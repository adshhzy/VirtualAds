"""Plan independent ad_params_N routes without changing their drawing order.

Only canonical positive route suffixes are read. ad_paste_data, way_ad_map,
brightness and the unsuffixed ad_params are ignored. Compatible routes share
ordered bounding boxes and effective source sizes; their matrices and scalar
opacity may differ. Other routes remain separate, including spatial opacity maps.

Planning reads source metadata, never pixels. It preserves the actual image
objects inside virtual slot mappings so the executor's identity/revision caches
continue to work. Geometry is rebuilt on each call, not cached by frame ID.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
import re

import numpy as np
import torch

if __package__:
    from .paste_render_executor import _bbox, _effective_ad_hw, _image_plan, _integer_tuple
    from .paste_route_preparation import PreparedAdSources
else:
    from paste_render_executor import _bbox, _effective_ad_hw, _image_plan, _integer_tuple
    from paste_route_preparation import PreparedAdSources


_ROUTE_KEY = re.compile(r"ad_params_([1-9][0-9]*)\Z")
_VIRTUAL_PLACEMENT = "__route_slot__"


@dataclass(frozen=True)
class RouteGroup:
    """One executor call; indices restore its outputs to RoutePlan.route_ids order."""

    indices: tuple[int, ...]
    params: list[dict]
    material_groups: list[dict]


@dataclass(frozen=True)
class RoutePlan:
    route_ids: tuple[int, ...]
    groups: tuple[RouteGroup, ...]


@dataclass(frozen=True)
class _PreparedRouteGroup(RouteGroup):
    sources: PreparedAdSources
    borrowed_parameters: tuple


@dataclass(frozen=True)
class _Slot:
    image: object
    box: tuple[int, int, int, int]
    source_hw: tuple[int, int]
    declared_hw: tuple[int, int] | None
    matrix: object
    matrix_device: str
    transparency: object
    image_plan: object
    source_metadata: tuple


def _host_metadata(value, name):
    """Keep coordinate/size validation from silently copying GPU values to CPU."""
    if isinstance(value, torch.Tensor) and value.device.type != "cpu":
        raise TypeError(f"{name} must be CPU coordinate/size metadata")
    return value


def _source_image(imgs_raw, pair, image_index, name):
    value = imgs_raw.get(pair)
    if value is None:
        raise KeyError(f"{name}: missing material for {pair!r}")
    if isinstance(value, Mapping):
        if "img" not in value:
            raise KeyError(f"{name}: material record must contain 'img'")
        value = value["img"]
    if isinstance(value, (tuple, list)):
        if image_index >= len(value):
            raise IndexError(f"{name}: image_index={image_index} outside device image list")
        value = value[image_index]
    if not isinstance(value, (np.ndarray, torch.Tensor)):
        raise TypeError(f"{name}: material must be a NumPy array or Tensor")
    return value


def _matrix_device(matrix, name):
    if isinstance(matrix, torch.Tensor):
        shape, device = tuple(matrix.shape), str(matrix.device)
    else:
        shape, device = np.asarray(matrix).shape, "cpu"
    if shape != (3, 3):
        raise ValueError(f"{name}: m_inv must be one [3,3] matrix per route and slot")
    return device


def _prepare_slot(param, imgs_raw, frame_hw, options, image_index, name, image_plans=None):
    if not isinstance(param, Mapping):
        raise TypeError(f"{name} must be a mapping")
    pair = (param["ad_placement_id"], param["ad_id"])
    image = _source_image(imgs_raw, pair, image_index, name)
    shape = tuple(image.shape)
    source_metadata = (shape, image.dtype, getattr(image, "device", "cpu"))
    plan = None if image_plans is None else image_plans.get(shape)
    if plan is None:
        plan = _image_plan(shape, 4, options.ad_layout, name)
        if image_plans is not None:
            image_plans[shape] = plan
    if plan[1][0] != 1:
        raise ValueError(f"{name}: one image per route required, got batch={plan[1][0]}")
    declared_hw = None
    if param.get("ad_h_w") is not None and (options.check_ad_h_w or options.resize_ads_to_ad_h_w):
        declared_hw = _integer_tuple(
            _host_metadata(param["ad_h_w"], f"{name}.ad_h_w"), 2, f"{name}.ad_h_w"
        )
    source_hw = _effective_ad_hw(
        plan[1][-2:],
        declared_hw,
        options.check_ad_h_w,
        options.resize_ads_to_ad_h_w,
        name,
    )
    box = _bbox(_host_metadata(param["bbox"], f"{name}.bbox"), frame_hw, f"{name}.bbox")
    matrix = param["m_inv"]
    return _Slot(
        image,
        box,
        source_hw,
        declared_hw,
        matrix,
        _matrix_device(matrix, name),
        param.get("transparency", 1.0),
        plan,
        source_metadata,
    )


def _stack_matrices(matrices, device):
    if len(matrices) == 1:
        return matrices[0]
    if device != "cpu":
        # All members already have this same device: no implicit D2H/cross-GPU copy.
        return torch.stack(matrices)
    return np.stack(
        [
            matrix.detach().numpy() if isinstance(matrix, torch.Tensor) else matrix
            for matrix in matrices
        ]
    )


def _combine_group(indices, routes, preparation=None, options=None, image_index=0):
    params = []
    material_groups = [{} for _ in indices]
    sources, signatures, slot_signatures, checks = [], [], [], {}
    shared_declarations = True
    borrowed_parameters = []
    for slot_index, first in enumerate(routes[indices[0]]):
        members = [routes[index][slot_index] for index in indices]
        pair = (_VIRTUAL_PLACEMENT, slot_index)
        for group, member in zip(material_groups, members):
            group[pair] = {"img": member.image}
        if isinstance(first.transparency, Real):
            values = [member.transparency for member in members]
            if preparation is None:
                transparency = np.asarray(values, dtype=np.float32)
            else:
                transparency = preparation.array((len(members),), np.float32)
                transparency[:] = values
                preparation.counts["opacity_vectors"] += 1
        else:
            # Non-scalar opacity routes are deliberately singletons.
            transparency = first.transparency
        matrices = [member.matrix for member in members]
        matrix = _prepared_matrices(matrices, first.matrix_device, preparation)
        param = dict(
            ad_placement_id=pair[0],
            ad_id=pair[1],
            bbox=first.box,
            m_inv=matrix,
            transparency=transparency,
        )
        if first.declared_hw is not None:
            param["ad_h_w"] = first.declared_hw
        params.append(param)
        if preparation is not None:
            for name in ("m_inv", "transparency"):
                value = param[name]
                if isinstance(value, np.ndarray) and value.base is preparation._active:
                    borrowed_parameters.append((param, name))
            # The legacy executor applies the generated first member's ad_h_w
            # to every source. Preserve its validation when declarations differ,
            # even if the planner's per-route effective sizes happen to match.
            shared_declarations &= all(
                member.declared_hw == first.declared_hw for member in members
            )
            slot_sources, slot_signature = [], []
            for member in members:
                image = member.image
                identity = id(image)
                # Snapshot comes from the same instant as image_plan. A custom
                # input Mapping can mutate an earlier source during a later get.
                shape, dtype, device = member.source_metadata
                check_key = (identity, shape, dtype, device)
                check = checks.get(check_key)
                if check is None:
                    check = ((image, shape, dtype, device), str(dtype), str(device))
                    checks[check_key] = check
                slot_sources.append((image, member.image_plan, member.source_hw))
                slot_signature.append(
                    (pair, identity, check[0][1], check[1], check[2], member.source_hw)
                )
            sources.append(slot_sources)
            slot_signatures.append(tuple(slot_signature))
            signatures.extend(slot_signature)
    if preparation is not None:
        described = PreparedAdSources(
            material_groups,
            params,
            options,
            image_index,
            (sources, signatures, slot_signatures) if shared_declarations else None,
            tuple(check[0] for check in checks.values()),
            preparation.counts,
        )
        return _PreparedRouteGroup(
            tuple(indices), params, material_groups, described, tuple(borrowed_parameters)
        )
    return RouteGroup(tuple(indices), params, material_groups)


def _prepared_matrices(matrices, device, preparation):
    """Borrow only uniform CPU float32/64 stacks; retain NumPy promotion otherwise."""
    if preparation is not None and len(matrices) > 1 and device == "cpu":
        arrays = []
        for matrix in matrices:
            if isinstance(matrix, torch.Tensor):
                if matrix.is_neg() or matrix.is_conj() or matrix.layout != torch.strided:
                    break
                matrix = matrix.detach().numpy()
            if not isinstance(matrix, np.ndarray) or matrix.dtype not in (np.float32, np.float64):
                break
            if arrays and matrix.dtype != arrays[0].dtype:
                break
            arrays.append(matrix)
        else:
            output = preparation.array((len(arrays), 3, 3), arrays[0].dtype)
            # np.stack keeps its established promotion/order semantics, but
            # writes into this call's slice rather than allocating another array.
            np.stack(arrays, out=output)
            preparation.counts["matrix_stacks"] += 1
            return output
    return _stack_matrices(matrices, device)


def build_route_plan(data_dict, imgs_raw, frame_hw, options):
    """Return ordered routes and compatible executor batches, without altering inputs.

    Route IDs are integer suffixes, sorted numerically. Empty advertised routes
    remain background-output routes; no route keys produces an empty plan. Device
    image lists use options.image_index, with None meaning a fixed index zero,
    independent of route ID. A missing exact (placement_id, ad_id) material fails.

    Different slot counts, ordered bboxes, effective source sizes, or matrix
    devices stay in separate batches. Non-Real opacity values keep a route alone
    so full maps and tensor values retain their original shape and ownership.
    Scalar opacity becomes float32 [B], before the executor applies its scale.
    """
    return _build_route_plan(data_dict, imgs_raw, frame_hw, options)


def _build_route_plan(data_dict, imgs_raw, frame_hw, options, preparation=None):
    if not isinstance(data_dict, Mapping) or not isinstance(imgs_raw, Mapping):
        raise TypeError("data_dict and imgs_raw must be mappings")
    frame_hw = _integer_tuple(_host_metadata(frame_hw, "frame_hw"), 2, "frame_hw")
    if min(frame_hw) < 1:
        raise ValueError("frame_hw must contain positive dimensions")
    image_index = 0 if options.image_index is None else options.image_index
    if not isinstance(image_index, int) or image_index < 0:
        raise ValueError("image_index must be a nonnegative integer")
    entries = []
    for key in data_dict:
        if not isinstance(key, str) or not key.startswith("ad_params_"):
            continue
        match = _ROUTE_KEY.fullmatch(key)
        if match is None:
            raise ValueError(
                f"invalid route key {key!r}; expected ad_params_N with positive integer N"
            )
        entries.append((int(match[1]), key))
    entries.sort()
    route_ids, routes = [], []
    compatible = {}
    image_plans = {} if preparation is not None else None
    for route_index, (route_id, key) in enumerate(entries):
        params = data_dict[key]
        if not isinstance(params, (list, tuple)):
            raise TypeError(f"{key} must be a list/tuple")
        slots = tuple(
            _prepare_slot(
                param, imgs_raw, frame_hw, options, image_index, f"{key}[{slot_index}]", image_plans
            )
            for slot_index, param in enumerate(params)
        )
        route_ids.append(route_id)
        routes.append(slots)
        signature = tuple((slot.box, slot.source_hw, slot.matrix_device) for slot in slots)
        unique_route = (
            route_index if any(not isinstance(slot.transparency, Real) for slot in slots) else None
        )
        compatible.setdefault((signature, unique_route), []).append(route_index)
    if preparation is None:
        groups = tuple(_combine_group(indices, routes) for indices in compatible.values())
    else:
        preparation.begin(sum(len(route) for route in routes))
        try:
            groups = tuple(
                _combine_group(indices, routes, preparation, options, image_index)
                for indices in compatible.values()
            )
        finally:
            preparation.finish()
    return RoutePlan(tuple(route_ids), groups)


__all__ = ["RouteGroup", "RoutePlan", "build_route_plan"]
