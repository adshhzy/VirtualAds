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
else:
    from paste_render_executor import _bbox, _effective_ad_hw, _image_plan, _integer_tuple


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
class _Slot:
    image: object
    box: tuple[int, int, int, int]
    source_hw: tuple[int, int]
    declared_hw: tuple[int, int] | None
    matrix: object
    matrix_device: str
    transparency: object


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


def _prepare_slot(param, imgs_raw, frame_hw, options, image_index, name):
    if not isinstance(param, Mapping):
        raise TypeError(f"{name} must be a mapping")
    pair = (param["ad_placement_id"], param["ad_id"])
    image = _source_image(imgs_raw, pair, image_index, name)
    plan = _image_plan(image.shape, 4, options.ad_layout, name)
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


def _combine_group(indices, routes):
    params = []
    material_groups = [{} for _ in indices]
    for slot_index, first in enumerate(routes[indices[0]]):
        members = [routes[index][slot_index] for index in indices]
        pair = (_VIRTUAL_PLACEMENT, slot_index)
        for group, member in zip(material_groups, members):
            group[pair] = {"img": member.image}
        if isinstance(first.transparency, Real):
            transparency = np.asarray([member.transparency for member in members], dtype=np.float32)
        else:
            # Non-scalar opacity routes are deliberately singletons.
            transparency = first.transparency
        param = dict(
            ad_placement_id=pair[0],
            ad_id=pair[1],
            bbox=first.box,
            m_inv=_stack_matrices([member.matrix for member in members], first.matrix_device),
            transparency=transparency,
        )
        if first.declared_hw is not None:
            param["ad_h_w"] = first.declared_hw
        params.append(param)
    return RouteGroup(tuple(indices), params, material_groups)


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
    for route_index, (route_id, key) in enumerate(entries):
        params = data_dict[key]
        if not isinstance(params, (list, tuple)):
            raise TypeError(f"{key} must be a list/tuple")
        slots = tuple(
            _prepare_slot(param, imgs_raw, frame_hw, options, image_index, f"{key}[{slot_index}]")
            for slot_index, param in enumerate(params)
        )
        route_ids.append(route_id)
        routes.append(slots)
        signature = tuple((slot.box, slot.source_hw, slot.matrix_device) for slot in slots)
        unique_route = (
            route_index if any(not isinstance(slot.transparency, Real) for slot in slots) else None
        )
        compatible.setdefault((signature, unique_route), []).append(route_index)
    groups = tuple(_combine_group(indices, routes) for indices in compatible.values())
    return RoutePlan(tuple(route_ids), groups)


__all__ = ["RouteGroup", "RoutePlan", "build_route_plan"]
