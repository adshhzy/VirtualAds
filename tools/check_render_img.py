"""Reproducible GPU checks for render_img, with independent per-route references.

SyntheticOwner provides demonstration-only camera/Gaussian callbacks. Neither the
small fixtures nor optional sample geometry constitute real video/material data.
The old render_frame reference shares the current camera history and completed
mask, isolating route assembly from mask filtering. A separate SciPy reference
checks mask union/feathering. No timings or production performance claims are made.

The optional object NPY is read with a NumPy-only pickle global allowlist; this
does not make arbitrary pickle globals executable. As with any array parser,
inputs should still have a reasonable file size and memory footprint.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import re

import numpy as np
from scipy.ndimage import convolve1d
import torch

from examples.live_render_demo import SyntheticOwner
from virtualads.paste_renderer import (
    ImageRenderer,
    RENDERER_VERSION,
    RenderOptions,
    initialize_renderer,
    release_renderer,
    render_frame,
)


HEIGHT, WIDTH = 96, 160
_ROUTE_KEY = re.compile(r"ad_params_([1-9][0-9]*)\Z")


class _NumpyUnpickler(pickle.Unpickler):
    """Permit only the constructors used by ordinary NumPy object-array files."""

    def find_class(self, module, name):
        if (module, name) == ("numpy", "ndarray"):
            return np.ndarray
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        if module in ("numpy.core.multiarray", "numpy._core.multiarray"):
            if name == "_reconstruct":
                return np._core.multiarray._reconstruct
            if name == "scalar":
                return np._core.multiarray.scalar
        raise pickle.UnpicklingError(f"disallowed pickle global: {module}.{name}")


def load_sample(path):
    """Load a scalar object NPY containing a dictionary, never allow_pickle=True."""
    with Path(path).open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY version {version}; expected 1.0 or 2.0")
        if shape != () or dtype != np.dtype(object):
            raise ValueError("sample must be a scalar object NPY containing a dictionary")
        value = _NumpyUnpickler(stream).load()
    if not isinstance(value, np.ndarray) or value.shape != () or value.dtype != object:
        raise ValueError("sample payload does not match the scalar object NPY header")
    data = value.item()
    if not isinstance(data, Mapping):
        raise TypeError("sample must contain a dictionary")
    return data


def _route_ids(data):
    return tuple(
        sorted(
            int(match[1])
            for key in data
            if isinstance(key, str) and (match := _ROUTE_KEY.fullmatch(key))
        )
    )


def _background(height=HEIGHT, width=WIDTH):
    ys, xs = np.indices((height, width))
    return np.stack((20 + xs % 43, 25 + ys % 37, 30 + (xs + ys) % 41), axis=-1).astype(np.uint8)


def _material(seed, shape):
    ys, xs = np.indices(shape)
    return np.stack(
        (
            (seed * 47 + xs * 11) % 220 + 20,
            (seed * 29 + ys * 19) % 220 + 20,
            (seed * 71 + xs * 7 + ys * 5) % 220 + 20,
            130 + (xs * 7 + ys * 13) % 126,
        ),
        axis=-1,
    ).astype(np.uint8)


def make_inputs(routes=(10, 2, 1), frame_id=0):
    """Distinct route assets/matrices with shared ordered, overlapping layouts."""
    data = {
        "frame_id": frame_id,
        "K": np.array(((150, 0, WIDTH / 2), (0, 152, HEIGHT / 2), (0, 0, 1)), np.float32),
        "w2c": np.eye(4, dtype=np.float32),
        "dist_coeffs": np.zeros(5, np.float32),
        "mask_coor_list": [],
        "imgs_raw_version": 0,
    }
    assets = {}
    for route in routes:
        parameters = []
        for index, (shape, box) in enumerate(
            zip(((24, 44), (20, 40)), ((10, 18, 74, 50), (44, 30, 108, 62)))
        ):
            height, width = shape
            left, top, right, bottom = box
            matrix = np.eye(3, dtype=np.float32)
            matrix[0, 0], matrix[1, 1] = (
                (width - 1) / (right - left - 1),
                (height - 1) / (bottom - top - 1),
            )
            matrix[0, 2] = (route % 3) * 0.3
            pair = (f"placement-{index}", route * 10 + index)
            parameters.append(
                dict(
                    ad_placement_id=pair[0],
                    ad_id=pair[1],
                    bbox=box,
                    ad_h_w=shape,
                    m_inv=matrix,
                    transparency=0.45 + (route % 3) * 0.2,
                )
            )
            pixels = _material(route * 3 + index, shape)
            assets[pair] = pixels if index == 0 else {"img": pixels}
        data[f"ad_params_{route}"] = parameters
    return data, _background(), assets


def _image(value, image_index):
    if isinstance(value, Mapping):
        value = value["img"]
    if isinstance(value, (tuple, list)):
        value = value[image_index]
    return value


def scipy_mask(data, shape, sigma, radius):
    """Independent clipped maximum union, then float32 x/y nearest convolution."""
    height, width = shape
    canvas = np.zeros(shape, dtype=np.float32)
    for item in data.get("mask_coor_list", []):
        x1, y1, x2, y2 = item["bbox"]
        left, top, right, bottom = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if left >= right or top >= bottom:
            continue
        source = item["mask"]
        if isinstance(source, torch.Tensor):
            source = source.detach().cpu().numpy()
        patch = np.asarray(source[top - y1 : bottom - y1, left - x1 : right - x1], np.float32)
        np.maximum(canvas[top:bottom, left:right], patch, out=canvas[top:bottom, left:right])
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (coordinates / sigma) ** 2)
    weights = (weights / weights.sum()).astype(np.float32)
    horizontal = convolve1d(canvas, weights, axis=1, mode="nearest")
    return np.clip(convolve1d(horizontal, weights, axis=0, mode="nearest"), 0, 1)


class _Checks:
    def __init__(self, device):
        self.device = torch.device(device)
        options = RenderOptions(frame_layout="HWC", ad_layout="HWC", mask_layout="HW")
        self.owner = SyntheticOwner()
        self.renderer = ImageRenderer(self.owner, options, device=device, history_capacity=9)
        self.reference = SyntheticOwner()
        initialize_renderer(self.reference, options)
        self.identity = (id(self.renderer.executor), id(self.owner._shared_paste_builder))
        self.held = deque(maxlen=2)
        self.cases = []
        self.route_checks = 0
        self.output_checks = 0
        self.max_mask_error = 0.0
        self.workspace_checks = 0
        self.max_history = 0
        self._workspace_state = None

    def _check_held(self, output=None):
        for prior, snapshot in self.held:
            assert np.array_equal(prior, snapshot), "an older NumPy output was overwritten"
            if output is not None:
                assert not np.shares_memory(prior, output), "outputs share mutable storage"
            self.output_checks += 1

    def _check_mask(self, data, shape):
        builder = self.renderer.mask_builder
        info = builder.cache_info()
        if self.device.type == "cuda":
            assert info["effective_backend"] == "triton", info
        expected = scipy_mask(data, shape, builder.sigma, builder.radius)
        actual = builder._output.cpu().numpy()
        error = float(np.max(np.abs(actual - expected), initial=0.0))
        assert error <= 1e-6, f"mask differs from SciPy: {error}"
        self.max_mask_error = max(self.max_mask_error, error)
        state = (
            shape,
            tuple(
                value.data_ptr()
                for value in (builder._canvas, builder._temporary, builder._output, builder._patch)
            ),
            info["workspace_allocations"],
            info["kernel_uploads"],
        )
        if self._workspace_state is not None and self._workspace_state[0] == shape:
            assert state == self._workspace_state, "fixed-resolution mask workspace was reallocated"
            self.workspace_checks += 1
        self._workspace_state = state
        return error

    def _reference_route(self, data, frame, assets, route):
        # Copies isolate camera filtering and any future preparation edits from
        # the caller's live records. Materials deliberately retain actual identity.
        params = deepcopy(data[f"ad_params_{route}"])
        frame_data = {"ad_params": params}
        for name in ("K", "dist_coeffs"):
            if name in data:
                frame_data[name] = deepcopy(data[name])
        image_index = self.renderer.executor.options.image_index or 0
        group = {}
        for param in params:
            pair = param["ad_placement_id"], param["ad_id"]
            value = assets.get(pair)
            if value is None:
                raise KeyError(pair)
            group[pair] = {"img": _image(value, image_index)}
        self.reference.imgs_raw_version = data.get("imgs_raw_version", 0)
        return render_frame(
            self.reference,
            frame,
            frame_data,
            data.get("frame_id", 0),
            mask_canvas=self.renderer.mask_builder._output,
            material_groups=[group],
            camera_history=self.renderer.history,
            device=self.device,
        )

    def render(
        self, name, data, frame, assets, *, expected_groups=None, background=False, changed=None
    ):
        self._check_held()
        camera_before = {
            key: deepcopy(data[key]) for key in ("K", "w2c", "dist_coeffs") if key in data
        }
        output = self.renderer.render_img(data, frame, assets)
        routes = _route_ids(data)
        assert self.renderer.route_ids == routes
        expected_shape = frame.shape if len(routes) == 1 else (len(routes), *frame.shape)
        assert output.shape == expected_shape and output.dtype == np.uint8
        assert output.flags.c_contiguous
        assert not np.shares_memory(frame, output), "result aliases the input frame"
        self._check_held(output)
        if changed is not None:
            assert self.held, "a visual change check requires a preceding output"
            equal = np.array_equal(output, self.held[-1][1])
            assert equal != changed, f"{name}: expected output changed={changed}"
        assert self.identity == (id(self.renderer.executor), id(self.owner._shared_paste_builder))
        assert len(self.renderer.history) <= 9
        self.max_history = max(self.max_history, len(self.renderer.history))
        for key, previous in camera_before.items():
            np.testing.assert_array_equal(data[key], previous, err_msg=f"input {key} mutated")
        groups = len(self.renderer.last_execution)
        if expected_groups is not None:
            assert groups == expected_groups, (name, groups, expected_groups)
        mask_error = None
        if any(data[f"ad_params_{route}"] for route in routes):
            mask_error = self._check_mask(data, frame.shape[:2])
        batch = output[None] if output.ndim == 3 else output
        for index, route in enumerate(routes):
            expected = self._reference_route(data, frame, assets, route)
            try:
                difference = int(
                    np.abs(batch[index].astype(np.int16) - expected.astype(np.int16)).max()
                )
                assert difference == 0, f"{name}: route {route} max uint8 difference {difference}"
                self.route_checks += 1
                if background:
                    np.testing.assert_array_equal(batch[index], frame)
            finally:
                del expected
        del batch
        if not routes:
            assert self.renderer.executor.cache_info()["retained_stack_bytes"] == 0
            assert not self.renderer.executor._asset_source_refs
            assert not self.owner.imgs_raw_groups
            assert self.owner._shared_paste_builder.cache_info()["entries"] == 0
            assert self.renderer.mask_builder.cache_info()["workspace_bytes"] == 0
            self._workspace_state = None
        else:
            self.held.append((output, output.copy()))
        self.cases.append(
            dict(
                case=name,
                route_ids=routes,
                groups=groups,
                max_uint8_diff=0,
                mask_max_abs_error=mask_error,
            )
        )
        return output

    def reset(self):
        self.renderer.reset_history()
        release_renderer(self.reference)
        self.reference.imgs_raw_groups = []
        self._workspace_state = None
        assert len(self.renderer.history) == 0
        assert self.renderer.executor.cache_info()["retained_stack_bytes"] == 0
        self._check_held()

    def close(self):
        try:
            self.renderer.close()
        finally:
            release_renderer(self.reference)
            self.reference.imgs_raw_groups = []
            self.reference.paste_data_stack = None
        self._check_held()
        info = dict(
            retained_stack_bytes=self.renderer.executor.cache_info()["retained_stack_bytes"],
            source_references=len(self.renderer.executor._asset_source_refs),
            kernel_entries=self.owner._shared_paste_builder.cache_info()["entries"],
            mask_workspace_bytes=self.renderer.mask_builder.cache_info()["workspace_bytes"],
            history_entries=len(self.renderer.history),
            material_groups=len(self.owner.imgs_raw_groups),
            owner_history_is_none=self.owner.paste_data_stack is None,
        )
        assert all(value == 0 for key, value in info.items() if key != "owner_history_is_none")
        assert info["owner_history_is_none"]
        self.renderer.close()  # Idempotent shutdown must preserve caller-held output too.
        self._check_held()
        try:
            self.renderer.render_img({}, np.empty((1, 1, 3), np.uint8), {})
        except RuntimeError as error:
            assert "closed" in str(error)
        else:
            raise AssertionError("render_img accepted a call after close")
        self.held.clear()
        return info

    def report(self, after_close):
        return dict(
            cases=self.cases,
            frames_checked=len(self.cases),
            route_images_checked=self.route_checks,
            max_uint8_diff=0,
            mask_max_abs_error=self.max_mask_error,
            mask_tolerance=1e-6,
            mask_workspace_reuse_checks=self.workspace_checks,
            old_output_checks=self.output_checks,
            retained_outputs_limit=2,
            max_history_entries=self.max_history,
            after_close=after_close,
        )


def _mask_items(kind, step=0):
    if kind == "empty":
        return []
    if kind == "ones":
        return [dict(bbox=(0, 0, WIDTH, HEIGHT), mask=np.ones((HEIGHT, WIDTH), np.float32))]
    if kind == "clipped":
        return [
            dict(bbox=(-8, -6, 24, 20), mask=np.ones((26, 32), np.float32)),
            dict(
                bbox=(WIDTH - 9, HEIGHT - 7, WIDTH + 6, HEIGHT + 5),
                mask=np.full((12, 15), 0.6, np.float32),
            ),
        ]
    left = 22 + step % 13
    return [
        dict(bbox=(left, 20, left + 24, 42), mask=np.full((22, 24), 0.8, np.float32)),
        dict(bbox=(30, 25, 56, 45), mask=np.full((20, 26), 0.5, np.float32)),
    ]


def _run_synthetic(checks, frames):
    frame_id = 0
    for routes in ((1,), (2, 1), (10, 2, 1), tuple(range(8, 0, -1))):
        data, frame, assets = make_inputs(routes, frame_id)
        checks.render(f"compatible_B{len(routes)}", data, frame, assets, expected_groups=1)
        frame_id += 1
    for variation in ("slot_count", "bbox", "source_size", "slot_order", "spatial_opacity"):
        data, frame, assets = make_inputs(frame_id=frame_id)
        params = data["ad_params_2"]
        if variation == "slot_count":
            params.pop()
        elif variation == "bbox":
            params[0]["bbox"] = (11, 18, 75, 50)
        elif variation == "source_size":
            param = params[0]
            assets[param["ad_placement_id"], param["ad_id"]] = _material(8, (26, 46))
            param["ad_h_w"] = (26, 46)
        elif variation == "slot_order":
            params.reverse()
        else:
            params[0]["transparency"] = np.full((32, 64), 0.6, np.float32)
        checks.render(variation, data, frame, assets, expected_groups=2)
        frame_id += 1

    data, frame, assets = make_inputs(frame_id=frame_id)
    checks.render("same_id_initial", data, frame, assets, expected_groups=1)
    data["ad_params_2"][0]["m_inv"][0, 2] += 2.5
    data["K"][0, 0] += 0.5
    data["dist_coeffs"][0] = 0.001
    checks.render("same_id_inplace_geometry", data, frame, assets, expected_groups=1, changed=True)
    data["ad_params_2"][0]["bbox"] = (12, 18, 76, 50)
    checks.render("same_id_inplace_bbox", data, frame, assets, expected_groups=2, changed=True)
    first_pair = next(iter(assets))
    assets[first_pair] = _material(23, (24, 44))
    checks.render("source_object_replacement", data, frame, assets, expected_groups=2, changed=True)
    _image(assets[first_pair], 0)[..., :3] = (240, 20, 170)
    data["imgs_raw_version"] += 1
    checks.render("inplace_pixels_revision", data, frame, assets, expected_groups=2, changed=True)
    _image(assets[first_pair], 0)[..., :3] = (20, 230, 60)
    checks.renderer.invalidate_assets()
    checks.reference._paste_batch_adapter.invalidate_assets()
    checks.render(
        "inplace_pixels_explicit_invalidation", data, frame, assets, expected_groups=2, changed=True
    )
    data.update(
        brightness=object(), ad_paste_data=object(), way_ad_map=object(), image_path=object()
    )
    checks.render("ignored_auxiliary_fields", data, frame, assets, expected_groups=2, changed=False)
    frame_id += 1

    for kind in ("dynamic", "dynamic", "empty", "ones", "clipped"):
        data["frame_id"] = frame_id
        data["mask_coor_list"] = _mask_items(kind, frame_id)
        checks.render(f"mask_{kind}_{frame_id}", data, frame, assets, background=kind == "ones")
        frame_id += 1
    checks.render("pause_B0", {}, frame, {}, expected_groups=0)
    checks.render("pause_B0_again", {}, frame, {}, expected_groups=0)
    data["frame_id"] = frame_id
    checks.render("resume_same_worker", data, frame, assets, expected_groups=2)
    frame_id += 1
    checks.render(
        "all_no_ads",
        {"ad_params_10": [], "ad_params_1": []},
        frame,
        {},
        expected_groups=1,
        background=True,
    )
    data["frame_id"] = frame_id
    data["ad_params_2"] = []
    checks.render("mixed_no_ads", data, frame, assets, expected_groups=2)
    checks.reset()
    data, frame, assets = make_inputs((1,), frame_id=0)
    checks.render("reset_frame_id_to_zero", data, frame, assets, expected_groups=1)

    # Reuse a bounded set of source assets; vary B, camera values, masks and live
    # matrices without retaining per-frame inputs or an unbounded output history.
    all_data, frame, assets = make_inputs(tuple(range(1, 9)))
    for step in range(frames):
        batch = (1, 2, 3, 8)[step % 4]
        current = {
            key: value for key, value in all_data.items() if not key.startswith("ad_params_")
        }
        current.update(
            {f"ad_params_{route}": all_data[f"ad_params_{route}"] for route in range(1, batch + 1)}
        )
        current["frame_id"] = step + 1
        current["dist_coeffs"][0] = (step % 7) * 0.0001
        current["w2c"][0, 3] = step * 0.001
        current["mask_coor_list"] = _mask_items("dynamic", step)
        for route in range(1, batch + 1):
            current[f"ad_params_{route}"][0]["m_inv"][0, 2] = (step % 5) * 0.1
        checks.render(
            f"bounded_loop_{step:04d}_B{batch}", current, frame, assets, expected_groups=1
        )


def _run_sample(device, path):
    data = load_sample(path)
    routes = _route_ids(data)
    if not routes:
        raise ValueError("sample has no ad_params_N routes")
    assets = {}
    for route in routes:
        for param in data[f"ad_params_{route}"]:
            pair = param["ad_placement_id"], param["ad_id"]
            if pair not in assets:
                assets[pair] = _material(len(assets) + 1, (200, 200))
    checks = _Checks(device)
    try:
        checks.render("provided_sample_synthetic_1080p", data, _background(1080, 1920), assets)
    finally:
        after_close = checks.close()
    report = checks.report(after_close)
    report["sample"] = dict(
        filename=Path(path).name,
        route_ids=routes,
        slot_counts=[len(data[f"ad_params_{route}"]) for route in routes],
        unique_material_pairs=len(assets),
        synthetic_frame_hw=[1080, 1920],
        synthetic_material_hw=[200, 200],
        real_video_or_materials=False,
    )
    return report


def run_checks(device, *, frames=100, sample=None):
    checks = _Checks(device)
    try:
        _run_synthetic(checks, frames)
    finally:
        after_close = checks.close()
    report = dict(
        renderer_version=RENDERER_VERSION,
        created_utc=datetime.now(timezone.utc).isoformat(),
        device=str(device),
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        synthetic_callbacks_only=True,
        finite_checks_are_not_a_production_leak_proof=True,
        finite_loop_frames=frames,
        synthetic=checks.report(after_close),
    )
    if sample is not None:
        report["sample"] = _run_sample(device, sample)
    return report


def main(argv=None):
    if not __debug__:
        raise RuntimeError("run this verification tool without -O; assertions must remain enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=Path)
    parser.add_argument(
        "--frames", type=int, default=100, help="Additional bounded dynamic-B frames (default: 100)"
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.frames < 0:
        parser.error("frames must be nonnegative")
    if torch.device(args.device).type != "cuda":
        parser.error("this complete regression requires a CUDA device")
    if args.sample is not None and not args.sample.is_file():
        parser.error(f"sample does not exist: {args.sample}")
    report = run_checks(args.device, frames=args.frames, sample=args.sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                output=str(args.output),
                frames_checked=report["synthetic"]["frames_checked"],
                route_images_checked=report["synthetic"]["route_images_checked"],
                max_uint8_diff=0,
                mask_max_abs_error=report["synthetic"]["mask_max_abs_error"],
            )
        )
    )


if __name__ == "__main__":
    main()
