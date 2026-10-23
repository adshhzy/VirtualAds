"""Compare complete render_img modes on an NPY configuration and two RGB BMPs.

This offline benchmark uses real supplied pixels with demonstration-only motion
callbacks (stationary camera, sigma 0.65, 5x5 Gaussian). It never changes runtime
defaults or accepts a nonzero image difference as an automatic-policy candidate.
The eager candidate replaces BOTH shared builder and executor; its shared mask
still uses the same Triton implementation as every other candidate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import time

import numpy as np
import torch

from benchmarks.benchmark_support import latency_statistics, rotated_order, serializable_arguments
from benchmarks.source_layout import REPOSITORY_ROOT, runtime_sources, source_file, source_key
from tools.check_render_img import load_sample
from examples.live_render_demo import SyntheticOwner
from virtualads.paste_renderer import ImageRenderer, RENDERER_VERSION, RenderExecutor, RenderOptions
from virtualads.paste_route_plan import build_route_plan
from virtualads.paste_shared_mask import SharedMaskBuilder
from virtualads.paste_shared_params import SharedParameterBuilder


ROOT = REPOSITORY_ROOT
ROUTE_KEY = re.compile(r"ad_params_([1-9][0-9]*)\Z")
BATCH, SLOTS, HISTORY_CAPACITY = 8, 16, 9
DEFAULT_CANDIDATES = (
    "eager",
    "dense_whole",
    "dense_c1",
    "dense_c2",
    "dense_c4",
    "separable_whole",
    "separable_c1",
    "separable_c2",
    "separable_c4",
)
CANDIDATES = DEFAULT_CANDIDATES + ("dense_auto",)


def read_bmp(path):
    """Read uncompressed Windows 24-bit BGR BMP into owned contiguous RGB bytes."""
    raw = Path(path).read_bytes()
    if len(raw) < 54 or raw[:2] != b"BM":
        raise ValueError(f"{path}: expected a Windows 24-bit BMP")
    offset, header_size = struct.unpack_from("<II", raw, 10)
    width, signed_height = struct.unpack_from("<ii", raw, 18)
    planes, bits = struct.unpack_from("<HH", raw, 26)
    compression = struct.unpack_from("<I", raw, 30)[0]
    if (
        header_size < 40
        or planes != 1
        or bits != 24
        or compression != 0
        or width <= 0
        or signed_height == 0
        or offset < 14 + header_size
    ):
        raise ValueError(f"{path}: only uncompressed 24-bit RGB BMP is supported")
    height = abs(signed_height)
    stride = (width * 3 + 3) & ~3
    if offset + stride * height > len(raw):
        raise ValueError(f"{path}: truncated BMP pixel data")
    rows = np.frombuffer(raw, np.uint8, count=stride * height, offset=offset)
    image = rows.reshape(height, stride)[:, : width * 3].reshape(height, width, 3)
    if signed_height > 0:
        image = image[::-1]
    return np.ascontiguousarray(image[..., ::-1])


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return getattr(value, "__qualname__", type(value).__name__)
    return value


def array_hash(array):
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expanded_fixture(sample):
    """Cycle original numeric routes, then their ordered slots, to exactly B8/S16."""
    route_keys = sorted(
        (key for key in sample if isinstance(key, str) and ROUTE_KEY.fullmatch(key)),
        key=lambda key: int(ROUTE_KEY.fullmatch(key)[1]),
    )
    if not route_keys or any(not sample[key] for key in route_keys):
        raise ValueError("sample must contain nonempty ad_params_N routes")
    data = deepcopy({key: value for key, value in sample.items() if key not in route_keys})
    for route_index in range(BATCH):
        original = sample[route_keys[route_index % len(route_keys)]]
        data[f"ad_params_{route_index + 1}"] = [
            deepcopy(original[slot % len(original)]) for slot in range(SLOTS)
        ]
    # Pixel contents remain fixed; only frame_id and, optionally, bboxes advance.
    data["imgs_raw_version"] = 0
    return data


def fixture_boxes(data):
    return tuple(
        tuple(tuple(int(value) for value in item["bbox"]) for item in data[f"ad_params_{route}"])
        for route in range(1, BATCH + 1)
    )


def layout_offsets(mode, boxes, frame_hw):
    """A bounded 12-step translation cycle, identical for every route and slot."""
    if mode == "static":
        return ((0, 0),)
    height, width = frame_hw
    flattened = [box for route in boxes for box in route]
    min_x = min(box[0] for box in flattened)
    min_y = min(box[1] for box in flattened)
    max_x = max(box[2] for box in flattened)
    max_y = max(box[3] for box in flattened)
    if min_x < 5 or max_x > width or min_y < 0 or max_y > height:
        raise ValueError("translated requires valid bboxes and at least 5 pixels of left margin")
    up, down = min_y >= 3, height - max_y >= 3
    if not up and not down:
        raise ValueError("translated requires at least 3 pixels of top or bottom margin")
    vertical = (0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1)
    if not up:
        vertical = tuple(abs(value) for value in vertical)
    elif not down:
        vertical = tuple(-abs(value) for value in vertical)
    return tuple((-(step % 6), vertical[step]) for step in range(12))


class Inputs:
    """Only user inputs change here; no layout, mask or grid is prepared ahead."""

    def __init__(self, template, offsets):
        self.data = deepcopy(template)
        self.boxes = fixture_boxes(self.data)
        self.offsets = offsets
        self.frame_id = int(self.data["frame_id"])

    def update(self, pattern_index):
        self.frame_id += 1
        self.data["frame_id"] = self.frame_id
        dx, dy = self.offsets[pattern_index % len(self.offsets)]
        if len(self.offsets) > 1:
            for route, boxes in enumerate(self.boxes, 1):
                for parameter, (x1, y1, x2, y2) in zip(self.data[f"ad_params_{route}"], boxes):
                    parameter["bbox"] = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        return self.data


def make_renderer(name, device):
    if name not in CANDIDATES:
        raise ValueError(f"unknown candidate: {name}")
    chunk_size = "auto" if name == "dense_auto" else 0
    if name.rsplit("_", 1)[-1] in ("c1", "c2", "c4"):
        chunk_size = int(name[-1])
    worker = ImageRenderer(
        SyntheticOwner(kernel_size=5),
        RenderOptions(
            frame_layout="HWC",
            ad_layout="HWC",
            resize_ads_to_ad_h_w=True,
            batch_chunk_size=chunk_size,
        ),
        device=device,
        use_separable=name.startswith("separable"),
        mask_sigma=1.5,
        mask_radius=3,
        history_capacity=HISTORY_CAPACITY,
    )
    try:
        if name == "eager":
            # Merely replacing worker.executor.options leaves the old Triton
            # builder callback attached. Replace both authoritative owner slots.
            worker.release_resources()
            builder = SharedParameterBuilder(
                acceleration="eager",
                normalization="legacy",
                camera_index_policy="legacy",
                distortion_inverse=True,
                iterations=10,
                distortion_align_corners=worker.executor.options.distortion_align_corners,
                enable_separable=False,
            )
            executor = RenderExecutor(
                replace(
                    worker.executor.options,
                    acceleration="eager",
                    blur_backend="torch",
                    warp_grids_builder=builder.build_warp_grids,
                    pinned_output=True,
                    batch_chunk_size=0,
                )
            )
            worker.owner._shared_paste_builder = builder
            worker.owner._paste_batch_adapter = executor
            worker.executor = executor
        # Explicit Triton prevents an accidental mask fallback in the eager row.
        worker.mask_builder.release_resources()
        worker.mask_builder = SharedMaskBuilder(sigma=1.5, radius=3, acceleration="triton")
        return worker
    except BaseException:
        worker.close()
        raise


def output_protocol(output, frame_shape):
    expected = (BATCH, *frame_shape)
    if not isinstance(output, np.ndarray) or output.shape != expected:
        raise AssertionError(f"expected independent NumPy {expected}, got {type(output)}")
    if output.dtype != np.uint8 or not output.flags.c_contiguous:
        raise AssertionError("expected contiguous uint8 output")


def check_plan(worker, data, assets, frame):
    plan = build_route_plan(data, assets, frame.shape[:2], worker.executor.options)
    if (
        len(plan.groups) != 1
        or len(plan.groups[0].indices) != BATCH
        or len(plan.groups[0].params) != SLOTS
    ):
        raise ValueError("expanded sample must form exactly one compatible B8/S16 layout group")


def memory_state(worker, device):
    torch.cuda.synchronize(device)
    return {
        "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(device),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "assets": worker.executor.cache_info(),
        "mask": worker.mask_builder.cache_info(),
        "gaussian": worker.owner._shared_paste_builder.cache_info(),
        "history_entries": len(worker.history),
        "source_references": len(worker.executor._asset_source_refs),
        "owner_material_groups": len(worker.owner.imgs_raw_groups),
    }


def effective_options(worker):
    options = {
        field.name: getattr(worker.executor.options, field.name) for field in fields(RenderOptions)
    }
    builder = worker.owner._shared_paste_builder
    return json_ready(
        {
            "executor": options,
            "shared_builder": {
                name: getattr(builder, name)
                for name in (
                    "acceleration",
                    "normalization",
                    "camera_index_policy",
                    "iterations",
                    "distortion_inverse",
                    "distortion_align_corners",
                    "enable_separable",
                )
            },
            "mask": {"acceleration": worker.mask_builder.acceleration, "sigma": 1.5, "radius": 3},
        }
    )


def assert_closed(state):
    remaining = (
        state["assets"]["retained_stack_bytes"],
        state["mask"]["workspace_bytes"],
        state["gaussian"]["entries"],
        state["history_entries"],
        state["source_references"],
        state["owner_material_groups"],
    )
    if any(remaining):
        raise AssertionError(f"close left library resources: {remaining}")


def references(template, offsets, frame, assets, args):
    """Dense whole baseline, copied into ordinary CPU-owned NumPy storage."""
    worker = make_renderer("dense_whole", args.device)
    inputs = Inputs(template, offsets)
    arrays, descriptions = [], []
    try:
        check_plan(worker, inputs.data, assets, frame)
        for step in range(max(HISTORY_CAPACITY, args.warmup)):
            output = worker.render_img(inputs.update(step), frame, assets)
            del output
        for pattern, offset in enumerate(offsets):
            output = worker.render_img(inputs.update(pattern), frame, assets)
            output_protocol(output, frame.shape)
            all_routes_equal = all(np.array_equal(output[0], row) for row in output[1:])
            if args.layout == "translated" and not all_routes_equal:
                raise AssertionError("translated reference compaction requires all 8 routes equal")
            # Static keeps the entire B8 reference, never assumes identical routes.
            reference = output.copy() if args.layout == "static" else output[0].copy()
            arrays.append(reference)
            descriptions.append(
                {
                    "pattern": pattern,
                    "offset_xy": offset,
                    "stored_shape": reference.shape,
                    "bytes": reference.nbytes,
                    "sha256": array_hash(reference),
                    "full_batch_sha256": array_hash(output),
                    "route_sha256": [array_hash(row) for row in output],
                    "all_8_routes_equal": all_routes_equal,
                }
            )
            del output
        configuration = effective_options(worker)
        execution = deepcopy(worker.last_execution)
    finally:
        worker.close()
    closed = memory_state(worker, args.device)
    assert_closed(closed)
    return arrays, {
        "candidate": "dense_whole",
        "patterns": descriptions,
        "total_cpu_bytes": sum(array.nbytes for array in arrays),
        "effective_options": configuration,
        "execution": execution,
        "after_close": closed,
        "meaning": "GPU dense whole render copied to CPU; not an independent CPU-math oracle",
    }


def image_difference(output, reference):
    maximum = changed_channels = changed_pixels = 0
    for route, image in enumerate(output):
        expected = reference[route] if reference.ndim == 4 else reference
        different = image != expected
        changed_channels += int(np.count_nonzero(different))
        changed_pixels += int(np.count_nonzero(np.any(different, axis=-1)))
        if different.any():
            maximum = max(maximum, int(np.abs(image.astype(np.int16) - expected).max()))
    return {
        "max_uint8_diff": maximum,
        "changed_channels": changed_channels,
        "changed_pixels": changed_pixels,
    }


def prepare_pattern_checks(worker, inputs, frame, assets, expected):
    """Measure each pattern once, retaining pixels only for nonzero differences.

    The CPU copies are bounded by the finite layout pattern count and released
    after this candidate/round. A zero-difference pattern uses the existing dense
    reference, including its broadcast view for compact translated references.
    """
    for step in range(HISTORY_CAPACITY):
        output = worker.render_img(inputs.update(step), frame, assets)
        del output
    known_outputs, descriptions = [], []
    for pattern, dense_reference in enumerate(expected):
        output = worker.render_img(inputs.update(pattern), frame, assets)
        output_protocol(output, frame.shape)
        started = time.perf_counter()
        difference = image_difference(output, dense_reference)
        difference_ms = (time.perf_counter() - started) * 1000.0
        candidate_reference = output.copy() if difference["max_uint8_diff"] else None
        known_outputs.append((candidate_reference, difference))
        descriptions.append(
            {
                "pattern": pattern,
                "difference_to_dense": difference,
                "full_batch_sha256": array_hash(output),
                "additional_cpu_reference_bytes": 0
                if candidate_reference is None
                else candidate_reference.nbytes,
                "comparison_target": "dense_reference"
                if candidate_reference is None
                else "candidate_reference_with_measured_dense_difference",
                "detailed_difference_ms": difference_ms,
            }
        )
        del output
    return known_outputs, descriptions


def validate_known_output(output, dense_reference, known_output):
    """Compare every byte; reuse exact counts only when all pixels are identical."""
    candidate_reference, measured_difference = known_output
    reference = dense_reference if candidate_reference is None else candidate_reference
    if np.array_equal(output, np.broadcast_to(reference, output.shape)):
        return measured_difference, False
    # Never assume a candidate is deterministic. Recompute its actual difference
    # to dense when any byte differs from the precomputed candidate output.
    return image_difference(output, dense_reference), True


def stats(samples):
    result = latency_statistics(samples, include_minimum=True)
    result.update(samples=len(samples), p99=float(np.percentile(samples, 99)), max=max(samples))
    return result


def ownership_check(worker, inputs, frame, assets):
    """Keep two outputs only here, separately from every timed interval."""
    first = worker.render_img(inputs.update(0), frame, assets)
    output_protocol(first, frame.shape)
    before = array_hash(first)
    second = worker.render_img(inputs.update(len(inputs.offsets) - 1), frame, assets)
    output_protocol(second, frame.shape)
    result = {
        "previous_unchanged": array_hash(first) == before,
        "independent_storage": not np.shares_memory(first, second),
    }
    del first, second
    if not all(result.values()):
        raise AssertionError(f"output ownership failed: {result}")
    return result


def run_phase(name, round_index, template, offsets, frame, assets, expected, args):
    worker = make_renderer(name, args.device)
    inputs = Inputs(template, offsets)
    samples, checks, validation_samples = [], [], []
    known_outputs = []
    unexpected_changes = []
    executions = Counter()
    try:
        check_plan(worker, inputs.data, assets, frame)
        known_outputs, pattern_checks = prepare_pattern_checks(
            worker, inputs, frame, assets, expected
        )
        # Restore steady GPU activity after the expensive detailed CPU counts.
        # No reference generation or detailed counting occurs in this warmup.
        actual_warmup = max(HISTORY_CAPACITY, args.warmup)
        for step in range(actual_warmup):
            output = worker.render_img(inputs.update(step), frame, assets)
            del output
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        before = memory_state(worker, args.device)
        options = effective_options(worker)
        for step in range(args.iterations):
            pattern = step % len(offsets)
            data = inputs.update(pattern)
            torch.cuda.synchronize(args.device)
            started = time.perf_counter()
            output = worker.render_img(data, frame, assets)
            torch.cuda.synchronize(args.device)
            samples.append((time.perf_counter() - started) * 1000.0)
            # Every timed call compares all eight routes exactly outside timing.
            # Equal pixels imply precisely the precomputed dense diff/counts.
            validation_started = time.perf_counter()
            output_protocol(output, frame.shape)
            difference, unexpected = validate_known_output(
                output, expected[pattern], known_outputs[pattern]
            )
            validation_samples.append((time.perf_counter() - validation_started) * 1000.0)
            checks.append(
                {"pattern": pattern, "unexpected_output_change": unexpected, **difference}
            )
            if unexpected:
                unexpected_changes.append({"step": step, "pattern": pattern, **difference})
            del output
            executions[json.dumps(json_ready(worker.last_execution), sort_keys=True)] += 1
        after = memory_state(worker, args.device)
        if after["mask"]["effective_backend"] != "triton":
            raise AssertionError("every mode must use the same Triton shared mask")
        ownership = ownership_check(worker, inputs, frame, assets)
    finally:
        known_outputs.clear()
        worker.close()
    closed = memory_state(worker, args.device)
    assert_closed(closed)
    return {
        "candidate": name,
        "round": round_index + 1,
        "warmup_calls": actual_warmup,
        "history_prefill_calls_before_pattern_checks": HISTORY_CAPACITY,
        "precomputed_pattern_checks": pattern_checks,
        "additional_cpu_reference_bytes": sum(
            check["additional_cpu_reference_bytes"] for check in pattern_checks
        ),
        "samples_ms": samples,
        "stats_ms": stats(samples),
        "validation_samples_ms": validation_samples,
        "validation_stats_ms": stats(validation_samples),
        "unexpected_output_changes": unexpected_changes,
        "checks": checks,
        "checked_calls": len(checks),
        "checked_route_images": BATCH * len(checks),
        "max_uint8_diff": max(check["max_uint8_diff"] for check in checks),
        "changed_channels": sum(check["changed_channels"] for check in checks),
        "changed_pixels": sum(check["changed_pixels"] for check in checks),
        "effective_options": options,
        "execution_counts": [
            {"execution": json.loads(key), "calls": value} for key, value in executions.items()
        ],
        "memory_before": before,
        "memory_after_timing": after,
        "after_close": closed,
        "timed_peak_cuda_delta_bytes": after["cuda_peak_allocated_bytes"]
        - before["cuda_allocated_bytes"],
        "ownership_separate_from_timing": ownership,
    }


def aggregate(phases, candidates, frame_shape):
    results = {}
    for name in candidates:
        matching = [phase for phase in phases if phase["candidate"] == name]
        if not matching:
            continue
        samples = [value for phase in matching for value in phase["samples_ms"]]
        calls = sum(phase["checked_calls"] for phase in matching)
        maximum = max(phase["max_uint8_diff"] for phase in matching)
        changed_channels = sum(phase["changed_channels"] for phase in matching)
        changed_pixels = sum(phase["changed_pixels"] for phase in matching)
        unexpected_changes = sum(len(phase["unexpected_output_changes"]) for phase in matching)
        validation_samples = [
            value for phase in matching for value in phase["validation_samples_ms"]
        ]
        results[name] = {
            "stats_ms": stats(samples),
            "round_means_ms": [p["stats_ms"]["mean"] for p in matching],
            "checked_calls": calls,
            "checked_route_images": calls * BATCH,
            "max_uint8_diff": maximum,
            "changed_channels": changed_channels,
            "changed_pixels": changed_pixels,
            "validation_stats_ms": stats(validation_samples),
            "unexpected_output_changes": unexpected_changes,
            "changed_pixel_ratio": changed_pixels
            / (calls * BATCH * frame_shape[0] * frame_shape[1]),
            "bitexact_in_checked_calls": maximum == 0,
            "image_gate_passed": maximum == 0 and unexpected_changes == 0,
            "automatic_policy_approved": False,
        }
    return results


def provenance(args):
    sources = set(runtime_sources(ROOT))
    sources.update(
        source_file(ROOT, name)
        for name in (
            Path(__file__).name,
            "source_layout.py",
            "benchmark_support.py",
            "check_render_img.py",
            "live_render_demo.py",
        )
    )
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    versions = {}
    for distribution in ("triton", "triton-windows", "scipy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "renderer_version": RENDERER_VERSION,
        "git_head": commit,
        "source_sha256": {source_key(path, ROOT): file_hash(path) for path in sorted(sources)},
        "input_sha256": {
            name: file_hash(getattr(args, name)) for name in ("sample", "frame_bmp", "ad_bmp")
        },
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "versions": versions,
        "gpu": torch.cuda.get_device_name(args.device),
        "torch_backend_settings": {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }


def save_progress(report, output):
    """Replace only the report this invocation reserved with exclusive creation."""
    output.write_text(
        json.dumps(json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample", type=Path, required=True, help="NumPy-only object NPY configuration"
    )
    parser.add_argument(
        "--frame-bmp", type=Path, required=True, help="24-bit uncompressed frame BMP"
    )
    parser.add_argument(
        "--ad-bmp", type=Path, required=True, help="24-bit source BMP; alpha 255 is appended"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new JSON report, never an existing file"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--candidates", nargs="+", choices=CANDIDATES, default=list(DEFAULT_CANDIDATES)
    )
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument(
        "--warmup", type=int, default=30, help="at least 9 calls actually run to fill history"
    )
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--layout", choices=("static", "translated"), default="static")
    args = parser.parse_args()
    if args.rounds <= 0 or args.iterations <= 0 or args.warmup < 0:
        parser.error("rounds/iterations must be positive; warmup must be nonnegative")
    if len(set(args.candidates)) != len(args.candidates):
        parser.error("candidates must not contain duplicates")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing report: {args.output}")
    for name in ("sample", "frame_bmp", "ad_bmp"):
        if not getattr(args, name).is_file():
            parser.error(f"{name} file does not exist: {getattr(args, name)}")
    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    if device.type != "cuda":
        parser.error("this full-render benchmark requires CUDA, including its eager candidate")
    if args.rounds % len(args.candidates):
        print(
            "Exploratory run: confirmation requires rounds to be a multiple of candidate count.",
            flush=True,
        )
    return args


def main():
    args = parse_arguments()
    torch.cuda.set_device(args.device)
    sample = load_sample(args.sample)
    frame, ad = read_bmp(args.frame_bmp), read_bmp(args.ad_bmp)
    rgba = np.concatenate((ad, np.full((*ad.shape[:2], 1), 255, dtype=np.uint8)), axis=-1)
    template = expanded_fixture(sample)
    assets = {
        (item["ad_placement_id"], item["ad_id"]): rgba
        for route in range(1, BATCH + 1)
        for item in template[f"ad_params_{route}"]
    }
    offsets = layout_offsets(args.layout, fixture_boxes(template), frame.shape[:2])
    report = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": serializable_arguments(args),
        "provenance": provenance(args),
        "balanced_candidate_order": args.rounds % len(args.candidates) == 0,
        "batch": BATCH,
        "slots_per_route": SLOTS,
        "frame_shape": frame.shape,
        "raw_ad_shape": ad.shape,
        "asset_lookup_pairs": len(assets),
        "distinct_pixel_arrays": 1,
        "offsets_xy": offsets,
        "fixture": "Cycle original numeric routes and ordered slots to B8/S16; same opaque RGB image for each pair; resize to supplied ad_h_w; m_inv never changed",
        "callbacks": "SyntheticOwner(kernel_size=5), stationary supplied camera, sigma=0.65; not production callbacks",
        "mask": "All candidates use Triton maximum-union plus Gaussian sigma1.5 radius3 once per full frame",
        "timing": "Synchronized wall clock around complete ImageRenderer.render_img and ready independent pinned NumPy; includes metadata, mask/camera/shared geometry, material/ROI packing, H2D and D2H",
        "exclusions": [
            "BMP/NPY I/O",
            "fixture and reference construction",
            "bbox/frame_id updates",
            "JIT/warmup",
            "every-call CPU image validation",
            "memory/ownership checks",
            "encoding/network",
        ],
        "output_retention": "Each timed output is validated and released before next timer; ownership separately holds two outputs. Static dense CPU reference is full B8; translated stores 12 single-route dense references after checking baseline routes equal. Each phase retains one extra ordinary CPU B8 reference only for each pattern with a nonzero candidate/dense difference, and releases these references before close. Zero-difference patterns allocate no extra pixel references.",
        "memory_scope": "CUDA counters are process-wide; one candidate alive at a time, allocator reserves from earlier candidates may remain. Reserved bytes alone do not imply a leak.",
        "image_gate": "Every timed call/all 8 routes must be bitexact for image eligibility; nonzero differences are recorded without abort. Passing is necessary, not sufficient, for policy admission; this tool changes no policy.",
        "validation_method": "Before final warmup, compute each candidate/pattern's exact dense difference and counts once. Every timed output is then compared in full with np.array_equal to dense (zero difference) or its candidate CPU copy (nonzero difference). Exact equality proves the stored dense counts apply to every pixel; this is not sampling or a tolerance. Unexpected pixel changes trigger actual detailed dense comparison and are reported. validation_samples_ms is outside render timing. Numerical and thread settings are unchanged.",
        "phases": [],
        "aggregate": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(json_ready(report), stream, ensure_ascii=False, indent=2)
    try:
        expected, report["references"] = references(template, offsets, frame, assets, args)
        save_progress(report, args.output)
        for round_index in range(args.rounds):
            for name in rotated_order(args.candidates, round_index):
                phase = run_phase(
                    name, round_index, template, offsets, frame, assets, expected, args
                )
                report["phases"].append(phase)
                report["aggregate"] = aggregate(report["phases"], args.candidates, frame.shape)
                save_progress(report, args.output)
                print(
                    json.dumps(
                        {
                            "round": round_index + 1,
                            "candidate": name,
                            "stats_ms": phase["stats_ms"],
                            "max_uint8_diff": phase["max_uint8_diff"],
                            "changed_channels": phase["changed_channels"],
                        }
                    ),
                    flush=True,
                )
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save_progress(report, args.output)
    print(
        json.dumps({"report": str(args.output), "aggregate": report["aggregate"]}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
