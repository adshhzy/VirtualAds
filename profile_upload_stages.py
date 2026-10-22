"""Diagnostic stage timings for a preserved or current B8/S16 render_img.

Use benchmark_upload_comparison.py for performance decisions. This tool adds
CPU wrappers and CUDA events, then separately records actual device activities.
Nested scopes and CPU/GPU times overlap and must never be summed together.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import wraps
import importlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import benchmark_render_img_modes as fixture
from check_streaming_stability import load_baseline, verify_baseline_sources
from live_render_demo import SyntheticOwner


class StageHooks:
    def __init__(self):
        self.mode = "off"
        self.originals = []
        self.events = {}
        self.begin()

    def begin(self):
        self.counts = defaultdict(int)
        self.cpu = defaultdict(float)
        self.used = []

    def add(self, obj, attr, name, cuda=True):
        original = getattr(obj, attr)

        @wraps(original)
        def wrapped(*args, **kwargs):
            if self.mode == "off":
                return original(*args, **kwargs)
            if self.mode == "profiler":
                with torch.profiler.record_function("stage::" + name):
                    return original(*args, **kwargs)
            index = self.counts[name]
            self.counts[name] += 1
            key = (name, index)
            if cuda:
                if key not in self.events:
                    self.events[key] = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                first, last = self.events[key]
                first.record()
            started = time.perf_counter()
            result = original(*args, **kwargs)
            self.cpu[name] += (time.perf_counter() - started) * 1000
            if cuda:
                last.record()
                self.used.append(key)
            return result

        self.originals.append((obj, attr, original))
        setattr(obj, attr, wrapped)

    def read(self):
        # Output is already ready; only final wrapper events may still be queued.
        torch.cuda.current_stream().synchronize()
        spans = defaultdict(float)
        for key in self.used:
            first, last = self.events[key]
            spans[key[0]] += first.elapsed_time(last)
        return {"cpu_ms": dict(self.cpu), "stream_ms": dict(spans), "calls": dict(self.counts)}

    def close(self):
        for obj, attr, original in reversed(self.originals):
            setattr(obj, attr, original)
        self.originals.clear()


def summarize_records(records):
    return {
        kind: {
            name: fixture.stats([row[kind].get(name, 0) for row in records])
            for name in records[0][kind]
        }
        for kind in ("cpu_ms", "stream_ms", "calls")
    }


def summarize_trace(path):
    """Sum kernel/memcpy activities, never CPU parents or GPU annotations."""
    events = json.loads(path.read_text(encoding="utf8"))["traceEvents"]
    scopes = [event for event in events if event.get("cat") == "user_annotation"]
    starts = sorted(event["ts"] for event in scopes if event["name"] == "stage::route_plan")
    ops = {
        event.get("args", {}).get("External id"): event
        for event in events
        if event.get("cat") == "cpu_op"
    }
    activities, transfers = defaultdict(list), defaultdict(list)
    for event in events:
        if event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        # Calls return ready NumPy: prior frame work cannot cross the next start.
        frame = sum(start <= event["ts"] for start in starts) - 1
        if frame < 1:
            continue
        size = event.get("args", {}).get("bytes")
        activities[(event["cat"], event["name"], size)].append(event["dur"])
        if event["cat"] == "gpu_memcpy":
            op = ops.get(event.get("args", {}).get("External id"))
            matches = (
                []
                if op is None
                else [
                    scope
                    for scope in scopes
                    if scope["ts"] <= op["ts"]
                    and scope["ts"] + scope["dur"] >= op["ts"] + op["dur"] - 0.01
                ]
            )
            scope = (
                min(matches, key=lambda item: item["dur"])["name"] if matches else "unattributed"
            )
            transfers[(scope, event["name"], size)].append(event["dur"])
    frames = len(starts) - 1
    if frames < 1:
        raise RuntimeError("trace must contain at least two annotated frames")

    def rows(values, label):
        return [
            {
                label: key[0],
                "name": key[1],
                "bytes": key[2],
                "calls_per_frame": len(items) / frames,
                "ms_per_frame": sum(items) / (frames * 1000),
            }
            for key, items in sorted(values.items(), key=lambda row: sum(row[1]), reverse=True)
        ]

    return {
        "frames": frames,
        "excluded_first_frame": True,
        "activities": rows(activities, "category"),
        "transfers": rows(transfers, "scope"),
    }


def install_hooks(hooks, worker, prefix):
    modules = {
        name: importlib.import_module(prefix + name)
        for name in (
            "paste_renderer",
            "paste_render_executor",
            "paste_shared_params",
            "paste_shared_kernels",
            "paste_gpu_batch",
            "paste_gpu_kernels",
            "paste_mask_kernels",
        )
    }
    api, execution = modules["paste_renderer"], modules["paste_render_executor"]
    # The three-argument entry point may use its private borrowed-input planner;
    # hook that actual call, without wrapping callbacks that gate its fast path.
    hooks.add(api, "build_route_plan", "route_plan", False)
    if hasattr(api, "_build_route_plan"):
        # Public planning calls its own module's helper, not this imported alias,
        # so these two wrappers measure mutually exclusive renderer branches.
        hooks.add(api, "_build_route_plan", "route_plan", False)
    hooks.add(worker, "_camera_inputs", "camera_snapshot", False)
    if hasattr(worker, "_frame_upload"):
        hooks.add(worker._frame_upload, "prepare", "frame_upload")
    else:
        hooks.add(api, "_as_tensor", "frame_upload")
    hooks.add(worker.mask_builder, "build", "shared_mask")
    if hasattr(worker.mask_builder, "_upload_host_patches"):
        hooks.add(worker.mask_builder, "_upload_host_patches", "mask_upload")
    hooks.add(modules["paste_mask_kernels"], "gaussian_mask_into", "mask_feather")
    hooks.add(api, "_prepare_shared_parameters", "shared_parameters")
    hooks.add(
        modules["paste_shared_params"], "compute_camera_filter_shared", "camera_filter", False
    )
    hooks.add(modules["paste_shared_kernels"], "build_distortion_grid_gpu", "distortion_grid")
    hooks.add(worker.owner._shared_paste_builder, "_motion_value", "gaussian_cache", False)
    for attr, name, cuda in (
        ("_prepare_frame", "frame_layout", False),
        ("_collect_ads", "asset_cache", False),
        ("_prepare_warp_inputs", "roi_parameters_and_grids", True),
        ("_prepare_canvas_inputs", "canvas_parameters", False),
        ("_prepare_motion_kernel", "motion_parameters", False),
        ("_export", "numpy_export", True),
    ):
        hooks.add(worker.executor, attr, name, cuda)
    hooks.add(execution, "_paste_gpu_batch", "core_total")
    hooks.add(modules["paste_gpu_kernels"], "_warp_paste_into_supported", "slot_warps")
    hooks.add(modules["paste_gpu_batch"].F, "grid_sample", "full_frame_distortion")
    hooks.add(modules["paste_gpu_kernels"], "blur_blend_to_uint8", "dense_blur_blend_uint8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sample", "frame-bmp", "ad-bmp", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="Preserved source; omit to profile current")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    if not __debug__ or args.steps < 1 or torch.device(args.device).type != "cuda":
        parser.error("requires CUDA, positive steps and enabled assertions")
    if args.output.exists():
        parser.error("output directory already exists; refusing overwrite")
    args.output.mkdir(parents=True)
    torch.cuda.set_device(args.device)
    prefix = ""
    if args.baseline:
        load_baseline(args.baseline)
        prefix = "_streaming_baseline."
    api = importlib.import_module(prefix + "paste_renderer")
    template = fixture.expanded_fixture(fixture.load_sample(args.sample))
    frame, ad = fixture.read_bmp(args.frame_bmp), fixture.read_bmp(args.ad_bmp)
    rgba = np.concatenate((ad, np.full((*ad.shape[:2], 1), 255, np.uint8)), axis=-1)
    assets = {
        (item["ad_placement_id"], item["ad_id"]): rgba
        for route in range(1, 9)
        for item in template[f"ad_params_{route}"]
    }
    inputs = fixture.Inputs(template, ((0, 0),))
    worker = api.ImageRenderer(
        SyntheticOwner(kernel_size=5),
        api.RenderOptions(frame_layout="HWC", ad_layout="HWC", resize_ads_to_ad_h_w=True),
        device=args.device,
        use_separable=False,
        mask_sigma=1.5,
        mask_radius=3,
        history_capacity=9,
    )
    hooks = StageHooks()
    report = {
        "source": str(args.baseline.resolve()) if args.baseline else "current",
        "runtime_sha256": {
            path.name: fixture.file_hash(path)
            for path in sorted(Path(api.__file__).resolve().parent.glob("paste*.py"))
        },
        "version": api.RENDERER_VERSION,
        "device": torch.cuda.get_device_name(),
        "cpu_threads": torch.get_num_threads(),
        "input_sha256": {
            name: fixture.file_hash(getattr(args, name))
            for name in ("sample", "frame_bmp", "ad_bmp")
        },
        "fixture": "3336 B8/S16 fixed bbox dense 5x5",
        "notes": [
            "CPU wall and GPU intervals overlap; nested scopes overlap.",
            "Profiler device activities are a separate diagnostic, not full-call latency.",
            "Stationary SyntheticOwner sigma 0.65, not production motion callbacks.",
            "Full-batch exact checks occur at phase endpoints, not every profiled frame.",
        ],
    }

    def render():
        return worker.render_img(inputs.update(0), frame, assets)

    def warm():
        for _ in range(50):
            output = render()
            del output

    def control(reference):
        warm()
        times = []
        for _ in range(args.steps):
            start = time.perf_counter()
            output = render()
            times.append((time.perf_counter() - start) * 1000)
            del output
        assert np.array_equal(render(), reference)
        return fixture.stats(times)

    try:
        warm()
        reference = render().copy()
        report["reference_sha256"] = fixture.array_hash(reference)
        report["baseline_before_ms"] = control(reference)
        install_hooks(hooks, worker, prefix)
        hooks.mode = "events"
        for _ in range(30):
            hooks.begin()
            output = render()
            hooks.read()
            del output
        records, elapsed = [], []
        for _ in range(args.steps):
            hooks.begin()
            start = time.perf_counter()
            output = render()
            elapsed.append((time.perf_counter() - start) * 1000)
            records.append(hooks.read())
            del output
        hooks.mode = "off"
        assert np.array_equal(render(), reference)
        report["event_instrumented_ms"] = fixture.stats(elapsed)
        report["stages"] = summarize_records(records)
        report["records"] = records
        warm()
        hooks.mode = "profiler"
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(10):
                output = render()
                del output
        trace = args.output / "trace.json"
        prof.export_chrome_trace(str(trace))
        report["device_activities"] = summarize_trace(trace)
        hooks.mode = "off"
        hooks.close()
        report["baseline_after_ms"] = control(reference)
        report["execution"] = worker.last_execution
        report["shared_info"] = worker.owner._shared_paste_builder.last_info
        report["assets"] = worker.executor.cache_info()
        report["shared_cache"] = worker.owner._shared_paste_builder.cache_info()
        report["mask_workspace"] = worker.mask_builder.cache_info()
        if hasattr(worker, "_route_preparation"):
            report["route_workspace"] = worker._route_preparation.cache_info()
        if hasattr(worker, "_frame_upload"):
            report["frame_upload"] = worker._frame_upload.cache_info()
        report["phase_endpoint_pixel_exact"] = True
        if args.baseline:
            verify_baseline_sources(args.baseline.resolve())
    finally:
        hooks.close()
        worker.close()
    (args.output / "report.json").write_text(
        json.dumps(fixture.json_ready(report), indent=2), encoding="utf8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "before_ms": report["baseline_before_ms"],
                "after_ms": report["baseline_after_ms"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
