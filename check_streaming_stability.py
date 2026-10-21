"""Bounded-state streaming correctness and allocation checks, not a speed benchmark.

--frames is the total measured call count, split between fixed and dynamic
phases. The same current builder/adapter survives both phases. Periodic old-code
oracles clear their assets/kernel cache before memory sampling. Neither the live
loop nor sampling calls gc.collect() or empty_cache(). Allocator reserved bytes
are reported separately from live allocations and are not classified as leaks.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import fields
import faulthandler
from functools import lru_cache
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
import weakref


def parse_arguments():
    from benchmark_paste_gpu_batch import parse_ad_size

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-label", default="6edf109")
    parser.add_argument("--frames", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument(
        "--ad-sizes", type=parse_ad_size, nargs="+", default=[(200, 200), (500, 500), (300, 600)]
    )
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--check-every", type=int, default=250)
    parser.add_argument("--change-every", type=int, default=32)
    parser.add_argument("--revision-every", type=int, default=17)
    parser.add_argument("--retain-outputs", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("results/streaming_stability.json"))
    cli = parser.parse_args()
    positive = [
        cli.frames,
        cli.warmup,
        cli.batch,
        *cli.batches,
        cli.slots,
        cli.sample_every,
        cli.check_every,
        cli.change_every,
        cli.revision_every,
        cli.retain_outputs,
        cli.max_samples,
    ]
    if min(positive) < 1 or cli.frames < 2 or min(cli.height, cli.width) < 8:
        parser.error("counts must be positive, frames >=2, and canvas dimensions >=8")
    if cli.kernel_size < 1 or cli.kernel_size % 2 != 1:
        parser.error("kernel-size must be positive and odd")
    if not cli.ad_sizes or any(min(hw) < 2 for hw in cli.ad_sizes):
        parser.error("source dimensions must be >=2")
    return cli


def source_hashes(directory, *, include_harness=False):
    paths = set(directory.glob("paste*.py"))
    if include_harness:
        paths.update(directory.glob("benchmark*.py"))
        paths.add(Path(__file__).resolve())
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def load_baseline(directory):
    """Use a private package and verify every imported runtime module's location."""
    directory = directory.resolve()
    if not directory.is_dir():
        raise ValueError(f"baseline directory does not exist: {directory}")
    package = ModuleType("_streaming_baseline")
    package.__path__ = [str(directory)]
    sys.modules[package.__name__] = package
    adapter = importlib.import_module(package.__name__ + ".paste_gpu_batch_adapter")
    shared = importlib.import_module(package.__name__ + ".paste_shared_params")
    verify_baseline_sources(directory)
    return adapter, shared


def verify_baseline_sources(directory):
    for name, module in tuple(sys.modules.items()):
        if name.startswith("_streaming_baseline."):
            if Path(module.__file__).resolve().parent != directory:
                raise RuntimeError(f"baseline module loaded from unexpected source: {name}")


@lru_cache(maxsize=1)
def _windows_memory_api():
    """Create ctypes types once; its pointer-type registry otherwise retains them."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
                "PrivateUsage",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    api = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    api.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    api.restype = wintypes.BOOL
    return Counters, kernel.GetCurrentProcess, api


def process_memory():
    """Current RSS/working set and private committed bytes without extra packages."""
    if os.name == "nt":
        import ctypes

        Counters, get_process, api = _windows_memory_api()
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not api(get_process(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"rss_bytes": counters.WorkingSetSize, "private_bytes": counters.PrivateUsage}
    statm = Path("/proc/self/statm")
    if statm.exists():
        return {
            "rss_bytes": int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE"),
            "private_bytes": None,
        }
    return {"rss_bytes": None, "private_bytes": None}


def cache_info(adapter):
    info = adapter.cache_info()
    return {**info, "sizes": {f"{h}x{w}": n for (h, w), n in info["sizes"].items()}}


def pinned_counter_issues(stats):
    """Keep raw counters, but flag values that cannot describe live host bytes."""
    current, reserved, peak = (
        stats.get(key)
        for key in ("allocated_bytes.current", "reserved_bytes.current", "allocated_bytes.peak")
    )
    if not all(isinstance(value, int) and value >= 0 for value in (current, reserved, peak)):
        return ["missing or negative host byte counters"]
    issues = []
    if current > reserved:
        issues.append("allocated current exceeds reserved current")
    if current > peak:
        issues.append("allocated current exceeds allocated peak")
    return issues


def memory_sample(path, device, retained_bytes):
    import torch

    result = {
        **process_memory(),
        "retained_output_bytes": retained_bytes,
        "assets": cache_info(path.adapter),
        "gaussian": path.builder.cache_info(),
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        stats = torch.cuda.memory_stats(device)
        result["cuda"] = {
            key: stats.get(key)
            for key in (
                "allocated_bytes.all.current",
                "reserved_bytes.all.current",
                "active_bytes.all.current",
                "inactive_split_bytes.all.current",
                "allocated_bytes.all.peak",
                "num_alloc_retries",
                "num_ooms",
                "num_device_alloc",
                "num_device_free",
            )
        }
        host_stats = getattr(torch.cuda.memory, "host_memory_stats", None)
        host = host_stats() if host_stats is not None else {}
        result["pinned"] = {
            key: host.get(key)
            for key in (
                "allocated_bytes.current",
                "reserved_bytes.current",
                "allocated_bytes.peak",
                "num_host_alloc",
                "num_host_free",
            )
        }
        result["pinned_stats_available"] = bool(host)
        result["pinned_counter_issues"] = pinned_counter_issues(host)
        result["pinned_live_bytes_valid"] = not result["pinned_counter_issues"]
    return result


class RetainedOutputs:
    """Keep a bounded consumer queue plus one phase anchor, checking ownership."""

    def __init__(self, limit):
        self.limit, self.queue = limit, deque()
        self.anchor = self.snapshot = None
        self.released = 0

    def add(self, output):
        import numpy as np

        if self.anchor is None:
            self.anchor, self.snapshot = output, output.copy()
        else:
            assert not np.shares_memory(output, self.anchor), "new output aliases first frame"
        for prior in self.queue:
            assert not np.shares_memory(output, prior), "consumer queue outputs alias"
        if self.queue:
            del prior
        self.queue.append(output)
        if len(self.queue) > self.limit:
            expired = self.queue.popleft()
            if expired is not self.anchor:
                reference = weakref.ref(expired)
                del expired
                assert reference() is None, "an evicted output is still retained by the renderer"
                self.released += 1

    def verify(self):
        import numpy as np

        if self.anchor is not None:
            np.testing.assert_array_equal(self.anchor, self.snapshot)

    def bytes(self):
        arrays = {id(array): array for array in self.queue}
        if self.anchor is not None:
            arrays[id(self.anchor)] = self.anchor
        return sum(array.nbytes for array in arrays.values())

    def clear(self):
        self.verify()
        references = [weakref.ref(array) for array in self.queue]
        if self.anchor is not None:
            references.append(weakref.ref(self.anchor))
        self.queue.clear()
        self.anchor = self.snapshot = None
        assert all(reference() is None for reference in references), (
            "outputs survived consumer release"
        )


class StreamCase:
    """One finite source bank and two fixtures; only the current path stays cached."""

    def __init__(self, cli, device, baseline_modules):
        import torch
        from benchmark_paste_gpu_batch import make_adapter_case, make_case
        from benchmark_shared_paste import RenderPath, SyntheticFixture

        self.cli, self.device = cli, device
        torch.manual_seed(24680)
        maximum_batch = max(cli.batch, *cli.batches)
        args, self.base_boxes = make_case(
            device,
            batch=maximum_batch,
            height=cli.height,
            width=cli.width,
            slots=cli.slots,
            realistic=True,
            kernel_size=cli.kernel_size,
            source_sizes=cli.ad_sizes,
        )
        self.sources = args[1]
        self.variants = {}
        for batch in sorted({cli.batch, *cli.batches}):
            for rotation in range(len(cli.ad_sizes)):
                ads = tuple(
                    self.sources[(slot + rotation) % cli.slots][:batch] for slot in range(cli.slots)
                )
                variant_args = (args[0], ads, *args[2:])
                owner, frame, *_ = make_adapter_case(variant_args, self.base_boxes)
                self.variants[batch, rotation] = (variant_args, owner.imgs_raw_groups, frame)
        fixture_options = SimpleNamespace(
            batch=cli.batch,
            height=cli.height,
            width=cli.width,
            slots=cli.slots,
            kernel_size=cli.kernel_size,
            ad_sizes=cli.ad_sizes,
            sigma_pattern="continuous",
            acceleration="triton" if device.type == "cuda" else "eager",
            dense_blur_backend="triton",
            normalization="legacy",
            camera_index_policy="legacy",
            align_corners=False,
            forward=False,
            inverse_iterations=10,
        )
        initial = self.variants[cli.batch, 0][0]
        self.current = RenderPath(
            "current",
            SyntheticFixture(
                SimpleNamespace(**vars(fixture_options)), device, initial, self.base_boxes
            ),
            optimized=True,
            separable=True,
        )
        self.oracle = RenderPath(
            "baseline",
            SyntheticFixture(
                SimpleNamespace(**vars(fixture_options)), device, initial, self.base_boxes
            ),
            optimized=True,
            separable=True,
        )
        old_adapter, old_shared = baseline_modules
        self.oracle.builder = old_shared.SharedParameterBuilder(
            acceleration=fixture_options.acceleration,
            normalization="legacy",
            camera_index_policy="legacy",
            enable_separable=True,
        )
        old_options = {
            field.name: (
                self.oracle.builder.build_warp_grids
                if field.name == "warp_grids_builder"
                else getattr(self.oracle.adapter.options, field.name)
            )
            for field in fields(old_adapter.AdapterOptions)
        }
        self.oracle.adapter = old_adapter.PasteBatchAdapter(
            old_adapter.AdapterOptions(**old_options)
        )
        self.revision = 0

    def prepare(self, phase, index, step, *, update_pixels=True):
        from benchmark_backend_comparison import moving_boxes

        if phase == "fixed":
            batch, rotation, layout = self.cli.batch, 0, "static"
        else:
            epoch = index // self.cli.change_every
            batch = self.cli.batches[epoch % len(self.cli.batches)]
            rotation = (epoch // len(self.cli.batches)) % len(self.cli.ad_sizes)
            layout = ("translate", "resize")[epoch % 2]
            if update_pixels and index % self.cli.revision_every == 0:
                # Actual video-like pixel changes, not only a synthetic cache-version bump.
                self.sources[0][:, :3].add_(0.125).fmod_(255.0)
                self.revision += 1
        variant, groups, frame = self.variants[batch, rotation]
        boxes = moving_boxes(self.base_boxes, step, layout, self.cli.height, self.cli.width)
        for path in (self.current, self.oracle):
            fixture = path.fixture
            fixture.cli.batch = batch
            fixture.image_args, fixture.frame, fixture.boxes = variant, frame, boxes
            fixture.owner.imgs_raw_groups = groups
            fixture.owner.imgs_raw_version = self.revision
            for param, box, ad in zip(fixture.data["ad_params"], boxes, variant[1]):
                param["bbox"], param["ad_h_w"] = box, tuple(ad.shape[-2:])
            fixture.set_frame(step)
        return {"batch": batch, "rotation": rotation, "layout": layout, "revision": self.revision}

    def compare(self, actual):
        import numpy as np

        try:
            expected = self.oracle.render()
            np.testing.assert_array_equal(actual, expected)
            assert self.current.adapter.last_execution == self.oracle.adapter.last_execution, (
                "batch auto execution policy differs from the preserved baseline"
            )
            assert not np.shares_memory(actual, expected)
        finally:
            # Oracle checks must not accumulate old revisions or Gaussian tensors.
            self.oracle.adapter.invalidate_assets()
            self.oracle.builder.clear_cache()
        return {"bitexact": True, "execution": self.current.adapter.last_execution.copy()}


def append_bounded(items, value, limit):
    items.append(value)
    if len(items) > limit:
        del items[0]


def scalar_metrics(sample):
    metrics = {key: sample[key] for key in ("rss_bytes", "private_bytes")}
    for category in ("cuda", "pinned"):
        metrics.update(
            {category + "." + key: value for key, value in sample.get(category, {}).items()}
        )
    for key in ("bytes", "current_bytes", "bucket_bytes", "retained_stack_bytes"):
        metrics["assets." + key] = sample["assets"].get(key)
    return {key: value for key, value in metrics.items() if isinstance(value, (int, float))}


def summarize_windows(windows):
    """Report growth, never equate allocator reserve growth with a proven leak."""
    if not windows:
        return {}
    first, last = scalar_metrics(windows[0]["memory"]), scalar_metrics(windows[-1]["memory"])
    return {
        key: {"first": first[key], "last": value, "delta": value - first[key]}
        for key, value in last.items()
        if key in first
    }


def run_phase(case, cli, device, name, count, report, step):
    import torch
    from benchmark_backend_comparison import check_contract
    from benchmark_support import latency_statistics, save_report

    phase = {
        "phase": name,
        "requested_frames": count,
        "completed_frames": 0,
        "windows": [],
        "baseline_checks": [],
        "status": "running",
    }
    report["phases"].append(phase)
    # Warm every finite B/source-order combination before measuring the dynamic phase.
    warmup = (
        max(cli.warmup, len(cli.batches) * len(cli.ad_sizes)) if name == "dynamic" else cli.warmup
    )
    for index in range(warmup):
        signature_index = index * cli.change_every if name == "dynamic" else index
        case.prepare(name, signature_index, step, update_pixels=False)
        output = case.current.render()
        if index < len(cli.batches) * len(cli.ad_sizes):
            case.compare(output)
        del output
        step += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    retained = RetainedOutputs(cli.retain_outputs)
    phase["after_warmup"] = memory_sample(case.current, device, 0)
    latencies = []
    elapsed_sum = 0.0
    try:
        for index in range(count):
            state = case.prepare(name, index, step)
            started = time.perf_counter()
            output = case.current.render()
            milliseconds = (time.perf_counter() - started) * 1000
            check_contract(output, state["batch"], cli.height, cli.width)
            retained.add(output)
            latencies.append(milliseconds)
            elapsed_sum += milliseconds
            if index % cli.check_every == 0 or index == count - 1:
                checked = case.compare(output)
                append_bounded(
                    phase["baseline_checks"], {"frame": index, **state, **checked}, cli.max_samples
                )
                retained.verify()
            del output
            phase["completed_frames"] = index + 1
            step += 1
            if (index + 1) % cli.sample_every == 0 or index == count - 1:
                retained.verify()
                sample = {
                    "frame": index + 1,
                    **state,
                    "render_ms": latency_statistics(latencies),
                    "memory": memory_sample(case.current, device, retained.bytes()),
                }
                append_bounded(phase["windows"], sample, cli.max_samples)
                latencies.clear()
                phase["window_deltas"] = summarize_windows(phase["windows"])
                save_report(report, cli.output)
                print(
                    json.dumps(
                        {
                            "phase": name,
                            "frame": index + 1,
                            "rss_bytes": sample["memory"]["rss_bytes"],
                            "assets": sample["memory"]["assets"],
                        }
                    ),
                    flush=True,
                )
        phase["render_mean_ms"] = elapsed_sum / count
        phase["weakref_releases"] = retained.released
        retained.clear()
        phase["after_consumer_release"] = memory_sample(case.current, device, 0)
        phase["status"] = "complete"
    finally:
        # Keep a bounded partial report on exceptions; do not hide failed assertions.
        save_report(report, cli.output)
    return step


def main():
    cli = parse_arguments()
    import torch
    from benchmark_support import save_report

    faulthandler.enable()
    device = torch.device(cli.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    else:
        torch.set_num_threads(1)
    baseline = cli.baseline.resolve()
    baseline_modules = load_baseline(baseline)
    report = {
        "scope": "synthetic streaming stability; bounded consumer and diagnostic history",
        "status": "running",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(cli).items()
        },
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "current_sources": source_hashes(Path(__file__).resolve().parent, include_harness=True),
        "baseline_sources": source_hashes(baseline),
        "phases": [],
        "notes": [
            "RSS includes input bank, consumer outputs and one ordinary NumPy anchor snapshot.",
            "GPU/pinned reserved bytes include allocator caches; growth alone is not a leak.",
            "Inconsistent pinned byte counters are preserved raw and flagged as invalid for live-byte analysis.",
            "Periodic baseline caches are cleared and its output released before sampling.",
            "Allocator peaks/cumulative calls are process-wide and include periodic oracle work.",
            "Sources, callbacks, B/signatures are synthetic; no decoder/encoder/network queue.",
            "Each phase keeps only the latest max_samples windows/check records.",
            "No per-frame gc.collect or empty_cache; live output storage remains independent.",
        ],
    }
    try:
        with torch.inference_mode():
            case = StreamCase(cli, device, baseline_modules)
            step = run_phase(case, cli, device, "fixed", cli.frames // 2, report, 0)
            run_phase(case, cli, device, "dynamic", cli.frames - cli.frames // 2, report, step)
            verify_baseline_sources(baseline)
        report["status"] = "complete"
        report["all_baseline_checks_bitexact"] = True
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save_report(report, cli.output)
    print(
        json.dumps({"status": report["status"], "frames": cli.frames, "output": str(cli.output)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
