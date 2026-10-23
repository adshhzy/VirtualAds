"""Full-call batch/bucketed comparison using synthetic, deterministic inputs.

Includes shared preparation, live ROI geometry, asset/packing/layout work and
completed independent NumPy output. Fixture updates alone are outside timing.
Every candidate owns its builder/adapter/camera state; immutable pixels are shared.
Differences of one uint8 unit are reported, timed, and fail the exactness gate.
Larger differences or invalid output contracts fail the run with a saved report.
No result from this tool changes the production execution policy.
"""

from __future__ import annotations

from benchmarks.source_layout import REPOSITORY_ROOT, runtime_sources, source_file, source_key

import argparse
from contextlib import contextmanager
from dataclasses import fields, replace
import fnmatch
import gc
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from benchmarks.benchmark_support import (
    latency_statistics,
    rotated_order,
    save_report,
    serializable_arguments,
)


CANDIDATES = {
    "batch_whole": ("batch", 0),
    "batch_auto": ("batch", "auto"),
    "bucketed": ("bucketed", 0),
    "batch_c4": ("batch", 4),
    "before_bucketed": ("bucketed", 0),
}
PROFILE_COUNTS = {"one": 1, "two": 2, "three": 3}
CHECK_STEPS = (0, 1, 7)


def parse_arguments():
    from benchmarks.benchmark_paste_gpu_batch import parse_ad_size

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--slots", type=int, nargs="+", default=[16])
    parser.add_argument("--profiles", choices=PROFILE_COUNTS, nargs="+", default=None)
    parser.add_argument(
        "--ad-sizes", type=parse_ad_size, nargs="+", default=[(200, 200), (500, 500), (300, 600)]
    )
    parser.add_argument(
        "--layouts",
        choices=("static", "translate", "resize"),
        nargs="+",
        default=["static", "translate", "resize"],
    )
    parser.add_argument(
        "--candidates",
        choices=CANDIDATES,
        nargs="+",
        default=["batch_whole", "batch_auto", "bucketed"],
    )
    parser.add_argument("--only-case", nargs="+", help="case ID glob(s), e.g. B8_S16_*_resize")
    parser.add_argument(
        "--baseline", type=Path, help="preserved source directory required for before_bucketed"
    )
    parser.add_argument(
        "--require-exact",
        choices=CANDIDATES,
        nargs="+",
        default=[],
        help="fail on any nonzero pixel difference for these selected candidates",
    )
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument(
        "--blur-backend", choices=("separable", "triton", "torch"), default="separable"
    )
    parser.add_argument("--acceleration", choices=("eager", "auto", "triton"))
    parser.add_argument("--sigma-pattern", choices=("continuous", "repeated"), default="continuous")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--check-timed-steps",
        action="store_true",
        help="check every timed frame index as well as steps 0, 1 and 7 before timing",
    )
    parser.add_argument(
        "--skip-cold-assets",
        action="store_true",
        help="omit the separate single-call cold asset measurement",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="separate untimed instrumentation/profile pass after main measurements",
    )
    parser.add_argument("--output", type=Path, default=Path("results/backend_comparison.json"))
    options = parser.parse_args()
    if min(*options.batches, *options.slots, options.iterations, options.rounds) < 1:
        parser.error("batches, slots, iterations and rounds must be positive")
    if min(options.height, options.width) < 8 or options.warmup < 0:
        parser.error("canvas dimensions must be >=8 and warmup nonnegative")
    if not 1 <= len(options.ad_sizes) <= 3 or any(min(size) < 2 for size in options.ad_sizes):
        parser.error("provide one to three source sizes, each dimension >=2")
    if options.kernel_size < 1 or options.kernel_size % 2 != 1:
        parser.error("kernel-size must be positive and odd")
    if options.profiles is None:
        options.profiles = [
            name for name, count in PROFILE_COUNTS.items() if count <= len(options.ad_sizes)
        ]
    if any(PROFILE_COUNTS[name] > len(options.ad_sizes) for name in options.profiles):
        parser.error("each profile requires enough --ad-sizes entries")
    if len(set(options.candidates)) != len(options.candidates):
        parser.error("candidate names must not repeat")
    if "before_bucketed" in options.candidates and options.baseline is None:
        parser.error("before_bucketed requires --baseline")
    if any(name not in options.candidates for name in options.require_exact):
        parser.error("--require-exact names must be included in --candidates")
    if options.device == "cpu" and "batch_c4" in options.candidates:
        parser.error("manual batch_c4 requires CUDA; omit it for CPU fixture checks")
    return options


def source_provenance():
    """Capture local source bytes before rendering; HEAD alone misses uncommitted edits."""
    script = Path(__file__).resolve()
    directory = REPOSITORY_ROOT
    sources = {
        script,
        *runtime_sources(directory),
        *(
            source_file(directory, name)
            for name in (
                "source_layout.py",
                "benchmark_shared_paste.py",
                "benchmark_paste_gpu_batch.py",
                "benchmark_support.py",
                "benchmark_dynamic_comparison.py",
            )
        ),
    }
    hashes = {
        source_key(path, directory): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(sources)
    }
    result = {
        "scope": "on-disk source snapshot at startup; includes harness, fixtures and runtime",
        "sha256_by_file": hashes,
        "combined_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "git_head": None,
        "git_worktree_dirty": None,
    }
    try:
        git_options = {
            "cwd": directory,
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": 5,
        }
        if sys.platform == "win32":
            git_options["creationflags"] = subprocess.CREATE_NO_WINDOW
        result["git_head"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], **git_options
        ).stdout.strip()
        result["git_worktree_dirty"] = bool(
            subprocess.run(["git", "status", "--porcelain"], **git_options).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError) as error:
        result["git_metadata_error"] = str(error)
    return result


def triton_versions():
    versions = {}
    for distribution in ("triton", "triton-windows"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    try:
        import triton
    except ImportError:
        versions["triton_module"] = None
    else:
        versions["triton_module"] = triton.__version__
    return versions


def baseline_source_provenance(directory):
    """Identify the preserved runtime used by the isolated before candidate."""
    sources = runtime_sources(directory)
    hashes = {
        source_key(path, directory): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    return {
        "directory": str(directory),
        "scope": "preserved paste*.py runtime; fixture and timer use the current harness",
        "sha256_by_file": hashes,
        "combined_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def synchronize(device):
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize(device)


def cache_snapshot(adapter):
    info = adapter.cache_info()
    return {**info, "sizes": {f"{h}x{w}": count for (h, w), count in info["sizes"].items()}}


def moving_boxes(base_boxes, step, layout, height, width):
    if layout == "static":
        return base_boxes
    updated = []
    for index, (x1, y1, x2, y2) in enumerate(base_boxes):
        # First two slots stay coincident, exercising transparent ROI overwrite.
        motion_index = 0 if index == 1 else index
        roi_height, roi_width = y2 - y1, x2 - x1
        if layout == "resize":
            roi_height += ((step + motion_index) % 5 - 2) * max(1, roi_height // 8)
            roi_width += ((step + motion_index) % 7 - 3) * max(1, roi_width // 12)
            roi_height = min(height - 1, max(2, roi_height))
            roi_width = min(width - 1, max(2, roi_width))
        x1 = (x1 + step * (3 + motion_index % 3)) % (width - roi_width + 1)
        y1 = (y1 + step * (2 + motion_index % 2)) % (height - roi_height + 1)
        updated.append((x1, y1, x1 + roi_width, y1 + roi_height))
    return tuple(updated)


def create_fixture(options, device, image_args, boxes, layout):
    from benchmarks.benchmark_shared_paste import SyntheticFixture

    class LayoutFixture(SyntheticFixture):
        def set_frame(self, step):
            self.boxes = moving_boxes(boxes, step, layout, self.cli.height, self.cli.width)
            for param, box in zip(self.data["ad_params"], self.boxes):
                param["bbox"] = box
            # Camera values and matrix coefficients update here; no grids,
            # bucket layouts, palettes or packed opacities are generated.
            super().set_frame(step)

    return LayoutFixture(SimpleNamespace(**vars(options)), device, image_args, boxes)


class CandidateState:
    def __init__(self, name, fixture, blur_backend, baseline_modules=None):
        from benchmarks.benchmark_shared_paste import RenderPath
        from virtualads.paste_gpu_batch_adapter import PasteBatchAdapter

        self.path = RenderPath(name, fixture, optimized=True, separable=blur_backend == "separable")
        backend, chunk = CANDIDATES[name]
        self.path.adapter = PasteBatchAdapter(
            replace(
                self.path.adapter.options,
                backend=backend,
                batch_chunk_size=chunk,
                blur_backend=blur_backend,
                blur_launch_config=None,
                profile_every=0,
            )
        )
        if name == "before_bucketed":
            baseline_adapter, baseline_shared = baseline_modules
            cli = fixture.cli
            self.path.builder = baseline_shared.SharedParameterBuilder(
                acceleration=cli.acceleration,
                normalization=cli.normalization,
                distortion_inverse=not cli.forward,
                iterations=cli.inverse_iterations,
                camera_index_policy=cli.camera_index_policy,
                distortion_align_corners=cli.align_corners,
                enable_separable=blur_backend == "separable",
            )
            baseline_options = {
                field.name: (
                    self.path.builder.build_warp_grids
                    if field.name == "warp_grids_builder"
                    else getattr(self.path.adapter.options, field.name)
                )
                for field in fields(baseline_adapter.AdapterOptions)
            }
            self.path.adapter = baseline_adapter.PasteBatchAdapter(
                baseline_adapter.AdapterOptions(**baseline_options)
            )
        self.previous = None
        self.snapshot = None

    def retain_checked(self, output):
        import numpy as np

        if self.previous is not None:
            if not np.array_equal(self.previous, self.snapshot):
                raise AssertionError("a retained NumPy output was overwritten")
            if np.shares_memory(output, self.previous):
                raise AssertionError("successive outputs share storage")
        self.previous = output
        self.snapshot = output.copy()


def check_contract(output, batch, height, width):
    import numpy as np

    expected = (height, width, 3) if batch == 1 else (batch, height, width, 3)
    if not isinstance(output, np.ndarray) or output.dtype != np.uint8:
        raise AssertionError("output must be NumPy uint8")
    if output.shape != expected or not output.flags.c_contiguous:
        raise AssertionError(f"expected contiguous {expected}, got {output.shape}/{output.strides}")


def output_difference(actual, expected):
    import numpy as np

    # Exact candidates need no large int16 difference image or per-pixel reduce.
    # This is outside timing and keeps the same full-array equality requirement.
    if np.array_equal(actual, expected):
        return {
            "max_abs_diff": 0,
            "different_pixel_ratio": 0.0,
            "different_value_ratio": 0.0,
            "bitexact": True,
        }
    difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    return {
        "max_abs_diff": int(difference.max()),
        "different_pixel_ratio": float(np.mean(np.any(difference != 0, axis=-1))),
        "different_value_ratio": float(np.mean(difference != 0)),
        "bitexact": False,
    }


def check_outputs(states, reference, options, batch, row):
    import numpy as np

    checks = {name: [] for name in states}
    row["correctness"] = checks
    checked_steps = set(CHECK_STEPS)
    timed_steps = range(options.warmup, options.warmup + options.iterations)
    if options.check_timed_steps:
        checked_steps.update(timed_steps)
    row["checked_steps"] = sorted(checked_steps)
    row["auto_correctness_gate_scope"] = "only checked_steps for this synthetic case"
    row["all_timed_steps_checked"] = checked_steps.issuperset(timed_steps)
    first_outputs = {}
    for step in row["checked_steps"]:
        reference.path.fixture.set_frame(step)
        expected = reference.path.render()
        check_contract(expected, batch, options.height, options.width)
        reference.retain_checked(expected)
        frozen_expected = expected.copy()
        for name, state in states.items():
            if state is reference:
                output = expected
            else:
                state.path.fixture.set_frame(step)
                output = state.path.render()
                check_contract(output, batch, options.height, options.width)
                state.retain_checked(output)
            if name not in first_outputs:
                first_outputs[name] = output, output.copy()
            difference = output_difference(output, frozen_expected)
            checks[name].append(
                {"step": step, **difference, "execution": state.path.adapter.last_execution.copy()}
            )
            allowed_difference = 0 if name in options.require_exact else 1
            if difference["max_abs_diff"] > allowed_difference:
                row["fatal_correctness_error"] = {"candidate": name, "step": step, **difference}
                raise AssertionError(
                    f"{name} step {step}: uint8 difference exceeds {allowed_difference}: {difference}"
                )
        if not np.array_equal(reference.previous, frozen_expected):
            raise AssertionError("reference output changed while checking other candidates")
        for name, state in states.items():
            original, original_snapshot = first_outputs[name]
            if not np.array_equal(original, original_snapshot):
                raise AssertionError(f"{name}: first output changed after later frame renders")
            if not np.array_equal(state.previous, state.snapshot):
                raise AssertionError(f"{name}: last output changed while checking other candidates")
    row["auto_correctness_gate"] = {
        name: all(item["bitexact"] for item in samples) for name, samples in checks.items()
    }
    for state in [*states.values(), reference]:
        state.snapshot = None  # Keep the previous output, not its diagnostic copy, during timing.


def measure_candidate(state, options, device):
    import torch

    path, adapter = state.path, state.path.adapter
    for step in range(options.warmup):
        path.fixture.set_frame(step)
        state.previous = path.render()
    synchronize(device)
    cache_before = cache_snapshot(adapter)
    kernel_cache_before = path.builder.cache_info()
    memory_before = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    rebuilds = repacks = 0
    for sample in range(options.iterations):
        step = options.warmup + sample
        path.fixture.set_frame(step)
        previous_layout_id = id(adapter._layout)
        previous_packed_assets_id = id(adapter._bucket_ads)
        synchronize(device)
        started = time.perf_counter()
        output = path.render()
        synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000
        # Replacement happens only after return/completion, identically for all candidates.
        state.previous = output
        rebuilt = id(adapter._layout) != previous_layout_id
        repacked = id(adapter._bucket_ads) != previous_packed_assets_id
        rebuilds += rebuilt
        repacks += repacked
        samples.append(
            {
                "step": step,
                "wall_ms": elapsed_ms,
                "layout_rebuilt": rebuilt,
                "assets_repacked": repacked,
                "execution": adapter.last_execution.copy(),
            }
        )
    return {
        "wall_ms": latency_statistics([item["wall_ms"] for item in samples], include_minimum=True),
        "samples": samples,
        "layout_rebuilds": rebuilds,
        "asset_repacks": repacks,
        "asset_cache_before": cache_before,
        "asset_cache_after": cache_snapshot(adapter),
        "kernel_cache_before": kernel_cache_before,
        "kernel_cache_after": path.builder.cache_info(),
        "memory": {
            "resident_allocated_before_bytes": memory_before,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
            if device.type == "cuda"
            else None,
            "peak_extra_allocated_bytes": (torch.cuda.max_memory_allocated(device) - memory_before)
            if device.type == "cuda"
            else None,
            "scope": "process-wide; other candidates' resident caches are included in the baseline",
        },
    }


def cold_assets_call(state, options, device):
    path, adapter = state.path, state.path.adapter
    # Prime this exact frame's kernels/layout, then invalidate only assets. The
    # invalidation is outside timing; reconstruction and packing occur inside it.
    step = options.warmup + options.iterations - 1
    path.fixture.set_frame(step)
    state.previous = path.render()
    synchronize(device)
    prior_layout = adapter._layout
    adapter.invalidate_assets()
    before = cache_snapshot(adapter)
    started = time.perf_counter()
    output = path.render()
    synchronize(device)
    elapsed = (time.perf_counter() - started) * 1000
    state.previous = output
    return {
        "cold_assets_ms": elapsed,
        "samples": 1,
        "not_a_p95_or_auto_basis": True,
        "kernels_preprimed_for_same_step": True,
        "layout_existed_before": prior_layout is not None,
        "layout_preserved_by_asset_invalidation": adapter._layout is prior_layout,
        "layout_rebuilt": adapter._layout is not prior_layout,
        "asset_cache_before": before,
        "asset_cache_after": cache_snapshot(adapter),
    }


@contextmanager
def count_bucket_operations(module):
    """Benchmark-only wrappers: never installed during headline timing."""
    names = (
        "make_bucket_layout",
        "pack_bucket_ads",
        "pack_bucket_grids",
        "pack_bucket_transparencies",
    )
    originals = {name: getattr(module, name) for name in names}
    counts = dict.fromkeys(names, 0)
    try:
        for name, function in originals.items():

            def counted(*args, _name=name, _function=function, **kwargs):
                counts[_name] += 1
                return _function(*args, **kwargs)

            setattr(module, name, counted)
        yield counts
    finally:
        for name, function in originals.items():
            setattr(module, name, function)


def diagnose_candidate(state, device):
    """Separate staged pass; its synchronization/monkeypatch overhead is not latency evidence."""
    path, adapter = state.path, state.path.adapter
    adapter.options = replace(adapter.options, profile_every=1)
    samples = []
    with count_bucket_operations(sys.modules[type(adapter).__module__]) as counts:
        for step in (0, 0, 1, 7):
            path.fixture.set_frame(step)
            synchronize(device)
            started = time.perf_counter()
            values = path.prepare()
            synchronize(device)
            preparation_ms = (time.perf_counter() - started) * 1000
            fixture = path.fixture
            state.previous = adapter(
                fixture.owner, fixture.frame, fixture.data, fixture.frame_id, fixture.way_id, values
            )
            samples.append(
                {
                    "step": step,
                    "shared_prepare_ready_ms": preparation_ms,
                    "adapter_profile": adapter.last_profile.copy(),
                }
            )
    return {
        "separate_diagnostic_pass": True,
        "samples": samples,
        "bucket_calls": counts,
        "asset_cache": cache_snapshot(adapter),
        "kernel_cache": path.builder.cache_info(),
    }


def run_case(options, device, batch, slots, profile, layout, report, baseline_modules=None):
    import torch
    from benchmarks.benchmark_paste_gpu_batch import make_case

    case_id = f"B{batch}_S{slots}_{profile}_{layout}"
    sizes = tuple(options.ad_sizes[: PROFILE_COUNTS[profile]])
    torch.manual_seed(options.seed)
    image_args, boxes = make_case(
        device,
        batch=batch,
        height=options.height,
        width=options.width,
        slots=slots,
        realistic=True,
        kernel_size=options.kernel_size,
        source_sizes=sizes,
    )
    if slots >= 2:
        boxes = (boxes[0], boxes[0], *boxes[2:])
    fixture_options = SimpleNamespace(
        batch=batch,
        slots=slots,
        height=options.height,
        width=options.width,
        kernel_size=options.kernel_size,
        ad_sizes=sizes,
        sigma_pattern=options.sigma_pattern,
        acceleration=options.acceleration,
        dense_blur_backend=options.blur_backend,
        normalization="legacy",
        camera_index_policy="legacy",
        align_corners=False,
        forward=False,
        inverse_iterations=10,
    )

    def state_for(name):
        fixture = create_fixture(fixture_options, device, image_args, boxes, layout)
        return CandidateState(name, fixture, options.blur_backend, baseline_modules)

    states = {name: state_for(name) for name in options.candidates}
    reference = states.get("batch_whole") or state_for("batch_whole")
    row = {
        "case_id": case_id,
        "batch": batch,
        "slots": slots,
        "profile": profile,
        "source_sizes": sizes,
        "layout": layout,
        "rounds": [],
        "fixture": "coincident first two slots when available; source alpha has transparent rows",
        "reference": "batch_whole, separate mutable state unless itself selected",
    }
    report["results"].append(row)
    try:
        check_outputs(states, reference, options, batch, row)
        save_report(report, options.output)
        print(
            json.dumps({"case_id": case_id, "auto_correctness_gate": row["auto_correctness_gate"]}),
            flush=True,
        )
        if not options.check_only:
            for round_index in range(options.rounds):
                measurements = []
                for name in rotated_order(options.candidates, round_index):
                    measurement = {
                        "candidate": name,
                        **measure_candidate(states[name], options, device),
                    }
                    measurements.append(measurement)
                    print(
                        json.dumps(
                            {
                                "case_id": case_id,
                                "round": round_index,
                                "candidate": name,
                                "wall_ms": measurement["wall_ms"],
                                "layout_rebuilds": measurement["layout_rebuilds"],
                            }
                        ),
                        flush=True,
                    )
                row["rounds"].append(measurements)
                save_report(report, options.output)
            row["summary"] = {
                name: latency_statistics(
                    [
                        sample["wall_ms"]
                        for round_items in row["rounds"]
                        for item in round_items
                        if item["candidate"] == name
                        for sample in item["samples"]
                    ],
                    include_minimum=True,
                )
                for name in options.candidates
            }
            if not options.skip_cold_assets:
                row["cold_assets"] = {
                    name: cold_assets_call(state, options, device) for name, state in states.items()
                }
        if options.diagnose:
            row["diagnostics"] = {
                name: diagnose_candidate(state_for(name), device) for name in options.candidates
            }
        row["status"] = "complete"
    except Exception as error:
        row["status"] = "failed"
        row["error"] = f"{type(error).__name__}: {error}"
        save_report(report, options.output)
        raise
    save_report(report, options.output)


def main():
    options = parse_arguments()
    import torch

    baseline_modules = None
    if "before_bucketed" in options.candidates:
        from benchmarks.benchmark_dynamic_comparison import load_baseline_modules

        options.baseline = options.baseline.resolve()
        baseline_modules = load_baseline_modules(options.baseline)

    device = torch.device(options.device)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("only CUDA and CPU fixture checks are supported")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    options.acceleration = options.acceleration or ("triton" if device.type == "cuda" else "eager")
    report = {
        "scope": "shared preparation through completed independent NumPy; synthetic callbacks",
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton_versions": triton_versions(),
        "source_provenance": source_provenance(),
        "baseline_source_provenance": baseline_source_provenance(options.baseline)
        if baseline_modules is not None
        else None,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "arguments": serializable_arguments(options),
        "results": [],
        "auto_rule": "bitexact on each case's checked_steps is necessary, not sufficient; nonzero differences are ineligible",
        "timing_notes": "fixture updates excluded; retained previous output per candidate; no wrappers in timed calls",
    }
    with torch.inference_mode():
        for batch in options.batches:
            for slots in options.slots:
                for profile in options.profiles:
                    for layout in options.layouts:
                        case_id = f"B{batch}_S{slots}_{profile}_{layout}"
                        if options.only_case and not any(
                            fnmatch.fnmatchcase(case_id, value) for value in options.only_case
                        ):
                            continue
                        run_case(
                            options, device, batch, slots, profile, layout, report, baseline_modules
                        )
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
    if not report["results"]:
        raise ValueError("no cases matched --only-case")
    save_report(report, options.output)


if __name__ == "__main__":
    main()
