"""Compare complete old/current render_img calls on the same B8/S16 fixture.

Both candidates use dense 5x5 blur and the unchanged automatic chunk policy.
The baseline is loaded as a separate relative-import package, including all its
paste modules. Every measured output is compared byte for byte with a baseline
reference outside the timer. These are stationary-camera demonstration callbacks,
not a replacement for production motion or Gaussian callbacks.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.machinery
import json
from pathlib import Path
import sys
import time
from types import ModuleType
import uuid

import numpy as np
import torch

from benchmarks.benchmark_render_img_modes import (
    BATCH,
    HISTORY_CAPACITY,
    SLOTS,
    Inputs,
    array_hash,
    assert_closed,
    effective_options,
    expanded_fixture,
    file_hash,
    fixture_boxes,
    image_difference,
    json_ready,
    layout_offsets,
    memory_state,
    output_protocol,
    provenance as environment_provenance,
    read_bmp,
    stats,
)
from benchmarks.benchmark_support import rotated_order, serializable_arguments
from benchmarks.source_layout import (
    REPOSITORY_ROOT,
    runtime_directory,
    runtime_sources,
    source_file,
    source_key,
)
from tools.check_render_img import load_sample
from examples.live_render_demo import SyntheticOwner
from virtualads import paste_renderer as current_api

ROOT = REPOSITORY_ROOT
CANDIDATES = ("baseline", "current")
REQUIRED_MODULES = (
    "paste_renderer",
    "paste_render_executor",
    "paste_shared_params",
    "paste_shared_mask",
    "paste_route_plan",
    "paste_gpu_batch",
)


def source_hashes(directory, *, include_harness=False):
    """Hash source contents, including uncommitted changes, outside render timing."""
    directory = Path(directory).resolve()
    paths = set(runtime_sources(directory))
    if include_harness:
        paths.update(
            source_file(directory, name)
            for name in (
                Path(__file__).name,
                "source_layout.py",
                "benchmark_render_img_modes.py",
                "benchmark_support.py",
                "check_render_img.py",
                "live_render_demo.py",
            )
        )
    files = {source_key(path, directory): file_hash(path) for path in sorted(paths)}
    combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    return {"directory": str(directory), "files": files, "combined_sha256": combined}


def validate_source_directory(directory):
    directory = runtime_directory(directory)
    paths = {path.stem: path for path in directory.glob("paste*.py")}
    required = set(REQUIRED_MODULES)
    for path in paths.values():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name.split(".")[0] for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module.split(".")[0]] if node.module else []
                if node.level and node.module is None:
                    names.extend(item.name for item in node.names)
            required.update(name for name in names if name.startswith("paste_"))
    missing = sorted(required.difference(paths))
    if missing:
        raise ValueError(f"incomplete baseline {directory}: missing {', '.join(missing)}")
    return directory, paths


class LoadedSources:
    """One isolated source package; close only after its workers have been closed."""

    def __init__(self, directory):
        self.directory, self.paths = validate_source_directory(directory)
        self.package_name = f"_paste_upload_baseline_{uuid.uuid4().hex}"
        self.provenance = source_hashes(self.directory)
        package = ModuleType(self.package_name)
        package.__package__ = self.package_name
        package.__path__ = [str(self.directory)]
        package.__spec__ = importlib.machinery.ModuleSpec(
            self.package_name, loader=None, is_package=True
        )
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules[self.package_name] = package
        try:
            # Import lazy kernels as well, before rendering and before verification.
            for name in sorted(self.paths):
                importlib.import_module(f"{self.package_name}.{name}")
            self.api = sys.modules[f"{self.package_name}.paste_renderer"]
            self.verify()
        except BaseException:
            self.close()
            raise

    def verify(self):
        """Check module locations and reject reexports from the current flat tree."""
        for name, path in self.paths.items():
            module = sys.modules.get(f"{self.package_name}.{name}")
            if module is None or Path(module.__file__).resolve() != path.resolve():
                raise RuntimeError(f"baseline module came from the wrong source: {name}")
            for value in vars(module).values():
                origin = (
                    value.__name__
                    if isinstance(value, ModuleType)
                    else getattr(value, "__module__", "")
                )
                parts = origin.split(".") if isinstance(origin, str) else []
                if "virtualads" in parts or any(part.startswith("paste_") for part in parts):
                    if not origin.startswith(f"{self.package_name}."):
                        raise RuntimeError(f"baseline {name} imports outside its package: {origin}")
        if source_hashes(self.directory) != self.provenance:
            raise RuntimeError("baseline source files changed after loading")

    def close(self):
        for name in tuple(sys.modules):
            if name == self.package_name or name.startswith(f"{self.package_name}."):
                del sys.modules[name]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def load_sources(directory):
    """Return LoadedSources with .api, .provenance, .verify() and .close()."""
    return LoadedSources(directory)


def make_worker(api, device):
    """Build the same public dense/auto worker using the supplied module's classes."""
    if isinstance(api, LoadedSources):
        api = api.api
    return api.ImageRenderer(
        SyntheticOwner(kernel_size=5),
        api.RenderOptions(
            frame_layout="HWC",
            ad_layout="HWC",
            resize_ads_to_ad_h_w=True,
            batch_chunk_size="auto",
        ),
        device=device,
        use_separable=False,
        mask_sigma=1.5,
        mask_radius=3,
        history_capacity=HISTORY_CAPACITY,
    )


def check_plan(api, worker, data, assets, frame):
    # The baseline must also plan with its own implementation.
    plan = api.build_route_plan(data, assets, frame.shape[:2], worker.executor.options)
    if (
        len(plan.groups) != 1
        or len(plan.groups[0].indices) != BATCH
        or len(plan.groups[0].params) != SLOTS
    ):
        raise ValueError("fixture must form exactly one compatible B8/S16 layout group")


def execution_snapshot(worker):
    return deepcopy(json_ready(worker.last_execution))


def compact_reference(output):
    """Use one route only after establishing that all eight reference routes agree."""
    equal = all(np.array_equal(output[0], route) for route in output[1:])
    return (output[0] if equal else output).copy(), equal


def validate_output(output, reference, execution, expected_execution, frame_shape):
    """No sampling or tolerance: compare all output bytes, then the routing decision."""
    output_protocol(output, frame_shape)
    if not np.array_equal(output, np.broadcast_to(reference, output.shape)):
        raise AssertionError(f"output differs from baseline: {image_difference(output, reference)}")
    if execution != expected_execution:
        raise AssertionError(
            f"execution policy differs from baseline: {execution!r} != {expected_execution!r}"
        )


def worker_memory(worker, device):
    state = memory_state(worker, device)
    background = getattr(worker, "_frame_upload", None)
    if background is not None:
        state["background_upload"] = background.cache_info()
    return json_ready(state)


def make_references(api, template, offsets, frame, assets, args):
    """Freeze mature-history baseline outputs in ordinary CPU storage, then close it."""
    worker = make_worker(api, args.device)
    inputs = Inputs(template, offsets)
    references, policies, descriptions = [], [], []
    try:
        check_plan(api, worker, inputs.data, assets, frame)
        for step in range(max(HISTORY_CAPACITY, args.warmup)):
            output = worker.render_img(inputs.update(step), frame, assets)
            del output
        for pattern, offset in enumerate(offsets):
            output = worker.render_img(inputs.update(pattern), frame, assets)
            output_protocol(output, frame.shape)
            reference, all_equal = compact_reference(output)
            references.append(reference)
            policies.append(execution_snapshot(worker))
            descriptions.append(
                {
                    "pattern": pattern,
                    "offset_xy": offset,
                    "stored_shape": reference.shape,
                    "bytes": reference.nbytes,
                    "sha256": array_hash(reference),
                    "full_batch_sha256": array_hash(output),
                    "route_sha256": [array_hash(route) for route in output],
                    "all_8_routes_equal": all_equal,
                    "execution": policies[-1],
                }
            )
            del output
        configuration = effective_options(worker)
    finally:
        worker.close()
    closed = worker_memory(worker, args.device)
    assert_closed(closed)
    return (
        references,
        policies,
        {
            "candidate": "baseline",
            "patterns": descriptions,
            "total_cpu_bytes": sum(array.nbytes for array in references),
            "effective_options": configuration,
            "after_close": closed,
            "meaning": "isolated old GPU implementation; not an independent CPU-math oracle",
        },
    )


def retain_output_check(worker, inputs, frame, assets, references, policies):
    """Retain one caller output across another render; caller also checks it after close."""
    first = worker.render_img(inputs.update(0), frame, assets)
    validate_output(first, references[0], execution_snapshot(worker), policies[0], frame.shape)
    snapshot = first.copy()
    pattern = len(inputs.offsets) - 1
    second = worker.render_img(inputs.update(pattern), frame, assets)
    validate_output(
        second, references[pattern], execution_snapshot(worker), policies[pattern], frame.shape
    )
    if np.shares_memory(first, second):
        raise AssertionError("successive caller outputs share storage")
    if not np.array_equal(first, snapshot):
        raise AssertionError("the next render overwrote a retained caller output")
    return first, snapshot


def run_phase(api, name, round_index, template, offsets, frame, assets, references, policies, args):
    worker = make_worker(api, args.device)
    inputs = Inputs(template, offsets)
    samples, validation_samples, pattern_indices = [], [], []
    held = snapshot = None
    try:
        check_plan(api, worker, inputs.data, assets, frame)
        for step in range(max(HISTORY_CAPACITY, args.warmup)):
            output = worker.render_img(inputs.update(step), frame, assets)
            del output
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        before = worker_memory(worker, args.device)
        configuration = effective_options(worker)
        for step in range(args.steps):
            pattern = step % len(offsets)
            data = inputs.update(pattern)
            torch.cuda.synchronize(args.device)
            started = time.perf_counter()
            output = worker.render_img(data, frame, assets)
            torch.cuda.synchronize(args.device)
            samples.append((time.perf_counter() - started) * 1000.0)
            validation_started = time.perf_counter()
            validate_output(
                output,
                references[pattern],
                execution_snapshot(worker),
                policies[pattern],
                frame.shape,
            )
            validation_samples.append((time.perf_counter() - validation_started) * 1000.0)
            pattern_indices.append(pattern)
            del output
        after = worker_memory(worker, args.device)
        held, snapshot = retain_output_check(worker, inputs, frame, assets, references, policies)
    finally:
        worker.close()
    if not np.array_equal(held, snapshot):
        raise AssertionError("close overwrote a retained caller output")
    del held, snapshot
    closed = worker_memory(worker, args.device)
    assert_closed(closed)
    return {
        "candidate": name,
        "round": round_index + 1,
        "warmup_calls": max(HISTORY_CAPACITY, args.warmup),
        "samples_ms": samples,
        "stats_ms": stats(samples),
        "validation_samples_ms": validation_samples,
        "validation_stats_ms": stats(validation_samples),
        "pattern_indices": pattern_indices,
        "pattern_counts": dict(Counter(pattern_indices)),
        "checked_calls": len(samples),
        "checked_route_images": len(samples) * BATCH,
        "max_uint8_diff": 0,
        "bitexact_in_checked_calls": True,
        "all_execution_policies_equal": True,
        "effective_options": configuration,
        "ownership": {"successive_outputs_independent": True, "unchanged_after_close": True},
        "memory_before_timing": before,
        "memory_after_timing": after,
        "memory_after_close": closed,
    }


def aggregate(phases):
    result = {}
    for name in CANDIDATES:
        matching = [phase for phase in phases if phase["candidate"] == name]
        if not matching:
            continue
        samples = [value for phase in matching for value in phase["samples_ms"]]
        result[name] = {
            "stats_ms": stats(samples),
            "round_means_ms": [phase["stats_ms"]["mean"] for phase in matching],
            "round_p95_ms": [phase["stats_ms"]["p95"] for phase in matching],
            "checked_calls": len(samples),
            "checked_route_images": BATCH * len(samples),
            "max_uint8_diff": 0,
            "bitexact_in_checked_calls": True,
            "all_execution_policies_equal": True,
        }
    if set(CANDIDATES).issubset(result):
        old, new = (result[name]["stats_ms"] for name in CANDIDATES)
        result["current_reduction_percent"] = {
            key: 100.0 * (old[key] - new[key]) / old[key] for key in ("mean", "p95")
        }
    return result


def save_progress(report, output):
    output.write_text(
        json.dumps(json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sample", "frame-bmp", "ad-bmp", "baseline", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--layouts", nargs="+", choices=("static", "translated"), default=["static", "translated"]
    )
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args(argv)
    if not __debug__:
        parser.error("run without -O so diagnostic assertions remain enabled")
    if args.rounds <= 0 or args.steps <= 0 or args.warmup < 0:
        parser.error("rounds/steps must be positive and warmup nonnegative")
    if len(set(args.layouts)) != len(args.layouts):
        parser.error("layouts must not contain duplicates")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing report: {args.output}")
    for name in ("sample", "frame_bmp", "ad_bmp"):
        if not getattr(args, name).is_file():
            parser.error(f"input does not exist: {getattr(args, name)}")
    if not args.baseline.is_dir():
        parser.error(f"baseline directory does not exist: {args.baseline}")
    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    if device.type != "cuda":
        parser.error("the complete public render_img benchmark requires CUDA")
    if args.rounds % len(CANDIDATES):
        print(
            "Exploratory order: use an even number of rounds for balanced confirmation.", flush=True
        )
    return args


def main(argv=None):
    args = parse_arguments(argv)
    torch.cuda.set_device(args.device)
    sample = load_sample(args.sample)
    template = expanded_fixture(sample)
    frame, ad = read_bmp(args.frame_bmp), read_bmp(args.ad_bmp)
    rgba = np.concatenate((ad, np.full((*ad.shape[:2], 1), 255, np.uint8)), axis=-1)
    assets = {
        (item["ad_placement_id"], item["ad_id"]): rgba
        for route in range(1, BATCH + 1)
        for item in template[f"ad_params_{route}"]
    }
    source_before = source_hashes(ROOT, include_harness=True)
    with load_sources(args.baseline) as baseline:
        report = {
            "status": "running",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "arguments": serializable_arguments(args),
            "provenance": environment_provenance(args),
            "current_source": source_before,
            "baseline_source": baseline.provenance,
            "versions": {
                "baseline": baseline.api.RENDERER_VERSION,
                "current": current_api.RENDERER_VERSION,
            },
            "balanced_candidate_order": args.rounds % len(CANDIDATES) == 0,
            "batch": BATCH,
            "slots_per_route": SLOTS,
            "frame_shape": frame.shape,
            "raw_ad_shape": ad.shape,
            "asset_lookup_pairs": len(assets),
            "distinct_source_arrays": 1,
            "fixture": "Original numeric routes and ordered slots cycle to B8/S16. All pairs use one opaque RGB source. Source pixels, camera and matrices are stationary; only frame_id and the specified bbox translation cycle change.",
            "callbacks": "Both candidates use the same SyntheticOwner(kernel_size=5), sigma0.65 for stationary supplied camera; these are demonstration callbacks.",
            "timing": "Synchronized wall clock around public ImageRenderer.render_img through ready independent NumPy. Includes route/cache/ROI/shared/mask preparation, uploads, GPU rendering and D2H. No runtime path is patched.",
            "exclusions": [
                "input I/O and fixture/reference construction",
                "bbox/frame_id input updates",
                "JIT/warmup",
                "every-call CPU validation",
                "memory/ownership checks",
                "encoding/network",
            ],
            "validation": "Every timed call compares all B8 pixels by np.array_equal against its frozen old-implementation pattern reference, and compares last_execution. Any nonzero image difference or policy change aborts. Validation is outside the timer but can affect inter-frame CPU load and device frequency.",
            "history": "Every worker warms at least nine stationary-camera frames; all patterns use the same mature camera state. These references are not valid for a moving-camera sequence.",
            "memory_scope": "CUDA counters are process-wide. One worker at a time; allocator reserves may survive previous phases. References are ordinary owned CPU arrays, compacted to HWC only after all baseline routes agree; every comparison still covers B8. Timed outputs are deleted before the next call. Ownership checks separately retain two outputs. Reserved memory alone does not establish a leak.",
            "cases": [],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(json_ready(report), stream, ensure_ascii=False, indent=2)
        try:
            for layout in args.layouts:
                offsets = layout_offsets(layout, fixture_boxes(template), frame.shape[:2])
                references, policies, reference_info = make_references(
                    baseline.api, template, offsets, frame, assets, args
                )
                case = {
                    "layout": layout,
                    "offsets_xy": offsets,
                    "status": "running",
                    "complete_pattern_cycles_per_phase": args.steps % len(offsets) == 0,
                    "references": reference_info,
                    "phases": [],
                    "aggregate": {},
                }
                report["cases"].append(case)
                save_progress(report, args.output)
                for round_index in range(args.rounds):
                    for name in rotated_order(CANDIDATES, round_index):
                        api = baseline.api if name == "baseline" else current_api
                        phase = run_phase(
                            api,
                            name,
                            round_index,
                            template,
                            offsets,
                            frame,
                            assets,
                            references,
                            policies,
                            args,
                        )
                        case["phases"].append(phase)
                        case["aggregate"] = aggregate(case["phases"])
                        baseline.verify()
                        if source_hashes(ROOT, include_harness=True) != source_before:
                            raise RuntimeError("current source files changed during measurement")
                        save_progress(report, args.output)
                        print(
                            json.dumps(
                                {
                                    "layout": layout,
                                    "candidate": name,
                                    "round": round_index + 1,
                                    "stats_ms": phase["stats_ms"],
                                    "exact": True,
                                    "policy_equal": True,
                                }
                            ),
                            flush=True,
                        )
                case["status"] = "complete"
                del references, policies
            report["status"] = "complete"
        except BaseException as exc:
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            save_progress(report, args.output)
        print(
            json.dumps(
                {
                    "report": str(args.output),
                    "results": {case["layout"]: case["aggregate"] for case in report["cases"]},
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
