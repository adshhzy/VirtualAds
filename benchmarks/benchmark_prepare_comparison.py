"""Balanced whole-call comparison of route preparation and mask uploads.

The isolated baseline is untouched. Current workers select each optimization by
instance-private diagnostic switches before warmup; no rendering method is
patched. Every timed B8/S16 NumPy result and auto decision is checked against
the old source with benchmark_upload_comparison's established timing loop.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks import benchmark_upload_comparison as base
from benchmarks.source_layout import source_file, source_key

CANDIDATES = ("baseline", "route_only", "mask_only", "both")
SWITCHES = {
    "route_only": (True, False),
    "mask_only": (False, True),
    "both": (True, True),
}


def configure_worker(worker, candidate):
    """Fail closed if a diagnostic switch is absent; never silently mislabel a run."""
    if candidate not in SWITCHES:
        raise ValueError(f"unknown current-source candidate: {candidate}")
    if not hasattr(worker, "_use_prepared_route_inputs") or not hasattr(
        worker.mask_builder, "_batch_upload_enabled"
    ):
        raise RuntimeError("current worker does not implement both preparation switches")
    route, mask = SWITCHES[candidate]
    worker._use_prepared_route_inputs = route
    worker.mask_builder._batch_upload_enabled = mask
    return worker


class CandidateAPI:
    """Construction-only proxy; public calls execute the actual selected runtime."""

    def __init__(self, api, candidate):
        if candidate not in SWITCHES:
            raise ValueError(f"unknown current-source candidate: {candidate}")
        self.api, self.candidate = api, candidate
        self.worker = None

    def __getattr__(self, name):
        return getattr(self.api, name)

    def ImageRenderer(self, *args, **kwargs):
        worker = self.api.ImageRenderer(*args, **kwargs)
        try:
            self.worker = configure_worker(worker, self.candidate)
            return self.worker
        except BaseException:
            worker.close()
            raise


def source_hashes(directory=base.ROOT):
    """Runtime plus all directly reused fixture, loader and measurement sources."""
    result = base.source_hashes(directory, include_harness=True)
    for name in (
        Path(__file__).name,
        "benchmark_shared_paste.py",
        "benchmark_paste_gpu_batch.py",
    ):
        path = source_file(directory, name)
        result["files"][source_key(path, directory)] = base.file_hash(path)
    result["files"] = dict(sorted(result["files"].items()))
    result["combined_sha256"] = hashlib.sha256(
        json.dumps(result["files"], sort_keys=True).encode("utf8")
    ).hexdigest()
    return result


def assert_uploads_released(state):
    """Check new upload storage as well as the legacy full-frame mask workspace."""
    upload = state["mask"].get("patch_upload")
    if upload is not None:
        for field in (
            "host_bytes",
            "device_bytes",
            "temporary_host_bytes",
            "temporary_device_bytes",
        ):
            if upload[field] != 0:
                raise AssertionError(f"close retained mask upload {field}: {upload[field]}")


def aggregate(phases, candidates):
    result = {}
    for name in candidates:
        matching = [phase for phase in phases if phase["candidate"] == name]
        samples = [value for phase in matching for value in phase["samples_ms"]]
        if not samples:
            continue
        result[name] = {
            "stats_ms": base.stats(samples),
            "round_means_ms": [phase["stats_ms"]["mean"] for phase in matching],
            "round_p95_ms": [phase["stats_ms"]["p95"] for phase in matching],
            "checked_calls": len(samples),
            "checked_route_images": base.BATCH * len(samples),
            "max_uint8_diff": 0,
            "all_execution_policies_equal": True,
        }
    if "baseline" in result:
        old = result["baseline"]["stats_ms"]
        for name in candidates:
            if name != "baseline" and name in result:
                new = result[name]["stats_ms"]
                result[name]["reduction_percent"] = {
                    key: 100 * (old[key] - new[key]) / old[key] for key in ("mean", "p95")
                }
    return result


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sample", "frame-bmp", "ad-bmp", "baseline", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument(
        "--layouts", nargs="+", choices=("static", "translated"), default=["static", "translated"]
    )
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="Allow an unbalanced candidate order; not confirmation evidence",
    )
    args = parser.parse_args(argv)
    if not __debug__:
        parser.error("run without -O so diagnostic assertions remain enabled")
    if args.rounds < 2 or args.steps <= 0 or args.warmup < 0:
        parser.error("rounds >= 2, positive steps and nonnegative warmup required")
    if len(set(args.layouts)) != len(args.layouts) or len(set(args.candidates)) != len(
        args.candidates
    ):
        parser.error("layouts and candidates must not contain duplicates")
    if "baseline" not in args.candidates or len(args.candidates) < 2:
        parser.error("include baseline and at least one current-source candidate")
    if args.rounds % len(args.candidates) and not args.exploratory:
        parser.error(
            "confirmation rounds must be a multiple of candidate count; or use --exploratory"
        )
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
    return args


def main(argv=None):
    args = parse_arguments(argv)
    torch.cuda.set_device(args.device)
    template = base.expanded_fixture(base.load_sample(args.sample))
    frame, ad = base.read_bmp(args.frame_bmp), base.read_bmp(args.ad_bmp)
    rgba = np.concatenate((ad, np.full((*ad.shape[:2], 1), 255, np.uint8)), axis=-1)
    assets = {
        (item["ad_placement_id"], item["ad_id"]): rgba
        for route in range(1, base.BATCH + 1)
        for item in template[f"ad_params_{route}"]
    }
    source_before = source_hashes()
    with base.load_sources(args.baseline) as baseline:
        report = {
            "status": "running",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "arguments": base.serializable_arguments(args),
            "provenance": base.environment_provenance(args),
            "cpu_threads": torch.get_num_threads(),
            "current_source": source_before,
            "baseline_source": baseline.provenance,
            "versions": {
                "baseline": baseline.api.RENDERER_VERSION,
                "current": base.current_api.RENDERER_VERSION,
            },
            "balanced_candidate_order": args.rounds % len(args.candidates) == 0,
            "candidate_switches": SWITCHES,
            "batch": base.BATCH,
            "slots_per_route": base.SLOTS,
            "frame_shape": frame.shape,
            "raw_ad_shape": ad.shape,
            "asset_lookup_pairs": len(assets),
            "distinct_source_arrays": 1,
            "fixture": "Original routes and ordered slots cycle to B8/S16; all assets use one opaque image. Only frame_id and specified bbox translation change. Camera, mask, matrices and material pixels are stationary.",
            "callbacks": "SyntheticOwner(kernel_size=5), stationary sigma0.65; demonstration callbacks, not production motion processing.",
            "timing": "Unmodified shared benchmark run_phase: synchronized wall clock around public render_img through ready independent NumPy. Includes route/asset/ROI/shared/mask preparation, uploads, GPU computation and D2H. Current worker switches set only during construction before warmup.",
            "exclusions": [
                "input I/O and reference construction",
                "input bbox/frame_id updates",
                "JIT/warmup",
                "every-call CPU validation",
                "memory/ownership checks",
                "encoding/network",
            ],
            "validation": "All B8 pixels of every timed output are checked by np.array_equal against isolated old-source mature-history references; auto decisions must match. Checks outside timers still affect cadence; inverse latency is not a live-stream throughput measurement.",
            "memory_scope": "One active worker at a time. CUDA counters are process-wide; allocator reserved memory can survive phases. Reference CPU arrays are independent. Ownership checks retain outputs separately. New upload working storage is recorded independently from full-frame mask storage.",
            "cases": [],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf8") as stream:
            json.dump(base.json_ready(report), stream, ensure_ascii=False, indent=2)
        try:
            for layout in args.layouts:
                offsets = base.layout_offsets(layout, base.fixture_boxes(template), frame.shape[:2])
                references, policies, reference_info = base.make_references(
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
                base.save_progress(report, args.output)
                for round_index in range(args.rounds):
                    for name in base.rotated_order(args.candidates, round_index):
                        api = (
                            baseline.api
                            if name == "baseline"
                            else CandidateAPI(base.current_api, name)
                        )
                        phase = base.run_phase(
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
                        assert_uploads_released(phase["memory_after_close"])
                        if name != "baseline":
                            route_state = api.worker._route_preparation.cache_info()
                            if route_state["retained_bytes"] != 0:
                                raise AssertionError("close retained route preparation storage")
                            phase["route_workspace_after_close"] = route_state
                            route_enabled, mask_enabled = SWITCHES[name]
                            mask_state = phase["memory_after_timing"]["mask"]["patch_upload"]
                            if bool(route_state["prepared_plans"]) != route_enabled:
                                raise AssertionError("route optimization switch was not exercised")
                            if bool(mask_state["uploads"]) != mask_enabled:
                                raise AssertionError("mask optimization switch was not exercised")
                        phase["candidate_switches"] = None if name == "baseline" else SWITCHES[name]
                        case["phases"].append(phase)
                        case["aggregate"] = aggregate(case["phases"], args.candidates)
                        baseline.verify()
                        if source_hashes() != source_before:
                            raise RuntimeError(
                                "current runtime or harness source changed during measurement"
                            )
                        base.save_progress(report, args.output)
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
            base.save_progress(report, args.output)
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
