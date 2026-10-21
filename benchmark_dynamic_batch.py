"""Dynamic-B, synchronous NumPy benchmark. Synthetic callbacks, not production.

Use --source-dir for a preserved source snapshot. All timings include shared
preparation and an independently owned, ready-to-read NumPy result. Candidate
order rotates each round; source inputs are identical for every candidate.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


CONFIGS = {
    "legacy": (128, 4, 4, 128, 4, 4),
    "64x4": (64, 4, 4, 64, 4, 4),
    "64x8": (64, 8, 4, 64, 8, 4),
    "128x8": (128, 8, 4, 128, 8, 4),
    "128x2": (128, 2, 4, 128, 2, 4),
    "256x2": (256, 2, 4, 256, 2, 4),
    "256x4": (256, 4, 4, 256, 4, 4),
    "128x8w8": (128, 8, 8, 128, 8, 8),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 3, 4, 5, 8, 12, 16])
    parser.add_argument("--chunks", type=int, nargs="+", default=[0])
    parser.add_argument("--configs", nargs="+", default=["default"])
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--blur-backend", choices=["separable", "triton"], default="separable")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    opt = parser.parse_args()
    if min(*opt.batches, opt.height, opt.width, opt.slots, opt.warmup, opt.iterations, opt.rounds) < 1:
        parser.error("sizes/counts must be positive")
    if any(c < 0 for c in opt.chunks):
        parser.error("chunks must be nonnegative")
    source = (opt.source_dir or Path(__file__).parent).resolve()
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import triton
    from benchmark_paste_gpu_batch import make_case
    from benchmark_shared_paste import SyntheticFixture, RenderPath
    import paste_gaussian_kernels as gaussian

    device = torch.device(opt.device)
    torch.cuda.set_device(device)
    raw_blur = gaussian.separable_blur_blend_to_uint8
    selected_config = None

    def configured_blur(*args, **kwargs):
        if selected_config is not None:
            kwargs["launch_config"] = selected_config
        return raw_blur(*args, **kwargs)

    gaussian.separable_blur_blend_to_uint8 = configured_blur
    report = {
        "scope": "shared preparation through ready independent NumPy; synthetic callbacks",
        "source_dir": str(source), "python": sys.version, "torch": torch.__version__,
        "triton": triton.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(opt).items()},
        "results": [],
    }
    opt.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        opt.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def summarize(values):
        return {"mean": float(np.mean(values)), "p50": float(np.median(values)),
                "p95": float(np.percentile(values, 95)), "min": float(np.min(values))}

    with torch.inference_mode():
        for batch in opt.batches:
            torch.manual_seed(24680)
            cli = SimpleNamespace(device=opt.device, batch=batch, height=opt.height, width=opt.width,
                slots=opt.slots, ad_sizes=[(200, 200), (500, 500), (300, 600)],
                kernel_size=opt.kernel_size, warmup=opt.warmup, iterations=opt.iterations,
                acceleration="triton", dense_blur_backend="triton", normalization="legacy",
                camera_index_policy="legacy", forward=False, inverse_iterations=10,
                align_corners=False, sigma_pattern="continuous")
            args, boxes = make_case(device, batch=batch, height=cli.height, width=cli.width,
                slots=cli.slots, kernel_size=cli.kernel_size, realistic=True, source_sizes=cli.ad_sizes)
            fixture = SyntheticFixture(cli, device, args, boxes)
            path = RenderPath("candidate", fixture, optimized=True, separable=opt.blur_backend == "separable")
            original_options = path.adapter.options
            candidates = [(chunk, config) for chunk in opt.chunks if chunk == 0 or chunk < batch
                          for config in opt.configs]
            if not candidates:
                candidates = [(0, opt.configs[0])]

            def select(chunk, config):
                nonlocal selected_config
                selected_config = None if config == "default" else CONFIGS.get(config)
                if config not in ("default", *CONFIGS):
                    parts = tuple(map(int, config.split(",")))
                    if len(parts) != 6:
                        raise ValueError("custom config must have six integers")
                    selected_config = parts
                if hasattr(original_options, "batch_chunk_size"):
                    path.adapter.options = replace(original_options, batch_chunk_size=chunk)
                elif chunk:
                    raise ValueError("source has no batch_chunk_size support")

            # Every candidate compares directly to the same unchunked reference.
            select(0, "default" if opt.source_dir else "legacy")
            fixture.set_frame(7)
            reference = path.render()
            for chunk, config in candidates:
                select(chunk, config)
                fixture.set_frame(7)
                actual = path.render()
                np.testing.assert_array_equal(actual, reference)
                assert actual.dtype == np.uint8 and actual.flags.c_contiguous
                del actual
            del reference
            row = {"batch": batch, "bitexact_candidates": True, "rounds": []}
            report["results"].append(row)
            for round_index in range(opt.rounds):
                order = candidates[round_index % len(candidates):] + candidates[:round_index % len(candidates)]
                measurements = []
                for chunk, config in order:
                    select(chunk, config)
                    for step in range(opt.warmup):
                        fixture.set_frame(step)
                        output = path.render()
                    torch.cuda.synchronize(device)
                    values = []
                    torch.cuda.reset_peak_memory_stats(device)
                    for step in range(opt.iterations):
                        fixture.set_frame(opt.warmup + step)
                        torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        output = path.render()
                        torch.cuda.synchronize(device)
                        values.append((time.perf_counter() - started) * 1000)
                    measurement = {"chunk": chunk, "config": config, "wall_ms": summarize(values),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
                    measurements.append(measurement)
                    print(json.dumps({"batch": batch, "round": round_index, **measurement}), flush=True)
                row["rounds"].append(measurements)
                save()
            del path, fixture, args, boxes, output
            gc.collect()
            torch.cuda.empty_cache()
    save()


if __name__ == "__main__":
    main()
