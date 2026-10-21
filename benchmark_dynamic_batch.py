"""Dynamic-B, synchronous NumPy benchmark. Synthetic callbacks, not production.

Use --source-dir for a preserved source snapshot. All timings include shared
preparation and an independently owned, ready-to-read NumPy result. Candidate
order rotates each round; source inputs are identical for every candidate.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import replace
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from benchmark_support import (
    latency_statistics,
    measure_render_path,
    rotated_order,
    save_report,
    serializable_arguments,
)

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


def parse_arguments():
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
    if (
        min(*opt.batches, opt.height, opt.width, opt.slots, opt.warmup, opt.iterations, opt.rounds)
        < 1
    ):
        parser.error("sizes/counts must be positive")
    if any(c < 0 for c in opt.chunks):
        parser.error("chunks must be nonnegative")
    return opt


def parse_launch_config(config):
    """Resolve a documented preset or six explicit horizontal/vertical values."""
    if config == "default":
        return None
    if config in CONFIGS:
        return CONFIGS[config]
    parts = tuple(map(int, config.split(",")))
    if len(parts) != 6:
        raise ValueError("custom config must have six integers")
    return parts


def fixture_options(opt, batch):
    return SimpleNamespace(
        device=opt.device,
        batch=batch,
        height=opt.height,
        width=opt.width,
        slots=opt.slots,
        ad_sizes=[(200, 200), (500, 500), (300, 600)],
        kernel_size=opt.kernel_size,
        warmup=opt.warmup,
        iterations=opt.iterations,
        acceleration="triton",
        dense_blur_backend="triton",
        normalization="legacy",
        camera_index_policy="legacy",
        forward=False,
        inverse_iterations=10,
        align_corners=False,
        sigma_pattern="continuous",
    )


def check_candidates(path, fixture, candidates, select, reference_config):
    """Every candidate must match one unchunked reference, with zero tolerance."""
    import numpy as np

    select(0, reference_config)
    fixture.set_frame(7)
    reference = path.render()
    expected = reference.copy()
    previous_output = previous_snapshot = None
    for chunk, config in candidates:
        select(chunk, config)
        # Two calls also detect reuse when the sweep has only one candidate.
        for _ in range(2):
            fixture.set_frame(7)
            actual = path.render()
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(reference, expected)
            assert not np.shares_memory(actual, reference), "candidate reused the reference output"
            if previous_output is not None:
                np.testing.assert_array_equal(previous_output, previous_snapshot)
                assert not np.shares_memory(actual, previous_output), "candidate output reused"
            assert actual.dtype == np.uint8 and actual.flags.c_contiguous
            previous_output, previous_snapshot = actual, actual.copy()


def validate_source_directory(source):
    """Reject incomplete snapshots instead of falling back to current modules."""
    required = [
        "benchmark_paste_gpu_batch",
        "benchmark_shared_paste",
        "paste_gpu_batch_adapter",
        "paste_gpu_batch",
        "paste_gpu_kernels",
        "paste_gaussian_kernels",
        "paste_shared_params",
        "paste_shared_kernels",
        "paste_shared_types",
        "paste_camera_filter",
    ]
    # Follow version-specific local imports as well: perf4.2 requires the
    # execution policy, whereas the preserved perf4.1 snapshot predates it.
    for name in required:
        module_path = source / (name + ".py")
        if not module_path.is_file():
            continue
        tree = ast.parse(module_path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                dependencies = [node.module] if node.module else [item.name for item in node.names]
            elif isinstance(node, ast.Import):
                dependencies = [item.name for item in node.names]
            else:
                continue
            for dependency in dependencies:
                if dependency.startswith("paste_") and dependency not in required:
                    required.append(dependency)
    missing = [name + ".py" for name in required if not (source / (name + ".py")).is_file()]
    if missing:
        raise ValueError(f"source snapshot is incomplete at {source}: {', '.join(missing)}")
    return required


def verify_loaded_sources(source, module_names):
    """Already-imported modules must also come from the requested snapshot."""
    for name in module_names:
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve().parent != source:
            raise RuntimeError(f"{name} loaded from {module.__file__}, expected {source}")


def main():
    opt = parse_arguments()
    source = (opt.source_dir or Path(__file__).parent).resolve()
    source_modules = validate_source_directory(source)
    sys.path.insert(0, str(source))
    import torch
    import triton
    from benchmark_paste_gpu_batch import make_case
    from benchmark_shared_paste import SyntheticFixture, RenderPath
    import paste_gaussian_kernels as gaussian

    verify_loaded_sources(source, source_modules)

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
        "source_dir": str(source),
        "python": sys.version,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "arguments": serializable_arguments(opt),
        "results": [],
    }
    with torch.inference_mode():
        for batch in opt.batches:
            torch.manual_seed(24680)
            cli = fixture_options(opt, batch)
            args, boxes = make_case(
                device,
                batch=batch,
                height=cli.height,
                width=cli.width,
                slots=cli.slots,
                kernel_size=cli.kernel_size,
                realistic=True,
                source_sizes=cli.ad_sizes,
            )
            fixture = SyntheticFixture(cli, device, args, boxes)
            path = RenderPath(
                "candidate", fixture, optimized=True, separable=opt.blur_backend == "separable"
            )
            original_options = path.adapter.options
            candidates = [
                (chunk, config)
                for chunk in opt.chunks
                if chunk == 0 or chunk < batch
                for config in opt.configs
            ]
            if not candidates:
                candidates = [(0, opt.configs[0])]

            def select(chunk, config, target=path, initial_options=original_options):
                nonlocal selected_config
                selected_config = parse_launch_config(config)
                if hasattr(initial_options, "batch_chunk_size"):
                    target.adapter.options = replace(initial_options, batch_chunk_size=chunk)
                elif chunk:
                    raise ValueError("source has no batch_chunk_size support")

            check_candidates(
                path, fixture, candidates, select, "default" if opt.source_dir else "legacy"
            )
            row = {"batch": batch, "bitexact_candidates": True, "rounds": []}
            report["results"].append(row)
            for round_index in range(opt.rounds):
                order = rotated_order(candidates, round_index)
                measurements = []
                for chunk, config in order:
                    select(chunk, config)
                    values = measure_render_path(
                        path,
                        warmup=opt.warmup,
                        iterations=opt.iterations,
                        device=device,
                        reset_peak_memory=True,
                    )
                    measurement = {
                        "chunk": chunk,
                        "config": config,
                        "wall_ms": latency_statistics(values, include_minimum=True),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    }
                    measurements.append(measurement)
                    print(
                        json.dumps({"batch": batch, "round": round_index, **measurement}),
                        flush=True,
                    )
                row["rounds"].append(measurements)
                save_report(report, opt.output)
            verify_loaded_sources(source, source_modules)
            del path, fixture, args, boxes, select
            gc.collect()
            torch.cuda.empty_cache()
    save_report(report, opt.output)


if __name__ == "__main__":
    main()
