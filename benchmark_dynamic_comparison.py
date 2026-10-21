"""Compare a preserved source snapshot against the current implementation.

Both versions run in one process with identical seeded inputs, separate mutable
camera state, and rotating measurement order. Includes synchronous NumPy output.
The baseline adapter and shared builder are imported from an isolated package.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import gc
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from benchmark_support import (
    latency_statistics,
    measure_render_path,
    rotated_order,
    save_report,
    serializable_arguments,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("results/baseline_perf4_1"))
    parser.add_argument("--baseline-label", default="perf4.1")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 3, 4, 5, 8, 12, 16])
    parser.add_argument(
        "--chunks", nargs="+", default=["auto"], help="auto or nonnegative integers"
    )
    parser.add_argument(
        "--configs", nargs="+", default=["default"], help="default or six comma-separated integers"
    )
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--ad-sizes", nargs="+", default=["200x200", "500x500", "300x600"])
    parser.add_argument(
        "--rounds", type=int, default=4, help="even rounds balance before/after order"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if (
        min(
            *options.batches,
            options.height,
            options.width,
            options.kernel_size,
            options.rounds,
            options.warmup,
            options.iterations,
        )
        < 1
    ):
        parser.error("counts/dimensions must be positive")
    candidates = {}
    for chunk in options.chunks:
        chunk_size = "auto" if chunk == "auto" else int(chunk)
        if chunk_size != "auto" and chunk_size < 0:
            parser.error("chunk size must be auto or nonnegative")
        for config in options.configs:
            launch = None if config == "default" else tuple(map(int, config.split(",")))
            candidates[f"chunk={chunk};blur={config}"] = (chunk_size, launch)
    if options.baseline_label in candidates:
        parser.error("baseline-label must differ from candidate names")
    return options, candidates


def load_baseline_modules(source_directory):
    """Keep baseline relative imports separate from the current source modules."""
    package = ModuleType("_paste_baseline")
    package.__path__ = [str(source_directory)]
    sys.modules[package.__name__] = package
    return (
        importlib.import_module("_paste_baseline.paste_gpu_batch_adapter"),
        importlib.import_module("_paste_baseline.paste_shared_params"),
    )


class ComparisonCase:
    """One batch's immutable GPU pixels and two independent rendering states."""

    def __init__(self, options, batch, device, source_sizes, baseline_modules, candidates):
        import torch
        from benchmark_paste_gpu_batch import make_case
        from benchmark_shared_paste import RenderPath, SyntheticFixture

        self.options, self.device, self.candidates = options, device, candidates
        self.baseline_name = options.baseline_label
        torch.manual_seed(24680)
        image_args, boxes = make_case(
            device,
            batch=batch,
            height=options.height,
            width=options.width,
            slots=16,
            realistic=True,
            kernel_size=options.kernel_size,
            source_sizes=source_sizes,
        )
        fixture_options = SimpleNamespace(
            batch=batch,
            height=options.height,
            width=options.width,
            slots=16,
            kernel_size=options.kernel_size,
            ad_sizes=source_sizes,
            sigma_pattern="continuous",
            acceleration="triton",
            normalization="legacy",
            camera_index_policy="legacy",
            forward=False,
            inverse_iterations=10,
            align_corners=False,
            dense_blur_backend="triton",
        )
        self.original = RenderPath(
            self.baseline_name,
            SyntheticFixture(fixture_options, device, image_args, boxes),
            optimized=True,
            separable=True,
        )
        baseline_adapter, baseline_shared = baseline_modules
        self.original.builder = baseline_shared.SharedParameterBuilder(
            acceleration="triton",
            normalization="legacy",
            camera_index_policy="legacy",
            enable_separable=True,
        )
        # Copy only fields supported by the preserved version. Its warp callback
        # must be bound to its own builder, never to the current implementation.
        baseline_options = {
            field.name: (
                self.original.builder.build_warp_grids
                if field.name == "warp_grids_builder"
                else getattr(self.original.adapter.options, field.name)
            )
            for field in fields(baseline_adapter.AdapterOptions)
        }
        self.original.adapter = baseline_adapter.PasteBatchAdapter(
            baseline_adapter.AdapterOptions(**baseline_options)
        )
        self.current = RenderPath(
            "current",
            SyntheticFixture(fixture_options, device, image_args, boxes),
            optimized=True,
            separable=True,
        )
        self.current_options = self.current.adapter.options

    def select(self, name):
        if name == self.baseline_name:
            return self.original
        chunk_size, launch_config = self.candidates[name]
        self.current.adapter.options = replace(
            self.current_options,
            batch_chunk_size=chunk_size,
            blur_launch_config=launch_config,
        )
        return self.current

    def check_outputs(self):
        """Compare directly to old modules and verify retained NumPy ownership."""
        import numpy as np

        previous_output = previous_snapshot = None
        for step in (0, 7):
            self.original.fixture.set_frame(step)
            expected = self.original.render()
            snapshot = expected.copy()
            for name in self.candidates:
                path = self.select(name)
                path.fixture.set_frame(step)
                actual = path.render()
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(expected, snapshot)
                if previous_output is not None:
                    np.testing.assert_array_equal(previous_output, previous_snapshot)
                    assert not np.shares_memory(actual, previous_output), "current output reused"
                assert actual.flags.c_contiguous and actual.dtype == np.uint8
                previous_output, previous_snapshot = actual, actual.copy()

    def measure_round(self, round_index):
        measurements = []
        names = [self.baseline_name, *self.candidates]
        for name in rotated_order(names, round_index):
            path = self.select(name)
            samples = measure_render_path(
                path,
                warmup=self.options.warmup,
                iterations=self.options.iterations,
                device=self.device,
            )
            measurements.append(
                {
                    "name": name,
                    "wall_ms": latency_statistics(samples),
                    "samples_ms": samples,
                    "execution": getattr(path.adapter, "last_execution", None),
                }
            )
        return measurements


def run_batch(options, batch, device, source_sizes, baseline_modules, candidates, report):
    case = ComparisonCase(options, batch, device, source_sizes, baseline_modules, candidates)
    case.check_outputs()
    row = {"batch": batch, "exact_snapshot_match": True, "rounds": []}
    report["results"].append(row)
    for round_index in range(options.rounds):
        measurements = case.measure_round(round_index)
        row["rounds"].append(measurements)
        for item in measurements:
            printable = {key: value for key, value in item.items() if key != "samples_ms"}
            print(json.dumps({"batch": batch, "round": round_index, **printable}), flush=True)
        save_report(report, options.output)
    row["summary"] = {
        name: latency_statistics(
            [
                value
                for measurements in row["rounds"]
                for item in measurements
                if item["name"] == name
                for value in item["samples_ms"]
            ]
        )
        for name in [options.baseline_label, *candidates]
    }
    save_report(report, options.output)


def main():
    options, candidates = parse_arguments()
    import torch
    import triton
    from benchmark_paste_gpu_batch import parse_ad_size

    baseline = options.baseline.resolve()
    baseline_modules = load_baseline_modules(baseline)
    device = torch.device(options.device)
    torch.cuda.set_device(device)
    source_sizes = [parse_ad_size(value) for value in options.ad_sizes]
    report = {
        "scope": "shared preparation through ready independent NumPy; exact original snapshot comparison",
        "baseline": str(baseline),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "python": sys.version,
        "arguments": serializable_arguments(options),
        "results": [],
    }
    with torch.inference_mode():
        for batch in options.batches:
            run_batch(options, batch, device, source_sizes, baseline_modules, candidates, report)
            gc.collect()
            torch.cuda.empty_cache()
    save_report(report, options.output)


if __name__ == "__main__":
    main()
