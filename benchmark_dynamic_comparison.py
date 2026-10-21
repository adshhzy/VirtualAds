"""Compare a preserved perf4.1 source snapshot against dynamic execution.

Both versions run in one process with identical seeded inputs, separate mutable
camera state, and alternating measurement order. Includes synchronous NumPy.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import gc
import importlib
import json
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, default=Path("results/baseline_perf4_1"))
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 3, 4, 5, 8, 12, 16])
    p.add_argument("--chunks", nargs="+", default=["auto"], help="auto or nonnegative integers")
    p.add_argument("--configs", nargs="+", default=["default"], help="default or six comma-separated integers")
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--kernel-size", type=int, default=9)
    p.add_argument("--ad-sizes", nargs="+", default=["200x200", "500x500", "300x600"])
    p.add_argument("--rounds", type=int, default=4, help="even rounds balance before/after order")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, required=True)
    cli = p.parse_args()
    if min(*cli.batches, cli.height, cli.width, cli.kernel_size, cli.rounds, cli.warmup, cli.iterations) < 1:
        p.error("counts/dimensions must be positive")
    import numpy as np
    import torch
    import triton
    from benchmark_paste_gpu_batch import make_case, parse_ad_size
    from benchmark_shared_paste import SyntheticFixture, RenderPath
    baseline = cli.baseline.resolve()
    package = ModuleType("_paste_baseline")
    package.__path__ = [str(baseline)]
    sys.modules[package.__name__] = package
    old_adapter = importlib.import_module("_paste_baseline.paste_gpu_batch_adapter")
    old_shared = importlib.import_module("_paste_baseline.paste_shared_params")
    device = torch.device(cli.device)
    torch.cuda.set_device(device)
    source_sizes = [parse_ad_size(value) for value in cli.ad_sizes]
    candidates = []
    for chunk in cli.chunks:
        chunk_value = "auto" if chunk == "auto" else int(chunk)
        if chunk_value != "auto" and chunk_value < 0:
            p.error("chunk size must be auto or nonnegative")
        for config in cli.configs:
            launch = None if config == "default" else tuple(map(int, config.split(",")))
            candidates.append((f"chunk={chunk};blur={config}", chunk_value, launch))
    report = dict(scope="shared preparation through ready independent NumPy; exact original snapshot comparison",
        baseline=str(baseline), gpu=torch.cuda.get_device_name(device), torch=torch.__version__,
        cuda=torch.version.cuda, triton=triton.__version__, python=sys.version,
        arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(cli).items()}, results=[])
    cli.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        cli.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def stats(values):
        return {"mean": float(np.mean(values)), "p50": float(np.median(values)),
                "p95": float(np.percentile(values, 95))}

    with torch.inference_mode():
        for batch in cli.batches:
            torch.manual_seed(24680)
            args, boxes = make_case(device, batch=batch, height=cli.height, width=cli.width,
                slots=16, realistic=True, kernel_size=cli.kernel_size, source_sizes=source_sizes)
            fixture_cli = SimpleNamespace(batch=batch, height=cli.height, width=cli.width, slots=16,
                kernel_size=cli.kernel_size, ad_sizes=source_sizes, sigma_pattern="continuous",
                acceleration="triton", normalization="legacy", camera_index_policy="legacy",
                forward=False, inverse_iterations=10, align_corners=False, dense_blur_backend="triton")
            original = RenderPath("perf4.1", SyntheticFixture(fixture_cli, device, args, boxes),
                                  optimized=True, separable=True)
            original.builder = old_shared.SharedParameterBuilder(acceleration="triton",
                normalization="legacy", camera_index_policy="legacy", enable_separable=True)
            original.adapter = old_adapter.PasteBatchAdapter(old_adapter.AdapterOptions(**{
                field.name: (original.builder.build_warp_grids if field.name == "warp_grids_builder"
                             else getattr(original.adapter.options, field.name))
                for field in fields(old_adapter.AdapterOptions)}))
            current = RenderPath("current", SyntheticFixture(fixture_cli, device, args, boxes),
                                 optimized=True, separable=True)
            current_options = current.adapter.options
            candidate_map = {name: (chunk, launch) for name,chunk,launch in candidates}

            def select(name):
                if name == "perf4.1":
                    return original
                chunk, launch = candidate_map[name]
                current.adapter.options = replace(current_options, batch_chunk_size=chunk,
                                                   blur_launch_config=launch)
                return current

            # Direct comparison to the pre-change modules, not successive tolerances.
            for step in (0, 7):
                original.fixture.set_frame(step)
                expected = original.render()
                snapshot = expected.copy()
                for name, _, _ in candidates:
                    path = select(name)
                    path.fixture.set_frame(step)
                    actual = path.render()
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(expected, snapshot)
                    assert actual.flags.c_contiguous and actual.dtype == np.uint8
                del actual, expected, snapshot
            row = dict(batch=batch, exact_snapshot_match=True, rounds=[])
            report["results"].append(row)
            names = ["perf4.1"] + [name for name, _, _ in candidates]
            for round_index in range(cli.rounds):
                order = names[round_index % len(names):] + names[:round_index % len(names)]
                measurements = []
                for name in order:
                    path = select(name)
                    for step in range(cli.warmup):
                        path.fixture.set_frame(step)
                        output = path.render()
                    torch.cuda.synchronize(device)
                    values = []
                    for step in range(cli.iterations):
                        path.fixture.set_frame(cli.warmup + step)
                        torch.cuda.synchronize(device)
                        start = time.perf_counter()
                        output = path.render()
                        torch.cuda.synchronize(device)
                        values.append((time.perf_counter() - start) * 1000)
                    item = dict(name=name, wall_ms=stats(values), samples_ms=values,
                        execution=getattr(path.adapter, "last_execution", None))
                    measurements.append(item)
                    print(json.dumps(dict(batch=batch, round=round_index,
                        **{k:v for k,v in item.items() if k != "samples_ms"})), flush=True)
                row["rounds"].append(measurements)
                save()
            row["summary"] = {name: stats([value for rnd in row["rounds"] for item in rnd
                if item["name"] == name for value in item["samples_ms"]]) for name in names}
            save()
            del original, current, path, args, boxes, output
            gc.collect()
            torch.cuda.empty_cache()
    save()


if __name__ == "__main__":
    main()
