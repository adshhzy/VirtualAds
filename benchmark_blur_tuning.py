"""Offline launch tuning for the two existing separable blur kernels.

This measures CUDA event spans for device kernels only: no shared preparation,
adapter, allocations, validation, or NumPy export is inside the timed interval.
It does not select or change the library's default launch configuration.

Example:
    python benchmark_blur_tuning.py --batches 1 2 3 4 5 8 12 16 --output results/blur_tuning.json
"""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import random
import statistics
import sys


BASELINE = (128, 4, 4)


def make_candidates(block_x, block_y, warps):
    # Avoid assigning more threads than output pixels; always keep the old
    # configuration even when the requested search space excludes it.
    candidates = [BASELINE]
    candidates.extend((x, y, w) for x, y, w in product(block_x, block_y, warps)
                      if x * y >= 32 * w)
    return list(dict.fromkeys(candidates))


def config_dict(config):
    return dict(zip(("block_x", "block_y", "num_warps"), config))


def make_fixture(torch, np, device, batch, height, width, seed):
    from paste_shared_params import _factor_kernel

    torch.manual_seed(seed + batch)
    # Premultiplied RGB in [0,255], alpha in [0,1], contiguous RGBA planes.
    # The shared HWC-backed frame and shared single-channel mask reproduce
    # the broadcast/noncontiguous views used by the regular shared benchmark.
    canvas = torch.rand((batch, 4, height, width), device=device, dtype=torch.float32)
    canvas[:, :3].mul_(255.0).mul_(canvas[:, 3:4])
    frames = (torch.rand((1, height, width, 3), device=device) * 255.0).permute(0, 3, 1, 2)
    frames = frames.expand(batch, 3, height, width)
    mask = (torch.rand((1, 1, height, width), device=device) * 0.35).expand(batch, 3, height, width)
    positions = np.arange(9, dtype=np.float32) - 4
    vector = np.exp(-(positions * positions) / (2.0 * 1.5 * 1.5)).astype(np.float32)
    vector /= vector.sum()
    dense = np.outer(vector, vector).astype(np.float32)
    factors, factor_error, reason = _factor_kernel(dense)
    if factors is None:
        raise AssertionError(f"9x9 fixture failed separability validation: {reason}")
    ky, kx = (torch.from_numpy(value).to(device=device) for value in factors)
    temp = torch.empty_like(canvas)
    reference_temp = torch.empty_like(canvas)
    output = torch.empty((batch, height, width, 3), device=device, dtype=torch.uint8)
    reference_output = torch.empty_like(output)
    return dict(canvas=canvas, frames=frames, mask=mask, ky=ky, kx=kx,
                temp=temp, reference_temp=reference_temp, output=output,
                reference_output=reference_output, factor_error=factor_error)


def horizontal_launcher(kernels, fixture, config, destination):
    canvas, kx = fixture["canvas"], fixture["kx"]
    batch, _, height, width = canvas.shape
    block_x, block_y, warps = config
    grid = ((width + block_x - 1) // block_x, (height + block_y - 1) // block_y, batch)
    kwargs = dict(H=height, W=width, C_B=canvas.stride(0), C_C=canvas.stride(1),
                  C_Y=canvas.stride(2), C_X=canvas.stride(3), K_S=kx.stride(0),
                  KW=kx.numel(), BLOCK_X=block_x, BLOCK_Y=block_y,
                  num_warps=warps, enable_fp_fusion=False)
    return lambda: kernels._horizontal_rgba[grid](canvas, kx, destination, **kwargs)


def vertical_launcher(kernels, fixture, config, destination):
    # Every vertical candidate reads the SAME immutable baseline horizontal
    # output, so the two tuning experiments remain independent.
    temp, frames, ky, mask = (fixture[key] for key in ("reference_temp", "frames", "ky", "mask"))
    batch, _, height, width = temp.shape
    block_x, block_y, warps = config
    grid = ((width + block_x - 1) // block_x, (height + block_y - 1) // block_y, batch)
    kwargs = dict(H=height, W=width, F_B=frames.stride(0), F_C=frames.stride(1),
                  F_Y=frames.stride(2), F_X=frames.stride(3), M_B=mask.stride(0),
                  M_C=mask.stride(1), M_Y=mask.stride(2), M_X=mask.stride(3),
                  K_S=ky.stride(0), KH=ky.numel(), LEGACY=True,
                  BLOCK_X=block_x, BLOCK_Y=block_y, num_warps=warps,
                  enable_fp_fusion=False)
    return lambda: kernels._vertical_blend_u8[grid](temp, frames, ky, mask, destination, **kwargs)


def measure_event_span(torch, device, invoke, warmup, iterations):
    for _ in range(warmup):
        invoke()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        invoke()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def tune_stage(torch, kernels, fixture, stage, candidates, cli, device, rng):
    horizontal = stage == "horizontal"
    destination = fixture["temp" if horizontal else "output"]
    reference = fixture["reference_temp" if horizontal else "reference_output"]
    factory = horizontal_launcher if horizontal else vertical_launcher
    records, launchers = {}, {}
    # Compile and validate every candidate before any timed round. Resource
    # exhaustion can reject one configuration; other failures abort the run.
    compile_order = list(candidates)
    rng.shuffle(compile_order)
    for config in compile_order:
        record = {"config": config_dict(config), "status": "pending", "event_span_ms_per_launch": []}
        records[config] = record
        invoke = factory(kernels, fixture, config, destination)
        try:
            invoke()
            torch.cuda.synchronize(device)
        except Exception as exc:
            if type(exc).__name__ != "OutOfResources":
                raise
            record.update(status="resource_error", error=str(exc))
            continue
        actual_bits = destination.view(torch.int32) if horizontal else destination
        expected_bits = reference.view(torch.int32) if horizontal else reference
        if not torch.equal(actual_bits, expected_bits):
            record.update(status="rejected_difference",
                          mismatched_elements=int(torch.count_nonzero(actual_bits != expected_bits).item()),
                          max_absolute_difference=float((destination.float() - reference.float()).abs().max().item()))
            continue
        record.update(status="passed", comparison="float32_bitexact" if horizontal else "uint8_exact")
        launchers[config] = invoke

    orders = []
    for round_index in range(cli.rounds):
        order = list(launchers)
        rng.shuffle(order)
        orders.append([config_dict(config) for config in order])
        for config in order:
            elapsed = measure_event_span(torch, device, launchers[config], cli.warmup, cli.iterations)
            records[config]["event_span_ms_per_launch"].append(elapsed)
        print(f"B={fixture['canvas'].shape[0]} {stage}: timed round {round_index + 1}/{cli.rounds}", flush=True)

    valid = []
    for record in records.values():
        samples = record["event_span_ms_per_launch"]
        if samples:
            record["timing_ms"] = dict(mean=statistics.mean(samples), median=statistics.median(samples),
                                       minimum=min(samples), maximum=max(samples))
            valid.append(record)
    valid.sort(key=lambda record: record["timing_ms"]["median"])
    if not valid or records[BASELINE]["status"] != "passed":
        raise RuntimeError(f"{stage}: baseline or all candidates failed validation")
    baseline_ms = records[BASELINE]["timing_ms"]["median"]
    for record in valid:
        record["median_saving_vs_baseline_ms"] = baseline_ms - record["timing_ms"]["median"]
    return {"baseline": records[BASELINE], "best_three": valid[:3],
            "all_candidates": [records[config] for config in candidates],
            "compile_order": [config_dict(config) for config in compile_order],
            "round_orders": orders}


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 3, 4, 5, 8, 12, 16])
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--iterations", type=int, default=50, help="launches enclosed by each CUDA-event pair")
    parser.add_argument("--warmup", type=int, default=5, help="untimed launches before each candidate measurement")
    parser.add_argument("--rounds", type=int, default=2, help="independently shuffled timing rounds; at least 2")
    parser.add_argument("--block-x", type=int, choices=(64, 128, 256), nargs="+", default=[64, 128, 256])
    parser.add_argument("--block-y", type=int, choices=(2, 4, 8), nargs="+", default=[2, 4, 8])
    parser.add_argument("--warps", type=int, choices=(2, 4, 8), nargs="+", default=[2, 4, 8])
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--output", type=Path, default=Path("results/blur_tuning.json"))
    cli = parser.parse_args()
    if min(cli.batches + [cli.height, cli.width, cli.iterations, cli.warmup]) < 1 or cli.rounds < 2:
        parser.error("batches/height/width/iterations/warmup must be positive; rounds must be >=2")
    if cli.seed < 0:
        parser.error("seed must be nonnegative")
    import numpy as np
    import torch
    import paste_gaussian_kernels as kernels

    device = torch.device(cli.device)
    if device.type != "cuda" or not torch.cuda.is_available() or not kernels.available():
        parser.exit(2, "A working CUDA GPU and Triton are required; no performance results were produced.\n")
    torch.cuda.set_device(device)
    candidates = make_candidates(cli.block_x, cli.block_y, cli.warps)
    report = dict(status="running", python=sys.version, python_executable=sys.executable,
                  torch=torch.__version__, cuda=torch.version.cuda, triton=kernels.triton.__version__,
                  gpu=torch.cuda.get_device_name(device), device=str(device),
                  arguments={**vars(cli), "output": str(cli.output.resolve())},
                  baseline=config_dict(BASELINE), candidates=[config_dict(config) for config in candidates],
                  fixture="synthetic contiguous FP32 premultiplied RGBA; shared HWC-backed background; shared one-channel mask; verified 9x9 sigma=1.5 kernel",
                  scope="CUDA event span per launch; may include CPU launch gaps; excludes allocation, validation, adapter, shared preparation and NumPy transfer; not end-to-end latency",
                  batches=[])
    write_report(cli.output, report)
    print(json.dumps({key: report[key] for key in ("gpu", "torch", "triton", "scope")}), flush=True)
    print(f"{len(candidates)} candidates per stage; {cli.rounds} rounds; {cli.iterations} launches per event pair", flush=True)
    with torch.inference_mode():
        for batch in dict.fromkeys(cli.batches):
            fixture = make_fixture(torch, np, device, batch, cli.height, cli.width, cli.seed)
            horizontal_launcher(kernels, fixture, BASELINE, fixture["reference_temp"])()
            vertical_launcher(kernels, fixture, BASELINE, fixture["reference_output"])()
            torch.cuda.synchronize(device)
            result = dict(batch=batch, factor_max_absolute_error=fixture["factor_error"],
                          canvas_stride=list(fixture["canvas"].stride()), frames_stride=list(fixture["frames"].stride()),
                          mask_stride=list(fixture["mask"].stride()))
            rng = random.Random(cli.seed + batch)
            for stage in ("horizontal", "vertical"):
                result[stage] = tune_stage(torch, kernels, fixture, stage, candidates, cli, device, rng)
                print(json.dumps({"batch": batch, "stage": stage,
                                  "best_three": result[stage]["best_three"]}), flush=True)
            report["batches"].append(result)
            write_report(cli.output, report)
            del fixture
    rejected = sum(candidate["status"] == "rejected_difference"
                   for batch in report["batches"] for stage in ("horizontal", "vertical")
                   for candidate in batch[stage]["all_candidates"])
    report.update(status="completed" if not rejected else "completed_with_correctness_rejections",
                  correctness_rejections=rejected)
    write_report(cli.output, report)
    print(f"Complete report: {cli.output.resolve()}", flush=True)
    return 1 if rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
