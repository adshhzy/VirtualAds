"""Shared reporting and timing helpers for offline benchmarks.

This module has no import-time GPU dependency. The timed interval always starts
after fixture updates and ends after CUDA completion; it includes rendering and
the synchronous NumPy result. Do not use these helpers in the production path.
"""

from __future__ import annotations

import json
from pathlib import Path
import time


def serializable_arguments(arguments):
    """Keep the CLI report JSON-compatible without altering the parsed options."""
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(arguments).items()
    }


def save_report(report, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


def latency_statistics(samples_ms, *, include_minimum=False):
    import numpy as np

    result = {
        "mean": float(np.mean(samples_ms)),
        "p50": float(np.median(samples_ms)),
        "p95": float(np.percentile(samples_ms, 95)),
    }
    if include_minimum:
        result["min"] = float(np.min(samples_ms))
    return result


def rotated_order(candidates, round_index):
    """Rotate the first candidate to balance ordering effects across rounds."""
    offset = round_index % len(candidates)
    return candidates[offset:] + candidates[:offset]


def measure_render_path(path, *, warmup, iterations, device, reset_peak_memory=False):
    """Return full-call samples while retaining the previous frame during each call.

    Retaining the output across iterations mirrors a consumer holding the prior
    frame while the next independent NumPy array is produced.
    """
    import torch

    for step in range(warmup):
        path.fixture.set_frame(step)
        output = path.render()
    torch.cuda.synchronize(device)
    if reset_peak_memory:
        torch.cuda.reset_peak_memory_stats(device)

    samples_ms = []
    for step in range(iterations):
        path.fixture.set_frame(warmup + step)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = path.render()
        torch.cuda.synchronize(device)
        samples_ms.append((time.perf_counter() - started) * 1000)
    del output
    return samples_ms
