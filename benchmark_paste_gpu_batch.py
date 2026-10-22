"""CUDA correctness and latency checks for paste_gpu_batch and its adapter.

Place beside paste_gpu_batch.py, paste_gpu_kernels.py and paste_gpu_batch_adapter.py.
Examples (compilation/warmup are excluded from reported latency):
    python benchmark_paste_gpu_batch.py --check-only --acceleration triton
    python benchmark_paste_gpu_batch.py --device cuda:3 --height 1080 --width 1920
    python benchmark_paste_gpu_batch.py --batch 2 --iterations 100 --kernel-size 9
    python benchmark_paste_gpu_batch.py --batch 8 --diagnose
    python benchmark_paste_gpu_batch.py --batch 8 --blur-backend triton --diagnose
    python benchmark_paste_gpu_batch.py --batch 8 --blur-backend triton --diagnose --ad-sizes 200x200 500x500 300x600

The core reference below is the original eager implementation, kept independent
of the optimized core. The adapter comparison uses the NEW adapter with eager
core/views disabled versus acceleration/views enabled; it is not a measurement
of the historical adapter. Timings use resident GPU inputs and warmed ad caches.
Use the production adapter's last_profile to measure real preprocessing/transfers.
Each latency sample is synchronized; this measures latency, not queued throughput.
--diagnose also measures GPU/pageable/pinned outputs, ready-output-only D2H,
and isolated GPU stages. These microbenchmarks are not additive pipeline timings.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

BENCHMARK_VERSION = "2026-10-22-perf4.8-uploads"
DIAGNOSTIC_ADAPTER_VERSION = "2026-10-22-perf4.8-uploads"


def parse_ad_size(value):
    """CLI sizes are HEIGHTxWIDTH, not bbox/output dimensions."""
    try:
        pieces = value.lower().split("x")
        if len(pieces) != 2:
            raise ValueError
        height, width = map(int, pieces)
        if min(height, width) < 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            "ad size must be positive HxW, e.g. 200x200 or 300x600"
        ) from None
    return height, width


def reference_paste(
    frames,
    ads,
    grids,
    distortion,
    kernel,
    mask,
    transparencies,
    *,
    bboxes,
    warp_align_corners=False,
    distortion_align_corners=False,
    warp_padding_mode="zeros",
    distortion_padding_mode="zeros",
    legacy_alpha=True,
):
    """Frozen pre-optimization eager math, including transparent ROI overwrite."""
    import torch
    import torch.nn.functional as F

    batch = ads[0].shape[0] if ads else frames.shape[0]
    height, width = frames.shape[-2:]
    canvas = frames.new_zeros((batch, 4, height, width))
    for source, grid, opacity, box in zip(ads, grids, transparencies, bboxes):
        x1, y1, x2, y2 = box
        grid = grid.expand(batch, -1, -1, -1)
        if legacy_alpha:
            warped = F.grid_sample(
                source,
                grid,
                mode="bilinear",
                padding_mode=warp_padding_mode,
                align_corners=warp_align_corners,
            )
            alpha = warped[:, 3:4] * (1.0 / 255.0) * opacity
            patch = torch.cat((warped[:, :3] * alpha, alpha), dim=1)
        else:
            source_alpha = source[:, 3:4] * (1.0 / 255.0)
            source = torch.cat((source[:, :3] * source_alpha, source_alpha), dim=1)
            patch = (
                F.grid_sample(
                    source,
                    grid,
                    mode="bilinear",
                    padding_mode=warp_padding_mode,
                    align_corners=warp_align_corners,
                )
                * opacity
            )
        canvas[:, :, y1:y2, x1:x2] = patch
    canvas = F.grid_sample(
        canvas,
        distortion.expand(batch, -1, -1, -1),
        mode="bilinear",
        padding_mode=distortion_padding_mode,
        align_corners=distortion_align_corners,
    )
    kh, kw = kernel.shape[-2:]
    if kernel.ndim == 4:
        canvas = F.conv2d(canvas, kernel, padding=(kh // 2, kw // 2), groups=4)
    else:
        canvas = F.conv2d(
            canvas.reshape(1, batch * 4, height, width),
            kernel.reshape(batch * 4, 1, kh, kw),
            padding=(kh // 2, kw // 2),
            groups=batch * 4,
        ).reshape(batch, 4, height, width)
    background = frames.expand(batch, -1, -1, -1)
    alpha = canvas[:, 3:4]
    color = canvas[:, :3] * alpha if legacy_alpha else canvas[:, :3]
    blended = color + background * (1.0 - alpha)
    blended = background * mask + blended * (1.0 - mask)
    return blended.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def identity_grid(height, width, device, align_corners=False):
    import torch

    if align_corners:
        y = (
            torch.linspace(-1, 1, height, device=device)
            if height > 1
            else torch.zeros(1, device=device)
        )
        x = (
            torch.linspace(-1, 1, width, device=device)
            if width > 1
            else torch.zeros(1, device=device)
        )
    else:
        y = (torch.arange(height, device=device) + 0.5) * (2.0 / height) - 1.0
        x = (torch.arange(width, device=device) + 0.5) * (2.0 / width) - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0)


def reference_homography(matrix, source_hw, box, canvas_xy, align):
    """Frozen eager homography formula, independent of accelerated grid builder."""
    import torch

    matrix = matrix.unsqueeze(0) if matrix.ndim == 2 else matrix
    x1, y1, x2, y2 = box
    y, x = torch.meshgrid(
        torch.arange(y2 - y1, device=matrix.device, dtype=matrix.dtype),
        torch.arange(x2 - x1, device=matrix.device, dtype=matrix.dtype),
        indexing="ij",
    )
    if canvas_xy:
        x, y = x + x1, y + y1
    projected = matrix @ torch.stack((x, y, torch.ones_like(x)), dim=0).reshape(1, 3, -1)
    z = projected[:, 2:3]
    valid = (z.abs() > 1e-8) & torch.isfinite(projected).all(dim=1, keepdim=True)
    xy = projected[:, :2] / torch.where(valid, z, torch.ones_like(z))
    sx, sy = xy[:, 0], xy[:, 1]
    sh, sw = source_hw
    gx, gy = (
        (2 * sx / (sw - 1) - 1, 2 * sy / (sh - 1) - 1)
        if align
        else ((2 * sx + 1) / sw - 1, (2 * sy + 1) / sh - 1)
    )
    grid = torch.stack((gx, gy), dim=-1)
    valid_grid = valid.transpose(1, 2) & torch.isfinite(grid).all(dim=-1, keepdim=True)
    return torch.where(valid_grid, grid, torch.full_like(grid, 1e6)).reshape(
        matrix.shape[0], y2 - y1, x2 - x1, 2
    )


def make_case(
    device,
    *,
    batch=1,
    height=48,
    width=64,
    slots=4,
    align=False,
    variant=0,
    kernel_size=3,
    realistic=False,
    source_sizes=None,
):
    import torch

    # HWC storage deliberately produces a noncontiguous NCHW background view.
    frames = (
        torch.rand((batch if variant % 2 else 1, height, width, 3), device=device) * 255
    ).permute(0, 3, 1, 2)
    roi_h = min(height - 2, max(5, height // (10 if realistic else 3)))
    roi_w = min(width - 2, max(7, width // (6 if realistic else 2)))
    if not realistic:
        roi_h -= variant % 2
        roi_w -= variant % 3
    boxes, ads, grids, opacity = [], [], [], []
    for i in range(slots):
        x1 = (i * max(1, roi_w // 2) + variant) % (width - roi_w + 1)
        y1 = (i * max(1, roi_h // 2) + variant) % (height - roi_h + 1)
        boxes.append((x1, y1, x1 + roi_w, y1 + roi_h))
        if source_sizes is not None:
            sh, sw = source_sizes[i % len(source_sizes)]
        else:
            sh, sw = (
                ((200, 200) if i % 2 == 0 else (500, 500))
                if realistic
                else ((13, 17) if i % 2 == 0 else (19, 23))
            )
        ad = (torch.rand((batch, sh, sw, 4), device=device) * 255).permute(0, 3, 1, 2)
        ad[:, 3, : sh // 3] = 0  # Transparent pixels must also overwrite old ROI.
        ads.append(ad.contiguous() if realistic else ad)
        n = batch if (variant + i) % 2 else 1
        grid = identity_grid(roi_h, roi_w, device, align).expand(n, -1, -1, -1).clone()
        grid += torch.randn_like(grid) * 0.015
        if not realistic:
            grid[:, 0, :3] = 2.3  # Exercise out-of-bounds border semantics.
        grids.append(grid)
        if realistic or i % 3 == 0:
            opacity.append(torch.full((1, 1, 1, 1), 0.83, device=device))
        elif i % 3 == 1:
            opacity.append(torch.rand((batch, 1, roi_h, roi_w), device=device))
        else:
            opacity.append(torch.rand((batch, 1, 1, 1), device=device))
    distortion = identity_grid(height, width, device, align)
    if not realistic:
        distortion = distortion.expand(batch if variant % 2 else 1, -1, -1, -1).clone()
        distortion += torch.randn_like(distortion) * 0.01
    # Both shared/per-group and one-/three-channel masks are covered.
    mask_shape = (batch if variant % 2 else 1, 3 if variant % 3 else 1, height, width)
    mask = torch.rand(mask_shape, device=device) * 0.35
    if not realistic and variant % 4 == 1:
        mask = mask[:, :, :1, :]
    elif not realistic and variant % 4 == 2:
        mask = mask[:, :, :, :1]
    kernel_shape = (
        (batch, 4, 1, kernel_size, kernel_size) if variant % 2 else (4, 1, kernel_size, kernel_size)
    )
    kernel = torch.rand(kernel_shape, device=device)
    kernel = kernel / kernel.sum(dim=(-2, -1), keepdim=True)
    args = (frames, tuple(ads), tuple(grids), distortion, kernel, mask, tuple(opacity))
    return args, tuple(boxes)


def assert_close(actual, expected, label, max_abs=1):
    import numpy as np
    import torch

    if isinstance(actual, np.ndarray):
        actual = torch.from_numpy(actual)
    if isinstance(expected, np.ndarray):
        expected = torch.from_numpy(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"{label}: shape/dtype differ: {actual.shape}/{actual.dtype}, "
            f"{expected.shape}/{expected.dtype}"
        )
    diff = (actual.to(device=expected.device, dtype=torch.int16) - expected.to(torch.int16)).abs()
    worst = diff.max().item()
    if worst > max_abs:
        raise AssertionError(
            f"{label}: max uint8 difference {worst} > {max_abs}; "
            f"mean difference {diff.float().mean().item():.6f}"
        )
    return worst


def make_adapter_case(args, boxes):
    frames, ads, grids, distortion, kernel, mask, opacity = args
    batch = ads[0].shape[0]
    groups = [
        {f"ad{i}": {"img": ad[b].permute(1, 2, 0)} for i, ad in enumerate(ads)}
        for b in range(batch)
    ]
    owner = SimpleNamespace(device=[str(frames.device)], imgs_raw_groups=groups, imgs_raw_version=0)
    params = [
        {
            "ad_placement_id": f"ad{i}",
            "ad_id": i,
            "bbox": box,
            "ad_h_w": tuple(ad.shape[-2:]),
            "transparency": trans,
        }
        for i, (box, ad, trans) in enumerate(zip(boxes, ads, opacity))
    ]
    paste_args = (
        frames.device,
        distortion,
        None,
        kernel,
        mask,
        {i: grid for i, grid in enumerate(grids)},
    )
    frame_hwc = frames.permute(0, 2, 3, 1)
    if frame_hwc.shape[0] == 1:
        frame_hwc = frame_hwc[0]
    return owner, frame_hwc, {"ad_params": params}, 0, 1, paste_args


def adapter_options(
    acceleration,
    *,
    fast=True,
    align=False,
    padding="zeros",
    legacy=True,
    blur_backend="torch",
    max_cached_sizes=2,
):
    from paste_gpu_batch_adapter import AdapterOptions

    return AdapterOptions(
        acceleration=acceleration,
        blur_backend=blur_backend,
        optimize_views=fast,
        max_cached_sizes=max_cached_sizes,
        cache_scalar_maps=fast,
        pinned_output=False,
        profile_every=0,
        backend="batch",
        frame_layout="HWC",
        ad_layout="HWC",
        mask_layout="BCHW",
        transparency_layout="BCHW",
        return_numpy=True,
        squeeze_single=True,
        warp_align_corners=align,
        distortion_align_corners=align,
        warp_padding_mode=padding,
        distortion_padding_mode=padding,
        legacy_alpha=legacy,
    )


def run_checks(device, acceleration, blur_backend="torch"):
    import torch
    from paste_gpu_batch import paste_gpu_batch
    from paste_gpu_batch_adapter import PasteBatchAdapter

    torch.manual_seed(12345)
    count, worst = 0, 0
    for batch in (1, 2):
        for align in (False, True):
            for legacy in (False, True):
                for padding in ("zeros", "border", "reflection"):
                    args, boxes = make_case(device, batch=batch, align=align, variant=count)
                    options = dict(
                        bboxes=boxes,
                        warp_align_corners=align,
                        distortion_align_corners=align,
                        warp_padding_mode=padding,
                        distortion_padding_mode=padding,
                        legacy_alpha=legacy,
                    )
                    expected = reference_paste(*args, **options)
                    actual = paste_gpu_batch(
                        *args, acceleration=acceleration, blur_backend=blur_backend, **options
                    )
                    label = f"B={batch}, align={align}, legacy={legacy}, padding={padding}"
                    worst = max(worst, assert_close(actual, expected, label))
                    adapter = PasteBatchAdapter(
                        adapter_options(
                            acceleration,
                            align=align,
                            padding=padding,
                            legacy=legacy,
                            blur_backend=blur_backend,
                        )
                    )
                    adapter_args = make_adapter_case(args, boxes)
                    converted = adapter(*adapter_args)
                    expected_cpu = expected.cpu().numpy()
                    if batch == 1:
                        expected_cpu = expected_cpu[0]
                    assert_close(converted, expected_cpu, "adapter " + label)
                    # Same adapter/source cache, fresh grids + shifted ROI next frame.
                    moved_args, moved_boxes = make_case(
                        device, batch=batch, align=align, variant=count + 1
                    )
                    moved_args = (
                        args[0],
                        args[1],
                        moved_args[2],
                        args[3],
                        args[4],
                        args[5],
                        moved_args[6],
                    )
                    changed = make_adapter_case(moved_args, moved_boxes)
                    changed = (adapter_args[0], *changed[1:])
                    expected_moved = reference_paste(
                        *moved_args, **dict(options, bboxes=moved_boxes)
                    )
                    expected_moved = expected_moved.cpu().numpy()
                    if batch == 1:
                        expected_moved = expected_moved[0]
                    assert_close(adapter(*changed), expected_moved, "changed geometry " + label)
                    count += 1

    # Independent exact assertion: transparent later slot clears earlier slot.
    frames = torch.full((1, 3, 16, 20), 73.0, device=device)
    opaque = torch.full((1, 4, 8, 8), 255.0, device=device)
    transparent = opaque.clone()
    transparent[:, 3] = 0
    grid = identity_grid(8, 8, device)
    args = (
        frames,
        (opaque, transparent),
        (grid, grid),
        identity_grid(16, 20, device),
        torch.ones((4, 1, 1, 1), device=device),
        torch.zeros((1, 1, 16, 20), device=device),
        (torch.ones((1, 1, 1, 1), device=device),) * 2,
    )
    actual = paste_gpu_batch(
        *args, bboxes=((3, 2, 11, 10),) * 2, acceleration=acceleration, blur_backend=blur_backend
    )
    assert_close(actual, frames.to(torch.uint8).permute(0, 2, 3, 1), "transparent overwrite", 0)

    # No-ad output must retain the classic HWC NumPy uint8 interface.
    background = torch.rand((16, 20, 3), device=device) * 300 - 20
    no_ads = PasteBatchAdapter(adapter_options(acceleration, blur_backend=blur_backend))
    no_ad_output = no_ads(SimpleNamespace(imgs_raw={}), background, {}, 0, 1, None)
    assert_close(
        no_ad_output, background.clamp(0, 255).to(torch.uint8).cpu().numpy(), "no ads HWC", 0
    )

    # Reuse the SAME matrix tensors while changing their values between frames.
    # Both coordinate conventions and both align_corners settings are tested.
    for align in (False, True):
        for canvas_xy in (False, True):
            args, boxes = make_case(device, batch=2, slots=2, align=align, variant=1)
            adapter_args = make_adapter_case(args, boxes)
            matrices = []
            for ad, box, param in zip(args[1], boxes, adapter_args[2]["ad_params"]):
                sh, sw = ad.shape[-2:]
                x1, y1, x2, y2 = box
                sx, sy = (sw - 1) / (x2 - x1), (sh - 1) / (y2 - y1)
                tx, ty = (-x1 * sx, -y1 * sy) if canvas_xy else (0, 0)
                matrix = torch.tensor(
                    [[sx, 0.01, tx], [0.02, sy, ty], [0.001, 0, 1]], device=device
                ).repeat(2, 1, 1)
                matrix[1, 0, 2] += 0.7
                param["m_inv"] = matrix
                matrices.append(matrix)
            options = replace(
                adapter_options(acceleration, align=align, blur_backend=blur_backend),
                grid_source="matrix",
                matrix_coordinates="canvas" if canvas_xy else "roi",
            )
            adapter = PasteBatchAdapter(options)
            for step in range(2):
                if step:
                    for matrix in matrices:
                        matrix[:, 0, 2].add_(1.3)
                        matrix[:, 1, 2].sub_(0.4)
                grids = tuple(
                    reference_homography(matrix, ad.shape[-2:], box, canvas_xy, align)
                    for matrix, ad, box in zip(matrices, args[1], boxes)
                )
                expected_args = (args[0], args[1], grids, *args[3:])
                expected = (
                    reference_paste(
                        *expected_args,
                        bboxes=boxes,
                        warp_align_corners=align,
                        distortion_align_corners=align,
                    )
                    .cpu()
                    .numpy()
                )
                assert_close(
                    adapter(*adapter_args),
                    expected,
                    f"updated matrix: align={align}, canvas_xy={canvas_xy}, step={step}",
                )
    print(
        f"PASS: {count} core + {count * 2} adapter cases; transparent overwrite; "
        f"no-ad HWC; 8 matrix-update cases; maximum core uint8 difference={worst}. "
        "Reflection tests use the documented fallback."
    )


def reference_blur_blend(canvas, frames, kernel, mask, *, legacy_alpha=True):
    """Independent PyTorch depthwise convolution + original blending math."""
    import torch
    import torch.nn.functional as F

    batch, _, height, width = canvas.shape
    kh, kw = kernel.shape[-2:]
    if kernel.ndim == 4:
        blurred = F.conv2d(canvas, kernel, padding=(kh // 2, kw // 2), groups=4)
    else:
        blurred = F.conv2d(
            canvas.reshape(1, batch * 4, height, width),
            kernel.reshape(batch * 4, 1, kh, kw),
            padding=(kh // 2, kw // 2),
            groups=batch * 4,
        ).reshape(batch, 4, height, width)
    background = frames.expand(batch, -1, -1, -1)
    alpha = blurred[:, 3:4]
    color = blurred[:, :3] * alpha if legacy_alpha else blurred[:, :3]
    result = color + background * (1.0 - alpha)
    result = background * mask + result * (1.0 - mask)
    return result.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def run_blur_kernel_checks(device, blur_callable=None):
    """Small direct-kernel suite; injectable callable also permits interpreter checks.

    blur_callable(canvas, frames, kernel, mask, legacy_alpha=bool) -> BHWC uint8.
    A supplied callable is responsible for selecting its own execution backend.
    This function does not catch numerical errors or label skipped CUDA tests passed.
    """
    import torch

    if blur_callable is None:
        from paste_gpu_kernels import blur_blend_to_uint8

        blur_callable = blur_blend_to_uint8
    torch.manual_seed(67890)
    count, worst = 0, 0
    for batch in (1, 2):
        for legacy in (False, True):
            for kh, kw in ((1, 1), (3, 5), (9, 9)):
                for per_batch in (False, True):
                    height, width = 7, 11  # 9x9 is taller than the input: edge padding matters.
                    canvas = torch.rand((batch, height, width, 4), device=device).permute(
                        0, 3, 1, 2
                    )
                    canvas[:, :3].mul_(255.0 * canvas[:, 3:4])
                    frame_batch = batch if per_batch else 1
                    frames = (
                        torch.rand((frame_batch, height, width, 3), device=device) * 255
                    ).permute(0, 3, 1, 2)
                    channels = 3 if legacy else 1
                    # Cover spatially broadcast and noncontiguous masks too.
                    mask = torch.rand(
                        (frame_batch, height, 1 if per_batch else width, channels), device=device
                    ).permute(0, 3, 1, 2)
                    shape = (batch, 4, 1, kw, kh) if per_batch else (4, 1, kw, kh)
                    kernel = torch.rand(shape, device=device).transpose(-1, -2)
                    kernel.div_(kernel.sum(dim=(-2, -1), keepdim=True))
                    if batch == 2 and not per_batch:
                        # A zero-stride batch axis is legal for shared weights.
                        kernel = kernel.unsqueeze(0).expand(batch, -1, -1, -1, -1)
                    steps = 1 if (kh, kw) == (1, 1) else 2
                    previous_expected = None
                    for step in range(steps):
                        if step:
                            # Mutate the original weights in place, retaining object identity.
                            writable = (
                                kernel[0] if kernel.ndim == 5 and kernel.stride(0) == 0 else kernel
                            )
                            writable.zero_()
                            writable[..., 0, 0] = 1.0
                        expected = reference_blur_blend(
                            canvas, frames, kernel, mask, legacy_alpha=legacy
                        )
                        actual = blur_callable(canvas, frames, kernel, mask, legacy_alpha=legacy)
                        label = (
                            f"fused blur: B={batch}, legacy={legacy}, kernel={kh}x{kw}, "
                            f"per_batch={per_batch}, update={step}"
                        )
                        worst = max(worst, assert_close(actual, expected, label))
                        if previous_expected is not None and torch.equal(
                            previous_expected, expected
                        ):
                            raise AssertionError(
                                f"{label}: weight update fixture did not change output"
                            )
                        previous_expected = expected
                        count += 1
    result = {"cases": count, "max_uint8_difference": worst}
    print(f"PASS: fused blur direct-kernel suite: {count} cases, max uint8 difference={worst}.")
    return result


def summarize(values):
    ordered = sorted(values)
    return {
        "mean": round(statistics.mean(values), 3),
        "p50": round(statistics.median(values), 3),
        "p95": round(ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)], 3),
    }


def measure(fn, device, warmup, iterations, cuda_events):
    import torch

    # First invocation may compile Triton kernels; do not include it in samples.
    for _ in range(warmup):
        _output = fn()
    torch.cuda.synchronize(device)
    wall, gpu = [], []
    start_event = torch.cuda.Event(enable_timing=True) if cuda_events else None
    end_event = torch.cuda.Event(enable_timing=True) if cuda_events else None
    for _ in range(iterations):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        if cuda_events:
            start_event.record()
        _output = fn()
        if cuda_events:
            end_event.record()
        torch.cuda.synchronize(device)
        wall.append((time.perf_counter() - start) * 1000)
        if cuda_events:
            gpu.append(start_event.elapsed_time(end_event))
    # Keep output alive until synchronization; do not copy tensors inside core timing.
    result = {"wall_ms": summarize(wall)}
    if cuda_events:
        result["cuda_event_span_ms"] = summarize(gpu)
    return result


def check_pinned_output_lifetime(adapter, adapter_args):
    """Keep a returned NumPy image alive while another frame uses the same adapter."""
    import numpy as np
    import torch

    previous = adapter(*adapter_args)
    frozen = previous.copy()
    changed = (adapter_args[0], torch.full_like(adapter_args[1], 251), *adapter_args[2:])
    following = adapter(*changed)
    if not np.array_equal(previous, frozen):
        raise AssertionError("pinned output lifetime: next call overwrote a retained NumPy result")
    if np.shares_memory(previous, following):
        raise AssertionError("pinned output lifetime: two live results share mutable memory")
    if np.array_equal(following, frozen):
        raise AssertionError(
            "pinned output lifetime fixture did not produce a different next frame"
        )


def diagnose_stages(cli, device, args, boxes):
    """Standalone measurements on already-prepared inputs; not sums of pipeline cost."""
    import torch
    import torch.nn.functional as F
    from paste_gpu_batch import _select_accelerator

    frames, ads, grids, distortion, kernel, mask, opacity = args
    batch, height, width = ads[0].shape[0], frames.shape[-2], frames.shape[-1]
    canvas = frames.new_zeros((batch, 4, height, width))
    accelerator = _select_accelerator(cli.acceleration, canvas, frames)

    def warps():
        # Rewrites every ROI; canvas outside the fixed ROIs remains zero.
        for ad, grid, trans, box in zip(ads, grids, opacity, boxes):
            if accelerator is not None:
                accelerator.warp_paste_into(
                    ad,
                    grid,
                    trans,
                    canvas,
                    box,
                    align_corners=False,
                    padding_mode="zeros",
                    legacy_alpha=True,
                )
            else:
                warped = F.grid_sample(
                    ad,
                    grid.expand(batch, -1, -1, -1),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                alpha = warped[:, 3:4] * (1.0 / 255.0) * trans
                x1, y1, x2, y2 = box
                canvas[:, :, y1:y2, x1:x2] = torch.cat((warped[:, :3] * alpha, alpha), dim=1)
        return canvas

    warps()

    def distort():
        return F.grid_sample(
            canvas,
            distortion.expand(batch, -1, -1, -1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    distorted = distort()
    kh, kw = kernel.shape[-2:]
    # The synthetic benchmark uses one shared kernel. Explicitly reject accidental
    # fixture changes instead of silently benchmarking a different convolution.
    if kernel.ndim != 4:
        raise ValueError("stage diagnosis currently requires a shared [4,1,kh,kw] kernel")

    def conv(value):
        return F.conv2d(value, kernel, padding=(kh // 2, kw // 2), groups=4)

    channels_last = distorted.contiguous(memory_format=torch.channels_last)
    blurred = conv(distorted)
    alternative = conv(channels_last)
    channels_last_matches = torch.allclose(alternative, blurred, rtol=2e-5, atol=1e-3)
    float_difference = (alternative - blurred).abs().max().item()
    del alternative

    def blend(value=blurred):
        if accelerator is not None:
            return accelerator.blend_to_uint8(value, frames, mask, legacy_alpha=True)
        alpha = value[:, 3:4]
        background = frames.expand(batch, -1, -1, -1)
        color = value[:, :3] * alpha + background * (1.0 - alpha)
        color = background * mask + color * (1.0 - mask)
        return color.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()

    common = dict(device=device, warmup=cli.warmup, iterations=cli.iterations, cuda_events=True)
    report = {
        "canvas_allocate_zero": measure(lambda: frames.new_zeros(canvas.shape), **common),
        "warps_into_existing_canvas": measure(warps, **common),
        "distortion": measure(distort, **common),
        "blur_nchw": measure(lambda: conv(distorted), **common),
        "blur_channels_last_preconverted": measure(lambda: conv(channels_last), **common),
        "blur_channels_last_with_input_conversion": measure(
            lambda: conv(distorted.contiguous(memory_format=torch.channels_last)), **common
        ),
        "final_blend_uint8": measure(blend, **common),
        "blur_channels_last_max_abs_float_diff": float_difference,
        "blur_channels_last_within_tolerance": channels_last_matches,
        "blur_channels_last_parity_status": (
            "within float tolerance; production output validation still required"
            if channels_last_matches
            else "NOT ELIGIBLE: experimental channels-last blur differs beyond float tolerance"
        ),
        "note": "Isolated warmed-input microbenchmarks; do not add stage timings. "
        "Preconverted channels-last excludes layout conversion; the other variant includes it. "
        "No channels-last setting is applied to the production core by this diagnostic.",
    }
    if cli.blur_backend == "triton":
        if accelerator is None:
            raise RuntimeError("fused blur diagnosis requires an active Triton accelerator")

        def fused():
            return accelerator.blur_blend_to_uint8(
                distorted, frames, kernel, mask, legacy_alpha=True
            )

        expected = reference_blur_blend(distorted, frames, kernel, mask, legacy_alpha=True)
        report["fused_blur_blend_max_uint8_difference"] = assert_close(
            fused(), expected, "diagnostic fused blur + blend"
        )
        report["torch_blur_then_accelerated_blend"] = measure(
            lambda: blend(conv(distorted)), **common
        )
        report["fused_blur_blend_uint8"] = measure(fused, **common)
    return report


def run_diagnostics(cli, device, args, boxes, adapter_args, pageable_adapter):
    import torch
    from paste_gpu_batch import paste_gpu_batch
    from paste_gpu_batch_adapter import PasteBatchAdapter

    options = adapter_options(
        cli.acceleration, blur_backend=cli.blur_backend, max_cached_sizes=cli.max_cached_sizes
    )
    gpu_adapter = PasteBatchAdapter(replace(options, return_numpy=False))
    pinned_adapter = PasteBatchAdapter(replace(options, pinned_output=True))
    expected = pageable_adapter(*adapter_args)
    gpu = gpu_adapter(*adapter_args)
    pinned = pinned_adapter(*adapter_args)
    assert_close(gpu.cpu().numpy(), expected, "diagnostic GPU output", 0)
    assert_close(pinned, expected, "diagnostic pinned output", 0)
    check_pinned_output_lifetime(pinned_adapter, adapter_args)
    del expected, gpu, pinned

    # Compute this once, then synchronize before EVERY copy sample. These D2H
    # results cannot include waits for the render kernels of the source frame.
    ready_output = paste_gpu_batch(
        *args, bboxes=boxes, acceleration=cli.acceleration, blur_backend=cli.blur_backend
    )
    torch.cuda.synchronize(device)
    payload_bytes = ready_output.numel() * ready_output.element_size()
    common = dict(device=device, warmup=cli.warmup, iterations=cli.iterations)
    report = {
        "requested_blur_backend": cli.blur_backend,
        "output_payload_bytes": payload_bytes,
        "output_payload_MB": payload_bytes / 1_000_000,
        "adapter_accelerated_gpu_output": measure(
            lambda: gpu_adapter(*adapter_args), cuda_events=True, **common
        ),
        "adapter_accelerated_pinned_cpu_output": measure(
            lambda: pinned_adapter(*adapter_args), cuda_events=False, **common
        ),
        "ready_output_d2h_pageable": measure(
            lambda: pageable_adapter._export(ready_output), cuda_events=False, **common
        ),
        "ready_output_d2h_pinned_owned": measure(
            lambda: pinned_adapter._export(ready_output), cuda_events=False, **common
        ),
        "pinned_output_lifetime_check": "passed: two live NumPy outputs stay independent",
    }
    if cli.blur_backend == "torch":
        report["adapter_accelerated_torch_blur_pinned_cpu_output"] = report[
            "adapter_accelerated_pinned_cpu_output"
        ]
    else:
        torch_blur_adapter = PasteBatchAdapter(
            replace(options, blur_backend="torch", pinned_output=True)
        )
        assert_close(
            pinned_adapter(*adapter_args),
            torch_blur_adapter(*adapter_args),
            "requested fused blur vs prior accelerated torch blur, both pinned",
        )
        report["adapter_accelerated_torch_blur_pinned_cpu_output"] = measure(
            lambda: torch_blur_adapter(*adapter_args), cuda_events=False, **common
        )
    report["additional_blur_saving_vs_torch_pinned_ms"] = round(
        report["adapter_accelerated_torch_blur_pinned_cpu_output"]["wall_ms"]["mean"]
        - report["adapter_accelerated_pinned_cpu_output"]["wall_ms"]["mean"],
        3,
    )
    for key in ("ready_output_d2h_pageable", "ready_output_d2h_pinned_owned"):
        milliseconds = report[key]["wall_ms"]["mean"]
        report[key]["effective_payload_GB_per_s"] = round(
            payload_bytes / (milliseconds * 1_000_000), 3
        )
    report["stages"] = diagnose_stages(cli, device, args, boxes)
    report["note"] = (
        "The top-level adapter_accelerated_cpu_output is the pageable comparison. "
        "Ready-output D2H timings include host allocation, transfer, synchronization and NumPy wrapping; "
        "effective_payload_GB_per_s is not raw PCIe bandwidth. Pinned outputs own independent storage. "
        "The torch_blur_pinned comparison uses the same prepared inputs, acceleration and output mode. "
        "All adapter variants are warmed separately. Timings are not additive."
    )
    return report


def prepare_benchmark_case(cli, device):
    """Check the requested source sizes even with --check-only; probe a repeated call."""
    from paste_gpu_batch import paste_gpu_batch
    from paste_gpu_batch_adapter import PasteBatchAdapter

    args, boxes = make_case(
        device,
        batch=cli.batch,
        height=cli.height,
        width=cli.width,
        slots=cli.slots,
        kernel_size=cli.kernel_size,
        realistic=True,
        source_sizes=cli.ad_sizes,
    )
    core_difference = assert_close(
        paste_gpu_batch(
            *args, bboxes=boxes, acceleration=cli.acceleration, blur_backend=cli.blur_backend
        ),
        reference_paste(*args, bboxes=boxes),
        "requested-size core output",
    )
    adapter_args = make_adapter_case(args, boxes)
    # Match common production metadata: Python numeric opacity each frame.
    for param in adapter_args[2]["ad_params"]:
        param["transparency"] = 0.83
    slow_adapter = PasteBatchAdapter(
        adapter_options("eager", fast=False, max_cached_sizes=cli.max_cached_sizes)
    )
    fast_adapter = PasteBatchAdapter(
        adapter_options(
            cli.acceleration, blur_backend=cli.blur_backend, max_cached_sizes=cli.max_cached_sizes
        )
    )
    expected = slow_adapter(*adapter_args)
    adapter_difference = assert_close(
        fast_adapter(*adapter_args), expected, "requested-size adapter output"
    )
    before = fast_adapter.cache_info()
    repeat_difference = assert_close(
        fast_adapter(*adapter_args), expected, "requested-size repeated output"
    )
    after = fast_adapter.cache_info()
    per_slot = [tuple(ad.shape[-2:]) for ad in args[1]]
    unique_sizes = list(dict.fromkeys(per_slot))
    counts = [{"hw": list(hw), "slots": per_slot.count(hw)} for hw in unique_sizes]
    repeat_hits = after["hits"] - before["hits"]
    repeat_misses = after["misses"] - before["misses"]
    capacity_sufficient = (
        len(unique_sizes) <= fast_adapter.options.max_cached_sizes
        and max(item["slots"] for item in counts) <= fast_adapter.options.max_entries_per_size
    )
    if capacity_sufficient and (repeat_misses != 0 or repeat_hits != cli.slots):
        raise AssertionError("unchanged source images should all hit a sufficiently sized cache")
    fixture = {
        "source_sizes_by_slot_hw": per_slot,
        "source_size_counts": counts,
        "distinct_source_sizes": len(unique_sizes),
        "same_slot_size_across_batch": True,
        "source_hw_is_not_output_roi_hw": True,
        "roi_hw_by_slot": [(y2 - y1, x2 - x1) for x1, y1, x2, y2 in boxes],
        "max_cached_sizes": cli.max_cached_sizes,
        "max_entries_per_size": fast_adapter.options.max_entries_per_size,
        "repeat_cache_hits": repeat_hits,
        "repeat_cache_misses": repeat_misses,
        "cached_sizes_after_repeat": [
            {"hw": list(hw), "entries": entries} for hw, entries in after["sizes"].items()
        ],
        "cache_capacity_sufficient": capacity_sufficient,
        "max_core_uint8_difference": core_difference,
        "max_adapter_uint8_difference": max(adapter_difference, repeat_difference),
    }
    print(
        f"PASS: requested mixed-size fixture: {len(unique_sizes)} sizes, {cli.slots} slots, "
        f"B={cli.batch}; repeated cache hits={repeat_hits}, misses={repeat_misses}."
    )
    if not capacity_sufficient:
        print(
            "NOTE: cache capacity is insufficient for this fixture; adapter timings include eviction/repacking."
        )
    return args, boxes, adapter_args, slow_adapter, fast_adapter, fixture


def run_benchmark(cli, device):
    from paste_gpu_batch import paste_gpu_batch

    args, boxes, adapter_args, slow_adapter, fast_adapter, fixture = prepare_benchmark_case(
        cli, device
    )

    def baseline():
        return reference_paste(*args, bboxes=boxes)

    def fast():
        return paste_gpu_batch(
            *args, bboxes=boxes, acceleration=cli.acceleration, blur_backend=cli.blur_backend
        )

    common = dict(device=device, warmup=cli.warmup, iterations=cli.iterations)
    report = {
        "fixture": fixture,
        "core_reference": measure(baseline, cuda_events=True, **common),
        "core_accelerated": measure(fast, cuda_events=True, **common),
        "adapter_eager_cpu_output": measure(
            lambda: slow_adapter(*adapter_args), cuda_events=False, **common
        ),
        "adapter_accelerated_cpu_output": measure(
            lambda: fast_adapter(*adapter_args), cuda_events=False, **common
        ),
    }
    report["adapter_mean_saving_ms"] = round(
        report["adapter_eager_cpu_output"]["wall_ms"]["mean"]
        - report["adapter_accelerated_cpu_output"]["wall_ms"]["mean"],
        3,
    )
    if cli.diagnose:
        report["diagnostics"] = run_diagnostics(
            cli, device, args, boxes, adapter_args, fast_adapter
        )
        pinned_ms = report["diagnostics"]["adapter_accelerated_pinned_cpu_output"]["wall_ms"][
            "mean"
        ]
        report["diagnostics"]["additional_pinned_saving_ms"] = round(
            report["adapter_accelerated_cpu_output"]["wall_ms"]["mean"] - pinned_ms, 3
        )
        report["diagnostics"]["total_pinned_saving_vs_eager_ms"] = round(
            report["adapter_eager_cpu_output"]["wall_ms"]["mean"] - pinned_ms, 3
        )
    print(json.dumps(report, indent=2))
    print(
        "CUDA event span includes host submission gaps; wall_ms is synchronized call latency. "
        "Adapter measurements include CPU uint8 NumPy output and the configured asset-cache behavior. "
        "Synthetic results are not a guarantee of production end-to-end savings."
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {BENCHMARK_VERSION}")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--acceleration", choices=("auto", "triton", "eager"), default="triton")
    parser.add_argument(
        "--blur-backend",
        choices=("torch", "triton"),
        default="torch",
        help="torch keeps the previous blur; triton explicitly tests fused blur+blend",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="also measure output transfer variants and isolated GPU stages",
    )
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument(
        "--ad-sizes",
        type=parse_ad_size,
        nargs="+",
        default=[(200, 200), (500, 500)],
        help="source HxW sizes assigned round-robin to slots; same slot size in every batch group",
    )
    parser.add_argument(
        "--max-cached-sizes",
        type=int,
        default=None,
        help="size-cache banks for every benchmark adapter; default: number of sizes actually used",
    )
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    cli = parser.parse_args()
    if min(cli.height, cli.width) < 8 or min(cli.batch, cli.slots, cli.iterations, cli.warmup) < 1:
        parser.error("height/width >= 8 and batch/slots/iterations/warmup >= 1 required")
    if cli.max_cached_sizes is None:
        cli.max_cached_sizes = len({cli.ad_sizes[i % len(cli.ad_sizes)] for i in range(cli.slots)})
    if cli.max_cached_sizes < 1:
        parser.error("max-cached-sizes must be >= 1")
    if cli.kernel_size < 1 or cli.kernel_size % 2 != 1:
        parser.error("kernel-size must be a positive odd number")
    if cli.blur_backend == "triton" and cli.acceleration == "eager":
        parser.error("--blur-backend triton requires --acceleration triton or auto")
    if cli.blur_backend == "triton" and cli.kernel_size > 15:
        parser.error("--blur-backend triton currently supports odd kernel sizes up to 15")
    return parser, cli


def main():
    parser, cli = parse_arguments()
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        parser.exit(
            2, "UNABLE TO TEST: PyTorch is not installed; no correctness/performance results.\n"
        )
    device = torch.device(cli.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.exit(
            2, "UNABLE TO TEST: a working CUDA GPU is required; no tests were skipped as passed.\n"
        )
    import paste_gpu_batch_adapter as adapter_module

    adapter_version = getattr(adapter_module, "ADAPTER_VERSION", "unknown")
    if (
        cli.diagnose or cli.blur_backend == "triton"
    ) and adapter_version != DIAGNOSTIC_ADAPTER_VERSION:
        parser.error(
            f"This diagnostic/fused-blur run requires adapter {DIAGNOSTIC_ADAPTER_VERSION}; loaded "
            f"{adapter_version} from {adapter_module.__file__}. "
            "Extract all files from the matching performance bundle into the same directory."
        )
    torch.cuda.set_device(device)
    print(
        json.dumps(
            {
                "benchmark_version": BENCHMARK_VERSION,
                "benchmark_path": str(Path(__file__).resolve()),
                "python": sys.version,
                "python_executable": sys.executable,
                "adapter_version": adapter_version,
                "adapter_path": str(Path(adapter_module.__file__).resolve()),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "arguments": vars(cli),
            },
            indent=2,
        )
    )
    with torch.inference_mode():
        run_checks(device, cli.acceleration, cli.blur_backend)
        if cli.blur_backend == "triton":
            run_blur_kernel_checks(device)
        if cli.check_only:
            *_, fixture = prepare_benchmark_case(cli, device)
            print(json.dumps({"fixture": fixture}, indent=2))
        else:
            run_benchmark(cli, device)


if __name__ == "__main__":
    main()
