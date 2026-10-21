"""Small correctness checks for choosing batch versus bucketed execution.

Run CPU checks without initializing CUDA:
    python check_backend_contracts.py --device cpu
Run GPU checks serially, outside any performance measurement:
    python check_backend_contracts.py --device cuda:0

CPU checks require exact eager outputs. CUDA additionally checks dense/separable
Triton paths against the frozen eager reference and each other. Their default
uint8 tolerance is one level, matching the existing accelerated checks; every
comparison reports whether it was actually bitexact. Use --max-uint8-diff 0 when
an execution-policy change requires strict bitexact output. This is a synthetic
correctness fixture and makes no performance or production-callback claims.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path


HEIGHT, WIDTH = 19, 29
SOURCE_SIZES = ((7, 9), (11, 6), (7, 9), (5, 12))
BOXES = ((1, 1, 13, 10), (5, 3, 26, 12), (2, 2, 7, 10), (4, 2, 15, 8))
AREA_CAPS = (64, 256)


def compare(actual, expected, label, tolerance):
    """Check shape/type as well as pixels, and retain exactness in the report."""
    import numpy as np
    import torch

    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy()
    if isinstance(expected, torch.Tensor):
        expected = expected.cpu().numpy()
    if actual.shape != expected.shape or actual.dtype != np.uint8 or expected.dtype != np.uint8:
        raise AssertionError(f"{label}: output shape/dtype differ")
    difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    maximum = int(difference.max())
    different = int(np.count_nonzero(difference))
    if maximum > tolerance:
        raise AssertionError(f"{label}: max uint8 difference {maximum} exceeds {tolerance}")
    return dict(label=label, bitexact=different == 0, different_values=different, max_abs=maximum)


def make_fixture(device, batch, align=False):
    """Deterministic mixed source/ROI sizes, overlaps and shared/per-B inputs."""
    import torch
    from benchmark_paste_gpu_batch import identity_grid

    def values(shape, offset=0):
        count = 1
        for length in shape:
            count *= length
        return ((torch.arange(count, device=device).reshape(shape) + offset) % 251).float()

    frames = values((batch, HEIGHT, WIDTH, 3), 17).permute(0, 3, 1, 2)
    ads, grids, opacity = [], [], []
    for slot, ((source_h, source_w), box) in enumerate(zip(SOURCE_SIZES, BOXES)):
        ad = values((batch, source_h, source_w, 4), slot * 31).permute(0, 3, 1, 2)
        ad[:, 3, :2] = 0
        if slot == 3:
            ad[:, 3] = 0  # A later fully transparent slot must erase earlier slots.
        ads.append(ad)
        x1, y1, x2, y2 = box
        roi_h, roi_w = y2 - y1, x2 - x1
        members = batch if slot % 2 else 1
        grid = identity_grid(roi_h, roi_w, device, align).expand(members, -1, -1, -1).clone()
        grid.mul_(0.93).add_(0.017 * slot)
        grid[:, 0, :2] = 1.3  # Exercise source-edge padding independently of ROI padding.
        if members > 1:
            grid.add_(torch.arange(members, device=device).view(-1, 1, 1, 1) * 0.007)
        grids.append(grid)
        shape = (batch, 1, roi_h, roi_w) if slot == 2 else (members, 1, 1, 1)
        opacity.append(values(shape, 83 + slot) / 251)
    distortion = identity_grid(HEIGHT, WIDTH, device, align)
    mask = values((1, 1, HEIGHT, WIDTH), 13) / 1000
    factor = torch.tensor([0.25, 0.5, 0.25], device=device)
    kernel = torch.outer(factor, factor)[None, None].expand(4, 1, 3, 3)
    args = (frames, tuple(ads), tuple(grids), distortion, kernel, mask, tuple(opacity))
    return args, (factor, factor)


def check_layouts():
    """Independent CPU ownership oracle, including original slot overwrite order."""
    import numpy as np
    from paste_gpu_batch import _plan_buckets

    layouts = [
        (SOURCE_SIZES, BOXES),
        (SOURCE_SIZES, ((2, 1, 14, 10), *BOXES[1:])),
        (SOURCE_SIZES, ((1, 1, 19, 7), *BOXES[1:])),
        (SOURCE_SIZES[::-1], BOXES[::-1]),
    ]
    count = 0
    for source_sizes, boxes in layouts:
        for caps in (None, AREA_CAPS):
            plan = _plan_buckets(source_sizes, boxes, (HEIGHT, WIDTH), caps)
            capacity = 1 + sum(
                len(group) * length for group, length in zip(plan["groups"], plan["lengths"])
            )
            palette = np.zeros(capacity, dtype=np.int64)
            expected = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
            for slot, (x1, y1, x2, y2) in enumerate(boxes):
                area = (x2 - x1) * (y2 - y1)
                patch = (slot + 1) * 1000 + np.arange(area)
                offset = plan["offsets"][slot]
                palette[offset : offset + area] = patch
                expected[y1:y2, x1:x2] = patch.reshape(y2 - y1, x2 - x1)
            actual = palette[np.asarray(plan["lookup"])].reshape(HEIGHT, WIDTH)
            np.testing.assert_array_equal(actual, expected)
            for members in plan["groups"]:
                assert len({source_sizes[slot] for slot in members}) == 1
                if caps is not None:
                    bins = {
                        next(
                            cap
                            for cap in caps
                            if cap
                            >= (boxes[slot][2] - boxes[slot][0]) * (boxes[slot][3] - boxes[slot][1])
                        )
                        for slot in members
                    }
                    assert len(bins) == 1
            count += 1
    return count


def render_bucket(args, boxes, options, area_caps):
    from paste_gpu_batch import (
        make_bucket_layout,
        pack_bucket_ads,
        pack_bucket_grids,
        pack_bucket_transparencies,
        paste_gpu_bucketed,
    )

    frames, ads, grids, distortion, kernel, mask, opacity = args
    layout = make_bucket_layout(ads, boxes, frames.shape[-2:], area_caps=area_caps)
    batch = ads[0].shape[0]
    return paste_gpu_bucketed(
        frames,
        pack_bucket_ads(ads, layout),
        pack_bucket_grids(grids, layout, batch),
        distortion,
        kernel,
        mask,
        pack_bucket_transparencies(opacity, layout, batch),
        layout["canvas_lut"],
        **options,
    )


def check_core(device, tolerance):
    from benchmark_paste_gpu_batch import reference_paste
    from paste_gpu_batch import paste_gpu_batch

    cases = [
        (1, False, "zeros", True),
        (3, False, "zeros", True),
        (3, True, "border", True),
        (1, True, "zeros", False),
        (3, False, "border", False),
    ]
    rows = []
    for index, (batch, align, padding, legacy) in enumerate(cases):
        args, factors = make_fixture(device, batch, align)
        semantic_options = dict(
            warp_align_corners=align,
            distortion_align_corners=align,
            warp_padding_mode=padding,
            legacy_alpha=legacy,
        )
        reference = reference_paste(*args, bboxes=BOXES, **semantic_options)
        routes = [("eager", "torch")]
        if device.type == "cuda":
            routes += [("triton", "triton"), ("triton", "separable")]
        for acceleration, blur in routes:
            options = dict(semantic_options, acceleration=acceleration, blur_backend=blur)
            if blur == "separable":
                options["blur_factors"] = factors
            label = f"case={index},B={batch},align={align},padding={padding},legacy={legacy},{blur}"
            limit = 0 if acceleration == "eager" else tolerance
            batch_output = paste_gpu_batch(*args, bboxes=BOXES, **options)
            bucket_output = render_bucket(args, BOXES, options, None if index < 2 else AREA_CAPS)
            rows.append(compare(batch_output, reference, label + ":batch/reference", limit))
            rows.append(compare(bucket_output, reference, label + ":bucketed/reference", limit))
            rows.append(compare(bucket_output, batch_output, label + ":bucketed/batch", limit))
    return rows


def check_transparent_overwrite(device):
    """A transparent later ROI restores background, rather than keeping the old ad."""
    import torch
    from benchmark_paste_gpu_batch import identity_grid
    from paste_gpu_batch import paste_gpu_batch

    frames = torch.full((1, 3, HEIGHT, WIDTH), 37.0, device=device)
    first = torch.full((1, 4, 3, 5), 210.0, device=device)
    first[:, 3] = 255
    second = first.clone()
    second[:, 3] = 0
    boxes = ((1, 1, 12, 11), (4, 3, 14, 12))
    grids = tuple(torch.zeros((1, y2 - y1, x2 - x1, 2), device=device) for x1, y1, x2, y2 in boxes)
    args = (
        frames,
        (first, second),
        grids,
        identity_grid(HEIGHT, WIDTH, device),
        torch.ones((4, 1, 1, 1), device=device),
        torch.zeros_like(frames[:, :1]),
        (torch.ones((1, 1, 1, 1), device=device),) * 2,
    )
    rows = []
    routes = [("eager", "torch")]
    if device.type == "cuda":
        routes.append(("triton", "triton"))
    for legacy in (True, False):
        for acceleration, blur in routes:
            options = dict(legacy_alpha=legacy, acceleration=acceleration, blur_backend=blur)
            batch = paste_gpu_batch(*args, bboxes=boxes, **options)
            bucket = render_bucket(args, boxes, options, None)
            # Interior pixels avoid numerical edge effects in the identity grid.
            for output in (batch, bucket):
                assert bool((output[0, 5:9, 6:10] == 37).all())
                assert bool((output[0, 3:8, 2:3] >= 209).all())
            rows.append(compare(bucket, batch, f"transparent overwrite,legacy={legacy},{blur}", 0))
    return rows


def check_adapter_mutations(device, tolerance):
    """Reuse live adapters across geometry, opacity and asset-cache transitions."""
    import numpy as np
    import torch
    from benchmark_paste_gpu_batch import adapter_options, identity_grid, make_adapter_case
    from paste_gpu_batch_adapter import PasteBatchAdapter

    args, _ = make_fixture(device, 3)
    call = list(make_adapter_case(args, BOXES))
    owner, _, data, _, _, paste_args = call
    options = replace(
        adapter_options("triton" if device.type == "cuda" else "eager", max_cached_sizes=4),
        pinned_output=device.type == "cuda",
        batch_chunk_size=0,
        area_caps=AREA_CAPS,
    )
    renderers = {
        backend: PasteBatchAdapter(replace(options, backend=backend))
        for backend in ("batch", "bucketed")
    }
    retained = {backend: [] for backend in renderers}
    rows = []

    def render_step(label, *, must_change=False):
        outputs = {}
        for backend, renderer in renderers.items():
            actual = renderer(*call)
            assert actual.shape == (3, HEIGHT, WIDTH, 3) and actual.flags.c_contiguous
            fresh = PasteBatchAdapter(replace(options, backend=backend))(*call)
            rows.append(compare(actual, fresh, f"{label}:{backend}/fresh", 0))
            if must_change:
                assert not np.array_equal(actual, retained[backend][-1][1]), (
                    f"{label}: ineffective fixture mutation"
                )
            for prior, snapshot in retained[backend]:
                np.testing.assert_array_equal(prior, snapshot)
                assert not np.shares_memory(actual, prior), "output storage reused"
            retained[backend].append((actual, actual.copy()))
            outputs[backend] = actual
        rows.append(
            compare(
                outputs["bucketed"],
                outputs["batch"],
                f"{label}:bucketed/batch",
                tolerance if device.type == "cuda" else 0,
            )
        )

    render_step("initial")
    packed = renderers["bucketed"]._bucket_ads
    render_step("same inputs")
    assert renderers["bucketed"]._bucket_ads is packed
    paste_args[-1][0].add_(0.11)
    render_step("grid updated in place, same frame_id", must_change=True)
    assert renderers["bucketed"]._bucket_ads is packed
    data["ad_params"][2]["transparency"].mul_(0.25)
    render_step("opacity updated in place", must_change=True)
    prior_layout = renderers["bucketed"]._layout
    data["ad_params"][0]["bbox"] = (2, 1, 14, 10)
    render_step("bbox moved, same ROI size", must_change=True)
    assert renderers["bucketed"]._layout is not prior_layout
    data["ad_params"][0]["bbox"] = (1, 1, 19, 7)
    paste_args[-1][0] = identity_grid(6, 18, device)
    render_step("ROI reshaped, same area", must_change=True)
    prior_layout = renderers["bucketed"]._layout
    packed = renderers["bucketed"]._bucket_ads
    for group in owner.imgs_raw_groups:
        group["ad0"]["img"] = torch.full((7, 9, 4), 149.0, device=device)
    render_step("new source object, same size and revision", must_change=True)
    assert renderers["bucketed"]._layout is prior_layout
    assert renderers["bucketed"]._bucket_ads is not packed
    for group in owner.imgs_raw_groups:
        group["ad0"]["img"] = torch.full((9, 8, 4), 173.0, device=device)
    data["ad_params"][0]["ad_h_w"] = (9, 8)
    render_step("new source object and size, unchanged revision", must_change=True)
    assert renderers["bucketed"]._layout is not prior_layout
    owner.imgs_raw_groups[0]["ad0"]["img"][..., :3].add_(23)
    owner.imgs_raw_version += 1
    render_step("source changed in place, revision incremented", must_change=True)
    data["ad_params"].reverse()
    render_step("slot order reversed", must_change=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-uint8-diff", type=int, choices=(0, 1), default=1)
    parser.add_argument("--output", type=Path)
    cli = parser.parse_args()
    import torch

    device = torch.device(cli.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("device must be cpu or cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA requested but unavailable; GPU checks were not run")
        torch.cuda.set_device(device)
    else:
        torch.set_num_threads(1)
    with torch.inference_mode():
        report = dict(
            scope="synthetic correctness only; no performance measurements",
            device=str(device),
            accelerated_uint8_tolerance=cli.max_uint8_diff,
            cpu_layout_checks=check_layouts(),
            comparisons=check_core(device, cli.max_uint8_diff),
        )
        report["comparisons"].extend(check_transparent_overwrite(device))
        report["comparisons"].extend(check_adapter_mutations(device, cli.max_uint8_diff))
    report["all_bitexact"] = all(row["bitexact"] for row in report["comparisons"])
    report["gpu_checks_run"] = device.type == "cuda"
    if cli.output:
        from benchmark_support import save_report

        save_report(report, cli.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
