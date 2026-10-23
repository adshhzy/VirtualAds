"""Bounded-workspace checks on the real B8/S16 fixture and dynamic routes.

This is a correctness/lifetime diagnostic, not a speed benchmark. Independent
old-source oracles run periodically, outside memory samples. Source pixels and
camera callbacks have the same limitations as benchmark_render_img_modes.py.
"""

from __future__ import annotations

from benchmarks.source_layout import REPOSITORY_ROOT, runtime_sources, source_key

import argparse
from collections import deque
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from benchmarks import benchmark_render_img_modes as fixture
from tools.check_streaming_stability import load_baseline, memory_sample, verify_baseline_sources
from examples.live_render_demo import SyntheticOwner


def workspace_state(worker):
    return {
        "frame": worker._frame_upload.cache_info(),
        "matrices": worker.owner._shared_paste_builder.cache_info()["warp_upload"],
        "opacity": worker.executor.cache_info()["opacity_upload"],
    }


def assert_released(state):
    assert state["frame"]["float32_bytes"] == 0
    for item in (state["frame"]["uint8_upload"], state["matrices"], state["opacity"]):
        assert item["host_bytes"] == item["device_bytes"] == 0, state


def allocation_state(state):
    return (
        state["frame"]["uint8_upload"]["reallocations"],
        state["frame"]["float32_allocations"],
        state["matrices"]["reallocations"],
        state["opacity"]["reallocations"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sample", "frame-bmp", "ad-bmp", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--frames", type=int, default=10000)
    parser.add_argument("--check-every", type=int, default=250)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if (
        not __debug__
        or args.frames < 2
        or args.check_every < 1
        or torch.device(args.device).type != "cuda"
    ):
        parser.error("requires CUDA, frames >=2, positive check-every and enabled assertions")
    if args.output.exists():
        parser.error("refusing to overwrite output")
    load_baseline(args.baseline)
    old = importlib.import_module("_streaming_baseline.paste_renderer")
    worker = fixture.make_renderer("dense_auto", args.device)
    oracle = old.ImageRenderer(
        SyntheticOwner(kernel_size=5),
        old.RenderOptions(frame_layout="HWC", ad_layout="HWC", resize_ads_to_ad_h_w=True),
        device=args.device,
        use_separable=False,
        history_capacity=9,
    )
    sample = fixture.expanded_fixture(fixture.load_sample(args.sample))
    frame = fixture.read_bmp(args.frame_bmp)
    image = fixture.read_bmp(args.ad_bmp)
    rgba = np.concatenate((image, np.full((*image.shape[:2], 1), 255, np.uint8)), axis=-1)
    assets = {
        (item["ad_placement_id"], item["ad_id"]): rgba
        for route in range(1, 9)
        for item in sample[f"ad_params_{route}"]
    }
    inputs = fixture.Inputs(sample, ((0, 0),))
    held = deque(maxlen=2)
    report = {
        "frames": args.frames,
        "arguments": vars(args),
        "current_version": fixture.RENDERER_VERSION,
        "baseline_version": old.RENDERER_VERSION,
        "runtime_sha256": {
            source_key(path, REPOSITORY_ROOT): fixture.file_hash(path)
            for path in runtime_sources(REPOSITORY_ROOT)
        },
        "baseline_sha256": {
            source_key(path, args.baseline): fixture.file_hash(path)
            for path in runtime_sources(args.baseline)
        },
        "input_sha256": {
            name: fixture.file_hash(getattr(args, name))
            for name in ("sample", "frame_bmp", "ad_bmp")
        },
        "checks": [],
        "memory": [],
        "held_output_checks": 0,
        "notes": [
            "Only listed oracle frames are pixel checked.",
            "Two returned outputs and CPU snapshots are deliberately retained.",
            "No per-frame gc.collect/empty_cache. Reserved memory is not a leak test.",
            "SyntheticOwner stationary sigma 0.65; no production motion callback.",
        ],
    }

    def check_held(output=None):
        for previous, snapshot in held:
            assert np.array_equal(previous, snapshot), "previous output overwritten"
            if output is not None:
                assert not np.shares_memory(previous, output)
            report["held_output_checks"] += 1

    def reference_check(data, pixels, output, label):
        # Feed the same mature history without retaining current source dictionaries.
        oracle.history.clear()
        for frame_id in worker.history:
            oracle.history[frame_id] = deepcopy(worker.history[frame_id])
        expected = oracle.render_img(deepcopy(data), pixels, assets)
        assert np.array_equal(output, expected), label
        assert worker.route_ids == oracle.route_ids
        assert worker.last_execution == oracle.last_execution
        check_held(output)
        report["checks"].append({"case": label, "routes": worker.route_ids, "max_uint8_diff": 0})
        del expected
        oracle.release_resources()

    def sample_memory(label):
        state = memory_sample(
            SimpleNamespace(adapter=worker.executor, builder=worker.owner._shared_paste_builder),
            torch.device(args.device),
            sum(output.nbytes for output, _ in held),
        )
        state.update(
            label=label,
            workspace=workspace_state(worker),
            history=len(worker.history),
            retained_snapshot_bytes=sum(snapshot.nbytes for _, snapshot in held),
        )
        assert state["history"] <= 9
        for item in (state["workspace"]["matrices"], state["workspace"]["opacity"]):
            assert item["capacity_bytes"] <= item["max_cached_bytes"]
        report["memory"].append(state)

    def dynamic_data(step):
        batch = (1, 2, 3, 4, 8, 16)[step % 6]
        slots = (1, 3, 7, 16)[(step // 6) % 4]
        data = inputs.update(0)
        result = {
            key: value for key, value in data.items() if not fixture.ROUTE_KEY.fullmatch(str(key))
        }
        for route in range(1, batch + 1):
            params = deepcopy(data[f"ad_params_{(route - 1) % 8 + 1}"][:slots])
            for slot, param in enumerate(params):
                x1, y1, x2, y2 = param["bbox"]
                dx, dy = -(step % 6), step % 3
                param["bbox"] = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
                # Live per-route values differ, even with shared layout/materials.
                param["m_inv"][0, 2] += 0.03 * (route % 3)
                param["transparency"] = 0.4 + ((route + slot + step) % 5) * 0.1
            result[f"ad_params_{route}"] = params
        return result

    try:
        for _ in range(100):
            output = worker.render_img(inputs.update(0), frame, assets)
            del output
        held.append((worker.render_img(inputs.update(0), frame, assets), None))
        held[-1] = (held[-1][0], held[-1][0].copy())
        allocations = allocation_state(workspace_state(worker))
        sample_memory("fixed_warm")
        half = args.frames // 2
        for step in range(half):
            data = inputs.update(0)
            frame[:8, :8] = step % 256
            if step and step % 500 == 0:
                rgba[:8, :8, :3] = (step // 500) % 256
                data["imgs_raw_version"] += 1
            output = worker.render_img(data, frame, assets)
            if step % args.check_every == 0 or step == half - 1:
                reference_check(data, frame, output, f"fixed-{step}")
                assert allocation_state(workspace_state(worker)) == allocations
                held.append((output, output.copy()))
            del output
            if step % args.check_every == 0 or step == half - 1:
                sample_memory(f"fixed-{step}")
                print(f"fixed {step + 1}/{half}", flush=True)
        # Warm every B/S combination to establish maximum retained capacities.
        for step in range(24):
            output = worker.render_img(dynamic_data(step), frame, assets)
            del output
        allocations = allocation_state(workspace_state(worker))
        sample_memory("dynamic_warm")
        for step in range(args.frames - half):
            frame[:8, :8] = (step + 17) % 256
            if step and step % 500 == 0:
                rgba[:8, :8, :3] = (step // 500 + 37) % 256
                inputs.data["imgs_raw_version"] += 1
            data = dynamic_data(step)
            output = worker.render_img(data, frame, assets)
            if step < 24 or step % args.check_every == 0 or step == args.frames - half - 1:
                reference_check(data, frame, output, f"dynamic-{step}")
                assert allocation_state(workspace_state(worker)) == allocations
                held.append((output, output.copy()))
            del output
            if step % args.check_every == 0 or step == args.frames - half - 1:
                sample_memory(f"dynamic-{step}")
                print(f"dynamic {step + 1}/{args.frames - half}", flush=True)
        # Pause/resume, same-frame in-place inputs, material revision and failure recovery.
        data = inputs.update(0)
        for key in tuple(data):
            if fixture.ROUTE_KEY.fullmatch(str(key)):
                del data[key]
        output = worker.render_img(data, frame, assets)
        assert output.shape[0] == 0
        assert_released(workspace_state(worker))
        del output
        inputs = fixture.Inputs(sample, ((0, 0),))
        inputs.frame_id = int(data["frame_id"]) + 1
        data = inputs.update(0)
        for change in range(4):
            if change:
                data["ad_params_1"][0]["m_inv"][0, 2] += 0.5
                data["ad_params_1"][0]["transparency"] = 0.3 + change / 10
                frame[0:10, 0:10] = 20 * change
            if change == 2:
                rgba[0:50, 0:50, :3] = 17
                data["imgs_raw_version"] += 1
            output = worker.render_img(data, frame, assets)
            reference_check(data, frame, output, f"resume-inplace-{change}")
            del output
        original_box = data["ad_params_1"][0]["bbox"]
        data["ad_params_1"][0]["bbox"] = (-1, 0, 10, 10)
        try:
            worker.render_img(data, frame, assets)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid bbox accepted")
        data["ad_params_1"][0]["bbox"] = original_box
        output = worker.render_img(data, frame, assets)
        reference_check(data, frame, output, "after-validation-error")
        del output
        # The background has been uploaded before mask validation fails. Reuse
        # must not overwrite an in-flight pinned source or return stale pixels.
        original_masks = data["mask_coor_list"]
        data["mask_coor_list"] = [{"bbox": (0, 0, 2, 2), "mask": np.zeros((1, 2), np.float32)}]
        try:
            worker.render_img(data, frame, assets)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid mask accepted")
        data["mask_coor_list"] = original_masks
        frame[0:10, 0:10] = 140
        output = worker.render_img(data, frame, assets)
        reference_check(data, frame, output, "after-upload-then-mask-error")
        del output
        check_held()
        worker.release_resources()
        assert_released(workspace_state(worker))
        check_held()
        output = worker.render_img(data, frame, assets)
        reference_check(data, frame, output, "after-explicit-release")
        del output
        sample_memory("before_close")
        verify_baseline_sources(args.baseline.resolve())
    finally:
        worker.close()
        oracle.close()
    assert_released(workspace_state(worker))
    check_held()
    held.clear()
    sample_memory("after_close_outputs_released")
    report["all_oracle_checks_exact"] = True
    report["fixed_and_dynamic_workspace_allocations_stable_after_warmup"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture.json_ready(report), indent=2), encoding="utf8")
    print(f"passed {len(report['checks'])} oracle frames; {args.output}", flush=True)


if __name__ == "__main__":
    main()
