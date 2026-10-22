"""Runnable three-argument render_img integration with a finite synthetic stream.

Run: python render_img_example.py --device cuda:0 --frames-per-phase 2
Optional: --output results/render_img_example.json (existing files are rejected).

SyntheticOwner supplies demonstration-only camera/motion/Gaussian callbacks. Use
the production owner's existing callbacks when integrating a real stream. This
example neither decodes video nor starts an encoder, HTTP service or network sink.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .live_render_demo import SyntheticOwner
    from .paste_renderer import ImageRenderer, RenderOptions
else:
    from live_render_demo import SyntheticOwner
    from paste_renderer import ImageRenderer, RenderOptions


HEIGHT, WIDTH = 96, 160
AD_SHAPES = ((24, 44), (20, 40))
AD_BOXES = ((10, 18, 74, 50), (82, 50, 146, 82))
PHASES = (
    ("initial_B1", (1,)),
    ("nonconsecutive_B3", (7, 1, 3)),
    ("replace_material_B3", (7, 1, 3)),
    ("inplace_revision_B3", (7, 1, 3)),
    ("paused_B0", ()),
    ("resume_B2", (7, 3)),
    ("no_ads_B2", (7, 3)),
    ("reset_history_B1", (1,)),
)


def _material(color, shape):
    """A uint8 RGB advertisement with an opaque fourth (alpha) channel."""
    image = np.empty((*shape, 4), dtype=np.uint8)
    image[..., :3] = color
    image[..., 3] = 255
    image[2:-2:5, 2:-2, :3] = 240
    return image


def _inputs():
    """Initialize camera, frame, material bank and per-route parameters once."""
    ys, xs = np.indices((HEIGHT, WIDTH))
    frame = np.stack((20 + xs // 4, 30 + ys // 3, 40 + (xs + ys) // 8), axis=-1)
    frame = frame.astype(np.uint8)
    mask = np.zeros((24, 24), dtype=np.uint8)
    data = {
        "K": np.array(((150, 0, WIDTH / 2), (0, 150, HEIGHT / 2), (0, 0, 1)), np.float32),
        "w2c": np.eye(4, dtype=np.float32),
        "dist_coeffs": np.zeros(5, dtype=np.float32),
        "mask_coor_list": [{"bbox": (42, 26, 66, 50), "mask": mask}],
        "imgs_raw_version": 0,
    }
    imgs_raw, route_params = {}, {}
    for route, color in ((1, (225, 45, 40)), (3, (40, 205, 65)), (7, (45, 80, 230))):
        parameters = []
        for slot, (shape, box) in enumerate(zip(AD_SHAPES, AD_BOXES)):
            key = (f"slot{slot}", route * 10 + slot)
            imgs_raw[key] = _material(color, shape)
            x1, y1, x2, y2 = box
            matrix = np.eye(3, dtype=np.float32)
            # m_inv maps ROI-local coordinates to material coordinates. The
            # translation is not the full-frame bbox origin.
            matrix[0, 0] = (shape[1] - 1) / (x2 - x1 - 1)
            matrix[1, 1] = (shape[0] - 1) / (y2 - y1 - 1)
            parameters.append(
                dict(
                    ad_placement_id=key[0],
                    ad_id=key[1],
                    bbox=box,
                    m_inv=matrix,
                    transparency=1.0,
                )
            )
        route_params[route] = parameters
    return data, frame, imgs_raw, route_params


def _update_frame(data, frame, route_params, step, frame_id):
    """Replace these synthetic updates with the current decoder/tracker results."""
    data["frame_id"] = frame_id
    data["w2c"][0, 3] = frame_id * 0.001
    frame[4:10, 4:20, 0] = 40 + step % 40
    shift = step % 3
    for parameters in route_params.values():
        for param, (x1, y1, x2, y2) in zip(parameters, AD_BOXES):
            param["bbox"] = (x1 + shift, y1, x2 + shift, y2)
            param["m_inv"][0, 2] = shift * 0.15
            param["transparency"] = 0.85 + (step % 2) * 0.15
    patch = data["mask_coor_list"][0]
    patch["bbox"] = (42 + shift, 26, 66 + shift, 50)
    patch["mask"].fill(0)
    patch["mask"][3:21, 6 + step % 2 : 16 + step % 2] = 1
    # Geometry, camera, mask and background updates do not need an asset revision.


def _closed_state(renderer):
    """Small end-of-demo ownership check, not a per-frame production probe."""
    owner = renderer.owner
    executor = renderer.executor.cache_info()
    shared = owner._shared_paste_builder.cache_info()
    mask = renderer.mask_builder.cache_info()
    frame = renderer._frame_upload.cache_info()
    uploads = (
        executor["opacity_upload"],
        shared["warp_upload"],
        mask["patch_upload"],
        frame["uint8_upload"],
    )
    return {
        "retained_asset_bytes": executor["retained_stack_bytes"],
        "gaussian_entries": shared["entries"],
        "mask_workspace_bytes": mask["workspace_bytes"] + mask["kernel_bytes"],
        "upload_host_bytes": sum(item["host_bytes"] for item in uploads),
        "upload_device_bytes": sum(item["device_bytes"] for item in uploads),
        "background_float32_bytes": frame["float32_bytes"],
        "route_workspace_bytes": renderer._route_preparation.cache_info()["retained_bytes"],
        "history_records": len(renderer.history),
        "owner_material_groups": len(owner.imgs_raw_groups),
        "owner_history_cleared": owner.paste_data_stack is None,
    }


def run_demo(device, frames_per_phase):
    """Exercise the same worker across route changes, edits, pause and recovery."""
    data, frame, imgs_raw, route_params = _inputs()
    owner = SyntheticOwner()  # Demonstration only; production supplies its own owner.
    options = RenderOptions(frame_layout="HWC", ad_layout="HWC", batch_chunk_size="auto")
    renderer = ImageRenderer(owner, options, device=device)
    render_img = renderer.render_img  # The requested three-argument online entry.
    rows = []
    held_output = held_snapshot = output = batch = None
    try:
        for phase_index, (phase, routes) in enumerate(PHASES):
            # Route membership changes only here; stale ad_params_N keys must be
            # removed. No keys means B=0; an existing key with [] means background.
            for key in tuple(data):
                if key.startswith("ad_params_"):
                    del data[key]
            for route in routes:
                data[f"ad_params_{route}"] = [] if phase == "no_ads_B2" else route_params[route]

            if phase == "replace_material_B3":
                # Replacing the array object is detected without changing revision.
                imgs_raw[("slot0", 10)] = _material((235, 190, 30), AD_SHAPES[0])
            elif phase == "inplace_revision_B3":
                # Same object, new pixels: bump one content version after all edits.
                imgs_raw[("slot0", 70)][..., :3] = (25, 220, 215)
                data["imgs_raw_version"] += 1
                # Alternatively call renderer.invalidate_assets() once after editing.
            elif phase == "reset_history_B1":
                # A new stream/seek starts a new camera timeline. Keep the worker.
                renderer.reset_history()

            row = {"phase": phase, "frames": 0, "routes": sorted(routes)}
            rows.append(row)
            for offset in range(frames_per_phase):
                step = phase_index * frames_per_phase + offset
                frame_id = offset if phase == "reset_history_B1" else step
                _update_frame(data, frame, route_params, step, frame_id)
                output = render_img(data, frame, imgs_raw)
                try:
                    # Initializer preserves the established B1=HWC contract;
                    # normalize only this local view for the synchronous consumer.
                    batch = output[None, ...] if output.ndim == 3 else output
                    assert renderer.route_ids == tuple(sorted(routes))
                    assert batch.shape == (len(routes), HEIGHT, WIDTH, 3)
                    assert output.dtype == np.uint8
                    if phase == "no_ads_B2":
                        assert all(np.array_equal(image, frame) for image in batch)
                    # In production, synchronously hand batch[index] to the route's
                    # encoder here; B=0 has no frames to send. This demo has no sink.
                    # An asynchronous consumer must own its result and bound/drain
                    # its queue. A single-route view can keep the whole batch alive.
                    if held_output is None and routes:
                        held_output, held_snapshot = output, output.copy()
                    row["frames"] += 1
                    row["output_shape"] = list(output.shape)
                    row["material_revision"] = data["imgs_raw_version"]
                finally:
                    # Release worker-local output references even if a sink fails.
                    output = batch = None
            print(json.dumps(row, ensure_ascii=False))
            # B0 releases working sets once on entry. Do not clear caches each
            # iteration; the same renderer automatically handles repeated pause.
    finally:
        try:
            renderer.close()
        finally:
            # In a real worker, first stop its source and finish/cancel downstream
            # consumption, then close the renderer and release business inputs.
            imgs_raw.clear()
            route_params.clear()
            data.clear()
            frame = output = batch = None

    released = _closed_state(renderer)
    assert released["owner_history_cleared"], "owner camera history is still attached"
    assert all(value == 0 for name, value in released.items() if name != "owner_history_cleared"), (
        "worker still owns cached resources"
    )
    assert np.array_equal(held_output, held_snapshot), "later frames/close overwrote an old result"
    held_output = held_snapshot = None
    return {
        "status": "passed",
        "scope": "finite synthetic render_img demonstration; no decoder/encoder/network",
        "device": str(device),
        "canvas": [HEIGHT, WIDTH],
        "total_calls": len(PHASES) * frames_per_phase,
        "phases": rows,
        "old_output_survived_close": True,
        "released": released,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames-per-phase", type=int, default=2)
    parser.add_argument("--output", type=Path, help="optional JSON report; never overwrite a file")
    args = parser.parse_args()
    if not __debug__:
        parser.error("run without -O: this finite demo uses assertions for its checks")
    if args.frames_per_phase < 2:
        parser.error("frames-per-phase must be >= 2 to demonstrate consecutive B=0 calls")
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}; choose a new report path")
    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("the public ImageRenderer initializer requires CUDA/Triton")
    torch.cuda.set_device(device)

    report = run_demo(device, args.frames_per_phase)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    print(json.dumps({"status": report["status"], "total_calls": report["total_calls"]}))


if __name__ == "__main__":
    main()
