"""Live worker integration and an executable synthetic lifecycle demonstration.

Import run_live_worker for business integration. Running this file uses explicitly
synthetic callbacks and a local checking sink, not a decoder, encoder or network.
See LIVE_RENDER_EXAMPLE.md for the packet and resource ownership contracts.
"""

from contextlib import ExitStack
from dataclasses import replace

import numpy as np

if __package__:
    from .paste_renderer import (
        BoundedCameraHistory,
        initialize_renderer,
        release_renderer,
        render_frame,
    )
else:
    from paste_renderer import (
        BoundedCameraHistory,
        initialize_renderer,
        release_renderer,
        render_frame,
    )


def _clear_worker_inputs(owner, history):
    """Drop only this worker's history and current group references."""
    if history is not None:
        history.clear()
    owner.paste_data_stack = None
    # None would enable the legacy owner.imgs_raw fallback; an empty list will not.
    owner.imgs_raw_groups = []


def run_live_worker(
    owner,
    existing_options,
    frame_source,
    sink,
    *,
    device,
    use_separable=False,
    history_capacity=9,
):
    """Own a source/sink lifetime and render serially on one GPU compute stream.

    owner must be dedicated to this worker and already implement the real camera,
    motion-to-sigma and Gaussian callbacks. This function installs its own bounded
    camera history. It does not own unrelated owner fields or external asset banks.

    Source packets contain frame_id, pts, frame, camera, data, mask, route_ids,
    material_groups and material_revision. route_ids is an immutable tuple of
    distinct nonempty strings, in the same order as the material_groups list.
    Camera K/w2c/dist_coeffs are CPU arrays; their small snapshots outlive a frame.

    sink.set_routes(tuple) reconciles output routes without rebuilding unchanged
    encoders. sink.write_batch(route_ids, numpy_output, pts=...) finishes this
    frame's consumption synchronously. Any deliberately retained outputs must be
    bounded and released by close(). Source and sink both implement close();
    source close stops producers, and sink close drains/cancels pending consumers.
    ExitStack attempts every cleanup even when initialization or cleanup raises.
    """
    stats = dict.fromkeys(
        (
            "received_frames",
            "rendered_frames",
            "paused_frames",
            "idle_releases",
            "camera_resets",
            "asset_invalidations",
            "route_changes",
            "output_images",
        ),
        0,
    )
    history = None
    with ExitStack() as cleanup:
        # Reverse execution order: source -> sink -> renderer -> worker inputs.
        cleanup.callback(lambda: _clear_worker_inputs(owner, history))
        cleanup.callback(release_renderer, owner)
        cleanup.callback(sink.close)
        cleanup.callback(frame_source.close)

        history = BoundedCameraHistory(capacity=history_capacity)
        executor = initialize_renderer(
            owner,
            replace(existing_options, batch_chunk_size="auto"),
            use_separable=use_separable,
            normalization="legacy",
            camera_index_policy="legacy",
            max_cached_sizes=existing_options.max_cached_sizes,
        )
        owner.paste_data_stack = history
        owner.imgs_raw_groups = []
        active_routes = ()
        has_rendered = False

        for packet in frame_source:
            stats["received_frames"] += 1
            if packet.get("reset_camera_history", False):
                history.reset()
                release_renderer(owner)
                owner.imgs_raw_groups = []
                has_rendered = False
                stats["camera_resets"] += 1

            frame_id = packet["frame_id"]
            camera = packet["camera"]
            record = {
                name: np.array(camera[name], copy=True) for name in ("K", "w2c", "dist_coeffs")
            }
            # Even B=0 frames keep the current camera record for later recovery.
            history[frame_id] = record
            groups, route_ids = packet["material_groups"], packet["route_ids"]
            if not isinstance(groups, (list, tuple)) or not isinstance(route_ids, tuple):
                raise TypeError("material_groups must be a list/tuple and route_ids a tuple")
            if len(groups) != len(route_ids):
                raise ValueError("route_ids and material_groups must have the same length")
            if route_ids != active_routes:
                if any(not isinstance(route, str) or not route for route in route_ids):
                    raise ValueError("route IDs must be nonempty strings")
                if len(set(route_ids)) != len(route_ids):
                    raise ValueError("route IDs must be unique")
                sink.set_routes(route_ids)
                active_routes = route_ids
                stats["route_changes"] += 1

            if not groups:
                # Release once on the transition into idle, never once per idle frame.
                if has_rendered:
                    release_renderer(owner)
                    owner.imgs_raw_groups = []
                    has_rendered = False
                    stats["idle_releases"] += 1
                stats["paused_frames"] += 1
                continue

            if packet.get("drop_asset_cache", False):
                executor.invalidate_assets()
                stats["asset_invalidations"] += 1
            # A content revision, not the frame counter. Static materials keep it.
            owner.imgs_raw_version = packet["material_revision"]
            data = dict(packet["data"])
            data["K"] = record["K"]
            data["dist_coeffs"] = record["dist_coeffs"].copy()
            output = render_frame(
                owner,
                packet["frame"],
                data,
                frame_id,
                mask_canvas=packet["mask"],
                material_groups=groups,
                device=device,
                way_id=packet.get("way_id", 1),
            )
            has_rendered = True
            stats["rendered_frames"] += 1
            stats["output_images"] += len(groups)
            try:
                sink.write_batch(route_ids, output, pts=packet["pts"])
            finally:
                # Do not keep this extra reference in the worker's error traceback.
                # A sink's own traceback/queue can still retain its argument.
                del output
    return stats


def main():
    """Run a small local demo; leave production callback selection to the caller."""
    import argparse
    import json
    from pathlib import Path

    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames-per-phase", type=int, default=3)
    parser.add_argument(
        "--separable", action="store_true", help="use the verified synthetic kernel"
    )
    parser.add_argument("--output", type=Path, default=Path("results/live_render_demo.json"))
    parser.add_argument(
        "--snapshot-dir", type=Path, help="optional first-route PPM per active phase"
    )
    cli = parser.parse_args()
    if cli.frames_per_phase < 2:
        parser.error("frames-per-phase must be >=2 to demonstrate repeated idle frames and reuse")
    device = torch.device(cli.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("the public initializer requires CUDA/Triton; CPU lifecycle tests use mocks")
    if cli.output.exists():
        parser.error(f"output already exists: {cli.output}; choose a new report path")
    if cli.snapshot_dir is not None and cli.snapshot_dir.exists():
        if not cli.snapshot_dir.is_dir() or any(cli.snapshot_dir.iterdir()):
            parser.error("snapshot-dir must be new or empty; existing files are not overwritten")
    torch.cuda.set_device(device)

    if __package__:
        from .live_render_demo import run_demo
    else:
        from live_render_demo import run_demo

    report = run_demo(cli, run_live_worker)
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    with cli.output.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    print(
        json.dumps(
            {"status": "passed", "output": str(cli.output), "worker": report["worker"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
