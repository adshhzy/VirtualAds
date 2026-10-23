"""Production worker wiring and finite demo: python -m examples.render_img_worker.

Demo callbacks and images are synthetic; production supplies owner/source/sink.
"""

import argparse
import json

import numpy as np
import torch

from virtualads import ImageRenderer, RenderOptions
from examples.live_render_demo import SyntheticOwner
from examples.render_img_example import (
    PHASES,
    AD_SHAPES,
    _inputs,
    _material,
    _update_frame,
    _closed_state,
)


def run_live(owner, source, sink, imgs_raw, *, device="cuda:0", resize_ads_to_ad_h_w=False):
    """Own source/sink shutdown; caller retains ownership of the material bank.

    Source yields data_dict/frame/pts packets, optionally reset_camera=True.
    Sink consumes BHWC synchronously and provides set_routes/write_batch/close.
    Use one owner/renderer per serial camera worker. Mutate inputs only between calls.
    """
    renderer = None
    output = batch = None
    try:
        renderer = ImageRenderer(
            owner,
            RenderOptions(
                frame_layout="HWC",
                ad_layout="HWC",
                batch_chunk_size="auto",
                resize_ads_to_ad_h_w=resize_ads_to_ad_h_w,
            ),
            device=device,
            use_separable=False,
            mask_sigma=1.5,
            mask_radius=3,
            history_capacity=9,
        )
        active_routes = None
        for packet in source:
            if packet.get("reset_camera", False):
                renderer.reset_history()
            output = renderer.render_img(packet["data_dict"], packet["frame"], imgs_raw)
            routes = renderer.route_ids
            try:
                if routes != active_routes:
                    sink.set_routes(routes)
                    active_routes = routes
                if routes:
                    batch = output[None] if output.ndim == 3 else output
                    sink.write_batch(routes, batch, pts=packet["pts"])
            finally:
                output = batch = None
    finally:
        try:
            source.close()
        finally:
            try:
                sink.close()
            finally:
                if renderer is not None:
                    renderer.close()
    return renderer  # Closed; useful for end-of-worker diagnostics only.


class DemoSource:
    """Finite decoder/tracker substitute; material changes happen between calls."""

    def __init__(self):
        self.data, self.frame, self.assets, self.params = _inputs()
        self.closed = False
        self.expected_routes = ()
        self.phase = ""
        self.calls = 0

    def __iter__(self):
        for phase_index, (phase, routes) in enumerate(PHASES):
            self.phase = phase
            self.expected_routes = tuple(sorted(routes))
            for key in tuple(self.data):
                if key.startswith("ad_params_"):
                    del self.data[key]
            for route in routes:
                self.data[f"ad_params_{route}"] = [] if phase == "no_ads_B2" else self.params[route]
            if phase == "replace_material_B3":
                self.assets[("slot0", 10)] = _material((235, 190, 30), AD_SHAPES[0])
            elif phase == "inplace_revision_B3":
                self.assets[("slot0", 70)][..., :3] = (25, 220, 215)
                self.data["imgs_raw_version"] += 1
            for offset in range(2):
                step = phase_index * 2 + offset
                reset = phase == "reset_history_B1" and offset == 0
                frame_id = offset if phase == "reset_history_B1" else step
                _update_frame(self.data, self.frame, self.params, step, frame_id)
                self.calls += 1
                yield dict(data_dict=self.data, frame=self.frame, pts=step, reset_camera=reset)

    def close(self):
        self.closed = True
        self.data.clear()
        self.params.clear()
        self.frame = None
        # assets are caller-owned and are cleared after sink/renderer shutdown.


class DemoSink:
    """Synchronous local checker, not an encoder or network sender."""

    def __init__(self, source, fail=False):
        self.source = source
        self.fail = fail
        self.closed = False
        self.routes = ()
        self.transitions = []
        self.outputs = 0
        self.held = self.snapshot = None

    def set_routes(self, routes):
        assert routes == self.source.expected_routes
        self.routes = routes
        self.transitions.append(routes)

    def write_batch(self, routes, batch, *, pts):
        assert routes == self.routes == self.source.expected_routes
        assert batch.shape == (len(routes), 96, 160, 3)
        assert batch.dtype == np.uint8
        if self.source.phase == "no_ads_B2":
            assert all(np.array_equal(image, self.source.frame) for image in batch)
        if self.held is None:
            self.held, self.snapshot = batch, batch.copy()
        assert np.array_equal(self.held, self.snapshot)
        self.outputs += len(routes)
        if self.fail:
            raise RuntimeError("demonstration sink failure")

    def close(self):
        assert self.source.closed
        self.closed = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not __debug__:
        parser.error("demo checks require assertions; do not use -O")
    torch.cuda.set_device(args.device)
    source = DemoSource()
    sink = DemoSink(source)
    try:
        renderer = run_live(SyntheticOwner(), source, sink, source.assets, device=args.device)
        released = _closed_state(renderer)
        assert all(
            value == 0 for name, value in released.items() if name != "owner_history_cleared"
        )
        assert released["owner_history_cleared"]
        assert source.closed and sink.closed and source.calls == 16
        assert () in sink.transitions
        assert np.array_equal(sink.held, sink.snapshot)
        print(
            json.dumps(
                dict(
                    status="passed",
                    calls=source.calls,
                    route_images=sink.outputs,
                    phases=len(PHASES),
                    released=released,
                )
            )
        )
    finally:
        source.assets.clear()
        sink.held = sink.snapshot = None

    # A failed consumer must still close the source, sink and renderer resources.
    source = DemoSource()
    sink = DemoSink(source, fail=True)
    owner = SyntheticOwner()
    try:
        try:
            run_live(owner, source, sink, source.assets, device=args.device)
        except RuntimeError as error:
            assert str(error) == "demonstration sink failure"
        else:
            raise AssertionError("sink failure was not propagated")
        assert source.closed and sink.closed
        assert owner.paste_data_stack is None and owner.imgs_raw_groups == []
        assert owner._paste_batch_adapter.cache_info()["retained_stack_bytes"] == 0
        print(json.dumps(dict(exception_cleanup="passed")))
    finally:
        source.assets.clear()
        sink.held = sink.snapshot = None


if __name__ == "__main__":
    main()
