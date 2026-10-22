"""Finite synthetic source/sink for the live worker example, not a video service.

The motion and Gaussian laws in SyntheticOwner are demonstration-only callbacks.
They must never replace a business owner's production methods. This module uses
the runtime API directly and does not import benchmark or test helpers. Optional
PPM files are local RGB snapshots; no RTMP, decoder, encoder or network is used.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import weakref

import numpy as np
import torch

if __package__:
    from .paste_renderer import RenderOptions
else:
    from paste_renderer import RenderOptions


HEIGHT, WIDTH = 96, 160
PHASES = (
    ("initial_B1", ("red",)),
    ("add_B3", ("red", "green", "blue")),
    ("reorder_replace_B2", ("blue", "red")),
    ("inplace_revision_B2", ("blue", "red")),
    ("paused_B0", ()),
    ("resume_B4", ("red", "green", "blue", "purple")),
    ("no_ads_B4", ("red", "green", "blue", "purple")),
    ("camera_reset_B1", ("purple",)),
)


class SyntheticOwner:
    """Explicit toy callbacks for a static synthetic camera and separable blur."""

    def __init__(self, kernel_size=5):
        if kernel_size not in (3, 5):
            raise ValueError("the synthetic Gaussian example supports kernel sizes 3 or 5")
        self.kernel_size = kernel_size
        self.gaussian_kernel_cache_version = 0

    def compute_camera_motion_rate(self, poses):
        positions = np.asarray([pose[:3, 3] for pose in poses], dtype=np.float32)
        return float(np.abs(np.diff(positions, axis=0)).mean()) if len(positions) > 1 else 0.0

    def motion_rate_to_sigma(self, motion_rate):
        return 0.65 + min(abs(float(motion_rate)), 1.0) * 0.1

    def create_gaussian_kernel(self, sigma):
        coordinates = np.arange(self.kernel_size, dtype=np.float32) - self.kernel_size // 2
        factor = np.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
        factor = (factor / factor.sum()).astype(np.float32)
        return np.outer(factor, factor).astype(np.float32)


def _material(color, shape, slot):
    """Build visibly different RGB advertisements with an opaque alpha channel."""
    height, width = shape
    image = np.empty((height, width, 4), dtype=np.float32)
    image[..., :3] = color
    image[..., 3] = 255
    image[3:-3:6, 3:-3, :3] = 245
    image[2:-2, width // 2 : width // 2 + 2, :3] = (25, 25, 25)
    if slot:
        image[height // 2 - 1 : height // 2 + 1, 2:-2, :3] = 245
    return image


def _group(color):
    return {
        (f"slot{slot}", slot): {"img": _material(color, shape, slot)}
        for slot, shape in enumerate(((24, 44), (20, 40)))
    }


def _ad_parameters():
    parameters = []
    for slot, (shape, box) in enumerate(
        zip(((24, 44), (20, 40)), ((10, 18, 74, 50), (82, 50, 146, 82)))
    ):
        height, width = shape
        x1, y1, x2, y2 = box
        matrix = np.eye(3, dtype=np.float32)
        matrix[0, 0] = (width - 1) / (x2 - x1 - 1)
        matrix[1, 1] = (height - 1) / (y2 - y1 - 1)
        parameters.append(
            dict(
                ad_placement_id=f"slot{slot}",
                ad_id=slot,
                bbox=box,
                ad_h_w=shape,
                m_inv=matrix,
                transparency=1.0,
            )
        )
    return parameters


class DemoSource:
    """Eight finite phases with independent camera records and monotonically increasing PTS.

    Image, calibration and geometry values stay stationary. Only the documented
    material edits, ordering, B, no-ad flag and camera reset change between phases.
    Post-frame observations are demonstration assertions, not worker hot-path code.
    """

    def __init__(self, owner, *, device, frames_per_phase=3):
        if type(frames_per_phase) is not int or frames_per_phase < 1:
            raise ValueError("frames_per_phase must be a positive Python integer")
        self.owner = owner
        self.frames_per_phase = frames_per_phase
        self.closed = False
        self._index = 0
        self._pending = None
        self._worker_refs = None
        self._history_ref = None
        self.current_phase = None
        self.revision = 0
        self.rows = [
            dict(phase=name, route_ids=list(routes), batch=len(routes), received=0, rendered=0)
            for name, routes in PHASES
        ]
        self.observations = dict(paused_frames_checked=0, worker_identity_checks=0)
        self.groups = {
            name: _group(color)
            for name, color in (
                ("red", (230, 45, 40)),
                ("green", (40, 210, 65)),
                ("blue", (45, 80, 235)),
                ("purple", (190, 50, 205)),
            )
        }
        ys, xs = np.indices((HEIGHT, WIDTH))
        background = np.stack((25 + xs // 5, 30 + ys // 3, 45 + (xs + ys) // 8), axis=-1)
        self.background = background.astype(np.uint8)
        self.frame = torch.as_tensor(self.background, device=device)
        self.mask = torch.zeros((HEIGHT, WIDTH, 1), dtype=torch.float32, device=device)
        self.parameters = _ad_parameters()

    def __iter__(self):
        return self

    def _observe_previous(self):
        if self._pending is None:
            return
        phase_index, prior_rendered = self._pending
        row = self.rows[phase_index]
        expected_calls = 1 if row["batch"] else 0
        assert row["rendered"] == prior_rendered + expected_calls, "unexpected sink call count"
        executor = self.owner._paste_batch_adapter
        builder = self.owner._shared_paste_builder
        if self._worker_refs is None:
            self._worker_refs = (weakref.ref(executor), weakref.ref(builder))
        assert self._worker_refs[0]() is executor and self._worker_refs[1]() is builder, (
            "worker objects were replaced during the stream"
        )
        self.observations["worker_identity_checks"] += 1
        history = self.owner.paste_data_stack
        if history is not None:
            if self._history_ref is None:
                self._history_ref = weakref.ref(history)
            assert len(history) <= history.capacity, "camera history exceeded its bound"
        if row["batch"] == 0:
            assert executor.cache_info()["retained_stack_bytes"] == 0, "paused assets retained"
            assert not executor._asset_source_refs, "paused source identities retained"
            assert builder.cache_info()["entries"] == 0, "paused Gaussian cache retained"
            assert not self.owner.imgs_raw_groups, "paused worker still references materials"
            self.observations["paused_frames_checked"] += 1
        elif row["phase"] != "no_ads_B4":
            assert executor._asset_key[0] == self.revision, "material revision was not applied"
        self._pending = None

    def __next__(self):
        if self.closed:
            raise StopIteration
        self._observe_previous()
        if self._index >= len(PHASES) * self.frames_per_phase:
            raise StopIteration
        phase_index, offset = divmod(self._index, self.frames_per_phase)
        name, routes = PHASES[phase_index]
        self.current_phase = phase_index
        if offset == 0 and name == "reorder_replace_B2":
            # Keep revision unchanged while replacing red's source objects.
            self.groups["red"] = _group((235, 190, 30))
        if offset == 0 and name == "inplace_revision_B2":
            # Keep object identities unchanged: only this explicit revision makes
            # cached blue pixels stale. Geometry/background remain exactly equal.
            for record in self.groups["blue"].values():
                record["img"][..., :3] = (25, 225, 220)
            self.revision += 1
        frame_id = offset if name == "camera_reset_B1" else self._index
        camera = {
            "K": np.array(((150, 0, WIDTH / 2), (0, 150, HEIGHT / 2), (0, 0, 1)), np.float32),
            "w2c": np.eye(4, dtype=np.float32),
            "dist_coeffs": np.zeros(5, dtype=np.float32),
        }
        row = self.rows[phase_index]
        row["received"] += 1
        row["material_revision"] = self.revision
        self._pending = (phase_index, row["rendered"])
        packet = dict(
            frame_id=frame_id,
            pts=self._index / 30.0,
            frame=self.frame,
            camera=camera,
            data={"ad_params": [] if name == "no_ads_B4" else self.parameters},
            mask=self.mask,
            route_ids=routes,
            material_groups=[self.groups[route] for route in routes],
            material_revision=self.revision,
            reset_camera_history=name == "camera_reset_B1" and offset == 0,
        )
        self._index += 1
        return packet

    def close(self):
        self.closed = True
        self.groups.clear()
        self.frame = self.mask = None
        self.parameters = []
        self._pending = None

    def history_was_released_or_cleared(self):
        history = self._history_ref() if self._history_ref is not None else None
        return history is None or len(history) == 0


def _digest(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


class CheckingSink:
    """Synchronously inspect results, retaining at most an anchor and previous output.

    Their ordinary NumPy snapshots detect overwrites; all held arrays are released
    by close(). PPM writes occur only for route zero of each phase's first batch.
    This sink demonstrates ownership and routing, not network delivery or timing.
    """

    def __init__(self, owner, source, *, snapshot_dir=None):
        self.owner, self.source = owner, source
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir is not None else None
        self.closed = False
        self.routes = None
        self.route_events = []
        self.snapshots = []
        self._last_pts = None
        self._anchor = self._anchor_snapshot = None
        self._previous = self._previous_snapshot = None
        self.checks = dict(output_batches=0, output_images=0, ownership_checks=0)

    def set_routes(self, route_ids):
        routes = tuple(route_ids)
        assert routes != self.routes, "set_routes repeated an unchanged route order"
        assert len(set(routes)) == len(routes), "duplicate routes"
        self.routes = routes
        self.route_events.append(list(routes))

    def _verify_retained(self):
        for output, snapshot in (
            (self._anchor, self._anchor_snapshot),
            (self._previous, self._previous_snapshot),
        ):
            if output is not None:
                np.testing.assert_array_equal(output, snapshot)
                self.checks["ownership_checks"] += 1

    def _verify_phase(self, row, digests):
        previous = self.source.rows
        name = row["phase"]
        if name == "add_B3":
            assert digests["red"] == previous[0]["route_sha256"]["red"]
        elif name == "reorder_replace_B2":
            assert digests["blue"] == previous[1]["route_sha256"]["blue"]
            assert digests["red"] != previous[1]["route_sha256"]["red"], "replacement was stale"
        elif name == "inplace_revision_B2":
            assert digests["red"] == previous[2]["route_sha256"]["red"]
            assert digests["blue"] != previous[2]["route_sha256"]["blue"], "revision was stale"
        elif name == "resume_B4":
            for route in ("red", "blue"):
                assert digests[route] == previous[3]["route_sha256"][route], "resume changed pixels"
        elif name == "camera_reset_B1":
            assert digests["purple"] == previous[5]["route_sha256"]["purple"]

    def write_batch(self, route_ids, output, *, pts):
        assert not self.closed
        routes = tuple(route_ids)
        assert routes and routes == self.routes, "output routing differs from configured routes"
        assert self._last_pts is None or pts > self._last_pts, "output PTS must increase"
        assert isinstance(output, np.ndarray) and output.dtype == np.uint8
        expected_shape = (HEIGHT, WIDTH, 3) if len(routes) == 1 else (len(routes), HEIGHT, WIDTH, 3)
        assert output.shape == expected_shape and output.flags.c_contiguous
        self._verify_retained()
        for retained in (self._anchor, self._previous):
            if retained is not None:
                assert not np.shares_memory(output, retained), "result storage was reused"
        batch = output[None] if len(routes) == 1 else output
        row = self.source.rows[self.source.current_phase]
        digests = {route: _digest(image) for route, image in zip(routes, batch)}
        if row["phase"] != "no_ads_B4":
            assert len(set(digests.values())) == len(routes), "material routes produced duplicates"
        if row["rendered"] == 0:
            self._verify_phase(row, digests)
            row["route_sha256"] = digests
            if self.snapshot_dir is not None:
                self.snapshot_dir.mkdir(parents=True, exist_ok=True)
                path = self.snapshot_dir / f"{self.source.current_phase + 1:02d}_{row['phase']}.ppm"
                with path.open("xb") as stream:
                    stream.write(f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii"))
                    stream.write(batch[0].tobytes())
                self.snapshots.append(str(path))
        else:
            assert digests == row["route_sha256"], "stationary phase produced changing pixels"
        if row["phase"] == "no_ads_B4":
            for image in batch:
                np.testing.assert_array_equal(image, self.source.background)
            assert self.owner._paste_batch_adapter.last_execution["reason"] == "no_ads"
        else:
            assert np.count_nonzero(batch[0] != self.source.background) > 1000, "ads not visible"
        if self._anchor is None:
            self._anchor, self._anchor_snapshot = output, output.copy()
        self._previous, self._previous_snapshot = output, output.copy()
        self._last_pts = pts
        row["rendered"] += 1
        self.checks["output_batches"] += 1
        self.checks["output_images"] += len(routes)

    def close(self):
        try:
            self._verify_retained()
        finally:
            self._anchor = self._anchor_snapshot = None
            self._previous = self._previous_snapshot = None
            self.closed = True


def run_demo(cli, run_live_worker):
    """Return a JSON-compatible demo report; the caller owns exclusive JSON creation."""
    owner = SyntheticOwner()
    source = DemoSource(owner, device=cli.device, frames_per_phase=cli.frames_per_phase)
    sink = CheckingSink(owner, source, snapshot_dir=cli.snapshot_dir)
    options = RenderOptions(
        frame_layout="HWC",
        ad_layout="HWC",
        mask_layout="HWC",
        transparency_layout="HW",
        max_cached_bytes=8 * 1024 * 1024,
        batch_chunk_size="auto",
    )
    worker = run_live_worker(
        owner,
        options,
        source,
        sink,
        device=cli.device,
        use_separable=cli.separable,
        history_capacity=9,
    )
    executor, builder = owner._paste_batch_adapter, owner._shared_paste_builder
    released = dict(
        source_closed=source.closed,
        sink_closed=sink.closed,
        retained_stack_bytes=executor.cache_info()["retained_stack_bytes"],
        source_references=len(executor._asset_source_refs),
        gaussian_entries=builder.cache_info()["entries"],
        kernel_context_references=len(builder._kernel_context_refs),
        history_empty=source.history_was_released_or_cleared(),
        owner_history_cleared=owner.paste_data_stack is None,
        owner_groups_empty=not owner.imgs_raw_groups,
        sink_outputs_released=sink._anchor is None and sink._previous is None,
    )
    assert released["source_closed"] and released["sink_closed"]
    assert all(
        released[key] == 0
        for key in (
            "retained_stack_bytes",
            "source_references",
            "gaussian_entries",
            "kernel_context_references",
        )
    )
    assert all(
        released[key]
        for key in (
            "history_empty",
            "owner_history_cleared",
            "owner_groups_empty",
            "sink_outputs_released",
        )
    )
    assert source.observations["paused_frames_checked"] == cli.frames_per_phase
    assert all(row["received"] == cli.frames_per_phase for row in source.rows)
    assert all(
        row["rendered"] == (cli.frames_per_phase if row["batch"] else 0) for row in source.rows
    )
    return dict(
        status="complete",
        scope="synthetic source and synchronous checking sink; local PPM snapshots, not RTMP",
        device=str(cli.device),
        separable=cli.separable,
        canvas=[HEIGHT, WIDTH],
        slots=2,
        frames_per_phase=cli.frames_per_phase,
        phases=source.rows,
        worker=worker,
        sink=sink.checks,
        route_changes=sink.route_events,
        observations=source.observations,
        after_close=released,
        snapshots=sink.snapshots,
        notes=[
            "Synthetic motion/Gaussian callbacks are demo-only, never production definitions.",
            "Background, calibration and geometry stay fixed; hashes isolate material changes.",
            "The sink retains at most two output arrays plus their ordinary NumPy snapshots.",
            "No timing or real video/network performance claim is made by this finite demo.",
        ],
    )
