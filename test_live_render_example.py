"""CPU lifecycle contracts for live integration; rendering is deliberately mocked."""

from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

import live_render_example as live
from paste_renderer import RenderOptions


def packet(frame_id, batch=1, *, revision=0, routes=None, camera=None, **extra):
    if camera is None:
        camera = {
            "K": np.eye(3, dtype=np.float32),
            "w2c": np.eye(4, dtype=np.float32),
            "dist_coeffs": np.arange(5, dtype=np.float32),
        }
    return {
        "frame_id": frame_id,
        "pts": frame_id * 0.04,
        "frame": object(),
        "camera": camera,
        "data": {"ad_params": [{"business_slot": 1}]},
        "mask": object(),
        "route_ids": tuple(f"route-{i}" for i in range(batch)) if routes is None else routes,
        "material_groups": [{"business_material": object()} for _ in range(batch)],
        "material_revision": revision,
        **extra,
    }


class FakeSource:
    def __init__(self, harness, packets):
        self.harness, self.packets = harness, packets

    def __iter__(self):
        self.harness.fail("iterate")
        yield from self.packets

    def close(self):
        self.harness.events.append("source.close")
        self.harness.fail("source.close")


class FakeSink:
    def __init__(self, harness):
        self.harness = harness
        self.routes, self.writes, self.output_refs = [], [], []

    def set_routes(self, routes):
        self.harness.events.append("set_routes")
        self.harness.fail("set_routes")
        self.routes.append(routes)

    def write_batch(self, routes, output, *, pts):
        self.harness.events.append("write_batch")
        self.harness.fail("write_batch")
        self.writes.append((routes, output.shape, pts))
        self.output_refs.append(weakref.ref(output))

    def close(self):
        self.harness.events.append("sink.close")
        self.harness.fail("sink.close")


class Harness:
    """Record real loop/history behavior while replacing only the GPU boundary."""

    def __init__(self, packets, *, failures=()):
        self.events, self.renders = [], []
        self.failures = set(failures)
        self.owner = SimpleNamespace(
            paste_data_stack="old worker history",
            imgs_raw_groups=["old groups"],
            imgs_raw={"unrelated": "business fallback"},
            unrelated=object(),
            compute_camera_motion_rate=object(),
            motion_rate_to_sigma=object(),
            create_gaussian_kernel=object(),
        )
        self.executor = SimpleNamespace(invalidate_assets=self.invalidate)
        self.source, self.sink = FakeSource(self, packets), FakeSink(self)
        self.history = None
        self.final_history_records = None
        self.options = RenderOptions(max_cached_sizes=5, legacy_alpha=False, batch_chunk_size=0)

    def fail(self, stage):
        if stage in self.failures:
            raise RuntimeError(stage)

    def initialize(self, owner, options, **kwargs):
        self.events.append("initialize")
        self.fail("initialize")
        owner._paste_batch_adapter = self.executor
        return self.executor

    def invalidate(self):
        self.events.append("invalidate_assets")
        self.fail("invalidate_assets")

    def render(self, owner, frame, data, frame_id, **kwargs):
        self.events.append("render")
        self.fail("render")
        history = owner.paste_data_stack
        self.history = history
        owner.imgs_raw_groups = kwargs["material_groups"]  # Public API's documented assignment.
        self.renders.append(
            {
                "owner": owner,
                "executor": owner._paste_batch_adapter,
                "frame": frame,
                "data": data,
                "frame_id": frame_id,
                "kwargs": kwargs,
                "revision": owner.imgs_raw_version,
                "history": {
                    key: {name: array.copy() for name, array in record.items()}
                    for key, record in history.items()
                },
                "coefficients_share_history": np.shares_memory(
                    data["dist_coeffs"], history[frame_id]["dist_coeffs"]
                ),
            }
        )
        # The production camera filter may replace/update this field. Exercise
        # isolation from both source buffers and retained raw history explicitly.
        data["dist_coeffs"].fill(999)
        batch = len(kwargs["material_groups"])
        shape = (2, 3, 3) if batch == 1 else (batch, 2, 3, 3)
        return np.full(shape, frame_id % 256, dtype=np.uint8)

    def release(self, owner):
        self.events.append("release")
        self.fail("release")

    def clear(self, original, owner, history):
        self.events.append("clear")
        self.history = history
        self.final_history_records = None if history is None else dict(history)
        original(owner, history)

    def run(self, **kwargs):
        original_clear = live._clear_worker_inputs
        with ExitStack() as patches:
            self.initialize_mock = patches.enter_context(
                patch.object(live, "initialize_renderer", side_effect=self.initialize)
            )
            self.render_mock = patches.enter_context(
                patch.object(live, "render_frame", side_effect=self.render)
            )
            self.release_mock = patches.enter_context(
                patch.object(live, "release_renderer", side_effect=self.release)
            )
            patches.enter_context(
                patch.object(
                    live,
                    "_clear_worker_inputs",
                    side_effect=lambda owner, history: self.clear(original_clear, owner, history),
                )
            )
            return live.run_live_worker(
                self.owner,
                self.options,
                self.source,
                self.sink,
                device="cuda:0",
                **kwargs,
            )


class LiveWorkerLifecycleTests(unittest.TestCase):
    def assert_cleaned(self, harness):
        self.assertEqual(harness.events[-4:], ["source.close", "sink.close", "release", "clear"])
        self.assertIsNone(harness.owner.paste_data_stack)
        self.assertEqual(harness.owner.imgs_raw_groups, [])
        if harness.history is not None:
            self.assertEqual(len(harness.history), 0)
            self.assertIsNone(harness.history.latest_frame_id)

    def test_one_initialization_idle_transition_and_resume_reuse(self):
        packets = [packet(i, batch) for i, batch in enumerate((2, 2, 0, 0, 1, 1))]
        harness = Harness(packets)
        stats = harness.run(use_separable=True)
        self.assertEqual(
            stats,
            {
                "received_frames": 6,
                "rendered_frames": 4,
                "paused_frames": 2,
                "idle_releases": 1,
                "camera_resets": 0,
                "asset_invalidations": 0,
                "route_changes": 3,
                "output_images": 6,
            },
        )
        self.assertTrue(all(type(value) is int for value in stats.values()))
        harness.initialize_mock.assert_called_once_with(
            harness.owner,
            replace(harness.options, batch_chunk_size="auto"),
            use_separable=True,
            normalization="legacy",
            camera_index_policy="legacy",
            max_cached_sizes=5,
        )
        self.assertEqual(harness.render_mock.call_count, 4)
        self.assertEqual(harness.release_mock.call_count, 2)  # Idle transition plus final cleanup.
        self.assertTrue(all(call["executor"] is harness.executor for call in harness.renders))
        self.assertEqual(harness.sink.routes, [("route-0", "route-1"), (), ("route-0",)])
        self.assertTrue(all(reference() is None for reference in harness.sink.output_refs))
        self.assert_cleaned(harness)

    def test_initial_idle_frames_do_not_render_or_repeatedly_release(self):
        # Idle input deliberately omits frame/data/mask/pts/revision, all unused.
        packets = []
        for index in range(3):
            value = packet(index, 0)
            packets.append(
                {key: value[key] for key in ("frame_id", "camera", "route_ids", "material_groups")}
            )
        harness = Harness(packets)
        stats = harness.run()
        self.assertEqual(stats["paused_frames"], 3)
        self.assertEqual(stats["idle_releases"], 0)
        self.assertEqual(stats["route_changes"], 0)
        harness.render_mock.assert_not_called()
        self.assertEqual(harness.release_mock.call_count, 1)
        self.assertEqual(tuple(harness.final_history_records), (0, 1, 2))
        self.assert_cleaned(harness)

    def test_material_revision_and_explicit_invalidation_are_not_frame_counters(self):
        packets = [
            packet(10, revision=7),
            packet(11, revision=7),
            packet(12, revision=8, drop_asset_cache=True),
        ]
        for value in packets[1:]:
            value["material_groups"] = packets[0]["material_groups"]
        harness = Harness(packets)
        stats = harness.run()
        self.assertEqual([value["revision"] for value in harness.renders], [7, 7, 8])
        self.assertEqual(harness.events.count("invalidate_assets"), 1)
        self.assertEqual(stats["asset_invalidations"], 1)
        self.assertEqual(len(harness.sink.routes), 1)
        self.assertLess(harness.events.index("invalidate_assets"), len(harness.events) - 4)
        self.assert_cleaned(harness)

    def test_camera_snapshots_survive_reused_source_buffers_and_filter_writes(self):
        first = packet(0)
        camera = first["camera"]
        raw = {name: value.copy() for name, value in camera.items()}

        def source():
            yield first
            for value in camera.values():
                value.fill(17)
            yield packet(1, camera=camera)

        harness = Harness(source())
        harness.run()
        second_history = harness.renders[1]["history"]
        for name, expected in raw.items():
            np.testing.assert_array_equal(second_history[0][name], expected)
            np.testing.assert_array_equal(second_history[1][name], np.full_like(expected, 17))
        self.assertTrue(all(not value["coefficients_share_history"] for value in harness.renders))
        self.assertNotIn("K", first["data"])
        self.assertNotIn("dist_coeffs", first["data"])
        self.assert_cleaned(harness)

    def test_history_is_bounded_and_reset_allows_frame_ids_to_restart(self):
        harness = Harness([packet(9), packet(10), packet(11), packet(0, reset_camera_history=True)])
        stats = harness.run(history_capacity=2)
        self.assertEqual(tuple(harness.renders[2]["history"]), (10, 11))
        self.assertEqual(tuple(harness.renders[3]["history"]), (0,))
        self.assertEqual(stats["camera_resets"], 1)
        self.assertEqual(harness.release_mock.call_count, 2)
        self.assertEqual(len(harness.sink.routes), 1)
        self.assert_cleaned(harness)

    def test_frame_id_rewind_without_reset_is_rejected_and_cleans_up(self):
        harness = Harness([packet(10), packet(0)])
        with self.assertRaisesRegex(ValueError, "monotonic"):
            harness.run()
        self.assertEqual(harness.render_mock.call_count, 1)
        self.assert_cleaned(harness)

    def test_route_order_changes_reconcile_once_and_pts_passes_through(self):
        packets = [packet(0, 2), packet(1, 2), packet(2, 2, routes=("route-1", "route-0"))]
        packets[2]["pts"] = object()
        packets[2]["way_id"] = 7
        harness = Harness(packets)
        stats = harness.run()
        self.assertEqual(stats["route_changes"], 2)
        self.assertEqual(harness.sink.routes, [("route-0", "route-1"), ("route-1", "route-0")])
        self.assertIs(harness.sink.writes[-1][2], packets[2]["pts"])
        self.assertEqual(harness.renders[-1]["kwargs"]["way_id"], 7)
        self.assertIs(
            harness.renders[-1]["kwargs"]["material_groups"], packets[2]["material_groups"]
        )

    def test_invalid_route_metadata_is_rejected_before_render(self):
        bad_values = [
            {"material_groups": {}},
            {"route_ids": ["route-0"]},
            {"route_ids": ()},
            {"route_ids": ("",)},
            {"route_ids": (1,)},
            {"route_ids": ("same", "same"), "material_groups": [{}, {}]},
        ]
        for changes in bad_values:
            with self.subTest(changes=changes):
                harness = Harness([packet(0, **changes)])
                with self.assertRaises((TypeError, ValueError)):
                    harness.run()
                harness.render_mock.assert_not_called()
                self.assert_cleaned(harness)

    def test_every_failure_stage_attempts_all_cleanup(self):
        for stage in (
            "initialize",
            "iterate",
            "set_routes",
            "render",
            "write_batch",
            "invalidate_assets",
            "source.close",
            "sink.close",
            "release",
        ):
            with self.subTest(stage=stage):
                harness = Harness([packet(0, drop_asset_cache=True)], failures=(stage,))
                with self.assertRaisesRegex(RuntimeError, stage):
                    harness.run()
                self.assert_cleaned(harness)

    def test_iterator_failure_after_output_still_cleans_up(self):
        def source():
            yield packet(0)
            raise RuntimeError("producer disconnected")

        harness = Harness(source())
        with self.assertRaisesRegex(RuntimeError, "producer disconnected"):
            harness.run()
        self.assertEqual(len(harness.sink.writes), 1)
        self.assert_cleaned(harness)

    def test_multiple_close_failures_do_not_skip_later_cleanup(self):
        harness = Harness([], failures=("source.close", "sink.close", "release"))
        with self.assertRaisesRegex(RuntimeError, "release"):
            harness.run()
        self.assert_cleaned(harness)

    def test_invalid_history_capacity_cleans_up_before_initialization(self):
        harness = Harness([])
        with self.assertRaises((TypeError, ValueError)):
            harness.run(history_capacity=0)
        harness.initialize_mock.assert_not_called()
        self.assert_cleaned(harness)

    def test_worker_does_not_replace_business_callbacks_or_unrelated_fields(self):
        harness = Harness([packet(0)])
        names = (
            "compute_camera_motion_rate",
            "motion_rate_to_sigma",
            "create_gaussian_kernel",
            "unrelated",
            "imgs_raw",
        )
        previous = {name: getattr(harness.owner, name) for name in names}
        harness.run()
        for name, value in previous.items():
            self.assertIs(getattr(harness.owner, name), value)
        self.assert_cleaned(harness)


if __name__ == "__main__":
    unittest.main()
