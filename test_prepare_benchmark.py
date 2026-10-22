"""CPU-only contracts for candidate selection and preparation diagnostics."""

from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import benchmark_prepare_comparison as benchmark
import check_prepare_streaming as streaming


class FakeWorker:
    def __init__(self, *args, **kwargs):
        self._use_prepared_route_inputs = None
        self.mask_builder = SimpleNamespace(_batch_upload_enabled=None)
        self.closed = False

    def close(self):
        self.closed = True


class CandidateTests(unittest.TestCase):
    def test_candidate_switches_are_independent_and_instance_local(self):
        for name, (route, mask) in benchmark.SWITCHES.items():
            worker = FakeWorker()
            other = FakeWorker()
            self.assertIs(benchmark.configure_worker(worker, name), worker)
            self.assertIs(worker._use_prepared_route_inputs, route)
            self.assertIs(worker.mask_builder._batch_upload_enabled, mask)
            self.assertIsNone(other._use_prepared_route_inputs)
            self.assertIsNone(other.mask_builder._batch_upload_enabled)

    def test_missing_switch_is_an_error_not_a_silent_benchmark(self):
        for field in ("route", "mask"):
            worker = FakeWorker()
            if field == "route":
                del worker._use_prepared_route_inputs
            else:
                del worker.mask_builder._batch_upload_enabled
            with self.assertRaisesRegex(RuntimeError, "both preparation switches"):
                benchmark.configure_worker(worker, "both")
        with self.assertRaises(ValueError):
            benchmark.configure_worker(FakeWorker(), "baseline")

    def test_api_proxy_only_configures_constructed_worker(self):
        api = SimpleNamespace(
            ImageRenderer=FakeWorker, RenderOptions=object(), build_route_plan=object()
        )
        proxy = benchmark.CandidateAPI(api, "route_only")
        worker = proxy.ImageRenderer("owner", "options", device="cuda:0")
        self.assertIsInstance(worker, FakeWorker)
        self.assertIs(proxy.RenderOptions, api.RenderOptions)
        self.assertIs(proxy.build_route_plan, api.build_route_plan)
        self.assertFalse(worker.closed)

    def test_construction_failure_closes_resources(self):
        worker = FakeWorker()
        del worker.mask_builder._batch_upload_enabled
        api = SimpleNamespace(ImageRenderer=lambda: worker)
        with self.assertRaises(RuntimeError):
            benchmark.CandidateAPI(api, "both").ImageRenderer()
        self.assertTrue(worker.closed)

    def test_rotated_order_balances_each_candidate_position(self):
        orders = [benchmark.base.rotated_order(benchmark.CANDIDATES, i) for i in range(4)]
        for position in range(4):
            self.assertEqual({order[position] for order in orders}, set(benchmark.CANDIDATES))

    def test_aggregate_keeps_all_candidates_and_pair_counts(self):
        phases = []
        for repeat in range(2):
            for index, name in enumerate(benchmark.CANDIDATES):
                samples = [float(10 - index), float(12 - index)]
                phases.append(
                    {
                        "candidate": name,
                        "round": repeat + 1,
                        "samples_ms": samples,
                        "stats_ms": benchmark.base.stats(samples),
                    }
                )
        result = benchmark.aggregate(phases, benchmark.CANDIDATES)
        self.assertEqual(set(result), set(benchmark.CANDIDATES))
        for candidate in result.values():
            self.assertEqual(candidate["checked_calls"], 4)
            self.assertEqual(candidate["checked_route_images"], 32)
        self.assertAlmostEqual(result["both"]["reduction_percent"]["mean"], 300 / 11)

    def test_closed_upload_check_includes_temporary_buffers(self):
        fields = ("host_bytes", "device_bytes", "temporary_host_bytes", "temporary_device_bytes")
        for field in fields:
            upload = dict.fromkeys(fields, 0)
            benchmark.assert_uploads_released({"mask": {"patch_upload": upload}})
            upload[field] = 1
            with self.assertRaisesRegex(AssertionError, field):
                benchmark.assert_uploads_released({"mask": {"patch_upload": upload}})
        benchmark.assert_uploads_released({"mask": {}})

    def test_all_measurement_dependency_files_are_hashed(self):
        result = benchmark.source_hashes()
        for name in (
            "benchmark_prepare_comparison.py",
            "benchmark_upload_comparison.py",
            "benchmark_render_img_modes.py",
            "benchmark_shared_paste.py",
            "benchmark_paste_gpu_batch.py",
            "live_render_demo.py",
            "paste_renderer.py",
        ):
            self.assertIn(name, result["files"])
        self.assertEqual(len(result["combined_sha256"]), 64)


class ArgumentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        directory = Path(self.temp.name)
        self.args = []
        for name in ("sample", "frame-bmp", "ad-bmp"):
            path = directory / name
            path.touch()
            self.args += ["--" + name, str(path)]
        self.args += ["--baseline", str(directory), "--output", str(directory / "report.json")]

    def error(self, extra, message):
        stream = StringIO()
        with redirect_stderr(stream), self.assertRaises(SystemExit):
            benchmark.parse_arguments(self.args + extra)
        self.assertIn(message, stream.getvalue())

    def test_default_four_candidate_order_and_balanced_pair(self):
        args = benchmark.parse_arguments(self.args)
        self.assertEqual(tuple(args.candidates), benchmark.CANDIDATES)
        pair = benchmark.parse_arguments(
            self.args + ["--candidates", "baseline", "both", "--rounds", "2"]
        )
        self.assertEqual(pair.candidates, ["baseline", "both"])

    def test_unbalanced_requires_explicit_exploration(self):
        self.error(["--rounds", "2"], "multiple of candidate count")
        args = benchmark.parse_arguments(self.args + ["--rounds", "2", "--exploratory"])
        self.assertTrue(args.exploratory)

    def test_invalid_scope_and_existing_output_rejected(self):
        self.error(["--candidates", "both"], "include baseline")
        self.error(["--candidates", "baseline", "both", "both"], "duplicates")
        self.error(["--device", "cpu"], "requires CUDA")
        self.error(["--rounds", "1"], "rounds >= 2")
        Path(self.temp.name, "report.json").touch()
        self.error([], "refusing to overwrite")


class MaskMutationTests(unittest.TestCase):
    def test_mask_mutates_in_place_preserving_shape_bbox_and_other_values(self):
        first = np.ones((10, 11), np.float32)
        second = np.ones((2, 3), np.float32)
        data = {
            "mask_coor_list": [
                {"bbox": (2, 4, 13, 14), "mask": first},
                {"bbox": (0, 0, 3, 2), "mask": second},
            ]
        }
        boxes = [entry["bbox"] for entry in data["mask_coor_list"]]
        streaming.mutate_masks(data, 2)
        self.assertIs(data["mask_coor_list"][0]["mask"], first)
        self.assertIs(data["mask_coor_list"][1]["mask"], second)
        self.assertTrue(np.all(first[:8, :8] == 0))
        self.assertTrue(np.all(first[8:, :] == 1))
        self.assertTrue(np.all(first[:, 8:] == 1))
        self.assertTrue(np.all(second == 1))
        self.assertEqual([entry["bbox"] for entry in data["mask_coor_list"]], boxes)

    def test_mask_sampling_period_covers_both_binary_values(self):
        mask = np.ones((3, 4), np.uint8)
        data = {"mask_coor_list": [{"mask": mask}]}
        values = []
        for sample_index in range(7):
            streaming.mutate_masks(data, sample_index * 250)
            values.append(float(mask[0, 0]))
        self.assertEqual(values, [0, 1, 1, 0, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
