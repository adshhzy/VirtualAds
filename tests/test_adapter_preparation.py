"""Input-boundary and output-lifetime checks without duplicate core validation."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from benchmarks.benchmark_paste_gpu_batch import adapter_options, make_adapter_case
from tools.check_backend_contracts import BOXES, make_fixture
from virtualads.paste_gpu_batch_adapter import PasteBatchAdapter


def make_call():
    args, _ = make_fixture(torch.device("cpu"), 3)
    call = list(make_adapter_case(args, BOXES))
    call[-1] = list(call[-1])
    return call


class PreparationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_invalid_dynamic_inputs_fail_before_rendering(self):
        # Exercise the public adapter boundary, including malformed later frames.
        # It must keep validating changing inputs even after a successful call.
        for field in (
            "frame",
            "bbox",
            "source",
            "grid",
            "opacity",
            "mask",
            "kernel_batch",
            "kernel_even",
        ):
            with self.subTest(field=field):
                call = make_call()
                adapter = PasteBatchAdapter(adapter_options("eager", max_cached_sizes=4))
                adapter(*call)
                if field == "frame":
                    call[1] = call[1][..., :2]
                elif field == "bbox":
                    call[2]["ad_params"][0]["bbox"] = (-1, 1, 13, 10)
                elif field == "source":
                    record = call[0].imgs_raw_groups[0]["ad0"]
                    record["img"] = record["img"][..., :3]
                elif field == "grid":
                    call[-1][5][0] = torch.zeros((2, 9, 12, 2))
                elif field == "opacity":
                    call[2]["ad_params"][0]["transparency"] = torch.ones((3, 2, 9, 12))
                elif field == "mask":
                    call[-1][4] = torch.ones((3, 2, 19, 29))
                elif field == "kernel_batch":
                    call[-1][3] = torch.ones((2, 4, 1, 3, 3))
                else:
                    call[-1][3] = torch.ones((4, 1, 2, 3))
                with patch.object(adapter, "_render_prepared") as render:
                    with self.assertRaises((TypeError, ValueError)):
                        adapter(*call)
                    render.assert_not_called()

    def test_map_plan_reuse_reads_current_values(self):
        call = make_call()
        options = adapter_options("eager", max_cached_sizes=4)
        adapter = PasteBatchAdapter(options)
        first = adapter(*call)
        snapshot = first.copy()
        call[-1][4].fill_(1)
        call[2]["ad_params"][0]["transparency"].fill_(0)
        actual = adapter(*call)
        expected = PasteBatchAdapter(options)(*call)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(first, snapshot)
        self.assertFalse(np.array_equal(first, actual))
        self.assertFalse(np.shares_memory(first, actual))

    def test_custom_builder_cannot_poison_cached_sources(self):
        for fail_inside in (False, True):
            with self.subTest(fail_inside=fail_inside):
                call = make_call()
                grids = tuple(call[-1][5].values())

                def bad_builder(*, ads, **kwargs):
                    batch, _, height, width = ads[0].shape
                    ads[0].resize_(batch, 3, height, width)
                    if fail_inside:
                        raise RuntimeError("callback failure")
                    return grids

                options = adapter_options("eager", max_cached_sizes=4)
                adapter = PasteBatchAdapter(
                    replace(options, grid_source="matrix", warp_grids_builder=bad_builder)
                )
                with patch.object(adapter, "_render_prepared") as render:
                    with self.assertRaises((RuntimeError, ValueError)):
                        adapter(*call)
                    render.assert_not_called()
                self.assertEqual(adapter.cache_info()["sizes"], {})
                adapter.options = options
                actual = adapter(*call)
                np.testing.assert_array_equal(actual, PasteBatchAdapter(options)(*call))

    def test_release_keeps_outputs_and_allows_reuse(self):
        for backend in ("batch", "bucketed"):
            with self.subTest(backend=backend):
                call = make_call()
                options = replace(adapter_options("eager", max_cached_sizes=4), backend=backend)
                adapter = PasteBatchAdapter(options)
                first = adapter(*call)
                expected = first.copy()
                adapter.release_resources()
                self.assertEqual(adapter.cache_info()["sizes"], {})
                self.assertIsNone(adapter._layout)
                self.assertFalse(adapter._map_plans)
                np.testing.assert_array_equal(first, expected)
                second = adapter(*call)
                np.testing.assert_array_equal(second, expected)
                self.assertFalse(np.shares_memory(first, second))


if __name__ == "__main__":
    unittest.main()
