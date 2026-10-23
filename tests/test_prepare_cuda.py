"""Opt-in CUDA mask-upload contracts against the preserved perf4.8 source.

Set PASTE_TEST_DEVICE=cuda:0 and, if needed, PASTE_PREPARE_BASELINE to the
directory restored from 6cebfce. These are exact output/lifetime checks, not
performance measurements. Run serially with other GPU work.
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from benchmarks.benchmark_upload_comparison import load_sources
from virtualads.paste_shared_mask import SharedMaskBuilder
from virtualads.paste_upload_buffer import PinnedUploadBuffer


CUDA_DEVICE = os.environ.get("PASTE_TEST_DEVICE", "")


@unittest.skipUnless(CUDA_DEVICE.startswith("cuda"), "set PASTE_TEST_DEVICE=cuda:0 explicitly")
class PrepareCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        directory = Path(
            os.environ.get("PASTE_PREPARE_BASELINE", "results/baseline_prepare_6cebfce")
        )
        cls.sources = load_sources(directory)
        cls.addClassCleanup(cls.sources.close)
        cls.old_mask_type = sys.modules[
            cls.sources.package_name + ".paste_shared_mask"
        ].SharedMaskBuilder

    def setUp(self):
        self.device = torch.device(CUDA_DEVICE)
        self.enterContext(torch.cuda.device(self.device))
        self.stream = torch.cuda.Stream(device=self.device)
        self.enterContext(torch.cuda.stream(self.stream))
        self.addCleanup(self.stream.synchronize)

    def builders(self, radius=3):
        current = SharedMaskBuilder(radius=radius, acceleration="triton")
        old = self.old_mask_type(radius=radius, acceleration="triton")
        self.addCleanup(current.release_resources)
        self.addCleanup(old.release_resources)
        return current, old

    def build(self, builder, items, shape=(17, 23)):
        return builder.build(items, height=shape[0], width=shape[1], device=self.device)

    def assert_exact(self, current, old, items, shape=(17, 23)):
        # A clone keeps each builder's borrowed result alive across later calls.
        expected = self.build(old, items, shape).clone()
        actual = self.build(current, items, shape).clone()
        self.stream.synchronize()
        self.assertTrue(torch.equal(actual, expected))

    def test_numeric_types_strides_clipping_and_overlap(self):
        current, old = self.builders()
        grid = np.linspace(0, 1, 120, dtype=np.float64).reshape(10, 12)
        readonly = grid.astype(np.float32)
        readonly.flags.writeable = False
        sources = [
            grid > 0.5,
            (grid > 0.5).astype(np.uint8),
            (grid > 0.5).astype(np.int64),
            grid.astype(np.float16),
            grid.astype(np.float32),
            grid,
            grid[:, ::-1],
            grid.astype(">f8"),
            readonly,
            torch.from_numpy(grid.astype(np.float32)).t().contiguous().t(),
            torch.from_numpy(grid.astype(np.float32)).to(torch.bfloat16),
            torch._neg_view(-torch.from_numpy(grid.astype(np.float32))),
        ]
        for source in sources:
            with self.subTest(dtype=str(source.dtype), strides=str(getattr(source, "strides", ""))):
                items = [
                    {"bbox": (-2, -3, 10, 7), "mask": source},
                    {"bbox": (4, 2, 16, 12), "mask": source},
                    {"bbox": (19, 13, 31, 23), "mask": source},
                ]
                self.assert_exact(current, old, items)
        self.assertGreater(current.cache_info()["patch_upload"]["uploads"], 0)

    def test_live_values_and_queued_consumers_reuse_storage(self):
        current, old = self.builders()
        items = [
            {"bbox": (1, 2, 13, 12), "mask": np.zeros((10, 12), np.float32)},
            {"bbox": (5, 4, 17, 14), "mask": np.zeros((10, 12), np.uint8)},
        ]
        references = []
        for step in range(40):
            items[0]["mask"][:] = (step % 9) / 8
            items[1]["mask"][:] = step % 2
            references.append(self.build(old, items).clone())
        self.stream.synchronize()
        consumers = []
        with patch("torch.cuda.synchronize", side_effect=AssertionError("global sync in loop")):
            for step in range(40):
                items[0]["mask"][:] = (step % 9) / 8
                items[1]["mask"][:] = step % 2
                if hasattr(torch.cuda, "_sleep"):
                    torch.cuda._sleep(200_000)
                consumers.append(self.build(current, items).clone())
        self.stream.synchronize()
        for actual, expected in zip(consumers, references):
            self.assertTrue(torch.equal(actual, expected))
        info = current.cache_info()["patch_upload"]
        self.assertEqual(info["reallocations"], 1)
        self.assertEqual(info["uploads"], 40)

    def test_gpu_and_mixed_sources_fall_back_then_resume_host_uploads(self):
        current, old = self.builders(radius=0)
        source = np.linspace(0, 1, 120, dtype=np.float32).reshape(10, 12)
        host = {"bbox": (0, 0, 12, 10), "mask": source}
        gpu = {"bbox": (3, 2, 15, 12), "mask": torch.from_numpy(source).to(self.device)}
        self.assert_exact(current, old, [host])
        allocations = current.cache_info()["patch_upload"]["reallocations"]
        uploads = current.cache_info()["patch_upload"]["uploads"]
        self.assert_exact(current, old, [gpu])
        self.assert_exact(current, old, [host, gpu])
        self.assertEqual(current.cache_info()["patch_upload"]["uploads"], uploads)
        source[:] = 0.3
        self.assert_exact(current, old, [host])
        self.assertEqual(current.cache_info()["patch_upload"]["reallocations"], allocations)

    def test_oversized_temporary_empty_and_release_reuse(self):
        current, old = self.builders()
        current._patch_upload = PinnedUploadBuffer(max_cached_bytes=16)
        items = [{"bbox": (0, 0, 12, 10), "mask": np.ones((10, 12), np.float32)}]
        self.assert_exact(current, old, items)
        info = current.cache_info()["patch_upload"]
        self.assertEqual(info["capacity_bytes"], 0)
        self.assertEqual(info["device_bytes"], 0)
        self.assertGreater(info["oversize_uploads"], 0)
        self.assert_exact(current, old, [])
        for _ in range(2):
            current.release_resources()
            self.assertEqual(current.cache_info()["patch_upload"]["host_bytes"], 0)
            self.assertEqual(current.cache_info()["patch_upload"]["device_bytes"], 0)
        self.assert_exact(current, old, items)

    def test_invalid_cpu_values_preserve_validation_and_recover(self):
        current, old = self.builders()
        good = [{"bbox": (0, 0, 12, 10), "mask": np.ones((10, 12), np.float32)}]
        self.assert_exact(current, old, good)
        for value in (np.nan, np.inf, -0.1, 1.1):
            invalid = deepcopy(good)
            invalid[0]["mask"][3, 2] = value
            for builder in (current, old):
                with self.assertRaises(ValueError):
                    self.build(builder, invalid)
            self.assert_exact(current, old, good)


if __name__ == "__main__":
    unittest.main()
