"""Packed opacity values preserve live updates, broadcasting and output ownership."""

from dataclasses import replace
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch

from paste_render_executor import RenderExecutor


class HostUploadDouble:
    """Exercise packing without initializing CUDA in the default CPU tests."""

    def __init__(self):
        self.uploads = 0

    def prepare(self, shape, *, dtype, device):
        self.host = torch.empty(shape, dtype=dtype)
        return self.host

    def upload(self):
        self.uploads += 1
        return self.host.clone()


class OpacityPackingTests(unittest.TestCase):
    def setUp(self):
        self.executor = RenderExecutor()
        self.transfer = HostUploadDouble()
        self.executor._opacity_upload = self.transfer

    def prepare(self, values, batch, scale=1.0):
        params = [{"transparency": value} for value in values]
        boxes = [(0, 0, 7, 5)] * len(params)
        options = replace(self.executor.options, transparency_scale=scale)
        return self.executor._prepare_opacity_maps(
            params, boxes, torch.device("cuda:0"), batch, options
        )

    def test_pack_live_mixed_scalar_and_batch_values(self):
        values = [np.array(0.2, np.float32), np.arange(8, dtype=np.float32)[::-1], torch.ones(1)]
        first = self.prepare(values, 8, 255)
        for actual, value in zip(first, values):
            array = value.numpy() if isinstance(value, torch.Tensor) else value
            expected = torch.as_tensor(np.array(array, copy=True)).view(-1, 1, 1, 1) / 255
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(self.transfer.uploads, 1)
        values[0][...] = 0.7
        values[1][...] = 3
        second = self.prepare(values, 8, 255)
        self.assertEqual(self.transfer.uploads, 2)
        self.assertNotEqual(first[0].item(), second[0].item())
        torch.testing.assert_close(second[1], torch.full((8, 1, 1, 1), 3.0) / 255, rtol=0, atol=0)

    def test_dynamic_batch_and_slot_counts_keep_broadcast_shapes(self):
        for batch, slots in ((8, 16), (1, 3), (3, 7), (16, 2), (2, 1)):
            values = [
                np.full(batch if slot % 2 else 1, slot / 16, np.float32) for slot in range(slots)
            ]
            result = self.prepare(values, batch)
            self.assertEqual(len(result), slots)
            for actual, value in zip(result, values):
                self.assertEqual(actual.shape, (value.size, 1, 1, 1))
                np.testing.assert_array_equal(actual.numpy().ravel(), value)

    def test_unsupported_values_use_original_map_validation(self):
        original = self.executor._map

        def on_cpu(value, batch, hw, channels, layout, scale, device, name):
            return original(value, batch, hw, channels, layout, scale, torch.device("cpu"), name)

        values = [0.5, np.ones((5, 7), np.float32), np.array([0.1, 0.3], np.float64)]
        with patch.object(self.executor, "_map", side_effect=on_cpu) as fallback:
            actual = self.prepare(values, 2)
            self.assertEqual(fallback.call_count, 3)
            self.assertEqual(self.transfer.uploads, 0)
            self.assertEqual(actual[1].shape, (1, 1, 5, 7))
            with self.assertRaises(ValueError):
                self.prepare([np.ones(3, np.float32)], 2)

    def test_tensor_logical_negative_view_retains_original_conversion(self):
        value = torch._neg_view(torch.tensor([0.2, 0.4]))
        original = self.executor._map

        def on_cpu(value, batch, hw, channels, layout, scale, device, name):
            return original(value, batch, hw, channels, layout, scale, torch.device("cpu"), name)

        with patch.object(self.executor, "_map", side_effect=on_cpu) as fallback:
            actual = self.prepare([value], 2)
        self.assertEqual(fallback.call_count, 1)
        self.assertEqual(self.transfer.uploads, 0)
        torch.testing.assert_close(actual[0].flatten(), torch.tensor([-0.2, -0.4]), rtol=0, atol=0)


@unittest.skipUnless(
    os.environ.get("PASTE_TEST_DEVICE", "").startswith("cuda"), "explicit CUDA run"
)
class OpacityUploadCudaTests(unittest.TestCase):
    def test_exact_scaling_reuse_and_release(self):
        device = torch.device(os.environ["PASTE_TEST_DEVICE"])
        executor = RenderExecutor()
        try:
            for scale in (1.0, 255.0, 0.7):
                options = replace(executor.options, transparency_scale=scale)
                for batch in (8, 1, 3, 16, 8):
                    values = [np.linspace(-0.2, 1.1, batch, dtype=np.float32) for _ in range(16)]
                    params = [{"transparency": value} for value in values]
                    boxes = [(0, 0, 7, 5)] * len(params)
                    actual = executor._prepare_opacity_maps(params, boxes, device, batch, options)
                    for value, result in zip(values, actual):
                        expected = executor._map(
                            value, batch, (5, 7), (1,), "auto", scale, device, "ref"
                        )
                        torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    snapshots = [result.cpu().numpy().copy() for result in actual]
                    for value in values:
                        value += 0.1
                    updated = executor._prepare_opacity_maps(params, boxes, device, batch, options)
                    self.assertFalse(np.array_equal(snapshots[0], updated[0].cpu().numpy()))
            pointers = [value.data_ptr() for value in updated]
            again = executor._prepare_opacity_maps(params, boxes, device, batch, options)
            self.assertEqual(pointers, [value.data_ptr() for value in again])
        finally:
            executor.release_resources()
        info = executor.cache_info()["opacity_upload"]
        self.assertEqual(info["host_bytes"], 0)
        self.assertEqual(info["device_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
