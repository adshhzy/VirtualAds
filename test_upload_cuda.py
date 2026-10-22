"""Opt-in real CUDA contracts for bounded upload workspaces.

Run serially with PASTE_TEST_DEVICE=cuda:0. With the variable unset (or set to
CPU), every test is skipped without querying CUDA. Consumers are queued on one
compute stream before its borrowed input storage is reused. Loop bodies forbid
global synchronize(); the upload helper's own H2D event waits remain permitted.
Small retained clone consumers are checked after the final stream wait.

Matrix references reproduce the old per-slot float32 transfer and use the same
Triton ROI kernel. These tests isolate transfer/layout equivalence, rather than
claiming that different geometry backends have identical rounding.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np
import torch

from paste_frame_upload import FrameUploadBuffer
from paste_shared_params import SharedParameterBuilder
from paste_upload_buffer import PinnedUploadBuffer


CUDA_DEVICE = os.environ.get("PASTE_TEST_DEVICE", "")


@unittest.skipUnless(CUDA_DEVICE.startswith("cuda"), "set PASTE_TEST_DEVICE=cuda:0 explicitly")
class UploadCudaTests(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.fail("PASTE_TEST_DEVICE requested CUDA, but CUDA is unavailable")
        self.device = torch.device(CUDA_DEVICE)
        self.enterContext(torch.cuda.device(self.device))
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.stream = torch.cuda.Stream(device=self.device)
        self.enterContext(torch.cuda.stream(self.stream))
        # Buffers register cleanup later, so they release pending host reads
        # first; this wait then completes any remaining clone/kernel consumers.
        self.addCleanup(self.stream.synchronize)

    def byte_buffer(self, budget=4096):
        buffer = PinnedUploadBuffer(max_cached_bytes=budget)
        self.addCleanup(buffer.release_resources)
        return buffer

    def frame_buffer(self, **kwargs):
        buffer = FrameUploadBuffer(**kwargs)
        self.addCleanup(buffer.release_resources)
        return buffer

    def matrix_builder(self):
        builder = SharedParameterBuilder(acceleration="triton", normalization="legacy")
        self.addCleanup(builder.release_resources)
        return builder

    @staticmethod
    def delay_stream():
        # Keep a short queue ahead of H2D so CPU rewrites exercise its event wait.
        # _sleep is only a test delay, not part of production or a timing result.
        if hasattr(torch.cuda, "_sleep"):
            torch.cuda._sleep(200_000)

    def assert_consumers(self, consumers):
        self.stream.synchronize()
        for actual, expected in consumers:
            np.testing.assert_array_equal(actual.cpu().numpy(), expected)

    def assert_upload_released(self, buffer):
        info = buffer.cache_info()
        self.assertEqual(info["host_bytes"], 0)
        self.assertEqual(info["device_bytes"], 0)
        self.assertEqual(info["capacity_bytes"], 0)
        self.assertFalse(info["copy_event_recorded"])

    def test_pinned_reuse_orders_h2d_and_clone_consumers_without_global_waits(self):
        buffer = self.byte_buffer()
        consumers, host_pointers, device_pointers = [], [], []
        with patch.object(torch.cuda, "synchronize", side_effect=AssertionError("global wait")):
            for step in range(12):
                self.delay_stream()
                host = buffer.prepare((17, 19), dtype=torch.uint8, device=self.device)
                self.assertTrue(host.is_pinned())
                host.fill_(step * 19)
                borrowed = buffer.upload()
                self.assertEqual(borrowed.device, self.device)
                consumers.append((borrowed.clone(), np.full((17, 19), step * 19, np.uint8)))
                host_pointers.append(host.data_ptr())
                device_pointers.append(borrowed.data_ptr())
                self.assertTrue(buffer.cache_info()["copy_event_recorded"])
            self.assertEqual(buffer.cache_info()["uploads"], 12)
            self.assertEqual(buffer.cache_info()["reallocations"], 1)
            # Releasing the pending host source is safe while clone consumers
            # remain queued on the same stream.
            buffer.release_resources()
        self.assertEqual(len(set(host_pointers)), 1)
        self.assertEqual(len(set(device_pointers)), 1)
        self.assert_upload_released(buffer)
        self.assert_consumers(consumers)

    def test_pinned_dynamic_shape_dtype_oversize_and_pending_release(self):
        buffer = self.byte_buffer(budget=128)
        requests = (
            ((4, 4), torch.uint8),
            ((8, 8), torch.uint8),
            ((2, 4), torch.uint8),
            ((4, 5), torch.float32),
            ((8, 5), torch.float32),  # 160 bytes: above the cached bound.
            ((2, 2), torch.float64),
        )
        consumers = []
        oversized = None
        with patch.object(torch.cuda, "synchronize", side_effect=AssertionError("global wait")):
            for step, (shape, dtype) in enumerate(requests):
                self.delay_stream()
                host = buffer.prepare(shape, dtype=dtype, device=self.device)
                host.fill_(step + 1)
                expected = host.numpy().copy()
                borrowed = buffer.upload()
                consumers.append((borrowed.clone(), expected))
                info = buffer.cache_info()
                self.assertLessEqual(info["capacity_bytes"], 128)
                if step == 4:
                    oversized = info
                    self.assertEqual(info["temporary_host_bytes"], 160)
                    self.assertEqual(info["device_bytes"], 0)
                    buffer.release_resources()
                    self.assert_upload_released(buffer)
            buffer.release_resources()
        self.assertIsNotNone(oversized)
        self.assertEqual(buffer.cache_info()["oversize_uploads"], 1)
        self.assertEqual(buffer.cache_info()["context_changes"], 3)
        self.assert_upload_released(buffer)
        self.assert_consumers(consumers)

    def test_compute_stream_change_is_rejected_until_release(self):
        buffer = self.byte_buffer()
        buffer.prepare((9,), dtype=torch.float32, device=self.device).fill_(7)
        other = torch.cuda.Stream(device=self.device)
        self.addCleanup(other.synchronize)
        with torch.cuda.stream(other):
            with self.assertRaisesRegex(RuntimeError, "same compute stream"):
                buffer.upload()
            with self.assertRaisesRegex(RuntimeError, "compute stream changed"):
                buffer.prepare((9,), dtype=torch.float32, device=self.device)
        first = buffer.upload().clone()
        buffer.release_resources()
        with torch.cuda.stream(other):
            buffer.prepare((9,), dtype=torch.float32, device=self.device).fill_(11)
            second = buffer.upload().clone()
            buffer.release_resources()
        other.synchronize()
        self.assert_consumers(
            [(first, np.full(9, 7, np.float32)), (second, np.full(9, 11, np.float32))]
        )

    def test_frame_uint8_conversion_exact_and_fixed_shape_allocations_stable(self):
        buffer = self.frame_buffer(max_uint8_bytes=4096, max_float32_bytes=16384)
        values = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        frame = values.copy()
        consumers, pointers = [], []
        with patch.object(torch.cuda, "synchronize", side_effect=AssertionError("global wait")):
            for step in range(8):
                frame[:] = (values.astype(np.uint16) + step * 17) % 256
                self.delay_stream()
                borrowed = buffer.prepare(frame, self.device)
                self.assertEqual(borrowed.dtype, torch.float32)
                pointers.append(borrowed.data_ptr())
                consumers.append((borrowed.clone(), frame.astype(np.float32)))
                self.assertEqual(buffer.cache_info()["float32_allocations"], 1)
                self.assertEqual(buffer.cache_info()["uint8_upload"]["reallocations"], 1)
            buffer.release_resources()
        self.assertEqual(len(set(pointers)), 1)
        self.assertEqual(buffer.cache_info()["float32_bytes"], 0)
        self.assert_upload_released(buffer._upload)
        self.assert_consumers(consumers)

    def test_frame_growth_shrink_layouts_and_oversize_preserve_queued_consumers(self):
        buffer = self.frame_buffer(max_uint8_bytes=1024, max_float32_bytes=4096)
        shapes = ((8, 8, 3), (16, 16, 3), (4, 4, 3), (3, 7, 11), (24, 24, 3), (8, 8, 3))
        consumers, info_rows, pointers = [], [], []
        with patch.object(torch.cuda, "synchronize", side_effect=AssertionError("global wait")):
            for step, shape in enumerate(shapes):
                # Slot 3 is a contiguous CHW CPU Tensor; other slots are HWC.
                values = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
                source = torch.from_numpy(values) if step == 3 else values
                borrowed = buffer.prepare(source, self.device)
                consumers.append((borrowed.clone(), values.astype(np.float32)))
                pointers.append(borrowed.data_ptr())
                info_rows.append(buffer.cache_info())
                self.assertLessEqual(info_rows[-1]["float32_bytes"], 4096)
                self.assertLessEqual(info_rows[-1]["uint8_upload"]["capacity_bytes"], 1024)
            buffer.release_resources()
        self.assertEqual(pointers[1], pointers[2])
        self.assertEqual(pointers[1], pointers[3])
        self.assertEqual(info_rows[3]["float32_allocations"], 2)
        self.assertEqual(info_rows[3]["uint8_upload"]["reallocations"], 2)
        self.assertEqual(info_rows[4]["float32_bytes"], 0)
        self.assertEqual(info_rows[4]["float32_oversize_allocations"], 1)
        self.assertEqual(info_rows[4]["uint8_upload"]["oversize_uploads"], 1)
        self.assertEqual(info_rows[4]["uint8_upload"]["device_bytes"], 0)
        self.assertEqual(info_rows[5]["float32_allocations"], 3)
        self.assertEqual(info_rows[5]["uint8_upload"]["reallocations"], 4)
        self.assert_consumers(consumers)

    def test_frame_strided_float_and_cuda_inputs_keep_compatible_conversion(self):
        buffer = self.frame_buffer()
        source = np.arange(8 * 11 * 3, dtype=np.uint8).reshape(8, 11, 3)
        variants = (
            source[:, ::2],
            source[:, ::-1],
            source.transpose(2, 0, 1),
            torch.from_numpy(source).transpose(0, 1),
            source.astype(np.float64),
            torch.from_numpy(source).to(self.device),
        )
        consumers = []
        for variant in variants:
            buffer.prepare(source, self.device)
            output = buffer.prepare(variant, self.device)
            original = variant
            if isinstance(original, np.ndarray) and any(stride < 0 for stride in original.strides):
                original = original.copy()
            reference = torch.as_tensor(original, device=self.device, dtype=torch.float32)
            self.assertEqual(output.stride(), reference.stride())
            consumers.append((output.clone(), reference.clone()))
            self.assertEqual(buffer.cache_info()["float32_bytes"], 0)
            self.assert_upload_released(buffer._upload)
        self.stream.synchronize()
        for actual, expected in consumers:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @staticmethod
    def host_matrices():
        first = np.array([[0.75, 0.1, 0.125], [0.0, 0.8, -0.25], [0.001, 0, 1]], np.float32)
        second = torch.from_numpy((first + np.eye(3, dtype=np.float32) * 0.125)[None].copy())
        batched = np.stack([first.copy() for _ in range(3)]).astype(np.float64)
        batched[:, 0, 2] += np.array([0.125, 0.25, 0.375])
        return [first, second, batched]

    def legacy_matrices(self, raw):
        # This is the previous mixed-rank per-slot conversion. Copy negative
        # strides just as the old _float_tensor converter did.
        result = []
        for value in raw:
            if not isinstance(value, torch.Tensor) and any(stride < 0 for stride in value.strides):
                value = value.copy()
            result.append(torch.as_tensor(value, dtype=torch.float32, device=self.device))
        return result

    def test_matrix_mixed_ranks_pack_once_and_same_backend_grids_stay_independent(self):
        from paste_shared_kernels import build_roi_grid_gpu

        builder = self.matrix_builder()
        raw = self.host_matrices()
        packed = builder._prepare_warp_matrices(raw, self.device)
        original_values = [np.asarray(value).astype(np.float32) for value in raw]
        consumers = [(value.clone(), expected) for value, expected in zip(packed, original_values)]
        self.assertEqual(builder.cache_info()["warp_upload"]["uploads"], 1)
        self.assertEqual(builder.last_info["warp_host_matrix_batches"], 1)
        self.assertEqual([tuple(value.shape) for value in packed], [(3, 3), (1, 3, 3), (3, 3, 3)])
        self.assertEqual(len({value.untyped_storage().data_ptr() for value in packed}), 1)
        params = [{"m_inv": value} for value in raw]
        sizes = ((7, 9), (8, 11), (9, 13))
        boxes = ((2, 3, 9, 8), (3, 4, 12, 11), (1, 2, 14, 11))
        ads = [torch.empty((3, 4, *size), device=self.device) for size in sizes]

        def build():
            return builder.build_warp_grids(
                params=params,
                ads=ads,
                boxes=boxes,
                device=self.device,
                batch=3,
                warp_align_corners=False,
                matrix_coordinates="roi",
            )

        def reference():
            return tuple(
                build_roi_grid_gpu(
                    size,
                    matrix,
                    (box[3] - box[1], box[2] - box[0]),
                    align_corners=False,
                    normalization="legacy",
                    eps=1e-8,
                )
                for size, box, matrix in zip(sizes, boxes, self.legacy_matrices(raw))
            )

        first, expected_first = build(), reference()
        first_snapshots = tuple(value.clone() for value in first)
        allocations = builder.cache_info()["warp_upload"]["reallocations"]
        raw[0][0, 2] += 0.25
        raw[1][0, 1, 2] -= 0.125
        raw[2][:, 0, 2] += 0.5
        with patch.object(torch.cuda, "synchronize", side_effect=AssertionError("global wait")):
            current, expected_current = build(), reference()
            self.assertEqual(builder.cache_info()["warp_upload"]["reallocations"], allocations)
            builder.release_resources()
        self.assert_consumers(consumers)
        for previous, snapshot, expected, updated, updated_expected in zip(
            first, first_snapshots, expected_first, current, expected_current
        ):
            torch.testing.assert_close(previous, expected, rtol=0, atol=0)
            torch.testing.assert_close(previous, snapshot, rtol=0, atol=0)
            torch.testing.assert_close(updated, updated_expected, rtol=0, atol=0)
            self.assertFalse(torch.equal(previous, updated))
        self.assert_upload_released(builder._warp_upload)

    def test_matrix_cpu_gpu_mix_keeps_fallback_and_does_not_reuse_old_values(self):
        builder = self.matrix_builder()
        raw = self.host_matrices()
        first = builder._prepare_warp_matrices(raw, self.device)
        consumers = [
            (value.clone(), np.asarray(source).astype(np.float32))
            for value, source in zip(first, raw)
        ]
        before = builder.cache_info()["warp_upload"]
        raw[2] = torch.as_tensor(raw[2], device=self.device, dtype=torch.float32)
        for step in range(2):
            raw[0][0, 2] += 0.25
            raw[2][:, 1, 2].add_(0.125)
            matrices = builder._prepare_warp_matrices(raw, self.device)
            expected = self.legacy_matrices(raw)
            self.assertIsNone(builder.last_info["warp_host_matrix_batches"])
            self.assertEqual(builder.cache_info()["warp_upload"]["uploads"], before["uploads"])
            self.assertEqual(
                builder.cache_info()["warp_upload"]["reallocations"], before["reallocations"]
            )
            self.assertEqual(matrices[2].data_ptr(), raw[2].data_ptr())
            for value, reference in zip(matrices, expected):
                # Queue snapshots before the next raw GPU matrix is edited.
                consumers.append((value.clone(), reference.clone()))
        builder.release_resources()
        self.stream.synchronize()
        for actual, expected in consumers:
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                np.testing.assert_array_equal(actual.cpu().numpy(), expected)
        self.assert_upload_released(builder._warp_upload)


if __name__ == "__main__":
    unittest.main()
