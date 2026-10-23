"""CPU and mocked-CUDA contracts for bounded upload staging and matrix packing.

No test allocates CUDA/pinned memory or runs GPU kernels. Fake CUDA events and
streams exercise lifecycle ordering; real CUDA behavior is checked separately
by the serial GPU regression harness.
"""

from contextlib import ExitStack
import gc
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch

from virtualads import paste_shared_params as shared
from virtualads import paste_upload_buffer as uploads


class FakeEvent:
    def __init__(self, log, **kwargs):
        self.log = log

    def record(self, stream):
        self.log.append(("record", stream.name))

    def synchronize(self):
        self.log.append(("wait",))


class FakeStream:
    def __init__(self, name="main"):
        self.name = name
        self.synchronizations = 0

    def synchronize(self):
        self.synchronizations += 1


class MockCuda:
    """Replace only allocation/stream/event services; tensor math stays on CPU."""

    def __init__(self):
        self.stack = ExitStack()
        self.log = []
        self.stream = FakeStream()
        self.real_empty = torch.empty

    def __enter__(self):
        def empty(*args, **kwargs):
            requested_device = torch.device(kwargs.pop("device", "cpu"))
            pinned = kwargs.pop("pin_memory", False)
            self.log.append(("allocate", requested_device.type, pinned, args[0]))
            target = "meta" if requested_device.type == "meta" else "cpu"
            return self.real_empty(*args, **kwargs, device=target)

        self.stack.enter_context(patch.object(uploads.torch, "empty", side_effect=empty))
        self.stack.enter_context(
            patch.object(
                uploads.torch.cuda, "current_stream", side_effect=lambda device: self.stream
            )
        )
        self.stack.enter_context(
            patch.object(
                uploads.torch.cuda, "Event", side_effect=lambda **kw: FakeEvent(self.log, **kw)
            )
        )
        return self

    def __exit__(self, *args):
        self.stack.__exit__(*args)


class UploadBufferTests(unittest.TestCase):
    def test_cpu_upload_is_uncached_independent_and_writable_outside_inference(self):
        buffer = uploads.PinnedUploadBuffer(64)
        host = buffer.prepare((2, 3), dtype=torch.float32, device="cpu")
        host.fill_(3)
        first = buffer.upload()
        host.fill_(9)
        torch.testing.assert_close(first, torch.full((2, 3), 3.0), rtol=0, atol=0)
        self.assertEqual(buffer.cache_info()["host_bytes"], 0)
        second_host = buffer.prepare((2, 3), dtype=torch.float32, device="cpu")
        second_host.fill_(7)
        second = buffer.upload()
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertFalse(first.requires_grad)
        self.assertIsNone(first.grad_fn)

    def test_invalid_configuration_and_protocol_fail_without_cuda(self):
        for budget in (-1, True, None, 1.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                uploads.PinnedUploadBuffer(budget)
        buffer = uploads.PinnedUploadBuffer()
        with self.assertRaises(RuntimeError):
            buffer.upload()
        for shape in (3, (-1,), (True,), (1.2,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                buffer.prepare(shape, dtype=torch.float32, device="cpu")
        with self.assertRaises(TypeError):
            buffer.prepare((1,), dtype="float32", device="cpu")
        with self.assertRaises(ValueError):
            buffer.prepare((1,), dtype=torch.float32, device="meta")

    def test_scalar_and_empty_shapes_keep_shape_and_ownership(self):
        buffer = uploads.PinnedUploadBuffer(0)
        for shape in ((), (0,), (2, 0, 3)):
            host = buffer.prepare(shape, dtype=torch.uint8, device="cpu")
            host.fill_(5)
            result = buffer.upload()
            self.assertEqual(tuple(result.shape), shape)
            self.assertEqual(buffer.cache_info()["capacity_bytes"], 0)

    def test_cuda_reuse_waits_before_cpu_write_and_grows_geometrically(self):
        with MockCuda() as cuda:
            buffer = uploads.PinnedUploadBuffer(64)
            host = buffer.prepare((4,), dtype=torch.float32, device="cuda:0")
            host.fill_(1)
            first = buffer.upload()
            original_pointer = first.data_ptr()
            host = buffer.prepare((3,), dtype=torch.float32, device="cuda:0")
            self.assertEqual(cuda.log[-1], ("wait",))
            host.fill_(2)
            second = buffer.upload()
            self.assertEqual(second.data_ptr(), original_pointer)
            self.assertEqual(buffer.cache_info()["reallocations"], 1)
            host = buffer.prepare((5,), dtype=torch.float32, device="cuda:0")
            host.fill_(3)
            third = buffer.upload()
            self.assertNotEqual(third.data_ptr(), original_pointer)
            self.assertEqual(buffer.cache_info()["capacity_bytes"], 32)
            buffer.release_resources()
            self.assertEqual(cuda.log[-1], ("wait",))
            self.assertEqual(buffer.cache_info()["host_bytes"], 0)
            self.assertEqual(buffer.cache_info()["device_bytes"], 0)
            self.assertEqual(cuda.stream.synchronizations, 0)

    def test_oversize_is_temporary_and_release_does_not_overwrite_borrowed_output(self):
        with MockCuda():
            buffer = uploads.PinnedUploadBuffer(16)
            host = buffer.prepare((8,), dtype=torch.float32, device="cuda:0")
            host.copy_(torch.arange(8))
            result = buffer.upload()
            snapshot = result.clone()
            info = buffer.cache_info()
            self.assertEqual(info["capacity_bytes"], 0)
            self.assertEqual(info["temporary_host_bytes"], 32)
            self.assertEqual(info["device_bytes"], 0)
            self.assertEqual(info["oversize_uploads"], 1)
            buffer.prepare((2,), dtype=torch.float32, device="cuda:0").fill_(0)
            buffer.upload()
            self.assertLessEqual(buffer.cache_info()["capacity_bytes"], 16)
            buffer.release_resources()
            torch.testing.assert_close(result, snapshot, rtol=0, atol=0)

    def test_dtype_and_device_changes_replace_context_and_keep_no_history(self):
        with MockCuda():
            buffer = uploads.PinnedUploadBuffer(128)
            for dtype, device in (
                (torch.float32, "cuda:0"),
                (torch.uint8, "cuda:0"),
                (torch.uint8, "cuda:1"),
            ):
                buffer.prepare((5,), dtype=dtype, device=device).fill_(4)
                output = buffer.upload()
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(buffer.cache_info()["device"], device)
                self.assertEqual(buffer.cache_info()["capacity_bytes"], 5 * dtype.itemsize)
            self.assertEqual(buffer.cache_info()["context_changes"], 3)

    def test_stream_switch_rejected_until_explicit_release(self):
        with MockCuda() as cuda:
            buffer = uploads.PinnedUploadBuffer()
            buffer.prepare((1,), dtype=torch.float32, device="cuda:0").fill_(1)
            cuda.stream = FakeStream("other")
            with self.assertRaises(RuntimeError):
                buffer.upload()
            with self.assertRaises(RuntimeError):
                buffer.prepare((1,), dtype=torch.float32, device="cuda:0")
            buffer.release_resources()
            buffer.prepare((1,), dtype=torch.float32, device="cuda:0").fill_(2)
            buffer.upload()

    def test_inference_context_switch_and_empty_cuda_shapes_need_no_new_semantics(self):
        with MockCuda():
            buffer = uploads.PinnedUploadBuffer()
            with torch.inference_mode():
                host = buffer.prepare((2,), dtype=torch.float32, device="cuda:0")
            host.fill_(7)
            output = buffer.upload()
            torch.testing.assert_close(output, torch.full((2,), 7.0), rtol=0, atol=0)
            empty = buffer.prepare((2, 0, 3), dtype=torch.float32, device="cuda:0")
            self.assertEqual(tuple(empty.shape), (2, 0, 3))
            self.assertEqual(tuple(buffer.upload().shape), (2, 0, 3))
            buffer.release_resources()

    def test_event_record_failure_synchronizes_only_own_stream(self):
        with MockCuda() as cuda:
            buffer = uploads.PinnedUploadBuffer()
            buffer.prepare((1,), dtype=torch.float32, device="cuda:0").fill_(1)
            with patch.object(buffer._event, "record", side_effect=RuntimeError("record failed")):
                with self.assertRaisesRegex(RuntimeError, "record failed"):
                    buffer.upload()
            self.assertEqual(cuda.stream.synchronizations, 1)
            buffer.release_resources()
            self.assertEqual(buffer.cache_info()["host_bytes"], 0)


class MatrixPackingTests(unittest.TestCase):
    def setUp(self):
        self.builder = shared.SharedParameterBuilder(acceleration="eager")

    def matrices(self):
        identity = np.eye(3, dtype=np.float32)
        batched = np.stack([identity * value for value in (1, 2, 3)])
        return [identity.copy(), identity[None].copy(), batched]

    def test_mixed_cpu_shapes_are_one_upload_with_original_shapes_and_values(self):
        raw = self.matrices()
        raw[1] = torch.from_numpy(raw[1].astype(np.float64))
        raw[2] = raw[2][:, :, ::-1]
        with MockCuda():
            matrices = self.builder._prepare_warp_matrices(raw, torch.device("cuda:0"))
            for value, matrix in zip(raw, matrices):
                expected = np.asarray(value).astype(np.float32)
                np.testing.assert_array_equal(matrix.numpy(), expected)
                self.assertEqual(tuple(matrix.shape), expected.shape)
            self.assertEqual(self.builder.last_info["warp_host_matrix_batches"], 1)
            self.assertEqual(self.builder.cache_info()["warp_upload"]["uploads"], 1)
            self.assertEqual(len({matrix.untyped_storage().data_ptr() for matrix in matrices}), 1)

    def test_same_object_edits_are_repacked_without_source_references(self):
        raw = self.matrices()
        source_ref = weakref.ref(raw[2])
        with MockCuda():
            first = self.builder._prepare_warp_matrices(raw, torch.device("cuda:0"))
            snapshot = first[2].clone()
            raw[2][:] += 0.125
            current = self.builder._prepare_warp_matrices(raw, torch.device("cuda:0"))
            np.testing.assert_array_equal(current[2].numpy(), raw[2])
            self.assertFalse(torch.equal(snapshot, current[2]))
            del raw
            gc.collect()
            self.assertIsNone(source_ref())

    def test_cpu_target_keeps_existing_stack_and_batched_alias_behavior(self):
        raw = self.matrices()
        with patch.object(
            self.builder._warp_upload, "prepare", side_effect=AssertionError("CPU staging")
        ):
            matrices = self.builder._prepare_warp_matrices(raw, torch.device("cpu"))
            self.assertEqual(self.builder.last_info["warp_host_matrix_batches"], 3)
            self.assertEqual(matrices[2].data_ptr(), torch.from_numpy(raw[2]).data_ptr())
            stacked = self.builder._prepare_warp_matrices(raw[:1] * 2, torch.device("cpu"))
            self.assertEqual(self.builder.last_info["warp_host_matrix_batches"], 1)
            self.assertNotEqual(stacked[0].data_ptr(), torch.from_numpy(raw[0]).data_ptr())

    def test_gpu_or_mixed_sources_use_original_conversion_fallback(self):
        raw = [np.eye(3, dtype=np.float32), torch.empty((3, 3), device="meta")]
        converted = (object(), object())
        with patch.object(shared, "_float_tensor", side_effect=converted) as convert:
            result = self.builder._prepare_warp_matrices(raw, torch.device("cuda:0"))
        self.assertEqual(result, converted)
        self.assertIs(convert.call_args_list[0].args[0], raw[0])
        self.assertIs(convert.call_args_list[1].args[0], raw[1])
        self.assertIsNone(self.builder.last_info["warp_host_matrix_batches"])

    def test_same_device_fallback_preserves_bounded_workspace_between_groups(self):
        with MockCuda():
            self.builder._prepare_warp_matrices(self.matrices(), torch.device("cuda:0"))
            before = self.builder.cache_info()["warp_upload"]
            raw = [torch.empty((3, 3), device="meta")]
            with patch.object(shared, "_float_tensor", return_value=object()):
                self.builder._prepare_warp_matrices(raw, torch.device("cuda:0"))
                same = self.builder.cache_info()["warp_upload"]
                self.assertEqual(same["capacity_bytes"], before["capacity_bytes"])
                self.assertEqual(same["reallocations"], before["reallocations"])
                self.builder._prepare_warp_matrices(raw, torch.device("cuda:1"))
                self.assertEqual(self.builder.cache_info()["warp_upload"]["capacity_bytes"], 0)

    def test_invalid_matrix_batches_fail_before_staging_or_device_allocation(self):
        shapes = ((0, 3, 3), (2, 3, 3), (1000000, 3, 3), (2, 3), (1, 1, 3, 3))
        for shape in shapes:
            # A zero-stride CPU view tests huge invalid B without allocating it.
            raw = np.broadcast_to(np.float32(1), shape)
            with self.subTest(shape=shape):
                with patch.object(
                    self.builder._warp_upload, "prepare", side_effect=AssertionError("allocated")
                ):
                    with self.assertRaises(ValueError):
                        self.builder.build_warp_grids(
                            params=[{"m_inv": raw}],
                            ads=[torch.empty(3, 4, 5, 7)],
                            boxes=[(0, 0, 6, 4)],
                            device="cuda:0",
                            batch=3,
                            warp_align_corners=False,
                        )

    def test_packed_geometry_matches_eager_math_and_old_grids_survive_next_frame(self):
        raw = self.matrices()
        parameters = [{"m_inv": value} for value in raw]
        ads = [torch.empty(3, 4, 7, 9) for _ in raw]
        boxes = [(1, 2, 7, 7), (2, 4, 8, 9), (3, 1, 9, 6)]
        reference = [
            shared.create_roi_grid_torch(
                (7, 9), torch.from_numpy(value), (5, 6), normalization="legacy", align_corners=False
            )
            for value in raw
        ]
        with MockCuda():
            first = self.builder.build_warp_grids(
                params=parameters,
                ads=ads,
                boxes=boxes,
                device="cuda:0",
                batch=3,
                warp_align_corners=False,
            )
            for output, expected in zip(first, reference):
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
            raw[0][0, 2] = 0.25
            second = self.builder.build_warp_grids(
                params=parameters,
                ads=ads,
                boxes=boxes,
                device="cuda:0",
                batch=3,
                warp_align_corners=False,
            )
            self.assertFalse(torch.equal(first[0], second[0]))
            torch.testing.assert_close(first[0], reference[0], rtol=0, atol=0)

    def test_gaussian_clear_keeps_workspace_and_full_release_clears_it(self):
        with MockCuda():
            self.builder._prepare_warp_matrices(self.matrices(), torch.device("cuda:0"))
            self.assertGreater(self.builder.cache_info()["warp_upload"]["capacity_bytes"], 0)
            self.builder.clear_cache()
            self.assertGreater(self.builder.cache_info()["warp_upload"]["capacity_bytes"], 0)
            self.builder.release_resources()
            info = self.builder.cache_info()["warp_upload"]
            self.assertEqual(info["host_bytes"], 0)
            self.assertEqual(info["device_bytes"], 0)
            self.assertEqual(info["capacity_bytes"], 0)
            self.assertEqual(self.builder.cache_info()["entries"], 0)


if __name__ == "__main__":
    unittest.main()
