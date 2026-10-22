"""CPU and mocked-CUDA coverage for visible-mask packing and resource lifetime.

These tests exercise real CPU conversion/union/blur and fake allocation events.
They do not substitute for the separate serial CUDA image/reference regression.
"""

from contextlib import contextmanager
import gc
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch

from paste_shared_mask import SharedMaskBuilder
from paste_upload_buffer import PinnedUploadBuffer
from test_shared_upload import MockCuda


@contextmanager
def mocked_cuda():
    """Keep builder allocations and kernels on CPU while exercising CUDA staging."""
    original_to = torch.Tensor.to

    def to(value, *args, **kwargs):
        device = kwargs.get("device")
        if device is not None and torch.device(device).type == "cuda":
            kwargs["device"] = "cpu"
        return original_to(value, *args, **kwargs)

    with MockCuda() as cuda, patch.object(torch.Tensor, "to", new=to):
        yield cuda


def build(builder, items, shape=(7, 9), device="cuda:0"):
    return builder.build(items, height=shape[0], width=shape[1], device=device)


def item(source, origin=(0, 0)):
    x, y = origin
    return {"bbox": (x, y, x + source.shape[1], y + source.shape[0]), "mask": source}


class MaskUploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_one_upload_covers_clipped_overlapping_patches_before_feather(self):
        source = np.linspace(0, 1, 30, dtype=np.float32).reshape(5, 6)
        patches = [
            item(source, (-2, -1)),
            item(np.full((4, 5), 0.75), (2, 3)),
            item(np.ones((3, 3), np.uint8), (30, 30)),
        ]
        for radius in (0, 1, 3):
            with self.subTest(radius=radius), mocked_cuda():
                builder = SharedMaskBuilder(radius=radius, acceleration="eager")
                reference = SharedMaskBuilder(radius=radius, acceleration="eager")
                reference._batch_upload_enabled = False
                actual = build(builder, patches)
                expected = build(reference, patches)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                info = builder.cache_info()["patch_upload"]
                self.assertEqual(info["uploads"], 1)
                self.assertEqual(info["capacity_bytes"], (4 * 4 + 4 * 5) * 4)
                self.assertEqual(reference.cache_info()["patch_upload"]["uploads"], 0)

    def test_dtype_stride_readonly_and_byte_order_match_old_conversion(self):
        base = np.linspace(0, 1, 30, dtype=np.float64).reshape(5, 6)
        readonly = base[::-1, ::-1]
        readonly.setflags(write=False)
        sources = [
            readonly,
            base.astype(">f8"),
            base.astype(np.longdouble),
            base.astype(np.float16),
            (base > 0.5),
            (base > 0.5).astype(np.uint64),
            torch.from_numpy(base.copy()).t().contiguous().t(),
            torch.from_numpy(base.copy()).bfloat16(),
            torch._neg_view(-torch.from_numpy(base.copy())),
            torch.from_numpy(base.copy()).requires_grad_(),
        ]
        with mocked_cuda():
            builder = SharedMaskBuilder(radius=0)
            reference = SharedMaskBuilder(radius=0)
            reference._batch_upload_enabled = False
            for source in sources:
                with self.subTest(dtype=source.dtype, type=type(source)):
                    actual = build(builder, [item(source, (-2, -1))]).clone()
                    expected = build(reference, [item(source, (-2, -1))])
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertFalse(actual.requires_grad)
                    self.assertIsNone(actual.grad_fn)

    def test_live_values_and_bbox_are_not_cached_and_sources_are_not_retained(self):
        source = np.zeros((3, 4), np.float32)
        source_ref = weakref.ref(source)
        with mocked_cuda() as cuda:
            builder = SharedMaskBuilder(radius=0)
            patches = [item(source, (1, 1))]
            first = build(builder, patches).clone()
            first_info = builder.cache_info()["patch_upload"]
            source.fill(0.5)
            patches[0]["bbox"] = (4, 4, 8, 7)
            second = build(builder, patches)
            self.assertFalse(torch.equal(first, second))
            self.assertEqual(float(second[4:7, 4:8].sum()), 6.0)
            self.assertEqual(float(second[:4].sum()), 0)
            self.assertEqual(builder.cache_info()["patch_upload"]["reallocations"], 1)
            self.assertEqual(first_info["capacity_bytes"], 3 * 4 * 4)
            self.assertIn(("wait",), cuda.log)
            del patches, source
            gc.collect()
            self.assertIsNone(source_ref())

    def test_validation_precedes_staging_and_preserves_previous_borrowed_output(self):
        valid = item(np.ones((2, 3), np.float32))
        with mocked_cuda():
            builder = SharedMaskBuilder(radius=0)
            previous = build(builder, [valid])
            snapshot = previous.clone()
            count = builder.cache_info()["patch_upload"]["uploads"]
            for invalid in (
                item(np.full((2, 3), np.nan)),
                item(np.full((2, 3), 1.01)),
                item(np.full((2, 3), -0.01)),
                {**valid, "bbox": (0, 0, 2, 2)},
            ):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ValueError):
                        build(builder, [valid, invalid])
                    self.assertEqual(builder.cache_info()["patch_upload"]["uploads"], count)
                    torch.testing.assert_close(previous, snapshot, rtol=0, atol=0)

    def test_offscreen_invalid_values_are_ignored_and_only_visible_area_is_packed(self):
        source = np.full((30, 40), np.nan, np.float64)
        source[28:, 37:] = 0.75
        with mocked_cuda():
            builder = SharedMaskBuilder(radius=0)
            actual = build(builder, [item(source, (-37, -28))], (2, 3))
            torch.testing.assert_close(actual, torch.full((2, 3), 0.75), rtol=0, atol=0)
            self.assertEqual(builder.cache_info()["patch_upload"]["capacity_bytes"], 24)

    def test_empty_or_disabled_path_does_not_allocate_and_retains_only_bounded_storage(self):
        with mocked_cuda():
            builder = SharedMaskBuilder(radius=0)
            build(builder, [])
            self.assertEqual(builder.cache_info()["patch_upload"]["reallocations"], 0)
            source = np.ones((2, 3), np.uint8)
            build(builder, [item(source)])
            previous = builder.cache_info()["patch_upload"]
            builder._batch_upload_enabled = False
            build(builder, [item(source)])
            build(builder, [])
            current = builder.cache_info()["patch_upload"]
            self.assertEqual(current["uploads"], 1)
            self.assertEqual(current["capacity_bytes"], previous["capacity_bytes"])

    def test_cpu_target_never_uses_staging_and_releases_prior_device_context(self):
        with mocked_cuda():
            builder = SharedMaskBuilder(radius=0)
            patches = [item(np.ones((2, 3), np.uint8))]
            build(builder, patches)
            self.assertGreater(builder.cache_info()["patch_upload"]["host_bytes"], 0)
            with patch.object(builder._patch_upload, "prepare", side_effect=AssertionError("CPU")):
                actual = build(builder, patches, device="cpu")
            self.assertEqual(float(actual.sum()), 6)
            self.assertEqual(builder.cache_info()["patch_upload"]["host_bytes"], 0)
            self.assertEqual(builder.cache_info()["patch_upload"]["device_bytes"], 0)

    def test_mixed_device_sources_decline_packing_without_touching_inputs(self):
        builder = SharedMaskBuilder()
        sources = [np.ones((2, 3), np.uint8), torch.empty((2, 3), device="meta")]
        patches = [((0, 0, 3, 2), source) for source in sources]
        with patch.object(builder._patch_upload, "prepare", side_effect=AssertionError("mixed")):
            self.assertIsNone(builder._upload_host_patches(patches, torch.device("cuda:0")))

    def test_negative_bit_tensor_uses_original_device_copy_instead_of_cpu_staging(self):
        # CPU copy and CUDA copy have different logical-view handling on the
        # supported torch build. Preserve the old path even for mixed inputs.
        source = torch._neg_view(-torch.full((2, 3), 0.5))
        patches = [item(np.ones((2, 3), np.uint8)), item(source, (3, 0))]
        with mocked_cuda():
            builder = SharedMaskBuilder(radius=0)
            with patch.object(builder._patch_upload, "prepare", side_effect=AssertionError("view")):
                actual = build(builder, patches)
            self.assertEqual(float(actual.sum()), 9)
            self.assertEqual(builder.cache_info()["patch_upload"]["uploads"], 0)

    def test_oversize_host_is_retired_when_following_frames_have_no_masks(self):
        with mocked_cuda():
            builder = SharedMaskBuilder(radius=0)
            builder._patch_upload = PinnedUploadBuffer(max_cached_bytes=16)
            build(builder, [item(np.ones((2, 3), np.uint8))])
            self.assertEqual(builder.cache_info()["patch_upload"]["temporary_host_bytes"], 24)
            build(builder, [])
            self.assertEqual(builder.cache_info()["patch_upload"]["temporary_host_bytes"], 0)
            self.assertEqual(builder.cache_info()["patch_upload"]["host_bytes"], 0)

    def test_over_budget_request_is_temporary_and_next_small_request_is_bounded(self):
        with mocked_cuda():
            builder = SharedMaskBuilder(radius=0)
            builder._patch_upload = PinnedUploadBuffer(max_cached_bytes=16)
            output = build(builder, [item(np.ones((2, 3), np.uint8))]).clone()
            info = builder.cache_info()["patch_upload"]
            self.assertEqual(info["capacity_bytes"], 0)
            self.assertEqual(info["temporary_host_bytes"], 24)
            self.assertEqual(info["device_bytes"], 0)
            self.assertEqual(info["oversize_uploads"], 1)
            build(builder, [item(np.zeros((1, 2), np.uint8))])
            info = builder.cache_info()["patch_upload"]
            self.assertEqual(info["capacity_bytes"], 8)
            self.assertEqual(info["temporary_host_bytes"], 0)
            self.assertEqual(float(output.sum()), 6)

    def test_failed_conversion_recovers_without_uploading_partial_host_values(self):
        patches = [item(np.ones((2, 3), np.uint8))]
        with mocked_cuda():
            builder = SharedMaskBuilder(radius=0)
            with patch("paste_shared_mask.np.copyto", side_effect=RuntimeError("host copy")):
                with self.assertRaisesRegex(RuntimeError, "host copy"):
                    build(builder, patches)
            self.assertEqual(builder.cache_info()["patch_upload"]["uploads"], 0)
            actual = build(builder, patches)
            self.assertEqual(float(actual.sum()), 6)
            self.assertEqual(builder.cache_info()["patch_upload"]["reallocations"], 1)

    def test_device_switch_keeps_one_context_and_release_waits_for_copy(self):
        patches = [item(np.ones((2, 3), np.uint8))]
        with mocked_cuda() as cuda:
            builder = SharedMaskBuilder(radius=0)
            first = build(builder, patches)
            snapshot = first.clone()
            second = build(builder, patches, device="cuda:1")
            info = builder.cache_info()["patch_upload"]
            self.assertEqual(info["device"], "cuda:1")
            self.assertEqual(info["host_bytes"], 24)
            self.assertIsNot(first, second)
            builder.release_resources()
            info = builder.cache_info()["patch_upload"]
            self.assertEqual(cuda.log[-1], ("wait",))
            self.assertEqual(info["host_bytes"], 0)
            self.assertEqual(info["device_bytes"], 0)
            self.assertEqual(builder.cache_info()["workspace_bytes"], 0)
            torch.testing.assert_close(first, snapshot, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
