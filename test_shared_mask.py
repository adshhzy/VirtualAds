"""CPU reference/ownership contracts for shared full-frame mask feathering."""

import unittest
import weakref

import numpy as np
import torch

from paste_shared_mask import SharedMaskBuilder


def feather_reference(mask, sigma=1.5, radius=3):
    """Independent NumPy tap accumulation with explicit replicated edges."""
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (coordinates / sigma) ** 2)
    weights = (weights / weights.sum()).astype(np.float32)
    weight_sum = np.float32(0)
    for weight in weights:
        weight_sum = np.float32(weight_sum + weight)
    height, width = mask.shape
    padded = np.pad(mask, ((0, 0), (radius, radius)), mode="edge")
    horizontal = np.zeros_like(mask, dtype=np.float32)
    for index, weight in enumerate(weights):
        horizontal += padded[:, index : index + width] * weight
    horizontal /= weight_sum
    padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode="edge")
    result = np.zeros_like(mask, dtype=np.float32)
    for index, weight in enumerate(weights):
        result += padded[index : index + height] * weight
    result /= weight_sum
    return np.clip(result, 0, 1)


def build(builder, patches, shape=(9, 13)):
    return builder.build(patches, height=shape[0], width=shape[1], device="cpu")


class SharedMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_union_maximum_clipping_and_input_ownership(self):
        source = np.arange(20, dtype=np.float32).reshape(4, 5) / 20
        overlap = torch.full((3, 4), 0.7)
        before_source, before_overlap = source.copy(), overlap.clone()
        patches = [
            {"bbox": (-2, -1, 3, 3), "mask": source},
            {"bbox": (1, 1, 5, 4), "mask": overlap},
            {"bbox": (10, 10, 12, 12), "mask": np.ones((2, 2), dtype=np.float32)},
        ]
        actual = build(SharedMaskBuilder(radius=0), patches, (4, 5))
        expected = np.zeros((4, 5), dtype=np.float32)
        expected[:3, :3] = source[1:, 2:]
        expected[1:4, 1:5] = np.maximum(expected[1:4, 1:5], 0.7)
        np.testing.assert_array_equal(actual.numpy(), expected)
        np.testing.assert_array_equal(source, before_source)
        torch.testing.assert_close(overlap, before_overlap, rtol=0, atol=0)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(actual.is_contiguous())

    def test_union_precedes_feather_and_duplicate_masks_do_not_add(self):
        left = np.zeros((9, 13), dtype=np.float32)
        right = left.copy()
        left[2:7, 1:8] = 1
        right[1:8, 5:12] = 0.8
        patches = [{"bbox": (0, 0, 13, 9), "mask": mask} for mask in (left, right, left)]
        actual = build(SharedMaskBuilder(acceleration="eager"), patches)
        expected = feather_reference(np.maximum(left, right))
        np.testing.assert_allclose(actual.numpy(), expected, rtol=0, atol=2e-7)
        wrong = np.maximum(feather_reference(left), feather_reference(right))
        self.assertGreater(float(np.max(np.abs(actual.numpy() - wrong))), 0.01)

    def test_empty_and_full_masks_preserve_replicated_frame_edges(self):
        for shape in ((1, 1), (1, 7), (8, 1), (4, 5)):
            with self.subTest(shape=shape):
                builder = SharedMaskBuilder()
                empty = build(builder, [], shape)
                torch.testing.assert_close(empty, torch.zeros_like(empty), rtol=0, atol=0)
                height, width = shape
                full = build(
                    builder,
                    [{"bbox": (0, 0, width, height), "mask": np.ones(shape, dtype=bool)}],
                    shape,
                )
                torch.testing.assert_close(full, torch.ones_like(full), rtol=0, atol=0)
                self.assertGreaterEqual(float(full.min()), 0)
                self.assertLessEqual(float(full.max()), 1)

    def test_dynamic_boxes_clear_old_values_and_return_borrowed_workspace(self):
        builder = SharedMaskBuilder(radius=0)
        first = build(builder, [{"bbox": (0, 0, 3, 3), "mask": np.ones((3, 3), np.float32)}])
        owned = first.clone()
        second = build(builder, [{"bbox": (10, 6, 13, 9), "mask": np.ones((3, 3), np.float32)}])
        self.assertIs(first, second)
        self.assertEqual(int(second[:3, :3].count_nonzero()), 0)
        self.assertEqual(int(second.count_nonzero()), 9)
        self.assertEqual(int(owned[:3, :3].count_nonzero()), 9)
        empty = build(builder, [])
        self.assertIs(empty, first)
        self.assertEqual(int(empty.count_nonzero()), 0)

    def test_weights_and_workspace_reuse_then_resolution_replacement(self):
        builder = SharedMaskBuilder()
        first = build(builder, [])
        kernel = builder._weights
        first_ref = weakref.ref(first)
        initial = builder.cache_info()
        self.assertEqual(initial["workspace_bytes"], 4 * 9 * 13 * 4)
        self.assertEqual(initial["patch_capacity_bytes"], 9 * 13 * 4)
        for _ in range(8):
            self.assertIs(build(builder, []), first)
        self.assertEqual(builder.cache_info()["workspace_allocations"], 1)
        self.assertEqual(builder.cache_info()["kernel_uploads"], 1)
        del first
        smaller = build(builder, [], (2, 3))
        self.assertIsNone(first_ref())
        self.assertEqual(smaller.shape, (2, 3))
        self.assertIs(builder._weights, kernel)
        self.assertEqual(builder.cache_info()["workspace_bytes"], 4 * 2 * 3 * 4)
        self.assertEqual(builder.cache_info()["workspace_allocations"], 2)
        self.assertEqual(builder.cache_info()["kernel_uploads"], 1)

    def test_release_drops_workspace_without_revoking_caller_tensor(self):
        builder = SharedMaskBuilder()
        output = build(builder, [{"bbox": (1, 1, 3, 3), "mask": np.ones((2, 2), np.float32)}])
        snapshot = output.clone()
        refs = [
            weakref.ref(value) for value in (builder._canvas, builder._temporary, builder._patch)
        ]
        builder.release_resources()
        self.assertTrue(all(reference() is None for reference in refs))
        self.assertEqual(builder.cache_info()["workspace_bytes"], 0)
        self.assertEqual(builder.cache_info()["kernel_bytes"], 0)
        self.assertIsNone(builder.cache_info()["shape"])
        torch.testing.assert_close(output, snapshot, rtol=0, atol=0)
        rebuilt = build(builder, [])
        self.assertIsNot(output, rebuilt)
        self.assertNotEqual(output.data_ptr(), rebuilt.data_ptr())
        torch.testing.assert_close(output, snapshot, rtol=0, atol=0)

    def test_visible_noncontiguous_readonly_and_integer_sources(self):
        numpy_source = np.arange(30, dtype=np.float64).reshape(5, 6) / 30
        numpy_source = numpy_source[::-1, ::-1]
        numpy_source.setflags(write=False)
        tensor_source = torch.arange(30, dtype=torch.float32).reshape(6, 5).t() / 30
        for source in (numpy_source, tensor_source, np.ones((5, 6), dtype=np.uint8)):
            with self.subTest(kind=type(source).__name__):
                actual = build(
                    SharedMaskBuilder(radius=0), [{"bbox": (-4, -3, 2, 2), "mask": source}], (2, 2)
                )
                expected = np.asarray(source)[3:, 4:].astype(np.float32)
                np.testing.assert_array_equal(actual.numpy(), expected)

    def test_completely_offscreen_patches_do_not_allocate_bbox_scratch(self):
        builder = SharedMaskBuilder(radius=0)
        source = np.ones((30, 40), dtype=np.float32)
        actual = build(builder, [{"bbox": (50, 50, 90, 80), "mask": source}], (2, 3))
        self.assertEqual(int(actual.count_nonzero()), 0)
        self.assertEqual(builder.cache_info()["patch_capacity_bytes"], 2 * 3 * 4)

    def test_custom_radius_and_sigma_match_reference(self):
        source = np.zeros((5, 7), dtype=np.float32)
        source[0, 0], source[3, 4] = 1, 0.4
        for radius, sigma in ((0, 1.5), (1, 0.8), (4, 2.2)):
            with self.subTest(radius=radius, sigma=sigma):
                actual = build(
                    SharedMaskBuilder(sigma=sigma, radius=radius),
                    [{"bbox": (0, 0, 7, 5), "mask": source}],
                    source.shape,
                )
                np.testing.assert_allclose(
                    actual.numpy(), feather_reference(source, sigma, radius), rtol=0, atol=2e-7
                )

    def test_full_mask_is_exact_for_custom_kernel_and_protects_uint8_background(self):
        shape = (17, 131)
        frame = torch.arange(shape[0] * shape[1] * 3).reshape(*shape, 3).remainder(256).float()
        for radius, sigma in ((0, 1.5), (1, 0.8), (3, 1.5), (4, 2.2), (6, 3.0)):
            with self.subTest(radius=radius, sigma=sigma):
                mask = build(
                    SharedMaskBuilder(sigma=sigma, radius=radius),
                    [{"bbox": (0, 0, shape[1], shape[0]), "mask": np.ones(shape, np.uint8)}],
                    shape,
                )
                torch.testing.assert_close(mask, torch.ones_like(mask), rtol=0, atol=0)
                actual = (frame * mask[..., None]).to(torch.uint8)
                torch.testing.assert_close(actual, frame.to(torch.uint8), rtol=0, atol=0)

    def test_bad_shapes_values_and_bbox_metadata_fail_before_reusing_output(self):
        valid = {"bbox": (0, 0, 2, 2), "mask": np.ones((2, 2), dtype=np.float32)}
        bad = [
            {**valid, "bbox": (0, 0, 3, 2)},
            {**valid, "bbox": (0, 0, 0, 2)},
            {**valid, "bbox": (0.0, 0, 2, 2)},
            {**valid, "bbox": (False, 0, 2, 2)},
            {**valid, "mask": np.ones((2, 2, 1))},
            {**valid, "mask": np.ones((2, 2), dtype=np.complex64)},
            {**valid, "mask": np.full((2, 2), 255, dtype=np.uint8)},
            {**valid, "mask": torch.full((2, 2), float("nan"))},
            {**valid, "mask": np.full((2, 2), -0.1)},
            {**valid, "mask": [[1, 1], [1, 1]]},
        ]
        builder = SharedMaskBuilder(radius=0)
        previous = build(builder, [valid])
        snapshot = previous.clone()
        for item in bad:
            with self.subTest(item=item):
                with self.assertRaises((TypeError, ValueError)):
                    build(builder, [item])
                torch.testing.assert_close(previous, snapshot, rtol=0, atol=0)

    def test_bad_configuration_and_cpu_triton_request_are_explicit(self):
        for kwargs in (
            {"sigma": 0},
            {"sigma": float("nan")},
            {"sigma": True},
            {"radius": -1},
            {"radius": 1.5},
            {"radius": True},
            {"acceleration": "unknown"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SharedMaskBuilder(**kwargs)
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            build(SharedMaskBuilder(acceleration="triton"), [])
        for height, width in ((0, 3), (2, -1), (True, 3), (2.5, 3)):
            with self.subTest(height=height, width=width), self.assertRaises(ValueError):
                SharedMaskBuilder().build([], height=height, width=width, device="cpu")


if __name__ == "__main__":
    unittest.main()
