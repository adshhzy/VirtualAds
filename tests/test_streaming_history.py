"""CPU regression tests for bounded camera history and Gaussian cache lifetime.

Run with the project's Python: ``python -m unittest test_streaming_history -v``.
The device-switch test mocks uploads; this suite never creates a CUDA context.
"""

from __future__ import annotations

import gc
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from scipy.ndimage import median_filter
import torch

from virtualads.paste_camera_filter import BoundedCameraHistory, compute_camera_filter_shared
from virtualads.paste_shared_params import SharedParameterBuilder


def camera_record(frame_id, dtype=np.float64):
    return {
        "K": np.eye(3, dtype=np.float32),
        "w2c": object(),
        "dist_coeffs": np.asarray([frame_id % 5, frame_id * 0.125, 0, 0, 0], dtype=dtype),
    }


class CameraOwner:
    def __init__(self, history):
        self.paste_data_stack = history
        self.calls = []

    def compute_camera_motion_rate(self, poses):
        self.calls.append(poses)
        return 0.25


class GaussianOwner:
    def __init__(self):
        self.gaussian_kernel_cache_version = 0
        self.sigma_calls = []
        self.kernel_calls = []

    def motion_rate_to_sigma(self, motion_rate):
        self.sigma_calls.append(motion_rate)
        return motion_rate

    def create_gaussian_kernel(self, sigma):
        self.kernel_calls.append(sigma)
        axis = np.asarray([1.0, 2.0 + float(sigma), 1.0], dtype=np.float64)
        kernel = np.outer(axis, axis)
        return (kernel / kernel.sum()).astype(np.float32)


class BoundedHistoryTests(unittest.TestCase):
    def test_capacity_replacement_and_explicit_seek_reset(self):
        history = BoundedCameraHistory(capacity=3)
        for frame_id in (1, 2, 4, 7):
            history[frame_id] = camera_record(frame_id)
        self.assertEqual(list(history), [2, 4, 7])
        self.assertEqual(history.latest_frame_id, 7)
        replacement = camera_record(7)
        history[7] = replacement
        self.assertIs(history[7], replacement)
        self.assertEqual(list(history), [2, 4, 7])
        for old_id in (0, 2, 6):
            with self.subTest(old_id=old_id), self.assertRaisesRegex(ValueError, "reset"):
                history[old_id] = camera_record(old_id)
        del history[7]
        with self.assertRaises(ValueError):
            history[6] = camera_record(6)
        history.reset()
        self.assertEqual(len(history), 0)
        self.assertIsNone(history.latest_frame_id)
        history[-2] = camera_record(-2)
        history.clear()
        history[-3] = camera_record(-3)
        self.assertEqual(list(history), [-3])

    def test_invalid_capacity_keys_and_record_types(self):
        for capacity in (0, -1, True, 1.5, "9"):
            with self.subTest(capacity=capacity), self.assertRaises((TypeError, ValueError)):
                BoundedCameraHistory(capacity)
        history = BoundedCameraHistory()
        for frame_id in (True, np.bool_(False), 1.5, "1"):
            with self.subTest(frame_id=frame_id), self.assertRaises(TypeError):
                history[frame_id] = camera_record(1)
        with self.assertRaises(TypeError):
            history[1] = None
        history[np.int64(1)] = camera_record(1)
        self.assertEqual(list(history), [1])

    def test_window_lookup_does_not_iterate_history(self):
        class NoIterationHistory(BoundedCameraHistory):
            def __iter__(self):
                raise AssertionError("window lookup must not scan all history keys")

        history = NoIterationHistory(capacity=20)
        for frame_id in range(20):
            history[frame_id] = camera_record(frame_id)
        self.assertEqual([key for key, _ in history.items_in_window(10, 2)], [8, 9, 10, 11, 12])
        owner = CameraOwner(history)
        compute_camera_filter_shared(owner, camera_record(10), 10)
        self.assertEqual(len(owner.calls), 1)

    def test_eviction_releases_records_without_touching_external_history(self):
        external = {1: camera_record(1)}
        retained_pose = external[1]["w2c"]
        # A weak-referenceable payload makes record ownership observable.
        payload = GaussianOwner()
        payload_ref = weakref.ref(payload)
        history = BoundedCameraHistory(capacity=1)
        history[1] = {"K": payload}
        del payload
        self.assertIsNotNone(payload_ref())
        history[2] = camera_record(2)
        self.assertIsNone(payload_ref())
        self.assertIs(external[1]["w2c"], retained_pose)
        self.assertEqual(len(external), 1)

    def test_dense_future_and_sparse_windows_match_original_median(self):
        cases = (
            (tuple(range(10, 19)), 14),  # Current frame plus four future records.
            ((1, 3, 4, 7, 8, 10, 14), 7),
            ((-8, -4, -3, 0, 2), -3),
        )
        for ids, current in cases:
            for policy in ("legacy", "corrected"):
                for dtype in (np.float32, np.float64):
                    with self.subTest(ids=ids, current=current, policy=policy, dtype=dtype):
                        records = {key: camera_record(key, dtype) for key in ids}
                        history = BoundedCameraHistory(capacity=len(records))
                        history.update(records)
                        plain_owner = CameraOwner(records)
                        bounded_owner = CameraOwner(history)
                        selected = [key for key in ids if abs(key - current) <= 4]
                        values = np.stack([records[key]["dist_coeffs"] for key in selected])
                        row = (
                            selected.index(current) if policy == "corrected" else len(selected) - 1
                        )
                        expected = median_filter(values, size=(7, 1), mode="reflect")[row]
                        actual, motion = compute_camera_filter_shared(
                            bounded_owner,
                            camera_record(current, dtype),
                            current,
                            index_policy=policy,
                        )
                        ordinary, ordinary_motion = compute_camera_filter_shared(
                            plain_owner, camera_record(current, dtype), current, index_policy=policy
                        )
                        np.testing.assert_array_equal(actual, expected)
                        np.testing.assert_array_equal(actual, ordinary)
                        self.assertEqual(actual.dtype, expected.dtype)
                        self.assertEqual(motion, ordinary_motion)
                        self.assertEqual(len(bounded_owner.calls), 1)
                        self.assertEqual(bounded_owner.calls, plain_owner.calls)

    def test_partial_records_fallback_and_absent_current_keep_old_behavior(self):
        history = BoundedCameraHistory()
        history[1] = camera_record(1)
        history[2] = {"K": np.eye(3)}
        history[3] = camera_record(3)
        owner = CameraOwner(history)
        data = camera_record(2)
        fallback = data["dist_coeffs"]
        result, motion = compute_camera_filter_shared(owner, data, 2)
        self.assertIs(result, fallback)
        self.assertIs(data["dist_coeffs"], fallback)
        self.assertEqual(motion, 0.0)
        self.assertEqual(owner.calls, [])
        history[4] = camera_record(4)
        for policy, error in (("corrected", ValueError), ("legacy", IndexError)):
            with self.subTest(policy=policy), self.assertRaises(error):
                compute_camera_filter_shared(owner, data, 2, index_policy=policy)

    def test_plain_mapping_still_validates_out_of_window_keys(self):
        history = {key: camera_record(key) for key in (1, 2, 3)}
        history["10000"] = {"K": np.eye(3)}
        owner = CameraOwner(history)
        with self.assertRaisesRegex(TypeError, "frame ID"):
            compute_camera_filter_shared(owner, camera_record(2), 2)
        self.assertEqual(owner.calls, [])
        self.assertEqual(len(history), 4)

    def test_live_coefficient_edits_and_filtered_dictionary_update(self):
        history = BoundedCameraHistory()
        for key in (1, 2, 3):
            history[key] = camera_record(key)
        owner = CameraOwner(history)
        data = camera_record(2)
        first, _ = compute_camera_filter_shared(owner, data, 2)
        self.assertIs(data["dist_coeffs"], first)
        for record in history.values():
            record["dist_coeffs"][:] = 17
        second, _ = compute_camera_filter_shared(owner, data, 2)
        np.testing.assert_array_equal(second, np.full(5, 17.0))
        self.assertFalse(np.array_equal(first, second))
        self.assertEqual(len(owner.calls), 2)


class KernelLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.builder = SharedParameterBuilder(acceleration="eager", cache_limit=3)
        self.device = torch.device("cpu")

    def render_kernel(self, owner, sigma):
        with torch.inference_mode():
            return self.builder._motion_value(owner, sigma, self.device)

    def test_exact_sigma_lru_stays_bounded_and_preserves_callback_types(self):
        owner = GaussianOwner()
        first = self.render_kernel(owner, np.float32(0.5))
        self.assertIs(self.render_kernel(owner, np.float32(0.5)), first)
        self.assertEqual(len(owner.sigma_calls), 2)
        self.assertEqual(len(owner.kernel_calls), 1)
        self.assertIsInstance(owner.kernel_calls[0], np.float32)
        self.render_kernel(owner, np.float64(0.5))
        self.assertEqual(len(owner.kernel_calls), 2)
        for sigma in (0.0, -0.0):
            self.render_kernel(owner, sigma)
        self.assertEqual(len(owner.kernel_calls), 4)
        for index in range(25):
            value = self.render_kernel(owner, 0.75 + index * 0.00001)
            self.assertLessEqual(self.builder.cache_info()["entries"], 3)
            self.assertIsNone(value.grad_fn)
            self.assertFalse(value.requires_grad)
        expected = owner.create_gaussian_kernel(0.75 + 24 * 0.00001)
        np.testing.assert_array_equal(value.numpy(), np.tile(expected[None, None], (4, 1, 1, 1)))

    def test_switch_owner_releases_previous_owner_and_kernel(self):
        old_owner = GaussianOwner()
        owner_ref = weakref.ref(old_owner)
        old_value = self.render_kernel(old_owner, 0.5)
        value_ref = weakref.ref(old_value)
        del old_owner, old_value
        self.assertIsNotNone(owner_ref())
        self.render_kernel(GaussianOwner(), 0.5)
        gc.collect()
        self.assertIsNone(owner_ref())
        self.assertIsNone(value_ref())
        self.assertEqual(self.builder.cache_info()["entries"], 1)

    def test_versions_callbacks_and_configuration_replace_entire_context(self):
        owner = GaussianOwner()
        for sigma in (0.5, 0.6):
            self.render_kernel(owner, sigma)
        owner.gaussian_kernel_cache_version += 1
        self.render_kernel(owner, 0.5)
        self.assertEqual(self.builder.cache_info()["entries"], 1)
        self.assertEqual(len(owner.kernel_calls), 3)
        self.builder.kernel_cache_version += 1
        self.render_kernel(owner, 0.5)
        self.assertEqual(len(owner.kernel_calls), 4)

        callback_owner = GaussianOwner()
        callback_ref = weakref.ref(callback_owner)
        owner.create_gaussian_kernel = callback_owner.create_gaussian_kernel
        self.render_kernel(owner, 0.5)
        del callback_owner
        self.assertIsNotNone(callback_ref())
        owner.create_gaussian_kernel = GaussianOwner.create_gaussian_kernel.__get__(owner)
        self.render_kernel(owner, 0.5)
        gc.collect()
        self.assertIsNone(callback_ref())
        self.assertEqual(self.builder.cache_info()["entries"], 1)

        self.builder.enable_separable = True
        spec = self.render_kernel(owner, 0.5)
        self.assertIsNotNone(spec.factors)
        for tensor in (spec.kernel, *spec.factors):
            self.assertIsNone(tensor.grad_fn)
            self.assertFalse(tensor.requires_grad)
        self.builder.cache_limit = 1
        self.render_kernel(owner, 0.6)
        self.assertEqual(self.builder.cache_info()["entries"], 1)

    def test_device_context_switch_drops_old_entries_without_cuda(self):
        owner = GaussianOwner()
        self.render_kernel(owner, 0.5)
        # Only the metadata context changes; the upload is replaced with a CPU
        # sentinel, so no CUDA allocation, initialization or synchronization occurs.
        sentinel = object()
        with patch(
            "virtualads.paste_shared_params._upload_motion_kernel", return_value=sentinel
        ) as upload:
            value = self.builder._motion_value(owner, 0.5, torch.device("cuda:0"))
        self.assertIs(value, sentinel)
        self.assertEqual(upload.call_args.args[-1], torch.device("cuda:0"))
        self.assertEqual(self.builder.cache_info()["entries"], 1)
        self.render_kernel(owner, 0.5)
        self.assertEqual(len(owner.kernel_calls), 3)

    def test_disabling_cache_releases_references_and_repeats_callback(self):
        owner = GaussianOwner()
        owner_ref = weakref.ref(owner)
        self.render_kernel(owner, 0.5)
        self.builder.cache_kernels = False
        self.render_kernel(owner, 0.5)
        self.render_kernel(owner, 0.5)
        self.assertEqual(self.builder.cache_info()["entries"], 0)
        self.assertEqual(len(owner.kernel_calls), 3)
        del owner
        gc.collect()
        self.assertIsNone(owner_ref())

    def test_clear_and_release_preserve_caller_outputs_and_allow_reuse(self):
        for method in (self.builder.clear_cache, self.builder.release_resources):
            with self.subTest(method=method.__name__):
                owner = GaussianOwner()
                owner_ref = weakref.ref(owner)
                result = self.render_kernel(owner, 0.5)
                snapshot = result.clone()
                del owner
                method()
                gc.collect()
                self.assertIsNone(owner_ref())
                self.assertEqual(self.builder.cache_info()["entries"], 0)
                self.assertEqual(self.builder.cache_info()["hits"], 0)
                self.assertEqual(self.builder.cache_info()["misses"], 0)
                torch.testing.assert_close(result, snapshot, rtol=0, atol=0)
        self.assertIsInstance(self.render_kernel(GaussianOwner(), 0.5), torch.Tensor)

    def test_prepare_keeps_camera_geometry_live_and_does_not_cache_graphs(self):
        owner = GaussianOwner()
        data = camera_record(1)
        coefficients = np.zeros(5, dtype=np.float32)
        mask = torch.ones((1, 1, 5, 7), dtype=torch.float32)

        def prepare():
            return self.builder.prepare(
                owner,
                data,
                1,
                height=5,
                width=7,
                device="cpu",
                mask_canvas=mask,
                camera_result=(coefficients, 0.5),
            )

        first = prepare()
        data["K"][0, 0] = 2.0
        coefficients[0] = 0.05
        second = prepare()
        self.assertFalse(torch.equal(first[1], second[1]))
        self.assertIs(first[3], second[3])
        first_grid_ref = weakref.ref(first[1])
        del first
        gc.collect()
        self.assertIsNone(first_grid_ref())
        self.assertIsNone(second[1].grad_fn)
        self.assertFalse(second[1].requires_grad)
        self.assertEqual(self.builder.cache_info()["entries"], 1)


if __name__ == "__main__":
    unittest.main()
