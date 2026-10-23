"""CPU contracts for uint8 staging and ImageRenderer workspace ownership.

The upload helper's CPU target exercises copying/conversion without CUDA. The
ImageRenderer integration tests opt into that target through a private test
seam; they do not claim to validate CUDA events, PCIe transfers or GPU kernels.
"""

from copy import deepcopy
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np
import torch

from virtualads import paste_frame_upload as uploads
from virtualads import paste_renderer as api
from virtualads.paste_render_executor import _as_tensor
from tests.test_render_img import CameraOwner, eager_initialize, make_inputs


class FrameUploadBufferTests(unittest.TestCase):
    def buffer(self, **kwargs):
        buffer = uploads.FrameUploadBuffer(**kwargs)
        self.addCleanup(buffer.release_resources)
        return buffer

    def test_exact_uint8_conversion_for_all_contiguous_layouts(self):
        hwc = np.arange(8 * 11 * 3, dtype=np.uint8).reshape(8, 11, 3)
        layouts = (
            hwc,
            hwc.transpose(2, 0, 1).copy(),
            hwc[None],
            hwc.transpose(2, 0, 1)[None].copy(),
        )
        buffer = self.buffer()
        for source in layouts:
            for frame in (source, torch.from_numpy(source)):
                with self.subTest(shape=frame.shape, tensor=isinstance(frame, torch.Tensor)):
                    self.assertTrue(buffer._contiguous_cpu_uint8(frame))
                    output = buffer._prepare_uint8(frame, torch.device("cpu"))
                    self.assertEqual(output.dtype, torch.float32)
                    self.assertTrue(output.is_contiguous())
                    np.testing.assert_array_equal(output.numpy(), source.astype(np.float32))
                    np.testing.assert_array_equal(np.asarray(frame), source)

    def test_each_frame_reads_live_pixels_and_reuses_float_capacity(self):
        buffer = self.buffer()
        frame = np.zeros((8, 11, 3), dtype=np.uint8)
        original = frame.copy()
        first = buffer._prepare_uint8(frame, torch.device("cpu"))
        pointer = first.data_ptr()
        allocations = buffer.cache_info()["float32_allocations"]
        np.testing.assert_array_equal(frame, original)
        frame.fill(237)
        second = buffer._prepare_uint8(frame, torch.device("cpu"))
        self.assertEqual(second.data_ptr(), pointer)
        self.assertEqual(buffer.cache_info()["float32_allocations"], allocations)
        np.testing.assert_array_equal(second.numpy(), frame.astype(np.float32))
        # The staging API is deliberately borrowed. Independent render outputs
        # are checked separately below, rather than promising independent inputs.
        self.assertTrue(torch.all(first == 237))

    def test_readonly_numpy_and_source_references_are_not_retained(self):
        buffer = self.buffer()
        for tensor in (False, True):
            frame = np.arange(60, dtype=np.uint8).reshape(4, 5, 3).copy()
            expected = frame.astype(np.float32)
            if tensor:
                frame = torch.from_numpy(frame)
            else:
                frame.flags.writeable = False
            reference = weakref.ref(frame)
            output = buffer._prepare_uint8(frame, torch.device("cpu"))
            np.testing.assert_array_equal(output.numpy(), expected)
            del frame
            self.assertIsNone(reference())

    def test_dynamic_shapes_keep_one_bounded_float_allocation(self):
        buffer = self.buffer(max_float32_bytes=128)
        first = buffer._prepare_uint8(np.zeros((2, 4, 3), np.uint8), torch.device("cpu"))
        self.assertEqual(buffer.cache_info()["float32_bytes"], 128)
        self.assertEqual(buffer.cache_info()["float32_allocations"], 1)
        smaller = buffer._prepare_uint8(np.zeros((2, 2, 3), np.uint8), torch.device("cpu"))
        self.assertEqual(smaller.data_ptr(), first.data_ptr())
        self.assertEqual(buffer.cache_info()["float32_allocations"], 1)
        oversized = buffer._prepare_uint8(np.zeros((3, 4, 3), np.uint8), torch.device("cpu"))
        info = buffer.cache_info()
        self.assertEqual(info["float32_bytes"], 0)
        self.assertEqual(info["last_temporary_float32_bytes"], 144)
        self.assertEqual(info["float32_oversize_allocations"], 1)
        self.assertIsNone(buffer._float_workspace)
        reference = weakref.ref(oversized)
        del oversized
        self.assertIsNone(reference())
        buffer._prepare_uint8(np.zeros((2, 2, 3), np.uint8), torch.device("cpu"))
        self.assertEqual(buffer.cache_info()["float32_allocations"], 2)
        buffer.release_resources()
        info = buffer.cache_info()
        self.assertEqual(info["float32_bytes"], 0)
        self.assertEqual(info["float32_allocations"], 2)
        self.assertEqual(info["last_temporary_float32_bytes"], 0)

    def test_disabled_float_cache_keeps_no_working_set(self):
        buffer = self.buffer(max_float32_bytes=0)
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        first = buffer._prepare_uint8(frame, torch.device("cpu"))
        second = buffer._prepare_uint8(frame, torch.device("cpu"))
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(buffer.cache_info()["float32_bytes"], 0)
        self.assertEqual(buffer.cache_info()["float32_oversize_allocations"], 2)

    def test_float_storage_can_be_reused_after_initial_inference_mode_call(self):
        buffer = self.buffer()
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        with torch.inference_mode():
            first = buffer._prepare_uint8(frame, torch.device("cpu"))
        frame.fill(39)
        second = buffer._prepare_uint8(frame, torch.device("cpu"))
        self.assertEqual(first.data_ptr(), second.data_ptr())
        np.testing.assert_array_equal(second.numpy(), frame.astype(np.float32))

    def test_unsupported_input_modes_preserve_original_converter(self):
        hwc = np.arange(8 * 11 * 3, dtype=np.uint8).reshape(8, 11, 3)
        frames = (
            hwc[:, ::2],
            hwc[:, ::-1],
            hwc.transpose(2, 0, 1),
            hwc.astype(np.float32),
            hwc.astype(np.float64),
            torch.from_numpy(hwc).transpose(0, 1),
            torch.ones(4, dtype=torch.float32),
            torch.empty((2, 3, 3), device="meta", dtype=torch.uint8),
        )
        buffer = self.buffer()
        for frame in frames:
            with self.subTest(shape=frame.shape, dtype=frame.dtype):
                self.assertFalse(buffer._contiguous_cpu_uint8(frame))
                buffer._prepare_uint8(hwc, torch.device("cpu"))
                sentinel = object()
                with patch.object(uploads, "_as_tensor", return_value=sentinel) as convert:
                    self.assertIs(buffer.prepare(frame, "cuda:0"), sentinel)
                    convert.assert_called_once_with(frame, torch.device("cuda:0"))
                self.assertEqual(buffer.cache_info()["float32_bytes"], 0)

    def test_cpu_targets_use_original_converter_even_for_contiguous_uint8(self):
        frame = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
        buffer = self.buffer()
        buffer._prepare_uint8(frame, torch.device("cpu"))
        with patch.object(uploads, "_as_tensor", wraps=_as_tensor) as convert:
            result = buffer.prepare(frame, "cpu")
        convert.assert_called_once_with(frame, torch.device("cpu"))
        self.assertEqual(buffer.cache_info()["float32_bytes"], 0)
        np.testing.assert_array_equal(result.numpy(), frame.astype(np.float32))

    def test_cuda_fast_path_stages_uint8_before_float_conversion(self):
        frame = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
        buffer = self.buffer()
        host = torch.empty(frame.shape, dtype=torch.uint8)
        upload = Mock()
        upload.prepare.return_value = host
        upload.upload.side_effect = lambda: host.clone()
        with patch.object(buffer, "_upload", upload):
            result = buffer.prepare(frame, "cuda:0")
        upload.prepare.assert_called_once_with(
            frame.shape, dtype=torch.uint8, device=torch.device("cuda:0")
        )
        upload.upload.assert_called_once_with()
        np.testing.assert_array_equal(host.numpy(), frame)
        np.testing.assert_array_equal(result.numpy(), frame.astype(np.float32))

    def test_release_clears_float_storage_even_if_upload_release_fails(self):
        buffer = self.buffer()
        buffer._prepare_uint8(np.zeros((2, 3, 3), np.uint8), torch.device("cpu"))
        with patch.object(buffer._upload, "release_resources", side_effect=RuntimeError("release")):
            with self.assertRaisesRegex(RuntimeError, "release"):
                buffer.release_resources()
        self.assertIsNone(buffer._float_workspace)
        self.assertEqual(buffer.cache_info()["float32_bytes"], 0)

    def test_invalid_float_budgets_are_rejected(self):
        for budget in (-1, True, 1.5, "128", None):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                uploads.FrameUploadBuffer(max_float32_bytes=budget)


class RenderImgFrameUploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.enterContext(patch.object(api, "initialize_renderer", side_effect=eager_initialize))

    def renderer(self, *, frame_layout="HWC", staged=True):
        renderer = api.ImageRenderer(
            CameraOwner(),
            api.RenderOptions(frame_layout=frame_layout, ad_layout="HWC"),
            device="cpu",
        )
        self.addCleanup(renderer.close)
        if staged:
            buffer = renderer._frame_upload

            def prepare_on_cpu(frame, device):
                if buffer._contiguous_cpu_uint8(frame):
                    return buffer._prepare_uint8(frame, torch.device("cpu"))
                return uploads.FrameUploadBuffer.prepare(buffer, frame, device)

            self.enterContext(patch.object(buffer, "prepare", side_effect=prepare_on_cpu))
        return renderer

    def assert_empty(self, renderer):
        info = renderer._frame_upload.cache_info()
        self.assertEqual(info["float32_bytes"], 0)
        self.assertEqual(info["uint8_upload"]["host_bytes"], 0)
        self.assertEqual(info["uint8_upload"]["device_bytes"], 0)

    def test_multigroup_uses_one_upload_and_matches_old_conversion(self):
        renderer, reference = self.renderer(), self.renderer(staged=False)
        data, frame, images = make_inputs()
        data["ad_params_2"][0]["bbox"] = (3, 4, 17, 15)
        buffer = renderer._frame_upload
        with patch.object(buffer, "_prepare_uint8", wraps=buffer._prepare_uint8) as prepare:
            actual = renderer.render_img(data, frame, images)
        prepare.assert_called_once()
        self.assertEqual(len(renderer.last_execution), 2)
        np.testing.assert_array_equal(actual, reference.render_img(deepcopy(data), frame, images))
        self.assertEqual(renderer.last_execution, reference.last_execution)

    def test_contiguous_layouts_and_cpu_tensors_match_old_conversion(self):
        data, hwc, images = make_inputs(routes=(1,))
        cases = (
            ("HWC", hwc),
            ("CHW", hwc.transpose(2, 0, 1).copy()),
            ("BHWC", hwc[None]),
            ("BCHW", hwc.transpose(2, 0, 1)[None].copy()),
        )
        for layout, source in cases:
            for frame in (source, torch.from_numpy(source)):
                with self.subTest(layout=layout, tensor=isinstance(frame, torch.Tensor)):
                    renderer = self.renderer(frame_layout=layout)
                    reference = self.renderer(frame_layout=layout, staged=False)
                    actual = renderer.render_img(data, frame, images)
                    expected = reference.render_img(deepcopy(data), frame, images)
                    np.testing.assert_array_equal(actual, expected)

    def test_inplace_frame_updates_dynamic_batch_and_independent_outputs(self):
        renderer, reference = self.renderer(), self.renderer(staged=False)
        data, frame, images = make_inputs()
        routes = {key: data.pop(key) for key in ("ad_params_1", "ad_params_2", "ad_params_10")}
        retained = []
        pointer = None
        for frame_id, route_keys in enumerate(((1,), (1, 2, 10), (10, 2), (1,))):
            for key in routes:
                data.pop(key, None)
            data.update(
                {f"ad_params_{route}": routes[f"ad_params_{route}"] for route in route_keys}
            )
            data["frame_id"] = frame_id
            frame[:] = (frame_id * 19 + 7, frame_id * 23 + 11, frame_id * 29 + 13)
            actual = renderer.render_img(data, frame, images)
            expected = reference.render_img(deepcopy(data), frame, images)
            np.testing.assert_array_equal(actual, expected)
            buffer = renderer._frame_upload
            current_pointer = buffer._float_workspace.data_ptr()
            if pointer is not None:
                self.assertEqual(current_pointer, pointer)
            pointer = current_pointer
            self.assertEqual(buffer.cache_info()["float32_allocations"], 1)
            for previous, snapshot in retained:
                np.testing.assert_array_equal(previous, snapshot)
                self.assertFalse(np.shares_memory(actual, previous))
            retained.append((actual, actual.copy()))
        renderer.close()
        for previous, snapshot in retained:
            np.testing.assert_array_equal(previous, snapshot)

    def test_fallback_dtypes_and_strides_match_and_drop_old_workspace(self):
        data, original, images = make_inputs(routes=(1,))
        strided = np.repeat(original, 2, axis=1)[:, ::2]
        frames = (
            strided,
            original[:, ::-1],
            original.astype(np.float32),
            original.astype(np.float64),
            torch.from_numpy(strided),
        )
        for frame in frames:
            with self.subTest(dtype=frame.dtype):
                renderer, reference = self.renderer(), self.renderer(staged=False)
                renderer.render_img(data, original, images)
                actual = renderer.render_img(data, frame, images)
                expected = reference.render_img(deepcopy(data), frame, images)
                np.testing.assert_array_equal(actual, expected)
                self.assert_empty(renderer)

    def test_pause_no_ads_release_reset_and_close_clear_inputs_but_keep_outputs(self):
        data, frame, images = make_inputs(routes=(1,))
        for action in ("pause", "no_ads", "release_resources", "reset_history", "close"):
            with self.subTest(action=action):
                renderer = self.renderer()
                output = renderer.render_img(data, frame, images)
                snapshot = output.copy()
                identities = (id(renderer.executor), id(renderer.owner._shared_paste_builder))
                self.assertGreater(renderer._frame_upload.cache_info()["float32_bytes"], 0)
                if action == "pause":
                    renderer.render_img({}, frame, {})
                    renderer.render_img({}, frame, {})
                elif action == "no_ads":
                    background = renderer.render_img({"ad_params_1": []}, frame, {})
                    np.testing.assert_array_equal(background, frame)
                else:
                    getattr(renderer, action)()
                self.assert_empty(renderer)
                np.testing.assert_array_equal(output, snapshot)
                if action != "close":
                    resumed = renderer.render_img(data, frame, images)
                    self.assertEqual(
                        identities,
                        (id(renderer.executor), id(renderer.owner._shared_paste_builder)),
                    )
                    np.testing.assert_array_equal(resumed, snapshot)
                    np.testing.assert_array_equal(output, snapshot)

    def test_outer_release_keeps_cleaning_after_mask_release_failure(self):
        renderer = self.renderer()
        data, frame, images = make_inputs(routes=(1,))
        renderer.render_img(data, frame, images)
        with patch.object(
            renderer.mask_builder, "release_resources", side_effect=RuntimeError("mask")
        ):
            with self.assertRaisesRegex(RuntimeError, "mask"):
                renderer.release_resources()
        self.assert_empty(renderer)
        self.assertEqual(renderer.owner.imgs_raw_groups, [])
        self.assertFalse(renderer._active)


if __name__ == "__main__":
    unittest.main()
