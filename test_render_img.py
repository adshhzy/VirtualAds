"""ImageRenderer contracts with real CPU math and a private eager initializer.

Only initialization is substituted: the route planner, shared camera/geometry,
mask preparation, executor and NumPy export remain real. This is a test seam,
not a promise that the public Triton initializer supports CPU rendering.
"""

from copy import deepcopy
from dataclasses import replace
import inspect
import unittest
from unittest.mock import patch

import numpy as np
import torch

import paste_renderer as api
from paste_render_executor import RenderExecutor
from paste_shared_params import SharedParameterBuilder


HEIGHT, WIDTH = 25, 37


class CameraOwner:
    """Small deterministic test callbacks; never production camera/blur laws."""

    def __init__(self):
        self.motion_calls = self.sigma_calls = self.kernel_calls = 0

    def compute_camera_motion_rate(self, poses):
        self.motion_calls += 1
        return float(len(poses)) / 100.0

    def motion_rate_to_sigma(self, motion_rate):
        self.sigma_calls += 1
        return 0.8

    def create_gaussian_kernel(self, sigma):
        self.kernel_calls += 1
        factor = np.array([1, 2, 1], dtype=np.float32) / 4
        return np.outer(factor, factor)


def eager_initialize(
    owner,
    base_options,
    *,
    use_separable=False,
    normalization="legacy",
    camera_index_policy="legacy",
    distortion_inverse=True,
    max_cached_sizes=3,
):
    """Preserve public setup conventions while selecting actual CPU operations."""
    builder = SharedParameterBuilder(
        acceleration="eager",
        normalization=normalization,
        camera_index_policy=camera_index_policy,
        distortion_inverse=distortion_inverse,
        distortion_align_corners=base_options.distortion_align_corners,
        enable_separable=use_separable,
    )
    executor = RenderExecutor(
        replace(
            base_options,
            backend="batch",
            acceleration="eager",
            blur_backend="torch",
            warp_grids_builder=builder.build_warp_grids,
            grid_source="matrix",
            matrix_coordinates="roi",
            pinned_output=False,
            return_numpy=True,
            squeeze_single=True,
            max_cached_sizes=max_cached_sizes,
            profile_every=0,
        )
    )
    owner._paste_batch_adapter = executor
    owner._shared_paste_builder = builder
    return executor


def material(route, slot, shape):
    ys, xs = np.indices(shape)
    image = np.stack(
        (
            (route * 23 + xs * 19 + ys * 7) % 190 + 40,
            (route * 37 + xs * 3 + ys * 29) % 180 + 50,
            (route * 11 + xs * 31 + ys * 5) % 200 + 30,
            150 + (xs * 13 + ys * 17 + slot) % 106,
        ),
        axis=-1,
    )
    return image.astype(np.float32)


def make_inputs(routes=(10, 2, 1), *, frame_id=0):
    """Unsorted routes share layout but have distinct live geometry/opacity."""
    ys, xs = np.indices((HEIGHT, WIDTH))
    frame = np.stack((xs % 17 + 4, ys % 19 + 6, (xs + ys) % 23 + 8), axis=-1)
    frame = frame.astype(np.uint8)
    data = {
        "frame_id": frame_id,
        "K": np.array([[45, 0, WIDTH / 2], [0, 44, HEIGHT / 2], [0, 0, 1]], np.float32),
        "w2c": np.eye(4, dtype=np.float32),
        "dist_coeffs": np.zeros(5, dtype=np.float32),
        "mask_coor_list": [],
    }
    images = {}
    for route in routes:
        params = []
        for slot, (shape, box) in enumerate(
            zip(((7, 9), (5, 11)), ((2, 3, 16, 14), (18, 10, 33, 23)))
        ):
            height, width = shape
            x1, y1, x2, y2 = box
            matrix = np.eye(3, dtype=np.float32)
            matrix[0, 0] = (width - 1) / (x2 - x1 - 1)
            matrix[1, 1] = (height - 1) / (y2 - y1 - 1)
            matrix[0, 2] = (route % 3) * 0.2
            matrix[1, 2] = -(route % 2) * 0.15
            placement, ad_id = f"slot-{slot}", route * 10 + slot
            params.append(
                dict(
                    ad_placement_id=placement,
                    ad_id=ad_id,
                    bbox=box,
                    ad_h_w=shape,
                    m_inv=matrix,
                    transparency=0.4 + (route % 3) * 0.2,
                )
            )
            image = material(route, slot, shape)
            # Both supported public forms must enter the same real execution path.
            images[(placement, ad_id)] = image if (route + slot) % 2 else {"img": image}
        data[f"ad_params_{route}"] = params
    return data, frame, images


def image_value(value):
    return value["img"] if isinstance(value, dict) else value


def plain_route_reference(data, frame, images, route, options, *, mask_sigma=1.5, mask_radius=3):
    """Run the existing explicit one-route API without ImageRenderer/planning."""
    owner = CameraOwner()
    eager_initialize(owner, options, max_cached_sizes=options.max_cached_sizes)
    owner.paste_data_stack = {
        data["frame_id"]: {
            key: np.array(data[key], copy=True) for key in ("K", "w2c", "dist_coeffs")
        }
    }
    owner.imgs_raw_version = data.get("imgs_raw_version", 0)
    params = deepcopy(data[f"ad_params_{route}"])
    owner.imgs_raw_groups = [
        {
            (param["ad_placement_id"], param["ad_id"]): {
                "img": image_value(images[(param["ad_placement_id"], param["ad_id"])])
            }
            for param in params
        }
    ]
    frame_data = {"K": data["K"], "dist_coeffs": data["dist_coeffs"], "ad_params": params}
    mask_builder = api.SharedMaskBuilder(sigma=mask_sigma, radius=mask_radius)
    try:
        mask = mask_builder.build(
            data.get("mask_coor_list", []), height=HEIGHT, width=WIDTH, device="cpu"
        )
        return api.render_frame(
            owner,
            frame,
            frame_data,
            data["frame_id"],
            mask_canvas=mask,
            device="cpu",
        )
    finally:
        api.release_renderer(owner)
        mask_builder.release_resources()


class RenderImgTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.initialize = self.enterContext(
            patch.object(api, "initialize_renderer", side_effect=eager_initialize)
        )

    def renderer(self, *, owner=None, options=None, **kwargs):
        owner = CameraOwner() if owner is None else owner
        if options is None:
            options = api.RenderOptions(frame_layout="HWC", ad_layout="HWC", max_cached_sizes=4)
        renderer = api.ImageRenderer(owner, options, device="cpu", **kwargs)
        self.addCleanup(renderer.close)
        return renderer

    def assert_output(self, output, batch):
        shape = (HEIGHT, WIDTH, 3) if batch == 1 else (batch, HEIGHT, WIDTH, 3)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.dtype, np.uint8)
        self.assertEqual(output.shape, shape)
        self.assertTrue(output.flags.c_contiguous)

    def assert_routes_match(self, renderer, data, frame, images):
        output = renderer.render_img(data, frame, images)
        self.assert_output(output, len(renderer.route_ids))
        batch = output[None] if output.ndim == 3 else output
        for index, route in enumerate(renderer.route_ids):
            expected = plain_route_reference(
                data,
                frame,
                images,
                route,
                renderer.executor.options,
                mask_sigma=renderer.mask_builder.sigma,
                mask_radius=renderer.mask_builder.radius,
            )
            np.testing.assert_array_equal(batch[index], expected)
        return output

    def test_public_signature_initialization_once_and_callback_identity(self):
        self.assertEqual(
            list(inspect.signature(api.ImageRenderer.render_img).parameters),
            ["self", "data_dict", "frame", "imgs_raw"],
        )
        owner = CameraOwner()
        callbacks = {
            name: getattr(owner, name).__func__
            for name in (
                "compute_camera_motion_rate",
                "motion_rate_to_sigma",
                "create_gaussian_kernel",
            )
        }
        renderer = api.ImageRenderer(owner, device="cpu")
        self.addCleanup(renderer.close)
        data, frame, images = make_inputs(routes=(1,))
        renderer.render_img(data, frame, images)
        renderer.render_img(data, frame, images)
        self.initialize.assert_called_once()
        self.assertEqual(renderer.executor.options.batch_chunk_size, "auto")
        self.assertIs(owner.paste_data_stack, renderer.history)
        for name, callback in callbacks.items():
            self.assertIs(getattr(owner, name).__func__, callback)
        with self.assertRaises(TypeError):
            renderer.render_img(data, frame, images, "extra")

    def test_compatible_routes_batch_once_and_match_explicit_single_routes(self):
        renderer = self.renderer()
        data, frame, images = make_inputs()
        with patch.object(
            renderer.executor, "render_prepared", wraps=renderer.executor.render_prepared
        ) as execute:
            output = renderer.render_img(data, frame, images)
        execute.assert_called_once()
        self.assertEqual(renderer.route_ids, (1, 2, 10))
        self.assert_output(output, 3)
        for index, route in enumerate(renderer.route_ids):
            expected = plain_route_reference(data, frame, images, route, renderer.executor.options)
            np.testing.assert_array_equal(output[index], expected)
        self.assertFalse(np.array_equal(output[0], output[1]))

    def test_incompatible_layouts_split_and_keep_numeric_output_order(self):
        for variation in ("bbox", "source_size", "slot_count", "slot_order"):
            with self.subTest(variation=variation):
                renderer = self.renderer()
                data, frame, images = make_inputs(routes=(10, 2, 1))
                params = data["ad_params_2"]
                if variation == "bbox":
                    params[0]["bbox"] = (3, 4, 17, 15)
                elif variation == "source_size":
                    param = params[0]
                    images[(param["ad_placement_id"], param["ad_id"])] = material(2, 0, (8, 10))
                    param["ad_h_w"] = (8, 10)
                elif variation == "slot_count":
                    params.pop()
                else:
                    params.reverse()
                with patch.object(
                    renderer.executor, "render_prepared", wraps=renderer.executor.render_prepared
                ) as execute:
                    output = renderer.render_img(data, frame, images)
                self.assertEqual(execute.call_count, 2)
                self.assertEqual(renderer.route_ids, (1, 2, 10))
                for index, route in enumerate(renderer.route_ids):
                    expected = plain_route_reference(
                        data, frame, images, route, renderer.executor.options
                    )
                    np.testing.assert_array_equal(output[index], expected)

    def test_legacy_auxiliary_fields_are_ignored(self):
        renderer = self.renderer()
        data, frame, images = make_inputs()
        expected = renderer.render_img(data, frame, images)
        data.update(
            brightness=object(),
            ad_paste_data=object(),
            way_ad_map=object(),
            image_path=object(),
            ad_params=object(),
        )
        actual = renderer.render_img(data, frame, images)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(renderer.route_ids, (1, 2, 10))

    def test_shared_camera_geometry_and_mask_run_once_for_multiple_groups(self):
        renderer = self.renderer()
        data, frame, images = make_inputs()
        data["ad_params_2"][0]["bbox"] = (3, 4, 17, 15)
        for frame_id in (0, 1):
            data["frame_id"] = frame_id
            renderer.render_img(data, frame, images)
        renderer.owner.motion_calls = renderer.owner.sigma_calls = 0
        data["frame_id"] = 2
        builder = renderer.owner._shared_paste_builder
        with (
            patch.object(builder, "prepare", wraps=builder.prepare) as prepare,
            patch.object(renderer.mask_builder, "build", wraps=renderer.mask_builder.build) as mask,
            patch.object(
                renderer.executor, "render_prepared", wraps=renderer.executor.render_prepared
            ) as execute,
        ):
            renderer.render_img(data, frame, images)
        prepare.assert_called_once()
        mask.assert_called_once()
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(renderer.owner.motion_calls, 1)
        self.assertEqual(renderer.owner.sigma_calls, 1)

    def test_empty_routes_return_background_without_camera_or_shared_preparation(self):
        renderer = self.renderer()
        _, frame, _ = make_inputs()
        builder = renderer.owner._shared_paste_builder
        with (
            patch.object(builder, "prepare", wraps=builder.prepare) as prepare,
            patch.object(renderer.mask_builder, "build", wraps=renderer.mask_builder.build) as mask,
        ):
            single = renderer.render_img({"ad_params_4": []}, frame, {})
            multiple = renderer.render_img(
                {"ad_params_10": [], "ad_params_2": [], "ad_params_1": []}, frame, {}
            )
        prepare.assert_not_called()
        mask.assert_not_called()
        self.assert_output(single, 1)
        self.assert_output(multiple, 3)
        np.testing.assert_array_equal(single, frame)
        np.testing.assert_array_equal(multiple, np.repeat(frame[None], 3, axis=0))
        self.assertFalse(np.shares_memory(single, frame))
        self.assertFalse(np.shares_memory(single, multiple))
        self.assertEqual(renderer.route_ids, (1, 2, 10))
        self.assertEqual(renderer.executor.last_execution["reason"], "no_ads")

    def test_same_frame_mask_updates_apply_to_every_group_and_preserve_old_output(self):
        renderer = self.renderer(mask_radius=0)
        data, frame, images = make_inputs()
        data["ad_params_2"][0]["bbox"] = (3, 4, 17, 15)
        protection = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        data["mask_coor_list"] = [{"bbox": (0, 0, WIDTH, HEIGHT), "mask": protection}]
        visible = self.assert_routes_match(renderer, data, frame, images)
        snapshot = visible.copy()
        self.assertFalse(np.array_equal(visible[0], frame))
        protection.fill(1)
        hidden = self.assert_routes_match(renderer, data, frame, images)
        np.testing.assert_array_equal(hidden, np.repeat(frame[None], 3, axis=0))
        np.testing.assert_array_equal(visible, snapshot)
        self.assertFalse(np.shares_memory(visible, hidden))
        protection.fill(0)
        restored = renderer.render_img(data, frame, images)
        np.testing.assert_array_equal(restored, snapshot)

    def test_mixed_empty_and_ad_routes_use_the_correct_output_positions(self):
        renderer = self.renderer()
        data, frame, images = make_inputs()
        data["ad_params_2"] = []
        output = self.assert_routes_match(renderer, data, frame, images)
        np.testing.assert_array_equal(output[1], frame)
        self.assertFalse(np.array_equal(output[0], frame))

    def test_pause_releases_once_on_active_transition_and_resumes_same_objects(self):
        renderer = self.renderer()
        data, frame, images = make_inputs(routes=(1,))
        identities = (id(renderer.executor), id(renderer.owner._shared_paste_builder))
        with patch.object(
            renderer, "release_resources", wraps=renderer.release_resources
        ) as release:
            initial_empty = renderer.render_img({}, frame, {})
            release.assert_not_called()
            active = renderer.render_img(data, frame, images)
            snapshot = active.copy()
            first_empty = renderer.render_img({}, frame, {})
            second_empty = renderer.render_img({}, frame, {})
            release.assert_called_once()
            for empty in (initial_empty, first_empty, second_empty):
                self.assert_output(empty, 0)
            self.assertIsNot(first_empty, second_empty)
            self.assertEqual(renderer.executor.cache_info()["retained_stack_bytes"], 0)
            self.assertFalse(renderer.owner.imgs_raw_groups)
            self.assertEqual(renderer.owner._shared_paste_builder.cache_info()["entries"], 0)
            data["frame_id"] = 1
            resumed = renderer.render_img(data, frame, images)
            self.assertEqual(
                identities, (id(renderer.executor), id(renderer.owner._shared_paste_builder))
            )
            np.testing.assert_array_equal(active, snapshot)
            np.testing.assert_array_equal(resumed, snapshot)

    def test_failed_active_frame_is_released_by_the_next_pause(self):
        renderer = self.renderer()
        data, frame, images = make_inputs(routes=(1,))
        original = renderer.executor.render_prepared

        def fail_after_render(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected post-render failure")

        with patch.object(renderer.executor, "render_prepared", side_effect=fail_after_render):
            with self.assertRaisesRegex(RuntimeError, "post-render failure"):
                renderer.render_img(data, frame, images)
        self.assertGreater(renderer.executor.cache_info()["retained_stack_bytes"], 0)
        with patch.object(
            renderer, "release_resources", wraps=renderer.release_resources
        ) as release:
            renderer.render_img({}, frame, {})
            renderer.render_img({}, frame, {})
        release.assert_called_once()
        self.assertEqual(renderer.executor.cache_info()["retained_stack_bytes"], 0)
        self.assertEqual(renderer.owner._shared_paste_builder.cache_info()["entries"], 0)
        self.assert_output(renderer.render_img(data, frame, images), 1)

    def test_history_is_bounded_independent_and_does_not_rewrite_input(self):
        renderer = self.renderer(history_capacity=5)
        data, frame, images = make_inputs(routes=(1,))
        original_keys = set(data)
        for frame_id in range(12):
            data["frame_id"] = frame_id
            data["dist_coeffs"][0] = frame_id * 0.0001
            data["w2c"][0, 3] = frame_id * 0.01
            snapshot = {name: data[name].copy() for name in ("K", "w2c", "dist_coeffs")}
            coefficients = data["dist_coeffs"]
            renderer.render_img(data, frame, images)
            self.assertIs(data["dist_coeffs"], coefficients)
            self.assertEqual(set(data), original_keys)
            for name in snapshot:
                np.testing.assert_array_equal(data[name], snapshot[name])
                np.testing.assert_array_equal(renderer.history[frame_id][name], snapshot[name])
                self.assertFalse(np.shares_memory(renderer.history[frame_id][name], data[name]))
            self.assertLessEqual(len(renderer.history), 5)
        self.assertEqual(tuple(renderer.history), (7, 8, 9, 10, 11))
        retained = renderer.history[11]["dist_coeffs"].copy()
        data["dist_coeffs"].fill(99)
        np.testing.assert_array_equal(renderer.history[11]["dist_coeffs"], retained)

    def test_revision_and_explicit_invalidation_refresh_inplace_pixels_same_frame(self):
        renderer = self.renderer()
        data, frame, images = make_inputs(routes=(1,))
        original = renderer.render_img(data, frame, images)
        snapshot = original.copy()
        key = next(iter(images))
        pixels = image_value(images[key])
        pixels[..., :3] = (230, 20, 30)
        data["imgs_raw_version"] = 1
        revised = self.assert_routes_match(renderer, data, frame, images)
        self.assertFalse(np.array_equal(revised, original))
        pixels[..., :3] = (20, 230, 210)
        renderer.invalidate_assets()
        invalidated = self.assert_routes_match(renderer, data, frame, images)
        self.assertFalse(np.array_equal(invalidated, revised))
        np.testing.assert_array_equal(original, snapshot)
        self.assertFalse(np.shares_memory(revised, invalidated))

    def test_source_replacement_and_dynamic_route_order_keep_old_outputs(self):
        renderer = self.renderer()
        data, frame, images = make_inputs()
        retained = []
        for routes in ((1,), (10, 2, 1), (10, 1), (2,)):
            current = {
                key: value for key, value in data.items() if not key.startswith("ad_params_")
            }
            current.update({f"ad_params_{route}": data[f"ad_params_{route}"] for route in routes})
            key = ("slot-0", routes[0] * 10)
            images[key] = material(routes[0] + len(retained) + 9, 0, (7, 9))
            output = self.assert_routes_match(renderer, current, frame, images)
            self.assertEqual(renderer.route_ids, tuple(sorted(routes)))
            for prior, snapshot in retained:
                np.testing.assert_array_equal(prior, snapshot)
                self.assertFalse(np.shares_memory(prior, output))
            retained.append((output, output.copy()))

    def test_reset_history_allows_frame_ids_to_restart_and_preserves_outputs(self):
        renderer = self.renderer()
        data, frame, images = make_inputs(routes=(1,), frame_id=7)
        output = renderer.render_img(data, frame, images)
        snapshot = output.copy()
        renderer.reset_history()
        self.assertEqual(len(renderer.history), 0)
        self.assertEqual(renderer.executor.cache_info()["retained_stack_bytes"], 0)
        self.assertEqual(renderer.owner._shared_paste_builder.cache_info()["entries"], 0)
        data["frame_id"] = 0
        restarted = renderer.render_img(data, frame, images)
        np.testing.assert_array_equal(restarted, snapshot)
        np.testing.assert_array_equal(output, snapshot)
        self.assertFalse(np.shares_memory(restarted, output))

    def test_context_close_is_idempotent_and_closed_operations_fail(self):
        renderer = self.renderer()
        data, frame, images = make_inputs(routes=(1,))
        with self.assertRaisesRegex(RuntimeError, "consumer failure"):
            with renderer as entered:
                self.assertIs(entered, renderer)
                output = renderer.render_img(data, frame, images)
                snapshot = output.copy()
                raise RuntimeError("consumer failure")
        with patch.object(api, "release_renderer", wraps=api.release_renderer) as release:
            renderer.close()
            release.assert_not_called()
        self.assertIsNone(renderer.owner.paste_data_stack)
        self.assertFalse(renderer.owner.imgs_raw_groups)
        self.assertEqual(renderer.route_ids, ())
        self.assertEqual(len(renderer.history), 0)
        np.testing.assert_array_equal(output, snapshot)
        self.assertTrue(images)
        for operation in (
            lambda: renderer.render_img(data, frame, images),
            renderer.reset_history,
            renderer.invalidate_assets,
        ):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                operation()

    def test_close_error_still_clears_history_and_closes_the_worker(self):
        renderer = self.renderer()
        data, frame, images = make_inputs(routes=(1,))
        renderer.render_img(data, frame, images)
        builder = renderer.owner._shared_paste_builder
        original_release = renderer.executor.release_resources
        with (
            patch.object(
                renderer.executor, "release_resources", side_effect=RuntimeError("release")
            ),
            patch.object(builder, "release_resources", wraps=builder.release_resources) as shared,
            patch.object(
                renderer.mask_builder,
                "release_resources",
                wraps=renderer.mask_builder.release_resources,
            ) as mask,
        ):
            with self.assertRaisesRegex(RuntimeError, "release"):
                renderer.close()
        shared.assert_called_once()
        mask.assert_called_once()
        self.assertEqual(len(renderer.history), 0)
        self.assertIsNone(renderer.owner.paste_data_stack)
        self.assertFalse(renderer.owner.imgs_raw_groups)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            renderer.render_img(data, frame, images)
        renderer.close()
        original_release()  # The injected failure intentionally skipped this real cleanup.

    def test_mask_release_error_still_drops_owner_materials_and_closes_worker(self):
        renderer = self.renderer()
        data, frame, images = make_inputs(routes=(1,))
        output = renderer.render_img(data, frame, images)
        snapshot = output.copy()
        original_release = renderer.mask_builder.release_resources
        with patch.object(
            renderer.mask_builder, "release_resources", side_effect=RuntimeError("mask release")
        ):
            with self.assertRaisesRegex(RuntimeError, "mask release"):
                renderer.close()
        self.assertEqual(renderer.executor.cache_info()["retained_stack_bytes"], 0)
        self.assertEqual(renderer.owner._shared_paste_builder.cache_info()["entries"], 0)
        self.assertFalse(renderer.owner.imgs_raw_groups)
        self.assertIsNone(renderer.owner.paste_data_stack)
        self.assertEqual(len(renderer.history), 0)
        self.assertEqual(renderer.last_execution, ())
        self.assertFalse(renderer._active)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            renderer.reset_history()
        renderer.close()
        np.testing.assert_array_equal(output, snapshot)
        original_release()  # The injected failure intentionally retained mask workspace.

    def test_frame_layout_and_single_original_frame_contract(self):
        renderer = self.renderer(options=api.RenderOptions(frame_layout="CHW", ad_layout="HWC"))
        data, frame, images = make_inputs(routes=(1,))
        self.assert_routes_match(renderer, data, np.transpose(frame, (2, 0, 1)), images)
        with self.assertRaisesRegex(ValueError, "one original frame"):
            renderer.render_img(data, np.repeat(frame.transpose(2, 0, 1)[None], 2, axis=0), images)


if __name__ == "__main__":
    unittest.main()
