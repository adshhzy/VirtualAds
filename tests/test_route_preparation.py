"""CPU-only contracts for per-call source metadata and borrowed route preparation.

Real eager rendering compares the private optimization with its instance-level
off switch. Public plans/custom hooks keep independently allocated parameters.
CUDA upload/event/kernel behavior is covered by the separate serial GPU checks.
"""

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch

from virtualads import paste_renderer as api
from virtualads import paste_route_plan as planning
from virtualads.paste_render_executor import RenderExecutor
from virtualads.paste_route_preparation import RoutePreparationWorkspace
from tests.test_render_img import CameraOwner, eager_initialize, image_value, make_inputs


class RoutePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.enterContext(patch.object(api, "initialize_renderer", side_effect=eager_initialize))

    def renderer(self, enabled=True):
        renderer = api.ImageRenderer(
            CameraOwner(), api.RenderOptions(frame_layout="HWC", ad_layout="HWC"), device="cpu"
        )
        renderer._use_prepared_route_inputs = enabled
        self.addCleanup(renderer.close)
        return renderer

    def test_standard_path_uses_metadata_and_skips_second_source_description(self):
        renderer = self.renderer()
        data, frame, images = make_inputs()
        with patch.object(
            renderer.executor, "_describe_ad_sources", side_effect=AssertionError("rescanned")
        ):
            output = renderer.render_img(data, frame, images)
        self.assertEqual(output.shape[0], 3)
        info = renderer._route_preparation.cache_info()
        self.assertEqual(info["prepared_plans"], 1)
        self.assertEqual(info["metadata_hits"], 1)
        self.assertEqual(info["matrix_stacks"], 2)
        self.assertEqual(info["opacity_vectors"], 2)
        self.assertEqual(info["metadata_fallbacks"], 0)

    def test_disabled_path_really_uses_public_planner_and_existing_description(self):
        renderer = self.renderer(False)
        data, frame, images = make_inputs()
        with (
            patch.object(api, "build_route_plan", wraps=api.build_route_plan) as plan,
            patch.object(
                renderer.executor,
                "_describe_ad_sources",
                wraps=renderer.executor._describe_ad_sources,
            ) as describe,
        ):
            renderer.render_img(data, frame, images)
        plan.assert_called_once()
        describe.assert_called_once()
        self.assertEqual(renderer._route_preparation.cache_info()["retained_bytes"], 0)

    def test_prepared_and_public_source_signatures_are_identical(self):
        renderer = self.renderer()
        data, frame, images = make_inputs()
        plan = planning._build_route_plan(
            data, images, frame.shape[:2], renderer.executor.options, renderer._route_preparation
        )
        group = plan.groups[0]
        description = group.sources.consume(
            group.material_groups, group.params, renderer.executor.options, 0
        )
        expected = renderer.executor._describe_ad_sources(group.material_groups, group.params, 0)
        self.assertEqual(description[1:], expected[1:])
        for actual_slot, expected_slot in zip(description[0], expected[0]):
            for actual, reference in zip(actual_slot, expected_slot):
                self.assertIs(actual[0], reference[0])
                self.assertEqual(actual[1:], reference[1:])
        self.assertIsNone(
            group.sources.consume(group.material_groups, group.params, renderer.executor.options, 0)
        )

    def test_layout_metadata_is_reused_only_inside_one_plan(self):
        renderer = self.renderer()
        data, frame, images = make_inputs()
        with patch.object(planning, "_image_plan", wraps=planning._image_plan) as image_plan:
            renderer.render_img(data, frame, images)
            self.assertEqual(image_plan.call_count, 2)
            renderer.render_img(data, frame, images)
            self.assertEqual(image_plan.call_count, 4)

    def test_uniform_matrix_dtype_preserved_and_mixed_dtype_uses_original_stack(self):
        renderer = self.renderer()
        for kinds in ((np.float32,) * 3, (np.float64,) * 3, (np.float32, np.float64, np.int64)):
            with self.subTest(kinds=kinds):
                data, frame, images = make_inputs()
                for route, dtype in zip((1, 2, 10), kinds):
                    for param in data[f"ad_params_{route}"]:
                        param["m_inv"] = param["m_inv"].astype(dtype)
                old = planning.build_route_plan(
                    data, images, frame.shape[:2], renderer.executor.options
                )
                before = renderer._route_preparation.cache_info()["matrix_stacks"]
                new = planning._build_route_plan(
                    data,
                    images,
                    frame.shape[:2],
                    renderer.executor.options,
                    renderer._route_preparation,
                )
                for a, b in zip(old.groups[0].params, new.groups[0].params):
                    self.assertEqual(a["m_inv"].dtype, b["m_inv"].dtype)
                    np.testing.assert_array_equal(a["m_inv"], b["m_inv"])
                    np.testing.assert_array_equal(a["transparency"], b["transparency"])
                delta = renderer._route_preparation.cache_info()["matrix_stacks"] - before
                self.assertEqual(delta, 2 if len(set(kinds)) == 1 else 0)

    def test_cpu_tensor_strided_matrices_and_non_scalar_opacity_match_public_plan(self):
        renderer = self.renderer()
        data, frame, images = make_inputs()
        for route in (1, 2, 10):
            first = data[f"ad_params_{route}"][0]
            first["m_inv"] = torch.from_numpy(first["m_inv"].T.copy()).T.requires_grad_()
        expected = planning.build_route_plan(
            data, images, frame.shape[:2], renderer.executor.options
        )
        actual = planning._build_route_plan(
            data, images, frame.shape[:2], renderer.executor.options, renderer._route_preparation
        )
        np.testing.assert_array_equal(
            actual.groups[0].params[0]["m_inv"], expected.groups[0].params[0]["m_inv"]
        )
        data["ad_params_2"][0]["transparency"] = np.array(0.3, dtype=np.float64)
        actual = planning._build_route_plan(
            data, images, frame.shape[:2], renderer.executor.options, renderer._route_preparation
        )
        expected = planning.build_route_plan(
            data, images, frame.shape[:2], renderer.executor.options
        )
        self.assertEqual([g.indices for g in actual.groups], [g.indices for g in expected.groups])
        single = next(group for group in actual.groups if group.indices == (1,))
        self.assertIs(single.params[0]["transparency"], data["ad_params_2"][0]["transparency"])

    def test_public_plan_arrays_are_not_overwritten_by_internal_reuse(self):
        renderer = self.renderer()
        data, frame, images = make_inputs()
        public = planning.build_route_plan(data, images, frame.shape[:2], renderer.executor.options)
        matrices = [param["m_inv"].copy() for param in public.groups[0].params]
        opacity = [param["transparency"].copy() for param in public.groups[0].params]
        for step in range(3):
            data["ad_params_2"][0]["m_inv"][0, 2] += 0.125
            data["ad_params_2"][0]["transparency"] = 0.1 * step
            renderer.render_img(data, frame, images)
        for param, matrix, alpha in zip(public.groups[0].params, matrices, opacity):
            np.testing.assert_array_equal(param["m_inv"], matrix)
            np.testing.assert_array_equal(param["transparency"], alpha)

    def test_multiple_layout_groups_do_not_overwrite_each_other(self):
        fast, ordinary = self.renderer(), self.renderer(False)
        data, frame, images = make_inputs(routes=(1, 2, 3, 4))
        for route in (3, 4):
            data[f"ad_params_{route}"][0]["bbox"] = (3, 4, 17, 15)
        output = fast.render_img(data, frame, images)
        reference = ordinary.render_img(deepcopy(data), frame, images)
        self.assertEqual(len(fast.last_execution), 2)
        np.testing.assert_array_equal(output, reference)
        self.assertEqual(fast._route_preparation.cache_info()["metadata_hits"], 2)

    def test_dynamic_routes_geometry_materials_and_revision_keep_results_exact(self):
        fast, ordinary = self.renderer(), self.renderer(False)
        data, frame, images = make_inputs()
        original = {key: data[key] for key in ("ad_params_1", "ad_params_2", "ad_params_10")}
        held = []
        for step, routes in enumerate(((1,), (1, 2, 10), (10, 2), (2, 1), (1, 2, 10))):
            for key in original:
                data.pop(key, None)
            data.update({f"ad_params_{route}": original[f"ad_params_{route}"] for route in routes})
            for route in routes:
                param = data[f"ad_params_{route}"][0]
                param["m_inv"][0, 2] += 0.125
                param["transparency"] = (step + route % 3 + 1) / 8
                if step == 3:
                    param["bbox"] = (3, 4, 17, 15)
            if step == 2:
                key = next(iter(images))
                images[key] = image_value(images[key]).copy()
                images[key][..., :3] = 211
            if step == 4:
                for value in images.values():
                    image_value(value)[..., :3] = 113
                data["imgs_raw_version"] = 1
            actual = fast.render_img(data, frame, images)
            expected = ordinary.render_img(deepcopy(data), frame, images)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(fast.last_execution, ordinary.last_execution)
            for previous, snapshot in held:
                np.testing.assert_array_equal(previous, snapshot)
                self.assertFalse(np.shares_memory(previous, actual))
            held[:] = [(actual, actual.copy())]

    def test_custom_executor_hook_keeps_independent_params_and_existing_call(self):
        class CustomExecutor(RenderExecutor):
            def __call__(self, owner, frame_input, data_dict, frame_id, way_id, paste_args):
                self.calls = getattr(self, "calls", 0) + 1
                self.seen = data_dict["ad_params"]
                return super().__call__(owner, frame_input, data_dict, frame_id, way_id, paste_args)

        renderer = self.renderer()
        renderer.executor = CustomExecutor(renderer.executor.options)
        renderer.owner._paste_batch_adapter = renderer.executor
        data, frame, images = make_inputs()
        renderer.render_img(data, frame, images)
        kept = renderer.executor.seen[0]["m_inv"]
        snapshot = kept.copy()
        data["ad_params_2"][0]["m_inv"][0, 2] += 0.25
        renderer.render_img(data, frame, images)
        self.assertEqual(renderer.executor.calls, 2)
        self.assertEqual(renderer._route_preparation.cache_info()["prepared_plans"], 0)
        np.testing.assert_array_equal(kept, snapshot)

    def test_custom_warp_callback_uses_existing_validation_and_independent_parameters(self):
        renderer = self.renderer()
        original = renderer.executor.options.warp_grids_builder
        seen = []

        def custom(**kwargs):
            seen.append(kwargs["params"][0]["m_inv"])
            return original(**kwargs)

        renderer.executor.options = replace(renderer.executor.options, warp_grids_builder=custom)
        data, frame, images = make_inputs()
        renderer.render_img(data, frame, images)
        snapshot = seen[0].copy()
        data["ad_params_2"][0]["m_inv"][0, 2] += 0.25
        renderer.render_img(data, frame, images)
        self.assertEqual(len(seen), 2)
        self.assertEqual(renderer._route_preparation.cache_info()["prepared_plans"], 0)
        np.testing.assert_array_equal(seen[0], snapshot)

    def test_owner_callback_source_metadata_mutation_falls_back_to_live_validation(self):
        renderer = self.renderer()
        data, frame, images = make_inputs(routes=(1,))
        source = image_value(next(iter(images.values())))
        original_shape = source.shape
        original = renderer.owner.create_gaussian_kernel

        def change_source(sigma):
            source.shape = (original_shape[1], original_shape[0], 4)
            return original(sigma)

        renderer.owner.create_gaussian_kernel = change_source
        with patch.object(
            renderer.executor, "_describe_ad_sources", wraps=renderer.executor._describe_ad_sources
        ) as describe:
            renderer.render_img(data, frame, images)
        describe.assert_called_once()
        self.assertEqual(renderer._route_preparation.cache_info()["metadata_fallbacks"], 1)

    def test_bounded_workspace_reuse_pause_failure_release_and_sources_not_retained(self):
        renderer = self.renderer()
        data, frame, images = make_inputs()
        result = renderer.render_img(data, frame, images)
        snapshot = result.copy()
        info = renderer._route_preparation.cache_info()
        renderer.render_img(data, frame, images)
        self.assertEqual(
            renderer._route_preparation.cache_info()["reallocations"], info["reallocations"]
        )
        data["mask_coor_list"] = [{"bbox": (0, 0, 2, 2), "mask": np.zeros((1, 2), np.float32)}]
        with self.assertRaises(ValueError):
            renderer.render_img(data, frame, images)
        renderer.render_img({}, frame, {})
        self.assertEqual(renderer._route_preparation.cache_info()["retained_bytes"], 0)
        data["mask_coor_list"] = []
        renderer.render_img(data, frame, images)
        references = [weakref.ref(image_value(value)) for value in images.values()]
        renderer.release_resources()
        del images
        self.assertTrue(all(reference() is None for reference in references))
        self.assertEqual(renderer._route_preparation.cache_info()["retained_bytes"], 0)
        np.testing.assert_array_equal(result, snapshot)

    def test_oversized_private_plan_is_temporary_and_no_active_alias_is_retained(self):
        renderer = self.renderer()
        renderer._route_preparation = RoutePreparationWorkspace(max_cached_bytes=32)
        data, frame, images = make_inputs()
        renderer.render_img(data, frame, images)
        info = renderer._route_preparation.cache_info()
        self.assertEqual(info["retained_bytes"], 0)
        self.assertEqual(info["temporary_allocations"], 1)
        self.assertIsNone(renderer._route_preparation._active)

    def test_workspace_invalid_budgets_and_empty_plan_release(self):
        for budget in (-1, True, 1.5, "32", None):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                RoutePreparationWorkspace(budget)
        workspace = RoutePreparationWorkspace(256)
        workspace.begin(1)
        workspace.finish()
        self.assertGreater(workspace.cache_info()["retained_bytes"], 0)
        workspace.begin(0)
        self.assertEqual(workspace.cache_info()["retained_bytes"], 0)
        self.assertIsNone(workspace._active)

    def test_missing_declared_resize_target_preserves_old_validation_error(self):
        data, frame, images = make_inputs(routes=(1, 2))
        first, second = data["ad_params_1"][0], data["ad_params_2"][0]
        first.pop("ad_h_w")
        images[(second["ad_placement_id"], second["ad_id"])] = np.zeros((8, 10, 4), np.float32)
        messages = []
        for enabled in (False, True):
            renderer = self.renderer(enabled)
            renderer.executor.options = replace(
                renderer.executor.options, resize_ads_to_ad_h_w=True
            )
            with self.assertRaises(ValueError) as caught:
                renderer.render_img(deepcopy(data), frame, images)
            messages.append(str(caught.exception))
        self.assertEqual(messages[0], messages[1])
        self.assertIn("requires ad_h_w", messages[0])

    def test_options_replaced_by_owner_callback_uses_live_validation(self):
        data, frame, images = make_inputs(routes=(1,))
        messages = []
        for enabled in (False, True):
            renderer = self.renderer(enabled)
            original = renderer.owner.create_gaussian_kernel

            def change_options(sigma):
                renderer.executor.options = replace(renderer.executor.options, ad_layout="CHW")
                return original(sigma)

            renderer.owner.create_gaussian_kernel = change_options
            with self.assertRaises(ValueError) as caught:
                renderer.render_img(deepcopy(data), frame, images)
            messages.append(str(caught.exception))
        self.assertEqual(messages[0], messages[1])

    def test_mapping_side_effect_keeps_original_plan_shape_in_metadata_guard(self):
        messages = []
        for enabled in (False, True):
            renderer = self.renderer(enabled)
            data, frame, images = make_inputs(routes=(1, 2))
            data["ad_params_1"] = data["ad_params_1"][:1]
            data["ad_params_2"] = data["ad_params_2"][:1]
            first, second = data["ad_params_1"][0], data["ad_params_2"][0]
            source = image_value(images[(first["ad_placement_id"], first["ad_id"])])
            later_pair = (second["ad_placement_id"], second["ad_id"])

            class ChangingMaterials(dict):
                def get(self, key, default=None):
                    if key == later_pair:
                        source.shape = (9, 7, 4)
                    return super().get(key, default)

            with self.assertRaises(ValueError) as caught:
                renderer.render_img(data, frame, ChangingMaterials(images))
            messages.append(str(caught.exception))
        self.assertEqual(messages[0], messages[1])
        self.assertIn("source H,W differs", messages[0])

    def test_callback_installing_custom_warp_detaches_only_borrowed_parameters(self):
        for routes in ((1,), (1, 2, 10)):
            with self.subTest(routes=routes):
                renderer = self.renderer()
                data, frame, images = make_inputs(routes=routes)
                builtin = renderer.executor.options.warp_grids_builder
                gaussian = renderer.owner.create_gaussian_kernel
                seen = []

                def custom(**kwargs):
                    seen.append(kwargs["params"])
                    return builtin(**kwargs)

                def install_hook(sigma):
                    renderer.executor.options = replace(
                        renderer.executor.options, warp_grids_builder=custom
                    )
                    return gaussian(sigma)

                renderer.owner.create_gaussian_kernel = install_hook
                renderer.render_img(data, frame, images)
                kept = seen[0][0]
                matrix_snapshot = np.asarray(kept["m_inv"]).copy()
                opacity_snapshot = kept["transparency"].copy()
                if len(routes) == 1:
                    self.assertIs(kept["m_inv"], data["ad_params_1"][0]["m_inv"])
                else:
                    self.assertFalse(
                        np.shares_memory(kept["m_inv"], renderer._route_preparation._storage)
                    )
                self.assertFalse(
                    np.shares_memory(kept["transparency"], renderer._route_preparation._storage)
                )
                renderer.executor.options = replace(
                    renderer.executor.options, warp_grids_builder=builtin
                )
                data["ad_params_1"][0]["transparency"] = 0.15
                renderer.render_img(data, frame, images)
                np.testing.assert_array_equal(kept["m_inv"], matrix_snapshot)
                np.testing.assert_array_equal(kept["transparency"], opacity_snapshot)

    def test_oversized_plan_failure_clears_active_and_keeps_no_sources(self):
        renderer = self.renderer()
        renderer._route_preparation = RoutePreparationWorkspace(0)
        data, frame, images = make_inputs()
        with patch.object(planning, "_prepared_matrices", side_effect=RuntimeError("packing")):
            with self.assertRaisesRegex(RuntimeError, "packing"):
                renderer.render_img(data, frame, images)
        self.assertIsNone(renderer._route_preparation._active)
        self.assertEqual(renderer._route_preparation.cache_info()["retained_bytes"], 0)
        self.assertFalse(renderer._active)


if __name__ == "__main__":
    unittest.main()
