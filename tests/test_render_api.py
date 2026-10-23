"""CPU contracts for the unified frame API and its legacy entry points."""

from dataclasses import FrozenInstanceError, fields, replace
import importlib
import inspect
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from benchmarks.benchmark_paste_gpu_batch import make_case
from benchmarks.benchmark_shared_paste import RenderPath, SyntheticFixture
from virtualads import dynamic_paste_example
from virtualads import paste_gpu_batch_adapter as legacy
from virtualads import paste_render_executor as execution
from virtualads import paste_renderer as api
from virtualads.shared_paste_example import prepare_shared_paste_args, setup_shared_paste
from virtualads import shared_paste_example


def fixture(batch=3, device="cpu"):
    """Use small synthetic callbacks explicitly, with eager execution on CPU."""
    torch.manual_seed(60827)
    device = torch.device(device)
    cli = SimpleNamespace(
        batch=batch,
        height=32,
        width=48,
        slots=3,
        kernel_size=3,
        ad_sizes=[(7, 9), (11, 6), (5, 12)],
        sigma_pattern="repeated",
        acceleration="eager",
        dense_blur_backend="torch",
        normalization="legacy",
        camera_index_policy="legacy",
        align_corners=False,
        forward=False,
        inverse_iterations=10,
    )
    args, boxes = make_case(
        device,
        batch=batch,
        height=cli.height,
        width=cli.width,
        slots=cli.slots,
        kernel_size=cli.kernel_size,
        source_sizes=cli.ad_sizes,
    )
    path = RenderPath("api-contract", SyntheticFixture(cli, device, args, boxes), optimized=True)
    owner = path.fixture.owner
    owner._paste_batch_adapter = path.adapter
    owner._shared_paste_builder = path.builder
    return path


def render(path, **kwargs):
    value = path.fixture
    return api.render_frame(
        value.owner,
        value.frame,
        value.data,
        value.frame_id,
        mask_canvas=kwargs.pop("mask_canvas", value.image_args[5]),
        **kwargs,
    )


def render_legacy(path, *, wrapper=False, mask=None):
    value = path.fixture
    prepared = prepare_shared_paste_args(
        value.owner,
        value.frame,
        value.data,
        value.frame_id,
        value.image_args[5] if mask is None else mask,
    )
    function = legacy.paste_gpu_batch_adapter if wrapper else value.owner._paste_batch_adapter
    return function(value.owner, value.frame, value.data, value.frame_id, value.way_id, prepared)


class RenderApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_aliases_are_the_same_classes(self):
        self.assertIs(api.RenderExecutor, execution.RenderExecutor)
        self.assertIs(legacy.PasteBatchAdapter, execution.RenderExecutor)
        self.assertIs(execution.PasteBatchAdapter, execution.RenderExecutor)
        self.assertIs(api.RenderOptions, execution.RenderOptions)
        self.assertIs(legacy.AdapterOptions, execution.RenderOptions)
        self.assertIs(execution.AdapterOptions, execution.RenderOptions)
        self.assertEqual(legacy.ADAPTER_VERSION, api.RENDERER_VERSION)
        self.assertIs(shared_paste_example.PasteBatchAdapter, api.RenderExecutor)
        self.assertIs(shared_paste_example.SharedParameterBuilder, api.SharedParameterBuilder)
        self.assertIs(dynamic_paste_example.prepare_shared_paste_args, prepare_shared_paste_args)

    def test_options_keep_positional_order_and_frozen_dataclass(self):
        expected = (
            "backend acceleration blur_backend optimize_views cache_scalar_maps pinned_output "
            "profile_every frame_layout ad_layout mask_layout transparency_layout mask_scale "
            "transparency_scale check_ad_h_w resize_ads_to_ad_h_w resize_align_corners "
            "resize_antialias warp_align_corners distortion_align_corners warp_padding_mode "
            "distortion_padding_mode legacy_alpha matrix_coordinates grid_source image_index "
            "cache_ads max_cached_sizes max_entries_per_size return_numpy squeeze_single "
            "area_caps distortion_grid_builder warp_grids_builder batch_chunk_size "
            "blur_launch_config max_cached_bytes"
        ).split()
        self.assertEqual([field.name for field in fields(api.RenderOptions)], expected)
        options = legacy.AdapterOptions("batch", "eager", "torch", False, False, False, 7)
        self.assertEqual(options.profile_every, 7)
        self.assertFalse(options.optimize_views)
        self.assertEqual(replace(options, profile_every=0).profile_every, 0)
        with self.assertRaises(FrozenInstanceError):
            options.profile_every = 0

    def test_legacy_imports_share_the_canonical_package_and_classes(self):
        public = importlib.import_module("paste_renderer")
        old = importlib.import_module("paste_gpu_batch_adapter")
        canonical = importlib.import_module("virtualads.paste_render_executor")
        self.assertIs(public, api)
        self.assertIs(old, legacy)
        self.assertIs(canonical, execution)
        self.assertIs(public.RenderOptions, old.AdapterOptions)
        self.assertIs(public.RenderExecutor, old.PasteBatchAdapter)
        self.assertIs(canonical.RenderExecutor, old.PasteBatchAdapter)

    def test_legacy_signatures_keep_required_positional_parameters(self):
        for function, names in (
            (
                legacy.paste_gpu_batch_adapter,
                ["self", "frame_input", "data_dict", "frame_id", "way_id", "paste_args"],
            ),
            (
                execution.RenderExecutor.__call__,
                ["self", "owner", "frame_input", "data_dict", "frame_id", "way_id", "paste_args"],
            ),
        ):
            parameters = list(inspect.signature(function).parameters.values())
            self.assertEqual([parameter.name for parameter in parameters], names)
            self.assertTrue(
                all(parameter.default is inspect.Parameter.empty for parameter in parameters)
            )
            self.assertTrue(
                all(
                    parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                    for parameter in parameters
                )
            )
        dynamic = inspect.signature(dynamic_paste_example.render_dynamic_frame).parameters
        for name in ("material_groups", "camera_history", "mask_canvas", "device"):
            self.assertEqual(dynamic[name].kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIs(dynamic[name].default, inspect.Parameter.empty)

    def test_new_render_matches_old_call_and_binding_function_exactly(self):
        for batch in (1, 3):
            with self.subTest(batch=batch):
                path = fixture(batch)
                actual = render(path)
                snapshot = actual.copy()
                for wrapper in (False, True):
                    expected = render_legacy(path, wrapper=wrapper)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertFalse(np.shares_memory(actual, expected))
                np.testing.assert_array_equal(actual, snapshot)
                self.assertEqual(actual.dtype, np.uint8)
                self.assertEqual(actual.shape, (32, 48, 3) if batch == 1 else (batch, 32, 48, 3))

    def test_shared_preparation_once_and_camera_result_passthrough(self):
        path = fixture()
        result = (np.zeros(5, dtype=np.float32), 0.25)
        with (
            patch.object(path.builder, "prepare", wraps=path.builder.prepare) as prepare,
            patch.object(
                path.adapter, "render_prepared", wraps=path.adapter.render_prepared
            ) as execute,
        ):
            actual = render(path, camera_result=result)
        prepare.assert_called_once()
        execute.assert_called_once()
        self.assertIs(prepare.call_args.kwargs["camera_result"], result)
        self.assertFalse(path.builder.last_info["camera_filter_called"])
        self.assertEqual(actual.shape[0], 3)

    def test_omitted_business_fields_are_preserved_and_explicit_fields_are_live(self):
        path = fixture()
        owner = path.fixture.owner
        original_groups, original_history = owner.imgs_raw_groups, owner.paste_data_stack
        render(path)
        self.assertIs(owner.imgs_raw_groups, original_groups)
        self.assertIs(owner.paste_data_stack, original_history)
        changed_groups = original_groups[:1]
        changed_history = dict(original_history)
        expected_keys = tuple(changed_history)
        actual = render(path, material_groups=changed_groups, camera_history=changed_history)
        self.assertEqual(actual.shape, (32, 48, 3))
        self.assertIs(owner.imgs_raw_groups, changed_groups)
        self.assertIs(owner.paste_data_stack, changed_history)
        self.assertEqual(tuple(changed_history), expected_keys)

    def test_dynamic_batch_reuses_executor_and_preserves_old_outputs(self):
        path = fixture()
        groups = path.fixture.owner.imgs_raw_groups
        saved = []
        for batch in (3, 1, 2, 3):
            output = render(path, material_groups=groups[:batch])
            np.testing.assert_array_equal(output, render_legacy(path))
            self.assertIs(path.fixture.owner._paste_batch_adapter, path.adapter)
            for old, snapshot in saved:
                np.testing.assert_array_equal(old, snapshot)
                self.assertFalse(np.shares_memory(old, output))
            saved.append((output, output.copy()))

    def test_same_frame_id_reads_updated_mask_geometry_and_material_revision(self):
        path = fixture()
        value = path.fixture
        mask = torch.zeros_like(value.image_args[5])
        first = render(path, mask_canvas=mask)
        snapshot = first.copy()
        previous = snapshot
        frame_id = value.frame_id
        for change in ("mask", "geometry", "material"):
            with self.subTest(change=change):
                if change == "mask":
                    mask.fill_(1)
                elif change == "geometry":
                    mask.zero_()
                    for matrix in value.matrices:
                        matrix[0, 2] += 3.0
                else:
                    for group in value.owner.imgs_raw_groups:
                        for record in group.values():
                            record["img"][..., :3].fill_(240)
                            record["img"][..., 3].fill_(255)
                    value.owner.imgs_raw_version += 1
                actual = render(path, mask_canvas=mask)
                np.testing.assert_array_equal(actual, render_legacy(path, mask=mask))
                self.assertFalse(np.array_equal(actual, previous))
                # A separate executor has no stale material/map entries. Comparing
                # only the two entry points on one cache would miss invalidation bugs.
                fresh = api.RenderExecutor(path.adapter.options)
                prepared = prepare_shared_paste_args(
                    value.owner, value.frame, value.data, value.frame_id, mask
                )
                expected = fresh(
                    value.owner, value.frame, value.data, value.frame_id, value.way_id, prepared
                )
                np.testing.assert_array_equal(actual, expected)
                previous = actual.copy()
                self.assertEqual(value.frame_id, frame_id)
        np.testing.assert_array_equal(first, snapshot)

    def test_custom_call_hook_is_preserved_by_old_and_new_entry_points(self):
        class CallHook(api.RenderExecutor):
            calls = 0

            def __call__(self, *args, **kwargs):
                self.calls += 1
                return super().__call__(*args, **kwargs)

        for has_ads in (False, True):
            with self.subTest(has_ads=has_ads):
                path = fixture(2)
                value = path.fixture
                path.adapter = CallHook(path.adapter.options)
                value.owner._paste_batch_adapter = path.adapter
                if not has_ads:
                    value.data["ad_params"] = []
                prepared = (
                    prepare_shared_paste_args(
                        value.owner, value.frame, value.data, value.frame_id, value.image_args[5]
                    )
                    if has_ads
                    else None
                )
                expected = legacy.paste_gpu_batch_adapter(
                    value.owner, value.frame, value.data, value.frame_id, value.way_id, prepared
                )
                self.assertEqual(path.adapter.calls, 1)
                with patch.object(path.builder, "prepare", wraps=path.builder.prepare) as prepare:
                    actual = render(path)
                    dynamic = dynamic_paste_example.render_dynamic_frame(
                        value.owner,
                        value.frame,
                        value.data,
                        value.frame_id,
                        material_groups=value.owner.imgs_raw_groups,
                        camera_history=value.owner.paste_data_stack,
                        mask_canvas=value.image_args[5],
                        device=value.device,
                    )
                self.assertEqual(path.adapter.calls, 3)
                self.assertEqual(prepare.call_count, 2 if has_ads else 0)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(dynamic, expected)

    def test_way_specific_no_ads_bypasses_shared_inputs(self):
        path = fixture(2)
        value = path.fixture
        # Missing K/dist_coeffs and opaque sentinel arguments must never reach preparation.
        data = {"ad_params": value.data["ad_params"], "ad_params_7": []}
        with patch.object(
            path.builder, "prepare", side_effect=AssertionError("prepared no-ad frame")
        ):
            actual = api.render_frame(
                value.owner,
                value.frame,
                data,
                value.frame_id,
                way_id=7,
                mask_canvas=object(),
                device=object(),
                camera_result=object(),
            )
        background = value.frame.clamp(0, 255).to(torch.uint8).numpy()
        np.testing.assert_array_equal(actual, np.repeat(background[None], 2, axis=0))
        self.assertEqual(path.adapter.last_execution["reason"], "no_ads")

    def test_way_specific_none_falls_back_and_invalid_parameters_skip_shared(self):
        path = fixture(1)
        expected = render(path)
        path.fixture.data["ad_params_7"] = None
        np.testing.assert_array_equal(render(path, way_id=7), expected)
        path.fixture.data["ad_params_7"] = "invalid"
        with patch.object(
            path.builder, "prepare", side_effect=AssertionError("prepared invalid params")
        ):
            with self.assertRaises((TypeError, ValueError)):
                render(path, way_id=7)

    def test_explicit_empty_material_groups_fail_before_preparation(self):
        path = fixture()
        original = path.fixture.owner.imgs_raw_groups
        for groups in ([], (), {}, "invalid"):
            with self.subTest(groups=groups), patch.object(path.builder, "prepare") as prepare:
                with self.assertRaisesRegex(ValueError, "material_groups"):
                    render(path, material_groups=groups)
                prepare.assert_not_called()
                self.assertIs(path.fixture.owner.imgs_raw_groups, original)

    def test_initialize_and_old_setup_return_contracts_without_touching_business_inputs(self):
        base = api.RenderOptions(
            legacy_alpha=False,
            warp_padding_mode="border",
            distortion_align_corners=True,
            frame_layout="CHW",
            mask_scale=255,
            batch_chunk_size=0,
        )
        for initializer in (api.initialize_renderer, setup_shared_paste):
            with self.subTest(initializer=initializer.__name__):
                history, groups = {}, [{"business": object()}]
                owner = SimpleNamespace(paste_data_stack=history, imgs_raw_groups=groups)
                result = initializer(owner, base, use_separable=True, max_cached_sizes=5)
                executor, builder = owner._paste_batch_adapter, owner._shared_paste_builder
                self.assertIs(
                    result, executor if initializer is api.initialize_renderer else builder
                )
                self.assertIs(owner.paste_data_stack, history)
                self.assertIs(owner.imgs_raw_groups, groups)
                self.assertEqual(
                    sum(isinstance(value, api.RenderExecutor) for value in vars(owner).values()), 1
                )
                self.assertEqual(executor.options.batch_chunk_size, 0)
                self.assertEqual(executor.options.blur_backend, "separable")
                self.assertEqual(executor.options.max_cached_sizes, 5)
                self.assertEqual(executor.options.frame_layout, "CHW")
                self.assertEqual(executor.options.mask_scale, 255)
                self.assertFalse(executor.options.legacy_alpha)
                self.assertEqual(executor.options.warp_padding_mode, "border")
                self.assertTrue(executor.options.distortion_align_corners)
                self.assertIs(executor.options.warp_grids_builder.__self__, builder)

    def test_dynamic_example_remains_a_thin_delegate_and_returns_builder(self):
        owner = SimpleNamespace(paste_data_stack={})
        builder = dynamic_paste_example.initialize_dynamic_worker(
            owner, api.RenderOptions(batch_chunk_size=0)
        )
        self.assertIs(builder, owner._shared_paste_builder)
        self.assertEqual(owner._paste_batch_adapter.options.batch_chunk_size, "auto")
        # Preserve the required old parameters and pass their exact objects through.
        parameters = dict(
            material_groups=[{}], camera_history={}, mask_canvas=object(), device="cpu", way_id=3
        )
        marker = object()
        with patch.object(dynamic_paste_example, "render_frame", return_value=marker) as delegated:
            actual = dynamic_paste_example.render_dynamic_frame(
                owner, "frame", "data", 9, **parameters
            )
        self.assertIs(actual, marker)
        delegated.assert_called_once_with(
            owner,
            "frame",
            "data",
            9,
            mask_canvas=parameters["mask_canvas"],
            device=parameters["device"],
            way_id=3,
        )
        self.assertIs(owner.imgs_raw_groups, parameters["material_groups"])
        self.assertIs(owner.paste_data_stack, parameters["camera_history"])
        with patch.object(dynamic_paste_example, "render_frame", return_value=marker):
            dynamic_paste_example.render_dynamic_frame(
                owner, "frame", "data", 9, **{**parameters, "camera_history": None}
            )
        self.assertIsNone(owner.paste_data_stack)

    def test_release_preserves_inputs_and_outputs_and_allows_reuse(self):
        path = fixture()
        owner = path.fixture.owner
        groups, history = owner.imgs_raw_groups, owner.paste_data_stack
        first = render(path)
        snapshot = first.copy()
        api.release_renderer(owner)
        self.assertIs(owner._paste_batch_adapter, path.adapter)
        self.assertIs(owner._shared_paste_builder, path.builder)
        self.assertIs(owner.imgs_raw_groups, groups)
        self.assertIs(owner.paste_data_stack, history)
        self.assertEqual(path.adapter.cache_info()["retained_stack_bytes"], 0)
        self.assertEqual(path.builder.cache_info()["entries"], 0)
        self.assertIsNone(path.builder._kernel_context)
        np.testing.assert_array_equal(first, snapshot)
        second = render(path)
        np.testing.assert_array_equal(second, snapshot)
        self.assertFalse(np.shares_memory(first, second))

    @unittest.skipUnless(
        os.environ.get("PASTE_TEST_DEVICE", "").startswith("cuda"),
        "set PASTE_TEST_DEVICE=cuda:0 explicitly to exercise public GPU initialization",
    )
    def test_public_initialize_render_release_gpu(self):
        """Optional real initialization path; all other tests remain CPU-only."""
        device = os.environ["PASTE_TEST_DEVICE"]
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        path = fixture(device=device)
        owner = path.fixture.owner
        path.adapter = api.initialize_renderer(owner, path.adapter.options, use_separable=True)
        path.builder = owner._shared_paste_builder
        groups = owner.imgs_raw_groups
        saved = []
        for batch in (3, 1, 2, 3):
            output = render(path, material_groups=groups[:batch])
            np.testing.assert_array_equal(output, render_legacy(path, wrapper=True))
            self.assertEqual(output.dtype, np.uint8)
            self.assertEqual(output.shape, (32, 48, 3) if batch == 1 else (batch, 32, 48, 3))
            for prior, snapshot in saved:
                np.testing.assert_array_equal(prior, snapshot)
                self.assertFalse(np.shares_memory(prior, output))
            saved.append((output, output.copy()))
        api.release_renderer(owner)
        self.assertEqual(path.adapter.cache_info()["retained_stack_bytes"], 0)
        self.assertEqual(path.builder.cache_info()["entries"], 0)
        rebuilt = render(path)
        np.testing.assert_array_equal(rebuilt, saved[-1][1])
        for prior, snapshot in saved:
            np.testing.assert_array_equal(prior, snapshot)
            self.assertFalse(np.shares_memory(prior, rebuilt))


if __name__ == "__main__":
    unittest.main()
