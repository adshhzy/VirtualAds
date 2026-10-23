"""CPU metadata and ownership checks for independent per-route configurations."""

from dataclasses import replace
import unittest

import numpy as np
import torch

from virtualads.paste_render_executor import RenderOptions
from virtualads.paste_route_plan import build_route_plan


def slot(placement="position", ad_id=1, *, box=(2, 3, 9, 8), opacity=1.0):
    return dict(
        ad_placement_id=placement,
        ad_id=ad_id,
        bbox=box,
        m_inv=np.eye(3, dtype=np.float32),
        transparency=opacity,
    )


def image(shape=(5, 7, 4)):
    return np.zeros(shape, dtype=np.uint8)


class RoutePlanTests(unittest.TestCase):
    def setUp(self):
        self.options = RenderOptions(ad_layout="HWC")
        self.frame_hw = (24, 32)

    def plan(self, data, assets, options=None):
        return build_route_plan(data, assets, self.frame_hw, options or self.options)

    def test_route_order_and_unrelated_fields(self):
        data = {
            "ad_params_10": [slot()],
            "brightness": object(),
            "ad_params_2": [slot()],
            "ad_paste_data": object(),
            "way_ad_map": object(),
            "ad_params": object(),
            "ad_params_1": [slot()],
            3: object(),
        }
        plan = self.plan(data, {("position", 1): image()})
        self.assertEqual(plan.route_ids, (1, 2, 10))
        self.assertEqual(plan.groups[0].indices, (0, 1, 2))
        self.assertEqual(plan.groups[0].params[0]["m_inv"].shape, (3, 3, 3))
        self.assertNotIn("brightness", plan.groups[0].params[0])

    def test_invalid_route_keys_and_values(self):
        for key in (
            "ad_params_0",
            "ad_params_01",
            "ad_params_-1",
            "ad_params_1.0",
            "ad_params_",
            "ad_params_1\n",
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.plan({key: []}, {})
        for value in (None, {}, "bad"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.plan({"ad_params_1": value}, {})
        with self.assertRaises(TypeError):
            self.plan({"ad_params_1": [None]}, {})

    def test_no_routes_and_empty_background_routes(self):
        plan = self.plan({"ad_params": []}, {})
        self.assertEqual((plan.route_ids, plan.groups), ((), ()))
        plan = self.plan({"ad_params_2": [], "ad_params_1": ()}, {})
        self.assertEqual(plan.route_ids, (1, 2))
        self.assertEqual(plan.groups[0].indices, (0, 1))
        self.assertEqual(plan.groups[0].params, [])
        self.assertEqual(plan.groups[0].material_groups, [{}, {}])

    def test_different_ids_share_layout_and_preserve_sources(self):
        first, second = slot("a", 10, opacity=64), slot("b", 20, opacity=192)
        second["m_inv"][0, 2] = 2
        a, b = image(), image()
        plan = self.plan(
            {"ad_params_1": [first], "ad_params_2": [second]}, {("a", 10): a, ("b", 20): b}
        )
        group = plan.groups[0]
        self.assertEqual(group.indices, (0, 1))
        param = group.params[0]
        pair = (param["ad_placement_id"], param["ad_id"])
        self.assertIs(group.material_groups[0][pair]["img"], a)
        self.assertIs(group.material_groups[1][pair]["img"], b)
        np.testing.assert_array_equal(param["transparency"], np.array([64, 192], np.float32))
        self.assertEqual(param["m_inv"][1, 0, 2], 2)
        self.assertEqual(first["ad_placement_id"], "a")
        self.assertEqual(second["ad_id"], 20)

    def test_layout_changes_split_groups_and_keep_output_indices(self):
        data = {
            "ad_params_1": [slot("a")],
            "ad_params_2": [slot("b", box=(3, 3, 10, 8))],
            "ad_params_3": [slot("c")],
            "ad_params_4": [slot("a"), slot("a")],
            "ad_params_5": [slot("d")],
        }
        assets = {(key, 1): image() for key in "abc"}
        assets["d", 1] = image((6, 7, 4))
        groups = self.plan(data, assets).groups
        self.assertEqual([group.indices for group in groups], [(0, 2), (1,), (3,), (4,)])

    def test_geometry_is_live_between_plans_and_single_matrix_is_preserved(self):
        first, second = slot(), slot()
        data = {"ad_params_1": [first], "ad_params_2": [second]}
        assets = {("position", 1): image()}
        before = self.plan(data, assets)
        second["m_inv"][1, 2] = 4
        after = self.plan(data, assets)
        self.assertEqual(before.groups[0].params[0]["m_inv"][1, 1, 2], 0)
        self.assertEqual(after.groups[0].params[0]["m_inv"][1, 1, 2], 4)
        second["bbox"] = (3, 3, 10, 8)
        split = self.plan(data, assets)
        self.assertEqual(len(split.groups), 2)
        self.assertIs(split.groups[1].params[0]["m_inv"], second["m_inv"])
        alpha = split.groups[0].params[0]["transparency"]
        self.assertEqual(alpha.shape, (1,))
        self.assertEqual(alpha.dtype, np.float32)

    def test_exact_asset_lookup_no_placement_or_path_fallback(self):
        param = slot()
        param["image_path"] = "must-not-be-opened.npy"
        with self.assertRaisesRegex(KeyError, "missing material"):
            self.plan({"ad_params_1": [param]}, {"position": image()})
        with self.assertRaises(KeyError):
            self.plan({"ad_params_1": [param]}, {("position", 1): {"image_path": "unused"}})
        with self.assertRaises(TypeError):
            self.plan({"ad_params_1": [param]}, {("position", 1): "filename.png"})

    def test_duplicate_ids_keep_distinct_ordered_virtual_slots(self):
        first = slot(box=(2, 3, 9, 8))
        second = slot(box=(4, 3, 11, 8), opacity=0)
        source = image()
        group = self.plan({"ad_params_1": [first, second]}, {("position", 1): source}).groups[0]
        self.assertEqual([param["bbox"] for param in group.params], [first["bbox"], second["bbox"]])
        pairs = [(param["ad_placement_id"], param["ad_id"]) for param in group.params]
        self.assertNotEqual(*pairs)
        self.assertEqual(len(group.material_groups[0]), 2)
        for pair in pairs:
            self.assertIs(group.material_groups[0][pair]["img"], source)

    def test_wrappers_device_lists_and_fixed_image_index(self):
        a, b = image(), torch.zeros((5, 7, 4))
        data = {"ad_params_2": [slot()], "ad_params_10": [slot()]}
        for value in (a, {"img": a}, [a, b], {"img": [a, b]}):
            group = self.plan(data, {("position", 1): value}).groups[0]
            self.assertIs(next(iter(group.material_groups[0].values()))["img"], a)
            self.assertIs(next(iter(group.material_groups[1].values()))["img"], a)
        group = self.plan(
            data, {("position", 1): {"img": [a, b]}}, replace(self.options, image_index=1)
        ).groups[0]
        self.assertIs(next(iter(group.material_groups[0].values()))["img"], b)
        with self.assertRaises(IndexError):
            self.plan(data, {("position", 1): []})

    def test_resize_grouping_uses_effective_sizes_and_retains_targets(self):
        a, b = slot("a"), slot("b")
        a["ad_h_w"] = b["ad_h_w"] = (8, 10)
        assets = {("a", 1): image(), ("b", 1): image((6, 9, 4))}
        data = {"ad_params_1": [a], "ad_params_2": [b]}
        self.assertEqual(len(self.plan(data, assets).groups), 2)
        resized = self.plan(data, assets, replace(self.options, resize_ads_to_ad_h_w=True))
        self.assertEqual(resized.groups[0].indices, (0, 1))
        self.assertEqual(resized.groups[0].params[0]["ad_h_w"], (8, 10))
        with self.assertRaises(ValueError):
            self.plan(data, assets, replace(self.options, check_ad_h_w=True))
        del b["ad_h_w"]
        with self.assertRaises(ValueError):
            self.plan(data, assets, replace(self.options, resize_ads_to_ad_h_w=True))

    def test_non_scalar_opacity_stays_in_independent_routes(self):
        opacity = np.ones((5, 7), np.float32)
        a, b, c = slot(opacity=opacity), slot(opacity=opacity), slot(opacity=torch.tensor(0.5))
        groups = self.plan(
            {"ad_params_1": [a], "ad_params_2": [b], "ad_params_3": [c]}, {("position", 1): image()}
        ).groups
        self.assertEqual([group.indices for group in groups], [(0,), (1,), (2,)])
        self.assertIs(groups[0].params[0]["transparency"], opacity)
        self.assertIs(groups[2].params[0]["transparency"], c["transparency"])

    def test_cpu_tensor_matrices_stack_without_changing_inputs(self):
        a, b = slot(), slot()
        b["m_inv"] = torch.eye(3, requires_grad=True)
        matrix = b["m_inv"]
        group = self.plan(
            {"ad_params_1": [a], "ad_params_2": [b]}, {("position", 1): image()}
        ).groups[0]
        self.assertIsInstance(group.params[0]["m_inv"], np.ndarray)
        self.assertEqual(group.params[0]["m_inv"].shape, (2, 3, 3))
        self.assertIs(b["m_inv"], matrix)
        self.assertTrue(matrix.requires_grad)

    def test_non_cpu_matrices_keep_their_device_without_host_conversion(self):
        # Meta tensors exercise device routing without using a GPU or reading values.
        a, b, c = slot(), slot(), slot()
        a["m_inv"] = torch.empty((3, 3), device="meta")
        c["m_inv"] = torch.empty((3, 3), device="meta")
        groups = self.plan(
            {"ad_params_1": [a], "ad_params_2": [b], "ad_params_3": [c]},
            {("position", 1): image()},
        ).groups
        self.assertEqual([group.indices for group in groups], [(0, 2), (1,)])
        matrices = groups[0].params[0]["m_inv"]
        self.assertEqual(matrices.device.type, "meta")
        self.assertEqual(matrices.shape, (2, 3, 3))
        self.assertIs(groups[1].params[0]["m_inv"], b["m_inv"])

    def test_coordinate_metadata_cannot_trigger_device_to_host_copies(self):
        assets = {("position", 1): image()}
        param = slot()
        param["bbox"] = torch.empty(4, device="meta")
        with self.assertRaisesRegex(TypeError, "CPU coordinate/size metadata"):
            self.plan({"ad_params_1": [param]}, assets)
        param = slot()
        param["ad_h_w"] = torch.empty(2, device="meta")
        with self.assertRaisesRegex(TypeError, "CPU coordinate/size metadata"):
            self.plan({"ad_params_1": [param]}, assets, replace(self.options, check_ad_h_w=True))

    def test_invalid_image_geometry_and_matrix_shapes(self):
        for key, value in (
            ("bbox", (0, 0, 33, 5)),
            ("bbox", (0, 0, 7.5, 5)),
            ("m_inv", np.eye(4)),
            ("m_inv", np.eye(3)[None]),
        ):
            param = slot()
            param[key] = value
            with (
                self.subTest(key=key, shape=getattr(value, "shape", None)),
                self.assertRaises(ValueError),
            ):
                self.plan({"ad_params_1": [param]}, {("position", 1): image()})
        with self.assertRaises(ValueError):
            self.plan({"ad_params_1": [slot()]}, {("position", 1): image((5, 7, 3))})


if __name__ == "__main__":
    unittest.main()
