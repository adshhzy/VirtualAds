"""Small CPU-only checks for streaming asset-cache limits and ownership.

These exercise metadata, conversion and retention without rendering or CUDA.
Run with: python -m unittest test_streaming_lifecycle
"""

from dataclasses import replace
import gc
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch

import paste_render_executor as adapter_module
from paste_asset_cache import DEFAULT_MAX_CACHED_BYTES, SizeAssetCache
from paste_gpu_batch_adapter import AdapterOptions, PasteBatchAdapter, _SizeAssetCache


def fixture(batch=2, sizes=((2, 3), (3, 5))):
    params = [dict(ad_placement_id=f"slot{slot}", ad_id=slot) for slot in range(len(sizes))]
    groups = [
        {
            f"slot{slot}": {"img": np.full((*hw, 4), 10 * group + slot, dtype=np.float32)}
            for slot, hw in enumerate(sizes)
        }
        for group in range(batch)
    ]
    return SimpleNamespace(imgs_raw_groups=groups, imgs_raw_version=0), params


def renderer(**overrides):
    return PasteBatchAdapter(AdapterOptions(ad_layout="HWC", **overrides))


def collect(adapter, owner, params, image_index=0):
    with torch.inference_mode():
        return adapter._collect_ads(
            owner, owner.imgs_raw_groups, params, torch.device("cpu"), image_index
        )


def assert_fresh(test, adapter, owner, params, image_index=0):
    actual = collect(adapter, owner, params, image_index)
    fresh = PasteBatchAdapter(replace(adapter.options, cache_ads=False))
    expected = collect(fresh, owner, params, image_index)
    test.assertEqual(len(actual), len(expected))
    for value, reference in zip(actual, expected):
        test.assertTrue(torch.equal(value, reference))
    return actual


class StreamingAssetCacheTests(unittest.TestCase):
    def test_budget_option_and_private_alias(self):
        self.assertIs(_SizeAssetCache, SizeAssetCache)
        self.assertEqual(AdapterOptions().max_cached_bytes, DEFAULT_MAX_CACHED_BYTES)
        self.assertEqual(renderer(max_cached_bytes=0).cache_info()["bytes"], 0)
        for invalid in (-1, True, 1.5, None):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "max_cached_bytes"),
            ):
                renderer(max_cached_bytes=invalid)

    def test_history_byte_limit_and_lru_order(self):
        cache = SizeAssetCache(max_sizes=2, max_entries=2, max_bytes=32)
        first, second, third = (torch.full((4,), float(value)) for value in range(3))
        cache.put((1, 1), "first", first, ())
        cache.put((2, 2), "second", second, ())
        self.assertIs(cache.get((1, 1), "first"), first)
        cache.put((1, 1), "third", third, ())
        self.assertEqual(cache.info()["bytes"], 32)
        self.assertIsNone(cache.get((2, 2), "second"))
        self.assertIs(cache.get((1, 1), "third"), third)
        self.assertEqual(cache.info()["evictions"], 1)
        cache.put((1, 1), "third", torch.zeros(20), ())
        self.assertEqual(cache.info()["bytes"], 16)
        self.assertIsNone(cache.get((1, 1), "third"))
        self.assertEqual(cache.info()["skipped_oversize"], 1)

    def test_history_size_and_entry_limits_remain_active(self):
        cache = SizeAssetCache(max_sizes=1, max_entries=1, max_bytes=1024)
        cache.put((2, 3), "old", torch.zeros(4), ())
        cache.put((2, 3), "new", torch.zeros(4), ())
        self.assertIsNone(cache.get((2, 3), "old"))
        cache.put((4, 5), "different-size", torch.zeros(4), ())
        self.assertEqual(cache.info()["sizes"], {(4, 5): 1})
        self.assertEqual(cache.info()["bytes"], 16)
        self.assertEqual(cache.info()["evictions"], 2)

    def test_current_set_over_budget_reuses_all_stacks(self):
        owner, params = fixture(sizes=((2, 3), (3, 5), (5, 6)))
        adapter = renderer(max_cached_bytes=32, max_cached_sizes=1, max_entries_per_size=1)
        first = collect(adapter, owner, params)
        initial = adapter.cache_info()
        self.assertGreater(initial["current_bytes"], initial["max_cached_bytes"])
        self.assertEqual(initial["bytes"], 0)
        with patch.object(adapter, "_resize_ad", side_effect=AssertionError("rebuilt current set")):
            again = collect(adapter, owner, params)
        self.assertIs(again, first)
        after = adapter.cache_info()
        self.assertEqual(after["misses"], initial["misses"])
        self.assertEqual(after["hits"] - initial["hits"], len(params))
        self.assertEqual(after["current_hits"], len(params))
        self.assertEqual(after["retained_stack_bytes"], after["current_bytes"])

    def test_history_and_current_byte_accounting_deduplicates_aliases(self):
        owner, params = fixture(batch=1, sizes=((2, 3),))
        adapter = renderer()
        collect(adapter, owner, params)
        info = adapter.cache_info()
        self.assertEqual(info["bytes"], 2 * 3 * 4 * 4)
        self.assertEqual(info["retained_stack_bytes"], info["current_bytes"])
        self.assertEqual(info["current_bytes"], info["bytes"])

    def test_current_refs_survive_history_eviction_and_release_on_invalidation(self):
        owner, params = fixture(batch=2)
        refs = [
            weakref.ref(group[param["ad_placement_id"]]["img"])
            for group in owner.imgs_raw_groups
            for param in params
        ]
        adapter = renderer(max_cached_bytes=0)
        collect(adapter, owner, params)
        owner.imgs_raw_groups.clear()
        gc.collect()
        self.assertTrue(all(reference() is not None for reference in refs))
        self.assertEqual(adapter.cache_info()["bytes"], 0)
        adapter.invalidate_assets()
        gc.collect()
        self.assertTrue(all(reference() is None for reference in refs))
        self.assertEqual(adapter.cache_info()["retained_stack_bytes"], 0)
        self.assertEqual(adapter.cache_info()["image_plans"], 0)

    def test_revision_changes_drop_old_sources_and_stacks_keep_counters(self):
        owner, params = fixture(batch=1, sizes=((2, 3),))
        adapter = renderer()
        previous_sources, previous_stacks = [], []
        for revision in range(40):
            owner.imgs_raw_groups[0]["slot0"]["img"] = np.full(
                (2, 3, 4), revision, dtype=np.float32
            )
            owner.imgs_raw_version = revision
            previous_sources.append(weakref.ref(owner.imgs_raw_groups[0]["slot0"]["img"]))
            result = collect(adapter, owner, params)
            previous_stacks.append(weakref.ref(result[0]))
            self.assertEqual(float(result[0][0, 0, 0, 0]), revision)
            self.assertEqual(adapter.cache_info()["sizes"], {(2, 3): 1})
        del result
        gc.collect()
        self.assertTrue(all(reference() is None for reference in previous_sources[:-1]))
        self.assertTrue(all(reference() is None for reference in previous_stacks[:-1]))
        info = adapter.cache_info()
        self.assertEqual(info["misses"], 40)
        self.assertEqual(info["revision_changes"], 39)
        self.assertEqual(info["evictions"], 39)
        owner.imgs_raw_version = 0
        collect(adapter, owner, params)
        self.assertEqual(adapter.cache_info()["misses"], 41)

    def test_current_metadata_detects_replacement_order_batch_shape_and_dtype(self):
        owner, params = fixture()
        adapter = renderer(max_cached_bytes=0)
        prior = assert_fresh(self, adapter, owner, params)
        owner.imgs_raw_groups[0]["slot0"]["img"] = np.full((2, 3, 4), 72, dtype=np.float32)
        current = assert_fresh(self, adapter, owner, params)
        self.assertFalse(torch.equal(current[0], prior[0]))
        owner.imgs_raw_groups.reverse()
        assert_fresh(self, adapter, owner, params)
        params.reverse()
        assert_fresh(self, adapter, owner, params)
        owner.imgs_raw_groups = owner.imgs_raw_groups[:1]
        assert_fresh(self, adapter, owner, params)
        image = torch.from_numpy(owner.imgs_raw_groups[0]["slot0"]["img"]).clone()
        owner.imgs_raw_groups[0]["slot0"]["img"] = image
        assert_fresh(self, adapter, owner, params)
        image.transpose_(0, 1)  # Same object, different source geometry and strides.
        reshaped = assert_fresh(self, adapter, owner, params)
        self.assertEqual(reshaped[1].shape[-2:], (3, 2))
        owner.imgs_raw_groups[0]["slot0"]["img"] = image.numpy().astype(np.uint8)
        assert_fresh(self, adapter, owner, params)

    def test_in_place_pixels_require_revision_and_resize_uses_live_metadata(self):
        owner, params = fixture(batch=1, sizes=((2, 3),))
        params[0]["ad_h_w"] = (3, 5)
        adapter = renderer(resize_ads_to_ad_h_w=True)
        first = collect(adapter, owner, params)
        owner.imgs_raw_groups[0]["slot0"]["img"].fill(93)
        owner.imgs_raw_version += 1
        changed = assert_fresh(self, adapter, owner, params)
        self.assertFalse(torch.equal(first[0], changed[0]))
        params[0]["ad_h_w"] = (4, 6)
        resized = assert_fresh(self, adapter, owner, params)
        self.assertEqual(resized[0].shape, (1, 4, 4, 6))

    def test_image_index_and_disabled_cache_do_not_reuse_old_pixels(self):
        owner, params = fixture(batch=1, sizes=((2, 3),))
        owner.imgs_raw_groups[0]["slot0"]["img"] = [
            np.full((2, 3, 4), value, dtype=np.float32) for value in (5, 82)
        ]
        adapter = renderer()
        first = collect(adapter, owner, params, 0)
        second = assert_fresh(self, adapter, owner, params, 1)
        self.assertFalse(torch.equal(first[0], second[0]))
        for change in ("no-revision", "disabled"):
            if change == "no-revision":
                owner.imgs_raw_version = None
            else:
                owner.imgs_raw_version = 2
                adapter.options = replace(adapter.options, cache_ads=False)
            collect(adapter, owner, params, 1)
            owner.imgs_raw_groups[0]["slot0"]["img"][1].fill(37)
            result = assert_fresh(self, adapter, owner, params, 1)
            self.assertEqual(float(result[0][0, 0, 0, 0]), 37)
            self.assertIsNone(adapter._ads)
            self.assertEqual(adapter._asset_source_refs, ())
            self.assertEqual(adapter.cache_info()["bytes"], 0)

    def test_shape_plans_are_bounded_metadata_only_and_errors_keep_current_name(self):
        owner, params = fixture(batch=3, sizes=((2, 3), (2, 3)))
        adapter = renderer()
        with patch.object(adapter_module, "_image_plan", wraps=adapter_module._image_plan) as plan:
            collect(adapter, owner, params)
            collect(adapter, owner, params)
        self.assertEqual(plan.call_count, 1)
        for width in range(1, 150):
            adapter._cached_image_plan((2, width, 4), 4, "HWC", "source")
        self.assertEqual(len(adapter._image_plans), 128)
        for key, value in adapter._image_plans.items():
            self.assertIsInstance(key[0], tuple)
            self.assertTrue(all(type(size) is int for size in key[0]))
            self.assertIsInstance(value, tuple)
            self.assertTrue(all(type(size) is int for part in value for size in part))
        for name in ("first-source", "second-source"):
            with self.assertRaisesRegex(ValueError, name):
                adapter._cached_image_plan((4, 4, 4), 4, "auto", name)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
