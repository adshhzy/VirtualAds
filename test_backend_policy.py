"""CPU-only metadata checks for conservative additions to the chunk policy.

No tensors, CUDA contexts or rendering are created. Tables are patched locally;
the checked-in additional profile table is populated only from measured results.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import paste_execution_policy as policy


class TensorMetadata:
    def __init__(self, shape, *, strides=None, device="cuda:0", contiguous=True, dtype="float32"):
        self.shape = tuple(shape)
        if strides is None:
            strides = tuple(math.prod(self.shape[index + 1 :]) for index in range(len(self.shape)))
        self._strides = tuple(strides)
        kind, _, index = device.partition(":")
        self.device = SimpleNamespace(type=kind, index=int(index) if index else None)
        self.dtype = dtype
        self._contiguous = contiguous

    def stride(self):
        return self._strides

    def is_contiguous(self):
        return self._contiguous

    def numel(self):
        return math.prod(self.shape)


def fixture(slots=4, source_pattern=((200, 200),), batch=8):
    source_hws = tuple(source_pattern[index % len(source_pattern)] for index in range(slots))
    frames = TensorMetadata((1, 3, 1080, 1920), strides=(6220800, 1, 5760, 3), contiguous=False)
    ads = tuple(TensorMetadata((batch, 4, *hw)) for hw in source_hws)
    grids = tuple(TensorMetadata((1, 108, 320, 2)) for _ in range(slots))
    distortion = TensorMetadata((1, 1080, 1920, 2))
    kernel = TensorMetadata((4, 1, 9, 9))
    mask = TensorMetadata((1, 1, 1080, 1920))
    opacity = tuple(TensorMetadata((1, 1, 1, 1)) for _ in range(slots))
    boxes = tuple((0, 0, 320, 108) for _ in range(slots))
    factors = (TensorMetadata((9,)), TensorMetadata((9,)))
    options = SimpleNamespace(
        backend="batch",
        acceleration="triton",
        blur_backend="separable",
        return_numpy=True,
        pinned_output=True,
        legacy_alpha=True,
        warp_align_corners=False,
        distortion_align_corners=False,
        warp_padding_mode="zeros",
        distortion_padding_mode="zeros",
    )
    return (frames, ads, grids, distortion, kernel, mask, opacity), boxes, options, factors


def profile_key(args, boxes):
    ads = args[1]
    return len(boxes), tuple(ad.shape[-2:] for ad in ads), ads[0].shape[0]


class AdditionalProfileTests(unittest.TestCase):
    def setUp(self):
        environment = patch.object(policy, "_measured_environment", return_value=True)
        self.environment = environment.start()
        self.addCleanup(environment.stop)
        table = patch.object(policy, "ADDITIONAL_PROFILES", {})
        table.start()
        self.addCleanup(table.stop)

    def choose(self, case):
        args, boxes, options, factors = case
        return policy.choose_execution(args, boxes, options, factors)

    def add_profile(self, case, value=(4, None)):
        args, boxes, _, _ = case
        policy.ADDITIONAL_PROFILES[profile_key(args, boxes)] = value

    def assert_fallback(self, case, reason=None):
        chunk, blur, actual_reason = self.choose(case)
        self.assertEqual((chunk, blur), (0, None))
        self.assertNotIn(actual_reason, (policy.PROFILE_ID, policy.ADDITIONAL_PROFILE_ID))
        if reason is not None:
            self.assertEqual(actual_reason, reason)

    def test_empty_table_preserves_original_profiles_and_reasons(self):
        for batch in range(1, 17):
            with self.subTest(batch=batch):
                case = fixture(16, policy.SOURCE_SIZES, batch)
                self.assertEqual(
                    self.choose(case),
                    (policy.CHUNK_BY_BATCH[batch], policy.BLUR_BY_BATCH[batch], policy.PROFILE_ID),
                )
        self.assert_fallback(fixture(), "unmeasured_frame_or_kernel")
        self.assert_fallback(fixture(16, ((200, 200),)), "unmeasured_slot_layout")
        self.assert_fallback(fixture(16, policy.SOURCE_SIZES, 17), "unmeasured_batch")

    def test_exact_additional_entries_for_each_slot_count_and_pattern(self):
        for slots in (4, 16, 32):
            for size_count in (1, 2, 3):
                for batch in (8, 16):
                    if slots == 16 and size_count == 3:
                        continue  # This layout is reserved for the original profile.
                    with self.subTest(slots=slots, size_count=size_count, batch=batch):
                        case = fixture(slots, policy.SOURCE_SIZES[:size_count], batch)
                        self.add_profile(case)
                        self.assertEqual(self.choose(case), (4, None, policy.ADDITIONAL_PROFILE_ID))

    def test_original_profile_cannot_be_overridden_or_bypass_missing_tables(self):
        case = fixture(16, policy.SOURCE_SIZES, 8)
        self.add_profile(case, (7, (64, 4, 4, 64, 4, 4)))
        launch = (128, 4, 4, 128, 4, 4)
        with (
            patch.object(policy, "CHUNK_BY_BATCH", {8: 2}),
            patch.object(policy, "BLUR_BY_BATCH", {8: launch}),
        ):
            self.assertEqual(self.choose(case), (2, launch, policy.PROFILE_ID))
            with patch.object(policy, "BLUR_BY_BATCH", {}):
                self.assert_fallback(case, "unmeasured_batch")
            with patch.object(policy, "CHUNK_BY_BATCH", {}):
                self.assert_fallback(case, "unmeasured_batch")

    def test_additional_entries_pair_chunk_and_blur_independently(self):
        case = fixture()
        launch = (128, 4, 4, 128, 4, 4)
        self.add_profile(case, (4, launch))
        with patch.object(policy, "CHUNK_BY_BATCH", {}), patch.object(policy, "BLUR_BY_BATCH", {}):
            self.assertEqual(self.choose(case), (4, launch, policy.ADDITIONAL_PROFILE_ID))

    def test_unknown_batch_slot_count_size_and_order_fall_back(self):
        case = fixture(4, policy.SOURCE_SIZES[:2], 8)
        self.add_profile(case)
        unknown = (
            fixture(4, policy.SOURCE_SIZES[:2], 16),
            fixture(32, policy.SOURCE_SIZES[:2], 8),
            fixture(4, tuple(reversed(policy.SOURCE_SIZES[:2])), 8),
            fixture(4, ((201, 200), (500, 500)), 8),
            fixture(4, policy.SOURCE_SIZES, 8),
        )
        for candidate in unknown:
            with self.subTest(key=profile_key(candidate[0], candidate[1])):
                self.assert_fallback(candidate)

    def test_frame_broadcast_kernel_and_slot_gates_apply_to_additions(self):
        case = fixture()
        self.add_profile(case)
        args, boxes, options, factors = case
        replacements = (
            (0, TensorMetadata((1, 3, 720, 1280))),
            (0, TensorMetadata((1, 3, 1080, 1920))),
            (3, TensorMetadata((8, 1080, 1920, 2))),
            (3, TensorMetadata((1, 1080, 1920, 2), contiguous=False)),
            (4, TensorMetadata((4, 1, 7, 7))),
            (5, TensorMetadata((1, 3, 1080, 1920))),
            (5, TensorMetadata((1, 1, 1080, 1920), strides=(0, 0, 0, 1))),
        )
        for index, value in replacements:
            with self.subTest(index=index, shape=value.shape):
                changed = list(args)
                changed[index] = value
                self.assert_fallback((tuple(changed), boxes, options, factors))
        for index, value in (
            (1, TensorMetadata((8, 4, 200, 200), contiguous=False)),
            (2, TensorMetadata((8, 108, 320, 2))),
            (2, TensorMetadata((1, 108, 320, 2), contiguous=False)),
            (6, TensorMetadata((8, 1, 1, 1))),
        ):
            with self.subTest(index=index, shape=value.shape):
                changed = list(args)
                changed[index] = (value, *args[index][1:])
                self.assert_fallback((tuple(changed), boxes, options, factors))
        self.assert_fallback((args, ((0, 0, 319, 108), *boxes[1:]), options, factors))

    def test_backend_output_geometry_and_factor_gates_apply_to_additions(self):
        case = fixture()
        self.add_profile(case)
        args, boxes, options, factors = case
        for name, value in (
            ("backend", "bucketed"),
            ("backend", "auto"),
            ("acceleration", "eager"),
            ("blur_backend", "torch"),
            ("return_numpy", False),
            ("pinned_output", False),
            ("legacy_alpha", False),
            ("warp_align_corners", True),
            ("distortion_align_corners", True),
            ("warp_padding_mode", "border"),
            ("distortion_padding_mode", "reflection"),
        ):
            with self.subTest(option=name, value=value):
                changed_options = SimpleNamespace(**{**vars(options), name: value})
                self.assert_fallback((args, boxes, changed_options, factors))
        for changed_factors in (
            None,
            (TensorMetadata((3, 3)), factors[1]),
            (TensorMetadata((9,), strides=(2,)), factors[1]),
            (TensorMetadata((9,), device="cuda:1"), factors[1]),
            (TensorMetadata((9,), dtype="float64"), factors[1]),
        ):
            self.assert_fallback((args, boxes, options, changed_factors))
        cpu_frames = TensorMetadata((1, 3, 1080, 1920), device="cpu")
        self.assert_fallback(((cpu_frames, *args[1:]), boxes, options, factors))
        self.environment.return_value = False
        self.assert_fallback(case, "unmeasured_environment")

    def test_each_call_rechecks_layout_and_preserves_live_bbox_positions(self):
        case = fixture(4, policy.SOURCE_SIZES[:2], 8)
        self.add_profile(case)
        args, boxes, options, factors = case
        moved = tuple((100, 200, 420, 308) for _ in boxes)
        self.assertEqual(
            self.choose((args, moved, options, factors)),
            (4, None, policy.ADDITIONAL_PROFILE_ID),
        )
        resized = ((100, 200, 420, 309), *moved[1:])
        self.assert_fallback((args, resized, options, factors), "unmeasured_slot_layout")
        reversed_ads = (*reversed(args[1]),)
        self.assert_fallback(((args[0], reversed_ads, *args[2:]), boxes, options, factors))
        self.assertEqual(self.choose(case), (4, None, policy.ADDITIONAL_PROFILE_ID))
        policy.ADDITIONAL_PROFILES.clear()
        self.assert_fallback(case, "unmeasured_frame_or_kernel")


class MeasuredTableTests(unittest.TestCase):
    def test_committed_table_selects_only_approved_combinations(self):
        with patch.object(policy, "_measured_environment", return_value=True):
            approved = 0
            for slots in (4, 16, 32):
                for size_count in (1, 2, 3):
                    for batch in (4, 8, 12, 16, 17):
                        case = fixture(slots, policy.SOURCE_SIZES[:size_count], batch)
                        args, boxes, options, factors = case
                        actual = policy.choose_execution(args, boxes, options, factors)
                        original = slots == 16 and size_count == 3 and batch <= 16
                        additional = batch in (8, 16) and (
                            slots == 4 or (slots == 16 and size_count < 3)
                        )
                        with self.subTest(slots=slots, sizes=size_count, batch=batch):
                            if original:
                                self.assertEqual(
                                    actual,
                                    (policy.CHUNK_BY_BATCH[batch], None, policy.PROFILE_ID),
                                )
                            elif additional:
                                approved += 1
                                self.assertEqual(actual, (4, None, policy.ADDITIONAL_PROFILE_ID))
                            else:
                                self.assertEqual(actual[:2], (0, None))
                                self.assertNotIn(
                                    actual[2], (policy.PROFILE_ID, policy.ADDITIONAL_PROFILE_ID)
                                )
            self.assertEqual(approved, len(policy.ADDITIONAL_PROFILES))


if __name__ == "__main__":
    unittest.main()
