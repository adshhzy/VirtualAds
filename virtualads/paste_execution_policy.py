"""Conservative, offline-measured execution profile. No online autotuning.

Only tensor metadata is inspected. Camera/grid/kernel values stay live. Unknown
hardware, shapes, layouts or batches use the original whole-batch path.
See DYNAMIC_BATCH_REVIEW.md and BACKEND_COMPARISON_REVIEW.md for measurements.
"""

from functools import lru_cache
import sys

PROFILE_ID = "4090-win-cu128-1080p-k9-v1"
# Filled from end-to-end measurements, never from isolated kernel timings.
CHUNK_BY_BATCH = {batch: (0 if batch <= 5 else 4) for batch in range(1, 17)}
# Explicit entries: each chunk decision is paired with the measured blur setup.
# None retains the original 128x4/four-warps kernels.
BLUR_BY_BATCH = {batch: None for batch in CHUNK_BY_BATCH}
SOURCE_SIZES = ((200, 200), (500, 500), (300, 600))

# Additional layouts passed full-call mean and P95 gains >=5% in both static
# and translated layouts, all four rounds faster, with bit-exact checked frames.
# A key is (slot_count, ordered_source_hws, batch), where
# ordered_source_hws contains one (H, W) pair per slot, including repetitions.
# Values pair the chunk size and blur launch configuration atomically.
# 32-slot cases did not pass both latency gates. The original 16-slot/three-size
# profile above retains its previously measured batch range and takes priority.
ADDITIONAL_PROFILE_ID = "4090-win-cu128-1080p-k9-layouts-v1"
ADDITIONAL_PROFILES = {
    (slots, tuple(SOURCE_SIZES[index % size_count] for index in range(slots)), batch): (4, None)
    for slots, size_count in ((4, 1), (4, 2), (4, 3), (16, 1), (16, 2))
    for batch in (8, 16)
}


@lru_cache(maxsize=16)
def _measured_environment(device_index):
    """Match the exact software/GPU used for the offline timing table."""
    import torch

    if (
        sys.platform != "win32"
        or torch.__version__ != "2.9.1+cu128"
        or torch.version.cuda != "12.8"
    ):
        return False
    try:
        import triton
        from importlib.metadata import version

        if triton.__version__ != "3.5.1" or version("triton-windows") != "3.5.1.post24":
            return False
    except (ImportError, ModuleNotFoundError):
        return False
    return torch.cuda.get_device_name(
        device_index
    ) == "NVIDIA GeForce RTX 4090" and torch.cuda.get_device_capability(device_index) == (8, 9)


def _supports_profile_backend(frames, options):
    """The measured path ends in independent pinned NumPy outputs on CUDA."""
    return (
        options.backend == "batch"
        and options.acceleration in ("auto", "triton")
        and options.blur_backend == "separable"
        and options.return_numpy
        and options.pinned_output
        and frames.device.type == "cuda"
    )


def _matches_frame_and_kernel(frames, kernel, boxes, options, blur_factors, *, slot_count=16):
    """Require the measured shared 1080p background and shared 9x9 kernel."""
    # NCHW view over one HWC frame: RGB is the innermost storage dimension.
    if tuple(frames.shape) != (1, 3, 1080, 1920) or frames.stride()[1:] != (1, 5760, 3):
        return False
    if tuple(kernel.shape) != (4, 1, 9, 9) or blur_factors is None:
        return False
    if tuple(factor.numel() for factor in blur_factors) != (9, 9):
        return False
    for factor in blur_factors:
        if (
            tuple(factor.shape) != (9,)
            or factor.stride() != (1,)
            or factor.device != frames.device
            or factor.dtype != frames.dtype
        ):
            return False
    return (
        len(boxes) == slot_count
        and options.legacy_alpha
        and not options.warp_align_corners
        and not options.distortion_align_corners
        and options.warp_padding_mode == "zeros"
        and options.distortion_padding_mode == "zeros"
    )


def _matches_shared_layout(mask, distortion):
    """Require one scalar-channel mask and one contiguous distortion grid."""
    if (
        tuple(mask.shape) != (1, 1, 1080, 1920)
        or mask.stride()[-2:] != (1920, 1)
        or tuple(distortion.shape) != (1, 1080, 1920, 2)
        or not distortion.is_contiguous()
    ):
        return False
    return True


def _matches_slot_layout(ads, grids, boxes, opacity, *, source_sizes=None):
    """Match source-size order, shared ROI grids and scalar slot opacity."""
    if source_sizes is None:
        source_sizes = SOURCE_SIZES
    for slot_index, (ad, grid, box, alpha) in enumerate(zip(ads, grids, boxes, opacity)):
        if (
            tuple(ad.shape[-2:]) != source_sizes[slot_index % len(source_sizes)]
            or not ad.is_contiguous()
            or (box[3] - box[1], box[2] - box[0]) != (108, 320)
            or tuple(grid.shape) != (1, 108, 320, 2)
            or not grid.is_contiguous()
            or tuple(alpha.shape) != (1, 1, 1, 1)
        ):
            return False
    return True


def _additional_profile(ads, boxes, batch):
    """Look up an exact extra layout without overriding the original profile.

    The original 16-slot/three-size layout always uses CHUNK_BY_BATCH and
    BLUR_BY_BATCH, including their missing-entry fallbacks. Camera values,
    pixels and bbox positions are not read or retained by this lookup.
    """
    source_hws = tuple(tuple(ad.shape[-2:]) for ad in ads)
    original_sources = tuple(SOURCE_SIZES[index % len(SOURCE_SIZES)] for index in range(16))
    if len(boxes) == 16 and source_hws == original_sources:
        return None, None
    profile = ADDITIONAL_PROFILES.get((len(boxes), source_hws, batch))
    return profile, source_hws if profile is not None else None


def choose_execution(args, boxes, options, blur_factors):
    """Return ``(chunk_size, blur_launch_config, reason)``; zero means whole B.

    ``args`` is the adapter's validated NCHW tensor tuple. These checks select
    a measured layout; they do not validate values or change rendering inputs.
    Keep the gate order stable because callers record the first fallback reason.
    """
    frames, ads, grids, distortion, kernel, mask, opacity = args
    if not _supports_profile_backend(frames, options):
        return 0, None, "unsupported_backend_or_output"
    batch = ads[0].shape[0]
    profile, source_sizes = (None, None)
    if ADDITIONAL_PROFILES:
        profile, source_sizes = _additional_profile(ads, boxes, batch)
    if profile is None and (batch not in CHUNK_BY_BATCH or batch not in BLUR_BY_BATCH):
        return 0, None, "unmeasured_batch"
    slot_count = 16 if profile is None else len(boxes)
    if not _matches_frame_and_kernel(
        frames, kernel, boxes, options, blur_factors, slot_count=slot_count
    ):
        return 0, None, "unmeasured_frame_or_kernel"
    if not _matches_shared_layout(mask, distortion):
        return 0, None, "unmeasured_broadcast_layout"
    if not _matches_slot_layout(ads, grids, boxes, opacity, source_sizes=source_sizes):
        return 0, None, "unmeasured_slot_layout"
    if not _measured_environment(frames.device.index):
        return 0, None, "unmeasured_environment"
    if profile is not None:
        chunk_size, blur_config = profile
        return chunk_size, blur_config, ADDITIONAL_PROFILE_ID
    return CHUNK_BY_BATCH[batch], BLUR_BY_BATCH.get(batch), PROFILE_ID
