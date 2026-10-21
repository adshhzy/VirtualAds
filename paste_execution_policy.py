"""Conservative, offline-measured execution profile. No online autotuning.

Only tensor metadata is inspected. Camera/grid/kernel values stay live. Unknown
hardware, shapes, layouts or batches use the original whole-batch path.
See DYNAMIC_BATCH_REVIEW.md for measurements and reproduction commands.
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


@lru_cache(maxsize=16)
def _measured_environment(device_index):
    import torch
    if sys.platform != "win32" or torch.__version__ != "2.9.1+cu128" or torch.version.cuda != "12.8":
        return False
    try:
        import triton
        from importlib.metadata import version
        if triton.__version__ != "3.5.1" or version("triton-windows") != "3.5.1.post24":
            return False
    except (ImportError, ModuleNotFoundError):
        return False
    return (torch.cuda.get_device_name(device_index) == "NVIDIA GeForce RTX 4090"
            and torch.cuda.get_device_capability(device_index) == (8, 9))


def choose_execution(args, boxes, options, blur_factors):
    """Return (chunk_size, blur_launch_config, reason); zero means whole batch."""
    frames, ads, grids, distortion, kernel, mask, opacity = args
    if (options.backend != "batch" or options.acceleration not in ("auto", "triton")
            or options.blur_backend != "separable" or not options.return_numpy
            or not options.pinned_output or frames.device.type != "cuda"):
        return 0, None, "unsupported_backend_or_output"
    batch = ads[0].shape[0]
    if batch not in CHUNK_BY_BATCH or batch not in BLUR_BY_BATCH:
        return 0, None, "unmeasured_batch"
    if (tuple(frames.shape) != (1, 3, 1080, 1920) or frames.stride()[1:] != (1, 5760, 3)
            or tuple(kernel.shape) != (4, 1, 9, 9) or blur_factors is None
            or tuple(k.numel() for k in blur_factors) != (9, 9)
            or any(tuple(k.shape) != (9,) or k.stride() != (1,) or k.device != frames.device
                   or k.dtype != frames.dtype for k in blur_factors)
            or len(boxes) != 16 or not options.legacy_alpha
            or options.warp_align_corners or options.distortion_align_corners
            or options.warp_padding_mode != "zeros" or options.distortion_padding_mode != "zeros"):
        return 0, None, "unmeasured_frame_or_kernel"
    if (tuple(mask.shape) != (1, 1, 1080, 1920) or mask.stride()[-2:] != (1920, 1)
            or tuple(distortion.shape) != (1, 1080, 1920, 2) or not distortion.is_contiguous()):
        return 0, None, "unmeasured_broadcast_layout"
    for i, (ad, grid, box, alpha) in enumerate(zip(ads, grids, boxes, opacity)):
        if (tuple(ad.shape[-2:]) != SOURCE_SIZES[i % len(SOURCE_SIZES)] or not ad.is_contiguous()
                or (box[3] - box[1], box[2] - box[0]) != (108, 320)
                or tuple(grid.shape) != (1, 108, 320, 2) or not grid.is_contiguous()
                or tuple(alpha.shape) != (1, 1, 1, 1)):
            return 0, None, "unmeasured_slot_layout"
    if not _measured_environment(frames.device.index):
        return 0, None, "unmeasured_environment"
    return CHUNK_BY_BATCH[batch], BLUR_BY_BATCH.get(batch), PROFILE_ID
