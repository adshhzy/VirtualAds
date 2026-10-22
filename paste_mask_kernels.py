"""Separable Gaussian feathering for a single full-frame float32 protection mask.

The two kernels use replicate image boundaries and caller-owned output buffers.
They do not implement RGBA motion blur or allocate full-frame intermediates.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def supported(*tensors):
    return triton is not None and all(
        isinstance(tensor, torch.Tensor)
        and tensor.is_cuda
        and tensor.dtype == torch.float32
        and tensor.is_contiguous()
        for tensor in tensors
    )


if triton is not None:

    @triton.jit
    def _horizontal_mask(
        Input,
        Weights,
        Output,
        H: tl.constexpr,
        W: tl.constexpr,
        RADIUS: tl.constexpr,
        BX: tl.constexpr,
        BY: tl.constexpr,
    ):
        x = tl.program_id(0) * BX + tl.arange(0, BX)[None, :]
        y = tl.program_id(1) * BY + tl.arange(0, BY)[:, None]
        active = (x < W) & (y < H)
        result = tl.full((BY, BX), 0.0, tl.float32)
        weight_sum = tl.full((), 0.0, tl.float32)
        for offset in tl.static_range(-RADIUS, RADIUS + 1):
            # Clamp the sampled coordinate, not the output coordinate: this
            # repeats the nearest edge pixel instead of fading frame borders.
            sample_x = tl.minimum(tl.maximum(x + offset, 0), W - 1)
            value = tl.load(Input + y * W + sample_x, mask=active, other=0.0)
            weight = tl.load(Weights + offset + RADIUS)
            result = result + value * weight
            weight_sum = weight_sum + weight
        # The stored float32 coefficients need not sum to exactly one. Use
        # this same accumulation order and correctly rounded division so an
        # all-one neighborhood remains exactly one, including frame edges.
        result = tl.div_rn(result, weight_sum)
        tl.store(Output + y * W + x, result, mask=active)

    @triton.jit
    def _vertical_mask(
        Input,
        Weights,
        Output,
        H: tl.constexpr,
        W: tl.constexpr,
        RADIUS: tl.constexpr,
        BX: tl.constexpr,
        BY: tl.constexpr,
    ):
        x = tl.program_id(0) * BX + tl.arange(0, BX)[None, :]
        y = tl.program_id(1) * BY + tl.arange(0, BY)[:, None]
        active = (x < W) & (y < H)
        result = tl.full((BY, BX), 0.0, tl.float32)
        weight_sum = tl.full((), 0.0, tl.float32)
        for offset in tl.static_range(-RADIUS, RADIUS + 1):
            sample_y = tl.minimum(tl.maximum(y + offset, 0), H - 1)
            value = tl.load(Input + sample_y * W + x, mask=active, other=0.0)
            weight = tl.load(Weights + offset + RADIUS)
            result = result + value * weight
            weight_sum = weight_sum + weight
        result = tl.div_rn(result, weight_sum)
        result = tl.minimum(tl.maximum(result, 0.0), 1.0)
        tl.store(Output + y * W + x, result, mask=active)


def gaussian_mask_into(canvas, temporary, output, weights, *, radius):
    """Write horizontal/vertical passes into disjoint, preallocated HW buffers."""
    tensors = (canvas, temporary, output, weights)
    if not supported(*tensors):
        raise ValueError("mask feathering requires contiguous float32 CUDA tensors and Triton")
    if canvas.ndim != 2 or min(canvas.shape) < 1:
        raise ValueError("canvas must be a nonempty HW tensor")
    if temporary.shape != canvas.shape or output.shape != canvas.shape:
        raise ValueError("temporary and output must match the HW canvas")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a nonnegative Python integer")
    if weights.ndim != 1 or weights.numel() != radius * 2 + 1:
        raise ValueError("weights must have length 2 * radius + 1")
    if any(tensor.device != canvas.device for tensor in tensors[1:]):
        raise ValueError("mask feathering tensors must share one CUDA device")
    if len({tensor.untyped_storage().data_ptr() for tensor in tensors}) != len(tensors):
        raise ValueError("mask feathering input, scratch, output and weights must not alias")
    height, width = canvas.shape
    block_x, block_y = 128, 4
    grid = (triton.cdiv(width, block_x), triton.cdiv(height, block_y))
    _horizontal_mask[grid](
        canvas,
        weights,
        temporary,
        height,
        width,
        radius,
        block_x,
        block_y,
        num_warps=4,
    )
    _vertical_mask[grid](
        temporary,
        weights,
        output,
        height,
        width,
        radius,
        block_x,
        block_y,
        num_warps=4,
    )
