"""Optional inference-only separable blur and blend kernels.

Call this path only after the actual 2D kernel has been verified to equal the
outer product ``ky[:, None] * kx[None, :]`` within the chosen numerical tolerance.
The kernel does not infer Gaussian parameters, factorize weights, normalize
them, cache their values, or silently approximate a nonseparable kernel.

Both factors are shared by every RGBA channel and batch element. The first
pass writes a float32 horizontal intermediate; the second pass performs the
vertical convolution and final alpha/mask blend directly to uint8 BHWC.
All convolution accumulation uses explicit float32 FMA. The blend preserves
the perf3 arithmetic order. Dense and separable convolution have different
floating summation orders, so target-GPU numerical and latency checks are
required before enabling this optional path.
"""

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError as exc:
    if exc.name not in ("triton", "triton.language"):
        raise
    triton = None
    tl = None


def available():
    return triton is not None


def supported(*tensors):
    """Metadata only: no numerical scan, CPU readback, or synchronization."""
    if triton is None or not tensors or not all(isinstance(t, torch.Tensor) for t in tensors):
        return False
    device = tensors[0].device
    return all(t.device.type == "cuda" and t.device == device and t.dtype == torch.float32
               and all(stride >= 0 for stride in t.stride())
               and not (torch.is_grad_enabled() and t.requires_grad) for t in tensors)


def separable_blur_blend_supported(canvas, frames, ky, kx, mask):
    if not supported(canvas, frames, ky, kx, mask):
        return False
    if canvas.ndim != 4 or canvas.shape[1] != 4 or min(canvas.shape[0], *canvas.shape[-2:]) < 1:
        return False
    batch, _, height, width = canvas.shape
    if frames.ndim != 4 or frames.shape[1] != 3:
        return False
    if mask.ndim != 4 or mask.shape[1] not in (1, 3):
        return False
    for value in (frames, mask):
        if value.shape[0] not in (1, batch) or value.shape[2] not in (1, height) or value.shape[3] not in (1, width):
            return False
    return all(value.ndim == 1 and 1 <= value.numel() <= 63 and value.numel() % 2 == 1
               for value in (ky, kx))


if triton is not None:
    @triton.jit
    def _horizontal_rgba(
        Canvas, Kx, Temp,
        H: tl.constexpr, W: tl.constexpr,
        C_B: tl.constexpr, C_C: tl.constexpr, C_Y: tl.constexpr, C_X: tl.constexpr,
        K_S: tl.constexpr, KW: tl.constexpr,
        BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr,
    ):
        x = tl.program_id(0)*BLOCK_X + tl.arange(0, BLOCK_X)[None, :]
        y = tl.program_id(1)*BLOCK_Y + tl.arange(0, BLOCK_Y)[:, None]
        batch = tl.program_id(2)
        active = (x < W) & (y < H)
        red = tl.full((BLOCK_Y, BLOCK_X), 0.0, tl.float32)
        green = tl.full((BLOCK_Y, BLOCK_X), 0.0, tl.float32)
        blue = tl.full((BLOCK_Y, BLOCK_X), 0.0, tl.float32)
        alpha = tl.full((BLOCK_Y, BLOCK_X), 0.0, tl.float32)
        for offset in tl.static_range(KW):
            source_x = x + offset - KW//2
            valid = active & (source_x >= 0) & (source_x < W)
            source_offset = batch*C_B + y*C_Y + source_x*C_X
            weight = tl.load(Kx + offset*K_S)
            red = tl.fma(tl.load(Canvas + source_offset, valid, 0.0), weight, red)
            green = tl.fma(tl.load(Canvas + source_offset + C_C, valid, 0.0), weight, green)
            blue = tl.fma(tl.load(Canvas + source_offset + 2*C_C, valid, 0.0), weight, blue)
            alpha = tl.fma(tl.load(Canvas + source_offset + 3*C_C, valid, 0.0), weight, alpha)
        out = batch*4*H*W + y*W + x
        tl.store(Temp + out, red, active)
        tl.store(Temp + out + H*W, green, active)
        tl.store(Temp + out + 2*H*W, blue, active)
        tl.store(Temp + out + 3*H*W, alpha, active)


    @triton.jit
    def _vertical_blend_u8(
        Temp, Frames, Ky, Mask, Out,
        H: tl.constexpr, W: tl.constexpr,
        F_B: tl.constexpr, F_C: tl.constexpr, F_Y: tl.constexpr, F_X: tl.constexpr,
        M_B: tl.constexpr, M_C: tl.constexpr, M_Y: tl.constexpr, M_X: tl.constexpr,
        K_S: tl.constexpr, KH: tl.constexpr, LEGACY: tl.constexpr,
        BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr,
    ):
        x = tl.program_id(0)*BLOCK_X + tl.arange(0, BLOCK_X)[None, :]
        y = tl.program_id(1)*BLOCK_Y + tl.arange(0, BLOCK_Y)[:, None]
        batch = tl.program_id(2)
        active = (x < W) & (y < H)
        red = tl.full((BLOCK_Y, BLOCK_X), 0.0, tl.float32)
        green = tl.full((BLOCK_Y, BLOCK_X), 0.0, tl.float32)
        blue = tl.full((BLOCK_Y, BLOCK_X), 0.0, tl.float32)
        alpha = tl.full((BLOCK_Y, BLOCK_X), 0.0, tl.float32)
        for offset in tl.static_range(KH):
            source_y = y + offset - KH//2
            valid = active & (source_y >= 0) & (source_y < H)
            source_offset = batch*4*H*W + source_y*W + x
            weight = tl.load(Ky + offset*K_S)
            red = tl.fma(tl.load(Temp + source_offset, valid, 0.0), weight, red)
            green = tl.fma(tl.load(Temp + source_offset + H*W, valid, 0.0), weight, green)
            blue = tl.fma(tl.load(Temp + source_offset + 2*H*W, valid, 0.0), weight, blue)
            alpha = tl.fma(tl.load(Temp + source_offset + 3*H*W, valid, 0.0), weight, alpha)
        frame_offset = batch*F_B + y*F_Y + x*F_X
        mask_offset = batch*M_B + y*M_Y + x*M_X
        bg_r = tl.load(Frames + frame_offset, active, 0.0)
        bg_g = tl.load(Frames + frame_offset + F_C, active, 0.0)
        bg_b = tl.load(Frames + frame_offset + 2*F_C, active, 0.0)
        mask_r = tl.load(Mask + mask_offset, active, 0.0)
        mask_g = tl.load(Mask + mask_offset + M_C, active, 0.0)
        mask_b = tl.load(Mask + mask_offset + 2*M_C, active, 0.0)
        if LEGACY:
            red, green, blue = red*alpha, green*alpha, blue*alpha
        red = red + bg_r*(1.0 - alpha)
        green = green + bg_g*(1.0 - alpha)
        blue = blue + bg_b*(1.0 - alpha)
        red = bg_r*mask_r + red*(1.0 - mask_r)
        green = bg_g*mask_g + green*(1.0 - mask_g)
        blue = bg_b*mask_b + blue*(1.0 - mask_b)
        red = tl.where(red == red, red, 0.0)
        green = tl.where(green == green, green, 0.0)
        blue = tl.where(blue == blue, blue, 0.0)
        red = tl.minimum(tl.maximum(red, 0.0), 255.0).to(tl.uint8)
        green = tl.minimum(tl.maximum(green, 0.0), 255.0).to(tl.uint8)
        blue = tl.minimum(tl.maximum(blue, 0.0), 255.0).to(tl.uint8)
        output_offset = ((batch*H + y)*W + x)*3
        tl.store(Out + output_offset, red, active)
        tl.store(Out + output_offset + 1, green, active)
        tl.store(Out + output_offset + 2, blue, active)


def separable_blur_blend_to_uint8(canvas, frames, ky, kx, mask, *, legacy_alpha=True):
    """Two-pass shared depthwise blur, then uint8 BHWC output.

    Factors are one-dimensional float32 CUDA tensors of odd lengths <=63.
    Their values are read on every invocation. Padding is zero, stride and
    dilation are one, and weights are not flipped (torch.conv2d convention).
    No input is modified; the output owns independent device storage.
    """
    if not separable_blur_blend_supported(canvas, frames, ky, kx, mask):
        raise ValueError("separable blur requires float32 CUDA inference tensors, broadcastable frames/mask, and 1D odd factors <=63 shared by all channels/batches")
    batch, _, height, width = canvas.shape
    frames = frames.expand(batch, 3, height, width)
    mask = mask.expand(batch, 3, height, width)
    temp = torch.empty((batch, 4, height, width), device=canvas.device, dtype=torch.float32)
    output = torch.empty((batch, height, width, 3), device=canvas.device, dtype=torch.uint8)
    block_x, block_y = 128, 4
    grid = (triton.cdiv(width, block_x), triton.cdiv(height, block_y), batch)
    with torch.cuda.device(canvas.device):
        _horizontal_rgba[grid](
            canvas, kx, temp, H=height, W=width,
            C_B=canvas.stride(0), C_C=canvas.stride(1), C_Y=canvas.stride(2), C_X=canvas.stride(3),
            K_S=kx.stride(0), KW=kx.numel(), BLOCK_X=block_x, BLOCK_Y=block_y,
            num_warps=4, enable_fp_fusion=False,
        )
        _vertical_blend_u8[grid](
            temp, frames, ky, mask, output, H=height, W=width,
            F_B=frames.stride(0), F_C=frames.stride(1), F_Y=frames.stride(2), F_X=frames.stride(3),
            M_B=mask.stride(0), M_C=mask.stride(1), M_Y=mask.stride(2), M_X=mask.stride(3),
            K_S=ky.stride(0), KH=ky.numel(), LEGACY=legacy_alpha,
            BLOCK_X=block_x, BLOCK_Y=block_y, num_warps=4, enable_fp_fusion=False,
        )
    return output
