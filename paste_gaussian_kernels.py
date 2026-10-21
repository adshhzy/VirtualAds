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

from numbers import Integral

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
    return all(
        t.device.type == "cuda"
        and t.device == device
        and t.dtype == torch.float32
        and all(stride >= 0 for stride in t.stride())
        and not (torch.is_grad_enabled() and t.requires_grad)
        for t in tensors
    )


def separable_blur_blend_supported(canvas, frames, ky, kx, mask):
    """Check the shared-factor route using only shape, stride and dtype metadata.

    canvas supplies the output B/H/W. Background and mask may broadcast in
    batch or spatial dimensions; both factors apply to all RGBA channels.
    Factorization and numerical validation belong to the parameter builder.
    """
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
        if (
            value.shape[0] not in (1, batch)
            or value.shape[2] not in (1, height)
            or value.shape[3] not in (1, width)
        ):
            return False
    return all(
        value.ndim == 1 and 1 <= value.numel() <= 63 and value.numel() % 2 == 1
        for value in (ky, kx)
    )


_DEFAULT_LAUNCH_CONFIG = (128, 4, 4, 128, 4, 4)


def _resolve_launch_config(launch_config):
    """Return two (block_x, block_y, warps) triples without querying CUDA.

    The first triple configures the horizontal pass; the second configures
    the vertical pass. None always means the original launch settings.
    """
    if launch_config is None:
        return _DEFAULT_LAUNCH_CONFIG
    if (
        not isinstance(launch_config, (tuple, list))
        or len(launch_config) != 6
        or any(
            isinstance(value, bool) or not isinstance(value, Integral) for value in launch_config
        )
    ):
        raise ValueError(
            "launch_config must contain six integers: (h_block_x, h_block_y, h_warps, v_block_x, v_block_y, v_warps)"
        )
    config = tuple(int(value) for value in launch_config)
    for stage, (block_x, block_y, warps) in zip(
        ("horizontal", "vertical"), (config[:3], config[3:])
    ):
        if (
            block_x not in (32, 64, 128, 256, 512)
            or block_y not in (1, 2, 4, 8, 16, 32)
            or warps not in (1, 2, 4, 8)
            or block_x * block_y > 4096
        ):
            raise ValueError(
                f"{stage} launch requires block_x in (32,64,128,256,512), block_y in (1,2,4,8,16,32), warps in (1,2,4,8), and block_x*block_y <=4096"
            )
    return config


if triton is not None:

    @triton.jit
    def _horizontal_rgba(
        Canvas,
        Kx,
        Temp,
        H: tl.constexpr,
        W: tl.constexpr,
        C_B: tl.constexpr,
        C_C: tl.constexpr,
        C_Y: tl.constexpr,
        C_X: tl.constexpr,
        K_S: tl.constexpr,
        KW: tl.constexpr,
        BLOCK_X: tl.constexpr,
        BLOCK_Y: tl.constexpr,
    ):
        # One program covers a [BLOCK_Y, BLOCK_X] tile in one batch group.
        # Temp is contiguous BCHW even when Canvas has noncontiguous strides.
        x = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)[None, :]
        y = tl.program_id(1) * BLOCK_Y + tl.arange(0, BLOCK_Y)[:, None]
        batch = tl.program_id(2)
        active = (x < W) & (y < H)
        red = tl.full((BLOCK_Y, BLOCK_X), 0.0, tl.float32)
        green = tl.full((BLOCK_Y, BLOCK_X), 0.0, tl.float32)
        blue = tl.full((BLOCK_Y, BLOCK_X), 0.0, tl.float32)
        alpha = tl.full((BLOCK_Y, BLOCK_X), 0.0, tl.float32)
        for offset in tl.static_range(KW):
            source_x = x + offset - KW // 2
            valid = active & (source_x >= 0) & (source_x < W)
            source_offset = batch * C_B + y * C_Y + source_x * C_X
            weight = tl.load(Kx + offset * K_S)
            red = tl.fma(tl.load(Canvas + source_offset, valid, 0.0), weight, red)
            green = tl.fma(tl.load(Canvas + source_offset + C_C, valid, 0.0), weight, green)
            blue = tl.fma(tl.load(Canvas + source_offset + 2 * C_C, valid, 0.0), weight, blue)
            alpha = tl.fma(tl.load(Canvas + source_offset + 3 * C_C, valid, 0.0), weight, alpha)
        out = batch * 4 * H * W + y * W + x
        tl.store(Temp + out, red, active)
        tl.store(Temp + out + H * W, green, active)
        tl.store(Temp + out + 2 * H * W, blue, active)
        tl.store(Temp + out + 3 * H * W, alpha, active)

    @triton.jit
    def _load_spatial_plane(
        Base, x, y, H: tl.constexpr, W: tl.constexpr, S_Y: tl.constexpr, S_X: tl.constexpr
    ):
        # Load only the stored spatial dimensions, then let the arithmetic
        # broadcast the VALUES. Broadcasting zero-stride pointers to a 2-D
        # tile before a masked load can fail Triton's coalescing pass.
        if S_Y == 0 and S_X == 0:
            value = tl.load(Base)
        elif S_Y == 0:
            value = tl.load(Base + x * S_X, x < W, 0.0)
        elif S_X == 0:
            value = tl.load(Base + y * S_Y, y < H, 0.0)
        else:
            value = tl.load(Base + y * S_Y + x * S_X, (x < W) & (y < H), 0.0)
        return value

    @triton.jit
    def _vertical_blend_u8(
        Temp,
        Frames,
        Ky,
        Mask,
        Out,
        H: tl.constexpr,
        W: tl.constexpr,
        F_B: tl.constexpr,
        F_C: tl.constexpr,
        F_Y: tl.constexpr,
        F_X: tl.constexpr,
        M_B: tl.constexpr,
        M_C: tl.constexpr,
        M_Y: tl.constexpr,
        M_X: tl.constexpr,
        K_S: tl.constexpr,
        KH: tl.constexpr,
        LEGACY: tl.constexpr,
        BLOCK_X: tl.constexpr,
        BLOCK_Y: tl.constexpr,
    ):
        # Consume horizontal FP32 values, then write three interleaved output
        # channels. Keep convolution FMA and the later blend rounding separate.
        x = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)[None, :]
        y = tl.program_id(1) * BLOCK_Y + tl.arange(0, BLOCK_Y)[:, None]
        batch = tl.program_id(2)
        active = (x < W) & (y < H)
        red = tl.full((BLOCK_Y, BLOCK_X), 0.0, tl.float32)
        green = tl.full((BLOCK_Y, BLOCK_X), 0.0, tl.float32)
        blue = tl.full((BLOCK_Y, BLOCK_X), 0.0, tl.float32)
        alpha = tl.full((BLOCK_Y, BLOCK_X), 0.0, tl.float32)
        for offset in tl.static_range(KH):
            source_y = y + offset - KH // 2
            valid = active & (source_y >= 0) & (source_y < H)
            source_offset = batch * 4 * H * W + source_y * W + x
            weight = tl.load(Ky + offset * K_S)
            red = tl.fma(tl.load(Temp + source_offset, valid, 0.0), weight, red)
            green = tl.fma(tl.load(Temp + source_offset + H * W, valid, 0.0), weight, green)
            blue = tl.fma(tl.load(Temp + source_offset + 2 * H * W, valid, 0.0), weight, blue)
            alpha = tl.fma(tl.load(Temp + source_offset + 3 * H * W, valid, 0.0), weight, alpha)
        frame_base = Frames + batch * F_B
        mask_base = Mask + batch * M_B
        bg_r = _load_spatial_plane(frame_base, x, y, H, W, F_Y, F_X)
        bg_g = _load_spatial_plane(frame_base + F_C, x, y, H, W, F_Y, F_X)
        bg_b = _load_spatial_plane(frame_base + 2 * F_C, x, y, H, W, F_Y, F_X)
        mask_r = _load_spatial_plane(mask_base, x, y, H, W, M_Y, M_X)
        mask_g = _load_spatial_plane(mask_base + M_C, x, y, H, W, M_Y, M_X)
        mask_b = _load_spatial_plane(mask_base + 2 * M_C, x, y, H, W, M_Y, M_X)
        if LEGACY:
            red, green, blue = red * alpha, green * alpha, blue * alpha
        red = red + bg_r * (1.0 - alpha)
        green = green + bg_g * (1.0 - alpha)
        blue = blue + bg_b * (1.0 - alpha)
        red = bg_r * mask_r + red * (1.0 - mask_r)
        green = bg_g * mask_g + green * (1.0 - mask_g)
        blue = bg_b * mask_b + blue * (1.0 - mask_b)
        red = tl.where(red == red, red, 0.0)
        green = tl.where(green == green, green, 0.0)
        blue = tl.where(blue == blue, blue, 0.0)
        red = tl.minimum(tl.maximum(red, 0.0), 255.0).to(tl.uint8)
        green = tl.minimum(tl.maximum(green, 0.0), 255.0).to(tl.uint8)
        blue = tl.minimum(tl.maximum(blue, 0.0), 255.0).to(tl.uint8)
        output_offset = ((batch * H + y) * W + x) * 3
        tl.store(Out + output_offset, red, active)
        tl.store(Out + output_offset + 1, green, active)
        tl.store(Out + output_offset + 2, blue, active)


def separable_blur_blend_to_uint8(
    canvas, frames, ky, kx, mask, *, legacy_alpha=True, launch_config=None
):
    """Two-pass shared depthwise blur, then uint8 BHWC output.

    Factors are one-dimensional float32 CUDA tensors of odd lengths <=63.
    Their values are read on every invocation. Padding is zero, stride and
    dilation are one, and weights are not flipped (torch.conv2d convention).
    No input is modified; the output owns independent device storage.

    canvas is [B,4,H,W]; frames and mask broadcast to [B,3,H,W]. The first
    pass produces a separate FP32 BCHW temporary, the second produces uint8
    BHWC. Expansion preserves input strides; no implicit contiguous copy is
    made. Host export and synchronization are the adapter's responsibility.

    launch_config optionally supplies (h_block_x, h_block_y, h_warps,
    v_block_x, v_block_y, v_warps) for offline tuning. None preserves the
    original 128x4 tile and four warps for each pass. Each block_x must be
    32/64/128/256/512, block_y 1/2/4/8/16/32, and warps 1/2/4/8, with at most
    4096 pixels per tile. Configurations still need target-GPU validation;
    this function neither benchmarks nor selects a configuration at runtime.
    """
    (
        horizontal_block_x,
        horizontal_block_y,
        horizontal_warps,
        vertical_block_x,
        vertical_block_y,
        vertical_warps,
    ) = _resolve_launch_config(launch_config)
    if not separable_blur_blend_supported(canvas, frames, ky, kx, mask):
        raise ValueError(
            "separable blur requires float32 CUDA inference tensors, broadcastable frames/mask, and 1D odd factors <=63 shared by all channels/batches"
        )
    batch, _, height, width = canvas.shape
    frames = frames.expand(batch, 3, height, width)
    mask = mask.expand(batch, 3, height, width)
    temp = torch.empty((batch, 4, height, width), device=canvas.device, dtype=torch.float32)
    output = torch.empty((batch, height, width, 3), device=canvas.device, dtype=torch.uint8)
    horizontal_grid = (
        triton.cdiv(width, horizontal_block_x),
        triton.cdiv(height, horizontal_block_y),
        batch,
    )
    vertical_grid = (
        triton.cdiv(width, vertical_block_x),
        triton.cdiv(height, vertical_block_y),
        batch,
    )
    with torch.cuda.device(canvas.device):
        _horizontal_rgba[horizontal_grid](
            canvas,
            kx,
            temp,
            H=height,
            W=width,
            C_B=canvas.stride(0),
            C_C=canvas.stride(1),
            C_Y=canvas.stride(2),
            C_X=canvas.stride(3),
            K_S=kx.stride(0),
            KW=kx.numel(),
            BLOCK_X=horizontal_block_x,
            BLOCK_Y=horizontal_block_y,
            num_warps=horizontal_warps,
            enable_fp_fusion=False,
        )
        _vertical_blend_u8[vertical_grid](
            temp,
            frames,
            ky,
            mask,
            output,
            H=height,
            W=width,
            F_B=frames.stride(0),
            F_C=frames.stride(1),
            F_Y=frames.stride(2),
            F_X=frames.stride(3),
            M_B=mask.stride(0),
            M_C=mask.stride(1),
            M_Y=mask.stride(2),
            M_X=mask.stride(3),
            K_S=ky.stride(0),
            KH=ky.numel(),
            LEGACY=legacy_alpha,
            BLOCK_X=vertical_block_x,
            BLOCK_Y=vertical_block_y,
            num_warps=vertical_warps,
            enable_fp_fusion=False,
        )
    return output
