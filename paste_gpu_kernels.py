"""Optional inference-only Triton kernels for paste_gpu_batch.

Install the Triton version supplied/supported by your PyTorch installation.
Importing this module remains safe when Triton is absent. The public wrappers
accept float32 CUDA tensors, preserve arbitrary nonnegative strides, and launch
on PyTorch's current stream for the tensor's device. No CPU tensor reads occur.

In the warp/homography paths, ROI positions, ROI sizes and tensor strides are
runtime kernel arguments, not constexprs: moving/resizing a bbox does not create
one kernel per bbox. Modes and block size are specialized. CUDA Graph replay still requires fixed buffers
and launch dimensions; dynamic bbox handling here is a JIT feature, not Graph
replay support.

grid_sample reflection and distortion stay on the PyTorch path. The optional
blur_blend_to_uint8 path fuses odd, at-most-15x15 depthwise blur with the final
blend/output. It reads the actual shared/per-batch, per-channel kernel each call;
it assumes neither separability nor sparsity, and caches no kernel values.
Canvas/layout strides and blur kernel SHAPE are compile-time specializations in
this path; changing only weights does not recompile. All arithmetic is float32,
with zero padding and convolution-as-cross-correlation as in torch.conv2d.
Numerical regression against the target GPU/PyTorch is required before deployment;
accumulation order and fused arithmetic may differ at uint8 thresholds.
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
    """Metadata-only check; no synchronization and no value/finite scans."""
    if triton is None or not tensors:
        return False
    device = tensors[0].device
    return all(
        tensor.device.type == "cuda"
        and tensor.device == device
        and tensor.dtype == torch.float32
        and all(stride >= 0 for stride in tensor.stride())
        and not (torch.is_grad_enabled() and tensor.requires_grad)
        for tensor in tensors
    )


def blur_blend_supported(canvas, frames, kernel, mask):
    """Metadata-only eligibility for optional dense blur + blend fusion.

    No finite/normalization/value checks or device-to-host reads are performed.
    Inputs must be finite. No restriction on kernel signs or channel equality.
    """
    if not supported(canvas, frames, kernel, mask):
        return False
    if canvas.ndim != 4 or canvas.shape[1] != 4:
        return False
    batch, _, height, width = canvas.shape
    if min(batch, height, width) < 1:
        return False
    if frames.ndim != 4 or frames.shape[1] != 3:
        return False
    if mask.ndim != 4 or mask.shape[1] not in (1, 3):
        return False
    for image in (frames, mask):
        if (
            image.shape[0] not in (1, batch)
            or image.shape[2] not in (1, height)
            or image.shape[3] not in (1, width)
        ):
            return False
    if kernel.ndim == 4:
        if kernel.shape[:2] != (4, 1):
            return False
    elif kernel.ndim == 5:
        if kernel.shape[:3] != (batch, 4, 1):
            return False
    else:
        return False
    return all(1 <= size <= 15 and size % 2 == 1 for size in kernel.shape[-2:])


if triton is not None:

    @triton.jit
    def _four_samples(S, base, Y0, X0, Y1, X1, S_Y, S_X, H, W, valid):
        # Return top-left, top-right, bottom-left, bottom-right in the same
        # order used by the four bilinear weights; out-of-image samples are 0.
        a = tl.load(
            S + base + Y0 * S_Y + X0 * S_X, valid & (Y0 >= 0) & (Y0 < H) & (X0 >= 0) & (X0 < W), 0
        )
        b = tl.load(
            S + base + Y0 * S_Y + X1 * S_X, valid & (Y0 >= 0) & (Y0 < H) & (X1 >= 0) & (X1 < W), 0
        )
        c = tl.load(
            S + base + Y1 * S_Y + X0 * S_X, valid & (Y1 >= 0) & (Y1 < H) & (X0 >= 0) & (X0 < W), 0
        )
        d = tl.load(
            S + base + Y1 * S_Y + X1 * S_X, valid & (Y1 >= 0) & (Y1 < H) & (X1 >= 0) & (X1 < W), 0
        )
        return a, b, c, d

    @triton.jit(
        do_not_specialize=[
            "N",
            "ROI_H",
            "ROI_W",
            "X_START",
            "Y_START",
            "SRC_H",
            "SRC_W",
            "S_B",
            "S_C",
            "S_Y",
            "S_X",
            "G_B",
            "G_Y",
            "G_X",
            "G_C",
            "O_B",
            "O_Y",
            "O_X",
            "D_B",
            "D_C",
            "D_Y",
            "D_X",
        ]
    )
    def _warp_paste(
        Source,
        Grid,
        Opacity,
        Dest,
        N,
        ROI_H,
        ROI_W,
        X_START,
        Y_START,
        SRC_H,
        SRC_W,
        S_B,
        S_C,
        S_Y,
        S_X,
        G_B,
        G_Y,
        G_X,
        G_C,
        O_B,
        O_Y,
        O_X,
        D_B,
        D_C,
        D_Y,
        D_X,
        ALIGN: tl.constexpr,
        BORDER: tl.constexpr,
        LEGACY: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # Flatten [B, ROI_H, ROI_W]; each lane writes all four RGBA channels.
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = idx < N
        local_x = idx % ROI_W
        local_y = (idx // ROI_W) % ROI_H
        batch = idx // (ROI_H * ROI_W)
        gbase = batch * G_B + local_y * G_Y + local_x * G_X
        gx = tl.load(Grid + gbase, active, 0)
        gy = tl.load(Grid + gbase + G_C, active, 0)
        # Match the documented grid_sample NaN-coordinate convention.
        gx = tl.where(gx == gx, gx, -1.0)
        gy = tl.where(gy == gy, gy, -1.0)
        if ALIGN:
            px = ((gx + 1.0) * 0.5) * (SRC_W - 1)
            py = ((gy + 1.0) * 0.5) * (SRC_H - 1)
        else:
            px = ((gx + 1.0) * SRC_W - 1.0) * 0.5
            py = ((gy + 1.0) * SRC_H - 1.0) * 0.5
        if BORDER:
            px = tl.minimum(tl.maximum(px, 0.0), SRC_W - 1.0)
            py = tl.minimum(tl.maximum(py, 0.0), SRC_H - 1.0)
            valid = active
        else:
            # Out-of-image values must never overflow float->integer conversion.
            valid = active & (px > -1.0) & (px < SRC_W) & (py > -1.0) & (py < SRC_H)
            px = tl.where(valid, px, 0.0)
            py = tl.where(valid, py, 0.0)
        x0 = tl.floor(px).to(tl.int32)
        y0 = tl.floor(py).to(tl.int32)
        x1, y1 = x0 + 1, y0 + 1
        w00 = (x1.to(tl.float32) - px) * (y1.to(tl.float32) - py)
        w01 = (px - x0.to(tl.float32)) * (y1.to(tl.float32) - py)
        w10 = (x1.to(tl.float32) - px) * (py - y0.to(tl.float32))
        w11 = (px - x0.to(tl.float32)) * (py - y0.to(tl.float32))
        a00, a01, a10, a11 = _four_samples(
            Source,
            batch * S_B + 3 * S_C,
            y0,
            x0,
            y1,
            x1,
            S_Y,
            S_X,
            SRC_H,
            SRC_W,
            valid,
        )
        opacity = tl.load(Opacity + batch * O_B + local_y * O_Y + local_x * O_X, active, 0)
        if LEGACY:
            sampled_alpha = ((a00 * w00 + a01 * w01) + a10 * w10) + a11 * w11
            alpha = (sampled_alpha * (1.0 / 255.0)) * opacity
        else:
            # Corrected alpha mode premultiplies EACH SOURCE SAMPLE before
            # interpolation. Multiplying sampled color by sampled alpha is wrong.
            a00, a01 = a00 * (1.0 / 255.0), a01 * (1.0 / 255.0)
            a10, a11 = a10 * (1.0 / 255.0), a11 * (1.0 / 255.0)
            alpha = (((a00 * w00 + a01 * w01) + a10 * w10) + a11 * w11) * opacity
        dst = batch * D_B + (local_y + Y_START) * D_Y + (local_x + X_START) * D_X
        for channel in tl.static_range(3):
            c00, c01, c10, c11 = _four_samples(
                Source,
                batch * S_B + channel * S_C,
                y0,
                x0,
                y1,
                x1,
                S_Y,
                S_X,
                SRC_H,
                SRC_W,
                valid,
            )
            if LEGACY:
                color = (((c00 * w00 + c01 * w01) + c10 * w10) + c11 * w11) * alpha
            else:
                c00, c01 = c00 * a00, c01 * a01
                c10, c11 = c10 * a10, c11 * a11
                color = (((c00 * w00 + c01 * w01) + c10 * w10) + c11 * w11) * opacity
            tl.store(Dest + dst + channel * D_C, color, active)
        # Write even alpha=0 pixels: preserve later-slot overwrite semantics.
        tl.store(Dest + dst + 3 * D_C, alpha, active)

    @triton.jit(
        do_not_specialize=[
            "N",
            "H",
            "W",
            "C_B",
            "C_C",
            "C_Y",
            "C_X",
            "F_B",
            "F_C",
            "F_Y",
            "F_X",
            "M_B",
            "M_C",
            "M_Y",
            "M_X",
        ]
    )
    def _blend_u8(
        Canvas,
        Frames,
        Mask,
        Out,
        N,
        H,
        W,
        C_B,
        C_C,
        C_Y,
        C_X,
        F_B,
        F_C,
        F_Y,
        F_X,
        M_B,
        M_C,
        M_Y,
        M_X,
        LEGACY: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = idx < N
        channel = idx % 3
        pixel = idx // 3
        x = pixel % W
        y = (pixel // W) % H
        batch = pixel // (H * W)
        cbase = batch * C_B + y * C_Y + x * C_X
        color = tl.load(Canvas + cbase + channel * C_C, active, 0)
        alpha = tl.load(Canvas + cbase + 3 * C_C, active, 0)
        background = tl.load(Frames + batch * F_B + channel * F_C + y * F_Y + x * F_X, active, 0)
        mask = tl.load(Mask + batch * M_B + channel * M_C + y * M_Y + x * M_X, active, 0)
        if LEGACY:
            color = color * alpha
        blended = color + background * (1.0 - alpha)
        blended = background * mask + blended * (1.0 - mask)
        # Inputs are finite image/mask data; NaNs are made zero for u8 export.
        blended = tl.where(blended == blended, blended, 0.0)
        result = tl.minimum(tl.maximum(blended, 0.0), 255.0).to(tl.uint8)
        tl.store(Out + idx, result, active)

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
    def _blur_blend_u8(
        Canvas,
        Frames,
        Kernel,
        Mask,
        Out,
        H: tl.constexpr,
        W: tl.constexpr,
        C_B: tl.constexpr,
        C_C: tl.constexpr,
        C_Y: tl.constexpr,
        C_X: tl.constexpr,
        F_B: tl.constexpr,
        F_C: tl.constexpr,
        F_Y: tl.constexpr,
        F_X: tl.constexpr,
        K_B: tl.constexpr,
        K_C: tl.constexpr,
        K_Y: tl.constexpr,
        K_X: tl.constexpr,
        M_B: tl.constexpr,
        M_C: tl.constexpr,
        M_Y: tl.constexpr,
        M_X: tl.constexpr,
        KH: tl.constexpr,
        KW: tl.constexpr,
        LEGACY: tl.constexpr,
        BLOCK_X: tl.constexpr,
        BLOCK_Y: tl.constexpr,
    ):
        # One program owns a small tile of adjacent output rows; x is the
        # contiguous lane dimension. Four accumulators avoid convolving alpha
        # separately for each RGB output, and omit the full float RGBA output.
        x = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)[None, :]
        y = tl.program_id(1) * BLOCK_Y + tl.arange(0, BLOCK_Y)[:, None]
        batch = tl.program_id(2)
        active = (x < W) & (y < H)
        acc_r = tl.full((BLOCK_Y, BLOCK_X), 0.0, tl.float32)
        acc_g = tl.full((BLOCK_Y, BLOCK_X), 0.0, tl.float32)
        acc_b = tl.full((BLOCK_Y, BLOCK_X), 0.0, tl.float32)
        acc_a = tl.full((BLOCK_Y, BLOCK_X), 0.0, tl.float32)

        # Loop over rows, unroll only columns: avoid a fully unrolled 225-tap
        # kernel while retaining fixed offsets within each source row.
        for ky in range(KH):
            source_y = y + ky - KH // 2
            row_valid = (source_y >= 0) & (source_y < H)
            for kx in tl.static_range(KW):
                source_x = x + kx - KW // 2
                valid = active & row_valid & (source_x >= 0) & (source_x < W)
                source_offset = batch * C_B + source_y * C_Y + source_x * C_X
                weight_offset = batch * K_B + ky * K_Y + kx * K_X
                # These are live pointer loads, not constexpr/cached weights.
                wr = tl.load(Kernel + weight_offset)
                wg = tl.load(Kernel + weight_offset + K_C)
                wb = tl.load(Kernel + weight_offset + 2 * K_C)
                wa = tl.load(Kernel + weight_offset + 3 * K_C)
                vr = tl.load(Canvas + source_offset, valid, 0.0)
                vg = tl.load(Canvas + source_offset + C_C, valid, 0.0)
                vb = tl.load(Canvas + source_offset + 2 * C_C, valid, 0.0)
                va = tl.load(Canvas + source_offset + 3 * C_C, valid, 0.0)
                # Explicit float32 FMA only for convolution accumulation.
                # Final alpha/mask operations retain their separate rounding
                # with enable_fp_fusion=False at launch.
                acc_r = tl.fma(vr, wr, acc_r)
                acc_g = tl.fma(vg, wg, acc_g)
                acc_b = tl.fma(vb, wb, acc_b)
                acc_a = tl.fma(va, wa, acc_a)

        frame_base = Frames + batch * F_B
        mask_base = Mask + batch * M_B
        bg_r = _load_spatial_plane(frame_base, x, y, H, W, F_Y, F_X)
        bg_g = _load_spatial_plane(frame_base + F_C, x, y, H, W, F_Y, F_X)
        bg_b = _load_spatial_plane(frame_base + 2 * F_C, x, y, H, W, F_Y, F_X)
        mask_r = _load_spatial_plane(mask_base, x, y, H, W, M_Y, M_X)
        mask_g = _load_spatial_plane(mask_base + M_C, x, y, H, W, M_Y, M_X)
        mask_b = _load_spatial_plane(mask_base + 2 * M_C, x, y, H, W, M_Y, M_X)
        if LEGACY:
            acc_r = acc_r * acc_a
            acc_g = acc_g * acc_a
            acc_b = acc_b * acc_a
        inverse_alpha = 1.0 - acc_a
        red = acc_r + bg_r * inverse_alpha
        green = acc_g + bg_g * inverse_alpha
        blue = acc_b + bg_b * inverse_alpha
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

    @triton.jit(
        do_not_specialize=[
            "N",
            "ROI_H",
            "ROI_W",
            "X_START",
            "Y_START",
            "SRC_H",
            "SRC_W",
            "S_B",
            "S_R",
            "S_C",
            "EPS",
        ]
    )
    def _homography_grid(
        Matrix,
        Out,
        N,
        ROI_H,
        ROI_W,
        X_START,
        Y_START,
        SRC_H,
        SRC_W,
        S_B,
        S_R,
        S_C,
        EPS,
        CANVAS_XY: tl.constexpr,
        ALIGN: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = idx < N
        x = (idx % ROI_W).to(tl.float32)
        y = ((idx // ROI_W) % ROI_H).to(tl.float32)
        batch = idx // (ROI_H * ROI_W)
        if CANVAS_XY:
            x, y = x + X_START, y + Y_START
        base = batch * S_B
        p0 = (
            tl.load(Matrix + base, active, 0) * x
            + tl.load(Matrix + base + S_C, active, 0) * y
            + tl.load(Matrix + base + 2 * S_C, active, 0)
        )
        p1 = (
            tl.load(Matrix + base + S_R, active, 0) * x
            + tl.load(Matrix + base + S_R + S_C, active, 0) * y
            + tl.load(Matrix + base + S_R + 2 * S_C, active, 0)
        )
        p2 = (
            tl.load(Matrix + base + 2 * S_R, active, 0) * x
            + tl.load(Matrix + base + 2 * S_R + S_C, active, 0) * y
            + tl.load(Matrix + base + 2 * S_R + 2 * S_C, active, 0)
        )
        valid = (
            (tl.abs(p2) > EPS)
            & (tl.abs(p0) < float("inf"))
            & (tl.abs(p1) < float("inf"))
            & (tl.abs(p2) < float("inf"))
        )
        safe_z = tl.where(valid, p2, 1.0)
        sx, sy = tl.div_rn(p0, safe_z), tl.div_rn(p1, safe_z)
        if ALIGN:
            gx = tl.div_rn(2 * sx, SRC_W - 1.0) - 1.0
            gy = tl.div_rn(2 * sy, SRC_H - 1.0) - 1.0
        else:
            gx = tl.div_rn(2 * sx + 1.0, SRC_W * 1.0) - 1.0
            gy = tl.div_rn(2 * sy + 1.0, SRC_H * 1.0) - 1.0
        valid = valid & (tl.abs(gx) < float("inf")) & (tl.abs(gy) < float("inf"))
        tl.store(Out + idx * 2, tl.where(valid, gx, 1e6), active)
        tl.store(Out + idx * 2 + 1, tl.where(valid, gy, 1e6), active)


def warp_paste_into(
    source,
    grid,
    opacity,
    canvas,
    bbox,
    *,
    align_corners=False,
    padding_mode="zeros",
    legacy_alpha=True,
):
    """Sample one advertisement and overwrite its ROI in a shared RGBA canvas.

    source is [B,4,src_h,src_w], grid is normalized [1|B,roi_h,roi_w,2],
    and opacity broadcasts to [B,1,roi_h,roi_w]. bbox uses canvas coordinates
    with exclusive right/bottom edges. Strides are passed through unchanged.
    The caller launches slots in order on one stream: even transparent later
    pixels overwrite earlier slots. Only canvas is modified.
    """
    if not supported(source, grid, opacity, canvas):
        raise ValueError("warp_paste_into requires inference float32 CUDA tensors and Triton")
    if padding_mode not in ("zeros", "border"):
        raise ValueError("Triton warp supports zeros/border; use PyTorch for reflection")
    x1, y1, x2, y2 = bbox
    roi_h, roi_w = y2 - y1, x2 - x1
    batch = source.shape[0]
    grid = grid.expand(batch, roi_h, roi_w, 2)
    opacity = opacity.expand(batch, 1, roi_h, roi_w)
    source_strides = source.stride()  # B, C, Y, X
    grid_strides = grid.stride()  # B, Y, X, coordinate
    opacity_batch_stride, _, opacity_row_stride, opacity_column_stride = opacity.stride()
    canvas_strides = canvas.stride()  # B, C, Y, X
    count = batch * roi_h * roi_w
    with torch.cuda.device(source.device):
        _warp_paste[(triton.cdiv(count, 256),)](
            source,
            grid,
            opacity,
            canvas,
            count,
            roi_h,
            roi_w,
            x1,
            y1,
            source.shape[-2],
            source.shape[-1],
            *source_strides,
            *grid_strides,
            opacity_batch_stride,
            opacity_row_stride,
            opacity_column_stride,
            *canvas_strides,
            ALIGN=align_corners,
            BORDER=padding_mode == "border",
            LEGACY=legacy_alpha,
            BLOCK=256,
            num_warps=4,
            enable_fp_fusion=False,
        )


def warp_bucket_patches(
    source,
    grids,
    opacity,
    *,
    align_corners=False,
    padding_mode="zeros",
    legacy_alpha=True,
):
    """Sample an entire bucket with exactly the batch path's FP32 arithmetic.

    source is [B,M,4,src_h,src_w], grids is [B,M,L,2], and opacity
    broadcasts to [B,M,1,L]. Fold B and M into the existing warp kernel's
    batch axis and treat each flattened ROI as a 1-by-L image. One launch
    produces [B,M,4,L] premultiplied patches, including transparent padding.
    No per-slot launch or conversion through PyTorch grid_sample occurs.
    """
    batch, members, channels, src_h, src_w = source.shape
    length = grids.shape[2]
    if channels != 4 or grids.shape != (batch, members, length, 2):
        raise ValueError("bucket source/grids must be [B,M,4,h,w] and [B,M,L,2]")
    flattened_source = source.reshape(batch * members, 4, src_h, src_w)
    flattened_grids = grids.reshape(batch * members, 1, length, 2)
    flattened_opacity = opacity.expand(batch, members, 1, length).reshape(
        batch * members, 1, 1, length
    )
    patches = source.new_empty((batch * members, 4, 1, length))
    warp_paste_into(
        flattened_source,
        flattened_grids,
        flattened_opacity,
        patches,
        (0, 0, length, 1),
        align_corners=align_corners,
        padding_mode=padding_mode,
        legacy_alpha=legacy_alpha,
    )
    return patches.reshape(batch, members, 4, length)


def blend_to_uint8(canvas, frames, mask, *, legacy_alpha=True):
    """Blend float32 [B,4,H,W] over frames and write independent uint8 BHWC.

    frames and mask broadcast to [B,3,H,W] through views, including zero
    strides. This route performs no blur and does not modify its inputs.
    """
    if not supported(canvas, frames, mask):
        raise ValueError("blend_to_uint8 requires inference float32 CUDA tensors and Triton")
    batch, _, height, width = canvas.shape
    frames = frames.expand(batch, 3, height, width)
    mask = mask.expand(batch, 3, height, width)
    output = torch.empty((batch, height, width, 3), device=canvas.device, dtype=torch.uint8)
    count = batch * height * width * 3
    with torch.cuda.device(canvas.device):
        _blend_u8[(triton.cdiv(count, 256),)](
            canvas,
            frames,
            mask,
            output,
            count,
            height,
            width,
            *canvas.stride(),
            *frames.stride(),
            *mask.stride(),
            LEGACY=legacy_alpha,
            BLOCK=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output


def blur_blend_to_uint8(canvas, frames, kernel, mask, *, legacy_alpha=True):
    """Optional one-kernel float32 depthwise blur + final uint8 BHWC output.

    canvas: [B,4,H,W], after distortion; no input tensor is modified.
    frames/mask broadcast to [B,3,H,W], including noncontiguous/zero strides.
    kernel: [4,1,kh,kw] or [B,4,1,kh,kw], odd kh/kw <=15; values are loaded
    afresh every invocation. Per-channel/per-batch weights may all differ.
    Padding is zero, stride/dilation=1; weights are not spatially flipped.
    Floating accumulation order can differ from cuDNN; target-GPU comparison
    and warm steady-state timing are required before enabling this path.
    """
    if not blur_blend_supported(canvas, frames, kernel, mask):
        raise ValueError(
            "blur_blend_to_uint8 requires float32 CUDA inference tensors, broadcastable frames/mask, and [4,1,kh,kw] or [B,4,1,kh,kw] with odd kh/kw <=15"
        )
    batch, _, height, width = canvas.shape
    frames = frames.expand(batch, 3, height, width)
    mask = mask.expand(batch, 3, height, width)
    if kernel.ndim == 4:
        # A shared [4,1,kh,kw] kernel has no batch axis: its batch stride is 0.
        kernel_strides = (0, kernel.stride(0), kernel.stride(2), kernel.stride(3))
    else:
        kernel_strides = (kernel.stride(0), kernel.stride(1), kernel.stride(3), kernel.stride(4))
    output = torch.empty((batch, height, width, 3), device=canvas.device, dtype=torch.uint8)
    block_x, block_y = 128, 4
    with torch.cuda.device(canvas.device):
        _blur_blend_u8[(triton.cdiv(width, block_x), triton.cdiv(height, block_y), batch)](
            canvas,
            frames,
            kernel,
            mask,
            output,
            H=height,
            W=width,
            C_B=canvas.stride(0),
            C_C=canvas.stride(1),
            C_Y=canvas.stride(2),
            C_X=canvas.stride(3),
            F_B=frames.stride(0),
            F_C=frames.stride(1),
            F_Y=frames.stride(2),
            F_X=frames.stride(3),
            K_B=kernel_strides[0],
            K_C=kernel_strides[1],
            K_Y=kernel_strides[2],
            K_X=kernel_strides[3],
            M_B=mask.stride(0),
            M_C=mask.stride(1),
            M_Y=mask.stride(2),
            M_X=mask.stride(3),
            KH=kernel.shape[-2],
            KW=kernel.shape[-1],
            LEGACY=legacy_alpha,
            BLOCK_X=block_x,
            BLOCK_Y=block_y,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output


def build_homography_grid(
    matrix, source_hw, bbox, *, matrix_uses_canvas_xy, align_corners=False, eps=1e-8
):
    """Build normalized [B,roi_h,roi_w,2] from validated [B,3,3] matrices.

    source_hw is the actual source image size. matrix_uses_canvas_xy chooses
    whether bbox's top-left offset is added before projection. This wrapper
    implements homography_roi_grid's valid-z/sentinel rule, not the shared
    builder's z+eps projection. Matrix values remain live GPU inputs.
    """
    if not supported(matrix):
        raise ValueError("build_homography_grid requires inference float32 CUDA tensor and Triton")
    x1, y1, x2, y2 = bbox
    roi_h, roi_w = y2 - y1, x2 - x1
    batch = matrix.shape[0]
    grid = torch.empty((batch, roi_h, roi_w, 2), device=matrix.device, dtype=matrix.dtype)
    count = batch * roi_h * roi_w
    with torch.cuda.device(matrix.device):
        _homography_grid[(triton.cdiv(count, 256),)](
            matrix,
            grid,
            count,
            roi_h,
            roi_w,
            x1,
            y1,
            source_hw[0],
            source_hw[1],
            *matrix.stride(),
            float(eps),
            CANVAS_XY=matrix_uses_canvas_xy,
            ALIGN=align_corners,
            BLOCK=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return grid
