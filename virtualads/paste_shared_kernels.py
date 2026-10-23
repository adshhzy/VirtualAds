"""Optional Triton builders for the caller's shared per-frame paste geometry.

These kernels reproduce the supplied original geometry formulas, including its
legacy normalized coordinates: 2*source/(size-1)-1 even when grid_sample uses
align_corners=False. Select normalization='matched' explicitly to use the
normalization corresponding to align_corners. This changes legacy sampling.

Camera/coefficients/matrix VALUES are live GPU inputs. No .item(), .tolist(),
CPU coefficient branches, geometry-value caches, or host synchronization occur.
Positions, sizes and strides are runtime arguments. ROI coordinates are local;
the perspective denominator is z+eps, not a valid-z clamp/sentinel. This is
deliberately different from paste_gpu_batch.homography_roi_grid.

Float32 CUDA inference only. Calibration must have nonzero fx/fy and meaningful
finite iterative states; singular/overflow behavior is not silently replaced.
Triton is optional at import time; callers choose their PyTorch fallback.
"""

import math
from numbers import Integral, Real

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
    """Metadata-only backend check; does not validate builder-specific shapes."""
    if triton is None or not tensors or not all(isinstance(t, torch.Tensor) for t in tensors):
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


def _size(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer dimension")
    return int(value)


def _hw(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be an (H, W) tuple/list")
    return _size(value[0], name + "[0]"), _size(value[1], name + "[1]")


def _aligned_normalization(normalization, align_corners, source_hw):
    """Choose the coordinate formula, keeping legacy independent of the sampler.

    True selects 2*x/(size-1)-1. False selects the half-pixel formula for
    align_corners=False. Legacy deliberately selects the former either way.
    """
    if normalization not in ("legacy", "matched"):
        raise ValueError("normalization must be 'legacy' or 'matched'")
    if not isinstance(align_corners, bool):
        raise ValueError("align_corners must be a bool")
    aligned = normalization == "legacy" or align_corners
    if aligned and min(source_hw) < 2:
        raise ValueError("legacy or align_corners=True normalization requires source H,W >=2")
    return aligned


if triton is not None:

    @triton.jit
    def _distortion_terms(x, y, k1, k2, p1, p2, k3):
        # Keep the original operation ordering, not a Horner rearrangement.
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        tx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        ty = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return radial, tx, ty

    @triton.jit(
        do_not_specialize=[
            "H",
            "W",
            "K_R",
            "K_C",
            "COEFF_STRIDE",
            "COEFF_COUNT",
            "ITERATIONS",
        ]
    )
    def _distortion_grid_kernel(
        K,
        Coefficients,
        Out,
        H,
        W,
        K_R,
        K_C,
        COEFF_STRIDE,
        COEFF_COUNT,
        ITERATIONS,
        INVERSE: tl.constexpr,
        ALIGNED_NORMALIZATION: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = index < H * W
        pixel_x = (index % W).to(tl.float32)
        pixel_y = (index // W).to(tl.float32)
        fx = tl.load(K)
        fy = tl.load(K + K_R + K_C)
        cx = tl.load(K + 2 * K_C)
        cy = tl.load(K + K_R + 2 * K_C)
        k1 = tl.load(Coefficients)
        k2 = tl.load(Coefficients + COEFF_STRIDE, COEFF_COUNT >= 2, 0.0)
        # Original behavior: lengths 2/3/4 use only k1,k2. At least five
        # coefficients are needed before tangential/k3 terms are present.
        p1 = tl.load(Coefficients + 2 * COEFF_STRIDE, COEFF_COUNT >= 5, 0.0)
        p2 = tl.load(Coefficients + 3 * COEFF_STRIDE, COEFF_COUNT >= 5, 0.0)
        k3 = tl.load(Coefficients + 4 * COEFF_STRIDE, COEFF_COUNT >= 5, 0.0)
        xn = tl.div_rn(pixel_x - cx, fx)
        yn = tl.div_rn(pixel_y - cy, fy)
        if INVERSE:
            x, y = xn, yn
            for _ in range(ITERATIONS):
                radial, tx, ty = _distortion_terms(x, y, k1, k2, p1, p2, k3)
                # Compute both updates from the same previous x/y. In
                # particular, ty cannot use the just-updated x value.
                next_x = tl.div_rn(xn - tx, radial)
                next_y = tl.div_rn(yn - ty, radial)
                x, y = next_x, next_y
        else:
            radial, tx, ty = _distortion_terms(xn, yn, k1, k2, p1, p2, k3)
            x = xn * radial + tx
            y = yn * radial + ty
        source_x = fx * x + cx
        source_y = fy * y + cy
        if ALIGNED_NORMALIZATION:
            gx = tl.div_rn(2.0 * source_x, W - 1.0) - 1.0
            gy = tl.div_rn(2.0 * source_y, H - 1.0) - 1.0
        else:
            gx = tl.div_rn(2.0 * source_x + 1.0, W * 1.0) - 1.0
            gy = tl.div_rn(2.0 * source_y + 1.0, H * 1.0) - 1.0
        tl.store(Out + index * 2, gx, active)
        tl.store(Out + index * 2 + 1, gy, active)

    @triton.jit(
        do_not_specialize=[
            "N",
            "ROI_H",
            "ROI_W",
            "SRC_H",
            "SRC_W",
            "M_B",
            "M_R",
            "M_C",
            "EPS",
        ]
    )
    def _roi_grid_kernel(
        Matrix,
        Out,
        N,
        ROI_H,
        ROI_W,
        SRC_H,
        SRC_W,
        M_B,
        M_R,
        M_C,
        EPS,
        ALIGNED_NORMALIZATION: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # Flatten [matrix_batch, ROI_H, ROI_W]. Projection starts at local
        # (0,0); the canvas bbox offset is intentionally absent from this API.
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = index < N
        x = (index % ROI_W).to(tl.float32)
        y = ((index // ROI_W) % ROI_H).to(tl.float32)
        batch = index // (ROI_H * ROI_W)
        base = batch * M_B
        sx = (
            tl.load(Matrix + base, active, 0.0) * x
            + tl.load(Matrix + base + M_C, active, 0.0) * y
            + tl.load(Matrix + base + 2 * M_C, active, 0.0)
        )
        sy = (
            tl.load(Matrix + base + M_R, active, 0.0) * x
            + tl.load(Matrix + base + M_R + M_C, active, 0.0) * y
            + tl.load(Matrix + base + M_R + 2 * M_C, active, 0.0)
        )
        z = (
            tl.load(Matrix + base + 2 * M_R, active, 0.0) * x
            + tl.load(Matrix + base + 2 * M_R + M_C, active, 0.0) * y
            + tl.load(Matrix + base + 2 * M_R + 2 * M_C, active, 0.0)
        )
        sx = tl.div_rn(sx, z + EPS)
        sy = tl.div_rn(sy, z + EPS)
        if ALIGNED_NORMALIZATION:
            gx = tl.div_rn(2.0 * sx, SRC_W - 1.0) - 1.0
            gy = tl.div_rn(2.0 * sy, SRC_H - 1.0) - 1.0
        else:
            gx = tl.div_rn(2.0 * sx + 1.0, SRC_W * 1.0) - 1.0
            gy = tl.div_rn(2.0 * sy + 1.0, SRC_H * 1.0) - 1.0
        tl.store(Out + index * 2, gx, active)
        tl.store(Out + index * 2 + 1, gy, active)


def build_distortion_grid_gpu(
    H,
    W,
    K,
    dist_coeffs,
    *,
    align_corners=False,
    inverse=False,
    iterations=10,
    normalization="legacy",
):
    """Return CUDA float32 [1,H,W,2] using the original camera formulas.

    K is [3,3], dist_coeffs is a nonempty 1D tensor, on the same CUDA device.
    Length 1 uses k1; lengths 2/3/4 use k1,k2; >=5 uses k1,k2,p1,p2,k3.
    Additional coefficients are ignored, exactly as in the supplied function.
    Tangential coefficients never trigger CPU reads or Python value branches.
    inverse=True performs simultaneous fixed-point updates, iterations times.

    This builds source-sampling coordinates only; it does not resample an
    image. The singleton batch grid can be shared by all material groups.
    Camera normalization, iteration, pixel projection and grid normalization
    run in one kernel with unchanged FP32 operation order.
    """
    H, W = _size(H, "H"), _size(W, "W")
    aligned = _aligned_normalization(normalization, align_corners, (H, W))
    if not isinstance(inverse, bool):
        raise ValueError("inverse must be a bool")
    if isinstance(iterations, bool) or not isinstance(iterations, Integral) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if not supported(K, dist_coeffs):
        raise ValueError(
            "build_distortion_grid_gpu requires Triton and float32 CUDA inference tensors on one device"
        )
    if K.shape != (3, 3) or dist_coeffs.ndim != 1 or dist_coeffs.numel() < 1:
        raise ValueError("K must be [3,3] and dist_coeffs must be nonempty 1D")
    grid = torch.empty((1, H, W, 2), device=K.device, dtype=torch.float32)
    with torch.cuda.device(K.device):
        _distortion_grid_kernel[(triton.cdiv(H * W, 256),)](
            K,
            dist_coeffs,
            grid,
            H,
            W,
            K.stride(0),
            K.stride(1),
            dist_coeffs.stride(0),
            dist_coeffs.numel(),
            int(iterations),
            INVERSE=inverse,
            ALIGNED_NORMALIZATION=aligned,
            BLOCK=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return grid


def build_roi_grid_gpu(
    source_hw, inv_matrix, roi_hw, *, align_corners=False, normalization="legacy", eps=1e-8
):
    """Return CUDA float32 [1|B,roi_h,roi_w,2] from ROI-local coordinates.

    inv_matrix is [3,3] or [B,3,3], mapping ROI-local xy to source pixels.
    Divide projected xy by projected z+eps, preserving the original formula;
    no absolute-z check, sign clamp, finite filtering, or sentinel is inserted.

    source_hw must describe the actual material size, whereas roi_hw is the
    destination rectangle size. They need not match. A single matrix yields
    a singleton-batch grid for later expansion without duplicating storage.
    """
    source_h, source_w = _hw(source_hw, "source_hw")
    roi_h, roi_w = _hw(roi_hw, "roi_hw")
    aligned = _aligned_normalization(normalization, align_corners, (source_h, source_w))
    if not isinstance(eps, Real) or isinstance(eps, bool) or not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be a finite nonnegative Python real number")
    if not supported(inv_matrix):
        raise ValueError("build_roi_grid_gpu requires Triton and a float32 CUDA inference tensor")
    matrix = inv_matrix.unsqueeze(0) if inv_matrix.ndim == 2 else inv_matrix
    if matrix.ndim != 3 or matrix.shape[-2:] != (3, 3) or matrix.shape[0] < 1:
        raise ValueError("inv_matrix must be [3,3] or [B,3,3] with B>=1")
    batch = matrix.shape[0]
    grid = torch.empty((batch, roi_h, roi_w, 2), device=matrix.device, dtype=torch.float32)
    count = batch * roi_h * roi_w
    with torch.cuda.device(matrix.device):
        _roi_grid_kernel[(triton.cdiv(count, 256),)](
            matrix,
            grid,
            count,
            roi_h,
            roi_w,
            source_h,
            source_w,
            matrix.stride(0),
            matrix.stride(1),
            matrix.stride(2),
            float(eps),
            ALIGNED_NORMALIZATION=aligned,
            BLOCK=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return grid
