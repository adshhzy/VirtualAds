"""Explicit per-frame preparation for shared camera and advertisement geometry.

Create one SharedParameterBuilder per worker/device and reuse it. Call prepare
once per frame, then pass the returned six-item paste_args to all material
groups that share that camera. There is deliberately NO frame_id result cache:
the same frame ID can be revisited with different live calibration values.

For multiple devices, call compute_camera_filter_shared once on the CPU and
pass its (coefficients, motion_rate) result as camera_result to each prepare.
The existing owner's motion_rate_to_sigma and create_gaussian_kernel callbacks
are used unchanged; this module does not invent a motion-to-blur relationship.

The bounded Gaussian cache assumes create_gaussian_kernel is pure for a given
sigma and callback configuration. Call clear_cache(), change
owner.gaussian_kernel_cache_version, or change builder.kernel_cache_version
when configuration changes. Disable it with cache_kernels=False. Sigma values
are never quantized. Camera, coefficient and homography VALUES are never cached.

Normalization 'legacy' reproduces 2*source/(size-1)-1 regardless of the sampling
align_corners flag. 'matched' changes normalization to match that flag. ROI
homographies use local ROI coordinates and divide by z+1e-8, matching the supplied
original implementation rather than the older adapter's valid-z sentinel rule.
Camera selection defaults to 'legacy' to keep performance comparisons neutral;
choose camera_index_policy='corrected' explicitly to fix the original row index.

This object is not thread safe. Use it serially on one CUDA stream, establishing
dependencies explicitly for inputs produced on other streams. Geometry creation
is inference-only. The torch builders below are CPU/GPU fallbacks and references;
the Triton builders are optional and never silently replace a requested backend.
"""

from __future__ import annotations

from collections import OrderedDict
from numbers import Integral, Real
import math

import numpy as np
import torch

if __package__:
    from .paste_camera_filter import compute_camera_filter_shared
else:
    from paste_camera_filter import compute_camera_filter_shared


def _size(value, name, *, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _hw(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be an (H, W) tuple/list")
    return _size(value[0], name + "[0]"), _size(value[1], name + "[1]")


def _aligned(normalization, align_corners, source_hw):
    if normalization not in ("legacy", "matched"):
        raise ValueError("normalization must be 'legacy' or 'matched'")
    if not isinstance(align_corners, bool):
        raise ValueError("align_corners must be a bool")
    aligned = normalization == "legacy" or align_corners
    if aligned and min(source_hw) < 2:
        raise ValueError("legacy or align_corners=True normalization requires source H,W >=2")
    return aligned


def _same_float32(*tensors):
    if not tensors or not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("grid builders require torch.Tensor inputs")
    device = tensors[0].device
    if any(t.dtype != torch.float32 or t.device != device for t in tensors):
        raise ValueError("grid builder inputs must be float32 on the same device")


def _normal_grid(source_x, source_y, source_hw, aligned):
    h, w = source_hw
    if aligned:
        gx, gy = 2.0*source_x/(w-1)-1.0, 2.0*source_y/(h-1)-1.0
    else:
        gx, gy = (2.0*source_x+1.0)/w-1.0, (2.0*source_y+1.0)/h-1.0
    return torch.stack((gx, gy), dim=-1)


def create_roi_grid_torch(source_hw, inv_matrix, roi_hw, *, align_corners=False,
                          normalization="legacy", eps=1e-8):
    """Return [1|B,roi_H,roi_W,2], using local xy and denominator z+eps.

    Matrix values are read live. No .item(), .tolist(), CPU transfer or finite-z
    clamp is performed. Float32 rounding follows this torch implementation.
    """
    source_hw, roi_hw = _hw(source_hw, "source_hw"), _hw(roi_hw, "roi_hw")
    aligned = _aligned(normalization, align_corners, source_hw)
    if isinstance(eps, bool) or not isinstance(eps, Real) or not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be a finite nonnegative Python real number")
    _same_float32(inv_matrix)
    matrix = inv_matrix.unsqueeze(0) if inv_matrix.ndim == 2 else inv_matrix
    if matrix.ndim != 3 or matrix.shape[-2:] != (3, 3) or matrix.shape[0] < 1:
        raise ValueError("inv_matrix must be [3,3] or [B,3,3] with B>=1")
    rh, rw = roi_hw
    yy, xx = torch.meshgrid(torch.arange(rh, device=matrix.device, dtype=torch.float32),
                           torch.arange(rw, device=matrix.device, dtype=torch.float32),
                           indexing="ij")
    coordinates = torch.stack((xx, yy, torch.ones_like(xx)), dim=0).reshape(1, 3, -1)
    projected = matrix @ coordinates
    xy = projected[:, :2] / (projected[:, 2:3] + float(eps))
    return _normal_grid(xy[:, 0], xy[:, 1], source_hw, aligned).reshape(
        matrix.shape[0], rh, rw, 2)


def _distortion_terms(x, y, k1, k2, p1, p2, k3):
    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r4*r2
    radial = 1.0 + k1*r2 + k2*r4 + k3*r6
    tx = 2.0*p1*x*y + p2*(r2 + 2.0*x*x)
    ty = p1*(r2 + 2.0*y*y) + 2.0*p2*x*y
    return radial, tx, ty


def create_distortion_grid_torch(H, W, K, dist_coeffs, *, align_corners=False,
                                 inverse=False, iterations=10, normalization="legacy"):
    """Return [1,H,W,2] with the supplied original distortion equations.

    Coefficient lengths 1 / 2..4 / >=5 mean k1 / k1,k2 / k1,k2,p1,p2,k3.
    inverse=True performs simultaneous fixed-point x/y updates. Coefficients
    and K remain live tensors; malformed calibration values are not repaired.
    """
    H, W = _size(H, "H"), _size(W, "W")
    aligned = _aligned(normalization, align_corners, (H, W))
    iterations = _size(iterations, "iterations", minimum=0)
    if not isinstance(inverse, bool):
        raise ValueError("inverse must be a bool")
    _same_float32(K, dist_coeffs)
    if K.shape != (3, 3) or dist_coeffs.ndim != 1 or dist_coeffs.numel() < 1:
        raise ValueError("K must be [3,3] and dist_coeffs a nonempty 1-D tensor")
    zero = dist_coeffs.new_zeros(())
    count = dist_coeffs.numel()
    k1 = dist_coeffs[0]
    k2 = dist_coeffs[1] if count >= 2 else zero
    p1, p2, k3 = (dist_coeffs[2], dist_coeffs[3], dist_coeffs[4]) if count >= 5 else (zero, zero, zero)
    fy, fx, cx, cy = K[1, 1], K[0, 0], K[0, 2], K[1, 2]
    yy, xx = torch.meshgrid(torch.arange(H, device=K.device, dtype=torch.float32),
                           torch.arange(W, device=K.device, dtype=torch.float32),
                           indexing="ij")
    xn, yn = (xx-cx)/fx, (yy-cy)/fy
    if inverse:
        x, y = xn, yn
        for _ in range(iterations):
            radial, tx, ty = _distortion_terms(x, y, k1, k2, p1, p2, k3)
            x, y = (xn-tx)/radial, (yn-ty)/radial
    else:
        radial, tx, ty = _distortion_terms(xn, yn, k1, k2, p1, p2, k3)
        x, y = xn*radial+tx, yn*radial+ty
    return _normal_grid(fx*x+cx, fy*y+cy, (H, W), aligned).unsqueeze(0)


def _float_tensor(value, device):
    if not isinstance(value, torch.Tensor) and hasattr(value, "strides"):
        if any(stride < 0 for stride in value.strides):
            value = value.copy()
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _host_numpy(value, name):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise TypeError(f"{name} must be CPU data; implicit device-to-host transfer is not allowed")
        value = value.detach().numpy()
    return np.asarray(value)


def _callback_identity(callback):
    return getattr(callback, "__self__", None), getattr(callback, "__func__", callback)


def _factor_kernel(kernel):
    """Validate a nonnegative rank-one kernel; return float32 factors or reason."""
    if not np.isfinite(kernel).all() or np.any(kernel < 0):
        return None, None, "kernel must be finite and nonnegative"
    h, w = kernel.shape
    work = kernel.astype(np.float64, copy=False)
    if not np.any(work):
        ky, kx = np.zeros(h, np.float32), np.ones(w, np.float32)
    else:
        row, column = np.unravel_index(np.argmax(work), work.shape)
        ky = work[:, column].astype(np.float32)
        kx = (work[row, :]/work[row, column]).astype(np.float32)
    reconstructed = ky[:, None]*kx[None, :]
    error = float(np.max(np.abs(reconstructed.astype(np.float64)-kernel)))
    if not np.allclose(reconstructed, kernel, rtol=1e-6, atol=1e-8):
        return None, error, "kernel is not rank one within rtol=1e-6, atol=1e-8"
    return (ky, kx), error, None


class SharedParameterBuilder:
    """Stateful Gaussian cache plus explicit, uncached per-frame geometry.

    enable_separable=True verifies the ACTUAL callback kernel is nonnegative
    and rank one before returning MotionKernelSpec(kernel, (ky,kx)). Failure
    raises with the residual; use enable_separable=False for a general 2-D
    callback kernel. The full kernel is retained for reference/dense execution.
    """

    def __init__(self, *, acceleration="triton", normalization="legacy",
                 distortion_inverse=True, iterations=10, camera_index_policy="legacy",
                 distortion_align_corners=False, enable_separable=False, cache_limit=16,
                 cache_kernels=True, kernel_cache_version=0):
        if acceleration not in ("eager", "auto", "triton"):
            raise ValueError("acceleration must be 'eager', 'auto' or 'triton'")
        if normalization not in ("legacy", "matched"):
            raise ValueError("normalization must be 'legacy' or 'matched'")
        if camera_index_policy not in ("legacy", "corrected"):
            raise ValueError("camera_index_policy must be 'legacy' or 'corrected'")
        for name, value in (("distortion_inverse", distortion_inverse),
                            ("distortion_align_corners", distortion_align_corners),
                            ("enable_separable", enable_separable), ("cache_kernels", cache_kernels)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool")
        self.acceleration, self.normalization = acceleration, normalization
        self.distortion_inverse = distortion_inverse
        self.iterations = _size(iterations, "iterations", minimum=0)
        self.camera_index_policy = camera_index_policy
        self.distortion_align_corners = distortion_align_corners
        self.enable_separable, self.cache_kernels = enable_separable, cache_kernels
        self.cache_limit = _size(cache_limit, "cache_limit")
        self.kernel_cache_version = kernel_cache_version
        self.clear_cache()

    def clear_cache(self):
        """Forget all cached Gaussian tensors/factors and reset cache statistics."""
        self._kernel_cache = OrderedDict()
        self._kernel_hits = self._kernel_misses = 0
        self.last_info = {}

    def cache_info(self):
        return {"entries": len(self._kernel_cache), "hits": self._kernel_hits,
                "misses": self._kernel_misses, "limit": self.cache_limit,
                "enabled": self.cache_kernels}

    def _grid_backend(self, *tensors):
        if self.acceleration == "eager":
            return None
        if __package__:
            from . import paste_shared_kernels
        else:
            import paste_shared_kernels
        if paste_shared_kernels.supported(*tensors):
            return paste_shared_kernels
        if self.acceleration == "triton":
            raise RuntimeError("SharedParameterBuilder acceleration='triton' requires Triton and float32 CUDA inference inputs; select 'eager' or 'auto' for a fallback")
        return None

    @staticmethod
    def _device(value):
        if isinstance(value, bool):
            raise ValueError("device must be a torch device, device string or CUDA integer index")
        device = torch.device("cuda", int(value)) if isinstance(value, Integral) else torch.device(value)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return device

    def _motion_value(self, owner, motion_rate, device):
        sigma_callback = getattr(owner, "motion_rate_to_sigma", None)
        kernel_callback = getattr(owner, "create_gaussian_kernel", None)
        if not callable(sigma_callback) or not callable(kernel_callback):
            raise TypeError("owner.motion_rate_to_sigma and owner.create_gaussian_kernel must be callable")
        callback_sigma = sigma_callback(motion_rate)
        sigma = callback_sigma
        if isinstance(sigma, torch.Tensor):
            if sigma.device.type != "cpu" or sigma.numel() != 1:
                raise TypeError("motion_rate_to_sigma must return a CPU scalar; no implicit GPU synchronization")
            sigma = sigma.detach().numpy().reshape(()).item()
        if isinstance(sigma, (bool, np.bool_)) or not isinstance(sigma, Real) or not math.isfinite(sigma):
            raise ValueError("motion_rate_to_sigma must return a finite real scalar sigma")
        # Keep the actual scalar/type in the key: no quantization, including
        # callbacks whose float32/float64 argument types affect their results.
        sigma_key = float(sigma)
        sigma_type = (type(callback_sigma), str(getattr(callback_sigma, "dtype", "")),
                      tuple(callback_sigma.shape) if isinstance(callback_sigma, torch.Tensor) else None)
        zero_sign = math.copysign(1.0, sigma) if sigma == 0 else None
        owner_version = getattr(owner, "gaussian_kernel_cache_version", 0)
        for version in (owner_version, self.kernel_cache_version):
            if isinstance(version, torch.Tensor):
                raise TypeError("Gaussian cache versions must be hashable Python values, not tensors")
            hash(version)
        sigma_identity, kernel_identity = _callback_identity(sigma_callback), _callback_identity(kernel_callback)
        key = (id(owner), tuple(map(id, sigma_identity)), tuple(map(id, kernel_identity)),
               sigma_type, type(sigma), sigma, zero_sign, str(device),
               owner_version, self.kernel_cache_version, self.enable_separable)
        cached = self._kernel_cache.get(key) if self.cache_kernels else None
        if cached is not None:
            self._kernel_cache.move_to_end(key)
            self._kernel_hits += 1
            value, info, _ = cached
            self.last_info.update(info, kernel_cache_hit=True)
            return value
        self._kernel_misses += 1
        # Preserve the callback's original sigma object/type; scalar conversion
        # above is for hashing/diagnostics only, never a new blur law.
        original = _host_numpy(kernel_callback(callback_sigma), "create_gaussian_kernel result")
        if original.ndim != 2 or min(original.shape) < 1 or any(size % 2 == 0 for size in original.shape):
            raise ValueError("create_gaussian_kernel must return a nonempty 2-D kernel with odd H,W")
        if original.dtype.kind != "f" or original.dtype.itemsize not in (4, 8):
            raise TypeError("create_gaussian_kernel must return float32 or float64 CPU weights")
        kernel_np = np.ascontiguousarray(original, dtype=np.float32)
        if not np.isfinite(kernel_np).all():
            raise ValueError("Gaussian kernel must remain finite after float32 conversion")
        factor_np, factor_error, reason = _factor_kernel(kernel_np) if self.enable_separable else (None, None, "disabled")
        info = dict(sigma=sigma_key, kernel_shape=tuple(kernel_np.shape),
                    separable=factor_np is not None, factor_error=factor_error,
                    separable_reason=reason)
        self.last_info.update(info, kernel_cache_hit=False)
        if self.enable_separable and factor_np is None:
            raise ValueError(f"enable_separable=True: {reason}; max absolute factor error={factor_error!r}. Use enable_separable=False with a dense blur backend for this callback kernel.")
        kh, kw = kernel_np.shape
        if factor_np is None:
            kernel = torch.from_numpy(kernel_np).to(device=device).reshape(1, 1, kh, kw).repeat(4, 1, 1, 1)
            value = kernel
        else:
            ky, kx = factor_np
            # One host-to-device transfer for dense reference weights + factors.
            packed = torch.from_numpy(np.concatenate((kernel_np.ravel(), ky, kx))).to(device=device)
            kernel = packed[:kh*kw].reshape(1, 1, kh, kw).repeat(4, 1, 1, 1)
            factors = (packed[kh*kw:kh*kw+kh], packed[kh*kw+kh:])
            if __package__:
                from .paste_shared_types import MotionKernelSpec
            else:
                from paste_shared_types import MotionKernelSpec
            value = MotionKernelSpec(kernel=kernel, factors=factors)
        if self.cache_kernels:
            # Hold identity objects so Python cannot recycle their IDs into a hit.
            self._kernel_cache[key] = (value, info, (owner, *sigma_identity, *kernel_identity))
            while len(self._kernel_cache) > self.cache_limit:
                self._kernel_cache.popitem(last=False)
        return value

    @torch.inference_mode()
    def prepare(self, owner, data_dict, frame_id, *, height, width, device,
                mask_canvas, camera_result=None):
        """Build a six-item original paste_args tuple, explicitly once per frame.

        camera_result=None calls compute_camera_filter_shared exactly once.
        Otherwise supply (coefficients, motion_rate) from an earlier CPU call;
        this method will not filter again or rewrite data_dict coefficients.
        Returned calibration tensors are float32 on device. Neither camera
        results nor distortion grids are reused based on frame_id.
        """
        height, width = _size(height, "height"), _size(width, "width")
        device = self._device(device)
        self.last_info = {"camera_filter_called": camera_result is None,
                          "camera_index_policy": self.camera_index_policy,
                          "normalization": self.normalization,
                          "requested_acceleration": self.acceleration}
        if camera_result is None:
            filtered, motion_rate = compute_camera_filter_shared(
                owner, data_dict, frame_id, index_policy=self.camera_index_policy)
        else:
            if not isinstance(camera_result, (tuple, list)) or len(camera_result) != 2:
                raise ValueError("camera_result must be (dist_coeffs_filtered, motion_rate)")
            filtered, motion_rate = camera_result
        K = _float_tensor(data_dict["K"], device)
        coefficients = _float_tensor(filtered, device)
        backend = self._grid_backend(K, coefficients)
        grid_fn = create_distortion_grid_torch if backend is None else backend.build_distortion_grid_gpu
        distortion = grid_fn(height, width, K, coefficients,
                            align_corners=self.distortion_align_corners,
                            inverse=self.distortion_inverse, iterations=self.iterations,
                            normalization=self.normalization)
        self.last_info["effective_grid_backend"] = "torch" if backend is None else "triton"
        motion_value = self._motion_value(owner, motion_rate, device)
        return device, distortion, coefficients, motion_value, mask_canvas, None

    @torch.inference_mode()
    def build_warp_grids(self, *, params, ads, boxes, device, batch,
                         warp_align_corners, **kwargs):
        """Bulk adapter hook: return one current ROI grid per advertisement slot.

        Source H,W come from actual collected ads, never ad_h_w. All-CPU [3,3]
        matrices are np.stack'ed into ONE transfer, then sliced on the device.
        GPU matrices are used live; no geometry values or input IDs are cached.
        Repeated slot IDs remain distinct because the result is an ordered tuple.
        """
        if not len(params) == len(ads) == len(boxes):
            raise ValueError("params, ads and boxes must have equal lengths")
        batch = _size(batch, "batch")
        device = self._device(device)
        if kwargs.get("matrix_coordinates") not in (None, "roi"):
            raise ValueError("SharedParameterBuilder uses original ROI-local m_inv; matrix_coordinates must be 'roi'")
        if not isinstance(warp_align_corners, bool):
            raise ValueError("warp_align_corners must be a bool")
        if not params:
            return ()
        raw_matrices = [param["m_inv"] for param in params]
        all_host = all(not isinstance(value, torch.Tensor) or value.device.type == "cpu"
                       for value in raw_matrices)
        host_matrices = [_host_numpy(value, f"m_inv[{i}]") for i, value in enumerate(raw_matrices)] if all_host else None
        if all_host and all(matrix.shape == (3, 3) for matrix in host_matrices):
            packed = _float_tensor(np.stack(host_matrices).astype(np.float32, copy=False), device)
            matrices = tuple(packed[i] for i in range(len(params)))
            self.last_info["warp_host_matrix_batches"] = 1
        else:
            matrices = tuple(_float_tensor(value, device) for value in raw_matrices)
            self.last_info["warp_host_matrix_batches"] = len(raw_matrices) if all_host else None
        grids = []
        for i, (ad, box, matrix) in enumerate(zip(ads, boxes, matrices)):
            if not isinstance(ad, torch.Tensor) or ad.ndim != 4 or ad.shape[:2] != (batch, 4):
                raise ValueError(f"ads[{i}] must be [B,4,H,W] with B={batch}")
            source_hw = tuple(ad.shape[-2:])
            if matrix.ndim not in (2, 3) or matrix.shape[-2:] != (3, 3):
                raise ValueError(f"m_inv[{i}] must be [3,3] or [1|B,3,3]")
            if matrix.ndim == 3 and matrix.shape[0] not in (1, batch):
                raise ValueError(f"m_inv[{i}] batch must be 1 or {batch}")
            if not isinstance(box, (tuple, list)) or len(box) != 4 or any(
                isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in box):
                raise ValueError(f"boxes[{i}] must contain four integer coordinates")
            x1, y1, x2, y2 = map(int, box)
            roi_hw = _hw((y2-y1, x2-x1), f"ROI[{i}]")
            backend = self._grid_backend(matrix)
            grid_fn = create_roi_grid_torch if backend is None else backend.build_roi_grid_gpu
            grids.append(grid_fn(source_hw, matrix, roi_hw,
                                 align_corners=warp_align_corners,
                                 normalization=self.normalization, eps=1e-8))
        self.last_info["warp_grid_slots"] = len(grids)
        return tuple(grids)


__all__ = ["SharedParameterBuilder", "create_distortion_grid_torch", "create_roi_grid_torch"]
