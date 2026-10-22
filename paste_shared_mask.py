"""One bounded workspace for unioning ROI masks and feathering the full frame.

Mask values are protection weights: 1 keeps the original background, 0 permits
the rendered advertisement. Inputs must be finite real values in [0, 1]; no
implicit conversion from a 0..255 convention is performed. Visible CPU values
are checked; CUDA values are a caller contract, avoiding readback/synchronization.

The returned HW tensor is borrowed workspace and may change on the next build.
Consume it on the same compute stream before another build, or clone it when
independent ownership is required. Use one builder per serial GPU worker.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Integral, Real

import numpy as np
import torch
import torch.nn.functional as F


def _integer(value, name, *, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _device(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("device must identify a CPU or CUDA device")
    result = (
        torch.device("cuda", int(value)) if isinstance(value, Integral) else torch.device(value)
    )
    if result.type == "cpu":
        return torch.device("cpu")
    if result.type != "cuda":
        raise ValueError("device must be CPU or CUDA")
    if result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    return result


def _mask_kernels():
    if __package__:
        from . import paste_mask_kernels
    else:
        import paste_mask_kernels
    return paste_mask_kernels


class SharedMaskBuilder:
    """Union masks once, then apply a normalized separable Gaussian to the canvas.

    Each input is {'bbox': (x1,y1,x2,y2), 'mask': HW ndarray/Tensor}, with
    exclusive right/bottom coordinates and mask shape (y2-y1, x2-x1). A bbox
    may extend outside the frame; only its visible intersection is copied.
    Masks are combined by maximum before feathering, never resized or blurred
    independently. Gaussian boundaries replicate the nearest frame pixel.

    sigma/radius are immutable configuration; construct another builder to change
    them. radius=0 disables feathering. Default radius=3 gives a 7x7 footprint.
    Device weights are reused across resolution changes. Only the latest device
    and resolution workspace is retained; no per-frame or per-resolution history.
    Eager tap accumulation is a reference path and allocates temporary padding
    and equality masks. The Triton path writes both passes into persistent workspace.
    """

    def __init__(self, sigma=1.5, radius=3, acceleration="auto"):
        if isinstance(sigma, (bool, np.bool_)) or not isinstance(sigma, Real):
            raise ValueError("sigma must be a finite positive real number")
        sigma = float(sigma)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be a finite positive real number")
        radius = _integer(radius, "radius", minimum=0)
        if acceleration not in ("auto", "triton", "eager"):
            raise ValueError("acceleration must be auto, triton or eager")
        self._sigma, self._radius, self._acceleration = sigma, radius, acceleration
        coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
        weights = np.exp(-0.5 * (coordinates / sigma) ** 2)
        self._host_weights = torch.from_numpy((weights / weights.sum()).astype(np.float32))
        self._weight_values = tuple(float(value) for value in self._host_weights.numpy())
        weight_sum = np.float32(0)
        for value in self._weight_values:
            weight_sum = np.float32(weight_sum + np.float32(value))
        self._weight_sum = float(weight_sum)
        self.release_resources()

    @property
    def sigma(self):
        return self._sigma

    @property
    def radius(self):
        return self._radius

    @property
    def acceleration(self):
        return self._acceleration

    def release_resources(self):
        """Drop workspace/device weights; retain only the small immutable CPU kernel.

        Call between builds, after its last consumer has finished. Caller-held
        borrowed tensors still own their old storage; release cannot revoke them.
        No CUDA synchronization, empty_cache or garbage collection is performed.
        """
        self._device = self._shape = None
        self._canvas = self._temporary = self._output = self._patch = self._weights = None
        self._builds = self._workspace_allocations = self._kernel_uploads = 0
        self._last_backend = None

    def cache_info(self):
        """Retained tensor bytes, excluding inputs, allocator reserves and eager temporaries."""
        workspace = (self._canvas, self._temporary, self._output, self._patch)
        return {
            "device": None if self._device is None else str(self._device),
            "shape": self._shape,
            "workspace_bytes": sum(
                value.numel() * value.element_size() for value in workspace if value is not None
            ),
            "patch_capacity_bytes": 0
            if self._patch is None
            else self._patch.numel() * self._patch.element_size(),
            "kernel_bytes": 0
            if self._weights is None
            else self._weights.numel() * self._weights.element_size(),
            "host_kernel_bytes": self._host_weights.numel() * self._host_weights.element_size(),
            "builds": self._builds,
            "workspace_allocations": self._workspace_allocations,
            "kernel_uploads": self._kernel_uploads,
            "effective_backend": self._last_backend,
        }

    def _describe_masks(self, items, height, width, device):
        if not isinstance(items, (list, tuple)):
            raise TypeError("mask_coor_list must be a list/tuple")
        visible = []
        for index, item in enumerate(items):
            name = f"mask_coor_list[{index}]"
            if not isinstance(item, Mapping):
                raise TypeError(f"{name} must be a mapping with bbox and mask")
            box, mask = item["bbox"], item["mask"]
            if not isinstance(box, (list, tuple, np.ndarray)) or len(box) != 4:
                raise ValueError(f"{name}.bbox must contain four integer coordinates")
            x1, y1, x2, y2 = (_integer(value, f"{name}.bbox") for value in box)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"{name}.bbox must have positive width and height")
            if not isinstance(mask, (np.ndarray, torch.Tensor)):
                raise TypeError(f"{name}.mask must be an HW ndarray or Tensor")
            if mask.ndim != 2 or tuple(mask.shape) != (y2 - y1, x2 - x1):
                raise ValueError(f"{name}.mask shape must match bbox height and width")
            if isinstance(mask, np.ndarray):
                if mask.dtype.kind not in "buif":
                    raise TypeError(f"{name}.mask must have a real numeric or boolean dtype")
            elif mask.is_complex() or mask.is_quantized or mask.layout != torch.strided:
                raise TypeError(f"{name}.mask must be a strided real tensor")
            elif mask.device.type != "cpu" and mask.device != device:
                raise ValueError(f"{name}.mask must be on CPU or on the target device")
            left, top, right, bottom = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if left >= right or top >= bottom:
                continue
            patch = mask[top - y1 : bottom - y1, left - x1 : right - x1]
            if isinstance(patch, np.ndarray):
                if not np.isfinite(patch).all() or np.any(patch < 0) or np.any(patch > 1):
                    raise ValueError(f"{name}.mask visible values must be finite and in [0, 1]")
            elif patch.device.type == "cpu":
                if (
                    not bool(torch.isfinite(patch).all())
                    or bool((patch < 0).any())
                    or bool((patch > 1).any())
                ):
                    raise ValueError(f"{name}.mask visible values must be finite and in [0, 1]")
            visible.append(((left, top, right, bottom), patch))
        return visible

    def _ensure_workspace(self, height, width, device):
        changed_device = device != self._device
        weights = self._weights
        if changed_device or weights is None:
            weights = self._host_weights.to(device=device)
        if changed_device or self._shape != (height, width):
            canvas = torch.empty((height, width), dtype=torch.float32, device=device)
            temporary = torch.empty_like(canvas) if self.radius else None
            output = torch.empty_like(canvas)
            patch = torch.empty(height * width, dtype=torch.float32, device=device)
            self._canvas, self._temporary, self._output, self._patch = (
                canvas,
                temporary,
                output,
                patch,
            )
            self._shape = (height, width)
            self._workspace_allocations += 1
        if changed_device or self._weights is None:
            self._weights = weights
            self._kernel_uploads += 1
        self._device = device

    def _blur_eager(self):
        """Accumulate the same ordered taps without an opaque convolution reduction."""
        radius = self.radius
        height, width = self._shape
        scratch = self._patch.view(height, width)
        for source, target, padding, horizontal in (
            (self._canvas, self._temporary, (radius, radius, 0, 0), True),
            (self._temporary, self._output, (0, 0, radius, radius), False),
        ):
            padded = F.pad(source[None, None], padding, mode="replicate")[0, 0]
            target.zero_()
            for offset, weight in enumerate(self._weight_values):
                samples = (
                    padded[:, offset : offset + width]
                    if horizontal
                    else padded[offset : offset + height]
                )
                torch.mul(samples, weight, out=scratch)
                target.add_(scratch)
            # Tensor/scalar division may use a reciprocal on some devices.
            # Preserve equal numerator/denominator exactly without host readback.
            ones = target == self._weight_sum
            target.div_(self._weight_sum).masked_fill_(ones, 1.0)
            del padded, samples, ones
        self._output.clamp_(0, 1)

    @torch.inference_mode()
    def build(self, mask_coor_list, *, height, width, device):
        """Return borrowed float32 HW protection weights, valid until the next build."""
        height = _integer(height, "height", minimum=1)
        width = _integer(width, "width", minimum=1)
        device = _device(device)
        if self.acceleration == "triton" and device.type != "cuda":
            raise RuntimeError("acceleration='triton' requires a CUDA device")
        patches = self._describe_masks(mask_coor_list, height, width, device)
        self._ensure_workspace(height, width, device)
        kernels = None
        if self.radius and self.acceleration != "eager" and device.type == "cuda":
            candidate = _mask_kernels()
            if candidate.supported(self._canvas, self._temporary, self._output, self._weights):
                kernels = candidate
            elif self.acceleration == "triton":
                raise RuntimeError("acceleration='triton' requires available Triton mask kernels")
        self._canvas.zero_()
        for (left, top, right, bottom), source in patches:
            if isinstance(source, np.ndarray):
                # Convert one visible patch at a time. Negative strides or a
                # read-only array need a bounded host copy; off-frame pixels
                # never enlarge this scratch. Common uint8/float32 inputs are
                # wrapped directly: copy_ converts into the existing scratch.
                source_array = source
                if not source_array.flags.writeable or any(
                    stride < 0 for stride in source_array.strides
                ):
                    source_array = source_array.copy(order="C")
                try:
                    source = torch.from_numpy(source_array)
                except (TypeError, ValueError):
                    source = torch.from_numpy(np.ascontiguousarray(source_array, dtype=np.float32))
                del source_array
            patch = self._patch[: (bottom - top) * (right - left)].view(bottom - top, right - left)
            patch.copy_(source)
            patch.clamp_(0, 1)
            region = self._canvas[top:bottom, left:right]
            torch.maximum(region, patch, out=region)
            del source
        if not self.radius:
            self._output.copy_(self._canvas)
            self._last_backend = "identity"
        elif kernels is None:
            self._blur_eager()
            self._last_backend = "eager"
        else:
            kernels.gaussian_mask_into(
                self._canvas, self._temporary, self._output, self._weights, radius=self.radius
            )
            self._last_backend = "triton"
        self._builds += 1
        return self._output
