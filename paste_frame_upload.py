"""Bounded staging for a serial worker's contiguous CPU uint8 background.

Upload the original bytes once, then convert into a reusable float32 device
workspace. Returned tensors are borrowed until the next prepare/release call;
they are internal render inputs, never caller-owned NumPy outputs. Other input
types and strides retain the executor's original conversion path and layout.
"""

from __future__ import annotations

import numpy as np
import torch

if __package__:
    from .paste_render_executor import _as_tensor
    from .paste_upload_buffer import PinnedUploadBuffer
else:
    from paste_render_executor import _as_tensor
    from paste_upload_buffer import PinnedUploadBuffer


class FrameUploadBuffer:
    """One current-device workspace, with bounded geometric capacity growth.

    The default limits retain up to 32 MiB each of pinned/GPU uint8 storage and
    128 MiB of float32 storage. They are budgets, not eager allocations. Larger
    inputs remain supported using temporary storage. Noncontiguous inputs use
    the old conversion, preserving its strides and therefore auto-policy gates.
    """

    def __init__(self, *, max_uint8_bytes=32 * 1024**2, max_float32_bytes=128 * 1024**2):
        if type(max_float32_bytes) is not int or max_float32_bytes < 0:
            raise ValueError("max_float32_bytes must be a nonnegative Python integer")
        self._upload = PinnedUploadBuffer(max_cached_bytes=max_uint8_bytes)
        self.max_float32_bytes = max_float32_bytes
        self._float_workspace = None
        self._float_capacity = 0
        self._float_allocations = 0
        self._oversize_allocations = 0
        self._last_temporary_bytes = 0

    @staticmethod
    def _contiguous_cpu_uint8(frame):
        if isinstance(frame, np.ndarray):
            return frame.dtype == np.uint8 and frame.flags.c_contiguous
        return (
            isinstance(frame, torch.Tensor)
            and frame.device.type == "cpu"
            and frame.dtype == torch.uint8
            and frame.layout == torch.strided
            and frame.is_contiguous()
        )

    def prepare(self, frame, device):
        """Return this frame's float32 values; inspect metadata, never cache pixels."""
        device = torch.device(device)
        if device.type != "cuda" or not self._contiguous_cpu_uint8(frame):
            # Drop stale upload context when the worker switches input mode.
            self.release_resources()
            return _as_tensor(frame, device)
        return self._prepare_uint8(frame, device)

    def _prepare_uint8(self, frame, device):
        """Keep CPU copying, byte upload and device conversion in that order."""
        host = self._upload.prepare(frame.shape, dtype=torch.uint8, device=device)
        if isinstance(frame, np.ndarray):
            # copyto supports read-only sources without making them writable or
            # creating a Tensor that aliases caller-owned NumPy storage.
            np.copyto(host.numpy(), frame, casting="no")
        else:
            host.copy_(frame, non_blocking=False)
        uploaded = self._upload.upload()
        output = self._float_output(uploaded)
        # All uint8 values are exactly representable in float32. Both operations
        # are ordered on the same worker stream; only uint8 crosses PCIe.
        output.copy_(uploaded)
        return output

    def _float_output(self, uploaded):
        elements = uploaded.numel()
        required_bytes = elements * 4
        workspace = self._float_workspace
        if workspace is not None and workspace.device != uploaded.device:
            self._float_workspace = workspace = None
            self._float_capacity = 0
        if required_bytes > self.max_float32_bytes:
            # Retain no historical large allocation. The current render call
            # owns this temporary until all of its layout groups are complete.
            self._float_workspace = None
            self._float_capacity = 0
            self._oversize_allocations += 1
            self._last_temporary_bytes = required_bytes
            with torch.inference_mode(False):
                return torch.empty_like(uploaded, dtype=torch.float32)
        self._last_temporary_bytes = 0
        if workspace is None or self._float_capacity < elements:
            capacity = max(1, self._float_capacity)
            while capacity < elements:
                capacity *= 2
            capacity = min(capacity, self.max_float32_bytes // 4)
            # The renderer normally runs under inference_mode, but ownership of
            # this reusable buffer must not depend on its first caller's mode.
            with torch.inference_mode(False):
                workspace = torch.empty(capacity, dtype=torch.float32, device=uploaded.device)
            self._float_workspace = workspace
            self._float_capacity = capacity
            self._float_allocations += 1
        return workspace[:elements].view(uploaded.shape)

    def release_resources(self):
        """Release retained inputs between calls; no caller-owned output is reused."""
        try:
            self._upload.release_resources()
        finally:
            self._float_workspace = None
            self._float_capacity = 0
            self._last_temporary_bytes = 0

    def cache_info(self):
        """Actual retained bytes and cumulative allocations, excluding outputs."""
        return dict(
            uint8_upload=self._upload.cache_info(),
            float32_bytes=self._float_capacity * 4,
            max_float32_bytes=self.max_float32_bytes,
            float32_allocations=self._float_allocations,
            float32_oversize_allocations=self._oversize_allocations,
            last_temporary_float32_bytes=self._last_temporary_bytes,
        )


__all__ = ["FrameUploadBuffer"]
