"""Bounded pinned staging for serial, inference-only CPU-to-CUDA uploads.

CUDA results are borrowed workspace views, valid until the next upload. Consume
them on the same compute stream before reusing this object. CPU targets instead
return independent tensors and do not retain a cached allocation. No source
array/tensor is stored by this helper.
"""

from __future__ import annotations

import math
from numbers import Integral

import torch


class PinnedUploadBuffer:
    """One device/dtype context with a bounded host and device capacity.

    max_cached_bytes limits EACH cached allocation: pinned host and CUDA device
    storage have identical capacities. Requests above that bound use temporary
    allocations, not a larger cache. An oversized host allocation remains alive
    only as pending-copy staging until the next prepare/release; the oversized
    device result is held only by its caller after upload.

    prepare() waits for the previous H2D event before allowing CPU writes. upload()
    queues a nonblocking copy and records an event on the current compute stream.
    Stream changes in a live CUDA context are rejected; release_resources() ends
    that context. It waits for this object's H2D copy, never globally synchronizes
    CUDA or empties allocator caches. Counters are cumulative across releases.
    """

    def __init__(self, max_cached_bytes=1024 * 1024):
        if type(max_cached_bytes) is not int or max_cached_bytes < 0:
            raise ValueError("max_cached_bytes must be a nonnegative Python integer")
        self.max_cached_bytes = max_cached_bytes
        self._host = self._device_storage = None
        self._shape = self._dtype = self._device = self._stream = None
        self._event = None
        self._event_recorded = False
        self._capacity = 0
        self._temporary = self._prepared = False
        self._reallocations = self._uploads = self._oversize_uploads = 0
        self._context_changes = 0

    def _wait_for_copy(self):
        if self._event_recorded:
            self._event.synchronize()
            self._event_recorded = False

    def release_resources(self):
        """Release owned allocations after their pending host read completes."""
        self._wait_for_copy()
        self._host = self._device_storage = None
        self._shape = self._dtype = self._device = self._stream = None
        self._event = None
        self._capacity = 0
        self._temporary = self._prepared = False

    @staticmethod
    def _dimensions(shape):
        if not isinstance(shape, (tuple, list, torch.Size)):
            raise ValueError("shape must be a tuple/list of nonnegative dimensions")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            for value in shape
        ):
            raise ValueError("shape must contain nonnegative integer dimensions")
        return tuple(int(value) for value in shape)

    @torch.no_grad()
    def prepare(self, shape, *, dtype, device):
        """Return a writable CPU view; do not write it after calling upload()."""
        shape = self._dimensions(shape)
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        device = torch.device(device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("upload device must be CPU or CUDA")
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.type == "cpu":
            self.release_resources()
            # Keep reusable storage writable even if a later call is made
            # outside the inference_mode context that first created it.
            with torch.inference_mode(False):
                self._host = torch.empty(shape, dtype=dtype, device="cpu")
            self._shape, self._dtype, self._device = shape, dtype, device
            self._temporary = self._prepared = True
            return self._host

        stream = torch.cuda.current_stream(device)
        context_changed = self._device != device or self._dtype != dtype
        if not context_changed and self._stream is not None and self._stream != stream:
            raise RuntimeError("upload buffer compute stream changed; release_resources first")
        self._wait_for_copy()
        if context_changed:
            self.release_resources()
            self._context_changes += 1
        elif self._temporary:
            self._host = self._device_storage = None
            self._capacity = 0
        self._device, self._dtype, self._stream = device, dtype, stream
        count = math.prod(shape)
        itemsize = dtype.itemsize
        cache_limit = self.max_cached_bytes // itemsize
        temporary = count > cache_limit
        if temporary or self._host is None or count > self._capacity:
            capacity = count if temporary else min(cache_limit, max(count, self._capacity * 2))
            with torch.inference_mode(False):
                host = torch.empty(capacity, dtype=dtype, device="cpu", pin_memory=True)
                target = torch.empty(capacity, dtype=dtype, device=device)
            self._host, self._device_storage = host, target
            self._capacity = 0 if temporary else capacity
            self._reallocations += 1
        if self._event is None:
            self._event = torch.cuda.Event(enable_timing=False)
        self._shape, self._temporary, self._prepared = shape, temporary, True
        return self._host[:count].view(shape)

    @torch.no_grad()
    def upload(self):
        """Upload the prepared values once and return a same-shape device view."""
        if not self._prepared:
            raise RuntimeError("prepare() must precede each upload()")
        if self._device.type == "cpu":
            result = self._host.clone()
            self.release_resources()
            return result
        stream = torch.cuda.current_stream(self._device)
        if stream != self._stream:
            raise RuntimeError("upload must use the same compute stream as prepare")
        count = math.prod(self._shape)
        result = self._device_storage[:count].view(self._shape)
        try:
            result.copy_(self._host[:count].view(self._shape), non_blocking=True)
            self._event.record(stream)
        except BaseException:
            # If recording fails after a copy was enqueued, still protect its
            # pinned source. Synchronize only this stream on the exception path.
            stream.synchronize()
            self._prepared = False
            raise
        self._event_recorded = True
        self._prepared = False
        self._uploads += 1
        if self._temporary:
            self._oversize_uploads += 1
            self._device_storage = None
        return result

    def cache_info(self):
        """Owned bytes, excluding caller-held views and allocator reserves."""
        host_bytes = 0 if self._host is None else self._host.numel() * self._host.element_size()
        device_bytes = (
            0
            if self._device_storage is None
            else self._device_storage.numel() * self._device_storage.element_size()
        )
        return {
            "device": None if self._device is None else str(self._device),
            "dtype": None if self._dtype is None else str(self._dtype),
            "shape": self._shape,
            "host_bytes": host_bytes,
            "device_bytes": device_bytes,
            "capacity_bytes": 0 if self._dtype is None else self._capacity * self._dtype.itemsize,
            "max_cached_bytes": self.max_cached_bytes,
            "temporary_host_bytes": host_bytes if self._temporary else 0,
            "temporary_device_bytes": device_bytes if self._temporary else 0,
            "reallocations": self._reallocations,
            "uploads": self._uploads,
            "oversize_uploads": self._oversize_uploads,
            "context_changes": self._context_changes,
            "copy_event_recorded": self._event_recorded,
        }


__all__ = ["PinnedUploadBuffer"]
