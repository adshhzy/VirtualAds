"""Private, per-call route metadata and one bounded CPU parameter workspace.

Only ImageRenderer's standard execution path borrows parameter views. Public
route plans and custom executor/warp callbacks retain their original ownership.
No input objects or parameter values are cached between calls here.
"""

import math

import numpy as np


class RoutePreparationWorkspace:
    def __init__(self, max_cached_bytes=1024 * 1024):
        if type(max_cached_bytes) is not int or max_cached_bytes < 0:
            raise ValueError("max_cached_bytes must be a nonnegative Python integer")
        self.max_cached_bytes = max_cached_bytes
        self._storage = self._active = None
        self._offset = 0
        self._reallocations = self._temporary_allocations = 0
        self.counts = dict(
            prepared_plans=0,
            matrix_stacks=0,
            opacity_vectors=0,
            metadata_hits=0,
            metadata_fallbacks=0,
        )

    def begin(self, member_count):
        self.counts["prepared_plans"] += 1
        # Per member: a float64 3x3 matrix, float32 opacity and alignment slack.
        required = member_count * 88
        if required > self.max_cached_bytes:
            self._storage = None
            self._active = np.empty(required, dtype=np.uint8)
            self._temporary_allocations += 1
        elif required:
            if self._storage is None or self._storage.nbytes < required:
                previous = 0 if self._storage is None else self._storage.nbytes
                capacity = min(self.max_cached_bytes, max(required, previous * 2))
                self._storage = np.empty(capacity, dtype=np.uint8)
                self._reallocations += 1
            self._active = self._storage
        else:
            self.release_resources()
        self._offset = 0

    def array(self, shape, dtype):
        dtype = np.dtype(dtype)
        start = (self._offset + 7) // 8 * 8
        end = start + math.prod(shape) * dtype.itemsize
        if self._active is None or end > self._active.nbytes:
            raise RuntimeError("route preparation workspace capacity was underestimated")
        self._offset = end
        return self._active[start:end].view(dtype).reshape(shape)

    def finish(self):
        # Plans own their temporary views until this render finishes. Keeping the
        # active pointer here would retain an oversized allocation between calls.
        self._active = None
        self._offset = 0

    def release_resources(self):
        self.finish()
        self._storage = None

    def cache_info(self):
        return {
            "retained_bytes": 0 if self._storage is None else self._storage.nbytes,
            "max_cached_bytes": self.max_cached_bytes,
            "reallocations": self._reallocations,
            "temporary_allocations": self._temporary_allocations,
            **self.counts,
        }


class PreparedAdSources:
    """Single-use description, with a live guard across shared owner callbacks."""

    def __init__(self, groups, params, options, image_index, description, checks, counts):
        self.groups, self.params = groups, params
        self.options, self.image_index = options, image_index
        self._description = description
        self._checks = checks
        self._counts = counts

    def consume(self, groups, params, options, image_index):
        description, self._description = self._description, None
        checks, self._checks = self._checks, ()
        if (
            groups is not self.groups
            or params is not self.params
            or options is not self.options
            or image_index != self.image_index
        ):
            self._counts["metadata_fallbacks"] += 1
            return None
        for image, shape, dtype, device in checks:
            if (
                tuple(image.shape) != shape
                or image.dtype != dtype
                or getattr(image, "device", "cpu") != device
            ):
                self._counts["metadata_fallbacks"] += 1
                return None
        self._counts["metadata_hits" if description is not None else "metadata_fallbacks"] += 1
        return description


class PreparedRouteData(dict):
    """Internal call envelope; no hidden keys reach the ordinary ad-params API."""

    def __init__(self, params, sources):
        super().__init__(ad_params=params)
        self.sources = sources
