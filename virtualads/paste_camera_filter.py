"""Shared, host-side camera filtering for the advertisement paste pipeline.

Call ``compute_camera_filter_shared`` ONCE per frame, then reuse its returned
``(dist_coeffs_filtered, motion_rate)`` for every material group / rendering way.
The function does not create a Gaussian kernel or implement a motion estimator:
it calls the existing ``owner.compute_camera_motion_rate(sorted_w2c_list)``.

The original code selected ``count(k != frame_id)`` after sorting its neighbors.
When the current frame is present this is the LAST row, not the current row.
``index_policy='legacy'`` preserves that selection for speed/regression baselines;
``'corrected'`` (the default here) selects the actual current frame. Do not change
this policy at the same time as evaluating a performance-only migration.

This module needs NumPy, not SciPy or PyTorch. For finite real coefficient data,
the selected result matches SciPy ``median_filter(..., size=(median_size, 1),
mode='reflect', origin=0)[index]``: only that row's reflected window is evaluated.
Float16/extended floats and nonfinite coefficients are explicitly rejected.
No accelerator-to-host transfer occurs implicitly. Only coefficient vectors
actually used in the NumPy median must be on CPU. Fallback coefficients and w2c
poses are passed through without inspecting their values or changing devices.
"""

from collections import OrderedDict
from collections.abc import Mapping, MutableMapping
from numbers import Integral

import numpy as np


def _integer(value, name, *, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


class BoundedCameraHistory(MutableMapping):
    """A worker-owned, bounded history for a monotonically advancing live stream.

    Write ``history[frame_id] = record`` in increasing frame order. Replacing the
    latest frame is allowed; an older ID requires an explicit ``reset()`` first
    (for a seek or a new stream). Records are retained by reference, not copied.
    Deleting a record does not rewind the latest ID; ``clear()`` also resets it.

    Capacity counts records, not frame IDs. For a dense +/-4 window, capacity 9
    covers filtering the center when up to four future records are already
    present. Filtering the latest frame needs at least 5 records for its past
    radius 4. A larger lookahead or delayed consumer needs more capacity. Sparse
    IDs are valid but may provide fewer than three eligible neighbors. Evicted
    records cannot be recovered by querying an old frame.

    This container never takes ownership of, or trims, an external history.
    Its window lookup costs O(radius), independent of stream duration. Use one
    instance per serial worker; caller-owned records must stay CPU-compatible
    when their coefficient vectors are consumed by the camera filter.
    """

    def __init__(self, capacity=9):
        self._capacity = _integer(capacity, "capacity", minimum=1)
        self._records = OrderedDict()
        self._latest_frame_id = None

    @property
    def capacity(self):
        return self._capacity

    @property
    def latest_frame_id(self):
        return self._latest_frame_id

    def __getitem__(self, frame_id):
        return self._records[_integer(frame_id, "frame_id")]

    def __setitem__(self, frame_id, record):
        frame_id = _integer(frame_id, "frame_id")
        if not isinstance(record, Mapping):
            raise TypeError("camera history record must be a mapping")
        if self._latest_frame_id is not None and frame_id < self._latest_frame_id:
            raise ValueError("camera history frame IDs must be monotonic; reset() before a seek")
        self._records[frame_id] = record
        self._latest_frame_id = frame_id
        if len(self._records) > self._capacity:
            self._records.popitem(last=False)

    def __delitem__(self, frame_id):
        del self._records[_integer(frame_id, "frame_id")]

    def __iter__(self):
        return iter(self._records)

    def __len__(self):
        return len(self._records)

    def reset(self):
        """Release all retained records and allow a new frame-ID sequence."""
        self._records.clear()
        self._latest_frame_id = None

    def clear(self):
        self.reset()

    def items_in_window(self, frame_id, radius=4):
        """Yield retained records in ascending ID order, without scanning history."""
        frame_id = _integer(frame_id, "frame_id")
        radius = _integer(radius, "radius", minimum=0)
        for key in range(frame_id - radius, frame_id + radius + 1):
            record = self._records.get(key)
            if record is not None:
                yield key, record


def _require_host(value, name):
    """Inspect device metadata only, before any NumPy conversion or callback."""
    device = getattr(value, "device", None)
    if device is not None:
        kind = getattr(device, "type", None)
        if kind is None:
            kind = str(device).split(":", 1)[0]
        if kind != "cpu":
            raise TypeError(
                f"{name} is on {device}; camera filtering requires CPU history. "
                "Transfer it explicitly before this function, outside the timed "
                "rendering path; implicit GPU synchronization is not allowed."
            )
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _require_host(item, f"{name}[{index}]")


def _coefficient_vector(value, name):
    _require_host(value, name)
    # A CPU torch.Tensor can be accepted without importing torch. ``detach``
    # does not transfer data; dtype/device conversion is deliberately absent.
    if hasattr(value, "detach") and hasattr(value, "numpy"):
        value = value.detach().numpy()
    vector = np.asarray(value)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a nonempty 1-D coefficient vector")
    dtype = vector.dtype
    if dtype.kind not in "biuf" or (dtype.kind == "f" and dtype.itemsize not in (4, 8)):
        raise TypeError(f"{name} must use real integer/bool/float32/float64 data, got {dtype}")
    if dtype.kind in "iu" and (np.any(vector > 2**53) or np.any(vector < -(2**53))):
        raise ValueError(
            f"{name}: integer coefficients outside [-2**53, 2**53] are unsupported; "
            "SciPy internally rounds such integers, so exact parity is not guaranteed"
        )
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite; SciPy median_filter has undefined NaN behavior")
    return vector


def _median_row_reflect(values, index, size):
    """Compute one row of a (size, 1) half-sample-symmetric median filter."""
    # The public caller validates 2-D, finite coefficient data and odd size.
    count = values.shape[0]
    radius = size // 2
    positions = np.arange(index - radius, index + radius + 1, dtype=np.int64)
    # SciPy 'reflect': ... c b a | a b c ...; unlike np.pad(mode='reflect'),
    # the edge sample is repeated. Modulo also handles windows longer than N.
    folded = positions % (2 * count)
    indices = np.where(folded < count, folded, 2 * count - 1 - folded)
    window = values[indices]
    # Odd footprint: select an existing value rather than averaging two values.
    # np.partition preserves the stacked dtype, as SciPy's default output does.
    return np.partition(window, radius, axis=0)[radius].copy()


def _select_neighbor_row(neighbors, frame_id, index_policy, legacy_index):
    """Choose a sorted history row without changing either index policy."""
    if index_policy == "corrected":
        index = next((i for i, item in enumerate(neighbors) if item[0] == frame_id), None)
        if index is None:
            raise ValueError(
                f"current frame {frame_id} is absent from eligible camera history; "
                "cannot select its filtered coefficients"
            )
        return index

    if legacy_index >= len(neighbors):
        raise IndexError(
            f"legacy current_index={legacy_index} is out of range for {len(neighbors)} "
            f"neighbors because current frame {frame_id} is absent"
        )
    return legacy_index


def _stack_neighbor_coefficients(neighbors):
    """Return CPU coefficients ``[neighbors, coefficients]`` and unchanged poses."""
    vectors = []
    poses = []
    coefficient_shape = None
    for key, pose, coefficients in neighbors:
        vector = _coefficient_vector(coefficients, f"paste_data_stack[{key}]['dist_coeffs']")
        if coefficient_shape is not None and vector.shape != coefficient_shape:
            raise ValueError("neighbor coefficient vectors must have the same length")
        coefficient_shape = vector.shape
        vectors.append(vector)
        poses.append(pose)
    return np.stack(vectors), poses


def compute_camera_filter_shared(
    owner,
    data_dict,
    frame_id,
    *,
    index_policy="corrected",
    neighbor_radius=4,
    median_size=7,
):
    """Return ``(dist_coeffs_filtered, motion_rate)`` using NumPy coefficient filtering.

    ``owner.paste_data_stack`` maps integer frame IDs to records with ``K``,
    ``w2c`` and one-dimensional ``dist_coeffs``. As in the supplied original,
    records with exactly one field are skipped. Neighbors satisfy
    ``abs(key - frame_id) <= neighbor_radius``: radius 4 matches the original
    ``abs(...) < 5`` (its comment saying +/-2 was inconsistent with its code).
    ``BoundedCameraHistory`` reads that window directly. Other mappings retain
    the original full traversal and validate every frame-ID key, even outside
    the window. The filter never deletes records from either kind of history.

    Policies:
      * corrected: select the sorted row whose key equals frame_id;
      * legacy: select the number of included records whose key is not frame_id,
        exactly reproducing the old index calculation. With the current frame
        present this chooses the last sorted row.

    With fewer than three eligible records, return the original
    ``data_dict['dist_coeffs']`` and 0.0. No median, motion callback, or dictionary
    update occurs. This also permits the current frame to be absent in this
    fallback branch. With at least three records and no current row, corrected
    raises ValueError; legacy raises IndexError, corresponding to the original
    out-of-range access. The function never silently selects a neighboring row.

    With at least three records, update ``data_dict['dist_coeffs']`` with the
    selected filtered vector, then call the existing motion callback with the
    ORIGINAL w2c objects in sorted frame order, exactly once. No data-value cache
    is used. Repeated calls can therefore modify shared history if data_dict is
    itself a history record, as in the original function; reuse the result.

    CPU torch tensors are accepted for median coefficient conversion. Neighbor
    coefficient tensors on accelerators are rejected before NumPy conversion.
    Fallback coefficients are not read by this function and may remain on their
    original device. Likewise w2c objects are passed unchanged to the existing
    callback, which controls their device handling. This is a host preparation
    step, not a CUDA Graph kernel.
    """
    if index_policy not in ("corrected", "legacy"):
        raise ValueError("index_policy must be 'corrected' or 'legacy'")
    frame_id = _integer(frame_id, "frame_id")
    neighbor_radius = _integer(neighbor_radius, "neighbor_radius", minimum=0)
    median_size = _integer(median_size, "median_size", minimum=1)
    if median_size % 2 != 1:
        raise ValueError("median_size must be odd")
    if not isinstance(data_dict, MutableMapping):
        raise TypeError("data_dict must be a mutable mapping")
    fallback = data_dict["dist_coeffs"]
    history = getattr(owner, "paste_data_stack", None)
    if not isinstance(history, Mapping):
        raise TypeError("owner.paste_data_stack must be a mapping")

    neighbors = []
    legacy_index = 0
    history_items = (
        history.items_in_window(frame_id, neighbor_radius)
        if isinstance(history, BoundedCameraHistory)
        else history.items()
    )
    for key, record in history_items:
        key = _integer(key, "paste_data_stack frame ID")
        if abs(key - frame_id) > neighbor_radius:
            continue
        if not isinstance(record, Mapping):
            raise TypeError(f"paste_data_stack[{key}] must be a mapping")
        if len(record) == 1:
            continue
        # Access the same fields as the original. K is not converted or filtered.
        record["K"]
        neighbors.append((key, record["w2c"], record["dist_coeffs"]))
        legacy_index += key != frame_id
    neighbors.sort(key=lambda item: item[0])
    if len(neighbors) < 3:
        return fallback, 0.0

    index = _select_neighbor_row(neighbors, frame_id, index_policy, legacy_index)
    motion_callback = getattr(owner, "compute_camera_motion_rate", None)
    if not callable(motion_callback):
        raise TypeError("owner.compute_camera_motion_rate must be callable")
    coefficients, poses = _stack_neighbor_coefficients(neighbors)
    filtered = _median_row_reflect(coefficients, index, median_size)
    data_dict["dist_coeffs"] = filtered
    motion_rate = motion_callback(poses)
    return filtered, motion_rate


__all__ = ["BoundedCameraHistory", "compute_camera_filter_shared"]
