"""Bounded historical advertisement stacks; no CUDA work or pixel inspection.

The adapter separately owns its current working set. This cache keeps reusable
past stacks within both size/count limits and a byte budget. Stored source
references prevent Python object-id reuse from producing false cache hits.
The byte budget counts owned stacks, not the source objects they keep alive.
"""

from collections import OrderedDict


DEFAULT_MAX_CACHED_BYTES = 512 * 1024 * 1024
_UNSET_REVISION = object()


def tensor_bytes(tensor):
    """Logical bytes of an owned, contiguous stack; reads metadata only."""
    return tensor.numel() * tensor.element_size()


class SizeAssetCache:
    """Two-level LRU by effective source size and ordered source identity.

    Byte-pressure eviction removes the oldest entry in the least recently used
    size bank. Revision changes discard older video frames without resetting
    cumulative counters. Zero bytes disables historical storage.
    """

    def __init__(self, max_sizes=2, max_entries=128, max_bytes=DEFAULT_MAX_CACHED_BYTES):
        if (
            type(max_sizes) is not int
            or max_sizes < 1
            or type(max_entries) is not int
            or max_entries < 1
        ):
            raise ValueError("cache capacities must be positive Python integers")
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("max_cached_bytes must be a nonnegative Python integer")
        self.max_sizes = max_sizes
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.banks = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.skipped = 0
        self.current_hits = 0
        self.revision_changes = 0
        self._revision = _UNSET_REVISION

    def set_revision(self, revision):
        """Keep only the current revision; report whether it changed."""
        if self._revision is not _UNSET_REVISION and revision == self._revision:
            return False
        if self._revision is not _UNSET_REVISION:
            self.revision_changes += 1
        self.evictions += sum(len(bank) for bank in self.banks.values())
        self.banks.clear()
        self.bytes = 0
        self._revision = revision
        return True

    def get(self, hw, key):
        bank = self.banks.get(hw)
        if bank is None or key not in bank:
            self.misses += 1
            return None
        self.banks.move_to_end(hw)
        bank.move_to_end(key)
        self.hits += 1
        return bank[key][0]

    def hit_current(self, slots):
        """Preserve per-slot hit counters when the complete current set matches."""
        self.hits += slots
        self.current_hits += slots

    def _evict_oldest(self, hw):
        bank = self.banks[hw]
        _, (value, _) = bank.popitem(last=False)
        self.bytes -= tensor_bytes(value)
        self.evictions += 1
        if not bank:
            del self.banks[hw]

    def put(self, hw, key, value, source_refs):
        size = tensor_bytes(value)
        bank = self.banks.get(hw)
        previous = bank.pop(key, None) if bank is not None else None
        if previous is not None:
            self.bytes -= tensor_bytes(previous[0])
            if not bank:
                del self.banks[hw]
        if self.max_bytes == 0 or size > self.max_bytes:
            self.skipped += 1
            return
        bank = self.banks.setdefault(hw, OrderedDict())
        self.banks.move_to_end(hw)
        bank[key] = (value, source_refs)
        bank.move_to_end(key)
        self.bytes += size
        while len(bank) > self.max_entries:
            self._evict_oldest(hw)
        while len(self.banks) > self.max_sizes:
            oldest_hw = next(iter(self.banks))
            while oldest_hw in self.banks:
                self._evict_oldest(oldest_hw)
        while self.bytes > self.max_bytes:
            self._evict_oldest(next(iter(self.banks)))

    def tensors(self):
        """Iterate retained stacks for explicit diagnostics, never on the hot path."""
        for bank in self.banks.values():
            for value, _ in bank.values():
                yield value

    def info(self):
        return {
            "sizes": {hw: len(bank) for hw, bank in self.banks.items()},
            "hits": self.hits,
            "misses": self.misses,
            "max_cached_sizes": self.max_sizes,
            "max_entries_per_size": self.max_entries,
            "bytes": self.bytes,
            "max_cached_bytes": self.max_bytes,
            "evictions": self.evictions,
            "skipped_oversize": self.skipped,
            "current_hits": self.current_hits,
            "revision_changes": self.revision_changes,
        }
