import threading
import time
from collections import OrderedDict
from typing import Any, Hashable


class TTLCache:
    """Small thread-safe LRU cache with per-entry TTL. In-process by design for a single
    instance; with multiple API replicas this moves to Redis (same keys, same TTLs)."""

    def __init__(self, maxsize: int, ttl_s: float):
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._data: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Any | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires, value = item
            if expires < time.monotonic():
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl_s, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class CorpusVersion:
    """Monotonic counter bumped whenever the searchable corpus changes (ingest / delete).
    It is part of every result-cache key, so stale results are never served."""

    def __init__(self):
        self._v = 0
        self._lock = threading.Lock()

    def bump(self) -> None:
        with self._lock:
            self._v += 1

    @property
    def value(self) -> int:
        return self._v
