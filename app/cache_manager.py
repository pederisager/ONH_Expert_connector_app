"""Simple cache wrapper for fetched snippets and embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from diskcache import Cache


class CacheManager:
    """Disk-backed cache with configurable TTL (retention days)."""

    def __init__(self, directory: Path, retention_days: int = 14, enabled: bool = True) -> None:
        self.directory = directory
        self.retention_days = retention_days
        self.enabled = enabled
        self._cache: Optional[Cache] = None

    @property
    def cache(self) -> Cache:
        if self._cache is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._cache = Cache(str(self.directory))
        return self._cache

    def get(self, key: str) -> Any:
        if not self.enabled:
            return None
        return self.cache.get(key)

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        expire = self.retention_days * 24 * 60 * 60
        self.cache.set(key, value, expire=expire)

    def clear_stale(self, keys: Iterable[str] | None = None) -> None:
        if not self.enabled:
            return
        if keys:
            for key in keys:
                self.cache.delete(key, retry=True)
        else:
            self.cache.cull()
