"""In-process sliding-window rate limiter.

Adequate for a single-process demo. A multi-worker deployment should swap the
backing store for Redis; the public interface here would not change.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque

from app.config import get_settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """Allow at most ``limit`` events per ``window`` seconds per key."""

    def __init__(self, limit: int | None = None, window: float = 60.0) -> None:
        self._limit = limit if limit is not None else get_settings().rate_limit_per_minute
        self._window = window
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> tuple[bool, int, float]:
        """Record an attempt for ``key``.

        Returns:
            ``(allowed, remaining, retry_after_seconds)``.
        """
        now = time.monotonic()
        async with self._lock:
            bucket = self._hits[key]
            cutoff = now - self._window
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self._limit:
                retry_after = max(0.0, self._window - (now - bucket[0]))
                logger.info("Rate limit hit for %s (retry in %.1fs)", key, retry_after)
                return False, 0, retry_after
            bucket.append(now)
            return True, self._limit - len(bucket), 0.0

    async def reset(self, key: str | None = None) -> None:
        """Clear one key's history, or all of them when ``key`` is ``None``."""
        async with self._lock:
            if key is None:
                self._hits.clear()
            else:
                self._hits.pop(key, None)


rate_limiter = RateLimiter()
