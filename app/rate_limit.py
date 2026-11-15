"""An in-memory request limit per client: at most N requests in any 60 seconds.

One process only: each API process keeps its own counts, which is enough for a single
container. It bounds how fast one address can spend the model budget; it is not a defence
against many addresses (the API's concurrency limit covers that).

The work per request does not grow with the number of clients: idle clients are swept at most
once per window, and when `max_keys` clients are being tracked, new ones share one bucket.
"""

import math
import threading
import time
from collections import deque

WINDOW_SECONDS = 60.0
MAX_KEYS = 10_000
OVERFLOW = "(overflow)"  # the shared bucket for clients beyond `max_keys`


class RateLimiter:
    """A sliding window per key (the client's IP address)."""

    def __init__(
        self, limit: int, window: float = WINDOW_SECONDS, max_keys: int = MAX_KEYS
    ) -> None:
        self.limit = limit
        self.window = window
        self.max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}
        self._swept_at: float | None = None
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> int | None:
        """Count a request. Returns None when it is allowed, else the seconds to wait."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._swept_at is None or now - self._swept_at >= self.window:
                self._sweep(now)
            if key not in self._hits and len(self._hits) >= self.max_keys:
                self._sweep(now)
                if len(self._hits) >= self.max_keys:
                    key = OVERFLOW
            hits = self._hits.setdefault(key, deque())
            self._expire(hits, now)
            if len(hits) >= self.limit:
                return max(1, math.ceil(hits[0] + self.window - now))
            hits.append(now)
            return None

    def _expire(self, hits: deque[float], now: float) -> None:
        while hits and hits[0] <= now - self.window:
            hits.popleft()

    def _sweep(self, now: float) -> None:
        """Forget every client with no request left in the window."""
        self._swept_at = now
        for key in list(self._hits):
            self._expire(self._hits[key], now)
            if not self._hits[key]:
                del self._hits[key]
