"""Lightweight in-memory sliding-window rate limiter.

Deliberately dependency-free: the API runs as a single uvicorn worker on
Hugging Face Spaces, so an in-process store is sufficient. If the app ever
scales to multiple workers/replicas, swap this for a Redis-backed limiter.

Usage (as a FastAPI dependency):

    limit_research = RateLimiter(limit=10, window_seconds=60, name="research")

    @router.post("/research", dependencies=[Depends(limit_research)])
    async def research(...): ...

NOTE: deliberately no `from __future__ import annotations` here — FastAPI
cannot resolve string annotations on callable *instances* (no __globals__),
which would silently turn the `request: Request` dependency parameter into a
required query parameter.
"""
import os
import time
from collections import deque

from fastapi import HTTPException, Request

# Safety cap on tracked client keys so a spoofed-IP flood cannot grow memory
# without bound. Oldest keys are evicted first.
_MAX_KEYS = 10_000


def _client_key(request: Request) -> str:
    """Best-effort client identity: first hop of X-Forwarded-For, else peer IP."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """Sliding-window request limiter, keyed by client IP.

    `time_func` is injectable for deterministic tests.
    """

    def __init__(self, limit: int, window_seconds: float = 60.0,
                 name: str = "default", time_func=time.monotonic):
        self.limit = max(1, int(limit))
        self.window = float(window_seconds)
        self.name = name
        self._now = time_func
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> float | None:
        """Record a hit for `key`. Returns None if allowed, else seconds to wait."""
        now = self._now()
        window_start = now - self.window

        bucket = self._hits.get(key)
        if bucket is None:
            if len(self._hits) >= _MAX_KEYS:
                # Evict the oldest-inserted key (dicts preserve insertion order)
                self._hits.pop(next(iter(self._hits)))
            bucket = deque()
            self._hits[key] = bucket

        while bucket and bucket[0] <= window_start:
            bucket.popleft()

        if len(bucket) >= self.limit:
            return max(0.0, bucket[0] + self.window - now)

        bucket.append(now)
        return None

    async def __call__(self, request: Request) -> None:
        retry_after = self.check(_client_key(request))
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded for {self.name}: "
                       f"{self.limit} requests per {int(self.window)}s.",
                headers={"Retry-After": str(int(retry_after) + 1)},
            )


def _env_int(var: str, default: int) -> int:
    try:
        return int(os.getenv(var, default))
    except (TypeError, ValueError):
        return default


# Per-endpoint limiters. Tune via environment without code changes.
limit_predict = RateLimiter(
    _env_int("RATE_LIMIT_PREDICT_PER_MIN", 60), 60.0, "predict")
limit_batch = RateLimiter(
    _env_int("RATE_LIMIT_BATCH_PER_MIN", 10), 60.0, "batch-predict")
limit_research = RateLimiter(
    _env_int("RATE_LIMIT_RESEARCH_PER_MIN", 10), 60.0, "research")
limit_analytics = RateLimiter(
    _env_int("RATE_LIMIT_ANALYTICS_PER_MIN", 30), 60.0, "analytics")
