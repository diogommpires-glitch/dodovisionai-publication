"""rate_limit.py — per-token-per-IP token-bucket rate limiter.

Zero-dep (stdlib only) async token bucket for FastAPI / Starlette routes.

Usage:
    from rate_limit import TokenBucket, get_limiter

    limiter = get_limiter()
    async def ws_detect(ws):
        ok = await limiter.try_acquire(ws_key(ws), cost=1.0)
        if not ok:
            await ws.close(code=4429, reason="rate limited")
            return
        ...

Configuration (env vars):
    DODO_WS_RATE_PER_SEC     requests/sec per (token, ip) pair  (default 30)
    DODO_WS_RATE_BURST       burst capacity                     (default 60)
"""
from __future__ import annotations
import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TokenBucket:
    """Single token bucket: continuously refills at `rate` tokens/sec up to
    `capacity`. `try_acquire(cost)` returns True if enough tokens are
    available, else False (no blocking)."""
    rate: float            # tokens per second
    capacity: float        # max burst
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()

    async def try_acquire(self, cost: float = 1.0) -> bool:
        async with self._lock:
            now = time.monotonic()
            dt = now - self.last_refill
            if dt > 0:
                self.tokens = min(self.capacity, self.tokens + dt * self.rate)
                self.last_refill = now
            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False


@dataclass
class MultiBucketLimiter:
    """Maintains a bucket per client key. LRU-evicts when too many keys."""
    rate: float
    capacity: float
    max_keys: int = 2048
    _buckets: Dict[str, TokenBucket] = field(default_factory=dict, init=False)

    async def try_acquire(self, key: str, cost: float = 1.0) -> bool:
        b = self._buckets.get(key)
        if b is None:
            # Cheap LRU evict: drop 10% of buckets when the dict is full
            if len(self._buckets) >= self.max_keys:
                for k in list(self._buckets.keys())[: self.max_keys // 10]:
                    self._buckets.pop(k, None)
            b = TokenBucket(rate=self.rate, capacity=self.capacity)
            self._buckets[key] = b
        return await b.try_acquire(cost)

    def stats(self) -> Dict[str, int]:
        return {"buckets": len(self._buckets),
                "rate_per_sec": self.rate,
                "burst": int(self.capacity)}


_LIMITER: Optional[MultiBucketLimiter] = None


def get_limiter() -> MultiBucketLimiter:
    """Lazily initialised process-wide limiter. Configured from env vars."""
    global _LIMITER
    if _LIMITER is None:
        rate = float(os.environ.get("DODO_WS_RATE_PER_SEC", "30"))
        burst = float(os.environ.get("DODO_WS_RATE_BURST", "60"))
        _LIMITER = MultiBucketLimiter(rate=rate, capacity=burst)
    return _LIMITER


def key_for_request(ip: str, token: Optional[str] = None) -> str:
    """Canonical key: (token, ip). Both present -> both matter."""
    return f"{token or 'anon'}|{ip or 'unknown'}"


__all__ = ["TokenBucket", "MultiBucketLimiter", "get_limiter", "key_for_request"]
