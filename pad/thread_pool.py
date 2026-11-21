"""thread_pool.py — v9.5.x dedicated thread pool for ONNX inference.

Why: FastAPI's event loop runs on a single thread. Every synchronous call
inside the WS handler blocks every other websocket message. A 150 ms
deepfake inference = 150 ms of frozen server. Offload these CPU/GPU-bound
calls to a dedicated pool so the event loop stays responsive.

Use pattern:

    from pad.thread_pool import run_in_pool
    score = await run_in_pool(score_patch_depth, face_crop)

Pool size is tuned for DirectML (single queue) + CPU fallback — one
executor thread per ONNX session is plenty; more threads just serialise
on the GPU anyway. Override with env `DODO_ONNX_WORKERS` (default 4).

The pool is process-global. Use `shutdown()` in tests to reset.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")


_WORKERS = int(os.environ.get("DODO_ONNX_WORKERS", "4"))
_POOL: concurrent.futures.ThreadPoolExecutor | None = None
_LOCK = threading.Lock()


def get_pool() -> concurrent.futures.ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        with _LOCK:
            if _POOL is None:
                _POOL = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_WORKERS,
                    thread_name_prefix="dodo-onnx",
                )
    return _POOL


async def run_in_pool(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run `fn(*args, **kwargs)` in the ONNX pool and await the result."""
    loop = asyncio.get_running_loop()
    pool = get_pool()
    if kwargs:
        return await loop.run_in_executor(pool, lambda: fn(*args, **kwargs))
    return await loop.run_in_executor(pool, fn, *args)


def run_sync(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Schedule + wait synchronously (for non-async callers)."""
    pool = get_pool()
    if kwargs:
        return pool.submit(lambda: fn(*args, **kwargs)).result()
    return pool.submit(fn, *args).result()


def shutdown(wait: bool = True) -> None:
    global _POOL
    with _LOCK:
        if _POOL is not None:
            _POOL.shutdown(wait=wait)
            _POOL = None


__all__ = ["get_pool", "run_in_pool", "run_sync", "shutdown"]
