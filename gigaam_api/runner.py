"""Serialization of blocking inference: one worker + queue limit (backpressure).

`Runner` guarantees no more than one inference at a time (the event loop is not blocked)
and rejects requests beyond `MAX_QUEUE` via `QueueFullError` (→ 503), instead of silently
waiting for hours. Knows nothing about the model or HTTP.
"""

import asyncio
import functools
import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class QueueFullError(Exception):
    """The inference queue is full (admitted ≥ MAX_QUEUE) — backpressure (→ 503)."""


class Runner:
    """A single `ThreadPoolExecutor(max_workers=1)` serializes inference; the admitted counter
    (in the queue + in progress) caps the load."""

    def __init__(self, max_queue: int) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        self._max_queue = max_queue
        self._inflight = 0
        # the admitted counter is mutated by both the event loop (run/try_acquire) and the
        # worker thread (release from the stream's done-callback) → protect it with a lock.
        self._lock = threading.Lock()

    def try_acquire(self) -> None:
        """Acquire a slot (backpressure). On overflow (admitted ≥ MAX_QUEUE) → QueueFullError.

        Synchronous — called in the handler BEFORE `StreamingResponse`, so that the 503 is sent
        before the headers (otherwise the stream would already be `200`)."""
        with self._lock:
            if self._inflight >= self._max_queue:
                logger.warning(
                    "queue full: inflight=%d max_queue=%d", self._inflight, self._max_queue
                )
                raise QueueFullError(f"inference queue is full (max_queue={self._max_queue})")
            self._inflight += 1
            logger.debug("inference enqueued: inflight=%d", self._inflight)

    def release(self) -> None:
        """Release a slot (the counterpart of `try_acquire`)."""
        with self._lock:
            self._inflight -= 1
            logger.debug("inference finished: inflight=%d", self._inflight)

    def submit(self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Future[T]:
        """Submit a blocking `fn` to the same single worker (serialization is preserved).

        Does NOT track admitted itself — the caller (the stream) does `try_acquire`/`release`,
        since the slot is held for the entire stream, not just until `submit` returns."""
        return self._executor.submit(functools.partial(fn, *args, **kwargs))

    async def run(self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        """Execute a blocking `fn` in the single worker. On overflow → QueueFullError."""
        self.try_acquire()
        try:
            loop = asyncio.get_running_loop()
            partial = functools.partial(fn, *args, **kwargs)
            return await loop.run_in_executor(self._executor, partial)
        finally:
            self.release()

    def shutdown(self) -> None:
        """Stop the pool (call in the lifespan on shutdown)."""
        # cancel_futures=True cancels queued tasks; an already-running thread runs to completion.
        self._executor.shutdown(wait=False, cancel_futures=True)
