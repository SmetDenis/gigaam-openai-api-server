"""Сериализация блокирующего инференса: один воркер + лимит очереди (backpressure).

`Runner` гарантирует не более одного инференса одновременно (event loop не блокируется)
и отклоняет запросы сверх `MAX_QUEUE` через `QueueFullError` (→ 503), вместо молчаливого
ожидания часами. Не знает про модель и HTTP (master §4.3, §11).
"""

import asyncio
import functools
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class QueueFullError(Exception):
    """Очередь инференса переполнена (admitted ≥ MAX_QUEUE) — backpressure (→ 503)."""


class Runner:
    """Один `ThreadPoolExecutor(max_workers=1)` сериализует инференс; счётчик admitted
    (в очереди + в работе) ограничивает нагрузку."""

    def __init__(self, max_queue: int) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        self._max_queue = max_queue
        self._inflight = 0

    async def run(self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        """Исполнить блокирующую `fn` в единственном воркере. При переполнении → QueueFullError."""
        if self._inflight >= self._max_queue:
            logger.warning(
                "очередь переполнена: inflight=%d max_queue=%d", self._inflight, self._max_queue
            )
            raise QueueFullError(f"очередь инференса переполнена (max_queue={self._max_queue})")
        self._inflight += 1
        logger.debug("инференс поставлен: inflight=%d", self._inflight)
        try:
            loop = asyncio.get_running_loop()
            partial = functools.partial(fn, *args, **kwargs)
            return await loop.run_in_executor(self._executor, partial)
        finally:
            self._inflight -= 1
            logger.debug("инференс завершён: inflight=%d", self._inflight)

    def shutdown(self) -> None:
        """Остановить пул (вызывать в lifespan на shutdown)."""
        # cancel_futures=True отменяет задачи в очереди; уже запущенный поток доработает до конца.
        self._executor.shutdown(wait=False, cancel_futures=True)
