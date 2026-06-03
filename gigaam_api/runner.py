"""Сериализация блокирующего инференса: один воркер + лимит очереди (backpressure).

`Runner` гарантирует не более одного инференса одновременно (event loop не блокируется)
и отклоняет запросы сверх `MAX_QUEUE` через `QueueFullError` (→ 503), вместо молчаливого
ожидания часами. Не знает про модель и HTTP (master §4.3, §11).
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
    """Очередь инференса переполнена (admitted ≥ MAX_QUEUE) — backpressure (→ 503)."""


class Runner:
    """Один `ThreadPoolExecutor(max_workers=1)` сериализует инференс; счётчик admitted
    (в очереди + в работе) ограничивает нагрузку."""

    def __init__(self, max_queue: int) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        self._max_queue = max_queue
        self._inflight = 0
        # admitted-счётчик меняют и event loop (run/try_acquire), и воркер-поток
        # (release из done-callback стрима) → защищаем lock'ом.
        self._lock = threading.Lock()

    def try_acquire(self) -> None:
        """Занять слот (backpressure). При переполнении (admitted ≥ MAX_QUEUE) → QueueFullError.

        Синхронный — вызывается в обработчике ДО `StreamingResponse`, чтобы 503 ушёл до
        отправки заголовков (стрим иначе уже был бы `200`)."""
        with self._lock:
            if self._inflight >= self._max_queue:
                logger.warning(
                    "очередь переполнена: inflight=%d max_queue=%d", self._inflight, self._max_queue
                )
                raise QueueFullError(f"очередь инференса переполнена (max_queue={self._max_queue})")
            self._inflight += 1
            logger.debug("инференс поставлен: inflight=%d", self._inflight)

    def release(self) -> None:
        """Освободить слот (парный к `try_acquire`)."""
        with self._lock:
            self._inflight -= 1
            logger.debug("инференс завершён: inflight=%d", self._inflight)

    def submit(self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Future[T]:
        """Поставить блокирующую `fn` в тот же единственный воркер (сериализация сохранена).

        Учёт admitted на себя НЕ берёт — вызывающий (стрим) сам делает `try_acquire`/`release`,
        т.к. слот держится на всё время стрима, а не до возврата `submit`."""
        return self._executor.submit(functools.partial(fn, *args, **kwargs))

    async def run(self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        """Исполнить блокирующую `fn` в единственном воркере. При переполнении → QueueFullError."""
        self.try_acquire()
        try:
            loop = asyncio.get_running_loop()
            partial = functools.partial(fn, *args, **kwargs)
            return await loop.run_in_executor(self._executor, partial)
        finally:
            self.release()

    def shutdown(self) -> None:
        """Остановить пул (вызывать в lifespan на shutdown)."""
        # cancel_futures=True отменяет задачи в очереди; уже запущенный поток доработает до конца.
        self._executor.shutdown(wait=False, cancel_futures=True)
