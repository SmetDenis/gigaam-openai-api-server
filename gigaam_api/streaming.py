"""Сборка SSE-событий для `stream=true`.

Контракт (проверен против OpenAI Speech-to-text, 06.2026):
- на каждый сегмент → `{"type":"transcript.text.delta","delta":"<text>"}`;
- в конце → `{"type":"transcript.text.done","text":"<полный текст>"}` → `data: [DONE]`;
- ошибка в источнике → `{"type":"error","error":{...}}` и закрытие (без `[DONE]`).

Инвариант (как у OpenAI): конкатенация всех `delta` == `done.text` == синхронному
ответу. Разделитель «уезжает» в начало следующего delta (первый — без пробела), поэтому
`"".join(delta) == " ".join(тексты сегментов)`.

Здесь — только рендер событий (чистый async-генератор поверх async-итератора сегментов).
Мост «блокирующий генератор движка → async» + heartbeat живут в `stream_segments` (см. ниже).
"""

import asyncio
import json
import logging
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator

from gigaam_api.asr.engine import SegmentTS
from gigaam_api.runner import Runner

logger = logging.getLogger(__name__)

# Интервал heartbeat'а: на медленном CPU один батч считается минутами без событий —
# периодический SSE-комментарий держит соединение против idle-таймаутов прокси.
STREAM_HEARTBEAT_SECONDS = 15.0

_HEARTBEAT_COMMENT = (
    ": keep-alive\n\n"  # SSE-комментарий: игнорируется клиентами, держит соединение
)


class _Heartbeat:
    """Маркер «событий пока нет» — рендерится в SSE-комментарий против idle-таймаутов прокси."""


HEARTBEAT = _Heartbeat()

# Элемент потока: либо готовый сегмент, либо heartbeat-маркер (когда событий ещё нет).
StreamItem = SegmentTS | _Heartbeat


def format_sse(data: dict[str, object] | str) -> str:
    """Собрать один SSE-фрейм `data: <payload>\\n\\n`.

    `str` отдаётся как есть (терминатор `[DONE]`); `dict` сериализуется компактным
    JSON с сохранением кириллицы (`ensure_ascii=False`).
    """
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"


def _error_event(exc: Exception) -> dict[str, object]:
    """OpenAI-формат error-события для ошибки в середине потока."""
    return {
        "type": "error",
        "error": {"message": str(exc), "type": "api_error", "param": None, "code": None},
    }


async def sse_transcription(
    items: AsyncIterator[StreamItem], *, request_id: str
) -> AsyncIterator[str]:
    """Отрендерить поток сегментов/heartbeat'ов в SSE-фреймы.

    `CancelledError`/`GeneratorExit` (клиент отключился) намеренно НЕ перехватываются —
    отдаём очистку наверх; перехватываем только настоящие ошибки источника → error-событие.
    """
    parts: list[str] = []
    try:
        async for item in items:
            if isinstance(item, _Heartbeat):
                yield _HEARTBEAT_COMMENT
                continue
            delta = item.text if not parts else " " + item.text
            parts.append(item.text)
            logger.debug(
                "request_id=%s delta idx=%d len=%d", request_id, len(parts) - 1, len(delta)
            )
            yield format_sse({"type": "transcript.text.delta", "delta": delta})
        full_text = " ".join(parts)
        logger.info(
            "request_id=%s stream done segments=%d chars=%d", request_id, len(parts), len(full_text)
        )
        yield format_sse({"type": "transcript.text.done", "text": full_text})
        yield format_sse("[DONE]")
    except Exception as exc:
        logger.exception("request_id=%s ошибка в SSE-потоке", request_id)
        yield format_sse(_error_event(exc))


class _Done:
    """Внутренний маркер очереди: продюсер исчерпал сегменты."""


_DONE = _Done()


async def stream_segments(
    runner: Runner,
    make_iter: Callable[[], Iterator[SegmentTS]],
    *,
    request_id: str,
    heartbeat_interval: float,
    cancel: threading.Event,
    on_done: Callable[[], None],
) -> AsyncGenerator[StreamItem, None]:
    """Мост: блокирующий генератор сегментов (в воркере Runner) → async-итератор.

    Сериализация инференса сохранена — продюсер идёт в тот же единственный воркер
    (`runner.submit`), а не во временный поток. Между потоком и event loop — `asyncio.Queue`
    (наполняется через `call_soon_threadsafe`); если за `heartbeat_interval` сегмент не пришёл,
    отдаём `HEARTBEAT` (рендерится в SSE-комментарий против idle-таймаутов прокси).

    Жизненный цикл:
    - `on_done` (release слота + удаление temp-файла) вызывается, когда продюсер РЕАЛЬНО
      завершился (воркер свободен) — через done-callback future, не когда потребитель дочитал;
    - `cancel` выставляется в `finally` (потребитель ушёл/отменён) → `iter_segments` остановится
      между батчами;
    - ошибка продюсера пробрасывается потребителю (→ `sse_transcription` отдаст error-событие).
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[SegmentTS | _Done | Exception] = asyncio.Queue()

    def produce() -> None:
        try:
            for segment in make_iter():
                loop.call_soon_threadsafe(queue.put_nowait, segment)
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)
        except Exception as exc:  # любая ошибка инференса → потребителю, поток не падает молча
            loop.call_soon_threadsafe(queue.put_nowait, exc)

    future = runner.submit(produce)
    future.add_done_callback(lambda _f: on_done())  # воркер свободен → release + cleanup
    logger.info("request_id=%s stream начат", request_id)
    try:
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=heartbeat_interval)
            except TimeoutError:
                yield HEARTBEAT
                continue
            if isinstance(message, _Done):
                return
            if isinstance(message, Exception):
                raise message
            yield message
    finally:
        cancel.set()
