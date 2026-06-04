"""Assembly of SSE events for `stream=true`.

Contract (verified against OpenAI Speech-to-text, 06.2026):
- per segment → `{"type":"transcript.text.delta","delta":"<text>"}`;
- at the end → `{"type":"transcript.text.done","text":"<full text>"}` → `data: [DONE]`;
- error in the source → `{"type":"error","error":{...}}` and close (no `[DONE]`).

Invariant (as in OpenAI): the concatenation of all `delta` == `done.text` == the synchronous
response. The separator "moves" to the start of the next delta (the first one has no leading space),
so `"".join(delta) == " ".join(segment texts)`.

Here — only event rendering (a pure async generator over an async iterator of segments).
The "blocking engine generator → async" bridge + heartbeat live in `stream_segments` (see below).
"""

import asyncio
import json
import logging
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator

from gigaam_api.asr.engine import SegmentTS
from gigaam_api.runner import Runner

logger = logging.getLogger(__name__)

# Heartbeat interval: on a slow CPU a single batch takes minutes without events —
# a periodic SSE comment keeps the connection alive against proxy idle timeouts.
STREAM_HEARTBEAT_SECONDS = 15.0

_HEARTBEAT_COMMENT = (
    ": keep-alive\n\n"  # SSE comment: ignored by clients, keeps the connection alive
)


class _Heartbeat:
    """Marker for "no events yet" — rendered into an SSE comment against proxy idle timeouts."""


HEARTBEAT = _Heartbeat()

# Stream item: either a ready segment or a heartbeat marker (when there are no events yet).
StreamItem = SegmentTS | _Heartbeat


def format_sse(data: dict[str, object] | str) -> str:
    """Assemble a single SSE frame `data: <payload>\\n\\n`.

    `str` is returned as is (the `[DONE]` terminator); `dict` is serialized to compact
    JSON preserving Cyrillic (`ensure_ascii=False`).
    """
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"


def _error_event(exc: Exception) -> dict[str, object]:
    """OpenAI-format error event for an error in the middle of the stream."""
    return {
        "type": "error",
        "error": {"message": str(exc), "type": "api_error", "param": None, "code": None},
    }


async def sse_transcription(
    items: AsyncIterator[StreamItem], *, request_id: str
) -> AsyncIterator[str]:
    """Render a stream of segments/heartbeats into SSE frames.

    `CancelledError`/`GeneratorExit` (the client disconnected) are intentionally NOT caught —
    we hand cleanup upward; we catch only genuine source errors → an error event.
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
        logger.exception("request_id=%s error in the SSE stream", request_id)
        yield format_sse(_error_event(exc))


class _Done:
    """Internal queue marker: the producer has exhausted the segments."""


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
    """Bridge: a blocking segment generator (in the Runner worker) → an async iterator.

    Inference serialization is preserved — the producer goes into the same single worker
    (`runner.submit`), not into a temporary thread. Between the thread and the event loop — an
    `asyncio.Queue` (filled via `call_soon_threadsafe`); if no segment arrived within
    `heartbeat_interval`, we yield `HEARTBEAT` (rendered into an SSE comment against proxy
    idle timeouts).

    Lifecycle:
    - `on_done` (release the slot + delete the temp file) is called when the producer has ACTUALLY
      finished (the worker is free) — via the future's done-callback, not when the consumer
      finished reading;
    - `cancel` is set in `finally` (the consumer left/was cancelled) → `iter_segments` will stop
      between batches;
    - a producer error is propagated to the consumer (→ `sse_transcription` emits an error event).
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[SegmentTS | _Done | Exception] = asyncio.Queue()

    def produce() -> None:
        try:
            for segment in make_iter():
                loop.call_soon_threadsafe(queue.put_nowait, segment)
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)
        # any inference error → to the consumer, the thread doesn't fail silently
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)

    future = runner.submit(produce)
    future.add_done_callback(lambda _f: on_done())  # worker is free → release + cleanup
    logger.info("request_id=%s stream started", request_id)
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
