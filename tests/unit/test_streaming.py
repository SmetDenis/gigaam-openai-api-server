"""Unit tests for the SSE renderer: event format, delta-concatenation invariant, heartbeat, error.

`sse_transcription` — a pure async generator over an async iterator of segments:
delta xN (prefix space) → done → [DONE]; HEARTBEAT marker → SSE comment;
exception in the source → error event and close (no [DONE] afterward).
"""

import json
import threading
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

import pytest

from gigaam_api.asr.engine import SegmentTS
from gigaam_api.runner import Runner
from gigaam_api.streaming import (
    HEARTBEAT,
    StreamItem,
    format_sse,
    sse_transcription,
    stream_segments,
)


async def _aiter(items: Sequence[StreamItem]) -> AsyncIterator[StreamItem]:
    for it in items:
        yield it


async def _aiter_then_raise(
    items: Sequence[StreamItem], exc: Exception
) -> AsyncIterator[StreamItem]:
    for it in items:
        yield it
    raise exc


async def _collect(agen: AsyncIterator[str]) -> list[str]:
    return [frame async for frame in agen]


def _data(frame: str) -> Any:
    """Parse the payload of an SSE frame `data: <json>\\n\\n` into an object/string."""
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    payload = frame[len("data: ") : -2]
    return payload if payload == "[DONE]" else json.loads(payload)


def _seg(text: str) -> SegmentTS:
    return SegmentTS(text=text, start=0.0, end=1.0)


# ------------------------------------------------------------------ format_sse


def test_format_sse_dict_is_compact_and_keeps_unicode() -> None:
    frame = format_sse({"type": "transcript.text.delta", "delta": "привет"})
    assert frame == 'data: {"type": "transcript.text.delta", "delta": "привет"}\n\n'


def test_format_sse_string_terminator() -> None:
    assert format_sse("[DONE]") == "data: [DONE]\n\n"


# ------------------------------------------------------------- sse_transcription


@pytest.mark.asyncio
async def test_emits_deltas_done_and_terminator() -> None:
    segs = [_seg("раз"), _seg("два"), _seg("три")]
    frames = await _collect(sse_transcription(_aiter(segs), request_id="rid"))

    payloads = [_data(f) for f in frames]
    deltas = [
        p["delta"] for p in payloads if isinstance(p, dict) and p["type"] == "transcript.text.delta"
    ]
    done = next(p for p in payloads if isinstance(p, dict) and p["type"] == "transcript.text.done")

    # the separator "shifts" to the start of the next delta
    assert deltas == ["раз", " два", " три"]
    assert "".join(deltas) == done["text"] == "раз два три"  # the OpenAI invariant
    assert payloads[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_done_text_matches_sync_join() -> None:
    # The full done text must match the synchronous ' '.join(segment texts).
    segs = [_seg("первый"), _seg("второй")]
    frames = await _collect(sse_transcription(_aiter(segs), request_id="rid"))
    done = next(_data(f) for f in frames if '"transcript.text.done"' in f)
    assert done["text"] == " ".join(s.text for s in segs)


@pytest.mark.asyncio
async def test_error_event_on_source_exception_and_no_done() -> None:
    agen = _aiter_then_raise([_seg("раз")], RuntimeError("inference failed"))
    frames = await _collect(sse_transcription(agen, request_id="rid"))

    payloads = [_data(f) for f in frames]
    assert payloads[0]["type"] == "transcript.text.delta"
    err = payloads[-1]
    assert err["type"] == "error"
    assert err["error"]["message"] == "inference failed"
    assert err["error"]["type"] == "api_error"
    # After an error the stream is closed: neither done nor [DONE].
    assert not any(p == "[DONE]" for p in payloads)
    assert not any(isinstance(p, dict) and p["type"] == "transcript.text.done" for p in payloads)


@pytest.mark.asyncio
async def test_heartbeat_marker_renders_comment_and_is_not_text() -> None:
    items: list[StreamItem] = [_seg("раз"), HEARTBEAT, _seg("два")]
    frames = await _collect(sse_transcription(_aiter(items), request_id="rid"))

    assert ": keep-alive\n\n" in frames  # an SSE comment, not a data event
    done = next(_data(f) for f in frames if '"transcript.text.done"' in f)
    assert done["text"] == "раз два"  # the heartbeat did not leak into the text


# ------------------------------------------- stream_segments (thread→async bridge)


@pytest.mark.asyncio
async def test_stream_segments_yields_items_then_done() -> None:
    runner = Runner(max_queue=4)
    closed = threading.Event()
    segs = [_seg("раз"), _seg("два")]
    try:
        out: list[StreamItem] = []
        async for item in stream_segments(
            runner,
            lambda: iter(segs),
            request_id="rid",
            heartbeat_interval=5.0,
            cancel=threading.Event(),
            on_done=closed.set,
        ):
            out.append(item)
        assert [s.text for s in out if isinstance(s, SegmentTS)] == ["раз", "два"]
        assert closed.wait(1.0)  # on_done is called once the producer finishes (the worker is free)
    finally:
        runner.shutdown()


@pytest.mark.asyncio
async def test_stream_segments_emits_heartbeat_when_producer_idle() -> None:
    runner = Runner(max_queue=4)

    def slow() -> Iterator[SegmentTS]:
        time.sleep(0.15)  # longer than heartbeat_interval → the consumer gets a chance to "idle"
        yield _seg("раз")

    try:
        items: list[StreamItem] = []
        async for item in stream_segments(
            runner,
            slow,
            request_id="rid",
            heartbeat_interval=0.05,
            cancel=threading.Event(),
            on_done=lambda: None,
        ):
            items.append(item)
        assert any(it is HEARTBEAT for it in items)  # at least one heartbeat during the idle period
        assert [s.text for s in items if isinstance(s, SegmentTS)] == ["раз"]
    finally:
        runner.shutdown()


@pytest.mark.asyncio
async def test_stream_segments_propagates_producer_error_and_still_cleans_up() -> None:
    runner = Runner(max_queue=4)
    closed = threading.Event()

    def boom() -> Iterator[SegmentTS]:
        yield _seg("раз")
        raise RuntimeError("inference failed")

    try:
        out: list[StreamItem] = []
        with pytest.raises(RuntimeError, match="inference failed"):
            async for item in stream_segments(
                runner,
                boom,
                request_id="rid",
                heartbeat_interval=5.0,
                cancel=threading.Event(),
                on_done=closed.set,
            ):
                out.append(item)
        assert [s.text for s in out if isinstance(s, SegmentTS)] == ["раз"]
        assert closed.wait(1.0)  # cleanup runs even on a producer error
    finally:
        runner.shutdown()


@pytest.mark.asyncio
async def test_stream_segments_sets_cancel_on_consumer_close() -> None:
    runner = Runner(max_queue=4)
    cancel = threading.Event()

    def cancellable() -> Iterator[SegmentTS]:
        i = 0
        while not cancel.is_set():  # like iter_segments — reacts to cancellation
            yield _seg(f"s{i}")
            time.sleep(0.02)
            i += 1

    try:
        agen = stream_segments(
            runner,
            cancellable,
            request_id="rid",
            heartbeat_interval=5.0,
            cancel=cancel,
            on_done=lambda: None,
        )
        first = await agen.__anext__()
        assert isinstance(first, SegmentTS)
        await agen.aclose()  # the consumer went away
        assert cancel.is_set()  # the bridge's finally set the cancel flag
    finally:
        runner.shutdown()
