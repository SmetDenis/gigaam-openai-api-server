"""Тесты POST /v1/audio/transcriptions: форматы ответа + unhappy-flow.

engine — фейк (через app.state.engine), runner — реальный Runner, probe_duration
замокан (фейковые байты ffprobe не прочтёт). Фокус: все форматы, 413/400, игнор
prompt/temperature, words по granularity, отмена → 499.
"""

import json
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import gigaam_api.api.transcriptions as tx
from gigaam_api.api.transcriptions import router as transcriptions_router
from gigaam_api.asr.engine import ASRResult, InferenceCancelledError, SegmentTS, WordTS
from gigaam_api.config import Settings, get_settings
from gigaam_api.errors import register_exception_handlers
from gigaam_api.runner import Runner

_RESULT = ASRResult(
    text="привет мир",
    duration=2.0,
    segments=[
        SegmentTS(
            text="привет мир",
            start=0.0,
            end=2.0,
            words=[WordTS("привет", 0.0, 0.5), WordTS("мир", 0.6, 1.0)],
        )
    ],
)


class _FakeEngine:
    def __init__(self, result: ASRResult | None = None, exc: Exception | None = None) -> None:
        self.model_name = "v3_ctc"
        self.device = "cpu"
        self._result = result
        self._exc = exc
        self.calls: list[tuple[str, bool]] = []

    def transcribe(
        self,
        wav_path: str,
        *,
        word_timestamps: bool,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ASRResult:
        self.calls.append((wav_path, word_timestamps))
        if self._exc is not None:
            raise self._exc
        assert self._result is not None
        return self._result

    def iter_segments(
        self,
        wav_path: str,
        *,
        word_timestamps: bool,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Iterator[SegmentTS]:
        self.calls.append((wav_path, word_timestamps))
        if self._exc is not None:
            raise self._exc
        assert self._result is not None
        yield from self._result.segments

    def info(self) -> dict[str, Any]:
        return {"model": self.model_name, "device": self.device, "loaded": True}


def _build(engine: _FakeEngine, settings: Settings | None = None) -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    if settings is not None:
        app.dependency_overrides[get_settings] = lambda: settings
    app.include_router(transcriptions_router)
    app.state.engine = engine
    app.state.runner = Runner(max_queue=8)
    return app


@pytest.fixture(autouse=True)
def _patch_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tx, "probe_duration", lambda path: 2.0)


def _post(client: TestClient, **kwargs: Any) -> Any:
    files = {"file": ("a.wav", b"RIFFfakebytes", "audio/wav")}
    return client.post("/v1/audio/transcriptions", files=files, **kwargs)


def test_json_format() -> None:
    client = TestClient(_build(_FakeEngine(result=_RESULT)))
    resp = _post(client)  # response_format по умолчанию = json
    assert resp.status_code == 200
    assert resp.json() == {"text": "привет мир"}


def test_text_format() -> None:
    client = TestClient(_build(_FakeEngine(result=_RESULT)))
    resp = _post(client, data={"response_format": "text"})
    assert resp.status_code == 200
    assert resp.text == "привет мир"


def test_verbose_json_default_has_segments_no_words() -> None:
    client = TestClient(_build(_FakeEngine(result=_RESULT)))
    body = _post(client, data={"response_format": "verbose_json"}).json()
    assert body["language"] == "russian"
    assert "segments" in body and "words" not in body


def test_verbose_json_word_granularity_adds_words() -> None:
    engine = _FakeEngine(result=_RESULT)
    client = TestClient(_build(engine))
    body = _post(
        client,
        data={"response_format": "verbose_json", "timestamp_granularities[]": ["word"]},
    ).json()
    assert [w["word"] for w in body["words"]] == ["привет", "мир"]
    assert engine.calls[0][1] is True


def test_srt_format() -> None:
    client = TestClient(_build(_FakeEngine(result=_RESULT)))
    resp = _post(client, data={"response_format": "srt"})
    assert resp.status_code == 200
    assert "00:00:00,000 --> 00:00:02,000" in resp.text


def test_vtt_format() -> None:
    client = TestClient(_build(_FakeEngine(result=_RESULT)))
    resp = _post(client, data={"response_format": "vtt"})
    assert resp.status_code == 200
    assert resp.text.startswith("WEBVTT")


def test_invalid_model_rejected_400() -> None:
    client = TestClient(_build(_FakeEngine(result=_RESULT)))
    resp = _post(client, data={"model": "gpt-4o"})
    assert resp.status_code == 400
    assert resp.json()["error"]["param"] == "model"


def test_invalid_response_format_rejected_400() -> None:
    client = TestClient(_build(_FakeEngine(result=_RESULT)))
    resp = _post(client, data={"response_format": "yaml"})
    assert resp.status_code == 400
    assert resp.json()["error"]["param"] == "response_format"


def test_oversize_upload_rejected_413() -> None:
    client = TestClient(_build(_FakeEngine(result=_RESULT), settings=Settings(MAX_UPLOAD_MB=1)))
    big = b"\x00" * (1024 * 1024 + 16)  # >1 МБ
    resp = client.post("/v1/audio/transcriptions", files={"file": ("big.wav", big, "audio/wav")})
    assert resp.status_code == 413


def test_prompt_and_temperature_ignored() -> None:
    client = TestClient(_build(_FakeEngine(result=_RESULT)))
    resp = _post(client, data={"prompt": "контекст", "temperature": "0.7"})
    assert resp.status_code == 200
    assert resp.json() == {"text": "привет мир"}


def test_missing_file_rejected_400() -> None:
    client = TestClient(_build(_FakeEngine(result=_RESULT)))
    resp = client.post("/v1/audio/transcriptions", data={"model": "v3_ctc"})
    assert resp.status_code == 400


def test_cancelled_inference_returns_499() -> None:
    engine = _FakeEngine(exc=InferenceCancelledError("отменено"))
    client = TestClient(_build(engine))
    resp = _post(client)
    assert resp.status_code == 499


def test_queue_full_returns_503() -> None:
    from gigaam_api.runner import QueueFullError

    class _FullRunner:
        async def run(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
            raise QueueFullError("очередь полна")

        def shutdown(self) -> None:
            pass

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(transcriptions_router)
    app.state.engine = _FakeEngine(result=_RESULT)
    app.state.runner = _FullRunner()
    resp = _post(TestClient(app))
    assert resp.status_code == 503


# ----------------------------------------------------------------- стриминг (SSE)

_MULTI = ASRResult(
    text="раз два",
    duration=2.0,
    segments=[SegmentTS("раз", 0.0, 1.0), SegmentTS("два", 1.0, 2.0)],
)


def _parse_sse(body: str) -> list[Any]:
    """Распарсить SSE-тело в список полезных нагрузок (`dict` или строка `[DONE]`)."""
    out: list[Any] = []
    for frame in body.split("\n\n"):
        line = frame.strip()
        if not line.startswith("data: "):
            continue  # пропускаем heartbeat-комментарии и пустые
        payload = line[len("data: ") :]
        out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


def test_stream_json_emits_sse_and_done_matches_sync() -> None:
    client = TestClient(_build(_FakeEngine(result=_MULTI)))
    resp = _post(client, data={"response_format": "json", "stream": "true"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    payloads = _parse_sse(resp.text)
    deltas = [p["delta"] for p in payloads if isinstance(p, dict) and p["type"].endswith(".delta")]
    done = next(p for p in payloads if isinstance(p, dict) and p["type"].endswith(".done"))
    assert deltas == ["раз", " два"]  # префикс-пробел: ''.join == done.text
    assert "".join(deltas) == done["text"] == _MULTI.text  # done идентичен синхронному тексту
    assert payloads[-1] == "[DONE]"


def test_stream_text_format_emits_sse() -> None:
    client = TestClient(_build(_FakeEngine(result=_MULTI)))
    resp = _post(client, data={"response_format": "text", "stream": "true"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert _parse_sse(resp.text)[-1] == "[DONE]"


def test_stream_verbose_json_falls_back_to_sync() -> None:
    client = TestClient(_build(_FakeEngine(result=_MULTI)))
    resp = _post(client, data={"response_format": "verbose_json", "stream": "true"})

    # Синхронный фоллбэк: обычный JSON, не SSE (verbose требует полного результата).
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert body["text"] == "раз два"
    assert "segments" in body
