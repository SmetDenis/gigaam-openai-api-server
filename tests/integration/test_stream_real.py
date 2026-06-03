"""Интеграция: SSE-стрим на реальной модели v3_ctc + длинный сэмпл (~40с).

End-to-end через HTTP: POST stream=true → собираем SSE-события → склеенный текст из
`transcript.text.done` обязан совпасть с синхронным `transcribe` на том же файле
(acceptance §05). DEVICE=cpu — детерминизм greedy-декода. Грейсфул-skip без сети/весов.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gigaam_api.api.transcriptions import router as transcriptions_router
from gigaam_api.asr.gigaam_engine import GigaAMEngine
from gigaam_api.config import Settings, get_settings
from gigaam_api.errors import register_exception_handlers
from gigaam_api.runner import Runner

pytestmark = pytest.mark.integration

_SAMPLE = Path(__file__).parent / "data" / "ru_long_sample.wav"


@pytest.fixture(scope="module")
def engine(tmp_path_factory: pytest.TempPathFactory) -> GigaAMEngine:
    if not _SAMPLE.exists():
        pytest.skip(f"нет длинного тест-сэмпла: {_SAMPLE}")
    cache = tmp_path_factory.mktemp("models")
    settings = Settings(MODEL="v3_ctc", DEVICE="cpu", MODELS_DIR=cache)
    try:
        return GigaAMEngine(settings)
    except Exception as exc:  # нет сети / CDN недоступен / веса не скачались
        pytest.skip(f"модель недоступна (нет сети/весов): {exc}")


@pytest.fixture
def client(engine: GigaAMEngine) -> Iterator[TestClient]:
    app = FastAPI()
    register_exception_handlers(app)
    app.dependency_overrides[get_settings] = lambda: Settings(MODEL="v3_ctc", DEVICE="cpu")
    app.include_router(transcriptions_router)
    app.state.engine = engine
    runner = Runner(max_queue=8)
    app.state.runner = runner
    try:
        yield TestClient(app)
    finally:
        runner.shutdown()


def _parse_sse(body: str) -> list[Any]:
    out: list[Any] = []
    for frame in body.split("\n\n"):
        line = frame.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


def test_stream_done_text_matches_sync(engine: GigaAMEngine, client: TestClient) -> None:
    reference = engine.transcribe(str(_SAMPLE), word_timestamps=False).text

    with _SAMPLE.open("rb") as f:
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("ru_long_sample.wav", f, "audio/wav")},
            data={"response_format": "json", "stream": "true"},
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    payloads = _parse_sse(resp.text)
    deltas = [p["delta"] for p in payloads if isinstance(p, dict) and p["type"].endswith(".delta")]
    done = next(p for p in payloads if isinstance(p, dict) and p["type"].endswith(".done"))

    assert payloads[-1] == "[DONE]"
    assert len(deltas) > 1, "длинное аудио должно дать несколько delta-сегментов"
    # Инвариант: склейка delta == done.text == синхронному результату.
    assert "".join(deltas) == done["text"] == reference
