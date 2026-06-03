"""Тест эндпоинта GET /health.

Юнит-тесты не запускают lifespan (plain TestClient), поэтому модель не грузится:
загруженный движок имитируем, подставляя фейк в app.state.engine.
"""

from collections.abc import Iterator

from fastapi.testclient import TestClient

from gigaam_api.asr.engine import ASRResult, EngineInfo, SegmentTS
from gigaam_api.config import Settings, get_settings
from gigaam_api.main import create_app


class _FakeEngine:
    """Двойник движка, удовлетворяющий ASREngine (runtime_checkable)."""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device

    def transcribe(self, wav_path: str, *, word_timestamps: bool) -> ASRResult:
        raise NotImplementedError

    def iter_segments(self, wav_path: str, *, word_timestamps: bool) -> Iterator[SegmentTS]:
        raise NotImplementedError

    def info(self) -> EngineInfo:
        return {"model": self.model_name, "device": self.device, "loaded": True}


def test_health_reports_loaded_engine() -> None:
    app = create_app()
    app.state.engine = _FakeEngine(model_name="v3_e2e_rnnt", device="cpu")
    client = TestClient(app)

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "model": "v3_e2e_rnnt",
        "device": "cpu",
        "loaded": True,
    }


def test_health_without_engine_reports_not_loaded() -> None:
    app = create_app()
    # Движок не загружен (lifespan не запускался) — эхо настроек, loaded=false.
    app.dependency_overrides[get_settings] = lambda: Settings()
    client = TestClient(app)

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "model": "v3_ctc",
        "device": "auto",
        "loaded": False,
    }
