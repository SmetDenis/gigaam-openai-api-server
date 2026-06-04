"""Lifespan tests: loading the engine into app.state and fail-fast on error.

GigaAMEngine is replaced by a fake/stub — weights are not downloaded. TestClient as a
context manager runs startup/shutdown (lifespan).
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import gigaam_api.asr.gigaam_engine as ge
from gigaam_api.asr.engine import ASRResult, EngineInfo, SegmentTS
from gigaam_api.config import Settings
from gigaam_api.main import create_app
from gigaam_api.runner import Runner


class _FakeEngine:
    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.MODEL
        self.device = "cpu"

    def transcribe(self, wav_path: str, *, word_timestamps: bool) -> ASRResult:
        raise NotImplementedError

    def iter_segments(self, wav_path: str, *, word_timestamps: bool) -> Iterator[SegmentTS]:
        raise NotImplementedError

    def info(self) -> EngineInfo:
        return {"model": self.model_name, "device": self.device, "loaded": True}


def test_lifespan_loads_engine_and_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ge, "GigaAMEngine", _FakeEngine)
    app = create_app()

    with TestClient(app) as client:
        assert isinstance(app.state.engine, _FakeEngine)
        assert isinstance(app.state.runner, Runner)
        assert client.get("/health").json()["loaded"] is True

    # shutdown released the engine and runner
    assert app.state.engine is None
    assert app.state.runner is None


def test_lifespan_fail_fast_on_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(settings: Settings) -> _FakeEngine:
        raise RuntimeError("weights download failed")

    monkeypatch.setattr(ge, "GigaAMEngine", _boom)
    app = create_app()

    with pytest.raises(RuntimeError, match="weights download failed"), TestClient(app):
        pass
