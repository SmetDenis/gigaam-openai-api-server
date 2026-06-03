"""Юнит-тесты GigaAMEngine (короткое аудио) на моках.

gigaam.load_model подменяется фейком — ни весов, ни сети. probe_duration
подменяется, чтобы управлять маршрутизацией по длительности без реальных файлов.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gigaam
import pytest

from gigaam_api.asr import gigaam_engine
from gigaam_api.asr.engine import ASRResult, SegmentTS, WordTS
from gigaam_api.audio import AudioDecodeError
from gigaam_api.config import Settings


class _FakeModel:
    """Минимальный двойник GigaAMASR: фиксирует вызовы и возвращает заданный результат."""

    def __init__(self, result: Any = None, raises: Exception | None = None) -> None:
        self._result = result
        self._raises = raises
        self.calls: list[tuple[str, bool]] = []

    def transcribe(self, wav_file: str, word_timestamps: bool = False) -> Any:
        self.calls.append((wav_file, word_timestamps))
        if self._raises is not None:
            raise self._raises
        return self._result


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {"DEVICE": "cpu", "MODEL": "v3_ctc", "MODELS_DIR": tmp_path}
    base.update(overrides)
    return Settings(**base)


def _patch_load_model(monkeypatch: pytest.MonkeyPatch, model: _FakeModel) -> None:
    monkeypatch.setattr(gigaam, "load_model", lambda *a, **k: model)


def _patch_probe(monkeypatch: pytest.MonkeyPatch, duration: float) -> None:
    monkeypatch.setattr(gigaam_engine, "probe_duration", lambda _path: duration)


def test_transcribe_maps_result_without_words(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _FakeModel(result=SimpleNamespace(text="привет мир", words=None))
    _patch_load_model(monkeypatch, model)
    _patch_probe(monkeypatch, 3.0)

    engine = gigaam_engine.GigaAMEngine(_settings(tmp_path))
    result = engine.transcribe("/tmp/a.wav", word_timestamps=False)

    assert result == ASRResult(
        text="привет мир",
        duration=3.0,
        segments=[SegmentTS(text="привет мир", start=0.0, end=3.0, words=None)],
    )
    assert model.calls == [("/tmp/a.wav", False)]


def test_transcribe_maps_words_when_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    words = [
        SimpleNamespace(text="привет", start=0.0, end=0.5),
        SimpleNamespace(text="мир", start=0.6, end=1.0),
    ]
    model = _FakeModel(result=SimpleNamespace(text="привет мир", words=words))
    _patch_load_model(monkeypatch, model)
    _patch_probe(monkeypatch, 1.2)

    engine = gigaam_engine.GigaAMEngine(_settings(tmp_path))
    result = engine.transcribe("/tmp/a.wav", word_timestamps=True)

    assert result.segments[0].words == [
        WordTS(text="привет", start=0.0, end=0.5),
        WordTS(text="мир", start=0.6, end=1.0),
    ]
    assert model.calls == [("/tmp/a.wav", True)]


def test_transcribe_translates_gigaam_decode_runtime_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _FakeModel(raises=RuntimeError("Failed to load audio"))
    _patch_load_model(monkeypatch, model)
    _patch_probe(monkeypatch, 5.0)

    engine = gigaam_engine.GigaAMEngine(_settings(tmp_path))
    with pytest.raises(AudioDecodeError):
        engine.transcribe("/tmp/broken.wav", word_timestamps=False)


def test_transcribe_propagates_unrelated_runtime_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Не маскируем посторонние RuntimeError (например, ошибки инференса).
    model = _FakeModel(raises=RuntimeError("CUDA kaboom"))
    _patch_load_model(monkeypatch, model)
    _patch_probe(monkeypatch, 5.0)

    engine = gigaam_engine.GigaAMEngine(_settings(tmp_path))
    with pytest.raises(RuntimeError, match="CUDA kaboom"):
        engine.transcribe("/tmp/a.wav", word_timestamps=False)


def test_info_reports_model_device_loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_load_model(monkeypatch, _FakeModel(result=SimpleNamespace(text="x", words=None)))

    engine = gigaam_engine.GigaAMEngine(_settings(tmp_path))
    assert engine.info() == {"model": "v3_ctc", "device": "cpu", "loaded": True}


def test_resolve_device_auto_prefers_mps_then_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert gigaam_engine._resolve_device("auto") == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert gigaam_engine._resolve_device("auto") == "cpu"


def test_resolve_device_explicit_passthrough() -> None:
    assert gigaam_engine._resolve_device("cpu") == "cpu"
    assert gigaam_engine._resolve_device("cuda") == "cuda"
