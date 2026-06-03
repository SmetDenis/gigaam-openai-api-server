"""Интеграция: longform на реальной модели v3_ctc + длинный сэмпл (~40с).

Сэмпл `ru_long_sample.wav` — обрезанный до 40с GigaAM long_example (реальная RU-речь
с паузами), закоммичен в data/. DEVICE=cpu — детерминизм и совпадение с прод-CPU.
Грейсфул-skip без сети/весов, чтобы make pre-commit оставался зелёным оффлайн.
"""

from itertools import pairwise
from pathlib import Path

import pytest

from gigaam_api.asr.engine import ASRResult
from gigaam_api.asr.gigaam_engine import GigaAMEngine
from gigaam_api.config import Settings

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


def test_longform_returns_multiple_monotonic_segments(engine: GigaAMEngine) -> None:
    result = engine.transcribe(str(_SAMPLE), word_timestamps=False)

    assert isinstance(result, ASRResult)
    assert result.text.strip(), "ожидали непустой транскрипт"
    assert result.duration == pytest.approx(40.0, abs=0.5)
    assert len(result.segments) > 1, "длинное аудио должно дать несколько сегментов"
    for seg in result.segments:
        assert seg.start <= seg.end
        assert seg.text.strip()
    # Границы чанков монотонны и не пересекаются.
    for prev, nxt in pairwise(result.segments):
        assert prev.start < nxt.start
        assert prev.end <= nxt.start + 1e-6


def test_longform_word_timestamps_are_global(engine: GigaAMEngine) -> None:
    result = engine.transcribe(str(_SAMPLE), word_timestamps=True)

    words = [w for seg in result.segments if seg.words for w in seg.words]
    assert len(words) > 0, "ожидали слова с таймстемпами"
    assert all(w.text for w in words)
    # Таймстемпы глобальны: в пределах длительности и неубывают по началу.
    for w in words:
        assert 0.0 <= w.start <= w.end <= result.duration + 0.5
    for prev, nxt in pairwise(words):
        assert prev.start <= nxt.start
