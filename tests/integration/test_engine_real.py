"""Integration: GigaAMEngine on the real v3_ctc model (integration marker).

Downloads the real weights into a tmp cache and recognizes a short Russian sample (~11s).
DEVICE=cpu — determinism and parity with prod CPU. Graceful skip if there is no
network/weights (CDN unavailable) — then make pre-commit stays green offline.
"""

from pathlib import Path

import pytest

from gigaam_api.asr.engine import ASRResult
from gigaam_api.asr.gigaam_engine import GigaAMEngine
from gigaam_api.config import Settings

pytestmark = pytest.mark.integration

_SAMPLE = Path(__file__).parent / "data" / "ru_short_sample.wav"


@pytest.fixture(scope="module")
def engine(tmp_path_factory: pytest.TempPathFactory) -> GigaAMEngine:
    if not _SAMPLE.exists():
        pytest.skip(f"no test sample: {_SAMPLE}")
    cache = tmp_path_factory.mktemp("models")
    settings = Settings(MODEL="v3_ctc", DEVICE="cpu", MODELS_DIR=cache)
    try:
        return GigaAMEngine(settings)
    except Exception as exc:  # no network / CDN unavailable / weights failed to download
        pytest.skip(f"model unavailable (no network/weights): {exc}")


def test_transcribe_short_returns_nonempty_text(engine: GigaAMEngine) -> None:
    result = engine.transcribe(str(_SAMPLE), word_timestamps=False)
    assert isinstance(result, ASRResult)
    assert result.text.strip(), "expected a non-empty transcript"
    assert result.duration > 0
    assert len(result.segments) == 1
    assert result.segments[0].words is None


def test_transcribe_short_with_word_timestamps(engine: GigaAMEngine) -> None:
    result = engine.transcribe(str(_SAMPLE), word_timestamps=True)
    words = result.segments[0].words
    assert words is not None and len(words) > 0
    assert all(w.text for w in words)
    assert all(w.start <= w.end for w in words)
