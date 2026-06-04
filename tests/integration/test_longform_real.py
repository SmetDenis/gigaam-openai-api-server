"""Integration: longform on the real v3_ctc model + a long sample (~40s).

The sample `ru_long_sample.wav` is a GigaAM long_example trimmed to 40s (real RU speech
with pauses), committed into data/. DEVICE=cpu — determinism and parity with prod CPU.
Graceful skip without network/weights, so that make pre-commit stays green offline.
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
        pytest.skip(f"no long test sample: {_SAMPLE}")
    cache = tmp_path_factory.mktemp("models")
    settings = Settings(MODEL="v3_ctc", DEVICE="cpu", MODELS_DIR=cache)
    try:
        return GigaAMEngine(settings)
    except Exception as exc:  # no network / CDN unavailable / weights failed to download
        pytest.skip(f"model unavailable (no network/weights): {exc}")


def test_longform_returns_multiple_monotonic_segments(engine: GigaAMEngine) -> None:
    result = engine.transcribe(str(_SAMPLE), word_timestamps=False)

    assert isinstance(result, ASRResult)
    assert result.text.strip(), "expected a non-empty transcript"
    assert result.duration == pytest.approx(40.0, abs=0.5)
    assert len(result.segments) > 1, "long audio should yield multiple segments"
    for seg in result.segments:
        assert seg.start <= seg.end
        assert seg.text.strip()
    # Chunk boundaries are monotonic and non-overlapping.
    for prev, nxt in pairwise(result.segments):
        assert prev.start < nxt.start
        assert prev.end <= nxt.start + 1e-6


def test_longform_word_timestamps_are_global(engine: GigaAMEngine) -> None:
    result = engine.transcribe(str(_SAMPLE), word_timestamps=True)

    words = [w for seg in result.segments if seg.words for w in seg.words]
    assert len(words) > 0, "expected words with timestamps"
    assert all(w.text for w in words)
    # Timestamps are global: within the duration and non-decreasing by start.
    for w in words:
        assert 0.0 <= w.start <= w.end <= result.duration + 0.5
    for prev, nxt in pairwise(words):
        assert prev.start <= nxt.start
