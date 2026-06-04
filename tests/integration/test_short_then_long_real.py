"""Regression: on a SINGLE engine instance a short request must not break longform.

Reproduces a live-service bug: `gigaam.transcribe` (the short path) is wrapped in
`@torch.inference_mode()` and caches the encoder's rotary `cos`/`sin` as inference tensors;
our longform calls `forward`/`_decode` directly and, without its own `inference_mode` wrapper,
failed on this cache with `RuntimeError: Inference tensors cannot be saved for backward`.

Other integration tests use a SEPARATE instance per file → they did not catch the bug. Here
we deliberately use one engine and the short→long order (as in the live service with a
single model).
"""

from pathlib import Path

import pytest

from gigaam_api.asr.engine import ASRResult
from gigaam_api.asr.gigaam_engine import GigaAMEngine
from gigaam_api.config import Settings

pytestmark = pytest.mark.integration

_SHORT = Path(__file__).parent / "data" / "ru_short_sample.wav"
_LONG = Path(__file__).parent / "data" / "ru_long_sample.wav"


@pytest.fixture
def engine(tmp_path_factory: pytest.TempPathFactory) -> GigaAMEngine:
    if not (_SHORT.exists() and _LONG.exists()):
        pytest.skip("no test samples (short and/or long)")
    cache = tmp_path_factory.mktemp("models")
    settings = Settings(MODEL="v3_ctc", DEVICE="cpu", MODELS_DIR=cache)
    try:
        return GigaAMEngine(settings)
    except Exception as exc:  # no network / CDN unavailable / weights failed to download
        pytest.skip(f"model unavailable (no network/weights): {exc}")


def test_short_then_long_same_engine(engine: GigaAMEngine) -> None:
    # The short path (gigaam.transcribe under inference_mode) "poisons" the rotary cos/sin cache.
    short = engine.transcribe(str(_SHORT), word_timestamps=False)
    assert short.text.strip(), "expected a non-empty short transcript"

    # Longform on the SAME instance must not fail on the cache's inference tensors.
    long = engine.transcribe(str(_LONG), word_timestamps=False)
    assert isinstance(long, ASRResult)
    assert long.text.strip(), "expected a non-empty longform transcript"
    assert len(long.segments) > 1, "long audio should yield several segments"
