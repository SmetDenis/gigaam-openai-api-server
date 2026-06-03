"""Регрессия: на ОДНОМ инстансе движка короткий запрос не должен ломать longform.

Воспроизводит баг живого сервиса: `gigaam.transcribe` (короткий путь) обёрнут в
`@torch.inference_mode()` и кэширует rotary `cos`/`sin` энкодера как inference-тензоры;
наш longform зовёт `forward`/`_decode` напрямую и без собственной `inference_mode`-обёртки
падал на этом кэше с `RuntimeError: Inference tensors cannot be saved for backward`.

Другие integration-тесты используют ОТДЕЛЬНЫЙ инстанс на файл → баг не ловили. Здесь
намеренно один движок и порядок short→long (как в живом сервисе с единственной моделью).
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
        pytest.skip("нет тест-сэмплов (short и/или long)")
    cache = tmp_path_factory.mktemp("models")
    settings = Settings(MODEL="v3_ctc", DEVICE="cpu", MODELS_DIR=cache)
    try:
        return GigaAMEngine(settings)
    except Exception as exc:  # нет сети / CDN недоступен / веса не скачались
        pytest.skip(f"модель недоступна (нет сети/весов): {exc}")


def test_short_then_long_same_engine(engine: GigaAMEngine) -> None:
    # Короткий путь (gigaam.transcribe под inference_mode) «отравляет» кэш rotary cos/sin.
    short = engine.transcribe(str(_SHORT), word_timestamps=False)
    assert short.text.strip(), "ожидали непустой короткий транскрипт"

    # Longform на ТОМ ЖЕ инстансе не должен падать на inference-тензорах кэша.
    long = engine.transcribe(str(_LONG), word_timestamps=False)
    assert isinstance(long, ASRResult)
    assert long.text.strip(), "ожидали непустой longform-транскрипт"
    assert len(long.segments) > 1, "длинное аудио должно дать несколько сегментов"
