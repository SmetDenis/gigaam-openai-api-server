"""Контракт ASR-движка: типы результата, Protocol и engine-level исключения.

Типы намеренно не зависят от gigaam — HTTP- и format-слой импортируют их, не зная
про конкретный backend инференса (см. docs/specs/00-master.md §4.3, D3).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypedDict, runtime_checkable


class EngineInfo(TypedDict):
    """Снимок состояния движка для GET /health."""

    model: str
    device: str
    loaded: bool


@dataclass(frozen=True)
class WordTS:
    """Слово с таймстемпами (секунды от начала аудио)."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class SegmentTS:
    """Сегмент транскрипта; `words` заполняется при word-level granularity."""

    text: str
    start: float
    end: float
    words: list[WordTS] | None = None


@dataclass(frozen=True)
class ASRResult:
    """Результат распознавания: полный текст, длительность и сегменты.

    Для короткого аудио (≤25с) `segments` — один сегмент `[0, duration]`.
    """

    text: str
    duration: float
    segments: list[SegmentTS]


class AudioTooLongError(Exception):
    """Аудио длиннее лимита MAX_AUDIO_SECONDS (longform-порог 25с обрабатывается внутри движка)."""


class InferenceCancelledError(Exception):
    """Инференс прерван по запросу (клиент отключился между батчами longform)."""


@runtime_checkable
class ASREngine(Protocol):
    """Контракт инференса; не знает про HTTP.

    Реализация (`GigaAMEngine`) держит загруженную модель и сериализуется снаружи
    (этап 04). longform-метод добавит этап 03. `runtime_checkable` нужен, чтобы
    /health мог сузить тип `app.state.engine` через isinstance, не импортируя gigaam.
    """

    model_name: str
    device: str

    def transcribe(
        self,
        wav_path: str,
        *,
        word_timestamps: bool,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ASRResult: ...

    def info(self) -> EngineInfo: ...
