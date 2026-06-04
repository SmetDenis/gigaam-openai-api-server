"""ASR engine contract: result types, Protocol and engine-level exceptions.

The types deliberately do not depend on gigaam — the HTTP and format layers import them
without knowing about the concrete inference backend.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol, TypedDict, runtime_checkable


class EngineInfo(TypedDict):
    """Engine state snapshot for GET /health."""

    model: str
    device: str
    loaded: bool


@dataclass(frozen=True)
class WordTS:
    """A word with timestamps (seconds from the start of the audio)."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class SegmentTS:
    """A transcript segment; `words` is populated at word-level granularity."""

    text: str
    start: float
    end: float
    words: list[WordTS] | None = None


@dataclass(frozen=True)
class ASRResult:
    """Recognition result: the full text, duration and segments.

    For short audio (≤25s) `segments` is a single segment `[0, duration]`.
    """

    text: str
    duration: float
    segments: list[SegmentTS]


class AudioTooLongError(Exception):
    """Audio is longer than the MAX_AUDIO_SECONDS limit.

    The 25s longform threshold is handled inside the engine.
    """


class InferenceCancelledError(Exception):
    """Inference aborted on request (the client disconnected between longform batches)."""


@runtime_checkable
class ASREngine(Protocol):
    """Inference contract; knows nothing about HTTP.

    The implementation (`GigaAMEngine`) holds the loaded model and is serialized externally
    (stage 04). The longform method is added by stage 03. `runtime_checkable` is needed so that
    /health can narrow the type of `app.state.engine` via isinstance without importing gigaam.
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

    def iter_segments(
        self,
        wav_path: str,
        *,
        word_timestamps: bool,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Iterator[SegmentTS]: ...

    def info(self) -> EngineInfo: ...
