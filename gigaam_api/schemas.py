"""Pydantic-модели ответа OpenAI-совместимого API (единый источник формы)."""

from pydantic import BaseModel


class TranscriptionJSON(BaseModel):
    """Ответ `response_format=json`."""

    text: str


class VerboseWord(BaseModel):
    """Слово в `verbose_json` (ключ `word`, не `text`)."""

    word: str
    start: float
    end: float


class VerboseSegment(BaseModel):
    """Сегмент в `verbose_json`. Недоступные у GigaAM поля — best-effort (master §6.3)."""

    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class VerboseTranscription(BaseModel):
    """Ответ `response_format=verbose_json`. `segments`/`words` включаются по granularity."""

    task: str
    language: str
    duration: float
    text: str
    segments: list[VerboseSegment] | None = None
    words: list[VerboseWord] | None = None


class ModelObject(BaseModel):
    """Элемент `GET /v1/models` (master §6.5)."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "gigaam"


class ModelsList(BaseModel):
    """Ответ `GET /v1/models`."""

    object: str = "list"
    data: list[ModelObject]
