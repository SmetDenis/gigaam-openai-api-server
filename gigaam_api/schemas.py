"""Pydantic response models for the OpenAI-compatible API (single source of shape)."""

from pydantic import BaseModel


class TranscriptionJSON(BaseModel):
    """Response for `response_format=json`."""

    text: str


class VerboseWord(BaseModel):
    """Word in `verbose_json` (key `word`, not `text`)."""

    word: str
    start: float
    end: float


class VerboseSegment(BaseModel):
    """Segment in `verbose_json`. Fields unavailable from GigaAM are best-effort."""

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
    """Response for `response_format=verbose_json`.

    `segments`/`words` are included per granularity.
    """

    task: str
    language: str
    duration: float
    text: str
    segments: list[VerboseSegment] | None = None
    words: list[VerboseWord] | None = None


class ModelObject(BaseModel):
    """Item of `GET /v1/models`."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "gigaam"


class ModelsList(BaseModel):
    """Response for `GET /v1/models`."""

    object: str = "list"
    data: list[ModelObject]
