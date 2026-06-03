"""Тесты Pydantic-схем ответа (форма OpenAI)."""

from gigaam_api.schemas import (
    ModelObject,
    ModelsList,
    VerboseSegment,
    VerboseTranscription,
    VerboseWord,
)


def test_models_list_shape() -> None:
    payload = ModelsList(data=[ModelObject(id="v3_ctc")]).model_dump()
    assert payload == {
        "object": "list",
        "data": [{"id": "v3_ctc", "object": "model", "created": 0, "owned_by": "gigaam"}],
    }


def test_verbose_transcription_omits_none_when_dumped() -> None:
    seg = VerboseSegment(
        id=0,
        seek=0,
        start=0.0,
        end=1.0,
        text="x",
        tokens=[],
        temperature=0.0,
        avg_logprob=0.0,
        compression_ratio=1.0,
        no_speech_prob=0.0,
    )
    model = VerboseTranscription(
        task="transcribe",
        language="russian",
        duration=1.0,
        text="x",
        segments=[seg],
        words=None,
    )
    dumped = model.model_dump(exclude_none=True)
    assert "segments" in dumped
    assert "words" not in dumped  # words=None → пропущено (granularity без word)
    _ = VerboseWord(word="x", start=0.0, end=1.0)  # модель словарного элемента существует
