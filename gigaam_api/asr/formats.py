"""Чистые функции рендера ASRResult в OpenAI-форматы. Не знают про HTTP/модель.

Недоступные у GigaAM поля verbose_json заполняются best-effort:
tokens=[], temperature/avg_logprob/no_speech_prob=0.0, seek=0; compression_ratio
считается честно (дёшево). Формат времени: SRT — запятая, VTT — точка.
"""

import zlib
from typing import Any

from gigaam_api.asr.engine import ASRResult
from gigaam_api.schemas import VerboseSegment, VerboseTranscription, VerboseWord

_TASK = "transcribe"
_LANGUAGE = "russian"


def to_json(result: ASRResult) -> dict[str, str]:
    """Рендер в формат `response_format=json` (только текст)."""
    return {"text": result.text}


def to_text(result: ASRResult) -> str:
    """Рендер в формат `response_format=text` (plain text)."""
    return result.text


def _compression_ratio(text: str) -> float:
    """len(bytes)/len(zlib.compress(bytes)) — как в Whisper (байты с обеих сторон).

    Пустой текст → 0.0 (без спорного деления). Для коротких сегментов значение может
    быть <1.0 (zlib-заголовок длиннее входа) — это корректно и совпадает с Whisper.
    """
    if not text:
        return 0.0
    encoded = text.encode("utf-8")
    return len(encoded) / len(zlib.compress(encoded))


def to_verbose_json(result: ASRResult, *, granularities: set[str]) -> dict[str, Any]:
    """Рендер в формат `response_format=verbose_json`.

    `granularities` управляет наличием ключей `segments` и `words`:
    - {"segment"} — только segments
    - {"word"} — только words
    - {"segment", "word"} — оба ключа
    Отсутствующие ключи исключаются через `exclude_none=True`.
    """
    segments = None
    if "segment" in granularities:
        segments = [
            VerboseSegment(
                id=i,
                seek=0,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                tokens=[],
                temperature=0.0,
                avg_logprob=0.0,
                compression_ratio=_compression_ratio(seg.text),
                no_speech_prob=0.0,
            )
            for i, seg in enumerate(result.segments)
        ]
    words = None
    if "word" in granularities:
        words = [
            VerboseWord(word=w.text, start=w.start, end=w.end)
            for seg in result.segments
            for w in (seg.words or [])
        ]
    model = VerboseTranscription(
        task=_TASK,
        language=_LANGUAGE,
        duration=result.duration,
        text=result.text,
        segments=segments,
        words=words,
    )
    return model.model_dump(exclude_none=True)


def _format_ts(seconds: float, *, sep: str) -> str:
    """Секунды → `HH:MM:SS{sep}mmm` (sep=',' для SRT, '.' для VTT). Отрицательное → 0."""
    millis_total: int = round(max(0.0, seconds) * 1000.0)
    hours, millis_total = divmod(millis_total, 3_600_000)
    minutes, millis_total = divmod(millis_total, 60_000)
    secs, millis = divmod(millis_total, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def to_srt(result: ASRResult) -> str:
    """Рендер в формат `response_format=srt` (SubRip Subtitle)."""
    out = ""
    for index, seg in enumerate(result.segments, start=1):
        start = _format_ts(seg.start, sep=",")
        end = _format_ts(seg.end, sep=",")
        out += f"{index}\n{start} --> {end}\n{seg.text}\n\n"
    return out


def to_vtt(result: ASRResult) -> str:
    """Рендер в формат `response_format=vtt` (WebVTT). Заголовок WEBVTT всегда присутствует."""
    out = "WEBVTT\n\n"
    for seg in result.segments:
        start = _format_ts(seg.start, sep=".")
        end = _format_ts(seg.end, sep=".")
        out += f"{start} --> {end}\n{seg.text}\n\n"
    return out
