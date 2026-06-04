"""Pure functions that render ASRResult into OpenAI formats. They know nothing about HTTP/model.

verbose_json fields not available from GigaAM are filled best-effort:
tokens=[], temperature/avg_logprob/no_speech_prob=0.0, seek=0; compression_ratio
is computed honestly (cheap). Time format: SRT — comma, VTT — dot.
"""

import zlib
from typing import Any

from gigaam_api.asr.engine import ASRResult
from gigaam_api.schemas import VerboseSegment, VerboseTranscription, VerboseWord

_TASK = "transcribe"
_LANGUAGE = "russian"


def to_json(result: ASRResult) -> dict[str, str]:
    """Render into the `response_format=json` format (text only)."""
    return {"text": result.text}


def to_text(result: ASRResult) -> str:
    """Render into the `response_format=text` format (plain text)."""
    return result.text


def _compression_ratio(text: str) -> float:
    """len(bytes)/len(zlib.compress(bytes)) — as in Whisper (bytes on both sides).

    Empty text → 0.0 (no questionable division). For short segments the value may
    be <1.0 (the zlib header is longer than the input) — this is correct and matches Whisper.
    """
    if not text:
        return 0.0
    encoded = text.encode("utf-8")
    return len(encoded) / len(zlib.compress(encoded))


def to_verbose_json(result: ASRResult, *, granularities: set[str]) -> dict[str, Any]:
    """Render into the `response_format=verbose_json` format.

    `granularities` controls the presence of the `segments` and `words` keys:
    - {"segment"} — segments only
    - {"word"} — words only
    - {"segment", "word"} — both keys
    Missing keys are excluded via `exclude_none=True`.
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
    """Seconds → `HH:MM:SS{sep}mmm` (sep=',' for SRT, '.' for VTT). Negative → 0."""
    millis_total: int = round(max(0.0, seconds) * 1000.0)
    hours, millis_total = divmod(millis_total, 3_600_000)
    minutes, millis_total = divmod(millis_total, 60_000)
    secs, millis = divmod(millis_total, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def to_srt(result: ASRResult) -> str:
    """Render into the `response_format=srt` format (SubRip Subtitle)."""
    out = ""
    for index, seg in enumerate(result.segments, start=1):
        start = _format_ts(seg.start, sep=",")
        end = _format_ts(seg.end, sep=",")
        out += f"{index}\n{start} --> {end}\n{seg.text}\n\n"
    return out


def to_vtt(result: ASRResult) -> str:
    """Render into the `response_format=vtt` format (WebVTT).

    The WEBVTT header is always present.
    """
    out = "WEBVTT\n\n"
    for seg in result.segments:
        start = _format_ts(seg.start, sep=".")
        end = _format_ts(seg.end, sep=".")
        out += f"{start} --> {end}\n{seg.text}\n\n"
    return out
