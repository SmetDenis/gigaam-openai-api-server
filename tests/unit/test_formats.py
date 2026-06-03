"""Тесты чистых рендер-функций formats (json/text/verbose_json/srt/vtt) + форматтер времени."""

import pytest

from gigaam_api.asr import formats
from gigaam_api.asr.engine import ASRResult, SegmentTS, WordTS


def _result() -> ASRResult:
    return ASRResult(
        text="привет мир пока",
        duration=5.0,
        segments=[
            SegmentTS(
                text="привет мир",
                start=0.0,
                end=2.0,
                words=[WordTS("привет", 0.0, 0.5), WordTS("мир", 0.6, 1.0)],
            ),
            SegmentTS(text="пока", start=3.0, end=5.0, words=[WordTS("пока", 3.0, 3.4)]),
        ],
    )


def test_to_json() -> None:
    assert formats.to_json(_result()) == {"text": "привет мир пока"}


def test_to_text() -> None:
    assert formats.to_text(_result()) == "привет мир пока"


def test_verbose_json_default_segment_only() -> None:
    out = formats.to_verbose_json(_result(), granularities={"segment"})
    assert out["task"] == "transcribe"
    assert out["language"] == "russian"
    assert out["duration"] == 5.0
    assert "segments" in out
    assert "words" not in out  # без word-granularity слов нет
    seg0 = out["segments"][0]
    assert seg0["id"] == 0 and seg0["seek"] == 0
    assert seg0["tokens"] == [] and seg0["temperature"] == 0.0
    assert seg0["avg_logprob"] == 0.0 and seg0["no_speech_prob"] == 0.0
    assert seg0["compression_ratio"] > 0.5  # байт/байт как у Whisper (кириллица не должна занижать)


def test_verbose_json_word_granularity_adds_words() -> None:
    out = formats.to_verbose_json(_result(), granularities={"segment", "word"})
    assert [w["word"] for w in out["words"]] == ["привет", "мир", "пока"]
    assert out["words"][0] == {"word": "привет", "start": 0.0, "end": 0.5}


def test_verbose_json_word_only_omits_segments() -> None:
    out = formats.to_verbose_json(_result(), granularities={"word"})
    assert "segments" not in out
    assert "words" in out


def test_compression_ratio_empty_text_is_zero() -> None:
    empty = ASRResult(text="", duration=1.0, segments=[SegmentTS("", 0.0, 1.0, None)])
    out = formats.to_verbose_json(empty, granularities={"segment"})
    assert out["segments"][0]["compression_ratio"] == 0.0


def test_to_srt() -> None:
    srt = formats.to_srt(_result())
    assert srt == (
        "1\n00:00:00,000 --> 00:00:02,000\nпривет мир\n\n2\n00:00:03,000 --> 00:00:05,000\nпока\n\n"
    )


def test_to_vtt() -> None:
    vtt = formats.to_vtt(_result())
    assert vtt == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.000\nпривет мир\n\n"
        "00:00:03.000 --> 00:00:05.000\nпока\n\n"
    )


def test_srt_empty_segments_is_empty_string() -> None:
    empty = ASRResult(text="", duration=0.0, segments=[])
    assert formats.to_srt(empty) == ""


def test_vtt_empty_segments_is_header_only() -> None:
    empty = ASRResult(text="", duration=0.0, segments=[])
    assert formats.to_vtt(empty) == "WEBVTT\n\n"


@pytest.mark.parametrize(
    "seconds, sep, expected",
    [
        (0.0, ",", "00:00:00,000"),
        (3661.5, ",", "01:01:01,500"),  # 1ч 1м 1.5с
        (3661.5, ".", "01:01:01.500"),
        (59.999, ",", "00:00:59,999"),
        (-1.0, ",", "00:00:00,000"),  # отрицательное клампится в 0
    ],
)
def test_format_ts(seconds: float, sep: str, expected: str) -> None:
    assert formats._format_ts(seconds, sep=sep) == expected
