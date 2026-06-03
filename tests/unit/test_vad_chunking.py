"""Юнит-тесты чистой функции чанкинга merge_intervals_to_chunks.

Алгоритм портирован дословно из gigaam/vad_utils.py::segment_audio_file
(_update_segments + цикл слияния). Здесь — самый важный юнит-тест этапа 03:
проверяем слияние, пороги min/max, разрезание сверхдлинных чанков и краевые случаи
на синтетических интервалах (без VAD и без модели).
"""

from itertools import pairwise

import pytest

from gigaam_api.asr.vad import merge_intervals_to_chunks

# Параметры по умолчанию (из config).
_PARAMS = {
    "min_duration": 15.0,
    "max_duration": 22.0,
    "strict_limit": 30.0,
    "new_chunk_threshold": 0.2,
}


def _chunks(
    intervals: list[tuple[float, float]], audio_duration: float
) -> list[tuple[float, float]]:
    return merge_intervals_to_chunks(intervals, audio_duration, **_PARAMS)


def test_empty_intervals_returns_empty() -> None:
    assert _chunks([], 10.0) == []


def test_single_interval_becomes_one_chunk() -> None:
    assert _chunks([(0.0, 10.0)], 10.0) == [(0.0, 10.0)]


def test_tiny_interval_below_threshold_is_dropped() -> None:
    # Длительность 0.1с < new_chunk_threshold (0.2) → финальный чанк не создаётся.
    assert _chunks([(0.0, 0.1)], 5.0) == []


def test_short_intervals_merge_into_single_chunk() -> None:
    # Несколько коротких речевых интервалов сливаются в один чанк (≥ min_duration),
    # пока не превышен max и не пройден min.
    intervals = [(0.0, 2.0), (3.0, 5.0), (6.0, 8.0), (9.0, 11.0), (12.0, 14.0), (15.0, 17.0)]
    chunks = _chunks(intervals, 20.0)
    assert chunks == [(0.0, 17.0)]
    assert chunks[0][1] - chunks[0][0] >= _PARAMS["min_duration"]


def test_new_chunk_starts_when_exceeding_max() -> None:
    # Первый интервал уже 16с (> min); следующий перевалит за max → flush, новый чанк.
    chunks = _chunks([(0.0, 16.0), (17.0, 33.0)], 33.0)
    assert chunks == [(0.0, 16.0), (17.0, 33.0)]
    # Границы монотонны и не пересекаются.
    for s, e in chunks:
        assert s <= e
    assert chunks[0][1] <= chunks[1][0]


def test_chunk_longer_than_strict_limit_is_split_into_equal_parts() -> None:
    # 70с > strict_limit(30) → int(70/30)+1 = 3 равные части ~23.33с, contiguous, сумма сохранена.
    chunks = _chunks([(0.0, 70.0)], 70.0)
    assert len(chunks) == 3
    expected = 70.0 / 3
    for s, e in chunks:
        assert (e - s) == pytest.approx(expected)
        assert (e - s) <= _PARAMS["strict_limit"]
    # Contiguity и полный охват [0, 70].
    assert chunks[0][0] == pytest.approx(0.0)
    assert chunks[-1][1] == pytest.approx(70.0)
    for prev, nxt in pairwise(chunks):
        assert prev[1] == pytest.approx(nxt[0])


def test_trailing_silence_does_not_extend_last_chunk() -> None:
    # Речь только в [0,10], дальше тишина до 30 → последний чанк кончается на 10, не на 30.
    assert _chunks([(0.0, 10.0)], 30.0) == [(0.0, 10.0)]


def test_interval_end_is_clamped_to_audio_duration() -> None:
    # Конец интервала за пределом длительности обрезается до audio_duration.
    chunks = _chunks([(0.0, 40.0)], 35.0)
    assert chunks[-1][1] == pytest.approx(35.0)
    # 35с > strict → разрез на 2 части по 17.5с.
    assert len(chunks) == 2
    assert chunks == pytest.approx([(0.0, 17.5), (17.5, 35.0)])


def test_negative_interval_start_is_clamped_to_zero() -> None:
    chunks = _chunks([(-1.0, 5.0)], 10.0)
    assert chunks == [(0.0, 5.0)]
