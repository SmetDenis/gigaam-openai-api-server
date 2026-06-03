"""Юнит-тесты обёрток Silero VAD: load_vad + speech_intervals.

Используют реальную JIT-модель Silero (бандлится в пакете silero-vad, без сети) —
поэтому это честный тест без моков. Populated-путь (реальная речь → непустые
интервалы) покрывается integration-тестом longform.
"""

import torch

from gigaam_api.asr.vad import load_vad, speech_intervals


def test_load_vad_returns_usable_model() -> None:
    model = load_vad()
    assert model is not None


def test_speech_intervals_on_silence_returns_empty_list() -> None:
    model = load_vad()
    wav = torch.zeros(16000 * 2)  # 2с тишины, float32 16kHz → речи нет
    intervals = speech_intervals(wav, model, threshold=0.5)
    assert intervals == []


def test_speech_intervals_returns_list_of_float_pairs() -> None:
    # Контракт возврата не зависит от наличия речи: list[(float, float)].
    model = load_vad()
    wav = torch.zeros(16000).float()
    intervals = speech_intervals(wav, model, threshold=0.5)
    assert isinstance(intervals, list)
    assert all(
        isinstance(pair, tuple)
        and len(pair) == 2
        and isinstance(pair[0], float)
        and isinstance(pair[1], float)
        for pair in intervals
    )
