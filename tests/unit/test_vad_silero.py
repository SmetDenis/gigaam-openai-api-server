"""Unit tests for the Silero VAD wrappers: load_vad + speech_intervals.

They use the real Silero JIT model (bundled in the silero-vad package, no network) —
so this is an honest test without mocks. The populated path (real speech → non-empty
intervals) is covered by the longform integration test.
"""

import torch

from gigaam_api.asr.vad import load_vad, speech_intervals


def test_load_vad_returns_usable_model() -> None:
    model = load_vad()
    assert model is not None


def test_speech_intervals_on_silence_returns_empty_list() -> None:
    model = load_vad()
    wav = torch.zeros(16000 * 2)  # 2s of silence, float32 16kHz → no speech
    intervals = speech_intervals(wav, model, threshold=0.5)
    assert intervals == []


def test_speech_intervals_returns_list_of_float_pairs() -> None:
    # The return contract does not depend on whether speech is present: list[(float, float)].
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
