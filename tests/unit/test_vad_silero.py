"""Unit tests for the Silero VAD wrappers: load_vad + speech_intervals.

They use the real Silero JIT model (bundled in the silero-vad package, no network) —
so this is an honest test without mocks. The populated path (real speech → non-empty
intervals) is covered by the longform integration test.
"""

import pytest
import torch

from gigaam_api.asr.vad import load_vad, speech_intervals


def test_load_vad_returns_usable_model() -> None:
    model = load_vad()
    assert model is not None


def test_load_vad_restores_torch_thread_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_vad must not leak silero's import-time side effect.

    silero_vad runs torch.set_num_threads(1) at *import* (silero_vad/model.py, module level)
    — it fires once, on the first import in a process. load_vad must capture the thread
    count BEFORE importing silero and restore it after, otherwise it silently pins all
    downstream inference to one thread (NUM_THREADS ignored, ~3x slower on a 4-core CPU; see
    the ADR). We evict silero_vad from sys.modules so the module-level set(1) re-fires during
    load_vad — reproducing the production first-import scenario deterministically (capturing
    the count after the import, as a naive fix would, restores the already-clobbered 1).
    """
    import sys

    for name in [m for m in sys.modules if m == "silero_vad" or m.startswith("silero_vad.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    original = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        load_vad()
        assert torch.get_num_threads() == 4
    finally:
        torch.set_num_threads(original)


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
