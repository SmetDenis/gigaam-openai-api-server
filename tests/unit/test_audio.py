"""Юнит-тесты audio.py: probe_duration и AudioDecodeError.

WAV генерируем stdlib-модулем `wave` (без ffmpeg/сети) — детерминированно.
ffprobe читает такой RIFF-WAV без проблем.
"""

import wave
from pathlib import Path

import pytest
import torch

from gigaam_api.audio import AudioDecodeError, decode_to_int16_16k_mono, probe_duration


def _write_silence_wav(
    path: Path, seconds: float, sample_rate: int = 16000, channels: int = 1
) -> None:
    n_frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * channels * n_frames)


def test_probe_duration_returns_seconds(tmp_path: Path) -> None:
    wav = tmp_path / "tone.wav"
    _write_silence_wav(wav, seconds=1.0)
    assert abs(probe_duration(str(wav)) - 1.0) < 0.05


def test_probe_duration_raises_on_corrupt_input(tmp_path: Path) -> None:
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"this is not audio")
    with pytest.raises(AudioDecodeError):
        probe_duration(str(bad))


def test_probe_duration_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AudioDecodeError):
        probe_duration(str(tmp_path / "does-not-exist.wav"))


def test_decode_returns_int16_mono_16k_tensor(tmp_path: Path) -> None:
    # 8kHz stereo silence → downmix(2→1) + resample(8k→16k): int16, 1-D, ~2× сэмплов.
    src = tmp_path / "stereo8k.wav"
    _write_silence_wav(src, seconds=0.5, sample_rate=8000, channels=2)
    wav = decode_to_int16_16k_mono(str(src))
    assert isinstance(wav, torch.Tensor)
    assert wav.dtype == torch.int16
    assert wav.ndim == 1
    assert abs(wav.numel() - int(0.5 * 16000)) < 200  # 0.5с @ 16kHz, допуск ресемплера
    assert int(wav.abs().max()) == 0  # тишина → нули


def test_decode_raises_on_corrupt_input(tmp_path: Path) -> None:
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"this is not audio")
    with pytest.raises(AudioDecodeError):
        decode_to_int16_16k_mono(str(bad))


def test_decode_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AudioDecodeError):
        decode_to_int16_16k_mono(str(tmp_path / "does-not-exist.wav"))
