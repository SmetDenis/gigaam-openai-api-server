"""Юнит-тесты audio.py: probe_duration и AudioDecodeError.

WAV генерируем stdlib-модулем `wave` (без ffmpeg/сети) — детерминированно.
ffprobe читает такой RIFF-WAV без проблем.
"""

import wave
from pathlib import Path

import pytest

from gigaam_api.audio import AudioDecodeError, probe_duration


def _write_silence_wav(path: Path, seconds: float, sample_rate: int = 16000) -> None:
    n_frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


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
