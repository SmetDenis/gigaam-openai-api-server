"""Unit tests for audio.py: probe_duration and AudioDecodeError.

We generate the WAV with the stdlib `wave` module (no ffmpeg/network) — deterministically.
ffprobe reads such a RIFF-WAV without issues.
"""

import wave
from pathlib import Path

import pytest
import torch

from gigaam_api.audio import (
    AudioDecodeError,
    AudioToolNotFoundError,
    decode_to_int16_16k_mono,
    probe_duration,
)


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


def test_probe_duration_raises_on_missing_audio_file(tmp_path: Path) -> None:
    with pytest.raises(AudioDecodeError):
        probe_duration(str(tmp_path / "does-not-exist.wav"))


def test_decode_returns_int16_mono_16k_tensor(tmp_path: Path) -> None:
    # 8kHz stereo silence → downmix(2→1) + resample(8k→16k): int16, 1-D, ~2x samples.
    src = tmp_path / "stereo8k.wav"
    _write_silence_wav(src, seconds=0.5, sample_rate=8000, channels=2)
    wav = decode_to_int16_16k_mono(str(src))
    assert isinstance(wav, torch.Tensor)
    assert wav.dtype == torch.int16
    assert wav.ndim == 1
    assert abs(wav.numel() - int(0.5 * 16000)) < 200  # 0.5s @ 16kHz, resampler tolerance
    assert int(wav.abs().max()) == 0  # silence → zeros


def test_decode_raises_on_corrupt_input(tmp_path: Path) -> None:
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"this is not audio")
    with pytest.raises(AudioDecodeError):
        decode_to_int16_16k_mono(str(bad))


def test_decode_raises_on_missing_audio_file(tmp_path: Path) -> None:
    with pytest.raises(AudioDecodeError):
        decode_to_int16_16k_mono(str(tmp_path / "does-not-exist.wav"))


def test_probe_duration_tool_missing_raises_tool_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    def _no_ffprobe(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(subprocess, "run", _no_ffprobe)
    with pytest.raises(AudioToolNotFoundError):
        probe_duration("/tmp/x.wav")


def test_probe_duration_bad_input_raises_decode_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    def _broken(*args: object, **kwargs: object) -> object:
        raise subprocess.CalledProcessError(returncode=1, cmd="ffprobe")

    monkeypatch.setattr(subprocess, "run", _broken)
    with pytest.raises(AudioDecodeError):
        probe_duration("/tmp/x.wav")
