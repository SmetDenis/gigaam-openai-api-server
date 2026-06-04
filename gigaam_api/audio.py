"""Audio handling: duration probe (ffprobe) and decode to int16 16k mono (ffmpeg).

The short recognition path delegates decoding to gigaam itself (`model.transcribe`
calls ffmpeg internally, see gigaam/preprocess.py::load_audio). The longform path
decodes on its own via `decode_to_int16_16k_mono` (int16 saves memory on long files).

torch is imported **lazily** inside `decode_to_int16_16k_mono`: the module stays
torch-free so that importing audio.py from the HTTP layer does not pull in torch.
"""

from __future__ import annotations

import logging
import subprocess
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


class AudioDecodeError(Exception):
    """Failed to read/decode audio (broken/unsupported file)."""


class AudioToolNotFoundError(Exception):
    """ffprobe/ffmpeg not found in PATH — a server-side environment problem (→ 500)."""


def probe_duration(path: str) -> float:
    """Return the audio duration in seconds via `ffprobe`.

    ffprobe reliably determines the duration for any format supported by
    ffmpeg. Any failure (missing file, broken input, unknown duration) →
    AudioDecodeError, so that raw subprocess errors are not propagated.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise AudioToolNotFoundError("ffprobe not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        logger.warning("ffprobe could not read audio: %s", path)
        raise AudioDecodeError(f"failed to read audio: {path}") from exc

    raw = proc.stdout.strip()
    try:
        duration = float(raw)
    except ValueError as exc:
        raise AudioDecodeError(f"ffprobe returned an invalid duration: {raw!r}") from exc

    logger.debug("probe_duration %s -> %.3fs", path, duration)
    return duration


def decode_to_int16_16k_mono(path: str) -> Tensor:
    """Decode audio into a 1-D **int16** torch.Tensor (16kHz mono) via ffmpeg.

    Like gigaam `load_audio`, but we return int16 (not float): on long files this
    halves memory usage (~1.15 GB/10h versus 2.3 GB in float). We convert to float
    chunk-by-chunk during batching. The tensor is read-only (shares memory with the
    ffmpeg buffer) — inference makes copies (`.float()`), the buffer itself is not mutated.
    """
    import torch

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-",
    ]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError as exc:
        raise AudioToolNotFoundError("ffmpeg not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        logger.warning("ffmpeg could not decode audio: %s", path)
        raise AudioDecodeError(f"failed to decode audio: {path}") from exc

    with warnings.catch_warnings():
        # torch.frombuffer on bytes yields a read-only tensor → UserWarning, as in gigaam.
        warnings.simplefilter("ignore", category=UserWarning)
        wav: Tensor = torch.frombuffer(raw, dtype=torch.int16)

    logger.debug(
        "decode_to_int16_16k_mono %s -> %d samples (%.1f MB int16)",
        path,
        wav.numel(),
        wav.numel() * 2 / 1024 / 1024,
    )
    return wav
