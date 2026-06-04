"""Silero VAD + chunking algorithm for longform recognition.

`merge_intervals_to_chunks` is a pure function, a verbatim port of the merge logic from
gigaam/vad_utils.py::segment_audio_file (_update_segments + loop). We only change the
source of speech intervals: pyannote → Silero.

silero/torch imports are lazy (inside functions) so that importing the module just for the
pure chunking function does not pull in the heavy stack. Silero weights are bundled in the
package (no network).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


def load_vad() -> object:
    """Load Silero VAD once (JIT from the package bundle; no network/cache needed).

    We return an opaque handle (passed to `speech_intervals`); the JIT stack is
    the same as GigaAM's (torch) — without onnxruntime, so as not to spawn thread pools
    on weak CPUs (e.g. ~4 cores; see the ADR in CLAUDE.md).
    """
    from silero_vad import load_silero_vad

    return load_silero_vad()


def speech_intervals(
    wav: Tensor, model: object, *, threshold: float, sampling_rate: int = 16000
) -> list[tuple[float, float]]:
    """Return speech intervals `(start, end)` in seconds via Silero.

    `wav` is float32 mono 16kHz (the whole signal; memory peak at this stage).
    The format is identical to pyannote: a list of `(start, end)` seconds.
    """
    from silero_vad import get_speech_timestamps

    timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sampling_rate,
        threshold=threshold,
        return_seconds=True,
    )
    return [(float(ts["start"]), float(ts["end"])) for ts in timestamps]


def merge_intervals_to_chunks(
    intervals: list[tuple[float, float]],
    audio_duration: float,
    *,
    min_duration: float,
    max_duration: float,
    strict_limit: float,
    new_chunk_threshold: float,
) -> list[tuple[float, float]]:
    """Merge speech intervals into chunks and return their boundaries (seconds).

    A port of `segment_audio_file` without waveform slicing: a pure function of the intervals and
    parameters (the engine does the slicing). Chunks longer than `strict_limit`
    are cut into equal parts (`int(d/strict_limit)+1`), as in upstream.
    """
    boundaries: list[tuple[float, float]] = []
    curr_duration = 0.0
    curr_start = 0.0
    curr_end = 0.0

    def _flush(curr_start: float, curr_end: float, curr_duration: float) -> None:
        if curr_duration > strict_limit:
            max_segments = int(curr_duration / strict_limit) + 1
            segment_duration = curr_duration / max_segments
            curr_end = curr_start + segment_duration
            for _ in range(max_segments - 1):
                boundaries.append((curr_start, curr_end))
                curr_start = curr_end
                curr_end += segment_duration
        boundaries.append((curr_start, curr_end))

    for raw_start, raw_end in intervals:
        start = max(0.0, raw_start)
        end = min(audio_duration, raw_end)
        if curr_duration == 0.0:
            curr_start = start
        elif curr_duration > new_chunk_threshold and (
            curr_duration + (end - curr_end) > max_duration or curr_duration > min_duration
        ):
            _flush(curr_start, curr_end, curr_duration)
            curr_start = start
        curr_end = end
        curr_duration = curr_end - curr_start

    if curr_duration > new_chunk_threshold:
        _flush(curr_start, curr_end, curr_duration)

    return boundaries
