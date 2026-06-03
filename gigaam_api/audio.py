"""Работа с аудио: probe длительности (ffprobe) и декод в int16 16k mono (ffmpeg).

Короткий путь распознавания делегирует декод самому gigaam (`model.transcribe`
внутри зовёт ffmpeg, см. gigaam/preprocess.py::load_audio). Longform-путь декодит
сам через `decode_to_int16_16k_mono` (int16 экономит память на длинных файлах).

torch импортируется **лениво** внутри `decode_to_int16_16k_mono`: модуль остаётся
torch-free, чтобы импорт audio.py из HTTP-слоя не тянул torch (master §4.3).
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
    """Не удалось прочитать/декодировать аудио (битый/неподдерживаемый файл)."""


class AudioToolNotFoundError(Exception):
    """ffprobe/ffmpeg не найден в PATH — серверная проблема окружения (→ 500)."""


def probe_duration(path: str) -> float:
    """Вернуть длительность аудио в секундах через `ffprobe`.

    ffprobe надёжно определяет длительность для любых форматов, поддерживаемых
    ffmpeg. Любой сбой (нет файла, битый ввод, неизвестная длительность) →
    AudioDecodeError, чтобы не пробрасывать сырые ошибки subprocess.
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
        raise AudioToolNotFoundError("ffprobe не найден в PATH") from exc
    except subprocess.CalledProcessError as exc:
        logger.warning("ffprobe не смог прочитать аудио: %s", path)
        raise AudioDecodeError(f"не удалось прочитать аудио: {path}") from exc

    raw = proc.stdout.strip()
    try:
        duration = float(raw)
    except ValueError as exc:
        raise AudioDecodeError(f"ffprobe вернул некорректную длительность: {raw!r}") from exc

    logger.debug("probe_duration %s -> %.3fs", path, duration)
    return duration


def decode_to_int16_16k_mono(path: str) -> Tensor:
    """Декодировать аудио в 1-D **int16** torch.Tensor (16kHz mono) через ffmpeg.

    Как gigaam `load_audio`, но возвращаем int16 (а не float): на длинных файлах это
    вдвое экономит память (~1.15 ГБ/10ч против 2.3 ГБ во float). Во float конвертируем
    срез-по-чанку при батчинге. Тензор read-only (делит память с буфером ffmpeg) —
    инференс делает копии (`.float()`), сам буфер не мутируем. См. master §11.
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
        raise AudioToolNotFoundError("ffmpeg не найден в PATH") from exc
    except subprocess.CalledProcessError as exc:
        logger.warning("ffmpeg не смог декодировать аудио: %s", path)
        raise AudioDecodeError(f"не удалось декодировать аудио: {path}") from exc

    with warnings.catch_warnings():
        # torch.frombuffer на bytes даёт read-only тензор → UserWarning, как в gigaam.
        warnings.simplefilter("ignore", category=UserWarning)
        wav: Tensor = torch.frombuffer(raw, dtype=torch.int16)

    logger.debug(
        "decode_to_int16_16k_mono %s -> %d samples (%.1f MB int16)",
        path,
        wav.numel(),
        wav.numel() * 2 / 1024 / 1024,
    )
    return wav
