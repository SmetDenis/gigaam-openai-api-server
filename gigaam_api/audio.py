"""Работа с аудио: probe длительности через ffprobe.

Короткий путь распознавания делегирует декод самому gigaam (`model.transcribe`
внутри зовёт ffmpeg, см. gigaam/preprocess.py::load_audio). Чанковая загрузка
(`decode_to_pcm16`) понадобится для longform — её добавит этап 03.
"""

import logging
import subprocess

logger = logging.getLogger(__name__)


class AudioDecodeError(Exception):
    """Не удалось прочитать/декодировать аудио (ffprobe/ffmpeg-сбой, битый файл)."""


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
        raise AudioDecodeError("ffprobe не найден в PATH") from exc
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
