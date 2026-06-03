"""PyTorch-реализация ASREngine поверх gigaam.load_model (короткое аудио ≤25с).

Резолв device (`auto`→cuda→mps→cpu) делаем сами: встроенный `auto` gigaam знает
только cuda→cpu, без MPS (docs/specs/00-master.md, ADR). Longform — этап 03.
"""

import logging
import time

import gigaam
import torch

from gigaam_api.asr.engine import (
    ASRResult,
    AudioTooLongError,
    EngineInfo,
    SegmentTS,
    WordTS,
)
from gigaam_api.audio import AudioDecodeError, probe_duration
from gigaam_api.config import Settings

logger = logging.getLogger(__name__)

# gigaam LONGFORM_THRESHOLD = 25 * 16000 сэмплов = ровно 25с при 16kHz.
SHORT_MAX_SECONDS = 25.0


def _resolve_device(setting: str) -> str:
    """Резолв DEVICE: `auto` → cuda→mps→cpu; явное значение возвращаем как есть."""
    if setting != "auto":
        return setting
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GigaAMEngine:
    """Обёртка над gigaam-моделью: загрузка один раз + транскрипция коротких аудио."""

    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.MODEL
        self.device = _resolve_device(settings.DEVICE)

        if self.device == "cpu":
            torch.set_num_threads(settings.NUM_THREADS)
        if self.device == "mps":
            logger.warning(
                "DEVICE=mps: при ошибках MPS установите PYTORCH_ENABLE_MPS_FALLBACK=1 "
                "(GigaAM на MPS upstream не тестируют; см. master §12)"
            )
        if settings.QUANTIZE_INT8:
            logger.warning("QUANTIZE_INT8=true игнорируется: int8 будет реализован на этапе 07")

        t0 = time.perf_counter()
        self._model = gigaam.load_model(
            settings.MODEL,
            device=self.device,
            download_root=str(settings.MODELS_DIR),
            fp16_encoder=True,  # на cpu — no-op (gigaam пропускает half() для cpu)
        )
        logger.info(
            "model loaded: %s device=%s cache=%s in %.1fs",
            self.model_name,
            self.device,
            settings.MODELS_DIR,
            time.perf_counter() - t0,
        )

    def transcribe(self, wav_path: str, *, word_timestamps: bool) -> ASRResult:
        duration = probe_duration(wav_path)
        logger.debug(
            "transcribe wav=%s duration=%.3fs word_timestamps=%s",
            wav_path,
            duration,
            word_timestamps,
        )
        if duration > SHORT_MAX_SECONDS:
            raise AudioTooLongError(
                f"аудио {duration:.1f}с длиннее {SHORT_MAX_SECONDS:.0f}с; "
                "longform появится на этапе 03"
            )

        t0 = time.perf_counter()
        try:
            result = self._model.transcribe(wav_path, word_timestamps=word_timestamps)
        except ValueError as exc:
            # Страховка у границы 25с: gigaam меряет длину по сэмплам, probe — секунды
            # исходника. Переводим только «too long»; прочие ValueError пробрасываем.
            if "too long" in str(exc).lower():
                raise AudioTooLongError(
                    f"аудио длиннее {SHORT_MAX_SECONDS:.0f}с (определено gigaam); "
                    "longform — этап 03"
                ) from exc
            raise
        except RuntimeError as exc:
            # gigaam/preprocess.load_audio бросает RuntimeError("Failed to load audio").
            if "failed to load audio" in str(exc).lower():
                raise AudioDecodeError(f"не удалось декодировать аудио: {wav_path}") from exc
            raise

        elapsed = time.perf_counter() - t0
        text: str = result.text
        words = (
            [WordTS(text=w.text, start=w.start, end=w.end) for w in result.words]
            if result.words is not None
            else None
        )
        logger.info(
            "transcribed wav=%s in %.2fs rtf=%.2f chars=%d words=%s",
            wav_path,
            elapsed,
            elapsed / duration if duration > 0 else 0.0,
            len(text),
            len(words) if words is not None else "n/a",
        )
        segment = SegmentTS(text=text, start=0.0, end=duration, words=words)
        return ASRResult(text=text, duration=duration, segments=[segment])

    def info(self) -> EngineInfo:
        return {"model": self.model_name, "device": self.device, "loaded": True}
