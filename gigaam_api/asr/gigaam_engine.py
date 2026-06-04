"""PyTorch implementation of ASREngine on top of gigaam.load_model (short + longform).

We resolve the device (`auto`→cuda→mps→cpu) ourselves: gigaam's built-in `auto` knows
only cuda→cpu, without MPS.

Duration-based routing inside the engine: ≤25s — the short path (delegate to
`model.transcribe`); otherwise — our own longform loop (Silero VAD + chunking +
batched `model.forward`/`model._decode`), without pyannote.
"""

import logging
import time
from collections.abc import Callable, Iterator
from itertools import islice

import gigaam
import torch
from torch import Tensor

from gigaam_api.asr.engine import (
    ASRResult,
    AudioTooLongError,
    EngineInfo,
    InferenceCancelledError,
    SegmentTS,
    WordTS,
)
from gigaam_api.asr.vad import load_vad, merge_intervals_to_chunks, speech_intervals
from gigaam_api.audio import AudioDecodeError, decode_to_int16_16k_mono, probe_duration
from gigaam_api.config import Settings

logger = logging.getLogger(__name__)

# gigaam LONGFORM_THRESHOLD = 25 * 16000 samples = exactly 25s at 16kHz.
SHORT_MAX_SECONDS = 25.0
SAMPLE_RATE = 16000


def _resolve_device(setting: str) -> str:
    """Resolve DEVICE: `auto` → cuda→mps→cpu; an explicit value is returned as is."""
    if setting != "auto":
        return setting
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _collate(wavs: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Stack waveform slices into a batch (zero-padded) + lengths. Port of gigaam collate."""
    lengths = torch.tensor([w.shape[-1] for w in wavs], dtype=torch.long)
    max_len = int(lengths.max().item())
    batch = torch.zeros(len(wavs), max_len, dtype=wavs[0].dtype)
    for i, wav in enumerate(wavs):
        batch[i, : wav.shape[-1]] = wav
    return batch, lengths


class GigaAMEngine:
    """Wrapper over the gigaam model: load once + short/longform transcription."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.model_name = settings.MODEL
        self.device = _resolve_device(settings.DEVICE)

        if self.device == "cpu":
            torch.set_num_threads(settings.NUM_THREADS)
        if self.device == "mps":
            logger.warning(
                "DEVICE=mps: on MPS errors set PYTORCH_ENABLE_MPS_FALLBACK=1 "
                "(GigaAM on MPS is not tested upstream)"
            )
        if settings.QUANTIZE_INT8:
            logger.warning("QUANTIZE_INT8=true is ignored: int8 will be implemented in stage 07")

        t0 = time.perf_counter()
        self._model = gigaam.load_model(
            settings.MODEL,
            device=self.device,
            download_root=str(settings.MODELS_DIR),
            fp16_encoder=True,  # on cpu — no-op (gigaam skips half() for cpu)
        )
        self._vad = load_vad()  # Silero JIT from the bundle (no network), once
        logger.info(
            "model loaded: %s device=%s cache=%s in %.1fs",
            self.model_name,
            self.device,
            settings.MODELS_DIR,
            time.perf_counter() - t0,
        )

    def transcribe(
        self,
        wav_path: str,
        *,
        word_timestamps: bool,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ASRResult:
        duration = probe_duration(wav_path)
        max_seconds = self._settings.MAX_AUDIO_SECONDS
        logger.debug(
            "transcribe wav=%s duration=%.3fs word_timestamps=%s",
            wav_path,
            duration,
            word_timestamps,
        )
        if max_seconds > 0 and duration > max_seconds:
            raise AudioTooLongError(
                f"audio {duration:.1f}s exceeds the limit MAX_AUDIO_SECONDS={max_seconds}s"
            )
        if duration <= SHORT_MAX_SECONDS:
            return self._transcribe_short(
                wav_path, duration, word_timestamps=word_timestamps, cancel_check=cancel_check
            )
        return self._transcribe_longform(
            wav_path, word_timestamps=word_timestamps, cancel_check=cancel_check
        )

    def iter_segments(
        self,
        wav_path: str,
        *,
        word_timestamps: bool,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Iterator[SegmentTS]:
        """Segments as they become ready (for SSE streaming).

        Routing as in `transcribe`: ≤25s — a single segment (delegate to the short path);
        otherwise — longform chunks, each yielded right after its batch is decoded.
        The `MAX_AUDIO_SECONDS` limit is checked on the first iteration (a safety net —
        the handler validates duration earlier, before the stream starts)."""
        duration = probe_duration(wav_path)
        max_seconds = self._settings.MAX_AUDIO_SECONDS
        logger.debug(
            "iter_segments wav=%s duration=%.3fs word_timestamps=%s",
            wav_path,
            duration,
            word_timestamps,
        )
        if max_seconds > 0 and duration > max_seconds:
            raise AudioTooLongError(
                f"audio {duration:.1f}s exceeds the limit MAX_AUDIO_SECONDS={max_seconds}s"
            )
        if duration <= SHORT_MAX_SECONDS:
            result = self._transcribe_short(
                wav_path, duration, word_timestamps=word_timestamps, cancel_check=cancel_check
            )
            yield from result.segments
            return
        int16, _duration, chunks = self._prepare_longform(wav_path)
        yield from self._iter_chunks(
            int16, chunks, word_timestamps=word_timestamps, cancel_check=cancel_check
        )

    def _transcribe_short(
        self,
        wav_path: str,
        duration: float,
        *,
        word_timestamps: bool,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ASRResult:
        """Short audio ≤25s: delegate decode+inference to gigaam itself.

        cancel_check is not checked on the short path (fast, non-cancellable); the parameter
        is passed only on the fallback to longform near the 25s boundary.
        """
        t0 = time.perf_counter()
        try:
            result = self._model.transcribe(wav_path, word_timestamps=word_timestamps)
        except ValueError as exc:
            # Near the 25s boundary gigaam measures length by samples, probe — by source seconds.
            # If gigaam considered the audio long — that's not an error, but longform.
            if "too long" in str(exc).lower():
                logger.debug("gigaam considered the audio long near the 25s boundary → longform")
                return self._transcribe_longform(
                    wav_path, word_timestamps=word_timestamps, cancel_check=cancel_check
                )
            raise
        except RuntimeError as exc:
            # gigaam/preprocess.load_audio raises RuntimeError("Failed to load audio").
            if "failed to load audio" in str(exc).lower():
                raise AudioDecodeError(f"failed to decode audio: {wav_path}") from exc
            raise

        elapsed = time.perf_counter() - t0
        text: str = result.text
        words = (
            [WordTS(text=w.text, start=w.start, end=w.end) for w in result.words]
            if result.words is not None
            else None
        )
        logger.info(
            "transcribed (short) wav=%s in %.2fs rtf=%.2f chars=%d words=%s",
            wav_path,
            elapsed,
            elapsed / duration if duration > 0 else 0.0,
            len(text),
            len(words) if words is not None else "n/a",
        )
        segment = SegmentTS(text=text, start=0.0, end=duration, words=words)
        return ASRResult(text=text, duration=duration, segments=[segment])

    def _prepare_longform(self, wav_path: str) -> tuple[Tensor, float, list[tuple[float, float]]]:
        """Decode int16 + Silero VAD + chunking. Returns (int16 signal, duration, boundaries).

        Memory peak — at the VAD stage (the whole signal in float); freed immediately.
        """
        int16 = decode_to_int16_16k_mono(wav_path)
        duration = int16.numel() / SAMPLE_RATE

        wav_f32 = int16.float() / 32768.0
        intervals = speech_intervals(wav_f32, self._vad, threshold=self._settings.VAD_THRESHOLD)
        del wav_f32

        chunks = merge_intervals_to_chunks(
            intervals,
            duration,
            min_duration=self._settings.VAD_MIN_DURATION,
            max_duration=self._settings.VAD_MAX_DURATION,
            strict_limit=self._settings.VAD_STRICT_LIMIT,
            new_chunk_threshold=self._settings.VAD_NEW_CHUNK_THRESHOLD,
        )
        speech_total = sum(end - start for start, end in intervals)
        logger.info(
            "VAD wav=%s intervals=%d speech=%.1fs chunks=%d",
            wav_path,
            len(intervals),
            speech_total,
            len(chunks),
        )
        logger.debug("chunk boundaries: %s", chunks)
        return int16, duration, chunks

    def _iter_chunks(
        self,
        int16: Tensor,
        chunks: list[tuple[float, float]],
        *,
        word_timestamps: bool,
        cancel_check: Callable[[], bool] | None,
    ) -> Iterator[SegmentTS]:
        """Batched inference over chunk boundaries; yield a segment once its batch is decoded.

        Cancellation is cooperative — checked at the start of each batch (between
        batches). The shared longform core: used by both the synchronous
        `_transcribe_longform` and the `iter_segments` stream.
        """
        if not chunks:
            return
        batch_size = self._settings.BATCH_SIZE
        n_batches = (len(chunks) + batch_size - 1) // batch_size
        chunks_iter = iter(chunks)
        for batch_idx in range(n_batches):
            if cancel_check is not None and cancel_check():
                logger.info(
                    "longform cancelled on batch %d/%d (cancel_check returned True)",
                    batch_idx + 1,
                    n_batches,
                )
                raise InferenceCancelledError("inference aborted on cancellation request")
            batch = list(islice(chunks_iter, batch_size))
            tb = time.perf_counter()
            slices = [
                int16[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)].float() / 32768.0
                for start, end in batch
            ]
            wav_pad, wav_lens = _collate(slices)
            wav_pad = wav_pad.to(self.device).to(self._model._dtype)
            wav_lens = wav_lens.to(self.device)
            # gigaam.transcribe/transcribe_longform are wrapped in @torch.inference_mode();
            # our longform calls forward/_decode directly → we wrap it ourselves. Without this
            # autograd is on, and the encoder's rotary cos/sin cache, created by the short path
            # (under inference_mode) as inference tensors, breaks forward:
            # "Inference tensors cannot be saved for backward" (short→long on a single engine).
            with torch.inference_mode():
                encoded, encoded_len = self._model.forward(wav_pad, wav_lens)
                decoded = self._model._decode(encoded, encoded_len, wav_lens, word_timestamps)
            for (text, words), (seg_start, seg_end) in zip(decoded, batch, strict=True):
                seg_words = None
                if word_timestamps:
                    seg_words = [
                        WordTS(
                            text=w.text,
                            start=round(w.start + seg_start, 3),
                            end=round(w.end + seg_start, 3),
                        )
                        for w in (words or [])
                    ]
                yield SegmentTS(text=text, start=seg_start, end=seg_end, words=seg_words)
            logger.info(
                "longform batch %d/%d samples=%d in %.2fs",
                batch_idx + 1,
                n_batches,
                int(wav_pad.shape[0] * wav_pad.shape[1]),
                time.perf_counter() - tb,
            )

    def _transcribe_longform(
        self,
        wav_path: str,
        *,
        word_timestamps: bool,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ASRResult:
        """Long audio >25s (sync): VAD → chunking → batched inference → join into ASRResult."""
        t0 = time.perf_counter()
        int16, duration, chunks = self._prepare_longform(wav_path)
        segments = list(
            self._iter_chunks(
                int16, chunks, word_timestamps=word_timestamps, cancel_check=cancel_check
            )
        )
        full_text = " ".join(seg.text for seg in segments)
        elapsed = time.perf_counter() - t0
        n_words = sum(len(seg.words) for seg in segments if seg.words is not None)
        logger.info(
            "transcribed (longform) wav=%s in %.2fs rtf=%.2f segments=%d words=%d",
            wav_path,
            elapsed,
            elapsed / duration if duration > 0 else 0.0,
            len(segments),
            n_words,
        )
        return ASRResult(text=full_text, duration=duration, segments=segments)

    def info(self) -> EngineInfo:
        return {"model": self.model_name, "device": self.device, "loaded": True}
