"""POST /v1/audio/transcriptions — синхронный OpenAI-совместимый эндпоинт.

Upload пишем на диск чанками (master §11: не грузить весь файл в RAM); инференс —
через Runner (сериализация + backpressure). Отмена longform — кооперативная: watcher
на request.is_disconnected() ставит threading.Event, прокинутый как cancel_check.
stream=true пока обрабатывается синхронно (TODO: SSE — этап 05,
см. docs/specs/05-sse-streaming.md).
"""

import asyncio
import contextlib
import logging
import os
import tempfile
import threading
import uuid
from typing import Annotated, cast

import anyio
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from gigaam_api.asr import formats
from gigaam_api.asr.engine import ASREngine, ASRResult, AudioTooLongError, InferenceCancelledError
from gigaam_api.audio import probe_duration
from gigaam_api.auth import require_auth
from gigaam_api.config import Settings, get_settings
from gigaam_api.errors import PayloadTooLargeError, error_response
from gigaam_api.runner import Runner

logger = logging.getLogger(__name__)
router = APIRouter()

_VALID_FORMATS = {"json", "text", "verbose_json", "srt", "vtt"}
_CHUNK = 1 << 20  # 1 МиБ


async def _watch_disconnect(request: Request, event: threading.Event) -> None:
    """Опрашивать disconnect клиента; при отключении — выставить флаг отмены.

    Запускается ТОЛЬКО как дочерняя задача anyio task group и снимается через
    `cancel_scope.cancel()`. Использовать raw `asyncio.create_task` + `task.cancel()`
    НЕЛЬЗЯ: `Request.is_disconnected()` внутри держит `anyio.CancelScope`, и raw-отмена
    с ней конфликтует — задача не завершается, `await task` дедлочит (ADR этапа 04).
    """
    while not await request.is_disconnected():
        await anyio.sleep(1.0)
    event.set()


def _render(result: ASRResult, fmt: str, granularities: set[str], request_id: str) -> Response:
    """Отрендерить ASRResult в ответ по формату (fmt уже провалидирован вызывающим)."""
    logger.debug("request_id=%s render format=%s chars=%d", request_id, fmt, len(result.text))
    if fmt == "json":
        return JSONResponse(content=formats.to_json(result))
    if fmt == "verbose_json":
        return JSONResponse(content=formats.to_verbose_json(result, granularities=granularities))
    if fmt == "text":
        return PlainTextResponse(content=formats.to_text(result))
    if fmt == "srt":
        return PlainTextResponse(
            content=formats.to_srt(result), media_type="text/plain; charset=utf-8"
        )
    return PlainTextResponse(content=formats.to_vtt(result), media_type="text/vtt; charset=utf-8")


@router.post("/v1/audio/transcriptions", dependencies=[Depends(require_auth)])
async def transcriptions(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile, File()],
    model: Annotated[str | None, Form()] = None,
    response_format: Annotated[str | None, Form()] = None,
    timestamp_granularities: Annotated[
        list[str] | None, Form(alias="timestamp_granularities[]")
    ] = None,
    language: Annotated[str | None, Form()] = None,  # принимается и игнорируется (GigaAM — RU)
    stream: Annotated[bool, Form()] = False,
    prompt: Annotated[str | None, Form()] = None,  # принимается и игнорируется
    temperature: Annotated[float | None, Form()] = None,  # принимается и игнорируется
) -> Response:
    request_id = uuid.uuid4().hex
    fmt = response_format or settings.DEFAULT_RESPONSE_FORMAT
    logger.info(
        "request_id=%s model=%s response_format=%s file=%s stream=%s",
        request_id,
        model,
        fmt,
        file.filename,
        stream,
    )

    if model is not None and model not in settings.ALLOWED_MODELS:
        return error_response(
            400,
            f"model '{model}' is not available",
            "invalid_request_error",
            param="model",
        )
    if fmt not in _VALID_FORMATS:
        return error_response(
            400,
            f"response_format '{fmt}' is not supported",
            "invalid_request_error",
            param="response_format",
        )

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
            while chunk := await file.read(_CHUNK):
                size += len(chunk)
                if max_bytes > 0 and size > max_bytes:
                    raise PayloadTooLargeError(
                        f"upload exceeds MAX_UPLOAD_MB={settings.MAX_UPLOAD_MB}"
                    )
                tmp.write(chunk)
        logger.debug("request_id=%s tmp=%s size=%d", request_id, tmp_path, size)

        loop = asyncio.get_running_loop()
        duration = await loop.run_in_executor(None, probe_duration, tmp_path)
        if settings.MAX_AUDIO_SECONDS > 0 and duration > settings.MAX_AUDIO_SECONDS:
            raise AudioTooLongError(
                f"audio {duration:.1f}s exceeds MAX_AUDIO_SECONDS={settings.MAX_AUDIO_SECONDS}"
            )

        word_timestamps = bool(timestamp_granularities and "word" in timestamp_granularities)
        granularities = set(timestamp_granularities) if timestamp_granularities else {"segment"}
        logger.debug(
            "request_id=%s duration=%.2fs word_timestamps=%s",
            request_id,
            duration,
            word_timestamps,
        )

        engine = cast(ASREngine, request.app.state.engine)
        runner = cast(Runner, request.app.state.runner)

        # Watcher disconnect'а и инференс — в одной anyio task group: при отключении
        # клиента watcher ставит cancel_event, longform прерывается между батчами.
        # Исход инференса захватываем ВНУТРИ группы и диспетчеризуем СНАРУЖИ, чтобы
        # исключения (напр. QueueFullError→503) не оборачивались в ExceptionGroup.
        cancel_event = threading.Event()
        result: ASRResult | None = None
        inference_error: BaseException | None = None
        async with anyio.create_task_group() as tg:
            tg.start_soon(_watch_disconnect, request, cancel_event)
            try:
                result = await runner.run(
                    engine.transcribe,
                    tmp_path,
                    word_timestamps=word_timestamps,
                    cancel_check=cancel_event.is_set,
                )
            except Exception as exc:  # захватываем, чтобы снять watcher и не плодить ExceptionGroup
                inference_error = exc
            finally:
                tg.cancel_scope.cancel()

        if isinstance(inference_error, InferenceCancelledError):
            logger.info("request_id=%s инференс отменён (клиент отключился)", request_id)
            return Response(status_code=499)
        if inference_error is not None:
            raise inference_error
        assert result is not None
        return _render(result, fmt, granularities, request_id)
    finally:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
