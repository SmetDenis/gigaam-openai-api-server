"""OpenAI-совместимый формат ошибок + регистрация exception handlers.

Тело: {"error":{message,type,param,code}}. Маппинг кодов: битый файл→400,
лимиты→400/413, инструмент не найден→500, очередь→503, auth→401,
непредусмотренное→500. 415 не используется (OpenAI отдаёт 400).
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from gigaam_api.asr.engine import AudioTooLongError
from gigaam_api.audio import AudioDecodeError, AudioToolNotFoundError
from gigaam_api.runner import QueueFullError

logger = logging.getLogger(__name__)


class AuthError(Exception):
    """Отсутствует/неверен Bearer-ключ (→ 401)."""


class PayloadTooLargeError(Exception):
    """Загрузка превысила MAX_UPLOAD_MB (→ 413)."""


class OpenAIErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class OpenAIError(BaseModel):
    error: OpenAIErrorDetail


def error_response(
    status_code: int,
    message: str,
    type_: str,
    *,
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    """Собрать JSONResponse в OpenAI-формате ошибки."""
    body = OpenAIError(error=OpenAIErrorDetail(message=message, type=type_, param=param, code=code))
    return JSONResponse(status_code=status_code, content=body.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    """Зарегистрировать все обработчики на приложении."""

    @app.exception_handler(AudioDecodeError)
    async def _decode(request: Request, exc: AudioDecodeError) -> JSONResponse:
        return error_response(400, str(exc), "invalid_request_error", param="file")

    @app.exception_handler(AudioTooLongError)
    async def _too_long(request: Request, exc: AudioTooLongError) -> JSONResponse:
        return error_response(400, str(exc), "invalid_request_error", param="file")

    @app.exception_handler(PayloadTooLargeError)
    async def _too_large(request: Request, exc: PayloadTooLargeError) -> JSONResponse:
        return error_response(413, str(exc), "invalid_request_error", param="file")

    @app.exception_handler(AudioToolNotFoundError)
    async def _tool(request: Request, exc: AudioToolNotFoundError) -> JSONResponse:
        logger.exception("инструмент аудио недоступен: %s", exc)
        return error_response(500, str(exc), "api_error")

    @app.exception_handler(QueueFullError)
    async def _queue(request: Request, exc: QueueFullError) -> JSONResponse:
        return error_response(503, str(exc), "server_error")

    @app.exception_handler(AuthError)
    async def _auth(request: Request, exc: AuthError) -> JSONResponse:
        message = str(exc) or "Incorrect API key provided."
        return error_response(401, message, "invalid_request_error", code="invalid_api_key")

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        param = str(errors[0]["loc"][-1]) if errors and errors[0].get("loc") else None
        return error_response(
            400, "invalid request parameters", "invalid_request_error", param=param
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("необработанная ошибка")
        return error_response(500, "internal server error", "api_error")
