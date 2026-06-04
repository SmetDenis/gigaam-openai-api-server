"""OpenAI-compatible error format + registration of exception handlers.

Body: {"error":{message,type,param,code}}. Code mapping: broken file→400,
limits→400/413, tool not found→500, queue→503, auth→401,
unexpected→500. 415 is not used (OpenAI returns 400).
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
    """Missing/invalid Bearer key (→ 401)."""


class PayloadTooLargeError(Exception):
    """Upload exceeded MAX_UPLOAD_MB (→ 413)."""


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
    """Build a JSONResponse in the OpenAI error format."""
    body = OpenAIError(error=OpenAIErrorDetail(message=message, type=type_, param=param, code=code))
    return JSONResponse(status_code=status_code, content=body.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    """Register all handlers on the application."""

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
        logger.exception("audio tool unavailable: %s", exc)
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
        logger.exception("unhandled error")
        return error_response(500, "internal server error", "api_error")
