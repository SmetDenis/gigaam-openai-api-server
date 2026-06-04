"""Tests of the OpenAI error format: each exception → the right code and body."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gigaam_api.asr.engine import AudioTooLongError
from gigaam_api.audio import AudioDecodeError, AudioToolNotFoundError
from gigaam_api.errors import AuthError, PayloadTooLargeError, register_exception_handlers
from gigaam_api.runner import QueueFullError


def _app_raising(exc: Exception) -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        raise exc

    return app


@pytest.mark.parametrize(
    "exc, status_code, type_",
    [
        (AudioDecodeError("corrupt audio file"), 400, "invalid_request_error"),
        (AudioTooLongError("audio is too long"), 400, "invalid_request_error"),
        (PayloadTooLargeError("upload exceeds size limit"), 413, "invalid_request_error"),
        (AudioToolNotFoundError("ffmpeg not found"), 500, "api_error"),
        (QueueFullError("inference queue is full"), 503, "server_error"),
        (AuthError("invalid API key"), 401, "invalid_request_error"),
    ],
)
def test_custom_exception_maps_to_openai_error(
    exc: Exception, status_code: int, type_: str
) -> None:
    client = TestClient(_app_raising(exc))
    resp = client.get("/boom")
    assert resp.status_code == status_code
    body = resp.json()
    assert set(body["error"]) == {"message", "type", "param", "code"}
    assert body["error"]["type"] == type_
    assert isinstance(body["error"]["message"], str)


def test_auth_error_sets_invalid_api_key_code() -> None:
    client = TestClient(_app_raising(AuthError("invalid API key")))
    assert client.get("/boom").json()["error"]["code"] == "invalid_api_key"


def test_unexpected_exception_maps_to_500_api_error() -> None:
    # Catch-all for unexpected errors (ffmpeg crash, etc.) — the flag below is needed.
    client = TestClient(_app_raising(RuntimeError("kaboom")), raise_server_exceptions=False)
    resp = client.get("/boom")
    assert resp.status_code == 500
    assert resp.json()["error"]["type"] == "api_error"


def test_request_validation_maps_to_400() -> None:
    from typing import Annotated

    from fastapi import Query

    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/need")
    def need(n: Annotated[int, Query()]) -> dict[str, int]:
        return {"n": n}

    resp = TestClient(app).get("/need")  # the required query param is missing
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"
    assert resp.json()["error"]["param"] == "n"


def test_auth_error_empty_message_uses_default() -> None:
    client = TestClient(_app_raising(AuthError("")))
    resp = client.get("/boom")
    assert resp.json()["error"]["message"] == "Incorrect API key provided."
