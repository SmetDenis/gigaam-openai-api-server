"""Bearer-auth tests: empty key → open; set → 401 without/with wrong, 200 with correct."""

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from gigaam_api.auth import require_auth
from gigaam_api.config import Settings, get_settings
from gigaam_api.errors import register_exception_handlers


def _app(api_key: str) -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    app.dependency_overrides[get_settings] = lambda: Settings(API_KEY=api_key)

    @app.get("/protected", dependencies=[Depends(require_auth)])
    def protected() -> dict[str, bool]:
        return {"ok": True}

    return app


def test_auth_disabled_when_key_empty() -> None:
    assert TestClient(_app("")).get("/protected").status_code == 200


def test_auth_missing_header_rejected() -> None:
    resp = TestClient(_app("secret")).get("/protected")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_api_key"


def test_auth_wrong_key_rejected() -> None:
    resp = TestClient(_app("secret")).get("/protected", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_auth_correct_key_accepted() -> None:
    resp = TestClient(_app("secret")).get("/protected", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
