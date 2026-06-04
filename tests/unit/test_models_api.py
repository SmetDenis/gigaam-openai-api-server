"""GET /v1/models tests: ALLOWED_MODELS list in OpenAI format + auth."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gigaam_api.api.models import router as models_router
from gigaam_api.config import Settings, get_settings
from gigaam_api.errors import register_exception_handlers


def _app(settings: Settings) -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    app.dependency_overrides[get_settings] = lambda: settings
    app.include_router(models_router)
    return app


def test_models_lists_allowed_models() -> None:
    resp = TestClient(_app(Settings())).get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["v3_ctc", "v3_e2e_ctc", "v3_rnnt", "v3_e2e_rnnt"]
    assert all(m["object"] == "model" and m["owned_by"] == "gigaam" for m in body["data"])


def test_models_requires_auth_when_key_set() -> None:
    client = TestClient(_app(Settings(API_KEY="secret")))
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200
