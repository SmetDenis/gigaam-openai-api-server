"""Тест эндпоинта GET /health."""

from fastapi.testclient import TestClient

from gigaam_api.config import Settings, get_settings
from gigaam_api.main import create_app


def test_health_ok() -> None:
    app = create_app()
    # Детерминизм: подменяем зависимость на дефолтные настройки.
    app.dependency_overrides[get_settings] = lambda: Settings()
    client = TestClient(app)

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "model": "v3_ctc",
        "device": "auto",
        "loaded": False,
    }
