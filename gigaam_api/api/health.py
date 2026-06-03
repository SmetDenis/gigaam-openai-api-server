"""GET /health — статус сервиса.

Если модель загружена (lifespan создал движок в app.state.engine) — отдаём её
реальные model/device и loaded=true. Иначе (движок не поднят) — эхо настроек,
loaded=false. Тип движка сужаем через isinstance(ASREngine), не импортируя gigaam.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict

from gigaam_api.asr.engine import ASREngine
from gigaam_api.config import Settings, get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    # Разрешаем поле `model` (по умолчанию pydantic защищает namespace `model_`).
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model: str
    device: str
    loaded: bool


@router.get("/health")
def health(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    logger.debug("health check requested")
    engine = getattr(request.app.state, "engine", None)
    if isinstance(engine, ASREngine):
        info = engine.info()
        return HealthResponse(
            status="ok",
            model=info["model"],
            device=info["device"],
            loaded=info["loaded"],
        )
    return HealthResponse(
        status="ok",
        model=settings.MODEL,
        device=settings.DEVICE,
        loaded=False,
    )
