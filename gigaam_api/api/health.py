"""GET /health — статус сервиса.

На этапе 01 модель ещё не загружается, поэтому `loaded=false`, а `device` —
это эхо настройки DEVICE (реальный резолв cuda→mps→cpu появится на этапе 02).
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

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
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    logger.debug("health check requested")
    return HealthResponse(
        status="ok",
        model=settings.MODEL,
        device=settings.DEVICE,
        loaded=False,
    )
