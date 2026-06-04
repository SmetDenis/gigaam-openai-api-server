"""GET /health — service status.

If the model is loaded (the lifespan created the engine in app.state.engine) — we return its
real model/device and loaded=true. Otherwise (the engine is not up) — echo the settings,
loaded=false. We narrow the engine type via isinstance(ASREngine), without importing gigaam.
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
    # Allow the `model` field (by default pydantic protects the `model_` namespace).
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
