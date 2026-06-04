"""GET /v1/models — the list of available models."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from gigaam_api.auth import require_auth
from gigaam_api.config import Settings, get_settings
from gigaam_api.schemas import ModelObject, ModelsList

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/v1/models", dependencies=[Depends(require_auth)])
def list_models(settings: Annotated[Settings, Depends(get_settings)]) -> ModelsList:
    logger.debug("models list requested (%d allowed)", len(settings.ALLOWED_MODELS))
    return ModelsList(data=[ModelObject(id=name) for name in settings.ALLOWED_MODELS])
