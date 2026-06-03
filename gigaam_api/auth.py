"""Bearer-аутентификация: один общий ключ из настроек.

Пустой API_KEY → auth выключен (LAN-dev). Иначе сравнение постоянным временем
(`secrets.compare_digest`). Несовпадение/отсутствие → AuthError (→ 401).
"""

import logging
import secrets
from typing import Annotated

from fastapi import Depends, Header

from gigaam_api.config import Settings, get_settings
from gigaam_api.errors import AuthError

logger = logging.getLogger(__name__)

_BEARER_PREFIX = "Bearer "


def require_auth(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    if not settings.API_KEY:
        logger.debug("auth выключен (пустой API_KEY)")
        return
    provided = ""
    if authorization and authorization.startswith(_BEARER_PREFIX):
        provided = authorization[len(_BEARER_PREFIX) :]
    if not provided or not secrets.compare_digest(provided, settings.API_KEY):
        logger.debug("auth не пройден")
        raise AuthError("Incorrect API key provided.")
    logger.debug("auth пройден")
