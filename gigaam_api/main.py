"""Точка входа FastAPI-приложения.

На этапе 01 — только настройка логирования, lifespan-заготовка и роутер /health.
Загрузка ASR-модели появится на этапе 02 (см. docs/specs/02-engine-short-audio.md).
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from gigaam_api import __version__
from gigaam_api.api.health import router as health_router
from gigaam_api.config import get_settings
from gigaam_api.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "starting gigaam-api v%s | model=%s device=%s log_level=%s",
            __version__,
            settings.MODEL,
            settings.DEVICE,
            settings.LOG_LEVEL,
        )
        # TODO(этап 02): загрузка ASR-модели здесь (docs/specs/02-engine-short-audio.md).
        yield

    app = FastAPI(title="GigaAM ASR", version=__version__, lifespan=lifespan)
    app.include_router(health_router)
    return app


app = create_app()
