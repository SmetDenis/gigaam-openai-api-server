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
        # Ленивый импорт: create_app() остаётся лёгким (без torch), модель грузится
        # только при реальном старте (первый старт = скачивание весов в MODELS_DIR).
        from gigaam_api.asr.gigaam_engine import GigaAMEngine

        try:
            engine = GigaAMEngine(settings)
        except Exception:
            logger.exception("ASR-модель не загрузилась — приложение не стартует (fail fast)")
            raise
        app.state.engine = engine
        logger.info("ASR engine ready: model=%s device=%s", engine.model_name, engine.device)
        try:
            yield
        finally:
            app.state.engine = None
            logger.info("ASR engine released")

    app = FastAPI(title="GigaAM ASR", version=__version__, lifespan=lifespan)
    app.include_router(health_router)
    return app


app = create_app()
