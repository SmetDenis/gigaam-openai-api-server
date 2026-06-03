"""Точка входа FastAPI-приложения.

create_app() остаётся лёгким (без torch): модель и Runner создаются в lifespan.
Роутеры (/health, /v1/audio/transcriptions, /v1/models) и OpenAI-обработчики ошибок
подключаются при создании приложения.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from gigaam_api import __version__
from gigaam_api.api.health import router as health_router
from gigaam_api.api.models import router as models_router
from gigaam_api.api.transcriptions import router as transcriptions_router
from gigaam_api.config import get_settings
from gigaam_api.errors import register_exception_handlers
from gigaam_api.logging_setup import setup_logging
from gigaam_api.runner import Runner

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
        # Ленивый импорт: create_app() остаётся без torch; модель грузится при старте.
        from gigaam_api.asr.gigaam_engine import GigaAMEngine

        try:
            engine = GigaAMEngine(settings)
        except Exception:
            logger.exception("ASR-модель не загрузилась — приложение не стартует (fail fast)")
            raise
        app.state.engine = engine
        app.state.runner = Runner(max_queue=settings.MAX_QUEUE)
        logger.info(
            "ASR engine ready: model=%s device=%s max_queue=%d",
            engine.model_name,
            engine.device,
            settings.MAX_QUEUE,
        )
        try:
            yield
        finally:
            app.state.runner.shutdown()
            app.state.runner = None
            app.state.engine = None
            logger.info("ASR engine and runner released")

    app = FastAPI(title="GigaAM ASR", version=__version__, lifespan=lifespan)
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(transcriptions_router)
    app.include_router(models_router)
    return app


app = create_app()
