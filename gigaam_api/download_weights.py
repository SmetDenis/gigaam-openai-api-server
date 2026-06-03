"""Прогрев весов: скачать/проверить веса модели в `MODELS_DIR` и выйти.

Запускается разово (CLI или одноразовый контейнер), HTTP-сервер не поднимает. Полезно
для предварительного прогрева на медленном/офлайн хосте — чтобы боевой старт сервиса
был мгновенным (кэш-хит в смонтированном volume), а не висел на скачивании в lifespan.

Запуск:
    python -m gigaam_api.download_weights
или (см. docker-compose.yml, профиль `tools`):
    docker compose --profile tools run --rm download-weights

Веса берёт `gigaam.load_model(download_root=MODELS_DIR)`; Silero VAD бандлится в пакете
(сеть не нужна). Создание `GigaAMEngine` повторно использует штатный путь загрузки сервиса.
"""

import logging

from gigaam_api import __version__
from gigaam_api.config import get_settings
from gigaam_api.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    setup_logging(settings)
    logger.info(
        "weight warmup v%s | model=%s device=%s models_dir=%s",
        __version__,
        settings.MODEL,
        settings.DEVICE,
        settings.MODELS_DIR,
    )
    # Ленивый импорт: торч/gigaam тянем только при реальном прогреве (конвенция проекта).
    from gigaam_api.asr.gigaam_engine import GigaAMEngine

    engine = GigaAMEngine(settings)
    logger.info(
        "weight warmup done: model=%s device=%s cache=%s",
        engine.model_name,
        engine.device,
        settings.MODELS_DIR,
    )


if __name__ == "__main__":
    main()
