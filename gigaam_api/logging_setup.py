"""Настройка stdlib logging по Settings (master §8).

Человекочитаемый формат по умолчанию; компактный JSON при LOG_JSON=true.
Без внешних зависимостей. Идемпотентно: повторный вызов не плодит хендлеры.
"""

import json
import logging

from gigaam_api.config import Settings

# Имя нашего хендлера — по нему обеспечивается идемпотентность setup_logging.
HANDLER_NAME = "gigaam-api"

_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


class JsonFormatter(logging.Formatter):
    """Минимальный JSON-форматтер: одна строка JSON на запись лога."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(settings: Settings) -> None:
    """Сконфигурировать root-логгер по settings. Безопасно вызывать повторно."""
    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL)

    # Идемпотентность: убираем ранее добавленный нами хендлер перед добавлением нового.
    for handler in list(root.handlers):
        if handler.get_name() == HANDLER_NAME:
            root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.set_name(HANDLER_NAME)
    handler.setFormatter(JsonFormatter() if settings.LOG_JSON else logging.Formatter(_TEXT_FORMAT))
    root.addHandler(handler)
