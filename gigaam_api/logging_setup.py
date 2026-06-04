"""Configure stdlib logging from Settings.

Human-readable format by default; compact JSON when LOG_JSON=true.
No external dependencies. Idempotent: a repeated call does not spawn extra handlers.
"""

import json
import logging

from gigaam_api.config import Settings

# The name of our handler — it is used to ensure idempotency of setup_logging.
HANDLER_NAME = "gigaam-api"

_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter: one JSON line per log record."""

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
    """Configure the root logger from settings. Safe to call repeatedly."""
    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL)

    # Idempotency: remove the handler we previously added before adding a new one.
    for handler in list(root.handlers):
        if handler.get_name() == HANDLER_NAME:
            root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.set_name(HANDLER_NAME)
    handler.setFormatter(JsonFormatter() if settings.LOG_JSON else logging.Formatter(_TEXT_FORMAT))
    root.addHandler(handler)
