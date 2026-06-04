"""setup_logging tests: idempotency, level, formatter selection."""

import logging

from gigaam_api.config import Settings
from gigaam_api.logging_setup import HANDLER_NAME, setup_logging


def _our_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if h.get_name() == HANDLER_NAME]


def test_setup_is_idempotent() -> None:
    settings = Settings()
    setup_logging(settings)
    setup_logging(settings)
    assert len(_our_handlers()) == 1


def test_level_applied() -> None:
    setup_logging(Settings(LOG_LEVEL="WARNING"))
    assert logging.getLogger().level == logging.WARNING


def test_json_formatter_outputs_json() -> None:
    setup_logging(Settings(LOG_JSON=True))
    handler = _our_handlers()[0]
    assert handler.formatter is not None
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello", None, None)
    out = handler.formatter.format(record)
    assert out.startswith("{")
    assert '"message": "hello"' in out
