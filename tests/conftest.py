"""Shared test fixtures: isolate Settings from the ambient environment and logging state."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

# All environment variables that Settings reads.
_SETTINGS_ENV_VARS = (
    "MODEL",
    "DEVICE",
    "API_KEY",
    "MODELS_DIR",
    "QUANTIZE_INT8",
    "BATCH_SIZE",
    "NUM_THREADS",
    "MAX_UPLOAD_MB",
    "MAX_AUDIO_SECONDS",
    "MAX_QUEUE",
    "VAD_MIN_DURATION",
    "VAD_MAX_DURATION",
    "VAD_STRICT_LIMIT",
    "VAD_NEW_CHUNK_THRESHOLD",
    "VAD_THRESHOLD",
    "HOST",
    "PORT",
    "LOG_LEVEL",
    "LOG_JSON",
    "DEFAULT_RESPONSE_FORMAT",
    "ALLOWED_MODELS",
)


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Isolate Settings: run from an empty directory (no ambient `.env`),
    clear all of the service's environment variables and the get_settings cache."""
    monkeypatch.chdir(tmp_path)
    for var in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    from gigaam_api.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """Snapshot and restore the root logger's state so that setup_logging
    in one test does not leak into others."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
