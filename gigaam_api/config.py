"""Конфигурация сервиса (pydantic-settings). Единственный источник настроек — `.env`.

Полный справочник переменных — в README (раздел «Конфигурация»).
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения. Имена полей совпадают с переменными окружения."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    MODEL: str = "v3_ctc"
    DEVICE: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    API_KEY: str = ""
    MODELS_DIR: Path = Path("/data/models")
    QUANTIZE_INT8: bool = False
    BATCH_SIZE: int = 4
    NUM_THREADS: int = 4
    MAX_UPLOAD_MB: int = 2048
    MAX_AUDIO_SECONDS: int = 36000
    MAX_QUEUE: int = Field(default=8, ge=1)  # 0 запретил бы все запросы (молчаливый 503)
    VAD_MIN_DURATION: float = 15.0
    VAD_MAX_DURATION: float = 22.0
    VAD_STRICT_LIMIT: float = 30.0
    VAD_NEW_CHUNK_THRESHOLD: float = 0.2
    VAD_THRESHOLD: float = 0.5
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_JSON: bool = False
    DEFAULT_RESPONSE_FORMAT: Literal["json", "text", "verbose_json", "srt", "vtt"] = "json"
    # NoDecode отключает JSON-парсинг complex-типа: значение приходит как CSV-строка.
    ALLOWED_MODELS: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["v3_ctc", "v3_e2e_ctc", "v3_rnnt", "v3_e2e_rnnt"]
    )

    @field_validator("ALLOWED_MODELS", mode="before")
    @classmethod
    def _split_allowed_models(cls, v: str | list[str]) -> list[str]:
        """Распарсить CSV-строку из env в список (пробелы и пустые элементы отбрасываются)."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    """Синглтон настроек (читается из окружения один раз)."""
    return Settings()
