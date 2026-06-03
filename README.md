# GigaAM ASR — OpenAI-совместимый сервис распознавания русской речи

Локальный домашний сервис ASR на базе [GigaAM](https://github.com/salute-developers/GigaAM),
выставляющий **OpenAI-совместимый** API (`POST /v1/audio/transcriptions`). Цель — прод на
Synology (Docker, CPU); разработка — на macOS (нативно через uv, опционально MPS).

> **Статус:** в разработке. Реализованы **этапы 01–03**: каркас + ASR-движок (PyTorch/GigaAM),
> распознавание **коротких (≤25с) и длинных (до ~10ч)** аудио — длинные через Silero VAD +
> чанкинг + батчевый longform-цикл (без pyannote), загрузка модели в lifespan, кэш весов в
> `MODELS_DIR`, `/health` с реальным статусом модели. HTTP-эндпоинт `transcriptions` и Docker —
> на следующих этапах. Источник истины — `docs/specs/` (начните с `docs/specs/README.md`
> и `docs/specs/00-master.md`).

## Требования

- **Python 3.12** (см. `.python-version`).
- **[uv](https://docs.astral.sh/uv/)** — менеджер пакетов и окружений.
- **ffmpeg** — обязателен: декодирование аудио (GigaAM грузит файлы через ffmpeg) и `ffprobe`
  для probe длительности.

## Быстрый старт

```bash
make install        # uv sync — установка зависимостей
cp .env.example .env # при необходимости отредактируйте
make run            # запуск сервиса (uvicorn --reload) на http://localhost:8000
```

Проверка здоровья:

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"v3_ctc","device":"cpu","loaded":true}
```

> Модель грузится один раз при старте (lifespan); первый старт скачивает веса в `MODELS_DIR`.
> После успешной загрузки `loaded=true`, а `device` — это **резолв** `DEVICE` (`auto` → cuda→mps→cpu).
> На dev-Mac `auto` → `mps`; при ошибках MPS установите `PYTORCH_ENABLE_MPS_FALLBACK=1`. Прод (Synology) — `cpu`.

> **Распознавание:** ядро движка (`GigaAMEngine.transcribe`) роутит по длительности: ≤25с —
> короткий путь (делегирует декод gigaam), иначе — longform (Silero VAD → чанкинг → батчи).
> `AudioTooLongError` теперь только при превышении `MAX_AUDIO_SECONDS` (дефолт 10ч; `0` = без лимита).
> Пик памяти на longform — на VAD-стадии (весь сигнал во float ≈ 2.3 ГБ/10ч); int16-буфер ~1.15 ГБ/10ч.
> HTTP-эндпоинт `POST /v1/audio/transcriptions` подключается на этапе 04.

## Команды (Makefile)

| Цель | Действие |
|---|---|
| `make install` | Установить зависимости (`uv sync`). |
| `make run` | Локальный запуск (uvicorn `--reload`). Переменные `HOST`/`PORT`. |
| `make lint` | `ruff check`. |
| `make format` | `ruff format` (применить). |
| `make format-check` | `ruff format --check`. |
| `make typecheck` | `mypy` (strict). |
| `make test` | Юнит-тесты (без `integration`). |
| `make test-integration` | Интеграционные тесты (реальная модель/сеть). |
| `make coverage` | Отчёт покрытия (без порога). |
| `make check` | `lint` + `format-check` + `typecheck` + `test` — быстрый цикл. |
| `make pre-commit` | Вся пачка тестов всех типов подряд (запускать после каждой задачи). |
| `make clean` | Удалить кэши инструментов. |

## Конфигурация

Все настройки читаются из `.env` (`pydantic-settings`). Полный список переменных и дефолтов —
в `.env.example` и `docs/specs/00-master.md §7`.

## Разработка

- Язык общения в проекте — русский; код/идентификаторы — английские.
- **TDD**, `mypy strict`, тестируем прагматично (ключевая/рисковая логика и happy-path).
- После каждой задачи — зелёный `make pre-commit`.
