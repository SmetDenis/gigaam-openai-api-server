# GigaAM ASR — OpenAI-совместимый сервис распознавания русской речи

Локальный домашний сервис ASR на базе [GigaAM](https://github.com/salute-developers/GigaAM),
выставляющий **OpenAI-совместимый** API (`POST /v1/audio/transcriptions`). Цель — прод на
Synology (Docker, CPU); разработка — на macOS (нативно через uv, опционально MPS).

> **Статус:** в разработке. Реализован **этап 01** (каркас, тулинг, `/health`).
> Полный API, инференс и Docker появятся на следующих этапах. Источник истины — `docs/specs/`
> (начните с `docs/specs/README.md` и `docs/specs/00-master.md`).

## Требования

- **Python 3.12** (см. `.python-version`).
- **[uv](https://docs.astral.sh/uv/)** — менеджер пакетов и окружений.
- **ffmpeg** — для декодирования аудио (понадобится на этапе 02+).

## Быстрый старт

```bash
make install        # uv sync — установка зависимостей
cp .env.example .env # при необходимости отредактируйте
make run            # запуск сервиса (uvicorn --reload) на http://localhost:8000
```

Проверка здоровья:

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"v3_ctc","device":"auto","loaded":false}
```

> На этапе 01 `loaded=false`, а `device` — это эхо настройки `DEVICE`
> (реальный резолв cuda→mps→cpu появится на этапе 02 при загрузке модели).

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
