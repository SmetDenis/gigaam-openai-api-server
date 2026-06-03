# 06 — Docker и деплой на Synology

> Этапный спек. Общий контекст — в [`00-master.md`](./00-master.md), особенно §1–2
> (среды), §9 (зависимости, torch CPU-индекс), §10–11. Самодостаточен.

## Цель
Упаковать сервис в Docker-образ для **Synology (linux/amd64, 4 ядра, 8 ГБ, без GPU)**, без
весов внутри образа (скачивание при первом старте в смонтированный volume), с
`docker-compose.yml`, healthcheck и Makefile-целями. Описать деплой на Synology Container Manager.
На Mac разработка остаётся нативной (uv), Docker — опционален.

## Предусловия
- Завершены этапы 01–05 (рабочий сервис, `make check` зелёный).
- Известно: прод — Synology x86_64 (см. master §2). На Mac dev Docker не обязателен.
- Прочитан master §9 (нюанс torch CPU-колёс), §10 (`/health` для healthcheck).

## Артефакты
```
Dockerfile                  # linux/amd64, python:3.12-slim, ffmpeg, uv, CPU-torch
docker-compose.yml          # env_file, volume весов, healthcheck, restart
.dockerignore
.env.example                # уже есть; убедиться, что MODELS_DIR=/data/models дефолт для контейнера
Makefile                    # +build-docker, up, down, logs, download-weights
README.md                   # раздел «Деплой на Synology»
```

## Задачи
1. **Dockerfile** (`--platform=linux/amd64`):
   - база `python:3.12-slim`; установить `ffmpeg` (apt) и `ca-certificates`; установить `uv`.
   - Скопировать `pyproject.toml`/`uv.lock`; **torch/torchaudio ставить из CPU-индекса**
     (`--index-url https://download.pytorch.org/whl/cpu`) отдельным шагом, чтобы не тянуть CUDA;
     затем `uv sync` остальное (см. master §9 — это ключевая интеграционная точка).
   - Скопировать `gigaam_api/`. Непривилегированный пользователь. `EXPOSE 8000`.
   - `ENV MODELS_DIR=/data/models`, `HF_HOME`/torch-hub cache → внутрь `/data/models` (чтобы Silero тоже кэшировался в volume).
   - CMD: `uvicorn gigaam_api.main:app --host 0.0.0.0 --port 8000` (без `--reload`).
   - HEALTHCHECK на `GET /health`.
   - Слои оптимизировать (кэш зависимостей отдельно от кода). Образ — без весов.
2. **`.dockerignore`**: `.venv`, `tmp/`, `docs/`, `tests/`, `.git`, кэши, `*.wav` и т.п.
3. **`docker-compose.yml`**:
   - сервис `gigaam-api`; `env_file: .env`; `ports: "8000:8000"`;
   - `volumes: ./models:/data/models` (или путь на Synology-томе) — кэш весов переживает пересоздание контейнера;
   - `restart: unless-stopped`; `healthcheck` (curl/python к `/health`);
   - комментарии про лимиты ресурсов (4 ядра / память) для Synology.
4. **Makefile-цели**:
   | Цель | Действие |
   |---|---|
   | `build-docker` | `docker build --platform linux/amd64 -t gigaam-api .` |
   | `up` | `docker compose up -d` |
   | `down` | `docker compose down` |
   | `logs` | `docker compose logs -f` |
   | `download-weights` | прогрев: запустить контейнер/скрипт, который вызывает загрузку модели в `./models` и выходит |
   | `test-integration` | `uv run pytest -m integration` |
5. **`download-weights`**: способ заранее скачать веса (GigaAM + Silero) в `./models` без поднятия
   полного сервиса (например, одноразовый запуск контейнера с командой, инициирующей `load_model`).
   Полезно для медленного/офлайн Synology.
6. **Финальный README.md (полный документ проекта)** — это финал, README доводится до полного вида:
   - что это и возможности; требования (Python 3.12, uv, ffmpeg, Docker);
   - быстрый старт: dev (uv, `make run`) и прод (Docker compose);
   - **примеры использования**: `curl` к `/v1/audio/transcriptions` для каждого `response_format`;
     пример с `stream=true` (SSE); настройка как «Custom OpenAI Compatible Whisper Provider»
     (base_url, api_key, model) — в т.ч. через `openai`-клиент;
   - **полный справочник `.env`** (таблица переменных — синхронно с master §7 / `.env.example`);
   - эндпоинты (`transcriptions`, `/v1/models`, `/health`); что игнорируется (`prompt`/`temperature`);
   - **деплой на Synology**: Container Manager, x86_64; проброс `.env` и volume `./models:/data/models`;
     первый старт качает веса (долго) → healthcheck healthy; задать `API_KEY`;
   - **ограничения/производительность**: скорость на 4 ядрах (RTF), многочасовые файлы лучше через
     `stream=true`; рекомендация моделей (CTC по умолчанию на CPU); best-effort поля в `verbose_json`;
   - **troubleshooting**: ffmpeg не найден, OOM на длинных файлах, MPS-фоллбэк на Mac, перекачка весов.

## Тесты / проверки
- Локальная сборка `make build-docker` успешна.
- `make up` → контейнер healthy; `GET /health` → 200 (через проброшенный порт).
- Короткий файл через `POST /v1/audio/transcriptions` → корректный ответ (smoke).
- Веса появились в `./models`; перезапуск контейнера не перекачивает.
- (CI-опционально) job сборки образа на linux/amd64.

## Debug-логи (этот этап)
- старт контейнера: `INFO` — версия, device=`cpu`, `MODELS_DIR`, `NUM_THREADS`.
- healthcheck-обращения: `DEBUG`.
- путь кэша весов и факт скачивания vs кэш-хит — `INFO`.

## Acceptance-критерии
- [ ] Образ собирается на linux/amd64; CPU-torch (без CUDA); образ без весов.
- [ ] `docker compose up` → healthy; `/health` отвечает.
- [ ] Веса скачиваются в смонтированный volume при первом старте, переиспользуются после.
- [ ] Smoke-тест транскрипции короткого файла проходит в контейнере.
- [ ] **Финальный README** полон: возможности, быстрый старт, curl/`openai`-примеры, справочник `.env`,
      деплой на Synology, ограничения, troubleshooting.
- [ ] `make pre-commit` (на хосте) остаётся зелёным.

## Definition of Done
Контейнер собирается и работает на amd64, веса монтируются; финальный README готов. Соблюдён
**общий DoD из master §14** (зелёный `make pre-commit`, трекер, актуальные `CLAUDE.md`/`README.md`).
Этап 06 → ✅ в трекере.
