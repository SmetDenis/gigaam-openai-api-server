# GigaAM ASR — OpenAI-совместимый сервис распознавания русской речи

Локальный домашний сервис ASR на базе [GigaAM](https://github.com/salute-developers/GigaAM),
выставляющий **OpenAI-совместимый** API (`POST /v1/audio/transcriptions`). Цель — прод на
Synology (Docker, CPU); разработка — на macOS (нативно через uv, опционально MPS).

> **Статус:** в разработке. Реализованы **этапы 01–04**: каркас + ASR-движок (PyTorch/GigaAM),
> распознавание **коротких (≤25с) и длинных (до ~10ч)** аудио (Silero VAD + чанкинг + батчевый
> longform-цикл, без pyannote), и полноценный **OpenAI-совместимый API** —
> `POST /v1/audio/transcriptions` (все форматы), `GET /v1/models`, Bearer-auth, OpenAI-формат
> ошибок, сериализация инференса через `Runner`. **Стриминг (`stream=true`)** — этап 05; **Docker** —
> этап 06. Источник истины — `docs/specs/` (начните с `docs/specs/README.md` и `docs/specs/00-master.md`).

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

## API (OpenAI-совместимый)

### `POST /v1/audio/transcriptions` (multipart/form-data)

| Поле | Поведение |
|---|---|
| `file` | **Обязательно.** Любой формат, поддерживаемый ffmpeg. |
| `model` | Валидируется по `ALLOWED_MODELS` (иначе `400`). Фактически используется загруженная модель сервиса. |
| `response_format` | `json` (деф) · `text` · `verbose_json` · `srt` · `vtt`. |
| `timestamp_granularities[]` | `segment` и/или `word` (влияет на наличие `segments`/`words` в `verbose_json`). |
| `language` | Принимается; GigaAM — только RU, на инференс не влияет. |
| `stream` | Принимается; **на этапе 04 ответ синхронный** (SSE — этап 05). |
| `prompt`, `temperature` | **Принимаются и игнорируются** (greedy-декодинг; prompt не поддержан). |

В `verbose_json` поля `tokens`/`avg_logprob`/`no_speech_prob`/`temperature` = `0.0`, `seek` = `0`
(GigaAM их не отдаёт — best-effort, безопасно для клиентских порогов Whisper);
`compression_ratio` считается честно (`len(b)/len(zlib.compress(b))`, байты).

Пример (curl):

```bash
curl -s http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@audio.mp3 \
  -F model=v3_ctc \
  -F response_format=verbose_json \
  -F "timestamp_granularities[]=word"
```

Пример (`openai` Python SDK):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="ваш-ключ")
with open("audio.mp3", "rb") as f:
    print(client.audio.transcriptions.create(file=f, model="v3_ctc").text)
```

### `GET /v1/models`
Список из `ALLOWED_MODELS` в формате OpenAI (`{"object":"list","data":[{"id","object":"model","owned_by":"gigaam"}]}`).

### Ошибки
OpenAI-формат `{"error":{message,type,param,code}}`. Коды: `400` (битый/неподдерживаемый файл,
лимит длительности, неверные параметры), `401` (нет/неверный ключ), `413` (превышен `MAX_UPLOAD_MB`),
`500` (внутренняя ошибка / ffmpeg недоступен), `503` (очередь переполнена, `MAX_QUEUE`).

### Конкурентность и отмена
Инференс сериализован (`Runner`, один воркер); сверх `MAX_QUEUE` запросов → `503`. При отключении
клиента во время длинного аудио инференс кооперативно прерывается между батчами (короткий путь ≤25с
не отменяется).

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
