# 00 — Master / Генеральный спек: GigaAM ASR — OpenAI-совместимый сервис

> Зонтичный документ. Описывает архитектуру, все принятые решения, конвенции,
> OpenAI-маппинг, конфигурацию, логирование, риски и трекер этапов.
> Этапные спеки (`01`..`07`) самодостаточны, но ссылаются сюда за общим контекстом.
>
> Дата: 2026-06-03. Язык общения в проекте: русский. Код/идентификаторы — английский.

---

## 1. Обзор и цель

Локальный домашний сервис распознавания **русской речи** на базе модели
[GigaAM](https://github.com/salute-developers/GigaAM), выставляющий
**OpenAI-совместимый** HTTP API (эндпоинт `POST /v1/audio/transcriptions`).
Пользователь отправляет аудиофайл одним запросом и получает полный транскрипт
в формате, совместимом с OpenAI Whisper API. Любой клиент, умеющий работать с
«Custom OpenAI Compatible Whisper Provider», должен работать с этим сервисом без
доработок.

### В scope (v1)
- `POST /v1/audio/transcriptions` (multipart) — синхронный, один запрос → полный транскрипт.
- Поддержка коротких (≤25с) и длинных (до ~10 часов) аудио через VAD-нарезку.
- Опциональный `stream=true` (SSE) — прогрессивная отдача сегментов для длинных файлов (защита от таймаутов).
- `GET /v1/models` — список доступных моделей.
- `GET /health` — статус сервиса.
- Форматы ответа: `json`, `text`, `verbose_json`, `srt`, `vtt`.
- Word-level и segment-level таймстемпы (`timestamp_granularities[]`).
- Bearer-аутентификация (один общий ключ из `.env`).

### Вне scope (v1)
- WebSocket / realtime-стриминг с микрофона.
- `POST /v1/audio/translations` (GigaAM не переводит — только русское ASR).
- Диаризация / мультиспикерность (спикер всегда один).
- Обучение / fine-tuning.

---

## 2. Глоссарий

| Термин | Значение |
|---|---|
| GigaAM | Семейство акустических моделей (Conformer ~240M параметров) от SberDevices. |
| CTC / RNN-T | Типы декодеров. CTC быстрее на CPU; RNN-T точнее, но медленнее. |
| e2e-модель | `v3_e2e_ctc` / `v3_e2e_rnnt` — выдают пунктуацию + нормализацию текста. |
| VAD | Voice Activity Detection — детектор речи; используется для нарезки длинных аудио. |
| Silero VAD | Лёгкий open-source VAD; в этом проекте заменяет pyannote из GigaAM. |
| longform | Обработка аудио длиннее 25с: нарезка на сегменты → батчевый инференс → склейка. |
| RTF | Real-Time Factor = время обработки / длительность аудио. RTF<1 — быстрее реального времени. |
| Engine | Абстракция backend инференса (`ASREngine`). v1: PyTorch. Позже возможен ONNX. |

---

## 3. Принятые решения (с обоснованием)

| # | Решение | Выбор | Почему |
|---|---|---|---|
| D1 | Прод-железо | Synology NAS, **x86_64, 4 ядра, 8 ГБ RAM, без GPU**, Docker | Реальная цель — домашний сервис. |
| D2 | Dev-железо | macOS M4 Max, Python venv (uv); Docker опционален; MPS опционален | Docker на Mac **не видит GPU** (Linux-VM без Metal). MPS только при нативном запуске. |
| D3 | Backend инференса | **PyTorch** (за абстракцией `ASREngine`) | Полный API GigaAM из коробки (`transcribe`, longform, word-timestamps); один код на cpu/mps/cuda. ONNX/Triton — позже. |
| D4 | VAD для длинных аудио | **Silero VAD** (вместо pyannote) | Лёгкий, без HF_TOKEN и лицензий, убирает тяжёлые зависимости (pyannote/transformers/numba), лучше для Python 3.12 и Docker. Спикер один → диаризация не нужна. |
| D5 | Python | **3.12** | Максимальная совместимость с torch/MPS; 3.14 слишком свежий (нестабильные torch-колёса). |
| D6 | Модель по умолчанию | **Конфигурируется через `.env`** (`MODEL`) | На Synology — CTC (быстрее на CPU), на Mac dev — `v3_e2e_rnnt`. |
| D7 | API-режим | Синхронный `POST /v1/audio/transcriptions` (полный транскрипт) + опц. `stream=true` (SSE) | Стандарт OpenAI; SSE решает таймауты на многочасовых файлах. |
| D8 | Эндпоинты | `transcriptions` + `/v1/models` + `/health` | translations пропускаем (нет перевода). |
| D9 | Auth | **Bearer-ключ из `.env`** (`API_KEY`) | Совместимо с OpenAI-клиентами; защита даже в LAN. |
| D10 | Доставка весов | **Скачивание при первом старте в смонтированный volume** | Образ лёгкий, веса не в образе. `make download-weights` для прогрева. |
| D11 | int8-квантизация | **Флаг `QUANTIZE_INT8`, off по умолчанию** | Ускорение на слабом CPU; включается после замеров качества. |
| D12 | Конкурентность | **Один экземпляр модели, инференс сериализован (1 воркер)** | Домашний сервис, 4 ядра — параллелить тяжёлые задачи смысла нет. |
| D13 | Логирование | **stdlib `logging`**, настраиваемый `LOG_LEVEL`, debug-логи в ключевых точках | Без внешних зависимостей, mypy-friendly. |
| D14 | Тулинг качества | **ruff (lint+format) + mypy (strict) + pytest (+coverage)** | Жёсткое тестирование по требованию. |
| D15 | Менеджер пакетов | **uv** | По требованию. |

---

## 4. Архитектура

### 4.1 Структура репозитория
```
gigaam-api/
  gigaam_api/                  # Python-пакет приложения
    __init__.py
    main.py                    # FastAPI app, lifespan (загрузка модели 1 раз)
    config.py                  # pydantic-settings ← .env
    logging_setup.py           # настройка stdlib logging (LOG_LEVEL, формат)
    auth.py                    # зависимость проверки Bearer-ключа
    audio.py                   # ffmpeg-загрузка, probe длительности, чанковая конвертация
    runner.py                  # сериализация инференса (1 воркер, run_in_executor)
    streaming.py               # генерация SSE-событий
    schemas.py                 # Pydantic-модели запроса/ответа (OpenAI-формат)
    errors.py                  # OpenAI-совместимый формат ошибок + обработчики
    api/
      __init__.py
      transcriptions.py        # POST /v1/audio/transcriptions
      models.py                # GET /v1/models
      health.py                # GET /health
    asr/
      __init__.py
      engine.py                # Protocol ASREngine + общие типы результата
      gigaam_engine.py         # PyTorch-реализация поверх gigaam.load_model
      vad.py                   # Silero VAD + алгоритм чанкинга
      formats.py               # result → json|verbose_json|text|srt|vtt
  tests/
    conftest.py
    unit/                      # с моком модели (без скачивания весов)
    integration/               # реальная модель (маркер `integration`)
  docs/specs/                  # эти спеки
  Makefile
  pyproject.toml               # uv, зависимости, конфиг ruff/mypy/pytest
  uv.lock
  .python-version              # 3.12
  Dockerfile
  docker-compose.yml
  .env.example
  .dockerignore
  .gitignore
  CLAUDE.md
  README.md
```

### 4.2 Поток данных
```
Клиент ──multipart──▶ FastAPI /v1/audio/transcriptions
   │
   ├─ auth.py: проверка Bearer
   ├─ сохранить upload во временный файл; probe длительности (audio.py)
   ├─ валидация лимитов (MAX_UPLOAD_MB, MAX_AUDIO_SECONDS)
   ├─ runner.py: поставить в очередь (1 воркер, run_in_executor)
   │     └─ asr/gigaam_engine.py:
   │           если duration ≤ 25с → model.transcribe(word_timestamps=...)
   │           иначе → longform: vad.py (Silero) → батчи → склейка сегментов
   ├─ asr/formats.py: result → нужный response_format
   └─ ответ (sync) ИЛИ SSE-поток (stream=true): delta по сегментам + done
```

### 4.3 Единицы и их ответственность
- `config.Settings` — единственный источник конфигурации; читается из `.env`.
- `ASREngine` (Protocol) — контракт инференса; не знает про HTTP.
- `GigaAMEngine` — обёртка над `gigaam.load_model`; реализует short и longform.
- `vad.segment_audio` — Silero VAD + чанкинг; возвращает `(сегменты, границы)`.
- `formats` — чистые функции рендеринга; не знают про модель.
- `runner.Runner` — сериализация блокирующего инференса; не знает про модель и HTTP.
- `api/*` — только HTTP-слой: парсинг, валидация, вызов engine через runner, рендер.

**Принцип:** HTTP-слой ⟂ инференс ⟂ форматирование ⟂ VAD — каждый тестируется изолированно.

---

## 5. Интеграция с GigaAM (факты для имплементации)

GigaAM ставится как зависимость из официального git-репозитория (см. §9). Используем:

```python
import gigaam
model = gigaam.load_model(
    model_name,            # "v3_ctc" | "v3_e2e_ctc" | "v3_rnnt" | "v3_e2e_rnnt" | путь к .ckpt
    fp16_encoder=True,     # на cpu автоматически не применяется
    use_flash=False,
    device=None,           # "cpu" | "mps" | "cuda"; None → авто (cuda или cpu)
    download_root=None,    # каталог кэша весов; None → ~/.cache/gigaam
)
```

Ключевые факты (из исходников GigaAM):
- `SAMPLE_RATE = 16000`. Аудио грузится через **ffmpeg** (любой формат → 16kHz mono PCM).
- `LONGFORM_THRESHOLD = 25 * 16000` сэмплов (25с). `model.transcribe(...)` **бросает `ValueError`** при превышении.
- `model.transcribe(wav_file: str, word_timestamps: bool=False) -> TranscriptionResult`,
  где `TranscriptionResult(text: str, words: Optional[list[Word]])`, `Word(text, start, end)`.
- Веса качаются с CDN (`https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM`), md5 проверяется.
  Для `e2e`-моделей и `v1_rnnt` дополнительно качается sentencepiece-токенизатор. Всё внутри `load_model`.
- `download_root` управляет каталогом кэша → подставляем `MODELS_DIR` (volume).
- Модели: e2e (`v3_e2e_ctc`/`v3_e2e_rnnt`) дают пунктуацию+нормализацию; обычные (`v3_ctc`/`v3_rnnt`) — нижний регистр без пунктуации.

### 5.1 Longform без pyannote (важно)
**НЕ вызывать `model.transcribe_longform`** — он импортирует `gigaam.vad_utils`, который тянет pyannote.
Вместо этого этап `03` реализует собственный longform-цикл, повторяющий логику
`GigaAMASR.transcribe_longform`, но с Silero VAD. Используются «приватные», но доступные методы:
- `model.forward(wav_pad: Tensor, wav_lens: Tensor) -> (encoded, encoded_len)` — принимает **сырой** батч waveform (внутри сам делает препроцессинг).
- `model._decode(encoded, encoded_len, wav_lens, word_timestamps: bool) -> list[tuple[str, list[Word]|None]]`.

Алгоритм чанкинга портируется из `gigaam/vad_utils.py::segment_audio_file` (параметры:
`max_duration=22.0`, `min_duration=15.0`, `strict_limit_duration=30.0`, `new_chunk_threshold=0.2`),
но источник речевых интервалов — **Silero** `get_speech_timestamps(..., return_seconds=True)`
вместо `pyannote pipeline(...).get_timeline().support()`. Формат интервалов идентичен — список `(start, end)` в секундах.

---

## 6. OpenAI-совместимость (маппинг)

### 6.1 Запрос `POST /v1/audio/transcriptions` (multipart/form-data)
| Поле | Тип | Поведение |
|---|---|---|
| `file` | UploadFile | **Обязательно.** Любой формат, поддерживаемый ffmpeg. |
| `model` | str | Валидируется по `ALLOWED_MODELS`; фактически используется загруженная модель сервиса. |
| `response_format` | str | `json`(деф) \| `text` \| `verbose_json` \| `srt` \| `vtt`. |
| `timestamp_granularities[]` | list[str] | `segment` и/или `word`. Влияет на наличие `words`/`segments` в `verbose_json`. |
| `language` | str | Принимаем; GigaAM — RU. Не влияет на инференс. |
| `stream` | bool | `true` → SSE (см. §6.4). |
| `prompt` | str | **Игнорируется** (не поддерживается). Задокументировать. |
| `temperature` | float | **Игнорируется** (greedy-декодинг). Задокументировать. |

### 6.2 Ответ `json`
```json
{ "text": "Полный распознанный текст." }
```

### 6.3 Ответ `verbose_json`
```json
{
  "task": "transcribe",
  "language": "russian",
  "duration": 123.45,
  "text": "Полный текст...",
  "segments": [
    { "id": 0, "seek": 0, "start": 0.0, "end": 4.2, "text": "...",
      "tokens": [], "temperature": 0.0, "avg_logprob": 0.0,
      "compression_ratio": 0.0, "no_speech_prob": 0.0 }
  ],
  "words": [ { "word": "...", "start": 0.0, "end": 0.3 } ]
}
```
GigaAM не отдаёт `tokens/avg_logprob/compression_ratio/no_speech_prob` — заполняем нейтральными
значениями (best-effort) и фиксируем это в README. `segments`/`words` включаются согласно
`timestamp_granularities[]` (по умолчанию `segment`).

### 6.4 Стриминг (`stream=true`) — SSE
- `Content-Type: text/event-stream`.
- По мере готовности каждого сегмента: `data: {"type":"transcript.text.delta","delta":"<текст сегмента + пробел>"}`.
- В конце: `data: {"type":"transcript.text.done","text":"<полный текст>"}`, затем `data: [DONE]`.
- Стриминг поддерживается для `response_format` `json`/`text` (для `verbose_json`/`srt`/`vtt` — синхронный ответ; при `stream=true` с этими форматами вернуть 400 с понятным сообщением). Зафиксировать контракт в README.

### 6.5 `GET /v1/models`
OpenAI-подобный список: `{"object":"list","data":[{"id":"<MODEL>","object":"model","owned_by":"gigaam"}]}`.

### 6.6 Формат ошибок (как у OpenAI)
```json
{ "error": { "message": "...", "type": "invalid_request_error", "param": null, "code": null } }
```
Коды: 400 (битый/неподдерживаемый ввод, лимиты), 401 (нет/неверный ключ), 413 (превышен размер), 415 (неподдерживаемый формат), 500 (внутренняя ошибка, в т.ч. ffmpeg).

---

## 7. Конфигурация (`.env`)

Читается через `pydantic-settings`. Все секреты — только в `.env` (в репо — `.env.example`).

| Переменная | Тип | Дефолт | Назначение |
|---|---|---|---|
| `MODEL` | str | `v3_ctc` | Имя модели GigaAM. |
| `DEVICE` | str | `auto` | `auto`\|`cpu`\|`mps`\|`cuda`. `auto`: cuda→mps→cpu. |
| `API_KEY` | str | `""` | Bearer-ключ. Пусто → auth выключен (см. D9; для прода задать). |
| `MODELS_DIR` | path | `/data/models` | Каталог кэша весов (volume). Передаётся в `download_root`. |
| `QUANTIZE_INT8` | bool | `false` | Динамическая int8-квантизация (этап 07). |
| `BATCH_SIZE` | int | `4` | Размер батча longform-инференса. |
| `NUM_THREADS` | int | `4` | `torch.set_num_threads`. |
| `MAX_UPLOAD_MB` | int | `2048` | Лимит размера загрузки. |
| `MAX_AUDIO_SECONDS` | int | `36000` | Лимит длительности (10ч). 0 = без лимита. |
| `VAD_MIN_DURATION` | float | `15.0` | Чанкинг: мин. длина сегмента. |
| `VAD_MAX_DURATION` | float | `22.0` | Чанкинг: целевой максимум. |
| `VAD_STRICT_LIMIT` | float | `30.0` | Чанкинг: жёсткий максимум (сверх — режем). |
| `VAD_NEW_CHUNK_THRESHOLD` | float | `0.2` | Чанкинг: порог нового чанка. |
| `VAD_THRESHOLD` | float | `0.5` | Silero: порог вероятности речи. |
| `HOST` | str | `0.0.0.0` | Хост uvicorn. |
| `PORT` | int | `8000` | Порт. |
| `LOG_LEVEL` | str | `INFO` | `DEBUG`\|`INFO`\|`WARNING`\|`ERROR`. |
| `LOG_JSON` | bool | `false` | JSON-формат логов (иначе human-readable). |
| `DEFAULT_RESPONSE_FORMAT` | str | `json` | Формат по умолчанию. |
| `ALLOWED_MODELS` | csv | `v3_ctc,v3_e2e_ctc,v3_rnnt,v3_e2e_rnnt` | Допустимые значения `model`. |

---

## 8. Логирование (debug-стратегия)

- stdlib `logging`; `logging.getLogger(__name__)` в каждом модуле; настройка в `logging_setup.py`.
- Уровень из `LOG_LEVEL`; при `LOG_JSON=true` — компактный JSON, иначе читаемый текст с timestamp/level/logger/msg.
- **DEBUG-точки (обязательно):**
  - входящий запрос: `model`, `response_format`, имя файла, размер, `stream`;
  - результат auth;
  - путь временного файла; результат probe длительности;
  - решение маршрутизации (short / longform) и почему;
  - VAD: число речевых интервалов, число чанков, суммарная длительность речи;
  - longform: прогресс по батчам (`batch i/N`, число сэмплов), время инференса на батч;
  - итог: общее время, **RTF**, число сегментов/слов;
  - выбранный формат и размер ответа.
- **Не логировать** содержимое аудио/текста на INFO; debug-тексты сегментов — только на DEBUG.
- Ошибки — со стектрейсом (`logger.exception`).
- Корреляция: генерировать `request_id` (uuid4) и прокидывать в логи запроса.

---

## 9. Зависимости (пиннинг)

**Рантайм:**
- `python = 3.12`
- `fastapi`, `uvicorn[standard]`, `python-multipart`, `pydantic>=2`, `pydantic-settings`
- `torch>=2.6`, `torchaudio>=2.6` (Mac: дефолтные колёса с MPS; Docker/Linux: **CPU-колёса** через `--index-url https://download.pytorch.org/whl/cpu`)
- `gigaam @ git+https://github.com/salute-developers/GigaAM.git@<PINNED_REV>` — **пиннинг ревизии обязателен**. Тянет core-зависимости GigaAM (hydra-core, omegaconf, onnx, onnxruntime, soundfile, sentencepiece, numpy, tqdm). torch ставим сами (не через extra), чтобы контролировать индекс/арх.
- `silero-vad` (pip-пакет: `load_silero_vad`, `get_speech_timestamps`)

**Dev:**
- `ruff`, `mypy`, `pytest`, `pytest-cov`, `pytest-asyncio`, `httpx` (TestClient), нужные `types-*`.

> Интеграционная точка риска: единый `uv.lock` для Mac(MPS) и Linux(CPU)-torch. Решение —
> в Dockerfile ставить torch/torchaudio из CPU-индекса явным шагом; на Mac — обычный `uv sync`.
> Детали — этап 06. Наличие совместимых с 3.12 колёс `gigaam`/torch проверить на этапе 02.

---

## 10. Тестирование (стандарт для всех этапов)

- **TDD**, но **прагматично, без оверинженеринга**: тестируем поведение и контракты, а не детали
  реализации. Покрываем рисковую/ключевую логику (VAD-чанкинг, `formats`, auth, маршрутизация,
  SSE-события, формат ошибок) и happy-path основных эндпоинтов. **Не** пишем тесты ради тестов,
  не дублируем проверки, не мокаем тривиальщину. Цель — уверенность, а не процент покрытия.
- **Юнит** (`tests/unit/`, без сети/весов): мок `ASREngine`; чистые функции `formats`, алгоритм
  чанкинга `vad` (на синтетических интервалах), `config`, `auth`, маршрутизация по длительности,
  SSE-события, формат ошибок.
- **Интеграция** (`tests/integration/`, маркер `integration`): реальная модель + короткий аудиосэмпл;
  проверка форматов и таймстемпов. Несколько проверок на ключевые сценарии — не больше.
- **Две команды:**
  - `make check` = `ruff check` + `ruff format --check` + `mypy` (strict) + `pytest -m "not integration"`.
    Быстрый внутренний цикл (без сети/весов).
  - **`make pre-commit`** = вся пачка тестов **всех типов один за другим**: `ruff check` →
    `ruff format --check` → `mypy` (strict) → юнит-тесты → интеграционные тесты. Запускается **после
    завершения каждой задачи**; «зелёный» `make pre-commit` — обязательное условие, чтобы считать
    задачу выполненной (и Definition of Done этапа). (Это **не** инструмент `pre-commit`, а Makefile-цель.)
- mypy strict: без `Any` в публичных сигнатурах; типизировать всё, что пишем.

---

## 11. Память, ресурсы, надёжность

- **Память (8 ГБ):** декодировать аудио в int16; во float конвертировать **по сегментам** (а не весь файл сразу). Пик ≈ веса (~1 ГБ) + ~1.2 ГБ на буфер даже для 10ч. `BATCH_SIZE` малый на CPU.
- **Конкурентность:** один экземпляр модели; инференс сериализован через `Runner` (один поток-воркер + `run_in_executor`), event loop не блокируется. Очередь с разумным лимитом.
- **Ошибки:** ffmpeg-сбой → 500 с понятным сообщением; неподдерживаемый/битый файл → 400; лимиты → 413/400.
- **Health:** `/health` → `{status, model, device, loaded}`; модель грузится в lifespan (первый старт = скачивание весов).

---

## 12. Риски (явно)

- **GigaAM на MPS** upstream не тестируют → возможно нужен `PYTORCH_ENABLE_MPS_FALLBACK=1`; на Mac это dev-ускорение, фоллбэк на cpu всегда доступен.
- **Скорость на 4 ядрах:** 10ч аудио = часы счёта (RTF может быть ≥1). Сервис батчевый; SSE даёт прогресс. Митигации: CTC-модель, int8 (этап 07), позже ONNX.
- **`gigaam` git-зависимость на Python 3.12** — проверить колёса torch/зависимостей на этапе 02; пиннинг ревизии.
- **Единый lock Mac/Linux для torch** — см. §9, решается в Dockerfile.

---

## 13. Трекер этапов

| Этап | Файл | Статус | Зависит от | Краткое описание |
|---|---|---|---|---|
| 00 | `00-master.md` | ✅ Готов | — | Зонтичный спек (этот документ). |
| 01 | `01-scaffolding.md` | ⬜ Не начат | — | Каркас, тулинг, логирование, FastAPI-скелет `/health`. |
| 02 | `02-engine-short-audio.md` | ⬜ Не начат | 01 | Config, ASR-движок (PyTorch), короткие аудио ≤25с, кэш весов. |
| 03 | `03-longform-vad.md` | ⬜ Не начат | 02 | Silero VAD + чанкинг + longform-цикл. |
| 04 | `04-openai-api.md` | ⬜ Не начат | 02, 03 | OpenAI-эндпоинты, schemas, formats, auth, runner. |
| 05 | `05-sse-streaming.md` | ⬜ Не начат | 04 | SSE-стриминг (`stream=true`). |
| 06 | `06-docker-deploy.md` | ⬜ Не начат | 01–05 | Docker (amd64), compose, volume, деплой на Synology. |
| 07 | `07-cpu-optimization.md` | ⬜ Опционально | 02–06 | int8-флаг, бенчмарки, задел под ONNX. |

> Легенда: ⬜ не начат · 🟡 в работе · ✅ готов. Реализующая сессия обновляет этот трекер.

---

## 14. Конвенции

- Имена модулей/функций/переменных — английские; докстринги/комментарии можно по-русски (как удобно).
- Все публичные функции типизированы; mypy strict.
- Чистые функции там, где возможно (особенно `formats`, `vad` чанкинг) — для лёгкого юнит-тестирования.
- Никаких сетевых вызовов в импортах модулей. Скачивание весов — только в lifespan/при первом обращении.
- **YAGNI:** не пишем код «на всякий случай» (лишние параметры, абстракции, ветки под гипотетическое
  будущее). Реализуем ровно то, что нужно текущей задаче. Исключение — `ASREngine` как осознанная
  точка расширения под ONNX (зафиксировано решением D3).
- **Без мёртвого кода:** неиспользуемые функции/импорты/параметры/файлы — удаляем сразу
  (ruff ловит часть; остальное — вручную). Не оставляем закомментированный код.
- **Архитектурные решения → в `CLAUDE.md`.** Любое принятое в ходе разработки архитектурное
  решение фиксируется в разделе «Архитектурные решения» файла `CLAUDE.md` (дата, решение, причина) —
  чтобы переиспользовать опыт между сессиями.
- **`CLAUDE.md` и `README.md` всегда актуальны.** Меняется поведение/команды/API/конфиг — синхронно
  правим оба документа в той же задаче. Расхождение доков с кодом считается дефектом.

### Общий Definition of Done (для каждой задачи и этапа)
1. «Зелёный» **`make pre-commit`** (вся пачка тестов всех типов один за другим, §10).
2. Обновлён **трекер этапов** (§13).
3. Новые архитектурные решения внесены в `CLAUDE.md` (раздел «Архитектурные решения»).
4. `CLAUDE.md` и `README.md` приведены в соответствие с изменениями.
5. Удалён неиспользуемый/«на всякий случай» код.
