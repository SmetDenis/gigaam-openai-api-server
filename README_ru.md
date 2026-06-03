# GigaAM ASR — OpenAI-совместимый сервис распознавания русской речи

> **Languages:** [English](README.md) · **Русский**

Self-hosted ASR-сервер на базе [GigaAM](https://github.com/salute-developers/GigaAM),
выставляющий **OpenAI-совместимый** API (`POST /v1/audio/transcriptions`). Любой клиент, умеющий
работать с «Custom OpenAI Compatible Whisper Provider», работает с этим сервисом без доработок.

Цель — **self-hosted OpenAI-совместимый сервер для личного использования**: работает в **Docker**-контейнере
(CPU) на любом хосте — Linux-сервере, NAS или мини-ПК; разработка — на **macOS** (нативно через uv,
опционально MPS).

## Возможности

- `POST /v1/audio/transcriptions` (multipart) — синхронно: один запрос → полный транскрипт.
- Короткие (≤25с) и длинные (до ~10ч) аудио: длинные режутся через **Silero VAD** + чанкинг и
  считаются батчами (без pyannote).
- Форматы ответа: `json`, `text`, `verbose_json`, `srt`, `vtt`.
- Word- и segment-level таймстемпы (`timestamp_granularities[]`).
- Опциональный **`stream=true`** (SSE) — прогрессивная отдача для `json`/`text` (защита от таймаутов
  на многочасовых файлах).
- `GET /v1/models`, `GET /health`.
- **Bearer-аутентификация** (один общий ключ из `.env`).
- Любой формат входного аудио, поддерживаемый ffmpeg.

**Вне scope:** перевод (`/v1/audio/translations` — GigaAM только русское ASR), WebSocket/realtime,
диаризация (спикер один).

## Требования

- **Железо (прод-хост).** Только CPU, GPU не нужен. Минимум **2 ядра / 2 ГБ RAM**; рекомендуется
  **4 ядра / 4–8 ГБ RAM** (зависит от нагрузки и длины аудио — длинным файлам нужно больше).
- **Python 3.12** (см. `.python-version`) — для нативного dev-запуска.
- **[uv](https://docs.astral.sh/uv/)** — менеджер пакетов и окружений (dev).
- **ffmpeg** (с `ffprobe`) — обязателен: декодирование аудио и probe длительности. **В Docker-образе
  ffmpeg+ffprobe уже встроены** (ставятся в образ из apt) — контейнер самодостаточен, на хосте
  ставить ничего не нужно. Только для нативного dev-запуска на macOS установите ffmpeg в систему
  (`brew install ffmpeg`).
- **Docker** + **Docker Compose** — для self-hosted деплоя. На dev-Mac Docker опционален.

---

## Быстрый старт

### Dev (macOS, нативно через uv)

```bash
make install              # uv sync — установка зависимостей
cp .env.example .env       # отредактируйте при необходимости (см. ниже про MODELS_DIR на Mac)
make run                  # uvicorn --reload на http://localhost:8000
```

Проверка здоровья:

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"v3_ctc","device":"mps","loaded":true}
```

> **MODELS_DIR на Mac.** Дефолт `MODELS_DIR=/data/models` рассчитан на контейнер и на macOS не
> записываем. Для нативного dev укажите в `.env` доступный путь, например `MODELS_DIR=./models`
> или `MODELS_DIR=~/.cache/gigaam`.

> Модель грузится один раз при старте (lifespan); первый старт скачивает веса в `MODELS_DIR`
> (на dev — с CDN, минуты). `device` — это **резолв** `DEVICE` (`auto` → cuda→mps→cpu): на dev-Mac
> `auto`→`mps`, в Docker-деплое→`cpu`. При ошибках MPS — `PYTORCH_ENABLE_MPS_FALLBACK=1`.

### Прод (Docker Compose)

Полная инструкция — в разделе [«Деплой (Docker Compose)»](#деплой-docker-compose). Вкратце: положить на хост
`docker-compose.yml` + `Dockerfile` + `.env`, создать каталог `./models` (владелец UID 1000),
поднять проект через `docker compose up -d`. Первый старт скачает веса → healthcheck станет `healthy`.

---

## API (OpenAI-совместимый)

Базовый URL: `http://<host>:8000/v1`. Аутентификация — заголовок `Authorization: Bearer <API_KEY>`
(если `API_KEY` в `.env` пуст — auth выключен).

### `POST /v1/audio/transcriptions` (multipart/form-data)

| Поле | Поведение |
|---|---|
| `file` | **Обязательно.** Любой формат, поддерживаемый ffmpeg. |
| `model` | Валидируется по `ALLOWED_MODELS` (иначе `400`). Фактически используется загруженная сервисом модель. |
| `response_format` | `json` (деф) · `text` · `verbose_json` · `srt` · `vtt`. |
| `timestamp_granularities[]` | `segment` и/или `word` (влияет на наличие `segments`/`words` в `verbose_json`). |
| `language` | Принимается; GigaAM — только RU, на инференс не влияет. |
| `stream` | `true` → **SSE-стриминг** для `json`/`text`; для `verbose_json`/`srt`/`vtt` — синхронный фоллбэк (полный ответ). |
| `prompt`, `temperature` | **Принимаются и игнорируются** (greedy-декодинг; prompt не поддержан). |

#### Примеры (`curl`)

`json` (по умолчанию) → `{"text":"..."}`:

```bash
curl -s http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@audio.mp3 \
  -F model=v3_ctc
```

`text` → просто текст:

```bash
curl -s http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@audio.mp3 -F model=v3_ctc -F response_format=text
```

`verbose_json` с word-таймстемпами (поля `segments`/`words`):

```bash
curl -s http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@audio.mp3 -F model=v3_ctc \
  -F response_format=verbose_json \
  -F "timestamp_granularities[]=segment" \
  -F "timestamp_granularities[]=word"
```

`srt` / `vtt` → субтитры:

```bash
curl -s http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@audio.mp3 -F model=v3_ctc -F response_format=srt
```

> В `verbose_json` поля `tokens`/`avg_logprob`/`no_speech_prob`/`temperature` = `0.0`, `seek` = `0`
> (GigaAM их не отдаёт — best-effort, безопасно для клиентских порогов Whisper); `compression_ratio`
> считается честно (`len(b)/len(zlib.compress(b))`, в байтах).

#### Пример (`openai` Python SDK)

Сервис настраивается как «Custom OpenAI Compatible Whisper Provider»: `base_url`, `api_key`, `model`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="ваш-ключ")
with open("audio.mp3", "rb") as f:
    print(client.audio.transcriptions.create(file=f, model="v3_ctc").text)
```

### Стриминг (`stream=true`, SSE)

Прогрессивная отдача транскрипта по [Server-Sent Events](https://developer.mozilla.org/docs/Web/API/Server-sent_events)
— чтобы многочасовые файлы не упирались в таймауты клиента/прокси. Поддерживается для
`response_format` `json`/`text`; для `verbose_json`/`srt`/`vtt` `stream` игнорируется (синхронный
полный ответ — эти форматы требуют целого результата).

- `Content-Type: text/event-stream` (+ `Cache-Control: no-cache`, `Connection: keep-alive`).
- На каждый готовый сегмент: `data: {"type":"transcript.text.delta","delta":"<кусок текста>"}`.
- В конце: `data: {"type":"transcript.text.done","text":"<полный текст>"}`, затем `data: [DONE]`.
- Ошибка в середине: `data: {"type":"error","error":{...}}` и закрытие потока (без `[DONE]`).
- Во время счёта батча периодически шлётся SSE-комментарий `: keep-alive` (раз в ~15с) —
  держит соединение против idle-таймаутов прокси (на CPU один батч может считаться минутами).

**Инвариант:** конкатенация всех `delta` точно равна `done.text` и **идентична синхронному** ответу
на тот же файл (разделитель-пробел уезжает в начало следующего `delta`).

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="ваш-ключ")
with open("audio.mp3", "rb") as f:
    stream = client.audio.transcriptions.create(file=f, model="v3_ctc", stream=True)
    for event in stream:
        if event.type == "transcript.text.delta":
            print(event.delta, end="", flush=True)
```

### `GET /v1/models`

Список из `ALLOWED_MODELS` в формате OpenAI:
`{"object":"list","data":[{"id":"v3_ctc","object":"model","owned_by":"gigaam"}, ...]}`.

### `GET /health`

`{"status":"ok","model":"<MODEL>","device":"<cpu|mps|cuda>","loaded":true}`. Используется как
Docker `HEALTHCHECK`.

### Ошибки

OpenAI-формат `{"error":{message,type,param,code}}`. Коды:

| Код | Когда |
|---|---|
| `400` | Битый/неподдерживаемый файл, превышен лимит длительности (`MAX_AUDIO_SECONDS`), неверные параметры. |
| `401` | Нет/неверный Bearer-ключ. |
| `413` | Превышен `MAX_UPLOAD_MB`. |
| `500` | Внутренняя ошибка / ffmpeg недоступен в PATH. |
| `503` | Очередь инференса переполнена (`MAX_QUEUE`). |

### Конкурентность и отмена

Инференс сериализован (`Runner`, один воркер); сверх `MAX_QUEUE` запросов → `503`. При отключении
клиента во время длинного аудио инференс кооперативно прерывается между батчами (короткий путь ≤25с
не отменяется).

### Модели (справка)

Сервис загружает **одну** модель (`MODEL` в `.env`); поле `model` в запросах валидируется по
`ALLOWED_MODELS` (фактически используется загруженная модель).

**Рекомендуемые — ASR поколения v3** (новейшее, лучшее качество):

| Модель | Декодер | Регистр/пунктуация | Когда брать |
|---|---|---|---|
| `v3_ctc` (дефолт) | CTC | нижний регистр, без пунктуации | быстрее на CPU — выбор по умолчанию для CPU-хостов |
| `v3_e2e_ctc` | CTC | пунктуация + нормализация регистра | нужна читаемость (знаки, регистр) при скорости CTC |
| `v3_rnnt` | RNN-T | нижний регистр, без пунктуации | выше точность, медленнее на CPU |
| `v3_e2e_rnnt` | RNN-T | пунктуация + нормализация регистра | максимум качества + пунктуация, медленнее всех |

- **CTC vs RNN-T:** CTC быстрее на CPU (рекомендуется на слабых CPU, напр. ~4 ядра); RNN-T точнее, но медленнее.
- **e2e** (есть только у v3): добавляют пунктуацию и нормализацию регистра прямо в вывод; обычные —
  «сырой» нижний регистр без знаков.

**Также скачиваются gigaam** (для смены `MODEL`): `v1_ctc`/`v1_rnnt`, `v2_ctc`/`v2_rnnt` — более старые
поколения ASR (e2e-варианты есть только у v3). Модели `*_ssl` (энкодеры-эмбеддинги) и `emo` (эмоции)
**сервис не поддерживает** — это не транскрипция (`model.transcribe` у них нет).

**Смена модели:** задайте `MODEL=<имя>` в `.env` (по желанию добавьте имя в `ALLOWED_MODELS`, чтобы
клиент мог слать его в поле `model`) и перезапустите контейнер (`docker compose up -d`). Первый запрос
скачает новые веса в `./models` (volume); ранее скачанные сохраняются — переключение без перекачки.

---

## Конфигурация (`.env`)

Все настройки читаются из `.env` (`pydantic-settings`). Пример — `.env.example`.

| Переменная | Тип | Дефолт | Назначение |
|---|---|---|---|
| `MODEL` | str | `v3_ctc` | Имя модели GigaAM (входит в `ALLOWED_MODELS`). |
| `DEVICE` | str | `auto` | `auto`\|`cpu`\|`mps`\|`cuda`. `auto`: cuda→mps→cpu. |
| `API_KEY` | str | `""` | Bearer-ключ. Пусто → auth выключен (в проде задайте). |
| `MODELS_DIR` | path | `/data/models` | Каталог кэша весов (volume). На Mac укажите локальный путь. |
| `QUANTIZE_INT8` | bool | `false` | Динамическая int8-квантизация (этап 07; пока игнорируется). |
| `BATCH_SIZE` | int | `4` | Размер батча longform-инференса. |
| `NUM_THREADS` | int | `4` | `torch.set_num_threads` (держите ≤ числу ядер). |
| `MAX_UPLOAD_MB` | int | `2048` | Лимит размера загрузки → `413`. |
| `MAX_AUDIO_SECONDS` | int | `36000` | Лимит длительности (10ч). `0` = без лимита. |
| `MAX_QUEUE` | int | `8` | Лимит очереди инференса (в очереди + в работе) → `503`. |
| `VAD_MIN_DURATION` | float | `15.0` | Чанкинг: мин. длина сегмента, сек. |
| `VAD_MAX_DURATION` | float | `22.0` | Чанкинг: целевой максимум, сек. |
| `VAD_STRICT_LIMIT` | float | `30.0` | Чанкинг: жёсткий максимум, сек. |
| `VAD_NEW_CHUNK_THRESHOLD` | float | `0.2` | Чанкинг: порог нового чанка, сек. |
| `VAD_THRESHOLD` | float | `0.5` | Silero: порог вероятности речи. |
| `HOST` | str | `0.0.0.0` | Хост uvicorn. |
| `PORT` | int | `8000` | Порт. |
| `LOG_LEVEL` | str | `INFO` | `DEBUG`\|`INFO`\|`WARNING`\|`ERROR`. |
| `LOG_JSON` | bool | `false` | JSON-формат логов (иначе human-readable). |
| `DEFAULT_RESPONSE_FORMAT` | str | `json` | Формат ответа по умолчанию. |
| `ALLOWED_MODELS` | csv | `v3_ctc,v3_e2e_ctc,v3_rnnt,v3_e2e_rnnt` | Допустимые значения поля `model`. |

---

## Деплой (Docker Compose)

Цель деплоя — любой self-hosted Docker-хост: Linux-сервер, NAS или мини-ПК (**x86_64**, без GPU;
минимум 2 ядра / 2 ГБ RAM, рекомендуется 4 ядра / 4–8 ГБ). Деплой — через `docker-compose.yml` (не `make`),
командой `docker compose` (UI управления контейнерами тоже подойдёт). Образ собирается на самом хосте
нативно (amd64) либо заранее на другой машине.

> **Образ самодостаточен.** ffmpeg + ffprobe, Python, torch (CPU) и все зависимости встроены в
> образ — на хосте ничего доустанавливать не нужно. Извне контейнеру требуется только интернет
> **на первом старте** (скачивание весов GigaAM с CDN в volume); далее работает офлайн из кэша.

### Шаги

1. **Файлы на хосте.** Положите в каталог проекта (напр. `/opt/gigaam-api`): `Dockerfile`,
   `docker-compose.yml`, `.dockerignore`, `pyproject.toml`, `uv.lock`, каталог `gigaam_api/`, и `.env`
   (скопируйте из `.env.example`, **задайте `API_KEY`**).
2. **Каталог весов и права.** Создайте подкаталог `models` и сделайте его владельцем **UID 1000**
   (под этим non-root пользователем работает контейнер, иначе нет прав записи весов):
   ```bash
   mkdir -p /opt/gigaam-api/models
   sudo chown -R 1000:1000 /opt/gigaam-api/models
   ```
3. **Сборка и запуск:**
   ```bash
   docker compose up -d --build
   ```
   (На x86_64-хосте образ собирается нативно, без эмуляции.)
4. **Первый старт качает веса** GigaAM в `./models` (минуты, зависит от канала). Контейнер будет в
   статусе «starting», `healthcheck` имеет `start_period: 600s` под скачивание → затем «healthy».
5. **Проверка:** `curl http://<host_ip>:8000/health` → `200` с `"loaded":true`.

### Volume и переустановка

`./models:/data/models` — кэш весов **переживает** пересоздание контейнера (повторно не качается).
`.env` и `./models` лежат на хосте, в образ не пекутся (образ лёгкий, без весов).

### Предварительный прогрев весов (опционально)

Чтобы боевой старт был мгновенным, веса можно скачать заранее, не поднимая сервис, — разовым
запуском профиля `tools`:

```bash
docker compose --profile tools run --rm download-weights
```

(На dev-Mac то же — `make download-weights`.)

### Лимиты ресурсов

В `docker-compose.yml` есть закомментированные `mem_limit`/`cpus` — подберите под железо хоста.
`NUM_THREADS` в `.env` держите ≤ числу выделенных ядер.

---

## Производительность и ограничения

- **Скорость CPU.** Сервис **батчевый, не realtime**: RTF (время счёта / длительность) на длинных
  файлах может быть ≥1 — 10ч аудио считаются часами. Больше ядер — быстрее (рекомендуется 4); на
  минимуме в 2 ядра пропускная способность пропорционально ниже. Для длинных файлов используйте
  `stream=true` (прогресс + защита от таймаутов).
- **Модель.** На CPU по умолчанию — `v3_ctc` (быстрее). RNN-T точнее, но заметно медленнее.
- **Память.** 2 ГБ хватает для коротких аудио; длинным файлам нужно больше. Пик на длинном файле ≈
  веса (~1 ГБ) + int16-буфер (~1.15 ГБ/10ч) + float на VAD-стадии (~2.3 ГБ/10ч) → многочасовым файлам
  нужно ~4–8 ГБ. При нехватке памяти — уменьшите `BATCH_SIZE` или режьте вход на части.
- **`verbose_json`.** Поля `tokens`/`avg_logprob`/`no_speech_prob`/`temperature`/`seek` — best-effort
  (`0.0`/`0`); `compression_ratio` — честный. GigaAM этих метрик не выдаёт.

---

## Troubleshooting

- **`ffmpeg`/`ffprobe` не найден (`500`).** В Docker ставится автоматически; при нативном dev —
  установите ffmpeg в систему (`brew install ffmpeg`) и проверьте `ffmpeg -version`.
- **Нет прав записи весов в volume (контейнер падает на старте).** Хостовый `./models` должен
  принадлежать UID 1000: `sudo chown -R 1000:1000 ./models`.
- **OOM на длинных файлах.** Уменьшите `BATCH_SIZE`, увеличьте лимит памяти контейнера, либо режьте
  вход на части. Пик памяти — на VAD-стадии длинного аудио.
- **Ошибки MPS на Mac.** `auto` на Mac резолвится в `mps`; при ошибках установите
  `PYTORCH_ENABLE_MPS_FALLBACK=1` (есть фоллбэк на CPU) или принудительно `DEVICE=cpu`.
- **Перекачать веса.** Удалите содержимое `./models` (volume) — при следующем старте веса скачаются заново.
- **Долгий первый старт / `unhealthy`.** Это скачивание весов. Дождитесь окончания (`start_period`
  600с); для медленного канала используйте предварительный прогрев (см. выше).

---

## Команды (Makefile)

Удобство для **разработки на Mac** (в проде деплой идёт без `make`, через `docker compose`).

| Цель | Действие |
|---|---|
| `make install` | Установить зависимости (`uv sync`). |
| `make run` | Локальный запуск (uvicorn `--reload`). Переменные `HOST`/`PORT`. |
| `make download-weights-local` | Прогрев весов **нативно** (uv, без Docker) в `MODELS_DIR` из `.env`. |
| `make check` | `lint` + `format-check` + `typecheck` + `test` — быстрый цикл. |
| `make pre-commit` | Вся пачка тестов всех типов подряд (после каждой задачи). |
| `make test` / `make test-integration` | Юнит / интеграционные тесты. |
| `make build-docker` | Сборка прод-образа (`--platform linux/amd64`). |
| `make up` / `make down` / `make logs` | `docker compose up -d` / `down` / `logs -f`. |
| `make download-weights` | Прогрев весов через **Docker** (разовый контейнер, профиль `tools`). |
| `make clean` | Удалить кэши инструментов. |

---

## Разработка

- Язык общения в проекте — русский; код/идентификаторы — английские.
- **TDD**, `mypy strict`, тестируем прагматично (ключевая/рисковая логика и happy-path).
- После каждой задачи — зелёный `make pre-commit`.
- Архитектура и проектные решения задокументированы в `CLAUDE.md` (гайд по проекту + ADR-лог).
