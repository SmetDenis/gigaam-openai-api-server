# 04 — OpenAI-совместимый API-слой (без стриминга)

> Этапный спек. Общий контекст — в [`00-master.md`](./00-master.md), особенно §6
> «OpenAI-совместимость» и §7 «Конфигурация». Самодостаточен.

## Цель
Выставить синхронный OpenAI-совместимый API: `POST /v1/audio/transcriptions` (multipart,
один запрос → полный транскрипт), `GET /v1/models`, Bearer-аутентификацию, OpenAI-формат
ошибок, рендер всех форматов (`json`/`text`/`verbose_json`/`srt`/`vtt`) и сериализацию
инференса через `Runner` (1 воркер). Стриминг (`stream=true`) — на этапе 05; здесь поле
принимается, но обрабатывается **синхронно** (решение зафиксировано в U4 ниже).

## Предусловия
- Завершены этапы 02 и 03 (engine: short + longform; `app.state.engine`).
- Прочитан master §6 (маппинг запрос/ответ, ошибки), §7 (конфиг), §10 (тесты).

## Уточнения по итогам верификации плана (2026-06-03)

> Зафиксированы после ресёрча OpenAI-совместимости и обсуждения. **Переопределяют** отдельные
> пункты раздела «Задачи» ниже (явно указано, какие). Полное обоснование — в ADR-логе `CLAUDE.md`.

**U1 (уточняет задачу 1) — кооперативная отмена longform.** ThreadPool-задачу прервать нельзя
(проверено: поток досчитает). Воркер один → брошенный многочасовой longform блокирует очередь для
всех. Поэтому:
- Контракт движка расширяем: `ASREngine.transcribe(wav_path, *, word_timestamps, cancel_check: Callable[[], bool] | None = None)`.
  Новое исключение `InferenceCancelledError` в `asr/engine.py`.
- `GigaAMEngine._transcribe_longform` проверяет `cancel_check()` **в начале каждой итерации батча**
  → при `True` бросает `InferenceCancelledError`. Short-путь (≤25с) **неотменяем** (терпимо).
- В `transcriptions.py`: async-watcher опрашивает `request.is_disconnected()` (~1с) → ставит
  `threading.Event`; `cancel_check=event.is_set` прокидывается через `runner.run`. Watcher гасится в
  `finally`. Пойманное `InferenceCancelledError` → лог + `Response(status_code=499)` (клиент уже отвалился).

**U2 (уточняет задачу 1) — лимит очереди.** Новый конфиг **`MAX_QUEUE: int = 8`** (добавлен в master §7,
синхронизировать `.env.example` и conftest `_SETTINGS_ENV_VARS`). `Runner` считает admitted (в очереди +
в работе); при `≥ MAX_QUEUE` → `QueueFullError` → **503** (до постановки в executor). **Таймаут запроса
НЕ вводим** (рубил бы легитимные многочасовые файлы; «брошенное задание» решает U1).

**U3 (переопределяет задачу 3) — маппинг ошибок, разделение причины.** Правим `audio.py` (root-cause):
- `FileNotFoundError` (ffmpeg/ffprobe не в PATH) → новое `AudioToolNotFoundError` → **500** (`type=api_error`).
- битый/неподдерживаемый файл (`CalledProcessError`, плохая длительность) → `AudioDecodeError` → **400**
  (`type=invalid_request_error`) — как реальный OpenAI («Unrecognized file format…» / «Audio file might
  be corrupted or unsupported»).
- **`UnsupportedFormatError` и код 415 НЕ вводим** — OpenAI их не использует (проверено). Из списка
  кастомных исключений задачи 3 `UnsupportedFormatError` исключается; добавляется `AudioToolNotFoundError`
  и `QueueFullError` (503).

**U4 (уточняет задачи 4–7) — OpenAI-детали:**
- `verbose_json`: `seek=0` для всех сегментов (безопасный совместимый дефолт); `compression_ratio` —
  честно per-segment `len(b)/len(zlib.compress(b))`, где `b=t.encode()` (байты/байты, как Whisper; `len(t)` в символах занизил бы кириллицу вдвое), с guard на пустой текст.
- `timestamp_granularities[]` принимаем через `Form(alias="timestamp_granularities[]")` + `list[str] | None`
  (канонический OpenAI-клиент шлёт поле именно с `[]` — проверено).
- `stream=true` → **синхронный ответ** (НЕ 400); TODO-якорь на этап 05.
- `GET /v1/models` → список **всего `ALLOWED_MODELS`** (каждая модель как объект), не только загруженной.
- media-type: `srt` → `text/plain; charset=utf-8`, `vtt` → `text/vtt; charset=utf-8`.

**U5 — доп. тест-файл:** сверх списка ниже добавляется `tests/unit/test_runner.py` (сериализация «1 за раз»,
`QueueFullError` при переполнении, корректный возврат результата, `shutdown`).

## Артефакты
```
gigaam_api/schemas.py             # Pydantic-модели ответа (verbose_json и т.д.)
gigaam_api/auth.py                # зависимость проверки Bearer-ключа
gigaam_api/errors.py              # OpenAI-формат ошибок + exception handlers
gigaam_api/runner.py              # Runner: сериализация инференса (1 поток + lock)
gigaam_api/asr/formats.py         # ASRResult → json|text|verbose_json|srt|vtt
gigaam_api/api/transcriptions.py  # POST /v1/audio/transcriptions
gigaam_api/api/models.py          # GET /v1/models
gigaam_api/main.py                # подключить роутеры, handlers, создать Runner в lifespan
tests/unit/test_formats.py        # рендер всех форматов (чистые функции)
tests/unit/test_auth.py
tests/unit/test_errors.py
tests/unit/test_transcriptions_api.py  # с моком engine через app.state
tests/unit/test_models_api.py
tests/unit/test_runner.py              # сериализация + QueueFullError + shutdown (см. U5)
```

## Задачи
1. **`runner.py`**: `class Runner` — единый `ThreadPoolExecutor(max_workers=1)` + `asyncio.Lock`
   (или очередь) для сериализации. `async def run(self, fn, *args, **kwargs)` исполняет блокирующую
   функцию в executor. Создаётся в lifespan, кладётся в `app.state.runner`. Гарантирует: одновременно
   не более одного инференса; event loop не блокируется.
   - **Лимит очереди (проверено):** при переполнении возвращать **503** (OpenAI-формат), а не молчаливо
     висеть часами (на 4 ядрах один longform — это часы). Лимит — в конфиг (напр. `MAX_QUEUE`).
   - **Отмена (проверено):** задачу в ThreadPool **нельзя прервать** при disconnect — поток досчитает.
     Реальная отмена только кооперативная: longform-цикл проверяет `await request.is_disconnected()`
     (через флаг, прокинутый в воркер) **между батчами** и прерывается. Для short — не отменяемо (терпимо).
2. **`auth.py`**: FastAPI-зависимость `require_auth` — читает `Authorization: Bearer <key>`,
   сравнивает с `settings.API_KEY`. Если `API_KEY` пуст → auth выключен (пропускать). Иначе при
   отсутствии/несовпадении → 401 в OpenAI-формате. Сравнение — `secrets.compare_digest`.
3. **`errors.py`**: модель `OpenAIError` (`{"error":{message,type,param,code}}`); собственные
   исключения → exception handlers с корректным HTTP-кодом. **Список переопределён в U3** (см. выше):
   `AudioDecodeError`→400, `AudioTooLongError`→400, `PayloadTooLargeError`→413,
   `AudioToolNotFoundError`→500, `QueueFullError`→503; **без `UnsupportedFormatError`/415**. Хендлер для
   `RequestValidationError` → 400 в OpenAI-формате; catch-all `Exception` → 500 (`logger.exception`).
4. **`schemas.py`**: Pydantic-модели ответа:
   - `TranscriptionJSON` (`text`);
   - `VerboseSegment` (`id, seek, start, end, text, tokens, temperature, avg_logprob, compression_ratio, no_speech_prob`);
   - `VerboseWord` (`word, start, end`);
   - `VerboseTranscription` (`task, language, duration, text, segments, words`);
   - `ModelsList` / `ModelObject`.
5. **`asr/formats.py`** (чистые функции, не знают про HTTP):
   - `to_json(result) -> dict`;
   - `to_text(result) -> str`;
   - `to_verbose_json(result, *, granularities: set[str]) -> dict` — заполнить недоступные поля
     best-effort (master §6.3): `tokens=[]`, `temperature=0.0`, `avg_logprob=0.0`, `no_speech_prob=0.0`,
     а `compression_ratio` считать честно `len(b)/len(zlib.compress(b))`, `b=text.encode()` (байты/байты, как Whisper; дёшево);
     `segments`/`words` включать по `granularities`;
   - `to_srt(result) -> str`, `to_vtt(result) -> str` — из `result.segments`:
     - SRT: `index`, строка `HH:MM:SS,mmm --> HH:MM:SS,mmm`, текст, пустая строка;
     - VTT: заголовок `WEBVTT\n\n`, время `HH:MM:SS.mmm --> HH:MM:SS.mmm`;
     - вынести форматирование времени в отдельные хелперы (тестировать отдельно).
6. **`api/transcriptions.py`**: `POST /v1/audio/transcriptions`, зависимость `require_auth`.
   - Принять multipart: `file` (req), `model`, `response_format`, `timestamp_granularities[]`,
     `language`, `stream`, `prompt`, `temperature` (последние два — принять и игнорировать).
   - Валидация: `model` ∈ `ALLOWED_MODELS` (иначе 400); `response_format` ∈ допустимых.
   - **Потоковая запись upload (проверено):** НЕ делать `await file.read()` целиком — Starlette `UploadFile`
     при `await read()` без аргумента грузит весь файл в RAM (2 ГБ при `MAX_UPLOAD_MB=2048` убьёт 8 ГБ).
     Писать на диск чанками: `while chunk := await file.read(1<<20): tmp.write(chunk)`, считая размер **по ходу**
     и обрывая на превышении `MAX_UPLOAD_MB` → **413** (`Content-Length` ненадёжен / может отсутствовать при
     chunked). Временный файл — контекст-менеджер, удалять в `finally`.
   - probe длительности; если > `MAX_AUDIO_SECONDS` (и лимит >0) → 400.
   - Определить `word_timestamps` из `timestamp_granularities` (есть `word`).
   - Через `runner.run(engine.transcribe, tmp_path, word_timestamps=...)` получить `ASRResult`.
   - Если `stream=true`: на этом этапе — либо синхронный ответ (упростить), либо 400
     «streaming реализуется на этапе 05». **Решение этапа: вернуть синхронно** (стрим включит этап 05);
     оставить TODO-якорь со ссылкой на спек 05.
   - Отрендерить по `response_format`: `json`/`verbose_json` → JSON; `text` → `PlainTextResponse`;
     `srt`/`vtt` → `PlainTextResponse` с нужным content-type.
7. **`api/models.py`**: `GET /v1/models` → список из `ALLOWED_MODELS` (или только загруженной `MODEL`).
   Формат master §6.5. Можно без auth или с auth — сделать как `transcriptions` (с auth).
8. **`main.py`**: подключить роутеры `transcriptions`, `models`; зарегистрировать exception handlers;
   создать `Runner` в lifespan.

## Тесты
- **unit** (`test_formats.py`): для фикстуры `ASRResult` с 2 сегментами и словами — проверить
  `to_json/to_text/to_verbose_json/to_srt/to_vtt`; отдельно — форматтеры времени (SRT-запятая vs VTT-точка,
  паддинг часов/минут/миллисекунд); `granularities` управляет наличием `words`/`segments`.
- **unit** (`test_auth.py`): пустой `API_KEY` → доступ открыт; заданный ключ → 401 без/с неверным, 200 с верным.
- **unit** (`test_errors.py`): каждое кастомное исключение → правильный код и тело OpenAI-формата.
- **unit** (`test_transcriptions_api.py`): подменить `app.state.engine` моком, `app.state.runner` —
  реальным/упрощённым; multipart-запрос маленьким файлом → проверить ответы для каждого `response_format`,
  413 при превышении размера, 400 при неверном `model`/`response_format`, игнор `prompt`/`temperature`.
- **unit** (`test_models_api.py`): `GET /v1/models` → ожидаемый список.

## Debug-логи (этот этап)
- входящий запрос: `INFO` — `request_id`(uuid4), `model`, `response_format`, имя файла, размер, `stream`.
- auth: `DEBUG` — результат (без самого ключа).
- `DEBUG` — путь tmp-файла, probe-длительность, решение `word_timestamps`.
- очередь/runner: `DEBUG` — постановка/старт/финиш инференса; `INFO` — общее время запроса.
- рендер: `DEBUG` — формат и размер ответа.

## Acceptance-критерии
- [ ] `POST /v1/audio/transcriptions` работает для всех `response_format`; ответы соответствуют master §6.
- [ ] Bearer-auth: при заданном `API_KEY` — 401 без/с неверным ключом; открыт при пустом ключе.
- [ ] Ошибки — в OpenAI-формате с верными кодами (400/401/413/500/503; **415 не используется**, см. U3).
- [ ] `prompt`/`temperature` принимаются и игнорируются; задокументировано.
- [ ] Инференс сериализован через `Runner` (не более одного одновременно), loop не блокируется.
- [ ] `GET /v1/models` отдаёт ожидаемый список.
- [ ] `make pre-commit` зелёный; mypy strict проходит.

## Definition of Done
Полноценный синхронный OpenAI-совместимый сервис распознавания файлов. Соблюдён **общий DoD из
master §14** (зелёный `make pre-commit`, трекер, актуальные `CLAUDE.md`/`README.md`). Этап 04 → ✅
в трекере. Поле `stream` принимается (пока синхронно), якорь для этапа 05 оставлен.
