# CLAUDE.md — GigaAM ASR (OpenAI-совместимый сервис)

Сервис распознавания **русской речи** на базе [GigaAM](https://github.com/salute-developers/GigaAM),
выставляющий **OpenAI-совместимый** API (`POST /v1/audio/transcriptions`). Запускается как
локальный домашний сервис на Synology (Docker) и разрабатывается на macOS.

## Источник истины — спеки

Проект строится **поэтапно, в разных сессиях**. Полные требования — в `docs/specs/`:

- **`docs/specs/README.md`** — план реализации по сессиям + рекомендации/советы (читать первым).
- **`docs/specs/00-master.md`** — генеральный спек: архитектура, ВСЕ решения, OpenAI-маппинг,
  конфиг, логирование, риски и **трекер этапов** (актуальный статус — там).
- `01-scaffolding.md` … `07-cpu-optimization.md` — самодостаточные этапные спеки.

**Перед любой работой:** прочитай `00-master.md` + спек текущего этапа. После работы —
обнови трекер этапов в `00-master.md` (⬜→🟡→✅).

## Статус

Этап **06 ✅** (Docker/деплой на Synology: multi-stage `python:3.12-slim`, **CPU-torch через
index+marker** в `pyproject.toml` — единый `uv.lock` Mac(MPS)/Linux(`2.12.0+cpu`), CUDA/nvidia/triton
больше не тянутся; **самодостаточный образ** — ffmpeg+ffprobe из apt, non-root UID 1000, healthcheck
через stdlib urllib; `docker-compose.yml` под **Synology Container Manager UI** (`env_file required:false`,
volume `./models:/data/models`, `start_period 600s` под первый скач весов, опц. профиль `tools` для
прогрева); `download_weights.py`; финальный README. Деплой = compose + UI, **без make** на проде;
Silero бандлится в пакете → volume только для весов GigaAM). Следующий — `07` (опц. CPU-оптимизация).
Этап **05 ✅** (SSE-стриминг `stream=true`: `transcript.text.delta`→`transcript.text.done`→`[DONE]`,
инвариант `"".join(delta)==done.text==sync` (префикс-пробел), heartbeat-комментарии ~15с против
idle-таймаутов; `iter_segments` (общий батч-цикл с longform); мост поток→async через `asyncio.Queue`
+ `Runner.submit` в тот же воркер; backpressure `try_acquire`→503 ДО заголовков; передача владения
temp-файлом стриму; verbose/srt/vtt+stream→синхронный фоллбэк; ошибка в потоке→`error`-событие).
Этап **04 ✅** (OpenAI-совместимый API: `POST /v1/audio/transcriptions` все форматы
`json`/`text`/`verbose_json`/`srt`/`vtt`, `GET /v1/models`, Bearer-auth, OpenAI-формат ошибок;
`Runner` (1 воркер + `MAX_QUEUE`→503); кооперативная отмена longform по disconnect через **anyio
task group**; upload чанками→413; probe-лимит→400). Этап **03 ✅** (longform >25с: Silero VAD (JIT)
→ чистая функция чанкинга `merge_intervals_to_chunks` (порт из gigaam) → батчевый
`model.forward`/`model._decode`; роутинг по длительности внутри движка; int16-декод для экономии
памяти; без pyannote). Этап **02 ✅** (ASR-движок PyTorch за `ASREngine`, короткие аудио ≤25с,
загрузка модели в lifespan, кэш весов в `MODELS_DIR`, `/health.loaded=true`). Этап **01 ✅** (каркас,
тулинг, логирование, FastAPI-скелет).
Актуальный трекер — в `00-master.md` §13.

## Архитектурные решения (ADR-лог)

> **Правило:** любое новое архитектурное решение, принятое в ходе разработки, **дописывается сюда**
> (дата · решение · причина) — чтобы переиспользовать опыт между сессиями. Это живой раздел.
> Подробное обоснование исходных решений — в `docs/specs/00-master.md` §3.

| Дата | Решение | Причина |
|---|---|---|
| 2026-06-03 | Backend инференса — **PyTorch** за абстракцией `ASREngine` (ONNX опционально, этап 07) | Полный API GigaAM из коробки; один код на cpu/mps/cuda. |
| 2026-06-03 | VAD длинных аудио — **Silero VAD** (НЕ pyannote) | Лёгкий, без HF_TOKEN/лицензий; убирает тяжёлые зависимости; спикер один. |
| 2026-06-03 | **Python 3.12**, менеджер **uv** | Совместимость torch/MPS; 3.14 слишком свежий. |
| 2026-06-03 | Модель — через `.env` (`MODEL`), по умолчанию CTC | CTC быстрее на CPU (Synology). |
| 2026-06-03 | Auth — Bearer-ключ из `.env` (`API_KEY`) | Совместимо с OpenAI-клиентами. |
| 2026-06-03 | Веса — скачиваются при первом старте в volume (`MODELS_DIR`), НЕ в образе | Лёгкий образ. |
| 2026-06-03 | API — синхронный `transcriptions` + опц. `stream=true` (SSE); эндпоинты `transcriptions`, `/v1/models`, `/health` | Стандарт OpenAI; SSE против таймаутов. translations/WebSocket — вне scope. |
| 2026-06-03 | Сборка пакета — **hatchling** (editable install через `uv sync`) | `gigaam_api` импортируется в pytest/mypy/uvicorn; `uvicorn gigaam_api.main:app` работает надёжно. |
| 2026-06-03 | Пиннинг — **нижние границы (`>=`) в `pyproject.toml` + точные пины в `uv.lock`** | Идиома uv: воспроизводимость через lock, апгрейд осознанно через `uv lock --upgrade`. |
| 2026-06-03 | `DEVICE=auto` резолвим **сами** (cuda→mps→cpu), в `load_model` передаём явный device (этап 02) | Встроенный auto GigaAM (`device=None`) = cuda→cpu, **без MPS**; MPS нужен на dev-Mac. На этапе 01 `/health.device` = эхо настройки. |
| 2026-06-03 | CSV-поля Settings (`ALLOWED_MODELS`) — **`Annotated[..., NoDecode]` + `field_validator`** | pydantic-settings по умолчанию парсит `list` как JSON; NoDecode + `split(",")` даёт CSV. |
| 2026-06-03 | ruff: отключены **RUF001/002/003** (ambiguous-unicode) | Ложные срабатывания на легитимную кириллицу в комментариях/докстрингах (конвенция — русский). |
| 2026-06-03 | gigaam пин **`6e4b027`** проверен: `torch==2.12.0`/`torchaudio==2.11.0` + `onnxruntime==1.23.2`/`onnx==1.19.1`/`numpy==2.4.6` ставятся на **Python 3.12 macOS arm64** (dev) | Этап 02 — блокер-проверка колёс пройдена; MPS доступен, CUDA нет. |
| 2026-06-03 | `uv add` хранит git-пин gigaam в **`[tool.uv.sources]`** (`rev=6e4b027`), зависимость объявлена как голое `gigaam` | Идиома uv; пин сохранён (rev + `uv.lock`), форма эквивалентна `gigaam @ git+...@rev`. |
| 2026-06-03 | Маршрутизация >25с: **pre-check `probe_duration`>25с → `AudioTooLongError`** + защитный перехват сырых исключений gigaam (`ValueError "too long"`→`AudioTooLongError`, `RuntimeError "failed to load audio"`→`AudioDecodeError`); прочие — пробрасываем | gigaam меряет длину по сэмплам, probe — секунды → у границы 25с возможен рассинхрон; не маскируем посторонние ошибки инференса. |
| 2026-06-03 | `ASREngine` расширен **`info()` + `@runtime_checkable`**; `/health` сужает тип `app.state.engine` через `isinstance`, **не импортируя gigaam/torch** | Принцип master §4.3 «HTTP ⟂ инференс»: HTTP-слой остаётся лёгким, `create_app()` без torch (ленивый импорт движка в lifespan). |
| 2026-06-03 | mypy: **per-module `ignore_missing_imports`** для `gigaam.*`/`silero_vad.*` | Нет py.typed/стабов; точечный override идиоматичнее широкого `# type: ignore`. |
| 2026-06-03 | Интеграционный сэмпл — **committed `tests/integration/data/ru_short_sample.wav`** (11.29с, RU; имя ≠ `example.wav`); тест на **cpu**, грейсфул-skip без сети/весов | `.gitignore` глобально игнорирует throwaway `example.wav` (его пишет `gigaam.utils`); отдельное имя сохраняет конвенцию и трекает фикстуру. cpu = детерминизм + прод-Synology. |
| 2026-06-03 (этап 03) | Silero backend — **JIT (`load_silero_vad(onnx=False)`), НЕ ONNX** | Один torch-стек с GigaAM; onnxruntime по умолчанию `intra_op_num_threads=0` (все ядра) → oversubscription с torch-пулом на 4 ядрах Synology. VAD не bottleneck (≈часы инференса vs минуты VAD). Веса бандлятся в пакете (без сети). Переключение в ONNX позже = 1 строка. |
| 2026-06-03 (этап 03) | **Роутинг внутри движка** (заменяет stage-02 строку выше): `probe_duration` → `>MAX_AUDIO_SECONDS`→`AudioTooLongError`; `≤25с`→short (делегируем `model.transcribe`, не трогаем); иначе→`_transcribe_longform`. `ValueError "too long"` у границы теперь → **fallback в longform** (не ошибка) | `AudioTooLongError` на обычном пути убран (спек 03 задача 6); short-путь не переписываем (минимум риска для hot-path); у границы 25с gigaam меряет по сэмплам → корректнее уйти в longform, чем падать. |
| 2026-06-03 (этап 03) | Longform — порт `gigaam/vad_utils.py::segment_audio_file`: **чистая функция `merge_intervals_to_chunks` (только границы)** + срез waveform/батчинг в engine; интервалы от Silero; инференс через приватные `model.forward`/`model._decode` (master §5.1); слова `+seg_start`, `round(...,3)` | Чистую логику слияния тестируем синтетикой изолированно (ядро этапа); `transcribe_longform` upstream не зовём (он тянет pyannote). |
| 2026-06-03 (этап 03) | Память: декод в **int16 `torch.Tensor`** (`torch.frombuffer`, без numpy); весь сигнал во float **только на VAD-стадию** → `del wav_f32` сразу; инференс — float по срезу-батчу | int16 вдвое экономит память (~1.15 ГБ/10ч); пик float — на VAD (≈2.3 ГБ/10ч), не на батчах; numpy не добавляем — остаёмся в torch-стеке. Ленивые импорты torch в `audio.py` (модуль остаётся torch-free для HTTP-слоя). |
| 2026-06-03 (этап 03) | Longform-фикстура — **committed `ru_long_sample.wav`** (40с, обрезка реального GigaAM `long_example.wav` через ffmpeg, mono 16k) | Реальная RU-речь с паузами → >1 чанк; НЕ `gigaam.utils.download_long_audio()` (wget в CWD). Грейсфул-skip без сети/весов. |
| 2026-06-03 (этап 04) | Отмена longform при disconnect — **кооперативная**: `ASREngine.transcribe` расширен опц. `cancel_check: Callable[[], bool] \| None`; longform проверяет в начале каждой итерации батча → `InferenceCancelledError`; API ставит watcher на `request.is_disconnected()` → `threading.Event`. Short-путь (≤25с) неотменяем. | ThreadPool-задачу прервать нельзя (проверено) — поток досчитает; воркер один → брошенный longform блокирует очередь для всех. Реальная отмена только кооперативная. |
| 2026-06-03 (этап 04) | Backpressure — один ключ **`MAX_QUEUE=8`**; `Runner` считает admitted (очередь+работа), при `≥MAX_QUEUE` → `QueueFullError`→**503**. Таймаут запроса **НЕ вводим**. | Дефолтный таймаут рубил бы легитимные многочасовые файлы (RTF≥1); «брошенное задание» решается отменой, а не грубым таймаутом. YAGNI. |
| 2026-06-03 (этап 04) | Маппинг ошибок — **разделить причину в `audio.py`**: `FileNotFoundError` (ffmpeg/ffprobe не в PATH) → новое `AudioToolNotFoundError`→**500** (`api_error`); битый/неподдерживаемый файл → `AudioDecodeError`→**400** (`invalid_request_error`). `UnsupportedFormatError`/**415 выкинуты**. | Реальный OpenAI на плохой аудиофайл отдаёт 400 `invalid_request_error` («Unrecognized file format…» / «Audio file might be corrupted…»), 415 не использует (проверено). Один код для клиентской и серверной причин — неверен (root-cause). |
| 2026-06-03 (этап 04) | OpenAI-уточнения — `timestamp_granularities[]` через `Form(alias="timestamp_granularities[]")`+`list[str]`; verbose `seek=0` + честный per-segment `compression_ratio`; `stream=true`=синхронный ответ до этапа 05; `/v1/models`=весь `ALLOWED_MODELS`. | Канонический OpenAI-клиент шлёт поле с `[]` (проверено). `seek=0` — безопасный совместимый дефолт; `compression_ratio` дёшев и осмысленен. Контракт фиксируется в README. |
| 2026-06-03 (этап 04) | `compression_ratio` — **байты/байты** `len(b)/len(zlib.compress(b))`, `b=text.encode()` (НЕ `len(text)` в символах). | Реальный Whisper считает байты с обеих сторон; для кириллицы (2 байта/символ) числитель в символах занижал бы ratio вдвое → порог галлюцинаций (>2.4) не сработал бы. Поймано на код-ревью этапа 04. |
| 2026-06-03 (этап 04) | Watcher disconnect'а — **только через `anyio.create_task_group()` + `cancel_scope.cancel()`**, НЕ через `asyncio.create_task` + `task.cancel()`/`await task`. Исход инференса захватываем внутри группы и диспетчеризуем снаружи (иначе `QueueFullError` обернётся в `ExceptionGroup` → 500 вместо 503). | `Request.is_disconnected()` (Starlette 1.2.x) внутри держит `anyio.CancelScope`; raw-asyncio отмена с ней конфликтует → watcher не завершается, `await watcher` **дедлочит** весь запрос (поймано faulthandler'ом: event loop idle в select, главный поток ждёт portal). Структурная anyio-отмена консистентна. |
| 2026-06-04 (этап 05) | Семантика delta — **префикс-пробел**: первый delta=`seg0.text`, последующие=`" "+segN.text`; `done.text=" ".join(сегменты)`. Инвариант: `"".join(delta)==done.text==синхронному`. Уточняет master §6.4 («+пробел в конце»). | Универсальный инвариант стриминга OpenAI (chat/responses/transcription, проверено по docs): склейка delta точно воспроизводит финальный текст. Суффикс-пробел дал бы хвостовой пробел → расхождение с `done`/sync (acceptance §05). |
| 2026-06-04 (этап 05) | Мост «блокирующий `iter_segments` → async» — **`asyncio.Queue` + `loop.call_soon_threadsafe`**, продюсер в **`Runner.submit` (тот же 1 воркер)**, НЕ временный поток. Очередь без `maxsize` (продюсер — bottleneck, никогда не блокируется на put). | Сериализация инференса сохранена (1 за раз), event loop не блокируется. `call_soon_threadsafe` — канонический мост поток→loop. heartbeat реализуем через `wait_for(queue.get(), 15s)` (отмена своей корутины безопасна), НЕ через `wait_for(__anext__())` чужого генератора (его отмена убила бы мост). |
| 2026-06-04 (этап 05) | Backpressure при стриме — **`runner.try_acquire()` в обработчике ДО `StreamingResponse`** (503 без заголовков); `release()` — в **done-callback future продюсера** (воркер реально свободен), не когда consumer дочитал. `_inflight` под `threading.Lock` (меняют loop и воркер-поток). | Async-генератор откладывает тело до первой итерации (после `200`) → 503 нужно отдать раньше. Release по завершению продюсера = inflight отражает занятость воркера, а не скорость клиента. |
| 2026-06-04 (этап 05) | **Владение temp-файлом передаётся стриму**: handler ставит флаг `streamed`, `finally` НЕ удаляет файл; удаляет `_cleanup` (done-callback продюсера), когда воркер закончил читать. | Handler возвращает `StreamingResponse` и его `finally` сработал бы СРАЗУ → удалил бы файл до чтения воркером (root-cause). Файл нужен на всё время инференса. |
| 2026-06-04 (этап 05) | Отмена стрима — **`cancel_event.set()` в `finally` генератора моста**; Starlette сам отменяет генератор при disconnect (uvicorn HTTP `spec_version=2.3 < 2.4` → ветка task group с `listen_for_disconnect`). НЕ переиспользуем anyio-watcher этапа 04. | `iter_segments` стопает между батчами (та же гранулярность, что sync-путь). `sse_transcription` ловит `Exception` (→ error-событие), но пропускает `CancelledError`/`GeneratorExit` (disconnect → только очистка). |
| 2026-06-04 (этап 05) | `verbose_json`/`srt`/`vtt` + `stream=true` → **синхронный фоллбэк** (игнор `stream`), НЕ 400 (отклонение от спека §05). Условие стрима: `stream and fmt in {json,text}`. | Большинство OpenAI-клиентов шлют `stream=true` по умолчанию и используют `verbose_json` → 400 ломал бы их. Предсказуемость сохранена: эти форматы всегда дают полный ответ. Спек §05 и master §6.4 обновлены. |
| 2026-06-04 (этап 05) | `iter_segments` — **общий батч-цикл `_iter_chunks`** (+ `_prepare_longform`), переиспользуемый sync-`_transcribe_longform` (через `list(...)`). Добавлен в `ASREngine` Protocol → фейки-движки в тестах реализуют его (runtime_checkable проверяет наличие метода → иначе `/health` ломается). | Один источник longform-логики (DRY); sync-поведение не изменилось. `iter_segments` для ≤25с делегирует короткому пути и yield'ит его единственный сегмент. |
| 2026-06-04 (этап 06) | CPU-torch — **`index+marker` в `pyproject.toml`** (НЕ «отдельный шаг в Dockerfile» из спека §06): `[[tool.uv.index]] pytorch-cpu` (`explicit=true`) + `[tool.uv.sources]` torch/torchaudio с маркером `sys_platform=='linux'`. Единый `uv.lock`: Mac → `torch 2.12.0` (PyPI, MPS), Linux → `2.12.0+cpu` (индекс). В Dockerfile просто `uv sync --frozen`. | Идиома uv 2026 (проверено по докам). Побочно `uv lock` **убрал из Linux-резолва весь CUDA-стек** (`nvidia-*`, `triton`, `cuda-*`) — старый lock тянул бы CUDA-torch в образ (гигабайты). Воспроизводимость через единый lock, без хрупкого `--no-install-package` в Dockerfile. |
| 2026-06-04 (этап 06) | Образ — **multi-stage `python:3.12-slim`**: builder (uv из `ghcr.io/astral-sh/uv` пин + `git` для git-gigaam, `uv sync --no-install-project` слой деп → COPY код → `uv sync`) + тонкий runtime (ffmpeg+ffprobe из apt, non-root **UID/GID 1000**, COPY `.venv`+`gigaam_api`). Платформа — на **build-time** (`docker build --platform linux/amd64`), НЕ хардкод в `FROM`. HEALTHCHECK — `python -c urllib` (в slim нет curl). `XDG_CACHE_HOME=/data/models/.cache`. | **Самодостаточный образ**: ffmpeg внутри (Synology его не имеет — критично). Кэш-слои деп отдельно от кода. `--platform` не в `FROM` → мультиарх-дружественно + быстрая нативная валидация на Mac (arm64, без qemu — проверено, сборка ~90с). UID 1000 + chown volume — права записи весов non-root. |
| 2026-06-04 (этап 06) | **Silero бандлится в pip-пакете** (`silero_vad/data/*.jit/.onnx` в site-packages) → volume нужен **только** для весов GigaAM (`MODELS_DIR`). Спек §06 «кэшировать Silero/HF в volume» **неактуален**: HF Hub / torch.hub проект не использует; `XDG_CACHE_HOME`→volume оставлен лишь защитной сетью для non-root. | Проверено `find .venv` — модель Silero в пакете, сеть/кэш для VAD не нужны (согласуется с ADR этапа 03). Не плодим лишние volume/ENV. |
| 2026-06-04 (этап 06, багфикс) | Longform-инференс (`_iter_chunks`: `forward`+`_decode`) обёрнут в **`torch.inference_mode()`** — как `gigaam.transcribe`/`transcribe_longform` (оба `@torch.inference_mode()`). Без обёртки автоград включён, и кэш rotary `cos`/`sin` энкодера, созданный коротким путём (под inference_mode) как inference-тензоры, ронял longform: `RuntimeError: Inference tensors cannot be saved for backward`. Проявлялось **только** в порядке short→long на ОДНОМ инстансе модели (живой сервис); integration-тесты с отдельным инстансом на файл баг не ловили. | Короткий путь делегирует `model.transcribe` (под inference_mode); longform звал `forward`/`_decode` напрямую без контекста → смешение inference-тензоров с автоградом. Регресс-тест `tests/integration/test_short_then_long_real.py` (один движок, short→long) воспроизводит (падал до фикса) и фиксирует. |
| 2026-06-04 (этап 06) | Деплой — **`docker-compose.yml` + Synology Container Manager UI, без `make` на проде** (требование пользователя). `make`-цели (`build-docker`/`up`/`down`/`logs`/`download-weights`) — только dev-удобство на Mac. `env_file` с `required:false` (стартует на дефолтах без `.env`). Прогрев весов — опц. compose-сервис `download-weights` (`profiles:["tools"]`) + модуль `gigaam_api/download_weights.py`; первый старт сервиса всё равно качает веса (`healthcheck start_period 600s`). | Synology UI не запускает `make`/`compose run` удобно → прод-путь должен «просто работать» из compose: первый `up` сам качает веса, `start_period` покрывает скачивание. Прогрев — для тех, кто хочет пред-скачать; не обязателен. |

<!-- Новые решения добавляй новой строкой выше этой подсказки. -->


## Критичные предостережения (root-cause, не нарушать)

1. **Docker на Mac НЕ видит GPU.** Контейнеры идут в Linux-VM без Metal → только CPU. MPS на Mac
   доступен лишь при **нативном** запуске (uv), не в Docker. Прод = Synology CPU.
2. **НЕ вызывать `model.transcribe_longform`** — он тянет pyannote. Longform делаем сами через
   Silero VAD + порт чанкинга GigaAM (master §5.1). Нигде не должно быть `import pyannote`.
3. **torch в Docker** — CPU-колёса (`download.pytorch.org/whl/cpu`), без CUDA. Реализовано через
   `index+marker` в `pyproject.toml` (этап 06): Linux→`2.12.0+cpu`, Mac→`2.12.0`; единый `uv.lock`.
   **Не возвращать** CUDA в Linux-резолв (раздул бы образ на гигабайты nvidia-пакетов).
6. **Образ самодостаточен** — ffmpeg+ffprobe встроены (apt). **Synology ffmpeg не имеет** → хостовые
   бинарники не использовать; всё внутри контейнера (`gigaam_api/audio.py` зовёт их из PATH образа).
7. **Прямой вызов `model.forward`/`model._decode`** (longform, `_iter_chunks`) **обязательно** в
   `torch.inference_mode()` — как `gigaam.transcribe`. Иначе автоград + inference-тензорный кэш rotary
   → `RuntimeError: Inference tensors cannot be saved for backward` (баг ловится только short→long на
   одном инстансе; тест `tests/integration/test_short_then_long_real.py`). Не убирать обёртку (важно
   для этапа 07: новые инференс-пути — тоже под `inference_mode`).
4. **MPS на Mac** может требовать `PYTORCH_ENABLE_MPS_FALLBACK=1` (GigaAM на MPS upstream не тестируют).
5. **Скорость на 4 ядрах:** 10ч аудио = часы счёта; сервис батчевый, не realtime. Длинные файлы — через `stream=true`.

## Команды (Makefile)

```
make install      # uv sync
make run          # локальный запуск (uvicorn --reload)
make download-weights-local  # прогрев весов нативно (uv, без Docker) в MODELS_DIR из .env
make check        # ruff + ruff format --check + mypy(strict) + pytest(unit) — быстрый цикл
make pre-commit   # ВСЯ пачка тестов всех типов подряд (check + integration). ← после КАЖДОЙ задачи, обязательно зелёный
make test         # юнит-тесты (без integration)
make test-integration  # реальная модель/сеть (маркер integration)
make build-docker / up / down / logs   # этап 06 (Docker, dev-удобство)
make download-weights  # прогрев весов через Docker (профиль tools)
```
> `make pre-commit` — это Makefile-цель, **не инструмент pre-commit**.

## Конвенции

- **Общение в проекте — на русском**; код/идентификаторы — английские (докстринги можно по-русски).
- **mypy strict**, **TDD** (тест → код), чистые функции для `formats`/VAD-чанкинга.
- **Тестируем прагматично, без оверинженеринга:** покрываем рисковую/ключевую логику и happy-path,
  не пишем тесты ради тестов и не гонимся за процентом покрытия.
- **YAGNI:** не пишем код «на всякий случай»; **удаляем неиспользуемый/мёртвый код сразу**
  (без закомментированных кусков). Осознанное исключение — абстракция `ASREngine` под ONNX.
- Никаких сетевых вызовов в импортах модулей; скачивание весов — только в lifespan.
- Логирование — stdlib `logging`, уровень из `LOG_LEVEL`; debug-логи в ключевых точках (master §8).
- **`CLAUDE.md` и `README.md` держим всегда актуальными:** изменилось поведение/команды/API/конфиг —
  правим оба в той же задаче. Новые архитектурные решения — в раздел «Архитектурные решения» выше.
- **После каждой задачи** и в конце каждого этапа — зелёный `make pre-commit` + обновлённый трекер.
- **Не коммитить без явной просьбы пользователя.**

## Изучение GigaAM

Склонированный репозиторий для изучения — в `tmp/GigaAM/` (не зависимость; зависимость ставится
из git, см. master §9). Полезные файлы для справки: `gigaam/model.py`, `gigaam/vad_utils.py`
(алгоритм чанкинга для порта), `gigaam/decoding.py`, `gigaam/__init__.py` (загрузка/кэш весов).
