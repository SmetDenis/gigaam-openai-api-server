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

Этап **03 ✅** (longform >25с: Silero VAD (JIT) → чистая функция чанкинга `merge_intervals_to_chunks`
(порт из gigaam) → батчевый `model.forward`/`model._decode`; роутинг по длительности внутри движка;
int16-декод для экономии памяти; без pyannote). Этап **02 ✅** (ASR-движок PyTorch за `ASREngine`,
короткие аудио ≤25с, загрузка модели в lifespan, кэш весов в `MODELS_DIR`, `/health.loaded=true`).
Этап **01 ✅** (каркас, тулинг, логирование, FastAPI-скелет). Следующий — `04` (OpenAI-эндпоинты,
schemas, formats, auth, runner). Актуальный трекер — в `00-master.md` §13.

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

<!-- Новые решения добавляй новой строкой выше этой подсказки. -->


## Критичные предостережения (root-cause, не нарушать)

1. **Docker на Mac НЕ видит GPU.** Контейнеры идут в Linux-VM без Metal → только CPU. MPS на Mac
   доступен лишь при **нативном** запуске (uv), не в Docker. Прод = Synology CPU.
2. **НЕ вызывать `model.transcribe_longform`** — он тянет pyannote. Longform делаем сами через
   Silero VAD + порт чанкинга GigaAM (master §5.1). Нигде не должно быть `import pyannote`.
3. **torch в Docker** ставить из CPU-индекса (`download.pytorch.org/whl/cpu`), не тянуть CUDA (master §9, этап 06).
4. **MPS на Mac** может требовать `PYTORCH_ENABLE_MPS_FALLBACK=1` (GigaAM на MPS upstream не тестируют).
5. **Скорость на 4 ядрах:** 10ч аудио = часы счёта; сервис батчевый, не realtime. Длинные файлы — через `stream=true`.

## Команды (Makefile)

```
make install      # uv sync
make run          # локальный запуск (uvicorn --reload)
make check        # ruff + ruff format --check + mypy(strict) + pytest(unit) — быстрый цикл
make pre-commit   # ВСЯ пачка тестов всех типов подряд (check + integration). ← после КАЖДОЙ задачи, обязательно зелёный
make test         # юнит-тесты (без integration)
make test-integration  # реальная модель/сеть (маркер integration)
make build-docker / up / down / logs / download-weights   # этап 06
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
