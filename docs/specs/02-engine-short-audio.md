# 02 — ASR-движок (PyTorch) + короткие аудио (≤25с)

> Этапный спек. Общий контекст и решения — в [`00-master.md`](./00-master.md)
> (особенно §5 «Интеграция с GigaAM» и §9 «Зависимости»). Самодостаточен.

## Цель
Подключить GigaAM (PyTorch) за абстракцией `ASREngine`, реализовать загрузку модели один раз,
скачивание/кэш весов в `MODELS_DIR`, загрузку аудио через ffmpeg и распознавание **коротких**
аудио (≤25с) через `model.transcribe`. Longform (этап 03) и HTTP-эндпоинты (этап 04)
здесь **не** делаются. Цель — рабочее ядро инференса с юнит-тестами на моках и
интеграционным тестом на реальной модели.

## Предусловия
- Завершён этап 01 (каркас, config, logging, `/health`).
- Доступен `ffmpeg` в PATH.
- Прочитан master §5 (API GigaAM, longform-факты), §7 (конфиг), §11 (память).

## Артефакты
```
gigaam_api/asr/__init__.py
gigaam_api/asr/engine.py          # Protocol ASREngine + типы результата
gigaam_api/asr/gigaam_engine.py   # PyTorch-реализация (short audio)
gigaam_api/audio.py               # ffmpeg-загрузка, probe длительности
gigaam_api/main.py                # обновить lifespan: грузить модель, /health.loaded=true
gigaam_api/api/health.py          # отражать реальный loaded/model/device
tests/unit/test_engine_short.py   # с моком gigaam-модели
tests/unit/test_audio.py
tests/integration/__init__.py
tests/integration/test_engine_real.py  # маркер integration, реальная модель
pyproject.toml                    # +torch, +torchaudio, +gigaam(git pin), +silero-vad
```

## Задачи
1. **Зависимости**: добавить `torch>=2.6`, `torchaudio>=2.6`,
   `gigaam @ git+https://github.com/salute-developers/GigaAM.git@<PINNED_REV>`, `silero-vad`.
   - Зафиксировать конкретную ревизию gigaam (тег/commit). Проверить, что колёса torch и
     зависимостей gigaam (onnx/onnxruntime/numpy и т.д.) ставятся на Python 3.12 (Mac arm64 и Linux amd64).
   - `silero-vad` добавляем уже здесь (используется на этапе 03), но интегрируем на 03.
2. **Типы результата** (`engine.py`) — собственные dataclass'ы (не зависящие от gigaam),
   чтобы HTTP/format-слой не импортировал gigaam:
   ```python
   @dataclass(frozen=True)
   class WordTS:    text: str; start: float; end: float
   @dataclass(frozen=True)
   class SegmentTS: text: str; start: float; end: float; words: list[WordTS] | None = None
   @dataclass(frozen=True)
   class ASRResult: text: str; duration: float; segments: list[SegmentTS]
   ```
   Для короткого аудио `segments` = один сегмент `[0, duration]`.
3. **Протокол** (`engine.py`):
   ```python
   class ASREngine(Protocol):
       model_name: str
       device: str
       def transcribe(self, wav_path: str, *, word_timestamps: bool) -> ASRResult: ...
       # longform добавит этап 03 (может быть отдельный метод transcribe_longform)
   ```
4. **`audio.py`**:
   - `probe_duration(path: str) -> float` — длительность в секундах (через `ffprobe` или `soundfile`; ffprobe надёжнее для любых форматов).
   - `decode_to_pcm16(path) -> ...` — заготовка потоковой/чанковой загрузки (понадобится на 03; здесь минимум — probe). Документировать, что короткий путь делегирует декод самому gigaam (`model.transcribe` сам зовёт ffmpeg).
   - Явно ловить ошибки ffmpeg → собственное исключение `AudioDecodeError`.
5. **`gigaam_engine.py`**:
   - `class GigaAMEngine` с конструктором `__init__(self, settings: Settings)`:
     - резолв устройства из `DEVICE` (`auto`: cuda→mps→cpu);
     - `torch.set_num_threads(settings.NUM_THREADS)` для cpu;
     - если `DEVICE` резолвится в `mps` — задокументировать установку `PYTORCH_ENABLE_MPS_FALLBACK=1` (см. master §12);
     - `gigaam.load_model(settings.MODEL, device=<resolved>, download_root=settings.MODELS_DIR)`;
     - `(int8: на этапе 07; здесь только заготовка-проверка флага)`.
   - `transcribe(wav_path, *, word_timestamps)`:
     - probe длительности; **если > 25с → бросить `AudioTooLongError`** (longform будет на этапе 03; пока явная ошибка с понятным текстом). Не падать с сырым `ValueError` gigaam.
     - вызвать `model.transcribe(wav_path, word_timestamps=word_timestamps)`;
     - сконвертировать `TranscriptionResult` → `ASRResult` (один сегмент `[0, duration]`,
       слова → `WordTS`).
   - Метод `info() -> dict` для `/health` (`model`, `device`, `loaded`).
6. **`main.py` lifespan**: создать `GigaAMEngine` (скачивание весов при первом старте),
   положить в `app.state.engine`; на shutdown — освободить. `/health.loaded=true` после успешной загрузки.
   На ошибке загрузки — лог `exception`, приложение не стартует (fail fast).
7. **`/health`**: брать данные из `app.state.engine.info()`.

## Тесты
- **unit** (`test_engine_short.py`): подменить `gigaam.load_model` моком, возвращающим
  фейковый `TranscriptionResult`; проверить маппинг в `ASRResult`, поведение `word_timestamps`,
  выброс `AudioTooLongError` при duration>25с (замокать `probe_duration`).
- **unit** (`test_audio.py`): `probe_duration` на крошечном сгенерированном wav; `AudioDecodeError` на битом вводе.
- **integration** (`test_engine_real.py`, маркер `integration`): реально загрузить лёгкую модель
  (например `v3_ctc`) и распознать короткий пример; проверить непустой текст и наличие слов при
  `word_timestamps=True`. Помечен так, чтобы НЕ запускаться в обычном `make test`.
  - **⚠️ Не использовать `gigaam.utils.download_short_audio()`** (проверено: `utils.py` зовёт
    `os.system('wget …')` и пишет в CWD; `wget` на macOS по умолчанию отсутствует → тест упадёт на dev).
    Держать маленький сэмпл в `tests/integration/data/` (в репозитории) или качать через `httpx`/`urllib`
    в `tmp`-каталог с cleanup.

## Debug-логи (этот этап)
- загрузка модели: `INFO` — имя, device, путь кэша; время загрузки; факт скачивания vs кэш-хит (если различимо).
- `transcribe`: `DEBUG` — `wav_path`, probe-длительность, `word_timestamps`; `INFO` — время инференса, RTF, длина текста.
- ошибки декода/слишком длинного файла — `warning`/`exception` с понятным сообщением.

## Acceptance-критерии
- [ ] Модель грузится один раз в lifespan; `/health` показывает `loaded=true`, верные `model`/`device`.
- [ ] `transcribe` на коротком аудио возвращает корректный `ASRResult` (текст + один сегмент; слова при запросе).
- [ ] Файл >25с → `AudioTooLongError` (не сырой ValueError).
- [ ] Веса кэшируются в `MODELS_DIR`; повторный старт не перекачивает.
- [ ] `make pre-commit` зелёный; mypy strict проходит (включая типы вокруг torch — где нужно, локальные `cast`/обёртки, минимум `ignore` с обоснованием).
- [ ] Интеграционный тест проходит локально при наличии сети/весов; включён в `make test-integration`.

## Definition of Done
Ядро инференса коротких аудио работает и протестировано; lifespan грузит модель.
Соблюдён **общий DoD из master §14** (зелёный `make pre-commit`, трекер, актуальные `CLAUDE.md`/`README.md`).
Этап 02 → ✅ в трекере. Зафиксированная ревизия `gigaam` и заметка о совместимости torch/Python 3.12 —
внесены в `CLAUDE.md` (раздел «Архитектурные решения») и README.
