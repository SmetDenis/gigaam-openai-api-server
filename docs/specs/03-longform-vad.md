# 03 — Длинные аудио: Silero VAD + чанкинг + longform-цикл

> Этапный спек. Общий контекст — в [`00-master.md`](./00-master.md), особенно
> §5.1 «Longform без pyannote». Самодостаточен.

## Цель
Реализовать распознавание аудио **длиннее 25с** (вплоть до ~10ч): нарезка на сегменты через
**Silero VAD** + алгоритм чанкинга (порт из GigaAM, без pyannote), батчевый инференс и склейку
сегментов с корректным пересчётом таймстемпов. Контроль памяти под 8 ГБ RAM. По завершении
`GigaAMEngine` умеет и short, и longform; роутинг по длительности — внутри engine.

## Предусловия
- Завершён этап 02 (engine, short audio, типы `ASRResult/SegmentTS/WordTS`, `audio.py`).
- В зависимостях есть `silero-vad` (добавлен на этапе 02).
- Прочитан master §5.1 (приватные методы `model.forward`/`model._decode`), §11 (память).

## Фон: исходный алгоритм GigaAM (что портируем)
Файл `gigaam/vad_utils.py::segment_audio_file` (см. репозиторий GigaAM):
- получает речевые интервалы от pyannote: `pipeline(wav).get_timeline().support()` — список `(start,end)` в секундах;
- сливает их в чанки по правилам: `max_duration=22.0`, `min_duration=15.0`,
  `strict_limit_duration=30.0`, `new_chunk_threshold=0.2`; чанки длиннее `strict_limit` режутся на равные части;
- возвращает `(segments: list[Tensor], boundaries: list[(start,end)])`, где `segments` — срезы waveform.

`GigaAMASR.transcribe_longform` затем батчит сегменты (`DataLoader` + `AudioDataset.collate`),
вызывает `model.forward(wav_pad, wav_lens)` → `model._decode(...)` и собирает `Segment`-ы,
прибавляя `seg_start` к таймстемпам слов.

**Мы заменяем только источник интервалов (pyannote → Silero). Алгоритм слияния и батчевый цикл — сохраняем.**

## Артефакты
```
gigaam_api/asr/vad.py             # Silero VAD + чанкинг (чистые функции, где возможно)
gigaam_api/asr/gigaam_engine.py   # +transcribe_longform; роутинг в transcribe по длительности
gigaam_api/audio.py               # потоковая/чанковая конвертация в 16kHz mono int16
tests/unit/test_vad_chunking.py   # алгоритм слияния на синтетических интервалах
tests/unit/test_engine_longform.py# longform с моками VAD и модели
tests/integration/test_longform_real.py  # маркер integration, реальная модель + длинный пример
```

## Задачи
1. **Загрузка Silero** (`vad.py`): через pip-пакет — `from silero_vad import load_silero_vad, get_speech_timestamps`.
   - `load_silero_vad()` — в свежих версиях пакета веса **бандлятся внутри пакета**, сеть и отдельный
     кэш не нужны (проверить для зафиксированной версии). Если конкретная версия всё же качает через
     torch.hub — направить кэш в `MODELS_DIR`.
   - Загружать один раз (как и GigaAM-модель), хранить в engine; не в импортах.
2. **Получение речевых интервалов** (`vad.py`):
   - `speech_intervals(wav_16k_mono, settings) -> list[tuple[float, float]]` (секунды), через
     `get_speech_timestamps(wav, model, sampling_rate=16000, threshold=settings.VAD_THRESHOLD, return_seconds=True)` →
     привести к списку `(start, end)`.
   - **⚠️ Память (проверено):** `get_speech_timestamps` принимает **весь** сигнал как float32 (≈ 2.3 ГБ/10ч)
     и на длинных файлах у Silero известен рост памяти (issue #356). Поэтому: подать float только на время VAD
     и сразу освободить (инференс потом — батчами из int16-буфера). Для очень длинных файлов рассмотреть
     потоковый `VADIterator` (чанки по 512 сэмплов), не загружая весь float. Не утверждать «float только батч».
3. **Алгоритм чанкинга** (`vad.py`, **чистая функция**, отдельно тестируемая):
   - `merge_intervals_to_chunks(intervals, audio_duration, *, min_duration, max_duration, strict_limit, new_chunk_threshold) -> list[tuple[float, float]]` (границы чанков в секундах).
   - Портировать ровно логику `_update_segments`/циклов из `segment_audio_file` (включая разрезание сверхдлинных чанков на равные части). Это чистая функция от списка интервалов и параметров — её и тестируем синтетикой.
4. **`audio.py`**: `decode_to_int16_16k_mono(path) -> np.ndarray|Tensor` (через ffmpeg, как в gigaam `load_audio`,
   но возвращаем int16, чтобы экономить память; во float конвертируем **по чанку** при формировании батча).
   Документировать пиковую память (master §11).
5. **`transcribe_longform`** в `GigaAMEngine`:
   - декодировать аудио в int16 16k mono (`audio.py`);
   - `intervals = speech_intervals(...)`; `chunks = merge_intervals_to_chunks(...)`;
   - если чанков нет → вернуть `ASRResult(text="", duration, segments=[])`;
   - батчами по `settings.BATCH_SIZE`: нарезать срезы waveform по границам, конвертировать в float,
     `collate` (паддинг + lengths), `model.forward(wav_pad, wav_lens)` → `model._decode(..., word_timestamps)`;
   - собрать `SegmentTS` для каждого чанка: `start/end` = границы чанка; слова — со сдвигом `+seg_start` и округлением до 3 знаков (как в upstream);
   - вернуть `ASRResult(text=" ".join(seg.text), duration, segments)`.
6. **Роутинг** в `GigaAMEngine.transcribe(...)`: длительность ≤25с → старый short-путь;
   иначе → `transcribe_longform`. Убрать `AudioTooLongError` из обычного пути (оставить только при
   превышении `MAX_AUDIO_SECONDS`).
7. **Память/устройство**: батч переносить на device+dtype как в upstream; на cpu без autocast.
   Профиль памяти (проверено): держим int16 целиком (~1.15 ГБ/10ч); для **инференса** float — только текущий батч.
   **Но пик float возникает раньше — на VAD-стадии** (весь сигнал float ≈ 2.3 ГБ/10ч, см. задачу 2), его надо
   освободить до старта батчей. Реальный пик ≈ веса + int16 + float-VAD (не «только батч»).

## Тесты
- **unit** (`test_vad_chunking.py`): на синтетических интервалах проверить — слияние коротких в чанк ≥min,
  не превышение max в типовых случаях, разрезание чанка > strict_limit на равные части, граничные случаи
  (пустой вход, один интервал, тишина в конце). Это самый важный юнит-тест этапа.
- **unit** (`test_engine_longform.py`): мок `speech_intervals` (вернуть фикс. интервалы) и мок модели
  (`forward`/`_decode` через monkeypatch) → проверить сборку `ASRResult`, корректный сдвиг таймстемпов слов,
  склейку текста, обработку «нет речи».
- **integration** (`test_longform_real.py`, маркер `integration`): реальная модель + длинный пример;
  проверить >1 сегмента, монотонность границ, непустой текст. **Не использовать
  `gigaam.utils.download_long_audio()`** (wget + запись в CWD, см. этап 02) — качать через `httpx`/`urllib`
  в `tmp` с cleanup или держать локальный сэмпл.

## Debug-логи (этот этап)
- VAD: `INFO` — число речевых интервалов, суммарная длительность речи, число чанков; `DEBUG` — границы чанков.
- longform: `INFO` — прогресс `batch i/N`, число сэмплов в батче, время на батч; итог — общее время, RTF, число сегментов/слов.
- decode: `DEBUG` — длительность, размер int16-буфера (МБ).

## Acceptance-критерии
- [ ] Аудио >25с распознаётся целиком; результат — несколько сегментов с монотонными границами.
- [ ] Таймстемпы слов глобальны (сдвинуты на начало чанка), округлены до 3 знаков.
- [ ] `merge_intervals_to_chunks` покрыт юнит-тестами (включая разрезание сверхдлинных и краевые случаи).
- [ ] pyannote НЕ импортируется нигде (проверяемо: нет `import pyannote`).
- [ ] Пиковая память на длинном файле в рамках 8 ГБ (учтён float-пик VAD-стадии, не только батч; замерить на реальном ~10ч).
- [ ] `make pre-commit` зелёный; интеграционный longform-тест проходит локально.

## Definition of Done
Engine поддерживает short + longform, роутинг по длительности внутри engine; VAD-чанкинг
протестирован. Соблюдён **общий DoD из master §14** (зелёный `make pre-commit`, трекер,
актуальные `CLAUDE.md`/`README.md`). Этап 03 → ✅ в трекере.
