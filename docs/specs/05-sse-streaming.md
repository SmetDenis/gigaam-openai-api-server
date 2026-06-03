# 05 — SSE-стриминг (`stream=true`)

> Этапный спек. Общий контекст — в [`00-master.md`](./00-master.md), особенно §6.4
> «Стриминг (SSE)». Самодостаточен.

## Цель
Реализовать прогрессивную отдачу транскрипта по Server-Sent Events при `stream=true` —
чтобы многочасовые файлы не упирались в таймауты клиента/прокси. Сегменты отдаются по мере
готовности (delta), в конце — финальное событие с полным текстом. Контракт совместим по форме
с OpenAI streaming transcription.

## Предусловия
- Завершён этап 04 (синхронный API, engine, runner, formats).
- Прочитан master §6.4 (формат SSE-событий) и §11 (сериализация инференса).

## Контракт SSE (фиксируем)
- `Content-Type: text/event-stream`, заголовки `Cache-Control: no-cache`, `Connection: keep-alive`.
- Поток событий (каждое — строка `data: <json>\n\n`):
  - на каждый готовый сегмент: `{"type":"transcript.text.delta","delta":"<текст сегмента>"}`
    (с завершающим пробелом-разделителем между сегментами);
  - в конце: `{"type":"transcript.text.done","text":"<полный текст>"}`;
  - затем терминатор: `data: [DONE]\n\n`.
- Поддерживается только для `response_format` ∈ {`json`, `text`}. Для `verbose_json`/`srt`/`vtt`
  при `stream=true` → **синхронный фоллбэк** (игнорируем `stream`, отдаём полный ответ).
  > **Уточнение по итогам реализации (этап 05):** исходно спек предписывал 400, но решено
  > отдавать синхронный фоллбэк — большинство OpenAI-клиентов шлют `stream=true` по умолчанию,
  > 400 ломал бы их на `verbose_json`. Предсказуемость сохраняется: эти форматы всегда дают полный ответ.
  > Контракт событий (`transcript.text.delta` → `transcript.text.done` → `[DONE]`) **проверен** против
  > официальной OpenAI Speech-to-text (06.2026) — совпадает дословно. Нюанс: реальный OpenAI стримит только
  > `json` (whisper-1 не стримит вовсе), поддержка `text` здесь — совместимое расширение.
- Ошибка в середине потока: отправить событие `{"type":"error","error":{...}}` (OpenAI-формат) и закрыть поток.

## Артефакты
```
gigaam_api/streaming.py           # сборка SSE-событий; генератор по сегментам
gigaam_api/asr/gigaam_engine.py   # +итеративный longform (yield сегментов по мере готовности)
gigaam_api/api/transcriptions.py  # ветка stream=true → StreamingResponse
tests/unit/test_streaming.py      # формат событий, терминатор, ошибка в потоке
tests/integration/test_stream_real.py  # маркер integration: реальный стрим длинного аудио
```

## Задачи
1. **Итеративный longform в engine**: добавить генератор
   `iter_segments(wav_path, *, word_timestamps) -> Iterator[SegmentTS]`, который выдаёт сегменты
   по мере обработки батчей (переиспользует логику этапа 03, но `yield` вместо накопления в список).
   Для короткого аудио (≤25с) — выдать один сегмент.
   - Блокирующий генератор исполняется в `Runner`-потоке; мост в async — через потокобезопасную
     очередь (`queue.Queue`) + `asyncio`-обёртка, либо `anyio.to_thread` с колбэком. Зафиксировать
     один подход; важно не блокировать event loop и сохранить сериализацию (1 инференс за раз).
2. **`streaming.py`**: `async def sse_transcription(segments_aiter, response_format) -> AsyncIterator[str]`:
   - на каждый сегмент → `delta`-событие (накапливать полный текст);
   - в конце → `done`-событие с полным текстом → `[DONE]`;
   - при исключении → `error`-событие → закрыть.
   - Хелпер `format_sse(data: dict | str) -> str`.
3. **`api/transcriptions.py`**: если `stream=true`:
   - проверить `response_format` ∈ {json,text} (иначе 400);
   - получить async-итератор сегментов (через engine.iter_segments в runner-потоке);
   - вернуть `fastapi.responses.StreamingResponse(sse_transcription(...), media_type="text/event-stream")`.
   - Синхронная ветка (этап 04) остаётся для `stream=false`.

## Тесты
- **unit** (`test_streaming.py`): подать фейковый async-итератор сегментов → проверить
  последовательность событий (`delta`×N → `done` → `[DONE]`), корректный полный текст,
  обработку исключения (событие `error` + закрытие). Проверить 400 для `verbose_json`+stream.
- **integration** (`test_stream_real.py`, маркер `integration`): реальный длинный пример,
  собрать события из потока → склеенный текст совпадает с синхронным результатом.

## Debug-логи (этот этап)
- старт стрима: `INFO` — `request_id`, `response_format`, режим stream.
- по каждому сегменту: `DEBUG` — индекс сегмента, длина delta.
- завершение: `INFO` — число сегментов, общий размер текста, общее время, RTF.
- ошибка в потоке: `exception`.

## Acceptance-критерии
- [ ] `stream=true` + `json`/`text` → корректный SSE-поток: `delta`-события, финальный `done`, `[DONE]`.
- [ ] Полный текст из `done` идентичен синхронному ответу на тот же файл.
- [ ] `stream=true` + `verbose_json`/`srt`/`vtt` → 400 с понятным сообщением.
- [ ] Event loop не блокируется; сериализация инференса сохранена (1 за раз).
- [ ] Ошибка в середине → `error`-событие + закрытие потока.
- [ ] `make pre-commit` зелёный.

## Definition of Done
Стриминг работает и протестирован; синхронный режим не сломан. Соблюдён **общий DoD из master §14**
(зелёный `make pre-commit`, трекер, актуальные `CLAUDE.md`/`README.md` — включая описание контракта SSE).
Этап 05 → ✅ в трекере.
