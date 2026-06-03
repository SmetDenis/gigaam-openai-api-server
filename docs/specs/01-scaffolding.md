# 01 — Каркас и тулинг

> Этапный спек. Общий контекст и решения — в [`00-master.md`](./00-master.md).
> Самодостаточен: реализуется в отдельной сессии без контекста брейншторма.

## Цель
Создать скелет проекта: uv-проект на Python 3.12, строгий тулинг (ruff + mypy strict +
pytest), Makefile, структуру пакета `gigaam_api`, настройку логирования и минимальное
FastAPI-приложение с `/health`. По завершении `make check` и `make run` работают,
сервис поднимается и отвечает на `/health`. Тяжёлые ML-зависимости (torch, gigaam, silero)
на этом этапе **не подключаются** — они появятся на этапе 02.

## Предусловия
- Пустой репозиторий в `/Users/smetdenis/work/smetdenis/gigaam-api`.
- Установлены `uv` и (для запуска) `ffmpeg` (понадобится позже).
- Прочитан `00-master.md` (§4 структура, §7 конфиг, §8 логирование, §10 тестирование).

## Артефакты (создаваемые файлы)
```
.python-version                 # "3.12"
pyproject.toml                  # проект, зависимости (только лёгкие), конфиг ruff/mypy/pytest
uv.lock                         # сгенерирован uv
.gitignore
.env.example                    # все переменные из master §7 с дефолтами
Makefile
README.md                       # краткое описание + команды (черновик)
gigaam_api/__init__.py
gigaam_api/main.py              # FastAPI app + lifespan + /health роутер
gigaam_api/config.py            # pydantic-settings Settings (полный набор из master §7)
gigaam_api/logging_setup.py     # setup_logging(settings)
gigaam_api/api/__init__.py
gigaam_api/api/health.py        # GET /health
tests/__init__.py
tests/conftest.py
tests/unit/__init__.py
tests/unit/test_config.py
tests/unit/test_health.py
tests/unit/test_logging_setup.py
```

## Задачи
1. **uv-проект**: `pyproject.toml` с `requires-python = ">=3.12,<3.13"`, `.python-version = 3.12`.
   Рантайм-зависимости этого этапа: `fastapi`, `uvicorn[standard]`, `pydantic>=2`,
   `pydantic-settings`, `python-multipart`. Dev-группа: `ruff`, `mypy`, `pytest`,
   `pytest-cov`, `pytest-asyncio`, `httpx`.
2. **Конфиг тулинга** в `pyproject.toml`:
   - `[tool.ruff]`: `line-length = 100`, включить правила `E,F,I,UP,B,SIM,RUF`; `[tool.ruff.format]` дефолтный.
   - `[tool.mypy]`: `strict = true`, `python_version = "3.12"`, `warn_unused_ignores = true`,
     `disallow_untyped_defs = true`, `plugins = ["pydantic.mypy"]`.
   - `[tool.pytest.ini_options]`: `testpaths = ["tests"]`, `markers = ["integration: требует реальной модели/сети"]`,
     `addopts = "-q --strict-markers"`, по умолчанию интеграционные тесты исключены
     (например, `-m "not integration"` в Makefile-цели юнит-прогона).
3. **`config.py`**: класс `Settings(BaseSettings)` со ВСЕМИ переменными из master §7
   (правильные типы, дефолты, `model_config = SettingsConfigDict(env_file=".env", extra="ignore")`).
   - `DEVICE`, `LOG_LEVEL`, `DEFAULT_RESPONSE_FORMAT` — валидировать допустимые значения.
   - `ALLOWED_MODELS` — парсить из csv в `list[str]`.
   - Функция `get_settings() -> Settings` с `@lru_cache` (синглтон).
4. **`logging_setup.py`**: `setup_logging(settings: Settings) -> None` — конфигурирует root logger
   по `LOG_LEVEL`; при `LOG_JSON=true` — JSON-формат (минимальный, без внешних либ: свой
   `logging.Formatter`), иначе человекочитаемый `%(asctime)s %(levelname)s %(name)s %(message)s`.
   Идемпотентность (не плодить хендлеры при повторном вызове).
5. **`main.py`**: `create_app() -> FastAPI`; `lifespan` (пока пустой, заготовка под загрузку модели
   на этапе 02 — оставить TODO-комментарий со ссылкой на спек 02); подключить роутер health;
   вызвать `setup_logging`. Экспортировать `app = create_app()`.
6. **`api/health.py`**: `GET /health` → `{"status":"ok","model":<MODEL>,"device":<resolved>,"loaded":false}`
   (`loaded=false`, т.к. модели ещё нет; на этапе 02 станет реальным).
7. **`.env.example`**: все переменные master §7 с дефолтами и краткими комментариями.
8. **Makefile** (цели; см. ниже).
9. **Тесты**:
   - `test_config.py`: загрузка дефолтов; парсинг `ALLOWED_MODELS`; валидация неверного `DEVICE`/`LOG_LEVEL` (ошибка).
   - `test_health.py`: `httpx`/TestClient → `GET /health` == 200 и ожидаемый JSON.
   - `test_logging_setup.py`: повторный вызов не плодит хендлеры; уровень применяется.
10. **README.md** (черновик): назначение, требования (Python 3.12, uv, ffmpeg), команды Makefile.

## Makefile (цели этого этапа; расширяется на следующих)
| Цель | Действие |
|---|---|
| `install` | `uv sync` |
| `run` | `uv run uvicorn gigaam_api.main:app --host $(HOST) --port $(PORT) --reload` (dev) |
| `lint` | `uv run ruff check .` |
| `format` | `uv run ruff format .` |
| `format-check` | `uv run ruff format --check .` |
| `typecheck` | `uv run mypy gigaam_api tests` |
| `test` | `uv run pytest -m "not integration"` (юнит, быстро) |
| `test-integration` | `uv run pytest -m integration` (реальная модель/сеть) |
| `check` | `lint` + `format-check` + `typecheck` + `test` (быстрый внутренний цикл) |
| `pre-commit` | **вся пачка всех типов тестов один за другим**: `lint` → `format-check` → `typecheck` → `test` → `test-integration`. Вызывается ПОСЛЕ КАЖДОЙ ЗАДАЧИ. **Это не инструмент pre-commit, а Makefile-цель.** |
| `clean` | удалить `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `__pycache__` |

> Нюанс `pre-commit`: на этапе 01 интеграционных тестов ещё нет — `pytest -m integration`
> вернёт код 5 («не собрано тестов»). Цель должна трактовать «нет тестов» как успех
> (например, обернуть: `... || [ $$? -eq 5 ]`). По мере добавления тестов на этапах 02+
> `test-integration` наполняется.
>
> Остальные цели (`download-weights`, `build-docker`, `up`, `down`, `logs`)
> добавляются на этапе 06 — оставить в Makefile секцию-заготовку с комментарием.

`.gitignore` уже создан в корне репозитория (Python/venv/кэши/`.env`/`models/`/`tmp/`) —
на этом этапе только убедиться, что он на месте и корректен.

## Debug-логи (этот этап)
- На старте приложения (`lifespan`): `INFO` — версия, выбранные `MODEL`/`DEVICE`/`LOG_LEVEL`.
- `/health`: `DEBUG` — факт обращения.

## Acceptance-критерии
- [ ] `uv sync` устанавливает зависимости без ошибок на Python 3.12.
- [ ] `make run` поднимает сервис; `GET /health` → 200 с корректным JSON.
- [ ] `make check` и `make pre-commit` — зелёные (ruff, ruff format --check, mypy strict, pytest).
- [ ] mypy strict проходит без `type: ignore` в написанном коде.
- [ ] `.env.example` содержит все переменные из master §7.
- [ ] Структура каталогов соответствует master §4.1 (в объёме этого этапа).

## Definition of Done
Все acceptance-критерии + **общий DoD из master §14** (зелёный `make pre-commit`, обновлён трекер,
актуальны `CLAUDE.md`/`README.md`, нет «на всякий случай»-кода). Этап 01 → ✅ в трекере.
README отражает реальные команды. Код не подключает torch/gigaam/silero.
