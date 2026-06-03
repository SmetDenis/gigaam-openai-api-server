# Makefile — команды разработки GigaAM ASR.
# `make pre-commit` запускается ПОСЛЕ КАЖДОЙ ЗАДАЧИ и должен быть зелёным.
# Это Makefile-цель, а не инструмент pre-commit.

HOST ?= 0.0.0.0
PORT ?= 8000

.PHONY: install run lint format format-check typecheck test test-integration coverage check pre-commit clean

install:  ## Установить зависимости (uv sync)
	uv sync

run:  ## Локальный запуск сервиса (uvicorn --reload)
	uv run uvicorn gigaam_api.main:app --host $(HOST) --port $(PORT) --reload

lint:  ## ruff check
	uv run ruff check .

format:  ## ruff format (применить)
	uv run ruff format .

format-check:  ## ruff format --check
	uv run ruff format --check .

typecheck:  ## mypy (strict)
	uv run mypy gigaam_api tests

test:  ## Юнит-тесты (без integration)
	uv run pytest -m "not integration"

test-integration:  ## Интеграционные тесты (реальная модель/сеть). Код выхода 5 ("нет тестов") трактуем как успех.
	uv run pytest -m integration || [ $$? -eq 5 ]

coverage:  ## Отчёт покрытия (без гейта)
	uv run pytest -m "not integration" --cov=gigaam_api --cov-report=term-missing

check: lint format-check typecheck test  ## Быстрый цикл: lint + format-check + mypy + unit

pre-commit: lint format-check typecheck test test-integration  ## Вся пачка тестов всех типов подряд

clean:  ## Удалить кэши инструментов
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# --- Этап 06 (Docker/деплой) — заготовка, наполняется позже ---
# download-weights:  ## Прогрев весов модели в volume
# 	uv run python -m gigaam_api.download_weights
# build-docker:  ## Сборка образа (amd64)
# 	docker build -t gigaam-api .
# up:  ## docker compose up -d
# 	docker compose up -d
# down:  ## docker compose down
# 	docker compose down
# logs:  ## docker compose logs -f
# 	docker compose logs -f
