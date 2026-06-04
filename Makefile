# Makefile — GigaAM ASR development commands.
# `make pre-commit` runs AFTER EVERY TASK and must be green.
# This is a Makefile target, not the pre-commit tool.

HOST ?= 0.0.0.0
PORT ?= 8000

IMAGE ?= gigaam-api:latest

.PHONY: install run download-weights-local lint format format-check typecheck test test-integration coverage check pre-commit clean \
        build-docker up down logs download-weights

install:  ## Install dependencies (uv sync)
	uv sync

run:  ## Local service run (uvicorn --reload)
	uv run uvicorn gigaam_api.main:app --host $(HOST) --port $(PORT) --reload

download-weights-local:  ## Warm up weights natively (uv, no Docker) into MODELS_DIR from .env
	uv run python -m gigaam_api.download_weights

lint:  ## ruff check
	uv run ruff check .

format:  ## ruff format (apply)
	uv run ruff format .

format-check:  ## ruff format --check
	uv run ruff format --check .

typecheck:  ## mypy (strict)
	uv run mypy gigaam_api tests

test:  ## Unit tests (no integration)
	uv run pytest -m "not integration"

test-integration:  ## Integration tests (real model/network). Exit code 5 ("no tests") is treated as success.
	uv run pytest -m integration || [ $$? -eq 5 ]

coverage:  ## Coverage report (no gate)
	uv run pytest -m "not integration" --cov=gigaam_api --cov-report=term-missing

check: lint format-check typecheck test  ## Fast loop: lint + format-check + mypy + unit

pre-commit: lint format-check typecheck test test-integration  ## The whole batch of all test types in a row

clean:  ## Remove tool caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# --- Docker/deployment (stage 06) ---
# A convenience for development on Mac. In production, deployment goes through docker-compose.yml +
# `docker compose` (no make). The production image is linux/amd64 (built on Mac via
# emulation; native on an amd64 host).

build-docker:  ## Build the production image (linux/amd64)
	docker build --platform linux/amd64 -t $(IMAGE) .

up:  ## docker compose up -d (start the service)
	docker compose up -d

down:  ## docker compose down (stop the service)
	docker compose down

logs:  ## docker compose logs -f (service logs)
	docker compose logs -f

download-weights:  ## Warm up weights into ./models (a one-off container, no service startup)
	docker compose --profile tools run --rm download-weights
