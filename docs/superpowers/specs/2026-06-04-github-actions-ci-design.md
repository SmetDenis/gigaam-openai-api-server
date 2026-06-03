# Дизайн: GitHub Actions CI (`.github/workflows/ci.yml`)

- **Дата:** 2026-06-04
- **Статус:** дизайн одобрен, ожидает ревью спеки
- **Прообраз:** `SmetDenis/ollama-to-openai/.github/workflows/ci.yml` (адаптирован под gigaam-api)

## Цель

Добавить CI на GitHub Actions: автоматические проверки кода (lint / format / type /
тесты) на каждый push в `master` и каждый PR, плюс сборка и публикация Docker-образа в
GitHub Container Registry (ghcr.io) для деплоя на Synology через `pull`.

## Решения (из обсуждения)

| Вопрос | Решение |
|---|---|
| Триггеры | `push` в `master` + `pull_request` (без фильтра таргет-ветки) |
| Объём проверок | **Все** проверки, **каждая отдельным шагом** (lint, format-check, typecheck, unit, integration); integration гоняется в основном workflow, без расписания |
| Docker-образ | Да — build & push в `ghcr.io` (теги `latest` + `sha`) |
| Кэш зависимостей | **Нет** (чистая установка каждый прогон — осознанный выбор) |
| Concurrency | Нет (без `concurrency`-логики) |

## Адаптации под проект (отличия от прообраза)

1. **Python 3.12**, а не 3.13 (`requires-python >=3.12,<3.13`, `.python-version`).
2. **Каждая проверка — отдельный шаг** через `make`-цели (`make lint`, `make format-check`,
   `make typecheck`, `make test`, `make test-integration`), а не один `make check`.
3. **Шаг установки ffmpeg** (`apt-get install -y ffmpeg`) — integration-тесты декодируют
   аудио через ffmpeg/ffprobe, а на `ubuntu-latest` они не гарантированы в PATH. Без этого
   шага integration упал бы на декоде (root-cause, в прообразе шага нет).
4. **Docker job — только push в master** (`if: github.event_name == 'push' && ...`): на PR
   образ не пушим (нет смысла + `GITHUB_TOKEN` форков не имеет прав `packages: write`).
5. **`packages: write`** выдаётся точечно на docker job, а не глобально (минимум привилегий).

## Версии actions (проверены по releases на 2026-06-04)

| Action | Тег | Примечание |
|---|---|---|
| `actions/checkout` | `@v6` | |
| `actions/setup-python` | `@v6` | |
| `astral-sh/setup-uv` | `@v8.2.0` | **immutable-теги** с v8.0.0 — `@v8`/`@v8.0` НЕ резолвятся, нужен полный тег |
| `docker/login-action` | `@v4` | последний v4.2.0, moving `@v4` доступен |
| `docker/metadata-action` | `@v6` | последний v6.1.0, moving `@v6` доступен |
| `docker/build-push-action` | `@v7` | последний major |

## Поведение

- Шаги в job `check` идут **последовательно, со стопом на первой ошибке** (стандартное
  fail-fast). Если lint упал — тяжёлый integration и Docker не запустятся (экономия времени).
- `docker` имеет `needs: check` — публикуется только при зелёных проверках.
- Образ собирается **нативно на amd64**-раннере (без эмуляции, в отличие от dev-Mac).
  Из-за torch CPU образ тяжёлый (несколько ГБ) — push заметен по времени.
- `metadata-action` нормализует имя образа в lowercase для ghcr
  (`SmetDenis/gigaam-api` → `ghcr.io/smetdenis/gigaam-api`).

## Полный файл `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [master]
  pull_request:

permissions:
  contents: read

jobs:
  check:
    name: Lint, Type Check & Test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6

      - name: Set up Python 3.12
        uses: actions/setup-python@v6
        with:
          python-version: "3.12"

      - name: Install uv
        uses: astral-sh/setup-uv@v8.2.0

      - name: Install ffmpeg (audio decode for integration tests)
        run: sudo apt-get update && sudo apt-get install -y ffmpeg

      - name: Install dependencies
        run: uv sync --frozen

      - name: Lint (ruff check)
        run: make lint

      - name: Format check (ruff format --check)
        run: make format-check

      - name: Type check (mypy strict)
        run: make typecheck

      - name: Unit tests
        run: make test

      - name: Integration tests (real model & weights)
        run: make test-integration

  docker:
    name: Build & Push Docker Image
    needs: check
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/master'
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v6

      - name: Log in to ghcr.io
        uses: docker/login-action@v4
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Docker metadata (tags & labels)
        id: meta
        uses: docker/metadata-action@v6
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=raw,value=latest
            type=sha,prefix=

      - name: Build & push
        uses: docker/build-push-action@v7
        with:
          context: .
          platforms: linux/amd64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
```

## Вне scope

- Кэширование зависимостей / Docker-слоёв (осознанно отключено).
- Матрица Python-версий (проект пинит одну — 3.12).
- Публикация по релизным тегам `v*` (можно добавить позже в `metadata-action`).
- Сканирование безопасности образа, multi-arch (только `linux/amd64` под Synology).

## Открытые вопросы для деплоя (не входят в этот файл, на будущее)

- В настройках GitHub-репозитория после первого push образа сделать пакет видимым/привязать
  к репозиторию (ghcr packages). Может потребоваться `Settings → Actions → Workflow
  permissions → Read and write` либо это покрывается job-level `packages: write`.
```
