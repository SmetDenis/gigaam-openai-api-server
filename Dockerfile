# syntax=docker/dockerfile:1.7
# GigaAM ASR — self-hosted image (linux/amd64, CPU-only torch, NO weights in the image).
#
# The platform is set at build time (`docker build --platform linux/amd64 ...`, see Makefile),
# NOT hardcoded in FROM — this keeps the image multi-arch friendly, and on an amd64 host
# it builds natively. torch/torchaudio are installed as CPU wheels from the index
# download.pytorch.org/whl/cpu (via [tool.uv.sources] + the marker sys_platform=='linux'
# in pyproject.toml) — without CUDA/nvidia packages. GigaAM weights are downloaded on first start
# into the mounted volume (MODELS_DIR=/data/models).

# ---- builder: install dependencies into an isolated /app/.venv via uv ----
FROM python:3.12-slim AS builder

# uv from the official image (version pin = reproducible build).
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

# git — for the git dependency gigaam (uv clones the repository and builds the package).
RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && rm -rf /var/lib/apt/lists/*

# UV_COMPILE_BYTECODE — .pyc at build time (faster start); UV_LINK_MODE=copy — no hardlinks
# (the cache mount is on a different FS); UV_PYTHON_DOWNLOADS=0 — use the base image's system CPython.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# 1) Dependency layer: cached until pyproject.toml/uv.lock change.
#    --no-install-project — the package itself is installed below, as a separate thin layer (after the code).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 2) Application code + installation of the package itself.
COPY gigaam_api ./gigaam_api
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---- runtime: thin image with only ffmpeg + the ready-made .venv ----
FROM python:3.12-slim AS runtime

# ffmpeg/ffprobe are required (audio decode + duration probe); ca-certificates — for
# HTTPS download of weights from the CDN on first start.
RUN apt-get update \
    && apt-get install --no-install-recommends -y ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged user with a fixed UID/GID 1000. IMPORTANT: the host directory
# ./models, mounted into /data/models, must be owned by UID 1000 (chown 1000:1000),
# otherwise non-root cannot write weights to the volume. See README → "Deploy (Docker Compose)".
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin app

# PATH to .venv; MODELS_DIR — GigaAM weights cache (volume); XDG_CACHE_HOME in the volume — a safety
# net so that any accidental cache does not hit the non-root-inaccessible ~/.cache (Silero
# is bundled in the package, the project uses no HF Hub/torch.hub).
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/data/models \
    XDG_CACHE_HOME=/data/models/.cache

# Volume mount point; owner — app (UID 1000), so non-root can write weights.
RUN mkdir -p /data/models && chown -R 1000:1000 /data/models

WORKDIR /app
COPY --from=builder --chown=1000:1000 /app/.venv /app/.venv
COPY --from=builder --chown=1000:1000 /app/gigaam_api /app/gigaam_api

USER 1000
EXPOSE 8000

# Healthcheck without curl (it is not in slim) — a thin request to /health via stdlib urllib.
# start-period is generous: the first start may download weights (minutes on a slow link),
# until they are loaded uvicorn does not yet respond. In compose start_period is set separately.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD ["python", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).status==200 else 1)"]

# Production run: without --reload.
CMD ["uvicorn", "gigaam_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
