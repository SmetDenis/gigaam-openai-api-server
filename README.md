# GigaAM ASR — OpenAI-compatible Russian speech recognition service

> **Languages:** **English** · [Русский](README_ru.md)

A self-hosted ASR server built on [GigaAM](https://github.com/salute-developers/GigaAM),
exposing an **OpenAI-compatible** API (`POST /v1/audio/transcriptions`). Any client that can talk to
a "Custom OpenAI Compatible Whisper Provider" works with this service unchanged.

The goal is a self-hosted OpenAI-compatible server for personal use: it runs as a **Docker** container
(CPU) on any host — a Linux server, NAS, or mini-PC — while development happens on **macOS** (natively
via uv, optionally MPS).

## Features

- `POST /v1/audio/transcriptions` (multipart) — synchronous: one request → full transcript.
- Short (≤25s) and long (up to ~10h) audio: long files are split via **Silero VAD** + chunking and
  processed in batches (no pyannote).
- Response formats: `json`, `text`, `verbose_json`, `srt`, `vtt`.
- Word- and segment-level timestamps (`timestamp_granularities[]`).
- Optional **`stream=true`** (SSE) — progressive delivery for `json`/`text` (protection against
  timeouts on multi-hour files).
- `GET /v1/models`, `GET /health`.
- **Bearer authentication** (single shared key from `.env`).
- Any input audio format supported by ffmpeg.

**Out of scope:** translation (`/v1/audio/translations` — GigaAM is Russian ASR only), WebSocket/realtime,
diarization (single speaker).

## Requirements

- **Host hardware (production).** CPU-only, no GPU required. Minimum **2 CPU cores / 2 GB RAM**;
  recommended **4 cores / 4–8 GB RAM** (depending on load and audio length — long files need more).
- **Python 3.12** (see `.python-version`) — for native dev runs.
- **[uv](https://docs.astral.sh/uv/)** — package and environment manager (dev).
- **ffmpeg** (with `ffprobe`) — required: audio decoding and duration probing. **The Docker image
  already bundles ffmpeg+ffprobe** (installed from apt) — the container is self-contained, nothing
  needs to be installed on the host. Only for native dev runs on macOS install ffmpeg on your system
  (`brew install ffmpeg`).
- **Docker** + **Docker Compose** — for self-hosted deployment. On a dev Mac, Docker is optional.

---

## Quick start

### Dev (macOS, native via uv)

```bash
make install              # uv sync — install dependencies
cp .env.example .env       # edit as needed (see the note on MODELS_DIR on Mac below)
make run                  # uvicorn --reload on http://localhost:8000
```

Health check:

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"v3_ctc","device":"mps","loaded":true}
```

> **MODELS_DIR on Mac.** The default `MODELS_DIR=/data/models` is meant for the container and is not
> writable on macOS. For native dev set an accessible path in `.env`, e.g. `MODELS_DIR=./models`
> or `MODELS_DIR=~/.cache/gigaam`.

> The model is loaded once at startup (lifespan); the first start downloads the weights into
> `MODELS_DIR` (on dev — from the CDN, takes minutes). `device` is the **resolution** of `DEVICE`
> (`auto` → cuda→mps→cpu): on a dev Mac `auto`→`mps`, in a Docker deployment→`cpu`. On MPS errors —
> `PYTORCH_ENABLE_MPS_FALLBACK=1`.

### Production (Docker Compose)

Full instructions are in the [Deploy (Docker Compose)](#deploy-docker-compose) section. In short: place
`docker-compose.yml` + `Dockerfile` + `.env` on the host, create a `./models` directory (owned by
UID 1000), and bring the project up with `docker compose up -d`. The first start downloads the weights →
the healthcheck becomes `healthy`.

---

## API (OpenAI-compatible)

Base URL: `http://<host>:8000/v1`. Authentication — the `Authorization: Bearer <API_KEY>` header
(if `API_KEY` in `.env` is empty, auth is disabled).

### `POST /v1/audio/transcriptions` (multipart/form-data)

| Field | Behavior |
|---|---|
| `file` | **Required.** Any format supported by ffmpeg. |
| `model` | Validated against `ALLOWED_MODELS` (otherwise `400`). The model actually loaded by the service is used. |
| `response_format` | `json` (default) · `text` · `verbose_json` · `srt` · `vtt`. |
| `timestamp_granularities[]` | `segment` and/or `word` (controls presence of `segments`/`words` in `verbose_json`). |
| `language` | Accepted; GigaAM is RU-only, has no effect on inference. |
| `stream` | `true` → **SSE streaming** for `json`/`text`; for `verbose_json`/`srt`/`vtt` — synchronous fallback (full response). |
| `prompt`, `temperature` | **Accepted and ignored** (greedy decoding; prompt is not supported). |

#### Examples (`curl`)

`json` (default) → `{"text":"..."}`:

```bash
curl -s http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@audio.mp3 \
  -F model=v3_ctc
```

`text` → plain text:

```bash
curl -s http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@audio.mp3 -F model=v3_ctc -F response_format=text
```

`verbose_json` with word timestamps (`segments`/`words` fields):

```bash
curl -s http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@audio.mp3 -F model=v3_ctc \
  -F response_format=verbose_json \
  -F "timestamp_granularities[]=segment" \
  -F "timestamp_granularities[]=word"
```

`srt` / `vtt` → subtitles:

```bash
curl -s http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@audio.mp3 -F model=v3_ctc -F response_format=srt
```

> In `verbose_json` the fields `tokens`/`avg_logprob`/`no_speech_prob`/`temperature` = `0.0`, `seek` = `0`
> (GigaAM does not provide them — best-effort, safe for Whisper client thresholds); `compression_ratio`
> is computed honestly (`len(b)/len(zlib.compress(b))`, in bytes).

#### Example (`openai` Python SDK)

The service is configured as a "Custom OpenAI Compatible Whisper Provider": `base_url`, `api_key`, `model`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-key")
with open("audio.mp3", "rb") as f:
    print(client.audio.transcriptions.create(file=f, model="v3_ctc").text)
```

### Streaming (`stream=true`, SSE)

Progressive delivery of the transcript via [Server-Sent Events](https://developer.mozilla.org/docs/Web/API/Server-sent_events)
— so multi-hour files don't hit client/proxy timeouts. Supported for `response_format` `json`/`text`;
for `verbose_json`/`srt`/`vtt` `stream` is ignored (synchronous full response — these formats require
the whole result).

- `Content-Type: text/event-stream` (+ `Cache-Control: no-cache`, `Connection: keep-alive`).
- For each ready segment: `data: {"type":"transcript.text.delta","delta":"<text chunk>"}`.
- At the end: `data: {"type":"transcript.text.done","text":"<full text>"}`, then `data: [DONE]`.
- Error mid-stream: `data: {"type":"error","error":{...}}` and the stream closes (no `[DONE]`).
- While a batch is being computed, an SSE comment `: keep-alive` is sent periodically (every ~15s) —
  it keeps the connection alive against proxy idle timeouts (on CPU a single batch can take minutes).

**Invariant:** the concatenation of all `delta`s exactly equals `done.text` and is **identical to the
synchronous** response for the same file (the separating space moves to the start of the next `delta`).

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-key")
with open("audio.mp3", "rb") as f:
    stream = client.audio.transcriptions.create(file=f, model="v3_ctc", stream=True)
    for event in stream:
        if event.type == "transcript.text.delta":
            print(event.delta, end="", flush=True)
```

### `GET /v1/models`

A list from `ALLOWED_MODELS` in OpenAI format:
`{"object":"list","data":[{"id":"v3_ctc","object":"model","owned_by":"gigaam"}, ...]}`.

### `GET /health`

`{"status":"ok","model":"<MODEL>","device":"<cpu|mps|cuda>","loaded":true}`. Used as the
Docker `HEALTHCHECK`.

### Errors

OpenAI format `{"error":{message,type,param,code}}`. Codes:

| Code | When |
|---|---|
| `400` | Corrupted/unsupported file, duration limit exceeded (`MAX_AUDIO_SECONDS`), invalid parameters. |
| `401` | Missing/invalid Bearer key. |
| `413` | `MAX_UPLOAD_MB` exceeded. |
| `500` | Internal error / ffmpeg not available in PATH. |
| `503` | Inference queue is full (`MAX_QUEUE`). |

### Concurrency and cancellation

Inference is serialized (`Runner`, a single worker); requests beyond `MAX_QUEUE` → `503`. When a client
disconnects during long audio, inference is cooperatively interrupted between batches (the short path
≤25s is not cancelled).

### Models (reference)

The service loads **one** model (`MODEL` in `.env`); the `model` field in requests is validated against
`ALLOWED_MODELS` (the loaded model is actually used).

**Recommended — generation v3 ASR** (newest, best quality):

| Model | Decoder | Case/punctuation | When to choose |
|---|---|---|---|
| `v3_ctc` (default) | CTC | lowercase, no punctuation | faster on CPU — the default choice for CPU hosts |
| `v3_e2e_ctc` | CTC | punctuation + case normalization | when you need readability (punctuation, case) at CTC speed |
| `v3_rnnt` | RNN-T | lowercase, no punctuation | higher accuracy, slower on CPU |
| `v3_e2e_rnnt` | RNN-T | punctuation + case normalization | maximum quality + punctuation, slowest of all |

- **CTC vs RNN-T:** CTC is faster on CPU (recommended on modest CPUs, e.g. ~4 cores); RNN-T is more accurate but slower.
- **e2e** (only available in v3): adds punctuation and case normalization directly to the output; the plain
  ones produce "raw" lowercase without punctuation.

**Also downloaded by gigaam** (for switching `MODEL`): `v1_ctc`/`v1_rnnt`, `v2_ctc`/`v2_rnnt` — older
ASR generations (e2e variants exist only in v3). The `*_ssl` (embedding encoders) and `emo` (emotion)
models are **not supported** by the service — they are not transcription (they have no `model.transcribe`).

**Switching the model:** set `MODEL=<name>` in `.env` (optionally add the name to `ALLOWED_MODELS` so the
client can send it in the `model` field) and restart the container (`docker compose up -d`). The first
request downloads the new weights into `./models` (volume); previously downloaded weights are kept —
switching without re-downloading.

---

## Configuration (`.env`)

All settings are read from `.env` (`pydantic-settings`). Example — `.env.example`.

| Variable | Type | Default | Purpose |
|---|---|---|---|
| `MODEL` | str | `v3_ctc` | GigaAM model name (must be in `ALLOWED_MODELS`). |
| `DEVICE` | str | `auto` | `auto`\|`cpu`\|`mps`\|`cuda`. `auto`: cuda→mps→cpu. |
| `API_KEY` | str | `""` | Bearer key. Empty → auth disabled (set it in production). |
| `MODELS_DIR` | path | `/data/models` | Weights cache directory (volume). On Mac set a local path. |
| `QUANTIZE_INT8` | bool | `false` | Dynamic int8 quantization (stage 07; ignored for now). |
| `BATCH_SIZE` | int | `4` | Longform inference batch size. |
| `NUM_THREADS` | int | `4` | `torch.set_num_threads` (keep ≤ the number of cores). |
| `MAX_UPLOAD_MB` | int | `2048` | Upload size limit → `413`. |
| `MAX_AUDIO_SECONDS` | int | `36000` | Duration limit (10h). `0` = no limit. |
| `MAX_QUEUE` | int | `8` | Inference queue limit (queued + in flight) → `503`. |
| `VAD_MIN_DURATION` | float | `15.0` | Chunking: min segment length, sec. |
| `VAD_MAX_DURATION` | float | `22.0` | Chunking: target maximum, sec. |
| `VAD_STRICT_LIMIT` | float | `30.0` | Chunking: hard maximum, sec. |
| `VAD_NEW_CHUNK_THRESHOLD` | float | `0.2` | Chunking: new-chunk threshold, sec. |
| `VAD_THRESHOLD` | float | `0.5` | Silero: speech probability threshold. |
| `HOST` | str | `0.0.0.0` | uvicorn host. |
| `PORT` | int | `8000` | Port. |
| `LOG_LEVEL` | str | `INFO` | `DEBUG`\|`INFO`\|`WARNING`\|`ERROR`. |
| `LOG_JSON` | bool | `false` | JSON log format (otherwise human-readable). |
| `DEFAULT_RESPONSE_FORMAT` | str | `json` | Default response format. |
| `ALLOWED_MODELS` | csv | `v3_ctc,v3_e2e_ctc,v3_rnnt,v3_e2e_rnnt` | Allowed values for the `model` field. |

---

## Deploy (Docker Compose)

The deployment target is any self-hosted Docker host — a Linux server, NAS, or mini-PC (**x86_64**,
no GPU; minimum 2 cores / 2 GB RAM, recommended 4 cores / 4–8 GB). Deployment is via `docker-compose.yml`
(not `make`), using the
`docker compose` CLI (a container-management UI works too). The image is built natively on the host
(amd64) or in advance on another machine.

> **The image is self-contained.** ffmpeg + ffprobe, Python, torch (CPU) and all dependencies are
> bundled into the image — nothing extra needs to be installed on the host. The container only needs
> internet access **on the first start** (downloading the GigaAM weights from the CDN into the volume);
> after that it works offline from the cache.

### Steps

1. **Files on the host.** Place into the project directory (e.g. `/opt/gigaam-api`): `Dockerfile`,
   `docker-compose.yml`, `.dockerignore`, `pyproject.toml`, `uv.lock`, the `gigaam_api/` directory,
   and `.env` (copy from `.env.example`, **set `API_KEY`**).
2. **Weights directory and permissions.** Create a `models` subdirectory and make it owned by **UID 1000**
   (the container runs as this non-root user, otherwise it cannot write the weights):
   ```bash
   mkdir -p /opt/gigaam-api/models
   sudo chown -R 1000:1000 /opt/gigaam-api/models
   ```
3. **Build and start:**
   ```bash
   docker compose up -d --build
   ```
   (On an x86_64 host the image is built natively, without emulation.)
4. **The first start downloads the weights** of GigaAM into `./models` (minutes, depends on bandwidth).
   The container will be in the "starting" status, the `healthcheck` has `start_period: 600s` to cover
   the download → then "healthy".
5. **Check:** `curl http://<host_ip>:8000/health` → `200` with `"loaded":true`.

### Volume and reinstall

`./models:/data/models` — the weights cache **survives** container recreation (no re-download).
`.env` and `./models` live on the host, they are not baked into the image (the image is light, without weights).

### Pre-warming the weights (optional)

To make the production start instant, the weights can be downloaded in advance, without bringing the
service up, with a one-off run of the `tools` profile:

```bash
docker compose --profile tools run --rm download-weights
```

(On a dev Mac the same — `make download-weights`.)

### Resource limits

`docker-compose.yml` has commented-out `mem_limit`/`cpus` — tune them to the host hardware.
Keep `NUM_THREADS` in `.env` ≤ the number of allocated cores.

---

## Performance and limitations

- **CPU speed.** The service is **batch, not realtime**: the RTF (compute time / duration) on long
  files can be ≥1 — 10h of audio takes hours to process. More cores help (recommended 4); on the
  minimum 2 cores expect proportionally slower throughput. For long files use `stream=true`
  (progress + timeout protection).
- **Model.** On CPU the default is `v3_ctc` (faster). RNN-T is more accurate but noticeably slower.
- **Memory.** 2 GB is enough for short audio; long files need more. The peak on a long file ≈ weights
  (~1 GB) + int16 buffer (~1.15 GB/10h) + float on the VAD stage (~2.3 GB/10h), so multi-hour files
  want ~4–8 GB. If memory runs out — reduce `BATCH_SIZE` or split the input into parts.
- **`verbose_json`.** The fields `tokens`/`avg_logprob`/`no_speech_prob`/`temperature`/`seek` are
  best-effort (`0.0`/`0`); `compression_ratio` is honest. GigaAM does not provide these metrics.

---

## Troubleshooting

- **`ffmpeg`/`ffprobe` not found (`500`).** In Docker it is installed automatically; on native dev —
  install ffmpeg on your system (`brew install ffmpeg`) and check `ffmpeg -version`.
- **No permission to write weights to the volume (the container crashes at startup).** The host
  `./models` must be owned by UID 1000: `sudo chown -R 1000:1000 ./models`.
- **OOM on long files.** Reduce `BATCH_SIZE`, increase the container memory limit, or split the input
  into parts. The memory peak is on the VAD stage of long audio.
- **MPS errors on Mac.** `auto` on Mac resolves to `mps`; on errors set
  `PYTORCH_ENABLE_MPS_FALLBACK=1` (there is a CPU fallback) or force `DEVICE=cpu`.
- **Re-download the weights.** Delete the contents of `./models` (volume) — on the next start the
  weights are downloaded again.
- **Slow first start / `unhealthy`.** This is the weights download. Wait for it to finish (`start_period`
  600s); for a slow connection use pre-warming (see above).

---

## Commands (Makefile)

A convenience for **development on Mac** (in production deployment goes without `make`, via `docker compose`).

| Target | Action |
|---|---|
| `make install` | Install dependencies (`uv sync`). |
| `make run` | Local run (uvicorn `--reload`). Variables `HOST`/`PORT`. |
| `make download-weights-local` | Pre-warm the weights **natively** (uv, without Docker) into `MODELS_DIR` from `.env`. |
| `make check` | `lint` + `format-check` + `typecheck` + `test` — the fast loop. |
| `make pre-commit` | The whole batch of tests of all types in a row (after each task). |
| `make test` / `make test-integration` | Unit / integration tests. |
| `make build-docker` | Build the production image (`--platform linux/amd64`). |
| `make up` / `make down` / `make logs` | `docker compose up -d` / `down` / `logs -f`. |
| `make download-weights` | Pre-warm the weights via **Docker** (one-off container, `tools` profile). |
| `make clean` | Remove tool caches. |

---

## Development

- The working language of the project is Russian; code/identifiers are English.
- **TDD**, `mypy strict`, pragmatic testing (key/risky logic and the happy path).
- After each task — a green `make pre-commit`.
- Architecture and design decisions are documented in `CLAUDE.md` (project guide + ADR log).
