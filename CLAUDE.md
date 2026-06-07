# CLAUDE.md — GigaAM ASR (OpenAI-compatible service)

A **Russian-speech** recognition service built on [GigaAM](https://github.com/salute-developers/GigaAM),
exposing an **OpenAI-compatible** API (`POST /v1/audio/transcriptions`). Self-hosted in Docker on any
CPU host (Linux server / NAS / mini-PC), developed on macOS. This file is the project guide; the
source of truth for behaviour is the code itself.

## Critical rules (root cause — do not violate)

1. **Production is CPU-only.** Docker on Mac runs in a Linux VM without Metal → no GPU. MPS works only
   on a **native** `uv` run, never in Docker. The target host is a self-hosted CPU box.
2. **Do longform ourselves** — Silero VAD + the ported GigaAM chunking. **Never call
   `model.transcribe_longform`** (it pulls pyannote); there must be no `import pyannote` anywhere.
3. **Always wrap direct `model.forward`/`model._decode` in `torch.inference_mode()`** (longform
   `_iter_chunks`, and any new stage-07 inference path). Without it, autograd + the short path's
   inference-tensor rotary cache → `RuntimeError: Inference tensors cannot be saved for backward`
   (only reproduces short→long on one instance; locked by `tests/integration/test_short_then_long_real.py`).
4. **Keep Linux torch on CPU wheels** (`2.12.0+cpu`, via `index+marker` in `pyproject.toml`). Never
   reintroduce CUDA into the Linux resolution — it would add gigabytes of nvidia/triton to the image.
5. **The image is self-contained:** call ffmpeg/ffprobe from the image's PATH (built in via apt;
   `gigaam_api/audio.py`). Never rely on host binaries — ffmpeg may be absent on the host.
6. **MPS on Mac** may need `PYTORCH_ENABLE_MPS_FALLBACK=1` (GigaAM on MPS is untested upstream).
7. **The service is batch, not realtime:** 10h of audio = hours of CPU compute (min 2 cores,
   recommended 4). Long files — via `stream=true`.

## Commands (Makefile)

- `make install` — `uv sync`.
- `make run` — local run (`uvicorn --reload`).
- `make check` — `lint` + `format-check` + `mypy(strict)` + unit tests — the fast loop.
- `make pre-commit` — `check` + integration tests; **must be green after every task**. (A Makefile
  target, not the pre-commit tool.)
- `make test` / `make test-integration` — unit / real-model tests.
- `make download-weights-local` — warm up weights natively (uv, no Docker) into `MODELS_DIR`.
- `make build-docker` / `up` / `down` / `logs` — Docker (dev convenience; prod uses `docker compose`).
- `make download-weights` — warm up weights via the Docker `tools` profile.
- Sub-targets: `lint`, `format`, `format-check`, `typecheck`, `coverage`, `clean`.

## Configuration

pydantic-settings loaded from `.env` (field names = env var names). Sources of truth:

- **Full reference table** — README "Configuration" section.
- **Editable template with defaults** — `.env.example`.
- **Schema/validation** — `gigaam_api/config.py` (`Settings`).

Key vars: `MODEL` (`v3_ctc`), `API_KEY` (empty ⇒ auth off), `DEVICE` (`auto`→cuda/mps/cpu),
`MODELS_DIR`, `MAX_QUEUE` (8), `BATCH_SIZE` (4), `NUM_THREADS` (4), `MAX_AUDIO_SECONDS` (36000),
`MAX_UPLOAD_MB` (2048), `VAD_*` chunking knobs, `ALLOWED_MODELS` (CSV).

## Conventions

- **mypy strict**, **TDD** (test → code), pure functions for `formats`/VAD chunking.
- **Test pragmatically:** cover risky/key logic and the happy path; don't chase coverage or write
  tests for tests' sake.
- **YAGNI:** delete unused/dead code immediately (no commented-out blocks). The one deliberate
  exception is the `ASREngine` abstraction kept for ONNX.
- No network calls at import time; weights download only in the lifespan.
- Logging via stdlib `logging`, level from `LOG_LEVEL`; debug logs at key points.
- Keep **CLAUDE.md and both READMEs current** in the same task that changes behaviour/commands/API/
  config; record new architectural decisions in the table below.
- Run **`make pre-commit` (green) after every task** and at the end of every stage.
- **Don't commit or push without an explicit request.**

## Architectural decisions (ADR log)

> Append any new architectural decision here as a `decision · reason` row — to reuse experience
> across sessions. This is a living section.

| Decision | Reason |
|---|---|
| Inference backend — **PyTorch** behind the `ASREngine` Protocol (ONNX optional, stage 07) | Full GigaAM API out of the box; one codebase for cpu/mps/cuda. |
| Long-audio VAD — **Silero VAD JIT** (`load_silero_vad(onnx=False)`), not pyannote, not onnxruntime | No HF_TOKEN/licenses; single torch stack; onnxruntime defaults to all-core threads → oversubscription against the torch pool on weak CPUs. Bundled in the pip package (no network); ONNX switch later = 1 line. |
| **Python 3.12** + package manager **uv** | torch/MPS compatibility; 3.14 too fresh. |
| Model via `MODEL` env, **CTC by default** | CTC is faster on the target CPU host. |
| Auth — **Bearer key from `API_KEY`** (empty ⇒ disabled) | OpenAI-client compatible. |
| Weights — **downloaded on first start into the `MODELS_DIR` volume**, not baked into the image | Lightweight image. |
| API surface — sync `POST /v1/audio/transcriptions` (+ optional `stream=true` SSE), `GET /v1/models`, `GET /health` | OpenAI standard; translations/WebSocket out of scope. |
| Package build — **hatchling** (editable via `uv sync`) | `gigaam_api` importable in pytest/mypy/uvicorn. |
| Pinning — **`>=` in `pyproject.toml`, exact pins in `uv.lock`**; gigaam git-pinned (`rev 6e4b027`) in `[tool.uv.sources]` as a bare `gigaam` dep | uv idiom: reproducibility via the lock, deliberate upgrades via `uv lock --upgrade`. |
| **`DEVICE=auto` resolved by us** (cuda→mps→cpu), explicit device passed to `load_model` | GigaAM's own auto skips MPS (cuda→cpu); MPS is needed on the dev Mac. |
| CSV settings (`ALLOWED_MODELS`) — **`Annotated[..., NoDecode]` + `field_validator`** | pydantic-settings parses `list` as JSON; NoDecode + `split(",")` gives CSV. |
| `ASREngine` — **`@runtime_checkable` Protocol** (`transcribe`/`info`/`iter_segments`); `/health` narrows the engine type without importing gigaam/torch | "HTTP ⟂ inference": the HTTP layer stays torch-free; engine imported lazily in the lifespan. |
| mypy — **per-module `ignore_missing_imports`** for `gigaam.*`/`silero_vad.*` | No py.typed/stubs; a targeted override beats a broad `# type: ignore`. |
| Duration routing **inside the engine**: `>MAX_AUDIO_SECONDS`→`AudioTooLongError`; `≤25s`→short (delegate to `model.transcribe`); else→longform. Near-boundary `ValueError "too long"`→fallback to longform | gigaam measures by samples, probe by seconds → near 25s going to longform beats failing; the hot short path stays untouched. |
| Longform — port of `gigaam vad_utils.segment_audio_file`: **pure `merge_intervals_to_chunks` (boundaries only)** + slicing/batching in the engine; inference via `model.forward`/`model._decode`; words `+seg_start`, `round(...,3)` | Pure merge logic is unit-tested in isolation; we never call upstream `transcribe_longform` (it pulls pyannote). |
| Longform memory — decode to an **int16 `torch.Tensor`** (`torch.frombuffer`, no numpy); a float copy only for the VAD stage, then freed | int16 halves memory (~1.15 GB/10h); the float peak is at VAD, not the batches; keeps `audio.py` torch-free for the HTTP layer (lazy torch imports). |
| Longform inference (`forward`/`_decode`) **must run under `torch.inference_mode()`** | Otherwise autograd + the short path's inference-tensor rotary cache → `RuntimeError: Inference tensors cannot be saved for backward`; regression-locked by `test_short_then_long_real.py`. |
| Longform cancellation on disconnect — **cooperative** `cancel_check` checked per batch → `InferenceCancelledError`; the API watches `request.is_disconnected()`. Short path is non-cancellable | A ThreadPool task can't be interrupted; one worker ⇒ an abandoned longform would block the queue for everyone. |
| Backpressure — single key **`MAX_QUEUE`** (admitted = queued + in-flight); over it → `QueueFullError`→**503**. **No request timeout** | A timeout would cut legitimate multi-hour files (RTF≥1); abandoned jobs are handled by cancellation, not a timeout. |
| Error mapping splits the cause: `AudioToolNotFoundError` (ffmpeg/ffprobe missing)→**500** `api_error`; `AudioDecodeError` (bad/unsupported file)→**400** `invalid_request_error`. No 415 | Real OpenAI returns 400 `invalid_request_error` for a bad audio file, never 415; one code for client + server causes is wrong. |
| OpenAI compatibility details — `timestamp_granularities[]` form alias; verbose `seek=0` + per-segment `compression_ratio`; `/v1/models` echoes `ALLOWED_MODELS` | Matches the canonical OpenAI client wire format. |
| `compression_ratio` — **bytes/bytes** `len(b)/len(zlib.compress(b))`, `b=text.encode()` | Whisper counts bytes; counting Cyrillic chars (2 bytes each) would halve the ratio and never trip the >2.4 hallucination threshold. |
| Disconnect watcher — **`anyio.create_task_group()` + `cancel_scope.cancel()`** (not raw `asyncio` cancel); the inference outcome is captured inside the group and dispatched outside | `Request.is_disconnected()` holds an anyio CancelScope; raw-asyncio cancel conflicts → the request deadlocks. Capturing outside keeps `QueueFullError`→503 (not wrapped in an ExceptionGroup→500). |
| SSE delta semantics — **prefix space**: first delta = `seg0.text`, rest = `" "+segN.text`; `done.text=" ".join(...)`. Invariant `"".join(delta)==done.text==sync` | The universal OpenAI streaming invariant: concatenated deltas reproduce the final text exactly. |
| Blocking-`iter_segments`→async bridge — **`asyncio.Queue` + `loop.call_soon_threadsafe`**, producer in `Runner.submit` (the same single worker), no `maxsize` | Keeps inference serialized, never blocks the loop; heartbeat via `wait_for(queue.get(), 15s)` (safe to cancel your own coroutine), not by cancelling someone else's generator. |
| Streaming backpressure — **`try_acquire()` BEFORE `StreamingResponse`** (503 without headers); `release()` in the producer's done-callback; `_inflight` under a lock | An async generator defers its body until after the 200 → the 503 must be decided earlier; release on producer completion = inflight tracks worker occupancy, not client read speed. |
| **Temp-file ownership handed to the stream** — the handler's `finally` doesn't delete; `_cleanup` (producer done-callback) deletes after the worker finishes reading | The handler returns immediately → its `finally` would delete the file before inference reads it. |
| Stream cancellation — **`cancel_event.set()` in the bridge generator's `finally`** (Starlette cancels the generator on disconnect); `sse_transcription` catches `Exception`→`error` event but lets `CancelledError`/`GeneratorExit` through | `iter_segments` stops between batches (same granularity as sync); disconnect = cleanup only. |
| `verbose_json`/`srt`/`vtt` + `stream=true` → **synchronous fallback** (stream only when `fmt in {json,text}`), not 400 | Most clients send `stream=true` by default with `verbose_json`; a 400 would break them. |
| `iter_segments` — **shared batch loop `_iter_chunks`** (+ `_prepare_longform`) reused by the sync `_transcribe_longform`; part of the `ASREngine` Protocol | Single source of longform logic (DRY); ≤25s delegates to the short path and yields one segment. |
| CPU-torch — **`index+marker` in `pyproject.toml`**: `[[tool.uv.index]] pytorch-cpu` (`explicit`) + torch/torchaudio sourced with marker `sys_platform=='linux'`. One `uv.lock`: Mac `2.12.0` (MPS), Linux `2.12.0+cpu` | uv idiom; as a side effect drops the entire CUDA stack (nvidia/triton) from the Linux resolution. Dockerfile just `uv sync --frozen`. |
| **`onnx` overridden to `>=1.21.0`** via `[tool.uv] override-dependencies` | `onnx` is transitive via gigaam, which hard-pins `onnx==1.19.*` (open high/medium GHSA advisories, fixed in 1.21.0); that pin made Dependabot security updates unresolvable. Safe: gigaam does no bare `import onnx` (only onnxruntime + `torch.onnx.export`) and this service never calls ONNX export; green `make pre-commit` incl. real-inference tests confirms it. |
| Image — **multi-stage `python:3.12-slim`**: builder (uv + git; deps layer, then code) + a thin runtime (ffmpeg from apt, non-root UID/GID 1000, healthcheck via stdlib urllib, `XDG_CACHE_HOME=/data/models/.cache`). Platform is build-time (`--platform`), not in `FROM` | Self-contained (ffmpeg may be absent on the host); dependency cache separate from code; multi-arch-friendly + fast native arm64 validation on Mac. |
| **Silero bundled in the pip package** ⇒ the volume is needed **only** for GigaAM weights (`MODELS_DIR`) | No HF Hub / torch.hub; `XDG_CACHE_HOME`→volume kept only as a non-root safety net. |
| Test fixtures — committed `tests/integration/data/ru_short_sample.wav` (~11s) and `ru_long_sample.wav` (40s, cut from GigaAM's `long_example.wav`); integration tests run on **cpu** and skip gracefully without weights/network | Tracked real RU speech (the long file has pauses ⇒ >1 chunk); names ≠ the throwaway `example.wav` that `.gitignore` drops; cpu = determinism + prod parity. |
| Deployment — **`docker-compose.yml` + `docker compose` CLI; no `make` in production**. First `up` downloads the weights (`healthcheck start_period 600s`); optional `tools`-profile `download-weights` service for warm-up | The prod path must "just work" from compose (incl. behind a NAS UI). `make` targets are dev convenience on Mac only. |
| Project language — **English** (comments/docstrings/log + error messages); ruff RUF001/002/003 enabled | RU README is `README_ru.md`. RU **speech transcripts** in tests are kept as ASR test data; mixed VTT/SRT lines carry a per-line `# noqa: RUF001`. |
| **`VAD_THRESHOLD` default 0.5 → 0.25** | Empirical (pooled WER over 14 FLEURS-ru files, clean + noise 15/5dB, both v3_ctc & v3_e2e_rnnt): 0.2–0.3 cuts WER 10–23% vs 0.5 — lower threshold = fewer false pauses = more context per chunk. Floor 0.2: at ~0.1 VAD stops seeing pauses → >30s blocks → mid-word arithmetic cuts. |
| **No overlap/snap chunk-boundary post-processing** (YAGNI) | Measured: natural RU speech never has >30s continuous blocks (max ~5–6s at threshold≥0.3) → the arithmetic cut that splits words almost never fires; selective-overlap == baseline. Energy-snap is unsafe (can't tell a word-internal energy dip from a pause; hurts in worst case). If the domain becomes pause-less (singing/dense), add **selective overlap+stitch on ARITH boundaries**, not snap. |

<!-- Append new decisions as a new row above this hint. -->

## GigaAM reference

[GigaAM](https://github.com/salute-developers/GigaAM) is the upstream Russian-ASR model/library this
service wraps — a git-pinned dependency (`rev 6e4b027` in `[tool.uv.sources]`), not vendored. We reuse
its inference (`model.transcribe`, `forward`/`_decode`) and port its VAD chunking. Key references:
[`gigaam/model.py`](https://github.com/salute-developers/GigaAM/blob/6e4b027/gigaam/model.py),
[`gigaam/vad_utils.py`](https://github.com/salute-developers/GigaAM/blob/6e4b027/gigaam/vad_utils.py)
(chunking algorithm), `gigaam/decoding.py`, `gigaam/__init__.py` (weights loading/cache).
