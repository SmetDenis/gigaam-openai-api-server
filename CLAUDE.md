# CLAUDE.md — GigaAM ASR (OpenAI-compatible service)

A **Russian-speech** recognition service built on [GigaAM](https://github.com/salute-developers/GigaAM),
exposing an **OpenAI-compatible** API (`POST /v1/audio/transcriptions`). Runs as a self-hosted
service in Docker (any CPU host: Linux server / NAS / mini-PC) and is developed on macOS.

> **This is the project's main guide document** (for Claude and developers): architecture, current
> status, ADR log of accepted decisions, and conventions. The source of truth for requirements is
> this file and the code itself.

## Status

Stage **06 ✅** (Docker / self-hosted deployment: multi-stage `python:3.12-slim`, **CPU-torch via
index+marker** in `pyproject.toml` — a single `uv.lock` for Mac(MPS)/Linux(`2.12.0+cpu`),
CUDA/nvidia/triton no longer pulled in; **self-contained image** — ffmpeg+ffprobe from apt, non-root
UID 1000, healthcheck via stdlib urllib; `docker-compose.yml` for the **`docker compose` CLI**
(`env_file required:false`, volume `./models:/data/models`, `start_period 600s` for the first weights
download, optional `tools` profile for warm-up); `download_weights.py`; final README. Deploy =
`docker compose up -d`, **no make** in production; Silero is bundled in the package → the volume is
only for GigaAM weights). Next — `07` (optional CPU optimization).
Stage **05 ✅** (SSE streaming `stream=true`: `transcript.text.delta`→`transcript.text.done`→`[DONE]`,
invariant `"".join(delta)==done.text==sync` (prefix space), heartbeat comments ~15s against
idle timeouts; `iter_segments` (shared batch loop with longform); thread→async bridge via
`asyncio.Queue` + `Runner.submit` into the same worker; backpressure `try_acquire`→503 BEFORE headers;
temp-file ownership handed to the stream; verbose/srt/vtt+stream → synchronous fallback; error in the
stream → `error` event).
Stage **04 ✅** (OpenAI-compatible API: `POST /v1/audio/transcriptions` all formats
`json`/`text`/`verbose_json`/`srt`/`vtt`, `GET /v1/models`, Bearer auth, OpenAI error format;
`Runner` (1 worker + `MAX_QUEUE`→503); cooperative longform cancellation on disconnect via an **anyio
task group**; chunked upload→413; probe limit→400). Stage **03 ✅** (longform >25s: Silero VAD (JIT)
→ pure chunking function `merge_intervals_to_chunks` (ported from gigaam) → batched
`model.forward`/`model._decode`; duration-based routing inside the engine; int16 decode to save
memory; no pyannote). Stage **02 ✅** (PyTorch ASR engine behind `ASREngine`, short audio ≤25s,
model load in the lifespan, weights cache in `MODELS_DIR`, `/health.loaded=true`). Stage **01 ✅**
(skeleton, tooling, logging, FastAPI skeleton).

**Stage tracker:**

| Stage | Topic | Status |
|---|---|---|
| 01 | Skeleton, tooling, logging, FastAPI skeleton `/health` | ✅ |
| 02 | Config, ASR engine (PyTorch), short audio ≤25s, weights cache | ✅ |
| 03 | Silero VAD + chunking + longform loop | ✅ |
| 04 | OpenAI endpoints, schemas, formats, auth, runner | ✅ |
| 05 | SSE streaming (`stream=true`); verbose/srt/vtt+stream → synchronous fallback | ✅ |
| 06 | Docker (amd64, CPU-torch), compose, volume, self-hosted deployment | ✅ |
| 07 | Optional CPU optimization: int8 flag, benchmarks, groundwork for ONNX | ⬜ optional |

## Architectural decisions (ADR log)

> **Rule:** any new architectural decision made during development is **appended here**
> (date · decision · reason) — to reuse experience across sessions. This is a living section.

| Date | Decision | Reason |
|---|---|---|
| 2026-06-03 | Inference backend — **PyTorch** behind the `ASREngine` abstraction (ONNX optional, stage 07) | Full GigaAM API out of the box; one codebase for cpu/mps/cuda. |
| 2026-06-03 | VAD for long audio — **Silero VAD** (NOT pyannote) | Lightweight, no HF_TOKEN/licenses; drops heavy dependencies; single speaker. |
| 2026-06-03 | **Python 3.12**, package manager **uv** | torch/MPS compatibility; 3.14 too fresh. |
| 2026-06-03 | Model — via `.env` (`MODEL`), CTC by default | CTC is faster on CPU (the target self-hosted host). |
| 2026-06-03 | Auth — Bearer key from `.env` (`API_KEY`) | Compatible with OpenAI clients. |
| 2026-06-03 | Weights — downloaded on first start into a volume (`MODELS_DIR`), NOT into the image | Lightweight image. |
| 2026-06-03 | API — synchronous `transcriptions` + optional `stream=true` (SSE); endpoints `transcriptions`, `/v1/models`, `/health` | OpenAI standard; SSE against timeouts. translations/WebSocket are out of scope. |
| 2026-06-03 | Package build — **hatchling** (editable install via `uv sync`) | `gigaam_api` is importable in pytest/mypy/uvicorn; `uvicorn gigaam_api.main:app` works reliably. |
| 2026-06-03 | Pinning — **lower bounds (`>=`) in `pyproject.toml` + exact pins in `uv.lock`** | The uv idiom: reproducibility via the lock, deliberate upgrades via `uv lock --upgrade`. |
| 2026-06-03 | We resolve `DEVICE=auto` **ourselves** (cuda→mps→cpu) and pass an explicit device to `load_model` (stage 02) | GigaAM's built-in auto (`device=None`) = cuda→cpu, **no MPS**; MPS is needed on the dev Mac. At stage 01 `/health.device` echoes the setting. |
| 2026-06-03 | CSV Settings fields (`ALLOWED_MODELS`) — **`Annotated[..., NoDecode]` + `field_validator`** | pydantic-settings parses `list` as JSON by default; NoDecode + `split(",")` gives CSV. |
| 2026-06-03 | ruff: **RUF001/002/003** (ambiguous-unicode) disabled | False positives on legitimate Cyrillic in existing comments/docstrings. — superseded 2026-06-04: project language switched to English, RUF001/002/003 re-enabled. |
| 2026-06-03 | gigaam pin **`6e4b027`** verified: `torch==2.12.0`/`torchaudio==2.11.0` + `onnxruntime==1.23.2`/`onnx==1.19.1`/`numpy==2.4.6` install on **Python 3.12 macOS arm64** (dev) | Stage 02 — wheel blocker check passed; MPS available, CUDA not. |
| 2026-06-03 | `uv add` stores the gigaam git pin in **`[tool.uv.sources]`** (`rev=6e4b027`), the dependency declared as bare `gigaam` | The uv idiom; the pin is preserved (rev + `uv.lock`), equivalent to `gigaam @ git+...@rev`. |
| 2026-06-03 | Routing >25s: **pre-check `probe_duration`>25s → `AudioTooLongError`** + defensive catch of raw gigaam exceptions (`ValueError "too long"`→`AudioTooLongError`, `RuntimeError "failed to load audio"`→`AudioDecodeError`); others re-raised | gigaam measures length by samples, probe by seconds → near the 25s boundary they can disagree; we don't mask unrelated inference errors. |
| 2026-06-03 | `ASREngine` extended with **`info()` + `@runtime_checkable`**; `/health` narrows the type of `app.state.engine` via `isinstance`, **without importing gigaam/torch** | The "HTTP ⟂ inference" principle: the HTTP layer stays light, `create_app()` has no torch (lazy engine import in the lifespan). |
| 2026-06-03 | mypy: **per-module `ignore_missing_imports`** for `gigaam.*`/`silero_vad.*` | No py.typed/stubs; a targeted override is more idiomatic than a broad `# type: ignore`. |
| 2026-06-03 | Integration sample — **committed `tests/integration/data/ru_short_sample.wav`** (11.29s, RU; name ≠ `example.wav`); test on **cpu**, graceful skip without network/weights | `.gitignore` globally ignores the throwaway `example.wav` (written by `gigaam.utils`); a separate name keeps the convention and tracks the fixture. cpu = determinism + prod CPU. |
| 2026-06-03 (stage 03) | Silero backend — **JIT (`load_silero_vad(onnx=False)`), NOT ONNX** | One torch stack with GigaAM; onnxruntime defaults to `intra_op_num_threads=0` (all cores) → oversubscription with the torch pool on weak CPUs (e.g. ~4 cores). VAD is not the bottleneck (≈hours of inference vs minutes of VAD). Weights are bundled in the package (no network). Switching to ONNX later = 1 line. |
| 2026-06-03 (stage 03) | **Routing inside the engine** (replaces the stage-02 row above): `probe_duration` → `>MAX_AUDIO_SECONDS`→`AudioTooLongError`; `≤25s`→short (delegate to `model.transcribe`, untouched); otherwise→`_transcribe_longform`. `ValueError "too long"` near the boundary now → **fallback to longform** (not an error) | `AudioTooLongError` removed from the normal path; the short path is not rewritten (minimal risk for the hot path); near the 25s boundary gigaam measures by samples → going to longform is more correct than failing. |
| 2026-06-03 (stage 03) | Longform — port of `gigaam/vad_utils.py::segment_audio_file`: **pure function `merge_intervals_to_chunks` (boundaries only)** + waveform slicing/batching in the engine; intervals from Silero; inference via private `model.forward`/`model._decode`; words `+seg_start`, `round(...,3)` | The pure merge logic is tested in isolation with synthetic data (the core of the stage); we don't call upstream `transcribe_longform` (it pulls pyannote). |
| 2026-06-03 (stage 03) | Memory: decode into an **int16 `torch.Tensor`** (`torch.frombuffer`, no numpy); the full signal goes to float **only for the VAD stage** → `del wav_f32` immediately; inference is float over the sliced batch | int16 halves memory (~1.15 GB/10h); the float peak is at VAD (≈2.3 GB/10h), not at the batches; we don't add numpy — staying in the torch stack. Lazy torch imports in `audio.py` (the module stays torch-free for the HTTP layer). |
| 2026-06-03 (stage 03) | Longform fixture — **committed `ru_long_sample.wav`** (40s, a cut of the real GigaAM `long_example.wav` via ffmpeg, mono 16k) | Real RU speech with pauses → >1 chunk; NOT `gigaam.utils.download_long_audio()` (wget into CWD). Graceful skip without network/weights. |
| 2026-06-03 (stage 04) | Longform cancellation on disconnect — **cooperative**: `ASREngine.transcribe` extended with optional `cancel_check: Callable[[], bool] \| None`; longform checks at the start of each batch iteration → `InferenceCancelledError`; the API sets a watcher on `request.is_disconnected()` → `threading.Event`. The short path (≤25s) is non-cancellable. | A ThreadPool task cannot be interrupted (verified) — the thread runs to completion; there is one worker → an abandoned longform blocks the queue for everyone. Real cancellation can only be cooperative. |
| 2026-06-03 (stage 04) | Backpressure — a single key **`MAX_QUEUE=8`**; `Runner` counts admitted (queue+work), at `≥MAX_QUEUE` → `QueueFullError`→**503**. A request timeout is **NOT introduced**. | A default timeout would cut legitimate multi-hour files (RTF≥1); the "abandoned job" problem is solved by cancellation, not a crude timeout. YAGNI. |
| 2026-06-03 (stage 04) | Error mapping — **split the cause in `audio.py`**: `FileNotFoundError` (ffmpeg/ffprobe not in PATH) → new `AudioToolNotFoundError`→**500** (`api_error`); broken/unsupported file → `AudioDecodeError`→**400** (`invalid_request_error`). `UnsupportedFormatError`/**415 removed**. | The real OpenAI returns 400 `invalid_request_error` for a bad audio file ("Unrecognized file format…" / "Audio file might be corrupted…"), it does not use 415 (verified). One code for client and server causes is wrong (root cause). |
| 2026-06-03 (stage 04) | OpenAI specifics — `timestamp_granularities[]` via `Form(alias="timestamp_granularities[]")`+`list[str]`; verbose `seek=0` + honest per-segment `compression_ratio`; `stream=true` = synchronous response until stage 05; `/v1/models` = the whole `ALLOWED_MODELS`. | The canonical OpenAI client sends the field with `[]` (verified). `seek=0` — a safe compatible default; `compression_ratio` is cheap and meaningful. The contract is fixed in the README. |
| 2026-06-03 (stage 04) | `compression_ratio` — **bytes/bytes** `len(b)/len(zlib.compress(b))`, `b=text.encode()` (NOT `len(text)` in characters). | Real Whisper counts bytes on both sides; for Cyrillic (2 bytes/char) a numerator in characters would halve the ratio → the hallucination threshold (>2.4) would never trigger. Caught in the stage-04 code review. |
| 2026-06-03 (stage 04) | The disconnect watcher — **only via `anyio.create_task_group()` + `cancel_scope.cancel()`**, NOT via `asyncio.create_task` + `task.cancel()`/`await task`. The inference outcome is captured inside the group and dispatched outside (otherwise `QueueFullError` is wrapped in an `ExceptionGroup` → 500 instead of 503). | `Request.is_disconnected()` (Starlette 1.2.x) holds an `anyio.CancelScope` inside; raw-asyncio cancellation conflicts with it → the watcher never finishes, `await watcher` **deadlocks** the whole request (caught by faulthandler: event loop idle in select, the main thread waiting on the portal). Structured anyio cancellation is consistent. |
| 2026-06-04 (stage 05) | Delta semantics — **prefix space**: the first delta = `seg0.text`, subsequent = `" "+segN.text`; `done.text=" ".join(segments)`. Invariant: `"".join(delta)==done.text==synchronous`. | The universal OpenAI streaming invariant (chat/responses/transcription, verified in the docs): concatenating deltas exactly reproduces the final text. A suffix space would leave a trailing space → mismatch with `done`/sync. |
| 2026-06-04 (stage 05) | The "blocking `iter_segments` → async" bridge — **`asyncio.Queue` + `loop.call_soon_threadsafe`**, the producer in **`Runner.submit` (the same single worker)**, NOT a temporary thread. The queue has no `maxsize` (the producer is the bottleneck, never blocks on put). | Inference serialization is preserved (one at a time), the event loop is not blocked. `call_soon_threadsafe` is the canonical thread→loop bridge. heartbeat is done via `wait_for(queue.get(), 15s)` (cancelling your own coroutine is safe), NOT via `wait_for(__anext__())` of someone else's generator (cancelling that would kill the bridge). |
| 2026-06-04 (stage 05) | Streaming backpressure — **`runner.try_acquire()` in the handler BEFORE `StreamingResponse`** (503 without headers); `release()` — in the **done-callback of the producer future** (the worker is actually free), not when the consumer finishes reading. `_inflight` under a `threading.Lock` (the loop and the worker thread both mutate it). | An async generator defers its body until the first iteration (after `200`) → the 503 must be sent earlier. Release on producer completion = inflight reflects worker occupancy, not client speed. |
| 2026-06-04 (stage 05) | **Temp-file ownership is handed to the stream**: the handler sets a `streamed` flag, `finally` does NOT delete the file; `_cleanup` (the producer done-callback) deletes it once the worker has finished reading. | The handler returns `StreamingResponse` and its `finally` would fire IMMEDIATELY → deleting the file before the worker reads it (root cause). The file is needed for the whole inference. |
| 2026-06-04 (stage 05) | Stream cancellation — **`cancel_event.set()` in the bridge generator's `finally`**; Starlette cancels the generator itself on disconnect (uvicorn HTTP `spec_version=2.3 < 2.4` → the task-group branch with `listen_for_disconnect`). We do NOT reuse the stage-04 anyio watcher. | `iter_segments` stops between batches (the same granularity as the sync path). `sse_transcription` catches `Exception` (→ error event) but lets `CancelledError`/`GeneratorExit` through (disconnect → cleanup only). |
| 2026-06-04 (stage 05) | `verbose_json`/`srt`/`vtt` + `stream=true` → **synchronous fallback** (ignore `stream`), NOT 400. The streaming condition: `stream and fmt in {json,text}`. | Most OpenAI clients send `stream=true` by default and use `verbose_json` → a 400 would break them. Predictability is preserved: these formats always return a full response. |
| 2026-06-04 (stage 05) | `iter_segments` — **a shared batch loop `_iter_chunks`** (+ `_prepare_longform`), reused by the sync `_transcribe_longform` (via `list(...)`). Added to the `ASREngine` Protocol → fake engines in tests implement it (runtime_checkable verifies the method exists → otherwise `/health` breaks). | A single source of longform logic (DRY); sync behaviour is unchanged. `iter_segments` for ≤25s delegates to the short path and yields its single segment. |
| 2026-06-04 (stage 06) | CPU-torch — **`index+marker` in `pyproject.toml`** (NOT a separate step in the Dockerfile): `[[tool.uv.index]] pytorch-cpu` (`explicit=true`) + `[tool.uv.sources]` torch/torchaudio with the marker `sys_platform=='linux'`. A single `uv.lock`: Mac → `torch 2.12.0` (PyPI, MPS), Linux → `2.12.0+cpu` (the index). In the Dockerfile just `uv sync --frozen`. | The uv 2026 idiom (verified in the docs). As a side effect `uv lock` **removed the entire CUDA stack from the Linux resolution** (`nvidia-*`, `triton`, `cuda-*`) — the old lock would have pulled CUDA-torch into the image (gigabytes). Reproducibility via a single lock, without a fragile `--no-install-package` in the Dockerfile. |
| 2026-06-04 (stage 06) | Image — **multi-stage `python:3.12-slim`**: builder (uv from a pinned `ghcr.io/astral-sh/uv` + `git` for git-gigaam, `uv sync --no-install-project` for the dependency layer → COPY code → `uv sync`) + a thin runtime (ffmpeg+ffprobe from apt, non-root **UID/GID 1000**, COPY `.venv`+`gigaam_api`). The platform is **build-time** (`docker build --platform linux/amd64`), NOT hardcoded in `FROM`. HEALTHCHECK — `python -c urllib` (no curl in slim). `XDG_CACHE_HOME=/data/models/.cache`. | **Self-contained image**: ffmpeg inside (it may be absent on the host — critical). Dependency cache layers separate from code. `--platform` not in `FROM` → multi-arch friendly + fast native validation on Mac (arm64, no qemu — verified, ~90s build). UID 1000 + chown of the volume — write permissions for non-root weights. |
| 2026-06-04 (stage 06) | **Silero is bundled in the pip package** (`silero_vad/data/*.jit/.onnx` in site-packages) → the volume is needed **only** for GigaAM weights (`MODELS_DIR`). The project uses no HF Hub / torch.hub; `XDG_CACHE_HOME`→volume is left only as a safety net for non-root. | Verified with `find .venv` — the Silero model is in the package, no network/cache needed for VAD (consistent with the stage-03 ADR). We don't multiply unnecessary volumes/ENV. |
| 2026-06-04 (stage 06, bugfix) | Longform inference (`_iter_chunks`: `forward`+`_decode`) is wrapped in **`torch.inference_mode()`** — like `gigaam.transcribe`/`transcribe_longform` (both `@torch.inference_mode()`). Without the wrapper autograd is on, and the encoder's rotary `cos`/`sin` cache, created by the short path (under inference_mode) as inference tensors, broke longform: `RuntimeError: Inference tensors cannot be saved for backward`. It only showed up in the order short→long on the SAME model instance (live service); integration tests with a separate instance per file did not catch it. | The short path delegates to `model.transcribe` (under inference_mode); longform called `forward`/`_decode` directly without the context → mixing inference tensors with autograd. The regression test `tests/integration/test_short_then_long_real.py` (one engine, short→long) reproduces it (failed before the fix) and locks it in. |
| 2026-06-04 (stage 06) | Deployment — **`docker-compose.yml` + the `docker compose` CLI, no `make` in production** (user requirement). The `make` targets (`build-docker`/`up`/`down`/`logs`/`download-weights`) are dev convenience on the Mac only. `env_file` with `required:false` (starts on defaults without `.env`). Weights warm-up — an optional compose service `download-weights` (`profiles:["tools"]`) + the module `gigaam_api/download_weights.py`; the service's first start downloads the weights anyway (`healthcheck start_period 600s`). | The production path must "just work" from compose (including via a NAS UI wrapper over compose): the first `up` downloads the weights itself, `start_period` covers the download. Warm-up is for those who want to pre-download; it is not mandatory. |
| 2026-06-04 | Project language convention switched to **English** (all comments, docstrings, log/error messages — including test error messages — translated to English); RUF001/002/003 re-enabled in `pyproject.toml`. | README_ru.md remains the Russian README. Russian **speech transcripts** in unit tests (e.g. `"раз два три"`, `"привет мир"`) are intentionally kept as domain ASR test data; the few mixed VTT/SRT subtitle strings carry a per-line `# noqa: RUF001` (keeps the rule active everywhere else). Consistency + lint guard against accidental Cyrillic look-alikes. |
| 2026-06-04 | **`onnx` pinned to `>=1.21.0` via `[tool.uv] override-dependencies`** (`pyproject.toml`). `onnx` is transitive via `gigaam`, which hard-pins `onnx==1.19.*`; that old version has open high/medium GHSA advisories (all fixed in 1.21.0). | Dependabot **security updates** could not bump `onnx` (the gigaam pin made the resolution unsatisfiable → `dependency_file_not_resolvable`, the Dependabot job failed on every run). uv `override-dependencies` forces a safe version over the upstream pin. **Safe because** gigaam never does a bare `import onnx` (only `onnxruntime` + `torch.onnx.export`), and this service never calls ONNX export — verified by green `make pre-commit` incl. real-inference integration tests (`test_short_then_long_real.py`). Single `uv.lock` (override applies on all platforms). |

<!-- Append new decisions as a new row above this hint. -->


## Critical cautions (root cause, do not violate)

1. **Docker on Mac does NOT see the GPU.** Containers run in a Linux VM without Metal → CPU only. MPS
   on Mac is available only with a **native** run (uv), not in Docker. Production = self-hosted CPU.
2. **Do NOT call `model.transcribe_longform`** — it pulls pyannote. We do longform ourselves via
   Silero VAD + the ported GigaAM chunking. There must be no `import pyannote` anywhere.
3. **torch in Docker** — CPU wheels (`download.pytorch.org/whl/cpu`), no CUDA. Implemented via
   `index+marker` in `pyproject.toml` (stage 06): Linux→`2.12.0+cpu`, Mac→`2.12.0`; a single `uv.lock`.
   **Do not bring back** CUDA into the Linux resolution (it would inflate the image by gigabytes of nvidia packages).
4. **The image is self-contained** — ffmpeg+ffprobe are built in (apt). **ffmpeg may be missing on the
   host** → do not use host binaries; everything is inside the container (`gigaam_api/audio.py` calls them from the image's PATH).
5. **A direct call to `model.forward`/`model._decode`** (longform, `_iter_chunks`) **must** be inside
   `torch.inference_mode()` — like `gigaam.transcribe`. Otherwise autograd + the inference-tensor rotary
   cache → `RuntimeError: Inference tensors cannot be saved for backward` (the bug is only caught
   short→long on a single instance; test `tests/integration/test_short_then_long_real.py`). Do not
   remove the wrapper (important for stage 07: new inference paths must also be under `inference_mode`).
6. **MPS on Mac** may require `PYTORCH_ENABLE_MPS_FALLBACK=1` (GigaAM on MPS is not tested upstream).
7. **CPU speed:** 10h of audio = hours of compute; the service is batch, not realtime (min. 2 cores,
   recommended 4). Long files — via `stream=true`.

## Commands (Makefile)

```
make install      # uv sync
make run          # local run (uvicorn --reload)
make download-weights-local  # warm up weights natively (uv, no Docker) into MODELS_DIR from .env
make check        # ruff + ruff format --check + mypy(strict) + pytest(unit) — the fast loop
make pre-commit   # the WHOLE batch of tests of all kinds in a row (check + integration). ← after EVERY task, must be green
make test         # unit tests (no integration)
make test-integration  # real model/network (the integration marker)
make build-docker / up / down / logs   # stage 06 (Docker, dev convenience)
make download-weights  # warm up weights via Docker (the tools profile)
```
> `make pre-commit` is a Makefile target, **not the pre-commit tool**.

## Conventions

- **mypy strict**, **TDD** (test → code), pure functions for `formats`/VAD chunking.
- **Test pragmatically, no over-engineering:** cover risky/key logic and the happy path,
  don't write tests for the sake of tests and don't chase coverage percentages.
- **YAGNI:** don't write code "just in case"; **delete unused/dead code immediately**
  (no commented-out blocks). The deliberate exception is the `ASREngine` abstraction for ONNX.
- No network calls in module imports; weights download — only in the lifespan.
- Logging — stdlib `logging`, level from `LOG_LEVEL`; debug logs at key points.
- **Keep `CLAUDE.md` and the README files always current:** if behaviour/commands/API/config
  change — update them in the same task. New architectural decisions — in the "Architectural decisions" section above.
- **After every task** and at the end of every stage — a green `make pre-commit` + an updated stage tracker (the "Status" section).
- **Do not commit without an explicit user request.**

## GigaAM reference

[GigaAM](https://github.com/salute-developers/GigaAM) is the upstream Russian-ASR model/library this
service wraps — installed as a git-pinned dependency (rev `6e4b027` in `[tool.uv.sources]`,
`pyproject.toml`), not vendored. We reuse its inference (`model.transcribe`, `forward`/`_decode`) and
port its VAD chunking. Key source files for reference:
[`gigaam/model.py`](https://github.com/salute-developers/GigaAM/blob/6e4b027/gigaam/model.py),
[`gigaam/vad_utils.py`](https://github.com/salute-developers/GigaAM/blob/6e4b027/gigaam/vad_utils.py)
(chunking algorithm), `gigaam/decoding.py`, `gigaam/__init__.py` (weights loading/cache).
