"""Weight warmup: download/verify the model weights into `MODELS_DIR` and exit.

Runs once (CLI or a one-shot container), does not start an HTTP server. Useful for
pre-warming on a slow/offline host — so that the production start of the service is
instant (cache hit in the mounted volume) instead of hanging on the download in the lifespan.

Run:
    python -m gigaam_api.download_weights
or (see docker-compose.yml, the `tools` profile):
    docker compose --profile tools run --rm download-weights

Weights are fetched by `gigaam.load_model(download_root=MODELS_DIR)`; Silero VAD is bundled
in the package (no network needed). Creating `GigaAMEngine` reuses the service's standard load path.
"""

import logging

from gigaam_api import __version__
from gigaam_api.config import get_settings
from gigaam_api.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    setup_logging(settings)
    logger.info(
        "weight warmup v%s | model=%s device=%s models_dir=%s",
        __version__,
        settings.MODEL,
        settings.DEVICE,
        settings.MODELS_DIR,
    )
    # Lazy import: pull in torch/gigaam only during an actual warmup (project convention).
    from gigaam_api.asr.gigaam_engine import GigaAMEngine

    engine = GigaAMEngine(settings)
    logger.info(
        "weight warmup done: model=%s device=%s cache=%s",
        engine.model_name,
        engine.device,
        settings.MODELS_DIR,
    )


if __name__ == "__main__":
    main()
