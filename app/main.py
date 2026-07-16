"""FastAPI application entry point.

Models are loaded once during the lifespan startup and stored on app.state —
they are never reloaded per request.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)

    # Heavy imports happen here (not at module import time) so tooling and
    # tests can import the app module quickly.
    from app.services.ocr_pipeline import OCRPipeline
    from app.services.paddle_service import PaddleDetectionService
    from app.services.trocr_service import TrOCRService

    logger.info("Starting OCR backend — loading models (one-time)...")
    detector = PaddleDetectionService(settings)
    recognizer = TrOCRService(settings)

    app.state.settings = settings
    app.state.pipeline = OCRPipeline(detector, recognizer, settings)
    logger.info("Models loaded. Service is ready.")

    yield

    logger.info("Shutting down OCR backend.")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="SOPAT Handwriting OCR API",
        description="Extracts handwritten text from images using PaddleOCR "
        "(detection) and TrOCR (recognition).",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    _settings = get_settings()
    uvicorn.run("app.main:app", host=_settings.host, port=_settings.port)
