"""HTTP routes. Business logic lives in the services layer."""

import time
import uuid

import anyio
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.api.schemas import HealthResponse, LineOut, OCRErrorResponse, OCRSuccessResponse
from app.config import Settings
from app.services.image_processing import InvalidImageError
from app.services.ocr_pipeline import NoTextDetectedError, OCRPipeline
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@router.post(
    "/ocr",
    response_model=OCRSuccessResponse,
    responses={
        400: {"model": OCRErrorResponse},
        413: {"model": OCRErrorResponse},
        415: {"model": OCRErrorResponse},
        422: {"model": OCRErrorResponse},
        500: {"model": OCRErrorResponse},
    },
)
async def ocr(request: Request, image: UploadFile = File(...)) -> JSONResponse:
    """Extract handwritten text from an uploaded image (multipart field: `image`)."""
    request_id = uuid.uuid4().hex[:12]
    settings: Settings = request.app.state.settings
    pipeline: OCRPipeline = request.app.state.pipeline
    start = time.perf_counter()

    content_type = (image.content_type or "").lower()
    if content_type not in settings.allowed_types:
        return _error(
            415,
            f"Unsupported file type '{content_type or 'unknown'}'. "
            f"Allowed: {', '.join(sorted(settings.allowed_types))}.",
            request_id,
        )

    data = await image.read()
    if not data:
        return _error(422, "Empty upload: the image file contains no data.", request_id)
    if len(data) > settings.max_upload_size_bytes:
        return _error(
            413,
            f"File too large ({len(data) / 1024 / 1024:.1f} MB). "
            f"Maximum allowed is {settings.max_upload_size_mb} MB.",
            request_id,
        )

    logger.info("[%s] OCR request: %s (%d bytes)", request_id, image.filename, len(data))

    try:
        # The pipeline is CPU-bound; run it in a worker thread so the event
        # loop keeps serving other requests.
        result = await anyio.to_thread.run_sync(pipeline.run, data)
    except InvalidImageError as exc:
        logger.warning("[%s] Invalid image: %s", request_id, exc)
        return _error(400, "Invalid or corrupted image file.", request_id)
    except NoTextDetectedError:
        logger.info("[%s] No handwriting detected.", request_id)
        return _error(422, "Unable to detect handwriting.", request_id)
    except Exception:
        logger.exception("[%s] OCR pipeline failed.", request_id)
        return _error(500, "Internal server error during OCR processing.", request_id)

    processing_time = round(time.perf_counter() - start, 3)
    logger.info(
        "[%s] Done in %.2fs (detection %.2fs, recognition %.2fs, %d lines)",
        request_id,
        processing_time,
        result.detection_time,
        result.recognition_time,
        len(result.lines),
    )

    payload = OCRSuccessResponse(
        text=result.text,
        confidence=result.confidence,
        lines=[LineOut(text=l.text, confidence=l.confidence) for l in result.lines],
        processing_time=processing_time,
        request_id=request_id,
    )
    return JSONResponse(status_code=200, content=payload.model_dump())


def _error(status_code: int, message: str, request_id: str) -> JSONResponse:
    body = OCRErrorResponse(error=message, request_id=request_id)
    return JSONResponse(status_code=status_code, content=body.model_dump())
