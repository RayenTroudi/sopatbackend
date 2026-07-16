"""Pydantic response models (JSON contract with the Flutter app)."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"


class LineOut(BaseModel):
    text: str
    confidence: float


class OCRSuccessResponse(BaseModel):
    success: bool = True
    text: str
    confidence: float
    lines: list[LineOut]
    processing_time: float
    request_id: str


class OCRErrorResponse(BaseModel):
    success: bool = False
    error: str
    request_id: str | None = None
