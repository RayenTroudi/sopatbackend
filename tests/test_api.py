"""API tests with mocked pipeline (no models required)."""

import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.api.routes import router
from app.config import Settings
from app.services.image_processing import InvalidImageError
from app.services.ocr_pipeline import LineResult, NoTextDetectedError, OCRResult


class FakePipeline:
    def __init__(self, result=None, error: Exception | None = None):
        self._result = result
        self._error = error

    def run(self, data: bytes) -> OCRResult:
        if self._error:
            raise self._error
        return self._result


def make_client(pipeline: FakePipeline) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.settings = Settings(max_upload_size_mb=1)
    app.state.pipeline = pipeline
    return TestClient(app)


def png_upload(size: tuple[int, int] = (100, 60)) -> dict:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(255, 255, 255)).save(buf, format="PNG")
    buf.seek(0)
    return {"image": ("test.png", buf, "image/png")}


@pytest.fixture
def success_result() -> OCRResult:
    return OCRResult(
        text="hello\nworld",
        confidence=0.91,
        lines=[LineResult("hello", 0.95), LineResult("world", 0.87)],
        detection_time=0.1,
        recognition_time=0.5,
    )


def test_health(success_result):
    client = make_client(FakePipeline(result=success_result))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ocr_success(success_result):
    client = make_client(FakePipeline(result=success_result))
    response = client.post("/ocr", files=png_upload())
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["text"] == "hello\nworld"
    assert body["confidence"] == 0.91
    assert len(body["lines"]) == 2
    assert body["processing_time"] >= 0
    assert body["request_id"]


def test_ocr_no_text_detected():
    client = make_client(FakePipeline(error=NoTextDetectedError()))
    response = client.post("/ocr", files=png_upload())
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error"] == "Unable to detect handwriting."


def test_ocr_invalid_image():
    client = make_client(FakePipeline(error=InvalidImageError("bad")))
    response = client.post("/ocr", files=png_upload())
    assert response.status_code == 400
    assert response.json()["success"] is False


def test_ocr_internal_error():
    client = make_client(FakePipeline(error=RuntimeError("boom")))
    response = client.post("/ocr", files=png_upload())
    assert response.status_code == 500
    assert response.json()["success"] is False


def test_ocr_unsupported_type(success_result):
    client = make_client(FakePipeline(result=success_result))
    files = {"image": ("doc.pdf", io.BytesIO(b"%PDF-"), "application/pdf")}
    response = client.post("/ocr", files=files)
    assert response.status_code == 415
    assert response.json()["success"] is False


def test_ocr_empty_upload(success_result):
    client = make_client(FakePipeline(result=success_result))
    files = {"image": ("empty.png", io.BytesIO(b""), "image/png")}
    response = client.post("/ocr", files=files)
    assert response.status_code == 422
    assert response.json()["success"] is False


def test_ocr_too_large(success_result):
    client = make_client(FakePipeline(result=success_result))
    big = io.BytesIO(b"\x89PNG" + b"0" * (2 * 1024 * 1024))
    files = {"image": ("big.png", big, "image/png")}
    response = client.post("/ocr", files=files)
    assert response.status_code == 413
    assert response.json()["success"] is False


def test_ocr_missing_field(success_result):
    client = make_client(FakePipeline(result=success_result))
    response = client.post("/ocr")
    assert response.status_code == 422
