"""API tests with mocked pipeline (no models required)."""

import io
import threading

import anyio
import httpx
import pytest
from httpx import ASGITransport
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


def make_client(pipeline: FakePipeline, **settings_kwargs) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    settings = Settings(max_upload_size_mb=1, **settings_kwargs)
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state.ocr_semaphore = anyio.Semaphore(settings.ocr_max_concurrency)
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


class BlockingPipeline:
    """Parks inside run() until released, so a second request can be observed
    contending for the concurrency slot."""

    def __init__(self, result):
        self._result = result
        self.entered = threading.Event()
        self.release = threading.Event()
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def run(self, data: bytes) -> OCRResult:
        with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.entered.set()
        self.release.wait(timeout=10)
        with self._lock:
            self.concurrent -= 1
        return self._result


def make_app(pipeline, **settings_kwargs):
    app = FastAPI()
    app.include_router(router)
    settings = Settings(max_upload_size_mb=1, **settings_kwargs)
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state.ocr_semaphore = anyio.Semaphore(settings.ocr_max_concurrency)
    return app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_pipeline_never_runs_two_at_once(success_result):
    """Two concurrent runs each hold model activations and OOM-killed the
    1 GB container. The second must queue rather than run in parallel."""
    pipeline = BlockingPipeline(success_result)
    transport = ASGITransport(app=make_app(pipeline))

    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        async with anyio.create_task_group() as tg:
            statuses = []

            async def post():
                response = await client.post("/ocr", files=png_upload())
                statuses.append(response.status_code)

            tg.start_soon(post)
            tg.start_soon(post)
            # Let the first request reach the pipeline, then confirm the
            # second is still queued outside it before releasing.
            await anyio.to_thread.run_sync(pipeline.entered.wait, 5)
            await anyio.sleep(0.3)
            assert pipeline.concurrent == 1
            pipeline.release.set()

    assert statuses == [200, 200]
    assert pipeline.max_concurrent == 1


@pytest.mark.anyio
async def test_returns_503_when_queue_wait_times_out(success_result):
    """A request that cannot get a slot in time returns a JSON 503 the client
    can explain, instead of hanging until the mobile timeout."""
    pipeline = BlockingPipeline(success_result)
    transport = ASGITransport(app=make_app(pipeline, ocr_queue_timeout_s=0.2))

    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        async with anyio.create_task_group() as tg:
            tg.start_soon(lambda: client.post("/ocr", files=png_upload()))
            await anyio.to_thread.run_sync(pipeline.entered.wait, 5)

            response = await client.post("/ocr", files=png_upload())
            assert response.status_code == 503
            body = response.json()
            assert body["success"] is False
            assert "busy" in body["error"].lower()

            pipeline.release.set()
