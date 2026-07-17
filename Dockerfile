FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    DEVICE=cpu

# System libraries needed by OpenCV and PaddleOCR, plus a C toolchain for
# dependencies that ship no prebuilt wheel (e.g. stringzilla)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first (much smaller than the default CUDA build)
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3.0" \
    && pip install -r requirements.txt

# Non-root user; model caches live in its home so no chown of large trees
RUN useradd --create-home appuser
ENV HF_HOME=/home/appuser/.cache/huggingface

USER appuser

# Pre-download models at build time (as appuser, separate layers) so
# container startup is fast and the image works offline. These layers come
# BEFORE the app code copy so code-only changes rebuild in seconds instead
# of re-downloading ~1.5 GB of model weights.
RUN python -c "\
from transformers import TrOCRProcessor, VisionEncoderDecoderModel; \
TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten'); \
VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')"

RUN python -c "\
from paddleocr import PaddleOCR; \
PaddleOCR(lang='en', use_angle_cls=True, det=True, rec=False, show_log=False)"

COPY --chown=appuser:appuser app ./app

EXPOSE 8000

# Shell form so ${PORT} expands — Railway (and most PaaS hosts) assign the
# port dynamically via this env var instead of always using 8000.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
