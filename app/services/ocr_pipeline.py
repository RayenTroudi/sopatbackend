"""End-to-end OCR pipeline: preprocess -> detect lines -> recognize -> merge."""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.config import Settings
from app.services import image_processing as imgproc
from app.utils.logger import get_logger

if TYPE_CHECKING:
    # Type-only imports keep torch/paddle out of the import chain so unit
    # tests and tooling can import this module without the heavy libraries.
    from app.services.paddle_service import PaddleDetectionService
    from app.services.trocr_service import TrOCRService

logger = get_logger(__name__)


class NoTextDetectedError(Exception):
    """Raised when no handwriting could be found in the image."""


@dataclass
class LineResult:
    text: str
    confidence: float


@dataclass
class OCRResult:
    text: str
    confidence: float
    lines: list[LineResult] = field(default_factory=list)
    detection_time: float = 0.0
    recognition_time: float = 0.0


class OCRPipeline:
    """Orchestrates the full pipeline. Holds references to the singleton
    detection and recognition services; stateless per request."""

    def __init__(
        self,
        detector: "PaddleDetectionService",
        recognizer: "TrOCRService",
        settings: Settings,
    ) -> None:
        self._detector = detector
        self._recognizer = recognizer
        self._settings = settings

    def run(self, image_bytes: bytes) -> OCRResult:
        """Run OCR on raw image bytes. Raises InvalidImageError or
        NoTextDetectedError on failure."""
        image = imgproc.load_image(image_bytes)
        image = imgproc.resize_if_needed(
            image,
            max_dim=self._settings.max_image_dimension,
            min_dim=self._settings.min_image_dimension,
        )
        image = imgproc.deskew(image)

        # Detection runs on an enhanced copy; crops for TrOCR come from the
        # original image because TrOCR was trained on natural grayscale strokes,
        # not hard-binarized ones.
        detection_image = imgproc.enhance_for_detection(image)

        det_start = time.perf_counter()
        regions = self._detector.detect(detection_image)
        detection_time = time.perf_counter() - det_start
        logger.info("Detected %d text region(s) in %.2fs", len(regions), detection_time)

        if not regions:
            raise NoTextDetectedError("Unable to detect handwriting.")

        crops = [imgproc.crop_region(image, region.box) for region in regions]
        crops = [c for c in crops if c.size > 0 and min(c.shape[:2]) >= 4]
        if not crops:
            raise NoTextDetectedError("Unable to detect handwriting.")

        rec_start = time.perf_counter()
        recognized = self._recognizer.recognize(crops)
        recognition_time = time.perf_counter() - rec_start
        logger.info("Recognized %d line(s) in %.2fs", len(recognized), recognition_time)

        lines = [
            LineResult(text=text, confidence=conf)
            for text, conf in recognized
            if text
        ]
        if not lines:
            raise NoTextDetectedError("Unable to detect handwriting.")

        full_text = "\n".join(line.text for line in lines)
        avg_confidence = round(sum(l.confidence for l in lines) / len(lines), 4)

        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            lines=lines,
            detection_time=round(detection_time, 3),
            recognition_time=round(recognition_time, 3),
        )
