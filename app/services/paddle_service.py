"""Text line detection using PaddleOCR (detection only — no recognition)."""

from dataclasses import dataclass

import numpy as np

from app.config import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TextRegion:
    """A detected text line: polygon corners and its bounding-box anchors
    used for reading-order sorting."""

    box: np.ndarray  # shape (4, 2), float
    top: float
    left: float
    height: float


class PaddleDetectionService:
    """Wraps PaddleOCR in detection-only mode. Instantiate once at startup."""

    def __init__(self, settings: Settings) -> None:
        # Imported lazily so unit tests can run without paddle installed.
        from paddleocr import PaddleOCR

        logger.info("Loading PaddleOCR detector (lang=%s)...", settings.paddle_lang)
        self._use_angle_cls = settings.use_angle_cls
        self._ocr = PaddleOCR(
            lang=settings.paddle_lang,
            use_angle_cls=self._use_angle_cls,
            det=True,
            rec=False,
            show_log=False,
            use_gpu=settings.device != "cpu",
        )
        logger.info("PaddleOCR detector ready (angle_cls=%s).", self._use_angle_cls)

    def detect(self, image: np.ndarray) -> list[TextRegion]:
        """Return detected text regions sorted in natural reading order
        (top to bottom, then left to right within a line band)."""
        result = self._ocr.ocr(image, rec=False, cls=self._use_angle_cls)
        boxes = result[0] if result and result[0] is not None else []

        regions: list[TextRegion] = []
        for raw_box in boxes:
            box = np.asarray(raw_box, dtype=np.float32)
            ys, xs = box[:, 1], box[:, 0]
            regions.append(
                TextRegion(
                    box=box,
                    top=float(ys.min()),
                    left=float(xs.min()),
                    height=float(ys.max() - ys.min()),
                )
            )

        return sort_reading_order(regions)


def sort_reading_order(regions: list[TextRegion]) -> list[TextRegion]:
    """Sort regions top-to-bottom, grouping boxes whose vertical centers fall
    within the same line band, then left-to-right inside each band."""
    if not regions:
        return []

    by_top = sorted(regions, key=lambda r: r.top)
    lines: list[list[TextRegion]] = [[by_top[0]]]
    for region in by_top[1:]:
        current_line = lines[-1]
        band_top = min(r.top for r in current_line)
        band_height = max(r.height for r in current_line)
        # Same line if the region starts within half a line-height of the band.
        if region.top <= band_top + band_height * 0.5:
            current_line.append(region)
        else:
            lines.append([region])

    ordered: list[TextRegion] = []
    for line in lines:
        ordered.extend(sorted(line, key=lambda r: r.left))
    return ordered
