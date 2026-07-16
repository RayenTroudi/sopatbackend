"""Image loading and preprocessing utilities for the OCR pipeline."""

import io

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.utils.logger import get_logger

logger = get_logger(__name__)


class InvalidImageError(Exception):
    """Raised when the uploaded bytes cannot be decoded as an image."""


def load_image(data: bytes) -> np.ndarray:
    """Decode raw upload bytes into an RGB numpy array.

    Applies EXIF orientation correction so photos taken on phones
    (the Flutter use case) are always upright.
    """
    if not data:
        raise InvalidImageError("Empty upload.")
    try:
        pil_image = Image.open(io.BytesIO(data))
        pil_image = ImageOps.exif_transpose(pil_image)
        pil_image = pil_image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"Could not decode image: {exc}") from exc
    return np.asarray(pil_image)


def resize_if_needed(image: np.ndarray, max_dim: int, min_dim: int) -> np.ndarray:
    """Keep the longest side within [min_dim, max_dim], preserving aspect ratio."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > max_dim:
        scale = max_dim / longest
    elif longest < min_dim:
        scale = min_dim / longest
    else:
        return image
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    logger.debug("Resizing image from %sx%s to %sx%s", w, h, *new_size)
    return cv2.resize(image, new_size, interpolation=interpolation)


def deskew(image: np.ndarray) -> np.ndarray:
    """Estimate global skew from text pixels and rotate to correct it.

    Skips correction for negligible angles to avoid interpolation blur.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 50:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.5 or abs(angle) > 20:
        # Tiny angles are noise; big ones are usually a mis-detected layout.
        return image

    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    logger.debug("Deskewing image by %.2f degrees", angle)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def enhance_for_detection(image: np.ndarray) -> np.ndarray:
    """Light cleanup that helps text detection without destroying strokes:
    grayscale -> denoise -> adaptive threshold, returned as 3-channel RGB
    because PaddleOCR expects a color image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def crop_region(image: np.ndarray, box: np.ndarray, padding: int = 4) -> np.ndarray:
    """Crop the axis-aligned bounding rectangle of a detection polygon,
    with a small padding so ascenders/descenders are not clipped."""
    h, w = image.shape[:2]
    xs, ys = box[:, 0], box[:, 1]
    x1 = max(0, int(np.floor(xs.min())) - padding)
    y1 = max(0, int(np.floor(ys.min())) - padding)
    x2 = min(w, int(np.ceil(xs.max())) + padding)
    y2 = min(h, int(np.ceil(ys.max())) + padding)
    return image[y1:y2, x1:x2]
