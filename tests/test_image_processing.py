"""Unit tests for image preprocessing (no models required)."""

import io

import numpy as np
import pytest
from PIL import Image

from app.services import image_processing as imgproc
from app.services.image_processing import InvalidImageError


def _png_bytes(width: int = 100, height: int = 60) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


class TestLoadImage:
    def test_valid_png(self):
        image = imgproc.load_image(_png_bytes())
        assert image.shape == (60, 100, 3)

    def test_empty_bytes(self):
        with pytest.raises(InvalidImageError):
            imgproc.load_image(b"")

    def test_garbage_bytes(self):
        with pytest.raises(InvalidImageError):
            imgproc.load_image(b"this is not an image")


class TestResize:
    def test_downscale_large(self):
        image = np.zeros((4000, 2000, 3), dtype=np.uint8)
        resized = imgproc.resize_if_needed(image, max_dim=2500, min_dim=300)
        assert max(resized.shape[:2]) == 2500

    def test_upscale_small(self):
        image = np.zeros((100, 50, 3), dtype=np.uint8)
        resized = imgproc.resize_if_needed(image, max_dim=2500, min_dim=300)
        assert max(resized.shape[:2]) == 300

    def test_no_change_in_range(self):
        image = np.zeros((500, 400, 3), dtype=np.uint8)
        resized = imgproc.resize_if_needed(image, max_dim=2500, min_dim=300)
        assert resized.shape == image.shape


class TestCropRegion:
    def test_crop_with_padding_clamped_to_bounds(self):
        image = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
        box = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        crop = imgproc.crop_region(image, box, padding=4)
        # Padding cannot go below 0; upper edge extends by 4.
        assert crop.shape[:2] == (14, 14)

    def test_crop_interior(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        box = np.array([[20, 30], [60, 30], [60, 50], [20, 50]], dtype=np.float32)
        crop = imgproc.crop_region(image, box, padding=0)
        assert crop.shape[:2] == (20, 40)


class TestEnhanceForDetection:
    def test_output_is_three_channel_binaryish(self):
        image = np.full((50, 50, 3), 128, dtype=np.uint8)
        enhanced = imgproc.enhance_for_detection(image)
        assert enhanced.shape == (50, 50, 3)
        assert set(np.unique(enhanced)).issubset({0, 255})
