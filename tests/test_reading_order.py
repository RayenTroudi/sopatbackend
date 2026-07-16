"""Unit tests for reading-order sorting of detected regions."""

import numpy as np

from app.services.paddle_service import TextRegion, sort_reading_order


def _region(left: float, top: float, width: float = 100, height: float = 20) -> TextRegion:
    box = np.array(
        [
            [left, top],
            [left + width, top],
            [left + width, top + height],
            [left, top + height],
        ],
        dtype=np.float32,
    )
    return TextRegion(box=box, top=top, left=left, height=height)


def test_empty_input():
    assert sort_reading_order([]) == []


def test_top_to_bottom():
    a, b, c = _region(0, 200), _region(0, 0), _region(0, 100)
    ordered = sort_reading_order([a, b, c])
    assert [r.top for r in ordered] == [0, 100, 200]


def test_left_to_right_within_same_line():
    right, left = _region(300, 10), _region(0, 12)
    ordered = sort_reading_order([right, left])
    assert [r.left for r in ordered] == [0, 300]


def test_slightly_offset_boxes_stay_on_same_line():
    # Two words of one handwritten line, second sits 8px lower.
    first, second = _region(0, 100), _region(150, 108)
    ordered = sort_reading_order([second, first])
    assert [r.left for r in ordered] == [0, 150]


def test_distinct_lines_not_merged():
    line1, line2 = _region(0, 0, height=20), _region(0, 40, height=20)
    ordered = sort_reading_order([line2, line1])
    assert [r.top for r in ordered] == [0, 40]
