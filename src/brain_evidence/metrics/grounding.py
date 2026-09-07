"""Bounding-box parsing and grounding metrics."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import TypeAlias

Box: TypeAlias = tuple[float, float, float, float]

_NUMBER = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
_PATTERNS = (
    re.compile(
        rf"\s*\[\s*{_NUMBER}\s*,\s*{_NUMBER}\s*,\s*"
        rf"{_NUMBER}\s*,\s*{_NUMBER}\s*\]\s*"
    ),
    re.compile(
        rf"\s*<box>\s*\(\s*{_NUMBER}\s*,\s*{_NUMBER}\s*\)\s*,\s*"
        rf"\(\s*{_NUMBER}\s*,\s*{_NUMBER}\s*\)\s*</box>\s*"
    ),
    re.compile(
        rf"\s*\(\s*{_NUMBER}\s*,\s*{_NUMBER}\s*\)\s*,\s*"
        rf"\(\s*{_NUMBER}\s*,\s*{_NUMBER}\s*\)\s*"
    ),
)


def _validate_box(coordinates: Sequence[float]) -> Box:
    if len(coordinates) != 4:
        raise ValueError("a box must contain exactly four coordinates")
    try:
        x_min, y_min, x_max, y_max = (float(value) for value in coordinates)
    except (TypeError, ValueError) as error:
        raise ValueError("box coordinates must be numeric") from error
    if not all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max)):
        raise ValueError("box coordinates must be finite")
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("box coordinates must have positive width and height")
    return x_min, y_min, x_max, y_max


def parse_box(value: str | Sequence[float]) -> Box:
    """Parse a generated box and reject malformed or degenerate geometry."""

    if not isinstance(value, str):
        return _validate_box(value)

    for pattern in _PATTERNS:
        match = pattern.fullmatch(value)
        if match is not None:
            return _validate_box(tuple(float(item) for item in match.groups()))
    raise ValueError("box must contain exactly two coordinate corners")


def box_iou(first: str | Sequence[float], second: str | Sequence[float]) -> float:
    """Return intersection-over-union for two valid axis-aligned boxes."""

    first_x_min, first_y_min, first_x_max, first_y_max = parse_box(first)
    second_x_min, second_y_min, second_x_max, second_y_max = parse_box(second)

    intersection_width = max(
        0.0, min(first_x_max, second_x_max) - max(first_x_min, second_x_min)
    )
    intersection_height = max(
        0.0, min(first_y_max, second_y_max) - max(first_y_min, second_y_min)
    )
    intersection = intersection_width * intersection_height
    first_area = (first_x_max - first_x_min) * (first_y_max - first_y_min)
    second_area = (second_x_max - second_x_min) * (second_y_max - second_y_min)
    return intersection / (first_area + second_area - intersection)


__all__ = ["Box", "box_iou", "parse_box"]
