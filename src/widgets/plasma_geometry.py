"""Window placement helpers for the plasma widget"""

from dataclasses import dataclass
from typing import NamedTuple, Optional

CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")


class Area(NamedTuple):
    """A screen or monitor rectangle"""

    x: int
    y: int
    width: int
    height: int


@dataclass
class WindowPlacement:
    """Where the widget sits and how big it is"""

    corner: str = "bottom-left"
    offset: int = 20  # Offset from edges in pixels
    window_x: int = 0
    window_y: int = 0
    width: int = 0
    height: int = 0


def corner_position(corner: str, area: Area, size: tuple[int, int], offset: int) -> Optional[tuple[int, int]]:
    """Top-left window coordinates that place a window of ``size`` in ``corner`` of ``area``"""
    width, height = size
    left = area.x + offset
    right = area.x + area.width - width - offset
    top = area.y + offset
    bottom = area.y + area.height - height - offset
    positions = {
        "top-left": (left, top),
        "top-right": (right, top),
        "bottom-left": (left, bottom),
        "bottom-right": (right, bottom),
    }
    return positions.get(corner)


def in_resize_corner(point: tuple[float, float], size: tuple[int, int], corner_size: int = 20) -> bool:
    """True if ``point`` lies in the bottom-right resize corner"""
    return point[0] > size[0] - corner_size and point[1] > size[1] - corner_size


def resized_size(start_size: int, delta: float, bounds: tuple[int, int]) -> float:
    """New square size after a resize drag of ``delta``, clamped to ``bounds`` (min, max)"""
    return max(bounds[0], min(bounds[1], start_size + delta))


def dominant_delta(dx: float, dy: float) -> float:
    """The larger of dx or dy (by magnitude), keeping its sign"""
    return dx if abs(dx) > abs(dy) else dy
