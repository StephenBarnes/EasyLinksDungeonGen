"""Geometry helpers for working with ports, rotations, and rectangles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class Rect:
    """Immutable axis-aligned rectangle using integer tile coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def max_x(self) -> int:
        """Right edge (exclusive)."""
        return self.x + self.width

    @property
    def max_y(self) -> int:
        """Bottom edge (exclusive)."""
        return self.y + self.height

    def overlaps(self, other: Rect) -> bool:
        """Return True when the interior of this rect intersects another rect."""
        if self.max_x <= other.x or other.max_x <= self.x:
            return False
        if self.max_y <= other.y or other.max_y <= self.y:
            return False
        return True

    def expand(self, margin: int) -> Rect:
        """Return a rect grown outward by ``margin`` tiles on all sides."""
        if margin == 0:
            return self
        return Rect(
            self.x - margin,
            self.y - margin,
            self.width + 2 * margin,
            self.height + 2 * margin,
        )

    def contains(self, point: Point) -> bool:
        """Return True if the provided point lies inside this rect."""
        px, py = point
        return self.x <= px < self.max_x and self.y <= py < self.max_y

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return the rect as an ``(x, y, width, height)`` tuple."""
        return self.x, self.y, self.width, self.height


def port_tiles_from_world_pos(world_x: float, world_y: float) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Derive the 1x2 or 2x1 tile footprint for a port center point."""
    frac_x = world_x - math.floor(world_x)
    frac_y = world_y - math.floor(world_y)
    if math.isclose(frac_x, 0.5, abs_tol=1e-6):
        base_y = int(round(world_y))
        return (
            (int(math.floor(world_x)), base_y),
            (int(math.ceil(world_x)), base_y),
        )
    if math.isclose(frac_y, 0.5, abs_tol=1e-6):
        base_x = int(round(world_x))
        return (
            (base_x, int(math.floor(world_y))),
            (base_x, int(math.ceil(world_y))),
        )
    raise ValueError(f"Port center must have exactly one half coordinate, got {(world_x, world_y)}")


def rotate_point(px: float, py: float, width: int, height: int, rotation: int) -> Tuple[float, float]:
    if rotation == 0:
        return px, py
    if rotation == 90:
        return py, width - 1 - px
    if rotation == 180:
        return width - 1 - px, height - 1 - py
    if rotation == 270:
        return height - 1 - py, px
    raise ValueError(f"Unsupported rotation {rotation}")


def rotate_direction(dx: int, dy: int, rotation: int) -> Tuple[int, int]:
    if rotation == 0:
        return dx, dy
    if rotation == 90:
        return dy, -dx
    if rotation == 180:
        return -dx, -dy
    if rotation == 270:
        return -dy, dx
    raise ValueError(f"Unsupported rotation {rotation}")
