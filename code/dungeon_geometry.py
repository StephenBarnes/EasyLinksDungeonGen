"""Geometry helpers for working with ports, rotations, and rectangles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple, Union


class Rotation(Enum):
    """Represents clockwise rotations in 90° increments."""

    DEG_0 = 0
    DEG_90 = 90
    DEG_180 = 180
    DEG_270 = 270

    @property
    def degrees(self) -> int:
        return self.value

    def quarter_turns(self) -> int:
        return (self.value // 90) % 4

    @classmethod
    def from_degrees(cls, value: int) -> Rotation:
        try:
            return cls(value % 360)
        except ValueError as exc:
            raise ValueError(f"Unsupported rotation {value}") from exc

    @classmethod
    def all(cls) -> Tuple[Rotation, ...]:
        return tuple(cls)

VALID_ROTATIONS = tuple(Rotation)


class Direction(Enum):
    """Cardinal directions with unit vectors on the tile grid."""

    NORTH = (0, -1)
    EAST = (1, 0)
    SOUTH = (0, 1)
    WEST = (-1, 0)

    @property
    def dx(self) -> int:
        return self.value[0]

    @property
    def dy(self) -> int:
        return self.value[1]

    @property
    def vector(self) -> Tuple[int, int]:
        return self.value

    def rotate(self, rotation: Rotation) -> Direction:
        return rotate_direction(self, rotation)

    def opposite(self) -> Direction:
        rotation = Rotation.DEG_180
        return self.rotate(rotation)

    @classmethod
    def from_tuple(cls, value: Tuple[int, int]) -> Direction:
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"Unsupported direction {value}") from exc

    def dot(self, other: Direction) -> int:
        return self.dx * other.dx + self.dy * other.dy


@dataclass(frozen=True, order=True)
class TilePos:
    """Integer tile coordinate."""

    x: int
    y: int

    def __iter__(self):
        yield self.x
        yield self.y

    def __getitem__(self, index: int) -> int:
        if index == 0:
            return self.x
        if index == 1:
            return self.y
        raise IndexError("TilePos only supports two coordinates")

    def to_tuple(self) -> Tuple[int, int]:
        return self.x, self.y

    @classmethod
    def from_tuple(cls, value: Tuple[int, int]) -> TilePos:
        return cls(*value)


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

    def contains(self, point: TilePos) -> bool:
        """Return True if the provided tile lies inside this rect."""
        return self.x <= point.x < self.max_x and self.y <= point.y < self.max_y

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return the rect as an ``(x, y, width, height)`` tuple."""
        return self.x, self.y, self.width, self.height


def port_tiles_from_world_pos(world_x: float, world_y: float) -> Tuple[TilePos, TilePos]:
    """Derive the 1x2 or 2x1 tile footprint for a port center point."""
    frac_x = world_x - math.floor(world_x)
    frac_y = world_y - math.floor(world_y)
    if math.isclose(frac_x, 0.5, abs_tol=1e-6):
        base_y = int(round(world_y))
        return (
            TilePos(int(math.floor(world_x)), base_y),
            TilePos(int(math.ceil(world_x)), base_y),
        )
    if math.isclose(frac_y, 0.5, abs_tol=1e-6):
        base_x = int(round(world_x))
        return (
            TilePos(base_x, int(math.floor(world_y))),
            TilePos(base_x, int(math.ceil(world_y))),
        )
    raise ValueError(f"Port center must have exactly one half coordinate, got {(world_x, world_y)}")


def rotate_point(px: float, py: float, width: int, height: int, rotation: Rotation) -> Tuple[float, float]:
    if rotation is Rotation.DEG_0:
        return px, py
    if rotation is Rotation.DEG_90:
        return py, width - 1 - px
    if rotation is Rotation.DEG_180:
        return width - 1 - px, height - 1 - py
    if rotation is Rotation.DEG_270:
        return height - 1 - py, px
    raise AssertionError(f"Unhandled rotation {rotation}")


def rotate_direction(direction: Direction, rotation: Rotation) -> Direction:
    dx, dy = direction.vector
    if rotation is Rotation.DEG_0:
        return direction
    if rotation is Rotation.DEG_90:
        return Direction.from_tuple((dy, -dx))
    if rotation is Rotation.DEG_180:
        return Direction.from_tuple((-dx, -dy))
    if rotation is Rotation.DEG_270:
        return Direction.from_tuple((-dy, dx))
    raise AssertionError(f"Unhandled rotation {rotation}")
