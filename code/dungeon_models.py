"""Core dataclasses used by the dungeon generator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Tuple

from dungeon_constants import VALID_ROTATIONS
from dungeon_geometry import port_tiles_from_world_pos, rotate_direction, rotate_point


@dataclass
class PortTemplate:
    """Doorway specification in template-local coordinates."""

    pos: Tuple[float, float]
    direction: Tuple[int, int]
    widths: FrozenSet[int]

    def __post_init__(self) -> None:
        dx, dy = self.direction
        if dx not in (-1, 0, 1) or dy not in (-1, 0, 1):
            raise ValueError(f"Port direction coords must be +-1 and 0, got {dx, dy}")
        if abs(dx) + abs(dy) != 1:
            raise ValueError(f"Port directions must be axis-aligned, got {dx, dy}")

        px, py = (float(self.pos[0]), float(self.pos[1]))
        if dx != 0:
            if not math.isclose(py - math.floor(py), 0.5, abs_tol=1e-6):
                raise ValueError(f"Vertical doorway must have .5 fractional y, got {py}")
            if not math.isclose(px, round(px), abs_tol=1e-6):
                raise ValueError(f"Vertical doorway must have integer x, got {px}")
        else:
            if not math.isclose(px - math.floor(px), 0.5, abs_tol=1e-6):
                raise ValueError(f"Horizontal doorway must have .5 fractional x, got {px}")
            if not math.isclose(py, round(py), abs_tol=1e-6):
                raise ValueError(f"Horizontal doorway must have integer y, got {py}")

        self.pos = (px, py)
        self.widths = frozenset(int(w) for w in self.widths)


@dataclass
class RoomTemplate:
    """Defines the blueprint for a type of room in its default rotation."""

    name: str
    size: Tuple[int, int]
    ports: List[PortTemplate]
    root_weight: float = 1.0  # Weight for random choice when choosing root room to place.
    direct_weight: float = 1.0  # Weight for random choice when creating direct-linked rooms.

    def __post_init__(self) -> None:
        width, height = self.size
        self.size = (int(width), int(height))


@dataclass(frozen=True)
class WorldPort:
    """Port information after converting to world coordinates."""

    pos: Tuple[float, float]
    tiles: Tuple[Tuple[int, int], Tuple[int, int]]
    direction: Tuple[int, int]
    widths: FrozenSet[int]


@dataclass(frozen=True)
class CorridorGeometry:
    """Holds the tile layout for a straight corridor between two ports."""

    tiles: Tuple[Tuple[int, int], ...]
    axis_index: int  # 0 for horizontal corridors, 1 for vertical
    port_axis_values: Tuple[int, int]


@dataclass
class Corridor:
    """Stores metadata for a carved corridor."""

    room_a_index: int
    port_a_index: int
    room_b_index: Optional[int]
    port_b_index: Optional[int]
    width: int
    geometry: CorridorGeometry
    component_id: int = -1
    joined_corridor_indices: Tuple[int, ...] = field(default_factory=tuple)
    junction_tiles: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)


@dataclass
class PlacedRoom:
    """Represents a room instance placed on the dungeon grid."""

    template: RoomTemplate
    x: int
    y: int
    rotation: int
    component_id: int = -1
    connected_port_indices: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.rotation not in VALID_ROTATIONS:
            raise ValueError(f"Unsupported rotation {self.rotation}")

    def get_available_port_indices(self) -> List[int]:
        """Return indices of template ports not yet used for a direct connection."""
        return [i for i in range(len(self.template.ports)) if i not in self.connected_port_indices]

    @property
    def width(self) -> int:
        if self.rotation in (0, 180):
            return self.template.size[0]
        return self.template.size[1]

    @property
    def height(self) -> int:
        if self.rotation in (0, 180):
            return self.template.size[1]
        return self.template.size[0]

    def get_bounds(self) -> Tuple[int, int, int, int]:
        """Returns the bounding box as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    def get_world_ports(self) -> List[WorldPort]:
        """Calculates the real-world positions and directions of ports after rotation."""
        world_ports: List[WorldPort] = []
        w, h = self.template.size

        for port in self.template.ports:
            rp_x, rp_y = rotate_point(port.pos[0], port.pos[1], w, h, self.rotation)
            rd_x, rd_y = rotate_direction(port.direction[0], port.direction[1], self.rotation)

            world_x = self.x + rp_x
            world_y = self.y + rp_y
            tiles = port_tiles_from_world_pos(world_x, world_y)

            world_ports.append(
                WorldPort(
                    pos=(world_x, world_y),
                    tiles=tiles,
                    direction=(rd_x, rd_y),
                    widths=port.widths,
                )
            )

        return world_ports
