"""Core dataclasses used by the dungeon generator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, List, Optional, Tuple

from dungeon_constants import DOOR_MACRO_ALIGNMENT_OFFSETS, MACRO_GRID_SIZE, VALID_ROTATIONS
from dungeon_geometry import port_tiles_from_world_pos, rotate_direction, rotate_point


class RoomKind(Enum):
    """Specifies the ways that a given RoomTemplate can be used."""
    STANDALONE = 0 # Created by step 1, either placed as a root or created directly linked to existing room.
    T_JUNCTION = 1 # Connects 2 corridors/rooms in line with each other, plus a third corridor/room perpendicular.
    BEND = 2 # Connects to 2 passages/rooms at right angles.
    FOUR_WAY = 3 # Connects to 4 passages, or poss


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
    kinds: FrozenSet[RoomKind]
    root_weight_middle: float = 1.0  # Weight when placing room near the dungeon center.
    root_weight_edge: float = 1.0  # Weight when placing room near the dungeon outskirts.
    root_weight_intermediate: float = 1.0  # Weight when placement is neither central nor edge.
    direct_weight: float = 1.0  # Weight for random choice when creating direct-linked rooms.
    preferred_center_facing_dir: Optional[Tuple[int, int]] = None
    allow_door_overlaps: bool = False

    def __post_init__(self) -> None:
        self.kinds = frozenset(self.kinds)
        self.validate()
        width, height = self.size
        self.size = (int(width), int(height))
    
    def validate(self):
        """Run several validations to check that our room templates and their ports obey constraints."""
        if self.size[0] <= 0 or self.size[1] <= 0:
            raise ValueError(f"Room {self.name} must have positive non-zero dimensions")

        if self.preferred_center_facing_dir is not None:
            dx, dy = self.preferred_center_facing_dir
            if (dx, dy) not in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                raise ValueError(
                    "preferred_center_facing_dir must be a cardinal direction tuple"
                )
            self.preferred_center_facing_dir = (int(dx), int(dy))

        if not self.ports:
            raise ValueError(f"Room {self.name} must define at least one port")

        if not self.kinds:
            raise ValueError(f"Room {self.name} must specify at least one RoomKind")

        self.validate_ports()
        
        if RoomKind.STANDALONE in self.kinds:
            self.validate_macrogrid_alignment()

    def validate_ports(self):
        """Validate per-port geometry and collect occupancy footprints at max width."""
        occupied_tiles: dict[tuple[int, int], int] = {}
        width, height = self.size
        eps = 1e-6

        def _max_width_tiles(port: PortTemplate, max_width: int) -> tuple[tuple[int, int], ...]:
            base_tiles = port_tiles_from_world_pos(port.pos[0], port.pos[1])
            tile_a, tile_b = base_tiles
            if tile_a[0] == tile_b[0]:
                # Vertical doorway (extends along Y).
                x = tile_a[0]
                y0, y1 = sorted((tile_a[1], tile_b[1]))
                extent = (max_width // 2) - 1
                start_y = y0 - extent
                end_y = start_y + max_width - 1
                return tuple((x, y) for y in range(start_y, end_y + 1))
            # Horizontal doorway (extends along X).
            y = tile_a[1]
            x0, x1 = sorted((tile_a[0], tile_b[0]))
            extent = (max_width // 2) - 1
            start_x = x0 - extent
            end_x = start_x + max_width - 1
            return tuple((x, y) for x in range(start_x, end_x + 1))

        for port_index, port in enumerate(self.ports):
            px, py = port.pos
            dx, dy = port.direction

            if not port.widths:
                raise ValueError(f"Room {self.name} port {port_index} must allow at least one width")

            for width_option in port.widths:
                if width_option <= 0:
                    raise ValueError(
                        f"Room {self.name} port {port_index} width {width_option} must be positive"
                    )
                if width_option % 2 != 0:
                    raise ValueError(
                        f"Room {self.name} port {port_index} width {width_option} must be even"
                    )

            max_width = max(port.widths)

            if dx == 0 and dy != 0:
                # Door on north/south edge, spans X axis inside room.
                expected_y = 0.0 if dy < 0 else float(height - 1)
                if not math.isclose(py, expected_y, abs_tol=eps):
                    raise ValueError(
                        f"Room {self.name} port {port_index} must lie on y={expected_y}, got {py}"
                    )
                if px < -eps or px > (width - 1) + eps:
                    raise ValueError(
                        f"Room {self.name} port {port_index} x={px} outside room width {width}"
                    )
                if max_width > width:
                    raise ValueError(
                        f"Room {self.name} port {port_index} width {max_width} exceeds room width {width}"
                    )
            elif dy == 0 and dx != 0:
                # Door on west/east edge, spans Y axis inside room.
                expected_x = 0.0 if dx < 0 else float(width - 1)
                if not math.isclose(px, expected_x, abs_tol=eps):
                    raise ValueError(
                        f"Room {self.name} port {port_index} must lie on x={expected_x}, got {px}"
                    )
                if py < -eps or py > (height - 1) + eps:
                    raise ValueError(
                        f"Room {self.name} port {port_index} y={py} outside room height {height}"
                    )
                if max_width > height:
                    raise ValueError(
                        f"Room {self.name} port {port_index} width {max_width} exceeds room height {height}"
                    )
            else:
                raise ValueError(
                    f"Room {self.name} port {port_index} must face outward along room boundary"
                )

            max_tiles = _max_width_tiles(port, max_width)
            for tile_x, tile_y in max_tiles:
                if tile_x < 0 or tile_x >= width or tile_y < 0 or tile_y >= height:
                    raise ValueError(
                        f"Room {self.name} port {port_index} width {max_width} exceeds room bounds"
                    )
                if (not self.allow_door_overlaps) and (tile_x, tile_y) in occupied_tiles:
                    raise ValueError(
                        f"Room {self.name} ports {occupied_tiles[(tile_x, tile_y)]} and {port_index} overlap at tile {(tile_x, tile_y)} when at max width"
                    )
                occupied_tiles[(tile_x, tile_y)] = port_index

    def validate_macrogrid_alignment(self):
        width, height = self.size
        eps = 1e-6
        for rotation in VALID_ROTATIONS:
            rotated_ports = []
            for port in self.ports:
                rp_x, rp_y = rotate_point(port.pos[0], port.pos[1], width, height, rotation)
                rd_x, rd_y = rotate_direction(port.direction[0], port.direction[1], rotation)
                rotated_ports.append(((rp_x, rp_y), (rd_x, rd_y)))

            ref_offset_x = None
            ref_offset_y = None
            for (rp_x, rp_y), direction in rotated_ports:
                try:
                    offset_x, offset_y = DOOR_MACRO_ALIGNMENT_OFFSETS[direction]
                except KeyError as exc:
                    raise ValueError(
                        f"Room {self.name} has unsupported port direction {direction}"
                    ) from exc

                # Each port must be able to anchor the room with an integer translation.
                shift_x = offset_x - rp_x
                shift_y = offset_y - rp_y
                if not math.isclose(shift_x, round(shift_x), abs_tol=eps):
                    raise ValueError(
                        f"Room {self.name} rotation {rotation} port cannot align to macro-grid with integer X shift"
                    )
                if not math.isclose(shift_y, round(shift_y), abs_tol=eps):
                    raise ValueError(
                        f"Room {self.name} rotation {rotation} port cannot align to macro-grid with integer Y shift"
                    )

                value_x = rp_x - offset_x
                value_y = rp_y - offset_y

                if ref_offset_x is None:
                    ref_offset_x = value_x
                    ref_offset_y = value_y
                else:
                    delta_x = value_x - ref_offset_x
                    delta_y = value_y - ref_offset_y
                    normalized_x = delta_x / float(MACRO_GRID_SIZE)
                    normalized_y = delta_y / float(MACRO_GRID_SIZE)
                    if not math.isclose(normalized_x, round(normalized_x), abs_tol=eps):
                        raise ValueError(
                            f"Room {self.name} rotation {rotation} ports misaligned on macro-grid (x)"
                        )
                    if not math.isclose(normalized_y, round(normalized_y), abs_tol=eps):
                        raise ValueError(
                            f"Room {self.name} rotation {rotation} ports misaligned on macro-grid (y)"
                        )

@dataclass(frozen=True)
class WorldPort:
    """Port information after converting to world coordinates."""

    pos: Tuple[float, float]
    tiles: Tuple[Tuple[int, int], Tuple[int, int]]
    direction: Tuple[int, int]
    widths: FrozenSet[int]


@dataclass(frozen=True)
class CorridorGeometry:
    """Holds the tile layout for a corridor between two ports."""

    tiles: Tuple[Tuple[int, int], ...]
    axis_index: Optional[int]  # 0 for horizontal, 1 for vertical, None for joints
    port_axis_values: Tuple[int, int]
    cross_coords: Tuple[int, ...] = ()


@dataclass
class Corridor:
    """Stores metadata for a carved corridor."""

    room_a_index: Optional[int]
    port_a_index: Optional[int]
    room_b_index: Optional[int]
    port_b_index: Optional[int]
    width: int
    geometry: CorridorGeometry
    component_id: int = -1


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
