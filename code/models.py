"""Core dataclasses used by the dungeon generator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple

from constants import door_macro_alignment_offsets
from geometry import (
    Direction,
    Rotation,
    Rect,
    TilePos,
    port_tiles_from_world_pos,
    rotate_direction,
    rotate_point,
    VALID_ROTATIONS,
)


class RoomKind(Enum):
    """Specifies the ways that a given RoomTemplate can be used."""
    STANDALONE = 0 # Created by step 1 when placing root rooms.
    T_JUNCTION = 1 # Connects 2 corridors/rooms in line with each other, plus a third corridor/room perpendicular.
    BEND = 2 # Connects to 2 passages/rooms at right angles.
    THROUGH = 3 # Used to break up long passages; connects to 2 passages on opposite sides.
    FOUR_WAY = 4 # Connects to 4 passages or room ports.
    DIRECT_LINKED = 5 # Created by step 1 when attaching rooms directly to the initial root tree.


@dataclass
class PortTemplate:
    """Doorway specification in template-local coordinates."""

    pos: Tuple[float, float]
    direction: Direction
    widths: FrozenSet[int]

    def __post_init__(self) -> None:
        direction = self.direction
        dx, dy = direction.dx, direction.dy
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
class RotatedPortTemplate:
    """Port geometry expressed in template-local coordinates after rotation."""

    pos: Tuple[float, float]
    tiles: Tuple[TilePos, TilePos]
    direction: Direction
    widths: FrozenSet[int]


@dataclass(frozen=True)
class RotatedRoomVariant:
    """Precomputed port data for a specific rotation of a room template."""

    rotation: Rotation
    ports: Tuple[RotatedPortTemplate, ...]
    ports_by_direction: Mapping[Direction, Tuple[int, ...]]


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
    first_root_weight: float = 1.0  # Weight for the very first root room.
    direct_linked_weight: float = 1.0  # Weight when selecting a direct-linked room during initial growth.
    t_junction_weight: float = 1.0  # Weight when selecting as a T-junction special room.
    bend_weight: float = 1.0  # Weight when selecting as a bend special room.
    four_way_weight: float = 1.0  # Weight when selecting as a 4-way intersection.
    through_weight: float = 1.0 # Weight when selecting as a through-room.
    preferred_center_facing_dir: Optional[Direction] = None
    allow_door_overlaps: bool = False
    macro_grid_size: int = 4
    is_symmetric_90: bool = field(init=False)
    is_symmetric_180: bool = field(init=False)
    _unique_rotations: Tuple[Rotation, ...] = field(init=False, repr=False)
    _rotation_alias_map: Dict[Rotation, Rotation] = field(init=False, repr=False)
    _rotation_variants: Dict[Rotation, RotatedRoomVariant] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        width, height = self.size
        self.size = (int(width), int(height))
        self.macro_grid_size = int(self.macro_grid_size)
        self.kinds = frozenset(self.kinds)
        self.root_weight_middle = float(self.root_weight_middle)
        self.root_weight_edge = float(self.root_weight_edge)
        self.root_weight_intermediate = float(self.root_weight_intermediate)
        self.first_root_weight = float(self.first_root_weight)
        self.direct_linked_weight = float(self.direct_linked_weight)
        self.t_junction_weight = float(self.t_junction_weight)
        self.bend_weight = float(self.bend_weight)
        self.four_way_weight = float(self.four_way_weight)
        self.validate()
        self._initialize_symmetry_flags()
        self._precompute_rotations()

    def validate(self):
        """Run several validations to check that our room templates and their ports obey constraints."""
        if self.size[0] <= 0 or self.size[1] <= 0:
            raise ValueError(f"Room {self.name} must have positive non-zero dimensions")

        if self.macro_grid_size <= 0:
            raise ValueError(f"Room {self.name} must specify a positive macro-grid size")

        if not self.ports:
            raise ValueError(f"Room {self.name} must define at least one port")

        if not self.kinds:
            raise ValueError(f"Room {self.name} must specify at least one RoomKind")

        if self.first_root_weight < 0:
            raise ValueError(f"Room {self.name} first-root weight must be non-negative")
        if self.direct_linked_weight < 0:
            raise ValueError(f"Room {self.name} direct-linked weight must be non-negative")
        if self.t_junction_weight < 0:
            raise ValueError(f"Room {self.name} T-junction weight must be non-negative")
        if self.bend_weight < 0:
            raise ValueError(f"Room {self.name} bend weight must be non-negative")
        if self.four_way_weight < 0:
            raise ValueError(f"Room {self.name} four-way weight must be non-negative")

        self.validate_ports()
        
        if RoomKind.STANDALONE in self.kinds:
            self.validate_macrogrid_alignment()

    def weight_for_kind(self, kind: RoomKind) -> float:
        """Return the selection weight for the given special-room kind."""
        if kind is RoomKind.DIRECT_LINKED:
            return self.direct_linked_weight
        if kind is RoomKind.T_JUNCTION:
            return self.t_junction_weight
        if kind is RoomKind.BEND:
            return self.bend_weight
        if kind is RoomKind.FOUR_WAY:
            return self.four_way_weight
        if kind is RoomKind.THROUGH:
            return self.through_weight
        return 1.0

    def validate_ports(self):
        """Validate per-port geometry and collect occupancy footprints at max width."""
        occupied_tiles: dict[TilePos, int] = {}
        width, height = self.size
        eps = 1e-6

        def _max_width_tiles(port: PortTemplate, max_width: int) -> tuple[TilePos, ...]:
            base_tiles = port_tiles_from_world_pos(port.pos[0], port.pos[1])
            tile_a, tile_b = base_tiles
            if tile_a.x == tile_b.x:
                # Vertical doorway (extends along Y).
                x = tile_a.x
                y0, y1 = sorted((tile_a.y, tile_b.y))
                extent = (max_width // 2) - 1
                start_y = y0 - extent
                end_y = start_y + max_width - 1
                return tuple(TilePos(x, y) for y in range(start_y, end_y + 1))
            # Horizontal doorway (extends along X).
            y = tile_a.y
            x0, x1 = sorted((tile_a.x, tile_b.x))
            extent = (max_width // 2) - 1
            start_x = x0 - extent
            end_x = start_x + max_width - 1
            return tuple(TilePos(x, y) for x in range(start_x, end_x + 1))

        for port_index, port in enumerate(self.ports):
            px, py = port.pos
            direction = port.direction
            dx, dy = direction.dx, direction.dy

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
            for tile in max_tiles:
                if tile.x < 0 or tile.x >= width or tile.y < 0 or tile.y >= height:
                    raise ValueError(
                        f"Room {self.name} port {port_index} width {max_width} exceeds room bounds"
                    )
                if (not self.allow_door_overlaps) and tile in occupied_tiles:
                    raise ValueError(
                        f"Room {self.name} ports {occupied_tiles[tile]} and {port_index} overlap at tile {tile.to_tuple()} when at max width"
                    )
                occupied_tiles[tile] = port_index

    def validate_macrogrid_alignment(self):
        width, height = self.size
        eps = 1e-6
        offsets = door_macro_alignment_offsets(self.macro_grid_size)
        for rotation in VALID_ROTATIONS:
            rotated_ports = []
            for port in self.ports:
                rp_x, rp_y = rotate_point(port.pos[0], port.pos[1], width, height, rotation)
                rotated_direction = rotate_direction(port.direction, rotation)
                rotated_ports.append(((rp_x, rp_y), rotated_direction))

            ref_offset_x = None
            ref_offset_y = None
            for (rp_x, rp_y), direction in rotated_ports:
                try:
                    offset_x, offset_y = offsets[direction]
                except KeyError as exc:
                    raise ValueError(
                        f"Room {self.name} has unsupported port direction {direction}"
                    ) from exc

                # Each port must be able to anchor the room with an integer translation.
                shift_x = offset_x - rp_x
                shift_y = offset_y - rp_y
                if not math.isclose(shift_x, round(shift_x), abs_tol=eps):
                    raise ValueError(
                        f"Room {self.name} rotation {rotation.degrees} port cannot align to macro-grid with integer X shift"
                    )
                if not math.isclose(shift_y, round(shift_y), abs_tol=eps):
                    raise ValueError(
                        f"Room {self.name} rotation {rotation.degrees} port cannot align to macro-grid with integer Y shift"
                    )

                value_x = rp_x - offset_x
                value_y = rp_y - offset_y

                if ref_offset_x is None:
                    ref_offset_x = value_x
                    ref_offset_y = value_y
                else:
                    delta_x = value_x - ref_offset_x
                    delta_y = value_y - ref_offset_y
                    normalized_x = delta_x / float(self.macro_grid_size)
                    normalized_y = delta_y / float(self.macro_grid_size)
                    if not math.isclose(normalized_x, round(normalized_x), abs_tol=eps):
                        raise ValueError(
                            f"Room {self.name} rotation {rotation.degrees} ports misaligned on macro-grid (x)"
                        )
                    if not math.isclose(normalized_y, round(normalized_y), abs_tol=eps):
                        raise ValueError(
                            f"Room {self.name} rotation {rotation.degrees} ports misaligned on macro-grid (y)"
                        )

    def _initialize_symmetry_flags(self) -> None:
        base_signature = self._rotation_signature(Rotation.DEG_0)
        symmetric_90 = self._has_rotational_symmetry(Rotation.DEG_90, base_signature)
        self.is_symmetric_90 = symmetric_90
        if symmetric_90:
            self.is_symmetric_180 = True
            return
        self.is_symmetric_180 = self._has_rotational_symmetry(Rotation.DEG_180, base_signature)

    def _has_rotational_symmetry(
        self,
        rotation: Rotation,
        base_signature: Tuple[Tuple[int, int, Tuple[int, int], Tuple[int, ...]], ...],
    ) -> bool:
        if rotation in (Rotation.DEG_90, Rotation.DEG_270) and self.size[0] != self.size[1]:
            return False
        if rotation is Rotation.DEG_0:
            return True
        rotated_signature = self._rotation_signature(rotation)
        return rotated_signature == base_signature

    def _rotation_signature(
        self,
        rotation: Rotation,
    ) -> Tuple[Tuple[int, int, Tuple[int, int], Tuple[int, ...]], ...]:
        width, height = self.size
        entries: List[Tuple[int, int, Tuple[int, int], Tuple[int, ...]]] = []
        for port in self.ports:
            rotated_x, rotated_y = rotate_point(port.pos[0], port.pos[1], width, height, rotation)
            rotated_dir = rotate_direction(port.direction, rotation)
            signature = (
                int(round(rotated_x * 2)),
                int(round(rotated_y * 2)),
                rotated_dir.value,
                tuple(sorted(port.widths)),
            )
            entries.append(signature)
        entries.sort()
        return tuple(entries)

    def _precompute_rotations(self) -> None:
        alias_map: Dict[Rotation, Rotation] = {}
        variants: Dict[Rotation, RotatedRoomVariant] = {}
        unique_rotations: List[Rotation] = []
        seen_signatures: Dict[Tuple[Tuple[int, int, Tuple[int, int], Tuple[int, ...]], ...], Rotation] = {}

        for rotation in VALID_ROTATIONS:
            signature = self._rotation_signature(rotation)
            canonical = seen_signatures.get(signature)
            if canonical is None:
                canonical = rotation
                seen_signatures[signature] = canonical
                unique_rotations.append(canonical)
                ports = self._build_rotated_ports(canonical)
                ports_by_dir: Dict[Direction, List[int]] = {}
                for idx, port in enumerate(ports):
                    ports_by_dir.setdefault(port.direction, []).append(idx)
                variants[canonical] = RotatedRoomVariant(
                    rotation=canonical,
                    ports=tuple(ports),
                    ports_by_direction={direction: tuple(indices) for direction, indices in ports_by_dir.items()},
                )
            alias_map[rotation] = canonical

        self._unique_rotations = tuple(unique_rotations)
        self._rotation_alias_map = alias_map
        self._rotation_variants = variants

    def _build_rotated_ports(self, rotation: Rotation) -> List[RotatedPortTemplate]:
        width, height = self.size
        ports: List[RotatedPortTemplate] = []
        for port in self.ports:
            rotated_x, rotated_y = rotate_point(port.pos[0], port.pos[1], width, height, rotation)
            rotated_dir = rotate_direction(port.direction, rotation)
            tiles = port_tiles_from_world_pos(rotated_x, rotated_y)
            ports.append(
                RotatedPortTemplate(
                    pos=(rotated_x, rotated_y),
                    tiles=tiles,
                    direction=rotated_dir,
                    widths=port.widths,
                )
            )
        return ports

    def unique_rotations(self) -> Tuple[Rotation, ...]:
        """Return rotations that produce distinct port layouts for this template."""
        return self._unique_rotations

    def canonical_rotation(self, rotation: Rotation) -> Rotation:
        """Map an arbitrary rotation to its canonical representative for this template."""
        return self._rotation_alias_map[rotation]

    def rotation_variant(self, rotation: Rotation) -> RotatedRoomVariant:
        """Return the precomputed variant for the requested rotation."""
        canonical = self.canonical_rotation(rotation)
        return self._rotation_variants[canonical]

    def rotated_ports(self, rotation: Rotation) -> Tuple[RotatedPortTemplate, ...]:
        """Return template-local ports for the requested rotation."""
        return self.rotation_variant(rotation).ports

@dataclass(frozen=True)
class WorldPort:
    """Port information after converting to world coordinates."""

    pos: Tuple[float, float]
    tiles: Tuple[TilePos, TilePos]
    direction: Direction
    widths: FrozenSet[int]


@dataclass(frozen=True)
class CorridorGeometry:
    """Holds the tile layout for a corridor between two ports."""

    tiles: Tuple[TilePos, ...]
    axis_index: int  # 0 for horizontal, 1 for vertical
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
    index: int = -1


@dataclass
class PlacedRoom:
    """Represents a room instance placed on the dungeon grid."""

    template: RoomTemplate
    x: int
    y: int
    rotation: Rotation
    connected_port_indices: set[int] = field(default_factory=set)
    index: int = -1

    def __post_init__(self) -> None:
        if not isinstance(self.rotation, Rotation):
            try:
                self.rotation = Rotation.from_degrees(int(self.rotation))
            except ValueError as exc:
                raise ValueError(f"Unsupported rotation {self.rotation}") from exc
        elif self.rotation not in VALID_ROTATIONS:
            raise ValueError(f"Unsupported rotation {self.rotation}")

    def get_available_port_indices(self) -> List[int]:
        """Return indices of template ports not yet used for a direct connection."""
        return [i for i in range(len(self.template.ports)) if i not in self.connected_port_indices]

    @property
    def width(self) -> int:
        if self.rotation in (Rotation.DEG_0, Rotation.DEG_180):
            return self.template.size[0]
        return self.template.size[1]

    @property
    def height(self) -> int:
        if self.rotation in (Rotation.DEG_0, Rotation.DEG_180):
            return self.template.size[1]
        return self.template.size[0]

    def get_bounds(self) -> Rect:
        """Return the bounding rectangle covering this room's footprint."""
        return Rect(self.x, self.y, self.width, self.height)

    def get_world_ports(self) -> List[WorldPort]:
        """Calculates the real-world positions and directions of ports after rotation."""
        variant = self.template.rotation_variant(self.rotation)
        x_offset = self.x
        y_offset = self.y
        world_ports: List[WorldPort] = []

        for rotated_port in variant.ports:
            world_x = x_offset + rotated_port.pos[0]
            world_y = y_offset + rotated_port.pos[1]
            tiles = tuple(
                TilePos(tile.x + x_offset, tile.y + y_offset) for tile in rotated_port.tiles
            )
            world_ports.append(
                WorldPort(
                    pos=(world_x, world_y),
                    tiles=tiles, # type: ignore
                    direction=rotated_port.direction,
                    widths=rotated_port.widths,
                )
            )

        return world_ports
