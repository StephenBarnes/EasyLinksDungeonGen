"""DungeonGenerator orchestrates the three implemented algorithm steps."""

from __future__ import annotations

import math
import random
import itertools
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Set, Tuple

from dungeon_constants import (
    DOOR_MACRO_ALIGNMENT_OFFSETS,
    MACRO_GRID_SIZE,
    MAX_CONNECTED_PLACEMENT_ATTEMPTS,
    VALID_ROTATIONS,
    MAX_CONSECUTIVE_LIMIT_FAILURES,
)
from dungeon_geometry import rotate_direction
from dungeon_models import RoomKind, Corridor, CorridorGeometry, PlacedRoom, RoomTemplate, WorldPort


@dataclass(frozen=True)
class PortRequirement:
    """Specifies the desired doorway placement for a special junction room."""

    center: Tuple[float, float]
    direction: Tuple[int, int]
    width: int
    inside_tiles: Tuple[Tuple[int, int], ...]
    outside_tiles: Tuple[Tuple[int, int], ...]
    source: str
    geometry: Optional[CorridorGeometry] = None
    room_index: Optional[int] = None
    port_index: Optional[int] = None
    corridor_idx: Optional[int] = None
    corridor_end: Optional[str] = None


class DungeonGenerator:
    """Manages the overall process of generating a dungeon floor layout."""

    def __init__(
        self,
        width: int,
        height: int,
        room_templates: List[RoomTemplate],
        direct_link_counts_probs: dict[int, float],
        num_rooms_to_place: int,
        min_room_separation: int,
        min_rooms_required: int = 6,
    ) -> None:
        self.width = width
        self.height = height

        self.room_templates = list(room_templates)
        self.standalone_room_templates = list(rt for rt in room_templates if RoomKind.STANDALONE in rt.kinds)
        self.bend_room_templates = list(rt for rt in room_templates if RoomKind.BEND in rt.kinds)
        self.t_junction_room_templates = list(rt for rt in room_templates if RoomKind.T_JUNCTION in rt.kinds)
        self.four_way_room_templates = list(rt for rt in room_templates if RoomKind.FOUR_WAY in rt.kinds)

        self.num_rooms_to_place = num_rooms_to_place
        # Minimum empty tiles between room bounding boxes, unless they connect at ports.
        self.min_room_separation = min_room_separation
        self.min_rooms_required = min_rooms_required
        self.placed_rooms: List[PlacedRoom] = []
        self.room_components: List[int] = []
        self.corridors: List[Corridor] = []
        self.corridor_components: List[int] = []
        self.corridor_tiles: Set[Tuple[int, int]] = set()
        self.corridor_tile_index: Dict[Tuple[int, int], List[int]] = {}
        self.room_corridor_links: Set[Tuple[int, int]] = set()
        self.grid = [[" " for _ in range(width)] for _ in range(height)]
        # Probability distribution for number of immediate direct links per room
        # Example: {0: 0.4, 1: 0.3, 2: 0.3}
        self.direct_link_counts_probs = dict(direct_link_counts_probs)
        self._next_component_id = 0

    @staticmethod
    def _expand_bounds(bounds: Tuple[int, int, int, int], margin: int) -> Tuple[int, int, int, int]:
        x, y, w, h = bounds
        return x - margin, y - margin, w + 2 * margin, h + 2 * margin

    @staticmethod
    def _rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        if ax + aw <= bx or bx + bw <= ax:
            return False
        if ay + ah <= by or by + bh <= ay:
            return False
        return True

    def _is_in_bounds(self, room: PlacedRoom) -> bool:
        x, y, w, h = room.get_bounds()
        return 0 <= x and 0 <= y and x + w <= self.width and y + h <= self.height

    def _rooms_overlap(self, candidate: PlacedRoom, existing: PlacedRoom, margin: int) -> bool:
        expanded_candidate = self._expand_bounds(candidate.get_bounds(), margin)
        return self._rects_overlap(expanded_candidate, existing.get_bounds())

    def _is_valid_room_position(self, new_room: PlacedRoom, anchor_room: Optional[PlacedRoom]) -> bool:
        if not self._is_in_bounds(new_room):
            return False

        for room in self.placed_rooms:
            margin = 0 if anchor_room is not None and room is anchor_room else self.min_room_separation
            if self._rooms_overlap(new_room, room, margin):
                return False
        return True

    def _is_valid_placement(self, new_room: PlacedRoom) -> bool:
        """Checks if a new room is in bounds and doesn't overlap existing rooms."""
        return self._is_valid_room_position(new_room, None)

    def _is_valid_placement_with_anchor(self, new_room: PlacedRoom, anchor_room: PlacedRoom) -> bool:
        """Validate placement allowing edge-adjacent contact with the anchor room only."""
        return self._is_valid_room_position(new_room, anchor_room)

    def _clear_grid(self) -> None:
        for row in self.grid:
            for x in range(self.width):
                row[x] = " "

    def _new_component_id(self) -> int:
        component_id = self._next_component_id
        self._next_component_id += 1
        return component_id

    def _register_room(self, room: PlacedRoom, component_id: int) -> None:
        room.component_id = component_id
        self.placed_rooms.append(room)
        self.room_components.append(component_id)

    def _register_corridor(self, corridor: Corridor, component_id: int) -> int:
        corridor.component_id = component_id
        self.corridors.append(corridor)
        self.corridor_components.append(component_id)
        new_index = len(self.corridors) - 1
        self._add_corridor_tiles(new_index)
        return new_index

    def _remove_corridor_tiles(self, corridor_idx: int) -> None:
        corridor = self.corridors[corridor_idx]
        for tile in corridor.geometry.tiles:
            owners = self.corridor_tile_index.get(tile)
            if owners is not None and corridor_idx in owners:
                owners[:] = [idx for idx in owners if idx != corridor_idx]
                if not owners:
                    del self.corridor_tile_index[tile]
            # Only remove from corridor_tiles if no other corridor uses it
            if tile not in self.corridor_tile_index:
                self.corridor_tiles.discard(tile)

    def _add_corridor_tiles(self, corridor_idx: int) -> None:
        corridor = self.corridors[corridor_idx]
        for tile in corridor.geometry.tiles:
            self.corridor_tiles.add(tile)
            self.corridor_tile_index.setdefault(tile, []).append(corridor_idx)

    def _merge_components(self, *component_ids: int) -> int:
        valid_ids = {cid for cid in component_ids if cid >= 0}
        if not valid_ids:
            raise ValueError("Cannot merge empty component set")

        target = min(valid_ids)

        for idx, comp in enumerate(self.room_components):
            if comp in valid_ids:
                self.room_components[idx] = target
                self.placed_rooms[idx].component_id = target

        for idx, comp in enumerate(self.corridor_components):
            if comp in valid_ids:
                self.corridor_components[idx] = target
                self.corridors[idx].component_id = target

        return target

    def get_component_summary(self) -> Dict[int, Dict[str, List[int]]]:
        """Return indices of rooms and corridors grouped by component id."""
        summary: Dict[int, Dict[str, List[int]]] = {}

        def ensure_entry(component_id: int) -> Dict[str, List[int]]:
            return summary.setdefault(
                component_id,
                {
                    "rooms": [],
                    "corridors": [],
                },
            )

        for idx, component_id in enumerate(self.room_components):
            comp_summary = ensure_entry(component_id)
            comp_summary["rooms"].append(idx)

        for idx, component_id in enumerate(self.corridor_components):
            comp_summary = ensure_entry(component_id)
            comp_summary["corridors"].append(idx)

        return summary

    def _random_rotation(self) -> int:
        return random.choice(VALID_ROTATIONS)

    def _random_macro_grid_point(self) -> Tuple[int, int]:
        max_macro_x = (self.width // MACRO_GRID_SIZE) - 1
        max_macro_y = (self.height // MACRO_GRID_SIZE) - 1
        if max_macro_x <= 1 or max_macro_y <= 1:
            raise ValueError("Grid too small to place rooms with macro-grid alignment")

        macro_x = random.randint(1, max_macro_x - 1) * MACRO_GRID_SIZE
        macro_y = random.randint(1, max_macro_y - 1) * MACRO_GRID_SIZE
        return macro_x, macro_y

    def _build_root_room_candidate(
        self, template: RoomTemplate, rotation: int, macro_x: int, macro_y: int
    ) -> PlacedRoom:
        anchor_port_index = random.randrange(len(template.ports))

        rotated_room = PlacedRoom(template, 0, 0, rotation)
        rotated_ports = rotated_room.get_world_ports()
        rotated_anchor_port = rotated_ports[anchor_port_index]

        try:
            offset_x, offset_y = DOOR_MACRO_ALIGNMENT_OFFSETS[rotated_anchor_port.direction]
        except KeyError as exc:
            raise ValueError(f"Unsupported port direction {rotated_anchor_port.direction}") from exc

        snapped_port_x = macro_x + offset_x
        snapped_port_y = macro_y + offset_y

        room_x = int(round(snapped_port_x - rotated_anchor_port.pos[0]))
        room_y = int(round(snapped_port_y - rotated_anchor_port.pos[1]))
        return PlacedRoom(template, room_x, room_y, rotation)

    @staticmethod
    def _categorize_side_distance(distance: float, span: int) -> str:
        if span <= 0:
            return "far"
        ratio = max(0.0, min(distance / float(span), 1.0))
        if ratio <= 0.3:
            return "close"
        if ratio >= 0.4:
            return "far"
        return "intermediate"

    def _describe_macro_position(self, macro_x: int, macro_y: int) -> Tuple[str, Dict[str, str]]:
        side_proximities = {
            "left": self._categorize_side_distance(macro_x, self.width),
            "right": self._categorize_side_distance(self.width - macro_x, self.width),
            "top": self._categorize_side_distance(macro_y, self.height),
            "bottom": self._categorize_side_distance(self.height - macro_y, self.height),
        }

        if any(value == "close" for value in side_proximities.values()):
            proximity = "edge"
        elif all(value == "far" for value in side_proximities.values()):
            proximity = "middle"
        else:
            proximity = "intermediate"

        return proximity, side_proximities

    def _select_root_rotation(
        self,
        template: RoomTemplate,
        placement_category: str,
        side_proximities: Dict[str, str],
    ) -> int:
        preferred_dir = template.preferred_center_facing_dir
        if placement_category != "edge" or preferred_dir is None:
            return self._random_rotation()

        inward_directions: List[Tuple[int, int]] = []
        if side_proximities.get("left") == "close":
            inward_directions.append((1, 0))
        if side_proximities.get("right") == "close":
            inward_directions.append((-1, 0))
        if side_proximities.get("top") == "close":
            inward_directions.append((0, 1))
        if side_proximities.get("bottom") == "close":
            inward_directions.append((0, -1))

        if not inward_directions:
            return self._random_rotation()

        rotation_weights: List[float] = []
        pdx, pdy = preferred_dir
        for rotation in VALID_ROTATIONS:
            rotated_dir = rotate_direction(pdx, pdy, rotation)
            weight = 1.0 if rotated_dir in inward_directions else 0.0
            rotation_weights.append(weight)

        return random.choices(VALID_ROTATIONS, weights=rotation_weights)[0]

    def _sample_num_direct_links(self) -> int:
        """Sample n using the configured probability distribution."""
        items = list(self.direct_link_counts_probs.items())
        total = sum(p for _, p in items)
        if total <= 0:
            # Fallback to default if misconfigured
            items = [(0, 0.4), (1, 0.3), (2, 0.3)]
            total = 1.0
        r = random.random()
        acc = 0.0
        for k, p in items:
            acc += p / total
            if r <= acc:
                return k
        return items[-1][0]

    def _attempt_place_connected_to(self, anchor_room: PlacedRoom) -> Optional[PlacedRoom]:
        """
        Attempt to place a new room that directly connects to one of the available
        ports on anchor_room. Chooses a random template, rotation, and compatible
        port facing opposite the anchor port. Returns the new PlacedRoom on success,
        otherwise None.
        """
        anchor_world_ports = anchor_room.get_world_ports()
        available_anchor_indices = anchor_room.get_available_port_indices()
        random.shuffle(available_anchor_indices)

        # Try each available port in random order
        for anchor_idx in available_anchor_indices:
            awp = anchor_world_ports[anchor_idx]
            ax, ay = awp.pos
            dx, dy = awp.direction
            # Target position for the new room's connecting port so the rooms are adjacent
            target_port_pos = (ax + dx, ay + dy)
            # Candidate attempt loop
            for _ in range(MAX_CONNECTED_PLACEMENT_ATTEMPTS):
                template = random.choices(
                    self.standalone_room_templates, weights=[rt.direct_weight for rt in self.standalone_room_templates]
                )[0]
                rotation = self._random_rotation()
                temp_room = PlacedRoom(template, 0, 0, rotation)
                rot_ports = temp_room.get_world_ports()
                # Find ports facing opposite direction
                compatible_port_indices = [
                    i for i, p in enumerate(rot_ports) if p.direction == (-dx, -dy) and (p.widths & awp.widths)
                ]
                if not compatible_port_indices:
                    continue
                cand_idx = random.choice(compatible_port_indices)
                cand_port = rot_ports[cand_idx]
                # Compute top-left for new room so its chosen port lands at target_port_pos
                rpx, rpy = cand_port.pos
                nx = int(round(target_port_pos[0] - rpx))
                ny = int(round(target_port_pos[1] - rpy))
                candidate = PlacedRoom(template, nx, ny, rotation)
                if self._is_valid_placement_with_anchor(candidate, anchor_room):
                    self._register_room(candidate, anchor_room.component_id)
                    anchor_room.connected_port_indices.add(anchor_idx)
                    candidate.connected_port_indices.add(cand_idx)
                    return candidate
        # No success on any port
        return None

    def _spawn_direct_links_recursive(self, from_room: PlacedRoom) -> int:
        """Recursively try to place 0-2 directly-connected rooms from from_room."""
        rooms_placed = 0
        n = self._sample_num_direct_links()
        for _ in range(n):
            child = self._attempt_place_connected_to(from_room)
            if child is not None:
                rooms_placed += 1
                rooms_placed += self._spawn_direct_links_recursive(child)
        return rooms_placed

    def _build_room_tile_lookup(self) -> Dict[Tuple[int, int], int]:
        """Map each room tile to its owning room index for collision checks."""
        tile_to_room: Dict[Tuple[int, int], int] = {}
        for idx, room in enumerate(self.placed_rooms):
            x, y, w, h = room.get_bounds()
            for ty in range(y, y + h):
                for tx in range(x, x + w):
                    tile_to_room[(tx, ty)] = idx
        return tile_to_room

    def _build_corridor_tile_lookup(self) -> Dict[Tuple[int, int], List[int]]:
        """Map corridor tiles to the corridors occupying them."""
        tile_to_corridors: Dict[Tuple[int, int], List[int]] = {}
        for corridor_idx, corridor in enumerate(self.corridors):
            for tile in corridor.geometry.tiles:
                tile_to_corridors.setdefault(tile, []).append(corridor_idx)
        return tile_to_corridors

    @staticmethod
    def _port_exit_axis_value(port: WorldPort, axis_index: int) -> int:
        """Return the first tile outside the room along the port's facing axis."""
        axis_values = [coord[axis_index] for coord in port.tiles]
        facing = port.direction[axis_index]
        if facing > 0:
            boundary = max(axis_values)
        else:
            boundary = min(axis_values)
        return boundary + facing

    @staticmethod
    def _corridor_cross_coords(center: float, width: int) -> List[int]:
        """Compute the perpendicular tile coordinates for a corridor with given width."""
        if width <= 0:
            raise ValueError("Corridor width must be positive")
        if width % 2 != 0:
            raise ValueError("Corridor widths are expected to be even")
        half = width // 2
        start = int(math.floor(center - (half - 0.5)))
        return list(range(start, start + width))

    def _build_segment_geometry(
        self,
        axis_index: int,
        start_axis: int,
        end_axis: int,
        cross_coords: Tuple[int, ...],
    ) -> Optional[CorridorGeometry]:
        """Construct geometry for a straight corridor segment between two axis values."""
        if start_axis == end_axis:
            return None
        if not cross_coords:
            return None

        step = 1 if end_axis > start_axis else -1
        axis_values = list(range(start_axis, end_axis, step))
        if not axis_values:
            return None

        tiles: List[Tuple[int, int]] = []
        for axis_value in axis_values:
            for cross in cross_coords:
                if axis_index == 0:
                    x, y = axis_value, cross
                else:
                    x, y = cross, axis_value
                if not (0 <= x < self.width and 0 <= y < self.height):
                    return None
                tiles.append((x, y))

        return CorridorGeometry(
            tiles=tuple(tiles),
            axis_index=axis_index,
            port_axis_values=(start_axis, end_axis),
            cross_coords=cross_coords,
        )

    @staticmethod
    def _corridor_cross_from_geometry(
        geometry: CorridorGeometry, axis_index: int
    ) -> Tuple[int, ...]:
        if geometry.cross_coords:
            return geometry.cross_coords
        cross_set: Set[int] = set()
        for tile in geometry.tiles:
            cross_set.add(tile[1 - axis_index])
        if not cross_set:
            raise ValueError("Unable to infer cross coordinates for corridor geometry")
        return tuple(sorted(cross_set))

    def _build_corridor_geometry(
        self,
        room_index_a: int,
        port_a: WorldPort,
        room_index_b: int,
        port_b: WorldPort,
        width: int,
        tile_to_room: Dict[Tuple[int, int], int],
    ) -> Optional[CorridorGeometry]:
        """Return the carved tiles for a straight corridor if it's valid."""
        dx1, dy1 = port_a.direction
        dx2, dy2 = port_b.direction
        if dx1 != -dx2 or dy1 != -dy2:
            return None

        axis_index = 0 if dx1 != 0 else 1
        if axis_index == 0:
            if not math.isclose(port_a.pos[1], port_b.pos[1], abs_tol=1e-6):
                return None
            center = port_a.pos[1]
        else:
            if not math.isclose(port_a.pos[0], port_b.pos[0], abs_tol=1e-6):
                return None
            center = port_a.pos[0]

        exit_a = self._port_exit_axis_value(port_a, axis_index)
        exit_b = self._port_exit_axis_value(port_b, axis_index)
        if exit_a == exit_b:
            return None

        axis_start = min(exit_a, exit_b)
        axis_end = max(exit_a, exit_b)
        cross_coords = self._corridor_cross_coords(center, width)

        tiles: List[Tuple[int, int]] = []
        for axis_value in range(axis_start, axis_end + 1):
            for cross_value in cross_coords:
                if axis_index == 0:
                    x, y = axis_value, cross_value
                else:
                    x, y = cross_value, axis_value
                if not (0 <= x < self.width and 0 <= y < self.height):
                    return None
                if (x, y) in tile_to_room:
                    return None
                tiles.append((x, y))

        allowed_axis_by_room = {
            room_index_a: exit_a,
            room_index_b: exit_b,
        }

        for tile in tiles:
            axis_value = tile[axis_index]
            neighbors = (
                (tile[0] + 1, tile[1]),
                (tile[0] - 1, tile[1]),
                (tile[0], tile[1] + 1),
                (tile[0], tile[1] - 1),
            )
            for neighbor in neighbors:
                neighbor_room = tile_to_room.get(neighbor)
                if neighbor_room is None:
                    continue
                allowed_axis = allowed_axis_by_room.get(neighbor_room)
                if allowed_axis is not None and axis_value == allowed_axis:
                    continue
                return None

        return CorridorGeometry(
            tiles=tuple(tiles),
            axis_index=axis_index,
            port_axis_values=(exit_a, exit_b),
            cross_coords=tuple(cross_coords),
        )

    @staticmethod
    def _port_center_from_tiles(inside_tiles: Tuple[Tuple[int, int], ...]) -> Tuple[float, float]:
        xs = {tile[0] for tile in inside_tiles}
        ys = {tile[1] for tile in inside_tiles}
        if len(xs) == 1:
            x = float(next(iter(xs)))
            min_y = min(ys)
            max_y = max(ys)
            y = (min_y + max_y) / 2.0
            return x, y
        if len(ys) == 1:
            y = float(next(iter(ys)))
            min_x = min(xs)
            max_x = max(xs)
            x = (min_x + max_x) / 2.0
            return x, y
        raise ValueError("Port tiles must form a straight line")

    def _build_port_requirement_from_segment(
        self,
        segment: Optional[CorridorGeometry],
        axis_index: int,
        source: str,
        *,
        expected_width: int,
        room_index: Optional[int] = None,
        port_index: Optional[int] = None,
        corridor_idx: Optional[int] = None,
        corridor_end: Optional[str] = None,
        junction_tiles: Optional[Iterable[Tuple[int, int]]] = None,
    ) -> Optional[PortRequirement]:
        if segment is None:
            return None
        start_axis, end_axis = segment.port_axis_values
        sign = 1 if end_axis > start_axis else -1
        boundary_axis = end_axis - sign
        outside_tiles = tuple(tile for tile in segment.tiles if tile[axis_index] == boundary_axis)
        if not outside_tiles:
            return None
        if axis_index == 0:
            direction = (-sign, 0)
        else:
            direction = (0, -sign)
        junction_set: Optional[Set[Tuple[int, int]]] = None
        if junction_tiles is not None:
            junction_set = set(junction_tiles)

        if junction_set is not None and all(tile in junction_set for tile in outside_tiles):
            inside_tiles = outside_tiles
        else:
            inside_tiles = tuple((tx - direction[0], ty - direction[1]) for tx, ty in outside_tiles)
        width = len(outside_tiles)
        if width != expected_width:
            return None
        center = self._port_center_from_tiles(inside_tiles)
        return PortRequirement(
            center=center,
            direction=direction,
            width=width,
            inside_tiles=tuple(sorted(inside_tiles)),
            outside_tiles=tuple(sorted(outside_tiles)),
            source=source,
            geometry=segment,
            room_index=room_index,
            port_index=port_index,
            corridor_idx=corridor_idx,
            corridor_end=corridor_end,
        )

    def _room_overlaps_disallowed_corridor_tiles(
        self,
        room: PlacedRoom,
        allowed_tiles: Set[Tuple[int, int]],
        allowed_corridors: Set[int],
    ) -> Tuple[bool, Dict[int, Set[Tuple[int, int]]]]:
        rx, ry, rw, rh = room.get_bounds()
        overlaps_by_corridor: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
        for ty in range(ry, ry + rh):
            for tx in range(rx, rx + rw):
                tile = (tx, ty)
                if tile not in self.corridor_tiles:
                    continue
                owners = self.corridor_tile_index.get(tile, [])
                if tile not in allowed_tiles:
                    if not owners or any(owner not in allowed_corridors for owner in owners):
                        return True, {}
                for owner in owners:
                    if owner in allowed_corridors:
                        overlaps_by_corridor[owner].add(tile)
        return False, {corridor: set(tiles) for corridor, tiles in overlaps_by_corridor.items()}

    @staticmethod
    def _world_port_tiles_for_width(port: WorldPort, width: int) -> Tuple[Tuple[int, int], ...]:
        if width <= 0 or width % 2 != 0:
            raise ValueError("Port width must be a positive even number")

        tile_a, tile_b = port.tiles
        if tile_a[0] == tile_b[0]:
            x = tile_a[0]
            y0, y1 = sorted((tile_a[1], tile_b[1]))
            extent = (width // 2) - 1
            start_y = y0 - extent
            end_y = start_y + width - 1
            return tuple((x, y) for y in range(start_y, end_y + 1))

        y = tile_a[1]
        x0, x1 = sorted((tile_a[0], tile_b[0]))
        extent = (width // 2) - 1
        start_x = x0 - extent
        end_x = start_x + width - 1
        return tuple((x, y) for x in range(start_x, end_x + 1))

    def _trim_geometry_for_room(
        self,
        geometry: CorridorGeometry,
        room: PlacedRoom,
    ) -> Optional[CorridorGeometry]:
        axis_index = geometry.axis_index
        if axis_index is None:
            return geometry

        start_axis, end_axis = geometry.port_axis_values
        step = 1 if end_axis > start_axis else -1
        rx, ry, rw, rh = room.get_bounds()

        def tile_inside(tile: Tuple[int, int]) -> bool:
            tx, ty = tile
            return rx <= tx < rx + rw and ry <= ty < ry + rh

        grouped: List[Tuple[int, List[Tuple[int, int]]]] = []
        current_axis: Optional[int] = None
        current_tiles: List[Tuple[int, int]] = []
        for tile in geometry.tiles:
            axis_value = tile[axis_index]
            if current_axis is None or axis_value != current_axis:
                if current_tiles:
                    assert current_axis is not None
                    grouped.append((current_axis, current_tiles))
                current_axis = axis_value
                current_tiles = []
            current_tiles.append(tile)
        if current_tiles:
            assert current_axis is not None
            grouped.append((current_axis, current_tiles))

        if not grouped:
            return None

        start_idx = 0
        end_idx = len(grouped)
        while start_idx < end_idx and any(tile_inside(tile) for tile in grouped[start_idx][1]):
            start_idx += 1
        while end_idx > start_idx and any(tile_inside(tile) for tile in grouped[end_idx - 1][1]):
            end_idx -= 1

        trimmed_groups = grouped[start_idx:end_idx]

        if not trimmed_groups:
            return None
        if start_idx == 0 and end_idx == len(grouped):
            return geometry

        trimmed_tiles = [tile for _, tiles in trimmed_groups for tile in tiles]
        new_start_axis = trimmed_groups[0][0]
        new_end_axis = trimmed_groups[-1][0] + step
        return CorridorGeometry(
            tiles=tuple(trimmed_tiles),
            axis_index=axis_index,
            port_axis_values=(new_start_axis, new_end_axis),
            cross_coords=geometry.cross_coords,
        )

    def _attempt_place_special_room(
        self,
        required_ports: List[PortRequirement],
        templates: List[RoomTemplate],
        allowed_overlap_tiles: Set[Tuple[int, int]],
        allowed_overlap_corridors: Set[int],
    ) -> Optional[Tuple[PlacedRoom, Dict[int, int], Dict[int, CorridorGeometry]]]:
        if not required_ports:
            return None
        template_candidates = list(templates)
        random.shuffle(template_candidates)

        for template in template_candidates:
            for rotation in VALID_ROTATIONS:
                base_room = PlacedRoom(template, 0, 0, rotation)
                rotated_ports = base_room.get_world_ports()
                if len(rotated_ports) < len(required_ports):
                    continue

                port_indices = list(range(len(rotated_ports)))
                for selected_ports in itertools.permutations(port_indices, len(required_ports)):
                    translation: Optional[Tuple[int, int]] = None
                    mapping: Dict[int, int] = {}
                    valid = True
                    for req_idx, port_idx in enumerate(selected_ports):
                        requirement = required_ports[req_idx]
                        rotated_port = rotated_ports[port_idx]
                        if rotated_port.direction != requirement.direction:
                            valid = False
                            break
                        if requirement.width not in rotated_port.widths:
                            valid = False
                            break

                        dx = requirement.center[0] - rotated_port.pos[0]
                        dy = requirement.center[1] - rotated_port.pos[1]
                        if translation is None:
                            if not (
                                math.isclose(dx, round(dx), abs_tol=1e-6)
                                and math.isclose(dy, round(dy), abs_tol=1e-6)
                            ):
                                valid = False
                                break
                            translation = (int(round(dx)), int(round(dy)))
                        else:
                            tx, ty = translation
                            if not (
                                math.isclose(rotated_port.pos[0] + tx, requirement.center[0], abs_tol=1e-6)
                                and math.isclose(rotated_port.pos[1] + ty, requirement.center[1], abs_tol=1e-6)
                            ):
                                valid = False
                                break
                        mapping[req_idx] = port_idx

                    if not valid or translation is None:
                        continue

                    candidate = PlacedRoom(template, translation[0], translation[1], rotation)
                    if not self._is_valid_placement(candidate):
                        continue
                    overlaps_blocked, _ = self._room_overlaps_disallowed_corridor_tiles(
                        candidate,
                        allowed_overlap_tiles,
                        allowed_overlap_corridors,
                    )
                    if overlaps_blocked:
                        continue

                    world_ports = candidate.get_world_ports()
                    ports_match = True
                    for req_idx, port_idx in mapping.items():
                        requirement = required_ports[req_idx]
                        world_port = world_ports[port_idx]
                        candidate_tiles = self._world_port_tiles_for_width(world_port, requirement.width)
                        if candidate_tiles != requirement.inside_tiles:
                            ports_match = False
                            break
                    if not ports_match:
                        continue

                    geometry_overrides: Dict[int, CorridorGeometry] = {}
                    for req_idx, requirement in enumerate(required_ports):
                        geometry = requirement.geometry
                        if geometry is None:
                            continue
                        trimmed = self._trim_geometry_for_room(geometry, candidate)
                        if trimmed is None:
                            ports_match = False
                            break
                        if trimmed is not geometry:
                            geometry_overrides[req_idx] = trimmed
                    if not ports_match:
                        continue

                    return candidate, mapping, geometry_overrides

        return None

    def _validate_room_corridor_clearance(self, room_index: int) -> None:
        room = self.placed_rooms[room_index]
        rx, ry, rw, rh = room.get_bounds()
        overlaps: List[Tuple[Tuple[int, int], List[int]]] = []
        for ty in range(ry, ry + rh):
            for tx in range(rx, rx + rw):
                tile = (tx, ty)
                corridors = self.corridor_tile_index.get(tile)
                if corridors:
                    overlaps.append((tile, list(corridors)))
        if overlaps:
            print(
                f"ERROR: room index {room_index} ({room.template.name}) overlaps corridor tiles:"
            )
            for tile, corridor_indices in overlaps:
                print(f"    tile {tile} -> corridors {corridor_indices}")
            reported: Set[int] = set()
            for _, corridor_indices in overlaps:
                for corridor_idx in corridor_indices:
                    if corridor_idx in reported:
                        continue
                    reported.add(corridor_idx)
                    corridor = self.corridors[corridor_idx]
                    geometry = corridor.geometry
                    print(
                        "    corridor"
                        f" {corridor_idx}: axis_index={geometry.axis_index},"
                        f" port_axis_values={geometry.port_axis_values},"
                        f" cross_coords={geometry.cross_coords},"
                        f" endpoints=({corridor.room_a_index}, {corridor.port_a_index}) ->"
                        f" ({corridor.room_b_index}, {corridor.port_b_index})"
                    )

    def _split_existing_corridor_geometries(
        self, corridor: Corridor, junction_tiles: Iterable[Tuple[int, int]]
    ) -> Tuple[Optional[CorridorGeometry], Optional[CorridorGeometry]]:
        geometry = corridor.geometry
        axis_index = geometry.axis_index
        if axis_index is None:
            return None, None
        axis_values = {tile[axis_index] for tile in junction_tiles}
        if not axis_values:
            return None, None
        cross_coords = self._corridor_cross_from_geometry(geometry, axis_index)
        start_axis, end_axis = geometry.port_axis_values
        axis_min = min(axis_values)
        axis_max = max(axis_values)
        direction = 1 if end_axis > start_axis else -1

        tiles_to_a: List[Tuple[int, int]] = []
        tiles_to_b: List[Tuple[int, int]] = []
        for tile in geometry.tiles:
            axis_value = tile[axis_index]
            if direction > 0:
                if axis_value < axis_min:
                    tiles_to_a.append(tile)
                elif axis_value > axis_max:
                    tiles_to_b.append(tile)
            else:
                if axis_value > axis_max:
                    tiles_to_a.append(tile)
                elif axis_value < axis_min:
                    tiles_to_b.append(tile)

        if not tiles_to_a or not tiles_to_b:
            return None, None

        if direction > 0:
            seg_a_axes = (start_axis, axis_min)
            seg_b_axes = (end_axis, axis_max)
        else:
            seg_a_axes = (start_axis, axis_max)
            seg_b_axes = (end_axis, axis_min)

        seg_to_a = CorridorGeometry(
            tiles=tuple(tiles_to_a),
            axis_index=axis_index,
            port_axis_values=seg_a_axes,
            cross_coords=cross_coords,
        )
        seg_to_b = CorridorGeometry(
            tiles=tuple(tiles_to_b),
            axis_index=axis_index,
            port_axis_values=seg_b_axes,
            cross_coords=cross_coords,
        )
        return seg_to_a, seg_to_b

    def _apply_existing_corridor_segments(
        self,
        corridor_idx: int,
        assignments: Dict[str, Tuple[PortRequirement, int]],
        junction_room_index: int,
        component_id: int,
    ) -> List[int]:
        corridor = self.corridors[corridor_idx]
        connected_indices: List[int] = []

        original_a = (corridor.room_a_index, corridor.port_a_index)
        original_b = (corridor.room_b_index, corridor.port_b_index)

        segments: List[Tuple[str, Corridor]] = []
        for end, original in (("a", original_a), ("b", original_b)):
            assignment = assignments.get(end)
            if assignment is None:
                continue
            requirement, junction_port_idx = assignment
            if requirement.geometry is None:
                continue
            other_room, other_port = original
            segment_corridor = Corridor(
                room_a_index=other_room,
                port_a_index=other_port,
                room_b_index=junction_room_index,
                port_b_index=junction_port_idx,
                width=corridor.width,
                geometry=requirement.geometry,
                component_id=component_id,
            )
            segments.append((end, segment_corridor))

        if not segments:
            return connected_indices

        # Remove the old geometry before replacing it.
        self._remove_corridor_tiles(corridor_idx)

        primary_end, primary_segment = segments[0]
        corridor.room_a_index = primary_segment.room_a_index
        corridor.port_a_index = primary_segment.port_a_index
        corridor.room_b_index = primary_segment.room_b_index
        corridor.port_b_index = primary_segment.port_b_index
        corridor.width = primary_segment.width
        corridor.geometry = primary_segment.geometry
        corridor.component_id = component_id
        self.corridor_components[corridor_idx] = component_id
        self._add_corridor_tiles(corridor_idx)
        connected_indices.append(corridor_idx)

        for _, segment in segments[1:]:
            new_idx = self._register_corridor(segment, component_id)
            connected_indices.append(new_idx)

        return connected_indices

    def _build_t_junction_geometry(
        self,
        room_index: int,
        port: WorldPort,
        width: int,
        tile_to_room: Dict[Tuple[int, int], int],
        tile_to_corridors: Dict[Tuple[int, int], List[int]],
    ) -> Optional[Tuple[CorridorGeometry, int, Tuple[Tuple[int, int], ...]]]:
        """Attempt to carve a straight corridor from a port to an existing corridor."""
        axis_index = 0 if port.direction[0] != 0 else 1
        direction = port.direction[axis_index]
        if direction == 0:
            return None

        cross_center = port.pos[1] if axis_index == 0 else port.pos[0]
        cross_coords = self._corridor_cross_coords(cross_center, width)
        exit_axis_value = self._port_exit_axis_value(port, axis_index)

        axis_value = exit_axis_value
        path_tiles: List[Tuple[int, int]] = []
        max_steps = max(self.width, self.height) + 1
        steps = 0

        while True:
            tiles_for_step: List[Tuple[int, int]] = []
            for cross in cross_coords:
                if axis_index == 0:
                    x, y = axis_value, cross
                else:
                    x, y = cross, axis_value

                if not (0 <= x < self.width and 0 <= y < self.height):
                    return None
                tiles_for_step.append((x, y))

            all_corridor = all(tile in self.corridor_tiles for tile in tiles_for_step)
            if all_corridor:
                intersecting_indices: Optional[Set[int]] = None
                for tile in tiles_for_step:
                    indices = set(tile_to_corridors.get(tile, []))
                    if not indices:
                        intersecting_indices = set()
                        break
                    if intersecting_indices is None:
                        intersecting_indices = indices
                    else:
                        intersecting_indices &= indices
                    if not intersecting_indices:
                        break

                if not intersecting_indices:
                    return None

                chosen_idx: Optional[int] = None
                for idx in sorted(intersecting_indices):
                    candidate = self.corridors[idx]
                    if (
                        candidate.geometry.axis_index is not None
                        and candidate.geometry.axis_index == axis_index
                    ):
                        continue
                    chosen_idx = idx
                    break

                if chosen_idx is None:
                    return None

                existing_geometry = self.corridors[chosen_idx].geometry
                existing_axis_index = existing_geometry.axis_index
                if existing_axis_index is None:
                    return None
                existing_cross_coords = existing_geometry.cross_coords or self._corridor_cross_from_geometry(
                    existing_geometry, existing_axis_index
                )

                intersection_tiles: Set[Tuple[int, int]] = set()
                for new_cross in cross_coords:
                    for existing_cross in existing_cross_coords:
                        if axis_index == 0:
                            # New corridor runs horizontally; existing corridor is vertical.
                            tile = (existing_cross, new_cross)
                        else:
                            # New corridor runs vertically; existing corridor is horizontal.
                            tile = (new_cross, existing_cross)
                        intersection_tiles.add(tile)

                geometry = CorridorGeometry(
                    tiles=tuple(path_tiles),
                    axis_index=axis_index,
                    port_axis_values=(exit_axis_value, axis_value),
                    cross_coords=tuple(cross_coords),
                )
                return geometry, chosen_idx, tuple(sorted(intersection_tiles))

            if any(tile in self.corridor_tiles for tile in tiles_for_step):
                return None

            for tile in tiles_for_step:
                if tile in tile_to_room:
                    return None
                neighbors = (
                    (tile[0] + 1, tile[1]),
                    (tile[0] - 1, tile[1]),
                    (tile[0], tile[1] + 1),
                    (tile[0], tile[1] - 1),
                )
                for neighbor in neighbors:
                    neighbor_room = tile_to_room.get(neighbor)
                    if neighbor_room is None:
                        continue
                    if neighbor_room != room_index:
                        return None

            path_tiles.extend(tiles_for_step)

            axis_value += direction
            steps += 1
            if steps > max_steps:
                return None

    def _list_available_ports(
        self, room_world_ports: List[List[WorldPort]]
    ) -> List[Tuple[int, int, WorldPort]]:
        """Gather all unused ports for the currently placed rooms."""
        available_ports: List[Tuple[int, int, WorldPort]] = []
        for room_index, room in enumerate(self.placed_rooms):
            world_ports = room_world_ports[room_index]
            for port_index in room.get_available_port_indices():
                available_ports.append((room_index, port_index, world_ports[port_index]))
        return available_ports

    def _plan_bend_room(
        self,
        room_a_idx: int,
        port_a_idx: int,
        room_b_idx: int,
        port_b_idx: int,
    ) -> Optional[Tuple[int, PlacedRoom, List[Tuple[int, int, int, CorridorGeometry]]]]:
        """Try to place a bend room linking two perpendicular ports."""
        if not self.bend_room_templates:
            return None

        room_a = self.placed_rooms[room_a_idx]
        room_b = self.placed_rooms[room_b_idx]
        ports_a = room_a.get_world_ports()
        ports_b = room_b.get_world_ports()
        port_a = ports_a[port_a_idx]
        port_b = ports_b[port_b_idx]

        dot = port_a.direction[0] * port_b.direction[0] + port_a.direction[1] * port_b.direction[1]
        if dot != 0:
            return None

        width_options = sorted(port_a.widths & port_b.widths)
        if not width_options:
            return None

        def port_is_horizontal(port: WorldPort) -> bool:
            return port.direction[0] != 0

        def port_is_vertical(port: WorldPort) -> bool:
            return port.direction[1] != 0

        port_infos = [
            {"room_idx": room_a_idx, "port_idx": port_a_idx, "port": port_a},
            {"room_idx": room_b_idx, "port_idx": port_b_idx, "port": port_b},
        ]

        horizontal_info = next((info for info in port_infos if port_is_horizontal(info["port"])), None)
        vertical_info = next((info for info in port_infos if port_is_vertical(info["port"])), None)
        if horizontal_info is None or vertical_info is None:
            return None

        horizontal_dir = horizontal_info["port"].direction
        vertical_dir = vertical_info["port"].direction

        tile_to_room = self._build_room_tile_lookup()
        candidate_room_index = len(self.placed_rooms)

        bend_templates = list(self.bend_room_templates)
        random.shuffle(bend_templates)

        for width in width_options:
            for template in bend_templates:
                for rotation in VALID_ROTATIONS:
                    temp_room = PlacedRoom(template, 0, 0, rotation)
                    rotated_ports = temp_room.get_world_ports()

                    horizontal_candidates = [
                        (idx, rp)
                        for idx, rp in enumerate(rotated_ports)
                        if rp.direction == (-horizontal_dir[0], -horizontal_dir[1])
                    ]
                    vertical_candidates = [
                        (idx, rp)
                        for idx, rp in enumerate(rotated_ports)
                        if rp.direction == (-vertical_dir[0], -vertical_dir[1])
                    ]

                    if not horizontal_candidates or not vertical_candidates:
                        continue

                    for bend_h_idx, bend_h_port in horizontal_candidates:
                        if width not in bend_h_port.widths:
                            continue
                        for bend_v_idx, bend_v_port in vertical_candidates:
                            if bend_v_idx == bend_h_idx:
                                continue
                            if width not in bend_v_port.widths:
                                continue

                            candidate_x = vertical_info["port"].pos[0] - bend_v_port.pos[0]
                            candidate_y = horizontal_info["port"].pos[1] - bend_h_port.pos[1]
                            if not math.isclose(candidate_x, round(candidate_x), abs_tol=1e-6):
                                continue
                            if not math.isclose(candidate_y, round(candidate_y), abs_tol=1e-6):
                                continue

                            placed_bend = PlacedRoom(template, int(round(candidate_x)), int(round(candidate_y)), rotation)
                            if not self._is_valid_placement(placed_bend):
                                continue

                            bx, by, bw, bh = placed_bend.get_bounds()
                            overlaps_corridor = False
                            for ty in range(by, by + bh):
                                for tx in range(bx, bx + bw):
                                    if (tx, ty) in self.corridor_tiles:
                                        overlaps_corridor = True
                                        break
                                if overlaps_corridor:
                                    break
                            if overlaps_corridor:
                                continue

                            tile_map_with_bend = dict(tile_to_room)
                            for ty in range(by, by + bh):
                                for tx in range(bx, bx + bw):
                                    tile_map_with_bend[(tx, ty)] = candidate_room_index

                            bend_world_ports = placed_bend.get_world_ports()
                            bend_world_h = bend_world_ports[bend_h_idx]
                            bend_world_v = bend_world_ports[bend_v_idx]

                            if width not in bend_world_h.widths or width not in bend_world_v.widths:
                                continue

                            geom_h = self._build_corridor_geometry(
                                horizontal_info["room_idx"],
                                horizontal_info["port"],
                                candidate_room_index,
                                bend_world_h,
                                width,
                                tile_map_with_bend,
                            )
                            if geom_h is None:
                                continue

                            geom_v = self._build_corridor_geometry(
                                vertical_info["room_idx"],
                                vertical_info["port"],
                                candidate_room_index,
                                bend_world_v,
                                width,
                                tile_map_with_bend,
                            )
                            if geom_v is None:
                                continue

                            if any(tile in self.corridor_tiles for tile in geom_h.tiles):
                                continue
                            if any(tile in self.corridor_tiles for tile in geom_v.tiles):
                                continue

                            tiles_h = set(geom_h.tiles)
                            tiles_v = set(geom_v.tiles)
                            if tiles_h & tiles_v:
                                continue

                            corridors = [
                                (
                                    horizontal_info["room_idx"],
                                    horizontal_info["port_idx"],
                                    bend_h_idx,
                                    geom_h,
                                ),
                                (
                                    vertical_info["room_idx"],
                                    vertical_info["port_idx"],
                                    bend_v_idx,
                                    geom_v,
                                ),
                            ]
                            return width, placed_bend, corridors

        return None

    def create_easy_links(self, step_num) -> int:
        """Implements Step 2: connect facing ports with straight corridors."""
        if not self.placed_rooms:
            raise ValueError("ERROR: no placed rooms.")

        initial_corridor_count = len(self.corridors)
        tile_to_room = self._build_room_tile_lookup()
        room_world_ports = [room.get_world_ports() for room in self.placed_rooms]
        available_ports = self._list_available_ports(room_world_ports)

        random.shuffle(available_ports)
        used_ports: Set[Tuple[int, int]] = set()
        connected_room_pairs: Set[Tuple[int, int]] = {
            tuple(sorted((corridor.room_a_index, corridor.room_b_index))) # type: ignore
            for corridor in self.corridors
            if corridor.room_b_index is not None
        } # type: ignore
        intersection_rooms_created = 0

        for i, (room_a_idx, port_a_idx, world_port_a) in enumerate(available_ports):
            key_a = (room_a_idx, port_a_idx)
            if key_a in used_ports:
                continue

            candidate_indices = list(range(i + 1, len(available_ports)))
            random.shuffle(candidate_indices)

            for j in candidate_indices:
                room_b_idx, port_b_idx, world_port_b = available_ports[j]
                key_b = (room_b_idx, port_b_idx)
                if key_b in used_ports:
                    continue
                if room_a_idx == room_b_idx:
                    continue

                room_pair: Tuple[int, int] = tuple(sorted((room_a_idx, room_b_idx))) # type: ignore
                if room_pair in connected_room_pairs:
                    continue

                common_widths = world_port_a.widths & world_port_b.widths
                if not common_widths:
                    continue

                viable_options: List[Tuple[int, CorridorGeometry]] = []
                for width in common_widths:
                    geometry = self._build_corridor_geometry(
                        room_a_idx,
                        world_port_a,
                        room_b_idx,
                        world_port_b,
                        width,
                        tile_to_room,
                    )
                    if geometry is not None:
                        viable_options.append((width, geometry))

                if not viable_options:
                    continue

                width, geometry = random.choice(viable_options)

                overlap_map: Dict[int, List[Tuple[int, int]]] = {}
                for tile in geometry.tiles:
                    for existing_idx in self.corridor_tile_index.get(tile, []):
                        overlap_map.setdefault(existing_idx, []).append(tile)

                if overlap_map:
                    if len(overlap_map) != 1:
                        continue

                    existing_idx, overlap_tiles = next(iter(overlap_map.items()))
                    existing_corridor = self.corridors[existing_idx]
                    if existing_corridor.geometry.axis_index is None or geometry.axis_index is None:
                        continue
                    if existing_corridor.geometry.axis_index == geometry.axis_index:
                        continue

                    intersection_axis_new = overlap_tiles[0][geometry.axis_index]
                    cross_coords_new = geometry.cross_coords or self._corridor_cross_from_geometry(
                        geometry, geometry.axis_index
                    )
                    seg_a = self._build_segment_geometry(
                        geometry.axis_index,
                        geometry.port_axis_values[0],
                        intersection_axis_new,
                        cross_coords_new,
                    )
                    seg_b = self._build_segment_geometry(
                        geometry.axis_index,
                        geometry.port_axis_values[1],
                        intersection_axis_new,
                        cross_coords_new,
                    )
                    if seg_a is None or seg_b is None:
                        continue

                    existing_axis_index = existing_corridor.geometry.axis_index
                    if existing_axis_index is None:
                        continue

                    existing_cross_coords = existing_corridor.geometry.cross_coords or self._corridor_cross_from_geometry(
                        existing_corridor.geometry, existing_axis_index
                    )

                    intersection_tiles: Set[Tuple[int, int]] = set()
                    for new_cross in cross_coords_new:
                        for existing_cross in existing_cross_coords:
                            if geometry.axis_index == 0:
                                tile = (existing_cross, new_cross)
                            else:
                                tile = (new_cross, existing_cross)
                            intersection_tiles.add(tile)

                    seg_existing_a, seg_existing_b = self._split_existing_corridor_geometries(
                        existing_corridor,
                        intersection_tiles,
                    )
                    if seg_existing_a is None or seg_existing_b is None:
                        continue

                    requirements: List[PortRequirement] = []
                    requirement_indices: Dict[str, int] = {}

                    def add_requirement(req: Optional[PortRequirement]) -> bool:
                        if req is None:
                            return False
                        requirement_indices[req.source] = len(requirements)
                        requirements.append(req)
                        return True

                    if not add_requirement(
                        self._build_port_requirement_from_segment(
                            seg_a,
                            geometry.axis_index,
                            "new_a",
                            expected_width=width,
                            room_index=room_a_idx,
                            port_index=port_a_idx,
                            junction_tiles=intersection_tiles,
                        )
                    ):
                        continue
                    if not add_requirement(
                        self._build_port_requirement_from_segment(
                            seg_b,
                            geometry.axis_index,
                            "new_b",
                            expected_width=width,
                            room_index=room_b_idx,
                            port_index=port_b_idx,
                            junction_tiles=intersection_tiles,
                        )
                    ):
                        continue
                    if not add_requirement(
                        self._build_port_requirement_from_segment(
                            seg_existing_a,
                            existing_axis_index,
                            "existing_a",
                            expected_width=existing_corridor.width,
                            corridor_idx=existing_idx,
                            corridor_end="a",
                            junction_tiles=intersection_tiles,
                        )
                    ):
                        continue
                    if not add_requirement(
                        self._build_port_requirement_from_segment(
                            seg_existing_b,
                            existing_axis_index,
                            "existing_b",
                            expected_width=existing_corridor.width,
                            corridor_idx=existing_idx,
                            corridor_end="b",
                            junction_tiles=intersection_tiles,
                        )
                    ):
                        continue

                    placement = self._attempt_place_special_room(
                        requirements,
                        self.four_way_room_templates,
                        allowed_overlap_tiles=intersection_tiles,
                        allowed_overlap_corridors={existing_idx},
                    )
                    if placement is None:
                        print("Failed to place four-way junction room. Will print out grid and indicate the intended position of room.")
                        print(requirements)
                        self.draw_to_grid()
                        # Mark the junction.
                        for x, y in intersection_tiles:
                            self.grid[y][x] = "*"
                        self.print_grid()
                        raise RuntimeError("Unable to place a four-way junction room with available templates.")


                    placed_room, port_mapping, geometry_overrides = placement
                    if geometry_overrides:
                        for req_idx, geometry_override in geometry_overrides.items():
                            requirements[req_idx] = replace(
                                requirements[req_idx], geometry=geometry_override
                            )
                    component_id = self._merge_components(
                        self.placed_rooms[room_a_idx].component_id,
                        self.placed_rooms[room_b_idx].component_id,
                        existing_corridor.component_id,
                    )

                    junction_room_index = len(self.placed_rooms)
                    self._register_room(placed_room, component_id)

                    for source in ("new_a", "new_b"):
                        req_idx = requirement_indices[source]
                        requirement = requirements[req_idx]
                        port_idx = port_mapping[req_idx]
                        geometry_segment = requirement.geometry
                        if geometry_segment is None:
                            continue
                        corridor = Corridor(
                            room_a_index=requirement.room_index,
                            port_a_index=requirement.port_index,
                            room_b_index=junction_room_index,
                            port_b_index=port_idx,
                            width=requirement.width,
                            geometry=geometry_segment,
                            component_id=component_id,
                        )
                        self._register_corridor(corridor, component_id)
                        if requirement.room_index is not None and requirement.port_index is not None:
                            self.placed_rooms[requirement.room_index].connected_port_indices.add(requirement.port_index)
                        self.placed_rooms[junction_room_index].connected_port_indices.add(port_idx)

                    existing_assignments: Dict[str, Tuple[PortRequirement, int]] = {}
                    for source in ("existing_a", "existing_b"):
                        req_idx = requirement_indices.get(source)
                        if req_idx is None:
                            continue
                        end_key = "a" if source.endswith("_a") else "b"
                        existing_assignments[end_key] = (requirements[req_idx], port_mapping[req_idx])
                        self.placed_rooms[junction_room_index].connected_port_indices.add(port_mapping[req_idx])

                    self._apply_existing_corridor_segments(
                        existing_idx,
                        existing_assignments,
                        junction_room_index,
                        component_id,
                    )

                    self.placed_rooms[room_a_idx].connected_port_indices.add(port_a_idx)
                    self.placed_rooms[room_b_idx].connected_port_indices.add(port_b_idx)
                    used_ports.add(key_a)
                    used_ports.add(key_b)
                    connected_room_pairs.add(room_pair)
                    intersection_rooms_created += 1

                    break

                else:
                    component_id = self._merge_components(
                        self.placed_rooms[room_a_idx].component_id,
                        self.placed_rooms[room_b_idx].component_id,
                    )

                    corridor = Corridor(
                        room_a_index=room_a_idx,
                        port_a_index=port_a_idx,
                        room_b_index=room_b_idx,
                        port_b_index=port_b_idx,
                        width=width,
                        geometry=geometry,
                        component_id=component_id,
                    )
                    self._register_corridor(corridor, component_id)
                    self.placed_rooms[room_a_idx].connected_port_indices.add(port_a_idx)
                    self.placed_rooms[room_b_idx].connected_port_indices.add(port_b_idx)
                    used_ports.add(key_a)
                    used_ports.add(key_b)
                    connected_room_pairs.add(room_pair)

                    break

        created = len(self.corridors) - initial_corridor_count
        print(
            f"Easylink step {step_num}: created {created} straight corridors "
            f"and placed {intersection_rooms_created} four-way rooms."
        )
        return created

    def create_easy_t_junctions(self, fill_probability: float, step_num: int) -> int:
        """Implements Step 3, 5, and 7: link ports to corridors with straight passages."""
        if not self.corridors:
            print(f"Easylink step {step_num}: skipped - no existing corridors to join.")
            return 0

        tile_to_room = self._build_room_tile_lookup()
        room_world_ports = [room.get_world_ports() for room in self.placed_rooms]

        existing_room_corridor_pairs = set(self.room_corridor_links)

        available_ports = self._list_available_ports(room_world_ports)
        random.shuffle(available_ports)

        created = 0
        junction_rooms_created = 0
        for room_idx, port_idx, world_port in available_ports:
            tile_to_corridors = self._build_corridor_tile_lookup()

            width_options = list(world_port.widths)
            random.shuffle(width_options)

            viable_options: List[Tuple[int, CorridorGeometry, int, Tuple[Tuple[int, int], ...]]] = []
            for width in width_options:
                result = self._build_t_junction_geometry(
                    room_idx,
                    world_port,
                    width,
                    tile_to_room,
                    tile_to_corridors,
                )
                if result is not None:
                    geometry, target_corridor_idx, junction_tiles = result
                    if (room_idx, target_corridor_idx) in existing_room_corridor_pairs:
                        continue
                    viable_options.append((width, geometry, target_corridor_idx, junction_tiles))

            if not viable_options:
                continue

            if random.random() > fill_probability:
                continue

            width, geometry, target_corridor_idx, junction_tiles = random.choice(viable_options)

            target_corridor = self.corridors[target_corridor_idx]
            if geometry.axis_index is None:
                continue
            existing_axis_index = target_corridor.geometry.axis_index
            if existing_axis_index is None:
                continue

            requirements: List[PortRequirement] = []
            requirement_indices: Dict[str, int] = {}

            def add_requirement(req: Optional[PortRequirement]) -> bool:
                if req is None:
                    return False
                requirement_indices[req.source] = len(requirements)
                requirements.append(req)
                return True

            if not add_requirement(
                self._build_port_requirement_from_segment(
                    geometry,
                    geometry.axis_index,
                    "new_branch",
                    expected_width=width,
                    room_index=room_idx,
                    port_index=port_idx,
                    junction_tiles=junction_tiles,
                )
            ):
                continue

            seg_existing_a, seg_existing_b = self._split_existing_corridor_geometries(
                target_corridor,
                junction_tiles,
            )
            if seg_existing_a is None or seg_existing_b is None:
                continue

            if not add_requirement(
                self._build_port_requirement_from_segment(
                    seg_existing_a,
                    existing_axis_index,
                    "existing_a",
                    expected_width=target_corridor.width,
                    corridor_idx=target_corridor_idx,
                    corridor_end="a",
                    junction_tiles=junction_tiles,
                )
            ):
                continue

            if not add_requirement(
                self._build_port_requirement_from_segment(
                    seg_existing_b,
                    existing_axis_index,
                    "existing_b",
                    expected_width=target_corridor.width,
                    corridor_idx=target_corridor_idx,
                    corridor_end="b",
                    junction_tiles=junction_tiles,
                )
            ):
                continue

            placement = self._attempt_place_special_room(
                requirements,
                self.t_junction_room_templates,
                allowed_overlap_tiles=set(junction_tiles),
                allowed_overlap_corridors={target_corridor_idx},
            )
            if placement is None:
                print("Failed to place T-junction room. Will print out grid and indicate the intended position of room.")
                print(requirements)
                self.draw_to_grid()
                # Mark the junction.
                for x, y in junction_tiles:
                    self.grid[y][x] = "*"
                # Mark the interior of the room we're trying to connect.
                self.mark_room_interior_on_grid(room_idx)
                self.print_grid()
                raise RuntimeError("Unable to place a T-junction room with available templates.")

            placed_room, port_mapping, geometry_overrides = placement
            if geometry_overrides:
                for req_idx, geometry_override in geometry_overrides.items():
                    requirements[req_idx] = replace(
                        requirements[req_idx], geometry=geometry_override
                    )
            component_id = self._merge_components(
                self.placed_rooms[room_idx].component_id,
                target_corridor.component_id,
            )

            junction_room_index = len(self.placed_rooms)
            self._register_room(placed_room, component_id)

            branch_idx = requirement_indices["new_branch"]
            branch_requirement = requirements[branch_idx]
            branch_geometry = branch_requirement.geometry
            if branch_geometry is None:
                continue
            branch_port_idx = port_mapping[branch_idx]
            new_corridor = Corridor(
                room_a_index=room_idx,
                port_a_index=port_idx,
                room_b_index=junction_room_index,
                port_b_index=branch_port_idx,
                width=width,
                geometry=branch_geometry,
                component_id=component_id,
            )
            new_corridor_idx = self._register_corridor(new_corridor, component_id)
            self.placed_rooms[room_idx].connected_port_indices.add(port_idx)
            self.placed_rooms[junction_room_index].connected_port_indices.add(branch_port_idx)
            self.room_corridor_links.add((room_idx, new_corridor_idx))
            existing_room_corridor_pairs.add((room_idx, new_corridor_idx))

            existing_assignments: Dict[str, Tuple[PortRequirement, int]] = {}
            for key in ("existing_a", "existing_b"):
                req_idx = requirement_indices.get(key)
                if req_idx is None:
                    continue
                end_key = "a" if key.endswith("_a") else "b"
                existing_assignments[end_key] = (requirements[req_idx], port_mapping[req_idx])
                self.placed_rooms[junction_room_index].connected_port_indices.add(port_mapping[req_idx])

            linked_indices = self._apply_existing_corridor_segments(
                target_corridor_idx,
                existing_assignments,
                junction_room_index,
                component_id,
            )

            existing_room_corridor_pairs.add((room_idx, target_corridor_idx))
            self.room_corridor_links.add((room_idx, target_corridor_idx))
            for idx in linked_indices:
                self.room_corridor_links.add((room_idx, idx))
                existing_room_corridor_pairs.add((room_idx, idx))

            self._validate_room_corridor_clearance(junction_room_index)

            created += 1
            junction_rooms_created += 1

        print(
            f"Easylink step {step_num}: created {created} corridor-to-corridor links "
            f"and placed {junction_rooms_created} T-junction rooms."
        )
        return created

    def create_bent_room_links(self) -> int:
        """Implements Step 4: link different components via 90-degree corridors."""
        if len(self.placed_rooms) < 2:
            print("Easylink step 4: skipped - not enough rooms to connect.")
            return 0

        if not self.bend_room_templates:
            print("Easylink step 4: skipped - no bend room templates available.")
            return 0

        if len({*self.room_components, *self.corridor_components}) <= 1:
            print("Easylink step 4: skipped - already fully connected.")
            return 0

        room_world_ports = [room.get_world_ports() for room in self.placed_rooms]
        available_ports = self._list_available_ports(room_world_ports)
        if len(available_ports) < 2:
            print("Easylink step 4: skipped - not enough unused ports.")
            return 0

        port_records = [
            {
                "room_idx": room_idx,
                "port_idx": port_idx,
                "port": world_port,
            }
            for room_idx, port_idx, world_port in available_ports
        ]

        candidates: List[Tuple[float, int, int, int, int, int]] = []
        for i, port_a_info in enumerate(port_records):
            for port_b_info in port_records[i + 1 :]:
                room_a_idx = port_a_info["room_idx"]
                room_b_idx = port_b_info["room_idx"]
                if self.placed_rooms[room_a_idx].component_id == self.placed_rooms[room_b_idx].component_id:
                    continue

                port_a = port_a_info["port"]
                port_b = port_b_info["port"]
                dot = port_a.direction[0] * port_b.direction[0] + port_a.direction[1] * port_b.direction[1]
                if dot != 0:
                    continue

                common_widths = port_a.widths & port_b.widths
                if not common_widths:
                    continue

                distance = abs(port_a.pos[0] - port_b.pos[0]) + abs(port_a.pos[1] - port_b.pos[1])
                min_width = min(common_widths)
                candidates.append(
                    (
                        float(distance),
                        int(min_width),
                        room_a_idx,
                        port_a_info["port_idx"],
                        room_b_idx,
                        port_b_info["port_idx"],
                    )
                )

        if not candidates:
            print("Easylink step 4: no viable bend room opportunities found.")
            return 0

        candidates.sort(key=lambda item: (item[0], item[1]))

        created = 0
        for _, _min_width, room_a_idx, port_a_idx, room_b_idx, port_b_idx in candidates:
            room_a = self.placed_rooms[room_a_idx]
            room_b = self.placed_rooms[room_b_idx]

            if port_a_idx in room_a.connected_port_indices:
                continue
            if port_b_idx in room_b.connected_port_indices:
                continue
            if room_a.component_id == room_b.component_id:
                continue

            plan = self._plan_bend_room(room_a_idx, port_a_idx, room_b_idx, port_b_idx)
            if plan is None:
                continue

            width, bend_room, corridor_plans = plan
            component_id = self._merge_components(room_a.component_id, room_b.component_id)

            bend_room_index = len(self.placed_rooms)
            self._register_room(bend_room, component_id)

            for existing_room_idx, existing_port_idx, bend_port_idx, geometry in corridor_plans:
                corridor = Corridor(
                    room_a_index=existing_room_idx,
                    port_a_index=existing_port_idx,
                    room_b_index=bend_room_index,
                    port_b_index=bend_port_idx,
                    width=width,
                    geometry=geometry,
                    component_id=component_id,
                )
                self._register_corridor(corridor, component_id)
                self.placed_rooms[existing_room_idx].connected_port_indices.add(existing_port_idx)
                self.placed_rooms[bend_room_index].connected_port_indices.add(bend_port_idx)

            created += 1

            if len({*self.room_components, *self.corridor_components}) <= 1:
                break

        if created == 0:
            print("Easylink step 4: no bend room placements succeeded.")
        else:
            print(f"Easylink step 4: created {created} bend rooms.")
        return created

    def place_rooms(self) -> None:
        """Implements Step 1: Randomly place rooms with macro-grid aligned ports."""
        print(f"Attempting to place {self.num_rooms_to_place} rooms...")
        placed_count = 0
        consecutive_limit_exceeded = 0

        for root_room_index in range(self.num_rooms_to_place):
            if placed_count >= self.num_rooms_to_place:
                break
            if consecutive_limit_exceeded >= MAX_CONSECUTIVE_LIMIT_FAILURES:
                print(f"Exceeded attempt limit {MAX_CONSECUTIVE_LIMIT_FAILURES} consecutive times, aborting further placement.")
                break

            placed_room: Optional[PlacedRoom] = None
            attempt = 0
            for attempt in range(20):
                macro_x, macro_y = self._random_macro_grid_point()
                placement_category, side_proximities = self._describe_macro_position(macro_x, macro_y)

                if placement_category == "middle":
                    template_weights = [rt.root_weight_middle for rt in self.standalone_room_templates]
                elif placement_category == "edge":
                    template_weights = [rt.root_weight_edge for rt in self.standalone_room_templates]
                else:
                    template_weights = [rt.root_weight_intermediate for rt in self.standalone_room_templates]

                if not any(weight > 0 for weight in template_weights):
                    template_weights = [1.0 for _ in self.standalone_room_templates]

                template = random.choices(self.standalone_room_templates, weights=template_weights)[0]
                rotation = self._select_root_rotation(template, placement_category, side_proximities)
                candidate_room = self._build_root_room_candidate(template, rotation, macro_x, macro_y)
                if self._is_valid_placement(candidate_room):
                    placed_room = candidate_room
                    break

            if placed_room is None:
                consecutive_limit_exceeded += 1
                print(f"Exceeded attempt limit when placing root room number {root_room_index}.")
                continue

            component_id = self._new_component_id()
            self._register_room(placed_room, component_id)
            placed_count += 1
            placed_count += self._spawn_direct_links_recursive(placed_room)
            consecutive_limit_exceeded = 0

            #print(f"Placed root room number {root_room_index} after {attempt} failed attempts.")
            #print(f"Placed root room is {placed_room.template.name} at {(placed_room.x, placed_room.y)}")

        print(f"Successfully placed {placed_count} rooms.")

    def draw_to_grid(self, draw_macrogrid: bool = False) -> None:
        """Renders the placed rooms and overlays all door ports."""
        self._clear_grid()
        # Fill rooms with a character so they are easy to distinguish.
        for room in self.placed_rooms:
            room_char = random.choice('OX/LNMW123456789')
            x, y, w, h = room.get_bounds()
            for j in range(h):
                for i in range(w):
                    self.grid[y + j][x + i] = room_char
        # Draw ports
        for room in self.placed_rooms:
            for port in room.get_world_ports():
                for tx, ty in port.tiles:
                    self.grid[ty][tx] = '█'
        # Draw corridors as floor tiles
        for corridor in self.corridors:
            for tx, ty in corridor.geometry.tiles:
                if self.grid[ty][tx] == '░':
                    print(f"Warning: tile {tx, ty} appears to be in multiple corridors")
                elif self.grid[ty][tx] == '█':
                    print(f"Warning: tile {tx, ty} is overlapping a room (on one of the port markers)")
                elif self.grid[ty][tx] != ' ':
                    print(f"Warning: tile {tx, ty} is in a room but also in a corridor")
                self.grid[ty][tx] = '░'
        if draw_macrogrid:
            self._draw_macrogrid_overlay()

    def _draw_macrogrid_overlay(self) -> None:
        """Add 2x2 boxes showing macro-grid squares where door ports can appear."""
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] != ' ':
                    continue
                if (0 < (x % MACRO_GRID_SIZE) < MACRO_GRID_SIZE - 1) or (
                    0 < (y % MACRO_GRID_SIZE) < MACRO_GRID_SIZE - 1
                ):
                    continue
                self.grid[y][x] = '░'

    def mark_room_interior_on_grid(self, room_idx: int) -> None:
        """Mark the interior of the specified room_idx with asterisks."""
        room = self.placed_rooms[room_idx]
        bounds = room.get_bounds()
        for x in range(bounds[0] + 1, bounds[0] + bounds[2] - 1):
            for y in range(bounds[1] + 1, bounds[1] + bounds[3] - 1):
                self.grid[y][x] = "*"

    def print_grid(self, horizontal_sep: str = "") -> None:
        """Prints the ASCII grid to the console."""
        for row in self.grid:
            print(horizontal_sep.join(row))
