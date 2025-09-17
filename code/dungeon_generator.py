"""DungeonGenerator orchestrates the three implemented algorithm steps."""

from __future__ import annotations

import math
import random
import itertools
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Set, Tuple

from component_manager import ComponentManager
from dungeon_config import DungeonConfig
from dungeon_geometry import Direction, Rotation, TilePos, rotate_direction, VALID_ROTATIONS
from dungeon_models import RoomKind, Corridor, CorridorGeometry, PlacedRoom, RoomTemplate, WorldPort
from growers import (
    run_bent_room_to_room_grower,
    run_room_to_corridor_grower,
    run_room_to_room_grower,
)
from growers.port_requirement import PortRequirement
from spatial_index import SpatialIndex


class DungeonGenerator:
    """Manages the overall process of generating a dungeon floor layout."""

    def __init__(self, config: DungeonConfig) -> None:
        self.config = config
        self.width = config.width
        self.height = config.height
        self.macro_grid_size = config.macro_grid_size

        self.room_templates = list(config.room_templates)
        self.standalone_room_templates = [rt for rt in self.room_templates if RoomKind.STANDALONE in rt.kinds]
        self.bend_room_templates = [rt for rt in self.room_templates if RoomKind.BEND in rt.kinds]
        self.t_junction_room_templates = [rt for rt in self.room_templates if RoomKind.T_JUNCTION in rt.kinds]
        self.four_way_room_templates = [rt for rt in self.room_templates if RoomKind.FOUR_WAY in rt.kinds]

        self.num_rooms_to_place = config.num_rooms_to_place
        # Minimum empty tiles between room bounding boxes, unless they connect at ports.
        self.min_room_separation = config.min_room_separation
        self.min_rooms_required = config.min_rooms_required
        self.placed_rooms: List[PlacedRoom] = []
        self.corridors: List[Corridor] = []
        self.room_corridor_links: Set[Tuple[int, int]] = set()
        self.grid = [[" " for _ in range(self.width)] for _ in range(self.height)]
        # Probability distribution for number of immediate direct links per room
        # Example: {0: 0.4, 1: 0.3, 2: 0.3}
        self.direct_link_counts_probs = dict(config.direct_link_counts_probs)
        self.component_manager = ComponentManager()
        self.spatial_index = SpatialIndex()
        self._door_macro_alignment_offsets = dict(config.door_macro_alignment_offsets)
        self.max_connected_placement_attempts = config.max_connected_placement_attempts
        self.max_consecutive_limit_failures = config.max_consecutive_limit_failures

    def _is_in_bounds(self, room: PlacedRoom) -> bool:
        bounds = room.get_bounds()
        return (
            0 <= bounds.x
            and 0 <= bounds.y
            and bounds.max_x <= self.width
            and bounds.max_y <= self.height
        )

    def _is_valid_room_position(
        self,
        new_room: PlacedRoom,
        anchor_room: Optional[PlacedRoom],
        *,
        ignore_corridors: Optional[Set[int]] = None,
    ) -> bool:
        if not self._is_in_bounds(new_room):
            return False

        bounds = new_room.get_bounds()
        corridor_exclusions = set(ignore_corridors or ())
        if not self.spatial_index.is_area_clear(bounds, ignore_corridors=corridor_exclusions):
            return False

        margin = self.min_room_separation
        if margin <= 0:
            return True

        expanded_bounds = bounds.expand(margin)
        ignore_rooms: Set[int] = set()
        if anchor_room is not None:
            try:
                anchor_index = self.placed_rooms.index(anchor_room)
            except ValueError:
                anchor_index = None
            if anchor_index is not None:
                ignore_rooms.add(anchor_index)

        return self.spatial_index.is_area_clear(
            expanded_bounds,
            ignore_rooms=ignore_rooms,
            ignore_corridors=corridor_exclusions,
        )

    def _is_valid_placement(
        self,
        new_room: PlacedRoom,
        *,
        ignore_corridors: Optional[Set[int]] = None,
    ) -> bool:
        """Checks if a new room is in bounds and doesn't overlap existing rooms."""
        return self._is_valid_room_position(new_room, None, ignore_corridors=ignore_corridors)

    def _is_valid_placement_with_anchor(
        self,
        new_room: PlacedRoom,
        anchor_room: PlacedRoom,
        *,
        ignore_corridors: Optional[Set[int]] = None,
    ) -> bool:
        """Validate placement allowing edge-adjacent contact with the anchor room only."""
        return self._is_valid_room_position(
            new_room,
            anchor_room,
            ignore_corridors=ignore_corridors,
        )

    def _clear_grid(self) -> None:
        for row in self.grid:
            for x in range(self.width):
                row[x] = " "

    def _new_component_id(self) -> int:
        return self.component_manager.new_component()

    def _register_room(self, room: PlacedRoom, component_id: int) -> None:
        self.placed_rooms.append(room)
        room_index = len(self.placed_rooms) - 1
        room.index = room_index
        root = self.component_manager.register_room(component_id)
        room.component_id = root
        self.spatial_index.add_room(room_index, room)

    def _register_corridor(self, corridor: Corridor, component_id: int) -> int:
        self.corridors.append(corridor)
        new_index = len(self.corridors) - 1
        corridor.index = new_index
        root = self.component_manager.register_corridor(component_id)
        corridor.component_id = root
        self.spatial_index.add_corridor(new_index, corridor.geometry.tiles)
        return new_index

    def _merge_components(self, *component_ids: int) -> int:
        return self.component_manager.union(*component_ids)

    def get_component_summary(self) -> Dict[int, Dict[str, List[int]]]:
        """Return indices of rooms and corridors grouped by component id."""
        return self.component_manager.component_summary()

    def _set_room_component(self, room_idx: int, component_id: int) -> int:
        root = self.component_manager.set_room_component(room_idx, component_id)
        self.placed_rooms[room_idx].component_id = root
        return root

    def _set_corridor_component(self, corridor_idx: int, component_id: int) -> int:
        root = self.component_manager.set_corridor_component(corridor_idx, component_id)
        self.corridors[corridor_idx].component_id = root
        return root

    def _normalize_room_component(self, room_idx: int) -> int:
        root = self.component_manager.room_component(room_idx)
        self.placed_rooms[room_idx].component_id = root
        return root

    def _normalize_corridor_component(self, corridor_idx: int) -> int:
        root = self.component_manager.corridor_component(corridor_idx)
        self.corridors[corridor_idx].component_id = root
        return root

    def _rooms_share_component(self, room_a_idx: int, room_b_idx: int) -> bool:
        return self._normalize_room_component(room_a_idx) == self._normalize_room_component(room_b_idx)

    def _random_macro_grid_point(self) -> Tuple[int, int]:
        max_macro_x = (self.width // self.macro_grid_size) - 1
        max_macro_y = (self.height // self.macro_grid_size) - 1
        if max_macro_x <= 1 or max_macro_y <= 1:
            raise ValueError("Grid too small to place rooms with macro-grid alignment")

        macro_x = random.randint(1, max_macro_x - 1) * self.macro_grid_size
        macro_y = random.randint(1, max_macro_y - 1) * self.macro_grid_size
        return macro_x, macro_y

    def _build_root_room_candidate(
        self, template: RoomTemplate, rotation: Rotation, macro_x: int, macro_y: int
    ) -> PlacedRoom:
        anchor_port_index = random.randrange(len(template.ports))

        rotated_room = PlacedRoom(template, 0, 0, rotation)
        rotated_ports = rotated_room.get_world_ports()
        rotated_anchor_port = rotated_ports[anchor_port_index]

        try:
            offset_x, offset_y = self._door_macro_alignment_offsets[rotated_anchor_port.direction]
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
    ) -> Rotation:
        preferred_dir = template.preferred_center_facing_dir
        if placement_category != "edge" or preferred_dir is None:
            return Rotation.random()

        inward_directions: List[Direction] = []
        if side_proximities.get("left") == "close":
            inward_directions.append(Direction.EAST)
        if side_proximities.get("right") == "close":
            inward_directions.append(Direction.WEST)
        if side_proximities.get("top") == "close":
            inward_directions.append(Direction.SOUTH)
        if side_proximities.get("bottom") == "close":
            inward_directions.append(Direction.NORTH)

        if not inward_directions:
            return Rotation.random()

        rotation_weights: List[float] = []
        for rotation in VALID_ROTATIONS:
            rotated_dir = rotate_direction(preferred_dir, rotation)
            weight = 1.0 if rotated_dir in inward_directions else 0.0
            rotation_weights.append(weight)

        if any(weight > 0 for weight in rotation_weights):
            return random.choices(VALID_ROTATIONS, weights=rotation_weights)[0]
        return Rotation.random()

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
        if anchor_room.index < 0:
            raise ValueError("Anchor room must be registered before creating connections")
        anchor_component_id = self._normalize_room_component(anchor_room.index)
        anchor_world_ports = anchor_room.get_world_ports()
        available_anchor_indices = anchor_room.get_available_port_indices()
        random.shuffle(available_anchor_indices)

        # Try each available port in random order
        for anchor_idx in available_anchor_indices:
            awp = anchor_world_ports[anchor_idx]
            ax, ay = awp.pos
            direction = awp.direction
            dx, dy = direction.dx, direction.dy
            # Target position for the new room's connecting port so the rooms are adjacent
            target_port_pos = (ax + dx, ay + dy)
            # Candidate attempt loop
            for _ in range(self.max_connected_placement_attempts):
                template = random.choices(
                    self.standalone_room_templates, weights=[rt.direct_weight for rt in self.standalone_room_templates]
                )[0]
                rotation = Rotation.random()
                temp_room = PlacedRoom(template, 0, 0, rotation)
                rot_ports = temp_room.get_world_ports()
                # Find ports facing opposite direction
                compatible_port_indices = [
                    i
                    for i, p in enumerate(rot_ports)
                    if p.direction == direction.opposite() and (p.widths & awp.widths)
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
                    self._register_room(candidate, anchor_component_id)
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

    @staticmethod
    def _port_exit_axis_value(port: WorldPort, axis_index: int) -> int:
        """Return the first tile outside the room along the port's facing axis."""
        axis_values = [coord[axis_index] for coord in port.tiles]
        facing = port.direction.dx if axis_index == 0 else port.direction.dy
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

        tiles: List[TilePos] = []
        for axis_value in axis_values:
            for cross in cross_coords:
                if axis_index == 0:
                    x, y = axis_value, cross
                else:
                    x, y = cross, axis_value
                if not (0 <= x < self.width and 0 <= y < self.height):
                    return None
                tiles.append(TilePos(x, y))

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
        extra_room_tiles: Optional[Dict[TilePos, int]] = None,
    ) -> Optional[CorridorGeometry]:
        """Return the carved tiles for a straight corridor if it's valid."""
        def room_owner(tile: TilePos) -> Optional[int]:
            if extra_room_tiles and tile in extra_room_tiles:
                return extra_room_tiles[tile]
            return self.spatial_index.get_room_at(tile)

        dx1, dy1 = port_a.direction.dx, port_a.direction.dy
        dx2, dy2 = port_b.direction.dx, port_b.direction.dy
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

        tiles: List[TilePos] = []
        for axis_value in range(axis_start, axis_end + 1):
            for cross_value in cross_coords:
                if axis_index == 0:
                    x, y = axis_value, cross_value
                else:
                    x, y = cross_value, axis_value
                if not (0 <= x < self.width and 0 <= y < self.height):
                    return None
                tile = TilePos(x, y)
                if room_owner(tile) is not None:
                    return None
                tiles.append(tile)

        allowed_axis_by_room = {
            room_index_a: exit_a,
            room_index_b: exit_b,
        }

        for tile in tiles:
            axis_value = tile[axis_index]
            neighbors = (
                TilePos(tile.x + 1, tile.y),
                TilePos(tile.x - 1, tile.y),
                TilePos(tile.x, tile.y + 1),
                TilePos(tile.x, tile.y - 1),
            )
            for neighbor in neighbors:
                neighbor_room = room_owner(neighbor)
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
    def _port_center_from_tiles(inside_tiles: Tuple[TilePos, ...]) -> Tuple[float, float]:
        xs = {tile.x for tile in inside_tiles}
        ys = {tile.y for tile in inside_tiles}
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
        junction_tiles: Optional[Iterable[TilePos]] = None,
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
            direction = Direction.WEST if sign > 0 else Direction.EAST
        else:
            direction = Direction.NORTH if sign > 0 else Direction.SOUTH
        junction_set: Optional[Set[TilePos]] = None
        if junction_tiles is not None:
            junction_set = set(junction_tiles)

        if junction_set is not None and all(tile in junction_set for tile in outside_tiles):
            inside_tiles = outside_tiles
        else:
            dx, dy = direction.dx, direction.dy
            inside_tiles = tuple(TilePos(tile.x - dx, tile.y - dy) for tile in outside_tiles)
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
        allowed_tiles: Set[TilePos],
        allowed_corridors: Set[int],
    ) -> Tuple[bool, Dict[int, Set[TilePos]]]:
        bounds = room.get_bounds()
        overlaps_by_corridor: Dict[int, Set[TilePos]] = defaultdict(set)
        for ty in range(bounds.y, bounds.max_y):
            for tx in range(bounds.x, bounds.max_x):
                tile = TilePos(tx, ty)
                owners = self.spatial_index.get_corridors_at(tile)
                if not owners:
                    continue
                if tile not in allowed_tiles and any(owner not in allowed_corridors for owner in owners):
                    return True, {}
                for owner in owners:
                    if owner in allowed_corridors:
                        overlaps_by_corridor[owner].add(tile)
        return False, {corridor: set(tiles) for corridor, tiles in overlaps_by_corridor.items()}

    @staticmethod
    def _world_port_tiles_for_width(port: WorldPort, width: int) -> Tuple[TilePos, ...]:
        if width <= 0 or width % 2 != 0:
            raise ValueError("Port width must be a positive even number")

        tile_a, tile_b = port.tiles
        if tile_a.x == tile_b.x:
            x = tile_a.x
            y0, y1 = sorted((tile_a.y, tile_b.y))
            extent = (width // 2) - 1
            start_y = y0 - extent
            end_y = start_y + width - 1
            return tuple(TilePos(x, y) for y in range(start_y, end_y + 1))

        y = tile_a.y
        x0, x1 = sorted((tile_a.x, tile_b.x))
        extent = (width // 2) - 1
        start_x = x0 - extent
        end_x = start_x + width - 1
        return tuple(TilePos(x, y) for x in range(start_x, end_x + 1))

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
        bounds = room.get_bounds()

        def tile_inside(tile: TilePos) -> bool:
            return bounds.contains(tile)

        grouped: List[Tuple[int, List[TilePos]]] = []
        current_axis: Optional[int] = None
        current_tiles: List[TilePos] = []
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
        allowed_overlap_tiles: Set[TilePos],
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
                    if not self._is_valid_placement(
                        candidate, ignore_corridors=allowed_overlap_corridors
                    ):
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
        bounds = room.get_bounds()
        overlaps: List[Tuple[TilePos, List[int]]] = []
        for ty in range(bounds.y, bounds.max_y):
            for tx in range(bounds.x, bounds.max_x):
                tile = TilePos(tx, ty)
                corridors = self.spatial_index.get_corridors_at(tile)
                if corridors:
                    overlaps.append((tile, list(corridors)))
        if overlaps:
            print(
                f"ERROR: room index {room_index} ({room.template.name}) overlaps corridor tiles:"
            )
            for tile, corridor_indices in overlaps:
                print(f"    tile {tile.to_tuple()} -> corridors {corridor_indices}")
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
        self, corridor: Corridor, junction_tiles: Iterable[TilePos]
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

        tiles_to_a: List[TilePos] = []
        tiles_to_b: List[TilePos] = []
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
        self.spatial_index.remove_corridor(corridor_idx)

        primary_end, primary_segment = segments[0]
        corridor.room_a_index = primary_segment.room_a_index
        corridor.port_a_index = primary_segment.port_a_index
        corridor.room_b_index = primary_segment.room_b_index
        corridor.port_b_index = primary_segment.port_b_index
        corridor.width = primary_segment.width
        corridor.geometry = primary_segment.geometry
        self._set_corridor_component(corridor_idx, component_id)
        self.spatial_index.add_corridor(corridor_idx, corridor.geometry.tiles)
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
    ) -> Optional[Tuple[CorridorGeometry, int, Tuple[TilePos, ...]]]:
        """Attempt to carve a straight corridor from a port to an existing corridor."""
        axis_index = 0 if port.direction.dx != 0 else 1
        direction = port.direction.dx if axis_index == 0 else port.direction.dy
        if direction == 0:
            return None

        def room_owner(tile: TilePos) -> Optional[int]:
            return self.spatial_index.get_room_at(tile)

        cross_center = port.pos[1] if axis_index == 0 else port.pos[0]
        cross_coords = self._corridor_cross_coords(cross_center, width)
        exit_axis_value = self._port_exit_axis_value(port, axis_index)

        axis_value = exit_axis_value
        path_tiles: List[TilePos] = []
        max_steps = max(self.width, self.height) + 1
        steps = 0

        while True:
            tiles_for_step: List[TilePos] = []
            for cross in cross_coords:
                if axis_index == 0:
                    x, y = axis_value, cross
                else:
                    x, y = cross, axis_value

                if not (0 <= x < self.width and 0 <= y < self.height):
                    return None
                tiles_for_step.append(TilePos(x, y))

            all_corridor = all(self.spatial_index.has_corridor_at(tile) for tile in tiles_for_step)
            if all_corridor:
                intersecting_indices: Optional[Set[int]] = None
                for tile in tiles_for_step:
                    indices = set(self.spatial_index.get_corridors_at(tile))
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

                intersection_tiles: Set[TilePos] = set()
                for new_cross in cross_coords:
                    for existing_cross in existing_cross_coords:
                        if axis_index == 0:
                            # New corridor runs horizontally; existing corridor is vertical.
                            tile = TilePos(existing_cross, new_cross)
                        else:
                            # New corridor runs vertically; existing corridor is horizontal.
                            tile = TilePos(new_cross, existing_cross)
                        intersection_tiles.add(tile)

                geometry = CorridorGeometry(
                    tiles=tuple(path_tiles),
                    axis_index=axis_index,
                    port_axis_values=(exit_axis_value, axis_value),
                    cross_coords=tuple(cross_coords),
                )
                return geometry, chosen_idx, tuple(sorted(intersection_tiles))

            if any(self.spatial_index.has_corridor_at(tile) for tile in tiles_for_step):
                return None

            for tile in tiles_for_step:
                if room_owner(tile) is not None:
                    return None
                neighbors = (
                    TilePos(tile.x + 1, tile.y),
                    TilePos(tile.x - 1, tile.y),
                    TilePos(tile.x, tile.y + 1),
                    TilePos(tile.x, tile.y - 1),
                )
                for neighbor in neighbors:
                    neighbor_room = room_owner(neighbor)
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

        if port_a.direction.dot(port_b.direction) != 0:
            return None

        width_options = sorted(port_a.widths & port_b.widths)
        if not width_options:
            return None

        def port_is_horizontal(port: WorldPort) -> bool:
            return port.direction.dx != 0

        def port_is_vertical(port: WorldPort) -> bool:
            return port.direction.dy != 0

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
                        if rp.direction == horizontal_dir.opposite()
                    ]
                    vertical_candidates = [
                        (idx, rp)
                        for idx, rp in enumerate(rotated_ports)
                        if rp.direction == vertical_dir.opposite()
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

                            bend_bounds = placed_bend.get_bounds()
                            overlaps_corridor = False
                            extra_room_tiles: Dict[TilePos, int] = {}
                            for ty in range(bend_bounds.y, bend_bounds.max_y):
                                for tx in range(bend_bounds.x, bend_bounds.max_x):
                                    tile = TilePos(tx, ty)
                                    if self.spatial_index.has_corridor_at(tile):
                                        overlaps_corridor = True
                                        break
                                    extra_room_tiles[tile] = candidate_room_index
                                if overlaps_corridor:
                                    break
                            if overlaps_corridor:
                                continue

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
                                extra_room_tiles,
                            )
                            if geom_h is None:
                                continue

                            geom_v = self._build_corridor_geometry(
                                vertical_info["room_idx"],
                                vertical_info["port"],
                                candidate_room_index,
                                bend_world_v,
                                width,
                                extra_room_tiles,
                            )
                            if geom_v is None:
                                continue

                            if any(self.spatial_index.has_corridor_at(tile) for tile in geom_h.tiles):
                                continue
                            if any(self.spatial_index.has_corridor_at(tile) for tile in geom_v.tiles):
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

    def grower_room_to_room(self) -> int:
        """Connect facing ports with straight corridors or four-way junctions."""
        if not self.placed_rooms:
            raise ValueError("ERROR: no placed rooms.")
        return run_room_to_room_grower(self)


    def grower_room_to_corridor(self, fill_probability: float) -> int:
        """Link room ports to nearby corridors via T-junction rooms."""
        if not self.corridors:
            print("Room-to-corridor grower: skipped - no existing corridors to join.")
            return 0
        return run_room_to_corridor_grower(self, fill_probability)


    def grower_bent_room_to_room(self) -> int:
        """Link rooms with perpendicular ports using bend rooms."""
        if len(self.placed_rooms) < 2:
            print("Bent-room-to-room grower: skipped - not enough rooms to connect.")
            return 0
        if not self.bend_room_templates:
            print("Bent-room-to-room grower: skipped - no bend room templates available.")
            return 0
        if self.component_manager.has_single_component():
            print("Bent-room-to-room grower: skipped - already fully connected.")
            return 0
        return run_bent_room_to_room_grower(self)


    def place_rooms(self) -> None:
        """Implements step 1: Randomly place rooms with macro-grid aligned ports."""
        print(f"Attempting to place {self.num_rooms_to_place} rooms...")
        placed_count = 0
        consecutive_limit_exceeded = 0

        for root_room_index in range(self.num_rooms_to_place):
            if placed_count >= self.num_rooms_to_place:
                break
            if consecutive_limit_exceeded >= self.max_consecutive_limit_failures:
                print(
                    f"Exceeded attempt limit {self.max_consecutive_limit_failures} consecutive times, aborting further placement."
                )
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
            bounds = room.get_bounds()
            for ty in range(bounds.y, bounds.max_y):
                for tx in range(bounds.x, bounds.max_x):
                    self.grid[ty][tx] = room_char
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
                if (0 < (x % self.macro_grid_size) < self.macro_grid_size - 1) or (
                    0 < (y % self.macro_grid_size) < self.macro_grid_size - 1
                ):
                    continue
                self.grid[y][x] = '.'

    def mark_room_interior_on_grid(self, room_idx: int) -> None:
        """Mark the interior of the specified room_idx with asterisks."""
        room = self.placed_rooms[room_idx]
        bounds = room.get_bounds()
        for x in range(bounds.x + 1, bounds.max_x - 1):
            for y in range(bounds.y + 1, bounds.max_y - 1):
                self.grid[y][x] = "*"

    def print_grid(self, horizontal_sep: str = "") -> None:
        """Prints the ASCII grid to the console."""
        for row in self.grid:
            print(horizontal_sep.join(row))
