"""Context object providing shared state and helper utilities for dungeon grower implementations."""

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from dungeon_config import DungeonConfig
from dungeon_layout import DungeonLayout
from geometry import Direction, TilePos
from growers.port_requirement import PortRequirement
from models import Corridor, CorridorGeometry, PlacedRoom, RoomKind, RoomTemplate, RotatedPortTemplate, WorldPort


@dataclass
class GrowerSeenState:
    """Tracks which layout entities a grower has inspected so far."""

    seen_rooms: Set[int] = field(default_factory=set)
    seen_corridors: Set[int] = field(default_factory=set)
    run_count: int = 0

    def note_seen(self, room_indices: Iterable[int], corridor_indices: Iterable[int]) -> None:
        self.seen_rooms.update(idx for idx in room_indices if idx is not None)
        self.seen_corridors.update(idx for idx in corridor_indices if idx is not None)

    def register_run(self) -> None:
        self.run_count += 1


@dataclass
class GrowerContext:
    """Encapsulates shared state and helpers for grower implementations."""

    config: DungeonConfig
    layout: DungeonLayout
    room_templates: Sequence[RoomTemplate]
    room_templates_by_kind: Mapping[RoomKind, Sequence[RoomTemplate]]
    grower_seen_state: Dict[str, GrowerSeenState] = field(default_factory=dict)

    def get_grower_seen_state(self, grower_name: str) -> GrowerSeenState:
        return self.grower_seen_state.setdefault(grower_name, GrowerSeenState())

    def get_room_templates(self, kind: RoomKind) -> Sequence[RoomTemplate]:
        """Return templates registered for the requested room kind."""
        return self.room_templates_by_kind.get(kind, ())

    def weighted_templates(
        self,
        kind: RoomKind,
        templates: Optional[Sequence[RoomTemplate]] = None,
    ) -> List[RoomTemplate]:
        """Return templates ordered by randomized weights for the given kind."""
        source = templates if templates is not None else self.get_room_templates(kind)
        weighted: List[Tuple[float, RoomTemplate]] = []
        zero_weight: List[RoomTemplate] = []
        for template in source:
            weight = template.weight_for_kind(kind)
            if weight <= 0.0:
                zero_weight.append(template)
                continue
            weighted.append((random.random() ** (1.0 / weight), template))

        if weighted:
            weighted.sort(key=lambda item: item[0], reverse=True)
            ordered = [template for _, template in weighted]
            ordered.extend(zero_weight)
            return ordered

        ordered = list(source)
        random.shuffle(ordered)
        return ordered

    # ------------------------------------------------------------------
    # Port and corridor utilities
    # ------------------------------------------------------------------
    @staticmethod
    def port_exit_axis_value(port: WorldPort | RotatedPortTemplate, axis_index: int) -> int:
        axis_values = [coord[axis_index] for coord in port.tiles]
        facing = port.direction.dx if axis_index == 0 else port.direction.dy
        if facing > 0:
            boundary = max(axis_values)
        else:
            boundary = min(axis_values)
        return boundary + facing

    @staticmethod
    def corridor_cross_coords(center: float, width: int) -> List[int]:
        if width <= 0:
            raise ValueError("Corridor width must be positive")
        if width % 2 != 0:
            raise ValueError("Corridor widths are expected to be even")
        half = width // 2
        start = int(math.floor(center - (half - 0.5)))
        return list(range(start, start + width))

    def build_segment_geometry(
        self,
        axis_index: int,
        start_axis: int,
        end_axis: int,
        cross_coords: Tuple[int, ...],
    ) -> Optional[CorridorGeometry]:
        if start_axis == end_axis or not cross_coords:
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
                if not (0 <= x < self.config.width and 0 <= y < self.config.height):
                    return None
                tiles.append(TilePos(x, y))

        geometry = CorridorGeometry(
            tiles=tuple(tiles),
            axis_index=axis_index,
            port_axis_values=(start_axis, end_axis),
            cross_coords=cross_coords,
        )
        if self.layout.would_create_long_parallel(geometry):
            return None
        return geometry

    @staticmethod
    def corridor_cross_from_geometry(
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

    def build_corridor_geometry(
        self,
        room_index_a: int,
        port_a: WorldPort,
        room_index_b: int,
        port_b: WorldPort,
        width: int,
        extra_room_tiles: Optional[Dict[TilePos, int]] = None,
    ) -> Optional[CorridorGeometry]:
        def room_owner(tile: TilePos) -> Optional[int]:
            if extra_room_tiles and tile in extra_room_tiles:
                return extra_room_tiles[tile]
            return self.layout.spatial_index.get_room_at(tile)

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

        exit_a = self.port_exit_axis_value(port_a, axis_index)
        exit_b = self.port_exit_axis_value(port_b, axis_index)
        if exit_a == exit_b:
            return None

        axis_start = min(exit_a, exit_b)
        axis_end = max(exit_a, exit_b)
        cross_coords = tuple(self.corridor_cross_coords(center, width))

        tiles: List[TilePos] = []
        for axis_value in range(axis_start, axis_end + 1):
            for cross_value in cross_coords:
                if axis_index == 0:
                    x, y = axis_value, cross_value
                else:
                    x, y = cross_value, axis_value
                if not (0 <= x < self.config.width and 0 <= y < self.config.height):
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

        geometry = CorridorGeometry(
            tiles=tuple(tiles),
            axis_index=axis_index,
            port_axis_values=(exit_a, exit_b),
            cross_coords=cross_coords,
        )
        if self.layout.would_create_long_parallel(geometry):
            return None
        return geometry

    @staticmethod
    def port_center_from_tiles(
        direction: Direction, inside_tiles: Tuple[TilePos, ...]
    ) -> Tuple[float, float]:
        if not inside_tiles:
            raise ValueError("At least one tile is required")

        unique_xs = sorted({tile.x for tile in inside_tiles})
        unique_ys = sorted({tile.y for tile in inside_tiles})
        if not unique_xs or not unique_ys:
            raise ValueError("Unable to determine port center from tiles")

        if direction in (Direction.NORTH, Direction.SOUTH):
            # Horizontal doorway: port positions in templates use the first tile's center
            # as the reference, so we mirror that instead of averaging tile centers.
            span = len(unique_xs)
            center_x = unique_xs[0] + (span / 2.0) - 0.5
            center_y = float(min(unique_ys)) if direction is Direction.NORTH else float(max(unique_ys))
        else:
            # Vertical doorway: align with the first tile along Y, matching template math.
            span = len(unique_ys)
            center_y = unique_ys[0] + (span / 2.0) - 0.5
            center_x = float(max(unique_xs)) if direction is Direction.EAST else float(min(unique_xs))

        return float(center_x), float(center_y)

    def build_port_requirement_from_segment(
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
        center = self.port_center_from_tiles(direction, inside_tiles)
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

    def room_overlaps_disallowed_corridor_tiles(
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
                owners = self.layout.spatial_index.get_corridors_at(tile)
                if not owners:
                    continue
                if tile not in allowed_tiles and any(owner not in allowed_corridors for owner in owners):
                    return True, {}
                for owner in owners:
                    if owner in allowed_corridors:
                        overlaps_by_corridor[owner].add(tile)
        return False, {corridor: set(tiles) for corridor, tiles in overlaps_by_corridor.items()}

    @staticmethod
    def world_port_tiles_for_width(port: WorldPort, width: int) -> Tuple[TilePos, ...]:
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

    def trim_geometry_for_room(
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

    def attempt_place_special_room(
        self,
        required_ports: List[PortRequirement],
        templates: Sequence[RoomTemplate],
        kind: RoomKind,
        allowed_overlap_tiles: Set[TilePos],
        allowed_overlap_corridors: Set[int],
    ) -> Optional[Tuple[PlacedRoom, Dict[int, int], Dict[int, CorridorGeometry]]]:
        if not required_ports:
            return None
        template_candidates = self.weighted_templates(kind, templates)

        def axis_index_for(direction: Direction) -> int:
            return 0 if direction.dx != 0 else 1

        def axis_dir_for(direction: Direction) -> int:
            return direction.dx if direction.dx != 0 else direction.dy

        def sort_key(axis_index: int, tile: TilePos) -> Tuple[int, int]:
            return (tile[1 - axis_index], tile[axis_index])

        def shift_tiles(tiles: Tuple[TilePos, ...], delta: int, axis_index: int) -> Tuple[TilePos, ...]:
            if axis_index == 0:
                return tuple(TilePos(tile.x + delta, tile.y) for tile in tiles)
            return tuple(TilePos(tile.x, tile.y + delta) for tile in tiles)

        def align_tiles(
            base_tiles: Tuple[TilePos, ...],
            target_tiles: Tuple[TilePos, ...],
            axis_index: int,
        ) -> Optional[int]:
            if not base_tiles or len(base_tiles) != len(target_tiles):
                return None
            base_sorted = sorted(base_tiles, key=lambda t: sort_key(axis_index, t))
            target_sorted = sorted(target_tiles, key=lambda t: sort_key(axis_index, t))
            deltas: Set[int] = set()
            for base_tile, target_tile in zip(base_sorted, target_sorted):
                if base_tile[1 - axis_index] != target_tile[1 - axis_index]:
                    return None
                deltas.add(target_tile[axis_index] - base_tile[axis_index])
            if len(deltas) != 1:
                return None
            return next(iter(deltas))

        def adjust_requirement(
            requirements_list: List[PortRequirement],
            idx: int,
            candidate_tiles: Tuple[TilePos, ...],
        ) -> Optional[PortRequirement]:
            requirement = requirements_list[idx]
            axis_index = axis_index_for(requirement.direction)
            geometry_tiles: Optional[Set[TilePos]] = None
            if requirement.geometry is not None:
                geometry_tiles = set(requirement.geometry.tiles)

            delta_inside = align_tiles(requirement.inside_tiles, candidate_tiles, axis_index)
            if delta_inside is not None:
                if delta_inside == 0 and candidate_tiles == requirement.inside_tiles:
                    return requirement
                new_inside = tuple(candidate_tiles)
                new_outside = shift_tiles(requirement.outside_tiles, delta_inside, axis_index)
                if geometry_tiles is not None and not all(tile in geometry_tiles for tile in new_outside):
                    return None
                new_center = self.port_center_from_tiles(requirement.direction, new_inside)
                return replace(requirement, center=new_center, inside_tiles=new_inside, outside_tiles=new_outside)

            delta_outside = align_tiles(requirement.outside_tiles, candidate_tiles, axis_index)
            if delta_outside is not None:
                axis_dir = axis_dir_for(requirement.direction)
                new_inside = tuple(candidate_tiles)
                new_outside = shift_tiles(new_inside, axis_dir, axis_index)
                if geometry_tiles is not None and not all(tile in geometry_tiles for tile in new_outside):
                    return None
                new_center = self.port_center_from_tiles(requirement.direction, new_inside)
                return replace(requirement, center=new_center, inside_tiles=new_inside, outside_tiles=new_outside)

            return None

        def compute_translation_candidates(
            requirement: PortRequirement, rotated_port: WorldPort
        ) -> List[Tuple[int, int]]:
            candidates: List[Tuple[int, int]] = []

            def add_candidate(center: Tuple[float, float]) -> None:
                dx = center[0] - rotated_port.pos[0]
                dy = center[1] - rotated_port.pos[1]
                if (
                    math.isclose(dx, round(dx), abs_tol=1e-6)
                    and math.isclose(dy, round(dy), abs_tol=1e-6)
                ):
                    candidate = (int(round(dx)), int(round(dy)))
                    if candidate not in candidates:
                        candidates.append(candidate)

            add_candidate(requirement.center)
            if requirement.outside_tiles:
                outside_center = self.port_center_from_tiles(requirement.direction, requirement.outside_tiles)
                add_candidate(outside_center)

            return candidates

        for template in template_candidates:
            for rotation in template.unique_rotations():
                base_room = PlacedRoom(template, 0, 0, rotation)
                rotated_ports = base_room.get_world_ports()
                if len(rotated_ports) < len(required_ports):
                    continue

                port_indices = list(range(len(rotated_ports)))
                for selected_ports in itertools.permutations(port_indices, len(required_ports)):
                    directions_ok = True
                    for req_idx, port_idx in enumerate(selected_ports):
                        requirement = required_ports[req_idx]
                        rotated_port = rotated_ports[port_idx]
                        if rotated_port.direction != requirement.direction:
                            directions_ok = False
                            break
                        if requirement.width not in rotated_port.widths:
                            directions_ok = False
                            break
                    if not directions_ok:
                        continue

                    first_requirement = required_ports[0]
                    first_port_idx = selected_ports[0]
                    translation_candidates = compute_translation_candidates(
                        first_requirement,
                        rotated_ports[first_port_idx],
                    )
                    if not translation_candidates:
                        continue

                    for translation in translation_candidates:
                        candidate = PlacedRoom(template, translation[0], translation[1], rotation)
                        if not self.layout.is_valid_placement(
                            candidate, ignore_corridors=allowed_overlap_corridors
                        ):
                            continue
                        overlaps_blocked, overlap_tiles = self.room_overlaps_disallowed_corridor_tiles(
                            candidate,
                            allowed_overlap_tiles,
                            allowed_overlap_corridors,
                        )
                        if overlaps_blocked:
                            continue

                        world_ports = candidate.get_world_ports()
                        adjusted_requirements = list(required_ports)
                        mapping: Dict[int, int] = {}
                        ports_match = True
                        for req_idx, port_idx in enumerate(selected_ports):
                            requirement = adjusted_requirements[req_idx]
                            world_port = world_ports[port_idx]
                            candidate_tiles = self.world_port_tiles_for_width(
                                world_port, requirement.width
                            )
                            if candidate_tiles != requirement.inside_tiles:
                                adjusted = adjust_requirement(adjusted_requirements, req_idx, candidate_tiles)
                                if adjusted is None:
                                    ports_match = False
                                    break
                                adjusted_requirements[req_idx] = adjusted
                            mapping[req_idx] = port_idx
                        if not ports_match:
                            continue

                        required_ports[:] = adjusted_requirements
                        requirements = adjusted_requirements

                        geometry_overrides: Dict[int, CorridorGeometry] = {}
                        for req_idx, requirement in enumerate(requirements):
                            geometry = requirement.geometry
                            if geometry is None:
                                continue
                            trimmed = self.trim_geometry_for_room(geometry, candidate)
                            if trimmed is None:
                                ports_match = False
                                break
                            if trimmed is not geometry:
                                geometry_overrides[req_idx] = trimmed
                        if not ports_match:
                            continue

                        return candidate, mapping, geometry_overrides

        return None

    def split_existing_corridor_geometries(
        self,
        corridor: Corridor,
        junction_tiles: Iterable[TilePos],
    ) -> Tuple[Optional[CorridorGeometry], Optional[CorridorGeometry]]:
        geometry = corridor.geometry
        axis_index = geometry.axis_index
        if axis_index is None:
            return None, None
        axis_values = {tile[axis_index] for tile in junction_tiles}
        if not axis_values:
            return None, None
        cross_coords = self.corridor_cross_from_geometry(geometry, axis_index)
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

    def apply_existing_corridor_segments(
        self,
        corridor_idx: int,
        assignments: Dict[str, Tuple[PortRequirement, int]],
        junction_room_index: int,
        component_id: int,
    ) -> List[int]:
        corridor = self.layout.corridors[corridor_idx]
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

        self.layout.spatial_index.remove_corridor(corridor_idx)

        primary_end, primary_segment = segments[0]
        corridor.room_a_index = primary_segment.room_a_index
        corridor.port_a_index = primary_segment.port_a_index
        corridor.room_b_index = primary_segment.room_b_index
        corridor.port_b_index = primary_segment.port_b_index
        corridor.width = primary_segment.width
        corridor.geometry = primary_segment.geometry
        self.layout.set_corridor_component(corridor_idx, component_id)
        self.layout.spatial_index.add_corridor(corridor_idx, corridor.geometry.tiles)
        self.layout.update_corridor_links(corridor_idx)
        connected_indices.append(corridor_idx)

        for _, segment in segments[1:]:
            new_idx = self.layout.register_corridor(segment, component_id)
            connected_indices.append(new_idx)

        return connected_indices

    def validate_room_corridor_clearance(self, room_index: int) -> None:
        room = self.layout.placed_rooms[room_index]
        bounds = room.get_bounds()
        overlaps: List[Tuple[TilePos, List[int]]] = []
        for ty in range(bounds.y, bounds.max_y):
            for tx in range(bounds.x, bounds.max_x):
                tile = TilePos(tx, ty)
                corridors = self.layout.spatial_index.get_corridors_at(tile)
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
                    corridor = self.layout.corridors[corridor_idx]
                    geometry = corridor.geometry
                    print(
                        "    corridor"
                        f" {corridor_idx}: axis_index={geometry.axis_index},"
                        f" port_axis_values={geometry.port_axis_values},"
                        f" cross_coords={geometry.cross_coords},"
                        f" endpoints=({corridor.room_a_index}, {corridor.port_a_index}) ->"
                        f" ({corridor.room_b_index}, {corridor.port_b_index})"
                    )

    def build_t_junction_geometry(
        self,
        room_index: int,
        port: WorldPort,
        width: int,
        *,
        target_corridor_indices: Optional[Set[int]] = None,
        max_axis_distance: Optional[int] = None,
    ) -> Optional[Tuple[CorridorGeometry, int, Tuple[TilePos, ...]]]:
        axis_index = 0 if port.direction.dx != 0 else 1
        direction = port.direction.dx if axis_index == 0 else port.direction.dy
        if direction == 0:
            return None

        def room_owner(tile: TilePos) -> Optional[int]:
            return self.layout.spatial_index.get_room_at(tile)

        cross_center = port.pos[1] if axis_index == 0 else port.pos[0]
        cross_coords = self.corridor_cross_coords(cross_center, width)
        exit_axis_value = self.port_exit_axis_value(port, axis_index)

        axis_value = exit_axis_value
        path_tiles: List[TilePos] = []
        max_steps_default = max(self.config.width, self.config.height) + 1
        max_steps = (
            min(max_axis_distance, max_steps_default)
            if max_axis_distance is not None
            else max_steps_default
        )
        steps = 0

        while True:
            tiles_for_step: List[TilePos] = []
            for cross in cross_coords:
                if axis_index == 0:
                    x, y = axis_value, cross
                else:
                    x, y = cross, axis_value

                if not (0 <= x < self.config.width and 0 <= y < self.config.height):
                    return None
                tiles_for_step.append(TilePos(x, y))

            all_corridor = all(self.layout.spatial_index.has_corridor_at(tile) for tile in tiles_for_step)
            if all_corridor:
                intersecting_indices: Optional[Set[int]] = None
                for tile in tiles_for_step:
                    indices = set(self.layout.spatial_index.get_corridors_at(tile))
                    if not indices:
                        intersecting_indices = set()
                        break
                    if intersecting_indices is None:
                        intersecting_indices = indices
                    else:
                        intersecting_indices &= indices
                    if not intersecting_indices:
                        break

                if target_corridor_indices is not None and intersecting_indices is not None:
                    intersecting_indices &= target_corridor_indices

                if not intersecting_indices:
                    return None

                chosen_idx: Optional[int] = None
                for idx in sorted(intersecting_indices):
                    candidate = self.layout.corridors[idx]
                    geometry = candidate.geometry
                    if geometry.axis_index is not None and geometry.axis_index == axis_index:
                        continue
                    if candidate.room_b_index is None:
                        continue
                    chosen_idx = idx
                    break

                if chosen_idx is None:
                    return None

                existing_geometry = self.layout.corridors[chosen_idx].geometry
                existing_axis_index = existing_geometry.axis_index
                if existing_axis_index is None:
                    return None
                existing_cross_coords = existing_geometry.cross_coords or self.corridor_cross_from_geometry(
                    existing_geometry, existing_axis_index
                )

                intersection_tiles: Set[TilePos] = set()
                for new_cross in cross_coords:
                    for existing_cross in existing_cross_coords:
                        if axis_index == 0:
                            tile = TilePos(existing_cross, new_cross)
                        else:
                            tile = TilePos(new_cross, existing_cross)
                        intersection_tiles.add(tile)

                geometry = CorridorGeometry(
                    tiles=tuple(path_tiles),
                    axis_index=axis_index,
                    port_axis_values=(exit_axis_value, axis_value),
                    cross_coords=tuple(cross_coords),
                )
                if self.layout.would_create_long_parallel(geometry):
                    return None
                return geometry, chosen_idx, tuple(sorted(intersection_tiles))

            if any(self.layout.spatial_index.has_corridor_at(tile) for tile in tiles_for_step):
                return None

            for tile in tiles_for_step:
                owner = room_owner(tile)
                if owner is not None:
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

    def list_available_ports(
        self, room_world_ports: List[List[WorldPort]]
    ) -> List[Tuple[int, int, WorldPort]]:
        available_ports: List[Tuple[int, int, WorldPort]] = []
        for room_index, room in enumerate(self.layout.placed_rooms):
            world_ports = room_world_ports[room_index]
            for port_index in room.get_available_port_indices():
                available_ports.append((room_index, port_index, world_ports[port_index]))
        return available_ports
