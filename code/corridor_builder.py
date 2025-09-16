"""Corridor construction and special-room placement helpers."""

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Set, Tuple, TYPE_CHECKING

from dungeon_constants import VALID_ROTATIONS
from dungeon_models import Corridor, CorridorGeometry, PlacedRoom, RoomTemplate, WorldPort
from dungeon_mixin_contract import DungeonContract


@dataclass(frozen=True)
class PortRequirement:
    """Specifies the doorway placement needed for a junction or bend room."""

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


class CorridorBuilderMixin(DungeonContract):
    """Implements corridor carving and the algorithm steps that use it."""

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

        cross_coords = tuple(self._corridor_cross_coords(center, width))

        segment = self._build_segment_geometry(axis_index, exit_a, exit_b, cross_coords)
        if segment is None:
            return None

        tiles = segment.tiles
        allowed_axis_by_room: Dict[int, int] = {}
        for tile in tiles:
            if tile in tile_to_room:
                owner = tile_to_room[tile]
                allowed_axis_by_room.setdefault(owner, tile[axis_index])

        exit_axis_values = segment.port_axis_values
        exit_a, exit_b = exit_axis_values
        for tile in tiles:
            owner = tile_to_room.get(tile)
            if owner is None:
                continue
            if owner not in (room_index_a, room_index_b):
                return None
            axis_value = tile[axis_index]
            if owner == room_index_a and axis_value == exit_a:
                continue
            if owner == room_index_b and axis_value == exit_b:
                continue
            return None

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
            for rotation in (0, 90, 180, 270):
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

                    placed_room = PlacedRoom(
                        template,
                        int(round(base_room.x + translation[0])),
                        int(round(base_room.y + translation[1])),
                        rotation,
                    )

                    overlaps, overlap_tiles_by_corridor = self._room_overlaps_disallowed_corridor_tiles(
                        placed_room,
                        allowed_tiles=allowed_overlap_tiles,
                        allowed_corridors=allowed_overlap_corridors,
                    )
                    if overlaps:
                        continue

                    geometry_overrides: Dict[int, CorridorGeometry] = {}
                    for req_idx, requirement in enumerate(required_ports):
                        rotated_port = placed_room.get_world_ports()[mapping[req_idx]]
                        inside_tiles = self._world_port_tiles_for_width(
                            rotated_port,
                            requirement.width,
                        )
                        outside_tiles = tuple(
                            (tx + rotated_port.direction[0], ty + rotated_port.direction[1])
                            for tx, ty in inside_tiles
                        )
                        geometry = requirement.geometry
                        if geometry is not None:
                            trimmed = self._trim_geometry_for_room(geometry, placed_room)
                            if trimmed is None:
                                break
                            geometry_overrides[req_idx] = trimmed
                    else:
                        return placed_room, mapping, geometry_overrides
        return None

    def _validate_room_corridor_clearance(self, room_index: int) -> None:
        room = self.placed_rooms[room_index]
        rx, ry, rw, rh = room.get_bounds()
        for ty in range(ry, ry + rh):
            for tx in range(rx, rx + rw):
                tile = (tx, ty)
                owners = self.corridor_tile_index.get(tile, [])
                if owners:
                    print(
                        f"Warning: New room at index {room_index} overlaps corridor tiles {tile}."
                        f" Owners={owners}."
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
                            tile = (existing_cross, new_cross)
                        else:
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

    def create_easy_links(self, step_num: int) -> int:
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
            tuple(sorted((corridor.room_a_index, corridor.room_b_index)))
            for corridor in self.corridors
            if corridor.room_b_index is not None
        }
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

                room_pair: Tuple[int, int] = tuple(sorted((room_a_idx, room_b_idx)))
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

                    existing_requirement_a = self._build_port_requirement_from_segment(
                        seg_existing_a,
                        existing_axis_index,
                        "existing_a",
                        expected_width=existing_corridor.width,
                        corridor_idx=existing_idx,
                        corridor_end="a",
                        junction_tiles=intersection_tiles,
                    )
                    existing_requirement_b = self._build_port_requirement_from_segment(
                        seg_existing_b,
                        existing_axis_index,
                        "existing_b",
                        expected_width=existing_corridor.width,
                        corridor_idx=existing_idx,
                        corridor_end="b",
                        junction_tiles=intersection_tiles,
                    )

                    if existing_requirement_a is None or existing_requirement_b is None:
                        continue

                    requirement_indices[existing_requirement_a.source] = len(requirements)
                    requirements.append(existing_requirement_a)
                    requirement_indices[existing_requirement_b.source] = len(requirements)
                    requirements.append(existing_requirement_b)

                    placement = self._attempt_place_special_room(
                        requirements,
                        self.four_way_room_templates,
                        allowed_overlap_tiles=set(intersection_tiles),
                        allowed_overlap_corridors={existing_idx},
                    )
                    if placement is None:
                        continue

                    new_room, port_mapping, geometry_overrides = placement
                    for req_idx, geometry_override in geometry_overrides.items():
                        requirements[req_idx] = replace(requirements[req_idx], geometry=geometry_override)

                    component_id = self._merge_components(
                        self.placed_rooms[room_a_idx].component_id,
                        self.placed_rooms[room_b_idx].component_id,
                        existing_corridor.component_id,
                    )

                    junction_room_index = len(self.placed_rooms)
                    self._register_room(new_room, component_id)

                    assignments: Dict[str, Tuple[PortRequirement, int]] = {}
                    for source in ("new_a", "new_b", "existing_a", "existing_b"):
                        req_idx = requirement_indices.get(source)
                        if req_idx is None:
                            continue
                        end_key = "a" if source.endswith("_a") else "b"
                        assignments[end_key] = (requirements[req_idx], port_mapping[req_idx])
                        self.placed_rooms[junction_room_index].connected_port_indices.add(port_mapping[req_idx])

                    self._apply_existing_corridor_segments(
                        existing_idx,
                        assignments,
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
                print(
                    "Failed to place T-junction room. Will print out grid and indicate the intended position of room."
                )
                print(requirements)
                self.draw_to_grid()
                for x, y in junction_tiles:
                    self.grid[y][x] = "*"
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
