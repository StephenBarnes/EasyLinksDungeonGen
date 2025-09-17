"""Data container for dungeon layout state."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import random

from component_manager import ComponentManager
from dungeon_config import DungeonConfig
from models import Corridor, CorridorGeometry, PlacedRoom

GraphNode = Tuple[str, int]
GraphEntity = Union[PlacedRoom, Corridor, Tuple[str, int]]
from spatial_index import SpatialIndex


class DungeonLayout:
    """Stores the mutable state for a dungeon layout."""

    def __init__(self, config: DungeonConfig) -> None:
        self.config = config
        self.placed_rooms: List[PlacedRoom] = []
        self.corridors: List[Corridor] = []
        self.room_corridor_links: Set[Tuple[int, int]] = set()
        self.room_room_links: Set[Tuple[int, int]] = set()
        self.component_manager = ComponentManager()
        self.spatial_index = SpatialIndex()
        self.grid = [[" " for _ in range(self.config.width)] for _ in range(self.config.height)]
        self._graph_distance_cache: Dict[Tuple[GraphNode, GraphNode], Optional[int]] = {}

    def new_component_id(self) -> int:
        return self.component_manager.new_component()

    def register_room(self, room: PlacedRoom, component_id: int) -> None:
        self.placed_rooms.append(room)
        room_index = len(self.placed_rooms) - 1
        room.index = room_index
        root = self.component_manager.register_room(component_id)
        room.component_id = root
        self.spatial_index.add_room(room_index, room)
        self._invalidate_graph_cache()

    def register_corridor(self, corridor: Corridor, component_id: int) -> int:
        self.corridors.append(corridor)
        new_index = len(self.corridors) - 1
        corridor.index = new_index
        root = self.component_manager.register_corridor(component_id)
        corridor.component_id = root
        self.spatial_index.add_corridor(new_index, corridor.geometry.tiles)
        if corridor.room_a_index is not None:
            self.room_corridor_links.add((corridor.room_a_index, new_index))
        if corridor.room_b_index is not None:
            self.room_corridor_links.add((corridor.room_b_index, new_index))
        self._invalidate_graph_cache()
        return new_index

    def merge_components(self, *component_ids: int) -> int:
        return self.component_manager.union(*component_ids)

    def get_component_summary(self) -> Dict[int, Dict[str, List[int]]]:
        return self.component_manager.component_summary()

    def set_room_component(self, room_idx: int, component_id: int) -> int:
        root = self.component_manager.set_room_component(room_idx, component_id)
        self.placed_rooms[room_idx].component_id = root
        return root

    def set_corridor_component(self, corridor_idx: int, component_id: int) -> int:
        root = self.component_manager.set_corridor_component(corridor_idx, component_id)
        self.corridors[corridor_idx].component_id = root
        return root

    def normalize_room_component(self, room_idx: int) -> int:
        root = self.component_manager.room_component(room_idx)
        self.placed_rooms[room_idx].component_id = root
        return root

    def normalize_corridor_component(self, corridor_idx: int) -> int:
        root = self.component_manager.corridor_component(corridor_idx)
        self.corridors[corridor_idx].component_id = root
        return root

    def rooms_share_component(self, room_a_idx: int, room_b_idx: int) -> bool:
        return (
            self.normalize_room_component(room_a_idx)
            == self.normalize_room_component(room_b_idx)
        )

    def add_room_room_link(self, room_a_idx: int, room_b_idx: int) -> None:
        if room_a_idx == room_b_idx:
            return
        if not (0 <= room_a_idx < len(self.placed_rooms)):
            raise IndexError(f"Room index {room_a_idx} out of range")
        if not (0 <= room_b_idx < len(self.placed_rooms)):
            raise IndexError(f"Room index {room_b_idx} out of range")
        key = tuple(sorted((room_a_idx, room_b_idx)))
        if key not in self.room_room_links:
            self.room_room_links.add(key) # type: ignore
            self._invalidate_graph_cache()

    def remove_corridor_links(self, corridor_idx: int) -> None:
        if corridor_idx < 0 or corridor_idx >= len(self.corridors):
            raise IndexError(f"Corridor index {corridor_idx} out of range")
        if not self.room_corridor_links:
            return
        self.room_corridor_links = {
            (room_idx, idx)
            for room_idx, idx in self.room_corridor_links
            if idx != corridor_idx
        }
        self._invalidate_graph_cache()

    def update_corridor_links(self, corridor_idx: int) -> None:
        if corridor_idx < 0 or corridor_idx >= len(self.corridors):
            raise IndexError(f"Corridor index {corridor_idx} out of range")
        self.remove_corridor_links(corridor_idx)
        corridor = self.corridors[corridor_idx]
        if corridor.room_a_index is not None:
            self.room_corridor_links.add((corridor.room_a_index, corridor_idx))
        if corridor.room_b_index is not None:
            self.room_corridor_links.add((corridor.room_b_index, corridor_idx))
        self._invalidate_graph_cache()

    @staticmethod
    def _geometry_cross_coords(geometry: CorridorGeometry) -> Tuple[int, ...]:
        axis_index = geometry.axis_index
        if axis_index is None:
            return ()
        if geometry.cross_coords:
            return tuple(sorted(set(geometry.cross_coords)))
        cross_values = {tile[1 - axis_index] for tile in geometry.tiles}
        return tuple(sorted(cross_values))

    def would_create_long_parallel(
        self,
        geometry: CorridorGeometry,
        *,
        skip_indices: Iterable[int] = (),
    ) -> bool:
        distance_threshold = self.config.max_parallel_corridor_perpendicular_distance
        overlap_threshold = self.config.max_parallel_corridor_overlap
        if distance_threshold <= 0 or overlap_threshold <= 0:
            return False

        axis_index = geometry.axis_index
        if axis_index is None:
            return False

        cross_coords = self._geometry_cross_coords(geometry)
        if not cross_coords:
            return False

        skip_set = set(skip_indices)
        axis_start, axis_end = geometry.port_axis_values
        axis_min = min(axis_start, axis_end)
        axis_max = max(axis_start, axis_end)

        for idx, existing in enumerate(self.corridors):
            if idx in skip_set:
                continue
            existing_geometry = existing.geometry
            if existing_geometry.axis_index != axis_index:
                continue

            existing_cross = self._geometry_cross_coords(existing_geometry)
            if not existing_cross:
                continue

            existing_start, existing_end = existing_geometry.port_axis_values
            existing_min = min(existing_start, existing_end)
            existing_max = max(existing_start, existing_end)
            axis_overlap = min(axis_max, existing_max) - max(axis_min, existing_min)
            if axis_overlap < overlap_threshold:
                continue

            cross_distance = min(
                abs(candidate - existing_cross_value)
                for candidate in cross_coords
                for existing_cross_value in existing_cross
            )
            if cross_distance < distance_threshold:
                return True

        return False

    def should_allow_connection(
        self,
        entity_a: GraphEntity,
        entity_b: GraphEntity,
        *,
        min_distance: Optional[int] = None,
    ) -> bool:
        threshold = self.config.min_intra_component_connection_distance
        if min_distance is not None:
            threshold = min_distance
        if threshold <= 0:
            return True

        node_a = self._normalize_graph_entity(entity_a)
        node_b = self._normalize_graph_entity(entity_b)
        component_a = self._component_id_for_node(node_a)
        component_b = self._component_id_for_node(node_b)
        if component_a < 0 or component_b < 0:
            return True
        if component_a != component_b:
            return True

        distance = self.graph_distance(node_a, node_b)
        if distance is None:
            return True
        return distance > threshold

    def graph_distance(self, entity_a: GraphEntity, entity_b: GraphEntity) -> Optional[int]:
        node_a = self._normalize_graph_entity(entity_a)
        node_b = self._normalize_graph_entity(entity_b)
        cache_key = self._graph_distance_cache_key(node_a, node_b)
        if cache_key in self._graph_distance_cache:
            return self._graph_distance_cache[cache_key]
        if node_a == node_b:
            self._graph_distance_cache[cache_key] = 0
            return 0

        corridor_to_rooms: Dict[int, Tuple[int, ...]] = {}
        room_to_corridors: Dict[int, List[int]] = {}
        for idx, corridor in enumerate(self.corridors):
            rooms: List[int] = []
            if corridor.room_a_index is not None:
                rooms.append(corridor.room_a_index)
            if corridor.room_b_index is not None:
                rooms.append(corridor.room_b_index)
            corridor_to_rooms[idx] = tuple(rooms)
            for room_idx in rooms:
                room_to_corridors.setdefault(room_idx, []).append(idx)

        room_to_rooms: Dict[int, Set[int]] = {}
        for room_a_idx, room_b_idx in self.room_room_links:
            room_to_rooms.setdefault(room_a_idx, set()).add(room_b_idx)
            room_to_rooms.setdefault(room_b_idx, set()).add(room_a_idx)

        visited: Set[GraphNode] = {node_a}
        queue = deque([(node_a, 0)])

        def neighbors(node: GraphNode) -> Iterable[GraphNode]:
            kind, idx = node
            if kind == "room":
                room_neighbors = room_to_rooms.get(idx, set())
                for other_room in room_neighbors:
                    yield ("room", other_room)
                for corridor_idx in room_to_corridors.get(idx, []):
                    yield ("corridor", corridor_idx)
            elif kind == "corridor":
                for room_idx in corridor_to_rooms.get(idx, ()):  # type: ignore[arg-type]
                    yield ("room", room_idx)
            else:
                raise ValueError(f"Unsupported graph node kind {kind}")

        result: Optional[int] = None
        while queue:
            current, distance = queue.popleft()
            if current == node_b:
                result = distance
                break
            for neighbor in neighbors(current):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

        self._graph_distance_cache[cache_key] = result
        return result

    def _graph_distance_cache_key(self, node_a: GraphNode, node_b: GraphNode) -> Tuple[GraphNode, GraphNode]:
        return (node_a, node_b) if node_a <= node_b else (node_b, node_a)

    def _normalize_graph_entity(self, entity: GraphEntity) -> GraphNode:
        if isinstance(entity, PlacedRoom):
            idx = entity.index
            if idx < 0:
                raise ValueError("PlacedRoom does not have a valid index")
            return ("room", idx)
        if isinstance(entity, Corridor):
            idx = entity.index
            if idx < 0:
                raise ValueError("Corridor does not have a valid index")
            return ("corridor", idx)
        if not isinstance(entity, tuple) or len(entity) != 2:
            raise TypeError("Graph entities must be PlacedRoom, Corridor, or ('kind', index) tuples")
        kind, idx = entity
        if kind not in {"room", "corridor"}:
            raise ValueError(f"Unsupported graph entity kind {kind}")
        if not isinstance(idx, int):
            raise TypeError("Graph entity index must be an integer")
        if kind == "room" and not (0 <= idx < len(self.placed_rooms)):
            raise IndexError(f"Room index {idx} out of range")
        if kind == "corridor" and not (0 <= idx < len(self.corridors)):
            raise IndexError(f"Corridor index {idx} out of range")
        return (kind, idx)

    def _component_id_for_node(self, node: GraphNode) -> int:
        kind, idx = node
        if kind == "room":
            return self.normalize_room_component(idx)
        return self.normalize_corridor_component(idx)

    def _invalidate_graph_cache(self) -> None:
        self._graph_distance_cache.clear()

    def _clear_grid(self) -> None:
        for row in self.grid:
            for x in range(self.config.width):
                row[x] = '.'

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
                elif self.grid[ty][tx] != '.':
                    print(f"Warning: tile {tx, ty} is in a room but also in a corridor")
                self.grid[ty][tx] = '░'
        if draw_macrogrid:
            self._draw_macrogrid_overlay()

    def _draw_macrogrid_overlay(self) -> None:
        """Add 2x2 boxes showing macro-grid squares where door ports can appear."""
        for y in range(self.config.height):
            for x in range(self.config.width):
                if self.grid[y][x] != '.':
                    continue
                if (0 < (x % self.config.macro_grid_size) < self.config.macro_grid_size - 1) or (
                    0 < (y % self.config.macro_grid_size) < self.config.macro_grid_size - 1
                ):
                    continue
                self.grid[y][x] = ';'

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

    def is_in_bounds(self, room: PlacedRoom) -> bool:
        bounds = room.get_bounds()
        return (
            0 <= bounds.x
            and 0 <= bounds.y
            and bounds.max_x <= self.config.width
            and bounds.max_y <= self.config.height
        )

    def is_valid_room_position(
        self,
        new_room: PlacedRoom,
        anchor_room: Optional[PlacedRoom],
        *,
        ignore_corridors: Optional[Set[int]] = None,
    ) -> bool:
        if not self.is_in_bounds(new_room):
            return False

        bounds = new_room.get_bounds()
        corridor_exclusions = set(ignore_corridors or ())
        if not self.spatial_index.is_area_clear(bounds, ignore_corridors=corridor_exclusions):
            return False

        margin = self.config.min_room_separation
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

    def is_valid_placement(
        self,
        new_room: PlacedRoom,
        *,
        ignore_corridors: Optional[Set[int]] = None,
    ) -> bool:
        """Checks if a new room is in bounds and doesn't overlap existing rooms."""
        return self.is_valid_room_position(new_room, None, ignore_corridors=ignore_corridors)

    def is_valid_placement_with_anchor(
        self,
        new_room: PlacedRoom,
        anchor_room: PlacedRoom,
        *,
        ignore_corridors: Optional[Set[int]] = None,
    ) -> bool:
        """Validate placement allowing edge-adjacent contact with the anchor room only."""
        return self.is_valid_room_position(
            new_room,
            anchor_room,
            ignore_corridors=ignore_corridors,
        )
