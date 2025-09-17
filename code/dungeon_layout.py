"""Data container for dungeon layout state."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import random

from component_manager import ComponentManager
from dungeon_config import DungeonConfig
from models import Corridor, PlacedRoom
from spatial_index import SpatialIndex


class DungeonLayout:
    """Stores the mutable state for a dungeon layout."""

    def __init__(self, config: DungeonConfig) -> None:
        self.config = config
        self.placed_rooms: List[PlacedRoom] = []
        self.corridors: List[Corridor] = []
        self.room_corridor_links: Set[Tuple[int, int]] = set()
        self.component_manager = ComponentManager()
        self.spatial_index = SpatialIndex()
        self.grid = [[" " for _ in range(self.config.width)] for _ in range(self.config.height)]

    def new_component_id(self) -> int:
        return self.component_manager.new_component()

    def register_room(self, room: PlacedRoom, component_id: int) -> None:
        self.placed_rooms.append(room)
        room_index = len(self.placed_rooms) - 1
        room.index = room_index
        root = self.component_manager.register_room(component_id)
        room.component_id = root
        self.spatial_index.add_room(room_index, room)

    def register_corridor(self, corridor: Corridor, component_id: int) -> int:
        self.corridors.append(corridor)
        new_index = len(self.corridors) - 1
        corridor.index = new_index
        root = self.component_manager.register_corridor(component_id)
        corridor.component_id = root
        self.spatial_index.add_corridor(new_index, corridor.geometry.tiles)
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

    def _clear_grid(self) -> None:
        for row in self.grid:
            for x in range(self.config.width):
                row[x] = " "

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
        for y in range(self.config.height):
            for x in range(self.config.width):
                if self.grid[y][x] != ' ':
                    continue
                if (0 < (x % self.config.macro_grid_size) < self.config.macro_grid_size - 1) or (
                    0 < (y % self.config.macro_grid_size) < self.config.macro_grid_size - 1
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