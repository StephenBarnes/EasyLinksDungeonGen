"""Data container for dungeon layout state."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from component_manager import ComponentManager
from models import Corridor, PlacedRoom
from spatial_index import SpatialIndex


class DungeonLayout:
    """Stores the mutable state for a dungeon layout."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.placed_rooms: List[PlacedRoom] = []
        self.corridors: List[Corridor] = []
        self.room_corridor_links: Set[Tuple[int, int]] = set()
        self.component_manager = ComponentManager()
        self.spatial_index = SpatialIndex()

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
