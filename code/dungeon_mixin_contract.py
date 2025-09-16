from __future__ import annotations

from typing import Dict, List, Protocol, Set, Tuple

from dungeon_models import Corridor, PlacedRoom, RoomTemplate


class DungeonContract(Protocol):
    """
    Defines the contract that the final DungeonGenerator class must fulfill.
    Mixins can rely on these methods and attributes being present.
    """

    # --- Attributes from DungeonGenerator ---
    width: int
    height: int
    room_templates: List[RoomTemplate]
    standalone_room_templates: List[RoomTemplate]
    bend_room_templates: List[RoomTemplate]
    t_junction_room_templates: List[RoomTemplate]
    four_way_room_templates: List[RoomTemplate]
    num_rooms_to_place: int
    min_room_separation: int
    direct_link_counts_probs: Dict[int, float]

    placed_rooms: List[PlacedRoom]
    room_components: List[int]
    corridors: List[Corridor]
    corridor_components: List[int]
    corridor_tiles: Set[Tuple[int, int]]
    corridor_tile_index: Dict[Tuple[int, int], List[int]]
    room_corridor_links: Set[Tuple[int, int]]

    grid: List[List[str]]
    _next_component_id: int

    # --- Methods from ComponentManagerMixin ---
    def _new_component_id(self) -> int:
        ...

    def _register_room(self, room: PlacedRoom, component_id: int) -> None:
        ...

    def _register_corridor(self, corridor: Corridor, component_id: int) -> int:
        ...

    def _remove_corridor_tiles(self, corridor_idx: int) -> None:
        ...

    def _add_corridor_tiles(self, corridor_idx: int) -> None:
        ...

    def _merge_components(self, *component_ids: int) -> int:
        ...

    # --- Method from RoomPlacementMixin ---
    def _is_valid_placement(self, new_room: PlacedRoom) -> bool:
        ...

    def mark_room_interior_on_grid(self, room_idx: int) -> None:
        ...

    def print_grid(self, horizontal_sep: str = "") -> None:
        ...