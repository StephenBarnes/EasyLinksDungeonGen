"""Core dungeon generator that orchestrates the algorithmic steps."""

from __future__ import annotations

from typing import Dict, List

from component_manager import ComponentManagerMixin
from corridor_builder import CorridorBuilderMixin
from grid_renderer import GridRendererMixin
from room_placement import RoomPlacementMixin
from dungeon_models import Corridor, PlacedRoom, RoomKind, RoomTemplate


class DungeonGenerator(
    ComponentManagerMixin,
    RoomPlacementMixin,
    CorridorBuilderMixin,
    GridRendererMixin,
):
    """Manages the overall process of generating a dungeon floor layout."""

    def __init__(
        self,
        width: int,
        height: int,
        room_templates: List[RoomTemplate],
        direct_link_counts_probs: Dict[int, float],
        num_rooms_to_place: int,
        min_room_separation: int,
        min_rooms_required: int = 6,
    ) -> None:
        self.width = width
        self.height = height

        self.room_templates = list(room_templates)
        self.standalone_room_templates = [
            rt for rt in room_templates if RoomKind.STANDALONE in rt.kinds
        ]
        self.bend_room_templates = [
            rt for rt in room_templates if RoomKind.BEND in rt.kinds
        ]
        self.t_junction_room_templates = [
            rt for rt in room_templates if RoomKind.T_JUNCTION in rt.kinds
        ]
        self.four_way_room_templates = [
            rt for rt in room_templates if RoomKind.FOUR_WAY in rt.kinds
        ]

        self.num_rooms_to_place = num_rooms_to_place
        self.min_room_separation = min_room_separation
        self.min_rooms_required = min_rooms_required
        self.direct_link_counts_probs = dict(direct_link_counts_probs)

        self.placed_rooms: List[PlacedRoom] = []
        self.room_components: List[int] = []
        self.corridors: List[Corridor] = []
        self.corridor_components: List[int] = []
        self.corridor_tiles = set()
        self.corridor_tile_index: Dict[tuple[int, int], List[int]] = {}
        self.room_corridor_links: set[tuple[int, int]] = set()

        self.grid: List[List[str]] = [[" " for _ in range(width)] for _ in range(height)]
        self._next_component_id = 0

    # The mixin methods provide the public API (place_rooms, create_easy_links, etc.).
    # No extra overrides are required here beyond the state initialisation above.
