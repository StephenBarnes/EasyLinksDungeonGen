"""Mixins and helpers for tracking room and corridor graph components."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from dungeon_models import Corridor, PlacedRoom


class ComponentManagerMixin:
    """Provides component bookkeeping utilities for `DungeonGenerator`.

    The concrete subclass is expected to define the following attributes:

    * `placed_rooms`: list of `PlacedRoom`
    * `room_components`: list parallel to `placed_rooms`
    * `corridors`: list of `Corridor`
    * `corridor_components`: list parallel to `corridors`
    * `_next_component_id`: integer counter
    * `corridor_tiles`: set of tile coordinates used by corridors
    * `corridor_tile_index`: mapping of tile to corridor indices occupying it
    """

    placed_rooms: List[PlacedRoom]
    room_components: List[int]
    corridors: List[Corridor]
    corridor_components: List[int]
    corridor_tiles: Set[Tuple[int, int]]
    corridor_tile_index: Dict[Tuple[int, int], List[int]]
    _next_component_id: int

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
