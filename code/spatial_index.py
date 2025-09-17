"""Spatial index for tracking tile occupancy by rooms and corridors."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Dict, Optional, Set, Tuple

from dungeon_geometry import Rect, TilePos
from dungeon_models import PlacedRoom


class SpatialIndex:
    """Caches tile occupancy to accelerate spatial lookups."""

    def __init__(self) -> None:
        self._tile_to_room: Dict[TilePos, int] = {}
        self._tile_to_corridors: Dict[TilePos, Set[int]] = {}
        self._corridor_to_tiles: Dict[int, Set[TilePos]] = {}

    def add_room(self, room_index: int, room: PlacedRoom) -> None:
        """Record all tiles occupied by ``room`` under ``room_index``."""
        for tile in self._iter_room_tiles(room):
            self._tile_to_room[tile] = room_index

    def remove_room(self, room_index: int, room: PlacedRoom) -> None:
        """Remove room tiles from the index if they still map to ``room_index``."""
        for tile in self._iter_room_tiles(room):
            if self._tile_to_room.get(tile) == room_index:
                self._tile_to_room.pop(tile, None)

    def add_corridor(self, corridor_index: int, tiles: Iterable[TilePos]) -> None:
        """Record all tiles occupied by a corridor."""
        tile_set = set(tiles)
        self._corridor_to_tiles[corridor_index] = tile_set
        for tile in tile_set:
            owners = self._tile_to_corridors.setdefault(tile, set())
            owners.add(corridor_index)

    def remove_corridor(self, corridor_index: int) -> None:
        """Remove a corridor's tiles from the index."""
        tiles = self._corridor_to_tiles.pop(corridor_index, None)
        if not tiles:
            return
        for tile in tiles:
            owners = self._tile_to_corridors.get(tile)
            if owners is None:
                continue
            owners.discard(corridor_index)
            if not owners:
                self._tile_to_corridors.pop(tile, None)

    def replace_corridor(self, corridor_index: int, tiles: Iterable[TilePos]) -> None:
        """Convenience helper for updating a corridor's footprint."""
        self.remove_corridor(corridor_index)
        self.add_corridor(corridor_index, tiles)

    def get_room_at(self, tile: TilePos) -> Optional[int]:
        """Return the room index occupying ``tile`` if any."""
        return self._tile_to_room.get(tile)

    def get_corridors_at(self, tile: TilePos) -> Tuple[int, ...]:
        """Return corridor indices occupying ``tile``."""
        owners = self._tile_to_corridors.get(tile)
        if not owners:
            return ()
        return tuple(sorted(owners))

    def has_corridor_at(self, tile: TilePos) -> bool:
        """Return True if any corridor uses ``tile``."""
        return tile in self._tile_to_corridors

    def is_area_clear(
        self,
        bounds: Rect,
        *,
        ignore_rooms: Optional[Set[int]] = None,
        ignore_corridors: Optional[Set[int]] = None,
    ) -> bool:
        """Return True if ``bounds`` has no occupants except the ignored ones."""
        ignore_rooms = ignore_rooms or set()
        ignore_corridors = ignore_corridors or set()
        for tile in self._iter_tiles_in_rect(bounds):
            room_owner = self._tile_to_room.get(tile)
            if room_owner is not None and room_owner not in ignore_rooms:
                return False
            owners = self._tile_to_corridors.get(tile)
            if owners and not owners.issubset(ignore_corridors):
                return False
        return True

    @staticmethod
    def _iter_room_tiles(room: PlacedRoom) -> Iterator[TilePos]:
        bounds = room.get_bounds()
        return SpatialIndex._iter_tiles_in_rect(bounds)

    @staticmethod
    def _iter_tiles_in_rect(bounds: Rect) -> Iterator[TilePos]:
        for ty in range(bounds.y, bounds.max_y):
            for tx in range(bounds.x, bounds.max_x):
                yield TilePos(tx, ty)

    def clear(self) -> None:
        """Remove all cached data."""
        self._tile_to_room.clear()
        self._tile_to_corridors.clear()
        self._corridor_to_tiles.clear()
