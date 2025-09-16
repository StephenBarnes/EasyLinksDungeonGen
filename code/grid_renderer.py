"""Render the dungeon state to an ASCII grid."""

from __future__ import annotations

import random
from typing import List

from dungeon_constants import MACRO_GRID_SIZE
from dungeon_models import Corridor, PlacedRoom


class GridRendererMixin:
    """Provides drawing helpers for visualizing the current layout."""

    width: int
    height: int
    grid: List[List[str]]
    placed_rooms: List[PlacedRoom]
    corridors: List[Corridor]

    def _clear_grid(self) -> None:
        for row in self.grid:
            for x in range(self.width):
                row[x] = " "

    def draw_to_grid(self, draw_macrogrid: bool = False) -> None:
        """Renders the placed rooms, ports, and corridors onto the ASCII grid."""
        self._clear_grid()
        for room in self.placed_rooms:
            room_char = random.choice('OX/LNMW123456789')
            x, y, w, h = room.get_bounds()
            for j in range(h):
                for i in range(w):
                    self.grid[y + j][x + i] = room_char
        for room in self.placed_rooms:
            for port in room.get_world_ports():
                for tx, ty in port.tiles:
                    self.grid[ty][tx] = '█'
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
                if (0 < (x % MACRO_GRID_SIZE) < MACRO_GRID_SIZE - 1) or (
                    0 < (y % MACRO_GRID_SIZE) < MACRO_GRID_SIZE - 1
                ):
                    continue
                self.grid[y][x] = '░'

    def mark_room_interior_on_grid(self, room_idx: int) -> None:
        """Mark the interior of the specified room_idx with asterisks."""
        room = self.placed_rooms[room_idx]
        bounds = room.get_bounds()
        for x in range(bounds[0] + 1, bounds[0] + bounds[2] - 1):
            for y in range(bounds[1] + 1, bounds[1] + bounds[3] - 1):
                self.grid[y][x] = "*"

    def print_grid(self, horizontal_sep: str = "") -> None:
        """Prints the ASCII grid to the console."""
        for row in self.grid:
            print(horizontal_sep.join(row))
