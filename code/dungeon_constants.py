"""Utility helpers that were formerly global constants."""

from __future__ import annotations

from dungeon_geometry import Direction


def door_macro_alignment_offsets(macro_grid_size: int) -> dict[Direction, tuple[float, float]]:
    """Return the door alignment offsets for a given macro-grid size."""

    door_frac_offset = macro_grid_size - 0.5
    door_whole_offset = float(macro_grid_size - 1)
    return {
        Direction.NORTH: (door_frac_offset, 0.0),
        Direction.SOUTH: (door_frac_offset, door_whole_offset),
        Direction.WEST: (0.0, door_frac_offset),
        Direction.EAST: (door_whole_offset, door_frac_offset),
    }


__all__ = ["door_macro_alignment_offsets"]
