"""Shared constants for the dungeon generation prototype."""

from __future__ import annotations

from dungeon_geometry import Direction, Rotation

MACRO_GRID_SIZE = 4
RANDOM_SEED = None  # Set to a number for reproducible behavior (for debugging); set to None to produce different dungeon on every run.

# Door centers must land on these macro-grid offsets per-facing direction.
DOOR_FRAC_OFFSET = MACRO_GRID_SIZE - 0.5
DOOR_WHOLE_OFFSET = float(MACRO_GRID_SIZE - 1)
DOOR_MACRO_ALIGNMENT_OFFSETS = {
    Direction.NORTH: (DOOR_FRAC_OFFSET, 0.0),
    Direction.SOUTH: (DOOR_FRAC_OFFSET, DOOR_WHOLE_OFFSET),
    Direction.WEST: (0.0, DOOR_FRAC_OFFSET),
    Direction.EAST: (DOOR_WHOLE_OFFSET, DOOR_FRAC_OFFSET),
}

MAX_CONNECTED_PLACEMENT_ATTEMPTS = 40
MAX_CONSECUTIVE_LIMIT_FAILURES = 5 # We abort adding new rooms after we've reached max placement attempts too many times in a row.
