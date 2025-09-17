"""Configuration container for the dungeon generation prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from constants import door_macro_alignment_offsets
from geometry import Direction
from models import RoomTemplate


@dataclass
class DungeonConfig:
    """Aggregates all tunable parameters for dungeon generation."""

    width: int
    height: int
    room_templates: Sequence[RoomTemplate]
    # Probability distribution for number of immediate direct links per room
    # Example: {0: 0.4, 1: 0.3, 2: 0.3}
    direct_link_counts_probs: Mapping[int, float]
    num_rooms_to_place: int
    # Minimum empty tiles between room bounding boxes, unless they connect at ports.
    min_room_separation: int
    min_rooms_required: int = 6
    macro_grid_size: int = 4
    random_seed: int | None = None
    max_connected_placement_attempts: int = 40
    max_consecutive_limit_failures: int = 5
    min_intra_component_connection_distance: int = 10
    max_desired_corridor_length: int = 10
    _door_macro_alignment_offsets: Mapping[Direction, tuple[float, float]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("DungeonConfig width and height must be positive")
        if self.num_rooms_to_place <= 0:
            raise ValueError("DungeonConfig num_rooms_to_place must be positive")
        if self.min_room_separation < 0:
            raise ValueError("DungeonConfig min_room_separation cannot be negative")
        if self.min_rooms_required <= 0:
            raise ValueError("DungeonConfig min_rooms_required must be positive")
        if self.macro_grid_size <= 0:
            raise ValueError("DungeonConfig macro_grid_size must be positive")
        if self.max_connected_placement_attempts <= 0:
            raise ValueError("DungeonConfig max_connected_placement_attempts must be positive")
        if self.max_consecutive_limit_failures <= 0:
            raise ValueError("DungeonConfig max_consecutive_limit_failures must be positive")
        if self.min_intra_component_connection_distance < 0:
            raise ValueError(
                "DungeonConfig min_intra_component_connection_distance must be non-negative"
            )

        self.room_templates = tuple(self.room_templates)
        if not self.room_templates:
            raise ValueError("DungeonConfig requires at least one room template")

        template_macro_sizes = {template.macro_grid_size for template in self.room_templates}
        if template_macro_sizes != {self.macro_grid_size}:
            raise ValueError(
                "All room templates must use the same macro_grid_size as the dungeon config"
            )

        self.direct_link_counts_probs = {
            int(count): float(prob) for count, prob in self.direct_link_counts_probs.items()
        }
        if not self.direct_link_counts_probs:
            raise ValueError("DungeonConfig requires direct_link_counts_probs entries")
        if any(prob < 0 for prob in self.direct_link_counts_probs.values()):
            raise ValueError("Direct link probabilities must be non-negative")
        if sum(self.direct_link_counts_probs.values()) <= 0:
            raise ValueError("Direct link probabilities must sum to a positive value")

        self._door_macro_alignment_offsets = door_macro_alignment_offsets(self.macro_grid_size)

    @property
    def door_macro_alignment_offsets(self) -> Mapping[Direction, tuple[float, float]]:
        return self._door_macro_alignment_offsets
