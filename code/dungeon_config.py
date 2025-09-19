"""Configuration container for the dungeon generation prototype."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from constants import door_macro_alignment_offsets
from geometry import Direction
from models import RoomTemplate


@dataclass(frozen=True)
class CorridorLengthDistribution:
    """Configurable distribution for the initial corridor growth step."""

    min_length: int
    max_length: int
    median_length: Optional[float] = None

    def __post_init__(self) -> None:
        min_length = int(self.min_length)
        max_length = int(self.max_length)
        if min_length <= 0:
            raise ValueError("Corridor min_length must be positive")
        if max_length < min_length:
            raise ValueError("Corridor max_length must be >= min_length")

        if self.median_length is None:
            median = (min_length + max_length) / 2.0
        else:
            median = float(self.median_length)
            if not (min_length <= median <= max_length):
                raise ValueError("Corridor median_length must lie within [min_length, max_length]")

        object.__setattr__(self, "min_length", min_length)
        object.__setattr__(self, "max_length", max_length)
        object.__setattr__(self, "median_length", median)

    def sample(self, rng: Optional[random.Random] = None) -> int:
        generator = rng if rng is not None else random
        value = generator.triangular(self.min_length, self.max_length, self.median_length)
        clamped = max(self.min_length, min(self.max_length, value))
        return int(round(clamped))

    @classmethod
    def default(cls) -> "CorridorLengthDistribution":
        return cls(min_length=4, max_length=12, median_length=6)


@dataclass
class DungeonConfig:
    """Aggregates all tunable parameters for dungeon generation."""

    width: int
    height: int
    room_templates: Sequence[RoomTemplate]

    # Probability distribution for number of immediate direct links per room
    direct_link_counts_probs: Mapping[int, float]
    # Number of rooms to place; only checked before placing root rooms, so we can place more.
    num_rooms_to_place: int
    # Minimum empty tiles between room bounding boxes, unless they connect at ports.
    min_room_separation: int
    # We don't place new links if they connect things in the same component that are closer than this number in the dungeon graph.
    min_intra_component_connection_distance: int
    # Corridors longer than this number can be split by through-corridor grower.
    max_desired_corridor_length: int
    # Ban long parallel corridors if they're closer than this along perpendicular axis.
    max_parallel_corridor_perpendicular_distance: int
    # Ban long parallel corridors if their overlap length is over this.
    max_parallel_corridor_overlap: int
    # Reject final dungeon if number of rooms in largest connected component is less than this.
    min_rooms_required: int

    macro_grid_size: int = 4
    random_seed: int | None = None
    collect_metrics: bool = False
    max_connected_placement_attempts: int = 40
    max_consecutive_limit_failures: int = 5
    bent_room_to_corridor_max_room_distance: int = 8
    bent_room_to_corridor_max_branch_distance: int | None = None
    initial_corridor_length: CorridorLengthDistribution | None = None
    _door_macro_alignment_offsets: Mapping[Direction, tuple[float, float]] = field(init=False, repr=False)
    _initial_corridor_length: CorridorLengthDistribution = field(init=False, repr=False)

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
        if self.max_parallel_corridor_perpendicular_distance < 0:
            raise ValueError(
                "DungeonConfig max_parallel_corridor_perpendicular_distance must be non-negative"
            )
        if self.max_parallel_corridor_overlap < 0:
            raise ValueError(
                "DungeonConfig min_parallel_corridor_overlap must be non-negative"
            )
        if self.bent_room_to_corridor_max_room_distance <= 0:
            raise ValueError(
                "DungeonConfig bent_room_to_corridor_max_room_distance must be positive"
            )
        if (
            self.bent_room_to_corridor_max_branch_distance is not None
            and self.bent_room_to_corridor_max_branch_distance <= 0
        ):
            raise ValueError(
                "DungeonConfig bent_room_to_corridor_max_branch_distance must be positive or None"
            )

        initial_corridor_length = (
            self.initial_corridor_length
            if self.initial_corridor_length is not None
            else CorridorLengthDistribution.default()
        )
        object.__setattr__(self, "initial_corridor_length", initial_corridor_length)

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
        self._initial_corridor_length = initial_corridor_length

    @property
    def door_macro_alignment_offsets(self) -> Mapping[Direction, tuple[float, float]]:
        return self._door_macro_alignment_offsets

    @property
    def initial_corridor_length_distribution(self) -> CorridorLengthDistribution:
        return self._initial_corridor_length
