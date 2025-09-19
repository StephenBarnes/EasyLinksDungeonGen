#!/usr/bin/env python3

from __future__ import annotations

import random

from dungeon_config import DungeonConfig, CorridorLengthDistribution
from dungeon_generator import DungeonGenerator
from room_templates import prototype_room_templates


def main() -> None:
    # Config for a big "realistic" dungeon - generally seems to be over 90% connected, though the generation process is slow.
    if True:
        config = DungeonConfig(
            width=150,
            height=150,
            room_templates=prototype_room_templates,
            direct_link_counts_probs={0: 0.45, 1: 0.3, 2: 0.2, 3: 0.05},
            num_rooms_to_place=60,
            min_room_separation=1,
            min_intra_component_connection_distance=10,
            corridor_length_for_split=8,
            max_parallel_corridor_perpendicular_distance=8,
            max_parallel_corridor_overlap=5,
            min_rooms_required=35,
            initial_corridor_length=CorridorLengthDistribution(
                min_length=5,
                max_length=25,
                median_length=10,
            ),
            first_root_center_fraction=.3,
            random_seed=None,
        )

    # Smaller config for rapid testing.
    if False:
        config = DungeonConfig(
            width=80,
            height=50,
            room_templates=prototype_room_templates,
            direct_link_counts_probs={0: 0.55, 1: 0.25, 2: 0.15, 3: 0.05},
            # direct_link_counts_probs={0: 1}, # For debugging
            num_rooms_to_place=15,
            min_room_separation=1,
            min_intra_component_connection_distance=10,
            corridor_length_for_split=8,
            max_parallel_corridor_perpendicular_distance=8,
            max_parallel_corridor_overlap=5,
            min_rooms_required=10,
            initial_corridor_length=CorridorLengthDistribution(
                min_length=5,
                max_length=25,
                median_length=10,
            ),
            random_seed=None,
        )


    seed = config.random_seed
    if seed is None:
        # Pick a random seed randomly and print it, so we can reproduce bugs by setting the seed in DungeonConfig for the next run.
        seed = random.randint(0, 1000000)
        config.random_seed = seed
    print(f"Using random seed {seed}")
    random.seed(seed)

    generator = DungeonGenerator(config)

    generator.generate()

    generator.layout.draw_to_grid(draw_macrogrid=True)
    generator.layout.print_grid(horizontal_sep="")

if __name__ == "__main__":
    main()
