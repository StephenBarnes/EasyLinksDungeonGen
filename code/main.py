#!/usr/bin/env python3

from __future__ import annotations

import random

from dungeon_constants import RANDOM_SEED
from dungeon_generator import DungeonGenerator
from dungeon_models import PortTemplate, RoomTemplate


def build_default_room_templates() -> list[RoomTemplate]:
    """Return the handcrafted room set from the prototype."""
    return [
        RoomTemplate(
            name="room_8x8_4doors",
            size=(8, 8),
            ports=[
                PortTemplate(pos=(3.5, 0), direction=(0, -1), widths=frozenset((2, 4))),
                PortTemplate(pos=(3.5, 7), direction=(0, 1), widths=frozenset({2, 4})),
                PortTemplate(pos=(0, 3.5), direction=(-1, 0), widths=frozenset({2, 4})),
                PortTemplate(pos=(7, 3.5), direction=(1, 0), widths=frozenset({2, 4})),
            ],
            root_weight_middle=1.5,
            root_weight_edge=0.4,
            root_weight_intermediate=1.0,
            direct_weight=1.5,
        ),
        RoomTemplate(
            name="room_8x10_5doors",
            size=(8, 10),
            ports=[
                PortTemplate(pos=(3.5, 9), direction=(0, 1), widths=frozenset({2, 4})),
                PortTemplate(pos=(0, 1.5), direction=(-1, 0), widths=frozenset({2})),
                PortTemplate(pos=(0, 5.5), direction=(-1, 0), widths=frozenset({2})),
                PortTemplate(pos=(7, 1.5), direction=(1, 0), widths=frozenset({2})),
                PortTemplate(pos=(7, 5.5), direction=(1, 0), widths=frozenset({2})),
            ],
            root_weight_middle=2.0,
            root_weight_edge=0.5,
            root_weight_intermediate=1.2,
        ),
        RoomTemplate(
            name="room_8x6_2doors",
            size=(8, 6),
            ports=[
                PortTemplate(pos=(0, 2.5), direction=(-1, 0), widths=frozenset({2, 4})),
                PortTemplate(pos=(7, 2.5), direction=(1, 0), widths=frozenset({2, 4})),
            ],
            root_weight_middle=1.0,
            root_weight_edge=1.0,
            root_weight_intermediate=1.0,
        ),
        RoomTemplate(
            name="room_6x6_90deg",
            size=(6, 6),
            ports=[
                PortTemplate(pos=(0, 1.5), direction=(-1, 0), widths=frozenset({2})),
                PortTemplate(pos=(3.5, 5), direction=(0, 1), widths=frozenset({2})),
            ],
            root_weight_middle=0.7,
            root_weight_edge=0.9,
            root_weight_intermediate=1.1,
        ),
        RoomTemplate(
            name="room_6x4_deadend",
            size=(6, 4),
            ports=[
                PortTemplate(pos=(2.5, 0), direction=(0, -1), widths=frozenset({2, 4})),
            ],
            root_weight_middle=0.1,
            root_weight_edge=2,
            root_weight_intermediate=0.3,
            preferred_center_facing_dir=(0, -1)
        ),
    ]


def main() -> None:
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    room_templates = build_default_room_templates()
    generator = DungeonGenerator(
        width=230,
        height=60,
        room_templates=room_templates,
        direct_link_counts_probs={0: 0.65, 1: 0.2, 2: 0.1, 3: 0.05},
        num_rooms_to_place=50,
        min_room_separation=1,
    )

    # Step 1: place rooms, some with direct links
    generator.place_rooms()

    # Step 2: create direct corridors
    generator.create_easy_links()

    # Step 3: create T-junctions with direct corridors
    num_created = 1
    while num_created > 0:
        num_created = generator.create_easy_t_junctions(fill_probability=1, step_num=3)

    # Step 4: create bent links between rooms
    num_created = generator.create_bent_room_links()

    # Step 5: create T-junctions with the new corridors, if any
    while num_created > 0:
        num_created = generator.create_easy_t_junctions(fill_probability=1, step_num=5)

    # Debug: check number of components
    print(f"Component count: {len(generator.get_component_summary())}")

    generator.draw_to_grid(draw_macrogrid=False)
    generator.print_grid(horizontal_sep="")

if __name__ == "__main__":
    main()
