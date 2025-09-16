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
            root_weight=1.5,
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
            root_weight=2.0,
        ),
        RoomTemplate(
            name="room_8x6_2doors",
            size=(8, 6),
            ports=[
                PortTemplate(pos=(0, 2.5), direction=(-1, 0), widths=frozenset({2, 4})),
                PortTemplate(pos=(7, 2.5), direction=(1, 0), widths=frozenset({2, 4})),
            ],
        ),
        RoomTemplate(
            name="room_6x6_90deg",
            size=(6, 6),
            ports=[
                PortTemplate(pos=(0, 1.5), direction=(-1, 0), widths=frozenset({2})),
                PortTemplate(pos=(3.5, 5), direction=(0, 1), widths=frozenset({2})),
            ],
        ),
        RoomTemplate(
            name="room_6x4_deadend",
            size=(6, 4),
            ports=[
                PortTemplate(pos=(2.5, 0), direction=(0, -1), widths=frozenset({2, 4})),
            ],
            root_weight=0.25,
            direct_weight=1.0,
        ),
    ]


def main() -> None:
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    room_templates = build_default_room_templates()
    generator = DungeonGenerator(
        width=100,
        height=40,
        room_templates=room_templates,
        direct_link_counts_probs={0: 0.65, 1: 0.2, 2: 0.1, 3: 0.05},
        num_rooms_to_place=10,
        min_room_separation=1,
    )

    generator.place_rooms()
    generator.create_easy_links()

    num_created = 1
    while num_created > 0:
        num_created = generator.create_easy_t_junctions(fill_probability=1)

    generator.draw_to_grid(draw_macrogrid=False)
    generator.print_grid(horizontal_sep="")


if __name__ == "__main__":
    main()
