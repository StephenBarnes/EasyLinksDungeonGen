#!/usr/bin/env python3

from __future__ import annotations

import random

from dungeon_config import DungeonConfig
from dungeon_generator import DungeonGenerator
from dungeon_models import PortTemplate, RoomTemplate, RoomKind
from dungeon_geometry import Direction


def build_default_room_templates() -> list[RoomTemplate]:
    """Return the handcrafted room set from the prototype."""
    return [
        RoomTemplate(
            name="room_8x8_4doors",
            size=(8, 8),
            ports=[
                PortTemplate(pos=(3.5, 0), direction=Direction.NORTH, widths=frozenset((2, 4))),
                PortTemplate(pos=(3.5, 7), direction=Direction.SOUTH, widths=frozenset((2, 4))),
                PortTemplate(pos=(0, 3.5), direction=Direction.WEST, widths=frozenset((2, 4))),
                PortTemplate(pos=(7, 3.5), direction=Direction.EAST, widths=frozenset((2, 4))),
            ],
            root_weight_middle=1.5,
            root_weight_edge=0.4,
            root_weight_intermediate=1.0,
            direct_weight=1.5,
            kinds=frozenset((RoomKind.STANDALONE, RoomKind.FOUR_WAY)),
        ),
        RoomTemplate(
            name="room_8x10_5doors",
            size=(8, 10),
            ports=[
                PortTemplate(pos=(3.5, 9), direction=Direction.SOUTH, widths=frozenset((2, 4))),
                PortTemplate(pos=(0, 1.5), direction=Direction.WEST, widths=frozenset((2,))),
                PortTemplate(pos=(0, 5.5), direction=Direction.WEST, widths=frozenset((2,))),
                PortTemplate(pos=(7, 1.5), direction=Direction.EAST, widths=frozenset((2,))),
                PortTemplate(pos=(7, 5.5), direction=Direction.EAST, widths=frozenset((2,))),
            ],
            root_weight_middle=2.0,
            root_weight_edge=0.5,
            root_weight_intermediate=1.2,
            kinds=frozenset((RoomKind.STANDALONE,)),
        ),
        RoomTemplate(
            name="room_8x6_2doors",
            size=(8, 6),
            ports=[
                PortTemplate(pos=(0, 2.5), direction=Direction.WEST, widths=frozenset((2, 4))),
                PortTemplate(pos=(7, 2.5), direction=Direction.EAST, widths=frozenset((2, 4))),
            ],
            root_weight_middle=1.0,
            root_weight_edge=1.0,
            root_weight_intermediate=1.0,
            kinds=frozenset((RoomKind.STANDALONE,)),
        ),
        RoomTemplate(
            name="room_6x6_90deg",
            size=(6, 6),
            ports=[
                PortTemplate(pos=(0, 1.5), direction=Direction.WEST, widths=frozenset((2,))),
                PortTemplate(pos=(3.5, 5), direction=Direction.SOUTH, widths=frozenset((2,))),
            ],
            root_weight_middle=0.7,
            root_weight_edge=0.9,
            root_weight_intermediate=1.1,
            kinds=frozenset((RoomKind.STANDALONE, RoomKind.BEND)),
        ),
        RoomTemplate(
            name="room_6x4_deadend",
            size=(6, 4),
            ports=[
                PortTemplate(pos=(2.5, 0), direction=Direction.NORTH, widths=frozenset((2, 4))),
            ],
            root_weight_middle=0.1,
            root_weight_edge=2,
            root_weight_intermediate=0.3,
            preferred_center_facing_dir=Direction.NORTH,
            kinds=frozenset((RoomKind.STANDALONE,)),
        ),

        # Special room templates for 90-degree bends.
        RoomTemplate(
            name="bend_2x2_right",
            size=(2, 2),
            ports=[
                PortTemplate(pos=(1, 0.5), direction=Direction.EAST, widths=frozenset((2,))),
                PortTemplate(pos=(0.5, 1), direction=Direction.SOUTH, widths=frozenset((2,))),
            ],
            kinds=frozenset((RoomKind.BEND,)),
            allow_door_overlaps=True,
        ),
        RoomTemplate(
            name="bend_2x2_left",
            size=(2, 2),
            ports=[
                PortTemplate(pos=(0, 0.5), direction=Direction.WEST, widths=frozenset((2,))),
                PortTemplate(pos=(0.5, 1), direction=Direction.SOUTH, widths=frozenset((2,))),
            ],
            kinds=frozenset((RoomKind.BEND,)),
            allow_door_overlaps=True,
        ),
        RoomTemplate(
            name="bend_2x4_right",
            size=(2, 4),
            ports=[
                PortTemplate(pos=(1, 1.5), direction=Direction.EAST, widths=frozenset((4,2))),
                PortTemplate(pos=(0.5, 3), direction=Direction.SOUTH, widths=frozenset((2,))),
            ],
            kinds=frozenset((RoomKind.BEND,)),
            allow_door_overlaps=True,
        ),
        RoomTemplate(
            name="bend_2x4_left",
            size=(2, 4),
            ports=[
                PortTemplate(pos=(0, 1.5), direction=Direction.WEST, widths=frozenset((4,2))),
                PortTemplate(pos=(0.5, 3), direction=Direction.SOUTH, widths=frozenset((2,))),
            ],
            kinds=frozenset((RoomKind.BEND,)),
            allow_door_overlaps=True,
        ),
        RoomTemplate(
            name="bend_4x4_right",
            size=(4, 4),
            ports=[
                PortTemplate(pos=(3, 1.5), direction=Direction.EAST, widths=frozenset((4,2))),
                PortTemplate(pos=(1.5, 3), direction=Direction.SOUTH, widths=frozenset((4,2))),
            ],
            kinds=frozenset((RoomKind.BEND,)),
            allow_door_overlaps=True,
        ),
        RoomTemplate(
            name="bend_4x4_left",
            size=(4, 4),
            ports=[
                PortTemplate(pos=(0, 1.5), direction=Direction.WEST, widths=frozenset((4,2))),
                PortTemplate(pos=(1.5, 3), direction=Direction.SOUTH, widths=frozenset((4,2))),
            ],
            kinds=frozenset((RoomKind.BEND,)),
            allow_door_overlaps=True,
        ),

        # Special room templates for T-junctions and 4-way intersections.
        RoomTemplate(
            name="junction_2x2",
            size=(2, 2),
            ports=[
                PortTemplate(pos=(0.5, 0), direction=Direction.NORTH, widths=frozenset((2,))),
                PortTemplate(pos=(0.5, 1), direction=Direction.SOUTH, widths=frozenset((2,))),
                PortTemplate(pos=(1, 0.5), direction=Direction.EAST, widths=frozenset((2,))),
                PortTemplate(pos=(0, 0.5), direction=Direction.WEST, widths=frozenset((2,))),
            ],
            kinds=frozenset((RoomKind.T_JUNCTION, RoomKind.FOUR_WAY)),
            allow_door_overlaps=True,
        ),
        RoomTemplate(
            name="junction_2x4",
            size=(2, 4),
            ports=[
                PortTemplate(pos=(0.5, 0), direction=Direction.NORTH, widths=frozenset((2,))),
                PortTemplate(pos=(0.5, 3), direction=Direction.SOUTH, widths=frozenset((2,))),
                PortTemplate(pos=(1, 1.5), direction=Direction.EAST, widths=frozenset((4,2))),
                PortTemplate(pos=(0, 1.5), direction=Direction.WEST, widths=frozenset((4,2))),
            ],
            kinds=frozenset((RoomKind.T_JUNCTION, RoomKind.FOUR_WAY)),
            allow_door_overlaps=True,
        ),
        RoomTemplate(
            name="junction_4x4",
            size=(4, 4),
            ports=[
                PortTemplate(pos=(1.5, 0), direction=Direction.NORTH, widths=frozenset((4,2))),
                PortTemplate(pos=(1.5, 3), direction=Direction.SOUTH, widths=frozenset((4,2))),
                PortTemplate(pos=(0, 1.5), direction=Direction.WEST, widths=frozenset((4,2))),
                PortTemplate(pos=(3, 1.5), direction=Direction.EAST, widths=frozenset((4,2))),
            ],
            kinds=frozenset((RoomKind.T_JUNCTION, RoomKind.FOUR_WAY)),
            allow_door_overlaps=True,
        ),
    ]


def main() -> None:
    room_templates = build_default_room_templates()
    config = DungeonConfig(
        width=120,
        height=50,
        room_templates=room_templates,
        direct_link_counts_probs={0: 0.55, 1: 0.25, 2: 0.15, 3: 0.05},
        num_rooms_to_place=30,
        min_room_separation=1,
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

    # Debug: check number of components is correct
    print(f"Component count: {len(generator.get_component_summary())}")

    generator.draw_to_grid(draw_macrogrid=True)
    generator.print_grid(horizontal_sep="")

if __name__ == "__main__":
    main()
