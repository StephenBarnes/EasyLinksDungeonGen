import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dungeon_config import DungeonConfig
from dungeon_layout import DungeonLayout
from geometry import Direction
from models import PortTemplate, RoomKind, RoomTemplate


@pytest.fixture
def standalone_template() -> RoomTemplate:
    return RoomTemplate(
        name="room_8x8_4doors",
        size=(8, 8),
        ports=[
            PortTemplate(pos=(3.5, 0.0), direction=Direction.NORTH, widths=frozenset((2, 4))),
            PortTemplate(pos=(3.5, 7.0), direction=Direction.SOUTH, widths=frozenset((2, 4))),
            PortTemplate(pos=(0.0, 3.5), direction=Direction.WEST, widths=frozenset((2, 4))),
            PortTemplate(pos=(7.0, 3.5), direction=Direction.EAST, widths=frozenset((2, 4))),
        ],
        kinds=frozenset((RoomKind.STANDALONE,)),
        root_weight_middle=1.5,
        root_weight_edge=0.4,
        root_weight_intermediate=1.0,
        direct_weight=1.5,
    )


@pytest.fixture
def dungeon_config(standalone_template: RoomTemplate) -> DungeonConfig:
    return DungeonConfig(
        width=40,
        height=40,
        room_templates=[standalone_template],
        direct_link_counts_probs={0: 0.5, 1: 0.5},
        num_rooms_to_place=5,
        min_room_separation=2,
        min_intra_component_connection_distance=3,
        max_desired_corridor_length=20,
        max_parallel_corridor_perpendicular_distance=2,
        max_parallel_corridor_overlap=4,
        min_rooms_required=1,
        macro_grid_size=4,
        max_connected_placement_attempts=20,
        max_consecutive_limit_failures=3,
    )


@pytest.fixture
def dungeon_layout(dungeon_config: DungeonConfig) -> DungeonLayout:
    return DungeonLayout(dungeon_config)
