import random

import pytest

from dungeon_config import CorridorLengthDistribution, DungeonConfig
from dungeon_layout import DungeonLayout
from geometry import Direction, Rotation
from models import RoomKind
from root_room_placer import RootRoomPlacer


@pytest.fixture
def root_room_placer(dungeon_config, dungeon_layout, standalone_template):
    room_templates_by_kind = {RoomKind.STANDALONE: [standalone_template]}
    return RootRoomPlacer(dungeon_config, dungeon_layout, room_templates_by_kind)


@pytest.mark.parametrize(
    "distance,span,expected",
    [
        (0, 40, "close"),
        (12, 40, "intermediate"),
        (20, 40, "far"),
        (5, 0, "far"),
    ],
)
def test_categorize_side_distance(distance, span, expected, root_room_placer):
    assert root_room_placer._categorize_side_distance(distance, span) == expected


def test_describe_macro_position_classifies_edge_and_middle(root_room_placer):
    category_edge, proximities_edge = root_room_placer._describe_macro_position(4, 36)
    category_middle, proximities_middle = root_room_placer._describe_macro_position(20, 20)

    assert category_edge == "edge"
    assert proximities_edge["left"] == "close"
    assert proximities_edge["bottom"] == "close"
    assert category_middle == "middle"
    assert all(value == "far" for value in proximities_middle.values())


def test_select_root_rotation_prefers_inward_direction(root_room_placer, standalone_template):
    template = standalone_template
    template.preferred_center_facing_dir = Direction.NORTH
    placement_category = "edge"
    side_proximities = {"left": "close", "right": "far", "top": "far", "bottom": "far"}

    rotation = root_room_placer._select_root_rotation(template, placement_category, side_proximities)

    assert rotation is Rotation.DEG_270


def test_select_root_rotation_falls_back_to_random(root_room_placer, monkeypatch, standalone_template):
    template = standalone_template
    template.preferred_center_facing_dir = None
    chosen_rotation = Rotation.DEG_180

    monkeypatch.setattr(Rotation, "random", staticmethod(lambda: chosen_rotation))

    rotation = root_room_placer._select_root_rotation(template, "edge", {"left": "close"})

    assert rotation is chosen_rotation


def test_build_root_room_candidate_aligns_using_macro_offsets(root_room_placer, monkeypatch, standalone_template):
    template = standalone_template
    rotation = Rotation.DEG_0
    macro_x, macro_y = 12, 16
    # Always pick the north-facing port so direction is predictable.
    monkeypatch.setattr(random, "randrange", lambda _: 0)

    candidate = root_room_placer._build_root_room_candidate(template, rotation, macro_x, macro_y)

    assert candidate.x % root_room_placer.config.macro_grid_size == 0
    assert candidate.y % root_room_placer.config.macro_grid_size == 0


def test_place_rooms_creates_single_component(root_room_placer):
    random.seed(12345)
    root_room_placer.place_rooms()

    component_sizes = root_room_placer.layout.get_component_sizes()
    assert len(component_sizes) == 1
    component_id = next(iter(component_sizes))
    layout = root_room_placer.layout
    assert len(layout.placed_rooms) >= 1
    assert all(room.component_id == component_id for room in layout.placed_rooms)
    assert all(corridor.component_id == component_id for corridor in layout.corridors)


def test_corridor_length_distribution_within_bounds(root_room_placer):
    dist = root_room_placer.config.initial_corridor_length_distribution
    for _ in range(20):
        value = dist.sample()
        assert dist.min_length <= value <= dist.max_length


def test_place_rooms_spawns_corridor_when_possible(standalone_template):
    state = random.getstate()
    try:
        random.seed(4242)
        config = DungeonConfig(
            width=60,
            height=60,
            room_templates=[standalone_template],
            direct_link_counts_probs={0: 1.0},
            num_rooms_to_place=6,
            min_room_separation=1,
            min_intra_component_connection_distance=3,
            max_desired_corridor_length=20,
            max_parallel_corridor_perpendicular_distance=10,
            max_parallel_corridor_overlap=8,
            min_rooms_required=1,
            macro_grid_size=4,
            max_connected_placement_attempts=40,
            max_consecutive_limit_failures=5,
            initial_corridor_length=CorridorLengthDistribution(
                min_length=4,
                max_length=4,
                median_length=4,
            ),
        )
        layout = DungeonLayout(config)
        placer = RootRoomPlacer(
            config,
            layout,
            {RoomKind.STANDALONE: [standalone_template]},
        )

        placer.place_rooms()

        assert len(layout.corridors) >= 1
        assert len(layout.room_corridor_links) >= 2
        assert len(layout.placed_rooms) >= 2
    finally:
        random.setstate(state)
