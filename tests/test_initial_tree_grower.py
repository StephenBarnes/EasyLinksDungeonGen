import random

import pytest

from dungeon_config import CorridorLengthDistribution, DungeonConfig
from dungeon_layout import DungeonLayout
from geometry import Direction, Rotation
from grower_context import GrowerContext
from growers.initial_tree import (
    InitialTreeHelper,
    categorize_side_distance,
    run_initial_tree_grower,
)
from models import RoomKind


@pytest.fixture
def tree_context(dungeon_config: DungeonConfig, dungeon_layout: DungeonLayout, standalone_template):
    room_templates = [standalone_template]
    room_templates_by_kind = {RoomKind.STANDALONE: room_templates}
    return GrowerContext(
        config=dungeon_config,
        layout=dungeon_layout,
        room_templates=room_templates,
        room_templates_by_kind=room_templates_by_kind,
    )


@pytest.fixture
def initial_tree_helper(tree_context: GrowerContext) -> InitialTreeHelper:
    return InitialTreeHelper(tree_context)


@pytest.mark.parametrize(
    "distance,span,expected",
    [
        (0, 40, "close"),
        (12, 40, "intermediate"),
        (20, 40, "far"),
        (5, 0, "far"),
    ],
)
def test_categorize_side_distance(distance, span, expected):
    assert categorize_side_distance(distance, span) == expected


def test_describe_macro_position_classifies_edge_and_middle(initial_tree_helper: InitialTreeHelper):
    category_edge, proximities_edge = initial_tree_helper._describe_macro_position(4, 36)
    category_middle, proximities_middle = initial_tree_helper._describe_macro_position(20, 20)

    assert category_edge == "edge"
    assert proximities_edge["left"] == "close"
    assert proximities_edge["bottom"] == "close"
    assert category_middle == "middle"
    assert all(value == "far" for value in proximities_middle.values())


def test_select_root_rotation_prefers_inward_direction(
    initial_tree_helper: InitialTreeHelper, standalone_template
):
    template = standalone_template
    template.preferred_center_facing_dir = Direction.NORTH
    placement_category = "edge"
    side_proximities = {"left": "close", "right": "far", "top": "far", "bottom": "far"}

    rotation = initial_tree_helper._select_root_rotation(
        template, placement_category, side_proximities
    )

    assert rotation is Rotation.DEG_270


def test_select_root_rotation_falls_back_to_random(
    initial_tree_helper: InitialTreeHelper, monkeypatch, standalone_template
):
    template = standalone_template
    template.preferred_center_facing_dir = None
    chosen_rotation = Rotation.DEG_180

    monkeypatch.setattr(Rotation, "random", staticmethod(lambda: chosen_rotation))

    rotation = initial_tree_helper._select_root_rotation(template, "edge", {"left": "close"})

    assert rotation is chosen_rotation


def test_build_root_room_candidate_aligns_using_macro_offsets(
    initial_tree_helper: InitialTreeHelper, monkeypatch, standalone_template
):
    template = standalone_template
    rotation = Rotation.DEG_0
    macro_x, macro_y = 12, 16
    monkeypatch.setattr(random, "randrange", lambda _: 0)

    candidate = initial_tree_helper._build_root_room_candidate(template, rotation, macro_x, macro_y)

    config = initial_tree_helper.config
    assert candidate.x % config.macro_grid_size == 0
    assert candidate.y % config.macro_grid_size == 0


def test_run_initial_tree_creates_single_component(tree_context: GrowerContext):
    random.seed(12345)
    run_initial_tree_grower(tree_context)

    component_sizes = tree_context.layout.get_component_sizes()
    assert len(component_sizes) == 1
    component_id = next(iter(component_sizes))
    layout = tree_context.layout
    assert len(layout.placed_rooms) >= 1
    assert all(room.component_id == component_id for room in layout.placed_rooms)
    assert all(corridor.component_id == component_id for corridor in layout.corridors)


def test_corridor_length_distribution_within_bounds(tree_context: GrowerContext):
    dist = tree_context.config.initial_corridor_length
    for _ in range(20):
        value = dist.sample()
        assert dist.min_length <= value <= dist.max_length


def test_run_initial_tree_spawns_corridor_when_possible(standalone_template):
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
            corridor_length_for_split=20,
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
        room_templates = [standalone_template]
        context = GrowerContext(
            config=config,
            layout=layout,
            room_templates=room_templates,
            room_templates_by_kind={RoomKind.STANDALONE: room_templates},
        )

        run_initial_tree_grower(context)

        assert len(layout.corridors) >= 1
        assert len(layout.room_corridor_links) >= 2
        assert len(layout.placed_rooms) >= 2
    finally:
        random.setstate(state)
