import random

import pytest

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
