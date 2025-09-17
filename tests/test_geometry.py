import math
import pytest

from geometry import (
    Direction,
    Rect,
    Rotation,
    TilePos,
    port_tiles_from_world_pos,
    rotate_direction,
    rotate_point,
)


def test_rotation_from_degrees_normalizes_and_computes_quarter_turns():
    rotation = Rotation.from_degrees(450)

    assert rotation is Rotation.DEG_90
    assert rotation.quarter_turns() == 1
    assert Rotation.DEG_180.quarter_turns() == 2


@pytest.mark.parametrize(
    "point,width,height,rotation,expected",
    [
        ((1, 2), 4, 6, Rotation.DEG_0, (1, 2)),
        ((1, 2), 4, 6, Rotation.DEG_90, (2, 4 - 1 - 1)),
        ((1, 2), 4, 6, Rotation.DEG_180, (4 - 1 - 1, 6 - 1 - 2)),
        ((1, 2), 4, 6, Rotation.DEG_270, (6 - 1 - 2, 1)),
    ],
)
def test_rotate_point_matches_expected_transform(point, width, height, rotation, expected):
    assert rotate_point(point[0], point[1], width, height, rotation) == pytest.approx(expected)


@pytest.mark.parametrize(
    "direction,rotation,expected",
    [
        (Direction.NORTH, Rotation.DEG_270, Direction.EAST),
        (Direction.SOUTH, Rotation.DEG_90, Direction.EAST),
        (Direction.WEST, Rotation.DEG_180, Direction.EAST),
    ],
)
def test_rotate_direction_produces_expected_cardinal(direction, rotation, expected):
    assert rotate_direction(direction, rotation) is expected


def test_port_tiles_from_world_pos_handles_horizontal_and_vertical_ports():
    horizontal_tiles = port_tiles_from_world_pos(4.5, 10.0)
    vertical_tiles = port_tiles_from_world_pos(3.0, 7.5)

    assert horizontal_tiles == (TilePos(4, 10), TilePos(5, 10))
    assert vertical_tiles == (TilePos(3, 7), TilePos(3, 8))

    with pytest.raises(ValueError):
        port_tiles_from_world_pos(2.0, 3.0)


@pytest.mark.parametrize(
    "rect_a,rect_b,expected",
    [
        (Rect(0, 0, 3, 3), Rect(2, 2, 3, 3), True),
        (Rect(0, 0, 2, 2), Rect(2, 2, 2, 2), False),
        (Rect(0, 0, 5, 5), Rect(5, 0, 3, 3), False),
    ],
)
def test_rect_overlaps(rect_a, rect_b, expected):
    assert rect_a.overlaps(rect_b) is expected
    assert rect_b.overlaps(rect_a) is expected


def test_rect_expand_grows_bounds_evenly():
    rect = Rect(2, 3, 4, 5)

    expanded = rect.expand(2)

    assert expanded.x == 0
    assert expanded.y == 1
    assert expanded.width == 8
    assert expanded.height == 9
    # Original rect should remain unchanged.
    assert rect == Rect(2, 3, 4, 5)
