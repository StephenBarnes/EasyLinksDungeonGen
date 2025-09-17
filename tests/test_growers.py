import random

from geometry import Rotation
from models import PlacedRoom, RoomKind
from growers.bent_room_to_corridor import run_bent_room_to_corridor_grower
from growers.bent_room_to_room import run_bent_room_to_room_grower
from growers.room_to_corridor import run_room_to_corridor_grower
from growers.room_to_room import run_room_to_room_grower


def _place_room(layout, template, x, y, *, rotation=Rotation.DEG_0, connected_ports=None) -> PlacedRoom:
    room = PlacedRoom(template, x, y, rotation)
    if connected_ports:
        room.connected_port_indices.update(connected_ports)
    component_id = layout.new_component_id()
    layout.register_room(room, component_id)
    return room


def _room_template_by_name(context, name: str):
    return next(template for template in context.room_templates if template.name == name)


def test_room_to_room_creates_four_way_intersection(make_context):
    random.seed(0)
    context = make_context()
    layout = context.layout
    template = _room_template_by_name(context, "room_8x8_4doors")

    for x, y in ((2, 10), (30, 10), (16, 2), (16, 26)):
        _place_room(layout, template, x, y)

    initial_rooms = len(layout.placed_rooms)

    created = run_room_to_room_grower(context)

    assert created == 4
    assert len(layout.placed_rooms) == initial_rooms + 1

    junction_index = len(layout.placed_rooms) - 1
    junction_room = layout.placed_rooms[junction_index]
    assert RoomKind.FOUR_WAY in junction_room.template.kinds

    assert len(layout.corridors) == 4
    assert {corridor.geometry.axis_index for corridor in layout.corridors} == {0, 1}
    for corridor in layout.corridors:
        assert junction_index in (corridor.room_a_index, corridor.room_b_index)

    assert context.layout.component_manager.has_single_component()


def test_bent_room_to_room_connects_diagonal_rooms(make_context):
    random.seed(0)
    context = make_context()
    layout = context.layout
    template = _room_template_by_name(context, "room_8x8_4doors")

    primary = _place_room(layout, template, 4, 24, connected_ports={0, 1, 2})
    partner = _place_room(layout, template, 32, 4, connected_ports={0, 2, 3})
    assert primary.index == 0
    assert partner.index == 1

    initial_rooms = len(layout.placed_rooms)

    created = run_bent_room_to_room_grower(context, stop_after_first=True)

    assert created == 1
    assert len(layout.placed_rooms) == initial_rooms + 1

    bend_room = layout.placed_rooms[-1]
    assert RoomKind.BEND in bend_room.template.kinds

    assert len(layout.corridors) == 2
    for corridor in layout.corridors:
        assert bend_room.index in (corridor.room_a_index, corridor.room_b_index)


def test_room_to_corridor_creates_t_junction(make_context):
    context = make_context()
    layout = context.layout
    template = _room_template_by_name(context, "room_8x8_4doors")

    random.seed(0)
    _place_room(layout, template, 4, 20)
    _place_room(layout, template, 28, 20)
    run_room_to_room_grower(context)

    north_room = _place_room(layout, template, 16, 4, connected_ports={0, 2, 3})

    initial_rooms = len(layout.placed_rooms)

    random.seed(1)
    created = run_room_to_corridor_grower(context, fill_probability=1.0)

    assert created == 1
    assert len(layout.placed_rooms) == initial_rooms + 1

    junction = layout.placed_rooms[-1]
    assert RoomKind.T_JUNCTION in junction.template.kinds

    assert len(layout.corridors) == 3
    branch = [
        corridor
        for corridor in layout.corridors
        if north_room.index in (corridor.room_a_index, corridor.room_b_index)
    ]
    assert len(branch) == 1
    assert branch[0].geometry.axis_index == 1


def test_bent_room_to_corridor_adds_bend_and_junction(make_context):
    random.seed(0)
    context = make_context()
    layout = context.layout
    template = _room_template_by_name(context, "room_8x8_4doors")

    _place_room(layout, template, 4, 20)
    _place_room(layout, template, 28, 20)
    run_room_to_room_grower(context)

    candidate = _place_room(layout, template, 0, 0, connected_ports={0, 1, 2})

    initial_rooms = len(layout.placed_rooms)

    random.seed(0)
    created = run_bent_room_to_corridor_grower(
        context,
        stop_after_first=True,
        fill_probability=1.0,
    )

    assert created == 1
    assert len(layout.placed_rooms) == initial_rooms + 2

    new_rooms = layout.placed_rooms[-2:]
    assert any(RoomKind.BEND in room.template.kinds for room in new_rooms)
    assert any(RoomKind.T_JUNCTION in room.template.kinds for room in new_rooms)

    assert len(layout.corridors) == 4
    candidate_links = [
        corridor
        for corridor in layout.corridors
        if candidate.index in (corridor.room_a_index, corridor.room_b_index)
    ]
    assert len(candidate_links) == 1
    bend_room = next(room for room in new_rooms if RoomKind.BEND in room.template.kinds)
    bend_links = [
        corridor
        for corridor in layout.corridors
        if bend_room.index in (corridor.room_a_index, corridor.room_b_index)
    ]
    assert len(bend_links) >= 2
