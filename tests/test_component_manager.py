import pytest

from component_manager import ComponentManager


def test_room_component_updates_to_canonical_root():
    manager = ComponentManager()
    room_a = manager.new_component()
    room_b = manager.new_component()

    manager.register_room(room_a)
    manager.register_room(room_b)

    manager.union(room_a, room_b)

    assert manager.room_component(0) == room_a
    assert manager.room_component(1) == room_a


def test_union_ignores_negative_ids_and_returns_root():
    manager = ComponentManager()
    corridor = manager.new_component()

    manager.register_corridor(corridor)

    root = manager.union(-1, corridor)

    assert root == corridor
    assert manager.corridor_component(0) == corridor


def test_union_rejects_all_negative_inputs():
    manager = ComponentManager()

    with pytest.raises(ValueError):
        manager.union(-1, -5)


def test_component_summary_and_sizes_merge_members():
    manager = ComponentManager()
    room_a = manager.new_component()
    room_b = manager.new_component()
    corridor = manager.new_component()

    manager.register_room(room_a)
    manager.register_room(room_b)
    manager.register_corridor(corridor)

    manager.union(room_a, room_b, corridor)

    assert manager.component_summary() == {
        room_a: {"rooms": [0, 1], "corridors": [0]}
    }
    assert manager.component_sizes() == {room_a: 3}


def test_has_single_component_reflects_connectivity():
    manager = ComponentManager()
    room = manager.new_component()
    corridor = manager.new_component()

    manager.register_room(room)
    manager.register_corridor(corridor)

    assert not manager.has_single_component()

    manager.union(room, corridor)

    assert manager.has_single_component()


def test_components_equal_rejects_negative_ids():
    manager = ComponentManager()
    room = manager.new_component()
    other = manager.new_component()

    assert not manager.components_equal(-1, room)

    manager.union(room, other)

    assert manager.components_equal(room, other)

def test_register_negative_component_keeps_placeholder_state():
    manager = ComponentManager()

    root = manager.register_room(-1)

    assert root == -1
    assert manager.room_component(0) == -1
    assert manager.has_single_component()
