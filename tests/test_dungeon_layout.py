from geometry import TilePos
from models import Corridor, CorridorGeometry


def test_geometry_cross_coords_collects_unique_sorted(dungeon_layout):
    geometry = CorridorGeometry(
        tiles=(
            TilePos(1, 5),
            TilePos(2, 5),
            TilePos(3, 6),
            TilePos(4, 6),
        ),
        axis_index=0,
        port_axis_values=(1, 4),
    )

    result = dungeon_layout._geometry_cross_coords(geometry)

    assert result == (5, 6)


def test_would_create_long_parallel_detects_close_overlap(dungeon_layout):
    layout = dungeon_layout
    existing_geometry = CorridorGeometry(
        tiles=tuple(TilePos(x, 10) for x in range(2, 11)),
        axis_index=0,
        port_axis_values=(2, 10),
        cross_coords=(10,),
    )
    existing_corridor = Corridor(
        room_a_index=None,
        port_a_index=None,
        room_b_index=None,
        port_b_index=None,
        width=2,
        geometry=existing_geometry,
    )
    existing_component = layout.new_component_id()
    layout.register_corridor(existing_corridor, existing_component)

    candidate_geometry = CorridorGeometry(
        tiles=tuple(TilePos(x, 11) for x in range(4, 13)),
        axis_index=0,
        port_axis_values=(4, 12),
        cross_coords=(11,),
    )

    assert layout.would_create_long_parallel(candidate_geometry) is True


def test_would_create_long_parallel_respects_skip_indices(dungeon_layout):
    layout = dungeon_layout
    candidate_geometry = CorridorGeometry(
        tiles=tuple(TilePos(x, 5) for x in range(4, 13)),
        axis_index=0,
        port_axis_values=(4, 12),
        cross_coords=(5,),
    )

    # With no existing corridors, should be False.
    assert layout.would_create_long_parallel(candidate_geometry) is False

    existing_geometry = CorridorGeometry(
        tiles=tuple(TilePos(x, 5) for x in range(2, 9)),
        axis_index=0,
        port_axis_values=(2, 8),
        cross_coords=(5,),
    )
    corridor = Corridor(
        room_a_index=None,
        port_a_index=None,
        room_b_index=None,
        port_b_index=None,
        width=2,
        geometry=existing_geometry,
    )
    layout.register_corridor(corridor, layout.new_component_id())

    assert layout.would_create_long_parallel(candidate_geometry, skip_indices=[0]) is False
