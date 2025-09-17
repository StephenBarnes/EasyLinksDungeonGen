from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

from geometry import TilePos
from models import Corridor, CorridorGeometry, PlacedRoom, WorldPort

from growers.base import (
    CandidateFinder,
    DungeonGrower,
    GeometryPlanner,
    GrowerApplier,
    GrowerStepResult,
)
from growers.port_requirement import PortRequirement

if TYPE_CHECKING:
    from dungeon_generator import DungeonGenerator


@dataclass(frozen=True)
class RoomToRoomCandidate:
    room_a_idx: int
    port_a_idx: int
    world_port_a: WorldPort
    room_b_idx: int
    port_b_idx: int
    world_port_b: WorldPort
    common_widths: Tuple[int, ...]
    room_pair: Tuple[int, int]


@dataclass(frozen=True)
class DirectRoomConnectionPlan:
    width: int
    geometry: CorridorGeometry


@dataclass(frozen=True)
class IntersectionRoomConnectionPlan:
    width: int
    requirements: Tuple[PortRequirement, ...]
    requirement_mapping: Dict[str, int]
    port_mapping: Dict[int, int]
    junction_room: PlacedRoom
    existing_corridor_idx: int


RoomToRoomPlan = Union[DirectRoomConnectionPlan, IntersectionRoomConnectionPlan]


class RoomToRoomCandidateFinder(CandidateFinder[RoomToRoomCandidate, RoomToRoomPlan]):
    def __init__(self) -> None:
        self._used_ports: Set[Tuple[int, int]] = set()
        self._connected_pairs: Set[Tuple[int, int]] = set()

    def find_candidates(self, generator: DungeonGenerator) -> Iterable[RoomToRoomCandidate]:
        self._used_ports.clear()
        self._connected_pairs = {
            tuple(sorted((corridor.room_a_index, corridor.room_b_index))) # type: ignore
            for corridor in generator.corridors
            if corridor.room_b_index is not None
        }
        room_world_ports = [room.get_world_ports() for room in generator.placed_rooms]
        available_ports = generator._list_available_ports(room_world_ports)
        random.shuffle(available_ports)

        def iterator() -> Iterator[RoomToRoomCandidate]:
            for i, (room_a_idx, port_a_idx, world_port_a) in enumerate(available_ports):
                key_a = (room_a_idx, port_a_idx)
                if key_a in self._used_ports:
                    continue
                candidate_indices = list(range(i + 1, len(available_ports)))
                random.shuffle(candidate_indices)
                for j in candidate_indices:
                    if key_a in self._used_ports:
                        break
                    room_b_idx, port_b_idx, world_port_b = available_ports[j]
                    key_b = (room_b_idx, port_b_idx)
                    if key_b in self._used_ports:
                        continue
                    if room_a_idx == room_b_idx:
                        continue
                    room_pair = tuple(sorted((room_a_idx, room_b_idx)))
                    if room_pair in self._connected_pairs:
                        continue
                    common_widths = world_port_a.widths & world_port_b.widths
                    if not common_widths:
                        continue
                    yield RoomToRoomCandidate(
                        room_a_idx=room_a_idx,
                        port_a_idx=port_a_idx,
                        world_port_a=world_port_a,
                        room_b_idx=room_b_idx,
                        port_b_idx=port_b_idx,
                        world_port_b=world_port_b,
                        common_widths=tuple(sorted(common_widths)),
                        room_pair=room_pair, # type: ignore
                    )

        return iterator()

    def on_success(
        self,
        generator: DungeonGenerator,
        candidate: RoomToRoomCandidate,
        plan: RoomToRoomPlan,
    ) -> None:
        self._used_ports.add((candidate.room_a_idx, candidate.port_a_idx))
        self._used_ports.add((candidate.room_b_idx, candidate.port_b_idx))
        self._connected_pairs.add(candidate.room_pair)


class RoomToRoomGeometryPlanner(GeometryPlanner[RoomToRoomCandidate, RoomToRoomPlan]):
    def plan(
        self,
        generator: DungeonGenerator,
        candidate: RoomToRoomCandidate,
    ) -> Optional[RoomToRoomPlan]:
        width_options = list(candidate.common_widths)
        random.shuffle(width_options)
        for width in width_options:
            geometry = generator._build_corridor_geometry(
                candidate.room_a_idx,
                candidate.world_port_a,
                candidate.room_b_idx,
                candidate.world_port_b,
                width,
                None,
            )
            if geometry is None:
                continue
            plan = self._build_plan_for_geometry(generator, candidate, width, geometry)
            if plan is not None:
                return plan
        return None

    def _build_plan_for_geometry(
        self,
        generator: DungeonGenerator,
        candidate: RoomToRoomCandidate,
        width: int,
        geometry: CorridorGeometry,
    ) -> Optional[RoomToRoomPlan]:
        overlap_map: Dict[int, List[TilePos]] = {}
        for tile in geometry.tiles:
            for existing_idx in generator.spatial_index.get_corridors_at(tile):
                overlap_map.setdefault(existing_idx, []).append(tile)

        if not overlap_map:
            return DirectRoomConnectionPlan(width=width, geometry=geometry)

        if len(overlap_map) != 1:
            return None

        existing_idx, overlap_tiles = next(iter(overlap_map.items()))
        existing_corridor = generator.corridors[existing_idx]
        if existing_corridor.geometry.axis_index is None or geometry.axis_index is None:
            return None
        if existing_corridor.geometry.axis_index == geometry.axis_index:
            return None

        intersection_axis_new = overlap_tiles[0][geometry.axis_index]
        cross_coords_new = geometry.cross_coords or generator._corridor_cross_from_geometry(
            geometry, geometry.axis_index
        )
        seg_a = generator._build_segment_geometry(
            geometry.axis_index,
            geometry.port_axis_values[0],
            intersection_axis_new,
            cross_coords_new,
        )
        seg_b = generator._build_segment_geometry(
            geometry.axis_index,
            geometry.port_axis_values[1],
            intersection_axis_new,
            cross_coords_new,
        )
        if seg_a is None or seg_b is None:
            return None

        existing_axis_index = existing_corridor.geometry.axis_index
        if existing_axis_index is None:
            return None
        existing_cross_coords = existing_corridor.geometry.cross_coords or generator._corridor_cross_from_geometry(
            existing_corridor.geometry, existing_axis_index
        )

        intersection_tiles: Set[TilePos] = set()
        for new_cross in cross_coords_new:
            for existing_cross in existing_cross_coords:
                if geometry.axis_index == 0:
                    tile = TilePos(existing_cross, new_cross)
                else:
                    tile = TilePos(new_cross, existing_cross)
                intersection_tiles.add(tile)

        seg_existing_a, seg_existing_b = generator._split_existing_corridor_geometries(
            existing_corridor,
            intersection_tiles,
        )
        if seg_existing_a is None or seg_existing_b is None:
            return None

        requirements: List[PortRequirement] = []
        requirement_indices: Dict[str, int] = {}

        def add_requirement(req: Optional[PortRequirement]) -> bool:
            if req is None:
                return False
            requirement_indices[req.source] = len(requirements)
            requirements.append(req)
            return True

        if not add_requirement(
            generator._build_port_requirement_from_segment(
                seg_a,
                geometry.axis_index,
                "new_a",
                expected_width=width,
                room_index=candidate.room_a_idx,
                port_index=candidate.port_a_idx,
                junction_tiles=intersection_tiles,
            )
        ):
            return None

        if not add_requirement(
            generator._build_port_requirement_from_segment(
                seg_b,
                geometry.axis_index,
                "new_b",
                expected_width=width,
                room_index=candidate.room_b_idx,
                port_index=candidate.port_b_idx,
                junction_tiles=intersection_tiles,
            )
        ):
            return None

        if not add_requirement(
            generator._build_port_requirement_from_segment(
                seg_existing_a,
                existing_axis_index,
                "existing_a",
                expected_width=existing_corridor.width,
                corridor_idx=existing_idx,
                corridor_end="a",
                junction_tiles=intersection_tiles,
            )
        ):
            return None

        if not add_requirement(
            generator._build_port_requirement_from_segment(
                seg_existing_b,
                existing_axis_index,
                "existing_b",
                expected_width=existing_corridor.width,
                corridor_idx=existing_idx,
                corridor_end="b",
                junction_tiles=intersection_tiles,
            )
        ):
            return None

        placement = generator._attempt_place_special_room(
            requirements,
            generator.four_way_room_templates,
            allowed_overlap_tiles=set(intersection_tiles),
            allowed_overlap_corridors={existing_idx},
        )
        if placement is None:
            return None

        placed_room, port_mapping, geometry_overrides = placement
        if geometry_overrides:
            for req_idx, geometry_override in geometry_overrides.items():
                requirements[req_idx] = replace(requirements[req_idx], geometry=geometry_override)
        mapping = dict(requirement_indices)
        return IntersectionRoomConnectionPlan(
            width=width,
            requirements=tuple(requirements),
            requirement_mapping=mapping,
            port_mapping=dict(port_mapping),
            junction_room=placed_room,
            existing_corridor_idx=existing_idx,
        )


class RoomToRoomApplier(GrowerApplier[RoomToRoomCandidate, RoomToRoomPlan]):
    def __init__(self) -> None:
        self._corridor_delta = 0
        self._junction_rooms_created = 0

    def apply(
        self,
        generator: DungeonGenerator,
        candidate: RoomToRoomCandidate,
        plan: RoomToRoomPlan,
    ) -> GrowerStepResult:
        before = len(generator.corridors)

        if isinstance(plan, DirectRoomConnectionPlan):
            component_id = generator._merge_components(
                generator._normalize_room_component(candidate.room_a_idx),
                generator._normalize_room_component(candidate.room_b_idx),
            )
            generator._set_room_component(candidate.room_a_idx, component_id)
            generator._set_room_component(candidate.room_b_idx, component_id)

            corridor = Corridor(
                room_a_index=candidate.room_a_idx,
                port_a_index=candidate.port_a_idx,
                room_b_index=candidate.room_b_idx,
                port_b_index=candidate.port_b_idx,
                width=plan.width,
                geometry=plan.geometry,
                component_id=component_id,
            )
            generator._register_corridor(corridor, component_id)
            generator.placed_rooms[candidate.room_a_idx].connected_port_indices.add(candidate.port_a_idx)
            generator.placed_rooms[candidate.room_b_idx].connected_port_indices.add(candidate.port_b_idx)
        else:
            component_id = generator._merge_components(
                generator._normalize_room_component(candidate.room_a_idx),
                generator._normalize_room_component(candidate.room_b_idx),
                generator._normalize_corridor_component(plan.existing_corridor_idx),
            )
            generator._set_room_component(candidate.room_a_idx, component_id)
            generator._set_room_component(candidate.room_b_idx, component_id)
            generator._set_corridor_component(plan.existing_corridor_idx, component_id)

            junction_room_index = len(generator.placed_rooms)
            generator._register_room(plan.junction_room, component_id)

            for key, source_room_idx, source_port_idx in (
                ("new_a", candidate.room_a_idx, candidate.port_a_idx),
                ("new_b", candidate.room_b_idx, candidate.port_b_idx),
            ):
                req_idx = plan.requirement_mapping.get(key)
                if req_idx is None:
                    continue
                requirement = plan.requirements[req_idx]
                geometry_segment = requirement.geometry
                if geometry_segment is None:
                    continue
                junction_port_idx = plan.port_mapping[req_idx]
                corridor = Corridor(
                    room_a_index=source_room_idx,
                    port_a_index=source_port_idx,
                    room_b_index=junction_room_index,
                    port_b_index=junction_port_idx,
                    width=plan.width,
                    geometry=geometry_segment,
                    component_id=component_id,
                )
                generator._register_corridor(corridor, component_id)
                generator.placed_rooms[source_room_idx].connected_port_indices.add(source_port_idx)
                generator.placed_rooms[junction_room_index].connected_port_indices.add(junction_port_idx)

            existing_assignments: Dict[str, Tuple[PortRequirement, int]] = {}
            for suffix in ("a", "b"):
                key = f"existing_{suffix}"
                req_idx = plan.requirement_mapping.get(key)
                if req_idx is None:
                    continue
                requirement = plan.requirements[req_idx]
                junction_port_idx = plan.port_mapping[req_idx]
                existing_assignments[suffix] = (requirement, junction_port_idx)
                generator.placed_rooms[junction_room_index].connected_port_indices.add(junction_port_idx)

            generator._apply_existing_corridor_segments(
                plan.existing_corridor_idx,
                existing_assignments,
                junction_room_index,
                component_id,
            )

            generator.placed_rooms[candidate.room_a_idx].connected_port_indices.add(candidate.port_a_idx)
            generator.placed_rooms[candidate.room_b_idx].connected_port_indices.add(candidate.port_b_idx)
            self._junction_rooms_created += 1

        delta = len(generator.corridors) - before
        if delta > 0:
            self._corridor_delta += delta
        return GrowerStepResult(applied=True)

    def finalize(self, generator: DungeonGenerator) -> int:
        print(
            "Room-to-room grower: created"
            f" {self._corridor_delta} straight corridors and placed"
            f" {self._junction_rooms_created} four-way rooms."
        )
        return self._corridor_delta


def run_room_to_room_grower(generator: DungeonGenerator) -> int:
    grower = DungeonGrower(
        name="room_to_room",
        candidate_finder=RoomToRoomCandidateFinder(),
        geometry_planner=RoomToRoomGeometryPlanner(),
        applier=RoomToRoomApplier(),
    )
    return grower.run(generator)
