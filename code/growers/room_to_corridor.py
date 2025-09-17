from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from geometry import TilePos
from models import Corridor, CorridorGeometry, PlacedRoom, RoomKind, WorldPort

from growers.base import (
    CandidateFinder,
    DungeonGrower,
    GeometryPlanner,
    GrowerApplier,
    GrowerStepResult,
)
from growers.port_requirement import PortRequirement

if TYPE_CHECKING:
    from grower_context import GrowerContext


@dataclass(frozen=True)
class RoomToCorridorCandidate:
    room_idx: int
    port_idx: int
    world_port: WorldPort


@dataclass(frozen=True)
class RoomToCorridorPlan:
    width: int
    branch_requirement_idx: int
    target_corridor_idx: int
    requirements: Tuple[PortRequirement, ...]
    requirement_mapping: Dict[str, int]
    port_mapping: Dict[int, int]
    junction_room: PlacedRoom


class RoomToCorridorCandidateFinder(CandidateFinder[RoomToCorridorCandidate, RoomToCorridorPlan]):
    def __init__(self) -> None:
        self._used_ports: Set[Tuple[int, int]] = set()

    def find_candidates(self, context: GrowerContext) -> Iterable[RoomToCorridorCandidate]:
        self._used_ports.clear()
        room_world_ports = [room.get_world_ports() for room in context.layout.placed_rooms]
        available_ports = context.list_available_ports(room_world_ports)
        random.shuffle(available_ports)

        def iterator() -> Iterator[RoomToCorridorCandidate]:
            for room_idx, port_idx, world_port in available_ports:
                key = (room_idx, port_idx)
                if key in self._used_ports:
                    continue
                yield RoomToCorridorCandidate(
                    room_idx=room_idx,
                    port_idx=port_idx,
                    world_port=world_port,
                )

        return iterator()

    def on_success(
        self,
        context: GrowerContext,
        candidate: RoomToCorridorCandidate,
        plan: RoomToCorridorPlan,
    ) -> None:
        self._used_ports.add((candidate.room_idx, candidate.port_idx))


class RoomToCorridorGeometryPlanner(GeometryPlanner[RoomToCorridorCandidate, RoomToCorridorPlan]):
    def __init__(self, fill_probability: float) -> None:
        self.fill_probability = fill_probability

    def plan(
        self,
        context: GrowerContext,
        candidate: RoomToCorridorCandidate,
    ) -> Optional[RoomToCorridorPlan]:
        width_options = list(candidate.world_port.widths)
        random.shuffle(width_options)

        viable_options: List[Tuple[int, CorridorGeometry, int, Tuple[TilePos, ...]]] = []
        existing_links = set(context.layout.room_corridor_links)
        for width in width_options:
            result = context.build_t_junction_geometry(
                candidate.room_idx,
                candidate.world_port,
                width,
            )
            if result is None:
                continue
            geometry, target_corridor_idx, junction_tiles = result
            if (candidate.room_idx, target_corridor_idx) in existing_links:
                continue
            viable_options.append((width, geometry, target_corridor_idx, junction_tiles))

        if not viable_options:
            return None
        if random.random() > self.fill_probability:
            return None

        width, geometry, target_corridor_idx, junction_tiles = random.choice(viable_options)
        return self._build_plan_for_option(
            context,
            candidate,
            width,
            geometry,
            target_corridor_idx,
            junction_tiles,
        )

    def _build_plan_for_option(
        self,
        context: GrowerContext,
        candidate: RoomToCorridorCandidate,
        width: int,
        geometry: CorridorGeometry,
        target_corridor_idx: int,
        junction_tiles: Tuple[TilePos, ...],
    ) -> Optional[RoomToCorridorPlan]:
        target_corridor = context.layout.corridors[target_corridor_idx]
        if geometry.axis_index is None:
            return None
        existing_axis_index = target_corridor.geometry.axis_index
        if existing_axis_index is None:
            return None

        requirements: List[PortRequirement] = []
        requirement_mapping: Dict[str, int] = {}

        def add_requirement(req: Optional[PortRequirement]) -> bool:
            if req is None:
                return False
            requirement_mapping[req.source] = len(requirements)
            requirements.append(req)
            return True

        if not add_requirement(
            context.build_port_requirement_from_segment(
                geometry,
                geometry.axis_index,
                "new_branch",
                expected_width=width,
                room_index=candidate.room_idx,
                port_index=candidate.port_idx,
                junction_tiles=junction_tiles,
            )
        ):
            return None

        seg_existing_a, seg_existing_b = context.split_existing_corridor_geometries(
            target_corridor,
            junction_tiles,
        )
        if seg_existing_a is None or seg_existing_b is None:
            return None

        if not add_requirement(
            context.build_port_requirement_from_segment(
                seg_existing_a,
                existing_axis_index,
                "existing_a",
                expected_width=target_corridor.width,
                corridor_idx=target_corridor_idx,
                corridor_end="a",
                junction_tiles=junction_tiles,
            )
        ):
            return None

        if not add_requirement(
            context.build_port_requirement_from_segment(
                seg_existing_b,
                existing_axis_index,
                "existing_b",
                expected_width=target_corridor.width,
                corridor_idx=target_corridor_idx,
                corridor_end="b",
                junction_tiles=junction_tiles,
            )
        ):
            return None

        placement = context.attempt_place_special_room(
            requirements,
            context.get_room_templates(RoomKind.T_JUNCTION),
            allowed_overlap_tiles=set(junction_tiles),
            allowed_overlap_corridors={target_corridor_idx},
        )
        if placement is None:
            print(
                "Failed to place T-junction room. Will print out grid and indicate the intended position of room."
            )
            print(requirements)
            context.layout.draw_to_grid()
            for x, y in junction_tiles:
                context.layout.grid[y][x] = "*"
            context.layout.mark_room_interior_on_grid(candidate.room_idx)
            context.layout.print_grid()
            raise RuntimeError("Unable to place a T-junction room with available templates.")

        placed_room, port_mapping, geometry_overrides = placement
        if geometry_overrides:
            for req_idx, geometry_override in geometry_overrides.items():
                requirements[req_idx] = replace(requirements[req_idx], geometry=geometry_override)
        mapping = dict(requirement_mapping)

        branch_idx = mapping.get("new_branch")
        if branch_idx is None:
            return None

        return RoomToCorridorPlan(
            width=width,
            branch_requirement_idx=branch_idx,
            target_corridor_idx=target_corridor_idx,
            requirements=tuple(requirements),
            requirement_mapping=mapping,
            port_mapping=dict(port_mapping),
            junction_room=placed_room,
        )


class RoomToCorridorApplier(GrowerApplier[RoomToCorridorCandidate, RoomToCorridorPlan]):
    def __init__(self) -> None:
        self._created = 0
        self._junction_rooms = 0

    def apply(
        self,
        context: GrowerContext,
        candidate: RoomToCorridorCandidate,
        plan: RoomToCorridorPlan,
    ) -> GrowerStepResult:
        component_id = context.layout.merge_components(
            context.layout.normalize_room_component(candidate.room_idx),
            context.layout.normalize_corridor_component(plan.target_corridor_idx),
        )
        context.layout.set_room_component(candidate.room_idx, component_id)
        context.layout.set_corridor_component(plan.target_corridor_idx, component_id)

        junction_room_index = len(context.layout.placed_rooms)
        context.layout.register_room(plan.junction_room, component_id)

        branch_requirement = plan.requirements[plan.branch_requirement_idx]
        branch_geometry = branch_requirement.geometry
        if branch_geometry is None:
            return GrowerStepResult(applied=False)
        branch_port_idx = plan.port_mapping[plan.branch_requirement_idx]
        new_corridor = Corridor(
            room_a_index=candidate.room_idx,
            port_a_index=candidate.port_idx,
            room_b_index=junction_room_index,
            port_b_index=branch_port_idx,
            width=plan.width,
            geometry=branch_geometry,
            component_id=component_id,
        )
        new_corridor_idx = context.layout.register_corridor(new_corridor, component_id)
        context.layout.placed_rooms[candidate.room_idx].connected_port_indices.add(candidate.port_idx)
        context.layout.placed_rooms[junction_room_index].connected_port_indices.add(branch_port_idx)
        context.layout.room_corridor_links.add((candidate.room_idx, new_corridor_idx))
        context.layout.room_corridor_links.add((candidate.room_idx, plan.target_corridor_idx))

        existing_assignments: Dict[str, Tuple[PortRequirement, int]] = {}
        for suffix in ("a", "b"):
            key = f"existing_{suffix}"
            req_idx = plan.requirement_mapping.get(key)
            if req_idx is None:
                continue
            requirement = plan.requirements[req_idx]
            junction_port_idx = plan.port_mapping[req_idx]
            existing_assignments[suffix] = (requirement, junction_port_idx)
            context.layout.placed_rooms[junction_room_index].connected_port_indices.add(junction_port_idx)

        linked_indices = context.apply_existing_corridor_segments(
            plan.target_corridor_idx,
            existing_assignments,
            junction_room_index,
            component_id,
        )

        for idx in linked_indices:
            context.layout.room_corridor_links.add((candidate.room_idx, idx))

        context.validate_room_corridor_clearance(junction_room_index)

        self._created += 1
        self._junction_rooms += 1
        return GrowerStepResult(applied=True)

    def finalize(self, context: GrowerContext) -> int:
        print(
            "Room-to-corridor grower: created"
            f" {self._created} corridor-to-corridor links and placed"
            f" {self._junction_rooms} T-junction rooms."
        )
        return self._created


def run_room_to_corridor_grower(
    context: GrowerContext, fill_probability: float
) -> int:
    if not context.layout.corridors:
        print("Room-to-corridor grower: skipped - no existing corridors to join.")
        return 0
    grower = DungeonGrower(
        name="room_to_corridor",
        candidate_finder=RoomToCorridorCandidateFinder(),
        geometry_planner=RoomToCorridorGeometryPlanner(fill_probability),
        applier=RoomToCorridorApplier(),
    )
    return grower.run(context)
