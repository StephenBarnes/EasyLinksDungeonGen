from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from geometry import TilePos, VALID_ROTATIONS
from models import Corridor, CorridorGeometry, PlacedRoom, RoomKind, RoomTemplate, WorldPort

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
class BentRoomToCorridorCandidate:
    room_idx: int
    port_idx: int
    world_port: WorldPort


@dataclass(frozen=True)
class BentRoomToCorridorPlan:
    width: int
    bend_room: PlacedRoom
    bend_room_port_idx: int
    bend_branch_port_idx: int
    room_to_bend_geometry: CorridorGeometry
    branch_geometry: CorridorGeometry
    target_corridor_idx: int
    branch_requirement_idx: int
    requirements: Tuple[PortRequirement, ...]
    requirement_mapping: Dict[str, int]
    port_mapping: Dict[int, int]
    junction_room: PlacedRoom


class BentRoomToCorridorCandidateFinder(
    CandidateFinder[BentRoomToCorridorCandidate, BentRoomToCorridorPlan]
):
    def __init__(self) -> None:
        self._used_ports: Set[Tuple[int, int]] = set()

    def find_candidates(self, context: GrowerContext) -> Iterable[BentRoomToCorridorCandidate]:
        self._used_ports.clear()
        room_world_ports = [room.get_world_ports() for room in context.layout.placed_rooms]
        available_ports = context.list_available_ports(room_world_ports)
        random.shuffle(available_ports)

        def iterator() -> Iterator[BentRoomToCorridorCandidate]:
            for room_idx, port_idx, world_port in available_ports:
                key = (room_idx, port_idx)
                if key in self._used_ports:
                    continue
                yield BentRoomToCorridorCandidate(
                    room_idx=room_idx,
                    port_idx=port_idx,
                    world_port=world_port,
                )

        return iterator()

    def on_success(
        self,
        context: GrowerContext,
        candidate: BentRoomToCorridorCandidate,
        plan: BentRoomToCorridorPlan,
    ) -> None:
        self._used_ports.add((candidate.room_idx, candidate.port_idx))


class BentRoomToCorridorGeometryPlanner(
    GeometryPlanner[BentRoomToCorridorCandidate, BentRoomToCorridorPlan]
):
    def __init__(self, max_room_distance: int = 8, fill_probability: float = 1.0) -> None:
        self.max_room_distance = max_room_distance
        self.fill_probability = fill_probability

    def plan(
        self,
        context: GrowerContext,
        candidate: BentRoomToCorridorCandidate,
    ) -> Optional[BentRoomToCorridorPlan]:
        if random.random() > self.fill_probability:
            return None
        width_options = list(candidate.world_port.widths)
        if not width_options:
            return None
        random.shuffle(width_options)

        bend_templates = context.weighted_templates(RoomKind.BEND)
        if not bend_templates:
            return None

        existing_links = set(context.layout.room_corridor_links)

        for width in width_options:
            plan = self._build_plan_for_width(
                context,
                candidate,
                width,
                bend_templates,
                existing_links,
            )
            if plan is not None:
                return plan
        return None

    def _build_plan_for_width(
        self,
        context: GrowerContext,
        candidate: BentRoomToCorridorCandidate,
        width: int,
        bend_templates: Sequence[RoomTemplate],
        existing_links: Set[Tuple[int, int]],
    ) -> Optional[BentRoomToCorridorPlan]:
        room_port = candidate.world_port
        direction = room_port.direction
        axis_index = 0 if direction.dx != 0 else 1
        axis_dir = direction.dx if axis_index == 0 else direction.dy
        if axis_dir == 0:
            return None

        candidate_exit_axis = context.port_exit_axis_value(room_port, axis_index)
        bend_room_index = len(context.layout.placed_rooms)

        distances = list(range(2, self.max_room_distance + 1))
        random.shuffle(distances)

        for template in bend_templates:
            for rotation in VALID_ROTATIONS:
                temp_room = PlacedRoom(template, 0, 0, rotation)
                rotated_ports = temp_room.get_world_ports()
                matching_indices = [
                    (idx, port)
                    for idx, port in enumerate(rotated_ports)
                    if port.direction == direction.opposite() and width in port.widths
                ]
                if not matching_indices:
                    continue

                branch_indices = [
                    (idx, port)
                    for idx, port in enumerate(rotated_ports)
                    if idx not in {idx for idx, _ in matching_indices}
                    and port.direction.dot(direction) == 0
                    and width in port.widths
                ]
                if not branch_indices:
                    continue

                for match_idx, match_port in matching_indices:
                    for branch_idx, branch_port in branch_indices:
                        plan = self._try_bend_position(
                            context,
                            candidate,
                            width,
                            temp_room,
                            match_idx,
                            match_port,
                            branch_idx,
                            branch_port,
                            axis_index,
                            axis_dir,
                            candidate_exit_axis,
                            bend_room_index,
                            distances,
                            existing_links,
                        )
                        if plan is not None:
                            return plan
        return None

    def _try_bend_position(
        self,
        context: GrowerContext,
        candidate: BentRoomToCorridorCandidate,
        width: int,
        bend_room_template: PlacedRoom,
        match_idx: int,
        match_port: WorldPort,
        branch_idx: int,
        branch_port: WorldPort,
        axis_index: int,
        axis_dir: int,
        candidate_exit_axis: int,
        bend_room_index: int,
        distances: Sequence[int],
        existing_links: Set[Tuple[int, int]],
    ) -> Optional[BentRoomToCorridorPlan]:
        room_port = candidate.world_port

        def aligned_translation(value: float) -> Optional[int]:
            if not math.isclose(value, round(value), abs_tol=1e-6):
                return None
            return int(round(value))

        # Align the bend room on the perpendicular axis so that the connecting ports share a cross coordinate.
        if axis_index == 0:
            perp_translation_value = room_port.pos[1] - match_port.pos[1]
        else:
            perp_translation_value = room_port.pos[0] - match_port.pos[0]
        perp_translation = aligned_translation(perp_translation_value)
        if perp_translation is None:
            return None

        match_port_exit_base = context.port_exit_axis_value(match_port, axis_index)

        for distance in distances:
            desired_exit = candidate_exit_axis + axis_dir * distance
            translation_axis_value = desired_exit - match_port_exit_base
            axis_translation = aligned_translation(translation_axis_value)
            if axis_translation is None:
                continue

            if axis_index == 0:
                tx, ty = axis_translation, perp_translation
            else:
                tx, ty = perp_translation, axis_translation

            placed_bend = PlacedRoom(
                bend_room_template.template,
                tx,
                ty,
                bend_room_template.rotation,
            )
            if not context.layout.is_valid_placement(placed_bend):
                continue

            bounds = placed_bend.get_bounds()
            extra_room_tiles: Dict[TilePos, int] = {}
            overlaps_corridor = False
            for ty_tile in range(bounds.y, bounds.max_y):
                for tx_tile in range(bounds.x, bounds.max_x):
                    tile = TilePos(tx_tile, ty_tile)
                    if context.layout.spatial_index.has_corridor_at(tile):
                        overlaps_corridor = True
                        break
                    extra_room_tiles[tile] = bend_room_index
                if overlaps_corridor:
                    break
            if overlaps_corridor:
                continue

            bend_world_ports = placed_bend.get_world_ports()
            bend_match_port = bend_world_ports[match_idx]
            bend_branch_port = bend_world_ports[branch_idx]
            if width not in bend_match_port.widths or width not in bend_branch_port.widths:
                continue

            geometry_room_to_bend = context.build_corridor_geometry(
                candidate.room_idx,
                room_port,
                bend_room_index,
                bend_match_port,
                width,
                extra_room_tiles,
            )
            if geometry_room_to_bend is None:
                continue
            if any(context.layout.spatial_index.has_corridor_at(tile) for tile in geometry_room_to_bend.tiles):
                continue

            branch_plan = self._build_branch_plan(
                context,
                candidate,
                width,
                bend_room_index,
                branch_idx,
                bend_branch_port,
                geometry_room_to_bend,
                extra_room_tiles,
                existing_links,
            )
            if branch_plan is None:
                continue

            return BentRoomToCorridorPlan(
                width=width,
                bend_room=placed_bend,
                bend_room_port_idx=match_idx,
                bend_branch_port_idx=branch_idx,
                room_to_bend_geometry=geometry_room_to_bend,
                branch_geometry=branch_plan.branch_geometry,
                target_corridor_idx=branch_plan.target_corridor_idx,
                branch_requirement_idx=branch_plan.branch_requirement_idx,
                requirements=branch_plan.requirements,
                requirement_mapping=branch_plan.requirement_mapping,
                port_mapping=branch_plan.port_mapping,
                junction_room=branch_plan.junction_room,
            )
        return None

    @dataclass(frozen=True)
    class _BranchPlan:
        branch_geometry: CorridorGeometry
        target_corridor_idx: int
        branch_requirement_idx: int
        requirements: Tuple[PortRequirement, ...]
        requirement_mapping: Dict[str, int]
        port_mapping: Dict[int, int]
        junction_room: PlacedRoom

    @staticmethod
    def _boundary_tiles(segment: CorridorGeometry, axis_index: Optional[int]) -> Set[TilePos]:
        if axis_index is None:
            return set()
        start_axis, end_axis = segment.port_axis_values
        if start_axis == end_axis:
            return set()
        sign = 1 if end_axis > start_axis else -1
        boundary_axis = end_axis - sign
        return {
            tile
            for tile in segment.tiles
            if (tile[axis_index] == boundary_axis)
        }

    @classmethod
    def _compute_allowed_overlap_tiles(
        cls,
        branch_geometry: CorridorGeometry,
        branch_axis: int,
        seg_existing_a: CorridorGeometry,
        seg_existing_b: CorridorGeometry,
        corridor_axis: int,
        junction_tiles: Iterable[TilePos],
    ) -> Set[TilePos]:
        allowed: Set[TilePos] = set(junction_tiles)
        allowed.update(cls._boundary_tiles(branch_geometry, branch_axis))
        allowed.update(cls._boundary_tiles(seg_existing_a, corridor_axis))
        allowed.update(cls._boundary_tiles(seg_existing_b, corridor_axis))
        return allowed

    def _build_branch_plan(
        self,
        context: GrowerContext,
        candidate: BentRoomToCorridorCandidate,
        width: int,
        bend_room_index: int,
        bend_branch_port_idx: int,
        bend_branch_port: WorldPort,
        geometry_room_to_bend: CorridorGeometry,
        extra_room_tiles: Dict[TilePos, int],
        existing_links: Set[Tuple[int, int]],
    ) -> Optional[_BranchPlan]:
        branch_result = context.build_t_junction_geometry(
            bend_room_index,
            bend_branch_port,
            width,
        )
        if branch_result is None:
            return None
        branch_geometry, target_corridor_idx, junction_tiles = branch_result

        if (candidate.room_idx, target_corridor_idx) in existing_links:
            return None
        if not context.layout.should_allow_connection(
            ("room", candidate.room_idx),
            ("corridor", target_corridor_idx),
        ):
            return None

        room_to_bend_tiles = set(geometry_room_to_bend.tiles)
        if any(tile in room_to_bend_tiles for tile in branch_geometry.tiles):
            return None
        bend_tiles = set(extra_room_tiles.keys())
        if any(tile in bend_tiles for tile in branch_geometry.tiles):
            return None

        branch_axis = branch_geometry.axis_index
        if branch_axis is None:
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
                branch_geometry,
                branch_axis,
                "new_branch",
                expected_width=width,
                room_index=bend_room_index,
                port_index=bend_branch_port_idx,
                junction_tiles=junction_tiles,
            )
        ):
            return None

        corridor = context.layout.corridors[target_corridor_idx]
        corridor_axis = corridor.geometry.axis_index
        if corridor_axis is None:
            return None

        seg_existing_a, seg_existing_b = context.split_existing_corridor_geometries(
            corridor,
            junction_tiles,
        )
        if seg_existing_a is None or seg_existing_b is None:
            return None

        if not add_requirement(
            context.build_port_requirement_from_segment(
                seg_existing_a,
                corridor_axis,
                "existing_a",
                expected_width=corridor.width,
                corridor_idx=target_corridor_idx,
                corridor_end="a",
                junction_tiles=junction_tiles,
            )
        ):
            return None

        if not add_requirement(
            context.build_port_requirement_from_segment(
                seg_existing_b,
                corridor_axis,
                "existing_b",
                expected_width=corridor.width,
                corridor_idx=target_corridor_idx,
                corridor_end="b",
                junction_tiles=junction_tiles,
            )
        ):
            return None

        allowed_overlap_tiles = self._compute_allowed_overlap_tiles(
            branch_geometry,
            branch_axis,
            seg_existing_a,
            seg_existing_b,
            corridor_axis,
            junction_tiles,
        )

        placement = context.attempt_place_special_room(
            requirements,
            context.get_room_templates(RoomKind.T_JUNCTION),
            RoomKind.T_JUNCTION,
            allowed_overlap_tiles=allowed_overlap_tiles,
            allowed_overlap_corridors={target_corridor_idx},
        )
        if placement is None:
            return None

        placed_room, port_mapping, geometry_overrides = placement
        if geometry_overrides:
            for req_idx, geometry_override in geometry_overrides.items():
                requirements[req_idx] = replace(requirements[req_idx], geometry=geometry_override)

        branch_requirement_idx = requirement_mapping.get("new_branch")
        if branch_requirement_idx is None:
            return None
        branch_requirement = requirements[branch_requirement_idx]
        branch_geometry_final = branch_requirement.geometry
        if branch_geometry_final is None:
            return None

        return BentRoomToCorridorGeometryPlanner._BranchPlan(
            branch_geometry=branch_geometry_final,
            target_corridor_idx=target_corridor_idx,
            branch_requirement_idx=branch_requirement_idx,
            requirements=tuple(requirements),
            requirement_mapping=dict(requirement_mapping),
            port_mapping=dict(port_mapping),
            junction_room=placed_room,
        )


class BentRoomToCorridorApplier(
    GrowerApplier[BentRoomToCorridorCandidate, BentRoomToCorridorPlan]
):
    def __init__(self, *, stop_after_first: bool) -> None:
        self._created = 0
        self._bend_rooms = 0
        self._junction_rooms = 0
        self._stop_after_first = stop_after_first

    def apply(
        self,
        context: GrowerContext,
        candidate: BentRoomToCorridorCandidate,
        plan: BentRoomToCorridorPlan,
    ) -> GrowerStepResult:
        component_id = context.layout.merge_components(
            context.layout.normalize_room_component(candidate.room_idx),
            context.layout.normalize_corridor_component(plan.target_corridor_idx),
        )
        context.layout.set_room_component(candidate.room_idx, component_id)
        context.layout.set_corridor_component(plan.target_corridor_idx, component_id)

        bend_room_index = len(context.layout.placed_rooms)
        context.layout.register_room(plan.bend_room, component_id)
        context.layout.set_room_component(bend_room_index, component_id)

        corridor_room_to_bend = Corridor(
            room_a_index=candidate.room_idx,
            port_a_index=candidate.port_idx,
            room_b_index=bend_room_index,
            port_b_index=plan.bend_room_port_idx,
            width=plan.width,
            geometry=plan.room_to_bend_geometry,
            component_id=component_id,
        )
        corridor_room_to_bend_idx = context.layout.register_corridor(
            corridor_room_to_bend,
            component_id,
        )
        context.layout.placed_rooms[candidate.room_idx].connected_port_indices.add(candidate.port_idx)
        context.layout.placed_rooms[bend_room_index].connected_port_indices.add(plan.bend_room_port_idx)

        junction_room_index = len(context.layout.placed_rooms)
        context.layout.register_room(plan.junction_room, component_id)
        context.layout.set_room_component(junction_room_index, component_id)

        branch_port_idx = plan.port_mapping.get(plan.branch_requirement_idx)
        branch_geometry = plan.requirements[plan.branch_requirement_idx].geometry
        if branch_port_idx is None or branch_geometry is None:
            return GrowerStepResult(applied=False)

        corridor_bend_to_junction = Corridor(
            room_a_index=bend_room_index,
            port_a_index=plan.bend_branch_port_idx,
            room_b_index=junction_room_index,
            port_b_index=branch_port_idx,
            width=plan.width,
            geometry=branch_geometry,
            component_id=component_id,
        )
        corridor_bend_to_junction_idx = context.layout.register_corridor(
            corridor_bend_to_junction,
            component_id,
        )
        context.layout.placed_rooms[bend_room_index].connected_port_indices.add(plan.bend_branch_port_idx)
        context.layout.placed_rooms[junction_room_index].connected_port_indices.add(branch_port_idx)

        existing_assignments: Dict[str, Tuple[PortRequirement, int]] = {}
        for suffix in ("a", "b"):
            key = f"existing_{suffix}"
            req_idx = plan.requirement_mapping.get(key)
            if req_idx is None:
                continue
            requirement = plan.requirements[req_idx]
            junction_port_idx = plan.port_mapping.get(req_idx)
            if junction_port_idx is None:
                continue
            existing_assignments[suffix] = (requirement, junction_port_idx)
            context.layout.placed_rooms[junction_room_index].connected_port_indices.add(junction_port_idx)

        linked_indices = context.apply_existing_corridor_segments(
            plan.target_corridor_idx,
            existing_assignments,
            junction_room_index,
            component_id,
        )

        context.layout.room_corridor_links.add((candidate.room_idx, plan.target_corridor_idx))
        for idx in linked_indices:
            context.layout.room_corridor_links.add((candidate.room_idx, idx))

        context.layout.room_corridor_links.add((candidate.room_idx, corridor_room_to_bend_idx))
        context.layout.room_corridor_links.add((bend_room_index, corridor_room_to_bend_idx))
        context.layout.room_corridor_links.add((bend_room_index, corridor_bend_to_junction_idx))

        context.validate_room_corridor_clearance(junction_room_index)

        self._created += 1
        self._bend_rooms += 1
        self._junction_rooms += 1
        stop = self._stop_after_first or context.layout.component_manager.has_single_component()
        return GrowerStepResult(applied=True, stop=stop)

    def finalize(self, context: GrowerContext) -> int:
        if self._created == 0:
            print("Bent-room-to-corridor grower: no placements succeeded.")
        else:
            print(
                "Bent-room-to-corridor grower: created"
                f" {self._created} room-bend-corridor links, placed"
                f" {self._bend_rooms} bend rooms and"
                f" {self._junction_rooms} T-junction rooms."
            )
        return self._created


def _run_bent_room_to_corridor_grower(
    context: GrowerContext,
    fill_probability: float,
    *,
    stop_after_first: bool,
    name: str,
) -> int:
    if not context.layout.corridors:
        print("Bent-room-to-corridor grower: skipped - no corridors available to join.")
        return 0
    if not context.get_room_templates(RoomKind.BEND):
        print("Bent-room-to-corridor grower: skipped - no bend room templates available.")
        return 0
    if not context.get_room_templates(RoomKind.T_JUNCTION):
        print("Bent-room-to-corridor grower: skipped - no T-junction templates available.")
        return 0
    grower = DungeonGrower(
        name=name,
        candidate_finder=BentRoomToCorridorCandidateFinder(),
        geometry_planner=BentRoomToCorridorGeometryPlanner(fill_probability=fill_probability),
        applier=BentRoomToCorridorApplier(stop_after_first=stop_after_first),
    )
    return grower.run(context)


def run_bent_room_to_corridor_grower(
    context: GrowerContext,
    stop_after_first: bool,
    fill_probability: float = 1.0,
) -> int:
    return _run_bent_room_to_corridor_grower(
        context,
        fill_probability,
        stop_after_first=stop_after_first,
        name="bent_room_to_corridor",
    )
