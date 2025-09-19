from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from geometry import TilePos
from models import Corridor, CorridorGeometry, PlacedRoom, RoomKind, RoomTemplate

from growers.base import (
    CandidateDependencies,
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
class CorridorSlice:
    axis_value: int
    tiles: Tuple[TilePos, ...]


@dataclass(frozen=True)
class ThroughCorridorCandidate:
    corridor_idx: int
    axis_index: int
    slices: Tuple[CorridorSlice, ...]


@dataclass(frozen=True)
class ThroughCorridorPlan:
    corridor_idx: int
    requirements: Tuple[PortRequirement, ...]
    requirement_mapping: Dict[str, int]
    port_mapping: Dict[int, int]
    junction_room: PlacedRoom


class ThroughCorridorCandidateFinder(
    CandidateFinder[ThroughCorridorCandidate, ThroughCorridorPlan]
):
    @staticmethod
    def _group_tiles_by_axis(
        geometry: CorridorGeometry, axis_index: int
    ) -> Tuple[CorridorSlice, ...]:
        grouped: Dict[int, List[TilePos]] = {}
        for tile in geometry.tiles:
            grouped.setdefault(tile[axis_index], []).append(tile)
        direction = 1 if geometry.port_axis_values[1] >= geometry.port_axis_values[0] else -1
        axis_values = sorted(grouped, reverse=(direction < 0))
        slices: List[CorridorSlice] = []
        for axis_value in axis_values:
            tiles = tuple(sorted(grouped[axis_value], key=lambda t: (t[1 - axis_index], t[axis_index])))
            slices.append(CorridorSlice(axis_value=axis_value, tiles=tiles))
        return tuple(slices)

    def find_candidates(self, context: GrowerContext) -> Iterable[ThroughCorridorCandidate]:
        max_length = context.config.corridor_length_for_split
        candidates: List[ThroughCorridorCandidate] = []
        for idx, corridor in enumerate(context.layout.corridors):
            geometry = corridor.geometry
            axis_index = geometry.axis_index
            if axis_index is None:
                continue
            if corridor.room_a_index is None or corridor.room_b_index is None:
                continue
            corridor_length = abs(geometry.port_axis_values[1] - geometry.port_axis_values[0])
            if corridor_length <= max_length:
                continue
            slices = self._group_tiles_by_axis(geometry, axis_index)
            if len(slices) <= 2:
                continue
            candidates.append(
                ThroughCorridorCandidate(
                    corridor_idx=idx,
                    axis_index=axis_index,
                    slices=slices,
                )
            )
        random.shuffle(candidates)

        def iterator() -> Iterator[ThroughCorridorCandidate]:
            yield from candidates

        return iterator()

    def dependencies(
        self,
        context: GrowerContext,
        candidate: ThroughCorridorCandidate,
    ) -> CandidateDependencies:
        return CandidateDependencies.from_iterables(
            corridors=(candidate.corridor_idx,)
        )


class ThroughCorridorGeometryPlanner(
    GeometryPlanner[ThroughCorridorCandidate, ThroughCorridorPlan]
):
    def plan(
        self,
        context: GrowerContext,
        candidate: ThroughCorridorCandidate,
    ) -> Optional[ThroughCorridorPlan]:
        templates = context.get_room_templates(RoomKind.THROUGH)
        if not templates:
            return None

        corridor = context.layout.corridors[candidate.corridor_idx]
        axis_index = candidate.axis_index
        slice_indices = list(range(1, len(candidate.slices) - 1))
        random.shuffle(slice_indices)

        for slice_idx in slice_indices:
            junction_tiles = candidate.slices[slice_idx].tiles
            plan = self._attempt_plan_for_slice(
                context,
                corridor,
                candidate.corridor_idx,
                axis_index,
                junction_tiles,
                templates,
            )
            if plan is not None:
                return plan
        return None

    def _attempt_plan_for_slice(
        self,
        context: GrowerContext,
        corridor: Corridor,
        corridor_idx: int,
        axis_index: int,
        junction_tiles: Tuple[TilePos, ...],
        templates: Sequence[RoomTemplate],
    ) -> Optional[ThroughCorridorPlan]:
        seg_existing_a, seg_existing_b = context.split_existing_corridor_geometries(
            corridor,
            junction_tiles,
        )
        if seg_existing_a is None or seg_existing_b is None:
            return None

        requirements: List[PortRequirement] = []
        requirement_mapping: Dict[str, int] = {}

        def add_requirement(
            name: str,
            segment: Optional[CorridorGeometry],
            corridor_end: str,
        ) -> bool:
            requirement = context.build_port_requirement_from_segment(
                segment,
                axis_index,
                name,
                expected_width=corridor.width,
                corridor_idx=corridor_idx,
                corridor_end=corridor_end,
                junction_tiles=junction_tiles,
            )
            if requirement is None:
                return False
            requirement_mapping[name] = len(requirements)
            requirements.append(requirement)
            return True

        if not add_requirement("existing_a", seg_existing_a, "a"):
            return None
        if not add_requirement("existing_b", seg_existing_b, "b"):
            return None

        placement = context.attempt_place_special_room(
            requirements,
            templates,
            RoomKind.THROUGH,
            allowed_overlap_tiles=set(corridor.geometry.tiles),
            allowed_overlap_corridors={corridor_idx},
        )
        if placement is None:
            return None

        placed_room, port_mapping, geometry_overrides = placement
        requirements_mut = list(requirements)
        for req_idx, geometry_override in geometry_overrides.items():
            requirements_mut[req_idx] = replace(
                requirements_mut[req_idx], geometry=geometry_override
            )

        return ThroughCorridorPlan(
            corridor_idx=corridor_idx,
            requirements=tuple(requirements_mut),
            requirement_mapping=dict(requirement_mapping),
            port_mapping=dict(port_mapping),
            junction_room=placed_room,
        )

    def dependencies(
        self,
        context: GrowerContext,
        candidate: ThroughCorridorCandidate,
        plan: ThroughCorridorPlan,
    ) -> CandidateDependencies:
        return CandidateDependencies.from_iterables(
            corridors=(plan.corridor_idx,)
        )


class ThroughCorridorApplier(GrowerApplier[ThroughCorridorCandidate, ThroughCorridorPlan]):
    def __init__(self) -> None:
        self._inserted = 0

    def apply(
        self,
        context: GrowerContext,
        candidate: ThroughCorridorCandidate,
        plan: ThroughCorridorPlan,
    ) -> GrowerStepResult:
        corridor_idx = plan.corridor_idx
        corridor = context.layout.corridors[corridor_idx]

        context.invalidate_corridor_index(corridor_idx)

        junction_room_index = len(context.layout.placed_rooms)
        context.layout.register_room(plan.junction_room)

        junction_room = context.layout.placed_rooms[junction_room_index]
        assignments: Dict[str, Tuple[PortRequirement, int]] = {}

        for suffix in ("a", "b"):
            key = f"existing_{suffix}"
            req_idx = plan.requirement_mapping.get(key)
            if req_idx is None:
                continue
            port_idx = plan.port_mapping[req_idx]
            requirement = plan.requirements[req_idx]
            assignments[suffix] = (requirement, port_idx)
            junction_room.connected_port_indices.add(port_idx)

        linked_indices = context.apply_existing_corridor_segments(
            corridor_idx,
            assignments,
            junction_room_index,
        )

        context.validate_room_corridor_clearance(junction_room_index)

        self._inserted += 1
        return GrowerStepResult(applied=bool(linked_indices))

    def finalize(self, context: GrowerContext) -> int:
        print(
            "Through-corridor grower: placed"
            f" {self._inserted} through-rooms to shorten corridors."
        )
        return self._inserted


def run_through_corridor_grower(context: GrowerContext) -> int:
    if not context.get_room_templates(RoomKind.THROUGH):
        print("Through-corridor grower: skipped - no through-room templates configured.")
        return 0
    grower = DungeonGrower(
        name="through_corridor",
        candidate_finder=ThroughCorridorCandidateFinder(),
        geometry_planner=ThroughCorridorGeometryPlanner(),
        applier=ThroughCorridorApplier(),
    )
    return grower.run(context)
