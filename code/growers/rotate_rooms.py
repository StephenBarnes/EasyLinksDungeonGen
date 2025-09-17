from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple, TYPE_CHECKING

from geometry import Rect, Rotation
from growers.base import (
    CandidateDependencies,
    CandidateFinder,
    DungeonGrower,
    GeometryPlanner,
    GrowerApplier,
    GrowerStepResult,
)

if TYPE_CHECKING:
    from grower_context import GrowerContext
    from models import PlacedRoom


@dataclass(frozen=True)
class RotateRoomCandidate:
    room_index: int
    rotation_options: Tuple[Rotation, ...]


@dataclass(frozen=True)
class RotateRoomPlan:
    new_rotation: Rotation
    bounds: Rect


def _rotation_options(template, current_rotation: Rotation) -> Tuple[Rotation, ...]:
    if template.is_symmetric_90:
        return ()
    if template.is_symmetric_180:
        deltas = (90, 270)
    else:
        deltas = (90, 180, 270)

    options: List[Rotation] = []
    seen: set[Rotation] = set()
    base_degrees = current_rotation.degrees
    for delta in deltas:
        rotation = Rotation.from_degrees(base_degrees + delta)
        if rotation == current_rotation:
            continue
        if rotation in seen:
            continue
        seen.add(rotation)
        options.append(rotation)
    return tuple(options)


class RotateRoomCandidateFinder(CandidateFinder[RotateRoomCandidate, RotateRoomPlan]):
    def find_candidates(self, context: GrowerContext) -> Iterable[RotateRoomCandidate]:
        candidates: List[RotateRoomCandidate] = []
        for room_index, room in enumerate(context.layout.placed_rooms):
            if room.connected_port_indices:
                continue
            rotation_options = _rotation_options(room.template, room.rotation)
            if not rotation_options:
                continue
            candidates.append(
                RotateRoomCandidate(
                    room_index=room_index,
                    rotation_options=rotation_options,
                )
            )
        random.shuffle(candidates)
        return candidates

    def dependencies(
        self,
        context: GrowerContext,
        candidate: RotateRoomCandidate,
    ) -> CandidateDependencies:
        return CandidateDependencies.from_iterables(rooms=(candidate.room_index,))


class RotateRoomGeometryPlanner(GeometryPlanner[RotateRoomCandidate, RotateRoomPlan]):
    def plan(
        self,
        context: GrowerContext,
        candidate: RotateRoomCandidate,
    ) -> RotateRoomPlan | None:
        room = context.layout.placed_rooms[candidate.room_index]
        rotation_order = list(candidate.rotation_options)
        random.shuffle(rotation_order)
        for rotation in rotation_order:
            bounds = self._compute_bounds(room, rotation)
            if not self._fits(context, room, bounds):
                continue
            return RotateRoomPlan(new_rotation=rotation, bounds=bounds)
        return None

    @staticmethod
    def _compute_bounds(room: PlacedRoom, rotation: Rotation) -> Rect:
        width, height = room.template.size
        if rotation in (Rotation.DEG_0, Rotation.DEG_180):
            rotated_width, rotated_height = width, height
        else:
            rotated_width, rotated_height = height, width
        return Rect(room.x, room.y, rotated_width, rotated_height)

    @staticmethod
    def _fits(context: GrowerContext, room: PlacedRoom, bounds: Rect) -> bool:
        if bounds.x < 0 or bounds.y < 0:
            return False
        if bounds.max_x > context.config.width or bounds.max_y > context.config.height:
            return False

        ignore_rooms = {room.index}

        spatial_index = context.layout.spatial_index
        if not spatial_index.is_area_clear(bounds, ignore_rooms=ignore_rooms):
            return False

        margin = context.config.min_room_separation
        if margin > 0:
            expanded = bounds.expand(margin)
            if not spatial_index.is_area_clear(expanded, ignore_rooms=ignore_rooms):
                return False

        return True


class RotateRoomApplier(GrowerApplier[RotateRoomCandidate, RotateRoomPlan]):
    def __init__(self) -> None:
        self._rotated = 0

    def apply(
        self,
        context: GrowerContext,
        candidate: RotateRoomCandidate,
        plan: RotateRoomPlan,
    ) -> GrowerStepResult:
        room = context.layout.placed_rooms[candidate.room_index]
        if room.rotation == plan.new_rotation:
            return GrowerStepResult(applied=False)

        context.invalidate_room_index(candidate.room_index)
        context.layout.spatial_index.remove_room(candidate.room_index, room)
        room.rotation = plan.new_rotation
        context.layout.spatial_index.add_room(candidate.room_index, room)
        self._rotated += 1
        return GrowerStepResult(applied=True)

    def finalize(self, context: GrowerContext) -> int:
        if self._rotated:
            print(f"Rotate-room grower: rotated {self._rotated} rooms.")
        return self._rotated


def run_rotate_rooms_grower(context: GrowerContext) -> int:
    grower = DungeonGrower(
        name="rotate_rooms",
        candidate_finder=RotateRoomCandidateFinder(),
        geometry_planner=RotateRoomGeometryPlanner(),
        applier=RotateRoomApplier(),
    )
    return grower.run(context)
