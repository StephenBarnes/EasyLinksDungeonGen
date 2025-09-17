"""Connect orthogonal room ports by inserting an intermediate bend room.

The grower enumerates every available room port, pairing ports from different
rooms whose facing directions are perpendicular and share at least one corridor
width. Candidate pairs are prioritized by Manhattan distance so that nearby
connections are tried first.

For each candidate, the planner iterates over compatible corridor widths, every
eligible bend-room template, and all four cardinal rotations. It rotates a
template, locates its ports that face the candidate ports, and translates the
room so the selected ports coincide with the world coordinates of the two
target ports. Placements are rejected if they require sub-tile offsets, collide
with existing rooms or corridors, or reuse corridor tiles.

Once a placement fits, the planner builds straight corridor geometries from
each existing room to the bend room, verifying that the tile sets do not overlap
and remain corridor-free inside the spatial index. When both corridors can be
constructed, the plan is accepted and the applier merges the room components,
registers the bend room, and commits the corridor geometries to the layout.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from geometry import TilePos
from models import Corridor, CorridorGeometry, PlacedRoom, RoomKind, WorldPort

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


@dataclass(frozen=True)
class BentRoomCandidate:
    room_a_idx: int
    port_a_idx: int
    port_a: WorldPort
    room_b_idx: int
    port_b_idx: int
    port_b: WorldPort
    min_width: int
    priority: Tuple[float, int]


@dataclass(frozen=True)
class BentRoomPlan:
    width: int
    bend_room: PlacedRoom
    corridor_plans: Tuple[Tuple[int, int, int, CorridorGeometry], ...]


class BentRoomCandidateFinder(CandidateFinder[BentRoomCandidate, BentRoomPlan]):
    def __init__(self) -> None:
        self._used_ports: Set[Tuple[int, int]] = set()

    def find_candidates(self, context: GrowerContext) -> Iterable[BentRoomCandidate]:
        self._used_ports.clear()
        room_world_ports = [room.get_world_ports() for room in context.layout.placed_rooms]
        available_ports = context.list_available_ports(room_world_ports)

        records = [
            {
                "room_idx": room_idx,
                "port_idx": port_idx,
                "port": world_port,
            }
            for room_idx, port_idx, world_port in available_ports
        ]

        candidates: List[BentRoomCandidate] = []

        for i, port_a_info in enumerate(records):
            for port_b_info in records[i + 1 :]:
                room_a_idx = port_a_info["room_idx"]
                room_b_idx = port_b_info["room_idx"]
                if not context.layout.should_allow_connection(
                    ("room", room_a_idx),
                    ("room", room_b_idx),
                ):
                    continue

                port_a = port_a_info["port"]
                port_b = port_b_info["port"]
                if port_a.direction.dot(port_b.direction) != 0:
                    continue

                common_widths = port_a.widths & port_b.widths
                if not common_widths:
                    continue

                distance = abs(port_a.pos[0] - port_b.pos[0]) + abs(port_a.pos[1] - port_b.pos[1])
                min_width = min(common_widths)
                candidates.append(
                    BentRoomCandidate(
                        room_a_idx=room_a_idx,
                        port_a_idx=port_a_info["port_idx"],
                        port_a=port_a,
                        room_b_idx=room_b_idx,
                        port_b_idx=port_b_info["port_idx"],
                        port_b=port_b,
                        min_width=int(min_width),
                        priority=(float(distance), int(min_width)),
                    )
                )

        candidates.sort(key=lambda c: c.priority)

        def iterator() -> Iterator[BentRoomCandidate]:
            for candidate in candidates:
                key_a = (candidate.room_a_idx, candidate.port_a_idx)
                key_b = (candidate.room_b_idx, candidate.port_b_idx)
                if key_a in self._used_ports or key_b in self._used_ports:
                    continue
                yield candidate

        return iterator()

    def on_success(
        self,
        context: GrowerContext,
        candidate: BentRoomCandidate,
        plan: BentRoomPlan,
    ) -> None:
        self._used_ports.add((candidate.room_a_idx, candidate.port_a_idx))
        self._used_ports.add((candidate.room_b_idx, candidate.port_b_idx))

    def dependencies(
        self,
        context: GrowerContext,
        candidate: BentRoomCandidate,
    ) -> CandidateDependencies:
        return CandidateDependencies.from_iterables(
            rooms=(candidate.room_a_idx, candidate.room_b_idx)
        )


class BentRoomGeometryPlanner(GeometryPlanner[BentRoomCandidate, BentRoomPlan]):
    def plan(
        self,
        context: GrowerContext,
        candidate: BentRoomCandidate,
    ) -> Optional[BentRoomPlan]:
        room_a = context.layout.placed_rooms[candidate.room_a_idx]
        room_b = context.layout.placed_rooms[candidate.room_b_idx]
        port_a = room_a.get_world_ports()[candidate.port_a_idx]
        port_b = room_b.get_world_ports()[candidate.port_b_idx]

        if port_a.direction.dot(port_b.direction) != 0:
            return None

        width_options = sorted(port_a.widths & port_b.widths)
        if not width_options:
            return None

        def port_is_horizontal(port: WorldPort) -> bool:
            return port.direction.dx != 0

        def port_is_vertical(port: WorldPort) -> bool:
            return port.direction.dy != 0

        port_infos = [
            {"room_idx": candidate.room_a_idx, "port_idx": candidate.port_a_idx, "port": port_a},
            {"room_idx": candidate.room_b_idx, "port_idx": candidate.port_b_idx, "port": port_b},
        ]

        horizontal_info = next((info for info in port_infos if port_is_horizontal(info["port"])), None)
        vertical_info = next((info for info in port_infos if port_is_vertical(info["port"])), None)
        if horizontal_info is None or vertical_info is None:
            return None

        horizontal_dir = horizontal_info["port"].direction
        vertical_dir = vertical_info["port"].direction

        candidate_room_index = len(context.layout.placed_rooms)

        bend_templates = context.weighted_templates(RoomKind.BEND)
        if not bend_templates:
            return None

        for width in width_options:
            for template in bend_templates:
                for rotation in template.unique_rotations():
                    variant = template.rotation_variant(rotation)

                    horizontal_indices = variant.ports_by_direction.get(
                        horizontal_dir.opposite(),
                        (),
                    )
                    vertical_indices = variant.ports_by_direction.get(
                        vertical_dir.opposite(),
                        (),
                    )
                    if not horizontal_indices or not vertical_indices:
                        continue

                    for bend_h_idx in horizontal_indices:
                        bend_h_port = variant.ports[bend_h_idx]
                        if width not in bend_h_port.widths:
                            continue
                        for bend_v_idx in vertical_indices:
                            if bend_v_idx == bend_h_idx:
                                continue
                            bend_v_port = variant.ports[bend_v_idx]
                            if width not in bend_v_port.widths:
                                continue

                            candidate_x = vertical_info["port"].pos[0] - bend_v_port.pos[0]
                            candidate_y = horizontal_info["port"].pos[1] - bend_h_port.pos[1]
                            if not math.isclose(candidate_x, round(candidate_x), abs_tol=1e-6):
                                continue
                            if not math.isclose(candidate_y, round(candidate_y), abs_tol=1e-6):
                                continue

                            placed_bend = PlacedRoom(
                                template,
                                int(round(candidate_x)),
                                int(round(candidate_y)),
                                rotation,
                            )
                            if not context.layout.is_valid_placement(placed_bend):
                                continue

                            bend_bounds = placed_bend.get_bounds()
                            overlaps_corridor = False
                            extra_room_tiles: Dict[TilePos, int] = {}
                            for ty in range(bend_bounds.y, bend_bounds.max_y):
                                for tx in range(bend_bounds.x, bend_bounds.max_x):
                                    tile = TilePos(tx, ty)
                                    if context.layout.spatial_index.has_corridor_at(tile):
                                        overlaps_corridor = True
                                        break
                                    extra_room_tiles[tile] = candidate_room_index
                                if overlaps_corridor:
                                    break
                            if overlaps_corridor:
                                continue

                            bend_world_ports = placed_bend.get_world_ports()
                            bend_world_h = bend_world_ports[bend_h_idx]
                            bend_world_v = bend_world_ports[bend_v_idx]

                            if width not in bend_world_h.widths or width not in bend_world_v.widths:
                                continue

                            geom_h = context.build_corridor_geometry(
                                horizontal_info["room_idx"],
                                horizontal_info["port"],
                                candidate_room_index,
                                bend_world_h,
                                width,
                                extra_room_tiles,
                            )
                            if geom_h is None:
                                continue

                            geom_v = context.build_corridor_geometry(
                                vertical_info["room_idx"],
                                vertical_info["port"],
                                candidate_room_index,
                                bend_world_v,
                                width,
                                extra_room_tiles,
                            )
                            if geom_v is None:
                                continue

                            if any(context.layout.spatial_index.has_corridor_at(tile) for tile in geom_h.tiles):
                                continue
                            if any(context.layout.spatial_index.has_corridor_at(tile) for tile in geom_v.tiles):
                                continue

                            tiles_h = set(geom_h.tiles)
                            tiles_v = set(geom_v.tiles)
                            if tiles_h & tiles_v:
                                continue

                            corridors = (
                                (
                                    horizontal_info["room_idx"],
                                    horizontal_info["port_idx"],
                                    bend_h_idx,
                                    geom_h,
                                ),
                                (
                                    vertical_info["room_idx"],
                                    vertical_info["port_idx"],
                                    bend_v_idx,
                                    geom_v,
                                ),
                            )
                            return BentRoomPlan(
                                width=width,
                                bend_room=placed_bend,
                                corridor_plans=corridors,
                            )

        return None


class BentRoomApplier(GrowerApplier[BentRoomCandidate, BentRoomPlan]):
    def __init__(self, stop_after_first: bool) -> None:
        self._created = 0
        self._stop_after_first = stop_after_first

    def apply(
        self,
        context: GrowerContext,
        candidate: BentRoomCandidate,
        plan: BentRoomPlan,
    ) -> GrowerStepResult:
        component_id = context.layout.merge_components(
            context.layout.normalize_room_component(candidate.room_a_idx),
            context.layout.normalize_room_component(candidate.room_b_idx),
        )
        context.layout.set_room_component(candidate.room_a_idx, component_id)
        context.layout.set_room_component(candidate.room_b_idx, component_id)

        bend_room_index = len(context.layout.placed_rooms)
        context.layout.register_room(plan.bend_room, component_id)

        for existing_room_idx, existing_port_idx, bend_port_idx, geometry in plan.corridor_plans:
            corridor = Corridor(
                room_a_index=existing_room_idx,
                port_a_index=existing_port_idx,
                room_b_index=bend_room_index,
                port_b_index=bend_port_idx,
                width=plan.width,
                geometry=geometry,
                component_id=component_id,
            )
            context.layout.register_corridor(corridor, component_id)
            context.layout.placed_rooms[existing_room_idx].connected_port_indices.add(existing_port_idx)
            context.layout.placed_rooms[bend_room_index].connected_port_indices.add(bend_port_idx)

        self._created += 1
        stop = self._stop_after_first or context.layout.component_manager.has_single_component()
        return GrowerStepResult(applied=True, stop=stop)

    def finalize(self, context: GrowerContext) -> int:
        if self._created == 0:
            print("Bent-room-to-room grower: no bend room placements succeeded.")
        else:
            print(f"Bent-room-to-room grower: created {self._created} bend rooms.")
        return self._created


def run_bent_room_to_room_grower(context: GrowerContext, stop_after_first: bool) -> int:
    if len(context.layout.placed_rooms) < 2:
        print("Bent-room-to-room grower: skipped - not enough rooms to connect.")
        return 0
    if not context.get_room_templates(RoomKind.BEND):
        print("Bent-room-to-room grower: skipped - no bend room templates available.")
        return 0
    if context.layout.component_manager.has_single_component():
        print("Bent-room-to-room grower: skipped - already fully connected.")
        return 0

    grower = DungeonGrower(
        name="bent_room_to_room",
        candidate_finder=BentRoomCandidateFinder(),
        geometry_planner=BentRoomGeometryPlanner(),
        applier=BentRoomApplier(stop_after_first=stop_after_first),
    )
    return grower.run(context)
