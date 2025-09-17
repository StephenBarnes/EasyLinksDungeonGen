from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Iterator, List, Optional, Set, Tuple

from dungeon_models import Corridor, CorridorGeometry, PlacedRoom, WorldPort

from growers.base import (
    CandidateFinder,
    DungeonGrower,
    GeometryPlanner,
    GrowerApplier,
    GrowerStepResult,
)

if TYPE_CHECKING:
    from dungeon_generator import DungeonGenerator


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

    def find_candidates(self, generator: DungeonGenerator) -> Iterable[BentRoomCandidate]:
        self._used_ports.clear()
        room_world_ports = [room.get_world_ports() for room in generator.placed_rooms]
        available_ports = generator._list_available_ports(room_world_ports)

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
                if generator._rooms_share_component(room_a_idx, room_b_idx):
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
        generator: DungeonGenerator,
        candidate: BentRoomCandidate,
        plan: BentRoomPlan,
    ) -> None:
        self._used_ports.add((candidate.room_a_idx, candidate.port_a_idx))
        self._used_ports.add((candidate.room_b_idx, candidate.port_b_idx))


class BentRoomGeometryPlanner(GeometryPlanner[BentRoomCandidate, BentRoomPlan]):
    def plan(
        self,
        generator: DungeonGenerator,
        candidate: BentRoomCandidate,
    ) -> Optional[BentRoomPlan]:
        plan = generator._plan_bend_room(
            candidate.room_a_idx,
            candidate.port_a_idx,
            candidate.room_b_idx,
            candidate.port_b_idx,
        )
        if plan is None:
            return None
        width, bend_room, corridor_plans = plan
        return BentRoomPlan(
            width=width,
            bend_room=bend_room,
            corridor_plans=tuple(corridor_plans),
        )


class BentRoomApplier(GrowerApplier[BentRoomCandidate, BentRoomPlan]):
    def __init__(self) -> None:
        self._created = 0

    def apply(
        self,
        generator: DungeonGenerator,
        candidate: BentRoomCandidate,
        plan: BentRoomPlan,
    ) -> GrowerStepResult:
        component_id = generator._merge_components(
            generator._normalize_room_component(candidate.room_a_idx),
            generator._normalize_room_component(candidate.room_b_idx),
        )
        generator._set_room_component(candidate.room_a_idx, component_id)
        generator._set_room_component(candidate.room_b_idx, component_id)

        bend_room_index = len(generator.placed_rooms)
        generator._register_room(plan.bend_room, component_id)

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
            generator._register_corridor(corridor, component_id)
            generator.placed_rooms[existing_room_idx].connected_port_indices.add(existing_port_idx)
            generator.placed_rooms[bend_room_index].connected_port_indices.add(bend_port_idx)

        self._created += 1
        stop = generator.component_manager.has_single_component()
        return GrowerStepResult(applied=True, stop=stop)

    def finalize(self, generator: DungeonGenerator) -> int:
        if self._created == 0:
            print("Bent-room-to-room grower: no bend room placements succeeded.")
        else:
            print(f"Bent-room-to-room grower: created {self._created} bend rooms.")
        return self._created


def run_bent_room_to_room_grower(generator: DungeonGenerator) -> int:
    grower = DungeonGrower(
        name="bent_room_to_room",
        candidate_finder=BentRoomCandidateFinder(),
        geometry_planner=BentRoomGeometryPlanner(),
        applier=BentRoomApplier(),
    )
    return grower.run(generator)
