from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from geometry import TilePos, VALID_ROTATIONS
from models import Corridor, CorridorGeometry, PlacedRoom, WorldPort

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
        room_world_ports = [room.get_world_ports() for room in generator.layout.placed_rooms]
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
                if generator.layout.rooms_share_component(room_a_idx, room_b_idx):
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
        if not generator.bend_room_templates:
            return None

        room_a = generator.layout.placed_rooms[candidate.room_a_idx]
        room_b = generator.layout.placed_rooms[candidate.room_b_idx]
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

        candidate_room_index = len(generator.layout.placed_rooms)

        bend_templates = list(generator.bend_room_templates)
        random.shuffle(bend_templates)

        for width in width_options:
            for template in bend_templates:
                for rotation in VALID_ROTATIONS:
                    temp_room = PlacedRoom(template, 0, 0, rotation)
                    rotated_ports = temp_room.get_world_ports()

                    horizontal_candidates = [
                        (idx, rp)
                        for idx, rp in enumerate(rotated_ports)
                        if rp.direction == horizontal_dir.opposite()
                    ]
                    vertical_candidates = [
                        (idx, rp)
                        for idx, rp in enumerate(rotated_ports)
                        if rp.direction == vertical_dir.opposite()
                    ]

                    if not horizontal_candidates or not vertical_candidates:
                        continue

                    for bend_h_idx, bend_h_port in horizontal_candidates:
                        if width not in bend_h_port.widths:
                            continue
                        for bend_v_idx, bend_v_port in vertical_candidates:
                            if bend_v_idx == bend_h_idx:
                                continue
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
                            if not generator.layout.is_valid_placement(placed_bend):
                                continue

                            bend_bounds = placed_bend.get_bounds()
                            overlaps_corridor = False
                            extra_room_tiles: Dict[TilePos, int] = {}
                            for ty in range(bend_bounds.y, bend_bounds.max_y):
                                for tx in range(bend_bounds.x, bend_bounds.max_x):
                                    tile = TilePos(tx, ty)
                                    if generator.layout.spatial_index.has_corridor_at(tile):
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

                            geom_h = generator._build_corridor_geometry(
                                horizontal_info["room_idx"],
                                horizontal_info["port"],
                                candidate_room_index,
                                bend_world_h,
                                width,
                                extra_room_tiles,
                            )
                            if geom_h is None:
                                continue

                            geom_v = generator._build_corridor_geometry(
                                vertical_info["room_idx"],
                                vertical_info["port"],
                                candidate_room_index,
                                bend_world_v,
                                width,
                                extra_room_tiles,
                            )
                            if geom_v is None:
                                continue

                            if any(generator.layout.spatial_index.has_corridor_at(tile) for tile in geom_h.tiles):
                                continue
                            if any(generator.layout.spatial_index.has_corridor_at(tile) for tile in geom_v.tiles):
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
    def __init__(self) -> None:
        self._created = 0

    def apply(
        self,
        generator: DungeonGenerator,
        candidate: BentRoomCandidate,
        plan: BentRoomPlan,
    ) -> GrowerStepResult:
        component_id = generator.layout.merge_components(
            generator.layout.normalize_room_component(candidate.room_a_idx),
            generator.layout.normalize_room_component(candidate.room_b_idx),
        )
        generator.layout.set_room_component(candidate.room_a_idx, component_id)
        generator.layout.set_room_component(candidate.room_b_idx, component_id)

        bend_room_index = len(generator.layout.placed_rooms)
        generator.layout.register_room(plan.bend_room, component_id)

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
            generator.layout.register_corridor(corridor, component_id)
            generator.layout.placed_rooms[existing_room_idx].connected_port_indices.add(existing_port_idx)
            generator.layout.placed_rooms[bend_room_index].connected_port_indices.add(bend_port_idx)

        self._created += 1
        stop = generator.layout.component_manager.has_single_component()
        return GrowerStepResult(applied=True, stop=stop)

    def finalize(self, generator: DungeonGenerator) -> int:
        if self._created == 0:
            print("Bent-room-to-room grower: no bend room placements succeeded.")
        else:
            print(f"Bent-room-to-room grower: created {self._created} bend rooms.")
        return self._created


def run_bent_room_to_room_grower(generator: DungeonGenerator) -> int:
    if len(generator.layout.placed_rooms) < 2:
        print("Bent-room-to-room grower: skipped - not enough rooms to connect.")
        return 0
    if not generator.bend_room_templates:
        print("Bent-room-to-room grower: skipped - no bend room templates available.")
        return 0
    if generator.layout.component_manager.has_single_component():
        print("Bent-room-to-room grower: skipped - already fully connected.")
        return 0

    grower = DungeonGrower(
        name="bent_room_to_room",
        candidate_finder=BentRoomCandidateFinder(),
        geometry_planner=BentRoomGeometryPlanner(),
        applier=BentRoomApplier(),
    )
    return grower.run(generator)
