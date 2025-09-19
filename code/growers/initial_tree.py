"""Initial tree grower responsible for seeding the dungeon layout."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from geometry import Direction, Rotation, TilePos, VALID_ROTATIONS, rotate_direction
from grower_context import GrowerContext
from growers.base import (
    CandidateDependencies,
    CandidateFinder,
    DungeonGrower,
    GeometryPlanner,
    GrowerApplier,
    GrowerStepResult,
)
from models import Corridor, CorridorGeometry, PlacedRoom, RoomKind, RoomTemplate, WorldPort


@dataclass(frozen=True)
class InitialTreeCandidate:
    """Candidate describing an available port on the current tree."""

    room_idx: int
    port_idx: int


@dataclass
class CorridorExpansionPlan:
    """Successful plan for extending a corridor to a new root room."""

    room: PlacedRoom
    room_port_index: int
    geometry: CorridorGeometry
    width: int


class InitialTreeHelper:
    """Helper that encapsulates shared logic for the initial tree grower."""

    def __init__(self, context: GrowerContext) -> None:
        self.context = context
        self.config = context.config
        self.layout = context.layout
        self.room_templates_by_kind = context.room_templates_by_kind
        self._standalone_templates = list(context.get_room_templates(RoomKind.STANDALONE))
        if not self._standalone_templates:
            raise ValueError("Initial tree growth requires standalone room templates")

        self._target_rooms = self.config.num_rooms_to_place
        self._pending_ports: Deque[Tuple[int, int]] = deque()
        self._processed_ports: Set[Tuple[int, int]] = set()
        self._rooms_created = 0
        self._root_component_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Public lifecycle helpers
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Ensure at least one root room exists and seed the candidate queue."""
        if not self.layout.placed_rooms:
            root_room = self._place_initial_root()
            if root_room is None:
                raise ValueError("ERROR: failed to place initial root room.")
            initial_children = self._spawn_direct_links_recursive(root_room)
            self._root_component_id = self.layout.normalize_room_component(root_room.index)
            rooms_to_seed = [root_room, *initial_children]
        else:
            rooms_to_seed = list(self.layout.placed_rooms)
            self._root_component_id = self.layout.normalize_room_component(
                rooms_to_seed[0].index
            )

        self.enqueue_rooms(rooms_to_seed)

    def enqueue_rooms(self, rooms: Iterable[PlacedRoom]) -> None:
        for room in rooms:
            if room.index < 0:
                continue
            for port_idx in room.get_available_port_indices():
                self._pending_ports.append((room.index, port_idx))

    # ------------------------------------------------------------------
    # Candidate iteration
    # ------------------------------------------------------------------
    def iter_candidates(self, context: GrowerContext) -> Iterable[InitialTreeCandidate]:
        while self._pending_ports and self.can_grow_more():
            room_idx, port_idx = self._pending_ports.popleft()
            key = (room_idx, port_idx)
            if key in self._processed_ports:
                continue
            room = context.layout.placed_rooms[room_idx]
            if port_idx in room.connected_port_indices:
                continue
            if self._root_component_id is not None and room.component_id != self._root_component_id:
                continue
            self._processed_ports.add(key)
            yield InitialTreeCandidate(room_idx=room_idx, port_idx=port_idx)

    def can_grow_more(self) -> bool:
        return len(self.layout.placed_rooms) < self._target_rooms

    # ------------------------------------------------------------------
    # Corridor planning and application
    # ------------------------------------------------------------------
    def plan_corridor(self, candidate: InitialTreeCandidate) -> Optional[CorridorExpansionPlan]:
        if not self.can_grow_more():
            return None

        anchor_room = self.layout.placed_rooms[candidate.room_idx]
        anchor_ports = anchor_room.get_world_ports()
        if candidate.port_idx >= len(anchor_ports):
            return None
        anchor_port = anchor_ports[candidate.port_idx]
        if not anchor_port.widths:
            return None

        max_attempts = self.config.max_connected_placement_attempts
        for _ in range(max_attempts):
            corridor_length = self.config.initial_corridor_length.sample()
            if corridor_length <= 0:
                continue
            plan = self._plan_corridor_expansion(
                anchor_room,
                candidate.port_idx,
                anchor_port,
                corridor_length,
            )
            if plan is not None:
                return plan
        return None

    def apply_plan(
        self,
        candidate: InitialTreeCandidate,
        plan: CorridorExpansionPlan,
    ) -> GrowerStepResult:
        anchor_room = self.layout.placed_rooms[candidate.room_idx]
        anchor_component = self.layout.normalize_room_component(anchor_room.index)

        self.layout.register_room(plan.room, anchor_component)
        plan.room.connected_port_indices.add(plan.room_port_index)
        anchor_room.connected_port_indices.add(candidate.port_idx)

        merged_component = self.layout.merge_components(
            anchor_component, plan.room.component_id
        )
        self.layout.set_room_component(anchor_room.index, merged_component)
        self.layout.set_room_component(plan.room.index, merged_component)
        self._root_component_id = merged_component

        corridor = Corridor(
            room_a_index=anchor_room.index,
            port_a_index=candidate.port_idx,
            room_b_index=plan.room.index,
            port_b_index=plan.room_port_index,
            width=plan.width,
            geometry=plan.geometry,
        )
        corridor_idx = self.layout.register_corridor(corridor, merged_component)
        self.layout.set_corridor_component(corridor_idx, merged_component)

        new_rooms = [plan.room]
        new_rooms.extend(self._spawn_direct_links_recursive(plan.room))
        self.enqueue_rooms(new_rooms)
        self._rooms_created += len(new_rooms)

        stop = not self.can_grow_more()
        return GrowerStepResult(applied=True, stop=stop)

    def finalize(self) -> int:
        print(f"Initial tree grower: created {self._rooms_created} rooms for the initial tree.")
        return self._rooms_created

    # ------------------------------------------------------------------
    # Internal helpers (mostly adapted from the original RootRoomPlacer)
    # ------------------------------------------------------------------
    def _place_initial_root(self) -> Optional[PlacedRoom]:
        max_attempts = self.config.max_connected_placement_attempts
        max_fail_windows = self.config.max_consecutive_limit_failures
        attempts = 0
        failures = 0

        while failures < max_fail_windows:
            macro_x, macro_y = self._random_macro_grid_point()
            placement_category, side_proximities = self._describe_macro_position(macro_x, macro_y)
            template = self._pick_root_template(placement_category)
            rotation = self._select_root_rotation(template, placement_category, side_proximities)
            candidate_room = self._build_root_room_candidate(template, rotation, macro_x, macro_y)
            if self.layout.is_valid_placement(candidate_room):
                component_id = self.layout.new_component_id()
                self.layout.register_room(candidate_room, component_id)
                return candidate_room

            attempts += 1
            if attempts >= max_attempts:
                attempts = 0
                failures += 1

        return None

    def _pick_root_template(self, placement_category: str) -> RoomTemplate:
        if placement_category == "middle":
            weights = [rt.root_weight_middle for rt in self._standalone_templates]
        elif placement_category == "edge":
            weights = [rt.root_weight_edge for rt in self._standalone_templates]
        else:
            weights = [rt.root_weight_intermediate for rt in self._standalone_templates]

        if not any(weight > 0 for weight in weights):
            weights = [1.0 for _ in self._standalone_templates]

        return random.choices(self._standalone_templates, weights=weights)[0]

    def _plan_corridor_expansion(
        self,
        anchor_room: PlacedRoom,
        anchor_port_idx: int,
        anchor_port: WorldPort,
        corridor_length: int,
    ) -> Optional[CorridorExpansionPlan]:
        direction = anchor_port.direction.opposite()
        width_options = list(anchor_port.widths)
        random.shuffle(width_options)
        templates = list(self._standalone_templates)
        random.shuffle(templates)

        for width in width_options:
            for template in templates:
                if len(anchor_room.template.ports) == 1 and len(template.ports) == 1:
                    continue
                if (
                    len(self._standalone_templates) > 1
                    and anchor_room.template.name == template.name
                ):
                    continue

                rotations = list(template.unique_rotations())
                random.shuffle(rotations)
                for rotation in rotations:
                    variant = template.rotation_variant(rotation)
                    candidate_indices = list(variant.ports_by_direction.get(direction, ()))
                    random.shuffle(candidate_indices)
                    for candidate_port_idx in candidate_indices:
                        rotated_port = variant.ports[candidate_port_idx]
                        if width not in rotated_port.widths:
                            continue
                        plan = self._build_corridor_plan_for_template(
                            anchor_room,
                            anchor_port_idx,
                            anchor_port,
                            template,
                            rotation,
                            candidate_port_idx,
                            corridor_length,
                            width,
                        )
                        if plan is not None:
                            return plan
        return None

    def _build_corridor_plan_for_template(
        self,
        anchor_room: PlacedRoom,
        anchor_port_idx: int,
        anchor_port: WorldPort,
        template: RoomTemplate,
        rotation: Rotation,
        candidate_port_idx: int,
        corridor_length: int,
        width: int,
    ) -> Optional[CorridorExpansionPlan]:
        candidate_room = self._compute_room_position_for_corridor(
            template,
            rotation,
            candidate_port_idx,
            anchor_port,
            corridor_length,
        )
        if candidate_room is None:
            return None
        if not self.layout.is_valid_placement(candidate_room):
            return None

        candidate_world_port = candidate_room.get_world_ports()[candidate_port_idx]
        temp_room_id = -(len(self.layout.placed_rooms) + 1)
        extra_room_tiles = self._build_extra_room_tiles(candidate_room, temp_room_id)

        geometry = self.context.build_corridor_geometry(
            anchor_room.index,
            anchor_port,
            temp_room_id,
            candidate_world_port,
            width,
            extra_room_tiles=extra_room_tiles,
        )
        if geometry is None or geometry.axis_index is None:
            return None

        axis_index = geometry.axis_index
        exit_anchor = self.context.port_exit_axis_value(anchor_port, axis_index)
        exit_candidate = self.context.port_exit_axis_value(candidate_world_port, axis_index)
        actual_length = abs(exit_candidate - exit_anchor)
        min_expected = max(1, corridor_length // 2)
        if actual_length < min_expected:
            return None

        for tile in geometry.tiles:
            if self.layout.spatial_index.get_corridors_at(tile):
                return None

        return CorridorExpansionPlan(
            room=candidate_room,
            room_port_index=candidate_port_idx,
            geometry=geometry,
            width=width,
        )

    def _compute_room_position_for_corridor(
        self,
        template: RoomTemplate,
        rotation: Rotation,
        port_idx: int,
        anchor_port: WorldPort,
        corridor_length: int,
    ) -> Optional[PlacedRoom]:
        variant = template.rotation_variant(rotation)
        rotated_port = variant.ports[port_idx]
        axis_index = 0 if anchor_port.direction.dx != 0 else 1
        direction_step = anchor_port.direction.dx if axis_index == 0 else anchor_port.direction.dy
        if direction_step == 0:
            return None

        cross_source = anchor_port.pos[1] if axis_index == 0 else anchor_port.pos[0]
        cross_local = rotated_port.pos[1] if axis_index == 0 else rotated_port.pos[0]
        cross_delta = cross_source - cross_local
        if not math.isclose(cross_delta, round(cross_delta), abs_tol=1e-6):
            return None
        cross_coord = int(round(cross_delta))

        exit_anchor = self.context.port_exit_axis_value(anchor_port, axis_index)
        target_exit = exit_anchor + direction_step * corridor_length

        tile_coords = [
            tile.x if axis_index == 0 else tile.y for tile in rotated_port.tiles
        ]
        boundary_local = max(tile_coords) if direction_step > 0 else min(tile_coords)
        axis_position = target_exit - direction_step - boundary_local
        if not math.isclose(axis_position, round(axis_position), abs_tol=1e-6):
            return None
        axis_coord = int(round(axis_position))

        if axis_index == 0:
            room_x = axis_coord
            room_y = cross_coord
        else:
            room_x = cross_coord
            room_y = axis_coord

        return PlacedRoom(template, room_x, room_y, rotation)

    def _build_extra_room_tiles(self, room: PlacedRoom, placeholder: int) -> Dict[TilePos, int]:
        bounds = room.get_bounds()
        tiles: Dict[TilePos, int] = {}
        for ty in range(bounds.y, bounds.max_y):
            for tx in range(bounds.x, bounds.max_x):
                tiles[TilePos(tx, ty)] = placeholder
        return tiles

    def _random_macro_grid_point(self) -> Tuple[int, int]:
        max_macro_x = (self.config.width // self.config.macro_grid_size) - 1
        max_macro_y = (self.config.height // self.config.macro_grid_size) - 1
        if max_macro_x <= 1 or max_macro_y <= 1:
            raise ValueError("Grid too small to place rooms with macro-grid alignment")

        macro_x = random.randint(1, max_macro_x - 1) * self.config.macro_grid_size
        macro_y = random.randint(1, max_macro_y - 1) * self.config.macro_grid_size
        return macro_x, macro_y

    def _build_root_room_candidate(
        self, template: RoomTemplate, rotation: Rotation, macro_x: int, macro_y: int
    ) -> PlacedRoom:
        anchor_port_index = random.randrange(len(template.ports))

        rotated_room = PlacedRoom(template, 0, 0, rotation)
        rotated_ports = rotated_room.get_world_ports()
        rotated_anchor_port = rotated_ports[anchor_port_index]

        offsets = self.config.door_macro_alignment_offsets
        try:
            offset_x, offset_y = offsets[rotated_anchor_port.direction]
        except KeyError as exc:
            raise ValueError(f"Unsupported port direction {rotated_anchor_port.direction}") from exc

        snapped_port_x = macro_x + offset_x
        snapped_port_y = macro_y + offset_y

        room_x = int(round(snapped_port_x - rotated_anchor_port.pos[0]))
        room_y = int(round(snapped_port_y - rotated_anchor_port.pos[1]))
        return PlacedRoom(template, room_x, room_y, rotation)

    @staticmethod
    def _categorize_side_distance(distance: float, span: int) -> str:
        if span <= 0:
            return "far"
        ratio = max(0.0, min(distance / float(span), 1.0))
        if ratio <= 0.2:
            return "close"
        if ratio >= 0.35:
            return "far"
        return "intermediate"

    def _describe_macro_position(self, macro_x: int, macro_y: int) -> Tuple[str, Dict[str, str]]:
        side_proximities = {
            "left": self._categorize_side_distance(macro_x, self.config.width),
            "right": self._categorize_side_distance(self.config.width - macro_x, self.config.width),
            "top": self._categorize_side_distance(macro_y, self.config.height),
            "bottom": self._categorize_side_distance(self.config.height - macro_y, self.config.height),
        }

        if any(value == "close" for value in side_proximities.values()):
            proximity = "edge"
        elif all(value == "far" for value in side_proximities.values()):
            proximity = "middle"
        else:
            proximity = "intermediate"

        return proximity, side_proximities

    def _select_root_rotation(
        self,
        template: RoomTemplate,
        placement_category: str,
        side_proximities: Dict[str, str],
    ) -> Rotation:
        preferred_dir = template.preferred_center_facing_dir
        if placement_category != "edge" or preferred_dir is None:
            return Rotation.random()

        inward_directions: List[Direction] = []
        if side_proximities.get("left") == "close":
            inward_directions.append(Direction.EAST)
        if side_proximities.get("right") == "close":
            inward_directions.append(Direction.WEST)
        if side_proximities.get("top") == "close":
            inward_directions.append(Direction.SOUTH)
        if side_proximities.get("bottom") == "close":
            inward_directions.append(Direction.NORTH)

        if not inward_directions:
            return Rotation.random()

        rotation_weights: List[float] = []
        for rotation in VALID_ROTATIONS:
            rotated_dir = rotate_direction(preferred_dir, rotation)
            weight = 1.0 if rotated_dir in inward_directions else 0.0
            rotation_weights.append(weight)

        if any(weight > 0 for weight in rotation_weights):
            return random.choices(VALID_ROTATIONS, weights=rotation_weights)[0]
        return Rotation.random()

    def _sample_num_direct_links(self) -> int:
        items = list(self.config.direct_link_counts_probs.items())
        total = sum(p for _, p in items)
        if total <= 0:
            items = [(0, 0.4), (1, 0.3), (2, 0.3)]
            total = 1.0
        r = random.random()
        acc = 0.0
        for k, p in items:
            acc += p / total
            if r <= acc:
                return k
        return items[-1][0]

    def _attempt_place_connected_to(self, anchor_room: PlacedRoom) -> Optional[PlacedRoom]:
        if anchor_room.index < 0:
            raise ValueError("Anchor room must be registered before creating connections")
        anchor_component_id = self.layout.normalize_room_component(anchor_room.index)
        anchor_world_ports = anchor_room.get_world_ports()
        available_anchor_indices = anchor_room.get_available_port_indices()
        random.shuffle(available_anchor_indices)
        standalone_templates: Sequence[RoomTemplate] = self.room_templates_by_kind.get(
            RoomKind.STANDALONE, ()
        )

        for anchor_idx in available_anchor_indices:
            awp = anchor_world_ports[anchor_idx]
            ax, ay = awp.pos
            direction = awp.direction
            dx, dy = direction.dx, direction.dy
            target_port_pos = (ax + dx, ay + dy)
            for _ in range(self.config.max_connected_placement_attempts):
                template = random.choices(
                    standalone_templates,
                    weights=[rt.direct_weight for rt in standalone_templates],
                )[0]
                if len(anchor_room.template.ports) == 1 and len(template.ports) == 1:
                    continue
                if anchor_room.template.name == template.name:
                    continue
                rotation = Rotation.random()
                temp_room = PlacedRoom(template, 0, 0, rotation)
                rot_ports = temp_room.get_world_ports()
                compatible_port_indices = [
                    i
                    for i, p in enumerate(rot_ports)
                    if p.direction == direction.opposite() and (p.widths & awp.widths)
                ]
                if not compatible_port_indices:
                    continue
                cand_idx = random.choice(compatible_port_indices)
                cand_port = rot_ports[cand_idx]
                rpx, rpy = cand_port.pos
                nx = int(round(target_port_pos[0] - rpx))
                ny = int(round(target_port_pos[1] - rpy))
                candidate = PlacedRoom(template, nx, ny, rotation)
                if self.layout.is_valid_placement_with_anchor(candidate, anchor_room):
                    self.layout.register_room(candidate, anchor_component_id)
                    self.layout.add_room_room_link(anchor_room.index, candidate.index)
                    anchor_room.connected_port_indices.add(anchor_idx)
                    candidate.connected_port_indices.add(cand_idx)
                    return candidate
        return None

    def _spawn_direct_links_recursive(self, from_room: PlacedRoom) -> List[PlacedRoom]:
        new_rooms: List[PlacedRoom] = []
        num_links = self._sample_num_direct_links()
        for _ in range(num_links):
            child = self._attempt_place_connected_to(from_room)
            if child is not None:
                new_rooms.append(child)
                new_rooms.extend(self._spawn_direct_links_recursive(child))
        return new_rooms


class InitialTreeCandidateFinder(
    CandidateFinder[InitialTreeCandidate, CorridorExpansionPlan]
):
    def __init__(self, helper: InitialTreeHelper) -> None:
        self.helper = helper

    def find_candidates(self, context: GrowerContext):
        return self.helper.iter_candidates(context)

    def dependencies(
        self, context: GrowerContext, candidate: InitialTreeCandidate
    ) -> CandidateDependencies:
        return CandidateDependencies.from_iterables(rooms=(candidate.room_idx,))


class InitialTreeGeometryPlanner(
    GeometryPlanner[InitialTreeCandidate, CorridorExpansionPlan]
):
    def __init__(self, helper: InitialTreeHelper) -> None:
        self.helper = helper

    def plan(
        self, context: GrowerContext, candidate: InitialTreeCandidate
    ) -> Optional[CorridorExpansionPlan]:
        return self.helper.plan_corridor(candidate)


class InitialTreeApplier(GrowerApplier[InitialTreeCandidate, CorridorExpansionPlan]):
    def __init__(self, helper: InitialTreeHelper) -> None:
        self.helper = helper

    def apply(
        self,
        context: GrowerContext,
        candidate: InitialTreeCandidate,
        plan: CorridorExpansionPlan,
    ) -> GrowerStepResult:
        return self.helper.apply_plan(candidate, plan)

    def finalize(self, context: GrowerContext) -> int:
        return self.helper.finalize()


def run_initial_tree_grower(context: GrowerContext) -> int:
    """Entry point that seeds the dungeon layout and grows the initial tree."""

    helper = InitialTreeHelper(context)
    helper.initialize()

    grower = DungeonGrower(
        name="initial_tree",
        candidate_finder=InitialTreeCandidateFinder(helper),
        geometry_planner=InitialTreeGeometryPlanner(helper),
        applier=InitialTreeApplier(helper),
    )
    return grower.run(context)


# Re-export selected helpers for unit tests.
categorize_side_distance = InitialTreeHelper._categorize_side_distance
