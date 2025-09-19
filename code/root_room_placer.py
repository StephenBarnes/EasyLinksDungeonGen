"""Utilities for populating a fresh layout with the initial set of rooms."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from dungeon_config import DungeonConfig
from dungeon_layout import DungeonLayout
from geometry import Direction, Rotation, TilePos, rotate_direction, VALID_ROTATIONS
from grower_context import GrowerContext
from models import Corridor, CorridorGeometry, PlacedRoom, RoomKind, RoomTemplate, WorldPort


@dataclass
class CorridorExpansionPlan:
    """Successful plan for extending a corridor to a new root room."""

    room: PlacedRoom
    room_port_index: int
    geometry: CorridorGeometry
    width: int


class RootRoomPlacer:
    """Populate an empty layout with root rooms and their direct links."""

    def __init__(
        self,
        config: DungeonConfig,
        layout: DungeonLayout,
        room_templates_by_kind: Mapping[RoomKind, Sequence[RoomTemplate]],
    ) -> None:
        self.config = config
        self.layout = layout
        self.room_templates_by_kind = room_templates_by_kind
        self._standalone_templates = list(room_templates_by_kind.get(RoomKind.STANDALONE, ()))
        if not self._standalone_templates:
            raise ValueError("Root room placement requires standalone room templates")
        all_templates: List[RoomTemplate] = []
        for templates in room_templates_by_kind.values():
            for template in templates:
                if template not in all_templates:
                    all_templates.append(template)
        self._all_templates: Tuple[RoomTemplate, ...] = tuple(all_templates)
        self._grower_context = GrowerContext(
            config=self.config,
            layout=self.layout,
            room_templates=self._all_templates,
            room_templates_by_kind=self.room_templates_by_kind,
        )
        self._corridor_length_distribution = self.config.initial_corridor_length_distribution

    def place_rooms(self) -> None:
        """Build a single connected component by growing corridors from unused ports."""
        target = self.config.num_rooms_to_place
        print(f"Attempting to place {target} rooms using corridor-driven growth...")

        root_room = self._place_initial_root()
        if root_room is None:
            raise ValueError("ERROR: failed to place initial root room.")

        initial_children = self._spawn_direct_links_recursive(root_room)
        processed_ports: Set[Tuple[int, int]] = set()
        port_queue: Deque[Tuple[int, int]] = deque()
        self._enqueue_available_ports(port_queue, [root_room, *initial_children])

        while len(self.layout.placed_rooms) < target and port_queue:
            room_idx, port_idx = port_queue.popleft()
            if (room_idx, port_idx) in processed_ports:
                continue
            processed_ports.add((room_idx, port_idx))

            room = self.layout.placed_rooms[room_idx]
            if port_idx in room.connected_port_indices:
                continue

            new_rooms = self._attempt_expand_via_corridor(room_idx, port_idx)
            if not new_rooms:
                continue
            self._enqueue_available_ports(port_queue, new_rooms)

        print(f"Successfully placed {len(self.layout.placed_rooms)} rooms.")

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

    def _enqueue_available_ports(
        self, port_queue: Deque[Tuple[int, int]], rooms: Iterable[PlacedRoom]
    ) -> None:
        for room in rooms:
            room_idx = room.index
            if room_idx < 0:
                continue
            for port_idx in room.get_available_port_indices():
                port_queue.append((room_idx, port_idx))

    def _sample_corridor_length(self) -> int:
        return self._corridor_length_distribution.sample()

    def _attempt_expand_via_corridor(
        self, room_idx: int, port_idx: int
    ) -> Optional[List[PlacedRoom]]:
        anchor_room = self.layout.placed_rooms[room_idx]
        anchor_ports = anchor_room.get_world_ports()
        if port_idx >= len(anchor_ports):
            return None
        anchor_port = anchor_ports[port_idx]
        if not anchor_port.widths:
            return None

        max_attempts = self.config.max_connected_placement_attempts
        for _ in range(max_attempts):
            corridor_length = self._sample_corridor_length()
            if corridor_length <= 0:
                continue
            plan = self._plan_corridor_expansion(
                anchor_room,
                port_idx,
                anchor_port,
                corridor_length,
            )
            if plan is None:
                continue
            return self._apply_corridor_plan(anchor_room, port_idx, plan)

        return None

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

        geometry = self._grower_context.build_corridor_geometry(
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
        exit_anchor = self._grower_context.port_exit_axis_value(anchor_port, axis_index)
        exit_candidate = self._grower_context.port_exit_axis_value(
            candidate_world_port, axis_index
        )
        if abs(exit_candidate - exit_anchor) != corridor_length:
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

        exit_anchor = self._grower_context.port_exit_axis_value(
            anchor_port, axis_index
        )
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

    def _apply_corridor_plan(
        self,
        anchor_room: PlacedRoom,
        anchor_port_idx: int,
        plan: CorridorExpansionPlan,
    ) -> List[PlacedRoom]:
        anchor_component = self.layout.normalize_room_component(anchor_room.index)
        self.layout.register_room(plan.room, anchor_component)
        plan.room.connected_port_indices.add(plan.room_port_index)
        anchor_room.connected_port_indices.add(anchor_port_idx)

        merged_component = self.layout.merge_components(
            anchor_component, plan.room.component_id
        )
        self.layout.set_room_component(anchor_room.index, merged_component)
        self.layout.set_room_component(plan.room.index, merged_component)
        corridor = Corridor(
            room_a_index=anchor_room.index,
            port_a_index=anchor_port_idx,
            room_b_index=plan.room.index,
            port_b_index=plan.room_port_index,
            width=plan.width,
            geometry=plan.geometry,
        )
        corridor_idx = self.layout.register_corridor(corridor, merged_component)
        self.layout.set_corridor_component(corridor_idx, merged_component)

        new_rooms = [plan.room]
        new_rooms.extend(self._spawn_direct_links_recursive(plan.room))
        return new_rooms

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
        standalone_templates = self.room_templates_by_kind.get(RoomKind.STANDALONE, ())

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
                # Special case: don't place a 1-door room directly linked to a 1-door room because we won't be able to connect them to the rest of the dungeon.
                if len(anchor_room.template.ports) == 1 and len(template.ports) == 1:
                    continue
                # Don't place the same room directly connected, to increase diversity of room types.
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
