"""Utilities for populating a fresh layout with the initial set of rooms."""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from dungeon_config import DungeonConfig
from dungeon_layout import DungeonLayout
from geometry import Direction, Rotation, rotate_direction, VALID_ROTATIONS
from models import PlacedRoom, RoomKind, RoomTemplate


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

    def place_rooms(self) -> None:
        """Randomly place root rooms and grow their direct links."""
        print(f"Attempting to place {self.config.num_rooms_to_place} rooms...")
        placed_count = 0
        consecutive_limit_exceeded = 0

        for root_room_index in range(self.config.num_rooms_to_place):
            if placed_count >= self.config.num_rooms_to_place:
                break
            if consecutive_limit_exceeded >= self.config.max_consecutive_limit_failures:
                print(
                    f"Exceeded attempt limit {self.config.max_consecutive_limit_failures} consecutive times, aborting further placement."
                )
                break

            placed_room: Optional[PlacedRoom] = None
            attempt = 0
            for attempt in range(20):
                macro_x, macro_y = self._random_macro_grid_point()
                placement_category, side_proximities = self._describe_macro_position(macro_x, macro_y)

                if placement_category == "middle":
                    template_weights = [rt.root_weight_middle for rt in self._standalone_templates]
                elif placement_category == "edge":
                    template_weights = [rt.root_weight_edge for rt in self._standalone_templates]
                else:
                    template_weights = [rt.root_weight_intermediate for rt in self._standalone_templates]

                if not any(weight > 0 for weight in template_weights):
                    template_weights = [1.0 for _ in self._standalone_templates]

                template = random.choices(self._standalone_templates, weights=template_weights)[0]
                rotation = self._select_root_rotation(template, placement_category, side_proximities)
                candidate_room = self._build_root_room_candidate(template, rotation, macro_x, macro_y)
                if self.layout.is_valid_placement(candidate_room):
                    placed_room = candidate_room
                    break

            if placed_room is None:
                consecutive_limit_exceeded += 1
                print(f"Exceeded attempt limit when placing root room number {root_room_index}.")
                continue

            component_id = self.layout.new_component_id()
            self.layout.register_room(placed_room, component_id)
            placed_count += 1
            placed_count += self._spawn_direct_links_recursive(placed_room)
            consecutive_limit_exceeded = 0

        print(f"Successfully placed {placed_count} rooms.")

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

    def _spawn_direct_links_recursive(self, from_room: PlacedRoom) -> int:
        rooms_placed = 0
        num_links = self._sample_num_direct_links()
        for _ in range(num_links):
            child = self._attempt_place_connected_to(from_room)
            if child is not None:
                rooms_placed += 1
                rooms_placed += self._spawn_direct_links_recursive(child)
        return rooms_placed
