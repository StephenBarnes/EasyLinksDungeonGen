"""Room placement helpers extracted from the monolithic generator."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from dungeon_constants import (
    DOOR_MACRO_ALIGNMENT_OFFSETS,
    MACRO_GRID_SIZE,
    MAX_CONNECTED_PLACEMENT_ATTEMPTS,
    MAX_CONSECUTIVE_LIMIT_FAILURES,
    VALID_ROTATIONS,
)
from dungeon_geometry import rotate_direction
from dungeon_models import PlacedRoom, RoomTemplate
from dungeon_mixin_contract import DungeonContract
import random


class RoomPlacementMixin(DungeonContract):
    """Shared routines implementing the algorithm's room placement logic."""

    def _expand_bounds(self, bounds: Tuple[int, int, int, int], margin: int) -> Tuple[int, int, int, int]:
        x, y, w, h = bounds
        return x - margin, y - margin, w + 2 * margin, h + 2 * margin

    def _rects_overlap(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        if ax + aw <= bx or bx + bw <= ax:
            return False
        if ay + ah <= by or by + bh <= ay:
            return False
        return True

    def _is_in_bounds(self, room: PlacedRoom) -> bool:
        x, y, w, h = room.get_bounds()
        return 0 <= x and 0 <= y and x + w <= self.width and y + h <= self.height

    def _rooms_overlap(self, candidate: PlacedRoom, existing: PlacedRoom, margin: int) -> bool:
        expanded_candidate = self._expand_bounds(candidate.get_bounds(), margin)
        return self._rects_overlap(expanded_candidate, existing.get_bounds())

    def _is_valid_room_position(self, new_room: PlacedRoom, anchor_room: Optional[PlacedRoom]) -> bool:
        if not self._is_in_bounds(new_room):
            return False

        for room in self.placed_rooms:
            margin = 0 if anchor_room is not None and room is anchor_room else self.min_room_separation
            if self._rooms_overlap(new_room, room, margin):
                return False
        return True

    def _is_valid_placement(self, new_room: PlacedRoom) -> bool:
        """Checks if a new room is in bounds and doesn't overlap existing rooms."""
        return self._is_valid_room_position(new_room, None)

    def _is_valid_placement_with_anchor(self, new_room: PlacedRoom, anchor_room: PlacedRoom) -> bool:
        """Validate placement allowing edge-adjacent contact with the anchor room only."""
        return self._is_valid_room_position(new_room, anchor_room)

    def _random_rotation(self) -> int:
        return random.choice(VALID_ROTATIONS)

    def _random_macro_grid_point(self) -> Tuple[int, int]:
        max_macro_x = (self.width // MACRO_GRID_SIZE) - 1
        max_macro_y = (self.height // MACRO_GRID_SIZE) - 1
        if max_macro_x <= 1 or max_macro_y <= 1:
            raise ValueError("Grid too small to place rooms with macro-grid alignment")

        macro_x = random.randint(1, max_macro_x - 1) * MACRO_GRID_SIZE
        macro_y = random.randint(1, max_macro_y - 1) * MACRO_GRID_SIZE
        return macro_x, macro_y

    def _build_root_room_candidate(
        self, template: RoomTemplate, rotation: int, macro_x: int, macro_y: int
    ) -> PlacedRoom:
        anchor_port_index = random.randrange(len(template.ports))

        rotated_room = PlacedRoom(template, 0, 0, rotation)
        rotated_ports = rotated_room.get_world_ports()
        rotated_anchor_port = rotated_ports[anchor_port_index]

        try:
            offset_x, offset_y = DOOR_MACRO_ALIGNMENT_OFFSETS[rotated_anchor_port.direction]
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
        if ratio <= 0.3:
            return "close"
        if ratio >= 0.4:
            return "far"
        return "intermediate"

    def _describe_macro_position(self, macro_x: int, macro_y: int) -> Tuple[str, Dict[str, str]]:
        side_proximities = {
            "left": self._categorize_side_distance(macro_x, self.width),
            "right": self._categorize_side_distance(self.width - macro_x, self.width),
            "top": self._categorize_side_distance(macro_y, self.height),
            "bottom": self._categorize_side_distance(self.height - macro_y, self.height),
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
    ) -> int:
        preferred_dir = template.preferred_center_facing_dir
        if placement_category != "edge" or preferred_dir is None:
            return self._random_rotation()

        inward_directions: List[Tuple[int, int]] = []
        if side_proximities.get("left") == "close":
            inward_directions.append((1, 0))
        if side_proximities.get("right") == "close":
            inward_directions.append((-1, 0))
        if side_proximities.get("top") == "close":
            inward_directions.append((0, 1))
        if side_proximities.get("bottom") == "close":
            inward_directions.append((0, -1))

        if not inward_directions:
            return self._random_rotation()

        rotation_weights: List[float] = []
        pdx, pdy = preferred_dir
        for rotation in VALID_ROTATIONS:
            rotated_dir = rotate_direction(pdx, pdy, rotation)
            weight = 1.0 if rotated_dir in inward_directions else 0.0
            rotation_weights.append(weight)

        return random.choices(VALID_ROTATIONS, weights=rotation_weights)[0]

    def _sample_num_direct_links(self) -> int:
        """Sample n using the configured probability distribution."""
        items = list(self.direct_link_counts_probs.items())
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
        """
        Attempt to place a new room that directly connects to one of the available
        ports on anchor_room. Chooses a random template, rotation, and compatible
        port facing opposite the anchor port. Returns the new PlacedRoom on success,
        otherwise None.
        """
        anchor_world_ports = anchor_room.get_world_ports()
        available_anchor_indices = anchor_room.get_available_port_indices()
        random.shuffle(available_anchor_indices)

        for anchor_idx in available_anchor_indices:
            awp = anchor_world_ports[anchor_idx]
            ax, ay = awp.pos
            dx, dy = awp.direction
            target_port_pos = (ax + dx, ay + dy)
            for _ in range(MAX_CONNECTED_PLACEMENT_ATTEMPTS):
                template = random.choices(
                    self.standalone_room_templates, weights=[rt.direct_weight for rt in self.standalone_room_templates]
                )[0]
                rotation = self._random_rotation()
                temp_room = PlacedRoom(template, 0, 0, rotation)
                rot_ports = temp_room.get_world_ports()
                compatible_port_indices = [
                    i for i, p in enumerate(rot_ports) if p.direction == (-dx, -dy) and (p.widths & awp.widths)
                ]
                if not compatible_port_indices:
                    continue
                cand_idx = random.choice(compatible_port_indices)
                cand_port = rot_ports[cand_idx]
                rpx, rpy = cand_port.pos
                nx = int(round(target_port_pos[0] - rpx))
                ny = int(round(target_port_pos[1] - rpy))
                candidate = PlacedRoom(template, nx, ny, rotation)
                if self._is_valid_placement_with_anchor(candidate, anchor_room):
                    self._register_room(candidate, anchor_room.component_id)
                    anchor_room.connected_port_indices.add(anchor_idx)
                    candidate.connected_port_indices.add(cand_idx)
                    return candidate
        return None

    def _spawn_direct_links_recursive(self, from_room: PlacedRoom) -> int:
        """Recursively try to place 0-2 directly-connected rooms from from_room."""
        rooms_placed = 0
        n = self._sample_num_direct_links()
        for _ in range(n):
            child = self._attempt_place_connected_to(from_room)
            if child is not None:
                rooms_placed += 1
                rooms_placed += self._spawn_direct_links_recursive(child)
        return rooms_placed

    def place_rooms(self) -> None:
        """Implements Step 1: Randomly place rooms with macro-grid aligned ports."""
        print(f"Attempting to place {self.num_rooms_to_place} rooms...")
        placed_count = 0
        consecutive_limit_exceeded = 0

        for root_room_index in range(self.num_rooms_to_place):
            if placed_count >= self.num_rooms_to_place:
                break
            if consecutive_limit_exceeded >= MAX_CONSECUTIVE_LIMIT_FAILURES:
                print(
                    f"Exceeded attempt limit {MAX_CONSECUTIVE_LIMIT_FAILURES} consecutive times, "
                    "aborting further placement."
                )
                break

            placed_room: Optional[PlacedRoom] = None
            attempt = 0
            for attempt in range(20):
                macro_x, macro_y = self._random_macro_grid_point()
                placement_category, side_proximities = self._describe_macro_position(macro_x, macro_y)

                if placement_category == "middle":
                    template_weights = [rt.root_weight_middle for rt in self.standalone_room_templates]
                elif placement_category == "edge":
                    template_weights = [rt.root_weight_edge for rt in self.standalone_room_templates]
                else:
                    template_weights = [rt.root_weight_intermediate for rt in self.standalone_room_templates]

                if not any(weight > 0 for weight in template_weights):
                    template_weights = [1.0 for _ in self.standalone_room_templates]

                template = random.choices(self.standalone_room_templates, weights=template_weights)[0]
                rotation = self._select_root_rotation(template, placement_category, side_proximities)
                candidate_room = self._build_root_room_candidate(template, rotation, macro_x, macro_y)
                if self._is_valid_placement(candidate_room):
                    placed_room = candidate_room
                    break

            if placed_room is None:
                consecutive_limit_exceeded += 1
                print(f"Exceeded attempt limit when placing root room number {root_room_index}.")
                continue

            component_id = self._new_component_id()
            self._register_room(placed_room, component_id)
            placed_count += 1
            placed_count += self._spawn_direct_links_recursive(placed_room)
            consecutive_limit_exceeded = 0

        print(f"Successfully placed {placed_count} rooms.")
