"""DungeonGenerator orchestrates the three steps of the easylink algorithm."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Set, Tuple

from dungeon_config import DungeonConfig
from geometry import Direction, Rotation, TilePos, rotate_direction, VALID_ROTATIONS
from grower_context import GrowerContext
from models import RoomKind, PlacedRoom, RoomTemplate, WorldPort
from growers import (
    run_bent_room_to_room_grower,
    run_room_to_corridor_grower,
    run_room_to_room_grower,
)
from dungeon_layout import DungeonLayout


class DungeonGenerator:
    """Manages the overall process of generating a dungeon floor layout."""

    def __init__(self, config: DungeonConfig) -> None:
        self.config = config
        self.layout = DungeonLayout(config)

        self.room_templates = list(config.room_templates)
        self.standalone_room_templates = [rt for rt in self.room_templates if RoomKind.STANDALONE in rt.kinds]
        self.bend_room_templates = [rt for rt in self.room_templates if RoomKind.BEND in rt.kinds]
        self.t_junction_room_templates = [rt for rt in self.room_templates if RoomKind.T_JUNCTION in rt.kinds]
        self.four_way_room_templates = [rt for rt in self.room_templates if RoomKind.FOUR_WAY in rt.kinds]

    def generate(self) -> None:
        """Generates the dungeon, by invoking dungeon-growers."""
        # Note: This function is incomplete. Currently it runs our implemented growers in a fairly arbitrary order, mostly for testing. The final version will have more growers, and will include step 3 (counting components, deleting smaller components, and accepting or rejecting the final connected dungeon map).

        # Step 1: Place rooms, some with direct links
        self.place_rooms()
        if not self.layout.placed_rooms:
            raise ValueError("ERROR: no placed rooms.")

        context = GrowerContext(
            config=self.config,
            layout=self.layout,
            room_templates=self.room_templates,
            standalone_room_templates=self.standalone_room_templates,
            bend_room_templates=self.bend_room_templates,
            t_junction_room_templates=self.t_junction_room_templates,
            four_way_room_templates=self.four_way_room_templates,
        )

        # Step 2: Run our growers repeatedly.
        run_room_to_room_grower(context)
        num_created = 1
        while num_created > 0:
            num_created = run_room_to_corridor_grower(context, fill_probability=1)
            num_created += run_room_to_room_grower(context)
        num_created = run_bent_room_to_room_grower(context)
        # Re-run other growers, if we created new rooms or corridors.
        while num_created > 0:
            num_created = run_room_to_room_grower(context)
            num_created += run_room_to_corridor_grower(context, fill_probability=1)
        
        # Additional growers will be invoked here, then step 3.

    def get_component_summary(self) -> Dict[int, Dict[str, List[int]]]:
        """Return indices of rooms and corridors grouped by component id."""
        return self.layout.get_component_summary()

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

        try:
            offset_x, offset_y = self.config._door_macro_alignment_offsets[rotated_anchor_port.direction]
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
        """Sample n using the configured probability distribution."""
        items = list(self.config.direct_link_counts_probs.items())
        total = sum(p for _, p in items)
        if total <= 0:
            # Fallback to default if misconfigured
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
        if anchor_room.index < 0:
            raise ValueError("Anchor room must be registered before creating connections")
        anchor_component_id = self.layout.normalize_room_component(anchor_room.index)
        anchor_world_ports = anchor_room.get_world_ports()
        available_anchor_indices = anchor_room.get_available_port_indices()
        random.shuffle(available_anchor_indices)

        # Try each available port in random order
        for anchor_idx in available_anchor_indices:
            awp = anchor_world_ports[anchor_idx]
            ax, ay = awp.pos
            direction = awp.direction
            dx, dy = direction.dx, direction.dy
            # Target position for the new room's connecting port so the rooms are adjacent
            target_port_pos = (ax + dx, ay + dy)
            # Candidate attempt loop
            for _ in range(self.config.max_connected_placement_attempts):
                template = random.choices(
                    self.standalone_room_templates, weights=[rt.direct_weight for rt in self.standalone_room_templates]
                )[0]
                rotation = Rotation.random()
                temp_room = PlacedRoom(template, 0, 0, rotation)
                rot_ports = temp_room.get_world_ports()
                # Find ports facing opposite direction
                compatible_port_indices = [
                    i
                    for i, p in enumerate(rot_ports)
                    if p.direction == direction.opposite() and (p.widths & awp.widths)
                ]
                if not compatible_port_indices:
                    continue
                cand_idx = random.choice(compatible_port_indices)
                cand_port = rot_ports[cand_idx]
                # Compute top-left for new room so its chosen port lands at target_port_pos
                rpx, rpy = cand_port.pos
                nx = int(round(target_port_pos[0] - rpx))
                ny = int(round(target_port_pos[1] - rpy))
                candidate = PlacedRoom(template, nx, ny, rotation)
                if self.layout.is_valid_placement_with_anchor(candidate, anchor_room):
                    self.layout.register_room(candidate, anchor_component_id)
                    anchor_room.connected_port_indices.add(anchor_idx)
                    candidate.connected_port_indices.add(cand_idx)
                    return candidate
        # No success on any port
        return None

    def _spawn_direct_links_recursive(self, from_room: PlacedRoom) -> int:
        """Recursively try to place directly-connected rooms from from_room."""
        rooms_placed = 0
        n = self._sample_num_direct_links()
        for _ in range(n):
            child = self._attempt_place_connected_to(from_room)
            if child is not None:
                rooms_placed += 1
                rooms_placed += self._spawn_direct_links_recursive(child)
        return rooms_placed

    def place_rooms(self) -> None:
        """Implements step 1: Randomly place rooms with macro-grid aligned ports."""
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

            #print(f"Placed root room number {root_room_index} after {attempt} failed attempts.")
            #print(f"Placed root room is {placed_room.template.name} at {(placed_room.x, placed_room.y)}")

        print(f"Successfully placed {placed_count} rooms.")
