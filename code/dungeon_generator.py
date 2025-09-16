"""DungeonGenerator orchestrates the three implemented algorithm steps."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Set, Tuple

from dungeon_constants import (
    DOOR_MACRO_ALIGNMENT_OFFSETS,
    MACRO_GRID_SIZE,
    MAX_CONNECTED_PLACEMENT_ATTEMPTS,
    VALID_ROTATIONS,
    MAX_CONSECUTIVE_LIMIT_FAILURES,
)
from dungeon_geometry import rotate_direction
from dungeon_models import (
    RoomKind,
    Corridor,
    CorridorGeometry,
    FourWayIntersection,
    PlacedRoom,
    RightAngleJoint,
    RoomTemplate,
    TJunction,
    WorldPort,
)


class DungeonGenerator:
    """Manages the overall process of generating a dungeon floor layout."""

    def __init__(
        self,
        width: int,
        height: int,
        room_templates: List[RoomTemplate],
        direct_link_counts_probs: dict[int, float],
        num_rooms_to_place: int,
        min_room_separation: int,
        min_rooms_required: int = 6,
    ) -> None:
        self.width = width
        self.height = height

        self.room_templates = list(room_templates)
        self.standalone_room_templates = list(rt for rt in room_templates if RoomKind.STANDALONE in rt.kinds)
        self.bend_room_templates = list(rt for rt in room_templates if RoomKind.BEND in rt.kinds)
        self.t_junction_room_templates = list(rt for rt in room_templates if RoomKind.T_JUNCTION in rt.kinds)
        self.four_way_room_templates = list(rt for rt in room_templates if RoomKind.FOUR_WAY in rt.kinds)

        self.num_rooms_to_place = num_rooms_to_place
        # Minimum empty tiles between room bounding boxes, unless they connect at ports.
        self.min_room_separation = min_room_separation
        self.min_rooms_required = min_rooms_required
        self.placed_rooms: List[PlacedRoom] = []
        self.room_components: List[int] = []
        self.corridors: List[Corridor] = []
        self.corridor_components: List[int] = []
        self.corridor_tiles: Set[Tuple[int, int]] = set()
        self.corridor_tile_index: Dict[Tuple[int, int], List[int]] = {}
        self.right_angle_joints: List[RightAngleJoint] = []
        self.right_angle_components: List[int] = []
        self.right_angle_tiles: Set[Tuple[int, int]] = set()
        self.t_junctions: List[TJunction] = []
        self.t_junction_components: List[int] = []
        self.t_junction_tiles: Set[Tuple[int, int]] = set()
        self.four_way_intersections: List[FourWayIntersection] = []
        self.four_way_components: List[int] = []
        self.four_way_tiles: Set[Tuple[int, int]] = set()
        self.grid = [[" " for _ in range(width)] for _ in range(height)]
        # Probability distribution for number of immediate direct links per room
        # Example: {0: 0.4, 1: 0.3, 2: 0.3}
        self.direct_link_counts_probs = dict(direct_link_counts_probs)
        self._next_component_id = 0

    @staticmethod
    def _expand_bounds(bounds: Tuple[int, int, int, int], margin: int) -> Tuple[int, int, int, int]:
        x, y, w, h = bounds
        return x - margin, y - margin, w + 2 * margin, h + 2 * margin

    @staticmethod
    def _rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
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

    def _clear_grid(self) -> None:
        for row in self.grid:
            for x in range(self.width):
                row[x] = " "

    def _new_component_id(self) -> int:
        component_id = self._next_component_id
        self._next_component_id += 1
        return component_id

    def _register_room(self, room: PlacedRoom, component_id: int) -> None:
        room.component_id = component_id
        self.placed_rooms.append(room)
        self.room_components.append(component_id)

    def _register_corridor(self, corridor: Corridor, component_id: int) -> int:
        corridor.component_id = component_id
        self.corridors.append(corridor)
        self.corridor_components.append(component_id)
        new_index = len(self.corridors) - 1
        self._add_corridor_tiles(new_index)
        return new_index

    def _register_right_angle_joint(self, joint: RightAngleJoint, component_id: int) -> int:
        joint.component_id = component_id
        self.right_angle_joints.append(joint)
        self.right_angle_components.append(component_id)
        idx = len(self.right_angle_joints) - 1
        for tile in joint.tiles:
            self.right_angle_tiles.add(tile)
        return idx

    def _register_t_junction(self, junction: TJunction, component_id: int) -> int:
        junction.component_id = component_id
        self.t_junctions.append(junction)
        self.t_junction_components.append(component_id)
        idx = len(self.t_junctions) - 1
        for tile in junction.tiles:
            self.t_junction_tiles.add(tile)
        return idx

    def _register_four_way(self, intersection: FourWayIntersection, component_id: int) -> int:
        intersection.component_id = component_id
        self.four_way_intersections.append(intersection)
        self.four_way_components.append(component_id)
        idx = len(self.four_way_intersections) - 1
        for tile in intersection.tiles:
            self.four_way_tiles.add(tile)
        return idx

    def _remove_corridor_tiles(self, corridor_idx: int) -> None:
        corridor = self.corridors[corridor_idx]
        for tile in corridor.geometry.tiles:
            owners = self.corridor_tile_index.get(tile)
            if owners is not None and corridor_idx in owners:
                owners[:] = [idx for idx in owners if idx != corridor_idx]
                if not owners:
                    del self.corridor_tile_index[tile]
            # Only remove from corridor_tiles if no other corridor uses it
            if tile not in self.corridor_tile_index:
                self.corridor_tiles.discard(tile)

    def _add_corridor_tiles(self, corridor_idx: int) -> None:
        corridor = self.corridors[corridor_idx]
        for tile in corridor.geometry.tiles:
            self.corridor_tiles.add(tile)
            self.corridor_tile_index.setdefault(tile, []).append(corridor_idx)

    def _merge_components(self, *component_ids: int) -> int:
        valid_ids = {cid for cid in component_ids if cid >= 0}
        if not valid_ids:
            raise ValueError("Cannot merge empty component set")

        target = min(valid_ids)

        for idx, comp in enumerate(self.room_components):
            if comp in valid_ids:
                self.room_components[idx] = target
                self.placed_rooms[idx].component_id = target

        for idx, comp in enumerate(self.corridor_components):
            if comp in valid_ids:
                self.corridor_components[idx] = target
                self.corridors[idx].component_id = target

        for idx, comp in enumerate(self.right_angle_components):
            if comp in valid_ids:
                self.right_angle_components[idx] = target
                self.right_angle_joints[idx].component_id = target

        for idx, comp in enumerate(self.t_junction_components):
            if comp in valid_ids:
                self.t_junction_components[idx] = target
                self.t_junctions[idx].component_id = target

        for idx, comp in enumerate(self.four_way_components):
            if comp in valid_ids:
                self.four_way_components[idx] = target
                self.four_way_intersections[idx].component_id = target

        return target

    def get_component_summary(self) -> Dict[int, Dict[str, List[int]]]:
        """Return indices of rooms and corridors grouped by component id."""
        summary: Dict[int, Dict[str, List[int]]] = {}

        def ensure_entry(component_id: int) -> Dict[str, List[int]]:
            return summary.setdefault(
                component_id,
                {
                    "rooms": [],
                    "corridors": [],
                    "right_angle_joints": [],
                    "t_junctions": [],
                    "four_way_intersections": [],
                },
            )

        for idx, component_id in enumerate(self.room_components):
            comp_summary = ensure_entry(component_id)
            comp_summary["rooms"].append(idx)

        for idx, component_id in enumerate(self.corridor_components):
            comp_summary = ensure_entry(component_id)
            comp_summary["corridors"].append(idx)

        for idx, component_id in enumerate(self.right_angle_components):
            comp_summary = ensure_entry(component_id)
            comp_summary["right_angle_joints"].append(idx)

        for idx, component_id in enumerate(self.t_junction_components):
            comp_summary = ensure_entry(component_id)
            comp_summary["t_junctions"].append(idx)

        for idx, component_id in enumerate(self.four_way_components):
            comp_summary = ensure_entry(component_id)
            comp_summary["four_way_intersections"].append(idx)

        return summary

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
        anchor_world_ports = anchor_room.get_world_ports()
        available_anchor_indices = anchor_room.get_available_port_indices()
        random.shuffle(available_anchor_indices)

        # Try each available port in random order
        for anchor_idx in available_anchor_indices:
            awp = anchor_world_ports[anchor_idx]
            ax, ay = awp.pos
            dx, dy = awp.direction
            # Target position for the new room's connecting port so the rooms are adjacent
            target_port_pos = (ax + dx, ay + dy)
            # Candidate attempt loop
            for _ in range(MAX_CONNECTED_PLACEMENT_ATTEMPTS):
                template = random.choices(
                    self.standalone_room_templates, weights=[rt.direct_weight for rt in self.standalone_room_templates]
                )[0]
                rotation = self._random_rotation()
                temp_room = PlacedRoom(template, 0, 0, rotation)
                rot_ports = temp_room.get_world_ports()
                # Find ports facing opposite direction
                compatible_port_indices = [
                    i for i, p in enumerate(rot_ports) if p.direction == (-dx, -dy) and (p.widths & awp.widths)
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
                if self._is_valid_placement_with_anchor(candidate, anchor_room):
                    self._register_room(candidate, anchor_room.component_id)
                    anchor_room.connected_port_indices.add(anchor_idx)
                    candidate.connected_port_indices.add(cand_idx)
                    return candidate
        # No success on any port
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

    def _build_room_tile_lookup(self) -> Dict[Tuple[int, int], int]:
        """Map each room tile to its owning room index for collision checks."""
        tile_to_room: Dict[Tuple[int, int], int] = {}
        for idx, room in enumerate(self.placed_rooms):
            x, y, w, h = room.get_bounds()
            for ty in range(y, y + h):
                for tx in range(x, x + w):
                    tile_to_room[(tx, ty)] = idx
        return tile_to_room

    def _build_corridor_tile_lookup(self) -> Dict[Tuple[int, int], List[int]]:
        """Map corridor tiles to the corridors occupying them."""
        tile_to_corridors: Dict[Tuple[int, int], List[int]] = {}
        for corridor_idx, corridor in enumerate(self.corridors):
            for tile in corridor.geometry.tiles:
                tile_to_corridors.setdefault(tile, []).append(corridor_idx)
        return tile_to_corridors

    @staticmethod
    def _port_exit_axis_value(port: WorldPort, axis_index: int) -> int:
        """Return the first tile outside the room along the port's facing axis."""
        axis_values = [coord[axis_index] for coord in port.tiles]
        facing = port.direction[axis_index]
        if facing > 0:
            boundary = max(axis_values)
        else:
            boundary = min(axis_values)
        return boundary + facing

    @staticmethod
    def _corridor_cross_coords(center: float, width: int) -> List[int]:
        """Compute the perpendicular tile coordinates for a corridor with given width."""
        if width <= 0:
            raise ValueError("Corridor width must be positive")
        if width % 2 != 0:
            raise ValueError("Corridor widths are expected to be even")
        half = width // 2
        start = int(math.floor(center - (half - 0.5)))
        return list(range(start, start + width))

    def _build_segment_geometry(
        self,
        axis_index: int,
        start_axis: int,
        end_axis: int,
        cross_coords: Tuple[int, ...],
    ) -> Optional[CorridorGeometry]:
        """Construct geometry for a straight corridor segment between two axis values."""
        if start_axis == end_axis:
            return None
        if not cross_coords:
            return None

        step = 1 if end_axis > start_axis else -1
        axis_values = list(range(start_axis, end_axis, step))
        if not axis_values:
            return None

        tiles: List[Tuple[int, int]] = []
        for axis_value in axis_values:
            for cross in cross_coords:
                if axis_index == 0:
                    x, y = axis_value, cross
                else:
                    x, y = cross, axis_value
                if not (0 <= x < self.width and 0 <= y < self.height):
                    return None
                tiles.append((x, y))

        return CorridorGeometry(
            tiles=tuple(tiles),
            axis_index=axis_index,
            port_axis_values=(start_axis, end_axis),
            cross_coords=cross_coords,
        )

    @staticmethod
    def _corridor_cross_from_geometry(
        geometry: CorridorGeometry, axis_index: int
    ) -> Tuple[int, ...]:
        if geometry.cross_coords:
            return geometry.cross_coords
        cross_set: Set[int] = set()
        for tile in geometry.tiles:
            cross_set.add(tile[1 - axis_index])
        if not cross_set:
            raise ValueError("Unable to infer cross coordinates for corridor geometry")
        return tuple(sorted(cross_set))

    def _corridor_endpoint_info(
        self, corridor: Corridor, endpoint: str
    ) -> Tuple[Optional[int], Optional[int], Optional[Tuple[str, int]]]:
        if endpoint == "a":
            return corridor.room_a_index, corridor.port_a_index, corridor.joint_a
        return corridor.room_b_index, corridor.port_b_index, corridor.joint_b

    def _set_corridor_endpoint_info(
        self,
        corridor: Corridor,
        endpoint: str,
        room_index: Optional[int],
        port_index: Optional[int],
        joint: Optional[Tuple[str, int]],
    ) -> None:
        if endpoint == "a":
            corridor.room_a_index = room_index
            corridor.port_a_index = port_index
            corridor.joint_a = joint
        else:
            corridor.room_b_index = room_index
            corridor.port_b_index = port_index
            corridor.joint_b = joint

    def _split_corridor_for_junction(
        self,
        corridor_idx: int,
        junction_axis: int,
        joint_ref: Tuple[str, int],
    ) -> List[int]:
        """Split a straight corridor at a junction and return involved corridor indices."""
        corridor = self.corridors[corridor_idx]
        geometry = corridor.geometry

        axis_index = geometry.axis_index
        if axis_index is None:
            return [corridor_idx]

        start_axis, end_axis = geometry.port_axis_values
        cross_coords = self._corridor_cross_from_geometry(geometry, axis_index)

        conn_a = self._corridor_endpoint_info(corridor, "a")
        conn_b = self._corridor_endpoint_info(corridor, "b")

        # Remove existing tiles before reassigning geometries.
        self._remove_corridor_tiles(corridor_idx)

        segment_a = self._build_segment_geometry(axis_index, start_axis, junction_axis, cross_coords)
        segment_b = self._build_segment_geometry(axis_index, end_axis, junction_axis, cross_coords)

        new_indices: List[int] = []

        if segment_a is not None:
            corridor.geometry = segment_a
            self._set_corridor_endpoint_info(
                corridor,
                "a",
                conn_a[0],
                conn_a[1],
                conn_a[2],
            )
            self._set_corridor_endpoint_info(corridor, "b", None, None, joint_ref)
            self._add_corridor_tiles(corridor_idx)
            new_indices.append(corridor_idx)
        else:
            # No tiles remain on the A side; treat the existing corridor as the B side segment.
            if segment_b is None:
                return [corridor_idx]
            corridor.geometry = segment_b
            self._set_corridor_endpoint_info(
                corridor,
                "a",
                conn_b[0],
                conn_b[1],
                conn_b[2],
            )
            self._set_corridor_endpoint_info(corridor, "b", None, None, joint_ref)
            self._add_corridor_tiles(corridor_idx)
            new_indices.append(corridor_idx)
            return new_indices

        if segment_b is not None:
            new_corridor = Corridor(
                room_a_index=conn_b[0],
                port_a_index=conn_b[1],
                room_b_index=None,
                port_b_index=None,
                width=corridor.width,
                geometry=segment_b,
                component_id=corridor.component_id,
                joint_a=conn_b[2],
                joint_b=joint_ref,
            )
            new_idx = self._register_corridor(new_corridor, corridor.component_id)
            new_indices.append(new_idx)
        else:
            # Endpoint B is directly at the junction.
            self._set_corridor_endpoint_info(corridor, "b", None, None, joint_ref)

        return new_indices

    def _replace_joint_reference(
        self,
        corridor_indices: List[int],
        old_ref: Tuple[str, int],
        new_ref: Tuple[str, int],
    ) -> None:
        for idx in corridor_indices:
            corridor = self.corridors[idx]
            if corridor.joint_a == old_ref:
                corridor.joint_a = new_ref
            if corridor.joint_b == old_ref:
                corridor.joint_b = new_ref

    def _build_corridor_geometry(
        self,
        room_index_a: int,
        port_a: WorldPort,
        room_index_b: int,
        port_b: WorldPort,
        width: int,
        tile_to_room: Dict[Tuple[int, int], int],
    ) -> Optional[CorridorGeometry]:
        """Return the carved tiles for a straight corridor if it's valid."""
        dx1, dy1 = port_a.direction
        dx2, dy2 = port_b.direction
        if dx1 != -dx2 or dy1 != -dy2:
            return None

        axis_index = 0 if dx1 != 0 else 1
        if axis_index == 0:
            if not math.isclose(port_a.pos[1], port_b.pos[1], abs_tol=1e-6):
                return None
            center = port_a.pos[1]
        else:
            if not math.isclose(port_a.pos[0], port_b.pos[0], abs_tol=1e-6):
                return None
            center = port_a.pos[0]

        exit_a = self._port_exit_axis_value(port_a, axis_index)
        exit_b = self._port_exit_axis_value(port_b, axis_index)
        if exit_a == exit_b:
            return None

        axis_start = min(exit_a, exit_b)
        axis_end = max(exit_a, exit_b)
        cross_coords = self._corridor_cross_coords(center, width)

        tiles: List[Tuple[int, int]] = []
        for axis_value in range(axis_start, axis_end + 1):
            for cross_value in cross_coords:
                if axis_index == 0:
                    x, y = axis_value, cross_value
                else:
                    x, y = cross_value, axis_value
                if not (0 <= x < self.width and 0 <= y < self.height):
                    return None
                if (x, y) in tile_to_room:
                    return None
                tiles.append((x, y))

        allowed_axis_by_room = {
            room_index_a: exit_a,
            room_index_b: exit_b,
        }

        for tile in tiles:
            axis_value = tile[axis_index]
            neighbors = (
                (tile[0] + 1, tile[1]),
                (tile[0] - 1, tile[1]),
                (tile[0], tile[1] + 1),
                (tile[0], tile[1] - 1),
            )
            for neighbor in neighbors:
                neighbor_room = tile_to_room.get(neighbor)
                if neighbor_room is None:
                    continue
                allowed_axis = allowed_axis_by_room.get(neighbor_room)
                if allowed_axis is not None and axis_value == allowed_axis:
                    continue
                return None

        return CorridorGeometry(
            tiles=tuple(tiles),
            axis_index=axis_index,
            port_axis_values=(exit_a, exit_b),
            cross_coords=tuple(cross_coords),
        )

    def _build_t_junction_geometry(
        self,
        room_index: int,
        port: WorldPort,
        width: int,
        tile_to_room: Dict[Tuple[int, int], int],
        tile_to_corridors: Dict[Tuple[int, int], List[int]],
    ) -> Optional[Tuple[CorridorGeometry, int, Tuple[Tuple[int, int], ...]]]:
        """Attempt to carve a straight corridor from a port to an existing corridor."""
        axis_index = 0 if port.direction[0] != 0 else 1
        direction = port.direction[axis_index]
        if direction == 0:
            return None

        cross_center = port.pos[1] if axis_index == 0 else port.pos[0]
        cross_coords = self._corridor_cross_coords(cross_center, width)
        exit_axis_value = self._port_exit_axis_value(port, axis_index)

        axis_value = exit_axis_value
        path_tiles: List[Tuple[int, int]] = []
        max_steps = max(self.width, self.height) + 1
        steps = 0

        while True:
            tiles_for_step: List[Tuple[int, int]] = []
            for cross in cross_coords:
                if axis_index == 0:
                    x, y = axis_value, cross
                else:
                    x, y = cross, axis_value

                if not (0 <= x < self.width and 0 <= y < self.height):
                    return None
                tiles_for_step.append((x, y))

            all_corridor = all(tile in self.corridor_tiles for tile in tiles_for_step)
            if all_corridor:
                intersecting_indices: Optional[Set[int]] = None
                for tile in tiles_for_step:
                    indices = set(tile_to_corridors.get(tile, []))
                    if not indices:
                        intersecting_indices = set()
                        break
                    if intersecting_indices is None:
                        intersecting_indices = indices
                    else:
                        intersecting_indices &= indices
                    if not intersecting_indices:
                        break

                if not intersecting_indices:
                    return None

                chosen_idx: Optional[int] = None
                for idx in sorted(intersecting_indices):
                    candidate = self.corridors[idx]
                    if (
                        candidate.geometry.axis_index is not None
                        and candidate.geometry.axis_index == axis_index
                    ):
                        continue
                    chosen_idx = idx
                    break

                if chosen_idx is None:
                    return None

                geometry = CorridorGeometry(
                    tiles=tuple(path_tiles),
                    axis_index=axis_index,
                    port_axis_values=(exit_axis_value, axis_value),
                    cross_coords=tuple(cross_coords),
                )
                return geometry, chosen_idx, tuple(tiles_for_step)

            if any(tile in self.corridor_tiles for tile in tiles_for_step):
                return None

            for tile in tiles_for_step:
                if tile in tile_to_room:
                    return None
                neighbors = (
                    (tile[0] + 1, tile[1]),
                    (tile[0] - 1, tile[1]),
                    (tile[0], tile[1] + 1),
                    (tile[0], tile[1] - 1),
                )
                for neighbor in neighbors:
                    neighbor_room = tile_to_room.get(neighbor)
                    if neighbor_room is None:
                        continue
                    if neighbor_room != room_index:
                        return None

            path_tiles.extend(tiles_for_step)

            axis_value += direction
            steps += 1
            if steps > max_steps:
                return None

    def _plan_right_angle_connection(
        self,
        port_a: WorldPort,
        port_b: WorldPort,
        width: int,
        tile_to_room: Dict[Tuple[int, int], int],
    ) -> Optional[Tuple[CorridorGeometry, CorridorGeometry, Tuple[Tuple[int, int], ...]]]:
        """Return straight legs and joint tiles for a right-angled connection."""
        dot = port_a.direction[0] * port_b.direction[0] + port_a.direction[1] * port_b.direction[1]
        if dot != 0:
            return None

        axis_a = 0 if port_a.direction[0] != 0 else 1
        axis_b = 0 if port_b.direction[0] != 0 else 1

        if axis_a == axis_b:
            return None

        if axis_a == 0:
            horizontal_port = port_a
            vertical_port = port_b
        else:
            horizontal_port = port_b
            vertical_port = port_a

        horizontal_exit = self._port_exit_axis_value(horizontal_port, 0)
        vertical_exit = self._port_exit_axis_value(vertical_port, 1)

        horizontal_direction = horizontal_port.direction[0]
        vertical_direction = vertical_port.direction[1]
        if horizontal_direction == 0 or vertical_direction == 0:
            return None

        horizontal_cross = self._corridor_cross_coords(horizontal_port.pos[1], width)
        vertical_cross = self._corridor_cross_coords(vertical_port.pos[0], width)

        bend_y_min = horizontal_cross[0]
        bend_y_max = horizontal_cross[-1]
        bend_x_min = vertical_cross[0]
        bend_x_max = vertical_cross[-1]

        if horizontal_direction > 0:
            if bend_x_min <= horizontal_exit:
                return None
            horizontal_axis_values = range(horizontal_exit, bend_x_min)
        else:
            if bend_x_max >= horizontal_exit:
                return None
            horizontal_axis_values = range(horizontal_exit, bend_x_max, -1)

        if vertical_direction > 0:
            if bend_y_min <= vertical_exit:
                return None
            vertical_axis_values = range(vertical_exit, bend_y_min)
        else:
            if bend_y_max >= vertical_exit:
                return None
            vertical_axis_values = range(vertical_exit, bend_y_max, -1)

        if (horizontal_direction > 0 and horizontal_exit >= bend_x_min) or (
            horizontal_direction < 0 and horizontal_exit <= bend_x_max
        ):
            return None

        if (vertical_direction > 0 and vertical_exit >= bend_y_min) or (
            vertical_direction < 0 and vertical_exit <= bend_y_max
        ):
            return None

        horizontal_tiles: List[Tuple[int, int]] = []
        vertical_tiles: List[Tuple[int, int]] = []

        def corridor_tile_blocked(tile: Tuple[int, int]) -> bool:
            if tile in tile_to_room:
                return True
            if tile in self.corridor_tiles:
                return True
            if tile in self.right_angle_tiles:
                return True
            if tile in self.t_junction_tiles:
                return True
            if tile in self.four_way_tiles:
                return True
            return False

        for axis_value in horizontal_axis_values:
            for y in horizontal_cross:
                if not (0 <= axis_value < self.width and 0 <= y < self.height):
                    return None
                tile = (axis_value, y)
                if corridor_tile_blocked(tile):
                    return None
                horizontal_tiles.append(tile)

        for axis_value in vertical_axis_values:
            for x in vertical_cross:
                if not (0 <= x < self.width and 0 <= axis_value < self.height):
                    return None
                tile = (x, axis_value)
                if corridor_tile_blocked(tile):
                    return None
                vertical_tiles.append(tile)

        if not horizontal_tiles or not vertical_tiles:
            return None

        joint_tiles: List[Tuple[int, int]] = []
        for x in range(bend_x_min, bend_x_max + 1):
            for y in range(bend_y_min, bend_y_max + 1):
                if not (0 <= x < self.width and 0 <= y < self.height):
                    return None
                tile = (x, y)
                if corridor_tile_blocked(tile):
                    return None
                joint_tiles.append(tile)

        last_horizontal_axis = horizontal_tiles[-1][0]
        horizontal_step = 1 if horizontal_direction > 0 else -1
        horizontal_joint_axis = last_horizontal_axis + horizontal_step

        last_vertical_axis = vertical_tiles[-1][1]
        vertical_step = 1 if vertical_direction > 0 else -1
        vertical_joint_axis = last_vertical_axis + vertical_step

        horizontal_geometry = CorridorGeometry(
            tiles=tuple(horizontal_tiles),
            axis_index=0,
            port_axis_values=(horizontal_exit, horizontal_joint_axis),
            cross_coords=tuple(horizontal_cross),
        )
        vertical_geometry = CorridorGeometry(
            tiles=tuple(vertical_tiles),
            axis_index=1,
            port_axis_values=(vertical_exit, vertical_joint_axis),
            cross_coords=tuple(vertical_cross),
        )

        if axis_a == 0:
            geom_a, geom_b = horizontal_geometry, vertical_geometry
        else:
            geom_a, geom_b = vertical_geometry, horizontal_geometry

        return geom_a, geom_b, tuple(joint_tiles)

    def _list_available_ports(
        self, room_world_ports: List[List[WorldPort]]
    ) -> List[Tuple[int, int, WorldPort]]:
        """Gather all unused ports for the currently placed rooms."""
        available_ports: List[Tuple[int, int, WorldPort]] = []
        for room_index, room in enumerate(self.placed_rooms):
            world_ports = room_world_ports[room_index]
            for port_index in room.get_available_port_indices():
                available_ports.append((room_index, port_index, world_ports[port_index]))
        return available_ports

    def create_easy_links(self) -> None:
        """Implements Step 2: connect facing ports with straight corridors."""
        if not self.placed_rooms:
            print("ERROR: no placed rooms.")
            return

        initial_corridor_count = len(self.corridors)
        tile_to_room = self._build_room_tile_lookup()
        room_world_ports = [room.get_world_ports() for room in self.placed_rooms]
        available_ports = self._list_available_ports(room_world_ports)

        random.shuffle(available_ports)
        used_ports: Set[Tuple[int, int]] = set()
        connected_room_pairs: Set[Tuple[int, int]] = {
            tuple(sorted((corridor.room_a_index, corridor.room_b_index))) # type: ignore
            for corridor in self.corridors
            if corridor.room_b_index is not None
        } # type: ignore

        for i, (room_a_idx, port_a_idx, world_port_a) in enumerate(available_ports):
            key_a = (room_a_idx, port_a_idx)
            if key_a in used_ports:
                continue

            candidate_indices = list(range(i + 1, len(available_ports)))
            random.shuffle(candidate_indices)

            for j in candidate_indices:
                room_b_idx, port_b_idx, world_port_b = available_ports[j]
                key_b = (room_b_idx, port_b_idx)
                if key_b in used_ports:
                    continue
                if room_a_idx == room_b_idx:
                    continue

                room_pair: Tuple[int, int] = tuple(sorted((room_a_idx, room_b_idx))) # type: ignore
                if room_pair in connected_room_pairs:
                    continue

                common_widths = world_port_a.widths & world_port_b.widths
                if not common_widths:
                    continue

                viable_options: List[Tuple[int, CorridorGeometry]] = []
                for width in common_widths:
                    geometry = self._build_corridor_geometry(
                        room_a_idx,
                        world_port_a,
                        room_b_idx,
                        world_port_b,
                        width,
                        tile_to_room,
                    )
                    if geometry is not None:
                        viable_options.append((width, geometry))

                if not viable_options:
                    continue

                width, geometry = random.choice(viable_options)

                overlap_map: Dict[int, List[Tuple[int, int]]] = {}
                for tile in geometry.tiles:
                    for existing_idx in self.corridor_tile_index.get(tile, []):
                        overlap_map.setdefault(existing_idx, []).append(tile)

                if overlap_map:
                    # Handle the simple case where the new corridor crosses exactly one existing corridor.
                    # More complex overlap scenarios (multiple intersections at once) are skipped for now.
                    if len(overlap_map) != 1:
                        continue

                    existing_idx, overlap_tiles = next(iter(overlap_map.items()))
                    existing_corridor = self.corridors[existing_idx]
                    if existing_corridor.geometry.axis_index is None or geometry.axis_index is None:
                        continue
                    if existing_corridor.geometry.axis_index == geometry.axis_index:
                        continue

                    component_id = self._merge_components(
                        self.placed_rooms[room_a_idx].component_id,
                        self.placed_rooms[room_b_idx].component_id,
                        existing_corridor.component_id,
                    )

                    pending_ref = ("four_way", -1)
                    intersection_axis_new = overlap_tiles[0][geometry.axis_index]
                    cross_coords_new = geometry.cross_coords or self._corridor_cross_from_geometry(
                        geometry, geometry.axis_index
                    )
                    seg_a = self._build_segment_geometry(
                        geometry.axis_index,
                        geometry.port_axis_values[0],
                        intersection_axis_new,
                        cross_coords_new,
                    )
                    seg_b = self._build_segment_geometry(
                        geometry.axis_index,
                        geometry.port_axis_values[1],
                        intersection_axis_new,
                        cross_coords_new,
                    )
                    if seg_a is None or seg_b is None:
                        continue

                    corridor_a = Corridor(
                        room_a_index=room_a_idx,
                        port_a_index=port_a_idx,
                        room_b_index=None,
                        port_b_index=None,
                        width=width,
                        geometry=seg_a,
                        component_id=component_id,
                        joint_b=pending_ref,
                    )
                    corridor_b = Corridor(
                        room_a_index=room_b_idx,
                        port_a_index=port_b_idx,
                        room_b_index=None,
                        port_b_index=None,
                        width=width,
                        geometry=seg_b,
                        component_id=component_id,
                        joint_b=pending_ref,
                    )

                    idx_a = self._register_corridor(corridor_a, component_id)
                    idx_b = self._register_corridor(corridor_b, component_id)
                    self.placed_rooms[room_a_idx].connected_port_indices.add(port_a_idx)
                    self.placed_rooms[room_b_idx].connected_port_indices.add(port_b_idx)
                    used_ports.add(key_a)
                    used_ports.add(key_b)
                    connected_room_pairs.add(room_pair)

                    existing_axis_value = overlap_tiles[0][existing_corridor.geometry.axis_index]
                    split_indices = self._split_corridor_for_junction(
                        existing_idx,
                        existing_axis_value,
                        pending_ref,
                    )

                    intersection_tiles = tuple(sorted({tile for tile in overlap_tiles}))
                    connected_indices = [idx_a, idx_b] + split_indices

                    intersection = FourWayIntersection(
                        tiles=intersection_tiles,
                        width=width,
                        connected_corridor_indices=tuple(sorted(set(connected_indices))), # type: ignore
                        component_id=component_id,
                    )
                    intersection_idx = self._register_four_way(intersection, component_id)
                    actual_ref = ("four_way", intersection_idx)
                    self._replace_joint_reference(connected_indices, pending_ref, actual_ref)

                    break

                else:
                    component_id = self._merge_components(
                        self.placed_rooms[room_a_idx].component_id,
                        self.placed_rooms[room_b_idx].component_id,
                    )

                    corridor = Corridor(
                        room_a_index=room_a_idx,
                        port_a_index=port_a_idx,
                        room_b_index=room_b_idx,
                        port_b_index=port_b_idx,
                        width=width,
                        geometry=geometry,
                        component_id=component_id,
                    )
                    self._register_corridor(corridor, component_id)
                    self.placed_rooms[room_a_idx].connected_port_indices.add(port_a_idx)
                    self.placed_rooms[room_b_idx].connected_port_indices.add(port_b_idx)
                    used_ports.add(key_a)
                    used_ports.add(key_b)
                    connected_room_pairs.add(room_pair)

                    break

        created = len(self.corridors) - initial_corridor_count
        total_four_way = sum(len(intersection.tiles) for intersection in self.four_way_intersections)
        print(
            f"Easylink step 2: created {created} straight corridors "
            f"(tracking {len(self.four_way_intersections)} 4-way intersections covering {total_four_way} tiles)."
        )

    def create_easy_t_junctions(self, fill_probability: float, step_num: int) -> int:
        """Implements Step 3, 5, and 7: link ports to corridors with straight passages."""
        if not self.corridors:
            print(f"Easylink step {step_num}: skipped - no existing corridors to join.")
            return 0

        tile_to_room = self._build_room_tile_lookup()
        room_world_ports = [room.get_world_ports() for room in self.placed_rooms]

        def build_existing_pairs() -> Set[Tuple[int, int]]:
            pairs: Set[Tuple[int, int]] = set()
            for junction in self.t_junctions:
                for corridor_idx in junction.connected_corridor_indices:
                    corridor = self.corridors[corridor_idx]
                    if corridor.room_a_index is not None:
                        pairs.add((corridor.room_a_index, corridor_idx))
                    if corridor.room_b_index is not None:
                        pairs.add((corridor.room_b_index, corridor_idx))
            return pairs

        existing_room_corridor_pairs = build_existing_pairs()

        available_ports = self._list_available_ports(room_world_ports)
        random.shuffle(available_ports)

        created = 0
        for room_idx, port_idx, world_port in available_ports:
            tile_to_corridors = self._build_corridor_tile_lookup()

            width_options = list(world_port.widths)
            random.shuffle(width_options)

            viable_options: List[Tuple[int, CorridorGeometry, int, Tuple[Tuple[int, int], ...]]] = []
            for width in width_options:
                result = self._build_t_junction_geometry(
                    room_idx,
                    world_port,
                    width,
                    tile_to_room,
                    tile_to_corridors,
                )
                if result is not None:
                    geometry, target_corridor_idx, junction_tiles = result
                    if (room_idx, target_corridor_idx) in existing_room_corridor_pairs:
                        continue
                    viable_options.append((width, geometry, target_corridor_idx, junction_tiles))

            if not viable_options:
                continue

            if random.random() > fill_probability:
                continue

            width, geometry, target_corridor_idx, junction_tiles = random.choice(viable_options)

            target_corridor = self.corridors[target_corridor_idx]
            component_id = self._merge_components(
                self.placed_rooms[room_idx].component_id,
                target_corridor.component_id,
            )

            pending_ref = ("t_junction", -1)
            new_corridor = Corridor(
                room_a_index=room_idx,
                port_a_index=port_idx,
                room_b_index=None,
                port_b_index=None,
                width=width,
                geometry=geometry,
                component_id=component_id,
                joint_a=None,
                joint_b=pending_ref,
            )

            new_corridor_idx = self._register_corridor(new_corridor, component_id)
            self.placed_rooms[room_idx].connected_port_indices.add(port_idx)

            axis_index = self.corridors[target_corridor_idx].geometry.axis_index
            if axis_index is None:
                continue
            junction_axis = junction_tiles[0][axis_index]

            linked_indices = self._split_corridor_for_junction(
                target_corridor_idx,
                junction_axis,
                pending_ref,
            )

            connected_indices = [new_corridor_idx] + linked_indices

            junction = TJunction(
                tiles=junction_tiles,
                width=width,
                connected_corridor_indices=tuple(sorted(set(connected_indices))), # type: ignore
                component_id=component_id,
            )
            junction_idx = self._register_t_junction(junction, component_id)
            actual_ref = ("t_junction", junction_idx)

            self._replace_joint_reference(connected_indices, pending_ref, actual_ref)

            existing_room_corridor_pairs.add((room_idx, target_corridor_idx))

            created += 1

        print(
            f"Easylink step {step_num}: created {created} corridor-to-corridor links "
            f"(tracking {len(self.t_junction_tiles)} T-junction tiles)."
        )
        return created

    def create_bent_room_links(self) -> int:
        """Implements Step 4: link different components via 90-degree corridors."""
        if len(self.placed_rooms) < 2:
            print("Easylink step 4: skipped - not enough rooms to connect.")
            return 0

        if len({*self.room_components, *self.corridor_components}) <= 1:
            print("Easylink step 4: skipped - already fully connected.")
            return 0

        tile_to_room = self._build_room_tile_lookup()
        room_world_ports = [room.get_world_ports() for room in self.placed_rooms]
        available_ports = self._list_available_ports(room_world_ports)
        if len(available_ports) < 2:
            print("Easylink step 4: skipped - not enough unused ports.")
            return 0

        port_records = [
            {
                "room_idx": room_idx,
                "port_idx": port_idx,
                "component_id": self.placed_rooms[room_idx].component_id,
                "port": world_port,
            }
            for room_idx, port_idx, world_port in available_ports
        ]

        candidates: List[Tuple[float, int, int, int, int, int, CorridorGeometry]] = []
        for i, port_a_info in enumerate(port_records):
            for port_b_info in port_records[i + 1 :]:
                if port_a_info["component_id"] == port_b_info["component_id"]:
                    continue

                port_a = port_a_info["port"]
                port_b = port_b_info["port"]
                dot = port_a.direction[0] * port_b.direction[0] + port_a.direction[1] * port_b.direction[1]
                if dot != 0:
                    continue

                common_widths = port_a.widths & port_b.widths
                if not common_widths:
                    continue

                chosen_geometries: Optional[Tuple[CorridorGeometry, CorridorGeometry]] = None
                chosen_joint_tiles: Optional[Tuple[Tuple[int, int], ...]] = None
                chosen_width: Optional[int] = None
                for width in sorted(common_widths):
                    plan = self._plan_right_angle_connection(
                        port_a,
                        port_b,
                        width,
                        tile_to_room,
                    )
                    if plan is not None:
                        geom_a, geom_b, joint_tiles = plan
                        chosen_geometries = (geom_a, geom_b)
                        chosen_joint_tiles = joint_tiles
                        chosen_width = width
                        break

                if chosen_geometries is None or chosen_joint_tiles is None or chosen_width is None:
                    continue

                distance = abs(port_a.pos[0] - port_b.pos[0]) + abs(port_a.pos[1] - port_b.pos[1])
                candidates.append(
                    (
                        float(distance),
                        chosen_width,
                        port_a_info["room_idx"],
                        port_a_info["port_idx"],
                        port_b_info["room_idx"],
                        port_b_info["port_idx"],
                        chosen_geometries, # type: ignore
                        chosen_joint_tiles,
                    )
                )

        if not candidates:
            print("Easylink step 4: no viable bent corridor opportunities found.")
            return 0

        candidates.sort(key=lambda item: (item[0], item[1]))

        created = 0
        for _, width, room_a_idx, port_a_idx, room_b_idx, port_b_idx, geometries, joint_tiles in candidates: # type: ignore
            room_a = self.placed_rooms[room_a_idx]
            room_b = self.placed_rooms[room_b_idx]

            if port_a_idx in room_a.connected_port_indices:
                continue
            if port_b_idx in room_b.connected_port_indices:
                continue
            if room_a.component_id == room_b.component_id:
                continue
            component_id = self._merge_components(room_a.component_id, room_b.component_id)

            joint = RightAngleJoint(
                tiles=joint_tiles,
                width=width,
                connected_corridor_indices=(), # type: ignore
                component_id=component_id,
            )
            joint_idx = self._register_right_angle_joint(joint, component_id)

            corridor_a = Corridor(
                room_a_index=room_a_idx,
                port_a_index=port_a_idx,
                room_b_index=None,
                port_b_index=None,
                width=width,
                geometry=geometries[0], # type: ignore
                component_id=component_id,
                joint_b=("right_angle", joint_idx),
            )
            corridor_b = Corridor(
                room_a_index=room_b_idx,
                port_a_index=port_b_idx,
                room_b_index=None,
                port_b_index=None,
                width=width,
                geometry=geometries[1], # type: ignore
                component_id=component_id,
                joint_b=("right_angle", joint_idx),
            )

            corridor_a_idx = self._register_corridor(corridor_a, component_id)
            corridor_b_idx = self._register_corridor(corridor_b, component_id)

            room_a.connected_port_indices.add(port_a_idx)
            room_b.connected_port_indices.add(port_b_idx)

            updated_joint = self.right_angle_joints[joint_idx]
            updated_joint.connected_corridor_indices = (corridor_a_idx, corridor_b_idx)
            self.right_angle_joints[joint_idx] = updated_joint

            created += 1

            if len({*self.room_components, *self.corridor_components}) <= 1:
                break

        print(f"Easylink step 4: created {created} bent corridors.")
        return created

    def place_rooms(self) -> None:
        """Implements Step 1: Randomly place rooms with macro-grid aligned ports."""
        print(f"Attempting to place {self.num_rooms_to_place} rooms...")
        placed_count = 0
        consecutive_limit_exceeded = 0

        for root_room_index in range(self.num_rooms_to_place):
            if placed_count >= self.num_rooms_to_place:
                break
            if consecutive_limit_exceeded >= MAX_CONSECUTIVE_LIMIT_FAILURES:
                print(f"Exceeded attempt limit {MAX_CONSECUTIVE_LIMIT_FAILURES} consecutive times, aborting further placement.")
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

            print(f"Placed root room number {root_room_index} after {attempt} failed attempts.")
            print(f"Placed root room is {placed_room.template.name} at {(placed_room.x, placed_room.y)}")

        print(f"Successfully placed {placed_count} rooms.")

    def draw_to_grid(self, draw_macrogrid: bool = False) -> None:
        """Renders the placed rooms and overlays all door ports."""
        self._clear_grid()
        # First fill rooms with a character so they are easy to distinguish.
        for room in self.placed_rooms:
            room_char = random.choice('OX/LNMW123456789')
            x, y, w, h = room.get_bounds()
            for j in range(h):
                for i in range(w):
                    self.grid[y + j][x + i] = room_char
        # Draw corridors as floor tiles
        for corridor in self.corridors:
            for tx, ty in corridor.geometry.tiles:
                self.grid[ty][tx] = '.'
        for joint in self.right_angle_joints:
            for tx, ty in joint.tiles:
                self.grid[ty][tx] = '.'
        for junction in self.t_junctions:
            for tx, ty in junction.tiles:
                self.grid[ty][tx] = '.'
        for intersection in self.four_way_intersections:
            for tx, ty in intersection.tiles:
                self.grid[ty][tx] = '.'
        # Then overlay ports
        for room in self.placed_rooms:
            for port in room.get_world_ports():
                for tx, ty in port.tiles:
                    self.grid[ty][tx] = '█'
        if draw_macrogrid:
            self._draw_macrogrid_overlay()

    def _draw_macrogrid_overlay(self) -> None:
        """Add 2x2 boxes showing macro-grid squares where door ports can appear."""
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] != ' ':
                    continue
                if (0 < (x % MACRO_GRID_SIZE) < MACRO_GRID_SIZE - 1) or (
                    0 < (y % MACRO_GRID_SIZE) < MACRO_GRID_SIZE - 1
                ):
                    continue
                self.grid[y][x] = '░'

    def print_grid(self, horizontal_sep: str = "") -> None:
        """Prints the ASCII grid to the console."""
        for row in self.grid:
            print(horizontal_sep.join(row))
