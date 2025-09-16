#!/usr/bin/env python3

import random
import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# --- Configuration Constants ---
MACRO_GRID_SIZE = 4
RANDOM_SEED = None # For reproducing bugs

# Door centers must land on these macro-grid offsets per-facing direction.
DOOR_FRAC_OFFSET = MACRO_GRID_SIZE - 0.5
DOOR_WHOLE_OFFSET = float(MACRO_GRID_SIZE - 1)
DOOR_MACRO_ALIGNMENT_OFFSETS = {
    (0, -1): (DOOR_FRAC_OFFSET, 0.0),   # North-facing ports
    (0, 1): (DOOR_FRAC_OFFSET, DOOR_WHOLE_OFFSET),    # South-facing ports
    (-1, 0): (0.0, DOOR_FRAC_OFFSET),   # West-facing ports
    (1, 0): (DOOR_WHOLE_OFFSET, DOOR_FRAC_OFFSET),    # East-facing ports
}

VALID_ROTATIONS: Tuple[int, ...] = (0, 90, 180, 270)
MAX_CONNECTED_PLACEMENT_ATTEMPTS = 40


def _port_tiles_from_world_pos(world_x: float, world_y: float) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Derive the 1x2 or 2x1 tile footprint for a port center point."""
    frac_x = world_x - math.floor(world_x)
    frac_y = world_y - math.floor(world_y)
    if math.isclose(frac_x, 0.5, abs_tol=1e-6):
        base_y = int(round(world_y))
        return (
            (int(math.floor(world_x)), base_y),
            (int(math.ceil(world_x)), base_y),
        )
    if math.isclose(frac_y, 0.5, abs_tol=1e-6):
        base_x = int(round(world_x))
        return (
            (base_x, int(math.floor(world_y))),
            (base_x, int(math.ceil(world_y))),
        )
    raise ValueError(f"Port center must have exactly one half coordinate, got {(world_x, world_y)}")


def _rotate_point(px: float, py: float, width: int, height: int, rotation: int) -> Tuple[float, float]:
    if rotation == 0:
        return px, py
    if rotation == 90:
        return py, width - 1 - px
    if rotation == 180:
        return width - 1 - px, height - 1 - py
    if rotation == 270:
        return height - 1 - py, px
    raise ValueError(f"Unsupported rotation {rotation}")


def _rotate_direction(dx: int, dy: int, rotation: int) -> Tuple[int, int]:
    if rotation == 0:
        return dx, dy
    if rotation == 90:
        return dy, -dx
    if rotation == 180:
        return -dx, -dy
    if rotation == 270:
        return -dy, dx
    raise ValueError(f"Unsupported rotation {rotation}")


@dataclass
class PortTemplate:
    """Doorway specification in template-local coordinates."""

    pos: Tuple[float, float]
    direction: Tuple[int, int]
    widths: FrozenSet[int]

    def __post_init__(self) -> None:
        dx, dy = self.direction
        if dx not in (-1, 0, 1) or dy not in (-1, 0, 1):
            raise ValueError(f"Port direction coords must be +-1 and 0, got {dx, dy}")
        if abs(dx) + abs(dy) != 1:
            raise ValueError(f"Port directions must be axis-aligned, got {dx, dy}")

        px, py = (float(self.pos[0]), float(self.pos[1]))
        if dx != 0:
            if not math.isclose(py - math.floor(py), 0.5, abs_tol=1e-6):
                raise ValueError(f"Vertical doorway must have .5 fractional y, got {py}")
            if not math.isclose(px, round(px), abs_tol=1e-6):
                raise ValueError(f"Vertical doorway must have integer x, got {px}")
        else:
            if not math.isclose(px - math.floor(px), 0.5, abs_tol=1e-6):
                raise ValueError(f"Horizontal doorway must have .5 fractional x, got {px}")
            if not math.isclose(py, round(py), abs_tol=1e-6):
                raise ValueError(f"Horizontal doorway must have integer y, got {py}")

        self.pos = (px, py)
        self.widths = frozenset(int(w) for w in self.widths)


@dataclass
class RoomTemplate:
    """Defines the blueprint for a type of room in its default rotation."""

    name: str
    size: Tuple[int, int]
    ports: List[PortTemplate]
    root_weight: float = 1.0 # Weight for random choice when choosing root room to place.
    direct_weight: float = 1.0 # Weight for random choice when creating direct-linked rooms.

    def __post_init__(self) -> None:
        width, height = self.size
        self.size = (int(width), int(height))


@dataclass(frozen=True)
class WorldPort:
    """Port information after converting to world coordinates."""

    pos: Tuple[float, float]
    tiles: Tuple[Tuple[int, int], Tuple[int, int]]
    direction: Tuple[int, int]
    widths: FrozenSet[int]


@dataclass(frozen=True)
class CorridorGeometry:
    """Holds the tile layout for a straight corridor between two ports."""

    tiles: Tuple[Tuple[int, int], ...]
    axis_index: int  # 0 for horizontal corridors, 1 for vertical
    port_axis_values: Tuple[int, int]


@dataclass
class Corridor:
    """Stores metadata for a carved corridor."""

    room_a_index: int
    port_a_index: int
    room_b_index: Optional[int]
    port_b_index: Optional[int]
    width: int
    geometry: CorridorGeometry
    component_id: int = -1
    joined_corridor_indices: Tuple[int, ...] = field(default_factory=tuple)
    junction_tiles: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)


@dataclass
class PlacedRoom:
    """Represents a room instance placed on the dungeon grid."""

    template: RoomTemplate
    x: int
    y: int
    rotation: int
    component_id: int = -1
    connected_port_indices: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.rotation not in VALID_ROTATIONS:
            raise ValueError(f"Unsupported rotation {self.rotation}")

    def get_available_port_indices(self) -> List[int]:
        """Return indices of template ports not yet used for a direct connection."""
        return [i for i in range(len(self.template.ports)) if i not in self.connected_port_indices]

    @property
    def width(self) -> int:
        if self.rotation in (0, 180):
            return self.template.size[0]
        return self.template.size[1]

    @property
    def height(self) -> int:
        if self.rotation in (0, 180):
            return self.template.size[1]
        return self.template.size[0]

    def get_bounds(self) -> Tuple[int, int, int, int]:
        """Returns the bounding box as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    def get_world_ports(self) -> List[WorldPort]:
        """Calculates the real-world positions and directions of ports after rotation."""
        world_ports: List[WorldPort] = []
        w, h = self.template.size

        for port in self.template.ports:
            rp_x, rp_y = _rotate_point(port.pos[0], port.pos[1], w, h, self.rotation)
            rd_x, rd_y = _rotate_direction(port.direction[0], port.direction[1], self.rotation)

            world_x = self.x + rp_x
            world_y = self.y + rp_y
            tiles = _port_tiles_from_world_pos(world_x, world_y)

            world_ports.append(
                WorldPort(
                    pos=(world_x, world_y),
                    tiles=tiles,
                    direction=(rd_x, rd_y),
                    widths=port.widths,
                )
            )

        return world_ports


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
        self.num_rooms_to_place = num_rooms_to_place
        self.min_room_separation = min_room_separation # Minimum empty tiles between room bounding boxes, unless they connect at ports.
        self.min_rooms_required = min_rooms_required
        self.placed_rooms: List[PlacedRoom] = []
        self.room_components: List[int] = []
        self.corridors: List[Corridor] = []
        self.corridor_components: List[int] = []
        self.corridor_tiles: Set[Tuple[int, int]] = set()
        self.four_way_junctions: Set[Tuple[int, int]] = set()
        self.t_junction_tiles: Set[Tuple[int, int]] = set()
        self.grid = [[' ' for _ in range(width)] for _ in range(height)]
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
                row[x] = ' '

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
        return len(self.corridors) - 1

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

        return target

    def get_component_summary(self) -> Dict[int, Dict[str, List[int]]]:
        """Return indices of rooms and corridors grouped by component id."""
        summary: Dict[int, Dict[str, List[int]]] = {}

        for idx, component_id in enumerate(self.room_components):
            comp_summary = summary.setdefault(component_id, {"rooms": [], "corridors": []})
            comp_summary["rooms"].append(idx)

        for idx, component_id in enumerate(self.corridor_components):
            comp_summary = summary.setdefault(component_id, {"rooms": [], "corridors": []})
            comp_summary["corridors"].append(idx)

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

    def _build_root_room_candidate(self, template: RoomTemplate, rotation: int) -> PlacedRoom:
        anchor_port_index = random.randrange(len(template.ports))
        macro_x, macro_y = self._random_macro_grid_point()

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

    def _sample_num_direct_links(self):
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

    def _attempt_place_connected_to(self, anchor_room):
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
                # Pick a random room template and rotation
                template = random.choices(self.room_templates, weights=[rt.direct_weight for rt in self.room_templates])[0]
                rotation = self._random_rotation()
                # Compute rotated ports for this template at origin
                temp_room = PlacedRoom(template, 0, 0, rotation)
                rot_ports = temp_room.get_world_ports()
                # Find ports facing opposite direction
                compatible_port_indices = [
                    i for i, p in enumerate(rot_ports)
                    if p.direction == (-dx, -dy) and (p.widths & awp.widths)
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
                # Validate allowing adjacency to the anchor room only
                if self._is_valid_placement_with_anchor(candidate, anchor_room):
                    # Place it and mark ports used on both rooms
                    self._register_room(candidate, anchor_room.component_id)
                    anchor_room.connected_port_indices.add(anchor_idx)
                    candidate.connected_port_indices.add(cand_idx)
                    return candidate
        # No success on any port
        return None

    def _spawn_direct_links_recursive(self, from_room):
        """Recursively try to place 0-2 directly-connected rooms from from_room. Returns number of rooms placed."""
        rooms_placed = 0
        n = self._sample_num_direct_links()
        for _ in range(n):
            child = self._attempt_place_connected_to(from_room)
            if child is not None:
                # Recurse from the newly placed room
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
            for tile in corridor.junction_tiles:
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
                    if candidate.geometry.axis_index == axis_index:
                        continue
                    chosen_idx = idx
                    break

                if chosen_idx is None:
                    return None

                geometry = CorridorGeometry(
                    tiles=tuple(path_tiles),
                    axis_index=axis_index,
                    port_axis_values=(exit_axis_value, axis_value),
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

    def create_easy_links(self) -> None:
        """Implements Step 2: connect facing ports with straight corridors."""
        if not self.placed_rooms:
            print("ERROR: no placed rooms.")
            return

        initial_corridor_count = len(self.corridors)
        tile_to_room = self._build_room_tile_lookup()
        room_world_ports = [room.get_world_ports() for room in self.placed_rooms]
        available_ports: List[Tuple[int, int, WorldPort]] = []
        for room_index, room in enumerate(self.placed_rooms):
            world_ports = room_world_ports[room_index]
            for port_index in room.get_available_port_indices():
                available_ports.append((room_index, port_index, world_ports[port_index]))

        random.shuffle(available_ports)
        used_ports: Set[Tuple[int, int]] = set()
        connected_room_pairs: Set[Tuple[int, int]] = {
            tuple(sorted((corridor.room_a_index, corridor.room_b_index)))
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

                room_pair = tuple(sorted((room_a_idx, room_b_idx)))
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
                connected_room_pairs.add(room_pair) #type: ignore

                for tile in geometry.tiles:
                    if tile in self.corridor_tiles:
                        self.four_way_junctions.add(tile)
                    self.corridor_tiles.add(tile)

                break

        created = len(self.corridors) - initial_corridor_count
        print(f"Easylink step 2: created {created} straight corridors (with {len(self.four_way_junctions)} tiles in 4-way junctions).")

    def create_easy_t_junctions(self, fill_probability: float) -> int:
        """Implements Step 3: link ports to corridors with straight passages. Returns number of corridors created."""
        if not self.corridors:
            print("Easylink step 3: skipped - no existing corridors to join.")
            return 0

        tile_to_room = self._build_room_tile_lookup()
        tile_to_corridors = self._build_corridor_tile_lookup()
        room_world_ports = [room.get_world_ports() for room in self.placed_rooms]

        existing_room_corridor_pairs: Set[Tuple[int, int]] = set()
        for corridor_idx, corridor in enumerate(self.corridors):
            if corridor.room_b_index is not None:
                continue
            for linked_corridor in corridor.joined_corridor_indices:
                existing_room_corridor_pairs.add((corridor.room_a_index, linked_corridor))

        available_ports: List[Tuple[int, int, WorldPort]] = []
        for room_idx, room in enumerate(self.placed_rooms):
            world_ports = room_world_ports[room_idx]
            for port_idx in room.get_available_port_indices():
                available_ports.append((room_idx, port_idx, world_ports[port_idx]))

        random.shuffle(available_ports)

        created = 0
        for room_idx, port_idx, world_port in available_ports:
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

            corridor = Corridor(
                room_a_index=room_idx,
                port_a_index=port_idx,
                room_b_index=None,
                port_b_index=None,
                width=width,
                geometry=geometry,
                component_id=component_id,
                joined_corridor_indices=(target_corridor_idx,),
                junction_tiles=junction_tiles,
            )

            new_corridor_idx = self._register_corridor(corridor, component_id)
            self.placed_rooms[room_idx].connected_port_indices.add(port_idx)

            for tile in geometry.tiles:
                if tile in self.corridor_tiles:
                    self.four_way_junctions.add(tile)
                self.corridor_tiles.add(tile)
                tile_to_corridors.setdefault(tile, []).append(new_corridor_idx)

            for tile in junction_tiles:
                self.t_junction_tiles.add(tile)
                tile_to_corridors.setdefault(tile, []).append(new_corridor_idx)

            target_junction_tiles = list(target_corridor.junction_tiles)
            for tile in junction_tiles:
                if tile not in target_junction_tiles:
                    target_junction_tiles.append(tile)
            target_corridor.junction_tiles = tuple(target_junction_tiles)
            target_corridor.joined_corridor_indices = tuple(
                sorted(set(target_corridor.joined_corridor_indices + (new_corridor_idx,)))
            )

            created += 1
            existing_room_corridor_pairs.add((room_idx, target_corridor_idx))

        print(
            f"Easylink step 3: created {created} corridor-to-corridor links "
            f"(tracking {len(self.t_junction_tiles)} T-junction tiles)."
        )
        return created

    def place_rooms(self) -> None:
        """Implements Step 1: Randomly place rooms with macro-grid aligned ports."""
        print(f"Attempting to place {self.num_rooms_to_place} rooms...")
        placed_count = 0

        for root_room_index in range(self.num_rooms_to_place):
            if placed_count >= self.num_rooms_to_place:
                break

            placed_room: Optional[PlacedRoom] = None
            attempt = 0
            for attempt in range(20):
                template = random.choices(self.room_templates, weights=[rt.root_weight for rt in self.room_templates])[0]
                rotation = self._random_rotation()
                candidate_room = self._build_root_room_candidate(template, rotation)
                if self._is_valid_placement(candidate_room):
                    placed_room = candidate_room
                    break

            if placed_room is None:
                print(f"Exceeded attempt limit when placing root room number {root_room_index}.")
                continue

            component_id = self._new_component_id()
            self._register_room(placed_room, component_id)
            placed_count += 1
            placed_count += self._spawn_direct_links_recursive(placed_room)

            print(f"Placed root room number {root_room_index} after {attempt} failed attempts.")
            print(f"Placed root room is {placed_room.template.name} at {(placed_room.x, placed_room.y)}")

        print(f"Successfully placed {placed_count} rooms.")


    def draw_to_grid(self, draw_macrogrid: bool = False):
        """Renders the placed rooms and overlays all door ports."""
        self._clear_grid()
        # First fill rooms with a character
        for room in self.placed_rooms:
            room_char = random.choice('OX/\\LNMW123456789')
            x, y, w, h = room.get_bounds()
            for j in range(h):
                for i in range(w):
                    self.grid[y + j][x + i] = room_char
        # Draw corridors as floor tiles
        for corridor in self.corridors:
            for tx, ty in corridor.geometry.tiles:
                self.grid[ty][tx] = '.'
        # Then overlay ports
        for room in self.placed_rooms:
            for port in room.get_world_ports():
                for tx, ty in port.tiles:
                    self.grid[ty][tx] = '█'
        # Add 2x2 boxes to mark out the macro-grid, specifically all squares where 1x2 or 2x1 door ports can appear.
        if draw_macrogrid:
            for y in range(self.height):
                for x in range(self.width):
                    if self.grid[y][x] != ' ':
                        continue
                    if (0 < (x % MACRO_GRID_SIZE) < MACRO_GRID_SIZE - 1) or (0 < (y % MACRO_GRID_SIZE) < MACRO_GRID_SIZE - 1):
                        continue
                    self.grid[y][x] = '░'

    
    def print_grid(self, horizontal_sep: str = ""):
        """Prints the ASCII grid to the console. Optional horizontal separater to compensate for character aspect ratio."""
        for i, row in enumerate(self.grid):
            print(horizontal_sep.join(row))

# --- Main Execution ---
if __name__ == "__main__":
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    # Define room modules for 4x4 macrogrid
    room_templates = [
        RoomTemplate(
            name="room_8x8_4doors",
            size=(8, 8),
            ports=[
                # Top
                PortTemplate(pos=(3.5, 0), direction=(0, -1), widths=frozenset((2,4))),
                # Bottom
                PortTemplate(pos=(3.5, 7), direction=(0, 1), widths=frozenset({2, 4})),
                # Left
                PortTemplate(pos=(0, 3.5), direction=(-1, 0), widths=frozenset({2, 4})),
                # Right
                PortTemplate(pos=(7, 3.5), direction=(1, 0), widths=frozenset({2, 4})),
            ],
            root_weight=1.5,
            direct_weight=1.5,
        ),
        RoomTemplate(
            name="room_8x10_5doors",
            size=(8, 10),
            ports=[
                # Bottom
                PortTemplate(pos=(3.5, 9), direction=(0, 1), widths=frozenset({2, 4})),
                # Left
                PortTemplate(pos=(0, 1.5), direction=(-1, 0), widths=frozenset({2})),
                PortTemplate(pos=(0, 5.5), direction=(-1, 0), widths=frozenset({2})),
                # Right
                PortTemplate(pos=(7, 1.5), direction=(1, 0), widths=frozenset({2})),
                PortTemplate(pos=(7, 5.5), direction=(1, 0), widths=frozenset({2})),
            ],
            root_weight=2.0,
        ),
        RoomTemplate(
            name="room_8x6_2doors",
            size=(8, 6),
            ports=[
                # Left
                PortTemplate(pos=(0, 2.5), direction=(-1, 0), widths=frozenset({2, 4})),
                # Right
                PortTemplate(pos=(7, 2.5), direction=(1, 0), widths=frozenset({2, 4})),
            ]
        ),
        RoomTemplate(
            name="room_6x6_90deg",
            size=(6, 6),
            ports=[
                # Left
                PortTemplate(pos=(0, 1.5), direction=(-1, 0), widths=frozenset({2})),
                # Bottom
                PortTemplate(pos=(3.5, 5), direction=(0, 1), widths=frozenset({2})),
            ]
        ),
        RoomTemplate(
            name="room_6x4_deadend",
            size=(6, 4),
            ports=[
                # Top
                PortTemplate(pos=(2.5, 0), direction=(0, -1), widths=frozenset({2, 4})),
            ],
            root_weight=0.25,
            direct_weight=1.0,
        ),
    ]

    # Create the generator
    generator = DungeonGenerator(
        width=100,
        height=40,
        room_templates=room_templates,
        direct_link_counts_probs={0: 0.65, 1: 0.2, 2: 0.1, 3: 0.05},
        num_rooms_to_place=10,
        min_room_separation=1,
    )

    # --- Step 1: Place Rooms ---
    generator.place_rooms()

    # --- Step 2: Create straight easy-link corridors ---
    generator.create_easy_links()

    # --- Step 3: Create straight T-junction corridors ---
    num_created = 1
    while num_created > 0:  # Re-run to create corridors to new corridors.
        num_created = generator.create_easy_t_junctions(fill_probability=1)

    # Render the final layout to the grid and print it
    generator.draw_to_grid(draw_macrogrid=False)
    generator.print_grid(horizontal_sep="")
