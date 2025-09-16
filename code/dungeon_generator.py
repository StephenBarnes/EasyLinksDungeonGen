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
)
from dungeon_models import Corridor, CorridorGeometry, PlacedRoom, RoomTemplate, WorldPort


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
        # Minimum empty tiles between room bounding boxes, unless they connect at ports.
        self.min_room_separation = min_room_separation
        self.min_rooms_required = min_rooms_required
        self.placed_rooms: List[PlacedRoom] = []
        self.room_components: List[int] = []
        self.corridors: List[Corridor] = []
        self.corridor_components: List[int] = []
        self.corridor_tiles: Set[Tuple[int, int]] = set()
        self.four_way_junctions: Set[Tuple[int, int]] = set()
        self.t_junction_tiles: Set[Tuple[int, int]] = set()
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
                    self.room_templates, weights=[rt.direct_weight for rt in self.room_templates]
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

                for tile in geometry.tiles:
                    if tile in self.corridor_tiles:
                        self.four_way_junctions.add(tile)
                    self.corridor_tiles.add(tile)

                break

        created = len(self.corridors) - initial_corridor_count
        print(
            f"Easylink step 2: created {created} straight corridors (with {len(self.four_way_junctions)} tiles in 4-way junctions)."
        )

    def create_easy_t_junctions(self, fill_probability: float) -> int:
        """Implements Step 3: link ports to corridors with straight passages."""
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

        available_ports = self._list_available_ports(room_world_ports)
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
                template = random.choices(
                    self.room_templates, weights=[rt.root_weight for rt in self.room_templates]
                )[0]
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
