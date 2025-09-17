# Ideas for refactoring dungeon_generator.py

## Data structures and models

### Introduce a SpatialIndex class.

Currently the code frequently rebuilds tile-to-room and tile-to-corridor lookups in `_build_room_tile_lookup` and `corridor_tile_index`. This is inefficient, and the logic is spread out.

Responsibilities:

* Maintain a grid or dictionary mapping tile coordinates `(x,y)` to the objects (rooms and corridors) that occupy them.
* Provide methods like `add(obj, bounds)`, `remove(obj)`, `is_area_clear(bounds, ignore_list=[...])`, `get_objects_at(tile)`.

The benefit of this is that all spatial query logic is centralized. When we add a room, we update the index once. All subsequent collision checks become much cleaner and faster. This completely replaces `corridor_tiles` and `corridor_tile_index`.

### Use a DSU (disjoint set union) for component management.

The current component management (`room_components`, `corridor_components`, `_merge_components`) involves iterating through lists to update component IDs. This is an O(N) operation for every merge.

A DSU (or Union-Find) data structure is designed for exactly this problem. Merging two components becomes a nearly constant-time operation.

You could create a `ComponentManager` class that wraps a DSU. It would track rooms and corridors by their list indices. `_merge_components(id1, id2)` becomes `component_manager.union(item1_id, item2_id)`.

### Introduce a DungeonConfig dataclass

Currently we're passing a long list of parameters to `DungeonGenerator.__init__`. We also have constants in `dungeon_constants.py`. It would be better to create a single `DungeonConfig` dataclass that holds all of the constructor's arguments, and some of those constants (such as the macro-grid size, the random seed, and numbers for max connected placement attempts and max consecutive limit failures.

## Simplifying complex logic

### `attempt_place_special_room` -> `JunctionFitter` class

This is the most complex method in the file. It tries to fit a template to a set of required ports by iterating through templates, rotations, and port permutations.

Refactoring suggestion: Create a `JunctionFitter` class.

* This class would take the List[PortRequirement] and the list of RoomTemplates.
* Its core method, find_fit(), would encapsulate all the permutation and validation logic.
* It could be broken down further internally: a method to check a single template/rotation pair, a method to calculate the required translation, etc.
* This isolates the hardest part of the algorithm into one place.

### Abstract the linking pattern

The methods `create_easy_links`, `create_easy_t_junctions`, `create_bent_room_links` share a similar structure:

* Find candidate connection points (port-to-port, port-to-corridor).
* For each candidate, check for geometric validity.
* If valid, plan the new geometry (corridor, junction room).
* If planned successfully, register the new objects and merge components.

Refactoring suggestion: abstract this pattern.

We could create a generic "`GeometryCreator`" that takes a `CandidateFinder`, a `GeometryPlanner`, and an `Applier`.

* For `create_easy_links`, the `CandidateFinder` would yield pairs of opposing ports.
* For `create_easy_t_junctions`, it would yield pairs of (port, corridor).

Recognizing the shared pattern is key. We might want to add more geometry creators in the future, which will share a similar structure; for example, we're planning to add a feature later that will find long straight corridors and consider splitting them up by placing a 2-door room template in the middle.

### Corridor geometry logic

Methods like `_build_corridor_geometry`, `_trim_geometry_for_room`, and `_split_existing_corridor_geometries` are pure geometric calculations.

Refactoring suggestion: Move these to a dedicated geometry utils module, such as the existing `dungeon_geometry.py` file.

These functions don't need access to the DungeonGenerator's self. They should take all the data they need as arguments (e.g., port positions, room bounds, existing tiles).

This makes them pure, stateless functions that are easy to test independently. For example: `build_straight_corridor(port_a, port_b, width, obstacles: SpatialIndex)`.