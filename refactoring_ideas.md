# Ideas for refactoring dungeon_generator.py

## Data structures and models

### Introduce a DungeonConfig dataclass

Currently we're passing a long list of parameters to `DungeonGenerator.__init__`. We also have constants in `dungeon_constants.py`. It would be better to create a single `DungeonConfig` dataclass that holds all of the constructor's arguments, and some of those constants (such as the macro-grid size, the random seed, and numbers for max connected placement attempts and max consecutive limit failures).

## Simplifying complex logic

### `attempt_place_special_room` -> `JunctionFitter` class

This is the most complex method in the file. It tries to fit a template to a set of required ports by iterating through templates, rotations, and port permutations.

Refactoring suggestion: Create a `JunctionFitter` class.

* This class would take the List[PortRequirement] and the list of RoomTemplates.
* Its core method, find_fit(), would encapsulate all the permutation and validation logic.
* It could be broken down further internally: a method to check a single template/rotation pair, a method to calculate the required translation, etc.
* This isolates the hardest part of the algorithm into one place.

### Abstract the dungeon-grower pattern

The methods `grower_room_to_room`, `grower_room_to_corridor`, `grower_bent_room_to_room` share a similar structure:

* Find candidate connection points (port-to-port, port-to-corridor).
* For each candidate, check for geometric validity.
* If valid, plan the new geometry (corridor, junction room).
* If planned successfully, register the new objects and merge components.

We want to abstract this pattern, by creating a generic `DungeonGrower` that takes a `CandidateFinder`, a `GeometryPlanner`, and an `Applier`.

* For `grow_room_to_room`, the `CandidateFinder` would yield pairs of opposing ports.
* For `grow_room_to_corridor`, it would yield pairs of (port, corridor).

Recognizing the shared pattern is key. We want to add more dungeon growers in the future, which will share a similar structure; for example, we're planning to add a feature later that will find long straight corridors and consider splitting them up by placing a 2-door room template in the middle.

### Corridor geometry logic

Methods like `_build_corridor_geometry`, `_trim_geometry_for_room`, and `_split_existing_corridor_geometries` are pure geometric calculations.

Refactoring suggestion: Move these to a dedicated geometry utils module, such as the existing `dungeon_geometry.py` file.

These functions don't need access to the DungeonGenerator's self. They should take all the data they need as arguments (e.g., port positions, room bounds, existing tiles).

This makes them pure, stateless functions that are easy to test independently. For example: `build_straight_corridor(port_a, port_b, width, obstacles: SpatialIndex)`.