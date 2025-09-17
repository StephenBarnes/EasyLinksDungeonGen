# Ideas for refactoring dungeon_generator.py

## More dataclasses

We still use Tuple[int, int] and Tuple[float, float] in several places, for coordinates. Rather refactor out to use the existing TilePos, and create a FloatTilePos for float-valued positions (for door ports).

Similarly axis_index uses an int (0 for horizontal, 1 for vertical), which should rather be an enum.

## PortRequirement source enum

In PortRequirement, the source is a string like "new_a" or "existing_b". This is brittle. An Enum would be more robust.

## `attempt_place_special_room` -> `JunctionFitter` class

The very complex logic in _attempt_place_special_room is a prime candidate for its own class. This class would specialize in the difficult task of finding a template that fits a set of geometric PortRequirements. We can move this to a new class.

Currently _attempt_place_special_room is the most complex method in the project. It tries to fit a template to a set of required ports by iterating through templates, rotations, and port permutations.

Refactoring suggestion: Create a `JunctionFitter` class.

* This class would take the List[PortRequirement] and the list of RoomTemplates.
* Its core method, find_fit(), would encapsulate all the permutation and validation logic.
* It could be broken down further internally: a method to check a single template/rotation pair, a method to calculate the required translation, etc.
* This isolates the hardest part of the algorithm into one place.

## Corridor geometry logic

Methods like `_build_corridor_geometry`, `_trim_geometry_for_room`, and `_split_existing_corridor_geometries` are pure geometric calculations.

Refactoring suggestion: Move these to a dedicated geometry utils module, such as the existing `dungeon_geometry.py` file.

These functions don't need access to the DungeonGenerator's self. They should take all the data they need as arguments (e.g., port positions, room bounds, existing tiles).

This makes them pure, stateless functions that are easy to test independently. For example: `build_straight_corridor(port_a, port_b, width, obstacles: SpatialIndex)`.