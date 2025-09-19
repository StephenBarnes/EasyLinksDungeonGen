- Add a RoomKind.DIRECT_LINKED.

- Add stop_after_first arg to through-room generator, so that we don't place too many through-rooms, especially since that reduces room template diversity.

- Add options to run growers only for specific rooms or corridors, to improve performance. When one step adds rooms or corridors, we only want subsequent growers to consider new connections involving those, not re-consider all of the possible options.

- Use different connection-distance thresholds for different growers.

- Place all root rooms in a smaller rectangle, say only in the central 80% of the map along each direction. Add a DungeonConfig field for this 80% value. Also adjust the _categorize_side_distance function in root_room_placer.py to adjust for the smaller size.

- Add support for weights when choosing room templates for T-junctions, 4-way junctions, and bend rooms. We generally want to prefer placing the bigger options if possible, and use the tiny 2x2 or 2x4 rooms only as a last resort.

- For rooms with only 1 connected port at the end, implement a contraction step where we try to move them inward and shorten the corridor.

- Ban creation of very long corridors. (Test how this affects total connectivity.)

- Implement tests.

- Implement integration testing to measure metrics over many random runs. This would allow testing the impact of tweaks and optimizations.
	- Measure performance of the entire gen algorithm. (Done.)
	- Measure some metrics about dungeon structure, such as the number of cycles and distribution of their sizes.

- Add a grower that looks for parallel corridors and creates a corridor between them, creating 2 T-junctions at the ends.

- Implement alternate algorithm where we first split the map in half. Then place a room in each half, so they can be directly linked. Then subsequent steps are each restricted to one half of the map. Could create more variety in layouts.

- Implement non-rectangular room templates, such as long diagonal rooms. We could fit this into the existing system by modelling them as a multi-room collection of rectangular rooms, which must be placed as a unit.

- Port to C#, and then to Unity. Then drive the dungeon generator by collecting data from the module prefab library and converting it into RoomTemplates and corridor template sets. Then run the dungeon generator on that extracted data. The dungeon generator should return a list of module names and locations and some data about e.g. which door ports are occupied and their widths, which is then passed on to the next phase of world generation (recursive resolution of slots into modules).


Bugs to fix:

- When placing special rooms (T-junctions, 4-ways, bends), we currently seem to be rejecting candidates that would directly touch an existing room, i.e. we are enforcing the min_room_separation. But the min_room_separation is only meant to apply in step 1, not to special rooms created by dungeon-growers.