- Place first root room in a smaller rectangle, say only in the central 30% of the map along each dimension. Add a DungeonConfig field for this 30% value. Also adjust the _categorize_side_distance function in initial tree grower to adjust for the smaller size.

- Instead of setting direction preference when creating each RoomTemplate, rather automatically infer it by checking which sides have door-ports on them. Eg the 90-degree bend templates should avoid having either door port facing map edge.

- Add a re-centering step? If dungeon is on one side of the map, move it closer to the middle, then try adding more rooms.

- Make initial-tree-grower respect the rules for long parallel corridors

- Confirm that initial grower isn't running graph distance checks, etc.

- Make it retry step 1, if we couldn't make enough rooms. Remove the accept-reject thing from dungeonconfig.

- Remove the rotate_rooms grower, which is obsolete now.

- Test that the code limiting growers to considering new rooms and corridors only is actually working.

- Use different connection-distance thresholds for different growers.

- For rooms with only 1 connected port at the end, implement a contraction step where we try to move them inward and shorten the corridor.

- Ban creation of very long corridors by step-2 growers. (Test how this affects total connectivity.)

- Improve benchmarker/metrics.
	- Track number of rooms created in total, and number above some acceptance threshold.
	- Add graph/network theory metrics to benchmarking algorithm.
		- Number of cycles.
		- Distribution of cycle size.

- Add a grower that looks for parallel corridors and creates a corridor between them, creating 2 T-junctions at the ends.

- Implement alternate algorithm where we first split the map in half. Then place a room in each half, so they can be directly linked. Then subsequent steps are each restricted to one half of the map. Could create more variety in layouts.

- Implement non-rectangular room templates, such as long diagonal rooms. We could fit this into the existing system by modelling them as a multi-room collection of rectangular rooms, which must be placed as a unit.

- Port to C#, and then to Unity. Then drive the dungeon generator by collecting data from the module prefab library and converting it into RoomTemplates and corridor template sets. Then run the dungeon generator on that extracted data. The dungeon generator should return a list of module names and locations and some data about e.g. which door ports are occupied and their widths, which is then passed on to the next phase of world generation (recursive resolution of slots into modules).


Bugs to fix:

- When placing special rooms (T-junctions, 4-ways, bends), we currently seem to be rejecting candidates that would directly touch an existing room, i.e. we are enforcing the min_room_separation. But the min_room_separation is only meant to apply in step 1, not to special rooms created by dungeon-growers.