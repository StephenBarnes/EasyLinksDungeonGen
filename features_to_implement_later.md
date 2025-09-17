- Implement remaining growers, after refactoring growers.
	- Add a grower that splits up overly long corridors, by placing a new room kind (marked as RoomKind.THROUGH) in the middle somewhere. Unclear where this should go in the ordering of steps.
	- Add a grower that tries to randomly rotate rooms that currently have none of their door ports connected; check if the rotated version's dimensions still fit. Then try to apply other growers again and see if they create new edges.
	- Consider adding a grower that looks for parallel corridors and creates a corridor between them, creating 2 T-junctions at the ends.
- Implement the step 2 loop that repeatedly runs growers.
- Implement step 3, deleting extra components and accepting or rejecting.

- Add support for weights when choosing room templates for T-junctions, 4-way junctions, and bend rooms. We generally want to prefer placing the bigger options if possible, and use the tiny 2x2 or 2x4 rooms only as a last resort.

- Add constraint: avoid making short loops. Meaning for instance if room 1 and room 2 are linked by a passage A, and room 1 is linked to another passage B, then we shouldn't create a passage C from room 2 to passage B. This would require maintaining a list of all passages connected to each room and all rooms connected to a given passage, and then checking the graph distance of things we want to connect that are in the same component, and then rejecting that connection if the graph distance is too short.

- Implement alternate algorithm where we first split the map in half. Then place a room in each half, so they can be directly linked. Then subsequent steps are each restricted to one half of the map. Could create more variety in layouts.

- Implement non-rectangular room templates, such as long diagonal rooms. We could fit this into the existing system by modelling them as a multi-room collection of rectangular rooms, which must be placed as a unit.

- Port to C#, and then to Unity. Then drive the dungeon generator by collecting data from the module prefab library and converting it into RoomTemplates and corridor template sets. Then run the dungeon generator on that extracted data. The dungeon generator should return a list of module names and locations and some data about e.g. which door ports are occupied and their widths, which is then passed on to the next phase of world generation (recursive resolution of slots into modules).


Bugs to fix:

- When placing special rooms (T-junctions, 4-ways, bends), we currently seem to be rejecting candidates that would directly touch an existing room, i.e. we are enforcing the min_room_separation. But the min_room_separation is only meant to apply in step 1, not to special rooms created by dungeon-growers.

- Rarely, the generator fails to find a 4-way intersection room that fits in a 4x4 region, despite the fact that a 4x4 room template has been created that should fit. Probably a subtle off-by-one in the code that checks if it fits.