- Implement algorithm steps 6-8, after refactor.

- Add support for weights when choosing room templates for T-junctions, 4-way junctions, and bend rooms. We generally want to prefer placing the bigger options if possible, and use the tiny 2x2 or 2x4 rooms only as a last resort.

- Add a step that splits up overly long corridors, by placing a new room kind (marked as RoomKind.THROUGH) in the middle somewhere. Unclear where this should go in the ordering of steps.

- Consider adding a step that looks for parallel corridors and creates a corridor between them, creating 2 T-junctions at the ends.

- Add constraint: avoid making short loops. Meaning for instance if room 1 and room 2 are linked by a passage A, and room 1 is linked to another passage B, then we shouldn't create a passage C from room 2 to passage B. This would require maintaining a list of all passages connected to each room and all rooms connected to a given passage, and then checking the graph distance of things we want to connect that are in the same component, and then rejecting that connection if the graph distance is too short.

- Implement alternate algorithm where we first split the map in half. Then place a room in each half, so they can be directly linked. Then subsequent steps are each restricted to one half of the map. Could create more variety in layouts.

- Implement non-rectangular room templates, such as long diagonal rooms. We could fit this into the existing system by modelling them as a multi-room collection of rectangular rooms, which must be placed as a unit.

- Port to C#, and then to Unity. Then drive the dungeon generator by collecting data from the module prefab library and converting it into RoomTemplates and corridor template sets. Then run the dungeon generator on that extracted data. The dungeon generator should return a list of module names and locations and some data about e.g. which door ports are occupied and their widths, which is then passed on to the next phase of world generation (recursive resolution of slots into modules).