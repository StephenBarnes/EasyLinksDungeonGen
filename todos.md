- Add validation for the room templates.
	- If one of the door ports is given, check that placing that door port in a valid position on the macrogrid will also put all other door ports at valid positions on the macrogrid.
	- Every room must have at least 1 door port.

- Add constraint: in step 3, don't make a corridor from room X that links to a corridor linked to a corridor linked to a corridor linked to room X. Or, just ban making parallel corridors in step 3.

- Add inward-facing argument to some rooms - when placed as root, close to edges of the map, they should always be rotated with that side facing away from the edge they're close to. Only affects step 1.
- Separate room template biases by distance from center. Dead-end rooms should be more common on the outskirts.
- Note, for these, we need to select the general region where we want to place a room, before we select which room to place. Maybe select the center, then select room, then shunt it a bit to make the door ports line up.