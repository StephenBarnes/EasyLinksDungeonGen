- Add validation for the room templates.
	- If one of the door ports is given, check that placing that door port in a valid position on the macrogrid will also put all other door ports at valid positions on the macrogrid.
	- Every room must have at least 1 door port.

- Add constraint: in step 3, don't make a corridor from room X that links to a corridor linked to a corridor linked to a corridor linked to room X. Or, just ban making parallel corridors in step 3.

- Implement alternate algorithm where we first split the map in half. Then place a room in each half, so they can be directly linked. Then subsequent steps are each restricted to one half of the map. Could create more variety in layouts.