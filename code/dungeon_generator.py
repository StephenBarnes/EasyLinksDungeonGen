"""DungeonGenerator orchestrates the three steps of the easylink algorithm."""

from __future__ import annotations

from typing import Dict, List

from dungeon_config import DungeonConfig
from grower_context import GrowerContext
from models import RoomKind, RoomTemplate
from growers import (
    run_bent_room_to_corridor_grower,
    run_bent_room_to_room_grower,
    run_room_to_corridor_grower,
    run_room_to_room_grower,
)
from dungeon_layout import DungeonLayout
from root_room_placer import RootRoomPlacer


class DungeonGenerator:
    """Manages the overall process of generating a dungeon floor layout."""

    def __init__(self, config: DungeonConfig) -> None:
        self.config = config
        self.layout = DungeonLayout(config)

        self.room_templates = list(config.room_templates)
        self.room_templates_by_kind: Dict[RoomKind, List[RoomTemplate]] = {
            kind: [] for kind in RoomKind
        }
        for template in self.room_templates:
            for kind in template.kinds:
                self.room_templates_by_kind[kind].append(template)

        self.root_room_placer = RootRoomPlacer(
            config=self.config,
            layout=self.layout,
            room_templates_by_kind=self.room_templates_by_kind,
        )

    def generate(self) -> None:
        """Generates the dungeon, by invoking dungeon-growers."""
        # Note: This function is incomplete. Currently it runs our implemented growers in a fairly arbitrary order, mostly for testing. The final version will have more growers, and will include step 3 (counting components, deleting smaller components, and accepting or rejecting the final connected dungeon map).

        # Step 1: Place rooms, some with direct links
        self.root_room_placer.place_rooms()
        if not self.layout.placed_rooms:
            raise ValueError("ERROR: no placed rooms.")

        context = GrowerContext(
            config=self.config,
            layout=self.layout,
            room_templates=self.room_templates,
            room_templates_by_kind=self.room_templates_by_kind,
        )

        # Step 2: Run our growers repeatedly. Re-run simpler rules until they terminate.
        run_room_to_room_grower(context)
        num_created = 1
        while num_created > 0:
            num_created = run_room_to_corridor_grower(context, fill_probability=1)
            num_created += run_room_to_room_grower(context)
        num_created = run_bent_room_to_room_grower(context)
        while num_created > 0:
            num_created = run_room_to_room_grower(context)
            num_created += run_room_to_corridor_grower(context, fill_probability=1)
        num_created = run_bent_room_to_corridor_grower(context, fill_probability=1)
        while num_created > 0:
            num_created = run_room_to_room_grower(context)
            num_created += run_room_to_corridor_grower(context, fill_probability=1)
        
        # Additional growers will be invoked here, then step 3.
