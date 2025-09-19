"""DungeonGenerator orchestrates the three steps of the easylink algorithm."""

from __future__ import annotations

from time import perf_counter
from typing import Callable, Dict, List

from dungeon_config import DungeonConfig
from grower_context import GrowerContext
from models import RoomKind, RoomTemplate
from growers import (
    run_bent_room_to_room_grower,
    run_room_to_corridor_grower,
    run_room_to_room_grower,
    run_bent_room_to_corridor_grower,
    run_through_corridor_grower,
    run_rotate_rooms_grower,
    run_initial_tree_grower,
)
from dungeon_layout import DungeonLayout
from metrics import GenerationMetrics


class DungeonGenerator:
    """Manages the overall process of generating a dungeon floor layout."""

    def __init__(self, config: DungeonConfig) -> None:
        self.config = config
        self.layout = DungeonLayout(config)
        self.metrics = GenerationMetrics() if config.collect_metrics else None

        self.room_templates = list(config.room_templates)
        self.room_templates_by_kind: Dict[RoomKind, List[RoomTemplate]] = {
            kind: [] for kind in RoomKind
        }
        for template in self.room_templates:
            for kind in template.kinds:
                self.room_templates_by_kind[kind].append(template)

    def _run_grower(
        self,
        name: str,
        func: Callable[..., int],
        *args,
        **kwargs,
    ) -> int:
        if self.metrics is None:
            return func(*args, **kwargs)

        rooms_before = len(self.layout.placed_rooms)
        corridors_before = len(self.layout.corridors)
        start = perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            duration = perf_counter() - start
            rooms_delta = len(self.layout.placed_rooms) - rooms_before
            corridors_delta = len(self.layout.corridors) - corridors_before
            self.metrics.record_grower_run(
                name,
                duration,
                rooms_delta,
                corridors_delta,
            )

    def generate(self) -> None:
        """Generates the dungeon, by invoking dungeon-growers."""
        # Note: This function is incomplete. Currently it runs our implemented growers in a fairly arbitrary order, mostly for testing. The final version will have more growers, and will include step 3 (counting components, deleting smaller components, and accepting or rejecting the final connected dungeon map).

        context = GrowerContext(
            config=self.config,
            layout=self.layout,
            room_templates=self.room_templates,
            room_templates_by_kind=self.room_templates_by_kind,
        )

        # Step 1: Place rooms, some with direct links, using the initial-tree grower
        self._run_grower("initial_tree", run_initial_tree_grower, context)
        if not self.layout.placed_rooms:
            raise ValueError("ERROR: no placed rooms.")

        # Step 2: Run our growers repeatedly. Re-run simpler rules until they terminate.
        def run_connection_growers() -> int:
            total_created = self._run_grower("room_to_room", run_room_to_room_grower, context)
            num_created_local = 1
            while num_created_local > 0:
                num_created_local = self._run_grower(
                    "room_to_corridor",
                    run_room_to_corridor_grower,
                    context,
                    fill_probability=1,
                )
                num_created_local += self._run_grower(
                    "room_to_room", run_room_to_room_grower, context
                )
                total_created += num_created_local

            num_created_local = self._run_grower(
                "bent_room_to_room",
                run_bent_room_to_room_grower,
                context,
                stop_after_first=True,
            )
            total_created += num_created_local
            while num_created_local > 0:
                num_created_local = self._run_grower(
                    "room_to_room", run_room_to_room_grower, context
                )
                num_created_local += self._run_grower(
                    "room_to_corridor",
                    run_room_to_corridor_grower,
                    context,
                    fill_probability=1,
                )
                total_created += num_created_local
                if num_created_local == 0:
                    num_created_local = self._run_grower(
                        "bent_room_to_room",
                        run_bent_room_to_room_grower,
                        context,
                        stop_after_first=True,
                    )
                    total_created += num_created_local

            num_created_local = self._run_grower(
                "bent_room_to_corridor",
                run_bent_room_to_corridor_grower,
                context,
                stop_after_first=True,
                fill_probability=1,
            )
            total_created += num_created_local
            while num_created_local > 0:
                num_created_local = self._run_grower(
                    "room_to_room", run_room_to_room_grower, context
                )
                num_created_local += self._run_grower(
                    "room_to_corridor",
                    run_room_to_corridor_grower,
                    context,
                    fill_probability=1,
                )
                total_created += num_created_local
                if num_created_local == 0:
                    num_created_local = self._run_grower(
                        "bent_room_to_corridor",
                        run_bent_room_to_corridor_grower,
                        context,
                        stop_after_first=True,
                        fill_probability=1,
                    )
                    total_created += num_created_local
            return total_created

        num_created = 1
        while num_created > 0:
            num_created = run_connection_growers()

        num_created = 1
        while num_created > 0:
            num_created = 0
            rotated_rooms = self._run_grower(
                "rotate_rooms", run_rotate_rooms_grower, context
            )
            if rotated_rooms > 0:
                num_created = run_connection_growers()

        num_created = self._run_grower(
            "through_corridor", run_through_corridor_grower, context
        )
        while num_created > 0:
            num_created = self._run_grower(
                "through_corridor", run_through_corridor_grower, context
            )
            num_created += run_connection_growers()

        # Additional growers will be invoked here, then step 3.
