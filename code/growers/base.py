from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, Optional, TypeVar

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grower_context import GrowerContext, GrowerSeenState


C = TypeVar("C")
P = TypeVar("P")


@dataclass
class GrowerStepResult:
    """Describes the outcome of applying a single grower plan."""

    applied: bool
    stop: bool = False


class CandidateFinder(Generic[C, P]):
    """Locate potential growth opportunities in the current dungeon state."""

    def find_candidates(self, context: GrowerContext) -> Iterable[C]:
        raise NotImplementedError

    def on_success(self, context: GrowerContext, candidate: C, plan: P) -> None:
        """Hook called when a plan for the candidate is successfully applied."""
        return None


class GeometryPlanner(Generic[C, P]):
    """Validate a candidate and compute geometry to add to the dungeon."""

    def plan(self, context: GrowerContext, candidate: C) -> Optional[P]:
        raise NotImplementedError


class GrowerApplier(Generic[C, P]):
    """Commit a planned geometry change to the dungeon state."""

    def apply(self, context: GrowerContext, candidate: C, plan: P) -> GrowerStepResult:
        raise NotImplementedError

    def finalize(self, context: GrowerContext) -> int:
        """Perform any final bookkeeping; return the grower's reported result."""
        return 0


class DungeonGrower(Generic[C, P]):
    """Coordinates finder, planner, and applier to execute a grower."""

    def __init__(
        self,
        name: str,
        candidate_finder: CandidateFinder[C, P],
        geometry_planner: GeometryPlanner[C, P],
        applier: GrowerApplier[C, P],
    ) -> None:
        self.name = name
        self.candidate_finder = candidate_finder
        self.geometry_planner = geometry_planner
        self.applier = applier

    def run(self, context: GrowerContext) -> int:
        """Execute the grower pipeline and return the aggregate result."""
        seen_state = context.get_grower_seen_state(self.name)
        self._record_seen_layout(context, seen_state)
        for candidate in self.candidate_finder.find_candidates(context):
            plan = self.geometry_planner.plan(context, candidate)
            if plan is None:
                continue
            result = self.applier.apply(context, candidate, plan)
            if result.applied:
                self.candidate_finder.on_success(context, candidate, plan)
            if result.stop:
                break
        try:
            return self.applier.finalize(context)
        finally:
            self._record_seen_layout(context, seen_state)
            seen_state.register_run()

    def _record_seen_layout(
        self,
        context: GrowerContext,
        seen_state: "GrowerSeenState",
    ) -> None:
        rooms = (room.index for room in context.layout.placed_rooms if room.index is not None)
        corridors = (
            corridor.index for corridor in context.layout.corridors if corridor.index is not None
        )
        seen_state.note_seen(rooms, corridors)
