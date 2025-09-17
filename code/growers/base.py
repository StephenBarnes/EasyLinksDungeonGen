from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, Optional, TypeVar

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dungeon_generator import DungeonGenerator


C = TypeVar("C")
P = TypeVar("P")


@dataclass
class GrowerStepResult:
    """Describes the outcome of applying a single grower plan."""

    applied: bool
    stop: bool = False


class CandidateFinder(Generic[C, P]):
    """Locate potential growth opportunities in the current dungeon state."""

    def find_candidates(self, generator: DungeonGenerator) -> Iterable[C]:
        raise NotImplementedError

    def on_success(self, generator: DungeonGenerator, candidate: C, plan: P) -> None:
        """Hook called when a plan for the candidate is successfully applied."""
        return None


class GeometryPlanner(Generic[C, P]):
    """Validate a candidate and compute geometry to add to the dungeon."""

    def plan(self, generator: DungeonGenerator, candidate: C) -> Optional[P]:
        raise NotImplementedError


class GrowerApplier(Generic[C, P]):
    """Commit a planned geometry change to the dungeon state."""

    def apply(self, generator: DungeonGenerator, candidate: C, plan: P) -> GrowerStepResult:
        raise NotImplementedError

    def finalize(self, generator: DungeonGenerator) -> int:
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

    def run(self, generator: DungeonGenerator) -> int:
        """Execute the grower pipeline and return the aggregate result."""
        for candidate in self.candidate_finder.find_candidates(generator):
            plan = self.geometry_planner.plan(generator, candidate)
            if plan is None:
                continue
            result = self.applier.apply(generator, candidate, plan)
            if result.applied:
                self.candidate_finder.on_success(generator, candidate, plan)
            if result.stop:
                break
        return self.applier.finalize(generator)
