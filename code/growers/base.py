from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, Optional, Tuple, TypeVar

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grower_context import GrowerContext


C = TypeVar("C")
P = TypeVar("P")


@dataclass
class GrowerStepResult:
    """Describes the outcome of applying a single grower plan."""

    applied: bool
    stop: bool = False


@dataclass(frozen=True)
class CandidateDependencies:
    """Identifies existing layout entities a candidate relies on."""

    rooms: Tuple[int, ...] = ()
    corridors: Tuple[int, ...] = ()

    @classmethod
    def from_iterables(
        cls,
        *,
        rooms: Iterable[int] = (),
        corridors: Iterable[int] = (),
    ) -> "CandidateDependencies":
        room_tuple = tuple(sorted({idx for idx in rooms if idx is not None}))
        corridor_tuple = tuple(sorted({idx for idx in corridors if idx is not None}))
        return cls(rooms=room_tuple, corridors=corridor_tuple)

    def merged_with(self, other: "CandidateDependencies") -> "CandidateDependencies":
        return CandidateDependencies.from_iterables(
            rooms=(*self.rooms, *other.rooms),
            corridors=(*self.corridors, *other.corridors),
        )


class CandidateFinder(Generic[C, P]):
    """Locate potential growth opportunities in the current dungeon state."""

    def find_candidates(self, context: GrowerContext) -> Iterable[C]:
        raise NotImplementedError

    def dependencies(
        self,
        context: GrowerContext,
        candidate: C,
    ) -> CandidateDependencies:
        return CandidateDependencies()

    def on_success(self, context: GrowerContext, candidate: C, plan: P) -> None:
        """Hook called when a plan for the candidate is successfully applied."""
        return None


class GeometryPlanner(Generic[C, P]):
    """Validate a candidate and compute geometry to add to the dungeon."""

    def plan(self, context: GrowerContext, candidate: C) -> Optional[P]:
        raise NotImplementedError

    def dependencies(
        self,
        context: GrowerContext,
        candidate: C,
        plan: P,
    ) -> CandidateDependencies:
        return CandidateDependencies()


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
        for candidate in self.candidate_finder.find_candidates(context):
            base_dependencies = self.candidate_finder.dependencies(context, candidate)
            if not context.should_consider_growth(
                self.name,
                rooms=base_dependencies.rooms,
                corridors=base_dependencies.corridors,
            ):
                continue
            plan = self.geometry_planner.plan(context, candidate)
            context.record_growth_seen(
                self.name,
                rooms=base_dependencies.rooms,
                corridors=base_dependencies.corridors,
            )
            if plan is None:
                continue
            plan_dependencies = self.geometry_planner.dependencies(
                context, candidate, plan
            )
            if plan_dependencies.rooms or plan_dependencies.corridors:
                context.record_growth_seen(
                    self.name,
                    rooms=plan_dependencies.rooms,
                    corridors=plan_dependencies.corridors,
                )
            result = self.applier.apply(context, candidate, plan)
            if result.applied:
                self.candidate_finder.on_success(context, candidate, plan)
            if result.stop:
                break
        result_value = self.applier.finalize(context)
        context.get_grower_seen_state(self.name).register_run(context.layout)
        return result_value
