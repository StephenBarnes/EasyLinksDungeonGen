"""Helpers for collecting instrumentation data during dungeon generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class GrowerMetrics:
    """Aggregated metrics for a single grower across invocations."""

    name: str
    invocations: int = 0
    total_time: float = 0.0
    total_rooms_added: int = 0
    total_corridors_added: int = 0

    def record(self, duration: float, rooms_delta: int, corridors_delta: int) -> None:
        self.invocations += 1
        self.total_time += duration
        self.total_rooms_added += rooms_delta
        self.total_corridors_added += corridors_delta

    def to_dict(self) -> Dict[str, float | int]:
        average_time = self.total_time / self.invocations if self.invocations else 0.0
        average_rooms = (
            self.total_rooms_added / self.invocations if self.invocations else 0.0
        )
        average_corridors = (
            self.total_corridors_added / self.invocations if self.invocations else 0.0
        )
        return {
            "invocations": self.invocations,
            "total_time": self.total_time,
            "average_time": average_time,
            "total_rooms_added": self.total_rooms_added,
            "average_rooms_added": average_rooms,
            "total_corridors_added": self.total_corridors_added,
            "average_corridors_added": average_corridors,
        }


@dataclass
class GenerationMetrics:
    """Container for grower metrics recorded during a generation run."""

    growers: Dict[str, GrowerMetrics] = field(default_factory=dict)

    def record_grower_run(
        self,
        name: str,
        duration: float,
        rooms_delta: int,
        corridors_delta: int,
    ) -> None:
        metrics = self.growers.get(name)
        if metrics is None:
            metrics = GrowerMetrics(name=name)
            self.growers[name] = metrics
        metrics.record(duration, rooms_delta, corridors_delta)

    def snapshot(self) -> Dict[str, Dict[str, float | int]]:
        return {name: metrics.to_dict() for name, metrics in self.growers.items()}

