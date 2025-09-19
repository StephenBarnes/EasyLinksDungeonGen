#!/usr/bin/env python3

# This file performs multiple runs of dungeon generation, collecting and reporting metrics.
# Used for testing both performance of the algorithm and quality of resulting dungeons.

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import math
import random
import statistics
import time
from typing import Callable, Dict, List

import networkx as nx

from dungeon_config import DungeonConfig, CorridorLengthDistribution
from dungeon_generator import DungeonGenerator
from room_templates import prototype_room_templates

# Default dungeon configuration mirrors the prototyping setup from main.py.
DEFAULT_CONFIG_KWARGS = dict(
    width=200,
    height=200,
    room_templates=prototype_room_templates,
    direct_link_counts_probs={0: 0.55, 1: 0.25, 2: 0.15, 3: 0.05},
    num_rooms_to_place=50,
    min_room_separation=1,
    min_intra_component_connection_distance=10,
    corridor_length_for_split=8,
    max_parallel_corridor_perpendicular_distance=8,
    max_parallel_corridor_overlap=5,
    min_rooms_required=10,
    initial_corridor_length=CorridorLengthDistribution(
        min_length=5,
        max_length=40,
        median_length=10,
    ),
    collect_metrics=True,
)

DEFAULT_ROOM_COMPLETION_THRESHOLD_RATIO = 0.8
DEFAULT_AREA_COVERAGE_THRESHOLD = 0.2
DEFAULT_CYCLE_COUNT_THRESHOLD = 1
DEFAULT_CYCLE_LENGTH_THRESHOLD = 5

PERCENTILES = [1.0, 5.0] + [float(value) for value in range(10, 100, 5)] + [99.0]


def build_config(seed: int) -> DungeonConfig:
    return DungeonConfig(random_seed=seed, **DEFAULT_CONFIG_KWARGS) # type: ignore


@dataclass
class GenerationRunResult:
    seed: int
    duration: float
    total_rooms: int
    total_corridors: int
    diversity_score: float
    gini_coefficient: float
    room_target: int
    room_acceptance_threshold: int
    meets_room_threshold: bool
    map_area: int
    bounding_box_area: int
    bounding_box_area_fraction: float
    cycle_count: int
    cycle_lengths: List[int]
    template_counts: Counter[str]
    grower_metrics: Dict[str, Dict[str, float | int]]


def gini_coefficient(counts: List[int]) -> float:
    """Compute the Gini coefficient for a list of non-negative counts."""
    data = [value for value in counts if value > 0]
    if not data:
        return 0.0
    data.sort()
    total = sum(data)
    if total <= 0:
        return 0.0
    n = len(data)
    weighted_sum = 0.0
    for index, value in enumerate(data, start=1):
        weighted_sum += index * value
    return (2.0 * weighted_sum) / (n * total) - (n + 1) / n


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    fraction = rank - lower
    return lower_value + (upper_value - lower_value) * fraction


def fraction_at_least(values: List[float], threshold: float) -> float:
    if not values:
        return float("nan")
    return sum(1 for value in values if value >= threshold) / len(values)


def format_fraction(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.1%}"


def format_value(value: float, formatter: Callable[[float], str] | None = None) -> str:
    numeric = float(value)
    if math.isnan(numeric):
        return "nan"
    if formatter is None:
        return f"{numeric:.3f}"
    return formatter(numeric)


def collected_percentiles(values: List[float]) -> List[tuple[float, float]]:
    return [(pct, percentile(values, pct)) for pct in PERCENTILES]


def compute_basic_stats(values: List[float]) -> Dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else float("nan"),
    }


@dataclass
class MetricDefinition:
    name: str
    values: List[float]
    value_formatter: Callable[[float], str] | None = None
    success_threshold: float | None = None
    success_label: str | None = None
    notes: str | None = None


def report_metric(definition: MetricDefinition) -> None:
    values = definition.values
    print(definition.name + ":")
    if not values:
        print("  (no data)")
        if definition.notes:
            print(f"  {definition.notes}")
        return

    stats = compute_basic_stats(values)
    mean = format_value(stats["mean"], definition.value_formatter)
    median = format_value(stats["median"], definition.value_formatter)
    minimum = format_value(stats["min"], definition.value_formatter)
    maximum = format_value(stats["max"], definition.value_formatter)
    stdev = format_value(stats["stdev"], definition.value_formatter)

    print(
        "  Count {count}, mean {mean}, median {median}, min {min}, max {max}, stdev {stdev}".format(
            count=len(values),
            mean=mean,
            median=median,
            min=minimum,
            max=maximum,
            stdev=stdev,
        )
    )

    percentile_parts = []
    for pct, value in collected_percentiles(values):
        label = f"p{int(pct)}" if float(pct).is_integer() else f"p{pct:g}"
        percentile_parts.append(f"{label}={format_value(value, definition.value_formatter)}")
    print("  Percentiles: " + ", ".join(percentile_parts))

    if definition.success_threshold is not None:
        success_rate = fraction_at_least(values, definition.success_threshold)
        label = definition.success_label or (
            ">= " + format_value(definition.success_threshold, definition.value_formatter)
        )
        print(
            "  Success rate {success} ({label})".format(
                success=format_fraction(success_rate),
                label=label,
            )
        )

    if definition.notes:
        print(f"  {definition.notes}")


def bounding_box_area(layout) -> tuple[int, int, float]:
    if not layout.placed_rooms:
        total_area = layout.config.width * layout.config.height
        return 0, total_area, 0.0

    min_x = min(room.x for room in layout.placed_rooms)
    max_x = max(room.x + room.width for room in layout.placed_rooms)
    min_y = min(room.y for room in layout.placed_rooms)
    max_y = max(room.y + room.height for room in layout.placed_rooms)

    width = max(0, max_x - min_x)
    height = max(0, max_y - min_y)
    area = width * height
    total_area = layout.config.width * layout.config.height
    fraction = area / total_area if total_area > 0 else 0.0
    return area, total_area, fraction


def build_room_graph(layout) -> nx.Graph:
    graph = nx.Graph()
    for room in layout.placed_rooms:
        graph.add_node(room.index)

    for corridor in layout.corridors:
        room_a = corridor.room_a_index
        room_b = corridor.room_b_index
        if room_a is not None and room_b is not None:
            graph.add_edge(room_a, room_b)

    for room_a, room_b in layout.room_room_links:
        graph.add_edge(room_a, room_b)

    return graph


def run_single_generation(seed: int, room_completion_ratio: float) -> GenerationRunResult:
    """Run one dungeon generation with the provided seed and collect metrics."""
    config = build_config(seed)

    random.seed(seed)

    generator = DungeonGenerator(config)

    start = time.perf_counter()
    generator.generate()
    end = time.perf_counter()

    layout = generator.layout
    total_rooms = len(layout.placed_rooms)

    room_target = layout.config.num_rooms_to_place
    room_threshold = math.floor(room_target * room_completion_ratio)
    meets_room_threshold = total_rooms >= room_threshold

    bbox_area, map_area, bbox_fraction = bounding_box_area(layout)

    graph = build_room_graph(layout)
    basis = nx.cycle_basis(graph)
    cycle_lengths = [len(cycle) for cycle in basis]
    cycle_count = len(cycle_lengths)

    template_counts: Counter[str] = Counter(
        room.template.name for room in layout.placed_rooms
    )
    gini = gini_coefficient(list(template_counts.values()))
    diversity = 1.0 - gini

    grower_metrics = generator.metrics.snapshot() if generator.metrics else {}

    return GenerationRunResult(
        seed=seed,
        duration=end - start,
        total_rooms=total_rooms,
        total_corridors=len(layout.corridors),
        diversity_score=diversity,
        gini_coefficient=gini,
        room_target=room_target,
        room_acceptance_threshold=room_threshold,
        meets_room_threshold=meets_room_threshold,
        map_area=map_area,
        bounding_box_area=bbox_area,
        bounding_box_area_fraction=bbox_fraction,
        cycle_count=cycle_count,
        cycle_lengths=cycle_lengths,
        template_counts=template_counts,
        grower_metrics=grower_metrics,
    )


def run_benchmark(
    num_runs: int, seed: int | None, room_completion_ratio: float
) -> List[GenerationRunResult]:
    """Run the generator multiple times and collect run-level metrics."""
    rng = random.Random(seed)

    results: List[GenerationRunResult] = []

    for _ in range(num_runs):
        run_seed = rng.randint(0, 1_000_000)
        result = run_single_generation(run_seed, room_completion_ratio)
        results.append(result)

    return results


def format_seconds(value: float) -> str:
    if value >= 1.0:
        return f"{value:.3f}s"
    return f"{value * 1000:.1f}ms"


def aggregate_grower_metrics(
    results: List[GenerationRunResult],
) -> Dict[str, Dict[str, float]]:
    totals: Dict[str, Dict[str, float]] = {}
    for result in results:
        for name, metrics in result.grower_metrics.items():
            aggregate = totals.setdefault(
                name,
                {
                    "invocations": 0.0,
                    "total_time": 0.0,
                    "total_rooms_added": 0.0,
                    "total_corridors_added": 0.0,
                },
            )
            aggregate["invocations"] += float(metrics.get("invocations", 0))
            aggregate["total_time"] += float(metrics.get("total_time", 0.0))
            aggregate["total_rooms_added"] += float(metrics.get("total_rooms_added", 0.0))
            aggregate["total_corridors_added"] += float(
                metrics.get("total_corridors_added", 0.0)
            )

    for aggregate in totals.values():
        invocations = aggregate["invocations"]
        aggregate["average_time"] = (
            aggregate["total_time"] / invocations if invocations else 0.0
        )
        aggregate["average_rooms_added"] = (
            aggregate["total_rooms_added"] / invocations if invocations else 0.0
        )
        aggregate["average_corridors_added"] = (
            aggregate["total_corridors_added"] / invocations if invocations else 0.0
        )
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the dungeon generator multiple times and report timing and quality statistics."
        )
    )
    parser.add_argument(
        "-n",
        "--runs",
        type=int,
        default=20,
        help="Number of dungeon generations to execute (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional seed for the benchmark harness RNG; keeps run seeds reproducible"
        ),
    )
    parser.add_argument(
        "--min-diversity",
        type=float,
        default=0.4,
        help=(
            "Minimum acceptable diversity score (1 - Gini coefficient) for room templates"
        ),
    )
    parser.add_argument(
        "--room-completion-threshold-ratio",
        type=float,
        default=DEFAULT_ROOM_COMPLETION_THRESHOLD_RATIO,
        help=(
            "Fraction of the target room count required for a run to be considered successful"
        ),
    )
    parser.add_argument(
        "--area-coverage-threshold",
        type=float,
        default=DEFAULT_AREA_COVERAGE_THRESHOLD,
        help="Minimum bounding-box area fraction of the map to consider coverage acceptable",
    )
    parser.add_argument(
        "--cycle-count-threshold",
        type=float,
        default=DEFAULT_CYCLE_COUNT_THRESHOLD,
        help="Minimum cycle count in the room graph for success evaluation",
    )
    parser.add_argument(
        "--cycle-length-threshold",
        type=float,
        default=DEFAULT_CYCLE_LENGTH_THRESHOLD,
        help="Minimum cycle length for success evaluation",
    )
    args = parser.parse_args()

    if args.runs <= 0:
        raise SystemExit("Number of runs must be a positive integer")
    if not (0.0 <= args.min_diversity <= 1.0):
        raise SystemExit("Minimum diversity must be within [0, 1]")
    if not (0.0 < args.room_completion_threshold_ratio <= 1.0):
        raise SystemExit("Room completion threshold ratio must be within (0, 1]")
    if not (0.0 <= args.area_coverage_threshold <= 1.0):
        raise SystemExit("Area coverage threshold must be within [0, 1]")
    if args.cycle_count_threshold < 0.0:
        raise SystemExit("Cycle count threshold must be non-negative")
    if args.cycle_length_threshold < 0.0:
        raise SystemExit("Cycle length threshold must be non-negative")

    results = run_benchmark(
        args.runs, args.seed, args.room_completion_threshold_ratio
    )
    if not results:
        raise SystemExit("No runs executed")

    durations = [result.duration for result in results]
    run_seeds = [result.seed for result in results]

    worst_duration = max(durations)
    worst_index = durations.index(worst_duration)
    worst_seed = run_seeds[worst_index]

    rooms_values = [float(result.total_rooms) for result in results]
    corridors_values = [float(result.total_corridors) for result in results]
    room_target = results[0].room_target
    room_threshold = results[0].room_acceptance_threshold
    bounding_fractions = [result.bounding_box_area_fraction for result in results]
    cycle_counts = [float(result.cycle_count) for result in results]
    cycle_lengths_all = [float(length) for result in results for length in result.cycle_lengths]

    diversity_scores = [r.diversity_score for r in results]

    for idx, result in enumerate(results, start=1):
        run_status = "ok" if result.meets_room_threshold else "low"
        cycle_lengths_display = (
            ", ".join(str(length) for length in result.cycle_lengths)
            if result.cycle_lengths
            else "-"
        )
        print(
            "Run {idx:02d}: {time} (seed {seed}) | rooms {rooms}/{target}"
            " (>= {threshold}? {status}) | diversity {diversity:.3f}".format(
                idx=idx,
                time=format_seconds(result.duration),
                seed=result.seed,
                rooms=result.total_rooms,
                target=room_target,
                threshold=room_threshold,
                status=run_status,
                diversity=result.diversity_score,
            )
        )
        print(
            "  bbox {fraction:.3f} of map (area {area}/{map_area}),"
            " cycles {cycles} [{lengths}]".format(
                fraction=result.bounding_box_area_fraction,
                area=result.bounding_box_area,
                map_area=result.map_area,
                cycles=result.cycle_count,
                lengths=cycle_lengths_display,
            )
        )

    total_template_counts: Counter[str] = Counter()
    total_rooms = 0
    for result in results:
        total_template_counts.update(result.template_counts)
        total_rooms += result.total_rooms

    grower_totals = aggregate_grower_metrics(results)

    print()
    print(f"Config runs: {args.runs}")
    print(
        f"Worst-case generation time: {format_seconds(worst_duration)} (seed {worst_seed})"
    )
    metrics_to_report = [
        MetricDefinition(
            name="Generation time",
            values=durations,
            value_formatter=lambda value: f"{value:.4f}s",
        ),
        MetricDefinition(
            name="Rooms placed",
            values=rooms_values,
            value_formatter=lambda value: f"{value:.0f}",
            success_threshold=float(room_threshold),
            success_label=(
                f">= {format_value(room_threshold, lambda value: f'{value:.0f}')}"
                f" (target {room_target})"
            ),
        ),
        MetricDefinition(
            name="Corridors placed",
            values=corridors_values,
            value_formatter=lambda value: f"{value:.0f}",
        ),
        MetricDefinition(
            name="Room diversity (1 - Gini)",
            values=diversity_scores,
            value_formatter=lambda value: f"{value:.3f}",
            success_threshold=args.min_diversity,
            success_label=f">= {args.min_diversity:.3f}",
        ),
        MetricDefinition(
            name="Bounding coverage",
            values=bounding_fractions,
            value_formatter=lambda value: f"{value:.1%}",
            success_threshold=args.area_coverage_threshold,
            success_label=(
                f">= {format_value(args.area_coverage_threshold, lambda value: f'{value:.1%}')}"
            ),
        ),
        MetricDefinition(
            name="Cycle count",
            values=cycle_counts,
            value_formatter=lambda value: f"{value:.1f}",
            success_threshold=args.cycle_count_threshold,
            success_label=(
                f">= {format_value(args.cycle_count_threshold, lambda value: f'{value:.1f}')}"
            ),
        ),
    ]

    cycle_length_notes = "No cycles observed across runs." if not cycle_lengths_all else None
    metrics_to_report.append(
        MetricDefinition(
            name="Cycle length",
            values=cycle_lengths_all,
            value_formatter=lambda value: f"{value:.1f}",
            success_threshold=args.cycle_length_threshold if cycle_lengths_all else None,
            success_label=(
                f">= {format_value(args.cycle_length_threshold, lambda value: f'{value:.1f}')}"
            )
            if cycle_lengths_all
            else None,
            notes=cycle_length_notes,
        )
    )

    for metric in metrics_to_report:
        print()
        report_metric(metric)

    if total_rooms > 0:
        print()
        print("Room template distribution across runs:")
        for template_name, count in total_template_counts.most_common():
            share = count / total_rooms
            print(
                f"  {template_name}: {count} rooms ({share:.1%} of {total_rooms} total rooms)"
            )

    if grower_totals:
        print()
        print("Grower performance summary:")
        for name, metrics in sorted(
            grower_totals.items(), key=lambda item: item[1]["total_time"], reverse=True
        ):
            invocations = int(metrics["invocations"])
            print(
                "  {name}: invocations={invocations}, total_time={total_time},"
                " avg_time={avg_time}, avg_rooms={avg_rooms:.2f}, avg_corridors={avg_corridors:.2f}".format(
                    name=name,
                    invocations=invocations,
                    total_time=format_seconds(metrics["total_time"]),
                    avg_time=format_seconds(metrics["average_time"]),
                    avg_rooms=metrics["average_rooms_added"],
                    avg_corridors=metrics["average_corridors_added"],
                )
            )


if __name__ == "__main__":
    main()
