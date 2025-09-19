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
from typing import Dict, List

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

ROOM_COMPLETION_THRESHOLD_RATIO = 0.8
AREA_COVERAGE_THRESHOLD = 0.2
CYCLE_COUNT_THRESHOLD = 1
CYCLE_LENGTH_THRESHOLD = 5


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


def describe_distribution(values: List[float]) -> str:
    if not values:
        return "mean nan, median nan, p10 nan, p90 nan"
    mean_val = statistics.mean(values)
    median_val = statistics.median(values)
    p10 = percentile(values, 10.0)
    p90 = percentile(values, 90.0)
    return (
        f"mean {mean_val:.3f}, median {median_val:.3f}, p10 {p10:.3f}, p90 {p90:.3f}"
    )


def fraction_at_least(values: List[float], threshold: float) -> float:
    if not values:
        return float("nan")
    return sum(1 for value in values if value >= threshold) / len(values)


def format_fraction(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.1%}"


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


def run_single_generation(seed: int) -> GenerationRunResult:
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
    room_threshold = math.floor(room_target * ROOM_COMPLETION_THRESHOLD_RATIO)
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


def run_benchmark(num_runs: int, seed: int | None) -> List[GenerationRunResult]:
    """Run the generator multiple times and collect run-level metrics."""
    rng = random.Random(seed)

    results: List[GenerationRunResult] = []

    for _ in range(num_runs):
        run_seed = rng.randint(0, 1_000_000)
        result = run_single_generation(run_seed)
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
    args = parser.parse_args()

    if args.runs <= 0:
        raise SystemExit("Number of runs must be a positive integer")

    results = run_benchmark(args.runs, args.seed)
    if not results:
        raise SystemExit("No runs executed")

    durations = [result.duration for result in results]
    run_seeds = [result.seed for result in results]

    mean_duration = statistics.mean(durations)
    median_duration = statistics.median(durations)
    worst_duration = max(durations)
    worst_index = durations.index(worst_duration)
    worst_seed = run_seeds[worst_index]

    rooms_values = [float(result.total_rooms) for result in results]
    room_target = results[0].room_target
    room_threshold = results[0].room_acceptance_threshold
    bounding_fractions = [result.bounding_box_area_fraction for result in results]
    cycle_counts = [float(result.cycle_count) for result in results]
    cycle_lengths_all = [float(length) for result in results for length in result.cycle_lengths]

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

    diversity_scores = [r.diversity_score for r in results]
    mean_diversity = statistics.mean(diversity_scores)
    median_diversity = statistics.median(diversity_scores)
    diversity_p5 = percentile(diversity_scores, 5.0)
    diversity_success = sum(
        1 for value in diversity_scores if value >= args.min_diversity
    ) / len(results)

    total_template_counts: Counter[str] = Counter()
    total_rooms = 0
    for result in results:
        total_template_counts.update(result.template_counts)
        total_rooms += result.total_rooms

    grower_totals = aggregate_grower_metrics(results)

    print()
    print(f"Config runs: {args.runs}")
    print(f"Mean generation time: {format_seconds(mean_duration)}")
    print(f"Median generation time: {format_seconds(median_duration)}")
    print(
        f"Worst-case generation time: {format_seconds(worst_duration)} (seed {worst_seed})"
    )
    print(
        "Room diversity (1 - Gini): mean {mean:.3f}, median {median:.3f},"
        " p5 {p5:.3f}, success rate {success:.1%} for threshold {threshold:.3f}".format(
            mean=mean_diversity,
            median=median_diversity,
            p5=diversity_p5,
            success=diversity_success,
            threshold=args.min_diversity,
        )
    )

    rooms_success = fraction_at_least(rooms_values, float(room_threshold))
    area_success = fraction_at_least(bounding_fractions, AREA_COVERAGE_THRESHOLD)
    cycle_success = fraction_at_least(cycle_counts, float(CYCLE_COUNT_THRESHOLD))
    cycle_length_success = fraction_at_least(
        cycle_lengths_all, float(CYCLE_LENGTH_THRESHOLD)
    )

    print(
        "Rooms placed: {distribution}, success rate {success} for"
        " threshold >= {threshold} (target {target}).".format(
            distribution=describe_distribution(rooms_values),
            success=format_fraction(rooms_success),
            threshold=room_threshold,
            target=room_target,
        )
    )
    print(
        "Bounding coverage: {distribution}, success rate {success} for"
        " threshold {threshold}.".format(
            distribution=describe_distribution(bounding_fractions),
            success=format_fraction(area_success),
            threshold=f"{AREA_COVERAGE_THRESHOLD:.0%}",
        )
    )
    print(
        "Cycle count: {distribution}, success rate {success} for"
        " threshold >= {threshold}.".format(
            distribution=describe_distribution(cycle_counts),
            success=format_fraction(cycle_success),
            threshold=CYCLE_COUNT_THRESHOLD,
        )
    )
    if cycle_lengths_all:
        print(
            "Cycle length: {distribution}, success rate {success} for"
            " threshold >= {threshold}.".format(
                distribution=describe_distribution(cycle_lengths_all),
                success=format_fraction(cycle_length_success),
                threshold=CYCLE_LENGTH_THRESHOLD,
            )
        )
    else:
        print("Cycle length: no cycles observed across runs.")

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
