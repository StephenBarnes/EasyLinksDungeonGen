#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import statistics
import time
from typing import List, Tuple

from dungeon_config import DungeonConfig
from dungeon_generator import DungeonGenerator
from room_templates import prototype_room_templates

# Default dungeon configuration mirrors the prototyping setup from main.py.
DEFAULT_CONFIG_KWARGS = dict(
    width=80,
    height=50,
    room_templates=prototype_room_templates,
    direct_link_counts_probs={0: 0.55, 1: 0.25, 2: 0.15, 3: 0.05},
    num_rooms_to_place=15,
    min_room_separation=1,
    min_intra_component_connection_distance=10,
    max_desired_corridor_length=8,
    max_parallel_corridor_perpendicular_distance=8,
    max_parallel_corridor_overlap=5,
    min_rooms_required=10,
)


def build_config(seed: int) -> DungeonConfig:
    return DungeonConfig(random_seed=seed, **DEFAULT_CONFIG_KWARGS) # type: ignore


def run_single_generation(seed: int) -> float:
    """Run one dungeon generation with the provided seed, returning elapsed seconds."""
    config = build_config(seed)

    # Keep the global RNG aligned with the config seed for reproducibility across runs.
    random.seed(seed)

    generator = DungeonGenerator(config)

    start = time.perf_counter()
    generator.generate()
    end = time.perf_counter()

    return end - start


def run_benchmark(num_runs: int, seed: int | None) -> Tuple[List[int], List[float]]:
    """Run the generator multiple times and collect seeds and durations."""
    rng = random.Random(seed)

    run_seeds: List[int] = []
    durations: List[float] = []

    for _ in range(num_runs):
        run_seed = rng.randint(0, 1_000_000)
        duration = run_single_generation(run_seed)
        run_seeds.append(run_seed)
        durations.append(duration)

    return run_seeds, durations


def format_seconds(value: float) -> str:
    if value >= 1.0:
        return f"{value:.3f}s"
    return f"{value * 1000:.1f}ms"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the dungeon generator multiple times and report timing statistics."
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
    args = parser.parse_args()

    if args.runs <= 0:
        raise SystemExit("Number of runs must be a positive integer")

    run_seeds, durations = run_benchmark(args.runs, args.seed)

    mean_duration = statistics.mean(durations)
    median_duration = statistics.median(durations)
    worst_duration = max(durations)
    worst_index = durations.index(worst_duration)
    worst_seed = run_seeds[worst_index]

    for idx, (seed, duration) in enumerate(zip(run_seeds, durations), start=1):
        print(f"Run {idx:02d}: {format_seconds(duration)} (seed {seed})")

    print()
    print(f"Config runs: {args.runs}")
    print(f"Mean generation time: {format_seconds(mean_duration)}")
    print(f"Median generation time: {format_seconds(median_duration)}")
    print(
        f"Worst-case generation time: {format_seconds(worst_duration)} (seed {worst_seed})"
    )


if __name__ == "__main__":
    main()
