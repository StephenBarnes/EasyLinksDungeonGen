#!/usr/bin/env python3
"""Compare two dungeon benchmark JSON files and highlight meaningful differences."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two dungeon benchmark JSON files. Provide zero or two paths; "
            "with zero paths the two newest benchmarks in the benchmarks/ directory are compared."
        )
    )
    parser.add_argument(
        "benchmark_files",
        nargs="*",
        help="Paths to two benchmark JSON files to compare.",
    )
    parser.add_argument(
        "--benchmarks-dir",
        default="benchmarks",
        help="Directory used when no explicit benchmark files are supplied (default: benchmarks).",
    )
    args = parser.parse_args()
    if len(args.benchmark_files) not in (0, 2):
        parser.error("Provide either zero or two benchmark files.")
    return args


def find_latest_benchmarks(directory: Path, count: int = 2) -> List[Path]:
    if not directory.exists():
        raise SystemExit(f"Benchmark directory '{directory}' does not exist.")
    candidates = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name.startswith("benchmark-") and path.suffix == ".json"
    )
    if len(candidates) < count:
        raise SystemExit(
            f"Expected at least {count} benchmark files in '{directory}', found {len(candidates)}."
        )
    return candidates[-count:]


def read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise SystemExit(f"Could not open benchmark file '{path}': {exc}.")


def format_path(parts: Iterable[str]) -> str:
    return ".".join(parts)


def format_basic(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        if math.isfinite(value):
            if math.isclose(value, round(value)):
                return str(int(round(value)))
            return f"{value:.6g}"
        return str(value)
    return str(value)


def compare_configs(a: Any, b: Any, path: Tuple[str, ...]) -> List[str]:
    differences: List[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a.keys()) | set(b.keys()))
        for key in keys:
            if key == "random_seed":
                continue
            sub_path = path + (key,)
            if key not in a:
                differences.append(
                    f"{format_path(sub_path)}: (missing) -> {format_basic(b[key])}"
                )
            elif key not in b:
                differences.append(
                    f"{format_path(sub_path)}: {format_basic(a[key])} -> (missing)"
                )
            else:
                differences.extend(compare_configs(a[key], b[key], sub_path))
        return differences
    if isinstance(a, list) and isinstance(b, list):
        if path and path[-1] == "room_templates":
            set_a = set(a)
            set_b = set(b)
            added = sorted(set_b - set_a)
            removed = sorted(set_a - set_b)
            changes = []
            if added:
                changes.append("added " + ", ".join(added))
            if removed:
                changes.append("removed " + ", ".join(removed))
            if changes:
                summary = ", and ".join(changes)
                differences.append(f"{format_path(path)}: {summary}")
            return differences
        if a != b:
            differences.append(
                f"{format_path(path)}: {format_basic(a)} -> {format_basic(b)}"
            )
        return differences
    if a != b:
        differences.append(f"{format_path(path)}: {format_basic(a)} -> {format_basic(b)}")
    return differences


def is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def flatten_numeric_leaves(node: Any, path: Tuple[str, ...], output: Dict[str, float]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            flatten_numeric_leaves(value, path + (key,), output)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            flatten_numeric_leaves(value, path + (str(index),), output)
    elif is_numeric(node):
        output[format_path(path)] = float(node)


def numbers_close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def format_percent(change: float) -> str:
    if math.isinf(change):
        return "(+inf%)" if change > 0 else "(-inf%)"
    sign = "+" if change > 0 else "-"
    return f"({sign}{abs(change):.1f}%)"


def compute_metric_changes(
    old_metrics: Mapping[str, float],
    new_metrics: Mapping[str, float],
) -> List[Tuple[float, str]]:
    changes: List[Tuple[float, str]] = []
    all_keys = set(old_metrics) | set(new_metrics)
    for key in all_keys:
        old_value = old_metrics.get(key)
        new_value = new_metrics.get(key)
        if old_value is None:
            description = f"{key}: (missing) -> {format_basic(new_value)}"
            changes.append((math.inf, description))
            continue
        if new_value is None:
            description = f"{key}: {format_basic(old_value)} -> (missing)"
            changes.append((math.inf, description))
            continue
        if numbers_close(old_value, new_value):
            continue
        delta = new_value - old_value
        if numbers_close(old_value, 0.0):
            percent_change = math.inf if delta > 0 else -math.inf
        else:
            percent_change = (delta / old_value) * 100.0
        description = (
            f"{key}: {format_basic(old_value)} -> {format_basic(new_value)} "
            f"{format_percent(percent_change)}"
        )
        sort_key = abs(percent_change) if math.isfinite(percent_change) else math.inf
        changes.append((sort_key, description))
    changes.sort(key=lambda item: (-item[0], item[1]))
    return changes


def main() -> None:
    args = parse_args()
    if args.benchmark_files:
        left_path, right_path = [Path(p) for p in args.benchmark_files]
    else:
        left_path, right_path = find_latest_benchmarks(Path(args.benchmarks_dir))
    left_data = read_json(left_path)
    right_data = read_json(right_path)

    print(f"Comparing:\n  A: {left_path}\n  B: {right_path}\n")

    config_a = left_data.get("benchmark_run_info", {}).get("dungeon_config", {})
    config_b = right_data.get("benchmark_run_info", {}).get("dungeon_config", {})
    config_diffs = compare_configs(config_a, config_b, ("benchmark_run_info", "dungeon_config"))

    print("Changes in benchmark_run_info.dungeon_config:")
    if config_diffs:
        for diff in sorted(config_diffs):
            print(f"  - {diff}")
    else:
        print("  (no changes)")
    print()

    aggregated_a = left_data.get("aggregated_results", {})
    aggregated_b = right_data.get("aggregated_results", {})

    gen_time_a: Dict[str, float] = {}
    gen_time_b: Dict[str, float] = {}
    if "generation_time" in aggregated_a:
        flatten_numeric_leaves(aggregated_a["generation_time"], ("generation_time",), gen_time_a)
    if "generation_time" in aggregated_b:
        flatten_numeric_leaves(aggregated_b["generation_time"], ("generation_time",), gen_time_b)
    gen_time_changes = compute_metric_changes(gen_time_a, gen_time_b)

    print("Changes in aggregated_results.generation_time:")
    if gen_time_changes:
        for _, change in gen_time_changes:
            print(f"  - {change}")
    else:
        print("  (no changes)")
    print()

    other_metrics_a: Dict[str, float] = {}
    other_metrics_b: Dict[str, float] = {}
    for key, value in aggregated_a.items():
        if key in {"generation_time", "worst_case_run"}:
            continue
        flatten_numeric_leaves(value, (key,), other_metrics_a)
    for key, value in aggregated_b.items():
        if key in {"generation_time", "worst_case_run"}:
            continue
        flatten_numeric_leaves(value, (key,), other_metrics_b)
    other_changes = compute_metric_changes(other_metrics_a, other_metrics_b)

    print("Changes in aggregated_results (excluding generation_time and worst_case_run):")
    if other_changes:
        for _, change in other_changes:
            print(f"  - {change}")
    else:
        print("  (no changes)")


if __name__ == "__main__":
    main()
