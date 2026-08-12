from __future__ import annotations

import argparse
from collections import defaultdict
from math import isfinite
from pathlib import Path

import pyarrow.parquet as pq

from trajectory.final_performance import FINAL_PERFORMANCE_PROTOCOL
from trajectory.records import OPTIMIZER_STATE_MODE
from trajectory.sampling import (
    MAX_SAMPLES_PER_RUN,
    MIN_SAMPLES_PER_RUN,
    get_sampling_spec,
)
from trajectory.validation import validate_trajectory_file

from experiments.phase1_batch_common import (
    algorithms,
    as_int_list,
    family_name,
    fe_total_for_dimension,
    load_config,
    make_shards,
    validate_dynamic_collection_config,
)


def _problem_id(suite: str, function: int, instance: int, dimension: int) -> str:
    suite_name = str(suite).lower()
    if suite_name == "bbob":
        return f"bbob_f{function:03d}_i{instance:02d}_d{dimension}"
    if suite_name in {"cec2017", "cec2022"}:
        return f"{suite_name}_f{function:02d}_d{dimension}"
    raise ValueError(f"unsupported benchmark suite for problem_id check: {suite}")


def _check_shard(
    config: dict,
    path: Path,
    final_performance_path: Path,
    function: int,
    dimension: int,
) -> dict:
    if not path.exists():
        raise ValueError(f"missing shard: {path}")

    validate_trajectory_file(path)
    table = pq.read_table(path)
    if "optimizer_state_mode" not in table.column_names:
        raise ValueError(f"{path}: missing optimizer_state_mode; regenerate this pre-native-continuation shard")
    rows = table.to_pylist()
    expected_algorithms = tuple(algorithms(config))
    expected_instances = tuple(as_int_list(config, "instances"))
    expected_seeds = tuple(as_int_list(config, "seeds"))
    fe_total = fe_total_for_dimension(config, dimension)
    sampling_spec = get_sampling_spec(str(config.get("sampling_protocol", "")))
    suite = str(config["suite"]).lower()
    expected_problem_ids = {_problem_id(suite, function, instance, dimension) for instance in expected_instances}
    expected_runs = len(expected_instances) * len(expected_seeds) * len(expected_algorithms)
    minimum_rows = expected_runs * sampling_spec.min_samples_per_run
    maximum_rows = expected_runs * sampling_spec.max_samples_per_run
    family = family_name(suite, function)
    population_size = int(config["population_size"])

    if not minimum_rows <= len(rows) <= maximum_rows:
        raise ValueError(
            f"{path}: expected {minimum_rows}..{maximum_rows} dynamic samples, got {len(rows)}"
        )

    families = {str(row["family"]) for row in rows}
    dimensions = {int(row["dimension"]) for row in rows}
    shard_algorithms = {str(row["algorithm"]) for row in rows}
    seeds = {int(row["seed"]) for row in rows}
    problem_ids = {str(row["problem_id"]) for row in rows}
    if families != {family}:
        raise ValueError(f"{path}: family coverage mismatch: {sorted(families)}")
    if dimensions != {dimension}:
        raise ValueError(f"{path}: dimension coverage mismatch: {sorted(dimensions)}")
    if shard_algorithms != set(expected_algorithms):
        raise ValueError(f"{path}: algorithm coverage mismatch: {sorted(shard_algorithms)}")
    if seeds != set(expected_seeds):
        raise ValueError(f"{path}: seed coverage mismatch: {sorted(seeds)}")
    if problem_ids != expected_problem_ids:
        raise ValueError(f"{path}: problem_id coverage mismatch: {sorted(problem_ids)}")
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        if str(row["optimizer_state_mode"]) != OPTIMIZER_STATE_MODE:
            raise ValueError(f"{path}: trajectory does not use native optimizer-state continuation")
        if str(row["sampling_protocol"]) != sampling_spec.protocol:
            raise ValueError(f"{path}: trajectory does not use the configured sampling protocol")
        if "window_statistics" not in row or "native_update_history" not in row:
            raise ValueError(f"{path}: trajectory is missing strict native-update window statistics")
        best_fitness = float(row["best_fitness"])
        if not isfinite(best_fitness):
            raise ValueError(f"{path}: best_fitness must be finite")
        if int(row["FE_total"]) != fe_total:
            raise ValueError(f"{path}: FE_total does not match the configured dimension budget")

        population = row["population"]
        fitness = row["fitness"]
        if len(population) != population_size or len(fitness) != population_size:
            raise ValueError(f"{path}: population checkpoint must contain {population_size} points and fitness values")
        for vector in population:
            if len(vector) != dimension:
                raise ValueError(f"{path}: population vector width must match dimension {dimension}")
            if any(not isfinite(float(value)) for value in vector):
                raise ValueError(f"{path}: population values must be finite")
        if any(not isfinite(float(value)) for value in fitness):
            raise ValueError(f"{path}: fitness values must be finite")

        grouped[(str(row["algorithm"]), str(row["problem_id"]), int(row["seed"]))].append(row)

    expected_run_count = expected_runs
    if len(grouped) != expected_run_count:
        raise ValueError(f"{path}: expected {expected_run_count} runs, got {len(grouped)}")

    for key, group in grouped.items():
        ordered = sorted(group, key=lambda item: int(item["FE"]))
        if not sampling_spec.min_samples_per_run <= len(ordered) <= sampling_spec.max_samples_per_run:
            raise ValueError(f"{path}: dynamic sample count mismatch for {key}: {len(ordered)}")
        best_values = [float(item["best_fitness"]) for item in ordered]
        if any(later > earlier for earlier, later in zip(best_values, best_values[1:])):
            raise ValueError(f"{path}: best_fitness must be non-increasing for {key}")

    final_rows = _check_final_performance_shard(
        path=final_performance_path,
        last_trajectory_rows={
            key: max(group, key=lambda row: int(row["FE"]))
            for key, group in grouped.items()
        },
        expected_problem_ids=expected_problem_ids,
        family=family,
        dimension=dimension,
        fe_total=fe_total,
        expected_algorithms=set(expected_algorithms),
        expected_seeds=set(expected_seeds),
    )

    return {
        "rows": len(rows),
        "runs": len(grouped),
        "family": family,
        "dimension": dimension,
        "algorithms": len(shard_algorithms),
        "seeds": len(seeds),
        "minimum_samples_per_run": sampling_spec.min_samples_per_run,
        "maximum_samples_per_run": sampling_spec.max_samples_per_run,
        "final_performance_rows": final_rows,
    }


def _check_final_performance_shard(
    *,
    path: Path,
    last_trajectory_rows: dict[tuple[str, str, int], dict],
    expected_problem_ids: set[str],
    family: str,
    dimension: int,
    fe_total: int,
    expected_algorithms: set[str],
    expected_seeds: set[int],
) -> int:
    if not path.exists():
        raise ValueError(f"missing complete-budget final-performance shard: {path}")
    rows = pq.read_table(path).to_pylist()
    expected_run_keys = set(last_trajectory_rows)
    if len(rows) != len(expected_run_keys):
        raise ValueError(
            f"{path}: expected {len(expected_run_keys)} complete-budget rows, got {len(rows)}"
        )
    keys = [
        (str(row["algorithm"]), str(row["problem_id"]), int(row["seed"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)) or set(keys) != expected_run_keys:
        raise ValueError(f"{path}: complete-budget run coverage differs from trajectory runs")
    if {str(row["problem_id"]) for row in rows} != expected_problem_ids:
        raise ValueError(f"{path}: final-performance problem coverage mismatch")
    if {str(row["family"]) for row in rows} != {family}:
        raise ValueError(f"{path}: final-performance family mismatch")
    if {int(row["dimension"]) for row in rows} != {dimension}:
        raise ValueError(f"{path}: final-performance dimension mismatch")
    if {str(row["algorithm"]) for row in rows} != expected_algorithms:
        raise ValueError(f"{path}: final-performance algorithm coverage mismatch")
    if {int(row["seed"]) for row in rows} != expected_seeds:
        raise ValueError(f"{path}: final-performance seed coverage mismatch")
    for row in rows:
        if int(row["FE_total"]) != fe_total or int(row["FE"]) != fe_total:
            raise ValueError(f"{path}: final performance must be recorded exactly at FE_total")
        if not isfinite(float(row["best_fitness"])):
            raise ValueError(f"{path}: final best_fitness must be finite")
        if str(row["optimizer_state_mode"]) != OPTIMIZER_STATE_MODE:
            raise ValueError(f"{path}: final performance optimizer-state mode mismatch")
        if str(row["final_performance_protocol"]) != FINAL_PERFORMANCE_PROTOCOL:
            raise ValueError(f"{path}: final-performance protocol mismatch")
        key = (str(row["algorithm"]), str(row["problem_id"]), int(row["seed"]))
        last_trajectory = last_trajectory_rows[key]
        if int(row["native_updates"]) < int(last_trajectory["native_updates"]):
            raise ValueError(f"{path}: final native_updates precede the last decision state for {key}")
        if float(row["best_fitness"]) > float(last_trajectory["best_fitness"]):
            raise ValueError(
                f"{path}: complete-budget best_fitness is worse than the last decision state for {key}"
            )
    return len(rows)


def check_config(
    path: Path,
    only_functions: list[int] | None = None,
    only_dimensions: list[int] | None = None,
) -> dict:
    config = load_config(path)
    validate_dynamic_collection_config(config)

    shard_summaries = []
    for shard in make_shards(config, only_functions, only_dimensions):
        shard_summaries.append(
            _check_shard(
                config,
                shard.output_path,
                shard.final_performance_path,
                shard.function,
                shard.dimension,
            )
        )

    total_rows = sum(int(summary["rows"]) for summary in shard_summaries)
    total_runs = sum(int(summary["runs"]) for summary in shard_summaries)
    total_final_performance_rows = sum(
        int(summary["final_performance_rows"]) for summary in shard_summaries
    )
    families = sorted({str(summary["family"]) for summary in shard_summaries})
    dimensions = sorted({int(summary["dimension"]) for summary in shard_summaries})
    rows_per_shard = sorted({int(summary["rows"]) for summary in shard_summaries})

    return {
        "path": str(path),
        "shards": len(shard_summaries),
        "rows": total_rows,
        "runs": total_runs,
        "final_performance_rows": total_final_performance_rows,
        "rows_per_shard": rows_per_shard,
        "families": families,
        "dimensions": dimensions,
        "algorithms": tuple(algorithms(config)),
        "seeds": tuple(as_int_list(config, "seeds")),
        "sampling_protocol": str(config["sampling_protocol"]),
        "samples_per_run": (MIN_SAMPLES_PER_RUN, MAX_SAMPLES_PER_RUN),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Phase 1 trajectory shard data quality.")
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    args = parser.parse_args()

    for config_path in args.configs:
        summary = check_config(Path(config_path), args.only_function, args.only_dimension)
        print(f"{summary['path']}:")
        print(f"  shards: {summary['shards']}")
        print(f"  rows: {summary['rows']}")
        print(f"  runs: {summary['runs']}")
        print(f"  final_performance_rows: {summary['final_performance_rows']}")
        print(f"  rows_per_shard: {summary['rows_per_shard']}")
        print(f"  families: {', '.join(summary['families'])}")
        print(f"  dimensions: {summary['dimensions']}")
        print(f"  algorithms: {', '.join(summary['algorithms'])}")
        print(f"  seed_count: {len(summary['seeds'])}")
        print(f"  sampling_protocol: {summary['sampling_protocol']}")
        print(f"  samples_per_run: {summary['samples_per_run'][0]}..{summary['samples_per_run'][1]}")


if __name__ == "__main__":
    main()
