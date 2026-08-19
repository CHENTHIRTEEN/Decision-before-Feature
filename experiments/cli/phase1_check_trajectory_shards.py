from __future__ import annotations

import argparse
from collections import defaultdict
from math import isclose, isfinite, log10
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
    function_id_name,
    landscape_family_name,
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
    expected_run_keys = {
        (algorithm, problem_id, seed)
        for algorithm in expected_algorithms
        for problem_id in expected_problem_ids
        for seed in expected_seeds
    }
    if not final_performance_path.exists():
        raise ValueError(f"missing attempted-run final-performance shard: {final_performance_path}")
    attempted_rows = pq.read_table(final_performance_path).to_pylist()
    attempted_by_key = {
        (str(row["algorithm"]), str(row["problem_id"]), int(row["seed"])): row
        for row in attempted_rows
    }
    if set(attempted_by_key) != expected_run_keys:
        raise ValueError("attempted-run final-performance coverage differs from the config")
    completed_run_keys = {
        key for key, row in attempted_by_key.items() if bool(row["path_completed"])
    }
    if rows:
        validate_trajectory_file(path)
    minimum_rows = len(completed_run_keys) * sampling_spec.min_samples_per_run
    maximum_rows = expected_runs * sampling_spec.max_samples_per_run
    function_id = function_id_name(suite, function)
    family = landscape_family_name(suite, function)
    population_size = int(config["population_size"])

    if not minimum_rows <= len(rows) <= maximum_rows:
        raise ValueError(
            f"{path}: expected {minimum_rows}..{maximum_rows} dynamic samples, got {len(rows)}"
        )

    families = {str(row["family"]) for row in rows}
    function_ids = {str(row["function_id"]) for row in rows}
    dimensions = {int(row["dimension"]) for row in rows}
    shard_algorithms = {str(row["algorithm"]) for row in rows}
    seeds = {int(row["seed"]) for row in rows}
    problem_ids = {str(row["problem_id"]) for row in rows}
    if not families.issubset({family}):
        raise ValueError(f"{path}: family coverage mismatch: {sorted(families)}")
    if not function_ids.issubset({function_id}):
        raise ValueError(f"{path}: function_id coverage mismatch: {sorted(function_ids)}")
    if not dimensions.issubset({dimension}):
        raise ValueError(f"{path}: dimension coverage mismatch: {sorted(dimensions)}")
    if not shard_algorithms.issubset(set(expected_algorithms)):
        raise ValueError(f"{path}: algorithm coverage mismatch: {sorted(shard_algorithms)}")
    if not seeds.issubset(set(expected_seeds)):
        raise ValueError(f"{path}: seed coverage mismatch: {sorted(seeds)}")
    if not problem_ids.issubset(expected_problem_ids):
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

    if not set(grouped).issubset(expected_run_keys):
        raise ValueError(f"{path}: trajectory contains an unplanned run")
    if not completed_run_keys.issubset(grouped):
        raise ValueError(f"{path}: completed runs are missing trajectory states")

    for key, group in grouped.items():
        ordered = sorted(group, key=lambda item: int(item["FE"]))
        lower = sampling_spec.min_samples_per_run if key in completed_run_keys else 0
        if not lower <= len(ordered) <= sampling_spec.max_samples_per_run:
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
        function_id=function_id,
        family=family,
        dimension=dimension,
        fe_total=fe_total,
        expected_algorithms=set(expected_algorithms),
        expected_seeds=set(expected_seeds),
        expected_run_keys=expected_run_keys,
        log10_gap_floor=float(config["log10_gap_floor"]),
        log10_gap_cap=float(config["log10_gap_cap"]),
        success_gap_target=float(config["success_gap_target"]),
    )

    return {
        "rows": len(rows),
        "runs": expected_runs,
        "trajectory_runs": len(grouped),
        "failed_runs": expected_runs - len(completed_run_keys),
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
    function_id: str,
    family: str,
    dimension: int,
    fe_total: int,
    expected_algorithms: set[str],
    expected_seeds: set[int],
    expected_run_keys: set[tuple[str, str, int]],
    log10_gap_floor: float,
    log10_gap_cap: float,
    success_gap_target: float,
) -> int:
    if not path.exists():
        raise ValueError(f"missing complete-budget final-performance shard: {path}")
    rows = pq.read_table(path).to_pylist()
    if len(rows) != len(expected_run_keys):
        raise ValueError(
            f"{path}: expected {len(expected_run_keys)} attempted-run rows, got {len(rows)}"
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
    if {str(row["function_id"]) for row in rows} != {function_id}:
        raise ValueError(f"{path}: final-performance function_id mismatch")
    if {int(row["dimension"]) for row in rows} != {dimension}:
        raise ValueError(f"{path}: final-performance dimension mismatch")
    if {str(row["algorithm"]) for row in rows} != expected_algorithms:
        raise ValueError(f"{path}: final-performance algorithm coverage mismatch")
    if {int(row["seed"]) for row in rows} != expected_seeds:
        raise ValueError(f"{path}: final-performance seed coverage mismatch")
    for row in rows:
        if int(row["FE_total"]) != fe_total or int(row["FE"]) != fe_total:
            raise ValueError(f"{path}: final performance must be recorded exactly at FE_total")
        status = str(row["run_status"])
        completed = bool(row["path_completed"])
        if status not in {"completed", "failed"} or completed != (status == "completed"):
            raise ValueError(f"{path}: run_status and path_completed are inconsistent")
        if int(row["planned_FE"]) != fe_total:
            raise ValueError(f"{path}: planned_FE must equal FE_total")
        effective_fe = int(row["effective_FE"])
        if not 0 <= effective_fe <= fe_total or (completed and effective_fe != fe_total):
            raise ValueError(f"{path}: effective_FE is inconsistent with run status")
        if completed and not isfinite(float(row["best_fitness"])):
            raise ValueError(f"{path}: completed final best_fitness must be finite")
        endpoint_values = (
            float(row["final_gap"]),
            float(row["log10_gap"]),
            float(row["log10_gap_floor"]),
            float(row["log10_gap_cap"]),
            float(row["success_gap_target"]),
        )
        if not all(isfinite(value) for value in endpoint_values):
            raise ValueError(f"{path}: final endpoint fields must be finite")
        if not isclose(float(row["log10_gap_floor"]), log10_gap_floor, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"{path}: log10_gap_floor differs from suite config")
        if not isclose(float(row["log10_gap_cap"]), log10_gap_cap, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"{path}: log10_gap_cap differs from suite config")
        if not isclose(float(row["success_gap_target"]), success_gap_target, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"{path}: success_gap_target differs from suite config")
        if completed:
            if not isfinite(float(row["benchmark_reference_value"])):
                raise ValueError(f"{path}: completed run requires a finite reference value")
            expected_gap = max(
                float(row["best_fitness"])
                - float(row["benchmark_reference_value"]),
                0.0,
            )
        else:
            expected_gap = log10_gap_cap
        if not isclose(float(row["final_gap"]), expected_gap, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{path}: final_gap is inconsistent with best_fitness and reference")
        expected_log_gap = log10(min(max(expected_gap, log10_gap_floor), log10_gap_cap))
        if not isclose(float(row["log10_gap"]), expected_log_gap, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{path}: log10_gap does not use the configured floor/cap")
        first_hit = row["first_hit_FE"]
        success = bool(row["success"])
        if success != (first_hit is not None):
            raise ValueError(f"{path}: success and first_hit_FE are inconsistent")
        if completed and success != (expected_gap <= success_gap_target):
            raise ValueError(f"{path}: completed endpoint success is inconsistent")
        if bool(row["target_hit_observed"]) != success:
            raise ValueError(f"{path}: target_hit_observed is inconsistent")
        if bool(row["target_hit_before_failure"]) != (success and not completed):
            raise ValueError(f"{path}: target_hit_before_failure is inconsistent")
        if bool(row["endpoint_success"]) != (success and completed):
            raise ValueError(f"{path}: endpoint_success is inconsistent")
        if first_hit is not None and not 1 <= int(first_hit) <= fe_total:
            raise ValueError(f"{path}: first_hit_FE must lie in [1, FE_total]")
        if str(row["optimizer_state_mode"]) != OPTIMIZER_STATE_MODE:
            raise ValueError(f"{path}: final performance optimizer-state mode mismatch")
        if str(row["final_performance_protocol"]) != FINAL_PERFORMANCE_PROTOCOL:
            raise ValueError(f"{path}: final-performance protocol mismatch")
        key = (str(row["algorithm"]), str(row["problem_id"]), int(row["seed"]))
        last_trajectory = last_trajectory_rows.get(key)
        if completed and last_trajectory is None:
            raise ValueError(f"{path}: completed run lacks a trajectory state for {key}")
        if last_trajectory is not None:
            if int(row["native_updates"]) < int(last_trajectory["native_updates"]):
                raise ValueError(f"{path}: final native_updates precede the last decision state for {key}")
            if completed and float(row["best_fitness"]) > float(last_trajectory["best_fitness"]):
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
