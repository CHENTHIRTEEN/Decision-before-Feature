from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
from math import ceil
from math import isfinite
from pathlib import Path

import pyarrow.parquet as pq

from experiments.phase1_batch_common import (
    algorithms,
    as_int_list,
    family_name,
    fe_total_for_dimension,
    load_config,
    make_shards,
)


def _checkpoint_fes(config: dict, dimension: int) -> tuple[int, ...]:
    fe_total = fe_total_for_dimension(config, dimension)
    population_size = int(config["population_size"])
    fes = []
    for ratio in config["checkpoint_ratios"]:
        value = Decimal(str(ratio)) * Decimal(fe_total)
        fe = int(min(fe_total, ceil(float(value) / population_size) * population_size))
        fes.append(fe)
    return tuple(fes)


def _problem_id(suite: str, function: int, instance: int, dimension: int) -> str:
    suite_name = str(suite).lower()
    if suite_name == "bbob":
        return f"bbob_f{function:03d}_i{instance:02d}_d{dimension}"
    if suite_name in {"cec2017", "cec2022"}:
        return f"{suite_name}_f{function:02d}_d{dimension}"
    raise ValueError(f"unsupported benchmark suite for problem_id check: {suite}")


def _check_shard(config: dict, path: Path, function: int, dimension: int) -> dict:
    if not path.exists():
        raise ValueError(f"missing shard: {path}")

    table = pq.read_table(path)
    rows = table.to_pylist()
    expected_algorithms = tuple(algorithms(config))
    expected_instances = tuple(as_int_list(config, "instances"))
    expected_seeds = tuple(as_int_list(config, "seeds"))
    expected_fes = _checkpoint_fes(config, dimension)
    fe_total = fe_total_for_dimension(config, dimension)
    expected_ratios = tuple(float(ratio) for ratio in config["checkpoint_ratios"])
    expected_ratio_by_fe = dict(zip(expected_fes, expected_ratios, strict=True))
    suite = str(config["suite"]).lower()
    expected_problem_ids = {_problem_id(suite, function, instance, dimension) for instance in expected_instances}
    expected_rows = len(expected_instances) * len(expected_seeds) * len(expected_algorithms) * len(expected_fes)
    family = family_name(suite, function)
    population_size = int(config["population_size"])

    if len(rows) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, got {len(rows)}")

    families = {str(row["family"]) for row in rows}
    dimensions = {int(row["dimension"]) for row in rows}
    shard_algorithms = {str(row["algorithm"]) for row in rows}
    seeds = {int(row["seed"]) for row in rows}
    problem_ids = {str(row["problem_id"]) for row in rows}
    fes = {int(row["FE"]) for row in rows}
    ratios = {float(row["FE_ratio"]) for row in rows}

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
    if fes != set(expected_fes):
        raise ValueError(f"{path}: checkpoint FE coverage mismatch: {sorted(fes)}")
    if ratios != set(expected_ratios):
        raise ValueError(f"{path}: FE_ratio coverage mismatch: {sorted(ratios)}")

    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        best_fitness = float(row["best_fitness"])
        if not isfinite(best_fitness):
            raise ValueError(f"{path}: best_fitness must be finite")
        fe = int(row["FE"])
        if float(row["FE_ratio"]) != expected_ratio_by_fe.get(fe):
            raise ValueError(f"{path}: FE_ratio does not match the formal checkpoint ratio for FE={fe}")

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

    expected_run_count = len(expected_algorithms) * len(expected_problem_ids) * len(expected_seeds)
    if len(grouped) != expected_run_count:
        raise ValueError(f"{path}: expected {expected_run_count} runs, got {len(grouped)}")

    for key, group in grouped.items():
        ordered = sorted(group, key=lambda item: int(item["FE"]))
        run_fes = tuple(int(item["FE"]) for item in ordered)
        run_ratios = tuple(float(item["FE_ratio"]) for item in ordered)
        if run_fes != expected_fes:
            raise ValueError(f"{path}: checkpoint coverage mismatch for {key}: {run_fes}")
        if run_ratios != expected_ratios:
            raise ValueError(f"{path}: FE_ratio coverage mismatch for {key}: {run_ratios}")
        best_values = [float(item["best_fitness"]) for item in ordered]
        if any(later > earlier for earlier, later in zip(best_values, best_values[1:])):
            raise ValueError(f"{path}: best_fitness must be non-increasing for {key}")

    return {
        "rows": len(rows),
        "runs": len(grouped),
        "family": family,
        "dimension": dimension,
        "algorithms": len(shard_algorithms),
        "seeds": len(seeds),
        "checkpoints": len(fes),
    }


def check_config(
    path: Path,
    only_functions: list[int] | None = None,
    only_dimensions: list[int] | None = None,
) -> dict:
    config = load_config(path)
    if str(config["suite"]).lower() not in {"bbob", "cec2017", "cec2022"}:
        raise ValueError("phase1-check-trajectory-shards supports suites: bbob, cec2017, cec2022")

    shard_summaries = []
    for shard in make_shards(config, only_functions, only_dimensions):
        shard_summaries.append(_check_shard(config, shard.output_path, shard.function, shard.dimension))

    total_rows = sum(int(summary["rows"]) for summary in shard_summaries)
    total_runs = sum(int(summary["runs"]) for summary in shard_summaries)
    families = sorted({str(summary["family"]) for summary in shard_summaries})
    dimensions = sorted({int(summary["dimension"]) for summary in shard_summaries})
    rows_per_shard = sorted({int(summary["rows"]) for summary in shard_summaries})

    return {
        "path": str(path),
        "shards": len(shard_summaries),
        "rows": total_rows,
        "runs": total_runs,
        "rows_per_shard": rows_per_shard,
        "families": families,
        "dimensions": dimensions,
        "algorithms": tuple(algorithms(config)),
        "seeds": tuple(as_int_list(config, "seeds")),
        "checkpoint_ratios": tuple(float(ratio) for ratio in config["checkpoint_ratios"]),
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
        print(f"  rows_per_shard: {summary['rows_per_shard']}")
        print(f"  families: {', '.join(summary['families'])}")
        print(f"  dimensions: {summary['dimensions']}")
        print(f"  algorithms: {', '.join(summary['algorithms'])}")
        print(f"  seed_count: {len(summary['seeds'])}")
        print(f"  checkpoint_ratios: {summary['checkpoint_ratios']}")


if __name__ == "__main__":
    main()
