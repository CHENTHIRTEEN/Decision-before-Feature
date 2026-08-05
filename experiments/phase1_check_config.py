from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from optimizers.registry import SUPPORTED_ALGORITHMS


BBOB_FUNCTIONS = set(range(1, 25))


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config


def _as_list(config: dict, name: str) -> list:
    values = config.get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list")
    return values


def _as_int_list(config: dict, name: str) -> list[int]:
    return [int(value) for value in _as_list(config, name)]


def _fe_total_for_dimension(config: dict, dimension: int) -> int:
    if "FE_total_by_dimension" in config:
        budgets = config["FE_total_by_dimension"]
        if not isinstance(budgets, dict):
            raise ValueError("FE_total_by_dimension must be a mapping")
        if dimension in budgets:
            return int(budgets[dimension])
        key = str(dimension)
        if key in budgets:
            return int(budgets[key])
        raise ValueError(f"missing FE_total_by_dimension budget for dimension {dimension}")
    if "FE_total" in config:
        return int(config["FE_total"])
    raise ValueError("config must define FE_total or FE_total_by_dimension")


def _validate_checkpoint_ratios(config: dict) -> tuple[float, ...]:
    ratios = tuple(float(value) for value in _as_list(config, "checkpoint_ratios"))
    previous = 0.0
    for ratio in ratios:
        if ratio <= previous or ratio > 1.0:
            raise ValueError("checkpoint_ratios must be strictly increasing and <= 1.0")
        previous = ratio
    return ratios


def validate_batch_config(path: Path) -> dict:
    config = _load_config(path)
    if str(config.get("suite", "")).lower() != "bbob":
        raise ValueError(f"{path}: suite must be bbob")

    functions = _as_int_list(config, "functions")
    instances = _as_int_list(config, "instances")
    dimensions = _as_int_list(config, "dimensions")
    seeds = _as_int_list(config, "seeds")
    algorithms = [str(value).lower() for value in _as_list(config, "algorithms")]
    unsupported = sorted(set(algorithms).difference(SUPPORTED_ALGORITHMS))
    if unsupported:
        raise ValueError(f"{path}: unsupported algorithms: {unsupported}")

    ratios = _validate_checkpoint_ratios(config)
    population_size = int(config["population_size"])
    if population_size < 4:
        raise ValueError(f"{path}: population_size must be at least 4")

    for dimension in dimensions:
        fe_total = _fe_total_for_dimension(config, dimension)
        if fe_total < population_size:
            raise ValueError(f"{path}: FE budget for {dimension}D must be at least population_size")
        if fe_total % population_size != 0:
            raise ValueError(f"{path}: FE budget for {dimension}D must be divisible by population_size")

    missing_functions = sorted(set(functions).difference(BBOB_FUNCTIONS))
    if missing_functions:
        raise ValueError(f"{path}: BBOB functions must be in 1..24, got {missing_functions}")

    return {
        "path": path,
        "functions": tuple(functions),
        "instances": tuple(instances),
        "dimensions": tuple(dimensions),
        "seeds": tuple(seeds),
        "algorithms": tuple(algorithms),
        "checkpoint_ratios": ratios,
        "fe_total_by_dimension": tuple((dimension, _fe_total_for_dimension(config, dimension)) for dimension in dimensions),
    }


def validate_config_pair(left: dict, right: dict) -> None:
    left_functions = set(left["functions"])
    right_functions = set(right["functions"])
    overlap = sorted(left_functions.intersection(right_functions))
    if overlap:
        raise ValueError(f"train/validation functions overlap: {overlap}")
    combined = left_functions.union(right_functions)
    if combined != BBOB_FUNCTIONS:
        missing = sorted(BBOB_FUNCTIONS.difference(combined))
        extra = sorted(combined.difference(BBOB_FUNCTIONS))
        raise ValueError(f"train/validation functions must cover 1..24; missing={missing}, extra={extra}")

    for field in ("instances", "dimensions", "seeds", "algorithms", "checkpoint_ratios", "fe_total_by_dimension"):
        if left[field] != right[field]:
            raise ValueError(f"train/validation {field} must match")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Phase 1 BBOB batch configuration.")
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()

    summaries = [validate_batch_config(Path(path)) for path in args.configs]
    if len(summaries) == 2:
        validate_config_pair(summaries[0], summaries[1])
    elif len(summaries) > 2:
        raise ValueError("provide one config, or exactly two configs for train/validation pair checking")

    for summary in summaries:
        run_count = (
            len(summary["functions"])
            * len(summary["instances"])
            * len(summary["dimensions"])
            * len(summary["seeds"])
            * len(summary["algorithms"])
        )
        row_count = run_count * len(summary["checkpoint_ratios"])
        print(f"{summary['path']}: {run_count} runs, {row_count} trajectory rows")


if __name__ == "__main__":
    main()

