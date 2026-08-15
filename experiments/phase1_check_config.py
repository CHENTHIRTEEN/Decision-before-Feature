from __future__ import annotations

import argparse
from pathlib import Path

from experiments.phase1_batch_common import (
    REQUIRED_ENDPOINT_FIELDS,
    algorithms,
    as_int_list,
    count_runs,
    fe_total_for_dimension,
    load_config,
    validate_dynamic_collection_config,
)
from trajectory.sampling import get_sampling_spec


BBOB_FUNCTIONS = set(range(1, 25))


def validate_batch_config(path: Path) -> dict:
    config = load_config(path)
    if str(config.get("suite", "")).lower() != "bbob":
        raise ValueError(f"{path}: suite must be bbob")
    try:
        validate_dynamic_collection_config(config)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error

    functions = as_int_list(config, "functions")
    instances = as_int_list(config, "instances")
    dimensions = as_int_list(config, "dimensions")
    seeds = as_int_list(config, "seeds")
    algorithm_names = algorithms(config)
    sampling_protocol = get_sampling_spec(str(config["sampling_protocol"])).protocol
    population_size = int(config["population_size"])

    missing_functions = sorted(set(functions).difference(BBOB_FUNCTIONS))
    if missing_functions:
        raise ValueError(f"{path}: BBOB functions must be in 1..24, got {missing_functions}")

    return {
        "path": path,
        "functions": tuple(functions),
        "instances": tuple(instances),
        "dimensions": tuple(dimensions),
        "seeds": tuple(seeds),
        "algorithms": tuple(algorithm_names),
        "population_size": population_size,
        "sampling_protocol": sampling_protocol,
        "function_family_protocol": str(config["function_family_protocol"]),
        "fe_total_by_dimension": tuple((dimension, fe_total_for_dimension(config, dimension)) for dimension in dimensions),
        "endpoint_config": tuple((field, config[field]) for field in REQUIRED_ENDPOINT_FIELDS),
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

    for field in (
        "instances",
        "dimensions",
        "seeds",
        "algorithms",
        "population_size",
        "sampling_protocol",
        "function_family_protocol",
        "fe_total_by_dimension",
        "endpoint_config",
    ):
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
        sampling_spec = get_sampling_spec(summary["sampling_protocol"])
        run_count = count_runs(
            load_config(summary["path"]),
            list(summary["functions"]),
            list(summary["dimensions"]),
        )
        minimum_rows = run_count * sampling_spec.min_samples_per_run
        maximum_rows = run_count * sampling_spec.max_samples_per_run
        print(
            f"{summary['path']}: {run_count} runs, "
            f"{minimum_rows}..{maximum_rows} trajectory rows, "
            f"sampling_protocol={sampling_spec.protocol}"
        )


if __name__ == "__main__":
    main()
