from __future__ import annotations

import argparse
import json
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
MABBOB_FORMAL_SPLIT = "mabbob_formal"
MABBOB_FORMAL_BOUNDARY_HANDLING = "reflect"
MABBOB_FORMAL_EFFICACY_REPETITIONS = 3
MABBOB_FORMAL_EFFICACY_AGGREGATION = "median"


def validate_batch_config(path: Path) -> dict:
    config = load_config(path)
    suite = str(config.get("suite", "")).lower()
    if suite not in {"bbob", "mabbob"}:
        raise ValueError(f"{path}: suite must be bbob or mabbob")
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

    if suite == "bbob":
        missing_functions = sorted(set(functions).difference(BBOB_FUNCTIONS))
        if missing_functions:
            raise ValueError(f"{path}: BBOB functions must be in 1..24, got {missing_functions}")
    else:
        validate_mabbob_formal_config(path, config, functions, dimensions)

    return {
        "path": path,
        "suite": suite,
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


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def _config_path(path_value: object, *, config_path: Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ".":
        return config_path.parent.joinpath(path).resolve()
    return Path.cwd().joinpath(path)


def validate_mabbob_formal_config(
    path: Path,
    config: dict,
    functions: list[int],
    dimensions: list[int],
) -> None:
    split = str(config.get("split", ""))
    if split != MABBOB_FORMAL_SPLIT:
        raise ValueError(f"{path}: MA-BBOB formal config must use split={MABBOB_FORMAL_SPLIT}")
    if str(config.get("boundary_handling", "")) != MABBOB_FORMAL_BOUNDARY_HANDLING:
        raise ValueError(f"{path}: MA-BBOB formal boundary_handling must be reflect")
    if int(config.get("efficacy_repetitions", 0)) != MABBOB_FORMAL_EFFICACY_REPETITIONS:
        raise ValueError(f"{path}: MA-BBOB formal efficacy_repetitions must be 3")
    if str(config.get("efficacy_aggregation", "")) != MABBOB_FORMAL_EFFICACY_AGGREGATION:
        raise ValueError(f"{path}: MA-BBOB formal efficacy_aggregation must be median")

    if "manifest_path" not in config:
        raise ValueError(f"{path}: MA-BBOB formal config must define manifest_path")
    if "selection_manifest_path" not in config:
        raise ValueError(f"{path}: MA-BBOB formal config must define selection_manifest_path")
    manifest_path = _config_path(config["manifest_path"], config_path=path)
    selection_manifest_path = _config_path(config["selection_manifest_path"], config_path=path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"{path}: manifest_path does not exist: {manifest_path}")
    if not selection_manifest_path.exists():
        raise FileNotFoundError(
            f"{path}: selection_manifest_path does not exist: {selection_manifest_path}"
        )

    manifest = _read_json(manifest_path)
    selection = _read_json(selection_manifest_path)
    manifest_entries = manifest.get("selected")
    selection_entries = selection.get("selected")
    if not isinstance(manifest_entries, list) or not manifest_entries:
        raise ValueError(f"{manifest_path}: selected entries must be a non-empty list")
    if not isinstance(selection_entries, list) or not selection_entries:
        raise ValueError(f"{selection_manifest_path}: selected entries must be a non-empty list")

    selected_ids = [int(value) for value in selection.get("selected_candidate_ids", [])]
    if not selected_ids:
        raise ValueError(f"{selection_manifest_path}: selected_candidate_ids must be non-empty")
    if functions != selected_ids:
        raise ValueError(
            f"{path}: functions must exactly match selected_candidate_ids; "
            f"functions={functions}, selected={selected_ids}"
        )
    if len(selected_ids) != int(selection.get("formal_target", len(selected_ids))):
        raise ValueError(f"{selection_manifest_path}: formal_target does not match selected IDs")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError(f"{selection_manifest_path}: selected_candidate_ids must be unique")

    manifest_by_id = {int(entry["candidate_id"]): entry for entry in manifest_entries}
    selection_by_id = {int(entry["candidate_id"]): entry for entry in selection_entries}
    missing_manifest_ids = sorted(set(selected_ids).difference(manifest_by_id))
    missing_selection_entries = sorted(set(selected_ids).difference(selection_by_id))
    if missing_manifest_ids:
        raise ValueError(f"{manifest_path}: missing selected candidate IDs {missing_manifest_ids}")
    if missing_selection_entries:
        raise ValueError(
            f"{selection_manifest_path}: missing selected entries {missing_selection_entries}"
        )
    leaked = [
        candidate_id
        for candidate_id in selected_ids
        if bool(manifest_by_id[candidate_id].get("is_val_component"))
        or bool(selection_by_id[candidate_id].get("is_val_component"))
    ]
    if leaked:
        raise ValueError(f"{path}: selected MA-BBOB candidates contain validation leakage: {leaked}")

    if len(dimensions) != 1:
        raise ValueError(f"{path}: MA-BBOB formal config must use one frozen dimension")
    dimension = int(dimensions[0])
    manifest_dimension = int(manifest.get("dimension", dimension))
    selection_dimension = int(selection.get("dimension", dimension))
    if dimension != manifest_dimension or dimension != selection_dimension:
        raise ValueError(
            f"{path}: config dimension must match manifest dimensions; "
            f"config={dimension}, manifest={manifest_dimension}, selection={selection_dimension}"
        )
    entry_dimension_mismatch = [
        candidate_id for candidate_id in selected_ids
        if int(manifest_by_id[candidate_id].get("dimension", dimension)) != dimension
    ]
    if entry_dimension_mismatch:
        raise ValueError(
            f"{path}: selected manifest entries have incompatible dimensions: "
            f"{entry_dimension_mismatch}"
        )


def validate_config_pair(left: dict, right: dict) -> None:
    if left["suite"] != "bbob" or right["suite"] != "bbob":
        raise ValueError("pair checking is only defined for BBOB train/validation configs")
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
    parser = argparse.ArgumentParser(description="Check Phase 1 BBOB or MA-BBOB batch configuration.")
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
