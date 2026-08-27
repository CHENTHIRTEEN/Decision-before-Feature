from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.mabbob import BBOB_VALIDATION_FUNCTIONS
from experiments.phase1_batch_common import (
    REQUIRED_ENDPOINT_FIELDS,
    algorithms,
    as_int_list,
    count_runs,
    fe_total_for_dimension,
    load_suite_configs,
    validate_dynamic_collection_config,
)
from trajectory.sampling import get_sampling_spec


BBOB_FUNCTIONS = set(range(1, 25))
MABBOB_FORMAL_SPLIT = "mabbob_formal"
MABBOB_VALIDATION_SPLIT = "mabbob_validation"
MABBOB_FORMAL_BOUNDARY_HANDLING = "reflect"
MABBOB_FORMAL_EFFICACY_REPETITIONS = 3
MABBOB_FORMAL_EFFICACY_AGGREGATION = "median"
MABBOB_SEEDS = (1, 2, 3, 4, 5)
MABBOB_DIMENSIONS = (10, 20, 40)


def validate_batch_config(path: Path) -> list[dict]:
    summaries = []
    for config in load_suite_configs(path):
        summaries.append(_validate_single_suite_config(path, config))
    return summaries


def _validate_single_suite_config(path: Path, config: dict) -> dict:
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
        boundary = str(config.get("boundary_handling", ""))
        if boundary not in {"clip", "reflect"}:
            raise ValueError(
                f"{path}: BBOB configs must declare boundary_handling explicitly (clip or reflect)"
            )
    else:
        validate_mabbob_formal_config(path, config, functions, dimensions)

    return {
        "path": path,
        "suite": suite,
        "config": config,
        "functions": tuple(functions),
        "instances": tuple(instances),
        "dimensions": tuple(dimensions),
        "seeds": tuple(seeds),
        "algorithms": tuple(algorithm_names),
        "population_size": population_size,
        "sampling_protocol": sampling_protocol,
        "function_family_protocol": str(config["function_family_protocol"]),
        "boundary_handling": str(config.get("boundary_handling", "")),
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


def _weight_support_within(entry: dict, allowed: set[int]) -> bool:
    """The leakage guard must hold on the actual weight support, not only the
    declared components list."""
    weights = entry.get("weights")
    if not isinstance(weights, (list, tuple)) or len(weights) != 24:
        return False
    support = {index + 1 for index, value in enumerate(weights) if float(value) > 0.0}
    return support.issubset(allowed)


def validate_mabbob_formal_config(
    path: Path,
    config: dict,
    functions: list[int],
    dimensions: list[int],
) -> None:
    split = str(config.get("split", ""))
    if split not in {MABBOB_FORMAL_SPLIT, MABBOB_VALIDATION_SPLIT}:
        raise ValueError(
            f"{path}: MA-BBOB config must use split={MABBOB_FORMAL_SPLIT} or "
            f"split={MABBOB_VALIDATION_SPLIT}"
        )
    seeds = as_int_list(config, "seeds")
    if tuple(seeds) != MABBOB_SEEDS:
        raise ValueError(f"{path}: MA-BBOB {split} seeds must be {list(MABBOB_SEEDS)}")
    if tuple(dimensions) != MABBOB_DIMENSIONS:
        raise ValueError(
            f"{path}: MA-BBOB {split} dimensions must be {list(MABBOB_DIMENSIONS)}"
        )
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

    if split == MABBOB_VALIDATION_SPLIT:
        allowed = set(BBOB_VALIDATION_FUNCTIONS)
        violations = [
            candidate_id
            for candidate_id in selected_ids
            if not bool(manifest_by_id[candidate_id].get("is_val_component"))
            or not bool(selection_by_id[candidate_id].get("is_val_component"))
            or not _weight_support_within(manifest_by_id[candidate_id], allowed)
            or not _weight_support_within(selection_by_id[candidate_id], allowed)
        ]
        if violations:
            raise ValueError(
                f"{path}: MA-BBOB validation candidates must keep every weight on the "
                f"validation functions {sorted(allowed)}; violations: {violations}"
            )
    else:
        train_scope = set(range(1, 25)).difference(BBOB_VALIDATION_FUNCTIONS)
        leaked = [
            candidate_id
            for candidate_id in selected_ids
            if bool(manifest_by_id[candidate_id].get("is_val_component"))
            or bool(selection_by_id[candidate_id].get("is_val_component"))
            or not _weight_support_within(manifest_by_id[candidate_id], train_scope)
            or not _weight_support_within(selection_by_id[candidate_id], train_scope)
        ]
        if leaked:
            raise ValueError(f"{path}: selected MA-BBOB candidates contain validation leakage: {leaked}")

    manifest_dimension = int(manifest.get("dimension", 10))
    if manifest_dimension not in [int(value) for value in dimensions]:
        raise ValueError(
            f"{path}: manifest reference dimension {manifest_dimension} must be one of the "
            f"configured dimensions {dimensions}"
        )
    dimension = manifest_dimension
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

    _check_mabbob_xopt_consistency(manifest_by_id, selected_ids, dimension, path=path)


def _check_mabbob_xopt_consistency(
    manifest_by_id: dict[int, dict],
    selected_ids: list[int],
    reference_dimension: int,
    *,
    path: Path,
) -> None:
    """The manifest xopt vector and the benchmark-side regeneration must be the
    same operator at the manifest reference dimension."""
    from benchmarks.mabbob import _xopt_from_mode

    for candidate_id in selected_ids:
        entry = manifest_by_id[candidate_id]
        stored = entry.get("xopt")
        if stored is None:
            continue
        mode = str(entry.get("xopt_mode", "uniform"))
        seed = int(entry.get("xopt_seed", 0))
        regenerated = _xopt_from_mode(reference_dimension, seed, mode)
        if not np.allclose(np.asarray(stored, dtype=float), regenerated, rtol=0.0, atol=0.0):
            raise ValueError(
                f"{path}: manifest xopt for candidate {candidate_id} does not match the "
                f"benchmark-side regeneration from xopt_mode/xopt_seed"
            )


def validate_config_pair(left: dict, right: dict) -> None:
    if left["suite"] != right["suite"]:
        raise ValueError("pair checking requires the same suite on both sides")
    left_functions = set(left["functions"])
    right_functions = set(right["functions"])
    overlap = sorted(left_functions.intersection(right_functions))
    if overlap:
        raise ValueError(f"train/validation functions overlap: {overlap}")

    if left["suite"] == "bbob":
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
        "boundary_handling",
        "fe_total_by_dimension",
        "endpoint_config",
    ):
        if left[field] != right[field]:
            raise ValueError(f"train/validation {field} must match")

    if left["boundary_handling"] != "reflect" or right["boundary_handling"] != "reflect":
        raise ValueError(
            "train/validation configs must use boundary_handling=reflect "
            "(clip is reserved for explicitly declared sensitivity analyses)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Phase 1 BBOB or MA-BBOB batch configuration.")
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()

    summaries_by_path = [validate_batch_config(Path(path)) for path in args.configs]
    if len(args.configs) == 2:
        left_by_suite = {summary["suite"]: summary for summary in summaries_by_path[0]}
        right_by_suite = {summary["suite"]: summary for summary in summaries_by_path[1]}
        if set(left_by_suite) != set(right_by_suite):
            raise ValueError(
                "paired configs must cover the same suites; "
                f"left={sorted(left_by_suite)}, right={sorted(right_by_suite)}"
            )
        for suite in sorted(left_by_suite):
            validate_config_pair(left_by_suite[suite], right_by_suite[suite])
    elif len(args.configs) > 2:
        raise ValueError("provide one config, or exactly two configs for train/validation pair checking")

    for summaries in summaries_by_path:
        for summary in summaries:
            sampling_spec = get_sampling_spec(summary["sampling_protocol"])
            run_count = count_runs(
                summary["config"],
                list(summary["functions"]),
                list(summary["dimensions"]),
            )
            minimum_rows = run_count * sampling_spec.min_samples_per_run
            maximum_rows = run_count * sampling_spec.max_samples_per_run
            print(
                f"{summary['path']} [{summary['suite']}]: {run_count} runs, "
                f"{minimum_rows}..{maximum_rows} trajectory rows, "
                f"sampling_protocol={sampling_spec.protocol}"
            )


if __name__ == "__main__":
    main()
