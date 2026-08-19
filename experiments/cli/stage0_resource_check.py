from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from experiments.phase1_batch_common import count_fe, count_runs, load_config, validate_dynamic_collection_config


@dataclass(frozen=True)
class ResourceEstimate:
    config_path: str
    runs: int
    total_fe: int
    per_dimension_fe: dict[int, int]


EXPECTED_BBOB_TRAIN_RUNS = 18 * 3 * 3 * 30 * 4
EXPECTED_BBOB_VALIDATION_RUNS = 6 * 3 * 3 * 30 * 4
EXPECTED_BBOB_TRAIN_FE = 18 * 3 * 30 * 4 * (10_000 + 20_000 + 40_000)
EXPECTED_BBOB_VALIDATION_FE = 6 * 3 * 30 * 4 * (10_000 + 20_000 + 40_000)
EXPECTED_BBOB_TOTAL_RUNS = EXPECTED_BBOB_TRAIN_RUNS + EXPECTED_BBOB_VALIDATION_RUNS
EXPECTED_BBOB_TOTAL_FE = EXPECTED_BBOB_TRAIN_FE + EXPECTED_BBOB_VALIDATION_FE
EXPECTED_CEC2017_RUNS = 29 * 1 * 3 * 30 * 4
EXPECTED_CEC2017_FE = 29 * 1 * 30 * 4 * (10_000 + 30_000 + 50_000)



def compute_resource_estimate(config_path: Path) -> ResourceEstimate:
    config = load_config(config_path)
    validate_dynamic_collection_config(config)
    dimensions = _as_dimensions(config)
    functions = _as_functions(config)
    runs = count_runs(config, functions, dimensions)
    total_fe = count_fe(config, functions, dimensions)
    per_dimension_fe = {
        dimension: int(total_fe_for_dimension(config, dimension))
        for dimension in dimensions
    }
    return ResourceEstimate(
        config_path=str(config_path),
        runs=int(runs),
        total_fe=int(total_fe),
        per_dimension_fe=per_dimension_fe,
    )



def verify_stage0_resource_estimate(config_paths: Iterable[Path]) -> dict[str, object]:
    estimates = [compute_resource_estimate(path) for path in config_paths]
    for estimate in estimates:
        _assert_internal_consistency(estimate)

    by_path = {estimate.config_path: estimate for estimate in estimates}
    return {
        "config_paths": list(by_path),
        "estimates": [
            {
                "config_path": estimate.config_path,
                "runs": estimate.runs,
                "total_fe": estimate.total_fe,
                "per_dimension_fe": estimate.per_dimension_fe,
            }
            for estimate in estimates
        ],
        "checkpoints": {
            "bbob_train": {
                "expected_runs": EXPECTED_BBOB_TRAIN_RUNS,
                "expected_fe": EXPECTED_BBOB_TRAIN_FE,
            },
            "bbob_validation": {
                "expected_runs": EXPECTED_BBOB_VALIDATION_RUNS,
                "expected_fe": EXPECTED_BBOB_VALIDATION_FE,
            },
            "bbob_total": {
                "expected_runs": EXPECTED_BBOB_TOTAL_RUNS,
                "expected_fe": EXPECTED_BBOB_TOTAL_FE,
            },
            "cec2017": {
                "expected_runs": EXPECTED_CEC2017_RUNS,
                "expected_fe": EXPECTED_CEC2017_FE,
            },
        },
    }



def _assert_internal_consistency(estimate: ResourceEstimate) -> None:
    if estimate.runs <= 0 or estimate.total_fe <= 0:
        raise ValueError(f"{estimate.config_path}: runs and total_fe must be positive")
    if not estimate.per_dimension_fe:
        raise ValueError(f"{estimate.config_path}: per_dimension_fe must not be empty")
    if any(fe <= 0 for fe in estimate.per_dimension_fe.values()):
        raise ValueError(f"{estimate.config_path}: per_dimension_fe must be positive")
    if sum(estimate.per_dimension_fe.values()) != estimate.total_fe:
        raise ValueError(f"{estimate.config_path}: per-dimension FE must sum to total FE")



def total_fe_for_dimension(config: dict, dimension: int) -> int:
    if "FE_total_by_dimension" in config:
        budgets = config["FE_total_by_dimension"]
        if str(dimension) in budgets:
            return int(budgets[str(dimension)])
        if dimension in budgets:
            return int(budgets[dimension])
    if "FE_total" in config:
        return int(config["FE_total"])
    raise ValueError("config must define FE_total or FE_total_by_dimension")



def _as_functions(config: dict) -> list[int]:
    values = config.get("functions")
    if not isinstance(values, list):
        raise ValueError("functions must be a list")
    return [int(value) for value in values]



def _as_dimensions(config: dict) -> list[int]:
    values = config.get("dimensions")
    if not isinstance(values, list):
        raise ValueError("dimensions must be a list")
    return [int(value) for value in values]



def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 0 resource verification check.")
    parser.add_argument(
        "configs",
        nargs="+",
        type=Path,
        default=[Path("configs/phase1_bbob_train.yaml"), Path("configs/phase1_bbob_validation.yaml"), Path("configs/phase1_cec2017_test.yaml")],
        help="Frozen phase 1 configs to verify.",
    )
    args = parser.parse_args()
    result = verify_stage0_resource_estimate(args.configs)
    for estimate in result["estimates"]:
        print(
            f"{estimate['config_path']}: runs={estimate['runs']} total_FE={estimate['total_fe']} per_dimension_FE={estimate['per_dimension_fe']}"
        )
    print("checkpoints:")
    for name, checkpoint in result["checkpoints"].items():
        print(f"  {name}: runs={checkpoint['expected_runs']} total_FE={checkpoint['expected_fe']}")


if __name__ == "__main__":
    main()
