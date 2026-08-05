from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import yaml

from benchmarks import make_problem
from optimizers import OptimizerSettings, run_optimizer
from optimizers.registry import SUPPORTED_ALGORITHMS
from trajectory import write_parquet


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _as_int_list(config: dict, name: str) -> list[int]:
    values = config.get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list")
    return [int(value) for value in values]


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


def _run_one(config: dict, algorithm: str, function: int, instance: int, dimension: int, seed: int) -> list:
    problem_config = {
        "suite": config["suite"],
        "function": function,
        "instance": instance,
        "dimension": dimension,
    }
    settings = OptimizerSettings(
        population_size=int(config["population_size"]),
        checkpoint_ratios=tuple(float(ratio) for ratio in config["checkpoint_ratios"]),
    )
    problem = make_problem(problem_config)
    try:
        return run_optimizer(
            algorithm=algorithm,
            problem=problem,
            seed=seed,
            fe_total=_fe_total_for_dimension(config, dimension),
            settings=settings,
        )
    finally:
        problem.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Phase 1 BBOB trajectories in batch.")
    parser.add_argument("--config", default="configs/phase1_bbob_mve.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    if str(config["suite"]).lower() != "bbob":
        raise ValueError("phase1-collect-batch currently supports only suite: bbob")

    algorithms = [str(name).lower() for name in config["algorithms"]]
    unsupported = sorted(set(algorithms).difference(SUPPORTED_ALGORITHMS))
    if unsupported:
        raise ValueError(f"unsupported algorithms: {unsupported}")

    functions = _as_int_list(config, "functions")
    instances = _as_int_list(config, "instances")
    dimensions = _as_int_list(config, "dimensions")
    seeds = _as_int_list(config, "seeds")

    records = []
    for function, instance, dimension, seed, algorithm in product(
        functions, instances, dimensions, seeds, algorithms
    ):
        records.extend(_run_one(config, algorithm, function, instance, dimension, seed))

    output_path = Path(args.output or config["output"])
    written = write_parquet(records, output_path)
    print(f"wrote {len(records)} trajectory records to {written}")


if __name__ == "__main__":
    main()
