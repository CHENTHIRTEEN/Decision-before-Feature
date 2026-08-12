from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from optimizers.registry import SUPPORTED_ALGORITHMS
from optimizers.settings import OptimizerSettings
from trajectory.sampling import get_sampling_spec


FROZEN_PHASE1_POPULATION_SIZE = 40
FROZEN_PHASE1_FE_PER_DIMENSION = 1000
SUPPORTED_PHASE1_SUITES = ("bbob", "cec2017", "cec2022")


@dataclass(frozen=True)
class Shard:
    suite: str
    function: int
    dimension: int
    output_path: Path

    @property
    def family(self) -> str:
        return family_name(self.suite, self.function)

    @property
    def final_performance_path(self) -> Path:
        return self.output_path.with_name("final_performance.parquet")


ShardOutputPairState = Literal["complete", "missing", "partial"]


def shard_output_pair_state(shard: Shard) -> ShardOutputPairState:
    trajectory_exists = shard.output_path.exists()
    final_performance_exists = shard.final_performance_path.exists()
    if trajectory_exists and final_performance_exists:
        return "complete"
    if not trajectory_exists and not final_performance_exists:
        return "missing"
    return "partial"


def require_complete_shard_outputs(shard: Shard) -> None:
    state = shard_output_pair_state(shard)
    if state == "complete":
        return
    if state == "missing":
        raise FileNotFoundError(
            "missing trajectory and complete-budget final-performance shard pair: "
            f"{shard.output_path.parent}"
        )
    raise FileNotFoundError(
        "incomplete shard output pair; trajectory and complete-budget final performance "
        f"must both exist: {shard.output_path.parent}"
    )


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config


def as_int_list(config: dict, name: str) -> list[int]:
    values = config.get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list")
    integers = [int(value) for value in values]
    if len(integers) != len(set(integers)):
        raise ValueError(f"{name} must not contain duplicate values")
    return integers


def algorithms(config: dict) -> list[str]:
    values = config.get("algorithms")
    if not isinstance(values, list) or not values:
        raise ValueError("algorithms must be a non-empty list")
    names = [str(value).lower() for value in values]
    unsupported = sorted(set(names).difference(SUPPORTED_ALGORITHMS))
    if unsupported:
        raise ValueError(f"unsupported algorithms: {unsupported}")
    if tuple(names) != SUPPORTED_ALGORITHMS:
        raise ValueError(
            "formal algorithm portfolio must be exactly de, pso, cmaes, shade in frozen order"
        )
    return names


def validate_dynamic_collection_config(config: dict) -> None:
    suite = str(config.get("suite", "")).lower()
    if suite not in SUPPORTED_PHASE1_SUITES:
        raise ValueError(
            "phase1 dynamic collection supports suites: bbob, cec2017, cec2022"
        )
    if "checkpoint_ratios" in config:
        raise ValueError(
            "checkpoint_ratios is not part of the frozen dynamic protocol; "
            "use sampling_protocol instead"
        )

    sampling_protocol = get_sampling_spec(
        str(config.get("sampling_protocol", ""))
    ).protocol
    population_size = int(config["population_size"])
    if population_size != FROZEN_PHASE1_POPULATION_SIZE:
        raise ValueError(
            "formal phase1 population_size must be exactly "
            f"{FROZEN_PHASE1_POPULATION_SIZE}"
        )

    algorithms(config)
    as_int_list(config, "functions")
    as_int_list(config, "instances")
    dimensions = as_int_list(config, "dimensions")
    as_int_list(config, "seeds")
    for dimension in dimensions:
        fe_total = fe_total_for_dimension(config, dimension)
        expected_fe_total = FROZEN_PHASE1_FE_PER_DIMENSION * dimension
        if fe_total != expected_fe_total:
            raise ValueError(
                f"formal phase1 FE_total for {dimension}D must be "
                f"{expected_fe_total}, got {fe_total}"
            )
        OptimizerSettings(
            population_size=population_size,
            sampling_protocol=sampling_protocol,
        ).validate(fe_total)


def fe_total_for_dimension(config: dict, dimension: int) -> int:
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


def split_name(config: dict) -> str:
    if "split" in config:
        return str(config["split"])
    output = Path(config["output"])
    stem = output.stem
    suffix = "_trajectories"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def family_name(suite: str, function: int) -> str:
    suite_name = str(suite).lower()
    if suite_name == "bbob":
        return f"bbob_f{int(function):03d}"
    if suite_name in {"cec2017", "cec2022"}:
        return f"{suite_name}_f{int(function):02d}"
    raise ValueError(f"unsupported benchmark suite for shard family: {suite}")


def selected_functions(config: dict, only_functions: list[int] | None = None) -> list[int]:
    functions = as_int_list(config, "functions")
    if only_functions is None:
        return functions
    requested = set(int(value) for value in only_functions)
    missing = sorted(requested.difference(functions))
    if missing:
        raise ValueError(f"requested functions are not in config: {missing}")
    return [function for function in functions if function in requested]


def selected_dimensions(config: dict, only_dimensions: list[int] | None = None) -> list[int]:
    dimensions = as_int_list(config, "dimensions")
    if only_dimensions is None:
        return dimensions
    requested = set(int(value) for value in only_dimensions)
    missing = sorted(requested.difference(dimensions))
    if missing:
        raise ValueError(f"requested dimensions are not in config: {missing}")
    return [dimension for dimension in dimensions if dimension in requested]


def shard_output_path(config: dict, function: int, dimension: int) -> Path:
    base_dir = Path(config["output"]).parent / split_name(config)
    return base_dir / family_name(str(config["suite"]), function) / f"dimension_{dimension}" / "trajectories.parquet"


def make_shards(
    config: dict,
    only_functions: list[int] | None = None,
    only_dimensions: list[int] | None = None,
) -> list[Shard]:
    return [
        Shard(
            suite=str(config["suite"]).lower(),
            function=function,
            dimension=dimension,
            output_path=shard_output_path(config, function, dimension),
        )
        for function in selected_functions(config, only_functions)
        for dimension in selected_dimensions(config, only_dimensions)
    ]


def count_runs(config: dict, functions: list[int], dimensions: list[int]) -> int:
    return (
        len(functions)
        * len(as_int_list(config, "instances"))
        * len(dimensions)
        * len(as_int_list(config, "seeds"))
        * len(algorithms(config))
    )


def count_fe(config: dict, functions: list[int], dimensions: list[int]) -> int:
    per_function = len(as_int_list(config, "instances")) * len(as_int_list(config, "seeds")) * len(algorithms(config))
    return len(functions) * per_function * sum(fe_total_for_dimension(config, dimension) for dimension in dimensions)
