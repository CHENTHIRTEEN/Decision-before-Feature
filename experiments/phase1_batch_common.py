from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from optimizers.registry import SUPPORTED_ALGORITHMS


@dataclass(frozen=True)
class Shard:
    function: int
    dimension: int
    output_path: Path

    @property
    def family(self) -> str:
        return f"bbob_f{self.function:03d}"


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
    return [int(value) for value in values]


def algorithms(config: dict) -> list[str]:
    values = config.get("algorithms")
    if not isinstance(values, list) or not values:
        raise ValueError("algorithms must be a non-empty list")
    names = [str(value).lower() for value in values]
    unsupported = sorted(set(names).difference(SUPPORTED_ALGORITHMS))
    if unsupported:
        raise ValueError(f"unsupported algorithms: {unsupported}")
    return names


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
    return base_dir / f"bbob_f{function:03d}" / f"dimension_{dimension}" / "trajectories.parquet"


def make_shards(
    config: dict,
    only_functions: list[int] | None = None,
    only_dimensions: list[int] | None = None,
) -> list[Shard]:
    return [
        Shard(function=function, dimension=dimension, output_path=shard_output_path(config, function, dimension))
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

