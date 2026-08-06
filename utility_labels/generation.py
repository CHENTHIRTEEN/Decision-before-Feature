from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from behavior.features import BEHAVIOR_FEATURE_COLUMNS
from behavior.validation import validate_behavior_file
from benchmarks import make_problem
from experiments.phase1_batch_common import fe_total_for_dimension, load_config, make_shards
from optimizers import OptimizerSettings
from optimizers.continuation import run_population_continuation
from utility_labels.fields import NEED_ELA_COLUMNS, UTILITY_COLUMNS, UTILITY_LAMBDAS, UTILITY_VALUE_COLUMNS


EPS = 1e-12
MIN_LABEL_RATIO = 0.12
FE_ANALYSIS_RATIO = 0.05


def generate_utility_labels(
    *,
    config_path: Path,
    behavior_root: Path,
    ela_root: Path,
    selection_reference_path: Path,
    output_path: Path,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    max_labels: int | None,
) -> dict[str, int | str]:
    config = load_config(config_path)
    split = Path(config["output"]).stem.removesuffix("_trajectories")
    selection = _read_selection_reference(selection_reference_path)
    ela_features = _read_ela_features(ela_root / split / "features.parquet")
    settings = OptimizerSettings(population_size=int(config["population_size"]), checkpoint_ratios=(1.0,))
    rows = []

    for shard in make_shards(config, only_functions, only_dimensions):
        trajectory_path = behavior_root / split / shard.family / f"dimension_{shard.dimension}" / "trajectories.parquet"
        behavior_path = trajectory_path.with_name("behavior.parquet")
        if not trajectory_path.exists():
            raise FileNotFoundError(f"missing trajectory shard: {trajectory_path}")
        if not behavior_path.exists():
            raise FileNotFoundError(f"missing behavior shard: {behavior_path}")
        validate_behavior_file(trajectory_path, behavior_path)
        trajectory_rows = pq.read_table(trajectory_path).to_pylist()
        behavior_rows = pq.read_table(behavior_path).to_pylist()
        for trajectory_row, behavior_row in zip(trajectory_rows, behavior_rows, strict=True):
            if not _eligible_for_label(trajectory_row, config):
                continue
            label = _generate_one_label(
                split=split,
                config=config,
                trajectory_row=trajectory_row,
                behavior_row=behavior_row,
                selection=selection,
                ela_features=ela_features,
                settings=settings,
            )
            rows.append(label)
            if max_labels is not None and len(rows) >= max_labels:
                break
        if max_labels is not None and len(rows) >= max_labels:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=_schema()), output_path)
    print(f"wrote {len(rows)} utility label rows to {output_path}")
    return {"rows": len(rows), "output": str(output_path)}


def _read_selection_reference(path: Path) -> dict[tuple[str, str, int, float], dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing selection reference: {path}")
    rows = pq.read_table(path).to_pylist()
    result = {}
    for row in rows:
        key = (str(row["split"]), str(row["problem_id"]), int(row["dimension"]), round(float(row["remaining_budget_ratio"]), 6))
        result[key] = row
    return result


def _read_ela_features(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing ELA feature file: {path}")
    return {str(row["problem_id"]): row for row in pq.read_table(path).to_pylist()}


def _eligible_for_label(row: dict, config: dict) -> bool:
    dimension = int(row["dimension"])
    fe_total = fe_total_for_dimension(config, dimension)
    fe_prefix = int(row["FE"])
    fe_analysis = int(FE_ANALYSIS_RATIO * fe_total)
    ratio = float(row["FE_ratio"])
    return ratio >= MIN_LABEL_RATIO and ratio < 1.0 and fe_prefix + fe_analysis < fe_total


def _generate_one_label(
    *,
    split: str,
    config: dict,
    trajectory_row: dict,
    behavior_row: dict,
    selection: dict[tuple[str, str, int, float], dict],
    ela_features: dict[str, dict],
    settings: OptimizerSettings,
) -> dict:
    function, instance, dimension = _parse_problem_id(str(trajectory_row["problem_id"]))
    fe_total = fe_total_for_dimension(config, dimension)
    fe_prefix = int(trajectory_row["FE"])
    fe_analysis = int(FE_ANALYSIS_RATIO * fe_total)
    fe_skip = fe_total - fe_prefix
    fe_ela = fe_total - fe_prefix - fe_analysis
    remaining_ratio = round(fe_ela / fe_total, 6)
    selection_row = selection.get((split, str(trajectory_row["problem_id"]), dimension, remaining_ratio))
    if selection_row is None:
        raise ValueError(
            "missing selection reference row for "
            f"split={split}, problem_id={trajectory_row['problem_id']}, remaining_ratio={remaining_ratio}"
        )
    ela_row = ela_features.get(str(trajectory_row["problem_id"]))
    if ela_row is None:
        raise ValueError(f"missing ELA feature row for {trajectory_row['problem_id']}")

    population = np.asarray(trajectory_row["population"], dtype=float)
    fitness = np.asarray(trajectory_row["fitness"], dtype=float)
    best_fitness = float(trajectory_row["best_fitness"])
    problem = make_problem(
        {
            "suite": "bbob",
            "function": function,
            "instance": instance,
            "dimension": dimension,
        }
    )
    try:
        generation = max(1, fe_prefix // int(config["population_size"]))
        skip = run_population_continuation(
            algorithm=str(selection_row["default_algorithm"]),
            problem=problem,
            seed=int(trajectory_row["seed"]),
            function=function,
            instance=instance,
            generation=generation,
            event=1,
            fe_budget=fe_skip,
            population=population,
            fitness=fitness,
            best_fitness=best_fitness,
            settings=settings,
        )
        ela = run_population_continuation(
            algorithm=str(selection_row["selected_algorithm"]),
            problem=problem,
            seed=int(trajectory_row["seed"]),
            function=function,
            instance=instance,
            generation=generation,
            event=2,
            fe_budget=fe_ela,
            population=population,
            fitness=fitness,
            best_fitness=best_fitness,
            settings=settings,
        )
    finally:
        problem.close()

    p_skip = float(skip.best_fitness)
    p_ela = float(ela.best_fitness)
    performance_gain_raw = p_skip - p_ela
    performance_gain_norm = performance_gain_raw / max(abs(p_skip), abs(p_ela), EPS)
    runtime_analysis = float(ela_row["runtime_analysis"])
    runtime_selection = float(selection_row["runtime_selection"])
    time_cost_norm = (runtime_analysis + runtime_selection) / max(float(skip.runtime_seconds), EPS)
    memory_cost_norm = 0.0
    utility_values = {
        column: performance_gain_norm - weight * time_cost_norm
        for column, weight in zip(UTILITY_VALUE_COLUMNS, UTILITY_LAMBDAS, strict=True)
    }
    need_values = {
        column: bool(utility_values[utility_column] > 0.0)
        for column, utility_column in zip(NEED_ELA_COLUMNS, UTILITY_VALUE_COLUMNS, strict=True)
    }

    return {
        "split": split,
        "problem_id": str(trajectory_row["problem_id"]),
        "family": str(trajectory_row["family"]),
        "dimension": dimension,
        "prefix_algorithm": str(trajectory_row["algorithm"]),
        "seed": int(trajectory_row["seed"]),
        "FE": int(trajectory_row["FE"]),
        "FE_ratio": float(trajectory_row["FE_ratio"]),
        "FE_total": int(fe_total),
        "FE_prefix": int(fe_prefix),
        "FE_analysis": int(fe_analysis),
        "FE_skip_optimization": int(fe_skip),
        "FE_ela_optimization": int(fe_ela),
        "default_algorithm": str(selection_row["default_algorithm"]),
        "selected_algorithm": str(selection_row["selected_algorithm"]),
        "p_skip": p_skip,
        "p_ela": p_ela,
        "performance_gain_raw": float(performance_gain_raw),
        "performance_gain_norm": float(performance_gain_norm),
        "runtime_analysis": runtime_analysis,
        "runtime_selection": runtime_selection,
        "runtime_skip_optimization": float(skip.runtime_seconds),
        "runtime_ela_optimization": float(ela.runtime_seconds),
        "time_cost_norm": float(time_cost_norm),
        "memory_cost_norm": memory_cost_norm,
        **utility_values,
        **need_values,
        **{column: behavior_row[column] for column in BEHAVIOR_FEATURE_COLUMNS},
    }


def _parse_problem_id(problem_id: str) -> tuple[int, int, int]:
    match = re.match(r"^bbob_f(\d{3})_i(\d+)_d(\d+)$", problem_id)
    if match is None:
        raise ValueError(f"invalid BBOB problem_id: {problem_id}")
    return tuple(int(value) for value in match.groups())


def _schema() -> pa.Schema:
    fields = [
        ("split", pa.string()),
        ("problem_id", pa.string()),
        ("family", pa.string()),
        ("dimension", pa.int32()),
        ("prefix_algorithm", pa.string()),
        ("seed", pa.int64()),
        ("FE", pa.int64()),
        ("FE_ratio", pa.float64()),
        ("FE_total", pa.int64()),
        ("FE_prefix", pa.int64()),
        ("FE_analysis", pa.int64()),
        ("FE_skip_optimization", pa.int64()),
        ("FE_ela_optimization", pa.int64()),
        ("default_algorithm", pa.string()),
        ("selected_algorithm", pa.string()),
        ("p_skip", pa.float64()),
        ("p_ela", pa.float64()),
        ("performance_gain_raw", pa.float64()),
        ("performance_gain_norm", pa.float64()),
        ("runtime_analysis", pa.float64()),
        ("runtime_selection", pa.float64()),
        ("runtime_skip_optimization", pa.float64()),
        ("runtime_ela_optimization", pa.float64()),
        ("time_cost_norm", pa.float64()),
        ("memory_cost_norm", pa.float64()),
    ]
    fields.extend((column, pa.float64()) for column in UTILITY_VALUE_COLUMNS)
    fields.extend((column, pa.bool_()) for column in NEED_ELA_COLUMNS)
    fields.extend((column, pa.float64()) for column in BEHAVIOR_FEATURE_COLUMNS)
    return pa.schema(fields)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline utility labels from dense Phase 1 checkpoints.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--behavior-root", type=Path, default=Path("results/phase1"))
    parser.add_argument("--ela-root", type=Path, default=Path("results/ela"))
    parser.add_argument(
        "--selection-reference",
        type=Path,
        default=Path("results/selection_reference/bbob_train/selection_reference.parquet"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    parser.add_argument("--max-labels", type=int, default=None)
    args = parser.parse_args()
    generate_utility_labels(
        config_path=args.config,
        behavior_root=args.behavior_root,
        ela_root=args.ela_root,
        selection_reference_path=args.selection_reference,
        output_path=args.output,
        only_functions=args.only_function,
        only_dimensions=args.only_dimension,
        max_labels=args.max_labels,
    )


if __name__ == "__main__":
    main()
