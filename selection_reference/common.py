from __future__ import annotations

from pathlib import Path
from math import isfinite

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from experiments.phase1_batch_common import (
    algorithms,
    as_int_list,
    fe_total_for_dimension,
    function_id_name,
    landscape_family_name,
    load_config,
    make_shards,
    require_complete_shard_outputs,
)
from trajectory.final_performance import FINAL_PERFORMANCE_PROTOCOL
from trajectory.records import OPTIMIZER_STATE_MODE
from trajectory.sampling import SAMPLING_PROTOCOL


def read_performance(
    config: dict,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
) -> pd.DataFrame:
    frames = []
    for shard in make_shards(config, only_functions, only_dimensions):
        require_complete_shard_outputs(shard)
        table = pq.read_table(
            shard.final_performance_path,
            columns=[
                "problem_id",
                "function_id",
                "family",
                "cv_group_id",
                "dimension",
                "algorithm",
                "seed",
                "FE",
                "FE_total",
                "native_updates",
                "best_fitness",
                "benchmark_reference_value",
                "final_gap",
                "log10_gap",
                "log10_gap_floor",
                "log10_gap_cap",
                "success_gap_target",
                "success",
                "first_hit_FE",
                "run_status",
                "path_completed",
                "planned_FE",
                "effective_FE",
                "observed_first_hit_FE",
                "target_hit_observed",
                "target_hit_before_failure",
                "endpoint_success",
                "failure_type",
                "failure_message",
                "optimizer_state_mode",
                "final_performance_protocol",
            ],
        )
        frame = table.to_pandas()
        if set(frame["optimizer_state_mode"].astype(str)) != {OPTIMIZER_STATE_MODE}:
            raise ValueError(
                "final-performance shard does not use native optimizer-state continuation: "
                f"{shard.final_performance_path}"
            )
        if set(frame["final_performance_protocol"].astype(str)) != {FINAL_PERFORMANCE_PROTOCOL}:
            raise ValueError(
                "final-performance shard protocol is inconsistent: "
                f"{shard.final_performance_path}"
            )
        if not (frame["FE"].astype(int) == frame["FE_total"].astype(int)).all():
            raise ValueError(
                "final-performance rows must be recorded at FE_total: "
                f"{shard.final_performance_path}"
            )
        statuses = frame["run_status"].astype(str)
        if not set(statuses).issubset({"completed", "failed"}):
            raise ValueError("final-performance run_status must be completed or failed")
        completed = statuses.eq("completed").to_numpy(dtype=bool)
        if not np.array_equal(frame["path_completed"].to_numpy(dtype=bool), completed):
            raise ValueError("final-performance path_completed and run_status disagree")
        if not np.array_equal(
            frame["planned_FE"].to_numpy(dtype=int),
            frame["FE_total"].to_numpy(dtype=int),
        ):
            raise ValueError("final-performance planned_FE must equal FE_total")
        effective = frame["effective_FE"].to_numpy(dtype=int)
        planned = frame["planned_FE"].to_numpy(dtype=int)
        if bool((effective < 0).any()) or bool((effective > planned).any()):
            raise ValueError("final-performance effective_FE lies outside the planned budget")
        if not np.array_equal(effective[completed], planned[completed]):
            raise ValueError("completed final-performance runs must consume planned FE")
        expected_problem_ids = _expected_problem_ids(
            config=config,
            function=shard.function,
            dimension=shard.dimension,
        )
        if set(frame["problem_id"].astype(str)) != expected_problem_ids:
            raise ValueError(
                "final-performance problem coverage differs from its configured shard: "
                f"{shard.final_performance_path}"
            )
        if set(frame["function_id"].astype(str)) != {
            function_id_name(str(config["suite"]), shard.function)
        }:
            raise ValueError(
                "final-performance function_id differs from its configured shard: "
                f"{shard.final_performance_path}"
            )
        if set(frame["family"].astype(str)) != {
            landscape_family_name(str(config["suite"]), shard.function)
        }:
            raise ValueError(
                "final-performance family differs from its configured shard: "
                f"{shard.final_performance_path}"
            )
        if set(frame["cv_group_id"].astype(str)) != {shard.cv_group_id}:
            raise ValueError(
                "final-performance cv_group_id differs from its configured shard: "
                f"{shard.final_performance_path}"
            )
        if set(frame["dimension"].astype(int)) != {int(shard.dimension)}:
            raise ValueError(
                "final-performance dimension differs from its configured shard: "
                f"{shard.final_performance_path}"
            )
        expected_fe_total = fe_total_for_dimension(config, shard.dimension)
        if set(frame["FE_total"].astype(int)) != {expected_fe_total}:
            raise ValueError(
                "final-performance budget differs from its configured shard: "
                f"{shard.final_performance_path}"
            )
        _check_final_against_trajectory(
            trajectory_path=shard.output_path,
            final_performance=frame,
        )
        frames.append(
            frame.drop(columns=["optimizer_state_mode", "final_performance_protocol"])
        )
    if not frames:
        raise ValueError("no complete-budget final-performance rows available for SBS")
    performance = pd.concat(frames, ignore_index=True)
    expected_algorithms = set(algorithms(config))
    expected_seeds = set(as_int_list(config, "seeds"))
    run_key = ["problem_id", "algorithm", "seed"]
    if performance.duplicated(run_key).any():
        raise ValueError("final-performance shards contain duplicate problem/algorithm/seed rows")
    for problem_id, group in performance.groupby("problem_id", sort=False):
        observed_algorithms = set(group["algorithm"].astype(str))
        observed_seeds = set(group["seed"].astype(int))
        if observed_algorithms != expected_algorithms:
            raise ValueError(
                f"{problem_id}: final-performance algorithms differ from the train config"
            )
        if observed_seeds != expected_seeds:
            raise ValueError(
                f"{problem_id}: final-performance seeds differ from the train config"
            )
        expected_rows = len(expected_algorithms) * len(expected_seeds)
        if len(group) != expected_rows:
            raise ValueError(
                f"{problem_id}: expected {expected_rows} complete-budget run rows, got {len(group)}"
            )
    return performance


def _check_final_against_trajectory(
    *,
    trajectory_path: Path,
    final_performance: pd.DataFrame,
) -> None:
    trajectory = pq.read_table(
        trajectory_path,
        columns=[
            "problem_id",
            "algorithm",
            "seed",
            "FE",
            "FE_total",
            "native_updates",
            "best_fitness",
            "sampling_protocol",
            "optimizer_state_mode",
        ],
    ).to_pandas()
    if trajectory.empty:
        if final_performance["path_completed"].astype(bool).any():
            raise ValueError(
                f"completed final-performance runs require trajectory states: {trajectory_path}"
            )
        return
    if set(trajectory["sampling_protocol"].astype(str)) != {SAMPLING_PROTOCOL}:
        raise ValueError(
            f"trajectory shard does not use the frozen sampling protocol: {trajectory_path}"
        )
    if set(trajectory["optimizer_state_mode"].astype(str)) != {OPTIMIZER_STATE_MODE}:
        raise ValueError(
            f"trajectory shard does not use native optimizer-state continuation: {trajectory_path}"
        )

    run_key = ["problem_id", "algorithm", "seed"]
    latest = (
        trajectory.sort_values([*run_key, "FE"])
        .groupby(run_key, as_index=False, sort=False)
        .tail(1)
    )
    if latest.duplicated(run_key).any() or final_performance.duplicated(run_key).any():
        raise ValueError(
            "trajectory/final-performance pair must contain one final comparison row per run: "
            f"{trajectory_path.parent}"
        )

    comparison = final_performance.merge(
        latest,
        on=run_key,
        how="outer",
        validate="one_to_one",
        suffixes=("_final", "_trajectory"),
        indicator=True,
    )
    if bool(comparison["_merge"].astype(str).eq("right_only").any()):
        raise ValueError(
            "trajectory contains a run without a final-performance attempted row: "
            f"{trajectory_path.parent}"
        )
    final_only = comparison["_merge"].astype(str).eq("left_only")
    if bool(
        final_only
        & comparison["path_completed"].fillna(False).astype(bool)
    ).any():
        raise ValueError("a completed final-performance run is missing trajectory states")
    paired = comparison[comparison["_merge"].astype(str).eq("both")].copy()
    completed_paired = paired[paired["path_completed"].astype(bool)].copy()
    if completed_paired.empty and bool(final_performance["path_completed"].astype(bool).any()):
        raise ValueError("completed final-performance runs have no paired trajectory states")
    if not (
        paired["FE_total_final"].astype(int)
        == paired["FE_total_trajectory"].astype(int)
    ).all():
        raise ValueError(
            "trajectory and final-performance budgets differ: "
            f"{trajectory_path.parent}"
        )
    if not (
        completed_paired["native_updates_final"].astype(int)
        >= completed_paired["native_updates_trajectory"].astype(int)
    ).all():
        raise ValueError(
            "final performance precedes the last trajectory state: "
            f"{trajectory_path.parent}"
        )
    if not (
        completed_paired["best_fitness_final"].astype(float)
        <= completed_paired["best_fitness_trajectory"].astype(float)
    ).all():
        raise ValueError(
            "final performance is worse than the last trajectory state: "
            f"{trajectory_path.parent}"
        )


def _expected_problem_ids(*, config: dict, function: int, dimension: int) -> set[str]:
    suite = str(config["suite"]).lower()
    if suite == "bbob":
        return {
            f"bbob_f{int(function):03d}_i{instance:02d}_d{int(dimension)}"
            for instance in as_int_list(config, "instances")
        }
    if suite in {"cec2017", "cec2022"}:
        return {f"{suite}_f{int(function):02d}_d{int(dimension)}"}
    raise ValueError(f"unsupported benchmark suite for final performance: {suite}")


def single_best_solver(
    performance: pd.DataFrame,
    portfolio_order: tuple[str, ...] | None = None,
) -> str:
    required = {
        "problem_id",
        "function_id",
        "family",
        "dimension",
        "algorithm",
        "seed",
        "FE",
        "FE_total",
        "best_fitness",
        "benchmark_reference_value",
        "final_gap",
        "log10_gap",
        "log10_gap_floor",
        "log10_gap_cap",
        "run_status",
        "path_completed",
        "planned_FE",
        "effective_FE",
    }
    missing = required.difference(performance.columns)
    if missing:
        raise ValueError(f"final-performance data missing required columns: {sorted(missing)}")
    if performance.empty:
        raise ValueError("final-performance data must not be empty")
    run_key = ["problem_id", "algorithm", "seed"]
    if performance.duplicated(run_key).any():
        raise ValueError("final-performance data must contain exactly one row per problem/algorithm/seed")
    if not (performance["FE"].astype(int) == performance["FE_total"].astype(int)).all():
        raise ValueError("SBS must use only complete-budget final-performance rows")
    completed = performance["path_completed"].astype(bool)
    if not np.array_equal(
        completed.to_numpy(dtype=bool),
        performance["run_status"].astype(str).eq("completed").to_numpy(dtype=bool),
    ):
        raise ValueError("SBS run_status and path_completed disagree")
    completed_values = performance.loc[completed, ["best_fitness", "benchmark_reference_value"]]
    if not completed_values.apply(
        lambda column: column.map(lambda value: isfinite(float(value))).all()
    ).all():
        raise ValueError("completed SBS final-performance values must be finite")
    numeric = performance[
        [
            "final_gap",
            "log10_gap",
            "log10_gap_floor",
            "log10_gap_cap",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("SBS endpoint and clipped log10-gap values must all be finite")
    floors = performance["log10_gap_floor"].to_numpy(dtype=float)
    caps = performance["log10_gap_cap"].to_numpy(dtype=float)
    if bool((floors <= 0.0).any()) or bool((caps <= floors).any()):
        raise ValueError("SBS clipped-gap bounds must satisfy 0 < floor < cap")
    expected_gap = performance["log10_gap_cap"].to_numpy(dtype=float).copy()
    expected_gap[completed.to_numpy(dtype=bool)] = np.maximum(
        performance.loc[completed, "best_fitness"].to_numpy(dtype=float)
        - performance.loc[completed, "benchmark_reference_value"].to_numpy(dtype=float),
        0.0,
    )
    if not np.allclose(
        performance["final_gap"].to_numpy(dtype=float),
        expected_gap,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("SBS final_gap is inconsistent with the benchmark reference")
    expected_log10_gap = np.log10(np.clip(expected_gap, floors, caps))
    if not np.allclose(
        performance["log10_gap"].to_numpy(dtype=float),
        expected_log10_gap,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("SBS log10_gap is inconsistent with the clipped endpoint gap")
    seed_sets = performance.groupby(["problem_id", "algorithm"])["seed"].agg(
        lambda values: tuple(sorted(int(value) for value in values))
    )
    expected_seed_set = seed_sets.iloc[0]
    if any(seed_set != expected_seed_set for seed_set in seed_sets.iloc[1:]):
        raise ValueError("each problem/algorithm pair must cover the same complete seed set")
    algorithms_by_problem = performance.groupby("problem_id")["algorithm"].agg(
        lambda values: tuple(sorted(set(str(value) for value in values)))
    )
    expected_algorithms = algorithms_by_problem.iloc[0]
    if any(value != expected_algorithms for value in algorithms_by_problem.iloc[1:]):
        raise ValueError("each problem must cover the same algorithm portfolio")
    function_by_problem = performance.groupby("problem_id")["function_id"].nunique()
    family_by_problem = performance.groupby("problem_id")["family"].nunique()
    if not bool((function_by_problem == 1).all()) or not bool((family_by_problem == 1).all()):
        raise ValueError("each SBS problem_id must belong to one function and one landscape family")
    if "cv_group_id" not in performance.columns:
        raise ValueError("final-performance data must contain cv_group_id; regenerate shards with the current protocol")
    cv_group_matches = performance.groupby("function_id")["cv_group_id"].nunique()
    if not bool((cv_group_matches == 1).all()):
        raise ValueError("cv_group_id must be unique per function_id")
    cv_group_by_problem = performance.groupby("problem_id")["cv_group_id"].nunique()
    if not bool((cv_group_by_problem == 1).all()):
        raise ValueError("each SBS problem_id must belong to one cv_group_id")
    problem_scores = performance.groupby(
        ["function_id", "family", "dimension", "problem_id", "algorithm"],
        as_index=False,
    )["log10_gap"].mean()
    dimension_scores = problem_scores.groupby(
        ["function_id", "dimension", "algorithm"],
        as_index=False,
    )["log10_gap"].mean()
    function_scores = dimension_scores.groupby(
        ["function_id", "algorithm"],
        as_index=False,
    )["log10_gap"].mean()
    algorithms_per_function = function_scores.groupby("function_id")["algorithm"].agg(
        lambda values: frozenset(str(value) for value in values)
    )
    observed_algorithms = frozenset(str(value) for value in performance["algorithm"])
    if not bool((algorithms_per_function == observed_algorithms).all()):
        raise ValueError("every SBS function must cover the complete algorithm portfolio")
    scores = function_scores.groupby("algorithm")["log10_gap"].mean()
    order = tuple(portfolio_order or sorted(scores.index.astype(str)))
    if set(order) != set(scores.index.astype(str)) or len(order) != len(set(order)):
        raise ValueError("SBS portfolio order must contain each observed algorithm exactly once")
    minimum = float(scores.min())
    tied = {
        str(algorithm)
        for algorithm, value in scores.items()
        if np.isclose(float(value), minimum, rtol=0.0, atol=1e-12)
    }
    return next(algorithm for algorithm in order if algorithm in tied)


def static_virtual_best_solver_rows(
    performance: pd.DataFrame,
    portfolio_order: tuple[str, ...],
) -> pd.DataFrame:
    """Select one complete-budget algorithm per static problem, then retain paired seeds."""
    # Reuse the complete-coverage and endpoint checks of the SBS calculation.
    single_best_solver(performance, portfolio_order=portfolio_order)
    required = {
        "problem_id",
        "function_id",
        "family",
        "dimension",
        "algorithm",
        "seed",
        "log10_gap",
        "final_gap",
        "success",
        "first_hit_FE",
        "FE_total",
    }
    missing = sorted(required.difference(performance.columns))
    if missing:
        raise ValueError(f"static VBS data missing required columns: {missing}")
    problem_key = ["function_id", "family", "problem_id", "dimension"]
    problem_scores = performance.groupby(
        [*problem_key, "algorithm"],
        sort=True,
        as_index=False,
        dropna=False,
    )["log10_gap"].mean()
    order = tuple(str(value) for value in portfolio_order)
    if set(order) != set(problem_scores["algorithm"].astype(str)) or len(order) != len(set(order)):
        raise ValueError("static VBS portfolio order must contain each observed algorithm exactly once")
    choices: list[dict[str, object]] = []
    for key_values, group in problem_scores.groupby(problem_key, sort=True, dropna=False):
        scores = {
            str(row["algorithm"]): float(row["log10_gap"])
            for _, row in group.iterrows()
        }
        minimum = min(scores.values())
        selected = next(
            algorithm
            for algorithm in order
            if np.isclose(scores[algorithm], minimum, rtol=0.0, atol=1e-12)
        )
        key_tuple = key_values if isinstance(key_values, tuple) else (key_values,)
        choices.append(
            {
                **dict(zip(problem_key, key_tuple, strict=True)),
                "selected_algorithm": selected,
                "selected_algorithm_seed_mean_log10_gap": float(scores[selected]),
                "selection_tie_order": ",".join(order),
            }
        )
    choice_frame = pd.DataFrame(choices)
    selected = choice_frame.merge(
        performance,
        left_on=[*problem_key, "selected_algorithm"],
        right_on=[*problem_key, "algorithm"],
        how="left",
        validate="one_to_many",
    )
    if selected.empty or selected.duplicated([*problem_key, "seed"]).any():
        raise RuntimeError("static VBS must retain exactly one selected outcome per problem and seed")
    expected_rows = performance[[*problem_key, "seed"]].drop_duplicates().shape[0]
    if len(selected) != expected_rows:
        raise RuntimeError("static VBS selected outcomes do not cover every problem and seed")
    selected.insert(0, "reference_name", "static_vbs")
    selected.insert(
        1,
        "reference_protocol",
        "problem_level_seed_aggregated_complete_budget_log10_gap_v1",
    )
    selected["selection_uses_per_seed_hindsight"] = False
    return selected.sort_values([*problem_key, "seed"], kind="mergesort").reset_index(drop=True)


def train_derived_sbs(train_config_path: Path) -> str:
    config = load_config(train_config_path)
    portfolio = tuple(algorithms(config))
    return single_best_solver(
        read_performance(config, None, None),
        portfolio_order=portfolio,
    )


def split_name(config: dict) -> str:
    if "split" in config:
        return str(config["split"])
    return Path(config["output"]).stem.removesuffix("_trajectories")
