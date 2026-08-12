from __future__ import annotations

from pathlib import Path
from math import isfinite

import pandas as pd
import pyarrow.parquet as pq

from experiments.phase1_batch_common import (
    algorithms,
    as_int_list,
    family_name,
    fe_total_for_dimension,
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
                "family",
                "dimension",
                "algorithm",
                "seed",
                "FE",
                "FE_total",
                "native_updates",
                "best_fitness",
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
        if set(frame["family"].astype(str)) != {
            family_name(str(config["suite"]), shard.function)
        }:
            raise ValueError(
                "final-performance family differs from its configured shard: "
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
        raise ValueError(f"trajectory shard is empty: {trajectory_path}")
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
    if set(comparison["_merge"].astype(str)) != {"both"}:
        raise ValueError(
            "trajectory and final-performance run coverage differ: "
            f"{trajectory_path.parent}"
        )
    if not (
        comparison["FE_total_final"].astype(int)
        == comparison["FE_total_trajectory"].astype(int)
    ).all():
        raise ValueError(
            "trajectory and final-performance budgets differ: "
            f"{trajectory_path.parent}"
        )
    if not (
        comparison["native_updates_final"].astype(int)
        >= comparison["native_updates_trajectory"].astype(int)
    ).all():
        raise ValueError(
            "final performance precedes the last trajectory state: "
            f"{trajectory_path.parent}"
        )
    if not (
        comparison["best_fitness_final"].astype(float)
        <= comparison["best_fitness_trajectory"].astype(float)
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
    required = {"problem_id", "algorithm", "seed", "FE", "FE_total", "best_fitness"}
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
    if not performance["best_fitness"].map(lambda value: isfinite(float(value))).all():
        raise ValueError("SBS final-performance values must all be finite")
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
    means = performance.groupby(["problem_id", "algorithm"], as_index=False)["best_fitness"].mean()
    means["rank"] = means.groupby("problem_id")["best_fitness"].rank(method="average", ascending=True)
    ranks = means.groupby("algorithm")["rank"].mean()
    order = tuple(portfolio_order or sorted(ranks.index.astype(str)))
    if set(order) != set(ranks.index.astype(str)) or len(order) != len(set(order)):
        raise ValueError("SBS portfolio order must contain each observed algorithm exactly once")
    minimum = float(ranks.min())
    tied = {
        str(algorithm)
        for algorithm, value in ranks.items()
        if float(value) == minimum
    }
    return next(algorithm for algorithm in order if algorithm in tied)


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
