from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
import pickle

import pandas as pd

from behavior.extraction import extract_behavior_file
from behavior_with_ela.local_landscape import LocalLandscapeRecorder
from behavior_with_ela.protocol import (
    BOUNDARY_HANDLING,
    ExperimentConfig,
    SuiteConfig,
    check_problem_availability,
    function_label,
    load_experiment_config,
    make_experiment_problem,
)
from optimizers import OptimizerRunResult, OptimizerSettings, run_optimizer
from trajectory import write_final_performance_parquet, write_parquet
from trajectory.sampling import get_sampling_spec


def collect_experiment_trajectories(
    *,
    config_path: str | Path,
    only_splits: tuple[str, ...] | None = None,
    only_functions: tuple[int, ...] | None = None,
    workers: int = 1,
    overwrite: bool = False,
) -> dict[str, int]:
    config = load_experiment_config(config_path)
    if workers < 1:
        raise ValueError("workers must be at least one")
    suites = _selected_suites(config, only_splits)
    tasks = [
        (suite, function)
        for suite in suites
        for function in _selected_functions(suite, only_functions)
    ]
    if not tasks:
        raise ValueError("no trajectory shards were selected")
    check_problem_availability(config, tasks)

    written = 0
    skipped = 0
    pending: list[tuple[SuiteConfig, int]] = []
    for suite, function in tasks:
        paths = collection_shard_paths(config, suite, function)
        existing = tuple(path.exists() for path in paths)
        if all(existing) and not overwrite:
            skipped += 1
            continue
        if any(existing) and not all(existing) and not overwrite:
            raise FileExistsError(
                "trajectory, final-performance and behavior outputs must be regenerated "
                f"together: {paths[0].parent}"
            )
        pending.append((suite, function))

    if workers == 1:
        for suite, function in pending:
            _collect_and_write_shard(config, suite, function, overwrite=overwrite)
            written += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _collect_and_write_shard,
                    config,
                    suite,
                    function,
                    overwrite,
                ): (suite.split, function)
                for suite, function in pending
            }
            for future in as_completed(futures):
                future.result()
                written += 1
    return {"written_shards": written, "skipped_shards": skipped}


def shard_paths(
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
) -> tuple[Path, Path, Path]:
    directory = (
        config.output_root
        / "trajectories"
        / suite.split
        / function_label(suite.suite, function)
        / f"dimension_{config.dimension}"
    )
    return (
        directory / "trajectories.parquet",
        directory / "final_performance.parquet",
        directory / "behavior.parquet",
    )


def local_landscape_shard_path(
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
) -> Path:
    trajectory_path, _, _ = shard_paths(config, suite, function)
    return trajectory_path.with_name("local_landscape.parquet")


def optimizer_checkpoint_shard_path(
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
) -> Path:
    trajectory_path, _, _ = shard_paths(config, suite, function)
    return trajectory_path.with_name("optimizer_checkpoints.parquet")


def collection_shard_paths(
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
) -> tuple[Path, Path, Path, Path, Path]:
    return (
        *shard_paths(config, suite, function),
        local_landscape_shard_path(config, suite, function),
        optimizer_checkpoint_shard_path(config, suite, function),
    )


def _collect_and_write_shard(
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    overwrite: bool,
) -> tuple[int, int]:
    (
        trajectory_path,
        final_path,
        behavior_path,
        local_path,
        checkpoint_path,
    ) = collection_shard_paths(config, suite, function)
    if overwrite:
        for path in (
            trajectory_path,
            final_path,
            behavior_path,
            local_path,
            checkpoint_path,
        ):
            if path.exists() and not path.is_file():
                raise IsADirectoryError(f"output target is not a file: {path}")
            path.unlink(missing_ok=True)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory_records = []
    final_records = []
    local_records: list[dict] = []
    checkpoint_records: list[dict] = []
    for instance, seed, algorithm in product(
        suite.instances,
        config.seeds,
        config.algorithms,
    ):
        result = _run_one(
            config=config,
            suite=suite,
            function=function,
            instance=instance,
            seed=seed,
            algorithm=algorithm,
        )
        trajectory_records.extend(result.trajectory_records)
        final_records.append(result.final_performance)
        local_records.extend(result.local_landscape_records)
        checkpoint_records.extend(result.optimizer_checkpoint_records)
    if not trajectory_records:
        raise RuntimeError(
            f"no decision-state trajectories were produced for {suite.split} function {function}"
        )
    write_parquet(trajectory_records, trajectory_path)
    write_final_performance_parquet(final_records, final_path)
    extract_behavior_file(trajectory_path, behavior_path)
    if len(local_records) != len(trajectory_records):
        raise RuntimeError(
            "local landscape and decision-state trajectory row counts must match"
        )
    if len(checkpoint_records) != len(trajectory_records):
        raise RuntimeError(
            "optimizer checkpoint and decision-state trajectory row counts must match"
        )
    pd.DataFrame(local_records).sort_values(
        ["problem_id", "algorithm", "seed", "FE", "decision_opportunity_index"],
        kind="mergesort",
    ).to_parquet(local_path, index=False)
    pd.DataFrame(checkpoint_records).sort_values(
        ["problem_id", "algorithm", "seed", "FE", "decision_opportunity_index"],
        kind="mergesort",
    ).to_parquet(checkpoint_path, index=False)
    return len(trajectory_records), len(final_records)


def _run_one(
    *,
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    instance: int,
    seed: int,
    algorithm: str,
) -> OptimizerRunResult:
    problem = make_experiment_problem(
        suite,
        function=function,
        instance=instance,
        dimension=config.dimension,
        boundary_handling=config.boundary_handling,
    )
    try:
        if problem.boundary_handling != BOUNDARY_HANDLING:
            raise ValueError("experiment problem did not preserve reflect boundary handling")
        settings = OptimizerSettings(
            population_size=config.population_size,
            sampling_protocol=config.sampling_protocol,
            boundary_handling=config.boundary_handling,
        )
        local = LocalLandscapeRecorder(
            config=config.local_landscape,
            lower_bounds=problem.lower_bounds,
            upper_bounds=problem.upper_bounds,
            seed=seed,
            suite=suite.suite,
            function=function,
            instance=instance,
            dimension=config.dimension,
            algorithm=algorithm,
        )
        checkpoint_records: list[dict] = []

        def capture_optimizer_state(record, opportunity_index, state) -> None:
            checkpoint_records.append(
                {
                    "problem_id": str(record.problem_id),
                    "algorithm": str(record.algorithm),
                    "seed": int(record.seed),
                    "FE": int(record.FE),
                    "decision_opportunity_index": int(opportunity_index),
                    "optimizer_state_payload": pickle.dumps(
                        state,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    ),
                }
            )

        result = run_optimizer(
            algorithm=algorithm,
            problem=problem,
            seed=seed,
            fe_total=config.fe_total,
            settings=settings,
            log10_gap_floor=config.log10_gap_floor,
            log10_gap_cap=config.log10_gap_cap,
            success_gap_target=config.success_gap_target,
            failure_loss_cap=config.failure_loss_cap,
            evaluation_callback=local.observe,
            decision_state_callback=lambda record, opportunity_index: local.snapshot(
                split=suite.split,
                problem_id=record.problem_id,
                function_id=record.function_id,
                family=record.family,
                cv_group_id=record.cv_group_id,
                fe=record.FE,
                fe_total=record.FE_total,
                native_updates=record.native_updates,
                decision_opportunity_index=opportunity_index,
            ),
            decision_optimizer_state_callback=capture_optimizer_state,
        )
        result.local_landscape_records.extend(local.records)
        result.optimizer_checkpoint_records.extend(checkpoint_records)
        _validate_collected_run(config, result)
        return result
    finally:
        problem.close()


def _validate_collected_run(
    config: ExperimentConfig,
    result: OptimizerRunResult,
) -> None:
    endpoint = result.final_performance
    if (
        str(endpoint.run_status) != "completed"
        or not bool(endpoint.path_completed)
        or int(endpoint.FE) != config.fe_total
        or int(endpoint.planned_FE) != config.fe_total
        or int(endpoint.effective_FE) != config.fe_total
    ):
        raise RuntimeError(
            "base trajectory did not complete its strict FE budget: "
            f"problem_id={endpoint.problem_id}, algorithm={endpoint.algorithm}, "
            f"seed={endpoint.seed}, status={endpoint.run_status}, "
            f"effective_FE={endpoint.effective_FE}, planned_FE={endpoint.planned_FE}; "
            f"{endpoint.failure_type}: {endpoint.failure_message}"
        )

    spec = get_sampling_spec(config.sampling_protocol)
    records = result.trajectory_records
    if not spec.min_samples_per_run <= len(records) <= spec.max_samples_per_run:
        raise RuntimeError(
            "base trajectory decision-state count is outside the sampling protocol: "
            f"problem_id={endpoint.problem_id}, algorithm={endpoint.algorithm}, "
            f"seed={endpoint.seed}, states={len(records)}, "
            f"expected={spec.min_samples_per_run}..{spec.max_samples_per_run}"
        )
    fe_values = tuple(int(record.FE) for record in records)
    if fe_values != tuple(sorted(set(fe_values))):
        raise RuntimeError("base trajectory decision-state FE values must increase uniquely")
    if any(
        int(record.FE_total) != config.fe_total
        or str(record.sampling_protocol) != config.sampling_protocol
        for record in records
    ):
        raise RuntimeError("base trajectory decision states differ from the configured protocol")

    observed_milestones = tuple(
        sorted(
            float(record.budget_milestone_ratio)
            for record in records
            if bool(record.is_budget_milestone)
            and record.budget_milestone_ratio is not None
        )
    )
    if observed_milestones != spec.budget_milestone_ratios:
        raise RuntimeError(
            "base trajectory does not contain exactly the predefined budget milestones"
        )
    for phase in ("early", "mid", "late"):
        event_only_count = sum(
            str(record.sampling_phase) == phase
            and bool(record.is_event_sample)
            and not bool(record.is_budget_milestone)
            for record in records
        )
        if event_only_count > spec.max_event_only_per_phase:
            raise RuntimeError(
                f"base trajectory exceeds the {phase} event-only state quota"
            )

    if len(result.local_landscape_records) != len(records):
        raise RuntimeError(
            "base trajectory local-landscape rows do not match decision states"
        )
    if len(result.optimizer_checkpoint_records) != len(records):
        raise RuntimeError(
            "base trajectory optimizer checkpoints do not match decision states"
        )


def _selected_suites(
    config: ExperimentConfig,
    only_splits: tuple[str, ...] | None,
) -> tuple[SuiteConfig, ...]:
    if only_splits is None:
        return config.suites
    requested = set(only_splits)
    missing = requested.difference(suite.split for suite in config.suites)
    if missing:
        raise ValueError(f"requested split is absent from config: {sorted(missing)}")
    return tuple(suite for suite in config.suites if suite.split in requested)


def _selected_functions(
    suite: SuiteConfig,
    only_functions: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if only_functions is None:
        return suite.functions
    requested = set(only_functions)
    return tuple(function for function in suite.functions if function in requested)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Behavior-with-ELA 10D decision-state trajectories."
    )
    parser.add_argument("--config", default="configs/behavior_with_ela_train.yaml")
    parser.add_argument("--only-split", action="append", default=None)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = collect_experiment_trajectories(
        config_path=args.config,
        only_splits=None if args.only_split is None else tuple(args.only_split),
        only_functions=(
            None if args.only_function is None else tuple(args.only_function)
        ),
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(
        f"wrote {summary['written_shards']} trajectory shards; "
        f"skipped {summary['skipped_shards']} existing shards"
    )


if __name__ == "__main__":
    main()
