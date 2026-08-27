from __future__ import annotations

import argparse
import os
import platform
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from benchmarks import make_problem
from experiments.phase1_batch_common import (
    load_suite_configs,
    load_config,
    make_shards,
    runtime_problem_config,
    split_name,
)
from landscape_queries.cheap import calculate_descriptor_cheap
from landscape_queries.sampling import sample_problem
from landscape_queries.specs import MAIN_QUERY_ID, get_query_spec
from optimizers import OptimizerSettings, advance_optimizer_state, initialize_optimizer_state
from selection_reference.action_losses import (
    BEHAVIOR_ONLY_FULL_BUDGET,
    QUERY_ADJUSTED_BUDGET,
    _TimedObjective,
    _evaluate_native_skip,
    _evaluate_one_candidate_action,
    _validate_replayed_checkpoint,
)
from utility_labels.fields import TIMING_REPLAY_STATUSES


COMPLETE_PATHS = (
    "skip",
    "query_joint",
    "query_matched_state_only",
    "sampling_only_continue_current",
    "behavior_only_full_budget",
)
STATE_COLUMNS = (
    "problem_id",
    "function_id",
    "family",
    "cv_group_id",
    "dimension",
    "prefix_algorithm",
    "seed",
    "FE",
)
TIMING_SOURCE = "measured_complete_policy_path"
TIMING_ORIGIN = "decision_state_to_terminal"
TIMING_ORDER_PROTOCOL = "cyclic_complete_path_v1"
TIMING_REPETITIONS = 3


def run_complete_path_replays(
    *,
    replay_plan_path: Path,
    config_paths: list[Path],
    output_dir: Path,
    workers: int,
    overwrite: bool,
    summarize_only: bool,
    only_storage_splits: list[str] | None,
    max_runs: int | None,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    plan = pq.read_table(replay_plan_path).to_pandas()
    configs = _suite_config_index(config_paths)
    plan = _attach_storage_splits(plan, configs)
    if only_storage_splits:
        requested = set(str(value) for value in only_storage_splits)
        unknown = sorted(requested.difference(set(plan["storage_split"].astype(str))))
        if unknown:
            raise ValueError(f"requested storage splits are absent from replay plan: {unknown}")
        plan = plan[plan["storage_split"].astype(str).isin(requested)].copy()
    _validate_plan(plan)
    run_columns = ["storage_split", *STATE_COLUMNS[:-1]]
    run_keys = (
        plan[run_columns]
        .drop_duplicates()
        .sort_values(run_columns, kind="mergesort")
        .reset_index(drop=True)
    )
    if max_runs is not None:
        run_keys = run_keys.iloc[: int(max_runs)].copy()
    if run_keys.empty:
        raise ValueError("replay plan selection contains no trajectory runs")
    plan = plan.merge(run_keys, on=run_columns, how="inner", validate="many_to_one")

    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    skipped = 0
    for run_index, run_key in run_keys.iterrows():
        output_path = shard_dir / f"run_{int(run_index):06d}.parquet"
        if output_path.exists() and not overwrite:
            skipped += 1
            continue
        mask = np.ones(len(plan), dtype=bool)
        for column in run_columns:
            mask &= plan[column].astype(str).eq(str(run_key[column])).to_numpy()
        run_plan = plan.loc[mask].copy()
        storage_split = str(run_key["storage_split"])
        jobs.append(
            {
                "run_index": int(run_index),
                "plan_records": run_plan.to_dict(orient="records"),
                "config": configs[storage_split],
                "storage_split": storage_split,
                "output_path": str(output_path),
                "overwrite": bool(overwrite),
            }
        )
    if not summarize_only:
        _execute_jobs(jobs, workers)
    expected_paths = [shard_dir / f"run_{int(index):06d}.parquet" for index in run_keys.index]
    missing = [path for path in expected_paths if not path.exists()]
    if missing:
        return {
            "planned_runs": int(len(run_keys)),
            "completed_run_shards": int(len(expected_paths) - len(missing)),
            "missing_run_shards": int(len(missing)),
            "skipped_existing_run_shards": int(skipped),
            "output": "",
        }
    frames = [pq.read_table(path).to_pandas() for path in expected_paths]
    timings = pd.concat(frames, ignore_index=True)
    _validate_timing_output(timings, plan)
    output_path = output_dir / "complete_path_timings.parquet"
    pq.write_table(pa.Table.from_pandas(timings, preserve_index=False), output_path)
    summary = _timing_summary(timings)
    summary.to_csv(output_dir / "complete_path_timing_summary.csv", index=False)
    pq.write_table(
        pa.Table.from_pandas(summary, preserve_index=False),
        output_dir / "complete_path_timing_summary.parquet",
    )
    return {
        "planned_runs": int(len(run_keys)),
        "completed_run_shards": int(len(expected_paths)),
        "missing_run_shards": 0,
        "skipped_existing_run_shards": int(skipped),
        "rows": int(len(timings)),
        "output": str(output_path),
    }


def _suite_config_index(config_paths: list[Path]) -> dict[str, dict]:
    configs: dict[str, dict] = {}
    for path in config_paths:
        combined = load_config(path)
        dataset_role = str(combined.get("dataset", combined.get("split", "")))
        for config in load_suite_configs(path):
            config = dict(config)
            config["dataset_role"] = dataset_role
            storage_split = split_name(config)
            if storage_split in configs:
                raise ValueError(f"duplicate storage split across replay configs: {storage_split}")
            configs[storage_split] = config
    if not configs:
        raise ValueError("complete-path replay requires at least one suite config")
    return configs


def _attach_storage_splits(plan: pd.DataFrame, configs: dict[str, dict]) -> pd.DataFrame:
    output = plan.copy()
    suite_by_split = {
        storage_split: str(config["suite"]).lower()
        for storage_split, config in configs.items()
    }
    role_by_split = {
        storage_split: str(config.get("dataset_role", ""))
        for storage_split, config in configs.items()
    }
    resolved: list[str] = []
    for row in output.to_dict(orient="records"):
        logical_role = str(row["split"])
        problem_id = str(row["problem_id"])
        suite = _suite_from_problem_id(problem_id)
        matches = [
            storage_split
            for storage_split, configured_suite in suite_by_split.items()
            if configured_suite == suite and role_by_split[storage_split] == logical_role
        ]
        if len(matches) != 1:
            raise ValueError(
                "cannot resolve one storage split for replay row: "
                f"role={logical_role}, suite={suite}, problem_id={problem_id}, matches={matches}"
            )
        resolved.append(matches[0])
    output["dataset_role"] = output["split"].astype(str)
    output["storage_split"] = resolved
    return output


def _suite_from_problem_id(problem_id: str) -> str:
    if problem_id.startswith("bbob_"):
        return "bbob"
    if problem_id.startswith("mabbob_"):
        return "mabbob"
    if problem_id.startswith("cec2017_"):
        return "cec2017"
    if problem_id.startswith("cec2022_"):
        return "cec2022"
    raise ValueError(f"unsupported replay problem_id: {problem_id}")


def _execute_jobs(jobs: list[dict[str, Any]], workers: int) -> None:
    if workers == 1:
        for job in jobs:
            result = _run_one_trajectory(job)
            print(f"wrote {result['rows']} timing rows to {result['output']}")
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_one_trajectory, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            print(f"wrote {result['rows']} timing rows to {result['output']}")


def _run_one_trajectory(job: dict[str, Any]) -> dict[str, Any]:
    output_path = Path(job["output_path"])
    if output_path.exists() and not bool(job["overwrite"]):
        return {"rows": int(pq.read_metadata(output_path).num_rows), "output": str(output_path)}
    plan = pd.DataFrame(job["plan_records"])
    config = dict(job["config"])
    storage_split = str(job["storage_split"])
    first = plan.iloc[0]
    function, instance, dimension = _parse_problem_id(str(first["problem_id"]))
    prefix_algorithm = str(first["prefix_algorithm"])
    seed = int(first["seed"])
    shard_specs = make_shards(config, [function], [dimension])
    if len(shard_specs) != 1:
        raise ValueError("replay run must resolve exactly one trajectory shard")
    shard = shard_specs[0]
    trajectories = pq.read_table(shard.output_path).to_pandas()
    trajectories = trajectories[
        trajectories["problem_id"].astype(str).eq(str(first["problem_id"]))
        & trajectories["algorithm"].astype(str).eq(prefix_algorithm)
        & trajectories["seed"].astype(int).eq(seed)
    ].sort_values(["FE", "decision_opportunity_index"], kind="mergesort")
    target_fes = set(plan["FE"].astype(int))
    trajectories = trajectories[trajectories["FE"].astype(int).isin(target_fes)].copy()
    if set(trajectories["FE"].astype(int)) != target_fes:
        raise ValueError("replay plan references decision states absent from trajectory shard")
    final_performance = pq.read_table(shard.final_performance_path).to_pandas()
    final_row = final_performance[
        final_performance["problem_id"].astype(str).eq(str(first["problem_id"]))
        & final_performance["algorithm"].astype(str).eq(prefix_algorithm)
        & final_performance["seed"].astype(int).eq(seed)
    ]
    if len(final_row) != 1:
        raise ValueError("replay run requires one final-performance row")
    final_row = final_row.iloc[0]
    final_first_hit = (
        None if pd.isna(final_row["first_hit_FE"]) else int(final_row["first_hit_FE"])
    )
    problem = make_problem(
        runtime_problem_config(
            config, function=function, instance=instance, dimension=dimension
        )
    )
    settings = OptimizerSettings(
        population_size=int(config["population_size"]), checkpoint_ratios=(1.0,)
    )
    state = initialize_optimizer_state(
        algorithm=prefix_algorithm, problem=problem, seed=seed, settings=settings
    )
    rows: list[dict[str, Any]] = []
    try:
        with threadpool_limits(limits=1):
            for trajectory_row in trajectories.to_dict(orient="records"):
                delta = int(trajectory_row["FE"]) - int(state.evaluations)
                if delta < 0:
                    raise ValueError("trajectory checkpoint FE moved backwards during replay")
                advance_optimizer_state(state=state, problem=problem, fe_budget=delta)
                _validate_replayed_checkpoint(state, trajectory_row)
                checkpoint_fe = int(trajectory_row["FE"])
                prefix_first_hit = (
                    final_first_hit
                    if final_first_hit is not None and final_first_hit <= checkpoint_fe
                    else None
                )
                state_plan = plan[plan["FE"].astype(int).eq(checkpoint_fe)]
                for learning_fold_role, role_plan in state_plan.groupby(
                    "learning_fold_role", sort=True, dropna=False
                ):
                    rows.extend(
                        _run_state_role(
                            role_plan=role_plan,
                            checkpoint_state=state,
                            problem=problem,
                            config=config,
                            storage_split=storage_split,
                            function=function,
                            instance=instance,
                            checkpoint_fe=checkpoint_fe,
                            prefix_first_hit_fe=prefix_first_hit,
                            learning_fold_role=str(learning_fold_role),
                        )
                    )
    finally:
        problem.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output_path)
    return {"rows": len(rows), "output": str(output_path)}


def _run_state_role(
    *,
    role_plan: pd.DataFrame,
    checkpoint_state,
    problem,
    config: dict,
    storage_split: str,
    function: int,
    instance: int,
    checkpoint_fe: int,
    prefix_first_hit_fe: int | None,
    learning_fold_role: str,
) -> list[dict[str, Any]]:
    if len(role_plan) != len(COMPLETE_PATHS):
        raise ValueError("each state/fold replay group must contain five paths")
    plan_by_path = {str(row["path"]): row for row in role_plan.to_dict(orient="records")}
    if set(plan_by_path) != set(COMPLETE_PATHS):
        raise ValueError("state/fold replay paths differ from the formal five-path set")
    timeout = float(config["timing_replay_timeout_seconds"])
    environment_id = _timing_environment_id()
    rows: list[dict[str, Any]] = []
    for repetition_index in range(TIMING_REPETITIONS):
        order = COMPLETE_PATHS[repetition_index:] + COMPLETE_PATHS[:repetition_index]
        for order_position, path in enumerate(order):
            plan_row = plan_by_path[path]
            rows.append(
                _execute_path(
                    plan_row=plan_row,
                    checkpoint_state=checkpoint_state,
                    problem=problem,
                    config=config,
                    storage_split=storage_split,
                    function=function,
                    instance=instance,
                    checkpoint_fe=checkpoint_fe,
                    prefix_first_hit_fe=prefix_first_hit_fe,
                    learning_fold_role=learning_fold_role,
                    repetition_index=repetition_index,
                    order_position=order_position,
                    timeout_seconds=timeout,
                    timing_environment_id=environment_id,
                )
            )
    return rows


def _execute_path(
    *,
    plan_row: dict[str, Any],
    checkpoint_state,
    problem,
    config: dict,
    storage_split: str,
    function: int,
    instance: int,
    checkpoint_fe: int,
    prefix_first_hit_fe: int | None,
    learning_fold_role: str,
    repetition_index: int,
    order_position: int,
    timeout_seconds: float,
    timing_environment_id: str,
) -> dict[str, Any]:
    path = str(plan_row["path"])
    selected_algorithm = str(plan_row["selected_algorithm"])
    path_started = perf_counter()
    query_effective_fe = 0
    query_best_gap = float("inf")
    query_first_hit_fe: int | None = None
    acquisition_completed = True
    acquisition_timed_out = False
    acquisition_failure = ""
    try:
        if path in {
            "query_joint",
            "query_matched_state_only",
            "sampling_only_continue_current",
        }:
            spec = get_query_spec(str(plan_row["query_id"]))
            if spec.query_id != MAIN_QUERY_ID:
                raise ValueError("formal replay currently supports the native main query only")
            acquisition_tracker = _TimedObjective(
                problem=problem,
                deadline=path_started + timeout_seconds,
                reference_value=float(problem.reference_value),
                success_gap_target=float(config["success_gap_target"]),
                first_evaluation_fe=checkpoint_fe + 1,
            )
            sample = sample_problem(
                problem=acquisition_tracker.wrapped_problem(),
                sample_design=spec.sample_design,
                base_seed=0,
                function=function,
                instance=instance,
                success_gap_target=float(config["success_gap_target"]),
                failure_loss_cap=float(config["failure_loss_cap"]),
            )
            query_effective_fe = int(sample["sample_effective_FE"])
            query_best_gap = float(sample["query_best_gap"])
            query_first_hit_fe = (
                None
                if sample["query_first_hit_offset"] is None
                else checkpoint_fe + int(sample["query_first_hit_offset"])
            )
            acquisition_completed = bool(sample["sample_path_completed"])
            acquisition_timed_out = bool(acquisition_tracker.timed_out)
            acquisition_failure = str(sample["sample_failure"])
            if acquisition_completed and bool(plan_row["descriptor_computation_required"]):
                calculate_descriptor_cheap(
                    np.asarray(sample["X"], dtype=float),
                    np.asarray(sample["y"], dtype=float),
                    np.asarray(sample["lower_bounds"], dtype=float),
                    np.asarray(sample["upper_bounds"], dtype=float),
                )
        elapsed = perf_counter() - path_started
        remaining_timeout = max(timeout_seconds - elapsed, np.finfo(float).eps)
        fe_total = int(plan_row["FE_total"])
        if path in {
            "query_joint",
            "query_matched_state_only",
            "sampling_only_continue_current",
        }:
            action_budget = fe_total - checkpoint_fe - int(
                get_query_spec(str(plan_row["query_id"])).sample_design.sample_size(
                    int(plan_row["dimension"])
                )
            )
            action_mode = QUERY_ADJUSTED_BUDGET
            action_start_fe = checkpoint_fe + int(
                get_query_spec(str(plan_row["query_id"])).sample_design.sample_size(
                    int(plan_row["dimension"])
                )
            )
        else:
            action_budget = fe_total - checkpoint_fe
            action_mode = BEHAVIOR_ONLY_FULL_BUDGET
            action_start_fe = checkpoint_fe
        if path == "skip":
            outcome = _evaluate_native_skip(
                checkpoint_state=checkpoint_state,
                problem=problem,
                fe_budget=action_budget,
                failure_loss_cap=float(config["failure_loss_cap"]),
                success_gap_target=float(config["success_gap_target"]),
                checkpoint_fe=checkpoint_fe,
                prefix_first_hit_fe=prefix_first_hit_fe,
                action_timeout_seconds=remaining_timeout,
            )
            action_gap = float(outcome["p_skip"])
            action_effective_fe = int(outcome["skip_effective_FE"])
            continuation_first_hit = outcome["skip_continuation_first_hit_FE"]
            action_completed = bool(outcome["skip_path_completed"])
            action_timed_out = bool(outcome["skip_timed_out"])
            action_failure = str(outcome["skip_failure_message"])
        else:
            outcome = _evaluate_one_candidate_action(
                checkpoint_state=checkpoint_state,
                problem=problem,
                target_algorithm=selected_algorithm,
                fe_budget=action_budget,
                seed=int(plan_row["seed"]),
                function=function,
                instance=instance,
                action_budget_mode=action_mode,
                failure_loss_cap=float(config["failure_loss_cap"]),
                success_gap_target=float(config["success_gap_target"]),
                action_start_fe=action_start_fe,
                prefix_first_hit_fe=prefix_first_hit_fe,
                action_timeout_seconds=remaining_timeout,
            )
            action_gap = float(outcome["action_loss"])
            action_effective_fe = int(outcome["effective_FE"])
            continuation_first_hit = outcome["continuation_first_hit_FE"]
            action_completed = bool(outcome["path_completed"])
            action_timed_out = bool(outcome["timed_out"])
            action_failure = str(outcome["failure_message"])
        observed_hits = [
            int(value)
            for value in (prefix_first_hit_fe, query_first_hit_fe, continuation_first_hit)
            if value is not None
        ]
        observed_first_hit = min(observed_hits) if observed_hits else None
        uses_query = path in {
            "query_joint",
            "query_matched_state_only",
            "sampling_only_continue_current",
        }
        planned_fe = int(plan_row["expected_planned_FE"])
        effective_fe = (
            checkpoint_fe
            + action_effective_fe
            + (query_effective_fe if uses_query else 0)
        )
        path_completed = bool(acquisition_completed and action_completed and effective_fe == planned_fe)
        timed_out = bool(acquisition_timed_out or action_timed_out)
        status = "completed" if path_completed else ("timed_out" if timed_out else "failed")
        terminal_gap = (
            min(action_gap, query_best_gap)
            if uses_query and acquisition_completed and action_completed
            else action_gap
        )
        runtime_seconds = max(perf_counter() - path_started, np.finfo(float).eps)
        target_hit_observed = observed_first_hit is not None
        return {
            "learning_fold_role": learning_fold_role,
            "split": storage_split,
            "dataset_role": str(plan_row["split"]),
            **{column: plan_row[column] for column in STATE_COLUMNS},
            "path": path,
            "repetition_index": int(repetition_index),
            "order_position": int(order_position),
            "runtime_seconds": float(runtime_seconds),
            "timing_repetitions": TIMING_REPETITIONS,
            "timing_order_protocol": TIMING_ORDER_PROTOCOL,
            "timing_source": TIMING_SOURCE,
            "timing_origin": TIMING_ORIGIN,
            "timing_environment_id": timing_environment_id,
            "thread_count": 1,
            "selected_algorithm": selected_algorithm,
            "terminal_gap": float(terminal_gap),
            "observed_first_hit_FE": observed_first_hit,
            "target_hit_observed": bool(target_hit_observed),
            "target_hit_before_failure": bool(target_hit_observed and not path_completed),
            "endpoint_success": bool(target_hit_observed and path_completed),
            "first_hit_FE": observed_first_hit,
            "success": bool(target_hit_observed),
            "planned_FE": planned_fe,
            "effective_FE": int(effective_fe),
            "timed_out": timed_out,
            "path_completed": path_completed,
            "timing_replay_status": status,
            "timing_replay_timeout_seconds": float(timeout_seconds),
            "failure_message": "; ".join(
                value for value in (acquisition_failure, action_failure) if value
            )[:500],
        }
    except Exception as exc:
        runtime_seconds = max(perf_counter() - path_started, np.finfo(float).eps)
        return {
            "learning_fold_role": learning_fold_role,
            "split": storage_split,
            "dataset_role": str(plan_row["split"]),
            **{column: plan_row[column] for column in STATE_COLUMNS},
            "path": path,
            "repetition_index": int(repetition_index),
            "order_position": int(order_position),
            "runtime_seconds": float(runtime_seconds),
            "timing_repetitions": TIMING_REPETITIONS,
            "timing_order_protocol": TIMING_ORDER_PROTOCOL,
            "timing_source": TIMING_SOURCE,
            "timing_origin": TIMING_ORIGIN,
            "timing_environment_id": timing_environment_id,
            "thread_count": 1,
            "selected_algorithm": selected_algorithm,
            "terminal_gap": float(config["failure_loss_cap"]),
            "observed_first_hit_FE": prefix_first_hit_fe,
            "target_hit_observed": bool(prefix_first_hit_fe is not None),
            "target_hit_before_failure": bool(prefix_first_hit_fe is not None),
            "endpoint_success": False,
            "first_hit_FE": prefix_first_hit_fe,
            "success": bool(prefix_first_hit_fe is not None),
            "planned_FE": int(plan_row["expected_planned_FE"]),
            "effective_FE": int(checkpoint_fe + query_effective_fe),
            "timed_out": bool(acquisition_timed_out),
            "path_completed": False,
            "timing_replay_status": "timed_out" if acquisition_timed_out else "failed",
            "timing_replay_timeout_seconds": float(timeout_seconds),
            "failure_message": f"{type(exc).__name__}: {exc}"[:500],
        }


def _validate_plan(plan: pd.DataFrame) -> None:
    required = {
        "learning_fold_role",
        "split",
        "storage_split",
        *STATE_COLUMNS,
        "path",
        "canonical_path_order",
        "selected_algorithm",
        "selected_equals_prefix",
        "handoff_required",
        "handoff_type",
        "query_id",
        "FE_total",
        "descriptor_computation_required",
        "expected_planned_FE",
        "timing_repetitions",
        "timing_order_protocol",
        "required_timing_source",
        "timing_origin",
    }
    missing = sorted(required.difference(plan.columns))
    if missing:
        raise ValueError(f"replay plan is missing columns: {missing}")
    if plan.empty:
        raise ValueError("replay plan is empty")
    if set(plan["path"].astype(str)) != set(COMPLETE_PATHS):
        raise ValueError("replay plan must contain exactly the five policy paths")
    if set(plan["timing_repetitions"].astype(int)) != {TIMING_REPETITIONS}:
        raise ValueError("replay plan must request exactly three repetitions")
    if set(plan["timing_order_protocol"].astype(str)) != {TIMING_ORDER_PROTOCOL}:
        raise ValueError("replay plan uses the wrong cyclic order protocol")
    if set(plan["required_timing_source"].astype(str)) != {TIMING_SOURCE}:
        raise ValueError("replay plan does not require measured complete policy paths")
    if set(plan["timing_origin"].astype(str)) != {TIMING_ORIGIN}:
        raise ValueError("replay timing must start at the decision state")


def _validate_timing_output(timings: pd.DataFrame, plan: pd.DataFrame) -> None:
    if set(timings["timing_replay_status"].astype(str)).difference(TIMING_REPLAY_STATUSES):
        raise ValueError("timing output contains an unsupported replay status")
    key = ["learning_fold_role", "dataset_role", *STATE_COLUMNS, "path", "repetition_index"]
    if timings.duplicated(key).any():
        raise ValueError("timing output contains duplicate fold/state/path repetitions")
    counts = timings.groupby(key[:-1], dropna=False).size()
    if not bool((counts == TIMING_REPETITIONS).all()):
        raise ValueError("each fold/state/path must contain exactly three repetitions")
    plan_keys = plan[["learning_fold_role", "split", *STATE_COLUMNS, "path"]].rename(
        columns={"split": "dataset_role"}
    )
    timing_keys = timings[key[:-1]].drop_duplicates()
    if not plan_keys.sort_values(list(plan_keys.columns)).reset_index(drop=True).equals(
        timing_keys.sort_values(list(timing_keys.columns)).reset_index(drop=True)
    ):
        raise ValueError("timing output does not cover the requested replay plan exactly")
    for _, group in timings.groupby(
        ["learning_fold_role", "dataset_role", *STATE_COLUMNS, "repetition_index"],
        dropna=False,
    ):
        if set(group["path"].astype(str)) != set(COMPLETE_PATHS):
            raise ValueError("each timing cycle must contain all five paths")
        if set(group["order_position"].astype(int)) != set(range(len(COMPLETE_PATHS))):
            raise ValueError("each timing cycle must use positions 0 through 4")


def _timing_summary(timings: pd.DataFrame) -> pd.DataFrame:
    return (
        timings.groupby(
            ["split", "dataset_role", "path", "timing_replay_status"],
            as_index=False,
            dropna=False,
        )
        .agg(rows=("runtime_seconds", "size"), median_runtime_seconds=("runtime_seconds", "median"))
        .sort_values(["dataset_role", "split", "path", "timing_replay_status"])
        .reset_index(drop=True)
    )


def _parse_problem_id(problem_id: str) -> tuple[int, int, int]:
    patterns = (
        r"^bbob_f(\d+)_i(\d+)_d(\d+)$",
        r"^mabbob_c(\d+)_i(\d+)_d(\d+)$",
        r"^cec(?:2017|2022)_f(\d+)_i?(\d*)_d(\d+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, problem_id)
        if match:
            function = int(match.group(1))
            instance = int(match.group(2)) if match.group(2) else 1
            return function, instance, int(match.group(3))
    raise ValueError(f"cannot parse replay problem_id: {problem_id}")


def _timing_environment_id() -> str:
    return ":".join(
        (
            platform.node() or "unknown-host",
            platform.machine() or "unknown-machine",
            platform.python_version(),
            "single-thread",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute fold-specific decision-state-to-terminal complete-policy timing replays."
    )
    parser.add_argument("--replay-plan", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=None,
        help="Combined train/validation config; repeat for both dataset roles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/descriptor_cheap_invariant/complete_path_replay"),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--only-storage-split", action="append", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    config_paths = args.config or [
        Path("configs/phase1_train.yaml"),
        Path("configs/phase1_validation.yaml"),
    ]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    summary = run_complete_path_replays(
        replay_plan_path=args.replay_plan,
        config_paths=config_paths,
        output_dir=args.output_dir,
        workers=args.workers,
        overwrite=args.overwrite,
        summarize_only=args.summarize_only,
        only_storage_splits=args.only_storage_split,
        max_runs=args.max_runs,
    )
    print(summary)


if __name__ == "__main__":
    main()
