from __future__ import annotations

import argparse
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks import make_problem
from benchmarks.core import Problem
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
from landscape_queries.specs import SAMPLE_DESIGN_SPECS, get_sample_design_spec
from optimizers import (
    NO_QUERY_TRANSFER_EVENT,
    OptimizerSettings,
    QUERY_TRANSFER_EVENT,
    advance_optimizer_state,
    clone_optimizer_state,
    initialize_optimizer_state,
    initialize_transferred_optimizer_state,
)
from optimizers.state import OptimizerState
from selection_reference.common import split_name, train_derived_sbs
from trajectory.sampling import SAMPLING_METADATA_COLUMNS, SAMPLING_METADATA_SCHEMA_FIELDS


EPS = 1e-12
MIN_LABEL_RATIO = 0.12
ACTION_LOSS_PROTOCOL = "shared_complete_state_observed_action_loss"
QUERY_ADJUSTED_BUDGET = "query_adjusted_budget"
BEHAVIOR_ONLY_FULL_BUDGET = "behavior_only_full_budget"
# Kept as a source-level alias because several callers use the descriptive
# constant name; the serialized value is uniquely `behavior_only_full_budget`.
FULL_REMAINING_BUDGET = BEHAVIOR_ONLY_FULL_BUDGET
PRE_RUN_QUERY_ADJUSTED_BUDGET = "pre_run_query_adjusted_budget"
ACTION_BUDGET_MODES = (
    QUERY_ADJUSTED_BUDGET,
    BEHAVIOR_ONLY_FULL_BUDGET,
    PRE_RUN_QUERY_ADJUSTED_BUDGET,
)
NOT_APPLICABLE = "not_applicable"
ACTION_ORDER_STREAM_CODE = 2026081401
EXECUTION_ORDER_PROTOCOL = "seeded_single_action_outcome_order_v1"
ACTION_OUTCOME_EXECUTIONS = 1
# Compatibility name used by the Selector reader.  Action outcomes are executed
# once; the separate selected complete-policy replay owns the three-repeat timing.
TIMING_REPETITIONS = ACTION_OUTCOME_EXECUTIONS
COMPLETE_PATH_TIMING_REPETITIONS = 3
COMPLETE_PATH_TIMING_ORDER_PROTOCOL = "cyclic_complete_path_v1"
FROZEN_PORTFOLIO_ORDER = ("de", "pso", "cmaes", "shade")
ACTION_BUDGET_STREAM_CODES = {
    QUERY_ADJUSTED_BUDGET: 11,
    BEHAVIOR_ONLY_FULL_BUDGET: 17,
    PRE_RUN_QUERY_ADJUSTED_BUDGET: 23,
}
STATE_KEY_COLUMNS = (
    "split",
    "problem_id",
    "function_id",
    "family",
    "cv_group_id",
    "dimension",
    "prefix_algorithm",
    "seed",
    "FE",
)


class _TimedObjective:
    """Evaluate one point at a time so a path deadline has exact FE accounting."""

    def __init__(
        self,
        *,
        problem: Problem,
        deadline: float,
        reference_value: float,
        success_gap_target: float,
        first_evaluation_fe: int,
    ) -> None:
        self.problem = problem
        self.deadline = float(deadline)
        self.reference_value = float(reference_value)
        self.success_gap_target = float(success_gap_target)
        self.first_evaluation_fe = int(first_evaluation_fe)
        self.evaluations = 0
        self.first_hit_fe: int | None = None
        self.timed_out = False

    def __call__(self, values: np.ndarray) -> np.ndarray:
        points = np.asarray(values, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        results: list[float] = []
        for point in points:
            if perf_counter() >= self.deadline:
                self.timed_out = True
                raise TimeoutError("action path exceeded action_timeout_seconds before evaluation")
            value = float(self.problem.evaluate(point)[0])
            self.evaluations += 1
            gap = max(value - self.reference_value, 0.0)
            if self.first_hit_fe is None and gap <= self.success_gap_target:
                self.first_hit_fe = self.first_evaluation_fe + self.evaluations - 1
            results.append(value)
            if perf_counter() >= self.deadline:
                self.timed_out = True
                raise TimeoutError("action path exceeded action_timeout_seconds after evaluation")
        return np.asarray(results, dtype=float)

    def wrapped_problem(self) -> Problem:
        return Problem(
            problem_id=self.problem.problem_id,
            function_id=self.problem.function_id,
            family=self.problem.family,
            dimension=self.problem.dimension,
            suite_code=self.problem.suite_code,
            function_number=self.problem.function_number,
            instance_number=self.problem.instance_number,
            bounds=self.problem.bounds.copy(),
            objective=self,
            reference_value=self.problem.reference_value,
        )


def generate_state_action_losses(
    *,
    config_path: Path,
    train_config_path: Path,
    sample_design_id: str | None,
    action_budget_mode: str,
    output_path: Path,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    default_algorithm: str | None,
    all_prefixes: bool,
    max_states: int | None,
    workers: int,
    overwrite: bool,
) -> dict[str, int | str]:
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"state-action loss output already exists; pass --overwrite: {output_path}")
        output_path.unlink()
    config = load_config(config_path)
    if action_budget_mode not in ACTION_BUDGET_MODES:
        raise ValueError(f"unsupported action budget mode: {action_budget_mode}")
    if action_budget_mode in {QUERY_ADJUSTED_BUDGET, PRE_RUN_QUERY_ADJUSTED_BUDGET}:
        if sample_design_id is None:
            raise ValueError("query-adjusted action losses require --sample-design-id")
        sample_design = get_sample_design_spec(sample_design_id)
        resolved_sample_design_id = sample_design.sample_design_id
    else:
        if sample_design_id is not None:
            raise ValueError("behavior-only full-budget action losses must not define a sample design")
        resolved_sample_design_id = None
    _validate_timing_config(config)
    split = split_name(config)
    suite = str(config["suite"]).lower()
    portfolio = tuple(str(value) for value in algorithms(config))
    if portfolio != FROZEN_PORTFOLIO_ORDER:
        raise ValueError(
            "formal action generation requires portfolio order de,pso,cmaes,shade"
        )
    default_filter: str | None = None
    if action_budget_mode == PRE_RUN_QUERY_ADJUSTED_BUDGET:
        if all_prefixes or default_algorithm is not None:
            raise ValueError("pre-run action losses have no prefix algorithm filter")
    elif not all_prefixes:
        default_filter = str(default_algorithm or train_derived_sbs(train_config_path))
        if default_filter not in portfolio:
            raise ValueError(
                f"prefix filter algorithm {default_filter!r} is not in the configured portfolio"
            )
    if workers < 1:
        raise ValueError("workers must be at least 1")
    settings = OptimizerSettings(population_size=int(config["population_size"]), checkpoint_ratios=(1.0,))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_specs = make_shards(config, only_functions, only_dimensions)
    if not shard_specs:
        raise ValueError("no shards selected for action-loss generation")

    results: list[tuple[int, list[dict], int]] = []
    if workers == 1 or len(shard_specs) == 1:
        for index, shard in enumerate(shard_specs):
            result = _evaluate_action_loss_shard(
                shard_index=index,
                shard=shard,
                split=split,
                suite=suite,
                config=config,
                action_budget_mode=action_budget_mode,
                sample_design_id=resolved_sample_design_id,
                settings=settings,
                portfolio=portfolio,
                default_algorithm=default_filter,
                all_prefixes=all_prefixes,
                max_states=None if max_states is None else max_states,
            )
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_action_loss_shard,
                    shard_index=index,
                    shard=shard,
                    split=split,
                    suite=suite,
                    config=config,
                    action_budget_mode=action_budget_mode,
                    sample_design_id=resolved_sample_design_id,
                    settings=settings,
                    portfolio=portfolio,
                    default_algorithm=default_filter,
                    all_prefixes=all_prefixes,
                    max_states=None if max_states is None else max_states,
                ): index
                for index, shard in enumerate(shard_specs)
            }
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item[0])

    writer: pq.ParquetWriter | None = None
    state_count = 0
    action_count = 0
    try:
        for _, rows, used_states in results:
            if rows:
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path,
                        _pre_run_schema()
                        if action_budget_mode == PRE_RUN_QUERY_ADJUSTED_BUDGET
                        else _schema(),
                    )
                writer.write_table(
                    pa.Table.from_pylist(
                        rows,
                        schema=(
                            _pre_run_schema()
                            if action_budget_mode == PRE_RUN_QUERY_ADJUSTED_BUDGET
                            else _schema()
                        ),
                    )
                )
            state_count += used_states
            action_count += len(rows)
    except BaseException:
        if writer is not None:
            writer.close()
            writer = None
        if output_path.exists():
            output_path.unlink()
        raise
    finally:
        if writer is not None:
            writer.close()
    if state_count == 0:
        raise ValueError("no eligible shared states were evaluated")
    print(f"wrote {action_count} action-loss rows for {state_count} shared states to {output_path}")
    return {
        "states": state_count,
        "rows": action_count,
        "action_budget_mode": action_budget_mode,
        "sample_design_id": resolved_sample_design_id or NOT_APPLICABLE,
        "output": str(output_path),
    }


def evaluate_candidate_actions(
    *,
    checkpoint_state: OptimizerState,
    problem: Problem,
    portfolio: tuple[str, ...],
    fe_budget: int,
    seed: int,
    function: int,
    instance: int,
    checkpoint_fe: int,
    action_budget_mode: str,
    failure_loss_cap: float,
    log10_gap_floor: float = 1.0e-12,
    log10_gap_cap: float = 1.0e20,
    success_gap_target: float = 1.0e-8,
    prefix_first_hit_fe: int | None = None,
    action_timeout_seconds: float = 3600.0,
) -> list[dict[str, float | str | int]]:
    prefix_algorithm = str(checkpoint_state.algorithm)
    if len(portfolio) != 4 or len(set(portfolio)) != 4:
        raise ValueError("action portfolio must contain exactly four unique algorithms")
    if prefix_algorithm not in portfolio:
        raise ValueError("prefix algorithm must belong to the action portfolio")
    reference_value = float(problem.reference_value) if problem.reference_value is not None else None
    if reference_value is None:
        raise ValueError("benchmark reference value is required for gap-based loss evaluation")
    _validate_failure_loss_cap(failure_loss_cap)
    if action_budget_mode not in ACTION_BUDGET_MODES:
        raise ValueError(f"unsupported action budget mode: {action_budget_mode}")
    order_rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                ACTION_ORDER_STREAM_CODE,
                int(function),
                int(instance),
                int(problem.dimension),
                int(checkpoint_fe),
                ACTION_BUDGET_STREAM_CODES[action_budget_mode],
            ]
        )
    )
    ordered_algorithms = tuple(np.asarray(portfolio, dtype=object)[order_rng.permutation(len(portfolio))])
    outcomes: list[dict[str, float | str | int]] = []
    for execution_order, target_algorithm_value in enumerate(ordered_algorithms):
        target_algorithm = str(target_algorithm_value)
        outcome = _evaluate_one_candidate_action(
            checkpoint_state=checkpoint_state,
            problem=problem,
            target_algorithm=target_algorithm,
            fe_budget=fe_budget,
            seed=seed,
            function=function,
            instance=instance,
            action_budget_mode=action_budget_mode,
            failure_loss_cap=failure_loss_cap,
            success_gap_target=success_gap_target,
            action_start_fe=int(checkpoint_fe),
            prefix_first_hit_fe=prefix_first_hit_fe,
            action_timeout_seconds=action_timeout_seconds,
        )
        outcome["execution_order"] = int(execution_order)
        outcomes.append(outcome)
    _complete_action_diagnostics(
        outcomes,
        prefix_algorithm=prefix_algorithm,
        portfolio=portfolio,
        log10_gap_floor=log10_gap_floor,
        log10_gap_cap=log10_gap_cap,
    )
    return outcomes


def _evaluate_one_candidate_action(
    *,
    checkpoint_state: OptimizerState,
    problem: Problem,
    target_algorithm: str,
    fe_budget: int,
    seed: int,
    function: int,
    instance: int,
    action_budget_mode: str,
    failure_loss_cap: float,
    success_gap_target: float,
    action_start_fe: int,
    prefix_first_hit_fe: int | None,
    action_timeout_seconds: float,
) -> dict[str, float | str | int]:
    prefix_algorithm = str(checkpoint_state.algorithm)
    reference_value = float(problem.reference_value) if problem.reference_value is not None else None
    if reference_value is None:
        raise ValueError("benchmark reference value is required for gap-based loss evaluation")
    action_status = "ok"
    failure_type = ""
    failure_message = ""
    path_started = perf_counter()
    tracker = _TimedObjective(
        problem=problem,
        deadline=path_started + float(action_timeout_seconds),
        reference_value=reference_value,
        success_gap_target=success_gap_target,
        first_evaluation_fe=int(action_start_fe) + 1,
    )
    timed_problem = tracker.wrapped_problem()
    if target_algorithm == prefix_algorithm:
        action = "continue_current"
        state = clone_optimizer_state(checkpoint_state)
        transition_mode = "native_optimizer_state"
        runtime_handoff = 0.0
    else:
        action = target_algorithm
        handoff_started = perf_counter()
        try:
            state = initialize_transferred_optimizer_state(
                algorithm=target_algorithm,
                source_state=checkpoint_state,
                problem=timed_problem,
                seed=seed,
                function=function,
                instance=instance,
                event=(
                    QUERY_TRANSFER_EVENT
                    if action_budget_mode == QUERY_ADJUSTED_BUDGET
                    else NO_QUERY_TRANSFER_EVENT
                ),
            )
        except Exception as exc:
            state = None
            action_status = "failed"
            failure_type = type(exc).__name__
            failure_message = str(exc)[:500]
        runtime_handoff = perf_counter() - handoff_started
        transition_mode = "population_transfer_initialization"
    action_started = perf_counter()
    if state is None:
        raw_loss = float(reference_value + failure_loss_cap)
    else:
        try:
            result = advance_optimizer_state(
                state=state,
                problem=timed_problem,
                fe_budget=fe_budget,
            )
            raw_loss = float(result.best_fitness)
            if not np.isfinite(raw_loss):
                raise FloatingPointError("action continuation returned a non-finite best fitness")
        except Exception as exc:
            action_status = "failed"
            failure_type = type(exc).__name__
            failure_message = str(exc)[:500]
            raw_loss = float(reference_value + failure_loss_cap)
    runtime_action = float(perf_counter() - action_started)
    gap_loss = min(max(raw_loss - reference_value, 0.0), float(failure_loss_cap))
    continuation_first_hit_fe = tracker.first_hit_fe
    path_completed = action_status == "ok"
    observed_first_hit_fe = (
        int(prefix_first_hit_fe)
        if prefix_first_hit_fe is not None
        else continuation_first_hit_fe
    )
    target_hit_observed = observed_first_hit_fe is not None
    target_hit_before_failure = bool(target_hit_observed and not path_completed)
    endpoint_success = bool(target_hit_observed and path_completed)
    return {
        "action": action,
        "target_algorithm": target_algorithm,
        "transition_mode": transition_mode,
        "action_status": action_status,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "action_loss": float(gap_loss),
        "loss_gap_raw": float(gap_loss),
        "loss_gap_norm": float(gap_loss),
        "runtime_handoff": float(runtime_handoff),
        "runtime_action_optimization": runtime_action,
        "prefix_first_hit_FE": prefix_first_hit_fe,
        "continuation_first_hit_FE": continuation_first_hit_fe,
        "observed_first_hit_FE": observed_first_hit_fe,
        "target_hit_observed": target_hit_observed,
        "target_hit_before_failure": target_hit_before_failure,
        "endpoint_success": endpoint_success,
        # Compatibility aliases retain the formal ERT semantics.
        "first_hit_FE": observed_first_hit_fe,
        "success": target_hit_observed,
        "planned_FE": int(fe_budget),
        "effective_FE": int(tracker.evaluations),
        "success_gap_target": float(success_gap_target),
        "timed_out": bool(tracker.timed_out),
        "path_completed": bool(path_completed),
    }


def _evaluate_native_skip(
    *,
    checkpoint_state: OptimizerState,
    problem: Problem,
    fe_budget: int,
    failure_loss_cap: float,
    success_gap_target: float,
    checkpoint_fe: int,
    prefix_first_hit_fe: int | None,
    action_timeout_seconds: float,
) -> dict[str, float | str]:
    reference_value = float(problem.reference_value) if problem.reference_value is not None else None
    if reference_value is None:
        raise ValueError("benchmark reference value is required for gap-based skip evaluation")
    _validate_failure_loss_cap(failure_loss_cap)
    started = perf_counter()
    tracker = _TimedObjective(
        problem=problem,
        deadline=started + float(action_timeout_seconds),
        reference_value=reference_value,
        success_gap_target=success_gap_target,
        first_evaluation_fe=int(checkpoint_fe) + 1,
    )
    timed_problem = tracker.wrapped_problem()
    status = "ok"
    failure_type = ""
    failure_message = ""
    try:
        result = advance_optimizer_state(
            state=clone_optimizer_state(checkpoint_state),
            problem=timed_problem,
            fe_budget=fe_budget,
        )
        raw_loss = float(result.best_fitness)
        if not np.isfinite(raw_loss):
            raise FloatingPointError("native skip continuation returned a non-finite best fitness")
    except Exception as exc:
        status = "failed"
        failure_type = type(exc).__name__
        failure_message = str(exc)[:500]
        raw_loss = float(reference_value + failure_loss_cap)
    gap_loss = min(max(raw_loss - reference_value, 0.0), float(failure_loss_cap))
    continuation_first_hit_fe = tracker.first_hit_fe
    path_completed = status == "ok"
    observed_first_hit_fe = (
        int(prefix_first_hit_fe)
        if prefix_first_hit_fe is not None
        else continuation_first_hit_fe
    )
    target_hit_observed = observed_first_hit_fe is not None
    target_hit_before_failure = bool(target_hit_observed and not path_completed)
    endpoint_success = bool(target_hit_observed and path_completed)
    return {
        "p_skip_raw": raw_loss,
        "p_skip": float(gap_loss),
        "loss_skip": float(gap_loss),
        "runtime_no_query_optimization": float(perf_counter() - started),
        "skip_status": status,
        "skip_failure_type": failure_type,
        "skip_failure_message": failure_message,
        "skip_prefix_first_hit_FE": prefix_first_hit_fe,
        "skip_continuation_first_hit_FE": continuation_first_hit_fe,
        "skip_observed_first_hit_FE": observed_first_hit_fe,
        "skip_target_hit_observed": target_hit_observed,
        "skip_target_hit_before_failure": target_hit_before_failure,
        "skip_endpoint_success": endpoint_success,
        # Compatibility aliases retain the formal ERT semantics.
        "skip_first_hit_FE": observed_first_hit_fe,
        "skip_success": target_hit_observed,
        "skip_planned_FE": int(fe_budget),
        "skip_effective_FE": int(tracker.evaluations),
        "success_gap_target": float(success_gap_target),
        "skip_timed_out": bool(tracker.timed_out),
        "skip_path_completed": bool(path_completed),
    }


def _evaluate_state_action_outcomes_once(
    *,
    checkpoint_state: OptimizerState,
    problem: Problem,
    portfolio: tuple[str, ...],
    action_budget: int,
    skip_budget: int,
    seed: int,
    function: int,
    instance: int,
    checkpoint_fe: int,
    action_budget_mode: str,
    failure_loss_cap: float,
    log10_gap_floor: float,
    log10_gap_cap: float,
    success_gap_target: float,
    action_start_fe: int,
    prefix_first_hit_fe: int | None,
    action_timeout_seconds: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Execute Skip and each candidate continuation exactly once.

    These executions provide the observed action-loss matrix.  Their runtimes
    are diagnostic only and must not be substituted for the later three-repeat
    selected complete-policy replay used by Utility.
    """
    path_labels = ("skip", *portfolio)
    order_rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                ACTION_ORDER_STREAM_CODE + 1,
                int(function),
                int(instance),
                int(problem.dimension),
                int(checkpoint_fe),
                ACTION_BUDGET_STREAM_CODES[action_budget_mode],
            ]
        )
    )
    base_order = tuple(
        str(value)
        for value in np.asarray(path_labels, dtype=object)[order_rng.permutation(len(path_labels))]
    )
    action_repeats: dict[str, list[dict[str, object]]] = {
        algorithm: [] for algorithm in portfolio
    }
    skip_repeats: list[dict[str, object]] = []
    for repetition_index in range(TIMING_REPETITIONS):
        rotation = repetition_index % len(base_order)
        repeated_order = base_order[rotation:] + base_order[:rotation]
        for order_position, path_label in enumerate(repeated_order):
            if path_label == "skip":
                record = _evaluate_native_skip(
                    checkpoint_state=checkpoint_state,
                    problem=problem,
                    fe_budget=skip_budget,
                    failure_loss_cap=failure_loss_cap,
                    success_gap_target=success_gap_target,
                    checkpoint_fe=checkpoint_fe,
                    prefix_first_hit_fe=prefix_first_hit_fe,
                    action_timeout_seconds=action_timeout_seconds,
                )
                record.update(
                    {
                        "timing_repetition_index": int(repetition_index),
                        "timing_order_position": int(order_position),
                    }
                )
                skip_repeats.append(record)
                continue
            record = _evaluate_one_candidate_action(
                checkpoint_state=checkpoint_state,
                problem=problem,
                target_algorithm=path_label,
                fe_budget=action_budget,
                seed=seed,
                function=function,
                instance=instance,
                action_budget_mode=action_budget_mode,
                failure_loss_cap=failure_loss_cap,
                success_gap_target=success_gap_target,
                action_start_fe=action_start_fe,
                prefix_first_hit_fe=prefix_first_hit_fe,
                action_timeout_seconds=action_timeout_seconds,
            )
            record.update(
                {
                    "timing_repetition_index": int(repetition_index),
                    "timing_order_position": int(order_position),
                }
            )
            action_repeats[path_label].append(record)
    outcomes = [
        _collapse_single_action_outcome(action_repeats[algorithm])
        for algorithm in portfolio
    ]
    _complete_action_diagnostics(
        outcomes,
        prefix_algorithm=str(checkpoint_state.algorithm),
        portfolio=portfolio,
        log10_gap_floor=log10_gap_floor,
        log10_gap_cap=log10_gap_cap,
    )
    return outcomes, _collapse_single_skip_outcome(skip_repeats)


def _collapse_single_action_outcome(records: list[dict[str, object]]) -> dict[str, object]:
    if len(records) != TIMING_REPETITIONS:
        raise RuntimeError("candidate action outcome must be executed exactly once")
    invariant_fields = (
        "action",
        "target_algorithm",
        "transition_mode",
        "action_status",
        "failure_type",
        "failure_message",
        "action_loss",
        "action_loss_raw",
        "loss_gap_raw",
        "prefix_first_hit_FE",
        "continuation_first_hit_FE",
        "observed_first_hit_FE",
        "target_hit_observed",
        "target_hit_before_failure",
        "endpoint_success",
        "first_hit_FE",
        "success",
        "planned_FE",
        "effective_FE",
        "success_gap_target",
        "timed_out",
        "path_completed",
    )
    first = records[0]
    for record in records[1:]:
        for field in invariant_fields:
            if record[field] != first[field]:
                raise RuntimeError(
                    f"candidate action outcome changed across fixed-state repetitions: {field}"
                )
    handoff_repeats = [float(record["runtime_handoff"]) for record in records]
    optimization_repeats = [float(record["runtime_action_optimization"]) for record in records]
    fresh_initialization_repeats = (
        [float(record["runtime_fresh_initialization"]) for record in records]
        if "runtime_fresh_initialization" in first
        else None
    )
    order_positions = [int(record["timing_order_position"]) for record in records]
    output = dict(first)
    output.update(
        {
            "execution_order": order_positions[0],
            "execution_order_repetitions": order_positions,
            "timing_repetition_indices": list(range(TIMING_REPETITIONS)),
            "runtime_handoff_repetitions": handoff_repeats,
            "runtime_action_optimization_repetitions": optimization_repeats,
            "runtime_handoff": float(np.median(handoff_repeats)),
            "runtime_action_optimization": float(np.median(optimization_repeats)),
            "timing_repetitions": TIMING_REPETITIONS,
            "timing_order_protocol": EXECUTION_ORDER_PROTOCOL,
        }
    )
    if fresh_initialization_repeats is not None:
        output["runtime_fresh_initialization_repetitions"] = (
            fresh_initialization_repeats
        )
        output["runtime_fresh_initialization"] = float(
            np.median(fresh_initialization_repeats)
        )
    output.pop("timing_repetition_index", None)
    output.pop("timing_order_position", None)
    return output


def _collapse_single_skip_outcome(records: list[dict[str, object]]) -> dict[str, object]:
    if len(records) != TIMING_REPETITIONS:
        raise RuntimeError("native Skip outcome must be executed exactly once")
    invariant_fields = (
        "p_skip_raw",
        "p_skip",
        "loss_skip",
        "skip_status",
        "skip_failure_type",
        "skip_failure_message",
        "skip_prefix_first_hit_FE",
        "skip_continuation_first_hit_FE",
        "skip_observed_first_hit_FE",
        "skip_target_hit_observed",
        "skip_target_hit_before_failure",
        "skip_endpoint_success",
        "skip_first_hit_FE",
        "skip_success",
        "skip_planned_FE",
        "skip_effective_FE",
        "success_gap_target",
        "skip_timed_out",
        "skip_path_completed",
    )
    first = records[0]
    for record in records[1:]:
        for field in invariant_fields:
            if record[field] != first[field]:
                raise RuntimeError(
                    f"native Skip outcome changed across fixed-state repetitions: {field}"
                )
    runtime_repeats = [float(record["runtime_no_query_optimization"]) for record in records]
    order_positions = [int(record["timing_order_position"]) for record in records]
    output = dict(first)
    output.update(
        {
            "runtime_no_query_optimization": float(np.median(runtime_repeats)),
            "runtime_no_query_optimization_repetitions": runtime_repeats,
            "runtime_no_query_handoff_repetitions": [0.0] * TIMING_REPETITIONS,
            "skip_execution_order": order_positions[0],
            "skip_execution_order_repetitions": order_positions,
            "timing_repetition_indices": list(range(TIMING_REPETITIONS)),
            "timing_repetitions": TIMING_REPETITIONS,
            "timing_order_protocol": EXECUTION_ORDER_PROTOCOL,
        }
    )
    output.pop("timing_repetition_index", None)
    output.pop("timing_order_position", None)
    return output


def _validate_timing_config(config: dict) -> None:
    repetitions = int(config.get("timing_repetitions", -1))
    protocol = str(config.get("timing_order_protocol", ""))
    if repetitions != COMPLETE_PATH_TIMING_REPETITIONS:
        raise ValueError(
            "formal selected complete-path replay requires "
            f"timing_repetitions={COMPLETE_PATH_TIMING_REPETITIONS}"
        )
    if protocol != COMPLETE_PATH_TIMING_ORDER_PROTOCOL:
        raise ValueError(
            "formal selected complete-path replay requires timing_order_protocol="
            f"{COMPLETE_PATH_TIMING_ORDER_PROTOCOL}"
        )


def _validate_failure_loss_cap(value: float) -> None:
    if not np.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError("failure_loss_cap must be finite and positive")


def _selector_log10_bounds(config: dict) -> tuple[float, float]:
    floor = float(config.get("log10_gap_floor", np.nan))
    cap = float(config.get("log10_gap_cap", np.nan))
    if not np.isfinite(floor) or not np.isfinite(cap) or not 0.0 < floor < cap:
        raise ValueError("log10 gap bounds must satisfy 0 < floor < cap")
    return floor, cap


def _evaluate_action_loss_shard(
    *,
    shard_index: int,
    shard,
    split: str,
    suite: str,
    config: dict,
    action_budget_mode: str,
    sample_design_id: str | None,
    settings: OptimizerSettings,
    portfolio: tuple[str, ...],
    default_algorithm: str | None,
    all_prefixes: bool,
    max_states: int | None,
) -> tuple[int, list[dict], int]:
    if action_budget_mode == PRE_RUN_QUERY_ADJUSTED_BUDGET:
        rows, used_states = _evaluate_pre_run_shard(
            split=split,
            suite=suite,
            config=config,
            function=shard.function,
            dimension=shard.dimension,
            sample_design_id=str(sample_design_id),
            settings=settings,
            portfolio=portfolio,
            max_states=max_states,
        )
    else:
        require_complete_shard_outputs(shard)
        trajectory_rows = pq.read_table(shard.output_path).to_pylist()
        final_performance_rows = pq.read_table(shard.final_performance_path).to_pylist()
        rows, used_states = _evaluate_shard_rows(
            split=split,
            suite=suite,
            config=config,
            trajectory_rows=trajectory_rows,
            final_performance_rows=final_performance_rows,
            sample_design_id=sample_design_id,
            action_budget_mode=action_budget_mode,
            settings=settings,
            portfolio=portfolio,
            default_algorithm=default_algorithm,
            all_prefixes=all_prefixes,
            max_states=max_states,
        )
    return shard_index, rows, used_states


def _evaluate_shard_rows(
    *,
    split: str,
    suite: str,
    config: dict,
    trajectory_rows: list[dict],
    final_performance_rows: list[dict],
    sample_design_id: str | None,
    action_budget_mode: str,
    settings: OptimizerSettings,
    portfolio: tuple[str, ...],
    default_algorithm: str | None,
    all_prefixes: bool,
    max_states: int | None,
) -> tuple[list[dict], int]:
    if max_states is not None and max_states <= 0:
        return [], 0
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for trajectory_row in trajectory_rows:
        key = (
            str(trajectory_row["problem_id"]),
            str(trajectory_row["algorithm"]),
            int(trajectory_row["seed"]),
        )
        grouped[key].append(trajectory_row)
    final_by_run: dict[tuple[str, str, int], dict] = {}
    for row in final_performance_rows:
        key = (str(row["problem_id"]), str(row["algorithm"]), int(row["seed"]))
        if key in final_by_run:
            raise ValueError("final performance contains duplicate problem/algorithm/seed rows")
        final_by_run[key] = row
    missing_final = sorted(set(grouped).difference(final_by_run))
    if missing_final:
        raise ValueError(
            "trajectory runs are missing final-performance attempted rows: "
            f"{missing_final[:8]}"
        )

    output_rows = []
    used_states = 0
    sample_design = (
        get_sample_design_spec(sample_design_id)
        if action_budget_mode == QUERY_ADJUSTED_BUDGET and sample_design_id is not None
        else None
    )
    failure_loss_cap = float(config.get("failure_loss_cap", np.nan))
    _validate_failure_loss_cap(failure_loss_cap)
    log10_gap_floor, log10_gap_cap = _selector_log10_bounds(config)
    success_gap_target = float(config.get("success_gap_target", np.nan))
    if not np.isfinite(success_gap_target) or success_gap_target <= 0.0:
        raise ValueError("success_gap_target must be finite and positive")
    action_timeout_seconds = float(config.get("action_timeout_seconds", np.nan))
    if not np.isfinite(action_timeout_seconds) or action_timeout_seconds <= 0.0:
        raise ValueError("action_timeout_seconds must be finite and positive")
    for (problem_id, prefix_algorithm, seed), trajectory_group in grouped.items():
        if not all_prefixes and prefix_algorithm != default_algorithm:
            continue
        function, instance, dimension = _parse_problem_id(problem_id, suite=suite)
        problem = make_problem({"suite": suite, "function": function, "instance": instance, "dimension": dimension})
        final_performance_row = final_by_run[(problem_id, prefix_algorithm, seed)]
        if not bool(final_performance_row.get("path_completed", True)):
            continue
        if "first_hit_FE" not in final_performance_row or "success" not in final_performance_row:
            raise ValueError("final performance is missing per-evaluation first-hit metadata")
        final_first_hit_fe = (
            None if final_performance_row["first_hit_FE"] is None else int(final_performance_row["first_hit_FE"])
        )
        if bool(final_performance_row["success"]) != (final_first_hit_fe is not None):
            raise ValueError("final-performance success must equal first_hit_FE is not null")
        reference_value = problem.reference_value
        if reference_value is None:
            raise ValueError(f"benchmark reference value is unavailable for {problem.problem_id}")
        state = initialize_optimizer_state(algorithm=prefix_algorithm, problem=problem, seed=seed, settings=settings)
        for trajectory_row in sorted(trajectory_group, key=lambda row: int(row["FE"])):
            checkpoint_fe = int(trajectory_row["FE"])
            delta = checkpoint_fe - int(state.evaluations)
            if delta < 0:
                raise ValueError(f"trajectory checkpoint FE moved backwards for {problem_id} {prefix_algorithm} seed={seed}")
            advance_optimizer_state(state=state, problem=problem, fe_budget=delta)
            _validate_replayed_checkpoint(state, trajectory_row)
            fe_total = fe_total_for_dimension(config, dimension)
            fe_query = sample_design.sample_size(dimension) if sample_design is not None else 0
            if not _eligible_for_action_loss(trajectory_row, config, fe_query=fe_query):
                continue
            prefix_first_hit_fe = (
                final_first_hit_fe if final_first_hit_fe is not None and final_first_hit_fe <= checkpoint_fe else None
            )
            if prefix_first_hit_fe is not None and not 0 < prefix_first_hit_fe <= checkpoint_fe:
                raise ValueError("prefix first_hit_FE must not follow the shared checkpoint")
            actual_fe_ratio = float(checkpoint_fe / fe_total)
            if float(trajectory_row["FE_ratio"]) != actual_fe_ratio:
                raise ValueError("trajectory FE_ratio must equal the actual FE / FE_total")
            missing_sampling = set(SAMPLING_METADATA_COLUMNS).difference(trajectory_row)
            if missing_sampling:
                raise ValueError(
                    "trajectory row is missing dynamic-sampling metadata: "
                    f"{sorted(missing_sampling)}"
                )
            if action_budget_mode == QUERY_ADJUSTED_BUDGET and (fe_query <= 0 or fe_query >= fe_total):
                raise ValueError(
                    "query sample budget must be positive and smaller than FE_total: "
                    f"sample_design_id={sample_design_id}, FE_query={fe_query}, FE_total={fe_total}"
                )
            action_budget = fe_total - checkpoint_fe - fe_query
            skip_budget = fe_total - checkpoint_fe
            if action_budget <= 0 or skip_budget <= 0:
                raise ValueError("eligible action-loss state must retain positive continuation budgets")
            outcomes, skip_outcome = _evaluate_state_action_outcomes_once(
                checkpoint_state=state,
                problem=problem,
                portfolio=portfolio,
                action_budget=action_budget,
                skip_budget=skip_budget,
                seed=seed,
                function=function,
                instance=instance,
                checkpoint_fe=checkpoint_fe,
                action_budget_mode=action_budget_mode,
                failure_loss_cap=failure_loss_cap,
                log10_gap_floor=log10_gap_floor,
                log10_gap_cap=log10_gap_cap,
                success_gap_target=success_gap_target,
                action_start_fe=checkpoint_fe + fe_query,
                prefix_first_hit_fe=prefix_first_hit_fe,
                action_timeout_seconds=action_timeout_seconds,
            )
            common = {
                "split": split,
                "problem_id": problem_id,
                "function_id": str(trajectory_row["function_id"]),
                "family": str(trajectory_row["family"]),
                "cv_group_id": str(trajectory_row.get("cv_group_id", trajectory_row["function_id"])),
                "dimension": dimension,
                "prefix_algorithm": prefix_algorithm,
                "prefix_scope": "all_portfolio" if all_prefixes else "filtered_diagnostic",
                "no_query_algorithm": prefix_algorithm,
                "seed": seed,
                "FE": checkpoint_fe,
                "FE_prefix": checkpoint_fe,
                "FE_ratio": actual_fe_ratio,
                "FE_total": fe_total,
                **{column: trajectory_row[column] for column in SAMPLING_METADATA_COLUMNS},
                "action_budget_mode": action_budget_mode,
                "sample_design_id": sample_design.sample_design_id if sample_design is not None else NOT_APPLICABLE,
                "sample_design_protocol": sample_design.protocol if sample_design is not None else NOT_APPLICABLE,
                "FE_query": fe_query,
                "FE_no_query_optimization": skip_budget,
                "FE_action_optimization": action_budget,
                "remaining_budget_ratio": float(action_budget / fe_total),
                "performance_value_mode": "raw_objective",
                "performance_loss_mode": "known_optimum_gap",
                "benchmark_reference_value": float(reference_value),
                **skip_outcome,
                "runtime_no_query_handoff": 0.0,
                "no_query_transition_mode": "native_optimizer_state",
                "failure_loss_cap": failure_loss_cap,
                "log10_gap_floor": log10_gap_floor,
                "log10_gap_cap": log10_gap_cap,
                "execution_order_protocol": EXECUTION_ORDER_PROTOCOL,
                "action_outcome_execution_count": ACTION_OUTCOME_EXECUTIONS,
                "action_runtime_role": "diagnostic_not_utility",
                "action_loss_protocol": ACTION_LOSS_PROTOCOL,
            }
            output_rows.extend({**common, **outcome} for outcome in outcomes)
            used_states += 1
            if max_states is not None and used_states >= max_states:
                return output_rows, used_states
    return output_rows, used_states


def _evaluate_pre_run_shard(
    *,
    split: str,
    suite: str,
    config: dict,
    function: int,
    dimension: int,
    sample_design_id: str,
    settings: OptimizerSettings,
    portfolio: tuple[str, ...],
    max_states: int | None,
) -> tuple[list[dict], int]:
    if max_states is not None and max_states <= 0:
        return [], 0
    sample_design = get_sample_design_spec(sample_design_id)
    fe_total = fe_total_for_dimension(config, dimension)
    fe_query = sample_design.sample_size(dimension)
    action_budget = fe_total - fe_query
    if action_budget <= int(settings.population_size):
        raise ValueError("pre-run query-adjusted budget cannot initialize the optimizer portfolio")
    failure_loss_cap = float(config.get("failure_loss_cap", np.nan))
    _validate_failure_loss_cap(failure_loss_cap)
    log10_gap_floor, log10_gap_cap = _selector_log10_bounds(config)
    success_gap_target = float(config.get("success_gap_target", np.nan))
    if not np.isfinite(success_gap_target) or success_gap_target <= 0.0:
        raise ValueError("success_gap_target must be finite and positive")
    action_timeout_seconds = float(config.get("action_timeout_seconds", np.nan))
    if not np.isfinite(action_timeout_seconds) or action_timeout_seconds <= 0.0:
        raise ValueError("action_timeout_seconds must be finite and positive")
    output_rows: list[dict] = []
    used_states = 0
    for instance in as_int_list(config, "instances"):
        problem = make_problem(
            {
                "suite": suite,
                "function": int(function),
                "instance": int(instance),
                "dimension": int(dimension),
            }
        )
        try:
            reference_value = problem.reference_value
            if reference_value is None:
                raise ValueError(f"benchmark reference value is unavailable for {problem.problem_id}")
            for seed in as_int_list(config, "seeds"):
                outcomes = _evaluate_pre_run_action_outcomes_once(
                    problem=problem,
                    portfolio=portfolio,
                    action_budget=action_budget,
                    settings=settings,
                    seed=int(seed),
                    function=int(function),
                    instance=int(instance),
                    failure_loss_cap=failure_loss_cap,
                    log10_gap_floor=log10_gap_floor,
                    log10_gap_cap=log10_gap_cap,
                    success_gap_target=success_gap_target,
                    action_start_fe=fe_query,
                    action_timeout_seconds=action_timeout_seconds,
                )
                common = {
                    "split": split,
                    "problem_id": problem.problem_id,
                    "function_id": function_id_name(suite, int(function)),
                    "family": landscape_family_name(suite, int(function)),
                    "cv_group_id": problem.cv_group_id,
                    "dimension": int(dimension),
                    "seed": int(seed),
                    "FE": 0,
                    "FE_prefix": 0,
                    "FE_ratio": 0.0,
                    "FE_total": int(fe_total),
                    "action_budget_mode": PRE_RUN_QUERY_ADJUSTED_BUDGET,
                    "sample_design_id": sample_design.sample_design_id,
                    "sample_design_protocol": sample_design.protocol,
                    "FE_query": int(fe_query),
                    "FE_action_optimization": int(action_budget),
                    "remaining_budget_ratio": float(action_budget / fe_total),
                    "performance_value_mode": "raw_objective",
                    "performance_loss_mode": "known_optimum_gap",
                    "benchmark_reference_value": float(reference_value),
                    "failure_loss_cap": float(failure_loss_cap),
                    "log10_gap_floor": log10_gap_floor,
                    "log10_gap_cap": log10_gap_cap,
                    "execution_order_protocol": EXECUTION_ORDER_PROTOCOL,
                    "action_outcome_execution_count": ACTION_OUTCOME_EXECUTIONS,
                    "action_runtime_role": "diagnostic_not_utility",
                    "action_loss_protocol": "pre_run_observed_algorithm_loss",
                }
                output_rows.extend({**common, **outcome} for outcome in outcomes)
                used_states += 1
                if max_states is not None and used_states >= max_states:
                    return output_rows, used_states
        finally:
            problem.close()
    return output_rows, used_states


def _evaluate_pre_run_action_outcomes_once(
    *,
    problem: Problem,
    portfolio: tuple[str, ...],
    action_budget: int,
    settings: OptimizerSettings,
    seed: int,
    function: int,
    instance: int,
    failure_loss_cap: float,
    log10_gap_floor: float,
    log10_gap_cap: float,
    success_gap_target: float,
    action_start_fe: int,
    action_timeout_seconds: float,
) -> list[dict[str, object]]:
    order_rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                ACTION_ORDER_STREAM_CODE + 2,
                int(function),
                int(instance),
                int(problem.dimension),
                ACTION_BUDGET_STREAM_CODES[PRE_RUN_QUERY_ADJUSTED_BUDGET],
            ]
        )
    )
    base_order = tuple(
        str(value)
        for value in np.asarray(portfolio, dtype=object)[order_rng.permutation(len(portfolio))]
    )
    repeated: dict[str, list[dict[str, object]]] = {algorithm: [] for algorithm in portfolio}
    for repetition_index in range(TIMING_REPETITIONS):
        rotation = repetition_index % len(base_order)
        order = base_order[rotation:] + base_order[:rotation]
        for order_position, algorithm in enumerate(order):
            record = _evaluate_one_pre_run_action(
                problem=problem,
                algorithm=algorithm,
                action_budget=action_budget,
                settings=settings,
                seed=seed,
                failure_loss_cap=failure_loss_cap,
                success_gap_target=success_gap_target,
                action_start_fe=action_start_fe,
                action_timeout_seconds=action_timeout_seconds,
            )
            record.update(
                {
                    "timing_repetition_index": repetition_index,
                    "timing_order_position": order_position,
                }
            )
            repeated[algorithm].append(record)
    outcomes = [_collapse_single_action_outcome(repeated[algorithm]) for algorithm in portfolio]
    _complete_pre_run_diagnostics(
        outcomes,
        portfolio=portfolio,
        log10_gap_floor=log10_gap_floor,
        log10_gap_cap=log10_gap_cap,
    )
    return outcomes


def _evaluate_one_pre_run_action(
    *,
    problem: Problem,
    algorithm: str,
    action_budget: int,
    settings: OptimizerSettings,
    seed: int,
    failure_loss_cap: float,
    success_gap_target: float,
    action_start_fe: int,
    action_timeout_seconds: float,
) -> dict[str, object]:
    reference_value = float(problem.reference_value) if problem.reference_value is not None else None
    if reference_value is None:
        raise ValueError("benchmark reference value is required for pre-run loss evaluation")
    status = "ok"
    failure_type = ""
    failure_message = ""
    path_started = perf_counter()
    tracker = _TimedObjective(
        problem=problem,
        deadline=path_started + float(action_timeout_seconds),
        reference_value=reference_value,
        success_gap_target=success_gap_target,
        first_evaluation_fe=int(action_start_fe) + 1,
    )
    timed_problem = tracker.wrapped_problem()
    initialization_started = path_started
    try:
        state = initialize_optimizer_state(
            algorithm=algorithm,
            problem=timed_problem,
            seed=seed,
            settings=settings,
        )
    except Exception as exc:
        state = None
        status = "failed"
        failure_type = type(exc).__name__
        failure_message = str(exc)[:500]
    runtime_initialization = float(perf_counter() - initialization_started)
    optimization_started = perf_counter()
    if state is None:
        raw_loss = float(reference_value + failure_loss_cap)
    else:
        try:
            remaining = int(action_budget) - int(state.evaluations)
            if remaining < 0:
                raise ValueError("pre-run action budget is smaller than native initialization")
            result = advance_optimizer_state(
                state=state,
                problem=timed_problem,
                fe_budget=remaining,
            )
            raw_loss = float(result.best_fitness)
            if not np.isfinite(raw_loss):
                raise FloatingPointError("pre-run action returned a non-finite best fitness")
        except Exception as exc:
            status = "failed"
            failure_type = type(exc).__name__
            failure_message = str(exc)[:500]
            raw_loss = float(reference_value + failure_loss_cap)
    runtime_optimization = float(perf_counter() - optimization_started)
    gap_loss = min(max(raw_loss - reference_value, 0.0), float(failure_loss_cap))
    path_completed = status == "ok"
    observed_first_hit_fe = tracker.first_hit_fe
    target_hit_observed = observed_first_hit_fe is not None
    target_hit_before_failure = bool(target_hit_observed and not path_completed)
    endpoint_success = bool(target_hit_observed and path_completed)
    return {
        "action": algorithm,
        "target_algorithm": algorithm,
        "transition_mode": "fresh_optimizer_initialization",
        "action_status": status,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "action_loss": float(gap_loss),
        "action_loss_raw": float(raw_loss),
        "loss_gap_raw": float(gap_loss),
        "loss_gap_norm": float(gap_loss),
        "runtime_handoff": 0.0,
        "runtime_fresh_initialization": runtime_initialization,
        "runtime_action_optimization": runtime_optimization,
        "prefix_first_hit_FE": None,
        "continuation_first_hit_FE": observed_first_hit_fe,
        "observed_first_hit_FE": observed_first_hit_fe,
        "target_hit_observed": target_hit_observed,
        "target_hit_before_failure": target_hit_before_failure,
        "endpoint_success": endpoint_success,
        # Compatibility aliases retain the formal ERT semantics.
        "first_hit_FE": observed_first_hit_fe,
        "success": target_hit_observed,
        "planned_FE": int(action_budget),
        "effective_FE": int(tracker.evaluations),
        "success_gap_target": float(success_gap_target),
        "timed_out": bool(tracker.timed_out),
        "path_completed": bool(path_completed),
    }


def _complete_pre_run_diagnostics(
    outcomes: list[dict[str, object]],
    *,
    portfolio: tuple[str, ...],
    log10_gap_floor: float,
    log10_gap_cap: float,
) -> None:
    targets = [str(row["target_algorithm"]) for row in outcomes]
    if len(outcomes) != len(portfolio) or set(targets) != set(portfolio):
        raise ValueError("pre-run outcomes require exactly one fresh run per portfolio algorithm")
    portfolio_index = {algorithm: index for index, algorithm in enumerate(portfolio)}
    ordered = sorted(
        outcomes,
        key=lambda row: (
            float(row["action_loss"]),
            portfolio_index[str(row["target_algorithm"])],
        ),
    )
    best_algorithm = str(ordered[0]["target_algorithm"])
    best_loss = float(ordered[0]["action_loss"])
    worst_loss = max(float(row["action_loss"]) for row in outcomes)
    scale = max(worst_loss - best_loss, EPS)
    for row in outcomes:
        row["best_observed_algorithm"] = best_algorithm
        row["best_observed_loss"] = best_loss
        row["loss_gap_norm"] = float((float(row["action_loss"]) - best_loss) / scale)
        row["log10_action_loss"] = _clipped_log10_loss(
            float(row["action_loss"]),
            floor=log10_gap_floor,
            cap=log10_gap_cap,
        )
        row["selector_target_loss"] = float(row["log10_action_loss"])


def _complete_action_diagnostics(
    outcomes: list[dict[str, float | str]],
    *,
    prefix_algorithm: str,
    portfolio: tuple[str, ...],
    log10_gap_floor: float,
    log10_gap_cap: float,
) -> None:
    targets = [str(row["target_algorithm"]) for row in outcomes]
    if len(outcomes) != len(portfolio) or set(targets) != set(portfolio):
        raise ValueError("each shared state must contain exactly one outcome per portfolio algorithm")
    native = [row for row in outcomes if row["transition_mode"] == "native_optimizer_state"]
    if len(native) != 1 or native[0]["target_algorithm"] != prefix_algorithm:
        raise ValueError("each shared state must contain exactly one native continue-current action")
    portfolio_index = {algorithm: index for index, algorithm in enumerate(portfolio)}
    ordered = sorted(
        outcomes,
        key=lambda row: (
            float(row["action_loss"]),
            portfolio_index[str(row["target_algorithm"])],
        ),
    )
    best_algorithm = str(ordered[0]["target_algorithm"])
    best_loss = float(ordered[0]["action_loss"])
    worst_loss = max(float(row["action_loss"]) for row in outcomes)
    scale = max(worst_loss - best_loss, EPS)
    continue_log_loss = _clipped_log10_loss(
        float(native[0]["action_loss"]),
        floor=log10_gap_floor,
        cap=log10_gap_cap,
    )
    for row in outcomes:
        row["best_observed_algorithm"] = best_algorithm
        row["best_observed_loss"] = best_loss
        row["loss_gap_norm"] = float((float(row["action_loss"]) - best_loss) / scale)
        row["log10_action_loss"] = _clipped_log10_loss(
            float(row["action_loss"]),
            floor=log10_gap_floor,
            cap=log10_gap_cap,
        )
        row["selector_target_loss"] = float(
            float(row["log10_action_loss"]) - continue_log_loss
        )


def _clipped_log10_loss(value: float, *, floor: float, cap: float) -> float:
    if not np.isfinite(float(floor)) or not np.isfinite(float(cap)) or not 0.0 < float(floor) < float(cap):
        raise ValueError("selector log10 gap bounds must satisfy 0 < floor < cap")
    if not np.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError("selector action loss must be finite and non-negative")
    return float(np.log10(np.clip(float(value), float(floor), float(cap))))


def _eligible_for_action_loss(row: dict, config: dict, *, fe_query: int) -> bool:
    fe_total = fe_total_for_dimension(config, int(row["dimension"]))
    fe_prefix = int(row["FE"])
    ratio = float(row["FE_ratio"])
    return ratio >= MIN_LABEL_RATIO and ratio < 1.0 and fe_prefix + fe_query < fe_total


def _validate_replayed_checkpoint(state: OptimizerState, trajectory_row: dict) -> None:
    row_pop = np.asarray(trajectory_row["population"], dtype=float)
    row_fit = np.asarray(trajectory_row["fitness"], dtype=float)
    if state.population.shape != row_pop.shape or state.fitness.shape != row_fit.shape:
        raise ValueError("trajectory population/fitness shape does not match replayed optimizer state")
    if not np.allclose(state.population, row_pop, rtol=1e-4, atol=5e-2, equal_nan=True):
        max_diff = float(np.max(np.abs(state.population - row_pop)))
        raise ValueError(
            f"trajectory population does not match replayed native optimizer state "
            f"(max diff {max_diff:.2e}); regenerate trajectories"
        )
    if not np.allclose(state.fitness, row_fit, rtol=1e-4, atol=5e-2, equal_nan=True):
        max_diff = float(np.max(np.abs(state.fitness - row_fit)))
        raise ValueError(
            f"trajectory fitness does not match replayed native optimizer state "
            f"(max diff {max_diff:.2e}); regenerate trajectories"
        )
    if not np.isclose(float(state.best_fitness), float(trajectory_row["best_fitness"]), rtol=1e-4, atol=5e-2):
        raise ValueError(
            f"trajectory best_fitness does not match replayed native optimizer state "
            f"({float(state.best_fitness):.12e} vs {float(trajectory_row['best_fitness']):.12e}); "
            f"regenerate trajectories"
        )
    if int(state.generation) != int(trajectory_row["native_updates"]):
        raise ValueError("trajectory native_updates does not match replayed optimizer generation; regenerate trajectories")


def _parse_problem_id(problem_id: str, *, suite: str) -> tuple[int, int, int]:
    suite_name = str(suite).lower()
    if suite_name == "bbob":
        match = re.match(r"^bbob_f(\d{3})_i(\d+)_d(\d+)$", problem_id)
        if match is None:
            raise ValueError(f"invalid BBOB problem_id: {problem_id}")
        return tuple(int(value) for value in match.groups())
    if suite_name in {"cec2017", "cec2022"}:
        match = re.match(rf"^{suite_name}_f(\d{{2}})_d(\d+)$", problem_id)
        if match is None:
            raise ValueError(f"invalid {suite_name.upper()} problem_id: {problem_id}")
        function, dimension = (int(value) for value in match.groups())
        return function, 1, dimension
    raise ValueError(f"unsupported benchmark suite for state-action loss generation: {suite}")


def _schema() -> pa.Schema:
    fields = [
        ("split", pa.string()),
        ("problem_id", pa.string()),
        ("function_id", pa.string()),
        ("family", pa.string()),
        ("cv_group_id", pa.string()),
        ("dimension", pa.int32()),
        ("prefix_algorithm", pa.string()),
        ("prefix_scope", pa.string()),
        ("no_query_algorithm", pa.string()),
        ("seed", pa.int64()),
        ("FE", pa.int64()),
        ("FE_prefix", pa.int64()),
        ("FE_ratio", pa.float64()),
        ("FE_total", pa.int64()),
        *SAMPLING_METADATA_SCHEMA_FIELDS,
        ("action_budget_mode", pa.string()),
        ("sample_design_id", pa.string()),
        ("sample_design_protocol", pa.string()),
        ("FE_query", pa.int64()),
        ("FE_no_query_optimization", pa.int64()),
        ("FE_action_optimization", pa.int64()),
        ("remaining_budget_ratio", pa.float64()),
        ("performance_value_mode", pa.string()),
        ("performance_loss_mode", pa.string()),
        ("benchmark_reference_value", pa.float64()),
        ("success_gap_target", pa.float64()),
        ("p_skip", pa.float64()),
        ("p_skip_raw", pa.float64()),
        ("loss_skip", pa.float64()),
        ("runtime_no_query_handoff", pa.float64()),
        ("runtime_no_query_optimization", pa.float64()),
        ("runtime_no_query_handoff_repetitions", pa.list_(pa.float64())),
        ("runtime_no_query_optimization_repetitions", pa.list_(pa.float64())),
        ("no_query_transition_mode", pa.string()),
        ("skip_status", pa.string()),
        ("skip_failure_type", pa.string()),
        ("skip_failure_message", pa.string()),
        ("skip_execution_order", pa.int32()),
        ("skip_execution_order_repetitions", pa.list_(pa.int32())),
        ("skip_prefix_first_hit_FE", pa.int64()),
        ("skip_continuation_first_hit_FE", pa.int64()),
        ("skip_observed_first_hit_FE", pa.int64()),
        ("skip_target_hit_observed", pa.bool_()),
        ("skip_target_hit_before_failure", pa.bool_()),
        ("skip_endpoint_success", pa.bool_()),
        ("skip_first_hit_FE", pa.int64()),
        ("skip_success", pa.bool_()),
        ("skip_planned_FE", pa.int64()),
        ("skip_effective_FE", pa.int64()),
        ("skip_timed_out", pa.bool_()),
        ("skip_path_completed", pa.bool_()),
        ("failure_loss_cap", pa.float64()),
        ("log10_gap_floor", pa.float64()),
        ("log10_gap_cap", pa.float64()),
        ("execution_order_protocol", pa.string()),
        ("action_outcome_execution_count", pa.int32()),
        ("action_runtime_role", pa.string()),
        ("timing_repetitions", pa.int32()),
        ("timing_repetition_indices", pa.list_(pa.int32())),
        ("timing_order_protocol", pa.string()),
        ("action", pa.string()),
        ("target_algorithm", pa.string()),
        ("transition_mode", pa.string()),
        ("action_status", pa.string()),
        ("failure_type", pa.string()),
        ("failure_message", pa.string()),
        ("execution_order", pa.int32()),
        ("execution_order_repetitions", pa.list_(pa.int32())),
        ("action_loss", pa.float64()),
        ("selector_target_loss", pa.float64()),
        ("loss_gap_raw", pa.float64()),
        ("loss_gap_norm", pa.float64()),
        ("runtime_handoff", pa.float64()),
        ("runtime_action_optimization", pa.float64()),
        ("runtime_handoff_repetitions", pa.list_(pa.float64())),
        ("runtime_action_optimization_repetitions", pa.list_(pa.float64())),
        ("prefix_first_hit_FE", pa.int64()),
        ("continuation_first_hit_FE", pa.int64()),
        ("observed_first_hit_FE", pa.int64()),
        ("target_hit_observed", pa.bool_()),
        ("target_hit_before_failure", pa.bool_()),
        ("endpoint_success", pa.bool_()),
        ("first_hit_FE", pa.int64()),
        ("success", pa.bool_()),
        ("planned_FE", pa.int64()),
        ("effective_FE", pa.int64()),
        ("timed_out", pa.bool_()),
        ("path_completed", pa.bool_()),
        ("best_observed_algorithm", pa.string()),
        ("best_observed_loss", pa.float64()),
        ("action_loss_protocol", pa.string()),
    ]
    return pa.schema(fields)


def _pre_run_schema() -> pa.Schema:
    return pa.schema(
        [
            ("split", pa.string()),
            ("problem_id", pa.string()),
            ("function_id", pa.string()),
            ("family", pa.string()),
            ("cv_group_id", pa.string()),
            ("dimension", pa.int32()),
            ("seed", pa.int64()),
            ("FE", pa.int64()),
            ("FE_prefix", pa.int64()),
            ("FE_ratio", pa.float64()),
            ("FE_total", pa.int64()),
            ("action_budget_mode", pa.string()),
            ("sample_design_id", pa.string()),
            ("sample_design_protocol", pa.string()),
            ("FE_query", pa.int64()),
            ("FE_action_optimization", pa.int64()),
            ("remaining_budget_ratio", pa.float64()),
            ("performance_value_mode", pa.string()),
            ("performance_loss_mode", pa.string()),
            ("benchmark_reference_value", pa.float64()),
            ("success_gap_target", pa.float64()),
            ("failure_loss_cap", pa.float64()),
            ("log10_gap_floor", pa.float64()),
            ("log10_gap_cap", pa.float64()),
            ("execution_order_protocol", pa.string()),
            ("action_outcome_execution_count", pa.int32()),
            ("action_runtime_role", pa.string()),
            ("timing_repetitions", pa.int32()),
            ("timing_repetition_indices", pa.list_(pa.int32())),
            ("timing_order_protocol", pa.string()),
            ("action", pa.string()),
            ("target_algorithm", pa.string()),
            ("transition_mode", pa.string()),
            ("action_status", pa.string()),
            ("failure_type", pa.string()),
            ("failure_message", pa.string()),
            ("execution_order", pa.int32()),
            ("execution_order_repetitions", pa.list_(pa.int32())),
            ("action_loss", pa.float64()),
            ("action_loss_raw", pa.float64()),
            ("selector_target_loss", pa.float64()),
            ("loss_gap_raw", pa.float64()),
            ("loss_gap_norm", pa.float64()),
            ("runtime_handoff", pa.float64()),
            ("runtime_fresh_initialization", pa.float64()),
            ("runtime_action_optimization", pa.float64()),
            ("runtime_handoff_repetitions", pa.list_(pa.float64())),
            ("runtime_fresh_initialization_repetitions", pa.list_(pa.float64())),
            ("runtime_action_optimization_repetitions", pa.list_(pa.float64())),
            ("prefix_first_hit_FE", pa.int64()),
            ("continuation_first_hit_FE", pa.int64()),
            ("observed_first_hit_FE", pa.int64()),
            ("target_hit_observed", pa.bool_()),
            ("target_hit_before_failure", pa.bool_()),
            ("endpoint_success", pa.bool_()),
            ("first_hit_FE", pa.int64()),
            ("success", pa.bool_()),
            ("planned_FE", pa.int64()),
            ("effective_FE", pa.int64()),
            ("timed_out", pa.bool_()),
            ("path_completed", pa.bool_()),
            ("best_observed_algorithm", pa.string()),
            ("best_observed_loss", pa.float64()),
            ("action_loss_protocol", pa.string()),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate every portfolio action from each shared optimizer state."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, default=Path("configs/phase1_bbob_train.yaml"))
    parser.add_argument("--action-budget-mode", choices=ACTION_BUDGET_MODES, required=True)
    parser.add_argument("--sample-design-id", choices=sorted(SAMPLE_DESIGN_SPECS), default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    parser.add_argument(
        "--default-algorithm",
        choices=("de", "pso", "cmaes", "shade"),
        default=None,
        help="Scoped consistency runs only; formal generation omits this and derives SBS from --train-config.",
    )
    parser.add_argument("--all-prefixes", action="store_true")
    parser.add_argument("--max-states", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    generate_state_action_losses(
        config_path=args.config,
        train_config_path=args.train_config,
        sample_design_id=args.sample_design_id,
        action_budget_mode=args.action_budget_mode,
        output_path=args.output,
        only_functions=args.only_function,
        only_dimensions=args.only_dimension,
        default_algorithm=args.default_algorithm,
        all_prefixes=args.all_prefixes,
        max_states=args.max_states,
        workers=args.workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
