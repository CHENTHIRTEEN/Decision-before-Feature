from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from behavior.features import BEHAVIOR_FEATURE_GROUPS, extract_behavior_rows, read_trajectory_rows
from behavior_with_ela.collection import (
    local_landscape_shard_path,
    optimizer_checkpoint_shard_path,
    shard_paths,
)
from behavior_with_ela.local_landscape import (
    LOCAL_LANDSCAPE_FEATURE_COLUMNS,
    LOCAL_LANDSCAPE_METADATA_COLUMNS,
)
from behavior_with_ela.protocol import (
    ExperimentConfig,
    SuiteConfig,
    function_label,
    load_experiment_config,
    make_experiment_problem,
    selected_mabbob_definitions,
    suite_code,
)
from benchmarks.core import Problem
from optimizers import (
    NO_QUERY_TRANSFER_EVENT,
    advance_optimizer_state,
    clone_optimizer_state,
    initialize_transferred_optimizer_state,
)
from optimizers.seeding import make_event_rng
from optimizers.state import NATIVE_STREAMS, OptimizerState
from trajectory.sampling import SAMPLING_METADATA_COLUMNS, get_sampling_spec
from utility_labels.efficacy import problem_scale_epsilon


REPLICATION_SELECTION_STREAM = 2026082701
ACTION_ORDER_STREAM = 2026082702
CONTINUATION_REPETITION_STREAM_OFFSET = 30_000
TRANSFER_REPETITION_EVENT_OFFSET = 40_000
ALGORITHM_CODES = {"pso": 2, "cmaes": 3, "shade": 4}
GFE_GATE_BEHAVIOR_FEATURE_COLUMNS = tuple(
    BEHAVIOR_FEATURE_GROUPS["B2+Motion+SearchMaturityLinear"]
)
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
    "decision_opportunity_index",
)


class _TrackedActionObjective:
    def __init__(
        self,
        *,
        problem: Problem,
        reference_value: float,
        success_gap_target: float,
        initial_best_fitness: float,
    ) -> None:
        self.problem = problem
        self.reference_value = float(reference_value)
        self.success_gap_target = float(success_gap_target)
        self.evaluations = 0
        self.first_hit_offset: int | None = None
        self.best_fitness = float(initial_best_fitness)
        self.improvement_offsets = [0]
        self.improvement_values = [float(initial_best_fitness)]

    def __call__(self, values: np.ndarray) -> np.ndarray:
        points = np.asarray(values, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        results: list[float] = []
        for point in points:
            value = float(self.problem.evaluate(point)[0])
            self.evaluations += 1
            if value < self.best_fitness:
                self.best_fitness = value
                self.improvement_offsets.append(self.evaluations)
                self.improvement_values.append(value)
            gap = max(value - self.reference_value, 0.0)
            if self.first_hit_offset is None and gap <= self.success_gap_target:
                self.first_hit_offset = self.evaluations
            results.append(value)
        return np.asarray(results, dtype=float)

    def wrapped_problem(self) -> Problem:
        return Problem(
            problem_id=self.problem.problem_id,
            function_id=self.problem.function_id,
            family=self.problem.family,
            cv_group_id=self.problem.cv_group_id,
            dimension=self.problem.dimension,
            suite_code=self.problem.suite_code,
            function_number=self.problem.function_number,
            instance_number=self.problem.instance_number,
            bounds=self.problem.bounds.copy(),
            objective=self,
            reference_value=self.problem.reference_value,
            boundary_handling=self.problem.boundary_handling,
        )


def build_action_datasets(
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
        for function in suite.functions
        if only_functions is None or function in set(only_functions)
    ]
    if not tasks:
        raise ValueError("no action-dataset shards were selected")

    written = 0
    skipped = 0
    pending: list[tuple[SuiteConfig, int]] = []
    for suite, function in tasks:
        outputs = action_shard_paths(config, suite, function)
        existing = tuple(path.exists() for path in outputs)
        if all(existing) and not overwrite:
            skipped += 1
            continue
        if any(existing) and not all(existing) and not overwrite:
            raise FileExistsError(
                f"action repetition and aggregate outputs must be regenerated together: {outputs[0].parent}"
            )
        pending.append((suite, function))

    if workers == 1:
        for suite, function in pending:
            _build_and_write_action_shard(config, suite, function, overwrite)
            written += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _build_and_write_action_shard,
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


def action_shard_paths(
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
) -> tuple[Path, Path]:
    directory = (
        config.output_root
        / "actions"
        / suite.split
        / function_label(suite.suite, function)
        / f"dimension_{config.dimension}"
    )
    return directory / "action_repetitions.parquet", directory / "action_gain_dataset.parquet"


def _build_and_write_action_shard(
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    overwrite: bool,
) -> tuple[int, int]:
    trajectory_path, final_path, _ = shard_paths(config, suite, function)
    local_path = local_landscape_shard_path(config, suite, function)
    checkpoint_path = optimizer_checkpoint_shard_path(config, suite, function)
    if (
        not trajectory_path.exists()
        or not final_path.exists()
        or not local_path.exists()
        or not checkpoint_path.exists()
    ):
        raise FileNotFoundError(
            f"missing trajectory inputs for {suite.split} function {function}; run behavior-ela-collect first"
        )
    repetition_path, dataset_path = action_shard_paths(config, suite, function)
    repetition_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        repetition_path.unlink(missing_ok=True)
        dataset_path.unlink(missing_ok=True)

    trajectory_rows = read_trajectory_rows(trajectory_path)
    final_performance = pd.read_parquet(final_path)
    local_landscape = pd.read_parquet(local_path)
    optimizer_checkpoints = pd.read_parquet(checkpoint_path)
    if not trajectory_rows:
        raise ValueError(f"trajectory shard is empty: {trajectory_path}")
    _validate_action_sources(
        config=config,
        suite=suite,
        trajectories=pd.DataFrame(trajectory_rows),
        final_performance=final_performance,
    )
    raw_rows, aggregate_rows = build_action_rows(
        config=config,
        suite=suite,
        function=function,
        trajectory_rows=trajectory_rows,
        local_landscape_rows=local_landscape.to_dict(orient="records"),
        optimizer_checkpoint_rows=optimizer_checkpoints.to_dict(orient="records"),
    )
    pd.DataFrame(raw_rows).to_parquet(repetition_path, index=False)
    pd.DataFrame(aggregate_rows).to_parquet(dataset_path, index=False)
    return len(raw_rows), len(aggregate_rows)


def _validate_action_sources(
    *,
    config: ExperimentConfig,
    suite: SuiteConfig,
    trajectories: pd.DataFrame,
    final_performance: pd.DataFrame,
) -> None:
    endpoint_required = {
        "problem_id",
        "algorithm",
        "seed",
        "FE",
        "FE_total",
        "run_status",
        "path_completed",
        "planned_FE",
        "effective_FE",
    }
    missing = sorted(endpoint_required.difference(final_performance.columns))
    if missing:
        raise ValueError(f"final-performance input is missing columns: {missing}")
    expected_runs = len(suite.instances) * len(config.seeds) * len(config.algorithms)
    if len(final_performance) != expected_runs:
        raise ValueError(
            "final-performance run count differs from the configured shard: "
            f"observed={len(final_performance)}, expected={expected_runs}"
        )
    complete = (
        final_performance["run_status"].astype(str).eq("completed")
        & final_performance["path_completed"].astype(bool)
        & final_performance["FE"].astype(int).eq(config.fe_total)
        & final_performance["FE_total"].astype(int).eq(config.fe_total)
        & final_performance["planned_FE"].astype(int).eq(config.fe_total)
        & final_performance["effective_FE"].astype(int).eq(config.fe_total)
    )
    if not complete.all():
        row = final_performance.loc[~complete].iloc[0]
        raise ValueError(
            "action labels require a completed base trajectory: "
            f"problem_id={row['problem_id']}, algorithm={row['algorithm']}, "
            f"seed={int(row['seed'])}, status={row['run_status']}, "
            f"effective_FE={int(row['effective_FE'])}"
        )

    trajectory_required = {
        "problem_id",
        "algorithm",
        "seed",
        "FE",
        "sampling_protocol",
    }
    missing = sorted(trajectory_required.difference(trajectories.columns))
    if missing:
        raise ValueError(f"trajectory input is missing columns: {missing}")
    run_counts = trajectories.groupby(
        ["problem_id", "algorithm", "seed"],
        sort=False,
    ).size()
    spec = get_sampling_spec(config.sampling_protocol)
    if len(run_counts) != expected_runs:
        raise ValueError(
            "trajectory run coverage differs from final performance: "
            f"observed={len(run_counts)}, expected={expected_runs}"
        )
    if not run_counts.between(
        spec.min_samples_per_run,
        spec.max_samples_per_run,
        inclusive="both",
    ).all():
        raise ValueError("trajectory state counts differ from the sampling protocol")
    if not trajectories["sampling_protocol"].astype(str).eq(
        config.sampling_protocol
    ).all():
        raise ValueError("trajectory rows use a different sampling protocol")


def build_action_rows(
    *,
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    trajectory_rows: list[dict],
    optimizer_checkpoint_rows: list[dict],
    local_landscape_rows: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    replication_plan = _build_replication_plan(
        config=config,
        suite=suite,
        function=function,
        trajectory_rows=trajectory_rows,
    )
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for row in trajectory_rows:
        key = (str(row["problem_id"]), str(row["algorithm"]), int(row["seed"]))
        grouped.setdefault(key, []).append(dict(row))
    local_by_state: dict[tuple[str, str, int, int, int], dict] = {}
    if local_landscape_rows is not None:
        for row in local_landscape_rows:
            key = (
                str(row["problem_id"]),
                str(row["algorithm"]),
                int(row["seed"]),
                int(row["FE"]),
                int(row["decision_opportunity_index"]),
            )
            if key in local_by_state:
                raise ValueError("local landscape rows contain a duplicate state")
            local_by_state[key] = dict(row)
    checkpoint_by_state: dict[tuple[str, str, int, int, int], bytes] = {}
    for row in optimizer_checkpoint_rows:
        key = (
            str(row["problem_id"]),
            str(row["algorithm"]),
            int(row["seed"]),
            int(row["FE"]),
            int(row["decision_opportunity_index"]),
        )
        if key in checkpoint_by_state:
            raise ValueError("optimizer checkpoint rows contain a duplicate state")
        checkpoint_by_state[key] = bytes(row["optimizer_state_payload"])

    repetition_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    component_functions = _component_functions(suite, function)

    for (problem_id, prefix_algorithm, seed), rows in grouped.items():
        if prefix_algorithm not in config.algorithms:
            raise ValueError(f"trajectory contains an algorithm outside the new portfolio: {prefix_algorithm}")
        ordered = sorted(rows, key=lambda item: int(item["FE"]))
        if len({int(row["FE"]) for row in ordered}) != len(ordered):
            raise ValueError("a trajectory run contains duplicate decision-state FE values")
        behavior_rows = extract_behavior_rows([dict(row) for row in ordered])
        behavior_by_fe = {int(row["FE"]): row for row in behavior_rows}
        instance = _problem_instance(ordered[0], suite.suite)
        problem = make_experiment_problem(
            suite,
            function=function,
            instance=instance,
            dimension=config.dimension,
            boundary_handling=config.boundary_handling,
        )
        try:
            reference = problem.reference_value
            if reference is None:
                raise ValueError(f"benchmark reference value is unavailable for {problem_id}")
            for opportunity_index, row in enumerate(ordered):
                checkpoint_fe = int(row["FE"])
                checkpoint_key = (
                    problem_id,
                    prefix_algorithm,
                    seed,
                    checkpoint_fe,
                    opportunity_index,
                )
                payload = checkpoint_by_state.get(checkpoint_key)
                if payload is None:
                    raise ValueError(
                        f"decision state is missing its optimizer checkpoint: {checkpoint_key}"
                    )
                state = pickle.loads(payload)
                _check_checkpoint_state(state, row)
                behavior = behavior_by_fe[checkpoint_fe]
                local_key = (
                    problem_id,
                    prefix_algorithm,
                    seed,
                    checkpoint_fe,
                    opportunity_index,
                )
                local_features = local_by_state.get(local_key)
                if local_landscape_rows is not None and local_features is None:
                    raise ValueError(
                        f"decision state is missing local landscape features: {local_key}"
                    )
                replication_key = _replication_state_key(row)
                replication_reason = replication_plan[replication_key]
                repetitions = (
                    config.replication.selected_state_repetitions
                    if replication_reason != "full_coverage_only"
                    else config.replication.full_coverage
                )
                state_rows = _evaluate_state_actions(
                    config=config,
                    suite=suite,
                    function=function,
                    instance=instance,
                    prefix_algorithm=prefix_algorithm,
                    seed=seed,
                    checkpoint_state=state,
                    problem=problem,
                    checkpoint_fe=checkpoint_fe,
                    repetitions=repetitions,
                )
                common = {
                    "split": suite.split,
                    "suite": suite.suite,
                    "problem_id": problem_id,
                    "function_id": str(row["function_id"]),
                    "family": str(row["family"]),
                    "cv_group_id": str(row.get("cv_group_id", row["function_id"])),
                    "component_functions": list(component_functions),
                    "dimension": config.dimension,
                    "boundary_handling": config.boundary_handling,
                    "prefix_algorithm": prefix_algorithm,
                    "seed": seed,
                    "FE": checkpoint_fe,
                    "FE_ratio": float(checkpoint_fe / config.fe_total),
                    "FE_total": config.fe_total,
                    "remaining_budget_ratio": float((config.fe_total - checkpoint_fe) / config.fe_total),
                    "prefix_best_fitness": float(row["best_fitness"]),
                    "benchmark_reference_value": float(reference),
                    "prefix_gap": max(
                        float(row["best_fitness"]) - float(reference),
                        0.0,
                    ),
                    "epsilon_p": float(
                        problem_scale_epsilon(
                            prefix_gap=max(
                                float(row["best_fitness"]) - float(reference),
                                0.0,
                            ),
                            problem_scale=1.0,
                        )
                    ),
                    "domain_gain_delta": float(config.domain_gain_delta),
                    "decision_opportunity_index": int(opportunity_index),
                    "planned_action_repetitions": int(repetitions),
                    "selected_for_additional_repetitions": bool(
                        repetitions > config.replication.full_coverage
                    ),
                    "replication_selection_reason": replication_reason,
                    "replication_stratum": _replication_stratum(row),
                    "state_id": _state_id(
                        suite.split,
                        problem_id,
                        prefix_algorithm,
                        seed,
                        checkpoint_fe,
                        opportunity_index,
                    ),
                    **{column: row[column] for column in SAMPLING_METADATA_COLUMNS},
                    **{
                        column: behavior[column]
                        for column in GFE_GATE_BEHAVIOR_FEATURE_COLUMNS
                    },
                    **(
                        {}
                        if local_features is None
                        else {
                            column: local_features[column]
                            for column in (
                                *LOCAL_LANDSCAPE_METADATA_COLUMNS,
                                *LOCAL_LANDSCAPE_FEATURE_COLUMNS,
                            )
                        }
                    ),
                }
                repetition_rows.extend({**common, **item} for item in state_rows)
                aggregate_rows.extend(
                    _aggregate_state_actions(
                        common=common,
                        rows=state_rows,
                        portfolio=config.algorithms,
                        practical_delta=config.domain_gain_delta,
                    )
                )
        finally:
            problem.close()
    return repetition_rows, aggregate_rows


def _evaluate_state_actions(
    *,
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    instance: int,
    prefix_algorithm: str,
    seed: int,
    checkpoint_state: OptimizerState,
    problem: Problem,
    checkpoint_fe: int,
    repetitions: int,
    fe_budget_override: int | None = None,
) -> list[dict]:
    fe_budget = (
        config.fe_total - int(checkpoint_fe)
        if fe_budget_override is None
        else int(fe_budget_override)
    )
    if fe_budget <= 0:
        raise ValueError("decision state must retain a positive terminal action budget")
    rows: list[dict] = []
    for replicate_id in range(repetitions):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    int(seed),
                    ACTION_ORDER_STREAM,
                    suite_code(suite.suite),
                    int(function),
                    int(instance),
                    int(problem.dimension),
                    int(checkpoint_fe),
                    int(replicate_id),
                ]
            )
        )
        order = tuple(
            str(value)
            for value in np.asarray(config.algorithms, dtype=object)[
                rng.permutation(len(config.algorithms))
            ]
        )
        replicate_rows: list[dict] = []
        for order_index, target_algorithm in enumerate(order):
            outcome = _evaluate_action(
                config=config,
                function=function,
                instance=instance,
                prefix_algorithm=prefix_algorithm,
                target_algorithm=target_algorithm,
                seed=seed,
                checkpoint_state=checkpoint_state,
                problem=problem,
                fe_budget=fe_budget,
                replicate_id=replicate_id,
            )
            outcome["execution_order"] = int(order_index)
            replicate_rows.append(outcome)
        continue_rows = [
            row for row in replicate_rows if bool(row["action_equals_prefix"])
        ]
        if len(continue_rows) != 1:
            raise RuntimeError("each state replicate must contain one continue-current action")
        continue_log_loss = float(continue_rows[0]["log10_action_loss"])
        for row in replicate_rows:
            gain = continue_log_loss - float(row["log10_action_loss"])
            row["action_gain_vs_continue"] = float(gain)
            row["action_gain_class"] = _gain_class(
                gain,
                config.domain_gain_delta,
            )
            rows.append(row)
    incomplete = [
        row
        for row in rows
        if str(row["action_status"]) != "completed"
        or int(row["effective_action_FE"]) != int(row["planned_action_FE"])
    ]
    if incomplete:
        first = incomplete[0]
        raise RuntimeError(
            "state-action continuation did not complete its FE budget: "
            f"candidate_action={first['candidate_action']}, "
            f"replicate_id={int(first['replicate_id'])}, "
            f"status={first['action_status']}, "
            f"effective_FE={int(first['effective_action_FE'])}, "
            f"planned_FE={int(first['planned_action_FE'])}; "
            f"{first['failure_type']}: {first['failure_message']}"
        )
    return rows


def _evaluate_action(
    *,
    config: ExperimentConfig,
    function: int,
    instance: int,
    prefix_algorithm: str,
    target_algorithm: str,
    seed: int,
    checkpoint_state: OptimizerState,
    problem: Problem,
    fe_budget: int,
    replicate_id: int,
) -> dict:
    reference = problem.reference_value
    if reference is None:
        raise ValueError("action loss requires a benchmark reference value")
    tracker = _TrackedActionObjective(
        problem=problem,
        reference_value=float(reference),
        success_gap_target=config.success_gap_target,
        initial_best_fitness=float(checkpoint_state.best_fitness),
    )
    tracked_problem = tracker.wrapped_problem()
    handoff_required = target_algorithm != prefix_algorithm
    transition_mode = (
        "population_transfer_initialization"
        if handoff_required
        else "native_optimizer_state"
    )
    status = "completed"
    failure_type = ""
    failure_message = ""
    try:
        if handoff_required:
            event = (
                NO_QUERY_TRANSFER_EVENT
                if replicate_id == 0
                else TRANSFER_REPETITION_EVENT_OFFSET + int(replicate_id)
            )
            state = initialize_transferred_optimizer_state(
                algorithm=target_algorithm,
                source_state=checkpoint_state,
                problem=tracked_problem,
                seed=seed,
                function=function,
                instance=instance,
                event=event,
            )
        else:
            state = clone_optimizer_state(checkpoint_state)
            if replicate_id > 0:
                rng = make_event_rng(
                    seed=seed,
                    stream_code=(
                        CONTINUATION_REPETITION_STREAM_OFFSET
                        + NATIVE_STREAMS[prefix_algorithm]
                    ),
                    suite_code=problem.suite_code,
                    function=function,
                    instance=instance,
                    dimension=problem.dimension,
                    generation=int(checkpoint_state.generation),
                    event=int(replicate_id),
                )
                state.rng_state = deepcopy(rng.bit_generator.state)
        advance_optimizer_state(
            state=state,
            problem=tracked_problem,
            fe_budget=fe_budget,
        )
        best_fitness = float(state.best_fitness)
        if not np.isfinite(best_fitness):
            raise FloatingPointError("action continuation returned a non-finite best value")
        gap = min(max(best_fitness - float(reference), 0.0), config.failure_loss_cap)
    except Exception as exc:
        status = "failed"
        failure_type = type(exc).__name__
        failure_message = str(exc)[:500]
        gap = config.failure_loss_cap
    log_loss = float(
        np.log10(np.clip(gap, config.log10_gap_floor, config.log10_gap_cap))
    )
    return {
        "replicate_id": int(replicate_id),
        "candidate_action": target_algorithm,
        "action": "continue_current" if not handoff_required else target_algorithm,
        "action_equals_prefix": bool(not handoff_required),
        "handoff_required": bool(handoff_required),
        "handoff_type": transition_mode,
        "action_status": status,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "planned_action_FE": int(fe_budget),
        "effective_action_FE": int(tracker.evaluations),
        "action_loss": float(gap),
        "log10_action_loss": log_loss,
        "first_hit_action_offset": tracker.first_hit_offset,
        "target_hit": bool(gap <= config.success_gap_target and status == "completed"),
        "action_improvement_offsets": list(tracker.improvement_offsets),
        "action_improvement_values": list(tracker.improvement_values),
    }


def _aggregate_state_actions(
    *,
    common: dict,
    rows: list[dict],
    portfolio: tuple[str, ...],
    practical_delta: float,
) -> list[dict]:
    by_action: dict[str, list[dict]] = {algorithm: [] for algorithm in portfolio}
    for row in rows:
        by_action[str(row["candidate_action"])].append(row)
    if any(not values for values in by_action.values()):
        raise RuntimeError("a state is missing one or more candidate actions")

    summaries: list[dict] = []
    for algorithm in portfolio:
        values = by_action[algorithm]
        gains = np.asarray([float(row["action_gain_vs_continue"]) for row in values])
        losses = np.asarray([float(row["action_loss"]) for row in values])
        log_losses = np.asarray([float(row["log10_action_loss"]) for row in values])
        gain_median = float(np.median(gains))
        signs = np.sign(gains)
        median_sign = np.sign(gain_median)
        if len(gains) > 1:
            gain_std = float(np.std(gains, ddof=1))
            gain_mean = float(np.mean(gains))
            half_width = float(
                student_t.ppf(0.975, df=len(gains) - 1)
                * gain_std
                / np.sqrt(len(gains))
            )
            ci_low: float | None = gain_mean - half_width
            ci_high: float | None = gain_mean + half_width
            interval_overlaps_equivalence: bool | None = bool(
                ci_low <= practical_delta and ci_high >= -practical_delta
            )
            probability_gain_above_zero: float | None = float(
                np.mean(gains > 0.0)
            )
            sign_flip_rate: float | None = float(np.mean(signs != median_sign))
        else:
            gain_std = None
            gain_mean = float(gains[0])
            ci_low = None
            ci_high = None
            interval_overlaps_equivalence = None
            probability_gain_above_zero = None
            sign_flip_rate = None
        summaries.append(
            {
                **common,
                "candidate_action": algorithm,
                "action": "continue_current" if algorithm == common["prefix_algorithm"] else algorithm,
                "action_equals_prefix": bool(algorithm == common["prefix_algorithm"]),
                "handoff_required": bool(algorithm != common["prefix_algorithm"]),
                "handoff_type": (
                    "native_optimizer_state"
                    if algorithm == common["prefix_algorithm"]
                    else "population_transfer_initialization"
                ),
                "action_repetitions": len(values),
                "action_loss": float(np.median(losses)),
                "log10_action_loss": float(np.median(log_losses)),
                "action_gain_vs_continue": gain_median,
                "action_gain_mean": gain_mean,
                "action_gain_std": gain_std,
                "action_gain_ci_low": ci_low,
                "action_gain_ci_high": ci_high,
                "action_gain_ci_confidence": 0.95,
                "action_gain_ci_defined": bool(len(gains) > 1),
                "action_gain_ci_overlaps_equivalence": interval_overlaps_equivalence,
                "action_gain_probability_above_zero": probability_gain_above_zero,
                "action_gain_sign_flip_rate": sign_flip_rate,
                "action_gain_class": _gain_class(gain_median, practical_delta),
                "all_action_paths_completed": bool(
                    all(str(row["action_status"]) == "completed" for row in values)
                ),
                "all_action_paths_used_planned_FE": bool(
                    all(
                        int(row["effective_action_FE"])
                        == int(row["planned_action_FE"])
                        for row in values
                    )
                ),
            }
        )
    portfolio_index = {algorithm: index for index, algorithm in enumerate(portfolio)}
    best = min(
        summaries,
        key=lambda row: (
            float(row["log10_action_loss"]),
            portfolio_index[str(row["candidate_action"])],
        ),
    )
    best_loss = float(best["log10_action_loss"])
    for row in summaries:
        row["best_observed_action"] = str(best["candidate_action"])
        row["acceptable_action"] = bool(
            float(row["log10_action_loss"]) - best_loss <= practical_delta
        )
    return summaries


def _build_replication_plan(
    *,
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    trajectory_rows: list[dict],
) -> dict[tuple[str, str, int, int], str]:
    replication = config.replication
    plan = {
        _replication_state_key(row): "full_coverage_only"
        for row in trajectory_rows
    }
    candidates: dict[str, list[dict]] = {}
    for row in trajectory_rows:
        key = _replication_state_key(row)
        if replication.include_event_states and bool(row["is_event_sample"]):
            plan[key] = "event_state"
            continue
        candidates.setdefault(_replication_stratum(row), []).append(row)

    fraction = float(replication.selected_state_fraction)
    if fraction <= 0.0:
        return plan
    for rows in candidates.values():
        selected_count = int(np.floor(len(rows) * fraction + 0.5))
        selected_count = min(len(rows), max(1, selected_count))
        ranked = sorted(
            rows,
            key=lambda row: (
                _replication_random_score(
                    config=config,
                    suite=suite,
                    function=function,
                    row=row,
                ),
                _replication_state_key(row),
            ),
        )
        for row in ranked[:selected_count]:
            plan[_replication_state_key(row)] = "prespecified_stratified_random"
    return plan


def _replication_random_score(
    *,
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    row: dict,
) -> float:
    prefix_algorithm = str(row["algorithm"])
    instance = _problem_instance(row, suite.suite)
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                config.replication.selection_seed,
                REPLICATION_SELECTION_STREAM,
                suite_code(suite.suite),
                int(function),
                int(instance),
                config.dimension,
                int(row["seed"]),
                ALGORITHM_CODES[prefix_algorithm],
                int(row["FE"]),
            ]
        )
    )
    return float(rng.random())


def _replication_state_key(row: dict) -> tuple[str, str, int, int]:
    return (
        str(row["problem_id"]),
        str(row["algorithm"]),
        int(row["seed"]),
        int(row["FE"]),
    )


def _replication_stratum(row: dict) -> str:
    state_type = "event" if bool(row["is_event_sample"]) else "milestone"
    return ":".join(
        (
            str(row["function_id"]),
            str(row["algorithm"]),
            str(row["sampling_phase"]),
            state_type,
        )
    )


def _check_checkpoint_state(state: OptimizerState, row: dict) -> None:
    population = np.asarray(row["population"], dtype=float)
    fitness = np.asarray(row["fitness"], dtype=float).reshape(-1)
    if state.population.shape != population.shape or state.fitness.shape != fitness.shape:
        raise ValueError("optimizer checkpoint shape differs from trajectory state")
    if not np.allclose(state.population, population, rtol=1e-12, atol=1e-12):
        raise ValueError("optimizer checkpoint population differs from trajectory state")
    if not np.allclose(state.fitness, fitness, rtol=1e-12, atol=1e-12):
        raise ValueError("optimizer checkpoint fitness differs from trajectory state")
    if not np.isclose(float(state.best_fitness), float(row["best_fitness"]), rtol=1e-12, atol=1e-12):
        raise ValueError("optimizer checkpoint best value differs from trajectory state")
    if int(state.generation) != int(row["native_updates"]):
        raise ValueError("optimizer checkpoint native-update index differs from trajectory state")
    if int(state.evaluations) != int(row["FE"]):
        raise ValueError("optimizer checkpoint FE differs from trajectory state")


def _component_functions(suite: SuiteConfig, function: int) -> tuple[int, ...]:
    if suite.suite == "bbob":
        return (int(function),)
    if suite.suite != "mabbob":
        return ()
    if suite.definitions_path is None:
        raise ValueError("MA-BBOB suite is missing definitions_path")
    row = selected_mabbob_definitions(suite.definitions_path)[int(function)]
    return tuple(int(value) for value in row["components"])


def _problem_instance(row: dict, suite: str) -> int:
    if suite != "bbob":
        return 1
    problem_id = str(row["problem_id"])
    marker = "_i"
    if marker not in problem_id:
        raise ValueError(f"cannot parse BBOB instance from {problem_id}")
    return int(problem_id.split(marker, 1)[1].split("_", 1)[0])


def _state_id(
    split: str,
    problem_id: str,
    prefix_algorithm: str,
    seed: int,
    fe: int,
    opportunity_index: int,
) -> str:
    return (
        f"{split}:{problem_id}:{prefix_algorithm}:seed{int(seed)}:"
        f"fe{int(fe)}:op{int(opportunity_index)}"
    )


def _gain_class(value: float, delta: float) -> str:
    if float(value) > float(delta):
        return "improve"
    if float(value) < -float(delta):
        return "degrade"
    return "equivalent"


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build terminal state-action outcomes and Behavior action-gain labels."
    )
    parser.add_argument("--config", default="configs/behavior_with_ela_train.yaml")
    parser.add_argument("--only-split", action="append", default=None)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = build_action_datasets(
        config_path=args.config,
        only_splits=None if args.only_split is None else tuple(args.only_split),
        only_functions=(
            None if args.only_function is None else tuple(args.only_function)
        ),
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(
        f"wrote {summary['written_shards']} action-dataset shards; "
        f"skipped {summary['skipped_shards']} existing shards"
    )


if __name__ == "__main__":
    main()
