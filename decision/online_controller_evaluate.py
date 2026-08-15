from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import qmc

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS, extract_behavior_rows
from behavior.streaming import StreamingBehaviorState
from benchmarks import Problem, make_problem
from decision.model_protocol import (
    FROZEN_THRESHOLD_MODE,
    SELECTED_MODEL_ALIAS,
    decision_scores,
    resolve_model_name,
)
from decision.matched_random import (
    MatchedRandomCalibration,
    load_matched_random_calibration,
    matched_random_target,
)
from experiments.phase1_batch_common import (
    TIMING_ORDER_PROTOCOL,
    TIMING_REPETITIONS,
    as_int_list,
    function_id_name,
    fe_total_for_dimension,
    load_config,
    selected_dimensions,
    selected_functions,
    split_name,
    validate_dynamic_collection_config,
)
from landscape_queries.batch_features import FEATURE_METADATA_COLUMNS
from landscape_queries.batch_sampling import SAMPLE_KEY_COLUMNS, default_sample_path
from landscape_queries.cheap import calculate_descriptor_cheap
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS, MAIN_QUERY_ID, get_query_spec
from optimizers import (
    OptimizerSettings,
    QUERY_TRANSFER_EVENT,
    advance_optimizer_state,
    initialize_optimizer_state,
    initialize_transferred_optimizer_state,
)
from selection_reference.model import StatewiseSelectorModel, load_selector_model, make_selector_features
from selection_reference.action_losses import FULL_REMAINING_BUDGET, NOT_APPLICABLE
from selection_reference.model import BEHAVIOR_ONLY_FULL_BUDGET_INPUT
from trajectory.sampling import (
    BUDGET_MILESTONE_RATIOS,
    EPS as SAMPLING_EPS,
    SAMPLING_METADATA_COLUMNS,
    SAMPLING_PROTOCOL,
    get_sampling_spec,
    sampling_phase,
)
from trajectory.records import TrajectoryRecord
from trajectory.window_statistics import NativeUpdateWindowRecorder


DEFAULT_CONFIG_PATH = Path("configs/phase1_cec2017_test.yaml")
DEFAULT_TRAIN_CONFIG_PATH = Path("configs/phase1_bbob_train.yaml")
DEFAULT_MODEL_NAME = SELECTED_MODEL_ALIAS
DEFAULT_THRESHOLD_MODE = FROZEN_THRESHOLD_MODE
DEFAULT_RANDOM_REPETITIONS = 30
DEFAULT_SAMPLING_PROTOCOL = SAMPLING_PROTOCOL
SAMPLING_PROTOCOLS = (DEFAULT_SAMPLING_PROTOCOL,)
QUERY_ONLY_SELECTOR_PROTOCOL = "query_only_observed_action_loss_regression"
BEHAVIOR_ONLY_THRESHOLD_MODE = "oof_behavior_utility_first_trigger"
BEHAVIOR_ONLY_TARGET_COLUMN = "u_behavior_only_full_budget_lamT_1"
ONLINE_TIMING_STREAM_CODE = 8117
ONLINE_POLICY_PATHS = (
    "sbs_no_query",
    "always_query",
    "pre_run_aas_fe0",
    "milestone_only_T0",
    "current_controller",
    "matched_trigger_behavior_only",
    "self_thresholded_behavior_only",
)


@dataclass(frozen=True)
class DecisionControllerModel:
    model: Any
    model_name: str
    model_family: str
    threshold_mode: str
    threshold: float
    feature_columns: list[str]
    training_summary_path: Path
    model_path: Path
    query_id: str
    query_protocol: str
    feature_group: str
    opportunity_scope: str


class MilestoneBehaviorState:
    """Build history-only Behavior rows at the frozen budget milestones."""

    def __init__(
        self,
        *,
        problem_id: str,
        function_id: str,
        family: str,
        dimension: int,
        algorithm: str,
        seed: int,
        fe_total: int,
        sampling_protocol: str,
    ) -> None:
        if sampling_protocol != SAMPLING_PROTOCOL:
            raise ValueError("milestone-only behavior state requires the frozen sampling protocol")
        self.problem_id = str(problem_id)
        self.function_id = str(function_id)
        self.family = str(family)
        self.dimension = int(dimension)
        self.algorithm = str(algorithm)
        self.seed = int(seed)
        self.fe_total = int(fe_total)
        self.sampling_protocol = str(sampling_protocol)
        self._milestones = tuple(float(value) for value in BUDGET_MILESTONE_RATIOS)
        self._milestone_index = 0
        self._window_recorder = NativeUpdateWindowRecorder()
        self._trajectory_rows: list[dict[str, Any]] = []

    @property
    def next_monitor_ratio(self) -> float | None:
        if self._milestone_index >= len(self._milestones):
            return None
        return self._milestones[self._milestone_index]

    def observe(
        self,
        *,
        fe: int,
        native_updates: int,
        population: np.ndarray,
        fitness: np.ndarray,
        best_fitness: float,
    ) -> None:
        self._window_recorder.observe(
            fe=fe,
            native_updates=native_updates,
            population=population,
            fitness=fitness,
            best_fitness=best_fitness,
        )

    def sample_milestone(self) -> dict[str, Any] | None:
        target = self.next_monitor_ratio
        if target is None:
            return None
        current = self._window_recorder.current_snapshot
        if float(current.fe / self.fe_total) + SAMPLING_EPS < float(target):
            return None
        target_fe = int(round(float(target) * self.fe_total))
        alignment_gap = int(current.fe - target_fe)
        if alignment_gap < 0 or alignment_gap >= int(current.population.shape[0]):
            raise ValueError("milestone-only state is not aligned to the first complete native update")
        windows, history = self._window_recorder.build(
            fe_total=self.fe_total,
            problem_id=self.problem_id,
            algorithm=self.algorithm,
        )
        metadata = {
            "sampling_protocol": self.sampling_protocol,
            "sampling_phase": sampling_phase(float(target)),
            "sampling_triggers": ["budget_milestone"],
            "is_budget_milestone": True,
            "budget_milestone_ratio": float(target),
            "is_event_sample": False,
            "monitor_target_ratio": float(target),
            "event_index_in_phase": None,
            "event_improvement_resume": False,
            "event_stagnation_onset": False,
            "event_rank_change": False,
            "event_elite_migration": False,
            "event_diversity_recovery": False,
            "event_improvement_resume_metric": 0.0,
            "event_stagnation_onset_metric": 0.0,
            "event_rank_change_metric": 0.0,
            "event_elite_migration_metric": 0.0,
            "event_diversity_recovery_metric": 0.0,
        }
        record = TrajectoryRecord.from_arrays(
            problem_id=self.problem_id,
            function_id=self.function_id,
            family=self.family,
            dimension=self.dimension,
            algorithm=self.algorithm,
            seed=self.seed,
            fe=int(current.fe),
            fe_total=self.fe_total,
            native_updates=int(current.native_updates),
            window_statistics=windows,
            native_update_history=history,
            population=current.population,
            fitness=current.fitness,
            best_fitness=float(current.best_fitness),
            sampling_metadata=metadata,
        )
        self._trajectory_rows.append(record.__dict__.copy())
        self._milestone_index += 1
        return {
            "problem_id": self.problem_id,
            "family": self.family,
            "dimension": self.dimension,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "FE": int(current.fe),
            "FE_ratio": float(current.fe / self.fe_total),
            **metadata,
            "bf_fe_ratio": float(current.fe / self.fe_total),
        }

    def full_behavior_state(self) -> dict[str, Any]:
        if not self._trajectory_rows:
            raise ValueError("milestone behavior features require an emitted milestone")
        return extract_behavior_rows(
            [row.copy() for row in self._trajectory_rows]
        )[-1]


@dataclass
class PathEvaluationTracker:
    benchmark_reference_value: float
    success_gap_target: float
    planned_total_fe: int
    phase: str = "continuation"
    total_evaluations: int = 0
    prefix_evaluations: int = 0
    query_evaluations: int = 0
    continuation_evaluations: int = 0
    first_hit_fe: int | None = None
    best_all: float = float("inf")
    best_optimizer: float = float("inf")
    best_query: float = float("inf")
    execution_context: dict[str, Any] = field(default_factory=dict)

    def set_phase(self, phase: str) -> None:
        if phase not in {"prefix", "query", "continuation"}:
            raise ValueError(f"unknown online evaluation phase: {phase}")
        self.phase = phase

    def remember(self, **values: Any) -> None:
        """Retain already-observed policy decisions and component costs for failure rows."""
        self.execution_context.update(values)

    def observe(self, values: np.ndarray) -> None:
        for value in np.asarray(values, dtype=float).reshape(-1):
            self.total_evaluations += 1
            if self.phase == "prefix":
                self.prefix_evaluations += 1
            elif self.phase == "query":
                self.query_evaluations += 1
            else:
                self.continuation_evaluations += 1
            if not np.isfinite(float(value)):
                continue
            numeric = float(value)
            self.best_all = min(self.best_all, numeric)
            if self.phase == "query":
                self.best_query = min(self.best_query, numeric)
            else:
                self.best_optimizer = min(self.best_optimizer, numeric)
            gap = max(numeric - float(self.benchmark_reference_value), 0.0)
            if self.first_hit_fe is None and gap <= float(self.success_gap_target):
                self.first_hit_fe = int(self.total_evaluations)


@dataclass(frozen=True)
class QueryExecution:
    features: dict[str, float | None]
    status: str
    failure: str
    runtime_sampling: float
    runtime_evaluation: float
    runtime_feature_computation: float
    runtime_total: float
    first_hit_offset: int | None
    best_gap: float


@dataclass(frozen=True)
class OnlineSelector:
    model: StatewiseSelectorModel

    @property
    def sbs_algorithm(self) -> str:
        return self.model.default_algorithm

    def select(
        self,
        query_features: dict[str, Any],
        behavior_features: dict[str, Any],
        remaining_ratio: float,
    ) -> tuple[str, float, str, float]:
        features = make_selector_features(
            behavior_features=behavior_features,
            query_features=query_features,
            query_feature_columns=self.model.query_feature_columns,
            remaining_budget_ratio=remaining_ratio,
        )
        selected, _, runtime_selection = self.model.select_one(features)
        return selected, float(remaining_ratio), "random_forest_action_loss_regression", runtime_selection


@dataclass(frozen=True)
class OnlineQueryOnlySelector:
    artifact: Any

    @property
    def query_id(self) -> str:
        return str(self.artifact.query_id)

    def select(
        self,
        query_features: dict[str, Any],
        remaining_ratio: float,
    ) -> tuple[str, float, str, float]:
        features = {
            **{
                column: np.nan if query_features.get(column) is None else float(query_features[column])
                for column in self.artifact.query_feature_columns
            },
            "remaining_budget_ratio": float(remaining_ratio),
        }
        started = perf_counter()
        frame = pd.DataFrame(
            [{column: features[column] for column in self.artifact.feature_columns}]
        )
        scores = np.asarray(
            self.artifact.model.predict(frame[list(self.artifact.feature_columns)]),
            dtype=float,
        ).reshape(1, -1)[0]
        targets = tuple(str(value) for value in self.artifact.target_algorithms)
        if len(scores) != len(targets) or not np.isfinite(scores).all():
            raise ValueError("query-only selector returned invalid action-loss predictions")
        score_map = dict(zip(targets, scores, strict=True))
        selected = min(targets, key=lambda algorithm: (score_map[algorithm], algorithm))
        return (
            selected,
            float(remaining_ratio),
            QUERY_ONLY_SELECTOR_PROTOCOL,
            float(perf_counter() - started),
        )


@dataclass(frozen=True)
class OnlineBehaviorOnlySelector:
    model: StatewiseSelectorModel

    @property
    def sbs_algorithm(self) -> str:
        return self.model.default_algorithm

    def select(
        self,
        behavior_features: dict[str, Any],
        remaining_ratio: float,
    ) -> tuple[str, float, str, float]:
        features = {
            **{
                column: behavior_features[column]
                for column in self.model.feature_columns
                if column != "remaining_budget_ratio"
            },
            "remaining_budget_ratio": float(remaining_ratio),
        }
        selected, _, runtime_selection = self.model.select_one(features)
        return (
            selected,
            float(remaining_ratio),
            str(self.model.protocol),
            runtime_selection,
        )


def evaluate_online_controller(
    *,
    query_id: str,
    query_feature_path: Path,
    query_sample_path: Path | None,
    config_path: Path,
    train_config_path: Path,
    selector_model_path: Path,
    pre_run_aas_selector_model_path: Path,
    behavior_only_selector_model_path: Path,
    matched_random_calibration_path: Path,
    training_summary_path: Path,
    milestone_only_training_summary_path: Path,
    behavior_only_training_summary_path: Path,
    output_dir: Path,
    model_name: str,
    threshold_mode: str,
    sampling_protocol: str,
    random_repetitions: int,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    only_seeds: list[int] | None,
    max_runs: int | None,
    sharded: bool,
    summarize_only: bool,
    workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if workers > 1 and not sharded and not summarize_only:
        raise ValueError("--workers > 1 is only supported with --sharded")
    config = load_config(config_path)
    validate_dynamic_collection_config(config)
    if str(config["suite"]).lower() not in {"cec2017", "cec2022"}:
        raise ValueError("online controller evaluation currently expects an external CEC suite")
    budget_milestone_ratios = _budget_milestone_ratios(config, sampling_protocol)
    decision_check_frequency = _decision_check_frequency(sampling_protocol)
    output_dir = _sampling_output_dir(output_dir, sampling_protocol)
    functions = selected_functions(config, only_functions)
    dimensions = selected_dimensions(config, only_dimensions)
    seeds = _selected_seeds(config, only_seeds)
    controller = _load_controller(training_summary_path, model_name, threshold_mode)
    milestone_only_controller = _load_controller(
        milestone_only_training_summary_path,
        controller.model_name,
        threshold_mode,
    )
    behavior_only_controller = _load_behavior_only_controller(
        behavior_only_training_summary_path,
        controller.model_name,
    )
    model_name = controller.model_name
    if controller.feature_group != "B3" or controller.opportunity_scope != "all_accepted":
        raise ValueError("Proposed online controller must use B3/all_accepted training artifacts")
    if (
        milestone_only_controller.feature_group != "T0"
        or milestone_only_controller.opportunity_scope != "milestone_only"
        or milestone_only_controller.feature_columns != ["bf_fe_ratio"]
    ):
        raise ValueError("milestone_only_T0 must use T0/milestone_only with X={bf_fe_ratio}")
    if milestone_only_controller.model_name != controller.model_name:
        raise ValueError("milestone_only_T0 must use the model family selected by Proposed")
    if (
        behavior_only_controller.feature_group != "B3"
        or behavior_only_controller.opportunity_scope != "all_accepted"
    ):
        raise ValueError("Behavior-only online controller must use B3/all_accepted artifacts")
    query_spec = get_query_spec(query_id)
    resolved_query_sample_path = query_sample_path or default_sample_path(
        query_spec.sample_design_id,
        split_name(config),
    )
    if not summarize_only and query_id != MAIN_QUERY_ID:
        raise ValueError(
            "online full-run replay currently supports descriptor_cheap_invariant only"
        )
    if controller.query_id != query_id or controller.query_protocol != query_spec.protocol:
        raise ValueError("Decision controller query protocol does not match the requested online evaluation")
    for deployed in (milestone_only_controller, behavior_only_controller):
        if deployed.query_id != query_id or deployed.query_protocol != query_spec.protocol:
            raise ValueError("deployed baseline controller query identity does not match Proposed")
    selector = None if summarize_only else _fit_online_selector(selector_model_path)
    behavior_only_selector = (
        None
        if summarize_only
        else _fit_behavior_only_selector(behavior_only_selector_model_path)
    )
    pre_run_aas_selector = (
        None
        if summarize_only
        else _load_query_only_selector(pre_run_aas_selector_model_path, query_id=query_id)
    )
    query_feature_rows = {} if summarize_only else _read_external_query_features(query_feature_path, query_id)
    query_sample_rows = (
        {}
        if summarize_only
        else _read_external_query_samples(
            resolved_query_sample_path,
            sample_design_id=query_spec.sample_design_id,
            expected_split=split_name(config),
        )
    )
    if selector is not None and selector.model.query_id != query_id:
        raise ValueError("selector model query_id does not match the requested online evaluation")
    if behavior_only_selector is not None and (
        behavior_only_selector.sbs_algorithm != selector.sbs_algorithm
    ):
        raise ValueError("Query and Behavior-only selectors must use the same frozen SBS")
    calibration = load_matched_random_calibration(
        matched_random_calibration_path,
        query_id=query_id,
        query_protocol=query_spec.protocol,
        feature_group=controller.feature_group,
        selected_model=controller.model_name,
        threshold_mode=threshold_mode,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if summarize_only:
        return _summarize_shards(
            config=config,
            config_path=config_path,
            train_config_path=train_config_path,
            training_summary_path=training_summary_path,
            output_dir=output_dir,
            model_name=model_name,
            threshold_mode=threshold_mode,
            sampling_protocol=sampling_protocol,
            budget_milestone_ratios=budget_milestone_ratios,
            decision_check_frequency=decision_check_frequency,
            controller=controller,
            milestone_only_controller=milestone_only_controller,
            behavior_only_controller=behavior_only_controller,
            random_repetitions=random_repetitions,
            matched_random_calibration_path=matched_random_calibration_path,
            pre_run_aas_selector_model_path=pre_run_aas_selector_model_path,
            behavior_only_selector_model_path=behavior_only_selector_model_path,
            milestone_only_training_summary_path=milestone_only_training_summary_path,
            behavior_only_training_summary_path=behavior_only_training_summary_path,
            only_functions=only_functions,
            only_dimensions=only_dimensions,
            only_seeds=only_seeds,
        )

    if sharded:
        return _evaluate_online_controller_sharded(
            config=config,
            config_path=config_path,
            train_config_path=train_config_path,
            training_summary_path=training_summary_path,
            output_dir=output_dir,
            model_name=model_name,
            threshold_mode=threshold_mode,
            sampling_protocol=sampling_protocol,
            budget_milestone_ratios=budget_milestone_ratios,
            decision_check_frequency=decision_check_frequency,
            controller=controller,
            milestone_only_controller=milestone_only_controller,
            behavior_only_controller=behavior_only_controller,
            selector=selector,
            behavior_only_selector=behavior_only_selector,
            pre_run_aas_selector=pre_run_aas_selector,
            calibration=calibration,
            matched_random_calibration_path=matched_random_calibration_path,
            pre_run_aas_selector_model_path=pre_run_aas_selector_model_path,
            behavior_only_selector_model_path=behavior_only_selector_model_path,
            milestone_only_training_summary_path=milestone_only_training_summary_path,
            behavior_only_training_summary_path=behavior_only_training_summary_path,
            query_feature_rows=query_feature_rows,
            query_sample_rows=query_sample_rows,
            random_repetitions=random_repetitions,
            functions=functions,
            dimensions=dimensions,
            seeds=seeds,
            only_functions=only_functions,
            only_dimensions=only_dimensions,
            only_seeds=only_seeds,
            max_runs=max_runs,
            workers=workers,
            overwrite=overwrite,
        )

    _check_output_paths(output_dir, overwrite)
    rows = []
    run_counter = 0
    for function in functions:
        for dimension in dimensions:
            fe_total = fe_total_for_dimension(config, dimension)
            for seed in seeds:
                if max_runs is not None and run_counter >= max_runs:
                    break
                rows.extend(
                    _evaluate_one_run(
                        config=config,
                        function=function,
                        dimension=dimension,
                        seed=seed,
                        fe_total=fe_total,
                        controller=controller,
                        milestone_only_controller=milestone_only_controller,
                        behavior_only_controller=behavior_only_controller,
                        selector=selector,
                        behavior_only_selector=behavior_only_selector,
                        pre_run_aas_selector=pre_run_aas_selector,
                        calibration=calibration,
                        query_feature_row=_query_feature_row(query_feature_rows, function=function, dimension=dimension),
                        query_sample_row=_query_sample_row(query_sample_rows, function=function, dimension=dimension),
                        sampling_protocol=sampling_protocol,
                        decision_check_frequency=decision_check_frequency,
                        random_repetitions=random_repetitions,
                    )
                )
                run_counter += 1
            if max_runs is not None and run_counter >= max_runs:
                break
        if max_runs is not None and run_counter >= max_runs:
            break

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("online controller evaluation produced no rows")
    return _write_online_summary(
        result=result,
        config_path=config_path,
        train_config_path=train_config_path,
        training_summary_path=training_summary_path,
        output_dir=output_dir,
        model_name=model_name,
        threshold_mode=threshold_mode,
        sampling_protocol=sampling_protocol,
        budget_milestone_ratios=budget_milestone_ratios,
        decision_check_frequency=decision_check_frequency,
        controller=controller,
        milestone_only_controller=milestone_only_controller,
        behavior_only_controller=behavior_only_controller,
        default_algorithm=selector.sbs_algorithm,
        random_repetitions=random_repetitions,
        matched_random_calibration_path=matched_random_calibration_path,
        pre_run_aas_selector_model_path=pre_run_aas_selector_model_path,
        behavior_only_selector_model_path=behavior_only_selector_model_path,
        milestone_only_training_summary_path=milestone_only_training_summary_path,
        behavior_only_training_summary_path=behavior_only_training_summary_path,
        run_mode="single_output",
        shards={},
    )


def _evaluate_online_controller_sharded(
    *,
    config: dict,
    config_path: Path,
    train_config_path: Path,
    training_summary_path: Path,
    output_dir: Path,
    model_name: str,
    threshold_mode: str,
    sampling_protocol: str,
    budget_milestone_ratios: tuple[float, ...],
    decision_check_frequency: str,
    controller: DecisionControllerModel,
    milestone_only_controller: DecisionControllerModel,
    behavior_only_controller: DecisionControllerModel,
    selector: OnlineSelector | None,
    behavior_only_selector: OnlineBehaviorOnlySelector | None,
    pre_run_aas_selector: OnlineQueryOnlySelector | None,
    calibration: MatchedRandomCalibration,
    matched_random_calibration_path: Path,
    pre_run_aas_selector_model_path: Path,
    behavior_only_selector_model_path: Path,
    milestone_only_training_summary_path: Path,
    behavior_only_training_summary_path: Path,
    query_feature_rows: dict[tuple[int, int], dict[str, Any]],
    query_sample_rows: dict[tuple[int, int], dict[str, Any]],
    random_repetitions: int,
    functions: list[int],
    dimensions: list[int],
    seeds: list[int],
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    only_seeds: list[int] | None,
    max_runs: int | None,
    workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    if selector is None or behavior_only_selector is None or pre_run_aas_selector is None:
        raise ValueError(
            "sharded online evaluation requires Query, Behavior-only, and pre-run query-only selectors"
        )
    jobs = []
    skipped_existing_shards = 0
    assigned_base_runs = 0
    for function in functions:
        for dimension in dimensions:
            if max_runs is not None and assigned_base_runs >= max_runs:
                break
            shard_dir = _shard_output_dir(output_dir, str(config["suite"]).lower(), function, dimension)
            shard_path = shard_dir / "online_policy_runs.parquet"
            if shard_path.exists() and not overwrite:
                print(f"skip existing online evaluation shard {shard_path}")
                skipped_existing_shards += 1
                continue

            shard_seeds = list(seeds)
            if max_runs is not None:
                remaining = max_runs - assigned_base_runs
                shard_seeds = shard_seeds[:remaining]
            if not shard_seeds:
                continue
            jobs.append(
                {
                    "config": config,
                    "function": int(function),
                    "dimension": int(dimension),
                    "seeds": [int(seed) for seed in shard_seeds],
                    "controller": controller,
                    "milestone_only_controller": milestone_only_controller,
                    "behavior_only_controller": behavior_only_controller,
                    "selector": selector,
                    "behavior_only_selector": behavior_only_selector,
                    "pre_run_aas_selector": pre_run_aas_selector,
                    "calibration": calibration,
                    "query_feature_row": _query_feature_row(query_feature_rows, function=function, dimension=dimension),
                    "query_sample_row": _query_sample_row(query_sample_rows, function=function, dimension=dimension),
                    "sampling_protocol": sampling_protocol,
                    "decision_check_frequency": decision_check_frequency,
                    "random_repetitions": int(random_repetitions),
                    "output_dir": output_dir,
                    "overwrite": bool(overwrite),
                }
            )
            assigned_base_runs += len(shard_seeds)
        if max_runs is not None and assigned_base_runs >= max_runs:
            break

    shard_results = _run_shard_jobs(jobs, workers)
    written_shards = sum(1 for result in shard_results if result["status"] == "written")
    worker_skipped_existing_shards = sum(1 for result in shard_results if result["status"] == "skipped_existing")
    skipped_existing_shards += worker_skipped_existing_shards
    executed_base_runs = sum(int(result["base_runs_executed"]) for result in shard_results)

    print(
        "finished sharded online evaluation: "
        f"{written_shards} written shards, "
        f"{skipped_existing_shards} existing shards skipped, "
        f"{executed_base_runs} base runs executed, "
        f"{workers} worker(s)"
    )
    return _summarize_shards(
        config=config,
        config_path=config_path,
        train_config_path=train_config_path,
        training_summary_path=training_summary_path,
        output_dir=output_dir,
        model_name=model_name,
        threshold_mode=threshold_mode,
        sampling_protocol=sampling_protocol,
        budget_milestone_ratios=budget_milestone_ratios,
        decision_check_frequency=decision_check_frequency,
        controller=controller,
        milestone_only_controller=milestone_only_controller,
        behavior_only_controller=behavior_only_controller,
        random_repetitions=random_repetitions,
        matched_random_calibration_path=matched_random_calibration_path,
        pre_run_aas_selector_model_path=pre_run_aas_selector_model_path,
        behavior_only_selector_model_path=behavior_only_selector_model_path,
        milestone_only_training_summary_path=milestone_only_training_summary_path,
        behavior_only_training_summary_path=behavior_only_training_summary_path,
        only_functions=only_functions,
        only_dimensions=only_dimensions,
        only_seeds=only_seeds,
        shard_run_summary={
            "written_shards": int(written_shards),
            "skipped_existing_shards": int(skipped_existing_shards),
            "executed_base_runs": int(executed_base_runs),
            "submitted_shards": int(len(jobs)),
            "workers": int(workers),
        },
    )


def _run_shard_jobs(jobs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    if not jobs:
        return []
    if workers == 1:
        results = []
        for job in jobs:
            result = _evaluate_online_controller_shard(job)
            _print_shard_result(result)
            results.append(result)
        return results

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_evaluate_online_controller_shard, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            _print_shard_result(result)
            results.append(result)
    return results


def _evaluate_online_controller_shard(job: dict[str, Any]) -> dict[str, Any]:
    config = job["config"]
    function = int(job["function"])
    dimension = int(job["dimension"])
    seeds = [int(seed) for seed in job["seeds"]]
    output_dir = Path(job["output_dir"])
    shard_dir = _shard_output_dir(output_dir, str(config["suite"]).lower(), function, dimension)
    shard_path = shard_dir / "online_policy_runs.parquet"
    if shard_path.exists() and not bool(job["overwrite"]):
        return {
            "status": "skipped_existing",
            "path": str(shard_path),
            "rows": 0,
            "base_runs_executed": 0,
        }

    shard_rows = []
    fe_total = fe_total_for_dimension(config, dimension)
    for seed in seeds:
        shard_rows.extend(
            _evaluate_one_run(
                config=config,
                function=function,
                dimension=dimension,
                seed=seed,
                fe_total=fe_total,
                controller=job["controller"],
                milestone_only_controller=job["milestone_only_controller"],
                behavior_only_controller=job["behavior_only_controller"],
                selector=job["selector"],
                behavior_only_selector=job["behavior_only_selector"],
                pre_run_aas_selector=job["pre_run_aas_selector"],
                calibration=job["calibration"],
                query_feature_row=job["query_feature_row"],
                query_sample_row=job["query_sample_row"],
                sampling_protocol=str(job["sampling_protocol"]),
                decision_check_frequency=str(job["decision_check_frequency"]),
                random_repetitions=int(job["random_repetitions"]),
            )
        )
    if not shard_rows:
        return {
            "status": "empty",
            "path": str(shard_path),
            "rows": 0,
            "base_runs_executed": 0,
        }
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_frame = pd.DataFrame(shard_rows)
    _write_frame(shard_frame, shard_dir / "online_policy_runs")
    return {
        "status": "written",
        "path": str(shard_path),
        "rows": int(len(shard_frame)),
        "base_runs_executed": int(len(seeds)),
    }


def _print_shard_result(result: dict[str, Any]) -> None:
    if result["status"] == "written":
        print(f"wrote {result['rows']} online policy rows to {result['path']}")
    elif result["status"] == "skipped_existing":
        print(f"skip existing online evaluation shard {result['path']}")
    elif result["status"] == "empty":
        print(f"skip empty online evaluation shard {result['path']}")
    else:
        print(f"finished online evaluation shard {result['path']} with status {result['status']}")


def _summarize_shards(
    *,
    config: dict,
    config_path: Path,
    train_config_path: Path,
    training_summary_path: Path,
    output_dir: Path,
    model_name: str,
    threshold_mode: str,
    sampling_protocol: str,
    budget_milestone_ratios: tuple[float, ...],
    decision_check_frequency: str,
    controller: DecisionControllerModel,
    milestone_only_controller: DecisionControllerModel,
    behavior_only_controller: DecisionControllerModel,
    random_repetitions: int,
    matched_random_calibration_path: Path,
    pre_run_aas_selector_model_path: Path,
    behavior_only_selector_model_path: Path,
    milestone_only_training_summary_path: Path,
    behavior_only_training_summary_path: Path,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    only_seeds: list[int] | None,
    shard_run_summary: dict[str, int] | None = None,
) -> dict[str, Any]:
    shard_paths = _existing_shard_paths(config, output_dir, only_functions, only_dimensions)
    if not shard_paths:
        raise ValueError(f"no online evaluation shard outputs found under {output_dir}")
    frames = [pq.read_table(path).to_pandas() for path in shard_paths]
    result = _normalize_legacy_online_columns(pd.concat(frames, ignore_index=True))
    if only_seeds is not None:
        requested = set(int(seed) for seed in only_seeds)
        result = result[result["seed"].astype(int).isin(requested)].reset_index(drop=True)
    if result.empty:
        raise ValueError("online evaluation shard rows are empty after filtering")
    default_algorithms = sorted(result["default_algorithm"].astype(str).unique().tolist())
    default_algorithm = default_algorithms[0] if len(default_algorithms) == 1 else ",".join(default_algorithms)
    return _write_online_summary(
        result=result,
        config_path=config_path,
        train_config_path=train_config_path,
        training_summary_path=training_summary_path,
        output_dir=output_dir,
        model_name=model_name,
        threshold_mode=threshold_mode,
        sampling_protocol=sampling_protocol,
        budget_milestone_ratios=budget_milestone_ratios,
        decision_check_frequency=decision_check_frequency,
        controller=controller,
        milestone_only_controller=milestone_only_controller,
        behavior_only_controller=behavior_only_controller,
        default_algorithm=default_algorithm,
        random_repetitions=random_repetitions,
        matched_random_calibration_path=matched_random_calibration_path,
        pre_run_aas_selector_model_path=pre_run_aas_selector_model_path,
        behavior_only_selector_model_path=behavior_only_selector_model_path,
        milestone_only_training_summary_path=milestone_only_training_summary_path,
        behavior_only_training_summary_path=behavior_only_training_summary_path,
        run_mode="sharded",
        shards={
            "discovered_shards": int(len(shard_paths)),
            "paths": [str(path) for path in shard_paths],
            **(shard_run_summary or {}),
        },
    )


def _write_online_summary(
    *,
    result: pd.DataFrame,
    config_path: Path,
    train_config_path: Path,
    training_summary_path: Path,
    output_dir: Path,
    model_name: str,
    threshold_mode: str,
    sampling_protocol: str,
    budget_milestone_ratios: tuple[float, ...],
    decision_check_frequency: str,
    controller: DecisionControllerModel,
    milestone_only_controller: DecisionControllerModel,
    behavior_only_controller: DecisionControllerModel,
    default_algorithm: str,
    random_repetitions: int,
    matched_random_calibration_path: Path,
    pre_run_aas_selector_model_path: Path,
    behavior_only_selector_model_path: Path,
    milestone_only_training_summary_path: Path,
    behavior_only_training_summary_path: Path,
    run_mode: str,
    shards: dict[str, Any],
) -> dict[str, Any]:
    result = _normalize_legacy_online_columns(result)
    policy_runs = _online_policy_run_frame(result)
    policy_summary = _policy_summary(policy_runs)
    relative_summary = _relative_summary(policy_runs)
    random_repetition_summary = _random_repetition_summary(result)

    _write_frame(result, output_dir / "online_policy_runs")
    _write_frame(policy_runs, output_dir / "online_policy_endpoints")
    _write_frame(policy_summary, output_dir / "online_policy_summary")
    _write_frame(relative_summary, output_dir / "online_relative_summary")
    _write_frame(random_repetition_summary, output_dir / "online_random_repetition_summary")

    base_runs = int(result[["problem_id", "dimension", "seed"]].drop_duplicates().shape[0])
    policy_timeouts = sorted(
        float(value) for value in result["policy_timeout_seconds"].unique()
    )
    if (
        len(policy_timeouts) != 1
        or not np.isfinite(policy_timeouts[0])
        or policy_timeouts[0] <= 0.0
    ):
        raise ValueError(
            "online outputs must use one finite positive policy_timeout_seconds"
        )
    summary = {
        "experiment": "cec_online_controller_evaluation",
        "query_id": str(result["query_id"].iloc[0]),
        "query_protocol": str(result["query_protocol"].iloc[0]),
        "sample_design_id": str(result["sample_design_id"].iloc[0]),
        "run_mode": run_mode,
        "config": str(config_path),
        "train_config": str(train_config_path),
        "training_summary": str(training_summary_path),
        "milestone_only_training_summary": str(milestone_only_training_summary_path),
        "model_name": model_name,
        "threshold_mode": threshold_mode,
        "sampling_protocol": sampling_protocol,
        "budget_milestone_ratios": [float(value) for value in budget_milestone_ratios],
        "decision_check_frequency": decision_check_frequency,
        "mean_decision_check_count": float(policy_runs["decision_check_count"].mean()),
        "threshold": float(controller.threshold),
        "feature_columns": controller.feature_columns,
        "default_algorithm": default_algorithm,
        "random_repetitions": random_repetitions,
        "matched_random_calibration": str(matched_random_calibration_path),
        "pre_run_aas_selector_model": str(pre_run_aas_selector_model_path),
        "behavior_only_selector_model": str(behavior_only_selector_model_path),
        "behavior_only_training_summary": str(behavior_only_training_summary_path),
        "behavior_only_threshold_mode": behavior_only_controller.threshold_mode,
        "milestone_only_threshold_mode": milestone_only_controller.threshold_mode,
        "milestone_only_threshold": float(milestone_only_controller.threshold),
        "policy_unit": "trajectory_first_trigger",
        "scientific_endpoint_stage": "one_pre_specified_stage_a_run_per_policy_path",
        "timing_repetitions": TIMING_REPETITIONS,
        "timing_order_protocol": TIMING_ORDER_PROTOCOL,
        "timing_replay_status_protocol": "stage_b_completed_timed_out_failed_v1",
        "policy_timeout_seconds": policy_timeouts[0],
        "runtime_measurement_scope": "full_run_fe0_to_terminal",
        "decision_state_future_path_runtime_measured": False,
        "raw_timing_rows": int(len(result)),
        "policy_endpoint_rows": int(len(policy_runs)),
        "base_runs": base_runs,
        "path_failure_weight": float(policy_runs["failure_weight"].astype(float).sum()),
        "timing_replay_instability_weight": float(
            policy_runs["timing_replay_instability_weight"].astype(float).sum()
        ),
        "timing_replay_failure_weight": float(
            policy_runs["timing_replay_failure_weight"].astype(float).sum()
        ),
        "timing_replay_timeout_weight": float(
            policy_runs["timing_replay_timeout_weight"].astype(float).sum()
        ),
        "coverage_weight": float(policy_runs["coverage_weight"].astype(float).sum()),
        "coverage_denominator": int(len(policy_runs)),
        "policies": sorted(result["policy_name"].astype(str).unique().tolist()),
        "shards": shards,
        "outputs": {
            "policy_runs": str(output_dir / "online_policy_runs.parquet"),
            "policy_endpoints": str(output_dir / "online_policy_endpoints.parquet"),
            "policy_summary": str(output_dir / "online_policy_summary.parquet"),
            "relative_summary": str(output_dir / "online_relative_summary.parquet"),
            "random_repetition_summary": str(output_dir / "online_random_repetition_summary.parquet"),
            "report": str(output_dir / "online_controller_evaluation_report.md"),
            "summary": str(output_dir / "online_controller_evaluation_summary.json"),
        },
        "data_leakage_check": {
            "external_rows_used_for_controller_fit": 0,
            "external_rows_used_for_threshold_fit": 0,
            "external_query_features_used_as_controller_input": False,
            "function_id_algorithm_id_or_optimizer_internal_parameters_used_as_controller_input": False,
            "controller_inputs_are_behavior_features_only": True,
        },
        "scope_notes": [
            "Never Query runs the SBS/default optimizer for the full budget without behavior monitoring, query, or selector computation.",
            "Behavior sampling defines the decision opportunities: every emitted dynamic state is also a possible query trigger point.",
            "always_query executes the fixed query at the first emitted state in the frozen sampling protocol.",
            "pre_run_aas_fe0 executes the query at FE=0, uses the separate query-only selector, and natively initializes its selected optimizer with budget B-FE_query.",
            "milestone_only_T0 evaluates X={FE_ratio} only at the 12 frozen budget milestones and uses its own BBOB-train OOF threshold.",
            "A failed called query is charged its FE and measured runtime, then falls back to the SBS/default optimizer without selector inference.",
            "A pre-specified Stage-A run fixes each scientific endpoint; three separate Stage-B replays in cyclic order determine only the wall-clock median.",
            "When the Stage-A scientific path completed, every completed Stage-B replay must reproduce its scientific endpoint; otherwise completed Stage-B replays must agree with one another and never replace the failed Stage-A endpoint.",
            "Stage-B replay status uses mutually exclusive completed/timed_out/failed categories; mixed status is retained as timing_replay_instability for failure sensitivity and is never selectively rerun.",
            "Online wall-clock fields measure the full FE=0-to-terminal run; they are not fold-specific decision-state future-path timings.",
            "runtime_component_sum_diagnostic is never substituted for measured full-run or decision-state wall-clock.",
            "Query samples are evaluated but never inserted into an optimizer population; optimizer-only and all-evaluation terminals are stored separately.",
            "matched_rate_random uses only the frozen BBOB-train OOF run-call rate and trigger-FE distribution.",
            "The 30 matched Random repetitions are averaged within each trajectory before policy aggregation.",
            "This runner does not construct the static per-problem full-budget four-algorithm VBS hindsight reference.",
            "The controller and threshold are frozen from BBOB train artifacts.",
            "In sharded mode, existing function/dimension shard outputs are skipped unless --overwrite is passed.",
            "Parallel workers execute independent function/dimension shards; summary outputs are written after shard completion.",
        ],
    }
    summary_path = output_dir / "online_controller_evaluation_summary.json"
    report_path = output_dir / "online_controller_evaluation_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(
        _markdown_report(
            summary=summary,
            policy_summary=policy_summary,
            relative_summary=relative_summary,
            random_repetition_summary=random_repetition_summary,
        ),
        encoding="utf-8",
    )
    print(f"wrote online policy runs to {output_dir / 'online_policy_runs.parquet'}")
    print(f"wrote online controller report to {report_path}")
    return summary


def _normalize_legacy_online_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "policy_name" in output.columns:
        output["policy_name"] = output["policy_name"].replace(
            {"traditional_aas": "pre_run_aas_fe0"}
        )
    legacy_runtime_columns = {
        "runtime_complete_path": "runtime_full_run_wall_clock",
        "runtime_complete_path_repetitions": "runtime_full_run_wall_clock_repetitions",
        "runtime_complete_path_median": "runtime_full_run_wall_clock_median",
        "skip_runtime_complete_path_median": "skip_runtime_full_run_wall_clock_median",
    }
    for legacy, active in legacy_runtime_columns.items():
        if legacy in output.columns and active not in output.columns:
            output = output.rename(columns={legacy: active})
    output = output.drop(
        columns=[column for column in ("runtime_total", "runtime_total_median") if column in output.columns]
    )
    if "runtime_measurement_scope" not in output.columns:
        output["runtime_measurement_scope"] = "full_run_fe0_to_terminal"
    if "decision_state_future_path_runtime_measured" not in output.columns:
        output["decision_state_future_path_runtime_measured"] = False
    if "runtime_decision_state_future_path_wall_clock" not in output.columns:
        output["runtime_decision_state_future_path_wall_clock"] = None
    return output


def _shard_output_dir(output_dir: Path, suite: str, function: int, dimension: int) -> Path:
    return output_dir / function_id_name(suite, function) / f"dimension_{int(dimension)}"


def _existing_shard_paths(
    config: dict,
    output_dir: Path,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
) -> list[Path]:
    suite = str(config["suite"]).lower()
    paths = []
    for function in selected_functions(config, only_functions):
        for dimension in selected_dimensions(config, only_dimensions):
            path = _shard_output_dir(output_dir, suite, function, dimension) / "online_policy_runs.parquet"
            if path.exists():
                paths.append(path)
    return sorted(paths)


def _check_output_paths(output_dir: Path, overwrite: bool) -> None:
    outputs = (
        output_dir / "online_policy_runs.csv",
        output_dir / "online_policy_runs.parquet",
        output_dir / "online_policy_endpoints.csv",
        output_dir / "online_policy_endpoints.parquet",
        output_dir / "online_policy_summary.csv",
        output_dir / "online_policy_summary.parquet",
        output_dir / "online_relative_summary.csv",
        output_dir / "online_relative_summary.parquet",
        output_dir / "online_random_repetition_summary.csv",
        output_dir / "online_random_repetition_summary.parquet",
        output_dir / "online_controller_evaluation_report.md",
        output_dir / "online_controller_evaluation_summary.json",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"online evaluation outputs already exist; pass --overwrite: {existing[0]}")


def _selected_seeds(config: dict, only_seeds: list[int] | None) -> list[int]:
    seeds = as_int_list(config, "seeds")
    if only_seeds is None:
        return seeds
    requested = set(int(seed) for seed in only_seeds)
    missing = sorted(requested.difference(seeds))
    if missing:
        raise ValueError(f"requested seeds are not in config: {missing}")
    return [seed for seed in seeds if seed in requested]


def _load_controller(training_summary_path: Path, model_name: str, threshold_mode: str) -> DecisionControllerModel:
    summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    if not summary.get("feature_group"):
        raise ValueError("Decision training summary does not define feature_group")
    feature_columns = [str(column) for column in summary.get("feature_columns", [])]
    if not feature_columns or not set(feature_columns).issubset(
        SELECTOR_BEHAVIOR_FEATURE_COLUMNS
    ):
        raise ValueError("controller feature columns must be a non-empty subset of behavior features")
    model_name = resolve_model_name(summary, model_name)
    model_path = _model_path(summary, model_name)
    threshold = _threshold(summary, model_name, threshold_mode)
    model_family = _model_family(summary, model_name)
    return DecisionControllerModel(
        model=joblib.load(model_path),
        model_name=model_name,
        model_family=model_family,
        threshold_mode=threshold_mode,
        threshold=threshold,
        feature_columns=feature_columns,
        training_summary_path=training_summary_path,
        model_path=model_path,
        query_id=str(summary.get("query_id", "")),
        query_protocol=str(summary.get("query_protocol", "")),
        feature_group=str(summary.get("feature_group", "")),
        opportunity_scope=str(summary.get("opportunity_scope", "")),
    )


def _read_external_query_features(path: Path, query_id: str) -> dict[tuple[int, int], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing external query feature table: {path}")
    spec = get_query_spec(query_id)
    frame = pq.read_table(path).to_pandas()
    expected_columns = set(FEATURE_METADATA_COLUMNS) | set(spec.feature_columns)
    observed_columns = set(frame.columns)
    if observed_columns != expected_columns:
        raise ValueError(
            "external query feature table does not exactly match the frozen whitelist; "
            f"missing={sorted(expected_columns - observed_columns)}, "
            f"extra={sorted(observed_columns - expected_columns)}"
        )
    required = {
        "problem_id",
        "function",
        "dimension",
        "query_id",
        "query_protocol",
        "query_preprocessing_id",
        "sample_design_id",
        "sample_seed",
        "runtime_query_sampling",
        "runtime_query_evaluation",
        "runtime_query_feature_computation",
        "runtime_query",
        "feature_status",
        "feature_failure",
        "feature_group_status",
        "additional_function_evaluations",
        "query_feature_columns",
        *spec.feature_columns,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"external query feature table is missing columns: {sorted(missing)}")
    if set(frame["query_id"].astype(str)) != {query_id}:
        raise ValueError("external query feature table contains the wrong query_id")
    if set(frame["query_protocol"].astype(str)) != {spec.protocol}:
        raise ValueError("external query feature table contains the wrong query_protocol")
    if set(frame["query_preprocessing_id"].astype(str)) != {spec.preprocessing_id}:
        raise ValueError("external query feature table contains the wrong preprocessing protocol")
    if set(frame["sample_design_id"].astype(str)) != {spec.sample_design_id}:
        raise ValueError("external query feature table contains the wrong sample design")
    expected_feature_columns = json.dumps(list(spec.feature_columns), ensure_ascii=False)
    if set(frame["query_feature_columns"].astype(str)) != {expected_feature_columns}:
        raise ValueError("external query feature table contains a non-frozen feature-column list")
    if (frame["additional_function_evaluations"].astype(int) != 0).any():
        raise ValueError("external query feature extraction reports additional objective evaluations")
    for row in frame.to_dict(orient="records"):
        group_status = json.loads(str(row["feature_group_status"]))
        if set(group_status) != set(spec.feature_groups):
            raise ValueError("external query feature group status does not cover the frozen groups")
        has_group_failure = any(str(status.get("status")) != "ok" for status in group_status.values())
        expected_status = "failed" if has_group_failure else "ok"
        if str(row["feature_status"]) != expected_status:
            raise ValueError("external query feature_status is inconsistent with group-level status")
    key = ["function", "dimension"]
    if frame.duplicated(key).any():
        raise ValueError("external query feature table contains duplicate function/dimension rows")
    runtimes = frame["runtime_query"].astype(float).to_numpy()
    if not np.isfinite(runtimes).all() or (runtimes < 0.0).any():
        raise ValueError("external query runtime must be finite and non-negative")
    expected_runtime = (
        frame["runtime_query_sampling"].astype(float)
        + frame["runtime_query_evaluation"].astype(float)
        + frame["runtime_query_feature_computation"].astype(float)
    ).to_numpy()
    if not np.allclose(runtimes, expected_runtime, rtol=0.0, atol=1e-12):
        raise ValueError("external runtime_query is not sampling evaluation plus feature computation")
    return {
        (int(row["function"]), int(row["dimension"])): row
        for row in frame.to_dict(orient="records")
    }


def _read_external_query_samples(
    path: Path,
    *,
    sample_design_id: str,
    expected_split: str,
) -> dict[tuple[int, int], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing external query sample table: {path}")
    frame = pq.read_table(path).to_pandas()
    required = {
        *SAMPLE_KEY_COLUMNS,
        "FE_total",
        "sampling_protocol",
        "sample_seed",
        "sample_size",
        "FE_query",
        "benchmark_reference_value",
        "success_gap_target",
        "query_success",
        "query_first_hit_offset",
        "query_best_gap",
        "lower_bounds",
        "upper_bounds",
        "X",
        "y",
        "sample_status",
        "sample_failure",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"external query sample table is missing columns: {sorted(missing)}")
    if set(frame["split"].astype(str)) != {str(expected_split)}:
        raise ValueError("external query sample table contains the wrong split")
    if set(frame["sample_design_id"].astype(str)) != {str(sample_design_id)}:
        raise ValueError("external query sample table contains the wrong sample design")
    if set(frame["sample_status"].astype(str)) != {"ok"}:
        raise ValueError("online evaluation requires complete query sample rows")
    if set(frame["instance"].astype(int)) != {1}:
        raise ValueError("external CEC query samples must use instance=1")
    key = ["function", "dimension"]
    if frame.duplicated(key).any():
        raise ValueError("external query sample table contains duplicate function/dimension rows")
    for row in frame.to_dict(orient="records"):
        _validate_saved_query_sample(row)
    return {
        (int(row["function"]), int(row["dimension"])): row
        for row in frame.to_dict(orient="records")
    }


def _query_feature_row(
    rows: dict[tuple[int, int], dict[str, Any]],
    *,
    function: int,
    dimension: int,
) -> dict[str, Any]:
    key = (int(function), int(dimension))
    if key not in rows:
        raise ValueError(f"missing saved query features for function={function}, dimension={dimension}")
    return rows[key]


def _query_sample_row(
    rows: dict[tuple[int, int], dict[str, Any]],
    *,
    function: int,
    dimension: int,
) -> dict[str, Any]:
    key = (int(function), int(dimension))
    if key not in rows:
        raise ValueError(f"missing saved query sample for function={function}, dimension={dimension}")
    return rows[key]


def _validate_saved_query_sample(row: dict[str, Any]) -> None:
    fe_query = int(row["FE_query"])
    x = np.asarray(row["X"], dtype=float)
    y = np.asarray(row["y"], dtype=float).reshape(-1)
    dimension = int(row["dimension"])
    if x.shape != (fe_query, dimension) or y.shape != (fe_query,):
        raise ValueError("saved query X/y shape does not match FE_query and dimension")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("saved query X/y must be finite")
    reference = float(row["benchmark_reference_value"])
    target = float(row["success_gap_target"])
    best_gap = float(row["query_best_gap"])
    if not np.isfinite(reference) or not np.isfinite(target) or target <= 0.0:
        raise ValueError("saved query endpoint metadata is invalid")
    gaps = np.maximum(y - reference, 0.0)
    hits = np.flatnonzero(gaps <= target)
    expected_offset = int(hits[0] + 1) if hits.size else None
    raw_offset = row["query_first_hit_offset"]
    observed_offset = None if pd.isna(raw_offset) else int(raw_offset)
    if observed_offset is not None and not 1 <= observed_offset <= fe_query:
        raise ValueError("query_first_hit_offset must lie in [1, FE_query]")
    if bool(row["query_success"]) != (observed_offset is not None):
        raise ValueError("query_success and query_first_hit_offset are inconsistent")
    if observed_offset != expected_offset:
        raise ValueError("saved query first-hit offset is inconsistent with y")
    if not np.isfinite(best_gap) or best_gap < 0.0 or not np.isclose(
        best_gap,
        float(np.min(gaps)),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("saved query_best_gap is inconsistent with y")


def _model_path(summary: dict[str, Any], model_name: str) -> Path:
    for artifact in summary.get("model_artifacts", []):
        if str(artifact.get("model_name")) == model_name:
            path = Path(str(artifact.get("model_path")))
            if not path.exists():
                raise FileNotFoundError(f"missing controller model artifact: {path}")
            return path
    raise ValueError(f"controller model artifact not found: {model_name}")


def _threshold(summary: dict[str, Any], model_name: str, threshold_mode: str) -> float:
    threshold_path = Path(str(summary.get("outputs", {}).get("decision_thresholds", "")))
    if not threshold_path.exists():
        raise FileNotFoundError(f"missing decision threshold table: {threshold_path}")
    thresholds = pq.read_table(threshold_path).to_pandas()
    row = thresholds[
        (thresholds["model_name"].astype(str) == model_name)
        & (thresholds["threshold_mode"].astype(str) == threshold_mode)
    ]
    if len(row) != 1:
        raise ValueError(f"expected one threshold row for {model_name}/{threshold_mode}, found {len(row)}")
    if int(row["validation_rows_used_for_threshold_fit"].iloc[0]) != 0:
        raise ValueError("controller threshold must be fit without held-out rows")
    return float(row["threshold"].iloc[0])


def _model_family(summary: dict[str, Any], model_name: str) -> str:
    fit_path = Path(str(summary.get("outputs", {}).get("model_fit_summary", "")))
    if not fit_path.exists():
        return ""
    rows = pq.read_table(fit_path).to_pandas()
    row = rows[rows["model_name"].astype(str) == model_name]
    return str(row["model_family"].iloc[0]) if len(row) else ""


def _fit_online_selector(selector_model_path: Path) -> OnlineSelector:
    model = load_selector_model(selector_model_path)
    spec = get_query_spec(model.query_id)
    if (
        model.query_protocol != spec.protocol
        or model.sample_design_id != spec.sample_design_id
        or tuple(model.query_feature_columns) != spec.feature_columns
    ):
        raise ValueError("online Query selector identity does not match its canonical query")
    return OnlineSelector(model=model)


def _fit_behavior_only_selector(selector_model_path: Path) -> OnlineBehaviorOnlySelector:
    model = load_selector_model(selector_model_path)
    if model.selector_input_mode != BEHAVIOR_ONLY_FULL_BUDGET_INPUT:
        raise ValueError("Behavior-only selector uses the wrong input mode")
    if model.action_budget_mode != FULL_REMAINING_BUDGET:
        raise ValueError("Behavior-only selector must preserve the full remaining FE budget")
    if model.query_id != NOT_APPLICABLE or model.query_feature_columns:
        raise ValueError("Behavior-only selector must not consume query identity or query features")
    expected_columns = (*SELECTOR_BEHAVIOR_FEATURE_COLUMNS, "remaining_budget_ratio")
    if tuple(model.feature_columns) != expected_columns:
        raise ValueError("Behavior-only selector feature contract is inconsistent")
    return OnlineBehaviorOnlySelector(model=model)


def _load_behavior_only_controller(
    training_summary_path: Path,
    model_name: str,
) -> DecisionControllerModel:
    summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    selected = resolve_model_name(summary, model_name)
    artifacts = summary.get("behavior_only_model_artifacts", [])
    artifact = next(
        (row for row in artifacts if str(row.get("model_name")) == selected),
        None,
    )
    if artifact is None:
        raise ValueError(f"Behavior-only Decision artifact not found for model: {selected}")
    model_path = Path(str(artifact.get("model_path", "")))
    if not model_path.exists():
        raise FileNotFoundError(f"missing Behavior-only Decision model artifact: {model_path}")
    feature_columns = [str(column) for column in summary.get("feature_columns", [])]
    if not feature_columns or not set(feature_columns).issubset(SELECTOR_BEHAVIOR_FEATURE_COLUMNS):
        raise ValueError("Behavior-only controller inputs must be Behavior features only")
    threshold_mode = str(
        summary.get("behavior_only_threshold_mode", BEHAVIOR_ONLY_THRESHOLD_MODE)
    )
    if threshold_mode != BEHAVIOR_ONLY_THRESHOLD_MODE:
        raise ValueError("Behavior-only controller uses an unsupported threshold mode")
    threshold = _threshold(summary, selected, threshold_mode)
    return DecisionControllerModel(
        model=joblib.load(model_path),
        model_name=selected,
        model_family=_model_family(summary, selected),
        threshold_mode=threshold_mode,
        threshold=threshold,
        feature_columns=feature_columns,
        training_summary_path=training_summary_path,
        model_path=model_path,
        query_id=str(summary.get("query_id", "")),
        query_protocol=str(summary.get("query_protocol", "")),
        feature_group=str(summary.get("feature_group", "")),
        opportunity_scope=str(summary.get("opportunity_scope", "")),
    )


def _load_query_only_selector(
    selector_model_path: Path,
    *,
    query_id: str,
) -> OnlineQueryOnlySelector:
    if not selector_model_path.exists():
        raise FileNotFoundError(
            f"missing separately trained pre-run AAS query-only selector: {selector_model_path}"
        )
    artifact = joblib.load(selector_model_path)
    required = (
        "protocol",
        "model",
        "target_algorithms",
        "query_id",
        "query_protocol",
        "sample_design_id",
        "query_feature_columns",
        "feature_columns",
        "selector_target_transform",
    )
    missing = [name for name in required if not hasattr(artifact, name)]
    if missing:
        raise ValueError(f"query-only selector artifact is missing attributes: {missing}")
    spec = get_query_spec(query_id)
    if str(artifact.protocol) != QUERY_ONLY_SELECTOR_PROTOCOL:
        raise ValueError("pre-run AAS selector does not use the query-only observed-action-loss protocol")
    if (
        str(artifact.query_id) != query_id
        or str(artifact.query_protocol) != spec.protocol
        or str(artifact.sample_design_id) != spec.sample_design_id
    ):
        raise ValueError("pre-run AAS selector does not match the requested query")
    if tuple(artifact.query_feature_columns) != spec.feature_columns:
        raise ValueError("pre-run AAS selector query columns do not match the canonical query ID")
    expected_features = spec.feature_columns + ("remaining_budget_ratio",)
    if tuple(artifact.feature_columns) != expected_features:
        raise ValueError("pre-run AAS selector must use only query features and remaining_budget_ratio")
    targets = tuple(str(value) for value in artifact.target_algorithms)
    if len(targets) != 4 or len(set(targets)) != 4 or set(targets) != {"de", "pso", "cmaes", "shade"}:
        raise ValueError("pre-run AAS selector must predict the four unique portfolio algorithms")
    return OnlineQueryOnlySelector(artifact=artifact)


def _budget_milestone_ratios(config: dict, sampling_protocol: str) -> tuple[float, ...]:
    if sampling_protocol == DEFAULT_SAMPLING_PROTOCOL:
        if str(config.get("sampling_protocol", "")) != SAMPLING_PROTOCOL:
            raise ValueError("online config does not use the frozen dynamic sampling protocol")
        return get_sampling_spec(sampling_protocol).budget_milestone_ratios
    raise ValueError(f"unsupported sampling protocol: {sampling_protocol}")


def _decision_check_frequency(sampling_protocol: str) -> str:
    if sampling_protocol == DEFAULT_SAMPLING_PROTOCOL:
        return "dynamic_budget_milestones_and_state_events"
    raise ValueError(f"unsupported sampling protocol: {sampling_protocol}")


def _sampling_output_dir(output_dir: Path, sampling_protocol: str) -> Path:
    if sampling_protocol == DEFAULT_SAMPLING_PROTOCOL:
        return output_dir
    return output_dir


def _evaluate_one_run(
    *,
    config: dict,
    function: int,
    dimension: int,
    seed: int,
    fe_total: int,
    controller: DecisionControllerModel,
    milestone_only_controller: DecisionControllerModel,
    behavior_only_controller: DecisionControllerModel,
    selector: OnlineSelector,
    behavior_only_selector: OnlineBehaviorOnlySelector,
    pre_run_aas_selector: OnlineQueryOnlySelector,
    calibration: MatchedRandomCalibration,
    query_feature_row: dict[str, Any],
    query_sample_row: dict[str, Any],
    sampling_protocol: str,
    decision_check_frequency: str,
    random_repetitions: int,
) -> list[dict[str, Any]]:
    suite = str(config["suite"]).lower()
    if int(config["timing_repetitions"]) != TIMING_REPETITIONS:
        raise ValueError(f"online timing requires exactly {TIMING_REPETITIONS} repetitions")
    if str(config["timing_order_protocol"]) != TIMING_ORDER_PROTOCOL:
        raise ValueError(f"online timing requires {TIMING_ORDER_PROTOCOL}")

    identity_problem = make_problem(
        {"suite": suite, "function": function, "instance": 1, "dimension": dimension}
    )
    try:
        problem_id = str(identity_problem.problem_id)
        problem_family = str(identity_problem.family)
    finally:
        identity_problem.close()
    _validate_query_feature_problem(
        query_feature_row,
        problem_id,
        fe_query=_query_fe(selector, dimension, fe_total),
        fe_total=fe_total,
    )
    _validate_query_sample_problem(
        query_sample_row,
        query_feature_row=query_feature_row,
        problem_id=problem_id,
        fe_query=_query_fe(selector, dimension, fe_total),
        fe_total=fe_total,
        success_gap_target=float(config["success_gap_target"]),
    )

    path_specs: list[dict[str, Any]] = [
        {"policy_name": name, "random_repetition": None}
        for name in ONLINE_POLICY_PATHS
    ]
    for repetition in range(random_repetitions):
        target_ratio = matched_random_target(
            calibration,
            problem_id=problem_id,
            prefix_algorithm=selector.sbs_algorithm,
            run_seed=seed,
            dimension=dimension,
            repetition=repetition,
        )
        path_specs.append(
            {
                "policy_name": "matched_rate_random",
                "random_repetition": int(repetition),
                "matched_trigger_fe_ratio": target_ratio,
            }
        )

    base_order = _online_timing_base_order(
        path_count=len(path_specs),
        function=function,
        dimension=dimension,
        seed=seed,
        random_repetitions=random_repetitions,
    )
    scientific_rows: dict[tuple[str, int | None], dict[str, Any]] = {}
    for scientific_order_position, spec_index in enumerate(base_order):
        spec = path_specs[int(spec_index)]
        scientific = _execute_online_policy_path(
            config=config,
            suite=suite,
            function=function,
            dimension=dimension,
            seed=seed,
            fe_total=fe_total,
            controller=controller,
            milestone_only_controller=milestone_only_controller,
            behavior_only_controller=behavior_only_controller,
            selector=selector,
            behavior_only_selector=behavior_only_selector,
            pre_run_aas_selector=pre_run_aas_selector,
            query_feature_row=query_feature_row,
            query_sample_row=query_sample_row,
            policy_spec=spec,
            sampling_protocol=sampling_protocol,
            decision_check_frequency=decision_check_frequency,
        )
        scientific.update(
            {
                "split": split_name(config),
                "suite": suite,
                "problem_id": problem_id,
                "family": problem_family,
                "function": int(function),
                "dimension": int(dimension),
                "seed": int(seed),
                "scientific_endpoint_stage": "pre_specified_stage_a_run",
                "scientific_run_order_position": int(scientific_order_position),
            }
        )
        scientific_key = (
            str(scientific["policy_name"]),
            scientific.get("random_repetition"),
        )
        if scientific_key in scientific_rows:
            raise RuntimeError(f"duplicate Stage-A online scientific path: {scientific_key}")
        scientific_rows[scientific_key] = scientific

    rows: list[dict[str, Any]] = []
    for timing_repetition in range(TIMING_REPETITIONS):
        rotation = timing_repetition % len(base_order)
        order = base_order[rotation:] + base_order[:rotation]
        for timing_order_position, spec_index in enumerate(order):
            spec = path_specs[int(spec_index)]
            row = _execute_online_policy_path(
                config=config,
                suite=suite,
                function=function,
                dimension=dimension,
                seed=seed,
                fe_total=fe_total,
                controller=controller,
                milestone_only_controller=milestone_only_controller,
                behavior_only_controller=behavior_only_controller,
                selector=selector,
                behavior_only_selector=behavior_only_selector,
                pre_run_aas_selector=pre_run_aas_selector,
                query_feature_row=query_feature_row,
                query_sample_row=query_sample_row,
                policy_spec=spec,
                sampling_protocol=sampling_protocol,
                decision_check_frequency=decision_check_frequency,
            )
            row.update(
                {
                    "split": split_name(config),
                    "suite": suite,
                    "problem_id": problem_id,
                    "family": problem_family,
                    "function": int(function),
                    "dimension": int(dimension),
                    "seed": int(seed),
                    "timing_repetition": int(timing_repetition),
                    "timing_order_position": int(timing_order_position),
                    "timing_order_protocol": TIMING_ORDER_PROTOCOL,
                    "timing_repetitions": TIMING_REPETITIONS,
                    "execution_stage": "stage_b_timing_replay",
                }
            )
            rows.append(row)
    return _attach_full_run_timing(rows, scientific_rows=scientific_rows)


def _online_timing_base_order(
    *,
    path_count: int,
    function: int,
    dimension: int,
    seed: int,
    random_repetitions: int,
) -> tuple[int, ...]:
    if path_count <= 0:
        raise ValueError("online full-run timing requires at least one policy path")
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                ONLINE_TIMING_STREAM_CODE,
                int(function),
                1,
                int(dimension),
                int(random_repetitions),
                int(path_count),
            ]
        )
    )
    return tuple(int(value) for value in rng.permutation(path_count))


def _tracked_problem(
    problem: Problem,
    *,
    tracker: PathEvaluationTracker,
    deadline: float,
) -> Problem:
    def objective(population: np.ndarray) -> np.ndarray:
        points = np.asarray(population, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        values: list[float] = []
        for point in points:
            if perf_counter() >= deadline:
                raise TimeoutError(
                    "online policy exceeded policy_timeout_seconds before evaluation"
                )
            value = float(problem.evaluate(point)[0])
            tracker.observe(np.asarray([value], dtype=float))
            values.append(value)
            if perf_counter() >= deadline:
                raise TimeoutError(
                    "online policy exceeded policy_timeout_seconds after evaluation"
                )
        return np.asarray(values, dtype=float)

    return Problem(
        problem_id=problem.problem_id,
        function_id=problem.function_id,
        family=problem.family,
        dimension=problem.dimension,
        suite_code=problem.suite_code,
        function_number=problem.function_number,
        instance_number=problem.instance_number,
        bounds=problem.bounds.copy(),
        objective=objective,
        reference_value=problem.reference_value,
        close_callback=problem.close,
    )


def _execute_online_policy_path(
    *,
    config: dict,
    suite: str,
    function: int,
    dimension: int,
    seed: int,
    fe_total: int,
    controller: DecisionControllerModel,
    milestone_only_controller: DecisionControllerModel,
    behavior_only_controller: DecisionControllerModel,
    selector: OnlineSelector,
    behavior_only_selector: OnlineBehaviorOnlySelector,
    pre_run_aas_selector: OnlineQueryOnlySelector,
    query_feature_row: dict[str, Any],
    query_sample_row: dict[str, Any],
    policy_spec: dict[str, Any],
    sampling_protocol: str,
    decision_check_frequency: str,
) -> dict[str, Any]:
    policy_name = str(policy_spec["policy_name"])
    random_repetition = policy_spec.get("random_repetition")
    path_started = perf_counter()
    timeout_seconds = float(config["policy_timeout_seconds"])
    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("policy_timeout_seconds must be finite and positive")
    deadline = path_started + timeout_seconds
    reference = float(query_sample_row["benchmark_reference_value"])
    tracker = PathEvaluationTracker(
        benchmark_reference_value=reference,
        success_gap_target=float(config["success_gap_target"]),
        planned_total_fe=int(fe_total),
    )
    default_algorithm = selector.sbs_algorithm
    tracker.remember(
        prefix_algorithm=default_algorithm,
        default_algorithm=default_algorithm,
        no_query_algorithm=default_algorithm,
        selected_algorithm=default_algorithm,
        selected_action="continue_current",
        selected_equals_default=True,
        selected_equals_prefix=True,
        handoff_required=False,
        handoff_type="native_optimizer_state",
        query_transition_mode="native_optimizer_state",
        optimizer_transition_mode="native_optimizer_state",
        selector_status="not_used",
        selector_remaining_budget_ratio=None,
        query_called=False,
        query_feature_status="not_called",
        query_feature_failure="",
        query_failure_fallback=False,
        trigger_FE=None,
        trigger_FE_ratio=None,
        decision_score=None,
        decision_threshold=None,
        decision_check_count=0,
        runtime_probe=0.0,
        runtime_query_sampling=0.0,
        runtime_query_evaluation=0.0,
        runtime_query_feature_computation=0.0,
        runtime_query=0.0,
        runtime_selection=0.0,
        runtime_handoff=0.0,
        runtime_after=0.0,
        **_empty_trigger_sampling_metadata(),
    )
    base_problem: Problem | None = None
    problem: Problem | None = None
    failure_class = ""
    failure_message = ""
    try:
        base_problem = make_problem(
            {
                "suite": suite,
                "function": function,
                "instance": 1,
                "dimension": dimension,
            }
        )
        if base_problem.reference_value is None or not np.isclose(
            float(base_problem.reference_value), reference, rtol=0.0, atol=1e-12
        ):
            raise ValueError("online benchmark reference does not match the query sample artifact")
        problem = _tracked_problem(base_problem, tracker=tracker, deadline=deadline)
        if policy_name == "sbs_no_query":
            row = _run_no_query_policy(
                problem=problem,
                tracker=tracker,
                config=config,
                seed=seed,
                fe_total=fe_total,
                selector=selector,
                policy_name=policy_name,
                sampling_protocol=sampling_protocol,
                decision_check_frequency=decision_check_frequency,
                repetition=random_repetition,
            )
        elif policy_name in {"pre_run_aas_fe0", "traditional_aas"}:
            row = _run_pre_run_aas_policy(
                problem=problem,
                tracker=tracker,
                deadline=deadline,
                config=config,
                seed=seed,
                fe_total=fe_total,
                selector=selector,
                pre_run_aas_selector=pre_run_aas_selector,
                query_feature_row=query_feature_row,
                query_sample_row=query_sample_row,
                sampling_protocol=sampling_protocol,
                decision_check_frequency="pre_run_fe0_query",
            )
        elif policy_name in {
            "matched_trigger_behavior_only",
            "self_thresholded_behavior_only",
        }:
            matched_controller = (
                controller
                if policy_name == "matched_trigger_behavior_only"
                else behavior_only_controller
            )
            row = _run_behavior_only_policy(
                problem=problem,
                tracker=tracker,
                config=config,
                function=function,
                seed=seed,
                fe_total=fe_total,
                query_selector=selector,
                behavior_selector=behavior_only_selector,
                controller=matched_controller,
                policy_name=policy_name,
                trigger_mode="controller",
                sampling_protocol=sampling_protocol,
                decision_check_frequency=(
                    "proposed_matched_first_trigger"
                    if policy_name == "matched_trigger_behavior_only"
                    else decision_check_frequency
                ),
                repetition=random_repetition,
            )
        elif policy_name in {
            "always_query",
            "milestone_only_T0",
            "current_controller",
            "matched_rate_random",
        }:
            target_ratio = policy_spec.get("matched_trigger_fe_ratio")
            if policy_name == "matched_rate_random" and target_ratio is None:
                row = _run_no_query_policy(
                    problem=problem,
                    tracker=tracker,
                    config=config,
                    seed=seed,
                    fe_total=fe_total,
                    selector=selector,
                    policy_name=policy_name,
                    sampling_protocol=sampling_protocol,
                    decision_check_frequency="train_oof_matched_first_trigger",
                    repetition=random_repetition,
                )
            else:
                trigger_mode = {
                    "always_query": "first_checkpoint",
                    "milestone_only_T0": "controller",
                    "current_controller": "controller",
                    "matched_rate_random": "matched_random",
                }[policy_name]
                row = _run_threshold_policy(
                    problem=problem,
                    tracker=tracker,
                    deadline=deadline,
                    config=config,
                    function=function,
                    seed=seed,
                    fe_total=fe_total,
                    selector=selector,
                    query_feature_row=query_feature_row,
                    query_sample_row=query_sample_row,
                    controller=(
                        controller
                        if policy_name == "current_controller"
                        else milestone_only_controller
                        if policy_name == "milestone_only_T0"
                        else None
                    ),
                    policy_name=policy_name,
                    trigger_mode=trigger_mode,
                    matched_trigger_fe_ratio=target_ratio,
                    sampling_protocol=sampling_protocol,
                    decision_check_frequency=(
                        "train_oof_matched_first_trigger"
                        if policy_name == "matched_rate_random"
                        else "budget_milestones_only"
                        if policy_name == "milestone_only_T0"
                        else decision_check_frequency
                    ),
                    decision_opportunity_scope=(
                        "milestone_only"
                        if policy_name == "milestone_only_T0"
                        else "all_accepted"
                    ),
                    repetition=random_repetition,
                )
        else:
            raise ValueError(f"unknown online policy path: {policy_name}")
    except Exception as exc:
        failure_class = "timeout" if isinstance(exc, TimeoutError) else type(exc).__name__
        failure_message = str(exc)[:500]
        row = _failed_online_policy_row(
            policy_name=policy_name,
            random_repetition=random_repetition,
            selector=selector,
            tracker=tracker,
            fe_total=fe_total,
            sampling_protocol=sampling_protocol,
            decision_check_frequency=decision_check_frequency,
            failure_class=failure_class,
            failure_message=failure_message,
        )
    finally:
        if problem is not None:
            problem.close()
        elif base_problem is not None:
            base_problem.close()
    runtime_full_run_wall_clock = float(perf_counter() - path_started)
    timed_out = failure_class == "timeout" or runtime_full_run_wall_clock > timeout_seconds
    if timed_out and not failure_class:
        failure_class = "timeout"
        failure_message = "online policy exceeded policy_timeout_seconds"
    return _finalize_online_policy_row(
        row=row,
        tracker=tracker,
        config=config,
        runtime_full_run_wall_clock=runtime_full_run_wall_clock,
        timed_out=timed_out,
        failure_class=failure_class,
        failure_message=failure_message,
    )


def _failed_online_policy_row(
    *,
    policy_name: str,
    random_repetition: int | None,
    selector: OnlineSelector,
    tracker: PathEvaluationTracker,
    fe_total: int,
    sampling_protocol: str,
    decision_check_frequency: str,
    failure_class: str,
    failure_message: str,
) -> dict[str, Any]:
    default_algorithm = selector.sbs_algorithm
    context = tracker.execution_context
    category = "controller" if policy_name == "current_controller" else "baseline"
    if policy_name == "matched_trigger_behavior_only":
        category = "diagnostic"
    runtime_probe = float(context.get("runtime_probe", 0.0))
    runtime_query_sampling = float(context.get("runtime_query_sampling", 0.0))
    runtime_query_evaluation = float(context.get("runtime_query_evaluation", 0.0))
    runtime_query_feature_computation = float(
        context.get("runtime_query_feature_computation", 0.0)
    )
    runtime_query = float(
        context.get(
            "runtime_query",
            runtime_query_sampling
            + runtime_query_evaluation
            + runtime_query_feature_computation,
        )
    )
    runtime_selection = float(context.get("runtime_selection", 0.0))
    runtime_handoff = float(context.get("runtime_handoff", 0.0))
    runtime_after = float(context.get("runtime_after", 0.0))
    prefix_algorithm = str(context.get("prefix_algorithm", default_algorithm))
    selected_algorithm = str(context.get("selected_algorithm", default_algorithm))
    selected_equals_prefix = bool(
        context.get("selected_equals_prefix", selected_algorithm == prefix_algorithm)
    )
    handoff_required = bool(context.get("handoff_required", not selected_equals_prefix))
    transition_mode = str(
        context.get(
            "optimizer_transition_mode",
            "native_optimizer_state"
            if selected_equals_prefix
            else "population_transfer_initialization",
        )
    )
    trigger_metadata = {
        column: context.get(column)
        for column in _empty_trigger_sampling_metadata()
    }
    observed_best = (
        float(tracker.best_all) if np.isfinite(tracker.best_all) else float("nan")
    )
    row = {
        "policy_name": policy_name,
        "policy_category": category,
        "sampling_protocol": sampling_protocol,
        "decision_check_frequency": decision_check_frequency,
        "random_repetition": random_repetition,
        "prefix_algorithm": prefix_algorithm,
        "default_algorithm": default_algorithm,
        "no_query_algorithm": default_algorithm,
        "selected_algorithm": selected_algorithm,
        "selected_action": str(
            context.get(
                "selected_action",
                "continue_current" if selected_equals_prefix else selected_algorithm,
            )
        ),
        "selected_equals_default": bool(
            context.get("selected_equals_default", selected_algorithm == default_algorithm)
        ),
        "selected_equals_prefix": selected_equals_prefix,
        "handoff_required": handoff_required,
        "handoff_type": str(context.get("handoff_type", transition_mode)),
        "query_transition_mode": str(
            context.get("query_transition_mode", transition_mode)
        ),
        "selector_status": str(context.get("selector_status", "path_failure")),
        "optimizer_transition_mode": transition_mode,
        "selector_target_transform": selector.model.selector_target_transform,
        "selector_remaining_budget_ratio": context.get(
            "selector_remaining_budget_ratio"
        ),
        "query_called": bool(
            context.get("query_called", bool(tracker.query_evaluations))
        ),
        "query_id": selector.model.query_id,
        "query_protocol": selector.model.query_protocol,
        "sample_design_id": selector.model.sample_design_id,
        "query_feature_status": str(context.get("query_feature_status", "not_called")),
        "query_feature_failure": str(
            context.get("query_feature_failure") or failure_message
        ),
        "query_failure_fallback": bool(context.get("query_failure_fallback", False)),
        "query_success": bool(context.get("query_success", False)),
        "query_first_hit_offset": context.get("query_first_hit_offset"),
        "query_best_gap": context.get("query_best_gap"),
        "trigger_FE": context.get("trigger_FE"),
        "trigger_FE_ratio": context.get("trigger_FE_ratio"),
        "decision_score": context.get("decision_score"),
        "decision_threshold": context.get("decision_threshold"),
        "decision_check_count": int(context.get("decision_check_count", 0)),
        "FE_total": int(fe_total),
        "FE_probe": int(tracker.prefix_evaluations),
        "FE_query": int(tracker.query_evaluations),
        "FE_after": int(tracker.continuation_evaluations),
        "FE_used": int(tracker.total_evaluations),
        "runtime_probe": runtime_probe,
        "runtime_query_sampling": runtime_query_sampling,
        "runtime_query_evaluation": runtime_query_evaluation,
        "runtime_query_feature_computation": runtime_query_feature_computation,
        "runtime_query": runtime_query,
        "runtime_selection": runtime_selection,
        "runtime_handoff": runtime_handoff,
        "runtime_fresh_initialization": float(
            context.get("runtime_fresh_initialization", 0.0)
        ),
        "runtime_after_includes_fresh_initialization": bool(
            context.get("runtime_after_includes_fresh_initialization", False)
        ),
        "runtime_after": runtime_after,
        "runtime_component_sum_diagnostic": float(
            runtime_probe
            + runtime_query
            + runtime_selection
            + runtime_handoff
            + runtime_after
        ),
        "final_performance": observed_best,
        "path_failure_class": failure_class,
        "path_failure_message": failure_message,
        **trigger_metadata,
    }
    return row


def _finalize_online_policy_row(
    *,
    row: dict[str, Any],
    tracker: PathEvaluationTracker,
    config: dict,
    runtime_full_run_wall_clock: float,
    timed_out: bool,
    failure_class: str,
    failure_message: str,
) -> dict[str, Any]:
    reference = float(tracker.benchmark_reference_value)
    failure_cap = float(config["failure_loss_cap"])
    gap_floor = float(config["log10_gap_floor"])
    gap_cap = float(config["log10_gap_cap"])
    planned_fe = int(tracker.planned_total_fe)
    if not np.isfinite(runtime_full_run_wall_clock) or runtime_full_run_wall_clock <= 0.0:
        raise ValueError("full-run wall-clock must be finite and positive")
    if not failure_class and int(tracker.total_evaluations) != planned_fe:
        failure_class = "EvaluationCountMismatch"
        failure_message = (
            f"online policy evaluated {tracker.total_evaluations} points, expected {planned_fe}"
        )
    optimizer_best = (
        float(tracker.best_optimizer) if np.isfinite(tracker.best_optimizer) else None
    )
    all_best = float(tracker.best_all) if np.isfinite(tracker.best_all) else None
    raw_gap = max(all_best - reference, 0.0) if all_best is not None else failure_cap
    if failure_class:
        endpoint_gap = failure_cap
        endpoint_best = float(reference + failure_cap)
    else:
        endpoint_gap = min(raw_gap, failure_cap)
        endpoint_best = float(all_best)
    clipped_gap = min(max(endpoint_gap, gap_floor), gap_cap)
    observed_first_hit_fe = tracker.first_hit_fe
    target_hit_observed = observed_first_hit_fe is not None
    path_completed = bool(
        not failure_class
        and not timed_out
        and int(tracker.total_evaluations) == planned_fe
    )
    target_hit_before_failure = bool(target_hit_observed and not path_completed)
    endpoint_success = bool(target_hit_observed and path_completed)
    ert_fe = int(observed_first_hit_fe) if target_hit_observed else planned_fe
    path_status = (
        "completed" if path_completed else "timed_out" if timed_out else "failed"
    )
    query_improved = bool(
        np.isfinite(tracker.best_query)
        and (
            not np.isfinite(tracker.best_optimizer)
            or float(tracker.best_query) < float(tracker.best_optimizer)
        )
    )
    row.update(
        {
            "benchmark_reference_value": reference,
            "success_gap_target": float(tracker.success_gap_target),
            "log10_gap_floor": gap_floor,
            "log10_gap_cap": gap_cap,
            "failure_loss_cap": failure_cap,
            "continuation_only_best": optimizer_best,
            "all_evaluations_best": all_best,
            "observed_final_gap": float(raw_gap),
            "query_sample_improved_terminal": query_improved,
            "final_performance": endpoint_best,
            "final_gap": float(endpoint_gap),
            "log10_gap": float(np.log10(clipped_gap)),
            "observed_first_hit_FE": observed_first_hit_fe,
            "target_hit_observed": target_hit_observed,
            "target_hit_before_failure": target_hit_before_failure,
            "endpoint_success": endpoint_success,
            # Compatibility aliases retain the formal ERT semantics.
            "target_hit": target_hit_observed,
            "first_hit_FE": observed_first_hit_fe,
            "success": target_hit_observed,
            "path_completed": path_completed,
            "planned_total_FE": planned_fe,
            "effective_total_FE": int(tracker.total_evaluations),
            "ert_FE_contribution": ert_fe,
            "policy_timeout_seconds": float(config["policy_timeout_seconds"]),
            "timeout": bool(timed_out),
            "path_status": path_status,
            "path_failure_class": str(failure_class),
            "path_failure_message": str(failure_message),
            "coverage_denominator": 1,
            "coverage_complete": path_completed,
            "runtime_full_run_wall_clock": runtime_full_run_wall_clock,
            "runtime_measurement_scope": "full_run_fe0_to_terminal",
            "decision_state_future_path_runtime_measured": False,
            "runtime_decision_state_future_path_wall_clock": None,
        }
    )
    return row


def _attach_full_run_timing(
    rows: list[dict[str, Any]],
    *,
    scientific_rows: dict[tuple[str, int | None], dict[str, Any]],
) -> list[dict[str, Any]]:
    timing_components = (
        "runtime_probe",
        "runtime_query_sampling",
        "runtime_query_evaluation",
        "runtime_query_feature_computation",
        "runtime_query",
        "runtime_selection",
        "runtime_handoff",
        "runtime_after",
        "runtime_full_run_wall_clock",
    )
    groups: dict[tuple[str, int | None], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["policy_name"]), row.get("random_repetition"))
        groups.setdefault(key, []).append(row)
    if set(groups) != set(scientific_rows):
        raise ValueError(
            "online Stage-A scientific paths and Stage-B timing paths must have identical coverage"
        )
    for key, records in groups.items():
        ordered = sorted(records, key=lambda record: int(record["timing_repetition"]))
        indices = [int(record["timing_repetition"]) for record in ordered]
        if indices != list(range(TIMING_REPETITIONS)):
            raise ValueError(f"online path {key} is missing one or more timing repetitions")
        scientific = scientific_rows[key]
        endpoint_fields = (
            "query_called",
            "selected_algorithm",
            "trigger_FE",
            "FE_probe",
            "FE_query",
            "FE_after",
            "continuation_only_best",
            "all_evaluations_best",
            "observed_first_hit_FE",
            "target_hit_observed",
            "target_hit_before_failure",
            "endpoint_success",
            "first_hit_FE",
            "final_gap",
            "log10_gap",
            "success",
        )
        replay_completed = [
            bool(record.get("coverage_complete"))
            and str(record.get("path_status")) == "completed"
            and not bool(record.get("timeout"))
            for record in ordered
        ]
        scientific_completed = (
            bool(scientific.get("coverage_complete"))
            and str(scientific.get("path_status")) == "completed"
            and not bool(scientific.get("timeout"))
        )
        scientific_status = (
            "completed"
            if scientific_completed
            else "timed_out"
            if bool(scientific.get("timeout"))
            else "failed"
        )
        completed_records = [
            record for record, completed in zip(ordered, replay_completed, strict=True) if completed
        ]
        internal_consistency: bool | None = None
        if len(completed_records) >= 2:
            reference = completed_records[0]
            internal_consistency = not any(
                not _timing_value_equal(reference.get(field), record.get(field))
                for field in endpoint_fields
                for record in completed_records[1:]
            )
            if not internal_consistency:
                raise ValueError(
                    f"online path {key} has internally inconsistent completed Stage-B outcomes"
                )
        stage_a_consistency: bool | None = None
        if scientific_completed and completed_records:
            stage_a_consistency = not any(
                not _timing_value_equal(scientific.get(field), record.get(field))
                for field in endpoint_fields
                for record in completed_records
            )
            if not stage_a_consistency:
                raise ValueError(
                    f"online path {key} has completed Stage-B outcomes inconsistent with Stage A"
                )
        replay_timeouts = [bool(record.get("timeout")) for record in ordered]
        replay_statuses = [
            "completed"
            if completed
            else "timed_out"
            if timed_out
            else "failed"
            for completed, timed_out in zip(
                replay_completed,
                replay_timeouts,
                strict=True,
            )
        ]
        timing_replay_instability = len(set(replay_statuses)) > 1
        stage_a_stage_b_completion_status_instability = any(
            completed != scientific_completed for completed in replay_completed
        )
        identity_fields = (
            "policy_name",
            "random_repetition",
            "default_algorithm",
            "query_id",
            "query_protocol",
            "sample_design_id",
            "planned_total_FE",
        )
        path_identity_consistent = not any(
            not _timing_value_equal(ordered[0].get(field), record.get(field))
            for field in identity_fields
            for record in ordered[1:]
        )
        if not path_identity_consistent:
            raise ValueError(f"online path {key} changed identity across timing repetitions")
        replay_outcome_fields = (
            "query_called",
            "selected_algorithm",
            "trigger_FE",
            "effective_total_FE",
            "continuation_only_best",
            "all_evaluations_best",
            "final_gap",
            "log10_gap",
            "first_hit_FE",
            "success",
            "observed_first_hit_FE",
            "target_hit_observed",
            "target_hit_before_failure",
            "endpoint_success",
            "path_completed",
            "ert_FE_contribution",
            "timeout",
            "path_status",
            "path_failure_class",
            "path_failure_message",
            "coverage_complete",
        )
        for record in records:
            for field in replay_outcome_fields:
                record[f"timing_replay_{field}"] = record.get(field)
        for component in timing_components:
            values = [float(record[component]) for record in ordered]
            if not np.isfinite(values).all() or any(value < 0.0 for value in values):
                raise ValueError(f"online path {key} has invalid {component} repetitions")
            if component == "runtime_full_run_wall_clock" and any(value <= 0.0 for value in values):
                raise ValueError(f"online path {key} has non-positive full-run timing")
            raw_observed_median = float(np.median(values))
            if component == "runtime_full_run_wall_clock":
                role_timeout = float(scientific["policy_timeout_seconds"])
                censored_values = [
                    value if completed else max(value, role_timeout)
                    for value, completed in zip(values, replay_completed, strict=True)
                ]
                median = float(np.median(censored_values))
            else:
                censored_values = values
                median = raw_observed_median
            for record in records:
                record[f"{component}_repetitions"] = values
                record[f"{component}_raw_observed_median"] = raw_observed_median
                record[f"{component}_median"] = median
                if component == "runtime_full_run_wall_clock":
                    record[f"{component}_censored_repetitions"] = censored_values
        positions = [int(record["timing_order_position"]) for record in ordered]
        for record in records:
            record["timing_repetition_indices"] = list(range(TIMING_REPETITIONS))
            record["timing_order_positions"] = positions
            record["scientific_endpoint_stage"] = "pre_specified_stage_a_run"
            record["scientific_path_status"] = scientific_status
            record["scientific_path_timed_out"] = bool(
                scientific_status == "timed_out"
            )
            record["scientific_path_completed"] = bool(
                scientific_status == "completed"
            )
            record["scientific_run_order_position"] = int(
                scientific["scientific_run_order_position"]
            )
            record["scientific_runtime_full_run_wall_clock"] = float(
                scientific["runtime_full_run_wall_clock"]
            )
            record["timing_replay_completed_repetitions"] = int(sum(replay_completed))
            record["timing_replay_timeout_repetitions"] = int(sum(replay_timeouts))
            record["timing_replay_failure_repetitions"] = int(
                sum(status == "failed" for status in replay_statuses)
            )
            record["timing_replay_effective_FE_repetitions"] = [
                int(value) for value in (
                    replay["effective_total_FE"] for replay in ordered
                )
            ]
            record["timing_replay_observed_first_hit_FE_repetitions"] = [
                replay.get("observed_first_hit_FE") for replay in ordered
            ]
            record["timing_replay_target_hit_observed_flags"] = [
                bool(replay.get("target_hit_observed")) for replay in ordered
            ]
            record["timing_replay_target_hit_before_failure_flags"] = [
                bool(replay.get("target_hit_before_failure")) for replay in ordered
            ]
            record["timing_replay_endpoint_success_flags"] = [
                bool(replay.get("endpoint_success")) for replay in ordered
            ]
            record["timing_replay_status_repetitions"] = replay_statuses
            record["timing_replay_timed_out_flags"] = replay_timeouts
            record["timing_replay_path_completed_flags"] = replay_completed
            record["timing_replay_path_identity_consistent"] = path_identity_consistent
            record[
                "completed_timing_replay_outcomes_internally_consistent"
            ] = internal_consistency
            record[
                "stage_a_to_completed_timing_replays_consistent"
            ] = stage_a_consistency
            record["timing_replay_status_instability"] = timing_replay_instability
            record[
                "stage_a_stage_b_completion_status_instability"
            ] = stage_a_stage_b_completion_status_instability
            # Compatibility alias: Stage-B status instability only.
            record["timing_replay_instability"] = timing_replay_instability
            record["timing_replay_status_stable"] = bool(
                not timing_replay_instability
            )
            for field, value in scientific.items():
                if not str(field).startswith("runtime_"):
                    record[field] = value

    skip_records = groups.get(("sbs_no_query", None))
    if skip_records is None:
        raise ValueError("online timing requires one three-repeat sbs_no_query reference")
    skip_median = float(skip_records[0]["runtime_full_run_wall_clock_median"])
    if not np.isfinite(skip_median) or skip_median <= 0.0:
        raise ValueError("Skip median full-run wall-clock must be finite and positive")
    for row in rows:
        median = float(row["runtime_full_run_wall_clock_median"])
        row["log10_runtime_ratio_vs_skip"] = float(np.log10(median / skip_median))
        row["skip_runtime_full_run_wall_clock_median"] = skip_median
    return rows


def _timing_value_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        return bool(np.isclose(float(left), float(right), rtol=1e-12, atol=1e-12))
    return bool(left == right)


def _run_no_query_policy(
    *,
    problem,
    tracker: PathEvaluationTracker,
    config: dict,
    seed: int,
    fe_total: int,
    selector: OnlineSelector,
    policy_name: str,
    sampling_protocol: str,
    decision_check_frequency: str,
    repetition: int | None,
) -> dict[str, Any]:
    default_algorithm = selector.sbs_algorithm
    tracker.remember(selector_status="not_used")
    settings = OptimizerSettings(
        population_size=int(config["population_size"]),
        checkpoint_ratios=(1.0,),
    )
    started = perf_counter()
    try:
        state = initialize_optimizer_state(
            algorithm=default_algorithm,
            problem=problem,
            seed=seed,
            settings=settings,
        )
    finally:
        runtime_initialization = perf_counter() - started
        tracker.remember(runtime_after=float(runtime_initialization))
    remaining_budget = int(fe_total) - int(state.evaluations)
    if remaining_budget < 0:
        raise ValueError("FE_total is smaller than the optimizer native initialization")
    continuation_started = perf_counter()
    try:
        after = advance_optimizer_state(
            state=state, problem=problem, fe_budget=remaining_budget
        )
    finally:
        runtime_optimization = float(
            runtime_initialization + perf_counter() - continuation_started
        )
        tracker.remember(runtime_after=runtime_optimization)
    runtime_optimization = float(runtime_initialization + after.runtime_seconds)
    tracker.remember(runtime_after=runtime_optimization)
    if int(state.evaluations) != int(fe_total):
        raise ValueError("Never Query did not consume the complete optimizer budget")
    row = _base_policy_row(
        policy_name=policy_name,
        policy_category="baseline",
        sampling_protocol=sampling_protocol,
        decision_check_frequency=decision_check_frequency,
        repetition=repetition,
        prefix_algorithm=default_algorithm,
        default_algorithm=default_algorithm,
        selected_algorithm=default_algorithm,
        selected_action="continue_current",
        selected_equals_default=True,
        selected_equals_prefix=True,
        handoff_required=False,
        selector_status="not_used",
        optimizer_transition_mode="native_optimizer_state",
        selector_target_transform=selector.model.selector_target_transform,
        selector_remaining_budget_ratio=None,
        query_called=False,
        query_id=selector.model.query_id,
        query_protocol=selector.model.query_protocol,
        sample_design_id=selector.model.sample_design_id,
        query_feature_status="not_called",
        query_feature_failure="",
        query_failure_fallback=False,
        trigger_fe=None,
        trigger_fe_ratio=None,
        decision_score=None,
        decision_threshold=None,
        decision_check_count=0,
        trigger_sampling_metadata=_empty_trigger_sampling_metadata(),
        fe_total=fe_total,
        fe_probe=0,
        fe_query=0,
        fe_after=fe_total,
        runtime_probe=0.0,
        runtime_query_sampling=0.0,
        runtime_query_evaluation=0.0,
        runtime_query_feature_computation=0.0,
        runtime_query=0.0,
        runtime_selection=0.0,
        runtime_handoff=0.0,
        runtime_after=runtime_optimization,
        final_performance=float(state.best_fitness),
    )
    return row


def _run_pre_run_aas_policy(
    *,
    problem,
    tracker: PathEvaluationTracker,
    deadline: float,
    config: dict,
    seed: int,
    fe_total: int,
    selector: OnlineSelector,
    pre_run_aas_selector: OnlineQueryOnlySelector,
    query_feature_row: dict[str, Any],
    query_sample_row: dict[str, Any],
    sampling_protocol: str,
    decision_check_frequency: str,
) -> dict[str, Any]:
    default_algorithm = selector.sbs_algorithm
    fe_query = _query_fe(selector, problem.dimension, fe_total)
    _validate_query_feature_problem(
        query_feature_row,
        problem.problem_id,
        fe_query=fe_query,
        fe_total=fe_total,
    )
    tracker.remember(
        query_called=True,
        trigger_FE=0,
        trigger_FE_ratio=0.0,
        optimizer_transition_mode="fresh_optimizer_initialization",
        handoff_type="fresh_optimizer_initialization",
        query_transition_mode="fresh_optimizer_initialization",
    )
    query_execution = _execute_real_query(
        problem=problem,
        tracker=tracker,
        deadline=deadline,
        query_feature_row=query_feature_row,
        query_sample_row=query_sample_row,
    )
    query_ok = query_execution.status == "ok"
    remaining_budget = int(fe_total) - int(fe_query)
    population_size = int(config["population_size"])
    if remaining_budget < population_size:
        raise ValueError("pre-run AAS leaves insufficient FE for native optimizer initialization")
    if query_ok:
        query_features = {
            column: query_execution.features.get(column)
            for column in pre_run_aas_selector.artifact.query_feature_columns
        }
        selection_started = perf_counter()
        try:
            (
                selected_algorithm,
                selector_remaining_budget_ratio,
                selector_status,
                runtime_selection,
            ) = pre_run_aas_selector.select(
                query_features,
                float(remaining_budget / fe_total),
            )
        finally:
            tracker.remember(runtime_selection=float(perf_counter() - selection_started))
        query_failure_fallback = False
    else:
        selected_algorithm = default_algorithm
        selector_remaining_budget_ratio = float(remaining_budget / fe_total)
        selector_status = "query_failure_fallback_default"
        runtime_selection = 0.0
        query_failure_fallback = True

    tracker.remember(
        prefix_algorithm=selected_algorithm,
        selected_algorithm=selected_algorithm,
        selected_action=selected_algorithm,
        selected_equals_default=selected_algorithm == default_algorithm,
        selected_equals_prefix=True,
        handoff_required=False,
        selector_status=selector_status,
        selector_remaining_budget_ratio=selector_remaining_budget_ratio,
        query_failure_fallback=query_failure_fallback,
        runtime_selection=float(runtime_selection),
    )

    settings = OptimizerSettings(population_size=population_size, checkpoint_ratios=(1.0,))
    tracker.set_phase("continuation")
    started = perf_counter()
    try:
        state = initialize_optimizer_state(
            algorithm=selected_algorithm,
            problem=problem,
            seed=seed,
            settings=settings,
        )
    finally:
        runtime_initialization = perf_counter() - started
        tracker.remember(
            runtime_after=float(runtime_initialization),
            runtime_fresh_initialization=float(runtime_initialization),
            runtime_after_includes_fresh_initialization=True,
        )
    continuation_budget = remaining_budget - int(state.evaluations)
    continuation_started = perf_counter()
    try:
        after = advance_optimizer_state(
            state=state,
            problem=problem,
            fe_budget=continuation_budget,
        )
    finally:
        tracker.remember(
            runtime_after=float(
                runtime_initialization + perf_counter() - continuation_started
            )
        )
    runtime_after = float(runtime_initialization + after.runtime_seconds)
    tracker.remember(runtime_after=runtime_after)
    if int(state.evaluations) != remaining_budget:
        raise ValueError("pre-run AAS optimizer did not consume B-FE_query evaluations")
    row = _base_policy_row(
        policy_name="pre_run_aas_fe0",
        policy_category="baseline",
        sampling_protocol=sampling_protocol,
        decision_check_frequency=decision_check_frequency,
        repetition=None,
        prefix_algorithm=selected_algorithm,
        default_algorithm=default_algorithm,
        selected_algorithm=selected_algorithm,
        selected_action=selected_algorithm,
        selected_equals_default=selected_algorithm == default_algorithm,
        selected_equals_prefix=True,
        handoff_required=False,
        selector_status=selector_status,
        optimizer_transition_mode="fresh_optimizer_initialization",
        selector_target_transform=str(pre_run_aas_selector.artifact.selector_target_transform),
        selector_remaining_budget_ratio=selector_remaining_budget_ratio,
        query_called=True,
        query_id=pre_run_aas_selector.query_id,
        query_protocol=str(pre_run_aas_selector.artifact.query_protocol),
        sample_design_id=str(pre_run_aas_selector.artifact.sample_design_id),
        query_feature_status=query_execution.status,
        query_feature_failure=query_execution.failure,
        query_failure_fallback=query_failure_fallback,
        trigger_fe=0,
        trigger_fe_ratio=0.0,
        decision_score=None,
        decision_threshold=None,
        decision_check_count=0,
        trigger_sampling_metadata=_empty_trigger_sampling_metadata(),
        fe_total=fe_total,
        fe_probe=0,
        fe_query=fe_query,
        fe_after=remaining_budget,
        runtime_probe=0.0,
        runtime_query_sampling=query_execution.runtime_sampling,
        runtime_query_evaluation=query_execution.runtime_evaluation,
        runtime_query_feature_computation=query_execution.runtime_feature_computation,
        runtime_query=query_execution.runtime_total,
        runtime_selection=runtime_selection,
        runtime_handoff=0.0,
        runtime_after=runtime_after,
        final_performance=float(state.best_fitness),
    )
    row.update(
        {
            "runtime_fresh_initialization": float(runtime_initialization),
            "runtime_after_includes_fresh_initialization": True,
            "query_success": query_execution.first_hit_offset is not None,
            "query_first_hit_offset": query_execution.first_hit_offset,
            "query_best_gap": query_execution.best_gap,
        }
    )
    return row


def _run_behavior_only_policy(
    *,
    problem,
    tracker: PathEvaluationTracker,
    config: dict,
    function: int,
    seed: int,
    fe_total: int,
    query_selector: OnlineSelector,
    behavior_selector: OnlineBehaviorOnlySelector,
    controller: DecisionControllerModel | None,
    policy_name: str,
    trigger_mode: str,
    sampling_protocol: str,
    decision_check_frequency: str,
    repetition: int | None,
    matched_trigger_fe: int | None = None,
) -> dict[str, Any]:
    default_algorithm = query_selector.sbs_algorithm
    if behavior_selector.sbs_algorithm != default_algorithm:
        raise ValueError("Behavior-only and Query selectors use different frozen SBS defaults")
    tracker.set_phase("prefix")
    population_size = int(config["population_size"])
    settings = OptimizerSettings(population_size=population_size, checkpoint_ratios=(1.0,))
    started = perf_counter()
    current_state = initialize_optimizer_state(
        algorithm=default_algorithm,
        problem=problem,
        seed=seed,
        settings=settings,
    )
    prefix_algorithm = str(current_state.algorithm)
    tracker.remember(
        prefix_algorithm=prefix_algorithm,
        runtime_probe=float(perf_counter() - started),
    )
    behavior_stream = StreamingBehaviorState(
        problem_id=problem.problem_id,
        function_id=problem.function_id,
        family=problem.family,
        dimension=problem.dimension,
        algorithm=default_algorithm,
        seed=seed,
        fe_total=fe_total,
        sampling_protocol=sampling_protocol,
    )
    behavior_stream.observe(
        fe=int(current_state.evaluations),
        native_updates=int(current_state.generation),
        population=current_state.population,
        fitness=current_state.fitness,
        best_fitness=current_state.best_fitness,
    )
    runtime_probe = perf_counter() - started
    current_fe = int(current_state.evaluations)
    triggered = False
    trigger_ratio: float | None = None
    trigger_score: float | None = None
    selected_algorithm = default_algorithm
    selector_remaining_budget_ratio: float | None = None
    selector_status = "not_used"
    runtime_selection = 0.0
    runtime_handoff = 0.0
    decision_check_count = 0
    trigger_sampling_metadata = _empty_trigger_sampling_metadata()

    while behavior_stream.next_monitor_ratio is not None:
        delta = min(population_size, fe_total - current_fe)
        if delta <= 0:
            break
        continuation = advance_optimizer_state(
            state=current_state,
            problem=problem,
            fe_budget=delta,
            on_native_update=lambda updated: behavior_stream.observe(
                fe=int(updated.evaluations),
                native_updates=int(updated.generation),
                population=updated.population,
                fitness=updated.fitness,
                best_fitness=updated.best_fitness,
            ),
        )
        runtime_probe += continuation.runtime_seconds
        tracker.remember(runtime_probe=float(runtime_probe))
        current_fe = int(current_state.evaluations)
        sampled_started = perf_counter()
        behavior_row = behavior_stream.sample_dynamic()
        if behavior_row is None:
            runtime_probe += perf_counter() - sampled_started
            continue
        decision_check_count += 1
        tracker.remember(decision_check_count=decision_check_count)
        should_trigger, trigger_score = _should_trigger(
            behavior_row=behavior_row,
            controller=controller,
            trigger_mode=trigger_mode,
            matched_trigger_fe=matched_trigger_fe,
        )
        runtime_probe += perf_counter() - sampled_started
        if not should_trigger:
            continue
        triggered = True
        trigger_ratio = float(behavior_row["FE_ratio"])
        trigger_sampling_metadata = {
            f"trigger_{column}": behavior_row[column]
            for column in SAMPLING_METADATA_COLUMNS
        }
        remaining_ratio = float((fe_total - current_fe) / fe_total)
        tracker.remember(
            trigger_FE=current_fe,
            trigger_FE_ratio=trigger_ratio,
            decision_score=trigger_score,
            decision_threshold=(
                float(controller.threshold) if controller is not None else None
            ),
            **trigger_sampling_metadata,
        )
        selection_started = perf_counter()
        try:
            (
                selected_algorithm,
                selector_remaining_budget_ratio,
                selector_status,
                runtime_selection,
            ) = behavior_selector.select(behavior_row, remaining_ratio)
        finally:
            tracker.remember(runtime_selection=float(perf_counter() - selection_started))
        selected_equals_prefix = selected_algorithm == prefix_algorithm
        tracker.remember(
            selected_algorithm=selected_algorithm,
            selected_action=(
                "continue_current" if selected_equals_prefix else selected_algorithm
            ),
            selected_equals_default=selected_algorithm == default_algorithm,
            selected_equals_prefix=selected_equals_prefix,
            handoff_required=not selected_equals_prefix,
            handoff_type=(
                "native_optimizer_state"
                if selected_equals_prefix
                else "population_transfer_initialization"
            ),
            query_transition_mode=(
                "native_optimizer_state"
                if selected_equals_prefix
                else "population_transfer_initialization"
            ),
            optimizer_transition_mode=(
                "native_optimizer_state"
                if selected_equals_prefix
                else "population_transfer_initialization"
            ),
            selector_status=selector_status,
            selector_remaining_budget_ratio=selector_remaining_budget_ratio,
            runtime_selection=float(runtime_selection),
        )
        break

    remaining_budget = max(fe_total - current_fe, 0)
    if triggered and selected_algorithm != prefix_algorithm:
        handoff_started = perf_counter()
        try:
            after_state = initialize_transferred_optimizer_state(
                algorithm=selected_algorithm,
                source_state=current_state,
                problem=problem,
                seed=seed,
                function=function,
                instance=1,
                event=QUERY_TRANSFER_EVENT,
            )
        finally:
            runtime_handoff = perf_counter() - handoff_started
            tracker.remember(runtime_handoff=float(runtime_handoff))
        transition_mode = "population_transfer_initialization"
    else:
        after_state = current_state
        transition_mode = "native_optimizer_state"
    tracker.set_phase("continuation")
    after_started = perf_counter()
    try:
        after = advance_optimizer_state(
            state=after_state,
            problem=problem,
            fe_budget=remaining_budget,
        )
    finally:
        tracker.remember(runtime_after=float(perf_counter() - after_started))
    tracker.remember(runtime_after=float(after.runtime_seconds))
    if int(after.evaluations) != remaining_budget:
        raise ValueError("Behavior-only continuation did not consume its assigned FE budget")
    if current_fe + int(after.evaluations) != fe_total:
        raise ValueError("Behavior-only policy must consume exactly FE_total evaluations")
    final_performance = float(after.state.best_fitness)
    selected_equals_prefix = selected_algorithm == prefix_algorithm
    handoff_required = bool(triggered and not selected_equals_prefix)
    if handoff_required != (transition_mode == "population_transfer_initialization"):
        raise ValueError("Behavior-only handoff relation is inconsistent")
    row = _base_policy_row(
        policy_name=policy_name,
        policy_category="diagnostic" if policy_name == "matched_trigger_behavior_only" else "baseline",
        sampling_protocol=sampling_protocol,
        decision_check_frequency=decision_check_frequency,
        repetition=repetition,
        prefix_algorithm=prefix_algorithm,
        default_algorithm=default_algorithm,
        selected_algorithm=selected_algorithm,
        selected_action="continue_current" if selected_equals_prefix else selected_algorithm,
        selected_equals_default=selected_algorithm == default_algorithm,
        selected_equals_prefix=selected_equals_prefix,
        handoff_required=handoff_required,
        selector_status=selector_status,
        optimizer_transition_mode=transition_mode,
        selector_target_transform=behavior_selector.model.selector_target_transform,
        selector_remaining_budget_ratio=selector_remaining_budget_ratio,
        query_called=False,
        query_id=query_selector.model.query_id,
        query_protocol=query_selector.model.query_protocol,
        sample_design_id=query_selector.model.sample_design_id,
        query_feature_status="not_called",
        query_feature_failure="",
        query_failure_fallback=False,
        trigger_fe=current_fe if triggered else None,
        trigger_fe_ratio=trigger_ratio,
        decision_score=trigger_score,
        decision_threshold=float(controller.threshold) if controller is not None else None,
        decision_check_count=decision_check_count,
        trigger_sampling_metadata=trigger_sampling_metadata,
        fe_total=fe_total,
        fe_probe=current_fe,
        fe_query=0,
        fe_after=int(after.evaluations),
        runtime_probe=runtime_probe,
        runtime_query_sampling=0.0,
        runtime_query_evaluation=0.0,
        runtime_query_feature_computation=0.0,
        runtime_query=0.0,
        runtime_selection=runtime_selection,
        runtime_handoff=runtime_handoff,
        runtime_after=float(after.runtime_seconds),
        final_performance=final_performance,
    )
    return row


def _run_threshold_policy(
    *,
    problem,
    tracker: PathEvaluationTracker,
    deadline: float,
    config: dict,
    function: int,
    seed: int,
    fe_total: int,
    selector: OnlineSelector,
    query_feature_row: dict[str, Any],
    query_sample_row: dict[str, Any],
    controller: DecisionControllerModel | None,
    policy_name: str,
    trigger_mode: str,
    sampling_protocol: str,
    decision_check_frequency: str,
    repetition: int | None,
    matched_trigger_fe_ratio: float | None = None,
    decision_opportunity_scope: str = "all_accepted",
) -> dict[str, Any]:
    default_algorithm = selector.sbs_algorithm
    population_size = int(config["population_size"])
    fe_query = _query_fe(selector, problem.dimension, fe_total)
    tracker.set_phase("prefix")
    settings = OptimizerSettings(population_size=population_size, checkpoint_ratios=(1.0,))
    started = perf_counter()
    current_state = initialize_optimizer_state(
        algorithm=default_algorithm,
        problem=problem,
        seed=seed,
        settings=settings,
    )
    prefix_algorithm = str(current_state.algorithm)
    tracker.remember(
        prefix_algorithm=prefix_algorithm,
        runtime_probe=float(perf_counter() - started),
    )
    if decision_opportunity_scope == "all_accepted":
        behavior_stream = StreamingBehaviorState(
            problem_id=problem.problem_id,
            function_id=problem.function_id,
            family=problem.family,
            dimension=problem.dimension,
            algorithm=default_algorithm,
            seed=seed,
            fe_total=fe_total,
            sampling_protocol=sampling_protocol,
        )
    elif decision_opportunity_scope == "milestone_only":
        behavior_stream = MilestoneBehaviorState(
            problem_id=problem.problem_id,
            function_id=problem.function_id,
            family=problem.family,
            dimension=problem.dimension,
            algorithm=default_algorithm,
            seed=seed,
            fe_total=fe_total,
            sampling_protocol=sampling_protocol,
        )
    else:
        raise ValueError(f"unsupported online decision opportunity scope: {decision_opportunity_scope}")
    behavior_stream.observe(
        fe=int(current_state.evaluations),
        native_updates=int(current_state.generation),
        population=current_state.population,
        fitness=current_state.fitness,
        best_fitness=current_state.best_fitness,
    )
    runtime_probe = perf_counter() - started
    current_fe = int(current_state.evaluations)
    triggered = False
    trigger_ratio = None
    trigger_score = None
    selected_algorithm = default_algorithm
    selector_remaining_budget_ratio = None
    selector_status = "not_used"
    runtime_query = 0.0
    runtime_query_sampling = 0.0
    runtime_query_evaluation = 0.0
    runtime_query_feature_computation = 0.0
    runtime_selection = 0.0
    runtime_handoff = 0.0
    query_execution: QueryExecution | None = None
    decision_check_count = 0
    trigger_sampling_metadata: dict[str, Any] = {
        f"trigger_{column}": None for column in SAMPLING_METADATA_COLUMNS
    }

    while behavior_stream.next_monitor_ratio is not None:
        delta = min(population_size, fe_total - current_fe)
        if delta <= 0:
            break
        continuation = advance_optimizer_state(
            state=current_state,
            problem=problem,
            fe_budget=delta,
            on_native_update=lambda updated: behavior_stream.observe(
                fe=int(updated.evaluations),
                native_updates=int(updated.generation),
                population=updated.population,
                fitness=updated.fitness,
                best_fitness=updated.best_fitness,
            ),
        )
        runtime_probe += continuation.runtime_seconds
        tracker.remember(runtime_probe=float(runtime_probe))
        current_fe = int(current_state.evaluations)
        sample_started = perf_counter()
        behavior_row = (
            behavior_stream.sample_dynamic()
            if decision_opportunity_scope == "all_accepted"
            else behavior_stream.sample_milestone()
        )
        if behavior_row is None:
            runtime_probe += perf_counter() - sample_started
            continue
        if current_fe + fe_query > fe_total:
            break
        decision_check_count += 1
        tracker.remember(decision_check_count=decision_check_count)
        should_trigger, trigger_score = _should_trigger(
            behavior_row=behavior_row,
            controller=controller,
            trigger_mode=trigger_mode,
            matched_trigger_fe_ratio=matched_trigger_fe_ratio,
        )
        runtime_probe += perf_counter() - sample_started
        if should_trigger:
            if decision_opportunity_scope == "milestone_only":
                feature_started = perf_counter()
                behavior_row = behavior_stream.full_behavior_state()
                runtime_probe += perf_counter() - feature_started
            triggered = True
            trigger_ratio = float(behavior_row["FE_ratio"])
            trigger_sampling_metadata = {
                f"trigger_{column}": behavior_row[column]
                for column in SAMPLING_METADATA_COLUMNS
            }
            tracker.remember(
                query_called=True,
                trigger_FE=current_fe,
                trigger_FE_ratio=trigger_ratio,
                decision_score=trigger_score,
                decision_threshold=(
                    float(controller.threshold) if controller is not None else None
                ),
                runtime_probe=float(runtime_probe),
                **trigger_sampling_metadata,
            )
            _validate_query_feature_problem(
                query_feature_row,
                problem.problem_id,
                fe_query=fe_query,
                fe_total=fe_total,
            )
            query_execution = _execute_real_query(
                problem=problem,
                tracker=tracker,
                deadline=deadline,
                query_feature_row=query_feature_row,
                query_sample_row=query_sample_row,
            )
            runtime_query_sampling = query_execution.runtime_sampling
            runtime_query_evaluation = query_execution.runtime_evaluation
            runtime_query_feature_computation = query_execution.runtime_feature_computation
            runtime_query = query_execution.runtime_total
            remaining = max(fe_total - current_fe - fe_query, 0)
            remaining_ratio = round(remaining / fe_total, 6)
            if query_execution.status == "ok":
                query_features = {
                    column: query_execution.features.get(column)
                    for column in selector.model.query_feature_columns
                }
                selection_started = perf_counter()
                try:
                    (
                        selected_algorithm,
                        selector_remaining_budget_ratio,
                        selector_status,
                        runtime_selection,
                    ) = selector.select(
                        query_features,
                        behavior_row,
                        remaining_ratio,
                    )
                finally:
                    tracker.remember(
                        runtime_selection=float(perf_counter() - selection_started)
                    )
            else:
                selected_algorithm = default_algorithm
                selector_remaining_budget_ratio = remaining_ratio
                selector_status = "query_failure_fallback_default"
                runtime_selection = 0.0
            selected_equals_prefix = selected_algorithm == prefix_algorithm
            planned_transition = (
                "native_optimizer_state"
                if selected_equals_prefix
                else "population_transfer_initialization"
            )
            tracker.remember(
                selected_algorithm=selected_algorithm,
                selected_action=(
                    "continue_current" if selected_equals_prefix else selected_algorithm
                ),
                selected_equals_default=selected_algorithm == default_algorithm,
                selected_equals_prefix=selected_equals_prefix,
                handoff_required=not selected_equals_prefix,
                handoff_type=planned_transition,
                query_transition_mode=planned_transition,
                optimizer_transition_mode=planned_transition,
                selector_status=selector_status,
                selector_remaining_budget_ratio=selector_remaining_budget_ratio,
                query_failure_fallback=(
                    selector_status == "query_failure_fallback_default"
                ),
                runtime_selection=float(runtime_selection),
            )
            break

    if triggered:
        remaining_budget = max(fe_total - current_fe - fe_query, 0)
        if selected_algorithm == prefix_algorithm:
            after_state = current_state
            transition_mode = "native_optimizer_state"
        else:
            handoff_started = perf_counter()
            try:
                after_state = initialize_transferred_optimizer_state(
                    algorithm=selected_algorithm,
                    source_state=current_state,
                    problem=problem,
                    seed=seed,
                    function=function,
                    instance=1,
                    event=QUERY_TRANSFER_EVENT,
                )
            finally:
                runtime_handoff = perf_counter() - handoff_started
                tracker.remember(runtime_handoff=float(runtime_handoff))
            transition_mode = "population_transfer_initialization"
        tracker.set_phase("continuation")
        after_started = perf_counter()
        try:
            after = advance_optimizer_state(
                state=after_state, problem=problem, fe_budget=remaining_budget
            )
        finally:
            tracker.remember(runtime_after=float(perf_counter() - after_started))
        tracker.remember(runtime_after=float(after.runtime_seconds))
        final_performance = float(after.state.best_fitness)
        runtime_after = float(after.runtime_seconds)
        fe_after = int(after.evaluations)
        if fe_after != remaining_budget:
            raise ValueError("query continuation did not consume its assigned FE budget")
        fe_used = int(current_fe + fe_query + fe_after)
    else:
        remaining_budget = max(fe_total - current_fe, 0)
        tracker.set_phase("continuation")
        after_started = perf_counter()
        try:
            after = advance_optimizer_state(
                state=current_state, problem=problem, fe_budget=remaining_budget
            )
        finally:
            tracker.remember(runtime_after=float(perf_counter() - after_started))
        tracker.remember(runtime_after=float(after.runtime_seconds))
        final_performance = float(after.state.best_fitness)
        runtime_after = float(after.runtime_seconds)
        fe_after = int(after.evaluations)
        if fe_after != remaining_budget:
            raise ValueError("Skip continuation did not consume its assigned FE budget")
        fe_used = int(current_fe + fe_after)
        transition_mode = "native_optimizer_state"

    if fe_used != fe_total:
        raise ValueError("online policy must charge exactly FE_total across probe, query, and continuation")
    policy_category = "controller" if policy_name == "current_controller" else "baseline"
    selected_equals_prefix = selected_algorithm == prefix_algorithm
    selected_action = "continue_current" if selected_equals_prefix else selected_algorithm
    handoff_required = not selected_equals_prefix
    if handoff_required != (transition_mode == "population_transfer_initialization"):
        raise ValueError("online handoff_required does not match optimizer_transition_mode")
    row = _base_policy_row(
        policy_name=policy_name,
        policy_category=policy_category,
        sampling_protocol=sampling_protocol,
        decision_check_frequency=decision_check_frequency,
        repetition=repetition,
        prefix_algorithm=prefix_algorithm,
        default_algorithm=default_algorithm,
        selected_algorithm=selected_algorithm,
        selected_action=selected_action,
        selected_equals_default=selected_algorithm == default_algorithm,
        selected_equals_prefix=selected_equals_prefix,
        handoff_required=handoff_required,
        selector_status=selector_status,
        optimizer_transition_mode=transition_mode,
        selector_target_transform=selector.model.selector_target_transform,
        selector_remaining_budget_ratio=selector_remaining_budget_ratio,
        query_called=triggered,
        query_id=selector.model.query_id,
        query_protocol=selector.model.query_protocol,
        sample_design_id=selector.model.sample_design_id,
        query_feature_status=(query_execution.status if query_execution is not None else "not_called"),
        query_feature_failure=(query_execution.failure if query_execution is not None else ""),
        query_failure_fallback=bool(triggered and selector_status == "query_failure_fallback_default"),
        trigger_fe=int(current_fe) if triggered else None,
        trigger_fe_ratio=float(trigger_ratio) if triggered else None,
        decision_score=float(trigger_score) if trigger_score is not None else None,
        decision_threshold=float(controller.threshold) if controller is not None else None,
        decision_check_count=decision_check_count,
        trigger_sampling_metadata=trigger_sampling_metadata,
        fe_total=fe_total,
        fe_probe=current_fe,
        fe_query=fe_query if triggered else 0,
        fe_after=fe_after,
        runtime_probe=runtime_probe,
        runtime_query_sampling=runtime_query_sampling,
        runtime_query_evaluation=runtime_query_evaluation,
        runtime_query_feature_computation=runtime_query_feature_computation,
        runtime_query=runtime_query,
        runtime_selection=runtime_selection,
        runtime_handoff=runtime_handoff,
        runtime_after=runtime_after,
        final_performance=final_performance,
    )
    if query_execution is not None:
        row.update(
            {
                "query_success": query_execution.first_hit_offset is not None,
                "query_first_hit_offset": query_execution.first_hit_offset,
                "query_best_gap": query_execution.best_gap,
            }
        )
    return row


def _should_trigger(
    *,
    behavior_row: dict[str, Any],
    controller: DecisionControllerModel | None,
    trigger_mode: str,
    matched_trigger_fe_ratio: float | None = None,
    matched_trigger_fe: int | None = None,
) -> tuple[bool, float | None]:
    if trigger_mode == "first_checkpoint":
        return True, None
    if trigger_mode == "never":
        return False, None
    if trigger_mode == "controller":
        if controller is None:
            raise ValueError("controller trigger mode requires a fitted Decision controller model")
        frame = pd.DataFrame([{column: behavior_row[column] for column in controller.feature_columns}])
        score = float(decision_scores(controller.model, frame)[0])
        return bool(score > controller.threshold), score
    if trigger_mode == "matched_random":
        if matched_trigger_fe_ratio is None:
            raise ValueError("matched-random trigger requires a train-OOF target FE ratio")
        return bool(float(behavior_row["FE_ratio"]) >= float(matched_trigger_fe_ratio)), None
    if trigger_mode == "matched_trigger":
        if matched_trigger_fe is None:
            raise ValueError("matched Behavior-only trigger requires the Proposed trigger FE")
        current_fe = int(behavior_row["FE"])
        if current_fe > int(matched_trigger_fe):
            raise ValueError("Behavior-only opportunity schedule skipped the Proposed trigger FE")
        return bool(current_fe == int(matched_trigger_fe)), None
    raise ValueError(f"unknown trigger_mode: {trigger_mode}")


def _query_fe(selector: OnlineSelector, dimension: int, fe_total: int) -> int:
    spec = get_query_spec(selector.model.query_id)
    if (
        selector.model.query_protocol != spec.protocol
        or selector.model.sample_design_id != spec.sample_design_id
    ):
        raise ValueError("online Selector and query specification use different query contracts")
    fe_query = int(spec.sample_design.sample_size(int(dimension)))
    if fe_query <= 0 or fe_query >= int(fe_total):
        raise ValueError("online query FE must be positive and smaller than FE_total")
    return fe_query


def _validate_query_feature_problem(
    row: dict[str, Any],
    problem_id: str,
    *,
    fe_query: int,
    fe_total: int,
) -> None:
    required = {
        "problem_id",
        "query_id",
        "query_protocol",
        "sample_design_id",
        "FE_query",
        "FE_total",
        "additional_function_evaluations",
    }
    missing = required.difference(row)
    if missing:
        raise ValueError(f"query feature row is missing online identity fields: {sorted(missing)}")
    spec = get_query_spec(str(row["query_id"]))
    if str(row["problem_id"]) != str(problem_id):
        raise ValueError("query feature row belongs to a different benchmark problem")
    if (
        str(row["query_protocol"]) != spec.protocol
        or str(row["sample_design_id"]) != spec.sample_design_id
    ):
        raise ValueError("query feature row does not match its canonical query protocol")
    if int(row["FE_query"]) != int(fe_query) or int(row["FE_total"]) != int(fe_total):
        raise ValueError("query feature FE metadata does not match the online run budget")
    if int(row["additional_function_evaluations"]) != 0:
        raise ValueError("query feature extraction must not add uncharged objective evaluations")


def _validate_query_sample_problem(
    row: dict[str, Any],
    *,
    query_feature_row: dict[str, Any],
    problem_id: str,
    fe_query: int,
    fe_total: int,
    success_gap_target: float,
) -> None:
    required = {
        "problem_id",
        "sample_design_id",
        "sampling_protocol",
        "sample_seed",
        "FE_query",
        "FE_total",
        "benchmark_reference_value",
        "success_gap_target",
        "query_success",
        "query_first_hit_offset",
        "query_best_gap",
        "X",
        "y",
    }
    missing = required.difference(row)
    if missing:
        raise ValueError(f"query sample row is missing online fields: {sorted(missing)}")
    spec = get_query_spec(str(query_feature_row["query_id"]))
    if str(row["problem_id"]) != str(problem_id):
        raise ValueError("query sample row belongs to a different benchmark problem")
    if (
        str(row["sample_design_id"]) != spec.sample_design_id
        or str(row["sampling_protocol"]) != spec.sample_design.protocol
    ):
        raise ValueError("query sample row does not match the requested sample design")
    if int(row["FE_query"]) != int(fe_query) or int(row["FE_total"]) != int(fe_total):
        raise ValueError("query sample FE metadata does not match the online budget")
    if int(row["sample_seed"]) != int(query_feature_row["sample_seed"]):
        raise ValueError("query sample and feature artifacts use different sample seeds")
    if not np.isclose(
        float(row["success_gap_target"]),
        float(success_gap_target),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("query sample success target does not match the suite config")
    _validate_saved_query_sample(row)


def _execute_real_query(
    *,
    problem: Problem,
    tracker: PathEvaluationTracker,
    deadline: float,
    query_feature_row: dict[str, Any],
    query_sample_row: dict[str, Any],
) -> QueryExecution:
    query_id = str(query_feature_row["query_id"])
    if query_id != MAIN_QUERY_ID:
        raise RuntimeError(
            "online real query replay currently supports descriptor_cheap_invariant only; "
            "pflacco paths require a callable isolated extractor with invariant preprocessing"
        )
    spec = get_query_spec(query_id)
    fe_query = int(query_sample_row["FE_query"])
    sample_seed = int(query_sample_row["sample_seed"])
    tracker.remember(
        query_called=True,
        query_feature_status="in_progress",
        query_feature_failure="",
    )

    sampling_started = perf_counter()
    if perf_counter() >= deadline:
        raise TimeoutError("online query exceeded policy_timeout_seconds before sampling")
    try:
        sampler = qmc.LatinHypercube(d=problem.dimension, seed=sample_seed)
        unit = sampler.random(n=fe_query)
        x = qmc.scale(unit, problem.lower_bounds, problem.upper_bounds)
        saved_x = np.asarray(query_sample_row["X"], dtype=float)
        if x.shape != saved_x.shape or not np.allclose(
            x, saved_x, rtol=0.0, atol=1e-12
        ):
            raise ValueError("regenerated LHS query does not match the saved sample_seed artifact")
    finally:
        runtime_sampling = float(perf_counter() - sampling_started)
        tracker.remember(
            runtime_query_sampling=runtime_sampling,
            runtime_query=runtime_sampling,
        )

    tracker.set_phase("query")
    evaluation_started = perf_counter()
    before = int(tracker.total_evaluations)
    try:
        y = np.asarray(problem.evaluate(x), dtype=float).reshape(-1)
    finally:
        runtime_evaluation = float(perf_counter() - evaluation_started)
        tracker.remember(
            runtime_query_evaluation=runtime_evaluation,
            runtime_query=float(runtime_sampling + runtime_evaluation),
        )
    if int(tracker.total_evaluations) - before != fe_query:
        raise RuntimeError("real query replay did not execute exactly FE_query objective evaluations")
    saved_y = np.asarray(query_sample_row["y"], dtype=float).reshape(-1)
    if y.shape != (fe_query,) or not np.isfinite(y).all():
        raise FloatingPointError("real query replay returned invalid objective values")
    if saved_y.shape != y.shape or not np.allclose(y, saved_y, rtol=1e-12, atol=1e-12):
        raise ValueError("real query objective values disagree with the saved sample artifact")
    reference = float(query_sample_row["benchmark_reference_value"])
    target = float(query_sample_row["success_gap_target"])
    gaps = np.maximum(y - reference, 0.0)
    hits = np.flatnonzero(gaps <= target)
    first_hit_offset = int(hits[0] + 1) if hits.size else None
    best_gap = float(np.min(gaps))
    tracker.remember(
        query_success=first_hit_offset is not None,
        query_first_hit_offset=first_hit_offset,
        query_best_gap=best_gap,
    )
    saved_offset_raw = query_sample_row["query_first_hit_offset"]
    saved_offset = None if pd.isna(saved_offset_raw) else int(saved_offset_raw)
    if (
        saved_offset != first_hit_offset
        or bool(query_sample_row["query_success"]) != bool(hits.size)
        or not np.isclose(
            float(query_sample_row["query_best_gap"]), best_gap, rtol=0.0, atol=1e-12
        )
    ):
        raise ValueError("real query first-hit endpoints disagree with the saved sample artifact")

    feature_started = perf_counter()
    failure = ""
    try:
        raw_features = calculate_descriptor_cheap(
            x,
            y,
            problem.lower_bounds,
            problem.upper_bounds,
        )
        features = {
            column: (
                float(raw_features[column])
                if np.isfinite(float(raw_features[column]))
                else None
            )
            for column in spec.feature_columns
        }
        mismatches = []
        for column in spec.feature_columns:
            actual = features[column]
            expected = query_feature_row.get(column)
            expected_missing = expected is None or pd.isna(expected)
            if actual is None and expected_missing:
                continue
            if actual is None or expected_missing or not np.isclose(
                float(actual), float(expected), rtol=1e-10, atol=1e-12
            ):
                mismatches.append(column)
        if mismatches:
            failure = "QueryFeatureMismatch: " + ",".join(mismatches[:16])
    except Exception as exc:
        features = {column: None for column in spec.feature_columns}
        failure = f"{type(exc).__name__}: {exc}"[:500]
    runtime_features = float(perf_counter() - feature_started)
    tracker.remember(
        runtime_query_feature_computation=runtime_features,
        runtime_query=float(runtime_sampling + runtime_evaluation + runtime_features),
        query_feature_status="failed" if failure else "ok",
        query_feature_failure=failure,
    )
    if perf_counter() >= deadline:
        raise TimeoutError("online query exceeded policy_timeout_seconds")
    expected_status = str(query_feature_row.get("feature_status", ""))
    status = "failed" if failure else "ok"
    if status != expected_status:
        failure = (
            failure
            or f"QueryFeatureStatusMismatch: replay={status}, saved={expected_status}"
        )
        status = "failed"
    tracker.remember(query_feature_status=status, query_feature_failure=failure)
    return QueryExecution(
        features=features,
        status=status,
        failure=failure,
        runtime_sampling=runtime_sampling,
        runtime_evaluation=runtime_evaluation,
        runtime_feature_computation=runtime_features,
        runtime_total=float(runtime_sampling + runtime_evaluation + runtime_features),
        first_hit_offset=first_hit_offset,
        best_gap=best_gap,
    )


def _empty_trigger_sampling_metadata() -> dict[str, Any]:
    return {f"trigger_{column}": None for column in SAMPLING_METADATA_COLUMNS}


def _base_policy_row(
    *,
    policy_name: str,
    policy_category: str,
    sampling_protocol: str,
    decision_check_frequency: str,
    repetition: int | None,
    prefix_algorithm: str,
    default_algorithm: str,
    selected_algorithm: str,
    selected_action: str,
    selected_equals_default: bool,
    selected_equals_prefix: bool,
    handoff_required: bool,
    selector_status: str,
    optimizer_transition_mode: str,
    selector_target_transform: str,
    selector_remaining_budget_ratio: float | None,
    query_called: bool,
    query_id: str,
    query_protocol: str,
    sample_design_id: str,
    query_feature_status: str,
    query_feature_failure: str,
    query_failure_fallback: bool,
    trigger_fe: int | None,
    trigger_fe_ratio: float | None,
    decision_score: float | None,
    decision_threshold: float | None,
    decision_check_count: int,
    trigger_sampling_metadata: dict[str, Any],
    fe_total: int,
    fe_probe: int,
    fe_query: int,
    fe_after: int,
    runtime_probe: float,
    runtime_query_sampling: float,
    runtime_query_evaluation: float,
    runtime_query_feature_computation: float,
    runtime_query: float,
    runtime_selection: float,
    runtime_handoff: float,
    runtime_after: float,
    final_performance: float,
) -> dict[str, Any]:
    fe_used = int(fe_probe) + int(fe_query) + int(fe_after)
    if fe_used != int(fe_total):
        raise ValueError("policy FE accounting must satisfy FE_probe + FE_query + FE_after = FE_total")
    runtime_values = np.asarray(
        [runtime_probe, runtime_query, runtime_selection, runtime_handoff, runtime_after],
        dtype=float,
    )
    if not np.isfinite(runtime_values).all() or (runtime_values < 0.0).any():
        raise ValueError("policy runtimes must be finite and non-negative")
    if bool(handoff_required) != (
        str(optimizer_transition_mode) == "population_transfer_initialization"
    ):
        raise ValueError("handoff_required must match the optimizer transition mode")
    if bool(selected_equals_prefix) == bool(handoff_required):
        raise ValueError("handoff_required must equal not selected_equals_prefix")
    return {
        "policy_name": str(policy_name),
        "policy_category": str(policy_category),
        "sampling_protocol": str(sampling_protocol),
        "decision_check_frequency": str(decision_check_frequency),
        "random_repetition": repetition,
        "prefix_algorithm": str(prefix_algorithm),
        "default_algorithm": str(default_algorithm),
        "no_query_algorithm": str(default_algorithm),
        "selected_algorithm": str(selected_algorithm),
        "selected_action": str(selected_action),
        "selected_equals_default": bool(selected_equals_default),
        "selected_equals_prefix": bool(selected_equals_prefix),
        "handoff_required": bool(handoff_required),
        "handoff_type": str(optimizer_transition_mode),
        "query_transition_mode": str(optimizer_transition_mode),
        "selector_status": str(selector_status),
        "optimizer_transition_mode": str(optimizer_transition_mode),
        "selector_target_transform": str(selector_target_transform),
        "selector_remaining_budget_ratio": selector_remaining_budget_ratio,
        "query_called": bool(query_called),
        "query_id": str(query_id),
        "query_protocol": str(query_protocol),
        "sample_design_id": str(sample_design_id),
        "query_feature_status": str(query_feature_status),
        "query_feature_failure": str(query_feature_failure),
        "query_failure_fallback": bool(query_failure_fallback),
        "query_success": False,
        "query_first_hit_offset": None,
        "query_best_gap": None,
        "trigger_FE": trigger_fe,
        "trigger_FE_ratio": trigger_fe_ratio,
        "decision_score": decision_score,
        "decision_threshold": decision_threshold,
        "decision_check_count": int(decision_check_count),
        "FE_total": int(fe_total),
        "FE_probe": int(fe_probe),
        "FE_query": int(fe_query),
        "FE_after": int(fe_after),
        "FE_used": fe_used,
        "runtime_probe": float(runtime_probe),
        "runtime_query_sampling": float(runtime_query_sampling),
        "runtime_query_evaluation": float(runtime_query_evaluation),
        "runtime_query_feature_computation": float(runtime_query_feature_computation),
        "runtime_query": float(runtime_query),
        "runtime_selection": float(runtime_selection),
        "runtime_handoff": float(runtime_handoff),
        "runtime_fresh_initialization": 0.0,
        "runtime_after_includes_fresh_initialization": False,
        "runtime_after": float(runtime_after),
        "runtime_component_sum_diagnostic": float(np.sum(runtime_values)),
        "final_performance": float(final_performance),
        **trigger_sampling_metadata,
    }


def _collapse_timing_repetition_frame(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "problem_id",
        "dimension",
        "seed",
        "policy_name",
        "random_repetition",
        "timing_repetition",
        "runtime_full_run_wall_clock_median",
        "runtime_full_run_wall_clock_raw_observed_median",
        "runtime_full_run_wall_clock_censored_repetitions",
        "scientific_endpoint_stage",
        "scientific_path_status",
        "scientific_path_timed_out",
        "scientific_path_completed",
        "timing_replay_instability",
        "timing_replay_completed_repetitions",
        "timing_replay_failure_repetitions",
        "timing_replay_timeout_repetitions",
        "timing_replay_status_repetitions",
        "timing_replay_timed_out_flags",
        "timing_replay_path_completed_flags",
        "timing_replay_path_identity_consistent",
        "completed_timing_replay_outcomes_internally_consistent",
        "stage_a_to_completed_timing_replays_consistent",
        "timing_replay_status_instability",
        "stage_a_stage_b_completion_status_instability",
        "policy_timeout_seconds",
        "log10_gap",
        "observed_first_hit_FE",
        "target_hit_observed",
        "target_hit_before_failure",
        "endpoint_success",
        "first_hit_FE",
        "success",
        "ert_FE_contribution",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"online timing rows are missing endpoint fields: {sorted(missing)}")
    key = ["problem_id", "dimension", "seed", "policy_name", "random_repetition"]
    collapsed: list[dict[str, Any]] = []
    for _, frame in rows.groupby(key, sort=True, dropna=False):
        ordered = frame.sort_values("timing_repetition", kind="mergesort")
        indices = ordered["timing_repetition"].astype(int).tolist()
        if indices != list(range(TIMING_REPETITIONS)):
            raise ValueError("online summary refuses paths without exactly three timing repetitions")
        stages = set(ordered["scientific_endpoint_stage"].astype(str))
        if stages != {"pre_specified_stage_a_run"}:
            raise ValueError("online endpoints must come from the pre-specified Stage-A run")
        scientific_statuses = set(ordered["scientific_path_status"].astype(str))
        if not scientific_statuses.issubset({"completed", "timed_out", "failed"}) or len(
            scientific_statuses
        ) != 1:
            raise ValueError("online paths must retain one valid Stage-A scientific status")
        scientific_status = next(iter(scientific_statuses))
        if not bool(
            (ordered["scientific_path_completed"].astype(bool) == (scientific_status == "completed")).all()
        ) or not bool(
            (ordered["scientific_path_timed_out"].astype(bool) == (scientific_status == "timed_out")).all()
        ):
            raise ValueError("online Stage-A status and timeout/completion flags disagree")
        replay_statuses = tuple(
            str(value) for value in ordered.iloc[0]["timing_replay_status_repetitions"]
        )
        replay_timed_out = tuple(
            bool(value) for value in ordered.iloc[0]["timing_replay_timed_out_flags"]
        )
        replay_completed = tuple(
            bool(value) for value in ordered.iloc[0]["timing_replay_path_completed_flags"]
        )
        if len(replay_statuses) != TIMING_REPETITIONS or not set(
            replay_statuses
        ).issubset({"completed", "timed_out", "failed"}):
            raise ValueError("online timing replays must retain three valid status values")
        if replay_completed != tuple(status == "completed" for status in replay_statuses) or replay_timed_out != tuple(
            status == "timed_out" for status in replay_statuses
        ):
            raise ValueError("online Stage-B status and timeout/completion flags disagree")
        expected_replay_counts = {
            "timing_replay_completed_repetitions": sum(replay_completed),
            "timing_replay_timeout_repetitions": sum(replay_timed_out),
            "timing_replay_failure_repetitions": sum(
                status == "failed" for status in replay_statuses
            ),
        }
        for column, expected in expected_replay_counts.items():
            if not bool((ordered[column].astype(int) == int(expected)).all()):
                raise ValueError(f"online {column} is inconsistent")
        if not bool(ordered["timing_replay_path_identity_consistent"].astype(bool).all()):
            raise ValueError("online timing replay path identity must remain consistent")
        completed_count = sum(replay_completed)
        internal_values = ordered[
            "completed_timing_replay_outcomes_internally_consistent"
        ].tolist()
        if completed_count >= 2:
            if not all(not pd.isna(value) and bool(value) for value in internal_values):
                raise ValueError("online completed timing replays must agree internally")
        elif any(not pd.isna(value) for value in internal_values):
            raise ValueError(
                "online internal completed-replay consistency must be null when not evaluable"
            )
        stage_a_values = ordered[
            "stage_a_to_completed_timing_replays_consistent"
        ].tolist()
        stage_a_applicable = scientific_status == "completed" and completed_count >= 1
        if stage_a_applicable:
            if not all(not pd.isna(value) and bool(value) for value in stage_a_values):
                raise ValueError("online completed timing replays must agree with Stage A")
        elif any(not pd.isna(value) for value in stage_a_values):
            raise ValueError(
                "online Stage-A replay consistency must be null when not evaluable"
            )
        expected_status_instability = len(set(replay_statuses)) > 1
        if not bool(
            (
                ordered["timing_replay_status_instability"].astype(bool)
                == expected_status_instability
            ).all()
        ):
            raise ValueError("online timing_replay_status_instability is inconsistent")
        expected_completion_instability = any(
            completed != (scientific_status == "completed")
            for completed in replay_completed
        )
        if not bool(
            (
                ordered["stage_a_stage_b_completion_status_instability"].astype(bool)
                == expected_completion_instability
            ).all()
        ):
            raise ValueError(
                "online Stage-A/Stage-B completion-status instability is inconsistent"
            )
        first_hit_values = ordered["observed_first_hit_FE"].tolist()
        if any(
            not _timing_value_equal(first_hit, alias)
            for first_hit, alias in zip(
                first_hit_values, ordered["first_hit_FE"].tolist(), strict=True
            )
        ):
            raise ValueError("online first_hit_FE must alias observed_first_hit_FE")
        target_hit = ordered["target_hit_observed"].astype(bool).to_numpy()
        expected_target_hit = np.asarray(
            [value is not None and not pd.isna(value) for value in first_hit_values],
            dtype=bool,
        )
        if not np.array_equal(target_hit, expected_target_hit) or not np.array_equal(
            ordered["success"].astype(bool).to_numpy(), target_hit
        ):
            raise ValueError("online target-hit aliases are inconsistent")
        scientific_completed_flags = ordered["scientific_path_completed"].astype(
            bool
        ).to_numpy()
        if not np.array_equal(
            ordered["target_hit_before_failure"].astype(bool).to_numpy(),
            target_hit & ~scientific_completed_flags,
        ) or not np.array_equal(
            ordered["endpoint_success"].astype(bool).to_numpy(),
            target_hit & scientific_completed_flags,
        ):
            raise ValueError("online target-hit/completion endpoint fields are inconsistent")
        row = ordered.iloc[0].to_dict()
        row["timing_raw_rows"] = TIMING_REPETITIONS
        row["success_weight"] = float(bool(row["success"]))
        row["target_hit_observed_weight"] = float(bool(row["target_hit_observed"]))
        row["endpoint_success_weight"] = float(bool(row["endpoint_success"]))
        row["query_call_weight"] = float(bool(row["query_called"]))
        row["query_failure_weight"] = float(
            bool(row["query_called"]) and str(row["query_feature_status"]) != "ok"
        )
        row["handoff_weight"] = float(bool(row["handoff_required"]))
        row["failure_weight"] = float(str(row["path_status"]) != "completed")
        row["timeout_weight"] = float(bool(row["timeout"]))
        row["coverage_weight"] = float(bool(row["coverage_complete"]))
        row["timing_replay_instability_weight"] = float(
            bool(row["timing_replay_instability"])
        )
        row["timing_replay_failure_weight"] = float(
            int(row["timing_replay_failure_repetitions"]) > 0
        )
        row["timing_replay_timeout_weight"] = float(
            int(row["timing_replay_timeout_repetitions"]) > 0
        )
        collapsed.append(row)
    return pd.DataFrame(collapsed)


def _online_policy_run_frame(rows: pd.DataFrame) -> pd.DataFrame:
    collapsed = _collapse_timing_repetition_frame(rows)
    fixed = collapsed[collapsed["policy_name"].astype(str) != "matched_rate_random"].copy()
    random_rows = collapsed[
        collapsed["policy_name"].astype(str) == "matched_rate_random"
    ].copy()
    if random_rows.empty:
        return fixed.reset_index(drop=True)
    run_key = ["problem_id", "dimension", "seed"]
    averaged: list[dict[str, Any]] = []
    numeric_means = (
        "log10_gap",
        "final_gap",
        "observed_final_gap",
        "runtime_full_run_wall_clock_median",
        "log10_runtime_ratio_vs_skip",
        "success_weight",
        "target_hit_observed_weight",
        "endpoint_success_weight",
        "query_call_weight",
        "query_failure_weight",
        "handoff_weight",
        "failure_weight",
        "timeout_weight",
        "coverage_weight",
        "timing_replay_instability_weight",
        "timing_replay_failure_weight",
        "timing_replay_timeout_weight",
        "ert_FE_contribution",
        "planned_total_FE",
        "effective_total_FE",
        "FE_query",
        "FE_used",
        "decision_check_count",
    )
    for _, frame in random_rows.groupby(run_key, sort=True, dropna=False):
        repetitions = sorted(frame["random_repetition"].astype(int).tolist())
        if repetitions != list(range(len(repetitions))):
            raise ValueError("matched Random repetitions must be consecutive within each run")
        row = frame.iloc[0].to_dict()
        for column in numeric_means:
            row[column] = float(frame[column].astype(float).mean())
        trigger = frame["trigger_FE_ratio"].dropna().astype(float)
        row["trigger_FE_ratio"] = float(trigger.mean()) if len(trigger) else None
        row["random_repetition"] = None
        row["random_repetitions_averaged_within_run"] = int(len(frame))
        averaged.append(row)
    fixed["random_repetitions_averaged_within_run"] = 1
    return pd.concat([fixed, pd.DataFrame(averaged)], ignore_index=True, sort=False)


def _policy_summary(policy_runs: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    layers = {
        "overall": [],
        "family": ["family"],
        "dimension": ["dimension"],
        "family_dimension": ["family", "dimension"],
    }
    for layer, columns in layers.items():
        for policy_name, policy_frame in policy_runs.groupby("policy_name", sort=True):
            if columns:
                groups = policy_frame.groupby(columns, dropna=False, sort=True)
            else:
                groups = [((), policy_frame)]
            for values, frame in groups:
                if columns and not isinstance(values, tuple):
                    values = (values,)
                group = dict(zip(columns, values, strict=False)) if columns else {}
                summary_rows.append(_summary_row(frame, layer, group, str(policy_name)))
    return pd.DataFrame(summary_rows).sort_values(["layer", "group", "policy_name"]).reset_index(drop=True)


def _summary_row(frame: pd.DataFrame, layer: str, group: dict[str, Any], policy_name: str) -> dict[str, Any]:
    success_count = float(frame["success_weight"].astype(float).sum())
    ert_numerator = float(frame["ert_FE_contribution"].astype(float).sum())
    return {
        "layer": layer,
        "group": _group_label(group),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "policy_name": policy_name,
        "runs": int(len(frame)),
        "mean_log10_gap": float(frame["log10_gap"].astype(float).mean()),
        "median_log10_gap": float(frame["log10_gap"].astype(float).median()),
        "success_rate": float(frame["success_weight"].astype(float).mean()),
        "target_hit_observed_rate": float(
            frame["target_hit_observed_weight"].astype(float).mean()
        ),
        "endpoint_success_rate": float(
            frame["endpoint_success_weight"].astype(float).mean()
        ),
        "ERT_FE": float(ert_numerator / success_count) if success_count > 0.0 else None,
        "ERT_numerator_FE": ert_numerator,
        "ERT_success_count": success_count,
        "query_call_rate": float(frame["query_call_weight"].astype(float).mean()),
        "query_failure_rate": float(frame["query_failure_weight"].astype(float).mean()),
        "handoff_rate": float(frame["handoff_weight"].astype(float).mean()),
        "failure_rate": float(frame["failure_weight"].astype(float).mean()),
        "timeout_rate": float(frame["timeout_weight"].astype(float).mean()),
        "coverage": float(frame["coverage_weight"].astype(float).sum() / len(frame)),
        "timing_replay_instability_rate": float(
            frame["timing_replay_instability_weight"].astype(float).mean()
        ),
        "timing_replay_failure_rate": float(
            frame["timing_replay_failure_weight"].astype(float).mean()
        ),
        "timing_replay_timeout_rate": float(
            frame["timing_replay_timeout_weight"].astype(float).mean()
        ),
        "mean_effective_total_FE": float(frame["effective_total_FE"].astype(float).mean()),
        "median_full_run_wall_clock": float(
            frame["runtime_full_run_wall_clock_median"].astype(float).median()
        ),
        "mean_log10_runtime_ratio_vs_skip": float(
            frame["log10_runtime_ratio_vs_skip"].astype(float).mean()
        ),
        "mean_trigger_FE_ratio": float(frame["trigger_FE_ratio"].dropna().mean()) if frame["trigger_FE_ratio"].notna().any() else None,
    }


def _relative_summary(policy_runs: pd.DataFrame) -> pd.DataFrame:
    baseline = policy_runs[policy_runs["policy_name"] == "sbs_no_query"][
        [
            "problem_id",
            "dimension",
            "seed",
            "log10_gap",
            "success_weight",
            "runtime_full_run_wall_clock_median",
            "ert_FE_contribution",
        ]
    ].rename(
        columns={
            "log10_gap": "sbs_no_query_log10_gap",
            "success_weight": "sbs_no_query_success_weight",
            "runtime_full_run_wall_clock_median": "sbs_no_query_runtime",
            "ert_FE_contribution": "sbs_no_query_ert_FE_contribution",
        }
    )
    if baseline.duplicated(["problem_id", "dimension", "seed"]).any():
        raise ValueError("SBS online endpoint rows must be unique by run")
    joined = policy_runs.merge(baseline, on=["problem_id", "dimension", "seed"], how="left")
    if joined["sbs_no_query_log10_gap"].isna().any():
        raise ValueError("online relative summary is missing a paired SBS path")
    joined["log10_gap_delta_vs_sbs_no_query"] = (
        joined["log10_gap"] - joined["sbs_no_query_log10_gap"]
    )
    joined["success_rate_delta_vs_sbs_no_query"] = (
        joined["success_weight"] - joined["sbs_no_query_success_weight"]
    )
    joined["paired_log10_runtime_ratio_vs_sbs_no_query"] = np.log10(
        joined["runtime_full_run_wall_clock_median"].astype(float)
        / joined["sbs_no_query_runtime"].astype(float)
    )
    summary_rows = []
    for policy_name, frame in joined.groupby("policy_name", sort=True):
        policy_success = float(frame["success_weight"].astype(float).sum())
        baseline_success = float(frame["sbs_no_query_success_weight"].astype(float).sum())
        policy_ert = (
            float(frame["ert_FE_contribution"].astype(float).sum() / policy_success)
            if policy_success > 0.0
            else None
        )
        baseline_ert = (
            float(
                frame["sbs_no_query_ert_FE_contribution"].astype(float).sum()
                / baseline_success
            )
            if baseline_success > 0.0
            else None
        )
        summary_rows.append(
            {
                "policy_name": str(policy_name),
                "runs": int(len(frame)),
                "mean_log10_gap_delta_vs_sbs_no_query": float(
                    frame["log10_gap_delta_vs_sbs_no_query"].mean()
                ),
                "median_log10_gap_delta_vs_sbs_no_query": float(
                    frame["log10_gap_delta_vs_sbs_no_query"].median()
                ),
                "mean_success_rate_delta_vs_sbs_no_query": float(
                    frame["success_rate_delta_vs_sbs_no_query"].mean()
                ),
                "mean_log10_runtime_ratio_vs_sbs_no_query": float(
                    frame["paired_log10_runtime_ratio_vs_sbs_no_query"].mean()
                ),
                "ERT_FE": policy_ert,
                "sbs_no_query_ERT_FE": baseline_ert,
                "ERT_FE_delta_vs_sbs_no_query": (
                    float(policy_ert - baseline_ert)
                    if policy_ert is not None and baseline_ert is not None
                    else None
                ),
                "query_call_rate": float(frame["query_call_weight"].mean()),
            }
        )
    return pd.DataFrame(summary_rows).sort_values(
        "mean_log10_gap_delta_vs_sbs_no_query"
    ).reset_index(drop=True)


def _random_repetition_summary(rows: pd.DataFrame) -> pd.DataFrame:
    collapsed = _collapse_timing_repetition_frame(rows)
    random_rows = collapsed[collapsed["policy_name"] == "matched_rate_random"].copy()
    if random_rows.empty:
        return pd.DataFrame()
    summary_rows = []
    for repetition, frame in random_rows.groupby("random_repetition", sort=True):
        success_count = float(frame["success_weight"].sum())
        numerator = float(frame["ert_FE_contribution"].sum())
        summary_rows.append(
            {
                "policy_name": "matched_rate_random",
                "random_repetition": int(repetition),
                "runs": int(len(frame)),
                "mean_log10_gap": float(frame["log10_gap"].mean()),
                "success_rate": float(frame["success_weight"].mean()),
                "ERT_FE": float(numerator / success_count) if success_count > 0.0 else None,
                "query_call_rate": float(frame["query_call_weight"].mean()),
                "median_full_run_wall_clock": float(
                    frame["runtime_full_run_wall_clock_median"].median()
                ),
            }
        )
    return pd.DataFrame(summary_rows)


def _write_frame(frame: pd.DataFrame, stem: Path) -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), stem.with_suffix(".parquet"))


def _markdown_report(
    *,
    summary: dict[str, Any],
    policy_summary: pd.DataFrame,
    relative_summary: pd.DataFrame,
    random_repetition_summary: pd.DataFrame,
) -> str:
    overall = policy_summary[policy_summary["layer"] == "overall"][
        [
            "policy_name",
            "runs",
            "mean_log10_gap",
            "median_log10_gap",
            "success_rate",
            "ERT_FE",
            "query_call_rate",
            "failure_rate",
            "coverage",
            "median_full_run_wall_clock",
            "mean_log10_runtime_ratio_vs_skip",
            "mean_trigger_FE_ratio",
        ]
    ].sort_values("mean_log10_gap")
    return "\n".join(
        [
            "# CEC online controller evaluation",
            "",
            "## 摘要",
            "",
            f"- Controller: `{summary['model_name']}`，阈值口径 `{summary['threshold_mode']}`。",
            f"- Default/SBS optimizer: `{summary['default_algorithm']}`。",
            f"- Sampling protocol: `{summary['sampling_protocol']}`。",
            f"- Decision-check frequency: `{summary['decision_check_frequency']}`。",
            f"- Budget milestone ratios: `{', '.join(str(value) for value in summary['budget_milestone_ratios'])}`。",
            f"- Full-policy timeout: `{summary['policy_timeout_seconds']}` seconds；与 action continuation 和 decision-state timing replay 的 timeout 分字段保存。",
            "- 评价单位是 trajectory-level first-trigger run；主性能端点为配置截断后的 `log10_gap`、success 与 ERT。",
            "- 每条政策路径先执行 1 次预指定 Stage-A 科学运行固定端点，再执行 "
            f"`{summary['timing_repetitions']}` 次 Stage-B 真实计时 replay，顺序协议 "
            f"`{summary['timing_order_protocol']}`；Stage-B 只决定 wall-clock 中位数。",
            "- wall-clock 字段是 full-run 口径；本执行器不生成 nested learning 所需的 decision-state future-path timing。",
            "- Controller 只使用实时 behavior features；CEC rows 不参与训练、预处理拟合或阈值选择。",
            "- Query 后的 Selection Reference 使用冻结的 BBOB-train statewise action-loss regressor，并连续接收 remaining budget；CEC rows 不参与 selector 拟合。",
            "- 每个动态采样状态同时是 behavior observation 点和可能触发固定 query 的 decision opportunity。",
            "- `always_query` 表示在当前 sampling protocol 的第一个决策机会必定执行固定 query 的 after-probe baseline。",
            "",
            "## Overall Policies",
            "",
            _markdown_table(overall),
            "",
            "## Relative To SBS/No-query",
            "",
            _markdown_table(relative_summary),
            "",
            "## Matched-rate Random Repetition Summary",
            "",
            _markdown_table(random_repetition_summary),
            "",
            "## Outputs",
            "",
            f"- Policy runs: `{summary['outputs']['policy_runs']}`",
            f"- Policy endpoints: `{summary['outputs']['policy_endpoints']}`",
            f"- Policy summary: `{summary['outputs']['policy_summary']}`",
            f"- Relative summary: `{summary['outputs']['relative_summary']}`",
        ]
    ) + "\n"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _group_label(group: dict[str, Any]) -> str:
    if not group:
        return "all"
    return ", ".join(f"{key}={value}" for key, value in group.items())


def _split_name(config: dict) -> str:
    if "split" in config:
        return str(config["split"])
    return Path(config["output"]).stem.removesuffix("_trajectories")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run online CEC controller evaluation with frozen BBOB controller.")
    parser.add_argument("--query-id", choices=sorted(LANDSCAPE_QUERY_SPECS), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH)
    parser.add_argument("--query-features", type=Path, default=None)
    parser.add_argument("--query-samples", type=Path, default=None)
    parser.add_argument("--selector-model", type=Path, default=None)
    parser.add_argument("--pre-run-aas-selector-model", type=Path, default=None)
    parser.add_argument("--behavior-only-selector-model", type=Path, default=None)
    parser.add_argument("--matched-random-calibration", type=Path, default=None)
    parser.add_argument("--training-summary", type=Path, default=None)
    parser.add_argument("--milestone-only-training-summary", type=Path, default=None)
    parser.add_argument("--behavior-only-training-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--threshold-mode", default=DEFAULT_THRESHOLD_MODE)
    parser.add_argument("--sampling-protocol", choices=SAMPLING_PROTOCOLS, default=DEFAULT_SAMPLING_PROTOCOL)
    parser.add_argument("--random-repetitions", type=int, default=DEFAULT_RANDOM_REPETITIONS)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    parser.add_argument("--only-seed", type=int, action="append", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--sharded", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    split = _split_name(config)
    query_features = args.query_features or Path("results/landscape_queries/features") / args.query_id / split / "features.parquet"
    selector_model = args.selector_model or Path("results/selection_reference") / args.query_id / "statewise_selector.joblib"
    training_summary = (
        args.training_summary
        or Path("results/decision")
        / args.query_id
        / "feature_group_ablation/B3/all_accepted/full_decision_model_training_summary.json"
    )
    milestone_only_training_summary = (
        args.milestone_only_training_summary
        or Path("results/decision")
        / args.query_id
        / "feature_group_ablation/T0/milestone_only/full_decision_model_training_summary.json"
    )
    behavior_only_training_summary = args.behavior_only_training_summary or training_summary
    pre_run_aas_selector_model = (
        args.pre_run_aas_selector_model
        or Path("results/selection_reference")
        / args.query_id
        / "query_only_fe0_selector.joblib"
    )
    behavior_only_selector_model = (
        args.behavior_only_selector_model
        or Path("results/selection_reference/behavior_only_full_budget/statewise_selector.joblib")
    )
    matched_random_calibration = (
        args.matched_random_calibration
        or Path("results/decision")
        / args.query_id
        / "controller_baseline_comparison/matched_random_calibration.json"
    )
    output_dir = args.output_dir or Path("results/decision") / args.query_id / split / "online_controller_evaluation"

    evaluate_online_controller(
        query_id=args.query_id,
        query_feature_path=query_features,
        query_sample_path=args.query_samples,
        config_path=args.config,
        train_config_path=args.train_config,
        selector_model_path=selector_model,
        pre_run_aas_selector_model_path=pre_run_aas_selector_model,
        behavior_only_selector_model_path=behavior_only_selector_model,
        matched_random_calibration_path=matched_random_calibration,
        training_summary_path=training_summary,
        milestone_only_training_summary_path=milestone_only_training_summary,
        behavior_only_training_summary_path=behavior_only_training_summary,
        output_dir=output_dir,
        model_name=args.model_name,
        threshold_mode=args.threshold_mode,
        sampling_protocol=args.sampling_protocol,
        random_repetitions=args.random_repetitions,
        only_functions=args.only_function,
        only_dimensions=args.only_dimension,
        only_seeds=args.only_seed,
        max_runs=args.max_runs,
        sharded=args.sharded,
        summarize_only=args.summarize_only,
        workers=args.workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
