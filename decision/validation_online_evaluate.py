from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from benchmarks import make_problem
from decision.model_protocol import PREDEFINED_THRESHOLD_MODE, decision_scores
from experiments.phase1_batch_common import (
    fe_total_for_dimension,
    load_suite_configs,
    make_shards,
    runtime_problem_config,
    split_name,
    validate_dynamic_collection_config,
)
from landscape_queries.cheap import calculate_descriptor_cheap
from landscape_queries.sampling import sample_problem
from landscape_queries.specs import MAIN_QUERY_ID, get_query_spec
from optimizers import OptimizerSettings, advance_optimizer_state, initialize_optimizer_state
from selection_reference.action_losses import (
    QUERY_ADJUSTED_BUDGET,
    _TimedObjective,
    _evaluate_native_skip,
    _evaluate_one_candidate_action,
    _validate_replayed_checkpoint,
)
from selection_reference.model import load_selector_model, make_selector_features


DEFAULT_ARTIFACT_DIR = Path(
    "results/decision/descriptor_cheap_invariant/feature_group_ablation/B3/all_accepted"
)
RUN_COLUMNS = (
    "split",
    "problem_id",
    "function_id",
    "family",
    "cv_group_id",
    "dimension",
    "prefix_algorithm",
    "seed",
)
STATE_COLUMNS = (*RUN_COLUMNS, "FE")
OUTCOME_NAME = "validation_online_policy_outcomes.parquet"
RESOURCE_NAME = "validation_online_deployment_metrics.parquet"
SUMMARY_NAME = "validation_online_evaluation_summary.parquet"
RESOURCE_SUMMARY_NAME = "validation_online_deployment_summary.parquet"
POLICY_NAME = "selected_nested_oof_query_full_hybrid"
TIMING_SOURCE = "measured_complete_online_policy_path"


def run_validation_online_evaluation(
    *,
    online_input_path: Path,
    decision_dataset_path: Path,
    training_summary_path: Path,
    selector_model_path: Path,
    validation_config_path: Path,
    output_dir: Path,
    workers: int,
    overwrite: bool,
    summarize_only: bool,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be at least one")
    configs = _config_index(validation_config_path)
    online = pq.read_table(online_input_path).to_pandas()
    decisions = pq.read_table(decision_dataset_path).to_pandas()
    prepared, protocol = _verify_and_prepare_policy_input(
        online=online,
        decisions=decisions,
        training_summary_path=training_summary_path,
        selector_model_path=selector_model_path,
        configs=configs,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    jobs = _make_jobs(
        prepared=prepared,
        configs=configs,
        decision_model_path=Path(protocol["model_path"]),
        feature_columns=list(protocol["feature_columns"]),
        threshold=float(protocol["threshold"]),
        selector_model_path=selector_model_path,
        shard_dir=shard_dir,
        overwrite=overwrite,
    )
    if not summarize_only:
        _execute_jobs(jobs, workers)
    missing = [Path(job["output_path"]) for job in jobs if not Path(job["output_path"]).exists()]
    if missing:
        return {
            "status": "incomplete",
            "planned_run_shards": int(len(jobs)),
            "missing_run_shards": int(len(missing)),
            "missing_examples": [str(path) for path in missing[:8]],
        }
    outcomes, resources = _collect_outputs(jobs)
    _validate_outputs(outcomes, resources, prepared, protocol)
    outcome_path = output_dir / OUTCOME_NAME
    resource_path = output_dir / RESOURCE_NAME
    pq.write_table(pa.Table.from_pandas(outcomes, preserve_index=False), outcome_path)
    pq.write_table(pa.Table.from_pandas(resources, preserve_index=False), resource_path)
    summary = _scientific_summary(outcomes)
    resource_summary = _resource_summary(resources)
    pq.write_table(
        pa.Table.from_pandas(summary, preserve_index=False), output_dir / SUMMARY_NAME
    )
    pq.write_table(
        pa.Table.from_pandas(resource_summary, preserve_index=False),
        output_dir / RESOURCE_SUMMARY_NAME,
    )
    result = {
        "status": "ok",
        "policy_name": POLICY_NAME,
        "model_name": protocol["model_name"],
        "threshold_mode": protocol["threshold_mode"],
        "threshold": protocol["threshold"],
        "selector_type": protocol["selector_type"],
        "selector_input_mode": protocol["selector_input_mode"],
        "runs": int(len(outcomes)),
        "query_calls": int(outcomes["query_called"].sum()),
        "completed_paths": int(outcomes["path_completed"].sum()),
        "target_hits": int(outcomes["target_hit_observed"].sum()),
        "no_query_native_gap_close_runs": int(
            outcomes.loc[
                ~outcomes["query_called"].to_numpy(bool),
                "native_endpoint_gap_close_rtol_1e_10_atol_1e_12",
            ].fillna(False).sum()
        ),
        "no_query_native_runs": int((~outcomes["query_called"].to_numpy(bool)).sum()),
        "no_query_native_target_hit_matches": int(
            outcomes.loc[
                ~outcomes["query_called"].to_numpy(bool),
                "native_endpoint_target_hit_match",
            ].fillna(False).sum()
        ),
        "max_abs_no_query_native_gap_difference": float(
            pd.to_numeric(
                outcomes["native_replay_gap_difference"], errors="coerce"
            ).abs().max()
        ),
        "max_abs_no_query_native_log10_gap_difference": float(
            pd.to_numeric(
                outcomes["native_replay_log10_gap_difference"], errors="coerce"
            ).abs().max()
        ),
        "outcome": str(outcome_path),
        "deployment_metrics": str(resource_path),
        "scientific_summary": str(output_dir / SUMMARY_NAME),
        "deployment_summary": str(output_dir / RESOURCE_SUMMARY_NAME),
        "wall_clock_excluded_from_labels_and_scientific_utility": True,
    }
    (output_dir / "validation_online_evaluation_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def _config_index(path: Path) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for config in load_suite_configs(path):
        config = dict(config)
        validate_dynamic_collection_config(config)
        storage_split = split_name(config)
        if storage_split in configs:
            raise ValueError(f"duplicate validation storage split: {storage_split}")
        configs[storage_split] = config
    if set(configs) != {"bbob_validation", "mabbob_validation"}:
        raise ValueError(
            "validation online evaluation requires bbob_validation and mabbob_validation"
        )
    return configs


def _verify_and_prepare_policy_input(
    *,
    online: pd.DataFrame,
    decisions: pd.DataFrame,
    training_summary_path: Path,
    selector_model_path: Path,
    configs: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_online = {
        *STATE_COLUMNS,
        "dataset_role",
        "model_name",
        "threshold_mode",
        "threshold",
        "decision_score",
        "decision_run_query",
        "selected_algorithm",
        "selected_equals_default",
        "selected_equals_prefix",
        "handoff_required",
        "handoff_type",
    }
    missing = sorted(required_online.difference(online.columns))
    if missing:
        raise ValueError(f"online evaluation input is missing columns: {missing}")
    if online.empty or set(online["dataset_role"].astype(str)) != {"validation"}:
        raise ValueError("online evaluation input must contain validation rows only")
    if set(online["split"].astype(str)) != set(configs):
        raise ValueError("online evaluation input does not cover both configured validation splits")
    if online.duplicated(list(STATE_COLUMNS)).any():
        raise ValueError("online evaluation input contains duplicate decision states")

    summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    model_name = str(summary.get("selected_model_name", ""))
    if model_name != "random_forest_regressor":
        raise ValueError(f"training summary selected unsupported Decision model: {model_name}")
    feature_columns = [str(value) for value in summary.get("feature_columns", [])]
    if not feature_columns or not all(value.startswith("bf_") for value in feature_columns):
        raise ValueError("the selected Decision model must use behavior features only")
    if str(summary.get("feature_group")) not in {
        "B2+Motion+SearchMaturity",
        "B2+Motion+SearchMaturityLinear",
        "B2+Motion+ExploreExploitRatio",
    } or str(summary.get("opportunity_scope")) != "all_accepted":
        raise ValueError("the Decision model must use one predefined maturity ablation/all_accepted")
    model_artifacts = {
        str(row["model_name"]): Path(str(row["model_path"]))
        for row in summary.get("model_artifacts", [])
    }
    model_path = model_artifacts.get(model_name)
    if model_path is None or not model_path.exists():
        raise FileNotFoundError("the selected Random Forest Regressor model artifact is missing")
    threshold_path = Path(str(summary["outputs"]["decision_thresholds"]))
    thresholds = pq.read_table(threshold_path).to_pandas()
    threshold_row = thresholds[
        thresholds["model_name"].astype(str).eq(model_name)
        & thresholds["threshold_mode"].astype(str).eq(PREDEFINED_THRESHOLD_MODE)
    ]
    if len(threshold_row) != 1:
        raise ValueError("the selected formal Decision threshold is not unique")
    threshold = float(threshold_row.iloc[0]["threshold"])
    if int(threshold_row.iloc[0]["validation_rows_used_for_threshold_fit"]) != 0:
        raise ValueError("validation rows must not be used to fit the threshold")
    if set(online["model_name"].astype(str)) != {model_name}:
        raise ValueError("online evaluation input uses a different Decision model")
    if set(online["threshold_mode"].astype(str)) != {PREDEFINED_THRESHOLD_MODE}:
        raise ValueError("online evaluation input uses a non-formal threshold mode")
    if not np.allclose(online["threshold"].to_numpy(float), threshold, rtol=0.0, atol=0.0):
        raise ValueError("online evaluation input threshold differs from the formal threshold")

    validation = decisions[decisions["dataset_role"].astype(str).eq("validation")].copy()
    missing_features = sorted(set(feature_columns).difference(validation.columns))
    if missing_features:
        raise ValueError(f"Decision dataset is missing behavior features: {missing_features}")
    endpoint_columns = [
        "p_query",
        "query_path_observed_first_hit_FE",
        "query_path_target_hit_observed",
        "query_path_endpoint_success",
    ]
    missing_endpoints = sorted(set(endpoint_columns).difference(validation.columns))
    if missing_endpoints:
        raise ValueError(
            f"Decision dataset is missing query endpoint fields: {missing_endpoints}"
        )
    feature_view = validation[
        [*STATE_COLUMNS, *feature_columns, *endpoint_columns]
    ].copy()
    if feature_view.duplicated(list(STATE_COLUMNS)).any():
        raise ValueError("Decision dataset contains duplicate validation states")
    prepared = online.merge(
        feature_view,
        on=list(STATE_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    if prepared[feature_columns].isna().all(axis=1).any():
        raise ValueError("online evaluation states are missing Decision behavior features")
    if len(prepared) != len(validation):
        raise ValueError("online input and validation Decision dataset do not have equal coverage")
    decision_model = joblib.load(model_path)
    recomputed_scores = np.asarray(
        decision_scores(decision_model, prepared[feature_columns]), dtype=float
    )
    if not np.allclose(
        recomputed_scores,
        prepared["decision_score"].to_numpy(float),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("recomputed Decision scores differ from online input")
    prepared["decision_score"] = recomputed_scores
    expected_calls = np.zeros(len(prepared), dtype=bool)
    run_key = list(RUN_COLUMNS)
    ordered = prepared.sort_values([*run_key, "FE"], kind="mergesort")
    for _, group in ordered.groupby(run_key, sort=False, dropna=False):
        candidates = group[group["decision_score"].to_numpy(float) > threshold]
        if not candidates.empty:
            expected_calls[prepared.index.get_indexer([candidates.index[0]])[0]] = True
    if not np.array_equal(expected_calls, prepared["decision_run_query"].to_numpy(bool)):
        raise ValueError("online input does not implement strict trajectory first-trigger")
    counts = prepared.groupby(run_key, dropna=False)["decision_run_query"].sum()
    if int(counts.max()) > 1:
        raise ValueError("a validation trajectory requests more than one query")

    selector = load_selector_model(selector_model_path)
    if (
        selector.selector_type != "dimension_aware_hybrid_selector"
        or selector.selector_input_mode != "query_full"
        or selector.action_budget_mode != QUERY_ADJUSTED_BUDGET
        or tuple(selector.pairwise_route_dimensions) != (40,)
        or selector.query_id != MAIN_QUERY_ID
    ):
        raise ValueError("the supplied Selector is not the formal query-full hybrid Selector")
    protocol = {
        "model_name": model_name,
        "model_path": str(model_path),
        "threshold_mode": PREDEFINED_THRESHOLD_MODE,
        "threshold": threshold,
        "feature_columns": feature_columns,
        "selector_type": str(selector.selector_type),
        "selector_input_mode": str(selector.selector_input_mode),
        "selector_target_transform": str(selector.selector_target_transform),
        "default_algorithm": str(selector.default_algorithm),
    }
    prepared.attrs["protocol"] = protocol
    return prepared, protocol


def _make_jobs(
    *,
    prepared: pd.DataFrame,
    configs: dict[str, dict[str, Any]],
    decision_model_path: Path,
    feature_columns: list[str],
    threshold: float,
    selector_model_path: Path,
    shard_dir: Path,
    overwrite: bool,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    grouped = prepared.groupby(["split", "function_id", "dimension"], sort=True)
    for job_index, ((storage_split, _, dimension), frame) in enumerate(grouped):
        config = configs[str(storage_split)]
        function, _, _ = _parse_problem_id(str(frame.iloc[0]["problem_id"]))
        shards = make_shards(config, [function], [int(dimension)])
        if len(shards) != 1:
            raise ValueError("online validation job must resolve one trajectory shard")
        output_path = shard_dir / f"job_{job_index:04d}.parquet"
        resource_path = shard_dir / f"job_{job_index:04d}__deployment.parquet"
        if overwrite:
            output_path.unlink(missing_ok=True)
            resource_path.unlink(missing_ok=True)
        jobs.append(
            {
                "records": frame.to_dict(orient="records"),
                "config": config,
                "trajectory_path": str(shards[0].output_path),
                "final_performance_path": str(shards[0].final_performance_path),
                "decision_model_path": str(decision_model_path),
                "feature_columns": feature_columns,
                "threshold": float(threshold),
                "selector_model_path": str(selector_model_path),
                "output_path": str(output_path),
                "resource_path": str(resource_path),
            }
        )
    return jobs


def _execute_jobs(jobs: list[dict[str, Any]], workers: int) -> None:
    pending = [
        job
        for job in jobs
        if not Path(job["output_path"]).exists()
        or not Path(job["resource_path"]).exists()
    ]
    if workers == 1:
        for job in pending:
            result = _run_job(job)
            print(f"wrote {result['runs']} online validation runs to {result['output']}")
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_job, job) for job in pending]
        for future in as_completed(futures):
            result = future.result()
            print(f"wrote {result['runs']} online validation runs to {result['output']}")


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    records = pd.DataFrame(job["records"])
    config = dict(job["config"])
    trajectories = pq.read_table(job["trajectory_path"]).to_pandas()
    final_performance = pq.read_table(job["final_performance_path"]).to_pandas()
    decision_model = joblib.load(job["decision_model_path"])
    selector = load_selector_model(Path(job["selector_model_path"]))
    outcomes: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    with threadpool_limits(limits=1):
        for _, run_states in records.groupby(list(RUN_COLUMNS), sort=True, dropna=False):
            outcome, resource = _run_one_policy(
                run_states=run_states.sort_values("FE", kind="mergesort"),
                trajectories=trajectories,
                final_performance=final_performance,
                config=config,
                decision_model=decision_model,
                feature_columns=[str(value) for value in job["feature_columns"]],
                threshold=float(job["threshold"]),
                selector=selector,
            )
            outcomes.append(outcome)
            resources.append(resource)
    output_path = Path(job["output_path"])
    resource_path = Path(job["resource_path"])
    pq.write_table(pa.Table.from_pylist(outcomes), output_path)
    pq.write_table(pa.Table.from_pylist(resources), resource_path)
    return {"runs": len(outcomes), "output": str(output_path)}


def _run_one_policy(
    *,
    run_states: pd.DataFrame,
    trajectories: pd.DataFrame,
    final_performance: pd.DataFrame,
    config: dict[str, Any],
    decision_model,
    feature_columns: list[str],
    threshold: float,
    selector,
) -> tuple[dict[str, Any], dict[str, Any]]:
    first = run_states.iloc[0]
    problem_id = str(first["problem_id"])
    function, instance, dimension = _parse_problem_id(problem_id)
    prefix_algorithm = str(first["prefix_algorithm"])
    seed = int(first["seed"])
    fe_total = fe_total_for_dimension(config, dimension)
    final_rows = final_performance[
        final_performance["problem_id"].astype(str).eq(problem_id)
        & final_performance["algorithm"].astype(str).eq(prefix_algorithm)
        & final_performance["seed"].astype(int).eq(seed)
    ]
    if len(final_rows) != 1:
        raise ValueError("each online run requires one native final-performance row")
    final_row = final_rows.iloc[0]
    native_first_hit = (
        None if pd.isna(final_row["observed_first_hit_FE"]) else int(final_row["observed_first_hit_FE"])
    )
    problem = make_problem(
        runtime_problem_config(
            config, function=function, instance=instance, dimension=dimension
        )
    )
    started = perf_counter()
    controller_started = perf_counter()
    trigger = None
    decision_check_count = 0
    for _, opportunity in run_states.iterrows():
        frame = pd.DataFrame(
            [{column: opportunity[column] for column in feature_columns}]
        )
        score = float(decision_scores(decision_model, frame)[0])
        if not np.isclose(
            score, float(opportunity["decision_score"]), rtol=1e-12, atol=1e-12
        ):
            raise ValueError("online sequential Decision score differs from online input")
        decision_check_count += 1
        if score > threshold:
            trigger = opportunity
            break
    controller_seconds = perf_counter() - controller_started
    expected_trigger_rows = run_states[
        run_states["decision_run_query"].to_numpy(bool)
    ]
    if len(expected_trigger_rows) > 1:
        raise ValueError("trajectory first-trigger produced more than one query")
    expected_trigger_fe = (
        None if expected_trigger_rows.empty else int(expected_trigger_rows.iloc[0]["FE"])
    )
    actual_trigger_fe = None if trigger is None else int(trigger["FE"])
    if actual_trigger_fe != expected_trigger_fe:
        raise ValueError("online sequential Decision trigger differs from online input")
    prefix_started = perf_counter()
    settings = OptimizerSettings(
        population_size=int(config["population_size"]), checkpoint_ratios=(1.0,)
    )
    state = initialize_optimizer_state(
        algorithm=prefix_algorithm, problem=problem, seed=seed, settings=settings
    )
    prefix_seconds = perf_counter() - prefix_started
    query_seconds = 0.0
    selector_seconds = 0.0
    handoff_seconds = 0.0
    continuation_seconds = 0.0
    query_called = trigger is not None
    query_feature_status = "not_called"
    query_failure = ""
    query_effective_fe = 0
    query_best_gap = float(config["failure_loss_cap"])
    query_first_hit: int | None = None
    query_completed = True
    query_timed_out = False
    selected_algorithm = str(selector.default_algorithm)
    selector_status = "not_used"
    selector_remaining_budget_ratio: float | None = None
    trigger_fe: int | None = None
    trigger_score: float | None = None
    trigger_check_count = int(decision_check_count)
    native_collection_terminal_gap: float | None = None
    native_replay_gap_difference: float | None = None
    native_replay_log10_gap_difference: float | None = None
    native_endpoint_gap_close: bool | None = None
    native_endpoint_target_hit_match: bool | None = None
    deadline = started + float(config["policy_timeout_seconds"])
    try:
        if trigger is None:
            checkpoint_fe = int(state.evaluations)
            prefix_first_hit = (
                native_first_hit
                if native_first_hit is not None and native_first_hit <= checkpoint_fe
                else None
            )
            continuation_started = perf_counter()
            outcome = _evaluate_native_skip(
                checkpoint_state=state,
                problem=problem,
                fe_budget=fe_total - checkpoint_fe,
                failure_loss_cap=float(config["failure_loss_cap"]),
                success_gap_target=float(config["success_gap_target"]),
                checkpoint_fe=checkpoint_fe,
                prefix_first_hit_fe=prefix_first_hit,
                action_timeout_seconds=max(deadline - perf_counter(), np.finfo(float).eps),
            )
            continuation_seconds = perf_counter() - continuation_started
            action_gap = float(outcome["p_skip"])
            action_effective_fe = int(outcome["skip_effective_FE"])
            continuation_first_hit = outcome["skip_continuation_first_hit_FE"]
            action_completed = bool(outcome["skip_path_completed"])
            action_timed_out = bool(outcome["skip_timed_out"])
            action_failure = str(outcome["skip_failure_message"])
            selected_algorithm = prefix_algorithm
            selector_status = "not_called"
            terminal_gap = action_gap
        else:
            trigger_fe = int(trigger["FE"])
            trigger_score = float(trigger["decision_score"])
            trajectory_row = trajectories[
                trajectories["problem_id"].astype(str).eq(problem_id)
                & trajectories["algorithm"].astype(str).eq(prefix_algorithm)
                & trajectories["seed"].astype(int).eq(seed)
                & trajectories["FE"].astype(int).eq(trigger_fe)
            ]
            if len(trajectory_row) != 1:
                raise ValueError("trigger state is absent or duplicated in the trajectory shard")
            prefix_started = perf_counter()
            advance_optimizer_state(
                state=state, problem=problem, fe_budget=trigger_fe - int(state.evaluations)
            )
            prefix_seconds += perf_counter() - prefix_started
            _validate_replayed_checkpoint(state, trajectory_row.iloc[0].to_dict())
            prefix_first_hit = (
                native_first_hit
                if native_first_hit is not None and native_first_hit <= trigger_fe
                else None
            )
            spec = get_query_spec(str(trigger["query_id"]))
            if spec.query_id != MAIN_QUERY_ID:
                raise ValueError("validation online evaluation supports the formal main query only")
            query_started = perf_counter()
            query_tracker = _TimedObjective(
                problem=problem,
                deadline=deadline,
                reference_value=float(problem.reference_value),
                success_gap_target=float(config["success_gap_target"]),
                first_evaluation_fe=trigger_fe + 1,
            )
            sample = sample_problem(
                problem=query_tracker.wrapped_problem(),
                sample_design=spec.sample_design,
                base_seed=0,
                function=function,
                instance=instance,
                success_gap_target=float(config["success_gap_target"]),
                failure_loss_cap=float(config["failure_loss_cap"]),
            )
            query_effective_fe = int(sample["sample_effective_FE"])
            query_best_gap = float(sample["query_best_gap"])
            query_completed = bool(sample["sample_path_completed"])
            query_timed_out = bool(query_tracker.timed_out)
            query_failure = str(sample["sample_failure"])
            query_first_hit = (
                None
                if sample["query_first_hit_offset"] is None
                else trigger_fe + int(sample["query_first_hit_offset"])
            )
            query_features: dict[str, Any]
            if query_completed:
                try:
                    raw_features = calculate_descriptor_cheap(
                        np.asarray(sample["X"], dtype=float),
                        np.asarray(sample["y"], dtype=float),
                        np.asarray(sample["lower_bounds"], dtype=float),
                        np.asarray(sample["upper_bounds"], dtype=float),
                    )
                    query_features = {
                        column: float(raw_features[column])
                        for column in selector.query_feature_columns
                    }
                    query_feature_status = "ok"
                except Exception as exc:
                    query_features = {}
                    query_feature_status = "failed"
                    query_failure = f"{type(exc).__name__}: {exc}"[:500]
            else:
                query_features = {}
                query_feature_status = "failed"
            query_seconds = perf_counter() - query_started
            fe_query_planned = int(spec.sample_design.sample_size(dimension))
            remaining_budget = fe_total - trigger_fe - fe_query_planned
            selector_remaining_budget_ratio = float(remaining_budget / fe_total)
            if query_feature_status == "ok":
                selector_features = make_selector_features(
                    behavior_features={
                        column: trigger[column]
                        for column in selector.feature_columns
                        if column.startswith("bf_")
                    },
                    query_features=query_features,
                    query_feature_columns=selector.query_feature_columns,
                    remaining_budget_ratio=selector_remaining_budget_ratio,
                )
                selector_started = perf_counter()
                selected_algorithm, _, measured_selector_seconds = selector.select_one(
                    selector_features, dimension=dimension
                )
                selector_seconds = max(
                    float(measured_selector_seconds), perf_counter() - selector_started
                )
                selector_status = str(selector.selector_type)
                if selected_algorithm != str(trigger["selected_algorithm"]):
                    raise ValueError(
                        "online query-full Selector prediction differs from online input"
                    )
            else:
                selected_algorithm = str(selector.default_algorithm)
                selector_status = "query_failure_fallback_default"
            continuation_started = perf_counter()
            outcome = _evaluate_one_candidate_action(
                checkpoint_state=state,
                problem=problem,
                target_algorithm=str(selected_algorithm),
                fe_budget=remaining_budget,
                seed=seed,
                function=function,
                instance=instance,
                action_budget_mode=QUERY_ADJUSTED_BUDGET,
                failure_loss_cap=float(config["failure_loss_cap"]),
                success_gap_target=float(config["success_gap_target"]),
                action_start_fe=trigger_fe + fe_query_planned,
                prefix_first_hit_fe=prefix_first_hit,
                action_timeout_seconds=max(deadline - perf_counter(), np.finfo(float).eps),
            )
            continuation_seconds = perf_counter() - continuation_started
            handoff_seconds = float(outcome["runtime_handoff"])
            action_gap = float(outcome["action_loss"])
            action_effective_fe = int(outcome["effective_FE"])
            continuation_first_hit = outcome["continuation_first_hit_FE"]
            action_completed = bool(outcome["path_completed"])
            action_timed_out = bool(outcome["timed_out"])
            action_failure = str(outcome["failure_message"])
            terminal_gap = (
                min(action_gap, query_best_gap)
                if query_completed and action_completed
                else action_gap
            )
        checkpoint_fe = int(state.evaluations)
        observed_hits = [
            int(value)
            for value in (prefix_first_hit, query_first_hit, continuation_first_hit)
            if value is not None
        ]
        observed_first_hit = min(observed_hits) if observed_hits else None
        effective_fe = checkpoint_fe + query_effective_fe + action_effective_fe
        path_completed = bool(query_completed and action_completed and effective_fe == fe_total)
        timed_out = bool(query_timed_out or action_timed_out)
        endpoint_status = "completed" if path_completed else ("timed_out" if timed_out else "failed")
        failure_message = "; ".join(
            value for value in (query_failure, action_failure) if value
        )[:500]
        target_hit_observed = observed_first_hit is not None
        if trigger is None and path_completed:
            native_collection_terminal_gap = float(final_row["final_gap"])
            native_replay_gap_difference = float(
                terminal_gap - native_collection_terminal_gap
            )
            replay_log = float(
                np.log10(
                    np.clip(
                        terminal_gap,
                        float(config["log10_gap_floor"]),
                        float(config["log10_gap_cap"]),
                    )
                )
            )
            native_log = float(final_row["log10_gap"])
            native_replay_log10_gap_difference = float(replay_log - native_log)
            native_endpoint_gap_close = bool(
                np.isclose(
                    terminal_gap,
                    native_collection_terminal_gap,
                    rtol=1e-10,
                    atol=1e-12,
                )
            )
            expected_hit = (
                None
                if pd.isna(final_row["observed_first_hit_FE"])
                else int(final_row["observed_first_hit_FE"])
            )
            native_endpoint_target_hit_match = bool(observed_first_hit == expected_hit)
        selected_equals_default = str(selected_algorithm) == str(selector.default_algorithm)
        selected_equals_prefix = str(selected_algorithm) == prefix_algorithm
        handoff_required = not selected_equals_prefix
        handoff_type = (
            "population_transfer_initialization"
            if handoff_required
            else "native_optimizer_state"
        )
        identity = {column: first[column] for column in RUN_COLUMNS}
        outcome_row = {
            **identity,
            "dataset_role": "validation",
            "policy_name": POLICY_NAME,
            "model_name": str(first["model_name"]),
            "threshold_mode": str(first["threshold_mode"]),
            "threshold": float(first["threshold"]),
            "decision_policy_unit": "trajectory_first_trigger",
            "decision_score_source": "training_after_parameter_fixed_model_recomputed",
            "decision_check_count": trigger_check_count,
            "query_called": bool(query_called),
            "query_call_count": int(query_called),
            "trigger_FE": trigger_fe,
            "trigger_FE_ratio": None if trigger_fe is None else float(trigger_fe / fe_total),
            "decision_score": trigger_score,
            "query_id": str(first["query_id"]),
            "query_protocol": str(first["query_protocol"]),
            "query_feature_status": query_feature_status,
            "query_failure_fallback": bool(query_called and query_feature_status != "ok"),
            "selector_type": str(selector.selector_type),
            "selector_input_mode": str(selector.selector_input_mode),
            "selector_target_transform": str(selector.selector_target_transform),
            "selector_status": selector_status,
            "selector_remaining_budget_ratio": selector_remaining_budget_ratio,
            "default_algorithm": str(selector.default_algorithm),
            "selected_algorithm": str(selected_algorithm),
            "selected_action": "continue_current" if selected_equals_prefix else str(selected_algorithm),
            "selected_equals_default": selected_equals_default,
            "selected_equals_prefix": selected_equals_prefix,
            "handoff_required": handoff_required,
            "handoff_type": handoff_type,
            "query_transition_mode": handoff_type,
            "FE_total": int(fe_total),
            "FE_prefix": int(checkpoint_fe),
            "FE_query_planned": int(
                get_query_spec(str(first["query_id"])).sample_design.sample_size(dimension)
                if query_called
                else 0
            ),
            "FE_query_effective": int(query_effective_fe),
            "FE_action_planned": int(
                fe_total
                - checkpoint_fe
                - (
                    get_query_spec(str(first["query_id"])).sample_design.sample_size(dimension)
                    if query_called
                    else 0
                )
            ),
            "FE_action_effective": int(action_effective_fe),
            "planned_FE": int(fe_total),
            "effective_FE": int(effective_fe),
            "benchmark_reference_value": float(problem.reference_value),
            "terminal_gap": float(terminal_gap),
            "terminal_log10_gap": float(
                np.log10(
                    np.clip(
                        terminal_gap,
                        float(config["log10_gap_floor"]),
                        float(config["log10_gap_cap"]),
                    )
                )
            ),
            "native_collection_terminal_gap": native_collection_terminal_gap,
            "native_replay_gap_difference": native_replay_gap_difference,
            "native_replay_log10_gap_difference": native_replay_log10_gap_difference,
            "native_endpoint_gap_close_rtol_1e_10_atol_1e_12": native_endpoint_gap_close,
            "native_endpoint_target_hit_match": native_endpoint_target_hit_match,
            "success_gap_target": float(config["success_gap_target"]),
            "observed_first_hit_FE": observed_first_hit,
            "target_hit_observed": bool(target_hit_observed),
            "target_hit_before_failure": bool(target_hit_observed and not path_completed),
            "endpoint_success": bool(target_hit_observed and path_completed),
            "endpoint_status": endpoint_status,
            "timed_out": timed_out,
            "path_completed": path_completed,
            "failure_message": failure_message,
            "scientific_endpoint_index": "function_evaluations",
        }
        wall_clock_seconds = max(perf_counter() - started, np.finfo(float).eps)
        resource_row = {
            **identity,
            "dataset_role": "validation",
            "policy_name": POLICY_NAME,
            "query_called": bool(query_called),
            "trigger_FE": trigger_fe,
            "selected_algorithm": str(selected_algorithm),
            "endpoint_status": endpoint_status,
            "wall_clock_seconds": float(wall_clock_seconds),
            "prefix_wall_clock_seconds": float(prefix_seconds),
            "controller_wall_clock_seconds": float(controller_seconds),
            "query_wall_clock_seconds": float(query_seconds),
            "selector_wall_clock_seconds": float(selector_seconds),
            "handoff_wall_clock_seconds": float(handoff_seconds),
            "continuation_wall_clock_seconds": float(continuation_seconds),
            "timing_source": TIMING_SOURCE,
            "deployment_metric_role": "resource_only",
            "excluded_from_labels": True,
            "excluded_from_scientific_utility": True,
        }
        return outcome_row, resource_row
    finally:
        problem.close()


def _collect_outputs(jobs: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = pd.concat(
        [pq.read_table(job["output_path"]).to_pandas() for job in jobs], ignore_index=True
    )
    resources = pd.concat(
        [pq.read_table(job["resource_path"]).to_pandas() for job in jobs], ignore_index=True
    )
    order = list(RUN_COLUMNS)
    return (
        outcomes.sort_values(order, kind="mergesort").reset_index(drop=True),
        resources.sort_values(order, kind="mergesort").reset_index(drop=True),
    )


def _validate_outputs(
    outcomes: pd.DataFrame,
    resources: pd.DataFrame,
    prepared: pd.DataFrame,
    protocol: dict[str, Any],
) -> None:
    expected_runs = prepared[list(RUN_COLUMNS)].drop_duplicates()
    for name, frame in (("outcome", outcomes), ("deployment", resources)):
        if len(frame) != len(expected_runs) or frame.duplicated(list(RUN_COLUMNS)).any():
            raise ValueError(f"{name} output does not contain exactly one row per trajectory")
        observed = frame[list(RUN_COLUMNS)].sort_values(list(RUN_COLUMNS)).reset_index(drop=True)
        expected = expected_runs.sort_values(list(RUN_COLUMNS)).reset_index(drop=True)
        observed["dimension"] = observed["dimension"].astype(int)
        expected["dimension"] = expected["dimension"].astype(int)
        observed["seed"] = observed["seed"].astype(int)
        expected["seed"] = expected["seed"].astype(int)
        if not observed.equals(expected):
            raise ValueError(f"{name} output does not cover all validation trajectories")
    if int(outcomes["query_call_count"].sum()) != int(prepared["decision_run_query"].sum()):
        raise ValueError("online output query count differs from first-trigger input")
    if int(outcomes["query_call_count"].max()) > 1:
        raise ValueError("online output queried a trajectory more than once")
    if set(outcomes["model_name"].astype(str)) != {protocol["model_name"]}:
        raise ValueError("online output model identity is inconsistent")
    if not np.allclose(outcomes["threshold"].to_numpy(float), protocol["threshold"], rtol=0, atol=0):
        raise ValueError("online output threshold is inconsistent")
    if not np.array_equal(
        outcomes["handoff_required"].to_numpy(bool),
        ~outcomes["selected_equals_prefix"].to_numpy(bool),
    ):
        raise ValueError("handoff_required must equal not selected_equals_prefix")
    expected_handoff = outcomes["handoff_type"].astype(str).eq(
        "population_transfer_initialization"
    ).to_numpy(bool)
    if not np.array_equal(outcomes["handoff_required"].to_numpy(bool), expected_handoff):
        raise ValueError("handoff_type is inconsistent with handoff_required")
    if not np.array_equal(
        outcomes["target_hit_observed"].to_numpy(bool),
        outcomes["observed_first_hit_FE"].notna().to_numpy(bool),
    ):
        raise ValueError("target_hit_observed is inconsistent with observed_first_hit_FE")
    expected_endpoint_success = (
        outcomes["target_hit_observed"].to_numpy(bool)
        & outcomes["path_completed"].to_numpy(bool)
    )
    if not np.array_equal(outcomes["endpoint_success"].to_numpy(bool), expected_endpoint_success):
        raise ValueError("endpoint_success is inconsistent")
    online_triggers = outcomes[outcomes["query_called"].to_numpy(bool)].copy()
    expected_triggers = prepared[
        prepared["decision_run_query"].to_numpy(bool)
    ][
        [
            *RUN_COLUMNS,
            "FE",
            "p_query",
            "query_path_observed_first_hit_FE",
            "query_path_target_hit_observed",
            "query_path_endpoint_success",
            "selected_algorithm",
            "selected_equals_default",
            "selected_equals_prefix",
            "handoff_required",
            "handoff_type",
        ]
    ].rename(columns={"FE": "trigger_FE"})
    paired_triggers = online_triggers.merge(
        expected_triggers,
        on=[*RUN_COLUMNS, "trigger_FE"],
        how="outer",
        validate="one_to_one",
        suffixes=("_online", "_expected"),
        indicator=True,
    )
    if not paired_triggers.empty and set(paired_triggers["_merge"].astype(str)) != {"both"}:
        raise ValueError("online trigger identities differ from online evaluation input")
    if not np.allclose(
        paired_triggers["terminal_gap"].to_numpy(float),
        paired_triggers["p_query"].to_numpy(float),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("online triggered terminal gaps differ from offline FE-indexed endpoints")
    online_hits = paired_triggers["observed_first_hit_FE"].fillna(-1).to_numpy(int)
    expected_hits = paired_triggers["query_path_observed_first_hit_FE"].fillna(-1).to_numpy(int)
    if not np.array_equal(online_hits, expected_hits):
        raise ValueError("online triggered first-hit FE differs from offline endpoint")
    if not np.array_equal(
        paired_triggers["target_hit_observed"].to_numpy(bool),
        paired_triggers["query_path_target_hit_observed"].to_numpy(bool),
    ) or not np.array_equal(
        paired_triggers["endpoint_success"].to_numpy(bool),
        paired_triggers["query_path_endpoint_success"].to_numpy(bool),
    ):
        raise ValueError("online triggered target-hit status differs from offline endpoint")
    for column in (
        "selected_algorithm",
        "selected_equals_default",
        "selected_equals_prefix",
        "handoff_required",
        "handoff_type",
    ):
        if not np.array_equal(
            paired_triggers[f"{column}_online"].to_numpy(),
            paired_triggers[f"{column}_expected"].to_numpy(),
        ):
            raise ValueError(f"online trigger relation differs for {column}")
    fe_sum = (
        outcomes["FE_prefix"].to_numpy(int)
        + outcomes["FE_query_effective"].to_numpy(int)
        + outcomes["FE_action_effective"].to_numpy(int)
    )
    if not np.array_equal(fe_sum, outcomes["effective_FE"].to_numpy(int)):
        raise ValueError("effective FE accounting is inconsistent")
    if any(
        "runtime" in column.lower() or "wall_clock" in column.lower()
        for column in outcomes.columns
    ):
        raise ValueError("scientific outcome table must not contain wall-clock columns")
    if not resources["excluded_from_labels"].all() or not resources[
        "excluded_from_scientific_utility"
    ].all():
        raise ValueError("deployment wall-clock must be excluded from labels and scientific Utility")


def _scientific_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    return (
        outcomes.groupby(["split", "dimension", "endpoint_status"], as_index=False)
        .agg(
            attempted_runs=("problem_id", "size"),
            completed_runs=("path_completed", "sum"),
            query_calls=("query_called", "sum"),
            target_hits=("target_hit_observed", "sum"),
            endpoint_successes=("endpoint_success", "sum"),
            median_terminal_log10_gap=("terminal_log10_gap", "median"),
        )
        .sort_values(["split", "dimension", "endpoint_status"])
        .reset_index(drop=True)
    )


def _resource_summary(resources: pd.DataFrame) -> pd.DataFrame:
    return (
        resources.groupby(["split", "dimension", "query_called", "endpoint_status"], as_index=False)
        .agg(
            runs=("problem_id", "size"),
            median_wall_clock_seconds=("wall_clock_seconds", "median"),
            mean_wall_clock_seconds=("wall_clock_seconds", "mean"),
        )
        .sort_values(["split", "dimension", "query_called", "endpoint_status"])
        .reset_index(drop=True)
    )


def _parse_problem_id(problem_id: str) -> tuple[int, int, int]:
    patterns = (
        r"^bbob_f(\d+)_i(\d+)_d(\d+)$",
        r"^mabbob_c(\d+)_i(\d+)_d(\d+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, problem_id)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
    raise ValueError(f"unsupported validation problem_id: {problem_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run validation online evaluation with the selected nested-OOF Decision model, "
            "formal first-trigger threshold, and query-full hybrid Selector."
        )
    )
    parser.add_argument(
        "--online-input",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "online_evaluation_input.parquet",
    )
    parser.add_argument(
        "--decision-dataset",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "decision_dataset.parquet",
    )
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "full_decision_model_training_summary.json",
    )
    parser.add_argument(
        "--selector-model",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "models/selector__query_full.joblib",
    )
    parser.add_argument(
        "--validation-config", type=Path, default=Path("configs/phase1_validation.yaml")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "validation_online_evaluation",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    result = run_validation_online_evaluation(
        online_input_path=args.online_input,
        decision_dataset_path=args.decision_dataset,
        training_summary_path=args.training_summary,
        selector_model_path=args.selector_model,
        validation_config_path=args.validation_config,
        output_dir=args.output_dir,
        workers=args.workers,
        overwrite=args.overwrite,
        summarize_only=args.summarize_only,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
