from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from decision.model_protocol import ACTIVE_MODEL_NAMES, FROZEN_THRESHOLD_MODE, SELECTED_MODEL_ALIAS
from decision.matched_random import (
    MatchedRandomCalibration,
    make_matched_random_calibration,
    matched_random_target,
)
from decision.query_contract import decision_query_root, validate_query_frame
from decision.sampling_opportunities import (
    STATE_KEY_COLUMNS,
    assert_aligned_decision_opportunities,
    assert_unique_state_keys,
    with_sampling_opportunity_type,
)
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS, get_query_spec
from trajectory.sampling import SAMPLING_METADATA_COLUMNS


DEFAULT_MODEL_NAME = SELECTED_MODEL_ALIAS
DEFAULT_THRESHOLD_MODE = FROZEN_THRESHOLD_MODE
DEFAULT_EXPECTED_SPLIT = "bbob_validation"
TARGET_COLUMN = "u_query_joint_lamT_1"
BEHAVIOR_TARGET_COLUMN = "u_behavior_only_full_budget_lamT_1"
BEHAVIOR_THRESHOLD_MODE = "oof_behavior_utility_first_trigger"
RUN_KEY_COLUMNS = STATE_KEY_COLUMNS[:-1]
ALL_ACCEPTED_OPPORTUNITIES = "all_accepted"
MILESTONE_ONLY_OPPORTUNITIES = "milestone_only"
GROUP_LAYERS = {
    "overall": [],
    "selected_equals_default": ["selected_equals_default"],
    "selected_equals_prefix": ["selected_equals_prefix"],
    "handoff_required": ["handoff_required"],
    "sampling_phase": ["sampling_phase"],
    "sampling_opportunity_type": ["sampling_opportunity_type"],
    "dimension": ["dimension"],
    "prefix_algorithm": ["prefix_algorithm"],
    "family": ["family"],
}


def compare_controller_baselines(
    *,
    query_id: str,
    predictions_path: Path,
    time_only_predictions_path: Path,
    behavior_only_predictions_path: Path,
    output_dir: Path,
    model_name: str,
    threshold_mode: str,
    random_repetitions: int,
    random_seed: int,
    expected_split: str,
    overwrite: bool,
) -> dict[str, Any]:
    _check_output_paths(output_dir, overwrite)
    predictions = _read_predictions(
        predictions_path, model_name, threshold_mode, expected_split, query_id
    )
    resolved_model_names = predictions["model_name"].astype(str).unique().tolist()
    if len(resolved_model_names) != 1:
        raise ValueError("current controller predictions must resolve to exactly one model")
    model_name = resolved_model_names[0]
    matched_call_rate, matched_trigger_fe_ratios = _train_oof_trigger_distribution(
        predictions_path=predictions_path,
        model_name=model_name,
        threshold_mode=threshold_mode,
        query_id=query_id,
    )
    training_identity = _training_identity(predictions_path)
    if str(training_identity["selected_model_name"]) != model_name:
        raise ValueError("training summary selected model does not match validation predictions")
    if str(training_identity["feature_group"]) != "B3":
        raise ValueError("the current controller baseline comparison requires B3 predictions")
    if str(training_identity["opportunity_scope"]) != ALL_ACCEPTED_OPPORTUNITIES:
        raise ValueError("the current controller baseline comparison requires all-accepted B3 predictions")
    calibration = make_matched_random_calibration(
        query_id=query_id,
        query_protocol=get_query_spec(query_id).protocol,
        feature_group=str(training_identity["feature_group"]),
        selected_model=model_name,
        threshold_mode=threshold_mode,
        run_call_rate=matched_call_rate,
        trigger_fe_ratios=matched_trigger_fe_ratios,
        seed=random_seed,
    )
    validate_time_only_training_summary(time_only_predictions_path, query_id)
    time_only_predictions = _read_predictions(
        time_only_predictions_path,
        model_name,
        threshold_mode,
        expected_split,
        query_id,
    )
    validate_behavior_only_training_summary(
        behavior_only_predictions_path,
        query_id=query_id,
        model_name=model_name,
    )
    behavior_only_predictions = _read_predictions(
        behavior_only_predictions_path,
        model_name,
        BEHAVIOR_THRESHOLD_MODE,
        expected_split,
        query_id,
        target_column=BEHAVIOR_TARGET_COLUMN,
    )
    _check_prediction_alignment(predictions, time_only_predictions)
    _check_behavior_prediction_alignment(predictions, behavior_only_predictions)
    policies = _policy_frames(
        predictions=predictions,
        time_only_predictions=time_only_predictions,
        behavior_only_predictions=behavior_only_predictions,
        threshold_mode=threshold_mode,
        calibration=calibration,
        random_repetitions=random_repetitions,
    )
    policy_summary = _policy_summary(policies)
    relative_summary = _relative_summary(policy_summary)
    best_policy_summary = _best_policy_summary(policy_summary)
    random_repetition_summary = _matched_random_repetition_summary(
        predictions=predictions,
        calibration=calibration,
        random_repetitions=random_repetitions,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(policy_summary, output_dir / "controller_baseline_policy_summary")
    _write_frame(relative_summary, output_dir / "controller_baseline_relative_summary")
    _write_frame(best_policy_summary, output_dir / "controller_baseline_best_policy")
    _write_frame(random_repetition_summary, output_dir / "controller_baseline_random_repetition_summary")
    calibration_path = calibration.write(output_dir / "matched_random_calibration.json")

    summary = {
        "experiment": "phase1_refined_sampling_controller_baseline_comparison",
        "query_id": query_id,
        "query_protocol": get_query_spec(query_id).protocol,
        "sample_design_id": get_query_spec(query_id).sample_design_id,
        "research_question": (
            "How does the dynamic Decision-before-Feature policy compare as a whole policy with SBS/Never Query, "
            "Always Query, matched-rate Random, milestone-only T0, and self-thresholded Behavior-only "
            f"under frozen {TARGET_COLUMN} labels?"
        ),
        "predictions_path": str(predictions_path),
        "time_only_predictions_path": str(time_only_predictions_path),
        "behavior_only_predictions_path": str(behavior_only_predictions_path),
        "model_name": model_name,
        "time_only_model_name": model_name,
        "time_only_input": {
            "mathematical_input": ["FE_ratio"],
            "implementation_input": ["bf_fe_ratio"],
        },
        "threshold_mode": threshold_mode,
        "expected_split": expected_split,
        "matched_random_run_call_rate": matched_call_rate,
        "matched_random_trigger_distribution_source": "selected_model_bbob_train_oof_first_trigger",
        "random_repetitions": random_repetitions,
        "matched_random_aggregation": "average_repetitions_within_trajectory_before_policy_summary",
        "random_seed": random_seed,
        "rows": int(len(predictions)),
        "reference_state_table_opportunity_scope": ALL_ACCEPTED_OPPORTUNITIES,
        "policy_eligible_opportunity_scopes": {
            "sbs_skip_reference": "not_applicable_no_trigger",
            "never_query": "not_applicable_no_trigger",
            "always_query": ALL_ACCEPTED_OPPORTUNITIES,
            "matched_rate_random": ALL_ACCEPTED_OPPORTUNITIES,
            "milestone_only_T0": MILESTONE_ONLY_OPPORTUNITIES,
            "self_thresholded_behavior_only": ALL_ACCEPTED_OPPORTUNITIES,
            "matched_trigger_behavior_only": ALL_ACCEPTED_OPPORTUNITIES,
            "current_controller": ALL_ACCEPTED_OPPORTUNITIES,
        },
        "policy_unit": "trajectory_first_trigger",
        "comparison_role": "whole_policy_baseline_comparison",
        "primary_rq2_contrast_included": False,
        "policies": sorted(policies["policy_name"].unique().tolist()),
        "outputs": {
            "policy_summary": str(output_dir / "controller_baseline_policy_summary.parquet"),
            "relative_summary": str(output_dir / "controller_baseline_relative_summary.parquet"),
            "best_policy": str(output_dir / "controller_baseline_best_policy.parquet"),
            "random_repetition_summary": str(output_dir / "controller_baseline_random_repetition_summary.parquet"),
            "matched_random_calibration": str(calibration_path),
            "report": str(output_dir / "controller_baseline_comparison_report.md"),
            "summary": str(output_dir / "controller_baseline_comparison_summary.json"),
        },
        "data_leakage_check": {
            "models_retrained": False,
            "utility_labels_regenerated": False,
            "expected_split_rows_used_for_controller_fit_or_threshold": False,
            "query_features_used_as_decision_input": False,
            "function_id_algorithm_id_or_optimizer_internal_parameters_used_as_input": False,
            "time_only_milestone_rows_are_subset_of_dynamic_controller_rows": True,
            "time_only_and_dynamic_controller_opportunity_sets_identical": False,
            "all_policies_use_identical_decision_opportunities": False,
            "matched_random_rate_or_timing_uses_validation_rows": False,
        },
        "scope_notes": [
            f"The whole-policy baseline comparison is expressed in the frozen {TARGET_COLUMN} label space.",
            "Every online policy contributes at most one first-trigger call and one Utility value per trajectory.",
            "The current B3 controller, Always Query, matched-rate Random, and Behavior-only policies may trigger "
            "on all accepted dynamic opportunities; milestone-only T0 may trigger only on the twelve budget milestones.",
            "Therefore current_controller versus milestone_only_T0 is a whole-policy comparison that combines "
            "opportunity scheduling, feature inputs, fitted scores, and thresholds. It is not the RQ2 Behavior increment.",
            "RQ2 is materialized separately by decision-compare-feature-groups using milestone-only B3 and "
            "milestone-only T0 on exactly aligned state rows.",
            "Matched Random repetitions are averaged within each trajectory before policy aggregation and are not independent replicates.",
            "sbs_skip_reference and never_query have identical zero utility because both retain the train-derived SBS "
            "skip path without calling the fixed query.",
            "Matched Random is reported only overall after within-trajectory averaging; assigning an averaged "
            "trajectory to one trigger-state stratum would be undefined.",
            "A distinct optimizer-level SBS performance comparison requires materializing P_sbs or final-performance "
            "columns; those fields are not present in the current controller prediction table.",
            "The static per-problem full-budget four-algorithm VBS hindsight reference is not derivable from "
            "Decision prediction rows and is not produced by this comparison.",
        ],
    }
    summary_path = output_dir / "controller_baseline_comparison_summary.json"
    report_path = output_dir / "controller_baseline_comparison_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(
        _markdown_report(
            policy_summary=policy_summary,
            relative_summary=relative_summary,
            best_policy_summary=best_policy_summary,
            random_repetition_summary=random_repetition_summary,
            model_name=model_name,
            threshold_mode=threshold_mode,
            matched_call_rate=matched_call_rate,
            random_repetitions=random_repetitions,
            expected_split=expected_split,
        ),
        encoding="utf-8",
    )
    print(f"wrote controller baseline policy summary to {output_dir / 'controller_baseline_policy_summary.parquet'}")
    print(f"wrote controller baseline relative summary to {output_dir / 'controller_baseline_relative_summary.parquet'}")
    print(f"wrote controller baseline report to {report_path}")
    return summary


def _check_output_paths(output_dir: Path, overwrite: bool) -> None:
    outputs = (
        output_dir / "controller_baseline_policy_summary.csv",
        output_dir / "controller_baseline_policy_summary.parquet",
        output_dir / "controller_baseline_relative_summary.csv",
        output_dir / "controller_baseline_relative_summary.parquet",
        output_dir / "controller_baseline_best_policy.csv",
        output_dir / "controller_baseline_best_policy.parquet",
        output_dir / "controller_baseline_random_repetition_summary.csv",
        output_dir / "controller_baseline_random_repetition_summary.parquet",
        output_dir / "matched_random_calibration.json",
        output_dir / "controller_baseline_comparison_report.md",
        output_dir / "controller_baseline_comparison_summary.json",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"controller baseline comparison outputs already exist; pass --overwrite: {existing[0]}")


def _read_predictions(
    path: Path,
    model_name: str,
    threshold_mode: str,
    expected_split: str,
    query_id: str,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pq.read_table(path).to_pandas()
    validate_query_frame(frame, query_id=query_id, artifact="controller prediction table")
    required = {
        "model_name",
        "split",
        "problem_id",
        "function_id",
        "family",
        "dimension",
        "prefix_algorithm",
        "seed",
        "FE",
        "FE_ratio",
        *SAMPLING_METADATA_COLUMNS,
        "default_algorithm",
        "no_query_algorithm",
        "selected_algorithm",
        "selected_action",
        "selected_equals_default",
        "selected_equals_prefix",
        "handoff_required",
        "handoff_type",
        "selector_target_transform",
        target_column,
        f"decision_run_query_{threshold_mode}",
        f"decision_utility_{threshold_mode}",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"prediction table is missing required columns: {missing}")
    if model_name == SELECTED_MODEL_ALIAS:
        if "selected_by_nested_oof" not in frame.columns:
            raise ValueError("prediction table does not identify the nested-OOF selected model")
        selected = frame[frame["selected_by_nested_oof"].astype(bool)]
        selected_names = selected["model_name"].astype(str).unique().tolist()
        if len(selected_names) != 1:
            raise ValueError("prediction table must identify exactly one nested-OOF selected model")
        model_name = selected_names[0]
    frame = frame[frame["model_name"] == model_name].copy()
    if frame.empty:
        raise ValueError(f"no prediction rows for model_name={model_name}")
    if frame["split"].nunique() != 1 or str(frame["split"].iloc[0]) != expected_split:
        raise ValueError(f"controller baseline comparison expects {expected_split} prediction rows")
    if not np.isfinite(frame[target_column].to_numpy(dtype=float)).all():
        raise ValueError(f"{target_column} contains non-finite values")
    frame = with_sampling_opportunity_type(frame, artifact="controller prediction table")
    assert_unique_state_keys(frame, artifact="controller prediction table")
    return frame.sort_values(list(STATE_KEY_COLUMNS)).reset_index(drop=True)


def _train_oof_trigger_distribution(
    *,
    predictions_path: Path,
    model_name: str,
    threshold_mode: str,
    query_id: str,
) -> tuple[float, np.ndarray]:
    train_oof_path = predictions_path.parent / "train_oof_predictions.parquet"
    train_oof = _read_predictions(
        train_oof_path,
        model_name,
        threshold_mode,
        "bbob_train",
        query_id,
    )
    call_column = f"decision_run_query_{threshold_mode}"
    calls = _first_trigger_mask(train_oof, train_oof[call_column].to_numpy(dtype=bool))
    called_runs = 0
    total_runs = 0
    trigger_fe_ratios: list[float] = []
    for _, run_frame in train_oof.groupby(list(RUN_KEY_COLUMNS), sort=True, dropna=False):
        ordered = _ordered_policy_run(run_frame)
        positions = train_oof.index.get_indexer(ordered.index)
        run_calls = calls[positions]
        total_runs += 1
        hit = np.flatnonzero(run_calls)
        if hit.size == 0:
            continue
        called_runs += 1
        trigger_fe_ratios.append(float(ordered.iloc[int(hit[0])]["FE_ratio"]))
    if total_runs == 0:
        raise ValueError("train OOF matched-random calibration requires at least one trajectory")
    return float(called_runs / total_runs), np.asarray(trigger_fe_ratios, dtype=float)


def _training_identity(predictions_path: Path) -> dict[str, Any]:
    summary_path = predictions_path.parent / "full_decision_model_training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = (
        "feature_group",
        "selected_model_name",
        "query_id",
        "query_protocol",
        "opportunity_scope",
    )
    missing = [field for field in required if not summary.get(field)]
    if missing:
        raise ValueError(f"Decision training summary is missing matched-random identity fields: {missing}")
    return summary


def validate_time_only_training_summary(path: Path, query_id: str) -> None:
    summary_path = path.parent / "full_decision_model_training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    spec = get_query_spec(query_id)
    if summary.get("query_id") != query_id or summary.get("query_protocol") != spec.protocol:
        raise ValueError("time-only training summary does not match the requested query protocol")
    if summary.get("feature_group") != "T0":
        raise ValueError("time-only predictions must come from the canonical feature_group=T0")
    if summary.get("opportunity_scope") != MILESTONE_ONLY_OPPORTUNITIES:
        raise ValueError("the formal time-only baseline must use opportunity_scope=milestone_only")
    if list(summary.get("feature_columns", [])) != ["bf_fe_ratio"]:
        raise ValueError("time-only training summary must use only bf_fe_ratio")
    if tuple(summary.get("models_trained", [])) != ACTIVE_MODEL_NAMES:
        raise ValueError("time-only training must use the same frozen three-model candidate set")
    if summary.get("threshold_modes") != ["zero", FROZEN_THRESHOLD_MODE]:
        raise ValueError("time-only training must use the frozen OOF threshold protocol")


def validate_behavior_only_training_summary(
    path: Path,
    *,
    query_id: str,
    model_name: str,
) -> None:
    summary = _training_identity(path)
    spec = get_query_spec(query_id)
    expected_main = {
        "query_id": query_id,
        "query_protocol": spec.protocol,
        "feature_group": "B3",
        "selected_model_name": model_name,
        "opportunity_scope": ALL_ACCEPTED_OPPORTUNITIES,
    }
    mismatch = {
        key: {"expected": value, "observed": summary.get(key)}
        for key, value in expected_main.items()
        if summary.get(key) != value
    }
    behavior_policy = summary.get("behavior_only_policy")
    if not isinstance(behavior_policy, dict):
        mismatch["behavior_only_policy"] = {
            "expected": "mapping",
            "observed": type(behavior_policy).__name__,
        }
    else:
        expected_behavior = {
            "target_column": BEHAVIOR_TARGET_COLUMN,
            "feature_group": "B3",
            "opportunity_scope": ALL_ACCEPTED_OPPORTUNITIES,
            "model_name": model_name,
        }
        mismatch.update(
            {
                f"behavior_only_policy.{key}": {
                    "expected": value,
                    "observed": behavior_policy.get(key),
                }
                for key, value in expected_behavior.items()
                if behavior_policy.get(key) != value
            }
        )
        threshold_modes = tuple(behavior_policy.get("threshold_modes", []))
        if BEHAVIOR_THRESHOLD_MODE not in threshold_modes:
            mismatch["behavior_only_policy.threshold_modes"] = {
                "expected_to_contain": BEHAVIOR_THRESHOLD_MODE,
                "observed": list(threshold_modes),
            }
    if mismatch:
        raise ValueError(f"Behavior-only training summary is inconsistent: {mismatch}")


def _check_prediction_alignment(current: pd.DataFrame, time_only: pd.DataFrame) -> None:
    assert_unique_state_keys(current, artifact="current-controller predictions")
    assert_unique_state_keys(time_only, artifact="time-only predictions")
    if not bool(time_only["is_budget_milestone"].to_numpy(dtype=bool).all()):
        raise ValueError("formal time-only predictions must contain budget milestones only")
    current_runs = set(map(tuple, current[list(RUN_KEY_COLUMNS)].itertuples(index=False, name=None)))
    time_only_runs = set(map(tuple, time_only[list(RUN_KEY_COLUMNS)].itertuples(index=False, name=None)))
    if current_runs != time_only_runs:
        raise ValueError("current and time-only predictions must cover the same trajectories")
    current_indexed = current.set_index(list(STATE_KEY_COLUMNS), drop=False)
    time_only_indexed = time_only.set_index(list(STATE_KEY_COLUMNS), drop=False)
    missing_keys = time_only_indexed.index.difference(current_indexed.index)
    if len(missing_keys):
        raise ValueError("time-only milestone states must be a subset of current-controller opportunities")
    aligned_current = current_indexed.loc[time_only_indexed.index].reset_index(drop=True)
    aligned_time_only = time_only_indexed.reset_index(drop=True)
    assert_aligned_decision_opportunities(
        aligned_current,
        aligned_time_only,
        reference_artifact="dynamic B3 milestone subset",
        candidate_artifact="milestone-only T0 predictions",
    )
    string_columns = (
        "split",
        "problem_id",
        "function_id",
        "family",
        "prefix_algorithm",
        "default_algorithm",
        "no_query_algorithm",
        "selected_algorithm",
        "selected_action",
        "handoff_type",
        "selector_target_transform",
    )
    integer_columns = ("dimension", "seed", "FE")
    boolean_columns = ("selected_equals_default", "selected_equals_prefix", "handoff_required")
    for column in string_columns:
        if not np.array_equal(
            aligned_current[column].astype(str).to_numpy(),
            aligned_time_only[column].astype(str).to_numpy(),
        ):
            raise ValueError(f"current and time-only prediction rows disagree on {column}")
    for column in integer_columns:
        if not np.array_equal(
            aligned_current[column].astype(int).to_numpy(),
            aligned_time_only[column].astype(int).to_numpy(),
        ):
            raise ValueError(f"current and time-only prediction rows disagree on {column}")
    for column in boolean_columns:
        if not np.array_equal(
            aligned_current[column].to_numpy(dtype=bool),
            aligned_time_only[column].to_numpy(dtype=bool),
        ):
            raise ValueError(f"current and time-only prediction rows disagree on {column}")
    for column in (TARGET_COLUMN,):
        if not np.allclose(
            aligned_current[column].to_numpy(dtype=float),
            aligned_time_only[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"current and time-only prediction rows disagree on {column}")


def _check_behavior_prediction_alignment(
    current: pd.DataFrame,
    behavior_only: pd.DataFrame,
) -> None:
    assert_aligned_decision_opportunities(
        current,
        behavior_only,
        reference_artifact="Query-controller predictions",
        candidate_artifact="Behavior-only predictions",
    )
    for column in (
        "split",
        "problem_id",
        "function_id",
        "family",
        "prefix_algorithm",
    ):
        if not np.array_equal(
            current[column].astype(str).to_numpy(),
            behavior_only[column].astype(str).to_numpy(),
        ):
            raise ValueError(f"Query and Behavior-only prediction rows disagree on {column}")
    for column in ("dimension", "seed", "FE"):
        if not np.array_equal(
            current[column].astype(int).to_numpy(),
            behavior_only[column].astype(int).to_numpy(),
        ):
            raise ValueError(f"Query and Behavior-only prediction rows disagree on {column}")


def _policy_frames(
    *,
    predictions: pd.DataFrame,
    time_only_predictions: pd.DataFrame,
    behavior_only_predictions: pd.DataFrame,
    threshold_mode: str,
    calibration: MatchedRandomCalibration,
    random_repetitions: int,
) -> pd.DataFrame:
    if random_repetitions <= 0:
        raise ValueError("random_repetitions must be positive")
    observed = predictions[TARGET_COLUMN].to_numpy(dtype=float)
    behavior_observed = behavior_only_predictions[BEHAVIOR_TARGET_COLUMN].to_numpy(dtype=float)
    current_call = _first_trigger_mask(
        predictions,
        predictions[f"decision_run_query_{threshold_mode}"].to_numpy(dtype=bool),
    )
    time_only_call = _project_first_trigger_mask(
        reference=predictions,
        candidate=time_only_predictions,
        candidate_triggers=time_only_predictions[
            f"decision_run_query_{threshold_mode}"
        ].to_numpy(dtype=bool),
    )
    always_call = _first_opportunity_mask(predictions)
    all_accepted_eligible = np.ones(len(predictions), dtype=bool)
    no_trigger_eligible = np.zeros(len(predictions), dtype=bool)
    milestone_eligible = predictions["is_budget_milestone"].to_numpy(dtype=bool)
    behavior_self_call = _first_trigger_mask(
        behavior_only_predictions,
        behavior_only_predictions[
            f"decision_run_query_{BEHAVIOR_THRESHOLD_MODE}"
        ].to_numpy(dtype=bool),
    )
    policy_specs = (
        (
            "sbs_skip_reference",
            "baseline",
            np.zeros(len(predictions), dtype=bool),
            "not_applicable_no_trigger",
            no_trigger_eligible,
        ),
        (
            "never_query",
            "baseline",
            np.zeros(len(predictions), dtype=bool),
            "not_applicable_no_trigger",
            no_trigger_eligible,
        ),
        ("always_query", "baseline", always_call, ALL_ACCEPTED_OPPORTUNITIES, all_accepted_eligible),
        (
            "milestone_only_T0",
            "learned_baseline",
            time_only_call,
            MILESTONE_ONLY_OPPORTUNITIES,
            milestone_eligible,
        ),
        (
            "current_controller",
            "controller",
            current_call,
            ALL_ACCEPTED_OPPORTUNITIES,
            all_accepted_eligible,
        ),
    )
    frames = []
    metadata_columns = [
            "split",
            "problem_id",
            "function_id",
            "family",
            "dimension",
            "prefix_algorithm",
            "seed",
            "FE",
            "FE_ratio",
            "sampling_phase",
            "sampling_opportunity_type",
            "default_algorithm",
            "no_query_algorithm",
            "selected_algorithm",
            "selected_action",
            "selected_equals_default",
            "selected_equals_prefix",
            "handoff_required",
            "handoff_type",
            "selector_target_transform",
            TARGET_COLUMN,
        ]
    if "decision_opportunity_index" in predictions.columns:
        metadata_columns.insert(metadata_columns.index("sampling_phase"), "decision_opportunity_index")
    metadata = predictions[metadata_columns].copy()
    for policy_name, policy_category, call, opportunity_scope, opportunity_eligible in policy_specs:
        frame = metadata.copy()
        frame.insert(0, "policy_name", policy_name)
        frame.insert(1, "policy_category", policy_category)
        frame.insert(2, "eligible_opportunity_scope", opportunity_scope)
        frame.insert(3, "eligible_opportunity", opportunity_eligible)
        frame["policy_trigger"] = call
        frame["query_called"] = call
        frame["policy_utility"] = np.where(call, observed, 0.0)
        frame["random_repetition"] = None
        frames.append(frame)
    for repetition in range(random_repetitions):
        call = _matched_random_first_trigger_mask(
            predictions,
            calibration=calibration,
            repetition=repetition,
        )
        frame = metadata.copy()
        frame.insert(0, "policy_name", "matched_rate_random")
        frame.insert(1, "policy_category", "baseline")
        frame.insert(2, "eligible_opportunity_scope", ALL_ACCEPTED_OPPORTUNITIES)
        frame.insert(3, "eligible_opportunity", all_accepted_eligible)
        frame["policy_trigger"] = call
        frame["query_called"] = call
        frame["policy_utility"] = np.where(call, observed, 0.0)
        frame["random_repetition"] = int(repetition)
        frames.append(frame)
    behavior_metadata_columns = [
            "split",
            "problem_id",
            "function_id",
            "family",
            "dimension",
            "prefix_algorithm",
            "seed",
            "FE",
            "FE_ratio",
            "sampling_phase",
            "sampling_opportunity_type",
            "default_algorithm",
            "no_query_algorithm",
            "selected_algorithm",
            "selected_action",
            "selected_equals_default",
            "selected_equals_prefix",
            "handoff_required",
            "handoff_type",
            "selector_target_transform",
            BEHAVIOR_TARGET_COLUMN,
        ]
    if "decision_opportunity_index" in behavior_only_predictions.columns:
        behavior_metadata_columns.insert(
            behavior_metadata_columns.index("sampling_phase"),
            "decision_opportunity_index",
        )
    behavior_metadata = behavior_only_predictions[behavior_metadata_columns].rename(
        columns={BEHAVIOR_TARGET_COLUMN: TARGET_COLUMN}
    )
    for policy_name, policy_category, call, opportunity_scope, opportunity_eligible in (
        (
            "matched_trigger_behavior_only",
            "diagnostic",
            current_call,
            ALL_ACCEPTED_OPPORTUNITIES,
            all_accepted_eligible,
        ),
        (
            "self_thresholded_behavior_only",
            "learned_baseline",
            behavior_self_call,
            ALL_ACCEPTED_OPPORTUNITIES,
            all_accepted_eligible,
        ),
    ):
        frame = behavior_metadata.copy()
        frame.insert(0, "policy_name", policy_name)
        frame.insert(1, "policy_category", policy_category)
        frame.insert(2, "eligible_opportunity_scope", opportunity_scope)
        frame.insert(3, "eligible_opportunity", opportunity_eligible)
        frame["policy_trigger"] = call
        frame["query_called"] = False
        frame["policy_utility"] = np.where(call, behavior_observed, 0.0)
        frame["random_repetition"] = None
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _ordered_policy_run(run_frame: pd.DataFrame) -> pd.DataFrame:
    order_columns = ["FE"]
    if "decision_opportunity_index" in run_frame.columns:
        if run_frame["decision_opportunity_index"].isna().any():
            raise ValueError("decision_opportunity_index contains missing values")
        order_columns.append("decision_opportunity_index")
    elif run_frame["FE"].duplicated().any():
        raise ValueError(
            "multiple policy opportunities share one FE without decision_opportunity_index"
        )
    return run_frame.sort_values(order_columns, kind="mergesort")


def _first_trigger_mask(frame: pd.DataFrame, candidates: np.ndarray) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=bool).reshape(-1)
    if len(candidates) != len(frame):
        raise ValueError("first-trigger candidate mask must align with the state table")
    if not frame.index.is_unique:
        raise ValueError("first-trigger state tables require unique row indices")
    mask = np.zeros(len(frame), dtype=bool)
    for _, run_frame in frame.groupby(list(RUN_KEY_COLUMNS), sort=True, dropna=False):
        ordered = _ordered_policy_run(run_frame)
        positions = frame.index.get_indexer(ordered.index)
        hit = np.flatnonzero(candidates[positions])
        if hit.size:
            mask[int(positions[int(hit[0])])] = True
    return mask


def _first_opportunity_mask(frame: pd.DataFrame) -> np.ndarray:
    return _first_trigger_mask(frame, np.ones(len(frame), dtype=bool))


def _project_first_trigger_mask(
    *,
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    candidate_triggers: np.ndarray,
) -> np.ndarray:
    candidate_first = _first_trigger_mask(candidate, candidate_triggers)
    trigger_keys = list(
        map(
            tuple,
            candidate.loc[candidate_first, list(STATE_KEY_COLUMNS)].itertuples(
                index=False, name=None
            ),
        )
    )
    reference_key_rows = list(
        map(
            tuple,
            reference[list(STATE_KEY_COLUMNS)].itertuples(index=False, name=None),
        )
    )
    reference_key_set = set(reference_key_rows)
    trigger_key_set = set(trigger_keys)
    missing = trigger_key_set.difference(reference_key_set)
    if len(missing):
        raise ValueError("projected first-trigger states are absent from the reference opportunity table")
    return np.asarray([key in trigger_key_set for key in reference_key_rows], dtype=bool)


def _matched_random_first_trigger_mask(
    frame: pd.DataFrame,
    *,
    calibration: MatchedRandomCalibration,
    repetition: int,
) -> np.ndarray:
    mask = np.zeros(len(frame), dtype=bool)
    for _, run_frame in frame.groupby(list(RUN_KEY_COLUMNS), sort=True, dropna=False):
        ordered = _ordered_policy_run(run_frame)
        first = ordered.iloc[0]
        target_ratio = matched_random_target(
            calibration,
            problem_id=str(first["problem_id"]),
            prefix_algorithm=str(first["prefix_algorithm"]),
            run_seed=int(first["seed"]),
            dimension=int(first["dimension"]),
            repetition=int(repetition),
        )
        if target_ratio is None:
            continue
        eligible = np.flatnonzero(ordered["FE_ratio"].to_numpy(dtype=float) >= target_ratio)
        if not len(eligible):
            continue
        position = int(frame.index.get_indexer([ordered.index[int(eligible[0])]])[0])
        if position < 0:
            raise RuntimeError("matched-random trigger row is not aligned with the state table")
        mask[position] = True
    return mask


def _policy_run_frame(policy_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = list(RUN_KEY_COLUMNS)
    if "random_repetition" in policy_frame.columns:
        group_columns.append("random_repetition")
    for run_key, run_frame in policy_frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(run_key, tuple):
            run_key = (run_key,)
        ordered = _ordered_policy_run(run_frame)
        opportunity_scopes = ordered["eligible_opportunity_scope"].astype(str).unique().tolist()
        if len(opportunity_scopes) != 1:
            raise ValueError("one policy trajectory must use exactly one eligible-opportunity scope")
        opportunity_eligible = ordered["eligible_opportunity"].to_numpy(dtype=bool)
        triggers = ordered["policy_trigger"].to_numpy(dtype=bool)
        query_calls = ordered["query_called"].to_numpy(dtype=bool)
        if bool(np.any(triggers & ~opportunity_eligible)):
            raise ValueError("a policy triggered outside its declared eligible-opportunity scope")
        if int(np.sum(triggers)) > 1 or int(np.sum(query_calls)) > 1:
            raise ValueError("each policy may trigger an action and call the query at most once per trajectory")
        observed = ordered[TARGET_COLUMN].to_numpy(dtype=float)
        triggered = bool(np.any(triggers))
        query_called = bool(np.any(query_calls))
        trigger = ordered.loc[triggers].iloc[0] if triggered else ordered.iloc[0]
        call_utility = float(trigger[TARGET_COLUMN]) if triggered else 0.0
        rows.append(
            {
                "policy_name": str(ordered["policy_name"].iloc[0]),
                "policy_category": str(ordered["policy_category"].iloc[0]),
                "eligible_opportunity_scope": opportunity_scopes[0],
                "eligible_opportunity_count": int(np.sum(opportunity_eligible)),
                "run_key": "|".join(str(value) for value in run_key),
                "base_run_key": "|".join(
                    str(value) for value in run_key[: len(RUN_KEY_COLUMNS)]
                ),
                "random_repetition": (
                    None
                    if len(run_key) == len(RUN_KEY_COLUMNS) or pd.isna(run_key[-1])
                    else int(run_key[-1])
                ),
                "function_id": str(ordered["function_id"].iloc[0]),
                "family": str(ordered["family"].iloc[0]),
                "dimension": int(ordered["dimension"].iloc[0]),
                "prefix_algorithm": str(ordered["prefix_algorithm"].iloc[0]),
                "sampling_phase": str(trigger["sampling_phase"]) if triggered else "not_called",
                "sampling_opportunity_type": (
                    str(trigger["sampling_opportunity_type"]) if triggered else "not_called"
                ),
                "selected_equals_default": bool(trigger["selected_equals_default"]) if triggered else None,
                "selected_equals_prefix": bool(trigger["selected_equals_prefix"]) if triggered else None,
                "handoff_required": bool(trigger["handoff_required"]) if triggered else None,
                "states": int(len(ordered)),
                "policy_trigger": triggered,
                "query_called": query_called,
                "policy_utility": call_utility,
                "trigger_weight": float(triggered),
                "query_call_weight": float(query_called),
                "beneficial_call_weight": float(triggered and call_utility > 0.0),
                "unhelpful_call_weight": float(triggered and call_utility <= 0.0),
                "captured_positive_utility": float(max(call_utility, 0.0)) if triggered else 0.0,
                "unhelpful_utility": float(min(call_utility, 0.0)) if triggered else 0.0,
                "trigger_observed_utility": call_utility,
                "best_available_positive_utility": float(max(0.0, float(np.max(observed)))),
                "has_positive_opportunity": bool(np.any(observed > 0.0)),
                "first_opportunity_utility": float(observed[0]),
            }
        )
    return pd.DataFrame(rows)


def _average_matched_random_within_run(run_frame: pd.DataFrame) -> pd.DataFrame:
    if run_frame.empty:
        return run_frame
    rows: list[dict[str, Any]] = []
    mean_columns = (
        "policy_utility",
        "trigger_observed_utility",
        "trigger_weight",
        "query_call_weight",
        "beneficial_call_weight",
        "unhelpful_call_weight",
        "captured_positive_utility",
        "unhelpful_utility",
    )
    invariant_columns = (
        "policy_name",
        "policy_category",
        "eligible_opportunity_scope",
        "eligible_opportunity_count",
        "function_id",
        "family",
        "dimension",
        "prefix_algorithm",
        "states",
        "best_available_positive_utility",
        "has_positive_opportunity",
        "first_opportunity_utility",
    )
    for base_run_key, frame in run_frame.groupby("base_run_key", sort=True, dropna=False):
        repetitions = sorted(frame["random_repetition"].astype(int).tolist())
        if repetitions != list(range(len(repetitions))):
            raise ValueError("matched Random repetitions must be consecutive within each trajectory")
        first = frame.iloc[0]
        row = {column: first[column] for column in invariant_columns}
        row.update(
            {
                "run_key": str(base_run_key),
                "base_run_key": str(base_run_key),
                "random_repetition": None,
                "sampling_phase": "averaged_random_repetitions",
                "sampling_opportunity_type": "averaged_random_repetitions",
                "selected_equals_default": None,
                "selected_equals_prefix": None,
                "handoff_required": None,
                "policy_trigger": float(frame["trigger_weight"].mean()),
                "query_called": float(frame["query_call_weight"].mean()),
                "random_repetitions_averaged_within_run": int(len(frame)),
            }
        )
        row.update({column: float(frame[column].mean()) for column in mean_columns})
        rows.append(row)
    return pd.DataFrame(rows)


def _policy_summary(policies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer, group_columns in GROUP_LAYERS.items():
        if not group_columns:
            for (_, _), state_frame in policies.groupby(["policy_name", "policy_category"], sort=True):
                policy_frame = _policy_run_frame(state_frame)
                if str(state_frame["policy_name"].iloc[0]) == "matched_rate_random":
                    policy_frame = _average_matched_random_within_run(policy_frame)
                rows.append(_policy_row(policy_frame, layer=layer, group={}))
            continue
        for (_, _), state_frame in policies.groupby(["policy_name", "policy_category"], sort=True):
            policy_frame = _policy_run_frame(state_frame)
            if str(state_frame["policy_name"].iloc[0]) == "matched_rate_random":
                # Repetitions define a within-trajectory Monte Carlo expectation.  A single
                # averaged trajectory can have probability mass in several trigger strata,
                # so assigning it to a synthetic stratum would be misleading.
                continue
            for group_values, frame in policy_frame.groupby(group_columns, dropna=False, sort=True):
                if not isinstance(group_values, tuple):
                    group_values = (group_values,)
                rows.append(_policy_row(frame, layer=layer, group=dict(zip(group_columns, group_values, strict=True))))
    result = pd.DataFrame(rows)
    sort_columns = ["layer", "group", "policy_category", "policy_name"]
    return result.sort_values(sort_columns).reset_index(drop=True)


def _single_string_value(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].astype(str).unique().tolist()
    if len(values) != 1:
        raise ValueError(f"policy summary requires one invariant {column}, observed {values}")
    return values[0]


def _policy_row(frame: pd.DataFrame, *, layer: str, group: dict[str, Any]) -> dict[str, Any]:
    utility = frame["policy_utility"].to_numpy(dtype=float)
    triggers = frame["trigger_weight"].to_numpy(dtype=float)
    query_calls = frame["query_call_weight"].to_numpy(dtype=float)
    beneficial_calls = frame["beneficial_call_weight"].to_numpy(dtype=float)
    unhelpful_calls = frame["unhelpful_call_weight"].to_numpy(dtype=float)
    positive_runs = frame["has_positive_opportunity"].to_numpy(dtype=bool)
    positive_utility_sum = float(np.sum(frame["best_available_positive_utility"].to_numpy(dtype=float)))
    captured_positive_utility_sum = float(
        np.sum(frame["captured_positive_utility"].to_numpy(dtype=float))
    )
    trigger_runs = float(np.sum(triggers))
    query_call_runs = float(np.sum(query_calls))
    return {
        "policy_name": str(frame["policy_name"].iloc[0]),
        "policy_category": str(frame["policy_category"].iloc[0]),
        "eligible_opportunity_scope": _single_string_value(
            frame,
            "eligible_opportunity_scope",
        ),
        "eligible_opportunity_rows": int(
            frame["eligible_opportunity_count"].to_numpy(dtype=int).sum()
        ),
        "layer": layer,
        "group": _group_label(group),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "sampling_phase": group.get("sampling_phase"),
        "sampling_opportunity_type": group.get("sampling_opportunity_type"),
        "prefix_algorithm": group.get("prefix_algorithm"),
        "selected_equals_default": group.get("selected_equals_default"),
        "selected_equals_prefix": group.get("selected_equals_prefix"),
        "handoff_required": group.get("handoff_required"),
        "policy_unit": "trajectory_first_trigger",
        "runs": int(len(frame)),
        "states": int(frame["states"].sum()),
        "observed_utility_gt_zero_rows": int(np.sum(positive_runs)),
        "observed_utility_gt_zero_rate": float(np.mean(positive_runs)),
        "policy_trigger_runs": trigger_runs,
        "policy_trigger_rate": float(np.mean(triggers)),
        "query_call_runs": query_call_runs,
        "query_call_rate": float(np.mean(query_calls)),
        "mean_observed_utility_under_triggers": (
            float(np.sum(utility) / trigger_runs)
            if trigger_runs > 0.0
            else 0.0
        ),
        "positive_run_capture_rate": float(np.sum(beneficial_calls) / max(np.sum(positive_runs), 1)),
        "utility_capture_rate": (
            captured_positive_utility_sum / positive_utility_sum if positive_utility_sum > 0.0 else 0.0
        ),
        "precision_u_gt_zero_under_triggers": float(
            np.sum(beneficial_calls) / max(trigger_runs, 1.0)
        ),
        "utility_le_zero_trigger_runs": float(np.sum(unhelpful_calls)),
        "utility_le_zero_rate_within_triggers": float(
            np.sum(unhelpful_calls) / max(trigger_runs, 1.0)
        ),
        "utility_le_zero_loss_sum": float(
            -np.sum(frame["unhelpful_utility"].to_numpy(dtype=float))
        ),
        "utility_sum": float(np.sum(utility)),
        "utility_mean": float(np.mean(utility)),
        "utility_median": float(np.median(utility)),
        "first_opportunity_mean_utility": float(np.mean(frame["first_opportunity_utility"])),
        "best_available_state_utility_mean": float(
            np.mean(frame["best_available_positive_utility"])
        ),
    }


def _relative_summary(policy_summary: pd.DataFrame) -> pd.DataFrame:
    overall = policy_summary[policy_summary["layer"] == "overall"].copy()
    baselines = overall[overall["policy_category"].isin({"baseline", "learned_baseline"})][
        [
            "policy_name",
            "eligible_opportunity_scope",
            "utility_mean",
            "utility_sum",
            "query_call_rate",
            "utility_capture_rate",
            "precision_u_gt_zero_under_triggers",
            "utility_le_zero_loss_sum",
        ]
    ].rename(
        columns={
            "policy_name": "baseline_policy",
            "eligible_opportunity_scope": "baseline_eligible_opportunity_scope",
            "utility_mean": "baseline_utility_mean",
            "utility_sum": "baseline_utility_sum",
            "query_call_rate": "baseline_query_call_rate",
            "utility_capture_rate": "baseline_utility_capture_rate",
            "precision_u_gt_zero_under_triggers": "baseline_precision_u_gt_zero_under_triggers",
            "utility_le_zero_loss_sum": "baseline_utility_le_zero_loss_sum",
        }
    )
    controller = overall[overall["policy_name"] == "current_controller"][
        [
            "eligible_opportunity_scope",
            "utility_mean",
            "utility_sum",
            "query_call_rate",
            "utility_capture_rate",
            "precision_u_gt_zero_under_triggers",
            "utility_le_zero_loss_sum",
        ]
    ].rename(
        columns={
            "eligible_opportunity_scope": "controller_eligible_opportunity_scope",
            "utility_mean": "controller_utility_mean",
            "utility_sum": "controller_utility_sum",
            "query_call_rate": "controller_query_call_rate",
            "utility_capture_rate": "controller_utility_capture_rate",
            "precision_u_gt_zero_under_triggers": "controller_precision_u_gt_zero_under_triggers",
            "utility_le_zero_loss_sum": "controller_utility_le_zero_loss_sum",
        }
    )
    if controller.empty:
        raise ValueError("current_controller row missing from policy summary")
    result = baselines.copy()
    for column, value in controller.iloc[0].items():
        result[column] = value
    result["utility_mean_delta_vs_baseline"] = result["controller_utility_mean"] - result["baseline_utility_mean"]
    result["utility_sum_delta_vs_baseline"] = result["controller_utility_sum"] - result["baseline_utility_sum"]
    result["query_call_rate_delta_vs_baseline"] = result["controller_query_call_rate"] - result["baseline_query_call_rate"]
    result["comparison_role"] = "whole_policy_baseline_comparison"
    milestone_t0 = result["baseline_policy"].astype(str).eq("milestone_only_T0")
    no_trigger = result["baseline_policy"].astype(str).isin(
        {"sbs_skip_reference", "never_query"}
    )
    result["opportunity_set_relation"] = np.select(
        [milestone_t0, no_trigger],
        [
            "milestone_subset_vs_dynamic_all_accepted",
            "no_trigger_baseline_vs_dynamic_all_accepted",
        ],
        default="same_all_accepted_eligible_schedule",
    )
    result["identical_eligible_opportunity_sets"] = ~milestone_t0 & ~no_trigger
    result["supports_rq2_behavior_increment"] = False
    result["interpretation"] = np.where(
        milestone_t0,
        (
            "difference combines opportunity scheduling, feature inputs, fitted scores, "
            "and first-trigger thresholds"
        ),
        "whole-policy difference under the policy-specific frozen protocol",
    )
    return result.reset_index(drop=True)


def _matched_random_repetition_summary(
    *,
    predictions: pd.DataFrame,
    calibration: MatchedRandomCalibration,
    random_repetitions: int,
) -> pd.DataFrame:
    if random_repetitions <= 0:
        raise ValueError("random_repetitions must be positive")
    rows = []
    for repetition in range(random_repetitions):
        call = _matched_random_first_trigger_mask(
            predictions,
            calibration=calibration,
            repetition=repetition,
        )
        state_frame = predictions.copy()
        state_frame.insert(0, "policy_name", "matched_rate_random")
        state_frame.insert(1, "policy_category", "baseline")
        state_frame["policy_trigger"] = call
        state_frame["query_called"] = call
        state_frame["policy_utility"] = np.where(call, state_frame[TARGET_COLUMN], 0.0)
        run_frame = _policy_run_frame(state_frame)
        utility = run_frame["policy_utility"].to_numpy(dtype=float)
        calls = run_frame["query_called"].to_numpy(dtype=bool)
        captured_positive = calls & (utility > 0.0)
        positive_utility_sum = float(
            np.sum(run_frame["best_available_positive_utility"].to_numpy(dtype=float))
        )
        call_runs = int(np.sum(calls))
        rows.append(
            {
                "repetition": repetition,
                "policy_unit": "trajectory_first_trigger",
                "runs": int(len(run_frame)),
                "states": int(run_frame["states"].sum()),
                "query_call_rate": float(np.mean(calls)),
                "utility_mean": float(np.mean(utility)),
                "utility_sum": float(np.sum(utility)),
                "utility_capture_rate": (
                    float(np.sum(utility[captured_positive]) / positive_utility_sum)
                    if positive_utility_sum > 0.0
                    else 0.0
                ),
                "precision_u_gt_zero_under_calls": float(np.sum(captured_positive) / max(call_runs, 1)),
            }
        )
    raw = pd.DataFrame(rows)
    summary_rows = []
    for metric in (
        "query_call_rate",
        "utility_mean",
        "utility_sum",
        "utility_capture_rate",
        "precision_u_gt_zero_under_calls",
    ):
        values = raw[metric].to_numpy(dtype=float)
        summary_rows.append(
            {
                "policy_name": "matched_rate_random",
                "metric": metric,
                "repetitions": random_repetitions,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if random_repetitions > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(summary_rows)


def _best_policy_summary(policy_summary: pd.DataFrame) -> pd.DataFrame:
    overall = policy_summary[policy_summary["layer"] == "overall"].copy()
    return overall.sort_values(["utility_mean", "utility_sum"], ascending=[False, False]).reset_index(drop=True)


def _group_label(group: dict[str, Any]) -> str:
    if not group:
        return "all"
    return ", ".join(f"{key}={value}" for key, value in group.items())


def _write_frame(frame: pd.DataFrame, stem: Path) -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), stem.with_suffix(".parquet"))


def _markdown_report(
    *,
    policy_summary: pd.DataFrame,
    relative_summary: pd.DataFrame,
    best_policy_summary: pd.DataFrame,
    random_repetition_summary: pd.DataFrame,
    model_name: str,
    threshold_mode: str,
    matched_call_rate: float,
    random_repetitions: int,
    expected_split: str,
) -> str:
    overall_columns = [
        "policy_name",
        "policy_category",
        "eligible_opportunity_scope",
        "policy_unit",
        "runs",
        "states",
        "utility_mean",
        "utility_sum",
        "query_call_rate",
        "utility_capture_rate",
        "precision_u_gt_zero_under_triggers",
        "utility_le_zero_loss_sum",
    ]
    relative_columns = [
        "baseline_policy",
        "baseline_eligible_opportunity_scope",
        "controller_eligible_opportunity_scope",
        "opportunity_set_relation",
        "supports_rq2_behavior_increment",
        "baseline_utility_mean",
        "controller_utility_mean",
        "utility_mean_delta_vs_baseline",
        "baseline_query_call_rate",
        "controller_query_call_rate",
        "query_call_rate_delta_vs_baseline",
        "interpretation",
    ]
    label_columns = [
        "policy_name",
        "group",
        "policy_unit",
        "runs",
        "utility_mean",
        "utility_sum",
        "query_call_rate",
        "utility_capture_rate",
        "precision_u_gt_zero_under_triggers",
    ]
    overall = best_policy_summary[overall_columns]
    action_relations = policy_summary[
        policy_summary["layer"].isin(("selected_equals_default", "selected_equals_prefix", "handoff_required"))
    ][["layer", *label_columns]].sort_values(["layer", "group", "utility_mean"], ascending=[True, True, False])
    lines = [
        "# Controller Baseline Comparison",
        "",
        "## 摘要",
        "",
        f"- 当前 controller：`{model_name}`，阈值口径为 `{threshold_mode}`。",
        f"- Time-only controller 使用同一模型 `{model_name}`，输入固定为 `X={{FE_ratio}}`（实现列 `bf_fe_ratio`）。",
        f"- Matched-rate Random 的 run-level 调用率 `{matched_call_rate:.6f}` 与触发 FE 分布只由 BBOB-train OOF 冻结。",
        f"- Matched-rate Random 使用 `{random_repetitions}` 个显式 `SeedSequence` 随机流，先在同一 trajectory 内平均，再进入政策汇总；逐 repetition 表仅作诊断。",
        f"- 指标在 `{expected_split}` 上按每 trajectory 首次触发计算，主口径是 `{TARGET_COLUMN}`。",
        "- `sbs_skip_reference` 与 `never_query` 在当前表中数值相同：都保留冻结的 SBS skip path，联合 Utility 为 0。",
        "",
        "## Overall Policies",
        "",
        _markdown_table(overall),
        "",
        "## Controller Relative To Baselines",
        "",
        _markdown_table(relative_summary[relative_columns]),
        "",
        "## Matched-rate Random Repetition Summary",
        "",
        _markdown_table(random_repetition_summary),
        "",
        "## Explicit Action-Relation Breakdown",
        "",
        _markdown_table(action_relations),
        "",
        "## 解释",
        "",
        "当 controller 的 mean utility 大于 0 时，它在该 Utility 口径下优于 `sbs_skip_reference` 和 `never_query`。",
        "`current_controller` 使用 all-accepted dynamic schedule，而 `milestone_only_T0` 只允许十二个预算里程碑；两者差异同时包含机会调度、输入特征、拟合分数和 first-trigger threshold，只能解释为整政策差异，不能归因为 Behavior 超出 FE ratio 的增量作用。",
        "RQ2 的唯一主要对比由 `decision-compare-feature-groups` 单独生成：milestone-only B3 与 milestone-only T0 使用严格相同的 state rows。",
        "与 `always_query` 的比较用于判断选择性调用固定 query 是否减少了无效调用。真正的 `pre_run_aas_fe0` 需要独立的 FE=0 query-only 运行，不由逐状态表伪造。",
        "Matched-rate Random 用于分离调用率和触发位置分布；评价集不参与调用率或触发分布拟合。",
    ]
    return "\n".join(lines) + "\n"


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the dynamic B3 controller with frozen whole-policy baselines. "
            "Dynamic B3 versus milestone-only T0 is not the RQ2 Behavior increment."
        )
    )
    parser.add_argument("--query-id", choices=sorted(LANDSCAPE_QUERY_SPECS), required=True)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--time-only-predictions", type=Path, default=None)
    parser.add_argument("--behavior-only-predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--threshold-mode", default=DEFAULT_THRESHOLD_MODE)
    parser.add_argument("--random-repetitions", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=1701)
    parser.add_argument("--expected-split", default=DEFAULT_EXPECTED_SPLIT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    query_root = decision_query_root(args.query_id)
    compare_controller_baselines(
        query_id=args.query_id,
        predictions_path=args.predictions
        or query_root / "feature_group_ablation/B3/all_accepted/validation_predictions.parquet",
        time_only_predictions_path=args.time_only_predictions
        or query_root / "feature_group_ablation/T0/milestone_only/validation_predictions.parquet",
        behavior_only_predictions_path=args.behavior_only_predictions
        or query_root
        / "feature_group_ablation/B3/all_accepted/validation_behavior_only_predictions.parquet",
        output_dir=args.output_dir or query_root / "controller_baseline_comparison",
        model_name=args.model_name,
        threshold_mode=args.threshold_mode,
        random_repetitions=args.random_repetitions,
        random_seed=args.random_seed,
        expected_split=args.expected_split,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
