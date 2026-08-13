from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from decision.model_protocol import FROZEN_THRESHOLD_MODE
from decision.query_contract import decision_query_root, validate_query_frame, validate_query_payload
from decision.sampling_opportunities import (
    STATE_KEY_COLUMNS,
    assert_aligned_decision_opportunities,
    assert_unique_state_keys,
    with_sampling_opportunity_type,
)
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS, get_query_spec
from trajectory.sampling import SAMPLING_METADATA_COLUMNS
from utility_labels.fields import NEED_QUERY_COLUMNS, UTILITY_VALUE_COLUMNS


DEFAULT_FEATURE_GROUPS = ("T0", "B1", "B2", "B3")
DEFAULT_TARGET_COLUMN = "u_query_lamT_1"
DEFAULT_AUXILIARY_LABEL_COLUMN = "need_query_lamT_1"
TARGET_COLUMN = DEFAULT_TARGET_COLUMN
AUXILIARY_LABEL_COLUMN = DEFAULT_AUXILIARY_LABEL_COLUMN
TRAIN_SPLIT = "bbob_train"
VALIDATION_SPLIT = "bbob_validation"
RUN_KEY_COLUMNS = (
    "split",
    "problem_id",
    "family",
    "dimension",
    "prefix_algorithm",
    "seed",
)
REQUIRED_BASE_COLUMNS = {
    "data_split",
    "model_name",
    "model_family",
    "split",
    "problem_id",
    "family",
    "dimension",
    "prefix_algorithm",
    "seed",
    "FE",
    "FE_ratio",
    *SAMPLING_METADATA_COLUMNS,
    "default_algorithm",
    "no_query_algorithm",
    "selection_reference_default_algorithm",
    "selector_target_transform",
    "selected_algorithm",
    "selected_equals_default",
    "selected_equals_prefix",
    "handoff_required",
    "skip_switches_from_prefix",
    "no_query_transition_mode",
    "query_transition_mode",
    "handoff_type",
    "decision_score",
    "decision_run_query_zero",
    "decision_utility_zero",
    f"decision_run_query_{FROZEN_THRESHOLD_MODE}",
    f"decision_utility_{FROZEN_THRESHOLD_MODE}",
}
GROUP_LAYERS = {
    "all": [],
    "selected_equals_default": ["selected_equals_default"],
    "selected_equals_prefix": ["selected_equals_prefix"],
    "handoff_required": ["handoff_required"],
    "sampling_phase": ["sampling_phase"],
    "sampling_opportunity_type": ["sampling_opportunity_type"],
    "dimension": ["dimension"],
    "prefix_algorithm": ["prefix_algorithm"],
    "family": ["family"],
}
PLOT_EXTENSIONS = ("png", "svg", "pdf")


def run_threshold_sweep(
    *,
    query_id: str,
    input_root: Path,
    feature_groups: list[str],
    output_dir: Path,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    overwrite: bool,
    target_column: str = DEFAULT_TARGET_COLUMN,
    auxiliary_label_column: str = DEFAULT_AUXILIARY_LABEL_COLUMN,
) -> dict[str, Any]:
    _set_utility_target_columns(
        target_column=target_column,
        auxiliary_label_column=auxiliary_label_column,
    )
    _check_args(threshold_min, threshold_max, threshold_step)
    if tuple(feature_groups) != DEFAULT_FEATURE_GROUPS:
        raise ValueError(
            "formal threshold comparison must use canonical T0/B1/B2/B3 exactly once and in order"
        )
    _check_output_paths(output_dir, overwrite)
    group_payloads = [
        _read_feature_group(input_root, feature_group, query_id) for feature_group in feature_groups
    ]
    _check_family_split(group_payloads)
    _check_decision_opportunity_alignment(group_payloads)

    dense_frames: list[pd.DataFrame] = []
    policy_frames: list[pd.DataFrame] = []
    distribution_frames: list[pd.DataFrame] = []
    threshold_policy_rows: list[dict[str, Any]] = []

    base_grid = _base_threshold_grid(threshold_min, threshold_max, threshold_step)
    for payload in group_payloads:
        feature_group = payload["feature_group"]
        train_oof = payload["train_oof"]
        validation = payload["validation"]
        existing_thresholds = payload["existing_thresholds"]
        for (model_name, model_family), model_train in train_oof.groupby(["model_name", "model_family"], sort=True):
            model_validation = validation[
                (validation["model_name"] == model_name) & (validation["model_family"] == model_family)
            ].copy()
            if model_validation.empty:
                raise ValueError(f"missing validation predictions for {feature_group}/{model_name}")
            thresholds = _model_threshold_grid(base_grid, model_train["decision_score"].to_numpy(dtype=float))

            train_dense = _sweep_metrics(
                frame=model_train,
                thresholds=thresholds,
                feature_group=feature_group,
                model_name=str(model_name),
                model_family=str(model_family),
                eval_split="train_oof",
                layer="all",
                group={},
                threshold_policy="sweep",
                uses_validation_utility_for_threshold=False,
            )
            dense_frames.append(train_dense)

            train_best_threshold = _best_threshold(train_dense)
            existing_threshold = existing_thresholds.get(str(model_name))
            policies = [
                {
                    "threshold_policy": "zero",
                    "threshold": 0.0,
                    "threshold_source_split": "fixed",
                    "deployable_policy": True,
                    "uses_validation_utility_for_threshold": False,
                },
                {
                    "threshold_policy": "frozen_oof_utility",
                    "threshold": existing_threshold,
                    "threshold_source_split": "train_oof",
                    "deployable_policy": True,
                    "uses_validation_utility_for_threshold": False,
                },
                {
                    "threshold_policy": "train_oof_sweep_best_check",
                    "threshold": train_best_threshold,
                    "threshold_source_split": "train_oof",
                    "deployable_policy": True,
                    "uses_validation_utility_for_threshold": False,
                },
            ]
            for policy in policies:
                if policy["threshold"] is None or not np.isfinite(float(policy["threshold"])):
                    continue
                threshold_policy_rows.append(
                    {
                        "feature_group": feature_group,
                        "model_name": str(model_name),
                        "model_family": str(model_family),
                        **policy,
                    }
                )
                policy_frames.append(
                    _policy_layer_summary(
                        train_oof,
                        model_validation,
                        feature_group=feature_group,
                        model_name=str(model_name),
                        model_family=str(model_family),
                        threshold_policy=str(policy["threshold_policy"]),
                        threshold=float(policy["threshold"]),
                        deployable_policy=bool(policy["deployable_policy"]),
                        uses_validation_utility_for_threshold=bool(policy["uses_validation_utility_for_threshold"]),
                    )
                )

        distribution_frames.append(
            _distribution_summary(train_oof, feature_group=feature_group, eval_split="train_oof")
        )
        distribution_frames.append(
            _distribution_summary(validation, feature_group=feature_group, eval_split="validation")
        )

    sweep_summary = pd.concat(dense_frames, ignore_index=True)
    best_thresholds = pd.concat(policy_frames, ignore_index=True)
    utility_distribution = pd.concat(distribution_frames, ignore_index=True)
    threshold_policy_table = pd.DataFrame(threshold_policy_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(sweep_summary, output_dir / "threshold_sweep_summary")
    _write_frame(best_thresholds, output_dir / "threshold_sweep_best_thresholds")
    _write_frame(utility_distribution, output_dir / "threshold_sweep_utility_distribution")
    _write_frame(threshold_policy_table, output_dir / "threshold_sweep_threshold_policies")

    plot_paths = _draw_plots(sweep_summary=sweep_summary, output_dir=output_dir)
    report_path = output_dir / "threshold_sweep_report.md"
    summary_path = output_dir / "threshold_sweep_summary.json"
    summary = {
        "experiment": "phase1_refined_sampling_threshold_sweep",
        "query_id": query_id,
        "query_protocol": get_query_spec(query_id).protocol,
        "sample_design_id": get_query_spec(query_id).sample_design_id,
        "research_question": (
            "Does the frozen run-level first-trigger threshold agree with a direct sweep over the same OOF scores, "
            "without using BBOB-validation Utility for threshold selection?"
        ),
        "input_root": str(input_root),
        "feature_groups": feature_groups,
        "target_column": TARGET_COLUMN,
        "auxiliary_label_column": AUXILIARY_LABEL_COLUMN,
        "decision_opportunity_set": "all accepted dynamic budget-milestone and causal-event rows",
        "threshold_grid": {
            "min": threshold_min,
            "max": threshold_max,
            "step": threshold_step,
            "train_oof_scores_added_per_model": True,
            "validation_threshold_grid_evaluated": False,
        },
        "threshold_policies": {
            "zero": "Run the fixed query when decision_score > 0 under the run-level first-trigger rule.",
            "frozen_oof_utility": "Use the run-level first-trigger threshold fitted from full BBOB-train family-OOF scores.",
            "train_oof_sweep_best_check": (
                "Recompute the best run-level first-trigger threshold on the same train-OOF scores as an implementation consistency check."
            ),
        },
        "rows": {
            "sweep_summary": int(len(sweep_summary)),
            "best_thresholds": int(len(best_thresholds)),
            "utility_distribution": int(len(utility_distribution)),
        },
        "outputs": {
            "sweep_summary": str(output_dir / "threshold_sweep_summary.parquet"),
            "best_thresholds": str(output_dir / "threshold_sweep_best_thresholds.parquet"),
            "utility_distribution": str(output_dir / "threshold_sweep_utility_distribution.parquet"),
            "threshold_policies": str(output_dir / "threshold_sweep_threshold_policies.parquet"),
            "plots": plot_paths,
            "report": str(report_path),
            "summary": str(summary_path),
        },
        "data_leakage_check": {
            "models_retrained": False,
            "utility_labels_regenerated": False,
            "original_utility_labels_modified": False,
            "decision_input_uses_query_features": False,
            "function_id_algorithm_id_or_optimizer_internal_parameters_used_as_input": False,
            "validation_utility_used_for_threshold_selection": False,
            "validation_threshold_grid_evaluated": False,
            "deployable_thresholds_use_validation_utility": False,
            "all_feature_groups_use_identical_decision_opportunities": True,
        },
    }
    report_path.write_text(
        _markdown_report(
            summary=summary,
            best_thresholds=best_thresholds,
            utility_distribution=utility_distribution,
            threshold_policy_table=threshold_policy_table,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote threshold sweep summary to {output_dir / 'threshold_sweep_summary.parquet'}")
    print(f"wrote threshold policy summary to {output_dir / 'threshold_sweep_best_thresholds.parquet'}")
    print(f"wrote threshold sweep report to {report_path}")
    return summary


def _check_args(threshold_min: float, threshold_max: float, threshold_step: float) -> None:
    if threshold_step <= 0.0:
        raise ValueError("threshold_step must be positive")
    if threshold_min >= threshold_max:
        raise ValueError("threshold_min must be smaller than threshold_max")


def _check_output_paths(output_dir: Path, overwrite: bool) -> None:
    names = (
        "threshold_sweep_summary",
        "threshold_sweep_best_thresholds",
        "threshold_sweep_utility_distribution",
        "threshold_sweep_threshold_policies",
    )
    outputs = []
    for name in names:
        outputs.extend([output_dir / f"{name}.csv", output_dir / f"{name}.parquet"])
    outputs.extend(
        [
            output_dir / "threshold_sweep_report.md",
            output_dir / "threshold_sweep_summary.json",
        ]
    )
    for stem in ("threshold_vs_summed_utility", "threshold_vs_call_rate", "utility_precision_call_curve"):
        outputs.extend(output_dir / f"{stem}.{extension}" for extension in PLOT_EXTENSIONS)
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"threshold sweep outputs already exist; pass --overwrite: {existing[0]}")


def _set_utility_target_columns(*, target_column: str, auxiliary_label_column: str) -> None:
    if target_column not in UTILITY_VALUE_COLUMNS:
        raise ValueError(f"target_column must be one of {list(UTILITY_VALUE_COLUMNS)}")
    if auxiliary_label_column not in NEED_QUERY_COLUMNS:
        raise ValueError(f"auxiliary_label_column must be one of {list(NEED_QUERY_COLUMNS)}")
    expected_label = NEED_QUERY_COLUMNS[UTILITY_VALUE_COLUMNS.index(target_column)]
    if auxiliary_label_column != expected_label:
        raise ValueError(f"{target_column} must use corresponding auxiliary label {expected_label}")
    global TARGET_COLUMN, AUXILIARY_LABEL_COLUMN
    TARGET_COLUMN = target_column
    AUXILIARY_LABEL_COLUMN = auxiliary_label_column


def _read_feature_group(input_root: Path, feature_group: str, query_id: str) -> dict[str, Any]:
    group_dir = input_root / feature_group
    train_path = group_dir / "train_oof_predictions.parquet"
    validation_path = group_dir / "validation_predictions.parquet"
    threshold_path = group_dir / "decision_thresholds.parquet"
    summary_path = group_dir / "full_decision_model_training_summary.json"
    for path in (train_path, validation_path, threshold_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(path)
    train_oof = pq.read_table(train_path).to_pandas()
    validation = pq.read_table(validation_path).to_pandas()
    validate_query_frame(train_oof, query_id=query_id, artifact=f"{feature_group} train OOF predictions")
    validate_query_frame(validation, query_id=query_id, artifact=f"{feature_group} validation predictions")
    _check_prediction_frame(train_oof, expected_data_split="train_oof", feature_group=feature_group)
    _check_prediction_frame(validation, expected_data_split="validation", feature_group=feature_group)
    train_oof = with_sampling_opportunity_type(train_oof, artifact=f"{feature_group} train OOF predictions")
    validation = with_sampling_opportunity_type(validation, artifact=f"{feature_group} validation predictions")
    thresholds = _read_existing_thresholds(threshold_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_query_payload(summary, query_id=query_id, artifact=f"{feature_group} training summary")
    if summary.get("feature_group") != feature_group:
        raise ValueError(f"training summary feature_group mismatch for {feature_group}")
    if summary.get("target_column") != TARGET_COLUMN:
        raise ValueError(
            f"{feature_group} training summary target_column must be {TARGET_COLUMN}, got {summary.get('target_column')}"
        )
    if summary.get("auxiliary_label_column") != AUXILIARY_LABEL_COLUMN:
        raise ValueError(
            f"{feature_group} training summary auxiliary_label_column must be {AUXILIARY_LABEL_COLUMN}, "
            f"got {summary.get('auxiliary_label_column')}"
        )
    return {
        "feature_group": feature_group,
        "train_oof": train_oof,
        "validation": validation,
        "existing_thresholds": thresholds,
        "summary": summary,
    }


def _check_prediction_frame(frame: pd.DataFrame, *, expected_data_split: str, feature_group: str) -> None:
    required_columns = set(REQUIRED_BASE_COLUMNS) | {TARGET_COLUMN, AUXILIARY_LABEL_COLUMN}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{feature_group} predictions missing required columns: {missing}")
    if set(frame["data_split"].astype(str).unique()) != {expected_data_split}:
        raise ValueError(f"{feature_group} predictions have unexpected data_split values")
    observed = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if observed.isna().any() or not np.isfinite(observed.to_numpy(dtype=float)).all():
        raise ValueError(f"{feature_group} {TARGET_COLUMN} must be non-null and finite")
    expected_need = observed.to_numpy(dtype=float) > 0.0
    if not np.array_equal(frame[AUXILIARY_LABEL_COLUMN].to_numpy(dtype=bool), expected_need):
        raise ValueError(f"{feature_group} {AUXILIARY_LABEL_COLUMN} must equal {TARGET_COLUMN} > 0")
    score = pd.to_numeric(frame["decision_score"], errors="coerce")
    if score.isna().any() or not np.isfinite(score.to_numpy(dtype=float)).all():
        raise ValueError(f"{feature_group} decision_score must be non-null and finite")


def _check_family_split(group_payloads: list[dict[str, Any]]) -> None:
    for payload in group_payloads:
        train_families = set(payload["train_oof"]["family"].astype(str))
        validation_families = set(payload["validation"]["family"].astype(str))
        overlap = sorted(train_families.intersection(validation_families))
        if overlap:
            raise ValueError(f"{payload['feature_group']} train and validation families overlap: {overlap}")


def _check_decision_opportunity_alignment(group_payloads: list[dict[str, Any]]) -> None:
    reference: dict[str, tuple[pd.DataFrame, str]] = {}
    for payload in group_payloads:
        feature_group = str(payload["feature_group"])
        for split_name in ("train_oof", "validation"):
            frame = payload[split_name]
            model_names = sorted(frame["model_name"].astype(str).unique().tolist())
            if not model_names:
                raise ValueError(f"{feature_group} {split_name} has no model rows")
            first_model = frame[frame["model_name"].astype(str) == model_names[0]].copy()
            assert_unique_state_keys(first_model, artifact=f"{feature_group} {split_name}/{model_names[0]}")
            for model_name in model_names[1:]:
                model_frame = frame[frame["model_name"].astype(str) == model_name].copy()
                assert_aligned_decision_opportunities(
                    first_model,
                    model_frame,
                    reference_artifact=f"{feature_group} {split_name}/{model_names[0]}",
                    candidate_artifact=f"{feature_group} {split_name}/{model_name}",
                )
            if split_name not in reference:
                reference[split_name] = (first_model, feature_group)
            else:
                reference_frame, reference_group = reference[split_name]
                assert_aligned_decision_opportunities(
                    reference_frame,
                    first_model,
                    reference_artifact=f"{reference_group} {split_name}",
                    candidate_artifact=f"{feature_group} {split_name}",
                )


def _read_existing_thresholds(path: Path) -> dict[str, float]:
    frame = pq.read_table(path).to_pandas()
    required = {"model_name", "threshold_mode", "threshold"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"decision threshold file missing columns: {missing}")
    rows = frame[frame["threshold_mode"].astype(str) == FROZEN_THRESHOLD_MODE].copy()
    return {str(row["model_name"]): float(row["threshold"]) for _, row in rows.iterrows()}


def _base_threshold_grid(threshold_min: float, threshold_max: float, threshold_step: float) -> np.ndarray:
    count = int(np.floor((threshold_max - threshold_min) / threshold_step)) + 1
    grid = threshold_min + np.arange(count + 1, dtype=float) * threshold_step
    grid = grid[grid <= threshold_max + threshold_step * 0.5]
    return np.unique(np.round(grid, 12))


def _model_threshold_grid(base_grid: np.ndarray, train_scores: np.ndarray) -> np.ndarray:
    return _threshold_candidates(np.concatenate([base_grid, train_scores.astype(float)]))


def _threshold_candidates(thresholds: np.ndarray) -> np.ndarray:
    return np.unique(
        np.concatenate(
            [
                np.asarray([-np.inf], dtype=float),
                np.asarray(thresholds, dtype=float),
                np.asarray([np.inf], dtype=float),
            ]
        )
    )


def _deployable_policy_name(threshold_policy: str) -> bool:
    return threshold_policy in {"zero", "frozen_oof_utility"}


def _run_group_key_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    columns = list(RUN_KEY_COLUMNS)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing run-key columns: {missing}")

    for protocol_column in ("query_protocol", "sampling_protocol"):
        if protocol_column in frame.columns and frame[protocol_column].nunique(dropna=False) > 1:
            raise ValueError(f"{protocol_column} is not constant within a run-group frame")

    if "FE_analysis_ratio" in frame.columns and frame["FE_analysis_ratio"].nunique(dropna=False) > 1:
        raise ValueError("FE_analysis_ratio is not constant within a run-group frame")

    return tuple(columns)


def _iter_run_groups(
    frame: pd.DataFrame,
    run_key_columns: tuple[str, ...],
):
    return frame.groupby(list(run_key_columns), sort=True, dropna=False)


def _ordered_run_frame(
    run_frame: pd.DataFrame,
    *,
    score_column: str,
    utility_column: str,
) -> pd.DataFrame:
    required = ["FE", score_column, utility_column]
    missing = [column for column in required if column not in run_frame.columns]
    if missing:
        raise ValueError(f"run frame missing columns: {missing}")

    order_columns = ["FE"]
    if "decision_opportunity_index" in run_frame.columns:
        if run_frame["decision_opportunity_index"].isna().any():
            raise ValueError("decision_opportunity_index contains NaN values")
        order_columns.append("decision_opportunity_index")
    elif run_frame["FE"].duplicated().any():
        raise ValueError(
            "multiple decision opportunities share the same FE, but decision_opportunity_index is unavailable"
        )

    ordered = run_frame.sort_values(order_columns, kind="mergesort").reset_index(drop=True)

    scores = ordered[score_column].to_numpy(dtype=float)
    utility = ordered[utility_column].to_numpy(dtype=float)

    if not np.all(np.isfinite(scores)):
        raise ValueError("decision scores contain NaN or infinite values")
    if not np.all(np.isfinite(utility)):
        raise ValueError("utility values contain NaN or infinite values")

    return ordered


def _first_trigger_run_contribution(
    ordered_run_frame: pd.DataFrame,
    *,
    threshold: float,
    score_column: str = "decision_score",
    utility_column: str = "u_query_lamT_1",
) -> tuple[bool, float, float | None, float | None]:
    scores = ordered_run_frame[score_column].to_numpy(dtype=float)
    observed = ordered_run_frame[utility_column].to_numpy(dtype=float)

    hit = np.flatnonzero(scores > threshold)
    if hit.size == 0:
        return False, 0.0, None, None

    first_index = int(hit[0])
    first_row = ordered_run_frame.iloc[first_index]
    first_fe = float(first_row["FE"])
    first_fe_ratio = float(first_row["FE_ratio"]) if "FE_ratio" in ordered_run_frame.columns else None
    first_utility = float(observed[first_index])
    return True, first_utility, first_fe, first_fe_ratio


def _run_best_available_positive_utility(
    ordered_run_frame: pd.DataFrame,
    *,
    score_column: str = "decision_score",
    utility_column: str = "u_query_lamT_1",
) -> float:
    scores = ordered_run_frame[score_column].to_numpy(dtype=float)
    utilities = ordered_run_frame[utility_column].to_numpy(dtype=float)
    if len(scores) == 0:
        return 0.0

    best = 0.0
    max_score_so_far = -np.inf
    for score, utility in zip(scores, utilities, strict=True):
        if score > max_score_so_far:
            if utility > best:
                best = float(utility)
            max_score_so_far = float(score)
    return float(max(0.0, best))


def _sweep_metrics(
    *,
    frame: pd.DataFrame,
    thresholds: np.ndarray,
    feature_group: str,
    model_name: str,
    model_family: str,
    eval_split: str,
    layer: str,
    group: dict[str, Any],
    threshold_policy: str,
    uses_validation_utility_for_threshold: bool,
    score_column: str = "decision_score",
    utility_column: str = "u_query_lamT_1",
) -> pd.DataFrame:
    run_key_columns = _run_group_key_columns(frame)

    prepared_runs: list[tuple[pd.DataFrame, float]] = []
    for _, run_frame in _iter_run_groups(frame, run_key_columns):
        ordered = _ordered_run_frame(
            run_frame,
            score_column=score_column,
            utility_column=utility_column,
        )
        best_available_positive_utility = _run_best_available_positive_utility(
            ordered,
            score_column=score_column,
            utility_column=utility_column,
        )
        prepared_runs.append((ordered, best_available_positive_utility))

    run_count = len(prepared_runs)
    if run_count == 0:
        raise ValueError("sweep_metrics requires at least one run")

    candidate_thresholds = _threshold_candidates(thresholds)
    oracle_positive_run_count = sum(best_available_positive_utility > 0.0 for _, best_available_positive_utility in prepared_runs)
    oracle_positive_utility_sum = float(sum(best_available_positive_utility for _, best_available_positive_utility in prepared_runs))

    rows: list[dict[str, Any]] = []
    for threshold in candidate_thresholds:
        call_runs = 0
        helpful_call_runs = 0
        unhelpful_call_runs = 0
        decision_utility_sum = 0.0
        selected_positive_utility_sum = 0.0
        unhelpful_call_cost_sum = 0.0
        trigger_fe_ratios: list[float] = []
        harmful_early_trigger_miss_runs = 0
        no_call_missed_positive_runs = 0

        for ordered, best_available_positive_utility in prepared_runs:
            triggered, first_utility, _, first_fe_ratio = _first_trigger_run_contribution(
                ordered,
                threshold=float(threshold),
                score_column=score_column,
                utility_column=utility_column,
            )

            if triggered:
                call_runs += 1
                decision_utility_sum += first_utility
                if first_fe_ratio is not None:
                    trigger_fe_ratios.append(first_fe_ratio)
                if first_utility > 0.0:
                    helpful_call_runs += 1
                    selected_positive_utility_sum += first_utility
                else:
                    unhelpful_call_runs += 1
                    unhelpful_call_cost_sum += -first_utility
                    if best_available_positive_utility > 0.0:
                        harmful_early_trigger_miss_runs += 1
            else:
                if best_available_positive_utility > 0.0:
                    no_call_missed_positive_runs += 1

        query_call_rate = _safe_ratio(call_runs, run_count)
        decision_mean_utility = _safe_ratio(decision_utility_sum, run_count)

        rows.append(
            {
                "feature_group": feature_group,
                "model_name": model_name,
                "model_family": model_family,
                "eval_split": eval_split,
                "layer": layer,
                "group": _group_label(group),
                "selected_equals_default": group.get("selected_equals_default"),
                "selected_equals_prefix": group.get("selected_equals_prefix"),
                "handoff_required": group.get("handoff_required"),
                "dimension": group.get("dimension"),
                "sampling_phase": group.get("sampling_phase"),
                "sampling_opportunity_type": group.get("sampling_opportunity_type"),
                "prefix_algorithm": group.get("prefix_algorithm"),
                "family": group.get("family"),
                "threshold_policy": threshold_policy,
                "threshold": float(threshold),
                "deployable_policy": _deployable_policy_name(threshold_policy),
                "uses_validation_utility_for_threshold": bool(uses_validation_utility_for_threshold),
                "runs": run_count,
                "oracle_u_gt_zero_runs": oracle_positive_run_count,
                "oracle_u_gt_zero_rate": _safe_ratio(oracle_positive_run_count, run_count),
                "oracle_positive_utility_sum": oracle_positive_utility_sum,
                "decision_query_call_runs": call_runs,
                "query_call_rate": query_call_rate,
                "helpful_call_runs": helpful_call_runs,
                "unhelpful_call_runs": unhelpful_call_runs,
                "precision_u_gt_zero_under_calls": _safe_ratio(helpful_call_runs, call_runs),
                "recall_positive_opportunity_runs": _safe_ratio(helpful_call_runs, oracle_positive_run_count),
                "selected_positive_utility_sum": selected_positive_utility_sum,
                "utility_capture_rate": _safe_ratio(selected_positive_utility_sum, oracle_positive_utility_sum),
                "decision_utility_sum": decision_utility_sum,
                "decision_mean_utility": decision_mean_utility,
                "mean_utility_under_calls": _safe_ratio(decision_utility_sum, call_runs),
                "unhelpful_call_cost_sum": unhelpful_call_cost_sum,
                "harmful_early_trigger_miss_runs": harmful_early_trigger_miss_runs,
                "no_call_missed_positive_runs": no_call_missed_positive_runs,
                "mean_trigger_FE_ratio": float(np.mean(trigger_fe_ratios)) if trigger_fe_ratios else float("nan"),
                "median_trigger_FE_ratio": float(np.median(trigger_fe_ratios)) if trigger_fe_ratios else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.zeros_like(np.asarray(numerator, dtype=float), dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    np.divide(numerator, denominator, out=output, where=denominator != 0.0)
    return output


def _best_threshold(sweep: pd.DataFrame) -> float:
    ordered = sweep.sort_values(
        ["decision_utility_sum", "query_call_rate", "threshold"],
        ascending=[False, True, False],
    )
    return float(ordered.iloc[0]["threshold"])


def _policy_layer_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_group: str,
    model_name: str,
    model_family: str,
    threshold_policy: str,
    threshold: float,
    deployable_policy: bool,
    uses_validation_utility_for_threshold: bool,
) -> pd.DataFrame:
    rows = []
    for eval_split, frame in (("train", train), ("validation", validation)):
        model_frame = frame[(frame["model_name"] == model_name) & (frame["model_family"] == model_family)].copy()
        for layer, group_columns in GROUP_LAYERS.items():
            if not group_columns:
                rows.append(
                    _single_threshold_row(
                        model_frame,
                        feature_group=feature_group,
                        model_name=model_name,
                        model_family=model_family,
                        eval_split=eval_split,
                        layer=layer,
                        group={},
                        threshold_policy=threshold_policy,
                        threshold=threshold,
                        deployable_policy=deployable_policy,
                        uses_validation_utility_for_threshold=uses_validation_utility_for_threshold,
                    )
                )
            else:
                for group_values, subset in model_frame.groupby(group_columns, dropna=False):
                    if not isinstance(group_values, tuple):
                        group_values = (group_values,)
                    rows.append(
                        _single_threshold_row(
                            subset,
                            feature_group=feature_group,
                            model_name=model_name,
                            model_family=model_family,
                            eval_split=eval_split,
                            layer=layer,
                            group=dict(zip(group_columns, group_values, strict=True)),
                            threshold_policy=threshold_policy,
                            threshold=threshold,
                            deployable_policy=deployable_policy,
                            uses_validation_utility_for_threshold=uses_validation_utility_for_threshold,
                        )
                    )
    return pd.DataFrame(rows)


def _single_threshold_row(
    frame: pd.DataFrame,
    *,
    feature_group: str,
    model_name: str,
    model_family: str,
    eval_split: str,
    layer: str,
    group: dict[str, Any],
    threshold_policy: str,
    threshold: float,
    deployable_policy: bool,
    uses_validation_utility_for_threshold: bool,
) -> dict[str, Any]:
    metrics = _sweep_metrics(
        frame=frame,
        thresholds=np.array([threshold], dtype=float),
        feature_group=feature_group,
        model_name=model_name,
        model_family=model_family,
        eval_split=eval_split,
        layer=layer,
        group=group,
        threshold_policy=threshold_policy,
        uses_validation_utility_for_threshold=uses_validation_utility_for_threshold,
    ).iloc[0]
    row = metrics.to_dict()
    row["deployable_policy"] = deployable_policy
    return row


def _distribution_summary(frame: pd.DataFrame, *, feature_group: str, eval_split: str) -> pd.DataFrame:
    rows = []
    for (model_name, model_family), model_frame in frame.groupby(["model_name", "model_family"], sort=True):
        for layer, group_columns in GROUP_LAYERS.items():
            if not group_columns:
                rows.append(
                    _distribution_row(
                        model_frame,
                        feature_group=feature_group,
                        model_name=str(model_name),
                        model_family=str(model_family),
                        eval_split=eval_split,
                        layer=layer,
                        group={},
                    )
                )
            else:
                for group_values, subset in model_frame.groupby(group_columns, dropna=False):
                    if not isinstance(group_values, tuple):
                        group_values = (group_values,)
                    rows.append(
                        _distribution_row(
                            subset,
                            feature_group=feature_group,
                            model_name=str(model_name),
                            model_family=str(model_family),
                            eval_split=eval_split,
                            layer=layer,
                            group=dict(zip(group_columns, group_values, strict=True)),
                        )
                    )
    return pd.DataFrame(rows)


def _distribution_row(
    frame: pd.DataFrame,
    *,
    feature_group: str,
    model_name: str,
    model_family: str,
    eval_split: str,
    layer: str,
    group: dict[str, Any],
) -> dict[str, Any]:
    observed = frame[TARGET_COLUMN].to_numpy(dtype=float)
    scores = frame["decision_score"].to_numpy(dtype=float)
    positive = observed > 0.0
    quantiles = [0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]
    observed_q = np.quantile(observed, quantiles)
    score_q = np.quantile(scores, quantiles)
    row = {
        "feature_group": feature_group,
        "model_name": model_name,
        "model_family": model_family,
        "eval_split": eval_split,
        "layer": layer,
        "group": _group_label(group),
        "selected_equals_default": group.get("selected_equals_default"),
        "selected_equals_prefix": group.get("selected_equals_prefix"),
        "handoff_required": group.get("handoff_required"),
        "dimension": group.get("dimension"),
        "sampling_phase": group.get("sampling_phase"),
        "sampling_opportunity_type": group.get("sampling_opportunity_type"),
        "prefix_algorithm": group.get("prefix_algorithm"),
        "family": group.get("family"),
        "rows": int(len(frame)),
        "positive_utility_rows": int(np.sum(positive)),
        "positive_utility_ratio": float(np.mean(positive)),
        "positive_utility_sum": float(np.sum(observed[positive])),
        "observed_utility_mean": float(np.mean(observed)),
        "observed_utility_std": float(np.std(observed)),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
    }
    for quantile, observed_value, score_value in zip(quantiles, observed_q, score_q, strict=True):
        label = f"q{int(quantile * 100):03d}"
        row[f"observed_utility_{label}"] = float(observed_value)
        row[f"score_{label}"] = float(score_value)
    return row


def _draw_plots(*, sweep_summary: pd.DataFrame, output_dir: Path) -> dict[str, list[str]]:
    train_oof = sweep_summary[
        (sweep_summary["eval_split"] == "train_oof")
        & (sweep_summary["layer"] == "all")
        & (sweep_summary["threshold_policy"] == "sweep")
    ].copy()
    plot_paths: dict[str, list[str]] = {}
    plot_paths["threshold_vs_summed_utility"] = _plot_threshold_metric(
        train_oof,
        output_dir=output_dir,
        stem="threshold_vs_summed_utility",
        y_column="decision_utility_sum",
        y_label="Run-level summed selected utility",
    )
    plot_paths["threshold_vs_call_rate"] = _plot_threshold_metric(
        train_oof,
        output_dir=output_dir,
        stem="threshold_vs_call_rate",
        y_column="query_call_rate",
        y_label="Run-level query call rate",
    )
    plot_paths["utility_precision_call_curve"] = _plot_utility_precision_call(train_oof, output_dir=output_dir)
    return plot_paths


def _plot_threshold_metric(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    stem: str,
    y_column: str,
    y_label: str,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(11, 6))
    for (feature_group, model_name), group in frame.groupby(["feature_group", "model_name"], sort=True):
        group = group.sort_values("threshold")
        ax.plot(group["threshold"], group[y_column], linewidth=1.2, label=f"{feature_group}/{model_name}")
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Threshold")
    ax.set_ylabel(y_label)
    ax.set_title(y_label + " across BBOB-train family-OOF thresholds under the run-level first-trigger rule")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return _save_plot(fig, output_dir, stem)


def _plot_utility_precision_call(frame: pd.DataFrame, *, output_dir: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for (feature_group, model_name), group in frame.groupby(["feature_group", "model_name"], sort=True):
        group = group.sort_values("query_call_rate")
        label = f"{feature_group}/{model_name}"
        axes[0].plot(
            group["precision_u_gt_zero_under_calls"],
            group["utility_capture_rate"],
            linewidth=1.2,
            label=label,
        )
        axes[1].plot(group["query_call_rate"], group["utility_capture_rate"], linewidth=1.2, label=label)
    axes[0].set_xlabel("Precision under Query calls")
    axes[0].set_ylabel("Utility capture rate")
    axes[0].set_title("Run-level utility capture vs precision")
    axes[1].set_xlabel("Run-level query call rate")
    axes[1].set_ylabel("Utility capture rate")
    axes[1].set_title("Run-level utility capture vs call rate")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[1].legend(fontsize=7, ncol=1)
    fig.tight_layout()
    return _save_plot(fig, output_dir, "utility_precision_call_curve")


def _save_plot(fig: Any, output_dir: Path, stem: str) -> list[str]:
    paths = []
    for extension in PLOT_EXTENSIONS:
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=180 if extension == "png" else None)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _write_frame(frame: pd.DataFrame, path_without_suffix: Path) -> None:
    frame.to_csv(path_without_suffix.with_suffix(".csv"), index=False)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path_without_suffix.with_suffix(".parquet"))


def _markdown_report(
    *,
    summary: dict[str, Any],
    best_thresholds: pd.DataFrame,
    utility_distribution: pd.DataFrame,
    threshold_policy_table: pd.DataFrame,
) -> str:
    validation_all = best_thresholds[
        (best_thresholds["eval_split"] == "validation") & (best_thresholds["layer"] == "all")
    ].copy()
    deployable = validation_all[validation_all["deployable_policy"]].sort_values(
        ["decision_utility_sum", "utility_capture_rate"], ascending=False
    )
    distribution_all = utility_distribution[
        (utility_distribution["eval_split"] == "validation") & (utility_distribution["layer"] == "all")
    ].sort_values(["feature_group", "model_name"])
    policy_columns = [
        "feature_group",
        "model_name",
        "threshold_policy",
        "threshold",
        "threshold_source_split",
        "deployable_policy",
        "uses_validation_utility_for_threshold",
    ]
    metric_columns = [
        "feature_group",
        "model_name",
        "threshold_policy",
        "threshold",
        "query_call_rate",
        "precision_u_gt_zero_under_calls",
        "recall_u_gt_zero",
        "utility_capture_rate",
        "decision_utility_sum",
        "average_selected_utility",
        "unhelpful_call_cost_sum",
    ]
    return "\n".join(
        [
            "# Threshold sweep report",
            "",
            "## Scope",
            "",
            "- Existing Decision predictions are reused; no model is retrained.",
            "- Utility labels are not regenerated or modified.",
            "- Threshold candidates are selected from BBOB-train family-OOF predictions only under the run-level first-trigger rule.",
            "- BBOB-validation Utility is evaluated only at thresholds frozen before validation under the same run-level first-trigger rule.",
            "- Metadata layers are used only for stratified reporting under the run-level first-trigger rule.",
            "",
            "## Threshold policies",
            "",
            _markdown_table(threshold_policy_table[policy_columns].sort_values(["feature_group", "model_name", "threshold_policy"])),
            "",
            "## Best deployable validation policies under the run-level first-trigger rule",
            "",
            _markdown_table(deployable[metric_columns].head(20)),
            "",
            "## Validation utility and score distribution",
            "",
            _markdown_table(
                distribution_all[
                    [
                        "feature_group",
                        "model_name",
                        "rows",
                        "positive_utility_ratio",
                        "positive_utility_sum",
                        "observed_utility_mean",
                        "observed_utility_q050",
                        "score_mean",
                        "score_q050",
                        "score_q095",
                    ]
                ]
            ),
            "",
            "## Output files",
            "",
            _markdown_table(pd.DataFrame([{"name": key, "path": value} for key, value in summary["outputs"].items()])),
            "",
        ]
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    def format_value(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return ", ".join(str(item) for item in value)
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value)

    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(format_value(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _group_label(group: dict[str, Any]) -> str:
    if not group:
        return "all"
    return "|".join(f"{key}={value}" for key, value in group.items())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check frozen run-level first-trigger Decision thresholds against BBOB-train family-OOF score sweeps."
    )
    parser.add_argument("--query-id", choices=sorted(LANDSCAPE_QUERY_SPECS), required=True)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--feature-groups", nargs="+", default=list(DEFAULT_FEATURE_GROUPS))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold-min", type=float, default=-0.5)
    parser.add_argument("--threshold-max", type=float, default=0.5)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--target-column", choices=UTILITY_VALUE_COLUMNS, default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--auxiliary-label-column", choices=NEED_QUERY_COLUMNS, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    auxiliary_label_column = args.auxiliary_label_column or NEED_QUERY_COLUMNS[
        UTILITY_VALUE_COLUMNS.index(args.target_column)
    ]
    query_root = decision_query_root(args.query_id)
    run_threshold_sweep(
        query_id=args.query_id,
        input_root=args.input_root or query_root / "feature_group_ablation",
        feature_groups=list(args.feature_groups),
        output_dir=args.output_dir or query_root / "threshold_sweep",
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
        overwrite=args.overwrite,
        target_column=args.target_column,
        auxiliary_label_column=auxiliary_label_column,
    )


if __name__ == "__main__":
    main()
