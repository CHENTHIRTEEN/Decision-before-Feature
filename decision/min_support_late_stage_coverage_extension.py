from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from behavior.features import BEHAVIOR_FEATURE_COLUMNS
from decision.min_support_changed_algorithm_aware_evaluate import _with_label_source
from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _check_target, _json_default, _read_labels
from decision.min_support_model_sensitivity import (
    _decision_scores,
    _decision_threshold_from_scores,
    _fit_model_spec,
    _model_specs,
)


STAGE_ORDER = ("early_stage", "middle_stage", "late_stage")
GROUP_LAYERS = {
    "overall": [],
    "family": ["family"],
    "dimension": ["dimension"],
    "fe_ratio": ["FE_ratio"],
    "problem_id": ["problem_id"],
    "seed": ["seed"],
    "family_dimension_fe_ratio": ["family", "dimension", "FE_ratio"],
    "problem_id_seed": ["problem_id", "seed"],
}
EVALUATION_DOMAINS = (
    "all_validation",
    "changed_algorithm_validation",
    "same_algorithm_reference",
    "target_holdout_all_rows",
    "target_holdout_changed_algorithm",
    "target_problem_changed_algorithm",
    "non_target_changed_algorithm",
)


def run_late_stage_coverage_extension(
    *,
    original_train_labels_path: Path,
    extension_train_labels_path: Path,
    validation_labels_path: Path,
    extension_config_path: Path,
    output_dir: Path,
    target_column: str,
    random_seed: int,
) -> dict[str, Any]:
    _check_target(target_column)
    original_train = _with_label_source(_read_labels(original_train_labels_path))
    extension_train = _with_label_source(_read_labels(extension_train_labels_path))
    validation = _with_label_source(_read_labels(validation_labels_path))
    config = _load_extension_config(extension_config_path)
    target_rows = _target_rows(config)

    original_train = _annotate_rows(original_train, target_rows, data_role="original_train")
    extension_train = _annotate_rows(extension_train, target_rows, data_role="extension_train")
    validation = _annotate_rows(validation, target_rows, data_role="validation")
    augmented_train = pd.concat([original_train, extension_train], ignore_index=True)

    train_sets = {
        "original_min_support_train": original_train,
        "late_stage_extended_train": augmented_train,
    }
    feature_columns = list(BEHAVIOR_FEATURE_COLUMNS)
    model_specs, unavailable_models = _model_specs(random_seed)

    label_summary = _label_summary(
        frames={
            "original_train": original_train,
            "extension_train": extension_train,
            "augmented_train": augmented_train,
            "validation": validation,
        },
        target_column=target_column,
    )
    thresholds_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    policy_frames: list[pd.DataFrame] = []
    score_summary_rows: list[dict[str, Any]] = []

    for train_name, train_frame in train_sets.items():
        for model_name, spec in model_specs.items():
            fitted = _fit_model_spec(spec, train_frame, feature_columns, target_column)
            train_scores = _decision_scores(fitted, spec, train_frame[feature_columns])
            validation_scores, prediction_runtime_per_state = _predict_validation(
                fitted=fitted,
                spec=spec,
                validation=validation,
                feature_columns=feature_columns,
            )
            thresholds = _thresholds_by_mode(
                train=train_frame,
                scores=train_scores,
                target_column=target_column,
            )
            thresholds_rows.extend(
                _threshold_rows(
                    train_name=train_name,
                    model_name=model_name,
                    spec=spec,
                    thresholds=thresholds,
                )
            )
            score_summary_rows.extend(
                _score_summary_rows(
                    train_name=train_name,
                    model_name=model_name,
                    spec=spec,
                    validation=validation,
                    scores=validation_scores,
                    target_column=target_column,
                )
            )

            for threshold_mode in ("zero", "train_utility", "stage_train_utility"):
                evaluated = _strategy_frame_with_thresholds(
                    validation=validation,
                    scores=validation_scores,
                    threshold_mode=threshold_mode,
                    thresholds=thresholds,
                    target_column=target_column,
                    random_seed=random_seed,
                    prediction_runtime_per_state=prediction_runtime_per_state,
                )
                evaluated.insert(0, "training_dataset", train_name)
                evaluated.insert(1, "model_name", model_name)
                evaluated.insert(2, "model_family", spec["model_family"])
                evaluated.insert(3, "score_semantics", spec["score_semantics"])
                evaluated.insert(4, "threshold_mode", threshold_mode)
                evaluated["label_source"] = validation["label_source"].to_numpy()
                evaluated["default_algorithm"] = validation["default_algorithm"].to_numpy()
                evaluated["selected_algorithm"] = validation["selected_algorithm"].to_numpy()
                evaluated["is_target_problem"] = validation["is_target_problem"].to_numpy(dtype=bool)
                evaluated["is_target_holdout_seed"] = validation["is_target_holdout_seed"].to_numpy(dtype=bool)
                prediction_frames.append(evaluated)

                for eval_domain in EVALUATION_DOMAINS:
                    domain_frame = _domain_frame(evaluated, eval_domain)
                    for layer, group_columns in GROUP_LAYERS.items():
                        policy_frames.append(
                            _policy_summary(
                                frame=domain_frame,
                                train_name=train_name,
                                model_name=model_name,
                                model_family=spec["model_family"],
                                score_semantics=spec["score_semantics"],
                                threshold_mode=threshold_mode,
                                eval_domain=eval_domain,
                                layer=layer,
                                group_columns=group_columns,
                                target_column=target_column,
                            )
                        )

    output_dir.mkdir(parents=True, exist_ok=True)
    augmented_train_path = output_dir / "augmented_train_labels.parquet"
    label_summary_path = output_dir / "late_stage_extension_label_summary.parquet"
    thresholds_path = output_dir / "late_stage_extension_thresholds.parquet"
    score_summary_path = output_dir / "late_stage_extension_score_summary.parquet"
    behavior_summary_path = output_dir / "late_stage_extension_behavior_feature_summary.parquet"
    predictions_path = output_dir / "late_stage_extension_predictions.parquet"
    policy_summary_path = output_dir / "late_stage_extension_policy_summary.parquet"
    summary_path = output_dir / "late_stage_extension_summary.json"

    threshold_summary = pd.DataFrame(thresholds_rows)
    score_summary = pd.DataFrame(score_summary_rows)
    behavior_summary = _behavior_feature_summary(
        frames={
            "original_train": original_train,
            "extension_train": extension_train,
            "augmented_train": augmented_train,
            "validation": validation,
        },
        target_column=target_column,
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    policy_summary = pd.concat(policy_frames, ignore_index=True)

    pq.write_table(pa.Table.from_pandas(augmented_train, preserve_index=False), augmented_train_path)
    pq.write_table(pa.Table.from_pandas(label_summary, preserve_index=False), label_summary_path)
    pq.write_table(pa.Table.from_pandas(threshold_summary, preserve_index=False), thresholds_path)
    pq.write_table(pa.Table.from_pandas(score_summary, preserve_index=False), score_summary_path)
    pq.write_table(pa.Table.from_pandas(behavior_summary, preserve_index=False), behavior_summary_path)
    pq.write_table(pa.Table.from_pandas(predictions, preserve_index=False), predictions_path)
    pq.write_table(pa.Table.from_pandas(policy_summary, preserve_index=False), policy_summary_path)

    family_overlap = sorted(set(augmented_train["family"].astype(str)).intersection(validation["family"].astype(str)))
    target_holdout_rows = validation[validation["is_target_holdout_seed"]]
    changed_target_holdout_rows = target_holdout_rows[target_holdout_rows["label_source"] == "changed_algorithm"]
    summary = {
        "experiment": "min_support_late_stage_coverage_extension_diagnostics",
        "research_question": (
            "Does adding minimal late-stage changed_algorithm training-label coverage allow behavior-only "
            "Decision Models to learn validation regions where U_ELA > 0?"
        ),
        "target_column": target_column,
        "original_train_labels": str(original_train_labels_path),
        "extension_train_labels": str(extension_train_labels_path),
        "validation_labels": str(validation_labels_path),
        "extension_config": str(extension_config_path),
        "rows": {
            "original_train": int(len(original_train)),
            "extension_train": int(len(extension_train)),
            "augmented_train": int(len(augmented_train)),
            "validation": int(len(validation)),
            "target_holdout_validation": int(len(target_holdout_rows)),
            "target_holdout_changed_algorithm_validation": int(len(changed_target_holdout_rows)),
        },
        "extension_label_coverage": _coverage_dict(extension_train, target_column),
        "validation_target_holdout_coverage": _coverage_dict(target_holdout_rows, target_column),
        "validation_target_holdout_changed_algorithm_coverage": _coverage_dict(
            changed_target_holdout_rows,
            target_column,
        ),
        "models": {
            name: {
                "model_family": spec["model_family"],
                "score_semantics": spec["score_semantics"],
                "training_target": spec["training_target"],
            }
            for name, spec in model_specs.items()
        },
        "unavailable_models": unavailable_models,
        "threshold_modes": ["zero", "train_utility", "stage_train_utility"],
        "stage_definition": {
            "early_stage": "FE_ratio <= 0.2",
            "middle_stage": "0.2 < FE_ratio < 0.5",
            "late_stage": "FE_ratio >= 0.5",
        },
        "evaluation_domains": list(EVALUATION_DOMAINS),
        "group_layers": GROUP_LAYERS,
        "feature_columns": feature_columns,
        "excluded_from_decision_input": [
            "function_id",
            "family",
            "problem_id",
            "dimension",
            "prefix_algorithm",
            "algorithm_parameters",
            "ELA feature columns",
            "selected_algorithm",
            "default_algorithm",
            "label_source",
            "target_holdout flags",
        ],
        "data_leakage_check": {
            "decision_input_uses_only_behavior_features": True,
            "ela_features_used_as_decision_input": False,
            "metadata_used_only_for_grouping": True,
            "original_utility_labels_modified": False,
            "formal_phase1_configs_modified": False,
            "formal_models_retrained": False,
            "diagnostic_models_only": True,
            "threshold_selected_from_validation": False,
            "diagnostic_train_validation_family_overlap": family_overlap,
            "overlap_is_intentional_late_stage_coverage_extension": True,
        },
        "outputs": {
            "augmented_train_labels": str(augmented_train_path),
            "label_summary": str(label_summary_path),
            "thresholds": str(thresholds_path),
            "score_summary": str(score_summary_path),
            "behavior_feature_summary": str(behavior_summary_path),
            "predictions": str(predictions_path),
            "policy_summary": str(policy_summary_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote augmented train labels to {augmented_train_path}")
    print(f"wrote late-stage extension label summary to {label_summary_path}")
    print(f"wrote late-stage extension thresholds to {thresholds_path}")
    print(f"wrote late-stage extension score summary to {score_summary_path}")
    print(f"wrote late-stage extension behavior feature summary to {behavior_summary_path}")
    print(f"wrote late-stage extension predictions to {predictions_path}")
    print(f"wrote late-stage extension policy summary to {policy_summary_path}")
    print(f"wrote late-stage extension summary to {summary_path}")
    return summary


def _load_extension_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config


def _target_rows(config: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {}
    for row in config.get("target_problem_seed_rows", []):
        problem_id = str(row["problem_id"])
        for seed in row.get("existing_holdout_validation_seeds", []):
            rows[(problem_id, int(seed))] = row
    return rows


def _annotate_rows(frame: pd.DataFrame, target_rows: dict[tuple[str, int], dict[str, Any]], data_role: str) -> pd.DataFrame:
    result = frame.copy()
    target_problem_ids = {problem_id for problem_id, _ in target_rows}
    target_pairs = set(target_rows)
    pairs = list(zip(result["problem_id"].astype(str), result["seed"].astype(int), strict=False))
    result["data_role"] = data_role
    result["stage"] = result["FE_ratio"].map(_stage)
    result["is_target_problem"] = result["problem_id"].astype(str).isin(target_problem_ids)
    result["is_target_holdout_seed"] = [pair in target_pairs for pair in pairs]
    return result


def _stage(fe_ratio: float) -> str:
    value = float(fe_ratio)
    if value <= 0.2:
        return "early_stage"
    if value < 0.5:
        return "middle_stage"
    return "late_stage"


def _predict_validation(
    *,
    fitted: Any,
    spec: dict[str, Any],
    validation: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, float]:
    started = perf_counter()
    scores = _decision_scores(fitted, spec, validation[feature_columns])
    runtime = perf_counter() - started
    return scores, runtime / max(len(validation), 1)


def _thresholds_by_mode(train: pd.DataFrame, scores: np.ndarray, target_column: str) -> dict[str, Any]:
    observed = train[target_column].to_numpy(dtype=float)
    stage_thresholds = {}
    for stage in STAGE_ORDER:
        mask = train["stage"].to_numpy() == stage
        if np.any(mask):
            stage_thresholds[stage] = _decision_threshold_from_scores(scores[mask], observed[mask])
        else:
            stage_thresholds[stage] = 0.0
    return {
        "zero": 0.0,
        "train_utility": _decision_threshold_from_scores(scores, observed),
        "stage_train_utility": stage_thresholds,
    }


def _threshold_rows(
    *,
    train_name: str,
    model_name: str,
    spec: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        {
            "training_dataset": train_name,
            "model_name": model_name,
            "model_family": spec["model_family"],
            "score_semantics": spec["score_semantics"],
            "threshold_mode": "zero",
            "stage": None,
            "threshold": float(thresholds["zero"]),
        },
        {
            "training_dataset": train_name,
            "model_name": model_name,
            "model_family": spec["model_family"],
            "score_semantics": spec["score_semantics"],
            "threshold_mode": "train_utility",
            "stage": None,
            "threshold": float(thresholds["train_utility"]),
        },
    ]
    rows.extend(
        {
            "training_dataset": train_name,
            "model_name": model_name,
            "model_family": spec["model_family"],
            "score_semantics": spec["score_semantics"],
            "threshold_mode": "stage_train_utility",
            "stage": stage,
            "threshold": float(threshold),
        }
        for stage, threshold in thresholds["stage_train_utility"].items()
    )
    return rows


def _strategy_frame_with_thresholds(
    *,
    validation: pd.DataFrame,
    scores: np.ndarray,
    threshold_mode: str,
    thresholds: dict[str, Any],
    target_column: str,
    random_seed: int,
    prediction_runtime_per_state: float,
) -> pd.DataFrame:
    frame = validation[
        [
            "split",
            "problem_id",
            "family",
            "dimension",
            "prefix_algorithm",
            "seed",
            "FE",
            "FE_ratio",
            "stage",
            "p_skip",
            "p_ela",
            "runtime_analysis",
            "runtime_selection",
            "runtime_skip_optimization",
            "runtime_ela_optimization",
            target_column,
        ]
    ].copy()
    observed = frame[target_column].to_numpy(dtype=float)
    if threshold_mode == "stage_train_utility":
        threshold_values = frame["stage"].map(thresholds["stage_train_utility"]).to_numpy(dtype=float)
    else:
        threshold_values = np.full(len(frame), float(thresholds[threshold_mode]))
    decision_run = scores > threshold_values
    random_run = _random_decisions(len(frame), random_seed)
    best_run = observed > 0.0

    frame["decision_score"] = scores.astype(float)
    frame["decision_threshold"] = threshold_values.astype(float)
    frame["decision_run_ela"] = decision_run
    frame["random_analysis_run_ela"] = random_run
    frame["best_observed_analysis_run_ela"] = best_run
    frame["utility_never_ela_sbs"] = 0.0
    frame["utility_always_ela_traditional_aas"] = observed
    frame["utility_random_analysis"] = np.where(random_run, observed, 0.0)
    frame["utility_decision_before_feature"] = np.where(decision_run, observed, 0.0)
    frame["utility_best_observed_analysis_action"] = np.maximum(observed, 0.0)
    frame["runtime_never_ela_sbs"] = frame["runtime_skip_optimization"]
    frame["runtime_always_ela_traditional_aas"] = (
        frame["runtime_analysis"] + frame["runtime_selection"] + frame["runtime_ela_optimization"]
    )
    frame["runtime_random_analysis"] = np.where(
        random_run,
        frame["runtime_always_ela_traditional_aas"],
        frame["runtime_never_ela_sbs"],
    )
    frame["runtime_decision_before_feature"] = (
        prediction_runtime_per_state
        + np.where(decision_run, frame["runtime_always_ela_traditional_aas"], frame["runtime_never_ela_sbs"])
    )
    frame["runtime_best_observed_analysis_action"] = np.where(
        best_run,
        frame["runtime_always_ela_traditional_aas"],
        frame["runtime_never_ela_sbs"],
    )
    return frame


def _random_decisions(size: int, random_seed: int) -> np.ndarray:
    seed_sequence = np.random.SeedSequence([int(random_seed), 202701, int(size)])
    rng = np.random.default_rng(seed_sequence)
    return rng.random(size) < 0.5


def _domain_frame(frame: pd.DataFrame, eval_domain: str) -> pd.DataFrame:
    if eval_domain == "all_validation":
        return frame
    if eval_domain == "changed_algorithm_validation":
        return frame[frame["label_source"] == "changed_algorithm"]
    if eval_domain == "same_algorithm_reference":
        return frame[frame["label_source"] == "same_algorithm"]
    if eval_domain == "target_holdout_all_rows":
        return frame[frame["is_target_holdout_seed"]]
    if eval_domain == "target_holdout_changed_algorithm":
        return frame[(frame["is_target_holdout_seed"]) & (frame["label_source"] == "changed_algorithm")]
    if eval_domain == "target_problem_changed_algorithm":
        return frame[(frame["is_target_problem"]) & (frame["label_source"] == "changed_algorithm")]
    if eval_domain == "non_target_changed_algorithm":
        return frame[(~frame["is_target_problem"]) & (frame["label_source"] == "changed_algorithm")]
    raise ValueError(f"unknown eval_domain: {eval_domain}")


def _policy_summary(
    *,
    frame: pd.DataFrame,
    train_name: str,
    model_name: str,
    model_family: str,
    score_semantics: str,
    threshold_mode: str,
    eval_domain: str,
    layer: str,
    group_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if not group_columns:
        return pd.DataFrame(
            [
                _policy_row(
                    frame=frame,
                    train_name=train_name,
                    model_name=model_name,
                    model_family=model_family,
                    score_semantics=score_semantics,
                    threshold_mode=threshold_mode,
                    eval_domain=eval_domain,
                    layer=layer,
                    group={},
                    target_column=target_column,
                )
            ]
        )
    rows = []
    for group_values, subset in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        rows.append(
            _policy_row(
                frame=subset,
                train_name=train_name,
                model_name=model_name,
                model_family=model_family,
                score_semantics=score_semantics,
                threshold_mode=threshold_mode,
                eval_domain=eval_domain,
                layer=layer,
                group=dict(zip(group_columns, group_values, strict=True)),
                target_column=target_column,
            )
        )
    return pd.DataFrame(rows)


def _policy_row(
    *,
    frame: pd.DataFrame,
    train_name: str,
    model_name: str,
    model_family: str,
    score_semantics: str,
    threshold_mode: str,
    eval_domain: str,
    layer: str,
    group: dict[str, Any],
    target_column: str,
) -> dict[str, Any]:
    observed = frame[target_column].to_numpy(dtype=float)
    need = observed > 0.0
    call = frame["decision_run_ela"].to_numpy(dtype=bool)
    positive_sum = float(np.sum(observed[need]))
    captured_sum = float(np.sum(observed[need & call]))
    unhelpful = observed[(~need) & call]
    return {
        "training_dataset": train_name,
        "model_name": model_name,
        "model_family": model_family,
        "score_semantics": score_semantics,
        "threshold_mode": threshold_mode,
        "eval_domain": eval_domain,
        "layer": layer,
        "group": _group_label(group),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "problem_id": group.get("problem_id"),
        "seed": group.get("seed"),
        "rows": int(len(frame)),
        "observed_utility_gt_zero_rows": int(np.sum(need)),
        "observed_utility_gt_zero_rate": float(np.mean(need)),
        "positive_utility_sum": positive_sum,
        "decision_ela_call_count": int(np.sum(call)),
        "decision_ela_call_rate": float(np.mean(call)),
        "captured_positive_rows": int(np.sum(need & call)),
        "missed_positive_rows": int(np.sum(need & ~call)),
        "unhelpful_call_rows": int(np.sum((~need) & call)),
        "positive_row_capture_rate": float(np.mean(call[need])) if np.any(need) else 0.0,
        "utility_capture_rate": captured_sum / positive_sum if positive_sum > 0.0 else 0.0,
        "captured_positive_utility_sum": captured_sum,
        "missed_positive_utility_sum": float(np.sum(observed[need & ~call])),
        "unhelpful_call_utility_sum": float(np.sum(unhelpful)),
        "unhelpful_call_cost_sum": float(-np.sum(unhelpful)),
        "decision_utility_sum": float(np.sum(np.where(call, observed, 0.0))),
        "decision_mean_utility": float(np.mean(np.where(call, observed, 0.0))),
        "always_ela_mean_utility": float(np.mean(observed)),
        "best_observed_action_mean_utility": float(np.mean(np.maximum(observed, 0.0))),
        "score_mean": float(frame["decision_score"].mean()),
        "score_median": float(frame["decision_score"].median()),
        "threshold_mean": float(frame["decision_threshold"].mean()),
    }


def _group_label(group: dict[str, Any]) -> str:
    if not group:
        return "overall"
    return "|".join(f"{key}={value}" for key, value in group.items())


def _label_summary(frames: dict[str, pd.DataFrame], target_column: str) -> pd.DataFrame:
    rows = []
    for data_split, frame in frames.items():
        for layer, group_columns in GROUP_LAYERS.items():
            if not group_columns:
                rows.append(_label_summary_row(data_split, frame, layer, {}, target_column))
                continue
            for group_values, subset in frame.groupby(group_columns, dropna=False):
                if not isinstance(group_values, tuple):
                    group_values = (group_values,)
                rows.append(
                    _label_summary_row(
                        data_split,
                        subset,
                        layer,
                        dict(zip(group_columns, group_values, strict=True)),
                        target_column,
                    )
                )
    return pd.DataFrame(rows)


def _label_summary_row(
    data_split: str,
    frame: pd.DataFrame,
    layer: str,
    group: dict[str, Any],
    target_column: str,
) -> dict[str, Any]:
    utility = frame[target_column].to_numpy(dtype=float)
    changed = frame["label_source"].to_numpy() == "changed_algorithm"
    return {
        "data_split": data_split,
        "layer": layer,
        "group": _group_label(group),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "problem_id": group.get("problem_id"),
        "seed": group.get("seed"),
        "rows": int(len(frame)),
        "changed_algorithm_rows": int(np.sum(changed)),
        "same_algorithm_rows": int(np.sum(~changed)),
        "utility_gt_zero_rows": int(np.sum(utility > 0.0)),
        "utility_gt_zero_rate": float(np.mean(utility > 0.0)),
        "positive_utility_sum": float(np.sum(utility[utility > 0.0])),
        "utility_sum": float(np.sum(utility)),
        "utility_mean": float(np.mean(utility)),
    }


def _score_summary_rows(
    *,
    train_name: str,
    model_name: str,
    spec: dict[str, Any],
    validation: pd.DataFrame,
    scores: np.ndarray,
    target_column: str,
) -> list[dict[str, Any]]:
    rows = []
    observed_all = validation[target_column].to_numpy(dtype=float)
    for eval_domain in EVALUATION_DOMAINS:
        frame = _domain_frame(validation.assign(decision_score=scores), eval_domain)
        if frame.empty:
            continue
        observed = frame[target_column].to_numpy(dtype=float)
        domain_scores = frame["decision_score"].to_numpy(dtype=float)
        rows.append(
            {
                "training_dataset": train_name,
                "model_name": model_name,
                "model_family": spec["model_family"],
                "score_semantics": spec["score_semantics"],
                "eval_domain": eval_domain,
                "rows": int(len(frame)),
                "utility_gt_zero_rows": int(np.sum(observed > 0.0)),
                "utility_gt_zero_rate": float(np.mean(observed > 0.0)),
                "score_mean": float(np.mean(domain_scores)),
                "score_median": float(np.median(domain_scores)),
                "score_min": float(np.min(domain_scores)),
                "score_q25": float(np.quantile(domain_scores, 0.25)),
                "score_q75": float(np.quantile(domain_scores, 0.75)),
                "score_max": float(np.max(domain_scores)),
                "score_positive_mean": _mean_or_none(domain_scores[observed > 0.0]),
                "score_positive_median": _median_or_none(domain_scores[observed > 0.0]),
                "score_non_positive_mean": _mean_or_none(domain_scores[observed <= 0.0]),
                "score_non_positive_median": _median_or_none(domain_scores[observed <= 0.0]),
                "global_observed_utility_mean": float(np.mean(observed_all)),
            }
        )
    return rows


def _mean_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def _median_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.median(values))


def _behavior_feature_summary(frames: dict[str, pd.DataFrame], target_column: str) -> pd.DataFrame:
    rows = []
    for data_split, frame in frames.items():
        domain_frames = {
            "all_rows": frame,
            "changed_algorithm": frame[frame["label_source"] == "changed_algorithm"],
            "same_algorithm": frame[frame["label_source"] == "same_algorithm"],
            "target_problem_changed_algorithm": frame[
                (frame["is_target_problem"]) & (frame["label_source"] == "changed_algorithm")
            ],
            "target_holdout_changed_algorithm": frame[
                (frame["is_target_holdout_seed"]) & (frame["label_source"] == "changed_algorithm")
            ],
        }
        for eval_domain, domain_frame in domain_frames.items():
            if domain_frame.empty:
                continue
            utility = domain_frame[target_column].to_numpy(dtype=float)
            role_frames = {
                "all_rows": domain_frame,
                "utility_gt_zero": domain_frame[utility > 0.0],
                "utility_le_zero": domain_frame[utility <= 0.0],
            }
            for role, role_frame in role_frames.items():
                if role_frame.empty:
                    continue
                for feature in BEHAVIOR_FEATURE_COLUMNS:
                    values = role_frame[feature].to_numpy(dtype=float)
                    values = values[np.isfinite(values)]
                    if values.size == 0:
                        continue
                    rows.append(
                        {
                            "data_split": data_split,
                            "eval_domain": eval_domain,
                            "utility_role": role,
                            "feature": feature,
                            "rows": int(len(role_frame)),
                            "finite_rows": int(values.size),
                            "mean": float(np.mean(values)),
                            "median": float(np.median(values)),
                            "std": float(np.std(values)),
                            "min": float(np.min(values)),
                            "q25": float(np.quantile(values, 0.25)),
                            "q75": float(np.quantile(values, 0.75)),
                            "max": float(np.max(values)),
                            "utility_gt_zero_rows": int(np.sum(role_frame[target_column].to_numpy(dtype=float) > 0.0)),
                        }
                    )
    return pd.DataFrame(rows)


def _coverage_dict(frame: pd.DataFrame, target_column: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "utility_gt_zero_rows": 0,
            "utility_gt_zero_rate": 0.0,
            "positive_utility_sum": 0.0,
        }
    utility = frame[target_column].to_numpy(dtype=float)
    return {
        "rows": int(len(frame)),
        "utility_gt_zero_rows": int(np.sum(utility > 0.0)),
        "utility_gt_zero_rate": float(np.mean(utility > 0.0)),
        "positive_utility_sum": float(np.sum(utility[utility > 0.0])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate min-support late-stage coverage extension diagnostics.")
    parser.add_argument(
        "--original-train-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_train/utility_labels.parquet"),
    )
    parser.add_argument(
        "--extension-train-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_train_late_stage_extension/utility_labels.parquet"),
    )
    parser.add_argument(
        "--validation-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_validation/utility_labels.parquet"),
    )
    parser.add_argument(
        "--extension-config",
        type=Path,
        default=Path("configs/min_support_bbob_train_late_stage_extension.yaml"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--random-seed", type=int, default=1701)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/late_stage_coverage_extension"),
    )
    args = parser.parse_args()
    run_late_stage_coverage_extension(
        original_train_labels_path=args.original_train_labels,
        extension_train_labels_path=args.extension_train_labels,
        validation_labels_path=args.validation_labels,
        extension_config_path=args.extension_config,
        output_dir=args.output_dir,
        target_column=args.target_column,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
