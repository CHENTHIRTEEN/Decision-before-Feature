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

from behavior.features import BEHAVIOR_FEATURE_COLUMNS
from decision.min_support_diagnostics import GROUP_LAYERS, _corr, _group_label
from decision.min_support_evaluate import (
    DEFAULT_TARGET_COLUMN,
    _check_family_split,
    _check_target,
    _decision_summary,
    _decision_threshold,
    _fit_model,
    _json_default,
    _prediction_summary,
    _read_labels,
    _strategy_frame,
)


def run_changed_algorithm_diagnostics(
    *,
    train_labels_path: Path,
    validation_labels_path: Path,
    output_dir: Path,
    target_column: str,
    random_seed: int,
) -> dict[str, Any]:
    train = _with_label_source(_read_labels(train_labels_path))
    validation = _with_label_source(_read_labels(validation_labels_path))
    _check_family_split(train, validation)
    _check_target(target_column)

    feature_columns = list(BEHAVIOR_FEATURE_COLUMNS)
    model = _fit_model(train, feature_columns, target_column, random_seed)
    thresholds = {
        "zero": _decision_threshold(
            model=model,
            train=train,
            feature_columns=feature_columns,
            target_column=target_column,
            threshold_mode="zero",
        ),
        "train_utility": _decision_threshold(
            model=model,
            train=train,
            feature_columns=feature_columns,
            target_column=target_column,
            threshold_mode="train_utility",
        ),
    }

    started = perf_counter()
    predictions = model.predict(validation[feature_columns])
    prediction_runtime = perf_counter() - started
    prediction_runtime_per_state = prediction_runtime / max(len(validation), 1)

    policy_rows = []
    prediction_rows = []
    for mode, threshold in thresholds.items():
        evaluated = _strategy_frame(
            validation=validation,
            predictions=predictions,
            target_column=target_column,
            threshold=threshold,
            random_analysis_probability=0.5,
            random_seed=random_seed,
            prediction_runtime_per_state=prediction_runtime_per_state,
        )
        evaluated.insert(0, "threshold_mode", mode)
        evaluated["label_source"] = validation["label_source"].to_numpy()
        evaluated["default_algorithm"] = validation["default_algorithm"].to_numpy()
        evaluated["selected_algorithm"] = validation["selected_algorithm"].to_numpy()
        prediction_rows.append(evaluated)
        for source in ("changed_algorithm", "same_algorithm"):
            source_frame = evaluated[evaluated["label_source"] == source]
            for layer, columns in GROUP_LAYERS.items():
                policy_rows.append(_policy_summary(source_frame, source, mode, threshold, layer, columns, target_column))

    policy_summary = pd.concat(policy_rows, ignore_index=True)
    threshold_predictions = pd.concat(prediction_rows, ignore_index=True)
    utility_summary = _utility_summary(validation, target_column)

    changed_validation = validation[validation["label_source"] == "changed_algorithm"]
    same_validation = validation[validation["label_source"] == "same_algorithm"]
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "changed_algorithm_threshold_policy_summary.parquet"
    utility_path = output_dir / "changed_algorithm_utility_summary.parquet"
    predictions_path = output_dir / "changed_algorithm_threshold_predictions.parquet"
    summary_path = output_dir / "changed_algorithm_diagnostic_summary.json"
    pq.write_table(pa.Table.from_pandas(policy_summary, preserve_index=False), policy_path)
    pq.write_table(pa.Table.from_pandas(utility_summary, preserve_index=False), utility_path)
    pq.write_table(pa.Table.from_pandas(threshold_predictions, preserve_index=False), predictions_path)

    summary = {
        "experiment": "min_support_changed_algorithm_diagnostics",
        "train_labels": str(train_labels_path),
        "validation_labels": str(validation_labels_path),
        "target_column": target_column,
        "rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "validation_changed_algorithm": int(len(changed_validation)),
            "validation_same_algorithm": int(len(same_validation)),
        },
        "thresholds": thresholds,
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
        ],
        "prediction": {
            "all_validation": _prediction_summary(
                observed=validation[target_column].to_numpy(dtype=float),
                predicted=predictions,
                mean_predicted=np.full(len(validation), float(train[target_column].mean())),
            ),
            "changed_algorithm_validation": _prediction_summary(
                observed=changed_validation[target_column].to_numpy(dtype=float),
                predicted=predictions[validation["label_source"].to_numpy() == "changed_algorithm"],
                mean_predicted=np.full(len(changed_validation), float(train[target_column].mean())),
            ),
            "same_algorithm_reference": _prediction_summary(
                observed=same_validation[target_column].to_numpy(dtype=float),
                predicted=predictions[validation["label_source"].to_numpy() == "same_algorithm"],
                mean_predicted=np.full(len(same_validation), float(train[target_column].mean())),
            ),
        },
        "decision": {
            mode: {
                "changed_algorithm_validation": _decision_summary(
                    observed=changed_validation[target_column].to_numpy(dtype=float),
                    predicted=predictions[validation["label_source"].to_numpy() == "changed_algorithm"],
                    threshold=threshold,
                ),
                "same_algorithm_reference": _decision_summary(
                    observed=same_validation[target_column].to_numpy(dtype=float),
                    predicted=predictions[validation["label_source"].to_numpy() == "same_algorithm"],
                    threshold=threshold,
                ),
            }
            for mode, threshold in thresholds.items()
        },
        "utility_distribution": {
            "changed_algorithm_validation": _source_distribution(changed_validation, target_column),
            "same_algorithm_reference": _source_distribution(same_validation, target_column),
        },
        "outputs": {
            "policy_summary": str(policy_path),
            "utility_summary": str(utility_path),
            "threshold_predictions": str(predictions_path),
            "summary": str(summary_path),
        },
        "data_leakage_check": {
            "family_split_overlap": [],
            "decision_input_uses_only_behavior_features": True,
            "label_source_used_only_for_evaluation_filter": True,
            "original_utility_labels_modified": False,
        },
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote changed-algorithm policy summary to {policy_path}")
    print(f"wrote changed-algorithm utility summary to {utility_path}")
    print(f"wrote changed-algorithm threshold predictions to {predictions_path}")
    print(f"wrote changed-algorithm diagnostic summary to {summary_path}")
    return summary


def _with_label_source(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["label_source"] = np.where(
        result["selected_algorithm"].astype(str) == result["default_algorithm"].astype(str),
        "same_algorithm",
        "changed_algorithm",
    )
    return result


def _policy_summary(
    frame: pd.DataFrame,
    source: str,
    mode: str,
    threshold: float,
    layer: str,
    group_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if not group_columns:
        return pd.DataFrame([_policy_row(frame, source, mode, threshold, layer, {}, target_column)])
    rows = []
    for group_values, subset in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group = dict(zip(group_columns, group_values, strict=True))
        rows.append(_policy_row(subset, source, mode, threshold, layer, group, target_column))
    return pd.DataFrame(rows)


def _policy_row(
    frame: pd.DataFrame,
    source: str,
    mode: str,
    threshold: float,
    layer: str,
    group: dict[str, Any],
    target_column: str,
) -> dict[str, Any]:
    observed_need = frame[target_column].to_numpy(dtype=float) > 0.0
    decision_run = frame["decision_run_ela"].to_numpy(dtype=bool)
    decision_utility = frame["utility_decision_before_feature"].to_numpy(dtype=float)
    return {
        "label_source": source,
        "threshold_mode": mode,
        "threshold": float(threshold),
        "layer": layer,
        "group": _group_label(group),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "problem_id": group.get("problem_id"),
        "rows": int(len(frame)),
        "observed_utility_gt_zero_rate": float(np.mean(observed_need)),
        "decision_ela_call_rate": float(np.mean(decision_run)),
        "decision_mean_utility": float(np.mean(decision_utility)),
        "decision_median_utility": float(np.median(decision_utility)),
        "always_ela_mean_utility": float(np.mean(frame["utility_always_ela_traditional_aas"])),
        "best_observed_action_mean_utility": float(np.mean(frame["utility_best_observed_analysis_action"])),
        "true_run_ela_count": int(np.sum(observed_need & decision_run)),
        "missed_run_ela_count": int(np.sum(observed_need & ~decision_run)),
        "unhelpful_run_ela_count": int(np.sum(~observed_need & decision_run)),
        "skip_when_unhelpful_count": int(np.sum(~observed_need & ~decision_run)),
    }


def _utility_summary(validation: pd.DataFrame, target_column: str) -> pd.DataFrame:
    frames = []
    for source in ("changed_algorithm", "same_algorithm"):
        source_frame = validation[validation["label_source"] == source]
        for layer, columns in GROUP_LAYERS.items():
            frames.append(_utility_layer_summary(source_frame, source, layer, columns, target_column))
    return pd.concat(frames, ignore_index=True)


def _utility_layer_summary(
    frame: pd.DataFrame,
    source: str,
    layer: str,
    group_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if not group_columns:
        return pd.DataFrame([_utility_row(frame, source, layer, {}, target_column)])
    rows = []
    for group_values, subset in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group = dict(zip(group_columns, group_values, strict=True))
        rows.append(_utility_row(subset, source, layer, group, target_column))
    return pd.DataFrame(rows)


def _utility_row(
    frame: pd.DataFrame,
    source: str,
    layer: str,
    group: dict[str, Any],
    target_column: str,
) -> dict[str, Any]:
    gain = frame["performance_gain_norm"].to_numpy(dtype=float)
    cost = frame["time_cost_norm"].to_numpy(dtype=float)
    utility = frame[target_column].to_numpy(dtype=float)
    positive = utility > 0.0
    return {
        "label_source": source,
        "layer": layer,
        "group": _group_label(group),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "problem_id": group.get("problem_id"),
        "rows": int(len(frame)),
        "utility_gt_zero_rows": int(np.sum(positive)),
        "utility_gt_zero_rate": float(np.mean(positive)),
        "performance_gain_gt_zero_rate": float(np.mean(gain > 0.0)),
        "mean_performance_gain_norm": float(np.mean(gain)),
        "median_performance_gain_norm": float(np.median(gain)),
        "mean_time_cost_norm": float(np.mean(cost)),
        "median_time_cost_norm": float(np.median(cost)),
        "mean_utility": float(np.mean(utility)),
        "median_utility": float(np.median(utility)),
        "sum_positive_utility": float(np.sum(utility[positive])),
        "mean_positive_utility": float(np.mean(utility[positive])) if np.any(positive) else 0.0,
        "corr_gain_utility": _corr(gain, utility),
        "corr_time_cost_utility": _corr(cost, utility),
    }


def _source_distribution(frame: pd.DataFrame, target_column: str) -> dict[str, float]:
    utility = frame[target_column].to_numpy(dtype=float)
    gain = frame["performance_gain_norm"].to_numpy(dtype=float)
    cost = frame["time_cost_norm"].to_numpy(dtype=float)
    positive = utility > 0.0
    return {
        "rows": int(len(frame)),
        "utility_gt_zero_rows": int(np.sum(positive)),
        "utility_gt_zero_rate": float(np.mean(positive)),
        "performance_gain_gt_zero_rate": float(np.mean(gain > 0.0)),
        "mean_performance_gain_norm": float(np.mean(gain)),
        "mean_time_cost_norm": float(np.mean(cost)),
        "mean_utility": float(np.mean(utility)),
        "median_utility": float(np.median(utility)),
        "sum_positive_utility": float(np.sum(utility[positive])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate min-support Decision behavior on changed-algorithm rows.")
    parser.add_argument(
        "--train-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_train/utility_labels.parquet"),
    )
    parser.add_argument(
        "--validation-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_validation/utility_labels.parquet"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--random-seed", type=int, default=1701)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/changed_algorithm_diagnostics"),
    )
    args = parser.parse_args()
    run_changed_algorithm_diagnostics(
        train_labels_path=args.train_labels,
        validation_labels_path=args.validation_labels,
        output_dir=args.output_dir,
        target_column=args.target_column,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
