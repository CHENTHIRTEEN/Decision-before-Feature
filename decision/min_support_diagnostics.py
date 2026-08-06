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
from decision.min_support_evaluate import (
    DEFAULT_TARGET_COLUMN,
    _check_family_split,
    _check_target,
    _decision_summary,
    _decision_threshold,
    _fit_model,
    _prediction_summary,
    _read_labels,
    _strategy_frame,
    _json_default,
)


GROUP_LAYERS = {
    "overall": [],
    "family": ["family"],
    "dimension": ["dimension"],
    "fe_ratio": ["FE_ratio"],
    "problem_id": ["problem_id"],
}


def run_diagnostics(
    *,
    train_labels_path: Path,
    validation_labels_path: Path,
    output_dir: Path,
    target_column: str,
    random_seed: int,
) -> dict[str, Any]:
    train = _read_labels(train_labels_path)
    validation = _read_labels(validation_labels_path)
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
    validation_predictions = model.predict(validation[feature_columns])
    prediction_runtime = perf_counter() - started
    prediction_runtime_per_state = prediction_runtime / max(len(validation), 1)

    relation_table = _relationship_table(train, validation, target_column)
    threshold_table, threshold_predictions = _threshold_tables(
        validation=validation,
        validation_predictions=validation_predictions,
        target_column=target_column,
        thresholds=thresholds,
        prediction_runtime_per_state=prediction_runtime_per_state,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    relationship_path = output_dir / "utility_relationships.parquet"
    threshold_path = output_dir / "threshold_policy_summary.parquet"
    prediction_path = output_dir / "threshold_predictions.parquet"
    summary_path = output_dir / "diagnostic_summary.json"
    pq.write_table(pa.Table.from_pandas(relation_table, preserve_index=False), relationship_path)
    pq.write_table(pa.Table.from_pandas(threshold_table, preserve_index=False), threshold_path)
    pq.write_table(pa.Table.from_pandas(threshold_predictions, preserve_index=False), prediction_path)

    observed = validation[target_column].to_numpy(dtype=float)
    summary = {
        "experiment": "min_support_diagnostics",
        "train_labels": str(train_labels_path),
        "validation_labels": str(validation_labels_path),
        "target_column": target_column,
        "rows": {"train": int(len(train)), "validation": int(len(validation))},
        "families": {
            "train": sorted(train["family"].astype(str).unique().tolist()),
            "validation": sorted(validation["family"].astype(str).unique().tolist()),
        },
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
        ],
        "thresholds": thresholds,
        "prediction": _prediction_summary(
            observed=observed,
            predicted=validation_predictions,
            mean_predicted=np.full_like(observed, fill_value=float(train[target_column].mean())),
        ),
        "threshold_decision": {
            mode: _decision_summary(observed=observed, predicted=validation_predictions, threshold=threshold)
            for mode, threshold in thresholds.items()
        },
        "utility_components": {
            "train": _component_summary(train, target_column),
            "validation": _component_summary(validation, target_column),
        },
        "outputs": {
            "relationships": str(relationship_path),
            "threshold_policy_summary": str(threshold_path),
            "threshold_predictions": str(prediction_path),
            "summary": str(summary_path),
        },
        "data_leakage_check": {
            "family_split_overlap": [],
            "decision_input_uses_only_behavior_features": True,
            "threshold_selected_from_validation": False,
        },
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote utility relationship diagnostics to {relationship_path}")
    print(f"wrote threshold policy diagnostics to {threshold_path}")
    print(f"wrote threshold predictions to {prediction_path}")
    print(f"wrote diagnostic summary to {summary_path}")
    return summary


def _relationship_table(train: pd.DataFrame, validation: pd.DataFrame, target_column: str) -> pd.DataFrame:
    frames = []
    for split_name, frame in (("train", train), ("validation", validation)):
        for layer, columns in GROUP_LAYERS.items():
            frames.append(_summarize_relationship(frame, split_name, layer, columns, target_column))
    return pd.concat(frames, ignore_index=True)


def _summarize_relationship(
    frame: pd.DataFrame,
    split_name: str,
    layer: str,
    group_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    if not group_columns:
        return pd.DataFrame([_relationship_row(frame, split_name, layer, {}, target_column)])
    rows = []
    for group_values, subset in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group = dict(zip(group_columns, group_values, strict=True))
        rows.append(_relationship_row(subset, split_name, layer, group, target_column))
    return pd.DataFrame(rows)


def _relationship_row(
    frame: pd.DataFrame,
    split_name: str,
    layer: str,
    group: dict[str, Any],
    target_column: str,
) -> dict[str, Any]:
    gain = frame["performance_gain_norm"].to_numpy(dtype=float)
    cost = frame["time_cost_norm"].to_numpy(dtype=float)
    utility = frame[target_column].to_numpy(dtype=float)
    return {
        "split": split_name,
        "layer": layer,
        "group": _group_label(group),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "problem_id": group.get("problem_id"),
        "rows": int(len(frame)),
        "mean_performance_gain_norm": float(np.mean(gain)),
        "median_performance_gain_norm": float(np.median(gain)),
        "mean_time_cost_norm": float(np.mean(cost)),
        "median_time_cost_norm": float(np.median(cost)),
        "mean_utility": float(np.mean(utility)),
        "median_utility": float(np.median(utility)),
        "utility_gt_zero_rate": float(np.mean(utility > 0.0)),
        "performance_gain_gt_zero_rate": float(np.mean(gain > 0.0)),
        "corr_gain_utility": _corr(gain, utility),
        "corr_time_cost_utility": _corr(cost, utility),
        "corr_gain_time_cost": _corr(gain, cost),
    }


def _threshold_tables(
    *,
    validation: pd.DataFrame,
    validation_predictions: np.ndarray,
    target_column: str,
    thresholds: dict[str, float],
    prediction_runtime_per_state: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    prediction_frames = []
    for mode, threshold in thresholds.items():
        evaluated = _strategy_frame(
            validation=validation,
            predictions=validation_predictions,
            target_column=target_column,
            threshold=threshold,
            random_analysis_probability=0.5,
            random_seed=1701,
            prediction_runtime_per_state=prediction_runtime_per_state,
        )
        evaluated.insert(0, "threshold_mode", mode)
        prediction_frames.append(evaluated)
        for layer, columns in GROUP_LAYERS.items():
            summaries.append(_threshold_summary(evaluated, mode, threshold, layer, columns, target_column))
    return pd.concat(summaries, ignore_index=True), pd.concat(prediction_frames, ignore_index=True)


def _threshold_summary(
    frame: pd.DataFrame,
    mode: str,
    threshold: float,
    layer: str,
    group_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    if not group_columns:
        return pd.DataFrame([_threshold_row(frame, mode, threshold, layer, {}, target_column)])
    rows = []
    for group_values, subset in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group = dict(zip(group_columns, group_values, strict=True))
        rows.append(_threshold_row(subset, mode, threshold, layer, group, target_column))
    return pd.DataFrame(rows)


def _threshold_row(
    frame: pd.DataFrame,
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
        "never_ela_mean_utility": 0.0,
        "always_ela_mean_utility": float(np.mean(frame["utility_always_ela_traditional_aas"])),
        "best_observed_action_mean_utility": float(np.mean(frame["utility_best_observed_analysis_action"])),
        "true_run_ela_count": int(np.sum(observed_need & decision_run)),
        "missed_run_ela_count": int(np.sum(observed_need & ~decision_run)),
        "unhelpful_run_ela_count": int(np.sum(~observed_need & decision_run)),
        "skip_when_unhelpful_count": int(np.sum(~observed_need & ~decision_run)),
    }


def _component_summary(frame: pd.DataFrame, target_column: str) -> dict[str, float]:
    return {
        "mean_performance_gain_norm": float(frame["performance_gain_norm"].mean()),
        "median_performance_gain_norm": float(frame["performance_gain_norm"].median()),
        "mean_time_cost_norm": float(frame["time_cost_norm"].mean()),
        "median_time_cost_norm": float(frame["time_cost_norm"].median()),
        "mean_utility": float(frame[target_column].mean()),
        "median_utility": float(frame[target_column].median()),
        "utility_gt_zero_rate": float((frame[target_column] > 0.0).mean()),
        "performance_gain_gt_zero_rate": float((frame["performance_gain_norm"] > 0.0).mean()),
    }


def _group_label(group: dict[str, Any]) -> str:
    if not group:
        return "overall"
    return "|".join(f"{name}={value}" for name, value in group.items())


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return None
    value = np.corrcoef(left, right)[0, 1]
    if not np.isfinite(value):
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose minimum-support utility labels and Decision thresholds.")
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
    parser.add_argument("--output-dir", type=Path, default=Path("results/decision/min_support/diagnostics"))
    args = parser.parse_args()
    run_diagnostics(
        train_labels_path=args.train_labels,
        validation_labels_path=args.validation_labels,
        output_dir=args.output_dir,
        target_column=args.target_column,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
