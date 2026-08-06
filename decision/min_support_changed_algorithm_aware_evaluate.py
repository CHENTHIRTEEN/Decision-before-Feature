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
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from behavior.features import BEHAVIOR_FEATURE_COLUMNS
from decision.min_support_diagnostics import GROUP_LAYERS, _group_label
from decision.min_support_evaluate import (
    DEFAULT_TARGET_COLUMN,
    _check_family_split,
    _check_target,
    _json_default,
    _prediction_summary,
    _read_labels,
    _strategy_frame,
)


EVALUATION_DOMAINS = {
    "changed_algorithm_validation": "changed_algorithm",
    "same_algorithm_reference": "same_algorithm",
}


def run_changed_algorithm_aware_evaluation(
    *,
    train_labels_path: Path,
    validation_labels_path: Path,
    output_dir: Path,
    target_column: str,
    random_seed: int,
    changed_weight: float,
    same_weight: float,
) -> dict[str, Any]:
    train = _with_label_source(_read_labels(train_labels_path))
    validation = _with_label_source(_read_labels(validation_labels_path))
    _check_family_split(train, validation)
    _check_target(target_column)
    _check_weights(changed_weight, same_weight)

    feature_columns = list(BEHAVIOR_FEATURE_COLUMNS)
    strategies = _fit_strategy_models(
        train=train,
        feature_columns=feature_columns,
        target_column=target_column,
        random_seed=random_seed,
        changed_weight=changed_weight,
        same_weight=same_weight,
    )

    prediction_summaries = []
    policy_summaries = []
    prediction_frames = []
    threshold_summary: dict[str, dict[str, float]] = {}

    for strategy_name, strategy in strategies.items():
        model = strategy["model"]
        threshold_train = strategy["threshold_train"]
        threshold_weights = strategy["threshold_weights"]
        thresholds = {
            "zero": 0.0,
            "train_utility": _weighted_decision_threshold(
                predictions=model.predict(threshold_train[feature_columns]),
                observed=threshold_train[target_column].to_numpy(dtype=float),
                sample_weight=threshold_weights,
            ),
        }
        threshold_summary[strategy_name] = thresholds

        started = perf_counter()
        predictions = model.predict(validation[feature_columns])
        prediction_runtime = perf_counter() - started
        prediction_runtime_per_state = prediction_runtime / max(len(validation), 1)

        prediction_summaries.extend(
            _prediction_domain_rows(
                strategy_name=strategy_name,
                validation=validation,
                predictions=predictions,
                target_column=target_column,
                train_mean=float(threshold_train[target_column].mean()),
            )
        )

        for threshold_mode, threshold in thresholds.items():
            evaluated = _strategy_frame(
                validation=validation,
                predictions=predictions,
                target_column=target_column,
                threshold=threshold,
                random_analysis_probability=0.5,
                random_seed=random_seed,
                prediction_runtime_per_state=prediction_runtime_per_state,
            )
            evaluated.insert(0, "training_strategy", strategy_name)
            evaluated.insert(1, "threshold_mode", threshold_mode)
            evaluated["label_source"] = validation["label_source"].to_numpy()
            evaluated["default_algorithm"] = validation["default_algorithm"].to_numpy()
            evaluated["selected_algorithm"] = validation["selected_algorithm"].to_numpy()
            prediction_frames.append(evaluated)

            for eval_domain, label_source in EVALUATION_DOMAINS.items():
                domain_frame = evaluated[evaluated["label_source"] == label_source]
                for layer, columns in GROUP_LAYERS.items():
                    policy_summaries.append(
                        _policy_summary(
                            frame=domain_frame,
                            training_strategy=strategy_name,
                            threshold_mode=threshold_mode,
                            threshold=threshold,
                            eval_domain=eval_domain,
                            layer=layer,
                            group_columns=columns,
                            target_column=target_column,
                        )
                    )

    policy_summary = pd.concat(policy_summaries, ignore_index=True)
    prediction_summary = pd.DataFrame(prediction_summaries)
    threshold_predictions = pd.concat(prediction_frames, ignore_index=True)
    utility_distribution = _utility_distribution(validation, target_column)

    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "changed_aware_policy_summary.parquet"
    prediction_summary_path = output_dir / "changed_aware_prediction_summary.parquet"
    predictions_path = output_dir / "changed_aware_predictions.parquet"
    utility_path = output_dir / "changed_aware_utility_distribution.parquet"
    summary_path = output_dir / "changed_aware_summary.json"
    pq.write_table(pa.Table.from_pandas(policy_summary, preserve_index=False), policy_path)
    pq.write_table(pa.Table.from_pandas(prediction_summary, preserve_index=False), prediction_summary_path)
    pq.write_table(pa.Table.from_pandas(threshold_predictions, preserve_index=False), predictions_path)
    pq.write_table(pa.Table.from_pandas(utility_distribution, preserve_index=False), utility_path)

    train_counts = train["label_source"].value_counts().sort_index().to_dict()
    validation_counts = validation["label_source"].value_counts().sort_index().to_dict()
    summary = {
        "experiment": "min_support_changed_algorithm_aware_decision_evaluation",
        "research_question": (
            "Does making the Decision Model training aware of selected/default algorithm changes improve "
            "ELA utility capture on changed-algorithm validation rows while limiting calls on same-algorithm "
            "reference rows?"
        ),
        "train_labels": str(train_labels_path),
        "validation_labels": str(validation_labels_path),
        "target_column": target_column,
        "rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "train_by_label_source": {str(key): int(value) for key, value in train_counts.items()},
            "validation_by_label_source": {str(key): int(value) for key, value in validation_counts.items()},
        },
        "training_strategies": {
            "full_label_rf": "Random Forest trained on all original utility labels.",
            "changed_only_rf": "Random Forest trained only on rows where selected_algorithm differs from default_algorithm.",
            "changed_weighted_rf": (
                "Random Forest trained on all original utility labels with higher sample weight for changed_algorithm rows."
            ),
        },
        "changed_weight": float(changed_weight),
        "same_weight": float(same_weight),
        "thresholds": threshold_summary,
        "train_utility_threshold_selection": {
            "full_label_rf": "unweighted mean policy utility on all train rows",
            "changed_only_rf": "unweighted mean policy utility on changed_algorithm train rows",
            "changed_weighted_rf": "weighted mean policy utility on all train rows using the configured label_source weights",
            "threshold_selected_from_validation": False,
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
            "label_source",
        ],
        "evaluation_domains": EVALUATION_DOMAINS,
        "outputs": {
            "policy_summary": str(policy_path),
            "prediction_summary": str(prediction_summary_path),
            "threshold_predictions": str(predictions_path),
            "utility_distribution": str(utility_path),
            "summary": str(summary_path),
        },
        "data_leakage_check": {
            "family_split_overlap": [],
            "decision_input_uses_only_behavior_features": True,
            "label_source_used_only_for_training_strategy_and_evaluation_grouping": True,
            "original_utility_labels_modified": False,
        },
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote changed-aware policy summary to {policy_path}")
    print(f"wrote changed-aware prediction summary to {prediction_summary_path}")
    print(f"wrote changed-aware threshold predictions to {predictions_path}")
    print(f"wrote changed-aware utility distribution to {utility_path}")
    print(f"wrote changed-aware summary to {summary_path}")
    return summary


def _with_label_source(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["label_source"] = np.where(
        result["selected_algorithm"].astype(str) == result["default_algorithm"].astype(str),
        "same_algorithm",
        "changed_algorithm",
    )
    return result


def _check_weights(changed_weight: float, same_weight: float) -> None:
    if changed_weight <= 0.0 or same_weight <= 0.0:
        raise ValueError("changed_weight and same_weight must be positive")


def _fit_strategy_models(
    *,
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    random_seed: int,
    changed_weight: float,
    same_weight: float,
) -> dict[str, dict[str, Any]]:
    changed_train = train[train["label_source"] == "changed_algorithm"].copy()
    if changed_train.empty:
        raise ValueError("changed_only_rf requires at least one changed_algorithm train row")

    changed_aware_weights = np.where(
        train["label_source"].to_numpy() == "changed_algorithm",
        float(changed_weight),
        float(same_weight),
    )
    return {
        "full_label_rf": {
            "model": _fit_rf(train, feature_columns, target_column, random_seed, sample_weight=None),
            "threshold_train": train,
            "threshold_weights": None,
        },
        "changed_only_rf": {
            "model": _fit_rf(changed_train, feature_columns, target_column, random_seed, sample_weight=None),
            "threshold_train": changed_train,
            "threshold_weights": None,
        },
        "changed_weighted_rf": {
            "model": _fit_rf(train, feature_columns, target_column, random_seed, sample_weight=changed_aware_weights),
            "threshold_train": train,
            "threshold_weights": changed_aware_weights,
        },
    }


def _fit_rf(
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    random_seed: int,
    sample_weight: np.ndarray | None,
) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=random_seed,
                    min_samples_leaf=3,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    if sample_weight is None:
        model.fit(train[feature_columns], train[target_column])
    else:
        model.fit(train[feature_columns], train[target_column], regressor__sample_weight=sample_weight)
    return model


def _weighted_decision_threshold(
    *,
    predictions: np.ndarray,
    observed: np.ndarray,
    sample_weight: np.ndarray | None,
) -> float:
    candidates = np.unique(np.concatenate(([0.0], predictions)))
    weights = np.ones(len(observed), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    best_threshold = 0.0
    best_utility = -np.inf
    for threshold in candidates:
        policy_utility = np.where(predictions > threshold, observed, 0.0)
        mean_utility = float(np.average(policy_utility, weights=weights))
        if mean_utility > best_utility:
            best_utility = mean_utility
            best_threshold = float(threshold)
    return best_threshold


def _prediction_domain_rows(
    *,
    strategy_name: str,
    validation: pd.DataFrame,
    predictions: np.ndarray,
    target_column: str,
    train_mean: float,
) -> list[dict[str, Any]]:
    rows = []
    for eval_domain, label_source in EVALUATION_DOMAINS.items():
        mask = validation["label_source"].to_numpy() == label_source
        domain_observed = validation.loc[mask, target_column].to_numpy(dtype=float)
        domain_predictions = predictions[mask]
        summary = _prediction_summary(
            observed=domain_observed,
            predicted=domain_predictions,
            mean_predicted=np.full(len(domain_observed), train_mean, dtype=float),
        )
        rows.append({"training_strategy": strategy_name, "eval_domain": eval_domain, **summary})
    return rows


def _policy_summary(
    *,
    frame: pd.DataFrame,
    training_strategy: str,
    threshold_mode: str,
    threshold: float,
    eval_domain: str,
    layer: str,
    group_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if not group_columns:
        return pd.DataFrame(
            [_policy_row(frame, training_strategy, threshold_mode, threshold, eval_domain, layer, {}, target_column)]
        )
    rows = []
    for group_values, subset in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group = dict(zip(group_columns, group_values, strict=True))
        rows.append(_policy_row(subset, training_strategy, threshold_mode, threshold, eval_domain, layer, group, target_column))
    return pd.DataFrame(rows)


def _policy_row(
    frame: pd.DataFrame,
    training_strategy: str,
    threshold_mode: str,
    threshold: float,
    eval_domain: str,
    layer: str,
    group: dict[str, Any],
    target_column: str,
) -> dict[str, Any]:
    observed = frame[target_column].to_numpy(dtype=float)
    observed_need = observed > 0.0
    decision_run = frame["decision_run_ela"].to_numpy(dtype=bool)
    decision_utility = frame["utility_decision_before_feature"].to_numpy(dtype=float)
    captured_positive = observed[observed_need & decision_run]
    missed_positive = observed[observed_need & ~decision_run]
    unhelpful_call = observed[(~observed_need) & decision_run]
    positive_utility_sum = float(np.sum(observed[observed_need]))
    captured_positive_utility_sum = float(np.sum(captured_positive))
    return {
        "training_strategy": training_strategy,
        "eval_domain": eval_domain,
        "threshold_mode": threshold_mode,
        "threshold": float(threshold),
        "layer": layer,
        "group": _group_label(group),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "problem_id": group.get("problem_id"),
        "rows": int(len(frame)),
        "observed_utility_gt_zero_rows": int(np.sum(observed_need)),
        "observed_utility_gt_zero_rate": float(np.mean(observed_need)),
        "positive_utility_sum": positive_utility_sum,
        "decision_ela_call_count": int(np.sum(decision_run)),
        "decision_ela_call_rate": float(np.mean(decision_run)),
        "true_run_ela_count": int(np.sum(observed_need & decision_run)),
        "missed_run_ela_count": int(np.sum(observed_need & ~decision_run)),
        "unhelpful_run_ela_count": int(np.sum((~observed_need) & decision_run)),
        "skip_when_unhelpful_count": int(np.sum((~observed_need) & ~decision_run)),
        "positive_row_capture_rate": float(np.mean(decision_run[observed_need])) if np.any(observed_need) else 0.0,
        "utility_capture_rate": (
            captured_positive_utility_sum / positive_utility_sum if positive_utility_sum > 0.0 else 0.0
        ),
        "captured_positive_utility_sum": captured_positive_utility_sum,
        "missed_positive_utility_sum": float(np.sum(missed_positive)),
        "unhelpful_call_utility_sum": float(np.sum(unhelpful_call)),
        "unhelpful_call_cost_sum": float(-np.sum(unhelpful_call)),
        "unhelpful_call_cost_mean": float(-np.mean(unhelpful_call)) if unhelpful_call.size else 0.0,
        "decision_utility_sum": float(np.sum(decision_utility)),
        "decision_mean_utility": float(np.mean(decision_utility)),
        "decision_median_utility": float(np.median(decision_utility)),
        "always_ela_mean_utility": float(np.mean(frame["utility_always_ela_traditional_aas"])),
        "best_observed_action_mean_utility": float(np.mean(frame["utility_best_observed_analysis_action"])),
    }


def _utility_distribution(validation: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows = []
    for eval_domain, label_source in EVALUATION_DOMAINS.items():
        frame = validation[validation["label_source"] == label_source]
        for layer, columns in GROUP_LAYERS.items():
            rows.append(_utility_layer(frame, eval_domain, layer, columns, target_column))
    return pd.concat(rows, ignore_index=True)


def _utility_layer(
    frame: pd.DataFrame,
    eval_domain: str,
    layer: str,
    group_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if not group_columns:
        return pd.DataFrame([_utility_row(frame, eval_domain, layer, {}, target_column)])
    rows = []
    for group_values, subset in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group = dict(zip(group_columns, group_values, strict=True))
        rows.append(_utility_row(subset, eval_domain, layer, group, target_column))
    return pd.DataFrame(rows)


def _utility_row(
    frame: pd.DataFrame,
    eval_domain: str,
    layer: str,
    group: dict[str, Any],
    target_column: str,
) -> dict[str, Any]:
    utility = frame[target_column].to_numpy(dtype=float)
    positive = utility > 0.0
    return {
        "eval_domain": eval_domain,
        "layer": layer,
        "group": _group_label(group),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "problem_id": group.get("problem_id"),
        "rows": int(len(frame)),
        "utility_gt_zero_rows": int(np.sum(positive)),
        "utility_gt_zero_rate": float(np.mean(positive)),
        "mean_utility": float(np.mean(utility)),
        "median_utility": float(np.median(utility)),
        "sum_positive_utility": float(np.sum(utility[positive])),
        "mean_positive_utility": float(np.mean(utility[positive])) if np.any(positive) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare changed-algorithm-aware min-support Decision Model training strategies."
    )
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
    parser.add_argument("--changed-weight", type=float, default=3.0)
    parser.add_argument("--same-weight", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/changed_aware_evaluation"),
    )
    args = parser.parse_args()
    run_changed_algorithm_aware_evaluation(
        train_labels_path=args.train_labels,
        validation_labels_path=args.validation_labels,
        output_dir=args.output_dir,
        target_column=args.target_column,
        random_seed=args.random_seed,
        changed_weight=args.changed_weight,
        same_weight=args.same_weight,
    )


if __name__ == "__main__":
    main()
