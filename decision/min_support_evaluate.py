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
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from behavior.features import BEHAVIOR_FEATURE_COLUMNS
from utility_labels.fields import UTILITY_VALUE_COLUMNS
from utility_labels.validation import validate_utility_label_file


DEFAULT_TARGET_COLUMN = "u_ela_lamT_1"
DEFAULT_RANDOM_ANALYSIS_PROBABILITY = 0.5


def train_and_evaluate(
    *,
    train_labels_path: Path,
    validation_labels_path: Path,
    output_summary_path: Path,
    output_predictions_path: Path,
    target_column: str,
    threshold_mode: str,
    random_analysis_probability: float,
    random_seed: int,
    selection_reference_path: Path | None,
) -> dict[str, Any]:
    train = _read_labels(train_labels_path)
    validation = _read_labels(validation_labels_path)
    _check_family_split(train, validation)
    _check_target(target_column)

    feature_columns = list(BEHAVIOR_FEATURE_COLUMNS)
    model = _fit_model(train, feature_columns, target_column, random_seed)
    mean_model = _fit_mean_model(train, feature_columns, target_column)
    threshold = _decision_threshold(
        model=model,
        train=train,
        feature_columns=feature_columns,
        target_column=target_column,
        threshold_mode=threshold_mode,
    )

    started = perf_counter()
    predictions = model.predict(validation[feature_columns])
    prediction_runtime = perf_counter() - started
    mean_predictions = mean_model.predict(validation[feature_columns])
    prediction_runtime_per_state = prediction_runtime / max(len(validation), 1)

    evaluated = _strategy_frame(
        validation=validation,
        predictions=predictions,
        target_column=target_column,
        threshold=threshold,
        random_analysis_probability=random_analysis_probability,
        random_seed=random_seed,
        prediction_runtime_per_state=prediction_runtime_per_state,
    )

    prediction_summary = _prediction_summary(
        observed=validation[target_column].to_numpy(dtype=float),
        predicted=predictions,
        mean_predicted=mean_predictions,
    )
    decision_summary = _decision_summary(
        observed=validation[target_column].to_numpy(dtype=float),
        predicted=predictions,
        threshold=threshold,
    )
    policy_summary = {
        "state_level": _policy_summary(evaluated, group_columns=None),
        "problem_level": _policy_summary(evaluated, group_columns=["problem_id"]),
        "family_level": _policy_summary(evaluated, group_columns=["family"]),
    }
    utility_distribution = _utility_distribution(validation, target_column)
    selection_reference_summary = _selection_reference_summary(selection_reference_path)

    output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(evaluated, preserve_index=False), output_predictions_path)

    summary = {
        "experiment": "min_support_decision_evaluation",
        "research_question": (
            "Can algorithm-agnostic behavior states predict ELA utility and support "
            "resource-aware analysis selection on held-out BBOB function families?"
        ),
        "train_labels": str(train_labels_path),
        "validation_labels": str(validation_labels_path),
        "target_column": target_column,
        "threshold_mode": threshold_mode,
        "decision_threshold": threshold,
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
        "rows": {"train": int(len(train)), "validation": int(len(validation))},
        "families": {
            "train": sorted(train["family"].astype(str).unique().tolist()),
            "validation": sorted(validation["family"].astype(str).unique().tolist()),
        },
        "baselines": [
            "never_ela_sbs",
            "always_ela_traditional_aas",
            "random_analysis",
            "decision_before_feature",
            "best_observed_analysis_action",
        ],
        "vbs_reference": selection_reference_summary,
        "prediction": prediction_summary,
        "decision": decision_summary,
        "utility_distribution": utility_distribution,
        "policy_summary": policy_summary,
        "outputs": {"predictions": str(output_predictions_path), "summary": str(output_summary_path)},
        "data_leakage_check": {
            "family_split_overlap": [],
            "decision_input_uses_only_behavior_features": True,
            "threshold_selected_from_validation": False,
        },
    }

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote min-support decision predictions to {output_predictions_path}")
    print(f"wrote min-support decision summary to {output_summary_path}")
    return summary


def _read_labels(path: Path) -> pd.DataFrame:
    validate_utility_label_file(path)
    return pq.read_table(path).to_pandas()


def _check_target(target_column: str) -> None:
    if target_column not in UTILITY_VALUE_COLUMNS:
        raise ValueError(f"target column must be one of {list(UTILITY_VALUE_COLUMNS)}")


def _check_family_split(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    overlap = sorted(set(train["family"].astype(str)).intersection(validation["family"].astype(str)))
    if overlap:
        raise ValueError(f"train and validation families must be disjoint: {overlap}")


def _fit_model(
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    random_seed: int,
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
    model.fit(train[feature_columns], train[target_column])
    return model


def _fit_mean_model(train: pd.DataFrame, feature_columns: list[str], target_column: str) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", DummyRegressor(strategy="mean")),
        ]
    )
    model.fit(train[feature_columns], train[target_column])
    return model


def _decision_threshold(
    *,
    model: Pipeline,
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    threshold_mode: str,
) -> float:
    if threshold_mode == "zero":
        return 0.0
    if threshold_mode != "train_utility":
        raise ValueError("threshold_mode must be zero or train_utility")

    predictions = model.predict(train[feature_columns])
    candidates = np.unique(np.concatenate(([0.0], predictions)))
    best_threshold = 0.0
    best_utility = -np.inf
    observed = train[target_column].to_numpy(dtype=float)
    for threshold in candidates:
        policy_utility = np.where(predictions > threshold, observed, 0.0)
        mean_utility = float(np.mean(policy_utility))
        if mean_utility > best_utility:
            best_utility = mean_utility
            best_threshold = float(threshold)
    return best_threshold


def _strategy_frame(
    *,
    validation: pd.DataFrame,
    predictions: np.ndarray,
    target_column: str,
    threshold: float,
    random_analysis_probability: float,
    random_seed: int,
    prediction_runtime_per_state: float,
) -> pd.DataFrame:
    if random_analysis_probability < 0.0 or random_analysis_probability > 1.0:
        raise ValueError("random_analysis_probability must be in [0, 1]")

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
    decision_run_ela = predictions > threshold
    random_run_ela = _random_analysis_decisions(len(frame), random_analysis_probability, random_seed)
    best_observed_run_ela = observed > 0.0

    frame["predicted_utility"] = predictions.astype(float)
    frame["decision_threshold"] = float(threshold)
    frame["decision_run_ela"] = decision_run_ela
    frame["random_analysis_run_ela"] = random_run_ela
    frame["best_observed_analysis_run_ela"] = best_observed_run_ela
    frame["utility_never_ela_sbs"] = 0.0
    frame["utility_always_ela_traditional_aas"] = observed
    frame["utility_random_analysis"] = np.where(random_run_ela, observed, 0.0)
    frame["utility_decision_before_feature"] = np.where(decision_run_ela, observed, 0.0)
    frame["utility_best_observed_analysis_action"] = np.maximum(observed, 0.0)
    frame["final_performance_never_ela_sbs"] = frame["p_skip"]
    frame["final_performance_always_ela_traditional_aas"] = frame["p_ela"]
    frame["final_performance_random_analysis"] = np.where(random_run_ela, frame["p_ela"], frame["p_skip"])
    frame["final_performance_decision_before_feature"] = np.where(decision_run_ela, frame["p_ela"], frame["p_skip"])
    frame["final_performance_best_observed_analysis_action"] = np.where(best_observed_run_ela, frame["p_ela"], frame["p_skip"])
    frame["runtime_never_ela_sbs"] = frame["runtime_skip_optimization"]
    frame["runtime_always_ela_traditional_aas"] = (
        frame["runtime_analysis"] + frame["runtime_selection"] + frame["runtime_ela_optimization"]
    )
    frame["runtime_random_analysis"] = np.where(
        random_run_ela,
        frame["runtime_always_ela_traditional_aas"],
        frame["runtime_never_ela_sbs"],
    )
    frame["runtime_decision_before_feature"] = (
        prediction_runtime_per_state
        + np.where(decision_run_ela, frame["runtime_always_ela_traditional_aas"], frame["runtime_never_ela_sbs"])
    )
    frame["runtime_best_observed_analysis_action"] = np.where(
        best_observed_run_ela,
        frame["runtime_always_ela_traditional_aas"],
        frame["runtime_never_ela_sbs"],
    )
    return frame


def _random_analysis_decisions(size: int, probability: float, random_seed: int) -> np.ndarray:
    seed_sequence = np.random.SeedSequence([int(random_seed), 202701, int(size)])
    rng = np.random.default_rng(seed_sequence)
    return rng.random(size) < probability


def _prediction_summary(observed: np.ndarray, predicted: np.ndarray, mean_predicted: np.ndarray) -> dict[str, float | None]:
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
        "r2": _finite_metric(lambda: r2_score(observed, predicted)),
        "spearman": _finite_metric(lambda: pd.Series(observed).corr(pd.Series(predicted), method="spearman")),
        "mean_predictor_mae": float(mean_absolute_error(observed, mean_predicted)),
        "mean_predictor_rmse": float(mean_squared_error(observed, mean_predicted) ** 0.5),
    }


def _decision_summary(observed: np.ndarray, predicted: np.ndarray, threshold: float) -> dict[str, float | None]:
    observed_need = observed > 0.0
    predicted_need = predicted > threshold
    result = {
        "accuracy": float(accuracy_score(observed_need, predicted_need)),
        "f1": float(f1_score(observed_need, predicted_need, zero_division=0)),
        "observed_ela_would_help_rate": float(np.mean(observed_need)),
        "decision_ela_call_rate": float(np.mean(predicted_need)),
    }
    result["auroc"] = _finite_metric(lambda: roc_auc_score(observed_need, predicted))
    return result


def _policy_summary(frame: pd.DataFrame, group_columns: list[str] | None) -> dict[str, dict[str, float]]:
    policies = [
        "never_ela_sbs",
        "always_ela_traditional_aas",
        "random_analysis",
        "decision_before_feature",
        "best_observed_analysis_action",
    ]
    if group_columns is None:
        source = frame
    else:
        source = frame.groupby(group_columns, as_index=False).mean(numeric_only=True)

    result = {}
    for policy in policies:
        utility = source[f"utility_{policy}"].to_numpy(dtype=float)
        performance = source[f"final_performance_{policy}"].to_numpy(dtype=float)
        runtime = source[f"runtime_{policy}"].to_numpy(dtype=float)
        result[policy] = {
            "mean_relative_utility": float(np.mean(utility)),
            "median_relative_utility": float(np.median(utility)),
            "mean_final_performance": float(np.mean(performance)),
            "median_final_performance": float(np.median(performance)),
            "mean_runtime_seconds": float(np.mean(runtime)),
        }
    result["random_analysis"]["ela_call_rate"] = float(np.mean(source["random_analysis_run_ela"]))
    result["decision_before_feature"]["ela_call_rate"] = float(np.mean(source["decision_run_ela"]))
    result["always_ela_traditional_aas"]["ela_call_rate"] = 1.0
    result["never_ela_sbs"]["ela_call_rate"] = 0.0
    result["best_observed_analysis_action"]["ela_call_rate"] = float(np.mean(source["best_observed_analysis_run_ela"]))
    return result


def _utility_distribution(frame: pd.DataFrame, target_column: str) -> dict[str, Any]:
    values = frame[target_column].to_numpy(dtype=float)
    by_family = (
        frame.assign(need_ela=frame[target_column] > 0.0)
        .groupby("family", as_index=False)
        .agg(
            rows=(target_column, "size"),
            mean_utility=(target_column, "mean"),
            median_utility=(target_column, "median"),
            need_ela_rate=("need_ela", "mean"),
        )
    )
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "need_ela_rate": float(np.mean(values > 0.0)),
        "by_family": by_family.to_dict(orient="records"),
    }


def _selection_reference_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "not_provided",
            "note": "Optimizer-level VBS performance is not contained in utility label rows.",
        }
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    frame = pq.read_table(path).to_pandas()
    if frame.empty:
        return {"status": "empty", "path": str(path)}
    return {
        "status": "available",
        "path": str(path),
        "rows": int(len(frame)),
        "default_algorithms": sorted(frame["default_algorithm"].astype(str).unique().tolist()),
        "selected_matches_vbs_rate": float((frame["selected_algorithm"] == frame["vbs_algorithm"]).mean()),
        "note": "VBS is reported as a selection reference upper bound; paired VBS continuation is not generated here.",
    }


def _finite_metric(compute: Any) -> float | None:
    try:
        value = compute()
    except ValueError:
        return None
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the minimum-support Decision Model.")
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--validation-labels", type=Path, required=True)
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--threshold-mode", choices=["zero", "train_utility"], default="zero")
    parser.add_argument("--random-analysis-probability", type=float, default=DEFAULT_RANDOM_ANALYSIS_PROBABILITY)
    parser.add_argument("--random-seed", type=int, default=1701)
    parser.add_argument("--selection-reference", type=Path, default=None)
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("results/decision/min_support/evaluation_summary.json"),
    )
    parser.add_argument(
        "--output-predictions",
        type=Path,
        default=Path("results/decision/min_support/validation_predictions.parquet"),
    )
    args = parser.parse_args()

    train_and_evaluate(
        train_labels_path=args.train_labels,
        validation_labels_path=args.validation_labels,
        output_summary_path=args.output_summary,
        output_predictions_path=args.output_predictions,
        target_column=args.target_column,
        threshold_mode=args.threshold_mode,
        random_analysis_probability=args.random_analysis_probability,
        random_seed=args.random_seed,
        selection_reference_path=args.selection_reference,
    )


if __name__ == "__main__":
    main()
