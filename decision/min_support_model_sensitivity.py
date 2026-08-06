from __future__ import annotations

import argparse
import importlib
import json
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, Ridge, TweedieRegressor
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from behavior.features import BEHAVIOR_FEATURE_COLUMNS
from decision.min_support_diagnostics import GROUP_LAYERS, _group_label
from decision.min_support_evaluate import (
    DEFAULT_TARGET_COLUMN,
    _check_family_split,
    _check_target,
    _json_default,
    _read_labels,
    _strategy_frame,
)
from decision.min_support_changed_algorithm_aware_evaluate import _with_label_source


EVALUATION_DOMAINS: dict[str, str | None] = {
    "all_validation": None,
    "changed_algorithm_validation": "changed_algorithm",
    "same_algorithm_reference": "same_algorithm",
}


def run_model_sensitivity(
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
    model_specs, unavailable_models = _model_specs(random_seed)

    model_rows = []
    threshold_rows = []
    score_rows = []
    policy_frames = []
    prediction_frames = []
    train_prediction_frames = []

    for model_name, spec in model_specs.items():
        fitted = _fit_model_spec(spec, train, feature_columns, target_column)
        train_scores = _decision_scores(fitted, spec, train[feature_columns])
        train_prediction_frames.append(
            _score_output_frame(
                frame=train,
                scores=train_scores,
                model_name=model_name,
                spec=spec,
                target_column=target_column,
                data_split="train",
            )
        )
        thresholds = {
            "zero": 0.0,
            "train_utility": _decision_threshold_from_scores(
                scores=train_scores,
                observed=train[target_column].to_numpy(dtype=float),
            ),
        }
        threshold_rows.extend(
            {
                "model_name": model_name,
                "model_family": spec["model_family"],
                "score_semantics": spec["score_semantics"],
                "threshold_mode": threshold_mode,
                "threshold": float(threshold),
            }
            for threshold_mode, threshold in thresholds.items()
        )

        started = perf_counter()
        validation_scores = _decision_scores(fitted, spec, validation[feature_columns])
        prediction_runtime = perf_counter() - started
        prediction_runtime_per_state = prediction_runtime / max(len(validation), 1)

        model_rows.append(
            {
                "model_name": model_name,
                "model_family": spec["model_family"],
                "score_semantics": spec["score_semantics"],
                "train_score_mean": float(np.mean(train_scores)),
                "validation_score_mean": float(np.mean(validation_scores)),
                "validation_prediction_runtime_seconds": float(prediction_runtime),
                "validation_prediction_runtime_per_state_seconds": float(prediction_runtime_per_state),
            }
        )
        score_rows.extend(
            _score_domain_rows(
                model_name=model_name,
                spec=spec,
                validation=validation,
                scores=validation_scores,
                target_column=target_column,
            )
        )

        for threshold_mode, threshold in thresholds.items():
            evaluated = _strategy_frame(
                validation=validation,
                predictions=validation_scores,
                target_column=target_column,
                threshold=threshold,
                random_analysis_probability=0.5,
                random_seed=random_seed,
                prediction_runtime_per_state=prediction_runtime_per_state,
            )
            evaluated.insert(0, "model_name", model_name)
            evaluated.insert(1, "model_family", spec["model_family"])
            evaluated.insert(2, "score_semantics", spec["score_semantics"])
            evaluated.insert(3, "threshold_mode", threshold_mode)
            evaluated["label_source"] = validation["label_source"].to_numpy()
            evaluated["default_algorithm"] = validation["default_algorithm"].to_numpy()
            evaluated["selected_algorithm"] = validation["selected_algorithm"].to_numpy()
            evaluated["decision_score"] = validation_scores.astype(float)
            prediction_frames.append(evaluated)

            for eval_domain, label_source in EVALUATION_DOMAINS.items():
                domain_frame = evaluated if label_source is None else evaluated[evaluated["label_source"] == label_source]
                for layer, columns in GROUP_LAYERS.items():
                    policy_frames.append(
                        _policy_summary(
                            frame=domain_frame,
                            model_name=model_name,
                            model_family=spec["model_family"],
                            score_semantics=spec["score_semantics"],
                            threshold_mode=threshold_mode,
                            threshold=threshold,
                            eval_domain=eval_domain,
                            layer=layer,
                            group_columns=columns,
                            target_column=target_column,
                        )
                    )

    model_summary = pd.DataFrame(model_rows)
    threshold_summary = pd.DataFrame(threshold_rows)
    score_summary = pd.DataFrame(score_rows)
    policy_summary = pd.concat(policy_frames, ignore_index=True)
    threshold_predictions = pd.concat(prediction_frames, ignore_index=True)
    train_predictions = pd.concat(train_prediction_frames, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model_sensitivity_models.parquet"
    threshold_path = output_dir / "model_sensitivity_thresholds.parquet"
    score_path = output_dir / "model_sensitivity_score_summary.parquet"
    policy_path = output_dir / "model_sensitivity_policy_summary.parquet"
    predictions_path = output_dir / "model_sensitivity_predictions.parquet"
    train_predictions_path = output_dir / "model_sensitivity_train_predictions.parquet"
    summary_path = output_dir / "model_sensitivity_summary.json"
    pq.write_table(pa.Table.from_pandas(model_summary, preserve_index=False), model_path)
    pq.write_table(pa.Table.from_pandas(threshold_summary, preserve_index=False), threshold_path)
    pq.write_table(pa.Table.from_pandas(score_summary, preserve_index=False), score_path)
    pq.write_table(pa.Table.from_pandas(policy_summary, preserve_index=False), policy_path)
    pq.write_table(pa.Table.from_pandas(threshold_predictions, preserve_index=False), predictions_path)
    pq.write_table(pa.Table.from_pandas(train_predictions, preserve_index=False), train_predictions_path)

    summary = {
        "experiment": "min_support_decision_model_sensitivity",
        "research_question": (
            "How much do common supervised model families change min-support Decision Model behavior "
            "under the unchanged utility-label and function-family validation protocol?"
        ),
        "train_labels": str(train_labels_path),
        "validation_labels": str(validation_labels_path),
        "target_column": target_column,
        "rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "train_by_label_source": {
                str(key): int(value) for key, value in train["label_source"].value_counts().sort_index().items()
            },
            "validation_by_label_source": {
                str(key): int(value) for key, value in validation["label_source"].value_counts().sort_index().items()
            },
        },
        "models": {
            name: {
                "model_family": spec["model_family"],
                "score_semantics": spec["score_semantics"],
                "training_target": spec["training_target"],
            }
            for name, spec in model_specs.items()
        },
        "unavailable_models": unavailable_models,
        "thresholds": threshold_summary.to_dict(orient="records"),
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
            "models": str(model_path),
            "thresholds": str(threshold_path),
            "score_summary": str(score_path),
            "policy_summary": str(policy_path),
            "threshold_predictions": str(predictions_path),
            "train_predictions": str(train_predictions_path),
            "summary": str(summary_path),
        },
        "data_leakage_check": {
            "family_split_overlap": [],
            "decision_input_uses_only_behavior_features": True,
            "label_source_used_only_for_evaluation_grouping": True,
            "original_utility_labels_modified": False,
            "threshold_selected_from_validation": False,
        },
        "classification_model_note": (
            "Logistic regression and LDA are trained on the auxiliary label U_ELA > 0 and use decision-function "
            "scores for thresholding; regression models are trained on the original utility value."
        ),
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote model sensitivity model summary to {model_path}")
    print(f"wrote model sensitivity thresholds to {threshold_path}")
    print(f"wrote model sensitivity score summary to {score_path}")
    print(f"wrote model sensitivity policy summary to {policy_path}")
    print(f"wrote model sensitivity predictions to {predictions_path}")
    print(f"wrote model sensitivity train predictions to {train_predictions_path}")
    print(f"wrote model sensitivity summary to {summary_path}")
    return summary


def _model_specs(random_seed: int) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    specs = {
        "random_forest_regression": {
            "model_family": "random_forest",
            "score_semantics": "predicted_utility",
            "training_target": "u_ela_regression",
            "estimator": Pipeline(
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
            ),
        },
        "knn_regression": {
            "model_family": "knn",
            "score_semantics": "predicted_utility",
            "training_target": "u_ela_regression",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("regressor", KNeighborsRegressor(n_neighbors=25, weights="distance")),
                ]
            ),
        },
        "mlp_regression": {
            "model_family": "mlp",
            "score_semantics": "predicted_utility",
            "training_target": "u_ela_regression",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "regressor",
                        MLPRegressor(
                            hidden_layer_sizes=(64, 32),
                            activation="relu",
                            alpha=1e-4,
                            learning_rate_init=1e-3,
                            max_iter=1000,
                            early_stopping=True,
                            validation_fraction=0.15,
                            random_state=random_seed,
                        ),
                    ),
                ]
            ),
        },
        "svm_regression": {
            "model_family": "svm",
            "score_semantics": "predicted_utility",
            "training_target": "u_ela_regression",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("regressor", SVR(C=1.0, epsilon=0.01, gamma="scale")),
                ]
            ),
        },
        "glm_tweedie_regression": {
            "model_family": "glm",
            "score_semantics": "predicted_utility",
            "training_target": "u_ela_regression",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "regressor",
                        TweedieRegressor(
                            power=0.0,
                            alpha=1.0,
                            link="identity",
                            max_iter=2000,
                        ),
                    ),
                ]
            ),
        },
        "rbf_kernel_ridge_regression": {
            "model_family": "kernel_regression",
            "score_semantics": "predicted_utility",
            "training_target": "u_ela_regression",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "kernel_features",
                        Nystroem(
                            kernel="rbf",
                            gamma=0.1,
                            n_components=256,
                            random_state=random_seed,
                        ),
                    ),
                    ("regressor", Ridge(alpha=1.0, random_state=random_seed)),
                ]
            ),
        },
        "logistic_regression_classifier": {
            "model_family": "logistic_regression",
            "score_semantics": "logit_score_for_u_ela_gt_zero",
            "training_target": "utility_gt_zero_classification",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=random_seed,
                        ),
                    ),
                ]
            ),
        },
        "linear_discriminant_analysis_classifier": {
            "model_family": "lda",
            "score_semantics": "lda_score_for_u_ela_gt_zero",
            "training_target": "utility_gt_zero_classification",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("classifier", LinearDiscriminantAnalysis(solver="svd")),
                ]
            ),
        },
    }
    unavailable_models = []
    try:
        lgb = importlib.import_module("lightgbm")
    except Exception as exc:
        unavailable_models.append(
            {"model_name": "lightgbm_regression", "model_family": "lightgbm", "reason": f"{type(exc).__name__}: {exc}"}
        )
    else:
        specs["lightgbm_regression"] = {
            "model_family": "lightgbm",
            "score_semantics": "predicted_utility",
            "training_target": "u_ela_regression",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "regressor",
                        lgb.LGBMRegressor(
                            n_estimators=300,
                            learning_rate=0.05,
                            num_leaves=31,
                            min_child_samples=5,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            random_state=random_seed,
                            n_jobs=-1,
                            verbosity=-1,
                        ),
                    ),
                ]
            ),
        }

    try:
        xgb = importlib.import_module("xgboost")
    except Exception as exc:
        unavailable_models.append(
            {"model_name": "xgboost_regression", "model_family": "xgboost", "reason": f"{type(exc).__name__}: {exc}"}
        )
    else:
        specs["xgboost_regression"] = {
            "model_family": "xgboost",
            "score_semantics": "predicted_utility",
            "training_target": "u_ela_regression",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "regressor",
                        xgb.XGBRegressor(
                            n_estimators=300,
                            max_depth=3,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            objective="reg:squarederror",
                            random_state=random_seed,
                            n_jobs=-1,
                            tree_method="hist",
                            verbosity=0,
                        ),
                    ),
                ]
            ),
        }
    return specs, unavailable_models


def _fit_model_spec(
    spec: dict[str, Any],
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> Pipeline:
    estimator = spec["estimator"]
    if spec["training_target"] == "utility_gt_zero_classification":
        target = train[target_column].to_numpy(dtype=float) > 0.0
    else:
        target = train[target_column]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(train[feature_columns], target)
    return estimator


def _decision_scores(model: Pipeline, spec: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    if spec["training_target"] == "utility_gt_zero_classification":
        classifier = model.named_steps["classifier"]
        transformed = model[:-1].transform(features)
        return classifier.decision_function(transformed).astype(float)
    return model.predict(features).astype(float)


def _decision_threshold_from_scores(scores: np.ndarray, observed: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0], scores)))
    best_threshold = 0.0
    best_utility = -np.inf
    for threshold in candidates:
        policy_utility = np.where(scores > threshold, observed, 0.0)
        mean_utility = float(np.mean(policy_utility))
        if mean_utility > best_utility:
            best_utility = mean_utility
            best_threshold = float(threshold)
    return best_threshold


def _score_output_frame(
    *,
    frame: pd.DataFrame,
    scores: np.ndarray,
    model_name: str,
    spec: dict[str, Any],
    target_column: str,
    data_split: str,
) -> pd.DataFrame:
    columns = [
        "split",
        "problem_id",
        "family",
        "dimension",
        "prefix_algorithm",
        "seed",
        "FE",
        "FE_ratio",
        "default_algorithm",
        "selected_algorithm",
        "label_source",
        target_column,
    ]
    result = frame[columns].copy()
    result.insert(0, "data_split", data_split)
    result.insert(1, "model_name", model_name)
    result.insert(2, "model_family", spec["model_family"])
    result.insert(3, "score_semantics", spec["score_semantics"])
    result.insert(4, "training_target", spec["training_target"])
    result["decision_score"] = scores.astype(float)
    return result


def _score_domain_rows(
    *,
    model_name: str,
    spec: dict[str, Any],
    validation: pd.DataFrame,
    scores: np.ndarray,
    target_column: str,
) -> list[dict[str, Any]]:
    rows = []
    for eval_domain, label_source in EVALUATION_DOMAINS.items():
        mask = np.ones(len(validation), dtype=bool) if label_source is None else (
            validation["label_source"].to_numpy() == label_source
        )
        observed = validation.loc[mask, target_column].to_numpy(dtype=float)
        domain_scores = scores[mask]
        observed_need = observed > 0.0
        row = {
            "model_name": model_name,
            "model_family": spec["model_family"],
            "score_semantics": spec["score_semantics"],
            "eval_domain": eval_domain,
            "rows": int(len(observed)),
            "utility_gt_zero_rate": float(np.mean(observed_need)),
            "score_mean": float(np.mean(domain_scores)),
            "score_median": float(np.median(domain_scores)),
            "score_spearman_with_utility": _finite_metric(
                lambda: pd.Series(observed).corr(pd.Series(domain_scores), method="spearman")
            ),
            "score_auroc_for_utility_gt_zero": _finite_metric(lambda: roc_auc_score(observed_need, domain_scores)),
            "score_zero_threshold_f1_for_utility_gt_zero": float(
                f1_score(observed_need, domain_scores > 0.0, zero_division=0)
            ),
        }
        if spec["score_semantics"] == "predicted_utility":
            row.update(
                {
                    "mae": float(mean_absolute_error(observed, domain_scores)),
                    "rmse": float(mean_squared_error(observed, domain_scores) ** 0.5),
                    "r2": _finite_metric(lambda: r2_score(observed, domain_scores)),
                }
            )
        else:
            row.update({"mae": None, "rmse": None, "r2": None})
        rows.append(row)
    return rows


def _policy_summary(
    *,
    frame: pd.DataFrame,
    model_name: str,
    model_family: str,
    score_semantics: str,
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
            [
                _policy_row(
                    frame,
                    model_name,
                    model_family,
                    score_semantics,
                    threshold_mode,
                    threshold,
                    eval_domain,
                    layer,
                    {},
                    target_column,
                )
            ]
        )
    rows = []
    for group_values, subset in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group = dict(zip(group_columns, group_values, strict=True))
        rows.append(
            _policy_row(
                subset,
                model_name,
                model_family,
                score_semantics,
                threshold_mode,
                threshold,
                eval_domain,
                layer,
                group,
                target_column,
            )
        )
    return pd.DataFrame(rows)


def _policy_row(
    frame: pd.DataFrame,
    model_name: str,
    model_family: str,
    score_semantics: str,
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
        "model_name": model_name,
        "model_family": model_family,
        "score_semantics": score_semantics,
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


def _finite_metric(compute: Any) -> float | None:
    try:
        value = compute()
    except ValueError:
        return None
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare common Decision Model families on min-support labels.")
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
        default=Path("results/decision/min_support/model_sensitivity"),
    )
    args = parser.parse_args()
    run_model_sensitivity(
        train_labels_path=args.train_labels,
        validation_labels_path=args.validation_labels,
        output_dir=args.output_dir,
        target_column=args.target_column,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
