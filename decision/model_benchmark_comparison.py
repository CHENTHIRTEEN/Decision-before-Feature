from __future__ import annotations

import argparse
import importlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import BayesianRidge, ElasticNet, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR

from behavior.features import BEHAVIOR_FEATURE_GROUPS
from decision.query_contract import decision_query_root, validate_query_frame, validate_query_payload
from decision.train_full_decision_model import (
    AUXILIARY_LABEL_COLUMN,
    METADATA_COLUMNS,
    TARGET_COLUMN,
    TOP_K_FRACTIONS,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    _check_dataset,
    _check_family_split,
    _feature_columns,
)
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS, get_query_spec


THRESHOLD_TOP_FRACTIONS = (0.05, 0.10, 0.20)
EPS = 1e-12


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    model_family: str
    estimator_name: str
    estimator: BaseEstimator
    objective: str


def run_model_benchmark_comparison(
    *,
    query_id: str,
    dataset_path: Path,
    schema_path: Path,
    output_dir: Path,
    feature_group: str,
    random_seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    _check_output_paths(output_dir, overwrite)
    dataset = pq.read_table(dataset_path).to_pandas()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validate_query_payload(schema, query_id=query_id, artifact="Decision schema")
    validate_query_frame(dataset, query_id=query_id, artifact="Decision dataset")
    feature_columns = _feature_columns(schema, feature_group)
    _check_dataset(dataset, feature_columns)

    train = dataset[dataset["split"] == TRAIN_SPLIT].copy()
    validation = dataset[dataset["split"] == VALIDATION_SPLIT].copy()
    _check_family_split(train, validation)

    model_specs, unavailable_models = _model_specs(random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    split_metric_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []
    ranking_frames: list[pd.DataFrame] = []
    train_prediction_frames: list[pd.DataFrame] = []
    validation_prediction_frames: list[pd.DataFrame] = []

    for spec in model_specs:
        train_target = _target_for_objective(train, spec.objective)
        started = perf_counter()
        fitted = _fit_estimator(spec.estimator, train, feature_columns, train_target)
        fit_seconds = perf_counter() - started

        started = perf_counter()
        train_scores = _decision_scores(fitted, train, feature_columns)
        train_prediction_seconds = perf_counter() - started
        started = perf_counter()
        validation_scores = _decision_scores(fitted, validation, feature_columns)
        validation_prediction_seconds = perf_counter() - started

        thresholds = _thresholds_from_train_scores(train_scores, train[TARGET_COLUMN].to_numpy(dtype=float))
        threshold_rows.extend(
            {
                "model_name": spec.model_name,
                "model_family": spec.model_family,
                "objective": spec.objective,
                "threshold_mode": threshold_mode,
                "threshold": float(threshold),
                "fit_split": TRAIN_SPLIT,
                "validation_rows_used_for_threshold_fit": 0,
            }
            for threshold_mode, threshold in thresholds.items()
        )
        fit_rows.append(
            {
                "model_name": spec.model_name,
                "model_family": spec.model_family,
                "objective": spec.objective,
                "estimator": spec.estimator_name,
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "fit_seconds": float(fit_seconds),
                "train_prediction_seconds": float(train_prediction_seconds),
                "validation_prediction_seconds": float(validation_prediction_seconds),
                "validation_prediction_seconds_per_row": float(validation_prediction_seconds / max(len(validation), 1)),
                "train_score_mean": float(np.mean(train_scores)),
                "validation_score_mean": float(np.mean(validation_scores)),
            }
        )

        train_predictions = _prediction_frame(
            frame=train,
            scores=train_scores,
            thresholds=thresholds,
            model_name=spec.model_name,
            model_family=spec.model_family,
            objective=spec.objective,
            eval_split=TRAIN_SPLIT,
        )
        validation_predictions = _prediction_frame(
            frame=validation,
            scores=validation_scores,
            thresholds=thresholds,
            model_name=spec.model_name,
            model_family=spec.model_family,
            objective=spec.objective,
            eval_split=VALIDATION_SPLIT,
        )
        split_metric_frames.append(_split_metric_summary(train_predictions, spec.objective))
        split_metric_frames.append(_split_metric_summary(validation_predictions, spec.objective))
        decision_frames.append(_decision_summary(train_predictions, thresholds))
        decision_frames.append(_decision_summary(validation_predictions, thresholds))
        ranking_frames.append(_ranking_summary(train_predictions))
        ranking_frames.append(_ranking_summary(validation_predictions))
        train_prediction_frames.append(train_predictions)
        validation_prediction_frames.append(validation_predictions)

    model_fit_summary = pd.DataFrame(fit_rows)
    threshold_summary = pd.DataFrame(threshold_rows)
    train_predictions = pd.concat(train_prediction_frames, ignore_index=True)
    validation_predictions = pd.concat(validation_prediction_frames, ignore_index=True)
    split_metric_summary = pd.concat(split_metric_frames, ignore_index=True)
    split_decision_summary = pd.concat(decision_frames, ignore_index=True)
    split_ranking_summary = pd.concat(ranking_frames, ignore_index=True)
    best_summary = _best_summary(split_metric_summary, split_decision_summary, split_ranking_summary)
    input_contract = _input_contract(feature_columns, train)

    _write_frame(input_contract, output_dir / "model_input_contract")
    _write_frame(model_fit_summary, output_dir / "model_fit_summary")
    _write_frame(threshold_summary, output_dir / "decision_thresholds")
    _write_parquet(train_predictions, output_dir / "train_predictions.parquet")
    _write_parquet(validation_predictions, output_dir / "validation_predictions.parquet")
    _write_frame(split_metric_summary, output_dir / "split_metric_summary")
    _write_frame(split_decision_summary, output_dir / "split_decision_summary")
    _write_frame(split_ranking_summary, output_dir / "split_ranking_summary")
    _write_frame(best_summary, output_dir / "model_benchmark_best_summary")

    summary = {
        "experiment": "phase1_refined_sampling_model_benchmark_comparison",
        "query_id": query_id,
        "query_protocol": get_query_spec(query_id).protocol,
        "sample_design_id": get_query_spec(query_id).sample_design_id,
        "dataset": str(dataset_path),
        "schema": str(schema_path),
        "target_column": TARGET_COLUMN,
        "auxiliary_label_column": AUXILIARY_LABEL_COLUMN,
        "feature_group": feature_group,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "train_split": TRAIN_SPLIT,
        "validation_split": VALIDATION_SPLIT,
        "rows": {"train": int(len(train)), "validation": int(len(validation))},
        "models_trained": [spec.model_name for spec in model_specs],
        "unavailable_models": unavailable_models,
        "threshold_modes": ["zero", "train_utility", "top_05", "top_10", "top_20"],
        "top_k_fractions": list(TOP_K_FRACTIONS),
        "classification_score_meaning": "classification models use probability or decision-function scores for ranking and threshold decisions",
        "outputs": {
            "model_input_contract": str(output_dir / "model_input_contract.parquet"),
            "model_fit_summary": str(output_dir / "model_fit_summary.parquet"),
            "decision_thresholds": str(output_dir / "decision_thresholds.parquet"),
            "train_predictions": str(output_dir / "train_predictions.parquet"),
            "validation_predictions": str(output_dir / "validation_predictions.parquet"),
            "split_metric_summary": str(output_dir / "split_metric_summary.parquet"),
            "split_decision_summary": str(output_dir / "split_decision_summary.parquet"),
            "split_ranking_summary": str(output_dir / "split_ranking_summary.parquet"),
            "best_summary": str(output_dir / "model_benchmark_best_summary.parquet"),
            "report": str(output_dir / "model_benchmark_comparison_report.md"),
            "summary": str(output_dir / "model_benchmark_comparison_summary.json"),
        },
        "data_leakage_check": {
            "decision_input_uses_only_behavior_features": True,
            "metadata_used_as_input": False,
            "function_id_algorithm_id_or_optimizer_internal_parameters_used_as_input": False,
            "query_features_used_as_input": False,
            "validation_rows_used_for_imputer_scaler_model_or_threshold_fit": 0,
        },
        "summary_layers": ["all", "label_source", "dimension", "FE_ratio"],
    }
    summary_path = output_dir / "model_benchmark_comparison_summary.json"
    report_path = output_dir / "model_benchmark_comparison_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(
        _markdown_report(
            summary=summary,
            input_contract=input_contract,
            model_fit_summary=model_fit_summary,
            split_metric_summary=split_metric_summary,
            split_decision_summary=split_decision_summary,
            split_ranking_summary=split_ranking_summary,
            best_summary=best_summary,
        ),
        encoding="utf-8",
    )
    print(f"wrote model benchmark summary to {summary_path}")
    print(f"wrote model benchmark report to {report_path}")
    return summary


def write_report_from_existing_outputs(*, output_dir: Path) -> Path:
    summary_path = output_dir / "model_benchmark_comparison_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    input_contract = pd.read_csv(output_dir / "model_input_contract.csv")
    model_fit_summary = pd.read_csv(output_dir / "model_fit_summary.csv")
    split_metric_summary = pd.read_csv(output_dir / "split_metric_summary.csv")
    split_decision_summary = pd.read_csv(output_dir / "split_decision_summary.csv")
    split_ranking_summary = pd.read_csv(output_dir / "split_ranking_summary.csv")
    best_summary = pd.read_csv(output_dir / "model_benchmark_best_summary.csv")
    report_path = output_dir / "model_benchmark_comparison_report.md"
    report_path.write_text(
        _markdown_report(
            summary=summary,
            input_contract=input_contract,
            model_fit_summary=model_fit_summary,
            split_metric_summary=split_metric_summary,
            split_decision_summary=split_decision_summary,
            split_ranking_summary=split_ranking_summary,
            best_summary=best_summary,
        ),
        encoding="utf-8",
    )
    print(f"wrote model benchmark report to {report_path}")
    return report_path


def _check_output_paths(output_dir: Path, overwrite: bool) -> None:
    outputs = (
        output_dir / "model_input_contract.csv",
        output_dir / "model_input_contract.parquet",
        output_dir / "model_fit_summary.csv",
        output_dir / "model_fit_summary.parquet",
        output_dir / "decision_thresholds.csv",
        output_dir / "decision_thresholds.parquet",
        output_dir / "train_predictions.parquet",
        output_dir / "validation_predictions.parquet",
        output_dir / "split_metric_summary.csv",
        output_dir / "split_metric_summary.parquet",
        output_dir / "split_decision_summary.csv",
        output_dir / "split_decision_summary.parquet",
        output_dir / "split_ranking_summary.csv",
        output_dir / "split_ranking_summary.parquet",
        output_dir / "model_benchmark_best_summary.csv",
        output_dir / "model_benchmark_best_summary.parquet",
        output_dir / "model_benchmark_comparison_report.md",
        output_dir / "model_benchmark_comparison_summary.json",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"model benchmark outputs already exist; pass --overwrite: {existing[0]}")


def _model_specs(random_seed: int) -> tuple[list[ModelSpec], list[dict[str, str]]]:
    specs = [
        ModelSpec(
            "dummy_mean_regression",
            "dummy",
            "DummyRegressor(strategy='mean')",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("regressor", DummyRegressor(strategy="mean"))]),
            "regression",
        ),
        ModelSpec(
            "linear_regression",
            "linear",
            "LinearRegression()",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("regressor", LinearRegression()),
                ]
            ),
            "regression",
        ),
        ModelSpec(
            "ridge_regression",
            "ridge",
            "Ridge(alpha=1.0)",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("regressor", Ridge(alpha=1.0, random_state=random_seed)),
                ]
            ),
            "regression",
        ),
        ModelSpec(
            "elastic_net_regression",
            "elastic_net",
            "ElasticNet(alpha=0.0005,l1_ratio=0.15)",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("regressor", ElasticNet(alpha=0.0005, l1_ratio=0.15, max_iter=5000, random_state=random_seed)),
                ]
            ),
            "regression",
        ),
        ModelSpec(
            "bayesian_ridge_regression",
            "bayesian_ridge",
            "BayesianRidge()",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("regressor", BayesianRidge()),
                ]
            ),
            "regression",
        ),
        ModelSpec(
            "linear_svm_regression",
            "linear_svm",
            "LinearSVR(C=1.0,epsilon=0.0)",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("regressor", LinearSVR(C=1.0, epsilon=0.0, max_iter=5000, random_state=random_seed)),
                ]
            ),
            "regression",
        ),
        ModelSpec(
            "rbf_nystroem_ridge_regression",
            "kernel_ridge_approx",
            "Nystroem(rbf,n_components=256)+Ridge(alpha=1.0)",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("kernel", Nystroem(kernel="rbf", gamma=0.2, n_components=256, random_state=random_seed)),
                    ("regressor", Ridge(alpha=1.0, random_state=random_seed)),
                ]
            ),
            "regression",
        ),
        ModelSpec(
            "rbf_nystroem_svm_regression",
            "kernel_svm_approx",
            "Nystroem(rbf,n_components=256)+LinearSVR(C=1.0)",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("kernel", Nystroem(kernel="rbf", gamma=0.2, n_components=256, random_state=random_seed)),
                    ("regressor", LinearSVR(C=1.0, epsilon=0.0, max_iter=5000, random_state=random_seed)),
                ]
            ),
            "regression",
        ),
        ModelSpec(
            "hist_gradient_boosting_regression",
            "hist_gradient_boosting",
            "HistGradientBoostingRegressor(max_iter=300)",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "regressor",
                        HistGradientBoostingRegressor(
                            max_iter=300,
                            learning_rate=0.05,
                            l2_regularization=0.01,
                            random_state=random_seed,
                        ),
                    ),
                ]
            ),
            "regression",
        ),
        ModelSpec(
            "random_forest_regression",
            "random_forest",
            "RandomForestRegressor(n_estimators=300,min_samples_leaf=3)",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "regressor",
                        RandomForestRegressor(
                            n_estimators=300,
                            min_samples_leaf=3,
                            random_state=random_seed,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "regression",
        ),
        ModelSpec(
            "extra_trees_regression",
            "extra_trees",
            "ExtraTreesRegressor(n_estimators=300,min_samples_leaf=3)",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "regressor",
                        ExtraTreesRegressor(
                            n_estimators=300,
                            min_samples_leaf=3,
                            random_state=random_seed,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "regression",
        ),
        ModelSpec(
            "mlp_regression",
            "mlp",
            "MLPRegressor(hidden_layer_sizes=(64,32),early_stopping=True)",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "regressor",
                        MLPRegressor(
                            hidden_layer_sizes=(64, 32),
                            activation="relu",
                            alpha=0.001,
                            batch_size=2048,
                            learning_rate_init=0.001,
                            max_iter=80,
                            early_stopping=True,
                            n_iter_no_change=8,
                            random_state=random_seed,
                        ),
                    ),
                ]
            ),
            "regression",
        ),
        ModelSpec(
            "softmax_logistic_classifier",
            "softmax_logistic",
            "LogisticRegression(class_weight='balanced')",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=random_seed,
                        ),
                    ),
                ]
            ),
            "classification",
        ),
        ModelSpec(
            "lda_classifier",
            "lda",
            "LinearDiscriminantAnalysis()",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("classifier", LinearDiscriminantAnalysis()),
                ]
            ),
            "classification",
        ),
        ModelSpec(
            "linear_svm_classifier",
            "linear_svm",
            "LinearSVC(class_weight='balanced')",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=random_seed),
                    ),
                ]
            ),
            "classification",
        ),
        ModelSpec(
            "mlp_classifier",
            "mlp_classifier",
            "MLPClassifier(hidden_layer_sizes=(64,32),early_stopping=True)",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        MLPClassifier(
                            hidden_layer_sizes=(64, 32),
                            activation="relu",
                            alpha=0.001,
                            batch_size=2048,
                            learning_rate_init=0.001,
                            max_iter=80,
                            early_stopping=True,
                            n_iter_no_change=8,
                            random_state=random_seed,
                        ),
                    ),
                ]
            ),
            "classification",
        ),
    ]
    unavailable: list[dict[str, str]] = []
    try:
        lgb = importlib.import_module("lightgbm")
    except Exception as exc:
        unavailable.append(
            {"model_name": "lightgbm_regression", "model_family": "lightgbm", "reason": f"{type(exc).__name__}: {exc}"}
        )
    else:
        specs.append(
            ModelSpec(
                "lightgbm_regression",
                "lightgbm",
                "LGBMRegressor(n_estimators=300,num_leaves=31)",
                Pipeline(
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
                "regression",
            )
        )

    try:
        xgb = importlib.import_module("xgboost")
    except Exception as exc:
        unavailable.append(
            {"model_name": "xgboost_regression", "model_family": "xgboost", "reason": f"{type(exc).__name__}: {exc}"}
        )
    else:
        specs.append(
            ModelSpec(
                "xgboost_regression",
                "xgboost",
                "XGBRegressor(n_estimators=300,max_depth=3)",
                Pipeline(
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
                "regression",
            )
        )
    return specs, unavailable


def _fit_estimator(
    estimator: BaseEstimator,
    train: pd.DataFrame,
    feature_columns: list[str],
    target: np.ndarray,
) -> BaseEstimator:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(train[feature_columns], target)
    return estimator


def _target_for_objective(frame: pd.DataFrame, objective: str) -> np.ndarray:
    if objective == "classification":
        return frame[AUXILIARY_LABEL_COLUMN].to_numpy(dtype=bool).astype(int)
    return frame[TARGET_COLUMN].to_numpy(dtype=float)


def _decision_scores(model: BaseEstimator, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(frame[feature_columns])
        if scores.ndim == 2 and scores.shape[1] >= 2:
            output = scores[:, 1]
        else:
            output = np.asarray(scores).reshape(-1)
    elif hasattr(model, "decision_function"):
        output = model.decision_function(frame[feature_columns])
    else:
        output = model.predict(frame[feature_columns])
    output = np.asarray(output, dtype=float).reshape(-1)
    if not np.isfinite(output).all():
        raise ValueError("model produced non-finite decision scores")
    return output


def _thresholds_from_train_scores(scores: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    thresholds = {
        "zero": 0.0,
        "train_utility": _best_utility_threshold(scores, observed),
    }
    for fraction in THRESHOLD_TOP_FRACTIONS:
        thresholds[f"top_{int(fraction * 100):02d}"] = _top_fraction_threshold(scores, fraction)
    return thresholds


def _best_utility_threshold(scores: np.ndarray, observed: np.ndarray) -> float:
    frame = pd.DataFrame({"score": scores.astype(float), "observed": observed.astype(float)})
    grouped = frame.groupby("score", as_index=True)["observed"].sum().sort_index()
    cumulative_leq = grouped.cumsum()
    total = float(grouped.sum())
    threshold_utility = total - cumulative_leq
    threshold_utility.loc[0.0] = float(frame.loc[frame["score"] > 0.0, "observed"].sum())
    return float(threshold_utility.idxmax())


def _top_fraction_threshold(scores: np.ndarray, fraction: float) -> float:
    if not 0.0 < fraction < 1.0:
        raise ValueError("top fraction threshold must be between 0 and 1")
    return float(np.quantile(scores.astype(float), 1.0 - fraction, method="higher"))


def _prediction_frame(
    *,
    frame: pd.DataFrame,
    scores: np.ndarray,
    thresholds: dict[str, float],
    model_name: str,
    model_family: str,
    objective: str,
    eval_split: str,
) -> pd.DataFrame:
    output = frame[list(METADATA_COLUMNS) + [TARGET_COLUMN, AUXILIARY_LABEL_COLUMN]].copy()
    output.insert(0, "eval_split", eval_split)
    output.insert(1, "model_name", model_name)
    output.insert(2, "model_family", model_family)
    output.insert(3, "objective", objective)
    output["decision_score"] = scores.astype(float)
    for threshold_mode, threshold in thresholds.items():
        calls = scores > threshold
        output[f"decision_run_query_{threshold_mode}"] = calls
        output[f"decision_utility_{threshold_mode}"] = np.where(calls, output[TARGET_COLUMN], 0.0)
    return output


def _split_metric_summary(frame: pd.DataFrame, objective: str) -> pd.DataFrame:
    rows = []
    for layer_frame, layer, group in _iter_layers(frame):
        rows.append(_split_metric_row(layer_frame, objective, layer, group))
    return pd.DataFrame(rows)


def _split_metric_row(
    frame: pd.DataFrame,
    objective: str,
    layer: str,
    group: dict[str, Any],
) -> dict[str, Any]:
    observed = frame[TARGET_COLUMN].to_numpy(dtype=float)
    predicted = frame["decision_score"].to_numpy(dtype=float)
    y_binary = observed > 0.0
    return {
        **_common_fields(frame, layer, group),
        "objective": objective,
        "mae": float(mean_absolute_error(observed, predicted)) if objective == "regression" else None,
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5) if objective == "regression" else None,
        "r2": _finite_metric(lambda: r2_score(observed, predicted)) if objective == "regression" else None,
        "pearson": _finite_metric(lambda: pd.Series(observed).corr(pd.Series(predicted), method="pearson")),
        "spearman": _finite_metric(lambda: pd.Series(observed).corr(pd.Series(predicted), method="spearman")),
        "roc_auc_u_gt_zero": _finite_binary_metric(lambda: roc_auc_score(y_binary, predicted), y_binary),
        "average_precision_u_gt_zero": _finite_binary_metric(lambda: average_precision_score(y_binary, predicted), y_binary),
        "score_mean": float(np.mean(predicted)),
        "score_median": float(np.median(predicted)),
    }


def _decision_summary(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    for layer_frame, layer, group in _iter_layers(frame):
        rows.extend(_decision_row(layer_frame, mode, threshold, layer, group) for mode, threshold in thresholds.items())
    return pd.DataFrame(rows)


def _decision_row(
    frame: pd.DataFrame,
    threshold_mode: str,
    threshold: float,
    layer: str,
    group: dict[str, Any],
) -> dict[str, Any]:
    observed = frame[TARGET_COLUMN].to_numpy(dtype=float)
    positive = observed > 0.0
    calls = frame[f"decision_run_query_{threshold_mode}"].to_numpy(dtype=bool)
    decision_utility = np.where(calls, observed, 0.0)
    captured_positive = positive & calls
    unhelpful_calls = (~positive) & calls
    positive_rows = int(np.sum(positive))
    positive_utility_sum = float(np.sum(observed[positive]))
    captured_positive_utility_sum = float(np.sum(observed[captured_positive]))
    call_rows = int(np.sum(calls))
    return {
        **_common_fields(frame, layer, group),
        "threshold_mode": threshold_mode,
        "threshold": float(threshold),
        "decision_query_call_rows": call_rows,
        "decision_query_call_rate": float(np.mean(calls)),
        "mean_observed_utility_under_calls": float(np.mean(observed[calls])) if call_rows else 0.0,
        "positive_rows_captured": int(np.sum(captured_positive)),
        "positive_row_capture_rate": float(np.sum(captured_positive) / max(positive_rows, 1)),
        "utility_capture_rate": (
            captured_positive_utility_sum / positive_utility_sum if positive_utility_sum > 0.0 else 0.0
        ),
        "precision_u_gt_zero_under_calls": float(np.sum(captured_positive) / max(call_rows, 1)),
        "unhelpful_call_rows": int(np.sum(unhelpful_calls)),
        "unhelpful_call_rate_within_calls": float(np.sum(unhelpful_calls) / max(call_rows, 1)),
        "unhelpful_call_cost_sum": float(-np.sum(observed[unhelpful_calls])),
        "decision_utility_sum": float(np.sum(decision_utility)),
        "decision_mean_utility": float(np.mean(decision_utility)),
        "always_query_mean_utility": float(np.mean(observed)),
        "never_query_mean_utility": 0.0,
        "best_observed_action_mean_utility": float(np.mean(np.maximum(observed, 0.0))),
    }


def _ranking_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer_frame, layer, group in _iter_layers(frame):
        rows.extend(_ranking_row(layer_frame, fraction, layer, group) for fraction in TOP_K_FRACTIONS)
    return pd.DataFrame(rows)


def _ranking_row(
    frame: pd.DataFrame,
    fraction: float,
    layer: str,
    group: dict[str, Any],
) -> dict[str, Any]:
    ranked = frame.sort_values("decision_score", ascending=False)
    top_k_rows = max(1, int(np.ceil(len(ranked) * fraction)))
    top = ranked.head(top_k_rows)
    observed = ranked[TARGET_COLUMN].to_numpy(dtype=float)
    top_observed = top[TARGET_COLUMN].to_numpy(dtype=float)
    positive = observed > 0.0
    top_positive = top_observed > 0.0
    positive_rows = int(np.sum(positive))
    positive_utility_sum = float(np.sum(observed[positive]))
    captured_positive_utility_sum = float(np.sum(top_observed[top_positive]))
    base_rate = float(np.mean(positive))
    top_rate = float(np.mean(top_positive))
    return {
        **_common_fields(frame, layer, group),
        "top_k_fraction": float(fraction),
        "top_k_rows": int(top_k_rows),
        "top_k_row_share": float(top_k_rows / max(len(ranked), 1)),
        "top_k_u_gt_zero_rate": top_rate,
        "lift_vs_base_rate": top_rate / base_rate if base_rate > 0.0 else None,
        "positive_rows_captured": int(np.sum(top_positive)),
        "positive_row_capture_rate": float(np.sum(top_positive) / max(positive_rows, 1)),
        "utility_capture_rate": (
            captured_positive_utility_sum / positive_utility_sum if positive_utility_sum > 0.0 else 0.0
        ),
        "top_k_mean_observed_utility": float(np.mean(top_observed)),
        "top_k_positive_utility_sum": captured_positive_utility_sum,
    }


def _iter_layers(frame: pd.DataFrame) -> list[tuple[pd.DataFrame, str, dict[str, Any]]]:
    layers: list[tuple[pd.DataFrame, str, dict[str, Any]]] = [(frame, "all", {})]
    for label_source, group in frame.groupby("label_source", dropna=False):
        layers.append((group, "label_source", {"label_source": _group_value(label_source)}))
    for dimension, group in frame.groupby("dimension", dropna=False):
        layers.append((group, "dimension", {"dimension": _group_value(dimension)}))
    for fe_ratio, group in frame.groupby("FE_ratio", dropna=False):
        layers.append((group, "FE_ratio", {"FE_ratio": _group_value(fe_ratio)}))
    return layers


def _group_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)


def _group_label(group: dict[str, Any]) -> str:
    if not group:
        return "all"
    return "|".join(f"{key}={value}" for key, value in group.items())


def _common_fields(frame: pd.DataFrame, layer: str, group: dict[str, Any]) -> dict[str, Any]:
    observed = frame[TARGET_COLUMN].to_numpy(dtype=float)
    positive = observed > 0.0
    return {
        "model_name": frame["model_name"].iloc[0],
        "model_family": frame["model_family"].iloc[0],
        "objective": frame["objective"].iloc[0],
        "eval_split": frame["eval_split"].iloc[0],
        "layer": layer,
        "group": _group_label(group),
        "label_source": group.get("label_source"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "rows": int(len(frame)),
        "u_gt_zero_rows": int(np.sum(positive)),
        "u_gt_zero_rate": float(np.mean(positive)),
        "mean_observed_utility": float(np.mean(observed)),
        "median_observed_utility": float(np.median(observed)),
        "positive_utility_sum": float(np.sum(observed[positive])),
    }


def _best_summary(
    metric: pd.DataFrame,
    decision: pd.DataFrame,
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    validation_metric = metric[(metric["eval_split"] == VALIDATION_SPLIT) & (metric["layer"] == "all")]
    validation_decision = decision[(decision["eval_split"] == VALIDATION_SPLIT) & (decision["layer"] == "all")]
    validation_ranking = ranking[(ranking["eval_split"] == VALIDATION_SPLIT) & (ranking["layer"] == "all")]
    rows.append(_best_row(validation_metric, "lowest_validation_rmse_regression_models", "rmse", ascending=True))
    rows.append(_best_row(validation_metric, "highest_validation_spearman", "spearman", ascending=False))
    rows.append(_best_row(validation_metric, "highest_validation_average_precision", "average_precision_u_gt_zero", ascending=False))
    rows.append(
        _best_row(
            validation_decision[validation_decision["threshold_mode"] == "train_utility"],
            "highest_train_utility_threshold_decision_mean_utility",
            "decision_mean_utility",
            ascending=False,
        )
    )
    rows.append(
        _best_row(
            validation_decision[validation_decision["threshold_mode"] == "top_10"],
            "highest_top10_threshold_decision_mean_utility",
            "decision_mean_utility",
            ascending=False,
        )
    )
    rows.append(
        _best_row(
            validation_ranking[np.isclose(validation_ranking["top_k_fraction"], 0.10)],
            "highest_top10_ranking_utility_capture_rate",
            "utility_capture_rate",
            ascending=False,
        )
    )
    return pd.DataFrame(rows)


def _best_row(frame: pd.DataFrame, criterion: str, metric: str, *, ascending: bool) -> dict[str, Any]:
    values = pd.to_numeric(frame.get(metric, pd.Series(dtype=float)), errors="coerce")
    finite = frame[np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))].copy()
    if finite.empty:
        return {"criterion": criterion, "metric": metric, "status": "unavailable"}
    row = finite.sort_values(metric, ascending=ascending).iloc[0].to_dict()
    selected = {
        "criterion": criterion,
        "metric": metric,
        "status": "available",
        "model_name": row.get("model_name"),
        "model_family": row.get("model_family"),
        "objective": row.get("objective"),
        "threshold_mode": row.get("threshold_mode"),
        "top_k_fraction": row.get("top_k_fraction"),
        metric: row.get(metric),
    }
    for optional in (
        "rmse",
        "spearman",
        "average_precision_u_gt_zero",
        "decision_mean_utility",
        "decision_query_call_rate",
        "utility_capture_rate",
        "precision_u_gt_zero_under_calls",
        "top_k_u_gt_zero_rate",
    ):
        if optional in row and optional not in selected:
            selected[optional] = row.get(optional)
    return selected


def _input_contract(feature_columns: list[str], train: pd.DataFrame) -> pd.DataFrame:
    feature_set = set(feature_columns)
    behavior_set = set().union(*[set(columns) for columns in BEHAVIOR_FEATURE_GROUPS.values()])
    return pd.DataFrame(
        [
            {
                "check": "x_columns_subset_of_behavior_feature_columns",
                "passed": feature_set.issubset(behavior_set),
                "detail": ",".join(feature_columns),
            },
            {
                "check": "fit_split_is_train_only",
                "passed": True,
                "detail": f"{TRAIN_SPLIT} rows={len(train)}; validation rows used for fit=0",
            },
            {
                "check": "metadata_retained_for_reporting_only",
                "passed": True,
                "detail": ",".join(METADATA_COLUMNS),
            },
        ]
    )


def _write_frame(frame: pd.DataFrame, path_without_suffix: Path) -> None:
    frame.to_csv(path_without_suffix.with_suffix(".csv"), index=False)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path_without_suffix.with_suffix(".parquet"))


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)


def _finite_metric(compute: Any) -> float | None:
    try:
        value = compute()
    except ValueError:
        return None
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def _finite_binary_metric(compute: Any, y_binary: np.ndarray) -> float | None:
    if len(np.unique(y_binary)) < 2:
        return None
    return _finite_metric(compute)


def _markdown_report(
    *,
    summary: dict[str, Any],
    input_contract: pd.DataFrame,
    model_fit_summary: pd.DataFrame,
    split_metric_summary: pd.DataFrame,
    split_decision_summary: pd.DataFrame,
    split_ranking_summary: pd.DataFrame,
    best_summary: pd.DataFrame,
) -> str:
    validation_metrics = split_metric_summary[
        (split_metric_summary["eval_split"] == VALIDATION_SPLIT) & (split_metric_summary["layer"] == "all")
    ].copy()
    validation_decision = split_decision_summary[
        (split_decision_summary["eval_split"] == VALIDATION_SPLIT) & (split_decision_summary["layer"] == "all")
    ].copy()
    validation_train_threshold = validation_decision[validation_decision["threshold_mode"] == "train_utility"].copy()
    validation_top10_threshold = validation_decision[validation_decision["threshold_mode"] == "top_10"].copy()
    validation_top10_ranking = split_ranking_summary[
        (split_ranking_summary["eval_split"] == VALIDATION_SPLIT)
        & (split_ranking_summary["layer"] == "all")
        & np.isclose(split_ranking_summary["top_k_fraction"], 0.10)
    ].copy()
    train_decision = split_decision_summary[
        (split_decision_summary["eval_split"] == TRAIN_SPLIT) & (split_decision_summary["layer"] == "all")
    ].copy()
    train_train_threshold = train_decision[train_decision["threshold_mode"] == "train_utility"].copy()
    target_models = [
        "lda_classifier",
        "softmax_logistic_classifier",
        "linear_svm_classifier",
        "ridge_regression",
    ]
    target_layer_decision = split_decision_summary[
        (split_decision_summary["eval_split"] == VALIDATION_SPLIT)
        & (split_decision_summary["threshold_mode"] == "train_utility")
        & (split_decision_summary["model_name"].isin(target_models))
        & (split_decision_summary["layer"].isin(["label_source", "dimension", "FE_ratio"]))
    ].copy()
    layer_table_columns = [
        "model_name",
        "group",
        "rows",
        "decision_mean_utility",
        "utility_capture_rate",
        "precision_u_gt_zero_under_calls",
        "unhelpful_call_cost_sum",
    ]
    return "\n".join(
        [
            "# BBOB Decision Model Benchmark Comparison",
            "",
            "## Scope",
            "",
            "- Dataset: existing phase1 refined sampling materialized BBOB Decision dataset.",
            f"- Train split: `{TRAIN_SPLIT}`; validation split: `{VALIDATION_SPLIT}`.",
            "- No validation rows were used to fit preprocessing, model parameters, or thresholds.",
            f"- Feature group: `{summary['feature_group']}` with {summary['feature_count']} behavior-only input columns.",
            "- Classification models are evaluated as decision-score models using probability or decision-function scores.",
            "",
            "## Input Contract",
            "",
            _markdown_table(input_contract),
            "",
            "## Models",
            "",
            _markdown_table(
                model_fit_summary[
                    [
                        "model_name",
                        "model_family",
                        "objective",
                        "fit_seconds",
                        "validation_prediction_seconds",
                    ]
                ].sort_values("fit_seconds")
            ),
            "",
            "## Best Rows",
            "",
            _markdown_table(best_summary),
            "",
            "## Validation Score Metrics",
            "",
            _markdown_table(
                validation_metrics[
                    [
                        "model_name",
                        "objective",
                        "rmse",
                        "spearman",
                        "roc_auc_u_gt_zero",
                        "average_precision_u_gt_zero",
                    ]
                ].sort_values("average_precision_u_gt_zero", ascending=False, na_position="last")
            ),
            "",
            "## Validation Decision: Train Utility Threshold",
            "",
            _markdown_table(
                validation_train_threshold[
                    [
                        "model_name",
                        "objective",
                        "decision_query_call_rate",
                        "decision_mean_utility",
                        "positive_row_capture_rate",
                        "utility_capture_rate",
                        "precision_u_gt_zero_under_calls",
                    ]
                ].sort_values("decision_mean_utility", ascending=False)
            ),
            "",
            "## Validation Decision: Top 10% Train-Score Threshold",
            "",
            _markdown_table(
                validation_top10_threshold[
                    [
                        "model_name",
                        "objective",
                        "decision_query_call_rate",
                        "decision_mean_utility",
                        "positive_row_capture_rate",
                        "utility_capture_rate",
                        "precision_u_gt_zero_under_calls",
                    ]
                ].sort_values("decision_mean_utility", ascending=False)
            ),
            "",
            "## Validation Top 10% Ranking",
            "",
            _markdown_table(
                validation_top10_ranking[
                    [
                        "model_name",
                        "objective",
                        "top_k_u_gt_zero_rate",
                        "lift_vs_base_rate",
                        "positive_row_capture_rate",
                        "utility_capture_rate",
                        "top_k_mean_observed_utility",
                    ]
                ].sort_values("utility_capture_rate", ascending=False)
            ),
            "",
            "## Train Decision: Train Utility Threshold",
            "",
            _markdown_table(
                train_train_threshold[
                    [
                        "model_name",
                        "objective",
                        "decision_query_call_rate",
                        "decision_mean_utility",
                        "positive_row_capture_rate",
                        "utility_capture_rate",
                        "precision_u_gt_zero_under_calls",
                    ]
                ].sort_values("decision_mean_utility", ascending=False)
            ),
            "",
            "## Target Models: Validation Decision by Label Source",
            "",
            _markdown_table(
                target_layer_decision[target_layer_decision["layer"] == "label_source"]
                .sort_values(["group", "decision_mean_utility"], ascending=[True, False])[layer_table_columns]
            ),
            "",
            "## Target Models: Validation Decision by Dimension",
            "",
            _markdown_table(
                target_layer_decision[target_layer_decision["layer"] == "dimension"]
                .sort_values(["dimension", "decision_mean_utility"], ascending=[True, False])[layer_table_columns]
            ),
            "",
            "## Target Models: Validation Decision by FE Ratio",
            "",
            _markdown_table(
                target_layer_decision[target_layer_decision["layer"] == "FE_ratio"]
                .sort_values(["FE_ratio", "decision_mean_utility"], ascending=[True, False])[layer_table_columns]
            ),
            "",
            "## Output Files",
            "",
            _markdown_table(pd.DataFrame([{"name": key, "path": value} for key, value in summary["outputs"].items()])),
            "",
        ]
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    def format_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value).replace("|", "\\|")

    headers = [str(column).replace("|", "\\|") for column in frame.columns]
    rows = [[format_value(value) for value in row] for row in frame.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Decision Model algorithms on BBOB train/validation data.")
    parser.add_argument("--query-id", choices=sorted(LANDSCAPE_QUERY_SPECS), required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--schema", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--feature-group", choices=sorted(BEHAVIOR_FEATURE_GROUPS), default="primary_with_maturity")
    parser.add_argument("--random-seed", type=int, default=1701)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    query_root = decision_query_root(args.query_id)
    materialized = query_root / "materialized_training_data"
    dataset = args.dataset or materialized / "decision_dataset.parquet"
    schema = args.schema or materialized / "decision_dataset_schema.json"
    output_dir = args.output_dir or query_root / "model_benchmark_comparison"
    if args.report_only:
        write_report_from_existing_outputs(output_dir=output_dir)
        return
    run_model_benchmark_comparison(
        query_id=args.query_id,
        dataset_path=dataset,
        schema_path=schema,
        output_dir=output_dir,
        feature_group=args.feature_group,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
