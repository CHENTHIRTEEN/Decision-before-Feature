from __future__ import annotations

import argparse
import importlib
import json
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from behavior.features import BEHAVIOR_FEATURE_COLUMNS, BEHAVIOR_FEATURE_GROUPS


DEFAULT_DATASET_PATH = Path("results/decision/phase1_refined_sampling/materialized_training_data/decision_dataset.parquet")
DEFAULT_SCHEMA_PATH = Path("results/decision/phase1_refined_sampling/materialized_training_data/decision_dataset_schema.json")
DEFAULT_READINESS_SUMMARY_PATH = Path(
    "results/decision/phase1_refined_sampling/full_training_readiness/full_training_readiness_summary.json"
)
DEFAULT_OUTPUT_DIR = Path("results/decision/phase1_refined_sampling/full_training")
TRAIN_SPLIT = "bbob_train"
VALIDATION_SPLIT = "bbob_validation"
TARGET_COLUMN = "u_ela_lamT_1"
AUXILIARY_LABEL_COLUMN = "need_ela_lamT_1"
METADATA_COLUMNS = (
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
)
FORBIDDEN_X_COLUMNS = {
    *METADATA_COLUMNS,
    TARGET_COLUMN,
    AUXILIARY_LABEL_COLUMN,
    "algorithm",
    "function_id",
    "function",
    "FE_total",
    "FE_prefix",
    "FE_analysis",
    "FE_skip_optimization",
    "FE_ela_optimization",
    "p_skip",
    "p_ela",
    "performance_gain_raw",
    "performance_gain_norm",
    "runtime_analysis",
    "runtime_selection",
    "runtime_skip_optimization",
    "runtime_ela_optimization",
    "time_cost_norm",
    "memory_cost_norm",
    "u_ela_lamT_0",
    "u_ela_lamT_025",
    "u_ela_lamT_05",
    "u_ela_lamT_2",
    "need_ela_lamT_0",
    "need_ela_lamT_025",
    "need_ela_lamT_05",
    "need_ela_lamT_2",
}
FORBIDDEN_X_NAME_FRAGMENTS = (
    "ela",
    "function",
    "algorithm",
    "selected",
    "default",
    "family",
    "problem",
    "dimension",
)
TOP_K_FRACTIONS = (0.05, 0.10, 0.20)
EPS = 1e-12


def train_full_decision_models(
    *,
    dataset_path: Path,
    schema_path: Path,
    readiness_summary_path: Path,
    output_dir: Path,
    overwrite: bool,
    random_seed: int,
    feature_group: str,
) -> dict[str, Any]:
    _check_output_paths(output_dir, overwrite)
    dataset = pq.read_table(dataset_path).to_pandas()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    readiness = _read_readiness_summary(readiness_summary_path)
    feature_columns = _feature_columns(schema, feature_group)
    _check_dataset(dataset, feature_columns)
    _check_readiness(readiness, dataset_path)

    train = dataset[dataset["split"] == TRAIN_SPLIT].copy()
    validation = dataset[dataset["split"] == VALIDATION_SPLIT].copy()
    _check_family_split(train, validation)

    model_specs, unavailable_models = _model_specs(random_seed)
    model_rows = []
    threshold_rows = []
    regression_frames = []
    decision_frames = []
    ranking_frames = []
    preprocessing_frames = []
    train_prediction_frames = []
    validation_prediction_frames = []
    model_artifacts = []

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for model_name, spec in model_specs.items():
        started = perf_counter()
        fitted = _fit_model(spec["estimator"], train, feature_columns)
        fit_seconds = perf_counter() - started

        train_scores = _predict_model(fitted, train, feature_columns)
        started = perf_counter()
        validation_scores = _predict_model(fitted, validation, feature_columns)
        validation_prediction_seconds = perf_counter() - started
        prediction_seconds_per_row = validation_prediction_seconds / max(len(validation), 1)

        thresholds = {
            "zero": 0.0,
            "train_utility": _decision_threshold_from_scores(
                scores=train_scores,
                observed=train[TARGET_COLUMN].to_numpy(dtype=float),
            ),
        }
        threshold_rows.extend(
            {
                "model_name": model_name,
                "model_family": spec["model_family"],
                "threshold_mode": threshold_mode,
                "threshold": float(threshold),
                "fit_split": TRAIN_SPLIT,
                "validation_rows_used_for_threshold_fit": 0,
            }
            for threshold_mode, threshold in thresholds.items()
        )

        model_artifact_path = model_dir / f"{model_name}.joblib"
        joblib.dump(fitted, model_artifact_path, compress=3)
        model_artifacts.append({"model_name": model_name, "model_path": str(model_artifact_path)})

        model_rows.append(
            {
                "model_name": model_name,
                "model_family": spec["model_family"],
                "estimator": spec["estimator_name"],
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "fit_seconds": float(fit_seconds),
                "validation_prediction_seconds": float(validation_prediction_seconds),
                "validation_prediction_seconds_per_row": float(prediction_seconds_per_row),
                "train_score_mean": float(np.mean(train_scores)),
                "validation_score_mean": float(np.mean(validation_scores)),
                "model_path": str(model_artifact_path),
            }
        )
        preprocessing_frames.append(
            _preprocessing_fit_summary(
                model=fitted,
                train=train,
                feature_columns=feature_columns,
                model_name=model_name,
                model_family=spec["model_family"],
            )
        )
        train_prediction_frames.append(
            _prediction_frame(
                frame=train,
                scores=train_scores,
                thresholds=thresholds,
                model_name=model_name,
                model_family=spec["model_family"],
                data_split="train",
            )
        )
        validation_predictions = _prediction_frame(
            frame=validation,
            scores=validation_scores,
            thresholds=thresholds,
            model_name=model_name,
            model_family=spec["model_family"],
            data_split="validation",
        )
        validation_prediction_frames.append(validation_predictions)

        regression_frames.append(
            _layer_metric_summary(
                frame=validation_predictions,
                model_name=model_name,
                model_family=spec["model_family"],
                row_fn=_regression_row,
            )
        )
        for threshold_mode, threshold in thresholds.items():
            decision_frames.append(
                _layer_metric_summary(
                    frame=validation_predictions,
                    model_name=model_name,
                    model_family=spec["model_family"],
                    row_fn=lambda layer_frame, layer, group, mn, mf, tm=threshold_mode, th=threshold: _decision_row(
                        layer_frame,
                        layer=layer,
                        group=group,
                        model_name=mn,
                        model_family=mf,
                        threshold_mode=tm,
                        threshold=th,
                    ),
                )
            )
        ranking_frames.append(
            _ranking_summary(
                frame=validation_predictions,
                model_name=model_name,
                model_family=spec["model_family"],
            )
        )

    model_fit_summary = pd.DataFrame(model_rows)
    threshold_summary = pd.DataFrame(threshold_rows)
    validation_regression_summary = pd.concat(regression_frames, ignore_index=True)
    validation_decision_summary = pd.concat(decision_frames, ignore_index=True)
    validation_ranking_summary = pd.concat(ranking_frames, ignore_index=True)
    preprocessing_fit_summary = pd.concat(preprocessing_frames, ignore_index=True)
    train_predictions = pd.concat(train_prediction_frames, ignore_index=True)
    validation_predictions = pd.concat(validation_prediction_frames, ignore_index=True)
    input_contract = _model_input_contract(feature_columns, train)

    _write_frame(input_contract, output_dir / "model_input_contract")
    _write_frame(preprocessing_fit_summary, output_dir / "preprocessing_fit_summary")
    _write_frame(model_fit_summary, output_dir / "model_fit_summary")
    _write_frame(threshold_summary, output_dir / "decision_thresholds")
    _write_frame(validation_regression_summary, output_dir / "validation_regression_summary")
    _write_frame(validation_decision_summary, output_dir / "validation_decision_summary")
    _write_frame(validation_ranking_summary, output_dir / "validation_ranking_summary")
    pq.write_table(pa.Table.from_pandas(train_predictions, preserve_index=False), output_dir / "train_predictions.parquet")
    pq.write_table(
        pa.Table.from_pandas(validation_predictions, preserve_index=False),
        output_dir / "validation_predictions.parquet",
    )

    summary = {
        "experiment": "phase1_refined_sampling_full_decision_model_training",
        "dataset": str(dataset_path),
        "schema": str(schema_path),
        "readiness_summary": str(readiness_summary_path),
        "target_column": TARGET_COLUMN,
        "auxiliary_label_column": AUXILIARY_LABEL_COLUMN,
        "feature_group": feature_group,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "available_feature_groups": {name: list(columns) for name, columns in BEHAVIOR_FEATURE_GROUPS.items()},
        "train_split": TRAIN_SPLIT,
        "validation_split": VALIDATION_SPLIT,
        "rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
        },
        "models_trained": sorted(model_specs.keys()),
        "unavailable_models": unavailable_models,
        "threshold_modes": ["zero", "train_utility"],
        "top_k_fractions": list(TOP_K_FRACTIONS),
        "preprocessing_contract": {
            "imputer": "SimpleImputer(strategy='median')",
            "imputer_fit_split": TRAIN_SPLIT,
            "scaler": "StandardScaler()",
            "scaler_fit_split": TRAIN_SPLIT,
            "validation_rows_used_for_fit": 0,
        },
        "excluded_from_decision_input": sorted(FORBIDDEN_X_COLUMNS),
        "metadata_usage": "metadata columns are retained in predictions and used only for stratified reporting",
        "model_artifacts": model_artifacts,
        "outputs": {
            "model_input_contract": str(output_dir / "model_input_contract.parquet"),
            "preprocessing_fit_summary": str(output_dir / "preprocessing_fit_summary.parquet"),
            "model_fit_summary": str(output_dir / "model_fit_summary.parquet"),
            "decision_thresholds": str(output_dir / "decision_thresholds.parquet"),
            "validation_regression_summary": str(output_dir / "validation_regression_summary.parquet"),
            "validation_decision_summary": str(output_dir / "validation_decision_summary.parquet"),
            "validation_ranking_summary": str(output_dir / "validation_ranking_summary.parquet"),
            "train_predictions": str(output_dir / "train_predictions.parquet"),
            "validation_predictions": str(output_dir / "validation_predictions.parquet"),
            "report": str(output_dir / "full_decision_model_training_report.md"),
            "summary": str(output_dir / "full_decision_model_training_summary.json"),
        },
        "data_leakage_check": {
            "family_split_overlap": [],
            "decision_input_uses_only_behavior_features": True,
            "metadata_used_as_input": False,
            "algorithm_identifier_used_as_input": False,
            "ela_features_used_as_input": False,
            "validation_rows_used_for_imputer_scaler_model_or_threshold_fit": 0,
        },
    }
    summary_path = output_dir / "full_decision_model_training_summary.json"
    report_path = output_dir / "full_decision_model_training_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(
        _markdown_report(
            summary=summary,
            input_contract=input_contract,
            preprocessing_fit_summary=preprocessing_fit_summary,
            model_fit_summary=model_fit_summary,
            threshold_summary=threshold_summary,
            regression_summary=validation_regression_summary,
            decision_summary=validation_decision_summary,
            ranking_summary=validation_ranking_summary,
        ),
        encoding="utf-8",
    )

    print(f"wrote full Decision training summary to {summary_path}")
    print(f"wrote full Decision training report to {report_path}")
    return summary


def _check_output_paths(output_dir: Path, overwrite: bool) -> None:
    outputs = (
        output_dir / "full_decision_model_training_summary.json",
        output_dir / "full_decision_model_training_report.md",
        output_dir / "model_input_contract.csv",
        output_dir / "model_input_contract.parquet",
        output_dir / "model_fit_summary.csv",
        output_dir / "model_fit_summary.parquet",
        output_dir / "preprocessing_fit_summary.csv",
        output_dir / "preprocessing_fit_summary.parquet",
        output_dir / "decision_thresholds.csv",
        output_dir / "decision_thresholds.parquet",
        output_dir / "validation_regression_summary.csv",
        output_dir / "validation_regression_summary.parquet",
        output_dir / "validation_decision_summary.csv",
        output_dir / "validation_decision_summary.parquet",
        output_dir / "validation_ranking_summary.csv",
        output_dir / "validation_ranking_summary.parquet",
        output_dir / "train_predictions.parquet",
        output_dir / "validation_predictions.parquet",
    )
    model_outputs = tuple((output_dir / "models").glob("*.joblib")) if (output_dir / "models").exists() else ()
    existing = [path for path in outputs if path.exists()] + list(model_outputs)
    if existing and not overwrite:
        raise FileExistsError(f"full training outputs already exist; pass --overwrite: {existing[0]}")


def _read_readiness_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"readiness summary is required before full training: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_columns(schema: dict[str, Any], feature_group: str) -> list[str]:
    schema_columns = list(schema.get("input_columns", []))
    if schema_columns != list(BEHAVIOR_FEATURE_COLUMNS):
        raise ValueError("schema input_columns must exactly equal BEHAVIOR_FEATURE_COLUMNS")
    if feature_group not in BEHAVIOR_FEATURE_GROUPS:
        raise ValueError(f"unknown feature group: {feature_group}")
    columns = list(BEHAVIOR_FEATURE_GROUPS[feature_group])
    missing_from_schema = sorted(set(columns).difference(schema_columns))
    if missing_from_schema:
        raise ValueError(f"feature group columns missing from materialized schema: {missing_from_schema}")
    exact_forbidden = sorted(set(columns).intersection(FORBIDDEN_X_COLUMNS))
    name_forbidden = [
        column
        for column in columns
        if any(fragment in column.lower() for fragment in FORBIDDEN_X_NAME_FRAGMENTS)
    ]
    if exact_forbidden:
        raise ValueError(f"Decision input contains forbidden columns: {exact_forbidden}")
    if name_forbidden:
        raise ValueError(f"Decision input contains forbidden name fragments: {name_forbidden}")
    return columns


def _check_dataset(dataset: pd.DataFrame, feature_columns: list[str]) -> None:
    required = set(METADATA_COLUMNS) | {TARGET_COLUMN, AUXILIARY_LABEL_COLUMN, *feature_columns}
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(f"materialized dataset missing required columns: {missing}")
    if set(dataset["split"].astype(str).unique()) != {TRAIN_SPLIT, VALIDATION_SPLIT}:
        raise ValueError(f"expected splits {TRAIN_SPLIT} and {VALIDATION_SPLIT}")
    target = pd.to_numeric(dataset[TARGET_COLUMN], errors="coerce")
    if dataset[TARGET_COLUMN].isna().any() or not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError(f"{TARGET_COLUMN} must be non-null and finite")
    if not np.array_equal(dataset[AUXILIARY_LABEL_COLUMN].to_numpy(dtype=bool), target.to_numpy(dtype=float) > 0.0):
        raise ValueError(f"{AUXILIARY_LABEL_COLUMN} must equal {TARGET_COLUMN} > 0")
    expected_label_source = np.where(
        dataset["selected_algorithm"].astype(str) == dataset["default_algorithm"].astype(str),
        "same_algorithm",
        "changed_algorithm",
    )
    if not np.array_equal(dataset["label_source"].to_numpy(dtype=str), expected_label_source):
        raise ValueError("label_source must match selected_algorithm == default_algorithm")
    for column in feature_columns:
        values = pd.to_numeric(dataset[column], errors="coerce")
        non_null = values.notna()
        invalid = non_null & ~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
        if invalid.any():
            raise ValueError(f"non-null behavior feature values must be finite: {column}")


def _check_readiness(readiness: dict[str, Any], dataset_path: Path) -> None:
    if not bool(readiness.get("can_start_full_training")):
        raise ValueError("full training readiness summary does not approve training")
    if Path(str(readiness.get("dataset_path"))) != dataset_path:
        raise ValueError("readiness summary dataset path does not match requested dataset")
    if list(readiness.get("input_columns", [])) != list(BEHAVIOR_FEATURE_COLUMNS):
        raise ValueError("readiness summary input columns must match full BEHAVIOR_FEATURE_COLUMNS")
    if readiness.get("target_column") != TARGET_COLUMN:
        raise ValueError("readiness summary target column does not match training target")


def _check_family_split(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    overlap = sorted(set(train["family"].astype(str)).intersection(validation["family"].astype(str)))
    if overlap:
        raise ValueError(f"train and validation families must be disjoint: {overlap}")


def _model_specs(random_seed: int) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    specs: dict[str, dict[str, Any]] = {
        "ridge_regression": {
            "model_family": "ridge",
            "estimator_name": "Ridge(alpha=1.0)",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("regressor", Ridge(alpha=1.0, random_state=random_seed)),
                ]
            ),
        },
        "random_forest_regression": {
            "model_family": "random_forest",
            "estimator_name": "RandomForestRegressor(n_estimators=300,min_samples_leaf=3)",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
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
            "estimator_name": "LGBMRegressor(n_estimators=300,num_leaves=31)",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
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
            "estimator_name": "XGBRegressor(n_estimators=300,max_depth=3)",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
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


def _fit_model(model: Pipeline, train: pd.DataFrame, feature_columns: list[str]) -> Pipeline:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train[feature_columns], train[TARGET_COLUMN])
    return model


def _predict_model(model: Pipeline, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    predictions = model.predict(frame[feature_columns]).astype(float)
    if not np.isfinite(predictions).all():
        raise ValueError("model produced non-finite predictions")
    return predictions


def _decision_threshold_from_scores(scores: np.ndarray, observed: np.ndarray) -> float:
    frame = pd.DataFrame({"score": scores.astype(float), "observed": observed.astype(float)})
    grouped = frame.groupby("score", as_index=True)["observed"].sum().sort_index()
    cumulative_leq = grouped.cumsum()
    total = float(grouped.sum())
    threshold_utility = total - cumulative_leq
    threshold_utility.loc[0.0] = float(frame.loc[frame["score"] > 0.0, "observed"].sum())
    best_threshold = float(threshold_utility.idxmax())
    return best_threshold


def _prediction_frame(
    *,
    frame: pd.DataFrame,
    scores: np.ndarray,
    thresholds: dict[str, float],
    model_name: str,
    model_family: str,
    data_split: str,
) -> pd.DataFrame:
    output = frame[list(METADATA_COLUMNS) + [TARGET_COLUMN, AUXILIARY_LABEL_COLUMN]].copy()
    output.insert(0, "data_split", data_split)
    output.insert(1, "model_name", model_name)
    output.insert(2, "model_family", model_family)
    output["decision_score"] = scores.astype(float)
    for threshold_mode, threshold in thresholds.items():
        output[f"decision_run_ela_{threshold_mode}"] = scores > threshold
        output[f"decision_utility_{threshold_mode}"] = np.where(scores > threshold, output[TARGET_COLUMN], 0.0)
    return output


def _model_input_contract(feature_columns: list[str], train: pd.DataFrame) -> pd.DataFrame:
    exact_forbidden = sorted(set(feature_columns).intersection(FORBIDDEN_X_COLUMNS))
    name_forbidden = [
        column
        for column in feature_columns
        if any(fragment in column.lower() for fragment in FORBIDDEN_X_NAME_FRAGMENTS)
    ]
    return pd.DataFrame(
        [
            {
                "check": "x_columns_subset_of_behavior_feature_columns",
                "passed": set(feature_columns).issubset(BEHAVIOR_FEATURE_COLUMNS),
                "detail": ",".join(feature_columns),
            },
            {
                "check": "forbidden_exact_columns_absent_from_x",
                "passed": len(exact_forbidden) == 0,
                "detail": ",".join(exact_forbidden),
            },
            {
                "check": "forbidden_name_fragments_absent_from_x",
                "passed": len(name_forbidden) == 0,
                "detail": ",".join(name_forbidden),
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


def _preprocessing_fit_summary(
    *,
    model: Pipeline,
    train: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    model_family: str,
) -> pd.DataFrame:
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    train_features = train[feature_columns]
    train_imputed = imputer.transform(train_features)
    train_mean = train_imputed.mean(axis=0)
    train_var = train_imputed.var(axis=0)
    n_samples_seen = _n_samples_seen(scaler)
    rows = []
    for index, column in enumerate(feature_columns):
        train_values = pd.to_numeric(train_features[column], errors="coerce")
        train_median = float(train_values.median())
        imputer_statistic = float(imputer.statistics_[index])
        scaler_mean = float(scaler.mean_[index])
        scaler_var = float(scaler.var_[index])
        rows.append(
            {
                "model_name": model_name,
                "model_family": model_family,
                "feature": column,
                "fit_split": TRAIN_SPLIT,
                "fit_rows": int(len(train)),
                "validation_rows_used_for_fit": 0,
                "imputer_strategy": str(imputer.strategy),
                "imputer_statistic": imputer_statistic,
                "train_raw_median": train_median,
                "imputer_statistic_matches_train_median": bool(
                    abs(imputer_statistic - train_median) <= EPS * max(1.0, abs(train_median))
                ),
                "scaler_n_samples_seen": int(n_samples_seen),
                "scaler_mean": scaler_mean,
                "train_imputed_mean": float(train_mean[index]),
                "scaler_mean_matches_train": bool(
                    abs(scaler_mean - train_mean[index]) <= EPS * max(1.0, abs(float(train_mean[index])))
                ),
                "scaler_var": scaler_var,
                "train_imputed_var": float(train_var[index]),
                "scaler_var_matches_train": bool(
                    abs(scaler_var - train_var[index]) <= EPS * max(1.0, abs(float(train_var[index])))
                ),
            }
        )
    return pd.DataFrame(rows)


def _layer_metric_summary(
    *,
    frame: pd.DataFrame,
    model_name: str,
    model_family: str,
    row_fn: Any,
) -> pd.DataFrame:
    rows = [row_fn(frame, "all_validation", {}, model_name, model_family)]
    for label_source, group in frame.groupby("label_source", dropna=False):
        rows.append(row_fn(group, "label_source", {"label_source": str(label_source)}, model_name, model_family))
    for dimension, group in frame.groupby("dimension", dropna=False):
        rows.append(row_fn(group, "dimension", {"dimension": int(dimension)}, model_name, model_family))
    for fe_ratio, group in frame.groupby("FE_ratio", dropna=False):
        rows.append(row_fn(group, "FE_ratio", {"FE_ratio": float(fe_ratio)}, model_name, model_family))
    for prefix_algorithm, group in frame.groupby("prefix_algorithm", dropna=False):
        rows.append(
            row_fn(
                group,
                "prefix_algorithm",
                {"prefix_algorithm": str(prefix_algorithm)},
                model_name,
                model_family,
            )
        )
    return pd.DataFrame(rows)


def _regression_row(
    frame: pd.DataFrame,
    layer: str,
    group: dict[str, Any],
    model_name: str,
    model_family: str,
) -> dict[str, Any]:
    observed = frame[TARGET_COLUMN].to_numpy(dtype=float)
    predicted = frame["decision_score"].to_numpy(dtype=float)
    return {
        **_common_fields(frame, layer, group, model_name, model_family),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
        "r2": _finite_metric(lambda: r2_score(observed, predicted)),
        "pearson": _finite_metric(lambda: pd.Series(observed).corr(pd.Series(predicted), method="pearson")),
        "spearman": _finite_metric(lambda: pd.Series(observed).corr(pd.Series(predicted), method="spearman")),
        "score_mean": float(np.mean(predicted)),
        "score_median": float(np.median(predicted)),
    }


def _decision_row(
    frame: pd.DataFrame,
    *,
    layer: str,
    group: dict[str, Any],
    model_name: str,
    model_family: str,
    threshold_mode: str,
    threshold: float,
) -> dict[str, Any]:
    observed = frame[TARGET_COLUMN].to_numpy(dtype=float)
    positive = observed > 0.0
    calls = frame[f"decision_run_ela_{threshold_mode}"].to_numpy(dtype=bool)
    decision_utility = np.where(calls, observed, 0.0)
    captured_positive = positive & calls
    unhelpful_calls = (~positive) & calls
    positive_rows = int(np.sum(positive))
    positive_utility_sum = float(np.sum(observed[positive]))
    captured_positive_utility_sum = float(np.sum(observed[captured_positive]))
    call_rows = int(np.sum(calls))
    return {
        **_common_fields(frame, layer, group, model_name, model_family),
        "threshold_mode": threshold_mode,
        "threshold": float(threshold),
        "decision_ela_call_rows": call_rows,
        "decision_ela_call_rate": float(np.mean(calls)),
        "mean_observed_utility_under_calls": float(np.mean(observed[calls])) if call_rows else 0.0,
        "positive_rows_captured": int(np.sum(captured_positive)),
        "positive_row_capture_rate": float(np.sum(captured_positive) / max(positive_rows, 1)),
        "utility_capture_rate": (
            captured_positive_utility_sum / positive_utility_sum if positive_utility_sum > 0.0 else 0.0
        ),
        "precision_u_gt_zero_under_calls": float(np.sum(captured_positive) / max(call_rows, 1)),
        "unhelpful_call_rows": int(np.sum(unhelpful_calls)),
        "unhelpful_call_rate_within_calls": float(np.sum(unhelpful_calls) / max(call_rows, 1)),
        "unhelpful_call_share_all_rows": float(np.mean(unhelpful_calls)),
        "unhelpful_call_cost_sum": float(-np.sum(observed[unhelpful_calls])),
        "decision_utility_sum": float(np.sum(decision_utility)),
        "decision_mean_utility": float(np.mean(decision_utility)),
        "always_ela_mean_utility": float(np.mean(observed)),
        "never_ela_mean_utility": 0.0,
        "best_observed_action_mean_utility": float(np.mean(np.maximum(observed, 0.0))),
    }


def _ranking_summary(frame: pd.DataFrame, model_name: str, model_family: str) -> pd.DataFrame:
    rows = []
    for layer_frame, layer, group in _iter_layers(frame):
        for fraction in TOP_K_FRACTIONS:
            rows.append(_ranking_row(layer_frame, layer, group, model_name, model_family, fraction))
    return pd.DataFrame(rows)


def _ranking_row(
    frame: pd.DataFrame,
    layer: str,
    group: dict[str, Any],
    model_name: str,
    model_family: str,
    fraction: float,
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
        **_common_fields(frame, layer, group, model_name, model_family),
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
    layers: list[tuple[pd.DataFrame, str, dict[str, Any]]] = [(frame, "all_validation", {})]
    for label_source, group in frame.groupby("label_source", dropna=False):
        layers.append((group, "label_source", {"label_source": str(label_source)}))
    for dimension, group in frame.groupby("dimension", dropna=False):
        layers.append((group, "dimension", {"dimension": int(dimension)}))
    for fe_ratio, group in frame.groupby("FE_ratio", dropna=False):
        layers.append((group, "FE_ratio", {"FE_ratio": float(fe_ratio)}))
    for prefix_algorithm, group in frame.groupby("prefix_algorithm", dropna=False):
        layers.append((group, "prefix_algorithm", {"prefix_algorithm": str(prefix_algorithm)}))
    return layers


def _common_fields(
    frame: pd.DataFrame,
    layer: str,
    group: dict[str, Any],
    model_name: str,
    model_family: str,
) -> dict[str, Any]:
    observed = frame[TARGET_COLUMN].to_numpy(dtype=float)
    positive = observed > 0.0
    return {
        "model_name": model_name,
        "model_family": model_family,
        "eval_split": VALIDATION_SPLIT,
        "layer": layer,
        "group": _group_label(group),
        "label_source": group.get("label_source"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "prefix_algorithm": group.get("prefix_algorithm"),
        "rows": int(len(frame)),
        "u_gt_zero_rows": int(np.sum(positive)),
        "u_gt_zero_rate": float(np.mean(positive)),
        "mean_observed_utility": float(np.mean(observed)),
        "median_observed_utility": float(np.median(observed)),
        "positive_utility_sum": float(np.sum(observed[positive])),
    }


def _group_label(group: dict[str, Any]) -> str:
    if not group:
        return "all"
    return "|".join(f"{key}={value}" for key, value in group.items())


def _write_frame(frame: pd.DataFrame, path_without_suffix: Path) -> None:
    frame.to_csv(path_without_suffix.with_suffix(".csv"), index=False)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path_without_suffix.with_suffix(".parquet"))


def _finite_metric(compute: Any) -> float | None:
    try:
        value = compute()
    except ValueError:
        return None
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def _n_samples_seen(scaler: StandardScaler) -> int:
    seen = scaler.n_samples_seen_
    if np.isscalar(seen):
        return int(seen)
    return int(np.asarray(seen).max())


def _markdown_report(
    *,
    summary: dict[str, Any],
    input_contract: pd.DataFrame,
    preprocessing_fit_summary: pd.DataFrame,
    model_fit_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    regression_summary: pd.DataFrame,
    decision_summary: pd.DataFrame,
    ranking_summary: pd.DataFrame,
) -> str:
    all_regression = regression_summary[regression_summary["layer"] == "all_validation"].sort_values("rmse")
    all_zero_decision = decision_summary[
        (decision_summary["layer"] == "all_validation") & (decision_summary["threshold_mode"] == "zero")
    ].sort_values("utility_capture_rate", ascending=False)
    all_train_threshold_decision = decision_summary[
        (decision_summary["layer"] == "all_validation") & (decision_summary["threshold_mode"] == "train_utility")
    ].sort_values("utility_capture_rate", ascending=False)
    all_top10 = ranking_summary[
        (ranking_summary["layer"] == "all_validation") & (np.isclose(ranking_summary["top_k_fraction"], 0.10))
    ].sort_values("utility_capture_rate", ascending=False)
    label_source_top10 = ranking_summary[
        (ranking_summary["layer"] == "label_source") & (np.isclose(ranking_summary["top_k_fraction"], 0.10))
    ].sort_values(["model_name", "label_source"])
    return "\n".join(
        [
            "# Full Decision Model training report",
            "",
            "## Scope",
            "",
            "- Dataset: formal phase1 refined sampling materialized Decision dataset.",
            "- Train split: `bbob_train`; validation split: `bbob_validation`.",
            "- No validation rows were used to fit imputer, scaler, model parameters, or thresholds.",
            "- Metadata is used only for reporting, splitting, and error analysis.",
            f"- Feature group: `{summary['feature_group']}` with {summary['feature_count']} input columns.",
            f"- Output directory: `{summary['outputs']['summary']}`.",
            "",
            "## Input contract",
            "",
            _markdown_table(input_contract),
            "",
            "## Models trained",
            "",
            _markdown_table(
                model_fit_summary[
                    [
                        "model_name",
                        "model_family",
                        "train_rows",
                        "validation_rows",
                        "fit_seconds",
                        "validation_prediction_seconds",
                    ]
                ]
            ),
            "",
            "## Preprocessing fit contract",
            "",
            _markdown_table(
                preprocessing_fit_summary[
                    [
                        "model_name",
                        "feature",
                        "fit_split",
                        "fit_rows",
                        "validation_rows_used_for_fit",
                        "imputer_statistic_matches_train_median",
                        "scaler_n_samples_seen",
                        "scaler_mean_matches_train",
                        "scaler_var_matches_train",
                    ]
                ].head(12)
            ),
            "",
            "## Thresholds",
            "",
            _markdown_table(threshold_summary),
            "",
            "## All-validation regression",
            "",
            _markdown_table(
                all_regression[
                    [
                        "model_name",
                        "rows",
                        "mae",
                        "rmse",
                        "r2",
                        "pearson",
                        "spearman",
                    ]
                ]
            ),
            "",
            "## All-validation decision at zero threshold",
            "",
            _markdown_table(
                all_zero_decision[
                    [
                        "model_name",
                        "decision_ela_call_rate",
                        "mean_observed_utility_under_calls",
                        "positive_row_capture_rate",
                        "utility_capture_rate",
                        "precision_u_gt_zero_under_calls",
                        "unhelpful_call_rate_within_calls",
                        "decision_mean_utility",
                    ]
                ]
            ),
            "",
            "## All-validation decision at train-derived utility threshold",
            "",
            _markdown_table(
                all_train_threshold_decision[
                    [
                        "model_name",
                        "decision_ela_call_rate",
                        "mean_observed_utility_under_calls",
                        "positive_row_capture_rate",
                        "utility_capture_rate",
                        "precision_u_gt_zero_under_calls",
                        "unhelpful_call_rate_within_calls",
                        "decision_mean_utility",
                    ]
                ]
            ),
            "",
            "## All-validation top 10% ranking",
            "",
            _markdown_table(
                all_top10[
                    [
                        "model_name",
                        "top_k_rows",
                        "top_k_u_gt_zero_rate",
                        "lift_vs_base_rate",
                        "positive_row_capture_rate",
                        "utility_capture_rate",
                        "top_k_mean_observed_utility",
                    ]
                ]
            ),
            "",
            "## changed/same top 10% ranking",
            "",
            _markdown_table(
                label_source_top10[
                    [
                        "model_name",
                        "label_source",
                        "top_k_rows",
                        "top_k_u_gt_zero_rate",
                        "lift_vs_base_rate",
                        "positive_row_capture_rate",
                        "utility_capture_rate",
                        "top_k_mean_observed_utility",
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
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value)

    if frame.empty:
        return ""
    headers = list(frame.columns)
    rows = [[format_value(value) for value in row] for row in frame.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train full Decision Models on the phase1 refined sampling dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--readiness-summary", type=Path, default=DEFAULT_READINESS_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-group", choices=sorted(BEHAVIOR_FEATURE_GROUPS), default="all_candidates")
    parser.add_argument("--random-seed", type=int, default=1701)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    train_full_decision_models(
        dataset_path=args.dataset,
        schema_path=args.schema,
        readiness_summary_path=args.readiness_summary,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        random_seed=args.random_seed,
        feature_group=args.feature_group,
    )


if __name__ == "__main__":
    main()
