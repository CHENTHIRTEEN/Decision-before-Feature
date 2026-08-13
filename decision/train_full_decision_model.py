from __future__ import annotations

import argparse
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
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from behavior.features import (
    BEHAVIOR_FEATURE_GROUPS,
    SELECTOR_BEHAVIOR_FEATURE_COLUMNS,
)
from decision.model_protocol import (
    ACTIVE_MODEL_NAMES,
    FROZEN_THRESHOLD_MODE,
    FULL_TRAIN_OOF_FOLDS,
    INNER_OOF_FOLDS,
    MODEL_SELECTION_METRIC,
    OUTER_OOF_FOLDS,
    THRESHOLD_NEIGHBORHOOD_QUANTILE,
    DecisionModelSpec,
    active_model_specs,
    decision_scores,
)
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS, get_query_spec
from selection_reference.model import SELECTION_REFERENCE_PROTOCOL, SELECTOR_TARGET_TRANSFORM
from trajectory.sampling import SAMPLING_METADATA_COLUMNS
from utility_labels.fields import NEED_QUERY_COLUMNS, RUNTIME_COST_COLUMNS, UTILITY_VALUE_COLUMNS


TRAIN_SPLIT = "bbob_train"
VALIDATION_SPLIT = "bbob_validation"
DEFAULT_TARGET_COLUMN = "u_query_lamT_1"
DEFAULT_AUXILIARY_LABEL_COLUMN = "need_query_lamT_1"
TARGET_COLUMN = DEFAULT_TARGET_COLUMN
AUXILIARY_LABEL_COLUMN = DEFAULT_AUXILIARY_LABEL_COLUMN
METADATA_COLUMNS = (
    "split",
    "problem_id",
    "family",
    "dimension",
    "prefix_algorithm",
    "seed",
    "FE",
    "FE_ratio",
    *SAMPLING_METADATA_COLUMNS,
    "query_id",
    "query_protocol",
    "sample_design_id",
    "default_algorithm",
    "no_query_algorithm",
    "selection_reference_default_algorithm",
    "selection_reference_protocol",
    "selector_prediction_source",
    "selector_target_transform",
    "selected_algorithm",
    "selected_action",
    "selected_equals_default",
    "selected_equals_prefix",
    "handoff_required",
    "best_observed_algorithm",
    "selected_matches_best_observed",
    "potential_gain_raw",
    "selector_regret_raw",
    "skip_switches_from_prefix",
    "no_query_transition_mode",
    "query_transition_mode",
    "handoff_type",
    *RUNTIME_COST_COLUMNS,
)
ACTION_RELATION_COLUMNS = (
    "selected_equals_default",
    "selected_equals_prefix",
    "handoff_required",
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
    "FE_query",
    "FE_no_query_optimization",
    "FE_query_optimization",
    "p_skip",
    "p_query",
    "performance_gain_raw",
    "performance_gain_norm",
    "runtime_query_sampling",
    "runtime_query_evaluation",
    "runtime_query_feature_computation",
    "runtime_query",
    "runtime_selection",
    "runtime_handoff",
    "runtime_no_query_handoff",
    "runtime_no_query_optimization",
    "runtime_query_optimization",
    "runtime_query_total",
    "runtime_no_query_total",
    "runtime_net",
    "time_cost_norm",
    "analysis_compute_cost_norm",
    "memory_cost_norm",
    "u_query_lamT_0",
    "u_query_lamT_025",
    "u_query_lamT_05",
    "u_query_lamT_2",
    "need_query_lamT_0",
    "need_query_lamT_025",
    "need_query_lamT_05",
    "need_query_lamT_2",
}
FORBIDDEN_X_NAME_FRAGMENTS = (
    "query",
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
RUN_KEY_COLUMNS = (
    "split",
    "problem_id",
    "family",
    "dimension",
    "prefix_algorithm",
    "seed",
)


class ConstantBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, positive_probability: float = 0.0):
        self.positive_probability = positive_probability

    def fit(self, X: Any, y: Any) -> "ConstantBinaryClassifier":
        probability = float(self.positive_probability)
        if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise ValueError("constant positive probability must be finite and in [0, 1]")
        self.classes_ = np.asarray([0, 1], dtype=int)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        probability = float(self.positive_probability)
        rows = len(X)
        return np.column_stack(
            [
                np.full(rows, 1.0 - probability, dtype=float),
                np.full(rows, probability, dtype=float),
            ]
        )

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def _threshold_candidates(thresholds: np.ndarray) -> np.ndarray:
    values = np.asarray(thresholds, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("threshold candidates require at least one finite score")
    lower = np.nextafter(float(np.min(values)), -np.inf)
    return np.unique(np.concatenate([np.asarray([lower], dtype=float), values]))


def train_full_decision_models(
    *,
    query_id: str,
    dataset_path: Path,
    schema_path: Path,
    output_dir: Path,
    overwrite: bool,
    random_seed: int,
    feature_group: str,
    target_column: str = DEFAULT_TARGET_COLUMN,
    auxiliary_label_column: str = DEFAULT_AUXILIARY_LABEL_COLUMN,
) -> dict[str, Any]:
    _set_utility_target_columns(
        target_column=target_column,
        auxiliary_label_column=auxiliary_label_column,
    )
    _check_output_paths(output_dir, overwrite)
    if feature_group not in {"T0", "B1", "B2", "B3"}:
        raise ValueError(
            "formal Decision training outputs must use canonical feature groups T0/B1/B2/B3"
        )
    dataset = pq.read_table(dataset_path).to_pandas()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    query_spec = get_query_spec(query_id)
    if schema.get("query_id") != query_id or schema.get("query_protocol") != query_spec.protocol:
        raise ValueError("Decision dataset schema does not match the requested query protocol")
    feature_columns = _feature_columns(schema, feature_group)
    _check_dataset(dataset, feature_columns)

    train = dataset[dataset["split"] == TRAIN_SPLIT].copy()
    validation = dataset[dataset["split"] == VALIDATION_SPLIT].copy()
    _check_family_split(train, validation)

    model_specs = active_model_specs(random_seed)
    model_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    regression_frames: list[pd.DataFrame] = []
    score_metric_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []
    ranking_frames: list[pd.DataFrame] = []
    preprocessing_frames: list[pd.DataFrame] = []
    train_oof_prediction_frames: list[pd.DataFrame] = []
    validation_prediction_frames: list[pd.DataFrame] = []
    nested_oof_prediction_frames: list[pd.DataFrame] = []
    oof_fold_frames: list[pd.DataFrame] = []
    model_artifacts: list[dict[str, str]] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for spec in model_specs:
        nested_predictions, nested_fold_summary = _nested_family_oof_predictions(
            spec=spec,
            train=train,
            feature_columns=feature_columns,
        )
        nested_oof_prediction_frames.append(nested_predictions)
        oof_fold_frames.append(nested_fold_summary)

        train_oof_scores, train_oof_fold_summary = _family_oof_scores(
            spec=spec,
            frame=train,
            feature_columns=feature_columns,
            requested_folds=FULL_TRAIN_OOF_FOLDS,
            fold_role="full_train_oof_threshold",
        )
        oof_fold_frames.append(train_oof_fold_summary)
        frozen_threshold = _decision_threshold_from_scores(
            frame=train,
            scores=train_oof_scores,
            observed=train[TARGET_COLUMN].to_numpy(dtype=float),
        )
        threshold_neighborhood_width = float(
            np.quantile(
                np.abs(train_oof_scores - frozen_threshold),
                THRESHOLD_NEIGHBORHOOD_QUANTILE,
            )
        )
        if not np.isfinite(threshold_neighborhood_width) or threshold_neighborhood_width < 0.0:
            raise RuntimeError("OOF threshold-neighborhood width must be finite and non-negative")

        started = perf_counter()
        fitted, used_constant_classifier, fit_positive_rows, fit_negative_rows = _fit_model(
            clone(spec.estimator),
            train,
            feature_columns,
            spec.objective,
        )
        fit_seconds = perf_counter() - started

        started = perf_counter()
        validation_scores = _predict_model(fitted, validation, feature_columns)
        validation_prediction_seconds = perf_counter() - started
        prediction_seconds_per_row = validation_prediction_seconds / max(len(validation), 1)

        thresholds = {
            "zero": 0.0,
            FROZEN_THRESHOLD_MODE: frozen_threshold,
        }
        threshold_rows.extend(
            {
                "model_name": spec.model_name,
                "model_family": spec.model_family,
                "objective": spec.objective,
                "threshold_mode": threshold_mode,
                "threshold": float(threshold),
                "threshold_source": "fixed_zero" if threshold_mode == "zero" else "full_train_family_oof",
                "fit_split": "fixed" if threshold_mode == "zero" else TRAIN_SPLIT,
                "oof_folds": 0 if threshold_mode == "zero" else FULL_TRAIN_OOF_FOLDS,
                "in_sample_train_rows_used_for_threshold_fit": 0,
                "validation_rows_used_for_threshold_fit": 0,
                "threshold_neighborhood_quantile": (
                    None
                    if threshold_mode == "zero"
                    else float(THRESHOLD_NEIGHBORHOOD_QUANTILE)
                ),
                "threshold_neighborhood_width": (
                    None
                    if threshold_mode == "zero"
                    else threshold_neighborhood_width
                ),
                "threshold_neighborhood_source": (
                    "not_applicable"
                    if threshold_mode == "zero"
                    else "full_train_family_oof_absolute_score_margin"
                ),
                "validation_rows_used_for_neighborhood_fit": 0,
            }
            for threshold_mode, threshold in thresholds.items()
        )

        model_artifact_path = model_dir / f"{spec.model_name}.joblib"
        joblib.dump(fitted, model_artifact_path, compress=3)
        model_artifacts.append({"model_name": spec.model_name, "model_path": str(model_artifact_path)})

        model_rows.append(
            {
                "model_name": spec.model_name,
                "model_family": spec.model_family,
                "objective": spec.objective,
                "estimator": spec.estimator_name,
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "train_positive_utility_rows": int(fit_positive_rows),
                "train_negative_utility_rows": int(fit_negative_rows),
                "uses_constant_classifier": bool(used_constant_classifier),
                "fit_seconds": float(fit_seconds),
                "validation_prediction_seconds": float(validation_prediction_seconds),
                "validation_prediction_seconds_per_row": float(prediction_seconds_per_row),
                "train_oof_score_mean": float(np.mean(train_oof_scores)),
                "validation_score_mean": float(np.mean(validation_scores)),
                "model_path": str(model_artifact_path),
            }
        )
        preprocessing_frames.append(
            _preprocessing_fit_summary(
                model=fitted,
                train=train,
                feature_columns=feature_columns,
                model_name=spec.model_name,
                model_family=spec.model_family,
            )
        )
        train_oof_prediction_frames.append(
            _prediction_frame(
                frame=train,
                scores=train_oof_scores,
                thresholds=thresholds,
                model_name=spec.model_name,
                model_family=spec.model_family,
                objective=spec.objective,
                data_split="train_oof",
            )
        )
        validation_predictions = _prediction_frame(
            frame=validation,
            scores=validation_scores,
            thresholds=thresholds,
            model_name=spec.model_name,
            model_family=spec.model_family,
            objective=spec.objective,
            data_split="validation",
        )
        validation_prediction_frames.append(validation_predictions)

        score_metric_frames.append(
            _layer_metric_summary(
                frame=validation_predictions,
                model_name=spec.model_name,
                model_family=spec.model_family,
                row_fn=lambda layer_frame, layer, group, mn, mf, objective=spec.objective: _score_metric_row(
                    layer_frame,
                    layer=layer,
                    group=group,
                    model_name=mn,
                    model_family=mf,
                    objective=objective,
                ),
            )
        )
        if spec.supports_utility_rmse:
            regression_frames.append(
                _layer_metric_summary(
                    frame=validation_predictions,
                    model_name=spec.model_name,
                    model_family=spec.model_family,
                    row_fn=_regression_row,
                )
            )
        for threshold_mode, threshold in thresholds.items():
            decision_frames.append(
                _layer_metric_summary(
                    frame=validation_predictions,
                    model_name=spec.model_name,
                    model_family=spec.model_family,
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
                model_name=spec.model_name,
                model_family=spec.model_family,
            )
        )

    nested_oof_predictions = pd.concat(nested_oof_prediction_frames, ignore_index=True)
    model_selection_summary = _nested_model_selection_summary(nested_oof_predictions, model_specs)
    selected_rows = model_selection_summary[model_selection_summary["selected_model"]]
    if len(selected_rows) != 1:
        raise RuntimeError("nested OOF model selection must select exactly one active candidate")
    selected_model_name = str(selected_rows.iloc[0]["model_name"])

    model_fit_summary = pd.DataFrame(model_rows)
    threshold_summary = pd.DataFrame(threshold_rows)
    model_fit_summary["selected_by_nested_oof"] = (
        model_fit_summary["model_name"].astype(str) == selected_model_name
    )
    threshold_summary["selected_by_nested_oof"] = (
        threshold_summary["model_name"].astype(str) == selected_model_name
    )
    validation_regression_summary = pd.concat(regression_frames, ignore_index=True)
    validation_score_summary = pd.concat(score_metric_frames, ignore_index=True)
    validation_decision_summary = pd.concat(decision_frames, ignore_index=True)
    validation_ranking_summary = pd.concat(ranking_frames, ignore_index=True)
    preprocessing_fit_summary = pd.concat(preprocessing_frames, ignore_index=True)
    train_oof_predictions = pd.concat(train_oof_prediction_frames, ignore_index=True)
    validation_predictions = pd.concat(validation_prediction_frames, ignore_index=True)
    oof_fold_summary = pd.concat(oof_fold_frames, ignore_index=True)
    for frame in (nested_oof_predictions, train_oof_predictions, validation_predictions):
        frame["selected_by_nested_oof"] = frame["model_name"].astype(str) == selected_model_name
    input_contract = _model_input_contract(feature_columns, train)

    _write_frame(input_contract, output_dir / "model_input_contract")
    _write_frame(preprocessing_fit_summary, output_dir / "preprocessing_fit_summary")
    _write_frame(model_fit_summary, output_dir / "model_fit_summary")
    _write_frame(threshold_summary, output_dir / "decision_thresholds")
    _write_frame(model_selection_summary, output_dir / "model_selection_summary")
    _write_frame(oof_fold_summary, output_dir / "oof_fold_summary")
    _write_frame(validation_regression_summary, output_dir / "validation_regression_summary")
    _write_frame(validation_score_summary, output_dir / "validation_score_summary")
    _write_frame(validation_decision_summary, output_dir / "validation_decision_summary")
    _write_frame(validation_ranking_summary, output_dir / "validation_ranking_summary")
    pq.write_table(
        pa.Table.from_pandas(nested_oof_predictions, preserve_index=False),
        output_dir / "nested_oof_predictions.parquet",
    )
    pq.write_table(
        pa.Table.from_pandas(train_oof_predictions, preserve_index=False),
        output_dir / "train_oof_predictions.parquet",
    )
    pq.write_table(
        pa.Table.from_pandas(validation_predictions, preserve_index=False),
        output_dir / "validation_predictions.parquet",
    )

    summary = {
        "experiment": "phase1_refined_sampling_full_decision_model_training",
        "dataset": str(dataset_path),
        "schema": str(schema_path),
        "query_id": query_id,
        "query_protocol": query_spec.protocol,
        "sample_design_id": query_spec.sample_design_id,
        "target_column": TARGET_COLUMN,
        "auxiliary_label_column": AUXILIARY_LABEL_COLUMN,
        "feature_group": feature_group,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "random_seed": int(random_seed),
        "available_feature_groups": {name: list(columns) for name, columns in BEHAVIOR_FEATURE_GROUPS.items()},
        "train_split": TRAIN_SPLIT,
        "validation_split": VALIDATION_SPLIT,
        "rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
        },
        "models_trained": list(ACTIVE_MODEL_NAMES),
        "model_selection_metric": MODEL_SELECTION_METRIC,
        "selected_model_name": selected_model_name,
        "selected_model_source": "nested_function_family_oof_on_bbob_train",
        "model_selection_tie_break": "frozen_candidate_order",
        "threshold_modes": ["zero", FROZEN_THRESHOLD_MODE],
        "oof_protocol": {
            "group_column": "family",
            "outer_folds": OUTER_OOF_FOLDS,
            "inner_folds": INNER_OOF_FOLDS,
            "full_train_threshold_folds": FULL_TRAIN_OOF_FOLDS,
            "outer_role": "unbiased_train-side_model_selection_evaluation",
            "inner_role": "outer-fold_threshold_fit",
            "full_train_oof_role": "frozen_threshold_fit_before_validation",
            "threshold_neighborhood": {
                "definition": "Q10(abs(full-train family-OOF score - frozen oof_utility threshold))",
                "quantile": THRESHOLD_NEIGHBORHOOD_QUANTILE,
                "role": "optional_post-training_online_review_only",
                "validation_rows_used": 0,
            },
        },
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
            "model_selection_summary": str(output_dir / "model_selection_summary.parquet"),
            "oof_fold_summary": str(output_dir / "oof_fold_summary.parquet"),
            "nested_oof_predictions": str(output_dir / "nested_oof_predictions.parquet"),
            "train_oof_predictions": str(output_dir / "train_oof_predictions.parquet"),
            "validation_regression_summary": str(output_dir / "validation_regression_summary.parquet"),
            "validation_score_summary": str(output_dir / "validation_score_summary.parquet"),
            "validation_decision_summary": str(output_dir / "validation_decision_summary.parquet"),
            "validation_ranking_summary": str(output_dir / "validation_ranking_summary.parquet"),
            "validation_predictions": str(output_dir / "validation_predictions.parquet"),
            "report": str(output_dir / "full_decision_model_training_report.md"),
            "summary": str(output_dir / "full_decision_model_training_summary.json"),
        },
        "data_leakage_check": {
            "family_split_overlap": [],
            "decision_input_uses_only_behavior_features": True,
            "metadata_used_as_input": False,
            "algorithm_identifier_used_as_input": False,
            "query_features_used_as_input": False,
            "model_selection_uses_validation_rows": False,
            "threshold_selection_uses_validation_rows": False,
            "oof_preprocessing_is_fit_within_each_family_fold": True,
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
            model_selection_summary=model_selection_summary,
            oof_fold_summary=oof_fold_summary,
            regression_summary=validation_regression_summary,
            score_summary=validation_score_summary,
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
        output_dir / "model_selection_summary.csv",
        output_dir / "model_selection_summary.parquet",
        output_dir / "oof_fold_summary.csv",
        output_dir / "oof_fold_summary.parquet",
        output_dir / "nested_oof_predictions.parquet",
        output_dir / "train_oof_predictions.parquet",
        output_dir / "validation_regression_summary.csv",
        output_dir / "validation_regression_summary.parquet",
        output_dir / "validation_score_summary.csv",
        output_dir / "validation_score_summary.parquet",
        output_dir / "validation_decision_summary.csv",
        output_dir / "validation_decision_summary.parquet",
        output_dir / "validation_ranking_summary.csv",
        output_dir / "validation_ranking_summary.parquet",
        output_dir / "validation_predictions.parquet",
    )
    model_outputs = tuple((output_dir / "models").glob("*.joblib")) if (output_dir / "models").exists() else ()
    existing = [path for path in outputs if path.exists()] + list(model_outputs)
    if existing and not overwrite:
        raise FileExistsError(f"full training outputs already exist; pass --overwrite: {existing[0]}")


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


def _feature_columns(schema: dict[str, Any], feature_group: str) -> list[str]:
    schema_columns = list(schema.get("input_columns", []))
    if schema_columns != list(SELECTOR_BEHAVIOR_FEATURE_COLUMNS):
        raise ValueError(
            "schema input_columns must exactly equal SELECTOR_BEHAVIOR_FEATURE_COLUMNS"
        )
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


def _validate_formal_feature_groups() -> None:
    expected_counts = {"T0": 1, "B1": 19, "B2": 25, "B3": 31}
    previous: tuple[str, ...] = ()
    groups: list[frozenset[str]] = []
    for name, expected_count in expected_counts.items():
        columns = tuple(BEHAVIOR_FEATURE_GROUPS[name])
        if len(columns) != expected_count or len(set(columns)) != expected_count:
            raise RuntimeError(f"formal feature group {name} must have {expected_count} unique inputs")
        if previous and columns[: len(previous)] != previous:
            raise RuntimeError("formal feature groups T0/B1/B2/B3 must be ordered nested prefixes")
        groups.append(frozenset(columns))
        previous = columns
    if len(set(groups)) != len(groups):
        raise RuntimeError("formal feature groups T0/B1/B2/B3 must be distinct")


_validate_formal_feature_groups()


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
    if not (dataset["prefix_algorithm"].astype(str) == dataset["default_algorithm"].astype(str)).all():
        raise ValueError("main Decision dataset must use the train-derived SBS as both prefix and default")
    if dataset["skip_switches_from_prefix"].astype(bool).any():
        raise ValueError("main Decision dataset must use native no-query continuation without a prefix switch")
    selected_equals_default = (
        dataset["selected_algorithm"].astype(str) == dataset["default_algorithm"].astype(str)
    ).to_numpy(dtype=bool)
    selected_equals_prefix = (
        dataset["selected_algorithm"].astype(str) == dataset["prefix_algorithm"].astype(str)
    ).to_numpy(dtype=bool)
    if not np.array_equal(dataset["selected_equals_default"].to_numpy(dtype=bool), selected_equals_default):
        raise ValueError("selected_equals_default is inconsistent")
    if not np.array_equal(dataset["selected_equals_prefix"].to_numpy(dtype=bool), selected_equals_prefix):
        raise ValueError("selected_equals_prefix is inconsistent")
    handoff_required = ~selected_equals_prefix
    if not np.array_equal(dataset["handoff_required"].to_numpy(dtype=bool), handoff_required):
        raise ValueError("handoff_required is inconsistent")
    if not np.array_equal(
        dataset["handoff_required"].to_numpy(dtype=bool),
        dataset["handoff_type"].astype(str).eq("population_transfer_initialization").to_numpy(dtype=bool),
    ):
        raise ValueError("handoff_required must match handoff_type")
    if set(dataset["selector_target_transform"].astype(str)) != {SELECTOR_TARGET_TRANSFORM}:
        raise ValueError("selector_target_transform is inconsistent")
    if set(dataset["selection_reference_protocol"].astype(str)) != {SELECTION_REFERENCE_PROTOCOL}:
        raise ValueError("selection_reference_protocol is inconsistent")
    for column in feature_columns:
        values = pd.to_numeric(dataset[column], errors="coerce")
        non_null = values.notna()
        invalid = non_null & ~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
        if invalid.any():
            raise ValueError(f"non-null behavior feature values must be finite: {column}")


def _check_family_split(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    overlap = sorted(set(train["family"].astype(str)).intersection(validation["family"].astype(str)))
    if overlap:
        raise ValueError(f"train and validation families must be disjoint: {overlap}")


def _target_for_objective(frame: pd.DataFrame, objective: str) -> np.ndarray:
    if objective == "classification":
        target = frame[AUXILIARY_LABEL_COLUMN].to_numpy(dtype=bool).astype(int)
        return target
    if objective == "regression":
        return frame[TARGET_COLUMN].to_numpy(dtype=float)
    raise ValueError(f"unknown Decision model objective: {objective}")


def _fit_model(
    model: Pipeline,
    train: pd.DataFrame,
    feature_columns: list[str],
    objective: str,
) -> tuple[Pipeline, bool, int, int]:
    target = _target_for_objective(train, objective)
    positive_rows = int(np.sum(target)) if objective == "classification" else int(np.sum(train[TARGET_COLUMN].to_numpy(dtype=float) > 0.0))
    negative_rows = int(len(target) - positive_rows)
    uses_constant_classifier = False
    if objective == "classification" and len(np.unique(target)) < 2:
        model = Pipeline(
            [
                ("imputer", clone(model.named_steps["imputer"])),
                ("scaler", clone(model.named_steps["scaler"])),
                ("classifier", ConstantBinaryClassifier(float(positive_rows / max(len(target), 1)))),
            ]
        )
        uses_constant_classifier = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train[feature_columns], target)
    return model, uses_constant_classifier, positive_rows, negative_rows


def _predict_model(model: Pipeline, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    return decision_scores(model, frame[feature_columns])


def _family_oof_scores(
    *,
    spec: DecisionModelSpec,
    frame: pd.DataFrame,
    feature_columns: list[str],
    requested_folds: int,
    fold_role: str,
    outer_fold: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    groups = frame["family"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(int(requested_folds), len(unique_groups))
    if n_splits < 2:
        raise ValueError(f"{fold_role} requires at least two function families")
    scores = np.full(len(frame), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold_index, (fit_index, holdout_index) in enumerate(splitter.split(frame, groups=groups)):
        fit_frame = frame.iloc[fit_index]
        holdout_frame = frame.iloc[holdout_index]
        fit_families = sorted(set(fit_frame["family"].astype(str)))
        holdout_families = sorted(set(holdout_frame["family"].astype(str)))
        overlap = sorted(set(fit_families).intersection(holdout_families))
        if overlap:
            raise RuntimeError(f"function-family OOF fold overlap: {overlap}")
        fitted, used_constant_classifier, fit_positive_rows, fit_negative_rows = _fit_model(
            clone(spec.estimator),
            fit_frame,
            feature_columns,
            spec.objective,
        )
        scores[holdout_index] = _predict_model(fitted, holdout_frame, feature_columns)
        fold_rows.append(
            {
                "model_name": spec.model_name,
                "model_family": spec.model_family,
                "objective": spec.objective,
                "fold_role": fold_role,
                "outer_fold": outer_fold,
                "fold_index": int(fold_index),
                "n_splits": int(n_splits),
                "fit_rows": int(len(fit_frame)),
                "holdout_rows": int(len(holdout_frame)),
                "fit_positive_utility_rows": int(fit_positive_rows),
                "fit_negative_utility_rows": int(fit_negative_rows),
                "uses_constant_classifier": bool(used_constant_classifier),
                "fit_families": ",".join(fit_families),
                "holdout_families": ",".join(holdout_families),
                "family_overlap_count": 0,
                "validation_rows_used": 0,
            }
        )
    if not np.isfinite(scores).all():
        raise RuntimeError(f"{fold_role} did not produce one finite OOF score per row")
    return scores, pd.DataFrame(fold_rows)


def _decision_run_key_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    missing = [column for column in RUN_KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing run-key columns: {missing}")
    return RUN_KEY_COLUMNS


def _iter_decision_run_groups(frame: pd.DataFrame):
    return frame.groupby(list(_decision_run_key_columns(frame)), sort=True, dropna=False)


def _ordered_decision_run_frame(run_frame: pd.DataFrame) -> pd.DataFrame:
    required = ["FE"]
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

    return run_frame.sort_values(order_columns, kind="mergesort")


def _first_trigger_run_utility(
    ordered_run_frame: pd.DataFrame,
    *,
    scores: np.ndarray,
    observed: np.ndarray,
    threshold: float,
) -> tuple[bool, float]:
    hit = np.flatnonzero(scores > threshold)
    if hit.size == 0:
        return False, 0.0
    return True, float(observed[int(hit[0])])


def _run_best_available_positive_utility(
    ordered_run_frame: pd.DataFrame,
    *,
    scores: np.ndarray,
    observed: np.ndarray,
) -> float:
    if len(scores) != len(ordered_run_frame) or len(observed) != len(ordered_run_frame):
        raise ValueError("scores and observed arrays must match the ordered run frame")
    best = 0.0
    for threshold in _threshold_candidates(np.unique(scores)):
        triggered, utility = _first_trigger_run_utility(
            ordered_run_frame,
            scores=scores,
            observed=observed,
            threshold=float(threshold),
        )
        if triggered and utility > best:
            best = float(utility)
    return float(max(0.0, best))


def _nested_family_oof_predictions(
    *,
    spec: DecisionModelSpec,
    train: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = train["family"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(OUTER_OOF_FOLDS, len(unique_groups))
    if n_splits < 2:
        raise ValueError("nested model selection requires at least two BBOB-train function families")
    prediction_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    splitter = GroupKFold(n_splits=n_splits)
    for outer_fold, (outer_fit_index, outer_holdout_index) in enumerate(splitter.split(train, groups=groups)):
        outer_fit = train.iloc[outer_fit_index]
        outer_holdout = train.iloc[outer_holdout_index]
        inner_scores, inner_fold_summary = _family_oof_scores(
            spec=spec,
            frame=outer_fit,
            feature_columns=feature_columns,
            requested_folds=INNER_OOF_FOLDS,
            fold_role="nested_inner_threshold",
            outer_fold=int(outer_fold),
        )
        threshold = _decision_threshold_from_scores(
            frame=outer_fit,
            scores=inner_scores,
            observed=outer_fit[TARGET_COLUMN].to_numpy(dtype=float),
        )
        fitted, used_constant_classifier, fit_positive_rows, fit_negative_rows = _fit_model(
            clone(spec.estimator),
            outer_fit,
            feature_columns,
            spec.objective,
        )
        holdout_scores = _predict_model(fitted, outer_holdout, feature_columns)
        calls = holdout_scores > threshold
        output = outer_holdout[list(METADATA_COLUMNS) + [TARGET_COLUMN, AUXILIARY_LABEL_COLUMN]].copy()
        output.insert(0, "data_split", "nested_train_oof")
        output.insert(1, "model_name", spec.model_name)
        output.insert(2, "model_family", spec.model_family)
        output.insert(3, "objective", spec.objective)
        output["outer_fold"] = int(outer_fold)
        output["decision_score"] = holdout_scores
        output["nested_oof_threshold"] = float(threshold)
        output["decision_run_query_nested_oof"] = calls
        output["decision_utility_nested_oof"] = np.where(calls, output[TARGET_COLUMN], 0.0)
        prediction_frames.append(output)

        fit_families = sorted(set(outer_fit["family"].astype(str)))
        holdout_families = sorted(set(outer_holdout["family"].astype(str)))
        overlap = sorted(set(fit_families).intersection(holdout_families))
        if overlap:
            raise RuntimeError(f"nested outer function-family fold overlap: {overlap}")
        fold_frames.append(inner_fold_summary)
        fold_frames.append(
            pd.DataFrame(
                [
                    {
                        "model_name": spec.model_name,
                        "model_family": spec.model_family,
                        "objective": spec.objective,
                        "fold_role": "nested_outer_evaluation",
                        "outer_fold": int(outer_fold),
                        "fold_index": int(outer_fold),
                        "n_splits": int(n_splits),
                        "fit_rows": int(len(outer_fit)),
                        "holdout_rows": int(len(outer_holdout)),
                        "fit_positive_utility_rows": int(fit_positive_rows),
                        "fit_negative_utility_rows": int(fit_negative_rows),
                        "uses_constant_classifier": bool(used_constant_classifier),
                        "fit_families": ",".join(fit_families),
                        "holdout_families": ",".join(holdout_families),
                        "family_overlap_count": 0,
                        "validation_rows_used": 0,
                        "threshold": float(threshold),
                        "threshold_source": "inner_function_family_oof",
                    }
                ]
            )
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if len(predictions) != len(train):
        raise RuntimeError("nested OOF predictions must contain exactly one row per BBOB-train state")
    return predictions, pd.concat(fold_frames, ignore_index=True)


def _decision_threshold_from_scores(
    frame: pd.DataFrame,
    scores: np.ndarray,
    observed: np.ndarray,
) -> float:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    observed = np.asarray(observed, dtype=float).reshape(-1)
    if len(scores) != len(observed) or len(scores) != len(frame) or not len(scores):
        raise ValueError("threshold fitting requires aligned non-empty frame, score, and Utility arrays")
    if not np.isfinite(scores).all() or not np.isfinite(observed).all():
        raise ValueError("threshold fitting requires finite scores and Utility values")

    thresholds = _threshold_candidates(np.unique(scores))
    utility_delta = np.zeros(len(thresholds) + 1, dtype=float)
    call_delta = np.zeros(len(thresholds) + 1, dtype=int)

    for _, run_frame in _iter_decision_run_groups(frame):
        ordered = _ordered_decision_run_frame(run_frame)
        run_positions = frame.index.get_indexer(ordered.index)
        if (run_positions < 0).any():
            raise RuntimeError("ordered run rows are not aligned with the threshold-fitting frame")
        run_scores = scores[run_positions]
        run_observed = observed[run_positions]
        max_previous_score = -np.inf
        for score, utility in zip(run_scores, run_observed, strict=True):
            start = int(np.searchsorted(thresholds, max_previous_score, side="left"))
            end = int(np.searchsorted(thresholds, float(score), side="left"))
            if start < end:
                utility_delta[start] += float(utility)
                utility_delta[end] -= float(utility)
                call_delta[start] += 1
                call_delta[end] -= 1
            if float(score) > max_previous_score:
                max_previous_score = float(score)

    decision_utility_sums = np.cumsum(utility_delta[:-1])
    call_runs = np.cumsum(call_delta[:-1])
    best_order = np.lexsort((thresholds, -call_runs, decision_utility_sums))
    best_threshold = float(thresholds[int(best_order[-1])])

    if not np.isfinite(best_threshold):
        raise RuntimeError("OOF threshold selection produced a non-finite threshold")
    return best_threshold


def _prediction_frame(
    *,
    frame: pd.DataFrame,
    scores: np.ndarray,
    thresholds: dict[str, float],
    model_name: str,
    model_family: str,
    objective: str,
    data_split: str,
) -> pd.DataFrame:
    output = frame[list(METADATA_COLUMNS) + [TARGET_COLUMN, AUXILIARY_LABEL_COLUMN]].copy()
    output.insert(0, "data_split", data_split)
    output.insert(1, "model_name", model_name)
    output.insert(2, "model_family", model_family)
    output.insert(3, "objective", objective)
    output["decision_score"] = scores.astype(float)
    for threshold_mode, threshold in thresholds.items():
        output[f"decision_run_query_{threshold_mode}"] = scores > threshold
        output[f"decision_utility_{threshold_mode}"] = np.where(scores > threshold, output[TARGET_COLUMN], 0.0)
    return output


def _nested_model_selection_summary(
    predictions: pd.DataFrame,
    model_specs: tuple[DecisionModelSpec, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidate_order = {name: index for index, name in enumerate(ACTIVE_MODEL_NAMES)}
    for spec in model_specs:
        frame = predictions[predictions["model_name"].astype(str) == spec.model_name]
        if len(frame) == 0:
            raise RuntimeError(f"missing nested OOF predictions for {spec.model_name}")
        observed = frame[TARGET_COLUMN].to_numpy(dtype=float)
        scores = frame["decision_score"].to_numpy(dtype=float)
        positive = observed > 0.0
        calls = frame["decision_run_query_nested_oof"].to_numpy(dtype=bool)
        decision_utility = frame["decision_utility_nested_oof"].to_numpy(dtype=float)
        captured_positive = positive & calls
        positive_utility_sum = float(np.sum(observed[positive]))
        captured_positive_utility_sum = float(np.sum(observed[captured_positive]))
        row = {
            "model_name": spec.model_name,
            "model_family": spec.model_family,
            "objective": spec.objective,
            "candidate_order": int(candidate_order[spec.model_name]),
            MODEL_SELECTION_METRIC: float(np.mean(decision_utility)),
            "nested_family_oof_decision_utility_sum": float(np.sum(decision_utility)),
            "nested_family_oof_query_call_rate": float(np.mean(calls)),
            "nested_family_oof_precision_u_gt_zero_under_calls": float(
                np.sum(captured_positive) / max(int(np.sum(calls)), 1)
            ),
            "nested_family_oof_utility_capture_rate": (
                captured_positive_utility_sum / positive_utility_sum if positive_utility_sum > 0.0 else 0.0
            ),
            "nested_family_oof_auroc": _finite_binary_metric(positive, scores, roc_auc_score),
            "nested_family_oof_average_precision": _finite_binary_metric(
                positive,
                scores,
                average_precision_score,
            ),
            "nested_family_oof_spearman": _finite_metric(
                lambda: pd.Series(observed).corr(pd.Series(scores), method="spearman")
            ),
            "nested_family_oof_rmse": (
                float(mean_squared_error(observed, scores) ** 0.5) if spec.supports_utility_rmse else None
            ),
            "rmse_applicable": bool(spec.supports_utility_rmse),
            "validation_rows_used_for_model_or_threshold_selection": 0,
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        [MODEL_SELECTION_METRIC, "candidate_order"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    summary.insert(0, "selection_rank", np.arange(1, len(summary) + 1, dtype=int))
    summary["selected_model"] = summary["selection_rank"] == 1
    return summary


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
                "passed": set(feature_columns).issubset(SELECTOR_BEHAVIOR_FEATURE_COLUMNS),
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
    for relation in ACTION_RELATION_COLUMNS:
        for relation_value, group in frame.groupby(relation, dropna=False):
            rows.append(
                row_fn(
                    group,
                    relation,
                    {relation: bool(relation_value)},
                    model_name,
                    model_family,
                )
            )
    for dimension, group in frame.groupby("dimension", dropna=False):
        rows.append(row_fn(group, "dimension", {"dimension": int(dimension)}, model_name, model_family))
    for phase, group in frame.groupby("sampling_phase", dropna=False):
        rows.append(
            row_fn(
                group,
                "sampling_phase",
                {"sampling_phase": str(phase)},
                model_name,
                model_family,
            )
        )
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


def _score_metric_row(
    frame: pd.DataFrame,
    *,
    layer: str,
    group: dict[str, Any],
    model_name: str,
    model_family: str,
    objective: str,
) -> dict[str, Any]:
    observed = frame[TARGET_COLUMN].to_numpy(dtype=float)
    scores = frame["decision_score"].to_numpy(dtype=float)
    positive = observed > 0.0
    rmse_applicable = objective == "regression"
    return {
        **_common_fields(frame, layer, group, model_name, model_family),
        "objective": objective,
        "auroc": _finite_binary_metric(positive, scores, roc_auc_score),
        "average_precision": _finite_binary_metric(positive, scores, average_precision_score),
        "spearman": _finite_metric(lambda: pd.Series(observed).corr(pd.Series(scores), method="spearman")),
        "rmse": float(mean_squared_error(observed, scores) ** 0.5) if rmse_applicable else None,
        "rmse_applicable": rmse_applicable,
        "rmse_not_applicable_reason": None if rmse_applicable else "classification_score_is_not_continuous_utility",
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
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
    calls = frame[f"decision_run_query_{threshold_mode}"].to_numpy(dtype=bool)
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
        "unhelpful_call_share_all_rows": float(np.mean(unhelpful_calls)),
        "unhelpful_call_cost_sum": float(-np.sum(observed[unhelpful_calls])),
        "decision_utility_sum": float(np.sum(decision_utility)),
        "decision_mean_utility": float(np.mean(decision_utility)),
        "always_query_mean_utility": float(np.mean(observed)),
        "never_query_mean_utility": 0.0,
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
    for relation in ACTION_RELATION_COLUMNS:
        for relation_value, group in frame.groupby(relation, dropna=False):
            layers.append((group, relation, {relation: bool(relation_value)}))
    for dimension, group in frame.groupby("dimension", dropna=False):
        layers.append((group, "dimension", {"dimension": int(dimension)}))
    for phase, group in frame.groupby("sampling_phase", dropna=False):
        layers.append((group, "sampling_phase", {"sampling_phase": str(phase)}))
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
        "selected_equals_default": group.get("selected_equals_default"),
        "selected_equals_prefix": group.get("selected_equals_prefix"),
        "handoff_required": group.get("handoff_required"),
        "dimension": group.get("dimension"),
        "sampling_phase": group.get("sampling_phase"),
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


def _finite_binary_metric(labels: np.ndarray, scores: np.ndarray, metric: Any) -> float | None:
    if len(np.unique(np.asarray(labels, dtype=bool))) < 2:
        return None
    return _finite_metric(lambda: metric(labels, scores))


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
    model_selection_summary: pd.DataFrame,
    oof_fold_summary: pd.DataFrame,
    regression_summary: pd.DataFrame,
    score_summary: pd.DataFrame,
    decision_summary: pd.DataFrame,
    ranking_summary: pd.DataFrame,
) -> str:
    all_regression = regression_summary[regression_summary["layer"] == "all_validation"].sort_values("rmse")
    all_scores = score_summary[score_summary["layer"] == "all_validation"].sort_values(
        "average_precision", ascending=False
    )
    all_zero_decision = decision_summary[
        (decision_summary["layer"] == "all_validation") & (decision_summary["threshold_mode"] == "zero")
    ].sort_values("utility_capture_rate", ascending=False)
    all_oof_threshold_decision = decision_summary[
        (decision_summary["layer"] == "all_validation")
        & (decision_summary["threshold_mode"] == FROZEN_THRESHOLD_MODE)
    ].sort_values("utility_capture_rate", ascending=False)
    all_top10 = ranking_summary[
        (ranking_summary["layer"] == "all_validation") & (np.isclose(ranking_summary["top_k_fraction"], 0.10))
    ].sort_values("utility_capture_rate", ascending=False)
    action_relation_top10 = ranking_summary[
        ranking_summary["layer"].isin(ACTION_RELATION_COLUMNS)
        & np.isclose(ranking_summary["top_k_fraction"], 0.10)
    ].sort_values(["model_name", "layer", "group"])
    return "\n".join(
        [
            "# Full Decision Model training report",
            "",
            "## Scope",
            "",
            "- Dataset: formal phase1 refined sampling materialized Decision dataset.",
            "- Train split: `bbob_train`; validation split: `bbob_validation`.",
            "- Active candidates are fixed to LDA, Logistic Regression, and Ridge.",
            "- Model selection uses nested function-family OOF decision utility on BBOB-train only.",
            "- Frozen thresholds use full BBOB-train function-family OOF scores; validation is evaluation only.",
            "- Metadata is used only for reporting, splitting, and error analysis.",
            f"- Feature group: `{summary['feature_group']}` with {summary['feature_count']} input columns.",
            f"- Selected model: `{summary['selected_model_name']}`.",
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
                        "objective",
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
            "## Nested BBOB-train OOF model selection",
            "",
            _markdown_table(model_selection_summary),
            "",
            "## OOF fold separation",
            "",
            _markdown_table(
                oof_fold_summary[
                    [
                        "model_name",
                        "fold_role",
                        "outer_fold",
                        "fold_index",
                        "fit_rows",
                        "holdout_rows",
                        "family_overlap_count",
                        "validation_rows_used",
                    ]
                ]
            ),
            "",
            "## All-validation auxiliary score metrics",
            "",
            _markdown_table(
                all_scores[
                    [
                        "model_name",
                        "objective",
                        "rows",
                        "auroc",
                        "average_precision",
                        "spearman",
                        "rmse",
                        "rmse_applicable",
                    ]
                ]
            ),
            "",
            "## All-validation continuous Utility regression (Ridge only)",
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
                        "decision_query_call_rate",
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
            "## All-validation decision at frozen OOF utility threshold",
            "",
            _markdown_table(
                all_oof_threshold_decision[
                    [
                        "model_name",
                        "decision_query_call_rate",
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
            "## Explicit action-relation top 10% ranking",
            "",
            _markdown_table(
                action_relation_top10[
                    [
                        "model_name",
                        "layer",
                        "group",
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
    parser = argparse.ArgumentParser(
        description="Select and train the frozen three-candidate Decision Model protocol with nested family OOF."
    )
    parser.add_argument("--query-id", choices=sorted(LANDSCAPE_QUERY_SPECS), required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--schema", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--feature-group",
        choices=("T0", "B1", "B2", "B3"),
        default="B3",
    )
    parser.add_argument("--random-seed", type=int, default=1701)
    parser.add_argument("--target-column", choices=UTILITY_VALUE_COLUMNS, default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--auxiliary-label-column", choices=NEED_QUERY_COLUMNS, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    auxiliary_label_column = args.auxiliary_label_column or NEED_QUERY_COLUMNS[
        UTILITY_VALUE_COLUMNS.index(args.target_column)
    ]

    materialized = Path("results/decision") / args.query_id / "materialized_training_data"
    train_full_decision_models(
        query_id=args.query_id,
        dataset_path=args.dataset or materialized / "decision_dataset.parquet",
        schema_path=args.schema or materialized / "decision_dataset_schema.json",
        output_dir=args.output_dir
        or Path("results/decision") / args.query_id / "feature_group_ablation" / args.feature_group,
        overwrite=args.overwrite,
        random_seed=args.random_seed,
        feature_group=args.feature_group,
        target_column=args.target_column,
        auxiliary_label_column=auxiliary_label_column,
    )


if __name__ == "__main__":
    main()
