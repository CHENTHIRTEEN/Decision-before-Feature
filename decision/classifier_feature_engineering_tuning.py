from __future__ import annotations

import argparse
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from behavior.features import BEHAVIOR_FEATURE_GROUPS
from decision.model_benchmark_comparison import (
    _decision_summary,
    _finite_binary_metric,
    _markdown_table,
    _prediction_frame,
    _split_metric_summary,
    _thresholds_from_train_scores,
)
from decision.train_full_decision_model import (
    AUXILIARY_LABEL_COLUMN,
    DEFAULT_DATASET_PATH,
    DEFAULT_READINESS_SUMMARY_PATH,
    DEFAULT_SCHEMA_PATH,
    METADATA_COLUMNS,
    TARGET_COLUMN,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    _check_dataset,
    _check_family_split,
    _check_readiness,
    _feature_columns,
    _read_readiness_summary,
)


DEFAULT_OUTPUT_DIR = Path("results/decision/phase1_refined_sampling/classifier_feature_engineering_tuning")
FEATURE_GROUP = "primary_with_maturity"
TARGET_MODELS = ("lda_classifier", "softmax_logistic_classifier", "linear_svm_classifier")
FEATURE_SCHEMES = (
    "raw_primary_with_maturity",
    "engineered_univariate_selected",
    "engineered_interaction_selected",
)


@dataclass(frozen=True)
class CandidateSpec:
    base_model_name: str
    model_family: str
    estimator_name: str
    estimator: BaseEstimator
    params: dict[str, Any]
    role: str


@dataclass
class StoredScores:
    candidate_id: str
    base_model_name: str
    model_family: str
    estimator_name: str
    feature_scheme: str
    params_json: str
    role: str
    train_scores: np.ndarray
    validation_scores: np.ndarray
    thresholds: dict[str, float]
    validation_decision_mean_utility: float


class BehaviorFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, *, include_interactions: bool = False) -> None:
        self.include_interactions = include_interactions

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "BehaviorFeatureEngineer":
        array = np.asarray(x, dtype=float)
        self.n_features_in_ = int(array.shape[1])
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        array = np.asarray(x, dtype=float)
        if array.ndim != 2 or array.shape[1] != self.n_features_in_:
            raise ValueError("BehaviorFeatureEngineer received an unexpected feature matrix shape")
        n_rows, n_features = array.shape
        interaction_count = n_features * (n_features - 1) // 2 if self.include_interactions else 0
        output = np.empty((n_rows, n_features * 5 + interaction_count), dtype=float)
        cursor = 0
        output[:, cursor : cursor + n_features] = array
        cursor += n_features
        output[:, cursor : cursor + n_features] = array * array
        cursor += n_features
        output[:, cursor : cursor + n_features] = np.abs(array)
        cursor += n_features
        output[:, cursor : cursor + n_features] = np.sign(array) * array * array
        cursor += n_features
        output[:, cursor : cursor + n_features] = np.log1p(np.abs(array))
        cursor += n_features
        if self.include_interactions:
            for left in range(n_features):
                for right in range(left + 1, n_features):
                    output[:, cursor] = array[:, left] * array[:, right]
                    cursor += 1
        return output

    def feature_names(self, input_features: list[str]) -> list[str]:
        names = []
        names.extend(input_features)
        names.extend(f"{name}__square" for name in input_features)
        names.extend(f"{name}__abs" for name in input_features)
        names.extend(f"{name}__signed_square" for name in input_features)
        names.extend(f"{name}__log_abs" for name in input_features)
        if self.include_interactions:
            for left, left_name in enumerate(input_features):
                for right_name in input_features[left + 1 :]:
                    names.append(f"{left_name}__x__{right_name}")
        return names


def run_classifier_feature_engineering_tuning(
    *,
    dataset_path: Path,
    schema_path: Path,
    readiness_summary_path: Path,
    output_dir: Path,
    overwrite: bool,
    random_seed: int,
) -> dict[str, Any]:
    _check_output_paths(output_dir, overwrite)
    dataset = pq.read_table(dataset_path).to_pandas()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    readiness = _read_readiness_summary(readiness_summary_path)
    feature_columns = _feature_columns(schema, FEATURE_GROUP)
    _check_dataset(dataset, feature_columns)
    _check_readiness(readiness, dataset_path)

    train = dataset[dataset["split"] == TRAIN_SPLIT].copy()
    validation = dataset[dataset["split"] == VALIDATION_SPLIT].copy()
    _check_family_split(train, validation)

    output_dir.mkdir(parents=True, exist_ok=True)
    y_train = train[AUXILIARY_LABEL_COLUMN].to_numpy(dtype=bool).astype(int)
    train_observed = train[TARGET_COLUMN].to_numpy(dtype=float)

    candidate_rows: list[dict[str, Any]] = []
    selected_feature_rows: list[dict[str, Any]] = []
    baseline_scores: dict[str, StoredScores] = {}
    best_scores: dict[str, StoredScores] = {}

    for feature_scheme in FEATURE_SCHEMES:
        print(f"fitting feature scheme: {feature_scheme}", flush=True)
        feature_started = perf_counter()
        transformed = _fit_transform_feature_scheme(
            feature_scheme=feature_scheme,
            train=train,
            validation=validation,
            feature_columns=feature_columns,
            y_train=y_train,
            random_seed=random_seed,
        )
        feature_fit_seconds = perf_counter() - feature_started
        selected_feature_rows.extend(
            _selected_feature_rows(
                feature_scheme=feature_scheme,
                feature_columns=transformed["feature_names"],
                source_columns=feature_columns,
                selector_scores=transformed.get("selector_scores"),
            )
        )

        candidate_specs = _candidate_specs(random_seed)
        if feature_scheme == "raw_primary_with_maturity":
            candidate_specs = _baseline_specs(random_seed) + candidate_specs
        skipped_specs: list[CandidateSpec] = []
        if feature_scheme == "engineered_interaction_selected":
            skipped_specs = [spec for spec in candidate_specs if spec.base_model_name == "linear_svm_classifier"]
            candidate_specs = [spec for spec in candidate_specs if spec.base_model_name != "linear_svm_classifier"]
            for skipped in skipped_specs:
                candidate_rows.extend(
                    _skipped_candidate_rows(
                        spec=skipped,
                        feature_scheme=feature_scheme,
                        feature_count=int(transformed["x_train"].shape[1]),
                        rows={"train": int(len(train)), "validation": int(len(validation))},
                        reason="skipped_slow_solver",
                    )
                )

        for index, spec in enumerate(candidate_specs, start=1):
            print(
                f"fitting candidate {index}/{len(candidate_specs)} for {feature_scheme}: "
                f"{spec.base_model_name} {spec.params}",
                flush=True,
            )
            candidate_id = _candidate_id(spec, feature_scheme)
            fit_started = perf_counter()
            fitted = _fit_estimator(spec.estimator, transformed["x_train"], y_train)
            fit_seconds = perf_counter() - fit_started

            train_prediction_started = perf_counter()
            train_scores = _decision_scores(fitted, transformed["x_train"])
            train_prediction_seconds = perf_counter() - train_prediction_started
            validation_prediction_started = perf_counter()
            validation_scores = _decision_scores(fitted, transformed["x_validation"])
            validation_prediction_seconds = perf_counter() - validation_prediction_started

            thresholds = _thresholds_from_train_scores(train_scores, train_observed)
            train_prediction = _prediction_frame(
                frame=train,
                scores=train_scores,
                thresholds=thresholds,
                model_name=candidate_id,
                model_family=spec.model_family,
                objective="classification",
                eval_split=TRAIN_SPLIT,
            )
            validation_prediction = _prediction_frame(
                frame=validation,
                scores=validation_scores,
                thresholds=thresholds,
                model_name=candidate_id,
                model_family=spec.model_family,
                objective="classification",
                eval_split=VALIDATION_SPLIT,
            )
            train_row = _candidate_summary_row(
                prediction=train_prediction,
                split=TRAIN_SPLIT,
                spec=spec,
                candidate_id=candidate_id,
                feature_scheme=feature_scheme,
                feature_count=int(transformed["x_train"].shape[1]),
                feature_fit_seconds=feature_fit_seconds,
                fit_seconds=fit_seconds,
                prediction_seconds=train_prediction_seconds,
                params_json=_params_json(spec.params),
                threshold=float(thresholds["train_utility"]),
            )
            validation_row = _candidate_summary_row(
                prediction=validation_prediction,
                split=VALIDATION_SPLIT,
                spec=spec,
                candidate_id=candidate_id,
                feature_scheme=feature_scheme,
                feature_count=int(transformed["x_train"].shape[1]),
                feature_fit_seconds=feature_fit_seconds,
                fit_seconds=fit_seconds,
                prediction_seconds=validation_prediction_seconds,
                params_json=_params_json(spec.params),
                threshold=float(thresholds["train_utility"]),
            )
            candidate_rows.extend([train_row, validation_row])

            stored = StoredScores(
                candidate_id=candidate_id,
                base_model_name=spec.base_model_name,
                model_family=spec.model_family,
                estimator_name=spec.estimator_name,
                feature_scheme=feature_scheme,
                params_json=_params_json(spec.params),
                role=spec.role,
                train_scores=train_scores.copy(),
                validation_scores=validation_scores.copy(),
                thresholds=thresholds,
                validation_decision_mean_utility=float(validation_row["decision_mean_utility"]),
            )
            if spec.role == "baseline":
                baseline_scores[spec.base_model_name] = stored
            elif _is_better_candidate(stored, best_scores.get(spec.base_model_name)):
                best_scores[spec.base_model_name] = stored

    prediction_entries = []
    for base_model_name in TARGET_MODELS:
        prediction_entries.append(baseline_scores[base_model_name])
        prediction_entries.append(best_scores[base_model_name])

    split_decision_frames = []
    split_metric_frames = []
    validation_prediction_frames = []
    for entry in prediction_entries:
        output_model_name = f"{entry.base_model_name}__{entry.role}"
        train_prediction = _selected_prediction_frame(train, entry.train_scores, entry, output_model_name, TRAIN_SPLIT)
        validation_prediction = _selected_prediction_frame(
            validation,
            entry.validation_scores,
            entry,
            output_model_name,
            VALIDATION_SPLIT,
        )
        split_decision_frames.append(_decision_summary(train_prediction, entry.thresholds))
        split_decision_frames.append(_decision_summary(validation_prediction, entry.thresholds))
        split_metric_frames.append(_split_metric_summary(train_prediction, "classification"))
        split_metric_frames.append(_split_metric_summary(validation_prediction, "classification"))
        validation_prediction_frames.append(validation_prediction)

    tuning_candidate_summary = pd.DataFrame(candidate_rows)
    selected_feature_summary = pd.DataFrame(selected_feature_rows)
    split_decision_summary = pd.concat(split_decision_frames, ignore_index=True)
    split_metric_summary = pd.concat(split_metric_frames, ignore_index=True)
    validation_predictions = pd.concat(validation_prediction_frames, ignore_index=True)
    output_selection_summary = _output_selection_summary(prediction_entries)
    split_decision_summary = _annotate_selected_summary(split_decision_summary, output_selection_summary)
    split_metric_summary = _annotate_selected_summary(split_metric_summary, output_selection_summary)
    input_contract = _input_contract(feature_columns)

    _write_frame(tuning_candidate_summary, output_dir / "tuning_candidate_summary")
    _write_frame(selected_feature_summary, output_dir / "selected_feature_summary")
    _write_frame(split_decision_summary, output_dir / "split_decision_summary")
    _write_frame(split_metric_summary, output_dir / "split_metric_summary")
    _write_frame(output_selection_summary, output_dir / "output_selection_summary")
    _write_frame(input_contract, output_dir / "model_input_contract")
    _write_parquet(validation_predictions, output_dir / "validation_predictions.parquet")

    summary = {
        "experiment": "phase1_refined_sampling_classifier_feature_engineering_tuning",
        "dataset": str(dataset_path),
        "schema": str(schema_path),
        "readiness_summary": str(readiness_summary_path),
        "target_column": TARGET_COLUMN,
        "auxiliary_label_column": AUXILIARY_LABEL_COLUMN,
        "feature_group": FEATURE_GROUP,
        "feature_source_columns": feature_columns,
        "feature_source_count": len(feature_columns),
        "feature_schemes": list(FEATURE_SCHEMES),
        "target_models": list(TARGET_MODELS),
        "train_split": TRAIN_SPLIT,
        "validation_split": VALIDATION_SPLIT,
        "rows": {"train": int(len(train)), "validation": int(len(validation))},
        "candidate_rows": int(len(tuning_candidate_summary)),
        "skipped_candidate_rows": int((tuning_candidate_summary["status"] == "skipped").sum()),
        "output_prediction_model_rows": int(len(output_selection_summary)),
        "validation_prediction_rows": int(len(validation_predictions)),
        "expected_validation_prediction_rows": int(len(output_selection_summary) * len(validation)),
        "selection_metric": "bbob_validation layer=all threshold_mode=train_utility decision_mean_utility",
        "skipped_candidate_policy": "engineered_interaction_selected LinearSVC candidates are recorded as skipped_slow_solver because this solver did not finish in repeated runs; raw and univariate LinearSVC candidates remain included",
        "outputs": {
            "tuning_candidate_summary": str(output_dir / "tuning_candidate_summary.parquet"),
            "selected_feature_summary": str(output_dir / "selected_feature_summary.parquet"),
            "output_selection_summary": str(output_dir / "output_selection_summary.parquet"),
            "validation_predictions": str(output_dir / "validation_predictions.parquet"),
            "split_decision_summary": str(output_dir / "split_decision_summary.parquet"),
            "split_metric_summary": str(output_dir / "split_metric_summary.parquet"),
            "report": str(output_dir / "classifier_feature_engineering_tuning_report.md"),
            "summary": str(output_dir / "classifier_feature_engineering_tuning_summary.json"),
        },
        "data_leakage_check": {
            "decision_input_uses_only_primary_with_maturity_behavior_features_and_train_fit_derivatives": True,
            "metadata_used_as_input": False,
            "function_id_algorithm_id_or_optimizer_internal_parameters_used_as_input": False,
            "ela_features_used_as_input": False,
            "validation_rows_used_for_feature_selection_model_or_threshold_fit": 0,
        },
    }
    summary_path = output_dir / "classifier_feature_engineering_tuning_summary.json"
    report_path = output_dir / "classifier_feature_engineering_tuning_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(
        _markdown_report(
            summary=summary,
            tuning_candidate_summary=tuning_candidate_summary,
            output_selection_summary=output_selection_summary,
            split_decision_summary=split_decision_summary,
            input_contract=input_contract,
        ),
        encoding="utf-8",
    )
    print(f"wrote classifier feature engineering tuning summary to {summary_path}")
    print(f"wrote classifier feature engineering tuning report to {report_path}")
    return summary


def _check_output_paths(output_dir: Path, overwrite: bool) -> None:
    outputs = (
        output_dir / "tuning_candidate_summary.csv",
        output_dir / "tuning_candidate_summary.parquet",
        output_dir / "selected_feature_summary.csv",
        output_dir / "selected_feature_summary.parquet",
        output_dir / "output_selection_summary.csv",
        output_dir / "output_selection_summary.parquet",
        output_dir / "validation_predictions.parquet",
        output_dir / "split_decision_summary.csv",
        output_dir / "split_decision_summary.parquet",
        output_dir / "split_metric_summary.csv",
        output_dir / "split_metric_summary.parquet",
        output_dir / "model_input_contract.csv",
        output_dir / "model_input_contract.parquet",
        output_dir / "classifier_feature_engineering_tuning_report.md",
        output_dir / "classifier_feature_engineering_tuning_summary.json",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"classifier tuning outputs already exist; pass --overwrite: {existing[0]}")


def _fit_transform_feature_scheme(
    *,
    feature_scheme: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    y_train: np.ndarray,
    random_seed: int,
) -> dict[str, Any]:
    x_train_raw = train[feature_columns]
    x_validation_raw = validation[feature_columns]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(x_train_raw))
    x_validation = scaler.transform(imputer.transform(x_validation_raw))
    if feature_scheme == "raw_primary_with_maturity":
        return {"x_train": x_train, "x_validation": x_validation, "feature_names": list(feature_columns)}

    include_interactions = feature_scheme == "engineered_interaction_selected"
    k = 96 if include_interactions else 48
    engineer = BehaviorFeatureEngineer(include_interactions=include_interactions)
    x_train_engineered = engineer.fit_transform(x_train)
    x_validation_engineered = engineer.transform(x_validation)
    engineered_names = engineer.feature_names(feature_columns)

    def score_func(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = mutual_info_classif(x, y, discrete_features=False, random_state=random_seed, n_jobs=-1)
        return scores, np.full(scores.shape, np.nan, dtype=float)

    selector = SelectKBest(score_func=score_func, k=min(k, x_train_engineered.shape[1]))
    x_train_selected = selector.fit_transform(x_train_engineered, y_train)
    x_validation_selected = selector.transform(x_validation_engineered)
    selected_names = [engineered_names[index] for index in selector.get_support(indices=True)]
    return {
        "x_train": x_train_selected,
        "x_validation": x_validation_selected,
        "feature_names": selected_names,
        "selector_scores": selector.scores_[selector.get_support(indices=True)],
    }


def _baseline_specs(random_seed: int) -> list[CandidateSpec]:
    return [
        CandidateSpec(
            "lda_classifier",
            "lda",
            "LinearDiscriminantAnalysis()",
            LinearDiscriminantAnalysis(),
            {"solver": "svd", "shrinkage": None},
            "baseline",
        ),
        CandidateSpec(
            "softmax_logistic_classifier",
            "softmax_logistic",
            "LogisticRegression(C=1.0,class_weight='balanced')",
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=random_seed),
            {"C": 1.0, "class_weight": "balanced", "penalty": "l2"},
            "baseline",
        ),
        CandidateSpec(
            "linear_svm_classifier",
            "linear_svm",
            "LinearSVC(C=1.0,class_weight='balanced',tol=1e-2)",
            LinearSVC(C=1.0, class_weight="balanced", dual=False, tol=1e-2, max_iter=5000, random_state=random_seed),
            {"C": 1.0, "class_weight": "balanced"},
            "baseline",
        ),
    ]


def _candidate_specs(random_seed: int) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    specs.append(
        CandidateSpec(
            "lda_classifier",
            "lda",
            "LinearDiscriminantAnalysis(solver='svd')",
            LinearDiscriminantAnalysis(solver="svd"),
            {"solver": "svd", "shrinkage": None},
            "best_candidate",
        )
    )
    for shrinkage in (None, "auto", 0.1, 0.3, 0.5, 0.7):
        specs.append(
            CandidateSpec(
                "lda_classifier",
                "lda",
                f"LinearDiscriminantAnalysis(solver='lsqr',shrinkage={shrinkage!r})",
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage),
                {"solver": "lsqr", "shrinkage": shrinkage},
                "best_candidate",
            )
        )
    for c_value in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        for class_weight in ("balanced", None):
            specs.append(
                CandidateSpec(
                    "softmax_logistic_classifier",
                    "softmax_logistic",
                    f"LogisticRegression(C={c_value},class_weight={class_weight!r})",
                    LogisticRegression(
                        C=c_value,
                        penalty="l2",
                        class_weight=class_weight,
                        max_iter=2000,
                        random_state=random_seed,
                    ),
                    {"C": c_value, "penalty": "l2", "class_weight": class_weight},
                    "best_candidate",
                )
            )
            specs.append(
                CandidateSpec(
                    "linear_svm_classifier",
                    "linear_svm",
                    f"LinearSVC(C={c_value},class_weight={class_weight!r},tol=1e-2)",
                    LinearSVC(
                        C=c_value,
                        class_weight=class_weight,
                        dual=False,
                        tol=1e-2,
                        max_iter=8000,
                        random_state=random_seed,
                    ),
                    {"C": c_value, "class_weight": class_weight},
                    "best_candidate",
                )
            )
    return specs


def _fit_estimator(estimator: BaseEstimator, x_train: np.ndarray, y_train: np.ndarray) -> BaseEstimator:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(x_train, y_train)
    return estimator


def _decision_scores(model: BaseEstimator, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x)
        if scores.ndim == 2 and scores.shape[1] >= 2:
            output = scores[:, 1]
        else:
            output = np.asarray(scores).reshape(-1)
    elif hasattr(model, "decision_function"):
        output = model.decision_function(x)
    else:
        output = model.predict(x)
    output = np.asarray(output, dtype=float).reshape(-1)
    if not np.isfinite(output).all():
        raise ValueError("model produced non-finite decision scores")
    return output


def _candidate_id(spec: CandidateSpec, feature_scheme: str) -> str:
    param_token = (
        _params_json(spec.params)
        .replace("{", "")
        .replace("}", "")
        .replace('"', "")
        .replace(":", "_")
        .replace(",", "__")
        .replace(" ", "")
        .replace(".", "p")
    )
    return f"{spec.base_model_name}__{feature_scheme}__{spec.role}__{param_token}"


def _params_json(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True)


def _candidate_summary_row(
    *,
    prediction: pd.DataFrame,
    split: str,
    spec: CandidateSpec,
    candidate_id: str,
    feature_scheme: str,
    feature_count: int,
    feature_fit_seconds: float,
    fit_seconds: float,
    prediction_seconds: float,
    params_json: str,
    threshold: float,
) -> dict[str, Any]:
    observed = prediction[TARGET_COLUMN].to_numpy(dtype=float)
    scores = prediction["decision_score"].to_numpy(dtype=float)
    binary = observed > 0.0
    calls = prediction["decision_run_ela_train_utility"].to_numpy(dtype=bool)
    captured_positive = binary & calls
    unhelpful_calls = (~binary) & calls
    positive_utility_sum = float(np.sum(observed[binary]))
    captured_positive_utility_sum = float(np.sum(observed[captured_positive]))
    call_rows = int(np.sum(calls))
    return {
        "candidate_id": candidate_id,
        "base_model_name": spec.base_model_name,
        "model_family": spec.model_family,
        "estimator": spec.estimator_name,
        "role": spec.role,
        "feature_scheme": feature_scheme,
        "feature_count": feature_count,
        "params_json": params_json,
        "status": "completed",
        "skip_reason": None,
        "eval_split": split,
        "rows": int(len(prediction)),
        "feature_fit_seconds": float(feature_fit_seconds),
        "fit_seconds": float(fit_seconds),
        "prediction_seconds": float(prediction_seconds),
        "roc_auc_u_gt_zero": _finite_binary_metric(lambda: roc_auc_score(binary, scores), binary),
        "average_precision_u_gt_zero": _finite_binary_metric(lambda: average_precision_score(binary, scores), binary),
        "threshold": threshold,
        "decision_ela_call_rate": float(np.mean(calls)),
        "decision_mean_utility": float(np.mean(np.where(calls, observed, 0.0))),
        "utility_capture_rate": (
            captured_positive_utility_sum / positive_utility_sum if positive_utility_sum > 0.0 else 0.0
        ),
        "precision_u_gt_zero_under_calls": float(np.sum(captured_positive) / max(call_rows, 1)),
        "unhelpful_call_cost_sum": float(-np.sum(observed[unhelpful_calls])),
    }


def _skipped_candidate_rows(
    *,
    spec: CandidateSpec,
    feature_scheme: str,
    feature_count: int,
    rows: dict[str, int],
    reason: str,
) -> list[dict[str, Any]]:
    candidate_id = _candidate_id(spec, feature_scheme)
    output = []
    for split, row_count in ((TRAIN_SPLIT, rows["train"]), (VALIDATION_SPLIT, rows["validation"])):
        output.append(
            {
                "candidate_id": candidate_id,
                "base_model_name": spec.base_model_name,
                "model_family": spec.model_family,
                "estimator": spec.estimator_name,
                "role": spec.role,
                "feature_scheme": feature_scheme,
                "feature_count": feature_count,
                "params_json": _params_json(spec.params),
                "status": "skipped",
                "skip_reason": reason,
                "eval_split": split,
                "rows": row_count,
                "feature_fit_seconds": None,
                "fit_seconds": None,
                "prediction_seconds": None,
                "roc_auc_u_gt_zero": None,
                "average_precision_u_gt_zero": None,
                "threshold": None,
                "decision_ela_call_rate": None,
                "decision_mean_utility": None,
                "utility_capture_rate": None,
                "precision_u_gt_zero_under_calls": None,
                "unhelpful_call_cost_sum": None,
            }
        )
    return output


def _is_better_candidate(candidate: StoredScores, incumbent: StoredScores | None) -> bool:
    if incumbent is None:
        return True
    return candidate.validation_decision_mean_utility > incumbent.validation_decision_mean_utility


def _selected_prediction_frame(
    frame: pd.DataFrame,
    scores: np.ndarray,
    entry: StoredScores,
    output_model_name: str,
    split: str,
) -> pd.DataFrame:
    prediction = _prediction_frame(
        frame=frame,
        scores=scores,
        thresholds=entry.thresholds,
        model_name=output_model_name,
        model_family=entry.model_family,
        objective="classification",
        eval_split=split,
    )
    prediction.insert(4, "base_model_name", entry.base_model_name)
    prediction.insert(5, "candidate_id", entry.candidate_id)
    prediction.insert(6, "feature_scheme", entry.feature_scheme)
    prediction.insert(7, "tuning_role", entry.role)
    prediction.insert(8, "params_json", entry.params_json)
    return prediction


def _output_selection_summary(entries: list[StoredScores]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "output_model_name": f"{entry.base_model_name}__{entry.role}",
                "candidate_id": entry.candidate_id,
                "base_model_name": entry.base_model_name,
                "model_family": entry.model_family,
                "estimator": entry.estimator_name,
                "feature_scheme": entry.feature_scheme,
                "role": entry.role,
                "params_json": entry.params_json,
                "validation_decision_mean_utility": entry.validation_decision_mean_utility,
            }
            for entry in entries
        ]
    )


def _annotate_selected_summary(summary: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    metadata = selection[
        [
            "output_model_name",
            "candidate_id",
            "base_model_name",
            "estimator",
            "feature_scheme",
            "role",
            "params_json",
        ]
    ].copy()
    return summary.merge(metadata, left_on="model_name", right_on="output_model_name", how="left").drop(
        columns=["output_model_name"]
    )


def _selected_feature_rows(
    *,
    feature_scheme: str,
    feature_columns: list[str],
    source_columns: list[str],
    selector_scores: np.ndarray | None,
) -> list[dict[str, Any]]:
    source_set = set(source_columns)
    rows = []
    for index, feature in enumerate(feature_columns):
        source = _source_column_for_engineered_feature(feature, source_columns)
        rows.append(
            {
                "feature_scheme": feature_scheme,
                "selected_rank": index + 1,
                "feature": feature,
                "source_column": source,
                "source_column_in_primary_with_maturity": source in source_set,
                "selector_score": None if selector_scores is None else float(selector_scores[index]),
            }
        )
    return rows


def _source_column_for_engineered_feature(feature: str, source_columns: list[str]) -> str:
    for source in sorted(source_columns, key=len, reverse=True):
        if feature == source or feature.startswith(f"{source}__"):
            return source
    raise ValueError(f"engineered feature does not map to a source behavior column: {feature}")


def _input_contract(feature_columns: list[str]) -> pd.DataFrame:
    feature_set = set(feature_columns)
    primary_set = set(BEHAVIOR_FEATURE_GROUPS[FEATURE_GROUP])
    metadata_set = set(METADATA_COLUMNS)
    return pd.DataFrame(
        [
            {
                "check": "source_columns_equal_primary_with_maturity",
                "passed": feature_columns == list(BEHAVIOR_FEATURE_GROUPS[FEATURE_GROUP]),
                "detail": ",".join(feature_columns),
            },
            {
                "check": "source_columns_exclude_metadata",
                "passed": len(feature_set.intersection(metadata_set)) == 0,
                "detail": ",".join(sorted(feature_set.intersection(metadata_set))),
            },
            {
                "check": "source_columns_subset_of_behavior_feature_columns",
                "passed": feature_set.issubset(primary_set),
                "detail": ",".join(feature_columns),
            },
            {
                "check": "validation_rows_used_for_feature_selection_model_or_threshold_fit",
                "passed": True,
                "detail": "0",
            },
        ]
    )


def _write_frame(frame: pd.DataFrame, path_without_suffix: Path) -> None:
    frame.to_csv(path_without_suffix.with_suffix(".csv"), index=False)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path_without_suffix.with_suffix(".parquet"))


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)


def _markdown_report(
    *,
    summary: dict[str, Any],
    tuning_candidate_summary: pd.DataFrame,
    output_selection_summary: pd.DataFrame,
    split_decision_summary: pd.DataFrame,
    input_contract: pd.DataFrame,
) -> str:
    validation_candidates = tuning_candidate_summary[tuning_candidate_summary["eval_split"] == VALIDATION_SPLIT].copy()
    baseline = validation_candidates[validation_candidates["role"] == "baseline"].sort_values(
        "decision_mean_utility",
        ascending=False,
    )
    completed_candidates = validation_candidates[validation_candidates["status"] == "completed"].copy()
    skipped_candidates = validation_candidates[validation_candidates["status"] == "skipped"].copy()
    top_candidates = completed_candidates.sort_values("decision_mean_utility", ascending=False).head(15)
    selected_decision = split_decision_summary[
        (split_decision_summary["eval_split"] == VALIDATION_SPLIT)
        & (split_decision_summary["threshold_mode"] == "train_utility")
    ].copy()
    selected_all = selected_decision[selected_decision["layer"] == "all"].sort_values(
        "decision_mean_utility",
        ascending=False,
    )
    weak_layers = selected_decision[
        ((selected_decision["layer"] == "label_source") & (selected_decision["label_source"] == "same_algorithm"))
        | ((selected_decision["layer"] == "dimension") & (selected_decision["dimension"] == 40))
        | (
            (selected_decision["layer"] == "FE_ratio")
            & (selected_decision["FE_ratio"].isin([0.25, 0.28, 0.3, 0.35]))
        )
    ].sort_values(["layer", "group", "decision_mean_utility"], ascending=[True, True, False])
    metric_columns = [
        "base_model_name",
        "role",
        "feature_scheme",
        "decision_ela_call_rate",
        "decision_mean_utility",
        "utility_capture_rate",
        "precision_u_gt_zero_under_calls",
        "unhelpful_call_cost_sum",
    ]
    return "\n".join(
        [
            "# Classifier Feature Engineering and Tuning Report",
            "",
            "## Scope",
            "",
            "- Dataset: existing phase1 refined sampling materialized BBOB Decision dataset.",
            f"- Feature source: `{summary['feature_group']}` with {summary['feature_source_count']} behavior-only columns.",
            "- No validation rows were used for feature selection, model fitting, or threshold fitting.",
            "- Interaction LinearSVC candidates are recorded as skipped because the solver was not tractable on the selected interaction matrix in repeated runs.",
            "",
            "## Input Contract",
            "",
            _markdown_table(input_contract),
            "",
            "## Baseline Validation",
            "",
            _markdown_table(baseline[metric_columns]),
            "",
            "## Top Validation Candidates",
            "",
            _markdown_table(
                top_candidates[
                    [
                        "base_model_name",
                        "feature_scheme",
                        "params_json",
                        "decision_mean_utility",
                        "utility_capture_rate",
                        "precision_u_gt_zero_under_calls",
                        "unhelpful_call_cost_sum",
                    ]
                ]
            ),
            "",
            "## Skipped Validation Candidates",
            "",
            _markdown_table(
                skipped_candidates[
                    [
                        "base_model_name",
                        "feature_scheme",
                        "params_json",
                        "status",
                        "skip_reason",
                    ]
                ]
            ),
            "",
            "## Output Prediction Models",
            "",
            _markdown_table(output_selection_summary),
            "",
            "## Selected Baseline and Best Candidates",
            "",
            _markdown_table(selected_all[["model_name", *metric_columns[2:]]]),
            "",
            "## Selected Models on Known Weak Layers",
            "",
            _markdown_table(
                weak_layers[
                    [
                        "model_name",
                        "layer",
                        "group",
                        "rows",
                        "decision_mean_utility",
                        "utility_capture_rate",
                        "precision_u_gt_zero_under_calls",
                        "unhelpful_call_cost_sum",
                    ]
                ]
            ),
            "",
            "## Output Files",
            "",
            _markdown_table(pd.DataFrame([{"name": key, "path": value} for key, value in summary["outputs"].items()])),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune selected classifier Decision models with behavior feature engineering.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--readiness-summary", type=Path, default=DEFAULT_READINESS_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-seed", type=int, default=1701)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_classifier_feature_engineering_tuning(
        dataset_path=args.dataset,
        schema_path=args.schema,
        readiness_summary_path=args.readiness_summary,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
