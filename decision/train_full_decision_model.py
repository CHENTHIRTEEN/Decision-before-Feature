from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

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
from decision.cluster_weighting import (
    CLUSTER_BALANCED_FIT,
    cluster_balanced_row_weights,
    fit_pipeline_with_weights,
)
from decision.model_protocol import (
    ACTIVE_MODEL_NAMES,
    BEHAVIOR_FROZEN_THRESHOLD_MODE,
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
from decision.nested_learning import (
    TRAIN_SPLIT as NESTED_TRAIN_SPLIT,
    VALIDATION_SPLIT as NESTED_VALIDATION_SPLIT,
    PreparedNestedLearningInputs,
    build_required_replay_plan,
    build_fold_learning_views,
    cv_group_fold_partitions,
    family_fold_partitions,
    prepare_nested_learning_inputs,
)
from experiments.phase1_batch_common import load_config
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS, get_query_spec
from selection_reference.common import read_performance
from selection_reference.model import (
    SELECTION_REFERENCE_PROTOCOL,
    SELECTOR_TARGET_TRANSFORM,
    save_selector_model,
)
from trajectory.sampling import SAMPLING_METADATA_COLUMNS
from utility_labels.fields import (
    BEHAVIOR_UTILITY_VALUE_COLUMNS,
    NEED_BEHAVIOR_ONLY_COLUMNS,
    NEED_QUERY_COLUMNS,
    RUNTIME_COST_COLUMNS,
    UTILITY_VALUE_COLUMNS,
)


TRAIN_SPLIT = "bbob_train"
VALIDATION_SPLIT = "bbob_validation"
DEFAULT_TARGET_COLUMN = "u_query_joint_lamT_1"
DEFAULT_AUXILIARY_LABEL_COLUMN = "need_query_joint_lamT_1"
DEFAULT_BEHAVIOR_TARGET_COLUMN = "u_behavior_only_full_budget_lamT_1"
DEFAULT_BEHAVIOR_AUXILIARY_LABEL_COLUMN = "need_behavior_only_full_budget_lamT_1"
TARGET_COLUMN = DEFAULT_TARGET_COLUMN
AUXILIARY_LABEL_COLUMN = DEFAULT_AUXILIARY_LABEL_COLUMN
METADATA_COLUMNS = (
    "split",
    "problem_id",
    "function_id",
    "family",
    "cv_group_id",
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
    "FE_action_optimization",
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
    "group",
)
TOP_K_FRACTIONS = (0.05, 0.10, 0.20)
EPS = 1e-12
RUN_KEY_COLUMNS = (
    "split",
    "problem_id",
    "function_id",
    "family",
    "cv_group_id",
    "dimension",
    "prefix_algorithm",
    "seed",
)
ALL_ACCEPTED_OPPORTUNITIES = "all_accepted"
MILESTONE_ONLY_OPPORTUNITIES = "milestone_only"
OPPORTUNITY_SCOPES = (ALL_ACCEPTED_OPPORTUNITIES, MILESTONE_ONLY_OPPORTUNITIES)
FORMAL_FEATURE_GROUPS = (
    "T0",
    "B1",
    "B2",
    "B2+Motion",
    "B2+Maturity",
    "B3",
)


class ConstantBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, positive_probability: float = 0.0):
        self.positive_probability = positive_probability

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: np.ndarray | None = None,
    ) -> "ConstantBinaryClassifier":
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
    prepared_inputs: PreparedNestedLearningInputs,
    output_dir: Path,
    overwrite: bool,
    random_seed: int,
    feature_group: str,
    opportunity_scope: str,
    expected_dimensions: Sequence[int],
    target_column: str = DEFAULT_TARGET_COLUMN,
    auxiliary_label_column: str = DEFAULT_AUXILIARY_LABEL_COLUMN,
    behavior_target_column: str = DEFAULT_BEHAVIOR_TARGET_COLUMN,
    behavior_auxiliary_label_column: str = DEFAULT_BEHAVIOR_AUXILIARY_LABEL_COLUMN,
) -> dict[str, Any]:
    if FULL_TRAIN_OOF_FOLDS != OUTER_OOF_FOLDS:
        raise RuntimeError(
            "full-train threshold OOF reuses outer-fold replay roles and therefore "
            "requires FULL_TRAIN_OOF_FOLDS == OUTER_OOF_FOLDS"
        )
    _set_utility_target_columns(
        target_column=target_column,
        auxiliary_label_column=auxiliary_label_column,
    )
    if behavior_target_column not in BEHAVIOR_UTILITY_VALUE_COLUMNS:
        raise ValueError(
            "behavior_target_column must be one of "
            f"{list(BEHAVIOR_UTILITY_VALUE_COLUMNS)}"
        )
    expected_behavior_label = NEED_BEHAVIOR_ONLY_COLUMNS[
        BEHAVIOR_UTILITY_VALUE_COLUMNS.index(behavior_target_column)
    ]
    if behavior_auxiliary_label_column != expected_behavior_label:
        raise ValueError(
            f"{behavior_target_column} must use {expected_behavior_label}"
        )
    _check_output_paths(output_dir, overwrite)
    if feature_group not in set(FORMAL_FEATURE_GROUPS):
        raise ValueError(
            "formal Decision training outputs must use one active six-group ablation condition"
        )
    if opportunity_scope not in OPPORTUNITY_SCOPES:
        raise ValueError(f"unsupported opportunity_scope: {opportunity_scope}")
    if prepared_inputs.query_spec.query_id != query_id:
        raise ValueError("prepared nested inputs do not match query_id")
    frozen_dimensions = _normalize_expected_dimensions(expected_dimensions)
    query_spec = prepared_inputs.query_spec
    feature_columns = _formal_feature_columns(feature_group)
    selection_feature_columns = _formal_feature_columns("B3")
    train_families = tuple(
        sorted(
            set(
                prepared_inputs.query_adjusted_states.loc[
                    prepared_inputs.query_adjusted_states["split"].astype(str).eq(TRAIN_SPLIT),
                    "family",
                ].astype(str)
            )
        )
    )
    validation_families = tuple(
        sorted(
            set(
                prepared_inputs.query_adjusted_states.loc[
                    prepared_inputs.query_adjusted_states["split"].astype(str).eq(VALIDATION_SPLIT),
                    "family",
                ].astype(str)
            )
        )
    )
    full_views = build_fold_learning_views(
        inputs=prepared_inputs,
        fit_families=train_families,
        holdout_families=validation_families,
        fit_split=NESTED_TRAIN_SPLIT,
        holdout_split=NESTED_VALIDATION_SPLIT,
        fold_role="full_train_final",
    )
    if (
        full_views.pre_run_query_only_selector is None
        or full_views.pre_run_fit_rows is None
        or full_views.pre_run_holdout_rows is None
    ):
        raise ValueError(
            "formal training requires fold-specific FE=0 pre-run Traditional-AAS outcomes"
        )
    train = _opportunity_view(full_views.fit_labels, opportunity_scope)
    validation = _opportunity_view(full_views.holdout_labels, opportunity_scope)
    _check_dataset(pd.concat([train, validation], ignore_index=True), feature_columns)
    _check_function_split(train, validation)
    _check_complete_dimension_coverage(
        train,
        expected_dimensions=frozen_dimensions,
        context="BBOB-train Decision data",
    )
    _check_complete_dimension_coverage(
        validation,
        expected_dimensions=frozen_dimensions,
        context="BBOB-validation Decision data",
    )

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
            inputs=prepared_inputs,
            feature_columns=selection_feature_columns,
            opportunity_scope=ALL_ACCEPTED_OPPORTUNITIES,
            expected_dimensions=frozen_dimensions,
        )
        nested_oof_prediction_frames.append(nested_predictions)
        oof_fold_frames.append(nested_fold_summary)

        train_oof_frame, train_oof_scores, train_oof_fold_summary = _end_to_end_family_oof_scores(
            spec=spec,
            inputs=prepared_inputs,
            feature_columns=feature_columns,
            requested_folds=FULL_TRAIN_OOF_FOLDS,
            fold_role="full_train_oof_threshold",
            opportunity_scope=opportunity_scope,
            target_column=TARGET_COLUMN,
            auxiliary_label_column=AUXILIARY_LABEL_COLUMN,
            expected_dimensions=frozen_dimensions,
        )
        oof_fold_frames.append(train_oof_fold_summary)
        frozen_threshold = _decision_threshold_from_scores(
            frame=train_oof_frame,
            scores=train_oof_scores,
            observed=train_oof_frame[TARGET_COLUMN].to_numpy(dtype=float),
            expected_dimensions=frozen_dimensions,
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
                "threshold_source": "fixed_zero" if threshold_mode == "zero" else "full_train_cv_group_oof",
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
                    else "full_train_cv_group_oof_absolute_score_margin"
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
                "fit_weight_mode": CLUSTER_BALANCED_FIT,
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
                frame=train_oof_frame,
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
    model_selection_summary = _nested_model_selection_summary(
        nested_oof_predictions,
        model_specs,
        expected_dimensions=frozen_dimensions,
    )
    selected_rows = model_selection_summary[model_selection_summary["selected_model"]]
    if len(selected_rows) != 1:
        raise RuntimeError("nested OOF model selection must select exactly one active candidate")
    selected_model_name = str(selected_rows.iloc[0]["model_name"])
    selected_spec = next(spec for spec in model_specs if spec.model_name == selected_model_name)

    behavior_train = _opportunity_view(
        full_views.fit_labels,
        ALL_ACCEPTED_OPPORTUNITIES,
    )
    behavior_validation = _opportunity_view(
        full_views.holdout_labels,
        ALL_ACCEPTED_OPPORTUNITIES,
    )
    behavior_oof_frame, behavior_oof_scores, behavior_oof_folds = _end_to_end_family_oof_scores(
        spec=selected_spec,
        inputs=prepared_inputs,
        feature_columns=selection_feature_columns,
        requested_folds=FULL_TRAIN_OOF_FOLDS,
        fold_role="full_train_behavior_oof_threshold",
        opportunity_scope=ALL_ACCEPTED_OPPORTUNITIES,
        target_column=behavior_target_column,
        auxiliary_label_column=behavior_auxiliary_label_column,
        expected_dimensions=frozen_dimensions,
    )
    oof_fold_frames.append(behavior_oof_folds)
    behavior_frozen_threshold = _decision_threshold_from_scores(
        frame=behavior_oof_frame,
        scores=behavior_oof_scores,
        observed=behavior_oof_frame[behavior_target_column].to_numpy(dtype=float),
        expected_dimensions=frozen_dimensions,
    )
    behavior_model, behavior_constant, behavior_positive, behavior_negative = _fit_model_for_target(
        clone(selected_spec.estimator),
        behavior_train,
        selection_feature_columns,
        selected_spec.objective,
        target_column=behavior_target_column,
        auxiliary_label_column=behavior_auxiliary_label_column,
    )
    behavior_validation_scores = _predict_model(
        behavior_model,
        behavior_validation,
        selection_feature_columns,
    )
    behavior_model_path = model_dir / f"{selected_model_name}__behavior_only.joblib"
    joblib.dump(behavior_model, behavior_model_path, compress=3)
    behavior_thresholds = {
        "zero": 0.0,
        BEHAVIOR_FROZEN_THRESHOLD_MODE: float(behavior_frozen_threshold),
    }
    threshold_rows.extend(
        [
            {
                "model_name": selected_model_name,
                "model_family": selected_spec.model_family,
                "objective": selected_spec.objective,
                "policy_target": "behavior_only_full_budget",
                "threshold_mode": "zero",
                "threshold": 0.0,
                "threshold_source": "fixed_zero",
                "fit_split": "fixed",
                "oof_folds": 0,
                "in_sample_train_rows_used_for_threshold_fit": 0,
                "validation_rows_used_for_threshold_fit": 0,
            },
            {
                "model_name": selected_model_name,
                "model_family": selected_spec.model_family,
                "objective": selected_spec.objective,
                "policy_target": "behavior_only_full_budget",
                "threshold_mode": BEHAVIOR_FROZEN_THRESHOLD_MODE,
                "threshold": float(behavior_frozen_threshold),
                "threshold_source": "fold_specific_upstream_full_train_cv_group_oof",
                "fit_split": TRAIN_SPLIT,
                "oof_folds": FULL_TRAIN_OOF_FOLDS,
                "in_sample_train_rows_used_for_threshold_fit": 0,
                "validation_rows_used_for_threshold_fit": 0,
            },
        ]
    )
    train_oof_behavior_predictions = _target_prediction_frame(
        frame=behavior_oof_frame,
        scores=behavior_oof_scores,
        thresholds=behavior_thresholds,
        model_name=selected_model_name,
        model_family=selected_spec.model_family,
        objective=selected_spec.objective,
        data_split="train_oof",
        target_column=behavior_target_column,
        auxiliary_label_column=behavior_auxiliary_label_column,
        policy_target="behavior_only_full_budget",
    )
    validation_behavior_predictions = _target_prediction_frame(
        frame=behavior_validation,
        scores=behavior_validation_scores,
        thresholds=behavior_thresholds,
        model_name=selected_model_name,
        model_family=selected_spec.model_family,
        objective=selected_spec.objective,
        data_split="validation",
        target_column=behavior_target_column,
        auxiliary_label_column=behavior_auxiliary_label_column,
        policy_target="behavior_only_full_budget",
    )
    model_artifacts.append(
        {
            "model_name": selected_model_name,
            "policy_target": "behavior_only_full_budget",
            "model_path": str(behavior_model_path),
        }
    )

    selector_models: list[tuple[str, Any]] = [
        ("query_full", full_views.query_selector),
        ("behavior_only_full_budget", full_views.behavior_only_selector),
        ("query_adjusted_state_only", full_views.state_only_selector),
        ("query_only", full_views.query_only_selector),
    ]
    if full_views.pre_run_query_only_selector is not None:
        selector_models.append(
            ("pre_run_query_only", full_views.pre_run_query_only_selector)
        )
    selector_artifacts: dict[str, str] = {}
    for selector_name, selector_model in selector_models:
        selector_path = model_dir / f"selector__{selector_name}.joblib"
        save_selector_model(selector_model, selector_path)
        selector_artifacts[selector_name] = str(selector_path)

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
    for frame in (
        nested_oof_predictions,
        train_oof_predictions,
        validation_predictions,
        train_oof_behavior_predictions,
        validation_behavior_predictions,
    ):
        frame["selected_by_nested_oof"] = frame["model_name"].astype(str) == selected_model_name
    input_contract = _model_input_contract(feature_columns, train)

    _write_frame(input_contract, output_dir / "model_input_contract")
    _write_frame(preprocessing_fit_summary, output_dir / "preprocessing_fit_summary")
    _write_frame(model_fit_summary, output_dir / "model_fit_summary")
    _write_frame(threshold_summary, output_dir / "decision_thresholds")
    _write_frame(model_selection_summary, output_dir / "model_selection_summary")
    _write_frame(oof_fold_summary, output_dir / "oof_fold_summary")
    _write_frame(full_views.selector_summary, output_dir / "selector_performance_summary")
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
    pq.write_table(
        pa.Table.from_pandas(train_oof_behavior_predictions, preserve_index=False),
        output_dir / "train_oof_behavior_only_predictions.parquet",
    )
    pq.write_table(
        pa.Table.from_pandas(validation_behavior_predictions, preserve_index=False),
        output_dir / "validation_behavior_only_predictions.parquet",
    )
    pq.write_table(
        pa.Table.from_pandas(full_views.pre_run_fit_rows, preserve_index=False),
        output_dir / "train_oof_pre_run_aas_selection.parquet",
    )
    pq.write_table(
        pa.Table.from_pandas(full_views.pre_run_holdout_rows, preserve_index=False),
        output_dir / "validation_pre_run_aas_selection.parquet",
    )

    summary = {
        "experiment": "phase1_refined_sampling_full_decision_model_training",
        "training_input_mode": "raw_fold_specific_upstream_inputs",
        "materialized_single_utility_table_used": False,
        "query_id": query_id,
        "query_protocol": query_spec.protocol,
        "sample_design_id": query_spec.sample_design_id,
        "target_column": TARGET_COLUMN,
        "auxiliary_label_column": AUXILIARY_LABEL_COLUMN,
        "behavior_only_target_column": behavior_target_column,
        "behavior_only_auxiliary_label_column": behavior_auxiliary_label_column,
        "behavior_only_threshold_mode": BEHAVIOR_FROZEN_THRESHOLD_MODE,
        "feature_group": feature_group,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "opportunity_scope": opportunity_scope,
        "expected_dimensions": list(frozen_dimensions),
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
        "selected_model_source": "nested_cv_group_oof_on_bbob_train",
        "model_selection_tie_break": "function_balanced_utility_then_frozen_candidate_order",
        "threshold_modes": ["zero", FROZEN_THRESHOLD_MODE],
        "behavior_only_policy": {
            "target_column": behavior_target_column,
            "auxiliary_label_column": behavior_auxiliary_label_column,
            "feature_group": "B3",
            "feature_columns": selection_feature_columns,
            "opportunity_scope": ALL_ACCEPTED_OPPORTUNITIES,
            "model_name": selected_model_name,
            "uses_constant_classifier": bool(behavior_constant),
            "train_positive_utility_rows": int(behavior_positive),
            "train_negative_utility_rows": int(behavior_negative),
            "threshold_modes": ["zero", BEHAVIOR_FROZEN_THRESHOLD_MODE],
            "frozen_threshold": float(behavior_frozen_threshold),
            "model_path": str(behavior_model_path),
        },
        "oof_protocol": {
            "group_column": "cv_group_id",
            "outer_folds": OUTER_OOF_FOLDS,
            "inner_folds": INNER_OOF_FOLDS,
            "full_train_threshold_folds": FULL_TRAIN_OOF_FOLDS,
            "outer_role": "train_side_candidate_selection_evaluation_only",
            "inner_role": "outer-fold_threshold_fit",
            "full_train_oof_role": "frozen_threshold_fit_before_validation",
            "upstream_components_refit_per_fold": [
                "function_balanced_SBS",
                "query_full_selector",
                "behavior_only_full_budget_selector",
                "query_adjusted_state_only_selector",
                "Utility_labels_from_measured_selected_complete_paths",
                "Decision_preprocessing_and_model",
            ],
            "threshold_objective": (
                "run_to_static_problem_to_fixed_dimension_to_function_balanced_"
                "first_trigger_utility"
            ),
            "threshold_tie_break": "lower_hierarchically_weighted_call_rate_then_larger_threshold",
            "fixed_dimension_coverage_required": True,
            "threshold_neighborhood": {
                "definition": "Q10(abs(full-train CV-group OOF score - fixed oof_utility threshold))",
                "quantile": THRESHOLD_NEIGHBORHOOD_QUANTILE,
                "role": "optional_post-training_online_review_only",
                "validation_rows_used": 0,
            },
        },
        "top_k_fractions": list(TOP_K_FRACTIONS),
        "preprocessing_contract": {
            "imputer": "WeightedMedianImputer(cluster-balanced fit-fold median)",
            "imputer_fit_split": TRAIN_SPLIT,
            "scaler": "StandardScaler(cluster-balanced fit-fold weights)",
            "scaler_fit_split": TRAIN_SPLIT,
            "fit_weight_mode": CLUSTER_BALANCED_FIT,
            "validation_rows_used_for_fit": 0,
        },
        "excluded_from_decision_input": sorted(FORBIDDEN_X_COLUMNS),
        "metadata_usage": "metadata columns are retained in predictions and used only for stratified reporting",
        "model_artifacts": model_artifacts,
        "selector_artifacts": selector_artifacts,
        "complete_path_timing_contract": {
            "source": "measured_complete_policy_path",
            "origin": "decision_state_to_terminal",
            "shared_prefix_cost_treatment": "sunk_before_decision_state",
            "repetitions": 3,
            "component_runtime_sum_accepted": False,
        },
        "outputs": {
            "model_input_contract": str(output_dir / "model_input_contract.parquet"),
            "preprocessing_fit_summary": str(output_dir / "preprocessing_fit_summary.parquet"),
            "model_fit_summary": str(output_dir / "model_fit_summary.parquet"),
            "decision_thresholds": str(output_dir / "decision_thresholds.parquet"),
            "model_selection_summary": str(output_dir / "model_selection_summary.parquet"),
            "oof_fold_summary": str(output_dir / "oof_fold_summary.parquet"),
            "selector_performance_summary": str(
                output_dir / "selector_performance_summary.parquet"
            ),
            "nested_oof_predictions": str(output_dir / "nested_oof_predictions.parquet"),
            "train_oof_predictions": str(output_dir / "train_oof_predictions.parquet"),
            "train_oof_behavior_only_predictions": str(
                output_dir / "train_oof_behavior_only_predictions.parquet"
            ),
            "validation_regression_summary": str(output_dir / "validation_regression_summary.parquet"),
            "validation_score_summary": str(output_dir / "validation_score_summary.parquet"),
            "validation_decision_summary": str(output_dir / "validation_decision_summary.parquet"),
            "validation_ranking_summary": str(output_dir / "validation_ranking_summary.parquet"),
            "validation_predictions": str(output_dir / "validation_predictions.parquet"),
            "validation_behavior_only_predictions": str(
                output_dir / "validation_behavior_only_predictions.parquet"
            ),
            "train_oof_pre_run_aas_selection": str(
                output_dir / "train_oof_pre_run_aas_selection.parquet"
            ),
            "validation_pre_run_aas_selection": str(
                output_dir / "validation_pre_run_aas_selection.parquet"
            ),
            "report": str(output_dir / "full_decision_model_training_report.md"),
            "summary": str(output_dir / "full_decision_model_training_summary.json"),
        },
        "data_leakage_check": {
            "function_id_split_overlap": [],
            "landscape_family_overlap_is_expected": True,
            "decision_input_uses_only_behavior_features": True,
            "metadata_used_as_input": False,
            "algorithm_identifier_used_as_input": False,
            "query_features_used_as_decision_model_input": False,
            "query_features_used_by_query_selector": True,
            "model_selection_uses_validation_rows": False,
            "threshold_selection_uses_validation_rows": False,
            "oof_preprocessing_is_fit_within_each_cv_group_fold": True,
            "oof_sbs_selectors_and_utility_are_rebuilt_within_each_cv_group_fold": True,
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
        output_dir / "selector_performance_summary.csv",
        output_dir / "selector_performance_summary.parquet",
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
        output_dir / "train_oof_behavior_only_predictions.parquet",
        output_dir / "validation_behavior_only_predictions.parquet",
        output_dir / "train_oof_pre_run_aas_selection.parquet",
        output_dir / "validation_pre_run_aas_selection.parquet",
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


def _formal_feature_columns(feature_group: str) -> list[str]:
    if feature_group not in BEHAVIOR_FEATURE_GROUPS:
        raise ValueError(f"unknown feature group: {feature_group}")
    columns = list(BEHAVIOR_FEATURE_GROUPS[feature_group])
    missing_from_contract = sorted(set(columns).difference(SELECTOR_BEHAVIOR_FEATURE_COLUMNS))
    if missing_from_contract:
        raise ValueError(
            f"feature group columns are outside the Decision behavior contract: {missing_from_contract}"
        )
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


def _opportunity_view(frame: pd.DataFrame, opportunity_scope: str) -> pd.DataFrame:
    if opportunity_scope == ALL_ACCEPTED_OPPORTUNITIES:
        view = frame.copy()
    elif opportunity_scope == MILESTONE_ONLY_OPPORTUNITIES:
        if "is_budget_milestone" not in frame.columns:
            raise ValueError("milestone-only Decision data require is_budget_milestone")
        view = frame.loc[frame["is_budget_milestone"].astype(bool)].copy()
        if view.empty:
            raise ValueError("milestone-only Decision data contain no budget milestones")
        if not view["is_budget_milestone"].astype(bool).all():
            raise RuntimeError("milestone-only opportunity filtering failed")
    else:
        raise ValueError(f"unsupported opportunity_scope: {opportunity_scope}")
    return view.reset_index(drop=True)


def _validate_formal_feature_groups() -> None:
    expected_counts = {
        "T0": 1,
        "B1": 19,
        "B2": 25,
        "B2+Motion": 28,
        "B2+Maturity": 28,
        "B3": 31,
    }
    groups: list[frozenset[str]] = []
    for name, expected_count in expected_counts.items():
        columns = tuple(BEHAVIOR_FEATURE_GROUPS[name])
        if len(columns) != expected_count or len(set(columns)) != expected_count:
            raise RuntimeError(f"formal feature group {name} must have {expected_count} unique inputs")
        groups.append(frozenset(columns))
    if tuple(BEHAVIOR_FEATURE_GROUPS["B1"]) != tuple(BEHAVIOR_FEATURE_GROUPS["B2"][:19]):
        raise RuntimeError("B2 must extend B1 in frozen column order")
    if tuple(BEHAVIOR_FEATURE_GROUPS["B2"]) != tuple(BEHAVIOR_FEATURE_GROUPS["B3"][:25]):
        raise RuntimeError("B3 must extend B2 in frozen column order")
    if not set(BEHAVIOR_FEATURE_GROUPS["B2"]).issubset(BEHAVIOR_FEATURE_GROUPS["B2+Motion"]):
        raise RuntimeError("B2+Motion must contain all B2 inputs")
    if not set(BEHAVIOR_FEATURE_GROUPS["B2"]).issubset(BEHAVIOR_FEATURE_GROUPS["B2+Maturity"]):
        raise RuntimeError("B2+Maturity must contain all B2 inputs")
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


def _check_function_split(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    overlap = sorted(
        set(train["function_id"].astype(str)).intersection(
            validation["function_id"].astype(str)
        )
    )
    if overlap:
        raise ValueError(f"train and validation function IDs must be disjoint: {overlap}")


def _normalize_expected_dimensions(expected_dimensions: Sequence[int]) -> tuple[int, ...]:
    dimensions = tuple(sorted(int(value) for value in expected_dimensions))
    if not dimensions or len(dimensions) != len(set(dimensions)):
        raise ValueError("expected_dimensions must be non-empty and unique")
    if any(value <= 0 for value in dimensions):
        raise ValueError("expected_dimensions must contain positive integers")
    return dimensions


def _check_complete_dimension_coverage(
    frame: pd.DataFrame,
    *,
    expected_dimensions: Sequence[int],
    context: str,
) -> tuple[int, ...]:
    required = {"function_id", "dimension"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing dimension-coverage columns: {missing}")
    expected = _normalize_expected_dimensions(expected_dimensions)
    if frame.empty:
        raise ValueError(f"{context} is empty")
    dimensions = pd.to_numeric(frame["dimension"], errors="coerce")
    dimension_values = dimensions.to_numpy(dtype=float)
    if (
        dimensions.isna().any()
        or not np.isfinite(dimension_values).all()
        or not np.equal(dimension_values, np.floor(dimension_values)).all()
    ):
        raise ValueError(f"{context} dimensions must be finite integers")
    normalized = frame.assign(dimension=dimensions.astype(int))
    observed_functions = tuple(sorted(normalized["function_id"].astype(str).unique()))
    if not observed_functions:
        raise ValueError(f"{context} contains no functions")
    incomplete: dict[str, tuple[int, ...]] = {}
    for function_id, function_frame in normalized.groupby(
        "function_id", sort=True, dropna=False
    ):
        observed = tuple(sorted(function_frame["dimension"].astype(int).unique()))
        if observed != expected:
            incomplete[str(function_id)] = observed
    if incomplete:
        details = "; ".join(
            f"{function_id}={list(observed)}"
            for function_id, observed in incomplete.items()
        )
        raise ValueError(
            f"{context} must cover exactly the frozen dimensions {list(expected)} "
            f"for every function; observed {details}"
        )
    return expected


def _target_for_objective(frame: pd.DataFrame, objective: str) -> np.ndarray:
    return _target_for_columns(
        frame,
        objective,
        target_column=TARGET_COLUMN,
        auxiliary_label_column=AUXILIARY_LABEL_COLUMN,
    )


def _target_for_columns(
    frame: pd.DataFrame,
    objective: str,
    *,
    target_column: str,
    auxiliary_label_column: str,
) -> np.ndarray:
    if objective == "classification":
        target = frame[auxiliary_label_column].to_numpy(dtype=bool).astype(int)
        return target
    if objective == "regression":
        return frame[target_column].to_numpy(dtype=float)
    raise ValueError(f"unknown Decision model objective: {objective}")


def _fit_model(
    model: Pipeline,
    train: pd.DataFrame,
    feature_columns: list[str],
    objective: str,
) -> tuple[Pipeline, bool, int, int]:
    return _fit_model_for_target(
        model,
        train,
        feature_columns,
        objective,
        target_column=TARGET_COLUMN,
        auxiliary_label_column=AUXILIARY_LABEL_COLUMN,
    )


def _fit_model_for_target(
    model: Pipeline,
    train: pd.DataFrame,
    feature_columns: list[str],
    objective: str,
    *,
    target_column: str,
    auxiliary_label_column: str,
) -> tuple[Pipeline, bool, int, int]:
    required = {target_column, auxiliary_label_column, *feature_columns}
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"Decision fit target view is missing columns: {missing}")
    target_values = train[target_column].to_numpy(dtype=float)
    labels = train[auxiliary_label_column].to_numpy(dtype=bool)
    if not np.isfinite(target_values).all() or not np.array_equal(
        labels,
        target_values > 0.0,
    ):
        raise ValueError(
            f"{auxiliary_label_column} must equal finite {target_column} > 0"
        )
    target = _target_for_columns(
        train,
        objective,
        target_column=target_column,
        auxiliary_label_column=auxiliary_label_column,
    )
    positive_rows = (
        int(np.sum(target))
        if objective == "classification"
        else int(np.sum(target_values > 0.0))
    )
    negative_rows = int(len(target) - positive_rows)
    sample_weight = cluster_balanced_row_weights(train)
    uses_constant_classifier = False
    if objective == "classification" and len(np.unique(target)) < 2:
        model = Pipeline(
            [
                ("imputer", clone(model.named_steps["imputer"])),
                ("scaler", clone(model.named_steps["scaler"])),
                (
                    "classifier",
                    ConstantBinaryClassifier(
                        float(np.average(target.astype(float), weights=sample_weight))
                    ),
                ),
            ]
        )
        uses_constant_classifier = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_pipeline_with_weights(
            model,
            train[feature_columns],
            target,
            sample_weight,
        )
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
    groups = (
        frame["cv_group_id"].astype(str).to_numpy()
        if "cv_group_id" in frame.columns
        else frame["family"].astype(str).to_numpy()
    )
    unique_groups = np.unique(groups)
    n_splits = min(int(requested_folds), len(unique_groups))
    if n_splits < 2:
        raise ValueError(f"{fold_role} requires at least two CV groups")
    scores = np.full(len(frame), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold_index, (fit_index, holdout_index) in enumerate(splitter.split(frame, groups=groups)):
        fit_frame = frame.iloc[fit_index]
        holdout_frame = frame.iloc[holdout_index]
        group_col = "cv_group_id" if "cv_group_id" in frame.columns else "family"
        fit_families = sorted(set(fit_frame[group_col].astype(str)))
        holdout_families = sorted(set(holdout_frame[group_col].astype(str)))
        overlap = sorted(set(fit_families).intersection(holdout_families))
        if overlap:
            raise RuntimeError(f"CV-group OOF fold overlap: {overlap}")
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


def _first_trigger_policy_arrays(
    frame: pd.DataFrame,
    *,
    scores: np.ndarray,
    observed: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    observed = np.asarray(observed, dtype=float).reshape(-1)
    if len(scores) != len(frame) or len(observed) != len(frame):
        raise ValueError("first-trigger policy requires aligned frame, score, and Utility arrays")
    if not frame.index.is_unique:
        raise ValueError("first-trigger policy requires a unique frame index")
    calls = np.zeros(len(frame), dtype=bool)
    decision_utility = np.zeros(len(frame), dtype=float)
    for _, run_frame in _iter_decision_run_groups(frame):
        ordered = _ordered_decision_run_frame(run_frame)
        positions = frame.index.get_indexer(ordered.index)
        if (positions < 0).any():
            raise RuntimeError("ordered run rows are not aligned with the first-trigger frame")
        hit = np.flatnonzero(scores[positions] > float(threshold))
        if hit.size == 0:
            continue
        trigger_position = int(positions[int(hit[0])])
        calls[trigger_position] = True
        decision_utility[trigger_position] = float(observed[trigger_position])
    return calls, decision_utility


def _first_trigger_run_summary(
    frame: pd.DataFrame,
    *,
    call_column: str,
    utility_column: str,
) -> pd.DataFrame:
    required = {call_column, utility_column, TARGET_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"first-trigger run summary missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for run_key, run_frame in _iter_decision_run_groups(frame):
        ordered = _ordered_decision_run_frame(run_frame)
        calls = ordered[call_column].to_numpy(dtype=bool)
        if int(np.sum(calls)) > 1:
            raise ValueError("a first-trigger policy may call the query at most once per trajectory")
        utilities = ordered[utility_column].to_numpy(dtype=float)
        observed = ordered[TARGET_COLUMN].to_numpy(dtype=float)
        if not np.isfinite(utilities).all() or not np.isfinite(observed).all():
            raise ValueError("first-trigger run summaries require finite Utility values")
        called = bool(np.any(calls))
        call_utility = float(utilities[calls][0]) if called else 0.0
        if called and not np.isclose(call_utility, float(observed[calls][0]), rtol=0.0, atol=EPS):
            raise ValueError("first-trigger decision Utility must equal observed Utility at the called state")
        if not np.allclose(utilities[~calls], 0.0, rtol=0.0, atol=EPS):
            raise ValueError("states other than the first trigger must contribute zero policy Utility")
        rows.append(
            {
                **dict(zip(RUN_KEY_COLUMNS, run_key, strict=True)),
                "run_key": "|".join(str(value) for value in run_key),
                "called": called,
                "call_utility": call_utility,
                "call_positive": bool(called and call_utility > 0.0),
                "unhelpful_call": bool(called and call_utility <= 0.0),
                "best_available_positive_utility": float(max(0.0, float(np.max(observed)))),
                "has_positive_opportunity": bool(np.any(observed > 0.0)),
                "first_opportunity_utility": float(observed[0]),
            }
        )
    if not rows:
        raise ValueError("first-trigger run summary requires at least one trajectory")
    return pd.DataFrame(rows)


def _function_balanced_run_mean(
    run_summary: pd.DataFrame,
    value_column: str,
    *,
    expected_dimensions: Sequence[int],
) -> float:
    required = {"function_id", "dimension", "problem_id", value_column}
    missing = sorted(required.difference(run_summary.columns))
    if missing:
        raise ValueError(f"function-balanced run summary is missing columns: {missing}")
    _check_complete_dimension_coverage(
        run_summary,
        expected_dimensions=expected_dimensions,
        context="function-balanced run summary",
    )
    problem_means = run_summary.groupby(
        ["function_id", "dimension", "problem_id"],
        as_index=False,
        dropna=False,
    )[value_column].mean()
    dimension_means = problem_means.groupby(
        ["function_id", "dimension"],
        as_index=False,
        dropna=False,
    )[value_column].mean()
    function_means = dimension_means.groupby("function_id", dropna=False)[value_column].mean()
    if function_means.empty or not np.isfinite(function_means.to_numpy(dtype=float)).all():
        raise ValueError("function-balanced mean requires finite values and non-empty functions")
    return float(function_means.mean())


def _train_family_values(inputs: PreparedNestedLearningInputs) -> tuple[str, ...]:
    group_column = (
        "cv_group_id"
        if "cv_group_id" in inputs.query_adjusted_states.columns
        else "family"
    )
    return tuple(
        sorted(
            set(
                inputs.query_adjusted_states.loc[
                    inputs.query_adjusted_states["split"].astype(str).eq(TRAIN_SPLIT),
                    group_column,
                ].astype(str)
            )
        )
    )


def _end_to_end_family_oof_scores(
    *,
    spec: DecisionModelSpec,
    inputs: PreparedNestedLearningInputs,
    feature_columns: list[str],
    requested_folds: int,
    fold_role: str,
    opportunity_scope: str,
    target_column: str,
    auxiliary_label_column: str,
    expected_dimensions: Sequence[int],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    partitions = cv_group_fold_partitions(
        cv_groups=_train_family_values(inputs),
        requested_folds=requested_folds,
    )
    holdout_frames: list[pd.DataFrame] = []
    holdout_scores: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_index, (fit_families, holdout_families) in enumerate(partitions):
        selector_fold_role = f"train_outer_{len(partitions)}_fold_{fold_index}"
        views = build_fold_learning_views(
            inputs=inputs,
            fit_families=fit_families,
            holdout_families=holdout_families,
            fold_role=selector_fold_role,
        )
        fit_frame = _opportunity_view(views.fit_labels, opportunity_scope)
        holdout_frame = _opportunity_view(views.holdout_labels, opportunity_scope)
        fitted, used_constant, positive_rows, negative_rows = _fit_model_for_target(
            clone(spec.estimator),
            fit_frame,
            feature_columns,
            spec.objective,
            target_column=target_column,
            auxiliary_label_column=auxiliary_label_column,
        )
        scores = _predict_model(fitted, holdout_frame, feature_columns)
        holdout_frame = holdout_frame.copy()
        holdout_frame["oof_fold"] = int(fold_index)
        holdout_frame["oof_selector_fold_role"] = selector_fold_role
        holdout_frames.append(holdout_frame)
        holdout_scores.append(scores)
        fold_rows.append(
            {
                "model_name": spec.model_name,
                "model_family": spec.model_family,
                "objective": spec.objective,
                "target_column": target_column,
                "fold_role": fold_role,
                "selector_fold_role": selector_fold_role,
                "outer_fold": None,
                "fold_index": int(fold_index),
                "n_splits": int(len(partitions)),
                "fit_rows": int(len(fit_frame)),
                "holdout_rows": int(len(holdout_frame)),
                "fit_positive_utility_rows": int(positive_rows),
                "fit_negative_utility_rows": int(negative_rows),
                "uses_constant_classifier": bool(used_constant),
                "fit_families": ",".join(fit_families),
                "holdout_families": ",".join(holdout_families),
                "family_overlap_count": 0,
                "validation_rows_used": 0,
                "upstream_components_refit_within_fold": True,
            }
        )
    frame = pd.concat(holdout_frames, ignore_index=True)
    scores = np.concatenate(holdout_scores).astype(float, copy=False)
    if len(frame) != len(scores) or not np.isfinite(scores).all():
        raise RuntimeError(f"{fold_role} did not produce one finite score per held-out row")
    group_col = "cv_group_id" if "cv_group_id" in frame.columns else "family"
    if set(frame[group_col].astype(str)) != set(_train_family_values(inputs)):
        raise RuntimeError(f"{fold_role} does not cover every BBOB-train CV group")
    return frame, scores, pd.DataFrame(fold_rows)


def _nested_family_oof_predictions(
    *,
    spec: DecisionModelSpec,
    inputs: PreparedNestedLearningInputs,
    feature_columns: list[str],
    opportunity_scope: str,
    expected_dimensions: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer_partitions = cv_group_fold_partitions(
        cv_groups=_train_family_values(inputs),
        requested_folds=OUTER_OOF_FOLDS,
    )
    prediction_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    for outer_fold, (outer_fit_families, outer_holdout_families) in enumerate(
        outer_partitions
    ):
        outer_role = f"train_outer_{len(outer_partitions)}_fold_{outer_fold}"
        outer_views = build_fold_learning_views(
            inputs=inputs,
            fit_families=outer_fit_families,
            holdout_families=outer_holdout_families,
            fold_role=outer_role,
        )
        outer_fit = _opportunity_view(outer_views.fit_labels, opportunity_scope)
        outer_holdout = _opportunity_view(
            outer_views.holdout_labels,
            opportunity_scope,
        )

        inner_frames: list[pd.DataFrame] = []
        inner_scores: list[np.ndarray] = []
        inner_partitions = cv_group_fold_partitions(
            cv_groups=outer_fit_families,
            requested_folds=INNER_OOF_FOLDS,
        )
        for inner_fold, (inner_fit_families, inner_holdout_families) in enumerate(
            inner_partitions
        ):
            inner_role = f"{outer_role}_inner_{len(inner_partitions)}_fold_{inner_fold}"
            inner_views = build_fold_learning_views(
                inputs=inputs,
                fit_families=inner_fit_families,
                holdout_families=inner_holdout_families,
                fold_role=inner_role,
            )
            inner_fit = _opportunity_view(inner_views.fit_labels, opportunity_scope)
            inner_holdout = _opportunity_view(
                inner_views.holdout_labels,
                opportunity_scope,
            )
            inner_model, inner_constant, inner_positive, inner_negative = _fit_model(
                clone(spec.estimator),
                inner_fit,
                feature_columns,
                spec.objective,
            )
            inner_holdout_scores = _predict_model(
                inner_model,
                inner_holdout,
                feature_columns,
            )
            inner_frames.append(inner_holdout)
            inner_scores.append(inner_holdout_scores)
            fold_frames.append(
                pd.DataFrame(
                    [
                        {
                            "model_name": spec.model_name,
                            "model_family": spec.model_family,
                            "objective": spec.objective,
                            "target_column": TARGET_COLUMN,
                            "fold_role": "nested_inner_threshold",
                            "selector_fold_role": inner_role,
                            "outer_fold": int(outer_fold),
                            "fold_index": int(inner_fold),
                            "n_splits": int(len(inner_partitions)),
                            "fit_rows": int(len(inner_fit)),
                            "holdout_rows": int(len(inner_holdout)),
                            "fit_positive_utility_rows": int(inner_positive),
                            "fit_negative_utility_rows": int(inner_negative),
                            "uses_constant_classifier": bool(inner_constant),
                            "fit_families": ",".join(inner_fit_families),
                            "holdout_families": ",".join(inner_holdout_families),
                            "family_overlap_count": 0,
                            "validation_rows_used": 0,
                            "upstream_components_refit_within_fold": True,
                        }
                    ]
                )
            )
        inner_frame = pd.concat(inner_frames, ignore_index=True)
        inner_score_values = np.concatenate(inner_scores).astype(float, copy=False)
        threshold = _decision_threshold_from_scores(
            frame=inner_frame,
            scores=inner_score_values,
            observed=inner_frame[TARGET_COLUMN].to_numpy(dtype=float),
            expected_dimensions=expected_dimensions,
        )
        fitted, used_constant, positive_rows, negative_rows = _fit_model(
            clone(spec.estimator),
            outer_fit,
            feature_columns,
            spec.objective,
        )
        scores = _predict_model(fitted, outer_holdout, feature_columns)
        calls, decision_utility = _first_trigger_policy_arrays(
            outer_holdout,
            scores=scores,
            observed=outer_holdout[TARGET_COLUMN].to_numpy(dtype=float),
            threshold=threshold,
        )
        output = outer_holdout[
            list(METADATA_COLUMNS) + [TARGET_COLUMN, AUXILIARY_LABEL_COLUMN]
        ].copy()
        output.insert(0, "data_split", "nested_train_oof")
        output.insert(1, "model_name", spec.model_name)
        output.insert(2, "model_family", spec.model_family)
        output.insert(3, "objective", spec.objective)
        output["outer_fold"] = int(outer_fold)
        output["decision_score"] = scores
        output["nested_oof_threshold"] = float(threshold)
        output["decision_run_query_nested_oof"] = calls
        output["decision_utility_nested_oof"] = decision_utility
        output["decision_policy_unit"] = "trajectory_first_trigger"
        output["selector_fold_role"] = outer_role
        prediction_frames.append(output)
        fold_frames.append(
            pd.DataFrame(
                [
                    {
                        "model_name": spec.model_name,
                        "model_family": spec.model_family,
                        "objective": spec.objective,
                        "target_column": TARGET_COLUMN,
                        "fold_role": "nested_outer_evaluation",
                        "selector_fold_role": outer_role,
                        "outer_fold": int(outer_fold),
                        "fold_index": int(outer_fold),
                        "n_splits": int(len(outer_partitions)),
                        "fit_rows": int(len(outer_fit)),
                        "holdout_rows": int(len(outer_holdout)),
                        "fit_positive_utility_rows": int(positive_rows),
                        "fit_negative_utility_rows": int(negative_rows),
                        "uses_constant_classifier": bool(used_constant),
                        "fit_families": ",".join(outer_fit_families),
                        "holdout_families": ",".join(outer_holdout_families),
                        "family_overlap_count": 0,
                        "validation_rows_used": 0,
                        "threshold": float(threshold),
                        "threshold_source": "inner_fold_specific_upstream_cv_group_oof",
                        "upstream_components_refit_within_fold": True,
                    }
                ]
            )
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    pred_group_col = "cv_group_id" if "cv_group_id" in predictions.columns else "family"
    if set(predictions[pred_group_col].astype(str)) != set(_train_family_values(inputs)):
        raise RuntimeError("nested OOF predictions do not cover every BBOB-train CV group")
    return predictions, pd.concat(fold_frames, ignore_index=True)


def _decision_threshold_from_scores(
    frame: pd.DataFrame,
    scores: np.ndarray,
    observed: np.ndarray,
    *,
    expected_dimensions: Sequence[int],
) -> float:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    observed = np.asarray(observed, dtype=float).reshape(-1)
    if len(scores) != len(observed) or len(scores) != len(frame) or not len(scores):
        raise ValueError("threshold fitting requires aligned non-empty frame, score, and Utility arrays")
    if not np.isfinite(scores).all() or not np.isfinite(observed).all():
        raise ValueError("threshold fitting requires finite scores and Utility values")

    thresholds = _threshold_candidates(np.unique(scores))
    utility_delta = np.zeros(len(thresholds) + 1, dtype=float)
    call_delta = np.zeros(len(thresholds) + 1, dtype=float)

    frozen_dimensions = _check_complete_dimension_coverage(
        frame,
        expected_dimensions=expected_dimensions,
        context="threshold-fitting Decision data",
    )
    run_table = frame[list(RUN_KEY_COLUMNS)].drop_duplicates().copy()
    runs_per_problem = run_table.groupby(
        ["function_id", "dimension", "problem_id"],
        dropna=False,
    ).size().rename("runs_per_problem")
    problems_per_dimension = (
        run_table[["function_id", "dimension", "problem_id"]]
        .drop_duplicates()
        .groupby(["function_id", "dimension"], dropna=False)
        .size()
        .rename("problems_per_dimension")
    )
    function_count = int(run_table["function_id"].astype(str).nunique())
    if function_count <= 0:
        raise ValueError("threshold fitting requires at least one function")
    weighted_runs = run_table.merge(
        runs_per_problem.reset_index(),
        on=["function_id", "dimension", "problem_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        problems_per_dimension.reset_index(),
        on=["function_id", "dimension"],
        how="left",
        validate="many_to_one",
    )
    weighted_runs["run_weight"] = 1.0 / (
        float(function_count)
        * float(len(frozen_dimensions))
        * weighted_runs["problems_per_dimension"].to_numpy(dtype=float)
        * weighted_runs["runs_per_problem"].to_numpy(dtype=float)
    )
    run_weights = {
        tuple(row[column] for column in RUN_KEY_COLUMNS): float(row["run_weight"])
        for _, row in weighted_runs.iterrows()
    }
    if not np.isclose(sum(run_weights.values()), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("function-balanced threshold weights must sum to one")

    for run_key, run_frame in _iter_decision_run_groups(frame):
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
                weighted_utility = float(utility) * run_weights[tuple(run_key)]
                utility_delta[start] += weighted_utility
                utility_delta[end] -= weighted_utility
                call_delta[start] += run_weights[tuple(run_key)]
                call_delta[end] -= run_weights[tuple(run_key)]
            if float(score) > max_previous_score:
                max_previous_score = float(score)

    decision_utility_sums = np.cumsum(utility_delta[:-1])
    weighted_call_rates = np.cumsum(call_delta[:-1])
    best_utility = float(np.max(decision_utility_sums))
    utility_ties = np.flatnonzero(
        np.isclose(decision_utility_sums, best_utility, rtol=0.0, atol=1e-12)
    )
    minimum_call_rate = float(np.min(weighted_call_rates[utility_ties]))
    call_ties = utility_ties[
        np.isclose(
            weighted_call_rates[utility_ties],
            minimum_call_rate,
            rtol=0.0,
            atol=1e-12,
        )
    ]
    best_threshold = float(np.max(thresholds[call_ties]))

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
        calls, decision_utility = _first_trigger_policy_arrays(
            frame,
            scores=scores,
            observed=frame[TARGET_COLUMN].to_numpy(dtype=float),
            threshold=float(threshold),
        )
        output[f"decision_run_query_{threshold_mode}"] = calls
        output[f"decision_utility_{threshold_mode}"] = decision_utility
    output["decision_policy_unit"] = "trajectory_first_trigger"
    return output


def _target_prediction_frame(
    *,
    frame: pd.DataFrame,
    scores: np.ndarray,
    thresholds: dict[str, float],
    model_name: str,
    model_family: str,
    objective: str,
    data_split: str,
    target_column: str,
    auxiliary_label_column: str,
    policy_target: str,
) -> pd.DataFrame:
    output = frame[
        list(METADATA_COLUMNS) + [target_column, auxiliary_label_column]
    ].copy()
    output.insert(0, "data_split", data_split)
    output.insert(1, "model_name", model_name)
    output.insert(2, "model_family", model_family)
    output.insert(3, "objective", objective)
    output.insert(4, "policy_target", policy_target)
    if policy_target == "behavior_only_full_budget":
        behavior_relation_columns = {
            "selected_algorithm": "behavior_selected_algorithm",
            "selected_action": "behavior_selected_action",
            "selected_equals_default": "behavior_selected_equals_default",
            "selected_equals_prefix": "behavior_selected_equals_prefix",
            "handoff_required": "behavior_handoff_required",
            "handoff_type": "behavior_handoff_type",
        }
        missing_behavior = sorted(
            set(behavior_relation_columns.values()).difference(frame.columns)
        )
        if missing_behavior:
            raise ValueError(
                "Behavior-only prediction output is missing action-relation columns: "
                f"{missing_behavior}"
            )
        for standard_column, behavior_column in behavior_relation_columns.items():
            output[standard_column] = frame[behavior_column].to_numpy()
        output["query_transition_mode"] = frame["behavior_handoff_type"].astype(str).to_numpy()
    output["decision_score"] = np.asarray(scores, dtype=float)
    for threshold_mode, threshold in thresholds.items():
        calls, decision_utility = _first_trigger_policy_arrays(
            frame,
            scores=scores,
            observed=frame[target_column].to_numpy(dtype=float),
            threshold=float(threshold),
        )
        output[f"decision_run_query_{threshold_mode}"] = calls
        output[f"decision_utility_{threshold_mode}"] = decision_utility
    output["decision_policy_unit"] = "trajectory_first_trigger"
    return output


def _nested_model_selection_summary(
    predictions: pd.DataFrame,
    model_specs: tuple[DecisionModelSpec, ...],
    *,
    expected_dimensions: Sequence[int],
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
        run_summary = _first_trigger_run_summary(
            frame,
            call_column="decision_run_query_nested_oof",
            utility_column="decision_utility_nested_oof",
        )
        calls = run_summary["called"].to_numpy(dtype=bool)
        decision_utility = run_summary["call_utility"].to_numpy(dtype=float)
        captured_positive = run_summary["call_positive"].to_numpy(dtype=bool)
        positive_utility_sum = float(np.sum(run_summary["best_available_positive_utility"]))
        captured_positive_utility_sum = float(np.sum(decision_utility[captured_positive]))
        row = {
            "model_name": spec.model_name,
            "model_family": spec.model_family,
            "objective": spec.objective,
            "candidate_order": int(candidate_order[spec.model_name]),
            MODEL_SELECTION_METRIC: _function_balanced_run_mean(
                run_summary,
                "call_utility",
                expected_dimensions=expected_dimensions,
            ),
            "nested_cv_group_oof_decision_utility_sum": float(np.sum(decision_utility)),
            "nested_cv_group_oof_runs": int(len(run_summary)),
            "nested_cv_group_oof_query_call_rate": float(np.mean(calls)),
            "nested_cv_group_oof_precision_u_gt_zero_under_calls": float(
                np.sum(captured_positive) / max(int(np.sum(calls)), 1)
            ),
            "nested_cv_group_oof_utility_capture_rate": (
                captured_positive_utility_sum / positive_utility_sum if positive_utility_sum > 0.0 else 0.0
            ),
            "nested_cv_group_oof_auroc": _finite_binary_metric(positive, scores, roc_auc_score),
            "nested_cv_group_oof_average_precision": _finite_binary_metric(
                positive,
                scores,
                average_precision_score,
            ),
            "nested_cv_group_oof_spearman": _finite_metric(
                lambda: pd.Series(observed).corr(pd.Series(scores), method="spearman")
            ),
            "nested_cv_group_oof_rmse": (
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
    call_column = f"decision_run_query_{threshold_mode}"
    utility_column = f"decision_utility_{threshold_mode}"
    run_summary = _first_trigger_run_summary(
        frame,
        call_column=call_column,
        utility_column=utility_column,
    )
    calls = run_summary["called"].to_numpy(dtype=bool)
    decision_utility = run_summary["call_utility"].to_numpy(dtype=float)
    captured_positive = run_summary["call_positive"].to_numpy(dtype=bool)
    unhelpful_calls = run_summary["unhelpful_call"].to_numpy(dtype=bool)
    positive_runs = int(np.sum(run_summary["has_positive_opportunity"].to_numpy(dtype=bool)))
    positive_utility_sum = float(np.sum(run_summary["best_available_positive_utility"]))
    captured_positive_utility_sum = float(np.sum(decision_utility[captured_positive]))
    call_runs = int(np.sum(calls))
    return {
        **_common_fields(frame, layer, group, model_name, model_family),
        "threshold_mode": threshold_mode,
        "threshold": float(threshold),
        "policy_unit": "trajectory_first_trigger",
        "runs": int(len(run_summary)),
        "decision_query_call_runs": call_runs,
        "decision_query_call_rate": float(np.mean(calls)),
        "mean_observed_utility_under_calls": float(np.mean(decision_utility[calls])) if call_runs else 0.0,
        "positive_runs_captured": int(np.sum(captured_positive)),
        "positive_run_capture_rate": float(np.sum(captured_positive) / max(positive_runs, 1)),
        "utility_capture_rate": (
            captured_positive_utility_sum / positive_utility_sum if positive_utility_sum > 0.0 else 0.0
        ),
        "precision_u_gt_zero_under_calls": float(np.sum(captured_positive) / max(call_runs, 1)),
        "unhelpful_call_runs": int(np.sum(unhelpful_calls)),
        "unhelpful_call_rate_within_calls": float(np.sum(unhelpful_calls) / max(call_runs, 1)),
        "unhelpful_call_share_all_runs": float(np.mean(unhelpful_calls)),
        "unhelpful_call_cost_sum": float(-np.sum(decision_utility[unhelpful_calls])),
        "decision_utility_sum": float(np.sum(decision_utility)),
        "decision_mean_utility": float(np.mean(decision_utility)),
        "always_query_mean_utility": float(np.mean(run_summary["first_opportunity_utility"])),
        "never_query_mean_utility": 0.0,
        "run_hindsight_max_positive_utility_mean": float(
            np.mean(run_summary["best_available_positive_utility"])
        ),
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
            "- Model selection uses nested CV-group (function-ID) OOF decision utility on BBOB-train only.",
            "- Fixed thresholds use full BBOB-train CV-group OOF scores; validation is evaluation only.",
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
                        "positive_run_capture_rate",
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
                        "positive_run_capture_rate",
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
        description=(
            "Fit fold-specific SBS/Selectors/Utility labels and train the frozen "
            "three-candidate Decision protocol with nested CV-group OOF."
        )
    )
    parser.add_argument("--query-id", choices=sorted(LANDSCAPE_QUERY_SPECS), required=True)
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/phase1_bbob_train.yaml"),
    )
    parser.add_argument("--query-action-loss", type=Path, action="append", required=True)
    parser.add_argument("--behavior-action-loss", type=Path, action="append", required=True)
    parser.add_argument("--behavior", type=Path, action="append", required=True)
    parser.add_argument("--query-feature", type=Path, action="append", required=True)
    parser.add_argument("--complete-path-timing", type=Path, action="append", default=None)
    parser.add_argument("--pre-run-action-loss", type=Path, action="append", required=True)
    parser.add_argument(
        "--emit-replay-plan",
        type=Path,
        default=None,
        help=(
            "Write the fold/state/path plan after fitting fold-specific Selectors. "
            "The plan must be executed by a real decision-state-to-terminal replay runner."
        ),
    )
    parser.add_argument("--replay-plan-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--feature-group",
        choices=FORMAL_FEATURE_GROUPS,
        default="B3",
    )
    parser.add_argument(
        "--opportunity-scope",
        choices=OPPORTUNITY_SCOPES,
        default=ALL_ACCEPTED_OPPORTUNITIES,
    )
    parser.add_argument("--random-seed", type=int, default=1701)
    parser.add_argument("--target-column", choices=UTILITY_VALUE_COLUMNS, default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--auxiliary-label-column", choices=NEED_QUERY_COLUMNS, default=None)
    parser.add_argument(
        "--behavior-target-column",
        choices=BEHAVIOR_UTILITY_VALUE_COLUMNS,
        default=DEFAULT_BEHAVIOR_TARGET_COLUMN,
    )
    parser.add_argument(
        "--behavior-auxiliary-label-column",
        choices=NEED_BEHAVIOR_ONLY_COLUMNS,
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if FULL_TRAIN_OOF_FOLDS != OUTER_OOF_FOLDS:
        raise RuntimeError(
            "replay-plan roles require FULL_TRAIN_OOF_FOLDS == OUTER_OOF_FOLDS"
        )
    auxiliary_label_column = args.auxiliary_label_column or NEED_QUERY_COLUMNS[
        UTILITY_VALUE_COLUMNS.index(args.target_column)
    ]
    behavior_auxiliary_label_column = (
        args.behavior_auxiliary_label_column
        or NEED_BEHAVIOR_ONLY_COLUMNS[
            BEHAVIOR_UTILITY_VALUE_COLUMNS.index(args.behavior_target_column)
        ]
    )
    config = load_config(args.train_config)
    performance = read_performance(config, None, None)
    prepared_inputs = prepare_nested_learning_inputs(
        query_id=args.query_id,
        performance=performance,
        query_action_loss_paths=list(args.query_action_loss),
        behavior_action_loss_paths=list(args.behavior_action_loss),
        behavior_paths=list(args.behavior),
        query_feature_paths=list(args.query_feature),
        complete_path_timing_paths=(
            None
            if args.complete_path_timing is None
            else list(args.complete_path_timing)
        ),
        log10_gap_floor=float(config["log10_gap_floor"]),
        log10_gap_cap=float(config["log10_gap_cap"]),
        pre_run_action_loss_paths=list(args.pre_run_action_loss),
    )
    replay_plan_path = args.emit_replay_plan
    if args.replay_plan_only and replay_plan_path is None:
        replay_plan_path = (
            Path("results/decision")
            / args.query_id
            / "fold_specific_selected_path_replay_plan.parquet"
        )
    if replay_plan_path is not None:
        if replay_plan_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"fold-specific replay plan already exists; pass --overwrite: {replay_plan_path}"
            )
        replay_plan = build_required_replay_plan(
            inputs=prepared_inputs,
            outer_folds=OUTER_OOF_FOLDS,
            inner_folds=INNER_OOF_FOLDS,
        )
        replay_plan_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(replay_plan, preserve_index=False),
            replay_plan_path,
        )
        print(f"wrote {len(replay_plan)} fold/state/path replay rows to {replay_plan_path}")
    if args.replay_plan_only:
        return
    if prepared_inputs.complete_path_timings is None:
        raise ValueError(
            "formal training requires --complete-path-timing from the executed "
            "fold-specific replay plan; action-component runtimes are not accepted"
        )

    default_output = (
        Path("results/decision")
        / args.query_id
        / "feature_group_ablation"
        / args.feature_group
        / args.opportunity_scope
    )
    train_full_decision_models(
        query_id=args.query_id,
        prepared_inputs=prepared_inputs,
        output_dir=args.output_dir or default_output,
        overwrite=args.overwrite,
        random_seed=args.random_seed,
        feature_group=args.feature_group,
        opportunity_scope=args.opportunity_scope,
        expected_dimensions=tuple(int(value) for value in config["dimensions"]),
        target_column=args.target_column,
        auxiliary_label_column=auxiliary_label_column,
        behavior_target_column=args.behavior_target_column,
        behavior_auxiliary_label_column=behavior_auxiliary_label_column,
    )


if __name__ == "__main__":
    main()
