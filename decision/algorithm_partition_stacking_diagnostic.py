from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from behavior.features import BEHAVIOR_FEATURE_COLUMNS


DEFAULT_DATASET_PATH = Path("results/decision/phase1_refined_sampling/materialized_training_data/decision_dataset.parquet")
DEFAULT_SCHEMA_PATH = Path("results/decision/phase1_refined_sampling/materialized_training_data/decision_dataset_schema.json")
DEFAULT_OUTPUT_DIR = Path("results/decision/phase1_refined_sampling/algorithm_partition_stacking_diagnostic")
TRAIN_SPLIT = "bbob_train"
VALIDATION_SPLIT = "bbob_validation"
TARGET_COLUMN = "u_ela_lamT_1"
ALGORITHMS = ("cmaes", "de", "pso", "shade")
STACK_SCORE_COLUMNS = tuple(f"score_from_{algorithm}_model" for algorithm in ALGORITHMS)
TOP_K_FRACTIONS = (0.05, 0.10, 0.20)
EPS = 1e-12


def run_algorithm_partition_stacking_diagnostic(
    *,
    dataset_path: Path,
    schema_path: Path,
    output_dir: Path,
    overwrite: bool,
    random_seed: int,
) -> dict[str, Any]:
    _check_output_paths(output_dir, overwrite)
    dataset = pq.read_table(dataset_path).to_pandas()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    feature_columns = _feature_columns(schema)
    _check_dataset(dataset, feature_columns)

    train = dataset[dataset["split"] == TRAIN_SPLIT].copy()
    validation = dataset[dataset["split"] == VALIDATION_SPLIT].copy()
    _check_train_validation(train, validation)

    global_model = _ridge_pipeline(random_seed)
    global_model.fit(train[feature_columns], train[TARGET_COLUMN])
    global_train_scores = global_model.predict(train[feature_columns]).astype(float)
    global_validation_scores = global_model.predict(validation[feature_columns]).astype(float)

    train_stack_scores, fold_summary = _family_oof_base_scores(
        train=train,
        feature_columns=feature_columns,
        random_seed=random_seed,
    )
    meta_model = Ridge(alpha=1.0, random_state=random_seed)
    meta_model.fit(train_stack_scores[list(STACK_SCORE_COLUMNS)], train[TARGET_COLUMN])
    stacked_train_scores = meta_model.predict(train_stack_scores[list(STACK_SCORE_COLUMNS)]).astype(float)
    scaled_meta_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0, random_state=random_seed)),
        ]
    )
    scaled_meta_model.fit(train_stack_scores[list(STACK_SCORE_COLUMNS)], train[TARGET_COLUMN])
    scaled_stacked_train_scores = scaled_meta_model.predict(train_stack_scores[list(STACK_SCORE_COLUMNS)]).astype(float)

    validation_stack_scores, base_domain_summary = _validation_base_scores(
        train=train,
        validation=validation,
        feature_columns=feature_columns,
        random_seed=random_seed,
    )
    stacked_validation_scores = meta_model.predict(validation_stack_scores[list(STACK_SCORE_COLUMNS)]).astype(float)
    scaled_stacked_validation_scores = scaled_meta_model.predict(validation_stack_scores[list(STACK_SCORE_COLUMNS)]).astype(float)

    stacked_train_oof_scores = _score_frame(
        train,
        data_split="train",
        model_name="algorithm_partition_stacked_ridge",
        decision_scores=stacked_train_scores,
        stack_scores=train_stack_scores,
    )
    scaled_stacked_train_oof_scores = _score_frame(
        train,
        data_split="train",
        model_name="algorithm_partition_stacked_scaled_ridge",
        decision_scores=scaled_stacked_train_scores,
        stack_scores=train_stack_scores,
    )
    stacked_validation_scores_frame = _score_frame(
        validation,
        data_split="validation",
        model_name="algorithm_partition_stacked_ridge",
        decision_scores=stacked_validation_scores,
        stack_scores=validation_stack_scores,
    )
    scaled_stacked_validation_scores_frame = _score_frame(
        validation,
        data_split="validation",
        model_name="algorithm_partition_stacked_scaled_ridge",
        decision_scores=scaled_stacked_validation_scores,
        stack_scores=validation_stack_scores,
    )
    global_validation_scores_frame = _score_frame(
        validation,
        data_split="validation",
        model_name="global_ridge_behavior",
        decision_scores=global_validation_scores,
        stack_scores=None,
    )

    regression_summary = _layer_metric_summary(
        validation=validation,
        model_scores={
            "global_ridge_behavior": global_validation_scores,
            "algorithm_partition_stacked_ridge": stacked_validation_scores,
            "algorithm_partition_stacked_scaled_ridge": scaled_stacked_validation_scores,
        },
        row_fn=_regression_row,
    )
    decision_summary = _layer_metric_summary(
        validation=validation,
        model_scores={
            "global_ridge_behavior": global_validation_scores,
            "algorithm_partition_stacked_ridge": stacked_validation_scores,
            "algorithm_partition_stacked_scaled_ridge": scaled_stacked_validation_scores,
        },
        row_fn=_decision_row,
    )
    ranking_summary = _ranking_summary(
        validation=validation,
        model_scores={
            "global_ridge_behavior": global_validation_scores,
            "algorithm_partition_stacked_ridge": stacked_validation_scores,
            "algorithm_partition_stacked_scaled_ridge": scaled_stacked_validation_scores,
        },
    )
    model_input_contract = _model_input_contract(feature_columns, fold_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(model_input_contract, output_dir / "model_input_contract")
    _write_frame(base_domain_summary, output_dir / "base_model_domain_summary")
    _write_frame(regression_summary, output_dir / "global_vs_stacked_regression_summary")
    _write_frame(decision_summary, output_dir / "global_vs_stacked_decision_summary")
    _write_frame(ranking_summary, output_dir / "global_vs_stacked_ranking_summary")
    _write_frame(fold_summary, output_dir / "oof_family_fold_summary")
    train_output = pd.concat([stacked_train_oof_scores, scaled_stacked_train_oof_scores], ignore_index=True)
    pq.write_table(pa.Table.from_pandas(train_output, preserve_index=False), output_dir / "stacked_train_oof_scores.parquet")
    validation_output = pd.concat(
        [global_validation_scores_frame, stacked_validation_scores_frame, scaled_stacked_validation_scores_frame],
        ignore_index=True,
    )
    pq.write_table(
        pa.Table.from_pandas(validation_output, preserve_index=False),
        output_dir / "stacked_validation_scores.parquet",
    )

    summary = {
        "experiment": "algorithm_partition_stacking_diagnostic",
        "dataset": str(dataset_path),
        "schema": str(schema_path),
        "target_column": TARGET_COLUMN,
        "feature_columns": feature_columns,
        "algorithms": list(ALGORITHMS),
        "stack_score_columns": list(STACK_SCORE_COLUMNS),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "oof_group_column": "family",
        "oof_folds": int(fold_summary["fold"].nunique()),
        "validation_rows_used_for_fit": 0,
        "main_complex_model_trained": False,
        "model_files_saved": False,
        "metadata_used_as_model_input": False,
        "algorithm_labels_used_as_global_or_meta_input": False,
        "diagnostic_boundary": (
            "prefix_algorithm is used for base-model partitioning and report grouping only; "
            "this is a diagnostic ablation, not the main deployable Decision Model."
        ),
        "outputs": {
            "model_input_contract": str(output_dir / "model_input_contract.parquet"),
            "base_model_domain_summary": str(output_dir / "base_model_domain_summary.parquet"),
            "regression_summary": str(output_dir / "global_vs_stacked_regression_summary.parquet"),
            "decision_summary": str(output_dir / "global_vs_stacked_decision_summary.parquet"),
            "ranking_summary": str(output_dir / "global_vs_stacked_ranking_summary.parquet"),
            "stacked_train_oof_scores": str(output_dir / "stacked_train_oof_scores.parquet"),
            "stacked_validation_scores": str(output_dir / "stacked_validation_scores.parquet"),
            "report": str(output_dir / "algorithm_partition_stacking_report.md"),
        },
    }
    summary_path = output_dir / "algorithm_partition_stacking_summary.json"
    report_path = output_dir / "algorithm_partition_stacking_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(
        _markdown_report(
            summary=summary,
            model_input_contract=model_input_contract,
            base_domain_summary=base_domain_summary,
            regression_summary=regression_summary,
            decision_summary=decision_summary,
            ranking_summary=ranking_summary,
        ),
        encoding="utf-8",
    )

    print(f"wrote algorithm-partition stacking summary to {summary_path}")
    print(f"wrote algorithm-partition stacking report to {report_path}")
    return summary


def _check_output_paths(output_dir: Path, overwrite: bool) -> None:
    outputs = (
        output_dir / "algorithm_partition_stacking_summary.json",
        output_dir / "algorithm_partition_stacking_report.md",
        output_dir / "model_input_contract.csv",
        output_dir / "model_input_contract.parquet",
        output_dir / "base_model_domain_summary.csv",
        output_dir / "base_model_domain_summary.parquet",
        output_dir / "global_vs_stacked_regression_summary.csv",
        output_dir / "global_vs_stacked_regression_summary.parquet",
        output_dir / "global_vs_stacked_decision_summary.csv",
        output_dir / "global_vs_stacked_decision_summary.parquet",
        output_dir / "global_vs_stacked_ranking_summary.csv",
        output_dir / "global_vs_stacked_ranking_summary.parquet",
        output_dir / "oof_family_fold_summary.csv",
        output_dir / "oof_family_fold_summary.parquet",
        output_dir / "stacked_train_oof_scores.parquet",
        output_dir / "stacked_validation_scores.parquet",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"algorithm-partition stacking outputs already exist; pass --overwrite: {existing[0]}")


def _feature_columns(schema: dict[str, Any]) -> list[str]:
    columns = list(schema.get("input_columns", []))
    if columns != list(BEHAVIOR_FEATURE_COLUMNS):
        raise ValueError("schema input_columns must exactly equal BEHAVIOR_FEATURE_COLUMNS")
    forbidden_fragments = ("ela", "function", "algorithm", "selected", "default", "family", "problem", "dimension")
    forbidden = [column for column in columns if any(fragment in column.lower() for fragment in forbidden_fragments)]
    if forbidden:
        raise ValueError(f"Decision input contains forbidden name fragments: {forbidden}")
    return columns


def _check_dataset(dataset: pd.DataFrame, feature_columns: list[str]) -> None:
    required = {
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
        TARGET_COLUMN,
        *feature_columns,
    }
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(f"materialized dataset missing required columns: {missing}")
    if set(dataset["split"].astype(str).unique()) != {TRAIN_SPLIT, VALIDATION_SPLIT}:
        raise ValueError(f"expected splits {TRAIN_SPLIT} and {VALIDATION_SPLIT}")
    if sorted(dataset["prefix_algorithm"].astype(str).unique().tolist()) != list(ALGORITHMS):
        raise ValueError(f"expected prefix algorithms {list(ALGORITHMS)}")
    target = dataset[TARGET_COLUMN].to_numpy(dtype=float)
    if dataset[TARGET_COLUMN].isna().any() or not np.isfinite(target).all():
        raise ValueError(f"{TARGET_COLUMN} must be non-null and finite")
    for column in feature_columns:
        values = pd.to_numeric(dataset[column], errors="coerce")
        non_null = values.notna()
        invalid = non_null & ~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
        if invalid.any():
            raise ValueError(f"non-null feature values must be finite: {column}")


def _check_train_validation(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    if train.empty:
        raise ValueError(f"missing train split rows: {TRAIN_SPLIT}")
    if validation.empty:
        raise ValueError(f"missing validation split rows: {VALIDATION_SPLIT}")
    for algorithm in ALGORITHMS:
        if train[train["prefix_algorithm"] == algorithm].empty:
            raise ValueError(f"missing train rows for prefix_algorithm={algorithm}")
        if validation[validation["prefix_algorithm"] == algorithm].empty:
            raise ValueError(f"missing validation rows for prefix_algorithm={algorithm}")


def _ridge_pipeline(random_seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0, random_state=random_seed)),
        ]
    )


def _family_oof_base_scores(
    *,
    train: pd.DataFrame,
    feature_columns: list[str],
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    families = np.array(sorted(train["family"].astype(str).unique().tolist()))
    n_splits = min(6, len(families))
    if n_splits < 2:
        raise ValueError("at least two train families are required for family-grouped OOF stacking")
    groups = train["family"].astype(str).to_numpy()
    fold_assignments = np.full(len(train), -1, dtype=int)
    score_matrix = pd.DataFrame(index=train.index)
    for column in STACK_SCORE_COLUMNS:
        score_matrix[column] = np.nan

    fold_rows = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (fit_positions, holdout_positions) in enumerate(splitter.split(train, groups=groups), start=1):
        fit_frame = train.iloc[fit_positions]
        holdout_frame = train.iloc[holdout_positions]
        holdout_families = sorted(holdout_frame["family"].astype(str).unique().tolist())
        fit_families = set(fit_frame["family"].astype(str).unique().tolist())
        if set(holdout_families).intersection(fit_families):
            raise ValueError("OOF fold family overlap detected")
        fold_assignments[holdout_positions] = fold
        row = {
            "fold": fold,
            "holdout_rows": int(len(holdout_frame)),
            "fit_rows_total": int(len(fit_frame)),
            "holdout_families": ",".join(holdout_families),
            "family_overlap_with_fit": False,
        }
        for algorithm, column in zip(ALGORITHMS, STACK_SCORE_COLUMNS, strict=True):
            algorithm_fit = fit_frame[fit_frame["prefix_algorithm"] == algorithm]
            if algorithm_fit.empty:
                raise ValueError(f"empty OOF fit rows for algorithm {algorithm} in fold {fold}")
            model = _ridge_pipeline(random_seed)
            model.fit(algorithm_fit[feature_columns], algorithm_fit[TARGET_COLUMN])
            score_matrix.loc[holdout_frame.index, column] = model.predict(holdout_frame[feature_columns]).astype(float)
            row[f"{algorithm}_fit_rows"] = int(len(algorithm_fit))
        fold_rows.append(row)

    if np.any(fold_assignments < 0):
        raise ValueError("some train rows did not receive OOF predictions")
    if score_matrix.isna().any().any():
        raise ValueError("OOF stack scores contain missing values")
    score_matrix.insert(0, "oof_fold", fold_assignments)
    return score_matrix, pd.DataFrame(fold_rows)


def _validation_base_scores(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_matrix = pd.DataFrame(index=validation.index)
    summary_rows = []
    for algorithm, column in zip(ALGORITHMS, STACK_SCORE_COLUMNS, strict=True):
        algorithm_train = train[train["prefix_algorithm"] == algorithm]
        algorithm_validation = validation[validation["prefix_algorithm"] == algorithm]
        model = _ridge_pipeline(random_seed)
        model.fit(algorithm_train[feature_columns], algorithm_train[TARGET_COLUMN])
        validation_scores = model.predict(validation[feature_columns]).astype(float)
        score_matrix[column] = validation_scores
        train_target = algorithm_train[TARGET_COLUMN].to_numpy(dtype=float)
        validation_target = algorithm_validation[TARGET_COLUMN].to_numpy(dtype=float)
        summary_rows.append(
            {
                "base_model": f"base_{algorithm}",
                "partition_column": "prefix_algorithm",
                "partition_value": algorithm,
                "train_rows": int(len(algorithm_train)),
                "validation_rows_same_prefix": int(len(algorithm_validation)),
                "train_u_gt_zero_rate": float(np.mean(train_target > 0.0)),
                "validation_same_prefix_u_gt_zero_rate": float(np.mean(validation_target > 0.0)),
                "model_input_columns": ",".join(feature_columns),
                "validation_rows_used_for_fit": 0,
            }
        )
    return score_matrix, pd.DataFrame(summary_rows)


def _score_frame(
    frame: pd.DataFrame,
    *,
    data_split: str,
    model_name: str,
    decision_scores: np.ndarray,
    stack_scores: pd.DataFrame | None,
) -> pd.DataFrame:
    result = frame[
        [
            "problem_id",
            "family",
            "dimension",
            "prefix_algorithm",
            "seed",
            "FE",
            "FE_ratio",
            "label_source",
            "default_algorithm",
            "selected_algorithm",
            TARGET_COLUMN,
        ]
    ].copy()
    result.insert(0, "data_split", data_split)
    result.insert(0, "model_name", model_name)
    result["decision_score"] = decision_scores
    if stack_scores is not None:
        for column in STACK_SCORE_COLUMNS:
            result[column] = stack_scores[column].to_numpy(dtype=float)
        if "oof_fold" in stack_scores.columns:
            result["oof_fold"] = stack_scores["oof_fold"].to_numpy(dtype=int)
    return result


def _layer_metric_summary(
    *,
    validation: pd.DataFrame,
    model_scores: dict[str, np.ndarray],
    row_fn: Any,
) -> pd.DataFrame:
    rows = []
    for layer, subsets in _evaluation_layers(validation).items():
        for group_label, mask in subsets:
            subset = validation.loc[mask]
            for model_name, scores in model_scores.items():
                rows.append(
                    row_fn(
                        layer=layer,
                        group_label=group_label,
                        model_name=model_name,
                        observed=subset[TARGET_COLUMN].to_numpy(dtype=float),
                        scores=scores[mask],
                        rows=len(subset),
                    )
                )
    return pd.DataFrame(rows)


def _evaluation_layers(validation: pd.DataFrame) -> dict[str, list[tuple[str, np.ndarray]]]:
    layers: dict[str, list[tuple[str, np.ndarray]]] = {
        "all_validation": [("all_validation", np.ones(len(validation), dtype=bool))],
        "label_source": [],
        "prefix_algorithm": [],
        "dimension": [],
        "FE_ratio": [],
    }
    for label_source in ("changed_algorithm", "same_algorithm"):
        layers["label_source"].append(
            (label_source, validation["label_source"].astype(str).to_numpy() == label_source)
        )
    for algorithm in ALGORITHMS:
        layers["prefix_algorithm"].append(
            (f"prefix_algorithm={algorithm}", validation["prefix_algorithm"].astype(str).to_numpy() == algorithm)
        )
    for dimension in sorted(validation["dimension"].astype(int).unique().tolist()):
        layers["dimension"].append((f"dimension={dimension}", validation["dimension"].to_numpy(dtype=int) == dimension))
    for ratio in sorted(validation["FE_ratio"].astype(float).unique().tolist()):
        layers["FE_ratio"].append((f"FE_ratio={ratio:.2f}", np.isclose(validation["FE_ratio"].to_numpy(dtype=float), ratio)))
    return layers


def _regression_row(
    *,
    layer: str,
    group_label: str,
    model_name: str,
    observed: np.ndarray,
    scores: np.ndarray,
    rows: int,
) -> dict[str, Any]:
    return {
        "layer": layer,
        "group_label": group_label,
        "model_name": model_name,
        "rows": int(rows),
        "u_gt_zero_rate": float(np.mean(observed > 0.0)),
        "mae": float(mean_absolute_error(observed, scores)),
        "rmse": float(mean_squared_error(observed, scores) ** 0.5),
        "r2": _finite_or_none(lambda: r2_score(observed, scores)),
        "pearson": _correlation(observed, scores, method="pearson"),
        "spearman": _correlation(observed, scores, method="spearman"),
    }


def _decision_row(
    *,
    layer: str,
    group_label: str,
    model_name: str,
    observed: np.ndarray,
    scores: np.ndarray,
    rows: int,
) -> dict[str, Any]:
    run = scores > 0.0
    positive = observed > 0.0
    captured_positive = observed[run & positive]
    positive_utility_sum = float(np.sum(observed[positive]))
    unhelpful = run & ~positive
    return {
        "layer": layer,
        "group_label": group_label,
        "model_name": model_name,
        "rows": int(rows),
        "call_rows": int(np.sum(run)),
        "call_rate": float(np.mean(run)),
        "called_mean_observed_utility": float(np.mean(observed[run])) if np.any(run) else 0.0,
        "captured_positive_rows": int(np.sum(run & positive)),
        "positive_row_capture_rate": float(np.mean(run[positive])) if np.any(positive) else 0.0,
        "captured_positive_utility_sum": float(np.sum(captured_positive)),
        "utility_capture_rate": (
            float(np.sum(captured_positive) / positive_utility_sum) if positive_utility_sum > 0.0 else 0.0
        ),
        "unhelpful_call_rows": int(np.sum(unhelpful)),
        "unhelpful_call_rate": float(np.mean(unhelpful)) if rows else 0.0,
        "unhelpful_call_cost_sum": float(-np.sum(observed[unhelpful])),
    }


def _ranking_summary(
    *,
    validation: pd.DataFrame,
    model_scores: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for layer, subsets in _evaluation_layers(validation).items():
        for group_label, mask in subsets:
            subset = validation.loc[mask]
            observed = subset[TARGET_COLUMN].to_numpy(dtype=float)
            positive = observed > 0.0
            positive_utility_sum = float(np.sum(observed[positive]))
            for model_name, scores in model_scores.items():
                subset_scores = scores[mask]
                order = np.lexsort(
                    (
                        subset["FE"].to_numpy(dtype=int),
                        subset["seed"].to_numpy(dtype=int),
                        subset["problem_id"].astype(str).to_numpy(),
                        -subset_scores,
                    )
                )
                for fraction in TOP_K_FRACTIONS:
                    top_k_rows = max(1, int(np.ceil(len(subset) * fraction)))
                    top_index = order[:top_k_rows]
                    top_positive = positive[top_index]
                    top_observed = observed[top_index]
                    rows.append(
                        {
                            "layer": layer,
                            "group_label": group_label,
                            "model_name": model_name,
                            "top_k_fraction": float(fraction),
                            "top_k_rows": int(top_k_rows),
                            "rows": int(len(subset)),
                            "u_gt_zero_rows": int(np.sum(positive)),
                            "u_gt_zero_rate": float(np.mean(positive)),
                            "top_k_u_gt_zero_rows": int(np.sum(top_positive)),
                            "top_k_positive_row_rate": float(np.mean(top_positive)),
                            "top_k_positive_row_lift": float(np.mean(top_positive) / max(float(np.mean(positive)), EPS)),
                            "top_k_positive_row_capture_rate": float(np.sum(top_positive) / max(int(np.sum(positive)), 1)),
                            "top_k_positive_utility_sum": float(np.sum(top_observed[top_positive])),
                            "top_k_utility_capture_rate": (
                                float(np.sum(top_observed[top_positive]) / positive_utility_sum)
                                if positive_utility_sum > 0.0
                                else 0.0
                            ),
                            "top_k_observed_u_mean": float(np.mean(top_observed)),
                        }
                    )
    return pd.DataFrame(rows)


def _model_input_contract(feature_columns: list[str], fold_summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "component": "global_ridge_behavior",
                "input_columns": ",".join(feature_columns),
                "uses_behavior_features": True,
                "uses_prefix_algorithm_as_input": False,
                "uses_algorithm_partitioning": False,
                "uses_selected_algorithm": False,
                "uses_default_algorithm": False,
                "uses_metadata_as_input": False,
                "validation_rows_used_for_fit": 0,
                "family_oof_used": False,
            },
            {
                "component": "algorithm_partition_base_models",
                "input_columns": ",".join(feature_columns),
                "uses_behavior_features": True,
                "uses_prefix_algorithm_as_input": False,
                "uses_algorithm_partitioning": True,
                "uses_selected_algorithm": False,
                "uses_default_algorithm": False,
                "uses_metadata_as_input": False,
                "validation_rows_used_for_fit": 0,
                "family_oof_used": True,
            },
            {
                "component": "stacked_meta_ridge",
                "input_columns": ",".join(STACK_SCORE_COLUMNS),
                "uses_behavior_features": False,
                "uses_prefix_algorithm_as_input": False,
                "uses_algorithm_partitioning": False,
                "uses_selected_algorithm": False,
                "uses_default_algorithm": False,
                "uses_metadata_as_input": False,
                "validation_rows_used_for_fit": 0,
                "family_oof_used": True,
                "train_only_preprocessing": "none",
            },
            {
                "component": "stacked_meta_scaled_ridge",
                "input_columns": ",".join(STACK_SCORE_COLUMNS),
                "uses_behavior_features": False,
                "uses_prefix_algorithm_as_input": False,
                "uses_algorithm_partitioning": False,
                "uses_selected_algorithm": False,
                "uses_default_algorithm": False,
                "uses_metadata_as_input": False,
                "validation_rows_used_for_fit": 0,
                "family_oof_used": True,
                "train_only_preprocessing": "StandardScaler fitted on train OOF base scores",
            },
            {
                "component": "family_oof_check",
                "input_columns": "family used only as OOF grouping key",
                "uses_behavior_features": False,
                "uses_prefix_algorithm_as_input": False,
                "uses_algorithm_partitioning": False,
                "uses_selected_algorithm": False,
                "uses_default_algorithm": False,
                "uses_metadata_as_input": False,
                "validation_rows_used_for_fit": 0,
                "family_oof_used": True,
                "oof_folds": int(fold_summary["fold"].nunique()),
                "oof_family_overlap_detected": bool(fold_summary["family_overlap_with_fit"].any()),
            },
        ]
    )


def _correlation(observed: np.ndarray, predicted: np.ndarray, *, method: str) -> float:
    if np.unique(predicted).size <= 1 or np.unique(observed).size <= 1:
        return 0.0
    value = pd.Series(observed).corr(pd.Series(predicted), method=method)
    if pd.isna(value):
        return 0.0
    return float(value)


def _finite_or_none(fn: Any) -> float | None:
    try:
        value = float(fn())
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return value


def _write_frame(frame: pd.DataFrame, path_without_suffix: Path) -> None:
    frame.to_csv(path_without_suffix.with_suffix(".csv"), index=False)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path_without_suffix.with_suffix(".parquet"))


def _markdown_report(
    *,
    summary: dict[str, Any],
    model_input_contract: pd.DataFrame,
    base_domain_summary: pd.DataFrame,
    regression_summary: pd.DataFrame,
    decision_summary: pd.DataFrame,
    ranking_summary: pd.DataFrame,
) -> str:
    overall_regression = regression_summary[regression_summary["layer"] == "all_validation"]
    changed_same_decision = decision_summary[
        (decision_summary["layer"] == "label_source")
        & decision_summary["group_label"].isin(["changed_algorithm", "same_algorithm"])
    ]
    top10 = ranking_summary[
        (ranking_summary["top_k_fraction"] == 0.10)
        & (
            (ranking_summary["layer"] == "all_validation")
            | (
                (ranking_summary["layer"] == "label_source")
                & ranking_summary["group_label"].isin(["changed_algorithm", "same_algorithm"])
            )
        )
    ]
    return "\n".join(
        [
            "# Algorithm-partition stacking diagnostic",
            "",
            "## Scope",
            "",
            "- Dual-track comparison: protocol-compliant global ridge baseline vs algorithm-partition stacking diagnostic.",
            "- The stacking track uses `prefix_algorithm` only for base-model partitioning and report grouping.",
            "- Meta model input is only the four base scores.",
            "- No model files were saved, and no main complex Decision Model was trained.",
            f"- OOF folds: {summary['oof_folds']} family-grouped folds.",
            "",
            "## Model input contract",
            "",
            _markdown_table(model_input_contract),
            "",
            "## Base model domain summary",
            "",
            _markdown_table(base_domain_summary),
            "",
            "## Overall validation regression",
            "",
            _markdown_table(overall_regression),
            "",
            "## changed/same decision metrics",
            "",
            _markdown_table(changed_same_decision),
            "",
            "## Top 10% ranking metrics",
            "",
            _markdown_table(top10),
            "",
            "## Interpretation boundary",
            "",
            "- `algorithm_partition_stacked_ridge` is a diagnostic ablation, not the main deployable Decision Model.",
            "- Any improvement should be reported as evidence that behavior-utility relations differ by prefix optimizer.",
            "- It should not be used to justify putting algorithm labels into the main Decision input.",
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

    headers = list(frame.columns)
    rows = [[format_value(value) for value in row] for row in frame.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run algorithm-partition stacking diagnostic for Decision Model labels.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    run_algorithm_partition_stacking_diagnostic(
        dataset_path=args.dataset,
        schema_path=args.schema,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
