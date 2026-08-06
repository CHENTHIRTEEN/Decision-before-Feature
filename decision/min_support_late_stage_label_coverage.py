from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from behavior.features import BEHAVIOR_FEATURE_COLUMNS
from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _check_target, _json_default
from decision.min_support_model_score_ranking import _add_score_ranks


GROUP_LAYERS = {
    "overall": [],
    "stage": ["stage"],
    "stage_family": ["stage", "family"],
    "stage_dimension": ["stage", "dimension"],
    "stage_fe_ratio": ["stage", "FE_ratio"],
    "stage_problem_id": ["stage", "problem_id"],
}


def run_late_stage_label_coverage_diagnostics(
    *,
    train_labels_path: Path,
    validation_labels_path: Path,
    train_predictions_path: Path,
    validation_predictions_path: Path,
    calibration_stage_summary_path: Path,
    calibration_neighbor_summary_path: Path,
    output_dir: Path,
    target_column: str,
    validation_threshold_mode: str,
    early_stage_max_fe_ratio: float,
    late_stage_min_fe_ratio: float,
) -> dict[str, Any]:
    _check_target(target_column)
    train_labels = _with_stage(_with_label_source(pq.read_table(train_labels_path).to_pandas()), early_stage_max_fe_ratio, late_stage_min_fe_ratio)
    validation_labels = _with_stage(
        _with_label_source(pq.read_table(validation_labels_path).to_pandas()),
        early_stage_max_fe_ratio,
        late_stage_min_fe_ratio,
    )
    _check_labels(train_labels, target_column)
    _check_labels(validation_labels, target_column)

    train_predictions = _rank_predictions(
        pq.read_table(train_predictions_path).to_pandas(),
        target_column=target_column,
        split_name="train",
        early_stage_max_fe_ratio=early_stage_max_fe_ratio,
        late_stage_min_fe_ratio=late_stage_min_fe_ratio,
    )
    validation_predictions = pq.read_table(validation_predictions_path).to_pandas()
    validation_predictions = validation_predictions[validation_predictions["threshold_mode"] == validation_threshold_mode].copy()
    validation_predictions = _rank_predictions(
        validation_predictions,
        target_column=target_column,
        split_name="validation",
        early_stage_max_fe_ratio=early_stage_max_fe_ratio,
        late_stage_min_fe_ratio=late_stage_min_fe_ratio,
    )
    calibration_context = _calibration_context(calibration_stage_summary_path, calibration_neighbor_summary_path)

    changed_labels = pd.concat(
        [
            train_labels.assign(split_for_coverage="train"),
            validation_labels.assign(split_for_coverage="validation"),
        ],
        ignore_index=True,
    )
    changed_labels = changed_labels[changed_labels["label_source"] == "changed_algorithm"].copy()

    coverage_summary = _coverage_summary(changed_labels, target_column)
    feature_range_summary = _feature_range_summary(changed_labels, target_column)
    score_support_gap = _score_support_gap(train_predictions, validation_predictions, target_column)
    coverage_conclusion = _coverage_conclusion(coverage_summary, score_support_gap)

    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = output_dir / "late_stage_label_coverage_summary.parquet"
    feature_range_path = output_dir / "late_stage_behavior_feature_ranges.parquet"
    score_gap_path = output_dir / "late_stage_score_support_gap.parquet"
    calibration_context_path = output_dir / "late_stage_calibration_context.parquet"
    summary_path = output_dir / "late_stage_label_coverage_summary.json"
    pq.write_table(pa.Table.from_pandas(coverage_summary, preserve_index=False), coverage_path)
    pq.write_table(pa.Table.from_pandas(feature_range_summary, preserve_index=False), feature_range_path)
    pq.write_table(pa.Table.from_pandas(score_support_gap, preserve_index=False), score_gap_path)
    pq.write_table(pa.Table.from_pandas(calibration_context, preserve_index=False), calibration_context_path)

    summary = {
        "experiment": "min_support_late_stage_label_coverage_diagnostics",
        "research_question": (
            "Does the min-support training set lack late-stage changed_algorithm rows with U_ELA > 0, "
            "creating a score-support gap for validation?"
        ),
        "target_column": target_column,
        "validation_threshold_mode": validation_threshold_mode,
        "stage_definition": {
            "early_stage": f"FE_ratio <= {early_stage_max_fe_ratio}",
            "middle_stage": f"{early_stage_max_fe_ratio} < FE_ratio < {late_stage_min_fe_ratio}",
            "late_stage": f"FE_ratio >= {late_stage_min_fe_ratio}",
        },
        "inputs": {
            "train_labels": str(train_labels_path),
            "validation_labels": str(validation_labels_path),
            "train_predictions": str(train_predictions_path),
            "validation_predictions": str(validation_predictions_path),
            "calibration_stage_summary": str(calibration_stage_summary_path),
            "calibration_neighbor_summary": str(calibration_neighbor_summary_path),
        },
        "rows": {
            "changed_label_rows": int(len(changed_labels)),
            "coverage_summary_rows": int(len(coverage_summary)),
            "feature_range_summary_rows": int(len(feature_range_summary)),
            "score_support_gap_rows": int(len(score_support_gap)),
            "calibration_context_rows": int(len(calibration_context)),
            **coverage_conclusion["row_counts"],
        },
        "coverage_conclusion": coverage_conclusion,
        "outputs": {
            "coverage_summary": str(coverage_path),
            "feature_range_summary": str(feature_range_path),
            "score_support_gap": str(score_gap_path),
            "calibration_context": str(calibration_context_path),
            "summary": str(summary_path),
        },
        "data_leakage_check": {
            "uses_existing_utility_labels": True,
            "uses_existing_model_scores_only": True,
            "uses_calibration_opportunity_output_for_context": True,
            "models_retrained": False,
            "threshold_selected_from_validation": False,
            "ela_features_used_as_decision_input": False,
            "metadata_used_only_for_grouping": True,
            "original_utility_labels_modified": False,
        },
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote late-stage label coverage summary to {coverage_path}")
    print(f"wrote late-stage behavior feature ranges to {feature_range_path}")
    print(f"wrote late-stage score support gap to {score_gap_path}")
    print(f"wrote late-stage calibration context to {calibration_context_path}")
    print(f"wrote late-stage label coverage summary JSON to {summary_path}")
    return summary


def _with_label_source(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["label_source"] = np.where(
        result["selected_algorithm"].astype(str) == result["default_algorithm"].astype(str),
        "same_algorithm",
        "changed_algorithm",
    )
    return result


def _with_stage(frame: pd.DataFrame, early_stage_max_fe_ratio: float, late_stage_min_fe_ratio: float) -> pd.DataFrame:
    result = frame.copy()
    fe_ratio = result["FE_ratio"].to_numpy(dtype=float)
    result["stage"] = np.select(
        [fe_ratio <= early_stage_max_fe_ratio, fe_ratio >= late_stage_min_fe_ratio],
        ["early_stage", "late_stage"],
        default="middle_stage",
    )
    return result


def _check_labels(frame: pd.DataFrame, target_column: str) -> None:
    required = {
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
        "stage",
        target_column,
        *BEHAVIOR_FEATURE_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"utility labels missing columns: {missing}")


def _rank_predictions(
    predictions: pd.DataFrame,
    *,
    target_column: str,
    split_name: str,
    early_stage_max_fe_ratio: float,
    late_stage_min_fe_ratio: float,
) -> pd.DataFrame:
    required = {
        "model_name",
        "model_family",
        "score_semantics",
        "problem_id",
        "family",
        "dimension",
        "prefix_algorithm",
        "seed",
        "FE",
        "FE_ratio",
        "label_source",
        "decision_score",
        target_column,
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"model predictions missing columns: {missing}")
    changed = predictions[predictions["label_source"] == "changed_algorithm"].copy()
    changed.insert(0, "split_for_coverage", split_name)
    changed = _with_stage(changed, early_stage_max_fe_ratio, late_stage_min_fe_ratio)
    ranked_frames = []
    for _, subset in changed.groupby(["model_name", "label_source"], dropna=False):
        ranked_frames.append(_add_score_ranks(subset))
    return pd.concat(ranked_frames, ignore_index=True)


def _coverage_summary(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows = []
    for layer, columns in GROUP_LAYERS.items():
        group_columns = ["split_for_coverage", *columns]
        for group_values, subset in frame.groupby(group_columns, dropna=False):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            group = dict(zip(group_columns, group_values, strict=True))
            rows.append(_coverage_row(subset, layer, group, columns, target_column))
    return pd.DataFrame(rows)


def _coverage_row(
    subset: pd.DataFrame,
    layer: str,
    group: dict[str, Any],
    reported_group_columns: list[str],
    target_column: str,
) -> dict[str, Any]:
    positive = subset[target_column] > 0.0
    positive_subset = subset[positive]
    return {
        "split_for_coverage": group["split_for_coverage"],
        "layer": layer,
        "group": _group_label({column: group[column] for column in reported_group_columns}),
        "stage": group.get("stage"),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "problem_id": group.get("problem_id"),
        "rows": int(len(subset)),
        "utility_gt_zero_rows": int(positive.sum()),
        "utility_gt_zero_rate": float(positive.mean()) if len(subset) else 0.0,
        "positive_utility_sum": float(subset.loc[positive, target_column].sum()),
        "utility_mean": float(subset[target_column].mean()),
        "unique_families": int(subset["family"].nunique()),
        "positive_unique_families": int(positive_subset["family"].nunique()),
        "positive_families": _joined_values(positive_subset["family"]),
        "positive_unique_problem_ids": int(positive_subset["problem_id"].nunique()),
        "positive_problem_ids": _joined_values(positive_subset["problem_id"]),
        "positive_fe_ratios": _joined_values(positive_subset["FE_ratio"]),
    }


def _feature_range_summary(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows = []
    labeled = frame.copy()
    labeled["utility_group"] = np.where(labeled[target_column] > 0.0, "utility_gt_zero", "utility_le_zero")
    all_rows = labeled.assign(utility_group="all_changed_algorithm")
    stacked = pd.concat([all_rows, labeled], ignore_index=True)
    for (split_name, stage, utility_group), subset in stacked.groupby(
        ["split_for_coverage", "stage", "utility_group"],
        dropna=False,
    ):
        base = {
            "split_for_coverage": split_name,
            "stage": stage,
            "utility_group": utility_group,
            "rows": int(len(subset)),
            "utility_gt_zero_rows": int((subset[target_column] > 0.0).sum()),
            "families": _joined_values(subset["family"]),
        }
        for feature in BEHAVIOR_FEATURE_COLUMNS:
            rows.append({**base, "feature": feature, **_numeric_stats(subset[feature])})
    return pd.DataFrame(rows)


def _score_support_gap(train_predictions: pd.DataFrame, validation_predictions: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows = []
    for model_name in sorted(set(train_predictions["model_name"].astype(str)).union(validation_predictions["model_name"].astype(str))):
        model_train = train_predictions[train_predictions["model_name"].astype(str) == model_name]
        model_validation = validation_predictions[validation_predictions["model_name"].astype(str) == model_name]
        for stage in ["early_stage", "middle_stage", "late_stage"]:
            train_stage = model_train[model_train["stage"] == stage]
            validation_stage = model_validation[model_validation["stage"] == stage]
            rows.append(_score_gap_row(model_name, stage, train_stage, validation_stage, target_column))
    return pd.DataFrame(rows)


def _score_gap_row(
    model_name: str,
    stage: str,
    train_stage: pd.DataFrame,
    validation_stage: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    train_positive = train_stage[train_stage[target_column] > 0.0]
    validation_positive = validation_stage[validation_stage[target_column] > 0.0]
    validation_top10 = validation_positive[validation_positive["score_quantile"] >= 0.9]
    train_top10 = train_positive[train_positive["score_quantile"] >= 0.9]
    if train_positive.empty and not validation_positive.empty:
        status = "no_train_positive_same_stage"
    elif validation_positive.empty:
        status = "no_validation_positive_same_stage"
    elif len(validation_top10) == 0:
        status = "validation_positive_scores_below_top10"
    else:
        status = "validation_positive_has_top10_score_support"
    return {
        "model_name": model_name,
        "model_family": _first_or_none(pd.concat([train_stage, validation_stage], ignore_index=True), "model_family"),
        "score_semantics": _first_or_none(pd.concat([train_stage, validation_stage], ignore_index=True), "score_semantics"),
        "stage": stage,
        "support_gap_status": status,
        "train_rows": int(len(train_stage)),
        "train_utility_gt_zero_rows": int(len(train_positive)),
        "train_utility_gt_zero_rate": float(len(train_positive) / len(train_stage)) if len(train_stage) else 0.0,
        "train_positive_families": _joined_values(train_positive["family"]),
        "train_positive_score_quantile_median": _median_or_none(train_positive["score_quantile"]),
        "train_positive_score_quantile_max": _max_or_none(train_positive["score_quantile"]),
        "train_positive_top10_rows": int(len(train_top10)),
        "train_positive_top10_capture_rate": float(len(train_top10) / len(train_positive)) if len(train_positive) else 0.0,
        "validation_rows": int(len(validation_stage)),
        "validation_utility_gt_zero_rows": int(len(validation_positive)),
        "validation_utility_gt_zero_rate": (
            float(len(validation_positive) / len(validation_stage)) if len(validation_stage) else 0.0
        ),
        "validation_positive_families": _joined_values(validation_positive["family"]),
        "validation_positive_score_quantile_median": _median_or_none(validation_positive["score_quantile"]),
        "validation_positive_score_quantile_max": _max_or_none(validation_positive["score_quantile"]),
        "validation_positive_top10_rows": int(len(validation_top10)),
        "validation_positive_top10_capture_rate": (
            float(len(validation_top10) / len(validation_positive)) if len(validation_positive) else 0.0
        ),
        "validation_positive_utility_sum": float(validation_positive[target_column].sum()),
    }


def _calibration_context(stage_summary_path: Path, neighbor_summary_path: Path) -> pd.DataFrame:
    frames = []
    if stage_summary_path.exists():
        stage = pq.read_table(stage_summary_path).to_pandas()
        keep = [
            "model_name",
            "threshold_scope",
            "target_stage",
            "train_rows_for_threshold",
            "train_utility_gt_zero_rows_for_threshold",
            "target_rows",
            "target_capture_rate",
            "target_captured_utility_sum",
            "target_utility_sum",
        ]
        stage = stage[[column for column in keep if column in stage.columns]].copy()
        stage.insert(0, "context_type", "calibration_stage_threshold")
        frames.append(stage)
    if neighbor_summary_path.exists():
        neighbor = pq.read_table(neighbor_summary_path).to_pandas()
        neighbor = neighbor[
            (neighbor["neighbor_k"] == 10)
            & (neighbor["feature_set"] == "without_fe_ratio")
            & neighbor["neighbor_pool"].isin(["changed_algorithm_all_train", "changed_algorithm_late_stage_train"])
        ].copy()
        keep = [
            "model_name",
            "neighbor_pool",
            "feature_set",
            "neighbor_k",
            "candidate_train_rows",
            "candidate_train_utility_gt_zero_rows",
            "nearest_unique_train_utility_gt_zero_rows",
            "per_target_neighbor_positive_rate_mean",
            "per_target_neighbor_mean_utility_mean",
            "per_target_neighbor_distance_median",
        ]
        neighbor = neighbor[[column for column in keep if column in neighbor.columns]].copy()
        neighbor.insert(0, "context_type", "calibration_neighbor_support")
        frames.append(neighbor)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _coverage_conclusion(coverage: pd.DataFrame, score_gap: pd.DataFrame) -> dict[str, Any]:
    stage_rows = coverage[(coverage["layer"] == "stage") & (coverage["stage"] == "late_stage")]
    train_late = stage_rows[stage_rows["split_for_coverage"] == "train"]
    validation_late = stage_rows[stage_rows["split_for_coverage"] == "validation"]
    train_late_positive = int(train_late["utility_gt_zero_rows"].sum()) if not train_late.empty else 0
    validation_late_positive = int(validation_late["utility_gt_zero_rows"].sum()) if not validation_late.empty else 0
    late_gap_models = score_gap[
        (score_gap["stage"] == "late_stage")
        & (score_gap["support_gap_status"] == "no_train_positive_same_stage")
    ]["model_name"].astype(str).tolist()
    if train_late_positive == 0 and validation_late_positive > 0:
        judgement = "min_support_train_lacks_late_stage_changed_algorithm_useful_ela_labels"
    elif train_late_positive > 0 and validation_late_positive > 0:
        judgement = "late_stage_labels_exist_in_train_and_validation"
    else:
        judgement = "late_stage_validation_has_no_useful_ela_labels"
    return {
        "judgement": judgement,
        "row_counts": {
            "train_changed_late_utility_gt_zero_rows": train_late_positive,
            "validation_changed_late_utility_gt_zero_rows": validation_late_positive,
            "late_stage_models_with_no_train_positive_same_stage": len(late_gap_models),
        },
        "late_stage_models_with_no_train_positive_same_stage": late_gap_models,
    }


def _numeric_stats(values: pd.Series) -> dict[str, float | int | None]:
    numeric = values.dropna().astype(float)
    if numeric.empty:
        return {"non_null_rows": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None, "mean": None}
    return {
        "non_null_rows": int(len(numeric)),
        "min": float(numeric.min()),
        "q25": float(numeric.quantile(0.25)),
        "median": float(numeric.median()),
        "q75": float(numeric.quantile(0.75)),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
    }


def _joined_values(values: pd.Series) -> str | None:
    cleaned = sorted({str(value) for value in values.dropna().unique()})
    return ",".join(cleaned) if cleaned else None


def _group_label(group: dict[str, Any]) -> str:
    if not group:
        return "overall"
    return "|".join(f"{key}={value}" for key, value in group.items())


def _median_or_none(values: pd.Series) -> float | None:
    numeric = values.dropna().astype(float)
    return float(numeric.median()) if len(numeric) else None


def _max_or_none(values: pd.Series) -> float | None:
    numeric = values.dropna().astype(float)
    return float(numeric.max()) if len(numeric) else None


def _first_or_none(frame: pd.DataFrame, column: str) -> str | None:
    if frame.empty or column not in frame.columns:
        return None
    return str(frame[column].iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose min-support late-stage changed-algorithm label coverage.")
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
    parser.add_argument(
        "--train-predictions",
        type=Path,
        default=Path("results/decision/min_support/model_sensitivity/model_sensitivity_train_predictions.parquet"),
    )
    parser.add_argument(
        "--validation-predictions",
        type=Path,
        default=Path("results/decision/min_support/model_sensitivity/model_sensitivity_predictions.parquet"),
    )
    parser.add_argument(
        "--calibration-stage-summary",
        type=Path,
        default=Path("results/decision/min_support/calibration_opportunity/calibration_stage_threshold_summary.parquet"),
    )
    parser.add_argument(
        "--calibration-neighbor-summary",
        type=Path,
        default=Path("results/decision/min_support/calibration_opportunity/calibration_neighbor_summary.parquet"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--validation-threshold-mode", default="zero")
    parser.add_argument("--early-stage-max-fe-ratio", type=float, default=0.2)
    parser.add_argument("--late-stage-min-fe-ratio", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/late_stage_label_coverage"),
    )
    args = parser.parse_args()
    run_late_stage_label_coverage_diagnostics(
        train_labels_path=args.train_labels,
        validation_labels_path=args.validation_labels,
        train_predictions_path=args.train_predictions,
        validation_predictions_path=args.validation_predictions,
        calibration_stage_summary_path=args.calibration_stage_summary,
        calibration_neighbor_summary_path=args.calibration_neighbor_summary,
        output_dir=args.output_dir,
        target_column=args.target_column,
        validation_threshold_mode=args.validation_threshold_mode,
        early_stage_max_fe_ratio=args.early_stage_max_fe_ratio,
        late_stage_min_fe_ratio=args.late_stage_min_fe_ratio,
    )


if __name__ == "__main__":
    main()
