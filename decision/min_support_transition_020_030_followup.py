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
from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _json_default, _read_labels
from utility_labels.fields import UTILITY_VALUE_COLUMNS


TARGET_FAMILIES = ("bbob_f005", "bbob_f024")
TARGET_STAGES = (0.20, 0.25, 0.30, 0.35)
STAGE_TOLERANCE = 0.003


def run_transition_followup(
    *,
    fe_transition_validation_labels_path: Path,
    followup_validation_labels_path: Path,
    output_dir: Path,
    target_column: str,
) -> dict[str, Any]:
    _check_target_column(target_column)
    fe_transition = _with_source(_read_labels(fe_transition_validation_labels_path), "fe_transition")
    followup = _with_source(_read_labels(followup_validation_labels_path), "transition_020_030_followup")
    combined = pd.concat([fe_transition, followup], ignore_index=True)
    combined = _with_label_source(combined)
    combined["comparison_stage"] = combined["FE_ratio"].map(_comparison_stage)
    availability = combined[
        (combined["family"].isin(TARGET_FAMILIES))
        & (combined["comparison_stage"].isin(TARGET_STAGES))
    ].copy()
    availability = _deduplicate_stage_rows(availability)

    target = availability[availability["label_source"] == "changed_algorithm"].copy()
    target["utility_role"] = np.where(target[target_column] > 0.0, "u_gt_zero", "u_le_zero")
    target["unhelpful_if_called"] = target[target_column] <= 0.0
    target["unhelpful_call_cost"] = np.where(target["unhelpful_if_called"], target["time_cost_norm"], 0.0)
    target["unhelpful_call_utility_loss"] = np.where(target["unhelpful_if_called"], -target[target_column], 0.0)

    availability_summary = _summarize_availability(availability, target_column)
    label_summary = _summarize_labels(target, target_column)
    unhelpful_cost_summary = _summarize_unhelpful_cost(target, target_column)
    behavior_distribution = _summarize_behavior_distribution(target)
    behavior_contrast = _summarize_behavior_contrast(behavior_distribution)

    output_dir.mkdir(parents=True, exist_ok=True)
    availability_path = output_dir / "transition_020_030_label_source_availability.parquet"
    rows_path = output_dir / "transition_020_030_target_changed_rows.parquet"
    label_path = output_dir / "transition_020_030_label_summary.parquet"
    cost_path = output_dir / "transition_020_030_unhelpful_cost_summary.parquet"
    behavior_path = output_dir / "transition_020_030_behavior_feature_distribution.parquet"
    contrast_path = output_dir / "transition_020_030_behavior_feature_contrast.parquet"
    summary_path = output_dir / "transition_020_030_followup_summary.json"

    pq.write_table(pa.Table.from_pandas(availability_summary, preserve_index=False), availability_path)
    pq.write_table(pa.Table.from_pandas(target, preserve_index=False), rows_path)
    pq.write_table(pa.Table.from_pandas(label_summary, preserve_index=False), label_path)
    pq.write_table(pa.Table.from_pandas(unhelpful_cost_summary, preserve_index=False), cost_path)
    pq.write_table(pa.Table.from_pandas(behavior_distribution, preserve_index=False), behavior_path)
    pq.write_table(pa.Table.from_pandas(behavior_contrast, preserve_index=False), contrast_path)

    summary = {
        "experiment": "min_support_transition_020_030_followup",
        "research_question": (
            "Does adding FE_ratio 0.25 clarify the transition from low to high U_ELA > 0 coverage "
            "between FE_ratio 0.20 and 0.30 for bbob_f005 and bbob_f024 changed_algorithm validation rows?"
        ),
        "target_column": target_column,
        "target_families": list(TARGET_FAMILIES),
        "comparison_stages": list(TARGET_STAGES),
        "inputs": {
            "fe_transition_validation_labels": str(fe_transition_validation_labels_path),
            "followup_validation_labels": str(followup_validation_labels_path),
        },
        "rows": {
            "label_source_availability": _availability_records(availability_summary),
            "target_changed_algorithm_rows": int(len(target)),
            "by_stage": _compact_stage_records(label_summary, group_columns=["comparison_stage"]),
            "by_family_stage": _compact_stage_records(label_summary, group_columns=["family", "comparison_stage"]),
        },
        "outputs": {
            "label_source_availability": str(availability_path),
            "target_rows": str(rows_path),
            "label_summary": str(label_path),
            "unhelpful_cost_summary": str(cost_path),
            "behavior_feature_distribution": str(behavior_path),
            "behavior_feature_contrast": str(contrast_path),
            "summary": str(summary_path),
        },
        "data_leakage_check": {
            "decision_input_uses_ela_features": False,
            "function_id_or_algorithm_id_used_as_decision_input": False,
            "original_utility_labels_modified": False,
            "formal_phase1_configs_modified": False,
        },
        "notes": [
            "Actual FE_ratio values can differ slightly from requested ratios because checkpoints are aligned to complete populations.",
            "The comparison_stage column maps actual 10D FE_ratio 0.252 to requested stage 0.25 and actual 0.352 to requested stage 0.35.",
        ],
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote transition label-source availability to {availability_path}")
    print(f"wrote transition target changed rows to {rows_path}")
    print(f"wrote transition label summary to {label_path}")
    print(f"wrote transition unhelpful cost summary to {cost_path}")
    print(f"wrote transition behavior feature distribution to {behavior_path}")
    print(f"wrote transition behavior feature contrast to {contrast_path}")
    print(f"wrote transition follow-up summary to {summary_path}")
    return summary


def _check_target_column(target_column: str) -> None:
    if target_column not in UTILITY_VALUE_COLUMNS:
        raise ValueError(f"target column must be one of {list(UTILITY_VALUE_COLUMNS)}")


def _with_source(frame: pd.DataFrame, label_dataset: str) -> pd.DataFrame:
    result = frame.copy()
    result["label_dataset"] = label_dataset
    return result


def _with_label_source(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["label_source"] = np.where(
        result["selected_algorithm"].astype(str) == result["default_algorithm"].astype(str),
        "same_algorithm",
        "changed_algorithm",
    )
    return result


def _comparison_stage(value: float) -> float | None:
    ratio = float(value)
    for stage in TARGET_STAGES:
        if abs(ratio - stage) <= STAGE_TOLERANCE:
            return float(stage)
    return None


def _deduplicate_stage_rows(frame: pd.DataFrame) -> pd.DataFrame:
    priority = {"transition_020_030_followup": 0, "fe_transition": 1}
    result = frame.copy()
    result["_source_priority"] = result["label_dataset"].map(priority).fillna(9).astype(int)
    key_columns = ["problem_id", "dimension", "prefix_algorithm", "seed", "comparison_stage"]
    result = result.sort_values(key_columns + ["_source_priority", "FE_ratio"])
    result = result.drop_duplicates(key_columns, keep="first")
    return result.drop(columns=["_source_priority"])


def _summarize_labels(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    group_layers = {
        "stage": ["comparison_stage"],
        "family_stage": ["family", "comparison_stage"],
        "dimension_stage": ["dimension", "comparison_stage"],
        "problem_stage": ["problem_id", "comparison_stage"],
        "family_dimension_stage": ["family", "dimension", "comparison_stage"],
    }
    summaries = []
    for layer, columns in group_layers.items():
        summaries.append(_summarize_groups(frame, layer=layer, group_columns=columns, target_column=target_column))
    return pd.concat(summaries, ignore_index=True)


def _summarize_availability(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    group_columns = ["family", "comparison_stage", "label_source"]
    rows = []
    for family in TARGET_FAMILIES:
        for stage in TARGET_STAGES:
            for label_source in ("changed_algorithm", "same_algorithm"):
                group = frame[
                    (frame["family"] == family)
                    & (frame["comparison_stage"] == stage)
                    & (frame["label_source"] == label_source)
                ]
                values = group[target_column].to_numpy(dtype=float)
                gt_zero = values > 0.0
                rows.append(
                    {
                        "family": family,
                        "comparison_stage": float(stage),
                        "label_source": label_source,
                        "rows": int(len(group)),
                        "u_gt_zero_count": int(gt_zero.sum()),
                        "u_gt_zero_rate": float(gt_zero.mean()) if len(group) else 0.0,
                        "u_mean": float(np.mean(values)) if len(group) else None,
                        "time_cost_norm_mean": float(group["time_cost_norm"].mean()) if len(group) else None,
                        "performance_gain_norm_mean": float(group["performance_gain_norm"].mean()) if len(group) else None,
                    }
                )
    summary = pd.DataFrame(rows)
    stage_rows = []
    for stage in TARGET_STAGES:
        for label_source in ("changed_algorithm", "same_algorithm"):
            group = frame[
                (frame["comparison_stage"] == stage)
                & (frame["label_source"] == label_source)
            ]
            values = group[target_column].to_numpy(dtype=float)
            gt_zero = values > 0.0
            stage_rows.append(
                {
                    "family": "all_target_families",
                    "comparison_stage": float(stage),
                    "label_source": label_source,
                    "rows": int(len(group)),
                    "u_gt_zero_count": int(gt_zero.sum()),
                    "u_gt_zero_rate": float(gt_zero.mean()) if len(group) else 0.0,
                    "u_mean": float(np.mean(values)) if len(group) else None,
                    "time_cost_norm_mean": float(group["time_cost_norm"].mean()) if len(group) else None,
                    "performance_gain_norm_mean": float(group["performance_gain_norm"].mean()) if len(group) else None,
                }
            )
    return pd.concat([pd.DataFrame(stage_rows), summary], ignore_index=True)


def _summarize_groups(
    frame: pd.DataFrame,
    *,
    layer: str,
    group_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    rows = []
    for group_key, group in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        values = group[target_column].to_numpy(dtype=float)
        gt_zero = values > 0.0
        unhelpful = values <= 0.0
        row = {
            "layer": layer,
            "rows": int(len(group)),
            "u_gt_zero_count": int(gt_zero.sum()),
            "u_gt_zero_rate": float(gt_zero.mean()) if len(group) else 0.0,
            "u_mean": float(np.mean(values)) if len(group) else 0.0,
            "u_median": float(np.median(values)) if len(group) else 0.0,
            "u_gt_zero_sum": float(np.sum(values[gt_zero])) if gt_zero.any() else 0.0,
            "unhelpful_if_called_count": int(unhelpful.sum()),
            "unhelpful_if_called_rate": float(unhelpful.mean()) if len(group) else 0.0,
            "unhelpful_call_utility_loss_sum": float(np.sum(-values[unhelpful])) if unhelpful.any() else 0.0,
            "time_cost_norm_mean": float(group["time_cost_norm"].mean()) if len(group) else 0.0,
            "performance_gain_norm_mean": float(group["performance_gain_norm"].mean()) if len(group) else 0.0,
        }
        row.update({column: value for column, value in zip(group_columns, group_key, strict=True)})
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_unhelpful_cost(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    cost_layers = {
        "stage": ["comparison_stage"],
        "family_stage": ["family", "comparison_stage"],
        "problem_stage": ["problem_id", "comparison_stage"],
        "dimension_stage": ["dimension", "comparison_stage"],
    }
    summaries = []
    for layer, columns in cost_layers.items():
        summary = _summarize_groups(frame, layer=layer, group_columns=columns, target_column=target_column)
        summaries.append(
            summary[
                [
                    *columns,
                    "layer",
                    "rows",
                    "unhelpful_if_called_count",
                    "unhelpful_if_called_rate",
                    "unhelpful_call_utility_loss_sum",
                    "time_cost_norm_mean",
                    "performance_gain_norm_mean",
                ]
            ]
        )
    return pd.concat(summaries, ignore_index=True)


def _summarize_behavior_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["family", "dimension", "comparison_stage", "utility_role"]
    for group_key, group in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        for feature in BEHAVIOR_FEATURE_COLUMNS:
            values = group[feature].dropna().to_numpy(dtype=float)
            row = {
                "behavior_feature": feature,
                "rows": int(len(group)),
                "non_null_rows": int(len(values)),
                "mean": float(np.mean(values)) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
                "p10": float(np.quantile(values, 0.10)) if len(values) else None,
                "p90": float(np.quantile(values, 0.90)) if len(values) else None,
                "std": float(np.std(values, ddof=0)) if len(values) else None,
            }
            row.update({column: value for column, value in zip(group_columns, group_key, strict=True)})
            rows.append(row)
    return pd.DataFrame(rows)


def _summarize_behavior_contrast(distribution: pd.DataFrame) -> pd.DataFrame:
    if distribution.empty:
        return distribution.copy()
    key_columns = ["family", "dimension", "comparison_stage", "behavior_feature"]
    wide = distribution.pivot_table(
        index=key_columns,
        columns="utility_role",
        values=["mean", "median", "p10", "p90", "rows"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{role}" for metric, role in wide.columns]
    wide = wide.reset_index()
    if "mean_u_gt_zero" in wide and "mean_u_le_zero" in wide:
        wide["mean_difference_u_gt_zero_minus_u_le_zero"] = wide["mean_u_gt_zero"] - wide["mean_u_le_zero"]
    if "median_u_gt_zero" in wide and "median_u_le_zero" in wide:
        wide["median_difference_u_gt_zero_minus_u_le_zero"] = wide["median_u_gt_zero"] - wide["median_u_le_zero"]
    if {"p10_u_gt_zero", "p90_u_gt_zero", "p10_u_le_zero", "p90_u_le_zero"}.issubset(wide.columns):
        overlap_low = np.maximum(wide["p10_u_gt_zero"], wide["p10_u_le_zero"])
        overlap_high = np.minimum(wide["p90_u_gt_zero"], wide["p90_u_le_zero"])
        total_low = np.minimum(wide["p10_u_gt_zero"], wide["p10_u_le_zero"])
        total_high = np.maximum(wide["p90_u_gt_zero"], wide["p90_u_le_zero"])
        wide["p10_p90_overlap_fraction"] = np.where(
            total_high > total_low,
            np.maximum(overlap_high - overlap_low, 0.0) / (total_high - total_low),
            np.nan,
        )
    return wide


def _compact_stage_records(summary: pd.DataFrame, *, group_columns: list[str]) -> list[dict[str, Any]]:
    subset = summary[summary["layer"] == "_".join(column.replace("comparison_", "") for column in group_columns)]
    if subset.empty:
        layer_name = "stage" if group_columns == ["comparison_stage"] else "family_stage"
        subset = summary[summary["layer"] == layer_name]
    columns = [
        *group_columns,
        "rows",
        "u_gt_zero_count",
        "u_gt_zero_rate",
        "u_mean",
        "unhelpful_if_called_count",
        "unhelpful_call_utility_loss_sum",
        "time_cost_norm_mean",
    ]
    return subset[[column for column in columns if column in subset.columns]].to_dict(orient="records")


def _availability_records(summary: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "family",
        "comparison_stage",
        "label_source",
        "rows",
        "u_gt_zero_count",
        "u_gt_zero_rate",
        "u_mean",
        "time_cost_norm_mean",
        "performance_gain_norm_mean",
    ]
    return summary[columns].to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare min_support 0.20-0.30 FE transition follow-up labels.")
    parser.add_argument(
        "--fe-transition-validation-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_validation_fe_transition/utility_labels.parquet"),
    )
    parser.add_argument(
        "--followup-validation-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_validation_transition_020_030_followup/utility_labels.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/transition_020_030_followup"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    args = parser.parse_args()
    run_transition_followup(
        fe_transition_validation_labels_path=args.fe_transition_validation_labels,
        followup_validation_labels_path=args.followup_validation_labels,
        output_dir=args.output_dir,
        target_column=args.target_column,
    )


if __name__ == "__main__":
    main()
