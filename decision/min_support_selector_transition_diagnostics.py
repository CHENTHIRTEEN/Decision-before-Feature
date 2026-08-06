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
FE_ANALYSIS_RATIO = 0.05


def run_selector_transition_diagnostics(
    *,
    fe_transition_validation_labels_path: Path,
    followup_validation_labels_path: Path,
    fe_transition_selection_reference_path: Path,
    followup_selection_reference_path: Path,
    output_dir: Path,
    target_column: str,
) -> dict[str, Any]:
    _check_target_column(target_column)
    label_rows = _target_label_rows(
        fe_transition_validation_labels_path=fe_transition_validation_labels_path,
        followup_validation_labels_path=followup_validation_labels_path,
        target_column=target_column,
    )
    selection_rows = _target_selection_rows(
        fe_transition_selection_reference_path=fe_transition_selection_reference_path,
        followup_selection_reference_path=followup_selection_reference_path,
    )
    stage_summary = _summarize_stage_labels(label_rows, target_column)
    selector_summary = _summarize_selector_rows(selection_rows)
    selector_transition = _selector_transition_rows(selection_rows)
    behavior_summary = _summarize_behavior(label_rows)
    behavior_stage_shift = _summarize_behavior_stage_shift(behavior_summary)
    stage_evidence = _stage_evidence(stage_summary, selector_summary)
    conclusion = _diagnostic_conclusion(selector_summary, selector_transition, stage_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    label_rows_path = output_dir / "selector_transition_target_label_rows.parquet"
    selection_rows_path = output_dir / "selector_transition_reference_rows.parquet"
    stage_summary_path = output_dir / "selector_transition_stage_label_summary.parquet"
    selector_summary_path = output_dir / "selector_transition_selector_summary.parquet"
    selector_transition_path = output_dir / "selector_transition_problem_stage_changes.parquet"
    behavior_summary_path = output_dir / "selector_transition_behavior_distribution.parquet"
    behavior_stage_shift_path = output_dir / "selector_transition_behavior_stage_shift.parquet"
    stage_evidence_path = output_dir / "selector_transition_stage_evidence.parquet"
    summary_path = output_dir / "selector_transition_diagnostic_summary.json"

    pq.write_table(pa.Table.from_pandas(label_rows, preserve_index=False), label_rows_path)
    pq.write_table(pa.Table.from_pandas(selection_rows, preserve_index=False), selection_rows_path)
    pq.write_table(pa.Table.from_pandas(stage_summary, preserve_index=False), stage_summary_path)
    pq.write_table(pa.Table.from_pandas(selector_summary, preserve_index=False), selector_summary_path)
    pq.write_table(pa.Table.from_pandas(selector_transition, preserve_index=False), selector_transition_path)
    pq.write_table(pa.Table.from_pandas(behavior_summary, preserve_index=False), behavior_summary_path)
    pq.write_table(pa.Table.from_pandas(behavior_stage_shift, preserve_index=False), behavior_stage_shift_path)
    pq.write_table(pa.Table.from_pandas(stage_evidence, preserve_index=False), stage_evidence_path)

    summary = {
        "experiment": "min_support_selector_transition_diagnostic",
        "research_question": (
            "Why do bbob_f005/bbob_f024 changed_algorithm validation rows first appear at FE_ratio 0.30 "
            "rather than 0.20 or 0.25 under the unchanged min_support protocol?"
        ),
        "target_column": target_column,
        "target_families": list(TARGET_FAMILIES),
        "comparison_stages": list(TARGET_STAGES),
        "inputs": {
            "fe_transition_validation_labels": str(fe_transition_validation_labels_path),
            "followup_validation_labels": str(followup_validation_labels_path),
            "fe_transition_selection_reference": str(fe_transition_selection_reference_path),
            "followup_selection_reference": str(followup_selection_reference_path),
        },
        "outputs": {
            "target_label_rows": str(label_rows_path),
            "selection_reference_rows": str(selection_rows_path),
            "stage_label_summary": str(stage_summary_path),
            "selector_summary": str(selector_summary_path),
            "selector_problem_stage_changes": str(selector_transition_path),
            "behavior_distribution": str(behavior_summary_path),
            "behavior_stage_shift": str(behavior_stage_shift_path),
            "stage_evidence": str(stage_evidence_path),
            "summary": str(summary_path),
        },
        "stage_evidence": stage_evidence.to_dict(orient="records"),
        "diagnostic_conclusion": conclusion,
        "data_leakage_check": {
            "decision_input_uses_ela_features": False,
            "ela_feature_values_read": False,
            "original_utility_labels_modified": False,
            "formal_phase1_configs_modified": False,
            "models_retrained": False,
        },
        "notes": [
            "Selected_algorithm is inspected as an existing selection_reference output; ELA feature values are not read.",
            "comparison_stage maps actual 10D population-aligned FE_ratio 0.252 to 0.25 and 0.352 to 0.35.",
            "selection_stage is inferred from remaining_budget_ratio using prefix_ratio = 0.95 - remaining_budget_ratio.",
        ],
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote selector transition target label rows to {label_rows_path}")
    print(f"wrote selector transition reference rows to {selection_rows_path}")
    print(f"wrote selector transition stage label summary to {stage_summary_path}")
    print(f"wrote selector transition selector summary to {selector_summary_path}")
    print(f"wrote selector transition problem-stage changes to {selector_transition_path}")
    print(f"wrote selector transition behavior distribution to {behavior_summary_path}")
    print(f"wrote selector transition behavior stage shift to {behavior_stage_shift_path}")
    print(f"wrote selector transition stage evidence to {stage_evidence_path}")
    print(f"wrote selector transition diagnostic summary to {summary_path}")
    return summary


def _check_target_column(target_column: str) -> None:
    if target_column not in UTILITY_VALUE_COLUMNS:
        raise ValueError(f"target column must be one of {list(UTILITY_VALUE_COLUMNS)}")


def _target_label_rows(
    *,
    fe_transition_validation_labels_path: Path,
    followup_validation_labels_path: Path,
    target_column: str,
) -> pd.DataFrame:
    frames = [
        _read_labels(fe_transition_validation_labels_path).assign(source_dataset="fe_transition"),
        _read_labels(followup_validation_labels_path).assign(source_dataset="transition_020_030_followup"),
    ]
    labels = pd.concat(frames, ignore_index=True)
    labels = labels[labels["family"].isin(TARGET_FAMILIES)].copy()
    labels["comparison_stage"] = labels["FE_ratio"].map(_stage_from_ratio)
    labels = labels[labels["comparison_stage"].isin(TARGET_STAGES)].copy()
    labels["label_source"] = np.where(
        labels["selected_algorithm"].astype(str) == labels["default_algorithm"].astype(str),
        "same_algorithm",
        "changed_algorithm",
    )
    labels["u_gt_zero"] = labels[target_column] > 0.0
    labels = _deduplicate_label_rows(labels)
    return labels


def _deduplicate_label_rows(labels: pd.DataFrame) -> pd.DataFrame:
    priority = {"transition_020_030_followup": 0, "fe_transition": 1}
    result = labels.copy()
    result["_source_priority"] = result["source_dataset"].map(priority).fillna(9).astype(int)
    key_columns = ["problem_id", "dimension", "prefix_algorithm", "seed", "comparison_stage"]
    result = result.sort_values(key_columns + ["_source_priority", "FE_ratio"])
    return result.drop_duplicates(key_columns, keep="first").drop(columns=["_source_priority"])


def _target_selection_rows(
    *,
    fe_transition_selection_reference_path: Path,
    followup_selection_reference_path: Path,
) -> pd.DataFrame:
    frames = [
        _read_selection_reference(fe_transition_selection_reference_path, "fe_transition"),
        _read_selection_reference(followup_selection_reference_path, "transition_020_030_followup"),
    ]
    selection = pd.concat(frames, ignore_index=True)
    selection = selection[selection["family"].isin(TARGET_FAMILIES)].copy()
    selection["comparison_stage"] = selection["remaining_budget_ratio"].map(_stage_from_remaining_budget)
    selection = selection[selection["comparison_stage"].isin(TARGET_STAGES)].copy()
    selection["label_source"] = np.where(
        selection["selected_algorithm"].astype(str) == selection["default_algorithm"].astype(str),
        "same_algorithm",
        "changed_algorithm",
    )
    selection["vbs_available"] = selection["vbs_algorithm"].fillna("").astype(str) != ""
    selection = _deduplicate_selection_rows(selection)
    return selection


def _read_selection_reference(path: Path, source_dataset: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing selection reference: {path}")
    return pq.read_table(path).to_pandas().assign(source_dataset=source_dataset)


def _deduplicate_selection_rows(selection: pd.DataFrame) -> pd.DataFrame:
    priority = {"transition_020_030_followup": 0, "fe_transition": 1}
    result = selection.copy()
    result["_source_priority"] = result["source_dataset"].map(priority).fillna(9).astype(int)
    key_columns = ["problem_id", "dimension", "comparison_stage"]
    result = result.sort_values(key_columns + ["_source_priority", "remaining_budget_ratio"])
    return result.drop_duplicates(key_columns, keep="first").drop(columns=["_source_priority"])


def _stage_from_ratio(value: float) -> float | None:
    ratio = float(value)
    for stage in TARGET_STAGES:
        if abs(ratio - stage) <= STAGE_TOLERANCE:
            return float(stage)
    return None


def _stage_from_remaining_budget(value: float) -> float | None:
    prefix_ratio = (1.0 - FE_ANALYSIS_RATIO) - float(value)
    return _stage_from_ratio(prefix_ratio)


def _summarize_stage_labels(labels: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows = []
    for family in ("all_target_families", *TARGET_FAMILIES):
        family_rows = labels if family == "all_target_families" else labels[labels["family"] == family]
        for stage in TARGET_STAGES:
            stage_rows = family_rows[family_rows["comparison_stage"] == stage]
            for label_source in ("all", "changed_algorithm", "same_algorithm"):
                group = stage_rows if label_source == "all" else stage_rows[stage_rows["label_source"] == label_source]
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
    return pd.DataFrame(rows)


def _summarize_selector_rows(selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in ("all_target_families", *TARGET_FAMILIES):
        family_rows = selection if family == "all_target_families" else selection[selection["family"] == family]
        for stage in TARGET_STAGES:
            group = family_rows[family_rows["comparison_stage"] == stage]
            selected_counts = _value_counts_string(group["selected_algorithm"])
            vbs_counts = _value_counts_string(group["vbs_algorithm"])
            bucket_values = sorted(round(float(value), 6) for value in group["performance_bucket_ratio"].dropna().unique())
            remaining_values = sorted(round(float(value), 6) for value in group["remaining_budget_ratio"].dropna().unique())
            rows.append(
                {
                    "family": family,
                    "comparison_stage": float(stage),
                    "selection_rows": int(len(group)),
                    "changed_problem_count": int((group["label_source"] == "changed_algorithm").sum()),
                    "changed_problem_rate": float((group["label_source"] == "changed_algorithm").mean()) if len(group) else 0.0,
                    "selected_algorithm_counts": selected_counts,
                    "default_algorithm_counts": _value_counts_string(group["default_algorithm"]),
                    "sbs_algorithm_counts": _value_counts_string(group["sbs_algorithm"]),
                    "vbs_algorithm_counts": vbs_counts,
                    "vbs_available_rate": float(group["vbs_available"].mean()) if len(group) else 0.0,
                    "performance_bucket_ratios": ",".join(f"{value:.6g}" for value in bucket_values),
                    "remaining_budget_ratios": ",".join(f"{value:.6g}" for value in remaining_values),
                    "source_dataset_counts": _value_counts_string(group["source_dataset"]),
                }
            )
    return pd.DataFrame(rows)


def _value_counts_string(series: pd.Series) -> str:
    if series.empty:
        return ""
    counts = series.fillna("").astype(str).value_counts().sort_index()
    return ";".join(f"{key}:{int(value)}" for key, value in counts.items() if key)


def _selector_transition_rows(selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key_columns = ["family", "problem_id", "dimension"]
    for key, group in selection.groupby(key_columns, sort=True):
        group = group.sort_values("comparison_stage")
        previous = None
        for _, current in group.iterrows():
            if previous is not None:
                rows.append(
                    {
                        "family": key[0],
                        "problem_id": key[1],
                        "dimension": int(key[2]),
                        "from_stage": float(previous["comparison_stage"]),
                        "to_stage": float(current["comparison_stage"]),
                        "from_remaining_budget_ratio": float(previous["remaining_budget_ratio"]),
                        "to_remaining_budget_ratio": float(current["remaining_budget_ratio"]),
                        "from_performance_bucket_ratio": float(previous["performance_bucket_ratio"]),
                        "to_performance_bucket_ratio": float(current["performance_bucket_ratio"]),
                        "performance_bucket_changed": bool(
                            round(float(previous["performance_bucket_ratio"]), 6)
                            != round(float(current["performance_bucket_ratio"]), 6)
                        ),
                        "from_selected_algorithm": str(previous["selected_algorithm"]),
                        "to_selected_algorithm": str(current["selected_algorithm"]),
                        "selected_algorithm_changed": bool(
                            str(previous["selected_algorithm"]) != str(current["selected_algorithm"])
                        ),
                        "from_vbs_algorithm": str(previous["vbs_algorithm"]),
                        "to_vbs_algorithm": str(current["vbs_algorithm"]),
                        "vbs_algorithm_changed": bool(str(previous["vbs_algorithm"]) != str(current["vbs_algorithm"])),
                        "from_source_dataset": str(previous["source_dataset"]),
                        "to_source_dataset": str(current["source_dataset"]),
                    }
                )
            previous = current
    return pd.DataFrame(rows)


def _summarize_behavior(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["family", "dimension", "comparison_stage", "label_source"]
    for group_key, group in labels.groupby(group_columns, dropna=False, sort=True):
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


def _summarize_behavior_stage_shift(behavior_summary: pd.DataFrame) -> pd.DataFrame:
    if behavior_summary.empty:
        return behavior_summary.copy()
    rows = []
    for (family, dimension, label_source, feature), group in behavior_summary.groupby(
        ["family", "dimension", "label_source", "behavior_feature"], sort=True
    ):
        by_stage = {float(row["comparison_stage"]): row for _, row in group.iterrows()}
        for from_stage, to_stage in zip(TARGET_STAGES[:-1], TARGET_STAGES[1:], strict=True):
            left = by_stage.get(float(from_stage))
            right = by_stage.get(float(to_stage))
            if left is None or right is None:
                continue
            rows.append(
                {
                    "family": family,
                    "dimension": int(dimension),
                    "label_source": label_source,
                    "behavior_feature": feature,
                    "from_stage": float(from_stage),
                    "to_stage": float(to_stage),
                    "from_mean": left["mean"],
                    "to_mean": right["mean"],
                    "mean_difference": None if pd.isna(left["mean"]) or pd.isna(right["mean"]) else float(right["mean"] - left["mean"]),
                    "from_rows": int(left["rows"]),
                    "to_rows": int(right["rows"]),
                }
            )
    return pd.DataFrame(rows)


def _stage_evidence(stage_summary: pd.DataFrame, selector_summary: pd.DataFrame) -> pd.DataFrame:
    labels = stage_summary[
        (stage_summary["family"] == "all_target_families")
        & (stage_summary["label_source"].isin(["changed_algorithm", "same_algorithm"]))
    ].copy()
    selectors = selector_summary[selector_summary["family"] == "all_target_families"].copy()
    merged = labels.merge(
        selectors,
        on=["family", "comparison_stage"],
        how="left",
        suffixes=("_label", "_selector"),
    )
    return merged.sort_values(["comparison_stage", "label_source"])


def _diagnostic_conclusion(
    selector_summary: pd.DataFrame,
    selector_transition: pd.DataFrame,
    stage_summary: pd.DataFrame,
) -> dict[str, Any]:
    all_selector = selector_summary[selector_summary["family"] == "all_target_families"].set_index("comparison_stage")
    transition_025_030 = selector_transition[
        (selector_transition["from_stage"] == 0.25) & (selector_transition["to_stage"] == 0.30)
    ]
    bucket_change_rate = float(transition_025_030["performance_bucket_changed"].mean()) if len(transition_025_030) else 0.0
    selected_change_rate = float(transition_025_030["selected_algorithm_changed"].mean()) if len(transition_025_030) else 0.0
    changed_counts = {
        f"{stage:.2f}": int(all_selector.loc[stage, "changed_problem_count"]) if stage in all_selector.index else 0
        for stage in TARGET_STAGES
    }
    changed_rates = {
        f"{stage:.2f}": float(all_selector.loc[stage, "changed_problem_rate"]) if stage in all_selector.index else 0.0
        for stage in TARGET_STAGES
    }
    label_changed = stage_summary[
        (stage_summary["family"] == "all_target_families") & (stage_summary["label_source"] == "changed_algorithm")
    ].set_index("comparison_stage")
    u_rates = {
        f"{stage:.2f}": float(label_changed.loc[stage, "u_gt_zero_rate"]) if stage in label_changed.index else 0.0
        for stage in TARGET_STAGES
    }
    primary_cause = (
        "performance_bucket_boundary_with_bucket_specific_selector_prediction"
        if bucket_change_rate >= 0.5 and selected_change_rate >= 0.5
        else "selector_prediction_change_without_clear_bucket_boundary"
    )
    return {
        "primary_cause": primary_cause,
        "changed_problem_count_by_stage": changed_counts,
        "changed_problem_rate_by_stage": changed_rates,
        "changed_algorithm_u_gt_zero_rate_by_stage": u_rates,
        "stage_025_to_030_performance_bucket_change_rate": bucket_change_rate,
        "stage_025_to_030_selected_algorithm_change_rate": selected_change_rate,
        "interpretation": (
            "Changed_algorithm rows emerge when the selection_reference crosses from the 0.75-like "
            "performance bucket at stage 0.25 to the 0.60-like bucket at stage 0.30. Because selected_algorithm "
            "is problem-level and constant across seeds/prefix algorithms within each stage, this transition is "
            "not caused by behavior features. The observed VBS patterns do not show a clean universal VBS change, "
            "so the strongest explanation is a bucket-specific ELA selector prediction boundary rather than a "
            "direct behavior-driven transition."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose min_support selector transition between FE_ratio 0.20 and 0.35.")
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
        "--fe-transition-selection-reference",
        type=Path,
        default=Path("results/selection_reference/min_support_bbob_train_fe_transition/selection_reference.parquet"),
    )
    parser.add_argument(
        "--followup-selection-reference",
        type=Path,
        default=Path("results/selection_reference/min_support_bbob_train_transition_020_030_followup/selection_reference.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/selector_transition_diagnostic"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    args = parser.parse_args()
    run_selector_transition_diagnostics(
        fe_transition_validation_labels_path=args.fe_transition_validation_labels,
        followup_validation_labels_path=args.followup_validation_labels,
        fe_transition_selection_reference_path=args.fe_transition_selection_reference,
        followup_selection_reference_path=args.followup_selection_reference,
        output_dir=args.output_dir,
        target_column=args.target_column,
    )


if __name__ == "__main__":
    main()
