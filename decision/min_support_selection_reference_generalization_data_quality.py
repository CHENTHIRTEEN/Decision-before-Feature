from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from decision.min_support_diagnostics import _group_label
from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _check_family_split, _check_target, _json_default, _read_labels
from decision.min_support_performance_bucket_sensitivity import run_performance_bucket_sensitivity


COVERAGE_GROUP_LAYERS: dict[str, list[str]] = {
    "overall": [],
    "split": ["dataset_role"],
    "family": ["dataset_role", "family"],
    "dimension": ["dataset_role", "dimension"],
    "fe_ratio": ["dataset_role", "FE_ratio"],
    "family_dimension_fe_ratio": ["dataset_role", "family", "dimension", "FE_ratio"],
    "label_source": ["dataset_role", "label_source"],
    "family_dimension_fe_ratio_label_source": ["dataset_role", "family", "dimension", "FE_ratio", "label_source"],
    "algorithm_pair": ["dataset_role", "default_algorithm", "selected_algorithm"],
    "family_dimension_fe_ratio_algorithm_pair": [
        "dataset_role",
        "family",
        "dimension",
        "FE_ratio",
        "label_source",
        "default_algorithm",
        "selected_algorithm",
    ],
}

GAP_GROUP_COLUMNS = ["dimension", "FE_ratio", "label_source", "default_algorithm", "selected_algorithm"]
TARGET_FAMILIES = ("bbob_f005", "bbob_f019", "bbob_f024")


def run_selection_reference_generalization_data_quality(
    *,
    train_labels_path: Path,
    validation_labels_path: Path,
    selection_reference_path: Path,
    output_dir: Path,
    target_column: str,
    fe_transition_validation_labels_path: Path,
    followup_validation_labels_path: Path,
    fe_transition_selection_reference_path: Path,
    followup_selection_reference_path: Path,
    fe_transition_trajectory_root: Path,
) -> dict[str, Any]:
    _check_target(target_column)
    train = _read_labels(train_labels_path).assign(dataset_role="train")
    validation = _read_labels(validation_labels_path).assign(dataset_role="validation")
    _check_family_split(train, validation)

    coverage_dir = output_dir / "coverage_support"
    bucket_dir = output_dir / "bucket_sparsity"
    coverage = _coverage_rows(train=train, validation=validation, selection_reference_path=selection_reference_path, target_column=target_column)
    coverage_map = _coverage_map(coverage, target_column)
    coverage_gap = _coverage_gap(coverage, target_column)
    coverage_summary = _coverage_summary(coverage, coverage_gap, target_column)

    coverage_dir.mkdir(parents=True, exist_ok=True)
    coverage_rows_path = coverage_dir / "coverage_support_rows.parquet"
    coverage_map_path = coverage_dir / "coverage_support_map.parquet"
    coverage_gap_path = coverage_dir / "coverage_support_gap.parquet"
    coverage_summary_path = coverage_dir / "coverage_support_summary.json"
    pq.write_table(pa.Table.from_pandas(coverage, preserve_index=False), coverage_rows_path)
    pq.write_table(pa.Table.from_pandas(coverage_map, preserve_index=False), coverage_map_path)
    pq.write_table(pa.Table.from_pandas(coverage_gap, preserve_index=False), coverage_gap_path)

    coverage_summary["outputs"] = {
        "coverage_rows": str(coverage_rows_path),
        "coverage_support_map": str(coverage_map_path),
        "coverage_support_gap": str(coverage_gap_path),
        "summary": str(coverage_summary_path),
    }
    coverage_summary_path.write_text(json.dumps(coverage_summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")

    bucket_summary = run_performance_bucket_sensitivity(
        fe_transition_validation_labels_path=fe_transition_validation_labels_path,
        followup_validation_labels_path=followup_validation_labels_path,
        fe_transition_selection_reference_path=fe_transition_selection_reference_path,
        followup_selection_reference_path=followup_selection_reference_path,
        fe_transition_trajectory_root=fe_transition_trajectory_root,
        output_dir=bucket_dir,
        target_column=target_column,
    )
    scenario_rows_path = Path(bucket_summary["outputs"]["scenario_rows"])
    scenario_summary_path = Path(bucket_summary["outputs"]["scenario_summary"])
    bucket_scenario_table_path = bucket_dir / "bucket_scenario_table.parquet"
    bucket_scenario_summary_path = bucket_dir / "bucket_scenario_summary.parquet"
    pq.write_table(pq.read_table(scenario_rows_path), bucket_scenario_table_path)
    pq.write_table(pq.read_table(scenario_summary_path), bucket_scenario_summary_path)

    summary_path = output_dir / "selection_reference_generalization_data_quality_summary.json"
    report_path = output_dir / "selection_reference_generalization_data_quality_report.md"
    summary = {
        "experiment": "selection_reference_generalization_data_quality_h1_h2",
        "research_question": (
            "Do train/validation coverage gaps and sparse nearest performance-bucket mapping explain the observed "
            "selection_reference generalization failures?"
        ),
        "target_column": target_column,
        "hypotheses": {
            "H1": "training coverage insufficiency",
            "H2": "sparse performance bucket mapping",
        },
        "inputs": {
            "train_labels": str(train_labels_path),
            "validation_labels": str(validation_labels_path),
            "selection_reference": str(selection_reference_path),
            "fe_transition_validation_labels": str(fe_transition_validation_labels_path),
            "followup_validation_labels": str(followup_validation_labels_path),
            "fe_transition_selection_reference": str(fe_transition_selection_reference_path),
            "followup_selection_reference": str(followup_selection_reference_path),
            "fe_transition_trajectory_root": str(fe_transition_trajectory_root),
        },
        "outputs": {
            "coverage_support": coverage_summary["outputs"],
            "bucket_sparsity": {
                **bucket_summary["outputs"],
                "bucket_scenario_table": str(bucket_scenario_table_path),
                "bucket_scenario_summary": str(bucket_scenario_summary_path),
            },
            "summary": str(summary_path),
            "report": str(report_path),
        },
        "coverage_conclusion": coverage_summary["diagnostic_conclusion"],
        "bucket_conclusion": bucket_summary["diagnostic_conclusion"],
        "data_leakage_check": {
            "original_utility_labels_modified": False,
            "selection_reference_modified": False,
            "formal_phase1_configs_modified": False,
            "utility_labels_regenerated": False,
            "models_retrained": False,
            "decision_input_uses_ela_features": False,
        },
        "notes": [
            "Coverage support uses existing utility labels and joins existing selection_reference metadata only for VBS comparison.",
            "Bucket sparsity scenarios reuse existing selection_reference outputs from neighboring stages; alternate utility labels are not generated.",
            "Rows where selected_algorithm equals default_algorithm are kept and reported as same_algorithm rather than removed.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(_markdown_report(summary, coverage_summary, bucket_summary), encoding="utf-8")

    print(f"wrote H1 coverage rows to {coverage_rows_path}")
    print(f"wrote H1 coverage support map to {coverage_map_path}")
    print(f"wrote H1 coverage gap to {coverage_gap_path}")
    print(f"wrote H1 coverage summary to {coverage_summary_path}")
    print(f"wrote H2 bucket scenario table to {bucket_scenario_table_path}")
    print(f"wrote H2 bucket scenario summary to {bucket_scenario_summary_path}")
    print(f"wrote combined summary to {summary_path}")
    print(f"wrote combined report to {report_path}")
    return summary


def _coverage_rows(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    selection_reference_path: Path,
    target_column: str,
) -> pd.DataFrame:
    labels = pd.concat([train, validation], ignore_index=True).copy()
    labels["label_source"] = np.where(
        labels["selected_algorithm"].astype(str) == labels["default_algorithm"].astype(str),
        "same_algorithm",
        "changed_algorithm",
    )
    labels["utility_gt_zero"] = labels[target_column] > 0.0
    labels["positive_utility"] = np.where(labels["utility_gt_zero"], labels[target_column], 0.0)
    labels["remaining_budget_key"] = (labels["FE_ela_optimization"] / labels["FE_total"]).round(6)

    selection = _read_selection_reference(selection_reference_path)
    selection["remaining_budget_key"] = selection["remaining_budget_ratio"].round(6)
    selection = selection[
        [
            "split",
            "problem_id",
            "dimension",
            "remaining_budget_key",
            "performance_bucket_ratio",
            "selected_algorithm",
            "vbs_algorithm",
            "selector_status",
        ]
    ].rename(
        columns={
            "selected_algorithm": "selection_reference_selected_algorithm",
            "performance_bucket_ratio": "selection_reference_performance_bucket_ratio",
        }
    )
    selection = selection.drop_duplicates(["split", "problem_id", "dimension", "remaining_budget_key"])

    joined = labels.merge(
        selection,
        on=["split", "problem_id", "dimension", "remaining_budget_key"],
        how="left",
        validate="many_to_one",
    )
    joined["selection_reference_joined"] = joined["selector_status"].notna()
    joined["selection_reference_selected_matches_label"] = (
        joined["selection_reference_selected_algorithm"].fillna("").astype(str) == joined["selected_algorithm"].fillna("").astype(str)
    )
    joined["vbs_available"] = joined["vbs_algorithm"].fillna("").astype(str) != ""
    joined["selected_matches_vbs"] = joined["vbs_available"] & (
        joined["selected_algorithm"].fillna("").astype(str) == joined["vbs_algorithm"].fillna("").astype(str)
    )
    joined["target_family"] = joined["family"].isin(TARGET_FAMILIES)
    return joined.drop(columns=["remaining_budget_key"])


def _read_selection_reference(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing selection_reference: {path}")
    return pq.read_table(path).to_pandas()


def _coverage_map(coverage: pd.DataFrame, target_column: str) -> pd.DataFrame:
    frames = []
    for layer, columns in COVERAGE_GROUP_LAYERS.items():
        frames.append(_grouped_coverage_summary(coverage, layer, columns, target_column))
    return pd.concat(frames, ignore_index=True)


def _grouped_coverage_summary(
    coverage: pd.DataFrame,
    layer: str,
    group_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    if not group_columns:
        return pd.DataFrame([_coverage_summary_row(coverage, layer, {}, target_column)])
    rows = []
    for group_values, subset in coverage.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group = dict(zip(group_columns, group_values, strict=True))
        rows.append(_coverage_summary_row(subset, layer, group, target_column))
    return pd.DataFrame(rows)


def _coverage_summary_row(frame: pd.DataFrame, layer: str, group: dict[str, Any], target_column: str) -> dict[str, Any]:
    utility = frame[target_column].to_numpy(dtype=float)
    gain = frame["performance_gain_norm"].to_numpy(dtype=float)
    cost = frame["time_cost_norm"].to_numpy(dtype=float)
    positive = utility > 0.0
    joined = frame["selection_reference_joined"].to_numpy(dtype=bool)
    selected_matches_vbs = frame["selected_matches_vbs"].to_numpy(dtype=bool)
    vbs_available = frame["vbs_available"].to_numpy(dtype=bool)
    return {
        "layer": layer,
        "group": _group_label(group),
        "dataset_role": group.get("dataset_role"),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "label_source": group.get("label_source"),
        "default_algorithm": group.get("default_algorithm"),
        "selected_algorithm": group.get("selected_algorithm"),
        "rows": int(len(frame)),
        "problem_count": int(frame["problem_id"].nunique()) if len(frame) else 0,
        "prefix_algorithm_count": int(frame["prefix_algorithm"].nunique()) if len(frame) else 0,
        "seed_count": int(frame["seed"].nunique()) if len(frame) else 0,
        "changed_algorithm_rows": int((frame["label_source"] == "changed_algorithm").sum()),
        "changed_algorithm_rate": float((frame["label_source"] == "changed_algorithm").mean()) if len(frame) else 0.0,
        "utility_gt_zero_rows": int(np.sum(positive)),
        "utility_gt_zero_rate": float(np.mean(positive)) if len(frame) else 0.0,
        "utility_sum": float(np.sum(utility)) if len(frame) else 0.0,
        "mean_utility": float(np.mean(utility)) if len(frame) else None,
        "sum_positive_utility": float(np.sum(utility[positive])) if len(frame) else 0.0,
        "mean_positive_utility": float(np.mean(utility[positive])) if np.any(positive) else 0.0,
        "mean_performance_gain_norm": float(np.mean(gain)) if len(frame) else None,
        "mean_time_cost_norm": float(np.mean(cost)) if len(frame) else None,
        "p_skip_mean": float(frame["p_skip"].mean()) if len(frame) else None,
        "p_ela_mean": float(frame["p_ela"].mean()) if len(frame) else None,
        "selection_reference_joined_rows": int(np.sum(joined)),
        "selection_reference_joined_rate": float(np.mean(joined)) if len(frame) else 0.0,
        "selection_reference_selected_mismatch_rows": int(
            (frame["selection_reference_joined"] & ~frame["selection_reference_selected_matches_label"]).sum()
        ),
        "vbs_available_rows": int(np.sum(vbs_available)),
        "vbs_available_rate": float(np.mean(vbs_available)) if len(frame) else 0.0,
        "selected_matches_vbs_rows": int(np.sum(selected_matches_vbs)),
        "selected_matches_vbs_rate": (
            float(np.sum(selected_matches_vbs) / np.sum(vbs_available)) if np.sum(vbs_available) > 0 else 0.0
        ),
        "default_algorithm_counts": _value_counts_string(frame["default_algorithm"]),
        "selected_algorithm_counts": _value_counts_string(frame["selected_algorithm"]),
        "vbs_algorithm_counts": _value_counts_string(frame["vbs_algorithm"]),
    }


def _coverage_gap(coverage: pd.DataFrame, target_column: str) -> pd.DataFrame:
    keep = [
        "dimension",
        "FE_ratio",
        "label_source",
        "default_algorithm",
        "selected_algorithm",
        "rows",
        "utility_gt_zero_rows",
        "utility_gt_zero_rate",
        "sum_positive_utility",
        "mean_utility",
        "selected_matches_vbs_rate",
    ]
    validation = _grouped_coverage_summary(
        coverage[coverage["dataset_role"] == "validation"], "gap_key", GAP_GROUP_COLUMNS, target_column
    )
    train = _grouped_coverage_summary(coverage[coverage["dataset_role"] == "train"], "gap_key", GAP_GROUP_COLUMNS, target_column)
    train = train[keep].rename(columns={column: f"train_{column}" for column in keep if column not in GAP_GROUP_COLUMNS})
    validation = validation[keep].rename(columns={column: f"validation_{column}" for column in keep if column not in GAP_GROUP_COLUMNS})
    result = train.merge(validation, on=GAP_GROUP_COLUMNS, how="outer")
    for column in ["rows", "utility_gt_zero_rows", "sum_positive_utility"]:
        result[f"train_{column}"] = result[f"train_{column}"].fillna(0)
        result[f"validation_{column}"] = result[f"validation_{column}"].fillna(0)
    for column in ["utility_gt_zero_rate", "mean_utility", "selected_matches_vbs_rate"]:
        result[f"train_{column}"] = result[f"train_{column}"].where(result[f"train_{column}"].notna(), np.nan)
        result[f"validation_{column}"] = result[f"validation_{column}"].where(result[f"validation_{column}"].notna(), np.nan)
    result["row_count_gap_validation_minus_train"] = result["validation_rows"] - result["train_rows"]
    result["u_gt_zero_count_gap_validation_minus_train"] = (
        result["validation_utility_gt_zero_rows"] - result["train_utility_gt_zero_rows"]
    )
    result["u_gt_zero_rate_gap_validation_minus_train"] = (
        result["validation_utility_gt_zero_rate"] - result["train_utility_gt_zero_rate"]
    )
    result["positive_utility_sum_gap_validation_minus_train"] = (
        result["validation_sum_positive_utility"] - result["train_sum_positive_utility"]
    )
    return result.sort_values(
        ["u_gt_zero_count_gap_validation_minus_train", "positive_utility_sum_gap_validation_minus_train"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _coverage_summary(coverage: pd.DataFrame, gap: pd.DataFrame, target_column: str) -> dict[str, Any]:
    overall = {}
    for dataset_role, frame in coverage.groupby("dataset_role", sort=True):
        changed = frame[frame["label_source"] == "changed_algorithm"]
        same = frame[frame["label_source"] == "same_algorithm"]
        overall[str(dataset_role)] = {
            "rows": int(len(frame)),
            "families": sorted(frame["family"].astype(str).unique().tolist()),
            "changed_algorithm_rows": int(len(changed)),
            "changed_algorithm_u_gt_zero_rows": int((changed[target_column] > 0.0).sum()),
            "changed_algorithm_u_gt_zero_rate": float((changed[target_column] > 0.0).mean()) if len(changed) else 0.0,
            "same_algorithm_rows": int(len(same)),
            "same_algorithm_u_gt_zero_rows": int((same[target_column] > 0.0).sum()),
            "same_algorithm_u_gt_zero_rate": float((same[target_column] > 0.0).mean()) if len(same) else 0.0,
            "selected_matches_vbs_rate": _safe_rate(frame["selected_matches_vbs"], frame["vbs_available"]),
            "selection_reference_joined_rate": float(frame["selection_reference_joined"].mean()) if len(frame) else 0.0,
        }

    validation_changed = coverage[
        (coverage["dataset_role"] == "validation")
        & (coverage["label_source"] == "changed_algorithm")
        & (coverage[target_column] > 0.0)
    ]
    target_family_rows = []
    for family, frame in validation_changed[validation_changed["family"].isin(TARGET_FAMILIES)].groupby("family", sort=True):
        target_family_rows.append(
            {
                "family": str(family),
                "positive_rows": int(len(frame)),
                "positive_utility_sum": float(frame[target_column].sum()),
                "FE_ratios": sorted(round(float(value), 6) for value in frame["FE_ratio"].dropna().unique()),
                "selected_algorithm_counts": _value_counts_string(frame["selected_algorithm"]),
            }
        )

    top_gaps = gap.head(12).replace({np.nan: None}).to_dict(orient="records")
    train_changed = overall.get("train", {}).get("changed_algorithm_u_gt_zero_rate", 0.0)
    validation_changed_rate = overall.get("validation", {}).get("changed_algorithm_u_gt_zero_rate", 0.0)
    return {
        "experiment": "selection_reference_h1_coverage_support",
        "target_column": target_column,
        "rows": {role: values["rows"] for role, values in overall.items()},
        "overall": overall,
        "target_validation_changed_positive_by_family": target_family_rows,
        "top_validation_minus_train_gaps": top_gaps,
        "diagnostic_conclusion": {
            "primary_pattern": "validation_changed_algorithm_positive_utility_exceeds_train_support",
            "train_changed_algorithm_u_gt_zero_rate": float(train_changed),
            "validation_changed_algorithm_u_gt_zero_rate": float(validation_changed_rate),
            "rate_gap_validation_minus_train": float(validation_changed_rate - train_changed),
            "interpretation": (
                "The support map compares existing train and validation utility labels without modifying labels or "
                "selection_reference. A larger validation changed_algorithm U_ELA>0 rate than train indicates that "
                "the fixed selection_reference is being evaluated in regions with limited training support."
            ),
        },
        "data_leakage_check": {
            "original_utility_labels_modified": False,
            "selection_reference_modified": False,
            "formal_phase1_configs_modified": False,
            "utility_labels_regenerated": False,
            "models_retrained": False,
        },
    }


def _markdown_report(summary: dict[str, Any], coverage_summary: dict[str, Any], bucket_summary: dict[str, Any]) -> str:
    coverage_conclusion = coverage_summary["diagnostic_conclusion"]
    bucket_conclusion = bucket_summary["diagnostic_conclusion"]
    lines = [
        "# selection_reference 泛化失败 H1/H2 最小诊断",
        "",
        "本报告只使用当前项目内已有 `utility_labels`、`selection_reference` 和 trajectory 输出；未修改原始结果，未重训模型，未生成新的 utility labels。",
        "",
        "## H1 训练覆盖不足",
        "",
        (
            f"- train changed_algorithm `U_ELA>0` rate: "
            f"`{coverage_conclusion['train_changed_algorithm_u_gt_zero_rate']:.6f}`"
        ),
        (
            f"- validation changed_algorithm `U_ELA>0` rate: "
            f"`{coverage_conclusion['validation_changed_algorithm_u_gt_zero_rate']:.6f}`"
        ),
        (
            f"- validation - train rate gap: "
            f"`{coverage_conclusion['rate_gap_validation_minus_train']:.6f}`"
        ),
        "",
        "目标 validation family 中 changed_algorithm 且 `U_ELA>0` 的分布：",
        "",
        "| family | positive rows | positive utility sum | FE ratios | selected_algorithm counts |",
        "|---|---:|---:|---|---|",
    ]
    for row in coverage_summary["target_validation_changed_positive_by_family"]:
        lines.append(
            f"| `{row['family']}` | {row['positive_rows']} | {row['positive_utility_sum']:.6f} | "
            f"`{row['FE_ratios']}` | `{row['selected_algorithm_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## H2 performance bucket 稀疏",
            "",
            f"- primary cause: `{bucket_conclusion['primary_cause']}`",
            (
                f"- 0.25 current bucket changed problem rate: "
                f"`{bucket_conclusion['stage_025_current_changed_problem_rate']:.6f}`"
            ),
            (
                f"- 0.25 lower-bucket proxy changed problem rate: "
                f"`{bucket_conclusion['stage_025_lower_bucket_changed_problem_rate']:.6f}`"
            ),
            (
                f"- 0.30 current bucket changed problem rate: "
                f"`{bucket_conclusion['stage_030_current_changed_problem_rate']:.6f}`"
            ),
            (
                f"- 0.30 upper-bucket proxy changed problem rate: "
                f"`{bucket_conclusion['stage_030_upper_bucket_changed_problem_rate']:.6f}`"
            ),
            "",
            "## 输出文件",
            "",
            f"- H1 coverage support map: `{summary['outputs']['coverage_support']['coverage_support_map']}`",
            f"- H1 coverage gap: `{summary['outputs']['coverage_support']['coverage_support_gap']}`",
            f"- H2 bucket scenario table: `{summary['outputs']['bucket_sparsity']['bucket_scenario_table']}`",
            f"- H2 bucket scenario summary: `{summary['outputs']['bucket_sparsity']['bucket_scenario_summary']}`",
            f"- combined summary: `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> float:
    denom = int(denominator.sum())
    if denom <= 0:
        return 0.0
    return float(numerator.sum() / denom)


def _value_counts_string(series: pd.Series) -> str:
    if series.empty:
        return ""
    counts = series.fillna("").astype(str).value_counts().sort_index()
    return ";".join(f"{key}:{int(value)}" for key, value in counts.items() if key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H1/H2 diagnostics for selection_reference generalization data quality.")
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
        "--selection-reference",
        type=Path,
        default=Path("results/selection_reference/min_support_bbob_train/selection_reference.parquet"),
    )
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
        "--fe-transition-trajectory-root",
        type=Path,
        default=Path("results/phase1/min_support_bbob_validation_fe_transition"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/selection_reference_generalization_data_quality"),
    )
    args = parser.parse_args()
    run_selection_reference_generalization_data_quality(
        train_labels_path=args.train_labels,
        validation_labels_path=args.validation_labels,
        selection_reference_path=args.selection_reference,
        output_dir=args.output_dir,
        target_column=args.target_column,
        fe_transition_validation_labels_path=args.fe_transition_validation_labels,
        followup_validation_labels_path=args.followup_validation_labels,
        fe_transition_selection_reference_path=args.fe_transition_selection_reference,
        followup_selection_reference_path=args.followup_selection_reference,
        fe_transition_trajectory_root=args.fe_transition_trajectory_root,
    )


if __name__ == "__main__":
    main()
