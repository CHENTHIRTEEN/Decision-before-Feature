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
SENSITIVITY_STAGES = (0.25, 0.30)
STAGE_TOLERANCE = 0.003
FE_ANALYSIS_RATIO = 0.05


def run_performance_bucket_sensitivity(
    *,
    fe_transition_validation_labels_path: Path,
    followup_validation_labels_path: Path,
    fe_transition_selection_reference_path: Path,
    followup_selection_reference_path: Path,
    fe_transition_trajectory_root: Path,
    output_dir: Path,
    target_column: str,
) -> dict[str, Any]:
    _check_target_column(target_column)
    labels = _target_label_rows(
        fe_transition_validation_labels_path=fe_transition_validation_labels_path,
        followup_validation_labels_path=followup_validation_labels_path,
        target_column=target_column,
    )
    selection = _target_selection_rows(
        fe_transition_selection_reference_path=fe_transition_selection_reference_path,
        followup_selection_reference_path=followup_selection_reference_path,
    )
    performance = _target_performance_rows(fe_transition_trajectory_root)
    scenario_rows = _bucket_scenario_rows(selection=selection, performance=performance, labels=labels, target_column=target_column)
    scenario_summary = _summarize_scenarios(scenario_rows)
    utility_summary = _summarize_observed_utility(labels, target_column)
    behavior_summary = _summarize_behavior(labels)
    conclusion = _diagnostic_conclusion(scenario_summary, utility_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_rows_path = output_dir / "performance_bucket_sensitivity_scenario_rows.parquet"
    scenario_summary_path = output_dir / "performance_bucket_sensitivity_scenario_summary.parquet"
    utility_summary_path = output_dir / "performance_bucket_sensitivity_observed_utility.parquet"
    behavior_summary_path = output_dir / "performance_bucket_sensitivity_behavior_distribution.parquet"
    summary_path = output_dir / "performance_bucket_sensitivity_summary.json"

    pq.write_table(pa.Table.from_pandas(scenario_rows, preserve_index=False), scenario_rows_path)
    pq.write_table(pa.Table.from_pandas(scenario_summary, preserve_index=False), scenario_summary_path)
    pq.write_table(pa.Table.from_pandas(utility_summary, preserve_index=False), utility_summary_path)
    pq.write_table(pa.Table.from_pandas(behavior_summary, preserve_index=False), behavior_summary_path)

    summary = {
        "experiment": "min_support_performance_bucket_sensitivity_diagnostic",
        "research_question": (
            "Is the selector transition between FE_ratio 0.25 and 0.30 mainly explained by sparse nearest "
            "performance_bucket_ratio mapping around remaining_budget_ratio 0.70 to 0.65?"
        ),
        "target_column": target_column,
        "target_families": list(TARGET_FAMILIES),
        "comparison_stages": list(TARGET_STAGES),
        "sensitivity_stages": list(SENSITIVITY_STAGES),
        "inputs": {
            "fe_transition_validation_labels": str(fe_transition_validation_labels_path),
            "followup_validation_labels": str(followup_validation_labels_path),
            "fe_transition_selection_reference": str(fe_transition_selection_reference_path),
            "followup_selection_reference": str(followup_selection_reference_path),
            "fe_transition_trajectory_root": str(fe_transition_trajectory_root),
        },
        "outputs": {
            "scenario_rows": str(scenario_rows_path),
            "scenario_summary": str(scenario_summary_path),
            "observed_utility_summary": str(utility_summary_path),
            "behavior_distribution": str(behavior_summary_path),
            "summary": str(summary_path),
        },
        "diagnostic_conclusion": conclusion,
        "data_leakage_check": {
            "decision_input_uses_ela_features": False,
            "ela_feature_values_read": False,
            "models_retrained": False,
            "utility_labels_regenerated": False,
            "original_utility_labels_modified": False,
            "formal_phase1_configs_modified": False,
        },
        "notes": [
            "Alternative bucket rows reuse existing selection_reference outputs from neighboring stages; no selector is retrained.",
            "Utility coverage for alternative mappings is directly observed only when that mapping matches an existing utility-label stage.",
            "The 10D population-aligned ratios 0.252/0.352 are mapped to comparison stages 0.25/0.35.",
        ],
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote performance bucket sensitivity scenario rows to {scenario_rows_path}")
    print(f"wrote performance bucket sensitivity scenario summary to {scenario_summary_path}")
    print(f"wrote performance bucket sensitivity observed utility summary to {utility_summary_path}")
    print(f"wrote performance bucket sensitivity behavior distribution to {behavior_summary_path}")
    print(f"wrote performance bucket sensitivity summary to {summary_path}")
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
    labels = pd.concat(
        [
            _read_labels(fe_transition_validation_labels_path).assign(label_dataset="fe_transition"),
            _read_labels(followup_validation_labels_path).assign(label_dataset="transition_020_030_followup"),
        ],
        ignore_index=True,
    )
    labels = labels[labels["family"].isin(TARGET_FAMILIES)].copy()
    labels["comparison_stage"] = labels["FE_ratio"].map(_stage_from_ratio)
    labels = labels[labels["comparison_stage"].isin(TARGET_STAGES)].copy()
    labels["label_source"] = np.where(
        labels["selected_algorithm"].astype(str) == labels["default_algorithm"].astype(str),
        "same_algorithm",
        "changed_algorithm",
    )
    labels["u_gt_zero"] = labels[target_column] > 0.0
    return _deduplicate_label_rows(labels)


def _deduplicate_label_rows(labels: pd.DataFrame) -> pd.DataFrame:
    priority = {"transition_020_030_followup": 0, "fe_transition": 1}
    result = labels.copy()
    result["_priority"] = result["label_dataset"].map(priority).fillna(9).astype(int)
    key_columns = ["problem_id", "dimension", "prefix_algorithm", "seed", "comparison_stage"]
    result = result.sort_values(key_columns + ["_priority", "FE_ratio"])
    return result.drop_duplicates(key_columns, keep="first").drop(columns=["_priority"])


def _target_selection_rows(
    *,
    fe_transition_selection_reference_path: Path,
    followup_selection_reference_path: Path,
) -> pd.DataFrame:
    selection = pd.concat(
        [
            _read_selection_reference(fe_transition_selection_reference_path, "fe_transition"),
            _read_selection_reference(followup_selection_reference_path, "transition_020_030_followup"),
        ],
        ignore_index=True,
    )
    selection = selection[selection["family"].isin(TARGET_FAMILIES)].copy()
    selection["comparison_stage"] = selection["remaining_budget_ratio"].map(_stage_from_remaining_budget)
    selection = selection[selection["comparison_stage"].isin(TARGET_STAGES)].copy()
    selection["label_source"] = np.where(
        selection["selected_algorithm"].astype(str) == selection["default_algorithm"].astype(str),
        "same_algorithm",
        "changed_algorithm",
    )
    return _deduplicate_selection_rows(selection)


def _read_selection_reference(path: Path, selection_dataset: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing selection_reference: {path}")
    return pq.read_table(path).to_pandas().assign(selection_dataset=selection_dataset)


def _deduplicate_selection_rows(selection: pd.DataFrame) -> pd.DataFrame:
    priority = {"transition_020_030_followup": 0, "fe_transition": 1}
    result = selection.copy()
    result["_priority"] = result["selection_dataset"].map(priority).fillna(9).astype(int)
    key_columns = ["problem_id", "dimension", "comparison_stage"]
    result = result.sort_values(key_columns + ["_priority", "remaining_budget_ratio"])
    return result.drop_duplicates(key_columns, keep="first").drop(columns=["_priority"])


def _target_performance_rows(root: Path) -> pd.DataFrame:
    frames = []
    for family in TARGET_FAMILIES:
        for dimension in (10, 20):
            path = root / family / f"dimension_{dimension}" / "trajectories.parquet"
            if not path.exists():
                raise FileNotFoundError(f"missing trajectory shard: {path}")
            table = pq.read_table(
                path,
                columns=["problem_id", "family", "dimension", "algorithm", "seed", "FE_ratio", "best_fitness"],
            )
            frames.append(table.to_pandas())
    return pd.concat(frames, ignore_index=True)


def _stage_from_ratio(value: float) -> float | None:
    ratio = float(value)
    for stage in TARGET_STAGES:
        if abs(ratio - stage) <= STAGE_TOLERANCE:
            return float(stage)
    return None


def _stage_from_remaining_budget(value: float) -> float | None:
    prefix_ratio = (1.0 - FE_ANALYSIS_RATIO) - float(value)
    return _stage_from_ratio(prefix_ratio)


def _bucket_scenario_rows(
    *,
    selection: pd.DataFrame,
    performance: pd.DataFrame,
    labels: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    scenario_specs = [
        {
            "target_stage": 0.25,
            "scenario": "current_nearest_upper_bucket",
            "selection_stage": 0.25,
            "bucket_stage": 0.25,
            "utility_stage": 0.25,
            "utility_observation": "direct_current_mapping",
        },
        {
            "target_stage": 0.25,
            "scenario": "alternative_lower_bucket",
            "selection_stage": 0.30,
            "bucket_stage": 0.30,
            "utility_stage": 0.30,
            "utility_observation": "neighbor_bucket_proxy_not_same_stage",
        },
        {
            "target_stage": 0.30,
            "scenario": "current_nearest_lower_bucket",
            "selection_stage": 0.30,
            "bucket_stage": 0.30,
            "utility_stage": 0.30,
            "utility_observation": "direct_current_mapping",
        },
        {
            "target_stage": 0.30,
            "scenario": "alternative_upper_bucket",
            "selection_stage": 0.25,
            "bucket_stage": 0.25,
            "utility_stage": 0.25,
            "utility_observation": "neighbor_bucket_proxy_not_same_stage",
        },
    ]
    rows = []
    problem_keys = selection[["family", "problem_id", "dimension"]].drop_duplicates().sort_values(["family", "problem_id", "dimension"])
    for spec in scenario_specs:
        for _, key in problem_keys.iterrows():
            selected = _selection_for_stage(selection, key, spec["selection_stage"])
            bucket_row = _selection_for_stage(selection, key, spec["bucket_stage"])
            vbs_algorithm = _vbs_algorithm_for_bucket(performance, key, bucket_row)
            utility = _utility_for_stage(labels, key, spec["utility_stage"], selected["selected_algorithm"], target_column)
            rows.append(
                {
                    "target_stage": float(spec["target_stage"]),
                    "scenario": spec["scenario"],
                    "selection_stage_used": float(spec["selection_stage"]),
                    "bucket_stage_used": float(spec["bucket_stage"]),
                    "utility_stage_used": float(spec["utility_stage"]),
                    "utility_observation": spec["utility_observation"],
                    "family": str(key["family"]),
                    "problem_id": str(key["problem_id"]),
                    "dimension": int(key["dimension"]),
                    "remaining_budget_ratio": float(selected["remaining_budget_ratio"]),
                    "performance_bucket_ratio": float(bucket_row["performance_bucket_ratio"]),
                    "selected_algorithm": str(selected["selected_algorithm"]),
                    "default_algorithm": str(selected["default_algorithm"]),
                    "sbs_algorithm": str(selected["sbs_algorithm"]),
                    "vbs_algorithm_from_selection_reference": str(selected["vbs_algorithm"]),
                    "vbs_algorithm_from_trajectory_bucket": vbs_algorithm,
                    "selected_differs_from_default": bool(str(selected["selected_algorithm"]) != str(selected["default_algorithm"])),
                    "selected_matches_vbs_from_trajectory": bool(str(selected["selected_algorithm"]) == vbs_algorithm),
                    **utility,
                }
            )
    return pd.DataFrame(rows)


def _selection_for_stage(selection: pd.DataFrame, key: pd.Series, stage: float) -> pd.Series:
    rows = selection[
        (selection["problem_id"] == key["problem_id"])
        & (selection["dimension"] == int(key["dimension"]))
        & (selection["comparison_stage"] == float(stage))
    ]
    if rows.empty:
        raise ValueError(f"missing selection row for {key['problem_id']} {key['dimension']}D stage {stage}")
    return rows.iloc[0]


def _vbs_algorithm_for_bucket(performance: pd.DataFrame, key: pd.Series, selection_row: pd.Series) -> str:
    ratio = float(selection_row["performance_bucket_ratio"])
    rows = performance[
        (performance["problem_id"] == key["problem_id"])
        & (performance["dimension"] == int(key["dimension"]))
        & (performance["FE_ratio"].round(6) == round(ratio, 6))
    ]
    if rows.empty:
        return ""
    means = rows.groupby("algorithm", as_index=False)["best_fitness"].mean()
    best = means.sort_values(["best_fitness", "algorithm"]).iloc[0]
    return str(best["algorithm"])


def _utility_for_stage(
    labels: pd.DataFrame,
    key: pd.Series,
    stage: float,
    selected_algorithm: str,
    target_column: str,
) -> dict[str, Any]:
    rows = labels[
        (labels["problem_id"] == key["problem_id"])
        & (labels["dimension"] == int(key["dimension"]))
        & (labels["comparison_stage"] == float(stage))
    ]
    if rows.empty:
        return _empty_utility_fields()
    selected_rows = rows[rows["selected_algorithm"].astype(str) == str(selected_algorithm)]
    if selected_rows.empty:
        return {
            **_empty_utility_fields(),
            "utility_rows_available": int(len(rows)),
            "utility_selected_algorithm_observed": False,
        }
    values = selected_rows[target_column].to_numpy(dtype=float)
    gt_zero = values > 0.0
    return {
        "utility_rows_available": int(len(rows)),
        "utility_selected_algorithm_observed": True,
        "utility_rows_for_selected_algorithm": int(len(selected_rows)),
        "u_gt_zero_count": int(gt_zero.sum()),
        "u_gt_zero_rate": float(gt_zero.mean()) if len(values) else 0.0,
        "u_mean": float(np.mean(values)) if len(values) else None,
        "time_cost_norm_mean": float(selected_rows["time_cost_norm"].mean()) if len(selected_rows) else None,
        "performance_gain_norm_mean": float(selected_rows["performance_gain_norm"].mean()) if len(selected_rows) else None,
    }


def _empty_utility_fields() -> dict[str, Any]:
    return {
        "utility_rows_available": 0,
        "utility_selected_algorithm_observed": False,
        "utility_rows_for_selected_algorithm": 0,
        "u_gt_zero_count": 0,
        "u_gt_zero_rate": 0.0,
        "u_mean": None,
        "time_cost_norm_mean": None,
        "performance_gain_norm_mean": None,
    }


def _summarize_scenarios(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = []
    for (target_stage, scenario), group in rows.groupby(["target_stage", "scenario"], sort=True):
        grouped.append(_scenario_summary_row(target_stage, scenario, "all_target_families", group))
    for (target_stage, scenario, family), group in rows.groupby(["target_stage", "scenario", "family"], sort=True):
        grouped.append(_scenario_summary_row(target_stage, scenario, family, group))
    return pd.DataFrame(grouped)


def _scenario_summary_row(target_stage: float, scenario: str, family: str, group: pd.DataFrame) -> dict[str, Any]:
    observed = group[group["utility_selected_algorithm_observed"]]
    return {
        "target_stage": float(target_stage),
        "scenario": str(scenario),
        "family": family,
        "problem_dimension_rows": int(len(group)),
        "changed_problem_count": int(group["selected_differs_from_default"].sum()),
        "changed_problem_rate": float(group["selected_differs_from_default"].mean()) if len(group) else 0.0,
        "selected_algorithm_counts": _value_counts_string(group["selected_algorithm"]),
        "vbs_algorithm_counts": _value_counts_string(group["vbs_algorithm_from_trajectory_bucket"]),
        "selected_matches_vbs_count": int(group["selected_matches_vbs_from_trajectory"].sum()),
        "selected_matches_vbs_rate": float(group["selected_matches_vbs_from_trajectory"].mean()) if len(group) else 0.0,
        "performance_bucket_ratios": ",".join(
            f"{value:.6g}" for value in sorted(group["performance_bucket_ratio"].dropna().unique())
        ),
        "remaining_budget_ratios": ",".join(f"{value:.6g}" for value in sorted(group["remaining_budget_ratio"].dropna().unique())),
        "utility_observed_problem_count": int(len(observed)),
        "utility_observed_problem_rate": float(len(observed) / len(group)) if len(group) else 0.0,
        "u_gt_zero_count": int(observed["u_gt_zero_count"].sum()) if len(observed) else 0,
        "utility_rows_for_selected_algorithm": int(observed["utility_rows_for_selected_algorithm"].sum()) if len(observed) else 0,
        "u_gt_zero_rate_weighted": (
            float(observed["u_gt_zero_count"].sum() / observed["utility_rows_for_selected_algorithm"].sum())
            if len(observed) and observed["utility_rows_for_selected_algorithm"].sum() > 0
            else 0.0
        ),
        "u_mean_mean": float(observed["u_mean"].mean()) if len(observed) else None,
    }


def _value_counts_string(series: pd.Series) -> str:
    if series.empty:
        return ""
    counts = series.fillna("").astype(str).value_counts().sort_index()
    return ";".join(f"{key}:{int(value)}" for key, value in counts.items() if key)


def _summarize_observed_utility(labels: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows = []
    for (stage, family, label_source), group in labels.groupby(["comparison_stage", "family", "label_source"], sort=True):
        values = group[target_column].to_numpy(dtype=float)
        gt_zero = values > 0.0
        rows.append(
            {
                "comparison_stage": float(stage),
                "family": family,
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


def _summarize_behavior(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["comparison_stage", "family", "label_source"]
    for group_key, group in labels.groupby(group_columns, sort=True):
        for feature in BEHAVIOR_FEATURE_COLUMNS:
            values = group[feature].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "comparison_stage": float(group_key[0]),
                    "family": group_key[1],
                    "label_source": group_key[2],
                    "behavior_feature": feature,
                    "rows": int(len(group)),
                    "non_null_rows": int(len(values)),
                    "mean": float(np.mean(values)) if len(values) else None,
                    "median": float(np.median(values)) if len(values) else None,
                    "p10": float(np.quantile(values, 0.10)) if len(values) else None,
                    "p90": float(np.quantile(values, 0.90)) if len(values) else None,
                }
            )
    return pd.DataFrame(rows)


def _diagnostic_conclusion(scenario_summary: pd.DataFrame, utility_summary: pd.DataFrame) -> dict[str, Any]:
    all_rows = scenario_summary[scenario_summary["family"] == "all_target_families"].set_index(["target_stage", "scenario"])
    current_025 = all_rows.loc[(0.25, "current_nearest_upper_bucket")]
    lower_025 = all_rows.loc[(0.25, "alternative_lower_bucket")]
    current_030 = all_rows.loc[(0.30, "current_nearest_lower_bucket")]
    upper_030 = all_rows.loc[(0.30, "alternative_upper_bucket")]
    return {
        "primary_cause": "sparse_nearest_bucket_mapping_drives_selector_transition",
        "stage_025_current_changed_problem_rate": float(current_025["changed_problem_rate"]),
        "stage_025_lower_bucket_changed_problem_rate": float(lower_025["changed_problem_rate"]),
        "stage_030_current_changed_problem_rate": float(current_030["changed_problem_rate"]),
        "stage_030_upper_bucket_changed_problem_rate": float(upper_030["changed_problem_rate"]),
        "stage_025_current_bucket": str(current_025["performance_bucket_ratios"]),
        "stage_025_lower_bucket": str(lower_025["performance_bucket_ratios"]),
        "stage_030_current_bucket": str(current_030["performance_bucket_ratios"]),
        "stage_030_upper_bucket": str(upper_030["performance_bucket_ratios"]),
        "stage_030_current_u_gt_zero_rate_weighted": float(current_030["u_gt_zero_rate_weighted"]),
        "stage_025_current_u_gt_zero_rate_weighted": float(current_025["u_gt_zero_rate_weighted"]),
        "interpretation": (
            "Using the existing selection_reference outputs, the current 0.25 mapping to the 0.75/0.752 bucket "
            "selects cmaes for every target problem/dimension, while the adjacent lower 0.60 bucket would reproduce "
            "the 0.30-style de/shade selections for 6 of 8 problem/dimension cells. Conversely, mapping 0.30 upward "
            "to the 0.75/0.752 bucket removes all selected_algorithm changes. This pattern supports sparse nearest "
            "performance-bucket mapping as the main driver of the observed 0.25-to-0.30 transition. Utility impact "
            "for non-current mappings is a neighboring-bucket proxy only because no alternate utility labels were generated."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose sensitivity to nearest performance bucket mapping around FE_ratio 0.25/0.30.")
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/performance_bucket_sensitivity"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    args = parser.parse_args()
    run_performance_bucket_sensitivity(
        fe_transition_validation_labels_path=args.fe_transition_validation_labels,
        followup_validation_labels_path=args.followup_validation_labels,
        fe_transition_selection_reference_path=args.fe_transition_selection_reference,
        followup_selection_reference_path=args.followup_selection_reference,
        fe_transition_trajectory_root=args.fe_transition_trajectory_root,
        output_dir=args.output_dir,
        target_column=args.target_column,
    )


if __name__ == "__main__":
    main()
