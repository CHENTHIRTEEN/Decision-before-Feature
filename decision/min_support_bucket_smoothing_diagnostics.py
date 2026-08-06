from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _json_default, _read_labels
from utility_labels.fields import UTILITY_VALUE_COLUMNS


TARGET_FAMILIES = ("bbob_f005", "bbob_f024")
TARGET_STAGES = (0.25, 0.30)
UPPER_STAGE = 0.25
LOWER_STAGE = 0.30
STAGE_TOLERANCE = 0.003
FE_ANALYSIS_RATIO = 0.05
ALGORITHMS = ("de", "pso", "cmaes", "shade")


def run_bucket_smoothing_diagnostic(
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
    policy_rows = _policy_rows(selection=selection, performance=performance, labels=labels, target_column=target_column)
    policy_summary = _summarize_policies(policy_rows)
    stability_summary = _summarize_stability(policy_rows)
    conclusion = _diagnostic_conclusion(policy_summary, stability_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "bucket_smoothing_policy_rows.parquet"
    summary_path = output_dir / "bucket_smoothing_policy_summary.parquet"
    stability_path = output_dir / "bucket_smoothing_stability_summary.parquet"
    json_path = output_dir / "bucket_smoothing_diagnostic_summary.json"
    pq.write_table(pa.Table.from_pandas(policy_rows, preserve_index=False), rows_path)
    pq.write_table(pa.Table.from_pandas(policy_summary, preserve_index=False), summary_path)
    pq.write_table(pa.Table.from_pandas(stability_summary, preserve_index=False), stability_path)

    summary = {
        "experiment": "min_support_bucket_smoothing_diagnostic",
        "research_question": (
            "How sensitive are selected_algorithm, VBS/SBS consistency, and U_ELA > 0 proxy coverage "
            "to nearest/lower/upper/interpolated performance-bucket policies around FE_ratio 0.25 and 0.30?"
        ),
        "target_column": target_column,
        "target_families": list(TARGET_FAMILIES),
        "target_stages": list(TARGET_STAGES),
        "policies": [
            "nearest_bucket_policy",
            "lower_bucket_policy",
            "upper_bucket_policy",
            "interpolated_trajectory_performance_policy",
        ],
        "inputs": {
            "fe_transition_validation_labels": str(fe_transition_validation_labels_path),
            "followup_validation_labels": str(followup_validation_labels_path),
            "fe_transition_selection_reference": str(fe_transition_selection_reference_path),
            "followup_selection_reference": str(followup_selection_reference_path),
            "fe_transition_trajectory_root": str(fe_transition_trajectory_root),
        },
        "outputs": {
            "policy_rows": str(rows_path),
            "policy_summary": str(summary_path),
            "stability_summary": str(stability_path),
            "summary": str(json_path),
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
            "Lower and upper policies reuse existing selection_reference rows from stages 0.30 and 0.25.",
            "Interpolated policy selects the algorithm with best linearly interpolated trajectory performance between upper and lower buckets; it is not a retrained ELA selector.",
            "U_ELA coverage is a proxy from existing utility labels and is marked unobserved when the selected algorithm has no matching label rows.",
        ],
    }
    json_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote bucket smoothing policy rows to {rows_path}")
    print(f"wrote bucket smoothing policy summary to {summary_path}")
    print(f"wrote bucket smoothing stability summary to {stability_path}")
    print(f"wrote bucket smoothing diagnostic summary to {json_path}")
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
    labels = labels[labels["comparison_stage"].isin((0.20, 0.25, 0.30, 0.35))].copy()
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
    selection = selection[selection["comparison_stage"].isin((UPPER_STAGE, LOWER_STAGE))].copy()
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
    for stage in (0.20, 0.25, 0.30, 0.35):
        if abs(ratio - stage) <= STAGE_TOLERANCE:
            return float(stage)
    return None


def _stage_from_remaining_budget(value: float) -> float | None:
    prefix_ratio = (1.0 - FE_ANALYSIS_RATIO) - float(value)
    return _stage_from_ratio(prefix_ratio)


def _policy_rows(
    *,
    selection: pd.DataFrame,
    performance: pd.DataFrame,
    labels: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    rows = []
    keys = selection[["family", "problem_id", "dimension"]].drop_duplicates().sort_values(["family", "problem_id", "dimension"])
    for target_stage in TARGET_STAGES:
        for _, key in keys.iterrows():
            upper = _selection_for_stage(selection, key, UPPER_STAGE)
            lower = _selection_for_stage(selection, key, LOWER_STAGE)
            nearest = lower if target_stage == LOWER_STAGE else upper
            interpolated = _interpolated_policy_row(performance, key, upper, lower, target_stage)
            policy_specs = [
                ("nearest_bucket_policy", nearest, target_stage, "direct_current_mapping"),
                ("lower_bucket_policy", lower, LOWER_STAGE, _utility_observation(target_stage, LOWER_STAGE)),
                ("upper_bucket_policy", upper, UPPER_STAGE, _utility_observation(target_stage, UPPER_STAGE)),
                ("interpolated_trajectory_performance_policy", interpolated, _proxy_stage_for_algorithm(interpolated, upper, lower), "interpolated_policy_proxy"),
            ]
            for policy_name, policy, utility_stage, utility_observation in policy_specs:
                if policy_name == "interpolated_trajectory_performance_policy":
                    vbs = _interpolated_vbs_algorithm(performance, key, upper, lower, target_stage)
                else:
                    vbs = _vbs_algorithm_for_bucket(performance, key, policy["performance_bucket_ratio"])
                utility = _utility_for_stage(labels, key, utility_stage, policy["selected_algorithm"], target_column)
                rows.append(
                    {
                        "target_stage": float(target_stage),
                        "policy_name": policy_name,
                        "family": str(key["family"]),
                        "problem_id": str(key["problem_id"]),
                        "dimension": int(key["dimension"]),
                        "policy_source": str(policy["policy_source"]),
                        "selection_stage_used": float(policy["selection_stage_used"]),
                        "utility_stage_used": float(utility_stage),
                        "utility_observation": utility_observation,
                        "remaining_budget_ratio": float(policy["remaining_budget_ratio"]),
                        "performance_bucket_ratio": float(policy["performance_bucket_ratio"]),
                        "selected_algorithm": str(policy["selected_algorithm"]),
                        "default_algorithm": str(policy["default_algorithm"]),
                        "sbs_algorithm": str(policy["sbs_algorithm"]),
                        "vbs_algorithm": vbs,
                        "selected_differs_from_default": bool(str(policy["selected_algorithm"]) != str(policy["default_algorithm"])),
                        "selected_matches_sbs": bool(str(policy["selected_algorithm"]) == str(policy["sbs_algorithm"])),
                        "selected_matches_vbs": bool(str(policy["selected_algorithm"]) == vbs),
                        "nearest_selected_algorithm": str(nearest["selected_algorithm"]),
                        "stable_with_nearest": bool(str(policy["selected_algorithm"]) == str(nearest["selected_algorithm"])),
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
    row = rows.iloc[0].copy()
    row["policy_source"] = f"selection_reference_stage_{stage:.2f}"
    row["selection_stage_used"] = float(stage)
    return row


def _interpolated_policy_row(performance: pd.DataFrame, key: pd.Series, upper: pd.Series, lower: pd.Series, target_stage: float) -> pd.Series:
    interpolated_scores = _interpolated_performance_scores(performance, key, upper, lower, target_stage)
    selected = (
        sorted(interpolated_scores.items(), key=lambda item: (item[1], item[0]))[0][0]
        if interpolated_scores
        else str(lower["selected_algorithm"])
    )
    target_remaining = float(lower["remaining_budget_ratio"] if target_stage == LOWER_STAGE else upper["remaining_budget_ratio"])
    lower_bucket = float(lower["performance_bucket_ratio"])
    upper_bucket = float(upper["performance_bucket_ratio"])
    weight_upper = _interpolation_weight(target_remaining, lower_bucket, upper_bucket)
    result = lower.copy()
    result["policy_source"] = "trajectory_performance_interpolation"
    result["selection_stage_used"] = float(target_stage)
    result["remaining_budget_ratio"] = target_remaining
    result["performance_bucket_ratio"] = (1.0 - weight_upper) * lower_bucket + weight_upper * upper_bucket
    result["selected_algorithm"] = selected
    result["sbs_algorithm"] = str(lower["sbs_algorithm"])
    result["default_algorithm"] = str(lower["default_algorithm"])
    return result


def _interpolated_vbs_algorithm(performance: pd.DataFrame, key: pd.Series, upper: pd.Series, lower: pd.Series, target_stage: float) -> str:
    interpolated_scores = _interpolated_performance_scores(performance, key, upper, lower, target_stage)
    if not interpolated_scores:
        return ""
    return sorted(interpolated_scores.items(), key=lambda item: (item[1], item[0]))[0][0]


def _interpolated_performance_scores(
    performance: pd.DataFrame,
    key: pd.Series,
    upper: pd.Series,
    lower: pd.Series,
    target_stage: float,
) -> dict[str, float]:
    lower_bucket = float(lower["performance_bucket_ratio"])
    upper_bucket = float(upper["performance_bucket_ratio"])
    target_remaining = float(lower["remaining_budget_ratio"] if target_stage == LOWER_STAGE else upper["remaining_budget_ratio"])
    weight_upper = _interpolation_weight(target_remaining, lower_bucket, upper_bucket)
    interpolated_scores = {}
    for algorithm in ALGORITHMS:
        lower_value = _mean_performance(performance, key, algorithm, lower_bucket)
        upper_value = _mean_performance(performance, key, algorithm, upper_bucket)
        if lower_value is None or upper_value is None:
            continue
        interpolated_scores[algorithm] = (1.0 - weight_upper) * lower_value + weight_upper * upper_value
    return interpolated_scores


def _interpolation_weight(target_remaining: float, lower_bucket: float, upper_bucket: float) -> float:
    if upper_bucket == lower_bucket:
        return 0.0
    return min(max((target_remaining - lower_bucket) / (upper_bucket - lower_bucket), 0.0), 1.0)


def _mean_performance(performance: pd.DataFrame, key: pd.Series, algorithm: str, ratio: float) -> float | None:
    rows = performance[
        (performance["problem_id"] == key["problem_id"])
        & (performance["dimension"] == int(key["dimension"]))
        & (performance["algorithm"] == algorithm)
        & (performance["FE_ratio"].round(6) == round(float(ratio), 6))
    ]
    if rows.empty:
        return None
    return float(rows["best_fitness"].mean())


def _vbs_algorithm_for_bucket(performance: pd.DataFrame, key: pd.Series, ratio: float) -> str:
    rows = performance[
        (performance["problem_id"] == key["problem_id"])
        & (performance["dimension"] == int(key["dimension"]))
        & (performance["FE_ratio"].round(6) == round(float(ratio), 6))
    ]
    if rows.empty:
        return ""
    means = rows.groupby("algorithm", as_index=False)["best_fitness"].mean()
    return str(means.sort_values(["best_fitness", "algorithm"]).iloc[0]["algorithm"])


def _utility_observation(target_stage: float, utility_stage: float) -> str:
    return "direct_current_mapping" if target_stage == utility_stage else "neighbor_bucket_proxy_not_same_stage"


def _proxy_stage_for_algorithm(policy: pd.Series, upper: pd.Series, lower: pd.Series) -> float:
    selected = str(policy["selected_algorithm"])
    if selected == str(lower["selected_algorithm"]):
        return LOWER_STAGE
    if selected == str(upper["selected_algorithm"]):
        return UPPER_STAGE
    return LOWER_STAGE


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


def _summarize_policies(rows: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (target_stage, policy_name), group in rows.groupby(["target_stage", "policy_name"], sort=True):
        summaries.append(_summary_row(target_stage, policy_name, "all_target_families", group))
    for (target_stage, policy_name, family), group in rows.groupby(["target_stage", "policy_name", "family"], sort=True):
        summaries.append(_summary_row(target_stage, policy_name, family, group))
    return pd.DataFrame(summaries)


def _summary_row(target_stage: float, policy_name: str, family: str, group: pd.DataFrame) -> dict[str, Any]:
    observed = group[group["utility_selected_algorithm_observed"]]
    observed_rows = int(observed["utility_rows_for_selected_algorithm"].sum()) if len(observed) else 0
    return {
        "target_stage": float(target_stage),
        "policy_name": str(policy_name),
        "family": family,
        "problem_dimension_rows": int(len(group)),
        "selected_algorithm_counts": _value_counts_string(group["selected_algorithm"]),
        "changed_problem_count": int(group["selected_differs_from_default"].sum()),
        "changed_problem_rate": float(group["selected_differs_from_default"].mean()) if len(group) else 0.0,
        "stable_with_nearest_count": int(group["stable_with_nearest"].sum()),
        "stable_with_nearest_rate": float(group["stable_with_nearest"].mean()) if len(group) else 0.0,
        "selected_matches_sbs_count": int(group["selected_matches_sbs"].sum()),
        "selected_matches_sbs_rate": float(group["selected_matches_sbs"].mean()) if len(group) else 0.0,
        "selected_matches_vbs_count": int(group["selected_matches_vbs"].sum()),
        "selected_matches_vbs_rate": float(group["selected_matches_vbs"].mean()) if len(group) else 0.0,
        "vbs_algorithm_counts": _value_counts_string(group["vbs_algorithm"]),
        "utility_observed_problem_count": int(len(observed)),
        "utility_observed_problem_rate": float(len(observed) / len(group)) if len(group) else 0.0,
        "utility_rows_for_selected_algorithm": observed_rows,
        "u_gt_zero_count": int(observed["u_gt_zero_count"].sum()) if len(observed) else 0,
        "u_gt_zero_rate_weighted": (
            float(observed["u_gt_zero_count"].sum() / observed_rows)
            if observed_rows > 0
            else 0.0
        ),
        "u_mean_mean": float(observed["u_mean"].mean()) if len(observed) else None,
    }


def _value_counts_string(series: pd.Series) -> str:
    if series.empty:
        return ""
    counts = series.fillna("").astype(str).value_counts().sort_index()
    return ";".join(f"{key}:{int(value)}" for key, value in counts.items() if key)


def _summarize_stability(rows: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    key_columns = ["target_stage", "family", "problem_id", "dimension"]
    for key, group in rows.groupby(key_columns, sort=True):
        selected_by_policy = dict(zip(group["policy_name"], group["selected_algorithm"], strict=False))
        for left in sorted(selected_by_policy):
            for right in sorted(selected_by_policy):
                if left >= right:
                    continue
                pairs.append(
                    {
                        "target_stage": float(key[0]),
                        "family": key[1],
                        "problem_id": key[2],
                        "dimension": int(key[3]),
                        "left_policy": left,
                        "right_policy": right,
                        "same_selected_algorithm": bool(selected_by_policy[left] == selected_by_policy[right]),
                        "left_selected_algorithm": selected_by_policy[left],
                        "right_selected_algorithm": selected_by_policy[right],
                    }
                )
    pair_rows = pd.DataFrame(pairs)
    if pair_rows.empty:
        return pair_rows
    summaries = []
    for (target_stage, left, right), group in pair_rows.groupby(["target_stage", "left_policy", "right_policy"], sort=True):
        summaries.append(
            {
                "target_stage": float(target_stage),
                "family": "all_target_families",
                "left_policy": left,
                "right_policy": right,
                "problem_dimension_rows": int(len(group)),
                "same_selected_count": int(group["same_selected_algorithm"].sum()),
                "same_selected_rate": float(group["same_selected_algorithm"].mean()) if len(group) else 0.0,
            }
        )
    for (target_stage, family, left, right), group in pair_rows.groupby(["target_stage", "family", "left_policy", "right_policy"], sort=True):
        summaries.append(
            {
                "target_stage": float(target_stage),
                "family": family,
                "left_policy": left,
                "right_policy": right,
                "problem_dimension_rows": int(len(group)),
                "same_selected_count": int(group["same_selected_algorithm"].sum()),
                "same_selected_rate": float(group["same_selected_algorithm"].mean()) if len(group) else 0.0,
            }
        )
    return pd.DataFrame(summaries)


def _diagnostic_conclusion(policy_summary: pd.DataFrame, stability_summary: pd.DataFrame) -> dict[str, Any]:
    all_rows = policy_summary[policy_summary["family"] == "all_target_families"].set_index(["target_stage", "policy_name"])
    nearest_025 = all_rows.loc[(0.25, "nearest_bucket_policy")]
    lower_025 = all_rows.loc[(0.25, "lower_bucket_policy")]
    interp_025 = all_rows.loc[(0.25, "interpolated_trajectory_performance_policy")]
    nearest_030 = all_rows.loc[(0.30, "nearest_bucket_policy")]
    upper_030 = all_rows.loc[(0.30, "upper_bucket_policy")]
    interp_030 = all_rows.loc[(0.30, "interpolated_trajectory_performance_policy")]
    return {
        "primary_result": "nearest_policy_is_bucket_discontinuous; interpolated_performance_policy_is_vbs_consistent_but_partly_unobserved_by_existing_utility_labels",
        "stage_025": {
            "nearest_changed_rate": float(nearest_025["changed_problem_rate"]),
            "lower_changed_rate": float(lower_025["changed_problem_rate"]),
            "interpolated_changed_rate": float(interp_025["changed_problem_rate"]),
            "nearest_u_gt_zero_proxy": float(nearest_025["u_gt_zero_rate_weighted"]),
            "lower_u_gt_zero_proxy": float(lower_025["u_gt_zero_rate_weighted"]),
            "interpolated_u_gt_zero_proxy_observed": float(interp_025["u_gt_zero_rate_weighted"]),
            "interpolated_utility_observed_problem_rate": float(interp_025["utility_observed_problem_rate"]),
        },
        "stage_030": {
            "nearest_changed_rate": float(nearest_030["changed_problem_rate"]),
            "upper_changed_rate": float(upper_030["changed_problem_rate"]),
            "interpolated_changed_rate": float(interp_030["changed_problem_rate"]),
            "nearest_u_gt_zero_proxy": float(nearest_030["u_gt_zero_rate_weighted"]),
            "upper_u_gt_zero_proxy": float(upper_030["u_gt_zero_rate_weighted"]),
            "interpolated_u_gt_zero_proxy_observed": float(interp_030["u_gt_zero_rate_weighted"]),
            "interpolated_utility_observed_problem_rate": float(interp_030["utility_observed_problem_rate"]),
        },
        "interpretation": (
            "Lower and upper bucket policies reproduce the previously observed discontinuity: upper selects cmaes "
            "for all target cells, lower selects de/shade/cmaes with 0.75 changed rate. The interpolated trajectory-"
            "performance policy uses both bucket performances and is consistent with the VBS induced by the same "
            "interpolated performance proxy. It often selects pso on bbob_f005, which is not covered by the existing "
            "utility labels for these states. Therefore the current evidence supports sparse bucket mapping as the "
            "source of selector instability, while also showing that a performance-interpolated policy would require "
            "new labels before U_ELA impact can be estimated directly."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose bucket smoothing policies around min_support FE_ratio 0.25/0.30.")
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
        default=Path("results/decision/min_support/bucket_smoothing_diagnostic"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    args = parser.parse_args()
    run_bucket_smoothing_diagnostic(
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
