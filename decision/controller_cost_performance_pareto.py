from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from decision.compare_controller_baselines import validate_time_only_training_summary
from decision.model_protocol import FROZEN_THRESHOLD_MODE, SELECTED_MODEL_ALIAS
from decision.query_contract import decision_query_root, validate_query_frame
from decision.sampling_opportunities import (
    STATE_KEY_COLUMNS,
    assert_aligned_decision_opportunities,
    assert_unique_state_keys,
    with_sampling_opportunity_type,
)
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS, get_query_spec
from trajectory.sampling import SAMPLING_METADATA_COLUMNS


DEFAULT_MODEL_NAME = SELECTED_MODEL_ALIAS
DEFAULT_THRESHOLD_MODE = FROZEN_THRESHOLD_MODE
TARGET_COLUMN = "u_query_lamT_1"
EPS = 1e-12
IDENTITY_COLUMNS = [
    *STATE_KEY_COLUMNS,
]
ALIGNMENT_COLUMNS = [
    "FE_ratio",
    "default_algorithm",
    "no_query_algorithm",
    "selected_algorithm",
    "selected_action",
    "selected_equals_default",
    "selected_equals_prefix",
    "handoff_required",
    "handoff_type",
    "selector_target_transform",
]
PREDICTION_IDENTITY_COLUMNS = [*IDENTITY_COLUMNS, *ALIGNMENT_COLUMNS]
GROUP_LAYERS = {
    "overall": [],
    "selected_equals_default": ["selected_equals_default"],
    "selected_equals_prefix": ["selected_equals_prefix"],
    "handoff_required": ["handoff_required"],
    "sampling_phase": ["sampling_phase"],
    "sampling_opportunity_type": ["sampling_opportunity_type"],
    "dimension": ["dimension"],
    "prefix_algorithm": ["prefix_algorithm"],
    "family": ["family"],
}
PARETO_AXIS_SETS = {
    "resource": {
        "title": "Resource cost vs final performance",
        "description": "直接围绕固定 query 调用资源解释 Decision-before-Feature。",
        "panels": [
            ("query_call_rate", "final_performance_mean", "Query call rate", "Mean final performance", "lower"),
            ("FE_query_mean", "final_performance_mean", "Mean FE used by the fixed query", "Mean final performance", "lower"),
            ("query_call_rate", "utility_mean", "Query call rate", "Mean utility", "higher"),
        ],
    },
    "utility": {
        "title": "Resource cost vs utility",
        "description": "直接检查所评估的固定 query 是否值得执行。",
        "panels": [
            ("query_call_rate", "utility_mean", "Query call rate", "Mean utility", "higher"),
            ("FE_query_mean", "utility_mean", "Mean FE used by the fixed query", "Mean utility", "higher"),
            ("query_call_rate", "utility_capture_rate", "Query call rate", "Utility capture rate", "higher"),
        ],
    },
    "diagnostic": {
        "title": "Call quality diagnostics",
        "description": "解释 controller 是否减少无效 query 调用。",
        "panels": [
            ("unhelpful_call_cost_sum", "utility_capture_rate", "Unhelpful call cost sum", "Utility capture rate", "higher"),
            ("query_call_rate", "precision_u_gt_zero_under_calls", "Query call rate", "Precision under calls", "higher"),
            ("unhelpful_call_rate_within_calls", "precision_u_gt_zero_under_calls", "Unhelpful call rate within calls", "Precision under calls", "higher"),
        ],
    },
    "runtime": {
        "title": "Wall-clock runtime diagnostics",
        "description": "检查当前实现运行时间，不作为唯一主结论。",
        "panels": [
            ("runtime_mean_seconds", "final_performance_mean", "Mean runtime seconds", "Mean final performance", "lower"),
            ("runtime_mean_seconds", "utility_mean", "Mean runtime seconds", "Mean utility", "higher"),
            ("runtime_mean_seconds", "query_call_rate", "Mean runtime seconds", "Query call rate", "lower"),
        ],
    },
}


def run_controller_cost_performance_pareto(
    *,
    query_id: str,
    utility_root: Path,
    predictions_path: Path,
    time_only_predictions_path: Path,
    model_fit_summary_path: Path,
    time_only_model_fit_summary_path: Path,
    output_dir: Path,
    model_name: str,
    threshold_mode: str,
    random_query_probability: float,
    random_repetitions: int,
    random_seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    _check_output_paths(output_dir, overwrite)
    labels = _read_validation_labels(utility_root, query_id)
    model_name = _resolve_prediction_model_name(predictions_path, model_name)
    predictions = _read_predictions(predictions_path, model_name, threshold_mode, query_id)
    validate_time_only_training_summary(time_only_predictions_path, query_id)
    time_only_predictions = _read_predictions(
        time_only_predictions_path,
        model_name,
        threshold_mode,
        query_id,
    )
    prediction_seconds_per_row = _prediction_seconds_per_row(model_fit_summary_path, model_name)
    time_only_prediction_seconds_per_row = _prediction_seconds_per_row(
        time_only_model_fit_summary_path,
        model_name,
    )
    frame = _join_labels_and_predictions(labels, predictions)
    frame = _join_time_only_predictions(frame, time_only_predictions, threshold_mode)
    policy_rows = _policy_rows(
        frame=frame,
        threshold_mode=threshold_mode,
        prediction_seconds_per_row=prediction_seconds_per_row,
        time_only_prediction_seconds_per_row=time_only_prediction_seconds_per_row,
        random_query_probability=random_query_probability,
        random_seed=random_seed,
    )
    policy_summary = _policy_summary(policy_rows)
    pareto_points = _pareto_points(policy_summary)
    pareto_frontier = _axis_set_frontiers(pareto_points)
    selector_ratio = _selector_selection_ratio(policy_rows)
    random_repetition_summary = _random_repetition_summary(
        frame=frame,
        random_query_probability=random_query_probability,
        random_repetitions=random_repetitions,
        random_seed=random_seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(policy_summary, output_dir / "cost_performance_policy_summary")
    _write_frame(pareto_points, output_dir / "cost_performance_pareto_points")
    _write_frame(pareto_frontier, output_dir / "pareto_axis_set_frontier")
    _write_frame(
        pareto_frontier[pareto_frontier["axis_set"] == "resource"].copy(),
        output_dir / "cost_performance_pareto_frontier",
    )
    _write_frame(selector_ratio, output_dir / "selector_selection_ratio")
    _write_frame(random_repetition_summary, output_dir / "random_query_repetition_summary")

    plot_paths = _draw_axis_set_plots(pareto_points=pareto_points, pareto_frontier=pareto_frontier, output_dir=output_dir)

    report_path = output_dir / "controller_cost_performance_pareto_report.md"
    summary_path = output_dir / "controller_cost_performance_pareto_summary.json"
    summary = {
        "experiment": "phase1_refined_sampling_controller_cost_performance_pareto",
        "query_id": query_id,
        "query_protocol": get_query_spec(query_id).protocol,
        "sample_design_id": get_query_spec(query_id).sample_design_id,
        "research_question": (
            "How do the current controller, time-only controller, SBS/No-query, Always Query, and Random Analysis "
            "compare on the validation cost-performance Pareto frontier, and how do their selector action ratios differ?"
        ),
        "utility_root": str(utility_root),
        "predictions_path": str(predictions_path),
        "time_only_predictions_path": str(time_only_predictions_path),
        "model_fit_summary_path": str(model_fit_summary_path),
        "time_only_model_fit_summary_path": str(time_only_model_fit_summary_path),
        "model_name": model_name,
        "time_only_model_name": model_name,
        "threshold_mode": threshold_mode,
        "prediction_seconds_per_row": prediction_seconds_per_row,
        "time_only_prediction_seconds_per_row": time_only_prediction_seconds_per_row,
        "random_query_probability": random_query_probability,
        "random_repetitions": random_repetitions,
        "random_seed": random_seed,
        "rows": int(len(frame)),
        "decision_opportunity_set": "all accepted dynamic budget-milestone and causal-event rows",
        "outputs": {
            "policy_summary": str(output_dir / "cost_performance_policy_summary.parquet"),
            "pareto_points": str(output_dir / "cost_performance_pareto_points.parquet"),
            "pareto_axis_set_frontier": str(output_dir / "pareto_axis_set_frontier.parquet"),
            "resource_frontier_compat": str(output_dir / "cost_performance_pareto_frontier.parquet"),
            "selector_selection_ratio": str(output_dir / "selector_selection_ratio.parquet"),
            "random_query_repetition_summary": str(output_dir / "random_query_repetition_summary.parquet"),
            "plots": plot_paths,
            "report": str(report_path),
            "summary": str(summary_path),
        },
        "data_leakage_check": {
            "models_retrained": False,
            "utility_labels_regenerated": False,
            "validation_labels_used_for_controller_fit_or_threshold": False,
            "decision_input_uses_query_features": False,
            "function_id_algorithm_id_or_optimizer_internal_parameters_used_as_input": False,
            "time_only_input_is_bf_fe_ratio_only": True,
            "all_policies_use_identical_decision_opportunities": True,
        },
        "metric_direction": {
            "final_performance_mean": "lower_is_better",
            "runtime_mean_seconds": "lower_is_better",
            "query_call_rate": "lower_is_better",
            "FE_query_mean": "lower_is_better",
            "unhelpful_call_cost_sum": "lower_is_better",
            "unhelpful_call_rate_within_calls": "lower_is_better",
            "utility_mean": "higher_is_better",
            "utility_capture_rate": "higher_is_better",
            "precision_u_gt_zero_under_calls": "higher_is_better",
        },
        "pareto_axis_sets": PARETO_AXIS_SETS,
        "scope_notes": [
            "SBS/No-query is represented by the train-derived SBS default skip path.",
            "Final performance is averaged in the current label scale and should be read with family-split context.",
            "Controller runtimes include the measured validation prediction overhead per row for the requested "
            "main and time-only models.",
        ],
    }
    report_path.write_text(
        _markdown_report(
            pareto_points=pareto_points,
            pareto_frontier=pareto_frontier,
            selector_ratio=selector_ratio,
            random_repetition_summary=random_repetition_summary,
            plot_paths=plot_paths,
            model_name=model_name,
            threshold_mode=threshold_mode,
            prediction_seconds_per_row=prediction_seconds_per_row,
            time_only_prediction_seconds_per_row=time_only_prediction_seconds_per_row,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote cost-performance policy summary to {output_dir / 'cost_performance_policy_summary.parquet'}")
    print(f"wrote Pareto frontier to {output_dir / 'pareto_axis_set_frontier.parquet'}")
    print(f"wrote selector selection ratios to {output_dir / 'selector_selection_ratio.parquet'}")
    print(f"wrote Pareto plots to {output_dir}")
    print(f"wrote report to {report_path}")
    return summary


def _check_output_paths(output_dir: Path, overwrite: bool) -> None:
    outputs = (
        output_dir / "cost_performance_policy_summary.csv",
        output_dir / "cost_performance_policy_summary.parquet",
        output_dir / "cost_performance_pareto_points.csv",
        output_dir / "cost_performance_pareto_points.parquet",
        output_dir / "cost_performance_pareto_frontier.csv",
        output_dir / "cost_performance_pareto_frontier.parquet",
        output_dir / "pareto_axis_set_frontier.csv",
        output_dir / "pareto_axis_set_frontier.parquet",
        output_dir / "selector_selection_ratio.csv",
        output_dir / "selector_selection_ratio.parquet",
        output_dir / "random_query_repetition_summary.csv",
        output_dir / "random_query_repetition_summary.parquet",
        output_dir / "cost_performance_pareto.png",
        output_dir / "cost_performance_pareto.svg",
        output_dir / "cost_performance_pareto.pdf",
        output_dir / "pareto_resource.png",
        output_dir / "pareto_resource.svg",
        output_dir / "pareto_resource.pdf",
        output_dir / "pareto_utility.png",
        output_dir / "pareto_utility.svg",
        output_dir / "pareto_utility.pdf",
        output_dir / "pareto_diagnostic.png",
        output_dir / "pareto_diagnostic.svg",
        output_dir / "pareto_diagnostic.pdf",
        output_dir / "pareto_runtime.png",
        output_dir / "pareto_runtime.svg",
        output_dir / "pareto_runtime.pdf",
        output_dir / "controller_cost_performance_pareto_report.md",
        output_dir / "controller_cost_performance_pareto_summary.json",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"cost-performance outputs already exist; pass --overwrite: {existing[0]}")


def _read_validation_labels(utility_root: Path, query_id: str) -> pd.DataFrame:
    paths = sorted((utility_root / "bbob_validation").glob("*/dimension_*/utility_labels.parquet"))
    if not paths:
        raise FileNotFoundError(f"no validation utility label shards under {utility_root}")
    frames = [pq.read_table(path).to_pandas() for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    validate_query_frame(frame, query_id=query_id, artifact="validation utility labels")
    required = {
        *IDENTITY_COLUMNS,
        *ALIGNMENT_COLUMNS,
        *SAMPLING_METADATA_COLUMNS,
        "p_skip",
        "p_query",
        "performance_gain_norm",
        "runtime_query",
        "runtime_selection",
        "runtime_no_query_optimization",
        "runtime_query_optimization",
        "FE_query",
        TARGET_COLUMN,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"validation utility labels missing columns: {missing}")
    frame = with_sampling_opportunity_type(frame, artifact="validation utility labels")
    assert_unique_state_keys(frame, artifact="validation utility labels")
    return frame


def _read_predictions(path: Path, model_name: str, threshold_mode: str, query_id: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pq.read_table(path).to_pandas()
    validate_query_frame(frame, query_id=query_id, artifact="validation predictions")
    required = {
        *PREDICTION_IDENTITY_COLUMNS,
        *SAMPLING_METADATA_COLUMNS,
        "model_name",
        f"decision_run_query_{threshold_mode}",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"validation predictions missing columns: {missing}")
    frame = frame[frame["model_name"] == model_name].copy()
    if frame.empty:
        raise ValueError(f"no validation predictions for model_name={model_name}")
    frame = with_sampling_opportunity_type(frame, artifact="validation predictions")
    assert_unique_state_keys(frame, artifact="validation predictions")
    return frame[
        [
            *PREDICTION_IDENTITY_COLUMNS,
            *SAMPLING_METADATA_COLUMNS,
            "sampling_opportunity_type",
            f"decision_run_query_{threshold_mode}",
        ]
    ].reset_index(drop=True)


def _resolve_prediction_model_name(path: Path, requested_model_name: str) -> str:
    if requested_model_name != SELECTED_MODEL_ALIAS:
        return requested_model_name
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pq.read_table(path, columns=["model_name", "selected_by_nested_oof"]).to_pandas()
    selected = frame[frame["selected_by_nested_oof"].astype(bool)]
    names = selected["model_name"].astype(str).unique().tolist()
    if len(names) != 1:
        raise ValueError("prediction table must identify exactly one nested-OOF selected model")
    return names[0]


def _prediction_seconds_per_row(path: Path, model_name: str) -> float:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pq.read_table(path).to_pandas()
    rows = frame[frame["model_name"] == model_name]
    if rows.empty:
        raise ValueError(f"no model fit summary row for model_name={model_name}")
    return float(rows["validation_prediction_seconds_per_row"].iloc[0])


def _join_labels_and_predictions(labels: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    assert_aligned_decision_opportunities(
        labels,
        predictions,
        reference_artifact="validation utility labels",
        candidate_artifact="current-controller predictions",
    )
    _check_action_alignment(labels, predictions, "current-controller predictions")
    decision_columns = [column for column in predictions.columns if column.startswith("decision_run_query_")]
    result = labels.merge(
        predictions[[*IDENTITY_COLUMNS, *decision_columns]],
        on=IDENTITY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    missing = result.filter(like="decision_run_query_").isna().any(axis=1)
    if bool(missing.any()):
        raise ValueError(f"missing controller predictions for {int(missing.sum())} utility-label rows")
    if len(result) != len(labels):
        raise ValueError("label/prediction join changed row count")
    return result


def _join_time_only_predictions(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    threshold_mode: str,
) -> pd.DataFrame:
    source_column = f"decision_run_query_{threshold_mode}"
    time_only_column = "decision_run_query_time_only"
    assert_aligned_decision_opportunities(
        frame,
        predictions,
        reference_artifact="validation utility labels",
        candidate_artifact="time-only predictions",
    )
    _check_action_alignment(frame, predictions, "time-only predictions")
    time_only = predictions.rename(columns={source_column: time_only_column})
    result = frame.merge(
        time_only[[*IDENTITY_COLUMNS, time_only_column]],
        on=IDENTITY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    missing = result[time_only_column].isna()
    if bool(missing.any()):
        raise ValueError(f"missing time-only predictions for {int(missing.sum())} utility-label rows")
    if len(result) != len(frame):
        raise ValueError("time-only prediction join changed row count")
    return result


def _check_action_alignment(reference: pd.DataFrame, candidate: pd.DataFrame, artifact: str) -> None:
    left = reference.sort_values(IDENTITY_COLUMNS).reset_index(drop=True)
    right = candidate.sort_values(IDENTITY_COLUMNS).reset_index(drop=True)
    for column in ALIGNMENT_COLUMNS:
        if column == "FE_ratio":
            equal = np.allclose(
                left[column].to_numpy(dtype=float),
                right[column].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
        else:
            equal = np.array_equal(left[column].to_numpy(), right[column].to_numpy())
        if not equal:
            raise ValueError(f"validation utility labels and {artifact} disagree on {column}")


def _policy_rows(
    *,
    frame: pd.DataFrame,
    threshold_mode: str,
    prediction_seconds_per_row: float,
    time_only_prediction_seconds_per_row: float,
    random_query_probability: float,
    random_seed: int,
) -> pd.DataFrame:
    if random_query_probability < 0.0 or random_query_probability > 1.0:
        raise ValueError("random_query_probability must be in [0, 1]")
    rng = np.random.default_rng(np.random.SeedSequence([int(random_seed), 20260811, len(frame)]))
    random_call = rng.random(len(frame)) < random_query_probability
    controller_call = frame[f"decision_run_query_{threshold_mode}"].to_numpy(dtype=bool)
    time_only_call = frame["decision_run_query_time_only"].to_numpy(dtype=bool)
    policy_specs = (
        ("sbs_skip_reference", "baseline", np.zeros(len(frame), dtype=bool), 0.0),
        ("always_query", "baseline", np.ones(len(frame), dtype=bool), 0.0),
        ("traditional_aas", "baseline", np.ones(len(frame), dtype=bool), 0.0),
        (f"random_query_p{int(round(random_query_probability * 100)):02d}", "baseline", random_call, 0.0),
        ("time_only_controller", "learned_baseline", time_only_call, time_only_prediction_seconds_per_row),
        ("current_controller", "controller", controller_call, prediction_seconds_per_row),
    )
    frames = []
    for policy_name, policy_category, call, overhead in policy_specs:
        policy = frame.copy()
        policy.insert(0, "policy_name", policy_name)
        policy.insert(1, "policy_category", policy_category)
        policy["run_query"] = call
        policy["policy_selected_algorithm"] = np.where(call, policy["selected_algorithm"], "skip_query")
        policy["policy_final_performance"] = np.where(call, policy["p_query"], policy["p_skip"])
        policy["policy_runtime_seconds"] = overhead + np.where(
            call,
            policy["runtime_query"] + policy["runtime_selection"] + policy["runtime_query_optimization"],
            policy["runtime_no_query_optimization"],
        )
        policy["policy_FE_query"] = np.where(call, policy["FE_query"], 0)
        denominator = np.maximum(np.maximum(np.abs(policy["p_skip"]), np.abs(policy["policy_final_performance"])), EPS)
        policy["policy_performance_gain_norm_vs_skip"] = (
            policy["p_skip"] - policy["policy_final_performance"]
        ) / denominator
        policy["policy_utility"] = np.where(call, policy[TARGET_COLUMN], 0.0)
        frames.append(policy)
    return pd.concat(frames, ignore_index=True)


def _policy_summary(policies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer, group_columns in GROUP_LAYERS.items():
        if not group_columns:
            for (_, _), frame in policies.groupby(["policy_name", "policy_category"], sort=True):
                rows.append(_policy_row(frame, layer=layer, group={}))
            continue
        for (_, _), policy_frame in policies.groupby(["policy_name", "policy_category"], sort=True):
            for group_values, frame in policy_frame.groupby(group_columns, dropna=False, sort=True):
                if not isinstance(group_values, tuple):
                    group_values = (group_values,)
                rows.append(_policy_row(frame, layer=layer, group=dict(zip(group_columns, group_values, strict=True))))
    return pd.DataFrame(rows).sort_values(["layer", "group", "policy_category", "policy_name"]).reset_index(drop=True)


def _policy_row(frame: pd.DataFrame, *, layer: str, group: dict[str, Any]) -> dict[str, Any]:
    observed = frame[TARGET_COLUMN].to_numpy(dtype=float)
    calls = frame["run_query"].to_numpy(dtype=bool)
    positive = observed > 0.0
    captured_positive = positive & calls
    unhelpful_calls = (~positive) & calls
    positive_utility_sum = float(np.sum(observed[positive]))
    captured_positive_sum = float(np.sum(observed[captured_positive]))
    call_rows = int(np.sum(calls))
    final_performance = frame["policy_final_performance"].to_numpy(dtype=float)
    runtime = frame["policy_runtime_seconds"].to_numpy(dtype=float)
    utility = frame["policy_utility"].to_numpy(dtype=float)
    return {
        "policy_name": str(frame["policy_name"].iloc[0]),
        "policy_category": str(frame["policy_category"].iloc[0]),
        "layer": layer,
        "group": _group_label(group),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "sampling_phase": group.get("sampling_phase"),
        "sampling_opportunity_type": group.get("sampling_opportunity_type"),
        "prefix_algorithm": group.get("prefix_algorithm"),
        "selected_equals_default": group.get("selected_equals_default"),
        "selected_equals_prefix": group.get("selected_equals_prefix"),
        "handoff_required": group.get("handoff_required"),
        "rows": int(len(frame)),
        "query_call_rows": call_rows,
        "query_call_rate": float(np.mean(calls)),
        "FE_query_mean": float(np.mean(frame["policy_FE_query"])),
        "runtime_mean_seconds": float(np.mean(runtime)),
        "runtime_median_seconds": float(np.median(runtime)),
        "final_performance_mean": float(np.mean(final_performance)),
        "final_performance_median": float(np.median(final_performance)),
        "final_performance_delta_vs_sbs_mean": float(np.mean(final_performance - frame["p_skip"].to_numpy(dtype=float))),
        "performance_gain_norm_vs_skip_mean": float(np.mean(frame["policy_performance_gain_norm_vs_skip"])),
        "utility_sum": float(np.sum(utility)),
        "utility_mean": float(np.mean(utility)),
        "utility_median": float(np.median(utility)),
        "observed_utility_gt_zero_rows": int(np.sum(positive)),
        "observed_utility_gt_zero_rate": float(np.mean(positive)),
        "positive_row_capture_rate": float(np.sum(captured_positive) / max(np.sum(positive), 1)),
        "utility_capture_rate": (
            captured_positive_sum / positive_utility_sum if positive_utility_sum > 0.0 else 0.0
        ),
        "precision_u_gt_zero_under_calls": float(np.sum(captured_positive) / max(call_rows, 1)),
        "unhelpful_call_rows": int(np.sum(unhelpful_calls)),
        "unhelpful_call_rate_within_calls": float(np.sum(unhelpful_calls) / max(call_rows, 1)),
        "unhelpful_call_cost_sum": float(-np.sum(observed[unhelpful_calls])),
    }


def _pareto_points(policy_summary: pd.DataFrame) -> pd.DataFrame:
    points = policy_summary[policy_summary["layer"] == "overall"].copy()
    points["plot_label"] = points["policy_name"].map(
        {
            "sbs_skip_reference": "SBS / No-query",
            "always_query": "Always Query",
            "traditional_aas": "Traditional AAS",
            "random_query_p50": "Random Query p=0.5",
            "time_only_controller": "Time-only controller",
            "current_controller": "Current controller",
        }
    ).fillna(points["policy_name"])
    return points.reset_index(drop=True)


def _axis_set_frontiers(points: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for axis_set, spec in PARETO_AXIS_SETS.items():
        for cost_column, quality_column, _x_label, _y_label, quality_direction in spec["panels"]:
            frames.append(
                _non_dominated(
                    points,
                    cost_column=cost_column,
                    quality_column=quality_column,
                    quality_direction=quality_direction,
                ).assign(
                    axis_set=axis_set,
                    frontier_cost_axis=cost_column,
                    quality_axis=quality_column,
                    quality_direction=quality_direction,
                )
            )
    return pd.concat(frames, ignore_index=True)


def _non_dominated(
    frame: pd.DataFrame,
    *,
    cost_column: str,
    quality_column: str,
    quality_direction: str,
) -> pd.DataFrame:
    rows = []
    for index, row in frame.iterrows():
        cost = float(row[cost_column])
        quality = float(row[quality_column])
        others = frame.drop(index)
        other_cost = others[cost_column].to_numpy(dtype=float)
        other_quality = others[quality_column].to_numpy(dtype=float)
        if quality_direction == "lower":
            quality_no_worse = other_quality <= quality
            quality_better = other_quality < quality
        elif quality_direction == "higher":
            quality_no_worse = other_quality >= quality
            quality_better = other_quality > quality
        else:
            raise ValueError(f"unknown quality direction: {quality_direction}")
        dominated = (
            (other_cost <= cost)
            & quality_no_worse
            & ((other_cost < cost) | quality_better)
        )
        if not bool(np.any(dominated)):
            rows.append(row)
    ascending_quality = quality_direction == "lower"
    return pd.DataFrame(rows).sort_values([cost_column, quality_column], ascending=[True, ascending_quality]).reset_index(drop=True)


def _selector_selection_ratio(policies: pd.DataFrame) -> pd.DataFrame:
    focus = policies[
        policies["policy_name"].isin(
            ["current_controller", "time_only_controller", "always_query", "random_query_p50"]
        )
    ].copy()
    rows = []
    for policy_name, policy_frame in focus.groupby("policy_name", sort=True):
        all_counts = policy_frame["policy_selected_algorithm"].value_counts(dropna=False).sort_index()
        for selected_algorithm, count in all_counts.items():
            rows.append(
                {
                    "policy_name": policy_name,
                    "denominator_scope": "all_rows",
                    "selected_algorithm": str(selected_algorithm),
                    "rows": int(len(policy_frame)),
                    "count": int(count),
                    "ratio": float(count / max(len(policy_frame), 1)),
                }
            )
        called = policy_frame[policy_frame["run_query"]].copy()
        called_counts = called["selected_algorithm"].value_counts(dropna=False).sort_index()
        for selected_algorithm, count in called_counts.items():
            rows.append(
                {
                    "policy_name": policy_name,
                    "denominator_scope": "called_rows",
                    "selected_algorithm": str(selected_algorithm),
                    "rows": int(len(called)),
                    "count": int(count),
                    "ratio": float(count / max(len(called), 1)),
                }
            )
    return pd.DataFrame(rows).sort_values(["denominator_scope", "policy_name", "selected_algorithm"]).reset_index(drop=True)


def _random_repetition_summary(
    *,
    frame: pd.DataFrame,
    random_query_probability: float,
    random_repetitions: int,
    random_seed: int,
) -> pd.DataFrame:
    if random_repetitions <= 0:
        raise ValueError("random_repetitions must be positive")
    rows = []
    for repetition in range(random_repetitions):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(random_seed), 20260812, len(frame), int(repetition)])
        )
        call = rng.random(len(frame)) < random_query_probability
        final_performance = np.where(call, frame["p_query"], frame["p_skip"])
        runtime = np.where(
            call,
            frame["runtime_query"] + frame["runtime_selection"] + frame["runtime_query_optimization"],
            frame["runtime_no_query_optimization"],
        )
        utility = np.where(call, frame[TARGET_COLUMN], 0.0)
        rows.append(
            {
                "repetition": repetition,
                "query_call_rate": float(np.mean(call)),
                "final_performance_mean": float(np.mean(final_performance)),
                "runtime_mean_seconds": float(np.mean(runtime)),
                "utility_mean": float(np.mean(utility)),
                "utility_sum": float(np.sum(utility)),
            }
        )
    raw = pd.DataFrame(rows)
    summary_rows = []
    for metric in ["query_call_rate", "final_performance_mean", "runtime_mean_seconds", "utility_mean", "utility_sum"]:
        values = raw[metric].to_numpy(dtype=float)
        summary_rows.append(
            {
                "policy_name": f"random_query_p{int(round(random_query_probability * 100)):02d}",
                "metric": metric,
                "repetitions": random_repetitions,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if random_repetitions > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(summary_rows)


def _draw_axis_set_plots(*, pareto_points: pd.DataFrame, pareto_frontier: pd.DataFrame, output_dir: Path) -> dict[str, dict[str, str]]:
    plot_paths: dict[str, dict[str, str]] = {}
    for axis_set, spec in PARETO_AXIS_SETS.items():
        png_path = output_dir / f"pareto_{axis_set}.png"
        svg_path = output_dir / f"pareto_{axis_set}.svg"
        pdf_path = output_dir / f"pareto_{axis_set}.pdf"
        _draw_axis_set_plot(
            points=pareto_points,
            frontier=pareto_frontier[pareto_frontier["axis_set"] == axis_set].copy(),
            axis_set=axis_set,
            spec=spec,
            png_path=png_path,
            svg_path=svg_path,
            pdf_path=pdf_path,
        )
        plot_paths[axis_set] = {"png": str(png_path), "svg": str(svg_path), "pdf": str(pdf_path)}
    # Keep the previous filename as a compatibility alias for the resource plot.
    for suffix in ("png", "svg", "pdf"):
        source = output_dir / f"pareto_resource.{suffix}"
        target = output_dir / f"cost_performance_pareto.{suffix}"
        target.write_bytes(source.read_bytes())
    return plot_paths


def _draw_axis_set_plot(
    *,
    points: pd.DataFrame,
    frontier: pd.DataFrame,
    axis_set: str,
    spec: dict[str, Any],
    png_path: Path,
    svg_path: Path,
    pdf_path: Path,
) -> None:
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    panels = spec["panels"]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.8), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    color = {
        "baseline": "#4C78A8",
        "learned_baseline": "#54A24B",
        "controller": "#F58518",
    }
    marker = {
        "baseline": "o",
        "learned_baseline": "^",
        "controller": "s",
    }
    for ax, (cost_column, quality_column, x_label, y_label, quality_direction) in zip(axes, panels, strict=True):
        for _, row in points.iterrows():
            ax.scatter(
                float(row[cost_column]),
                float(row[quality_column]),
                s=78,
                color=color.get(str(row["policy_category"]), "#666666"),
                marker=marker.get(str(row["policy_category"]), "o"),
                edgecolor="black",
                linewidth=0.7,
                label=str(row["plot_label"]),
                zorder=3,
            )
            ax.annotate(
                str(row["plot_label"]),
                (float(row[cost_column]), float(row[quality_column])),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
            )
        axis_frontier = frontier[frontier["frontier_cost_axis"] == cost_column].copy()
        if not axis_frontier.empty:
            ordered = axis_frontier.sort_values(cost_column)
            ax.plot(
                ordered[cost_column].to_numpy(dtype=float),
                ordered[quality_column].to_numpy(dtype=float),
                color="#222222",
                linewidth=1.2,
                linestyle="--",
                label="Pareto frontier",
                zorder=2,
            )
        ax.set_xlabel(f"{x_label} (lower is better)")
        direction_label = "lower is better" if quality_direction == "lower" else "higher is better"
        ax.set_title(x_label)
        ax.set_ylabel(f"{y_label} ({direction_label})")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    fig.legend(by_label.values(), by_label.keys(), frameon=False, loc="outside lower center", ncol=5)
    fig.suptitle(f"{spec['title']} ({axis_set})", y=1.04)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(
    *,
    pareto_points: pd.DataFrame,
    pareto_frontier: pd.DataFrame,
    selector_ratio: pd.DataFrame,
    random_repetition_summary: pd.DataFrame,
    plot_paths: dict[str, dict[str, str]],
    model_name: str,
    threshold_mode: str,
    prediction_seconds_per_row: float,
    time_only_prediction_seconds_per_row: float,
) -> str:
    point_columns = [
        "policy_name",
        "policy_category",
        "rows",
        "final_performance_mean",
        "runtime_mean_seconds",
        "utility_mean",
        "query_call_rate",
        "FE_query_mean",
        "utility_capture_rate",
        "precision_u_gt_zero_under_calls",
        "unhelpful_call_cost_sum",
    ]
    frontier_columns = [
        "axis_set",
        "policy_name",
        "frontier_cost_axis",
        "quality_axis",
        "quality_direction",
        "query_call_rate",
        "FE_query_mean",
        "runtime_mean_seconds",
        "final_performance_mean",
        "utility_mean",
        "utility_capture_rate",
        "precision_u_gt_zero_under_calls",
        "unhelpful_call_cost_sum",
    ]
    selector_columns = ["policy_name", "denominator_scope", "selected_algorithm", "rows", "count", "ratio"]
    frontier_sections = []
    for axis_set, spec in PARETO_AXIS_SETS.items():
        axis_frontier = pareto_frontier[pareto_frontier["axis_set"] == axis_set].copy()
        frontier_sections.extend(
            [
                f"## {axis_set}: {spec['title']}",
                "",
                spec["description"],
                "",
                f"图文件：`{plot_paths[axis_set]['png']}`",
                "",
                _markdown_table(axis_frontier[frontier_columns]),
                "",
            ]
        )
    report = [
        "# Controller Cost-Performance Pareto",
        "",
        "## 摘要",
        "",
        f"- 当前 controller：`{model_name}`，阈值口径 `{threshold_mode}`。",
        f"- Time-only controller 使用同一模型 `{model_name}`，输入固定为 `X={{FE_ratio}}`（实现列 `bf_fe_ratio`）。",
        f"- controller 每行预测开销：`{prediction_seconds_per_row:.9f}` 秒，已计入 runtime。",
        f"- Time-only controller 每行预测开销：`{time_only_prediction_seconds_per_row:.9f}` 秒，已计入 runtime。",
        "- final performance 与 runtime 都按越小越好解释；utility 按越大越好解释。",
        "- SBS/No-query 使用训练集确定的 SBS default skip path。",
        "",
        "## Pareto Points",
        "",
        _markdown_table(pareto_points[point_columns]),
        "",
        *frontier_sections,
        "## Selector Selection Ratio",
        "",
        _markdown_table(selector_ratio[selector_columns]),
        "",
        "## Random Analysis Repetition Summary",
        "",
        _markdown_table(random_repetition_summary),
        "",
        "## 解释",
        "",
        "Pareto frontier 按 axis-set 分别计算。每个 panel 使用一个成本轴和一个质量轴：成本轴总是越低越好；质量轴根据 `quality_direction` 解释。"
        "如果一个策略在指定成本轴不更高、质量轴不更差，并且至少一个指标更优，则被比较策略不在该 panel 的 frontier 上。"
        "Time-only controller 用于判断主 Controller 的成本—性能关系是否提供阶段信息之外的增量。"
        "selector 比例中，`called_rows` 只统计实际执行固定 query 的行，`all_rows` 将未执行 query 记为 `skip_query`。",
    ]
    return "\n".join(report) + "\n"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _group_label(group: dict[str, Any]) -> str:
    if not group:
        return "all"
    return ", ".join(f"{key}={value}" for key, value in group.items())


def _write_frame(frame: pd.DataFrame, stem: Path) -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), stem.with_suffix(".parquet"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build phase1 controller cost-performance Pareto comparison.")
    parser.add_argument("--query-id", choices=sorted(LANDSCAPE_QUERY_SPECS), required=True)
    parser.add_argument("--utility-root", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--time-only-predictions", type=Path, default=None)
    parser.add_argument("--model-fit-summary", type=Path, default=None)
    parser.add_argument("--time-only-model-fit-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--threshold-mode", default=DEFAULT_THRESHOLD_MODE)
    parser.add_argument("--random-query-probability", type=float, default=0.5)
    parser.add_argument("--random-repetitions", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=1701)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    query_root = decision_query_root(args.query_id)
    feature_group_root = query_root / "feature_group_ablation/B3"
    time_only_root = query_root / "feature_group_ablation/T0"
    run_controller_cost_performance_pareto(
        query_id=args.query_id,
        utility_root=args.utility_root or Path("results/utility_labels") / args.query_id,
        predictions_path=args.predictions or feature_group_root / "validation_predictions.parquet",
        time_only_predictions_path=args.time_only_predictions or time_only_root / "validation_predictions.parquet",
        model_fit_summary_path=args.model_fit_summary or feature_group_root / "model_fit_summary.parquet",
        time_only_model_fit_summary_path=args.time_only_model_fit_summary
        or time_only_root / "model_fit_summary.parquet",
        output_dir=args.output_dir or query_root / "controller_cost_performance_pareto",
        model_name=args.model_name,
        threshold_mode=args.threshold_mode,
        random_query_probability=args.random_query_probability,
        random_repetitions=args.random_repetitions,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
