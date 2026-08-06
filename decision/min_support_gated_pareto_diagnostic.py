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

from decision.min_support_diagnostics import GROUP_LAYERS, _group_label
from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _json_default
from decision.min_support_model_sensitivity import EVALUATION_DOMAINS


MODEL_FAMILIES = ("lightgbm", "random_forest", "xgboost")
BASELINE_POLICIES = ("no_ela_sbs", "always_ela_traditional_aas", "random_analysis_p50", "best_observed_analysis_action")
EVAL_DOMAINS = ("all_validation", "changed_algorithm_validation", "same_algorithm_reference")


def run_gated_pareto_diagnostic(
    *,
    validation_predictions_path: Path,
    train_predictions_path: Path,
    ablation_policy_summary_path: Path,
    output_dir: Path,
    target_column: str,
    dataset_name: str,
) -> dict[str, Any]:
    validation = _read_validation_predictions(validation_predictions_path, target_column)
    train = _read_train_predictions(train_predictions_path, target_column)
    baselines = _read_baselines(ablation_policy_summary_path)
    validation = validation[validation["model_family"].isin(MODEL_FAMILIES)].copy()
    train = train[train["model_family"].isin(MODEL_FAMILIES)].copy()

    threshold_table = _threshold_table(train, validation, target_column)
    policy_rows = _policy_summary(validation, baselines, threshold_table, target_column)
    pareto_points = _pareto_points(policy_rows)
    frontier = _frontier_rows(pareto_points)
    conclusion = _diagnostic_conclusion(pareto_points, frontier)

    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = output_dir / "gated_pareto_thresholds.parquet"
    policy_path = output_dir / "gated_pareto_policy_summary.parquet"
    points_path = output_dir / "gated_pareto_points.parquet"
    frontier_path = output_dir / "gated_pareto_frontier.parquet"
    png_path = output_dir / "gated_cost_performance_pareto.png"
    pdf_path = output_dir / "gated_cost_performance_pareto.pdf"
    svg_path = output_dir / "gated_cost_performance_pareto.svg"
    json_path = output_dir / "gated_pareto_summary.json"

    pq.write_table(pa.Table.from_pandas(threshold_table, preserve_index=False), threshold_path)
    pq.write_table(pa.Table.from_pandas(policy_rows, preserve_index=False), policy_path)
    pq.write_table(pa.Table.from_pandas(pareto_points, preserve_index=False), points_path)
    pq.write_table(pa.Table.from_pandas(frontier, preserve_index=False), frontier_path)
    _draw_pareto(pareto_points, frontier, png_path, pdf_path, svg_path, dataset_name)

    summary = {
        "experiment": "min_support_gated_pareto_diagnostic",
        "research_question": (
            "Can score-only gating variants move LightGBM/RF/XGBoost Decision-before-Feature policies onto a better "
            "cost-performance Pareto frontier without retraining or changing the utility-label protocol?"
        ),
        "dataset_name": dataset_name,
        "target_column": target_column,
        "inputs": {
            "validation_predictions": str(validation_predictions_path),
            "train_predictions": str(train_predictions_path),
            "ablation_policy_summary": str(ablation_policy_summary_path),
        },
        "models": list(MODEL_FAMILIES),
        "gate_definitions": _gate_definitions(),
        "score_row_selection": (
            "Validation predictions may contain multiple threshold_mode rows for the same model/state. "
            "This diagnostic keeps one score row per model/state by selecting threshold_mode == 'zero' when available; "
            "all gated policies are recomputed from decision_score and do not reuse the stored threshold_mode decision."
        ),
        "outputs": {
            "thresholds": str(threshold_path),
            "policy_summary": str(policy_path),
            "pareto_points": str(points_path),
            "pareto_frontier": str(frontier_path),
            "png": str(png_path),
            "pdf": str(pdf_path),
            "svg": str(svg_path),
            "summary": str(json_path),
        },
        "diagnostic_conclusion": conclusion,
        "data_leakage_check": {
            "models_retrained": False,
            "utility_labels_regenerated": False,
            "original_utility_labels_modified": False,
            "decision_input_uses_ela_features": False,
            "formal_phase1_configs_modified": False,
            "validation_utility_used_only_by_descriptive_family_stage_threshold": True,
        },
    }
    json_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote gated Pareto thresholds to {threshold_path}")
    print(f"wrote gated Pareto policy summary to {policy_path}")
    print(f"wrote gated Pareto points to {points_path}")
    print(f"wrote gated Pareto frontier to {frontier_path}")
    print(f"wrote gated Pareto plot to {png_path}")
    print(f"wrote gated Pareto summary to {json_path}")
    return summary


def _gate_definitions() -> list[dict[str, Any]]:
    return [
        {
            "gate_name": "zero_threshold",
            "deployable": True,
            "definition": "Run ELA when model score > 0.",
        },
        {
            "gate_name": "changed_train_utility_threshold",
            "deployable": True,
            "definition": "Use one utility-maximizing score threshold estimated from train rows where selected_algorithm differs from default_algorithm.",
        },
        {
            "gate_name": "stage_train_utility_threshold",
            "deployable": True,
            "definition": "Use a utility-maximizing score threshold estimated from train rows at the same FE_ratio; no validation labels are used.",
        },
        {
            "gate_name": "family_stage_train_utility_threshold",
            "deployable": True,
            "definition": "Use a utility-maximizing score threshold estimated from train rows with the same family and FE_ratio; no validation labels are used.",
        },
        {
            "gate_name": "same_algorithm_score_guard_q95",
            "deployable": True,
            "definition": "Run ELA only when score is above the 95th percentile of same_algorithm train scores.",
        },
        {
            "gate_name": "changed_algorithm_label_guard",
            "deployable": False,
            "definition": "Diagnostic bound: keep zero-threshold calls only on validation rows where selected_algorithm differs from default_algorithm.",
        },
        {
            "gate_name": "family_stage_validation_descriptive_threshold",
            "deployable": False,
            "definition": "Descriptive opportunity check: choose thresholds within validation family and FE_ratio groups. This uses validation utility and is excluded from deployable frontiers.",
        },
    ]


def _read_validation_predictions(path: Path, target_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing predictions file: {path}")
    frame = pq.read_table(path).to_pandas()
    required = {
        "model_name",
        "model_family",
        "score_semantics",
        "threshold_mode",
        "problem_id",
        "family",
        "dimension",
        "FE_ratio",
        "p_skip",
        "p_ela",
        "runtime_analysis",
        "runtime_selection",
        "runtime_skip_optimization",
        "runtime_ela_optimization",
        "runtime_always_ela_traditional_aas",
        "runtime_never_ela_sbs",
        "runtime_decision_before_feature",
        "decision_run_ela",
        "decision_score",
        "label_source",
        target_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"predictions file is missing required columns: {missing}")
    if "zero" in set(frame["threshold_mode"].astype(str)):
        frame = frame[frame["threshold_mode"].astype(str) == "zero"].copy()
    else:
        identity_columns = [
            "model_name",
            "model_family",
            "score_semantics",
            "split",
            "problem_id",
            "family",
            "dimension",
            "prefix_algorithm",
            "seed",
            "FE",
            "FE_ratio",
        ]
        frame = frame.sort_values("threshold_mode").drop_duplicates(identity_columns, keep="first").copy()
    frame["score_source_threshold_mode"] = frame["threshold_mode"].astype(str)
    return frame


def _read_train_predictions(path: Path, target_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing train predictions file: {path}")
    frame = pq.read_table(path).to_pandas()
    required = {
        "model_name",
        "model_family",
        "score_semantics",
        "family",
        "FE_ratio",
        "decision_score",
        "label_source",
        target_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"train predictions file is missing required columns: {missing}")
    return frame


def _read_baselines(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing ablation policy summary: {path}")
    frame = pq.read_table(path).to_pandas()
    rows = frame[(frame["policy_name"].isin(BASELINE_POLICIES)) & (frame["layer"].isin(GROUP_LAYERS))].copy()
    rows["gate_name"] = rows["policy_name"]
    rows["gate_category"] = np.where(rows["policy_name"] == "best_observed_analysis_action", "reference_upper_bound", "baseline")
    rows["deployable_policy"] = rows["policy_name"] != "best_observed_analysis_action"
    rows["uses_validation_utility_for_threshold"] = False
    rows["threshold_value"] = np.nan
    rows["threshold_detail"] = ""
    rows["model_name"] = ""
    rows["model_family"] = ""
    rows["score_semantics"] = ""
    return rows


def _threshold_table(train: pd.DataFrame, validation: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_name, model_family, score_semantics), model_train in train.groupby(
        ["model_name", "model_family", "score_semantics"],
        sort=True,
    ):
        model_validation = validation[
            (validation["model_name"] == model_name)
            & (validation["model_family"] == model_family)
            & (validation["score_semantics"] == score_semantics)
        ]
        rows.append(
            {
                "model_name": model_name,
                "model_family": model_family,
                "score_semantics": score_semantics,
                "gate_name": "zero_threshold",
                "threshold_scope": "global",
                "family": "",
                "FE_ratio": np.nan,
                "threshold_value": 0.0,
                "deployable_policy": True,
                "uses_validation_utility_for_threshold": False,
            }
        )
        changed_train = model_train[model_train["label_source"] == "changed_algorithm"]
        rows.append(
            {
                "model_name": model_name,
                "model_family": model_family,
                "score_semantics": score_semantics,
                "gate_name": "changed_train_utility_threshold",
                "threshold_scope": "changed_train",
                "family": "",
                "FE_ratio": np.nan,
                "threshold_value": _utility_threshold(changed_train["decision_score"], changed_train[target_column]),
                "deployable_policy": True,
                "uses_validation_utility_for_threshold": False,
            }
        )
        same_train = model_train[model_train["label_source"] == "same_algorithm"]
        rows.append(
            {
                "model_name": model_name,
                "model_family": model_family,
                "score_semantics": score_semantics,
                "gate_name": "same_algorithm_score_guard_q95",
                "threshold_scope": "same_train_q95",
                "family": "",
                "FE_ratio": np.nan,
                "threshold_value": float(same_train["decision_score"].quantile(0.95)) if len(same_train) else np.inf,
                "deployable_policy": True,
                "uses_validation_utility_for_threshold": False,
            }
        )
        rows.append(
            {
                "model_name": model_name,
                "model_family": model_family,
                "score_semantics": score_semantics,
                "gate_name": "changed_algorithm_label_guard",
                "threshold_scope": "validation_label_source_guard",
                "family": "",
                "FE_ratio": np.nan,
                "threshold_value": 0.0,
                "deployable_policy": False,
                "uses_validation_utility_for_threshold": False,
            }
        )
        for fe_ratio, stage_train in model_train.groupby("FE_ratio", sort=True):
            rows.append(
                {
                    "model_name": model_name,
                    "model_family": model_family,
                    "score_semantics": score_semantics,
                    "gate_name": "stage_train_utility_threshold",
                    "threshold_scope": "train_FE_ratio",
                    "family": "",
                    "FE_ratio": float(fe_ratio),
                    "threshold_value": _utility_threshold(stage_train["decision_score"], stage_train[target_column]),
                    "deployable_policy": True,
                    "uses_validation_utility_for_threshold": False,
                }
            )
        for (family, fe_ratio), family_stage_train in model_train.groupby(["family", "FE_ratio"], sort=True):
            rows.append(
                {
                    "model_name": model_name,
                    "model_family": model_family,
                    "score_semantics": score_semantics,
                    "gate_name": "family_stage_train_utility_threshold",
                    "threshold_scope": "train_family_FE_ratio",
                    "family": str(family),
                    "FE_ratio": float(fe_ratio),
                    "threshold_value": _utility_threshold(
                        family_stage_train["decision_score"], family_stage_train[target_column]
                    ),
                    "deployable_policy": True,
                    "uses_validation_utility_for_threshold": False,
                }
            )
        for (family, fe_ratio), group in model_validation.groupby(["family", "FE_ratio"], sort=True):
            rows.append(
                {
                    "model_name": model_name,
                    "model_family": model_family,
                    "score_semantics": score_semantics,
                    "gate_name": "family_stage_validation_descriptive_threshold",
                    "threshold_scope": "validation_family_FE_ratio",
                    "family": str(family),
                    "FE_ratio": float(fe_ratio),
                    "threshold_value": _utility_threshold(group["decision_score"], group[target_column]),
                    "deployable_policy": False,
                    "uses_validation_utility_for_threshold": True,
                }
            )
    return pd.DataFrame(rows)


def _utility_threshold(scores: pd.Series, utility: pd.Series) -> float:
    if len(scores) == 0:
        return np.inf
    score_values = scores.to_numpy(dtype=float)
    utility_values = utility.to_numpy(dtype=float)
    candidates = np.unique(np.concatenate(([0.0], score_values)))
    best_threshold = 0.0
    best_utility = -np.inf
    for threshold in candidates:
        mean_utility = float(np.mean(np.where(score_values > threshold, utility_values, 0.0)))
        if mean_utility > best_utility:
            best_utility = mean_utility
            best_threshold = float(threshold)
    return best_threshold


def _policy_summary(
    validation: pd.DataFrame,
    baselines: pd.DataFrame,
    thresholds: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    rows = [baselines]
    for (model_name, model_family, score_semantics), model_frame in validation.groupby(
        ["model_name", "model_family", "score_semantics"],
        sort=True,
    ):
        model_thresholds = thresholds[thresholds["model_name"] == model_name]
        for gate_name in [
            "zero_threshold",
            "changed_train_utility_threshold",
            "stage_train_utility_threshold",
            "family_stage_train_utility_threshold",
            "same_algorithm_score_guard_q95",
            "changed_algorithm_label_guard",
            "family_stage_validation_descriptive_threshold",
        ]:
            gate_frame = _apply_gate(model_frame, model_thresholds, gate_name, target_column)
            for eval_domain, label_source in EVALUATION_DOMAINS.items():
                domain = gate_frame if label_source is None else gate_frame[gate_frame["label_source"] == label_source]
                for layer, columns in GROUP_LAYERS.items():
                    rows.append(
                        _grouped_rows(
                            frame=domain,
                            gate_name=gate_name,
                            model_name=str(model_name),
                            model_family=str(model_family),
                            score_semantics=str(score_semantics),
                            eval_domain=eval_domain,
                            layer=layer,
                            group_columns=columns,
                            target_column=target_column,
                        )
                    )
    return pd.concat(rows, ignore_index=True)


def _apply_gate(
    frame: pd.DataFrame,
    thresholds: pd.DataFrame,
    gate_name: str,
    target_column: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["gate_name"] = gate_name
    result["policy_name"] = f"gated_decision_{gate_name}"
    result["gate_category"] = "gated_proposed"
    result["deployable_policy"] = bool(thresholds[thresholds["gate_name"] == gate_name]["deployable_policy"].iloc[0])
    result["uses_validation_utility_for_threshold"] = bool(
        thresholds[thresholds["gate_name"] == gate_name]["uses_validation_utility_for_threshold"].iloc[0]
    )
    result["threshold_detail"] = gate_name
    if gate_name in {"zero_threshold", "changed_train_utility_threshold", "same_algorithm_score_guard_q95"}:
        threshold = float(thresholds[thresholds["gate_name"] == gate_name]["threshold_value"].iloc[0])
        run = result["decision_score"].to_numpy(dtype=float) > threshold
        threshold_values = np.full(len(result), threshold)
    elif gate_name == "stage_train_utility_threshold":
        threshold_map = thresholds[thresholds["gate_name"] == gate_name].set_index("FE_ratio")["threshold_value"].to_dict()
        threshold_values = result["FE_ratio"].map(threshold_map).fillna(np.inf).to_numpy(dtype=float)
        run = result["decision_score"].to_numpy(dtype=float) > threshold_values
    elif gate_name == "family_stage_train_utility_threshold":
        threshold_map = thresholds[thresholds["gate_name"] == gate_name].set_index(["family", "FE_ratio"])[
            "threshold_value"
        ].to_dict()
        threshold_values = np.array(
            [threshold_map.get((str(row.family), float(row.FE_ratio)), np.inf) for row in result.itertuples()],
            dtype=float,
        )
        run = result["decision_score"].to_numpy(dtype=float) > threshold_values
    elif gate_name == "changed_algorithm_label_guard":
        threshold_values = np.zeros(len(result), dtype=float)
        run = (result["decision_score"].to_numpy(dtype=float) > 0.0) & (
            result["label_source"].to_numpy(dtype=str) == "changed_algorithm"
        )
    elif gate_name == "family_stage_validation_descriptive_threshold":
        threshold_map = thresholds[thresholds["gate_name"] == gate_name].set_index(["family", "FE_ratio"])[
            "threshold_value"
        ].to_dict()
        threshold_values = np.array(
            [threshold_map.get((str(row.family), float(row.FE_ratio)), np.inf) for row in result.itertuples()],
            dtype=float,
        )
        run = result["decision_score"].to_numpy(dtype=float) > threshold_values
    else:
        raise ValueError(f"unknown gate: {gate_name}")
    result["gated_threshold"] = threshold_values
    result["gated_run_ela"] = run
    observed = result[target_column].to_numpy(dtype=float)
    result["utility_gated_decision"] = np.where(run, observed, 0.0)
    result["final_performance_gated_decision"] = np.where(run, result["p_ela"], result["p_skip"])
    prediction_overhead = _prediction_overhead(result)
    result["runtime_gated_decision"] = prediction_overhead + np.where(
        run,
        result["runtime_always_ela_traditional_aas"],
        result["runtime_never_ela_sbs"],
    )
    return result


def _prediction_overhead(frame: pd.DataFrame) -> np.ndarray:
    original_action_runtime = np.where(
        frame["decision_run_ela"].to_numpy(dtype=bool),
        frame["runtime_always_ela_traditional_aas"].to_numpy(dtype=float),
        frame["runtime_never_ela_sbs"].to_numpy(dtype=float),
    )
    overhead = frame["runtime_decision_before_feature"].to_numpy(dtype=float) - original_action_runtime
    return np.maximum(overhead, 0.0)


def _grouped_rows(
    *,
    frame: pd.DataFrame,
    gate_name: str,
    model_name: str,
    model_family: str,
    score_semantics: str,
    eval_domain: str,
    layer: str,
    group_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if not group_columns:
        return pd.DataFrame(
            [
                _policy_row(
                    frame=frame,
                    gate_name=gate_name,
                    model_name=model_name,
                    model_family=model_family,
                    score_semantics=score_semantics,
                    eval_domain=eval_domain,
                    layer=layer,
                    group={},
                    target_column=target_column,
                )
            ]
        )
    rows = []
    for group_values, subset in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        rows.append(
            _policy_row(
                frame=subset,
                gate_name=gate_name,
                model_name=model_name,
                model_family=model_family,
                score_semantics=score_semantics,
                eval_domain=eval_domain,
                layer=layer,
                group=dict(zip(group_columns, group_values, strict=True)),
                target_column=target_column,
            )
        )
    return pd.DataFrame(rows)


def _policy_row(
    *,
    frame: pd.DataFrame,
    gate_name: str,
    model_name: str,
    model_family: str,
    score_semantics: str,
    eval_domain: str,
    layer: str,
    group: dict[str, Any],
    target_column: str,
) -> dict[str, Any]:
    observed = frame[target_column].to_numpy(dtype=float)
    utility = frame["utility_gated_decision"].to_numpy(dtype=float)
    run = frame["gated_run_ela"].to_numpy(dtype=bool)
    observed_help = observed > 0.0
    captured_positive = observed[observed_help & run]
    unhelpful_call = observed[(~observed_help) & run]
    positive_utility_sum = float(np.sum(observed[observed_help]))
    captured_positive_sum = float(np.sum(captured_positive))
    return {
        "policy_name": f"gated_decision_{gate_name}",
        "gate_name": gate_name,
        "gate_category": "gated_proposed",
        "model_name": model_name,
        "model_family": model_family,
        "score_semantics": score_semantics,
        "eval_domain": eval_domain,
        "layer": layer,
        "group": _group_label(group),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "problem_id": group.get("problem_id"),
        "deployable_policy": bool(frame["deployable_policy"].iloc[0]),
        "uses_validation_utility_for_threshold": bool(frame["uses_validation_utility_for_threshold"].iloc[0]),
        "threshold_value_mean": float(np.mean(frame["gated_threshold"])),
        "rows": int(len(frame)),
        "observed_utility_gt_zero_rows": int(np.sum(observed_help)),
        "observed_utility_gt_zero_rate": float(np.mean(observed_help)),
        "ela_call_count": int(np.sum(run)),
        "ela_call_rate": float(np.mean(run)),
        "positive_row_capture_rate": float(np.mean(run[observed_help])) if np.any(observed_help) else 0.0,
        "utility_capture_rate": captured_positive_sum / positive_utility_sum if positive_utility_sum > 0.0 else 0.0,
        "captured_positive_utility_sum": captured_positive_sum,
        "unhelpful_call_count": int(np.sum((~observed_help) & run)),
        "unhelpful_call_utility_sum": float(np.sum(unhelpful_call)),
        "unhelpful_call_cost_sum": float(-np.sum(unhelpful_call)),
        "utility_sum": float(np.sum(utility)),
        "utility_mean": float(np.mean(utility)),
        "utility_median": float(np.median(utility)),
        "final_performance_mean": float(np.mean(frame["final_performance_gated_decision"])),
        "runtime_mean_seconds": float(np.mean(frame["runtime_gated_decision"])),
        "runtime_median_seconds": float(np.median(frame["runtime_gated_decision"])),
    }


def _pareto_points(policy_rows: pd.DataFrame) -> pd.DataFrame:
    points = policy_rows[(policy_rows["layer"] == "overall") & (policy_rows["eval_domain"].isin(EVAL_DOMAINS))].copy()
    points["plot_label"] = points.apply(_plot_label, axis=1)
    return points.reset_index(drop=True)


def _plot_label(row: pd.Series) -> str:
    policy = str(row["policy_name"])
    if policy == "no_ela_sbs":
        return "No ELA / SBS"
    if policy == "always_ela_traditional_aas":
        return "Always ELA"
    if policy == "random_analysis_p50":
        return "Random p=0.5"
    if policy == "best_observed_analysis_action":
        return "Best observed action"
    short_gate = {
        "zero_threshold": "zero",
        "changed_train_utility_threshold": "changed-thr",
        "stage_train_utility_threshold": "stage-thr",
        "family_stage_train_utility_threshold": "family-stage-thr",
        "same_algorithm_score_guard_q95": "same-q95",
        "changed_algorithm_label_guard": "changed-label",
        "family_stage_validation_descriptive_threshold": "family-stage-desc",
    }.get(str(row["gate_name"]), str(row["gate_name"]))
    return f"{row['model_family']} {short_gate}"


def _frontier_rows(points: pd.DataFrame) -> pd.DataFrame:
    frames = []
    eligible = points[(points["deployable_policy"]) & (~points["uses_validation_utility_for_threshold"])]
    for eval_domain, group in eligible.groupby("eval_domain", sort=True):
        for cost_column in ["ela_call_rate", "runtime_mean_seconds"]:
            frontier = _non_dominated(group, cost_column, "utility_mean").assign(frontier_cost_axis=cost_column)
            frames.append(frontier)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _non_dominated(frame: pd.DataFrame, cost_column: str, quality_column: str) -> pd.DataFrame:
    ordered = frame.sort_values([cost_column, quality_column], ascending=[True, False]).copy()
    keep = []
    best_quality = -np.inf
    for index, row in ordered.iterrows():
        quality = float(row[quality_column])
        if quality > best_quality:
            keep.append(index)
            best_quality = quality
    return ordered.loc[keep].copy()


def _draw_pareto(
    points: pd.DataFrame,
    frontier: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
    svg_path: Path,
    dataset_name: str,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 140,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    colors = {
        "baseline": "#4C78A8",
        "reference_upper_bound": "#54A24B",
        "gated_proposed": "#F58518",
    }
    markers = {
        "baseline": "o",
        "reference_upper_bound": "*",
        "gated_proposed": "s",
    }
    titles = {
        "all_validation": "All validation",
        "changed_algorithm_validation": "Changed algorithm rows",
        "same_algorithm_reference": "Same algorithm reference",
    }
    for col, eval_domain in enumerate(EVAL_DOMAINS):
        domain_points = points[points["eval_domain"] == eval_domain]
        domain_frontier = frontier[
            (frontier["eval_domain"] == eval_domain) & (frontier["frontier_cost_axis"] == "ela_call_rate")
        ]
        _draw_panel(
            axes[0, col],
            domain_points,
            domain_frontier,
            x_column="ela_call_rate",
            title=titles[eval_domain],
            colors=colors,
            markers=markers,
            draw_frontier=True,
        )
        domain_runtime_frontier = frontier[
            (frontier["eval_domain"] == eval_domain) & (frontier["frontier_cost_axis"] == "runtime_mean_seconds")
        ]
        _draw_panel(
            axes[1, col],
            domain_points,
            domain_runtime_frontier,
            x_column="runtime_mean_seconds",
            title="",
            colors=colors,
            markers=markers,
            draw_frontier=True,
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    fig.legend(by_label.values(), by_label.keys(), loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle(f"Gated cost-performance Pareto diagnostic ({dataset_name})", y=1.02, fontsize=13)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def _draw_panel(
    ax: plt.Axes,
    domain_points: pd.DataFrame,
    frontier: pd.DataFrame,
    *,
    x_column: str,
    title: str,
    colors: dict[str, str],
    markers: dict[str, str],
    draw_frontier: bool,
) -> None:
    frontier_labels = set(frontier["plot_label"].astype(str)) if not frontier.empty else set()
    label_names = {"No ELA / SBS", "Always ELA", "Random p=0.5", "Best observed action", *frontier_labels}
    for _, row in domain_points.iterrows():
        category = str(row["gate_category"])
        deployable = bool(row["deployable_policy"]) and not bool(row["uses_validation_utility_for_threshold"])
        alpha = 0.9 if deployable else 0.35
        ax.scatter(
            float(row[x_column]),
            float(row["utility_mean"]),
            s=120 if category == "reference_upper_bound" else 54,
            marker=markers[category],
            color=colors[category],
            edgecolor="black",
            linewidth=0.6,
            alpha=alpha,
            label=str(row["plot_label"]),
        )
        if str(row["plot_label"]) in label_names:
            ax.annotate(
                str(row["plot_label"]),
                (float(row[x_column]), float(row["utility_mean"])),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
                alpha=0.85,
            )
    if draw_frontier and not frontier.empty:
        front = frontier.sort_values(x_column)
        ax.plot(front[x_column], front["utility_mean"], color="#222222", linewidth=1.5, linestyle="--", label="Deployable frontier")
    ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle=":")
    ax.set_xlabel("ELA call rate" if x_column == "ela_call_rate" else "Mean runtime (s)")
    ax.set_ylabel("Mean relative utility")
    ax.set_title(title)
    ax.grid(True, alpha=0.22)


def _diagnostic_conclusion(points: pd.DataFrame, frontier: pd.DataFrame) -> dict[str, Any]:
    def top_rows(frame: pd.DataFrame, n: int = 5) -> list[dict[str, Any]]:
        columns = [
            "eval_domain",
            "plot_label",
            "gate_category",
            "model_family",
            "gate_name",
            "deployable_policy",
            "uses_validation_utility_for_threshold",
            "ela_call_rate",
            "runtime_mean_seconds",
            "utility_mean",
            "utility_sum",
            "utility_capture_rate",
            "unhelpful_call_cost_sum",
        ]
        return frame[columns].sort_values("utility_sum", ascending=False).head(n).to_dict(orient="records")

    all_points = points[points["eval_domain"] == "all_validation"]
    changed_points = points[points["eval_domain"] == "changed_algorithm_validation"]
    same_points = points[points["eval_domain"] == "same_algorithm_reference"]
    deployable = points[(points["deployable_policy"]) & (~points["uses_validation_utility_for_threshold"])]
    return {
        "frontier": top_rows(frontier, n=20),
        "best_deployable_all_validation": top_rows(deployable[deployable["eval_domain"] == "all_validation"], n=5),
        "best_deployable_changed_algorithm_validation": top_rows(
            deployable[deployable["eval_domain"] == "changed_algorithm_validation"], n=5
        ),
        "best_deployable_same_algorithm_reference": top_rows(
            deployable[deployable["eval_domain"] == "same_algorithm_reference"], n=5
        ),
        "best_descriptive_family_stage_all_validation": top_rows(
            all_points[all_points["gate_name"] == "family_stage_validation_descriptive_threshold"], n=3
        ),
        "best_descriptive_family_stage_changed_algorithm_validation": top_rows(
            changed_points[changed_points["gate_name"] == "family_stage_validation_descriptive_threshold"], n=3
        ),
        "best_descriptive_family_stage_same_algorithm_reference": top_rows(
            same_points[same_points["gate_name"] == "family_stage_validation_descriptive_threshold"], n=3
        ),
        "interpretation": (
            "Deployable frontier excludes best_observed_analysis_action, changed_algorithm_label_guard, and "
            "family_stage_validation_descriptive_threshold. The descriptive family-stage threshold is included only "
            "to estimate calibration opportunity under existing predictions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gated min-support cost-performance Pareto diagnostics.")
    parser.add_argument(
        "--validation-predictions",
        type=Path,
        default=Path("results/decision/min_support/fe_transition_model_sensitivity/model_sensitivity_predictions.parquet"),
    )
    parser.add_argument(
        "--train-predictions",
        type=Path,
        default=Path("results/decision/min_support/fe_transition_model_sensitivity/model_sensitivity_train_predictions.parquet"),
    )
    parser.add_argument(
        "--ablation-policy-summary",
        type=Path,
        default=Path("results/decision/min_support/ablation_comparison/ablation_policy_summary.parquet"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--dataset-name", default="fe_transition_model_sensitivity")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/gated_pareto_diagnostic"),
    )
    args = parser.parse_args()
    run_gated_pareto_diagnostic(
        validation_predictions_path=args.validation_predictions,
        train_predictions_path=args.train_predictions,
        ablation_policy_summary_path=args.ablation_policy_summary,
        output_dir=args.output_dir,
        target_column=args.target_column,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
