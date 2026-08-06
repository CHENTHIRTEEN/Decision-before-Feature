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
from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _check_target, _json_default


EVALUATION_DOMAINS = {
    "changed_algorithm_validation": "changed_algorithm",
    "same_algorithm_reference": "same_algorithm",
}

GROUP_LAYERS = {
    "overall": [],
    "stage": ["stage"],
    "fe_ratio": ["FE_ratio"],
    "family": ["family"],
    "dimension": ["dimension"],
    "problem_id": ["problem_id"],
    "stage_family": ["stage", "family"],
    "stage_dimension": ["stage", "dimension"],
    "stage_problem_id": ["stage", "problem_id"],
}

def run_stage_threshold_diagnostics(
    *,
    predictions_path: Path,
    train_predictions_path: Path,
    behavior_overlap_role_rows_path: Path,
    output_dir: Path,
    target_column: str,
    early_stage_max_fe_ratio: float,
    late_stage_min_fe_ratio: float,
) -> dict[str, Any]:
    _check_target(target_column)
    _check_stage_bounds(early_stage_max_fe_ratio, late_stage_min_fe_ratio)

    predictions = pq.read_table(predictions_path).to_pandas()
    _check_prediction_columns(predictions, target_column)
    predictions = _with_stage(predictions, early_stage_max_fe_ratio, late_stage_min_fe_ratio)
    train_predictions = _read_train_predictions(train_predictions_path, target_column)
    if not train_predictions.empty:
        train_predictions = _with_stage(train_predictions, early_stage_max_fe_ratio, late_stage_min_fe_ratio)

    available_policy = _available_policy_summary(predictions, target_column)
    stage_thresholds = _stage_thresholds_from_train(train_predictions, target_column)
    if stage_thresholds.empty:
        stage_policy = _stage_train_utility_unavailable_rows(predictions)
    else:
        stage_policy = _stage_train_utility_policy_summary(predictions, stage_thresholds, target_column)
    policy_summary = pd.concat([available_policy, stage_policy], ignore_index=True)
    threshold_summary = _threshold_summary(
        predictions,
        stage_thresholds,
        early_stage_max_fe_ratio,
        late_stage_min_fe_ratio,
    )
    stage_role_summary = _stage_role_summary(
        behavior_overlap_role_rows_path,
        early_stage_max_fe_ratio,
        late_stage_min_fe_ratio,
        target_column,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "stage_threshold_policy_summary.parquet"
    threshold_path = output_dir / "stage_threshold_threshold_summary.parquet"
    stage_role_path = output_dir / "stage_threshold_behavior_overlap_roles.parquet"
    summary_path = output_dir / "stage_threshold_summary.json"
    pq.write_table(pa.Table.from_pandas(policy_summary, preserve_index=False), policy_path)
    pq.write_table(pa.Table.from_pandas(threshold_summary, preserve_index=False), threshold_path)
    pq.write_table(pa.Table.from_pandas(stage_role_summary, preserve_index=False), stage_role_path)

    summary = {
        "experiment": "min_support_stage_aware_threshold_diagnostics",
        "predictions": str(predictions_path),
        "train_predictions": str(train_predictions_path),
        "behavior_overlap_role_rows": str(behavior_overlap_role_rows_path),
        "target_column": target_column,
        "stage_definition": {
            "early_stage": f"FE_ratio <= {early_stage_max_fe_ratio}",
            "middle_stage": f"{early_stage_max_fe_ratio} < FE_ratio < {late_stage_min_fe_ratio}",
            "late_stage": f"FE_ratio >= {late_stage_min_fe_ratio}",
        },
        "threshold_modes": {
            "zero": "existing model_sensitivity zero-threshold policy",
            "train_utility": "existing model_sensitivity global train-utility threshold policy",
            "stage_train_utility": (
                "threshold selected within each train-stage group from model_sensitivity train predictions, then "
                "applied to validation rows in the matching stage"
            ),
        },
        "rows": {
            "prediction_rows": int(len(predictions)),
            "train_prediction_rows": int(len(train_predictions)),
            "policy_summary_rows": int(len(policy_summary)),
            "threshold_summary_rows": int(len(threshold_summary)),
            "stage_role_summary_rows": int(len(stage_role_summary)),
        },
        "outputs": {
            "policy_summary": str(policy_path),
            "threshold_summary": str(threshold_path),
            "stage_role_summary": str(stage_role_path),
            "summary": str(summary_path),
        },
        "data_leakage_check": {
            "uses_existing_model_sensitivity_predictions_only_for_policy_metrics": True,
            "uses_behavior_overlap_output_only_for_role_stage_context": True,
            "models_retrained": False,
            "threshold_selected_from_validation": False,
            "stage_train_utility_threshold_computed": bool(not stage_thresholds.empty),
            "stage_train_utility_selected_from_train_predictions": bool(not stage_thresholds.empty),
            "ela_features_used_as_decision_input": False,
            "original_utility_labels_modified": False,
        },
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote stage threshold policy summary to {policy_path}")
    print(f"wrote stage threshold threshold summary to {threshold_path}")
    print(f"wrote stage threshold behavior-overlap role summary to {stage_role_path}")
    print(f"wrote stage threshold summary to {summary_path}")
    return summary


def _check_stage_bounds(early_stage_max_fe_ratio: float, late_stage_min_fe_ratio: float) -> None:
    if early_stage_max_fe_ratio < 0.0 or late_stage_min_fe_ratio > 1.0:
        raise ValueError("stage FE_ratio bounds must be within [0, 1]")
    if early_stage_max_fe_ratio >= late_stage_min_fe_ratio:
        raise ValueError("early_stage_max_fe_ratio must be less than late_stage_min_fe_ratio")


def _check_prediction_columns(predictions: pd.DataFrame, target_column: str) -> None:
    required = {
        "model_name",
        "model_family",
        "score_semantics",
        "threshold_mode",
        "decision_threshold",
        "decision_score",
        "decision_run_ela",
        "label_source",
        "family",
        "dimension",
        "FE_ratio",
        "problem_id",
        target_column,
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"model sensitivity predictions missing columns: {missing}")


def _read_train_predictions(path: Path, target_column: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    train_predictions = pq.read_table(path).to_pandas()
    required = {
        "model_name",
        "model_family",
        "score_semantics",
        "decision_score",
        "family",
        "dimension",
        "FE_ratio",
        "problem_id",
        target_column,
    }
    missing = sorted(required.difference(train_predictions.columns))
    if missing:
        raise ValueError(f"model sensitivity train predictions missing columns: {missing}")
    return train_predictions


def _with_stage(frame: pd.DataFrame, early_stage_max_fe_ratio: float, late_stage_min_fe_ratio: float) -> pd.DataFrame:
    result = frame.copy()
    fe_ratio = result["FE_ratio"].to_numpy(dtype=float)
    result["stage"] = np.select(
        [fe_ratio <= early_stage_max_fe_ratio, fe_ratio >= late_stage_min_fe_ratio],
        ["early_stage", "late_stage"],
        default="middle_stage",
    )
    return result


def _available_policy_summary(predictions: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows = []
    available = predictions[predictions["threshold_mode"].isin(["zero", "train_utility"])].copy()
    for (model_name, threshold_mode), model_frame in available.groupby(["model_name", "threshold_mode"], dropna=False):
        for eval_domain, label_source in EVALUATION_DOMAINS.items():
            domain_frame = model_frame[model_frame["label_source"] == label_source]
            for layer, columns in GROUP_LAYERS.items():
                rows.append(
                    _policy_layer_summary(
                        frame=domain_frame,
                        model_name=str(model_name),
                        threshold_mode=str(threshold_mode),
                        eval_domain=eval_domain,
                        layer=layer,
                        group_columns=columns,
                        target_column=target_column,
                    )
                )
    return pd.concat(rows, ignore_index=True)


def _stage_train_utility_policy_summary(
    predictions: pd.DataFrame,
    stage_thresholds: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    base = predictions[predictions["threshold_mode"] == "zero"].copy()
    threshold_lookup = {
        (str(row["model_name"]), str(row["stage"])): float(row["threshold"])
        for _, row in stage_thresholds.iterrows()
        if row["status"] == "computed"
    }
    base["threshold_mode"] = "stage_train_utility"
    base["decision_threshold"] = [
        threshold_lookup[(str(model_name), str(stage))]
        for model_name, stage in zip(base["model_name"], base["stage"], strict=True)
    ]
    base["decision_run_ela"] = base["decision_score"].to_numpy(dtype=float) > base["decision_threshold"].to_numpy(
        dtype=float
    )

    rows = []
    for model_name, model_frame in base.groupby("model_name", dropna=False):
        for eval_domain, label_source in EVALUATION_DOMAINS.items():
            domain_frame = model_frame[model_frame["label_source"] == label_source]
            for layer, columns in GROUP_LAYERS.items():
                rows.append(
                    _policy_layer_summary(
                        frame=domain_frame,
                        model_name=str(model_name),
                        threshold_mode="stage_train_utility",
                        eval_domain=eval_domain,
                        layer=layer,
                        group_columns=columns,
                        target_column=target_column,
                    )
                )
    return pd.concat(rows, ignore_index=True)


def _policy_layer_summary(
    *,
    frame: pd.DataFrame,
    model_name: str,
    threshold_mode: str,
    eval_domain: str,
    layer: str,
    group_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if not group_columns:
        return pd.DataFrame([_policy_row(frame, model_name, threshold_mode, eval_domain, layer, {}, target_column)])
    rows = []
    for group_values, subset in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group = dict(zip(group_columns, group_values, strict=True))
        rows.append(_policy_row(subset, model_name, threshold_mode, eval_domain, layer, group, target_column))
    return pd.DataFrame(rows)


def _policy_row(
    frame: pd.DataFrame,
    model_name: str,
    threshold_mode: str,
    eval_domain: str,
    layer: str,
    group: dict[str, Any],
    target_column: str,
) -> dict[str, Any]:
    observed = frame[target_column].to_numpy(dtype=float)
    observed_gt_zero = observed > 0.0
    decision_run = frame["decision_run_ela"].to_numpy(dtype=bool)
    captured_positive = observed[observed_gt_zero & decision_run]
    missed_positive = observed[observed_gt_zero & ~decision_run]
    unhelpful_call = observed[(~observed_gt_zero) & decision_run]
    positive_utility_sum = float(np.sum(observed[observed_gt_zero]))
    captured_positive_utility_sum = float(np.sum(captured_positive))
    return {
        **_common_fields(frame, model_name, threshold_mode, eval_domain, layer, group),
        "status": "computed",
        "rows": int(len(frame)),
        "utility_gt_zero_rows": int(np.sum(observed_gt_zero)),
        "utility_gt_zero_rate": float(np.mean(observed_gt_zero)),
        "positive_utility_sum": positive_utility_sum,
        "decision_ela_call_rows": int(np.sum(decision_run)),
        "decision_ela_call_rate": float(np.mean(decision_run)),
        "true_run_ela_count": int(np.sum(observed_gt_zero & decision_run)),
        "missed_run_ela_count": int(np.sum(observed_gt_zero & ~decision_run)),
        "unhelpful_run_ela_count": int(np.sum((~observed_gt_zero) & decision_run)),
        "positive_row_capture_rate": float(np.mean(decision_run[observed_gt_zero])) if np.any(observed_gt_zero) else 0.0,
        "utility_capture_rate": (
            captured_positive_utility_sum / positive_utility_sum if positive_utility_sum > 0.0 else 0.0
        ),
        "captured_positive_utility_sum": captured_positive_utility_sum,
        "missed_positive_utility_sum": float(np.sum(missed_positive)),
        "unhelpful_call_utility_sum": float(np.sum(unhelpful_call)),
        "unhelpful_call_cost_sum": float(-np.sum(unhelpful_call)),
        "decision_utility_sum": float(np.sum(np.where(decision_run, observed, 0.0))),
        "decision_mean_utility": float(np.mean(np.where(decision_run, observed, 0.0))),
        "note": None,
    }


def _stage_train_utility_unavailable_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    reference = predictions[predictions["threshold_mode"] == "zero"].copy()
    rows = []
    for model_name, model_frame in reference.groupby("model_name", dropna=False):
        for eval_domain, label_source in EVALUATION_DOMAINS.items():
            domain_frame = model_frame[model_frame["label_source"] == label_source]
            for layer, columns in GROUP_LAYERS.items():
                if domain_frame.empty:
                    continue
                if not columns:
                    rows.append(_unavailable_row(domain_frame, str(model_name), eval_domain, layer, {}))
                    continue
                for group_values, subset in domain_frame.groupby(columns, dropna=False):
                    if not isinstance(group_values, tuple):
                        group_values = (group_values,)
                    group = dict(zip(columns, group_values, strict=True))
                    rows.append(_unavailable_row(subset, str(model_name), eval_domain, layer, group))
    return pd.DataFrame(rows)


def _unavailable_row(
    frame: pd.DataFrame,
    model_name: str,
    eval_domain: str,
    layer: str,
    group: dict[str, Any],
) -> dict[str, Any]:
    row = {
        **_common_fields(frame, model_name, "stage_train_utility", eval_domain, layer, group),
        "status": "not_computable_from_existing_outputs",
        "rows": int(len(frame)),
        "utility_gt_zero_rows": None,
        "utility_gt_zero_rate": None,
        "positive_utility_sum": None,
        "decision_ela_call_rows": None,
        "decision_ela_call_rate": None,
        "true_run_ela_count": None,
        "missed_run_ela_count": None,
        "unhelpful_run_ela_count": None,
        "positive_row_capture_rate": None,
        "utility_capture_rate": None,
        "captured_positive_utility_sum": None,
        "missed_positive_utility_sum": None,
        "unhelpful_call_utility_sum": None,
        "unhelpful_call_cost_sum": None,
        "decision_utility_sum": None,
        "decision_mean_utility": None,
        "note": (
            "stage_train_utility requires train-set model scores grouped by FE_ratio/stage; current "
            "model_sensitivity train predictions are missing or empty"
        ),
    }
    row["threshold"] = None
    row["threshold_min"] = None
    row["threshold_max"] = None
    row["threshold_unique_values"] = 0
    return row


def _threshold_summary(
    predictions: pd.DataFrame,
    stage_thresholds: pd.DataFrame,
    early_stage_max_fe_ratio: float,
    late_stage_min_fe_ratio: float,
) -> pd.DataFrame:
    available = (
        predictions[predictions["threshold_mode"].isin(["zero", "train_utility"])]
        .groupby(["model_name", "model_family", "score_semantics", "threshold_mode"], as_index=False)
        .agg(threshold=("decision_threshold", "first"), threshold_unique_values=("decision_threshold", "nunique"))
    )
    if stage_thresholds.empty:
        stage_rows = []
        for _, row in predictions[predictions["threshold_mode"] == "zero"][
            ["model_name", "model_family", "score_semantics"]
        ].drop_duplicates().iterrows():
            stage_rows.append(
                {
                    "model_name": row["model_name"],
                    "model_family": row["model_family"],
                    "score_semantics": row["score_semantics"],
                    "threshold_mode": "stage_train_utility",
                    "stage": None,
                    "threshold": None,
                    "threshold_unique_values": 0,
                    "train_rows": 0,
                    "train_policy_mean_utility": None,
                    "status": "not_computable_from_existing_outputs",
                    "stage_definition": (
                        f"early<= {early_stage_max_fe_ratio}; middle between; late>= {late_stage_min_fe_ratio}"
                    ),
                }
            )
        stage_threshold_frame = pd.DataFrame(stage_rows)
    else:
        stage_threshold_frame = stage_thresholds.copy()
        stage_threshold_frame["threshold_mode"] = "stage_train_utility"
        stage_threshold_frame["threshold_unique_values"] = 1
        stage_threshold_frame["stage_definition"] = (
            f"early<= {early_stage_max_fe_ratio}; middle between; late>= {late_stage_min_fe_ratio}"
        )
    available["status"] = "computed"
    available["stage_definition"] = None
    available["stage"] = None
    available["train_rows"] = None
    available["train_policy_mean_utility"] = None
    return pd.concat([available, stage_threshold_frame], ignore_index=True)


def _stage_thresholds_from_train(train_predictions: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if train_predictions.empty:
        return pd.DataFrame()
    rows = []
    for (model_name, stage), subset in train_predictions.groupby(["model_name", "stage"], dropna=False):
        threshold, train_policy_mean_utility = _decision_threshold_from_scores(
            scores=subset["decision_score"].to_numpy(dtype=float),
            observed=subset[target_column].to_numpy(dtype=float),
        )
        rows.append(
            {
                "model_name": str(model_name),
                "model_family": str(subset["model_family"].iloc[0]),
                "score_semantics": str(subset["score_semantics"].iloc[0]),
                "stage": str(stage),
                "threshold": float(threshold),
                "train_rows": int(len(subset)),
                "train_utility_gt_zero_rows": int((subset[target_column] > 0.0).sum()),
                "train_policy_mean_utility": float(train_policy_mean_utility),
                "status": "computed",
            }
        )
    return pd.DataFrame(rows)


def _decision_threshold_from_scores(scores: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.concatenate(([0.0], scores)))
    best_threshold = 0.0
    best_utility = -np.inf
    for threshold in candidates:
        policy_utility = np.where(scores > threshold, observed, 0.0)
        mean_utility = float(np.mean(policy_utility))
        if mean_utility > best_utility:
            best_utility = mean_utility
            best_threshold = float(threshold)
    return best_threshold, float(best_utility)


def _stage_role_summary(
    behavior_overlap_role_rows_path: Path,
    early_stage_max_fe_ratio: float,
    late_stage_min_fe_ratio: float,
    target_column: str,
) -> pd.DataFrame:
    if not behavior_overlap_role_rows_path.exists():
        return pd.DataFrame(
            [
                {
                    "status": "missing_behavior_overlap_role_rows",
                    "path": str(behavior_overlap_role_rows_path),
                }
            ]
        )
    role_rows = pq.read_table(behavior_overlap_role_rows_path).to_pandas()
    if role_rows.empty:
        return pd.DataFrame()
    role_rows = _with_stage(role_rows, early_stage_max_fe_ratio, late_stage_min_fe_ratio)
    overall = role_rows[role_rows["layer"] == "overall"].copy()
    if overall.empty:
        return pd.DataFrame()
    return (
        overall.groupby(["model_name_for_overlap", "threshold_mode_for_overlap", "role", "stage"], as_index=False)
        .agg(
            rows=("role", "size"),
            mean_decision_score=("decision_score", "mean"),
            median_decision_score=("decision_score", "median"),
            mean_utility=(target_column, "mean") if target_column in overall.columns else ("FE_ratio", "size"),
        )
        .rename(
            columns={
                "model_name_for_overlap": "model_name",
                "threshold_mode_for_overlap": "threshold_mode",
            }
        )
    )


def _common_fields(
    frame: pd.DataFrame,
    model_name: str,
    threshold_mode: str,
    eval_domain: str,
    layer: str,
    group: dict[str, Any],
) -> dict[str, Any]:
    thresholds = frame["decision_threshold"].dropna().to_numpy(dtype=float)
    unique_thresholds = np.unique(thresholds) if thresholds.size else np.array([], dtype=float)
    threshold = float(unique_thresholds[0]) if unique_thresholds.size == 1 else None
    return {
        "model_name": model_name,
        "model_family": str(frame["model_family"].iloc[0]),
        "score_semantics": str(frame["score_semantics"].iloc[0]),
        "eval_domain": eval_domain,
        "threshold_mode": threshold_mode,
        "threshold": threshold,
        "threshold_min": float(np.min(thresholds)) if thresholds.size else None,
        "threshold_max": float(np.max(thresholds)) if thresholds.size else None,
        "threshold_unique_values": int(unique_thresholds.size),
        "layer": layer,
        "group": _group_label(group),
        "stage": group.get("stage"),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "problem_id": group.get("problem_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate min-support stage-aware threshold diagnostics.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/decision/min_support/model_sensitivity/model_sensitivity_predictions.parquet"),
    )
    parser.add_argument(
        "--train-predictions",
        type=Path,
        default=Path("results/decision/min_support/model_sensitivity/model_sensitivity_train_predictions.parquet"),
    )
    parser.add_argument(
        "--behavior-overlap-role-rows",
        type=Path,
        default=Path("results/decision/min_support/behavior_overlap/behavior_overlap_role_rows.parquet"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--early-stage-max-fe-ratio", type=float, default=0.2)
    parser.add_argument("--late-stage-min-fe-ratio", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/stage_threshold_diagnostics"),
    )
    args = parser.parse_args()
    run_stage_threshold_diagnostics(
        predictions_path=args.predictions,
        train_predictions_path=args.train_predictions,
        behavior_overlap_role_rows_path=args.behavior_overlap_role_rows,
        output_dir=args.output_dir,
        target_column=args.target_column,
        early_stage_max_fe_ratio=args.early_stage_max_fe_ratio,
        late_stage_min_fe_ratio=args.late_stage_min_fe_ratio,
    )


if __name__ == "__main__":
    main()
