from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _check_target, _json_default


MODEL_FAMILIES = ("lightgbm", "random_forest", "xgboost")
EVALUATION_DOMAINS: dict[str, str | None] = {
    "all_validation": None,
    "changed_algorithm_validation": "changed_algorithm",
    "same_algorithm_reference": "same_algorithm",
}
TRAIN_FAMILY_STAGE_GATE = "family_stage_train_utility_threshold"
VALIDATION_FAMILY_STAGE_GATE = "family_stage_validation_descriptive_threshold"
STAGE_TRAIN_GATE = "stage_train_utility_threshold"


def run_family_stage_threshold_transfer_diagnostic(
    *,
    validation_predictions_path: Path,
    train_predictions_path: Path,
    gated_thresholds_path: Path,
    output_dir: Path,
    target_column: str,
) -> dict[str, Any]:
    _check_target(target_column)
    validation = _read_validation_predictions(validation_predictions_path, target_column)
    train = _read_train_predictions(train_predictions_path, target_column)
    thresholds = _read_thresholds(gated_thresholds_path)

    train_support = _train_family_stage_support(train, thresholds, target_column)
    transfer = _validation_transfer_summary(validation, train, thresholds, target_column)
    threshold_shift = _threshold_distribution_by_fe_ratio(thresholds, train_support, transfer)
    reason_summary = _failure_reason_summary(transfer)
    conclusion = _diagnostic_conclusion(train, validation, transfer, reason_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_support_path = output_dir / "family_stage_train_support.parquet"
    transfer_path = output_dir / "family_stage_threshold_transfer.parquet"
    threshold_shift_path = output_dir / "family_stage_threshold_shift_by_fe_ratio.parquet"
    reason_summary_path = output_dir / "family_stage_transfer_failure_reasons.parquet"
    summary_path = output_dir / "family_stage_threshold_transfer_summary.json"

    pq.write_table(pa.Table.from_pandas(train_support, preserve_index=False), train_support_path)
    pq.write_table(pa.Table.from_pandas(transfer, preserve_index=False), transfer_path)
    pq.write_table(pa.Table.from_pandas(threshold_shift, preserve_index=False), threshold_shift_path)
    pq.write_table(pa.Table.from_pandas(reason_summary, preserve_index=False), reason_summary_path)

    summary = {
        "experiment": "min_support_family_stage_threshold_transfer_diagnostic",
        "research_question": (
            "Why does the train-derived family-stage threshold fail to transfer from min-support train rows "
            "to function-family validation rows for LightGBM, RF, and XGBoost?"
        ),
        "target_column": target_column,
        "models": list(MODEL_FAMILIES),
        "inputs": {
            "validation_predictions": str(validation_predictions_path),
            "train_predictions": str(train_predictions_path),
            "gated_thresholds": str(gated_thresholds_path),
        },
        "outputs": {
            "train_support": str(train_support_path),
            "transfer": str(transfer_path),
            "threshold_shift_by_fe_ratio": str(threshold_shift_path),
            "failure_reasons": str(reason_summary_path),
            "summary": str(summary_path),
        },
        "rows": {
            "validation_score_rows": int(len(validation)),
            "train_score_rows": int(len(train)),
            "train_support_rows": int(len(train_support)),
            "transfer_rows": int(len(transfer)),
            "threshold_shift_rows": int(len(threshold_shift)),
            "failure_reason_rows": int(len(reason_summary)),
        },
        "split_support": {
            "train_families": sorted(map(str, train["family"].unique())),
            "validation_families": sorted(map(str, validation["family"].unique())),
            "family_intersection": sorted(set(map(str, train["family"].unique())).intersection(map(str, validation["family"].unique()))),
            "train_fe_ratios": sorted(float(x) for x in train["FE_ratio"].unique()),
            "validation_fe_ratios": sorted(float(x) for x in validation["FE_ratio"].unique()),
        },
        "diagnostic_conclusion": conclusion,
        "data_leakage_check": {
            "uses_existing_model_sensitivity_predictions": True,
            "uses_existing_gated_pareto_thresholds": True,
            "models_retrained": False,
            "utility_labels_regenerated": False,
            "original_utility_labels_modified": False,
            "validation_utility_used_only_for_descriptive_threshold_reference": True,
            "ela_features_used_as_decision_input": False,
            "metadata_used_only_for_diagnostic_grouping": True,
            "formal_phase1_configs_modified": False,
        },
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote train family-stage support to {train_support_path}")
    print(f"wrote family-stage threshold transfer to {transfer_path}")
    print(f"wrote threshold shift summary to {threshold_shift_path}")
    print(f"wrote failure reason summary to {reason_summary_path}")
    print(f"wrote family-stage transfer summary to {summary_path}")
    return summary


def _read_validation_predictions(path: Path, target_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing validation predictions: {path}")
    frame = pq.read_table(path).to_pandas()
    required = {
        "model_name",
        "model_family",
        "score_semantics",
        "threshold_mode",
        "decision_score",
        "label_source",
        "family",
        "dimension",
        "FE_ratio",
        "problem_id",
        target_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"validation predictions missing columns: {missing}")
    frame = frame[frame["model_family"].isin(MODEL_FAMILIES)].copy()
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
        identity_columns = [column for column in identity_columns if column in frame.columns]
        frame = frame.sort_values("threshold_mode").drop_duplicates(identity_columns, keep="first").copy()
    return frame.reset_index(drop=True)


def _read_train_predictions(path: Path, target_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing train predictions: {path}")
    frame = pq.read_table(path).to_pandas()
    required = {
        "model_name",
        "model_family",
        "score_semantics",
        "decision_score",
        "label_source",
        "family",
        "dimension",
        "FE_ratio",
        "problem_id",
        target_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"train predictions missing columns: {missing}")
    return frame[frame["model_family"].isin(MODEL_FAMILIES)].copy().reset_index(drop=True)


def _read_thresholds(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing gated thresholds: {path}")
    frame = pq.read_table(path).to_pandas()
    required = {
        "model_name",
        "model_family",
        "score_semantics",
        "gate_name",
        "family",
        "FE_ratio",
        "threshold_value",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"gated thresholds missing columns: {missing}")
    return frame[frame["model_family"].isin(MODEL_FAMILIES)].copy().reset_index(drop=True)


def _train_family_stage_support(
    train: pd.DataFrame,
    thresholds: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    threshold_lookup = _threshold_lookup(thresholds, TRAIN_FAMILY_STAGE_GATE, ["model_name", "family", "FE_ratio"])
    rows: list[dict[str, Any]] = []
    for keys, group in train.groupby(["model_name", "model_family", "score_semantics", "family", "FE_ratio"], sort=True):
        model_name, model_family, score_semantics, family, fe_ratio = keys
        threshold = threshold_lookup.get((str(model_name), str(family), float(fe_ratio)), np.inf)
        row = {
            "model_name": str(model_name),
            "model_family": str(model_family),
            "score_semantics": str(score_semantics),
            "family": str(family),
            "FE_ratio": float(fe_ratio),
            "threshold_value": float(threshold),
            **_score_stats(group["decision_score"], "train_score"),
            **_utility_stats(group, target_column, "train"),
            **_policy_metrics(group, threshold, target_column, "train_family_stage_threshold_on_train"),
        }
        changed = group[group["label_source"] == "changed_algorithm"]
        same = group[group["label_source"] == "same_algorithm"]
        row.update(_utility_stats(changed, target_column, "train_changed"))
        row.update(_utility_stats(same, target_column, "train_same"))
        rows.append(row)
    return pd.DataFrame(rows)


def _validation_transfer_summary(
    validation: pd.DataFrame,
    train: pd.DataFrame,
    thresholds: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    train_family_threshold = _threshold_lookup(thresholds, TRAIN_FAMILY_STAGE_GATE, ["model_name", "family", "FE_ratio"])
    validation_desc_threshold = _threshold_lookup(
        thresholds, VALIDATION_FAMILY_STAGE_GATE, ["model_name", "family", "FE_ratio"]
    )
    stage_threshold = _threshold_lookup(thresholds, STAGE_TRAIN_GATE, ["model_name", "FE_ratio"])
    train_stage_stats = _train_stage_stats(train, thresholds, target_column)
    train_families_by_model = {
        str(model_name): set(map(str, group["family"].unique()))
        for model_name, group in train.groupby("model_name", sort=False)
    }

    rows: list[dict[str, Any]] = []
    for (model_name, model_family, score_semantics, family, fe_ratio), model_group in validation.groupby(
        ["model_name", "model_family", "score_semantics", "family", "FE_ratio"],
        sort=True,
    ):
        for eval_domain, label_source in EVALUATION_DOMAINS.items():
            group = model_group if label_source is None else model_group[model_group["label_source"] == label_source]
            if group.empty:
                continue
            family_key = (str(model_name), str(family), float(fe_ratio))
            stage_key = (str(model_name), float(fe_ratio))
            train_threshold = train_family_threshold.get(family_key, np.inf)
            desc_threshold = validation_desc_threshold.get(family_key, np.inf)
            stage_train_threshold = stage_threshold.get(stage_key, np.inf)
            train_stage = train_stage_stats.get(stage_key, {})
            threshold_exists = np.isfinite(train_threshold)
            family_seen = str(family) in train_families_by_model.get(str(model_name), set())
            desc_metrics = _policy_metrics(group, desc_threshold, target_column, "validation_descriptive_threshold")
            train_metrics = _policy_metrics(group, train_threshold, target_column, "train_family_stage_threshold")
            stage_metrics = _policy_metrics(group, stage_train_threshold, target_column, "stage_train_threshold")
            row = {
                "eval_domain": eval_domain,
                "model_name": str(model_name),
                "model_family": str(model_family),
                "score_semantics": str(score_semantics),
                "family": str(family),
                "FE_ratio": float(fe_ratio),
                "family_seen_in_train": bool(family_seen),
                "train_family_stage_threshold_exists": bool(threshold_exists),
                "train_family_stage_threshold": float(train_threshold) if threshold_exists else np.nan,
                "validation_descriptive_threshold": float(desc_threshold) if np.isfinite(desc_threshold) else np.nan,
                "stage_train_threshold": float(stage_train_threshold) if np.isfinite(stage_train_threshold) else np.nan,
                **_score_stats(group["decision_score"], "validation_score"),
                **_utility_stats(group, target_column, "validation"),
                **desc_metrics,
                **train_metrics,
                **stage_metrics,
            }
            row.update(train_stage)
            row.update(_threshold_quantile_fields(group, train_threshold, desc_threshold, stage_train_threshold))
            row.update(_failure_reason(row))
            rows.append(row)
    return pd.DataFrame(rows)


def _train_stage_stats(
    train: pd.DataFrame,
    thresholds: pd.DataFrame,
    target_column: str,
) -> dict[tuple[str, float], dict[str, Any]]:
    stage_threshold = _threshold_lookup(thresholds, STAGE_TRAIN_GATE, ["model_name", "FE_ratio"])
    result: dict[tuple[str, float], dict[str, Any]] = {}
    for (model_name, fe_ratio), group in train.groupby(["model_name", "FE_ratio"], sort=True):
        threshold = stage_threshold.get((str(model_name), float(fe_ratio)), np.inf)
        row: dict[str, Any] = {}
        row.update(_score_stats(group["decision_score"], "train_stage_score"))
        row.update(_utility_stats(group, target_column, "train_stage"))
        row.update(_policy_metrics(group, threshold, target_column, "stage_train_threshold_on_train"))
        result[(str(model_name), float(fe_ratio))] = row
    return result


def _threshold_distribution_by_fe_ratio(
    thresholds: pd.DataFrame,
    train_support: pd.DataFrame,
    transfer: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_name, model_family, fe_ratio), group in thresholds.groupby(["model_name", "model_family", "FE_ratio"], sort=True):
        if pd.isna(fe_ratio):
            continue
        train_thresholds = group[group["gate_name"] == TRAIN_FAMILY_STAGE_GATE]["threshold_value"]
        desc_thresholds = group[group["gate_name"] == VALIDATION_FAMILY_STAGE_GATE]["threshold_value"]
        stage_threshold = group[group["gate_name"] == STAGE_TRAIN_GATE]["threshold_value"]
        transfer_group = transfer[
            (transfer["model_name"] == model_name)
            & (transfer["FE_ratio"] == float(fe_ratio))
            & (transfer["eval_domain"] == "changed_algorithm_validation")
        ]
        support_group = train_support[(train_support["model_name"] == model_name) & (train_support["FE_ratio"] == float(fe_ratio))]
        row = {
            "model_name": str(model_name),
            "model_family": str(model_family),
            "FE_ratio": float(fe_ratio),
            "train_family_threshold_groups": int(len(train_thresholds)),
            "validation_descriptive_threshold_groups": int(len(desc_thresholds)),
            "stage_train_threshold": float(stage_threshold.iloc[0]) if len(stage_threshold) else np.nan,
            **_series_stats(train_thresholds, "train_family_threshold"),
            **_series_stats(desc_thresholds, "validation_descriptive_threshold"),
            "threshold_median_delta_train_minus_validation_desc": _safe_median(train_thresholds)
            - _safe_median(desc_thresholds),
            "train_stage_positive_rows": int(support_group["train_utility_gt_zero_rows"].sum()) if not support_group.empty else 0,
            "train_stage_rows": int(support_group["train_rows"].sum()) if not support_group.empty else 0,
            "validation_changed_positive_rows": int(transfer_group["validation_utility_gt_zero_rows"].sum())
            if not transfer_group.empty
            else 0,
            "validation_descriptive_utility_sum": float(
                transfer_group["validation_descriptive_threshold_utility_sum"].sum()
            )
            if not transfer_group.empty
            else 0.0,
            "train_family_stage_transfer_utility_sum": float(
                transfer_group["train_family_stage_threshold_utility_sum"].sum()
            )
            if not transfer_group.empty
            else 0.0,
        }
        row["train_stage_positive_rate"] = (
            row["train_stage_positive_rows"] / row["train_stage_rows"] if row["train_stage_rows"] else 0.0
        )
        row["validation_changed_positive_rate_over_transfer_rows"] = (
            row["validation_changed_positive_rows"] / int(transfer_group["validation_rows"].sum())
            if not transfer_group.empty and int(transfer_group["validation_rows"].sum()) > 0
            else 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _failure_reason_summary(transfer: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in transfer.groupby(["eval_domain", "model_name", "model_family", "primary_failure_reason"], sort=True):
        eval_domain, model_name, model_family, reason = keys
        rows.append(
            {
                "eval_domain": str(eval_domain),
                "model_name": str(model_name),
                "model_family": str(model_family),
                "primary_failure_reason": str(reason),
                "family_stage_groups": int(len(group)),
                "rows": int(group["validation_rows"].sum()),
                "validation_utility_gt_zero_rows": int(group["validation_utility_gt_zero_rows"].sum()),
                "validation_positive_utility_sum": float(group["validation_positive_utility_sum"].sum()),
                "train_family_stage_utility_sum": float(group["train_family_stage_threshold_utility_sum"].sum()),
                "validation_descriptive_utility_sum": float(group["validation_descriptive_threshold_utility_sum"].sum()),
                "stage_train_threshold_utility_sum": float(group["stage_train_threshold_utility_sum"].sum()),
                "train_family_stage_unhelpful_call_cost_sum": float(
                    group["train_family_stage_threshold_unhelpful_call_cost_sum"].sum()
                ),
                "validation_descriptive_unhelpful_call_cost_sum": float(
                    group["validation_descriptive_threshold_unhelpful_call_cost_sum"].sum()
                ),
                "stage_train_unhelpful_call_cost_sum": float(group["stage_train_threshold_unhelpful_call_cost_sum"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _diagnostic_conclusion(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    transfer: pd.DataFrame,
    reason_summary: pd.DataFrame,
) -> dict[str, Any]:
    changed = transfer[transfer["eval_domain"] == "changed_algorithm_validation"]
    all_validation = transfer[transfer["eval_domain"] == "all_validation"]
    summary_rows = []
    for model_family, group in changed.groupby("model_family", sort=True):
        missing_groups = int((~group["train_family_stage_threshold_exists"]).sum())
        groups = int(len(group))
        desc_utility = float(group["validation_descriptive_threshold_utility_sum"].sum())
        train_utility = float(group["train_family_stage_threshold_utility_sum"].sum())
        stage_utility = float(group["stage_train_threshold_utility_sum"].sum())
        summary_rows.append(
            {
                "model_family": str(model_family),
                "changed_algorithm_family_stage_groups": groups,
                "missing_train_family_stage_threshold_groups": missing_groups,
                "missing_train_family_stage_threshold_rate": missing_groups / groups if groups else 0.0,
                "changed_algorithm_validation_descriptive_utility_sum": desc_utility,
                "changed_algorithm_train_family_stage_transfer_utility_sum": train_utility,
                "changed_algorithm_stage_train_threshold_utility_sum": stage_utility,
                "descriptive_minus_train_family_stage_utility_sum": desc_utility - train_utility,
            }
        )
    reason_totals = (
        reason_summary[reason_summary["eval_domain"] == "changed_algorithm_validation"]
        .sort_values("validation_descriptive_utility_sum", ascending=False)
        .head(12)
        .to_dict(orient="records")
    )
    return {
        "train_families": sorted(map(str, train["family"].unique())),
        "validation_families": sorted(map(str, validation["family"].unique())),
        "family_overlap_count": int(len(set(map(str, train["family"].unique())).intersection(map(str, validation["family"].unique())))),
        "model_level_changed_algorithm_summary": summary_rows,
        "changed_algorithm_reason_totals": reason_totals,
        "all_validation_train_family_stage_utility_sum": float(
            all_validation["train_family_stage_threshold_utility_sum"].sum()
        ),
        "all_validation_validation_descriptive_utility_sum": float(
            all_validation["validation_descriptive_threshold_utility_sum"].sum()
        ),
        "interpretation": (
            "The exact family-stage train threshold is not transferable under the current function-family split when "
            "validation families are unseen in train. Score-shift and conservative-threshold fields are retained for "
            "secondary analysis, but missing family-stage threshold support is the first condition to inspect."
        ),
    }


def _threshold_lookup(thresholds: pd.DataFrame, gate_name: str, key_columns: list[str]) -> dict[tuple[Any, ...], float]:
    subset = thresholds[thresholds["gate_name"] == gate_name]
    lookup: dict[tuple[Any, ...], float] = {}
    for _, row in subset.iterrows():
        values = []
        for column in key_columns:
            value = row[column]
            values.append(float(value) if column == "FE_ratio" else str(value))
        lookup[tuple(values)] = float(row["threshold_value"])
    return lookup


def _score_stats(values: pd.Series, prefix: str) -> dict[str, Any]:
    return _series_stats(values.astype(float), prefix)


def _series_stats(values: pd.Series, prefix: str) -> dict[str, Any]:
    clean = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_q10": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_q90": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_count": int(len(clean)),
        f"{prefix}_mean": float(clean.mean()),
        f"{prefix}_std": float(clean.std(ddof=0)),
        f"{prefix}_min": float(clean.min()),
        f"{prefix}_q10": float(clean.quantile(0.10)),
        f"{prefix}_q25": float(clean.quantile(0.25)),
        f"{prefix}_median": float(clean.median()),
        f"{prefix}_q75": float(clean.quantile(0.75)),
        f"{prefix}_q90": float(clean.quantile(0.90)),
        f"{prefix}_max": float(clean.max()),
    }


def _utility_stats(frame: pd.DataFrame, target_column: str, prefix: str) -> dict[str, Any]:
    if frame.empty:
        return {
            f"{prefix}_rows": 0,
            f"{prefix}_utility_gt_zero_rows": 0,
            f"{prefix}_utility_gt_zero_rate": 0.0,
            f"{prefix}_utility_mean": np.nan,
            f"{prefix}_utility_sum": 0.0,
            f"{prefix}_positive_utility_sum": 0.0,
        }
    utility = frame[target_column].to_numpy(dtype=float)
    positive = utility > 0.0
    return {
        f"{prefix}_rows": int(len(frame)),
        f"{prefix}_utility_gt_zero_rows": int(np.sum(positive)),
        f"{prefix}_utility_gt_zero_rate": float(np.mean(positive)),
        f"{prefix}_utility_mean": float(np.mean(utility)),
        f"{prefix}_utility_sum": float(np.sum(utility)),
        f"{prefix}_positive_utility_sum": float(np.sum(utility[positive])),
    }


def _policy_metrics(frame: pd.DataFrame, threshold: float, target_column: str, prefix: str) -> dict[str, Any]:
    if frame.empty or not np.isfinite(threshold):
        return {
            f"{prefix}_threshold_available": bool(np.isfinite(threshold)),
            f"{prefix}_call_count": 0,
            f"{prefix}_call_rate": 0.0,
            f"{prefix}_positive_row_capture_rate": 0.0,
            f"{prefix}_utility_capture_rate": 0.0,
            f"{prefix}_captured_positive_utility_sum": 0.0,
            f"{prefix}_unhelpful_call_count": 0,
            f"{prefix}_unhelpful_call_cost_sum": 0.0,
            f"{prefix}_utility_sum": 0.0,
            f"{prefix}_utility_mean": 0.0,
        }
    score = frame["decision_score"].to_numpy(dtype=float)
    utility = frame[target_column].to_numpy(dtype=float)
    run = score > float(threshold)
    positive = utility > 0.0
    captured_positive_sum = float(np.sum(utility[run & positive]))
    positive_utility_sum = float(np.sum(utility[positive]))
    unhelpful = utility[run & (~positive)]
    policy_utility = np.where(run, utility, 0.0)
    return {
        f"{prefix}_threshold_available": True,
        f"{prefix}_call_count": int(np.sum(run)),
        f"{prefix}_call_rate": float(np.mean(run)),
        f"{prefix}_positive_row_capture_rate": float(np.mean(run[positive])) if np.any(positive) else 0.0,
        f"{prefix}_utility_capture_rate": captured_positive_sum / positive_utility_sum if positive_utility_sum > 0.0 else 0.0,
        f"{prefix}_captured_positive_utility_sum": captured_positive_sum,
        f"{prefix}_unhelpful_call_count": int(np.sum(run & (~positive))),
        f"{prefix}_unhelpful_call_cost_sum": float(-np.sum(unhelpful)),
        f"{prefix}_utility_sum": float(np.sum(policy_utility)),
        f"{prefix}_utility_mean": float(np.mean(policy_utility)),
    }


def _threshold_quantile_fields(
    frame: pd.DataFrame,
    train_threshold: float,
    desc_threshold: float,
    stage_threshold: float,
) -> dict[str, Any]:
    score = frame["decision_score"].to_numpy(dtype=float)
    return {
        "train_family_stage_threshold_validation_score_quantile": _threshold_quantile(score, train_threshold),
        "validation_descriptive_threshold_validation_score_quantile": _threshold_quantile(score, desc_threshold),
        "stage_train_threshold_validation_score_quantile": _threshold_quantile(score, stage_threshold),
        "train_minus_validation_descriptive_threshold": float(train_threshold - desc_threshold)
        if np.isfinite(train_threshold) and np.isfinite(desc_threshold)
        else np.nan,
        "stage_train_minus_validation_descriptive_threshold": float(stage_threshold - desc_threshold)
        if np.isfinite(stage_threshold) and np.isfinite(desc_threshold)
        else np.nan,
    }


def _threshold_quantile(score: np.ndarray, threshold: float) -> float:
    if len(score) == 0 or not np.isfinite(threshold):
        return np.nan
    return float(np.mean(score <= float(threshold)))


def _failure_reason(row: dict[str, Any]) -> dict[str, Any]:
    opportunity_gap = (
        float(row["validation_descriptive_threshold_utility_sum"])
        - float(row["train_family_stage_threshold_utility_sum"])
    )
    has_validation_opportunity = (
        int(row["validation_utility_gt_zero_rows"]) > 0
        or float(row["validation_descriptive_threshold_utility_sum"]) > 0.0
    )
    missing_family_stage = not bool(row["train_family_stage_threshold_exists"])
    label_coverage_gap = (
        int(row.get("train_stage_utility_gt_zero_rows", 0)) == 0 and int(row["validation_utility_gt_zero_rows"]) > 0
    )
    threshold_over_conservative = (
        bool(row.get("stage_train_threshold_threshold_available", False))
        and float(row["stage_train_threshold_call_rate"]) <= 0.25 * max(float(row["validation_descriptive_threshold_call_rate"]), 1e-12)
        and float(row.get("stage_train_minus_validation_descriptive_threshold", 0.0)) > 0.0
        and int(row.get("train_stage_utility_gt_zero_rows", 0)) > 0
    )
    score_shift = (
        bool(row.get("stage_train_threshold_threshold_available", False))
        and float(row.get("stage_train_threshold_validation_score_quantile", 0.0)) >= 0.90
        and float(row["validation_descriptive_threshold_call_rate"]) > 0.0
        and int(row.get("train_stage_utility_gt_zero_rows", 0)) > 0
    )
    if not has_validation_opportunity:
        primary = "no_validation_descriptive_opportunity"
    elif missing_family_stage:
        primary = "missing_train_family_stage_threshold_due_to_family_split"
    elif label_coverage_gap:
        primary = "train_stage_label_coverage_gap"
    elif score_shift:
        primary = "score_distribution_shift_against_train_threshold"
    elif threshold_over_conservative:
        primary = "train_threshold_objective_too_conservative"
    else:
        primary = "mixed_or_small_sample_transfer_gap"
    return {
        "opportunity_gap_validation_desc_minus_train_family_stage": opportunity_gap,
        "has_validation_descriptive_opportunity": bool(has_validation_opportunity),
        "missing_train_family_stage_threshold_flag": bool(missing_family_stage),
        "train_stage_label_coverage_gap_flag": bool(label_coverage_gap),
        "score_distribution_shift_flag": bool(score_shift),
        "train_threshold_objective_too_conservative_flag": bool(threshold_over_conservative),
        "primary_failure_reason": primary,
    }


def _safe_median(values: pd.Series) -> float:
    clean = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.median()) if len(clean) else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose family-stage threshold transfer in min-support results.")
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
        "--gated-thresholds",
        type=Path,
        default=Path("results/decision/min_support/gated_pareto_diagnostic/gated_pareto_thresholds.parquet"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/family_stage_threshold_transfer"),
    )
    args = parser.parse_args()
    run_family_stage_threshold_transfer_diagnostic(
        validation_predictions_path=args.validation_predictions,
        train_predictions_path=args.train_predictions,
        gated_thresholds_path=args.gated_thresholds,
        output_dir=args.output_dir,
        target_column=args.target_column,
    )


if __name__ == "__main__":
    main()
