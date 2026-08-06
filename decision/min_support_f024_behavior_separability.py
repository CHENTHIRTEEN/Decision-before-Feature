from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from behavior.features import BEHAVIOR_FEATURE_COLUMNS
from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _check_target, _json_default


MODEL_NAMES = ("random_forest_regression", "logistic_regression_classifier")
TRAINING_DATASETS = ("original_min_support_train", "late_stage_extended_train")
DOMAINS = ("target_problem_changed_late", "target_holdout_changed_late")
GROUP_LAYERS = {
    "overall": [],
    "problem_id": ["problem_id"],
    "dimension": ["dimension"],
}
TOP_K_FRACTIONS = (0.05, 0.10, 0.20)


def run_f024_behavior_separability(
    *,
    validation_labels_path: Path,
    extension_train_labels_path: Path,
    predictions_path: Path,
    extension_config_path: Path,
    output_dir: Path,
    target_column: str,
) -> dict[str, Any]:
    _check_target(target_column)
    config = _load_config(extension_config_path)
    target_problem_ids = _target_problem_ids(config)
    target_holdout_pairs = _target_holdout_pairs(config)

    validation = _annotate(
        pd.read_parquet(validation_labels_path),
        target_problem_ids=target_problem_ids,
        target_holdout_pairs=target_holdout_pairs,
        data_split="validation",
    )
    extension_train = _annotate(
        pd.read_parquet(extension_train_labels_path),
        target_problem_ids=target_problem_ids,
        target_holdout_pairs=target_holdout_pairs,
        data_split="extension_train",
    )
    predictions = pd.read_parquet(predictions_path)

    target_validation = _target_frame(validation, holdout_only=False)
    target_holdout = _target_frame(validation, holdout_only=True)
    target_extension = _target_frame(extension_train, holdout_only=False)
    score_rows = _score_rows(
        predictions,
        validation,
        target_problem_ids,
        target_holdout_pairs,
        target_column,
    )

    validation_target_fe050 = target_validation[target_validation["FE_ratio"].round(6) == 0.5].copy()
    validation_target_holdout_fe050 = target_holdout[target_holdout["FE_ratio"].round(6) == 0.5].copy()
    extension_target_fe050 = target_extension[target_extension["FE_ratio"].round(6) == 0.5].copy()

    distribution_overlap = _distribution_overlap_table(
        frames={
            "validation_target": target_validation,
            "validation_target_holdout": target_holdout,
            "validation_target_fe050": validation_target_fe050,
            "validation_target_holdout_fe050": validation_target_holdout_fe050,
            "extension_train_target": target_extension,
            "extension_train_target_fe050": extension_target_fe050,
        },
        target_column=target_column,
    )
    single_feature_thresholds = _single_feature_threshold_table(
        frames={
            "validation_target": target_validation,
            "validation_target_holdout": target_holdout,
            "validation_target_fe050": validation_target_fe050,
            "validation_target_holdout_fe050": validation_target_holdout_fe050,
            "extension_train_target": target_extension,
            "extension_train_target_fe050": extension_target_fe050,
        },
        target_column=target_column,
    )
    score_rank_summary = _score_rank_summary(score_rows, target_column)
    score_failure_samples = _score_failure_samples(score_rows, target_column)

    output_dir.mkdir(parents=True, exist_ok=True)
    distribution_path = output_dir / "f024_behavior_distribution_overlap.parquet"
    thresholds_path = output_dir / "f024_single_feature_threshold_separability.parquet"
    score_rank_path = output_dir / "f024_score_rank_summary.parquet"
    failures_path = output_dir / "f024_score_ranking_failure_samples.parquet"
    summary_path = output_dir / "f024_behavior_separability_summary.json"

    pq.write_table(pa.Table.from_pandas(distribution_overlap, preserve_index=False), distribution_path)
    pq.write_table(pa.Table.from_pandas(single_feature_thresholds, preserve_index=False), thresholds_path)
    pq.write_table(pa.Table.from_pandas(score_rank_summary, preserve_index=False), score_rank_path)
    pq.write_table(pa.Table.from_pandas(score_failure_samples, preserve_index=False), failures_path)

    summary = {
        "experiment": "min_support_f024_behavior_separability_diagnostics",
        "research_question": (
            "Can existing behavior features separate f024 target changed_algorithm late-stage rows "
            "where U_ELA > 0 from rows where U_ELA <= 0?"
        ),
        "target_column": target_column,
        "inputs": {
            "validation_labels": str(validation_labels_path),
            "extension_train_labels": str(extension_train_labels_path),
            "predictions": str(predictions_path),
            "extension_config": str(extension_config_path),
        },
        "rows": {
            "validation_target_problem_changed_late": int(len(target_validation)),
            "validation_target_holdout_changed_late": int(len(target_holdout)),
            "validation_target_holdout_changed_late_fe050": int(len(validation_target_holdout_fe050)),
            "extension_train_target_changed_late": int(len(target_extension)),
            "validation_target_problem_utility_gt_zero_rows": int((target_validation[target_column] > 0.0).sum()),
            "validation_target_holdout_utility_gt_zero_rows": int((target_holdout[target_column] > 0.0).sum()),
            "validation_target_holdout_fe050_utility_gt_zero_rows": int(
                (validation_target_holdout_fe050[target_column] > 0.0).sum()
            ),
            "extension_train_target_utility_gt_zero_rows": int((target_extension[target_column] > 0.0).sum()),
        },
        "target_problem_ids": sorted(target_problem_ids),
        "model_names": list(MODEL_NAMES),
        "training_datasets": list(TRAINING_DATASETS),
        "feature_columns": list(BEHAVIOR_FEATURE_COLUMNS),
        "excluded_from_decision_input": [
            "function_id",
            "family",
            "problem_id",
            "dimension",
            "prefix_algorithm",
            "algorithm_parameters",
            "ELA feature columns",
            "selected_algorithm",
            "default_algorithm",
            "label_source",
            "target flags",
        ],
        "interpretation": _interpretation(
            single_feature_thresholds=single_feature_thresholds,
            score_rank_summary=score_rank_summary,
        ),
        "data_leakage_check": {
            "uses_existing_predictions_only": True,
            "formal_models_retrained": False,
            "decision_input_uses_only_behavior_features": True,
            "ela_features_used_as_decision_input": False,
            "metadata_used_only_for_grouping": True,
            "original_utility_labels_modified": False,
            "formal_phase1_configs_modified": False,
            "threshold_selected_from_validation_for_formal_model": False,
        },
        "outputs": {
            "distribution_overlap": str(distribution_path),
            "single_feature_thresholds": str(thresholds_path),
            "score_rank_summary": str(score_rank_path),
            "score_failure_samples": str(failures_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote f024 behavior distribution overlap to {distribution_path}")
    print(f"wrote f024 single-feature threshold separability to {thresholds_path}")
    print(f"wrote f024 score rank summary to {score_rank_path}")
    print(f"wrote f024 score ranking failure samples to {failures_path}")
    print(f"wrote f024 behavior separability summary to {summary_path}")
    return summary


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config


def _target_problem_ids(config: dict[str, Any]) -> set[str]:
    return {str(row["problem_id"]) for row in config.get("target_problem_seed_rows", [])}


def _target_holdout_pairs(config: dict[str, Any]) -> set[tuple[str, int]]:
    pairs = set()
    for row in config.get("target_problem_seed_rows", []):
        problem_id = str(row["problem_id"])
        for seed in row.get("existing_holdout_validation_seeds", []):
            pairs.add((problem_id, int(seed)))
    return pairs


def _annotate(
    frame: pd.DataFrame,
    *,
    target_problem_ids: set[str],
    target_holdout_pairs: set[tuple[str, int]],
    data_split: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["analysis_data_split"] = data_split
    result["label_source"] = np.where(
        result["selected_algorithm"].astype(str) == result["default_algorithm"].astype(str),
        "same_algorithm",
        "changed_algorithm",
    )
    result["stage"] = np.where(result["FE_ratio"].astype(float) >= 0.5, "late_stage", "not_late_stage")
    result["is_target_problem"] = result["problem_id"].astype(str).isin(target_problem_ids)
    pairs = list(zip(result["problem_id"].astype(str), result["seed"].astype(int), strict=False))
    result["is_target_holdout_seed"] = [pair in target_holdout_pairs for pair in pairs]
    return result


def _target_frame(frame: pd.DataFrame, *, holdout_only: bool) -> pd.DataFrame:
    mask = (
        frame["is_target_problem"].to_numpy(dtype=bool)
        & (frame["label_source"].to_numpy() == "changed_algorithm")
        & (frame["FE_ratio"].to_numpy(dtype=float) >= 0.5)
    )
    if holdout_only:
        mask &= frame["is_target_holdout_seed"].to_numpy(dtype=bool)
    return frame.loc[mask].copy()


def _score_rows(
    predictions: pd.DataFrame,
    validation: pd.DataFrame,
    target_problem_ids: set[str],
    target_holdout_pairs: set[tuple[str, int]],
    target_column: str,
) -> pd.DataFrame:
    frame = predictions[
        (predictions["threshold_mode"] == "zero")
        & (predictions["model_name"].isin(MODEL_NAMES))
        & (predictions["training_dataset"].isin(TRAINING_DATASETS))
    ].copy()
    frame["is_target_problem"] = frame["problem_id"].astype(str).isin(target_problem_ids)
    pairs = list(zip(frame["problem_id"].astype(str), frame["seed"].astype(int), strict=False))
    frame["is_target_holdout_seed"] = [pair in target_holdout_pairs for pair in pairs]
    frame = frame[
        (frame["is_target_problem"])
        & (frame["label_source"] == "changed_algorithm")
        & (frame["FE_ratio"].astype(float) >= 0.5)
    ].copy()
    frame["utility_gt_zero"] = frame[target_column].astype(float) > 0.0
    key_columns = ["split", "problem_id", "dimension", "prefix_algorithm", "seed", "FE", "FE_ratio"]
    feature_frame = validation[key_columns + list(BEHAVIOR_FEATURE_COLUMNS)].copy()
    return frame.merge(feature_frame, on=key_columns, how="left", validate="many_to_one")


def _distribution_overlap_table(frames: dict[str, pd.DataFrame], target_column: str) -> pd.DataFrame:
    rows = []
    for data_split, frame in frames.items():
        for layer, group_columns in GROUP_LAYERS.items():
            grouped = [((), frame)] if not group_columns else frame.groupby(group_columns, dropna=False)
            for group_values, subset in grouped:
                group = _group_dict(group_columns, group_values)
                for feature in BEHAVIOR_FEATURE_COLUMNS:
                    rows.append(
                        _distribution_overlap_row(
                            data_split=data_split,
                            layer=layer,
                            group=group,
                            frame=subset,
                            feature=feature,
                            target_column=target_column,
                        )
                    )
    return pd.DataFrame(rows)


def _distribution_overlap_row(
    *,
    data_split: str,
    layer: str,
    group: dict[str, Any],
    frame: pd.DataFrame,
    feature: str,
    target_column: str,
) -> dict[str, Any]:
    positive = _finite_values(frame.loc[frame[target_column] > 0.0, feature])
    non_positive = _finite_values(frame.loc[frame[target_column] <= 0.0, feature])
    pos_stats = _stats(positive)
    neg_stats = _stats(non_positive)
    auc = _rank_auc(positive, non_positive)
    return {
        "data_split": data_split,
        "layer": layer,
        "group": _group_label(group),
        "problem_id": group.get("problem_id"),
        "dimension": group.get("dimension"),
        "feature": feature,
        "rows": int(len(frame)),
        "positive_rows": int((frame[target_column] > 0.0).sum()),
        "non_positive_rows": int((frame[target_column] <= 0.0).sum()),
        "positive_finite_rows": int(len(positive)),
        "non_positive_finite_rows": int(len(non_positive)),
        **{f"positive_{key}": value for key, value in pos_stats.items()},
        **{f"non_positive_{key}": value for key, value in neg_stats.items()},
        "iqr_overlap_ratio": _interval_overlap_ratio(
            pos_stats.get("q25"),
            pos_stats.get("q75"),
            neg_stats.get("q25"),
            neg_stats.get("q75"),
        ),
        "range_overlap_ratio": _interval_overlap_ratio(
            pos_stats.get("min"),
            pos_stats.get("max"),
            neg_stats.get("min"),
            neg_stats.get("max"),
        ),
        "rank_auc_positive_greater": auc,
        "rank_separation": None if auc is None else float(abs(auc - 0.5) * 2.0),
        "median_gap_abs": _abs_gap(pos_stats.get("median"), neg_stats.get("median")),
    }


def _single_feature_threshold_table(frames: dict[str, pd.DataFrame], target_column: str) -> pd.DataFrame:
    rows = []
    for data_split, frame in frames.items():
        for layer, group_columns in GROUP_LAYERS.items():
            grouped = [((), frame)] if not group_columns else frame.groupby(group_columns, dropna=False)
            for group_values, subset in grouped:
                group = _group_dict(group_columns, group_values)
                for feature in BEHAVIOR_FEATURE_COLUMNS:
                    rows.append(
                        _best_single_feature_threshold(
                            data_split=data_split,
                            layer=layer,
                            group=group,
                            frame=subset,
                            feature=feature,
                            target_column=target_column,
                        )
                    )
    return pd.DataFrame(rows)


def _best_single_feature_threshold(
    *,
    data_split: str,
    layer: str,
    group: dict[str, Any],
    frame: pd.DataFrame,
    feature: str,
    target_column: str,
) -> dict[str, Any]:
    finite = frame[np.isfinite(frame[feature].to_numpy(dtype=float))].copy()
    values = finite[feature].to_numpy(dtype=float)
    observed = finite[target_column].to_numpy(dtype=float)
    labels = observed > 0.0
    base = {
        "data_split": data_split,
        "layer": layer,
        "group": _group_label(group),
        "problem_id": group.get("problem_id"),
        "dimension": group.get("dimension"),
        "feature": feature,
        "rows": int(len(frame)),
        "finite_rows": int(len(finite)),
        "positive_rows": int(np.sum(frame[target_column].to_numpy(dtype=float) > 0.0)),
        "finite_positive_rows": int(np.sum(labels)) if len(finite) else 0,
    }
    if len(finite) == 0 or len(np.unique(labels)) < 2:
        return {**base, **_empty_threshold_metrics()}

    thresholds = _candidate_thresholds(values)
    best = None
    for direction in ("greater", "less_equal"):
        for threshold in thresholds:
            calls = values > threshold if direction == "greater" else values <= threshold
            metrics = _classification_metrics(calls, labels, observed)
            candidate = {
                "direction": direction,
                "threshold": float(threshold),
                **metrics,
            }
            if best is None or _threshold_sort_key(candidate) > _threshold_sort_key(best):
                best = candidate
    return {**base, **best}


def _score_rank_summary(score_rows: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows = []
    for domain, domain_frame in {
        "target_problem_changed_late": score_rows,
        "target_holdout_changed_late": score_rows[score_rows["is_target_holdout_seed"]],
        "target_problem_changed_late_fe050": score_rows[score_rows["FE_ratio"].round(6) == 0.5],
        "target_holdout_changed_late_fe050": score_rows[
            (score_rows["is_target_holdout_seed"]) & (score_rows["FE_ratio"].round(6) == 0.5)
        ],
    }.items():
        for (training_dataset, model_name), subset in domain_frame.groupby(["training_dataset", "model_name"]):
            ranked = _with_descending_rank(subset)
            observed = ranked[target_column].to_numpy(dtype=float)
            positive = observed > 0.0
            row = {
                "eval_domain": domain,
                "training_dataset": training_dataset,
                "model_name": model_name,
                "rows": int(len(ranked)),
                "positive_rows": int(np.sum(positive)),
                "score_mean": float(ranked["decision_score"].mean()),
                "score_median": float(ranked["decision_score"].median()),
                "positive_score_mean": _mean_or_none(ranked.loc[positive, "decision_score"].to_numpy(dtype=float)),
                "positive_score_median": _median_or_none(ranked.loc[positive, "decision_score"].to_numpy(dtype=float)),
                "non_positive_score_mean": _mean_or_none(ranked.loc[~positive, "decision_score"].to_numpy(dtype=float)),
                "non_positive_score_median": _median_or_none(ranked.loc[~positive, "decision_score"].to_numpy(dtype=float)),
                "positive_mean_score_percentile": _mean_or_none(
                    ranked.loc[positive, "score_percentile_desc"].to_numpy(dtype=float)
                ),
                "positive_median_rank_desc": _median_or_none(ranked.loc[positive, "score_rank_desc"].to_numpy(dtype=float)),
                "zero_threshold_call_rows": int((ranked["decision_score"] > 0.0).sum()),
                "zero_threshold_positive_capture_rows": int(((ranked["decision_score"] > 0.0) & positive).sum()),
                "zero_threshold_unhelpful_call_rows": int(((ranked["decision_score"] > 0.0) & ~positive).sum()),
            }
            for fraction in TOP_K_FRACTIONS:
                k = max(1, int(np.ceil(len(ranked) * fraction)))
                top = ranked["score_rank_desc"].to_numpy(dtype=float) <= k
                row[f"top_{int(fraction * 100)}pct_k"] = int(k)
                row[f"top_{int(fraction * 100)}pct_positive_capture_rows"] = int(np.sum(top & positive))
                row[f"top_{int(fraction * 100)}pct_positive_capture_rate"] = (
                    float(np.mean(top[positive])) if np.any(positive) else 0.0
                )
                row[f"top_{int(fraction * 100)}pct_unhelpful_rows"] = int(np.sum(top & ~positive))
            rows.append(row)
    return pd.DataFrame(rows)


def _score_failure_samples(score_rows: pd.DataFrame, target_column: str) -> pd.DataFrame:
    frames = []
    for domain, domain_frame in {
        "target_problem_changed_late": score_rows,
        "target_holdout_changed_late": score_rows[score_rows["is_target_holdout_seed"]],
        "target_problem_changed_late_fe050": score_rows[score_rows["FE_ratio"].round(6) == 0.5],
        "target_holdout_changed_late_fe050": score_rows[
            (score_rows["is_target_holdout_seed"]) & (score_rows["FE_ratio"].round(6) == 0.5)
        ],
    }.items():
        for (training_dataset, model_name), subset in domain_frame.groupby(["training_dataset", "model_name"]):
            ranked = _with_descending_rank(subset)
            positive = ranked[target_column].to_numpy(dtype=float) > 0.0
            non_positive_median = float(ranked.loc[~positive, "decision_score"].median()) if np.any(~positive) else np.nan
            top20_k = max(1, int(np.ceil(len(ranked) * 0.20)))
            candidates = [
                (
                    "missed_positive_below_non_positive_median",
                    ranked[positive & (ranked["decision_score"] <= non_positive_median)],
                ),
                (
                    "missed_positive_not_top20pct",
                    ranked[positive & (ranked["score_rank_desc"] > top20_k)],
                ),
                (
                    "top20pct_unhelpful_high_score",
                    ranked[(~positive) & (ranked["score_rank_desc"] <= top20_k)],
                ),
                (
                    "zero_threshold_unhelpful_call",
                    ranked[(~positive) & (ranked["decision_score"] > 0.0)],
                ),
            ]
            for failure_role, failure_frame in candidates:
                selected = failure_frame.sort_values("decision_score", ascending=failure_role.startswith("missed")).head(30)
                if selected.empty:
                    continue
                selected = selected.copy()
                selected.insert(0, "failure_role", failure_role)
                selected.insert(1, "eval_domain", domain)
                frames.append(selected)
    if not frames:
        return pd.DataFrame()
    columns = [
        "failure_role",
        "eval_domain",
        "training_dataset",
        "model_name",
        "problem_id",
        "dimension",
        "prefix_algorithm",
        "seed",
        "FE",
        "FE_ratio",
        target_column,
        "decision_score",
        "score_rank_desc",
        "score_percentile_desc",
        "decision_run_ela",
        "is_target_holdout_seed",
        *BEHAVIOR_FEATURE_COLUMNS,
    ]
    return pd.concat(frames, ignore_index=True)[columns]


def _with_descending_rank(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values("decision_score", ascending=False).copy()
    result["score_rank_desc"] = np.arange(1, len(result) + 1)
    if len(result) == 1:
        result["score_percentile_desc"] = 1.0
    else:
        result["score_percentile_desc"] = 1.0 - (result["score_rank_desc"] - 1.0) / (len(result) - 1.0)
    return result


def _group_dict(group_columns: list[str], group_values: Any) -> dict[str, Any]:
    if not group_columns:
        return {}
    if not isinstance(group_values, tuple):
        group_values = (group_values,)
    return dict(zip(group_columns, group_values, strict=True))


def _group_label(group: dict[str, Any]) -> str:
    if not group:
        return "overall"
    return "|".join(f"{key}={value}" for key, value in group.items())


def _finite_values(series: pd.Series) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _stats(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"mean": None, "median": None, "std": None, "min": None, "q25": None, "q75": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def _interval_overlap_ratio(a_low: float | None, a_high: float | None, b_low: float | None, b_high: float | None) -> float | None:
    if None in (a_low, a_high, b_low, b_high):
        return None
    a_width = max(float(a_high) - float(a_low), 0.0)
    b_width = max(float(b_high) - float(b_low), 0.0)
    denom = min(a_width, b_width)
    if denom <= 0.0:
        return None
    overlap = max(0.0, min(float(a_high), float(b_high)) - max(float(a_low), float(b_low)))
    return float(overlap / denom)


def _rank_auc(positive: np.ndarray, non_positive: np.ndarray) -> float | None:
    if positive.size == 0 or non_positive.size == 0:
        return None
    comparisons = positive[:, None] - non_positive[None, :]
    greater = np.sum(comparisons > 0.0)
    ties = np.sum(comparisons == 0.0)
    return float((greater + 0.5 * ties) / comparisons.size)


def _abs_gap(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(abs(left - right))


def _candidate_thresholds(values: np.ndarray) -> np.ndarray:
    unique = np.unique(values)
    if unique.size == 1:
        return np.array([unique[0]], dtype=float)
    mids = (unique[:-1] + unique[1:]) / 2.0
    return np.concatenate(([unique[0] - 1e-12], mids, [unique[-1] + 1e-12]))


def _classification_metrics(calls: np.ndarray, labels: np.ndarray, observed: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum(calls & labels))
    fp = int(np.sum(calls & ~labels))
    fn = int(np.sum(~calls & labels))
    tn = int(np.sum(~calls & ~labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    positive_sum = float(np.sum(observed[labels]))
    captured_sum = float(np.sum(observed[calls & labels]))
    unhelpful = observed[calls & ~labels]
    return {
        "true_run_ela_rows": tp,
        "unhelpful_call_rows": fp,
        "missed_positive_rows": fn,
        "skip_when_unhelpful_rows": tn,
        "decision_call_rows": int(np.sum(calls)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / len(labels)),
        "utility_capture_rate": captured_sum / positive_sum if positive_sum > 0.0 else 0.0,
        "captured_positive_utility_sum": captured_sum,
        "unhelpful_call_cost_sum": float(-np.sum(unhelpful)),
        "decision_mean_utility": float(np.mean(np.where(calls, observed, 0.0))),
    }


def _threshold_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["f1"]),
        float(row["decision_mean_utility"]),
        float(row["utility_capture_rate"]),
        -float(row["unhelpful_call_cost_sum"]),
    )


def _empty_threshold_metrics() -> dict[str, Any]:
    return {
        "direction": None,
        "threshold": None,
        "true_run_ela_rows": 0,
        "unhelpful_call_rows": 0,
        "missed_positive_rows": None,
        "skip_when_unhelpful_rows": None,
        "decision_call_rows": 0,
        "precision": None,
        "recall": None,
        "f1": None,
        "accuracy": None,
        "utility_capture_rate": None,
        "captured_positive_utility_sum": None,
        "unhelpful_call_cost_sum": None,
        "decision_mean_utility": None,
    }


def _mean_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def _median_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.median(values))


def _interpretation(single_feature_thresholds: pd.DataFrame, score_rank_summary: pd.DataFrame) -> dict[str, Any]:
    target_holdout_thresholds = single_feature_thresholds[
        (single_feature_thresholds["data_split"] == "validation_target_holdout_fe050")
        & (single_feature_thresholds["layer"] == "overall")
        & (single_feature_thresholds["feature"] != "bf_fe_ratio")
    ]
    best_f1 = target_holdout_thresholds["f1"].dropna().max()
    score_focus = score_rank_summary[
        (score_rank_summary["eval_domain"] == "target_holdout_changed_late_fe050")
        & (score_rank_summary["training_dataset"] == "late_stage_extended_train")
        & (score_rank_summary["model_name"].isin(MODEL_NAMES))
    ]
    rf_row = score_focus[score_focus["model_name"] == "random_forest_regression"]
    logistic_row = score_focus[score_focus["model_name"] == "logistic_regression_classifier"]
    return {
        "best_single_feature_f1_on_target_holdout": None if pd.isna(best_f1) else float(best_f1),
        "rf_top20_positive_capture_rate": _first_or_none(rf_row, "top_20pct_positive_capture_rate"),
        "logistic_top20_positive_capture_rate": _first_or_none(logistic_row, "top_20pct_positive_capture_rate"),
        "needs_algorithm_agnostic_feature_extension": bool(
            (pd.isna(best_f1) or float(best_f1) < 0.55)
            and (_first_or_none(rf_row, "top_20pct_positive_capture_rate") or 0.0) < 0.50
        ),
        "reason": (
            "Set to true when single behavior features have weak target-holdout separability "
            "and RF scores fail to rank at least half of U_ELA>0 rows into the top 20%."
        ),
    }


def _first_or_none(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty:
        return None
    value = frame.iloc[0][column]
    if pd.isna(value):
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose f024 behavior separability for min-support follow-up.")
    parser.add_argument(
        "--validation-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_validation/utility_labels.parquet"),
    )
    parser.add_argument(
        "--extension-train-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_train_late_stage_f024_followup/utility_labels_fe050.parquet"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/decision/min_support/late_stage_f024_followup/late_stage_extension_predictions.parquet"),
    )
    parser.add_argument(
        "--extension-config",
        type=Path,
        default=Path("configs/min_support_bbob_train_late_stage_f024_followup.yaml"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/f024_behavior_separability"),
    )
    args = parser.parse_args()
    run_f024_behavior_separability(
        validation_labels_path=args.validation_labels,
        extension_train_labels_path=args.extension_train_labels,
        predictions_path=args.predictions,
        extension_config_path=args.extension_config,
        output_dir=args.output_dir,
        target_column=args.target_column,
    )


if __name__ == "__main__":
    main()
