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
from decision.min_support_diagnostics import GROUP_LAYERS, _group_label
from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _check_target, _json_default, _read_labels


EVAL_DOMAIN = "changed_algorithm_validation"
EVAL_LABEL_SOURCE = "changed_algorithm"
PAIRWISE_COMPARISONS = (
    ("utility_gt_zero", "top_score_unhelpful_called"),
    ("utility_gt_zero", "low_score_missed_positive"),
    ("top_score_unhelpful_called", "low_score_missed_positive"),
)
MERGE_KEY_COLUMNS = [
    "problem_id",
    "family",
    "dimension",
    "prefix_algorithm",
    "seed",
    "FE",
    "FE_ratio",
    "default_algorithm",
    "selected_algorithm",
]


def run_behavior_overlap_diagnostics(
    *,
    predictions_path: Path,
    ranked_rows_path: Path,
    validation_labels_path: Path,
    output_dir: Path,
    target_column: str,
    top_score_quantile_min: float,
    low_score_quantile_max: float,
) -> dict[str, Any]:
    _check_target(target_column)
    _check_quantile(top_score_quantile_min, "top_score_quantile_min")
    _check_quantile(low_score_quantile_max, "low_score_quantile_max")

    predictions = pq.read_table(predictions_path).to_pandas()
    ranked_rows = pq.read_table(ranked_rows_path).to_pandas()
    labels = _read_labels(validation_labels_path)
    _check_inputs(predictions, ranked_rows, labels, target_column)

    changed = predictions[predictions["label_source"] == EVAL_LABEL_SOURCE].copy()
    labels_for_join = labels[[*MERGE_KEY_COLUMNS, *BEHAVIOR_FEATURE_COLUMNS]].copy()
    enriched = changed.merge(labels_for_join, on=MERGE_KEY_COLUMNS, how="left", validate="many_to_one")
    missing_behavior_rows = int(enriched[list(BEHAVIOR_FEATURE_COLUMNS)].isna().all(axis=1).sum())
    if missing_behavior_rows:
        raise ValueError(f"behavior features missing after merge for {missing_behavior_rows} prediction rows")

    role_count_frames = []
    feature_summary_frames = []
    pairwise_frames = []
    role_row_frames = []
    for (model_name, threshold_mode), model_frame in enriched.groupby(["model_name", "threshold_mode"], dropna=False):
        for layer, columns in GROUP_LAYERS.items():
            if not columns:
                ranked = _add_group_ranks(model_frame)
                roles = _assign_roles(
                    ranked,
                    target_column=target_column,
                    top_score_quantile_min=top_score_quantile_min,
                    low_score_quantile_max=low_score_quantile_max,
                )
                role_count_frames.append(_role_count_row(roles, str(model_name), str(threshold_mode), layer, {}))
                feature_summary_frames.append(
                    _feature_summary(roles, str(model_name), str(threshold_mode), layer, {}, target_column)
                )
                pairwise_frames.append(_pairwise_overlap(roles, str(model_name), str(threshold_mode), layer, {}))
                role_row_frames.append(_role_rows(roles, str(model_name), str(threshold_mode), layer, {}, target_column))
                continue

            for group_values, subset in model_frame.groupby(columns, dropna=False):
                if not isinstance(group_values, tuple):
                    group_values = (group_values,)
                group = dict(zip(columns, group_values, strict=True))
                ranked = _add_group_ranks(subset)
                roles = _assign_roles(
                    ranked,
                    target_column=target_column,
                    top_score_quantile_min=top_score_quantile_min,
                    low_score_quantile_max=low_score_quantile_max,
                )
                role_count_frames.append(_role_count_row(roles, str(model_name), str(threshold_mode), layer, group))
                feature_summary_frames.append(
                    _feature_summary(roles, str(model_name), str(threshold_mode), layer, group, target_column)
                )
                pairwise_frames.append(_pairwise_overlap(roles, str(model_name), str(threshold_mode), layer, group))
                role_row_frames.append(_role_rows(roles, str(model_name), str(threshold_mode), layer, group, target_column))

    role_counts = pd.DataFrame(role_count_frames)
    feature_summary = pd.concat(feature_summary_frames, ignore_index=True)
    pairwise_overlap = pd.concat(pairwise_frames, ignore_index=True)
    role_rows = pd.concat(role_row_frames, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    role_counts_path = output_dir / "behavior_overlap_role_counts.parquet"
    feature_summary_path = output_dir / "behavior_overlap_feature_summary.parquet"
    pairwise_path = output_dir / "behavior_overlap_pairwise.parquet"
    role_rows_path = output_dir / "behavior_overlap_role_rows.parquet"
    summary_path = output_dir / "behavior_overlap_summary.json"
    pq.write_table(pa.Table.from_pandas(role_counts, preserve_index=False), role_counts_path)
    pq.write_table(pa.Table.from_pandas(feature_summary, preserve_index=False), feature_summary_path)
    pq.write_table(pa.Table.from_pandas(pairwise_overlap, preserve_index=False), pairwise_path)
    pq.write_table(pa.Table.from_pandas(role_rows, preserve_index=False), role_rows_path)

    summary = {
        "experiment": "min_support_behavior_overlap_diagnostics",
        "predictions": str(predictions_path),
        "ranked_rows": str(ranked_rows_path),
        "validation_labels": str(validation_labels_path),
        "target_column": target_column,
        "eval_domain": EVAL_DOMAIN,
        "eval_label_source": EVAL_LABEL_SOURCE,
        "role_definitions": {
            "utility_gt_zero": f"{target_column} > 0 within changed_algorithm validation rows",
            "top_score_unhelpful_called": (
                f"{target_column} <= 0, decision_run_ela is true, and group score_quantile >= "
                f"{top_score_quantile_min}"
            ),
            "low_score_missed_positive": (
                f"{target_column} > 0, decision_run_ela is false, and group score_quantile <= "
                f"{low_score_quantile_max}"
            ),
        },
        "ranking_scope": "score ranks and quantiles are recomputed within each model, threshold mode, and layer group",
        "behavior_features": list(BEHAVIOR_FEATURE_COLUMNS),
        "rows": {
            "prediction_rows": int(len(predictions)),
            "ranked_rows_read": int(len(ranked_rows)),
            "changed_algorithm_prediction_rows": int(len(changed)),
            "role_count_rows": int(len(role_counts)),
            "feature_summary_rows": int(len(feature_summary)),
            "pairwise_overlap_rows": int(len(pairwise_overlap)),
            "role_rows": int(len(role_rows)),
        },
        "outputs": {
            "role_counts": str(role_counts_path),
            "feature_summary": str(feature_summary_path),
            "pairwise_overlap": str(pairwise_path),
            "role_rows": str(role_rows_path),
            "summary": str(summary_path),
        },
        "data_leakage_check": {
            "uses_existing_model_sensitivity_predictions": True,
            "uses_existing_model_score_ranking_output": True,
            "behavior_features_used_for_diagnostics_only": True,
            "ela_features_used_as_decision_input": False,
            "original_utility_labels_modified": False,
        },
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote behavior overlap role counts to {role_counts_path}")
    print(f"wrote behavior overlap feature summary to {feature_summary_path}")
    print(f"wrote behavior overlap pairwise summary to {pairwise_path}")
    print(f"wrote behavior overlap role rows to {role_rows_path}")
    print(f"wrote behavior overlap summary to {summary_path}")
    return summary


def _check_quantile(value: float, name: str) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _check_inputs(
    predictions: pd.DataFrame,
    ranked_rows: pd.DataFrame,
    labels: pd.DataFrame,
    target_column: str,
) -> None:
    prediction_required = {
        "model_name",
        "model_family",
        "score_semantics",
        "threshold_mode",
        "decision_score",
        "decision_run_ela",
        "decision_threshold",
        "label_source",
        target_column,
        *MERGE_KEY_COLUMNS,
    }
    ranked_required = {
        "model_name_for_rank",
        "threshold_mode_for_rank",
        "eval_domain",
        "row_role",
        "score_rank",
        "score_quantile",
        target_column,
    }
    label_required = {target_column, *MERGE_KEY_COLUMNS, *BEHAVIOR_FEATURE_COLUMNS}
    missing_prediction = sorted(prediction_required.difference(predictions.columns))
    missing_ranked = sorted(ranked_required.difference(ranked_rows.columns))
    missing_label = sorted(label_required.difference(labels.columns))
    if missing_prediction:
        raise ValueError(f"model sensitivity predictions missing columns: {missing_prediction}")
    if missing_ranked:
        raise ValueError(f"model score ranking rows missing columns: {missing_ranked}")
    if missing_label:
        raise ValueError(f"validation labels missing columns: {missing_label}")


def _add_group_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["group_score_rank"] = ranked["decision_score"].rank(method="min", ascending=False).astype(float)
    if len(ranked) <= 1:
        ranked["group_score_quantile"] = 1.0
    else:
        ranked["group_score_quantile"] = 1.0 - ((ranked["group_score_rank"] - 1.0) / float(len(ranked) - 1))
    return ranked


def _assign_roles(
    frame: pd.DataFrame,
    *,
    target_column: str,
    top_score_quantile_min: float,
    low_score_quantile_max: float,
) -> pd.DataFrame:
    result = frame.copy()
    observed = result[target_column].to_numpy(dtype=float)
    decision_run = result["decision_run_ela"].to_numpy(dtype=bool)
    quantile = result["group_score_quantile"].to_numpy(dtype=float)
    result["role_utility_gt_zero"] = observed > 0.0
    result["role_top_score_unhelpful_called"] = (observed <= 0.0) & decision_run & (quantile >= top_score_quantile_min)
    result["role_low_score_missed_positive"] = (observed > 0.0) & (~decision_run) & (quantile <= low_score_quantile_max)
    return result


def _role_count_row(
    frame: pd.DataFrame,
    model_name: str,
    threshold_mode: str,
    layer: str,
    group: dict[str, Any],
) -> dict[str, Any]:
    return {
        **_common_fields(frame, model_name, threshold_mode, layer, group),
        "rows": int(len(frame)),
        "utility_gt_zero_rows": int(frame["role_utility_gt_zero"].sum()),
        "top_score_unhelpful_called_rows": int(frame["role_top_score_unhelpful_called"].sum()),
        "low_score_missed_positive_rows": int(frame["role_low_score_missed_positive"].sum()),
        "decision_ela_call_rows": int(frame["decision_run_ela"].sum()),
        "score_quantile_min": float(frame["group_score_quantile"].min()),
        "score_quantile_median": float(frame["group_score_quantile"].median()),
        "score_quantile_max": float(frame["group_score_quantile"].max()),
    }


def _feature_summary(
    frame: pd.DataFrame,
    model_name: str,
    threshold_mode: str,
    layer: str,
    group: dict[str, Any],
    target_column: str,
) -> pd.DataFrame:
    rows = []
    role_masks = _role_masks(frame)
    for role_name, mask in role_masks.items():
        role_frame = frame[mask]
        for feature in BEHAVIOR_FEATURE_COLUMNS:
            values = role_frame[feature].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    **_common_fields(frame, model_name, threshold_mode, layer, group),
                    "role": role_name,
                    "feature": feature,
                    "rows": int(len(role_frame)),
                    "non_missing_rows": int(values.size),
                    "target_mean": _mean_or_none(role_frame[target_column].dropna().to_numpy(dtype=float)),
                    "score_quantile_median": _median_or_none(
                        role_frame["group_score_quantile"].dropna().to_numpy(dtype=float)
                    ),
                    **_distribution_stats(values),
                }
            )
    return pd.DataFrame(rows)


def _pairwise_overlap(
    frame: pd.DataFrame,
    model_name: str,
    threshold_mode: str,
    layer: str,
    group: dict[str, Any],
) -> pd.DataFrame:
    role_masks = _role_masks(frame)
    rows = []
    for left_role, right_role in PAIRWISE_COMPARISONS:
        left_frame = frame[role_masks[left_role]]
        right_frame = frame[role_masks[right_role]]
        for feature in BEHAVIOR_FEATURE_COLUMNS:
            left = left_frame[feature].dropna().to_numpy(dtype=float)
            right = right_frame[feature].dropna().to_numpy(dtype=float)
            separability = _rank_separability(left, right)
            rows.append(
                {
                    **_common_fields(frame, model_name, threshold_mode, layer, group),
                    "left_role": left_role,
                    "right_role": right_role,
                    "feature": feature,
                    "left_rows": int(len(left_frame)),
                    "right_rows": int(len(right_frame)),
                    "left_non_missing_rows": int(left.size),
                    "right_non_missing_rows": int(right.size),
                    "left_median": _median_or_none(left),
                    "right_median": _median_or_none(right),
                    "median_difference_left_minus_right": (
                        float(np.median(left) - np.median(right)) if left.size and right.size else None
                    ),
                    "left_mean": _mean_or_none(left),
                    "right_mean": _mean_or_none(right),
                    "mean_difference_left_minus_right": (
                        float(np.mean(left) - np.mean(right)) if left.size and right.size else None
                    ),
                    "rank_auc_left_gt_right": separability["rank_auc_left_gt_right"],
                    "separability_auc": separability["separability_auc"],
                    "separability_direction": separability["separability_direction"],
                    "iqr_overlap_fraction": _iqr_overlap_fraction(left, right),
                    "feature_distinguishability": _feature_distinguishability(
                        separability["separability_auc"],
                        _iqr_overlap_fraction(left, right),
                    ),
                }
            )
    return pd.DataFrame(rows)


def _role_rows(
    frame: pd.DataFrame,
    model_name: str,
    threshold_mode: str,
    layer: str,
    group: dict[str, Any],
    target_column: str,
) -> pd.DataFrame:
    rows = []
    for role_name, mask in _role_masks(frame).items():
        subset = frame[mask].copy()
        if subset.empty:
            continue
        subset.insert(0, "role", role_name)
        subset.insert(0, "group", _group_label(group))
        subset.insert(0, "layer", layer)
        subset.insert(0, "threshold_mode_for_overlap", threshold_mode)
        subset.insert(0, "model_name_for_overlap", model_name)
        rows.append(
            subset[
                [
                    "model_name_for_overlap",
                    "threshold_mode_for_overlap",
                    "layer",
                    "group",
                    "role",
                    "model_name",
                    "model_family",
                    "score_semantics",
                    "threshold_mode",
                    "problem_id",
                    "family",
                    "dimension",
                    "FE_ratio",
                    "prefix_algorithm",
                    "seed",
                    "FE",
                    "default_algorithm",
                    "selected_algorithm",
                    "decision_score",
                    "group_score_rank",
                    "group_score_quantile",
                    "decision_run_ela",
                    target_column,
                    *BEHAVIOR_FEATURE_COLUMNS,
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _role_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "utility_gt_zero": frame["role_utility_gt_zero"],
        "top_score_unhelpful_called": frame["role_top_score_unhelpful_called"],
        "low_score_missed_positive": frame["role_low_score_missed_positive"],
    }


def _common_fields(
    frame: pd.DataFrame,
    model_name: str,
    threshold_mode: str,
    layer: str,
    group: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model_family": str(frame["model_family"].iloc[0]),
        "score_semantics": str(frame["score_semantics"].iloc[0]),
        "eval_domain": EVAL_DOMAIN,
        "threshold_mode": threshold_mode,
        "threshold": float(frame["decision_threshold"].iloc[0]),
        "layer": layer,
        "group": _group_label(group),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "problem_id": group.get("problem_id"),
    }


def _distribution_stats(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def _rank_separability(left: np.ndarray, right: np.ndarray) -> dict[str, float | str | None]:
    if left.size == 0 or right.size == 0:
        return {
            "rank_auc_left_gt_right": None,
            "separability_auc": None,
            "separability_direction": None,
        }
    combined = np.concatenate([left, right])
    ranks = pd.Series(combined).rank(method="average").to_numpy(dtype=float)
    left_rank_sum = float(np.sum(ranks[: left.size]))
    auc = (left_rank_sum - left.size * (left.size + 1) / 2.0) / float(left.size * right.size)
    separability_auc = max(auc, 1.0 - auc)
    direction = "left_higher" if auc >= 0.5 else "right_higher"
    return {
        "rank_auc_left_gt_right": float(auc),
        "separability_auc": float(separability_auc),
        "separability_direction": direction,
    }


def _iqr_overlap_fraction(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0 or right.size == 0:
        return None
    left_q25, left_q75 = np.quantile(left, [0.25, 0.75])
    right_q25, right_q75 = np.quantile(right, [0.25, 0.75])
    overlap = max(0.0, min(left_q75, right_q75) - max(left_q25, right_q25))
    union = max(left_q75, right_q75) - min(left_q25, right_q25)
    if union <= 0.0:
        return 1.0 if np.isclose(left_q25, right_q25) else 0.0
    return float(overlap / union)


def _feature_distinguishability(separability_auc: float | None, iqr_overlap: float | None) -> str:
    if separability_auc is None or iqr_overlap is None:
        return "insufficient_rows"
    if separability_auc >= 0.80 and iqr_overlap <= 0.25:
        return "strong_single_feature_separation"
    if separability_auc >= 0.70 and iqr_overlap <= 0.50:
        return "moderate_single_feature_separation"
    return "weak_single_feature_separation"


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if values.size else None


def _median_or_none(values: np.ndarray) -> float | None:
    return float(np.median(values)) if values.size else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose behavior feature overlap for min-support model errors.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/decision/min_support/model_sensitivity/model_sensitivity_predictions.parquet"),
    )
    parser.add_argument(
        "--ranked-rows",
        type=Path,
        default=Path("results/decision/min_support/model_score_ranking/model_score_ranked_rows_of_interest.parquet"),
    )
    parser.add_argument(
        "--validation-labels",
        type=Path,
        default=Path("results/utility_labels/min_support_bbob_validation/utility_labels.parquet"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--top-score-quantile-min", type=float, default=0.90)
    parser.add_argument("--low-score-quantile-max", type=float, default=0.50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/behavior_overlap"),
    )
    args = parser.parse_args()
    run_behavior_overlap_diagnostics(
        predictions_path=args.predictions,
        ranked_rows_path=args.ranked_rows,
        validation_labels_path=args.validation_labels,
        output_dir=args.output_dir,
        target_column=args.target_column,
        top_score_quantile_min=args.top_score_quantile_min,
        low_score_quantile_max=args.low_score_quantile_max,
    )


if __name__ == "__main__":
    main()
