from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from decision.min_support_diagnostics import GROUP_LAYERS, _group_label
from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _check_target, _json_default


EVALUATION_DOMAINS = {
    "changed_algorithm_validation": "changed_algorithm",
    "same_algorithm_reference": "same_algorithm",
}

TOP_K_SPECS = [
    ("top_1_row", "rows", 1.0),
    ("top_3_rows", "rows", 3.0),
    ("top_5_rows", "rows", 5.0),
    ("top_10_rows", "rows", 10.0),
    ("top_1pct", "fraction", 0.01),
    ("top_2pct", "fraction", 0.02),
    ("top_5pct", "fraction", 0.05),
    ("top_10pct", "fraction", 0.10),
    ("top_20pct", "fraction", 0.20),
]


def run_model_score_ranking_diagnostics(
    *,
    predictions_path: Path,
    output_dir: Path,
    target_column: str,
) -> dict[str, Any]:
    _check_target(target_column)
    predictions = pq.read_table(predictions_path).to_pandas()
    _check_prediction_columns(predictions, target_column)

    rank_frames = []
    topk_frames = []
    interest_frames = []

    for model_name, model_frame in predictions.groupby("model_name", dropna=False):
        for threshold_mode, threshold_frame in model_frame.groupby("threshold_mode", dropna=False):
            for eval_domain, label_source in EVALUATION_DOMAINS.items():
                domain_frame = threshold_frame[threshold_frame["label_source"] == label_source].copy()
                if domain_frame.empty:
                    continue
                domain_ranked = _add_score_ranks(domain_frame)
                interest_frames.append(
                    _rows_of_interest(
                        domain_ranked,
                        str(model_name),
                        str(threshold_mode),
                        eval_domain,
                        target_column,
                    )
                )
                for layer, columns in GROUP_LAYERS.items():
                    rank_frames.append(
                        _rank_layer_summary(
                            frame=domain_ranked,
                            model_name=str(model_name),
                            threshold_mode=str(threshold_mode),
                            eval_domain=eval_domain,
                            layer=layer,
                            group_columns=columns,
                            target_column=target_column,
                        )
                    )
                    topk_frames.append(
                        _topk_layer_summary(
                            frame=domain_ranked,
                            model_name=str(model_name),
                            threshold_mode=str(threshold_mode),
                            eval_domain=eval_domain,
                            layer=layer,
                            group_columns=columns,
                            target_column=target_column,
                        )
                    )

    rank_summary = pd.concat(rank_frames, ignore_index=True)
    topk_summary = pd.concat(topk_frames, ignore_index=True)
    ranked_rows_of_interest = pd.concat(interest_frames, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    rank_summary_path = output_dir / "model_score_rank_summary.parquet"
    topk_summary_path = output_dir / "model_score_topk_summary.parquet"
    interest_rows_path = output_dir / "model_score_ranked_rows_of_interest.parquet"
    summary_path = output_dir / "model_score_ranking_summary.json"
    pq.write_table(pa.Table.from_pandas(rank_summary, preserve_index=False), rank_summary_path)
    pq.write_table(pa.Table.from_pandas(topk_summary, preserve_index=False), topk_summary_path)
    pq.write_table(pa.Table.from_pandas(ranked_rows_of_interest, preserve_index=False), interest_rows_path)

    summary = {
        "experiment": "min_support_model_score_ranking_diagnostics",
        "predictions": str(predictions_path),
        "target_column": target_column,
        "rows": {
            "prediction_rows": int(len(predictions)),
            "rank_summary_rows": int(len(rank_summary)),
            "topk_summary_rows": int(len(topk_summary)),
            "ranked_rows_of_interest": int(len(ranked_rows_of_interest)),
        },
        "models": sorted(predictions["model_name"].astype(str).unique().tolist()),
        "threshold_modes": sorted(predictions["threshold_mode"].astype(str).unique().tolist()),
        "evaluation_domains": EVALUATION_DOMAINS,
        "ranking_definition": {
            "score_order": "higher decision_score ranks earlier",
            "score_rank": "1 is the highest score within a model, threshold mode, evaluation domain, and layer group",
            "score_quantile": "1.0 is highest score and 0.0 is lowest score within the same ranking set",
            "unhelpful_called_row": f"{target_column} <= 0 and decision_run_ela is true",
        },
        "top_k_specs": [{"top_k_label": label, "kind": kind, "value": value} for label, kind, value in TOP_K_SPECS],
        "outputs": {
            "rank_summary": str(rank_summary_path),
            "topk_summary": str(topk_summary_path),
            "ranked_rows_of_interest": str(interest_rows_path),
            "summary": str(summary_path),
        },
        "data_leakage_check": {
            "uses_existing_model_sensitivity_predictions_only": True,
            "decision_input_uses_only_existing_model_scores": True,
            "ela_features_used_as_decision_input": False,
            "original_utility_labels_modified": False,
        },
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote model score rank summary to {rank_summary_path}")
    print(f"wrote model score top-k summary to {topk_summary_path}")
    print(f"wrote model score ranked rows of interest to {interest_rows_path}")
    print(f"wrote model score ranking summary to {summary_path}")
    return summary


def _check_prediction_columns(predictions: pd.DataFrame, target_column: str) -> None:
    required = {
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
        "label_source",
        "default_algorithm",
        "selected_algorithm",
        "decision_score",
        "decision_run_ela",
        target_column,
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"model sensitivity predictions missing required columns: {missing}")


def _add_score_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["score_rank"] = ranked["decision_score"].rank(method="min", ascending=False).astype(float)
    if len(ranked) <= 1:
        ranked["score_quantile"] = 1.0
    else:
        ranked["score_quantile"] = 1.0 - ((ranked["score_rank"] - 1.0) / float(len(ranked) - 1))
    return ranked


def _rank_layer_summary(
    *,
    frame: pd.DataFrame,
    model_name: str,
    threshold_mode: str,
    eval_domain: str,
    layer: str,
    group_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    if not group_columns:
        ranked = _add_score_ranks(frame)
        return pd.DataFrame([_rank_row(ranked, model_name, threshold_mode, eval_domain, layer, {}, target_column)])
    rows = []
    for group_values, subset in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group = dict(zip(group_columns, group_values, strict=True))
        ranked = _add_score_ranks(subset)
        rows.append(_rank_row(ranked, model_name, threshold_mode, eval_domain, layer, group, target_column))
    return pd.DataFrame(rows)


def _topk_layer_summary(
    *,
    frame: pd.DataFrame,
    model_name: str,
    threshold_mode: str,
    eval_domain: str,
    layer: str,
    group_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    if not group_columns:
        ranked = _add_score_ranks(frame)
        return pd.DataFrame(
            _topk_rows(ranked, model_name, threshold_mode, eval_domain, layer, {}, target_column)
        )
    rows = []
    for group_values, subset in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group = dict(zip(group_columns, group_values, strict=True))
        ranked = _add_score_ranks(subset)
        rows.extend(_topk_rows(ranked, model_name, threshold_mode, eval_domain, layer, group, target_column))
    return pd.DataFrame(rows)


def _rank_row(
    frame: pd.DataFrame,
    model_name: str,
    threshold_mode: str,
    eval_domain: str,
    layer: str,
    group: dict[str, Any],
    target_column: str,
) -> dict[str, Any]:
    observed = frame[target_column].to_numpy(dtype=float)
    score = frame["decision_score"].to_numpy(dtype=float)
    positive = observed > 0.0
    unhelpful_called = (observed <= 0.0) & frame["decision_run_ela"].to_numpy(dtype=bool)
    positive_score = score[positive]
    unhelpful_score = score[unhelpful_called]
    return {
        **_common_fields(frame, model_name, threshold_mode, eval_domain, layer, group),
        "rows": int(len(frame)),
        "utility_gt_zero_rows": int(np.sum(positive)),
        "utility_gt_zero_rate": float(np.mean(positive)),
        "decision_ela_call_rows": int(np.sum(frame["decision_run_ela"].to_numpy(dtype=bool))),
        "unhelpful_called_rows": int(np.sum(unhelpful_called)),
        "unhelpful_called_rate": float(np.mean(unhelpful_called)),
        "score_mean": float(np.mean(score)),
        "score_median": float(np.median(score)),
        "positive_score_mean": _mean_or_none(positive_score),
        "positive_score_median": _median_or_none(positive_score),
        "unhelpful_called_score_mean": _mean_or_none(unhelpful_score),
        "unhelpful_called_score_median": _median_or_none(unhelpful_score),
        "positive_minus_unhelpful_called_score_mean": (
            float(np.mean(positive_score) - np.mean(unhelpful_score))
            if positive_score.size and unhelpful_score.size
            else None
        ),
        **_rank_stats(frame.loc[positive, "score_rank"], "positive_score_rank", larger_is_better=False),
        **_rank_stats(frame.loc[positive, "score_quantile"], "positive_score_quantile", larger_is_better=True),
        **_rank_stats(
            frame.loc[unhelpful_called, "score_rank"],
            "unhelpful_called_score_rank",
            larger_is_better=False,
        ),
        **_rank_stats(
            frame.loc[unhelpful_called, "score_quantile"],
            "unhelpful_called_score_quantile",
            larger_is_better=True,
        ),
    }


def _topk_rows(
    frame: pd.DataFrame,
    model_name: str,
    threshold_mode: str,
    eval_domain: str,
    layer: str,
    group: dict[str, Any],
    target_column: str,
) -> list[dict[str, Any]]:
    observed = frame[target_column].to_numpy(dtype=float)
    positive = observed > 0.0
    decision_run = frame["decision_run_ela"].to_numpy(dtype=bool)
    unhelpful_called = (observed <= 0.0) & decision_run
    positive_utility_sum = float(np.sum(observed[positive]))
    ranked = frame.sort_values(["decision_score", "problem_id", "seed", "FE"], ascending=[False, True, True, True])
    rows = []
    for top_k_label, kind, value in TOP_K_SPECS:
        top_k_rows = _top_k_count(len(ranked), kind, value)
        top_index = ranked.index[:top_k_rows]
        top_positive = positive[frame.index.get_indexer(top_index)]
        top_unhelpful_called = unhelpful_called[frame.index.get_indexer(top_index)]
        top_observed = observed[frame.index.get_indexer(top_index)]
        rows.append(
            {
                **_common_fields(frame, model_name, threshold_mode, eval_domain, layer, group),
                "rows": int(len(frame)),
                "top_k_label": top_k_label,
                "top_k_kind": kind,
                "top_k_value": float(value),
                "top_k_rows": int(top_k_rows),
                "utility_gt_zero_rows": int(np.sum(positive)),
                "unhelpful_called_rows": int(np.sum(unhelpful_called)),
                "top_k_utility_gt_zero_rows": int(np.sum(top_positive)),
                "top_k_positive_row_capture_rate": (
                    float(np.sum(top_positive) / np.sum(positive)) if np.any(positive) else 0.0
                ),
                "top_k_positive_row_rate": float(np.mean(top_positive)) if top_k_rows else 0.0,
                "top_k_positive_utility_sum": float(np.sum(top_observed[top_positive])),
                "top_k_positive_utility_capture_rate": (
                    float(np.sum(top_observed[top_positive]) / positive_utility_sum)
                    if positive_utility_sum > 0.0
                    else 0.0
                ),
                "top_k_unhelpful_called_rows": int(np.sum(top_unhelpful_called)),
                "top_k_unhelpful_called_rate": float(np.mean(top_unhelpful_called)) if top_k_rows else 0.0,
                "top_k_unhelpful_called_share": (
                    float(np.sum(top_unhelpful_called) / np.sum(unhelpful_called))
                    if np.any(unhelpful_called)
                    else 0.0
                ),
                "top_k_unhelpful_called_utility_sum": float(np.sum(top_observed[top_unhelpful_called])),
                "top_k_unhelpful_called_cost_sum": float(-np.sum(top_observed[top_unhelpful_called])),
            }
        )
    return rows


def _rows_of_interest(
    frame: pd.DataFrame,
    model_name: str,
    threshold_mode: str,
    eval_domain: str,
    target_column: str,
) -> pd.DataFrame:
    observed = frame[target_column].to_numpy(dtype=float)
    unhelpful_called = (observed <= 0.0) & frame["decision_run_ela"].to_numpy(dtype=bool)
    utility_gt_zero = observed > 0.0
    interest = frame[utility_gt_zero | unhelpful_called].copy()
    if interest.empty:
        return pd.DataFrame()
    interest.insert(0, "eval_domain", eval_domain)
    interest.insert(0, "threshold_mode_for_rank", threshold_mode)
    interest.insert(0, "model_name_for_rank", model_name)
    interest["row_role"] = np.where(interest[target_column] > 0.0, "utility_gt_zero", "unhelpful_called")
    keep_columns = [
        "model_name_for_rank",
        "threshold_mode_for_rank",
        "eval_domain",
        "row_role",
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
        "label_source",
        "default_algorithm",
        "selected_algorithm",
        "decision_score",
        "score_rank",
        "score_quantile",
        "decision_run_ela",
        target_column,
    ]
    return interest[keep_columns]


def _common_fields(
    frame: pd.DataFrame,
    model_name: str,
    threshold_mode: str,
    eval_domain: str,
    layer: str,
    group: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model_family": str(frame["model_family"].iloc[0]),
        "score_semantics": str(frame["score_semantics"].iloc[0]),
        "eval_domain": eval_domain,
        "threshold_mode": threshold_mode,
        "threshold": float(frame["decision_threshold"].iloc[0]),
        "layer": layer,
        "group": _group_label(group),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "problem_id": group.get("problem_id"),
    }


def _rank_stats(values: pd.Series, prefix: str, *, larger_is_better: bool) -> dict[str, float | None]:
    numeric = values.to_numpy(dtype=float)
    if numeric.size == 0:
        return {
            f"{prefix}_best": None,
            f"{prefix}_median": None,
            f"{prefix}_mean": None,
            f"{prefix}_worst": None,
        }
    return {
        f"{prefix}_best": float(np.max(numeric) if larger_is_better else np.min(numeric)),
        f"{prefix}_median": float(np.median(numeric)),
        f"{prefix}_mean": float(np.mean(numeric)),
        f"{prefix}_worst": float(np.min(numeric) if larger_is_better else np.max(numeric)),
    }


def _top_k_count(rows: int, kind: str, value: float) -> int:
    if rows <= 0:
        return 0
    if kind == "rows":
        return min(rows, max(1, int(value)))
    if kind == "fraction":
        return min(rows, max(1, int(np.ceil(rows * value))))
    raise ValueError(f"unknown top-k kind: {kind}")


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if values.size else None


def _median_or_none(values: np.ndarray) -> float | None:
    return float(np.median(values)) if values.size else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose min-support model score ranking behavior.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/decision/min_support/model_sensitivity/model_sensitivity_predictions.parquet"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/model_score_ranking"),
    )
    args = parser.parse_args()
    run_model_score_ranking_diagnostics(
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        target_column=args.target_column,
    )


if __name__ == "__main__":
    main()
