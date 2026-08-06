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
from decision.min_support_model_score_ranking import TOP_K_SPECS, _add_score_ranks, _rank_stats, _top_k_count


EVALUATION_DOMAINS = {
    "all_train": None,
    "changed_algorithm_train": "changed_algorithm",
    "same_algorithm_train": "same_algorithm",
}

GROUP_LAYERS = {
    "overall": [],
    "family": ["family"],
    "dimension": ["dimension"],
    "fe_ratio": ["FE_ratio"],
    "problem_id": ["problem_id"],
}


def run_train_score_ranking_diagnostics(
    *,
    train_predictions_path: Path,
    output_dir: Path,
    target_column: str,
) -> dict[str, Any]:
    _check_target(target_column)
    train_predictions = pq.read_table(train_predictions_path).to_pandas()
    _check_train_prediction_columns(train_predictions, target_column)

    rank_frames = []
    topk_frames = []
    positive_row_frames = []

    for model_name, model_frame in train_predictions.groupby("model_name", dropna=False):
        for eval_domain, label_source in EVALUATION_DOMAINS.items():
            domain_frame = (
                model_frame.copy()
                if label_source is None
                else model_frame[model_frame["label_source"] == label_source].copy()
            )
            if domain_frame.empty:
                continue
            for layer, columns in GROUP_LAYERS.items():
                rank_frames.append(
                    _rank_layer_summary(
                        frame=domain_frame,
                        model_name=str(model_name),
                        eval_domain=eval_domain,
                        layer=layer,
                        group_columns=columns,
                        target_column=target_column,
                    )
                )
                topk_frames.append(
                    _topk_layer_summary(
                        frame=domain_frame,
                        model_name=str(model_name),
                        eval_domain=eval_domain,
                        layer=layer,
                        group_columns=columns,
                        target_column=target_column,
                    )
                )
            positive_row_frames.append(_positive_rows(domain_frame, str(model_name), eval_domain, target_column))

    rank_summary = pd.concat(rank_frames, ignore_index=True)
    topk_summary = pd.concat(topk_frames, ignore_index=True)
    positive_rows = pd.concat(positive_row_frames, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    rank_summary_path = output_dir / "train_score_rank_summary.parquet"
    topk_summary_path = output_dir / "train_score_topk_summary.parquet"
    positive_rows_path = output_dir / "train_score_ranked_positive_rows.parquet"
    summary_path = output_dir / "train_score_ranking_summary.json"
    pq.write_table(pa.Table.from_pandas(rank_summary, preserve_index=False), rank_summary_path)
    pq.write_table(pa.Table.from_pandas(topk_summary, preserve_index=False), topk_summary_path)
    pq.write_table(pa.Table.from_pandas(positive_rows, preserve_index=False), positive_rows_path)

    summary = {
        "experiment": "min_support_train_score_ranking_diagnostics",
        "research_question": (
            "Where do U_ELA > 0 training rows appear in each already trained model's score ordering?"
        ),
        "train_predictions": str(train_predictions_path),
        "target_column": target_column,
        "rows": {
            "train_prediction_rows": int(len(train_predictions)),
            "rank_summary_rows": int(len(rank_summary)),
            "topk_summary_rows": int(len(topk_summary)),
            "ranked_positive_rows": int(len(positive_rows)),
        },
        "models": sorted(train_predictions["model_name"].astype(str).unique().tolist()),
        "evaluation_domains": EVALUATION_DOMAINS,
        "group_layers": GROUP_LAYERS,
        "ranking_definition": {
            "score_order": "higher decision_score ranks earlier",
            "score_rank": "1 is the highest score within a model, evaluation domain, and layer group",
            "score_quantile": "1.0 is highest score and 0.0 is lowest score within the same ranking set",
        },
        "top_k_specs": [{"top_k_label": label, "kind": kind, "value": value} for label, kind, value in TOP_K_SPECS],
        "outputs": {
            "rank_summary": str(rank_summary_path),
            "topk_summary": str(topk_summary_path),
            "ranked_positive_rows": str(positive_rows_path),
            "summary": str(summary_path),
        },
        "data_leakage_check": {
            "uses_existing_model_sensitivity_train_predictions_only": True,
            "models_retrained": False,
            "threshold_selected_from_validation": False,
            "ela_features_used_as_decision_input": False,
            "metadata_used_only_for_grouping": True,
            "original_utility_labels_modified": False,
        },
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote train score rank summary to {rank_summary_path}")
    print(f"wrote train score top-k summary to {topk_summary_path}")
    print(f"wrote train score ranked positive rows to {positive_rows_path}")
    print(f"wrote train score ranking summary to {summary_path}")
    return summary


def _check_train_prediction_columns(train_predictions: pd.DataFrame, target_column: str) -> None:
    required = {
        "data_split",
        "model_name",
        "model_family",
        "score_semantics",
        "training_target",
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
        target_column,
    }
    missing = sorted(required.difference(train_predictions.columns))
    if missing:
        raise ValueError(f"model sensitivity train predictions missing required columns: {missing}")
    splits = set(train_predictions["data_split"].astype(str).unique())
    if splits != {"train"}:
        raise ValueError(f"expected only train predictions, got data_split values: {sorted(splits)}")


def _rank_layer_summary(
    *,
    frame: pd.DataFrame,
    model_name: str,
    eval_domain: str,
    layer: str,
    group_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    if not group_columns:
        ranked = _add_score_ranks(frame)
        return pd.DataFrame([_rank_row(ranked, model_name, eval_domain, layer, {}, target_column)])
    rows = []
    for group_values, subset in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group = dict(zip(group_columns, group_values, strict=True))
        ranked = _add_score_ranks(subset)
        rows.append(_rank_row(ranked, model_name, eval_domain, layer, group, target_column))
    return pd.DataFrame(rows)


def _topk_layer_summary(
    *,
    frame: pd.DataFrame,
    model_name: str,
    eval_domain: str,
    layer: str,
    group_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    if not group_columns:
        ranked = _add_score_ranks(frame)
        return pd.DataFrame(_topk_rows(ranked, model_name, eval_domain, layer, {}, target_column))
    rows = []
    for group_values, subset in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group = dict(zip(group_columns, group_values, strict=True))
        ranked = _add_score_ranks(subset)
        rows.extend(_topk_rows(ranked, model_name, eval_domain, layer, group, target_column))
    return pd.DataFrame(rows)


def _rank_row(
    frame: pd.DataFrame,
    model_name: str,
    eval_domain: str,
    layer: str,
    group: dict[str, Any],
    target_column: str,
) -> dict[str, Any]:
    observed = frame[target_column].to_numpy(dtype=float)
    score = frame["decision_score"].to_numpy(dtype=float)
    positive = observed > 0.0
    positive_score = score[positive]
    return {
        **_common_fields(frame, model_name, eval_domain, layer, group),
        "rows": int(len(frame)),
        "utility_gt_zero_rows": int(np.sum(positive)),
        "utility_gt_zero_rate": float(np.mean(positive)),
        "positive_utility_sum": float(np.sum(observed[positive])),
        "score_mean": float(np.mean(score)),
        "score_median": float(np.median(score)),
        "positive_score_mean": _mean_or_none(positive_score),
        "positive_score_median": _median_or_none(positive_score),
        **_rank_stats(frame.loc[positive, "score_rank"], "positive_score_rank", larger_is_better=False),
        **_rank_stats(frame.loc[positive, "score_quantile"], "positive_score_quantile", larger_is_better=True),
    }


def _topk_rows(
    frame: pd.DataFrame,
    model_name: str,
    eval_domain: str,
    layer: str,
    group: dict[str, Any],
    target_column: str,
) -> list[dict[str, Any]]:
    observed = frame[target_column].to_numpy(dtype=float)
    positive = observed > 0.0
    positive_utility_sum = float(np.sum(observed[positive]))
    ranked = frame.sort_values(["decision_score", "problem_id", "seed", "FE"], ascending=[False, True, True, True])
    rows = []
    for top_k_label, kind, value in TOP_K_SPECS:
        top_k_rows = _top_k_count(len(ranked), kind, value)
        top_index = ranked.index[:top_k_rows]
        top_positions = frame.index.get_indexer(top_index)
        top_positive = positive[top_positions]
        top_observed = observed[top_positions]
        rows.append(
            {
                **_common_fields(frame, model_name, eval_domain, layer, group),
                "rows": int(len(frame)),
                "top_k_label": top_k_label,
                "top_k_kind": kind,
                "top_k_value": float(value),
                "top_k_rows": int(top_k_rows),
                "utility_gt_zero_rows": int(np.sum(positive)),
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
            }
        )
    return rows


def _positive_rows(
    frame: pd.DataFrame,
    model_name: str,
    eval_domain: str,
    target_column: str,
) -> pd.DataFrame:
    ranked = _add_score_ranks(frame)
    positive = ranked[ranked[target_column] > 0.0].copy()
    if positive.empty:
        return pd.DataFrame()
    positive.insert(0, "eval_domain", eval_domain)
    positive.insert(0, "model_name_for_rank", model_name)
    keep_columns = [
        "model_name_for_rank",
        "eval_domain",
        "model_name",
        "model_family",
        "score_semantics",
        "training_target",
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
        target_column,
    ]
    return positive[keep_columns]


def _common_fields(
    frame: pd.DataFrame,
    model_name: str,
    eval_domain: str,
    layer: str,
    group: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model_family": str(frame["model_family"].iloc[0]),
        "score_semantics": str(frame["score_semantics"].iloc[0]),
        "eval_domain": eval_domain,
        "layer": layer,
        "group": _group_label(group),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "problem_id": group.get("problem_id"),
    }


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if values.size else None


def _median_or_none(values: np.ndarray) -> float | None:
    return float(np.median(values)) if values.size else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose min-support train score ranking behavior.")
    parser.add_argument(
        "--train-predictions",
        type=Path,
        default=Path("results/decision/min_support/model_sensitivity/model_sensitivity_train_predictions.parquet"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/train_score_ranking"),
    )
    args = parser.parse_args()
    run_train_score_ranking_diagnostics(
        train_predictions_path=args.train_predictions,
        output_dir=args.output_dir,
        target_column=args.target_column,
    )


if __name__ == "__main__":
    main()
