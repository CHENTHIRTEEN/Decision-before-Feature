from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from decision.min_support_diagnostics import GROUP_LAYERS
from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _check_target, _json_default


DOMAIN_PAIRS = {
    "changed_algorithm": {
        "train": "changed_algorithm_train",
        "validation": "changed_algorithm_validation",
    },
    "same_algorithm": {
        "train": "same_algorithm_train",
        "validation": "same_algorithm_reference",
    },
}

GROUP_KEY_COLUMNS = ["model_name", "layer", "group", "family", "dimension", "FE_ratio", "problem_id"]
RANK_METRIC_COLUMNS = [
    "rows",
    "utility_gt_zero_rows",
    "utility_gt_zero_rate",
    "score_mean",
    "score_median",
    "positive_score_mean",
    "positive_score_median",
    "positive_score_rank_best",
    "positive_score_rank_median",
    "positive_score_rank_mean",
    "positive_score_rank_worst",
    "positive_score_quantile_best",
    "positive_score_quantile_median",
    "positive_score_quantile_mean",
    "positive_score_quantile_worst",
]
TOPK_METRIC_COLUMNS = [
    "rows",
    "top_k_rows",
    "utility_gt_zero_rows",
    "top_k_utility_gt_zero_rows",
    "top_k_positive_row_capture_rate",
    "top_k_positive_row_rate",
    "top_k_positive_utility_sum",
    "top_k_positive_utility_capture_rate",
]


def run_score_generalization_diagnostics(
    *,
    train_rank_summary_path: Path,
    train_topk_summary_path: Path,
    validation_rank_summary_path: Path,
    validation_topk_summary_path: Path,
    validation_predictions_path: Path,
    output_dir: Path,
    target_column: str,
    validation_threshold_mode: str,
) -> dict[str, Any]:
    _check_target(target_column)

    train_rank = pq.read_table(train_rank_summary_path).to_pandas()
    train_topk = pq.read_table(train_topk_summary_path).to_pandas()
    validation_rank = pq.read_table(validation_rank_summary_path).to_pandas()
    validation_topk = pq.read_table(validation_topk_summary_path).to_pandas()
    validation_predictions = pq.read_table(validation_predictions_path).to_pandas()
    _check_rank_summary(train_rank, is_validation=False)
    _check_topk_summary(train_topk, is_validation=False)
    _check_rank_summary(validation_rank, is_validation=True)
    _check_topk_summary(validation_topk, is_validation=True)
    _check_validation_predictions(validation_predictions, target_column, validation_threshold_mode)

    validation_rank = validation_rank[validation_rank["threshold_mode"] == validation_threshold_mode].copy()
    validation_topk = validation_topk[validation_topk["threshold_mode"] == validation_threshold_mode].copy()

    rank_shift = _rank_shift_summary(train_rank, validation_rank)
    topk_shift = _topk_shift_summary(train_topk, validation_topk)
    distribution_shift = _positive_distribution_summary(train_rank, validation_rank)

    output_dir.mkdir(parents=True, exist_ok=True)
    rank_shift_path = output_dir / "score_generalization_rank_shift.parquet"
    topk_shift_path = output_dir / "score_generalization_topk_shift.parquet"
    distribution_shift_path = output_dir / "score_generalization_positive_distribution.parquet"
    summary_path = output_dir / "score_generalization_summary.json"
    pq.write_table(pa.Table.from_pandas(rank_shift, preserve_index=False), rank_shift_path)
    pq.write_table(pa.Table.from_pandas(topk_shift, preserve_index=False), topk_shift_path)
    pq.write_table(pa.Table.from_pandas(distribution_shift, preserve_index=False), distribution_shift_path)

    prediction_domain_rows = {
        domain: int(
            len(
                validation_predictions[
                    (validation_predictions["threshold_mode"] == validation_threshold_mode)
                    & (validation_predictions["label_source"] == label_source)
                ]
            )
        )
        for domain, label_source in {
            "changed_algorithm_validation": "changed_algorithm",
            "same_algorithm_reference": "same_algorithm",
        }.items()
    }
    summary = {
        "experiment": "min_support_train_vs_validation_score_generalization_diagnostics",
        "research_question": (
            "How do U_ELA > 0 score ranks and top-k capture change from training rows to "
            "function-family validation rows for the already trained min-support models?"
        ),
        "baseline_or_reference": (
            "in-sample train score ranking compared with validation score ranking under the unchanged "
            "function-family validation protocol"
        ),
        "target_column": target_column,
        "validation_threshold_mode_for_ranking_outputs": validation_threshold_mode,
        "domain_pairs": DOMAIN_PAIRS,
        "group_layers": GROUP_LAYERS,
        "inputs": {
            "train_rank_summary": str(train_rank_summary_path),
            "train_topk_summary": str(train_topk_summary_path),
            "validation_rank_summary": str(validation_rank_summary_path),
            "validation_topk_summary": str(validation_topk_summary_path),
            "validation_predictions": str(validation_predictions_path),
        },
        "rows": {
            "train_rank_summary_rows": int(len(train_rank)),
            "train_topk_summary_rows": int(len(train_topk)),
            "validation_rank_summary_rows_for_threshold": int(len(validation_rank)),
            "validation_topk_summary_rows_for_threshold": int(len(validation_topk)),
            "validation_prediction_rows_for_threshold": int(
                (validation_predictions["threshold_mode"] == validation_threshold_mode).sum()
            ),
            "validation_prediction_rows_by_domain": prediction_domain_rows,
            "rank_shift_rows": int(len(rank_shift)),
            "topk_shift_rows": int(len(topk_shift)),
            "positive_distribution_rows": int(len(distribution_shift)),
        },
        "outputs": {
            "rank_shift": str(rank_shift_path),
            "topk_shift": str(topk_shift_path),
            "positive_distribution": str(distribution_shift_path),
            "summary": str(summary_path),
        },
        "data_leakage_check": {
            "uses_existing_train_score_ranking": True,
            "uses_existing_model_score_ranking": True,
            "uses_existing_model_sensitivity_predictions_for_validation_context": True,
            "models_retrained": False,
            "threshold_selected_from_validation": False,
            "ela_features_used_as_decision_input": False,
            "metadata_used_only_for_grouping": True,
            "original_utility_labels_modified": False,
        },
        "notes": [
            "Family and problem_id groups are expected to be mostly split-specific under function-family validation.",
            "Dimension and FE_ratio groups can be directly matched across train and validation in this min-support run.",
            "Validation score ranks are read from the selected threshold mode; score ordering itself is independent of the threshold.",
        ],
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote score generalization rank shift to {rank_shift_path}")
    print(f"wrote score generalization top-k shift to {topk_shift_path}")
    print(f"wrote score generalization positive distribution to {distribution_shift_path}")
    print(f"wrote score generalization summary to {summary_path}")
    return summary


def _check_rank_summary(frame: pd.DataFrame, *, is_validation: bool) -> None:
    required = set(GROUP_KEY_COLUMNS + RANK_METRIC_COLUMNS + ["eval_domain", "model_family", "score_semantics"])
    if is_validation:
        required.add("threshold_mode")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"rank summary missing columns: {missing}")


def _check_topk_summary(frame: pd.DataFrame, *, is_validation: bool) -> None:
    required = set(
        GROUP_KEY_COLUMNS
        + TOPK_METRIC_COLUMNS
        + ["eval_domain", "model_family", "score_semantics", "top_k_label", "top_k_kind", "top_k_value"]
    )
    if is_validation:
        required.add("threshold_mode")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"top-k summary missing columns: {missing}")


def _check_validation_predictions(
    predictions: pd.DataFrame,
    target_column: str,
    validation_threshold_mode: str,
) -> None:
    required = {
        "model_name",
        "threshold_mode",
        "label_source",
        "decision_score",
        "family",
        "dimension",
        "FE_ratio",
        "problem_id",
        target_column,
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"model sensitivity predictions missing columns: {missing}")
    modes = set(predictions["threshold_mode"].astype(str).unique())
    if validation_threshold_mode not in modes:
        raise ValueError(
            f"validation threshold mode {validation_threshold_mode!r} not found in predictions: {sorted(modes)}"
        )


def _rank_shift_summary(train_rank: pd.DataFrame, validation_rank: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for comparison_domain, domains in DOMAIN_PAIRS.items():
        train = _domain_rank_frame(train_rank, domains["train"], comparison_domain, "train")
        validation = _domain_rank_frame(validation_rank, domains["validation"], comparison_domain, "validation")
        merged = _merge_shift_frames(
            train,
            validation,
            merge_columns=["comparison_domain", *GROUP_KEY_COLUMNS],
        )
        merged["positive_score_quantile_median_delta"] = (
            merged["validation_positive_score_quantile_median"] - merged["train_positive_score_quantile_median"]
        )
        merged["positive_score_quantile_mean_delta"] = (
            merged["validation_positive_score_quantile_mean"] - merged["train_positive_score_quantile_mean"]
        )
        merged["utility_gt_zero_rate_delta"] = (
            merged["validation_utility_gt_zero_rate"] - merged["train_utility_gt_zero_rate"]
        )
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def _topk_shift_summary(train_topk: pd.DataFrame, validation_topk: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for comparison_domain, domains in DOMAIN_PAIRS.items():
        train = _domain_topk_frame(train_topk, domains["train"], comparison_domain, "train")
        validation = _domain_topk_frame(validation_topk, domains["validation"], comparison_domain, "validation")
        merged = _merge_shift_frames(
            train,
            validation,
            merge_columns=[
                "comparison_domain",
                *GROUP_KEY_COLUMNS,
                "top_k_label",
                "top_k_kind",
                "top_k_value",
            ],
        )
        merged["top_k_positive_row_capture_rate_delta"] = (
            merged["validation_top_k_positive_row_capture_rate"] - merged["train_top_k_positive_row_capture_rate"]
        )
        merged["top_k_positive_utility_capture_rate_delta"] = (
            merged["validation_top_k_positive_utility_capture_rate"]
            - merged["train_top_k_positive_utility_capture_rate"]
        )
        merged["top_k_positive_row_rate_delta"] = (
            merged["validation_top_k_positive_row_rate"] - merged["train_top_k_positive_row_rate"]
        )
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def _positive_distribution_summary(train_rank: pd.DataFrame, validation_rank: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for comparison_domain, domains in DOMAIN_PAIRS.items():
        frames.append(_distribution_frame(train_rank, domains["train"], comparison_domain, "train"))
        frames.append(_distribution_frame(validation_rank, domains["validation"], comparison_domain, "validation"))
    distribution = pd.concat(frames, ignore_index=True)
    distribution["positive_row_share_within_layer"] = distribution.groupby(
        ["split", "comparison_domain", "model_name", "layer"], dropna=False
    )["utility_gt_zero_rows"].transform(_share)
    distribution["row_share_within_layer"] = distribution.groupby(
        ["split", "comparison_domain", "model_name", "layer"], dropna=False
    )["rows"].transform(_share)
    return distribution


def _domain_rank_frame(
    frame: pd.DataFrame,
    eval_domain: str,
    comparison_domain: str,
    prefix: str,
) -> pd.DataFrame:
    columns = ["model_family", "score_semantics", "eval_domain", *GROUP_KEY_COLUMNS, *RANK_METRIC_COLUMNS]
    result = frame.loc[frame["eval_domain"] == eval_domain, columns].copy()
    result.insert(0, "comparison_domain", comparison_domain)
    result = result.rename(columns={column: f"{prefix}_{column}" for column in RANK_METRIC_COLUMNS})
    result = result.rename(
        columns={
            "model_family": f"{prefix}_model_family",
            "score_semantics": f"{prefix}_score_semantics",
            "eval_domain": f"{prefix}_eval_domain",
        }
    )
    return result


def _domain_topk_frame(
    frame: pd.DataFrame,
    eval_domain: str,
    comparison_domain: str,
    prefix: str,
) -> pd.DataFrame:
    columns = [
        "model_family",
        "score_semantics",
        "eval_domain",
        *GROUP_KEY_COLUMNS,
        "top_k_label",
        "top_k_kind",
        "top_k_value",
        *TOPK_METRIC_COLUMNS,
    ]
    result = frame.loc[frame["eval_domain"] == eval_domain, columns].copy()
    result.insert(0, "comparison_domain", comparison_domain)
    result = result.rename(columns={column: f"{prefix}_{column}" for column in TOPK_METRIC_COLUMNS})
    result = result.rename(
        columns={
            "model_family": f"{prefix}_model_family",
            "score_semantics": f"{prefix}_score_semantics",
            "eval_domain": f"{prefix}_eval_domain",
        }
    )
    return result


def _distribution_frame(
    frame: pd.DataFrame,
    eval_domain: str,
    comparison_domain: str,
    split: str,
) -> pd.DataFrame:
    columns = [
        "model_name",
        "model_family",
        "score_semantics",
        "eval_domain",
        "layer",
        "group",
        "family",
        "dimension",
        "FE_ratio",
        "problem_id",
        "rows",
        "utility_gt_zero_rows",
        "utility_gt_zero_rate",
        "positive_score_rank_median",
        "positive_score_quantile_median",
        "positive_score_quantile_mean",
    ]
    result = frame.loc[frame["eval_domain"] == eval_domain, columns].copy()
    result.insert(0, "comparison_domain", comparison_domain)
    result.insert(0, "split", split)
    return result


def _merge_shift_frames(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    merge_columns: list[str],
) -> pd.DataFrame:
    merged = train.merge(validation, on=merge_columns, how="outer", indicator=True)
    merged["group_status"] = merged["_merge"].map(
        {
            "both": "matched",
            "left_only": "train_only",
            "right_only": "validation_only",
        }
    )
    merged = merged.drop(columns=["_merge"])
    return merged


def _share(values: pd.Series) -> pd.Series:
    total = float(values.sum())
    if total == 0.0:
        return pd.Series(0.0, index=values.index)
    return values.astype(float) / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose min-support train-vs-validation score generalization.")
    parser.add_argument(
        "--train-rank-summary",
        type=Path,
        default=Path("results/decision/min_support/train_score_ranking/train_score_rank_summary.parquet"),
    )
    parser.add_argument(
        "--train-topk-summary",
        type=Path,
        default=Path("results/decision/min_support/train_score_ranking/train_score_topk_summary.parquet"),
    )
    parser.add_argument(
        "--validation-rank-summary",
        type=Path,
        default=Path("results/decision/min_support/model_score_ranking/model_score_rank_summary.parquet"),
    )
    parser.add_argument(
        "--validation-topk-summary",
        type=Path,
        default=Path("results/decision/min_support/model_score_ranking/model_score_topk_summary.parquet"),
    )
    parser.add_argument(
        "--validation-predictions",
        type=Path,
        default=Path("results/decision/min_support/model_sensitivity/model_sensitivity_predictions.parquet"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--validation-threshold-mode", default="zero")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/score_generalization"),
    )
    args = parser.parse_args()
    run_score_generalization_diagnostics(
        train_rank_summary_path=args.train_rank_summary,
        train_topk_summary_path=args.train_topk_summary,
        validation_rank_summary_path=args.validation_rank_summary,
        validation_topk_summary_path=args.validation_topk_summary,
        validation_predictions_path=args.validation_predictions,
        output_dir=args.output_dir,
        target_column=args.target_column,
        validation_threshold_mode=args.validation_threshold_mode,
    )


if __name__ == "__main__":
    main()
