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
from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _check_target, _json_default
from decision.min_support_model_score_ranking import _add_score_ranks


MERGE_KEY_COLUMNS = ["problem_id", "family", "dimension", "prefix_algorithm", "seed", "FE", "FE_ratio"]
COMPARISON_DOMAINS = {
    "changed_algorithm": "changed_algorithm",
    "same_algorithm": "same_algorithm",
}
ROLE_ORDER = [
    "train_high_capture_positive_top10",
    "train_low_score_positive_reference",
    "validation_high_score_positive_top10",
    "validation_low_capture_positive_below_top10",
]
GROUP_LAYERS = {
    "overall": [],
    "family": ["family"],
    "dimension": ["dimension"],
    "fe_ratio": ["FE_ratio"],
    "family_fe_ratio": ["family", "FE_ratio"],
    "problem_id": ["problem_id"],
}


def run_feature_distribution_shift_diagnostics(
    *,
    train_behavior_root: Path,
    validation_behavior_root: Path,
    train_predictions_path: Path,
    validation_predictions_path: Path,
    score_generalization_distribution_path: Path,
    output_dir: Path,
    target_column: str,
    validation_threshold_mode: str,
    top_score_quantile_min: float,
) -> dict[str, Any]:
    _check_target(target_column)
    _check_quantile(top_score_quantile_min)

    train_behavior = _read_behavior_rows(train_behavior_root)
    validation_behavior = _read_behavior_rows(validation_behavior_root)
    train_predictions = pq.read_table(train_predictions_path).to_pandas()
    validation_predictions = pq.read_table(validation_predictions_path).to_pandas()
    score_distribution = pq.read_table(score_generalization_distribution_path).to_pandas()
    _check_predictions(train_predictions, target_column, require_threshold=False)
    _check_predictions(validation_predictions, target_column, require_threshold=True)
    _check_behavior(train_behavior)
    _check_behavior(validation_behavior)

    validation_predictions = validation_predictions[
        validation_predictions["threshold_mode"] == validation_threshold_mode
    ].copy()
    train_enriched = _enrich_predictions(
        predictions=train_predictions,
        behavior=train_behavior,
        split_name="train",
        target_column=target_column,
        top_score_quantile_min=top_score_quantile_min,
    )
    validation_enriched = _enrich_predictions(
        predictions=validation_predictions,
        behavior=validation_behavior,
        split_name="validation",
        target_column=target_column,
        top_score_quantile_min=top_score_quantile_min,
    )
    role_rows = pd.concat([train_enriched, validation_enriched], ignore_index=True)
    selected_role_rows = role_rows[role_rows["shift_role"].isin(ROLE_ORDER)].copy()

    role_feature_summary = _role_feature_summary(selected_role_rows, target_column)
    feature_shift_summary = _feature_shift_summary(selected_role_rows)
    migration_summary = _migration_summary(score_distribution)
    score_shift_summary = _score_shift_summary(selected_role_rows, target_column)

    output_dir.mkdir(parents=True, exist_ok=True)
    role_rows_path = output_dir / "feature_shift_role_rows.parquet"
    role_feature_summary_path = output_dir / "feature_shift_role_feature_summary.parquet"
    feature_shift_summary_path = output_dir / "feature_shift_pairwise_summary.parquet"
    migration_summary_path = output_dir / "feature_shift_family_fe_ratio_migration.parquet"
    score_shift_summary_path = output_dir / "feature_shift_score_summary.parquet"
    summary_path = output_dir / "feature_shift_summary.json"
    pq.write_table(pa.Table.from_pandas(selected_role_rows, preserve_index=False), role_rows_path)
    pq.write_table(pa.Table.from_pandas(role_feature_summary, preserve_index=False), role_feature_summary_path)
    pq.write_table(pa.Table.from_pandas(feature_shift_summary, preserve_index=False), feature_shift_summary_path)
    pq.write_table(pa.Table.from_pandas(migration_summary, preserve_index=False), migration_summary_path)
    pq.write_table(pa.Table.from_pandas(score_shift_summary, preserve_index=False), score_shift_summary_path)

    summary = {
        "experiment": "min_support_feature_distribution_shift_diagnostics",
        "research_question": (
            "Do behavior feature distributions shift from train regions where U_ELA > 0 rows are high-scored "
            "to validation regions where U_ELA > 0 rows are not captured by high model scores?"
        ),
        "target_column": target_column,
        "top_score_quantile_min": top_score_quantile_min,
        "role_definitions": {
            "train_high_capture_positive_top10": (
                f"train rows with {target_column} > 0 and score_quantile >= {top_score_quantile_min}"
            ),
            "validation_low_capture_positive_below_top10": (
                f"validation rows with {target_column} > 0 and score_quantile < {top_score_quantile_min}"
            ),
            "train_low_score_positive_reference": (
                f"train rows with {target_column} > 0 and score_quantile < {top_score_quantile_min}"
            ),
            "validation_high_score_positive_top10": (
                f"validation rows with {target_column} > 0 and score_quantile >= {top_score_quantile_min}"
            ),
        },
        "inputs": {
            "train_behavior_root": str(train_behavior_root),
            "validation_behavior_root": str(validation_behavior_root),
            "train_predictions": str(train_predictions_path),
            "validation_predictions": str(validation_predictions_path),
            "score_generalization_distribution": str(score_generalization_distribution_path),
        },
        "rows": {
            "train_behavior_rows": int(len(train_behavior)),
            "validation_behavior_rows": int(len(validation_behavior)),
            "train_prediction_rows": int(len(train_predictions)),
            "validation_prediction_rows_for_threshold": int(len(validation_predictions)),
            "selected_role_rows": int(len(selected_role_rows)),
            "role_feature_summary_rows": int(len(role_feature_summary)),
            "feature_shift_summary_rows": int(len(feature_shift_summary)),
            "migration_summary_rows": int(len(migration_summary)),
            "score_shift_summary_rows": int(len(score_shift_summary)),
            "train_missing_behavior_rows": int(train_enriched["missing_behavior_features"].sum()),
            "validation_missing_behavior_rows": int(validation_enriched["missing_behavior_features"].sum()),
        },
        "behavior_features": list(BEHAVIOR_FEATURE_COLUMNS),
        "outputs": {
            "role_rows": str(role_rows_path),
            "role_feature_summary": str(role_feature_summary_path),
            "feature_shift_summary": str(feature_shift_summary_path),
            "family_fe_ratio_migration": str(migration_summary_path),
            "score_shift_summary": str(score_shift_summary_path),
            "summary": str(summary_path),
        },
        "data_leakage_check": {
            "uses_original_behavior_feature_files": True,
            "uses_existing_model_scores_only": True,
            "uses_existing_score_generalization_output": True,
            "models_retrained": False,
            "threshold_selected_from_validation": False,
            "ela_features_used_as_decision_input": False,
            "metadata_used_only_for_grouping": True,
            "original_utility_labels_modified": False,
        },
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote feature shift role rows to {role_rows_path}")
    print(f"wrote feature shift role-feature summary to {role_feature_summary_path}")
    print(f"wrote feature shift pairwise summary to {feature_shift_summary_path}")
    print(f"wrote feature shift family/FE-ratio migration to {migration_summary_path}")
    print(f"wrote feature shift score summary to {score_shift_summary_path}")
    print(f"wrote feature shift summary to {summary_path}")
    return summary


def _read_behavior_rows(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("*/dimension_*/behavior.parquet"))
    if not paths:
        raise ValueError(f"no behavior.parquet files found under {root}")
    frames = [pq.read_table(path).to_pandas() for path in paths]
    result = pd.concat(frames, ignore_index=True)
    return result.rename(columns={"algorithm": "prefix_algorithm"})


def _check_quantile(value: float) -> None:
    if not (0.0 < value < 1.0):
        raise ValueError("top_score_quantile_min must be in (0, 1)")


def _check_behavior(frame: pd.DataFrame) -> None:
    required = set(MERGE_KEY_COLUMNS + list(BEHAVIOR_FEATURE_COLUMNS))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"behavior rows missing columns: {missing}")


def _check_predictions(frame: pd.DataFrame, target_column: str, *, require_threshold: bool) -> None:
    required = {
        "model_name",
        "model_family",
        "score_semantics",
        "label_source",
        "default_algorithm",
        "selected_algorithm",
        "decision_score",
        target_column,
        *MERGE_KEY_COLUMNS,
    }
    if require_threshold:
        required.add("threshold_mode")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"prediction rows missing columns: {missing}")


def _enrich_predictions(
    *,
    predictions: pd.DataFrame,
    behavior: pd.DataFrame,
    split_name: str,
    target_column: str,
    top_score_quantile_min: float,
) -> pd.DataFrame:
    columns = [
        "model_name",
        "model_family",
        "score_semantics",
        *MERGE_KEY_COLUMNS,
        "label_source",
        "default_algorithm",
        "selected_algorithm",
        "decision_score",
        target_column,
    ]
    if "threshold_mode" in predictions.columns:
        columns.insert(3, "threshold_mode")
    enriched = predictions[columns].merge(
        behavior[[*MERGE_KEY_COLUMNS, *BEHAVIOR_FEATURE_COLUMNS]],
        on=MERGE_KEY_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    enriched.insert(0, "split_for_shift", split_name)
    enriched["comparison_domain"] = enriched["label_source"].map(
        {
            "changed_algorithm": "changed_algorithm",
            "same_algorithm": "same_algorithm",
        }
    )
    enriched = enriched[enriched["comparison_domain"].isin(COMPARISON_DOMAINS)].copy()
    ranked_frames = []
    for _, subset in enriched.groupby(["model_name", "comparison_domain"], dropna=False):
        ranked_frames.append(_add_score_ranks(subset))
    enriched = pd.concat(ranked_frames, ignore_index=True)
    positive = enriched[target_column].to_numpy(dtype=float) > 0.0
    high_score = enriched["score_quantile"].to_numpy(dtype=float) >= top_score_quantile_min
    if split_name == "train":
        roles = np.select(
            [positive & high_score, positive & ~high_score],
            ["train_high_capture_positive_top10", "train_low_score_positive_reference"],
            default="other_train_rows",
        )
    else:
        roles = np.select(
            [positive & high_score, positive & ~high_score],
            ["validation_high_score_positive_top10", "validation_low_capture_positive_below_top10"],
            default="other_validation_rows",
        )
    enriched["shift_role"] = roles
    enriched["missing_behavior_features"] = enriched[list(BEHAVIOR_FEATURE_COLUMNS)].isna().all(axis=1)
    return enriched


def _role_feature_summary(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows = []
    for layer, columns in GROUP_LAYERS.items():
        group_columns = ["split_for_shift", "comparison_domain", "model_name", "shift_role", *columns]
        for group_values, subset in frame.groupby(group_columns, dropna=False):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            group = dict(zip(group_columns, group_values, strict=True))
            base = {
                "split_for_shift": group["split_for_shift"],
                "comparison_domain": group["comparison_domain"],
                "model_name": group["model_name"],
                "shift_role": group["shift_role"],
                "layer": layer,
                "group": _group_label({column: group[column] for column in columns}),
                "family": group.get("family"),
                "dimension": group.get("dimension"),
                "FE_ratio": group.get("FE_ratio"),
                "problem_id": group.get("problem_id"),
                "rows": int(len(subset)),
                "utility_sum": float(subset[target_column].sum()),
                "score_mean": float(subset["decision_score"].mean()),
                "score_quantile_median": float(subset["score_quantile"].median()),
            }
            for feature in BEHAVIOR_FEATURE_COLUMNS:
                rows.append({**base, "feature": feature, **_numeric_stats(subset[feature])})
    return pd.DataFrame(rows)


def _feature_shift_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    train_role = "train_high_capture_positive_top10"
    validation_role = "validation_low_capture_positive_below_top10"
    for comparison_domain, domain_frame in frame.groupby("comparison_domain", dropna=False):
        for model_name, model_frame in domain_frame.groupby("model_name", dropna=False):
            train = model_frame[model_frame["shift_role"] == train_role]
            validation = model_frame[model_frame["shift_role"] == validation_role]
            for feature in BEHAVIOR_FEATURE_COLUMNS:
                rows.append(
                    _shift_row(
                        comparison_domain=str(comparison_domain),
                        model_name=str(model_name),
                        feature=feature,
                        train=train[feature],
                        validation=validation[feature],
                    )
                )
    return pd.DataFrame(rows)


def _score_shift_summary(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows = []
    for group_values, subset in frame.groupby(["split_for_shift", "comparison_domain", "model_name", "shift_role"]):
        split_for_shift, comparison_domain, model_name, shift_role = group_values
        rows.append(
            {
                "split_for_shift": split_for_shift,
                "comparison_domain": comparison_domain,
                "model_name": model_name,
                "shift_role": shift_role,
                "rows": int(len(subset)),
                "utility_sum": float(subset[target_column].sum()),
                "decision_score_mean": float(subset["decision_score"].mean()),
                "decision_score_median": float(subset["decision_score"].median()),
                "score_rank_median": float(subset["score_rank"].median()),
                "score_quantile_median": float(subset["score_quantile"].median()),
            }
        )
    return pd.DataFrame(rows)


def _migration_summary(score_distribution: pd.DataFrame) -> pd.DataFrame:
    required = {
        "split",
        "comparison_domain",
        "model_name",
        "layer",
        "group",
        "rows",
        "utility_gt_zero_rows",
        "utility_gt_zero_rate",
        "positive_row_share_within_layer",
        "positive_score_quantile_median",
    }
    missing = sorted(required.difference(score_distribution.columns))
    if missing:
        raise ValueError(f"score generalization positive distribution missing columns: {missing}")
    rows = []
    for layer in ["family", "fe_ratio", "family_fe_ratio", "problem_id"]:
        if layer not in set(score_distribution["layer"].astype(str).unique()):
            continue
        layer_frame = score_distribution[
            (score_distribution["layer"] == layer) & (score_distribution["utility_gt_zero_rows"] > 0)
        ]
        for (comparison_domain, model_name), subset in layer_frame.groupby(["comparison_domain", "model_name"]):
            train_top = _top_distribution_group(subset[subset["split"] == "train"])
            validation_top = _top_distribution_group(subset[subset["split"] == "validation"])
            rows.append(
                {
                    "comparison_domain": comparison_domain,
                    "model_name": model_name,
                    "layer": layer,
                    **_prefixed_group(train_top, "train"),
                    **_prefixed_group(validation_top, "validation"),
                    "positive_score_quantile_median_delta": (
                        _value_or_nan(validation_top, "positive_score_quantile_median")
                        - _value_or_nan(train_top, "positive_score_quantile_median")
                    ),
                    "utility_gt_zero_rate_delta": (
                        _value_or_nan(validation_top, "utility_gt_zero_rate")
                        - _value_or_nan(train_top, "utility_gt_zero_rate")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _top_distribution_group(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    ordered = frame.sort_values(
        ["positive_row_share_within_layer", "utility_gt_zero_rate", "utility_gt_zero_rows"],
        ascending=False,
    )
    return ordered.iloc[0]


def _prefixed_group(row: pd.Series | None, prefix: str) -> dict[str, Any]:
    if row is None:
        return {
            f"{prefix}_group": None,
            f"{prefix}_rows": None,
            f"{prefix}_utility_gt_zero_rows": None,
            f"{prefix}_utility_gt_zero_rate": None,
            f"{prefix}_positive_row_share_within_layer": None,
            f"{prefix}_positive_score_quantile_median": None,
        }
    return {
        f"{prefix}_group": row["group"],
        f"{prefix}_rows": int(row["rows"]),
        f"{prefix}_utility_gt_zero_rows": int(row["utility_gt_zero_rows"]),
        f"{prefix}_utility_gt_zero_rate": float(row["utility_gt_zero_rate"]),
        f"{prefix}_positive_row_share_within_layer": float(row["positive_row_share_within_layer"]),
        f"{prefix}_positive_score_quantile_median": float(row["positive_score_quantile_median"]),
    }


def _value_or_nan(row: pd.Series | None, column: str) -> float:
    if row is None:
        return float("nan")
    return float(row[column])


def _shift_row(
    *,
    comparison_domain: str,
    model_name: str,
    feature: str,
    train: pd.Series,
    validation: pd.Series,
) -> dict[str, Any]:
    train_stats = _numeric_stats(train)
    validation_stats = _numeric_stats(validation)
    pooled_scale = _pooled_std(train, validation)
    mean_delta = _delta(validation_stats["mean"], train_stats["mean"])
    median_delta = _delta(validation_stats["median"], train_stats["median"])
    return {
        "comparison_domain": comparison_domain,
        "model_name": model_name,
        "feature": feature,
        **{f"train_{key}": value for key, value in train_stats.items()},
        **{f"validation_{key}": value for key, value in validation_stats.items()},
        "validation_minus_train_mean": mean_delta,
        "validation_minus_train_median": median_delta,
        "standardized_mean_delta": mean_delta / pooled_scale if mean_delta is not None and pooled_scale > 0.0 else None,
    }


def _numeric_stats(values: pd.Series) -> dict[str, float | int | None]:
    numeric = values.dropna().astype(float)
    if numeric.empty:
        return {
            "rows": 0,
            "mean": None,
            "std": None,
            "median": None,
            "q25": None,
            "q75": None,
            "min": None,
            "max": None,
        }
    return {
        "rows": int(len(numeric)),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)),
        "median": float(numeric.median()),
        "q25": float(numeric.quantile(0.25)),
        "q75": float(numeric.quantile(0.75)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def _pooled_std(left: pd.Series, right: pd.Series) -> float:
    left_values = left.dropna().astype(float).to_numpy()
    right_values = right.dropna().astype(float).to_numpy()
    if left_values.size == 0 or right_values.size == 0:
        return 0.0
    left_var = float(np.var(left_values))
    right_var = float(np.var(right_values))
    return float(np.sqrt((left_var + right_var) / 2.0))


def _delta(left: float | int | None, right: float | int | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _group_label(group: dict[str, Any]) -> str:
    if not group:
        return "overall"
    return "|".join(f"{key}={value}" for key, value in group.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose min-support behavior feature distribution shift.")
    parser.add_argument(
        "--train-behavior-root",
        type=Path,
        default=Path("results/phase1/min_support_bbob_train"),
    )
    parser.add_argument(
        "--validation-behavior-root",
        type=Path,
        default=Path("results/phase1/min_support_bbob_validation"),
    )
    parser.add_argument(
        "--train-predictions",
        type=Path,
        default=Path("results/decision/min_support/model_sensitivity/model_sensitivity_train_predictions.parquet"),
    )
    parser.add_argument(
        "--validation-predictions",
        type=Path,
        default=Path("results/decision/min_support/model_sensitivity/model_sensitivity_predictions.parquet"),
    )
    parser.add_argument(
        "--score-generalization-distribution",
        type=Path,
        default=Path(
            "results/decision/min_support/score_generalization/score_generalization_positive_distribution.parquet"
        ),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--validation-threshold-mode", default="zero")
    parser.add_argument("--top-score-quantile-min", type=float, default=0.90)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/feature_distribution_shift"),
    )
    args = parser.parse_args()
    run_feature_distribution_shift_diagnostics(
        train_behavior_root=args.train_behavior_root,
        validation_behavior_root=args.validation_behavior_root,
        train_predictions_path=args.train_predictions,
        validation_predictions_path=args.validation_predictions,
        score_generalization_distribution_path=args.score_generalization_distribution,
        output_dir=args.output_dir,
        target_column=args.target_column,
        validation_threshold_mode=args.validation_threshold_mode,
        top_score_quantile_min=args.top_score_quantile_min,
    )


if __name__ == "__main__":
    main()
