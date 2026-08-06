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
TARGET_FAMILY = "bbob_f024"
TARGET_FE_RATIO = 0.5
TARGET_LABEL_SOURCE = "changed_algorithm"
FEATURE_SETS = {
    "all_behavior_features": list(BEHAVIOR_FEATURE_COLUMNS),
    "without_fe_ratio": [column for column in BEHAVIOR_FEATURE_COLUMNS if column != "bf_fe_ratio"],
}
NEIGHBOR_K_VALUES = [5, 10, 25]


def run_calibration_opportunity_diagnostics(
    *,
    train_behavior_root: Path,
    validation_behavior_root: Path,
    train_predictions_path: Path,
    validation_predictions_path: Path,
    feature_shift_pairwise_path: Path,
    feature_shift_migration_path: Path,
    output_dir: Path,
    target_column: str,
    validation_threshold_mode: str,
    early_stage_max_fe_ratio: float,
    late_stage_min_fe_ratio: float,
) -> dict[str, Any]:
    _check_target(target_column)
    train_behavior = _read_behavior_rows(train_behavior_root)
    validation_behavior = _read_behavior_rows(validation_behavior_root)
    train_predictions = pq.read_table(train_predictions_path).to_pandas()
    validation_predictions = pq.read_table(validation_predictions_path).to_pandas()
    feature_shift_pairwise = pq.read_table(feature_shift_pairwise_path).to_pandas()
    feature_shift_migration = pq.read_table(feature_shift_migration_path).to_pandas()
    _check_behavior(train_behavior)
    _check_behavior(validation_behavior)
    _check_predictions(train_predictions, target_column, require_threshold=False)
    _check_predictions(validation_predictions, target_column, require_threshold=True)

    validation_predictions = validation_predictions[
        validation_predictions["threshold_mode"] == validation_threshold_mode
    ].copy()
    train = _enrich_predictions(train_predictions, train_behavior, target_column)
    validation = _enrich_predictions(validation_predictions, validation_behavior, target_column)
    train = _with_stage(train, early_stage_max_fe_ratio, late_stage_min_fe_ratio)
    validation = _with_stage(validation, early_stage_max_fe_ratio, late_stage_min_fe_ratio)

    target_rows = validation[
        (validation["label_source"] == TARGET_LABEL_SOURCE)
        & (validation["family"] == TARGET_FAMILY)
        & np.isclose(validation["FE_ratio"].to_numpy(dtype=float), TARGET_FE_RATIO)
        & (validation[target_column] > 0.0)
    ].copy()
    if target_rows.empty:
        raise ValueError("no target validation rows found for changed_algorithm bbob_f024 FE_ratio=0.5 with U_ELA > 0")

    stage_summary = _stage_threshold_summary(train, target_rows, target_column)
    neighbor_summary, neighbor_rows = _neighbor_opportunity_summary(train, target_rows, target_column)
    target_summary = _target_area_summary(target_rows, target_column)
    feature_shift_context = _feature_shift_context(feature_shift_pairwise, feature_shift_migration)

    output_dir.mkdir(parents=True, exist_ok=True)
    stage_summary_path = output_dir / "calibration_stage_threshold_summary.parquet"
    neighbor_summary_path = output_dir / "calibration_neighbor_summary.parquet"
    neighbor_rows_path = output_dir / "calibration_neighbor_rows.parquet"
    target_summary_path = output_dir / "calibration_target_area_summary.parquet"
    feature_shift_context_path = output_dir / "calibration_feature_shift_context.parquet"
    summary_path = output_dir / "calibration_opportunity_summary.json"
    pq.write_table(pa.Table.from_pandas(stage_summary, preserve_index=False), stage_summary_path)
    pq.write_table(pa.Table.from_pandas(neighbor_summary, preserve_index=False), neighbor_summary_path)
    pq.write_table(pa.Table.from_pandas(neighbor_rows, preserve_index=False), neighbor_rows_path)
    pq.write_table(pa.Table.from_pandas(target_summary, preserve_index=False), target_summary_path)
    pq.write_table(pa.Table.from_pandas(feature_shift_context, preserve_index=False), feature_shift_context_path)

    summary = {
        "experiment": "min_support_calibration_opportunity_diagnostics",
        "research_question": (
            "Can the changed_algorithm validation U_ELA > 0 rows in bbob_f024 at FE_ratio=0.5 be covered by "
            "stage-wise score calibration or by train rows with similar behavior features?"
        ),
        "target_area": {
            "label_source": TARGET_LABEL_SOURCE,
            "family": TARGET_FAMILY,
            "FE_ratio": TARGET_FE_RATIO,
            "condition": f"{target_column} > 0",
        },
        "target_column": target_column,
        "validation_threshold_mode": validation_threshold_mode,
        "stage_definition": {
            "early_stage": f"FE_ratio <= {early_stage_max_fe_ratio}",
            "middle_stage": f"{early_stage_max_fe_ratio} < FE_ratio < {late_stage_min_fe_ratio}",
            "late_stage": f"FE_ratio >= {late_stage_min_fe_ratio}",
        },
        "feature_sets": FEATURE_SETS,
        "neighbor_k_values": NEIGHBOR_K_VALUES,
        "inputs": {
            "train_behavior_root": str(train_behavior_root),
            "validation_behavior_root": str(validation_behavior_root),
            "train_predictions": str(train_predictions_path),
            "validation_predictions": str(validation_predictions_path),
            "feature_shift_pairwise": str(feature_shift_pairwise_path),
            "feature_shift_migration": str(feature_shift_migration_path),
        },
        "rows": {
            "train_behavior_rows": int(len(train_behavior)),
            "validation_behavior_rows": int(len(validation_behavior)),
            "train_prediction_rows": int(len(train_predictions)),
            "validation_prediction_rows_for_threshold": int(len(validation_predictions)),
            "target_area_rows": int(len(target_rows)),
            "target_area_unique_state_rows": int(len(target_rows[MERGE_KEY_COLUMNS].drop_duplicates())),
            "stage_summary_rows": int(len(stage_summary)),
            "neighbor_summary_rows": int(len(neighbor_summary)),
            "neighbor_rows": int(len(neighbor_rows)),
            "target_summary_rows": int(len(target_summary)),
            "feature_shift_context_rows": int(len(feature_shift_context)),
            "train_missing_behavior_rows": int(train["missing_behavior_features"].sum()),
            "validation_missing_behavior_rows": int(validation["missing_behavior_features"].sum()),
        },
        "outputs": {
            "stage_threshold_summary": str(stage_summary_path),
            "neighbor_summary": str(neighbor_summary_path),
            "neighbor_rows": str(neighbor_rows_path),
            "target_area_summary": str(target_summary_path),
            "feature_shift_context": str(feature_shift_context_path),
            "summary": str(summary_path),
        },
        "data_leakage_check": {
            "uses_existing_model_scores_only": True,
            "uses_existing_behavior_features_only": True,
            "uses_feature_distribution_shift_output_for_context": True,
            "models_retrained": False,
            "threshold_selected_from_validation": False,
            "ela_features_used_as_decision_input": False,
            "metadata_used_only_for_grouping": True,
            "original_utility_labels_modified": False,
        },
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote calibration stage threshold summary to {stage_summary_path}")
    print(f"wrote calibration neighbor summary to {neighbor_summary_path}")
    print(f"wrote calibration neighbor rows to {neighbor_rows_path}")
    print(f"wrote calibration target area summary to {target_summary_path}")
    print(f"wrote calibration feature shift context to {feature_shift_context_path}")
    print(f"wrote calibration opportunity summary to {summary_path}")
    return summary


def _read_behavior_rows(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("*/dimension_*/behavior.parquet"))
    if not paths:
        raise ValueError(f"no behavior.parquet files found under {root}")
    frames = [pq.read_table(path).to_pandas() for path in paths]
    return pd.concat(frames, ignore_index=True).rename(columns={"algorithm": "prefix_algorithm"})


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


def _enrich_predictions(predictions: pd.DataFrame, behavior: pd.DataFrame, target_column: str) -> pd.DataFrame:
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
    ranked_frames = []
    for _, subset in enriched.groupby(["model_name", "label_source"], dropna=False):
        ranked_frames.append(_add_score_ranks(subset))
    enriched = pd.concat(ranked_frames, ignore_index=True)
    enriched["missing_behavior_features"] = enriched[list(BEHAVIOR_FEATURE_COLUMNS)].isna().all(axis=1)
    return enriched


def _with_stage(frame: pd.DataFrame, early_stage_max_fe_ratio: float, late_stage_min_fe_ratio: float) -> pd.DataFrame:
    result = frame.copy()
    fe_ratio = result["FE_ratio"].to_numpy(dtype=float)
    result["stage"] = np.select(
        [fe_ratio <= early_stage_max_fe_ratio, fe_ratio >= late_stage_min_fe_ratio],
        ["early_stage", "late_stage"],
        default="middle_stage",
    )
    return result


def _stage_threshold_summary(train: pd.DataFrame, target_rows: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows = []
    for model_name, model_target in target_rows.groupby("model_name", dropna=False):
        model_train = train[(train["model_name"] == model_name) & (train["label_source"] == TARGET_LABEL_SOURCE)]
        target_stage = str(model_target["stage"].iloc[0])
        stage_train = model_train[model_train["stage"] == target_stage]
        for threshold_scope, threshold_train in [
            ("changed_algorithm_all_train", model_train),
            ("changed_algorithm_same_stage_train", stage_train),
        ]:
            threshold, train_policy_mean = _decision_threshold_from_scores(
                threshold_train["decision_score"].to_numpy(dtype=float),
                threshold_train[target_column].to_numpy(dtype=float),
            )
            called = model_target["decision_score"].to_numpy(dtype=float) > threshold
            observed = model_target[target_column].to_numpy(dtype=float)
            rows.append(
                {
                    "model_name": model_name,
                    "model_family": str(model_target["model_family"].iloc[0]),
                    "score_semantics": str(model_target["score_semantics"].iloc[0]),
                    "threshold_scope": threshold_scope,
                    "target_stage": target_stage,
                    "threshold": float(threshold),
                    "train_rows_for_threshold": int(len(threshold_train)),
                    "train_utility_gt_zero_rows_for_threshold": int((threshold_train[target_column] > 0.0).sum()),
                    "train_policy_mean_utility": float(train_policy_mean),
                    "target_rows": int(len(model_target)),
                    "target_score_mean": float(model_target["decision_score"].mean()),
                    "target_score_median": float(model_target["decision_score"].median()),
                    "target_score_quantile_median": float(model_target["score_quantile"].median()),
                    "target_called_rows": int(called.sum()),
                    "target_capture_rate": float(called.mean()),
                    "target_captured_utility_sum": float(observed[called].sum()),
                    "target_utility_sum": float(observed.sum()),
                }
            )
    return pd.DataFrame(rows)


def _neighbor_opportunity_summary(
    train: pd.DataFrame,
    target_rows: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    neighbor_row_frames = []
    for model_name, model_target in target_rows.groupby("model_name", dropna=False):
        model_train = train[(train["model_name"] == model_name) & (train["label_source"] == TARGET_LABEL_SOURCE)]
        for pool_name, pool in [
            ("changed_algorithm_all_train", model_train),
            ("changed_algorithm_late_stage_train", model_train[model_train["stage"] == "late_stage"]),
            (
                "changed_algorithm_positive_train",
                model_train[model_train[target_column] > 0.0],
            ),
        ]:
            if pool.empty:
                continue
            for feature_set_name, feature_columns in FEATURE_SETS.items():
                distances = _distance_matrix(
                    pool=pool,
                    target=model_target,
                    feature_columns=feature_columns,
                )
                for k in NEIGHBOR_K_VALUES:
                    k_eff = min(k, len(pool))
                    neighbor_positions = np.argsort(distances, axis=0)[:k_eff, :]
                    flat_neighbor_positions = np.unique(neighbor_positions.reshape(-1))
                    neighbor_rows = pool.iloc[flat_neighbor_positions].copy()
                    neighbor_rows.insert(0, "neighbor_pool", pool_name)
                    neighbor_rows.insert(1, "feature_set", feature_set_name)
                    neighbor_rows.insert(2, "neighbor_k", k)
                    neighbor_rows.insert(3, "target_model_name", model_name)
                    neighbor_row_frames.append(neighbor_rows)

                    per_target_positive_rates = []
                    per_target_mean_utilities = []
                    per_target_median_distances = []
                    for target_index in range(neighbor_positions.shape[1]):
                        target_neighbors = pool.iloc[neighbor_positions[:, target_index]]
                        per_target_positive_rates.append(float((target_neighbors[target_column] > 0.0).mean()))
                        per_target_mean_utilities.append(float(target_neighbors[target_column].mean()))
                        per_target_median_distances.append(float(np.median(distances[neighbor_positions[:, target_index], target_index])))
                    observed = model_target[target_column].to_numpy(dtype=float)
                    summary_rows.append(
                        {
                            "model_name": model_name,
                            "model_family": str(model_target["model_family"].iloc[0]),
                            "score_semantics": str(model_target["score_semantics"].iloc[0]),
                            "neighbor_pool": pool_name,
                            "feature_set": feature_set_name,
                            "neighbor_k": int(k),
                            "candidate_train_rows": int(len(pool)),
                            "candidate_train_utility_gt_zero_rows": int((pool[target_column] > 0.0).sum()),
                            "target_rows": int(len(model_target)),
                            "target_utility_sum": float(observed.sum()),
                            "target_score_quantile_median": float(model_target["score_quantile"].median()),
                            "nearest_unique_train_rows": int(len(neighbor_rows)),
                            "nearest_unique_train_utility_gt_zero_rows": int((neighbor_rows[target_column] > 0.0).sum()),
                            "nearest_unique_train_utility_gt_zero_rate": float((neighbor_rows[target_column] > 0.0).mean()),
                            "nearest_unique_train_mean_utility": float(neighbor_rows[target_column].mean()),
                            "per_target_neighbor_positive_rate_mean": float(np.mean(per_target_positive_rates)),
                            "per_target_neighbor_positive_rate_median": float(np.median(per_target_positive_rates)),
                            "per_target_neighbor_mean_utility_mean": float(np.mean(per_target_mean_utilities)),
                            "per_target_neighbor_distance_median": float(np.median(per_target_median_distances)),
                            "nearest_unique_train_score_quantile_median": float(neighbor_rows["score_quantile"].median()),
                            "nearest_unique_train_score_median": float(neighbor_rows["decision_score"].median()),
                        }
                    )
    neighbor_rows = pd.concat(neighbor_row_frames, ignore_index=True) if neighbor_row_frames else pd.DataFrame()
    return pd.DataFrame(summary_rows), neighbor_rows


def _target_area_summary(target_rows: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows = []
    for model_name, subset in target_rows.groupby("model_name", dropna=False):
        observed = subset[target_column].to_numpy(dtype=float)
        rows.append(
            {
                "model_name": model_name,
                "model_family": str(subset["model_family"].iloc[0]),
                "score_semantics": str(subset["score_semantics"].iloc[0]),
                "rows": int(len(subset)),
                "utility_sum": float(observed.sum()),
                "utility_mean": float(observed.mean()),
                "decision_score_mean": float(subset["decision_score"].mean()),
                "decision_score_median": float(subset["decision_score"].median()),
                "score_quantile_median": float(subset["score_quantile"].median()),
                "score_quantile_min": float(subset["score_quantile"].min()),
                "score_quantile_max": float(subset["score_quantile"].max()),
                "top10_score_rows": int((subset["score_quantile"] >= 0.9).sum()),
                "top10_score_capture_rate": float((subset["score_quantile"] >= 0.9).mean()),
            }
        )
    return pd.DataFrame(rows)


def _feature_shift_context(pairwise: pd.DataFrame, migration: pd.DataFrame) -> pd.DataFrame:
    pair_required = {"comparison_domain", "model_name", "feature", "standardized_mean_delta"}
    migration_required = {"comparison_domain", "model_name", "layer", "train_group", "validation_group"}
    pair_missing = sorted(pair_required.difference(pairwise.columns))
    migration_missing = sorted(migration_required.difference(migration.columns))
    if pair_missing:
        raise ValueError(f"feature shift pairwise missing columns: {pair_missing}")
    if migration_missing:
        raise ValueError(f"feature shift migration missing columns: {migration_missing}")
    pair_context = pairwise[pairwise["comparison_domain"] == TARGET_LABEL_SOURCE].copy()
    pair_context["context_type"] = "feature_shift"
    migration_context = migration[migration["comparison_domain"] == TARGET_LABEL_SOURCE].copy()
    migration_context["context_type"] = "region_migration"
    return pd.concat(
        [
            pair_context.assign(layer=None, train_group=None, validation_group=None),
            migration_context.assign(feature=None, standardized_mean_delta=None),
        ],
        ignore_index=True,
        sort=False,
    )


def _decision_threshold_from_scores(scores: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    if scores.size == 0:
        return 0.0, 0.0
    candidates = np.unique(np.concatenate(([0.0], scores)))
    best_threshold = 0.0
    best_utility = -np.inf
    for threshold in candidates:
        utility = np.where(scores > threshold, observed, 0.0)
        mean_utility = float(np.mean(utility))
        if mean_utility > best_utility:
            best_utility = mean_utility
            best_threshold = float(threshold)
    return best_threshold, float(best_utility)


def _distance_matrix(pool: pd.DataFrame, target: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    pool_values = pool[feature_columns].astype(float)
    target_values = target[feature_columns].astype(float)
    medians = pool_values.median(axis=0).fillna(0.0)
    scales = pool_values.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    pool_scaled = ((pool_values.fillna(medians) - medians) / scales).to_numpy(dtype=float)
    target_scaled = ((target_values.fillna(medians) - medians) / scales).to_numpy(dtype=float)
    diff = pool_scaled[:, None, :] - target_scaled[None, :, :]
    return np.sqrt(np.mean(diff * diff, axis=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose min-support calibration opportunity without retraining.")
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
        "--feature-shift-pairwise",
        type=Path,
        default=Path("results/decision/min_support/feature_distribution_shift/feature_shift_pairwise_summary.parquet"),
    )
    parser.add_argument(
        "--feature-shift-migration",
        type=Path,
        default=Path(
            "results/decision/min_support/feature_distribution_shift/feature_shift_family_fe_ratio_migration.parquet"
        ),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--validation-threshold-mode", default="zero")
    parser.add_argument("--early-stage-max-fe-ratio", type=float, default=0.2)
    parser.add_argument("--late-stage-min-fe-ratio", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/calibration_opportunity"),
    )
    args = parser.parse_args()
    run_calibration_opportunity_diagnostics(
        train_behavior_root=args.train_behavior_root,
        validation_behavior_root=args.validation_behavior_root,
        train_predictions_path=args.train_predictions,
        validation_predictions_path=args.validation_predictions,
        feature_shift_pairwise_path=args.feature_shift_pairwise,
        feature_shift_migration_path=args.feature_shift_migration,
        output_dir=args.output_dir,
        target_column=args.target_column,
        validation_threshold_mode=args.validation_threshold_mode,
        early_stage_max_fe_ratio=args.early_stage_max_fe_ratio,
        late_stage_min_fe_ratio=args.late_stage_min_fe_ratio,
    )


if __name__ == "__main__":
    main()
