from __future__ import annotations

import argparse
import json
from math import ceil, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from decision.min_support_evaluate import DEFAULT_TARGET_COLUMN, _check_target, _json_default
from decision.min_support_f024_behavior_separability import _annotate, _load_config, _target_holdout_pairs, _target_problem_ids


EPS = 1e-12
ELITE_FRACTION = 0.20
BOUNDARY_ABS_THRESHOLD = 4.5
WINDOW_LONG = 0.10

CANDIDATE_FEATURE_DEFINITIONS: tuple[dict[str, str], ...] = (
    {
        "feature": "cf_fitness_iqr_norm",
        "definition": (
            "Population fitness interquartile range divided by max(abs(median fitness), fitness IQR, eps)."
        ),
        "calculation_window": "current checkpoint population and fitness values",
        "expected_failure_type": (
            "Missed U_ELA>0 rows where spatial diversity looks ordinary but the population still contains "
            "substantial fitness heterogeneity."
        ),
    },
    {
        "feature": "cf_elite_fitness_gap_norm",
        "definition": (
            "Median fitness minus top-20% elite median fitness, divided by max(abs(median fitness), fitness IQR, eps)."
        ),
        "calculation_window": "current checkpoint population and fitness values",
        "expected_failure_type": (
            "Rows where a small elite subset is substantially better than the population bulk, suggesting that "
            "the selector change may matter even when best-fitness progress features are ambiguous."
        ),
    },
    {
        "feature": "cf_elite_centroid_shift_norm",
        "definition": (
            "Euclidean distance between the top-20% elite centroid and the population centroid, normalized by "
            "sqrt(dimension) and current mean pairwise population distance."
        ),
        "calculation_window": "current checkpoint population and fitness values",
        "expected_failure_type": (
            "Rows where useful ELA occurs when good solutions sit away from the population mass; mean pairwise "
            "diversity alone does not encode this elite-population geometry."
        ),
    },
    {
        "feature": "cf_best_centroid_distance_norm",
        "definition": (
            "Euclidean distance between the current population-best point and the population centroid, normalized "
            "by sqrt(dimension) and current mean pairwise population distance."
        ),
        "calculation_window": "current checkpoint population and fitness values",
        "expected_failure_type": (
            "Score-ranking failures where a single promising point is spatially displaced but elite-average "
            "features are diluted by the remaining population."
        ),
    },
    {
        "feature": "cf_rank_distance_correlation",
        "definition": (
            "Spearman correlation between distance to the current population-best point and fitness value inside "
            "the population; higher values mean worse individuals tend to be farther from the current best."
        ),
        "calculation_window": "current checkpoint population and fitness values",
        "expected_failure_type": (
            "Rows where current behavior features cannot distinguish coherent local structure from mixed or "
            "misleading population-fitness structure."
        ),
    },
    {
        "feature": "cf_population_axis_anisotropy",
        "definition": (
            "Largest eigenvalue of the population coordinate covariance divided by covariance trace."
        ),
        "calculation_window": "current checkpoint population coordinates",
        "expected_failure_type": (
            "Rows with ridge-like or axis-concentrated population shapes that have similar pairwise diversity "
            "to unhelpful called rows."
        ),
    },
    {
        "feature": "cf_boundary_contact_rate",
        "definition": (
            "Fraction of population coordinates with absolute coordinate value at least 4.5 in the BBOB search box."
        ),
        "calculation_window": "current checkpoint population coordinates",
        "expected_failure_type": (
            "Rows where late-stage f024 behavior is dominated by boundary contact or clipping-like population "
            "states not represented by diversity and convergence rates."
        ),
    },
    {
        "feature": "cf_best_path_directness_w10",
        "definition": (
            "Net displacement of the population-best point from the 0.10 FE-ratio anchor to the current checkpoint "
            "divided by the cumulative population-best path length over the same checkpoint window."
        ),
        "calculation_window": "last 0.10 FE-ratio trajectory window ending at the current checkpoint",
        "expected_failure_type": (
            "Missed U_ELA>0 rows where the best point keeps moving in a directed way, versus unhelpful high-score "
            "rows with erratic or nearly stationary late-stage best-point movement."
        ),
    },
)

CANDIDATE_FEATURE_COLUMNS = tuple(row["feature"] for row in CANDIDATE_FEATURE_DEFINITIONS)

GROUP_LAYERS = {
    "overall": [],
    "family": ["family"],
    "dimension": ["dimension"],
    "FE_ratio": ["FE_ratio"],
    "problem_id": ["problem_id"],
    "family_dimension_FE_ratio_problem_id": ["family", "dimension", "FE_ratio", "problem_id"],
}


def run_f024_behavior_feature_candidate_diagnostics(
    *,
    validation_labels_path: Path,
    extension_train_labels_path: Path,
    validation_trajectory_paths: list[Path],
    extension_trajectory_paths: list[Path],
    extension_config_path: Path,
    separability_dir: Path,
    output_dir: Path,
    target_column: str,
) -> dict[str, Any]:
    _check_target(target_column)
    config = _load_config(extension_config_path)
    target_problem_ids = _target_problem_ids(config)
    target_holdout_pairs = _target_holdout_pairs(config)

    validation_labels = _annotate(
        pd.read_parquet(validation_labels_path),
        target_problem_ids=target_problem_ids,
        target_holdout_pairs=target_holdout_pairs,
        data_split="validation",
    )
    extension_labels = _annotate(
        pd.read_parquet(extension_train_labels_path),
        target_problem_ids=target_problem_ids,
        target_holdout_pairs=target_holdout_pairs,
        data_split="extension_train",
    )

    validation_candidates = _candidate_features_from_trajectories(validation_trajectory_paths)
    extension_candidates = _candidate_features_from_trajectories(extension_trajectory_paths)

    validation = _merge_candidate_features(validation_labels, validation_candidates)
    extension_train = _merge_candidate_features(extension_labels, extension_candidates)

    target_validation = _target_frame(validation, holdout_only=False)
    target_holdout = _target_frame(validation, holdout_only=True)
    target_extension = _target_frame(extension_train, holdout_only=False)

    frames = {
        "validation_target_changed_late": target_validation,
        "validation_target_holdout_changed_late": target_holdout,
        "validation_target_changed_late_fe050": target_validation[target_validation["FE_ratio"].round(6) == 0.5].copy(),
        "validation_target_holdout_changed_late_fe050": target_holdout[target_holdout["FE_ratio"].round(6) == 0.5].copy(),
        "extension_train_target_changed_late": target_extension,
    }

    candidate_values = pd.concat(
        [
            _candidate_value_frame("validation", target_validation, target_column),
            _candidate_value_frame("extension_train", target_extension, target_column),
        ],
        ignore_index=True,
    )
    distribution_overlap = _distribution_overlap_table(frames, target_column)
    single_feature_thresholds = _single_feature_threshold_table(frames, target_column)
    existing_comparison = _existing_vs_candidate_comparison(
        candidate_thresholds=single_feature_thresholds,
        existing_thresholds_path=separability_dir / "f024_single_feature_threshold_separability.parquet",
    )
    failure_candidate_values = _failure_candidate_value_table(
        separability_dir / "f024_score_ranking_failure_samples.parquet",
        validation_candidates,
        target_column,
    )
    failure_threshold_coverage = _failure_threshold_coverage_table(
        failure_candidate_values=failure_candidate_values,
        candidate_thresholds=single_feature_thresholds,
        target_column=target_column,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    values_path = output_dir / "f024_candidate_behavior_feature_values.parquet"
    distribution_path = output_dir / "f024_candidate_distribution_overlap.parquet"
    thresholds_path = output_dir / "f024_candidate_single_feature_thresholds.parquet"
    comparison_path = output_dir / "f024_candidate_vs_existing_threshold_summary.parquet"
    failure_values_path = output_dir / "f024_candidate_score_failure_values.parquet"
    failure_coverage_path = output_dir / "f024_candidate_failure_threshold_coverage.parquet"
    definitions_path = output_dir / "f024_candidate_feature_definitions.md"
    summary_path = output_dir / "f024_behavior_feature_candidate_summary.json"

    _write_parquet(candidate_values, values_path)
    _write_parquet(distribution_overlap, distribution_path)
    _write_parquet(single_feature_thresholds, thresholds_path)
    _write_parquet(existing_comparison, comparison_path)
    _write_parquet(failure_candidate_values, failure_values_path)
    _write_parquet(failure_threshold_coverage, failure_coverage_path)
    definitions_path.write_text(_definitions_markdown(), encoding="utf-8")

    summary = {
        "experiment": "min_support_f024_behavior_feature_candidate_diagnostics",
        "research_question": (
            "Which minimal algorithm-agnostic population/fitness behavior feature candidates may separate "
            "f024 changed_algorithm late-stage rows with U_ELA > 0 from rows with U_ELA <= 0?"
        ),
        "target_column": target_column,
        "candidate_features": list(CANDIDATE_FEATURE_DEFINITIONS),
        "inputs": {
            "validation_labels": str(validation_labels_path),
            "extension_train_labels": str(extension_train_labels_path),
            "validation_trajectories": [str(path) for path in validation_trajectory_paths],
            "extension_trajectories": [str(path) for path in extension_trajectory_paths],
            "extension_config": str(extension_config_path),
            "f024_behavior_separability_dir": str(separability_dir),
        },
        "rows": {
            name: {
                "rows": int(len(frame)),
                "utility_gt_zero_rows": int((frame[target_column] > 0.0).sum()),
                "candidate_complete_rows": int(frame[list(CANDIDATE_FEATURE_COLUMNS)].notna().all(axis=1).sum()),
            }
            for name, frame in frames.items()
        },
        "target_problem_ids": sorted(target_problem_ids),
        "interpretation": _interpretation(single_feature_thresholds, existing_comparison),
        "data_leakage_check": {
            "formal_models_retrained": False,
            "formal_phase1_configs_modified": False,
            "original_utility_labels_modified": False,
            "ela_features_used_as_decision_input": False,
            "function_id_used_as_decision_input": False,
            "problem_id_used_only_for_grouping": True,
            "algorithm_id_used_only_for_alignment_with_prefix_trajectory": True,
            "algorithm_internal_parameters_used": False,
            "candidate_features_computed_from_existing_trajectory_population_and_fitness": True,
            "thresholds_are_diagnostic_single_feature_summaries": True,
        },
        "outputs": {
            "candidate_feature_values": str(values_path),
            "distribution_overlap": str(distribution_path),
            "single_feature_thresholds": str(thresholds_path),
            "candidate_vs_existing_threshold_summary": str(comparison_path),
            "score_failure_candidate_values": str(failure_values_path),
            "failure_threshold_coverage": str(failure_coverage_path),
            "feature_definitions": str(definitions_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    print(f"wrote f024 candidate behavior feature values to {values_path}")
    print(f"wrote f024 candidate distribution overlap to {distribution_path}")
    print(f"wrote f024 candidate single-feature thresholds to {thresholds_path}")
    print(f"wrote f024 candidate vs existing threshold summary to {comparison_path}")
    print(f"wrote f024 candidate score failure values to {failure_values_path}")
    print(f"wrote f024 candidate failure threshold coverage to {failure_coverage_path}")
    print(f"wrote f024 candidate feature definitions to {definitions_path}")
    print(f"wrote f024 behavior feature candidate summary to {summary_path}")
    return summary


def _candidate_features_from_trajectories(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_parquet(path))
    trajectories = pd.concat(frames, ignore_index=True)
    rows = []
    for _, group in trajectories.groupby(["algorithm", "problem_id", "seed"], sort=False, dropna=False):
        ordered = group.sort_values("FE").reset_index(drop=True)
        checkpoint_stats = [_current_candidate_stats(row) for _, row in ordered.iterrows()]
        best_positions = [_best_position(row) for _, row in ordered.iterrows()]
        for index, row in ordered.iterrows():
            anchor_index = _find_anchor_index(ordered, index, WINDOW_LONG)
            candidate_row = {
                "problem_id": str(row["problem_id"]),
                "family": str(row["family"]),
                "dimension": int(row["dimension"]),
                "prefix_algorithm": str(row["algorithm"]),
                "seed": int(row["seed"]),
                "FE": int(row["FE"]),
                "FE_ratio": float(row["FE_ratio"]),
            }
            candidate_row.update(
                {
                    feature: checkpoint_stats[index][feature]
                    for feature in CANDIDATE_FEATURE_COLUMNS
                    if feature != "cf_best_path_directness_w10"
                }
            )
            candidate_row["cf_best_path_directness_w10"] = _best_path_directness(best_positions, anchor_index, index)
            rows.append(candidate_row)
    return pd.DataFrame(rows)


def _current_candidate_stats(row: pd.Series) -> dict[str, float | None]:
    population = _population(row)
    fitness = _fitness(row)
    dimension = int(row["dimension"])
    best_index = int(np.argmin(fitness))
    best = population[best_index]
    centroid = np.mean(population, axis=0)
    diversity = _mean_pairwise_distance(population, dimension)

    ordered = np.argsort(fitness)
    elite_count = max(1, int(ceil(len(fitness) * ELITE_FRACTION)))
    elite_indices = ordered[:elite_count]
    elite_population = population[elite_indices]
    elite_fitness = fitness[elite_indices]

    fitness_q25 = float(np.quantile(fitness, 0.25))
    fitness_q75 = float(np.quantile(fitness, 0.75))
    fitness_iqr = fitness_q75 - fitness_q25
    fitness_median = float(np.median(fitness))
    fitness_scale = max(abs(fitness_median), abs(fitness_q25), abs(fitness_q75), fitness_iqr, EPS)

    elite_centroid = np.mean(elite_population, axis=0)
    elite_median = float(np.median(elite_fitness))
    normalized_diversity = max(diversity, EPS)

    return {
        "cf_internal_diversity": diversity,
        "cf_fitness_iqr_norm": float(fitness_iqr / fitness_scale),
        "cf_elite_fitness_gap_norm": float((fitness_median - elite_median) / fitness_scale),
        "cf_elite_centroid_shift_norm": float(_scaled_distance(elite_centroid, centroid, dimension) / normalized_diversity),
        "cf_best_centroid_distance_norm": float(_scaled_distance(best, centroid, dimension) / normalized_diversity),
        "cf_rank_distance_correlation": _rank_distance_correlation(population, fitness, best),
        "cf_population_axis_anisotropy": _axis_anisotropy(population),
        "cf_boundary_contact_rate": float(np.mean(np.abs(population) >= BOUNDARY_ABS_THRESHOLD)),
    }


def _merge_candidate_features(labels: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["problem_id", "dimension", "prefix_algorithm", "seed", "FE", "FE_ratio"]
    return labels.merge(candidates[key_columns + list(CANDIDATE_FEATURE_COLUMNS)], on=key_columns, how="left", validate="many_to_one")


def _target_frame(frame: pd.DataFrame, *, holdout_only: bool) -> pd.DataFrame:
    mask = (
        frame["is_target_problem"].to_numpy(dtype=bool)
        & (frame["label_source"].to_numpy() == "changed_algorithm")
        & (frame["FE_ratio"].to_numpy(dtype=float) >= 0.5)
    )
    if holdout_only:
        mask &= frame["is_target_holdout_seed"].to_numpy(dtype=bool)
    return frame.loc[mask].copy()


def _candidate_value_frame(source: str, frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    columns = [
        "analysis_data_split",
        "split",
        "family",
        "problem_id",
        "dimension",
        "prefix_algorithm",
        "seed",
        "FE",
        "FE_ratio",
        "default_algorithm",
        "selected_algorithm",
        "label_source",
        "is_target_holdout_seed",
        target_column,
        *CANDIDATE_FEATURE_COLUMNS,
    ]
    result = frame.copy()
    result["analysis_data_split"] = source
    if target_column not in result.columns:
        result[target_column] = np.nan
    return result[[column for column in columns if column in result.columns]]


def _distribution_overlap_table(frames: dict[str, pd.DataFrame], target_column: str) -> pd.DataFrame:
    rows = []
    for data_split, frame in frames.items():
        for layer, group_columns in GROUP_LAYERS.items():
            grouped = [((), frame)] if not group_columns else frame.groupby(group_columns, dropna=False)
            for group_values, subset in grouped:
                group = _group_dict(group_columns, group_values)
                for feature in CANDIDATE_FEATURE_COLUMNS:
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
    non_pos_stats = _stats(non_positive)
    auc = _rank_auc(positive, non_positive)
    return {
        "data_split": data_split,
        "layer": layer,
        "group": _group_label(group),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "problem_id": group.get("problem_id"),
        "feature": feature,
        "rows": int(len(frame)),
        "positive_rows": int((frame[target_column] > 0.0).sum()),
        "non_positive_rows": int((frame[target_column] <= 0.0).sum()),
        "positive_finite_rows": int(len(positive)),
        "non_positive_finite_rows": int(len(non_positive)),
        **{f"positive_{key}": value for key, value in pos_stats.items()},
        **{f"non_positive_{key}": value for key, value in non_pos_stats.items()},
        "iqr_overlap_ratio": _interval_overlap_ratio(pos_stats["q25"], pos_stats["q75"], non_pos_stats["q25"], non_pos_stats["q75"]),
        "range_overlap_ratio": _interval_overlap_ratio(pos_stats["min"], pos_stats["max"], non_pos_stats["min"], non_pos_stats["max"]),
        "rank_auc_positive_greater": auc,
        "rank_separation": None if auc is None else float(abs(auc - 0.5) * 2.0),
        "median_gap_abs": _abs_gap(pos_stats["median"], non_pos_stats["median"]),
    }


def _single_feature_threshold_table(frames: dict[str, pd.DataFrame], target_column: str) -> pd.DataFrame:
    rows = []
    for data_split, frame in frames.items():
        for layer, group_columns in GROUP_LAYERS.items():
            grouped = [((), frame)] if not group_columns else frame.groupby(group_columns, dropna=False)
            for group_values, subset in grouped:
                group = _group_dict(group_columns, group_values)
                for feature in CANDIDATE_FEATURE_COLUMNS:
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
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "problem_id": group.get("problem_id"),
        "feature": feature,
        "rows": int(len(frame)),
        "finite_rows": int(len(finite)),
        "positive_rows": int(np.sum(frame[target_column].to_numpy(dtype=float) > 0.0)),
        "finite_positive_rows": int(np.sum(labels)) if len(finite) else 0,
    }
    if len(finite) == 0 or len(np.unique(labels)) < 2:
        return {**base, **_empty_threshold_metrics()}

    best = None
    for direction in ("greater", "less_equal"):
        for threshold in _candidate_thresholds(values):
            calls = values > threshold if direction == "greater" else values <= threshold
            metrics = _classification_metrics(calls, labels, observed)
            candidate = {"direction": direction, "threshold": float(threshold), **metrics}
            if best is None or _threshold_sort_key(candidate) > _threshold_sort_key(best):
                best = candidate
    return {**base, **best}


def _existing_vs_candidate_comparison(candidate_thresholds: pd.DataFrame, existing_thresholds_path: Path) -> pd.DataFrame:
    frames = []
    selected_columns = [
        "data_split",
        "layer",
        "group",
        "family",
        "dimension",
        "FE_ratio",
        "problem_id",
        "feature",
        "rows",
        "positive_rows",
        "direction",
        "threshold",
        "precision",
        "recall",
        "f1",
        "utility_capture_rate",
        "unhelpful_call_cost_sum",
        "decision_mean_utility",
    ]
    candidate = candidate_thresholds[selected_columns].copy()
    candidate.insert(0, "feature_source", "candidate_population_fitness_behavior")
    frames.append(candidate)
    if existing_thresholds_path.exists():
        existing = pd.read_parquet(existing_thresholds_path).copy()
        for column in ("family", "FE_ratio"):
            if column not in existing.columns:
                existing[column] = np.nan
        existing = existing[[column for column in selected_columns if column in existing.columns]].copy()
        existing.insert(0, "feature_source", "existing_min_support_behavior")
        frames.append(existing)
    return pd.concat(frames, ignore_index=True)


def _failure_candidate_value_table(failure_path: Path, validation_candidates: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if not failure_path.exists():
        return pd.DataFrame()
    failures = pd.read_parquet(failure_path).copy()
    key_columns = ["problem_id", "dimension", "prefix_algorithm", "seed", "FE", "FE_ratio"]
    merged = failures.merge(validation_candidates[key_columns + list(CANDIDATE_FEATURE_COLUMNS)], on=key_columns, how="left", validate="many_to_one")
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
        *CANDIDATE_FEATURE_COLUMNS,
    ]
    return merged[[column for column in columns if column in merged.columns]]


def _failure_threshold_coverage_table(
    *,
    failure_candidate_values: pd.DataFrame,
    candidate_thresholds: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    if failure_candidate_values.empty:
        return pd.DataFrame()
    domain_to_split = {
        "target_problem_changed_late": "validation_target_changed_late",
        "target_holdout_changed_late": "validation_target_holdout_changed_late",
        "target_problem_changed_late_fe050": "validation_target_changed_late_fe050",
        "target_holdout_changed_late_fe050": "validation_target_holdout_changed_late_fe050",
    }
    rows = []
    for eval_domain, split_name in domain_to_split.items():
        thresholds = candidate_thresholds[
            (candidate_thresholds["data_split"] == split_name)
            & (candidate_thresholds["layer"] == "overall")
            & candidate_thresholds["threshold"].notna()
        ]
        domain_failures = failure_candidate_values[failure_candidate_values["eval_domain"] == eval_domain]
        if domain_failures.empty or thresholds.empty:
            continue
        for _, threshold_row in thresholds.iterrows():
            feature = str(threshold_row["feature"])
            values = domain_failures[feature].to_numpy(dtype=float)
            finite = np.isfinite(values)
            if str(threshold_row["direction"]) == "greater":
                candidate_calls = values > float(threshold_row["threshold"])
            else:
                candidate_calls = values <= float(threshold_row["threshold"])
            for (failure_role, training_dataset, model_name), subset in domain_failures.groupby(
                ["failure_role", "training_dataset", "model_name"], dropna=False
            ):
                positions = domain_failures.index.get_indexer(subset.index)
                local_calls = candidate_calls[positions]
                local_finite = finite[positions]
                observed = subset[target_column].to_numpy(dtype=float)
                utility_gt_zero = observed > 0.0
                rows.append(
                    {
                        "eval_domain": eval_domain,
                        "failure_role": failure_role,
                        "training_dataset": training_dataset,
                        "model_name": model_name,
                        "feature": feature,
                        "direction": threshold_row["direction"],
                        "threshold": threshold_row["threshold"],
                        "failure_rows": int(len(positions)),
                        "finite_rows": int(np.sum(local_finite)),
                        "utility_gt_zero_rows": int(np.sum(utility_gt_zero)),
                        "candidate_call_rows": int(np.sum(local_calls & local_finite)),
                        "candidate_call_rate": float(np.mean(local_calls[local_finite])) if np.any(local_finite) else None,
                        "candidate_capture_utility_gt_zero_rows": int(np.sum(local_calls & local_finite & utility_gt_zero)),
                        "candidate_call_unhelpful_rows": int(np.sum(local_calls & local_finite & ~utility_gt_zero)),
                    }
                )
    return pd.DataFrame(rows)


def _population(row: pd.Series) -> np.ndarray:
    population_value = row["population"]
    population = np.vstack([np.asarray(member, dtype=float) for member in population_value])
    if population.ndim != 2:
        raise ValueError("population must be two-dimensional")
    if population.shape[1] != int(row["dimension"]):
        raise ValueError("population width must match dimension")
    return population


def _fitness(row: pd.Series) -> np.ndarray:
    fitness = np.asarray(row["fitness"], dtype=float).reshape(-1)
    if fitness.shape[0] != len(row["population"]):
        raise ValueError("fitness length must match population rows")
    return fitness


def _best_position(row: pd.Series) -> np.ndarray:
    population = _population(row)
    fitness = _fitness(row)
    return population[int(np.argmin(fitness))]


def _mean_pairwise_distance(population: np.ndarray, dimension: int) -> float:
    if population.shape[0] < 2:
        return 0.0
    deltas = population[:, None, :] - population[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    upper = distances[np.triu_indices(population.shape[0], k=1)]
    return float(np.mean(upper) / sqrt(dimension))


def _scaled_distance(left: np.ndarray, right: np.ndarray, dimension: int) -> float:
    return float(np.linalg.norm(left - right) / sqrt(dimension))


def _rank_distance_correlation(population: np.ndarray, fitness: np.ndarray, best: np.ndarray) -> float | None:
    distances = np.linalg.norm(population - best, axis=1)
    if np.allclose(distances, distances[0]) or np.allclose(fitness, fitness[0]):
        return None
    distance_ranks = _ordinal_ranks(distances)
    fitness_ranks = _ordinal_ranks(fitness)
    return _pearson_corr(distance_ranks, fitness_ranks)


def _axis_anisotropy(population: np.ndarray) -> float | None:
    if population.shape[0] < 2:
        return None
    centered = population - np.mean(population, axis=0)
    covariance = np.cov(centered, rowvar=False)
    values = np.linalg.eigvalsh(covariance)
    total = float(np.sum(np.maximum(values, 0.0)))
    if total <= EPS:
        return None
    return float(np.max(values) / total)


def _best_path_directness(best_positions: list[np.ndarray], anchor_index: int | None, current_index: int) -> float | None:
    if anchor_index is None or current_index <= anchor_index:
        return None
    segment = best_positions[anchor_index : current_index + 1]
    steps = [float(np.linalg.norm(current - previous)) for previous, current in zip(segment[:-1], segment[1:], strict=False)]
    path_length = float(np.sum(steps))
    if path_length <= EPS:
        return 0.0
    net = float(np.linalg.norm(segment[-1] - segment[0]))
    return float(net / path_length)


def _find_anchor_index(rows: pd.DataFrame, current_index: int, window: float) -> int | None:
    target = float(rows.iloc[current_index]["FE_ratio"]) - window
    anchor = None
    for index in range(current_index):
        if float(rows.iloc[index]["FE_ratio"]) <= target + EPS:
            anchor = index
        else:
            break
    return anchor


def _ordinal_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


def _pearson_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denom = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denom <= EPS:
        return None
    return float(np.dot(left_centered, right_centered) / denom)


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
    true_run_ela = int(np.sum(calls & labels))
    unhelpful_calls = int(np.sum(calls & ~labels))
    missed = int(np.sum(~calls & labels))
    skipped_unhelpful = int(np.sum(~calls & ~labels))
    precision = true_run_ela / (true_run_ela + unhelpful_calls) if true_run_ela + unhelpful_calls else 0.0
    recall = true_run_ela / (true_run_ela + missed) if true_run_ela + missed else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    positive_sum = float(np.sum(observed[labels]))
    captured_sum = float(np.sum(observed[calls & labels]))
    unhelpful = observed[calls & ~labels]
    return {
        "true_run_ela_rows": true_run_ela,
        "unhelpful_call_rows": unhelpful_calls,
        "missed_positive_rows": missed,
        "skip_when_unhelpful_rows": skipped_unhelpful,
        "decision_call_rows": int(np.sum(calls)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((true_run_ela + skipped_unhelpful) / len(labels)),
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


def _interpretation(single_feature_thresholds: pd.DataFrame, comparison: pd.DataFrame) -> dict[str, Any]:
    focus = single_feature_thresholds[
        (single_feature_thresholds["data_split"] == "validation_target_holdout_changed_late_fe050")
        & (single_feature_thresholds["layer"] == "overall")
    ].copy()
    if focus.empty:
        return {"candidate_signal_found": False, "reason": "No target holdout FE_ratio=0.50 candidate rows were available."}
    best = focus.sort_values(["f1", "decision_mean_utility", "utility_capture_rate"], ascending=False).head(1).iloc[0]
    existing_focus = comparison[
        (comparison["feature_source"] == "existing_min_support_behavior")
        & (comparison["data_split"] == "validation_target_holdout_fe050")
        & (comparison["layer"] == "overall")
    ]
    existing_best_f1 = None if existing_focus.empty else float(existing_focus["f1"].dropna().max())
    candidate_best_f1 = None if pd.isna(best["f1"]) else float(best["f1"])
    return {
        "best_candidate_feature_on_target_holdout_fe050": str(best["feature"]),
        "best_candidate_direction": None if pd.isna(best["direction"]) else str(best["direction"]),
        "best_candidate_threshold": None if pd.isna(best["threshold"]) else float(best["threshold"]),
        "best_candidate_f1": candidate_best_f1,
        "best_candidate_precision": None if pd.isna(best["precision"]) else float(best["precision"]),
        "best_candidate_recall": None if pd.isna(best["recall"]) else float(best["recall"]),
        "best_candidate_utility_capture_rate": None if pd.isna(best["utility_capture_rate"]) else float(best["utility_capture_rate"]),
        "existing_best_single_feature_f1": existing_best_f1,
        "candidate_exceeds_existing_best_f1": (
            candidate_best_f1 is not None and existing_best_f1 is not None and candidate_best_f1 > existing_best_f1
        ),
        "recommendation": _candidate_recommendation(candidate_best_f1, existing_best_f1),
    }


def _candidate_recommendation(candidate_best_f1: float | None, existing_best_f1: float | None) -> str:
    if candidate_best_f1 is None:
        return "No candidate threshold is estimable on the f024 target holdout FE_ratio=0.50 subset."
    if existing_best_f1 is None:
        return "Use candidate diagnostics as the first behavior-feature extension evidence for this subset."
    if candidate_best_f1 >= existing_best_f1 + 0.05:
        return "Promote the best candidate to the next min_support extractor investigation, using a separate train/validation check."
    if candidate_best_f1 >= existing_best_f1:
        return "Treat the best candidate as a low-cost supplementary feature; verify stability before changing any formal extractor."
    return "Current candidates do not improve single-feature separability over existing behavior features on this holdout subset."


def _definitions_markdown() -> str:
    lines = [
        "# f024 behavior feature candidates",
        "",
        "These candidates are min_support diagnostics only. They use existing trajectory population and fitness values, and do not use ELA features, function id, algorithm id, or optimizer internal parameters as Decision input.",
        "",
    ]
    for item in CANDIDATE_FEATURE_DEFINITIONS:
        lines.extend(
            [
                f"## {item['feature']}",
                "",
                f"- Definition: {item['definition']}",
                f"- Calculation window: {item['calculation_window']}",
                f"- Expected failure type: {item['expected_failure_type']}",
                "",
            ]
        )
    return "\n".join(lines)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose f024 min-support behavior feature candidates.")
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
        "--validation-trajectories",
        type=Path,
        nargs="+",
        default=[
            Path("results/phase1/min_support_bbob_validation/bbob_f024/dimension_10/trajectories.parquet"),
            Path("results/phase1/min_support_bbob_validation/bbob_f024/dimension_20/trajectories.parquet"),
        ],
    )
    parser.add_argument(
        "--extension-trajectories",
        type=Path,
        nargs="+",
        default=[
            Path("results/phase1/min_support_bbob_train_late_stage_f024_followup/bbob_f024/dimension_10/trajectories.parquet"),
            Path("results/phase1/min_support_bbob_train_late_stage_f024_followup/bbob_f024/dimension_20/trajectories.parquet"),
        ],
    )
    parser.add_argument(
        "--extension-config",
        type=Path,
        default=Path("configs/min_support_bbob_train_late_stage_f024_followup.yaml"),
    )
    parser.add_argument(
        "--separability-dir",
        type=Path,
        default=Path("results/decision/min_support/f024_behavior_separability"),
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decision/min_support/f024_behavior_feature_candidates"),
    )
    args = parser.parse_args()
    run_f024_behavior_feature_candidate_diagnostics(
        validation_labels_path=args.validation_labels,
        extension_train_labels_path=args.extension_train_labels,
        validation_trajectory_paths=args.validation_trajectories,
        extension_trajectory_paths=args.extension_trajectories,
        extension_config_path=args.extension_config,
        separability_dir=args.separability_dir,
        output_dir=args.output_dir,
        target_column=args.target_column,
    )


if __name__ == "__main__":
    main()
