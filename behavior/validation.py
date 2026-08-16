from __future__ import annotations

from math import isclose, isfinite
from pathlib import Path

import pyarrow.parquet as pq

from behavior.features import (
    BEHAVIOR_COLUMNS,
    BEHAVIOR_FEATURE_COLUMNS,
    BEHAVIOR_METADATA_COLUMNS,
    BEHAVIOR_WINDOW_METADATA_COLUMNS,
    extract_behavior_rows,
)
from trajectory.validation import validate_trajectory_file
from trajectory.window_statistics import WINDOW_RATIOS


FORBIDDEN_COLUMN_FRAGMENTS = (
    "query",
    "utility",
    "p_skip",
    "p_query",
    "cost",
    "inertia",
    "c1",
    "c2",
    "mutation",
    "strategy",
    "sigma",
    "algorithm_id",
)

BOUNDED_FEATURE_COLUMNS = (
    "bf_improvement_frequency_w02",
    "bf_covariance_spectral_concentration",
    "bf_stagnation_w10",
    "bf_centroid_shift_coherence_w05",
    "bf_fitness_quantile_improvement_fraction_w02",
    "bf_search_maturity",
    "bf_search_maturity_linear",
    "bf_population_overlap_w05",
)

NON_NEGATIVE_FEATURE_COLUMNS = (
    "bf_fe_ratio",
    "bf_diversity_mean_pairwise",
    "bf_fitness_diversity",
    "bf_fitness_diversity_rel",
    "bf_population_wasserstein_rate_w05",
    "bf_centroid_shift_rate_w05",
    "bf_fitness_wasserstein_rate_w02",
    "bf_elite_concentration",
)

CORRELATION_FEATURE_COLUMNS = (
    "bf_best_distance_fitness_corr",
)

SLOPE_FEATURE_COLUMNS = (
    "bf_best_fitness_slope_rel_w05",
)

EPS = 1e-12


def validate_behavior_rows(trajectory_rows: list[dict], behavior_rows: list[dict]) -> dict[str, int]:
    if len(trajectory_rows) != len(behavior_rows):
        raise ValueError("behavior output row count must match trajectory input row count")
    if not behavior_rows:
        raise ValueError("behavior output contains no rows")

    columns = set(behavior_rows[0])
    missing = set(BEHAVIOR_COLUMNS).difference(columns)
    if missing:
        raise ValueError(f"missing behavior columns: {sorted(missing)}")
    unexpected = columns.difference(BEHAVIOR_COLUMNS)
    if unexpected:
        raise ValueError(f"unexpected behavior columns: {sorted(unexpected)}")
    _check_forbidden_columns(columns)

    expected_rows = extract_behavior_rows([trajectory_row.copy() for trajectory_row in trajectory_rows])
    for trajectory_row, behavior_row, expected_row in zip(trajectory_rows, behavior_rows, expected_rows, strict=True):
        for column in BEHAVIOR_METADATA_COLUMNS:
            if behavior_row[column] != trajectory_row[column]:
                raise ValueError(f"metadata column {column} differs from trajectory input")
        if behavior_row["bf_fe_ratio"] != trajectory_row["FE_ratio"]:
            raise ValueError("bf_fe_ratio must match FE_ratio")
        if behavior_row["bf_diversity_mean_pairwise"] < 0.0:
            raise ValueError("bf_diversity_mean_pairwise must be non-negative")
        _check_fitness_iqr_consistency(trajectory_row, behavior_row)

        for suffix in ("w02", "w05", "w10"):
            ratio_column = f"effective_window_ratio_{suffix}"
            fe_column = f"effective_window_fe_{suffix}"
            updates_column = f"effective_native_updates_{suffix}"
            values = (behavior_row[ratio_column], behavior_row[fe_column], behavior_row[updates_column])
            if any(value is None for value in values):
                if not all(value is None for value in values):
                    raise ValueError(f"effective {suffix} window metadata must be all null or all present")
            else:
                ratio, fe, native_updates = values
                if float(ratio) <= 0.0 or int(fe) <= 0 or int(native_updates) < 0:
                    raise ValueError(f"effective {suffix} window metadata is out of range")
                if not isclose(
                    float(ratio),
                    int(fe) / int(trajectory_row["FE_total"]),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    raise ValueError(f"effective {suffix} window ratio must equal FE span / FE_total")
                nominal = WINDOW_RATIOS[suffix]
                population_size = len(trajectory_row["population"])
                target_fe = int(round(nominal * int(trajectory_row["FE_total"])))
                if not target_fe <= int(fe) < target_fe + population_size:
                    raise ValueError(f"effective {suffix} window must be aligned within one native update")

        for column in BEHAVIOR_WINDOW_METADATA_COLUMNS:
            value = behavior_row[column]
            expected = expected_row[column]
            if value is None or expected is None:
                if value is not None or expected is not None:
                    raise ValueError(f"{column} does not match the current trajectory; regenerate behavior")
            elif value != expected:
                raise ValueError(f"{column} does not match the current trajectory; regenerate behavior")

        for column in NON_NEGATIVE_FEATURE_COLUMNS:
            value = behavior_row[column]
            if value is not None and value < 0.0:
                raise ValueError(f"{column} must be non-negative or null")

        for column in BOUNDED_FEATURE_COLUMNS:
            value = behavior_row[column]
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{column} must be in [0, 1] or null")

        for column in CORRELATION_FEATURE_COLUMNS:
            value = behavior_row[column]
            if value is not None and not -1.0 <= value <= 1.0:
                raise ValueError(f"{column} must be in [-1, 1] or null")

        for column in SLOPE_FEATURE_COLUMNS:
            value = behavior_row[column]
            if value is not None and not isfinite(float(value)):
                raise ValueError(f"{column} must be finite or null")

        for column in BEHAVIOR_FEATURE_COLUMNS:
            value = behavior_row[column]
            if value is not None and not isfinite(float(value)):
                raise ValueError(f"{column} must be finite or null")
            expected = expected_row[column]
            if value is None or expected is None:
                if value is not None or expected is not None:
                    raise ValueError(f"{column} does not match the current trajectory; regenerate behavior")
            elif float(value) != float(expected):
                raise ValueError(f"{column} does not match the current trajectory; regenerate behavior")

    return {"rows": len(behavior_rows)}


def _check_fitness_iqr_consistency(trajectory_row: dict, behavior_row: dict) -> None:
    statistics = trajectory_row["window_statistics"]
    behavior_ratio = float(behavior_row["bf_fitness_diversity_rel"])
    baselines = []
    current_iqrs = []
    ratios = []
    for item in statistics:
        baseline = float(item["fitness_iqr_baseline"])
        current_iqr = float(item["fitness_iqr_current"])
        ratio = float(item["fitness_iqr_rel"])
        values = (baseline, current_iqr, ratio)
        if not all(isfinite(value) for value in values):
            raise ValueError("fitness IQR window statistics must be finite")
        if baseline < 0.0 or current_iqr < 0.0 or ratio < 0.0:
            raise ValueError("fitness IQR window statistics must be non-negative")
        expected_ratio = current_iqr / max(baseline, EPS)
        if not isclose(ratio, expected_ratio, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("fitness_iqr_rel must equal current IQR / initialization IQR")
        baselines.append(baseline)
        current_iqrs.append(current_iqr)
        ratios.append(ratio)

    for values, name in (
        (baselines, "fitness_iqr_baseline"),
        (current_iqrs, "fitness_iqr_current"),
        (ratios, "fitness_iqr_rel"),
    ):
        if not all(isclose(value, values[0], rel_tol=1e-12, abs_tol=1e-12) for value in values[1:]):
            raise ValueError(f"{name} must be identical across w02/w05/w10")
    if not isclose(behavior_ratio, ratios[0], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("bf_fitness_diversity_rel must equal trajectory fitness_iqr_rel")

    history_ratio = float(trajectory_row["native_update_history"][-1]["fitness_iqr_rel"])
    if not isclose(behavior_ratio, history_ratio, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("bf_fitness_diversity_rel must equal the current native-update IQR ratio")


def validate_behavior_file(input_path: str | Path, output_path: str | Path) -> dict[str, int]:
    validate_trajectory_file(input_path)
    trajectory_rows = pq.read_table(input_path).to_pylist()
    behavior_rows = pq.read_table(output_path).to_pylist()
    return validate_behavior_rows(trajectory_rows, behavior_rows)


def _check_forbidden_columns(columns: set[str]) -> None:
    for column in columns:
        lowered = column.lower()
        for fragment in FORBIDDEN_COLUMN_FRAGMENTS:
            if fragment in lowered:
                raise ValueError(f"forbidden behavior output column: {column}")
