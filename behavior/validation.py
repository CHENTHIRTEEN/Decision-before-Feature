from __future__ import annotations

from math import isfinite
from pathlib import Path

import pyarrow.parquet as pq

from behavior.features import BEHAVIOR_COLUMNS, BEHAVIOR_FEATURE_COLUMNS, BEHAVIOR_METADATA_COLUMNS, extract_behavior_rows
from trajectory.validation import validate_trajectory_file


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
    "function_id",
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
