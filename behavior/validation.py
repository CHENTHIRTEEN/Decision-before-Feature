from __future__ import annotations

from math import isfinite
from pathlib import Path

import pyarrow.parquet as pq

from behavior.features import BEHAVIOR_COLUMNS, BEHAVIOR_FEATURE_COLUMNS, BEHAVIOR_METADATA_COLUMNS
from trajectory.validation import validate_trajectory_file


FORBIDDEN_COLUMN_FRAGMENTS = (
    "ela",
    "utility",
    "p_skip",
    "p_ela",
    "cost",
    "inertia",
    "c1",
    "c2",
    "mutation",
    "strategy",
    "covariance",
    "sigma",
    "algorithm_id",
    "function_id",
)

BOUNDED_FEATURE_COLUMNS = (
    "bf_improvement_frequency_w02",
    "bf_directional_entropy_w05",
    "bf_stagnation_w10",
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

    for trajectory_row, behavior_row in zip(trajectory_rows, behavior_rows):
        for column in BEHAVIOR_METADATA_COLUMNS:
            if behavior_row[column] != trajectory_row[column]:
                raise ValueError(f"metadata column {column} differs from trajectory input")
        if behavior_row["bf_fe_ratio"] != trajectory_row["FE_ratio"]:
            raise ValueError("bf_fe_ratio must match FE_ratio")
        if behavior_row["bf_diversity_mean_pairwise"] < 0.0:
            raise ValueError("bf_diversity_mean_pairwise must be non-negative")

        for column in BOUNDED_FEATURE_COLUMNS:
            value = behavior_row[column]
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{column} must be in [0, 1] or null")

        for column in BEHAVIOR_FEATURE_COLUMNS:
            value = behavior_row[column]
            if value is not None and not isfinite(float(value)):
                raise ValueError(f"{column} must be finite or null")

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
