from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from trajectory.records import TrajectoryRecord


WINDOW_STATISTIC_TYPE = pa.struct(
    [
        ("suffix", pa.string()),
        ("nominal_window_ratio", pa.float64()),
        ("anchor_FE", pa.int64()),
        ("anchor_native_updates", pa.int64()),
        ("anchor_best_fitness", pa.float64()),
        ("anchor_diversity_mean_pairwise", pa.float64()),
        ("anchor_distance_to_best", pa.float64()),
        ("population_wasserstein_distance", pa.float64()),
        ("centroid_shift_distance", pa.float64()),
        ("fitness_quantile_improvement_fraction", pa.float64()),
        ("fitness_mean_improvement", pa.float64()),
        ("fitness_wasserstein_distance", pa.float64()),
        ("anchor_fitness_abs_mean", pa.float64()),
        ("population_overlap", pa.float64()),
    ]
)
NATIVE_UPDATE_STATISTIC_TYPE = pa.struct(
    [
        ("FE", pa.int64()),
        ("FE_ratio", pa.float64()),
        ("native_updates", pa.int64()),
        ("best_fitness", pa.float64()),
        ("diversity_mean_pairwise", pa.float64()),
    ]
)


TRAJECTORY_SCHEMA = pa.schema(
    [
        ("problem_id", pa.string()),
        ("family", pa.string()),
        ("dimension", pa.int32()),
        ("algorithm", pa.string()),
        ("seed", pa.int64()),
        ("FE", pa.int64()),
        ("FE_ratio", pa.float64()),
        ("FE_total", pa.int64()),
        ("native_updates", pa.int64()),
        ("window_statistics", pa.list_(WINDOW_STATISTIC_TYPE)),
        ("native_update_history", pa.list_(NATIVE_UPDATE_STATISTIC_TYPE)),
        ("population", pa.list_(pa.list_(pa.float64()))),
        ("fitness", pa.list_(pa.float64())),
        ("best_fitness", pa.float64()),
        ("optimizer_state_mode", pa.string()),
    ]
)


def write_parquet(records: list[TrajectoryRecord], output_path: str | Path) -> Path:
    if not records:
        raise ValueError("no trajectory records to write")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([record.__dict__ for record in records], schema=TRAJECTORY_SCHEMA)
    pq.write_table(table, path)
    return path
