from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from trajectory.records import TrajectoryRecord


TRAJECTORY_SCHEMA = pa.schema(
    [
        ("problem_id", pa.string()),
        ("family", pa.string()),
        ("dimension", pa.int32()),
        ("algorithm", pa.string()),
        ("seed", pa.int64()),
        ("FE", pa.int64()),
        ("FE_ratio", pa.float64()),
        ("population", pa.list_(pa.list_(pa.float64()))),
        ("fitness", pa.list_(pa.float64())),
        ("best_fitness", pa.float64()),
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

