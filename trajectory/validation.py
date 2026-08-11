from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from trajectory.records import OPTIMIZER_STATE_MODE


REQUIRED_COLUMNS = {
    "problem_id",
    "family",
    "dimension",
    "algorithm",
    "seed",
    "FE",
    "FE_ratio",
    "FE_total",
    "native_updates",
    "population",
    "fitness",
    "best_fitness",
    "optimizer_state_mode",
}


def validate_trajectory_file(path: str | Path) -> dict[str, int]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    missing = REQUIRED_COLUMNS.difference(table.column_names)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if not rows:
        raise ValueError("trajectory file contains no rows")

    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        if row["optimizer_state_mode"] != OPTIMIZER_STATE_MODE:
            raise ValueError(
                "trajectory was not generated with native optimizer-state continuation; regenerate the shard"
            )
        if not 0.0 < row["FE_ratio"] <= 1.0:
            raise ValueError("FE_ratio must be in (0, 1]")
        if int(row["FE_total"]) <= 0 or not 0 < int(row["FE"]) <= int(row["FE_total"]):
            raise ValueError("FE must be in (0, FE_total]")
        if int(row["native_updates"]) < 0:
            raise ValueError("native_updates must be non-negative")
        if len(row["population"]) != len(row["fitness"]):
            raise ValueError("population and fitness lengths must match")
        grouped[(row["algorithm"], row["problem_id"], row["seed"])].append(row)

    for key, group in grouped.items():
        ordered = sorted(group, key=lambda item: item["FE"])
        fes = [item["FE"] for item in ordered]
        totals = {int(item["FE_total"]) for item in ordered}
        native_updates = [int(item["native_updates"]) for item in ordered]
        best = [item["best_fitness"] for item in ordered]
        if fes != sorted(set(fes)):
            raise ValueError(f"FE must be strictly increasing for {key}")
        if any(later > earlier for earlier, later in zip(best, best[1:])):
            raise ValueError(f"best_fitness must be non-increasing for {key}")
        if len(totals) != 1:
            raise ValueError(f"FE_total must be constant for {key}")
        if any(later < earlier for earlier, later in zip(native_updates, native_updates[1:])):
            raise ValueError(f"native_updates must be non-decreasing for {key}")

    return {"rows": len(rows), "runs": len(grouped)}
