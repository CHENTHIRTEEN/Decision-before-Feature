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
    "window_statistics",
    "native_update_history",
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
        _validate_window_statistics(row)
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


def _validate_window_statistics(row: dict) -> None:
    statistics = row["window_statistics"]
    if not isinstance(statistics, list) or {str(item["suffix"]) for item in statistics} != {"w02", "w05", "w10"}:
        raise ValueError("trajectory row must contain w02/w05/w10 native-update window statistics")
    history = row["native_update_history"]
    if not isinstance(history, list) or len(history) < 2:
        raise ValueError("trajectory row must contain native-update scalar history")
    history_fes = [int(item["FE"]) for item in history]
    history_updates = [int(item["native_updates"]) for item in history]
    if history_fes != sorted(set(history_fes)) or history_updates != sorted(set(history_updates)):
        raise ValueError("native-update scalar history must be strictly increasing")
    if history_fes[-1] != int(row["FE"]) or history_updates[-1] != int(row["native_updates"]):
        raise ValueError("native-update scalar history must end at the formal checkpoint")
    if any(float(item["FE_ratio"]) != int(item["FE"]) / int(row["FE_total"]) for item in history):
        raise ValueError("native-update scalar history FE_ratio is inconsistent")
    for item in statistics:
        suffix = str(item["suffix"])
        nominal = {"w02": 0.02, "w05": 0.05, "w10": 0.10}[suffix]
        if float(item["nominal_window_ratio"]) != nominal:
            raise ValueError(f"{suffix} nominal window ratio is inconsistent")
        if int(item["anchor_FE"]) not in history_fes:
            raise ValueError(f"{suffix} anchor is missing from native-update scalar history")
        anchor_index = history_fes.index(int(item["anchor_FE"]))
        if int(item["anchor_native_updates"]) != history_updates[anchor_index]:
            raise ValueError(f"{suffix} anchor native_updates is inconsistent")
        target_span = int(round(nominal * int(row["FE_total"])))
        actual_span = int(row["FE"]) - int(item["anchor_FE"])
        if not target_span <= actual_span < target_span + len(row["population"]):
            raise ValueError(f"{suffix} window is not aligned within one complete native update")
