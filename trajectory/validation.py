from __future__ import annotations

from collections import defaultdict
from math import isclose, isfinite
from pathlib import Path

import pyarrow.parquet as pq

from trajectory.records import OPTIMIZER_STATE_MODE
from trajectory.sampling import (
    BUDGET_MILESTONE_RATIOS,
    DIVERSITY_RECOVERY_THRESHOLD,
    ELITE_MIGRATION_THRESHOLD,
    EPS,
    EVENT_ONLY_MIN_GAP_RATIO,
    EVENT_NAMES,
    MAX_EVENT_ONLY_PER_PHASE,
    MAX_SAMPLES_PER_RUN,
    MIN_SAMPLES_PER_RUN,
    MONITOR_RATIOS,
    RANK_CHANGE_THRESHOLD,
    SAMPLING_METADATA_COLUMNS,
    SAMPLING_PHASES,
    SAMPLING_PROTOCOL,
    STAGNATION_ONSET_THRESHOLD,
    is_budget_milestone,
    sampling_metrics,
    sampling_phase,
)
from trajectory.window_statistics import WINDOW_RATIOS


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
    *SAMPLING_METADATA_COLUMNS,
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
        _validate_sampling_metadata(row)
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
        _validate_run_sampling(ordered, key)

    return {"rows": len(rows), "runs": len(grouped)}


def _validate_sampling_metadata(row: dict) -> None:
    if str(row["sampling_protocol"]) != SAMPLING_PROTOCOL:
        raise ValueError("trajectory sampling protocol is inconsistent")
    actual_ratio = int(row["FE"]) / int(row["FE_total"])
    if not isclose(float(row["FE_ratio"]), actual_ratio, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("dynamic trajectory FE_ratio must equal FE / FE_total")
    target = float(row["monitor_target_ratio"])
    if not any(isclose(target, ratio, rel_tol=0.0, abs_tol=1e-12) for ratio in MONITOR_RATIOS):
        raise ValueError("monitor_target_ratio must belong to the frozen monitor grid")
    target_fe = int(round(target * int(row["FE_total"])) )
    alignment_gap = int(row["FE"]) - target_fe
    if alignment_gap < 0:
        raise ValueError("dynamic sample must use the first complete update not earlier than its monitor target")
    if alignment_gap >= len(row["population"]):
        raise ValueError("dynamic sample must be aligned within one complete population update")
    if str(row["sampling_phase"]) != sampling_phase(target):
        raise ValueError("sampling phase is inconsistent with monitor_target_ratio")

    milestone = bool(row["is_budget_milestone"])
    milestone_ratio = row["budget_milestone_ratio"]
    if milestone != is_budget_milestone(target):
        raise ValueError("budget-milestone flag is inconsistent")
