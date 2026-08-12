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
    alignment_gap = int(row["FE"]) - target * int(row["FE_total"])
    if alignment_gap < -1e-12:
        raise ValueError("dynamic sample must use the first complete update not earlier than its monitor target")
    if alignment_gap >= len(row["population"]) - 1e-12:
        raise ValueError("dynamic sample must be aligned within one complete population update")
    if str(row["sampling_phase"]) != sampling_phase(target):
        raise ValueError("sampling phase is inconsistent with monitor_target_ratio")

    milestone = bool(row["is_budget_milestone"])
    milestone_ratio = row["budget_milestone_ratio"]
    if milestone != is_budget_milestone(target):
        raise ValueError("budget-milestone flag is inconsistent")
    if milestone:
        if milestone_ratio is None or not isclose(
            float(milestone_ratio), target, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("budget_milestone_ratio must equal monitor_target_ratio")
    elif milestone_ratio is not None:
        raise ValueError("event-only rows must not define budget_milestone_ratio")

    event_flags = [bool(row[f"event_{name}"]) for name in EVENT_NAMES]
    event_metrics = [float(row[f"event_{name}_metric"]) for name in EVENT_NAMES]
    if not all(isfinite(value) for value in event_metrics):
        raise ValueError("event sampling metrics must all be finite")
    _validate_event_metric_values(row)
    if bool(row["is_event_sample"]) != any(event_flags):
        raise ValueError("is_event_sample must equal the union of event flags")
    triggers = tuple(str(value) for value in row["sampling_triggers"])
    expected_triggers = tuple(
        name
        for name, enabled in zip(EVENT_NAMES, event_flags, strict=True)
        if enabled
    )
    if milestone:
        expected_triggers = ("budget_milestone", *expected_triggers)
    if triggers != expected_triggers:
        raise ValueError("sampling_triggers must equal the active milestone and event flags")
    if not milestone and not any(event_flags):
        raise ValueError("trajectory rows must be emitted by a milestone or accepted event")


def _validate_run_sampling(rows: list[dict], key: tuple[str, str, int]) -> None:
    if not MIN_SAMPLES_PER_RUN <= len(rows) <= MAX_SAMPLES_PER_RUN:
        raise ValueError(f"dynamic sampling must emit 12..18 rows for {key}")
    milestone_ratios = tuple(
        float(row["budget_milestone_ratio"])
        for row in rows
        if bool(row["is_budget_milestone"])
    )
    if milestone_ratios != BUDGET_MILESTONE_RATIOS:
        raise ValueError(f"budget milestone coverage mismatch for {key}: {milestone_ratios}")
    monitor_targets = [float(row["monitor_target_ratio"]) for row in rows]
    monitor_indexes = [
        next(
            index
            for index, ratio in enumerate(MONITOR_RATIOS)
            if isclose(target, ratio, rel_tol=0.0, abs_tol=1e-12)
        )
        for target in monitor_targets
    ]
    if monitor_indexes != sorted(set(monitor_indexes)):
        raise ValueError(f"monitor targets must be strictly increasing and unique for {key}")
    event_only = [row for row in rows if not bool(row["is_budget_milestone"])]
    event_ratios = [float(row["FE_ratio"]) for row in event_only]
    if any(
        later - earlier < EVENT_ONLY_MIN_GAP_RATIO - EPS
        for earlier, later in zip(event_ratios, event_ratios[1:])
    ):
        raise ValueError(
            f"event-only samples violate the {EVENT_ONLY_MIN_GAP_RATIO:.2f} FE-ratio gap for {key}"
        )
    for phase in SAMPLING_PHASES:
        phase_rows = [row for row in event_only if str(row["sampling_phase"]) == phase]
        if len(phase_rows) > MAX_EVENT_ONLY_PER_PHASE:
            raise ValueError(f"event-only phase quota exceeded for {key}/{phase}")
        indexes = [int(row["event_index_in_phase"]) for row in phase_rows]
        if indexes != list(range(1, len(phase_rows) + 1)):
            raise ValueError(f"event-only indexes are inconsistent for {key}/{phase}")
    if any(row["event_index_in_phase"] is not None for row in rows if bool(row["is_budget_milestone"])):
        raise ValueError(f"budget milestone rows must not consume event-only quota for {key}")
    _validate_observable_event_sequence(rows, monitor_indexes, key)


def _validate_event_metric_values(row: dict) -> None:
    metrics = {
        name: float(row[f"event_{name}_metric"])
        for name in EVENT_NAMES
    }
    if not 0.0 <= metrics["improvement_resume"] <= 1.0:
        raise ValueError("improvement-resume metric must be a frequency in [0, 1]")
    if metrics["stagnation_onset"] < 0.0:
        raise ValueError("stagnation-onset metric must be non-negative")
    if metrics["elite_migration"] < 0.0 or metrics["diversity_recovery"] < 0.0:
        raise ValueError("elite-migration and diversity-recovery metrics must be non-negative")

    necessary_thresholds = {
        "improvement_resume": metrics["improvement_resume"] > EPS,
        "stagnation_onset": metrics["stagnation_onset"] >= STAGNATION_ONSET_THRESHOLD - EPS,
        "rank_change": abs(metrics["rank_change"]) >= RANK_CHANGE_THRESHOLD - EPS,
        "elite_migration": metrics["elite_migration"] >= ELITE_MIGRATION_THRESHOLD - EPS,
        "diversity_recovery": metrics["diversity_recovery"] >= DIVERSITY_RECOVERY_THRESHOLD - EPS,
    }
    for name in EVENT_NAMES:
        if bool(row[f"event_{name}"]) and not necessary_thresholds[name]:
            raise ValueError(f"event_{name} fired without satisfying its frozen threshold")

    recomputed = sampling_metrics(
        window_statistics=row["window_statistics"],
        native_update_history=row["native_update_history"],
        dimension=int(row["dimension"]),
        # The precise last-improvement FE can precede the retained 10% history.
        # Passing the recorded span still lets the other four metrics be
        # independently reconstructed from the persisted native-update data.
        stagnation_span_ratio=metrics["stagnation_onset"],
    )
    for name in (
        "improvement_resume",
        "rank_change",
        "elite_migration",
        "diversity_recovery",
    ):
        column = f"event_{name}_metric"
        if not isclose(float(row[column]), float(recomputed[column]), rel_tol=0.0, abs_tol=EPS):
            raise ValueError(f"{column} is inconsistent with persisted native-update windows")
    _validate_stagnation_metric(row, metrics["stagnation_onset"])


def _validate_stagnation_metric(row: dict, recorded: float) -> None:
    history = row["native_update_history"]
    improvements = [
        int(right["FE"])
        for left, right in zip(history, history[1:])
        if _is_strict_improvement(float(left["best_fitness"]), float(right["best_fitness"]))
    ]
    if improvements:
        expected = (int(row["FE"]) - improvements[-1]) / int(row["FE_total"])
        if not isclose(recorded, expected, rel_tol=0.0, abs_tol=EPS):
            raise ValueError(
                "event_stagnation_onset_metric is inconsistent with the latest persisted improvement"
            )
        return

    retained_span = (int(row["FE"]) - int(history[0]["FE"])) / int(row["FE_total"])
    if recorded + EPS < retained_span:
        raise ValueError(
            "event_stagnation_onset_metric implies an improvement absent from persisted native-update history"
        )


def _validate_observable_event_sequence(
    rows: list[dict],
    monitor_indexes: list[int],
    key: tuple[str, str, int],
) -> None:
    """Check rearm constraints that are observable in emitted trajectory rows.

    Non-emitted monitor points do not persist covariance or population geometry,
    so this deliberately avoids claiming a full replay for rank/elite events.
    Adjacent monitor points, however, leave no hidden rearm opportunity. A
    repeated stagnation event additionally requires an intervening strict best
    fitness improvement, because that is its only rearm mechanism.
    """
    previous_firing: dict[str, tuple[dict, int]] = {}
    for row, monitor_index in zip(rows, monitor_indexes, strict=True):
        for name in EVENT_NAMES:
            if not bool(row[f"event_{name}"]):
                continue
            previous = previous_firing.get(name)
            if previous is not None:
                previous_row, previous_index = previous
                if monitor_index == previous_index + 1 and name != "stagnation_onset":
                    raise ValueError(
                        f"event_{name} repeated at adjacent monitor points without a rearm opportunity for {key}"
                    )
                if name == "stagnation_onset":
                    improvement = _intervening_strict_improvement(
                        rows,
                        after_native_updates=int(previous_row["native_updates"]),
                        through_native_updates=int(row["native_updates"]),
                    )
                    if improvement is False:
                        raise ValueError(
                            f"event_stagnation_onset repeated without an intervening strict improvement for {key}"
                        )
            previous_firing[name] = (row, monitor_index)


def _intervening_strict_improvement(
    rows: list[dict],
    *,
    after_native_updates: int,
    through_native_updates: int,
) -> bool | None:
    history_by_update: dict[int, tuple[int, float]] = {}
    for row in rows:
        for item in row["native_update_history"]:
            native_updates = int(item["native_updates"])
            if after_native_updates <= native_updates <= through_native_updates:
                fe = int(item["FE"])
                value = float(item["best_fitness"])
                previous = history_by_update.get(native_updates)
                if previous is not None and (
                    previous[0] != fe
                    or not isclose(previous[1], value, rel_tol=0.0, abs_tol=EPS)
                ):
                    raise ValueError("overlapping native-update histories disagree on best_fitness")
                history_by_update[native_updates] = (fe, value)
    expected_updates = set(range(after_native_updates, through_native_updates + 1))
    if set(history_by_update) != expected_updates:
        return None
    ordered = [history_by_update[index] for index in sorted(history_by_update)]
    for (_, left), (_, right) in zip(ordered, ordered[1:]):
        if _is_strict_improvement(left, right):
            return True
    return False


def _is_strict_improvement(previous_best: float, current_best: float) -> bool:
    threshold = EPS * max(1.0, abs(float(previous_best)))
    return float(previous_best) - float(current_best) > threshold


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
        nominal = WINDOW_RATIOS[suffix]
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

        required_window_fields = {
            "population_wasserstein_distance",
            "centroid_shift_distance",
            "population_chamfer_distance",
            "elite_centroid_shift",
            "covariance_trace_current",
            "covariance_trace_anchor",
            "covariance_trace_ratio",
            "covariance_trace_change",
            "covariance_effective_rank_current",
            "covariance_effective_rank_anchor",
            "covariance_effective_rank",
            "covariance_effective_rank_change",
            "fitness_quantile_improvement_fraction",
            "fitness_mean_improvement",
            "fitness_wasserstein_distance",
            "fitness_iqr_baseline",
            "fitness_iqr_current",
            "fitness_iqr_rel",
            "population_overlap",
        }
        missing_fields = required_window_fields.difference(item)
        if missing_fields:
            raise ValueError(f"{suffix} window statistics are incomplete: {sorted(missing_fields)}")

    required_history_fields = {
        "FE",
        "FE_ratio",
        "native_updates",
        "best_fitness",
        "diversity_mean_pairwise",
        "fitness_iqr",
        "fitness_iqr_rel",
    }
    for item in history:
        missing_fields = required_history_fields.difference(item)
        if missing_fields:
            raise ValueError(f"native-update scalar history is incomplete: {sorted(missing_fields)}")
