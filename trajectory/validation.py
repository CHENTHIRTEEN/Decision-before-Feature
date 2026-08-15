from __future__ import annotations

import json
from collections import defaultdict
from math import isclose, isfinite
from pathlib import Path

import pyarrow.parquet as pq

from benchmarks.bbob import bbob_function_id, bbob_landscape_family
from trajectory.query import (
    TRAJECTORY_QUERY_FEATURE_COLUMNS,
    TRAJECTORY_QUERY_ID,
    TRAJECTORY_QUERY_EVENT_CODE,
    TRAJECTORY_QUERY_PREPROCESSING_ID,
    TRAJECTORY_QUERY_PROTOCOL,
    TRAJECTORY_QUERY_SOURCE_MODE,
    TRAJECTORY_QUERY_STREAM_CODE,
)
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
    "function_id",
    "family",
    "cv_group_id",
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
FORBIDDEN_DECISION_INPUT_COLUMNS = {
    "benchmark_reference_value",
    "known_optimum",
    "reference_value",
    "gap",
    "loss_gap",
    "best_known_gap",
}
FORBIDDEN_DECISION_INPUT_FRAGMENTS = (
    "benchmark_reference",
    "known_optimum",
    "optimality_gap",
    "loss_gap",
)


def validate_trajectory_file(path: str | Path) -> dict[str, int]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    missing = REQUIRED_COLUMNS.difference(table.column_names)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if not rows:
        raise ValueError("trajectory file contains no rows")
    forbidden = sorted(
        column
        for column in table.column_names
        if column.lower() in FORBIDDEN_DECISION_INPUT_COLUMNS
        or any(fragment in column.lower() for fragment in FORBIDDEN_DECISION_INPUT_FRAGMENTS)
        or column.lower().endswith("_gap")
    )
    if forbidden:
        raise ValueError(
            "trajectory/Behavior inputs must not contain benchmark references or gap fields: "
            f"{forbidden}"
        )

    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        _validate_function_metadata(row)
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
        if not isfinite(float(row["best_fitness"])):
            raise ValueError("trajectory best_fitness must be finite")
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


def validate_trajectory_query_file(
    path: str | Path,
    *,
    trajectory_path: str | Path | None = None,
) -> dict[str, int]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    required = {
        "split",
        "problem_id",
        "function_id",
        "family",
        "cv_group_id",
        "dimension",
        "algorithm",
        "seed",
        "FE",
        "FE_ratio",
        "FE_total",
        "native_updates",
        "query_id",
        "query_protocol",
        "query_source_mode",
        "query_preprocessing_id",
        "query_feature_columns",
        "trajectory_query_reservoir_size",
        "trajectory_query_seen_count",
        "trajectory_sample_count",
        "trajectory_sample_coverage_ratio",
        "reservoir_stream_code",
        "reservoir_event_code",
        "trajectory_query_runtime",
        "feature_status",
        "feature_count",
        *TRAJECTORY_QUERY_FEATURE_COLUMNS,
    }
    missing = required.difference(table.column_names)
    if missing:
        raise ValueError(f"trajectory query file is missing columns: {sorted(missing)}")
    if not rows:
        raise ValueError("trajectory query file contains no rows")

    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    keys: set[tuple[str, str, int, int]] = set()
    for row in rows:
        _validate_function_metadata(row)
        identity = (
            str(row["query_id"]),
            str(row["query_protocol"]),
            str(row["query_source_mode"]),
            str(row["query_preprocessing_id"]),
        )
        expected_identity = (
            TRAJECTORY_QUERY_ID,
            TRAJECTORY_QUERY_PROTOCOL,
            TRAJECTORY_QUERY_SOURCE_MODE,
            TRAJECTORY_QUERY_PREPROCESSING_ID,
        )
        if identity != expected_identity:
            raise ValueError("trajectory query identity/protocol fields are inconsistent")
        fe = int(row["FE"])
        fe_total = int(row["FE_total"])
        seen_count = int(row["trajectory_query_seen_count"])
        reservoir_size = int(row["trajectory_query_reservoir_size"])
        sample_count = int(row["trajectory_sample_count"])
        native_updates = int(row["native_updates"])
        if not 0 < fe <= fe_total or seen_count != fe:
            raise ValueError("trajectory query seen_count must equal its emitted integer FE")
        if native_updates < 0:
            raise ValueError("trajectory query native_updates must be non-negative")
        if int(row["reservoir_stream_code"]) != TRAJECTORY_QUERY_STREAM_CODE:
            raise ValueError("trajectory query reservoir_stream_code is inconsistent")
        if int(row["reservoir_event_code"]) != TRAJECTORY_QUERY_EVENT_CODE:
            raise ValueError("trajectory query reservoir_event_code is inconsistent")
        if not isfinite(float(row["trajectory_query_runtime"])) or float(row["trajectory_query_runtime"]) < 0.0:
            raise ValueError("trajectory query runtime must be finite and non-negative")
        try:
            feature_columns = json.loads(str(row["query_feature_columns"]))
        except json.JSONDecodeError as exc:
            raise ValueError("trajectory query feature whitelist is not valid JSON") from exc
        if feature_columns != list(TRAJECTORY_QUERY_FEATURE_COLUMNS):
            raise ValueError("trajectory query feature whitelist is inconsistent")
        if reservoir_size <= 0 or sample_count != min(seen_count, reservoir_size):
            raise ValueError("trajectory query sample_count is inconsistent with reservoir sampling")
        expected_ratio = fe / fe_total
        if not isclose(float(row["FE_ratio"]), expected_ratio, rel_tol=0.0, abs_tol=EPS):
            raise ValueError("trajectory query FE_ratio must equal FE / FE_total")
        expected_coverage = sample_count / seen_count
        if not isclose(
            float(row["trajectory_sample_coverage_ratio"]),
            expected_coverage,
            rel_tol=0.0,
            abs_tol=EPS,
        ):
            raise ValueError("trajectory query coverage ratio is inconsistent")
        finite_count = sum(
            row[column] is not None and isfinite(float(row[column]))
            for column in TRAJECTORY_QUERY_FEATURE_COLUMNS
        )
        expected_status = "ok" if finite_count == len(TRAJECTORY_QUERY_FEATURE_COLUMNS) else "failed"
        if str(row["feature_status"]) != expected_status or int(row["feature_count"]) != finite_count:
            raise ValueError("trajectory query feature status/count is inconsistent")
        run_key = (str(row["problem_id"]), str(row["algorithm"]), int(row["seed"]))
        grouped[run_key].append(row)
        key = (*run_key, fe)
        if key in keys:
            raise ValueError("trajectory query state keys must be unique")
        keys.add(key)

    for run_key, run_rows in grouped.items():
        ordered_rows = sorted(run_rows, key=lambda row: int(row["FE"]))
        fes = [int(row["FE"]) for row in ordered_rows]
        if fes != sorted(set(fes)):
            raise ValueError(f"trajectory query FE must be strictly increasing for {run_key}")
        native_updates = [int(row["native_updates"]) for row in ordered_rows]
        if any(later <= earlier for earlier, later in zip(native_updates, native_updates[1:])):
            raise ValueError(f"trajectory query native_updates must be strictly increasing for {run_key}")

    if trajectory_path is not None:
        trajectory = pq.read_table(trajectory_path).to_pylist()
        trajectory_by_key = {
            (str(row["problem_id"]), str(row["algorithm"]), int(row["seed"]), int(row["FE"])): row
            for row in trajectory
        }
        trajectory_keys = set(trajectory_by_key)
        if keys != trajectory_keys:
            missing_query = len(trajectory_keys.difference(keys))
            extra_query = len(keys.difference(trajectory_keys))
            raise ValueError(
                "trajectory query snapshots must cover exactly the emitted trajectory states: "
                f"missing_query={missing_query}, extra_query={extra_query}"
            )
        query_by_key = {
            (str(row["problem_id"]), str(row["algorithm"]), int(row["seed"]), int(row["FE"])): row
            for row in rows
        }
        for key, query_row in query_by_key.items():
            trajectory_row = trajectory_by_key[key]
            if int(query_row["native_updates"]) != int(trajectory_row["native_updates"]):
                raise ValueError(f"trajectory query native_updates disagree with trajectory state for {key}")
            if int(query_row["FE_total"]) != int(trajectory_row["FE_total"]):
                raise ValueError(f"trajectory query FE_total disagrees with trajectory state for {key}")
    return {"rows": len(rows), "runs": len(grouped)}


def _validate_function_metadata(row: dict) -> None:
    problem_id = str(row["problem_id"])
    function, _ = parse_problem_id(problem_id)
    if problem_id.startswith("bbob_"):
        expected_function_id = bbob_function_id(function)
        expected_family = bbob_landscape_family(function)
    elif problem_id.startswith("cec2017_"):
        expected_function_id = f"cec2017_f{function:02d}"
        expected_family = "cec2017_unassigned_landscape_family"
    elif problem_id.startswith("cec2022_"):
        expected_function_id = f"cec2022_f{function:02d}"
        expected_family = "cec2022_unassigned_landscape_family"
    else:
        raise ValueError(f"unsupported problem_id for function metadata: {problem_id}")
    if str(row["function_id"]) != expected_function_id:
        raise ValueError("function_id is inconsistent with problem_id")
    if str(row["family"]) != expected_family:
        raise ValueError("landscape family is inconsistent with problem_id")
    if "cv_group_id" not in row:
        raise ValueError("cv_group_id is missing; all trajectory and query rows must carry cv_group_id")
    if str(row["cv_group_id"]) != expected_function_id:
        raise ValueError("cv_group_id is inconsistent with function_id")


def _validate_sampling_metadata(row: dict) -> None:
    if str(row["sampling_protocol"]) != SAMPLING_PROTOCOL:
        raise ValueError("trajectory sampling protocol is inconsistent")
    actual_ratio = int(row["FE"]) / int(row["FE_total"])
    if not isclose(float(row["FE_ratio"]), actual_ratio, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("dynamic trajectory FE_ratio must equal FE / FE_total")
    target = float(row["monitor_target_ratio"])
    if not any(isclose(target, ratio, rel_tol=0.0, abs_tol=1e-12) for ratio in MONITOR_RATIOS):
        raise ValueError("monitor_target_ratio must belong to the frozen monitor grid")
    target_fe = int(round(target * int(row["FE_total"])))
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
    if milestone:
        if milestone_ratio is None or not isclose(
            float(milestone_ratio), target, rel_tol=0.0, abs_tol=EPS
        ):
            raise ValueError("budget_milestone_ratio must equal the milestone monitor target")
    elif milestone_ratio is not None:
        raise ValueError("non-milestone rows must not define budget_milestone_ratio")

    event_flags = {
        name: bool(row[f"event_{name}"])
        for name in EVENT_NAMES
    }
    event_sample = bool(row["is_event_sample"])
    if event_sample != any(event_flags.values()):
        raise ValueError("is_event_sample must equal the disjunction of the event flags")
    expected_triggers = (["budget_milestone"] if milestone else []) + [
        name for name in EVENT_NAMES if event_flags[name]
    ]
    if list(row["sampling_triggers"]) != expected_triggers:
        raise ValueError("sampling_triggers are inconsistent with milestone and event flags")

    event_index = row["event_index_in_phase"]
    if milestone:
        if event_index is not None:
            raise ValueError("milestone rows must not consume an event-only phase index")
    else:
        if not event_sample:
            raise ValueError("every emitted non-milestone row must be an event-only sample")
        if event_index is None or not 1 <= int(event_index) <= MAX_EVENT_ONLY_PER_PHASE:
            raise ValueError("event-only rows require a valid event_index_in_phase")

    metric_columns = tuple(f"event_{name}_metric" for name in EVENT_NAMES)
    for column in metric_columns:
        if not isfinite(float(row[column])):
            raise ValueError(f"sampling event metric must be finite: {column}")
    recomputed = sampling_metrics(
        window_statistics=row["window_statistics"],
        native_update_history=row["native_update_history"],
        dimension=int(row["dimension"]),
        stagnation_span_ratio=float(row["event_stagnation_onset_metric"]),
    )
    for column in metric_columns:
        if not isclose(
            float(row[column]), float(recomputed[column]), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"sampling event metric is inconsistent with native-update history: {column}")

    if event_flags["improvement_resume"] and float(row["event_improvement_resume_metric"]) <= EPS:
        raise ValueError("improvement_resume requires a non-zero improvement-frequency metric")
    if event_flags["stagnation_onset"] and (
        float(row["event_stagnation_onset_metric"]) < STAGNATION_ONSET_THRESHOLD - EPS
    ):
        raise ValueError("stagnation_onset flag is below its frozen threshold")
    if event_flags["rank_change"] and (
        abs(float(row["event_rank_change_metric"])) < RANK_CHANGE_THRESHOLD - EPS
    ):
        raise ValueError("rank_change flag is below its frozen threshold")
    if event_flags["elite_migration"] and (
        float(row["event_elite_migration_metric"]) < ELITE_MIGRATION_THRESHOLD - EPS
    ):
        raise ValueError("elite_migration flag is below its frozen threshold")
    if event_flags["diversity_recovery"] and (
        float(row["event_diversity_recovery_metric"]) < DIVERSITY_RECOVERY_THRESHOLD - EPS
    ):
        raise ValueError("diversity_recovery flag is below its frozen threshold")


def _validate_window_statistics(row: dict) -> None:
    windows = row["window_statistics"]
    history = row["native_update_history"]
    if not isinstance(windows, list) or not isinstance(history, list):
        raise ValueError("window_statistics and native_update_history must be lists")
    if [str(item["suffix"]) for item in windows] != list(WINDOW_RATIOS):
        raise ValueError("window statistics must contain w02, w05, and w10 in frozen order")
    if not history:
        raise ValueError("native_update_history must not be empty")

    current_fe = int(row["FE"])
    fe_total = int(row["FE_total"])
    current_updates = int(row["native_updates"])
    population_size = len(row["population"])
    if population_size <= 0:
        raise ValueError("trajectory population must not be empty")
    if any(len(point) != int(row["dimension"]) for point in row["population"]):
        raise ValueError("trajectory population points must match dimension")

    history_fes: list[int] = []
    history_updates: list[int] = []
    history_best: list[float] = []
    history_by_fe: dict[int, dict] = {}
    for item in history:
        fe = int(item["FE"])
        updates = int(item["native_updates"])
        values = (
            float(item["FE_ratio"]),
            float(item["best_fitness"]),
            float(item["diversity_mean_pairwise"]),
            float(item["fitness_iqr"]),
            float(item["fitness_iqr_rel"]),
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("native_update_history values must be finite")
        if not isclose(values[0], fe / fe_total, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("native_update_history FE_ratio must equal FE / FE_total")
        if values[2] < 0.0 or values[3] < 0.0 or values[4] < 0.0:
            raise ValueError("native_update_history scale statistics must be non-negative")
        history_fes.append(fe)
        history_updates.append(updates)
        history_best.append(values[1])
        history_by_fe[fe] = item

    if history_fes != sorted(set(history_fes)):
        raise ValueError("native_update_history FE values must be strictly increasing")
    if any(later <= earlier for earlier, later in zip(history_updates, history_updates[1:])):
        raise ValueError("native_update_history native_updates must be strictly increasing")
    if any(later > earlier for earlier, later in zip(history_best, history_best[1:])):
        raise ValueError("native_update_history best_fitness must be non-increasing")
    if history_fes[-1] != current_fe or history_updates[-1] != current_updates:
        raise ValueError("native_update_history must end at the emitted trajectory state")
    if not isclose(history_best[-1], float(row["best_fitness"]), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("native_update_history final best_fitness is inconsistent")

    nonnegative_fields = {
        "anchor_diversity_mean_pairwise",
        "anchor_distance_to_best",
        "population_wasserstein_distance",
        "centroid_shift_distance",
        "population_chamfer_distance",
        "elite_centroid_shift",
        "covariance_trace_current",
        "covariance_trace_anchor",
        "covariance_trace_ratio",
        "covariance_effective_rank_current",
        "covariance_effective_rank_anchor",
        "covariance_effective_rank",
        "fitness_wasserstein_distance",
        "fitness_iqr_baseline",
        "fitness_iqr_current",
        "fitness_iqr_rel",
    }
    bounded_unit_fields = {
        "fitness_quantile_improvement_fraction",
        "population_overlap",
    }
    numeric_fields = nonnegative_fields | bounded_unit_fields | {
        "anchor_best_fitness",
        "covariance_trace_change",
        "covariance_effective_rank_change",
        "fitness_mean_improvement",
    }
    anchors: dict[str, int] = {}
    for item in windows:
        suffix = str(item["suffix"])
        nominal_ratio = float(item["nominal_window_ratio"])
        if not isclose(nominal_ratio, WINDOW_RATIOS[suffix], rel_tol=0.0, abs_tol=EPS):
            raise ValueError(f"{suffix} nominal window ratio is inconsistent")
        for field in numeric_fields:
            value = float(item[field])
            if not isfinite(value):
                raise ValueError(f"window statistic must be finite: {suffix}.{field}")
            if field in nonnegative_fields and value < 0.0:
                raise ValueError(f"window statistic must be non-negative: {suffix}.{field}")
            if field in bounded_unit_fields and not 0.0 <= value <= 1.0:
                raise ValueError(f"window statistic must be in [0, 1]: {suffix}.{field}")

        anchor_fe = int(item["anchor_FE"])
        anchors[suffix] = anchor_fe
        if anchor_fe not in history_by_fe:
            raise ValueError(f"{suffix} anchor must be a retained complete native update")
        target_span = int(round(WINDOW_RATIOS[suffix] * fe_total))
        target_anchor_fe = current_fe - target_span
        eligible_fes = [fe for fe in history_fes if fe <= target_anchor_fe]
        if not eligible_fes or anchor_fe != max(eligible_fes):
            raise ValueError(f"{suffix} anchor must be the latest complete update not after its target")
        actual_span = current_fe - anchor_fe
        if actual_span < target_span or actual_span >= target_span + population_size:
            raise ValueError(f"{suffix} realized span must be within one native population update")

        anchor_history = history_by_fe[anchor_fe]
        if int(item["anchor_native_updates"]) != int(anchor_history["native_updates"]):
            raise ValueError(f"{suffix} anchor native update index is inconsistent")
        if not isclose(
            float(item["anchor_best_fitness"]),
            float(anchor_history["best_fitness"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{suffix} anchor best_fitness is inconsistent")
        if not isclose(
            float(item["anchor_diversity_mean_pairwise"]),
            float(anchor_history["diversity_mean_pairwise"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{suffix} anchor diversity is inconsistent")
        if not isclose(
            float(item["fitness_iqr_current"]),
            float(history[-1]["fitness_iqr"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not isclose(
            float(item["fitness_iqr_rel"]),
            float(history[-1]["fitness_iqr_rel"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{suffix} current fitness-spread statistics are inconsistent")
        if not isclose(
            float(item["covariance_effective_rank"]),
            float(item["covariance_effective_rank_current"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{suffix} covariance effective-rank aliases are inconsistent")
        if float(item["covariance_effective_rank_current"]) > int(row["dimension"]) + 1e-9:
            raise ValueError(f"{suffix} covariance effective rank exceeds dimension")

        trace_anchor = float(item["covariance_trace_anchor"])
        trace_current = float(item["covariance_trace_current"])
        expected_trace_ratio = trace_current / max(abs(trace_anchor), EPS)
        expected_trace_change = (trace_current - trace_anchor) / max(abs(trace_anchor), EPS)
        if not isclose(
            float(item["covariance_trace_ratio"]), expected_trace_ratio, rel_tol=1e-12, abs_tol=1e-12
        ) or not isclose(
            float(item["covariance_trace_change"]), expected_trace_change, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"{suffix} covariance trace ratios are inconsistent")
        rank_anchor = float(item["covariance_effective_rank_anchor"])
        rank_current = float(item["covariance_effective_rank_current"])
        expected_rank_change = (rank_current - rank_anchor) / max(abs(rank_anchor), EPS)
        if not isclose(
            float(item["covariance_effective_rank_change"]),
            expected_rank_change,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{suffix} covariance effective-rank change is inconsistent")
        expected_iqr_rel = float(item["fitness_iqr_current"]) / max(
            float(item["fitness_iqr_baseline"]), EPS
        )
        if not isclose(
            float(item["fitness_iqr_rel"]), expected_iqr_rel, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"{suffix} fitness IQR normalization is inconsistent")

    if not anchors["w02"] >= anchors["w05"] >= anchors["w10"]:
        raise ValueError("window anchors must be ordered from the short to the long window")
    if history_fes[0] != anchors["w10"]:
        raise ValueError("native_update_history must begin at the w10 anchor")


def _validate_run_sampling(ordered: list[dict], key: tuple[str, str, int]) -> None:
    if not MIN_SAMPLES_PER_RUN <= len(ordered) <= MAX_SAMPLES_PER_RUN:
        raise ValueError(
            f"trajectory run must contain {MIN_SAMPLES_PER_RUN}--{MAX_SAMPLES_PER_RUN} rows for {key}"
        )
    targets = [float(row["monitor_target_ratio"]) for row in ordered]
    if targets != sorted(set(targets)):
        raise ValueError(f"emitted monitor targets must be strictly increasing for {key}")

    milestone_targets = [
        float(row["budget_milestone_ratio"])
        for row in ordered
        if bool(row["is_budget_milestone"])
    ]
    if len(milestone_targets) != len(BUDGET_MILESTONE_RATIOS) or any(
        not isclose(observed, expected, rel_tol=0.0, abs_tol=EPS)
        for observed, expected in zip(milestone_targets, BUDGET_MILESTONE_RATIOS)
    ):
        raise ValueError(f"trajectory run does not contain every frozen budget milestone exactly once for {key}")

    event_only_rows = [row for row in ordered if not bool(row["is_budget_milestone"])]
    if len(event_only_rows) > len(SAMPLING_PHASES) * MAX_EVENT_ONLY_PER_PHASE:
        raise ValueError(f"trajectory run exceeds the frozen event-only quota for {key}")
    event_only_ratios = [float(row["FE_ratio"]) for row in event_only_rows]
    if any(
        later - earlier < EVENT_ONLY_MIN_GAP_RATIO - EPS
        for earlier, later in zip(event_only_ratios, event_only_ratios[1:])
    ):
        raise ValueError(f"event-only trajectory rows violate the frozen FE-ratio gap for {key}")

    for phase in SAMPLING_PHASES:
        phase_rows = [row for row in event_only_rows if str(row["sampling_phase"]) == phase]
        if len(phase_rows) > MAX_EVENT_ONLY_PER_PHASE:
            raise ValueError(f"trajectory run exceeds the {phase} event-only quota for {key}")
        indices = [int(row["event_index_in_phase"]) for row in phase_rows]
        if indices != list(range(1, len(phase_rows) + 1)):
            raise ValueError(f"event-only phase indices must be consecutive for {key} phase={phase}")
