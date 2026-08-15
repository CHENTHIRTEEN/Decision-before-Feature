from __future__ import annotations

import argparse
from math import isclose, isfinite
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from landscape_queries.specs import QUERY_PREPROCESSING_VERSION, get_query_spec
from experiments.phase1_batch_common import TIMING_REPETITIONS
from selection_reference.model import SELECTION_REFERENCE_PROTOCOL, SELECTOR_TARGET_TRANSFORM
from trajectory.sampling import (
    EVENT_NAMES,
    SAMPLING_METADATA_COLUMNS,
    SAMPLING_PROTOCOL,
    is_budget_milestone,
    sampling_phase,
)
from utility_labels.fields import (
    BEHAVIOR_UTILITY_VALUE_COLUMNS,
    MATCHED_ACQUISITION_PATH_SUFFIXES,
    NEED_BEHAVIOR_ONLY_COLUMNS,
    NEED_QUERY_COLUMNS,
    QUERY_DESCRIPTOR_USE_INCREMENT_COLUMNS,
    QUERY_MATCHED_STATE_ONLY_UTILITY_COLUMNS,
    QUERY_OPERATIONAL_INCREMENT_COLUMNS,
    QUERY_SAMPLING_DIRECT_INCREMENT_COLUMNS,
    QUERY_STATE_ONLY_VS_SAMPLING_INCREMENT_COLUMNS,
    SAMPLING_ONLY_UTILITY_COLUMNS,
    SCIENTIFIC_ENDPOINT_SOURCE,
    SCIENTIFIC_PATH_STATUS_PROTOCOL,
    TIMING_REPLAY_STATUSES,
    TIMING_REPLAY_STATUS_PROTOCOL,
    UTILITY_LAMBDAS,
    UTILITY_VALUE_COLUMNS,
)


EPS = 1e-12
UTILITY_FORMULA_PROTOCOL = (
    "clipped_log10_gap_difference_minus_lambda_log10_complete_path_runtime_ratio"
)
RETIRED_COLUMNS = {
    "u_query_lamT_0",
    "u_query_lamT_025",
    "u_query_lamT_05",
    "u_query_lamT_1",
    "u_query_lamT_2",
    "u_behavior_lamT_1",
    "u_query_information_increment_lamT_1",
    "query_information_increment_lamT_0",
    "query_information_increment_lamT_025",
    "query_information_increment_lamT_05",
    "query_information_increment_lamT_1",
    "query_information_increment_lamT_2",
    "performance_norm_scale",
    "time_cost_norm",
    "skip_timed_out",
    "skip_path_completed",
    "query_path_timed_out",
    "query_path_completed",
    "behavior_path_timed_out",
    "behavior_path_completed",
    "completed_timing_replay_outcomes_consistent",
}


def validate_utility_label_file(path: str | Path) -> dict[str, int]:
    table = pq.read_table(path)
    retired = sorted(RETIRED_COLUMNS.intersection(table.column_names))
    if retired:
        raise ValueError(f"utility label file contains retired columns: {retired}")
    rows = table.to_pylist()
    if not rows:
        raise ValueError("utility label file contains no rows")
    for row in rows:
        _validate_row(row)
    return {"rows": len(rows)}


def _validate_row(row: dict) -> None:
    missing_sampling = set(SAMPLING_METADATA_COLUMNS).difference(row)
    if missing_sampling:
        raise ValueError(
            "utility label is missing dynamic-sampling metadata: "
            f"{sorted(missing_sampling)}"
        )
    spec = get_query_spec(str(row["query_id"]))
    if str(row["query_protocol"]) != spec.protocol:
        raise ValueError("query_protocol does not match query_id")
    if str(row["query_preprocessing_id"]) != QUERY_PREPROCESSING_VERSION:
        raise ValueError("query_preprocessing_id does not match the frozen preprocessing contract")
    if str(row["sample_design_id"]) != spec.sample_design_id:
        raise ValueError("sample_design_id does not match query_id")
    if str(row["selection_reference_protocol"]) != SELECTION_REFERENCE_PROTOCOL:
        raise ValueError("query Utility label uses an unsupported Selection Reference protocol")
    if str(row["selector_target_transform"]) != SELECTOR_TARGET_TRANSFORM:
        raise ValueError("selector_target_transform does not match the frozen target transform")
    if str(row["utility_formula_protocol"]) != UTILITY_FORMULA_PROTOCOL:
        raise ValueError("Utility formula protocol is inconsistent")
    _validate_fe_and_sampling(row)
    _validate_action_relations(row)
    _validate_performance(row)
    _validate_complete_path_timing(row)
    _validate_utility_values(row)
    _validate_ert(row)


def _validate_fe_and_sampling(row: dict) -> None:
    if int(row["FE_prefix"]) + int(row["FE_no_query_optimization"]) != int(row["FE_total"]):
        raise ValueError("no-query FE ledger is inconsistent")
    if int(row["FE_prefix"]) != int(row["FE"]):
        raise ValueError("FE_prefix must equal the sampled trajectory FE")
    actual_fe_ratio = int(row["FE"]) / int(row["FE_total"])
    if float(row["FE_ratio"]) != actual_fe_ratio:
        raise ValueError("utility-label FE_ratio must equal actual FE / FE_total")
    _validate_sampling_metadata(row, actual_fe_ratio=actual_fe_ratio)
    if (
        int(row["FE_prefix"])
        + int(row["FE_query"])
        + int(row["FE_action_optimization"])
        != int(row["FE_total"])
    ):
        raise ValueError("query FE ledger is inconsistent")
    expected_query_fe = get_query_spec(str(row["query_id"])).sample_design.sample_size(
        int(row["dimension"])
    )
    if int(row["FE_query"]) != expected_query_fe:
        raise ValueError("FE_query does not match the frozen sample design")


def _validate_action_relations(row: dict) -> None:
    default = str(row["default_algorithm"])
    prefix = str(row["prefix_algorithm"])
    selected = str(row["selected_algorithm"])
    if str(row["no_query_algorithm"]) != default:
        raise ValueError("no_query_algorithm must equal default_algorithm")
    expected_default = selected == default
    expected_prefix = selected == prefix
    expected_handoff = not expected_prefix
    checks = {
        "selected_equals_default": expected_default,
        "selected_equals_prefix": expected_prefix,
        "handoff_required": expected_handoff,
        "skip_switches_from_prefix": default != prefix,
    }
    for column, expected in checks.items():
        if bool(row[column]) != expected:
            raise ValueError(f"{column} is inconsistent")
    expected_transition = (
        "native_optimizer_state" if expected_prefix else "population_transfer_initialization"
    )
    if str(row["query_transition_mode"]) != expected_transition:
        raise ValueError("query_transition_mode is inconsistent")
    if str(row["handoff_type"]) != expected_transition:
        raise ValueError("handoff_type must equal query_transition_mode")
    if expected_handoff != (str(row["handoff_type"]) == "population_transfer_initialization"):
        raise ValueError("handoff_required must match handoff_type")

    behavior_selected = str(row["behavior_selected_algorithm"])
    behavior_equals_default = behavior_selected == default
    behavior_equals_prefix = behavior_selected == prefix
    behavior_handoff = not behavior_equals_prefix
    behavior_checks = {
        "behavior_selected_equals_default": behavior_equals_default,
        "behavior_selected_equals_prefix": behavior_equals_prefix,
        "behavior_handoff_required": behavior_handoff,
    }
    for column, expected in behavior_checks.items():
        if bool(row[column]) != expected:
            raise ValueError(f"{column} is inconsistent")
    expected_behavior_transition = (
        "native_optimizer_state"
        if behavior_equals_prefix
        else "population_transfer_initialization"
    )
    if str(row["behavior_handoff_type"]) != expected_behavior_transition:
        raise ValueError("behavior_handoff_type is inconsistent")

    for path_prefix in ("query_matched_state_only", "sampling_only"):
        path_selected = str(row[f"{path_prefix}_selected_algorithm"])
        path_equals_default = path_selected == default
        path_equals_prefix = path_selected == prefix
        path_handoff = not path_equals_prefix
        for suffix, expected in (
            ("selected_equals_default", path_equals_default),
            ("selected_equals_prefix", path_equals_prefix),
            ("handoff_required", path_handoff),
        ):
            if bool(row[f"{path_prefix}_{suffix}"]) != expected:
                raise ValueError(f"{path_prefix}_{suffix} is inconsistent")
        expected_path_transition = (
            "native_optimizer_state"
            if path_equals_prefix
            else "population_transfer_initialization"
        )
        if str(row[f"{path_prefix}_handoff_type"]) != expected_path_transition:
            raise ValueError(f"{path_prefix}_handoff_type is inconsistent")
    if str(row["sampling_only_selected_algorithm"]) != prefix:
        raise ValueError("sampling-only path must continue the prefix algorithm")


def _validate_performance(row: dict) -> None:
    floor = float(row["log10_gap_floor"])
    cap = float(row["log10_gap_cap"])
    if not isfinite(floor) or not isfinite(cap) or not 0.0 < floor < cap:
        raise ValueError("log10 gap bounds must satisfy 0 < floor < cap")
    gaps = {
        "skip": float(row["p_skip"]),
        "query_joint": float(row["p_query"]),
        "behavior_only_full_budget": float(row["behavior_p_behavior"]),
        "query_matched_state_only": float(row["p_query_matched_state_only"]),
        "sampling_only_continue_current": float(
            row["p_sampling_only_continue_current"]
        ),
    }
    if any(not isfinite(value) or value < 0.0 for value in gaps.values()):
        raise ValueError("Utility gaps must be finite and non-negative")
    expected_logs = {
        name: float(np.log10(np.clip(value, floor, cap)))
        for name, value in gaps.items()
    }
    saved_logs = {
        "skip": float(row["log10_gap_skip"]),
        "query_joint": float(row["log10_gap_query_joint"]),
        "behavior_only_full_budget": float(row["log10_gap_behavior_only_full_budget"]),
        "query_matched_state_only": float(
            row["log10_gap_query_matched_state_only"]
        ),
        "sampling_only_continue_current": float(
            row["log10_gap_sampling_only_continue_current"]
        ),
    }
    for name, expected in expected_logs.items():
        if not isclose(saved_logs[name], expected, rel_tol=0.0, abs_tol=EPS):
            raise ValueError(f"saved {name} clipped log10 gap is inconsistent")
    improvements = {
        "query_joint_log10_gap_improvement": expected_logs["skip"] - expected_logs["query_joint"],
        "behavior_only_log10_gap_improvement": (
            expected_logs["skip"] - expected_logs["behavior_only_full_budget"]
        ),
        "query_vs_behavior_log10_gap_improvement": (
            expected_logs["behavior_only_full_budget"] - expected_logs["query_joint"]
        ),
    }
    for column, expected in improvements.items():
        if not isclose(float(row[column]), expected, rel_tol=0.0, abs_tol=EPS):
            raise ValueError(f"{column} is inconsistent")
    matched_gap_increments = {
        "query_descriptor_use_log10_gap_increment": (
            expected_logs["query_matched_state_only"] - expected_logs["query_joint"]
        ),
        "query_sampling_direct_log10_gap_increment": (
            expected_logs["skip"] - expected_logs["sampling_only_continue_current"]
        ),
    }
    for column, expected in matched_gap_increments.items():
        if not isclose(float(row[column]), expected, rel_tol=0.0, abs_tol=EPS):
            raise ValueError(f"{column} is inconsistent")
    continuation_gap = float(row["continuation_only_gap"])
    sample_gap = float(row["query_sample_best_gap"])
    query_path_completed = bool(row["scientific_query_joint_path_completed"])
    operational_gap = (
        min(continuation_gap, sample_gap) if query_path_completed else continuation_gap
    )
    if not isclose(float(row["p_query"]), operational_gap, rel_tol=0.0, abs_tol=EPS):
        raise ValueError(
            "p_query may include query sample best only when the Stage-A continuation completed"
        )
    expected_sample_improvement = query_path_completed and sample_gap < continuation_gap
    if bool(row["query_sample_improved_terminal"]) != expected_sample_improvement:
        raise ValueError("query_sample_improved_terminal is inconsistent")
    if not isclose(
        float(row["query_sample_terminal_gap_improvement"]),
        continuation_gap - operational_gap,
        rel_tol=0.0,
        abs_tol=EPS,
    ):
        raise ValueError("query sample terminal improvement is inconsistent")


def _validate_complete_path_timing(row: dict) -> None:
    if str(row["scientific_endpoint_source"]) != SCIENTIFIC_ENDPOINT_SOURCE:
        raise ValueError("scientific endpoints must come from the Stage-A Selection Reference")
    if int(row["timing_repetitions"]) != TIMING_REPETITIONS:
        raise ValueError("Utility timing must use exactly three repetitions")
    if str(row["timing_order_protocol"]) != "cyclic_complete_path_v1":
        raise ValueError("Utility timing uses the wrong cyclic order protocol")
    if str(row["timing_replay_status_protocol"]) != TIMING_REPLAY_STATUS_PROTOCOL:
        raise ValueError("Utility timing uses the wrong Stage-B replay-status protocol")
    replay_timeout = float(row["timing_replay_timeout_seconds"])
    if not isfinite(replay_timeout) or replay_timeout <= 0.0:
        raise ValueError("timing_replay_timeout_seconds must be finite and positive")
    if not str(row["complete_path_timing_source"]):
        raise ValueError("Utility label must retain its complete-path timing source")
    if str(row["complete_path_timing_origin"]) != "decision_state_to_terminal":
        raise ValueError(
            "nested Utility timing must start at the saved decision state; the shared prefix is sunk"
        )
    if not str(row["timing_environment_id"]):
        raise ValueError("Utility timing requires a non-empty timing_environment_id")
    if int(row["thread_count"]) <= 0:
        raise ValueError("Utility timing thread_count must be a positive integer")
    paths = (
        "skip",
        "query_joint",
        "query_matched_state_only",
        "sampling_only_continue_current",
        "behavior_only_full_budget",
    )
    medians: dict[str, float] = {}
    replay_instabilities: list[bool] = []
    for path in paths:
        repeats = np.asarray(row[f"runtime_{path}_repetitions"], dtype=float)
        censored_repeats = np.asarray(
            row[f"runtime_{path}_censored_repetitions"],
            dtype=float,
        )
        compatibility_censored_repeats = np.asarray(
            row[f"runtime_{path}_failure_worst_case_repetitions"],
            dtype=float,
        )
        positions = tuple(int(value) for value in row[f"timing_order_positions_{path}"])
        statuses = tuple(
            str(value) for value in row[f"timing_replay_status_repetitions_{path}"]
        )
        effective_fe = tuple(
            int(value)
            for value in row[f"timing_replay_effective_FE_repetitions_{path}"]
        )
        observed_first_hits = tuple(
            None if value is None else int(value)
            for value in row[f"timing_replay_observed_first_hit_FE_repetitions_{path}"]
        )
        target_hit_flags = tuple(
            bool(value)
            for value in row[f"timing_replay_target_hit_observed_flags_{path}"]
        )
        hit_before_failure_flags = tuple(
            bool(value)
            for value in row[
                f"timing_replay_target_hit_before_failure_flags_{path}"
            ]
        )
        endpoint_success_flags = tuple(
            bool(value)
            for value in row[f"timing_replay_endpoint_success_flags_{path}"]
        )
        timed_out_flags = tuple(
            bool(value) for value in row[f"timing_replay_timed_out_flags_{path}"]
        )
        completed_flags = tuple(
            bool(value)
            for value in row[f"timing_replay_path_completed_flags_{path}"]
        )
        if repeats.shape != (TIMING_REPETITIONS,) or not np.isfinite(repeats).all():
            raise ValueError(f"{path} complete-path runtimes must contain three finite values")
        if bool((repeats <= 0.0).any()):
            raise ValueError(f"{path} complete-path runtimes must be strictly positive")
        if (
            censored_repeats.shape != (TIMING_REPETITIONS,)
            or not np.isfinite(censored_repeats).all()
            or bool((censored_repeats <= 0.0).any())
        ):
            raise ValueError(
                f"{path} censored runtimes must contain three positive finite values"
            )
        if (
            compatibility_censored_repeats.shape != (TIMING_REPETITIONS,)
            or not np.isfinite(compatibility_censored_repeats).all()
            or bool((compatibility_censored_repeats <= 0.0).any())
        ):
            raise ValueError(
                f"{path} compatibility censored runtimes must contain three positive finite values"
            )
        if len(positions) != TIMING_REPETITIONS:
            raise ValueError(f"{path} timing order positions must contain three values")
        if len(set(positions)) != TIMING_REPETITIONS or not set(positions).issubset(
            set(range(len(paths)))
        ):
            raise ValueError(
                f"{path} must occupy three distinct cyclic positions from 0,...,4"
            )
        replay_lists = (
            statuses,
            effective_fe,
            observed_first_hits,
            target_hit_flags,
            hit_before_failure_flags,
            endpoint_success_flags,
            timed_out_flags,
            completed_flags,
        )
        if any(len(values) != TIMING_REPETITIONS for values in replay_lists):
            raise ValueError(f"{path} must retain all three Stage-B replay outcomes")
        if not set(statuses).issubset(TIMING_REPLAY_STATUSES):
            raise ValueError(f"{path} contains an unsupported timing replay status")
        expected_completed = tuple(status == "completed" for status in statuses)
        expected_timed_out = tuple(status == "timed_out" for status in statuses)
        if completed_flags != expected_completed or timed_out_flags != expected_timed_out:
            raise ValueError(f"{path} replay status and timeout/completion flags disagree")
        expected_target_hit = tuple(value is not None for value in observed_first_hits)
        if target_hit_flags != expected_target_hit:
            raise ValueError(f"{path} replay target-hit flags disagree with observed first hits")
        if hit_before_failure_flags != tuple(
            hit and not completed
            for hit, completed in zip(
                expected_target_hit, expected_completed, strict=True
            )
        ):
            raise ValueError(f"{path} replay target-hit-before-failure flags are inconsistent")
        if endpoint_success_flags != tuple(
            hit and completed
            for hit, completed in zip(
                expected_target_hit, expected_completed, strict=True
            )
        ):
            raise ValueError(f"{path} replay endpoint-success flags are inconsistent")
        planned_column = {
            "skip": "skip_planned_FE",
            "query_joint": "query_path_planned_FE",
            "query_matched_state_only": "query_matched_state_only_path_planned_FE",
            "sampling_only_continue_current": "sampling_only_path_planned_FE",
            "behavior_only_full_budget": "behavior_path_planned_FE",
        }[path]
        planned_fe = int(row[planned_column])
        if any(value < 0 or value > planned_fe for value in effective_fe):
            raise ValueError(f"{path} replay effective FE lies outside its planned budget")
        expected_censored = np.asarray(
            [
                runtime if status == "completed" else max(runtime, replay_timeout)
                for runtime, status in zip(repeats, statuses, strict=True)
            ],
            dtype=float,
        )
        if not np.allclose(
            censored_repeats,
            expected_censored,
            rtol=0.0,
            atol=EPS,
        ):
            raise ValueError(f"{path} censored runtime repetitions are inconsistent")
        if not np.allclose(
            compatibility_censored_repeats,
            expected_censored,
            rtol=0.0,
            atol=EPS,
        ):
            raise ValueError(
                f"{path} failure-worst-case repetitions must alias censored repetitions"
            )
        expected_counts = {
            f"timing_replay_completed_repetitions_{path}": sum(expected_completed),
            f"timing_replay_timeout_repetitions_{path}": sum(expected_timed_out),
            f"timing_replay_failure_repetitions_{path}": sum(
                status == "failed" for status in statuses
            ),
        }
        for column, expected in expected_counts.items():
            if int(row[column]) != int(expected):
                raise ValueError(f"{column} is inconsistent")
        path_instability = len(set(statuses)) > 1
        if bool(row[f"timing_replay_status_instability_{path}"]) != path_instability:
            raise ValueError(f"timing_replay_status_instability_{path} is inconsistent")
        if bool(row[f"timing_replay_instability_{path}"]) != path_instability:
            raise ValueError(f"timing_replay_instability_{path} is inconsistent")
        replay_instabilities.append(path_instability)
        medians[path] = float(np.median(expected_censored))
        if not isclose(
            float(row[f"runtime_{path}_median"]),
            medians[path],
            rel_tol=0.0,
            abs_tol=EPS,
        ):
            raise ValueError(f"runtime_{path}_median is inconsistent")
        if not isclose(
            float(row[f"runtime_{path}_raw_observed_median"]),
            float(np.median(repeats)),
            rel_tol=0.0,
            abs_tol=EPS,
        ):
            raise ValueError(f"runtime_{path}_raw_observed_median is inconsistent")
        if not isclose(
            float(row[f"runtime_{path}_failure_worst_case_median"]),
            medians[path],
            rel_tol=0.0,
            abs_tol=EPS,
        ):
            raise ValueError(f"runtime_{path}_failure_worst_case_median is inconsistent")
        if not bool(row[f"timing_replay_path_identity_consistent_{path}"]):
            raise ValueError(f"timing_replay_path_identity_consistent_{path} must be true")
        internal_consistency = row[
            f"completed_timing_replay_outcomes_internally_consistent_{path}"
        ]
        expected_internal_applicable = sum(expected_completed) >= 2
        if expected_internal_applicable:
            if internal_consistency is None or not bool(internal_consistency):
                raise ValueError(
                    f"completed timing replay internal consistency for {path} must be true"
                )
        elif internal_consistency is not None:
            raise ValueError(
                f"completed timing replay internal consistency for {path} must be null"
            )
        scientific_prefix = {
            "skip": "scientific_skip",
            "query_joint": "scientific_query_joint",
            "query_matched_state_only": "scientific_query_matched_state_only",
            "sampling_only_continue_current": (
                "scientific_sampling_only_continue_current"
            ),
            "behavior_only_full_budget": "scientific_behavior_only_full_budget",
        }[path]
        stage_a_completed = bool(row[f"{scientific_prefix}_path_completed"])
        stage_a_consistency = row[
            f"stage_a_to_completed_timing_replays_consistent_{path}"
        ]
        expected_stage_a_applicable = stage_a_completed and any(expected_completed)
        if expected_stage_a_applicable:
            if stage_a_consistency is None or not bool(stage_a_consistency):
                raise ValueError(
                    f"Stage-A-to-completed-replay consistency for {path} must be true"
                )
        elif stage_a_consistency is not None:
            raise ValueError(
                f"Stage-A-to-completed-replay consistency for {path} must be null"
            )
        completion_status_instability = any(
            completed != stage_a_completed for completed in expected_completed
        )
        if bool(
            row[f"stage_a_stage_b_completion_status_instability_{path}"]
        ) != completion_status_instability:
            raise ValueError(
                f"stage_a_stage_b_completion_status_instability_{path} is inconsistent"
            )
    if bool(row["timing_replay_instability"]) != any(replay_instabilities):
        raise ValueError("timing_replay_instability is inconsistent")
    for repetition in range(TIMING_REPETITIONS):
        observed = {
            int(row[f"timing_order_positions_{path}"][repetition])
            for path in paths
        }
        if observed != set(range(len(paths))):
            raise ValueError("each repetition must execute all five paths at unique positions")
    expected_costs = {
        "time_cost_log10_ratio": np.log10(medians["query_joint"] / medians["skip"]),
        "behavior_time_cost_log10_ratio": np.log10(
            medians["behavior_only_full_budget"] / medians["skip"]
        ),
        "query_vs_behavior_time_cost_log10_ratio": np.log10(
            medians["query_joint"] / medians["behavior_only_full_budget"]
        ),
    }
    for column, expected in expected_costs.items():
        if not isclose(float(row[column]), float(expected), rel_tol=0.0, abs_tol=EPS):
            raise ValueError(f"{column} is inconsistent")
    memory_status = str(row["peak_memory_measurement_status"])
    memory_values = [
        float("nan") if row[column] is None else float(row[column])
        for column in (
            "peak_memory_bytes_skip",
            "peak_memory_bytes_query_joint",
            "peak_memory_bytes_behavior_only_full_budget",
            *(
                f"peak_memory_bytes_{suffix}"
                for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
            ),
        )
    ]
    if memory_status == "not_available":
        if any(isfinite(value) for value in memory_values):
            raise ValueError("unavailable peak-memory endpoints must remain null/non-finite")
    elif memory_status == "measured_complete_path":
        if any(not isfinite(value) or value < 0.0 for value in memory_values):
            raise ValueError("measured peak-memory endpoints must be finite and non-negative")
    else:
        raise ValueError("unsupported peak_memory_measurement_status")


def _validate_utility_values(row: dict) -> None:
    query_gain = float(row["query_joint_log10_gap_improvement"])
    behavior_gain = float(row["behavior_only_log10_gap_improvement"])
    query_vs_behavior_gain = float(row["query_vs_behavior_log10_gap_improvement"])
    query_cost = float(row["time_cost_log10_ratio"])
    behavior_cost = float(row["behavior_time_cost_log10_ratio"])
    increment_cost = float(row["query_vs_behavior_time_cost_log10_ratio"])
    state_only_gain = float(row["log10_gap_skip"]) - float(
        row["log10_gap_query_matched_state_only"]
    )
    sampling_only_gain = float(row["query_sampling_direct_log10_gap_increment"])
    descriptor_use_gain = float(row["query_descriptor_use_log10_gap_increment"])
    state_only_vs_sampling_gain = float(
        row["log10_gap_sampling_only_continue_current"]
    ) - float(row["log10_gap_query_matched_state_only"])
    skip_runtime = float(row["runtime_skip_median"])
    query_runtime = float(row["runtime_query_joint_median"])
    state_only_runtime = float(row["runtime_query_matched_state_only_median"])
    sampling_only_runtime = float(row["runtime_sampling_only_continue_current_median"])
    state_only_cost = float(np.log10(state_only_runtime / skip_runtime))
    sampling_only_cost = float(np.log10(sampling_only_runtime / skip_runtime))
    descriptor_use_cost = float(np.log10(query_runtime / state_only_runtime))
    state_only_vs_sampling_cost = float(
        np.log10(state_only_runtime / sampling_only_runtime)
    )
    for (
        weight,
        query_column,
        behavior_column,
        increment_column,
        state_only_column,
        sampling_only_column,
        descriptor_use_column,
        state_only_vs_sampling_column,
        sampling_direct_column,
        query_need,
        behavior_need,
    ) in zip(
        UTILITY_LAMBDAS,
        UTILITY_VALUE_COLUMNS,
        BEHAVIOR_UTILITY_VALUE_COLUMNS,
        QUERY_OPERATIONAL_INCREMENT_COLUMNS,
        QUERY_MATCHED_STATE_ONLY_UTILITY_COLUMNS,
        SAMPLING_ONLY_UTILITY_COLUMNS,
        QUERY_DESCRIPTOR_USE_INCREMENT_COLUMNS,
        QUERY_STATE_ONLY_VS_SAMPLING_INCREMENT_COLUMNS,
        QUERY_SAMPLING_DIRECT_INCREMENT_COLUMNS,
        NEED_QUERY_COLUMNS,
        NEED_BEHAVIOR_ONLY_COLUMNS,
        strict=True,
    ):
        expected_query = query_gain - float(weight) * query_cost
        expected_behavior = behavior_gain - float(weight) * behavior_cost
        expected_increment = query_vs_behavior_gain - float(weight) * increment_cost
        expected_state_only = state_only_gain - float(weight) * state_only_cost
        expected_sampling_only = sampling_only_gain - float(weight) * sampling_only_cost
        expected_descriptor_use = descriptor_use_gain - float(weight) * descriptor_use_cost
        expected_state_only_vs_sampling = state_only_vs_sampling_gain - float(
            weight
        ) * state_only_vs_sampling_cost
        expected_sampling_direct = sampling_only_gain - float(weight) * sampling_only_cost
        checks = {
            query_column: expected_query,
            behavior_column: expected_behavior,
            increment_column: expected_increment,
            state_only_column: expected_state_only,
            sampling_only_column: expected_sampling_only,
            descriptor_use_column: expected_descriptor_use,
            state_only_vs_sampling_column: expected_state_only_vs_sampling,
            sampling_direct_column: expected_sampling_direct,
        }
        for column, expected in checks.items():
            if not isclose(float(row[column]), expected, rel_tol=0.0, abs_tol=EPS):
                raise ValueError(f"{column} is inconsistent")
        if not isclose(
            float(row[query_column]) - float(row[behavior_column]),
            float(row[increment_column]),
            rel_tol=0.0,
            abs_tol=EPS,
        ):
            raise ValueError("query operational increment lost exact Utility additivity")
        if bool(row[query_need]) != (expected_query > 0.0):
            raise ValueError(f"{query_need} is inconsistent")
        if bool(row[behavior_need]) != (expected_behavior > 0.0):
            raise ValueError(f"{behavior_need} is inconsistent")
        if not isclose(
            float(row[query_column]),
            float(row[descriptor_use_column])
            + float(row[state_only_vs_sampling_column])
            + float(row[sampling_direct_column]),
            rel_tol=0.0,
            abs_tol=EPS,
        ):
            raise ValueError("matched-acquisition Utility decomposition lost additivity")


def _validate_ert(row: dict) -> None:
    target = float(row["success_gap_target"])
    if not isfinite(target) or target <= 0.0:
        raise ValueError("success_gap_target must be finite and positive")
    if str(row["scientific_path_status_protocol"]) != SCIENTIFIC_PATH_STATUS_PROTOCOL:
        raise ValueError("scientific paths use the wrong Stage-A status protocol")
    scientific_prefixes = {
        "skip": "scientific_skip",
        "query_path": "scientific_query_joint",
        "query_matched_state_only_path": "scientific_query_matched_state_only",
        "sampling_only_path": "scientific_sampling_only_continue_current",
        "behavior_path": "scientific_behavior_only_full_budget",
    }
    for prefix in scientific_prefixes:
        observed_first_hit = row[f"{prefix}_observed_first_hit_FE"]
        target_hit_observed = bool(row[f"{prefix}_target_hit_observed"])
        target_hit_before_failure = bool(row[f"{prefix}_target_hit_before_failure"])
        endpoint_success = bool(row[f"{prefix}_endpoint_success"])
        first_hit = row[f"{prefix}_first_hit_FE"]
        success = bool(row[f"{prefix}_success"])
        scientific_prefix = scientific_prefixes[prefix]
        status = str(row[f"{scientific_prefix}_status"])
        timed_out = bool(row[f"{scientific_prefix}_timed_out"])
        completed = bool(row[f"{scientific_prefix}_path_completed"])
        if status not in TIMING_REPLAY_STATUSES:
            raise ValueError(f"{scientific_prefix}_status is unsupported")
        if completed != (status == "completed") or timed_out != (status == "timed_out"):
            raise ValueError(f"{scientific_prefix} status and flags disagree")
        expected_target_hit = observed_first_hit is not None
        if target_hit_observed != expected_target_hit:
            raise ValueError(
                f"{prefix} target_hit_observed is inconsistent with observed_first_hit_FE"
            )
        if first_hit != observed_first_hit:
            raise ValueError(f"{prefix} first_hit_FE must alias observed_first_hit_FE")
        if success != target_hit_observed:
            raise ValueError(f"{prefix} success must alias target_hit_observed")
        if target_hit_before_failure != (target_hit_observed and not completed):
            raise ValueError(f"{prefix} target_hit_before_failure is inconsistent")
        if endpoint_success != (target_hit_observed and completed):
            raise ValueError(f"{prefix} endpoint_success is inconsistent")
        planned = int(row[f"{prefix}_planned_FE"])
        effective = int(row[f"{prefix}_effective_FE"])
        if planned <= 0 or effective < 0 or effective > planned:
            raise ValueError(f"{prefix} planned/effective FE are inconsistent")
        if observed_first_hit is not None and not 0 < int(observed_first_hit) <= planned:
            raise ValueError(
                f"{prefix} observed_first_hit_FE is outside the planned path budget"
            )


def _validate_sampling_metadata(row: dict, *, actual_fe_ratio: float) -> None:
    if str(row["sampling_protocol"]) != SAMPLING_PROTOCOL:
        raise ValueError("utility-label sampling_protocol is inconsistent")
    target = float(row["monitor_target_ratio"])
    if actual_fe_ratio + EPS < target:
        raise ValueError("utility-label sample FE precedes monitor_target_ratio")
    if str(row["sampling_phase"]) != sampling_phase(target):
        raise ValueError("utility-label sampling_phase is inconsistent")
    milestone = bool(row["is_budget_milestone"])
    if milestone != is_budget_milestone(target):
        raise ValueError("utility-label budget milestone flag is inconsistent")
    milestone_ratio = row["budget_milestone_ratio"]
    if milestone:
        if milestone_ratio is None or float(milestone_ratio) != target:
            raise ValueError("utility-label budget_milestone_ratio is inconsistent")
    elif milestone_ratio is not None:
        raise ValueError("event-only utility labels must not define budget_milestone_ratio")
    flags = {name: bool(row[f"event_{name}"]) for name in EVENT_NAMES}
    if bool(row["is_event_sample"]) != any(flags.values()):
        raise ValueError("utility-label is_event_sample is inconsistent")
    expected_triggers = ["budget_milestone"] if milestone else []
    expected_triggers.extend(name for name in EVENT_NAMES if flags[name])
    if list(row["sampling_triggers"]) != expected_triggers:
        raise ValueError("utility-label sampling_triggers are inconsistent")
    event_index = row["event_index_in_phase"]
    if milestone and event_index is not None:
        raise ValueError("budget milestones must not consume the event-only quota")
    if not milestone and event_index is None:
        raise ValueError("event-only utility labels must define event_index_in_phase")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate paired offline Utility labels.")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    summary = validate_utility_label_file(args.input)
    print(f"validated {summary['rows']} paired Utility label rows")


if __name__ == "__main__":
    main()
