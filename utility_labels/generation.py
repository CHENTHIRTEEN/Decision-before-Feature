from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.phase1_batch_common import (
    TIMING_ORDER_PROTOCOL,
    TIMING_REPETITIONS,
    load_config,
    selected_dimensions,
    selected_functions,
    split_name,
)
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS, get_query_spec
from selection_reference.model import (
    BEHAVIOR_ONLY_SELECTION_REFERENCE_PROTOCOL,
    SELECTION_REFERENCE_PROTOCOL,
    SELECTOR_TARGET_TRANSFORM,
    STATE_ONLY_INPUT,
)
from selection_reference.action_losses import (
    BEHAVIOR_ONLY_FULL_BUDGET,
    QUERY_ADJUSTED_BUDGET,
    STATE_KEY_COLUMNS,
)
from trajectory.sampling import SAMPLING_METADATA_COLUMNS, SAMPLING_METADATA_SCHEMA_FIELDS
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
PAIRED_UTILITY_PROTOCOL = "joint_query_selector_with_matched_acquisition_controls"
COMPLETE_PATH_TIMING_SOURCE = "measured_complete_policy_path"
COMPLETE_PATH_TIMING_ORIGIN = "decision_state_to_terminal"


def paired_utility_label_view(
    *,
    query_selection: pd.DataFrame,
    behavior_selection: pd.DataFrame,
    query_adjusted_behavior_selection: pd.DataFrame,
    sampling_only_selection: pd.DataFrame,
    complete_path_timings: pd.DataFrame,
    query_id: str,
    log10_gap_floor: float,
    log10_gap_cap: float,
) -> pd.DataFrame:
    """Build the joint, full-budget, and matched-acquisition Utility estimands.

    Selector action choice uses clipped log10 gap advantage relative to native
    continuation. Utility uses clipped log10 terminal-gap differences and the
    log10 ratio of three-repeat complete-path wall-clock medians.
    """
    spec = get_query_spec(query_id)
    key = list(STATE_KEY_COLUMNS)
    floor = float(log10_gap_floor)
    cap = float(log10_gap_cap)
    if not np.isfinite(floor) or not np.isfinite(cap) or not 0.0 < floor < cap:
        raise ValueError("Utility log10 gap bounds must satisfy 0 < floor < cap")
    query_required = {
        *key,
        "query_id",
        "query_protocol",
        "action_budget_mode",
        "p_skip",
        "p_skip_raw",
        "loss_skip",
        "skip_status",
        "skip_failure_type",
        "skip_failure_message",
        "skip_prefix_first_hit_FE",
        "skip_continuation_first_hit_FE",
        "skip_observed_first_hit_FE",
        "skip_target_hit_observed",
        "skip_target_hit_before_failure",
        "skip_endpoint_success",
        "p_query",
        "skip_first_hit_FE",
        "skip_success",
        "skip_planned_FE",
        "skip_effective_FE",
        "skip_timed_out",
        "skip_path_completed",
        "query_path_observed_first_hit_FE",
        "query_path_target_hit_observed",
        "query_path_target_hit_before_failure",
        "query_path_endpoint_success",
        "query_path_first_hit_FE",
        "query_path_success",
        "query_path_planned_FE",
        "query_path_effective_FE",
        "query_path_timed_out",
        "query_path_completed",
        "continuation_only_gap",
        "query_sample_best_gap",
        "query_sample_improved_terminal",
        "query_sample_terminal_gap_improvement",
        "selected_algorithm",
        "selected_equals_default",
        "selected_equals_prefix",
        "handoff_required",
        "default_algorithm",
        "no_query_algorithm",
        "selected_transition_mode",
        "handoff_type",
        "best_observed_algorithm",
        "best_observed_loss",
        "selector_target_transform",
        "selection_reference_protocol",
    }
    behavior_required = {
        *key,
        "action_budget_mode",
        "p_skip",
        "p_skip_raw",
        "loss_skip",
        "skip_status",
        "skip_failure_type",
        "skip_failure_message",
        "skip_prefix_first_hit_FE",
        "skip_continuation_first_hit_FE",
        "skip_observed_first_hit_FE",
        "skip_target_hit_observed",
        "skip_target_hit_before_failure",
        "skip_endpoint_success",
        "p_behavior",
        "skip_first_hit_FE",
        "skip_success",
        "skip_planned_FE",
        "skip_effective_FE",
        "skip_timed_out",
        "skip_path_completed",
        "no_query_algorithm",
        "no_query_transition_mode",
        "behavior_path_observed_first_hit_FE",
        "behavior_path_target_hit_observed",
        "behavior_path_target_hit_before_failure",
        "behavior_path_endpoint_success",
        "behavior_path_first_hit_FE",
        "behavior_path_success",
        "behavior_path_planned_FE",
        "behavior_path_effective_FE",
        "behavior_path_timed_out",
        "behavior_path_completed",
        "selected_algorithm",
        "selected_equals_default",
        "selected_equals_prefix",
        "handoff_required",
        "handoff_type",
        "selector_target_transform",
        "selection_reference_protocol",
    }
    state_only_required = {
        *key,
        "action_budget_mode",
        "p_query",
        "selected_action_loss",
        "selected_algorithm",
        "selected_action",
        "selected_equals_default",
        "selected_equals_prefix",
        "handoff_required",
        "handoff_type",
        "query_path_observed_first_hit_FE",
        "query_path_target_hit_observed",
        "query_path_target_hit_before_failure",
        "query_path_endpoint_success",
        "query_path_first_hit_FE",
        "query_path_success",
        "query_path_planned_FE",
        "query_path_effective_FE",
        "query_path_timed_out",
        "query_path_completed",
        "selector_input_mode",
    }
    sampling_only_required = {
        *state_only_required,
        "descriptor_computation_required",
    }
    missing_query = sorted(query_required.difference(query_selection.columns))
    missing_behavior = sorted(behavior_required.difference(behavior_selection.columns))
    missing_state_only = sorted(
        state_only_required.difference(query_adjusted_behavior_selection.columns)
    )
    missing_sampling_only = sorted(
        sampling_only_required.difference(sampling_only_selection.columns)
    )
    if missing_query:
        raise ValueError(f"query Selector view is missing columns: {missing_query}")
    if missing_behavior:
        raise ValueError(f"behavior-only Selector view is missing columns: {missing_behavior}")
    if missing_state_only:
        raise ValueError(
            "query-adjusted Behavior-only diagnostic view is missing columns: "
            f"{missing_state_only}"
        )
    if missing_sampling_only:
        raise ValueError(
            "sampling-only continue-current view is missing columns: "
            f"{missing_sampling_only}"
        )
    if (
        query_selection.empty
        or behavior_selection.empty
        or query_adjusted_behavior_selection.empty
        or sampling_only_selection.empty
    ):
        raise ValueError("paired Utility construction requires non-empty Selector views")
    if (
        query_selection.duplicated(key).any()
        or behavior_selection.duplicated(key).any()
        or query_adjusted_behavior_selection.duplicated(key).any()
        or sampling_only_selection.duplicated(key).any()
    ):
        raise ValueError("paired Utility Selector views must contain unique state keys")
    if set(query_selection["query_id"].astype(str)) != {query_id}:
        raise ValueError("query Selector view uses the wrong query_id")
    if set(query_selection["query_protocol"].astype(str)) != {spec.protocol}:
        raise ValueError("query Selector view uses the wrong query protocol")
    if set(query_selection["action_budget_mode"].astype(str)) != {QUERY_ADJUSTED_BUDGET}:
        raise ValueError("joint query path requires query-adjusted action outcomes")
    if set(behavior_selection["action_budget_mode"].astype(str)) != {
        BEHAVIOR_ONLY_FULL_BUDGET
    }:
        raise ValueError("behavior-only path requires full-budget action outcomes")
    if set(query_adjusted_behavior_selection["action_budget_mode"].astype(str)) != {
        QUERY_ADJUSTED_BUDGET
    }:
        raise ValueError("predictive-increment diagnostic requires query-adjusted outcomes")
    if set(query_adjusted_behavior_selection["selector_input_mode"].astype(str)) != {
        STATE_ONLY_INPUT
    }:
        raise ValueError("predictive-increment diagnostic must use the state-only Selector")
    if sampling_only_selection["descriptor_computation_required"].astype(bool).any():
        raise ValueError("sampling-only path must not compute landscape descriptors")
    if not sampling_only_selection["selected_equals_prefix"].astype(bool).all():
        raise ValueError("sampling-only path must continue the prefix algorithm")
    if set(query_selection["selection_reference_protocol"].astype(str)) != {
        SELECTION_REFERENCE_PROTOCOL
    }:
        raise ValueError("query Selector view uses an unsupported protocol")
    if set(behavior_selection["selection_reference_protocol"].astype(str)) != {
        BEHAVIOR_ONLY_SELECTION_REFERENCE_PROTOCOL
    }:
        raise ValueError("behavior-only Selector view uses an unsupported protocol")

    behavior_columns = [
        *key,
        "p_skip",
        "p_skip_raw",
        "loss_skip",
        "skip_status",
        "skip_failure_type",
        "skip_failure_message",
        "skip_prefix_first_hit_FE",
        "skip_continuation_first_hit_FE",
        "skip_observed_first_hit_FE",
        "skip_target_hit_observed",
        "skip_target_hit_before_failure",
        "skip_endpoint_success",
        "skip_first_hit_FE",
        "skip_success",
        "skip_planned_FE",
        "skip_effective_FE",
        "no_query_algorithm",
        "no_query_transition_mode",
        "p_behavior",
        "skip_timed_out",
        "skip_path_completed",
        "selected_algorithm",
        "selected_action",
        "selected_equals_default",
        "selected_equals_prefix",
        "handoff_required",
        "handoff_type",
        "behavior_path_observed_first_hit_FE",
        "behavior_path_target_hit_observed",
        "behavior_path_target_hit_before_failure",
        "behavior_path_endpoint_success",
        "behavior_path_first_hit_FE",
        "behavior_path_success",
        "behavior_path_planned_FE",
        "behavior_path_effective_FE",
        "behavior_path_timed_out",
        "behavior_path_completed",
        "selector_prediction_source",
        "selector_input_mode",
        "selection_reference_protocol",
    ]
    paired = query_selection.merge(
        behavior_selection[behavior_columns].rename(
            columns={
                column: (
                    column
                    if column.startswith("behavior_path_")
                    else f"behavior_{column}"
                )
                for column in behavior_columns
                if column not in key
            }
        ),
        on=key,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if set(paired["_merge"].astype(str)) != {"both"}:
        raise ValueError("query and behavior-only Selector views must have identical state coverage")
    paired = paired.drop(columns="_merge")
    skip_loss = paired["p_skip"].to_numpy(dtype=float)
    for column in ("p_skip", "p_skip_raw", "loss_skip"):
        if not np.allclose(
            paired[column].to_numpy(dtype=float),
            paired[f"behavior_{column}"].to_numpy(dtype=float),
            rtol=0.0,
            atol=EPS,
        ):
            raise ValueError(f"paired Utility paths disagree on Stage-A {column}")
    for column in (
        "skip_target_hit_observed",
        "skip_target_hit_before_failure",
        "skip_endpoint_success",
        "skip_success",
        "skip_timed_out",
        "skip_path_completed",
    ):
        behavior_column = f"behavior_{column}"
        if not np.array_equal(
            paired[column].to_numpy(dtype=bool),
            paired[behavior_column].to_numpy(dtype=bool),
        ):
            raise ValueError(f"paired Utility paths disagree on Stage-A {column}")
    for column in (
        "skip_status",
        "skip_failure_type",
        "skip_failure_message",
        "no_query_algorithm",
        "no_query_transition_mode",
    ):
        if not np.array_equal(
            paired[column].astype(str).to_numpy(),
            paired[f"behavior_{column}"].astype(str).to_numpy(),
        ):
            raise ValueError(f"paired Utility paths disagree on Stage-A {column}")
    for column in (
        "skip_prefix_first_hit_FE",
        "skip_continuation_first_hit_FE",
        "skip_observed_first_hit_FE",
        "skip_first_hit_FE",
    ):
        if not np.array_equal(
            paired[column].fillna(-1).to_numpy(dtype=np.int64),
            paired[f"behavior_{column}"].fillna(-1).to_numpy(dtype=np.int64),
        ):
            raise ValueError(f"paired Utility paths disagree on Stage-A {column}")
    for column in ("skip_planned_FE", "skip_effective_FE"):
        if not np.array_equal(
            paired[column].to_numpy(dtype=np.int64),
            paired[f"behavior_{column}"].to_numpy(dtype=np.int64),
        ):
            raise ValueError(f"paired Utility paths disagree on Stage-A {column}")
    scientific_status_inputs = {
        "skip": ("skip_timed_out", "skip_path_completed"),
        "query_joint": ("query_path_timed_out", "query_path_completed"),
        "behavior_only_full_budget": (
            "behavior_path_timed_out",
            "behavior_path_completed",
        ),
    }
    paired["scientific_path_status_protocol"] = SCIENTIFIC_PATH_STATUS_PROTOCOL
    for path, (timed_out_column, completed_column) in scientific_status_inputs.items():
        timed_out_values = paired[timed_out_column].to_numpy(dtype=bool)
        completed_values = paired[completed_column].to_numpy(dtype=bool)
        if bool((timed_out_values & completed_values).any()):
            raise ValueError(f"Stage-A {path} cannot be both timed out and completed")
        paired[f"scientific_{path}_timed_out"] = timed_out_values
        paired[f"scientific_{path}_path_completed"] = completed_values
        paired[f"scientific_{path}_status"] = np.where(
            completed_values,
            "completed",
            np.where(timed_out_values, "timed_out", "failed"),
        )
    for prefix, completed_column in (
        ("skip", "skip_path_completed"),
        ("query_path", "query_path_completed"),
        ("behavior_path", "behavior_path_completed"),
    ):
        observed = paired[f"{prefix}_observed_first_hit_FE"]
        target_hit = observed.notna().to_numpy(dtype=bool)
        completed_values = paired[completed_column].to_numpy(dtype=bool)
        if not np.array_equal(
            paired[f"{prefix}_target_hit_observed"].to_numpy(dtype=bool), target_hit
        ):
            raise ValueError(f"{prefix} target_hit_observed is inconsistent")
        if not np.array_equal(
            paired[f"{prefix}_first_hit_FE"].fillna(-1).to_numpy(dtype=np.int64),
            observed.fillna(-1).to_numpy(dtype=np.int64),
        ):
            raise ValueError(f"{prefix} first_hit_FE must alias observed_first_hit_FE")
        if not np.array_equal(
            paired[f"{prefix}_success"].to_numpy(dtype=bool), target_hit
        ):
            raise ValueError(f"{prefix} success must alias target_hit_observed")
        if not np.array_equal(
            paired[f"{prefix}_target_hit_before_failure"].to_numpy(dtype=bool),
            target_hit & ~completed_values,
        ):
            raise ValueError(f"{prefix} target_hit_before_failure is inconsistent")
        if not np.array_equal(
            paired[f"{prefix}_endpoint_success"].to_numpy(dtype=bool),
            target_hit & completed_values,
        ):
            raise ValueError(f"{prefix} endpoint_success is inconsistent")
    matched_columns = [
        *key,
        "p_query",
        "selected_action_loss",
        "selected_algorithm",
        "selected_action",
        "selected_equals_default",
        "selected_equals_prefix",
        "handoff_required",
        "handoff_type",
        "query_path_observed_first_hit_FE",
        "query_path_target_hit_observed",
        "query_path_target_hit_before_failure",
        "query_path_endpoint_success",
        "query_path_first_hit_FE",
        "query_path_success",
        "query_path_planned_FE",
        "query_path_effective_FE",
        "query_path_timed_out",
        "query_path_completed",
    ]

    state_only_rename = {
        "p_query": "p_query_matched_state_only",
        "selected_action_loss": "query_matched_state_only_continuation_only_gap",
        **{
            column: f"query_matched_state_only_{column}"
            for column in (
                "selected_algorithm",
                "selected_action",
                "selected_equals_default",
                "selected_equals_prefix",
                "handoff_required",
                "handoff_type",
            )
        },
        **{
            column: column.replace(
                "query_path_",
                "query_matched_state_only_path_",
            )
            for column in matched_columns
            if column.startswith("query_path_")
        },
    }
    state_only = query_adjusted_behavior_selection[matched_columns].rename(
        columns=state_only_rename
    )
    paired = paired.merge(state_only, on=key, how="left", validate="one_to_one")
    if paired["p_query_matched_state_only"].isna().any():
        raise ValueError("query-matched state-only path has incomplete state coverage")

    sampling_rename = {
        "p_query": "p_sampling_only_continue_current",
        "selected_action_loss": "sampling_only_continuation_only_gap",
        **{
            column: f"sampling_only_{column}"
            for column in (
                "selected_algorithm",
                "selected_action",
                "selected_equals_default",
                "selected_equals_prefix",
                "handoff_required",
                "handoff_type",
            )
        },
        **{
            column: column.replace("query_path_", "sampling_only_path_")
            for column in matched_columns
            if column.startswith("query_path_")
        },
    }
    sampling_only = sampling_only_selection[matched_columns].rename(
        columns=sampling_rename
    )
    paired = paired.merge(sampling_only, on=key, how="left", validate="one_to_one")
    if paired["p_sampling_only_continue_current"].isna().any():
        raise ValueError("sampling-only path has incomplete state coverage")
    for scientific_path, endpoint_prefix in (
        ("query_matched_state_only", "query_matched_state_only_path"),
        ("sampling_only_continue_current", "sampling_only_path"),
    ):
        timed_out_values = paired[f"{endpoint_prefix}_timed_out"].to_numpy(dtype=bool)
        completed_values = paired[f"{endpoint_prefix}_completed"].to_numpy(dtype=bool)
        if bool((timed_out_values & completed_values).any()):
            raise ValueError(
                f"Stage-A {scientific_path} cannot be both timed out and completed"
            )
        paired[f"scientific_{scientific_path}_timed_out"] = timed_out_values
        paired[f"scientific_{scientific_path}_path_completed"] = completed_values
        paired[f"scientific_{scientific_path}_status"] = np.where(
            completed_values,
            "completed",
            np.where(timed_out_values, "timed_out", "failed"),
        )
        observed = paired[f"{endpoint_prefix}_observed_first_hit_FE"]
        target_hit = observed.notna().to_numpy(dtype=bool)
        if not np.array_equal(
            paired[f"{endpoint_prefix}_target_hit_observed"].to_numpy(dtype=bool),
            target_hit,
        ):
            raise ValueError(f"{endpoint_prefix} target-hit fields are inconsistent")
        if not np.array_equal(
            paired[f"{endpoint_prefix}_first_hit_FE"].fillna(-1).to_numpy(
                dtype=np.int64
            ),
            observed.fillna(-1).to_numpy(dtype=np.int64),
        ):
            raise ValueError(f"{endpoint_prefix} first-hit alias is inconsistent")
        if not np.array_equal(
            paired[f"{endpoint_prefix}_success"].to_numpy(dtype=bool),
            target_hit,
        ):
            raise ValueError(f"{endpoint_prefix} success alias is inconsistent")
        if not np.array_equal(
            paired[f"{endpoint_prefix}_target_hit_before_failure"].to_numpy(dtype=bool),
            target_hit & ~completed_values,
        ):
            raise ValueError(f"{endpoint_prefix} hit-before-failure is inconsistent")
        if not np.array_equal(
            paired[f"{endpoint_prefix}_endpoint_success"].to_numpy(dtype=bool),
            target_hit & completed_values,
        ):
            raise ValueError(f"{endpoint_prefix} endpoint success is inconsistent")
    query_loss = paired["p_query"].to_numpy(dtype=float)
    query_continuation_loss = paired["continuation_only_gap"].to_numpy(dtype=float)
    behavior_loss = paired["behavior_p_behavior"].to_numpy(dtype=float)
    state_only_loss = paired["p_query_matched_state_only"].to_numpy(dtype=float)
    sampling_only_loss = paired["p_sampling_only_continue_current"].to_numpy(
        dtype=float
    )
    losses = np.column_stack(
        [skip_loss, query_loss, behavior_loss, state_only_loss, sampling_only_loss]
    )
    if not np.isfinite(losses).all() or (losses < 0.0).any():
        raise ValueError("paired Utility losses must be finite and non-negative")
    clipped_skip = np.clip(skip_loss, floor, cap)
    clipped_query = np.clip(query_loss, floor, cap)
    clipped_behavior = np.clip(behavior_loss, floor, cap)
    clipped_state_only = np.clip(state_only_loss, floor, cap)
    clipped_sampling_only = np.clip(sampling_only_loss, floor, cap)
    log_skip = np.log10(clipped_skip)
    log_query = np.log10(clipped_query)
    log_behavior = np.log10(clipped_behavior)
    log_state_only = np.log10(clipped_state_only)
    log_sampling_only = np.log10(clipped_sampling_only)
    query_gain = log_skip - log_query
    behavior_gain = log_skip - log_behavior
    query_vs_behavior_gain = log_behavior - log_query
    state_only_gain = log_skip - log_state_only
    sampling_only_gain = log_skip - log_sampling_only
    descriptor_use_gap_increment = log_state_only - log_query
    state_only_vs_sampling_gap_increment = log_sampling_only - log_state_only
    sampling_direct_gap_increment = sampling_only_gain
    state_only_gap = paired[
        "query_matched_state_only_continuation_only_gap"
    ].to_numpy(dtype=float)
    if not np.isfinite(state_only_gap).all() or (state_only_gap < 0.0).any():
        raise ValueError("query-adjusted state-only selected gaps must be finite and non-negative")
    predictive_increment = np.log10(np.clip(state_only_gap, floor, cap)) - np.log10(
        np.clip(query_continuation_loss, floor, cap)
    )

    paired = _join_complete_path_timings(
        paired=paired,
        complete_path_timings=complete_path_timings,
        key=key,
    )
    skip_runtime = paired["runtime_skip_median"].to_numpy(dtype=float)
    query_runtime = paired["runtime_query_joint_median"].to_numpy(dtype=float)
    behavior_runtime = paired[
        "runtime_behavior_only_full_budget_median"
    ].to_numpy(dtype=float)
    state_only_runtime = paired[
        "runtime_query_matched_state_only_median"
    ].to_numpy(dtype=float)
    sampling_only_runtime = paired[
        "runtime_sampling_only_continue_current_median"
    ].to_numpy(dtype=float)
    query_time_cost = np.log10(query_runtime / skip_runtime)
    behavior_time_cost = np.log10(behavior_runtime / skip_runtime)
    query_vs_behavior_time_cost = np.log10(query_runtime / behavior_runtime)
    state_only_time_cost = np.log10(state_only_runtime / skip_runtime)
    sampling_only_time_cost = np.log10(sampling_only_runtime / skip_runtime)
    descriptor_use_time_cost = np.log10(query_runtime / state_only_runtime)
    state_only_vs_sampling_time_cost = np.log10(
        state_only_runtime / sampling_only_runtime
    )

    output = paired.copy()
    output["utility_protocol"] = PAIRED_UTILITY_PROTOCOL
    output["utility_formula_protocol"] = (
        "clipped_log10_gap_difference_minus_lambda_log10_complete_path_runtime_ratio"
    )
    output["log10_gap_floor"] = floor
    output["log10_gap_cap"] = cap
    output["log10_gap_skip"] = log_skip
    output["log10_gap_query_joint"] = log_query
    output["log10_gap_behavior_only_full_budget"] = log_behavior
    output["log10_gap_query_matched_state_only"] = log_state_only
    output["log10_gap_sampling_only_continue_current"] = log_sampling_only
    output["query_joint_log10_gap_improvement"] = query_gain
    output["behavior_only_log10_gap_improvement"] = behavior_gain
    output["query_vs_behavior_log10_gap_improvement"] = query_vs_behavior_gain
    output["query_feature_predictive_increment_log10_gap"] = predictive_increment
    output["query_descriptor_use_log10_gap_increment"] = descriptor_use_gap_increment
    output["query_sampling_direct_log10_gap_increment"] = sampling_direct_gap_increment
    output["performance_gain_raw"] = skip_loss - query_loss
    output["performance_gain_norm"] = query_gain
    output["behavior_performance_gain_norm"] = behavior_gain
    output["runtime_query_total"] = query_runtime
    output["runtime_no_query_total"] = skip_runtime
    output["runtime_behavior_total"] = behavior_runtime
    output["runtime_query_optimization"] = output[
        "runtime_selected_action_optimization"
    ].to_numpy(dtype=float)
    output["runtime_net"] = query_runtime - skip_runtime
    output["time_cost_log10_ratio"] = query_time_cost
    output["behavior_time_cost_log10_ratio"] = behavior_time_cost
    output["query_vs_behavior_time_cost_log10_ratio"] = query_vs_behavior_time_cost
    output["peak_memory_measurement_status"] = "not_available"
    output["peak_memory_bytes_skip"] = np.nan
    output["peak_memory_bytes_query_joint"] = np.nan
    output["peak_memory_bytes_behavior_only_full_budget"] = np.nan
    for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES:
        output[f"peak_memory_bytes_{suffix}"] = np.nan
    output["selection_reference_default_algorithm"] = output["default_algorithm"].astype(str)
    output["skip_switches_from_prefix"] = (
        output["default_algorithm"].astype(str)
        != output["prefix_algorithm"].astype(str)
    )
    output["query_transition_mode"] = output["selected_transition_mode"].astype(str)
    output["potential_gain_raw"] = (
        output["p_skip"].to_numpy(dtype=float)
        - output["best_observed_loss"].to_numpy(dtype=float)
    )
    output["selector_regret_raw"] = (
        output["selected_action_loss"].to_numpy(dtype=float)
        - output["best_observed_loss"].to_numpy(dtype=float)
    )
    output["selected_matches_best_observed"] = (
        output["selected_algorithm"].astype(str)
        == output["best_observed_algorithm"].astype(str)
    )
    for (
        weight,
        joint_column,
        behavior_column,
        increment_column,
        state_only_column,
        sampling_only_column,
        descriptor_use_column,
        state_only_vs_sampling_column,
        sampling_direct_column,
        need_column,
        behavior_need_column,
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
        joint = query_gain - float(weight) * query_time_cost
        behavior = behavior_gain - float(weight) * behavior_time_cost
        increment = query_vs_behavior_gain - float(weight) * query_vs_behavior_time_cost
        state_only_utility = state_only_gain - float(weight) * state_only_time_cost
        sampling_only_utility = sampling_only_gain - float(weight) * sampling_only_time_cost
        descriptor_use_increment = descriptor_use_gap_increment - float(
            weight
        ) * descriptor_use_time_cost
        state_only_vs_sampling_increment = state_only_vs_sampling_gap_increment - float(
            weight
        ) * state_only_vs_sampling_time_cost
        sampling_direct_increment = sampling_direct_gap_increment - float(
            weight
        ) * sampling_only_time_cost
        output[joint_column] = joint
        output[behavior_column] = behavior
        output[increment_column] = increment
        output[state_only_column] = state_only_utility
        output[sampling_only_column] = sampling_only_utility
        output[descriptor_use_column] = descriptor_use_increment
        output[state_only_vs_sampling_column] = state_only_vs_sampling_increment
        output[sampling_direct_column] = sampling_direct_increment
        output[need_column] = joint > 0.0
        output[behavior_need_column] = behavior > 0.0
        if not np.allclose(
            output[joint_column].to_numpy(dtype=float)
            - output[behavior_column].to_numpy(dtype=float),
            output[increment_column].to_numpy(dtype=float),
            rtol=0.0,
            atol=EPS,
        ):
            raise RuntimeError("query operational Utility increment lost exact additivity")
        if not np.allclose(
            output[joint_column].to_numpy(dtype=float),
            output[descriptor_use_column].to_numpy(dtype=float)
            + output[state_only_vs_sampling_column].to_numpy(dtype=float)
            + output[sampling_direct_column].to_numpy(dtype=float),
            rtol=0.0,
            atol=EPS,
        ):
            raise RuntimeError("matched-acquisition Utility decomposition lost exact additivity")
    return output.sort_values(key).reset_index(drop=True)


def _join_complete_path_timings(
    *,
    paired: pd.DataFrame,
    complete_path_timings: pd.DataFrame,
    key: list[str],
) -> pd.DataFrame:
    required = {
        *key,
        "path",
        "repetition_index",
        "order_position",
        "runtime_seconds",
        "timing_repetitions",
        "timing_order_protocol",
        "timing_source",
        "timing_origin",
        "timing_environment_id",
        "thread_count",
        "selected_algorithm",
        "terminal_gap",
        "observed_first_hit_FE",
        "target_hit_observed",
        "target_hit_before_failure",
        "endpoint_success",
        "first_hit_FE",
        "success",
        "planned_FE",
        "effective_FE",
        "timed_out",
        "path_completed",
        "timing_replay_status",
        "timing_replay_timeout_seconds",
    }
    missing = sorted(required.difference(complete_path_timings.columns))
    if missing:
        raise ValueError(f"complete-path timing input is missing columns: {missing}")
    timings = complete_path_timings.copy()
    expected_paths = (
        "skip",
        "query_joint",
        "query_matched_state_only",
        "sampling_only_continue_current",
        "behavior_only_full_budget",
    )
    if set(timings["path"].astype(str)) != set(expected_paths):
        raise ValueError("complete-path timing input must contain exactly five policy paths")
    if set(timings["timing_repetitions"].astype(int)) != {TIMING_REPETITIONS}:
        raise ValueError("complete-path timing input must use exactly three repetitions")
    if set(timings["timing_order_protocol"].astype(str)) != {
        TIMING_ORDER_PROTOCOL
    }:
        raise ValueError("complete-path timing input uses the wrong cyclic order protocol")
    if set(timings["timing_source"].astype(str)) != {COMPLETE_PATH_TIMING_SOURCE}:
        raise ValueError(
            "Utility timing must be measured over complete policy paths; "
            "component-runtime sums are not accepted"
        )
    if set(timings["timing_origin"].astype(str)) != {COMPLETE_PATH_TIMING_ORIGIN}:
        raise ValueError(
            "nested Utility timing must run from the saved decision state to the "
            "terminal budget; the shared prefix is a sunk cost"
        )
    runtime = timings["runtime_seconds"].to_numpy(dtype=float)
    if not np.isfinite(runtime).all() or (runtime <= 0.0).any():
        raise ValueError("complete-path runtimes must be finite and strictly positive")
    if timings["timing_environment_id"].isna().any() or bool(
        timings["timing_environment_id"].astype(str).str.len().eq(0).any()
    ):
        raise ValueError("complete-path timing requires a non-empty timing_environment_id")
    if bool((timings["thread_count"].astype(int) <= 0).any()):
        raise ValueError("complete-path timing thread_count must be a positive integer")
    replay_timeout = timings["timing_replay_timeout_seconds"].to_numpy(dtype=float)
    if not np.isfinite(replay_timeout).all() or bool((replay_timeout <= 0.0).any()):
        raise ValueError("timing replay timeout seconds must be finite and strictly positive")
    terminal_gaps = timings["terminal_gap"].to_numpy(dtype=float)
    if not np.isfinite(terminal_gaps).all() or bool((terminal_gaps < 0.0).any()):
        raise ValueError("complete-path terminal gaps must be finite and non-negative")
    planned = timings["planned_FE"].to_numpy(dtype=int)
    effective = timings["effective_FE"].to_numpy(dtype=int)
    if bool((planned <= 0).any()) or bool((effective < 0).any()) or bool((effective > planned).any()):
        raise ValueError("complete-path planned/effective FE accounting is inconsistent")
    completed = timings["path_completed"].to_numpy(dtype=bool)
    timed_out = timings["timed_out"].to_numpy(dtype=bool)
    statuses = timings["timing_replay_status"].astype(str)
    if not set(statuses).issubset(TIMING_REPLAY_STATUSES):
        raise ValueError(
            "timing_replay_status must be one of completed, timed_out, or failed"
        )
    if not np.array_equal(completed, statuses.eq("completed").to_numpy(dtype=bool)):
        raise ValueError(
            "path_completed must be true exactly for timing_replay_status=completed"
        )
    if not np.array_equal(timed_out, statuses.eq("timed_out").to_numpy(dtype=bool)):
        raise ValueError(
            "timed_out must be true exactly for timing_replay_status=timed_out"
        )
    target_hit_observed = timings["observed_first_hit_FE"].notna().to_numpy(dtype=bool)
    if not np.array_equal(
        timings["target_hit_observed"].to_numpy(dtype=bool), target_hit_observed
    ):
        raise ValueError(
            "timing replay target_hit_observed must equal observed_first_hit_FE is not null"
        )
    if not np.array_equal(
        timings["first_hit_FE"].fillna(-1).to_numpy(dtype=np.int64),
        timings["observed_first_hit_FE"].fillna(-1).to_numpy(dtype=np.int64),
    ):
        raise ValueError("timing replay first_hit_FE must alias observed_first_hit_FE")
    if not np.array_equal(
        timings["success"].to_numpy(dtype=bool), target_hit_observed
    ):
        raise ValueError("timing replay success must alias target_hit_observed")
    if not np.array_equal(
        timings["target_hit_before_failure"].to_numpy(dtype=bool),
        target_hit_observed & ~completed,
    ):
        raise ValueError("timing replay target_hit_before_failure is inconsistent")
    if not np.array_equal(
        timings["endpoint_success"].to_numpy(dtype=bool),
        target_hit_observed & completed,
    ):
        raise ValueError("timing replay endpoint_success is inconsistent")
    timing_key = [*key, "path", "repetition_index"]
    if timings.duplicated(timing_key).any():
        raise ValueError("complete-path timing input contains duplicate repetitions")
    counts = timings.groupby([*key, "path"], dropna=False).size()
    if not bool((counts == TIMING_REPETITIONS).all()):
        raise ValueError("every state/path must contain exactly three timing repetitions")
    repetitions = timings.groupby([*key, "path"], dropna=False)["repetition_index"].agg(
        lambda values: tuple(sorted(int(value) for value in values))
    )
    if not bool((repetitions == tuple(range(TIMING_REPETITIONS))).all()):
        raise ValueError("complete-path repetition indices must be exactly 0,1,2")
    for _, group in timings.groupby([*key, "repetition_index"], dropna=False):
        if set(group["path"].astype(str)) != set(expected_paths):
            raise ValueError("each timing repetition must contain all five policy paths")
        if set(group["order_position"].astype(int)) != set(range(len(expected_paths))):
            raise ValueError("each timing repetition must use unique cyclic order positions 0,...,4")
    path_positions = timings.groupby([*key, "path"], dropna=False)["order_position"].agg(
        lambda values: frozenset(int(value) for value in values)
    )
    if not path_positions.map(lambda values: len(values) == TIMING_REPETITIONS).all():
        raise ValueError(
            "three-repeat cyclic timing requires each path to occupy three distinct order positions"
        )
    environment_counts = timings.groupby(key, dropna=False)["timing_environment_id"].nunique()
    thread_counts = timings.groupby(key, dropna=False)["thread_count"].nunique()
    timeout_counts = timings.groupby(key, dropna=False)[
        "timing_replay_timeout_seconds"
    ].nunique()
    if (
        not bool((environment_counts == 1).all())
        or not bool((thread_counts == 1).all())
        or not bool((timeout_counts == 1).all())
    ):
        raise ValueError(
            "all paths and repetitions at one state must share one timing environment, "
            "thread count, and replay timeout"
        )
    invariant_fields = ("selected_algorithm", "planned_FE")
    for state_path, repetitions_frame in timings.groupby([*key, "path"], dropna=False):
        first = repetitions_frame.iloc[0]
        for column in invariant_fields:
            values = repetitions_frame[column]
            equal = values.astype(str).eq(str(first[column])).all()
            if not bool(equal):
                raise ValueError(
                    "complete-path identity changed across timing repetitions: "
                    f"state_path={state_path}, field={column}"
                )
    expected_algorithms_by_path = {
        "skip": paired[key + ["no_query_algorithm"]].rename(
            columns={"no_query_algorithm": "selected_algorithm"}
        ),
        "query_joint": paired[key + ["selected_algorithm"]],
        "query_matched_state_only": paired[
            key + ["query_matched_state_only_selected_algorithm"]
        ].rename(
            columns={
                "query_matched_state_only_selected_algorithm": "selected_algorithm"
            }
        ),
        "sampling_only_continue_current": paired[
            key + ["sampling_only_selected_algorithm"]
        ].rename(columns={"sampling_only_selected_algorithm": "selected_algorithm"}),
        "behavior_only_full_budget": paired[
            key + ["behavior_selected_algorithm"]
        ].rename(columns={"behavior_selected_algorithm": "selected_algorithm"}),
    }
    for path, expected in expected_algorithms_by_path.items():
        observed = timings[timings["path"].astype(str) == path][
            key + ["selected_algorithm"]
        ].drop_duplicates()
        comparison = expected.merge(
            observed,
            on=key,
            how="outer",
            suffixes=("_expected", "_timed"),
            indicator=True,
            validate="one_to_one",
        )
        if set(comparison["_merge"].astype(str)) != {"both"} or not (
            comparison["selected_algorithm_expected"].astype(str)
            == comparison["selected_algorithm_timed"].astype(str)
        ).all():
            raise ValueError(f"{path} complete-path timings use a different selected action")
    expected_endpoints_by_path = {
        "skip": paired[
            key
            + [
                "p_skip",
                "skip_observed_first_hit_FE",
                "skip_target_hit_observed",
                "skip_endpoint_success",
                "skip_first_hit_FE",
                "skip_success",
                "skip_planned_FE",
                "scientific_skip_path_completed",
            ]
        ].rename(
            columns={
                "p_skip": "expected_terminal_gap",
                "skip_observed_first_hit_FE": "expected_observed_first_hit_FE",
                "skip_target_hit_observed": "expected_target_hit_observed",
                "skip_endpoint_success": "expected_endpoint_success",
                "skip_first_hit_FE": "expected_first_hit_FE",
                "skip_success": "expected_success",
                "skip_planned_FE": "expected_planned_FE",
                "scientific_skip_path_completed": "expected_path_completed",
            }
        ),
        "query_joint": paired[
            key
            + [
                "p_query",
                "query_path_observed_first_hit_FE",
                "query_path_target_hit_observed",
                "query_path_endpoint_success",
                "query_path_first_hit_FE",
                "query_path_success",
                "query_path_planned_FE",
                "scientific_query_joint_path_completed",
            ]
        ].rename(
            columns={
                "p_query": "expected_terminal_gap",
                "query_path_observed_first_hit_FE": "expected_observed_first_hit_FE",
                "query_path_target_hit_observed": "expected_target_hit_observed",
                "query_path_endpoint_success": "expected_endpoint_success",
                "query_path_first_hit_FE": "expected_first_hit_FE",
                "query_path_success": "expected_success",
                "query_path_planned_FE": "expected_planned_FE",
                "scientific_query_joint_path_completed": "expected_path_completed",
            }
        ),
        "query_matched_state_only": paired[
            key
            + [
                "p_query_matched_state_only",
                "query_matched_state_only_path_observed_first_hit_FE",
                "query_matched_state_only_path_target_hit_observed",
                "query_matched_state_only_path_endpoint_success",
                "query_matched_state_only_path_first_hit_FE",
                "query_matched_state_only_path_success",
                "query_matched_state_only_path_planned_FE",
                "scientific_query_matched_state_only_path_completed",
            ]
        ].rename(
            columns={
                "p_query_matched_state_only": "expected_terminal_gap",
                "query_matched_state_only_path_observed_first_hit_FE": (
                    "expected_observed_first_hit_FE"
                ),
                "query_matched_state_only_path_target_hit_observed": (
                    "expected_target_hit_observed"
                ),
                "query_matched_state_only_path_endpoint_success": (
                    "expected_endpoint_success"
                ),
                "query_matched_state_only_path_first_hit_FE": "expected_first_hit_FE",
                "query_matched_state_only_path_success": "expected_success",
                "query_matched_state_only_path_planned_FE": "expected_planned_FE",
                "scientific_query_matched_state_only_path_completed": (
                    "expected_path_completed"
                ),
            }
        ),
        "sampling_only_continue_current": paired[
            key
            + [
                "p_sampling_only_continue_current",
                "sampling_only_path_observed_first_hit_FE",
                "sampling_only_path_target_hit_observed",
                "sampling_only_path_endpoint_success",
                "sampling_only_path_first_hit_FE",
                "sampling_only_path_success",
                "sampling_only_path_planned_FE",
                "scientific_sampling_only_continue_current_path_completed",
            ]
        ].rename(
            columns={
                "p_sampling_only_continue_current": "expected_terminal_gap",
                "sampling_only_path_observed_first_hit_FE": (
                    "expected_observed_first_hit_FE"
                ),
                "sampling_only_path_target_hit_observed": "expected_target_hit_observed",
                "sampling_only_path_endpoint_success": "expected_endpoint_success",
                "sampling_only_path_first_hit_FE": "expected_first_hit_FE",
                "sampling_only_path_success": "expected_success",
                "sampling_only_path_planned_FE": "expected_planned_FE",
                "scientific_sampling_only_continue_current_path_completed": (
                    "expected_path_completed"
                ),
            }
        ),
        "behavior_only_full_budget": paired[
            key
            + [
                "behavior_p_behavior",
                "behavior_path_observed_first_hit_FE",
                "behavior_path_target_hit_observed",
                "behavior_path_endpoint_success",
                "behavior_path_first_hit_FE",
                "behavior_path_success",
                "behavior_path_planned_FE",
                "scientific_behavior_only_full_budget_path_completed",
            ]
        ].rename(
            columns={
                "behavior_p_behavior": "expected_terminal_gap",
                "behavior_path_observed_first_hit_FE": "expected_observed_first_hit_FE",
                "behavior_path_target_hit_observed": "expected_target_hit_observed",
                "behavior_path_endpoint_success": "expected_endpoint_success",
                "behavior_path_first_hit_FE": "expected_first_hit_FE",
                "behavior_path_success": "expected_success",
                "behavior_path_planned_FE": "expected_planned_FE",
                "scientific_behavior_only_full_budget_path_completed": (
                    "expected_path_completed"
                ),
            }
        ),
    }
    consistency_by_state_path: dict[tuple[tuple[object, ...], str], dict[str, object]] = {}
    for path, expected in expected_endpoints_by_path.items():
        path_rows = timings[timings["path"].astype(str) == path].copy()
        coverage = expected[key].merge(
            path_rows[key].drop_duplicates(),
            on=key,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if set(coverage["_merge"].astype(str)) != {"both"}:
            raise ValueError(f"{path} timing coverage does not match its Stage-A outcomes")
        comparison = path_rows.merge(expected, on=key, how="left", validate="many_to_one")
        if not np.array_equal(
            comparison["expected_planned_FE"].to_numpy(dtype=int),
            comparison["planned_FE"].to_numpy(dtype=int),
        ):
            raise ValueError(f"{path} timing replays use the wrong planned FE")
        for state_key, state_replays in comparison.groupby(key, dropna=False, sort=False):
            state_key_tuple = state_key if isinstance(state_key, tuple) else (state_key,)
            completed_replays = state_replays[
                state_replays["timing_replay_status"].astype(str).eq("completed")
            ]
            stage_a_completed = bool(state_replays["expected_path_completed"].iloc[0])
            internal_consistent: bool | None = None
            if len(completed_replays) >= 2:
                internal_consistent = _completed_timing_endpoints_match(
                    completed_replays,
                    terminal_gap=float(completed_replays.iloc[0]["terminal_gap"]),
                    observed_first_hit_fe=completed_replays.iloc[0][
                        "observed_first_hit_FE"
                    ],
                    target_hit_observed=bool(
                        completed_replays.iloc[0]["target_hit_observed"]
                    ),
                    endpoint_success=bool(completed_replays.iloc[0]["endpoint_success"]),
                )
                if not internal_consistent:
                    raise ValueError(
                        f"completed {path} timing replays disagree internally: "
                        f"state={state_key_tuple}"
                    )
            stage_a_consistent: bool | None = None
            if stage_a_completed and not completed_replays.empty:
                stage_a_consistent = _completed_timing_endpoints_match(
                    completed_replays,
                    terminal_gap=float(state_replays["expected_terminal_gap"].iloc[0]),
                    observed_first_hit_fe=state_replays[
                        "expected_observed_first_hit_FE"
                    ].iloc[0],
                    target_hit_observed=bool(
                        state_replays["expected_target_hit_observed"].iloc[0]
                    ),
                    endpoint_success=bool(
                        state_replays["expected_endpoint_success"].iloc[0]
                    ),
                )
                if not stage_a_consistent:
                    raise ValueError(
                        f"completed {path} timing replays disagree with Stage A: "
                        f"state={state_key_tuple}"
                    )
            replay_statuses = state_replays["timing_replay_status"].astype(str).tolist()
            replay_completed = state_replays["path_completed"].to_numpy(dtype=bool)
            consistency_by_state_path[(state_key_tuple, path)] = {
                "path_identity_consistent": True,
                "completed_outcomes_internally_consistent": internal_consistent,
                "stage_a_to_completed_replays_consistent": stage_a_consistent,
                "status_instability": len(set(replay_statuses)) > 1,
                "stage_a_stage_b_completion_status_instability": bool(
                    (replay_completed != stage_a_completed).any()
                ),
            }
    aggregate_rows: list[dict[str, object]] = []
    for state_key, state in timings.groupby(key, dropna=False, sort=False):
        row = dict(zip(key, state_key if isinstance(state_key, tuple) else (state_key,), strict=True))
        sources: list[str] = []
        replay_instability = False
        state_key_tuple = state_key if isinstance(state_key, tuple) else (state_key,)
        for path in expected_paths:
            path_rows = state[state["path"].astype(str) == path].sort_values("repetition_index")
            values = [float(value) for value in path_rows["runtime_seconds"]]
            positions = [int(value) for value in path_rows["order_position"]]
            replay_statuses = [str(value) for value in path_rows["timing_replay_status"]]
            replay_effective_fe = [int(value) for value in path_rows["effective_FE"]]
            replay_observed_first_hit_fe = [
                None if pd.isna(value) else int(value)
                for value in path_rows["observed_first_hit_FE"]
            ]
            replay_target_hit_flags = [
                bool(value) for value in path_rows["target_hit_observed"]
            ]
            replay_hit_before_failure_flags = [
                bool(value) for value in path_rows["target_hit_before_failure"]
            ]
            replay_endpoint_success_flags = [
                bool(value) for value in path_rows["endpoint_success"]
            ]
            replay_timeout_flags = [bool(value) for value in path_rows["timed_out"]]
            replay_completion_flags = [bool(value) for value in path_rows["path_completed"]]
            suffix = {
                "skip": "skip",
                "query_joint": "query_joint",
                "query_matched_state_only": "query_matched_state_only",
                "sampling_only_continue_current": "sampling_only_continue_current",
                "behavior_only_full_budget": "behavior_only_full_budget",
            }[path]
            timeout_seconds = float(path_rows["timing_replay_timeout_seconds"].iloc[0])
            censored_values = [
                value if status == "completed" else max(value, timeout_seconds)
                for value, status in zip(values, replay_statuses, strict=True)
            ]
            row[f"runtime_{suffix}_repetitions"] = values
            row[f"runtime_{suffix}_raw_observed_median"] = float(np.median(values))
            row[f"runtime_{suffix}_censored_repetitions"] = censored_values
            row[f"runtime_{suffix}_median"] = float(np.median(censored_values))
            # Transitional compatibility aliases; both are exactly the censored values.
            row[f"runtime_{suffix}_failure_worst_case_repetitions"] = (
                censored_values
            )
            row[f"runtime_{suffix}_failure_worst_case_median"] = float(
                np.median(censored_values)
            )
            row[f"timing_order_positions_{suffix}"] = positions
            row[f"timing_replay_status_repetitions_{suffix}"] = replay_statuses
            row[f"timing_replay_effective_FE_repetitions_{suffix}"] = replay_effective_fe
            row[f"timing_replay_observed_first_hit_FE_repetitions_{suffix}"] = (
                replay_observed_first_hit_fe
            )
            row[f"timing_replay_target_hit_observed_flags_{suffix}"] = (
                replay_target_hit_flags
            )
            row[f"timing_replay_target_hit_before_failure_flags_{suffix}"] = (
                replay_hit_before_failure_flags
            )
            row[f"timing_replay_endpoint_success_flags_{suffix}"] = (
                replay_endpoint_success_flags
            )
            row[f"timing_replay_timed_out_flags_{suffix}"] = replay_timeout_flags
            row[f"timing_replay_path_completed_flags_{suffix}"] = replay_completion_flags
            row[f"timing_replay_completed_repetitions_{suffix}"] = int(
                sum(replay_completion_flags)
            )
            row[f"timing_replay_timeout_repetitions_{suffix}"] = int(
                sum(replay_timeout_flags)
            )
            row[f"timing_replay_failure_repetitions_{suffix}"] = int(
                sum(status == "failed" for status in replay_statuses)
            )
            consistency = consistency_by_state_path[(state_key_tuple, path)]
            status_instability = bool(consistency["status_instability"])
            row[f"timing_replay_path_identity_consistent_{suffix}"] = bool(
                consistency["path_identity_consistent"]
            )
            row[f"completed_timing_replay_outcomes_internally_consistent_{suffix}"] = (
                consistency["completed_outcomes_internally_consistent"]
            )
            row[f"stage_a_to_completed_timing_replays_consistent_{suffix}"] = (
                consistency["stage_a_to_completed_replays_consistent"]
            )
            row[f"timing_replay_status_instability_{suffix}"] = status_instability
            row[f"stage_a_stage_b_completion_status_instability_{suffix}"] = bool(
                consistency["stage_a_stage_b_completion_status_instability"]
            )
            # Compatibility alias: Stage-B status instability only.
            row[f"timing_replay_instability_{suffix}"] = status_instability
            replay_instability = replay_instability or status_instability
            sources.extend(str(value) for value in path_rows["timing_source"])
        row["scientific_endpoint_source"] = SCIENTIFIC_ENDPOINT_SOURCE
        row["timing_repetitions"] = TIMING_REPETITIONS
        row["timing_order_protocol"] = TIMING_ORDER_PROTOCOL
        row["timing_replay_status_protocol"] = TIMING_REPLAY_STATUS_PROTOCOL
        row["timing_replay_timeout_seconds"] = float(
            state["timing_replay_timeout_seconds"].iloc[0]
        )
        row["timing_replay_instability"] = replay_instability
        row["complete_path_timing_origin"] = COMPLETE_PATH_TIMING_ORIGIN
        row["timing_environment_id"] = str(state["timing_environment_id"].iloc[0])
        row["thread_count"] = int(state["thread_count"].iloc[0])
        row["complete_path_timing_source"] = ";".join(sorted(set(sources)))
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    joined = paired.merge(aggregate, on=key, how="left", validate="one_to_one")
    if joined["runtime_skip_median"].isna().any():
        raise ValueError("complete-path timing coverage does not match paired Selector states")
    return joined


def _completed_timing_endpoints_match(
    completed_replays: pd.DataFrame,
    *,
    terminal_gap: float,
    observed_first_hit_fe: object,
    target_hit_observed: bool,
    endpoint_success: bool,
) -> bool:
    if completed_replays.empty:
        raise ValueError("completed timing endpoint comparison requires at least one replay")
    expected_hit = -1 if pd.isna(observed_first_hit_fe) else int(observed_first_hit_fe)
    return bool(
        np.allclose(
            completed_replays["terminal_gap"].to_numpy(dtype=float),
            float(terminal_gap),
            rtol=0.0,
            atol=EPS,
        )
        and np.array_equal(
            completed_replays["observed_first_hit_FE"].fillna(-1).to_numpy(
                dtype=np.int64
            ),
            np.full(len(completed_replays), expected_hit, dtype=np.int64),
        )
        and np.array_equal(
            completed_replays["target_hit_observed"].to_numpy(dtype=bool),
            np.full(len(completed_replays), bool(target_hit_observed), dtype=bool),
        )
        and np.array_equal(
            completed_replays["endpoint_success"].to_numpy(dtype=bool),
            np.full(len(completed_replays), bool(endpoint_success), dtype=bool),
        )
    )


def generate_utility_labels(
    *,
    query_id: str,
    config_path: Path,
    query_selection_reference_path: Path,
    behavior_selection_reference_path: Path,
    query_adjusted_behavior_reference_path: Path,
    sampling_only_reference_path: Path,
    complete_path_timings_path: Path,
    output_path: Path,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    max_labels: int | None,
    overwrite: bool = False,
) -> dict[str, int | str]:
    spec = get_query_spec(query_id)
    config = load_config(config_path)
    split = split_name(config)
    functions = set(selected_functions(config, only_functions))
    dimensions = set(selected_dimensions(config, only_dimensions))
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"utility label output already exists; pass --overwrite: {output_path}")
    query_reference = pq.read_table(query_selection_reference_path).to_pandas()
    behavior_reference = pq.read_table(behavior_selection_reference_path).to_pandas()
    state_only_reference = pq.read_table(query_adjusted_behavior_reference_path).to_pandas()
    sampling_only_reference = pq.read_table(sampling_only_reference_path).to_pandas()
    complete_path_timings = pq.read_table(complete_path_timings_path).to_pandas()

    def filter_rows(frame: pd.DataFrame) -> pd.DataFrame:
        mask = frame["split"].astype(str).eq(split) & frame["dimension"].astype(int).isin(dimensions)
        mask &= frame["function_id"].astype(str).map(_function_from_id).isin(functions)
        return frame.loc[mask].copy()

    query_reference = filter_rows(query_reference)
    behavior_reference = filter_rows(behavior_reference)
    state_only_reference = filter_rows(state_only_reference)
    sampling_only_reference = filter_rows(sampling_only_reference)
    complete_path_timings = filter_rows(complete_path_timings)
    if query_reference.empty:
        raise ValueError(f"selection reference contains no rows for {query_id} split={split}")
    if set(query_reference["query_id"].astype(str)) != {query_id}:
        raise ValueError("query Selection Reference uses the wrong query_id")
    if set(query_reference["query_protocol"].astype(str)) != {spec.protocol}:
        raise ValueError("query Selection Reference uses the wrong query protocol")
    labels = paired_utility_label_view(
        query_selection=query_reference,
        behavior_selection=behavior_reference,
        query_adjusted_behavior_selection=state_only_reference,
        sampling_only_selection=sampling_only_reference,
        complete_path_timings=complete_path_timings,
        query_id=query_id,
        log10_gap_floor=float(config["log10_gap_floor"]),
        log10_gap_cap=float(config["log10_gap_cap"]),
    )
    if max_labels is not None:
        labels = labels.iloc[: int(max_labels)].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(labels[list(utility_schema().names)], schema=utility_schema(), preserve_index=False),
        output_path,
    )
    print(f"wrote {len(labels)} paired {query_id} utility label rows to {output_path}")
    return {"query_id": query_id, "rows": len(labels), "output": str(output_path)}


def utility_schema() -> pa.Schema:
    from utility_labels.fields import UTILITY_COLUMNS

    sampling_fields = {
        name: (data_type, nullable)
        for name, data_type, nullable in SAMPLING_METADATA_SCHEMA_FIELDS
    }
    string_columns = {
        "split",
        "problem_id",
        "function_id",
        "family",
        "prefix_algorithm",
        "query_id",
        "query_protocol",
        "query_preprocessing_id",
        "query_feature_columns",
        "sample_design_id",
        "default_algorithm",
        "no_query_algorithm",
        "selection_reference_default_algorithm",
        "selection_reference_protocol",
        "selector_prediction_source",
        "selector_target_transform",
        "selected_algorithm",
        "selected_action",
        "no_query_transition_mode",
        "query_transition_mode",
        "handoff_type",
        "behavior_selected_algorithm",
        "behavior_selected_action",
        "behavior_handoff_type",
        "query_matched_state_only_selected_algorithm",
        "query_matched_state_only_selected_action",
        "query_matched_state_only_handoff_type",
        "sampling_only_selected_algorithm",
        "sampling_only_selected_action",
        "sampling_only_handoff_type",
        "performance_value_mode",
        "performance_loss_mode",
        "best_observed_algorithm",
        "utility_formula_protocol",
        "scientific_endpoint_source",
        "scientific_path_status_protocol",
        "scientific_skip_status",
        "scientific_query_joint_status",
        "scientific_behavior_only_full_budget_status",
        "scientific_query_matched_state_only_status",
        "scientific_sampling_only_continue_current_status",
        "timing_order_protocol",
        "timing_replay_status_protocol",
        "complete_path_timing_source",
        "complete_path_timing_origin",
        "timing_environment_id",
        "peak_memory_measurement_status",
    }
    boolean_columns = {
        "selected_equals_default",
        "selected_equals_prefix",
        "handoff_required",
        "skip_switches_from_prefix",
        "selected_matches_best_observed",
        "query_sample_improved_terminal",
        "behavior_selected_equals_default",
        "behavior_selected_equals_prefix",
        "behavior_handoff_required",
        "query_matched_state_only_selected_equals_default",
        "query_matched_state_only_selected_equals_prefix",
        "query_matched_state_only_handoff_required",
        "sampling_only_selected_equals_default",
        "sampling_only_selected_equals_prefix",
        "sampling_only_handoff_required",
        "skip_target_hit_observed",
        "skip_target_hit_before_failure",
        "skip_endpoint_success",
        "skip_success",
        "query_path_target_hit_observed",
        "query_path_target_hit_before_failure",
        "query_path_endpoint_success",
        "query_path_success",
        "behavior_path_target_hit_observed",
        "behavior_path_target_hit_before_failure",
        "behavior_path_endpoint_success",
        "behavior_path_success",
        *{
            f"{prefix}_{suffix}"
            for prefix in ("query_matched_state_only_path", "sampling_only_path")
            for suffix in (
                "target_hit_observed",
                "target_hit_before_failure",
                "endpoint_success",
                "success",
            )
        },
        "scientific_skip_timed_out",
        "scientific_skip_path_completed",
        "scientific_query_joint_timed_out",
        "scientific_query_joint_path_completed",
        "scientific_behavior_only_full_budget_timed_out",
        "scientific_behavior_only_full_budget_path_completed",
        "scientific_query_matched_state_only_timed_out",
        "scientific_query_matched_state_only_path_completed",
        "scientific_sampling_only_continue_current_timed_out",
        "scientific_sampling_only_continue_current_path_completed",
        *{
            f"{prefix}_{path}"
            for prefix in (
                "timing_replay_instability",
                "timing_replay_path_identity_consistent",
                "completed_timing_replay_outcomes_internally_consistent",
                "stage_a_to_completed_timing_replays_consistent",
                "timing_replay_status_instability",
                "stage_a_stage_b_completion_status_instability",
            )
            for path in (
                "skip",
                "query_joint",
                "behavior_only_full_budget",
                *MATCHED_ACQUISITION_PATH_SUFFIXES,
            )
        },
        "timing_replay_instability",
        *NEED_QUERY_COLUMNS,
        *NEED_BEHAVIOR_ONLY_COLUMNS,
    }
    integer_columns = {
        "dimension",
        "seed",
        "FE",
        "FE_prefix",
        "FE_query",
        "FE_no_query_optimization",
        "FE_action_optimization",
        "FE_total",
        "timing_repetitions",
        "thread_count",
        "skip_observed_first_hit_FE",
        "skip_first_hit_FE",
        "skip_planned_FE",
        "skip_effective_FE",
        "query_path_observed_first_hit_FE",
        "query_path_first_hit_FE",
        "query_path_planned_FE",
        "query_path_effective_FE",
        "behavior_path_observed_first_hit_FE",
        "behavior_path_first_hit_FE",
        "behavior_path_planned_FE",
        "behavior_path_effective_FE",
        *{
            f"{prefix}_{suffix}"
            for prefix in ("query_matched_state_only_path", "sampling_only_path")
            for suffix in (
                "observed_first_hit_FE",
                "first_hit_FE",
                "planned_FE",
                "effective_FE",
            )
        },
        *{
            f"timing_replay_{status}_repetitions_{path}"
            for status in ("completed", "timeout", "failure")
            for path in (
                "skip",
                "query_joint",
                "behavior_only_full_budget",
                *MATCHED_ACQUISITION_PATH_SUFFIXES,
            )
        },
    }
    list_float_columns = {
        *{
            f"runtime_{path}_{suffix}"
            for path in (
                "skip",
                "query_joint",
                "behavior_only_full_budget",
                *MATCHED_ACQUISITION_PATH_SUFFIXES,
            )
            for suffix in (
                "repetitions",
                "censored_repetitions",
                "failure_worst_case_repetitions",
            )
        },
    }
    list_int32_columns = {
        *{
            f"timing_order_positions_{path}"
            for path in (
                "skip",
                "query_joint",
                "behavior_only_full_budget",
                *MATCHED_ACQUISITION_PATH_SUFFIXES,
            )
        },
    }
    list_int64_columns = {
        *{
            f"timing_replay_{quantity}_repetitions_{path}"
            for quantity in ("effective_FE", "observed_first_hit_FE")
            for path in (
                "skip",
                "query_joint",
                "behavior_only_full_budget",
                *MATCHED_ACQUISITION_PATH_SUFFIXES,
            )
        },
    }
    list_string_columns = {
        *{
            f"timing_replay_status_repetitions_{path}"
            for path in (
                "skip",
                "query_joint",
                "behavior_only_full_budget",
                *MATCHED_ACQUISITION_PATH_SUFFIXES,
            )
        },
    }
    list_boolean_columns = {
        *{
            f"timing_replay_{quantity}_{path}"
            for quantity in (
                "timed_out_flags",
                "path_completed_flags",
                "target_hit_observed_flags",
                "target_hit_before_failure_flags",
                "endpoint_success_flags",
            )
            for path in (
                "skip",
                "query_joint",
                "behavior_only_full_budget",
                *MATCHED_ACQUISITION_PATH_SUFFIXES,
            )
        },
    }
    fields: list[pa.Field] = []
    for column in UTILITY_COLUMNS:
        if column in sampling_fields:
            data_type, nullable = sampling_fields[column]
            fields.append(pa.field(column, data_type, nullable=nullable))
        elif column in string_columns:
            fields.append(pa.field(column, pa.string()))
        elif column in boolean_columns:
            fields.append(pa.field(column, pa.bool_()))
        elif column in integer_columns:
            fields.append(pa.field(column, pa.int64()))
        elif column in list_float_columns:
            fields.append(pa.field(column, pa.list_(pa.float64())))
        elif column in list_int32_columns:
            fields.append(pa.field(column, pa.list_(pa.int32())))
        elif column in list_int64_columns:
            fields.append(pa.field(column, pa.list_(pa.int64())))
        elif column in list_string_columns:
            fields.append(pa.field(column, pa.list_(pa.string())))
        elif column in list_boolean_columns:
            fields.append(pa.field(column, pa.list_(pa.bool_())))
        else:
            fields.append(pa.field(column, pa.float64()))
    return pa.schema(fields)


def _function_from_id(function_id: str) -> int:
    import re

    match = re.search(r"(\d+)$", function_id)
    if not match:
        raise ValueError(f"cannot infer function number from function_id: {function_id}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paired Query-joint, Behavior-only, and query-operational Utility labels."
    )
    parser.add_argument("--query-id", choices=sorted(LANDSCAPE_QUERY_SPECS), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--query-selection-reference", type=Path, required=True)
    parser.add_argument("--behavior-selection-reference", type=Path, required=True)
    parser.add_argument("--query-adjusted-behavior-reference", type=Path, required=True)
    parser.add_argument("--sampling-only-reference", type=Path, required=True)
    parser.add_argument("--complete-path-timings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    parser.add_argument("--max-labels", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate_utility_labels(
        query_id=args.query_id,
        config_path=args.config,
        query_selection_reference_path=args.query_selection_reference,
        behavior_selection_reference_path=args.behavior_selection_reference,
        query_adjusted_behavior_reference_path=args.query_adjusted_behavior_reference,
        sampling_only_reference_path=args.sampling_only_reference,
        complete_path_timings_path=args.complete_path_timings,
        output_path=args.output,
        only_functions=args.only_function,
        only_dimensions=args.only_dimension,
        max_labels=args.max_labels,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
