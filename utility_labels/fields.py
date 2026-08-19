from __future__ import annotations

from trajectory.sampling import SAMPLING_METADATA_COLUMNS

SCIENTIFIC_ENDPOINT_SOURCE = "stage_a_selection_reference_outcome"
SCIENTIFIC_PATH_STATUS_PROTOCOL = "stage_a_completed_timed_out_failed_v1"
TIMING_REPLAY_STATUS_PROTOCOL = "stage_b_completed_timed_out_failed_v1"
TIMING_REPLAY_STATUSES = frozenset({"completed", "timed_out", "failed"})

EFFICACY_FORMULA_PROTOCOL = "equal_total_fe_log_gap_ratio_v1"
EFFICACY_COLUMNS = (
    "g_fe",
    "g_fe_bounded",
    "g_fe_gt_zero",
    "g_fe_gt_practical",
    "epsilon_p",
    "delta_practical",
)

UTILITY_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0)

# ── 方案 A 主功效标签（等总 FE，runtime 不进入主标签）──────────
# g_fe 是连续主标签，g_fe_gt_zero 是对应布尔标签。
# 旧 u_query_joint_lamT_* / need_query_joint_lamT_* 仅作兼容诊断。
PRIMARY_EFFICACY_VALUE_COLUMN = "g_fe"
PRIMARY_EFFICACY_LABEL_COLUMN = "g_fe_gt_zero"
PRIMARY_EFFICACY_COLUMNS = (
    PRIMARY_EFFICACY_VALUE_COLUMN,
    PRIMARY_EFFICACY_LABEL_COLUMN,
)

# Utility 标签（旧口径保留为兼容字段）
UTILITY_VALUE_COLUMNS = (
    "u_query_joint_lamT_0",
    "u_query_joint_lamT_025",
    "u_query_joint_lamT_05",
    "u_query_joint_lamT_1",
    "u_query_joint_lamT_2",
)
NEED_QUERY_COLUMNS = (
    "need_query_joint_lamT_0",
    "need_query_joint_lamT_025",
    "need_query_joint_lamT_05",
    "need_query_joint_lamT_1",
    "need_query_joint_lamT_2",
)
BEHAVIOR_UTILITY_VALUE_COLUMNS = (
    "u_behavior_only_full_budget_lamT_0",
    "u_behavior_only_full_budget_lamT_025",
    "u_behavior_only_full_budget_lamT_05",
    "u_behavior_only_full_budget_lamT_1",
    "u_behavior_only_full_budget_lamT_2",
)
NEED_BEHAVIOR_ONLY_COLUMNS = (
    "need_behavior_only_full_budget_lamT_0",
    "need_behavior_only_full_budget_lamT_025",
    "need_behavior_only_full_budget_lamT_05",
    "need_behavior_only_full_budget_lamT_1",
    "need_behavior_only_full_budget_lamT_2",
)
QUERY_OPERATIONAL_INCREMENT_COLUMNS = (
    "query_operational_increment_lamT_0",
    "query_operational_increment_lamT_025",
    "query_operational_increment_lamT_05",
    "query_operational_increment_lamT_1",
    "query_operational_increment_lamT_2",
)
QUERY_MATCHED_STATE_ONLY_UTILITY_COLUMNS = (
    "u_query_matched_state_only_lamT_0",
    "u_query_matched_state_only_lamT_025",
    "u_query_matched_state_only_lamT_05",
    "u_query_matched_state_only_lamT_1",
    "u_query_matched_state_only_lamT_2",
)
SAMPLING_ONLY_UTILITY_COLUMNS = (
    "u_sampling_only_continue_current_lamT_0",
    "u_sampling_only_continue_current_lamT_025",
    "u_sampling_only_continue_current_lamT_05",
    "u_sampling_only_continue_current_lamT_1",
    "u_sampling_only_continue_current_lamT_2",
)
QUERY_DESCRIPTOR_USE_INCREMENT_COLUMNS = (
    "query_descriptor_use_increment_lamT_0",
    "query_descriptor_use_increment_lamT_025",
    "query_descriptor_use_increment_lamT_05",
    "query_descriptor_use_increment_lamT_1",
    "query_descriptor_use_increment_lamT_2",
)
QUERY_STATE_ONLY_VS_SAMPLING_INCREMENT_COLUMNS = (
    "query_state_only_vs_sampling_increment_lamT_0",
    "query_state_only_vs_sampling_increment_lamT_025",
    "query_state_only_vs_sampling_increment_lamT_05",
    "query_state_only_vs_sampling_increment_lamT_1",
    "query_state_only_vs_sampling_increment_lamT_2",
)
QUERY_SAMPLING_DIRECT_INCREMENT_COLUMNS = (
    "query_sampling_direct_increment_lamT_0",
    "query_sampling_direct_increment_lamT_025",
    "query_sampling_direct_increment_lamT_05",
    "query_sampling_direct_increment_lamT_1",
    "query_sampling_direct_increment_lamT_2",
)
MATCHED_ACQUISITION_PATH_SUFFIXES = (
    "query_matched_state_only",
    "sampling_only_continue_current",
)
RUNTIME_COST_COLUMNS = (
    "runtime_query_sampling",
    "runtime_query_evaluation",
    "runtime_query_feature_computation",
    "runtime_query",
    "runtime_selection",
    "runtime_handoff",
    "runtime_no_query_handoff",
    "runtime_no_query_optimization",
    "runtime_query_optimization",
    "runtime_query_total",
    "runtime_no_query_total",
    "runtime_net",
    "time_cost_log10_ratio",
    "behavior_time_cost_log10_ratio",
    "query_vs_behavior_time_cost_log10_ratio",
)
UTILITY_COLUMNS = (
    "split",
    "problem_id",
    "function_id",
    "family",
    "dimension",
    "prefix_algorithm",
    "seed",
    "FE",
    "FE_ratio",
    *SAMPLING_METADATA_COLUMNS,
    "query_id",
    "query_protocol",
    "query_preprocessing_id",
    "query_feature_columns",
    "sample_design_id",
    "FE_total",
    "FE_prefix",
    "FE_query",
    "FE_no_query_optimization",
    "FE_action_optimization",
    "default_algorithm",
    "no_query_algorithm",
    "selection_reference_default_algorithm",
    "selection_reference_protocol",
    "selector_prediction_source",
    "selector_target_transform",
    "selected_algorithm",
    "selected_action",
    "selected_equals_default",
    "selected_equals_prefix",
    "handoff_required",
    "skip_switches_from_prefix",
    "no_query_transition_mode",
    "query_transition_mode",
    "handoff_type",
    "behavior_selected_algorithm",
    "behavior_selected_action",
    "behavior_selected_equals_default",
    "behavior_selected_equals_prefix",
    "behavior_handoff_required",
    "behavior_handoff_type",
    "query_matched_state_only_selected_algorithm",
    "query_matched_state_only_selected_action",
    "query_matched_state_only_selected_equals_default",
    "query_matched_state_only_selected_equals_prefix",
    "query_matched_state_only_handoff_required",
    "query_matched_state_only_handoff_type",
    "sampling_only_selected_algorithm",
    "sampling_only_selected_action",
    "sampling_only_selected_equals_default",
    "sampling_only_selected_equals_prefix",
    "sampling_only_handoff_required",
    "sampling_only_handoff_type",
    "performance_value_mode",
    "performance_loss_mode",
    "benchmark_reference_value",
    "p_skip",
    "p_query",
    "behavior_p_behavior",
    "p_query_matched_state_only",
    "p_sampling_only_continue_current",
    "p_skip_raw",
    "p_query_raw",
    "loss_skip",
    "loss_query",
    "action_loss",
    "best_observed_algorithm",
    "best_observed_loss",
    "selected_matches_best_observed",
    "potential_gain_raw",
    "selector_regret_raw",
    "performance_gain_raw",
    "performance_gain_norm",
    "performance_gain_gap_raw",
    "performance_gain_norm_gap",
    "utility_formula_protocol",
    "log10_gap_floor",
    "log10_gap_cap",
    "log10_gap_skip",
    "log10_gap_query_joint",
    "log10_gap_behavior_only_full_budget",
    "log10_gap_query_matched_state_only",
    "log10_gap_sampling_only_continue_current",
    "query_joint_log10_gap_improvement",
    "behavior_only_log10_gap_improvement",
    "query_vs_behavior_log10_gap_improvement",
    "continuation_only_gap",
    "query_sample_best_gap",
    "query_sample_improved_terminal",
    "query_sample_terminal_gap_improvement",
    "query_feature_predictive_increment_log10_gap",
    "query_descriptor_use_log10_gap_increment",
    "query_sampling_direct_log10_gap_increment",
    "success_gap_target",
    "skip_observed_first_hit_FE",
    "skip_target_hit_observed",
    "skip_target_hit_before_failure",
    "skip_endpoint_success",
    "skip_first_hit_FE",
    "skip_success",
    "skip_planned_FE",
    "skip_effective_FE",
    "query_path_observed_first_hit_FE",
    "query_path_target_hit_observed",
    "query_path_target_hit_before_failure",
    "query_path_endpoint_success",
    "query_path_first_hit_FE",
    "query_path_success",
    "query_path_planned_FE",
    "query_path_effective_FE",
    "behavior_path_observed_first_hit_FE",
    "behavior_path_target_hit_observed",
    "behavior_path_target_hit_before_failure",
    "behavior_path_endpoint_success",
    "behavior_path_first_hit_FE",
    "behavior_path_success",
    "behavior_path_planned_FE",
    "behavior_path_effective_FE",
    "query_matched_state_only_path_observed_first_hit_FE",
    "query_matched_state_only_path_target_hit_observed",
    "query_matched_state_only_path_target_hit_before_failure",
    "query_matched_state_only_path_endpoint_success",
    "query_matched_state_only_path_first_hit_FE",
    "query_matched_state_only_path_success",
    "query_matched_state_only_path_planned_FE",
    "query_matched_state_only_path_effective_FE",
    "sampling_only_path_observed_first_hit_FE",
    "sampling_only_path_target_hit_observed",
    "sampling_only_path_target_hit_before_failure",
    "sampling_only_path_endpoint_success",
    "sampling_only_path_first_hit_FE",
    "sampling_only_path_success",
    "sampling_only_path_planned_FE",
    "sampling_only_path_effective_FE",
    "scientific_endpoint_source",
    "scientific_path_status_protocol",
    "scientific_skip_status",
    "scientific_skip_timed_out",
    "scientific_skip_path_completed",
    "scientific_query_joint_status",
    "scientific_query_joint_timed_out",
    "scientific_query_joint_path_completed",
    "scientific_behavior_only_full_budget_status",
    "scientific_behavior_only_full_budget_timed_out",
    "scientific_behavior_only_full_budget_path_completed",
    "scientific_query_matched_state_only_status",
    "scientific_query_matched_state_only_timed_out",
    "scientific_query_matched_state_only_path_completed",
    "scientific_sampling_only_continue_current_status",
    "scientific_sampling_only_continue_current_timed_out",
    "scientific_sampling_only_continue_current_path_completed",
    *RUNTIME_COST_COLUMNS,
    "runtime_skip_median",
    "runtime_query_joint_median",
    "runtime_behavior_only_full_budget_median",
    *tuple(f"runtime_{suffix}_median" for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES),
    "runtime_skip_raw_observed_median",
    "runtime_query_joint_raw_observed_median",
    "runtime_behavior_only_full_budget_raw_observed_median",
    *tuple(
        f"runtime_{suffix}_raw_observed_median"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "runtime_skip_failure_worst_case_median",
    "runtime_query_joint_failure_worst_case_median",
    "runtime_behavior_only_full_budget_failure_worst_case_median",
    *tuple(
        f"runtime_{suffix}_failure_worst_case_median"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "runtime_skip_repetitions",
    "runtime_query_joint_repetitions",
    "runtime_behavior_only_full_budget_repetitions",
    *tuple(f"runtime_{suffix}_repetitions" for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES),
    "runtime_skip_censored_repetitions",
    "runtime_query_joint_censored_repetitions",
    "runtime_behavior_only_full_budget_censored_repetitions",
    *tuple(
        f"runtime_{suffix}_censored_repetitions"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "runtime_skip_failure_worst_case_repetitions",
    "runtime_query_joint_failure_worst_case_repetitions",
    "runtime_behavior_only_full_budget_failure_worst_case_repetitions",
    *tuple(
        f"runtime_{suffix}_failure_worst_case_repetitions"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "timing_order_positions_skip",
    "timing_order_positions_query_joint",
    "timing_order_positions_behavior_only_full_budget",
    *tuple(
        f"timing_order_positions_{suffix}" for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "timing_replay_status_repetitions_skip",
    "timing_replay_status_repetitions_query_joint",
    "timing_replay_status_repetitions_behavior_only_full_budget",
    *tuple(
        f"timing_replay_status_repetitions_{suffix}"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "timing_replay_effective_FE_repetitions_skip",
    "timing_replay_effective_FE_repetitions_query_joint",
    "timing_replay_effective_FE_repetitions_behavior_only_full_budget",
    *tuple(
        f"timing_replay_effective_FE_repetitions_{suffix}"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "timing_replay_observed_first_hit_FE_repetitions_skip",
    "timing_replay_observed_first_hit_FE_repetitions_query_joint",
    "timing_replay_observed_first_hit_FE_repetitions_behavior_only_full_budget",
    *tuple(
        f"timing_replay_observed_first_hit_FE_repetitions_{suffix}"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "timing_replay_target_hit_observed_flags_skip",
    "timing_replay_target_hit_observed_flags_query_joint",
    "timing_replay_target_hit_observed_flags_behavior_only_full_budget",
    *tuple(
        f"timing_replay_target_hit_observed_flags_{suffix}"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "timing_replay_target_hit_before_failure_flags_skip",
    "timing_replay_target_hit_before_failure_flags_query_joint",
    "timing_replay_target_hit_before_failure_flags_behavior_only_full_budget",
    *tuple(
        f"timing_replay_target_hit_before_failure_flags_{suffix}"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "timing_replay_endpoint_success_flags_skip",
    "timing_replay_endpoint_success_flags_query_joint",
    "timing_replay_endpoint_success_flags_behavior_only_full_budget",
    *tuple(
        f"timing_replay_endpoint_success_flags_{suffix}"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "timing_replay_timed_out_flags_skip",
    "timing_replay_timed_out_flags_query_joint",
    "timing_replay_timed_out_flags_behavior_only_full_budget",
    *tuple(
        f"timing_replay_timed_out_flags_{suffix}"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "timing_replay_path_completed_flags_skip",
    "timing_replay_path_completed_flags_query_joint",
    "timing_replay_path_completed_flags_behavior_only_full_budget",
    *tuple(
        f"timing_replay_path_completed_flags_{suffix}"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "timing_replay_completed_repetitions_skip",
    "timing_replay_completed_repetitions_query_joint",
    "timing_replay_completed_repetitions_behavior_only_full_budget",
    *tuple(
        f"timing_replay_completed_repetitions_{suffix}"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "timing_replay_timeout_repetitions_skip",
    "timing_replay_timeout_repetitions_query_joint",
    "timing_replay_timeout_repetitions_behavior_only_full_budget",
    *tuple(
        f"timing_replay_timeout_repetitions_{suffix}"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "timing_replay_failure_repetitions_skip",
    "timing_replay_failure_repetitions_query_joint",
    "timing_replay_failure_repetitions_behavior_only_full_budget",
    *tuple(
        f"timing_replay_failure_repetitions_{suffix}"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "timing_replay_instability_skip",
    "timing_replay_instability_query_joint",
    "timing_replay_instability_behavior_only_full_budget",
    *tuple(
        f"timing_replay_instability_{suffix}"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "timing_replay_instability",
    "timing_replay_path_identity_consistent_skip",
    "timing_replay_path_identity_consistent_query_joint",
    "timing_replay_path_identity_consistent_behavior_only_full_budget",
    *tuple(
        f"timing_replay_path_identity_consistent_{suffix}"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "completed_timing_replay_outcomes_internally_consistent_skip",
    "completed_timing_replay_outcomes_internally_consistent_query_joint",
    "completed_timing_replay_outcomes_internally_consistent_behavior_only_full_budget",
    *tuple(
        f"completed_timing_replay_outcomes_internally_consistent_{suffix}"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "stage_a_to_completed_timing_replays_consistent_skip",
    "stage_a_to_completed_timing_replays_consistent_query_joint",
    "stage_a_to_completed_timing_replays_consistent_behavior_only_full_budget",
    *tuple(
        f"stage_a_to_completed_timing_replays_consistent_{suffix}"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "timing_replay_status_instability_skip",
    "timing_replay_status_instability_query_joint",
    "timing_replay_status_instability_behavior_only_full_budget",
    *tuple(
        f"timing_replay_status_instability_{suffix}"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "stage_a_stage_b_completion_status_instability_skip",
    "stage_a_stage_b_completion_status_instability_query_joint",
    "stage_a_stage_b_completion_status_instability_behavior_only_full_budget",
    *tuple(
        f"stage_a_stage_b_completion_status_instability_{suffix}"
        for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES
    ),
    "timing_repetitions",
    "timing_order_protocol",
    "timing_replay_status_protocol",
    "timing_replay_timeout_seconds",
    "timing_environment_id",
    "thread_count",
    "complete_path_timing_source",
    "complete_path_timing_origin",
    "peak_memory_measurement_status",
    "peak_memory_bytes_skip",
    "peak_memory_bytes_query_joint",
    "peak_memory_bytes_behavior_only_full_budget",
    *tuple(f"peak_memory_bytes_{suffix}" for suffix in MATCHED_ACQUISITION_PATH_SUFFIXES),
    *UTILITY_VALUE_COLUMNS,
    *BEHAVIOR_UTILITY_VALUE_COLUMNS,
    *QUERY_OPERATIONAL_INCREMENT_COLUMNS,
    *QUERY_MATCHED_STATE_ONLY_UTILITY_COLUMNS,
    *SAMPLING_ONLY_UTILITY_COLUMNS,
    *QUERY_DESCRIPTOR_USE_INCREMENT_COLUMNS,
    *QUERY_STATE_ONLY_VS_SAMPLING_INCREMENT_COLUMNS,
    *QUERY_SAMPLING_DIRECT_INCREMENT_COLUMNS,
    *NEED_QUERY_COLUMNS,
    *NEED_BEHAVIOR_ONLY_COLUMNS,
    *EFFICACY_COLUMNS,
)
