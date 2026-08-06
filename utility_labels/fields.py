from __future__ import annotations

from behavior.features import BEHAVIOR_FEATURE_COLUMNS


UTILITY_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0)

UTILITY_COLUMNS = (
    "split",
    "problem_id",
    "family",
    "dimension",
    "prefix_algorithm",
    "seed",
    "FE",
    "FE_ratio",
    "FE_total",
    "FE_prefix",
    "FE_analysis",
    "FE_skip_optimization",
    "FE_ela_optimization",
    "default_algorithm",
    "selected_algorithm",
    "p_skip",
    "p_ela",
    "performance_gain_raw",
    "performance_gain_norm",
    "runtime_analysis",
    "runtime_selection",
    "runtime_skip_optimization",
    "runtime_ela_optimization",
    "time_cost_norm",
    "memory_cost_norm",
    "u_ela_lamT_0",
    "u_ela_lamT_025",
    "u_ela_lamT_05",
    "u_ela_lamT_1",
    "u_ela_lamT_2",
    "need_ela_lamT_0",
    "need_ela_lamT_025",
    "need_ela_lamT_05",
    "need_ela_lamT_1",
    "need_ela_lamT_2",
) + BEHAVIOR_FEATURE_COLUMNS


UTILITY_VALUE_COLUMNS = (
    "u_ela_lamT_0",
    "u_ela_lamT_025",
    "u_ela_lamT_05",
    "u_ela_lamT_1",
    "u_ela_lamT_2",
)


NEED_ELA_COLUMNS = (
    "need_ela_lamT_0",
    "need_ela_lamT_025",
    "need_ela_lamT_05",
    "need_ela_lamT_1",
    "need_ela_lamT_2",
)
