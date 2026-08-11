from __future__ import annotations

import argparse
from math import isclose, isfinite
from pathlib import Path

import pyarrow.parquet as pq

from landscape_queries.specs import get_query_spec
from utility_labels.fields import NEED_QUERY_COLUMNS, UTILITY_LAMBDAS, UTILITY_VALUE_COLUMNS


def validate_utility_label_file(path: str | Path) -> dict[str, int]:
    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError("utility label file contains no rows")
    for row in rows:
        _validate_row(row)
    return {"rows": len(rows)}


def _validate_row(row: dict) -> None:
    spec = get_query_spec(str(row["query_id"]))
    if str(row["query_protocol"]) != spec.protocol:
        raise ValueError("query_protocol does not match query_id")
    if str(row["sample_design_id"]) != spec.sample_design_id:
        raise ValueError("sample_design_id does not match query_id")
    if int(row["FE_prefix"]) + int(row["FE_no_query_optimization"]) != int(row["FE_total"]):
        raise ValueError("no-query FE ledger is inconsistent")
    if (
        int(row["FE_prefix"])
        + int(row["FE_query"])
        + int(row["FE_query_optimization"])
        != int(row["FE_total"])
    ):
        raise ValueError("query FE ledger is inconsistent")
    expected_fe = spec.sample_design.sample_size(int(row["dimension"]))
    if int(row["FE_query"]) != expected_fe:
        raise ValueError("FE_query does not match the frozen sample design")
    expected_selected_equals_default = str(row["selected_algorithm"]) == str(row["default_algorithm"])
    expected_selected_equals_prefix = str(row["selected_algorithm"]) == str(row["prefix_algorithm"])
    expected_skip_switch = str(row["default_algorithm"]) != str(row["prefix_algorithm"])
    if str(row["no_query_algorithm"]) != str(row["default_algorithm"]):
        raise ValueError("no_query_algorithm must equal default_algorithm")
    if bool(row["selected_equals_default"]) != expected_selected_equals_default:
        raise ValueError("selected_equals_default is inconsistent")
    if bool(row["selected_equals_prefix"]) != expected_selected_equals_prefix:
        raise ValueError("selected_equals_prefix is inconsistent")
    if bool(row["skip_switches_from_prefix"]) != expected_skip_switch:
        raise ValueError("skip_switches_from_prefix is inconsistent")
    expected_no_query_mode = (
        "population_transfer_initialization" if expected_skip_switch else "native_optimizer_state"
    )
    expected_query_mode = (
        "native_optimizer_state" if expected_selected_equals_prefix else "population_transfer_initialization"
    )
    if str(row["no_query_transition_mode"]) != expected_no_query_mode:
        raise ValueError("no_query_transition_mode is inconsistent")
    if str(row["query_transition_mode"]) != expected_query_mode:
        raise ValueError("query_transition_mode is inconsistent")
    if str(row["handoff_type"]) != expected_query_mode:
        raise ValueError("handoff_type must equal query_transition_mode")
    if not isclose(float(row["p_query"]), float(row["selected_action_loss"]), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("p_query must equal selected_action_loss")
    p_skip = float(row["p_skip"])
    p_query = float(row["p_query"])
    best = float(row["best_observed_loss"])
    selector_regret = p_query - best
    potential_gain = p_skip - best
    performance_gain = p_skip - p_query
    if selector_regret < -1e-12:
        raise ValueError("selector_regret_raw cannot be negative")
    checks = {
        "selector_regret_raw": selector_regret,
        "potential_gain_raw": potential_gain,
        "performance_gain_raw": performance_gain,
    }
    for column, expected in checks.items():
        if not isclose(float(row[column]), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{column} is inconsistent")
    if not isclose(
        float(row["performance_gain_raw"]),
        float(row["potential_gain_raw"]) - float(row["selector_regret_raw"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("raw utility decomposition is inconsistent")
    scale = max(abs(p_skip), abs(p_query), 1e-12)
    normalized = {
        "performance_norm_scale": scale,
        "performance_gain_norm": performance_gain / scale,
        "potential_gain_norm": potential_gain / scale,
        "selector_regret_decomposition_norm": selector_regret / scale,
    }
    for column, expected in normalized.items():
        if not isclose(float(row[column]), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{column} is inconsistent")
    runtime_columns = (
        "runtime_query",
        "runtime_selection",
        "runtime_no_query_optimization",
        "runtime_query_optimization",
    )
    if any(not isfinite(float(row[column])) or float(row[column]) < 0.0 for column in runtime_columns):
        raise ValueError("runtime fields must be finite and non-negative")
    expected_time_cost = (float(row["runtime_query"]) + float(row["runtime_selection"])) / max(
        float(row["runtime_no_query_optimization"]), 1e-12
    )
    if not isclose(float(row["time_cost_norm"]), expected_time_cost, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("time_cost_norm is inconsistent")
    if float(row["memory_cost_norm"]) != 0.0:
        raise ValueError("v1 memory_cost_norm must be 0.0")
    for utility_column, need_column, weight in zip(
        UTILITY_VALUE_COLUMNS,
        NEED_QUERY_COLUMNS,
        UTILITY_LAMBDAS,
        strict=True,
    ):
        expected_utility = performance_gain / scale - weight * expected_time_cost
        if not isclose(float(row[utility_column]), expected_utility, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{utility_column} is inconsistent")
        if bool(row[need_column]) != (expected_utility > 0.0):
            raise ValueError(f"{need_column} must equal {utility_column} > 0")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate query-specific offline utility labels.")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    summary = validate_utility_label_file(args.input)
    print(f"validated {summary['rows']} query utility label rows")


if __name__ == "__main__":
    main()
