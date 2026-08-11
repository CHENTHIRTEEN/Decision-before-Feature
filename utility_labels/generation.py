from __future__ import annotations

import argparse
from math import isfinite
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.phase1_batch_common import load_config, selected_dimensions, selected_functions, split_name
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS, get_query_spec
from selection_reference.model import SELECTION_REFERENCE_PROTOCOL
from utility_labels.fields import NEED_QUERY_COLUMNS, UTILITY_LAMBDAS, UTILITY_VALUE_COLUMNS


EPS = 1e-12


def generate_utility_labels(
    *,
    query_id: str,
    config_path: Path,
    selection_reference_path: Path,
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
    reference = pq.read_table(selection_reference_path).to_pylist()
    rows = []
    for row in reference:
        if str(row["split"]) != split or int(row["dimension"]) not in dimensions:
            continue
        function = _function_from_family(str(row["family"]))
        if function not in functions:
            continue
        rows.append(_utility_row(row=row, query_id=query_id))
        if max_labels is not None and len(rows) >= max_labels:
            break
    if not rows:
        raise ValueError(f"selection reference contains no rows for {query_id} split={split}")
    if any(str(row["query_protocol"]) != spec.protocol for row in rows):
        raise ValueError("utility labels do not match the frozen query protocol")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=utility_schema()), output_path)
    print(f"wrote {len(rows)} {query_id} utility label rows to {output_path}")
    return {"query_id": query_id, "rows": len(rows), "output": str(output_path)}


def _utility_row(*, row: dict, query_id: str) -> dict:
    spec = get_query_spec(query_id)
    if row.get("selection_reference_protocol") != SELECTION_REFERENCE_PROTOCOL:
        raise ValueError("selection reference uses an unsupported protocol")
    if str(row.get("query_id")) != query_id or str(row.get("query_protocol")) != spec.protocol:
        raise ValueError("selection reference query identity does not match the requested utility target")
    if str(row.get("sample_design_id")) != spec.sample_design_id:
        raise ValueError("selection reference and query spec use different query-FE budgets")

    prefix_algorithm = str(row["prefix_algorithm"])
    default_algorithm = str(row["default_algorithm"])
    no_query_algorithm = str(row["no_query_algorithm"])
    if no_query_algorithm != default_algorithm:
        raise ValueError("no_query_algorithm must equal default_algorithm")
    selected_algorithm = str(row["selected_algorithm"])
    selected_equals_default = selected_algorithm == default_algorithm
    selected_equals_prefix = selected_algorithm == prefix_algorithm
    skip_switches_from_prefix = default_algorithm != prefix_algorithm
    expected_action = "continue_current" if selected_equals_prefix else selected_algorithm
    if str(row["selected_action"]) != expected_action:
        raise ValueError("selection reference selected_action is inconsistent")
    no_query_transition_mode = (
        "population_transfer_initialization" if skip_switches_from_prefix else "native_optimizer_state"
    )
    query_transition_mode = (
        "native_optimizer_state" if selected_equals_prefix else "population_transfer_initialization"
    )
    if str(row["no_query_transition_mode"]) != no_query_transition_mode:
        raise ValueError("no-query transition mode is inconsistent")
    if str(row["selected_transition_mode"]) != query_transition_mode:
        raise ValueError("query transition mode is inconsistent")
    handoff_type = str(row["handoff_type"])
    if handoff_type != query_transition_mode:
        raise ValueError("handoff_type must equal the query transition mode")

    p_skip = float(row["p_skip"])
    p_query = float(row["selected_action_loss"])
    best_observed_loss = float(row["best_observed_loss"])
    potential_gain_raw = p_skip - best_observed_loss
    selector_regret_raw = p_query - best_observed_loss
    performance_gain_raw = p_skip - p_query
    if not np.isclose(performance_gain_raw, potential_gain_raw - selector_regret_raw, rtol=0.0, atol=EPS):
        raise ValueError("performance gain decomposition is inconsistent")
    scale = max(abs(p_skip), abs(p_query), EPS)
    performance_gain_norm = performance_gain_raw / scale
    potential_gain_norm = potential_gain_raw / scale
    selector_regret_norm = selector_regret_raw / scale
    runtime_query = float(row["runtime_query"])
    runtime_selection = float(row["runtime_selection"])
    runtime_skip = float(row["runtime_no_query_optimization"])
    runtime_selected = float(row["runtime_selected_action_optimization"])
    runtimes = (runtime_query, runtime_selection, runtime_skip, runtime_selected)
    if any(not isfinite(value) or value < 0.0 for value in runtimes):
        raise ValueError("query and optimization runtimes must be finite and non-negative")
    time_cost_norm = (runtime_query + runtime_selection) / max(runtime_skip, EPS)
    utility_values = {
        column: performance_gain_norm - weight * time_cost_norm
        for column, weight in zip(UTILITY_VALUE_COLUMNS, UTILITY_LAMBDAS, strict=True)
    }
    need_values = {
        column: bool(utility_values[utility_column] > 0.0)
        for column, utility_column in zip(NEED_QUERY_COLUMNS, UTILITY_VALUE_COLUMNS, strict=True)
    }
    return {
        "split": str(row["split"]),
        "problem_id": str(row["problem_id"]),
        "family": str(row["family"]),
        "dimension": int(row["dimension"]),
        "prefix_algorithm": prefix_algorithm,
        "seed": int(row["seed"]),
        "FE": int(row["FE"]),
        "FE_ratio": float(row["FE_ratio"]),
        "query_id": query_id,
        "query_protocol": spec.protocol,
        "query_feature_columns": str(row["query_feature_columns"]),
        "sample_design_id": spec.sample_design_id,
        "FE_total": int(row["FE_total"]),
        "FE_prefix": int(row["FE"]),
        "FE_query": int(row["FE_query"]),
        "FE_no_query_optimization": int(row["FE_no_query_optimization"]),
        "FE_query_optimization": int(row["FE_query_optimization"]),
        "default_algorithm": default_algorithm,
        "no_query_algorithm": no_query_algorithm,
        "selection_reference_default_algorithm": default_algorithm,
        "selection_reference_protocol": str(row["selection_reference_protocol"]),
        "selector_prediction_source": str(row["selector_prediction_source"]),
        "selected_algorithm": selected_algorithm,
        "selected_action": str(row["selected_action"]),
        "selected_equals_default": selected_equals_default,
        "selected_equals_prefix": selected_equals_prefix,
        "skip_switches_from_prefix": skip_switches_from_prefix,
        "no_query_transition_mode": no_query_transition_mode,
        "query_transition_mode": query_transition_mode,
        "handoff_type": handoff_type,
        "p_skip": p_skip,
        "p_query": p_query,
        "selected_action_loss": p_query,
        "best_observed_algorithm": str(row["best_observed_algorithm"]),
        "best_observed_loss": best_observed_loss,
        "selected_matches_best_observed": bool(selected_algorithm == str(row["best_observed_algorithm"])),
        "potential_gain_raw": float(potential_gain_raw),
        "selector_regret_raw": float(selector_regret_raw),
        "performance_norm_scale": float(scale),
        "potential_gain_norm": float(potential_gain_norm),
        "selector_regret_decomposition_norm": float(selector_regret_norm),
        "performance_gain_raw": float(performance_gain_raw),
        "performance_gain_norm": float(performance_gain_norm),
        "runtime_query": runtime_query,
        "runtime_selection": runtime_selection,
        "runtime_no_query_optimization": runtime_skip,
        "runtime_query_optimization": runtime_selected,
        "time_cost_norm": float(time_cost_norm),
        "memory_cost_norm": 0.0,
        **utility_values,
        **need_values,
    }


def _function_from_family(family: str) -> int:
    try:
        return int(family.rsplit("f", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cannot parse function number from family: {family}") from exc


def utility_schema() -> pa.Schema:
    fields = [
        ("split", pa.string()),
        ("problem_id", pa.string()),
        ("family", pa.string()),
        ("dimension", pa.int32()),
        ("prefix_algorithm", pa.string()),
        ("seed", pa.int64()),
        ("FE", pa.int64()),
        ("FE_ratio", pa.float64()),
        ("query_id", pa.string()),
        ("query_protocol", pa.string()),
        ("query_feature_columns", pa.string()),
        ("sample_design_id", pa.string()),
        ("FE_total", pa.int64()),
        ("FE_prefix", pa.int64()),
        ("FE_query", pa.int64()),
        ("FE_no_query_optimization", pa.int64()),
        ("FE_query_optimization", pa.int64()),
        ("default_algorithm", pa.string()),
        ("no_query_algorithm", pa.string()),
        ("selection_reference_default_algorithm", pa.string()),
        ("selection_reference_protocol", pa.string()),
        ("selector_prediction_source", pa.string()),
        ("selected_algorithm", pa.string()),
        ("selected_action", pa.string()),
        ("selected_equals_default", pa.bool_()),
        ("selected_equals_prefix", pa.bool_()),
        ("skip_switches_from_prefix", pa.bool_()),
        ("no_query_transition_mode", pa.string()),
        ("query_transition_mode", pa.string()),
        ("handoff_type", pa.string()),
        ("p_skip", pa.float64()),
        ("p_query", pa.float64()),
        ("selected_action_loss", pa.float64()),
        ("best_observed_algorithm", pa.string()),
        ("best_observed_loss", pa.float64()),
        ("selected_matches_best_observed", pa.bool_()),
        ("potential_gain_raw", pa.float64()),
        ("selector_regret_raw", pa.float64()),
        ("performance_norm_scale", pa.float64()),
        ("potential_gain_norm", pa.float64()),
        ("selector_regret_decomposition_norm", pa.float64()),
        ("performance_gain_raw", pa.float64()),
        ("performance_gain_norm", pa.float64()),
        ("runtime_query", pa.float64()),
        ("runtime_selection", pa.float64()),
        ("runtime_no_query_optimization", pa.float64()),
        ("runtime_query_optimization", pa.float64()),
        ("time_cost_norm", pa.float64()),
        ("memory_cost_norm", pa.float64()),
    ]
    fields.extend((column, pa.float64()) for column in UTILITY_VALUE_COLUMNS)
    fields.extend((column, pa.bool_()) for column in NEED_QUERY_COLUMNS)
    return pa.schema(fields)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate query-specific offline utility labels.")
    parser.add_argument("--query-id", choices=sorted(LANDSCAPE_QUERY_SPECS), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    parser.add_argument("--max-labels", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate_utility_labels(
        query_id=args.query_id,
        config_path=args.config,
        selection_reference_path=args.selection_reference,
        output_path=args.output,
        only_functions=args.only_function,
        only_dimensions=args.only_dimension,
        max_labels=args.max_labels,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
