"""Quantify benchmark-boundary scale effects in the current CMA-ES prefix.

This is a standalone sensitivity analysis.  It does not load or modify the
Decision model, its preprocessing, its threshold, or any utility label.  The
analysis replays the current CMA-ES prefix under the formal BBOB and CEC
configuration, counts out-of-bound proposals immediately before the existing
reflection operator, and records population geometry at the predefined
budget milestones.

The primary comparison is:

* BBOB-train functions, instance 1, dimensions 10/20/40, seeds 1--5;
* the five current CEC2017 functions, instance 1, dimensions 10/20/30/50,
  seeds 1--5.

The BBOB instance restriction matches CEC's single-instance protocol and is
limited to this sensitivity analysis.  It does not replace the formal
Decision training data.

Outputs are written below
``results/dataset_analysis/domain_scale_sensitivity``:

* replay-level reflection and initial-coverage tables;
* milestone-level raw-coordinate and unit-cube distance tables;
* exact-dimension scale decomposition tables;
* four static diagnostic figures and a Chinese Markdown report.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import optimizers.state as optimizer_state_module
from benchmarks import make_problem
from experiments.phase1_batch_common import (
    as_int_list,
    fe_total_for_dimension,
    load_config,
    load_suite_configs,
    runtime_problem_config,
)
from optimizers import OptimizerSettings, advance_optimizer_state, initialize_optimizer_state
from trajectory.sampling import BUDGET_MILESTONE_RATIOS, sampling_phase


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO / "results/dataset_analysis/domain_scale_sensitivity"
BBOB_CONFIG_PATH = REPO / "configs/phase1_train.yaml"
CEC_CONFIG_PATH = REPO / "configs/cec2017_distribution_shift.yaml"
POPULATION_SIZE = 40
SAMPLING_PROTOCOL = "phase1_dynamic_budget_event_v1"
DOMAIN_WIDTH_RATIO = 20.0
MILESTONES = tuple(float(value) for value in (*BUDGET_MILESTONE_RATIOS, 1.0))
MILESTONE_NAMES = tuple(
    [f"m{int(round(value * 100)):02d}" for value in BUDGET_MILESTONE_RATIOS] + ["terminal"]
)
ALL_MILESTONE_NAMES = ("initial", *MILESTONE_NAMES)
MILESTONE_ORDER = {name: index for index, name in enumerate(ALL_MILESTONE_NAMES)}
SUITE_LABELS = {"bbob": "BBOB", "cec2017": "CEC2017"}
SUITE_COLORS = {"bbob": "#2F5D8C", "cec2017": "#C96B27"}
PAPER = "#FBFBFA"
CHARCOAL = "#2F3136"
GREY = "#8A929B"
GREY_LIGHT = "#D9DDE2"


@dataclass
class ReplaySpec:
    suite: str
    function: int
    instance: int
    dimension: int
    seed: int
    fe_total: int
    boundary_handling: str
    function_family_protocol: str


class BoundaryCounter:
    """Count proposed coordinates before the existing boundary operator."""

    def __init__(self) -> None:
        self.total_coordinates = 0
        self.outside_coordinates = 0
        self.total_vectors = 0
        self.outside_vectors = 0
        self.initial_total_coordinates = 0
        self.initial_outside_coordinates = 0
        self.initial_total_vectors = 0
        self.initial_outside_vectors = 0
        self.post_initial_total_coordinates = 0
        self.post_initial_outside_coordinates = 0
        self.post_initial_total_vectors = 0
        self.post_initial_outside_vectors = 0
        self.boundary_calls = 0

    def add(self, values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> None:
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            matrix = array.reshape(1, -1)
        elif array.ndim == 2:
            matrix = array
        else:
            raise ValueError(f"boundary proposal must be one- or two-dimensional, got {array.shape}")
        outside = (matrix < lower[None, :]) | (matrix > upper[None, :])
        coordinates = int(outside.size)
        outside_coordinates = int(np.count_nonzero(outside))
        vectors = int(matrix.shape[0])
        outside_vectors = int(np.count_nonzero(np.any(outside, axis=1)))
        self.total_coordinates += coordinates
        self.outside_coordinates += outside_coordinates
        self.total_vectors += vectors
        self.outside_vectors += outside_vectors
        if self.boundary_calls == 0:
            self.initial_total_coordinates += coordinates
            self.initial_outside_coordinates += outside_coordinates
            self.initial_total_vectors += vectors
            self.initial_outside_vectors += outside_vectors
        else:
            self.post_initial_total_coordinates += coordinates
            self.post_initial_outside_coordinates += outside_coordinates
            self.post_initial_total_vectors += vectors
            self.post_initial_outside_vectors += outside_vectors
        self.boundary_calls += 1

    def post_initial_snapshot(self) -> tuple[int, int, int, int]:
        return (
            self.post_initial_total_coordinates,
            self.post_initial_outside_coordinates,
            self.post_initial_total_vectors,
            self.post_initial_outside_vectors,
        )


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _phase_for_ratio(ratio: float) -> str:
    if ratio < 0.20:
        return "pre_monitor"
    if ratio <= 0.30 + 1.0e-12:
        return sampling_phase(min(max(ratio, 0.20), 0.30))
    if ratio < 0.50:
        return "mid"
    if ratio <= 0.60 + 1.0e-12:
        return "late"
    return "post_monitor"


def _dimension_label(suite: str, dimension: int) -> str:
    return f"{SUITE_LABELS[str(suite)]} {int(dimension)}D"


def _unit_population(population: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    span = upper - lower
    if np.any(span <= 0.0):
        raise ValueError("problem bounds must have positive span")
    return np.clip((np.asarray(population, dtype=float) - lower[None, :]) / span[None, :], 0.0, 1.0)


def _mean_pairwise_distance(population: np.ndarray) -> float:
    if population.shape[0] < 2:
        return 0.0
    deltas = population[:, None, :] - population[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    upper = distances[np.triu_indices(population.shape[0], k=1)]
    return float(np.mean(upper))


def _population_metrics(
    population: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    milestone: str,
    fe: int,
    fe_total: int,
) -> dict[str, Any]:
    raw = np.asarray(population, dtype=float)
    unit = _unit_population(raw, lower, upper)
    axis_ranges = np.max(unit, axis=0) - np.min(unit, axis=0)
    raw_distance = _mean_pairwise_distance(raw)
    unit_distance = _mean_pairwise_distance(unit)
    dimension = int(raw.shape[1])
    return {
        "milestone": milestone,
        "phase": "initial" if milestone == "initial" else _phase_for_ratio(float(fe / fe_total)),
        "FE": int(fe),
        "FE_ratio": float(fe / fe_total),
        "raw_pairwise_distance": raw_distance,
        "unit_pairwise_distance": unit_distance,
        "raw_pairwise_distance_per_sqrt_dimension": float(raw_distance / np.sqrt(dimension)),
        "unit_pairwise_distance_per_sqrt_dimension": float(unit_distance / np.sqrt(dimension)),
        # These are the coverage statistics of the population at this
        # milestone.  The initial population values are copied to the run
        # table under the explicit ``initial_*`` names below.
        "axis_coverage_mean": float(np.mean(axis_ranges)),
        "axis_coverage_median": float(np.median(axis_ranges)),
        "axis_coverage_min": float(np.min(axis_ranges)),
        "axis_coverage_max": float(np.max(axis_ranges)),
    }


def _counter_rates(counter: BoundaryCounter, prefix: str = "") -> dict[str, Any]:
    def key(name: str) -> str:
        return f"{prefix}{name}" if prefix else name

    if prefix == "initial_":
        total_coordinates = counter.initial_total_coordinates
        outside_coordinates = counter.initial_outside_coordinates
        total_vectors = counter.initial_total_vectors
        outside_vectors = counter.initial_outside_vectors
    elif prefix == "post_initial_":
        total_coordinates = counter.post_initial_total_coordinates
        outside_coordinates = counter.post_initial_outside_coordinates
        total_vectors = counter.post_initial_total_vectors
        outside_vectors = counter.post_initial_outside_vectors
    else:
        total_coordinates = counter.total_coordinates
        outside_coordinates = counter.outside_coordinates
        total_vectors = counter.total_vectors
        outside_vectors = counter.outside_vectors
    return {
        key("proposed_coordinate_count"): int(total_coordinates),
        key("reflected_coordinate_count"): int(outside_coordinates),
        key("proposed_vector_count"): int(total_vectors),
        key("reflected_vector_count"): int(outside_vectors),
        key("reflection_coordinate_rate"): _rate(outside_coordinates, total_coordinates),
        key("reflection_vector_rate"): _rate(outside_vectors, total_vectors),
    }


def _make_specs() -> tuple[list[ReplaySpec], dict[str, Any]]:
    bbob_configs = [
        config for config in load_suite_configs(BBOB_CONFIG_PATH) if str(config["suite"]).lower() == "bbob"
    ]
    if len(bbob_configs) != 1:
        raise ValueError("phase1_train.yaml must expose exactly one BBOB configuration")
    bbob = bbob_configs[0]
    cec = load_config(CEC_CONFIG_PATH)
    common = {
        "population_size": int(bbob["population_size"]),
        "sampling_protocol": str(bbob["sampling_protocol"]),
        "bbob_boundary_handling": str(bbob.get("boundary_handling", "reflect")),
        "cec_boundary_handling": str(cec.get("boundary_handling", "reflect")),
    }
    if common["population_size"] != POPULATION_SIZE:
        raise ValueError(f"expected population size {POPULATION_SIZE}, got {common['population_size']}")
    if common["sampling_protocol"] != SAMPLING_PROTOCOL:
        raise ValueError("replay must use the predefined dynamic sampling protocol")

    specs: list[ReplaySpec] = []
    for function in as_int_list(bbob, "functions"):
        for dimension in as_int_list(bbob, "dimensions"):
            for seed in as_int_list(bbob, "seeds"):
                specs.append(
                    ReplaySpec(
                        suite="bbob",
                        function=function,
                        instance=1,
                        dimension=dimension,
                        seed=seed,
                        fe_total=fe_total_for_dimension(bbob, dimension),
                        boundary_handling=common["bbob_boundary_handling"],
                        function_family_protocol=str(bbob["function_family_protocol"]),
                    )
                )
    for function in as_int_list(cec, "functions"):
        for dimension in as_int_list(cec, "dimensions"):
            for seed in as_int_list(cec, "seeds"):
                specs.append(
                    ReplaySpec(
                        suite="cec2017",
                        function=function,
                        instance=1,
                        dimension=dimension,
                        seed=seed,
                        fe_total=fe_total_for_dimension(cec, dimension),
                        boundary_handling=common["cec_boundary_handling"],
                        function_family_protocol=str(cec["function_family_protocol"]),
                    )
                )
    metadata = {
        "bbob_functions": as_int_list(bbob, "functions"),
        "bbob_dimensions": as_int_list(bbob, "dimensions"),
        "bbob_instances": [1],
        "bbob_seeds": as_int_list(bbob, "seeds"),
        "cec_functions": as_int_list(cec, "functions"),
        "cec_dimensions": as_int_list(cec, "dimensions"),
        "cec_instances": as_int_list(cec, "instances"),
        "cec_seeds": as_int_list(cec, "seeds"),
        "population_size": common["population_size"],
        "sampling_protocol": common["sampling_protocol"],
        "bbob_boundary_handling": common["bbob_boundary_handling"],
        "cec_boundary_handling": common["cec_boundary_handling"],
        "bbob_config": str(BBOB_CONFIG_PATH),
        "cec_config": str(CEC_CONFIG_PATH),
    }
    return specs, metadata


def _run_replay(spec: ReplaySpec) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    problem_config = {
        "suite": spec.suite,
        "function": spec.function,
        "instance": spec.instance,
        "dimension": spec.dimension,
        "boundary_handling": spec.boundary_handling,
    }
    if spec.suite == "cec2017":
        problem_config["function_family_protocol"] = spec.function_family_protocol
    problem = make_problem(problem_config)
    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)
    counter = BoundaryCounter()
    original_boundary_handler = optimizer_state_module._apply_boundary_handling

    def counted_boundary_handler(
        values: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        *,
        boundary_handling: str,
    ) -> np.ndarray:
        if str(boundary_handling) == "reflect":
            counter.add(values, np.asarray(lower_bounds, dtype=float), np.asarray(upper_bounds, dtype=float))
        return original_boundary_handler(
            values,
            lower_bounds,
            upper_bounds,
            boundary_handling=boundary_handling,
        )

    optimizer_state_module._apply_boundary_handling = counted_boundary_handler
    metric_rows: list[dict[str, Any]] = []
    phase_counts: dict[str, dict[str, int]] = {}
    previous_post_initial = counter.post_initial_snapshot()
    target_index = 0
    settings = OptimizerSettings(
        population_size=POPULATION_SIZE,
        sampling_protocol=SAMPLING_PROTOCOL,
        boundary_handling=spec.boundary_handling,
    )
    try:
        settings.validate(spec.fe_total)
        state = initialize_optimizer_state(
            algorithm="cmaes",
            problem=problem,
            seed=spec.seed,
            settings=settings,
        )
        initial = _population_metrics(
            state.population,
            lower,
            upper,
            milestone="initial",
            fe=state.evaluations,
            fe_total=spec.fe_total,
        )
        metric_rows.append(initial)
        target_fes = [int(round(ratio * spec.fe_total)) for ratio in MILESTONES]

        def on_native_update(updated: Any) -> None:
            nonlocal previous_post_initial, target_index
            current_post_initial = counter.post_initial_snapshot()
            delta = tuple(
                int(current_post_initial[index] - previous_post_initial[index])
                for index in range(4)
            )
            phase = _phase_for_ratio(float(updated.evaluations / spec.fe_total))
            bucket = phase_counts.setdefault(
                phase,
                {
                    "proposed_coordinates": 0,
                    "reflected_coordinates": 0,
                    "proposed_vectors": 0,
                    "reflected_vectors": 0,
                },
            )
            bucket["proposed_coordinates"] += delta[0]
            bucket["reflected_coordinates"] += delta[1]
            bucket["proposed_vectors"] += delta[2]
            bucket["reflected_vectors"] += delta[3]
            previous_post_initial = current_post_initial
            while target_index < len(target_fes) and int(updated.evaluations) >= target_fes[target_index]:
                milestone = MILESTONE_NAMES[target_index]
                metric_rows.append(
                    _population_metrics(
                        updated.population,
                        lower,
                        upper,
                        milestone=milestone,
                        fe=int(updated.evaluations),
                        fe_total=spec.fe_total,
                    )
                )
                target_index += 1

        while int(state.evaluations) < spec.fe_total:
            advance_optimizer_state(
                state=state,
                problem=problem,
                fe_budget=spec.fe_total - int(state.evaluations),
                on_native_update=on_native_update,
            )
        if target_index != len(target_fes):
            raise RuntimeError(
                f"replay did not reach all milestone targets: {target_index}/{len(target_fes)}"
            )
    finally:
        optimizer_state_module._apply_boundary_handling = original_boundary_handler
        problem.close()

    run_row: dict[str, Any] = {
        "suite": spec.suite,
        "suite_label": SUITE_LABELS[spec.suite],
        "function": int(spec.function),
        "function_label": f"F{int(spec.function):02d}",
        "instance": int(spec.instance),
        "dimension": int(spec.dimension),
        "dimension_label": _dimension_label(spec.suite, spec.dimension),
        "seed": int(spec.seed),
        "FE_total": int(spec.fe_total),
        "boundary_handling": spec.boundary_handling,
        "replay_algorithm": "cmaes",
        "boundary_call_count": int(counter.boundary_calls),
    }
    run_row.update(_counter_rates(counter))
    run_row.update(_counter_rates(counter, "initial_"))
    run_row.update(_counter_rates(counter, "post_initial_"))
    for phase in ("pre_monitor", "early", "mid", "late", "post_monitor"):
        bucket = phase_counts.get(
            phase,
            {
                "proposed_coordinates": 0,
                "reflected_coordinates": 0,
                "proposed_vectors": 0,
                "reflected_vectors": 0,
            },
        )
        run_row[f"{phase}_reflection_coordinate_rate"] = _rate(
            bucket["reflected_coordinates"], bucket["proposed_coordinates"]
        )
        run_row[f"{phase}_reflection_vector_rate"] = _rate(
            bucket["reflected_vectors"], bucket["proposed_vectors"]
        )
    initial_metrics = metric_rows[0]
    for column in (
        "raw_pairwise_distance",
        "unit_pairwise_distance",
        "raw_pairwise_distance_per_sqrt_dimension",
        "unit_pairwise_distance_per_sqrt_dimension",
        "axis_coverage_mean",
        "axis_coverage_median",
        "axis_coverage_min",
        "axis_coverage_max",
    ):
        run_column = (
            column.replace("axis_coverage", "initial_axis_coverage")
            if column.startswith("axis_coverage")
            else column
        )
        run_row[run_column] = initial_metrics[column]
    for row in metric_rows:
        row.update(
            {
                "suite": spec.suite,
                "suite_label": SUITE_LABELS[spec.suite],
                "function": int(spec.function),
                "function_label": f"F{int(spec.function):02d}",
                "instance": int(spec.instance),
                "dimension": int(spec.dimension),
                "dimension_label": _dimension_label(spec.suite, spec.dimension),
                "seed": int(spec.seed),
                "FE_total": int(spec.fe_total),
                "boundary_handling": spec.boundary_handling,
                "replay_algorithm": "cmaes",
            }
        )
    return run_row, metric_rows


def _run_all(specs: list[ReplaySpec], workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    if workers <= 1:
        results = ((_run_replay(spec), spec) for spec in specs)
        for result, spec in results:
            run_row, rows = result
            run_rows.append(run_row)
            metric_rows.extend(rows)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_replay, spec): spec for spec in specs}
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    run_row, rows = future.result()
                except Exception as exc:
                    failures.append(
                        f"{spec.suite} F{spec.function:02d} {spec.dimension}D seed={spec.seed}: {type(exc).__name__}: {exc}"
                    )
                    continue
                run_rows.append(run_row)
                metric_rows.extend(rows)
    if failures:
        raise RuntimeError("replay failures:\n" + "\n".join(sorted(failures)))
    runs = pd.DataFrame(run_rows).sort_values(
        ["suite", "function", "dimension", "seed"], ignore_index=True
    )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["suite", "function", "dimension", "seed", "FE"], ignore_index=True
    )
    expected_runs = len(specs)
    if len(runs) != expected_runs or len(runs.drop_duplicates(["suite", "function", "dimension", "seed"])) != expected_runs:
        raise ValueError(f"replay run count mismatch: expected {expected_runs}, got {len(runs)}")
    expected_metrics = expected_runs * (1 + len(MILESTONES))
    if len(metrics) != expected_metrics:
        raise ValueError(f"replay milestone count mismatch: expected {expected_metrics}, got {len(metrics)}")
    return runs, metrics


def _summary_table(frame: pd.DataFrame, columns: list[str], group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, observed=True, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_columns, keys)}
        row["n_runs"] = int(len(group))
        for column in columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(float)
            row[f"{column}_median"] = float(np.median(values)) if len(values) else float("nan")
            row[f"{column}_q25"] = float(np.quantile(values, 0.25)) if len(values) else float("nan")
            row[f"{column}_q75"] = float(np.quantile(values, 0.75)) if len(values) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _build_scale_decomposition(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = _summary_table(
        metrics,
        [
            "raw_pairwise_distance",
            "unit_pairwise_distance",
            "raw_pairwise_distance_per_sqrt_dimension",
            "unit_pairwise_distance_per_sqrt_dimension",
        ],
        ["suite", "dimension", "milestone", "phase"],
    )
    rows: list[dict[str, Any]] = []
    for cec_dimension, bbob_dimension, comparison in (
        (10, 10, "exact dimension"),
        (20, 20, "exact dimension"),
        (30, 40, "nearest high-dimensional reference"),
        (50, 40, "nearest high-dimensional reference"),
    ):
        cec = summary[summary["suite"].eq("cec2017") & summary["dimension"].eq(cec_dimension)]
        bbob = summary[summary["suite"].eq("bbob") & summary["dimension"].eq(bbob_dimension)]
        if cec.empty or bbob.empty:
            continue
        for milestone in sorted(
            set(cec["milestone"]) & set(bbob["milestone"]),
            key=lambda value: MILESTONE_ORDER[str(value)],
        ):
            cec_row = cec[cec["milestone"].eq(milestone)].iloc[0]
            bbob_row = bbob[bbob["milestone"].eq(milestone)].iloc[0]
            cec_raw = float(cec_row["raw_pairwise_distance_median"])
            bbob_raw = float(bbob_row["raw_pairwise_distance_median"])
            cec_unit = float(cec_row["unit_pairwise_distance_median"])
            bbob_unit = float(bbob_row["unit_pairwise_distance_median"])
            cec_raw_sqrt = float(cec_row["raw_pairwise_distance_per_sqrt_dimension_median"])
            bbob_raw_sqrt = float(bbob_row["raw_pairwise_distance_per_sqrt_dimension_median"])
            cec_unit_sqrt = float(cec_row["unit_pairwise_distance_per_sqrt_dimension_median"])
            bbob_unit_sqrt = float(bbob_row["unit_pairwise_distance_per_sqrt_dimension_median"])
            raw_ratio = float(cec_raw / max(bbob_raw, np.finfo(float).tiny))
            unit_ratio = float(cec_unit / max(bbob_unit, np.finfo(float).tiny))
            log_raw_ratio = float(np.log(raw_ratio))
            log_boundary_scale = float(np.log(DOMAIN_WIDTH_RATIO))
            log_structural_residual = float(np.log(unit_ratio))
            component_magnitude = abs(log_boundary_scale) + abs(log_structural_residual)
            rows.append(
                {
                    "cec_dimension": cec_dimension,
                    "bbob_reference_dimension": bbob_dimension,
                    "comparison": comparison,
                    "milestone": milestone,
                    "phase": str(cec_row["phase"]),
                    "domain_width_ratio": DOMAIN_WIDTH_RATIO,
                    "cec_raw_distance_median": cec_raw,
                    "bbob_raw_distance_median": bbob_raw,
                    "cec_to_bbob_raw_distance_ratio": raw_ratio,
                    "raw_ratio_after_domain_scale": float(raw_ratio / DOMAIN_WIDTH_RATIO),
                    "cec_unit_distance_median": cec_unit,
                    "bbob_unit_distance_median": bbob_unit,
                    "cec_to_bbob_unit_distance_ratio": unit_ratio,
                    "unit_distance_difference_cec_minus_bbob": float(cec_unit - bbob_unit),
                    "cec_raw_distance_per_sqrt_dimension_median": cec_raw_sqrt,
                    "bbob_raw_distance_per_sqrt_dimension_median": bbob_raw_sqrt,
                    "cec_to_bbob_raw_per_sqrt_dimension_ratio": float(cec_raw_sqrt / max(bbob_raw_sqrt, np.finfo(float).tiny)),
                    "cec_unit_distance_per_sqrt_dimension_median": cec_unit_sqrt,
                    "bbob_unit_distance_per_sqrt_dimension_median": bbob_unit_sqrt,
                    "cec_to_bbob_unit_per_sqrt_dimension_ratio": float(cec_unit_sqrt / max(bbob_unit_sqrt, np.finfo(float).tiny)),
                    # Exact log-scale decomposition:
                    # log(raw CEC/BBOB) = log(domain-width ratio) +
                    # log(unit-cube CEC/BBOB).  The last term is the
                    # normalized geometry/function/trajectory residual.
                    "log_raw_distance_ratio": log_raw_ratio,
                    "log_boundary_scale_contribution": log_boundary_scale,
                    "log_structural_residual": log_structural_residual,
                    "log_decomposition_residual": float(
                        log_raw_ratio - log_boundary_scale - log_structural_residual
                    ),
                    "boundary_scale_magnitude_share": float(
                        abs(log_boundary_scale) / component_magnitude
                        if component_magnitude
                        else np.nan
                    ),
                    "structure_magnitude_share": float(
                        abs(log_structural_residual) / component_magnitude
                        if component_magnitude
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _build_scale_attribution_summary(decomposition: pd.DataFrame) -> pd.DataFrame:
    """Summarize the additive log-scale components for exact dimensions."""

    exact = decomposition[decomposition["comparison"].eq("exact dimension")].copy()
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [
        (f"{int(dimension)}D", group)
        for dimension, group in exact.groupby("cec_dimension", sort=True)
    ]
    if not exact.empty:
        groups.append(("pooled_exact", exact))
    for dimension_label, group in groups:
        rows.append(
            {
                "dimension_group": dimension_label,
                "n_milestones": int(len(group)),
                "median_raw_distance_ratio": float(group["cec_to_bbob_raw_distance_ratio"].median()),
                "median_raw_ratio_after_domain_scale": float(group["raw_ratio_after_domain_scale"].median()),
                "median_unit_distance_ratio": float(group["cec_to_bbob_unit_distance_ratio"].median()),
                "median_log_raw_distance_ratio": float(group["log_raw_distance_ratio"].median()),
                "boundary_log_contribution": float(group["log_boundary_scale_contribution"].median()),
                "median_log_structural_residual": float(group["log_structural_residual"].median()),
                "median_abs_log_raw_distance_ratio": float(group["log_raw_distance_ratio"].abs().median()),
                "median_abs_log_structural_residual": float(group["log_structural_residual"].abs().median()),
                "median_boundary_scale_magnitude_share": float(group["boundary_scale_magnitude_share"].median()),
                "median_structure_magnitude_share": float(group["structure_magnitude_share"].median()),
                "max_abs_log_decomposition_residual": float(group["log_decomposition_residual"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Microsoft YaHei",
            "font.sans-serif": ["Microsoft YaHei", "PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
            "axes.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "axes.edgecolor": GREY,
            "axes.labelcolor": CHARCOAL,
            "xtick.color": CHARCOAL,
            "ytick.color": CHARCOAL,
            "text.color": CHARCOAL,
            "axes.grid": True,
            "grid.color": GREY_LIGHT,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "axes.axisbelow": True,
            "savefig.dpi": 220,
        }
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def _boxplot_by_group(
    ax: plt.Axes,
    frame: pd.DataFrame,
    value_column: str,
    groups: list[tuple[str, int]],
    *,
    ylabel: str,
) -> None:
    data: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[str] = []
    for suite, dimension in groups:
        values = pd.to_numeric(
            frame[frame["suite"].eq(suite) & frame["dimension"].eq(dimension)][value_column],
            errors="coerce",
        ).dropna().to_numpy(float)
        if len(values) == 0:
            continue
        data.append(values)
        labels.append(_dimension_label(suite, dimension).replace(" ", "\n", 1))
        colors.append(SUITE_COLORS[suite])
    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.62,
        showfliers=False,
        medianprops={"color": CHARCOAL, "linewidth": 1.3},
        whiskerprops={"color": GREY, "linewidth": 0.9},
        capprops={"color": GREY, "linewidth": 0.9},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_edgecolor(CHARCOAL)
    ax.axhline(0.0, color=CHARCOAL, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=8)
    ax.set_ylabel(ylabel)


def _figure_initial_diagnostics(runs: pd.DataFrame, output: Path) -> None:
    groups = [
        ("bbob", 10), ("cec2017", 10),
        ("bbob", 20), ("cec2017", 20),
        ("bbob", 40), ("cec2017", 30), ("cec2017", 50),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.0))
    _boxplot_by_group(axes[0, 0], runs, "raw_pairwise_distance", groups, ylabel="原始坐标平均两两距离")
    axes[0, 0].set_title("初始群体：原始坐标距离", fontsize=11, weight="bold")
    _boxplot_by_group(axes[0, 1], runs, "unit_pairwise_distance", groups, ylabel="单位立方体平均两两距离")
    axes[0, 1].set_title("初始群体：单位立方体距离", fontsize=11, weight="bold")
    _boxplot_by_group(axes[1, 0], runs, "initial_axis_coverage_mean", groups, ylabel="平均逐维覆盖率")
    axes[1, 0].set_title("初始群体：平均逐维覆盖率", fontsize=11, weight="bold")
    _boxplot_by_group(axes[1, 1], runs, "initial_reflection_coordinate_rate", groups, ylabel="初始候选坐标反射比例")
    axes[1, 1].set_title("初始 CMA-ES 群体：越界坐标比例", fontsize=11, weight="bold")
    fig.suptitle("BBOB 与 CEC 的初始搜索空间诊断", x=0.07, ha="left", y=0.985, fontsize=15, weight="bold")
    fig.text(
        0.07,
        0.945,
        "箱线图为当前 CMA-ES replay 的 run-level 分布；CEC 30/50D 与 BBOB 40D 仅作近邻维度参考，不能视为同维度比较。",
        fontsize=9.5,
        color=GREY,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.88, wspace=0.24, hspace=0.32)
    _save_figure(fig, output / "fig01_initial_search_space_diagnostics.png")


def _figure_distance_over_budget(metrics: pd.DataFrame, output: Path) -> None:
    frame = metrics[metrics["milestone"].ne("initial")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.5), sharex=True)
    linestyles = {10: "-", 20: "--", 30: ":", 40: "-.", 50: (0, (3, 1, 1, 1))}
    for (suite, dimension), group in frame.groupby(["suite", "dimension"], sort=True):
        summary = group.groupby("milestone", as_index=False).agg(
            FE_ratio=("FE_ratio", "median"),
            raw=("raw_pairwise_distance", "median"),
            unit=("unit_pairwise_distance", "median"),
        )
        summary["milestone_order"] = summary["milestone"].map({name: index for index, name in enumerate(MILESTONE_NAMES)})
        summary = summary.sort_values("milestone_order")
        label = _dimension_label(suite, dimension)
        style = linestyles.get(int(dimension), "-")
        axes[0].plot(summary["FE_ratio"], summary["raw"], color=SUITE_COLORS[suite], linestyle=style, linewidth=1.7, marker="o", markersize=3.5, label=label)
        axes[1].plot(summary["FE_ratio"], summary["unit"], color=SUITE_COLORS[suite], linestyle=style, linewidth=1.7, marker="o", markersize=3.5, label=label)
    for ax, ylabel, title in (
        (axes[0], "原始坐标平均两两距离", "原始坐标距离随预算阶段变化"),
        (axes[1], "单位立方体平均两两距离", "单位立方体距离随预算阶段变化"),
    ):
        ax.set_xlabel("FE ratio（replay milestone）")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xticks([0.2, 0.3, 0.5, 0.6, 1.0], ["0.20", "0.30", "0.50", "0.60", "1.00"])
        ax.grid(axis="y")
    axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    fig.suptitle("BBOB 与 CEC 的群体距离轨迹", x=0.07, ha="left", y=0.985, fontsize=15, weight="bold")
    fig.text(
        0.07,
        0.925,
        "颜色区分 benchmark，线型区分维度；原始距离保留坐标单位，单位立方体距离用于观察去除统一边界尺度后的轨迹差异。",
        fontsize=9.5,
        color=GREY,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.80, wspace=0.22)
    _save_figure(fig, output / "fig02_distance_over_budget.png")


def _figure_reflection_rates(runs: pd.DataFrame, output: Path) -> None:
    groups = [
        ("bbob", 10), ("cec2017", 10),
        ("bbob", 20), ("cec2017", 20),
        ("bbob", 40), ("cec2017", 30), ("cec2017", 50),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.3), sharey=True)
    for ax, value_column, title in (
        (axes[0], "initial_reflection_coordinate_rate", "初始 CMA-ES 生成"),
        (axes[1], "post_initial_reflection_coordinate_rate", "初始群体之后的完整 replay"),
    ):
        positions = np.arange(len(groups), dtype=float)
        medians = []
        lower = []
        upper = []
        colors = []
        labels = []
        for suite, dimension in groups:
            values = pd.to_numeric(
                runs[runs["suite"].eq(suite) & runs["dimension"].eq(dimension)][value_column],
                errors="coerce",
            ).dropna().to_numpy(float)
            medians.append(float(np.median(values)) if len(values) else np.nan)
            lower.append(float(np.quantile(values, 0.25)) if len(values) else np.nan)
            upper.append(float(np.quantile(values, 0.75)) if len(values) else np.nan)
            colors.append(SUITE_COLORS[suite])
            labels.append(_dimension_label(suite, dimension).replace(" ", "\n", 1))
        medians = np.asarray(medians, dtype=float)
        yerr = np.vstack([medians - np.asarray(lower), np.asarray(upper) - medians])
        ax.errorbar(
            positions,
            medians,
            yerr=yerr,
            fmt="none",
            ecolor=GREY,
            elinewidth=1.0,
            capsize=3,
            zorder=2,
        )
        ax.scatter(positions, medians, c=colors, s=70, edgecolor=CHARCOAL, linewidth=0.6, zorder=3)
        ax.set_xticks(positions, labels, fontsize=8)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_ylabel("越界候选坐标比例")
        ax.set_ylim(bottom=0.0)
        ax.grid(axis="y")
    fig.suptitle("CMA-ES 边界反射比例", x=0.07, ha="left", y=0.985, fontsize=15, weight="bold")
    fig.text(
        0.07,
        0.94,
        "点为 5 个 seed 的中位数，误差线为跨 seed 的四分位区间；统计对象是反射前越界候选，而非修复后落在边界的点。",
        fontsize=9.5,
        color=GREY,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.83, wspace=0.16)
    _save_figure(fig, output / "fig03_reflection_rates.png")


def _figure_scale_decomposition(decomposition: pd.DataFrame, output: Path) -> None:
    exact = decomposition[decomposition["comparison"].eq("exact dimension")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.9), sharex=True)
    colors = {10: "#2F5D8C", 20: "#C08A2D"}
    for dimension, group in exact.groupby("cec_dimension", sort=True):
        order = group["milestone"].map(MILESTONE_ORDER)
        group = group.assign(_order=order).sort_values("_order")
        x = group["milestone"].map(MILESTONE_ORDER).to_numpy(float)
        axes[0].plot(x, group["cec_to_bbob_raw_distance_ratio"], marker="o", linewidth=1.7, color=colors[int(dimension)], label=f"{dimension}D")
        axes[1].plot(x, group["cec_to_bbob_unit_distance_ratio"], marker="o", linewidth=1.7, color=colors[int(dimension)], label=f"{dimension}D")
    axes[0].axhline(DOMAIN_WIDTH_RATIO, color=CHARCOAL, linestyle="--", linewidth=1.0, label="纯边界尺度预期 = 20")
    axes[1].axhline(1.0, color=CHARCOAL, linestyle="--", linewidth=1.0, label="归一化后相等 = 1")
    for ax, title, ylabel in (
        (axes[0], "原始距离比：CEC / BBOB", "原始坐标距离比"),
        (axes[1], "单位立方体距离比：CEC / BBOB", "单位立方体距离比"),
    ):
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("FE ratio（replay milestone）")
        ax.set_xticks(
            range(len(ALL_MILESTONE_NAMES)),
            ["初始" if name == "initial" else name.replace("m", "") for name in ALL_MILESTONE_NAMES],
            rotation=30,
        )
        ax.grid(axis="y")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("边界尺度与函数/轨迹结构的可分解诊断", x=0.08, ha="left", y=0.985, fontsize=15, weight="bold")
    fig.text(
        0.08,
        0.925,
        "原始距离比接近 20 表示坐标单位差异占主导；单位立方体距离比偏离 1 则表示归一化后仍存在轨迹/函数结构差异。",
        fontsize=9.5,
        color=GREY,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.17, top=0.80, wspace=0.22)
    _save_figure(fig, output / "fig04_scale_decomposition.png")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "NA"
    return f"{numeric:.{digits}g}"


def _write_report(
    output: Path,
    metadata: dict[str, Any],
    runs: pd.DataFrame,
    metrics: pd.DataFrame,
    reflection_summary: pd.DataFrame,
    coverage_summary: pd.DataFrame,
    decomposition: pd.DataFrame,
    attribution_summary: pd.DataFrame,
) -> None:
    def median_where(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
        values = pd.to_numeric(frame.loc[mask, column], errors="coerce").dropna().to_numpy(float)
        return float(np.median(values)) if len(values) else float("nan")

    bbob = runs[runs["suite"].eq("bbob")]
    cec = runs[runs["suite"].eq("cec2017")]
    bbob_raw = median_where(bbob, pd.Series(True, index=bbob.index), "raw_pairwise_distance")
    cec_raw = median_where(cec, pd.Series(True, index=cec.index), "raw_pairwise_distance")
    bbob_unit = median_where(bbob, pd.Series(True, index=bbob.index), "unit_pairwise_distance")
    cec_unit = median_where(cec, pd.Series(True, index=cec.index), "unit_pairwise_distance")
    bbob_coverage = median_where(bbob, pd.Series(True, index=bbob.index), "initial_axis_coverage_mean")
    cec_coverage = median_where(cec, pd.Series(True, index=cec.index), "initial_axis_coverage_mean")
    bbob_initial_reflection = median_where(bbob, pd.Series(True, index=bbob.index), "initial_reflection_coordinate_rate")
    cec_initial_reflection = median_where(cec, pd.Series(True, index=cec.index), "initial_reflection_coordinate_rate")
    bbob_post_reflection = median_where(bbob, pd.Series(True, index=bbob.index), "post_initial_reflection_coordinate_rate")
    cec_post_reflection = median_where(cec, pd.Series(True, index=cec.index), "post_initial_reflection_coordinate_rate")

    lines = [
        "# BBOB/CEC 搜索空间尺度敏感性分析",
        "",
        "> 目标：在不改变 Decision 模型、预处理、阈值或主标签的前提下，区分边界尺度与函数/轨迹结构对当前 CMA-ES 行为差异的影响。",
        "",
        "## 结论摘要",
        "",
        f"1. 本次 replay 覆盖 {len(runs)} 条 CMA-ES run、{len(metrics)} 个群体检查点；BBOB 使用当前训练函数的 instance 1，CEC 使用当前 online 测评的 5 个函数和 instance 1。",
        f"2. 当前 factory 实测 BBOB 每维边界宽度为 10，CEC2017 为 200，边界宽度比为 {DOMAIN_WIDTH_RATIO:g}。因此原始坐标距离出现约 20 倍的纯单位尺度差异是预期的。",
        f"3. replay 的初始平均逐维覆盖率中位数为 BBOB={_fmt(bbob_coverage, 5)}、CEC={_fmt(cec_coverage, 5)}；如果两者接近，说明 uniform/相对跨度初始化后的覆盖率并未因绝对边界大小而系统改变。",
        f"4. 初始 CMA-ES 反射坐标比例中位数为 BBOB={_fmt(bbob_initial_reflection, 5)}、CEC={_fmt(cec_initial_reflection, 5)}；初始群体之后的完整 replay 比例为 BBOB={_fmt(bbob_post_reflection, 5)}、CEC={_fmt(cec_post_reflection, 5)}。后者差异包含函数选择、最优点位置和轨迹演化的影响，不能只解释为坐标单位。",
        f"5. 原始距离与单位立方体距离满足严格的对数分解：`log(raw CEC/BBOB) = log({DOMAIN_WIDTH_RATIO:g}) + log(unit-cube CEC/BBOB)`。因此 `log(20)` 是边界单位尺度项，第二项是归一化几何、函数响应和搜索轨迹的结构项；两者的阶段级数值见 `scale_decomposition.csv`。",
        "",
        "## 统计口径",
        "",
        "- 算法：当前 online 前缀 `cmaes`；population size=40；使用当前正式 FE budget 和 `reflect` 边界处理。",
        "- BBOB：当前 `phase1_train.yaml` 的 BBOB 函数、10/20/40D、seeds 1--5，仅取 instance 1 以匹配 CEC 的单 instance 口径。",
        "- CEC：当前 `cec2017_distribution_shift.yaml` 的 F01/F05/F09/F20/F24、10/20/30/50D、seeds 1--5。",
        "- 原始坐标距离：种群 40 个点的平均两两欧氏距离，保留真实坐标单位。",
        "- 单位立方体距离：将每个坐标按 `(x-lower)/(upper-lower)` 映射到 `[0,1]` 后计算相同的平均两两距离。",
        "- 初始群体覆盖率：初始群体在单位立方体中每一维的 `max-min`，再对维度取平均。该定义避免高维超矩形体积因样本稀疏而快速接近 0。",
        "- 反射比例：反射函数调用前，越出相应边界的候选坐标数 / 全部候选坐标数；同时保存候选向量比例。它不是修复后边界点比例。",
        f"- 尺度归因：在同维度比较中，原始距离比的边界尺度项固定为 `log({DOMAIN_WIDTH_RATIO:g})={np.log(DOMAIN_WIDTH_RATIO):.6f}`；结构项为单位立方体距离比的自然对数。`boundary_scale_magnitude_share` 仅表示两项绝对对数幅度中的边界项比例，遇到两项方向相反时不等同于结果差异的可加百分比。",
        "",
        "## 全局中位数（描述性）",
        "",
        "| suite | raw distance | unit-cube distance | initial coverage | initial reflection | post-initial reflection |",
        "|---|---:|---:|---:|---:|---:|",
        f"| BBOB | {_fmt(bbob_raw, 6)} | {_fmt(bbob_unit, 6)} | {_fmt(bbob_coverage, 6)} | {_fmt(bbob_initial_reflection, 6)} | {_fmt(bbob_post_reflection, 6)} |",
        f"| CEC2017 | {_fmt(cec_raw, 6)} | {_fmt(cec_unit, 6)} | {_fmt(cec_coverage, 6)} | {_fmt(cec_initial_reflection, 6)} | {_fmt(cec_post_reflection, 6)} |",
        "",
        "这里的全局中位数用于快速定位，不替代按维度和 FE milestone 的分层表。原始距离受到维度与边界单位共同影响；单位立方体距离更接近函数/轨迹结构诊断。",
        "",
        "## 分层结果",
        "",
        "`reflection_summary.csv` 保存按 suite×dimension 的反射比例中位数和四分位区间；`initial_coverage_summary.csv` 保存初始距离和覆盖率；`distance_milestone_summary.csv` 保存每个预定义 FE milestone 的距离分布。",
        "",
    ]
    if not decomposition.empty:
        lines.extend(
            [
                "### 同维度尺度归因摘要（含初始群体）",
                "",
                "| 维度 | 比较点数 | 中位原始距离比 | 中位原始比值 / 20 | 中位结构项 `log(unit ratio)` | 边界项绝对幅度占比* | 结构项绝对幅度占比* |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in attribution_summary.iterrows():
            lines.append(
                f"| `{row['dimension_group']}` | {int(row['n_milestones'])} | {_fmt(row['median_raw_distance_ratio'], 5)} | {_fmt(row['median_raw_ratio_after_domain_scale'], 5)} | {_fmt(row['median_log_structural_residual'], 5)} | {_fmt(row['median_boundary_scale_magnitude_share'], 5)} | {_fmt(row['median_structure_magnitude_share'], 5)} |"
            )
        lines.extend(
            [
                "",
                "* 两个幅度占比由 `|log(20)|` 与 `|log(unit ratio)|` 归一化得到；若边界项与结构项方向相反，二者会发生抵消，所以该表用于比较组件幅度，不把它解释为结果差异的因果百分比。",
                "",
                "### 10D/20D 的阶段级尺度分解",
                "",
                "| CEC 维度 | FE milestone | 原始距离比 CEC/BBOB | 原始比值 / 20 | 单位立方体距离比 |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for _, row in decomposition[decomposition["comparison"].eq("exact dimension")].iterrows():
            lines.append(
                f"| {int(row['cec_dimension'])} | `{row['milestone']}` | {_fmt(row['cec_to_bbob_raw_distance_ratio'], 5)} | {_fmt(row['raw_ratio_after_domain_scale'], 5)} | {_fmt(row['cec_to_bbob_unit_distance_ratio'], 5)} |"
            )
        lines.extend(
            [
                "",
                "解释时，`原始比值 / 20` 与单位立方体距离比在数值上相同；越接近 1，越说明归一化后的群体几何相似。原始距离比与 20 的差异由结构项承担。",
            ]
        )
    lines.extend(
        [
            "",
            "## 与 Decision 未触发问题的关系",
            "",
            "1. 如果原始坐标距离的差异约为 20 倍，但单位立方体距离和初始覆盖率接近，那么绝对搜索边界主要改变了坐标单位，而不是直接把初始群体覆盖率推离训练域。",
            "2. 如果反射比例在初始阶段接近、但完整 replay 后差异扩大，则差异主要发生在目标函数驱动的 CMA-ES 轨迹演化阶段；这与当前 CEC 行为特征的联合分布偏移相符。",
            "3. 如果单位立方体距离、覆盖率或反射比例仍有系统差异，那么边界尺度通过最优点位置、函数变换和边界交互改变搜索轨迹，最终可能间接影响 `bf_*` 输入和 Decision score。该分析本身不把这种关系表述为单一因素的因果结论。",
            "4. 本敏感性分析不重新计算 Decision score，不将 raw distance、reflection rate 或 coverage 加入模型输入，也不改写主实验结论。",
            "",
            "## 输出文件",
            "",
            "- `reflection_replay_runs.csv`：每条 replay 的反射比例、初始覆盖率和初始距离。",
            "- `distance_milestones.csv`：每条 replay 在 initial、预定义 FE milestones 和 terminal 的距离/覆盖率。",
            "- `reflection_summary.csv`：suite×dimension 分层反射比例。",
            "- `initial_coverage_summary.csv`：suite×dimension 分层初始距离和覆盖率。",
            "- `distance_milestone_summary.csv`：suite×dimension×milestone 分层距离。",
            "- `scale_decomposition.csv`：CEC/BBOB 原始距离比、单位立方体距离比和边界尺度基准。",
            "- `scale_attribution_summary.csv`：同维度比较的对数尺度归因摘要。",
            "- `fig01_initial_search_space_diagnostics.png` 至 `fig04_scale_decomposition.png`：四张诊断图。",
            "",
            "## 下一步建议",
            "",
            "若需要继续定位 Decision score 偏移，下一步应把本报告的 unit-cube 轨迹差异与 29 个 `bf_*` 特征逐列关联，优先检查那些在单位立方体距离差异较大且模型 split importance 较高的特征；仍保持模型、阈值和主标签不变。",
        ]
    )
    (output / "domain_scale_sensitivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(output_dir: Path, workers: int, overwrite: bool) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"输出目录已有内容：{output_dir}；如需重跑请使用 --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    specs, metadata = _make_specs()
    runs, metrics = _run_all(specs, workers)
    reflection_summary = _summary_table(
        runs,
        [
            "initial_reflection_coordinate_rate",
            "post_initial_reflection_coordinate_rate",
            "reflection_coordinate_rate",
            "initial_reflection_vector_rate",
            "post_initial_reflection_vector_rate",
            "reflection_vector_rate",
        ],
        ["suite", "dimension"],
    )
    coverage_summary = _summary_table(
        runs,
        [
            "raw_pairwise_distance",
            "unit_pairwise_distance",
            "raw_pairwise_distance_per_sqrt_dimension",
            "unit_pairwise_distance_per_sqrt_dimension",
            "initial_axis_coverage_mean",
            "initial_axis_coverage_median",
        ],
        ["suite", "dimension"],
    )
    distance_summary = _summary_table(
        metrics,
        [
            "raw_pairwise_distance",
            "unit_pairwise_distance",
            "raw_pairwise_distance_per_sqrt_dimension",
            "unit_pairwise_distance_per_sqrt_dimension",
            "axis_coverage_mean",
        ],
        ["suite", "dimension", "milestone", "phase"],
    )
    decomposition = _build_scale_decomposition(metrics)
    attribution_summary = _build_scale_attribution_summary(decomposition)

    runs.to_csv(output_dir / "reflection_replay_runs.csv", index=False)
    metrics.to_csv(output_dir / "distance_milestones.csv", index=False)
    reflection_summary.to_csv(output_dir / "reflection_summary.csv", index=False)
    coverage_summary.to_csv(output_dir / "initial_coverage_summary.csv", index=False)
    distance_summary.to_csv(output_dir / "distance_milestone_summary.csv", index=False)
    decomposition.to_csv(output_dir / "scale_decomposition.csv", index=False)
    attribution_summary.to_csv(output_dir / "scale_attribution_summary.csv", index=False)
    metadata.update(
        {
            "analysis": "domain_scale_sensitivity",
            "status": "ok",
            "run_count": int(len(runs)),
            "metric_row_count": int(len(metrics)),
            "milestones": list(MILESTONES),
            "domain_width_ratio": DOMAIN_WIDTH_RATIO,
            "decision_model_used": False,
            "decision_threshold_used": False,
            "utility_labels_used": False,
            "feature_input_columns_used": [],
            "reflection_definition": "out_of_bound_proposed_coordinate_before_existing_reflect_operator",
            "sources": {
                "bbob_config": str(BBOB_CONFIG_PATH),
                "cec_config": str(CEC_CONFIG_PATH),
                "benchmark_factory": str(REPO / "benchmarks/factory.py"),
                "optimizer_state": str(REPO / "optimizers/state.py"),
            },
        }
    )
    (output_dir / "domain_scale_sensitivity_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    _set_plot_style()
    _figure_initial_diagnostics(runs, output_dir)
    _figure_distance_over_budget(metrics, output_dir)
    _figure_reflection_rates(runs, output_dir)
    _figure_scale_decomposition(decomposition, output_dir)
    _write_report(
        output_dir,
        metadata,
        runs,
        metrics,
        reflection_summary,
        coverage_summary,
        decomposition,
        attribution_summary,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    result = run_analysis(args.output_dir, args.workers, args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
