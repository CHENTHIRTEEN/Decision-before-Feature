from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import log10
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd


BOOTSTRAP_REPETITIONS = 10_000
EFFECT_INTERVAL_LEVEL = 0.95
SIMULTANEOUS_SENSITIVITY_ALPHA = 0.05
SIGN_FLIP_EXACT_MAX_FUNCTIONS = 16
SIGN_FLIP_MONTE_CARLO_REPETITIONS = 100_000
SIGN_FLIP_STREAM_CODE = 7301

FunctionSamplingMode = Literal["fixed_functions", "resample_functions"]
FUNCTION_SAMPLING_MODES = ("fixed_functions", "resample_functions")

FUNCTION_COLUMN = "function_id"
PROBLEM_COLUMNS = ("problem_id", "dimension")
RUN_COLUMNS = (
    "split",
    FUNCTION_COLUMN,
    "problem_id",
    "dimension",
    "prefix_algorithm",
    "seed",
)


@dataclass(frozen=True)
class OperationalTolerance:
    endpoint: str
    lower: float
    upper: float
    scale: str


OPERATIONAL_TOLERANCES = {
    "decision_utility": OperationalTolerance("decision_utility", -0.01, 0.01, "utility"),
    "log10_gap": OperationalTolerance("log10_gap", -0.05, 0.05, "log10_gap"),
    "runtime_log10_ratio": OperationalTolerance(
        "runtime_log10_ratio",
        log10(0.95),
        log10(1.05),
        "log10_ratio",
    ),
    "call_rate": OperationalTolerance("call_rate", -0.05, 0.05, "absolute_proportion"),
    "success_rate": OperationalTolerance(
        "success_rate", -0.05, 0.05, "absolute_proportion"
    ),
}


def aggregate_state_values(
    frame: pd.DataFrame,
    *,
    value_column: str,
    expected_dimensions: Sequence[int],
    run_columns: Sequence[str] = RUN_COLUMNS,
) -> dict[str, pd.DataFrame | float | tuple[int, ...]]:
    """Apply state -> run -> problem -> fixed-dimension stratum -> function weighting."""
    required_hierarchy_columns = {FUNCTION_COLUMN, *PROBLEM_COLUMNS}
    if not required_hierarchy_columns.issubset(run_columns):
        raise ValueError(
            "run_columns must retain function_id, problem_id, and dimension for hierarchical aggregation"
        )
    dimensions = _validate_function_dimension_coverage(
        frame,
        expected_dimensions=expected_dimensions,
        artifact="hierarchical aggregation input",
    )
    _require_columns(frame, {*run_columns, value_column})
    values = pd.to_numeric(frame[value_column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{value_column} must contain finite values")

    run = (
        frame.assign(**{value_column: values})
        .groupby(list(run_columns), sort=True, as_index=False, dropna=False)[value_column]
        .mean()
    )
    problem_group = [FUNCTION_COLUMN, *PROBLEM_COLUMNS]
    problem = run.groupby(problem_group, sort=True, as_index=False, dropna=False)[value_column].mean()
    dimension = problem.groupby(
        [FUNCTION_COLUMN, "dimension"],
        sort=True,
        as_index=False,
        dropna=False,
    )[value_column].mean()
    function = dimension.groupby(
        FUNCTION_COLUMN,
        sort=True,
        as_index=False,
        dropna=False,
    )[value_column].mean()
    if function.empty:
        raise ValueError("hierarchical aggregation produced no function values")
    return {
        "run": run,
        "problem": problem,
        "dimension": dimension,
        "function": function,
        "estimate": float(function[value_column].mean()),
        "expected_dimensions": dimensions,
    }


def paired_run_effects(
    frame: pd.DataFrame,
    *,
    policy_column: str,
    value_column: str,
    treatment: str,
    reference: str,
    run_columns: Sequence[str] = RUN_COLUMNS,
    repetition_column: str | None = None,
) -> pd.DataFrame:
    """Return one complete-pair treatment-minus-reference effect per trajectory.

    The explicit ``dropna`` below defines a complete-pair estimate only.  This
    helper must not be used as an attempted-population result: its consumer must
    separately report attempted coverage and apply the prespecified two-direction
    failure sensitivity before drawing a suite-level conclusion.
    """
    required = {*run_columns, policy_column, value_column}
    if repetition_column is not None:
        required.add(repetition_column)
    _require_columns(frame, required)
    subset = frame[frame[policy_column].astype(str).isin({treatment, reference})].copy()
    if repetition_column is not None:
        repetition_keys = [*run_columns, policy_column, repetition_column]
        if subset.duplicated(repetition_keys, keep=False).any():
            raise ValueError("timing input must contain one value per policy, trajectory, and repetition")
        subset = (
            subset.groupby([*run_columns, policy_column], sort=True, as_index=False, dropna=False)[value_column]
            .median()
        )
    duplicates = subset.duplicated([*run_columns, policy_column], keep=False)
    if duplicates.any():
        raise ValueError("paired policy input must contain one value per policy and trajectory")
    wide = subset.pivot(index=list(run_columns), columns=policy_column, values=value_column)
    missing = {treatment, reference}.difference(wide.columns.astype(str))
    if missing:
        raise ValueError(f"paired policy input is missing policies: {sorted(missing)}")
    paired = wide[[treatment, reference]].dropna().reset_index()
    paired["paired_effect"] = (
        pd.to_numeric(paired[treatment], errors="coerce")
        - pd.to_numeric(paired[reference], errors="coerce")
    )
    if not np.isfinite(paired["paired_effect"].to_numpy(dtype=float)).all():
        raise ValueError("paired effects must be finite")
    output = paired[[*run_columns, "paired_effect"]].copy()
    output.attrs["estimate_population"] = "complete_pairs_only"
    output.attrs["requires_attempted_coverage_report"] = True
    output.attrs["requires_prespecified_failure_sensitivity"] = True
    return output


def paired_hierarchical_interval(
    run_effects: pd.DataFrame,
    *,
    expected_dimensions: Sequence[int],
    function_sampling_mode: FunctionSamplingMode,
    value_column: str = "paired_effect",
    repetitions: int = BOOTSTRAP_REPETITIONS,
    interval_level: float = EFFECT_INTERVAL_LEVEL,
    seed: int,
    analysis_code: int,
    endpoint_code: int,
) -> dict[str, float | int | str | bool | tuple[int, ...]]:
    """Bootstrap paired effects conditional on the fixed static-problem set.

    Optimizer runs are resampled within every fixed problem.  Functions are
    fixed for the primary finite-set interval and may be resampled only for the
    separately labelled function-composition sensitivity.
    """
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    if not 0.0 < interval_level < 1.0:
        raise ValueError("interval_level must lie in (0, 1)")
    if function_sampling_mode not in FUNCTION_SAMPLING_MODES:
        raise ValueError(
            "function_sampling_mode must be exactly fixed_functions or resample_functions"
        )
    _require_columns(run_effects, {FUNCTION_COLUMN, *PROBLEM_COLUMNS, value_column})
    dimensions = _validate_function_dimension_coverage(
        run_effects,
        expected_dimensions=expected_dimensions,
        artifact="hierarchical interval input",
    )
    values = pd.to_numeric(run_effects[value_column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{value_column} must contain finite run-level effects")
    frame = run_effects.assign(**{value_column: values}).reset_index(drop=True)
    run_key = [column for column in RUN_COLUMNS if column in frame.columns]
    if run_key and frame.duplicated(run_key, keep=False).any():
        raise ValueError("hierarchical interval input must contain one effect per trajectory")
    functions = tuple(sorted(frame[FUNCTION_COLUMN].astype(str).unique()))
    if len(functions) < 2:
        raise ValueError("hierarchical interval requires at least two functions")

    replicates = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [int(seed), int(analysis_code), int(endpoint_code), int(repetition)]
            )
        )
        if function_sampling_mode == "fixed_functions":
            sampled_function_indices = np.arange(len(functions), dtype=int)
        else:
            sampled_function_indices = rng.integers(0, len(functions), size=len(functions))
        function_means: list[float] = []
        for function_index in sampled_function_indices:
            function_frame = frame[frame[FUNCTION_COLUMN].astype(str) == functions[int(function_index)]]
            dimension_means: list[float] = []
            for dimension_value in dimensions:
                dimension_frame = function_frame[
                    function_frame["dimension"].astype(int).eq(int(dimension_value))
                ]
                problem_ids = tuple(
                    sorted(dimension_frame["problem_id"].astype(str).unique())
                )
                problem_means: list[float] = []
                for problem_id in problem_ids:
                    run_values = dimension_frame.loc[
                        dimension_frame["problem_id"].astype(str).eq(problem_id),
                        value_column,
                    ].to_numpy(dtype=float)
                    sampled_runs = run_values[
                        rng.integers(0, len(run_values), size=len(run_values))
                    ]
                    problem_means.append(float(np.mean(sampled_runs)))
                dimension_means.append(float(np.mean(problem_means)))
            function_means.append(float(np.mean(dimension_means)))
        replicates[repetition] = float(np.mean(function_means))

    alpha = 1.0 - interval_level
    lower, upper = np.quantile(replicates, [alpha / 2.0, 1.0 - alpha / 2.0])
    observed = aggregate_state_values(
        frame,
        value_column=value_column,
        expected_dimensions=dimensions,
        run_columns=tuple(column for column in RUN_COLUMNS if column in frame.columns),
    )
    return {
        "estimate": float(observed["estimate"]),
        "lower": float(lower),
        "upper": float(upper),
        "interval_level": float(interval_level),
        "repetitions": int(repetitions),
        "function_sampling_mode": str(function_sampling_mode),
        "function_resampling_performed": bool(function_sampling_mode == "resample_functions"),
        "static_problem_sampling_mode": "fixed_static_problems",
        "optimizer_run_resampling_performed": True,
        "functions": int(len(functions)),
        "dimensions": int(len(dimensions)),
        "expected_dimensions": dimensions,
        "problems": int(frame[[FUNCTION_COLUMN, *PROBLEM_COLUMNS]].drop_duplicates().shape[0]),
        "runs": int(len(frame)),
    }


def paired_ert_strata(
    frame: pd.DataFrame,
    *,
    policy_column: str,
    ert_contribution_column: str,
    success_column: str,
    treatment: str,
    reference: str,
    expected_dimensions: Sequence[int],
    run_columns: Sequence[str] = RUN_COLUMNS,
) -> pd.DataFrame:
    """Return absolute ERT and the paired log-ratio for each function x dimension.

    ``success_column`` is a run-level binary target-hit indicator, not an
    endpoint-completion rate or an already aggregated success probability.
    Static problems are weighted equally within a stratum.  A problem's numerator
    and success mass are first averaged over its paired optimizer runs, after which
    the problem means are averaged.  ERT is their ratio, not an average of run-level
    ratios.  Zero-success strata remain explicit: a one-sided zero gives an infinite
    ERT contrast and a two-sided zero gives an undefined contrast.
    """
    paired = _paired_ert_run_rows(
        frame,
        policy_column=policy_column,
        ert_contribution_column=ert_contribution_column,
        success_column=success_column,
        treatment=treatment,
        reference=reference,
        run_columns=run_columns,
    )
    dimensions = _validate_function_dimension_coverage(
        paired,
        expected_dimensions=expected_dimensions,
        artifact="paired ERT input",
    )
    rows: list[dict[str, float | int | str]] = []
    for function in sorted(paired[FUNCTION_COLUMN].astype(str).unique()):
        function_frame = paired[paired[FUNCTION_COLUMN].astype(str).eq(function)]
        for dimension in dimensions:
            stratum = function_frame[
                function_frame["dimension"].astype(int).eq(int(dimension))
            ]
            components = _ert_stratum_components(stratum, rng=None)
            contrast, status = _ert_log10_ratio(
                treatment_numerator=components["treatment_numerator"],
                treatment_success=components["treatment_success"],
                reference_numerator=components["reference_numerator"],
                reference_success=components["reference_success"],
            )
            rows.append(
                {
                    FUNCTION_COLUMN: function,
                    "dimension": int(dimension),
                    "static_problems": int(stratum["problem_id"].astype(str).nunique()),
                    "paired_runs": int(len(stratum)),
                    "treatment_ert_numerator_FE": float(
                        components["treatment_numerator"]
                    ),
                    "treatment_success_mass": float(components["treatment_success"]),
                    "treatment_ERT_FE": _ert_from_components(
                        components["treatment_numerator"],
                        components["treatment_success"],
                    ),
                    "reference_ert_numerator_FE": float(
                        components["reference_numerator"]
                    ),
                    "reference_success_mass": float(components["reference_success"]),
                    "reference_ERT_FE": _ert_from_components(
                        components["reference_numerator"],
                        components["reference_success"],
                    ),
                    "log10_ERT_ratio_treatment_vs_reference": contrast,
                    "zero_success_status": status,
                }
            )
    return pd.DataFrame(rows).sort_values(
        [FUNCTION_COLUMN, "dimension"], kind="mergesort"
    ).reset_index(drop=True)


def paired_hierarchical_ert_log10_ratio_interval(
    frame: pd.DataFrame,
    *,
    policy_column: str,
    ert_contribution_column: str,
    success_column: str,
    treatment: str,
    reference: str,
    expected_dimensions: Sequence[int],
    function_sampling_mode: FunctionSamplingMode,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    interval_level: float = EFFECT_INTERVAL_LEVEL,
    seed: int,
    analysis_code: int,
    endpoint_code: int,
    run_columns: Sequence[str] = RUN_COLUMNS,
) -> dict[str, float | int | str | bool | tuple[int, ...]]:
    """Bootstrap the paired finite-set mean log10 ERT ratio.

    ``success_column`` must contain one binary target-hit value per policy and
    trajectory.  Aggregated rates are not valid inputs.
    Every replicate keeps the finite static-problem set fixed and jointly
    resamples paired optimizer runs within each problem.  ERT is recomputed
    separately in every function x dimension stratum before the
    log10 treatment/reference ratio is averaged equally over fixed dimensions and
    then functions.  No zero-success stratum or replicate is silently dropped.
    One-sided zero success is retained as +/- infinity; two-sided zero success
    contributes explicit undefined bootstrap mass.  Extended-real interval bounds
    allocate that mass conservatively to both tails.  ``interval_status``
    distinguishes a finite interval, an unbounded interval, and an undefined
    observed contrast; it does not change merely because one rare bootstrap
    replicate has zero hits.
    """
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    if not 0.0 < interval_level < 1.0:
        raise ValueError("interval_level must lie in (0, 1)")
    if function_sampling_mode not in FUNCTION_SAMPLING_MODES:
        raise ValueError(
            "function_sampling_mode must be exactly fixed_functions or resample_functions"
        )
    paired = _paired_ert_run_rows(
        frame,
        policy_column=policy_column,
        ert_contribution_column=ert_contribution_column,
        success_column=success_column,
        treatment=treatment,
        reference=reference,
        run_columns=run_columns,
    )
    dimensions = _validate_function_dimension_coverage(
        paired,
        expected_dimensions=expected_dimensions,
        artifact="paired ERT bootstrap input",
    )
    functions = tuple(sorted(paired[FUNCTION_COLUMN].astype(str).unique()))
    if len(functions) < 2:
        raise ValueError("paired ERT interval requires at least two functions")

    observed_strata = paired_ert_strata(
        frame,
        policy_column=policy_column,
        ert_contribution_column=ert_contribution_column,
        success_column=success_column,
        treatment=treatment,
        reference=reference,
        expected_dimensions=dimensions,
        run_columns=run_columns,
    )
    observed_function_effects = [
        _extended_mean(
            function_rows["log10_ERT_ratio_treatment_vs_reference"].to_numpy(
                dtype=float
            )
        )
        for _, function_rows in observed_strata.groupby(
            FUNCTION_COLUMN, sort=True, dropna=False
        )
    ]
    observed_estimate = _extended_mean(np.asarray(observed_function_effects, dtype=float))
    observed_status_counts = _zero_success_status_counts(observed_strata)

    replicates = np.empty(repetitions, dtype=float)
    bootstrap_status_totals = {
        "finite": 0,
        "treatment_zero_success": 0,
        "reference_zero_success": 0,
        "both_zero_success": 0,
    }
    repetitions_with_zero_success = 0
    repetitions_with_undefined_contrast = 0
    for repetition in range(repetitions):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [int(seed), int(analysis_code), int(endpoint_code), int(repetition)]
            )
        )
        if function_sampling_mode == "fixed_functions":
            sampled_function_indices = np.arange(len(functions), dtype=int)
        else:
            sampled_function_indices = rng.integers(
                0, len(functions), size=len(functions)
            )
        function_effects: list[float] = []
        repetition_has_zero_success = False
        for function_index in sampled_function_indices:
            function = functions[int(function_index)]
            function_frame = paired[
                paired[FUNCTION_COLUMN].astype(str).eq(function)
            ]
            stratum_effects: list[float] = []
            for dimension in dimensions:
                stratum = function_frame[
                    function_frame["dimension"].astype(int).eq(int(dimension))
                ]
                components = _ert_stratum_components(stratum, rng=rng)
                contrast, status = _ert_log10_ratio(
                    treatment_numerator=components["treatment_numerator"],
                    treatment_success=components["treatment_success"],
                    reference_numerator=components["reference_numerator"],
                    reference_success=components["reference_success"],
                )
                bootstrap_status_totals[status] += 1
                repetition_has_zero_success = repetition_has_zero_success or status != "finite"
                stratum_effects.append(contrast)
            function_effects.append(
                _extended_mean(np.asarray(stratum_effects, dtype=float))
            )
        replicates[repetition] = _extended_mean(
            np.asarray(function_effects, dtype=float)
        )
        repetitions_with_zero_success += int(repetition_has_zero_success)
        repetitions_with_undefined_contrast += int(np.isnan(replicates[repetition]))

    lower, upper, bootstrap_undefined_mass = (
        _extended_empirical_interval_with_undefined_mass(
            replicates,
            interval_level=interval_level,
        )
    )
    observed_undefined = bool(np.isnan(observed_estimate))
    bounds_undefined = bool(np.isnan(lower) or np.isnan(upper))
    interval_unbounded = bool(
        not bounds_undefined
        and (
            np.isinf(observed_estimate)
            or np.isinf(lower)
            or np.isinf(upper)
        )
    )
    if observed_undefined:
        interval_status = "undefined_observed_contrast"
    elif interval_unbounded and bootstrap_undefined_mass > 0.0:
        interval_status = "unbounded_with_bootstrap_undefined_mass"
    elif interval_unbounded:
        interval_status = "unbounded"
    elif bootstrap_undefined_mass > 0.0:
        interval_status = "finite_with_bootstrap_undefined_mass"
    else:
        interval_status = "finite"
    interval_established = bool(
        not observed_undefined
        and not bounds_undefined
    )
    return {
        "estimate_log10_ERT_ratio": float(observed_estimate),
        "lower": float(lower),
        "upper": float(upper),
        "interval_level": float(interval_level),
        "interval_established": interval_established,
        "interval_status": interval_status,
        "interval_unbounded": interval_unbounded,
        "repetitions": int(repetitions),
        "function_sampling_mode": str(function_sampling_mode),
        "function_resampling_performed": bool(
            function_sampling_mode == "resample_functions"
        ),
        "static_problem_sampling_mode": "fixed_static_problems",
        "optimizer_run_resampling_performed": True,
        "functions": int(len(functions)),
        "dimensions": int(len(dimensions)),
        "expected_dimensions": dimensions,
        "problems": int(
            paired[[FUNCTION_COLUMN, *PROBLEM_COLUMNS]].drop_duplicates().shape[0]
        ),
        "paired_runs": int(len(paired)),
        "observed_finite_strata": int(observed_status_counts["finite"]),
        "observed_treatment_zero_success_strata": int(
            observed_status_counts["treatment_zero_success"]
        ),
        "observed_reference_zero_success_strata": int(
            observed_status_counts["reference_zero_success"]
        ),
        "observed_both_zero_success_strata": int(
            observed_status_counts["both_zero_success"]
        ),
        "bootstrap_finite_strata": int(bootstrap_status_totals["finite"]),
        "bootstrap_treatment_zero_success_strata": int(
            bootstrap_status_totals["treatment_zero_success"]
        ),
        "bootstrap_reference_zero_success_strata": int(
            bootstrap_status_totals["reference_zero_success"]
        ),
        "bootstrap_both_zero_success_strata": int(
            bootstrap_status_totals["both_zero_success"]
        ),
        "bootstrap_repetitions_with_zero_success": int(
            repetitions_with_zero_success
        ),
        "bootstrap_repetitions_with_undefined_contrast": int(
            repetitions_with_undefined_contrast
        ),
        "bootstrap_defined_repetitions": int(
            repetitions - repetitions_with_undefined_contrast
        ),
        "bootstrap_undefined_mass": float(bootstrap_undefined_mass),
        "bootstrap_undefined_mass_interval_rule": (
            "conservative_two_tail_allocation_on_extended_real_line_v1"
        ),
        "zero_success_rule": (
            "one_policy_zero_success_is_infinite_ert_and_signed_infinite_log10_ratio;"
            "both_policies_zero_success_is_explicit_undefined_mass;"
            "undefined_mass_is_allocated_conservatively_to_both_interval_tails;"
            "no_stratum_or_replicate_is_silently_dropped"
        ),
        "aggregation": (
            "ert_within_function_dimension_then_equal_dimensions_then_equal_functions"
        ),
    }


def operational_tolerance_position_from_interval(
    interval: dict[str, float | int | str | bool | tuple[int, ...]],
    *,
    tolerance: OperationalTolerance,
    family_size: int,
) -> dict[str, float | int | str | bool]:
    required_level = bonferroni_sensitivity_interval_level(family_size)
    if not np.isclose(float(interval["interval_level"]), required_level):
        raise ValueError(
            "operational-tolerance sensitivity requires the Bonferroni simultaneous "
            f"interval level {required_level:.12g} for family_size={family_size}"
        )
    lower = float(interval["lower"])
    upper = float(interval["upper"])
    if lower >= tolerance.lower and upper <= tolerance.upper:
        position = "entire_interval_inside_operational_tolerance"
    elif upper < tolerance.lower:
        position = "entire_interval_below_operational_tolerance"
    elif lower > tolerance.upper:
        position = "entire_interval_above_operational_tolerance"
    else:
        position = "interval_crosses_operational_tolerance_boundary"
    return {
        "endpoint": tolerance.endpoint,
        "scale": tolerance.scale,
        "operational_tolerance_lower": float(tolerance.lower),
        "operational_tolerance_upper": float(tolerance.upper),
        "interval_lower": lower,
        "interval_upper": upper,
        "family_size": int(family_size),
        "bonferroni_simultaneous_interval_level": float(required_level),
        "interval_position_relative_to_operational_tolerance": position,
        "inference_role": "operational_tolerance_sensitivity",
        "supports_confirmatory_equivalence": False,
        "tolerance_basis": "project_prespecified_without_independent_domain_basis",
    }


def paired_operational_tolerance_sensitivity_table(
    run_effects: pd.DataFrame,
    *,
    expected_comparisons: Sequence[str],
    expected_dimensions: Sequence[int],
    function_sampling_mode: FunctionSamplingMode,
    tolerance: OperationalTolerance,
    seed: int,
    analysis_code: int,
    endpoint_code: int,
    comparison_column: str = "comparison",
    value_column: str = "paired_effect",
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> pd.DataFrame:
    """Describe intervals relative to project-specific operational tolerances.

    The actual comparison set is checked before the family size and interval
    level are computed.  All contrasts use the same explicit integer bootstrap
    stream inputs so matched hierarchy layouts receive aligned resamples.
    These outputs are sensitivity descriptions, not equivalence conclusions.
    """
    _require_columns(run_effects, {comparison_column, value_column})
    expected = _validate_comparison_family(
        run_effects,
        comparison_column=comparison_column,
        expected_comparisons=expected_comparisons,
        artifact="operational-tolerance sensitivity input",
    )
    family_size = len(expected)
    interval_level = bonferroni_sensitivity_interval_level(family_size)
    dimensions = _normalize_expected_dimensions(expected_dimensions)

    rows: list[dict[str, float | int | str | bool | tuple[int, ...]]] = []
    for comparison in expected:
        comparison_effects = run_effects[
            run_effects[comparison_column].astype(str).eq(comparison)
        ].copy()
        interval = paired_hierarchical_interval(
            comparison_effects,
            expected_dimensions=dimensions,
            function_sampling_mode=function_sampling_mode,
            value_column=value_column,
            repetitions=repetitions,
            interval_level=interval_level,
            seed=seed,
            analysis_code=analysis_code,
            endpoint_code=endpoint_code,
        )
        tolerance_position = operational_tolerance_position_from_interval(
            interval,
            tolerance=tolerance,
            family_size=family_size,
        )
        rows.append(
            {
                comparison_column: comparison,
                "estimate": float(interval["estimate"]),
                "lower": float(interval["lower"]),
                "upper": float(interval["upper"]),
                "interval_level": float(interval["interval_level"]),
                "repetitions": int(interval["repetitions"]),
                "function_sampling_mode": str(interval["function_sampling_mode"]),
                "static_problem_sampling_mode": str(
                    interval["static_problem_sampling_mode"]
                ),
                "optimizer_run_resampling_performed": bool(
                    interval["optimizer_run_resampling_performed"]
                ),
                "functions": int(interval["functions"]),
                "expected_dimensions": dimensions,
                "problems": int(interval["problems"]),
                "runs": int(interval["runs"]),
                "family_expected_comparisons": "|".join(expected),
                **tolerance_position,
            }
        )
    output = pd.DataFrame(rows)
    position_counts = output[
        "interval_position_relative_to_operational_tolerance"
    ].value_counts()
    for position in (
        "entire_interval_inside_operational_tolerance",
        "entire_interval_below_operational_tolerance",
        "entire_interval_above_operational_tolerance",
        "interval_crosses_operational_tolerance_boundary",
    ):
        output[f"family_{position}_count"] = int(position_counts.get(position, 0))
    return output


def bonferroni_sensitivity_interval_level(family_size: int) -> float:
    """Return 1-alpha/m for descriptive family-wise 95% intervals."""
    if int(family_size) <= 0:
        raise ValueError("sensitivity family_size must be positive")
    return float(
        1.0 - (SIMULTANEOUS_SENSITIVITY_ALPHA / int(family_size))
    )


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(tuple(p_values), dtype=float)
    if (
        values.ndim != 1
        or len(values) == 0
        or not np.isfinite(values).all()
        or np.any((values < 0.0) | (values > 1.0))
    ):
        raise ValueError("Holm adjustment requires finite p-values in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * float(values[index])))
        adjusted[index] = running
    return adjusted


def function_level_sign_flip_p_value(
    run_effects: pd.DataFrame,
    *,
    expected_dimensions: Sequence[int],
    value_column: str = "paired_effect",
    seed: int,
    analysis_code: int,
    endpoint_code: int,
    monte_carlo_repetitions: int = SIGN_FLIP_MONTE_CARLO_REPETITIONS,
) -> dict[str, float | int | str | bool]:
    """Assumption-sensitive sign-flip diagnostic for fixed function effects."""
    function = _function_effects(
        run_effects,
        value_column=value_column,
        expected_dimensions=expected_dimensions,
    )
    effects = function[value_column].to_numpy(dtype=float)
    observed = abs(float(np.mean(effects)))
    function_count = len(effects)
    tolerance = np.finfo(float).eps * max(1.0, observed) * 8.0

    if function_count <= SIGN_FLIP_EXACT_MAX_FUNCTIONS:
        exceedances = 0
        total = 2**function_count
        for signs in product((-1.0, 1.0), repeat=function_count):
            statistic = abs(float(np.mean(effects * np.asarray(signs, dtype=float))))
            exceedances += int(statistic + tolerance >= observed)
        p_value = exceedances / total
        method = "exact_function_level_paired_sign_flip_two_sided"
        repetitions = total
    else:
        if monte_carlo_repetitions <= 0:
            raise ValueError("Monte Carlo sign-flip repetitions must be positive")
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    int(seed),
                    SIGN_FLIP_STREAM_CODE,
                    int(analysis_code),
                    int(endpoint_code),
                ]
            )
        )
        exceedances = 0
        remaining = int(monte_carlo_repetitions)
        chunk_size = 4096
        while remaining:
            current = min(chunk_size, remaining)
            signs = rng.integers(0, 2, size=(current, function_count), dtype=np.int8)
            signs = signs.astype(float) * 2.0 - 1.0
            statistics = np.abs(np.mean(signs * effects[None, :], axis=1))
            exceedances += int(np.count_nonzero(statistics + tolerance >= observed))
            remaining -= current
        p_value = (exceedances + 1.0) / (monte_carlo_repetitions + 1.0)
        method = "monte_carlo_function_level_paired_sign_flip_two_sided"
        repetitions = int(monte_carlo_repetitions)

    return {
        "estimate": float(np.mean(effects)),
        "raw_p_value": float(p_value),
        "functions": int(function_count),
        "method": method,
        "null_exchangeability_assumption": "paired_function_effects_are_sign_exchangeable",
        "sign_flip_repetitions": int(repetitions),
        "inference_role": "assumption_sensitive_auxiliary",
        "supports_function_superpopulation_inference": False,
    }


def holm_family_table(
    frame: pd.DataFrame,
    *,
    comparison_column: str = "comparison",
    p_value_column: str = "raw_p_value",
    expected_comparisons: Sequence[str],
) -> pd.DataFrame:
    """Apply auxiliary Holm adjustment after checking family completeness."""
    _require_columns(frame, {comparison_column, p_value_column})
    expected = _validate_comparison_family(
        frame,
        comparison_column=comparison_column,
        expected_comparisons=expected_comparisons,
        artifact="Holm input",
        require_one_row_per_comparison=True,
    )
    output = frame.copy()
    output["holm_adjusted_p_value"] = holm_adjust(
        pd.to_numeric(output[p_value_column], errors="coerce").to_numpy(dtype=float)
    )
    output["holm_family_size"] = len(expected)
    output["inference_role"] = "assumption_sensitive_auxiliary"
    output["supports_function_superpopulation_inference"] = False
    return output


def _function_effects(
    run_effects: pd.DataFrame,
    *,
    value_column: str,
    expected_dimensions: Sequence[int],
) -> pd.DataFrame:
    _require_columns(run_effects, {FUNCTION_COLUMN, *PROBLEM_COLUMNS, value_column})
    _validate_function_dimension_coverage(
        run_effects,
        expected_dimensions=expected_dimensions,
        artifact="function-level test input",
    )
    values = pd.to_numeric(run_effects[value_column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{value_column} must contain finite run-level effects")
    frame = run_effects.assign(**{value_column: values})
    run_key = [column for column in RUN_COLUMNS if column in frame.columns]
    if run_key and frame.duplicated(run_key, keep=False).any():
        raise ValueError("function-level test input must contain one effect per trajectory")
    problem = frame.groupby(
        [FUNCTION_COLUMN, *PROBLEM_COLUMNS],
        sort=True,
        as_index=False,
        dropna=False,
    )[value_column].mean()
    dimension = problem.groupby(
        [FUNCTION_COLUMN, "dimension"],
        sort=True,
        as_index=False,
        dropna=False,
    )[value_column].mean()
    function = dimension.groupby(
        FUNCTION_COLUMN,
        sort=True,
        as_index=False,
        dropna=False,
    )[value_column].mean()
    if len(function) < 2:
        raise ValueError("function-level test requires at least two functions")
    return function


def _paired_ert_run_rows(
    frame: pd.DataFrame,
    *,
    policy_column: str,
    ert_contribution_column: str,
    success_column: str,
    treatment: str,
    reference: str,
    run_columns: Sequence[str],
) -> pd.DataFrame:
    required_hierarchy_columns = {FUNCTION_COLUMN, *PROBLEM_COLUMNS}
    if not required_hierarchy_columns.issubset(run_columns):
        raise ValueError(
            "run_columns must retain function_id, problem_id, and dimension for paired ERT"
        )
    if str(treatment) == str(reference):
        raise ValueError("paired ERT treatment and reference must differ")
    _require_columns(
        frame,
        {
            *run_columns,
            policy_column,
            ert_contribution_column,
            success_column,
        },
    )
    subset = frame[
        frame[policy_column].astype(str).isin({str(treatment), str(reference)})
    ].copy()
    if subset.empty:
        raise ValueError("paired ERT input contains neither requested policy")
    duplicate_key = [*run_columns, policy_column]
    if subset.duplicated(duplicate_key, keep=False).any():
        raise ValueError("paired ERT input must contain one row per policy and trajectory")

    numerator = pd.to_numeric(subset[ert_contribution_column], errors="coerce")
    success = pd.to_numeric(subset[success_column], errors="coerce")
    if numerator.isna().any() or not np.isfinite(numerator.to_numpy(dtype=float)).all():
        raise ValueError("ERT contributions must be finite")
    if bool((numerator <= 0.0).any()):
        raise ValueError("ERT contributions must be strictly positive")
    if success.isna().any() or not np.isfinite(success.to_numpy(dtype=float)).all():
        raise ValueError("ERT success values must be finite")
    if not success.isin((0.0, 1.0)).all():
        raise ValueError(
            "ERT target-hit values must be binary 0/1 for every run-level policy row"
        )
    subset = subset.assign(
        _ert_numerator=numerator.to_numpy(dtype=float),
        _ert_success=success.to_numpy(dtype=float),
    )

    treatment_rows = subset[
        subset[policy_column].astype(str).eq(str(treatment))
    ][[*run_columns, "_ert_numerator", "_ert_success"]].rename(
        columns={
            "_ert_numerator": "treatment_numerator",
            "_ert_success": "treatment_success",
        }
    )
    reference_rows = subset[
        subset[policy_column].astype(str).eq(str(reference))
    ][[*run_columns, "_ert_numerator", "_ert_success"]].rename(
        columns={
            "_ert_numerator": "reference_numerator",
            "_ert_success": "reference_success",
        }
    )
    paired = treatment_rows.merge(
        reference_rows,
        on=list(run_columns),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not paired["_merge"].astype(str).eq("both").all():
        counts = paired["_merge"].astype(str).value_counts().to_dict()
        raise ValueError(
            "paired ERT input has incomplete treatment/reference trajectories: "
            f"{counts}"
        )
    paired = paired.drop(columns="_merge")
    if paired.empty:
        raise ValueError("paired ERT input produced no complete trajectories")
    return paired.sort_values(list(run_columns), kind="mergesort").reset_index(drop=True)


def _ert_stratum_components(
    stratum: pd.DataFrame,
    *,
    rng: np.random.Generator | None,
) -> dict[str, float]:
    if stratum.empty:
        raise ValueError("ERT function-dimension stratum must not be empty")
    problem_ids = tuple(sorted(stratum["problem_id"].astype(str).unique()))
    problem_components: list[np.ndarray] = []
    component_columns = [
        "treatment_numerator",
        "treatment_success",
        "reference_numerator",
        "reference_success",
    ]
    for problem_index in np.arange(len(problem_ids), dtype=int):
        problem = stratum[
            stratum["problem_id"].astype(str).eq(problem_ids[int(problem_index)])
        ]
        values = problem[component_columns].to_numpy(dtype=float)
        if rng is not None:
            values = values[rng.integers(0, len(values), size=len(values))]
        problem_components.append(np.mean(values, axis=0))
    means = np.mean(np.vstack(problem_components), axis=0)
    return {
        column: float(value)
        for column, value in zip(component_columns, means, strict=True)
    }


def _ert_from_components(numerator: float, success: float) -> float:
    if success == 0.0:
        return float("inf")
    return float(numerator / success)


def _ert_log10_ratio(
    *,
    treatment_numerator: float,
    treatment_success: float,
    reference_numerator: float,
    reference_success: float,
) -> tuple[float, str]:
    treatment_zero = treatment_success == 0.0
    reference_zero = reference_success == 0.0
    if treatment_zero and reference_zero:
        return float("nan"), "both_zero_success"
    if treatment_zero:
        return float("inf"), "treatment_zero_success"
    if reference_zero:
        return float("-inf"), "reference_zero_success"
    treatment_ert = _ert_from_components(treatment_numerator, treatment_success)
    reference_ert = _ert_from_components(reference_numerator, reference_success)
    return float(log10(treatment_ert) - log10(reference_ert)), "finite"


def _extended_mean(values: np.ndarray) -> float:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("extended-real mean requires a non-empty one-dimensional array")
    if np.isnan(values).any():
        return float("nan")
    has_positive_infinity = bool(np.isposinf(values).any())
    has_negative_infinity = bool(np.isneginf(values).any())
    if has_positive_infinity and has_negative_infinity:
        return float("nan")
    if has_positive_infinity:
        return float("inf")
    if has_negative_infinity:
        return float("-inf")
    return float(np.mean(values))


def _extended_empirical_interval_with_undefined_mass(
    values: np.ndarray,
    *,
    interval_level: float,
) -> tuple[float, float, float]:
    """Return conservative extended-real bounds and explicit undefined mass.

    Undefined values arise when both policies have zero successes in at least one
    aggregated contrast.  They have no order on the extended real line, so the
    lower bound assigns all undefined mass below the defined values and the upper
    bound assigns it above them.  This keeps every replicate in the uncertainty
    accounting without making interval existence depend on whether one rare
    undefined replicate happened to be sampled.
    """
    if values.ndim != 1 or len(values) == 0:
        raise ValueError(
            "extended empirical interval requires a non-empty one-dimensional array"
        )
    if not 0.0 < interval_level < 1.0:
        raise ValueError("interval_level must lie in (0, 1)")
    undefined = np.isnan(values)
    undefined_mass = float(np.mean(undefined))
    defined = np.sort(values[~undefined])
    if len(defined) == 0:
        return float("-inf"), float("inf"), undefined_mass

    alpha_tail = (1.0 - interval_level) / 2.0
    defined_mass = 1.0 - undefined_mass

    def empirical_quantile(probability: float) -> float:
        index = int(np.ceil(probability * len(defined))) - 1
        index = min(max(index, 0), len(defined) - 1)
        return float(defined[index])

    if undefined_mass >= alpha_tail:
        lower = float("-inf")
        upper = float("inf")
    else:
        lower_probability = (alpha_tail - undefined_mass) / defined_mass
        upper_probability = (1.0 - alpha_tail) / defined_mass
        lower = empirical_quantile(lower_probability)
        upper = empirical_quantile(upper_probability)
    return lower, upper, undefined_mass


def _zero_success_status_counts(strata: pd.DataFrame) -> dict[str, int]:
    expected = (
        "finite",
        "treatment_zero_success",
        "reference_zero_success",
        "both_zero_success",
    )
    observed = strata["zero_success_status"].astype(str).value_counts().to_dict()
    unexpected = set(observed).difference(expected)
    if unexpected:
        raise ValueError(f"ERT strata contain unsupported zero-success states: {unexpected}")
    return {status: int(observed.get(status, 0)) for status in expected}


def _normalize_expected_dimensions(expected_dimensions: Sequence[int]) -> tuple[int, ...]:
    raw_values = tuple(expected_dimensions)
    if not raw_values:
        raise ValueError("expected_dimensions must be non-empty")
    values: list[int] = []
    for value in raw_values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("expected_dimensions must contain integers, not booleans")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("expected_dimensions must contain integers") from error
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("expected_dimensions must contain finite integers")
        values.append(int(numeric))
    if any(value <= 0 for value in values):
        raise ValueError("expected_dimensions must contain positive integers")
    if len(values) != len(set(values)):
        raise ValueError("expected_dimensions must contain unique dimensions")
    return tuple(sorted(values))


def _validate_function_dimension_coverage(
    frame: pd.DataFrame,
    *,
    expected_dimensions: Sequence[int],
    artifact: str,
) -> tuple[int, ...]:
    _require_columns(frame, {FUNCTION_COLUMN, "dimension"})
    expected = _normalize_expected_dimensions(expected_dimensions)
    if frame.empty:
        raise ValueError(f"{artifact} must contain at least one row")
    if frame[FUNCTION_COLUMN].isna().any():
        raise ValueError(f"{artifact} contains missing function identifiers")
    numeric_dimensions = pd.to_numeric(frame["dimension"], errors="coerce")
    if numeric_dimensions.isna().any() or not np.isfinite(
        numeric_dimensions.to_numpy(dtype=float)
    ).all():
        raise ValueError(f"{artifact} contains non-finite dimensions")
    integer_dimensions = numeric_dimensions.astype(int)
    if not np.array_equal(
        numeric_dimensions.to_numpy(dtype=float),
        integer_dimensions.to_numpy(dtype=float),
    ):
        raise ValueError(f"{artifact} dimensions must be integers")

    family_values = frame[FUNCTION_COLUMN].astype(str)
    mismatches: list[str] = []
    for function in sorted(family_values.unique()):
        observed = tuple(
            sorted(set(integer_dimensions.loc[family_values.eq(function)].astype(int)))
        )
        if observed != expected:
            mismatches.append(f"{function}: observed={observed}")
    if mismatches:
        raise ValueError(
            f"{artifact} requires every function to cover expected_dimensions={expected}; "
            + "; ".join(mismatches)
        )
    return expected


def _validate_comparison_family(
    frame: pd.DataFrame,
    *,
    comparison_column: str,
    expected_comparisons: Sequence[str],
    artifact: str,
    require_one_row_per_comparison: bool = False,
) -> tuple[str, ...]:
    _require_columns(frame, {comparison_column})
    expected = tuple(str(value) for value in expected_comparisons)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected comparisons must be non-empty and unique")
    if frame[comparison_column].isna().any():
        raise ValueError(f"{artifact} contains missing comparison identifiers")
    actual_rows = tuple(frame[comparison_column].astype(str))
    actual = set(actual_rows)
    if actual != set(expected):
        raise ValueError(
            f"{artifact} must cover the complete prespecified comparison family: "
            f"expected={expected}, observed={tuple(sorted(actual))}"
        )
    if require_one_row_per_comparison and len(actual_rows) != len(expected):
        raise ValueError(f"{artifact} must contain each prespecified comparison exactly once")
    return expected


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"hierarchical inference input is missing columns: {sorted(missing)}")
