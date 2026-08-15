from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.stats import qmc

from benchmarks.core import Problem
from landscape_queries.specs import SampleDesignSpec


QUERY_SAMPLE_STREAM_CODE = 2501


def make_query_sample_seed(
    *,
    base_seed: int,
    function: int,
    instance: int,
    dimension: int,
    sample_design: SampleDesignSpec,
) -> int:
    sequence = np.random.SeedSequence(
        [
            int(base_seed),
            QUERY_SAMPLE_STREAM_CODE,
            int(function),
            int(instance),
            int(dimension),
            int(sample_design.design_code),
        ]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def sample_problem(
    *,
    problem: Problem,
    sample_design: SampleDesignSpec,
    base_seed: int,
    function: int,
    instance: int,
    success_gap_target: float,
    failure_loss_cap: float,
) -> dict[str, object]:
    sample_seed = make_query_sample_seed(
        base_seed=base_seed,
        function=function,
        instance=instance,
        dimension=problem.dimension,
        sample_design=sample_design,
    )
    sample_size = sample_design.sample_size(problem.dimension)
    sampling_started = perf_counter()
    failure_type = ""
    failure_message = ""
    try:
        sampler = qmc.LatinHypercube(d=problem.dimension, seed=sample_seed)
        unit = sampler.random(n=sample_size)
        x_planned = qmc.scale(unit, problem.lower_bounds, problem.upper_bounds)
    except Exception as exc:
        x_planned = np.empty((0, problem.dimension), dtype=float)
        failure_type = type(exc).__name__
        failure_message = str(exc)[:500]
    runtime_sampling = perf_counter() - sampling_started
    evaluation_started = perf_counter()
    evaluated_points: list[np.ndarray] = []
    evaluated_values: list[float] = []
    if not failure_type:
        for point in x_planned:
            try:
                value_array = np.asarray(
                    problem.evaluate(np.asarray(point, dtype=float).reshape(1, -1)),
                    dtype=float,
                ).reshape(-1)
                if value_array.shape != (1,):
                    raise ValueError("landscape-query objective returned an unexpected shape")
                value = float(value_array[0])
                evaluated_points.append(np.asarray(point, dtype=float))
                evaluated_values.append(value)
                if not np.isfinite(value):
                    raise FloatingPointError(
                        "landscape-query objective returned a non-finite value"
                    )
            except Exception as exc:
                failure_type = type(exc).__name__
                failure_message = str(exc)[:500]
                break
    runtime_evaluation = perf_counter() - evaluation_started
    x = np.asarray(evaluated_points, dtype=float).reshape(-1, problem.dimension)
    y = np.asarray(evaluated_values, dtype=float).reshape(-1)
    path_completed = not failure_type and len(y) == sample_size
    if not path_completed and not failure_type:
        failure_type = "IncompleteQuerySample"
        failure_message = "query sampling did not execute its planned evaluations"
    target = float(success_gap_target)
    cap = float(failure_loss_cap)
    if not np.isfinite(target) or target <= 0.0 or not np.isfinite(cap) or cap <= 0.0:
        raise ValueError("success_gap_target must be finite and positive")
    reference = problem.reference_value
    if reference is None:
        failure_type = failure_type or "MissingBenchmarkReference"
        failure_message = failure_message or "query success recording requires a benchmark reference value"
        path_completed = False
        gaps = np.full(len(y), np.inf, dtype=float)
    else:
        gaps = np.where(
            np.isfinite(y),
            np.maximum(y - float(reference), 0.0),
            np.inf,
        )
    hits = np.flatnonzero(gaps <= target)
    first_hit_offset = int(hits[0] + 1) if hits.size else None
    finite_gaps = gaps[np.isfinite(gaps)]
    query_best_gap = float(np.min(finite_gaps)) if finite_gaps.size else cap
    return {
        "sample_seed": sample_seed,
        "sample_size": sample_size,
        "FE_query": int(len(y)),
        "FE_query_planned": sample_size,
        "runtime_query_sampling": float(runtime_sampling),
        "runtime_query_evaluation": float(runtime_evaluation),
        "runtime_sampling_evaluation": float(runtime_sampling + runtime_evaluation),
        "benchmark_reference_value": (
            None if reference is None else float(reference)
        ),
        "success_gap_target": target,
        "query_success": bool(hits.size),
        "query_first_hit_offset": first_hit_offset,
        "query_best_gap": query_best_gap,
        "lower_bounds": np.asarray(problem.lower_bounds, dtype=float).tolist(),
        "upper_bounds": np.asarray(problem.upper_bounds, dtype=float).tolist(),
        "X": x.tolist(),
        "y": y.tolist(),
        "sample_status": "ok" if path_completed else "failed",
        "sample_path_completed": bool(path_completed),
        "sample_planned_FE": int(sample_size),
        "sample_effective_FE": int(len(y)),
        "sample_observed_first_hit_FE": first_hit_offset,
        "sample_target_hit_observed": bool(first_hit_offset is not None),
        "sample_target_hit_before_failure": bool(
            first_hit_offset is not None and not path_completed
        ),
        "sample_endpoint_success": bool(first_hit_offset is not None and path_completed),
        "sample_timed_out": False,
        "sample_failure_type": failure_type,
        "sample_failure_message": failure_message,
        "sample_failure": (
            "" if not failure_type else f"{failure_type}: {failure_message}"[:500]
        ),
    }
