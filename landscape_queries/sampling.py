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
) -> dict[str, object]:
    sample_seed = make_query_sample_seed(
        base_seed=base_seed,
        function=function,
        instance=instance,
        dimension=problem.dimension,
        sample_design=sample_design,
    )
    sample_size = sample_design.sample_size(problem.dimension)
    started = perf_counter()
    sampler = qmc.LatinHypercube(d=problem.dimension, seed=sample_seed)
    unit = sampler.random(n=sample_size)
    x = qmc.scale(unit, problem.lower_bounds, problem.upper_bounds)
    y = np.asarray(problem.evaluate(x), dtype=float).reshape(-1)
    runtime = perf_counter() - started
    if x.shape != (sample_size, problem.dimension) or y.shape != (sample_size,):
        raise ValueError("landscape-query sample has an unexpected shape")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("landscape-query sampling produced non-finite X or y")
    return {
        "sample_seed": sample_seed,
        "sample_size": sample_size,
        "FE_query": sample_size,
        "runtime_sampling_evaluation": float(runtime),
        "lower_bounds": np.asarray(problem.lower_bounds, dtype=float).tolist(),
        "upper_bounds": np.asarray(problem.upper_bounds, dtype=float).tolist(),
        "X": x.tolist(),
        "y": y.tolist(),
    }
