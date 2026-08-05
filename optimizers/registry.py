from __future__ import annotations

from benchmarks.core import Problem
from optimizers.cmaes import run_cmaes
from optimizers.pymoo_adapters import run_pymoo_optimizer
from optimizers.settings import OptimizerSettings
from optimizers.shade import run_shade


SUPPORTED_ALGORITHMS = ("de", "pso", "cmaes", "shade")


def run_optimizer(
    *,
    algorithm: str,
    problem: Problem,
    seed: int,
    fe_total: int,
    settings: OptimizerSettings,
) -> list:
    key = algorithm.lower()
    if key in {"de", "pso"}:
        return run_pymoo_optimizer(
            algorithm_name=key,
            problem=problem,
            seed=seed,
            fe_total=fe_total,
            settings=settings,
        )
    if key == "cmaes":
        return run_cmaes(problem=problem, seed=seed, fe_total=fe_total, settings=settings)
    if key == "shade":
        return run_shade(problem=problem, seed=seed, fe_total=fe_total, settings=settings)
    raise ValueError(f"unsupported optimizer: {algorithm}")
