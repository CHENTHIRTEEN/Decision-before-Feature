from __future__ import annotations

import cma
import numpy as np

from benchmarks.core import Problem
from optimizers.seeding import derive_seed
from optimizers.settings import OptimizerSettings
from trajectory.recorder import TrajectoryRecorder


def run_cmaes(
    *,
    problem: Problem,
    seed: int,
    fe_total: int,
    settings: OptimizerSettings,
) -> list:
    settings.validate(fe_total)
    recorder = TrajectoryRecorder(settings.checkpoint_ratios)
    lower = problem.lower_bounds
    upper = problem.upper_bounds
    x0 = ((lower + upper) / 2.0).tolist()
    sigma = 0.3 * float(np.mean(upper - lower))
    strategy = cma.CMAEvolutionStrategy(
        x0,
        sigma,
        {
            "bounds": [lower.tolist(), upper.tolist()],
            "popsize": settings.population_size,
            "seed": derive_seed(seed, 303),
            "verbose": -9,
            "verb_log": 0,
        },
    )
    fe = 0
    best_fitness = np.inf

    while fe < fe_total:
        population = np.asarray(strategy.ask(number=settings.population_size), dtype=float)
        population = np.clip(population, lower, upper)
        fitness = problem.evaluate(population)
        strategy.tell(population.tolist(), fitness.tolist())
        fe += settings.population_size
        best_fitness = min(best_fitness, float(np.min(fitness)))
        recorder.observe(
            problem=problem,
            algorithm="cmaes",
            seed=seed,
            fe=fe,
            fe_total=fe_total,
            population=population,
            fitness=fitness,
            best_fitness=best_fitness,
        )

    return recorder.records

