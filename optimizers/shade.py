from __future__ import annotations

import numpy as np

from benchmarks.core import Problem
from optimizers.seeding import make_rng
from optimizers.settings import OptimizerSettings
from trajectory.recorder import TrajectoryRecorder


def _sample_cauchy_positive(rng: np.random.Generator, center: float) -> float:
    value = center + 0.1 * rng.standard_cauchy()
    attempts = 0
    while value <= 0.0 and attempts < 100:
        value = center + 0.1 * rng.standard_cauchy()
        attempts += 1
    return float(np.clip(value if value > 0.0 else center, 1e-8, 1.0))


def run_shade(
    *,
    problem: Problem,
    seed: int,
    fe_total: int,
    settings: OptimizerSettings,
    memory_size: int = 5,
) -> list:
    settings.validate(fe_total)
    if memory_size < 1:
        raise ValueError("memory_size must be positive")

    # DetPy 2.0.0 exposes SHADE, but its constructor requires database setup even
    # when no database connection is requested. This local implementation keeps
    # Phase 1 focused on observable trajectory collection.
    rng = make_rng(seed, 404)
    recorder = TrajectoryRecorder(settings.checkpoint_ratios)
    population_size = settings.population_size
    lower = problem.lower_bounds
    upper = problem.upper_bounds
    dimension = problem.dimension

    population = rng.uniform(lower, upper, size=(population_size, dimension))
    fitness = problem.evaluate(population)
    fe = population_size
    best_fitness = float(np.min(fitness))
    archive = np.empty((0, dimension), dtype=float)
    memory_f = np.full(memory_size, 0.5, dtype=float)
    memory_cr = np.full(memory_size, 0.5, dtype=float)
    memory_index = 0

    recorder.observe(
        problem=problem,
        algorithm="shade",
        seed=seed,
        fe=fe,
        fe_total=fe_total,
        population=population,
        fitness=fitness,
        best_fitness=best_fitness,
    )

    while fe < fe_total:
        trial_population = np.empty_like(population)
        f_values = np.empty(population_size, dtype=float)
        cr_values = np.empty(population_size, dtype=float)
        rank = np.argsort(fitness)
        union = np.vstack([population, archive]) if len(archive) else population

        for i in range(population_size):
            memory_slot = int(rng.integers(memory_size))
            f = _sample_cauchy_positive(rng, memory_f[memory_slot])
            cr = float(np.clip(rng.normal(memory_cr[memory_slot], 0.1), 0.0, 1.0))
            f_values[i] = f
            cr_values[i] = cr

            pbest_limit = max(2, int(np.ceil(rng.uniform(2.0 / population_size, 0.2) * population_size)))
            pbest = population[int(rng.choice(rank[:pbest_limit]))]
            candidates = np.delete(np.arange(population_size), i)
            r1 = int(rng.choice(candidates))
            r2 = int(rng.integers(len(union)))
            while r2 == i:
                r2 = int(rng.integers(len(union)))

            mutant = population[i] + f * (pbest - population[i]) + f * (population[r1] - union[r2])
            mutant = np.clip(mutant, lower, upper)
            crossover_mask = rng.random(dimension) < cr
            crossover_mask[int(rng.integers(dimension))] = True
            trial_population[i] = np.where(crossover_mask, mutant, population[i])

        trial_fitness = problem.evaluate(trial_population)
        fe += population_size
        improved = trial_fitness < fitness
        if np.any(improved):
            old_population = population[improved].copy()
            improvements = fitness[improved] - trial_fitness[improved]
            archive = np.vstack([archive, old_population])
            if len(archive) > population_size:
                keep = rng.choice(len(archive), size=population_size, replace=False)
                archive = archive[keep]

            successful_f = f_values[improved]
            successful_cr = cr_values[improved]
            weights = improvements / np.sum(improvements)
            memory_f[memory_index] = np.sum(weights * successful_f * successful_f) / np.sum(weights * successful_f)
            memory_cr[memory_index] = np.sum(weights * successful_cr)
            memory_index = (memory_index + 1) % memory_size

            population[improved] = trial_population[improved]
            fitness[improved] = trial_fitness[improved]

        best_fitness = min(best_fitness, float(np.min(fitness)))
        recorder.observe(
            problem=problem,
            algorithm="shade",
            seed=seed,
            fe=fe,
            fe_total=fe_total,
            population=population,
            fitness=fitness,
            best_fitness=best_fitness,
        )

    return recorder.records
