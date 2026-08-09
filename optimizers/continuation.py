from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cma
import numpy as np

from benchmarks.core import Problem
from optimizers.seeding import make_event_rng
from optimizers.settings import OptimizerSettings


ALGORITHM_STREAMS = {
    "de": 1101,
    "pso": 1202,
    "cmaes": 1303,
    "shade": 1404,
}


@dataclass(frozen=True)
class ContinuationResult:
    best_fitness: float
    runtime_seconds: float
    evaluations: int
    population: np.ndarray | None = None
    fitness: np.ndarray | None = None


def run_population_continuation(
    *,
    algorithm: str,
    problem: Problem,
    seed: int,
    function: int,
    instance: int,
    generation: int,
    event: int,
    fe_budget: int,
    population: np.ndarray,
    fitness: np.ndarray,
    best_fitness: float,
    settings: OptimizerSettings,
) -> ContinuationResult:
    """Continue from checkpoint population state without optimizer-internal state transfer."""
    if fe_budget < 0:
        raise ValueError("fe_budget must be non-negative")
    if fe_budget == 0:
        return ContinuationResult(float(best_fitness), 0.0, 0, np.asarray(population, dtype=float).copy(), np.asarray(fitness, dtype=float).copy())

    pop = np.asarray(population, dtype=float).copy()
    fit = np.asarray(fitness, dtype=float).reshape(-1).copy()
    if pop.ndim != 2 or pop.shape[1] != problem.dimension:
        raise ValueError("population shape must match problem dimension")
    if fit.shape[0] != pop.shape[0]:
        raise ValueError("fitness length must match population rows")

    key = algorithm.lower()
    if key not in ALGORITHM_STREAMS:
        raise ValueError(f"unsupported continuation algorithm: {algorithm}")
    rng = make_event_rng(
        seed=seed,
        stream_code=ALGORITHM_STREAMS[key],
        function=function,
        instance=instance,
        dimension=problem.dimension,
        generation=generation,
        event=event,
    )
    started = perf_counter()
    if key == "de":
        best, evaluations = _continue_de(problem, rng, fe_budget, pop, fit, float(best_fitness))
    elif key == "pso":
        best, evaluations = _continue_pso(problem, rng, fe_budget, pop, fit, float(best_fitness))
    elif key == "cmaes":
        best, evaluations = _continue_cmaes(problem, rng, fe_budget, pop, fit, float(best_fitness), settings)
    else:
        best, evaluations = _continue_shade(problem, rng, fe_budget, pop, fit, float(best_fitness))
    return ContinuationResult(best, perf_counter() - started, evaluations, pop.copy(), fit.copy())


def _continue_de(
    problem: Problem,
    rng: np.random.Generator,
    fe_budget: int,
    population: np.ndarray,
    fitness: np.ndarray,
    best_fitness: float,
) -> tuple[float, int]:
    lower = problem.lower_bounds
    upper = problem.upper_bounds
    pop_size, dimension = population.shape
    evaluations = 0
    while evaluations < fe_budget:
        for i in range(pop_size):
            if evaluations >= fe_budget:
                break
            choices = np.delete(np.arange(pop_size), i)
            r1, r2, r3 = rng.choice(choices, size=3, replace=False)
            mutant = np.clip(population[r1] + 0.5 * (population[r2] - population[r3]), lower, upper)
            mask = rng.random(dimension) < 0.9
            mask[int(rng.integers(dimension))] = True
            trial = np.where(mask, mutant, population[i])
            trial_fitness = float(problem.evaluate(trial)[0])
            evaluations += 1
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                best_fitness = min(best_fitness, trial_fitness)
    return float(best_fitness), evaluations


def _continue_pso(
    problem: Problem,
    rng: np.random.Generator,
    fe_budget: int,
    population: np.ndarray,
    fitness: np.ndarray,
    best_fitness: float,
) -> tuple[float, int]:
    lower = problem.lower_bounds
    upper = problem.upper_bounds
    span = upper - lower
    pop_size, dimension = population.shape
    velocity = rng.normal(0.0, 0.05 * span, size=population.shape)
    personal_best = population.copy()
    personal_best_fitness = fitness.copy()
    best_index = int(np.argmin(fitness))
    global_best = population[best_index].copy()
    evaluations = 0
    while evaluations < fe_budget:
        for i in range(pop_size):
            if evaluations >= fe_budget:
                break
            r1 = rng.random(dimension)
            r2 = rng.random(dimension)
            velocity[i] = 0.72 * velocity[i] + 1.49 * r1 * (personal_best[i] - population[i]) + 1.49 * r2 * (
                global_best - population[i]
            )
            population[i] = np.clip(population[i] + velocity[i], lower, upper)
            value = float(problem.evaluate(population[i])[0])
            evaluations += 1
            fitness[i] = value
            if value < personal_best_fitness[i]:
                personal_best[i] = population[i]
                personal_best_fitness[i] = value
            if value < best_fitness:
                best_fitness = value
                global_best = population[i].copy()
    return float(best_fitness), evaluations


def _continue_cmaes(
    problem: Problem,
    rng: np.random.Generator,
    fe_budget: int,
    population: np.ndarray,
    fitness: np.ndarray,
    best_fitness: float,
    settings: OptimizerSettings,
) -> tuple[float, int]:
    lower = problem.lower_bounds
    upper = problem.upper_bounds
    center = population[int(np.argmin(fitness))].copy()
    spread = np.mean(np.std(population, axis=0))
    sigma = float(np.clip(spread, 1e-6, 0.3 * np.mean(upper - lower)))
    seed_value = int(rng.integers(0, np.iinfo(np.uint32).max))
    strategy = cma.CMAEvolutionStrategy(
        center.tolist(),
        sigma,
        {
            "bounds": [lower.tolist(), upper.tolist()],
            "popsize": settings.population_size,
            "seed": seed_value,
            "verbose": -9,
            "verb_log": 0,
        },
    )
    evaluations = 0
    while evaluations < fe_budget:
        batch = min(settings.population_size, fe_budget - evaluations)
        candidates = np.asarray(strategy.ask(number=batch), dtype=float)
        candidates = np.clip(candidates, lower, upper)
        values = problem.evaluate(candidates)
        strategy.tell(candidates.tolist(), values.tolist())
        population[: len(candidates)] = candidates
        fitness[: len(values)] = values
        evaluations += int(len(values))
        best_fitness = min(best_fitness, float(np.min(values)))
    return float(best_fitness), evaluations


def _sample_cauchy_positive(rng: np.random.Generator, center: float) -> float:
    value = center + 0.1 * rng.standard_cauchy()
    attempts = 0
    while value <= 0.0 and attempts < 100:
        value = center + 0.1 * rng.standard_cauchy()
        attempts += 1
    return float(np.clip(value if value > 0.0 else center, 1e-8, 1.0))


def _continue_shade(
    problem: Problem,
    rng: np.random.Generator,
    fe_budget: int,
    population: np.ndarray,
    fitness: np.ndarray,
    best_fitness: float,
) -> tuple[float, int]:
    lower = problem.lower_bounds
    upper = problem.upper_bounds
    pop_size, dimension = population.shape
    memory_f = np.full(5, 0.5, dtype=float)
    memory_cr = np.full(5, 0.5, dtype=float)
    memory_index = 0
    archive = np.empty((0, dimension), dtype=float)
    evaluations = 0
    while evaluations < fe_budget:
        improved_f = []
        improved_cr = []
        improvements = []
        rank = np.argsort(fitness)
        union = np.vstack([population, archive]) if len(archive) else population
        for i in range(pop_size):
            if evaluations >= fe_budget:
                break
            slot = int(rng.integers(len(memory_f)))
            f = _sample_cauchy_positive(rng, memory_f[slot])
            cr = float(np.clip(rng.normal(memory_cr[slot], 0.1), 0.0, 1.0))
            pbest_limit = max(2, int(np.ceil(rng.uniform(2.0 / pop_size, 0.2) * pop_size)))
            pbest = population[int(rng.choice(rank[:pbest_limit]))]
            candidates = np.delete(np.arange(pop_size), i)
            r1 = int(rng.choice(candidates))
            r2 = int(rng.integers(len(union)))
            mutant = population[i] + f * (pbest - population[i]) + f * (population[r1] - union[r2])
            mutant = np.clip(mutant, lower, upper)
            mask = rng.random(dimension) < cr
            mask[int(rng.integers(dimension))] = True
            trial = np.where(mask, mutant, population[i])
            value = float(problem.evaluate(trial)[0])
            evaluations += 1
            if value < fitness[i]:
                archive = np.vstack([archive, population[i].copy()])
                if len(archive) > pop_size:
                    archive = archive[rng.choice(len(archive), size=pop_size, replace=False)]
                improvements.append(float(fitness[i] - value))
                improved_f.append(f)
                improved_cr.append(cr)
                population[i] = trial
                fitness[i] = value
                best_fitness = min(best_fitness, value)
        if improvements:
            weights = np.asarray(improvements, dtype=float)
            weights = weights / np.sum(weights)
            sf = np.asarray(improved_f, dtype=float)
            scr = np.asarray(improved_cr, dtype=float)
            memory_f[memory_index] = np.sum(weights * sf * sf) / max(np.sum(weights * sf), 1e-12)
            memory_cr[memory_index] = np.sum(weights * scr)
            memory_index = (memory_index + 1) % len(memory_f)
    return float(best_fitness), evaluations
