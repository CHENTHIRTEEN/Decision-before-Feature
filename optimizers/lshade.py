from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from benchmarks.core import Problem
from optimizers.seeding import make_rng
from optimizers.state import _apply_boundary_handling, _sample_cauchy_positive
from optimizers.settings import OptimizerSettings


LSHADE_STREAM_CODE = 505
LSHADE_MEMORY_SIZE = 6
LSHADE_MIN_POPULATION_SIZE = 4
LSHADE_P_MIN = 0.11
LSHADE_P_MAX = 0.20


@dataclass
class LShadeState:
    population: np.ndarray
    fitness: np.ndarray
    archive: np.ndarray
    memory_f: np.ndarray
    memory_cr: np.ndarray
    memory_index: int
    generation: int
    rng_state: dict
    evaluations: int
    max_evaluations: int
    initial_population_size: int
    minimum_population_size: int
    best_fitness: float
    best_position: np.ndarray
    pending_population: np.ndarray | None = None
    pending_fitness: np.ndarray | None = None
    pending_f: np.ndarray | None = None
    pending_cr: np.ndarray | None = None
    pending_index: int = 0

    @property
    def algorithm(self) -> str:
        return "lshade"


def initialize_lshade_state(
    *,
    problem: Problem,
    seed: int,
    settings: OptimizerSettings,
    fe_total: int,
    on_evaluation: Callable[[np.ndarray, float], None] | None = None,
) -> LShadeState:
    """Initialize the additional L-SHADE comparison baseline.

    The main project portfolio remains unchanged. This state uses the standard
    current-to-pbest/1 mutation, success-history memories, external archive,
    and linear population-size reduction with the project's fixed population
    size and boundary handling.
    """
    settings.validate(int(fe_total))
    population_size = int(settings.population_size)
    minimum_population_size = min(population_size, LSHADE_MIN_POPULATION_SIZE)
    rng = make_rng(
        seed,
        LSHADE_STREAM_CODE,
        suite_code=problem.suite_code,
        function=problem.function_number,
        instance=problem.instance_number,
        dimension=problem.dimension,
    )
    population = rng.uniform(
        problem.lower_bounds,
        problem.upper_bounds,
        size=(population_size, problem.dimension),
    )
    fitness = np.asarray(problem.evaluate(population), dtype=float).reshape(-1)
    if on_evaluation is not None:
        for point, value in zip(population, fitness, strict=True):
            on_evaluation(np.asarray(point, dtype=float), float(value))
    best_index = int(np.argmin(fitness))
    return LShadeState(
        population=population,
        fitness=fitness,
        archive=np.empty((0, problem.dimension), dtype=float),
        memory_f=np.full(LSHADE_MEMORY_SIZE, 0.5, dtype=float),
        memory_cr=np.full(LSHADE_MEMORY_SIZE, 0.5, dtype=float),
        memory_index=0,
        generation=1,
        rng_state=deepcopy(rng.bit_generator.state),
        evaluations=population_size,
        max_evaluations=int(fe_total),
        initial_population_size=population_size,
        minimum_population_size=minimum_population_size,
        best_fitness=float(fitness[best_index]),
        best_position=population[best_index].copy(),
    )


def advance_lshade_state(
    *,
    state: LShadeState,
    problem: Problem,
    fe_budget: int,
    on_native_update: Callable[[LShadeState], None] | None = None,
    on_evaluation: Callable[[np.ndarray, float], None] | None = None,
) -> int:
    """Advance L-SHADE by exactly ``fe_budget`` objective evaluations."""
    if int(fe_budget) < 0:
        raise ValueError("fe_budget must be non-negative")
    if int(state.evaluations) + int(fe_budget) > int(state.max_evaluations):
        raise ValueError("L-SHADE advance exceeds its declared FE_total")
    completed = 0
    while completed < int(fe_budget):
        if state.pending_population is None:
            _start_generation(state, problem)
        remaining = len(state.population) - int(state.pending_index)
        batch = min(remaining, int(fe_budget) - completed)
        start = int(state.pending_index)
        stop = start + batch
        values = np.asarray(
            problem.evaluate(state.pending_population[start:stop]), dtype=float
        ).reshape(-1)
        state.pending_fitness[start:stop] = values
        for offset, value in enumerate(values):
            position = state.pending_population[start + offset]
            if float(value) < float(state.best_fitness):
                state.best_fitness = float(value)
                state.best_position = np.asarray(position, dtype=float).copy()
            if on_evaluation is not None:
                on_evaluation(np.asarray(position, dtype=float), float(value))
        state.pending_index = stop
        state.evaluations += batch
        completed += batch
        if int(state.pending_index) == len(state.population):
            _finish_generation(state, problem)
            if on_native_update is not None:
                on_native_update(state)
    return completed


def _start_generation(state: LShadeState, problem: Problem) -> None:
    rng = _restore_rng(state.rng_state)
    population = state.population
    population_size, dimension = population.shape
    rank = np.argsort(state.fitness, kind="mergesort")
    union = (
        np.vstack([population, state.archive])
        if len(state.archive)
        else population
    )
    trials = np.empty_like(population)
    f_values = np.empty(population_size, dtype=float)
    cr_values = np.empty(population_size, dtype=float)

    for index in range(population_size):
        memory_slot = int(rng.integers(len(state.memory_f)))
        f_value = _sample_cauchy_positive(rng, state.memory_f[memory_slot])
        cr_value = float(
            np.clip(rng.normal(state.memory_cr[memory_slot], 0.1), 0.0, 1.0)
        )
        p_fraction = float(rng.uniform(LSHADE_P_MIN, LSHADE_P_MAX))
        pbest_count = max(2, int(np.ceil(p_fraction * population_size)))
        pbest = population[int(rng.choice(rank[:pbest_count]))]
        r1 = _sample_population_donor(rng, population_size, index)
        r2 = _sample_union_donor(rng, len(union), population_size, index, r1)
        mutant = population[index] + f_value * (pbest - population[index])
        mutant = mutant + f_value * (population[r1] - union[r2])
        mask = rng.random(dimension) < cr_value
        mask[int(rng.integers(dimension))] = True
        trial = np.where(mask, mutant, population[index])
        trials[index] = _apply_boundary_handling(
            trial,
            problem.lower_bounds,
            problem.upper_bounds,
            boundary_handling=getattr(problem, "boundary_handling", "clip"),
        )
        f_values[index] = f_value
        cr_values[index] = cr_value

    state.pending_population = trials
    state.pending_fitness = np.full(population_size, np.nan, dtype=float)
    state.pending_f = f_values
    state.pending_cr = cr_values
    state.pending_index = 0
    state.rng_state = deepcopy(rng.bit_generator.state)


def _finish_generation(state: LShadeState, problem: Problem) -> None:
    improved = state.pending_fitness < state.fitness
    if np.any(improved):
        replaced = state.population[improved].copy()
        state.archive = np.vstack([state.archive, replaced])
        successful_f = state.pending_f[improved]
        successful_cr = state.pending_cr[improved]
        improvements = state.fitness[improved] - state.pending_fitness[improved]
        state.population[improved] = state.pending_population[improved]
        state.fitness[improved] = state.pending_fitness[improved]
        _update_success_history(state, successful_f, successful_cr, improvements)

    target_size = int(
        round(
            state.initial_population_size
            + (state.minimum_population_size - state.initial_population_size)
            * float(state.evaluations)
            / float(state.max_evaluations)
        )
    )
    target_size = max(
        state.minimum_population_size,
        min(state.initial_population_size, target_size),
    )
    if target_size < len(state.population):
        order = np.argsort(state.fitness, kind="mergesort")[:target_size]
        state.population = state.population[order].copy()
        state.fitness = state.fitness[order].copy()

    archive_limit = len(state.population)
    if len(state.archive) > archive_limit:
        rng = _restore_rng(state.rng_state)
        keep = rng.choice(len(state.archive), size=archive_limit, replace=False)
        state.archive = state.archive[np.asarray(keep, dtype=int)].copy()
        state.rng_state = deepcopy(rng.bit_generator.state)

    state.pending_population = None
    state.pending_fitness = None
    state.pending_f = None
    state.pending_cr = None
    state.pending_index = 0
    state.generation += 1


def _update_success_history(
    state: LShadeState,
    successful_f: np.ndarray,
    successful_cr: np.ndarray,
    improvements: np.ndarray,
) -> None:
    total = float(np.sum(improvements))
    if not np.isfinite(total) or total <= 0.0:
        return
    weights = improvements / total
    denominator = max(float(np.sum(weights * successful_f)), np.finfo(float).eps)
    state.memory_f[state.memory_index] = float(
        np.sum(weights * successful_f * successful_f) / denominator
    )
    state.memory_cr[state.memory_index] = float(np.sum(weights * successful_cr))
    state.memory_index = (state.memory_index + 1) % len(state.memory_f)


def _sample_population_donor(
    rng: np.random.Generator,
    population_size: int,
    current_index: int,
) -> int:
    candidates = np.delete(np.arange(population_size, dtype=int), int(current_index))
    return int(rng.choice(candidates))


def _sample_union_donor(
    rng: np.random.Generator,
    union_size: int,
    population_size: int,
    current_index: int,
    first_donor: int,
) -> int:
    candidates = np.arange(union_size, dtype=int)
    while True:
        candidate = int(rng.choice(candidates))
        if candidate == int(current_index):
            continue
        if candidate < population_size and candidate == int(first_donor):
            continue
        return candidate


def _restore_rng(rng_state: dict) -> np.random.Generator:
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(rng_state)
    return rng
