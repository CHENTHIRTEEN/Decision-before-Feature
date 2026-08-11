from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import exp, log, sqrt
from time import perf_counter
from typing import TypeAlias

import numpy as np

from benchmarks.core import Problem
from optimizers.seeding import make_event_rng, make_rng
from optimizers.settings import OptimizerSettings


NATIVE_STREAMS = {
    "de": 101,
    "pso": 202,
    "cmaes": 303,
    "shade": 404,
}

TRANSFER_STREAMS = {
    "de": 1101,
    "pso": 1202,
    "cmaes": 1303,
    "shade": 1404,
}

DE_MUTATION_FACTOR = 0.5
DE_CROSSOVER_RATE = 0.9
PSO_INERTIA = 0.72
PSO_COGNITIVE = 1.49
PSO_SOCIAL = 1.49
PSO_MAX_VELOCITY_RATE = 0.2
SHADE_MEMORY_SIZE = 5
NO_QUERY_TRANSFER_EVENT = 1
QUERY_TRANSFER_EVENT = 2


@dataclass
class DEState:
    population: np.ndarray
    fitness: np.ndarray
    generation: int
    rng_state: dict
    evaluations: int
    best_fitness: float
    best_position: np.ndarray
    mutation_factor: float
    crossover_rate: float
    pending_population: np.ndarray | None = None
    pending_fitness: np.ndarray | None = None
    pending_index: int = 0

    @property
    def algorithm(self) -> str:
        return "de"


@dataclass
class PSOState:
    positions: np.ndarray
    fitness: np.ndarray
    velocities: np.ndarray
    personal_bests: np.ndarray
    personal_best_fitness: np.ndarray
    global_best: np.ndarray
    global_best_fitness: float
    generation: int
    rng_state: dict
    evaluations: int
    best_fitness: float
    best_position: np.ndarray
    inertia: float
    cognitive: float
    social: float
    max_velocity_rate: float
    pending_positions: np.ndarray | None = None
    pending_velocities: np.ndarray | None = None
    pending_fitness: np.ndarray | None = None
    pending_index: int = 0

    @property
    def algorithm(self) -> str:
        return "pso"

    @property
    def population(self) -> np.ndarray:
        return self.positions


@dataclass
class CMAESStrategyState:
    weights: np.ndarray
    mu_effective: float
    c_c: float
    c_sigma: float
    c_1: float
    c_mu: float
    damping: float
    chi_n: float
    eigenvectors: np.ndarray
    axis_scales: np.ndarray
    inverse_sqrt_covariance: np.ndarray
    pending_population: np.ndarray | None = None
    pending_fitness: np.ndarray | None = None
    pending_index: int = 0


@dataclass
class CMAESState:
    population: np.ndarray
    fitness: np.ndarray
    mean: np.ndarray
    covariance_matrix: np.ndarray
    sigma: float
    evolution_path_c: np.ndarray
    evolution_path_sigma: np.ndarray
    strategy_state: CMAESStrategyState
    generation: int
    rng_state: dict
    evaluations: int
    best_fitness: float
    best_position: np.ndarray

    @property
    def algorithm(self) -> str:
        return "cmaes"


@dataclass
class SHADEState:
    population: np.ndarray
    fitness: np.ndarray
    memory_f: np.ndarray
    memory_cr: np.ndarray
    archive: np.ndarray
    memory_index: int
    generation: int
    rng_state: dict
    evaluations: int
    best_fitness: float
    best_position: np.ndarray
    pending_population: np.ndarray | None = None
    pending_fitness: np.ndarray | None = None
    pending_f: np.ndarray | None = None
    pending_cr: np.ndarray | None = None
    pending_index: int = 0

    @property
    def algorithm(self) -> str:
        return "shade"


OptimizerState: TypeAlias = DEState | PSOState | CMAESState | SHADEState


@dataclass(frozen=True)
class StateAdvanceResult:
    state: OptimizerState
    evaluations: int
    runtime_seconds: float

    @property
    def best_fitness(self) -> float:
        return float(self.state.best_fitness)

    @property
    def population(self) -> np.ndarray:
        return np.asarray(self.state.population, dtype=float)

    @property
    def fitness(self) -> np.ndarray:
        return np.asarray(self.state.fitness, dtype=float)


def initialize_optimizer_state(
    *,
    algorithm: str,
    problem: Problem,
    seed: int,
    settings: OptimizerSettings,
) -> OptimizerState:
    """Initialize an optimizer and evaluate its first native population."""
    population_size = int(settings.population_size)
    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    key = _algorithm_key(algorithm)
    rng = make_rng(seed, NATIVE_STREAMS[key])
    if key == "cmaes":
        state = _empty_cmaes_state(problem, population_size, rng)
        _advance_cmaes(state, problem, population_size)
        return state

    population = rng.uniform(
        problem.lower_bounds,
        problem.upper_bounds,
        size=(population_size, problem.dimension),
    )
    fitness = np.asarray(problem.evaluate(population), dtype=float)
    best_index = int(np.argmin(fitness))
    common = {
        "fitness": fitness.copy(),
        "generation": 1,
        "rng_state": _rng_state(rng),
        "evaluations": population_size,
        "best_fitness": float(fitness[best_index]),
        "best_position": population[best_index].copy(),
    }
    if key == "de":
        return DEState(
            population=population.copy(),
            mutation_factor=DE_MUTATION_FACTOR,
            crossover_rate=DE_CROSSOVER_RATE,
            **common,
        )
    if key == "pso":
        span = problem.upper_bounds - problem.lower_bounds
        velocities = rng.uniform(
            -PSO_MAX_VELOCITY_RATE * span,
            PSO_MAX_VELOCITY_RATE * span,
            size=population.shape,
        )
        common["rng_state"] = _rng_state(rng)
        return PSOState(
            positions=population.copy(),
            velocities=velocities,
            personal_bests=population.copy(),
            personal_best_fitness=fitness.copy(),
            global_best=population[best_index].copy(),
            global_best_fitness=float(fitness[best_index]),
            inertia=PSO_INERTIA,
            cognitive=PSO_COGNITIVE,
            social=PSO_SOCIAL,
            max_velocity_rate=PSO_MAX_VELOCITY_RATE,
            **common,
        )
    return SHADEState(
        population=population.copy(),
        memory_f=np.full(SHADE_MEMORY_SIZE, 0.5, dtype=float),
        memory_cr=np.full(SHADE_MEMORY_SIZE, 0.5, dtype=float),
        archive=np.empty((0, problem.dimension), dtype=float),
        memory_index=0,
        **common,
    )


def initialize_transferred_optimizer_state(
    *,
    algorithm: str,
    source_state: OptimizerState,
    problem: Problem,
    seed: int,
    function: int,
    instance: int,
    event: int,
) -> OptimizerState:
    """Initialize a different optimizer once from an algorithm-independent checkpoint state."""
    key = _algorithm_key(algorithm)
    if key == source_state.algorithm:
        raise ValueError("same-algorithm paths must clone and continue the native state")
    population = np.asarray(source_state.population, dtype=float).copy()
    fitness = np.asarray(source_state.fitness, dtype=float).reshape(-1).copy()
    _validate_population(problem, population, fitness)
    rng = make_event_rng(
        seed=seed,
        stream_code=TRANSFER_STREAMS[key],
        function=function,
        instance=instance,
        dimension=problem.dimension,
        generation=int(source_state.generation),
        event=event,
    )
    best_fitness = float(source_state.best_fitness)
    best_position = np.asarray(source_state.best_position, dtype=float).copy()
    if key == "de":
        return DEState(
            population=population,
            fitness=fitness,
            generation=0,
            rng_state=_rng_state(rng),
            evaluations=0,
            best_fitness=best_fitness,
            best_position=best_position,
            mutation_factor=DE_MUTATION_FACTOR,
            crossover_rate=DE_CROSSOVER_RATE,
        )
    if key == "pso":
        span = problem.upper_bounds - problem.lower_bounds
        velocities = rng.uniform(
            -PSO_MAX_VELOCITY_RATE * span,
            PSO_MAX_VELOCITY_RATE * span,
            size=population.shape,
        )
        return PSOState(
            positions=population,
            fitness=fitness,
            velocities=velocities,
            personal_bests=population.copy(),
            personal_best_fitness=fitness.copy(),
            global_best=best_position.copy(),
            global_best_fitness=best_fitness,
            generation=0,
            rng_state=_rng_state(rng),
            evaluations=0,
            best_fitness=best_fitness,
            best_position=best_position,
            inertia=PSO_INERTIA,
            cognitive=PSO_COGNITIVE,
            social=PSO_SOCIAL,
            max_velocity_rate=PSO_MAX_VELOCITY_RATE,
        )
    if key == "cmaes":
        return _transferred_cmaes_state(
            problem=problem,
            population=population,
            fitness=fitness,
            best_fitness=best_fitness,
            best_position=best_position,
            rng=rng,
        )
    return SHADEState(
        population=population,
        fitness=fitness,
        memory_f=np.full(SHADE_MEMORY_SIZE, 0.5, dtype=float),
        memory_cr=np.full(SHADE_MEMORY_SIZE, 0.5, dtype=float),
        archive=np.empty((0, problem.dimension), dtype=float),
        memory_index=0,
        generation=0,
        rng_state=_rng_state(rng),
        evaluations=0,
        best_fitness=best_fitness,
        best_position=best_position,
    )


def clone_optimizer_state(state: OptimizerState) -> OptimizerState:
    """Create an independent checkpoint, including the exact RNG state."""
    return deepcopy(state)


def advance_optimizer_state(
    *,
    state: OptimizerState,
    problem: Problem,
    fe_budget: int,
) -> StateAdvanceResult:
    """Advance an existing native optimizer state by exactly ``fe_budget`` evaluations."""
    if fe_budget < 0:
        raise ValueError("fe_budget must be non-negative")
    _validate_population(problem, state.population, state.fitness)
    started = perf_counter()
    if isinstance(state, DEState):
        evaluations = _advance_de(state, problem, fe_budget)
    elif isinstance(state, PSOState):
        evaluations = _advance_pso(state, problem, fe_budget)
    elif isinstance(state, CMAESState):
        evaluations = _advance_cmaes(state, problem, fe_budget)
    elif isinstance(state, SHADEState):
        evaluations = _advance_shade(state, problem, fe_budget)
    else:
        raise TypeError(f"unsupported optimizer state: {type(state).__name__}")
    if evaluations != fe_budget:
        raise RuntimeError(f"optimizer advanced by {evaluations} evaluations, expected {fe_budget}")
    return StateAdvanceResult(state=state, evaluations=evaluations, runtime_seconds=perf_counter() - started)


def _advance_de(state: DEState, problem: Problem, fe_budget: int) -> int:
    completed = 0
    while completed < fe_budget:
        if state.pending_population is None:
            _start_de_generation(state, problem)
        remaining = len(state.population) - state.pending_index
        batch = min(remaining, fe_budget - completed)
        start = state.pending_index
        stop = start + batch
        values = problem.evaluate(state.pending_population[start:stop])
        state.pending_fitness[start:stop] = values
        for offset, value in enumerate(values):
            _update_best(state, float(value), state.pending_population[start + offset])
        state.pending_index = stop
        state.evaluations += batch
        completed += batch
        if state.pending_index == len(state.population):
            improved = state.pending_fitness < state.fitness
            state.population[improved] = state.pending_population[improved]
            state.fitness[improved] = state.pending_fitness[improved]
            state.pending_population = None
            state.pending_fitness = None
            state.pending_index = 0
            state.generation += 1
    return completed


def _start_de_generation(state: DEState, problem: Problem) -> None:
    rng = _restore_rng(state.rng_state)
    source = state.population.copy()
    pop_size, dimension = source.shape
    trials = np.empty_like(source)
    indices = np.arange(pop_size)
    for i in range(pop_size):
        choices = np.delete(indices, i)
        r1, r2, r3 = rng.choice(choices, size=3, replace=False)
        mutant = source[r1] + state.mutation_factor * (source[r2] - source[r3])
        mutant = np.clip(mutant, problem.lower_bounds, problem.upper_bounds)
        mask = rng.random(dimension) < state.crossover_rate
        mask[int(rng.integers(dimension))] = True
        trials[i] = np.where(mask, mutant, source[i])
    state.pending_population = trials
    state.pending_fitness = np.full(pop_size, np.nan, dtype=float)
    state.pending_index = 0
    state.rng_state = _rng_state(rng)


def _advance_pso(state: PSOState, problem: Problem, fe_budget: int) -> int:
    completed = 0
    while completed < fe_budget:
        if state.pending_positions is None:
            _start_pso_generation(state, problem)
        remaining = len(state.positions) - state.pending_index
        batch = min(remaining, fe_budget - completed)
        start = state.pending_index
        stop = start + batch
        values = problem.evaluate(state.pending_positions[start:stop])
        state.pending_fitness[start:stop] = values
        for offset, value in enumerate(values):
            _update_best(state, float(value), state.pending_positions[start + offset])
        state.pending_index = stop
        state.evaluations += batch
        completed += batch
        if state.pending_index == len(state.positions):
            improved = state.pending_fitness < state.personal_best_fitness
            state.personal_bests[improved] = state.pending_positions[improved]
            state.personal_best_fitness[improved] = state.pending_fitness[improved]
            best_index = int(np.argmin(state.personal_best_fitness))
            state.global_best = state.personal_bests[best_index].copy()
            state.global_best_fitness = float(state.personal_best_fitness[best_index])
            state.positions = state.pending_positions
            state.velocities = state.pending_velocities
            state.fitness = state.pending_fitness
            state.pending_positions = None
            state.pending_velocities = None
            state.pending_fitness = None
            state.pending_index = 0
            state.generation += 1
    return completed


def _start_pso_generation(state: PSOState, problem: Problem) -> None:
    rng = _restore_rng(state.rng_state)
    shape = state.positions.shape
    r1 = rng.random(shape)
    r2 = rng.random(shape)
    velocities = (
        state.inertia * state.velocities
        + state.cognitive * r1 * (state.personal_bests - state.positions)
        + state.social * r2 * (state.global_best - state.positions)
    )
    maximum = state.max_velocity_rate * (problem.upper_bounds - problem.lower_bounds)
    velocities = np.clip(velocities, -maximum, maximum)
    positions = np.clip(state.positions + velocities, problem.lower_bounds, problem.upper_bounds)
    state.pending_positions = positions
    state.pending_velocities = velocities
    state.pending_fitness = np.full(len(positions), np.nan, dtype=float)
    state.pending_index = 0
    state.rng_state = _rng_state(rng)


def _advance_cmaes(state: CMAESState, problem: Problem, fe_budget: int) -> int:
    completed = 0
    strategy = state.strategy_state
    while completed < fe_budget:
        if strategy.pending_population is None:
            _start_cmaes_generation(state, problem)
        remaining = len(strategy.pending_population) - strategy.pending_index
        batch = min(remaining, fe_budget - completed)
        start = strategy.pending_index
        stop = start + batch
        values = problem.evaluate(strategy.pending_population[start:stop])
        strategy.pending_fitness[start:stop] = values
        for offset, value in enumerate(values):
            _update_best(state, float(value), strategy.pending_population[start + offset])
        strategy.pending_index = stop
        state.evaluations += batch
        completed += batch
        if strategy.pending_index == len(strategy.pending_population):
            _finish_cmaes_generation(state)
    return completed


def _start_cmaes_generation(state: CMAESState, problem: Problem) -> None:
    strategy = state.strategy_state
    rng = _restore_rng(state.rng_state)
    z = rng.standard_normal((len(state.population), problem.dimension))
    y = (z * strategy.axis_scales) @ strategy.eigenvectors.T
    population = state.mean + state.sigma * y
    population = np.clip(population, problem.lower_bounds, problem.upper_bounds)
    strategy.pending_population = population
    strategy.pending_fitness = np.full(len(population), np.nan, dtype=float)
    strategy.pending_index = 0
    state.rng_state = _rng_state(rng)


def _finish_cmaes_generation(state: CMAESState) -> None:
    strategy = state.strategy_state
    population = strategy.pending_population
    fitness = strategy.pending_fitness
    order = np.argsort(fitness)
    mu = len(strategy.weights)
    selected = population[order[:mu]]
    old_mean = state.mean.copy()
    state.mean = np.sum(strategy.weights[:, None] * selected, axis=0)
    selected_steps = (selected - old_mean) / max(state.sigma, np.finfo(float).tiny)
    weighted_step = np.sum(strategy.weights[:, None] * selected_steps, axis=0)
    state.evolution_path_sigma = (
        (1.0 - strategy.c_sigma) * state.evolution_path_sigma
        + sqrt(strategy.c_sigma * (2.0 - strategy.c_sigma) * strategy.mu_effective)
        * (strategy.inverse_sqrt_covariance @ weighted_step)
    )
    path_norm = float(np.linalg.norm(state.evolution_path_sigma))
    generation_number = state.generation + 1
    normalization = sqrt(max(1.0 - (1.0 - strategy.c_sigma) ** (2 * generation_number), np.finfo(float).eps))
    h_sigma = float(
        path_norm / normalization / strategy.chi_n
        < 1.4 + 2.0 / (len(state.mean) + 1.0)
    )
    state.evolution_path_c = (
        (1.0 - strategy.c_c) * state.evolution_path_c
        + h_sigma
        * sqrt(strategy.c_c * (2.0 - strategy.c_c) * strategy.mu_effective)
        * weighted_step
    )
    rank_mu = np.zeros_like(state.covariance_matrix)
    for weight, step in zip(strategy.weights, selected_steps, strict=True):
        rank_mu += weight * np.outer(step, step)
    covariance_multiplier = (
        1.0
        - strategy.c_1
        - strategy.c_mu
        + strategy.c_1 * (1.0 - h_sigma) * strategy.c_c * (2.0 - strategy.c_c)
    )
    state.covariance_matrix = (
        covariance_multiplier * state.covariance_matrix
        + strategy.c_1 * np.outer(state.evolution_path_c, state.evolution_path_c)
        + strategy.c_mu * rank_mu
    )
    state.covariance_matrix = 0.5 * (state.covariance_matrix + state.covariance_matrix.T)
    state.sigma *= exp((strategy.c_sigma / strategy.damping) * (path_norm / strategy.chi_n - 1.0))
    state.sigma = float(max(state.sigma, np.finfo(float).tiny))
    _update_cmaes_eigensystem(state)
    state.population = population
    state.fitness = fitness
    strategy.pending_population = None
    strategy.pending_fitness = None
    strategy.pending_index = 0
    state.generation += 1


def _advance_shade(state: SHADEState, problem: Problem, fe_budget: int) -> int:
    completed = 0
    while completed < fe_budget:
        if state.pending_population is None:
            _start_shade_generation(state, problem)
        remaining = len(state.population) - state.pending_index
        batch = min(remaining, fe_budget - completed)
        start = state.pending_index
        stop = start + batch
        values = problem.evaluate(state.pending_population[start:stop])
        state.pending_fitness[start:stop] = values
        for offset, value in enumerate(values):
            _update_best(state, float(value), state.pending_population[start + offset])
        state.pending_index = stop
        state.evaluations += batch
        completed += batch
        if state.pending_index == len(state.population):
            _finish_shade_generation(state)
    return completed


def _start_shade_generation(state: SHADEState, problem: Problem) -> None:
    rng = _restore_rng(state.rng_state)
    population = state.population
    pop_size, dimension = population.shape
    rank = np.argsort(state.fitness)
    union = np.vstack([population, state.archive]) if len(state.archive) else population
    trials = np.empty_like(population)
    f_values = np.empty(pop_size, dtype=float)
    cr_values = np.empty(pop_size, dtype=float)
    indices = np.arange(pop_size)
    for i in range(pop_size):
        slot = int(rng.integers(len(state.memory_f)))
        f_value = _sample_cauchy_positive(rng, state.memory_f[slot])
        cr_value = float(np.clip(rng.normal(state.memory_cr[slot], 0.1), 0.0, 1.0))
        f_values[i] = f_value
        cr_values[i] = cr_value
        pbest_limit = max(2, int(np.ceil(rng.uniform(2.0 / pop_size, 0.2) * pop_size)))
        pbest = population[int(rng.choice(rank[:pbest_limit]))]
        r1 = int(rng.choice(np.delete(indices, i)))
        r2 = int(rng.integers(len(union)))
        while r2 == i:
            r2 = int(rng.integers(len(union)))
        mutant = population[i] + f_value * (pbest - population[i]) + f_value * (population[r1] - union[r2])
        mutant = np.clip(mutant, problem.lower_bounds, problem.upper_bounds)
        mask = rng.random(dimension) < cr_value
        mask[int(rng.integers(dimension))] = True
        trials[i] = np.where(mask, mutant, population[i])
    state.pending_population = trials
    state.pending_fitness = np.full(pop_size, np.nan, dtype=float)
    state.pending_f = f_values
    state.pending_cr = cr_values
    state.pending_index = 0
    state.rng_state = _rng_state(rng)


def _finish_shade_generation(state: SHADEState) -> None:
    improved = state.pending_fitness < state.fitness
    if np.any(improved):
        old_population = state.population[improved].copy()
        improvements = state.fitness[improved] - state.pending_fitness[improved]
        state.archive = np.vstack([state.archive, old_population])
        if len(state.archive) > len(state.population):
            rng = _restore_rng(state.rng_state)
            keep = rng.choice(len(state.archive), size=len(state.population), replace=False)
            state.archive = state.archive[keep]
            state.rng_state = _rng_state(rng)
        successful_f = state.pending_f[improved]
        successful_cr = state.pending_cr[improved]
        weights = improvements / np.sum(improvements)
        denominator = max(float(np.sum(weights * successful_f)), np.finfo(float).eps)
        state.memory_f[state.memory_index] = float(np.sum(weights * successful_f * successful_f) / denominator)
        state.memory_cr[state.memory_index] = float(np.sum(weights * successful_cr))
        state.memory_index = (state.memory_index + 1) % len(state.memory_f)
        state.population[improved] = state.pending_population[improved]
        state.fitness[improved] = state.pending_fitness[improved]
    state.pending_population = None
    state.pending_fitness = None
    state.pending_f = None
    state.pending_cr = None
    state.pending_index = 0
    state.generation += 1


def _empty_cmaes_state(
    problem: Problem,
    population_size: int,
    rng: np.random.Generator,
) -> CMAESState:
    dimension = problem.dimension
    mean = (problem.lower_bounds + problem.upper_bounds) / 2.0
    sigma = 0.3 * float(np.mean(problem.upper_bounds - problem.lower_bounds))
    covariance = np.eye(dimension, dtype=float)
    strategy = _cmaes_strategy_state(dimension, population_size, covariance)
    return CMAESState(
        population=np.empty((population_size, dimension), dtype=float),
        fitness=np.full(population_size, np.inf, dtype=float),
        mean=mean,
        covariance_matrix=covariance,
        sigma=sigma,
        evolution_path_c=np.zeros(dimension, dtype=float),
        evolution_path_sigma=np.zeros(dimension, dtype=float),
        strategy_state=strategy,
        generation=0,
        rng_state=_rng_state(rng),
        evaluations=0,
        best_fitness=float("inf"),
        best_position=mean.copy(),
    )


def _transferred_cmaes_state(
    *,
    problem: Problem,
    population: np.ndarray,
    fitness: np.ndarray,
    best_fitness: float,
    best_position: np.ndarray,
    rng: np.random.Generator,
) -> CMAESState:
    population_size, dimension = population.shape
    weights = _cmaes_weights(population_size)
    order = np.argsort(fitness)
    mean = np.sum(weights[:, None] * population[order[: len(weights)]], axis=0)
    centered = population - np.mean(population, axis=0)
    sample_covariance = centered.T @ centered / max(population_size - 1, 1)
    span_mean = float(np.mean(problem.upper_bounds - problem.lower_bounds))
    sigma = float(max(np.mean(np.std(population, axis=0)), 1e-6 * span_mean))
    covariance = sample_covariance / max(sigma * sigma, np.finfo(float).tiny)
    covariance += np.eye(dimension, dtype=float) * 1e-12
    strategy = _cmaes_strategy_state(dimension, population_size, covariance)
    return CMAESState(
        population=population,
        fitness=fitness,
        mean=mean,
        covariance_matrix=covariance,
        sigma=sigma,
        evolution_path_c=np.zeros(dimension, dtype=float),
        evolution_path_sigma=np.zeros(dimension, dtype=float),
        strategy_state=strategy,
        generation=0,
        rng_state=_rng_state(rng),
        evaluations=0,
        best_fitness=best_fitness,
        best_position=best_position,
    )


def _cmaes_strategy_state(
    dimension: int,
    population_size: int,
    covariance: np.ndarray,
) -> CMAESStrategyState:
    weights = _cmaes_weights(population_size)
    mu_effective = float(1.0 / np.sum(weights * weights))
    c_c = float((4.0 + mu_effective / dimension) / (dimension + 4.0 + 2.0 * mu_effective / dimension))
    c_sigma = float((mu_effective + 2.0) / (dimension + mu_effective + 5.0))
    c_1 = float(2.0 / ((dimension + 1.3) ** 2 + mu_effective))
    c_mu = float(
        min(
            1.0 - c_1,
            2.0 * (mu_effective - 2.0 + 1.0 / mu_effective) / ((dimension + 2.0) ** 2 + mu_effective),
        )
    )
    damping = float(1.0 + 2.0 * max(0.0, sqrt((mu_effective - 1.0) / (dimension + 1.0)) - 1.0) + c_sigma)
    chi_n = float(sqrt(dimension) * (1.0 - 1.0 / (4.0 * dimension) + 1.0 / (21.0 * dimension * dimension)))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-30)
    axis_scales = np.sqrt(eigenvalues)
    inverse = eigenvectors @ np.diag(1.0 / axis_scales) @ eigenvectors.T
    return CMAESStrategyState(
        weights=weights,
        mu_effective=mu_effective,
        c_c=c_c,
        c_sigma=c_sigma,
        c_1=c_1,
        c_mu=c_mu,
        damping=damping,
        chi_n=chi_n,
        eigenvectors=eigenvectors,
        axis_scales=axis_scales,
        inverse_sqrt_covariance=inverse,
    )


def _update_cmaes_eigensystem(state: CMAESState) -> None:
    eigenvalues, eigenvectors = np.linalg.eigh(state.covariance_matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-30)
    state.covariance_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    strategy = state.strategy_state
    strategy.eigenvectors = eigenvectors
    strategy.axis_scales = np.sqrt(eigenvalues)
    strategy.inverse_sqrt_covariance = eigenvectors @ np.diag(1.0 / strategy.axis_scales) @ eigenvectors.T


def _cmaes_weights(population_size: int) -> np.ndarray:
    mu = population_size // 2
    weights = np.asarray([log(mu + 0.5) - log(index + 1.0) for index in range(mu)], dtype=float)
    return weights / np.sum(weights)


def _sample_cauchy_positive(rng: np.random.Generator, center: float) -> float:
    value = center + 0.1 * rng.standard_cauchy()
    attempts = 0
    while value <= 0.0 and attempts < 100:
        value = center + 0.1 * rng.standard_cauchy()
        attempts += 1
    return float(np.clip(value if value > 0.0 else center, 1e-8, 1.0))


def _update_best(state: OptimizerState, value: float, position: np.ndarray) -> None:
    if value < state.best_fitness:
        state.best_fitness = float(value)
        state.best_position = np.asarray(position, dtype=float).copy()


def _validate_population(problem: Problem, population: np.ndarray, fitness: np.ndarray) -> None:
    pop = np.asarray(population, dtype=float)
    fit = np.asarray(fitness, dtype=float).reshape(-1)
    if pop.ndim != 2 or pop.shape[1] != problem.dimension:
        raise ValueError("population shape must match problem dimension")
    if len(pop) != len(fit):
        raise ValueError("fitness length must match population rows")
    if not np.isfinite(pop).all() or not np.isfinite(fit).all():
        raise ValueError("population and fitness must contain only finite values")


def _algorithm_key(algorithm: str) -> str:
    key = str(algorithm).lower()
    if key not in NATIVE_STREAMS:
        raise ValueError(f"unsupported optimizer: {algorithm}")
    return key


def _rng_state(rng: np.random.Generator) -> dict:
    return deepcopy(rng.bit_generator.state)


def _restore_rng(rng_state: dict) -> np.random.Generator:
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(rng_state)
    return rng
