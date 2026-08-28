"""Independent implementations of the ten optimizers used by AS-LGBM RGI.

These implementations are intentionally isolated from the project's four
stateful optimizers.  They reproduce the algorithm pool needed by the
literature baseline and expose the same run-level fields needed for both
final-value and target-FE summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


PAPER_ALGORITHMS = (
    "ABC",
    "ACO",
    "CMA-ES",
    "CSO",
    "DE",
    "FEP",
    "GA",
    "PSO",
    "SA",
    "RAND",
)

_ALGORITHM_ALIASES = {
    "ABC": "ABC",
    "ACO": "ACO",
    "CMAES": "CMA-ES",
    "CMA-ES": "CMA-ES",
    "CSO": "CSO",
    "DE": "DE",
    "FEP": "FEP",
    "GA": "GA",
    "PSO": "PSO",
    "SA": "SA",
    "RAND": "RAND",
}
_FAILURE_VALUE = 1.0e100


@dataclass
class RGIAlgorithmResult:
    algorithm: str
    run_seed: int
    fe_total: int
    effective_fe: int
    best_value: float | None
    final_value: float | None
    first_hit_fe: int | None
    target_hit: bool
    performance_value: float
    run_status: str
    failure_type: str | None = None
    failure_message: str | None = None


class _EvaluationBudget:
    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        *,
        fe_total: int,
        target_value: float | None,
        target_tolerance: float,
    ) -> None:
        self.objective = objective
        self.fe_total = int(fe_total)
        self.target_value = target_value
        self.target_tolerance = float(target_tolerance)
        self.evaluations = 0
        self.best_value = np.inf
        self.final_value: float | None = None
        self.first_hit_fe: int | None = None

    @property
    def remaining(self) -> int:
        return self.fe_total - self.evaluations

    def evaluate(self, point: np.ndarray) -> float:
        if self.evaluations >= self.fe_total:
            raise RuntimeError("RGI evaluation budget exhausted")
        value = float(self.objective(np.asarray(point, dtype=float)))
        return self._record(value)

    def _record(self, value: float) -> float:
        if not np.isfinite(value):
            value = _FAILURE_VALUE
        self.evaluations += 1
        self.final_value = value
        if value < self.best_value:
            self.best_value = value
        if (
            self.target_value is not None
            and self.first_hit_fe is None
            and value <= self.target_value + self.target_tolerance
        ):
            self.first_hit_fe = self.evaluations
        return value

    def evaluate_many(self, points: np.ndarray) -> np.ndarray:
        candidates = np.asarray(points, dtype=float)
        if len(candidates) > self.remaining:
            raise RuntimeError("RGI evaluation batch exceeds the remaining FE budget")
        try:
            vector_values = np.asarray(self.objective(candidates), dtype=float)
        except (TypeError, ValueError):
            vector_values = np.asarray([], dtype=float)
        if vector_values.shape != (len(candidates),):
            values = np.empty(len(candidates), dtype=float)
            for index, point in enumerate(candidates):
                values[index] = self.evaluate(point)
            return values

        values = np.empty(len(candidates), dtype=float)
        for index, value in enumerate(vector_values):
            values[index] = self._record(float(value))
        return values


def _clip(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(points, lower), upper)


def _initial_population(
    budget: _EvaluationBudget,
    rng: np.random.Generator,
    *,
    population_size: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    population = rng.uniform(lower, upper, size=(population_size, len(lower)))
    values = budget.evaluate_many(population)
    return population, values


def _algorithm_abc(
    budget: _EvaluationBudget,
    rng: np.random.Generator,
    population_size: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    population, values = _initial_population(
        budget, rng, population_size=population_size, lower=lower, upper=upper
    )
    trials = np.zeros(population_size, dtype=int)
    stagnation_limit = max(20, 2 * len(lower))
    while budget.remaining:
        candidates = np.empty_like(population)
        scout_mask = trials >= stagnation_limit
        for index in range(population_size):
            if scout_mask[index]:
                candidates[index] = rng.uniform(lower, upper)
                continue
            other = int(rng.integers(0, population_size - 1))
            if other >= index:
                other += 1
            phi = rng.uniform(-1.0, 1.0, size=len(lower))
            candidates[index] = _clip(
                population[index] + phi * (population[index] - population[other]),
                lower,
                upper,
            )
        candidate_values = budget.evaluate_many(candidates)
        improved = candidate_values < values
        population[improved] = candidates[improved]
        values[improved] = candidate_values[improved]
        trials[improved] = 0
        trials[~improved] += 1
        trials[scout_mask] = 0


def _algorithm_aco(
    budget: _EvaluationBudget,
    rng: np.random.Generator,
    population_size: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    archive, values = _initial_population(
        budget, rng, population_size=population_size, lower=lower, upper=upper
    )
    while budget.remaining:
        order = np.argsort(values, kind="mergesort")
        archive = archive[order]
        values = values[order]
        weights = np.exp(-np.arange(population_size) / max(population_size / 3.0, 1.0))
        weights /= np.sum(weights)
        selected = rng.choice(population_size, size=population_size, p=weights)
        coordinate_spread = np.mean(np.abs(archive - archive[selected]), axis=0)
        sigma = 0.85 * coordinate_spread + 1.0e-6 * (upper - lower)
        candidates = archive[selected] + rng.normal(
            0.0, sigma, size=(population_size, len(lower))
        )
        candidates = _clip(candidates, lower, upper)
        candidate_values = budget.evaluate_many(candidates)
        archive = np.concatenate((archive, candidates), axis=0)
        values = np.concatenate((values, candidate_values), axis=0)
        keep = np.argsort(values, kind="mergesort")[:population_size]
        archive = archive[keep]
        values = values[keep]


def _algorithm_cmaes(
    budget: _EvaluationBudget,
    rng: np.random.Generator,
    population_size: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    dimension = len(lower)
    mean = rng.uniform(lower, upper)
    sigma = 0.3 * float(np.mean(upper - lower))
    coordinate_scale = np.ones(dimension, dtype=float)
    weights = np.log(population_size / 2.0 + 0.5) - np.log(
        np.arange(1, population_size + 1, dtype=float)
    )
    weights = np.maximum(weights, 0.0)
    weights /= np.sum(weights)
    while budget.remaining:
        noise = rng.normal(0.0, 1.0, size=(population_size, dimension))
        candidates = _clip(mean + sigma * noise * coordinate_scale, lower, upper)
        values = budget.evaluate_many(candidates)
        order = np.argsort(values, kind="mergesort")
        selected_noise = noise[order]
        old_mean = mean.copy()
        mean = np.sum(candidates[order] * weights[:, None], axis=0)
        centered = (mean - old_mean) / max(sigma, 1.0e-12)
        coordinate_scale = np.sqrt(
            0.9 * coordinate_scale**2 + 0.1 * np.square(centered)
        )
        coordinate_scale = np.clip(coordinate_scale, 0.05, 20.0)
        improvement = float(values[order[0]] - np.median(values))
        sigma *= float(np.exp(np.clip(-0.03 * improvement / (abs(np.median(values)) + 1.0), -0.2, 0.2)))
        sigma = float(np.clip(sigma, 1.0e-8, 2.0 * np.max(upper - lower)))
        # Referencing the selected noise keeps the adaptation tied to the
        # sampled generation and avoids an unused stochastic branch.
        if not np.isfinite(selected_noise).all():
            raise FloatingPointError("CMA-ES generated a non-finite search direction")


def _algorithm_cso(
    budget: _EvaluationBudget,
    rng: np.random.Generator,
    population_size: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    population, values = _initial_population(
        budget, rng, population_size=population_size, lower=lower, upper=upper
    )
    while budget.remaining:
        mean = np.mean(population, axis=0)
        permutation = rng.permutation(population_size)
        candidates = population.copy()
        for left in range(0, population_size, 2):
            first = int(permutation[left])
            second = int(permutation[left + 1])
            if values[first] <= values[second]:
                winner, loser = first, second
            else:
                winner, loser = second, first
            r1 = rng.random(len(lower))
            r2 = rng.random(len(lower))
            candidates[loser] = _clip(
                population[loser]
                + r1 * (population[winner] - population[loser])
                + r2 * (mean - population[loser]),
                lower,
                upper,
            )
        candidate_values = budget.evaluate_many(candidates)
        improved = candidate_values < values
        population[improved] = candidates[improved]
        values[improved] = candidate_values[improved]


def _algorithm_de(
    budget: _EvaluationBudget,
    rng: np.random.Generator,
    population_size: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    population, values = _initial_population(
        budget, rng, population_size=population_size, lower=lower, upper=upper
    )
    indices = np.arange(population_size)
    for _ in range(max(0, budget.remaining // population_size)):
        trials = np.empty_like(population)
        for index in range(population_size):
            available = np.delete(indices, index)
            a, b, c = rng.choice(available, size=3, replace=False)
            mutant = population[a] + 0.5 * (population[b] - population[c])
            mask = rng.random(len(lower)) < 0.9
            mask[int(rng.integers(0, len(lower)))] = True
            trials[index] = _clip(
                np.where(mask, mutant, population[index]), lower, upper
            )
        trial_values = budget.evaluate_many(trials)
        improved = trial_values < values
        population[improved] = trials[improved]
        values[improved] = trial_values[improved]


def _algorithm_fep(
    budget: _EvaluationBudget,
    rng: np.random.Generator,
    population_size: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    population, values = _initial_population(
        budget, rng, population_size=population_size, lower=lower, upper=upper
    )
    scale = (upper - lower) / np.sqrt(2.0 * np.sqrt(len(lower)))
    while budget.remaining:
        offspring = _clip(
            population + rng.normal(0.0, scale, size=population.shape), lower, upper
        )
        offspring_values = budget.evaluate_many(offspring)
        combined = np.concatenate((population, offspring), axis=0)
        combined_values = np.concatenate((values, offspring_values), axis=0)
        keep = np.argsort(combined_values, kind="mergesort")[:population_size]
        population = combined[keep]
        values = combined_values[keep]


def _tournament_index(values: np.ndarray, rng: np.random.Generator) -> int:
    first, second = rng.integers(0, len(values), size=2)
    return int(first if values[first] <= values[second] else second)


def _algorithm_ga(
    budget: _EvaluationBudget,
    rng: np.random.Generator,
    population_size: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    population, values = _initial_population(
        budget, rng, population_size=population_size, lower=lower, upper=upper
    )
    elite_count = max(2, population_size // 20)
    while budget.remaining:
        order = np.argsort(values, kind="mergesort")
        offspring = np.empty_like(population)
        offspring[:elite_count] = population[order[:elite_count]]
        for index in range(elite_count, population_size):
            parent_a = population[_tournament_index(values, rng)]
            parent_b = population[_tournament_index(values, rng)]
            mix = rng.random(len(lower))
            child = mix * parent_a + (1.0 - mix) * parent_b
            mutation = rng.random(len(lower)) < 1.0 / len(lower)
            child = child + mutation * rng.normal(
                0.0, 0.1 * (upper - lower), size=len(lower)
            )
            offspring[index] = _clip(child, lower, upper)
        offspring_values = budget.evaluate_many(offspring)
        population, values = offspring, offspring_values


def _algorithm_pso(
    budget: _EvaluationBudget,
    rng: np.random.Generator,
    population_size: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    population, values = _initial_population(
        budget, rng, population_size=population_size, lower=lower, upper=upper
    )
    velocity = rng.normal(0.0, 0.1 * (upper - lower), size=population.shape)
    personal_best = population.copy()
    personal_values = values.copy()
    global_index = int(np.argmin(personal_values))
    global_best = personal_best[global_index].copy()
    while budget.remaining:
        inertia = 0.7
        cognitive = rng.random(population.shape) * 1.5
        social = rng.random(population.shape) * 1.5
        velocity = (
            inertia * velocity
            + cognitive * (personal_best - population)
            + social * (global_best - population)
        )
        population = _clip(population + velocity, lower, upper)
        values = budget.evaluate_many(population)
        improved = values < personal_values
        personal_best[improved] = population[improved]
        personal_values[improved] = values[improved]
        global_index = int(np.argmin(personal_values))
        global_best = personal_best[global_index].copy()


def _algorithm_sa(
    budget: _EvaluationBudget,
    rng: np.random.Generator,
    population_size: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    del population_size
    current = rng.uniform(lower, upper)
    current_value = budget.evaluate(current)
    temperature = max(abs(current_value), 1.0)
    scale = 0.2 * (upper - lower)
    while budget.remaining:
        candidate = _clip(current + rng.normal(0.0, scale), lower, upper)
        candidate_value = budget.evaluate(candidate)
        delta = candidate_value - current_value
        acceptance = delta <= 0.0 or rng.random() < np.exp(
            np.clip(-delta / max(temperature, 1.0e-12), -700.0, 0.0)
        )
        if acceptance:
            current, current_value = candidate, candidate_value
        temperature = max(1.0e-12, temperature * 0.9995)
        scale *= 0.9998


def _algorithm_rand(
    budget: _EvaluationBudget,
    rng: np.random.Generator,
    population_size: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    while budget.remaining:
        batch_size = min(population_size, budget.remaining)
        points = rng.uniform(lower, upper, size=(batch_size, len(lower)))
        budget.evaluate_many(points)


_ALGORITHM_RUNNERS = {
    "ABC": _algorithm_abc,
    "ACO": _algorithm_aco,
    "CMA-ES": _algorithm_cmaes,
    "CSO": _algorithm_cso,
    "DE": _algorithm_de,
    "FEP": _algorithm_fep,
    "GA": _algorithm_ga,
    "PSO": _algorithm_pso,
    "SA": _algorithm_sa,
    "RAND": _algorithm_rand,
}


def canonical_algorithm_name(name: str) -> str:
    key = str(name).strip().upper()
    try:
        return _ALGORITHM_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"unsupported RGI algorithm {name!r}; choose from {PAPER_ALGORITHMS}"
        ) from exc


def run_algorithm(
    *,
    algorithm: str,
    objective: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    dimension: int,
    lower_bound: float,
    upper_bound: float,
    population_size: int = 100,
    fe_total: int = 10_000,
    target_value: float | None = None,
    target_tolerance: float = 0.0,
    run_seed: int = 0,
) -> RGIAlgorithmResult:
    """Run one RGI algorithm and return a complete run-level outcome."""
    name = canonical_algorithm_name(algorithm)
    if dimension < 1:
        raise ValueError("dimension must be at least 1")
    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    if fe_total < population_size:
        raise ValueError("fe_total must be at least population_size")
    if name != "SA" and fe_total % population_size != 0:
        raise ValueError(
            "fe_total must be a multiple of population_size for population algorithms"
        )
    if lower_bound >= upper_bound:
        raise ValueError("lower_bound must be smaller than upper_bound")
    if target_tolerance < 0.0:
        raise ValueError("target_tolerance must be non-negative")
    if name == "CSO" and population_size % 2:
        raise ValueError("CSO requires an even population_size")

    lower = np.full(dimension, float(lower_bound), dtype=float)
    upper = np.full(dimension, float(upper_bound), dtype=float)
    budget = _EvaluationBudget(
        objective,
        fe_total=fe_total,
        target_value=target_value,
        target_tolerance=target_tolerance,
    )
    try:
        _ALGORITHM_RUNNERS[name](budget, rng, population_size, lower, upper)
        if budget.evaluations != fe_total:
            raise RuntimeError(
                f"algorithm used {budget.evaluations} FE, expected {fe_total}"
            )
        return RGIAlgorithmResult(
            algorithm=name,
            run_seed=int(run_seed),
            fe_total=fe_total,
            effective_fe=budget.evaluations,
            best_value=float(budget.best_value),
            final_value=float(budget.final_value),
            first_hit_fe=budget.first_hit_fe,
            target_hit=budget.first_hit_fe is not None,
            performance_value=(
                float(budget.first_hit_fe)
                if budget.first_hit_fe is not None
                else (-1.0 if target_value is not None else float(budget.best_value))
            ),
            run_status="completed",
        )
    except Exception as exc:
        failed_performance = (
            -1.0
            if target_value is not None
            else (
                float(budget.best_value)
                if np.isfinite(budget.best_value)
                else -1.0
            )
        )
        return RGIAlgorithmResult(
            algorithm=name,
            run_seed=int(run_seed),
            fe_total=fe_total,
            effective_fe=budget.evaluations,
            best_value=(None if not np.isfinite(budget.best_value) else float(budget.best_value)),
            final_value=budget.final_value,
            first_hit_fe=budget.first_hit_fe,
            target_hit=budget.first_hit_fe is not None,
            performance_value=(
                float(budget.first_hit_fe)
                if budget.first_hit_fe is not None
                else failed_performance
            ),
            run_status="failed",
            failure_type=type(exc).__name__,
            failure_message=str(exc),
        )


__all__ = [
    "PAPER_ALGORITHMS",
    "RGIAlgorithmResult",
    "canonical_algorithm_name",
    "run_algorithm",
]
