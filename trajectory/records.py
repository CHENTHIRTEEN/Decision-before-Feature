from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryRecord:
    problem_id: str
    family: str
    dimension: int
    algorithm: str
    seed: int
    FE: int
    FE_ratio: float
    population: list[list[float]]
    fitness: list[float]
    best_fitness: float

    @classmethod
    def from_arrays(
        cls,
        *,
        problem_id: str,
        family: str,
        dimension: int,
        algorithm: str,
        seed: int,
        fe: int,
        fe_total: int,
        population: np.ndarray,
        fitness: np.ndarray,
        best_fitness: float,
    ) -> "TrajectoryRecord":
        pop = np.asarray(population, dtype=float)
        fit = np.asarray(fitness, dtype=float).reshape(-1)
        if pop.ndim != 2:
            raise ValueError("population must be a two-dimensional array")
        if pop.shape[0] != fit.shape[0]:
            raise ValueError("population and fitness must contain the same number of rows")
        return cls(
            problem_id=problem_id,
            family=family,
            dimension=int(dimension),
            algorithm=algorithm,
            seed=int(seed),
            FE=int(fe),
            FE_ratio=float(fe / fe_total),
            population=pop.tolist(),
            fitness=fit.tolist(),
            best_fitness=float(best_fitness),
        )

