from __future__ import annotations

from dataclasses import dataclass

import numpy as np


OPTIMIZER_STATE_MODE = "native_optimizer_state"


@dataclass(frozen=True)
class TrajectoryRecord:
    problem_id: str
    family: str
    dimension: int
    algorithm: str
    seed: int
    FE: int
    FE_ratio: float
    FE_total: int
    native_updates: int
    population: list[list[float]]
    fitness: list[float]
    best_fitness: float
    optimizer_state_mode: str

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
        native_updates: int,
        population: np.ndarray,
        fitness: np.ndarray,
        best_fitness: float,
        fe_ratio: float | None = None,
    ) -> "TrajectoryRecord":
        pop = np.asarray(population, dtype=float)
        fit = np.asarray(fitness, dtype=float).reshape(-1)
        if pop.ndim != 2:
            raise ValueError("population must be a two-dimensional array")
        if pop.shape[0] != fit.shape[0]:
            raise ValueError("population and fitness must contain the same number of rows")
        if int(fe_total) <= 0 or not 0 < int(fe) <= int(fe_total):
            raise ValueError("trajectory FE must be in (0, FE_total]")
        if int(native_updates) < 0:
            raise ValueError("native_updates must be non-negative")
        return cls(
            problem_id=problem_id,
            family=family,
            dimension=int(dimension),
            algorithm=algorithm,
            seed=int(seed),
            FE=int(fe),
            FE_ratio=float(fe_ratio if fe_ratio is not None else fe / fe_total),
            FE_total=int(fe_total),
            native_updates=int(native_updates),
            population=pop.tolist(),
            fitness=fit.tolist(),
            best_fitness=float(best_fitness),
            optimizer_state_mode=OPTIMIZER_STATE_MODE,
        )
