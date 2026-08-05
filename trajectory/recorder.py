from __future__ import annotations

import numpy as np

from benchmarks.core import Problem
from trajectory.records import TrajectoryRecord


class TrajectoryRecorder:
    def __init__(self, checkpoint_ratios: tuple[float, ...]):
        self._checkpoint_ratios = tuple(checkpoint_ratios)
        self._next_checkpoint = 0
        self.records: list[TrajectoryRecord] = []
        self._last_recorded_fe: int | None = None

    def observe(
        self,
        *,
        problem: Problem,
        algorithm: str,
        seed: int,
        fe: int,
        fe_total: int,
        population: np.ndarray,
        fitness: np.ndarray,
        best_fitness: float,
    ) -> None:
        if self._next_checkpoint >= len(self._checkpoint_ratios):
            return
        fe_ratio = fe / fe_total
        if fe_ratio < self._checkpoint_ratios[self._next_checkpoint]:
            return

        while (
            self._next_checkpoint < len(self._checkpoint_ratios)
            and fe_ratio >= self._checkpoint_ratios[self._next_checkpoint]
        ):
            self._next_checkpoint += 1

        if self._last_recorded_fe == fe:
            return
        self.records.append(
            TrajectoryRecord.from_arrays(
                problem_id=problem.problem_id,
                family=problem.family,
                dimension=problem.dimension,
                algorithm=algorithm,
                seed=seed,
                fe=fe,
                fe_total=fe_total,
                population=population,
                fitness=fitness,
                best_fitness=best_fitness,
            )
        )
        self._last_recorded_fe = fe

