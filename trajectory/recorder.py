from __future__ import annotations

import numpy as np

from benchmarks.core import Problem
from trajectory.records import TrajectoryRecord
from trajectory.window_statistics import (
    NativeUpdateWindowRecorder,
)


class TrajectoryRecorder:
    def __init__(self, checkpoint_ratios: tuple[float, ...]):
        self._checkpoint_ratios = tuple(checkpoint_ratios)
        self._next_checkpoint = 0
        self.records: list[TrajectoryRecord] = []
        self._last_recorded_fe: int | None = None
        self._window_recorder = NativeUpdateWindowRecorder()

    def observe(
        self,
        *,
        problem: Problem,
        algorithm: str,
        seed: int,
        fe: int,
        fe_total: int,
        native_updates: int,
        population: np.ndarray,
        fitness: np.ndarray,
        best_fitness: float,
    ) -> None:
        if self._next_checkpoint >= len(self._checkpoint_ratios):
            return
        self._window_recorder.observe(
            fe=fe,
            native_updates=native_updates,
            population=population,
            fitness=fitness,
            best_fitness=best_fitness,
        )
        fe_ratio = fe / fe_total
        checkpoint_ratio = self._checkpoint_ratios[self._next_checkpoint]
        if fe_ratio < checkpoint_ratio:
            return

        while (
            self._next_checkpoint < len(self._checkpoint_ratios)
            and fe_ratio >= self._checkpoint_ratios[self._next_checkpoint]
        ):
            self._next_checkpoint += 1

        if self._last_recorded_fe == fe:
            return
        window_statistics, native_update_history = self._window_recorder.build(fe_total=fe_total)
        self.records.append(
            TrajectoryRecord.from_arrays(
                problem_id=problem.problem_id,
                family=problem.family,
                dimension=problem.dimension,
                algorithm=algorithm,
                seed=seed,
                fe=fe,
                fe_total=fe_total,
                native_updates=native_updates,
                window_statistics=window_statistics,
                native_update_history=native_update_history,
                population=population,
                fitness=fitness,
                best_fitness=best_fitness,
                fe_ratio=checkpoint_ratio,
            )
        )
        self._last_recorded_fe = fe
