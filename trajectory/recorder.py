from __future__ import annotations

import numpy as np

from benchmarks.core import Problem
from trajectory.records import TrajectoryRecord
from trajectory.sampling import (
    DynamicSamplingPolicy,
    SamplingDecision,
    sampling_metrics,
)
from trajectory.window_statistics import NativeUpdateWindowRecorder


class TrajectoryRecorder:
    def __init__(
        self,
        *,
        sampling_protocol: str,
    ) -> None:
        self._sampling_policy = DynamicSamplingPolicy(sampling_protocol)
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
        if self._sampling_policy.complete:
            return
        self._window_recorder.observe(
            fe=fe,
            native_updates=native_updates,
            population=population,
            fitness=fitness,
            best_fitness=best_fitness,
        )
        self._observe_dynamic(
            problem=problem,
            algorithm=algorithm,
            seed=seed,
            fe=fe,
            fe_total=fe_total,
            native_updates=native_updates,
            population=population,
            fitness=fitness,
            best_fitness=best_fitness,
        )

    def _observe_dynamic(self, **state) -> None:
        policy = self._sampling_policy
        policy.observe_update(fe=int(state["fe"]), best_fitness=float(state["best_fitness"]))
        pending = policy.pending_monitor_ratios(int(state["fe"]) / int(state["fe_total"]))
        if not pending:
            return
        window_statistics, native_update_history = self._window_recorder.build(
            fe_total=int(state["fe_total"]),
            problem_id=state["problem"].problem_id,
            algorithm=str(state["algorithm"]),
        )
        metrics = sampling_metrics(
            window_statistics=window_statistics,
            native_update_history=native_update_history,
            dimension=state["problem"].dimension,
            stagnation_span_ratio=policy.stagnation_span_ratio(
                current_fe=int(state["fe"]), fe_total=int(state["fe_total"])
            ),
        )
        decision = policy.decide_pending(
            monitor_target_ratios=pending,
            actual_fe_ratio=int(state["fe"]) / int(state["fe_total"]),
            **metrics,
        )
        if not decision.should_emit:
            return
        self._append_record(
            **state,
            window_statistics=window_statistics,
            native_update_history=native_update_history,
            sampling_decision=decision,
        )

    def _append_record(
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
        window_statistics: list[dict],
        native_update_history: list[dict],
        sampling_decision: SamplingDecision,
    ) -> None:
        if self._last_recorded_fe == int(fe):
            raise ValueError("trajectory sampling must emit at most one row per FE")
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
                sampling_metadata=sampling_decision.metadata(),
            )
        )
        self._last_recorded_fe = int(fe)
