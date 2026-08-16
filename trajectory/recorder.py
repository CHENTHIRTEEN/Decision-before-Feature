from __future__ import annotations

import numpy as np

from benchmarks.core import Problem
from trajectory.query import TrajectoryQueryReservoir, parse_problem_id
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
        trajectory_query_enabled: bool = False,
        trajectory_query_reservoir_size: int | None = None,
        trajectory_query_split: str | None = None,
    ) -> None:
        self._sampling_policy = DynamicSamplingPolicy(sampling_protocol)
        self.records: list[TrajectoryRecord] = []
        self.trajectory_query_records: list[dict] = []
        self._last_recorded_fe: int | None = None
        self._window_recorder = NativeUpdateWindowRecorder()
        self._query_reservoir: TrajectoryQueryReservoir | None = None
        self._trajectory_query_enabled = bool(trajectory_query_enabled)
        self._trajectory_query_reservoir_size = trajectory_query_reservoir_size
        self._trajectory_query_split = (
            None if trajectory_query_split is None else str(trajectory_query_split)
        )
        if self._trajectory_query_enabled and not self._trajectory_query_split:
            raise ValueError("enabled trajectory queries require an explicit non-empty split")

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

    def observe_evaluation(self, *, problem: Problem, algorithm: str, seed: int, point: np.ndarray, value: float) -> None:
        if not self._trajectory_query_enabled:
            return
        if self._query_reservoir is None:
            self._query_reservoir = TrajectoryQueryReservoir(
                problem_id=problem.problem_id,
                function_id=problem.function_id,
                family=problem.family,
                cv_group_id=problem.cv_group_id,
                dimension=problem.dimension,
                algorithm=algorithm,
                seed=seed,
                lower_bounds=problem.lower_bounds,
                upper_bounds=problem.upper_bounds,
                reservoir_size=self._trajectory_query_reservoir_size,
            )
        self._query_reservoir.observe(point, value)

    def build_trajectory_query_snapshot(
        self,
        *,
        problem: Problem,
        algorithm: str,
        seed: int,
        fe: int,
        fe_total: int,
        native_updates: int,
    ) -> dict | None:
        if self._query_reservoir is None:
            return None
        if not self._trajectory_query_split:
            raise RuntimeError("trajectory query split was not configured")
        if not self.records or int(self.records[-1].FE) != int(fe):
            raise ValueError("trajectory query snapshots must attach to the latest emitted trajectory state")
        if int(self.records[-1].native_updates) != int(native_updates):
            raise ValueError("trajectory query native_updates must match the latest emitted trajectory state")
        if self.trajectory_query_records and int(self.trajectory_query_records[-1]["FE"]) >= int(fe):
            raise ValueError("trajectory query snapshot FE values must be strictly increasing")
        function, instance = parse_problem_id(problem.problem_id)
        snapshot = self._query_reservoir.snapshot(
            split=self._trajectory_query_split,
            problem_id=problem.problem_id,
            function_id=problem.function_id,
            family=problem.family,
            function=function,
            instance=instance,
            algorithm=algorithm,
            seed=seed,
            fe=fe,
            fe_total=fe_total,
            native_updates=native_updates,
        )
        self.trajectory_query_records.append(snapshot)
        return snapshot

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
                function_id=problem.function_id,
                family=problem.family,
                cv_group_id=problem.cv_group_id,
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
        if self._trajectory_query_enabled:
            self.build_trajectory_query_snapshot(
                problem=problem,
                algorithm=algorithm,
                seed=seed,
                fe=fe,
                fe_total=fe_total,
                native_updates=native_updates,
            )
