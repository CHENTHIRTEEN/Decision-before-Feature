from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from behavior.features import extract_behavior_rows
from trajectory.records import TrajectoryRecord
from trajectory.sampling import (
    DynamicSamplingPolicy,
    SamplingDecision,
    sampling_metrics,
)
from trajectory.window_statistics import NativeUpdateWindowRecorder


@dataclass(frozen=True)
class DecisionObservation:
    behavior_state: dict[str, Any]
    decision_score: float | None
    run_query: bool | None


class StreamingBehaviorState:
    """Shared causal state interface for offline collection and online evaluation."""

    def __init__(
        self,
        *,
        problem_id: str,
        function_id: str,
        family: str,
        dimension: int,
        algorithm: str,
        seed: int,
        fe_total: int,
        sampling_protocol: str | None = None,
    ) -> None:
        self.problem_id = str(problem_id)
        self.function_id = str(function_id)
        self.family = str(family)
        self.dimension = int(dimension)
        self.algorithm = str(algorithm)
        self.seed = int(seed)
        self.fe_total = int(fe_total)
        if self.fe_total <= 0:
            raise ValueError("FE_total must be positive")
        self._window_recorder = NativeUpdateWindowRecorder()
        self._sampling_policy = (
            DynamicSamplingPolicy(sampling_protocol) if sampling_protocol is not None else None
        )
        self._trajectory_rows: list[dict[str, Any]] = []
        self._latest_record: TrajectoryRecord | None = None
        self._latest_behavior_state: dict[str, Any] | None = None
        self._latest_record_emitted = False

    def observe(
        self,
        *,
        fe: int,
        native_updates: int,
        population: np.ndarray,
        fitness: np.ndarray,
        best_fitness: float,
    ) -> None:
        """Observe one completed native optimizer update without emitting a sample."""
        self._window_recorder.observe(
            fe=fe,
            native_updates=native_updates,
            population=population,
            fitness=fitness,
            best_fitness=best_fitness,
        )
        if self._sampling_policy is not None:
            self._sampling_policy.observe_update(fe=int(fe), best_fitness=float(best_fitness))

    @property
    def next_monitor_ratio(self) -> float | None:
        if self._sampling_policy is None:
            return None
        return self._sampling_policy.next_monitor_ratio

    def sample_dynamic(self) -> dict[str, Any] | None:
        """Emit the current state when a frozen milestone or causal event accepts it."""
        policy = self._sampling_policy
        if policy is None:
            raise ValueError("dynamic sampling was not configured for this behavior stream")
        current = self._window_recorder.current_snapshot
        actual_ratio = float(current.fe / self.fe_total)
        pending = policy.pending_monitor_ratios(actual_ratio)
        if not pending:
            return None
        window_statistics, native_update_history = self._window_recorder.build(
            fe_total=self.fe_total,
            problem_id=self.problem_id,
            algorithm=self.algorithm,
        )
        metrics = sampling_metrics(
            window_statistics=window_statistics,
            native_update_history=native_update_history,
            dimension=self.dimension,
            stagnation_span_ratio=policy.stagnation_span_ratio(
                current_fe=current.fe,
                fe_total=self.fe_total,
            ),
        )
        decision = policy.decide_pending(
            monitor_target_ratios=pending,
            actual_fe_ratio=actual_ratio,
            **metrics,
        )
        if not decision.should_emit:
            return None
        self._set_record(
            current=current,
            window_statistics=window_statistics,
            native_update_history=native_update_history,
            sampling_decision=decision,
        )
        return self.emit_features()

    def _set_record(
        self,
        *,
        current,
        window_statistics: list[dict],
        native_update_history: list[dict],
        sampling_decision: SamplingDecision,
    ) -> None:
        record = TrajectoryRecord.from_arrays(
            problem_id=self.problem_id,
            function_id=self.function_id,
            family=self.family,
            dimension=self.dimension,
            algorithm=self.algorithm,
            seed=self.seed,
            fe=current.fe,
            fe_total=self.fe_total,
            native_updates=current.native_updates,
            window_statistics=window_statistics,
            native_update_history=native_update_history,
            population=current.population,
            fitness=current.fitness,
            best_fitness=current.best_fitness,
            sampling_metadata=sampling_decision.metadata(),
        )
        self._latest_record = record
        self._latest_behavior_state = None
        self._latest_record_emitted = False

    def emit_features(self) -> dict[str, Any]:
        """Emit the same behavior state representation used by offline extraction."""
        if self._latest_record is None:
            raise ValueError("update_window must be called before emit_features")
        if self._latest_record_emitted:
            raise ValueError("the current behavior state has already been emitted")
        self._trajectory_rows.append(self._latest_record.__dict__.copy())
        self._latest_behavior_state = extract_behavior_rows(
            [row.copy() for row in self._trajectory_rows]
        )[-1]
        self._latest_record_emitted = True
        return dict(self._latest_behavior_state)

    def maybe_decide(
        self,
        *,
        score: Callable[[dict[str, Any]], float],
        threshold: float,
    ) -> DecisionObservation:
        """Evaluate a frozen controller threshold on the most recently emitted state."""
        if self._latest_behavior_state is None:
            raise ValueError("emit_features must be called before maybe_decide")
        decision_score = float(score(dict(self._latest_behavior_state)))
        return DecisionObservation(
            behavior_state=dict(self._latest_behavior_state),
            decision_score=decision_score,
            run_query=bool(decision_score > float(threshold)),
        )

    @property
    def trajectory_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(row.copy() for row in self._trajectory_rows)
