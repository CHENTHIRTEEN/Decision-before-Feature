from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass

import numpy as np

from behavior.features import extract_behavior_rows
from behavior_with_ela.analysis_v5.task13.task13_replay import SAMPLING_METADATA
from trajectory.records import TrajectoryRecord


PRIMARY_WINDOW_FE = 500
FITNESS_TOL = 1e-12
FITNESS_SCALE_FLOOR = 1e-3


@dataclass
class AgentObservation:
    fe: int
    fitness: float


class AgentHistoryTracker:
    """Track Task15A-compatible search-slot lineages on natural trajectories."""

    def __init__(self, state) -> None:
        self.agent_ids = np.asarray(
            [f"a{index:03d}" for index in range(len(state.population))],
            dtype=object,
        )
        self.history: dict[str, list[AgentObservation]] = {
            str(agent_id): [
                AgentObservation(fe=int(state.evaluations), fitness=float(value))
            ]
            for agent_id, value in zip(self.agent_ids, state.fitness, strict=True)
        }
        self._parent_ids: np.ndarray | None = None
        self._parent_fitness: np.ndarray | None = None
        self._evaluated_fitness: list[float] = []
        self._cso_losers: np.ndarray | None = None

    def clone(self) -> "AgentHistoryTracker":
        return copy.deepcopy(self)

    def start_generation(self, state) -> None:
        pending = getattr(state, "pending_population", None)
        if pending is None:
            pending = getattr(state, "pending_positions", None)
        if pending is not None:
            raise ValueError("generation tracking must start before pending candidates exist")
        self._parent_ids = self.agent_ids.copy()
        self._parent_fitness = np.asarray(state.fitness, dtype=float).copy()
        self._evaluated_fitness = []
        self._cso_losers = None

    def observe_evaluation(self, state, value: float) -> None:
        if self._parent_ids is None or self._parent_fitness is None:
            raise RuntimeError("missing generation parent state")
        if state.algorithm == "cso" and self._cso_losers is None:
            self._cso_losers = np.asarray(state.pending_loser_indices, dtype=int).copy()
        self._evaluated_fitness.append(float(value))

    def finish_generation(self, state) -> None:
        if self._parent_ids is None or self._parent_fitness is None:
            raise RuntimeError("missing generation parent state")
        if state.algorithm in {"shade", "lshade"}:
            trial_fitness = np.asarray(self._evaluated_fitness, dtype=float)
            if len(trial_fitness) != len(self._parent_ids):
                raise RuntimeError("SHADE lineage tracking did not observe a complete update")
            selected_fitness = np.minimum(self._parent_fitness, trial_fitness)
            if state.algorithm == "lshade" and len(state.population) < len(self._parent_ids):
                keep = np.argsort(selected_fitness, kind="mergesort")[: len(state.population)]
                self.agent_ids = self._parent_ids[keep]
            else:
                self.agent_ids = self._parent_ids.copy()
        elif state.algorithm == "cso":
            if self._cso_losers is None or len(self._evaluated_fitness) != len(self._cso_losers):
                raise RuntimeError("CSO lineage tracking did not observe a complete update")
            self.agent_ids = self._parent_ids.copy()
        else:
            raise ValueError(f"unsupported Task16A algorithm: {state.algorithm}")
        if len(self.agent_ids) != len(state.population):
            raise RuntimeError("agent lineage count does not match active population")
        for agent_id, value in zip(self.agent_ids, state.fitness, strict=True):
            self.history.setdefault(str(agent_id), []).append(
                AgentObservation(fe=int(state.evaluations), fitness=float(value))
            )
        self._parent_ids = None
        self._parent_fitness = None
        self._evaluated_fitness = []
        self._cso_losers = None

    def individual_primitives(self, state) -> list[dict]:
        current_fitness = np.asarray(state.fitness, dtype=float)
        q25, median, q75 = np.quantile(current_fitness, [0.25, 0.50, 0.75])
        scale = max(
            float(q75 - q25),
            FITNESS_SCALE_FLOOR * max(1.0, abs(float(median))),
            np.finfo(float).eps,
        )
        final_fe = int(state.evaluations)
        rows = []
        for index, agent_id_value in enumerate(self.agent_ids):
            agent_id = str(agent_id_value)
            history = self.history[agent_id]
            transitions = [
                (earlier, later)
                for earlier, later in zip(history[:-1], history[1:], strict=True)
                if later.fe > final_fe - PRIMARY_WINDOW_FE
            ]
            improvements = [
                max(float(earlier.fitness - later.fitness), 0.0)
                for earlier, later in transitions
            ]
            progress = float(sum(improvements) / max(len(transitions) * scale, np.finfo(float).eps))
            last_improvement_fe = int(history[0].fe)
            for earlier, later in zip(history[:-1], history[1:], strict=True):
                tolerance = FITNESS_TOL * max(1.0, abs(float(earlier.fitness)))
                if float(earlier.fitness - later.fitness) > tolerance:
                    last_improvement_fe = int(later.fe)
            stagnation_age = int(min(max(final_fe - last_improvement_fe, 0), PRIMARY_WINDOW_FE))
            rows.append(
                {
                    "agent_id": agent_id,
                    "population_index": int(index),
                    "individual_stagnation_age_FE": stagnation_age,
                    "individual_recent_progress": progress,
                    "history_length": int(len(history)),
                }
            )
        stagnation_order = sorted(
            range(len(rows)),
            key=lambda i: (-rows[i]["individual_stagnation_age_FE"], rows[i]["agent_id"]),
        )
        progress_order = sorted(
            range(len(rows)),
            key=lambda i: (rows[i]["individual_recent_progress"], rows[i]["agent_id"]),
        )
        for rank, index in enumerate(stagnation_order, start=1):
            rows[index]["selection_rank_stagnation"] = int(rank)
        for rank, index in enumerate(progress_order, start=1):
            rows[index]["selection_rank_progress"] = int(rank)
        return rows


def extract_task16a_probes(*, problem, state, algorithm: str, seed: int, recorder, native_updates: int) -> dict:
    recorder_copy = copy.deepcopy(recorder)
    windows, history = recorder_copy.build(
        fe_total=10000,
        problem_id=problem.problem_id,
        algorithm=algorithm,
    )
    snapshot = recorder_copy.current_snapshot
    record = TrajectoryRecord.from_arrays(
        problem_id=problem.problem_id,
        function_id=problem.function_id,
        family=problem.family,
        cv_group_id=problem.cv_group_id,
        dimension=problem.dimension,
        algorithm=algorithm,
        seed=int(seed),
        fe=int(snapshot.fe),
        fe_total=10000,
        native_updates=int(native_updates),
        window_statistics=windows,
        native_update_history=history,
        population=snapshot.population,
        fitness=snapshot.fitness,
        best_fitness=snapshot.best_fitness,
        sampling_metadata=dict(SAMPLING_METADATA),
    )
    behavior = extract_behavior_rows([dataclasses.asdict(record)])
    if len(behavior) != 1:
        raise RuntimeError("expected one behavior row for one source state")
    row = behavior[0]
    productivity = float(row["bf_fitness_distribution_improvement_rate_w02"])
    entropy = float(1.0 - row["bf_centroid_shift_coherence_w05"])
    stagnation = float(row["bf_stagnation_w10"])
    values = np.asarray([productivity, entropy, stagnation], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Task16A probes must be finite")
    return {
        "probe_productivity": productivity,
        "probe_entropy": entropy,
        "probe_stagnation": stagnation,
        "probe_productivity_source": "bf_fitness_distribution_improvement_rate_w02",
        "probe_entropy_source": "1-bf_centroid_shift_coherence_w05",
        "probe_stagnation_source": "bf_stagnation_w10",
        "effective_window_fe_w02": int(row["effective_window_fe_w02"]),
        "effective_window_fe_w05": int(row["effective_window_fe_w05"]),
        "effective_window_fe_w10": int(row["effective_window_fe_w10"]),
        "effective_native_updates_w02": int(row["effective_native_updates_w02"]),
        "effective_native_updates_w05": int(row["effective_native_updates_w05"]),
        "effective_native_updates_w10": int(row["effective_native_updates_w10"]),
    }
