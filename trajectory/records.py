from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trajectory.sampling import SAMPLING_METADATA_COLUMNS


OPTIMIZER_STATE_MODE = "native_optimizer_state_with_dynamic_sampling_v4"


@dataclass(frozen=True)
class TrajectoryRecord:
    problem_id: str
    function_id: str
    family: str
    dimension: int
    algorithm: str
    seed: int
    FE: int
    FE_ratio: float
    FE_total: int
    native_updates: int
    window_statistics: list[dict]
    native_update_history: list[dict]
    population: list[list[float]]
    fitness: list[float]
    best_fitness: float
    optimizer_state_mode: str
    sampling_protocol: str
    sampling_phase: str
    sampling_triggers: list[str]
    is_budget_milestone: bool
    budget_milestone_ratio: float | None
    is_event_sample: bool
    monitor_target_ratio: float
    event_index_in_phase: int | None
    event_improvement_resume: bool
    event_stagnation_onset: bool
    event_rank_change: bool
    event_elite_migration: bool
    event_diversity_recovery: bool
    event_improvement_resume_metric: float
    event_stagnation_onset_metric: float
    event_rank_change_metric: float
    event_elite_migration_metric: float
    event_diversity_recovery_metric: float

    @classmethod
    def from_arrays(
        cls,
        *,
        problem_id: str,
        function_id: str,
        family: str,
        dimension: int,
        algorithm: str,
        seed: int,
        fe: int,
        fe_total: int,
        native_updates: int,
        window_statistics: list[dict],
        native_update_history: list[dict],
        population: np.ndarray,
        fitness: np.ndarray,
        best_fitness: float,
        sampling_metadata: dict,
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
        metadata = _sampling_metadata(sampling_metadata)
        return cls(
            problem_id=problem_id,
            function_id=function_id,
            family=family,
            dimension=int(dimension),
            algorithm=algorithm,
            seed=int(seed),
            FE=int(fe),
            FE_ratio=float(fe / fe_total),
            FE_total=int(fe_total),
            native_updates=int(native_updates),
            window_statistics=[dict(item) for item in window_statistics],
            native_update_history=[dict(item) for item in native_update_history],
            population=pop.tolist(),
            fitness=fit.tolist(),
            best_fitness=float(best_fitness),
            optimizer_state_mode=OPTIMIZER_STATE_MODE,
            **metadata,
        )


def _sampling_metadata(values: dict) -> dict:
    missing = set(SAMPLING_METADATA_COLUMNS).difference(values)
    extra = set(values).difference(SAMPLING_METADATA_COLUMNS)
    if missing or extra:
        raise ValueError(f"sampling metadata columns mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    return {column: values[column] for column in SAMPLING_METADATA_COLUMNS}
