from __future__ import annotations

from dataclasses import dataclass

from benchmarks.core import Problem
from optimizers.state import advance_optimizer_state, initialize_optimizer_state
from optimizers.settings import OptimizerSettings
from trajectory.final_performance import FinalPerformanceRecord
from trajectory.recorder import TrajectoryRecorder
from trajectory.records import TrajectoryRecord


SUPPORTED_ALGORITHMS = ("de", "pso", "cmaes", "shade")


@dataclass(frozen=True)
class OptimizerRunResult:
    trajectory_records: list[TrajectoryRecord]
    final_performance: FinalPerformanceRecord


def run_optimizer(
    *,
    algorithm: str,
    problem: Problem,
    seed: int,
    fe_total: int,
    settings: OptimizerSettings,
) -> OptimizerRunResult:
    key = algorithm.lower()
    if key not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"unsupported optimizer: {algorithm}")
    settings.validate(fe_total)
    if settings.sampling_protocol is None:
        raise ValueError("run_optimizer requires an explicit trajectory sampling protocol")
    recorder = TrajectoryRecorder(
        sampling_protocol=settings.sampling_protocol,
    )
    state = initialize_optimizer_state(
        algorithm=key,
        problem=problem,
        seed=seed,
        settings=settings,
    )
    recorder.observe(
        problem=problem,
        algorithm=key,
        seed=seed,
        fe=state.evaluations,
        fe_total=fe_total,
        native_updates=state.generation,
        population=state.population,
        fitness=state.fitness,
        best_fitness=state.best_fitness,
    )
    while state.evaluations < fe_total:
        advance_optimizer_state(
            state=state,
            problem=problem,
            fe_budget=fe_total - state.evaluations,
            on_native_update=lambda updated: recorder.observe(
                problem=problem,
                algorithm=key,
                seed=seed,
                fe=updated.evaluations,
                fe_total=fe_total,
                native_updates=updated.generation,
                population=updated.population,
                fitness=updated.fitness,
                best_fitness=updated.best_fitness,
            ),
        )
    final_performance = FinalPerformanceRecord.from_optimizer_state(
        problem_id=problem.problem_id,
        family=problem.family,
        dimension=problem.dimension,
        algorithm=key,
        seed=seed,
        fe=state.evaluations,
        fe_total=fe_total,
        native_updates=state.generation,
        best_fitness=state.best_fitness,
    )
    return OptimizerRunResult(
        trajectory_records=recorder.records,
        final_performance=final_performance,
    )
