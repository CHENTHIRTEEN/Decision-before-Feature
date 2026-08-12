from __future__ import annotations

from benchmarks.core import Problem
from optimizers.state import advance_optimizer_state, initialize_optimizer_state
from optimizers.settings import OptimizerSettings
from trajectory.recorder import TrajectoryRecorder


SUPPORTED_ALGORITHMS = ("de", "pso", "cmaes", "shade")


def run_optimizer(
    *,
    algorithm: str,
    problem: Problem,
    seed: int,
    fe_total: int,
    settings: OptimizerSettings,
) -> list:
    key = algorithm.lower()
    if key not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"unsupported optimizer: {algorithm}")
    settings.validate(fe_total)
    recorder = TrajectoryRecorder(settings.checkpoint_ratios)
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
    return recorder.records
