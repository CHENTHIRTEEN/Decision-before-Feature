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
    trajectory_query_records: list[dict]


def run_optimizer(
    *,
    algorithm: str,
    problem: Problem,
    seed: int,
    fe_total: int,
    settings: OptimizerSettings,
    log10_gap_floor: float,
    log10_gap_cap: float,
    success_gap_target: float,
    failure_loss_cap: float,
    trajectory_query_split: str | None = None,
) -> OptimizerRunResult:
    key = algorithm.lower()
    if key not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"unsupported optimizer: {algorithm}")
    settings.validate(fe_total)
    if settings.sampling_protocol is None:
        raise ValueError("run_optimizer requires an explicit trajectory sampling protocol")
    recorder = TrajectoryRecorder(
        sampling_protocol=settings.sampling_protocol,
        trajectory_query_enabled=trajectory_query_split is not None,
        trajectory_query_split=trajectory_query_split,
    )
    if problem.reference_value is None:
        raise ValueError("formal optimizer run requires a benchmark reference value")
    evaluation_count = 0
    first_hit_fe: int | None = None

    def observe_evaluation(point, value) -> None:
        nonlocal evaluation_count, first_hit_fe
        evaluation_count += 1
        gap = max(float(value) - float(problem.reference_value), 0.0)
        if first_hit_fe is None and gap <= float(success_gap_target):
            first_hit_fe = evaluation_count
        recorder.observe_evaluation(
            problem=problem,
            algorithm=key,
            seed=seed,
            point=point,
            value=value,
        )

    state = None
    run_failure: Exception | None = None
    try:
        state = initialize_optimizer_state(
            algorithm=key,
            problem=problem,
            seed=seed,
            settings=settings,
            on_evaluation=observe_evaluation,
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
                on_evaluation=observe_evaluation,
            )
        if evaluation_count != state.evaluations:
            raise RuntimeError("per-evaluation endpoint observer count does not match optimizer FE")
    except Exception as exc:
        run_failure = exc
    if run_failure is not None:
        final_performance = FinalPerformanceRecord.from_failure(
            problem_id=problem.problem_id,
            function_id=problem.function_id,
            family=problem.family,
            cv_group_id=problem.cv_group_id,
            dimension=problem.dimension,
            algorithm=key,
            seed=seed,
            fe_total=fe_total,
            effective_fe=evaluation_count,
            native_updates=0 if state is None else state.generation,
            best_fitness=None if state is None else state.best_fitness,
            benchmark_reference_value=float(problem.reference_value),
            failure_loss_cap=failure_loss_cap,
            log10_gap_floor=log10_gap_floor,
            log10_gap_cap=log10_gap_cap,
            success_gap_target=success_gap_target,
            first_hit_fe=first_hit_fe,
            failure_type=type(run_failure).__name__,
            failure_message=str(run_failure),
        )
        return OptimizerRunResult(
            trajectory_records=recorder.records,
            final_performance=final_performance,
            trajectory_query_records=recorder.trajectory_query_records,
        )
    if state is None:
        raise RuntimeError("optimizer state is unavailable after a completed run")
    final_performance = FinalPerformanceRecord.from_optimizer_state(
        problem_id=problem.problem_id,
        function_id=problem.function_id,
        family=problem.family,
        cv_group_id=problem.cv_group_id,
        dimension=problem.dimension,
        algorithm=key,
        seed=seed,
        fe=state.evaluations,
        fe_total=fe_total,
        native_updates=state.generation,
        best_fitness=state.best_fitness,
        benchmark_reference_value=float(problem.reference_value),
        log10_gap_floor=log10_gap_floor,
        log10_gap_cap=log10_gap_cap,
        success_gap_target=success_gap_target,
        first_hit_fe=first_hit_fe,
    )
    return OptimizerRunResult(
        trajectory_records=recorder.records,
        final_performance=final_performance,
        trajectory_query_records=recorder.trajectory_query_records,
    )
