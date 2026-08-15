from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log10

import pyarrow as pa

from trajectory.records import OPTIMIZER_STATE_MODE


FINAL_PERFORMANCE_PROTOCOL = "attempted_native_optimizer_run_with_failure_endpoints"


@dataclass(frozen=True)
class FinalPerformanceRecord:
    problem_id: str
    function_id: str
    family: str
    dimension: int
    algorithm: str
    seed: int
    FE: int
    FE_total: int
    native_updates: int
    best_fitness: float | None
    benchmark_reference_value: float | None
    final_gap: float
    log10_gap: float
    log10_gap_floor: float
    log10_gap_cap: float
    success_gap_target: float
    success: bool
    first_hit_FE: int | None
    run_status: str
    path_completed: bool
    planned_FE: int
    effective_FE: int
    observed_first_hit_FE: int | None
    target_hit_observed: bool
    target_hit_before_failure: bool
    endpoint_success: bool
    failure_type: str
    failure_message: str
    optimizer_state_mode: str
    final_performance_protocol: str
    cv_group_id: str = ""

    @classmethod
    def from_optimizer_state(
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
        best_fitness: float,
        benchmark_reference_value: float,
        log10_gap_floor: float,
        log10_gap_cap: float,
        success_gap_target: float,
        first_hit_fe: int | None,
        cv_group_id: str = "",
    ) -> "FinalPerformanceRecord":
        if int(fe_total) <= 0 or int(fe) != int(fe_total):
            raise ValueError("final performance must be recorded exactly at FE_total")
        if int(native_updates) < 0:
            raise ValueError("final native_updates must be non-negative")
        if not isfinite(float(best_fitness)):
            raise ValueError("final best_fitness must be finite")
        reference = float(benchmark_reference_value)
        floor = float(log10_gap_floor)
        cap = float(log10_gap_cap)
        target = float(success_gap_target)
        if not all(isfinite(value) for value in (reference, floor, cap, target)):
            raise ValueError("final performance endpoint values must be finite")
        if not 0.0 < floor < target < cap:
            raise ValueError("final performance endpoints must satisfy 0 < floor < target < cap")
        hit = None if first_hit_fe is None else int(first_hit_fe)
        if hit is not None and not 1 <= hit <= int(fe_total):
            raise ValueError("first_hit_FE must lie in [1, FE_total]")
        final_gap = max(float(best_fitness) - reference, 0.0)
        success = final_gap <= target
        if success != (hit is not None):
            raise ValueError("success and first_hit_FE are inconsistent")
        clipped_gap = min(max(final_gap, floor), cap)
        return cls(
            problem_id=str(problem_id),
            function_id=str(function_id),
            family=str(family),
            dimension=int(dimension),
            algorithm=str(algorithm),
            seed=int(seed),
            FE=int(fe),
            FE_total=int(fe_total),
            native_updates=int(native_updates),
            best_fitness=float(best_fitness),
            benchmark_reference_value=reference,
            final_gap=float(final_gap),
            log10_gap=float(log10(clipped_gap)),
            log10_gap_floor=floor,
            log10_gap_cap=cap,
            success=bool(success),
            success_gap_target=target,
            first_hit_FE=hit,
            run_status="completed",
            path_completed=True,
            planned_FE=int(fe_total),
            effective_FE=int(fe),
            observed_first_hit_FE=hit,
            target_hit_observed=bool(hit is not None),
            target_hit_before_failure=False,
            endpoint_success=bool(hit is not None),
            failure_type="",
            failure_message="",
            optimizer_state_mode=OPTIMIZER_STATE_MODE,
            final_performance_protocol=FINAL_PERFORMANCE_PROTOCOL,
            cv_group_id=str(cv_group_id or function_id),
        )

    @classmethod
    def from_failure(
        cls,
        *,
        problem_id: str,
        function_id: str,
        family: str,
        dimension: int,
        algorithm: str,
        seed: int,
        fe_total: int,
        effective_fe: int,
        native_updates: int,
        best_fitness: float | None,
        benchmark_reference_value: float | None,
        failure_loss_cap: float,
        log10_gap_floor: float,
        log10_gap_cap: float,
        success_gap_target: float,
        first_hit_fe: int | None,
        failure_type: str,
        failure_message: str,
        cv_group_id: str = "",
    ) -> "FinalPerformanceRecord":
        planned = int(fe_total)
        effective = int(effective_fe)
        updates = int(native_updates)
        cap = float(failure_loss_cap)
        floor = float(log10_gap_floor)
        gap_cap = float(log10_gap_cap)
        target = float(success_gap_target)
        if planned <= 0 or effective < 0 or effective > planned:
            raise ValueError("failed run planned/effective FE accounting is inconsistent")
        if updates < 0:
            raise ValueError("failed run native_updates must be non-negative")
        if not all(isfinite(value) for value in (cap, floor, gap_cap, target)):
            raise ValueError("failed run endpoint values must be finite")
        if cap != gap_cap or not 0.0 < floor < target < gap_cap:
            raise ValueError("failed run endpoint bounds are inconsistent")
        best = None if best_fitness is None else float(best_fitness)
        reference = (
            None
            if benchmark_reference_value is None
            else float(benchmark_reference_value)
        )
        if best is not None and not isfinite(best):
            best = None
        if reference is not None and not isfinite(reference):
            reference = None
        hit = None if first_hit_fe is None else int(first_hit_fe)
        if hit is not None and not 1 <= hit <= max(effective, 1):
            raise ValueError("failed run first_hit_FE lies outside its observed evaluations")
        message = str(failure_message)[:500]
        failure_name = str(failure_type)
        if not failure_name:
            raise ValueError("failed run requires a failure_type")
        return cls(
            problem_id=str(problem_id),
            function_id=str(function_id),
            family=str(family),
            dimension=int(dimension),
            algorithm=str(algorithm),
            seed=int(seed),
            FE=planned,
            FE_total=planned,
            native_updates=updates,
            best_fitness=best,
            benchmark_reference_value=reference,
            final_gap=cap,
            log10_gap=float(log10(gap_cap)),
            log10_gap_floor=floor,
            log10_gap_cap=gap_cap,
            success_gap_target=target,
            success=bool(hit is not None),
            first_hit_FE=hit,
            run_status="failed",
            path_completed=False,
            planned_FE=planned,
            effective_FE=effective,
            observed_first_hit_FE=hit,
            target_hit_observed=bool(hit is not None),
            target_hit_before_failure=bool(hit is not None),
            endpoint_success=False,
            failure_type=failure_name,
            failure_message=message,
            optimizer_state_mode=OPTIMIZER_STATE_MODE,
            final_performance_protocol=FINAL_PERFORMANCE_PROTOCOL,
            cv_group_id=str(cv_group_id or function_id),
        )


FINAL_PERFORMANCE_SCHEMA = pa.schema(
    [
        ("problem_id", pa.string()),
        ("function_id", pa.string()),
        ("family", pa.string()),
        ("cv_group_id", pa.string()),
        ("dimension", pa.int32()),
        ("algorithm", pa.string()),
        ("seed", pa.int64()),
        ("FE", pa.int64()),
        ("FE_total", pa.int64()),
        ("native_updates", pa.int64()),
        ("best_fitness", pa.float64()),
        ("benchmark_reference_value", pa.float64()),
        ("final_gap", pa.float64()),
        ("log10_gap", pa.float64()),
        ("log10_gap_floor", pa.float64()),
        ("log10_gap_cap", pa.float64()),
        ("success_gap_target", pa.float64()),
        ("success", pa.bool_()),
        ("first_hit_FE", pa.int64()),
        ("run_status", pa.string()),
        ("path_completed", pa.bool_()),
        ("planned_FE", pa.int64()),
        ("effective_FE", pa.int64()),
        ("observed_first_hit_FE", pa.int64()),
        ("target_hit_observed", pa.bool_()),
        ("target_hit_before_failure", pa.bool_()),
        ("endpoint_success", pa.bool_()),
        ("failure_type", pa.string()),
        ("failure_message", pa.string()),
        ("optimizer_state_mode", pa.string()),
        ("final_performance_protocol", pa.string()),
    ]
)
