from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import pyarrow as pa

from trajectory.records import OPTIMIZER_STATE_MODE


FINAL_PERFORMANCE_PROTOCOL = "complete_budget_native_optimizer_run"


@dataclass(frozen=True)
class FinalPerformanceRecord:
    problem_id: str
    family: str
    dimension: int
    algorithm: str
    seed: int
    FE: int
    FE_total: int
    native_updates: int
    best_fitness: float
    optimizer_state_mode: str
    final_performance_protocol: str

    @classmethod
    def from_optimizer_state(
        cls,
        *,
        problem_id: str,
        family: str,
        dimension: int,
        algorithm: str,
        seed: int,
        fe: int,
        fe_total: int,
        native_updates: int,
        best_fitness: float,
    ) -> "FinalPerformanceRecord":
        if int(fe_total) <= 0 or int(fe) != int(fe_total):
            raise ValueError("final performance must be recorded exactly at FE_total")
        if int(native_updates) < 0:
            raise ValueError("final native_updates must be non-negative")
        if not isfinite(float(best_fitness)):
            raise ValueError("final best_fitness must be finite")
        return cls(
            problem_id=str(problem_id),
            family=str(family),
            dimension=int(dimension),
            algorithm=str(algorithm),
            seed=int(seed),
            FE=int(fe),
            FE_total=int(fe_total),
            native_updates=int(native_updates),
            best_fitness=float(best_fitness),
            optimizer_state_mode=OPTIMIZER_STATE_MODE,
            final_performance_protocol=FINAL_PERFORMANCE_PROTOCOL,
        )


FINAL_PERFORMANCE_SCHEMA = pa.schema(
    [
        ("problem_id", pa.string()),
        ("family", pa.string()),
        ("dimension", pa.int32()),
        ("algorithm", pa.string()),
        ("seed", pa.int64()),
        ("FE", pa.int64()),
        ("FE_total", pa.int64()),
        ("native_updates", pa.int64()),
        ("best_fitness", pa.float64()),
        ("optimizer_state_mode", pa.string()),
        ("final_performance_protocol", pa.string()),
    ]
)
