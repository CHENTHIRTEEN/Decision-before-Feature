from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.core import Problem
from benchmarks.factory import make_problem
from optimizers import OptimizerSettings, advance_optimizer_state, clone_optimizer_state, initialize_optimizer_state
from optimizers.registry import SUPPORTED_ALGORITHMS
from trajectory.final_performance import FinalPerformanceRecord


DEFAULT_ALGORITHMS = SUPPORTED_ALGORITHMS
DEFAULT_SEEDS = (1, 2, 3)
DEFAULT_FUNCTIONS = (1, 2, 3)
DEFAULT_DIMENSIONS = (10,)
DEFAULT_INSTANCE = 1
DEFAULT_POPULATION_SIZE = 40
DEFAULT_FE_TOTAL_BY_DIMENSION = {10: 1000}
DEFAULT_CHECKPOINT_FES = (40, 80, 120, 200, 400, 1000)


class ReferenceOptimizerFactory:
    def __init__(self, name: str) -> None:
        self.name = name

    def supported(self) -> bool:
        return self.name in SUPPORTED_ALGORITHMS

    def label(self) -> str:
        return self.name



def run_algorithm_consistency_check(
    *,
    suite: str,
    algorithms: tuple[str, ...],
    functions: tuple[int, ...],
    dimensions: tuple[int, ...],
    seeds: tuple[int, ...],
    instance: int,
    population_size: int,
    checkpoint_fes: tuple[int, ...],
    max_runs: int | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    supported = tuple(a.lower() for a in algorithms)
    if tuple(sorted(set(supported))) != tuple(sorted(set(SUPPORTED_ALGORITHMS))):
        raise ValueError(f"check must cover the four supported algorithms: {SUPPORTED_ALGORITHMS}")

    run_count = 0
    for function in functions:
        for dimension in dimensions:
            fe_total = _fe_total_for_dimension(dimension)
            for seed in seeds:
                for algorithm in supported:
                    if max_runs is not None and run_count >= max_runs:
                        break
                    run_count += 1
                    row = _check_single_run(
                        suite=suite,
                        function=function,
                        instance=instance,
                        dimension=dimension,
                        seed=seed,
                        population_size=population_size,
                        fe_total=fe_total,
                        checkpoint_fes=checkpoint_fes,
                        algorithm=algorithm,
                    )
                    rows.append(row)
                if max_runs is not None and run_count >= max_runs:
                    break
            if max_runs is not None and run_count >= max_runs:
                break
        if max_runs is not None and run_count >= max_runs:
            break

    if not rows:
        raise ValueError("consistency check produced no rows")
    return {
        "suite": suite,
        "algorithms": list(sorted(set(supported))),
        "rows": rows,
        "summary": _summarize_rows(rows),
    }



def _check_single_run(
    *,
    suite: str,
    function: int,
    instance: int,
    dimension: int,
    seed: int,
    population_size: int,
    fe_total: int,
    checkpoint_fes: tuple[int, ...],
    algorithm: str,
) -> dict[str, Any]:
    problem = _make_problem(suite=suite, function=function, instance=instance, dimension=dimension)
    try:
        settings = OptimizerSettings(population_size=population_size, checkpoint_ratios=(1.0,))
        reference = initialize_optimizer_state(
            algorithm=algorithm,
            problem=problem,
            seed=seed,
            settings=settings,
        )
        clone = clone_optimizer_state(reference)
        _assert_state_equal(reference, clone, f"{algorithm} clone at FE={population_size}")

        previous_fe = population_size
        checkpoint_records: list[dict[str, Any]] = []
        uninterrupted = initialize_optimizer_state(
            algorithm=algorithm,
            problem=problem,
            seed=seed,
            settings=settings,
        )
        for checkpoint_fe in checkpoint_fes:
            if checkpoint_fe < previous_fe:
                raise ValueError("checkpoint_fes must be increasing")
            delta = checkpoint_fe - previous_fe
            if delta:
                advance_optimizer_state(state=reference, problem=problem, fe_budget=delta)
                advance_optimizer_state(state=uninterrupted, problem=problem, fe_budget=delta)
            restored = clone_optimizer_state(reference)
            advance_optimizer_state(state=restored, problem=problem, fe_budget=0)
            _assert_state_equal(reference, restored, f"{algorithm} restore at FE={checkpoint_fe}")
            _assert_state_equal(reference, uninterrupted, f"{algorithm} uninterrupted at FE={checkpoint_fe}")
            checkpoint_records.append(
                {
                    "checkpoint_fe": int(checkpoint_fe),
                    "clone_exact": True,
                    "restore_exact": True,
                    "uninterrupted_exact": True,
                }
            )
            previous_fe = checkpoint_fe

        state = initialize_optimizer_state(
            algorithm=algorithm,
            problem=problem,
            seed=seed,
            settings=settings,
        )
        advance_optimizer_state(state=state, problem=problem, fe_budget=fe_total - population_size)
        final_performance = FinalPerformanceRecord.from_optimizer_state(
            problem_id=problem.problem_id,
            function_id=problem.function_id,
            family=problem.family,
            dimension=problem.dimension,
            algorithm=algorithm,
            seed=seed,
            fe=fe_total,
            fe_total=fe_total,
            native_updates=getattr(state, "generation", 0),
            best_fitness=float(state.best_fitness),
            benchmark_reference_value=float(problem.reference_value),
            log10_gap_floor=1e-12,
            log10_gap_cap=1e20,
            success_gap_target=1e-8,
            first_hit_fe=population_size,
        )
        return {
            "algorithm": algorithm,
            "function": int(function),
            "instance": int(instance),
            "dimension": int(dimension),
            "seed": int(seed),
            "checkpoint_records": checkpoint_records,
            "final_performance": asdict(final_performance),
        }
    finally:
        problem.close()



def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    algorithms = sorted({row["algorithm"] for row in rows})
    by_algorithm: dict[str, dict[str, Any]] = {}
    for algorithm in algorithms:
        algo_rows = [row for row in rows if row["algorithm"] == algorithm]
        by_algorithm[algorithm] = {
            "runs": len(algo_rows),
            "all_clone_exact": all(all(check["clone_exact"] for check in row["checkpoint_records"]) for row in algo_rows),
            "all_restore_exact": all(all(check["restore_exact"] for check in row["checkpoint_records"]) for row in algo_rows),
            "all_uninterrupted_exact": all(all(check["uninterrupted_exact"] for check in row["checkpoint_records"]) for row in algo_rows),
        }
    return {
        "rows": len(rows),
        "algorithms": algorithms,
        "by_algorithm": by_algorithm,
    }



def _make_problem(*, suite: str, function: int, instance: int, dimension: int) -> Problem:
    if suite.lower() == "bbob":
        return make_problem(
            {
                "suite": "bbob",
                "function": function,
                "instance": instance,
                "dimension": dimension,
            }
        )
    raise ValueError("consistency check currently targets BBOB-style problems")



def _assert_state_equal(left: Any, right: Any, context: str) -> None:
    if type(left) is not type(right):
        raise ValueError(f"{context}: state types differ: {type(left).__name__} != {type(right).__name__}")
    if hasattr(left, "__dict__"):
        for key in left.__dict__:
            lvalue = getattr(left, key)
            rvalue = getattr(right, key)
            if isinstance(lvalue, np.ndarray):
                if not np.array_equal(lvalue, rvalue, equal_nan=True):
                    raise ValueError(f"{context}: ndarray field {key} differs")
            elif isinstance(lvalue, (list, tuple)):
                if len(lvalue) != len(rvalue):
                    raise ValueError(f"{context}: sequence field {key} length differs")
            elif lvalue != rvalue:
                raise ValueError(f"{context}: field {key} differs: {lvalue!r} != {rvalue!r}")
        return
    raise TypeError(f"unsupported state type for consistency comparison: {type(left).__name__}")



def _fe_total_for_dimension(dimension: int) -> int:
    return int(DEFAULT_FE_TOTAL_BY_DIMENSION.get(int(dimension), 1000 * int(dimension)))



def main() -> None:
    parser = argparse.ArgumentParser(description="Run the four-algorithm consistency check.")
    parser.add_argument("--suite", default="bbob")
    parser.add_argument("--algorithm", action="append", default=None)
    parser.add_argument("--function", type=int, action="append", default=None)
    parser.add_argument("--dimension", type=int, action="append", default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--instance", type=int, default=DEFAULT_INSTANCE)
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--checkpoint-fe", type=int, action="append", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    algorithms = tuple(args.algorithm or DEFAULT_ALGORITHMS)
    functions = tuple(args.function or DEFAULT_FUNCTIONS)
    dimensions = tuple(args.dimension or DEFAULT_DIMENSIONS)
    seeds = tuple(args.seed or DEFAULT_SEEDS)
    checkpoint_fes = tuple(args.checkpoint_fe or DEFAULT_CHECKPOINT_FES)

    result = run_algorithm_consistency_check(
        suite=args.suite,
        algorithms=algorithms,
        functions=functions,
        dimensions=dimensions,
        seeds=seeds,
        instance=args.instance,
        population_size=args.population_size,
        checkpoint_fes=checkpoint_fes,
        max_runs=args.max_runs,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload, encoding="utf-8")
        print(f"wrote consistency check to {args.output}")


if __name__ == "__main__":
    main()
