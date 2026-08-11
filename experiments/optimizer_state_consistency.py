from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass

import numpy as np

from benchmarks import make_problem
from optimizers import (
    OptimizerSettings,
    advance_optimizer_state,
    clone_optimizer_state,
    initialize_optimizer_state,
)
from optimizers.registry import SUPPORTED_ALGORITHMS


DEFAULT_CHECKPOINT_FES = (200, 280, 320, 400, 600)


def check_optimizer_state_consistency(
    *,
    algorithms: tuple[str, ...],
    function: int,
    instance: int,
    dimension: int,
    seed: int,
    population_size: int,
    checkpoint_fes: tuple[int, ...],
) -> list[dict[str, int | str | bool]]:
    if not checkpoint_fes:
        raise ValueError("checkpoint_fes must not be empty")
    if tuple(sorted(set(checkpoint_fes))) != checkpoint_fes:
        raise ValueError("checkpoint_fes must be strictly increasing")
    if checkpoint_fes[0] < population_size:
        raise ValueError("first checkpoint FE must include the initial population")

    rows = []
    for algorithm in algorithms:
        key = str(algorithm).lower()
        if key not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"unsupported optimizer: {algorithm}")
        reference_problem = _problem(function, instance, dimension)
        restored_problem = _problem(function, instance, dimension)
        uninterrupted_problem = _problem(function, instance, dimension)
        try:
            settings = OptimizerSettings(population_size=population_size, checkpoint_ratios=(1.0,))
            reference = initialize_optimizer_state(
                algorithm=key,
                problem=reference_problem,
                seed=seed,
                settings=settings,
            )
            restored = initialize_optimizer_state(
                algorithm=key,
                problem=restored_problem,
                seed=seed,
                settings=settings,
            )
            previous_fe = population_size
            for checkpoint_fe in checkpoint_fes:
                delta = checkpoint_fe - previous_fe
                advance_optimizer_state(state=reference, problem=reference_problem, fe_budget=delta)
                advance_optimizer_state(state=restored, problem=restored_problem, fe_budget=delta)
                restored = clone_optimizer_state(restored)
                _assert_state_equal(reference, restored, f"{key} at FE={checkpoint_fe}")
                rows.append(
                    {
                        "algorithm": key,
                        "checkpoint_FE": int(checkpoint_fe),
                        "checkpoint_restore_exact": True,
                    }
                )
                previous_fe = checkpoint_fe

            uninterrupted = initialize_optimizer_state(
                algorithm=key,
                problem=uninterrupted_problem,
                seed=seed,
                settings=settings,
            )
            advance_optimizer_state(
                state=uninterrupted,
                problem=uninterrupted_problem,
                fe_budget=checkpoint_fes[-1] - population_size,
            )
            _assert_state_equal(reference, uninterrupted, f"{key} uninterrupted final state")
            rows[-1]["uninterrupted_final_exact"] = True
        finally:
            reference_problem.close()
            restored_problem.close()
            uninterrupted_problem.close()
    return rows


def _problem(function: int, instance: int, dimension: int):
    return make_problem(
        {
            "suite": "bbob",
            "function": function,
            "instance": instance,
            "dimension": dimension,
        }
    )


def _assert_state_equal(left, right, context: str) -> None:
    difference = _first_difference(left, right, "state")
    if difference is not None:
        raise ValueError(f"native optimizer-state consistency failed for {context}: {difference}")


def _first_difference(left, right, path: str) -> str | None:
    if type(left) is not type(right):
        return f"{path} type differs: {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, np.ndarray):
        if not np.array_equal(left, right, equal_nan=True):
            return f"{path} array differs"
        return None
    if is_dataclass(left):
        for field in fields(left):
            difference = _first_difference(getattr(left, field.name), getattr(right, field.name), f"{path}.{field.name}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return f"{path} keys differ"
        for key in left:
            difference = _first_difference(left[key], right[key], f"{path}[{key!r}]")
            if difference is not None:
                return difference
        return None
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return f"{path} length differs"
        for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
            difference = _first_difference(left_value, right_value, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if left != right:
        return f"{path} differs: {left!r} != {right!r}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check native optimizer-state checkpoint/restore consistency on a real BBOB problem."
    )
    parser.add_argument("--algorithm", action="append", choices=SUPPORTED_ALGORITHMS, default=None)
    parser.add_argument("--function", type=int, default=1)
    parser.add_argument("--instance", type=int, default=1)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--checkpoint-fe", type=int, action="append", default=None)
    args = parser.parse_args()
    algorithms = tuple(args.algorithm or SUPPORTED_ALGORITHMS)
    checkpoint_fes = tuple(args.checkpoint_fe or DEFAULT_CHECKPOINT_FES)
    rows = check_optimizer_state_consistency(
        algorithms=algorithms,
        function=args.function,
        instance=args.instance,
        dimension=args.dimension,
        seed=args.seed,
        population_size=args.population_size,
        checkpoint_fes=checkpoint_fes,
    )
    for row in rows:
        final = " and uninterrupted final state" if row.get("uninterrupted_final_exact") else ""
        print(
            f"{row['algorithm']} FE={row['checkpoint_FE']}: "
            f"checkpoint restore exact{final}"
        )


if __name__ == "__main__":
    main()
