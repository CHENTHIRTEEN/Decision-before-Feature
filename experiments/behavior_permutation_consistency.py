from __future__ import annotations

import argparse

import numpy as np

from behavior.features import (
    BEHAVIOR_FEATURE_COLUMNS,
    BEHAVIOR_WINDOW_METADATA_COLUMNS,
    extract_behavior_rows,
)
from behavior.validation import validate_behavior_rows
from benchmarks import make_problem
from optimizers import OptimizerSettings, run_optimizer
from optimizers.registry import SUPPORTED_ALGORITHMS


DEFAULT_CHECKPOINT_RATIOS = (0.20, 0.25, 0.28, 0.30, 0.35, 0.40)
PERMUTATION_STREAM_CODE = 20260812


def check_population_permutation_consistency(
    *,
    algorithms: tuple[str, ...],
    function: int,
    instance: int,
    dimension: int,
    seed: int,
    population_size: int,
    fe_total: int,
) -> list[dict[str, int | str | bool]]:
    settings = OptimizerSettings(
        population_size=population_size,
        checkpoint_ratios=DEFAULT_CHECKPOINT_RATIOS,
    )
    compared_columns = BEHAVIOR_WINDOW_METADATA_COLUMNS + BEHAVIOR_FEATURE_COLUMNS
    summaries = []
    for algorithm_code, algorithm in enumerate(algorithms, start=1):
        problem = make_problem(
            {
                "suite": "bbob",
                "function": function,
                "instance": instance,
                "dimension": dimension,
            }
        )
        try:
            records = run_optimizer(
                algorithm=algorithm,
                problem=problem,
                seed=seed,
                fe_total=fe_total,
                settings=settings,
            )
        finally:
            problem.close()

        trajectory_rows = [record.__dict__.copy() for record in records]
        original = extract_behavior_rows([row.copy() for row in trajectory_rows])
        validate_behavior_rows(trajectory_rows, original)

        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), PERMUTATION_STREAM_CODE, int(algorithm_code)])
        )
        permuted_rows = []
        for row in trajectory_rows:
            permuted = row.copy()
            order = rng.permutation(len(row["population"]))
            permuted["population"] = np.asarray(row["population"], dtype=float)[order].tolist()
            permuted["fitness"] = np.asarray(row["fitness"], dtype=float)[order].tolist()
            permuted_rows.append(permuted)

        reordered = extract_behavior_rows([row.copy() for row in permuted_rows])
        validate_behavior_rows(permuted_rows, reordered)
        _assert_behavior_equal(original, reordered, compared_columns, algorithm)
        _assert_strict_native_update_windows(trajectory_rows, algorithm, population_size)
        summaries.append(
            {
                "algorithm": algorithm,
                "checkpoints": len(original),
                "compared_columns": len(compared_columns),
                "permutation_invariant": True,
            }
        )
    return summaries


def _assert_strict_native_update_windows(
    trajectory_rows: list[dict],
    algorithm: str,
    population_size: int,
) -> None:
    nominal_ratios = {"w02": 0.02, "w05": 0.05, "w10": 0.10}
    for row in trajectory_rows:
        history_fes = {int(item["FE"]) for item in row["native_update_history"]}
        for statistic in row["window_statistics"]:
            suffix = str(statistic["suffix"])
            target_span = int(round(nominal_ratios[suffix] * int(row["FE_total"])))
            actual_span = int(row["FE"]) - int(statistic["anchor_FE"])
            if int(statistic["anchor_FE"]) not in history_fes:
                raise ValueError(f"{algorithm} {suffix} anchor is not a recorded native update")
            if not target_span <= actual_span < target_span + population_size:
                raise ValueError(f"{algorithm} {suffix} is not aligned to the strict native-update window")


def _assert_behavior_equal(
    original: list[dict],
    reordered: list[dict],
    columns: tuple[str, ...],
    algorithm: str,
) -> None:
    if len(original) != len(reordered):
        raise ValueError(f"{algorithm} behavior row count changed after population permutation")
    for row_index, (left, right) in enumerate(zip(original, reordered, strict=True)):
        for column in columns:
            if left[column] != right[column]:
                raise ValueError(
                    f"population permutation changed {algorithm} row={row_index} column={column}: "
                    f"{left[column]!r} != {right[column]!r}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check behavior invariance to independent checkpoint population-row permutations on real BBOB."
    )
    parser.add_argument("--algorithm", action="append", choices=SUPPORTED_ALGORITHMS, default=None)
    parser.add_argument("--function", type=int, default=1)
    parser.add_argument("--instance", type=int, default=1)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--fe-total", type=int, default=2000)
    args = parser.parse_args()
    summaries = check_population_permutation_consistency(
        algorithms=tuple(args.algorithm or SUPPORTED_ALGORITHMS),
        function=args.function,
        instance=args.instance,
        dimension=args.dimension,
        seed=args.seed,
        population_size=args.population_size,
        fe_total=args.fe_total,
    )
    for summary in summaries:
        print(
            f"{summary['algorithm']}: {summary['checkpoints']} checkpoints, "
            f"{summary['compared_columns']} behavior/window columns unchanged after independent row permutations"
        )


if __name__ == "__main__":
    main()
