from __future__ import annotations

import argparse

import numpy as np

from benchmarks.factory import make_problem


def main() -> None:
    parser = argparse.ArgumentParser(description="Check public benchmark adapter construction.")
    parser.add_argument("--suite", action="append", default=None)
    parser.add_argument("--function", type=int, action="append", default=None)
    parser.add_argument("--dimension", type=int, action="append", default=None)
    args = parser.parse_args()

    suites = [str(value).lower() for value in (args.suite or ["cec2017", "cec2022", "mabbob"])]
    functions = [int(value) for value in (args.function or [1])]
    dimensions = [int(value) for value in (args.dimension or [10])]
    failed = []
    for suite in suites:
        for function in functions:
            for dimension in dimensions:
                try:
                    problem = make_problem({"suite": suite, "function": function, "dimension": dimension})
                    try:
                        values = problem.evaluate(np.zeros((1, dimension)))
                        print(f"{problem.problem_id}: ok value={float(values[0])}")
                    finally:
                        problem.close()
                except Exception as exc:
                    failed.append((suite, function, dimension, exc))
                    print(f"{suite}_f{function:02d}_d{dimension}: failed {type(exc).__name__}: {exc}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
