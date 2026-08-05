from __future__ import annotations

import argparse

import numpy as np

from benchmarks.factory import make_problem


def main() -> None:
    parser = argparse.ArgumentParser(description="Check public benchmark adapter construction.")
    parser.add_argument("--dimension", type=int, default=10)
    args = parser.parse_args()

    for suite in ("cec2017", "cec2022"):
        problem = make_problem({"suite": suite, "function": 1, "dimension": args.dimension})
        values = problem.evaluate(np.zeros((1, args.dimension)))
        print(f"{problem.problem_id}: {float(values[0])}")


if __name__ == "__main__":
    main()
