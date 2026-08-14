from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from benchmarks.bbob import make_bbob_problem
from benchmarks.cec import make_cec_problem
from benchmarks.core import Problem, coerce_reference_value


_PROBLEM_ID_PATTERNS = {
    "bbob": re.compile(r"^bbob_f(\d{3})_i(\d+)_d(\d+)$"),
    "cec2017": re.compile(r"^cec2017_f(\d{2})_d(\d+)$"),
    "cec2022": re.compile(r"^cec2022_f(\d{2})_d(\d+)$"),
}


def make_problem(config: dict) -> Problem:
    suite = str(config["suite"]).lower()
    dimension = int(config["dimension"])

    if suite == "bbob":
        return make_bbob_problem(
            function=int(config["function"]),
            dimension=dimension,
            instance=int(config["instance"]),
        )
    if suite in {"cec2017", "cec2022"}:
        return make_cec_problem(
            year=int(suite.removeprefix("cec")),
            function=int(config["function"]),
            dimension=dimension,
        )

    raise ValueError(f"unsupported benchmark suite: {suite}")


@lru_cache(maxsize=None)
def problem_bounds(problem_id: str) -> tuple[np.ndarray, np.ndarray]:
    config = _problem_config_from_id(problem_id)
    problem = make_problem(config)
    try:
        return problem.lower_bounds.copy(), problem.upper_bounds.copy()
    finally:
        problem.close()


def _problem_config_from_id(problem_id: str) -> dict[str, int | str]:
    for suite, pattern in _PROBLEM_ID_PATTERNS.items():
        match = pattern.match(problem_id)
        if match is None:
            continue
        if suite == "bbob":
            function, instance, dimension = (int(value) for value in match.groups())
            return {
                "suite": suite,
                "function": function,
                "instance": instance,
                "dimension": dimension,
            }
        function, dimension = (int(value) for value in match.groups())
        return {
            "suite": suite,
            "function": function,
            "dimension": dimension,
        }
    raise ValueError(f"unsupported problem_id: {problem_id}")
