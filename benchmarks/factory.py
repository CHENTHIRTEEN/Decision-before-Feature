from __future__ import annotations

from benchmarks.bbob import make_bbob_problem
from benchmarks.cec import make_cec_problem
from benchmarks.core import Problem


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
