from __future__ import annotations

import numpy as np
from cocoex import Suite

from benchmarks.core import Problem


def make_bbob_problem(function: int, dimension: int, instance: int) -> Problem:
    suite_options = f"function_indices:{function} dimensions:{dimension} instance_indices:{instance}"
    suite = Suite("bbob", "", suite_options)
    coco_problem = suite.get_problem_by_function_dimension_instance(function, dimension, instance)
    if coco_problem is None:
        raise ValueError(
            f"COCO BBOB problem not found for function={function}, dimension={dimension}, instance={instance}"
        )

    bounds = np.column_stack(
        [
            np.asarray(coco_problem.lower_bounds, dtype=float),
            np.asarray(coco_problem.upper_bounds, dtype=float),
        ]
    )
    problem_id = f"bbob_f{function:03d}_i{instance:02d}_d{dimension}"

    def objective(population: np.ndarray) -> np.ndarray:
        return np.asarray([float(coco_problem(row)) for row in population], dtype=float)

    def close_problem() -> None:
        coco_problem.free()

    return Problem(
        problem_id=problem_id,
        family=f"bbob_f{function:03d}",
        dimension=dimension,
        bounds=bounds,
        objective=objective,
        close_callback=close_problem,
    )
