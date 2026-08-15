from __future__ import annotations

import numpy as np
from cocoex import BareProblem, Suite

from benchmarks.core import Problem, coerce_reference_value


BBOB_FUNCTION_FAMILY_PROTOCOL = "bbob_five_landscape_groups_v1"
BBOB_FUNCTION_FAMILY_RANGES = (
    (range(1, 6), "bbob_separable_f01_f05"),
    (range(6, 10), "bbob_low_or_moderate_conditioning_f06_f09"),
    (range(10, 15), "bbob_high_conditioning_unimodal_f10_f14"),
    (range(15, 20), "bbob_multimodal_adequate_global_structure_f15_f19"),
    (range(20, 25), "bbob_multimodal_weak_global_structure_f20_f24"),
)


def bbob_function_id(function: int) -> str:
    value = int(function)
    if not 1 <= value <= 24:
        raise ValueError(f"BBOB function must lie in [1, 24], got {value}")
    return f"bbob_f{value:03d}"


def bbob_landscape_family(function: int) -> str:
    value = int(function)
    for function_range, family in BBOB_FUNCTION_FAMILY_RANGES:
        if value in function_range:
            return family
    raise ValueError(f"BBOB function must lie in [1, 24], got {value}")


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

    reference_value = coerce_reference_value(
        coco_problem,
        ("fopt", "reference_value", "best_value"),
    )
    if reference_value is None:
        reference_problem = BareProblem("bbob", function, dimension, instance)
        reference_value = float(reference_problem.best_value())

    return Problem(
        problem_id=problem_id,
        function_id=bbob_function_id(function),
        family=bbob_landscape_family(function),
        dimension=dimension,
        suite_code=1,
        function_number=int(function),
        instance_number=int(instance),
        bounds=bounds,
        objective=objective,
        reference_value=reference_value,
        close_callback=close_problem,
    )
