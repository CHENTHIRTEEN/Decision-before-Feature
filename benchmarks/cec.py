from __future__ import annotations

import importlib
import warnings

import numpy as np

from benchmarks.core import Problem, coerce_reference_value


SUPPORTED_CEC_YEARS = {2017, 2022}


def _load_cec_class(year: int, function: int):
    if year not in SUPPORTED_CEC_YEARS:
        raise ValueError(f"unsupported CEC year: {year}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")
        module = importlib.import_module(f"opfunu.cec_based.cec{year}")

    class_name = f"F{function}{year}"
    function_class = getattr(module, class_name, None)
    if function_class is None:
        raise ValueError(f"OPFUNU class {class_name} is not available")
    return function_class


def make_cec_problem(year: int, function: int, dimension: int) -> Problem:
    function_class = _load_cec_class(year, function)
    default_function = function_class()
    supported_dimensions = getattr(default_function, "dim_supported", None)
    if supported_dimensions is not None and dimension not in supported_dimensions:
        raise ValueError(
            f"CEC{year} F{function} supports dimensions {supported_dimensions}, got {dimension}"
        )

    function_object = function_class(ndim=dimension)
    bounds = np.column_stack(
        [
            np.asarray(function_object.lb, dtype=float),
            np.asarray(function_object.ub, dtype=float),
        ]
    )
    problem_id = f"cec{year}_f{function:02d}_d{dimension}"

    def objective(population: np.ndarray) -> np.ndarray:
        return np.asarray([float(function_object.evaluate(row)) for row in population], dtype=float)

    return Problem(
        problem_id=problem_id,
        function_id=f"cec{year}_f{function:02d}",
        family=f"cec{year}_unassigned_landscape_family",
        dimension=dimension,
        suite_code=2 if int(year) == 2017 else 3,
        function_number=int(function),
        instance_number=1,
        bounds=bounds,
        objective=objective,
        reference_value=coerce_reference_value(function_object, ("bias", "fopt", "best_value", "optimum")),
        cv_group_id=f"cec{year}_f{function:02d}",
        boundary_handling="reflect",
    )
