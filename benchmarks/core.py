from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


Objective = Callable[[np.ndarray], np.ndarray]
CloseCallback = Callable[[], None]


@dataclass(frozen=True)
class Problem:
    problem_id: str
    function_id: str
    family: str
    dimension: int
    suite_code: int
    function_number: int
    instance_number: int
    bounds: np.ndarray
    objective: Objective
    reference_value: float | None = None
    close_callback: CloseCallback | None = None

    def __post_init__(self) -> None:
        if not str(self.problem_id) or not str(self.function_id) or not str(self.family):
            raise ValueError("problem_id, function_id, and family must be non-empty")
        for name in ("suite_code", "function_number", "instance_number"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, value)
        bounds = np.asarray(self.bounds, dtype=float)
        if bounds.shape != (self.dimension, 2):
            raise ValueError("bounds must have shape (dimension, 2)")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("each lower bound must be smaller than the upper bound")
        if self.reference_value is not None:
            reference_value = float(self.reference_value)
            if not np.isfinite(reference_value):
                raise ValueError("reference_value must be finite when provided")
            object.__setattr__(self, "reference_value", reference_value)
        object.__setattr__(self, "bounds", bounds)

    @property
    def lower_bounds(self) -> np.ndarray:
        return self.bounds[:, 0]

    @property
    def upper_bounds(self) -> np.ndarray:
        return self.bounds[:, 1]

    def evaluate(self, population: np.ndarray) -> np.ndarray:
        values = np.asarray(population, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[1] != self.dimension:
            raise ValueError("population dimension does not match problem dimension")
        fitness = np.asarray(self.objective(values), dtype=float).reshape(-1)
        if fitness.shape[0] != values.shape[0]:
            raise ValueError("objective must return one fitness value per population row")
        return fitness

    def close(self) -> None:
        if self.close_callback is not None:
            self.close_callback()


def coerce_reference_value(source: object, attr_names: tuple[str, ...]) -> float | None:
    for attr_name in attr_names:
        if not hasattr(source, attr_name):
            continue
        value = getattr(source, attr_name)
        if value is None:
            continue
        try:
            reference_value = float(np.asarray(value).reshape(()))
        except (TypeError, ValueError):
            continue
        if np.isfinite(reference_value):
            return reference_value
    return None
