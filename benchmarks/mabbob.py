from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from benchmarks.bbob import make_bbob_problem
from benchmarks.core import Problem


MABBOB_FUNCTION_FAMILY_PROTOCOL = "mabbob_affine_combination_v1"
MABBOB_FUNCTION_FAMILY = "mabbob_affine_combination"

BBOB_TRAIN_FUNCTIONS = (
    1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 15, 16, 17, 18, 20, 21, 22, 23
)
BBOB_VALIDATION_FUNCTIONS = (5, 9, 13, 14, 19, 24)
ALL_BBOB_FUNCTIONS = tuple(range(1, 25))

DEFAULT_MABBOB_SCALES = (
    11.0, 17.5, 12.3, 12.6, 11.5, 15.3, 12.1, 15.3,
    15.2, 17.4, 13.4, 20.4, 12.9, 10.4, 12.3, 10.3,
    9.8, 10.6, 10.0, 14.7, 10.7, 10.8, 9.0, 12.1,
)


@dataclass(frozen=True)
class MABBOBCandidate:
    candidate_id: int
    components: tuple[int, ...]
    weights: tuple[float, ...]
    instances: tuple[int, ...]
    xopt_seed: int
    weight_seed: int
    scale_factors: tuple[float, ...] = DEFAULT_MABBOB_SCALES


@dataclass(frozen=True)
class MABBOBConfig:
    candidate_id: int
    dimension: int
    instance: int = 1
    xopt_seed: int | None = None
    weight_seed: int | None = None
    components: tuple[int, ...] = BBOB_TRAIN_FUNCTIONS
    scales: tuple[float, ...] = DEFAULT_MABBOB_SCALES


class _FallbackManyAffine:
    def __init__(
        self,
        candidate_id: int,
        n_variables: int,
        components: tuple[int, ...],
        weights: tuple[float, ...],
        instances: tuple[int, ...],
        xopt: np.ndarray,
        scale_factors: tuple[float, ...],
    ) -> None:
        self.candidate_id = int(candidate_id)
        self.n_variables = int(n_variables)
        self.components = components
        self.weights = weights
        self.instances = instances
        self.xopt = np.asarray(xopt, dtype=float)
        self.scale_factors = scale_factors
        self.bounds = np.column_stack(
            [np.full(self.n_variables, -5.0, dtype=float), np.full(self.n_variables, 5.0, dtype=float)]
        )
        self.problem_id = f"mabbob_c{self.candidate_id:03d}_i01_d{self.n_variables}"
        self._reference_value = 0.0
        self._bbob_problems: list[Any] = []
        for fi, inst in zip(self.components, self.instances):
            try:
                self._bbob_problems.append(
                    make_bbob_problem(function=fi, dimension=self.n_variables, instance=inst)
                )
            except Exception:
                self._bbob_problems.append(None)

    def __call__(self, x: np.ndarray) -> float:
        values = np.asarray(x, dtype=float).reshape(-1)
        result = 0.0
        for idx, (fi, w, inst, scale) in enumerate(
            zip(self.components, self.weights, self.instances, self.scale_factors)
        ):
            component = self._bbob_problems[idx]
            if w <= 0.0 or component is None:
                continue
            optimum = getattr(component, "optimum", None)
            xopt = np.asarray(getattr(optimum, "x", np.zeros(self.n_variables)), dtype=float).reshape(-1)
            shift = values + xopt - self.xopt
            evaluation = component(shift.tolist())
            raw_value = float(np.asarray(evaluation).reshape(-1)[0])
            reference_value = float(getattr(component, "reference_value", 0.0) or 0.0)
            f0 = np.clip(raw_value - reference_value, 1e-12, 1e20)
            f0 = (np.log10(f0) + 8.0) / scale
            result += f0 * w
        return float(10.0 ** (10.0 * result - 8.0))

    @property
    def optimum(self) -> Any:
        class _Optimum:
            def __init__(self, dimension: int) -> None:
                self.x = np.zeros(dimension, dtype=float)
                self.y = 0.0
        return _Optimum(self.n_variables)

    def close(self) -> None:
        for p in self._bbob_problems:
            if p is not None:
                p.close()


try:
    from ioh import problem as ioh_problem_module
except Exception:
    ioh_problem_module = None


def candidate_definition(candidate_id: int) -> tuple[tuple[int, ...], np.ndarray]:
    candidate = int(candidate_id)
    if candidate < 1 or candidate > 200:
        raise ValueError(f"candidate_id must be in [1, 200], got {candidate}")

    pairs = [
        (left, right)
        for index, left in enumerate(BBOB_TRAIN_FUNCTIONS)
        for right in BBOB_TRAIN_FUNCTIONS[index + 1 :]
    ]
    triples: list[tuple[int, int, int]] = []
    for first_index, first in enumerate(BBOB_TRAIN_FUNCTIONS):
        for second_index in range(first_index + 1, len(BBOB_TRAIN_FUNCTIONS)):
            for third in BBOB_TRAIN_FUNCTIONS[second_index + 1 :]:
                triples.append((first, BBOB_TRAIN_FUNCTIONS[second_index], third))

    def pair_weights(left: int, right: int, alpha: float) -> np.ndarray:
        weights = np.zeros(24, dtype=float)
        weights[left - 1] = alpha
        weights[right - 1] = 1.0 - alpha
        return weights

    def triple_weights(triple: tuple[int, int, int], profile: tuple[float, float, float]) -> np.ndarray:
        weights = np.zeros(24, dtype=float)
        for index, component in enumerate(triple):
            weights[component - 1] = profile[index]
        return weights

    if candidate <= 90:
        pair_index, alpha_index = divmod(candidate - 1, 3)
        left, right = pairs[pair_index % len(pairs)]
        return (left, right), pair_weights(left, right, (0.2, 0.5, 0.8)[alpha_index])

    if candidate <= 100:
        triple_index, dominant_index = divmod(candidate - 91, 3)
        triple = triples[triple_index % len(triples)]
        profiles = ((0.5, 0.3, 0.2), (0.2, 0.5, 0.3), (0.3, 0.2, 0.5))
        return triple, triple_weights(triple, profiles[dominant_index])

    if candidate <= 190:
        pair_index, alpha_index = divmod(candidate - 101, 3)
        left, right = pairs[(pair_index + 30) % len(pairs)]
        return (left, right), pair_weights(left, right, (0.15, 0.45, 0.85)[alpha_index])

    triple_index, dominant_index = divmod(candidate - 191, 3)
    triple = triples[(triple_index + 20) % len(triples)]
    profiles = ((0.45, 0.35, 0.20), (0.20, 0.45, 0.35), (0.35, 0.20, 0.45))
    return triple, triple_weights(triple, profiles[dominant_index])


def _make_ioh_many_affine(config: MABBOBConfig):
    if ioh_problem_module is None:
        return None
    try:
        return ioh_problem_module.ManyAffine(int(config.instance), int(config.dimension))
    except Exception:
        return None


def _random_weights(candidate_id: int, size: int = 24, threshold: float = 0.85) -> np.ndarray:
    rng = np.random.default_rng(2000 + int(candidate_id))
    weights = rng.uniform(0.0, 1.0, size=size)
    order = np.argsort(weights)
    top_two = order[-2:]
    weights[top_two] = np.maximum(weights[top_two], threshold)
    cutoff = min(float(threshold), float(np.partition(weights, -3)[-3]))
    weights = np.where(weights >= cutoff, weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        weights[:] = 0.0
        weights[top_two] = 0.5
        total = float(np.sum(weights))
    return weights / total


def _make_controlled_many_affine(config: MABBOBConfig) -> Any:
    if ioh_problem_module is None:
        return None
    try:
        xopt = np.random.default_rng(int(config.xopt_seed or config.instance)).uniform(
            low=-5.0, high=5.0, size=(config.dimension,)
        )
        weights = _random_weights(int(config.candidate_id), size=24)
        instances = tuple(int(config.instance) for _ in range(24))
        return ioh_problem_module.ManyAffine(
            xopt=xopt.tolist(),
            weights=weights.tolist(),
            instances=list(instances),
            n_variables=config.dimension,
            scale_factors=list(DEFAULT_MABBOB_SCALES),
        )
    except Exception:
        return None


def _candidate_instances(candidate_id: int) -> tuple[int, ...]:
    return tuple(1 for _ in range(24))


def _candidate_components(candidate_id: int) -> tuple[int, ...]:
    components, _ = candidate_definition(candidate_id)
    return components


def _candidate_weights(candidate_id: int) -> tuple[float, ...]:
    _, weights = candidate_definition(candidate_id)
    return tuple(float(value) for value in weights)


def _make_fallback_many_affine(config: MABBOBConfig) -> _FallbackManyAffine:
    xopt_seed = int(config.xopt_seed or config.instance)
    rng = np.random.default_rng(xopt_seed)
    xopt = rng.uniform(low=-5.0, high=5.0, size=(config.dimension,))
    weights = _candidate_weights(config.candidate_id)
    components = _candidate_components(config.candidate_id)
    instances = _candidate_instances(config.candidate_id)
    return _FallbackManyAffine(
        candidate_id=config.candidate_id,
        n_variables=config.dimension,
        components=components,
        weights=weights,
        instances=instances,
        xopt=xopt,
        scale_factors=config.scales,
    )


def make_mabbob_problem(
    candidate_id: int,
    dimension: int,
    instance: int = 1,
    boundary_handling: str = "clip",
) -> Problem:
    config = MABBOBConfig(
        candidate_id=int(candidate_id),
        dimension=int(dimension),
        instance=int(instance),
        xopt_seed=instance,
        weight_seed=instance,
    )
    base_problem = _make_controlled_many_affine(config)
    if base_problem is None:
        base_problem = _make_ioh_many_affine(config)
    if base_problem is None:
        base_problem = _make_fallback_many_affine(config)

    bounds = np.column_stack(
        [
            np.full(config.dimension, -5.0, dtype=float),
            np.full(config.dimension, 5.0, dtype=float),
        ]
    )
    problem_id = f"mabbob_c{config.candidate_id:03d}_i{config.instance:02d}_d{config.dimension}"

    def objective(population: np.ndarray) -> np.ndarray:
        return np.asarray([float(base_problem(row)) for row in population], dtype=float)

    reference_value = getattr(base_problem, "reference_value", None)
    if reference_value is None:
        reference_value = getattr(getattr(base_problem, "optimum", None), "y", None)
    if reference_value is None:
        reference_value = 0.0

    def close_problem() -> None:
        if hasattr(base_problem, "close"):
            base_problem.close()

    return Problem(
        problem_id=problem_id,
        function_id=f"mabbob_c{config.candidate_id:03d}",
        family=MABBOB_FUNCTION_FAMILY,
        dimension=config.dimension,
        suite_code=4,
        function_number=config.candidate_id,
        instance_number=config.instance,
        bounds=bounds,
        objective=objective,
        reference_value=float(reference_value),
        close_callback=close_problem,
        cv_group_id=f"mabbob_c{config.candidate_id:03d}",
        boundary_handling=str(boundary_handling),
    )
