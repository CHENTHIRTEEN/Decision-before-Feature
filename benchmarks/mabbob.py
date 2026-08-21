from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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
class MABBOBDefinition:
    candidate_id: int
    dimension: int
    components: tuple[int, ...]
    weights: tuple[float, ...]
    instances: tuple[int, ...]
    xopt: np.ndarray
    scale_factors: tuple[float, ...] = DEFAULT_MABBOB_SCALES
    bridge_type: str = "unknown"
    xopt_mode: str = "uniform"


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


_PAIRWISE_PROFILES = (0.2, 0.5, 0.8)
_PAIRWISE_PROFILES_EXTENDED = (0.15, 0.45, 0.85)
_TRIPLE_PROFILES = ((0.5, 0.3, 0.2), (0.2, 0.5, 0.3), (0.3, 0.2, 0.5))
_TRIPLE_PROFILES_EXTENDED = ((0.45, 0.35, 0.20), (0.20, 0.45, 0.35), (0.35, 0.20, 0.45))


def _pairwise_candidates() -> list[tuple[int, int]]:
    return [
        (left, right)
        for index, left in enumerate(BBOB_TRAIN_FUNCTIONS)
        for right in BBOB_TRAIN_FUNCTIONS[index + 1 :]
    ]


def _triple_candidates() -> list[tuple[int, int, int]]:
    triples: list[tuple[int, int, int]] = []
    for first_index, first in enumerate(BBOB_TRAIN_FUNCTIONS):
        for second_index in range(first_index + 1, len(BBOB_TRAIN_FUNCTIONS)):
            for third in BBOB_TRAIN_FUNCTIONS[second_index + 1 :]:
                triples.append((first, BBOB_TRAIN_FUNCTIONS[second_index], third))
    return triples


def _bridge_type(candidate_id: int, arity: int) -> str:
    if arity == 1:
        return "anchor"
    if arity == 2:
        return "pairwise_bridge"
    if arity == 3:
        return "sparse_3way_bridge"
    if arity == 4:
        return "sparse_4way_bridge"
    return "dense_bridge"


def _weights_from_components(components: tuple[int, ...], profiles: tuple[float, ...] | tuple[tuple[float, ...], ...]) -> np.ndarray:
    weights = np.zeros(24, dtype=float)
    if len(components) == 1:
        weights[components[0] - 1] = 1.0
        return weights
    if len(components) == 2:
        left, right = components
        if len(profiles) != 3:
            raise ValueError("pairwise profiles must contain exactly three weights")
        weights[left - 1] = float(profiles[0])
        weights[right - 1] = float(profiles[1])
        return weights
    for index, component in enumerate(components):
        weights[component - 1] = float(profiles[index])
    return weights


def candidate_definition(candidate_id: int) -> tuple[tuple[int, ...], np.ndarray]:
    candidate = int(candidate_id)
    if candidate < 1 or candidate > 200:
        raise ValueError(f"candidate_id must be in [1, 200], got {candidate}")

    pairs = _pairwise_candidates()
    triples = _triple_candidates()

    if candidate <= 90:
        pair_index, alpha_index = divmod(candidate - 1, 3)
        left, right = pairs[pair_index % len(pairs)]
        weights = np.zeros(24, dtype=float)
        alpha = _PAIRWISE_PROFILES[alpha_index]
        weights[left - 1] = alpha
        weights[right - 1] = 1.0 - alpha
        return (left, right), weights

    if candidate <= 100:
        triple_index, dominant_index = divmod(candidate - 91, 3)
        triple = triples[triple_index % len(triples)]
        weights = np.zeros(24, dtype=float)
        for index, component in enumerate(triple):
            weights[component - 1] = _TRIPLE_PROFILES[dominant_index][index]
        return triple, weights

    if candidate <= 190:
        pair_index, alpha_index = divmod(candidate - 101, 3)
        left, right = pairs[(pair_index + 30) % len(pairs)]
        weights = np.zeros(24, dtype=float)
        alpha = _PAIRWISE_PROFILES_EXTENDED[alpha_index]
        weights[left - 1] = alpha
        weights[right - 1] = 1.0 - alpha
        return (left, right), weights

    triple_index, dominant_index = divmod(candidate - 191, 3)
    triple = triples[(triple_index + 20) % len(triples)]
    weights = np.zeros(24, dtype=float)
    for index, component in enumerate(triple):
        weights[component - 1] = _TRIPLE_PROFILES_EXTENDED[dominant_index][index]
    return triple, weights


def candidate_metadata(candidate_id: int) -> dict[str, Any]:
    components, weights = candidate_definition(candidate_id)
    active = tuple(
        index + 1 for index, value in enumerate(weights) if float(value) > 0.0
    )
    dominant_index = int(np.argmax(weights)) + 1 if np.any(weights > 0.0) else 0
    dominant_weight = float(np.max(weights)) if np.any(weights > 0.0) else 0.0
    metadata = {
        "candidate_id": int(candidate_id),
        "components": tuple(int(component) for component in components),
        "weights": tuple(float(value) for value in weights),
        "active_components": active,
        "arity": len(components),
        "bridge_type": _bridge_type(int(candidate_id), len(components)),
        "dominant_component": dominant_index,
        "dominant_weight": dominant_weight,
        "is_val_component": any(component in BBOB_VALIDATION_FUNCTIONS for component in components),
    }
    return metadata


def _candidate_instances(candidate_id: int, components: tuple[int, ...]) -> tuple[int, ...]:
    seed = 3000 + int(candidate_id)
    rng = np.random.default_rng(seed)
    instances = np.ones(24, dtype=int)
    for index, component in enumerate(components):
        # Keep the active components distinct while staying in a safe, low-numbered instance range.
        instances[component - 1] = int(rng.integers(1, 11))
    return tuple(int(value) for value in instances)


def _xopt_from_mode(dimension: int, seed: int, mode: str = "uniform") -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    if mode == "center":
        return np.zeros(dimension, dtype=float)
    if mode == "boundary":
        values = rng.uniform(low=-4.9, high=4.9, size=dimension)
        if dimension > 0:
            anchor = int(rng.integers(0, dimension))
            values[anchor] = 4.95 if rng.random() < 0.5 else -4.95
        return values
    if mode != "uniform":
        raise ValueError(f"unsupported xopt mode: {mode}")
    return rng.uniform(low=-5.0, high=5.0, size=dimension)


def _make_definition(
    config: MABBOBConfig,
    *,
    manifest_entry: Mapping[str, Any] | None = None,
) -> MABBOBDefinition:
    if manifest_entry is None:
        components, weights = candidate_definition(config.candidate_id)
        instances = _candidate_instances(config.candidate_id, components)
        xopt_mode = "uniform"
        bridge_type = _bridge_type(config.candidate_id, len(components))
        xopt_seed = int(config.xopt_seed or config.instance)
        xopt = _xopt_from_mode(config.dimension, xopt_seed, xopt_mode)
        return MABBOBDefinition(
            candidate_id=int(config.candidate_id),
            dimension=int(config.dimension),
            components=tuple(int(component) for component in components),
            weights=tuple(float(value) for value in weights),
            instances=instances,
            xopt=xopt,
            scale_factors=tuple(float(value) for value in config.scales),
            bridge_type=bridge_type,
            xopt_mode=xopt_mode,
        )

    components_raw = manifest_entry.get("components")
    weights_raw = manifest_entry.get("weights")
    instances_raw = manifest_entry.get("instances")
    if not isinstance(components_raw, (list, tuple)) or not components_raw:
        raise ValueError("manifest entry must contain non-empty components")
    if not isinstance(weights_raw, (list, tuple)):
        raise ValueError("manifest entry must contain weights")
    if not isinstance(instances_raw, (list, tuple)):
        raise ValueError("manifest entry must contain instances")

    components = tuple(int(value) for value in components_raw)
    if len(weights_raw) != 24:
        raise ValueError("manifest entry weights must have length 24")
    if len(instances_raw) != 24:
        raise ValueError("manifest entry instances must have length 24")

    weights = tuple(float(value) for value in weights_raw)
    instances = tuple(int(value) for value in instances_raw)
    xopt_mode = str(manifest_entry.get("xopt_mode", "uniform"))
    xopt_seed = int(manifest_entry.get("xopt_seed", config.xopt_seed or config.instance))
    bridge_type = str(manifest_entry.get("bridge_type", _bridge_type(config.candidate_id, len(components))))
    scale_factors_raw = manifest_entry.get("scale_factors", config.scales)
    scale_factors = tuple(float(value) for value in scale_factors_raw)
    if len(scale_factors) != 24:
        raise ValueError("manifest entry scale_factors must have length 24")
    xopt_raw = manifest_entry.get("xopt")
    if xopt_raw is None:
        xopt = _xopt_from_mode(config.dimension, xopt_seed, xopt_mode)
    else:
        xopt = np.asarray(xopt_raw, dtype=float).reshape(-1)
        if xopt.shape != (config.dimension,):
            raise ValueError("manifest entry xopt has incompatible dimension")
    return MABBOBDefinition(
        candidate_id=int(config.candidate_id),
        dimension=int(config.dimension),
        components=components,
        weights=weights,
        instances=instances,
        xopt=xopt,
        scale_factors=scale_factors,
        bridge_type=bridge_type,
        xopt_mode=xopt_mode,
    )


def _make_ioh_many_affine(definition: MABBOBDefinition):
    if ioh_problem_module is None:
        return None
    try:
        return ioh_problem_module.ManyAffine(
            xopt=np.asarray(definition.xopt, dtype=float).tolist(),
            weights=list(definition.weights),
            instances=list(definition.instances),
            n_variables=int(definition.dimension),
            scale_factors=list(definition.scale_factors),
        )
    except Exception:
        return None


def _make_controlled_many_affine(definition: MABBOBDefinition) -> Any:
    return _make_ioh_many_affine(definition)


def _candidate_components(candidate_id: int) -> tuple[int, ...]:
    components, _ = candidate_definition(candidate_id)
    return components


def _candidate_weights(candidate_id: int) -> tuple[float, ...]:
    _, weights = candidate_definition(candidate_id)
    return tuple(float(value) for value in weights)


def _make_fallback_many_affine(definition: MABBOBDefinition) -> _FallbackManyAffine:
    return _FallbackManyAffine(
        candidate_id=definition.candidate_id,
        n_variables=definition.dimension,
        components=definition.components,
        weights=definition.weights,
        instances=definition.instances,
        xopt=np.asarray(definition.xopt, dtype=float),
        scale_factors=definition.scale_factors,
    )


def make_mabbob_problem(
    candidate_id: int,
    dimension: int,
    instance: int = 1,
    boundary_handling: str = "clip",
    *,
    manifest_entry: Mapping[str, Any] | None = None,
) -> Problem:
    config = MABBOBConfig(
        candidate_id=int(candidate_id),
        dimension=int(dimension),
        instance=int(instance),
        xopt_seed=instance,
        weight_seed=instance,
    )
    definition = _make_definition(config, manifest_entry=manifest_entry)
    base_problem = _make_controlled_many_affine(definition)
    if base_problem is None:
        base_problem = _make_fallback_many_affine(definition)

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


def make_mabbob_problem_from_manifest_entry(
    candidate_id: int,
    dimension: int,
    manifest_entry: Mapping[str, Any],
    instance: int = 1,
    boundary_handling: str = "clip",
) -> Problem:
    return make_mabbob_problem(
        candidate_id=candidate_id,
        dimension=dimension,
        instance=instance,
        boundary_handling=boundary_handling,
        manifest_entry=manifest_entry,
    )
