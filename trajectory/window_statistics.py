from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.optimize import linear_sum_assignment

from benchmarks.factory import problem_bounds


EPS = 1e-12
WINDOW_RATIOS = {
    "w02": 0.02,
    "w05": 0.05,
    "w10": 0.10,
}


@dataclass(frozen=True)
class NativeUpdateSnapshot:
    fe: int
    native_updates: int
    population: np.ndarray
    fitness: np.ndarray
    best_fitness: float


class NativeUpdateWindowRecorder:
    def __init__(self) -> None:
        self.snapshots: list[NativeUpdateSnapshot] = []

    def observe(
        self,
        *,
        fe: int,
        native_updates: int,
        population: np.ndarray,
        fitness: np.ndarray,
        best_fitness: float,
    ) -> None:
        snapshot = make_native_update_snapshot(
            fe=fe,
            native_updates=native_updates,
            population=population,
            fitness=fitness,
            best_fitness=best_fitness,
        )
        if self.snapshots:
            previous = self.snapshots[-1]
            if snapshot.fe == previous.fe:
                return
            if snapshot.fe < previous.fe or snapshot.native_updates <= previous.native_updates:
                raise ValueError("native-update snapshots must be strictly increasing")
        self.snapshots.append(snapshot)

    def build(self, *, fe_total: int, problem_id: str) -> tuple[list[dict], list[dict]]:
        windows, history = build_window_statistics(self.snapshots, fe_total=fe_total, problem_id=problem_id)
        retained_from_fe = int(history[0]["FE"])
        self.snapshots = [snapshot for snapshot in self.snapshots if snapshot.fe >= retained_from_fe]
        return windows, history


def make_native_update_snapshot(
    *,
    fe: int,
    native_updates: int,
    population: np.ndarray,
    fitness: np.ndarray,
    best_fitness: float,
) -> NativeUpdateSnapshot:
    pop = np.asarray(population, dtype=float)
    fit = np.asarray(fitness, dtype=float).reshape(-1)
    if pop.ndim != 2 or pop.shape[0] != fit.shape[0] or pop.shape[0] == 0:
        raise ValueError("native-update snapshot population and fitness are inconsistent")
    if not np.isfinite(pop).all() or not np.isfinite(fit).all() or not np.isfinite(float(best_fitness)):
        raise ValueError("native-update snapshot values must be finite")
    return NativeUpdateSnapshot(
        fe=int(fe),
        native_updates=int(native_updates),
        population=pop.copy(),
        fitness=fit.copy(),
        best_fitness=float(best_fitness),
    )


def build_window_statistics(
    snapshots: list[NativeUpdateSnapshot],
    *,
    fe_total: int,
    problem_id: str,
) -> tuple[list[dict], list[dict]]:
    if not snapshots:
        raise ValueError("native-update history must not be empty")
    current = snapshots[-1]
    population_size = int(current.population.shape[0])
    if fe_total <= 0:
        raise ValueError("FE_total must be positive")
    windows = []
    anchors: dict[str, NativeUpdateSnapshot] = {}
    for suffix, nominal_ratio in WINDOW_RATIOS.items():
        target_span = int(round(nominal_ratio * fe_total))
        target_fe = int(current.fe - target_span)
        anchor = _latest_snapshot_not_after(snapshots[:-1], target_fe)
        if anchor is None:
            raise ValueError(f"missing native-update history for strict {suffix} window")
        span = int(current.fe - anchor.fe)
        if span < target_span or span >= target_span + population_size:
            raise ValueError(f"native-update-aligned {suffix} window exceeds one update of FE quantization")
        anchors[suffix] = anchor
        windows.append(
            _window_row(
                suffix=suffix,
                nominal_ratio=nominal_ratio,
                current=current,
                anchor=anchor,
                problem_id=problem_id,
            )
        )

    long_anchor = anchors["w10"]
    history = [
        {
            "FE": int(snapshot.fe),
            "FE_ratio": float(snapshot.fe / fe_total),
            "native_updates": int(snapshot.native_updates),
            "best_fitness": float(snapshot.best_fitness),
            "diversity_mean_pairwise": _mean_pairwise_distance(snapshot.population, problem_id=problem_id),
        }
        for snapshot in snapshots
        if snapshot.fe >= long_anchor.fe
    ]
    if not history or int(history[0]["FE"]) != long_anchor.fe or int(history[-1]["FE"]) != current.fe:
        raise ValueError("native-update scalar history does not cover the strict 10% window")
    return windows, history


def _latest_snapshot_not_after(
    snapshots: list[NativeUpdateSnapshot],
    target_fe: int,
) -> NativeUpdateSnapshot | None:
    anchor = None
    for snapshot in snapshots:
        if snapshot.fe <= target_fe:
            anchor = snapshot
        else:
            break
    return anchor


def _window_row(
    *,
    suffix: str,
    nominal_ratio: float,
    current: NativeUpdateSnapshot,
    anchor: NativeUpdateSnapshot,
    problem_id: str,
) -> dict:
    if current.population.shape != anchor.population.shape:
        raise ValueError("window endpoint populations must have identical shapes")
    dimension = int(current.population.shape[1])
    current_population, current_fitness = _ordered_arrays(current.population, current.fitness)
    anchor_population, anchor_fitness = _ordered_arrays(anchor.population, anchor.fitness)
    current_diversity = _mean_pairwise_distance(current_population, problem_id=problem_id)
    anchor_diversity = _mean_pairwise_distance(anchor_population, problem_id=problem_id)
    anchor_distance_to_best = _mean_distance_to_best(anchor_population, anchor_fitness, problem_id=problem_id)

    current_scaled = _scale_population_to_unit_cube(current_population, problem_id=problem_id)
    anchor_scaled = _scale_population_to_unit_cube(anchor_population, problem_id=problem_id)
    pairwise_distances = np.linalg.norm(anchor_scaled[:, None, :] - current_scaled[None, :, :], axis=2)
    anchor_indices, current_indices = linear_sum_assignment(pairwise_distances)
    population_wasserstein = float(np.mean(pairwise_distances[anchor_indices, current_indices]))
    centroid_shift = float(np.linalg.norm(np.mean(current_scaled, axis=0) - np.mean(anchor_scaled, axis=0)))

    anchor_quantiles = np.sort(anchor_fitness)
    current_quantiles = np.sort(current_fitness)
    quantile_improvements = anchor_quantiles - current_quantiles
    threshold = EPS * np.maximum(1.0, np.abs(anchor_quantiles))
    fitness_scale = max(float(np.mean(np.abs(anchor_quantiles))), EPS)
    radius = 0.05 * max(current_diversity, EPS)
    nearest = np.min(np.linalg.norm(anchor_scaled[:, None, :] - current_scaled[None, :, :], axis=2), axis=0)
    return {
        "suffix": suffix,
        "nominal_window_ratio": float(nominal_ratio),
        "anchor_FE": int(anchor.fe),
        "anchor_native_updates": int(anchor.native_updates),
        "anchor_best_fitness": float(anchor.best_fitness),
        "anchor_diversity_mean_pairwise": float(anchor_diversity),
        "anchor_distance_to_best": float(anchor_distance_to_best),
        "population_wasserstein_distance": population_wasserstein,
        "centroid_shift_distance": centroid_shift,
        "fitness_quantile_improvement_fraction": float(np.mean(quantile_improvements > threshold)),
        "fitness_mean_improvement": float(np.mean(quantile_improvements)),
        "fitness_wasserstein_distance": float(np.mean(np.abs(quantile_improvements))),
        "anchor_fitness_abs_mean": fitness_scale,
        "population_overlap": float(np.mean(nearest <= radius)),
    }


def _ordered_arrays(population: np.ndarray, fitness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sort_keys = [fitness]
    sort_keys.extend(population[:, axis] for axis in reversed(range(population.shape[1])))
    order = np.lexsort(tuple(sort_keys))
    return population[order], fitness[order]


def _mean_pairwise_distance(population: np.ndarray, *, problem_id: str) -> float:
    if population.shape[0] < 2:
        return 0.0
    scaled = _scale_population_to_unit_cube(population, problem_id=problem_id)
    deltas = scaled[:, None, :] - scaled[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    upper = distances[np.triu_indices(scaled.shape[0], k=1)]
    return float(np.mean(upper))


def _mean_distance_to_best(population: np.ndarray, fitness: np.ndarray, *, problem_id: str) -> float:
    best = population[int(np.argmin(fitness))]
    scaled = _scale_population_to_unit_cube(population, problem_id=problem_id)
    best_scaled = _scale_population_to_unit_cube(best.reshape(1, -1), problem_id=problem_id)[0]
    distances = np.linalg.norm(scaled - best_scaled, axis=1)
    return float(np.mean(distances))


def _scale_population_to_unit_cube(population: np.ndarray, *, problem_id: str) -> np.ndarray:
    lower, upper = problem_bounds(problem_id)
    span = upper - lower
    if np.any(span <= 0.0):
        raise ValueError(f"invalid problem bounds for {problem_id}")
    return np.clip((np.asarray(population, dtype=float) - lower) / span, 0.0, 1.0)
