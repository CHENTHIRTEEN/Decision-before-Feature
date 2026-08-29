from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from benchmarks.factory import problem_bounds


EPS = 1e-12
WINDOW_RATIOS = {
    "w02": 0.02,
    "w05": 0.05,
    "w10": 0.10,
}


@dataclass
class NativeUpdateSnapshot:
    fe: int
    native_updates: int
    population: np.ndarray
    fitness: np.ndarray
    best_fitness: float


class NativeUpdateWindowRecorder:
    def __init__(self) -> None:
        self.snapshots: list[NativeUpdateSnapshot] = []
        self._initial_fitness_iqr: float | None = None

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
        else:
            self._initial_fitness_iqr = _fitness_iqr(snapshot.fitness)
        self.snapshots.append(snapshot)

    def build(self, *, fe_total: int, problem_id: str, algorithm: str) -> tuple[list[dict], list[dict]]:
        if not self.snapshots:
            raise ValueError("native-update history must not be empty")
        if self._initial_fitness_iqr is None:
            raise ValueError("native-update normalization baseline is missing")
        windows, history = build_window_statistics(
            self.snapshots,
            fe_total=fe_total,
            problem_id=problem_id,
            algorithm=algorithm,
            initial_fitness_iqr=self._initial_fitness_iqr,
        )
        retained_from_fe = int(history[0]["FE"])
        self.snapshots = [snapshot for snapshot in self.snapshots if snapshot.fe >= retained_from_fe]
        return windows, history

    @property
    def current_snapshot(self) -> NativeUpdateSnapshot:
        if not self.snapshots:
            raise ValueError("native-update history must not be empty")
        return self.snapshots[-1]


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
    algorithm: str,
    initial_fitness_iqr: float,
) -> tuple[list[dict], list[dict]]:
    if not snapshots:
        raise ValueError("native-update history must not be empty")
    current = snapshots[-1]
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
        # the allowed quantization excess is one native update measured at the
        # anchor region (the update that bounds the anchor from above), not the
        # current population size; for constant-population solvers the two are
        # identical, but a shrinking population (L-SHADE) would otherwise fail
        # although the window respects the native-update-aligned contract
        anchor_index = next(
            (index for index, snapshot in enumerate(snapshots[:-1]) if snapshot is anchor),
            len(snapshots) - 2,
        )
        next_after_anchor = snapshots[anchor_index + 1] if anchor_index + 1 < len(snapshots) else current
        local_update_gap = int(next_after_anchor.fe - anchor.fe)
        if span < target_span or span >= target_span + local_update_gap:
            raise ValueError(f"native-update-aligned {suffix} window exceeds one update of FE quantization")
        anchors[suffix] = anchor
        windows.append(
            _window_row(
                suffix=suffix,
                nominal_ratio=nominal_ratio,
                current=current,
                anchor=anchor,
                problem_id=problem_id,
                algorithm=algorithm,
                initial_fitness_iqr=initial_fitness_iqr,
            )
        )

    long_anchor = anchors["w10"]
    history = []
    for snapshot in snapshots:
        if snapshot.fe < long_anchor.fe:
            continue
        fitness_iqr = _fitness_iqr(snapshot.fitness)
        history.append(
            {
                "FE": int(snapshot.fe),
                "FE_ratio": float(snapshot.fe / fe_total),
                "native_updates": int(snapshot.native_updates),
                "best_fitness": float(snapshot.best_fitness),
                "diversity_mean_pairwise": _mean_pairwise_distance(
                    snapshot.population,
                    problem_id=problem_id,
                ),
                "fitness_iqr": float(fitness_iqr),
                "fitness_iqr_rel": float(fitness_iqr / max(initial_fitness_iqr, EPS)),
            }
        )
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
    algorithm: str,
    initial_fitness_iqr: float,
) -> dict:
    if current.population.ndim != 2 or anchor.population.ndim != 2:
        raise ValueError("window endpoint populations must be two-dimensional")
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
    elite_centroid_shift = float(
        np.linalg.norm(
            np.mean(_elite_subset(current_scaled, current_fitness), axis=0)
            - np.mean(_elite_subset(anchor_scaled, anchor_fitness), axis=0)
        )
    )
    covariance_trace_current = _covariance_trace(current_scaled)
    covariance_trace_anchor = _covariance_trace(anchor_scaled)
    covariance_trace_change = _relative_change(covariance_trace_current, covariance_trace_anchor)
    covariance_effective_rank_current = _covariance_effective_rank(current_scaled)
    covariance_effective_rank_anchor = _covariance_effective_rank(anchor_scaled)
    covariance_effective_rank_change = _relative_change(
        covariance_effective_rank_current,
        covariance_effective_rank_anchor,
    )
    covariance_trace_ratio = covariance_trace_current / max(abs(covariance_trace_anchor), EPS)

    anchor_quantiles = np.sort(anchor_fitness)
    current_quantiles = np.sort(current_fitness)
    if anchor_quantiles.shape == current_quantiles.shape:
        # legacy path: sorted endpoints of equal length cancel positionally
        quantile_improvements = anchor_quantiles - current_quantiles
        quantile_anchor_levels = anchor_quantiles
    else:
        # shrinking-population solvers (e.g. L-SHADE) change NP every update,
        # so the endpoints have different lengths; compare the quantile
        # functions on a common probability grid of the larger size
        levels = (np.arange(max(anchor_quantiles.size, current_quantiles.size)) + 0.5) / max(
            anchor_quantiles.size, current_quantiles.size
        )
        quantile_anchor_levels = np.quantile(anchor_fitness, levels)
        quantile_improvements = quantile_anchor_levels - np.quantile(current_fitness, levels)
    threshold = EPS * np.maximum(1.0, np.abs(quantile_anchor_levels))
    fitness_iqr_baseline = initial_fitness_iqr
    fitness_iqr_current = _fitness_iqr(current_fitness)
    fitness_iqr_rel = fitness_iqr_current / max(fitness_iqr_baseline, EPS)
    population_overlap = _population_overlap(
        anchor_scaled,
        current_scaled,
        current_diversity=current_diversity,
        algorithm=algorithm,
    )
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
        "population_chamfer_distance": float(
            _population_overlap_chamfer_distance(anchor_scaled, current_scaled)
        ),
        "elite_centroid_shift": elite_centroid_shift,
        "covariance_trace_current": float(covariance_trace_current),
        "covariance_trace_anchor": float(covariance_trace_anchor),
        "covariance_trace_ratio": float(covariance_trace_ratio),
        "covariance_trace_change": covariance_trace_change,
        "covariance_effective_rank_current": float(covariance_effective_rank_current),
        "covariance_effective_rank_anchor": float(covariance_effective_rank_anchor),
        "covariance_effective_rank": float(covariance_effective_rank_current),
        "covariance_effective_rank_change": covariance_effective_rank_change,
        "fitness_quantile_improvement_fraction": float(np.mean(quantile_improvements > threshold)),
        "fitness_mean_improvement": float(np.mean(quantile_improvements)),
        "fitness_wasserstein_distance": float(np.mean(np.abs(quantile_improvements))),
        "fitness_iqr_baseline": float(fitness_iqr_baseline),
        "fitness_iqr_current": float(fitness_iqr_current),
        "fitness_iqr_rel": float(fitness_iqr_rel),
        "population_overlap": float(population_overlap),
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


def _fitness_iqr(fitness: np.ndarray) -> float:
    q75, q25 = np.percentile(np.asarray(fitness, dtype=float).reshape(-1), [75.0, 25.0])
    return float(q75 - q25)


def _elite_subset(population_scaled: np.ndarray, fitness: np.ndarray) -> np.ndarray:
    if population_scaled.shape[0] == 0:
        return population_scaled
    elite_count = min(population_scaled.shape[0], max(2, int(np.ceil(0.20 * population_scaled.shape[0]))))
    elite_indices = np.argsort(fitness)[:elite_count]
    return population_scaled[elite_indices]


def _covariance_trace(population_scaled: np.ndarray) -> float:
    if population_scaled.shape[0] < 2:
        return 0.0
    centered = population_scaled - np.mean(population_scaled, axis=0)
    return float(np.trace(centered.T @ centered))


def _covariance_effective_rank(population_scaled: np.ndarray) -> float:
    if population_scaled.shape[0] < 2:
        return 0.0
    centered = population_scaled - np.mean(population_scaled, axis=0)
    eigenvalues = np.maximum(np.linalg.eigvalsh(centered.T @ centered), 0.0)
    total = float(np.sum(eigenvalues))
    if total <= EPS:
        return 0.0
    probabilities = eigenvalues / total
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, EPS))))
    return float(np.exp(entropy))


def _population_overlap(
    anchor_scaled: np.ndarray,
    current_scaled: np.ndarray,
    *,
    current_diversity: float,
    algorithm: str,
) -> float:
    if anchor_scaled.shape[0] == 0 or current_scaled.shape[0] == 0:
        return 0.0
    if algorithm.lower() == "cmaes":
        return _population_overlap_chamfer(anchor_scaled, current_scaled, current_diversity=current_diversity)
    return _population_overlap_nearest(anchor_scaled, current_scaled, current_diversity=current_diversity)


def _population_overlap_nearest(
    anchor_scaled: np.ndarray,
    current_scaled: np.ndarray,
    *,
    current_diversity: float,
) -> float:
    radius = 0.05 * max(current_diversity, EPS)
    nearest = np.min(np.linalg.norm(anchor_scaled[:, None, :] - current_scaled[None, :, :], axis=2), axis=0)
    return float(np.mean(nearest <= radius))


def _population_overlap_chamfer_distance(
    anchor_scaled: np.ndarray,
    current_scaled: np.ndarray,
) -> float:
    forward = np.min(np.linalg.norm(anchor_scaled[:, None, :] - current_scaled[None, :, :], axis=2), axis=0)
    backward = np.min(np.linalg.norm(current_scaled[:, None, :] - anchor_scaled[None, :, :], axis=2), axis=0)
    return float(0.5 * (float(np.mean(forward)) + float(np.mean(backward))))


def _population_overlap_chamfer(
    anchor_scaled: np.ndarray,
    current_scaled: np.ndarray,
    *,
    current_diversity: float,
) -> float:
    radius = 0.05 * max(current_diversity, EPS)
    chamfer = _population_overlap_chamfer_distance(anchor_scaled, current_scaled)
    return float(np.clip(1.0 - chamfer / max(radius, EPS), 0.0, 1.0))


def _relative_change(current: float, anchor: float) -> float:
    return float((current - anchor) / max(abs(anchor), EPS))


def _scale_population_to_unit_cube(population: np.ndarray, *, problem_id: str) -> np.ndarray:
    lower, upper = problem_bounds(problem_id)
    span = upper - lower
    if np.any(span <= 0.0):
        raise ValueError(f"invalid problem bounds for {problem_id}")
    return np.clip((np.asarray(population, dtype=float) - lower) / span, 0.0, 1.0)
