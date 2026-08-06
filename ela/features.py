from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.stats import qmc

from benchmarks.core import Problem
from optimizers.seeding import derive_seed


ELA_FEATURE_COLUMNS = (
    "ela_y_min",
    "ela_y_max",
    "ela_y_mean",
    "ela_y_std",
    "ela_y_median",
    "ela_y_iqr",
    "ela_y_skew",
    "ela_y_kurtosis",
    "ela_x_mean_pairwise",
    "ela_x_std_pairwise",
    "ela_x_best_dist_center",
    "ela_x_mean_dist_center",
    "ela_corr_y_dist_center",
    "ela_corr_y_nn_dist",
    "ela_linear_r2",
    "ela_linear_gradient_norm",
)


def extract_ela_for_problem(
    *,
    problem: Problem,
    seed: int,
    fe_analysis: int,
    function: int,
    instance: int,
) -> dict[str, float | int | str]:
    if fe_analysis <= 0:
        raise ValueError("fe_analysis must be positive")
    started = perf_counter()
    sample = _latin_hypercube_sample(
        lower=problem.lower_bounds,
        upper=problem.upper_bounds,
        n_samples=fe_analysis,
        seed=derive_seed(seed + function * 1000 + instance * 100000 + problem.dimension, 505),
    )
    values = problem.evaluate(sample)
    features = _native_ela_features(sample, values, problem.lower_bounds, problem.upper_bounds)
    runtime = perf_counter() - started
    return {
        **features,
        "FE_analysis": int(fe_analysis),
        "runtime_analysis": float(runtime),
        "feature_status": "ok",
        "feature_count": int(len(features)),
        "feature_failure": "",
    }


def _latin_hypercube_sample(
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=len(lower), seed=int(seed))
    unit = sampler.random(n=int(n_samples))
    return qmc.scale(unit, lower, upper)


def _native_ela_features(sample: np.ndarray, values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    x = np.asarray(sample, dtype=float)
    y = np.asarray(values, dtype=float).reshape(-1)
    if x.ndim != 2 or y.shape[0] != x.shape[0]:
        raise ValueError("ELA sample and values have inconsistent shapes")

    y_std = float(np.std(y))
    centered = y - float(np.mean(y))
    skew = float(np.mean(centered**3) / max(y_std**3, 1e-12))
    kurtosis = float(np.mean(centered**4) / max(y_std**4, 1e-12))
    distances = _pairwise_distances(x, len(lower))
    center = (lower + upper) / 2.0
    dist_center = np.linalg.norm(x - center, axis=1) / np.sqrt(len(lower))
    best = x[int(np.argmin(y))]
    nn_dist = _nearest_neighbor_distances(x, len(lower))
    linear_r2, gradient_norm = _linear_model_features(x, y, lower, upper)
    return {
        "ela_y_min": float(np.min(y)),
        "ela_y_max": float(np.max(y)),
        "ela_y_mean": float(np.mean(y)),
        "ela_y_std": y_std,
        "ela_y_median": float(np.median(y)),
        "ela_y_iqr": float(np.percentile(y, 75) - np.percentile(y, 25)),
        "ela_y_skew": skew,
        "ela_y_kurtosis": kurtosis,
        "ela_x_mean_pairwise": float(np.mean(distances)) if distances.size else 0.0,
        "ela_x_std_pairwise": float(np.std(distances)) if distances.size else 0.0,
        "ela_x_best_dist_center": float(np.linalg.norm(best - center) / np.sqrt(len(lower))),
        "ela_x_mean_dist_center": float(np.mean(dist_center)),
        "ela_corr_y_dist_center": _safe_corr(y, dist_center),
        "ela_corr_y_nn_dist": _safe_corr(y, nn_dist),
        "ela_linear_r2": linear_r2,
        "ela_linear_gradient_norm": gradient_norm,
    }


def _pairwise_distances(x: np.ndarray, dimension: int) -> np.ndarray:
    if x.shape[0] < 2:
        return np.asarray([], dtype=float)
    deltas = x[:, None, :] - x[None, :, :]
    matrix = np.linalg.norm(deltas, axis=2) / np.sqrt(dimension)
    return matrix[np.triu_indices(x.shape[0], k=1)]


def _nearest_neighbor_distances(x: np.ndarray, dimension: int) -> np.ndarray:
    if x.shape[0] < 2:
        return np.zeros(x.shape[0], dtype=float)
    deltas = x[:, None, :] - x[None, :, :]
    matrix = np.linalg.norm(deltas, axis=2) / np.sqrt(dimension)
    np.fill_diagonal(matrix, np.inf)
    return np.min(matrix, axis=1)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std <= 1e-12 or right_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _linear_model_features(x: np.ndarray, y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[float, float]:
    scaled = (x - lower) / np.maximum(upper - lower, 1e-12)
    design = np.column_stack([np.ones(scaled.shape[0]), scaled])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    prediction = design @ coefficients
    total = float(np.sum((y - np.mean(y)) ** 2))
    residual = float(np.sum((y - prediction) ** 2))
    r2 = 0.0 if total <= 1e-12 else max(0.0, 1.0 - residual / total)
    gradient_norm = float(np.linalg.norm(coefficients[1:]))
    return float(r2), gradient_norm
