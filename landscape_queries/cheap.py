from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

from landscape_queries.specs import DESCRIPTOR_CHEAP_COLUMNS


def calculate_descriptor_cheap(
    sample: np.ndarray,
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    x = np.asarray(sample, dtype=float)
    y = np.asarray(values, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    if x.ndim != 2 or y.shape[0] != x.shape[0] or x.shape[1] != lower.size:
        raise ValueError("descriptor sample, values, and bounds have inconsistent shapes")

    y_std = float(np.std(y))
    centered = y - float(np.mean(y))
    skew = float(np.mean(centered**3) / y_std**3) if y_std > 1e-12 else float("nan")
    kurtosis = float(np.mean(centered**4) / y_std**4) if y_std > 1e-12 else float("nan")
    distances = _pairwise_distances(x, lower.size)
    center = (lower + upper) / 2.0
    dist_center = np.linalg.norm(x - center, axis=1) / np.sqrt(lower.size)
    best = x[int(np.argmin(y))]
    nn_dist = _nearest_neighbor_distances(x, lower.size)
    linear_r2, gradient_norm = _linear_model_features(x, y, lower, upper)
    features = {
        "descriptor_y_min": float(np.min(y)),
        "descriptor_y_max": float(np.max(y)),
        "descriptor_y_mean": float(np.mean(y)),
        "descriptor_y_std": y_std,
        "descriptor_y_median": float(np.median(y)),
        "descriptor_y_iqr": float(np.percentile(y, 75) - np.percentile(y, 25)),
        "descriptor_y_skew": skew,
        "descriptor_y_kurtosis": kurtosis,
        "descriptor_x_mean_pairwise": float(np.mean(distances)) if distances.size else float("nan"),
        "descriptor_x_std_pairwise": float(np.std(distances)) if distances.size else float("nan"),
        "descriptor_x_best_dist_center": float(np.linalg.norm(best - center) / np.sqrt(lower.size)),
        "descriptor_x_mean_dist_center": float(np.mean(dist_center)),
        "descriptor_corr_y_dist_center": _safe_corr(y, dist_center),
        "descriptor_corr_y_nn_dist": _safe_corr(y, nn_dist),
        "descriptor_linear_r2": linear_r2,
        "descriptor_linear_gradient_norm": gradient_norm,
    }
    if tuple(features) != DESCRIPTOR_CHEAP_COLUMNS:
        raise ValueError("descriptor_cheap output does not match the frozen feature whitelist")
    return features


def _pairwise_distances(x: np.ndarray, dimension: int) -> np.ndarray:
    if x.shape[0] < 2:
        return np.asarray([], dtype=float)
    return np.asarray(pdist(x, metric="euclidean"), dtype=float) / np.sqrt(dimension)


def _nearest_neighbor_distances(x: np.ndarray, dimension: int) -> np.ndarray:
    if x.shape[0] < 2:
        return np.full(x.shape[0], np.nan, dtype=float)
    distances, _ = cKDTree(x).query(x, k=2, workers=1)
    return np.asarray(distances[:, 1], dtype=float) / np.sqrt(dimension)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _linear_model_features(
    x: np.ndarray,
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[float, float]:
    scaled = (x - lower) / np.maximum(upper - lower, 1e-12)
    design = np.column_stack([np.ones(scaled.shape[0]), scaled])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    prediction = design @ coefficients
    total = float(np.sum((y - np.mean(y)) ** 2))
    residual = float(np.sum((y - prediction) ** 2))
    r2 = float("nan") if total <= 1e-12 else 1.0 - residual / total
    return r2, float(np.linalg.norm(coefficients[1:]))
