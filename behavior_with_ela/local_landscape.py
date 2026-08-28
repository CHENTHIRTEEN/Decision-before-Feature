from __future__ import annotations

from collections import deque
from time import perf_counter
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist, squareform

from behavior_with_ela.protocol import LocalLandscapeConfig, suite_code


EPS = 1e-12
LOCAL_LANDSCAPE_PROTOCOL = "trajectory_local_landscape_v1"
LOCAL_LANDSCAPE_STREAM = 2026082801
LOCAL_LANDSCAPE_BOOTSTRAP_STREAM = 2026082802
ALGORITHM_CODES = {"pso": 2, "cmaes": 3, "shade": 4}

LOCAL_LANDSCAPE_STREAMING_COLUMNS = (
    "lf_y_mean",
    "lf_y_std",
    "lf_y_skew",
    "lf_y_kurtosis",
    "lf_y_q10",
    "lf_y_q50",
    "lf_y_q90",
)
LOCAL_LANDSCAPE_META_MODEL_COLUMNS = (
    "lf_linear_r2",
    "lf_quadratic_r2",
    "lf_quadratic_gain_over_linear",
    "lf_model_residual_rel",
    "lf_design_condition",
    "lf_linear_r2_change",
    "lf_quadratic_r2_change",
)
LOCAL_LANDSCAPE_INFORMATION_COLUMNS = (
    "lf_information_transition_entropy",
    "lf_information_sign_change_rate",
    "lf_information_ruggedness",
)
LOCAL_LANDSCAPE_GEOMETRY_COLUMNS = (
    "lf_local_fdc",
    "lf_elite_dispersion_ratio",
    "lf_elite_dispersion_shift",
    "lf_nbc_mean_distance",
    "lf_nbc_distance_cv",
    "lf_nbc_cluster_proxy",
)
LOCAL_LANDSCAPE_POINT_COLUMNS = (
    *LOCAL_LANDSCAPE_STREAMING_COLUMNS,
    *LOCAL_LANDSCAPE_META_MODEL_COLUMNS,
    *LOCAL_LANDSCAPE_INFORMATION_COLUMNS,
    *LOCAL_LANDSCAPE_GEOMETRY_COLUMNS,
)
LOCAL_LANDSCAPE_UNCERTAINTY_BASES = (
    "lf_local_fdc",
    "lf_elite_dispersion_ratio",
    "lf_linear_r2",
    "lf_information_transition_entropy",
)
LOCAL_LANDSCAPE_UNCERTAINTY_COLUMNS = tuple(
    f"{base}_{suffix}"
    for base in LOCAL_LANDSCAPE_UNCERTAINTY_BASES
    for suffix in ("bootstrap_mean", "bootstrap_std", "bootstrap_ci_width")
)
LOCAL_LANDSCAPE_FEATURE_COLUMNS = (
    *LOCAL_LANDSCAPE_POINT_COLUMNS,
    *LOCAL_LANDSCAPE_UNCERTAINTY_COLUMNS,
)
LOCAL_LANDSCAPE_METADATA_COLUMNS = (
    "local_landscape_protocol",
    "local_landscape_extra_FE",
    "local_landscape_seen_evaluations",
    "local_landscape_sample_count",
    "local_landscape_uniform_count",
    "local_landscape_recent_count",
    "local_landscape_elite_count",
    "local_landscape_bootstrap_repetitions",
    "local_landscape_bootstrap_sample_size",
    "local_landscape_bootstrap_confidence",
    "local_landscape_runtime_seconds",
)


class LocalLandscapeRecorder:
    def __init__(
        self,
        *,
        config: LocalLandscapeConfig,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        seed: int,
        suite: str,
        function: int,
        instance: int,
        dimension: int,
        algorithm: str,
    ) -> None:
        self.config = config
        self.lower_bounds = np.asarray(lower_bounds, dtype=float).reshape(-1)
        self.upper_bounds = np.asarray(upper_bounds, dtype=float).reshape(-1)
        self.seed = int(seed)
        self.suite = str(suite)
        self.function = int(function)
        self.instance = int(instance)
        self.dimension = int(dimension)
        self.algorithm = str(algorithm).lower()
        if self.lower_bounds.shape != (self.dimension,) or self.upper_bounds.shape != (
            self.dimension,
        ):
            raise ValueError("local landscape bounds must match dimension")
        if not np.isfinite(self.lower_bounds).all() or not np.isfinite(
            self.upper_bounds
        ).all():
            raise ValueError("local landscape bounds must be finite")
        if np.any(self.lower_bounds >= self.upper_bounds):
            raise ValueError("local landscape bounds are invalid")
        if self.algorithm not in ALGORITHM_CODES:
            raise ValueError(f"unsupported local landscape algorithm: {algorithm}")

        self.uniform_capacity = int(
            round(config.reservoir_size * config.uniform_fraction)
        )
        self.recent_capacity = int(
            round(config.reservoir_size * config.recent_fraction)
        )
        self.elite_capacity = (
            config.reservoir_size - self.uniform_capacity - self.recent_capacity
        )
        if min(
            self.uniform_capacity,
            self.recent_capacity,
            self.elite_capacity,
        ) <= 0:
            raise ValueError("each local landscape reservoir component must be non-empty")
        self._rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    self.seed,
                    LOCAL_LANDSCAPE_STREAM,
                    suite_code(self.suite),
                    self.function,
                    self.instance,
                    self.dimension,
                    ALGORITHM_CODES[self.algorithm],
                    config.reservoir_size,
                ]
            )
        )
        self._seen = 0
        self._uniform: list[tuple[int, np.ndarray, float]] = []
        self._recent: deque[tuple[int, np.ndarray, float]] = deque(
            maxlen=self.recent_capacity
        )
        self._elite: list[tuple[int, np.ndarray, float]] = []
        self._all_values: list[float] = []
        self._best_point: np.ndarray | None = None
        self._best_value = float("inf")
        self._previous_linear_r2: float | None = None
        self._previous_quadratic_r2: float | None = None
        self.records: list[dict[str, Any]] = []

    @property
    def seen_count(self) -> int:
        return self._seen

    def observe(self, point: np.ndarray, value: float) -> None:
        x = np.asarray(point, dtype=float).reshape(-1)
        y = float(value)
        if x.shape != (self.dimension,) or not np.isfinite(x).all() or not np.isfinite(y):
            raise ValueError("local landscape recorder requires finite evaluations")
        self._seen += 1
        item = (self._seen, x.copy(), y)
        self._all_values.append(y)
        self._recent.append(item)
        if y < self._best_value:
            self._best_value = y
            self._best_point = x.copy()

        if len(self._uniform) < self.uniform_capacity:
            self._uniform.append(item)
        else:
            index = int(self._rng.integers(self._seen))
            if index < self.uniform_capacity:
                self._uniform[index] = item

        if len(self._elite) < self.elite_capacity:
            self._elite.append(item)
        else:
            worst_index = max(
                range(len(self._elite)),
                key=lambda index: (self._elite[index][2], self._elite[index][0]),
            )
            if y < self._elite[worst_index][2]:
                self._elite[worst_index] = item

    def snapshot(
        self,
        *,
        split: str,
        problem_id: str,
        function_id: str,
        family: str,
        cv_group_id: str,
        fe: int,
        fe_total: int,
        native_updates: int,
        decision_opportunity_index: int,
    ) -> dict[str, Any]:
        if int(fe) != self._seen:
            raise ValueError("local landscape snapshot FE must equal observed evaluations")
        if self._best_point is None or not self._all_values:
            raise ValueError("local landscape snapshot requires evaluated points")
        started = perf_counter()
        sample = [*self._uniform, *self._recent, *self._elite]
        x = np.asarray([item[1] for item in sample], dtype=float)
        y = np.asarray([item[2] for item in sample], dtype=float)
        x_scaled = _scale_x(x, self.lower_bounds, self.upper_bounds)
        y_scaled, center, scale = _scale_y(y)
        all_y_scaled = (np.asarray(self._all_values, dtype=float) - center) / scale
        point_features = self._point_features(
            x=x_scaled,
            y=y_scaled,
            all_y=all_y_scaled,
        )
        uncertainty = self._bootstrap_features(
            x=x_scaled,
            y=y_scaled,
            fe=int(fe),
        )
        payload: dict[str, Any] = {
            "split": str(split),
            "problem_id": str(problem_id),
            "function_id": str(function_id),
            "family": str(family),
            "cv_group_id": str(cv_group_id),
            "dimension": self.dimension,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "FE": int(fe),
            "FE_ratio": float(fe / fe_total),
            "FE_total": int(fe_total),
            "native_updates": int(native_updates),
            "decision_opportunity_index": int(decision_opportunity_index),
            "local_landscape_protocol": LOCAL_LANDSCAPE_PROTOCOL,
            "local_landscape_extra_FE": 0,
            "local_landscape_seen_evaluations": self._seen,
            "local_landscape_sample_count": len(sample),
            "local_landscape_uniform_count": len(self._uniform),
            "local_landscape_recent_count": len(self._recent),
            "local_landscape_elite_count": len(self._elite),
            "local_landscape_bootstrap_repetitions": self.config.bootstrap_repetitions,
            "local_landscape_bootstrap_sample_size": self.config.bootstrap_sample_size,
            "local_landscape_bootstrap_confidence": self.config.bootstrap_confidence,
            **point_features,
            **uncertainty,
        }
        payload["local_landscape_runtime_seconds"] = float(
            perf_counter() - started
        )
        missing = set(LOCAL_LANDSCAPE_FEATURE_COLUMNS).difference(payload)
        if missing:
            raise RuntimeError(f"local landscape snapshot is missing features: {sorted(missing)}")
        self.records.append(payload)
        self._previous_linear_r2 = _finite_or_none(payload["lf_linear_r2"])
        self._previous_quadratic_r2 = _finite_or_none(payload["lf_quadratic_r2"])
        return payload

    def _point_features(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        all_y: np.ndarray,
    ) -> dict[str, float | None]:
        y_mean = float(np.mean(all_y))
        y_std = float(np.std(all_y))
        centered = all_y - y_mean
        y_skew = (
            float(np.mean(centered**3) / y_std**3) if y_std > EPS else 0.0
        )
        y_kurtosis = (
            float(np.mean(centered**4) / y_std**4) if y_std > EPS else 0.0
        )
        linear_r2, linear_residual_rel, design_condition = _meta_model(
            x,
            y,
            quadratic=False,
        )
        quadratic_r2, quadratic_residual_rel, _ = _meta_model(
            x,
            y,
            quadratic=True,
        )
        information = _information_features(
            all_y[-self.config.information_window_fe :]
        )
        local_fdc = _local_fdc(x, y)
        dispersion_ratio, dispersion_shift = _dispersion_features(
            x,
            y,
            elite_fraction=self.config.elite_sample_fraction,
        )
        nbc_mean, nbc_cv, nbc_cluster = _nbc_features(x, y)
        return {
            "lf_y_mean": y_mean,
            "lf_y_std": y_std,
            "lf_y_skew": y_skew,
            "lf_y_kurtosis": y_kurtosis,
            "lf_y_q10": float(np.quantile(all_y, 0.10)),
            "lf_y_q50": float(np.quantile(all_y, 0.50)),
            "lf_y_q90": float(np.quantile(all_y, 0.90)),
            "lf_linear_r2": linear_r2,
            "lf_quadratic_r2": quadratic_r2,
            "lf_quadratic_gain_over_linear": (
                None
                if linear_r2 is None or quadratic_r2 is None
                else float(quadratic_r2 - linear_r2)
            ),
            "lf_model_residual_rel": (
                quadratic_residual_rel
                if quadratic_residual_rel is not None
                else linear_residual_rel
            ),
            "lf_design_condition": design_condition,
            "lf_linear_r2_change": (
                None
                if linear_r2 is None or self._previous_linear_r2 is None
                else float(linear_r2 - self._previous_linear_r2)
            ),
            "lf_quadratic_r2_change": (
                None
                if quadratic_r2 is None or self._previous_quadratic_r2 is None
                else float(quadratic_r2 - self._previous_quadratic_r2)
            ),
            **information,
            "lf_local_fdc": local_fdc,
            "lf_elite_dispersion_ratio": dispersion_ratio,
            "lf_elite_dispersion_shift": dispersion_shift,
            "lf_nbc_mean_distance": nbc_mean,
            "lf_nbc_distance_cv": nbc_cv,
            "lf_nbc_cluster_proxy": nbc_cluster,
        }

    def _bootstrap_features(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        fe: int,
    ) -> dict[str, float | None]:
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    self.seed,
                    LOCAL_LANDSCAPE_BOOTSTRAP_STREAM,
                    suite_code(self.suite),
                    self.function,
                    self.instance,
                    self.dimension,
                    ALGORITHM_CODES[self.algorithm],
                    int(fe),
                ]
            )
        )
        values = {base: [] for base in LOCAL_LANDSCAPE_UNCERTAINTY_BASES}
        information_y = np.asarray(self._all_values, dtype=float)[
            -self.config.information_window_fe :
        ]
        _, information_center, information_scale = _scale_y(information_y)
        information_y = (information_y - information_center) / information_scale
        transitions = _information_transition_pairs(information_y)
        for _ in range(self.config.bootstrap_repetitions):
            indices = rng.integers(
                0,
                len(x),
                size=min(len(x), self.config.bootstrap_sample_size),
            )
            xb = x[indices]
            yb = y[indices]
            values["lf_local_fdc"].append(_local_fdc(xb, yb))
            dispersion, _ = _dispersion_features(
                xb,
                yb,
                elite_fraction=self.config.elite_sample_fraction,
            )
            values["lf_elite_dispersion_ratio"].append(dispersion)
            linear_r2, _, _ = _meta_model(xb, yb, quadratic=False)
            values["lf_linear_r2"].append(linear_r2)
            values["lf_information_transition_entropy"].append(
                _bootstrap_information_entropy(transitions, rng)
            )
        alpha = (1.0 - self.config.bootstrap_confidence) / 2.0
        output: dict[str, float | None] = {}
        for base, raw in values.items():
            array = np.asarray(
                [value for value in raw if value is not None and np.isfinite(value)],
                dtype=float,
            )
            if array.size < 2:
                mean = std = width = None
            else:
                mean = float(np.mean(array))
                std = float(np.std(array, ddof=1))
                width = float(
                    np.quantile(array, 1.0 - alpha) - np.quantile(array, alpha)
                )
            output[f"{base}_bootstrap_mean"] = mean
            output[f"{base}_bootstrap_std"] = std
            output[f"{base}_bootstrap_ci_width"] = width
        return output


def _scale_x(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (x - lower) / np.maximum(upper - lower, EPS)


def _scale_y(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(y, dtype=float).reshape(-1)
    center = float(np.median(values))
    scale = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    if scale <= EPS:
        scale = float(np.median(np.abs(values - center)))
    if scale <= EPS:
        scale = float(np.std(values))
    if scale <= EPS:
        scale = 1.0
    return (values - center) / scale, center, scale


def _linear_design(x: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(x), dtype=float), x))


def _quadratic_design(x: np.ndarray) -> np.ndarray:
    columns = [np.ones(len(x), dtype=float)]
    columns.extend(x[:, index] for index in range(x.shape[1]))
    columns.extend(x[:, index] ** 2 for index in range(x.shape[1]))
    columns.extend(
        x[:, left] * x[:, right]
        for left in range(x.shape[1])
        for right in range(left + 1, x.shape[1])
    )
    return np.column_stack(columns)


def _meta_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    quadratic: bool,
) -> tuple[float | None, float | None, float | None]:
    design = _quadratic_design(x) if quadratic else _linear_design(x)
    if len(y) <= design.shape[1]:
        return None, None, None
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    prediction = design @ coefficients
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = None if total <= EPS else float(1.0 - residual / total)
    residual_rel = (
        None
        if float(np.std(y)) <= EPS
        else float(np.sqrt(residual / len(y)) / np.std(y))
    )
    condition = float(np.linalg.cond(design))
    condition = min(condition, 1.0e20) if np.isfinite(condition) else 1.0e20
    return r2, residual_rel, condition


def _information_symbols(y: np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=float).reshape(-1)
    if len(values) < 2:
        return np.asarray([], dtype=int)
    differences = np.diff(values)
    threshold = 0.05 * max(float(np.std(values)), EPS)
    return np.where(differences > threshold, 1, np.where(differences < -threshold, -1, 0))


def _information_transition_pairs(y: np.ndarray) -> np.ndarray:
    symbols = _information_symbols(y)
    if len(symbols) < 2:
        return np.empty((0, 2), dtype=int)
    return np.column_stack((symbols[:-1], symbols[1:]))


def _transition_entropy(pairs: np.ndarray) -> float | None:
    if len(pairs) == 0:
        return None
    encoded = (pairs[:, 0] + 1) * 3 + (pairs[:, 1] + 1)
    counts = np.bincount(encoded, minlength=9).astype(float)
    probabilities = counts[counts > 0.0] / float(np.sum(counts))
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(9.0))


def _information_features(y: np.ndarray) -> dict[str, float | None]:
    symbols = _information_symbols(y)
    pairs = _information_transition_pairs(y)
    nonzero = symbols[symbols != 0]
    sign_change = (
        None
        if len(nonzero) < 2
        else float(np.mean(nonzero[1:] != nonzero[:-1]))
    )
    ruggedness = (
        None
        if len(symbols) < 2
        else float(np.mean(np.abs(np.diff(symbols))) / 2.0)
    )
    return {
        "lf_information_transition_entropy": _transition_entropy(pairs),
        "lf_information_sign_change_rate": sign_change,
        "lf_information_ruggedness": ruggedness,
    }


def _bootstrap_information_entropy(
    pairs: np.ndarray,
    rng: np.random.Generator,
) -> float | None:
    if len(pairs) == 0:
        return None
    sample = pairs[rng.integers(0, len(pairs), size=len(pairs))]
    return _transition_entropy(sample)


def _local_fdc(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or float(np.std(y)) <= EPS:
        return None
    best = x[int(np.argmin(y))]
    distance = np.linalg.norm(x - best, axis=1) / np.sqrt(x.shape[1])
    return _safe_corr(distance, y)


def _dispersion_features(
    x: np.ndarray,
    y: np.ndarray,
    *,
    elite_fraction: float,
) -> tuple[float | None, float | None]:
    if len(x) < 3:
        return None, None
    all_distances = pdist(x) / np.sqrt(x.shape[1])
    elite_count = max(2, int(np.ceil(len(x) * elite_fraction)))
    elite = x[np.argsort(y, kind="mergesort")[:elite_count]]
    elite_distances = pdist(elite) / np.sqrt(x.shape[1])
    if len(all_distances) == 0 or len(elite_distances) == 0:
        return None, None
    all_mean = float(np.mean(all_distances))
    elite_mean = float(np.mean(elite_distances))
    ratio = None if all_mean <= EPS else float(elite_mean / all_mean)
    return ratio, float(elite_mean - all_mean)


def _nbc_features(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    if len(x) < 3:
        return None, None, None
    distances = squareform(pdist(x)) / np.sqrt(x.shape[1])
    nearest_better: list[float] = []
    for index in range(len(x)):
        better = np.flatnonzero(y < y[index])
        if len(better):
            nearest_better.append(float(np.min(distances[index, better])))
    if len(nearest_better) < 2:
        return None, None, None
    values = np.asarray(nearest_better, dtype=float)
    mean = float(np.mean(values))
    cv = None if mean <= EPS else float(np.std(values) / mean)
    q25, q75 = np.quantile(values, (0.25, 0.75))
    long_edge_threshold = float(q75 + 1.5 * (q75 - q25))
    cluster_proxy = float(np.mean(values > long_edge_threshold))
    return mean, cv, cluster_proxy


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if float(np.std(left)) <= EPS or float(np.std(right)) <= EPS:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None
