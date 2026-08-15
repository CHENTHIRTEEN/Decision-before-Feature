from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from landscape_queries.cheap import calculate_descriptor_cheap
from landscape_queries.specs import DESCRIPTOR_CHEAP_COLUMNS


TRAJECTORY_QUERY_PROTOCOL = "trajectory_query_reservoir_v1"
TRAJECTORY_QUERY_ID = "descriptor_cheap_invariant"
TRAJECTORY_QUERY_SOURCE_MODE = "trajectory_reservoir_zero_extra_fe"
TRAJECTORY_QUERY_PREPROCESSING_ID = "unit_cube_x__median_iqr_y_v1"
TRAJECTORY_QUERY_FEATURE_COLUMNS = DESCRIPTOR_CHEAP_COLUMNS
TRAJECTORY_QUERY_FEATURE_GROUP = "descriptor_cheap"
TRAJECTORY_QUERY_RESERVOIR_PER_DIMENSION = 50
TRAJECTORY_QUERY_STREAM_CODE = 2026081101
TRAJECTORY_QUERY_EVENT_CODE = 1
TRAJECTORY_QUERY_SUITE_CODES = {
    "bbob": 1,
    "cec2017": 2,
    "cec2022": 3,
}
TRAJECTORY_QUERY_ALGORITHM_CODES = {
    "de": 101,
    "pso": 202,
    "cmaes": 303,
    "shade": 404,
}

TRAJECTORY_QUERY_SCHEMA = pa.schema(
    [
        ("split", pa.string()),
        ("problem_id", pa.string()),
        ("function_id", pa.string()),
        ("family", pa.string()),
        ("function", pa.int64()),
        ("instance", pa.int64()),
        ("dimension", pa.int64()),
        ("algorithm", pa.string()),
        ("seed", pa.int64()),
        ("FE", pa.int64()),
        ("FE_ratio", pa.float64()),
        ("FE_total", pa.int64()),
        ("native_updates", pa.int64()),
        ("query_id", pa.string()),
        ("query_protocol", pa.string()),
        ("query_source_mode", pa.string()),
        ("query_preprocessing_id", pa.string()),
        ("query_feature_columns", pa.string()),
        ("trajectory_query_reservoir_size", pa.int64()),
        ("trajectory_query_seen_count", pa.int64()),
        ("trajectory_sample_count", pa.int64()),
        ("trajectory_sample_coverage_ratio", pa.float64()),
        ("reservoir_stream_code", pa.int64()),
        ("reservoir_event_code", pa.int64()),
        ("trajectory_query_runtime", pa.float64()),
        ("feature_status", pa.string()),
        ("feature_count", pa.int64()),
        ("feature_failure", pa.string()),
        ("feature_group_status", pa.string()),
        ("feature_nonfinite", pa.string()),
        *((column, pa.float64()) for column in TRAJECTORY_QUERY_FEATURE_COLUMNS),
    ]
)


class TrajectoryQueryReservoir:
    def __init__(
        self,
        *,
        problem_id: str,
        function_id: str,
        family: str,
        dimension: int,
        algorithm: str,
        seed: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        reservoir_size: int | None = None,
    ) -> None:
        self.problem_id = str(problem_id)
        self.function_id = str(function_id)
        self.family = str(family)
        self.dimension = int(dimension)
        self.algorithm = str(algorithm).lower()
        self.seed = int(seed)
        self.lower_bounds = np.asarray(lower_bounds, dtype=float).reshape(-1).copy()
        self.upper_bounds = np.asarray(upper_bounds, dtype=float).reshape(-1).copy()
        self.reservoir_size = int(
            TRAJECTORY_QUERY_RESERVOIR_PER_DIMENSION * self.dimension
            if reservoir_size is None
            else reservoir_size
        )
        if self.dimension <= 0:
            raise ValueError("trajectory query dimension must be positive")
        if self.seed < 0:
            raise ValueError("trajectory query seed must be non-negative")
        if self.lower_bounds.shape != (self.dimension,) or self.upper_bounds.shape != (self.dimension,):
            raise ValueError("trajectory query bounds must match dimension")
        if not np.isfinite(self.lower_bounds).all() or not np.isfinite(self.upper_bounds).all():
            raise ValueError("trajectory query bounds must be finite")
        if np.any(self.lower_bounds >= self.upper_bounds):
            raise ValueError("trajectory query lower bounds must be smaller than upper bounds")
        if self.algorithm not in TRAJECTORY_QUERY_ALGORITHM_CODES:
            raise ValueError(f"unsupported trajectory query algorithm: {algorithm}")
        if self.reservoir_size <= 0:
            raise ValueError("trajectory query reservoir size must be positive")
        suite_code, function, unit_number = _problem_seed_components(self.problem_id)
        self._rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    self.seed,
                    unit_number,
                    TRAJECTORY_QUERY_STREAM_CODE,
                    0,
                    TRAJECTORY_QUERY_ALGORITHM_CODES[self.algorithm],
                    TRAJECTORY_QUERY_EVENT_CODE,
                    suite_code,
                    function,
                    self.dimension,
                    self.reservoir_size,
                ]
            )
        )
        self._seen_count = 0
        self._points: list[np.ndarray] = []
        self._values: list[float] = []

    @property
    def seen_count(self) -> int:
        return self._seen_count

    @property
    def sample_count(self) -> int:
        return len(self._points)

    def observe(self, point: np.ndarray, value: float) -> None:
        candidate = np.asarray(point, dtype=float).reshape(-1)
        if candidate.size != self.dimension:
            raise ValueError("trajectory query reservoir point dimension mismatch")
        if not np.isfinite(candidate).all() or not np.isfinite(float(value)):
            raise ValueError("trajectory query reservoir can only store finite evaluations")
        self._seen_count += 1
        if len(self._points) < self.reservoir_size:
            self._points.append(candidate.copy())
            self._values.append(float(value))
            return
        index = int(self._rng.integers(self._seen_count))
        if index < self.reservoir_size:
            self._points[index] = candidate.copy()
            self._values[index] = float(value)

    def snapshot(
        self,
        *,
        split: str,
        problem_id: str,
        function_id: str,
        family: str,
        function: int,
        instance: int | None,
        algorithm: str,
        seed: int,
        fe: int,
        fe_total: int,
        native_updates: int,
    ) -> dict[str, Any]:
        if self.sample_count == 0:
            raise ValueError("trajectory query reservoir must contain at least one evaluated point")
        if (
            str(problem_id) != self.problem_id
            or str(function_id) != self.function_id
            or str(family) != self.family
        ):
            raise ValueError("trajectory query snapshot problem metadata does not match its reservoir")
        if str(algorithm).lower() != self.algorithm or int(seed) != self.seed:
            raise ValueError("trajectory query snapshot run metadata does not match its reservoir")
        if int(fe_total) <= 0 or not 0 < int(fe) <= int(fe_total):
            raise ValueError("trajectory query snapshot FE must lie in (0, FE_total]")
        if int(native_updates) < 0:
            raise ValueError("trajectory query snapshot native_updates must be non-negative")
        if int(fe) != self.seen_count:
            raise ValueError("trajectory query snapshot FE must equal the number of evaluations observed so far")
        started = perf_counter()
        x = np.asarray(self._points, dtype=float)
        y = np.asarray(self._values, dtype=float)
        raw_features = calculate_descriptor_cheap(x, y, self.lower_bounds, self.upper_bounds)
        runtime = perf_counter() - started
        nonfinite = [column for column, value in raw_features.items() if not np.isfinite(float(value))]
        feature_status = "ok" if not nonfinite else "failed"
        payload: dict[str, Any] = {
            "split": str(split),
            "problem_id": str(problem_id),
            "function_id": str(function_id),
            "family": str(family),
            "function": int(function),
            "instance": int(instance) if instance is not None else None,
            "dimension": int(self.dimension),
            "algorithm": str(algorithm),
            "seed": int(seed),
            "FE": int(fe),
            "FE_ratio": float(fe / fe_total),
            "FE_total": int(fe_total),
            "native_updates": int(native_updates),
            "query_id": TRAJECTORY_QUERY_ID,
            "query_protocol": TRAJECTORY_QUERY_PROTOCOL,
            "query_source_mode": TRAJECTORY_QUERY_SOURCE_MODE,
            "query_preprocessing_id": TRAJECTORY_QUERY_PREPROCESSING_ID,
            "query_feature_columns": json.dumps(list(TRAJECTORY_QUERY_FEATURE_COLUMNS), ensure_ascii=False),
            "trajectory_query_reservoir_size": int(self.reservoir_size),
            "trajectory_query_seen_count": int(self.seen_count),
            "trajectory_sample_count": int(self.sample_count),
            "trajectory_sample_coverage_ratio": float(self.sample_count / max(self.seen_count, 1)),
            "reservoir_stream_code": int(TRAJECTORY_QUERY_STREAM_CODE),
            "reservoir_event_code": int(TRAJECTORY_QUERY_EVENT_CODE),
            "trajectory_query_runtime": float(runtime),
            "feature_status": feature_status,
            "feature_count": int(len(TRAJECTORY_QUERY_FEATURE_COLUMNS) - len(nonfinite)),
            "feature_failure": json.dumps(
                []
                if not nonfinite
                else [{"group": TRAJECTORY_QUERY_FEATURE_GROUP, "error": "non-finite descriptors"}],
                ensure_ascii=False,
            ),
            "feature_group_status": json.dumps(
                {
                    TRAJECTORY_QUERY_FEATURE_GROUP: {
                        "status": feature_status,
                        "runtime_seconds": float(runtime),
                        "nonfinite_columns": nonfinite,
                        "warnings": [],
                        "error": "",
                    }
                },
                sort_keys=True,
                ensure_ascii=False,
            ),
            "feature_nonfinite": json.dumps(
                {TRAJECTORY_QUERY_FEATURE_GROUP: nonfinite} if nonfinite else {},
                sort_keys=True,
                ensure_ascii=False,
            ),
        }
        for column in TRAJECTORY_QUERY_FEATURE_COLUMNS:
            payload[column] = None if column in nonfinite else float(raw_features[column])
        return payload

    def write(self, path: str | Path, **metadata: Any) -> Path:
        payload = self.snapshot(**metadata)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([payload], schema=TRAJECTORY_QUERY_SCHEMA), output_path)
        return output_path


def parse_problem_id(problem_id: str) -> tuple[int, int | None]:
    value = str(problem_id)
    bbob = re.match(r"^bbob_f(\d{3})_i(\d+)_d\d+$", value)
    if bbob is not None:
        return int(bbob.group(1)), int(bbob.group(2))
    cec = re.match(r"^cec(?:2017|2022)_f(\d{2})_d\d+$", value)
    if cec is not None:
        return int(cec.group(1)), None
    raise ValueError(f"unsupported problem_id for trajectory query reservoir: {problem_id}")


def _problem_seed_components(problem_id: str) -> tuple[int, int, int]:
    value = str(problem_id)
    function, instance = parse_problem_id(value)
    if value.startswith("bbob_"):
        suite = "bbob"
    elif value.startswith("cec2017_"):
        suite = "cec2017"
    elif value.startswith("cec2022_"):
        suite = "cec2022"
    else:
        raise ValueError(f"unsupported problem_id for trajectory query reservoir: {problem_id}")
    unit_number = int(instance) if instance is not None else 1
    return TRAJECTORY_QUERY_SUITE_CODES[suite], int(function), unit_number
