from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks.factory import problem_bounds
from landscape_queries.cheap import calculate_descriptor_cheap
from landscape_queries.specs import DESCRIPTOR_CHEAP_COLUMNS


TRAJECTORY_QUERY_PROTOCOL = "trajectory_query_reservoir_v1"
TRAJECTORY_QUERY_ID = "trajectory_descriptor_cheap_16"
TRAJECTORY_QUERY_SOURCE_ID = "descriptor_cheap_invariant"
TRAJECTORY_QUERY_PREPROCESSING_ID = "unit_cube_x__median_iqr_y_v1"
TRAJECTORY_QUERY_FEATURE_COLUMNS = DESCRIPTOR_CHEAP_COLUMNS
TRAJECTORY_QUERY_FEATURE_GROUP = "trajectory_descriptor_cheap"
TRAJECTORY_QUERY_RESERVOIR_PER_DIMENSION = 50

TRAJECTORY_QUERY_SCHEMA = pa.schema(
    [
        ("split", pa.string()),
        ("problem_id", pa.string()),
        ("family", pa.string()),
        ("function", pa.int64()),
        ("instance", pa.int64()),
        ("dimension", pa.int64()),
        ("algorithm", pa.string()),
        ("seed", pa.int64()),
        ("FE", pa.int64()),
        ("FE_ratio", pa.float64()),
        ("FE_total", pa.int64()),
        ("query_id", pa.string()),
        ("query_protocol", pa.string()),
        ("query_source_id", pa.string()),
        ("query_preprocessing_id", pa.string()),
        ("query_feature_columns", pa.string()),
        ("trajectory_query_reservoir_size", pa.int64()),
        ("trajectory_query_seen_count", pa.int64()),
        ("trajectory_sample_count", pa.int64()),
        ("trajectory_sample_coverage_ratio", pa.float64()),
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
        family: str,
        dimension: int,
        algorithm: str,
        seed: int,
        reservoir_size: int | None = None,
    ) -> None:
        self.problem_id = str(problem_id)
        self.family = str(family)
        self.dimension = int(dimension)
        self.algorithm = str(algorithm)
        self.seed = int(seed)
        self.reservoir_size = int(reservoir_size or (TRAJECTORY_QUERY_RESERVOIR_PER_DIMENSION * self.dimension))
        if self.reservoir_size <= 0:
            raise ValueError("trajectory query reservoir size must be positive")
        self._rng = np.random.default_rng(
            _semantic_seed(
                TRAJECTORY_QUERY_PROTOCOL,
                self.problem_id,
                self.family,
                self.dimension,
                self.algorithm,
                self.seed,
                self.reservoir_size,
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
        family: str,
        function: int,
        instance: int | None,
        algorithm: str,
        seed: int,
        fe: int,
        fe_total: int,
    ) -> dict[str, Any]:
        if self.sample_count == 0:
            raise ValueError("trajectory query reservoir must contain at least one evaluated point")
        started = perf_counter()
        x = np.asarray(self._points, dtype=float)
        y = np.asarray(self._values, dtype=float)
        lower, upper = problem_bounds(problem_id)
        raw_features = calculate_descriptor_cheap(x, y, lower, upper)
        runtime = perf_counter() - started
        nonfinite = [column for column, value in raw_features.items() if not np.isfinite(float(value))]
        feature_status = "ok" if not nonfinite else "partial"
        payload: dict[str, Any] = {
            "split": str(split),
            "problem_id": str(problem_id),
            "family": str(family),
            "function": int(function),
            "instance": int(instance) if instance is not None else None,
            "dimension": int(self.dimension),
            "algorithm": str(algorithm),
            "seed": int(seed),
            "FE": int(fe),
            "FE_ratio": float(fe / fe_total),
            "FE_total": int(fe_total),
            "query_id": TRAJECTORY_QUERY_ID,
            "query_protocol": TRAJECTORY_QUERY_PROTOCOL,
            "query_source_id": TRAJECTORY_QUERY_SOURCE_ID,
            "query_preprocessing_id": TRAJECTORY_QUERY_PREPROCESSING_ID,
            "query_feature_columns": json.dumps(list(TRAJECTORY_QUERY_FEATURE_COLUMNS), ensure_ascii=False),
            "trajectory_query_reservoir_size": int(self.reservoir_size),
            "trajectory_query_seen_count": int(self.seen_count),
            "trajectory_sample_count": int(self.sample_count),
            "trajectory_sample_coverage_ratio": float(self.sample_count / max(self.seen_count, 1)),
            "trajectory_query_runtime": float(runtime),
            "feature_status": feature_status,
            "feature_count": int(len(TRAJECTORY_QUERY_FEATURE_COLUMNS) - len(nonfinite)),
            "feature_failure": json.dumps([], ensure_ascii=False),
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


def _semantic_seed(*parts: Any) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def parse_problem_id(problem_id: str) -> tuple[int, int | None]:
    value = str(problem_id)
    bbob = re.match(r"^bbob_f(\d{3})_i(\d+)_d\d+$", value)
    if bbob is not None:
        return int(bbob.group(1)), int(bbob.group(2))
    cec = re.match(r"^cec(?:2017|2022)_f(\d{2})_d\d+$", value)
    if cec is not None:
        return int(cec.group(1)), None
    raise ValueError(f"unsupported problem_id for trajectory query reservoir: {problem_id}")
