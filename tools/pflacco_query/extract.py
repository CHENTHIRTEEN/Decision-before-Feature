from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pflacco.classical_ela_features import (
    calculate_dispersion,
    calculate_ela_distribution,
    calculate_ela_level,
    calculate_information_content,
    calculate_nbc,
    calculate_pca,
)
from pflacco.misc_features import calculate_fitness_distance_correlation


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from landscape_queries.batch_features import feature_schema  # noqa: E402
from landscape_queries.cheap import preprocess_query_sample  # noqa: E402
from landscape_queries.specs import (  # noqa: E402
    PFLACCO_GROUP_COLUMNS,
    get_query_spec,
)


PFLACCO_VERSION = "1.2.2"
INFORMATION_CONTENT_STREAM_CODE = 3705
GROUP_FUNCTIONS: dict[str, Callable[..., dict[str, object]]] = {
    "pca": calculate_pca,
    "nbc": calculate_nbc,
    "dispersion": calculate_dispersion,
    "information_content": calculate_information_content,
    "ela_distribution": calculate_ela_distribution,
    "ela_level": calculate_ela_level,
    "fitness_distance_correlation": calculate_fitness_distance_correlation,
}


def extract_pflacco_features(
    *,
    query_id: str,
    sample_path: Path,
    output_path: Path,
    overwrite: bool,
) -> dict[str, object]:
    spec = get_query_spec(query_id)
    if spec.backend != "pflacco_1.2.2":
        raise ValueError("isolated pflacco extractor accepts only pflacco_standard or pflacco_broad")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"pflacco feature output already exists; pass --overwrite: {output_path}")
    rows = pq.read_table(sample_path).to_pylist()
    if not rows:
        raise ValueError("query sample input contains no rows")

    output_rows = []
    for row in rows:
        if row.get("sample_status") != "ok":
            raise ValueError(f"cannot extract pflacco features from failed sample: {row['problem_id']}")
        if str(row["sample_design_id"]) != spec.sample_design_id:
            raise ValueError(f"{query_id} requires sample design {spec.sample_design_id}")
        if str(row["sampling_protocol"]) != spec.sample_design.protocol:
            raise ValueError(f"{query_id} requires sampling protocol {spec.sample_design.protocol}")
        started = perf_counter()
        x_scaled, y_scaled = preprocess_query_sample(
            np.asarray(row["X"], dtype=float),
            np.asarray(row["y"], dtype=float),
            np.asarray(row["lower_bounds"], dtype=float),
            np.asarray(row["upper_bounds"], dtype=float),
        )
        x = pd.DataFrame(x_scaled)
        y = pd.Series(y_scaled)
        values: dict[str, float | None] = {}
        group_status: dict[str, dict[str, object]] = {}
        failures: list[dict[str, str]] = []
        nonfinite: dict[str, list[str]] = {}
        for group in spec.feature_groups:
            group_started = perf_counter()
            expected_columns = PFLACCO_GROUP_COLUMNS[group]
            kwargs: dict[str, object] = {}
            if group == "nbc":
                kwargs["dist_tie_breaker"] = "first"
            elif group == "information_content":
                kwargs["seed"] = _information_content_seed(row=row, query_code=spec.query_code)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    result = GROUP_FUNCTIONS[group](x.copy(), y.copy(), **kwargs)
                    _validate_group_output(group, result, expected_columns)
                    group_nonfinite = []
                    for column in expected_columns:
                        value = float(result[column]) if result[column] is not None else float("nan")
                        if np.isfinite(value):
                            values[column] = value
                        else:
                            values[column] = None
                            group_nonfinite.append(column)
                    if group_nonfinite:
                        nonfinite[group] = group_nonfinite
                    group_status[group] = {
                        "status": "ok",
                        "runtime_seconds": float(perf_counter() - group_started),
                        "nonfinite_columns": group_nonfinite,
                        "warnings": [str(item.message) for item in caught],
                        "error": "",
                    }
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    for column in expected_columns:
                        values[column] = None
                    failures.append({"group": group, "error": message})
                    group_status[group] = {
                        "status": "failed",
                        "runtime_seconds": float(perf_counter() - group_started),
                        "nonfinite_columns": [],
                        "warnings": [str(item.message) for item in caught],
                        "error": message,
                    }

        runtime_feature = perf_counter() - started
        if tuple(values) != spec.feature_columns:
            raise ValueError("pflacco feature output order does not match the versioned whitelist")
        output_rows.append(
            {
                **{
                    name: row[name]
                    for name in (
                        "split",
                        "problem_id",
                        "function_id",
                        "family",
                        "function",
                        "instance",
                        "dimension",
                        "FE_total",
                        "sample_design_id",
                        "sampling_protocol",
                        "sample_seed",
                        "sample_size",
                        "FE_query",
                        "runtime_query_sampling",
                        "runtime_query_evaluation",
                        "runtime_sampling_evaluation",
                        "benchmark_reference_value",
                        "success_gap_target",
                        "query_success",
                        "query_first_hit_offset",
                        "query_best_gap",
                    )
                },
                "query_id": spec.query_id,
                "query_protocol": spec.protocol,
                "query_preprocessing_id": spec.preprocessing_id,
                "runtime_feature_computation": float(runtime_feature),
                "runtime_query_feature_computation": float(runtime_feature),
                "runtime_query": float(row["runtime_sampling_evaluation"] + runtime_feature),
                "feature_status": "failed" if failures else "ok",
                "feature_count": int(sum(value is not None for value in values.values())),
                "feature_failure": json.dumps(failures, sort_keys=True, ensure_ascii=False),
                "feature_group_status": json.dumps(group_status, sort_keys=True, ensure_ascii=False),
                "feature_nonfinite": json.dumps(nonfinite, sort_keys=True, ensure_ascii=False),
                "additional_function_evaluations": 0,
                "query_feature_columns": json.dumps(list(spec.feature_columns), ensure_ascii=False),
                **values,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(output_rows, schema=feature_schema(spec.feature_columns)), output_path)
    print(f"wrote {len(output_rows)} {query_id} feature rows to {output_path}")
    return {"query_id": query_id, "rows": len(output_rows), "output": str(output_path)}


def _information_content_seed(*, row: dict[str, object], query_code: int) -> int:
    sequence = np.random.SeedSequence(
        [
            int(row["sample_seed"]),
            INFORMATION_CONTENT_STREAM_CODE,
            int(row["function"]),
            int(row["instance"]),
            int(row["dimension"]),
            int(query_code),
        ]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _validate_group_output(
    group: str,
    result: dict[str, object],
    expected_columns: tuple[str, ...],
) -> None:
    runtime_column = {
        "pca": "pca.costs_runtime",
        "nbc": "nbc.costs_runtime",
        "dispersion": "disp.costs_runtime",
        "information_content": "ic.costs_runtime",
        "ela_distribution": "ela_distr.costs_runtime",
        "ela_level": "ela_level.costs_runtime",
        "fitness_distance_correlation": "fitness_distance.costs_runtime",
    }[group]
    expected_returned = set(expected_columns) | {runtime_column}
    observed = set(result)
    if observed != expected_returned:
        raise ValueError(
            f"{group} returned columns outside the pflacco 1.2.2 whitelist; "
            f"missing={sorted(expected_returned - observed)}, extra={sorted(observed - expected_returned)}"
        )
    additional = [column for column in observed if "additional_function_eval" in column]
    if additional:
        raise ValueError(f"{group} unexpectedly reports extra objective evaluations: {additional}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a predefined pflacco 1.2.2 landscape query from saved samples.")
    parser.add_argument("--query-id", choices=("pflacco_standard_invariant", "pflacco_broad_invariant"), required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    extract_pflacco_features(
        query_id=args.query_id,
        sample_path=args.samples,
        output_path=args.output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
