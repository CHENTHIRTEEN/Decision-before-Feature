from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from landscape_queries.cheap import calculate_descriptor_cheap
from landscape_queries.specs import DESCRIPTOR_CHEAP_COLUMNS, MAIN_QUERY_ID, get_query_spec


FEATURE_METADATA_COLUMNS = (
    "split",
    "problem_id",
    "family",
    "function",
    "instance",
    "dimension",
    "FE_total",
    "query_id",
    "query_protocol",
    "sample_design_id",
    "sampling_protocol",
    "sample_seed",
    "sample_size",
    "FE_query",
    "runtime_sampling_evaluation",
    "runtime_feature_computation",
    "runtime_query",
    "feature_status",
    "feature_count",
    "feature_failure",
    "feature_group_status",
    "feature_nonfinite",
    "additional_function_evaluations",
    "query_feature_columns",
)


def default_feature_path(query_id: str, split: str) -> Path:
    return Path("results/landscape_queries/features") / query_id / split / "features.parquet"


def extract_descriptor_features(
    *,
    sample_path: Path,
    output_path: Path | None,
    overwrite: bool,
) -> dict[str, int | str]:
    spec = get_query_spec(MAIN_QUERY_ID)
    rows = pq.read_table(sample_path).to_pylist()
    if not rows:
        raise ValueError("query sample input contains no rows")
    splits = {str(row["split"]) for row in rows}
    if len(splits) != 1:
        raise ValueError("one feature output must contain exactly one split")
    output = output_path or default_feature_path(spec.query_id, next(iter(splits)))
    if output.exists() and not overwrite:
        raise FileExistsError(f"query feature output already exists; pass --overwrite: {output}")

    output_rows = []
    for row in rows:
        if row.get("sample_status") != "ok":
            raise ValueError(f"cannot extract features from failed sample row: {row['problem_id']}")
        if str(row["sample_design_id"]) != spec.sample_design_id:
            raise ValueError(f"{spec.query_id} requires sample design {spec.sample_design_id}")
        started = perf_counter()
        group_status: dict[str, dict[str, object]] = {}
        failures: list[str] = []
        try:
            raw_features = calculate_descriptor_cheap(
                np.asarray(row["X"], dtype=float),
                np.asarray(row["y"], dtype=float),
                np.asarray(row["lower_bounds"], dtype=float),
                np.asarray(row["upper_bounds"], dtype=float),
            )
        except Exception as exc:
            raw_features = {column: float("nan") for column in spec.feature_columns}
            message = f"{type(exc).__name__}: {exc}"
            failures.append(message)
        runtime_feature = perf_counter() - started
        nonfinite = [column for column, value in raw_features.items() if not np.isfinite(float(value))]
        group_status["descriptor_cheap"] = {
            "status": "failed" if failures else "ok",
            "runtime_seconds": float(runtime_feature),
            "nonfinite_columns": nonfinite,
            "warnings": [],
            "error": failures[0] if failures else "",
        }
        features = {
            column: None if column in nonfinite else float(raw_features[column])
            for column in spec.feature_columns
        }
        output_rows.append(
            {
                **{
                    name: row[name]
                    for name in (
                        "split",
                        "problem_id",
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
                        "runtime_sampling_evaluation",
                    )
                },
                "query_id": spec.query_id,
                "query_protocol": spec.protocol,
                "runtime_feature_computation": float(runtime_feature),
                "runtime_query": float(row["runtime_sampling_evaluation"] + runtime_feature),
                "feature_status": "failed" if failures else "ok",
                "feature_count": int(len(spec.feature_columns) - len(nonfinite)),
                "feature_failure": json.dumps(failures, ensure_ascii=False),
                "feature_group_status": json.dumps(group_status, sort_keys=True, ensure_ascii=False),
                "feature_nonfinite": json.dumps(
                    {"descriptor_cheap": nonfinite} if nonfinite else {},
                    sort_keys=True,
                    ensure_ascii=False,
                ),
                "additional_function_evaluations": 0,
                "query_feature_columns": json.dumps(list(spec.feature_columns), ensure_ascii=False),
                **features,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(output_rows, schema=feature_schema(spec.feature_columns)), output)
    print(f"wrote {len(output_rows)} {spec.query_id} feature rows to {output}")
    return {"rows": len(output_rows), "query_id": spec.query_id, "output": str(output)}


def feature_schema(feature_columns: tuple[str, ...]) -> pa.Schema:
    fields = [
        ("split", pa.string()),
        ("problem_id", pa.string()),
        ("family", pa.string()),
        ("function", pa.int32()),
        ("instance", pa.int32()),
        ("dimension", pa.int32()),
        ("FE_total", pa.int64()),
        ("query_id", pa.string()),
        ("query_protocol", pa.string()),
        ("sample_design_id", pa.string()),
        ("sampling_protocol", pa.string()),
        ("sample_seed", pa.int64()),
        ("sample_size", pa.int64()),
        ("FE_query", pa.int64()),
        ("runtime_sampling_evaluation", pa.float64()),
        ("runtime_feature_computation", pa.float64()),
        ("runtime_query", pa.float64()),
        ("feature_status", pa.string()),
        ("feature_count", pa.int32()),
        ("feature_failure", pa.string()),
        ("feature_group_status", pa.string()),
        ("feature_nonfinite", pa.string()),
        ("additional_function_evaluations", pa.int64()),
        ("query_feature_columns", pa.string()),
    ]
    fields.extend((column, pa.float64()) for column in feature_columns)
    return pa.schema(fields)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the fixed 16-dimensional descriptor_cheap query.")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    extract_descriptor_features(sample_path=args.samples, output_path=args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
