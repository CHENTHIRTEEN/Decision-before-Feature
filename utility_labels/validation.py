from __future__ import annotations

import argparse
from math import isfinite
from pathlib import Path

import pyarrow.parquet as pq

from utility_labels.fields import NEED_ELA_COLUMNS, UTILITY_COLUMNS, UTILITY_VALUE_COLUMNS


def validate_utility_label_file(path: str | Path) -> dict[str, int]:
    table = pq.read_table(path)
    columns = set(table.column_names)
    missing = set(UTILITY_COLUMNS).difference(columns)
    if missing:
        raise ValueError(f"missing utility label columns: {sorted(missing)}")
    unexpected = columns.difference(UTILITY_COLUMNS)
    if unexpected:
        raise ValueError(f"unexpected utility label columns: {sorted(unexpected)}")
    rows = table.to_pylist()
    if not rows:
        raise ValueError("utility label file contains no rows")
    for row in rows:
        if row["FE_ratio"] < 0.12 or row["FE_ratio"] >= 1.0:
            raise ValueError("utility label FE_ratio must be in [0.12, 1.0)")
        if row["FE_prefix"] + row["FE_analysis"] + row["FE_ela_optimization"] != row["FE_total"]:
            raise ValueError("ELA branch FE budget does not sum to FE_total")
        if row["FE_prefix"] + row["FE_skip_optimization"] != row["FE_total"]:
            raise ValueError("skip branch FE budget does not sum to FE_total")
        if row["memory_cost_norm"] != 0.0:
            raise ValueError("v1 memory_cost_norm must be 0.0")
        for column in UTILITY_VALUE_COLUMNS:
            value = row[column]
            if value is None or not isfinite(float(value)):
                raise ValueError(f"{column} must be finite")
        for column in NEED_ELA_COLUMNS:
            if not isinstance(row[column], bool):
                raise ValueError(f"{column} must be boolean")
    return {"rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate offline utility label output.")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    summary = validate_utility_label_file(args.input)
    print(f"validated {summary['rows']} utility label rows")


if __name__ == "__main__":
    main()
