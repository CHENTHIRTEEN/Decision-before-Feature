from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.phase1_batch_common import algorithms, load_config
from selection_reference.common import read_performance, static_virtual_best_solver_rows


def build_static_vbs(
    *,
    config_path: Path,
    output_path: Path,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    overwrite: bool,
) -> dict[str, int | str]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"static VBS output already exists; pass --overwrite: {output_path}"
        )
    config = load_config(config_path)
    performance = read_performance(config, only_functions, only_dimensions)
    rows = static_virtual_best_solver_rows(
        performance,
        portfolio_order=tuple(str(value) for value in algorithms(config)),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(rows, preserve_index=False), output_path)
    print(f"wrote {len(rows)} static problem-level VBS run outcomes to {output_path}")
    return {
        "rows": int(len(rows)),
        "problems": int(rows["problem_id"].nunique()),
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the static problem-level VBS reference by selecting one algorithm "
            "from seed-aggregated complete-budget log10 gaps, never per seed."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_static_vbs(
        config_path=args.config,
        output_path=args.output,
        only_functions=args.only_function,
        only_dimensions=args.only_dimension,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
