from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks import make_problem
from ela.features import ELA_FEATURE_COLUMNS, extract_ela_for_problem
from experiments.phase1_batch_common import as_int_list, fe_total_for_dimension, load_config, selected_dimensions, selected_functions


ELA_METADATA_COLUMNS = (
    "split",
    "problem_id",
    "family",
    "function",
    "instance",
    "dimension",
    "FE_total",
    "FE_analysis",
    "runtime_analysis",
    "feature_status",
    "feature_count",
    "feature_failure",
)


def default_output_path(config: dict) -> Path:
    output = Path(config["output"])
    split = output.stem.removesuffix("_trajectories")
    return Path("results") / "ela" / split / "features.parquet"


def extract_ela_for_config(
    *,
    config_path: Path,
    output_path: Path | None,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
) -> dict[str, int | str]:
    config = load_config(config_path)
    if str(config["suite"]).lower() != "bbob":
        raise ValueError("ela-extract-batch currently supports only suite: bbob")
    split = Path(config["output"]).stem.removesuffix("_trajectories")
    functions = selected_functions(config, only_functions)
    dimensions = selected_dimensions(config, only_dimensions)
    rows = []
    for function in functions:
        for dimension in dimensions:
            fe_total = fe_total_for_dimension(config, dimension)
            fe_analysis = int(0.05 * fe_total)
            for instance in as_int_list(config, "instances"):
                problem = make_problem(
                    {
                        "suite": "bbob",
                        "function": function,
                        "instance": instance,
                        "dimension": dimension,
                    }
                )
                try:
                    try:
                        features = extract_ela_for_problem(
                            problem=problem,
                            seed=0,
                            fe_analysis=fe_analysis,
                            function=function,
                            instance=instance,
                        )
                    except Exception as exc:
                        features = {
                            column: None for column in ELA_FEATURE_COLUMNS
                        }
                        features.update(
                            {
                                "FE_analysis": fe_analysis,
                                "runtime_analysis": 0.0,
                                "feature_status": "failed",
                                "feature_count": 0,
                                "feature_failure": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    rows.append(
                        {
                            "split": split,
                            "problem_id": problem.problem_id,
                            "family": problem.family,
                            "function": int(function),
                            "instance": int(instance),
                            "dimension": int(dimension),
                            "FE_total": int(fe_total),
                            **features,
                        }
                    )
                finally:
                    problem.close()

    output = output_path or default_output_path(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    schema = _schema()
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, output)
    print(f"wrote {len(rows)} ELA feature rows to {output}")
    return {"rows": len(rows), "output": str(output)}


def _schema() -> pa.Schema:
    fields = [
        ("split", pa.string()),
        ("problem_id", pa.string()),
        ("family", pa.string()),
        ("function", pa.int32()),
        ("instance", pa.int32()),
        ("dimension", pa.int32()),
        ("FE_total", pa.int64()),
    ]
    fields.extend((column, pa.float64()) for column in ELA_FEATURE_COLUMNS)
    fields.extend(
        [
            ("FE_analysis", pa.int64()),
            ("runtime_analysis", pa.float64()),
            ("feature_status", pa.string()),
            ("feature_count", pa.int32()),
            ("feature_failure", pa.string()),
        ]
    )
    return pa.schema(fields)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract BBOB ELA features for Phase 1 problem instances.")
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    args = parser.parse_args()

    if args.output is not None and len(args.config) != 1:
        raise ValueError("--output can be used only with a single --config")
    for config_path in args.config:
        extract_ela_for_config(
            config_path=config_path,
            output_path=args.output,
            only_functions=args.only_function,
            only_dimensions=args.only_dimension,
        )


if __name__ == "__main__":
    main()
