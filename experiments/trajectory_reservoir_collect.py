from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks import make_problem
from optimizers import OptimizerSettings, run_optimizer
from trajectory.writer import write_query_parquet
from trajectory.validation import validate_trajectory_query_file
from experiments.phase1_batch_common import (
    algorithms,
    as_int_list,
    fe_total_for_dimension,
    load_config,
    selected_dimensions,
    selected_functions,
    validate_dynamic_collection_config,
)


def _run_one(
    config: dict,
    algorithm: str,
    function: int,
    instance: int,
    dimension: int,
    seed: int,
) -> list[dict]:
    problem = make_problem(
        {
            "suite": config["suite"],
            "function": function,
            "instance": instance,
            "dimension": dimension,
        }
    )
    settings = OptimizerSettings(
        population_size=int(config["population_size"]),
        sampling_protocol=str(config["sampling_protocol"]),
    )
    try:
        result = run_optimizer(
            algorithm=algorithm,
            problem=problem,
            seed=seed,
            fe_total=fe_total_for_dimension(config, dimension),
            settings=settings,
            log10_gap_floor=float(config["log10_gap_floor"]),
            log10_gap_cap=float(config["log10_gap_cap"]),
            success_gap_target=float(config["success_gap_target"]),
            failure_loss_cap=float(config["failure_loss_cap"]),
            trajectory_query_split=str(
                config.get("split", Path(str(config["output"])).name)
            ),
        )
        return result.trajectory_query_records
    finally:
        problem.close()


def _collect_records(
    config: dict,
    functions: list[int],
    dimensions: list[int],
    algorithms_: list[str],
) -> list[dict]:
    query_records: list[dict] = []
    for function, instance, dimension, seed, algorithm in product(
        functions, as_int_list(config, "instances"), dimensions, as_int_list(config, "seeds"), algorithms_
    ):
        query_records.extend(_run_one(config, algorithm, function, instance, dimension, seed))
    return query_records


def _write_output(records: list[dict], output_path: Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not output.is_file():
        raise IsADirectoryError(f"output target is not a file: {output}")
    with TemporaryDirectory(prefix=".trajectory-query-output-", dir=output.parent) as temp_dir:
        temp_path = Path(temp_dir) / output.name
        write_query_parquet(records, temp_path)
        output.unlink(missing_ok=True)
        temp_path.replace(output)
    validate_trajectory_query_file(output)
    return output


def _collect_shards(
    config: dict,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    overwrite: bool,
    workers: int,
) -> None:
    algorithms_ = algorithms(config)
    functions = selected_functions(config, only_functions)
    dimensions = selected_dimensions(config, only_dimensions)
    written = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for function in functions:
            for dimension in dimensions:
                shard_dir = Path(config["output"]).parent / f"trajectory_query_{function}_{dimension}"
                output_path = shard_dir / "trajectory_query.parquet"
                if output_path.exists() and not overwrite:
                    print(f"skip existing reservoir query output {output_path}")
                    continue
                futures.append(
                    executor.submit(
                        _collect_and_write_shard,
                        config,
                        function,
                        dimension,
                        output_path,
                        algorithms_,
                    )
                )
        for future in as_completed(futures):
            written_path, row_count = future.result()
            print(f"wrote {row_count} trajectory-query rows to {written_path}")
            written += 1
    print(f"finished {written} trajectory-query shard outputs")


def _collect_and_write_shard(
    config: dict,
    function: int,
    dimension: int,
    output_path: Path,
    algorithms_: list[str],
) -> tuple[str, int]:
    records = _collect_records(config, [function], [dimension], algorithms_)
    written = _write_output(records, output_path)
    return str(written), len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Phase 1 trajectory reservoir queries in batch.")
    parser.add_argument("--config", default="configs/phase1_bbob_train.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--sharded", action="store_true")
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    validate_dynamic_collection_config(config)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    if args.sharded:
        _collect_shards(config, args.only_function, args.only_dimension, args.overwrite, args.workers)
        return

    output_path = Path(args.output or (Path(config["output"]).parent / "trajectory_query.parquet"))
    functions = selected_functions(config, args.only_function)
    dimensions = selected_dimensions(config, args.only_dimension)
    records = _collect_records(config, functions, dimensions, algorithms(config))
    written = _write_output(records, output_path)
    print(f"wrote {len(records)} trajectory-query rows to {written}")


if __name__ == "__main__":
    main()
