from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

from benchmarks import make_problem
from optimizers import OptimizerSettings, run_optimizer
from trajectory import write_parquet
from experiments.phase1_batch_common import (
    algorithms,
    as_int_list,
    fe_total_for_dimension,
    load_config,
    make_shards,
    selected_dimensions,
    selected_functions,
)


def _run_one(config: dict, algorithm: str, function: int, instance: int, dimension: int, seed: int) -> list:
    problem_config = {
        "suite": config["suite"],
        "function": function,
        "instance": instance,
        "dimension": dimension,
    }
    settings = OptimizerSettings(
        population_size=int(config["population_size"]),
        checkpoint_ratios=tuple(float(ratio) for ratio in config["checkpoint_ratios"]),
    )
    problem = make_problem(problem_config)
    try:
        return run_optimizer(
            algorithm=algorithm,
            problem=problem,
            seed=seed,
            fe_total=fe_total_for_dimension(config, dimension),
            settings=settings,
        )
    finally:
        problem.close()


def _collect_records(config: dict, functions: list[int], dimensions: list[int], algorithms_: list[str]) -> list:
    records = []
    for function, instance, dimension, seed, algorithm in product(
        functions, as_int_list(config, "instances"), dimensions, as_int_list(config, "seeds"), algorithms_
    ):
        records.extend(_run_one(config, algorithm, function, instance, dimension, seed))
    return records


def _collect_shards(
    config: dict,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    overwrite: bool,
    workers: int = 1,
) -> None:
    algorithms_ = algorithms(config)
    shards = make_shards(config, only_functions, only_dimensions)
    skipped_count = 0
    pending_shards = []
    for shard in shards:
        if shard.output_path.exists() and not overwrite:
            print(f"skip existing shard {shard.output_path}")
            skipped_count += 1
            continue
        pending_shards.append(shard)

    written_count = 0
    if workers <= 1 or len(pending_shards) <= 1:
        for shard in pending_shards:
            records = _collect_records(config, [shard.function], [shard.dimension], algorithms_)
            written = write_parquet(records, shard.output_path)
            print(f"wrote {len(records)} trajectory records to {written}")
            written_count += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_collect_and_write_shard, config, shard.function, shard.dimension, shard.output_path, algorithms_): shard
                for shard in pending_shards
            }
            for future in as_completed(futures):
                records_count, written = future.result()
                print(f"wrote {records_count} trajectory records to {written}")
                written_count += 1
    print(f"finished {written_count} written shards, {skipped_count} skipped shards")


def _collect_and_write_shard(
    config: dict,
    function: int,
    dimension: int,
    output_path: Path,
    algorithms_: list[str],
) -> tuple[int, str]:
    records = _collect_records(config, [function], [dimension], algorithms_)
    written = write_parquet(records, output_path)
    return len(records), str(written)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Phase 1 BBOB trajectories in batch.")
    parser.add_argument("--config", default="configs/phase1_bbob_mve.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--sharded", action="store_true")
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Shard-level worker processes for --sharded collection.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if str(config["suite"]).lower() not in {"bbob", "cec2017", "cec2022"}:
        raise ValueError("phase1-collect-batch supports suites: bbob, cec2017, cec2022")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    if args.sharded:
        _collect_shards(config, args.only_function, args.only_dimension, args.overwrite, args.workers)
        return

    output_path = Path(args.output or config["output"])
    if output_path.suffix != ".parquet":
        raise ValueError("non-sharded collection requires a .parquet output file; use --sharded for shard directory output")
    functions = selected_functions(config, args.only_function)
    dimensions = selected_dimensions(config, args.only_dimension)
    records = _collect_records(config, functions, dimensions, algorithms(config))

    written = write_parquet(records, output_path)
    print(f"wrote {len(records)} trajectory records to {written}")


if __name__ == "__main__":
    main()
