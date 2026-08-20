from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks import make_problem
from optimizers import OptimizerRunResult, OptimizerSettings, run_optimizer
from trajectory import write_final_performance_parquet, write_parquet
from trajectory.final_performance import FinalPerformanceRecord
from experiments.phase1_batch_common import (
    algorithms,
    as_int_list,
    fe_total_for_dimension,
    function_id_name,
    landscape_family_name,
    load_config,
    make_shards,
    selected_dimensions,
    selected_functions,
    shard_output_pair_state,
    validate_dynamic_collection_config,
)


def _run_one(
    config: dict,
    algorithm: str,
    function: int,
    instance: int,
    dimension: int,
    seed: int,
) -> OptimizerRunResult:
    problem_config = {
        "suite": config["suite"],
        "function": function,
        "instance": instance,
        "dimension": dimension,
    }
    if str(config["suite"]).lower() == "mabbob":
        problem_config["candidate_id"] = function
    settings = OptimizerSettings(
        population_size=int(config["population_size"]),
        sampling_protocol=str(config["sampling_protocol"]),
    )
    problem = None
    try:
        problem = make_problem(problem_config)
        return run_optimizer(
            algorithm=algorithm,
            problem=problem,
            seed=seed,
            fe_total=fe_total_for_dimension(config, dimension),
            settings=settings,
            log10_gap_floor=float(config["log10_gap_floor"]),
            log10_gap_cap=float(config["log10_gap_cap"]),
            success_gap_target=float(config["success_gap_target"]),
            failure_loss_cap=float(config["failure_loss_cap"]),
        )
    except Exception as exc:
        suite = str(config["suite"]).lower()
        problem_id = (
            f"bbob_f{int(function):03d}_i{int(instance):02d}_d{int(dimension)}"
            if suite == "bbob"
            else f"{suite}_f{int(function):02d}_d{int(dimension)}"
        )
        if suite == "mabbob":
            problem_id = f"mabbob_c{int(function):03d}_i{int(instance):02d}_d{int(dimension)}"
        failure = FinalPerformanceRecord.from_failure(
            problem_id=problem_id,
            function_id=function_id_name(suite, function),
            family=landscape_family_name(suite, function),
            cv_group_id=function_id_name(suite, function),
            dimension=dimension,
            algorithm=algorithm,
            seed=seed,
            fe_total=fe_total_for_dimension(config, dimension),
            effective_fe=0,
            native_updates=0,
            best_fitness=None,
            benchmark_reference_value=None,
            failure_loss_cap=float(config["failure_loss_cap"]),
            log10_gap_floor=float(config["log10_gap_floor"]),
            log10_gap_cap=float(config["log10_gap_cap"]),
            success_gap_target=float(config["success_gap_target"]),
            first_hit_fe=None,
            failure_type=type(exc).__name__,
            failure_message=str(exc),
        )
        return OptimizerRunResult(
            trajectory_records=[],
            final_performance=failure,
            trajectory_query_records=[],
        )
    finally:
        if problem is not None:
            problem.close()


def _collect_records(
    config: dict,
    functions: list[int],
    dimensions: list[int],
    algorithms_: list[str],
) -> tuple[list, list]:
    trajectory_records: list = []
    final_performance_records: list = []
    for function, instance, dimension, seed, algorithm in product(
        functions, as_int_list(config, "instances"), dimensions, as_int_list(config, "seeds"), algorithms_
    ):
        result = _run_one(config, algorithm, function, instance, dimension, seed)
        trajectory_records.extend(result.trajectory_records)
        final_performance_records.append(result.final_performance)
    return trajectory_records, final_performance_records


def _write_output_pair(
    trajectory_records: list,
    final_performance_records: list,
    output_path: Path,
    final_performance_path: Path,
) -> tuple[Path, Path]:
    trajectory_path = Path(output_path)
    final_path = Path(final_performance_path)
    if trajectory_path.parent.resolve() != final_path.parent.resolve():
        raise ValueError("trajectory and final-performance outputs must share one directory")

    output_dir = trajectory_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if len({trajectory_path.resolve(), final_path.resolve()}) != 2:
        raise ValueError("trajectory and final-performance outputs must use different paths")
    with TemporaryDirectory(prefix=".phase1-output-pair-", dir=output_dir) as temp_dir:
        temp_root = Path(temp_dir)
        temporary_trajectory = temp_root / trajectory_path.name
        temporary_final = temp_root / final_path.name
        write_parquet(trajectory_records, temporary_trajectory)
        write_final_performance_parquet(
            final_performance_records,
            temporary_final,
        )

        targets = (trajectory_path, final_path)
        non_files = [path for path in targets if path.exists() and not path.is_file()]
        if non_files:
            raise IsADirectoryError(f"output target is not a file: {non_files[0]}")

        for target in targets:
            target.unlink(missing_ok=True)
        temporary_trajectory.replace(trajectory_path)
        temporary_final.replace(final_path)

    return trajectory_path, final_path


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
        output_state = shard_output_pair_state(shard)
        if output_state == "partial" and not overwrite:
            raise FileExistsError(
                "trajectory and final-performance shards must be regenerated together; "
                f"pass --overwrite: {shard.output_path.parent}"
            )
        if output_state == "complete" and not overwrite:
            print(f"skip existing shard pair {shard.output_path.parent}")
            skipped_count += 1
            continue
        pending_shards.append(shard)

    written_count = 0
    if workers <= 1 or len(pending_shards) <= 1:
        for shard in pending_shards:
            records, final_records = _collect_records(
                config, [shard.function], [shard.dimension], algorithms_
            )
            written, final_written = _write_output_pair(
                records,
                final_records,
                shard.output_path,
                shard.final_performance_path,
            )
            print(
                f"wrote {len(records)} trajectory records to {written}; "
                f"wrote {len(final_records)} complete-budget rows to {final_written}"
            )
            written_count += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _collect_and_write_shard,
                    config,
                    shard.function,
                    shard.dimension,
                    shard.output_path,
                    shard.final_performance_path,
                    algorithms_,
                ): shard
                for shard in pending_shards
            }
            for future in as_completed(futures):
                records_count, final_count, written, final_written = future.result()
                print(
                    f"wrote {records_count} trajectory records to {written}; "
                    f"wrote {final_count} complete-budget rows to {final_written}"
                )
                written_count += 1
    print(f"finished {written_count} written shards, {skipped_count} skipped shards")


def _collect_and_write_shard(
    config: dict,
    function: int,
    dimension: int,
    output_path: Path,
    final_performance_path: Path,
    algorithms_: list[str],
) -> tuple[int, int, str, str]:
    records, final_records = _collect_records(config, [function], [dimension], algorithms_)
    written, final_written = _write_output_pair(
        records,
        final_records,
        output_path,
        final_performance_path,
    )
    return len(records), len(final_records), str(written), str(final_written)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Phase 1 BBOB trajectories in batch.")
    parser.add_argument("--config", default="configs/phase1_bbob_train.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--sharded", action="store_true")
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Shard-level worker processes for --sharded collection.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    validate_dynamic_collection_config(config)
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
    records, final_records = _collect_records(config, functions, dimensions, algorithms(config))

    final_path = output_path.with_name(f"{output_path.stem}_final_performance.parquet")
    written, final_written = _write_output_pair(
        records,
        final_records,
        output_path,
        final_path,
    )
    print(
        f"wrote {len(records)} trajectory records to {written}; "
        f"wrote {len(final_records)} complete-budget rows to {final_written}"
    )


if __name__ == "__main__":
    main()
