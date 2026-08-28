"""Batch generation and ten-algorithm execution for AS-LGBM RGI."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from optimizers.seeding import make_indexed_rng

from .rgi_generation import (
    RGI_GENERATION_STREAM,
    RGIInstance,
    decode_rgi,
    generate_rgi_instances,
)
from .rgi_optimizers import PAPER_ALGORITHMS, run_algorithm


RGI_RUN_STREAM = 5102
DEFAULT_INSTANCE_COUNT = 200_000
DEFAULT_REPEATS = 30
DEFAULT_DIMENSION = 10
DEFAULT_FE_TOTAL = 10_000
DEFAULT_POPULATION_SIZE = 100


def _performance_column(algorithm: str, repeat: int) -> str:
    safe_algorithm = algorithm.replace("-", "_")
    return f"{safe_algorithm}_run_{repeat:02d}"


def _instance_batches(
    instances: Iterable[RGIInstance],
    *,
    batch_size: int,
) -> Iterable[list[RGIInstance]]:
    batch: list[RGIInstance] = []
    for instance in instances:
        batch.append(instance)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _run_instance(
    instance: RGIInstance,
    *,
    root_seed: int,
    repeats: int,
    algorithms: Sequence[str],
    population_size: int,
    fe_total: int,
    performance_metric: str,
    target_value: float | None,
    target_tolerance: float,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    if performance_metric not in {"final_best", "target_fe"}:
        raise ValueError("performance_metric must be final_best or target_fe")
    if performance_metric == "target_fe" and target_value is None:
        raise ValueError("target_fe performance requires target_value")

    objective = decode_rgi(
        instance.rpn,
        dimension=instance.dimension,
        lower_bound=instance.lower_bound,
        upper_bound=instance.upper_bound,
    )
    run_rows: list[dict[str, object]] = []
    performance_row: dict[str, object] = {"instance_id": instance.instance_id}
    for algorithm_index, algorithm in enumerate(algorithms):
        for repeat in range(1, repeats + 1):
            run_rng = make_indexed_rng(
                seed=root_seed,
                unit_number=instance.instance_id,
                stream_code=RGI_RUN_STREAM,
                generation=repeat,
                target=algorithm_index,
                event=1,
            )
            run_seed_rng = make_indexed_rng(
                seed=root_seed,
                unit_number=instance.instance_id,
                stream_code=RGI_RUN_STREAM,
                generation=repeat,
                target=algorithm_index,
                event=2,
            )
            run_seed = int(run_seed_rng.integers(0, 2**32, dtype=np.uint32))
            outcome = run_algorithm(
                algorithm=algorithm,
                objective=objective,
                rng=run_rng,
                dimension=instance.dimension,
                lower_bound=instance.lower_bound,
                upper_bound=instance.upper_bound,
                population_size=population_size,
                fe_total=fe_total,
                target_value=target_value if performance_metric == "target_fe" else None,
                target_tolerance=target_tolerance,
                run_seed=run_seed,
            )
            performance_row[_performance_column(algorithm, repeat)] = outcome.performance_value
            run_rows.append(
                {
                    "instance_id": instance.instance_id,
                    "instance_seed": instance.instance_seed,
                    "algorithm": outcome.algorithm,
                    "algorithm_index": algorithm_index,
                    "repeat": repeat,
                    "run_seed": outcome.run_seed,
                    "tree_depth": instance.tree_depth,
                    "dimension": instance.dimension,
                    "fe_total": outcome.fe_total,
                    "effective_fe": outcome.effective_fe,
                    "best_value": outcome.best_value,
                    "final_value": outcome.final_value,
                    "first_hit_fe": outcome.first_hit_fe,
                    "target_hit": outcome.target_hit,
                    "performance_metric": performance_metric,
                    "performance_value": outcome.performance_value,
                    "run_status": outcome.run_status,
                    "failure_type": outcome.failure_type,
                    "failure_message": outcome.failure_message,
                }
            )
    return instance.to_record(), run_rows, performance_row


def _prepare_output(output_dir: Path, *, overwrite: bool) -> tuple[Path, Path, Path]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    subdirectories = (
        output_dir / "rgi_instances",
        output_dir / "algorithm_runs",
        output_dir / "performance_runs",
    )
    existing = [path for path in subdirectories if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "RGI output already exists; choose a new directory or pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    if overwrite:
        for path in subdirectories:
            if path.exists():
                shutil.rmtree(path)
    for path in subdirectories:
        path.mkdir(parents=True, exist_ok=True)
    return subdirectories


def generate_and_run_rgi(
    *,
    output_dir: Path,
    count: int = DEFAULT_INSTANCE_COUNT,
    root_seed: int = 20260827,
    start_instance_id: int = 1,
    batch_size: int = 100,
    dimension: int = DEFAULT_DIMENSION,
    min_depth: int = 5,
    max_depth: int = 8,
    lower_bound: float = -10.0,
    upper_bound: float = 10.0,
    algorithms: Sequence[str] = PAPER_ALGORITHMS,
    repeats: int = DEFAULT_REPEATS,
    population_size: int = DEFAULT_POPULATION_SIZE,
    fe_total: int = DEFAULT_FE_TOTAL,
    performance_metric: str = "final_best",
    target_value: float | None = None,
    target_tolerance: float = 0.0,
    overwrite: bool = False,
) -> dict[str, object]:
    """Generate RGI instances, run the ten algorithms and write parquet shards.

    The three output directories contain matching ``part-XXXXXX.parquet``
    shards keyed by ``instance_id``:

    - ``rgi_instances`` stores the expression-tree RPN and its metadata;
    - ``algorithm_runs`` stores one row per algorithm and repeat;
    - ``performance_runs`` stores one wide 300-run row per instance, suitable
      for the existing AS-LGBM Soft-ERT reader.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if len(algorithms) != len(set(algorithms)) or len(algorithms) < 2:
        raise ValueError("algorithms must contain at least two unique names")
    if tuple(algorithms) != PAPER_ALGORITHMS:
        raise ValueError(
            "RGI result columns use the paper algorithm order: "
            f"{PAPER_ALGORITHMS}"
        )
    if performance_metric not in {"final_best", "target_fe"}:
        raise ValueError("performance_metric must be final_best or target_fe")
    if performance_metric == "target_fe" and target_value is None:
        raise ValueError("target_fe performance requires target_value")

    instance_dir, run_dir, performance_dir = _prepare_output(
        output_dir, overwrite=overwrite
    )
    instances = generate_rgi_instances(
        count=count,
        root_seed=root_seed,
        start_instance_id=start_instance_id,
        dimension=dimension,
        min_depth=min_depth,
        max_depth=max_depth,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    written_instances = 0
    for batch_number, batch in enumerate(
        _instance_batches(instances, batch_size=batch_size), start=1
    ):
        instance_rows: list[dict[str, object]] = []
        run_rows: list[dict[str, object]] = []
        performance_rows: list[dict[str, object]] = []
        for instance in batch:
            instance_row, rows, performance_row = _run_instance(
                instance,
                root_seed=root_seed,
                repeats=repeats,
                algorithms=algorithms,
                population_size=population_size,
                fe_total=fe_total,
                performance_metric=performance_metric,
                target_value=target_value,
                target_tolerance=target_tolerance,
            )
            instance_rows.append(instance_row)
            run_rows.extend(rows)
            performance_rows.append(performance_row)

        suffix = f"part-{batch_number:06d}.parquet"
        pd.DataFrame(instance_rows).to_parquet(instance_dir / suffix, index=False)
        pd.DataFrame(run_rows).to_parquet(run_dir / suffix, index=False)
        pd.DataFrame(performance_rows).to_parquet(
            performance_dir / suffix, index=False
        )
        written_instances += len(batch)
        print(
            f"wrote RGI shard {batch_number}: {len(batch)} instances, "
            f"{len(run_rows)} algorithm runs"
        )

    metadata = {
        "method": "AS-LGBM RGI batch generation",
        "instance_count": written_instances,
        "first_instance_id": start_instance_id,
        "last_instance_id": start_instance_id + written_instances - 1,
        "root_seed": int(root_seed),
        "generation_stream_code": RGI_GENERATION_STREAM,
        "run_stream_code": RGI_RUN_STREAM,
        "dimension": dimension,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "tree_depth_range": [min_depth, max_depth],
        "tree_generation": "full",
        "operator_codes": list(range(10, 25)),
        "operator_weights": [2, 2, 2, 4, 4, 2, 2, 3, 3, 3, 3, 40, 40, 20, 20],
        "operator_count_in_code": 15,
        "repeats_per_algorithm": repeats,
        "algorithms": list(algorithms),
        "population_size": population_size,
        "fe_total": fe_total,
        "performance_metric": performance_metric,
        "target_value": target_value,
        "target_tolerance": target_tolerance,
        "result_layout": {
            "instances": "rgi_instances/part-*.parquet",
            "long_runs": "algorithm_runs/part-*.parquet",
            "wide_performance": "performance_runs/part-*.parquet",
        },
        "wide_performance_columns": "{algorithm}_run_{repeat:02d}; CMA-ES uses CMA_ES in column names",
        "note": (
            "This is an independent literature baseline data producer. It does not enter "
            "the Decision-before-Feature main four-algorithm pipeline."
        ),
    }
    metadata_path = Path(output_dir).resolve() / "rgi_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=DEFAULT_INSTANCE_COUNT)
    parser.add_argument("--root-seed", type=int, default=20260827)
    parser.add_argument("--start-instance-id", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--min-depth", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--lower-bound", type=float, default=-10.0)
    parser.add_argument("--upper-bound", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--fe-total", type=int, default=DEFAULT_FE_TOTAL)
    parser.add_argument(
        "--performance-metric",
        choices=("final_best", "target_fe"),
        default="final_best",
        help="final_best saves the best objective; target_fe saves first-hit FE or -1.",
    )
    parser.add_argument("--target-value", type=float, default=None)
    parser.add_argument("--target-tolerance", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    metadata = generate_and_run_rgi(
        output_dir=args.output_dir,
        count=args.count,
        root_seed=args.root_seed,
        start_instance_id=args.start_instance_id,
        batch_size=args.batch_size,
        dimension=args.dimension,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound,
        repeats=args.repeats,
        population_size=args.population_size,
        fe_total=args.fe_total,
        performance_metric=args.performance_metric,
        target_value=args.target_value,
        target_tolerance=args.target_tolerance,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
