from __future__ import annotations

import argparse
from pathlib import Path

from experiments.phase1_batch_common import (
    algorithms,
    count_fe,
    count_runs,
    fe_total_for_dimension,
    load_config,
    make_shards,
    selected_dimensions,
    selected_functions,
    validate_dynamic_collection_config,
)
from trajectory.sampling import get_sampling_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Print Phase 1 benchmark shard collection plan.")
    parser.add_argument("--config", default="configs/phase1_bbob_train.yaml")
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    validate_dynamic_collection_config(config)

    functions = selected_functions(config, args.only_function)
    dimensions = selected_dimensions(config, args.only_dimension)
    shards = make_shards(config, args.only_function, args.only_dimension)
    run_count = count_runs(config, functions, dimensions)
    sampling_spec = get_sampling_spec(str(config.get("sampling_protocol", "")))
    minimum_rows = run_count * sampling_spec.min_samples_per_run
    maximum_rows = run_count * sampling_spec.max_samples_per_run
    total_fe = count_fe(config, functions, dimensions)

    print(f"config: {args.config}")
    print(f"shards: {len(shards)}")
    print(f"runs: {run_count}")
    print(f"trajectory_rows: {minimum_rows}..{maximum_rows}")
    print(f"final_performance_rows: {run_count}")
    print(f"sampling_protocol: {sampling_spec.protocol}")
    print(f"budget_milestones_per_run: {sampling_spec.min_samples_per_run}")
    print(
        "event_only_samples_per_run: "
        f"0..{sampling_spec.max_samples_per_run - sampling_spec.min_samples_per_run}"
    )
    print(f"total_FE: {total_fe}")
    print(f"algorithms: {', '.join(algorithms(config))}")
    print("FE_total_by_dimension:")
    for dimension in dimensions:
        print(f"  {dimension}: {fe_total_for_dimension(config, dimension)}")
    print("outputs:")
    for shard in shards:
        trajectory_state = "exists" if shard.output_path.exists() else "missing"
        final_state = "exists" if shard.final_performance_path.exists() else "missing"
        print(f"  {shard.output_path} [{trajectory_state}]")
        print(f"  {shard.final_performance_path} [{final_state}]")


if __name__ == "__main__":
    main()
