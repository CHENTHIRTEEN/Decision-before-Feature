from __future__ import annotations

import argparse
from pathlib import Path

from experiments.phase1_batch_common import (
    algorithms,
    as_int_list,
    count_fe,
    count_runs,
    fe_total_for_dimension,
    load_config,
    make_shards,
    selected_dimensions,
    selected_functions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print Phase 1 BBOB shard collection plan.")
    parser.add_argument("--config", default="configs/phase1_bbob_train.yaml")
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if str(config["suite"]).lower() != "bbob":
        raise ValueError("phase1-plan-shards currently supports only suite: bbob")

    functions = selected_functions(config, args.only_function)
    dimensions = selected_dimensions(config, args.only_dimension)
    shards = make_shards(config, args.only_function, args.only_dimension)
    run_count = count_runs(config, functions, dimensions)
    row_count = run_count * len(as_int_list(config, "checkpoint_ratios"))
    total_fe = count_fe(config, functions, dimensions)

    print(f"config: {args.config}")
    print(f"shards: {len(shards)}")
    print(f"runs: {run_count}")
    print(f"trajectory_rows: {row_count}")
    print(f"total_FE: {total_fe}")
    print(f"algorithms: {', '.join(algorithms(config))}")
    print("FE_total_by_dimension:")
    for dimension in dimensions:
        print(f"  {dimension}: {fe_total_for_dimension(config, dimension)}")
    print("outputs:")
    for shard in shards:
        state = "exists" if shard.output_path.exists() else "missing"
        print(f"  {shard.output_path} [{state}]")


if __name__ == "__main__":
    main()

