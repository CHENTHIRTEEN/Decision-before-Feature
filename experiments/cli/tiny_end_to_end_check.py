from __future__ import annotations

import argparse
from pathlib import Path

from experiments.cli.algorithm_consistency_check import run_algorithm_consistency_check
from experiments.cli.stage0_resource_check import verify_stage0_resource_estimate


DEFAULT_CONFIGS = (
    Path("configs/phase1_bbob_train.yaml"),
    Path("configs/phase1_bbob_validation.yaml"),
    Path("configs/phase1_cec2017_test.yaml"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny end-to-end verification check.")
    parser.add_argument("--max-runs", type=int, default=1)
    args = parser.parse_args()

    stage0 = verify_stage0_resource_estimate(DEFAULT_CONFIGS)
    consistency = run_algorithm_consistency_check(
        suite="bbob",
        algorithms=("de", "pso", "cmaes", "shade"),
        functions=(1,),
        dimensions=(10,),
        seeds=(1,),
        instance=1,
        population_size=40,
        checkpoint_fes=(40,),
        max_runs=args.max_runs,
    )
    print(stage0["checkpoints"]["bbob_total"])
    print(consistency["summary"])


if __name__ == "__main__":
    main()
