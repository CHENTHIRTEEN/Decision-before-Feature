from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from benchmarks import make_problem
from optimizers import OptimizerSettings, run_optimizer
from optimizers.registry import SUPPORTED_ALGORITHMS
from trajectory import write_parquet


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Phase 1 optimizer trajectories.")
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    settings = OptimizerSettings(
        population_size=int(config["population_size"]),
        checkpoint_ratios=tuple(float(ratio) for ratio in config["checkpoint_ratios"]),
    )
    fe_total = int(config["FE_total"])
    seed = int(config["seed"])
    algorithms = [name.lower() for name in config["algorithms"]]
    unsupported = sorted(set(algorithms).difference(SUPPORTED_ALGORITHMS))
    if unsupported:
        raise ValueError(f"unsupported algorithms: {unsupported}")

    records = []
    for algorithm in algorithms:
        problem = make_problem(config)
        try:
            records.extend(
                run_optimizer(
                    algorithm=algorithm,
                    problem=problem,
                    seed=seed,
                    fe_total=fe_total,
                    settings=settings,
                )
            )
        finally:
            problem.close()

    output_path = Path(args.output or config["output"])
    written = write_parquet(records, output_path)
    print(f"wrote {len(records)} trajectory records to {written}")


if __name__ == "__main__":
    main()
