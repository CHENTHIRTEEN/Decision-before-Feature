from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from benchmarks import make_problem
from experiments.phase1_batch_common import (
    algorithms,
    fe_total_for_dimension,
    load_config,
    selected_dimensions,
    selected_functions,
    validate_dynamic_collection_config,
)
from optimizers import OptimizerSettings, run_optimizer
from trajectory import write_final_performance_parquet, write_parquet


def _run_config(config: dict, boundary_handling: str) -> tuple[list, list]:
    records: list = []
    final_records: list = []
    for function, instance, dimension, seed, algorithm in product(
        selected_functions(config),
        [int(v) for v in config["instances"]],
        selected_dimensions(config),
        [int(v) for v in config["seeds"]],
        algorithms(config),
    ):
        problem = make_problem(
            {
                "suite": config["suite"],
                "function": function,
                "instance": instance,
                "dimension": dimension,
                "candidate_id": function,
                "boundary_handling": boundary_handling,
            }
        )
        settings = OptimizerSettings(
            population_size=int(config["population_size"]),
            sampling_protocol=str(config["sampling_protocol"]),
            boundary_handling=boundary_handling,
        )
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
        )
        records.extend(result.trajectory_records)
        final_records.append(result.final_performance)
        problem.close()
    return records, final_records


def _write_output(base_dir: Path, boundary_handling: str, records: list, final_records: list) -> tuple[Path, Path]:
    output_dir = base_dir / boundary_handling
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectories.parquet"
    final_path = output_dir / "final_performance.parquet"
    with TemporaryDirectory(prefix=f".clip-reflect-{boundary_handling}-", dir=output_dir) as temp_dir:
        temp_root = Path(temp_dir)
        temporary_trajectory = temp_root / trajectory_path.name
        temporary_final = temp_root / final_path.name
        write_parquet(records, temporary_trajectory)
        write_final_performance_parquet(final_records, temporary_final)
        for target in (trajectory_path, final_path):
            target.unlink(missing_ok=True)
        temporary_trajectory.replace(trajectory_path)
        temporary_final.replace(final_path)
    return trajectory_path, final_path


def _compare_outputs(base_dir: Path) -> pd.DataFrame:
    clip = pd.read_parquet(base_dir / "clip" / "trajectories.parquet")
    reflect = pd.read_parquet(base_dir / "reflect" / "trajectories.parquet")
    keys = ["problem_id", "algorithm", "seed", "FE"]
    rows = []
    for label in ["w02", "w05", "w10"]:
        clip_rows = []
        reflect_rows = []
        for row in clip.itertuples(index=False):
            for window in row.window_statistics:
                if window["suffix"] == label:
                    clip_rows.append(
                        {
                            "problem_id": row.problem_id,
                            "algorithm": row.algorithm,
                            "seed": row.seed,
                            "FE": row.FE,
                            **window,
                        }
                    )
                    break
        for row in reflect.itertuples(index=False):
            for window in row.window_statistics:
                if window["suffix"] == label:
                    reflect_rows.append(
                        {
                            "problem_id": row.problem_id,
                            "algorithm": row.algorithm,
                            "seed": row.seed,
                            "FE": row.FE,
                            **window,
                        }
                    )
                    break
        clip_df = pd.DataFrame(clip_rows)
        reflect_df = pd.DataFrame(reflect_rows)
        merged = clip_df.merge(reflect_df, on=keys, suffixes=("_clip", "_reflect"))
        for metric in [
            "population_overlap",
            "anchor_diversity_mean_pairwise",
            "centroid_shift_distance",
            "covariance_trace_current",
            "covariance_effective_rank_current",
            "fitness_iqr_rel",
            "fitness_mean_improvement",
        ]:
            delta = merged[f"{metric}_reflect"] - merged[f"{metric}_clip"]
            rows.append(
                {
                    "window": label,
                    "metric": metric,
                    "mean_delta": float(delta.mean()),
                    "abs_mean_delta": float(delta.abs().mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a clip-vs-reflect boundary sensitivity pilot.")
    parser.add_argument("--config", default="configs/phase1_mabbob_formal.yaml")
    parser.add_argument("--boundary-handlings", nargs="+", default=["clip", "reflect"])
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    parser.add_argument("--only-seed", type=int, action="append", default=None)
    parser.add_argument("--output-dir", default="results/clip_reflect_sensitivity")
    parser.add_argument("--compare", action="store_true", help="Print a summary comparison after both runs")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    validate_dynamic_collection_config(config)
    base_dir = Path(args.output_dir)
    original_functions = selected_functions(config, args.only_function)
    original_dimensions = selected_dimensions(config, args.only_dimension)
    seeds = [int(v) for v in (args.only_seed or config["seeds"])]
    config = dict(config)
    config["functions"] = original_functions
    config["dimensions"] = original_dimensions
    config["seeds"] = seeds
    for boundary_handling in args.boundary_handlings:
        config["boundary_handling"] = boundary_handling
        records, final_records = _run_config(config, boundary_handling)
        trajectory_path, final_path = _write_output(base_dir, boundary_handling, records, final_records)
        print(f"{boundary_handling}: wrote {len(records)} trajectory rows to {trajectory_path}")
        print(f"{boundary_handling}: wrote {len(final_records)} final rows to {final_path}")

    if args.compare and set(args.boundary_handlings) >= {"clip", "reflect"}:
        summary = _compare_outputs(base_dir)
        print("\ncomparison summary")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
