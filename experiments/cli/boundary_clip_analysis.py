from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.phase1_batch_common import load_config, make_shards, validate_dynamic_collection_config


BOUNDARY_TOL = 1e-12


def _boundary_stats(population: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    pop = np.asarray(population, dtype=float)
    lower = np.asarray(lower, dtype=float).reshape(1, -1)
    upper = np.asarray(upper, dtype=float).reshape(1, -1)
    if pop.ndim != 2:
        raise ValueError("population must be a 2D array")
    if pop.shape[1] != lower.shape[1] or pop.shape[1] != upper.shape[1]:
        raise ValueError("population and bounds dimension mismatch")

    on_lower = np.isclose(pop, lower, atol=BOUNDARY_TOL, rtol=0.0)
    on_upper = np.isclose(pop, upper, atol=BOUNDARY_TOL, rtol=0.0)
    on_boundary = on_lower | on_upper

    coord_fraction = float(np.mean(on_boundary))
    point_fraction = float(np.mean(np.any(on_boundary, axis=1)))
    lower_fraction = float(np.mean(on_lower))
    upper_fraction = float(np.mean(on_upper))
    both_fraction = float(np.mean(np.any(on_lower, axis=1) & np.any(on_upper, axis=1)))
    edge_mass = float(np.mean(np.any(on_boundary, axis=1)))
    per_point_hits = np.sum(on_boundary, axis=1)
    max_hits = float(np.max(per_point_hits))
    mean_hits = float(np.mean(per_point_hits))
    return {
        "boundary_coord_fraction": coord_fraction,
        "boundary_point_fraction": point_fraction,
        "boundary_lower_fraction": lower_fraction,
        "boundary_upper_fraction": upper_fraction,
        "boundary_both_sides_fraction": both_fraction,
        "boundary_edge_mass": edge_mass,
        "boundary_max_hits_per_point": max_hits,
        "boundary_mean_hits_per_point": mean_hits,
    }


def analyze_config(path: Path, only_functions: list[int] | None = None, only_dimensions: list[int] | None = None) -> pd.DataFrame:
    config = load_config(path)
    validate_dynamic_collection_config(config)
    lower_dim_map: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, object]] = []

    for shard in make_shards(config, only_functions, only_dimensions):
        trajectory_path = shard.output_path
        if not trajectory_path.exists():
            raise FileNotFoundError(f"missing trajectory file: {trajectory_path}")
        df = pd.read_parquet(trajectory_path)
        if df.empty:
            continue
        key = (shard.function_id, shard.dimension)
        if key not in lower_dim_map:
            from benchmarks.factory import problem_bounds

            problem_id_for_bounds = f"{shard.function_id}_i01_d{shard.dimension}"
            lower_dim_map[key] = problem_bounds(problem_id_for_bounds)
        lower, upper = lower_dim_map[key]
        for row in df.itertuples(index=False):
            population = np.vstack(row.population).astype(float)
            stats = _boundary_stats(population, lower, upper)
            rows.append(
                {
                    "problem_id": str(row.problem_id),
                    "function_id": str(row.function_id),
                    "family": str(row.family),
                    "cv_group_id": str(row.cv_group_id),
                    "dimension": int(row.dimension),
                    "algorithm": str(row.algorithm),
                    "seed": int(row.seed),
                    "FE": int(row.FE),
                    "FE_ratio": float(row.FE_ratio),
                    "best_fitness": float(row.best_fitness),
                    **stats,
                }
            )
    if not rows:
        raise ValueError("no trajectory rows found for analysis")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> None:
    print(f"rows: {len(df)}")
    print(f"runs: {df[['problem_id', 'algorithm', 'seed']].drop_duplicates().shape[0]}")
    print(f"problems: {df['problem_id'].nunique()}")
    print(f"algorithms: {sorted(df['algorithm'].unique().tolist())}")
    print()

    overall = {
        "boundary_coord_fraction": df["boundary_coord_fraction"].mean(),
        "boundary_point_fraction": df["boundary_point_fraction"].mean(),
        "boundary_edge_mass": df["boundary_edge_mass"].mean(),
        "boundary_lower_fraction": df["boundary_lower_fraction"].mean(),
        "boundary_upper_fraction": df["boundary_upper_fraction"].mean(),
        "boundary_max_hits_per_point": df["boundary_max_hits_per_point"].mean(),
        "boundary_mean_hits_per_point": df["boundary_mean_hits_per_point"].mean(),
    }
    print("overall boundary summary")
    for key, value in overall.items():
        print(f"  {key}: {value:.6f}")
    print()

    by_algo = df.groupby("algorithm").agg(
        boundary_coord_fraction=("boundary_coord_fraction", "mean"),
        boundary_point_fraction=("boundary_point_fraction", "mean"),
        boundary_edge_mass=("boundary_edge_mass", "mean"),
        boundary_max_hits_per_point=("boundary_max_hits_per_point", "mean"),
        boundary_mean_hits_per_point=("boundary_mean_hits_per_point", "mean"),
        best_fitness=("best_fitness", "mean"),
    )
    print("by algorithm")
    print(by_algo.sort_values("boundary_edge_mass", ascending=False).to_string(float_format=lambda x: f"{x:.6f}"))
    print()

    by_problem = df.groupby("problem_id").agg(
        boundary_coord_fraction=("boundary_coord_fraction", "mean"),
        boundary_point_fraction=("boundary_point_fraction", "mean"),
        boundary_edge_mass=("boundary_edge_mass", "mean"),
        best_fitness=("best_fitness", "mean"),
    )
    print("top 15 problems by boundary point fraction")
    print(by_problem.sort_values("boundary_point_fraction", ascending=False).head(15).to_string(float_format=lambda x: f"{x:.6f}"))
    print()

    quantiles = pd.cut(df["FE_ratio"], bins=[0.0, 0.1, 0.2, 0.4, 0.6, 1.0], include_lowest=True)
    by_phase = df.groupby(quantiles).agg(
        boundary_coord_fraction=("boundary_coord_fraction", "mean"),
        boundary_point_fraction=("boundary_point_fraction", "mean"),
        boundary_edge_mass=("boundary_edge_mass", "mean"),
        best_fitness=("best_fitness", "mean"),
    )
    print("by FE ratio phase")
    print(by_phase.to_string(float_format=lambda x: f"{x:.6f}"))
    print()

    corr_cols = ["boundary_coord_fraction", "boundary_point_fraction", "boundary_edge_mass", "best_fitness"]
    corr = df[corr_cols].corr(numeric_only=True)
    print("correlation matrix")
    print(corr.to_string(float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze boundary clipping accumulation in pilot trajectories.")
    parser.add_argument("config", help="YAML config used to generate the trajectories")
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    parser.add_argument("--output", type=str, default=None, help="Optional CSV path for per-row boundary statistics")
    args = parser.parse_args()

    df = analyze_config(Path(args.config), args.only_function, args.only_dimension)
    summarize(df)
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nwrote per-row boundary stats to {output_path}")


if __name__ == "__main__":
    main()
