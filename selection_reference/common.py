from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from experiments.phase1_batch_common import load_config, make_shards
from trajectory.records import OPTIMIZER_STATE_MODE


def read_performance(
    config: dict,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
) -> pd.DataFrame:
    frames = []
    for shard in make_shards(config, only_functions, only_dimensions):
        if not shard.output_path.exists():
            raise FileNotFoundError(f"missing trajectory shard: {shard.output_path}")
        if "optimizer_state_mode" not in pq.read_schema(shard.output_path).names:
            raise ValueError(f"trajectory shard predates native optimizer-state continuation: {shard.output_path}")
        table = pq.read_table(
            shard.output_path,
            columns=[
                "problem_id",
                "family",
                "dimension",
                "algorithm",
                "seed",
                "FE",
                "FE_ratio",
                "best_fitness",
                "optimizer_state_mode",
            ],
        )
        frame = table.to_pandas()
        if set(frame["optimizer_state_mode"].astype(str)) != {OPTIMIZER_STATE_MODE}:
            raise ValueError(f"trajectory shard does not use native optimizer-state continuation: {shard.output_path}")
        frames.append(frame.drop(columns=["optimizer_state_mode"]))
    if not frames:
        raise ValueError("no trajectory rows available for selection reference")
    return pd.concat(frames, ignore_index=True)


def single_best_solver(performance: pd.DataFrame) -> str:
    final = performance.sort_values("FE_ratio").groupby(["problem_id", "algorithm"], as_index=False).tail(1)
    means = final.groupby(["problem_id", "algorithm"], as_index=False)["best_fitness"].mean()
    means["rank"] = means.groupby("problem_id")["best_fitness"].rank(method="average", ascending=True)
    ranks = means.groupby("algorithm")["rank"].mean().sort_values()
    return str(ranks.index[0])


def train_derived_sbs(train_config_path: Path) -> str:
    config = load_config(train_config_path)
    return single_best_solver(read_performance(config, None, None))


def split_name(config: dict) -> str:
    if "split" in config:
        return str(config["split"])
    return Path(config["output"]).stem.removesuffix("_trajectories")
