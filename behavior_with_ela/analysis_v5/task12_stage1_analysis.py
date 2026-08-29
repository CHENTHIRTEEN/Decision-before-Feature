"""Task 12 Stage 1 analysis: strength, static pairwise complementarity (DCM),
rule-based screening and the pre-registered DE-GA hypothesis test.

Practical relations use per-FE noise deltas calibrated from the Stage-1
repetition subset (deterministic 10 percent, R=3, function-balanced q50/q95).
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SHARDS = ROOT / "behavior_with_ela/results/portfolio_screening/task12/stage1/shards"
HEAVY = ROOT / "behavior_with_ela/results/analysis_v5/task12"
LIGHT = ROOT / "behavior_with_ela/analysis_v5/task12"
CANDIDATES = ("pso", "lbestpso", "de", "shade", "lshade", "ga", "cso")
FES = (2000, 4000, 6000, 10000)
BOOTSTRAP_STREAM = 2026083011


def json_write(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float))


def load_runs() -> pd.DataFrame:
    frames = []
    for shard in sorted(SHARDS.iterdir()):
        if shard.is_dir():
            frames.append(pd.read_parquet(shard / "runs.parquet"))
    return pd.concat(frames, ignore_index=True)


def function_balanced(frame: pd.DataFrame, column: str) -> float:
    return float(frame.groupby("cv_group_id")[column].mean().mean())


def noise_deltas(runs: pd.DataFrame) -> pd.DataFrame:
    repeated = runs.loc[runs["replicate_id"].gt(0) | runs["replicate_id"].eq(0)]
    grouped = (
        repeated.groupby(["suite", "cv_group_id", "candidate", "seed", "FE"])[
            "log10_gap"
        ]
        .agg(["median", "count"])
        .reset_index()
    )
    merged = repeated.merge(
        grouped, on=["suite", "cv_group_id", "candidate", "seed", "FE"], how="left"
    )
    merged["absolute_deviation"] = (
        merged["log10_gap"] - merged["median"]
    ).abs()
    valid = merged.loc[merged["count"].gt(1)]
    rows = []
    for (suite, fe), group in valid.groupby(["suite", "FE"], sort=False):
        per_function = group.groupby("cv_group_id")["absolute_deviation"]
        quantiles = per_function.quantile([0.50, 0.95]).unstack()
        rows.append(
            {
                "FE": int(fe),
                "suite": suite,
                "delta_50_function_balanced": float(quantiles[0.50].mean()),
                "delta_95_function_balanced": float(quantiles[0.95].mean()),
                "states": int(len(group)),
            }
        )
    table = pd.DataFrame(rows)
    return table


def main() -> None:
    HEAVY.mkdir(parents=True, exist_ok=True)
    LIGHT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    base = runs.loc[runs["replicate_id"].eq(0)].copy()
    deltas = noise_deltas(runs)
    deltas.to_parquet(HEAVY / "stage1_noise_deltas.parquet", index=False)
    delta_lookup = {
        (int(row.FE), row.suite): float(row.delta_95_function_balanced)
        for row in deltas.itertuples()
    }

    # ---- performance table ----
    perf_rows = []
    pivot_by_fe = {}
    for fe in FES:
        subset = base.loc[base["FE"].eq(fe)]
        pivot = subset.pivot_table(
            index=["problem_id", "seed"], columns="candidate",
            values="log10_gap", aggfunc="first",
        )
        meta = subset.drop_duplicates(["problem_id", "seed"])[
            ["problem_id", "seed", "suite", "cv_group_id", "family"]
        ]
        pivot = pivot.join(meta.set_index(["problem_id", "seed"]))
        pivot_by_fe[fe] = pivot
        vbs = pivot[list(CANDIDATES)].min(axis=1)
        previous = pivot_by_fe.get(fe - 1000)
        for candidate in CANDIDATES:
            losses = pivot[candidate]
            pool = [name for name in CANDIDATES if name != candidate]
            delta = deltas.copy()
            suite_specific = subset
            # practical winner stats with per-suite deltas
            win_counts = 0
            top2_counts = 0
            exclusive_counts = 0
            exclusive_mass = []
            for suite_name in subset["suite"].unique():
                d = float(
                    delta_lookup.get((fe, suite_name), np.nan)
                )
                block = pivot.loc[pivot["suite"].eq(suite_name), list(CANDIDATES)]
                others = block[pool].min(axis=1)
                win_counts += int((block[candidate] < others - d).sum())
                ranked = block.rank(axis=1, method="min")
                top2_counts += int((ranked[candidate] <= 2.0).sum())
                exclusive_counts += int(
                    block[pool]
                    .gt(block[candidate] + d, axis=0)
                    .all(axis=1)
                    .sum()
                )
                exclusive_mass.extend(
                    (others - block[candidate] - d).clip(lower=0.0).tolist()
                )
            stagnation = 0.0
            if previous is not None:
                same = (
                    pivot[candidate].to_numpy()
                    >= previous.loc[pivot.index, candidate].to_numpy() - 1e-12
                )
                stagnation = float(same.mean())
            perf_rows.append(
                {
                    "FE": fe,
                    "candidate": candidate,
                    "mean_log10_gap": float(losses.mean()),
                    "median_log10_gap": float(losses.median()),
                    "function_balanced_log10_gap": function_balanced(subset.loc[subset["candidate"].eq(candidate)], "log10_gap"),
                    "practical_win_rate": win_counts / max(len(pivot), 1),
                    "top2_rate": top2_counts / max(len(pivot), 1),
                    "exclusive_practical_win_rate": exclusive_counts / max(len(pivot), 1),
                    "exclusive_gain_mass_median": float(np.median(exclusive_mass)) if exclusive_mass else 0.0,
                    "exclusive_gain_mass_mean": float(np.mean(exclusive_mass)) if exclusive_mass else 0.0,
                    "mean_regret_to_vbs": float((losses - vbs).mean()),
                    "stagnation_rate_per_1000fe": stagnation,
                }
            )
    performance = pd.DataFrame(perf_rows)
    performance.to_parquet(HEAVY / "natural_candidate_performance.parquet", index=False)

    # ---- static pairwise DCM ----
    dcm_rows = []
    for fe in FES:
        pivot = pivot_by_fe[fe]
        for suite_name in pivot["suite"].unique():
            d = float(delta_lookup.get((fe, suite_name), np.nan))
            block = pivot.loc[pivot["suite"].eq(suite_name), list(CANDIDATES)]
            for a, b in combinations(CANDIDATES, 2):
                diff = block[a] - block[b]
                tie = float((diff.abs() <= d).mean())
                a_better = float((diff < -d).mean())
                b_better = float((diff > d).mean())
                c = a_better - b_better
                dcm_rows.append(
                    {
                        "FE": fe,
                        "suite": suite_name,
                        "candidate_a": a,
                        "candidate_b": b,
                        "P_tie": tie,
                        "P_a_better": a_better,
                        "P_b_better": b_better,
                        "C": c,
                        "DCM": float((tie + abs(c)) / 2.0),
                    }
                )
    dcm = pd.DataFrame(dcm_rows)
    dcm.to_parquet(HEAVY / "static_pairwise_dcm.parquet", index=False)

    # ---- marginal VBS value (leave-one-out, per FE) ----
    marginal_rows = []
    for fe in FES:
        pivot = pivot_by_fe[fe]
        for candidate in CANDIDATES:
            pool = [name for name in CANDIDATES if name != candidate]
            with_candidate = pivot[list(CANDIDATES)].min(axis=1)
            without = pivot[pool].min(axis=1)
            delta = without - with_candidate
            marginal_rows.append(
                {
                    "FE": fe,
                    "candidate": candidate,
                    "marginal_vbs_improvement_mean": float(delta.mean()),
                    "marginal_vbs_improvement_median": float(delta.median()),
                    "positive_share": float((delta > 0).mean()),
                }
            )
    marginal = pd.DataFrame(marginal_rows)
    marginal.to_parquet(HEAVY / "marginal_vbs.parquet", index=False)

    # ---- H-DEGA pre-registered test ----
    dega = {
        "static": [],
    }
    for fe in FES:
        for suite_name in pivot_by_fe[fe]["suite"].unique():
            d = float(delta_lookup.get((fe, suite_name), np.nan))
            block = pivot_by_fe[fe].loc[pivot_by_fe[fe]["suite"].eq(suite_name), list(CANDIDATES)]
            diff = block["de"] - block["ga"]
            tie = float((diff.abs() <= d).mean())
            de_better = float((diff < -d).mean())
            ga_better = float((diff > d).mean())
            dega["static"].append(
                {
                    "FE": fe,
                    "suite": suite_name,
                    "P_tie": tie,
                    "P_de_better": de_better,
                    "P_ga_better": ga_better,
                    "DCM": float((tie + abs(de_better - ga_better)) / 2.0),
                    "de_exclusive_win": float(
                        block.drop(columns=["de"]).gt(block["de"] + d, axis=0).all(axis=1).mean()
                    ),
                    "ga_exclusive_win": float(
                        block.drop(columns=["ga"]).gt(block["ga"] + d, axis=0).all(axis=1).mean()
                    ),
                }
            )
    json_write(dega, LIGHT / "h_dega_static.json")

    # dump light tables
    performance.to_parquet(LIGHT / "natural_candidate_performance.parquet", index=False)
    dcm.to_parquet(LIGHT / "static_pairwise_dcm.parquet", index=False)
    marginal.to_parquet(LIGHT / "marginal_vbs.parquet", index=False)
    strength = performance.copy()
    strength.to_parquet(LIGHT / "candidate_strength_metrics.parquet", index=False)

    print(performance.loc[performance.FE.eq(6000)].round(4).to_string())
    print(dcm.loc[(dcm.FE.eq(6000)) & (dcm.suite.eq("bbob"))].round(3).to_string())
    print(deltas.round(4).to_string())


if __name__ == "__main__":
    main()
