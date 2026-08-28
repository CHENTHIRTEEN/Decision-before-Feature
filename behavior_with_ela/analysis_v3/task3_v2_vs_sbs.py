"""Task 9C: complete policy-level comparison of the v2 carrier against SBS.

SBS = the train-derived static best solver (cmaes) run from FE=0 to FE_total.
Two scenarios: all-prefix diagnostic and the deployable CMAES-start scenario.
All quantities are recomputed from existing run-level artifacts.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import common  # noqa: F401
from common import (
    BOOTSTRAP_STREAM,
    json_dumps,
    load_v2_first_trigger_runs,
    noise_deltas,
    save_heavy_table,
    save_table,
)

TASK = "task3"
BOOTSTRAP_REPLICATES = 10_000


def scenario_runs(runs: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if scenario == "all_prefix":
        return runs
    if scenario == "cmaes_start":
        return runs.loc[runs["prefix_algorithm"].astype(str).eq("cmaes")]
    raise ValueError(scenario)


def paired_table(runs: pd.DataFrame, deltas: dict[str, float]) -> pd.DataFrame:
    merged = common.gain_over_sbs(runs)
    band = float(deltas["delta_50"])
    gain = merged["gain_over_sbs"].to_numpy(dtype=float)
    merged["comparison_band"] = np.select(
        [gain > band, gain < -band],
        ["v2_better", "v2_worse"],
        default="equivalent",
    )
    return merged


def scenario_summary(
    paired: pd.DataFrame,
    deltas: dict[str, float],
    vbs_loss: float,
) -> dict:
    band = float(deltas["delta_50"])
    gain = paired["gain_over_sbs"].to_numpy(dtype=float)
    policy_fb = common.function_balanced(
        paired["selected_terminal_log10_loss"], paired["cv_group_id"]
    )
    sbs_fb = common.function_balanced(
        paired["sbs_terminal_log10_loss"], paired["cv_group_id"]
    )
    denominator = sbs_fb - vbs_loss
    return {
        "run_count": int(len(paired)),
        "function_balanced_v2_terminal_log10_loss": float(policy_fb),
        "function_balanced_sbs_terminal_log10_loss": float(sbs_fb),
        "function_balanced_paired_gain": common.function_balanced(
            paired["gain_over_sbs"], paired["cv_group_id"]
        ),
        "median_paired_gain": float(np.median(gain)),
        "mean_paired_gain": float(np.mean(gain)),
        "fraction_v2_better_beyond_delta50": float((gain > band).mean()),
        "fraction_equivalent_within_delta50": float(
            (np.abs(gain) <= band).mean()
        ),
        "fraction_v2_worse_beyond_delta50": float((gain < -band).mean()),
        "fraction_v2_better_raw": float((gain > 0.0).mean()),
        "fraction_v2_worse_raw": float((gain < 0.0).mean()),
        "v2_success_rate": float(
            (
                paired["selected_terminal_log10_loss"].to_numpy(dtype=float)
                <= common.SUCCESS_LOG10_TARGET + 1e-12
            ).mean()
        ),
        "sbs_success_rate": float(
            (
                paired["sbs_terminal_log10_loss"].to_numpy(dtype=float)
                <= common.SUCCESS_LOG10_TARGET + 1e-12
            ).mean()
        ),
        "vbs_sbs_gap_closed_fraction": (
            float((sbs_fb - policy_fb) / denominator) if denominator > 1e-12 else float("nan")
        ),
        "vbs_function_balanced_terminal_log10_loss": float(vbs_loss),
    }


def function_level(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for function, group in paired.groupby("cv_group_id", sort=True):
        gain = group["gain_over_sbs"].to_numpy(dtype=float)
        rows.append(
            {
                "cv_group_id": str(function),
                "runs": int(len(group)),
                "mean_paired_gain": float(gain.mean()),
                "median_paired_gain": float(np.median(gain)),
                "share_v2_better_raw": float((gain > 0.0).mean()),
                "mean_v2_terminal_log10_loss": float(
                    group["selected_terminal_log10_loss"].mean()
                ),
                "mean_sbs_terminal_log10_loss": float(
                    group["sbs_terminal_log10_loss"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(
    paired: pd.DataFrame,
    *,
    scenario: str,
    split: str,
) -> dict:
    per_function = (
        paired.groupby("cv_group_id")["gain_over_sbs"].mean().to_numpy(dtype=float)
    )
    count = per_function.size
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                BOOTSTRAP_STREAM,
                1 if split == "bbob_train_oof" else 2,
                1 if scenario == "all_prefix" else 2,
            ]
        ).generate_state(4)
    )
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sample = per_function[rng.integers(0, count, size=count)]
        draws[index] = float(sample.mean())
    return {
        "scenario": scenario,
        "split": split,
        "function_count": int(count),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
        "point_function_balanced": float(per_function.mean()),
    }


def main() -> None:
    config = common.load_experiment_config(common.TRAIN_CONFIG)
    deltas = noise_deltas(config)
    train_runs, validation_runs, threshold = load_v2_first_trigger_runs()
    static = pd.read_parquet(
        common.RESULTS / "baselines/phase1/static_portfolio_summary.parquet"
    )
    vbs_loss = {
        (row["evaluation_split"], row["portfolio_reference"]): float(
            row["function_balanced_mean_log10_gap"]
        )
        for _, row in static.iterrows()
    }

    # consistency check: SBS from continue rows vs the static portfolio summary
    consistency = {}
    for split, runs in (
        ("bbob_train_oof", train_runs),
        ("bbob_validation", validation_runs),
    ):
        reference = common.sbs_reference(runs)
        from_continue = float(reference["sbs_terminal_log10_loss"].mean())
        static_value = float(
            static.loc[
                static["evaluation_split"].eq(
                    "train" if split == "bbob_train_oof" else "validation"
                )
                & static["portfolio_reference"].eq("sbs"),
                "function_balanced_mean_log10_gap",
            ].iloc[0]
        )
        consistency[split] = {
            "sbs_mean_from_continue_rows": from_continue,
            "sbs_static_summary_value": static_value,
            "absolute_difference": abs(from_continue - static_value),
        }

    per_function_rows = []
    bootstrap_rows = []
    summaries = []
    paired_frames = []
    for split, runs in (
        ("bbob_train_oof", train_runs),
        ("bbob_validation", validation_runs),
    ):
        for scenario in ("all_prefix", "cmaes_start"):
            subset = scenario_runs(runs, scenario)
            paired = paired_table(subset, deltas)
            paired["scenario"] = scenario
            paired["evaluation_split"] = split
            paired_frames.append(paired)
            vbs = vbs_loss[
                ("train" if split == "bbob_train_oof" else "validation", "vbs")
            ]
            summary = scenario_summary(paired, deltas, vbs)
            summary["evaluation_split"] = split
            summary["scenario"] = scenario
            summaries.append(summary)
            function_table = function_level(paired)
            function_table.insert(0, "evaluation_split", split)
            function_table.insert(0, "scenario", scenario)
            per_function_rows.append(function_table)
            bootstrap_rows.append(
                bootstrap_ci(paired, scenario=scenario, split=split)
            )

    paired_all = pd.concat(paired_frames, ignore_index=True)
    save_heavy_table(paired_all, "v2_vs_sbs_paired_runs.parquet", TASK)
    save_table(pd.concat(per_function_rows, ignore_index=True), "per_function_paired_gain.parquet", TASK)
    save_table(pd.DataFrame(bootstrap_rows), "bootstrap_ci.parquet", TASK)
    save_table(summaries, "scenario_summary.json", TASK)
    save_table(consistency, "sbs_consistency_check.json", TASK)

    # selected-algorithm decomposition for the deployable scenario
    cmaes_start = paired_all.loc[
        paired_all["scenario"].eq("cmaes_start")
    ]
    decomposition = (
        cmaes_start.groupby(["evaluation_split", "selected_algorithm"])
        .size()
        .reset_index(name="runs")
    )
    save_table(decomposition, "cmaes_start_selected_algorithm.parquet", TASK)
    print(f"[{TASK}] done; threshold={threshold:.6f}", flush=True)
    print(json_dumps(summaries), flush=True)
    print(json_dumps(consistency), flush=True)
    print(json_dumps(bootstrap_rows), flush=True)
    print(decomposition.to_string(), flush=True)


if __name__ == "__main__":
    main()
