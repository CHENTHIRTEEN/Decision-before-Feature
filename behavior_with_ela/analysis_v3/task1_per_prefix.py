"""Task 9A: per-prefix deployability of behavior_action_loss_regression_v2.

Decomposes the fixed v2 first-trigger policy by initial (prefix) algorithm on
train grouped-family OOF and the untouched BBOB validation split. Reuses the
existing run-level artifacts only; no objective evaluation is executed.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import common  # noqa: F401
from common import (
    json_dumps,
    load_v2_first_trigger_runs,
    policy_metrics,
    save_table,
    target_distribution,
)

TASK = "task1"


def prefix_block(runs: pd.DataFrame, deltas: dict[str, float]) -> pd.DataFrame:
    rows = []
    for prefix, group in runs.groupby("prefix_algorithm", sort=True):
        triggered = group.loc[group["switch_triggered"].astype(bool)]
        selected_fe = triggered["selected_FE"].dropna()
        rows.append(
            {
                "prefix_algorithm": str(prefix),
                **policy_metrics(group, deltas),
                "selected_fe_p10": (
                    float(np.quantile(selected_fe.to_numpy(dtype=float), 0.10))
                    if len(selected_fe)
                    else float("nan")
                ),
                "selected_fe_p50": (
                    float(np.quantile(selected_fe.to_numpy(dtype=float), 0.50))
                    if len(selected_fe)
                    else float("nan")
                ),
                "selected_fe_p90": (
                    float(np.quantile(selected_fe.to_numpy(dtype=float), 0.90))
                    if len(selected_fe)
                    else float("nan")
                ),
                "stay_with_prefix_rate": float(
                    (~group["switch_triggered"].astype(bool)).mean()
                ),
                "mean_selected_gain": float(
                    group["selected_action_gain"].mean()
                ),
                "median_selected_gain": float(
                    group["selected_action_gain"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    train_runs, validation_runs, threshold = load_v2_first_trigger_runs()
    config = common.load_experiment_config(common.TRAIN_CONFIG)
    deltas = common.noise_deltas(config)

    summary_rows = []
    target_rows = []
    harmful_rows = []
    for split, runs in (
        ("bbob_train_oof", train_runs),
        ("bbob_validation", validation_runs),
    ):
        prefix_table = prefix_block(runs, deltas)
        prefix_table.insert(0, "evaluation_split", split)
        summary_rows.append(prefix_table)
        target = target_distribution(runs)
        target.insert(0, "evaluation_split", split)
        target_rows.append(target)
        triggered = runs.loc[runs["switch_triggered"].astype(bool)]
        for prefix, group in triggered.groupby("prefix_algorithm", sort=True):
            gains = group["selected_action_gain"].to_numpy(dtype=float)
            harmful_rows.append(
                {
                    "evaluation_split": split,
                    "prefix_algorithm": str(prefix),
                    "triggered_runs": int(len(group)),
                    "gain_below_zero": int((gains < 0.0).sum()),
                    "gain_below_delta_50": int(
                        (gains < -deltas["delta_50"]).sum()
                    ),
                    "gain_below_delta_95": int(
                        (gains < -deltas["delta_95"]).sum()
                    ),
                    "rate_below_zero": float((gains < 0.0).mean()),
                    "rate_below_delta_50": float(
                        (gains < -deltas["delta_50"]).mean()
                    ),
                    "rate_below_delta_95": float(
                        (gains < -deltas["delta_95"]).mean()
                    ),
                }
            )

    summary = pd.concat(summary_rows, ignore_index=True)
    targets = pd.concat(target_rows, ignore_index=True)
    harmful = pd.DataFrame(harmful_rows)
    save_table(summary, "per_prefix_policy_summary.parquet", TASK)
    save_table(targets, "per_prefix_target_distribution.parquet", TASK)
    save_table(harmful, "per_prefix_harmful_rates.parquet", TASK)

    # overall switch decomposition on both splits
    decomposition = {}
    for split, runs in (
        ("bbob_train_oof", train_runs),
        ("bbob_validation", validation_runs),
    ):
        by_prefix = runs.groupby("prefix_algorithm")["switch_triggered"].mean()
        target_of_switch = (
            runs.loc[runs["switch_triggered"].astype(bool)]
            .groupby(["prefix_algorithm", "selected_algorithm"])
            .size()
        )
        decomposition[split] = {
            "switch_rate_by_prefix": {key: float(value) for key, value in by_prefix.items()},
            "target_counts": {
                f"{prefix}->{target}": int(count)
                for (prefix, target), count in target_of_switch.items()
            },
            "overall_switch_rate": float(runs["switch_triggered"].mean()),
        }

    payload = {
        "threshold": threshold,
        "noise_deltas": deltas,
        "decomposition": decomposition,
        "policy_summary": json.loads(summary.to_json(orient="records")),
    }
    save_table(payload, "summary.json", TASK)
    print(f"[{TASK}] done", flush=True)
    print(summary.to_string(), flush=True)
    print(json_dumps(decomposition), flush=True)


if __name__ == "__main__":
    main()
