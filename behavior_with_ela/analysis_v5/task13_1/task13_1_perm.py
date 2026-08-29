"""Task 13.1J: within-problem permutation study with N_perm = 100 (RF carrier).

Identical protocol to the Task 13 O2 control: permute the behavior values
within each (problem, current, FE) group, re-run the 4-train/1-test
leave-one-seed-out comparison of the W0 group-mean baseline against the W2
behavior policy, and record the function-balanced delta per suite. W1 is
omitted here because it is permutation-irrelevant for the W0-vs-W2 null;
the W2 leg uses exactly the same carrier, folds and feature set as Task 13.
The unpermuted delta is recomputed once with the same code as a consistency
check against the committed Task 13 result.
"""
from __future__ import annotations

import json
import resource
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS
from behavior_with_ela.analysis_v5.task13.task13_analysis import make_carrier

ROOT = Path(__file__).resolve().parents[3]
T13_HEAVY = ROOT / "behavior_with_ela/results/analysis_v5/task13"
T13_1_LIGHT = ROOT / "behavior_with_ela/analysis_v5/task13_1"
N_PERM = 100
WORKERS = 8
LOSS_COLS = [f"loss_shade", "loss_lshade", "loss_cso"]


def load_dataset() -> pd.DataFrame:
    dataset = pd.read_parquet(T13_HEAVY / "behavior_action_dataset_task13.parquet")
    dummies = pd.get_dummies(dataset["current_algorithm"], prefix="cur", dtype=float)
    return pd.concat([dataset.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)


def w2_within_loso(dataset: pd.DataFrame, bf_cols: list[str], behavior_perm: np.ndarray | None) -> pd.DataFrame:
    """Return per-state realized losses of W0 (group-train mean) and W2 under
    the leave-one-seed-out protocol; behavior_perm optionally permutes the
    behavior matrix rows beforehand."""
    truth = dataset[LOSS_COLS].to_numpy(dtype=float)
    contexts = dataset[["cur_shade", "cur_lshade", "cur_cso", "FE_ratio"]].to_numpy(dtype=float)
    behavior = dataset[bf_cols].to_numpy(dtype=float)
    if behavior_perm is not None:
        behavior = behavior[behavior_perm]
    X = np.hstack([contexts, behavior])
    records = []
    for keys, positions in dataset.groupby(["problem_id", "current_algorithm", "FE"], sort=False).groups.items():
        idx = np.asarray(positions)
        for pos in idx:
            train = idx[idx != pos]
            train_mean = truth[train].mean(axis=0)
            model = make_carrier("rf")
            model.fit(X[train], truth[train])
            prediction = model.predict(X[pos : pos + 1])[0]
            records.append(
                {
                    "state_id": dataset.at[pos, "state_id"],
                    "suite": dataset.at[pos, "suite"],
                    "cv_group_id": dataset.at[pos, "cv_group_id"],
                    "realized_W0": float(truth[pos, int(np.argmin(train_mean))]),
                    "realized_W2": float(truth[pos, int(np.argmin(prediction))]),
                }
            )
    return pd.DataFrame(records)


def run_repeats(repeat_ids: list[int]) -> list[dict]:
    dataset = load_dataset()
    bf_cols = [c for c in dataset.columns if c in set(SELECTOR_BEHAVIOR_FEATURE_COLUMNS)]
    rng = np.random.default_rng(np.random.SeedSequence([2026090216, repeat_ids[0]]).generate_state(4))
    group_keys = dataset.groupby(["problem_id", "current_algorithm", "FE"], sort=False).groups
    rows = []
    for repeat in repeat_ids:
        perm_index = np.arange(len(dataset))
        for _, idx in group_keys.items():
            positions = np.asarray(idx)
            perm_index[positions] = positions[rng.permutation(len(positions))]
        result = w2_within_loso(dataset, bf_cols, behavior_perm=perm_index)
        for suite_name, group in result.groupby("suite", sort=False):
            delta = float(
                (group["realized_W0"] - group["realized_W2"])
                .groupby(group["cv_group_id"])
                .mean()
                .mean()
            )
            rows.append({"repeat": repeat, "suite": suite_name, "delta_within_perm": delta})
        print(f"[perm] repeat {repeat} done", flush=True)
    return rows


def main() -> None:
    T13_1_LIGHT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    dataset = load_dataset()
    bf_cols = [c for c in dataset.columns if c in set(__import__("behavior.features", fromlist=["SELECTOR_BEHAVIOR_FEATURE_COLUMNS"]).SELECTOR_BEHAVIOR_FEATURE_COLUMNS)]

    observed = w2_within_loso(dataset, bf_cols, behavior_perm=None)
    observed_rows = []
    for suite_name, group in observed.groupby("suite", sort=False):
        observed_rows.append(
            {
                "suite": suite_name,
                "delta_within_observed": float(
                    (group["realized_W0"] - group["realized_W2"])
                    .groupby(group["cv_group_id"])
                    .mean()
                    .mean()
                ),
            }
        )
    observed_table = pd.DataFrame(observed_rows)
    observed_table.to_parquet(T13_1_LIGHT / "within_problem_permutation_observed.parquet", index=False)

    chunks = [list(range(i, N_PERM, WORKERS)) for i in range(WORKERS)]
    all_rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for part in executor.map(run_repeats, chunks):
            all_rows.extend(part)
    perm_table = pd.DataFrame(all_rows).sort_values(["suite", "repeat"]).reset_index(drop=True)
    perm_table.to_parquet(T13_1_LIGHT / "within_problem_permutation_100.parquet", index=False)

    summary = []
    for suite_name, group in perm_table.groupby("suite", sort=False):
        deltas = group["delta_within_perm"].to_numpy()
        delta_obs = float(observed_table.loc[observed_table["suite"].eq(suite_name), "delta_within_observed"].iloc[0])
        p_value = float((1 + int(np.sum(deltas >= delta_obs))) / (1 + N_PERM))
        summary.append(
            {
                "suite": suite_name,
                "n_perm": N_PERM,
                "delta_within_observed": delta_obs,
                "null_mean": float(deltas.mean()),
                "null_std": float(deltas.std(ddof=1)),
                "null_q95": float(np.quantile(deltas, 0.95)),
                "null_q975": float(np.quantile(deltas, 0.975)),
                "empirical_p_one_sided": p_value,
            }
        )
    summary_table = pd.DataFrame(summary)
    summary_table.to_parquet(T13_1_LIGHT / "within_problem_permutation_summary.parquet", index=False)
    summary_table["wall_seconds"] = perf_counter() - started
    summary_table["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(summary_table.round(5).to_string())


if __name__ == "__main__":
    main()
