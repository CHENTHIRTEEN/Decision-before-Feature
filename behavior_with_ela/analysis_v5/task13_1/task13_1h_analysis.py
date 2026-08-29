"""Task 13.1-H: fold-local margin calibration hygiene patch (zero new FE).

The Task 13.1 deployment noise scale pooled all development repetitions, so
in every leave-cv_group-out fold the held-out group's repetition variability
contributed to the threshold scale (OOF calibration leakage, not action-label
leakage). This patch re-estimates the action-specific scale per fold from the
training groups only, re-applies the pre-registered kappa grid, and compares
the fold-local policies against the committed pooled-scale policies.
"""
from __future__ import annotations

import json
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from behavior_with_ela.analysis_v5.task12_1_analysis import (
    SOLVERS,
    TASK12_HEAVY,
    fb_mean,
    load_shard,
)
from behavior_with_ela.analysis_v5.task13_1.task13_1_analysis import BOOTSTRAP_DRAWS

ROOT = Path(__file__).resolve().parents[3]
T13_HEAVY = ROOT / "behavior_with_ela/results/analysis_v5/task13"
T13_1_HEAVY = ROOT / "behavior_with_ela/results/analysis_v5/task13_1"
T13_1_LIGHT = ROOT / "behavior_with_ela/analysis_v5/task13_1"
BOOTSTRAP_STREAM = 2026090217
KAPPA_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)
SCALES = ("max", "sum")
LOSS_COLS = [f"loss_{s}" for s in SOLVERS]


def fold_local_scales(branches: pd.DataFrame, groups: list[str]) -> tuple[dict, pd.DataFrame]:
    """delta_{a,95}^{(-g)} per held-out group: mean over training groups of the
    per-group Q95 replicate deviation, with the Task 13.1 solver-cell mapping
    (continue cell belongs to the current solver)."""
    medians = branches.groupby(["state_id", "candidate_action"])["loss_1000"].transform("median")
    branches = branches.assign(deviation=(branches["loss_1000"] - medians).abs())
    per_group_q95 = branches.groupby(["cv_group_id", "solver_cell"])["deviation"].quantile(0.95)
    per_group_q95 = per_group_q95.unstack("solver_cell")
    scales = {}
    detail_rows = []
    for held_out in groups:
        training = [g for g in groups if g != held_out]
        block = per_group_q95.loc[per_group_q95.index.isin(training)]
        scales[held_out] = {a: float(block[a].mean()) for a in SOLVERS}
        for a in SOLVERS:
            detail_rows.append(
                {
                    "held_out_group": held_out,
                    "solver": a,
                    "delta_95_fold_local": float(block[a].mean()),
                    "n_training_groups": len(training),
                }
            )
    return scales, pd.DataFrame(detail_rows)


def main() -> None:
    T13_1_LIGHT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    states = pd.read_parquet(TASK12_HEAVY / "dynamic_screening_states.parquet")
    dataset = pd.read_parquet(T13_HEAVY / "behavior_action_dataset_task13.parquet")
    policy_rows = pd.read_parquet(T13_1_HEAVY / "margin_policy_rows.parquet")
    lookup = pd.read_parquet(T13_1_HEAVY / "strong_baseline_policy_rows.parquet")[
        ["state_id", "lookup_loss"]
    ]
    delta_ctx = pd.read_parquet(T13_HEAVY / "pairwise_delta_context.parquet").set_index("state_id")
    committed_summary = pd.read_parquet(T13_1_LIGHT / "margin_policy_summary.parquet")

    # ---- H1: fold-local action noise scales ----
    branches = load_shard("branches")
    meta = states.set_index("state_id")[["cv_group_id"]]
    branches = branches.merge(meta, left_on="state_id", right_index=True, validate="many_to_one")
    branches["solver_cell"] = np.where(
        branches["candidate_action"].eq("continue"), branches["current_algorithm"], branches["candidate_action"]
    )
    groups = sorted(dataset["cv_group_id"].unique())
    scales, scale_detail = fold_local_scales(branches, groups)
    pooled = (
        pd.read_parquet(T13_1_LIGHT / "pooled_action_noise_scale.parquet")
        .set_index("solver")["delta_pool_95"]
        .to_dict()
    )
    scale_detail["pooled_delta_95"] = scale_detail["solver"].map(pooled)
    scale_detail["abs_diff_vs_pooled"] = (
        scale_detail["delta_95_fold_local"] - scale_detail["pooled_delta_95"]
    ).abs()
    scale_detail.to_parquet(T13_1_LIGHT / "fold_local_action_noise_scale.parquet", index=False)

    # ---- H2/H3: fold-local pair scales + fixed kappa grid ----
    frame = policy_rows.merge(lookup, on="state_id", validate="many_to_one")
    frame["cv_group"] = frame["cv_group_id"]
    delta_state = delta_ctx

    def state_pair_delta(state_id: str, current: str, action: str) -> float:
        if current == action:
            return 0.0
        return float(max(delta_state.at[state_id, f"delta_cell_{current}"], delta_state.at[state_id, f"delta_cell_{action}"]))

    scale_columns = {}
    for scale in SCALES:
        d_max, d_sum = [], []
        for _, row in frame.iterrows():
            s = scales[row["cv_group"]]
            d_max.append(max(s[row["current_algorithm"]], s[row["pred_alt_algorithm"]]))
            d_sum.append(s[row["current_algorithm"]] + s[row["pred_alt_algorithm"]])
        frame[f"fold_scale_{scale}"] = d_max if scale == "max" else d_sum
    for scale in SCALES:
        for kappa in KAPPA_GRID:
            column = f"m2_fold_{scale}_k{int(kappa * 10):02d}"
            selected_list, realized_list = [], []
            solver_loss = dataset.set_index("state_id")[LOSS_COLS]
            for row in frame.itertuples(index=False):
                if row.pred_margin > kappa * getattr(row, f"fold_scale_{scale}"):
                    action = row.pred_alt_algorithm
                    loss = float(solver_loss.at[row.state_id, f"loss_{action}"])
                else:
                    action = row.current_algorithm
                    loss = float(row.continue_loss)
                selected_list.append(action)
                realized_list.append(loss)
            frame[f"selected_{column}"] = selected_list
            frame[f"realized_{column}"] = realized_list
            pair_delta = np.array(
                [
                    state_pair_delta(s, c, a)
                    for s, c, a in zip(frame["state_id"], frame["current_algorithm"], frame[f"selected_{column}"])
                ]
            )
            frame[f"harmful_{column}"] = frame[f"realized_{column}"] > (
                frame["continue_loss"].to_numpy() + pair_delta
            )
            frame[f"harmful_mass_{column}"] = np.clip(
                frame[f"realized_{column}"].to_numpy(dtype=float)
                - frame["continue_loss"].to_numpy()
                - pair_delta,
                0.0,
                None,
            )
    frame.to_parquet(T13_1_HEAVY / "fold_local_margin_policy_rows.parquet", index=False)
    lookup_map = lookup.set_index("state_id")["lookup_loss"]

    def summarize(frame: pd.DataFrame, prefix: str) -> list[dict]:
        rows = []
        for scale in SCALES:
            for kappa in KAPPA_GRID:
                column = prefix + f"{scale}_k{int(kappa * 10):02d}"
                selected_col = f"selected_{column}"
                realized_col = f"realized_{column}"
                for suite_name, group in frame.groupby("suite", sort=False):
                    is_switch = group[selected_col].ne(group["current_algorithm"])
                    required = group["switch_required"].astype(bool)
                    switched_required = group.loc[required & is_switch]
                    lookup_vals = lookup_map.reindex(group["state_id"]).to_numpy(dtype=float)
                    rows.append(
                        {
                            "policy": f"{prefix}{scale}|kappa={kappa}",
                            "scale": scale,
                            "kappa": kappa,
                            "suite": suite_name,
                            "realized_fb_loss": fb_mean(group, realized_col),
                            "gain_vs_continue": float(
                                (group["continue_loss"] - group[realized_col])
                                .groupby(group["cv_group_id"])
                                .mean()
                                .mean()
                            ),
                            "gain_vs_lookup": float(
                                pd.Series(lookup_vals - group[realized_col].to_numpy(dtype=float))
                                .groupby(group["cv_group_id"].to_numpy())
                                .mean()
                                .mean()
                            ),
                            "switch_rate": float(is_switch.mean()),
                            "harmful_rate": float(group[f"harmful_{column}"].mean()),
                            "harmful_mass": float(group[f"harmful_mass_{column}"].mean()),
                            "switch_precision": float(group.loc[is_switch, "switch_required"].mean())
                            if is_switch.any()
                            else np.nan,
                            "switch_recall": float(len(switched_required) / max(int(required.sum()), 1)),
                            "unnecessary_switch_rate": float(
                                (~group.loc[is_switch, "switch_required"]).mean()
                            )
                            if is_switch.any()
                            else np.nan,
                        }
                    )
        return rows

    # ---- reproduce the pooled summary first (stop-condition step) ----
    pooled_reproduced = True
    for _, row in committed_summary.loc[committed_summary["tag"].eq("m2")].iterrows():
        column = f"m2_k{int(row['kappa'] * 10):02d}_{row['scale']}"
        part = frame.loc[frame["suite"].eq(row["suite"])]
        value = fb_mean(part, f"realized_{column}")
        if abs(value - row["realized_fb_loss"]) > 1e-9:
            pooled_reproduced = False
    if not pooled_reproduced:
        raise SystemExit("[task13.1H] pooled-scale reproduction FAILED -> STOP")

    summary_rows = summarize(frame, "m2_fold_")
    fold_summary = pd.DataFrame(summary_rows)
    fold_summary.to_parquet(T13_1_LIGHT / "fold_local_margin_policy_summary.parquet", index=False)

    # ---- H4: fold-local vs pooled paired comparison (5000 draws) ----
    def fb_difference(frame: pd.DataFrame, upper: np.ndarray, lower: np.ndarray, stream: int) -> tuple[float, float, float]:
        work = pd.DataFrame({"cv_group_id": frame["cv_group_id"].to_numpy(), "d": upper - lower})
        groups_local = sorted(work["cv_group_id"].unique())
        means = work.groupby("cv_group_id")["d"].mean()
        rng = np.random.default_rng(
            np.random.SeedSequence([BOOTSTRAP_STREAM + stream, len(groups_local)]).generate_state(4)
        )
        draws = np.empty(BOOTSTRAP_DRAWS)
        for draw in range(BOOTSTRAP_DRAWS):
            sample = rng.choice(groups_local, size=len(groups_local), replace=True)
            draws[draw] = np.mean([means[g] for g in sample])
        return float(means.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))

    compare_rows = []
    stream = 0
    for scale in SCALES:
        for kappa in KAPPA_GRID:
            fold_col = f"realized_m2_fold_{scale}_k{int(kappa * 10):02d}"
            pooled_col = f"realized_m2_k{int(kappa * 10):02d}_{scale}"
            for suite_name, group in frame.groupby("suite", sort=False):
                point, low, high = fb_difference(
                    group,
                    group[fold_col].to_numpy(dtype=float),
                    group[pooled_col].to_numpy(dtype=float),
                    stream,
                )
                compare_rows.append(
                    {
                        "scale": scale,
                        "kappa": kappa,
                        "suite": suite_name,
                        "fb_fold_local": float(fb_mean(group, fold_col)),
                        "fb_pooled": float(fb_mean(group, pooled_col)),
                        "fold_minus_pooled": point,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
                stream += 1
    comparison = pd.DataFrame(compare_rows)
    comparison.to_parquet(T13_1_LIGHT / "fold_local_vs_pooled_bootstrap.parquet", index=False)

    elapsed = perf_counter() - started
    pd.DataFrame(
        [
            {
                "phase": "task13_1h_hygiene_patch",
                "new_objective_fe": 0,
                "wall_seconds": elapsed,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                "note": "fold-local calibration only; no replay, no refit, no new kappa",
            }
        ]
    ).to_parquet(T13_1_HEAVY / "task13_1h_resource_ledger.parquet", index=False)

    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print("=== fold-local scale spread vs pooled ===")
        spread = scale_detail.pivot_table(index="held_out_group", columns="solver", values="delta_95_fold_local")
        print(spread.round(4).to_string())
        print("per-solver abs diff vs pooled:", scale_detail.groupby("solver")["abs_diff_vs_pooled"].mean().round(4).to_dict())
        print("=== fold-local policy summary (m2) ===")
        print(fold_summary.round(4).to_string())
        print("=== fold-local vs pooled (fold_minus_pooled, fb loss difference) ===")
        print(comparison.round(5).to_string())


if __name__ == "__main__":
    main()
