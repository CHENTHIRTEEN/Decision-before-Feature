"""Task 13.1 A-I/M/N: strong deployment baselines, predicted-margin risk
policies, calibration and Pareto diagnostics, transition risk stratification.

Zero new objective FE: every quantity reuses the Task 12/13 artifacts. The
deployment margin scale is derived only from training-domain repetitions of
the stage-2 branches, pooled over both suites (no suite label, no problem
identity, no validation statistics enter the policy constants). The margin
policies are a fixed pre-registered kappa grid; no threshold is selected.
"""
from __future__ import annotations

import json
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS
from behavior_with_ela.analysis_v5.task12_1_analysis import (
    SOLVERS,
    TASK12_HEAVY,
    assignment_loss,
    fb_mean,
    load_shard,
    oof_assignment,
)
from behavior_with_ela.analysis_v5.task13.task13_analysis import (
    make_carrier,
    run_grouped_oof,
)

ROOT = Path(__file__).resolve().parents[3]
T13_HEAVY = ROOT / "behavior_with_ela/results/analysis_v5/task13"
T13_1_HEAVY = ROOT / "behavior_with_ela/results/analysis_v5/task13_1"
T13_1_LIGHT = ROOT / "behavior_with_ela/analysis_v5/task13_1"
BOOTSTRAP_STREAM = 2026090215
BOOTSTRAP_DRAWS = 5000
KAPPA_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)
SCALES = ("max", "sum")
LOSS_COLS = [f"loss_{s}" for s in SOLVERS]


def fb_of_difference(frame: pd.DataFrame, upper: np.ndarray, lower: np.ndarray) -> tuple[float, float, float]:
    work = pd.DataFrame({"cv_group_id": frame["cv_group_id"].to_numpy(), "d": upper - lower})
    groups = sorted(work["cv_group_id"].unique())
    means = work.groupby("cv_group_id")["d"].mean()
    rng = np.random.default_rng(
        np.random.SeedSequence([BOOTSTRAP_STREAM, len(groups)]).generate_state(4)
    )
    draws = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        sample = rng.choice(groups, size=len(groups), replace=True)
        draws[draw] = np.mean([means[g] for g in sample])
    point = float(means.mean())
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    T13_1_HEAVY.mkdir(parents=True, exist_ok=True)
    T13_1_LIGHT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    dataset = pd.read_parquet(T13_HEAVY / "behavior_action_dataset_task13.parquet")
    preds = pd.read_parquet(T13_HEAVY / "oof_policy_rows.parquet")
    delta_ctx = pd.read_parquet(T13_HEAVY / "pairwise_delta_context.parquet")

    # ------------------------------------------------------------------
    # A1: reproduce the RF M2 OOF from the fixed Task 13 code path
    # ------------------------------------------------------------------
    bf_cols = [c for c in dataset.columns if c in set(SELECTOR_BEHAVIOR_FEATURE_COLUMNS)]
    if len(bf_cols) != 28:
        raise SystemExit("[task13.1A] behavior contract mismatch")
    dummies = pd.get_dummies(dataset["current_algorithm"], prefix="cur", dtype=float)
    dataset = pd.concat([dataset.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    cur_cols = list(dummies.columns)
    m2_features = [*cur_cols, "FE_ratio", *bf_cols]
    fresh = run_grouped_oof(dataset, {"M2_current_FE_behavior": m2_features}, "rf")
    saved_m2 = (
        preds.loc[preds["model"].eq("M2_current_FE_behavior") & preds["carrier"].eq("rf")][
            ["state_id", "pred_shade", "pred_lshade", "pred_cso", "selected", "realized_loss"]
        ]
        .sort_values("state_id")
        .reset_index(drop=True)
    )
    fresh_sorted = fresh.sort_values("state_id").reset_index(drop=True)
    pred_diff = float(
        np.max(
            np.abs(
                fresh_sorted[["pred_shade", "pred_lshade", "pred_cso"]].to_numpy(dtype=float)
                - saved_m2[["pred_shade", "pred_lshade", "pred_cso"]].to_numpy(dtype=float)
            )
        )
    )
    selected_match = bool((fresh_sorted["selected"] == saved_m2["selected"]).all())
    realized_match = bool(np.allclose(fresh_sorted["realized_loss"], saved_m2["realized_loss"], atol=1e-12))
    audit = {
        "head_commit": "c55563f",
        "m2_oof_reproduced": bool(pred_diff <= 1e-12 and selected_match and realized_match),
        "max_pred_abs_diff": pred_diff,
        "selected_all_match": selected_match,
        "realized_all_match": realized_match,
    }

    # ------------------------------------------------------------------
    # B1: empirical current+FE OOF lookup (must reproduce Task 12.1)
    # ------------------------------------------------------------------
    solver = pd.read_parquet(TASK12_HEAVY / "dynamic_solver_loss_matrix.parquet")
    states = pd.read_parquet(TASK12_HEAVY / "dynamic_screening_states.parquet")
    frame = solver.merge(states[["state_id", "family", "instance"]], on="state_id", validate="many_to_one")
    lookup_parts = []
    lookup_check = {}
    ladder_ref = pd.read_parquet(
        ROOT / "behavior_with_ela/analysis_v5/task12_1/oracle_ladder_current_conditioned.parquet"
    ).set_index("suite")["L_current_FE_oof"]
    reference = {suite: float(ladder_ref.loc[suite]) for suite in ("bbob", "mabbob")}
    for suite_name, group in frame.groupby("suite", sort=False):
        group = group.reset_index(drop=True)
        assignment = oof_assignment(group, ["current_algorithm", "FE"], SOLVERS)
        group["lookup_loss"] = assignment_loss(group, assignment)
        value = fb_mean(group, "lookup_loss")
        lookup_check[suite_name] = {
            "fb_lookup_loss": value,
            "reference": reference[suite_name],
            "abs_diff": abs(value - reference[suite_name]),
        }
        lookup_parts.append(group[["state_id", "lookup_loss"]])
    lookup = pd.concat(lookup_parts, ignore_index=True)
    audit["lookup_check"] = lookup_check

    if not audit["m2_oof_reproduced"] or any(v["abs_diff"] > 1e-9 for v in lookup_check.values()):
        (T13_1_LIGHT / "15a_input_audit.json").write_text(json.dumps(audit, indent=2))
        raise SystemExit("[task13.1A/B] reproduction FAILED -> STOP")

    # ------------------------------------------------------------------
    # strong-baseline ladder rows
    # ------------------------------------------------------------------
    m2 = saved_m2.rename(columns={"selected": "selected_m2", "realized_loss": "realized_m2"})
    m0 = (
        preds.loc[preds["model"].eq("M0_current_FE") & preds["carrier"].eq("rf")][["state_id", "realized_loss"]]
        .rename(columns={"realized_loss": "realized_m0"})
    )
    m1 = (
        preds.loc[preds["model"].eq("M1_behavior") & preds["carrier"].eq("rf")][
            ["state_id", "pred_shade", "pred_lshade", "pred_cso", "selected", "realized_loss"]
        ]
        .rename(columns={"selected": "selected_m1", "realized_loss": "realized_m1"})
    )
    base = dataset[
        ["state_id", "suite", "problem_id", "cv_group_id", "seed", "current_algorithm", "FE",
         "family", "continue_loss", "switch_required", "A_ND_members"]
    ]
    ladder = (
        base.merge(lookup, on="state_id", validate="many_to_one")
        .merge(m0, on="state_id", validate="many_to_one")
        .merge(m1, on="state_id", validate="many_to_one")
        .merge(m2, on="state_id", validate="many_to_one")
        .merge(delta_ctx, on="state_id", validate="many_to_one")
    )
    ladder.to_parquet(T13_1_HEAVY / "strong_baseline_policy_rows.parquet", index=False)

    # ------------------------------------------------------------------
    # C: paired comparisons (5000 draws)
    # ------------------------------------------------------------------
    comparisons = {
        "C1_continue_minus_M2": ("continue_loss", "realized_m2"),
        "C2_lookup_minus_M2": ("lookup_loss", "realized_m2"),
        "C3_M0_minus_M2": ("realized_m0", "realized_m2"),
        "C4_lookup_minus_M1": ("lookup_loss", "realized_m1"),
    }
    comp_rows = []
    stream = 0
    for name, (upper_col, lower_col) in comparisons.items():
        for suite_name, group in ladder.groupby("suite", sort=False):
            point, low, high = fb_of_difference(
                group, group[upper_col].to_numpy(dtype=float), group[lower_col].to_numpy(dtype=float)
            )
            comp_rows.append({"comparison": name, "suite": suite_name, "fb_mean": point, "ci_low": low, "ci_high": high})
            stream += 1
    comparisons_table = pd.DataFrame(comp_rows)
    comparisons_table.to_parquet(T13_1_LIGHT / "strong_baseline_pairwise_bootstrap.parquet", index=False)

    # ------------------------------------------------------------------
    # E: pooled action-specific noise scale from training-domain repetitions
    # ------------------------------------------------------------------
    branches = load_shard("branches")
    state_meta = states.set_index("state_id")[["cv_group_id"]]
    branches = branches.merge(state_meta, left_on="state_id", right_index=True, validate="many_to_one")
    branches["solver_cell"] = np.where(
        branches["candidate_action"].eq("continue"), branches["current_algorithm"], branches["candidate_action"]
    )
    medians = branches.groupby(["state_id", "candidate_action"])["loss_1000"].transform("median")
    branches["deviation"] = (branches["loss_1000"] - medians).abs()
    scale_rows = []
    for solver_name, group in branches.groupby("solver_cell", sort=False):
        per_group = group.groupby("cv_group_id")["deviation"].quantile(0.95)
        scale_rows.append(
            {
                "solver": solver_name,
                "delta_pool_95": float(per_group.mean()),
                "n_groups": int(per_group.size),
                "n_repeated_cells": int(group.loc[group["replicate_id"].gt(0)].groupby("state_id").ngroups),
            }
        )
    pooled_scale = pd.DataFrame(scale_rows)
    pooled_scale.to_parquet(T13_1_LIGHT / "pooled_action_noise_scale.parquet", index=False)
    delta_pool = pooled_scale.set_index("solver")["delta_pool_95"].to_dict()

    # ------------------------------------------------------------------
    # D: predicted margins for M2 (primary) and M1 (diagnostic)
    # ------------------------------------------------------------------
    dataset_indexed = dataset.set_index("state_id")

    def margins(pred_frame: pd.DataFrame) -> pd.DataFrame:
        records = []
        for state_id, row in pred_frame.set_index("state_id").iterrows():
            current = dataset_indexed.at[state_id, "current_algorithm"]
            alternatives = [s for s in SOLVERS if s != current]
            alt = min(alternatives, key=lambda s: row[f"pred_{s}"])
            records.append(
                {
                    "state_id": state_id,
                    "current_algorithm": current,
                    "pred_current": float(row[f"pred_{current}"]),
                    "pred_alt": float(row[f"pred_{alt}"]),
                    "pred_alt_algorithm": alt,
                    "pred_margin": float(row[f"pred_{current}"] - row[f"pred_{alt}"]),
                }
            )
        frame = pd.DataFrame(records)
        frame["pair_scale_max"] = [
            max(delta_pool[c], delta_pool[a]) for c, a in zip(frame["current_algorithm"], frame["pred_alt_algorithm"])
        ]
        frame["pair_scale_sum"] = [
            delta_pool[c] + delta_pool[a] for c, a in zip(frame["current_algorithm"], frame["pred_alt_algorithm"])
        ]
        return frame

    margins_m2 = margins(saved_m2)
    margins_m1 = margins(m1)

    # ------------------------------------------------------------------
    # F/H: fixed kappa grid policies + full metric set
    # ------------------------------------------------------------------
    delta_state = delta_ctx.set_index("state_id")
    lookup_map = lookup.set_index("state_id")["lookup_loss"]

    def build_policy_frame(margin_frame: pd.DataFrame, tag: str) -> pd.DataFrame:
        frame = margin_frame.merge(
            dataset[
                ["state_id", "suite", "problem_id", "cv_group_id", "seed", "FE", "switch_required", "A_ND_members", "continue_loss"]
            ],
            on="state_id",
            validate="many_to_one",
        )
        solver_loss = dataset.set_index("state_id")[LOSS_COLS]
        for scale in SCALES:
            for kappa in KAPPA_GRID:
                column = f"{tag}_k{int(kappa * 10):02d}_{scale}"
                selected_list, realized_list = [], []
                for row in frame.itertuples(index=False):
                    if row.pred_margin > kappa * getattr(row, f"pair_scale_{scale}"):
                        action = row.pred_alt_algorithm
                        loss = float(solver_loss.at[row.state_id, f"loss_{action}"])
                    else:
                        action = row.current_algorithm
                        loss = float(row.continue_loss)
                    selected_list.append(action)
                    realized_list.append(loss)
                frame[f"selected_{column}"] = selected_list
                realized_col = f"realized_{column}"
                frame[realized_col] = realized_list
                pair_delta = np.array(
                    [
                        0.0
                        if c == a
                        else float(
                            max(delta_state.at[s, f"delta_cell_{c}"], delta_state.at[s, f"delta_cell_{a}"])
                        )
                        for s, c, a in zip(frame["state_id"], frame["current_algorithm"], frame[f"selected_{column}"])
                    ]
                )
                frame[f"pair_state_delta_{column}"] = pair_delta
                realized_array = frame[realized_col].to_numpy(dtype=float)
                frame[f"harmful_{column}"] = realized_array > (frame["continue_loss"].to_numpy() + pair_delta)
                frame[f"harmful_mass_{column}"] = np.clip(
                    realized_array - frame["continue_loss"].to_numpy() - pair_delta, 0.0, None
                )
        return frame

    margins_m2_full = build_policy_frame(margins_m2, "m2")
    margins_m1_full = build_policy_frame(margins_m1, "m1")
    margins_m2_full.to_parquet(T13_1_HEAVY / "margin_policy_rows.parquet", index=False)
    margins_m1_full.to_parquet(T13_1_HEAVY / "margin_policy_rows_m1.parquet", index=False)

    def summarize(frame: pd.DataFrame, tag: str) -> list[dict]:
        rows = []
        for scale in SCALES:
            for kappa in KAPPA_GRID:
                column = f"{tag}_k{int(kappa * 10):02d}_{scale}"
                selected_col = f"selected_{column}"
                realized_col = f"realized_{column}"
                for suite_name, group in frame.groupby("suite", sort=False):
                    is_switch = group[selected_col].ne(group["current_algorithm"])
                    gain = group["continue_loss"] - group[realized_col]
                    required = group["switch_required"].astype(bool)
                    switched_required = group.loc[required & is_switch]
                    lookup_vals = lookup_map.reindex(group["state_id"]).to_numpy()
                    rows.append(
                        {
                            "policy": f"{tag}|{scale}|kappa={kappa}",
                            "tag": tag,
                            "scale": scale,
                            "kappa": kappa,
                            "suite": suite_name,
                            "realized_fb_loss": fb_mean(group, realized_col),
                            "gain_vs_continue": float(
                                (group["continue_loss"] - group[realized_col])
                                .groupby(group["cv_group_id"]).mean().mean()
                            ),
                            "gain_vs_lookup": float(
                                pd.Series(lookup_vals - group[realized_col].to_numpy())
                                .groupby(group["cv_group_id"].to_numpy())
                                .mean()
                                .mean()
                            ),
                            "switch_rate": float(is_switch.mean()),
                            "harmful_rate": float(group[f"harmful_{column}"].mean()),
                            "harmful_mass": float(group[f"harmful_mass_{column}"].mean()),
                            "switch_precision": float(group.loc[is_switch, "switch_required"].mean()) if is_switch.any() else np.nan,
                            "switch_recall": float(len(switched_required) / max(int(required.sum()), 1)),
                            "unnecessary_switch_rate": float((~group.loc[is_switch, "switch_required"]).mean()) if is_switch.any() else np.nan,
                            "gain_per_executed_switch": float(gain[is_switch].mean()) if is_switch.any() else np.nan,
                        }
                    )
        return rows

    summary_rows = summarize(margins_m2_full, "m2") + summarize(margins_m1_full, "m1")
    policy_summary_table = pd.DataFrame(summary_rows)
    policy_summary_table.to_parquet(T13_1_LIGHT / "margin_policy_summary.parquet", index=False)

    # per-policy paired bootstrap vs continue and vs lookup (m2 primary)
    boot_rows = []
    stream = 200
    for scale in SCALES:
        for kappa in KAPPA_GRID:
            column = f"m2_k{int(kappa * 10):02d}_{scale}"
            realized_col = f"realized_{column}"
            for baseline_name in ("continue", "lookup"):
                for suite_name, group in margins_m2_full.groupby("suite", sort=False):
                    if baseline_name == "continue":
                        base_values = group["continue_loss"].to_numpy(dtype=float)
                    else:
                        base_values = lookup_map.reindex(group["state_id"]).to_numpy(dtype=float)
                    point, low, high = fb_of_difference(group, base_values, group[realized_col].to_numpy(dtype=float))
                    boot_rows.append(
                        {
                            "policy": f"m2|{scale}|kappa={kappa}",
                            "baseline": baseline_name,
                            "suite": suite_name,
                            "fb_gain": point,
                            "ci_low": low,
                            "ci_high": high,
                        }
                    )
                    stream += 1
    pd.DataFrame(boot_rows).to_parquet(T13_1_LIGHT / "margin_policy_bootstrap.parquet", index=False)

    # ------------------------------------------------------------------
    # G: margin calibration (pooled deciles + strata)
    # ------------------------------------------------------------------
    calib = margins_m2_full.merge(dataset[["state_id", "FE"]], on="state_id", validate="many_to_one")
    solver_loss_indexed = dataset.set_index("state_id")[LOSS_COLS]
    calib["realized_switch_gain"] = [
        float(continue_loss_value - solver_loss_indexed.at[state_id, f"loss_{alt}"])
        for state_id, alt, continue_loss_value in zip(
            calib["state_id"], calib["pred_alt_algorithm"], calib["continue_loss"]
        )
    ]
    calib["margin_bin"] = pd.qcut(calib["pred_margin"], 10, labels=False, duplicates="drop")
    calib_rows = []
    for stratum_name, part in (
        ("pooled", calib),
        ("bbob", calib.loc[calib["suite"].eq("bbob")]),
        ("mabbob", calib.loc[calib["suite"].eq("mabbob")]),
    ):
        for bin_index, bin_group in part.groupby("margin_bin", observed=True):
            calib_rows.append(
                {
                    "stratum": stratum_name,
                    "bin": int(bin_index),
                    "n": int(len(bin_group)),
                    "mean_pred_margin": float(bin_group["pred_margin"].mean()),
                    "mean_realized_gain": float(bin_group["realized_switch_gain"].mean()),
                    "median_realized_gain": float(bin_group["realized_switch_gain"].median()),
                    "harmful_probability": float((bin_group["realized_switch_gain"] < -bin_group["pair_scale_max"]).mean()),
                    "switch_required_probability": float(bin_group["switch_required"].mean()),
                }
            )
    calibration = pd.DataFrame(calib_rows)
    calibration.to_parquet(T13_1_LIGHT / "margin_calibration_bins.parquet", index=False)

    # ------------------------------------------------------------------
    # I: risk-performance table + Pareto flag (harmful rate vs fb loss)
    # ------------------------------------------------------------------
    policy_cols = policy_summary_table[
        ["policy", "tag", "scale", "kappa", "suite", "switch_rate", "harmful_rate", "realized_fb_loss", "gain_vs_continue", "gain_vs_lookup"]
    ]
    baseline_rows = []
    for suite_name, group in ladder.groupby("suite", sort=False):
        for name, column in (
            ("continue", "continue_loss"),
            ("lookup", "lookup_loss"),
            ("rf_m0", "realized_m0"),
            ("rf_m1", "realized_m1"),
            ("rf_m2", "realized_m2"),
        ):
            switch_rate = np.nan
            if name in ("rf_m1", "rf_m2"):
                selected_col = "selected_m1" if name == "rf_m1" else "selected_m2"
                switch_rate = float(group[selected_col].ne(group["current_algorithm"]).mean())
            baseline_rows.append(
                {
                    "policy": name,
                    "tag": "baseline",
                    "scale": "",
                    "kappa": np.nan,
                    "suite": suite_name,
                    "switch_rate": switch_rate,
                    "harmful_rate": np.nan,
                    "realized_fb_loss": fb_mean(group, column),
                    "gain_vs_continue": np.nan,
                    "gain_vs_lookup": np.nan,
                }
            )
    pareto_frame = pd.concat([policy_cols, pd.DataFrame(baseline_rows)], ignore_index=True)
    pareto_frame["pareto_on_harmful_vs_loss"] = False
    for suite_name, group in pareto_frame.groupby("suite", sort=False):
        sub = group.loc[group["harmful_rate"].notna()]
        values = sub[["harmful_rate", "realized_fb_loss"]].to_numpy(dtype=float)
        optimal = np.array(
            [
                not np.any(
                    (values[:, 0] <= values[i, 0])
                    & (values[:, 1] <= values[i, 1])
                    & ((values[:, 0] < values[i, 0]) | (values[:, 1] < values[i, 1]))
                )
                for i in range(len(sub))
            ]
        )
        pareto_frame.loc[sub.index, "pareto_on_harmful_vs_loss"] = optimal
    pareto_frame.to_parquet(T13_1_LIGHT / "risk_performance_pareto.parquet", index=False)

    # ------------------------------------------------------------------
    # N: transition risk stratification (raw M2 = kappa 0)
    # ------------------------------------------------------------------
    transition_rows = []
    raw_selected = margins_m2_full["selected_m2_k00_max"]
    switched = margins_m2_full.loc[raw_selected.ne(margins_m2_full["current_algorithm"])]
    for (suite_name, current, target), group in switched.groupby(
        ["suite", "current_algorithm", "selected_m2_k00_max"], sort=False
    ):
        gains = group["continue_loss"] - group["realized_m2_k00_max"]
        transition_rows.append(
            {
                "suite": suite_name,
                "transition": f"{current} -> {target}",
                "count": int(len(group)),
                "mean_realized_gain": float(gains.mean()),
                "harmful_rate": float(group["harmful_m2_k00_max"].mean()),
                "switch_required_precision": float(group["switch_required"].mean()),
            }
        )
    transitions = pd.DataFrame(transition_rows)
    transitions.to_parquet(T13_1_LIGHT / "transition_risk_stratification.parquet", index=False)

    audit["delta_pool"] = delta_pool
    (T13_1_LIGHT / "15a_input_audit.json").write_text(json.dumps(audit, indent=2))

    elapsed = perf_counter() - started
    pd.DataFrame(
        [
            {
                "phase": "task13_1_analysis",
                "new_objective_fe": 0,
                "wall_seconds": elapsed,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                "note": "zero-FE reuse of Task 12/13 artifacts; permutation study runs separately",
            }
        ]
    ).to_parquet(T13_1_HEAVY / "task13_1_resource_ledger.parquet", index=False)

    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print("=== A1 audit ===")
        print(json.dumps(audit, indent=1, default=float))
        print("=== C comparisons ===")
        print(comparisons_table.round(4).to_string())
        print("=== pooled noise scale ===")
        print(pooled_scale.round(4).to_string())
        print("=== policy summary (m2) ===")
        print(policy_summary_table.loc[policy_summary_table["tag"].eq("m2")].round(4).to_string())
        print("=== policy summary (m1) ===")
        print(policy_summary_table.loc[policy_summary_table["tag"].eq("m1")].round(4).to_string())
        print("=== calibration (pooled) ===")
        print(calibration.loc[calibration["stratum"].eq("pooled")].round(4).to_string())
        print("=== transitions ===")
        print(transitions.round(4).to_string())


if __name__ == "__main__":
    main()
