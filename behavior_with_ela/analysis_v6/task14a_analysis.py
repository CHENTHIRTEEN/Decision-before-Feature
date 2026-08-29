"""Task 14A analysis: post-handoff noise calibration, practical action space,
oracle ladder, absorbing-state audit, natural-vs-post comparison, reset
confound effects and the frozen margin-policy confirmation diagnostic.

Zero new objective FE beyond the committed collection; all statistics reuse
the collected parquets. The margin diagnostic applies the Task 13 frozen
carrier (same pipeline/parameters) fitted once on the full development set
to the post-handoff states with the Task 13.1 pooled deployment scales; it
is a confirmation diagnostic only and cannot re-select kappa.
"""
from __future__ import annotations

import json
import resource
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS
from behavior_with_ela.analysis_v5.task12_1_analysis import (
    SOLVERS,
    fb_mean,
    fb_series,
)
from behavior_with_ela.analysis_v5.task13.task13_analysis import (
    make_carrier,
)

ROOT = Path(__file__).resolve().parents[2]
T14A_HEAVY = ROOT / "behavior_with_ela/results/analysis_v6/task14a"
T14A_LIGHT = ROOT / "behavior_with_ela/analysis_v6/task14a"
T13_HEAVY = ROOT / "behavior_with_ela/results/analysis_v5/task13"
T13_1_LIGHT = ROOT / "behavior_with_ela/analysis_v5/task13_1"
BOOTSTRAP_STREAM = 2026090223
BOOTSTRAP_DRAWS = 5000
LOSS_COLS = [f"loss_{s}" for s in SOLVERS]


def fb_diff_bootstrap(
    frame: pd.DataFrame, upper: np.ndarray, lower: np.ndarray, stream_offset: int
) -> tuple[float, float, float]:
    work = pd.DataFrame({"cv_group_id": frame["cv_group_id"].to_numpy(), "d": upper - lower})
    groups = sorted(work["cv_group_id"].unique())
    means = work.groupby("cv_group_id")["d"].mean()
    rng = np.random.default_rng(
        np.random.SeedSequence([BOOTSTRAP_STREAM + stream_offset, len(groups)]).generate_state(4)
    )
    draws = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        sample = rng.choice(groups, size=len(groups), replace=True)
        draws[draw] = np.mean([means[g] for g in sample])
    return float(means.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    T14A_LIGHT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    states = pd.read_parquet(T14A_HEAVY / "post_handoff_states.parquet")
    branches = pd.read_parquet(T14A_HEAVY / "post_handoff_action_outcomes_1000.parquet")
    resets = pd.read_parquet(T14A_HEAVY / "reset_control_outcomes.parquet")
    behavior = pd.read_parquet(T14A_HEAVY / "post_handoff_behavior.parquet")
    bf_cols = [c for c in behavior.columns if c in set(SELECTOR_BEHAVIOR_FEATURE_COLUMNS)]
    behavior_global = behavior.loc[behavior["behavior_kind"].eq("global")][
        ["state_id", *bf_cols]
    ]

    base = branches.loc[branches["replicate_id"].eq(0)]
    cont = base.loc[base["candidate_action"].eq("continue")].set_index("state_id")["loss_1000"]
    switch = (
        base.loc[base["candidate_action"].ne("continue")]
        .pivot_table(index="state_id", columns="candidate_action", values="loss_1000", aggfunc="first")
    )
    frame = states.merge(
        behavior_global, on="state_id", validate="many_to_one"
    )
    frame["continue_loss"] = cont.reindex(frame["state_id"]).to_numpy()
    for s in SOLVERS:
        frame[f"switch_{s}"] = switch[s].reindex(frame["state_id"]).to_numpy()
    for s in SOLVERS:
        frame[f"loss_{s}"] = np.where(
            frame["current_algorithm"].eq(s),
            frame["continue_loss"],
            frame[f"switch_{s}"],
        )
    frame["FE_ratio"] = frame["FE"] / 10000.0
    if frame[LOSS_COLS].isna().any().any():
        raise SystemExit("[task14a] missing solver-matrix cells")

    # ---- post-handoff practical noise (10% x R=3, outcome-blind) ----
    meta = states.set_index("state_id")[["cv_group_id", "suite"]]
    medians = branches.groupby(["state_id", "candidate_action"])["loss_1000"].transform("median")
    branches = branches.assign(deviation=(branches["loss_1000"] - medians).abs())
    branches_meta = branches.merge(meta, left_on="state_id", right_index=True, validate="many_to_one")
    branches_meta["solver_cell"] = np.where(
        branches_meta["candidate_action"].eq("continue"),
        branches_meta["current_algorithm"],
        branches_meta["candidate_action"],
    )
    noise_rows = []
    for (solver_cell,), group in branches_meta.groupby(["solver_cell"], sort=False):
        per_group = group.groupby("cv_group_id")["deviation"].quantile([0.50, 0.95]).unstack()
        noise_rows.append(
            {
                "solver": solver_cell,
                "stratum": "pooled",
                "delta_50": float(per_group[0.50].mean()),
                "delta_95": float(per_group[0.95].mean()),
                "repeated_cells": int(
                    group.loc[group["replicate_id"].gt(0)].groupby("state_id").ngroups
                ),
            }
        )
    for (solver_cell, route), group in branches_meta.groupby(["solver_cell", "route"], sort=False):
        per_group = group.groupby("cv_group_id")["deviation"].quantile(0.95)
        noise_rows.append(
            {
                "solver": solver_cell,
                "stratum": f"route:{route}",
                "delta_50": np.nan,
                "delta_95": float(per_group.mean()),
                "repeated_cells": int(
                    group.loc[group["replicate_id"].gt(0)].groupby("state_id").ngroups
                ),
            }
        )
    noise = pd.DataFrame(noise_rows)
    noise.to_parquet(T14A_LIGHT / "post_handoff_noise_deltas.parquet", index=False)
    delta_by_solver = (
        noise.loc[noise["stratum"].eq("pooled")].set_index("solver")["delta_95"].to_dict()
    )

    # ---- practical action sets (max primary, quad/sum sensitivity) ----
    delta_ctx = pd.DataFrame(index=frame.index)
    for s in SOLVERS:
        # the solver-cell noise merges the solver's continue and switch cells
        # (same semantics as the Task 13.1 pooled deployment scale)
        delta_ctx[f"delta_cell_{s}"] = float(delta_by_solver[s])
    values = frame[LOSS_COLS].to_numpy(dtype=float)
    n = len(frame)
    index_by_solver = {s: i for i, s in enumerate(SOLVERS)}
    current_index = frame["current_algorithm"].map(index_by_solver).to_numpy()
    results = {}
    set_rows = []
    dcm_rows = []
    for rule in ("max", "quadrature", "sum"):
        delta_matrix = np.zeros((n, 3, 3))
        for i, a in enumerate(SOLVERS):
            for j, b in enumerate(SOLVERS):
                if i == j:
                    continue
                da = delta_ctx[f"delta_cell_{a}"].to_numpy()
                db = delta_ctx[f"delta_cell_{b}"].to_numpy()
                if rule == "max":
                    delta_matrix[:, i, j] = np.maximum(da, db)
                elif rule == "quadrature":
                    delta_matrix[:, i, j] = np.sqrt(da**2 + db**2)
                else:
                    delta_matrix[:, i, j] = da + db
        dominates = values[:, :, None] < values[:, None, :] - delta_matrix
        non_dominated = ~dominates.any(axis=1)
        in_nd = non_dominated[np.arange(n), current_index]
        results[rule] = {"non_dominated": non_dominated, "delta": delta_matrix}
        for suite_name, mask in (
            ("pooled", np.ones(n, dtype=bool)),
            ("bbob", frame["suite"].eq("bbob").to_numpy()),
            ("mabbob", frame["suite"].eq("mabbob").to_numpy()),
        ):
            sub_nd = non_dominated[mask]
            sub_in = in_nd[mask]
            set_rows.append(
                {
                    "rule": rule,
                    "suite": suite_name,
                    "switch_required_rate": float((~sub_in).mean()),
                    "P_current_in_A_ND": float(sub_in.mean()),
                    "P_unique_winner": float((sub_nd.sum(axis=1) == 1).mean()),
                    "P_A_ND_empty": float((sub_nd.sum(axis=1) == 0).mean()),
                    "E_A_ND_size": float(sub_nd.sum(axis=1).mean()),
                }
            )
        if rule in ("max", "sum"):
            for a, b in combinations(SOLVERS, 2):
                d = delta_matrix[:, index_by_solver[a], index_by_solver[b]]
                la, lb = values[:, index_by_solver[a]], values[:, index_by_solver[b]]
                tie = float((np.abs(la - lb) <= d).mean())
                a_better = float((la < lb - d).mean())
                b_better = float((lb < la - d).mean())
                dcm_rows.append(
                    {
                        "rule": rule,
                        "candidate_a": a,
                        "candidate_b": b,
                        "P_tie": tie,
                        "P_a_better": a_better,
                        "P_b_better": b_better,
                        "DCM": float((tie + abs(a_better - b_better)) / 2.0),
                    }
                )
    practical_sets = pd.DataFrame(set_rows)
    practical_sets.to_parquet(T14A_LIGHT / "post_handoff_practical_action_sets.parquet", index=False)
    dcm = pd.DataFrame(dcm_rows)
    dcm.to_parquet(T14A_LIGHT / "post_handoff_dcm.parquet", index=False)

    primary = "max"
    non_dominated = results[primary]["non_dominated"]
    frame["A_ND_size"] = non_dominated.sum(axis=1)
    frame["current_in_A_ND"] = in_nd
    frame["switch_required"] = ~in_nd

    # ---- oracle ladder (P0-P4) ----
    assignment_store = {}

    def context_policy(keys: list[str]) -> pd.Series:
        cache_key = "|".join(keys)
        if cache_key not in assignment_store:
            loss_cols = LOSS_COLS
            per = (
                frame.groupby([*keys, "cv_group_id"], sort=False)[loss_cols]
                .mean()
                .groupby(keys, sort=False)
                .mean()
            )
            choice = per.idxmin(axis=1).str.replace("loss_", "", regex=False)
            assignment_store[cache_key] = pd.Series(
                np.asarray(frame.set_index(keys).index.map(choice)), index=frame.index
            )
        return assignment_store[cache_key]

    frame["P0_continue"] = frame["continue_loss"]
    for name, keys in (
        ("P1_current", ["current_algorithm"]),
        ("P2_route", ["source_algorithm", "current_algorithm"]),
        ("P3_route_source_FE", ["source_algorithm", "current_algorithm", "source_checkpoint_fe"]),
    ):
        assignment = context_policy(keys)
        frame[name] = [row[f"loss_{a}"] for (_, row), a in zip(frame.iterrows(), assignment)]
    frame["P4_statewise"] = frame[LOSS_COLS].min(axis=1)
    ladder_columns = ["P0_continue", "P1_current", "P2_route", "P3_route_source_FE", "P4_statewise"]

    ladder_rows = []
    stream = 0
    for suite_name, group in (
        ("pooled", frame),
        ("bbob", frame.loc[frame["suite"].eq("bbob")]),
        ("mabbob", frame.loc[frame["suite"].eq("mabbob")]),
    ):
        row = {"suite": suite_name}
        for column in ladder_columns:
            row[f"L_{column}"] = fb_mean(group, column)
        groups = sorted(group["cv_group_id"].unique())
        group_means = {c: fb_series(group, c).to_dict() for c in ladder_columns}
        rng = np.random.default_rng(
            np.random.SeedSequence([BOOTSTRAP_STREAM + stream, len(groups)]).generate_state(4)
        )
        stream += 1
        draws = {c: np.empty(5000) for c in ladder_columns}
        for draw in range(5000):
            sample = rng.choice(groups, size=len(groups), replace=True)
            for c in ladder_columns:
                draws[c][draw] = np.mean([group_means[c][g] for g in sample])
        for c in ladder_columns:
            row[f"{c}_ci_low"] = float(np.quantile(draws[c], 0.025))
            row[f"{c}_ci_high"] = float(np.quantile(draws[c], 0.975))
        row["delta_post_routeFE_statewise"] = row["L_P3_route_source_FE"] - row["L_P4_statewise"]
        row["delta_routeFE_minus_route"] = row["L_P3_route_source_FE"] - row["L_P2_route"]
        ladder_rows.append(row)
    ladder = pd.DataFrame(ladder_rows)
    ladder.to_parquet(T14A_LIGHT / "post_handoff_oracle_headroom.parquet", index=False)

    # per-route/per-current breakdown of the switch-required rate
    strata_rows = []
    for stratum, keys in (
        ("route", ["source_algorithm", "current_algorithm"]),
        ("current", ["current_algorithm"]),
        ("source", ["source_algorithm"]),
        ("source_FE", ["source_checkpoint_fe"]),
    ):
        for key, part in frame.groupby(keys, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            strata_rows.append(
                {
                    "stratum": stratum,
                    "key": "|".join(str(k) for k in key),
                    "n": len(part),
                    "switch_required_rate": float(part["switch_required"].mean()),
                    "P_current_in_A_ND": float(part["current_in_A_ND"].mean()),
                    "L_statewise": fb_mean(part, "P4_statewise"),
                    "L_P3": fb_mean(part, "P3_route_source_FE"),
                }
            )
    strata = pd.DataFrame(strata_rows)
    strata.to_parquet(T14A_HEAVY / "post_handoff_strata.parquet", index=False)

    # ---- absorbing-state audit ----
    absorbing_rows = []
    for current, part in frame.groupby("current_algorithm", sort=False):
        absorbing_rows.append(
            {
                "current": current,
                "P_current_in_A_ND": float(part["current_in_A_ND"].mean()),
                "switch_required_rate": float(part["switch_required"].mean()),
                "L_P3": fb_mean(part, "P3_route_source_FE"),
                "L_P4_statewise": fb_mean(part, "P4_statewise"),
                "residual": fb_mean(part, "P3_route_source_FE") - fb_mean(part, "P4_statewise"),
            }
        )
    absorbing = pd.DataFrame(absorbing_rows)
    absorbing["absorbing_state_risk_flag"] = (
        absorbing["P_current_in_A_ND"] > 0.90
    ) & (absorbing["residual"].abs() < 0.05)
    absorbing.to_parquet(T14A_LIGHT / "post_handoff_absorbing_audit.parquet", index=False)

    # ---- natural vs post-handoff comparison ----
    natural_rows = []
    natural_reference = {
        "switch_required_rate": {"bbob": 0.2578, "mabbob": 0.2648},
        "DCM_range": (0.254, 0.370),
        "delta_dynamic_old": {"bbob": 0.1137, "mabbob": 0.0854},
    }
    for suite_name in ("bbob", "mabbob"):
        post = practical_sets.loc[
            practical_sets["suite"].eq(suite_name) & practical_sets["rule"].eq("max")
        ].iloc[0]
        natural_rows.append(
            {
                "suite": suite_name,
                "natural_switch_required_rate": natural_reference["switch_required_rate"][suite_name],
                "post_switch_required_rate": float(post["switch_required_rate"]),
                "natural_DCM_min": natural_reference["DCM_range"][0],
                "natural_DCM_max": natural_reference["DCM_range"][1],
                "post_DCM_min": float(
                    dcm.loc[dcm["rule"].eq("max"), "DCM"].min()
                ),
                "post_DCM_max": float(
                    dcm.loc[dcm["rule"].eq("max"), "DCM"].max()
                ),
                "natural_delta_dynamic_old": natural_reference["delta_dynamic_old"][suite_name],
                "post_delta_routeFE_statewise": float(
                    ladder.loc[ladder["suite"].eq(suite_name), "delta_post_routeFE_statewise"].iloc[0]
                ),
            }
        )
    natural_vs_post = pd.DataFrame(natural_rows)
    natural_vs_post.to_parquet(T14A_LIGHT / "natural_vs_post_handoff_metrics.parquet", index=False)

    # ---- reset confound effects ----
    cont_map = base.loc[base["candidate_action"].eq("continue")].set_index("state_id")["loss_1000"]
    reset_effect_rows = []
    stream = 50
    for current, switch_target in (("shade", "lshade"), ("lshade", "shade")):
        part_states = frame.loc[frame["current_algorithm"].eq(current)]
        part_cont = cont_map.reindex(part_states["state_id"]).to_numpy(dtype=float)
        part_switch = (
            base.loc[
                base["candidate_action"].eq(switch_target) & base["replicate_id"].eq(0)
            ]
            .set_index("state_id")
            .loc[part_states["state_id"], "loss_1000"]
            .to_numpy(dtype=float)
        )
        part_reset = (
            resets.loc[resets["current_algorithm"].eq(current)]
            .set_index("state_id")
            .loc[part_states["state_id"], "loss_1000"]
            .to_numpy(dtype=float)
        )
        for suite_name, mask in (
            ("pooled", np.ones(len(part_states), dtype=bool)),
            ("bbob", part_states["suite"].eq("bbob").to_numpy()),
            ("mabbob", part_states["suite"].eq("mabbob").to_numpy()),
        ):
            work = part_states.loc[mask]
            g_reset, g_reset_low, g_reset_high = fb_diff_bootstrap(
                work, part_cont[mask], part_reset[mask], stream
            )
            g_switch, g_switch_low, g_switch_high = fb_diff_bootstrap(
                work, part_cont[mask], part_switch[mask], stream + 1
            )
            delta_point, delta_low, delta_high = fb_diff_bootstrap(
                work, part_reset[mask], part_switch[mask], stream + 2
            )
            reset_effect_rows.append(
                {
                    "current": current,
                    "switch_target": switch_target,
                    "suite": suite_name,
                    "G_reset_native_minus_reset": g_reset,
                    "G_reset_ci_low": g_reset_low,
                    "G_reset_ci_high": g_reset_high,
                    "G_switch_native_minus_switch": g_switch,
                    "G_switch_ci_low": g_switch_low,
                    "G_switch_ci_high": g_switch_high,
                    "delta_solver_specific_reset_minus_switch": delta_point,
                    "delta_ci_low": delta_low,
                    "delta_ci_high": delta_high,
                    "delta_exceeds_post_noise": bool(abs(delta_point) > 0.05),
                }
            )
            stream += 3
    reset_effects = pd.DataFrame(reset_effect_rows)
    reset_effects.to_parquet(T14A_LIGHT / "reset_effect_summary.parquet", index=False)

    # ---- frozen margin-policy confirmation diagnostic (16h) ----
    dev = pd.read_parquet(T13_HEAVY / "behavior_action_dataset_task13.parquet")
    dummies = pd.get_dummies(dev["current_algorithm"], prefix="cur", dtype=float)
    dev = pd.concat([dev.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    features = [*dummies.columns, "FE_ratio", *bf_cols]
    model = make_carrier("rf")
    model.fit(dev[features].to_numpy(dtype=float), dev[LOSS_COLS].to_numpy(dtype=float))

    post_dummies = pd.get_dummies(frame["current_algorithm"], prefix="cur", dtype=float)
    for column in dummies.columns:
        if column not in post_dummies:
            post_dummies[column] = 0.0
    post_dummies = post_dummies[dummies.columns]
    X_post = pd.concat([post_dummies.reset_index(drop=True), frame[["FE_ratio", *bf_cols]].reset_index(drop=True)], axis=1)
    pred = model.predict(X_post.to_numpy(dtype=float))
    frame["pred_shade"], frame["pred_lshade"], frame["pred_cso"] = pred[:, 0], pred[:, 1], pred[:, 2]
    pred_current = pred[np.arange(n), current_index]
    pred_alt_index = np.array(
        [min((i for i in range(3) if i != ci), key=lambda i: pred[r, i]) for r, ci in enumerate(current_index)]
    )
    alt_solver = np.array([SOLVERS[i] for i in pred_alt_index])
    margin = pred_current - pred[np.arange(n), pred_alt_index]
    frame["pred_margin"] = margin
    frame["pred_alt_algorithm"] = alt_solver

    pooled_delta = (
        pd.read_parquet(T13_1_LIGHT / "pooled_action_noise_scale.parquet")
        .set_index("solver")["delta_pool_95"]
        .to_dict()
    )
    frame["pair_scale_max"] = [
        max(pooled_delta[c], pooled_delta[a]) for c, a in zip(frame["current_algorithm"], frame["pred_alt_algorithm"])
    ]
    diag_rows = []
    realized_switch = np.array(
        [frame.at[r, f"switch_{s}"] for r, s in zip(frame.index, alt_solver)]
    )
    cont_array = frame["continue_loss"].to_numpy(dtype=float)
    groups_array = frame["cv_group_id"].to_numpy()
    suites_array = frame["suite"].to_numpy()
    for kappa in (0.0, 0.5, 1.0):
        do_switch = margin > kappa * frame["pair_scale_max"].to_numpy()
        realized = np.where(do_switch, realized_switch, cont_array)
        selected_solver = np.where(do_switch, alt_solver, frame["current_algorithm"].to_numpy())
        state_delta = np.array(
            [
                0.0
                if c == a
                else float(
                    max(delta_ctx.at[r, f"delta_cell_{c}"], delta_ctx.at[r, f"delta_cell_{a}"])
                )
                for r, c, a in zip(frame.index, frame["current_algorithm"], selected_solver)
            ]
        )
        d_gain = cont_array - realized
        harmful = realized > cont_array + state_delta
        for suite_name, mask in (
            ("pooled", np.ones(n, dtype=bool)),
            ("bbob", suites_array == "bbob"),
            ("mabbob", suites_array == "mabbob"),
        ):
            diag_rows.append(
                {
                    "kappa": kappa,
                    "suite": suite_name,
                    "switch_rate": float(do_switch[mask].mean()),
                    "realized_fb_loss": float(
                        pd.Series(-d_gain[mask]).groupby(groups_array[mask]).mean().mean()
                    ),
                    "gain_vs_continue": float(
                        pd.Series(d_gain[mask]).groupby(groups_array[mask]).mean().mean()
                    ),
                    "harmful_rate": float(harmful[mask].mean()),
                }
            )
    margin_diag = pd.DataFrame(diag_rows)
    margin_diag.to_parquet(T14A_LIGHT / "margin_policy_confirmation_diagnostic.parquet", index=False)

    elapsed = perf_counter() - started
    pd.DataFrame(
        [
            {
                "phase": "task14a_analysis",
                "new_objective_fe": 0,
                "wall_seconds": elapsed,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                "note": "analysis on collected post-handoff artifacts",
            }
        ]
    ).to_parquet(T14A_HEAVY / "task14a_analysis_ledger.parquet", index=False)

    frame.to_parquet(T14A_HEAVY / "post_handoff_analysis_frame.parquet", index=False)
    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print("=== practical sets ===")
        print(practical_sets.round(4).to_string())
        print("=== post-handoff DCM (max) ===")
        print(dcm.loc[dcm["rule"].eq("max")].round(4).to_string())
        print("=== ladder ===")
        print(ladder.round(4).to_string())
        print("=== absorbing ===")
        print(absorbing.round(4).to_string())
        print("=== natural vs post ===")
        print(natural_vs_post.round(4).to_string())
        print("=== reset effects ===")
        print(reset_effects.round(4).to_string())
        print("=== margin diagnostic ===")
        print(margin_diag.round(4).to_string())
        print("=== noise ===")
        print(noise.head(8).round(4).to_string())


if __name__ == "__main__":
    main()
