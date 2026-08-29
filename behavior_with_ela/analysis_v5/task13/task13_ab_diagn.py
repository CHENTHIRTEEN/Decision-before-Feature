"""Task 13A/13B: zero-FE diagnostics that gate the behavior replay.

13A re-estimates the winner's-curse bias of the replicate-0 statewise best
observed action with an independent estimator: the winner cell selected on
replicate 0 is re-scored by the mean of replicates 1 and 2 only, so the
selection outcome never enters the reference value (Task 12.1 used the
median of r0..r2, which contains r0).

13B re-computes the current-preserving practical action space under three
pre-registered pairwise-delta combination rules (max, quadrature, sum of the
Task 12 per-action deltas) and verdicts whether the balanced portfolio stays
non-degenerate under the most conservative rule. A FRAGILE verdict stops the
round before any replay.
"""
from __future__ import annotations

import json
import resource
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from behavior_with_ela.analysis_v5.task12_1_analysis import (
    CHECKPOINT_FES,
    DIRECT_PAIR_MIN_N,
    HEAVY as T12_HEAVY,
    SOLVERS,
    TASK12_HEAVY,
    conditional_entropy,
    entropy_of,
    fb_mean,
    fb_series,
    load_shard,
    statewise_pairwise_delta,
)

ROOT = Path(__file__).resolve().parents[3]
HEAVY = ROOT / "behavior_with_ela/results/analysis_v5/task13"
LIGHT = ROOT / "behavior_with_ela/analysis_v5/task13"
BOOTSTRAP_STREAM = 2026090113
BOOTSTRAP_DRAWS = 2000
BIDIRECTIONAL_MIN = 0.05  # pre-specified: both direction probabilities >= this
STABLE_MIN_SWITCH = 0.10
FRAGILE_MAX_SWITCH = 0.05

DELTA_RULES = ("max", "quadrature", "sum")


def bootstrap_fb_ci(frame: pd.DataFrame, column: str, stream_offset: int) -> tuple[float, float, float]:
    groups = sorted(frame["cv_group_id"].unique())
    means = fb_series(frame, column).to_dict()
    rng = np.random.default_rng(
        np.random.SeedSequence([BOOTSTRAP_STREAM + stream_offset, len(groups)]).generate_state(4)
    )
    draws = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        sample = rng.choice(groups, size=len(groups), replace=True)
        draws[draw] = np.mean([means[g] for g in sample])
    return (
        float(fb_series(frame, column).mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


# --------------------------------------------------------------------------
# 13A independent winner's-curse diagnostic
# --------------------------------------------------------------------------
def independent_oracle_bias(branches: pd.DataFrame, base: pd.DataFrame, frame3: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = branches.pivot_table(
        index=["state_id", "candidate_action"], columns="replicate_id", values="loss_1000"
    )
    cell_r0 = base.set_index(["state_id", "candidate_action"])["loss_1000"]
    rows = []
    for state_id, state_row in frame3.set_index("state_id").iterrows():
        cells = {s: ("continue" if state_row["current_algorithm"] == s else s) for s in SOLVERS}
        losses0 = {}
        for s in SOLVERS:
            try:
                losses0[s] = float(cell_r0.loc[(state_id, cells[s])])
            except KeyError:
                losses0[s] = np.nan
        if any(np.isnan(v) for v in losses0.values()):
            continue
        best = min(losses0, key=losses0.get)
        try:
            rep = wide.loc[(state_id, cells[best])]
        except KeyError:
            continue
        if 1 in rep.index and 2 in rep.index and np.isfinite(rep.loc[1]) and np.isfinite(rep.loc[2]):
            meta = state_row
            rows.append(
                {
                    "state_id": state_id,
                    "suite": meta["suite"],
                    "cv_group_id": meta["cv_group_id"],
                    "current_algorithm": meta["current_algorithm"],
                    "FE": int(meta["FE"]),
                    "winner_cell": cells[best],
                    "r0": losses0[best],
                    "independent_mean_r12": float((rep.loc[1] + rep.loc[2]) / 2.0),
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise SystemExit("[task13A] no repeated winner cells found")
    detail["bias_independent"] = detail["independent_mean_r12"] - detail["r0"]

    strata_rows = []
    stream = 0
    for suite_name, group in detail.groupby("suite", sort=False):
        mean, low, high = bootstrap_fb_ci(group, "bias_independent", stream_offset=stream)
        stream += 10
        strata_rows.append(
            {
                "suite": suite_name,
                "stratum": "overall",
                "current": "all",
                "FE": 0,
                "n": len(group),
                "pooled_mean": float(group["bias_independent"].mean()),
                "fb_mean": mean,
                "ci_low": low,
                "ci_high": high,
                "coverage": "ok" if len(group) >= DIRECT_PAIR_MIN_N else "INSUFFICIENT_diagnostic_only",
            }
        )
        for (current, fe), part in group.groupby(["current_algorithm", "FE"]):
            strata_rows.append(
                {
                    "suite": suite_name,
                    "stratum": "current_x_FE",
                    "current": current,
                    "FE": int(fe),
                    "n": len(part),
                    "pooled_mean": float(part["bias_independent"].mean()),
                    "fb_mean": fb_mean(part, "bias_independent"),
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "coverage": "ok" if len(part) >= DIRECT_PAIR_MIN_N else "INSUFFICIENT_diagnostic_only",
                }
            )
    return detail, pd.DataFrame(strata_rows)


# --------------------------------------------------------------------------
# 13B delta-rule sensitivity
# --------------------------------------------------------------------------
def set_valued_from_delta_matrix(frame: pd.DataFrame, solvers, delta_matrix: np.ndarray) -> pd.DataFrame:
    """Generalization of the Task 12.1 set-valued construction to an arbitrary
    pre-computed (n, |A|, |A|) pairwise delta matrix (diagonal unused)."""
    frame = frame.reset_index(drop=True)
    loss_cols = [f"loss_{s}" for s in solvers]
    values = frame[loss_cols].to_numpy()
    n = len(frame)
    dominates = values[:, :, None] < values[:, None, :] - delta_matrix
    non_dominated = ~dominates.any(axis=1)
    set_size = non_dominated.sum(axis=1)
    in_nd = np.array([non_dominated[row, solvers.index(c)] for row, c in enumerate(frame["current_algorithm"])])
    switch_required = ~in_nd
    raw_argmin = values.argmin(axis=1)
    targets, op_actions = [], []
    for row in range(n):
        candidates = np.where(non_dominated[row])[0]
        best = candidates[np.argmin(values[row, candidates])] if len(candidates) else int(raw_argmin[row])
        if switch_required[row]:
            targets.append(solvers[best])
            op_actions.append(solvers[best])
        else:
            targets.append("")
            op_actions.append("continue")
    frame["A_ND_size"] = set_size
    frame["A_ND_empty"] = set_size == 0
    frame["A_ND_members"] = ["|".join(solvers[i] for i in np.where(non_dominated[row])[0]) for row in range(n)]
    frame["current_in_A_ND"] = in_nd
    frame["switch_required"] = switch_required
    frame["switch_target"] = targets
    frame["operational_action"] = op_actions
    frame["optional_switch"] = in_nd & (set_size > 1)
    return frame


def build_delta_matrix(frame: pd.DataFrame, cell_delta: pd.DataFrame, rule: str) -> np.ndarray:
    solvers = SOLVERS
    n = len(frame)
    matrix = np.zeros((n, len(solvers), len(solvers)))
    for a_index, a in enumerate(solvers):
        for b_index, b in enumerate(solvers):
            if a == b:
                continue
            da = cell_delta[f"delta_cell_{a}"].to_numpy()
            db = cell_delta[f"delta_cell_{b}"].to_numpy()
            if rule == "max":
                matrix[:, a_index, b_index] = np.maximum(da, db)
            elif rule == "quadrature":
                matrix[:, a_index, b_index] = np.sqrt(da**2 + db**2)
            elif rule == "sum":
                matrix[:, a_index, b_index] = da + db
            else:
                raise ValueError(rule)
    return matrix


def main() -> None:
    HEAVY.mkdir(parents=True, exist_ok=True)
    LIGHT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    states = pd.read_parquet(TASK12_HEAVY / "dynamic_screening_states.parquet")
    base = pd.read_parquet(TASK12_HEAVY / "dynamic_action_outcomes_1000.parquet")
    solver = pd.read_parquet(TASK12_HEAVY / "dynamic_solver_loss_matrix.parquet")
    noise = pd.read_parquet(TASK12_HEAVY / "dynamic_noise_deltas.parquet")
    branches = load_shard("branches")
    t12_sets = pd.read_parquet(T12_HEAVY / "practical_action_sets.parquet")

    frame3 = solver.merge(
        states[["state_id", "family", "instance"]], on="state_id", validate="many_to_one"
    )
    if not frame3["state_id"].tolist() == states["state_id"].tolist():
        frame3 = states[["state_id"]].merge(frame3, on="state_id", validate="one_to_one")

    # ---- 13A ----
    detail, strata = independent_oracle_bias(branches, base, frame3)
    detail.to_parquet(HEAVY / "independent_oracle_bias_states.parquet", index=False)
    strata.to_parquet(LIGHT / "independent_oracle_bias.parquet", index=False)

    # ---- 13B ----
    per_action = noise.set_index(["suite", "candidate_action"])["delta_95_function_balanced"]
    delta_ctx = statewise_pairwise_delta(frame3, per_action, SOLVERS)
    delta_ctx = delta_ctx.set_index("state_id").loc[frame3["state_id"]].reset_index()

    sens_rows = []
    variation_rows = []
    per_rule_sets = {}
    for rule in DELTA_RULES:
        parts = []
        for suite_name, group in frame3.groupby("suite", sort=False):
            group = group.reset_index(drop=True)
            ctx = delta_ctx.set_index("state_id").loc[group["state_id"]].reset_index()
            matrix = build_delta_matrix(group, ctx, rule)
            sv = set_valued_from_delta_matrix(group, SOLVERS, matrix)
            sv["suite"] = suite_name
            parts.append(sv)
        sv = pd.concat(parts, ignore_index=True)
        per_rule_sets[rule] = sv
        for suite_name, group in sv.groupby("suite", sort=False):
            group = group.reset_index(drop=True)
            ctx = delta_ctx.set_index("state_id").loc[group["state_id"]]
            cell_delta_arrays = {s: ctx[f"delta_cell_{s}"].to_numpy() for s in SOLVERS}
            row = {
                "rule": rule,
                "suite": suite_name,
                "switch_required_rate": float(group["switch_required"].mean()),
                "P_current_in_A_ND": float(group["current_in_A_ND"].mean()),
                "P_A_ND_size_gt1": float(group["A_ND_size"].gt(1).mean()),
                "P_A_ND_empty": float(group["A_ND_empty"].mean()),
                "H_operational": entropy_of(group["operational_action"].value_counts()),
                "optional_switch_rate": float(group["optional_switch"].mean()),
            }
            for a, b in combinations(SOLVERS, 2):
                la = group[f"loss_{a}"].to_numpy()
                lb = group[f"loss_{b}"].to_numpy()
                da = cell_delta_arrays[a]
                db = cell_delta_arrays[b]
                if rule == "max":
                    d = np.maximum(da, db)
                elif rule == "quadrature":
                    d = np.sqrt(da**2 + db**2)
                else:
                    d = da + db
                a_better = float((la < lb - d).mean())
                b_better = float((lb < la - d).mean())
                row[f"P_{a}_better_{b}"] = a_better
                row[f"P_{b}_better_{a}"] = b_better
                row[f"DCM_{a}|{b}"] = float(
                    ((np.abs(la - lb) <= d).mean() + abs(a_better - b_better)) / 2.0
                )
            sens_rows.append(row)
        # within-trajectory variation
        for suite_name, group in sv.groupby("suite", sort=False):
            status = group.groupby(["problem_id", "seed", "current_algorithm"])["switch_required"].nunique()
            target_required = group.loc[group["switch_required"]].groupby(
                ["problem_id", "seed", "current_algorithm"]
            )["switch_target"].nunique()
            variation_rows.append(
                {
                    "rule": rule,
                    "suite": suite_name,
                    "P_switch_status_varies": float(status.ge(2).mean()),
                    "P_target_varies": float(target_required.ge(2).mean()),
                }
            )
    sensitivity = pd.DataFrame(sens_rows)
    variation = pd.DataFrame(variation_rows)
    sensitivity.to_parquet(LIGHT / "delta_sensitivity_max_quad_sum.parquet", index=False)
    variation.to_parquet(LIGHT / "delta_sensitivity_variation.parquet", index=False)
    for rule, sv in per_rule_sets.items():
        sv.to_parquet(HEAVY / f"practical_action_sets_{rule}.parquet", index=False)

    # continuity check: max rule must reproduce the Task 12.1 pairwise sets
    check = per_rule_sets["max"].sort_values("state_id").reset_index(drop=True)
    ref = t12_sets.sort_values("state_id").reset_index(drop=True)
    if not np.allclose(
        check["switch_required"].to_numpy(dtype=float), ref["switch_required"].to_numpy(dtype=float)
    ):
        raise SystemExit("[task13B] max-rule set construction does not reproduce Task 12.1")

    # ---- verdict ----
    verdict_rows = []
    rules_summary = {}
    for rule in DELTA_RULES:
        sub = sensitivity.loc[sensitivity["rule"].eq(rule)]
        min_switch = float(sub["switch_required_rate"].min())
        min_dir = np.inf
        for suite_name in sub["suite"].unique():
            row = sub.loc[sub["suite"].eq(suite_name)].iloc[0]
            for a, b in combinations(SOLVERS, 2):
                min_dir = min(min_dir, row[f"P_{a}_better_{b}"], row[f"P_{b}_better_{a}"])
        rules_summary[rule] = {"min_switch_required": min_switch, "min_direction_probability": float(min_dir)}
    sum_rule = rules_summary["sum"]
    quad_rule = rules_summary["quadrature"]
    if quad_rule["min_switch_required"] < FRAGILE_MAX_SWITCH:
        verdict = "FRAGILE"
    elif sum_rule["min_switch_required"] < STABLE_MIN_SWITCH or sum_rule["min_direction_probability"] < BIDIRECTIONAL_MIN:
        verdict = "MODERATE"
    else:
        verdict = "STABLE"
    for rule in DELTA_RULES:
        verdict_rows.append({"rule": rule, **rules_summary[rule]})
    verdict_table = pd.DataFrame(verdict_rows)
    verdict_table["verdict"] = verdict
    verdict_table.to_parquet(LIGHT / "delta_sensitivity_verdict.parquet", index=False)

    summary = {
        "independent_bias_overall": {
            row["suite"]: {
                "n": int(row["n"]),
                "pooled_mean": float(row["pooled_mean"]),
                "fb_mean": float(row["fb_mean"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "coverage": row["coverage"],
            }
            for _, row in strata.loc[strata["stratum"].eq("overall")].iterrows()
        },
        "delta_rules": rules_summary,
        "delta_verdict": verdict,
        "gate": "PROCEED" if verdict != "FRAGILE" else "STOP_BEFORE_REPLAY",
        "wall_seconds": perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    (LIGHT / "13ab_gate_summary.json").write_text(json.dumps(summary, indent=2))

    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print("=== 13A independent oracle bias (overall) ===")
        print(strata.loc[strata["stratum"].eq("overall")].round(4).to_string())
        print("=== 13A by current x FE ===")
        print(strata.loc[strata["stratum"].eq("current_x_FE")].round(4).to_string())
        print("=== 13B sensitivity ===")
        print(sensitivity.round(4).to_string())
        print("=== 13B variation ===")
        print(variation.round(4).to_string())
        print("=== verdict ===")
        print(json.dumps(summary["delta_rules"], indent=1), verdict, summary["gate"])


if __name__ == "__main__":
    main()
