"""Task 12P: CMA-ES add-back dominance control (runs only after P_balanced
{shade, lshade, cso} was frozen from the Stage-2 selection analysis).

Extends the frozen portfolio with CMA-ES as a fourth action on the identical
Stage-2 states (the isolated add-back branches), recomputes the action-space
metrics, and answers whether CMA-ES collapses the actionable dynamic space.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from behavior_with_ela.protocol import suite_code

ROOT = Path(__file__).resolve().parents[2]
HEAVY = ROOT / "behavior_with_ela/results/analysis_v5/task12"
SHARDS = ROOT / "behavior_with_ela/results/portfolio_screening/task12/stage2/shards"
LIGHT = ROOT / "behavior_with_ela/analysis_v5/task12"
CANDIDATES = ("shade", "lshade", "cso")
DOMINANCE_CONTROL = "cmaes"
PERMUTATION_STREAM = 2026083015


def json_write(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float))


def entropy_of(counts: pd.Series) -> float:
    values = counts.to_numpy(dtype=float)
    probabilities = values / values.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def main() -> None:
    solver = pd.read_parquet(HEAVY / "dynamic_solver_loss_matrix.parquet")
    addback = []
    for shard in sorted(SHARDS.iterdir()):
        if shard.is_dir():
            addback.append(pd.read_parquet(shard / "addback.parquet"))
    addback = pd.concat(addback, ignore_index=True)
    base_addback = addback.loc[addback["replicate_id"].eq(0)]
    cmaes_loss = base_addback.set_index("state_id")["loss_1000"].rename(f"loss_{DOMINANCE_CONTROL}")
    full = solver.join(cmaes_loss, how="left", validate="many_to_one")
    if full[f"loss_{DOMINANCE_CONTROL}"].isna().any():
        raise RuntimeError("add-back branches do not cover every state")

    solvers = [*CANDIDATES, DOMINANCE_CONTROL]
    loss_columns = [f"loss_{name}" for name in solvers]
    order = {name: index for index, name in enumerate(solvers)}

    def practical_best(frame: pd.DataFrame, deltas: dict) -> pd.Series:
        chosen = {}
        for state_id, row in frame.iterrows():
            d = deltas.get(row["suite"], np.nan)
            values = row[loss_columns]
            tied = [
                name
                for name in solvers
                if row[f"loss_{name}"] <= values.min() + d
            ]
            chosen[state_id] = sorted(tied, key=lambda name: order[name])[0]
        return pd.Series(chosen)

    suite_delta95 = {
        suite: float(group["delta_95_function_balanced"].mean())
        for suite, group in pd.read_parquet(
            LIGHT / "dynamic_noise_deltas.parquet"
        ).groupby("suite")
    }

    full["best_action_practical"] = practical_best(full, suite_delta95)
    full["state_best_loss"] = full[loss_columns].min(axis=1)

    rows = []
    for suite_name, group in full.groupby("suite", sort=False):
        d = suite_delta95.get(suite_name, np.nan)
        candidate_fb = (
            group.groupby(["cv_group_id"])[loss_columns]
            .mean()
            .mean()
        )
        sbs = candidate_fb.idxmin().replace("loss_", "")
        group = group.copy()
        group["sbs_loss"] = group[f"loss_{sbs}"]
        pf_action = (
            group.groupby(["problem_id", "FE", "cv_group_id"])[loss_columns]
            .mean()
            .groupby(["problem_id", "FE"])
            .mean()
        )
        pf_best = pf_action.idxmin(axis=1)
        group["problem_fe_best_loss"] = [
            row[pf_best[(row["problem_id"], row["FE"])]]
            for _, row in group.iterrows()
        ]
        dominance = group["best_action_practical"].value_counts(normalize=True)
        rows.append(
            {
                "suite": suite_name,
                "sbs_algorithm_4": sbs,
                "L_SBS_4": float(group["sbs_loss"].mean()),
                "L_statewise_4": float(group["state_best_loss"].mean()),
                "delta_portfolio_4": float(
                    function_balanced_mean(group, "sbs_loss")
                    - function_balanced_mean(group, "state_best_loss")
                ),
                "delta_dynamic_4": float(
                    function_balanced_mean(group, "problem_fe_best_loss")
                    - function_balanced_mean(group, "state_best_loss")
                ),
                "cmaes_practical_win_rate": float(
                    group["best_action_practical"].eq(DOMINANCE_CONTROL).mean()
                ),
                "cmaes_practical_win_rate_bbob_states": float(
                    group.loc[group["suite"].eq("bbob"), "best_action_practical"]
                    .eq(DOMINANCE_CONTROL)
                    .mean()
                ),
                "max_practical_dominance_4": float(dominance.max()),
                "practical_entropy_4": entropy_of(
                    group["best_action_practical"].value_counts()
                ),
                "P_varies_4": float(
                    group.groupby(["problem_id", "seed", "current_algorithm"])[
                        "best_action_practical"
                    ]
                    .nunique()
                    .ge(2)
                    .mean()
                ),
                "delta_95": d,
            }
        )
    table = pd.DataFrame(rows)

    # entropy/varies computed per suite need the conditional version for a fair
    # comparison with the 3-solver analysis: recompute H(A*|problem,FE)
    cond_rows = []
    for suite_name, group in full.groupby("suite", sort=False):
        cond_rows.append(
            {
                "suite": suite_name,
                "H_best_practical": entropy_of(
                    group["best_action_practical"].value_counts()
                ),
                "H_best_given_problem_FE_practical": (
                    lambda frame: float(
                        sum(
                            (len(part) / len(frame))
                            * entropy_of(part["best_action_practical"].value_counts())
                            for _, part in frame.groupby(["problem_id", "FE"])
                        )
                    )
                )(group),
            }
        )
    conditional = pd.DataFrame(cond_rows)
    table = table.merge(conditional, on="suite", how="left")

    table.to_parquet(LIGHT / "cmaes_addback_metrics.parquet", index=False)
    print(table.round(4).to_string())

    # verdict against the 3-solver baseline
    baseline = pd.read_parquet(LIGHT / "portfolio_subset_metrics.parquet")
    verdicts = {}
    for _, row in table.iterrows():
        suite_name = row["suite"]
        base_row = baseline.loc[baseline["suite"].eq(suite_name)].iloc[0]
        delta_ratio = float(row["delta_dynamic_4"] / base_row["delta_dynamic"]) if base_row["delta_dynamic"] else np.nan
        if row["cmaes_practical_win_rate"] >= 0.85 or delta_ratio <= 0.2:
            verdict = "STRONG COLLAPSE"
        elif row["cmaes_practical_win_rate"] >= 0.55 or delta_ratio <= 0.6:
            verdict = "PARTIAL COLLAPSE"
        else:
            verdict = "NO COLLAPSE"
        verdicts[suite_name] = {
            "cmaes_practical_win_rate": float(row["cmaes_practical_win_rate"]),
            "delta_dynamic_3_solvers": float(base_row["delta_dynamic"]),
            "delta_dynamic_4_solvers": float(row["delta_dynamic_4"]),
            "delta_dynamic_ratio_4_over_3": delta_ratio,
            "max_practical_dominance_4": float(row["max_practical_dominance_4"]),
            "verdict": verdict,
        }
    json_write(verdicts, LIGHT / "cmaes_addback_verdict.json")
    print(json.dumps(verdicts, indent=1))


def function_balanced_mean(frame: pd.DataFrame, column: str) -> float:
    return float(frame.groupby("cv_group_id")[column].mean().mean())


if __name__ == "__main__":
    main()
