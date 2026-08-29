"""Task 12 Stage 2 analysis: dynamic noise, action-space metrics, three-layer
oracle headroom, within-problem variation, dynamic DCM, portfolio subsets,
permutation null sensitivity and the isolated CMA-ES add-back control.

CMA-ES add-back columns are loaded ONLY in the add-back function so the
portfolio selection above is structurally blind to them.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from behavior_with_ela.protocol import suite_code

ROOT = Path(__file__).resolve().parents[2]
SHARDS = ROOT / "behavior_with_ela/results/portfolio_screening/task12/stage2/shards"
HEAVY = ROOT / "behavior_with_ela/results/analysis_v5/task12"
LIGHT = ROOT / "behavior_with_ela/analysis_v5/task12"
STAGE1_LIGHT = ROOT / "behavior_with_ela/analysis_v5/task12"
CANDIDATES = ("shade", "lshade", "cso")
DOMINANCE_CONTROL = "cmaes"
CHECKPOINT_FES = (2000, 4000, 6000)
PERMUTATION_STREAM = 2026083013
BOOTSTRAP_STREAM = 2026083014


def json_write(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float))


def load_stage2() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    states, branches, addback, ledgers = [], [], [], []
    for shard in sorted(SHARDS.iterdir()):
        if not shard.is_dir():
            continue
        states.append(pd.read_parquet(shard / "states.parquet"))
        branches.append(pd.read_parquet(shard / "branches.parquet"))
        addback.append(pd.read_parquet(shard / "addback.parquet"))
        ledgers.append(pd.read_parquet(shard / "ledger.parquet"))
    return (
        pd.concat(states, ignore_index=True),
        pd.concat(branches, ignore_index=True),
        pd.concat(addback, ignore_index=True),
        pd.concat(ledgers, ignore_index=True),
    )


def function_balanced(frame: pd.DataFrame, column: str) -> float:
    return float(frame.groupby("cv_group_id")[column].mean().mean())


def entropy_of(counts: pd.Series) -> float:
    values = counts.to_numpy(dtype=float)
    probabilities = values / values.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def conditional_entropy(frame: pd.DataFrame, group_columns: list[str], label_column: str) -> float:
    total = len(frame)
    value = 0.0
    for _, group in frame.groupby(group_columns, sort=False):
        value += (len(group) / total) * entropy_of(group[label_column].value_counts())
    return float(value)


def dynamic_noise(branches: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        branches.groupby(["state_id", "candidate_action", "replicate_id"])["loss_1000"]
        .first()
        .reset_index()
    )
    meta = states.drop_duplicates("state_id")[["state_id", "suite", "cv_group_id"]]
    grouped = grouped.merge(meta, on="state_id", how="left", validate="many_to_one")
    rows = []
    for (suite, action), group in grouped.groupby(["suite", "candidate_action"], sort=False):
        state_groups = group.groupby(["state_id", "candidate_action"])["loss_1000"]
        medians = state_groups.transform("median")
        deviation = (group["loss_1000"] - medians).abs()
        frame = pd.DataFrame(
            {
                "deviation": deviation,
                "cv_group_id": group["cv_group_id"],
            }
        )
        per_function = frame.groupby("cv_group_id")["deviation"]
        quantiles = per_function.quantile([0.50, 0.95]).unstack()
        rows.append(
            {
                "suite": suite,
                "candidate_action": action,
                "delta_50_function_balanced": float(quantiles[0.50].mean()),
                "delta_95_function_balanced": float(quantiles[0.95].mean()),
                "pooled_delta_50": float(frame["deviation"].quantile(0.50)),
                "pooled_delta_95": float(frame["deviation"].quantile(0.95)),
                "samples": int(len(frame)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    HEAVY.mkdir(parents=True, exist_ok=True)
    LIGHT.mkdir(parents=True, exist_ok=True)
    states, branches, addback, ledgers = load_stage2()
    base = branches.loc[branches["replicate_id"].eq(0)].copy()
    base.to_parquet(HEAVY / "dynamic_action_outcomes_1000.parquet", index=False)
    states.to_parquet(HEAVY / "dynamic_screening_states.parquet", index=False)
    ledgers.to_parquet(HEAVY / "task12_resource_ledger.parquet", index=False)

    # ---- noise calibration ----
    noise = dynamic_noise(branches, states)
    noise.to_parquet(HEAVY / "dynamic_noise_deltas.parquet", index=False)
    noise.to_parquet(LIGHT / "dynamic_noise_deltas.parquet", index=False)
    pooled_delta95 = float(
        noise.groupby("candidate_action")["delta_95_function_balanced"].mean().mean()
    )
    suite_delta95 = {
        suite: float(group["delta_95_function_balanced"].mean())
        for suite, group in noise.groupby("suite")
    }

    # ---- solver-semantics loss matrix ----
    # solver X at state s: continue_loss if current == X else switch-to-X loss.
    # Every (state, solver) cell is then defined, and oracles commit to one
    # solver per problem / problem+FE over the SAME state set.
    continue_losses = (
        base.loc[base["candidate_action"].eq("continue")]
        .set_index("state_id")["loss_1000"]
        .rename("continue_loss")
    )
    switch_losses = base.loc[base["candidate_action"].ne("continue")].pivot_table(
        index="state_id", columns="candidate_action", values="loss_1000", aggfunc="first"
    )
    states_indexed = states.set_index("state_id")
    solver_frame = states_indexed[["suite", "cv_group_id", "problem_id", "FE", "seed", "current_algorithm"]].copy()
    for solver in CANDIDATES:
        solver_frame[f"loss_{solver}"] = np.where(
            states_indexed["current_algorithm"].eq(solver),
            continue_losses.reindex(solver_frame.index),
            switch_losses[solver].reindex(solver_frame.index),
        )
    if solver_frame[[f"loss_{s}" for s in CANDIDATES]].isna().any().any():
        raise RuntimeError("solver loss matrix has undefined cells")

    # ---- dynamic best action per state (raw argmin and practical) ----
    loss_columns = [f"loss_{solver}" for solver in CANDIDATES]
    solver_frame["best_action_raw"] = solver_frame[loss_columns].idxmin(axis=1).str.replace("loss_", "")
    suite_of = solver_frame["suite"]
    best_practical = {}
    for state_id, row in solver_frame.iterrows():
        d = suite_delta95.get(row["suite"], pooled_delta95)
        values = row[loss_columns]
        tied = [solver for solver in CANDIDATES if row[f"loss_{solver}"] <= values.min() + d]
        best_practical[state_id] = tied[0]
    solver_frame["best_action_practical"] = pd.Series(best_practical)
    solver_frame.to_parquet(HEAVY / "dynamic_solver_loss_matrix.parquet")

    # ---- 22K dynamic action distribution & entropies ----
    entropy_rows = []
    for label, column in (("raw", "best_action_raw"), ("practical", "best_action_practical")):
        entropy_rows.append(
            {
                "label": label,
                "H_best": entropy_of(solver_frame[column].value_counts()),
                "H_best_given_current": conditional_entropy(solver_frame, ["current_algorithm"], column),
                "H_best_given_problem": conditional_entropy(solver_frame, ["problem_id"], column),
                "H_best_given_problem_FE": conditional_entropy(solver_frame, ["problem_id", "FE"], column),
                "H_best_given_current_problem_FE": conditional_entropy(
                    solver_frame, ["current_algorithm", "problem_id", "FE"], column
                ),
            }
        )
    entropies = pd.DataFrame(entropy_rows)
    entropies.to_parquet(LIGHT / "dynamic_action_entropy.parquet", index=False)

    distribution = (
        solver_frame.groupby(["current_algorithm", "best_action_practical"], sort=False)
        .size()
        .reset_index(name="count")
    )
    totals = solver_frame.groupby("current_algorithm", sort=False).size().rename("total")
    distribution = distribution.merge(totals.reset_index(), on="current_algorithm")
    distribution["share"] = distribution["count"] / distribution["total"]
    distribution.to_parquet(LIGHT / "dynamic_best_action_distribution.parquet", index=False)

    # ---- 23L three-layer oracle headroom on the solver matrix ----
    headroom_rows = []
    augmented = {}
    for suite_name, suite_group in solver_frame.groupby("suite", sort=False):
        d = suite_delta95.get(suite_name, pooled_delta95)
        candidate_fb = (
            suite_group.groupby(["cv_group_id"])[[f"loss_{s}" for s in CANDIDATES]]
            .mean()
            .mean()
        )
        sbs = candidate_fb.idxmin().replace("loss_", "")
        suite_group = suite_group.copy()
        suite_group["sbs_loss"] = suite_group[f"loss_{sbs}"]
        suite_group["state_best_loss"] = suite_group[loss_columns].min(axis=1)
        problem_action = (
            suite_group.groupby(["problem_id", "cv_group_id"])[loss_columns]
            .mean()
            .groupby("problem_id")
            .mean()
        )
        problem_best = problem_action.idxmin(axis=1)
        suite_group["problem_best_loss"] = [
            row[problem_best[row["problem_id"]]] for _, row in suite_group.iterrows()
        ]
        pf_action = (
            suite_group.groupby(["problem_id", "FE", "cv_group_id"])[loss_columns]
            .mean()
            .groupby(["problem_id", "FE"])
            .mean()
        )
        pf_best = pf_action.idxmin(axis=1)
        suite_group["problem_fe_best_loss"] = [
            row[pf_best[(row["problem_id"], row["FE"])]]
            for _, row in suite_group.iterrows()
        ]
        headroom_rows.append(
            {
                "suite": suite_name,
                "sbs_algorithm": sbs,
                "L_SBS": function_balanced(suite_group.reset_index(), "sbs_loss"),
                "L_problem_static": function_balanced(suite_group.reset_index(), "problem_best_loss"),
                "L_problem_FE": function_balanced(suite_group.reset_index(), "problem_fe_best_loss"),
                "L_statewise": function_balanced(suite_group.reset_index(), "state_best_loss"),
                "delta_portfolio": function_balanced(suite_group.reset_index(), "sbs_loss")
                - function_balanced(suite_group.reset_index(), "state_best_loss"),
                "delta_problem": function_balanced(suite_group.reset_index(), "problem_best_loss")
                - function_balanced(suite_group.reset_index(), "state_best_loss"),
                "delta_dynamic": function_balanced(suite_group.reset_index(), "problem_fe_best_loss")
                - function_balanced(suite_group.reset_index(), "state_best_loss"),
                "delta_95": d,
                "sbs_fb_loss_per_solver": json.dumps(
                    {solver: float(value) for solver, value in candidate_fb.items()}
                ),
            }
        )
        # exclusive-win metrics per solver (dynamic)
        best_gain_columns = suite_group[loss_columns].min(axis=1)
        for solver in CANDIDATES:
            others = [column for column in loss_columns if column != f"loss_{solver}"]
            exclusive = suite_group[others].gt(suite_group[f"loss_{solver}"] + d, axis=0).all(axis=1)
            rival_best = suite_group[others].min(axis=1)
            headroom_rows[-1][f"exclusive_win_{solver}"] = float(exclusive.mean())
            headroom_rows[-1][f"exclusive_gain_mass_{solver}"] = float(
                (rival_best - suite_group[f"loss_{solver}"] - d).clip(lower=0.0).mean()
            )
        augmented[suite_name] = suite_group
    headroom = pd.DataFrame(headroom_rows)
    headroom.to_parquet(LIGHT / "dynamic_oracle_headroom.parquet", index=False)
    augmented_frames = augmented

    # ---- function-level bootstrap CI for the three headrooms ----
    bootstrap_rows = []
    for suite_name, suite_group in augmented_frames.items():
        functions = sorted(suite_group["cv_group_id"].unique())
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [BOOTSTRAP_STREAM, suite_code(suite_name), len(functions)]
            ).generate_state(4)
        )
        columns = {
            "L_SBS": "sbs_loss",
            "L_problem_static": "problem_best_loss",
            "L_problem_FE": "problem_fe_best_loss",
            "L_statewise": "state_best_loss",
        }
        draws = {name: [] for name in columns}
        for _ in range(2000):
            sample = rng.choice(functions, size=len(functions), replace=True)
            blocks = [suite_group.loc[suite_group["cv_group_id"].eq(name)] for name in sample]
            resampled = pd.concat(blocks, ignore_index=True)
            for name, column in columns.items():
                draws[name].append(float(resampled[column].mean()))
        for name in columns:
            bootstrap_rows.append(
                {
                    "suite": suite_name,
                    "quantity": name,
                    "mean": float(np.mean(draws[name])),
                    "ci_low": float(np.quantile(draws[name], 0.025)),
                    "ci_high": float(np.quantile(draws[name], 0.975)),
                }
            )
        for name, (upper, lower) in (
            ("delta_portfolio", ("sbs_loss", "state_best_loss")),
            ("delta_problem", ("problem_best_loss", "state_best_loss")),
            ("delta_dynamic", ("problem_fe_best_loss", "state_best_loss")),
        ):
            deltas = []
            for _ in range(2000):
                sample = rng.choice(functions, size=len(functions), replace=True)
                blocks = [suite_group.loc[suite_group["cv_group_id"].eq(name)] for name in sample]
                resampled = pd.concat(blocks, ignore_index=True)
                deltas.append(float(resampled[upper].mean() - resampled[lower].mean()))
            bootstrap_rows.append(
                {
                    "suite": suite_name,
                    "quantity": name,
                    "mean": float(np.mean(deltas)),
                    "ci_low": float(np.quantile(deltas, 0.025)),
                    "ci_high": float(np.quantile(deltas, 0.975)),
                }
            )
    bootstrap_table = pd.DataFrame(bootstrap_rows)
    bootstrap_table.to_parquet(LIGHT / "oracle_headroom_bootstrap.parquet", index=False)

    # ---- 24M within-problem dynamic variation ----
    variation_rows = []
    for (suite_name, problem_id, seed, current), group in solver_frame.groupby(
        ["suite", "problem_id", "seed", "current_algorithm"], sort=False
    ):
        best_actions = group.sort_values("FE")["best_action_practical"].dropna()
        variation_rows.append(
            {
                "suite": suite_name,
                "problem_id": problem_id,
                "seed": seed,
                "current": current,
                "distinct_practical_best": int(best_actions.nunique()),
                "varies": int(best_actions.nunique() >= 2),
            }
        )
    variation = pd.DataFrame(variation_rows)
    variation.to_parquet(HEAVY / "within_problem_dynamic_variation.parquet", index=False)
    variation_summary = (
        variation.groupby("suite")
        .agg(trajectories=("varies", "size"), P_varies=("varies", "mean"))
        .reset_index()
    )
    variation_summary.to_parquet(LIGHT / "within_problem_dynamic_variation.parquet", index=False)

    # ---- 25N dynamic pairwise DCM on the solver matrix ----
    dcm_rows = []
    for suite_name, suite_group in solver_frame.groupby("suite", sort=False):
        d = suite_delta95.get(suite_name, pooled_delta95)
        for a, b in combinations(CANDIDATES, 2):
            diff = suite_group[f"loss_{a}"] - suite_group[f"loss_{b}"]
            tie = float((diff.abs() <= d).mean())
            a_better = float((diff < -d).mean())
            b_better = float((diff > d).mean())
            dcm_rows.append(
                {
                    "suite": suite_name,
                    "candidate_a": a,
                    "candidate_b": b,
                    "P_tie": tie,
                    "P_a_better": a_better,
                    "P_b_better": b_better,
                    "DCM": float((tie + abs(a_better - b_better)) / 2.0),
                    "matrix": "dynamic",
                }
            )
    natural = pd.read_parquet(STAGE1_LIGHT / "static_pairwise_dcm.parquet")
    for row in natural.itertuples():
        if row.candidate_a in CANDIDATES and row.candidate_b in CANDIDATES:
            dcm_rows.append(
                {
                    "suite": row.suite,
                    "candidate_a": row.candidate_a,
                    "candidate_b": row.candidate_b,
                    "P_tie": row.P_tie,
                    "P_a_better": row.P_a_better,
                    "P_b_better": row.P_b_better,
                    "DCM": row.DCM,
                    "matrix": f"natural_fe{row.FE}",
                }
            )
    dcm = pd.DataFrame(dcm_rows)
    dcm.to_parquet(LIGHT / "dynamic_pairwise_dcm.parquet", index=False)

    # ---- 34 permutation null sensitivity ----
    permutation = {"H_best_given_problem_FE": {}, "P_varies": {}}
    for suite_name, suite_group in solver_frame.groupby("suite", sort=False):
        observed = conditional_entropy(
            suite_group, ["problem_id", "FE"], "best_action_practical"
        )
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [PERMUTATION_STREAM, suite_code(suite_name)]
            ).generate_state(4)
        )
        labels = suite_group["best_action_practical"].to_numpy().copy()
        within_values = labels.copy()
        shuffled = suite_group.copy()
        null_entropy_global = []
        null_varies = []
        for _ in range(200):
            labels = rng.permutation(labels)
            shuffled["null_label"] = labels
            null_entropy_global.append(
                conditional_entropy(shuffled, ["problem_id", "FE"], "null_label")
            )
            for _, indexes in suite_group.groupby(["problem_id", "FE"]).groups.items():
                positions = suite_group.index.get_indexer(indexes)
                within_values[positions] = rng.permutation(within_values[positions])
            shuffled["within_label"] = within_values
            varies = (
                shuffled.groupby(["problem_id", "seed", "current_algorithm"])["within_label"]
                .nunique()
                .ge(2)
                .mean()
            )
            null_varies.append(float(varies))
        permutation["H_best_given_problem_FE"][suite_name] = {
            "observed": observed,
            "null_mean_global_shuffle": float(np.mean(null_entropy_global)),
            "null_p05_global_shuffle": float(np.quantile(null_entropy_global, 0.05)),
            "marginal_entropy": entropy_of(suite_group["best_action_practical"].value_counts()),
        }
        permutation["P_varies"][suite_name] = {
            "observed": float(
                variation.loc[variation["suite"].eq(suite_name), "varies"].mean()
            ),
            "null_mean": float(np.mean(null_varies)),
            "null_p95": float(np.quantile(null_varies, 0.95)),
        }
    json_write(permutation, LIGHT / "permutation_null_sensitivity.json")

    # ---- 27 portfolio subset metrics (single 3-solver subset from KEEP set) ----
    subset_rows = []
    for suite_name, suite_group in solver_frame.groupby("suite", sort=False):
        d = suite_delta95.get(suite_name, pooled_delta95)
        candidate_fb = (
            suite_group.groupby(["cv_group_id"])[[f"loss_{s}" for s in CANDIDATES]]
            .mean()
            .mean()
        )
        sbs = candidate_fb.idxmin().replace("loss_", "")
        suite_group = suite_group.copy()
        suite_group["sbs_loss"] = suite_group[f"loss_{sbs}"]
        suite_group["state_best_loss"] = suite_group[loss_columns].min(axis=1)
        pf_action = (
            suite_group.groupby(["problem_id", "FE", "cv_group_id"])[loss_columns]
            .mean()
            .groupby(["problem_id", "FE"])
            .mean()
        )
        pf_best = pf_action.idxmin(axis=1)
        suite_group["problem_fe_best_loss"] = [
            row[pf_best[(row["problem_id"], row["FE"])]]
            for _, row in suite_group.iterrows()
        ]
        dominance = suite_group["best_action_practical"].value_counts(normalize=True)
        subset_rows.append(
            {
                "suite": suite_name,
                "subset": "+".join(CANDIDATES),
                "size": len(CANDIDATES),
                "sbs_algorithm": sbs,
                "L_SBS": function_balanced(suite_group.reset_index(), "sbs_loss"),
                "L_problem_FE": function_balanced(suite_group.reset_index(), "problem_fe_best_loss"),
                "L_statewise": function_balanced(suite_group.reset_index(), "state_best_loss"),
                "delta_portfolio": function_balanced(suite_group.reset_index(), "sbs_loss")
                - function_balanced(suite_group.reset_index(), "state_best_loss"),
                "delta_dynamic": function_balanced(suite_group.reset_index(), "problem_fe_best_loss")
                - function_balanced(suite_group.reset_index(), "state_best_loss"),
                "max_practical_dominance": float(dominance.max()),
                "practical_entropy": entropy_of(
                    suite_group["best_action_practical"].value_counts()
                ),
            }
        )
    subsets = pd.DataFrame(subset_rows)
    subsets.to_parquet(LIGHT / "portfolio_subset_metrics.parquet", index=False)

    print(noise.to_string())
    print(entropies.round(4).to_string())
    print(headroom.round(4).to_string())
    print(variation_summary.round(4).to_string())
    print(subsets.round(4).to_string())
    print(json.dumps(permutation, indent=1, default=str))


if __name__ == "__main__":
    main()
