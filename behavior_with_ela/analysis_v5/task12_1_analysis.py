"""Task 12.1: robustness re-analysis of the Task 12 dynamic performance
residuals and structural feasibility of a progress gate.

Zero new objective evaluations: every quantity reuses the Task 12 Stage-2
artifacts (1,890 candidate-current states x {continue, 2 switches} x
replicate-0, plus the ~10 percent R=3 repetition cells and the isolated
CMA-ES add-back branches). The checkpoint current log-gap (ell_t), which the
Stage-2 tables never stored, is recovered from the Task 12 Stage-1 natural
marks after a two-identity bit-level alignment check; if that check failed
the script stops instead of mixing sources.

No model is trained in this round (no Behavior selector, no progress
predictor, no re-selection of the portfolio), and no objective function is
evaluated.
"""
from __future__ import annotations

import json
import resource
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
TASK12_HEAVY = ROOT / "behavior_with_ela/results/analysis_v5/task12"
TASK12_SHARDS = ROOT / "behavior_with_ela/results/portfolio_screening/task12/stage2/shards"
STAGE1_SHARDS = ROOT / "behavior_with_ela/results/portfolio_screening/task12/stage1/shards"
HEAVY = ROOT / "behavior_with_ela/results/analysis_v5/task12_1"
LIGHT = ROOT / "behavior_with_ela/analysis_v5/task12_1"
SOLVERS = ("shade", "lshade", "cso")
SOLVERS4 = ("shade", "lshade", "cso", "cmaes")
CHECKPOINT_FES = (2000, 4000, 6000)
HORIZON = 1000
TRIGGER_QUANTILES = (0.20, 0.30, 0.40)
DIRECT_PAIR_MIN_N = 30  # pre-specified minimum for a direct pairwise delta
BOOTSTRAP_STREAM = 2026083112
BOOTSTRAP_DRAWS = 2000


class SystemStop(Exception):
    pass


def fb_mean(frame: pd.DataFrame, column: str) -> float:
    return float(frame.groupby("cv_group_id")[column].mean().mean())


def fb_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("cv_group_id")[column].mean()


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


def load_shard(name: str) -> pd.DataFrame:
    frames = []
    for shard in sorted(TASK12_SHARDS.iterdir()):
        if shard.is_dir():
            frames.append(pd.read_parquet(shard / f"{name}.parquet"))
    return pd.concat(frames, ignore_index=True)


def load_stage1_natural_marks() -> pd.DataFrame:
    frames = []
    for shard in sorted(STAGE1_SHARDS.iterdir()):
        if shard.is_dir():
            frames.append(pd.read_parquet(shard / "runs.parquet"))
    runs = pd.concat(frames, ignore_index=True)
    keep = (
        runs["replicate_id"].eq(0)
        & runs["candidate"].isin(SOLVERS)
        & runs["FE"].isin([*CHECKPOINT_FES, 3000, 5000, 7000, 10000])
    )
    return runs.loc[keep]


# --------------------------------------------------------------------------
# 12.1R alignment check (existing traces only; no replay is executed)
# --------------------------------------------------------------------------
def verify_alignment(states: pd.DataFrame, base: pd.DataFrame, stage1: pd.DataFrame) -> pd.DataFrame:
    """Identity A: stage-1 FE=10000 mark == stage-2 terminal_loss_current_run.
    Identity B: stage-1 FE=t+1000 mark == stage-2 continue-branch loss from
    checkpoint t. Both must hold bit-exactly for every state."""
    terminal = stage1.loc[stage1["FE"].eq(10000)][["problem_id", "seed", "candidate", "log10_gap"]]
    check = states.merge(
        terminal,
        left_on=["problem_id", "seed", "current_algorithm"],
        right_on=["problem_id", "seed", "candidate"],
        how="left",
        validate="many_to_one",
    )
    if check["log10_gap"].isna().any():
        raise SystemStop("stage-1 traces do not cover every stage-2 state")
    check = check.rename(columns={"log10_gap": "stage1_terminal"})
    check["terminal_abs_diff"] = (check["stage1_terminal"] - check["terminal_loss_current_run"]).abs()

    continue_rows = base.loc[base["candidate_action"].eq("continue")][["state_id", "FE", "loss_1000"]]
    continue_rows = continue_rows.merge(
        states[["state_id", "problem_id", "seed", "current_algorithm"]], on="state_id", validate="many_to_one"
    )
    cont_parts = []
    for fe in CHECKPOINT_FES:
        marks = stage1.loc[stage1["FE"].eq(fe + HORIZON)][["problem_id", "seed", "candidate", "log10_gap"]]
        merged = continue_rows.loc[continue_rows["FE"].eq(fe)].merge(
            marks,
            left_on=["problem_id", "seed", "current_algorithm"],
            right_on=["problem_id", "seed", "candidate"],
            how="left",
            validate="many_to_one",
        )
        if merged["log10_gap"].isna().any():
            raise SystemStop(f"stage-1 marks missing for FE={fe} continuation")
        merged["continue_abs_diff"] = (merged["log10_gap"] - merged["loss_1000"]).abs()
        cont_parts.append(merged[["state_id", "continue_abs_diff"]])
    cont_check = pd.concat(cont_parts, ignore_index=True)

    alignment = check[["state_id", "terminal_abs_diff"]].merge(
        cont_check, on="state_id", validate="one_to_one"
    )
    if alignment["terminal_abs_diff"].max() > 1e-12 or alignment["continue_abs_diff"].max() > 1e-12:
        raise SystemStop(
            "stage-1 traces do not reproduce stage-2 states bit-for-bit; per the "
            "pre-specified stop rule the checkpoint gap must not be mixed in"
        )
    return alignment


# --------------------------------------------------------------------------
# 12.1A semantics checklist (facts from tables and collection code)
# --------------------------------------------------------------------------
def semantics_checklist(
    states_raw: pd.DataFrame,
    base: pd.DataFrame,
    branches: pd.DataFrame,
    solver: pd.DataFrame,
    suite_legacy: dict,
) -> dict:
    loss_cols = [f"loss_{s}" for s in SOLVERS]
    losses = solver[loss_cols].to_numpy()
    suite_delta = solver["suite"].map(suite_legacy).to_numpy()
    min_loss = losses.min(axis=1)
    # Task 12 practical tie set: {a : L_a <= min + delta_suite}
    in_tie = losses <= min_loss[:, None] + suite_delta[:, None]
    is_tie = in_tie.sum(axis=1) > 1
    raw_argmin = solver["best_action_raw"].to_numpy()
    practical = solver["best_action_practical"].to_numpy()
    tie_resolved_away_from_raw = practical != raw_argmin
    return {
        "1_all_states_natural_current": bool(
            states_raw["source_algorithm"].eq(states_raw["current_algorithm"]).all()
            and states_raw["route"].eq("current_" + states_raw["current_algorithm"]).all()
        ),
        "2_no_real_post_handoff_states": bool(
            states_raw["handoff_performed"].eq(False).all() and states_raw["dwell_FE"].eq(states_raw["FE"]).all()
        ),
        "3_source_equals_current_always": bool(
            states_raw["source_algorithm"].eq(states_raw["current_algorithm"]).all()
        ),
        "4_handoff_performed_always_false": bool(states_raw["handoff_performed"].eq(False).all()),
        "5_states_store_checkpoint_log_gap": bool("checkpoint_log10_gap" in states_raw.columns),
        "5_note": "absent in Task 12 states; recovered in task12_1 from stage-1 marks (states_with_checkpoint_gap.parquet)",
        "6_states_store_bg_behavior": bool(any(c.startswith("bg_") for c in states_raw.columns)),
        "7_states_store_bs_behavior": bool(any(c.startswith("bs_") for c in states_raw.columns)),
        "8_practical_best_uses_tied_first": True,
        "8_note": "task12_stage2_analysis.py resolves the tie set with `tied[0]`",
        "8b_tie_set_gt1_share": float(is_tie.mean()),
        "8c_tie_resolved_away_from_raw_argmin_share": float(tie_resolved_away_from_raw.mean()),
        "9_candidate_order_fixed": True,
        "9_note": "CANDIDATES = ('shade', 'lshade', 'cso') defines the tie order",
        "10_oracle_lacks_current_condition": True,
        "10_note": "Task 12 oracles condition on problem / problem+FE only",
        "11_addback_lacks_cmaes_current_states": True,
        "11_note": "cmaes appears only as an action branch, never as current_algorithm",
        "12_repetition_fraction_state_action": float(
            branches.loc[branches["replicate_id"].gt(0), ["state_id", "candidate_action"]]
            .drop_duplicates()
            .shape[0]
        )
        / float(base[["state_id", "candidate_action"]].drop_duplicates().shape[0]),
        "13_continue_branch_is_t_plus_1000_current_outcome": "verified bit-exactly against stage-1 marks (replay_alignment.parquet)",
        "14_horizon_exact_1000_evaluations": "enforced by RuntimeError assert in _run_horizon_branch",
    }


# --------------------------------------------------------------------------
# delta semantics (12.1F)
# --------------------------------------------------------------------------
def addback_noise(addback: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    meta = states.drop_duplicates("state_id")[["state_id", "suite", "cv_group_id"]]
    frame = addback.merge(meta, on="state_id", validate="many_to_one")
    medians = frame.groupby("state_id")["loss_1000"].transform("median")
    frame["deviation"] = (frame["loss_1000"] - medians).abs()
    rows = []
    for suite_name, group in frame.groupby("suite", sort=False):
        per_function = group.groupby("cv_group_id")["deviation"].quantile([0.50, 0.95]).unstack()
        rows.append(
            {
                "suite": suite_name,
                "candidate_action": "cmaes",
                "delta_50_function_balanced": float(per_function[0.50].mean()),
                "delta_95_function_balanced": float(per_function[0.95].mean()),
                "repeated_cells": int(group.loc[group["replicate_id"].gt(0)].groupby("state_id").ngroups),
            }
        )
    return pd.DataFrame(rows)


def direct_pairwise_deltas(branches: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    wide = branches.pivot_table(
        index=["state_id", "candidate_action"], columns="replicate_id", values="loss_1000"
    )
    meta = states.set_index("state_id")[["suite", "current_algorithm"]]
    rows = []
    for suite_name in sorted(states["suite"].unique()):
        suite_states = meta.index[meta["suite"].eq(suite_name)]
        for i, j in combinations(SOLVERS, 2):
            diffs = []
            paired = 0
            for state_id in suite_states:
                current = meta.at[state_id, "current_algorithm"]
                raw_i = "continue" if current == i else i
                raw_j = "continue" if current == j else j
                try:
                    xi = wide.loc[(state_id, raw_i)].dropna()
                    xj = wide.loc[(state_id, raw_j)].dropna()
                except KeyError:
                    continue
                shared = xi.index.intersection(xj.index)
                if len(shared) < 2:
                    continue
                d = (xi.loc[shared] - xj.loc[shared]).to_numpy()
                diffs.extend(d - np.median(d))
                paired += 1
            rows.append(
                {
                    "suite": suite_name,
                    "pair": f"{i}|{j}",
                    "n_paired_states": paired,
                    "estimate": "direct_replicate_matched",
                    "delta_95": float(np.quantile(np.abs(diffs), 0.95)) if diffs else np.nan,
                    "sufficient": paired >= DIRECT_PAIR_MIN_N,
                }
            )
    return pd.DataFrame(rows)


def statewise_pairwise_delta(frame: pd.DataFrame, per_action: pd.Series, solvers) -> pd.DataFrame:
    """delta_ab(s) = max(delta_raw(s,a), delta_raw(s,b)); the raw action behind
    a solver cell is `continue` when the cell equals the current solver, else
    the switch-to-a branch."""
    delta = pd.DataFrame(index=frame.index)
    delta["state_id"] = frame["state_id"].to_numpy()
    for s in solvers:
        raw = np.where(frame["current_algorithm"].eq(s), "continue", s)
        delta[f"delta_cell_{s}"] = [
            float(per_action.loc[(suite, action)]) for suite, action in zip(frame["suite"], raw)
        ]
    return delta


# --------------------------------------------------------------------------
# policies (12.1B/C)
# --------------------------------------------------------------------------
def descriptive_assignment(frame: pd.DataFrame, keys: list[str], solvers) -> pd.Series:
    loss_cols = [f"loss_{s}" for s in solvers]
    per = frame.groupby([*keys, "cv_group_id"], sort=False)[loss_cols].mean().groupby(keys, sort=False).mean()
    choice = per.idxmin(axis=1).str.replace("loss_", "", regex=False)
    assigned = frame.set_index(keys).index.map(choice)
    return pd.Series(np.asarray(assigned), index=frame.index)


def oof_assignment(frame: pd.DataFrame, keys: list[str], solvers) -> pd.Series:
    loss_cols = [f"loss_{s}" for s in solvers]
    assigned = pd.Series(index=frame.index, dtype=object)
    for group in sorted(frame["cv_group_id"].unique()):
        train = frame.loc[frame["cv_group_id"].ne(group)]
        per = train.groupby([*keys, "cv_group_id"], sort=False)[loss_cols].mean().groupby(keys, sort=False).mean()
        choice = per.idxmin(axis=1).str.replace("loss_", "", regex=False)
        hold = frame.index[frame["cv_group_id"].eq(group)]
        assigned.loc[hold] = np.asarray(frame.loc[hold].set_index(keys).index.map(choice))
    if assigned.isna().any():
        raise RuntimeError("OOF policy left states unassigned")
    return assigned


def loso_seed_assignment(frame: pd.DataFrame, keys: list[str], solvers) -> pd.Series:
    loss_cols = [f"loss_{s}" for s in solvers]
    assigned = pd.Series(index=frame.index, dtype=object)
    for _, positions in frame.groupby(keys, sort=False).groups.items():
        idx = pd.Index(positions)
        for pos in idx:
            seed = frame.at[pos, "seed"]
            train = frame.loc[idx.difference(pd.Index([pos]))]
            train = train.loc[train["seed"].ne(seed)]
            assigned.at[pos] = train[loss_cols].mean().idxmin().replace("loss_", "")
    if assigned.isna().any():
        raise RuntimeError("LOSO-seed policy left states unassigned")
    return assigned


def assignment_loss(frame: pd.DataFrame, assignment: pd.Series) -> pd.Series:
    return pd.Series(
        [row[f"loss_{a}"] for (_, row), a in zip(frame.iterrows(), assignment)], index=frame.index
    )


def bootstrap_fb(frame: pd.DataFrame, columns: list[str], stream_offset: int) -> dict[str, tuple[float, float, float]]:
    groups = sorted(frame["cv_group_id"].unique())
    group_means = {col: fb_series(frame, col).to_dict() for col in columns}
    rng = np.random.default_rng(
        np.random.SeedSequence([BOOTSTRAP_STREAM + stream_offset, len(groups), len(columns)]).generate_state(4)
    )
    draws = {col: np.empty(BOOTSTRAP_DRAWS) for col in columns}
    for draw in range(BOOTSTRAP_DRAWS):
        sample = rng.choice(groups, size=len(groups), replace=True)
        for col in columns:
            draws[col][draw] = np.mean([group_means[col][g] for g in sample])
    return {
        col: (
            float(fb_series(frame, col).mean()),
            float(np.quantile(draws[col], 0.025)),
            float(np.quantile(draws[col], 0.975)),
        )
        for col in columns
    }


# --------------------------------------------------------------------------
# set-valued practical actions (12.1G/H)
# --------------------------------------------------------------------------
def set_valued_frame(
    frame: pd.DataFrame,
    solvers,
    semantics: str,
    delta_ctx: pd.DataFrame,
    legacy_delta: float,
    pooled_by_action: dict,
) -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    delta_ctx = delta_ctx.reset_index(drop=True)
    if len(frame) != len(delta_ctx) or not frame["state_id"].eq(delta_ctx["state_id"]).all():
        raise SystemStop("delta context rows do not align with the state frame")
    loss_cols = [f"loss_{s}" for s in solvers]
    values = frame[loss_cols].to_numpy()
    n = len(frame)

    if semantics == "pairwise":
        # zeros keep the diagonal defined: self-dominance L_a < L_a - 0 is false
        delta_matrix = np.zeros((n, len(solvers), len(solvers)))
        for a_index, a in enumerate(solvers):
            for b_index, b in enumerate(solvers):
                if a == b:
                    continue
                delta_matrix[:, a_index, b_index] = np.maximum(
                    delta_ctx[f"delta_cell_{a}"].to_numpy(), delta_ctx[f"delta_cell_{b}"].to_numpy()
                )
    elif semantics == "legacy":
        delta_matrix = np.full((n, len(solvers), len(solvers)), float(legacy_delta))
    else:  # pooled action-pair sensitivity
        delta_matrix = np.zeros((n, len(solvers), len(solvers)))
        for a_index, a in enumerate(solvers):
            for b_index, b in enumerate(solvers):
                if a == b:
                    continue
                raw_a = np.where(frame["current_algorithm"].eq(a), "continue", a)
                raw_b = np.where(frame["current_algorithm"].eq(b), "continue", b)
                delta_matrix[:, a_index, b_index] = [
                    max(pooled_by_action[x], pooled_by_action[y]) for x, y in zip(raw_a, raw_b)
                ]

    dominates = values[:, :, None] < values[:, None, :] - delta_matrix  # a strictly better than b beyond delta
    dominated = dominates.any(axis=1)
    non_dominated = ~dominated

    set_size = non_dominated.sum(axis=1)
    in_nd = np.array([non_dominated[row, solvers.index(c)] for row, c in enumerate(frame["current_algorithm"])])
    switch_required = ~in_nd

    raw_argmin = values.argmin(axis=1)
    targets, op_actions = [], []
    for row in range(n):
        candidates = np.where(non_dominated[row])[0]
        if len(candidates):
            best = candidates[np.argmin(values[row, candidates])]
        else:
            # intransitive pairwise dominance can leave the practical set empty;
            # the operational summary then falls back to the raw best observed
            # action and the state is flagged separately
            best = int(raw_argmin[row])
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


# --------------------------------------------------------------------------
# progress gate (12.1J-P)
# --------------------------------------------------------------------------
def auroc_ap_bootstrap(frame: pd.DataFrame, score_column: str, label_column: str) -> dict:
    pooled_auc = float(roc_auc_score(frame[label_column], frame[score_column]))
    pooled_ap = float(average_precision_score(frame[label_column], frame[score_column]))
    groups = sorted(frame["cv_group_id"].unique())
    rng = np.random.default_rng(
        np.random.SeedSequence([BOOTSTRAP_STREAM + 50, len(groups)]).generate_state(4)
    )
    aucs, aps = [], []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = rng.choice(groups, size=len(groups), replace=True)
        resampled = pd.concat(
            [frame.loc[frame["cv_group_id"].eq(g)] for g in sample], ignore_index=True
        )
        if resampled[label_column].nunique() < 2:
            continue
        aucs.append(roc_auc_score(resampled[label_column], resampled[score_column]))
        aps.append(average_precision_score(resampled[label_column], resampled[score_column]))
    return {
        "auc_pooled": pooled_auc,
        "auc_ci_low": float(np.quantile(aucs, 0.025)),
        "auc_ci_high": float(np.quantile(aucs, 0.975)),
        "ap_pooled": pooled_ap,
        "ap_ci_low": float(np.quantile(aps, 0.025)),
        "ap_ci_high": float(np.quantile(aps, 0.975)),
        "base_rate": float(frame[label_column].mean()),
        "n": int(len(frame)),
        "n_positives": int(frame[label_column].sum()),
    }


def trigger_sensitivity(frame: pd.DataFrame, stratification: str) -> pd.DataFrame:
    rows = []
    for suite_name, suite_group in frame.groupby("suite", sort=False):
        base_rate = float(suite_group["switch_required"].mean())
        z = suite_group["switch_required"].to_numpy(dtype=bool)
        for q in TRIGGER_QUANTILES:
            if stratification == "suite_global":
                threshold = suite_group["progress_R"].quantile(q)
                triggered = (suite_group["progress_R"] <= threshold).to_numpy()
            else:
                threshold = suite_group.groupby(["current_algorithm", "FE"])["progress_R"].transform(
                    lambda s: s.quantile(q)
                )
                triggered = (suite_group["progress_R"] <= threshold).to_numpy()
            rho = float(triggered.mean())
            precision = float(z[triggered].mean()) if triggered.any() else np.nan
            rows.append(
                {
                    "suite": suite_name,
                    "stratification": stratification,
                    "q": q,
                    "trigger_rate": rho,
                    "switch_recall": float(z[triggered].sum() / max(z.sum(), 1)),
                    "switch_precision": precision,
                    "enrichment": precision / base_rate if precision == precision else np.nan,
                    "missed_switch_rate": float((z & ~triggered).sum() / max(z.sum(), 1)),
                    "action_call_reduction": 1.0 - rho,
                    "cross_action_branch_reduction": 1.0 - rho,
                    "P_Z_given_high_progress": float(z[~triggered].mean()),
                    "E_G_practical_high_progress": float(suite_group.loc[~triggered, "G_practical"].mean()),
                    "base_rate": base_rate,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    HEAVY.mkdir(parents=True, exist_ok=True)
    LIGHT.mkdir(parents=True, exist_ok=True)
    started_total = perf_counter()

    states_raw = pd.read_parquet(TASK12_HEAVY / "dynamic_screening_states.parquet")
    base = pd.read_parquet(TASK12_HEAVY / "dynamic_action_outcomes_1000.parquet")
    branches = load_shard("branches")
    addback = load_shard("addback")
    solver = pd.read_parquet(TASK12_HEAVY / "dynamic_solver_loss_matrix.parquet")
    noise_task12 = pd.read_parquet(TASK12_HEAVY / "dynamic_noise_deltas.parquet")
    stage1 = load_stage1_natural_marks()

    # ---- 12.1R alignment + checkpoint gap recovery (zero FE) ----
    alignment = verify_alignment(states_raw, base, stage1)
    alignment.to_parquet(HEAVY / "replay_alignment.parquet", index=False)
    alignment.to_parquet(LIGHT / "replay_alignment.parquet", index=False)
    (LIGHT / "alignment_summary.json").write_text(
        json.dumps(
            {
                "replay_executed": False,
                "max_terminal_abs_diff": float(alignment["terminal_abs_diff"].max()),
                "max_continue_abs_diff": float(alignment["continue_abs_diff"].max()),
                "n_states_checked": int(len(alignment)),
                "checkpoint_gap_source": "task12 stage1 natural marks (replicate 0)",
            },
            indent=2,
        )
    )
    marks = stage1.loc[stage1["FE"].isin(CHECKPOINT_FES)][["problem_id", "seed", "candidate", "FE", "log10_gap"]]
    gap = states_raw[["state_id", "problem_id", "seed", "current_algorithm", "FE"]].merge(
        marks,
        left_on=["problem_id", "seed", "current_algorithm", "FE"],
        right_on=["problem_id", "seed", "candidate", "FE"],
        how="left",
        validate="many_to_one",
    ).rename(columns={"log10_gap": "checkpoint_log10_gap"})
    if gap["checkpoint_log10_gap"].isna().any():
        raise SystemStop("checkpoint gap recovery incomplete")
    states = states_raw.merge(gap[["state_id", "checkpoint_log10_gap"]], on="state_id", validate="one_to_one")
    states.to_parquet(HEAVY / "states_with_checkpoint_gap.parquet", index=False)

    # ---- solver frame in states row order ----
    frame3 = solver.merge(
        states[["state_id", "family", "instance", "checkpoint_log10_gap"]], on="state_id", validate="many_to_one"
    )
    if not frame3["state_id"].tolist() == states["state_id"].tolist():
        frame3 = states[["state_id"]].merge(frame3, on="state_id", validate="one_to_one")
    loss_cols3 = [f"loss_{s}" for s in SOLVERS]
    cont_loss = base.loc[base["candidate_action"].eq("continue")].set_index("state_id")["loss_1000"]
    switch_loss = base.loc[base["candidate_action"].ne("continue")].pivot_table(
        index="state_id", columns="candidate_action", values="loss_1000", aggfunc="first"
    )
    frame3["continue_loss"] = cont_loss.reindex(frame3["state_id"]).to_numpy()
    for s in SOLVERS:
        frame3[f"switch_{s}"] = switch_loss[s].reindex(frame3["state_id"]).to_numpy()

    # ---- delta semantics (12.1F) ----
    per_action = noise_task12.set_index(["suite", "candidate_action"])["delta_95_function_balanced"]
    suite_legacy = noise_task12.groupby("suite")["delta_95_function_balanced"].mean().to_dict()
    pooled_by_action = noise_task12.set_index("candidate_action")["pooled_delta_95"].to_dict()
    delta_ctx3 = statewise_pairwise_delta(frame3, per_action, SOLVERS)

    # ---- 12.1A checklist (on the original Task 12 tables) ----
    checklist = semantics_checklist(states_raw, base, branches, solver, suite_legacy)
    (LIGHT / "12a_data_semantics_checklist.json").write_text(
        json.dumps(checklist, indent=2, ensure_ascii=False, default=float)
    )

    direct = direct_pairwise_deltas(branches, states)
    cmaes_noise = addback_noise(addback, states)
    cmaes_delta_map = dict(zip(cmaes_noise["suite"], cmaes_noise["delta_95_function_balanced"]))
    pairwise_out = direct.copy()
    conservative = []
    for row in pairwise_out.itertuples(index=False):
        i, j = row.pair.split("|")
        suite_group = states.loc[states["suite"].eq(row.suite), "current_algorithm"]
        d_i = max(
            per_action.loc[(row.suite, "continue")], per_action.loc[(row.suite, i)]
        )
        d_j = max(
            per_action.loc[(row.suite, "continue")], per_action.loc[(row.suite, j)]
        )
        conservative.append(float(max(d_i, d_j)))
    pairwise_out["fallback_conservative_delta_95"] = conservative
    pairwise_out.to_parquet(LIGHT / "pairwise_noise_deltas.parquet", index=False)
    cmaes_noise.to_parquet(LIGHT / "cmaes_action_noise.parquet", index=False)

    # ---- 12.1B/C/D ladder + OOF + bootstrap ----
    ladder_rows, boot_rows = [], []
    for suite_index, (suite_name, group) in enumerate(frame3.groupby("suite", sort=False)):
        group = group.copy().reset_index(drop=True)
        policies = {
            "L_current_desc": descriptive_assignment(group, ["current_algorithm"], SOLVERS),
            "L_current_FE_desc": descriptive_assignment(group, ["current_algorithm", "FE"], SOLVERS),
            "L_problem_desc": descriptive_assignment(group, ["problem_id"], SOLVERS),
            "L_problem_FE_desc": descriptive_assignment(group, ["problem_id", "FE"], SOLVERS),
            "L_problem_current_FE_desc": descriptive_assignment(
                group, ["problem_id", "current_algorithm", "FE"], SOLVERS
            ),
            "L_problem_current_FE_loso_seed": loso_seed_assignment(
                group, ["problem_id", "current_algorithm", "FE"], SOLVERS
            ),
            "L_current_oof": oof_assignment(group, ["current_algorithm"], SOLVERS),
            "L_current_FE_oof": oof_assignment(group, ["current_algorithm", "FE"], SOLVERS),
        }
        policy_columns = list(policies)
        for name, assignment in policies.items():
            group[name] = assignment_loss(group, assignment)
        sbs_means = group.groupby("cv_group_id")[loss_cols3].mean().mean()
        sbs_solver = sbs_means.idxmin().replace("loss_", "")
        group["L_SBS"] = group[f"loss_{sbs_solver}"]
        group["L_statewise"] = group[loss_cols3].min(axis=1)
        ladder = {name: fb_mean(group, name) for name in ["L_SBS", *policy_columns, "L_statewise"]}
        ladder_rows.append(
            {
                "suite": suite_name,
                "SBS_solver": sbs_solver,
                **{k: float(v) for k, v in ladder.items()},
                "delta_portfolio": ladder["L_SBS"] - ladder["L_statewise"],
                "delta_deploy_residual": ladder["L_current_FE_oof"] - ladder["L_statewise"],
                "delta_context_residual": ladder["L_problem_current_FE_desc"] - ladder["L_statewise"],
                "delta_problem_info": ladder["L_current_FE_oof"] - ladder["L_problem_current_FE_loso_seed"],
                "delta_dynamic_old": ladder["L_problem_FE_desc"] - ladder["L_statewise"],
                "delta_problem_old": ladder["L_problem_desc"] - ladder["L_statewise"],
                "delta_current_FE_desc": ladder["L_current_FE_desc"] - ladder["L_statewise"],
                "delta_95_legacy_suite": suite_legacy[suite_name],
            }
        )
        boot = bootstrap_fb(group, ["L_SBS", *policy_columns, "L_statewise"], stream_offset=suite_index * 10)
        for quantity, (mean, low, high) in boot.items():
            boot_rows.append(
                {"suite": suite_name, "quantity": quantity, "mean": mean, "ci_low": low, "ci_high": high}
            )
        for stream, (name, upper, lower) in enumerate(
            (
                ("delta_portfolio", "L_SBS", "L_statewise"),
                ("delta_deploy_residual", "L_current_FE_oof", "L_statewise"),
                ("delta_context_residual", "L_problem_current_FE_desc", "L_statewise"),
                ("delta_problem_info", "L_current_FE_oof", "L_problem_current_FE_loso_seed"),
                ("delta_dynamic_old", "L_problem_FE_desc", "L_statewise"),
            )
        ):
            paired = _paired_group_bootstrap(
                group, upper, lower, stream_offset=100 + suite_index * 10 + stream
            )
            boot_rows.append(
                {
                    "suite": suite_name,
                    "quantity": name,
                    "mean": float(fb_mean(group, upper) - fb_mean(group, lower)),
                    "ci_low": float(np.quantile(paired, 0.025)),
                    "ci_high": float(np.quantile(paired, 0.975)),
                }
            )
    ladder_table = pd.DataFrame(ladder_rows)
    ladder_table.to_parquet(LIGHT / "oracle_ladder_current_conditioned.parquet", index=False)
    boot_table = pd.DataFrame(boot_rows)
    boot_table.to_parquet(LIGHT / "oracle_headroom_current_conditioned_bootstrap.parquet", index=False)

    # ---- 12.1E winner's-curse diagnostic on repeated argmin cells ----
    cell_median = branches.groupby(["state_id", "candidate_action"])["loss_1000"].median()
    cell_replicates = branches.groupby(["state_id", "candidate_action"])["replicate_id"].nunique()
    repeated_mask = cell_replicates.ge(2)
    cell_r0 = base.set_index(["state_id", "candidate_action"])["loss_1000"]
    states_idx = states.set_index("state_id")
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
        if bool(repeated_mask.loc[(state_id, cells[best])]):
            meta = states_idx.loc[state_id]
            rows.append(
                {
                    "suite": meta["suite"],
                    "current_algorithm": meta["current_algorithm"],
                    "FE": int(meta["FE"]),
                    "cv_group_id": meta["cv_group_id"],
                    "bias": float(cell_median.loc[(state_id, cells[best])] - losses0[best]),
                }
            )
    inflation = pd.DataFrame(rows)

    inflation_rows = []
    for suite_name, group in inflation.groupby("suite", sort=False):
        inflation_rows.append(
            {
                "suite": suite_name,
                "stratum": "overall",
                "current": "all",
                "FE": 0,
                "n": len(group),
                "bias_fb": fb_mean(group, "bias"),
                "bias_pooled_mean": float(group["bias"].mean()),
            }
        )
        for (current, fe), part in group.groupby(["current_algorithm", "FE"]):
            inflation_rows.append(
                {
                    "suite": suite_name,
                    "stratum": "current_x_FE",
                    "current": current,
                    "FE": int(fe),
                    "n": len(part),
                    "bias_fb": fb_mean(part, "bias"),
                    "bias_pooled_mean": float(part["bias"].mean()),
                }
            )
    inflation_table = pd.DataFrame(inflation_rows)
    inflation_table["coverage_note"] = np.where(
        inflation_table["n"] < DIRECT_PAIR_MIN_N, "INSUFFICIENT_COVERAGE_diagnostic_only", "ok"
    )
    inflation_table.to_parquet(LIGHT / "oracle_inflation_repeated_subset.parquet", index=False)

    # ---- 12.1G/H set-valued action space ----
    sets_frames = {}
    for semantics in ("pairwise", "legacy", "pooled"):
        parts = []
        for suite_name, group in frame3.groupby("suite", sort=False):
            group = group.reset_index(drop=True)
            ctx = delta_ctx3.set_index("state_id").loc[group["state_id"]].reset_index()
            parts.append(
                set_valued_frame(
                    group, SOLVERS, semantics, ctx, suite_legacy[suite_name], pooled_by_action
                )
            )
        sets_frames[semantics] = pd.concat(parts, ignore_index=True)
    sv_primary = sets_frames["pairwise"]
    sv_primary.to_parquet(HEAVY / "practical_action_sets.parquet", index=False)
    sv_primary.to_parquet(LIGHT / "practical_action_sets.parquet", index=False)

    action_metric_rows = []
    for semantics, sv in sets_frames.items():
        for suite_name, group in sv.groupby("suite", sort=False):
            action_metric_rows.append(
                {
                    "semantics": semantics,
                    "suite": suite_name,
                    "P_current_in_A_ND": float(group["current_in_A_ND"].mean()),
                    "P_A_ND_size_1": float(group["A_ND_size"].eq(1).mean()),
                    "P_A_ND_size_gt1": float(group["A_ND_size"].gt(1).mean()),
                    "P_A_ND_empty": float(group["A_ND_empty"].mean()),
                    "E_A_ND_size": float(group["A_ND_size"].mean()),
                    "switch_required_rate": float(group["switch_required"].mean()),
                    "optional_switch_rate": float(group["optional_switch"].mean()),
                    "H_operational": entropy_of(group["operational_action"].value_counts()),
                    "H_Z": entropy_of(group["switch_required"].astype(int).value_counts()),
                    "H_Z_given_current": conditional_entropy(group, ["current_algorithm"], "switch_required"),
                    "H_Z_given_current_FE": conditional_entropy(
                        group, ["current_algorithm", "FE"], "switch_required"
                    ),
                    "H_Z_given_problem_current_FE": conditional_entropy(
                        group, ["problem_id", "current_algorithm", "FE"], "switch_required"
                    ),
                }
            )
    action_metrics = pd.DataFrame(action_metric_rows)
    action_metrics.to_parquet(LIGHT / "switch_required_distribution.parquet", index=False)

    trans_rows = []
    for suite_name, group in sv_primary.groupby("suite", sort=False):
        for current, part in group.groupby("current_algorithm", sort=False):
            row = {"suite": suite_name, "current": current, "n": int(len(part))}
            row["P_continue"] = float((~part["switch_required"]).mean())
            required = part.loc[part["switch_required"]]
            for s in SOLVERS:
                if s == current:
                    continue
                row[f"P_target_{s}_given_switch"] = (
                    float(required["switch_target"].eq(s).mean()) if len(required) else np.nan
                )
                row[f"P_target_{s}_joint"] = float(part["switch_target"].eq(s).mean())
            trans_rows.append(row)
    transition = pd.DataFrame(trans_rows)
    transition.to_parquet(LIGHT / "current_preserving_transition_matrix.parquet", index=False)

    strata_rows = []
    for suite_name, group in sv_primary.groupby("suite", sort=False):
        for stratum, keys in (("FE", ["FE"]), ("current", ["current_algorithm"]), ("family", ["family"])):
            for key, part in group.groupby(keys, sort=False):
                if not isinstance(key, tuple):
                    key = (key,)
                strata_rows.append(
                    {
                        "suite": suite_name,
                        "stratum": stratum,
                        "key": "|".join(str(k) for k in key),
                        "n": len(part),
                        "switch_required_rate": float(part["switch_required"].mean()),
                        "P_current_in_A_ND": float(part["current_in_A_ND"].mean()),
                    }
                )
    pd.DataFrame(strata_rows).to_parquet(HEAVY / "switch_required_strata.parquet", index=False)

    variation_rows = []
    for (suite_name, problem_id, seed, current), part in sv_primary.groupby(
        ["suite", "problem_id", "seed", "current_algorithm"], sort=False
    ):
        part = part.sort_values("FE")
        required_targets = part.loc[part["switch_required"], "switch_target"]
        variation_rows.append(
            {
                "suite": suite_name,
                "problem_id": problem_id,
                "seed": seed,
                "current": current,
                "switch_status_varies": int(part["switch_required"].nunique() >= 2),
                "target_varies_within_switch_required": (
                    int(required_targets.nunique() >= 2) if len(required_targets) >= 2 else np.nan
                ),
            }
        )
    variation = pd.DataFrame(variation_rows)
    variation_summary = (
        variation.groupby("suite")
        .agg(
            trajectories=("switch_status_varies", "size"),
            P_switch_status_varies=("switch_status_varies", "mean"),
            P_target_varies=("target_varies_within_switch_required", "mean"),
        )
        .reset_index()
    )
    variation_summary.to_parquet(LIGHT / "within_trajectory_switch_variation.parquet", index=False)

    # ---- 12.1I DCM robustness ----
    dcm_rows = []
    for semantics, sv in sets_frames.items():
        for suite_name, group in sv.groupby("suite", sort=False):
            for a, b in combinations(SOLVERS, 2):
                la = group[f"loss_{a}"].to_numpy()
                lb = group[f"loss_{b}"].to_numpy()
                if semantics == "pairwise":
                    ctx = delta_ctx3.set_index("state_id").loc[group["state_id"]]
                    d = np.maximum(ctx[f"delta_cell_{a}"].to_numpy(), ctx[f"delta_cell_{b}"].to_numpy())
                elif semantics == "legacy":
                    d = np.full(len(group), suite_legacy[suite_name])
                else:
                    raw_a = np.where(group["current_algorithm"].eq(a), "continue", a)
                    raw_b = np.where(group["current_algorithm"].eq(b), "continue", b)
                    d = np.array([max(pooled_by_action[x], pooled_by_action[y]) for x, y in zip(raw_a, raw_b)])
                tie = float((np.abs(la - lb) <= d).mean())
                a_better = float((la < lb - d).mean())
                b_better = float((lb < la - d).mean())
                dcm_rows.append(
                    {
                        "semantics": semantics,
                        "suite": suite_name,
                        "candidate_a": a,
                        "candidate_b": b,
                        "P_tie": tie,
                        "P_a_better": a_better,
                        "P_b_better": b_better,
                        "DCM": float((tie + abs(a_better - b_better)) / 2.0),
                    }
                )
    dcm = pd.DataFrame(dcm_rows)
    dcm.to_parquet(LIGHT / "dynamic_dcm_robustness.parquet", index=False)

    # ---- 12.1J-P progress gate ----
    prog = sv_primary[
        ["state_id", "switch_required", "suite", "cv_group_id", "current_algorithm", "FE"]
    ].copy()
    fx = frame3.set_index("state_id")
    prog["checkpoint_log10_gap"] = states_idx.loc[prog["state_id"], "checkpoint_log10_gap"].to_numpy()
    prog["continue_loss"] = fx.loc[prog["state_id"], "continue_loss"].to_numpy()
    switch_cols = [f"switch_{s}" for s in SOLVERS]
    switch_losses = fx.loc[prog["state_id"], switch_cols]
    prog["G_best_switch"] = prog["continue_loss"].to_numpy() - switch_losses.min(axis=1).to_numpy()
    prog["best_switch_solver"] = (
        switch_losses.idxmin(axis=1).str.replace("switch_", "").to_numpy()
    )
    ctx_prog = delta_ctx3.set_index("state_id").loc[prog["state_id"]].reset_index(drop=True)
    pair_delta = np.array(
        [
            max(row[f"delta_cell_{c}"], row[f"delta_cell_{t}"])
            for row, c, t in zip(
                ctx_prog.to_dict("records"),
                prog["current_algorithm"],
                prog["best_switch_solver"],
            )
        ]
    )
    prog["G_practical"] = np.maximum(prog["G_best_switch"].to_numpy() - pair_delta, 0.0)
    prog["progress_delta"] = prog["checkpoint_log10_gap"] - prog["continue_loss"]
    prog["progress_R"] = 10.0 * prog["progress_delta"]
    prog["score_neg_progress"] = -prog["progress_R"]
    prog.to_parquet(HEAVY / "realized_progress_1000.parquet", index=False)

    assoc_rows = []
    for suite_name, group in prog.groupby("suite", sort=False):
        stats = auroc_ap_bootstrap(group, "score_neg_progress", "switch_required")
        stats.update(
            {
                "suite": suite_name,
                "stratum": "overall",
                "spearman_negR_G_best_switch": float(
                    spearmanr(group["score_neg_progress"], group["G_best_switch"]).statistic
                ),
                "spearman_negR_G_practical": float(
                    spearmanr(group["score_neg_progress"], group["G_practical"]).statistic
                ),
            }
        )
        assoc_rows.append(stats)
        for current, part in group.groupby("current_algorithm", sort=False):
            if part["switch_required"].nunique() < 2:
                continue
            assoc_rows.append(
                {
                    "suite": suite_name,
                    "stratum": f"current={current}",
                    "auc_pooled": float(roc_auc_score(part["switch_required"], part["score_neg_progress"])),
                    "ap_pooled": float(
                        average_precision_score(part["switch_required"], part["score_neg_progress"])
                    ),
                    "base_rate": float(part["switch_required"].mean()),
                    "n": len(part),
                    "n_positives": int(part["switch_required"].sum()),
                    "spearman_negR_G_best_switch": float(
                        spearmanr(part["score_neg_progress"], part["G_best_switch"]).statistic
                    ),
                }
            )
        for fe, part in group.groupby("FE", sort=False):
            if part["switch_required"].nunique() < 2:
                continue
            assoc_rows.append(
                {
                    "suite": suite_name,
                    "stratum": f"FE={int(fe)}",
                    "auc_pooled": float(roc_auc_score(part["switch_required"], part["score_neg_progress"])),
                    "ap_pooled": float(
                        average_precision_score(part["switch_required"], part["score_neg_progress"])
                    ),
                    "base_rate": float(part["switch_required"].mean()),
                    "n": len(part),
                    "n_positives": int(part["switch_required"].sum()),
                    "spearman_negR_G_best_switch": float(
                        spearmanr(part["score_neg_progress"], part["G_best_switch"]).statistic
                    ),
                }
            )
    association = pd.DataFrame(assoc_rows)
    association.to_parquet(LIGHT / "progress_switch_association.parquet", index=False)

    trigger_tables = pd.concat(
        [trigger_sensitivity(prog, "suite_global"), trigger_sensitivity(prog, "current_FE_stratified")],
        ignore_index=True,
    )
    trigger_tables.to_parquet(LIGHT / "progress_trigger_quantile_sensitivity.parquet", index=False)

    cont_reps = branches.loc[branches["candidate_action"].eq("continue")]
    noise_rows = []
    for state_id, part in cont_reps.groupby("state_id", sort=False):
        if part["replicate_id"].nunique() < 3:
            continue
        meta = states_idx.loc[state_id]
        r_values = 10.0 * (meta["checkpoint_log10_gap"] - part.sort_values("replicate_id")["loss_1000"].to_numpy())
        noise_rows.append(
            {
                "suite": meta["suite"],
                "current_algorithm": meta["current_algorithm"],
                "cv_group_id": meta["cv_group_id"],
                "deviation": float(np.abs(r_values - np.median(r_values)).max()),
            }
        )
    progress_noise_rows = []
    noise_frame = pd.DataFrame(noise_rows)
    for (suite_name, current), part in noise_frame.groupby(["suite", "current_algorithm"], sort=False):
        per_function = part.groupby("cv_group_id")["deviation"].quantile([0.50, 0.95]).unstack()
        progress_noise_rows.append(
            {
                "suite": suite_name,
                "current": current,
                "n_states": len(part),
                "delta_progress_50_fb": float(per_function[0.50].mean()),
                "delta_progress_95_fb": float(per_function[0.95].mean()),
            }
        )
    progress_noise = pd.DataFrame(progress_noise_rows)
    progress_noise["status"] = "diagnostic_only_not_a_final_threshold"
    progress_noise.to_parquet(LIGHT / "progress_noise_diagnostic.parquet", index=False)

    # ---- 12.1S add-back robustness (4 actions, one-step) ----
    cmaes_loss = addback.loc[addback["replicate_id"].eq(0)].set_index("state_id")["loss_1000"]
    frame4 = frame3.copy()
    frame4["loss_cmaes"] = cmaes_loss.reindex(frame4["state_id"]).to_numpy()
    if frame4["loss_cmaes"].isna().any():
        raise SystemStop("add-back branches do not cover every state")
    delta_ctx4 = statewise_pairwise_delta(frame4, per_action, SOLVERS)
    delta_ctx4["delta_cell_cmaes"] = frame4["suite"].map(cmaes_delta_map).to_numpy()
    if delta_ctx4["delta_cell_cmaes"].isna().any():
        raise SystemStop("cmaes action delta unavailable for a suite")

    addback_rows = []
    loss_cols4 = [f"loss_{s}" for s in SOLVERS4]
    for suite_index, (suite_name, group) in enumerate(frame4.groupby("suite", sort=False)):
        group = group.copy().reset_index(drop=True)
        ctx = delta_ctx4.set_index("state_id").loc[group["state_id"]].reset_index()
        sv4 = set_valued_frame(group, SOLVERS4, "pairwise", ctx, np.nan, pooled_by_action)
        g = sv4.copy()
        g["L_current_FE_oof4"] = assignment_loss(g, oof_assignment(g, ["current_algorithm", "FE"], SOLVERS4))
        g["L_problem_current_FE_desc4"] = assignment_loss(
            g, descriptive_assignment(g, ["problem_id", "current_algorithm", "FE"], SOLVERS4)
        )
        g["L_statewise4"] = g[loss_cols4].min(axis=1)
        sbs_means4 = g.groupby("cv_group_id")[loss_cols4].mean().mean()
        sbs4 = sbs_means4.idxmin().replace("loss_", "")
        g["L_SBS4"] = g[f"loss_{sbs4}"]
        addback_rows.append(
            {
                "suite": suite_name,
                "SBS_4": sbs4,
                "L_SBS_4": fb_mean(g, "L_SBS4"),
                "L_current_FE_oof_4": fb_mean(g, "L_current_FE_oof4"),
                "L_problem_current_FE_desc_4": fb_mean(g, "L_problem_current_FE_desc4"),
                "L_statewise_4": fb_mean(g, "L_statewise4"),
                "delta_deploy_residual_4": fb_mean(g, "L_current_FE_oof4") - fb_mean(g, "L_statewise4"),
                "delta_context_residual_4": fb_mean(g, "L_problem_current_FE_desc4")
                - fb_mean(g, "L_statewise4"),
                "P_cmaes_in_A_ND": float(g["A_ND_members"].str.contains("cmaes").mean()),
                "P_A_ND_singleton_cmaes": float(g["A_ND_members"].eq("cmaes").mean()),
                "P_A_ND_empty": float(g["A_ND_empty"].mean()),
                "switch_required_rate_4": float(g["switch_required"].mean()),
                "P_continue_4": float((~g["switch_required"]).mean()),
                "H_operational_4": entropy_of(g["operational_action"].value_counts()),
                "E_A_ND_size_4": float(g["A_ND_size"].mean()),
                "cmaes_delta_95": cmaes_delta_map[suite_name],
                "switch_required_rate_3": float(
                    sv_primary.loc[sv_primary["suite"].eq(suite_name), "switch_required"].mean()
                ),
            }
        )
    addback_table = pd.DataFrame(addback_rows)
    base_residual = ladder_table.set_index("suite")["delta_deploy_residual"]
    addback_table["delta_deploy_residual_3"] = addback_table["suite"].map(base_residual)
    addback_table["residual_ratio_4_over_3"] = (
        addback_table["delta_deploy_residual_4"] / addback_table["delta_deploy_residual_3"]
    )

    def collapse_verdict(row) -> str:
        if row["P_A_ND_singleton_cmaes"] >= 0.85 or row["residual_ratio_4_over_3"] <= 0.2:
            return "ONE-STEP ADD-BACK: STRONG COLLAPSE"
        if row["P_A_ND_singleton_cmaes"] >= 0.55 or row["residual_ratio_4_over_3"] <= 0.6:
            return "ONE-STEP ADD-BACK: PARTIAL COLLAPSE"
        return "ONE-STEP ADD-BACK: NO COLLAPSE"

    addback_table["verdict"] = addback_table.apply(collapse_verdict, axis=1)
    addback_table.to_parquet(LIGHT / "cmaes_addback_robustness.parquet", index=False)

    # ---- resource ledger ----
    ledger_rows = [
        {
            "phase": "task12_1_zero_fe_reanalysis",
            "new_action_label_fe": 0,
            "natural_replay_fe": 0,
            "replay_executed": False,
            "wall_seconds": perf_counter() - started_total,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "note": "checkpoint gaps recovered from stage-1 natural marks after bit-exact alignment check",
        }
    ]
    pd.DataFrame(ledger_rows).to_parquet(HEAVY / "task12_1_resource_ledger.parquet", index=False)

    # ---- console summary ----
    with pd.option_context("display.width", 220, "display.max_columns", 50):
        print("=== ladder ===")
        print(ladder_table.round(4).to_string())
        print("=== bootstrap (deltas) ===")
        print(boot_table.loc[boot_table["quantity"].str.startswith("delta")].round(4).to_string())
        print("=== oracle inflation ===")
        print(inflation_table.head(10).round(4).to_string())
        print("=== action metrics ===")
        print(action_metrics.round(4).to_string())
        print("=== transition ===")
        print(transition.round(4).to_string())
        print("=== dcm ===")
        print(dcm.round(4).to_string())
        print("=== association ===")
        print(association.round(4).to_string())
        print("=== trigger ===")
        print(trigger_tables.round(4).to_string())
        print("=== progress noise ===")
        print(progress_noise.round(4).to_string())
        print("=== addback ===")
        print(addback_table.round(4).to_string())
        print("=== within-trajectory variation ===")
        print(variation_summary.round(4).to_string())
        print("=== cmaes noise ===")
        print(cmaes_noise.round(4).to_string())


def _paired_group_bootstrap(
    frame: pd.DataFrame, upper: str, lower: str, stream_offset: int = 0
) -> np.ndarray:
    groups = sorted(frame["cv_group_id"].unique())
    upper_means = fb_series(frame, upper).to_dict()
    lower_means = fb_series(frame, lower).to_dict()
    rng = np.random.default_rng(
        np.random.SeedSequence([BOOTSTRAP_STREAM + 200 + stream_offset, len(groups)]).generate_state(4)
    )
    deltas = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        sample = rng.choice(groups, size=len(groups), replace=True)
        deltas[draw] = np.mean([upper_means[g] for g in sample]) - np.mean([lower_means[g] for g in sample])
    return deltas


if __name__ == "__main__":
    try:
        main()
    except SystemStop as stop:
        print(f"[task12_1] STOP: {stop}")
        raise
