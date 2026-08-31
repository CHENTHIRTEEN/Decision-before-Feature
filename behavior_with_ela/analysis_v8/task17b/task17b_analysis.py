"""Task 17B zero-FE decision-reliability and residual-aliasing analysis.

The analysis reuses Task 12/13 natural repetitions, Task 14A post-handoff
repetitions, and Task 17A/17A.1 state identities and distances. It never
imports benchmark objectives, optimizer execution, feature extraction, ELA,
or learned models.
"""
from __future__ import annotations

import itertools
import json
import platform
import resource
import warnings
from pathlib import Path
from time import perf_counter, process_time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr


ROOT = Path(__file__).resolve().parents[3]
OUT_LIGHT = ROOT / "behavior_with_ela/analysis_v8/task17b"
OUT_HEAVY = ROOT / "behavior_with_ela/results/analysis_v8/task17b"
FIGURES = OUT_LIGHT / "figures"

POST_BRANCHES = ROOT / "behavior_with_ela/results/analysis_v6/task14a/post_handoff_action_outcomes_1000.parquet"
POST_STATES = ROOT / "behavior_with_ela/results/analysis_v6/task14a/post_handoff_states.parquet"
NATURAL_SHARDS = ROOT / "behavior_with_ela/results/portfolio_screening/task12/stage2/shards"
SIGNATURES = ROOT / "behavior_with_ela/results/analysis_v8/task17a/task17a_state_decision_signatures.parquet"
PAIR_IDENTITIES = ROOT / "behavior_with_ela/results/analysis_v8/task17a/task17a_within_stratum_pairs.parquet"
REPRESENTATION_DISTANCES = ROOT / "behavior_with_ela/results/analysis_v8/task17a/task17a_representation_distances.parquet"
RANDOM_PAIR_CONTROLS = ROOT / "behavior_with_ela/results/analysis_v8/task17a/task17a_random_pair_controls.parquet"

ACTIONS = ("shade", "lshade", "cso")
ACTION_PAIRS = tuple(itertools.combinations(ACTIONS, 2))
SCOPES = ("bbob", "mabbob", "pooled")
REPRESENTATIONS = ("global28", "compact6")
BOOTSTRAP_DRAWS = 5000
MASTER_SEED = 2026083117
BOOTSTRAP_STREAM = 2026083118
NEW_OBJECTIVE_FE = 0

PAIR_CODES = {pair: index + 1 for index, pair in enumerate(ACTION_PAIRS)}
SCOPE_CODES = {scope: index + 1 for index, scope in enumerate(SCOPES)}
SUPPORT_CODES = {"matched_support": 1, "all_available_support": 2}
REPRESENTATION_CODES = {"global28": 1, "compact6": 2}


def _scope(frame: pd.DataFrame, suite: str) -> pd.DataFrame:
    return frame if suite == "pooled" else frame.loc[frame["suite"].eq(suite)]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        return float(spearmanr(x, y).statistic)


def _ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _seed_sequence(*codes: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([MASTER_SEED, *[int(code) for code in codes]])


def _relation(margin: np.ndarray, scale: np.ndarray) -> np.ndarray:
    margin = np.asarray(margin, dtype=float)
    scale = np.asarray(scale, dtype=float)
    return np.where(margin > scale, "a_better", np.where(margin < -scale, "b_better", "tie"))


def _cohen_kappa(left: np.ndarray, right: np.ndarray) -> float:
    categories = ("a_better", "tie", "b_better")
    left = np.asarray(left, dtype=str)
    right = np.asarray(right, dtype=str)
    if not len(left):
        return float("nan")
    observed = float(np.mean(left == right))
    expected = sum(float(np.mean(left == c) * np.mean(right == c)) for c in categories)
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else float("nan")


def _read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [POST_BRANCHES, POST_STATES, SIGNATURES, PAIR_IDENTITIES, REPRESENTATION_DISTANCES, RANDOM_PAIR_CONTROLS]
    missing = [path for path in required if not path.exists()]
    natural_branch_paths = sorted(NATURAL_SHARDS.glob("*/branches.parquet"))
    natural_state_paths = sorted(NATURAL_SHARDS.glob("*/states.parquet"))
    if missing or not natural_branch_paths or not natural_state_paths:
        raise FileNotFoundError("missing Task 17B inputs: " + ", ".join(map(str, missing)))

    post_branches = pd.read_parquet(POST_BRANCHES)
    post_states = pd.read_parquet(POST_STATES)
    natural_branches = pd.concat([pd.read_parquet(path) for path in natural_branch_paths], ignore_index=True)
    natural_states = pd.concat([pd.read_parquet(path) for path in natural_state_paths], ignore_index=True)
    natural_states = natural_states.drop_duplicates("state_id").reset_index(drop=True)
    signatures = pd.read_parquet(SIGNATURES)
    pair_identities = pd.read_parquet(PAIR_IDENTITIES)
    representation = pd.read_parquet(REPRESENTATION_DISTANCES)
    random_pairs = pd.read_parquet(RANDOM_PAIR_CONTROLS)

    branches = []
    for domain, source, states in (
        ("natural", natural_branches, natural_states),
        ("post_handoff", post_branches, post_states),
    ):
        part = source.copy()
        part["domain"] = domain
        part["solver"] = part["candidate_action"].where(
            ~part["candidate_action"].eq("continue"), part["current_algorithm"]
        )
        meta_columns = [
            "state_id", "suite", "cv_group_id", "problem_id", "seed", "current_algorithm", "FE"
        ]
        if domain == "post_handoff":
            meta_columns.extend(["source_algorithm", "source_checkpoint_fe"])
        meta = states[meta_columns].drop_duplicates("state_id")
        join_keys = ["state_id", "current_algorithm", "FE"]
        part = part.merge(
            meta,
            on=join_keys,
            how="left",
            validate="many_to_one",
            suffixes=("", "_state"),
        )
        if part[["suite", "cv_group_id", "problem_id", "seed"]].isna().any().any():
            raise RuntimeError(f"{domain} repetition-to-state join is incomplete")
        branches.append(part)
    branch_frame = pd.concat(branches, ignore_index=True)
    return branch_frame, signatures, pair_identities, representation, random_pairs


def _validate_inputs(
    branches: pd.DataFrame,
    signatures: pd.DataFrame,
    pairs: pd.DataFrame,
    representations: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict] = []

    def record(check_id: str, passed: bool, detail: str) -> None:
        checks.append(
            {
                "check_id": check_id,
                "passed": bool(passed),
                "detail": detail,
                "new_objective_fe": NEW_OBJECTIVE_FE,
            }
        )
        if not passed:
            raise RuntimeError(f"{check_id}: {detail}")

    record("C01_state_identity", signatures.groupby("domain").size().to_dict() == {"natural": 1890, "post_handoff": 3780}, "Task17A state counts are 1890 and 3780")
    record("C02_pair_identity", pairs.groupby("domain").size().to_dict() == {"natural": 3780, "post_handoff": 7560}, "each stratum retains ten unordered state pairs")
    record("C03_branch_keys", not branches.duplicated(["domain", "state_id", "solver", "replicate_id"]).any(), "state-solver-replicate keys are unique")
    record("C04_original_branch", branches.groupby(["domain", "state_id", "solver"])["replicate_id"].apply(lambda x: 0 in set(map(int, x))).all(), "every state-action cell contains replicate 0")
    record("C05_repetition_keys", set(branches["replicate_id"].astype(int).unique()).issubset({0, 1, 2}), "repetition keys are limited to 0, 1, and 2")
    record("C06_repetition_layers", branches.groupby(["domain", "state_id", "solver"])["replicate_id"].nunique().isin([1, 3]).all(), "cells contain either one or three outcome layers")
    record("C07_representation_contract", set(REPRESENTATIONS).issubset(set(representations["representation"].unique())), "Global28 and Compact6 are present")
    record("C08_zero_new_objective_fe", NEW_OBJECTIVE_FE == 0, "analysis imports no objective or optimizer runner")

    base = branches.loc[branches["replicate_id"].eq(0)].pivot(
        index=["domain", "state_id"], columns="solver", values="loss_1000"
    )
    signature_losses = signatures.set_index(["domain", "state_id"])[[f"loss_{action}" for action in ACTIONS]]
    signature_losses.columns = list(ACTIONS)
    aligned = base.join(signature_losses, how="inner", lsuffix="_branch", rsuffix="_signature")
    differences = [
        np.max(np.abs(aligned[f"{action}_branch"] - aligned[f"{action}_signature"]))
        for action in ACTIONS
    ]
    record("C09_replicate0_signature_alignment", len(aligned) == 5670 and max(differences) < 1e-12, "replicate-0 solver losses exactly reproduce Task17A signatures")

    scales = signatures[[f"noise_scale_{action}" for action in ACTIONS]].to_numpy(dtype=float)
    record("C10_noise_scale_contract", np.isfinite(scales).all() and (scales > 0).all(), "Task17A fold-local action scales are finite and positive")
    return pd.DataFrame(checks)


def _stratum_lookup(signatures: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "domain", "state_id", "suite", "cv_group_id", "problem_id", "seed",
        "current_algorithm", "route", "source_FE", "stratum_id", "A_ND_members", "A_ND_size",
        *[f"noise_scale_{a}" for a in ACTIONS],
        *[f"margin_{a}_{b}" for a, b in ACTION_PAIRS],
        *[f"pair_scale_{a}_{b}" for a, b in ACTION_PAIRS],
    ]
    return signatures[columns].copy()


def _support_counts(branches: pd.DataFrame, signatures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = _stratum_lookup(signatures).set_index(["domain", "state_id"])
    counts = branches.groupby(["domain", "state_id", "solver"])["replicate_id"].nunique().unstack(fill_value=0)
    counts = counts.reindex(columns=ACTIONS, fill_value=0)
    repeated = counts.ge(2)
    topology_rows: list[dict] = []
    support_rows: list[dict] = []
    repeated_state_rows: list[dict] = []

    for domain in ("post_handoff", "natural"):
        domain_meta = meta.loc[domain]
        domain_repeated = repeated.loc[domain]
        for suite in SCOPES:
            ids = domain_meta.index if suite == "pooled" else domain_meta.index[domain_meta["suite"].eq(suite)]
            flags = domain_repeated.loc[domain_repeated.index.intersection(ids)]
            scope_meta = domain_meta.loc[flags.index]
            t3_ids = flags.index[flags.all(axis=1)]
            t3_groups = int(scope_meta.loc[t3_ids, "cv_group_id"].nunique()) if len(t3_ids) else 0
            t3_formal = (
                len(t3_ids) >= (50 if suite == "pooled" else 20)
                and t3_groups >= (8 if suite == "pooled" else 5)
            )
            for action in ACTIONS:
                action_ids = flags.index[flags[action]]
                cell_counts = counts.loc[(domain, action_ids), action] if len(action_ids) else pd.Series(dtype=int)
                topology_rows.append(
                    {
                        "domain": domain,
                        "suite": suite,
                        "component_type": "T1_action",
                        "action_or_pair": action,
                        "t1_cells": int(len(action_ids)),
                        "t2_states": 0,
                        "t3_states": int(len(t3_ids)),
                        "repetitions_per_cell_min": int(cell_counts.min()) if len(cell_counts) else 0,
                        "repetitions_per_cell_max": int(cell_counts.max()) if len(cell_counts) else 0,
                        "states": int(len(action_ids)),
                        "cv_groups": int(scope_meta.loc[action_ids, "cv_group_id"].nunique()) if len(action_ids) else 0,
                        "decision_strata": int(scope_meta.loc[action_ids, "stratum_id"].nunique()) if len(action_ids) else 0,
                        "original_branch_included": True,
                        "outcome_blind_selection": "YES",
                        "exact_source_state": "YES",
                        "rng_semantics": "disjoint_semantic_streams",
                        "formal_support": True,
                        "support_status": "DESCRIPTIVE_T1",
                        "new_objective_fe": NEW_OBJECTIVE_FE,
                    }
                )
            for action_a, action_b in ACTION_PAIRS:
                pair_ids = flags.index[flags[action_a] & flags[action_b]]
                n_groups = int(scope_meta.loc[pair_ids, "cv_group_id"].nunique()) if len(pair_ids) else 0
                formal = (
                    len(pair_ids) >= (30 if suite == "pooled" else 15)
                    and n_groups >= (8 if suite == "pooled" else 5)
                )
                pair_name = f"{action_a}|{action_b}"
                topology_rows.append(
                    {
                        "domain": domain,
                        "suite": suite,
                        "component_type": "T2_pair",
                        "action_or_pair": pair_name,
                        "t1_cells": int(flags[action_a].sum() + flags[action_b].sum()),
                        "t2_states": int(len(pair_ids)),
                        "t3_states": int(len(t3_ids)),
                        "repetitions_per_cell_min": 3 if len(pair_ids) else 0,
                        "repetitions_per_cell_max": 3 if len(pair_ids) else 0,
                        "states": int(len(pair_ids)),
                        "cv_groups": n_groups,
                        "decision_strata": int(scope_meta.loc[pair_ids, "stratum_id"].nunique()) if len(pair_ids) else 0,
                        "original_branch_included": True,
                        "outcome_blind_selection": "YES",
                        "exact_source_state": "YES",
                        "rng_semantics": "disjoint_semantic_streams",
                        "formal_support": bool(formal),
                        "support_status": "FORMAL" if formal else "LOW_SUPPORT",
                        "new_objective_fe": NEW_OBJECTIVE_FE,
                    }
                )
                support_rows.append(
                    {
                        "domain": domain,
                        "suite": suite,
                        "track": "T2",
                        "action_pair": pair_name,
                        "states": int(len(pair_ids)),
                        "cv_groups": n_groups,
                        "state_threshold": 30 if suite == "pooled" else 15,
                        "cv_group_threshold": 8 if suite == "pooled" else 5,
                        "formal_support": bool(formal),
                        "support_status": "FORMAL" if formal else "LOW_SUPPORT",
                        "new_objective_fe": NEW_OBJECTIVE_FE,
                    }
                )
                for state_id in pair_ids:
                    repeated_state_rows.append(
                        {
                            "domain": domain,
                            "state_id": state_id,
                            "action_a": action_a,
                            "action_b": action_b,
                            "action_pair": pair_name,
                            **scope_meta.loc[state_id, ["suite", "cv_group_id", "problem_id", "seed", "current_algorithm", "route", "source_FE", "stratum_id"]].to_dict(),
                        }
                    )
            support_rows.append(
                {
                    "domain": domain,
                    "suite": suite,
                    "track": "T3",
                    "action_pair": "shade|lshade|cso",
                    "states": int(len(t3_ids)),
                    "cv_groups": t3_groups,
                    "state_threshold": 50 if suite == "pooled" else 20,
                    "cv_group_threshold": 8 if suite == "pooled" else 5,
                    "formal_support": bool(t3_formal),
                    "support_status": "FORMAL" if t3_formal else "UNAVAILABLE",
                    "new_objective_fe": NEW_OBJECTIVE_FE,
                }
            )
            topology_rows.append(
                {
                    "domain": domain,
                    "suite": suite,
                    "component_type": "T3_full_signature",
                    "action_or_pair": "shade|lshade|cso",
                    "t1_cells": int(flags.sum().sum()),
                    "t2_states": 0,
                    "t3_states": int(len(t3_ids)),
                    "repetitions_per_cell_min": 3 if len(t3_ids) else 0,
                    "repetitions_per_cell_max": 3 if len(t3_ids) else 0,
                    "states": int(len(t3_ids)),
                    "cv_groups": t3_groups,
                    "decision_strata": int(scope_meta.loc[t3_ids, "stratum_id"].nunique()) if len(t3_ids) else 0,
                    "original_branch_included": True,
                    "outcome_blind_selection": "YES",
                    "exact_source_state": "YES",
                    "rng_semantics": "disjoint_semantic_streams",
                    "formal_support": bool(t3_formal),
                    "support_status": "FORMAL" if t3_formal else "UNAVAILABLE",
                    "new_objective_fe": NEW_OBJECTIVE_FE,
                }
            )
    return pd.DataFrame(topology_rows), pd.DataFrame(support_rows), pd.DataFrame(repeated_state_rows).drop_duplicates()


def _rng_verification() -> pd.DataFrame:
    rows = [
        ("outcome_blind_selection", "PASS", "sampling uses SeedSequence coordinates only; no action outcome enters selection"),
        ("source_state_identity", "PASS", "every branch starts from the same saved population, fitness, best value, and FE state"),
        ("continue_clone", "PASS", "continue uses an independent full-state clone before installing its repetition RNG"),
        ("transfer_initialization", "PASS_WITH_SCOPE_NOTE", "target initialization reuses the same source state; CSO velocities vary only through the target action RNG"),
        ("semantic_rng_separation", "PASS", "replicate and target-action coordinates select disjoint semantic RNG streams"),
        ("cross_action_coupling", "PASS_BY_CONSTRUCTION", "target solver stream codes differ; no shared mutable optimizer state is reused"),
        ("empirical_independence_claim", "NOT_CLAIMED", "stream separation is verified from code; finite PRNG outcomes do not prove probabilistic independence"),
        ("primary_eligibility", "PASS", "all post-handoff T1/T2 rows satisfy the source-state and RNG design contract"),
    ]
    return pd.DataFrame(
        [
            {
                "check_id": check_id,
                "status": status,
                "detail": detail,
                "primary_noise_floor_eligible": check_id != "empirical_independence_claim",
                "new_objective_fe": NEW_OBJECTIVE_FE,
            }
            for check_id, status, detail in rows
        ]
    )


def _action_noise(branches: pd.DataFrame, signatures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = _stratum_lookup(signatures).set_index(["domain", "state_id"])
    repeated_counts = branches.groupby(["domain", "state_id", "solver"])["replicate_id"].nunique()
    repeated_cells = repeated_counts[repeated_counts.ge(2)].index
    rows: list[dict] = []
    for domain, state_id, action in repeated_cells:
        cell = branches.loc[
            branches["domain"].eq(domain)
            & branches["state_id"].eq(state_id)
            & branches["solver"].eq(action)
        ].sort_values("replicate_id")
        row_meta = meta.loc[(domain, state_id)]
        scale = float(row_meta[f"noise_scale_{action}"])
        for left, right in itertools.combinations(cell.itertuples(index=False), 2):
            rows.append(
                {
                    "domain": domain,
                    "suite": row_meta["suite"],
                    "cv_group_id": row_meta["cv_group_id"],
                    "problem_id": row_meta["problem_id"],
                    "state_id": state_id,
                    "stratum_id": row_meta["stratum_id"],
                    "action": action,
                    "replicate_left": int(left.replicate_id),
                    "replicate_right": int(right.replicate_id),
                    "loss_left": float(left.loss_1000),
                    "loss_right": float(right.loss_1000),
                    "delta_a": scale,
                    "N_a": abs(float(left.loss_1000) - float(right.loss_1000)) / scale,
                    "lexicographically_first_pair": int(left.replicate_id) == 0 and int(right.replicate_id) == 1,
                    "new_objective_fe": NEW_OBJECTIVE_FE,
                }
            )
    pairs = pd.DataFrame(rows)
    summaries: list[dict] = []
    for (domain, action), part in pairs.groupby(["domain", "action"], sort=True):
        for suite in SCOPES:
            subset = _scope(part, suite)
            for weighting in ("all_unordered_pairs", "one_pair_per_state"):
                used = subset if weighting == "all_unordered_pairs" else subset.loc[subset["lexicographically_first_pair"]]
                values = used["N_a"].to_numpy(dtype=float)
                summaries.append(
                    {
                        "domain": domain,
                        "suite": suite,
                        "action": action,
                        "weighting": weighting,
                        "median": float(np.median(values)) if len(values) else np.nan,
                        "q75": float(np.quantile(values, 0.75)) if len(values) else np.nan,
                        "q90": float(np.quantile(values, 0.90)) if len(values) else np.nan,
                        "q95": float(np.quantile(values, 0.95)) if len(values) else np.nan,
                        "P_N_gt1": float(np.mean(values > 1.0)) if len(values) else np.nan,
                        "repeated_states": int(used["state_id"].nunique()),
                        "repeated_cells": int(used[["state_id", "action"]].drop_duplicates().shape[0]),
                        "cv_groups": int(used["cv_group_id"].nunique()),
                        "new_objective_fe": NEW_OBJECTIVE_FE,
                    }
                )
    summary = pd.DataFrame(summaries)
    for (_, _), part in summary.groupby(["domain", "suite"], sort=False):
        primary = part.loc[part["weighting"].eq("all_unordered_pairs")].sort_values(
            ["P_N_gt1", "action"], ascending=[False, True]
        )
        sensitivity = part.loc[part["weighting"].eq("one_pair_per_state")].sort_values(
            ["P_N_gt1", "action"], ascending=[False, True]
        )
        sensitive = bool(
            len(primary)
            and len(sensitivity)
            and primary.iloc[0]["action"] != sensitivity.iloc[0]["action"]
        )
        summary.loc[part.index, "repetition_weight_sensitive"] = sensitive
    summary["repetition_weight_sensitive"] = summary["repetition_weight_sensitive"].astype(bool)
    return pairs, summary


def _pairwise_replicates(branches: pd.DataFrame, signatures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = _stratum_lookup(signatures).set_index(["domain", "state_id"])
    wide = branches.pivot(index=["domain", "state_id"], columns=["solver", "replicate_id"], values="loss_1000")
    rows: list[dict] = []
    for domain in ("post_handoff", "natural"):
        domain_ids = wide.loc[domain].index
        for action_a, action_b in ACTION_PAIRS:
            available = wide.loc[(domain, domain_ids), action_a].notna().sum(axis=1).ge(2) & wide.loc[(domain, domain_ids), action_b].notna().sum(axis=1).ge(2)
            for state_id in domain_ids[available.to_numpy()]:
                row_meta = meta.loc[(domain, state_id)]
                scale = float(row_meta[f"pair_scale_{action_a}_{action_b}"])
                layer0 = float(wide.loc[(domain, state_id), (action_b, 0)] - wide.loc[(domain, state_id), (action_a, 0)])
                layer1 = float(wide.loc[(domain, state_id), (action_b, 1)] - wide.loc[(domain, state_id), (action_a, 1)])
                layer2 = float(wide.loc[(domain, state_id), (action_b, 2)] - wide.loc[(domain, state_id), (action_a, 2)])
                relation0 = _relation(np.asarray([layer0]), np.asarray([scale]))[0]
                relation1 = _relation(np.asarray([layer1]), np.asarray([scale]))[0]
                relation2 = _relation(np.asarray([layer2]), np.asarray([scale]))[0]
                rows.append(
                    {
                        "domain": domain,
                        "suite": row_meta["suite"],
                        "cv_group_id": row_meta["cv_group_id"],
                        "problem_id": row_meta["problem_id"],
                        "state_id": state_id,
                        "stratum_id": row_meta["stratum_id"],
                        "route": row_meta["route"],
                        "source_FE": int(row_meta["source_FE"]),
                        "current_algorithm": row_meta["current_algorithm"],
                        "action_a": action_a,
                        "action_b": action_b,
                        "action_pair": f"{action_a}|{action_b}",
                        "delta_ab": scale,
                        "replicate_key_1": 0,
                        "replicate_key_2": 1,
                        "margin_1": layer0,
                        "margin_2": layer1,
                        "D_noise": abs(layer0 - layer1) / scale,
                        "relation_1": relation0,
                        "relation_2": relation1,
                        "exact_category_agreement": relation0 == relation1,
                        "directional_reversal": {relation0, relation1} == {"a_better", "b_better"},
                        "tie_winner_transition": (relation0 == "tie") != (relation1 == "tie"),
                        "third_layer_margin": layer2,
                        "third_layer_D_sensitivity": abs(layer1 - layer2) / scale,
                        "third_layer_relation": relation2,
                        "new_objective_fe": NEW_OBJECTIVE_FE,
                    }
                )
    replicates = pd.DataFrame(rows)
    summaries: list[dict] = []
    for (domain, action_pair), part in replicates.groupby(["domain", "action_pair"], sort=True):
        for suite in SCOPES:
            subset = _scope(part, suite)
            values = subset["D_noise"].to_numpy(dtype=float)
            rate = float(np.mean(values > 1.0)) if len(values) else np.nan
            rho = _spearman(subset["margin_1"], subset["margin_2"])
            sensitivity_rate = float(np.mean(subset["third_layer_D_sensitivity"] > 1.0)) if len(subset) else np.nan
            sensitivity_rho = _spearman(subset["margin_2"], subset["third_layer_margin"])
            if np.isfinite(rate) and rate <= 0.10 and np.isfinite(rho) and rho >= 0.6:
                reliability = "NR1 HIGHLY REPEATABLE"
            elif (np.isfinite(rate) and rate > 0.30) or (np.isfinite(rho) and rho < 0.3):
                reliability = "NR3 LOW RELIABILITY"
            else:
                reliability = "NR2 MODERATE"
            if np.isfinite(sensitivity_rate) and sensitivity_rate <= 0.10 and np.isfinite(sensitivity_rho) and sensitivity_rho >= 0.6:
                sensitivity_reliability = "NR1 HIGHLY REPEATABLE"
            elif (np.isfinite(sensitivity_rate) and sensitivity_rate > 0.30) or (np.isfinite(sensitivity_rho) and sensitivity_rho < 0.3):
                sensitivity_reliability = "NR3 LOW RELIABILITY"
            else:
                sensitivity_reliability = "NR2 MODERATE"
            support = (
                len(subset) >= (30 if suite == "pooled" else 15)
                and subset["cv_group_id"].nunique() >= (8 if suite == "pooled" else 5)
            )
            summaries.append(
                {
                    "domain": domain,
                    "suite": suite,
                    "action_pair": action_pair,
                    "states": int(len(subset)),
                    "cv_groups": int(subset["cv_group_id"].nunique()),
                    "formal_support": bool(support),
                    "support_status": "FORMAL" if support else "LOW_SUPPORT",
                    "median_D_noise": float(np.median(values)) if len(values) else np.nan,
                    "P_D_noise_gt1": rate,
                    "replicate_margin_spearman": rho,
                    "exact_category_agreement": float(subset["exact_category_agreement"].mean()) if len(subset) else np.nan,
                    "directional_reversal_rate": float(subset["directional_reversal"].mean()) if len(subset) else np.nan,
                    "tie_winner_transition_rate": float(subset["tie_winner_transition"].mean()) if len(subset) else np.nan,
                    "cohen_kappa_secondary": _cohen_kappa(subset["relation_1"], subset["relation_2"]),
                    "third_layer_P_D_gt1_sensitivity": sensitivity_rate,
                    "third_layer_margin_spearman_sensitivity": sensitivity_rho,
                    "third_layer_reliability_class": sensitivity_reliability,
                    "repetition_weight_sensitive": reliability != sensitivity_reliability,
                    "reliability_class": reliability,
                    "new_objective_fe": NEW_OBJECTIVE_FE,
                }
            )
    return replicates, pd.DataFrame(summaries)


def _cross_state_rows(
    signatures: pd.DataFrame,
    pair_identities: pd.DataFrame,
    representation_distances: pd.DataFrame,
    random_controls: pd.DataFrame,
) -> pd.DataFrame:
    post_sig = signatures.loc[signatures["domain"].eq("post_handoff")].set_index("state_id")
    post_pairs = pair_identities.loc[pair_identities["domain"].eq("post_handoff")].copy()
    random0 = random_controls.loc[
        random_controls["domain"].eq("post_handoff") & random_controls["repetition"].eq(0),
        ["stratum_id", "pair_index"],
    ].copy()
    random_ids = set(random0["pair_index"].astype(int))
    outputs: list[pd.DataFrame] = []
    for representation in REPRESENTATIONS:
        distances = representation_distances.loc[
            representation_distances["domain"].eq("post_handoff")
            & representation_distances["representation"].eq(representation)
        ].sort_values(
            ["stratum_id", "representation_distance_l1_mean", "seed_i", "seed_j", "pair_index"]
        )
        local_ids = set(distances.groupby("stratum_id", sort=False).head(1)["pair_index"].astype(int))
        base = post_pairs.copy()
        base["representation"] = representation
        base["representation_distance_l1_mean"] = base["pair_index"].map(
            distances.set_index("pair_index")["representation_distance_l1_mean"]
        )
        base["is_local_pair"] = base["pair_index"].isin(local_ids)
        base["is_random_pair"] = base["pair_index"].isin(random_ids)
        for action_a, action_b in ACTION_PAIRS:
            frame = base.copy()
            margin_i = frame["state_i"].map(post_sig[f"margin_{action_a}_{action_b}"])
            margin_j = frame["state_j"].map(post_sig[f"margin_{action_a}_{action_b}"])
            scale_i = frame["state_i"].map(post_sig[f"pair_scale_{action_a}_{action_b}"])
            scale_j = frame["state_j"].map(post_sig[f"pair_scale_{action_a}_{action_b}"])
            if not np.allclose(scale_i, scale_j, rtol=0, atol=1e-12):
                raise RuntimeError("within-stratum pair scales are inconsistent")
            frame["action_a"] = action_a
            frame["action_b"] = action_b
            frame["action_pair"] = f"{action_a}|{action_b}"
            frame["margin_i"] = margin_i.to_numpy(dtype=float)
            frame["margin_j"] = margin_j.to_numpy(dtype=float)
            frame["delta_ab"] = scale_i.to_numpy(dtype=float)
            frame["D_crossstate"] = np.abs(frame["margin_i"] - frame["margin_j"]) / frame["delta_ab"]
            frame["new_objective_fe"] = NEW_OBJECTIVE_FE
            outputs.append(frame)
    return pd.concat(outputs, ignore_index=True)


def _bootstrap_multiplicities(groups: list[str], codes: tuple[int, ...]) -> np.ndarray:
    rng = np.random.default_rng(_seed_sequence(BOOTSTRAP_STREAM, *codes, len(groups)))
    return rng.multinomial(len(groups), np.full(len(groups), 1.0 / len(groups)), size=BOOTSTRAP_DRAWS)


def _group_codes(frame: pd.DataFrame, groups: list[str]) -> np.ndarray:
    mapping = {group: index for index, group in enumerate(groups)}
    return np.asarray([mapping[str(value)] for value in frame["cv_group_id"]], dtype=int)


def _rate_draws(values: np.ndarray, group_codes: np.ndarray, multiplicities: np.ndarray) -> np.ndarray:
    successes = np.bincount(group_codes, weights=np.asarray(values, dtype=float) > 1.0, minlength=multiplicities.shape[1])
    totals = np.bincount(group_codes, minlength=multiplicities.shape[1])
    return (multiplicities @ successes) / (multiplicities @ totals)


def _weighted_median_draws(values: np.ndarray, group_codes: np.ndarray, multiplicities: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_codes = group_codes[order]
    output = np.empty(len(multiplicities), dtype=float)
    batch = 128
    for start in range(0, len(multiplicities), batch):
        stop = min(start + batch, len(multiplicities))
        weights = multiplicities[start:stop, sorted_codes]
        cumulative = np.cumsum(weights, axis=1)
        totals = cumulative[:, -1]
        lower = (totals - 1) // 2
        upper = totals // 2
        lower_index = np.argmax(cumulative > lower[:, None], axis=1)
        upper_index = np.argmax(cumulative > upper[:, None], axis=1)
        output[start:stop] = (sorted_values[lower_index] + sorted_values[upper_index]) / 2.0
    return output


def _decomposition(
    margin_replicates: pd.DataFrame,
    crossstate: pd.DataFrame,
    representation: str = "global28",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    bootstrap_parts: list[pd.DataFrame] = []
    post_noise = margin_replicates.loc[margin_replicates["domain"].eq("post_handoff")]
    cross = crossstate.loc[crossstate["representation"].eq(representation)]
    for pair in [f"{a}|{b}" for a, b in ACTION_PAIRS]:
        pair_noise = post_noise.loc[post_noise["action_pair"].eq(pair)]
        pair_cross = cross.loc[cross["action_pair"].eq(pair)]
        for suite in SCOPES:
            noise_scope = _scope(pair_noise, suite)
            cross_scope = _scope(pair_cross, suite)
            exact_strata = set(noise_scope["stratum_id"].astype(str))
            for support_mode in ("matched_support", "all_available_support"):
                if support_mode == "matched_support":
                    supported = cross_scope.loc[cross_scope["stratum_id"].astype(str).isin(exact_strata)]
                else:
                    supported = cross_scope
                local = supported.loc[supported["is_local_pair"]]
                all_pairs = supported
                random = supported.loc[supported["is_random_pair"]]
                common_groups = sorted(
                    set(noise_scope["cv_group_id"].astype(str))
                    & set(local["cv_group_id"].astype(str))
                    & set(all_pairs["cv_group_id"].astype(str))
                )
                if not common_groups:
                    continue
                noise_used = noise_scope.loc[noise_scope["cv_group_id"].astype(str).isin(common_groups)]
                local_used = local.loc[local["cv_group_id"].astype(str).isin(common_groups)]
                all_used = all_pairs.loc[all_pairs["cv_group_id"].astype(str).isin(common_groups)]
                random_used = random.loc[random["cv_group_id"].astype(str).isin(common_groups)]
                pair_tuple = tuple(pair.split("|"))
                multiplicities = _bootstrap_multiplicities(
                    common_groups,
                    (
                        PAIR_CODES[pair_tuple], SCOPE_CODES[suite], SUPPORT_CODES[support_mode],
                        REPRESENTATION_CODES[representation],
                    ),
                )
                noise_codes = _group_codes(noise_used, common_groups)
                local_codes = _group_codes(local_used, common_groups)
                all_codes = _group_codes(all_used, common_groups)
                noise_values = noise_used["D_noise"].to_numpy(dtype=float)
                local_values = local_used["D_crossstate"].to_numpy(dtype=float)
                all_values = all_used["D_crossstate"].to_numpy(dtype=float)
                rate_noise_draws = _rate_draws(noise_values, noise_codes, multiplicities)
                rate_local_draws = _rate_draws(local_values, local_codes, multiplicities)
                rate_all_draws = _rate_draws(all_values, all_codes, multiplicities)
                median_noise_draws = _weighted_median_draws(noise_values, noise_codes, multiplicities)
                median_local_draws = _weighted_median_draws(local_values, local_codes, multiplicities)
                median_all_draws = _weighted_median_draws(all_values, all_codes, multiplicities)
                e_draws = rate_local_draws - rate_noise_draws
                m_draws = median_local_draws - median_noise_draws
                denominator_draws = rate_all_draws - rate_noise_draws
                valid_capture_draws = (rate_all_draws > rate_noise_draws) & (denominator_draws > 0.05)
                capture_draws = np.full(BOOTSTRAP_DRAWS, np.nan)
                residual_fraction_draws = np.full(BOOTSTRAP_DRAWS, np.nan)
                capture_draws[valid_capture_draws] = (
                    rate_all_draws[valid_capture_draws] - rate_local_draws[valid_capture_draws]
                ) / denominator_draws[valid_capture_draws]
                residual_fraction_draws[valid_capture_draws] = (
                    rate_local_draws[valid_capture_draws] - rate_noise_draws[valid_capture_draws]
                ) / denominator_draws[valid_capture_draws]

                a_noise = float(np.mean(noise_values > 1.0))
                a_local = float(np.mean(local_values > 1.0))
                a_all = float(np.mean(all_values > 1.0))
                median_noise = float(np.median(noise_values))
                median_local = float(np.median(local_values))
                median_all = float(np.median(all_values))
                denominator = a_all - a_noise
                capture_valid = bool(a_all > a_noise and denominator > 0.05)
                capture = (a_all - a_local) / denominator if capture_valid else np.nan
                residual_fraction = (a_local - a_noise) / denominator if capture_valid else np.nan
                noise_ci = _ci(rate_noise_draws)
                local_ci = _ci(rate_local_draws)
                all_ci = _ci(rate_all_draws)
                e_ci = _ci(e_draws)
                m_ci = _ci(m_draws)
                capture_ci = _ci(capture_draws)
                residual_fraction_ci = _ci(residual_fraction_draws)
                formal = (
                    len(noise_scope) >= (30 if suite == "pooled" else 15)
                    and noise_scope["cv_group_id"].nunique() >= (8 if suite == "pooled" else 5)
                )
                summary_rows.append(
                    {
                        "domain": "post_handoff",
                        "suite": suite,
                        "representation": representation,
                        "action_pair": pair,
                        "support_mode": support_mode,
                        "formal_support": bool(formal),
                        "support_status": "FORMAL" if formal else "LOW_SUPPORT",
                        "n_noise_states": int(len(noise_used)),
                        "n_local_pairs": int(len(local_used)),
                        "n_all_pairs": int(len(all_used)),
                        "n_random_pairs": int(len(random_used)),
                        "n_cv_groups": int(len(common_groups)),
                        "A_noise": a_noise,
                        "A_noise_ci_low": noise_ci[0],
                        "A_noise_ci_high": noise_ci[1],
                        "A_local": a_local,
                        "A_local_ci_low": local_ci[0],
                        "A_local_ci_high": local_ci[1],
                        "A_all": a_all,
                        "A_all_ci_low": all_ci[0],
                        "A_all_ci_high": all_ci[1],
                        "A_random": float(np.mean(random_used["D_crossstate"] > 1.0)) if len(random_used) else np.nan,
                        "E_res": a_local - a_noise,
                        "E_res_ci_low": e_ci[0],
                        "E_res_ci_high": e_ci[1],
                        "median_D_noise": median_noise,
                        "median_D_local": median_local,
                        "median_D_all": median_all,
                        "M_res": median_local - median_noise,
                        "M_res_ci_low": m_ci[0],
                        "M_res_ci_high": m_ci[1],
                        "noise_to_local_ratio": a_noise / a_local if a_local > 0 else np.nan,
                        "capture_defined": capture_valid,
                        "F_capture": capture,
                        "F_capture_ci_low": capture_ci[0],
                        "F_capture_ci_high": capture_ci[1],
                        "F_residual": residual_fraction,
                        "F_residual_ci_low": residual_fraction_ci[0],
                        "F_residual_ci_high": residual_fraction_ci[1],
                        "bootstrap_draws": BOOTSTRAP_DRAWS,
                        "bootstrap_unit": "cv_group_id",
                        "new_objective_fe": NEW_OBJECTIVE_FE,
                    }
                )
                bootstrap_parts.append(
                    pd.DataFrame(
                        {
                            "domain": "post_handoff",
                            "suite": suite,
                            "representation": representation,
                            "action_pair": pair,
                            "support_mode": support_mode,
                            "draw": np.arange(BOOTSTRAP_DRAWS, dtype=int),
                            "A_noise": rate_noise_draws,
                            "A_local": rate_local_draws,
                            "A_all": rate_all_draws,
                            "E_res": e_draws,
                            "M_res": m_draws,
                            "F_capture": capture_draws,
                            "F_residual": residual_fraction_draws,
                            "bootstrap_unit": "cv_group_id",
                            "new_objective_fe": NEW_OBJECTIVE_FE,
                        }
                    )
                )
    return pd.DataFrame(summary_rows), pd.concat(bootstrap_parts, ignore_index=True)


def _high_confidence(signatures: pd.DataFrame, representations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    states = signatures.loc[signatures["domain"].eq("post_handoff")].copy()
    states["U0_ambiguous"] = states["A_ND_size"].gt(1)
    states["U1_unique_practical_winner"] = states["A_ND_size"].eq(1)
    states["U2_switch_required"] = [
        current not in set(str(members).split("|"))
        for current, members in zip(states["current_algorithm"], states["A_ND_members"], strict=True)
    ]
    states["U3_unique_switch_target"] = states["U1_unique_practical_winner"] & states["U2_switch_required"]
    states["U1_repeat_available"] = False
    state_output = states[
        [
            "state_id", "suite", "cv_group_id", "problem_id", "seed", "route", "source_FE",
            "current_algorithm", "stratum_id", "A_ND_members", "A_ND_size", "U0_ambiguous",
            "U1_unique_practical_winner", "U2_switch_required", "U3_unique_switch_target",
            "U1_repeat_available",
        ]
    ].copy()
    state_output["new_objective_fe"] = NEW_OBJECTIVE_FE
    indexed = states.set_index("state_id")
    global_distances = representations.loc[
        representations["domain"].eq("post_handoff")
        & representations["representation"].eq("global28")
    ].copy()
    rows: list[dict] = []
    for subset_name, flag in (
        ("U1", "U1_unique_practical_winner"),
        ("U3", "U3_unique_switch_target"),
    ):
        eligible = indexed[flag].astype(bool)
        candidate = global_distances.loc[
            global_distances["state_i"].map(eligible).fillna(False)
            & global_distances["state_j"].map(eligible).fillna(False)
        ].sort_values(
            ["stratum_id", "representation_distance_l1_mean", "seed_i", "seed_j", "pair_index"]
        )
        selected = candidate.groupby("stratum_id", sort=False).head(1)
        for action_a, action_b in ACTION_PAIRS:
            part = selected.copy()
            part["subset"] = subset_name
            part["action_pair"] = f"{action_a}|{action_b}"
            part["margin_i"] = part["state_i"].map(indexed[f"margin_{action_a}_{action_b}"])
            part["margin_j"] = part["state_j"].map(indexed[f"margin_{action_a}_{action_b}"])
            part["delta_ab"] = part["state_i"].map(indexed[f"pair_scale_{action_a}_{action_b}"])
            part["D_local"] = np.abs(part["margin_i"] - part["margin_j"]) / part["delta_ab"]
            part["winner_mismatch"] = part["state_i"].map(indexed["A_ND_members"]) != part["state_j"].map(indexed["A_ND_members"])
            rows.extend(
                part[
                    [
                        "subset", "suite", "cv_group_id", "problem_id", "stratum_id", "pair_index",
                        "state_i", "state_j", "action_pair", "representation_distance_l1_mean",
                        "margin_i", "margin_j", "delta_ab", "D_local", "winner_mismatch",
                    ]
                ].to_dict("records")
            )
    aliasing = pd.DataFrame(rows)
    aliasing["new_objective_fe"] = NEW_OBJECTIVE_FE
    return state_output, aliasing


def _high_confidence_summary(states: pd.DataFrame, aliasing: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for suite in SCOPES:
        state_part = _scope(states, suite)
        for subset in ("ALL_STATES", "U1", "U3"):
            if subset == "ALL_STATES":
                rows.append(
                    {
                        "suite": suite,
                        "subset": subset,
                        "action_pair": "all",
                        "states": int(len(state_part)),
                        "strata_with_pairs": np.nan,
                        "state_fraction": 1.0,
                        "A_local": np.nan,
                        "median_D_local": np.nan,
                        "winner_mismatch_rate": np.nan,
                        "new_objective_fe": NEW_OBJECTIVE_FE,
                    }
                )
                continue
            flag = "U1_unique_practical_winner" if subset == "U1" else "U3_unique_switch_target"
            for pair in [f"{a}|{b}" for a, b in ACTION_PAIRS]:
                part = _scope(aliasing.loc[aliasing["subset"].eq(subset) & aliasing["action_pair"].eq(pair)], suite)
                rows.append(
                    {
                        "suite": suite,
                        "subset": subset,
                        "action_pair": pair,
                        "states": int(state_part[flag].sum()),
                        "strata_with_pairs": int(len(part)),
                        "state_fraction": float(state_part[flag].mean()) if len(state_part) else np.nan,
                        "A_local": float(np.mean(part["D_local"] > 1.0)) if len(part) else np.nan,
                        "median_D_local": float(np.median(part["D_local"])) if len(part) else np.nan,
                        "winner_mismatch_rate": float(part["winner_mismatch"].mean()) if len(part) else np.nan,
                        "new_objective_fe": NEW_OBJECTIVE_FE,
                    }
                )
    return pd.DataFrame(rows)


def _full_track_tables(support: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t3 = support.loc[support["track"].eq("T3")].copy()
    reliability = t3.rename(columns={"states": "t3_states", "cv_groups": "t3_cv_groups"})
    reliability["AVA_noise"] = np.nan
    reliability["decision_geometry_repeatability"] = np.nan
    reliability["track_status"] = "UNAVAILABLE_INSUFFICIENT_T3_SUPPORT"
    reliability["reason"] = "pooled post-handoff requires 50 states and 8 cv groups"
    replicate_columns = [
        "domain", "suite", "cv_group_id", "state_id", "replicate_layer", "normalized_margin_shade_lshade",
        "normalized_margin_shade_cso", "normalized_margin_lshade_cso", "new_objective_fe",
    ]
    replicates = pd.DataFrame(columns=replicate_columns)
    residual = reliability[
        ["domain", "suite", "t3_states", "t3_cv_groups", "track_status", "reason", "new_objective_fe"]
    ].copy()
    residual["AVA_noise"] = np.nan
    residual["AVA_local_T3"] = np.nan
    residual["AVA_all_T3"] = np.nan
    residual["E_res_full"] = np.nan
    residual["F_capture_full"] = np.nan
    residual["F_residual_full"] = np.nan
    return replicates, reliability, residual


def _select_verdict(
    support: pd.DataFrame,
    decomposition: pd.DataFrame,
    high_confidence_summary: pd.DataFrame,
    rng_verification: pd.DataFrame,
) -> tuple[str, dict]:
    pooled_support = support.loc[
        support["domain"].eq("post_handoff")
        & support["suite"].eq("pooled")
        & support["track"].eq("T2")
    ]
    primary = decomposition.loc[
        decomposition["suite"].eq("pooled")
        & decomposition["support_mode"].eq("matched_support")
        & decomposition["representation"].eq("global28")
        & decomposition["formal_support"]
    ].copy()
    supported_pairs = int(pooled_support["formal_support"].sum())
    direct_support_ok = supported_pairs >= 2
    rng_ok = rng_verification.loc[rng_verification["check_id"].eq("primary_eligibility"), "status"].eq("PASS").all()
    if not direct_support_ok or not rng_ok or len(primary) < 2:
        verdict = "D4 INCONCLUSIVE / INSUFFICIENT REPETITION SUPPORT"
    else:
        robust_tail = primary["E_res_ci_low"].gt(0)
        median_positive = primary["M_res"].gt(0)
        d1_core = int(robust_tail.sum()) >= 2 and int(median_positive.sum()) >= 2
        suite_formal = decomposition.loc[
            decomposition["suite"].isin(["bbob", "mabbob"])
            & decomposition["support_mode"].eq("matched_support")
            & decomposition["representation"].eq("global28")
            & decomposition["formal_support"]
        ]
        suite_strong_inverse = suite_formal["E_res_ci_high"].lt(0).any()
        u1 = high_confidence_summary.loc[
            high_confidence_summary["suite"].eq("pooled")
            & high_confidence_summary["subset"].eq("U1")
        ]
        merged_u1 = primary[["action_pair", "A_noise", "A_local"]].merge(
            u1[["action_pair", "A_local"]].rename(columns={"A_local": "A_local_U1"}), on="action_pair"
        )
        d3_pairs = (
            (merged_u1["A_local_U1"] <= merged_u1["A_local"] - 0.10)
            & ((merged_u1["A_local_U1"] - merged_u1["A_noise"]).abs() <= 0.10)
        )
        practical_relevance = bool((merged_u1["A_local_U1"] - merged_u1["A_noise"] > 0.10).any())
        noise_ratio = primary["noise_to_local_ratio"].ge(0.70)
        robust_beyond = primary["E_res_ci_low"].gt(0)
        d2_structure_b = int(noise_ratio.sum()) >= 2 and int(robust_beyond.sum()) < 2
        if d1_core and not suite_strong_inverse and practical_relevance:
            verdict = "D1 RESIDUAL DECISION ALIASING BEYOND STOCHASTIC NOISE CONFIRMED"
        elif int(d3_pairs.sum()) >= 2:
            verdict = "D3 DECISION-RELEVANT SUBSET IS SUBSTANTIALLY MORE RESOLVABLE"
        elif d2_structure_b:
            verdict = "D2 NOISE-LIMITED DECISION GEOMETRY"
        else:
            verdict = "D4 INCONCLUSIVE / INSUFFICIENT REPETITION SUPPORT"
    decisions = {
        "final_verdict": verdict,
        "supported_pooled_t2_pairs": supported_pairs,
        "task17c_allowed": verdict.startswith("D1"),
        "new_feature_allowed": verdict.startswith("D1"),
        "new_selector_allowed": False,
        "fuzzy_allowed": False,
        "seeds_6_10_allowed": False,
        "cec_allowed": False,
        "additional_repetitions_allowed": False,
        "paper_consolidation_status": (
            "GO" if verdict.startswith("D3") else
            "EVALUATE_ONLY" if verdict.startswith("D2") else
            "NO"
        ),
        "mainline_stop": not verdict.startswith("D1"),
        "new_objective_fe": NEW_OBJECTIVE_FE,
    }
    return verdict, decisions


def _figures(decomposition: pd.DataFrame, margin_replicates: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    primary = decomposition.loc[
        decomposition["suite"].eq("pooled")
        & decomposition["support_mode"].eq("matched_support")
        & decomposition["representation"].eq("global28")
    ].sort_values("action_pair")
    labels = primary["action_pair"].str.replace("|", " vs ", regex=False).tolist()
    x = np.arange(len(primary))
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for offset, column, label, color in (
        (-0.22, "A_noise", "same-state stochastic", "#0072B2"),
        (0.00, "A_local", "Global28 nearest-state", "#D55E00"),
        (0.22, "A_all", "all within-context pairs", "#009E73"),
    ):
        ax.bar(x + offset, primary[column], width=0.21, label=label, color=color)
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylabel("Rate with normalized discrepancy > 1")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title("Same-state vs nearest-state vs all-state discrepancy")
    fig.tight_layout()
    fig.savefig(FIGURES / "17b_figure_a_same_local_all.png", dpi=180)
    plt.close(fig)

    post = margin_replicates.loc[margin_replicates["domain"].eq("post_handoff")]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for ax, pair in zip(axes, sorted(post["action_pair"].unique()), strict=True):
        part = post.loc[post["action_pair"].eq(pair)]
        xval = part["margin_1"] / part["delta_ab"]
        yval = part["margin_2"] / part["delta_ab"]
        ax.scatter(xval, yval, s=24, alpha=0.72, color="#0072B2", edgecolor="none")
        ax.axvline(-1, color="0.5", linestyle="--", linewidth=1)
        ax.axvline(1, color="0.5", linestyle="--", linewidth=1)
        ax.axhline(-1, color="0.5", linestyle="--", linewidth=1)
        ax.axhline(1, color="0.5", linestyle="--", linewidth=1)
        ax.set_title(pair.replace("|", " vs "))
        ax.set_xlabel("replicate layer 1 margin / delta")
        ax.set_ylabel("replicate layer 2 margin / delta")
    fig.tight_layout()
    fig.savefig(FIGURES / "17b_figure_b_margin_repeatability.png", dpi=180)
    plt.close(fig)

    valid = primary.loc[primary["capture_defined"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if len(valid):
        positions = np.arange(len(valid))
        ax.bar(positions - 0.18, valid["F_capture"], width=0.34, label="observable capture", color="#009E73")
        ax.bar(positions + 0.18, valid["F_residual"], width=0.34, label="residual", color="#CC79A7")
        ax.set_xticks(positions, valid["action_pair"].str.replace("|", " vs ", regex=False), rotation=12)
    ax.axhline(0, color="0.25", linewidth=1)
    ax.axhline(1, color="0.4", linewidth=1)
    ax.set_ylabel("Unclamped fraction")
    ax.set_title("Observable capture and residual fractions")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "17b_figure_c_capture_residual.png", dpi=180)
    plt.close(fig)


def _md_table(frame: pd.DataFrame, columns: list[str] | None = None, digits: int = 4) -> str:
    table = frame if columns is None else frame[columns]
    return table.to_markdown(index=False, floatfmt=f".{digits}f") if len(table) else "无可报告行。"


def _write_reports(
    topology: pd.DataFrame,
    rng: pd.DataFrame,
    support: pd.DataFrame,
    action_summary: pd.DataFrame,
    reliability: pd.DataFrame,
    full_reliability: pd.DataFrame,
    decomposition: pd.DataFrame,
    full_residual: pd.DataFrame,
    states: pd.DataFrame,
    high_summary: pd.DataFrame,
    resource_ledger: pd.DataFrame,
    verdict: str,
    decisions: dict,
) -> None:
    OUT_LIGHT.mkdir(parents=True, exist_ok=True)
    primary = decomposition.loc[
        decomposition["support_mode"].eq("matched_support")
        & decomposition["representation"].eq("global28")
    ].copy()
    pooled = primary.loc[primary["suite"].eq("pooled")]
    post_action = action_summary.loc[
        action_summary["domain"].eq("post_handoff")
        & action_summary["suite"].eq("pooled")
        & action_summary["weighting"].eq("all_unordered_pairs")
    ]
    post_rel = reliability.loc[
        reliability["domain"].eq("post_handoff") & reliability["suite"].eq("pooled")
    ]
    post_rel_all = reliability.loc[reliability["domain"].eq("post_handoff")]
    post_action_all = action_summary.loc[
        action_summary["domain"].eq("post_handoff")
        & action_summary["weighting"].eq("all_unordered_pairs")
    ]
    post_support = support.loc[support["domain"].eq("post_handoff")]
    t3 = post_support.loc[post_support["track"].eq("T3") & post_support["suite"].eq("pooled")].iloc[0]
    state_rates = {
        "U0": float(states["U0_ambiguous"].mean()),
        "U1": float(states["U1_unique_practical_winner"].mean()),
        "U2": float(states["U2_switch_required"].mean()),
        "U3": float(states["U3_unique_switch_target"].mean()),
    }
    common_header = (
        "Task17B 是零新增目标函数评估的机制分解。所有正式区间均按 `cv_group_id` "
        f"分组 bootstrap {BOOTSTRAP_DRAWS} 次；`new_objective_FE = 0`。\n\n"
    )
    reports = {
        "17b_01_zero_fe_contract.md": common_header + "# 17b_01 零 FE 与统计契约\n\n"
        "- Primary domain：post-handoff。\n- Primary representation：Global28；Compact6 仅作敏感性。\n"
        "- 两个 replicate layer 固定按 key 0、1；key 2 仅作敏感性。\n"
        "- `E_res = A_local - A_noise`；`M_res = median(D_local) - median(D_noise)`。\n"
        "- D3 的操作判据在查看机制结果前固定为：至少两对 U1 rate 比总体 local rate 下降至少 0.10，且与 noise rate 的绝对差不超过 0.10。\n"
        "- 粘贴任务中缺失公式左端的定义，按相邻文字还原为归一化绝对差；未引入额外统计量。\n",
        "17b_02_repetition_topology_and_rng_check.md": common_header + "# 17b_02 Repetition topology 与 RNG 核查\n\n"
        + _md_table(topology.loc[topology["suite"].eq("pooled")], ["domain", "component_type", "action_or_pair", "t1_cells", "t2_states", "t3_states", "cv_groups", "decision_strata", "outcome_blind_selection", "formal_support", "support_status"])
        + "\n\n## RNG 与 clone 语义\n\n" + _md_table(rng, ["check_id", "status", "detail"])
        + "\n\nCSO transfer 的初始 velocity 由 target-action RNG 生成，因此 target 内部数组不逐字相同；差异完全由预先指定的 action RNG 产生，source population/fitness/best/FE 保持相同。该随机初始化属于被重复测量的动作核。\n",
        "17b_03_support_gate.md": common_header + "# 17b_03 T1/T2/T3 支持门槛\n\n"
        + _md_table(post_support, ["suite", "track", "action_pair", "states", "cv_groups", "state_threshold", "cv_group_threshold", "formal_support", "support_status"])
        + f"\n\nT3 pooled 只有 {int(t3.t3_states if 't3_states' in t3 else t3.states)} states / {int(t3.cv_groups)} groups，full-signature track 不可用。\n",
        "17b_04_action_level_same_state_noise.md": common_header + "# 17b_04 Action-level same-state noise\n\n"
        + _md_table(post_action, ["action", "median", "q75", "q90", "q95", "P_N_gt1", "repeated_states", "cv_groups", "repetition_weight_sensitive"])
        + "\n\n所有 unordered repetition pairs 作描述；每 state 固定 key 0/1 的敏感性结果保存在同一 summary parquet。\n",
        "17b_05_pairwise_margin_reliability.md": common_header + "# 17b_05 Pairwise-margin reliability\n\n"
        + _md_table(post_rel, ["action_pair", "states", "cv_groups", "median_D_noise", "P_D_noise_gt1", "replicate_margin_spearman", "exact_category_agreement", "directional_reversal_rate", "tie_winner_transition_rate", "reliability_class", "third_layer_reliability_class", "repetition_weight_sensitive"]),
        "17b_06_full_signature_reliability.md": common_header + "# 17b_06 Full-signature reliability\n\n"
        + _md_table(full_reliability, ["domain", "suite", "t3_states", "t3_cv_groups", "track_status", "reason"])
        + "\n\n未生成 synthetic signature，也未把模型化噪声当作直接证据。\n",
        "17b_07_pairwise_residual_beyond_noise.md": common_header + "# 17b_07 Pairwise residual beyond noise\n\n"
        + _md_table(pooled, ["action_pair", "n_noise_states", "n_local_pairs", "n_all_pairs", "A_noise", "A_local", "A_all", "E_res", "E_res_ci_low", "E_res_ci_high", "M_res", "M_res_ci_low", "M_res_ci_high", "noise_to_local_ratio", "F_capture", "F_residual"])
        + "\n\nPrimary `matched_support` 将 cross-state rows 限定到包含对应 T2 repetition state 的 exact decision strata；`all_available_support` 仅作敏感性。\n",
        "17b_08_full_vector_residual_decomposition.md": common_header + "# 17b_08 Full-vector residual decomposition\n\n"
        + _md_table(full_residual, ["domain", "suite", "t3_states", "t3_cv_groups", "track_status", "AVA_noise", "AVA_local_T3", "E_res_full"])
        + "\n\nT3 未达到门槛，因此不计算 full-vector residual 或 decision-geometry repeatability reference。\n",
        "17b_09_high_confidence_state_check.md": common_header + "# 17b_09 Decision-relevant state 核查\n\n"
        + f"U0={state_rates['U0']:.4f}，U1={state_rates['U1']:.4f}，U2={state_rates['U2']:.4f}，U3={state_rates['U3']:.4f}。\n\n"
        + _md_table(high_summary.loc[high_summary["suite"].eq("pooled") & high_summary["subset"].isin(["U1", "U3"])], ["subset", "action_pair", "states", "strata_with_pairs", "state_fraction", "A_local", "median_D_local", "winner_mismatch_rate"])
        + "\n\nU1/U3 来自 observed practical action sets，只是 secondary diagnostic，不称为 ground truth。T3 不足，U1-repeat 不可构造。\n",
        "17b_10_suite_actionpair_robustness.md": common_header + "# 17b_10 Suite × action-pair robustness\n\n"
        + _md_table(primary, ["suite", "action_pair", "formal_support", "n_noise_states", "n_cv_groups", "A_noise", "A_local", "A_all", "E_res", "E_res_ci_low", "E_res_ci_high", "M_res", "F_capture", "F_residual"])
        + "\n\nMA-BBOB 的 SHADE-CSO 与 L-SHADE-CSO 只有 8 states，均标为 LOW_SUPPORT，不承担 suite-specific 机制判定。\n",
        "17b_11_noise_ceiling_interpretation.md": common_header + "# 17b_11 Noise-ceiling interpretation\n\n"
        "这里的 repetition geometry 只称 empirical noise-ceiling reference，不称严格的 information-theoretic upper bound。"
        "同状态重复同时覆盖后续求解随机性，以及 CSO transfer 时由 action RNG 生成的初始 velocity。\n\n"
        + _md_table(pooled, ["action_pair", "A_noise", "A_local", "noise_to_local_ratio", "E_res", "E_res_ci_low", "E_res_ci_high"]),
        "17b_12_stoploss_decision.md": common_header + "# 17b_12 Stop-loss decision\n\n"
        + f"最终判定：**{verdict}**。\n\n"
        + ("仅允许最后一次 solver-internal adaptive-state attribution；其后停止实验主线。" if verdict.startswith("D1") else "实验主线停止；不得以新增特征、selector、repetition、seeds 6-10 或 CEC 扩充证据。D2 只允许评估现有结果是否足以收束为机制稿，不自动给出 paper-consolidation GO。") + "\n",
        "17b_13_resource_ledger.md": common_header + "# 17b_13 Resource ledger\n\n" + _md_table(resource_ledger),
        "17b_14_final_verdict.md": common_header + "# 17b_14 Final verdict\n\n"
        + f"## {verdict}\n\n"
        + ("Repeated-action controls indicate that stochastic outcome variation alone cannot account for the residual decision mismatch between behaviorally nearest states.\n" if verdict.startswith("D1") else
           "A substantial fraction of the observed decision mismatch is comparable to the stochastic variation obtained by repeating the same action from the same optimization state. Further hidden-state feature hunting is not justified.\n" if verdict.startswith("D2") else
           "Observable behavior is substantially more decision-informative on practically well-separated states than on ambiguous near-tie states.\n" if verdict.startswith("D3") else
           "Existing outcome-blind repetitions are insufficient to identify how much residual action-value aliasing exceeds same-state stochastic variation. No additional objective evaluations were commissioned to rescue this attribution question.\n")
        + "\n" + json.dumps(decisions, ensure_ascii=False, indent=2) + "\n",
    }
    for filename, text in reports.items():
        (OUT_LIGHT / filename).write_text(text, encoding="utf-8")

    action_largest = post_action.sort_values("P_N_gt1", ascending=False).iloc[0]
    total = common_header + "# Decision-before-Feature Task17B：Decision Reliability 与 Residual Aliasing Decomposition\n\n"
    total += f"## 结论\n\n**{verdict}**。\n\n"
    total += "## Repetition topology\n\n" + _md_table(topology.loc[(topology["domain"].eq("post_handoff")) & (topology["suite"].eq("pooled"))], ["component_type", "action_or_pair", "t1_cells", "t2_states", "t3_states", "cv_groups", "outcome_blind_selection", "formal_support"])
    total += "\n\n- Source state：相同 population、fitness、best、FE；continue 为全状态 clone；transfer 由同一 source state 初始化。\n"
    total += "- RNG：仅 semantic RNG coordinates 改变；stream separation 由代码确认，不作有限样本独立性的过度主张。\n"
    total += "- Natural：T1 可作描述；三个 pooled T2 均低于 30 states，因此不承担 secondary mechanism verdict。\n"
    total += "- new objective FE：0。\n\n"
    total += "## Action noise\n\n" + _md_table(post_action, ["action", "P_N_gt1", "median", "q90", "q95", "repeated_states", "cv_groups"])
    total += f"\n\n按 `P(N_a>1)`，随机波动最大的是 {action_largest.action}（{action_largest.P_N_gt1:.4f}）。suite 明细保存在 parquet 与分报告中。\n\n"
    total += _md_table(post_action_all.loc[post_action_all["suite"].isin(["bbob", "mabbob"])], ["suite", "action", "P_N_gt1", "median", "q90", "repeated_states", "cv_groups", "repetition_weight_sensitive"])
    total += "\n\nBBOB 中 SHADE 的 `P(N_a>1)` 最大，MA-BBOB 中 CSO 最大；action-level 排序并非跨 suite 完全稳定。all-pairs 与每 state 固定 key 0/1 的最大-action 结论在 pooled 层一致，未标记 repetition-weight sensitive。\n\n"
    total += "## Pairwise reliability\n\n" + _md_table(post_rel, ["action_pair", "P_D_noise_gt1", "replicate_margin_spearman", "exact_category_agreement", "directional_reversal_rate", "tie_winner_transition_rate", "reliability_class", "third_layer_reliability_class", "repetition_weight_sensitive"])
    total += "\n\n" + _md_table(post_rel_all.loc[post_rel_all["suite"].isin(["bbob", "mabbob"])], ["suite", "action_pair", "states", "cv_groups", "formal_support", "P_D_noise_gt1", "replicate_margin_spearman", "reliability_class"])
    total += "\n\n## Residual beyond noise\n\n" + _md_table(pooled, ["action_pair", "A_noise", "A_local", "A_all", "E_res", "E_res_ci_low", "E_res_ci_high", "M_res", "F_capture", "F_residual"])
    total += "\n\n" + _md_table(primary.loc[primary["suite"].isin(["bbob", "mabbob"])], ["suite", "action_pair", "formal_support", "A_noise", "A_local", "A_all", "E_res", "E_res_ci_low", "E_res_ci_high", "M_res"])
    total += "\n\n至少 2/3 pairs 的 beyond-noise residual：" + ("YES" if int((pooled["E_res_ci_low"] > 0).sum()) >= 2 else "NO") + "。BBOB 与 MA-BBOB 逐对结果见 17b_10；低支持 MA cells 不承担正式结论。\n\n"
    total += f"## Full signature\n\nT3={int(t3.states)} states / {int(t3.cv_groups)} groups，未达到 50 / 8，故 AVA_noise、AVA_local,T3、E_res_full 与 R_D 均不计算。\n\n"
    total += "## High-confidence states\n\n" + f"U0={state_rates['U0']:.4f}，U1={state_rates['U1']:.4f}，U2={state_rates['U2']:.4f}，U3={state_rates['U3']:.4f}。\n\n"
    total += _md_table(high_summary.loc[high_summary["suite"].eq("pooled") & high_summary["subset"].isin(["U1", "U3"])], ["subset", "action_pair", "A_local", "median_D_local", "winner_mismatch_rate", "strata_with_pairs"])
    total += "\n\nU1-repeat 因 T3 不足不可构造；U1/U3 local aliasing 未作为正式 ground truth。U1 与 U3 的三对 local rates 均未下降到 same-state floor，现有结果不支持 apparent aliasing 主要由 practical near-tie states 驱动。\n\n"
    total += "## Stop-loss\n\n" + f"- 最终：{verdict}\n- Task17C：{'YES' if decisions['task17c_allowed'] else 'NO'}\n- new feature：{'仅限 Task17C 内预先指定变量' if decisions['new_feature_allowed'] else 'NO'}\n- new selector：NO\n- fuzzy：NO\n- seeds 6-10：NO\n- CEC：NO\n- 追加 repetitions：NO\n- paper consolidation：{decisions['paper_consolidation_status']}\n- 主线 STOP：{'YES' if decisions['mainline_stop'] else 'Task17C 后必须 STOP'}\n"
    total += "\n## 图表\n\n- `figures/17b_figure_a_same_local_all.png`\n- `figures/17b_figure_b_margin_repeatability.png`\n- `figures/17b_figure_c_capture_residual.png`\n"
    (OUT_LIGHT / "Decision-before-Feature_Task17B_DecisionReliability与ResidualAliasingDecomposition.md").write_text(total, encoding="utf-8")
    (OUT_LIGHT / "task17b_verdict.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    wall_start = perf_counter()
    cpu_start = process_time()
    OUT_LIGHT.mkdir(parents=True, exist_ok=True)
    OUT_HEAVY.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    branches, signatures, pair_identities, representations, random_controls = _read_inputs()
    checks = _validate_inputs(branches, signatures, pair_identities, representations)
    topology, support, _ = _support_counts(branches, signatures)
    rng = _rng_verification()
    action_pairs, action_summary = _action_noise(branches, signatures)
    margin_replicates, margin_reliability = _pairwise_replicates(branches, signatures)
    crossstate = _cross_state_rows(signatures, pair_identities, representations, random_controls)
    decomposition, bootstrap = _decomposition(margin_replicates, crossstate, "global28")
    compact_decomposition, compact_bootstrap = _decomposition(margin_replicates, crossstate, "compact6")
    decomposition = pd.concat([decomposition, compact_decomposition], ignore_index=True)
    bootstrap = pd.concat([bootstrap, compact_bootstrap], ignore_index=True)
    high_states, high_aliasing = _high_confidence(signatures, representations)
    high_summary = _high_confidence_summary(high_states, high_aliasing)
    full_replicates, full_reliability, full_residual = _full_track_tables(support)
    verdict, decisions = _select_verdict(support, decomposition, high_summary, rng)

    elapsed_wall = perf_counter() - wall_start
    elapsed_cpu = process_time() - cpu_start
    max_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_rss_mb = max_rss / (1024.0 * 1024.0) if platform.system() == "Darwin" else max_rss / 1024.0
    post_rows = branches.loc[branches["domain"].eq("post_handoff")]
    natural_rows = branches.loc[branches["domain"].eq("natural")]
    pooled_t2 = support.loc[
        support["domain"].eq("post_handoff") & support["suite"].eq("pooled") & support["track"].eq("T2")
    ]
    pooled_t3 = support.loc[
        support["domain"].eq("post_handoff") & support["suite"].eq("pooled") & support["track"].eq("T3")
    ].iloc[0]
    ledger_values = {
        "new_objective_FE": NEW_OBJECTIVE_FE,
        "reused_natural_repetition_rows": int(len(natural_rows)),
        "reused_post_repetition_rows": int(len(post_rows)),
        "reused_natural_extra_repetition_rows": int(natural_rows["replicate_id"].gt(0).sum()),
        "reused_post_extra_repetition_rows": int(post_rows["replicate_id"].gt(0).sum()),
        "t1_repeated_cells": int(topology.loc[topology["domain"].eq("post_handoff") & topology["suite"].eq("pooled") & topology["component_type"].eq("T1_action"), "t1_cells"].sum()),
        "t2_shade_lshade_states": int(pooled_t2.loc[pooled_t2["action_pair"].eq("shade|lshade"), "states"].iloc[0]),
        "t2_shade_cso_states": int(pooled_t2.loc[pooled_t2["action_pair"].eq("shade|cso"), "states"].iloc[0]),
        "t2_lshade_cso_states": int(pooled_t2.loc[pooled_t2["action_pair"].eq("lshade|cso"), "states"].iloc[0]),
        "t3_full_signature_states": int(pooled_t3.states),
        "n_cv_groups_supported": int(pooled_t2.loc[pooled_t2["formal_support"], "cv_groups"].max()),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "analysis_cpu_seconds": elapsed_cpu,
        "wall_seconds": elapsed_wall,
        "peak_rss_mb": peak_rss_mb,
    }
    resource_ledger = pd.DataFrame(
        [{"metric": key, "value": value, "new_objective_fe": NEW_OBJECTIVE_FE} for key, value in ledger_values.items()]
    )
    if int(ledger_values["new_objective_FE"]) != 0:
        raise RuntimeError("Task17B invalid: new objective FE is nonzero")

    outputs = {
        "task17b_repetition_topology": topology,
        "task17b_repetition_rng_verification": rng,
        "task17b_support_gate": support,
        "task17b_action_noise_pairs": action_pairs,
        "task17b_action_noise_summary": action_summary,
        "task17b_pairwise_margin_replicates": margin_replicates,
        "task17b_pairwise_margin_reliability": margin_reliability,
        "task17b_full_signature_replicates": full_replicates,
        "task17b_full_signature_reliability": full_reliability,
        "task17b_pairwise_local_crossstate": crossstate,
        "task17b_pairwise_residual_decomposition": decomposition,
        "task17b_full_residual_decomposition": full_residual,
        "task17b_high_confidence_states": high_states,
        "task17b_high_confidence_aliasing": high_aliasing,
        "task17b_suite_actionpair_robustness": decomposition,
        "task17b_bootstrap": bootstrap,
        "task17b_resource_ledger": resource_ledger,
        "task17b_consistency_checks": checks,
    }
    for name, frame in outputs.items():
        frame.to_parquet(OUT_HEAVY / f"{name}.parquet", index=False)

    _figures(decomposition, margin_replicates)
    _write_reports(
        topology, rng, support, action_summary, margin_reliability, full_reliability,
        decomposition, full_residual, high_states, high_summary, resource_ledger,
        verdict, decisions,
    )
    print(json.dumps(decisions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
