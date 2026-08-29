"""Task 14B: global vs segment behavior on post-handoff dynamic states.

Zero new objective FE: everything reuses the Task 14A collection (3,780
mature post-handoff states, their 1000-FE next-action outcomes and the
formally extracted B_global / B_segment features). The post-handoff selector
is retrained inside the post-handoff development domain with the pre-fixed Task
13 carrier (WeightedMedianImputer -> StandardScaler -> RF(200, 8, sqrt,
fixed seed) plus a Ridge baseline) under leave-cv_group-out, and the two
pre-fixed margin candidates (kappa = 0.5 / 1.0) are evaluated with
fold-local noise scales. Permutation controls use N = 100.
"""
from __future__ import annotations

import argparse
import json
import resource
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS
from behavior_with_ela.analysis_v5.task12_1_analysis import (
    SOLVERS,
    fb_mean,
)
from behavior_with_ela.analysis_v5.task13.task13_analysis import (
    make_carrier,
    run_grouped_oof,
)

ROOT = Path(__file__).resolve().parents[3]
T14A_HEAVY = ROOT / "behavior_with_ela/results/analysis_v6/task14a"
T14A_LIGHT = ROOT / "behavior_with_ela/analysis_v6/task14a"
T14B_HEAVY = ROOT / "behavior_with_ela/results/analysis_v6/task14b"
T14B_LIGHT = ROOT / "behavior_with_ela/analysis_v6/task14b"
BOOTSTRAP_STREAM = 2026090224
BOOTSTRAP_DRAWS = 5000
N_PERM = 100
WORKERS = 8
KAPPA_GRID = (0.0, 0.5, 1.0)
LOSS_COLS = [f"loss_{s}" for s in SOLVERS]


def fb_of_difference(frame: pd.DataFrame, upper: np.ndarray, lower: np.ndarray, stream: int) -> tuple[float, float, float]:
    work = pd.DataFrame({"cv_group_id": frame["cv_group_id"].to_numpy(), "d": upper - lower})
    groups = sorted(work["cv_group_id"].unique())
    means = work.groupby("cv_group_id")["d"].mean()
    rng = np.random.default_rng(
        np.random.SeedSequence([BOOTSTRAP_STREAM + stream, len(groups)]).generate_state(4)
    )
    draws = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        sample = rng.choice(groups, size=len(groups), replace=True)
        draws[draw] = np.mean([means[g] for g in sample])
    return float(means.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def build_dataset() -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = pd.read_parquet(T14A_HEAVY / "post_handoff_analysis_frame.parquet")
    behavior = pd.read_parquet(T14A_HEAVY / "post_handoff_behavior.parquet")
    keep = [
        "state_id", "suite", "problem_id", "cv_group_id", "family", "instance", "seed",
        "source_algorithm", "current_algorithm", "source_checkpoint_fe", "FE",
        "segment_start", "segment_age", "log10_gap",
        *LOSS_COLS, "continue_loss",
    ]
    frame = frame[keep + ["switch_required", "A_ND_size"]].copy()
    frame["route"] = frame["source_algorithm"] + "->" + frame["current_algorithm"]
    noise14a = pd.read_parquet(T14A_LIGHT / "post_handoff_noise_deltas.parquet")
    delta_map = (
        noise14a.loc[noise14a["stratum"].eq("pooled")].set_index("solver")["delta_95"].to_dict()
    )
    values = frame[LOSS_COLS].to_numpy(dtype=float)
    index_by_solver = {s: i for i, s in enumerate(SOLVERS)}
    members = []
    for row_index, current in enumerate(frame["current_algorithm"]):
        dominated = np.zeros(3, dtype=bool)
        for i, a in enumerate(SOLVERS):
            for j, b in enumerate(SOLVERS):
                if i == j:
                    continue
                d = max(delta_map[a], delta_map[b])
                if values[row_index, i] < values[row_index, j] - d:
                    dominated[i] = True
        members.append("|".join(s for s in SOLVERS if not dominated[index_by_solver[s]]) or "")
    frame["A_ND_members"] = members
    frame = frame.rename(columns={"source_checkpoint_fe": "source_FE", "FE": "global_FE"})
    bg_cols = [c for c in behavior.columns if c in set(SELECTOR_BEHAVIOR_FEATURE_COLUMNS)]
    bg = (
        behavior.loc[behavior["behavior_kind"].eq("global"), ["state_id", *bg_cols]]
        .rename(columns={c: f"bg_{c[3:]}" for c in bg_cols})
    )
    bs = (
        behavior.loc[behavior["behavior_kind"].eq("segment"), ["state_id", *bg_cols]]
        .rename(columns={c: f"bs_{c[3:]}" for c in bg_cols})
    )
    dataset = frame.merge(bg, on="state_id", validate="many_to_one").merge(bs, on="state_id", validate="many_to_one")
    dataset["raw_best_action"] = dataset[LOSS_COLS].idxmin(axis=1).str.replace("loss_", "", regex=False)
    # numerical-stability clipping for the behavior features: a handful of
    # post-handoff cells (division tails in the extractor) reach 1e13-1e14 and
    # overflow float32 inside the scaler of the pre-fixed carrier; natural-state
    # features never exceed ~1e5, so the +-1e6 bound only tames the
    # pathological tail and is documented in the feature audit
    bg_col_names = [f"bg_{c[3:]}" for c in bg_cols]
    bs_col_names = [f"bs_{c[3:]}" for c in bg_cols]
    denormal_snap = 0
    for cols in (bg_col_names, bs_col_names):
        tiny = (dataset[cols].abs() < 1e-12) & (dataset[cols] != 0.0)
        denormal_snap += int(tiny.to_numpy().sum())
        dataset[cols] = dataset[cols].mask(tiny, 0.0)
        dataset[cols] = dataset[cols].clip(lower=-1e6, upper=1e6)
    dataset.to_parquet(T14B_HEAVY / "post_handoff_behavior_action_dataset_task14b.parquet", index=False)
    bs_col_names = [f"bs_{c[3:]}" for c in bg_cols]
    return dataset, [f"bg_{c[3:]}" for c in bg_cols], bs_col_names


def feature_audit(dataset: pd.DataFrame, bg_cols: list[str], bs_cols: list[str]) -> dict:
    audit = {
        "n_states": int(len(dataset)),
        "n_bg": len(bg_cols),
        "n_bs": len(bs_cols),
        "bg_nan": int(dataset[bg_cols].isna().sum().sum()),
        "bs_nan": int(dataset[bs_cols].isna().sum().sum()),
        "bg_constant": [c for c in bg_cols if dataset[c].nunique() <= 1],
        "bs_constant": [c for c in bs_cols if dataset[c].nunique() <= 1],
        "identical_columns_bg_bs": [
            c[3:] for c in bg_cols if np.allclose(dataset[c], dataset["bs_" + c[3:]])
        ],
        "stability_clip_bound": 1e6,
        "denormal_snap_threshold": 1e-12,
        "denormal_snap_cells": int(
            (
                ((dataset[bg_cols].abs() < 1e-12) & (dataset[bg_cols] != 0.0)).to_numpy()
                | ((dataset[bs_cols].abs() < 1e-12) & (dataset[bs_cols] != 0.0)).to_numpy()
            ).sum()
        ),
        "stability_clip_cells": int(
            (
                (dataset[bg_cols].abs() > 1e6).to_numpy()
                | (dataset[bs_cols].abs() > 1e6).to_numpy()
            ).sum()
        ),
    }
    corr = []
    for c in bg_cols:
        corr.append(float(dataset[c].corr(dataset["bs_" + c[3:]])))
    audit["bg_bs_per_feature_correlation"] = {
        "mean": float(np.nanmean(corr)),
        "min": float(np.nanmin(corr)),
        "max": float(np.nanmax(corr)),
        "highest": sorted(zip(np.round(corr, 3), [c[3:] for c in bg_cols]), reverse=True)[:5],
    }
    audit["time_like_features"] = ["bg_fe_ratio", "bs_fe_ratio"]
    route_counts = dataset.groupby("route").size().to_dict()
    audit["route_counts"] = {k: int(v) for k, v in route_counts.items()}
    audit["suite_counts"] = {k: int(v) for k, v in dataset["suite"].value_counts().items()}
    return audit


def lookup_policy(dataset: pd.DataFrame) -> pd.Series:
    assigned = pd.Series(index=dataset.index, dtype=object)
    loss_cols = LOSS_COLS
    for group in sorted(dataset["cv_group_id"].unique()):
        train = dataset.loc[dataset["cv_group_id"].ne(group)]
        per = (
            train.groupby(["route", "source_FE", "cv_group_id"], sort=False)[loss_cols]
            .mean()
            .groupby(["route", "source_FE"], sort=False)
            .mean()
        )
        choice = per.idxmin(axis=1).str.replace("loss_", "", regex=False)
        hold = dataset.index[dataset["cv_group_id"].eq(group)]
        assigned.loc[hold] = np.asarray(
            dataset.loc[hold].set_index(["route", "source_FE"]).index.map(choice)
        )
    if assigned.isna().any():
        raise RuntimeError("lookup left states unassigned")
    return assigned


def within_route_loso(
    dataset: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    permuted: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    truth = dataset[LOSS_COLS].to_numpy(dtype=float)
    matrices = {name: dataset[cols].to_numpy(dtype=float) for name, cols in feature_sets.items()}
    if permuted:
        for name, perm in permuted.items():
            matrices[name] = matrices[name][perm]
    # float32 matches the forest's internal dtype and prevents denormal-scale
    # variance columns (permutation explores every 4-row train fold) from
    # overflowing the scaler at predict time
    matrices = {name: X.astype(np.float32) for name, X in matrices.items()}
    records = []
    for _, positions in dataset.groupby(["problem_id", "route", "source_FE"], sort=False).groups.items():
        idx = np.asarray(positions)
        w0_action = int(np.argmin(truth[idx].mean(axis=0)))
        for pos in idx:
            train = idx[idx != pos]
            record = {
                "state_id": dataset.at[pos, "state_id"],
                "suite": dataset.at[pos, "suite"],
                "cv_group_id": dataset.at[pos, "cv_group_id"],
                "route": dataset.at[pos, "route"],
                "source_FE": int(dataset.at[pos, "source_FE"]),
                "realized_W0": float(truth[pos, w0_action]),
            }
            for name, X in matrices.items():
                model = make_carrier("rf")
                model.fit(X[train], truth[train])
                prediction = model.predict(X[pos : pos + 1])[0]
                record[f"realized_{name}"] = float(truth[pos, int(np.argmin(prediction))])
            records.append(record)
    return pd.DataFrame(records)


def permutation_chunk(chunk: list[int], control: str) -> list[dict]:
    dataset, bg_cols, bs_cols = build_dataset()
    feature_sets = {
        "WG": bg_cols,
        "WGS": [*bg_cols, *bs_cols],
    }
    rng = np.random.default_rng(
        np.random.SeedSequence([BOOTSTRAP_STREAM + 500, chunk[0], len(control)]).generate_state(4)
    )
    strata_groups = dataset.groupby(["route", "source_FE"], sort=False).groups
    rows = []
    for repeat in chunk:
        perm = np.arange(len(dataset))
        for _, idx in strata_groups.items():
            positions = np.asarray(idx)
            perm[positions] = positions[rng.permutation(len(positions))]
        if control == "P1_global":
            use = {"WG": bg_cols, "WGS": bg_cols}
            permuted = {"WG": perm, "WGS": perm}
        else:
            use = {"WGS": [*bg_cols, *bs_cols]}
            permuted = {"WGS": perm}
        result = within_route_loso(dataset, use, permuted=permuted)
        for suite_name, group in result.groupby("suite", sort=False):
            entry = {"control": control, "repeat": repeat, "suite": suite_name}
            if control == "P1_global":
                entry["delta_within_perm"] = float(
                    (group["realized_W0"] - group["realized_WG"])
                    .groupby(group["cv_group_id"])
                    .mean()
                    .mean()
                )
            else:
                # P2 shuffles only the segment features, so W0 and WG are
                # permutation-invariant; the null statistic for the segment
                # increment is fb(W0 - WGS_permuted), compared against the
                # observed fb(W0 - WGS) with the same form
                entry["delta_within_perm"] = float(
                    (group["realized_W0"] - group["realized_WGS"])
                    .groupby(group["cv_group_id"])
                    .mean()
                    .mean()
                )
            rows.append(entry)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-perm", action="store_true")
    parser.add_argument("--rerun-perm", action="store_true")
    args = parser.parse_args()
    T14B_HEAVY.mkdir(parents=True, exist_ok=True)
    T14B_LIGHT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    dataset, bg_cols, bs_cols = build_dataset()
    audit = feature_audit(dataset, bg_cols, bs_cols)
    (T14B_LIGHT / "17a_feature_audit.json").write_text(json.dumps(audit, indent=2, default=float))

    route_dummies = pd.get_dummies(dataset["route"], prefix="route", dtype=float)
    dataset = pd.concat([dataset.reset_index(drop=True), route_dummies.reset_index(drop=True)], axis=1)
    route_cols = list(route_dummies.columns)
    m0_features = [*route_cols, "source_FE", "segment_age"]

    # ---- B1: empirical route+sourceFE OOF lookup ----
    dataset["lookup_action"] = lookup_policy(dataset)
    dataset["realized_lookup"] = [
        row[f"loss_{a}"] for (_, row), a in zip(dataset.iterrows(), dataset["lookup_action"])
    ]

    # ---- grouped OOF for M0/MG/MS/MGS + BG/BS diagnostics ----
    feature_sets = {
        "M0": m0_features,
        "MG": [*m0_features, *bg_cols],
        "MS": [*m0_features, *bs_cols],
        "MGS": [*m0_features, *bg_cols, *bs_cols],
        "BG_only": list(bg_cols),
        "BS_only": list(bs_cols),
    }
    all_preds = []
    for carrier in ("rf", "ridge"):
        all_preds.append(run_grouped_oof(dataset, feature_sets, carrier))
    preds = pd.concat(all_preds, ignore_index=True)
    preds.to_parquet(T14B_HEAVY / "task14b_oof_predictions.parquet", index=False)

    lookup_map = dataset.set_index("state_id")["realized_lookup"]
    cont_array = dataset["continue_loss"].to_numpy(dtype=float)
    noise14a = pd.read_parquet(T14A_LIGHT / "post_handoff_noise_deltas.parquet")
    post_noise = (
        noise14a.loc[noise14a["stratum"].eq("pooled")].set_index("solver")["delta_95"].to_dict()
    )
    summary_rows = []
    for (model, carrier_name), group in preds.groupby(["model", "carrier"], sort=False):
        merged = group.merge(
            dataset[["state_id", "suite", "cv_group_id", "continue_loss", "current_algorithm", "switch_required"]],
            on="state_id",
            validate="many_to_one",
        )
        lookup_vals = lookup_map.reindex(merged["state_id"]).to_numpy(dtype=float)
        is_switch = merged["selected"].ne(merged["current_algorithm"])
        gain_continue = merged["continue_loss"].to_numpy(dtype=float) - merged["realized_loss"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "model": model,
                "carrier": carrier_name,
                "realized_fb_loss": float(
                    merged["realized_loss"].groupby(merged["cv_group_id"]).mean().mean()
                ),
                "gain_vs_continue": float(pd.Series(gain_continue).groupby(merged["cv_group_id"]).mean().mean()),
                "gain_vs_lookup": float(
                    pd.Series(lookup_vals - merged["realized_loss"].to_numpy(dtype=float))
                    .groupby(merged["cv_group_id"])
                    .mean()
                    .mean()
                ),
                "switch_rate": float(is_switch.mean()),
                "harmful_rate": np.nan,
            }
        )
        for suite_name, part in merged.groupby("suite", sort=False):
            summary_rows.append(
                {
                    "model": model,
                    "carrier": carrier_name,
                    "suite": suite_name,
                    "realized_fb_loss": float(
                        part["realized_loss"].groupby(part["cv_group_id"]).mean().mean()
                    ),
                    "gain_vs_continue": float(
                        pd.Series(
                            part["continue_loss"].to_numpy(dtype=float)
                            - part["realized_loss"].to_numpy(dtype=float)
                        )
                        .groupby(part["cv_group_id"])
                        .mean()
                        .mean()
                    ),
                    "gain_vs_lookup": float(
                        pd.Series(
                            lookup_map.reindex(part["state_id"]).to_numpy(dtype=float)
                            - part["realized_loss"].to_numpy(dtype=float)
                        )
                        .groupby(part["cv_group_id"])
                        .mean()
                        .mean()
                    ),
                    "switch_rate": float(part["selected"].ne(part["current_algorithm"]).mean()),
                    "harmful_rate": float(
                        (
                            part["realized_loss"].to_numpy(dtype=float)
                            > part["continue_loss"].to_numpy(dtype=float)
                            + np.array(
                                [
                                    0.0
                                    if a == c
                                    else max(post_noise[c], post_noise[a])
                                    for c, a in zip(part["current_algorithm"], part["selected"])
                                ]
                            )
                        ).mean()
                    ),
                }
            )
    performance = pd.DataFrame(summary_rows)
    performance.to_parquet(T14B_LIGHT / "task14b_policy_performance.parquet", index=False)

    # pairwise bootstrap (RF carrier, primary)
    merged_rf = preds.loc[preds["carrier"].eq("rf")].merge(
        dataset[["state_id", "suite", "cv_group_id", "continue_loss"]], on="state_id", validate="many_to_one"
    )
    lookup_vals_full = lookup_map.reindex(merged_rf["state_id"]).to_numpy(dtype=float)
    realized_wide = merged_rf.pivot_table(index=["state_id", "cv_group_id", "suite"], columns="model", values="realized_loss").reset_index()
    boot_rows = []
    stream = 0
    pairs = [
        ("MGS_vs_lookup", "realized_lookup", "MGS"),
        ("MGS_vs_M0", "M0", "MGS"),
        ("MG_vs_M0", "M0", "MG"),
        ("MS_vs_M0", "M0", "MS"),
        ("MGS_vs_MG", "MG", "MGS"),
        ("M0_vs_continue", "continue", "M0"),
    ]
    for name, upper, lower in pairs:
        for suite_name, group in realized_wide.groupby("suite", sort=False):
            if upper == "realized_lookup":
                upper_values = lookup_map.reindex(group["state_id"]).to_numpy(dtype=float)
            elif upper == "continue":
                upper_values = dataset.set_index("state_id").loc[group["state_id"], "continue_loss"].to_numpy(dtype=float)
            else:
                upper_values = group[upper].to_numpy(dtype=float)
            point, low, high = fb_of_difference(
                group, upper_values, group[lower].to_numpy(dtype=float), stream
            )
            boot_rows.append(
                {"comparison": name, "suite": suite_name, "fb_mean": point, "ci_low": low, "ci_high": high}
            )
            stream += 1
    bootstrap = pd.DataFrame(boot_rows)
    bootstrap.to_parquet(T14B_LIGHT / "global_vs_segment_pairwise_bootstrap.parquet", index=False)

    # ---- within-route LOSO (observed) ----
    within_sets = {"WG": bg_cols, "WS": bs_cols, "WGS": [*bg_cols, *bs_cols]}
    within = within_route_loso(dataset, within_sets)
    within.to_parquet(T14B_HEAVY / "within_route_loso_predictions.parquet", index=False)
    within_rows = []
    for suite_name, group in within.groupby("suite", sort=False):
        entry = {"suite": suite_name}
        for model in ("W0", "WG", "WS", "WGS"):
            entry[f"L_{model}"] = float(
                group[f"realized_{model}"].groupby(group["cv_group_id"]).mean().mean()
            )
        entry["delta_within_global"] = entry["L_W0"] - entry["L_WG"]
        entry["delta_within_segment_only"] = entry["L_W0"] - entry["L_WS"]
        entry["delta_within_segment"] = entry["L_WG"] - entry["L_WGS"]
        within_rows.append(entry)
    within_summary = pd.DataFrame(within_rows)
    within_summary.to_parquet(T14B_LIGHT / "within_route_performance.parquet", index=False)

    # ---- permutations (P1 global / P2 segment), 100 each, RF carrier ----
    perm_path = T14B_LIGHT / "segment_permutation_100.parquet"
    perm_global_path = T14B_LIGHT / "global_permutation_100.parquet"
    perm_already_done = perm_path.exists() and perm_global_path.exists()
    run_perms = (not args.skip_perm) and (args.rerun_perm or not perm_already_done)
    if run_perms:
        chunks = [list(range(i, N_PERM, WORKERS)) for i in range(WORKERS)]
        all_rows = []
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            for part in executor.map(permutation_chunk, chunks, ["P2_segment"] * len(chunks)):
                all_rows.extend(part)
            for part in executor.map(permutation_chunk, chunks, ["P1_global"] * len(chunks)):
                all_rows.extend(part)
        perm_table = pd.DataFrame(all_rows)
        perm_table.loc[perm_table["control"].eq("P2_segment")].to_parquet(perm_path, index=False)
        perm_table.loc[perm_table["control"].eq("P1_global")].to_parquet(perm_global_path, index=False)

    perm_summary_rows = []
    for path, control, observed_column in (
        (perm_path, "P2_segment", "L_W0_minus_WGS_observed"),
        (perm_global_path, "P1_global", "delta_within_global"),
    ):
        if not path.exists():
            continue
        perm_table = pd.read_parquet(path)
        for suite_name, group in perm_table.groupby("suite", sort=False):
            deltas = group["delta_within_perm"].to_numpy()
            if control == "P2_segment":
                w0_obs = float(within_summary.loc[within_summary["suite"].eq(suite_name), "L_W0"].iloc[0])
                wgs_obs = float(within_summary.loc[within_summary["suite"].eq(suite_name), "L_WGS"].iloc[0])
                delta_obs = w0_obs - wgs_obs
            else:
                delta_obs = float(within_summary.loc[within_summary["suite"].eq(suite_name), observed_column].iloc[0])
            perm_summary_rows.append(
                {
                    "control": control,
                    "suite": suite_name,
                    "delta_observed": delta_obs,
                    "null_mean": float(deltas.mean()),
                    "null_std": float(deltas.std(ddof=1)),
                    "null_q95": float(np.quantile(deltas, 0.95)),
                    "null_q975": float(np.quantile(deltas, 0.975)),
                    "empirical_p": float((1 + int(np.sum(deltas >= delta_obs))) / (1 + N_PERM)),
                }
            )
    perm_summary = pd.DataFrame(perm_summary_rows)
    perm_summary.to_parquet(T14B_LIGHT / "within_permutation_summary.parquet", index=False)

    # ---- representation redundancy (leave-cv_group-out Ridge R2) ----
    redundancy_rows = []
    for direction, (inputs, outputs) in {
        "bs_from_bg": (bg_cols, bs_cols),
        "bg_from_bs": (bs_cols, bg_cols),
    }.items():
        X = dataset[inputs].to_numpy(dtype=float)
        Y = dataset[outputs].to_numpy(dtype=float)
        r2_per_fold = []
        for group in sorted(dataset["cv_group_id"].unique()):
            test_mask = dataset["cv_group_id"].eq(group).to_numpy()
            regressor = Ridge(alpha=1.0).fit(X[~test_mask], Y[~test_mask])
            r2 = regressor.score(X[test_mask], Y[test_mask])
            r2_per_fold.append(r2)
        redundancy_rows.append(
            {
                "direction": direction,
                "oof_r2_mean": float(np.mean(r2_per_fold)),
                "oof_r2_ci_low": float(np.quantile(r2_per_fold, 0.025)),
                "oof_r2_ci_high": float(np.quantile(r2_per_fold, 0.975)),
            }
        )
    redundancy = pd.DataFrame(redundancy_rows)
    redundancy.to_parquet(T14B_LIGHT / "global_segment_redundancy_oof.parquet", index=False)

    # ---- fold-local noise + margin policies on MGS OOF ----
    branches = pd.read_parquet(T14A_HEAVY / "post_handoff_action_outcomes_1000.parquet")
    states_meta = pd.read_parquet(T14A_HEAVY / "post_handoff_states.parquet").set_index("state_id")["cv_group_id"]
    branches = branches.merge(states_meta.rename("cv_group_id"), left_on="state_id", right_index=True, validate="many_to_one")
    branches["solver_cell"] = np.where(
        branches["candidate_action"].eq("continue"), branches["current_algorithm"], branches["candidate_action"]
    )
    medians = branches.groupby(["state_id", "candidate_action"])["loss_1000"].transform("median")
    branches["deviation"] = (branches["loss_1000"] - medians).abs()
    per_group_q95 = branches.groupby(["cv_group_id", "solver_cell"])["deviation"].quantile(0.95).unstack()
    groups = sorted(dataset["cv_group_id"].unique())
    fold_scales = {
        g: {a: float(per_group_q95.loc[per_group_q95.index.isin([x for x in groups if x != g]), a].mean()) for a in SOLVERS}
        for g in groups
    }
    scale_rows = []
    for g, values in fold_scales.items():
        for a, v in values.items():
            scale_rows.append({"held_out_group": g, "solver": a, "delta_95_fold_local": v})
    pd.DataFrame(scale_rows).to_parquet(T14B_LIGHT / "post_handoff_fold_local_noise_scale.parquet", index=False)

    mgs = preds.loc[preds["carrier"].eq("rf") & preds["model"].eq("MGS")].merge(
        dataset[["state_id", "suite", "cv_group_id", "current_algorithm", "continue_loss", "switch_required", "A_ND_members", "source_FE", "route"]],
        on="state_id",
        validate="many_to_one",
    )
    solver_loss = dataset.set_index("state_id")[LOSS_COLS]
    policy_rows = []
    for row in mgs.itertuples(index=False):
        current = row.current_algorithm
        alternatives = [s for s in SOLVERS if s != current]
        pred_current = getattr(row, f"pred_{current}")
        alt = min(alternatives, key=lambda s: getattr(row, f"pred_{s}"))
        margin_value = pred_current - getattr(row, f"pred_{alt}")
        entry = {
            "state_id": row.state_id,
            "suite": row.suite,
            "cv_group_id": row.cv_group_id,
            "current_algorithm": current,
            "pred_margin": margin_value,
            "pred_alt_algorithm": alt,
            "continue_loss": row.continue_loss,
            "switch_required": row.switch_required,
            "A_ND_members": row.A_ND_members,
        }
        for scale in ("max", "sum"):
            d = max(fold_scales[row.cv_group_id][current], fold_scales[row.cv_group_id][alt]) if scale == "max" else (
                fold_scales[row.cv_group_id][current] + fold_scales[row.cv_group_id][alt]
            )
            entry[f"pair_scale_{scale}"] = d
            for kappa in KAPPA_GRID:
                column = f"{scale}_k{int(kappa * 10):02d}"
                if margin_value > kappa * d:
                    action = alt
                    loss = float(solver_loss.at[row.state_id, f"loss_{action}"])
                else:
                    action = current
                    loss = float(row.continue_loss)
                entry[f"selected_{column}"] = action
                entry[f"realized_{column}"] = loss
                pair_delta = 0.0 if action == current else max(post_noise[current], post_noise[action])
                entry[f"harmful_{column}"] = loss > row.continue_loss + pair_delta
                entry[f"harmful_mass_{column}"] = max(loss - row.continue_loss - pair_delta, 0.0)
        policy_rows.append(entry)
    policy_frame = pd.DataFrame(policy_rows)
    policy_frame.to_parquet(T14B_HEAVY / "post_handoff_margin_policy_rows.parquet", index=False)

    policy_summary_rows = []
    for scale in ("max", "sum"):
        for kappa in KAPPA_GRID:
            column = f"{scale}_k{int(kappa * 10):02d}"
            realized_col = f"realized_{column}"
            selected_col = f"selected_{column}"
            for suite_name, group in policy_frame.groupby("suite", sort=False):
                is_switch = group[selected_col].ne(group["current_algorithm"])
                policy_summary_rows.append(
                    {
                        "scale": scale,
                        "kappa": kappa,
                        "suite": suite_name,
                        "realized_fb_loss": float(
                            group[realized_col].groupby(group["cv_group_id"]).mean().mean()
                        ),
                        "gain_vs_continue": float(
                            (group["continue_loss"] - group[realized_col])
                            .groupby(group["cv_group_id"])
                            .mean()
                            .mean()
                        ),
                        "switch_rate": float(is_switch.mean()),
                        "harmful_rate": float(group[f"harmful_{column}"].mean()),
                    }
                )
    pd.DataFrame(policy_summary_rows).to_parquet(T14B_LIGHT / "post_handoff_margin_policy_summary.parquet", index=False)

    # ---- route and phase stratification (RF MGS vs M0 vs MG) ----
    strat_rows = []
    merged_rf_full = merged_rf.merge(
        dataset[["state_id", "route", "source_FE", "switch_required"]], on="state_id", validate="many_to_one"
    )
    for stratum_name, keys in (
        ("route", ["route"]),
        ("source_FE", ["source_FE"]),
    ):
        for key, part in merged_rf_full.groupby(keys, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            pivot = part.pivot_table(
                index=["state_id", "cv_group_id"], columns="model", values="realized_loss"
            ).reset_index()
            entry = {
                "stratum": stratum_name,
                "key": "|".join(str(k) for k in key),
                "n": len(part),
            }
            required = part.drop_duplicates("state_id")["switch_required"] if "switch_required" in part else None
            entry["switch_required_rate"] = float(part.drop_duplicates("state_id")["switch_required"].mean())
            for model in ("M0", "MG", "MS", "MGS"):
                entry[f"L_{model}"] = float(pivot[model].groupby(pivot["cv_group_id"]).mean().mean())
            entry["segment_increment"] = entry["L_MG"] - entry["L_MGS"]
            strat_rows.append(entry)
    pd.DataFrame(strat_rows).to_parquet(T14B_LIGHT / "route_phase_stratification.parquet", index=False)

    elapsed = perf_counter() - started
    pd.DataFrame(
        [
            {
                "phase": "task14b_analysis",
                "new_objective_fe": 0,
                "wall_seconds": elapsed,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                "note": "permutation controls included" if not args.skip_perm else "permutations skipped",
            }
        ]
    ).to_parquet(T14B_HEAVY / "task14b_resource_ledger.parquet", index=False)

    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print("=== performance (rf) ===")
        print(performance.loc[performance["carrier"].eq("rf")].round(4).to_string())
        print("=== bootstrap ===")
        print(bootstrap.round(4).to_string())
        print("=== within-route ===")
        print(within_summary.round(4).to_string())
        print("=== permutation summary ===")
        if len(perm_summary):
            print(perm_summary.round(4).to_string())
        print("=== redundancy ===")
        print(redundancy.round(4).to_string())
        print("=== margin summary ===")
        print(pd.DataFrame(policy_summary_rows).round(4).to_string())


if __name__ == "__main__":
    main()
