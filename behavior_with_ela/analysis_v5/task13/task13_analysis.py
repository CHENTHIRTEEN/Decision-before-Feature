"""Task 13E-13R: behavior feature audit, behavior-action dataset, grouped
out-of-fold policy evaluation (M0 current+FE vs M1 behavior vs M2 combined,
plus M3/M4 auxiliary), within-problem leave-one-seed-out test, shuffle and
time-proxy controls, feature-contribution diagnostic and the non-deployable
problem-identity diagnostic.

Zero new action-label FE: the true action outcomes are the Task 12 stage-2
1000-FE solver-semantics loss matrix; the behavior features come from the
Task 13C deterministic replay. The deployment carrier is the formal project
pipeline WeightedMedianImputer -> StandardScaler -> RandomForestRegressor
(n_estimators=200, max_depth=8, max_features='sqrt', fixed random_state);
Ridge (alpha=1.0, same imputer/scaler) is the single allowed low-complexity
baseline. No hyperparameter search, no other models.

Pre-specified analysis constants: paired cv_group bootstrap (2000 draws) for
every incremental gain; leave-cv_group-out folds identical in philosophy to
the Task 12.1 L_current+FE^OOF baseline; shuffle controls use explicit
SeedSequence streams.
"""
from __future__ import annotations

import json
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS
from behavior_with_ela.analysis_v5.task12_1_analysis import (
    SOLVERS,
    TASK12_HEAVY,
    fb_mean,
    fb_series,
    statewise_pairwise_delta,
)
from decision.cluster_weighting import WeightedMedianImputer

ROOT = Path(__file__).resolve().parents[3]
T13_HEAVY = ROOT / "behavior_with_ela/results/analysis_v5/task13"
T13_LIGHT = ROOT / "behavior_with_ela/analysis_v5/task13"
BOOTSTRAP_STREAM = 2026090114
BOOTSTRAP_DRAWS = 2000
RF_RANDOM_STATE = 2026090113
SHUFFLE_REPEATS_RF = 100
SHUFFLE_REPEATS_RF_WITHIN = 10
N_SELECTOR_BEHAVIOR = 28
TIME_LIKE_FEATURES = ("bf_fe_ratio",)
LOSS_COLS = [f"loss_{s}" for s in SOLVERS]


def make_carrier(kind: str):
    if kind == "rf":
        return Pipeline(
            [
                ("imputer", WeightedMedianImputer()),
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=8,
                        max_features="sqrt",
                        random_state=RF_RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    if kind == "ridge":
        return Pipeline(
            [
                ("imputer", WeightedMedianImputer()),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1.0)),
            ]
        )
    raise ValueError(kind)


def paired_group_bootstrap(frame: pd.DataFrame, value_column: str, stream_offset: int) -> tuple[float, float, float]:
    groups = sorted(frame["cv_group_id"].unique())
    means = fb_series(frame, value_column).to_dict()
    rng = np.random.default_rng(
        np.random.SeedSequence([BOOTSTRAP_STREAM + stream_offset, len(groups)]).generate_state(4)
    )
    draws = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        sample = rng.choice(groups, size=len(groups), replace=True)
        draws[draw] = np.mean([means[g] for g in sample])
    return (
        float(fb_series(frame, value_column).mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def build_dataset() -> tuple[pd.DataFrame, list[str]]:
    states = pd.read_parquet(TASK12_HEAVY / "dynamic_screening_states.parquet")
    solver = pd.read_parquet(TASK12_HEAVY / "dynamic_solver_loss_matrix.parquet")
    base = pd.read_parquet(TASK12_HEAVY / "dynamic_action_outcomes_1000.parquet")
    sets = pd.read_parquet(T13_HEAVY / "practical_action_sets_max.parquet")
    behavior = pd.read_parquet(T13_HEAVY / "behavior_global_features.parquet")

    cont = base.loc[base["candidate_action"].eq("continue")].set_index("state_id")["loss_1000"]
    frame = solver.merge(states[["state_id", "family", "instance"]], on="state_id", validate="many_to_one")
    frame = frame.merge(
        sets[["state_id", "switch_required", "A_ND_members", "A_ND_size", "switch_target"]],
        on="state_id",
        validate="many_to_one",
    )
    frame["continue_loss"] = cont.reindex(frame["state_id"]).to_numpy()
    frame["raw_best_action"] = frame[LOSS_COLS].idxmin(axis=1).str.replace("loss_", "", regex=False)
    frame["FE_ratio"] = frame["FE"] / 10000.0

    bf_cols = [
        c
        for c in behavior.columns
        if c in set(SELECTOR_BEHAVIOR_FEATURE_COLUMNS)
    ]
    if len(bf_cols) != N_SELECTOR_BEHAVIOR:
        raise SystemExit(f"selector behavior contract mismatch: {len(bf_cols)}")
    dataset = frame.merge(behavior[["state_id", *bf_cols]], on="state_id", validate="one_to_one")
    dataset.to_parquet(T13_HEAVY / "behavior_action_dataset_task13.parquet", index=False)

    noise = pd.read_parquet(TASK12_HEAVY / "dynamic_noise_deltas.parquet")
    per_action = noise.set_index(["suite", "candidate_action"])["delta_95_function_balanced"]
    delta_ctx = statewise_pairwise_delta(frame, per_action, SOLVERS)
    delta_ctx.to_parquet(T13_HEAVY / "pairwise_delta_context.parquet", index=False)
    return dataset, bf_cols


def feature_audit(dataset: pd.DataFrame, bf_cols: list[str]) -> dict:
    audit = {"n_states": int(len(dataset)), "n_behavior_features": len(bf_cols)}
    audit["nan_by_column"] = {
        c: int(dataset[c].isna().sum()) for c in bf_cols if dataset[c].isna().any()
    }
    audit["inf_total"] = int(np.isinf(dataset[bf_cols].to_numpy(dtype=float)).sum())
    audit["constant_columns"] = [c for c in bf_cols if dataset[c].nunique(dropna=True) <= 1]
    audit["time_like_columns"] = list(TIME_LIKE_FEATURES)
    audit["maturity_columns_excluded"] = [
        "bf_search_maturity",
        "bf_search_maturity_linear",
        "bf_explore_exploit_ratio",
    ]
    availability = {}
    for solver in SOLVERS:
        part = dataset.loc[dataset["current_algorithm"].eq(solver), bf_cols]
        availability[solver] = {
            "n": int(len(part)),
            "nan_total": int(part.isna().sum().sum()),
            "all_features_available_rows": int(part.notna().all(axis=1).sum()),
        }
    audit["availability_by_current_solver"] = availability
    audit["availability_by_FE"] = {
        str(fe): int(part[bf_cols].notna().all(axis=1).sum())
        for fe, part in dataset.groupby("FE")
    }
    return audit


def _current_one_hot(dataset: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(dataset["current_algorithm"], prefix="cur", dtype=float)
    return dummies


def run_grouped_oof(
    dataset: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    carrier: str,
    permuted_behavior: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Leave-cv_group-out predictions for every feature set. When
    permuted_behavior is given (model -> feature frame), the behavior columns
    of that model are replaced by the shuffled values before fitting."""
    truth = dataset[LOSS_COLS].to_numpy(dtype=float)
    out = []
    for model_name, features in feature_sets.items():
        matrix = dataset[features]
        if permuted_behavior is not None and model_name in permuted_behavior:
            matrix = permuted_behavior[model_name]
        X = matrix.to_numpy(dtype=float)
        pred = np.full_like(truth, np.nan)
        fold_id = np.empty(len(dataset), dtype=object)
        for group in sorted(dataset["cv_group_id"].unique()):
            test_mask = dataset["cv_group_id"].eq(group).to_numpy()
            model = make_carrier(carrier)
            model.fit(X[~test_mask], truth[~test_mask])
            pred[test_mask] = model.predict(X[test_mask])
            fold_id[test_mask] = f"holdout_{group}"
        selected_index = pred.argmin(axis=1)
        frame = pd.DataFrame(
            {
                "state_id": dataset["state_id"].to_numpy(),
                "model": model_name,
                "carrier": carrier,
                "fold_id": fold_id,
                "selected": [SOLVERS[i] for i in selected_index],
                "realized_loss": truth[np.arange(len(truth)), selected_index],
                "pred_shade": pred[:, 0],
                "pred_lshade": pred[:, 1],
                "pred_cso": pred[:, 2],
            }
        )
        out.append(frame)
    return pd.concat(out, ignore_index=True)


def policy_summary(preds: pd.DataFrame, dataset: pd.DataFrame, delta_pair: pd.Series) -> pd.DataFrame:
    merged = preds.merge(
        dataset[["state_id", "suite", "cv_group_id", "seed", "continue_loss", "current_algorithm", "switch_required", "FE"]],
        on="state_id",
        validate="many_to_one",
    )
    merged["gain_vs_continue"] = merged["continue_loss"] - merged["realized_loss"]
    merged["is_switch"] = merged["selected"].ne(merged["current_algorithm"])
    merged["harmful"] = merged["realized_loss"] > merged["continue_loss"] + merged["delta_pair"]
    rows = []
    for (model, carrier_name, suite_name), group in merged.groupby(["model", "carrier", "suite"], sort=False):
        rows.append(
            {
                "model": model,
                "carrier": carrier_name,
                "suite": suite_name,
                "realized_fb_loss": fb_mean(group, "realized_loss"),
                "gain_vs_continue_fb": fb_mean(group, "gain_vs_continue"),
                "harmful_rate": float(group["harmful"].mean()),
                "switch_rate": float(group["is_switch"].mean()),
                "switch_precision": float(
                    group.loc[group["is_switch"], "switch_required"].mean()
                )
                if group["is_switch"].any()
                else np.nan,
                "switch_recall": float(
                    group.loc[group["switch_required"], "is_switch"].mean()
                )
                if group["switch_required"].any()
                else np.nan,
                "unnecessary_switch_rate": float(
                    (~group.loc[group["is_switch"], "switch_required"]).mean()
                )
                if group["is_switch"].any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows), merged


def prediction_diagnostics(preds: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    truth = dataset.set_index("state_id")[LOSS_COLS]
    rows = []
    for (model, carrier_name, suite_name), group in preds.groupby(["model", "carrier", "suite"], sort=False):
        sub = group.set_index("state_id")
        ordered = sub.loc[truth.index.intersection(sub.index)]
        y = truth.loc[ordered.index]
        pred = ordered[["pred_shade", "pred_lshade", "pred_cso"]].to_numpy(dtype=float)
        yv = y.to_numpy(dtype=float)
        row = {"model": model, "suite": suite_name, "carrier": carrier_name}
        for index, solver in enumerate(SOLVERS):
            rho = spearmanr(pred[:, index], yv[:, index]).statistic
            row[f"spearman_{solver}"] = float(rho)
        row["mae"] = float(mean_absolute_error(yv, pred))
        row["rmse"] = float(mean_squared_error(yv, pred) ** 0.5)
        correct_top1 = 0
        total_top1 = 0
        pair_correct, pair_total = 0, 0
        for i in range(len(yv)):
            true_best = int(np.argmin(yv[i]))
            pred_best = int(np.argmin(pred[i]))
            total_top1 += 1
            correct_top1 += int(pred_best == true_best)
            for a, b in combinations(range(3), 2):
                pair_total += 1
                pair_correct += int((pred[i, a] < pred[i, b]) == (yv[i, a] < yv[i, b]))
        row["raw_top1_accuracy"] = correct_top1 / total_top1
        row["pairwise_ordering_accuracy"] = pair_correct / pair_total
        rows.append(row)
    return pd.DataFrame(rows)


def within_problem_loso(dataset: pd.DataFrame, bf_cols: list[str], carrier: str, behavior_perm: np.ndarray | None = None) -> pd.DataFrame:
    """Leave-one-seed-out within (problem, current, FE) groups. W0 predicts
    each action by the group-train mean; W1/W2 fit the carrier on the 4-row
    train part. When behavior_perm is given, behavior values are permuted
    (per repeat) before fitting."""
    truth = dataset[LOSS_COLS].to_numpy(dtype=float)
    groups = dataset.groupby(["problem_id", "current_algorithm", "FE"], sort=False).groups
    contexts = dataset[["cur_shade", "cur_lshade", "cur_cso", "FE_ratio"]].to_numpy(dtype=float)
    B = dataset[bf_cols].to_numpy(dtype=float)
    if behavior_perm is not None:
        B = B[behavior_perm]
    X1 = B
    X2 = np.hstack([contexts, B])
    rows = []
    for key, positions in groups.items():
        idx = np.asarray(positions)
        for pos in idx:
            train = idx[idx != pos]
            mean_pred = truth[train].mean(axis=0)
            for model_name, X in (("W0", None), ("W1", X1), ("W2", X2)):
                if model_name == "W0":
                    p = mean_pred
                else:
                    model = make_carrier(carrier)
                    model.fit(X[train], truth[train])
                    p = model.predict(X[pos : pos + 1])[0]
                sel = int(np.argmin(p))
                rows.append(
                    {
                        "state_id": dataset.at[pos, "state_id"],
                        "suite": dataset.at[pos, "suite"],
                        "cv_group_id": dataset.at[pos, "cv_group_id"],
                        "problem_id": key[0],
                        "current_algorithm": key[1],
                        "FE": key[2],
                        "seed": int(dataset.at[pos, "seed"]),
                        "model": model_name,
                        "carrier": carrier,
                        "realized_loss": float(truth[pos, sel]),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="reload finished shuffle-control results instead of recomputing")
    args = parser.parse_args()
    T13_HEAVY.mkdir(parents=True, exist_ok=True)
    T13_LIGHT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    dataset, bf_cols = build_dataset()
    audit = feature_audit(dataset, bf_cols)
    (T13_LIGHT / "13e_behavior_feature_audit.json").write_text(json.dumps(audit, indent=2))

    dummies = _current_one_hot(dataset)
    dataset = pd.concat([dataset.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    cur_cols = list(dummies.columns)
    feature_sets = {
        "M0_current_FE": [*cur_cols, "FE_ratio"],
        "M1_behavior": list(bf_cols),
        "M2_current_FE_behavior": [*cur_cols, "FE_ratio", *bf_cols],
        "M3_current_only": list(cur_cols),
        "M4_FE_only": ["FE_ratio"],
    }

    # pairwise max-rule delta between the continue cell and each solver cell
    delta_ctx = pd.read_parquet(T13_HEAVY / "pairwise_delta_context.parquet")
    delta_map = delta_ctx.set_index("state_id")
    def delta_pair_for(row) -> float:
        current = row["current_algorithm"]
        selected = row["selected"]
        if selected == current:
            return 0.0
        return float(max(delta_map.at[row["state_id"], f"delta_cell_{current}"],
                         delta_map.at[row["state_id"], f"delta_cell_{selected}"]))

    # ---- grouped OOF, both carriers ----
    all_preds = []
    for carrier in ("rf", "ridge"):
        preds = run_grouped_oof(dataset, feature_sets, carrier)
        all_preds.append(preds)
    preds = pd.concat(all_preds, ignore_index=True)
    preds = preds.merge(
        delta_ctx[["state_id", "delta_cell_shade", "delta_cell_lshade", "delta_cell_cso"]],
        on="state_id",
        validate="many_to_one",
    )
    preds = preds.merge(
        dataset[["state_id", "current_algorithm"]], on="state_id", validate="many_to_one"
    )
    preds["delta_pair"] = preds.apply(delta_pair_for, axis=1)
    preds = preds.drop(columns=["current_algorithm"])
    summary, merged_full = policy_summary(preds, dataset, preds["delta_pair"])

    # incremental gains with paired bootstrap CIs (primary: M2 vs M0)
    gain_rows = []
    stream = 0
    merged_full.to_parquet(T13_HEAVY / "oof_policy_rows.parquet", index=False)
    for carrier in ("rf", "ridge"):
        for suite_name in ("bbob", "mabbob"):
            part = merged_full.loc[merged_full["suite"].eq(suite_name) & merged_full["carrier"].eq(carrier)]
            pivot = part.pivot_table(index=["state_id", "cv_group_id"], columns="model", values="realized_loss").reset_index()
            pivot["delta_M0_minus_M2"] = pivot["M0_current_FE"] - pivot["M2_current_FE_behavior"]
            pivot["delta_M0_minus_M1"] = pivot["M0_current_FE"] - pivot["M1_behavior"]
            pivot["delta_M2_minus_M1"] = pivot["M2_current_FE_behavior"] - pivot["M1_behavior"]
            for quantity, offset in (
                ("delta_M0_minus_M2", 0),
                ("delta_M0_minus_M1", 7),
                ("delta_M2_minus_M1", 13),
            ):
                mean, low, high = paired_group_bootstrap(pivot, quantity, stream_offset=stream + offset)
                gain_rows.append(
                    {
                        "carrier": carrier,
                        "suite": suite_name,
                        "quantity": quantity,
                        "fb_mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
            stream += 20
    gains = pd.DataFrame(gain_rows)

    # ---- time-proxy sensitivity (13P): drop bf_fe_ratio from M2 ----
    feature_sets_time = {
        "M2_full": feature_sets["M2_current_FE_behavior"],
        "M2_minus_time": [c for c in feature_sets["M2_current_FE_behavior"] if c not in TIME_LIKE_FEATURES],
    }
    time_rows = []
    for carrier in ("rf", "ridge"):
        tp = run_grouped_oof(dataset, feature_sets_time, carrier)
        tp = tp.merge(dataset[["state_id", "suite", "cv_group_id"]], on="state_id", validate="many_to_one")
        for (model, suite_name), group in tp.groupby(["model", "suite"], sort=False):
            time_rows.append(
                {
                    "carrier": carrier,
                    "suite": suite_name,
                    "model": model,
                    "realized_fb_loss": fb_mean(group, "realized_loss"),
                }
            )
    time_proxy = pd.DataFrame(time_rows)

    # ---- within-problem LOSO (13N) ----
    within_preds = []
    for carrier in ("rf", "ridge"):
        within_preds.append(within_problem_loso(dataset, bf_cols, carrier))
    within = pd.concat(within_preds, ignore_index=True)
    within.to_parquet(T13_HEAVY / "within_problem_loso_predictions.parquet", index=False)
    within_summary_rows = []
    for carrier in ("rf", "ridge"):
        for suite_name, group in within.loc[within["carrier"].eq(carrier)].groupby("suite", sort=False):
            pivot = group.pivot_table(index=["state_id", "cv_group_id"], columns="model", values="realized_loss").reset_index()
            pivot["delta_within"] = pivot["W0"] - pivot["W2"]
            pivot["delta_within_W1"] = pivot["W0"] - pivot["W1"]
            mean, low, high = paired_group_bootstrap(pivot, "delta_within", stream_offset=40 if carrier == "rf" else 60)
            mean1, low1, high1 = paired_group_bootstrap(pivot, "delta_within_W1", stream_offset=(45 if carrier == "rf" else 65))
            within_summary_rows.append(
                {
                    "carrier": carrier,
                    "suite": suite_name,
                    "L_W0": fb_mean(pivot, "W0"),
                    "L_W1": fb_mean(pivot, "W1"),
                    "L_W2": fb_mean(pivot, "W2"),
                    "delta_within_W0_minus_W2": mean,
                    "delta_within_ci_low": low,
                    "delta_within_ci_high": high,
                    "delta_within_W0_minus_W1": mean1,
                    "delta_within_W1_ci_low": low1,
                    "delta_within_W1_ci_high": high1,
                }
            )
    within_summary = pd.DataFrame(within_summary_rows)
    within_summary.to_parquet(T13_LIGHT / "within_problem_performance.parquet", index=False)

    # ---- shuffle controls (13O) ----
    resume_path = T13_LIGHT / "shuffle_control_results.parquet"
    if args.resume and resume_path.exists():
        shuffle_table = pd.read_parquet(resume_path)
        print(f"[shuffle] reloaded {len(shuffle_table)} rows", flush=True)
    else:
        shuffle_rows = []
        rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_STREAM + 90]).generate_state(4))
        # O1: shuffle behavior within (current, FE) stratum, refit M2, full grouped OOF
        for repeat in range(SHUFFLE_REPEATS_RF):
            permuted = dataset[bf_cols].copy()
            for _, idx in dataset.groupby(["current_algorithm", "FE"]).groups.items():
                positions = np.asarray(idx)
                permuted.iloc[positions] = permuted.iloc[rng.permutation(positions)].to_numpy()
            feature_sets_o1 = {"M2_current_FE_behavior": feature_sets["M2_current_FE_behavior"]}
            perm = {"M2_current_FE_behavior": permuted}
            for carrier in ("rf", "ridge"):
                if carrier == "rf" and repeat >= SHUFFLE_REPEATS_RF:
                    continue
                result = run_grouped_oof(dataset, feature_sets_o1, carrier, permuted_behavior=perm)
                result = result.merge(dataset[["state_id", "suite", "cv_group_id", "continue_loss"]], on="state_id", validate="many_to_one")
                for suite_name, group in result.groupby("suite", sort=False):
                    m0 = merged_full.loc[
                        merged_full["suite"].eq(suite_name)
                        & merged_full["model"].eq("M0_current_FE")
                        & merged_full["carrier"].eq(carrier),
                        ["state_id", "realized_loss"],
                    ].rename(columns={"realized_loss": "m0_loss"})
                    g = group.merge(m0, on="state_id", validate="many_to_one")
                    shuffle_rows.append(
                        {
                            "control": "O1_shuffle_within_current_FE",
                            "carrier": carrier,
                            "repeat": repeat,
                            "suite": suite_name,
                            "L_M2_shuffled": fb_mean(g, "realized_loss"),
                            "L_M0_reference": fb_mean(g, "m0_loss"),
                            "delta_shuffled_M0_minus_M2": fb_mean(g, "m0_loss") - fb_mean(g, "realized_loss"),
                        }
                    )
            if repeat % 20 == 0:
                print(f"[O1] repeat {repeat}", flush=True)

        # O2: shuffle behavior within (problem, current, FE) group, within-problem LOSO
        group_keys = dataset.groupby(["problem_id", "current_algorithm", "FE"], sort=False).groups
        for repeat in range(max(SHUFFLE_REPEATS_RF_WITHIN, 100)):
            perm_index = np.arange(len(dataset))
            for _, idx in group_keys.items():
                positions = np.asarray(idx)
                perm_index[positions] = positions[rng.permutation(len(positions))]
            carrier_filter = ("ridge",) if repeat >= SHUFFLE_REPEATS_RF_WITHIN else ("ridge", "rf")
            for carrier in carrier_filter:
                result = within_problem_loso(dataset, bf_cols, carrier, behavior_perm=perm_index)
                pivot = result.pivot_table(index=["state_id", "cv_group_id", "suite"], columns="model", values="realized_loss").reset_index()
                for suite_name, group in pivot.groupby("suite", sort=False):
                    shuffle_rows.append(
                        {
                            "control": "O2_shuffle_within_problem",
                            "carrier": carrier,
                            "repeat": repeat,
                            "suite": suite_name,
                            "L_W2_shuffled": fb_mean(group, "W2"),
                            "L_W1_shuffled": fb_mean(group, "W1"),
                            "L_W0_reference": fb_mean(group, "W0"),
                            "delta_shuffled_W0_minus_W2": fb_mean(group, "W0") - fb_mean(group, "W2"),
                        }
                    )
            if repeat % 20 == 0:
                print(f"[O2] repeat {repeat}", flush=True)
        shuffle_table = pd.DataFrame(shuffle_rows)
        shuffle_table.to_parquet(resume_path, index=False)

    # ---- feature contribution diagnostic (13Q, RF only, full-data fit) ----
    importance_rows = []
    for suite_name, group in dataset.groupby("suite", sort=False):
        X = group[feature_sets["M2_current_FE_behavior"]].to_numpy(dtype=float)
        y = group[LOSS_COLS].to_numpy(dtype=float)
        model = make_carrier("rf")
        model.fit(X, y)
        imp = permutation_importance(model, X, y, n_repeats=5, random_state=RF_RANDOM_STATE, n_jobs=1)
        for name, values in zip(feature_sets["M2_current_FE_behavior"], imp.importances_mean):
            importance_rows.append({"suite": suite_name, "feature": name, "importance_mean": float(values)})
    importance = pd.DataFrame(importance_rows)
    importance.to_parquet(T13_LIGHT / "feature_importance_diagnostic.parquet", index=False)

    # ---- non-deployable problem-identity diagnostic (13R) ----
    problem_dummies = pd.get_dummies(dataset["problem_id"], prefix="prob", dtype=float)
    dataset_diag = pd.concat([dataset.reset_index(drop=True), problem_dummies.reset_index(drop=True)], axis=1)
    prob_cols = list(problem_dummies.columns)
    diag_sets = {
        "D_problem_current_FE": [*prob_cols, *cur_cols, "FE_ratio"],
        "D_problem_current_FE_behavior": [*prob_cols, *cur_cols, "FE_ratio", *bf_cols],
    }
    diag_rows = []
    diag_preds = []
    for carrier in ("rf", "ridge"):
        dp = run_grouped_oof(dataset_diag, diag_sets, carrier)
        dp = dp.merge(dataset[["state_id", "suite", "cv_group_id"]], on="state_id", validate="many_to_one")
        diag_preds.append(dp)
        for (model, suite_name), group in dp.groupby(["model", "suite"], sort=False):
            diag_rows.append(
                {
                    "carrier": carrier,
                    "suite": suite_name,
                    "model": model + " (NON-DEPLOYABLE DIAGNOSTIC)",
                    "realized_fb_loss": fb_mean(group, "realized_loss"),
                }
            )
    diag_table = pd.concat(diag_preds, ignore_index=True)
    diag_table.to_parquet(T13_HEAVY / "oof_problem_id_diagnostic_predictions.parquet", index=False)

    # ---- save main prediction & performance tables (29 requirements) ----
    merged_rf = merged_full.loc[merged_full["carrier"].eq("rf")]
    wide = merged_rf.pivot_table(
        index=["state_id", "fold_id", "suite", "cv_group_id", "seed", "current_algorithm", "FE"],
        columns="model",
        values=["pred_shade", "pred_lshade", "pred_cso", "realized_loss"],
        aggfunc="first",
    )
    wide.columns = [f"{stat.replace('realized_loss', 'realized')}_{model}" for stat, model in wide.columns]
    wide = wide.reset_index()
    selected = merged_full.pivot_table(index="state_id", columns="model", values="selected", aggfunc="first")
    selected.columns = [f"selected_{m}" for m in selected.columns]
    wide = wide.merge(selected.reset_index(), on="state_id", validate="one_to_one")
    truth_cols = dataset.set_index("state_id")[[*LOSS_COLS, "continue_loss", "switch_required"]]
    wide = wide.merge(truth_cols.reset_index(), on="state_id", validate="one_to_one")
    delta_per_state = merged_rf[merged_rf["model"].eq("M0_current_FE")][["state_id", "delta_pair"]]
    wide = wide.merge(delta_per_state, on="state_id", validate="one_to_one")
    wide["harmful_M2"] = wide["realized_M2_current_FE_behavior"] > (
        wide["continue_loss"] + wide["delta_pair"]
    )
    wide.to_parquet(T13_HEAVY / "oof_action_loss_predictions.parquet", index=False)
    summary.to_parquet(T13_LIGHT / "oof_policy_performance.parquet", index=False)
    gains.to_parquet(T13_LIGHT / "oof_increment_gains.parquet", index=False)
    time_proxy.to_parquet(T13_LIGHT / "time_proxy_sensitivity.parquet", index=False)
    diagnostics = prediction_diagnostics(merged_full, dataset)
    diagnostics.to_parquet(T13_LIGHT / "oof_prediction_diagnostics.parquet", index=False)
    pd.DataFrame(diag_rows).to_parquet(T13_LIGHT / "problem_id_diagnostic.parquet", index=False)

    elapsed = perf_counter() - started
    ledger = pd.DataFrame(
        [
            {
                "phase": "task13_analysis",
                "state_reconstruction_fe": 3780000,
                "new_action_label_fe": 0,
                "wall_seconds": elapsed,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                "note": "replay FE executed once in task13_replay.py; analysis itself is zero-FE",
            }
        ]
    )
    ledger.to_parquet(T13_HEAVY / "task13_resource_ledger.parquet", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print("=== OOF policy performance ===")
        print(summary.round(4).to_string())
        print("=== incremental gains ===")
        print(gains.round(4).to_string())
        print("=== time proxy ===")
        print(time_proxy.round(4).to_string())
        print("=== within-problem ===")
        print(within_summary.round(4).to_string())
        print("=== prediction diagnostics ===")
        print(diagnostics.round(4).to_string())
        print("=== problem-id diagnostic ===")
        print(pd.DataFrame(diag_rows).round(4).to_string())
        print("=== shuffle controls summary ===")
        obs = shuffle_table.groupby(["control", "carrier", "suite"])["delta_shuffled_M0_minus_M2"].agg(["mean", "quantile"])
        print(
            shuffle_table.groupby(["control", "carrier", "suite"])
            .agg(
                n=("repeat", "size"),
                delta_mean=("delta_shuffled_M0_minus_M2", "mean"),
                delta_q025=("delta_shuffled_M0_minus_M2", lambda s: np.nanquantile(s, 0.025)),
                delta_q975=("delta_shuffled_M0_minus_M2", lambda s: np.nanquantile(s, 0.975)),
            )
            .round(4)
            .to_string()
        )
        print("=== feature importance (top 10 per suite) ===")
        for suite_name, group in importance.groupby("suite", sort=False):
            print(suite_name, group.nlargest(10, "importance_mean")[["feature", "importance_mean"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
