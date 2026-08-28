"""Task 9F: grouped re-analysis of Behavior / Local Landscape redundancy.

Fixes the optimistic 80/20 row split of analysis_v2 task5: each state appears in
three action rows with identical feature vectors, so row-level splits leaked
duplicate states between train and test. Here every state contributes one row
(state dedup) and predictions are made under two grouping schemes:
  A. leave-BBOB-family-out (MA rows kept in training under the component guard)
  B. grouped by problem_id (every function instance held out as a block)

Both directions (bf->lf and lf->bf) are evaluated with multi-output RF.
Also restates the A1 vs A3 OOF action disagreement under corrected naming.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import common  # noqa: F401
from common import (
    V2_HEAVY,
    json_dumps,
    load_train_val,
    save_table,
)

sys.path.insert(0, str(common.ROOT))

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS as BF_COLS  # noqa: E402
from behavior_with_ela.local_landscape import (  # noqa: E402
    LOCAL_LANDSCAPE_POINT_COLUMNS as LF_COLS,
)
from behavior_with_ela.model import STATE_KEY, _ma_overlaps_heldout  # noqa: E402

TASK = "task6"
FOLD_STREAM = 2026082919
N_FOLDS = 5


def state_frame(train: pd.DataFrame) -> pd.DataFrame:
    states = train.loc[train["action_equals_prefix"].astype(bool)].copy()
    dedup_keys = list(STATE_KEY)
    if states.duplicated(dedup_keys).any():
        raise RuntimeError("state deduplication left duplicate rows")
    return states.reset_index(drop=True)


def fold_assignments(groups: list[str]) -> dict[str, int]:
    order = np.random.default_rng(
        np.random.SeedSequence([FOLD_STREAM, len(groups)]).generate_state(1)
    ).permutation(len(groups))
    return {
        group: int(position % N_FOLDS)
        for position, group in zip(order, groups)
    }


def r2_per_target(
    truth: np.ndarray,
    predicted: np.ndarray,
    targets: list[str],
    fold_names: list[np.ndarray],
) -> pd.DataFrame:
    rows = []
    for index, target in enumerate(targets):
        y = truth[:, index]
        p = predicted[:, index]
        ss_res = float(((y - p) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        pooled = 1.0 - ss_res / max(ss_tot, 1e-12)
        fold_r2 = []
        for mask in fold_names:
            if mask.sum() < 30:
                continue
            y_f = y[mask]
            p_f = p[mask]
            ss_tot_f = float(((y_f - y_f.mean()) ** 2).sum())
            fold_r2.append(
                1.0 - float(((y_f - p_f) ** 2).sum()) / max(ss_tot_f, 1e-12)
            )
        rows.append(
            {
                "target": target,
                "pooled_oof_r2": pooled,
                "median_fold_r2": float(np.median(fold_r2)),
                "share_folds_r2_above_0.5": float(
                    np.mean([value > 0.5 for value in fold_r2])
                ),
                "share_folds_r2_below_0": float(
                    np.mean([value < 0.0 for value in fold_r2])
                ),
                "folds": len(fold_r2),
            }
        )
    return pd.DataFrame(rows)


def run_scheme(
    states: pd.DataFrame,
    inputs: list[str],
    targets: list[str],
    scheme: str,
) -> pd.DataFrame:
    if scheme == "leave_family_out":
        bbob = states.loc[states["suite"].astype(str).eq("bbob")]
        families = tuple(sorted(set(bbob["family"].astype(str))))
        folds = []
        for heldout_family in families:
            heldout_functions = set(
                bbob.loc[
                    bbob["family"].astype(str).eq(heldout_family), "function_id"
                ].astype(str)
            )
            mask = ~(
                states["suite"].astype(str).eq("bbob")
                & states["family"].astype(str).eq(heldout_family)
            )
            ma_safe = ~states.apply(
                lambda row: _ma_overlaps_heldout(row, heldout_functions), axis=1
            )
            fit_states = states.loc[mask & ma_safe]
            eval_states = bbob.loc[
                bbob["family"].astype(str).eq(heldout_family)
            ]
            folds.append((heldout_family, fit_states, eval_states))
    elif scheme == "grouped_by_problem":
        assignment = fold_assignments(
            sorted(states["problem_id"].astype(str).unique())
        )
        fold_of = states["problem_id"].astype(str).map(assignment)
        folds = []
        for fold in range(N_FOLDS):
            fit_states = states.loc[~fold_of.eq(fold)]
            eval_states = states.loc[fold_of.eq(fold)]
            folds.append((f"problem_fold_{fold}", fit_states, eval_states))
    else:
        raise ValueError(scheme)

    predicted_blocks = []
    truth_blocks = []
    fold_masks = []
    for fold_number, (fold_name, fit_states, eval_states) in enumerate(
        folds, start=1
    ):
        medians = fit_states[inputs + targets].median(numeric_only=True)
        x_fit = fit_states[inputs].fillna(medians).to_numpy(dtype=float)
        y_fit = fit_states[targets].fillna(medians).to_numpy(dtype=float)
        x_eval = eval_states[inputs].fillna(medians).to_numpy(dtype=float)
        random_state = int(
            np.random.SeedSequence(
                [FOLD_STREAM, 7, fold_number, len(inputs)]
            ).generate_state(1, dtype=np.uint32)[0]
        )
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=random_state,
            n_jobs=4,
        )
        model.fit(x_fit, y_fit)
        predicted_blocks.append(model.predict(x_eval))
        truth_blocks.append(eval_states[targets].to_numpy(dtype=float))
        fold_masks.append(
            np.full(len(eval_states), fold_number, dtype=int)
        )
        print(
            f"[{task_label(scheme, inputs)}] fold {fold_number}/{len(folds)} done "
            f"({len(eval_states)} eval states)",
            flush=True,
        )
    predicted = np.concatenate(predicted_blocks, axis=0)
    truth = np.concatenate(truth_blocks, axis=0)
    fold_array = np.concatenate(fold_masks)
    masks = [fold_array == number for number in sorted(set(fold_array.tolist()))]
    table = r2_per_target(truth, predicted, targets, masks)
    table.insert(0, "direction", "bf_to_lf" if inputs == list(BF_COLS) else "lf_to_bf")
    table.insert(0, "scheme", scheme)
    return table


def task_label(scheme: str, inputs: list[str]) -> str:
    direction = "bf_to_lf" if inputs == list(BF_COLS) else "lf_to_bf"
    return f"{scheme}:{direction}"


def a1_a3_disagreement() -> dict:
    oof = pd.read_parquet(
        common.RESULTS
        / "model/local_landscape_increment_action_loss/train_oof_action_predictions.parquet",
        columns=[
            "phase2_feature_group", "problem_id", "prefix_algorithm", "seed",
            "FE", "decision_opportunity_index", "candidate_action",
            "predicted_action_gain", "action_gain_vs_continue",
        ],
    )
    a1 = oof.loc[oof["phase2_feature_group"].eq("A1_behavior")]
    a3 = oof.loc[oof["phase2_feature_group"].eq("A3_behavior_local_landscape")]
    key = ["problem_id", "prefix_algorithm", "seed", "FE", "decision_opportunity_index"]
    merged = a1.merge(
        a3.drop(columns="phase2_feature_group"),
        on=key + ["candidate_action"],
        suffixes=("_a1", "_a3"),
    )
    state_first = (
        merged.sort_values("predicted_action_gain_a1", ascending=False, kind="mergesort")
        .groupby(key, sort=False)
        .head(2)
    )

    def top_actions(frame: pd.DataFrame, column: str):
        best = (
            frame.sort_values(column, ascending=False, kind="mergesort")
            .groupby(key, sort=False)
            .head(1)
            .set_index(key)
        )
        return best["candidate_action"], best["action_gain_vs_continue_a1"]

    a1_action, a1_true = top_actions(state_first, "predicted_action_gain_a1")
    a3_action, a3_true = top_actions(state_first, "predicted_action_gain_a3")
    paired = pd.DataFrame(
        {
            "a1_action": a1_action,
            "a3_action": a3_action,
            "a1_true": a1_true,
            "a3_true": a3_true,
        }
    ).dropna()
    disagree = paired.loc[paired["a1_action"] != paired["a3_action"]]
    advantage = (
        disagree["a3_true"].to_numpy(dtype=float)
        - disagree["a1_true"].to_numpy(dtype=float)
    )
    return {
        "states": int(len(paired)),
        "top1_disagreement_rate": float(len(disagree) / max(len(paired), 1)),
        "a3_better_count": int((advantage > 0).sum()),
        "a1_better_count": int((advantage < 0).sum()),
        "mean_true_gain_delta_if_forcing_a3": (
            float(advantage.mean()) if len(advantage) else 0.0
        ),
        "median_true_gain_delta_if_forcing_a3": (
            float(np.median(advantage)) if len(advantage) else 0.0
        ),
        "naming_note": (
            "A1/A3 disagreement is restated as actionable-overlap evidence; the "
            "prefix-conditioned one-switch upper bound naming applies to the "
            "best-observed quantities elsewhere in analysis_v3"
        ),
    }


def main() -> None:
    config, validation_config, bundle, delta, train, validation = load_train_val()
    states = state_frame(train)
    print(f"[{TASK}] states={len(states)}", flush=True)

    tables = []
    summaries = {}
    for scheme in ("leave_family_out", "grouped_by_problem"):
        for inputs, targets in (
            (list(BF_COLS), list(LF_COLS)),
            (list(LF_COLS), list(BF_COLS)),
        ):
            table = run_scheme(states, inputs, targets, scheme)
            tables.append(table)
            direction = table["direction"].iloc[0]
            summaries[f"{scheme}:{direction}"] = {
                "median_target_pooled_r2": float(
                    table["pooled_oof_r2"].median()
                ),
                "share_targets_pooled_r2_above_0.5": float(
                    (table["pooled_oof_r2"] > 0.5).mean()
                ),
                "share_targets_negative_pooled_r2": float(
                    (table["pooled_oof_r2"] < 0.0).mean()
                ),
            }
    result = pd.concat(tables, ignore_index=True)
    save_table(result, "grouped_cross_predictability.csv", TASK)

    disagreement = a1_a3_disagreement()
    payload = {
        "states": int(len(states)),
        "bf_features": len(BF_COLS),
        "lf_features": len(LF_COLS),
        "schemes": summaries,
        "a1_a3_oof_disagreement": disagreement,
        "permutation_importance_note": (
            "block permutation importance from analysis_v2 task5 was computed on "
            "training rows of the full model and is not re-reported here; the "
            "grouped OOF cross-predictability above is the mechanism evidence"
        ),
    }
    save_table(payload, "summary.json", TASK)
    print(f"[{TASK}] done", flush=True)
    print(json_dumps(summaries), flush=True)
    print(json_dumps(disagreement), flush=True)


if __name__ == "__main__":
    main()
