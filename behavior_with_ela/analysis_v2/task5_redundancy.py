"""Task 5: Behavior vs Local Landscape redundancy and disagreement analysis."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import common  # noqa: F401
from common import V2_HEAVY, json_dumps, load_train_val, save_table

sys.path.insert(0, str(common.ROOT))

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS as COLS  # noqa: E402
from behavior_with_ela.local_landscape import (  # noqa: E402
    LOCAL_LANDSCAPE_POINT_COLUMNS as LCOLS,
)
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

TASK = "task5"
SUBSAMPLE = 20_000
IMPORTANCE_SAMPLE = 5_000


def main() -> None:
    config, validation_config, bundle, delta, train, validation = load_train_val()
    heavy = V2_HEAVY / TASK
    heavy.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # 1. pairwise Spearman correlation, bf block vs lf block
    sub = train.iloc[rng.choice(len(train), size=SUBSAMPLE, replace=False)]
    corr = sub[list(COLS) + list(LCOLS)].corr(method="spearman")
    cross = corr.loc[list(COLS), list(LCOLS)]
    save_table(cross, "spearman_bf_vs_lf.csv", TASK)
    abs_cross = cross.abs()
    redundancy = {
        "mean_abs_cross_correlation": float(abs_cross.mean().mean()),
        "max_abs_cross_correlation": float(abs_cross.max().max()),
        "share_pairs_above_0.5": float((abs_cross > 0.5).mean().mean()),
        "share_pairs_above_0.7": float((abs_cross > 0.7).mean().mean()),
        "bf_features_with_any_lf_partner_above_0.7": int(
            (abs_cross > 0.7).any(axis=1).sum()
        ),
        "lf_features_with_any_bf_partner_above_0.7": int(
            (abs_cross > 0.7).any(axis=0).sum()
        ),
    }

    # 2. cross-group predictability (multi-output RF, both directions)
    sample = train.iloc[rng.choice(len(train), size=SUBSAMPLE, replace=False)]
    directions = {}
    for name, (inputs, targets) in {
        "bf_to_lf": (list(COLS), list(LCOLS)),
        "lf_to_bf": (list(LCOLS), list(COLS)),
    }.items():
        block = sample[inputs + targets]
        block = block.fillna(block.median(numeric_only=True))
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=0,
            n_jobs=4,
        )
        split = int(len(block) * 0.8)
        model.fit(block.iloc[:split][inputs], block.iloc[:split][targets])
        predicted = model.predict(block.iloc[split:][inputs])
        truth = block.iloc[split:][targets].to_numpy(dtype=float)
        ss_res = ((truth - predicted) ** 2).sum(axis=0)
        ss_tot = ((truth - truth.mean(axis=0)) ** 2).sum(axis=0)
        r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
        directions[name] = {
            target: float(value) for target, value in zip(targets, r2)
        }
    save_table(
        pd.DataFrame(
            [
                {"direction": name, "target": t, "oof_r2": r2}
                for name, mapping in directions.items()
                for t, r2 in mapping.items()
            ]
        ),
        "cross_group_predictability.csv",
        TASK,
    )
    summary_r2 = {
        name: {
            "median_target_r2": float(np.median(list(mapping.values()))),
            "share_targets_r2_above_0.5": float(
                np.mean([v > 0.5 for v in mapping.values()])
            ),
        }
        for name, mapping in directions.items()
    }

    # 3. A1 vs A3 disagreement with true outcomes (train OOF)
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
    m = a1.merge(
        a3.drop(columns="phase2_feature_group"),
        on=key + ["candidate_action"],
        suffixes=("_a1", "_a3"),
    )
    m["disagree_rank"] = (
        m["predicted_action_gain_a1"] > m["predicted_action_gain_a3"]
    )
    state_first = (
        m.sort_values("predicted_action_gain_a1", ascending=False, kind="mergesort")
        .groupby(key, sort=False)
        .head(2)
    )
    def top_actions(frame, column):
        best = frame.sort_values(column, ascending=False, kind="mergesort").groupby(
            key, sort=False
        ).head(1).set_index(key)
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
    disagreement_stats = {
        "states": int(len(paired)),
        "top1_disagreement_rate": float(len(disagree) / max(len(paired), 1)),
        "a3_better_count": int((advantage > 0).sum()),
        "a1_better_count": int((advantage < 0).sum()),
        "mean_true_gain_delta_if_forcing_a3": float(advantage.mean()) if len(advantage) else 0.0,
        "median_true_gain_delta_if_forcing_a3": float(np.median(advantage)) if len(advantage) else 0.0,
    }

    # run-level first-trigger disagreement (train OOF)
    runs = pd.read_parquet(
        common.RESULTS
        / "model/local_landscape_increment_action_loss/train_first_trigger_runs.parquet"
    )
    piv = runs.pivot_table(
        index=["problem_id", "prefix_algorithm", "seed"],
        columns="phase2_feature_group",
        values=["selected_algorithm", "selected_FE", "selected_action_gain"],
        aggfunc="first",
    )
    run_disagreement = {
        "runs": int(len(piv)),
        "same_selected_action": float(
            (
                piv[("selected_algorithm", "A1_behavior")]
                == piv[("selected_algorithm", "A3_behavior_local_landscape")]
            ).mean()
        ),
        "same_selected_fe": float(
            (
                piv[("selected_FE", "A1_behavior")].fillna(-1)
                == piv[("selected_FE", "A3_behavior_local_landscape")].fillna(-1)
            ).mean()
        ),
        "mean_gain_delta_forcing_a3": float(
            (
                piv[("selected_action_gain", "A3_behavior_local_landscape")]
                - piv[("selected_action_gain", "A1_behavior")]
            ).mean()
        ),
    }

    # 4. drop-group permutation importance on the A3 action-loss model
    import joblib

    abundle = joblib.load(
        common.RESULTS
        / "model/local_landscape_increment_action_loss/action_loss_models.joblib"
    )
    model = abundle["models"]["A3_behavior_local_landscape"]
    from behavior_with_ela.baselines import _action_loss_state_matrix

    states, targets = _action_loss_state_matrix(train, common.PORTFOLIO)
    idx = rng.choice(len(states), size=IMPORTANCE_SAMPLE, replace=False)
    xs = states.iloc[idx][list(COLS) + list(LCOLS)]
    ys = targets[idx]
    scoring = ["neg_mean_absolute_error"]
    perm = permutation_importance(
        model, xs, ys, scoring=scoring, n_repeats=3, random_state=0, n_jobs=1
    )
    importances = perm[scoring[0]].importances_mean
    bf_importance = float(np.abs(importances[: len(COLS)]).sum())
    lf_importance = float(np.abs(importances[len(COLS) :]).sum())
    block_importance = {
        "bf_block_importance": bf_importance,
        "lf_block_importance": lf_importance,
        "lf_share": lf_importance / max(bf_importance + lf_importance, 1e-12),
    }

    result = {
        "correlation": redundancy,
        "cross_group_r2": summary_r2,
        "state_level_disagreement": disagreement_stats,
        "run_level_disagreement": run_disagreement,
        "block_permutation_importance": block_importance,
    }
    (common.V2 / TASK / "summary.json").write_text(json_dumps(result))
    print(f"[{TASK}] done: {json_dumps(result)[:400]}", flush=True)


if __name__ == "__main__":
    main()
