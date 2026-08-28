"""Task 6: action-uncertainty diagnostic on the action-loss RF ensemble.

Tree-level variance, top1-top2 loss margin, switch-gain margin and tree
disagreement are compared against realized decision errors. Diagnostic only.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common  # noqa: F401
from common import V2_HEAVY, json_dumps, load_train_val, save_table

sys.path.insert(0, str(common.ROOT))

from behavior_with_ela.baselines import _action_loss_state_matrix  # noqa: E402
import joblib  # noqa: E402

TASK = "task6"


def main() -> None:
    config, validation_config, bundle, delta, train, validation = load_train_val()
    heavy = V2_HEAVY / TASK
    heavy.mkdir(parents=True, exist_ok=True)
    abundle = joblib.load(
        common.RESULTS
        / "model/local_landscape_increment_action_loss/action_loss_models.joblib"
    )
    model = abundle["models"]["A1_behavior"]
    tree_model = model.named_steps["regressor"]
    imputer = model.named_steps["imputer"]
    trees = tree_model.estimators_

    bbob = train.loc[train["suite"].astype(str).eq("bbob")]
    states, targets = _action_loss_state_matrix(bbob, common.PORTFOLIO)
    features = list(abundle["feature_groups"]["A1_behavior"])
    x = imputer.transform(states[features].to_numpy(dtype=float))
    print(f"[{TASK}] states={len(states)}, trees={len(trees)}", flush=True)

    tree_pred = np.stack([tree.predict(x) for tree in trees], axis=0)
    mean_pred = tree_pred.mean(axis=0)
    pred_std = tree_pred.std(axis=0, ddof=1)

    prefix = states["prefix_algorithm"].astype(str).to_numpy()
    prefix_idx = np.array([common.PORTFOLIO.index(a) for a in prefix])
    row_index = np.arange(len(states))
    prefix_loss_pred = mean_pred[row_index, prefix_idx]
    switch_mask = np.ones_like(mean_pred, dtype=bool)
    switch_mask[row_index, prefix_idx] = False

    switch_gain = np.where(
        switch_mask, prefix_loss_pred[:, None] - mean_pred, -np.inf
    )
    best_switch = switch_gain.argmax(axis=1)
    predicted_switch_gain = switch_gain.max(axis=1)
    top2_losses = np.sort(mean_pred, axis=1)
    loss_margin = top2_losses[:, 1] - top2_losses[:, 0]

    best_action = mean_pred.argmin(axis=1)
    tree_best = tree_pred.argmin(axis=2)
    tree_disagreement = (tree_best != best_action[None, :]).mean(axis=0)

    true_best = targets.argmin(axis=1)
    true_switch_gain = np.where(
        switch_mask, targets[row_index, prefix_idx][:, None] - targets, -np.inf
    )
    realized_switch_gain = true_switch_gain.max(axis=1)
    decision_switch = predicted_switch_gain > float(abundle["thresholds"]["A1_behavior"])
    realized_switch_helps = realized_switch_gain > float(abundle["practical_gain_delta"])
    harmful = decision_switch & ~(realized_switch_gain > -float(abundle["practical_gain_delta"]))
    missed = ~decision_switch & realized_switch_helps

    uncertainty = {
        "tree_variance_of_best_switch": pred_std[np.arange(len(states)), best_switch],
        "loss_margin": loss_margin,
        "switch_gain_margin": np.abs(predicted_switch_gain),
        "tree_disagreement": tree_disagreement,
    }
    labels = {
        "harmful_switch": harmful,
        "missed_helpful_switch": missed,
    }
    rows = []
    for target_name, target in labels.items():
        if target.sum() < 10 or (~target).sum() < 10:
            continue
        for feature_name, values in uncertainty.items():
            rows.append(
                {
                    "target": target_name,
                    "feature": feature_name,
                    "positive_rate": float(target.mean()),
                    "auc_low_identifies_risk": float(
                        roc_auc_score(target, -values)
                    ),
                    "mean_for_risk": float(values[target].mean()),
                    "mean_for_safe": float(values[~target].mean()),
                }
            )
    diag = pd.DataFrame(rows)
    save_table(diag, "uncertainty_auc.csv", TASK)

    deciles = pd.qcut(
        uncertainty["switch_gain_margin"], 10, labels=False, duplicates="drop"
    )
    decile_table = pd.DataFrame(
        {
            "margin_decile": deciles,
            "harmful": harmful,
            "realized_switch_gain": realized_switch_gain,
            "decision_switch": decision_switch,
        }
    ).groupby("margin_decile", sort=True)["harmful"].mean()
    decile_table.to_csv(
        common.V2 / TASK / "harmful_rate_by_margin_decile.csv",
        index=True,
    )

    summary = {
        "states": int(len(states)),
        "decision_switch_rate": float(decision_switch.mean()),
        "harmful_switch_rate": float(harmful.mean()),
        "missed_helpful_rate": float(missed.mean()),
        "practical_delta": float(abundle["practical_gain_delta"]),
        "threshold_A1": float(abundle["thresholds"]["A1_behavior"]),
    }
    (common.V2 / TASK / "summary.json").write_text(json_dumps(summary))
    print(f"[{TASK}] done: {json_dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
