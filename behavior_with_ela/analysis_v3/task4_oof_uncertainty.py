"""Task 9D: corrected OOF action-uncertainty diagnostic for the v2 carrier.

Two fixes over analysis_v2 task6:
1. risk labels use the realized gain of the *selected* switch action, not the
   best observed switch gain (the old label was an unattainable upper bound);
2. tree-level uncertainty is computed from family-OOF fold models, not from the
   full-train ensemble on its own training states.

Harmful definitions: H0 gain<0, H50 gain<-delta_50, H95 gain<-delta_95.
No objective evaluation is executed.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import common  # noqa: F401
from common import (
    V2_HEAVY,
    json_dumps,
    load_train_val,
    noise_deltas,
    save_heavy_table,
    save_table,
)

sys.path.insert(0, str(common.ROOT))

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS  # noqa: E402
from behavior_with_ela.baselines import (  # noqa: E402
    _action_loss_state_matrix,
    _fit_action_loss_model,
)
from behavior_with_ela.model import _ma_overlaps_heldout  # noqa: E402

TASK = "task4"


def vote_entropy(counts: np.ndarray) -> np.ndarray:
    total = counts.sum(axis=1)
    probability = counts / np.maximum(total, 1)[:, None]
    log_probability = np.where(probability > 0.0, np.log2(probability), 0.0)
    return -(probability * log_probability).sum(axis=1)


def auc_rows(
    labels: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for label_name, label in labels.items():
        if label.sum() < 10 or (~label).sum() < 10:
            continue
        for feature_name, values in features.items():
            finite = np.isfinite(values)
            if finite.sum() < len(values):
                continue
            auc_value = float(roc_auc_score(label, values))
            rows.append(
                {
                    "label": label_name,
                    "feature": feature_name,
                    "positive_rate": float(label.mean()),
                    "auc_high_value_risk": auc_value,
                    "auc_low_value_risk": float(roc_auc_score(label, -values)),
                    "mean_value_risk": float(values[label].mean()),
                    "mean_value_safe": float(values[~label].mean()),
                }
            )
    return pd.DataFrame(rows)


def pr_auc_rows(
    labels: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for label_name, label in labels.items():
        if label.sum() < 10 or (~label).sum() < 10:
            continue
        for feature_name, values in features.items():
            # orient the score so higher means more risk
            auc_high = float(roc_auc_score(label, values))
            oriented = values if auc_high >= 0.5 else -values
            rows.append(
                {
                    "label": label_name,
                    "feature": feature_name,
                    "pr_auc": float(average_precision_score(label, oriented)),
                    "orientation": "high_is_risk" if auc_high >= 0.5 else "low_is_risk",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    config, validation_config, bundle, delta, train, validation = load_train_val()
    deltas = noise_deltas(config)
    delta_95 = float(deltas["delta_95"])
    delta_50 = float(deltas["delta_50"])

    oof = pd.read_parquet(V2_HEAVY / "task1/oof_predictions.parquet")
    thresholds = pd.read_parquet(V2_HEAVY / "task1/thresholds.parquet")
    threshold = float(
        thresholds.loc[thresholds["selected_threshold"], "threshold"].iloc[0]
    )

    bbob = train.loc[train["suite"].astype(str).eq("bbob")].copy()
    families = tuple(sorted(set(bbob["family"].astype(str))))

    state_blocks = []
    for fold_number, heldout_family in enumerate(families, start=1):
        heldout_functions = set(
            bbob.loc[bbob["family"].astype(str).eq(heldout_family), "function_id"]
            .astype(str)
        )
        train_mask = ~(
            train["suite"].astype(str).eq("bbob")
            & train["family"].astype(str).eq(heldout_family)
        )
        ma_safe = ~train.apply(
            lambda row: _ma_overlaps_heldout(row, heldout_functions), axis=1
        )
        fold_train = train.loc[train_mask & ma_safe]
        model = _fit_action_loss_model(
            fold_train, config, fold_number=fold_number
        )
        regressor = model.named_steps["regressor"]
        imputer = model.named_steps["imputer"]

        fold_eval = bbob.loc[bbob["family"].astype(str).eq(heldout_family)]
        states, targets = _action_loss_state_matrix(fold_eval, common.PORTFOLIO)
        x = imputer.transform(
            states[list(SELECTOR_BEHAVIOR_FEATURE_COLUMNS)].to_numpy(dtype=float)
        )
        tree_prediction = np.stack(
            [tree.predict(x) for tree in regressor.estimators_], axis=0
        )
        mean_prediction = tree_prediction.mean(axis=0)

        prefix = states["prefix_algorithm"].astype(str).to_numpy()
        prefix_index = np.array([common.PORTFOLIO.index(a) for a in prefix])
        rows = np.arange(len(states))
        predicted_continue = mean_prediction[rows, prefix_index]
        switch_mask = np.ones_like(mean_prediction, dtype=bool)
        switch_mask[rows, prefix_index] = False
        predicted_switch_gain = np.where(
            switch_mask, predicted_continue[:, None] - mean_prediction, -np.inf
        )
        best_switch = predicted_switch_gain.argmax(axis=1)
        best_switch_gain = predicted_switch_gain.max(axis=1)
        decision_switch = best_switch_gain > threshold

        realized_selected_gain = (
            targets[rows, prefix_index] - targets[rows, best_switch]
        )
        true_switch_gain = np.where(
            switch_mask, targets[rows, prefix_index][:, None] - targets, -np.inf
        )
        realized_best_gain = true_switch_gain.max(axis=1)
        exists_positive_gain = np.where(
            switch_mask, targets[rows, prefix_index][:, None] - targets, -np.inf
        ).max(axis=1) > 0.0

        # uncertainty features
        sorted_losses = np.sort(mean_prediction, axis=1)
        tree_best = tree_prediction.argmin(axis=2)
        ensemble_best = mean_prediction.argmin(axis=1)
        disagreement = (tree_best != ensemble_best[None, :]).mean(axis=0)
        vote_counts = np.stack(
            [
                (tree_best == action).sum(axis=0)
                for action in range(len(common.PORTFOLIO))
            ],
            axis=1,
        ).astype(float)
        selected_tree_loss = tree_prediction[:, rows, best_switch]
        block = pd.DataFrame(
            {
                "problem_id": states["problem_id"].to_numpy(),
                "prefix_algorithm": prefix,
                "seed": states["seed"].to_numpy(dtype=int),
                "FE": states["FE"].to_numpy(dtype=int),
                "decision_opportunity_index": states[
                    "decision_opportunity_index"
                ].to_numpy(dtype=int),
                "cv_group_id": states["cv_group_id"].to_numpy(),
                "predicted_continue_loss": predicted_continue,
                "selected_switch_action": [
                    common.PORTFOLIO[index] for index in best_switch
                ],
                "predicted_selected_switch_gain": best_switch_gain,
                "top1_top2_loss_margin": sorted_losses[:, 1] - sorted_losses[:, 0],
                "tree_argmin_disagreement": disagreement,
                "selected_action_tree_std": selected_tree_loss.std(axis=0, ddof=1),
                "tree_argmin_vote_entropy": vote_entropy(vote_counts),
                "decision_switch": decision_switch,
                "realized_selected_switch_gain": realized_selected_gain,
                "realized_best_switch_gain": realized_best_gain,
                "exists_true_positive_gain": exists_positive_gain,
                "oof_fold": fold_number,
                "heldout_family": heldout_family,
            }
        )
        state_blocks.append(block)
        print(
            f"[{TASK}] fold {fold_number} ({heldout_family}): "
            f"states={len(states)}, switch_rate={decision_switch.mean():.4f}",
            flush=True,
        )

    table = pd.concat(state_blocks, ignore_index=True)
    expected_states = bbob[
        ["problem_id", "prefix_algorithm", "seed", "FE", "decision_opportunity_index"]
    ].drop_duplicates()
    if len(table) != len(expected_states):
        raise RuntimeError("OOF uncertainty coverage differs from BBOB states")

    # consistency with the task1 OOF predictions (identical fold models):
    # selected-switch rows must reproduce the same predicted losses
    oof_switch = oof[
        [
            "problem_id", "prefix_algorithm", "seed", "FE",
            "decision_opportunity_index", "candidate_action",
            "predicted_candidate_log10_loss", "predicted_continue_log10_loss",
        ]
    ]
    check = table.merge(
        oof_switch,
        left_on=[
            "problem_id", "prefix_algorithm", "seed", "FE",
            "decision_opportunity_index", "selected_switch_action",
        ],
        right_on=[
            "problem_id", "prefix_algorithm", "seed", "FE",
            "decision_opportunity_index", "candidate_action",
        ],
        how="inner",
        validate="one_to_one",
    )
    if len(check) != len(table):
        raise RuntimeError(
            "selected switch rows are not covered by task1 OOF predictions: "
            f"matched={len(check)}, states={len(table)}"
        )
    loss_deviation = float(
        np.abs(
            (
                check["predicted_continue_loss"]
                - check["predicted_selected_switch_gain"]
            ).to_numpy(dtype=float)
            - check["predicted_candidate_log10_loss"].to_numpy(dtype=float)
        ).max()
    )
    continue_deviation = float(
        np.abs(
            check["predicted_continue_loss"].to_numpy(dtype=float)
            - check["predicted_continue_log10_loss"].to_numpy(dtype=float)
        ).max()
    )
    if max(loss_deviation, continue_deviation) > 1e-9:
        raise RuntimeError(
            "OOF tree-ensemble means deviate from task1 OOF predictions: "
            f"{max(loss_deviation, continue_deviation)}"
        )

    labels = {
        "harmful_H0": table["decision_switch"].to_numpy()
        & (table["realized_selected_switch_gain"].to_numpy(dtype=float) < 0.0),
        "harmful_H50": table["decision_switch"].to_numpy()
        & (
            table["realized_selected_switch_gain"].to_numpy(dtype=float)
            < -delta_50
        ),
        "harmful_H95": table["decision_switch"].to_numpy()
        & (
            table["realized_selected_switch_gain"].to_numpy(dtype=float)
            < -delta_95
        ),
    }
    decision = table["decision_switch"].to_numpy()
    labels["missed_helpful_switch_gt_delta95"] = (~decision) & (
        table["realized_best_switch_gain"].to_numpy(dtype=float) > delta_95
    )
    features = {
        "predicted_selected_switch_gain": table[
            "predicted_selected_switch_gain"
        ].to_numpy(dtype=float),
        "top1_top2_loss_margin": table["top1_top2_loss_margin"].to_numpy(dtype=float),
        "tree_argmin_disagreement": table["tree_argmin_disagreement"].to_numpy(
            dtype=float
        ),
        "selected_action_tree_std": table["selected_action_tree_std"].to_numpy(
            dtype=float
        ),
        "tree_argmin_vote_entropy": table["tree_argmin_vote_entropy"].to_numpy(
            dtype=float
        ),
    }

    roc_table = auc_rows(labels, features)
    pr_table = pr_auc_rows(labels, features)
    save_table(roc_table, "uncertainty_roc_auc.csv", TASK)
    save_table(pr_table, "uncertainty_pr_auc.csv", TASK)

    decile_rows = []
    for feature_name in (
        "predicted_selected_switch_gain",
        "tree_argmin_disagreement",
        "selected_action_tree_std",
    ):
        values = features[feature_name]
        deciles = pd.qcut(values, 10, labels=False, duplicates="drop")
        for label_name, label in labels.items():
            frame = pd.DataFrame(
                {"decile": deciles, "label": label}
            ).groupby("decile", sort=True)["label"].agg(["mean", "sum", "size"])
            for decile, row in frame.iterrows():
                decile_rows.append(
                    {
                        "feature": feature_name,
                        "label": label_name,
                        "decile": int(decile),
                        "risk_rate": float(row["mean"]),
                        "risk_count": int(row["sum"]),
                        "states": int(row["size"]),
                    }
                )
    save_table(pd.DataFrame(decile_rows), "risk_by_decile.csv", TASK)

    # contrast with the old (incorrect) best-switch risk label
    old_label = decision & (
        table["realized_best_switch_gain"].to_numpy(dtype=float) < -delta_95
    )
    old_new_contrast = {
        "decision_switch_rate": float(decision.mean()),
        "harmful_H95_selected_label_count": int(labels["harmful_H95"].sum()),
        "harmful_H95_old_best_switch_label_count": int(old_label.sum()),
        "old_label_includes_selected_harmless_runs": int(
            (old_label & ~labels["harmful_H95"]).sum()
        ),
        "selected_label_misses_old_label_runs": int(
            (labels["harmful_H95"] & ~old_label).sum()
        ),
        "harmful_H0_count": int(labels["harmful_H0"].sum()),
        "harmful_H50_count": int(labels["harmful_H50"].sum()),
        "missed_helpful_count": int(labels["missed_helpful_switch_gt_delta95"].sum()),
    }

    save_heavy_table(table, "oof_action_uncertainty_states.parquet", TASK)
    payload = {
        "threshold": threshold,
        "delta_50": delta_50,
        "delta_95": delta_95,
        "states": int(len(table)),
        "contrast_old_vs_new_label": old_new_contrast,
    }
    save_table(payload, "summary.json", TASK)
    print(f"[{TASK}] done", flush=True)
    print(json_dumps(payload), flush=True)
    print(roc_table.to_string(), flush=True)
    print(pr_table.to_string(), flush=True)


if __name__ == "__main__":
    main()
