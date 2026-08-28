"""Task 1: formalize behavior_action_loss_regression_v2 as the main candidate.

Multi-output RandomForestRegressor on the 28 Behavior features with the
predicted-loss advantage score, evaluated with grouped family OOF and compared
against the Phase 1 baseline policy panel plus score-level regression metrics.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import common  # noqa: F401
from common import (
    V2_HEAVY,
    harmful_switch_rate,
    json_dumps,
    load_train_val,
    per_candidate_scores,
    policy_block,
    save_table,
    switch_fe_quantiles,
)

sys.path.insert(0, str(common.ROOT))

from behavior_with_ela.baselines import (  # noqa: E402
    _action_loss_family_oof_predictions,
    _fit_action_loss_model,
    _predict_action_loss_rows,
)
from behavior_with_ela.model import (  # noqa: E402
    fit_first_trigger_threshold,
    replay_first_trigger,
)

TASK = "task1"
MODEL_PROTOCOL = "behavior_action_loss_regression_v2"


def main() -> None:
    config, validation_config, bundle, delta, train, validation = load_train_val()
    heavy = V2_HEAVY / TASK
    heavy.mkdir(parents=True, exist_ok=True)

    print(f"[{TASK}] grouped family OOF for {MODEL_PROTOCOL}", flush=True)
    heavy.mkdir(parents=True, exist_ok=True)
    (common.V2 / TASK).mkdir(parents=True, exist_ok=True)
    oof_path = heavy / "oof_predictions.parquet"
    if oof_path.exists():
        oof = pd.read_parquet(oof_path)
        thresholds = pd.read_parquet(heavy / "thresholds.parquet")
        selected = float(
            thresholds.loc[thresholds["selected_threshold"], "threshold"].iloc[0]
        )
        runs = pd.read_parquet(heavy / "train_first_trigger_runs.parquet")
        print(f"[{TASK}] reused cached OOF: {len(oof)} rows", flush=True)
    else:
        oof = _action_loss_family_oof_predictions(train, config)
        thresholds, selected, runs = fit_first_trigger_threshold(
            action_rows=train, action_predictions=oof, practical_delta=delta
        )
        oof.to_parquet(heavy / "oof_predictions.parquet", index=False)
        thresholds.to_parquet(heavy / "thresholds.parquet", index=False)
        runs.to_parquet(heavy / "train_first_trigger_runs.parquet", index=False)
    oof["phase2_feature_group"] = "v2"

    model = _fit_action_loss_model(train, config, fold_number=90_001)
    predictions_val = _predict_action_loss_rows(
        model=model, action_rows=validation, practical_delta=delta
    )
    validation_runs = replay_first_trigger(
        action_rows=validation,
        action_predictions=predictions_val,
        threshold=selected,
        practical_delta=delta,
        default_algorithm=str(bundle["default_algorithm"]),
    )
    validation_runs.to_parquet(
        heavy / "validation_first_trigger_runs.parquet", index=False
    )

    # score-level metrics: MAE / Spearman / pairwise ranking / top-1 (incl. continue)
    def score_metrics(pred: pd.DataFrame) -> pd.DataFrame:
        from scipy.stats import spearmanr

        rows = []
        key = ["problem_id", "prefix_algorithm", "seed", "FE", "decision_opportunity_index"]
        for candidate, group in pred.groupby("candidate_action", sort=True):
            errors = (
                group["predicted_candidate_log10_loss"].to_numpy(dtype=float)
                - group["log10_action_loss"].to_numpy(dtype=float)
            )
            rho = float(
                spearmanr(
                    group["predicted_action_gain"].to_numpy(dtype=float),
                    group["action_gain_vs_continue"].to_numpy(dtype=float),
                ).statistic
            )
            rows.append(
                {
                    "candidate_action": str(candidate),
                    "rows": int(len(group)),
                    "loss_mae": float(np.abs(errors).mean()),
                    "loss_rmse": float(np.sqrt((errors**2).mean())),
                    "gain_spearman": rho,
                }
            )
        frame = pd.DataFrame(rows)

        left = pred[
            key + ["candidate_action", "predicted_candidate_log10_loss",
                   "predicted_action_gain", "log10_action_loss"]
        ]
        merged = left.merge(left, on=key, suffixes=("", "_other"))
        merged = merged.loc[
            merged["candidate_action"].astype(str)
            < merged["candidate_action_other"].astype(str)
        ]
        pred_higher = (
            merged["predicted_action_gain"]
            > merged["predicted_action_gain_other"]
        )
        true_higher = (
            merged["log10_action_loss"] < merged["log10_action_loss_other"]
        )
        pairwise = float((pred_higher == true_higher).mean())

        continue_rows = train.loc[
            train["action_equals_prefix"].astype(bool),
            key + ["candidate_action", "log10_action_loss"],
        ].rename(
            columns={
                "candidate_action": "prefix_action",
                "log10_action_loss": "continue_true_loss",
            }
        )
        state_pred = pred.sort_values(
            ["predicted_candidate_log10_loss"], ascending=True, kind="mergesort"
        ).groupby(key, sort=False).head(1)[
            key + ["candidate_action", "predicted_candidate_log10_loss",
                   "predicted_continue_log10_loss"]
        ]
        state_true = train.loc[
            train["split"].astype(str).eq("bbob_train")
        ].groupby(key, sort=False)["log10_action_loss"].min().rename(
            "state_best_true_loss"
        ).reset_index()
        top = state_pred.merge(continue_rows, on=key, how="left")
        top = top.merge(state_true, on=key, how="left")
        pred_best_loss = np.minimum(
            top["predicted_candidate_log10_loss"].to_numpy(dtype=float),
            top["predicted_continue_log10_loss"].to_numpy(dtype=float),
        )
        pred_best_is_continue = (
            top["predicted_continue_log10_loss"].to_numpy(dtype=float)
            <= top["predicted_candidate_log10_loss"].to_numpy(dtype=float)
        )
        top["pred_best_action"] = np.where(
            pred_best_is_continue, top["prefix_action"], top["candidate_action"]
        )
        switch_losses = pred.pivot_table(
            index=key, columns="candidate_action", values="log10_action_loss",
            aggfunc="first",
        ).reset_index()
        top = top.merge(switch_losses, on=key, suffixes=("", "_sw"))
        switch_frame = top[list(common.PORTFOLIO)]
        true_switch_min = switch_frame.min(axis=1)
        true_switch_argmin = switch_frame.idxmin(axis=1)
        true_best_is_continue = (
            top["continue_true_loss"].to_numpy(dtype=float)
            <= true_switch_min.to_numpy(dtype=float)
        )
        top["true_best_action"] = np.where(
            true_best_is_continue, top["prefix_action"], true_switch_argmin
        )
        top1 = float(
            (
                top["pred_best_action"].astype(str)
                == top["true_best_action"].astype(str)
            ).mean()
        )
        frame.attrs["pairwise_ranking_accuracy"] = pairwise
        frame.attrs["top1_action_accuracy"] = top1
        return frame

    oof_scores = score_metrics(oof)
    oof_scores.to_csv(common.V2 / TASK / "score_metrics_oof.csv", index=False)

    policy_train = {
        **policy_block(runs),
        "harmful_switch_rate": harmful_switch_rate(runs, delta),
        **switch_fe_quantiles(runs),
    }
    policy_val = {
        **policy_block(validation_runs),
        "harmful_switch_rate": harmful_switch_rate(validation_runs, delta),
        **switch_fe_quantiles(validation_runs),
    }

    # unified comparison panel
    panel = pd.read_parquet(
        common.RESULTS / "baselines/phase1/policy_summary.parquet"
    )
    v2_rows = []
    for split, policy in (
        ("bbob_train_oof_or_reference", policy_train),
        ("bbob_validation", policy_val),
    ):
        v2_rows.append(
            {
                "evaluation_split": split,
                "policy_name": MODEL_PROTOCOL,
                "run_count": policy["run_count"],
                "function_balanced_mean_gain": policy["function_balanced_mean_gain"],
                "function_balanced_mean_normalized_regret": policy[
                    "function_balanced_mean_normalized_regret"
                ],
                "switch_rate": policy["switch_rate"],
                "acceptable_policy_rate": policy["acceptable_policy_rate"],
            }
        )
    unified = pd.concat([panel, pd.DataFrame(v2_rows)], ignore_index=True)
    save_table(unified, "unified_policy_panel.parquet", TASK)

    summary = {
        "model_protocol": MODEL_PROTOCOL,
        "decision_threshold": float(selected),
        "pairwise_ranking_accuracy_oof": float(
            oof_scores.attrs["pairwise_ranking_accuracy"]
        ),
        "top1_action_accuracy_oof": float(oof_scores.attrs["top1_action_accuracy"]),
        "policy_train_oof": policy_train,
        "policy_validation": policy_val,
    }
    (common.V2 / TASK / "summary.json").write_text(json_dumps(summary))
    print(f"[{TASK}] done: threshold={selected:.6f}", flush=True)


if __name__ == "__main__":
    main()
