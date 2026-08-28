"""Task 9E: re-validate selected MA-BBOB augmentation on the regression carrier.

R-B  = BBOB train only (28 Behavior features, identical multi-output RF)
R-BM = BBOB + selected MA-BBOB (the current v2 protocol, restated from artifacts)

Same hyperparameters, same family-OOF protocol, same threshold protocol,
same practical delta, same validation. Additionally a MA-BBOB definition-grouped
5-fold OOF regression diagnostic. No objective evaluation is executed.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common  # noqa: F401
from common import (
    V2_HEAVY,
    json_dumps,
    load_train_val,
    noise_deltas,
    policy_metrics,
    save_heavy_table,
    save_table,
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

TASK = "task5"


def score_metrics(pred: pd.DataFrame, action_rows: pd.DataFrame) -> dict:
    from behavior_with_ela.model import STATE_KEY

    rows = []
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
                "gain_spearman": rho,
            }
        )
    per_candidate = pd.DataFrame(rows)

    key = list(STATE_KEY)
    left = pred[
        key
        + [
            "candidate_action",
            "predicted_candidate_log10_loss",
            "predicted_action_gain",
            "log10_action_loss",
        ]
    ]
    merged = left.merge(left, on=key, suffixes=("", "_other"))
    merged = merged.loc[
        merged["candidate_action"].astype(str)
        < merged["candidate_action_other"].astype(str)
    ]
    pairwise = float(
        (
            (
                merged["predicted_action_gain"]
                > merged["predicted_action_gain_other"]
            )
            == (
                merged["log10_action_loss"] < merged["log10_action_loss_other"]
            )
        ).mean()
    )

    continue_rows = (
        action_rows.loc[action_rows["action_equals_prefix"].astype(bool)]
        .sort_values(key, kind="mergesort")
        .groupby(key, sort=False, as_index=False)
        .nth(0)[key + ["candidate_action", "log10_action_loss"]]
        .rename(
            columns={
                "candidate_action": "prefix_action",
                "log10_action_loss": "continue_true_loss",
            }
        )
    )
    switch_losses = pred.pivot_table(
        index=key, columns="candidate_action", values="log10_action_loss", aggfunc="first"
    ).reset_index()
    state_pred = pred.sort_values(
        "predicted_candidate_log10_loss", ascending=True, kind="mergesort"
    ).groupby(key, sort=False).head(1)[
        key + ["candidate_action", "predicted_candidate_log10_loss"]
    ]
    top = state_pred.merge(continue_rows, on=key, how="left")
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
    pred_best_is_continue = (
        top["continue_true_loss"].to_numpy(dtype=float)
        <= top["predicted_candidate_log10_loss"].to_numpy(dtype=float)
    )
    top["pred_best_action"] = np.where(
        pred_best_is_continue, top["prefix_action"], top["candidate_action"]
    )
    top1 = float(
        (
            top["pred_best_action"].astype(str)
            == top["true_best_action"].astype(str)
        ).mean()
    )
    return {
        "per_candidate": per_candidate,
        "pairwise_ranking_accuracy": pairwise,
        "top1_action_accuracy": top1,
    }


def main() -> None:
    config, validation_config, bundle, delta, train, validation = load_train_val()
    deltas = noise_deltas(config)
    default_algorithm = str(bundle["default_algorithm"])
    train_bbob = train.loc[train["suite"].astype(str).eq("bbob")].copy()

    heavy = common.V3_HEAVY / TASK
    heavy.mkdir(parents=True, exist_ok=True)
    oof_path = heavy / "oof_rb.parquet"
    if oof_path.exists():
        oof_rb = pd.read_parquet(oof_path)
        thresholds_rb = pd.read_parquet(heavy / "thresholds_rb.parquet")
        selected_rb = float(
            thresholds_rb.loc[thresholds_rb["selected_threshold"], "threshold"].iloc[0]
        )
        runs_rb_train = pd.read_parquet(heavy / "runs_rb_train.parquet")
        print(f"[{TASK}] reused cached R-B OOF", flush=True)
    else:
        print(f"[{TASK}] R-B family OOF", flush=True)
        oof_rb = _action_loss_family_oof_predictions(train_bbob, config)
        thresholds_rb, selected_rb, runs_rb_train = fit_first_trigger_threshold(
            action_rows=train_bbob,
            action_predictions=oof_rb,
            practical_delta=delta,
        )
        oof_rb.to_parquet(oof_path, index=False)
        thresholds_rb.to_parquet(heavy / "thresholds_rb.parquet", index=False)
        runs_rb_train.to_parquet(heavy / "runs_rb_train.parquet", index=False)
    runs_rb_train["policy_name"] = "rb_bbob_only_regression"

    scores_rb = score_metrics(oof_rb, train_bbob)

    model_rb = _fit_action_loss_model(train_bbob, config, fold_number=90_003)
    validation_scores_rb = _predict_action_loss_rows(
        model=model_rb, action_rows=validation, practical_delta=delta
    )
    runs_rb_validation = replay_first_trigger(
        action_rows=validation,
        action_predictions=validation_scores_rb,
        threshold=selected_rb,
        practical_delta=delta,
        default_algorithm=default_algorithm,
    )
    runs_rb_validation["policy_name"] = "rb_bbob_only_regression"

    # restated R-BM (= v2) from the fixed task1 artifacts
    v2_train_runs, v2_validation_runs, v2_threshold = (
        common.load_v2_first_trigger_runs()
    )
    v2_train_runs["policy_name"] = "rbm_bbob_mabbob_regression_v2"
    v2_validation_runs["policy_name"] = "rbm_bbob_mabbob_regression_v2"

    rows = []
    for model_name, runs_train, runs_val in (
        ("rb_bbob_only_regression", runs_rb_train, runs_rb_validation),
        ("rbm_bbob_mabbob_regression_v2", v2_train_runs, v2_validation_runs),
    ):
        for split, runs in (
            ("bbob_train_oof", runs_train),
            ("bbob_validation", runs_val),
        ):
            rows.append(
                {
                    "model": model_name,
                    "evaluation_split": split,
                    **policy_metrics(runs, deltas),
                }
            )
    policy_summary = pd.DataFrame(rows)

    # paired gain over SBS on validation (deployability context)
    sbs_rows = []
    for model_name, runs in (
        ("rb_bbob_only_regression", runs_rb_validation),
        ("rbm_bbob_mabbob_regression_v2", v2_validation_runs),
    ):
        paired = common.gain_over_sbs(runs)
        sbs_rows.append(
            {
                "model": model_name,
                "all_prefix_function_balanced_gain_over_sbs": common.function_balanced(
                    paired["gain_over_sbs"], paired["cv_group_id"]
                ),
                "cmaes_start_function_balanced_gain_over_sbs": common.function_balanced(
                    paired.loc[
                        paired["prefix_algorithm"].astype(str).eq("cmaes"),
                        "gain_over_sbs",
                    ],
                    paired.loc[
                        paired["prefix_algorithm"].astype(str).eq("cmaes"),
                        "cv_group_id",
                    ],
                ),
            }
        )
    sbs_table = pd.DataFrame(sbs_rows)

    # per-candidate regression quality table
    quality_rows = []
    for model_name, pred, action_frame in (
        ("rb_bbob_only_regression", scores_rb["per_candidate"], train_bbob),
    ):
        frame = pred.copy()
        frame.insert(0, "model", model_name)
        quality_rows.append(frame)
    quality = pd.concat(quality_rows, ignore_index=True)

    score_rows = [
        {
            "model": "rb_bbob_only_regression",
            "pairwise_ranking_accuracy_oof": scores_rb["pairwise_ranking_accuracy"],
            "top1_action_accuracy_oof": scores_rb["top1_action_accuracy"],
        }
    ]
    # restated v2 OOF score metrics from task1
    task1_summary = json.loads(
        (common.V3.parent / "analysis_v2/task1/summary.json").read_text()
    )
    score_rows.append(
        {
            "model": "rbm_bbob_mabbob_regression_v2",
            "pairwise_ranking_accuracy_oof": task1_summary[
                "pairwise_ranking_accuracy_oof"
            ],
            "top1_action_accuracy_oof": task1_summary["top1_action_accuracy_oof"],
        }
    )
    score_table = pd.DataFrame(score_rows)

    # MA-BBOB definition-grouped 5-fold OOF regression diagnostic
    assignment_path = (
        common.V3.parent / "analysis_v2/task7/definition_fold_assignment.json"
    )
    fold_of = json.loads(assignment_path.read_text())
    ma = train.loc[train["suite"].astype(str).eq("mabbob")].copy()
    ma["definition_fold"] = ma["function_id"].astype(str).map(fold_of)
    if ma["definition_fold"].isna().any():
        raise RuntimeError("MA definition fold assignment is incomplete")
    definition_rows = []
    for fold in range(5):
        fit_rows = pd.concat(
            [
                train.loc[train["suite"].astype(str).eq("bbob")],
                ma.loc[~ma["definition_fold"].eq(fold)],
            ],
            ignore_index=True,
        )
        model = _fit_action_loss_model(fit_rows, config, fold_number=70_100 + fold)
        held = ma.loc[ma["definition_fold"].eq(fold)]
        predictions = _predict_action_loss_rows(
            model=model, action_rows=held, practical_delta=delta
        )
        for definition, group in predictions.groupby("function_id", sort=True):
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
            definition_rows.append(
                {
                    "fold": fold,
                    "definition": str(definition),
                    "rows": int(len(group)),
                    "loss_mae": float(np.abs(errors).mean()),
                    "gain_spearman": rho,
                }
            )
        print(f"[{TASK}] MA definition fold {fold} done", flush=True)
    definition_table = pd.DataFrame(definition_rows)

    save_heavy_table(runs_rb_validation, "runs_rb_validation.parquet", TASK)
    save_table(policy_summary, "augmentation_regression_policy_summary.parquet", TASK)
    save_table(score_table, "augmentation_regression_score_summary.parquet", TASK)
    save_table(quality, "rb_per_candidate_regression_quality_oof.csv", TASK)
    save_table(definition_table, "mabbob_definition_regression_oof.csv", TASK)
    save_table(sbs_table, "gain_over_sbs_validation.parquet", TASK)
    payload = {
        "rb_threshold": float(selected_rb),
        "rbm_threshold": float(v2_threshold),
        "definition_fold_assignment_source": "analysis_v2/task7/definition_fold_assignment.json",
    }
    save_table(payload, "summary.json", TASK)
    print(f"[{TASK}] done", flush=True)
    print(policy_summary.to_string(), flush=True)
    print(score_table.to_string(), flush=True)
    print(sbs_table.to_string(), flush=True)
    print(
        definition_table.groupby("fold")[["loss_mae", "gain_spearman"]]
        .mean()
        .to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()
