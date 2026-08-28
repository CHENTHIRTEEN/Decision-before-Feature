"""Task 2: practical gain delta sensitivity across repetition quantiles.

Retrains the three-class classifier and time-only model per quantile on the
relabelled train data (grouped family OOF only). The action-loss regression
carrier uses its existing OOF scores with per-delta threshold replays.
Validation is evaluated only at the originally fixed quantile (0.95).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import common  # noqa: F401
from common import (
    TRAIN_CONFIG,
    V2_HEAVY,
    json_dumps,
    load_train_val,
    harmful_switch_rate,
    policy_block,
    save_table,
)

sys.path.insert(0, str(common.ROOT))

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS as COLS  # noqa: E402
from behavior_with_ela.baselines import (  # noqa: E402
    _action_loss_family_oof_predictions,
    _fit_action_loss_model,
    _predict_action_loss_rows,
)
from behavior_with_ela.model import (  # noqa: E402
    _action_metrics,
    _family_oof_predictions,
    apply_practical_gain_delta,
    estimate_noise_gain_delta,
    fit_first_trigger_threshold,
    predict_action_rows,
    read_action_datasets,
    replay_first_trigger,
)
from behavior_with_ela.protocol import load_experiment_config  # noqa: E402

TASK = "task2"
QUANTILES = (0.50, 0.75, 0.80, 0.90, 0.95)
DOMAIN_DELTA = 0.05


def main() -> None:
    config, validation_config, bundle, fixed_delta, train_raw, validation = (
        load_train_val()
    )
    heavy = V2_HEAVY / TASK
    heavy.mkdir(parents=True, exist_ok=True)

    # action repetition frame for noise delta (BBOB switch actions only)
    reps_parts = [
        pd.read_parquet(f)
        for f in sorted(
            (common.RESULTS / "actions").glob("*_train/*/dimension_10/action_repetitions.parquet")
        )
    ]
    reps = pd.concat(reps_parts, ignore_index=True)

    action_loss_oof = pd.read_parquet(
        common.RESULTS
        / "model/local_landscape_increment_action_loss/train_oof_action_predictions.parquet",
        columns=[
            "phase2_feature_group",
            "problem_id", "prefix_algorithm", "seed", "FE",
            "decision_opportunity_index", "candidate_action",
            "predicted_action_class", "predicted_improve_probability",
        ],
    )
    action_loss_oof = action_loss_oof.loc[
        action_loss_oof["phase2_feature_group"].eq("A1_behavior")
    ].drop(columns="phase2_feature_group")

    noise_rows = []
    summary_rows = []
    for quantile in QUANTILES:
        noise_table, noise_delta = estimate_noise_gain_delta(
            reps, quantile=float(quantile)
        )
        practical = max(DOMAIN_DELTA, float(noise_delta))
        noise_table["quantile"] = quantile
        noise_rows.append(noise_table)

        train = apply_practical_gain_delta(train_raw.copy(), practical)
        labels = train["action_gain_class"].value_counts(normalize=True)
        print(
            f"[{TASK}] q={quantile}: noise={noise_delta:.4f} practical={practical:.4f} "
            f"labels={labels.round(3).to_dict()}",
            flush=True,
        )

        # three-class behavior classifier retrained on relabelled data
        oof = _family_oof_predictions(train, config)
        thresholds, selected, runs = fit_first_trigger_threshold(
            action_rows=train, action_predictions=oof, practical_delta=practical
        )
        metrics = _action_metrics(oof)
        summary_rows.append(
            {
                "carrier": "action_gain_classifier",
                "quantile": quantile,
                "practical_delta": practical,
                "improve_share": float(labels.get("improve", 0.0)),
                "equivalent_share": float(labels.get("equivalent", 0.0)),
                "degrade_share": float(labels.get("degrade", 0.0)),
                **metrics,
                **policy_block(runs),
                "harmful_switch_rate": harmful_switch_rate(runs, practical),
            }
        )
        oof.to_parquet(heavy / f"oof_classifier_q{quantile:.2f}.parquet", index=False)

        # time-only model retrained on relabelled data
        oof_time = _family_oof_predictions(
            train, config, feature_columns=("bf_fe_ratio",)
        )
        t_thresholds, t_selected, t_runs = fit_first_trigger_threshold(
            action_rows=train,
            action_predictions=oof_time,
            practical_delta=practical,
        )
        summary_rows.append(
            {
                "carrier": "time_only",
                "quantile": quantile,
                "practical_delta": practical,
                "improve_share": float(labels.get("improve", 0.0)),
                "equivalent_share": float(labels.get("equivalent", 0.0)),
                "degrade_share": float(labels.get("degrade", 0.0)),
                **policy_block(t_runs),
                "harmful_switch_rate": harmful_switch_rate(t_runs, practical),
            }
        )

        # action-loss regression carrier: fixed scores, per-delta threshold replay
        al_thresholds, al_selected, al_runs = fit_first_trigger_threshold(
            action_rows=train,
            action_predictions=action_loss_oof,
            practical_delta=practical,
        )
        summary_rows.append(
            {
                "carrier": "action_loss_regression",
                "quantile": quantile,
                "practical_delta": practical,
                "improve_share": float(labels.get("improve", 0.0)),
                "equivalent_share": float(labels.get("equivalent", 0.0)),
                "degrade_share": float(labels.get("degrade", 0.0)),
                **policy_block(al_runs),
                "harmful_switch_rate": harmful_switch_rate(al_runs, practical),
            }
        )
        print(
            f"[{TASK}] q={quantile} done: classifier gain="
            f"{summary_rows[-3]['function_balanced_mean_gain']:.4f} "
            f"time={summary_rows[-2]['function_balanced_mean_gain']:.4f} "
            f"action-loss={summary_rows[-1]['function_balanced_mean_gain']:.4f}",
            flush=True,
        )

    noise_all = pd.concat(noise_rows, ignore_index=True)
    save_table(noise_all, "noise_delta_per_family.parquet", TASK)
    summary_table = pd.DataFrame(summary_rows)
    save_table(summary_table, "delta_sensitivity_summary.parquet", TASK)

    # per-family noise distribution and strata at the fixed quantile 0.95
    fixed_noise = noise_all.loc[noise_all["quantile"].eq(0.95)]
    distribution = {
        "median": float(fixed_noise["function_noise_gain_delta"].median()),
        "iqr": float(
            fixed_noise["function_noise_gain_delta"].quantile(0.75)
            - fixed_noise["function_noise_gain_delta"].quantile(0.25)
        ),
        "q90": float(fixed_noise["function_noise_gain_delta"].quantile(0.90)),
        "q95": float(fixed_noise["function_noise_gain_delta"].quantile(0.95)),
        "max": float(fixed_noise["function_noise_gain_delta"].max()),
    }
    reps_bbob = reps.loc[
        reps["suite"].astype(str).eq("bbob")
        & ~reps["action_equals_prefix"].astype(bool)
        & reps["action_status"].astype(str).eq("completed")
    ].copy()
    reps_bbob["phase"] = pd.cut(
        reps_bbob["FE"].to_numpy(dtype=float) / reps_bbob["FE_total"].to_numpy(dtype=float),
        bins=[0.0, 1 / 3, 2 / 3, 1.0],
        labels=["early", "mid", "late"],
    )
    strata = []
    for column in ("prefix_algorithm", "candidate_action", "phase"):
        grouped = (
            reps_bbob.groupby(column, sort=True, observed=True)["action_gain_vs_continue"]
            .std()
        )
        for value, std in grouped.items():
            strata.append(
                {
                    "stratum": column,
                    "value": str(value),
                    "within_state_gain_std": float(std),
                }
            )
    (common.V2 / TASK).mkdir(parents=True, exist_ok=True)
    (common.V2 / TASK / "noise_distribution_q95.json").write_text(
        json_dumps({"distribution": distribution, "strata_std": strata})
    )
    print(f"[{TASK}] done", flush=True)


if __name__ == "__main__":
    main()
