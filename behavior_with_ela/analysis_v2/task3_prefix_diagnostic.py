"""Task 3: CMA-ES candidate weakness diagnosis + prefix-aware diagnostic model.

Model U = Behavior only (Phase 1 reference)
Model P = Behavior + one-hot prefix algorithm identity (diagnostic ablation)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import common  # noqa: F401
from common import (
    V2_HEAVY,
    add_prefix_onehot,
    harmful_switch_rate,
    json_dumps,
    load_train_val,
    policy_block,
    save_table,
)

sys.path.insert(0, str(common.ROOT))

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS as COLS  # noqa: E402
from behavior_with_ela.model import (  # noqa: E402
    _action_metrics,
    _family_oof_predictions,
    _fit_models,
    fit_first_trigger_threshold,
    predict_action_rows,
    replay_first_trigger,
)

TASK = "task3"
FEATURES_P = tuple(COLS) + ("prefix_pso", "prefix_shade", "prefix_cmaes")


def main() -> None:
    config, validation_config, bundle, delta, train, validation = load_train_val()
    train = add_prefix_onehot(train)
    validation = add_prefix_onehot(validation)
    heavy = V2_HEAVY / TASK
    heavy.mkdir(parents=True, exist_ok=True)

    model_u_oof = pd.read_parquet(
        common.RESULTS / "model/behavior_action_gain/oof_action_predictions.parquet"
    )

    print(f"[{TASK}] Model P: {len(FEATURES_P)} features", flush=True)
    oof_p = _family_oof_predictions(train, config, feature_columns=FEATURES_P)
    thresholds_p, selected_p, runs_p = fit_first_trigger_threshold(
        action_rows=train, action_predictions=oof_p, practical_delta=delta
    )
    oof_p.to_parquet(heavy / "oof_model_p.parquet", index=False)

    models_p = _fit_models(
        train, config, fold_number=80_001, feature_columns=FEATURES_P
    )
    predictions_p = predict_action_rows(
        models_p, validation, feature_columns=FEATURES_P
    )
    validation_runs_p = replay_first_trigger(
        action_rows=validation,
        action_predictions=predictions_p,
        threshold=selected_p,
        practical_delta=delta,
        default_algorithm=str(bundle["default_algorithm"]),
    )
    validation_runs_p.to_parquet(
        heavy / "validation_runs_model_p.parquet", index=False
    )

    rows = []
    for name, oof, runs, val_runs, metrics_val in (
        (
            "U_behavior_only",
            model_u_oof,
            None,
            None,
            None,
        ),
        (
            "P_behavior_prefix",
            oof_p,
            runs_p,
            validation_runs_p,
            None,
        ),
    ):
        row = {"model": name}
        if name == "U_behavior_only":
            import json

            phase1 = json.loads(
                (
                    common.RESULTS
                    / "model/behavior_action_gain/training_summary.json"
                ).read_text()
            )
            row.update(
                {
                    "evaluation_split": "bbob_train_oof",
                    **{
                        k: v
                        for k, v in phase1["action_metrics"].items()
                    },
                    **{
                        k: v
                        for k, v in phase1["first_trigger_metrics"].items()
                    },
                }
            )
        else:
            row.update(
                {
                    "evaluation_split": "bbob_train_oof",
                    **_action_metrics(oof_p),
                    **policy_block(runs_p),
                    "harmful_switch_rate": harmful_switch_rate(runs_p, delta),
                }
            )
        rows.append(row)
    rows.append(
        {
            "model": "P_behavior_prefix",
            "evaluation_split": "bbob_validation",
            **_action_metrics(predictions_p),
            **policy_block(validation_runs_p),
            "harmful_switch_rate": harmful_switch_rate(validation_runs_p, delta),
        }
    )
    import json

    phase1 = json.loads(
        (
            common.RESULTS / "model/behavior_action_gain/training_summary.json"
        ).read_text()
    )
    rows.append(
        {
            "model": "U_behavior_only",
            "evaluation_split": "bbob_validation",
            "improve_average_precision": phase1["validation"].get(
                "improve_average_precision", float("nan")
            ),
            "function_balanced_mean_gain": phase1["validation"][
                "function_balanced_mean_gain"
            ],
            "function_balanced_mean_normalized_regret": phase1["validation"][
                "mean_normalized_one_switch_regret"
            ],
        }
    )
    save_table(pd.DataFrame([r for r in rows if r]), "model_summary.csv", TASK)

    # prefix x candidate diagnostic table (Model P OOF)
    diag_rows = []
    for (prefix, candidate), group in oof_p.groupby(
        ["prefix_algorithm", "candidate_action"], sort=True
    ):
        binary = group["action_gain_class"].astype(str).eq("improve").astype(int)
        finite = np.isfinite(group["predicted_improve_probability"])
        rho = float(
            np.corrcoef(
                group["predicted_improve_probability"][finite],
                group["action_gain_vs_continue"][finite],
            )[0, 1]
        )
        diag_rows.append(
            {
                "prefix_algorithm": str(prefix),
                "candidate_action": str(candidate),
                "rows": int(len(group)),
                "improve_prevalence": float(binary.mean()),
                "improve_ap": (
                    float(average_precision_score(binary, group["predicted_improve_probability"]))
                    if binary.nunique() > 1
                    else float("nan")
                ),
                "pearson_pred_vs_true_gain": rho,
                "mean_true_gain": float(group["action_gain_vs_continue"].mean()),
            }
        )
    diag = pd.DataFrame(diag_rows)
    save_table(diag, "prefix_candidate_diagnostic_p.csv", TASK)

    # same table for Model U (existing OOF) for direct comparison
    diag_rows_u = []
    for (prefix, candidate), group in model_u_oof.groupby(
        ["prefix_algorithm", "candidate_action"], sort=True
    ):
        binary = group["action_gain_class"].astype(str).eq("improve").astype(int)
        diag_rows_u.append(
            {
                "prefix_algorithm": str(prefix),
                "candidate_action": str(candidate),
                "rows": int(len(group)),
                "improve_prevalence": float(binary.mean()),
                "improve_ap": (
                    float(average_precision_score(binary, group["predicted_improve_probability"]))
                    if binary.nunique() > 1
                    else float("nan")
                ),
                "mean_true_gain": float(group["action_gain_vs_continue"].mean()),
            }
        )
    save_table(pd.DataFrame(diag_rows_u), "prefix_candidate_diagnostic_u.csv", TASK)

    # FE-phase stratification for CMA-ES candidate
    oof_all = oof_p.copy()
    oof_all["fe_ratio_bin"] = pd.cut(
        oof_all["FE"].to_numpy(dtype=float) / oof_all["FE_total"].to_numpy(dtype=float),
        bins=[0.0, 1 / 3, 2 / 3, 1.0],
        labels=["early", "mid", "late"],
    )
    cmaes = oof_all.loc[oof_all["candidate_action"].astype(str).eq("cmaes")]
    phase_rows = []
    for (phase, prefix), group in cmaes.groupby(["fe_ratio_bin", "prefix_algorithm"], observed=True, sort=True):
        binary = group["action_gain_class"].astype(str).eq("improve").astype(int)
        phase_rows.append(
            {
                "phase": str(phase),
                "prefix_algorithm": str(prefix),
                "rows": int(len(group)),
                "improve_prevalence": float(binary.mean()),
                "improve_ap": (
                    float(average_precision_score(binary, group["predicted_improve_probability"]))
                    if binary.nunique() > 1
                    else float("nan")
                ),
                "mean_true_gain": float(group["action_gain_vs_continue"].mean()),
            }
        )
    save_table(pd.DataFrame(phase_rows), "cmaes_phase_diagnostic.csv", TASK)
    print(f"[{TASK}] done", flush=True)


if __name__ == "__main__":
    main()
