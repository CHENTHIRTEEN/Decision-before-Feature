"""Task 7: does selected MA-BBOB augmentation improve behavior->action-value generalization?

Model B  = BBOB train only
Model BM = BBOB + selected MA-BBOB (Phase 1 reference, re-evaluated here)
Plus a MA-BBOB definition-grouped 5-fold OOF diagnostic (score level only).
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
from sklearn.metrics import average_precision_score  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

TASK = "task7"
MA_FOLD_STREAM = 2026082917


def main() -> None:
    config, validation_config, bundle, delta, train, validation = load_train_val()
    heavy = V2_HEAVY / TASK
    heavy.mkdir(parents=True, exist_ok=True)
    rows = []

    # ---- Model B: BBOB only ----
    train_bbob = train.loc[train["suite"].astype(str).eq("bbob")].copy()
    print(f"[{TASK}] Model B (BBOB only): {len(train_bbob)} rows", flush=True)
    oof_b = _family_oof_predictions(train_bbob, config)
    thresholds_b, selected_b, runs_b = fit_first_trigger_threshold(
        action_rows=train_bbob, action_predictions=oof_b, practical_delta=delta
    )
    rows.append(
        {
            "model": "B_bbob_only",
            "evaluation_split": "bbob_train_oof",
            **_action_metrics(oof_b),
            **policy_block(runs_b),
            "harmful_switch_rate": harmful_switch_rate(runs_b, delta),
        }
    )
    models_b = _fit_models(train_bbob, config, fold_number=70_001)
    predictions_b = predict_action_rows(models_b, validation)
    validation_runs_b = replay_first_trigger(
        action_rows=validation,
        action_predictions=predictions_b,
        threshold=selected_b,
        practical_delta=delta,
        default_algorithm=str(bundle["default_algorithm"]),
    )
    rows.append(
        {
            "model": "B_bbob_only",
            "evaluation_split": "bbob_validation",
            **_action_metrics(predictions_b),
            **policy_block(validation_runs_b),
            "harmful_switch_rate": harmful_switch_rate(validation_runs_b, delta),
        }
    )
    oof_b.to_parquet(heavy / "oof_model_b.parquet", index=False)
    validation_runs_b.to_parquet(heavy / "validation_runs_model_b.parquet", index=False)

    # ---- Model BM: BBOB + MA-BBOB (Phase 1 reference numbers, restated) ----
    phase1_summary = (
        common.RESULTS / "model/behavior_action_gain/training_summary.json"
    ).read_text()
    import json

    phase1 = json.loads(phase1_summary)
    rows.append(
        {
            "model": "BM_bbob_mabbob",
            "evaluation_split": "bbob_train_oof",
            "improve_average_precision": phase1["action_metrics"][
                "improve_average_precision"
            ],
            "balanced_accuracy": phase1["action_metrics"]["balanced_accuracy"],
            "macro_f1": phase1["action_metrics"]["macro_f1"],
            "function_balanced_mean_gain": phase1["first_trigger_metrics"][
                "function_balanced_mean_gain"
            ],
            "function_balanced_mean_normalized_regret": phase1[
                "first_trigger_metrics"
            ]["mean_normalized_one_switch_regret"],
        }
    )
    rows.append(
        {
            "model": "BM_bbob_mabbob",
            "evaluation_split": "bbob_validation",
            "function_balanced_mean_gain": phase1["validation"][
                "function_balanced_mean_gain"
            ],
            "function_balanced_mean_normalized_regret": phase1["validation"][
                "mean_normalized_one_switch_regret"
            ],
        }
    )

    # ---- MA-BBOB definition-grouped 5-fold OOF (score level) ----
    ma = train.loc[train["suite"].astype(str).eq("mabbob")].copy()
    definitions = sorted(ma["function_id"].astype(str).unique())
    order = np.random.default_rng(
        np.random.SeedSequence([MA_FOLD_STREAM, config.dimension]).generate_state(1)
    ).permutation(len(definitions))
    fold_of = {
        definition: int(position % 5)
        for position, definition in zip(order, definitions)
    }
    ma["definition_fold"] = ma["function_id"].astype(str).map(fold_of)
    diag_rows = []
    for fold in range(5):
        held = ma.loc[ma["definition_fold"].eq(fold)]
        fit_rows = pd.concat(
            [train.loc[train["suite"].astype(str).eq("bbob")], ma.loc[~ma["definition_fold"].eq(fold)]],
            ignore_index=True,
        )
        models = _fit_models(fit_rows, config, fold_number=70_100 + fold)
        predictions = predict_action_rows(models, held)
        per_def = []
        for definition, group in predictions.groupby("function_id", sort=True):
            binary = group["action_gain_class"].astype(str).eq("improve").astype(int)
            rho = float(
                spearmanr(
                    group["predicted_improve_probability"].to_numpy(dtype=float),
                    group["action_gain_vs_continue"].to_numpy(dtype=float),
                ).statistic
            )
            per_def.append(
                {
                    "model": "BM_definition_oof",
                    "fold": fold,
                    "function_id": str(definition),
                    "rows": int(len(group)),
                    "improve_prevalence": float(binary.mean()),
                    "improve_ap": (
                        float(average_precision_score(binary, group["predicted_improve_probability"]))
                        if binary.nunique() > 1
                        else float("nan")
                    ),
                    "gain_spearman": rho,
                }
            )
        diag_rows.extend(per_def)
        frame = pd.DataFrame(per_def)
        print(
            f"[{TASK}] MA definition fold {fold}: AP={frame['improve_ap'].mean():.4f} "
            f"rho={frame['gain_spearman'].mean():.4f}",
            flush=True,
        )
    save_table(pd.DataFrame(diag_rows), "mabbob_definition_oof.csv", TASK)

    summary = pd.DataFrame(rows)
    save_table(summary, "augmentation_summary.parquet", TASK)
    (common.V2 / TASK / "definition_fold_assignment.json").write_text(
        json_dumps(fold_of)
    )
    print(f"[{TASK}] done", flush=True)


if __name__ == "__main__":
    main()
