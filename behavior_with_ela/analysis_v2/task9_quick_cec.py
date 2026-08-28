"""Quick CEC2017 transfer sanity check for the frozen v2 regression carrier.

3 functions (F1, F10, F29) x 10 seeds x 3 prefixes, online one-switch policy.
Development diagnostic only - not the formal Task 9 evaluation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import common  # noqa: F401
from common import TRAIN_CONFIG, json_dumps, load_train_val

sys.path.insert(0, str(common.ROOT))

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS as COLS  # noqa: E402
from behavior_with_ela.baselines import _fit_action_loss_model  # noqa: E402
from behavior_with_ela.online import (  # noqa: E402
    REGRESSION_ONLINE_PROTOCOL,
    evaluate_one_switch_online,
)
from behavior_with_ela.online_baselines import run_static_optimizer_policy  # noqa: E402
from behavior_with_ela.protocol import load_experiment_config  # noqa: E402

TASK = "task9_quick_cec"
FUNCTIONS = (1, 10, 29)
CEC_CONFIG = common.ROOT / "configs/behavior_with_ela_cec2017.yaml"
OUT = common.RESULTS / "online" / "cec2017_quick3"


def build_v2_bundle() -> Path:
    config, validation_config, bundle1, delta, train, validation = load_train_val()
    thresholds = pd.read_parquet(
        common.RESULTS / "analysis_v2/task1/thresholds.parquet"
    )
    selected = float(
        thresholds.loc[thresholds["selected_threshold"], "threshold"].iloc[0]
    )
    regressor = _fit_action_loss_model(train, config, fold_number=90_001)
    online_bundle = {
        "model_protocol": REGRESSION_ONLINE_PROTOCOL,
        "feature_columns": tuple(COLS),
        "portfolio": tuple(config.algorithms),
        "models": {"regressor": regressor},
        "decision_threshold": selected,
        "practical_gain_delta": delta,
        "default_algorithm": str(bundle1["default_algorithm"]),
        "dimension": config.dimension,
        "FE_total": config.fe_total,
        "population_size": config.population_size,
        "sampling_protocol": config.sampling_protocol,
        "boundary_handling": config.boundary_handling,
    }
    heavy = common.V2_HEAVY / TASK
    heavy.mkdir(parents=True, exist_ok=True)
    path = heavy / "v2_online_bundle.joblib"
    joblib.dump(online_bundle, path)
    return path


def static_reference(config) -> pd.DataFrame:
    suite = config.suite("cec2017")
    rows = []
    for function in FUNCTIONS:
        for seed in config.seeds:
            for prefix in config.algorithms:
                outcome, _, _ = run_static_optimizer_policy(
                    config=config,
                    suite=suite,
                    function=function,
                    instance=suite.instances[0],
                    seed=seed,
                    prefix_algorithm=prefix,
                    default_algorithm="cmaes",
                    policy_name="continue_current",
                )
                rows.append(outcome)
    return pd.DataFrame(rows)


def main() -> None:
    config = load_experiment_config(CEC_CONFIG)
    v2_bundle = build_v2_bundle()
    print(f"[{TASK}] v2 online bundle written: {v2_bundle}", flush=True)

    print(f"[{TASK}] online evaluation: v2 regression carrier", flush=True)
    evaluate_one_switch_online(
        config_path=CEC_CONFIG,
        model_path=v2_bundle,
        output_dir=OUT / "v2_regression",
        only_functions=FUNCTIONS,
        initial_algorithm="all",
        workers=8,
        overwrite=True,
    )
    print(f"[{TASK}] online evaluation: three-class classifier reference", flush=True)
    evaluate_one_switch_online(
        config_path=CEC_CONFIG,
        model_path=common.PHASE1_MODEL,
        output_dir=OUT / "action_gain_classifier",
        only_functions=FUNCTIONS,
        initial_algorithm="all",
        workers=8,
        overwrite=True,
    )

    print(f"[{TASK}] static continue-current reference", flush=True)
    static = static_reference(config)
    static["policy_name"] = "continue_current"
    static.to_parquet(OUT / "static_continue.parquet", index=False)

    frames = []
    for name in ("v2_regression", "action_gain_classifier"):
        frame = pd.read_parquet(OUT / name / "online_policy_outcomes.parquet")
        frame["policy_name"] = name
        frames.append(frame)
    learned = pd.concat(frames, ignore_index=True)
    key = ["function_id", "seed", "prefix_algorithm"]
    cont = static[key + ["log10_gap"]].rename(columns={"log10_gap": "continue_gap"})
    merged = learned.merge(cont, on=key, how="left")
    merged["relative_gain_vs_continue"] = (
        merged["continue_gap"] - merged["log10_gap"]
    )
    static_mean = static.groupby("function_id")["log10_gap"].mean()

    summary_rows = []
    for (policy, function), group in merged.groupby(["policy_name", "function_id"]):
        summary_rows.append(
            {
                "policy_name": policy,
                "function_id": function,
                "runs": int(len(group)),
                "mean_log10_gap": float(group["log10_gap"].mean()),
                "mean_relative_gain_vs_continue": float(
                    group["relative_gain_vs_continue"].mean()
                ),
                "mean_vs_static_function_mean": float(
                    group["log10_gap"].mean() - static_mean.loc[function]
                ),
                "switch_rate": float(group["switch_triggered"].mean()),
                "median_selected_FE": float(
                    group.loc[
                        group["switch_triggered"], "selected_FE"
                    ].median()
                    if group["switch_triggered"].any()
                    else float("nan")
                ),
                "success_rate": float(group["success"].mean()),
            }
        )
    table = pd.DataFrame(summary_rows)
    common.save_table(table, "quick_cec_summary.csv", TASK)
    function_balanced = (
        table.groupby("policy_name")[
            ["mean_log10_gap", "mean_relative_gain_vs_continue", "switch_rate"]
        ]
        .mean()
    )
    print(function_balanced.round(4).to_string(), flush=True)
    (common.V2 / TASK / "function_balanced.json").write_text(
        json_dumps(function_balanced.to_dict(orient="index"))
    )
    print(f"[{TASK}] done", flush=True)


if __name__ == "__main__":
    main()
