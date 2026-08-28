"""Quick CEC2017 transfer check under the per-opportunity switch rule.

Protocol change (user decision, 2026-08-29): every decision opportunity whose
best predicted switch advantage exceeds the threshold triggers a switch
(no one-switch cap). Static continue-current reference is reused from the
first-trigger round (it never switches).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import common  # noqa: F401
from common import json_dumps
from common import V2_HEAVY

sys.path.insert(0, str(common.ROOT))

from behavior_with_ela.online import evaluate_one_switch_online  # noqa: E402

TASK = "task9_quick_cec"
FUNCTIONS = (1, 10, 29)
CEC_CONFIG = common.ROOT / "configs/behavior_with_ela_cec2017.yaml"
OUT = common.RESULTS / "online" / "cec2017_quick3_multiswitch"
BUNDLE = V2_HEAVY / "task9_quick_cec" / "v2_online_bundle.joblib"


def main() -> None:
    for name, model in (
        ("v2_regression", BUNDLE),
        ("action_gain_classifier", common.PHASE1_MODEL),
    ):
        print(f"[{TASK}] per-opportunity online: {name}", flush=True)
        evaluate_one_switch_online(
            config_path=CEC_CONFIG,
            model_path=model,
            output_dir=OUT / name,
            only_functions=FUNCTIONS,
            initial_algorithm="all",
            workers=8,
            switch_rule="per_opportunity",
            overwrite=True,
        )

    frames = []
    for name in ("v2_regression", "action_gain_classifier"):
        frame = pd.read_parquet(OUT / name / "online_policy_outcomes.parquet")
        frame["policy_name"] = name
        frames.append(frame)
    learned = pd.concat(frames, ignore_index=True)
    static = pd.read_parquet(
        common.RESULTS / "online/cec2017_quick3/static_continue.parquet"
    )
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
                "switch_rate": float(group["switch_triggered"].mean()),
                "mean_switch_count": float(group["switch_count"].mean()),
                "p90_switch_count": float(group["switch_count"].quantile(0.9)),
                "max_switch_count": int(group["switch_count"].max()),
                "median_switch_FE": float(
                    pd.Series(
                        [fe for fe in group["selected_FE"].dropna()]
                    ).median()
                ),
                "success_rate": float(group["success"].mean()),
            }
        )
    table = pd.DataFrame(summary_rows)
    common.save_table(table, "multiswitch_summary.csv", TASK)
    function_balanced = table.groupby("policy_name")[
        [
            "mean_log10_gap",
            "mean_relative_gain_vs_continue",
            "switch_rate",
            "mean_switch_count",
            "max_switch_count",
        ]
    ].mean()
    print(function_balanced.round(4).to_string(), flush=True)
    (common.V2 / TASK / "multiswitch_function_balanced.json").write_text(
        json_dumps(function_balanced.to_dict(orient="index"))
    )
    print(f"[{TASK}] done", flush=True)


if __name__ == "__main__":
    main()
