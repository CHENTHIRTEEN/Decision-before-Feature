"""Task 4: Local Landscape group ablation (code-defined L1-L4 groups).

Feature groups are the pre-defined column tuples in local_landscape.py:
L1 streaming fitness distribution, L2 meta-model, L3 information content, L4 geometry.
"""
from __future__ import annotations

import sys

import pandas as pd

import common  # noqa: F401
from common import (
    TRAIN_CONFIG,
    V2_HEAVY,
    add_prefix_onehot,
    harmful_switch_rate,
    json_dumps,
    load_train_val,
    policy_block,
    save_table,
    switch_fe_quantiles,
)

sys.path.insert(0, str(common.ROOT))

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS  # noqa: E402
from behavior_with_ela.local_landscape import (  # noqa: E402
    LOCAL_LANDSCAPE_GEOMETRY_COLUMNS,
    LOCAL_LANDSCAPE_INFORMATION_COLUMNS,
    LOCAL_LANDSCAPE_META_MODEL_COLUMNS,
    LOCAL_LANDSCAPE_STREAMING_COLUMNS,
)
from behavior_with_ela.model import (  # noqa: E402
    RUN_KEY,
    _family_oof_predictions,
    _fit_models,
    fit_first_trigger_threshold,
    predict_action_rows,
    replay_first_trigger,
)
from behavior_with_ela.phase2 import _paired_increment  # noqa: E402

TASK = "task4"
GROUPS = {
    "L1_fitness_distribution": tuple(LOCAL_LANDSCAPE_STREAMING_COLUMNS),
    "L2_meta_model": tuple(LOCAL_LANDSCAPE_META_MODEL_COLUMNS),
    "L3_information_content": tuple(LOCAL_LANDSCAPE_INFORMATION_COLUMNS),
    "L4_geometry_nbc": tuple(LOCAL_LANDSCAPE_GEOMETRY_COLUMNS),
}


def main() -> None:
    config, validation_config, bundle, delta, train, validation = load_train_val()
    train = add_prefix_onehot(train)
    validation = add_prefix_onehot(validation)
    heavy = V2_HEAVY / TASK
    heavy.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_runs = []
    for index, (group_name, columns) in enumerate(GROUPS.items(), start=1):
        variants = {
            f"M2_{group_name}": tuple(columns),
            f"M1_{group_name}": tuple(SELECTOR_BEHAVIOR_FEATURE_COLUMNS)
            + tuple(columns),
        }
        for variant, feature_columns in variants.items():
            print(f"[{TASK}] {variant}: {len(feature_columns)} features", flush=True)
            oof = _family_oof_predictions(
                train, config, feature_columns=feature_columns
            )
            thresholds, selected, runs = fit_first_trigger_threshold(
                action_rows=train,
                action_predictions=oof,
                practical_delta=delta,
            )
            oof["feature_variant"] = variant
            runs["feature_variant"] = variant
            thresholds["feature_variant"] = variant
            oof.to_parquet(heavy / f"oof_{variant}.parquet", index=False)
            runs.to_parquet(heavy / f"runs_{variant}.parquet", index=False)
            models = _fit_models(
                train,
                config,
                fold_number=60_000 + index,
                feature_columns=feature_columns,
            )
            predictions = predict_action_rows(
                models, validation, feature_columns=feature_columns
            )
            validation_runs = replay_first_trigger(
                action_rows=validation,
                action_predictions=predictions,
                threshold=selected,
                practical_delta=delta,
                default_algorithm=str(bundle["default_algorithm"]),
            )
            validation_runs["feature_variant"] = variant
            validation_runs.to_parquet(
                heavy / f"validation_runs_{variant}.parquet", index=False
            )
            summaries.append(
                {
                    "evaluation_split": "bbob_train_oof",
                    "feature_variant": variant,
                    "feature_count": len(feature_columns),
                    **policy_block(runs),
                    "harmful_switch_rate": harmful_switch_rate(runs, delta),
                }
            )
            summaries.append(
                {
                    "evaluation_split": "bbob_validation",
                    "feature_variant": variant,
                    "feature_count": len(feature_columns),
                    **policy_block(validation_runs),
                    "harmful_switch_rate": harmful_switch_rate(
                        validation_runs, delta
                    ),
                    **switch_fe_quantiles(validation_runs),
                }
            )
            all_runs.append(runs)

    summary_table = pd.DataFrame(summaries)
    save_table(summary_table, "model_summary.parquet", TASK)

    m1_runs = pd.read_parquet(
        common.RESULTS
        / "model/local_landscape_increment/train_first_trigger_runs.parquet"
    )
    m1_runs = m1_runs.loc[
        m1_runs["phase2_feature_group"].eq("M1_behavior")
    ].copy()
    m1_runs["feature_variant"] = "M1_behavior_reference"
    contrast_rows = []
    for group_name in GROUPS:
        variant = f"M1_{group_name}"
        right = pd.concat(
            [r for r in all_runs if r["feature_variant"].eq(variant)],
            ignore_index=True,
        )
        combined = pd.concat([m1_runs, right], ignore_index=True)
        contrast = _paired_increment(
            combined,
            split="bbob_train_oof",
            left="M1_behavior_reference",
            right=variant,
            contrast=f"{variant}_minus_M1_behavior",
        )
        contrast_rows.append(contrast)
    contrast_table = pd.concat(contrast_rows, ignore_index=True)
    save_table(contrast_table, "paired_contrasts_vs_M1.parquet", TASK)
    (common.V2 / TASK / "summary.json").write_text(
        json_dumps({"groups": {k: list(v) for k, v in GROUPS.items()}})
    )
    print(f"[{TASK}] done", flush=True)


if __name__ == "__main__":
    main()
