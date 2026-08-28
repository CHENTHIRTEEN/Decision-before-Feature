from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS
from behavior_with_ela.baselines import (
    _action_loss_family_oof_predictions,
    _fit_action_loss_model,
    _predict_action_loss_rows,
)
from behavior_with_ela.local_landscape import (
    LOCAL_LANDSCAPE_POINT_COLUMNS,
    LOCAL_LANDSCAPE_UNCERTAINTY_COLUMNS,
)
from behavior_with_ela.model import (
    apply_practical_gain_delta,
    fit_first_trigger_threshold,
    read_action_datasets,
    replay_first_trigger,
)
from behavior_with_ela.phase2 import _paired_increment, _run_metrics
from behavior_with_ela.protocol import load_experiment_config


PHASE2_ACTION_LOSS_PROTOCOL = "per_state_rf_local_landscape_increment_action_loss_v1"
PHASE2_ACTION_LOSS_FEATURE_GROUPS = {
    "A1_behavior": tuple(SELECTOR_BEHAVIOR_FEATURE_COLUMNS),
    "A3_behavior_local_landscape": (
        *SELECTOR_BEHAVIOR_FEATURE_COLUMNS,
        *LOCAL_LANDSCAPE_POINT_COLUMNS,
    ),
    "A4_behavior_local_landscape_uncertainty": (
        *SELECTOR_BEHAVIOR_FEATURE_COLUMNS,
        *LOCAL_LANDSCAPE_POINT_COLUMNS,
        *LOCAL_LANDSCAPE_UNCERTAINTY_COLUMNS,
    ),
}


def compare_action_loss_local_landscape_models(
    *,
    train_config_path: str | Path,
    phase1_model_path: str | Path,
    output_dir: str | Path,
    validation_config_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = load_experiment_config(train_config_path)
    phase1_bundle = joblib.load(phase1_model_path)
    practical_delta = float(phase1_bundle["practical_gain_delta"])
    train = apply_practical_gain_delta(
        read_action_datasets(config),
        practical_delta,
    )

    output = Path(output_dir)
    expected = (
        output / "action_loss_models.joblib",
        output / "train_oof_action_predictions.parquet",
        output / "train_first_trigger_runs.parquet",
        output / "thresholds.parquet",
        output / "model_summary.parquet",
        output / "increment_contrasts.parquet",
        output / "phase2_action_loss_summary.json",
    )
    if any(path.exists() for path in expected) and not overwrite:
        raise FileExistsError(
            f"action-loss Phase 2 outputs already exist; pass --overwrite: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in expected:
            path.unlink(missing_ok=True)
        (output / "validation_first_trigger_runs.parquet").unlink(missing_ok=True)

    all_predictions: list[pd.DataFrame] = []
    all_runs: list[pd.DataFrame] = []
    all_thresholds: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    full_models: dict[str, Any] = {}
    selected_thresholds: dict[str, float] = {}
    for group_index, (group_name, feature_columns) in enumerate(
        PHASE2_ACTION_LOSS_FEATURE_GROUPS.items(),
        start=1,
    ):
        oof = _action_loss_family_oof_predictions(
            train,
            config,
            feature_columns=tuple(feature_columns),
        )
        thresholds, selected_threshold, runs = fit_first_trigger_threshold(
            action_rows=train,
            action_predictions=oof,
            practical_delta=practical_delta,
        )
        oof["phase2_feature_group"] = group_name
        runs["phase2_feature_group"] = group_name
        thresholds["phase2_feature_group"] = group_name
        all_predictions.append(oof)
        all_runs.append(runs)
        all_thresholds.append(thresholds)
        full_models[group_name] = _fit_action_loss_model(
            train,
            config,
            fold_number=40_000 + group_index,
            feature_columns=tuple(feature_columns),
        )
        selected_thresholds[group_name] = float(selected_threshold)
        summaries.append(
            {
                "evaluation_split": "bbob_train_oof",
                "phase2_feature_group": group_name,
                "feature_count": len(feature_columns),
                **_run_metrics(runs),
            }
        )

    train_predictions = pd.concat(all_predictions, ignore_index=True)
    train_runs = pd.concat(all_runs, ignore_index=True)
    threshold_table = pd.concat(all_thresholds, ignore_index=True)
    all_contrasts = [
        _paired_increment(
            train_runs,
            split="bbob_train_oof",
            left="A1_behavior",
            right="A3_behavior_local_landscape",
            contrast="A3_minus_A1_local_landscape_given_behavior_action_loss",
        ),
        _paired_increment(
            train_runs,
            split="bbob_train_oof",
            left="A3_behavior_local_landscape",
            right="A4_behavior_local_landscape_uncertainty",
            contrast="A4_minus_A3_uncertainty_given_point_estimates_action_loss",
        ),
    ]

    if validation_config_path is not None:
        validation_config = load_experiment_config(validation_config_path)
        validation = apply_practical_gain_delta(
            read_action_datasets(validation_config),
            practical_delta,
        )
        validation_runs_output: list[pd.DataFrame] = []
        for group_name, feature_columns in PHASE2_ACTION_LOSS_FEATURE_GROUPS.items():
            predictions = _predict_action_loss_rows(
                model=full_models[group_name],
                action_rows=validation,
                practical_delta=practical_delta,
                feature_columns=tuple(feature_columns),
            )
            runs = replay_first_trigger(
                action_rows=validation,
                action_predictions=predictions,
                threshold=selected_thresholds[group_name],
                practical_delta=practical_delta,
                default_algorithm=str(phase1_bundle["default_algorithm"]),
            )
            runs["phase2_feature_group"] = group_name
            validation_runs_output.append(runs)
            summaries.append(
                {
                    "evaluation_split": "bbob_validation",
                    "phase2_feature_group": group_name,
                    "feature_count": len(feature_columns),
                    **_run_metrics(runs),
                }
            )
        validation_runs = pd.concat(validation_runs_output, ignore_index=True)
        all_contrasts.extend(
            [
                _paired_increment(
                    validation_runs,
                    split="bbob_validation",
                    left="A1_behavior",
                    right="A3_behavior_local_landscape",
                    contrast="A3_minus_A1_local_landscape_given_behavior_action_loss",
                ),
                _paired_increment(
                    validation_runs,
                    split="bbob_validation",
                    left="A3_behavior_local_landscape",
                    right="A4_behavior_local_landscape_uncertainty",
                    contrast="A4_minus_A3_uncertainty_given_point_estimates_action_loss",
                ),
            ]
        )
        validation_runs.to_parquet(
            output / "validation_first_trigger_runs.parquet",
            index=False,
        )

    summary_table = pd.DataFrame(summaries)
    contrast_table = pd.concat(all_contrasts, ignore_index=True)
    train_predictions.to_parquet(
        output / "train_oof_action_predictions.parquet",
        index=False,
    )
    train_runs.to_parquet(output / "train_first_trigger_runs.parquet", index=False)
    threshold_table.to_parquet(output / "thresholds.parquet", index=False)
    summary_table.to_parquet(output / "model_summary.parquet", index=False)
    contrast_table.to_parquet(output / "increment_contrasts.parquet", index=False)
    joblib.dump(
        {
            "model_protocol": PHASE2_ACTION_LOSS_PROTOCOL,
            "feature_groups": PHASE2_ACTION_LOSS_FEATURE_GROUPS,
            "models": full_models,
            "thresholds": selected_thresholds,
            "practical_gain_delta": practical_delta,
            "default_algorithm": str(phase1_bundle["default_algorithm"]),
            "portfolio": config.algorithms,
            "dimension": config.dimension,
            "FE_total": config.fe_total,
            "population_size": config.population_size,
            "sampling_protocol": config.sampling_protocol,
            "boundary_handling": config.boundary_handling,
            "local_landscape_config": config.local_landscape,
        },
        output / "action_loss_models.joblib",
    )
    result = {
        "model_protocol": PHASE2_ACTION_LOSS_PROTOCOL,
        "feature_groups": {
            name: list(columns)
            for name, columns in PHASE2_ACTION_LOSS_FEATURE_GROUPS.items()
        },
        "practical_gain_delta": practical_delta,
        "train_state_action_rows": int(len(train)),
        "train_oof_prediction_rows": int(len(train_predictions)),
        "validation_included": validation_config_path is not None,
        "validation_rows_used_for_fit": 0,
        "validation_rows_used_for_threshold_fit": 0,
    }
    with (output / "phase2_action_loss_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Local Landscape increments on the behavior action-loss RF carrier."
        )
    )
    parser.add_argument("--train-config", default="configs/behavior_with_ela_train.yaml")
    parser.add_argument("--validation-config", default=None)
    parser.add_argument(
        "--phase1-model",
        default=(
            "behavior_with_ela/results/model/behavior_action_gain/models.joblib"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "behavior_with_ela/results/model/local_landscape_increment_action_loss"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = compare_action_loss_local_landscape_models(
        train_config_path=args.train_config,
        phase1_model_path=args.phase1_model,
        validation_config_path=args.validation_config,
        output_dir=args.output,
        overwrite=args.overwrite,
    )
    print(
        f"compared {len(summary['feature_groups'])} action-loss feature groups with "
        f"{summary['train_oof_prediction_rows']} OOF switch predictions"
    )


if __name__ == "__main__":
    main()
