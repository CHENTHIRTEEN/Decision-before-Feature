from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS
from behavior_with_ela.local_landscape import (
    LOCAL_LANDSCAPE_POINT_COLUMNS,
    LOCAL_LANDSCAPE_UNCERTAINTY_COLUMNS,
)
from behavior_with_ela.model import (
    RUN_KEY,
    _action_metrics,
    _family_oof_predictions,
    _fit_models,
    apply_practical_gain_delta,
    fit_first_trigger_threshold,
    predict_action_rows,
    read_action_datasets,
    replay_first_trigger,
)
from behavior_with_ela.protocol import load_experiment_config


PHASE2_MODEL_PROTOCOL = "per_action_rf_local_landscape_increment_v1"
PHASE2_FEATURE_GROUPS = {
    "M0_time_only": ("bf_fe_ratio",),
    "M1_behavior": tuple(SELECTOR_BEHAVIOR_FEATURE_COLUMNS),
    "M2_local_landscape": tuple(LOCAL_LANDSCAPE_POINT_COLUMNS),
    "M3_behavior_local_landscape": (
        *SELECTOR_BEHAVIOR_FEATURE_COLUMNS,
        *LOCAL_LANDSCAPE_POINT_COLUMNS,
    ),
    "M4_behavior_local_landscape_uncertainty": (
        *SELECTOR_BEHAVIOR_FEATURE_COLUMNS,
        *LOCAL_LANDSCAPE_POINT_COLUMNS,
        *LOCAL_LANDSCAPE_UNCERTAINTY_COLUMNS,
    ),
}


def compare_local_landscape_models(
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
    _validate_phase2_features(train)

    output = Path(output_dir)
    expected = (
        output / "phase2_models.joblib",
        output / "train_oof_action_predictions.parquet",
        output / "train_first_trigger_runs.parquet",
        output / "thresholds.parquet",
        output / "model_summary.parquet",
        output / "increment_contrasts.parquet",
        output / "phase2_summary.json",
    )
    if any(path.exists() for path in expected) and not overwrite:
        raise FileExistsError(f"Phase 2 outputs already exist; pass --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in expected:
            path.unlink(missing_ok=True)
        (output / "validation_action_predictions.parquet").unlink(missing_ok=True)
        (output / "validation_first_trigger_runs.parquet").unlink(missing_ok=True)

    all_predictions: list[pd.DataFrame] = []
    all_runs: list[pd.DataFrame] = []
    all_thresholds: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    full_models: dict[str, Any] = {}
    selected_thresholds: dict[str, float] = {}
    for group_index, (group_name, feature_columns) in enumerate(
        PHASE2_FEATURE_GROUPS.items(),
        start=1,
    ):
        oof = _family_oof_predictions(
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
        full_models[group_name] = _fit_models(
            train,
            config,
            fold_number=30_000 + group_index,
            feature_columns=tuple(feature_columns),
        )
        selected_thresholds[group_name] = float(selected_threshold)
        metrics = _action_metrics(oof)
        summaries.append(
            {
                "evaluation_split": "bbob_train_oof",
                "phase2_feature_group": group_name,
                "feature_count": len(feature_columns),
                **metrics,
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
            left="M1_behavior",
            right="M3_behavior_local_landscape",
            contrast="M3_minus_M1_local_landscape_given_behavior",
        ),
        _paired_increment(
            train_runs,
            split="bbob_train_oof",
            left="M3_behavior_local_landscape",
            right="M4_behavior_local_landscape_uncertainty",
            contrast="M4_minus_M3_uncertainty_given_point_estimates",
        ),
    ]

    validation_predictions_output: list[pd.DataFrame] = []
    validation_runs_output: list[pd.DataFrame] = []
    if validation_config_path is not None:
        validation_config = load_experiment_config(validation_config_path)
        validation = apply_practical_gain_delta(
            read_action_datasets(validation_config),
            practical_delta,
        )
        _validate_phase2_features(validation)
        for group_name, feature_columns in PHASE2_FEATURE_GROUPS.items():
            predictions = predict_action_rows(
                full_models[group_name],
                validation,
                feature_columns=tuple(feature_columns),
            )
            runs = replay_first_trigger(
                action_rows=validation,
                action_predictions=predictions,
                threshold=selected_thresholds[group_name],
                practical_delta=practical_delta,
                default_algorithm=str(phase1_bundle["default_algorithm"]),
            )
            predictions["phase2_feature_group"] = group_name
            runs["phase2_feature_group"] = group_name
            validation_predictions_output.append(predictions)
            validation_runs_output.append(runs)
            summaries.append(
                {
                    "evaluation_split": "bbob_validation",
                    "phase2_feature_group": group_name,
                    "feature_count": len(feature_columns),
                    **_action_metrics(predictions),
                    **_run_metrics(runs),
                }
            )
        validation_predictions = pd.concat(
            validation_predictions_output,
            ignore_index=True,
        )
        validation_runs = pd.concat(validation_runs_output, ignore_index=True)
        all_contrasts.extend(
            [
                _paired_increment(
                    validation_runs,
                    split="bbob_validation",
                    left="M1_behavior",
                    right="M3_behavior_local_landscape",
                    contrast="M3_minus_M1_local_landscape_given_behavior",
                ),
                _paired_increment(
                    validation_runs,
                    split="bbob_validation",
                    left="M3_behavior_local_landscape",
                    right="M4_behavior_local_landscape_uncertainty",
                    contrast="M4_minus_M3_uncertainty_given_point_estimates",
                ),
            ]
        )
        validation_predictions.to_parquet(
            output / "validation_action_predictions.parquet",
            index=False,
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
            "model_protocol": PHASE2_MODEL_PROTOCOL,
            "feature_groups": PHASE2_FEATURE_GROUPS,
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
        output / "phase2_models.joblib",
    )
    result = {
        "model_protocol": PHASE2_MODEL_PROTOCOL,
        "feature_groups": {
            name: list(columns) for name, columns in PHASE2_FEATURE_GROUPS.items()
        },
        "practical_gain_delta": practical_delta,
        "train_state_action_rows": int(len(train)),
        "train_oof_prediction_rows": int(len(train_predictions)),
        "validation_included": validation_config_path is not None,
        "validation_rows_used_for_fit": 0,
        "validation_rows_used_for_threshold_fit": 0,
    }
    with (output / "phase2_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def _validate_phase2_features(rows: pd.DataFrame) -> None:
    required = {
        *SELECTOR_BEHAVIOR_FEATURE_COLUMNS,
        *LOCAL_LANDSCAPE_POINT_COLUMNS,
        *LOCAL_LANDSCAPE_UNCERTAINTY_COLUMNS,
        "local_landscape_extra_FE",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Phase 2 action data is missing features: {missing}")
    if not rows["local_landscape_extra_FE"].astype(int).eq(0).all():
        raise ValueError("Phase 2 local landscape features must use zero extra FE")
    for group_name, columns in PHASE2_FEATURE_GROUPS.items():
        if len(columns) != len(set(columns)):
            raise RuntimeError(f"Phase 2 feature group contains duplicate columns: {group_name}")


def _run_metrics(runs: pd.DataFrame) -> dict[str, float | int]:
    function_means = runs.groupby("cv_group_id", sort=False)[
        [
            "selected_action_gain",
            "selected_terminal_log10_loss",
            "one_switch_regret",
            "normalized_one_switch_regret",
        ]
    ].mean()
    return {
        "run_count": int(len(runs)),
        "function_balanced_mean_gain": float(
            function_means["selected_action_gain"].mean()
        ),
        "function_balanced_mean_log10_loss": float(
            function_means["selected_terminal_log10_loss"].mean()
        ),
        "function_balanced_mean_regret": float(
            function_means["one_switch_regret"].mean()
        ),
        "function_balanced_mean_normalized_regret": float(
            function_means["normalized_one_switch_regret"].mean()
        ),
        "median_gain": float(runs["selected_action_gain"].median()),
        "median_regret": float(runs["one_switch_regret"].median()),
        "median_normalized_regret": float(
            runs["normalized_one_switch_regret"].median()
        ),
        "switch_rate": float(runs["switch_triggered"].mean()),
        "acceptable_policy_rate": float(runs["acceptable_policy"].mean()),
    }


def _paired_increment(
    runs: pd.DataFrame,
    *,
    split: str,
    left: str,
    right: str,
    contrast: str,
) -> pd.DataFrame:
    keys = list(RUN_KEY)
    columns = [
        *keys,
        "cv_group_id",
        "selected_action_gain",
        "selected_terminal_log10_loss",
        "one_switch_regret",
    ]
    left_rows = runs.loc[runs["phase2_feature_group"].astype(str).eq(left), columns]
    right_rows = runs.loc[runs["phase2_feature_group"].astype(str).eq(right), columns]
    paired = left_rows.merge(
        right_rows,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_left", "_right"),
    )
    if len(paired) != len(left_rows) or len(paired) != len(right_rows):
        raise RuntimeError(f"Phase 2 paired contrast has incomplete run coverage: {contrast}")
    if not paired["cv_group_id_left"].astype(str).equals(
        paired["cv_group_id_right"].astype(str)
    ):
        raise RuntimeError("paired Phase 2 runs disagree on function group")
    paired["gain_difference"] = (
        paired["selected_action_gain_right"]
        - paired["selected_action_gain_left"]
    )
    paired["log10_loss_difference"] = (
        paired["selected_terminal_log10_loss_right"]
        - paired["selected_terminal_log10_loss_left"]
    )
    paired["regret_difference"] = (
        paired["one_switch_regret_right"] - paired["one_switch_regret_left"]
    )
    function_effects = (
        paired.groupby("cv_group_id_left", as_index=False)[
            ["gain_difference", "log10_loss_difference", "regret_difference"]
        ]
        .mean()
        .rename(columns={"cv_group_id_left": "cv_group_id"})
    )
    function_effects.insert(0, "evaluation_split", split)
    function_effects.insert(1, "contrast", contrast)
    function_effects["left_feature_group"] = left
    function_effects["right_feature_group"] = right
    function_effects["function_balanced_gain_difference"] = float(
        function_effects["gain_difference"].mean()
    )
    function_effects["function_balanced_log10_loss_difference"] = float(
        function_effects["log10_loss_difference"].mean()
    )
    function_effects["function_balanced_regret_difference"] = float(
        function_effects["regret_difference"].mean()
    )
    return function_effects


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare time, Behavior, and trajectory-derived Local Landscape models."
    )
    parser.add_argument("--train-config", default="configs/behavior_with_ela_train.yaml")
    parser.add_argument("--validation-config", default=None)
    parser.add_argument(
        "--phase1-model",
        default="results/behavior_with_ela/model/behavior_action_gain/models.joblib",
    )
    parser.add_argument(
        "--output",
        default="results/behavior_with_ela/model/local_landscape_increment",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = compare_local_landscape_models(
        train_config_path=args.train_config,
        phase1_model_path=args.phase1_model,
        validation_config_path=args.validation_config,
        output_dir=args.output,
        overwrite=args.overwrite,
    )
    print(
        f"compared {len(summary['feature_groups'])} Phase 2 feature groups with "
        f"{summary['train_oof_prediction_rows']} OOF switch predictions"
    )


if __name__ == "__main__":
    main()
