from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from decision.model_protocol import (
    ACTIVE_MODEL_NAMES,
    FROZEN_THRESHOLD_MODE,
    FULL_TRAIN_OOF_FOLDS,
    INNER_OOF_FOLDS,
    MODEL_SELECTION_METRIC,
    OUTER_OOF_FOLDS,
    THRESHOLD_NEIGHBORHOOD_QUANTILE,
    active_model_specs,
)


BANNED_DECISION_MODEL_FRAGMENTS = (
    "random_forest",
    "xgboost",
    "lightgbm",
    "mlp",
)


def check_model_protocol(training_dir: Path | None = None) -> dict[str, Any]:
    specs = active_model_specs(1701)
    observed_names = tuple(spec.model_name for spec in specs)
    if observed_names != ACTIVE_MODEL_NAMES:
        raise ValueError(f"active Decision candidates changed: {observed_names}")
    objectives = {spec.model_name: spec.objective for spec in specs}
    expected_objectives = {
        "lda_classifier": "classification",
        "logistic_regression_classifier": "classification",
        "ridge_regression": "regression",
    }
    if objectives != expected_objectives:
        raise ValueError(f"Decision objective mapping changed: {objectives}")
    if [spec.model_name for spec in specs if spec.supports_utility_rmse] != ["ridge_regression"]:
        raise ValueError("continuous Utility RMSE must be restricted to Ridge")
    banned = [
        spec.model_name
        for spec in specs
        if any(fragment in spec.model_name.lower() for fragment in BANNED_DECISION_MODEL_FRAGMENTS)
    ]
    if banned:
        raise ValueError(f"complex models returned to the active Decision candidate set: {banned}")

    result: dict[str, Any] = {
        "status": "ok",
        "active_model_names": list(ACTIVE_MODEL_NAMES),
        "objectives": objectives,
        "model_selection_metric": MODEL_SELECTION_METRIC,
        "frozen_threshold_mode": FROZEN_THRESHOLD_MODE,
        "threshold_neighborhood_quantile": THRESHOLD_NEIGHBORHOOD_QUANTILE,
        "oof_folds": {
            "outer": OUTER_OOF_FOLDS,
            "inner": INNER_OOF_FOLDS,
            "full_train_threshold": FULL_TRAIN_OOF_FOLDS,
        },
        "artifact_checks_run": training_dir is not None,
    }
    if training_dir is not None:
        result["artifacts"] = _check_training_artifacts(training_dir)
    return result


def _check_training_artifacts(training_dir: Path) -> dict[str, Any]:
    summary_path = training_dir / "full_decision_model_training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if tuple(summary.get("models_trained", [])) != ACTIVE_MODEL_NAMES:
        raise ValueError("training summary model candidates do not match the active protocol")
    if summary.get("model_selection_metric") != MODEL_SELECTION_METRIC:
        raise ValueError("training summary uses the wrong model-selection metric")
    selected_name = str(summary.get("selected_model_name", ""))
    if selected_name not in ACTIVE_MODEL_NAMES:
        raise ValueError("training summary does not identify one active selected model")
    if summary.get("threshold_modes") != ["zero", FROZEN_THRESHOLD_MODE]:
        raise ValueError("training summary threshold modes do not match the frozen protocol")

    selection = pq.read_table(training_dir / "model_selection_summary.parquet").to_pandas()
    if tuple(selection.sort_values("candidate_order")["model_name"].astype(str)) != ACTIVE_MODEL_NAMES:
        raise ValueError("model-selection table candidates do not match the frozen order")
    if int(selection["selected_model"].astype(bool).sum()) != 1:
        raise ValueError("model-selection table must select exactly one candidate")
    selected_table_name = str(selection.loc[selection["selected_model"].astype(bool), "model_name"].iloc[0])
    if selected_table_name != selected_name:
        raise ValueError("selected model differs between summary and model-selection table")
    if (selection["validation_rows_used_for_model_or_threshold_selection"].astype(int) != 0).any():
        raise ValueError("BBOB-validation rows were used for model or threshold selection")

    thresholds = pq.read_table(training_dir / "decision_thresholds.parquet").to_pandas()
    expected_pairs = {(name, mode) for name in ACTIVE_MODEL_NAMES for mode in ("zero", FROZEN_THRESHOLD_MODE)}
    observed_pairs = set(zip(thresholds["model_name"].astype(str), thresholds["threshold_mode"].astype(str)))
    if observed_pairs != expected_pairs or len(thresholds) != len(expected_pairs):
        raise ValueError("decision-threshold table does not contain exactly two modes per active model")
    if (thresholds["validation_rows_used_for_threshold_fit"].astype(int) != 0).any():
        raise ValueError("BBOB-validation rows were used to fit a Decision threshold")
    if (thresholds["in_sample_train_rows_used_for_threshold_fit"].astype(int) != 0).any():
        raise ValueError("in-sample training predictions were used to fit a Decision threshold")
    oof_thresholds = thresholds[
        thresholds["threshold_mode"].astype(str) == FROZEN_THRESHOLD_MODE
    ]
    if len(oof_thresholds) != len(ACTIVE_MODEL_NAMES):
        raise ValueError("one OOF threshold-neighborhood width is required per active model")
    if not np.allclose(
        oof_thresholds["threshold_neighborhood_quantile"].astype(float).to_numpy(),
        THRESHOLD_NEIGHBORHOOD_QUANTILE,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("threshold-neighborhood quantile does not match the frozen Q10 protocol")
    widths = oof_thresholds["threshold_neighborhood_width"].astype(float).to_numpy()
    if not np.isfinite(widths).all() or (widths < 0.0).any():
        raise ValueError("threshold-neighborhood widths must be finite and non-negative")
    if (oof_thresholds["validation_rows_used_for_neighborhood_fit"].astype(int) != 0).any():
        raise ValueError("BBOB-validation rows were used to fit a threshold-neighborhood width")

    folds = pq.read_table(training_dir / "oof_fold_summary.parquet").to_pandas()
    if (folds["family_overlap_count"].astype(int) != 0).any():
        raise ValueError("landscape families overlap within at least one OOF fold")
    if (folds["validation_rows_used"].astype(int) != 0).any():
        raise ValueError("BBOB-validation rows entered an OOF fold")
    expected_roles = {"nested_inner_threshold", "nested_outer_evaluation", "full_train_oof_threshold"}
    if set(folds["fold_role"].astype(str)) != expected_roles:
        raise ValueError("OOF fold table does not cover the three frozen fold roles")

    scores = pq.read_table(training_dir / "validation_score_summary.parquet").to_pandas()
    all_validation = scores[scores["layer"].astype(str) == "all_validation"]
    classifier = all_validation[all_validation["objective"].astype(str) == "classification"]
    ridge = all_validation[all_validation["model_name"].astype(str) == "ridge_regression"]
    if classifier["rmse_applicable"].astype(bool).any() or classifier["rmse"].notna().any():
        raise ValueError("classification scores must not be assigned continuous Utility RMSE")
    if len(ridge) != 1 or not bool(ridge["rmse_applicable"].iloc[0]) or not np.isfinite(float(ridge["rmse"].iloc[0])):
        raise ValueError("Ridge must report one finite all-validation continuous Utility RMSE")

    predictions = pq.read_table(
        training_dir / "validation_predictions.parquet",
        columns=["model_name", "selected_by_nested_oof"],
    ).to_pandas()
    selected_prediction_names = predictions.loc[
        predictions["selected_by_nested_oof"].astype(bool), "model_name"
    ].astype(str).unique().tolist()
    if selected_prediction_names != [selected_name]:
        raise ValueError("validation predictions do not identify the selected model consistently")
    return {
        "training_dir": str(training_dir),
        "selected_model_name": selected_name,
        "selection_rows": int(len(selection)),
        "threshold_rows": int(len(thresholds)),
        "oof_fold_rows": int(len(folds)),
        "validation_score_rows": int(len(scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the fixed three-model nested landscape-family OOF Decision protocol."
    )
    parser.add_argument("--training-dir", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(check_model_protocol(args.training_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
