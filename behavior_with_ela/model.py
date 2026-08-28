from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
)
from sklearn.pipeline import Pipeline

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS
from behavior_with_ela.action_dataset import action_shard_paths
from behavior_with_ela.collection import shard_paths
from behavior_with_ela.protocol import (
    CORE_PORTFOLIO,
    ExperimentConfig,
    load_experiment_config,
)


MODEL_PROTOCOL = "per_action_behavior_rf_three_class"
THRESHOLD_PROTOCOL = "bbob_family_oof_first_trigger_terminal_gain"
NOISE_DELTA_PROTOCOL = "bbob_function_balanced_within_state_absolute_deviation_quantile"
MODEL_STREAM_CODE = 2026082703
CALIBRATION_BINS = 10
RUN_KEY = ("problem_id", "prefix_algorithm", "seed")
STATE_KEY = (*RUN_KEY, "FE", "decision_opportunity_index")


def train_behavior_action_models(
    *,
    train_config_path: str | Path,
    output_dir: str | Path,
    validation_config_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    train_config = load_experiment_config(train_config_path)
    train = read_action_datasets(train_config)
    repetitions = read_action_repetitions(train_config)
    noise_table, noise_delta = estimate_noise_gain_delta(
        repetitions,
        quantile=train_config.noise_delta_quantile,
    )
    practical_delta = max(float(train_config.domain_gain_delta), noise_delta)
    train = apply_practical_gain_delta(train, practical_delta)
    _validate_training_frame(train, train_config)
    output = Path(output_dir)
    expected_outputs = (
        output / "models.joblib",
        output / "oof_action_predictions.parquet",
        output / "oof_first_trigger_runs.parquet",
        output / "threshold_summary.parquet",
        output / "noise_delta_summary.parquet",
        output / "oof_calibration.parquet",
        output / "training_summary.json",
    )
    if any(path.exists() for path in expected_outputs) and not overwrite:
        raise FileExistsError(
            f"model outputs already exist; pass --overwrite: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in expected_outputs:
            path.unlink(missing_ok=True)
        (output / "validation_action_predictions.parquet").unlink(missing_ok=True)
        (output / "validation_first_trigger_runs.parquet").unlink(missing_ok=True)
        (output / "validation_calibration.parquet").unlink(missing_ok=True)

    oof = _family_oof_predictions(train, train_config)
    threshold_table, threshold, oof_runs = fit_first_trigger_threshold(
        action_rows=train,
        action_predictions=oof,
        practical_delta=practical_delta,
    )
    models = _fit_models(train, train_config, fold_number=0)
    default_algorithm, sbs_table = _train_derived_sbs(train_config)
    bundle = {
        "model_protocol": MODEL_PROTOCOL,
        "threshold_protocol": THRESHOLD_PROTOCOL,
        "feature_columns": tuple(SELECTOR_BEHAVIOR_FEATURE_COLUMNS),
        "portfolio": train_config.algorithms,
        "models": models,
        "decision_threshold": float(threshold),
        "domain_gain_delta": float(train_config.domain_gain_delta),
        "noise_gain_delta": float(noise_delta),
        "noise_delta_quantile": float(train_config.noise_delta_quantile),
        "noise_delta_protocol": NOISE_DELTA_PROTOCOL,
        "practical_gain_delta": float(practical_delta),
        "default_algorithm": default_algorithm,
        "dimension": train_config.dimension,
        "FE_total": train_config.fe_total,
        "population_size": train_config.population_size,
        "sampling_protocol": train_config.sampling_protocol,
        "boundary_handling": train_config.boundary_handling,
    }
    joblib.dump(bundle, output / "models.joblib")
    oof.to_parquet(output / "oof_action_predictions.parquet", index=False)
    oof_runs.to_parquet(output / "oof_first_trigger_runs.parquet", index=False)
    threshold_table.to_parquet(output / "threshold_summary.parquet", index=False)
    noise_table.to_parquet(output / "noise_delta_summary.parquet", index=False)
    calibration_table(oof).to_parquet(output / "oof_calibration.parquet", index=False)
    sbs_table.to_parquet(output / "sbs_training_summary.parquet", index=False)

    action_metrics = _action_metrics(oof)
    selected_threshold = threshold_table.loc[
        threshold_table["selected_threshold"],
    ].iloc[0]
    summary: dict[str, Any] = {
        "model_protocol": MODEL_PROTOCOL,
        "threshold_protocol": THRESHOLD_PROTOCOL,
        "portfolio": list(train_config.algorithms),
        "feature_columns": list(SELECTOR_BEHAVIOR_FEATURE_COLUMNS),
        "feature_count": len(SELECTOR_BEHAVIOR_FEATURE_COLUMNS),
        "train_rows": int(len(train)),
        "train_switch_rows": int((~train["action_equals_prefix"].astype(bool)).sum()),
        "oof_switch_rows": int(len(oof)),
        "oof_runs": int(len(oof_runs)),
        "decision_threshold": float(threshold),
        "default_algorithm": default_algorithm,
        "domain_gain_delta": float(train_config.domain_gain_delta),
        "noise_gain_delta": float(noise_delta),
        "noise_delta_quantile": float(train_config.noise_delta_quantile),
        "noise_delta_protocol": NOISE_DELTA_PROTOCOL,
        "practical_gain_delta": float(practical_delta),
        "action_metrics": action_metrics,
        "first_trigger_metrics": {
            "function_balanced_mean_gain": float(
                selected_threshold["function_balanced_mean_gain"]
            ),
            "mean_gain": float(selected_threshold["mean_gain"]),
            "median_gain": float(selected_threshold["median_gain"]),
            "switch_rate": float(selected_threshold["switch_rate"]),
            "acceptable_policy_rate": float(
                selected_threshold["acceptable_policy_rate"]
            ),
            "mean_normalized_one_switch_regret": float(
                selected_threshold["mean_normalized_one_switch_regret"]
            ),
        },
        "validation_rows_used_for_model_fit": 0,
        "validation_rows_used_for_threshold_fit": 0,
    }

    if validation_config_path is not None:
        validation_config = load_experiment_config(validation_config_path)
        validation = read_action_datasets(validation_config)
        validation = apply_practical_gain_delta(validation, practical_delta)
        _validate_evaluation_frame(validation, validation_config, bundle)
        validation_predictions = predict_action_rows(models, validation)
        validation_runs = replay_first_trigger(
            action_rows=validation,
            action_predictions=validation_predictions,
            threshold=threshold,
            practical_delta=practical_delta,
            default_algorithm=default_algorithm,
        )
        validation_predictions.to_parquet(
            output / "validation_action_predictions.parquet", index=False
        )
        validation_runs.to_parquet(
            output / "validation_first_trigger_runs.parquet", index=False
        )
        calibration_table(validation_predictions).to_parquet(
            output / "validation_calibration.parquet", index=False
        )
        summary["validation"] = {
            "switch_rows": int(len(validation_predictions)),
            "runs": int(len(validation_runs)),
            "function_balanced_mean_gain": _function_balanced_mean(
                validation_runs, "selected_action_gain"
            ),
            "mean_gain": float(validation_runs["selected_action_gain"].mean()),
            "median_gain": float(validation_runs["selected_action_gain"].median()),
            "switch_rate": float(validation_runs["switch_triggered"].mean()),
            "acceptable_policy_rate": float(
                validation_runs["acceptable_policy"].mean()
            ),
            "mean_normalized_one_switch_regret": float(
                validation_runs["normalized_one_switch_regret"].mean()
            ),
        }

    with (output / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def read_action_datasets(config: ExperimentConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for suite in config.suites:
        for function in suite.functions:
            _, path = action_shard_paths(config, suite, function)
            if not path.exists():
                raise FileNotFoundError(
                    f"missing action-gain dataset {path}; run behavior-ela-build-actions first"
                )
            frames.append(pd.read_parquet(path))
    if not frames:
        raise ValueError("no action-gain datasets were found")
    return pd.concat(frames, ignore_index=True)


def read_action_repetitions(config: ExperimentConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for suite in config.suites:
        for function in suite.functions:
            path, _ = action_shard_paths(config, suite, function)
            if not path.exists():
                raise FileNotFoundError(
                    f"missing action repetition data {path}; run behavior-ela-build-actions first"
                )
            frames.append(pd.read_parquet(path))
    if not frames:
        raise ValueError("no action repetition datasets were found")
    return pd.concat(frames, ignore_index=True)


def estimate_noise_gain_delta(
    repetitions: pd.DataFrame,
    *,
    quantile: float,
) -> tuple[pd.DataFrame, float]:
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("noise delta quantile must be in (0, 1)")
    required = {
        *STATE_KEY,
        "suite",
        "cv_group_id",
        "candidate_action",
        "action_equals_prefix",
        "action_status",
        "action_gain_vs_continue",
        "replicate_id",
    }
    missing = sorted(required.difference(repetitions.columns))
    if missing:
        raise ValueError(f"action repetition data is missing columns: {missing}")
    eligible = repetitions.loc[
        repetitions["suite"].astype(str).eq("bbob")
        & ~repetitions["action_equals_prefix"].astype(bool)
        & repetitions["action_status"].astype(str).eq("completed")
    ].copy()
    group_columns = [*STATE_KEY, "candidate_action"]
    counts = eligible.groupby(group_columns, sort=False)["replicate_id"].transform("size")
    eligible = eligible.loc[counts > 1].copy()
    if eligible.empty:
        raise ValueError(
            "noise delta estimation requires repeated completed BBOB switch actions"
        )
    medians = eligible.groupby(group_columns, sort=False)[
        "action_gain_vs_continue"
    ].transform("median")
    eligible["within_state_absolute_gain_deviation"] = np.abs(
        eligible["action_gain_vs_continue"].to_numpy(dtype=float)
        - medians.to_numpy(dtype=float)
    )
    rows: list[dict[str, Any]] = []
    for function_id, function_rows in eligible.groupby("cv_group_id", sort=True):
        deviations = function_rows[
            "within_state_absolute_gain_deviation"
        ].to_numpy(dtype=float)
        rows.append(
            {
                "noise_delta_protocol": NOISE_DELTA_PROTOCOL,
                "cv_group_id": str(function_id),
                "noise_delta_quantile": float(quantile),
                "function_noise_gain_delta": float(
                    np.quantile(deviations, float(quantile))
                ),
                "repeated_state_action_count": int(
                    function_rows[group_columns].drop_duplicates().shape[0]
                ),
                "repetition_row_count": int(len(function_rows)),
            }
        )
    table = pd.DataFrame(rows)
    noise_delta = float(table["function_noise_gain_delta"].mean())
    if not np.isfinite(noise_delta) or noise_delta < 0.0:
        raise RuntimeError("estimated noise gain delta must be finite and non-negative")
    table["function_balanced_noise_gain_delta"] = noise_delta
    return table, noise_delta


def apply_practical_gain_delta(
    rows: pd.DataFrame,
    practical_delta: float,
) -> pd.DataFrame:
    delta = float(practical_delta)
    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError("practical gain delta must be finite and non-negative")
    result = rows.copy()
    gains = result["action_gain_vs_continue"].to_numpy(dtype=float)
    result["action_gain_class"] = np.select(
        [gains > delta, gains < -delta],
        ["improve", "degrade"],
        default="equivalent",
    )
    best_loss = result.groupby(list(STATE_KEY), sort=False)[
        "log10_action_loss"
    ].transform("min")
    result["acceptable_action"] = (
        result["log10_action_loss"].to_numpy(dtype=float)
        - best_loss.to_numpy(dtype=float)
        <= delta
    )
    result["practical_gain_delta"] = delta
    return result


def predict_action_rows(
    models: dict[str, Pipeline],
    action_rows: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = SELECTOR_BEHAVIOR_FEATURE_COLUMNS,
) -> pd.DataFrame:
    switch_rows = action_rows.loc[
        ~action_rows["action_equals_prefix"].astype(bool)
    ].copy()
    predictions: list[pd.DataFrame] = []
    for algorithm in CORE_PORTFOLIO:
        rows = switch_rows.loc[
            switch_rows["candidate_action"].astype(str).eq(algorithm)
        ].copy()
        if rows.empty:
            continue
        model = models[algorithm]
        features = rows[list(feature_columns)]
        rows["predicted_action_class"] = model.predict(features).astype(str)
        rows["predicted_improve_probability"] = _class_probability(
            model,
            features,
            "improve",
        )
        predictions.append(rows)
    if not predictions:
        raise ValueError("no switch-action predictions were produced")
    return pd.concat(predictions, ignore_index=True)


def fit_first_trigger_threshold(
    *,
    action_rows: pd.DataFrame,
    action_predictions: pd.DataFrame,
    practical_delta: float,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    scores = np.unique(
        action_predictions["predicted_improve_probability"].to_numpy(dtype=float)
    )
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("OOF improve probabilities must be finite and non-empty")
    thresholds = _threshold_candidates(scores)
    table = _sweep_first_trigger_thresholds(
        action_rows=action_rows,
        action_predictions=action_predictions,
        thresholds=thresholds,
        practical_delta=practical_delta,
    )
    ordered = table.sort_values(
        ["function_balanced_mean_gain", "switch_rate", "threshold"],
        ascending=[False, True, False],
        kind="mergesort",
    )
    selected_index = int(ordered.index[0])
    table["selected_threshold"] = table.index == selected_index
    selected = float(table.loc[selected_index, "threshold"])
    oof_runs = replay_first_trigger(
        action_rows=action_rows,
        action_predictions=action_predictions,
        threshold=selected,
        practical_delta=practical_delta,
        default_algorithm=None,
    )
    return table, selected, oof_runs


def _first_trigger_run_context(
    *,
    action_rows: pd.DataFrame,
    action_predictions: pd.DataFrame,
) -> dict[str, Any]:
    bbob = action_rows.loc[action_rows["suite"].astype(str).eq("bbob")].copy()
    prediction_key = [*STATE_KEY, "candidate_action"]
    score_columns = [
        *prediction_key,
        "predicted_action_class",
        "predicted_improve_probability",
    ]
    predictions = action_predictions[score_columns].copy()
    switch = bbob.loc[~bbob["action_equals_prefix"].astype(bool)].merge(
        predictions,
        on=prediction_key,
        how="inner",
        validate="one_to_one",
    )
    continue_rows = bbob.loc[bbob["action_equals_prefix"].astype(bool)].copy()
    if switch.empty or continue_rows.empty:
        raise ValueError("first-trigger replay requires switch and continue action rows")
    portfolio_order = {algorithm: index for index, algorithm in enumerate(CORE_PORTFOLIO)}
    switch["candidate_order"] = (
        switch["candidate_action"].astype(str).map(portfolio_order).astype(int)
    )

    run_columns = list(RUN_KEY)
    state_best = (
        switch.sort_values(
            [
                "FE",
                "decision_opportunity_index",
                "predicted_improve_probability",
                "candidate_order",
            ],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .groupby([*run_columns, "FE", "decision_opportunity_index"], sort=False)
        .head(1)
        .sort_values([*run_columns, "FE", "decision_opportunity_index"], kind="mergesort")
        .reset_index(drop=True)
    )
    switch = switch.sort_values(
        [*run_columns, "FE", "decision_opportunity_index", "candidate_action"],
        kind="mergesort",
    )
    continue_first = (
        continue_rows.sort_values(
            [*run_columns, "FE", "decision_opportunity_index"], kind="mergesort"
        )
        .groupby(run_columns, sort=False, as_index=False)
        .nth(0)
    )
    best_gain = switch.groupby(run_columns, sort=False)["action_gain_vs_continue"].max()

    runs = state_best[run_columns].drop_duplicates().reset_index(drop=True)
    runs = runs.merge(
        continue_first[["problem_id", "prefix_algorithm", "seed", "log10_action_loss"]].rename(
            columns={"log10_action_loss": "continue_terminal_log10_loss"}
        ),
        on=["problem_id", "prefix_algorithm", "seed"],
        how="left",
        validate="one_to_one",
    )
    runs = runs.merge(
        best_gain.rename("best_observed_one_switch_gain").reset_index(),
        on=run_columns,
        how="left",
        validate="one_to_one",
    )
    for column in ("function_id", "family", "cv_group_id"):
        runs[column] = runs["problem_id"].map(
            dict(zip(state_best["problem_id"], state_best[column]))
        )
    if runs[["continue_terminal_log10_loss", "best_observed_one_switch_gain"]].isna().any().any():
        raise RuntimeError("first-trigger run context is missing run-level outcomes")
    runs["best_observed_one_switch_gain"] = runs[
        "best_observed_one_switch_gain"
    ].clip(lower=0.0)
    runs = runs.sort_values(run_columns, kind="mergesort").reset_index(drop=True)

    member_positions = state_best.groupby(run_columns, sort=False).indices
    context_scores: list[np.ndarray] = []
    context_suffix_positions: list[np.ndarray] = []
    context_offsets = np.zeros(len(runs) + 1, dtype=np.int64)
    scores = state_best["predicted_improve_probability"].to_numpy(dtype=float)
    for index, run_row in enumerate(runs[run_columns].itertuples(index=False, name=None)):
        positions = member_positions[run_row]
        begin = int(context_offsets[index])
        context_offsets[index + 1] = begin + int(positions.size)
        run_scores = scores[np.sort(positions)]
        order = np.argsort(run_scores, kind="mergesort")
        positions_in_run = np.arange(run_scores.size, dtype=np.int64)
        suffix_min = np.minimum.accumulate(positions_in_run[order][::-1])[::-1]
        context_scores.append(run_scores[order])
        context_suffix_positions.append(suffix_min)
    return {
        "runs": runs,
        "state_best_gains": state_best["action_gain_vs_continue"].to_numpy(dtype=float),
        "state_best_losses": state_best["log10_action_loss"].to_numpy(dtype=float),
        "run_offsets": context_offsets,
        "run_sorted_scores": context_scores,
        "run_suffix_positions": context_suffix_positions,
    }


def _sweep_first_trigger_thresholds(
    *,
    action_rows: pd.DataFrame,
    action_predictions: pd.DataFrame,
    thresholds: np.ndarray,
    practical_delta: float,
) -> pd.DataFrame:
    context = _first_trigger_run_context(
        action_rows=action_rows,
        action_predictions=action_predictions,
    )
    runs = context["runs"]
    run_count = len(runs)
    threshold_count = int(len(thresholds))
    thresholds_array = np.asarray(thresholds, dtype=float)
    selected_gain = np.zeros((threshold_count, run_count), dtype=float)
    switched = np.zeros((threshold_count, run_count), dtype=bool)
    for run_index in range(run_count):
        begin = int(context["run_offsets"][run_index])
        end = int(context["run_offsets"][run_index + 1])
        sorted_scores = context["run_sorted_scores"][run_index]
        suffix_positions = context["run_suffix_positions"][run_index]
        size = sorted_scores.size
        below = np.searchsorted(sorted_scores, thresholds_array, side="right")
        eligible = below < size
        first_positions = suffix_positions[below[eligible]]
        selected_gain[eligible, run_index] = context["state_best_gains"][begin:end][
            first_positions
        ]
        switched[eligible, run_index] = True
    best_gain = runs["best_observed_one_switch_gain"].to_numpy(dtype=float)
    regret = np.where(switched, best_gain[None, :] - selected_gain, best_gain[None, :])
    practical = float(practical_delta)
    regret_scale = np.maximum(best_gain, max(practical, 1e-12))
    normalized_regret = regret / regret_scale[None, :]
    acceptable = regret <= practical

    group_ids = runs["cv_group_id"].astype(str).to_numpy()
    function_balanced = np.zeros(threshold_count, dtype=float)
    for group in sorted(set(group_ids.tolist())):
        function_balanced += selected_gain[:, group_ids == group].mean(axis=1)
    function_balanced /= max(len(set(group_ids.tolist())), 1)

    return pd.DataFrame(
        {
            "threshold": thresholds_array,
            "function_balanced_mean_gain": function_balanced,
            "mean_gain": selected_gain.mean(axis=1),
            "median_gain": np.median(selected_gain, axis=1),
            "switch_rate": switched.mean(axis=1),
            "acceptable_policy_rate": acceptable.mean(axis=1),
            "mean_normalized_one_switch_regret": normalized_regret.mean(axis=1),
            "run_count": np.full(threshold_count, run_count, dtype=np.int64),
        }
    )


def replay_first_trigger(
    *,
    action_rows: pd.DataFrame,
    action_predictions: pd.DataFrame,
    threshold: float,
    practical_delta: float,
    default_algorithm: str | None,
) -> pd.DataFrame:
    bbob = action_rows.loc[action_rows["suite"].astype(str).eq("bbob")].copy()
    prediction_key = [*STATE_KEY, "candidate_action"]
    score_columns = [
        *prediction_key,
        "predicted_action_class",
        "predicted_improve_probability",
    ]
    predictions = action_predictions[score_columns].copy()
    switch = bbob.loc[~bbob["action_equals_prefix"].astype(bool)].merge(
        predictions,
        on=prediction_key,
        how="inner",
        validate="one_to_one",
    )
    continue_rows = bbob.loc[bbob["action_equals_prefix"].astype(bool)].copy()
    if switch.empty or continue_rows.empty:
        raise ValueError("first-trigger replay requires switch and continue action rows")
    portfolio_order = {algorithm: index for index, algorithm in enumerate(CORE_PORTFOLIO)}
    switch["candidate_order"] = (
        switch["candidate_action"].astype(str).map(portfolio_order).astype(int)
    )

    run_rows: list[dict[str, Any]] = []
    for run_key, run_switch in switch.groupby(list(RUN_KEY), sort=False):
        ordered = run_switch.sort_values(
            ["FE", "decision_opportunity_index", "candidate_action"],
            kind="mergesort",
        )
        state_best = (
            ordered.sort_values(
                [
                    "FE",
                    "decision_opportunity_index",
                    "predicted_improve_probability",
                    "candidate_order",
                ],
                ascending=[True, True, False, True],
                kind="mergesort",
            )
            .groupby(["FE", "decision_opportunity_index"], sort=False)
            .head(1)
            .sort_values(["FE", "decision_opportunity_index"], kind="mergesort")
        )
        eligible = state_best.loc[
            state_best["predicted_improve_probability"].to_numpy(dtype=float)
            > float(threshold)
        ]
        first = ordered.iloc[0]
        prefix_algorithm = str(first["prefix_algorithm"])
        run_continue = continue_rows.loc[
            continue_rows["problem_id"].astype(str).eq(str(run_key[0]))
            & continue_rows["prefix_algorithm"].astype(str).eq(prefix_algorithm)
            & continue_rows["seed"].astype(int).eq(int(run_key[2]))
        ].sort_values(["FE", "decision_opportunity_index"], kind="mergesort")
        if run_continue.empty:
            raise RuntimeError("first-trigger run is missing its continue-current outcome")
        continue_log10_loss = float(run_continue.iloc[0]["log10_action_loss"])
        best_observed_gain = max(
            0.0,
            float(ordered["action_gain_vs_continue"].max()),
        )
        if eligible.empty:
            selected_algorithm = prefix_algorithm
            selected_gain = 0.0
            selected_fe = None
            selected_opportunity = None
            selected_score = None
            switch_triggered = False
            selected_log10_loss = continue_log10_loss
        else:
            selected = eligible.iloc[0]
            selected_algorithm = str(selected["candidate_action"])
            selected_gain = float(selected["action_gain_vs_continue"])
            selected_fe = int(selected["FE"])
            selected_opportunity = int(selected["decision_opportunity_index"])
            selected_score = float(selected["predicted_improve_probability"])
            switch_triggered = True
            selected_log10_loss = float(selected["log10_action_loss"])
        selected_equals_prefix = selected_algorithm == prefix_algorithm
        one_switch_regret = float(best_observed_gain - selected_gain)
        regret_scale = max(
            float(best_observed_gain),
            float(practical_delta),
            1e-12,
        )
        run_rows.append(
            {
                "problem_id": str(run_key[0]),
                "prefix_algorithm": prefix_algorithm,
                "seed": int(run_key[2]),
                "function_id": str(first["function_id"]),
                "family": str(first["family"]),
                "cv_group_id": str(first["cv_group_id"]),
                "threshold": float(threshold),
                "switch_triggered": bool(switch_triggered),
                "selected_algorithm": selected_algorithm,
                "selected_FE": selected_fe,
                "selected_decision_opportunity_index": selected_opportunity,
                "selected_score": selected_score,
                "selected_action_gain": selected_gain,
                "selected_terminal_log10_loss": selected_log10_loss,
                "continue_terminal_log10_loss": continue_log10_loss,
                "best_observed_one_switch_gain": best_observed_gain,
                "one_switch_regret": one_switch_regret,
                "normalized_one_switch_regret": float(
                    one_switch_regret / regret_scale
                ),
                "normalized_regret_denominator": regret_scale,
                "acceptable_policy": bool(
                    best_observed_gain - selected_gain <= float(practical_delta)
                ),
                "selected_equals_default": (
                    None
                    if default_algorithm is None
                    else bool(selected_algorithm == default_algorithm)
                ),
                "selected_equals_prefix": bool(selected_equals_prefix),
                "handoff_required": bool(not selected_equals_prefix),
                "handoff_type": (
                    "native_optimizer_state"
                    if selected_equals_prefix
                    else "population_transfer_initialization"
                ),
            }
        )
    result = pd.DataFrame(run_rows)
    if result.empty:
        raise ValueError("first-trigger replay produced no run rows")
    return result.sort_values(list(RUN_KEY), kind="mergesort").reset_index(drop=True)


def _family_oof_predictions(
    train: pd.DataFrame,
    config: ExperimentConfig,
    *,
    feature_columns: tuple[str, ...] = SELECTOR_BEHAVIOR_FEATURE_COLUMNS,
) -> pd.DataFrame:
    bbob = train.loc[train["suite"].astype(str).eq("bbob")].copy()
    families = tuple(sorted(set(bbob["family"].astype(str))))
    if len(families) < 2:
        raise ValueError("BBOB family OOF requires at least two landscape families")
    predictions: list[pd.DataFrame] = []
    for fold_number, heldout_family in enumerate(families, start=1):
        heldout_functions = set(
            bbob.loc[
                bbob["family"].astype(str).eq(heldout_family), "function_id"
            ].astype(str)
        )
        train_mask = ~(
            train["suite"].astype(str).eq("bbob")
            & train["family"].astype(str).eq(heldout_family)
        )
        ma_safe = ~train.apply(
            lambda row: _ma_overlaps_heldout(row, heldout_functions),
            axis=1,
        )
        fold_train = train.loc[train_mask & ma_safe].copy()
        fold_eval = bbob.loc[bbob["family"].astype(str).eq(heldout_family)].copy()
        models = _fit_models(
            fold_train,
            config,
            fold_number=fold_number,
            feature_columns=feature_columns,
        )
        predicted = predict_action_rows(
            models,
            fold_eval,
            feature_columns=feature_columns,
        )
        predicted["oof_fold"] = int(fold_number)
        predicted["heldout_family"] = heldout_family
        predicted["heldout_function_count"] = len(heldout_functions)
        predictions.append(predicted)
    result = pd.concat(predictions, ignore_index=True)
    expected = bbob.loc[~bbob["action_equals_prefix"].astype(bool)]
    keys = [*STATE_KEY, "candidate_action"]
    if len(result) != len(expected):
        raise RuntimeError(
            f"OOF prediction coverage mismatch: predicted={len(result)}, expected={len(expected)}"
        )
    if result.duplicated(keys).any():
        raise RuntimeError("OOF predictions contain duplicate state-action rows")
    return result


def _fit_models(
    rows: pd.DataFrame,
    config: ExperimentConfig,
    *,
    fold_number: int,
    feature_columns: tuple[str, ...] = SELECTOR_BEHAVIOR_FEATURE_COLUMNS,
) -> dict[str, Pipeline]:
    switch_rows = rows.loc[~rows["action_equals_prefix"].astype(bool)].copy()
    models: dict[str, Pipeline] = {}
    for target_index, algorithm in enumerate(config.algorithms):
        target = switch_rows.loc[
            switch_rows["candidate_action"].astype(str).eq(algorithm)
        ].copy()
        if target.empty:
            raise ValueError(f"no training rows are available for target algorithm {algorithm}")
        random_state = int(
            np.random.SeedSequence(
                [
                    MODEL_STREAM_CODE,
                    int(fold_number),
                    int(target_index),
                    config.dimension,
                ]
            ).generate_state(1, dtype=np.uint32)[0]
        )
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=8,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        weights = _block_function_weights(target)
        model.fit(
            target[list(feature_columns)],
            target["action_gain_class"].astype(str),
            classifier__sample_weight=weights,
        )
        models[algorithm] = model
    return models


def _block_function_weights(rows: pd.DataFrame) -> np.ndarray:
    required = {"suite", "cv_group_id"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"training rows are missing weighting columns: {missing}")
    if rows.empty:
        raise ValueError("training weights require at least one row")

    frame = rows.reset_index(drop=True)
    levels: list[tuple[str, ...]] = [("suite",), ("cv_group_id",)]
    if "prefix_algorithm" in frame.columns:
        levels.append(("prefix_algorithm",))
    run_columns = tuple(
        column for column in ("problem_id", "seed") if column in frame.columns
    )
    if run_columns:
        levels.append(run_columns)
    state_columns = tuple(
        column
        for column in ("FE", "decision_opportunity_index")
        if column in frame.columns
    )
    if state_columns:
        levels.append(state_columns)

    weights = np.zeros(len(frame), dtype=float)

    def assign_equal_mass(
        positions: np.ndarray,
        level_index: int,
        total_mass: float,
    ) -> None:
        if level_index == len(levels):
            weights[positions] = float(total_mass) / len(positions)
            return
        columns = levels[level_index]
        grouped: dict[tuple[str, ...], list[int]] = {}
        for position in positions:
            key = tuple(str(frame.at[int(position), column]) for column in columns)
            grouped.setdefault(key, []).append(int(position))
        child_mass = float(total_mass) / len(grouped)
        for key in sorted(grouped):
            assign_equal_mass(
                np.asarray(grouped[key], dtype=int),
                level_index + 1,
                child_mass,
            )

    assign_equal_mass(np.arange(len(frame), dtype=int), 0, 1.0)
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise RuntimeError("training weights must be finite and positive")
    return weights * (len(weights) / float(np.sum(weights)))


def _class_probability(
    model: Pipeline,
    features: pd.DataFrame,
    label: str,
) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    classes = np.asarray(model.classes_).astype(str)
    matches = np.flatnonzero(classes == str(label))
    if len(matches) == 0:
        return np.zeros(len(features), dtype=float)
    if len(matches) != 1:
        raise RuntimeError("classifier exposes duplicate class labels")
    return probabilities[:, int(matches[0])]


def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    ordered = np.unique(np.asarray(scores, dtype=float))
    if ordered.size == 1:
        return np.asarray([ordered[0] - 1e-12, ordered[0]], dtype=float)
    midpoints = (ordered[:-1] + ordered[1:]) / 2.0
    return np.concatenate(([ordered[0] - 1e-12], midpoints, [ordered[-1]]))


def _action_metrics(oof: pd.DataFrame) -> dict[str, float]:
    truth = oof["action_gain_class"].astype(str)
    predicted = oof["predicted_action_class"].astype(str)
    binary_truth = truth.eq("improve").astype(int)
    probability = oof["predicted_improve_probability"].to_numpy(dtype=float)
    calibration = calibration_table(oof)
    overall = calibration.loc[
        calibration["calibration_scope"].astype(str).eq("all_switch_actions")
    ]
    expected_calibration_error = float(
        np.average(
            overall["absolute_calibration_error"].to_numpy(dtype=float),
            weights=overall["calibration_weight"].to_numpy(dtype=float),
        )
    )
    return {
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "improve_average_precision": (
            float(average_precision_score(binary_truth, probability))
            if binary_truth.nunique() > 1
            else float("nan")
        ),
        "improve_brier": float(brier_score_loss(binary_truth, probability)),
        "improve_expected_calibration_error": expected_calibration_error,
    }


def calibration_table(
    predictions: pd.DataFrame,
    *,
    bins: int = CALIBRATION_BINS,
) -> pd.DataFrame:
    if bins < 2:
        raise ValueError("calibration requires at least two probability bins")
    required = {
        "suite",
        "cv_group_id",
        "candidate_action",
        "action_gain_class",
        "predicted_improve_probability",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"calibration predictions are missing columns: {missing}")
    calibration_rows: list[dict[str, Any]] = []
    groups: list[tuple[str, str | None, pd.DataFrame]] = [
        ("all_switch_actions", None, predictions)
    ]
    groups.extend(
        ("candidate_action", str(action), group)
        for action, group in predictions.groupby("candidate_action", sort=True)
    )
    for scope, action, group in groups:
        frame = group.copy()
        probabilities = frame["predicted_improve_probability"].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or np.any(
            (probabilities < 0.0) | (probabilities > 1.0)
        ):
            raise ValueError("calibration probabilities must lie in [0, 1]")
        frame["calibration_bin"] = np.minimum(
            np.floor(probabilities * bins).astype(int),
            bins - 1,
        )
        frame["calibration_weight"] = _block_function_weights(frame)
        frame["observed_improve"] = frame["action_gain_class"].astype(str).eq(
            "improve"
        )
        for bin_index, bin_rows in frame.groupby("calibration_bin", sort=True):
            weights = bin_rows["calibration_weight"].to_numpy(dtype=float)
            mean_probability = float(
                np.average(
                    bin_rows["predicted_improve_probability"].to_numpy(dtype=float),
                    weights=weights,
                )
            )
            observed_rate = float(
                np.average(
                    bin_rows["observed_improve"].to_numpy(dtype=float),
                    weights=weights,
                )
            )
            calibration_rows.append(
                {
                    "calibration_scope": scope,
                    "candidate_action": action,
                    "calibration_bin": int(bin_index),
                    "probability_lower": float(bin_index / bins),
                    "probability_upper": float((bin_index + 1) / bins),
                    "row_count": int(len(bin_rows)),
                    "calibration_weight": float(np.sum(weights)),
                    "mean_predicted_improve_probability": mean_probability,
                    "observed_improve_rate": observed_rate,
                    "absolute_calibration_error": abs(
                        mean_probability - observed_rate
                    ),
                }
            )
    result = pd.DataFrame(calibration_rows)
    if result.empty:
        raise ValueError("calibration produced no populated probability bins")
    return result


def _train_derived_sbs(
    config: ExperimentConfig,
) -> tuple[str, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for suite in config.suites:
        for function in suite.functions:
            _, final_path, _ = shard_paths(config, suite, function)
            if not final_path.exists():
                raise FileNotFoundError(f"missing final-performance data: {final_path}")
            frame = pd.read_parquet(final_path)
            frame["suite"] = suite.suite
            frames.append(frame)
    final = pd.concat(frames, ignore_index=True)
    final = final.loc[final["algorithm"].astype(str).isin(config.algorithms)].copy()
    per_function = (
        final.groupby(["suite", "function_id", "algorithm"], as_index=False)[
            "log10_gap"
        ].mean()
    )
    per_suite = (
        per_function.groupby(["suite", "algorithm"], as_index=False)["log10_gap"]
        .mean()
        .rename(columns={"log10_gap": "suite_mean_log10_gap"})
    )
    summary = (
        per_suite.groupby("algorithm", as_index=False)["suite_mean_log10_gap"]
        .mean()
        .rename(columns={"suite_mean_log10_gap": "block_balanced_mean_log10_gap"})
    )
    order = {algorithm: index for index, algorithm in enumerate(config.algorithms)}
    summary["portfolio_order"] = summary["algorithm"].map(order).astype(int)
    selected = summary.sort_values(
        ["block_balanced_mean_log10_gap", "portfolio_order"],
        kind="mergesort",
    ).iloc[0]
    summary["selected_sbs"] = summary["algorithm"].astype(str).eq(
        str(selected["algorithm"])
    )
    return str(selected["algorithm"]), summary


def _function_balanced_mean(rows: pd.DataFrame, column: str) -> float:
    return float(rows.groupby("cv_group_id")[column].mean().mean())


def _ma_overlaps_heldout(row: pd.Series, heldout_functions: set[str]) -> bool:
    if str(row["suite"]) != "mabbob":
        return False
    raw = row["component_functions"]
    components = np.asarray(raw, dtype=int).reshape(-1)
    labels = {f"bbob_f{int(value):03d}" for value in components}
    return bool(labels.intersection(heldout_functions))


def _validate_training_frame(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> None:
    _validate_action_frame(frame, config)
    if not set(frame["suite"].astype(str)).issuperset({"bbob", "mabbob"}):
        raise ValueError("formal training requires both BBOB and selected MA-BBOB")


def _validate_action_frame(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> None:
    required = {
        *SELECTOR_BEHAVIOR_FEATURE_COLUMNS,
        "suite",
        "family",
        "cv_group_id",
        "component_functions",
        "candidate_action",
        "prefix_algorithm",
        "action_equals_prefix",
        "action_gain_class",
        "action_gain_vs_continue",
        "log10_action_loss",
        "all_action_paths_completed",
        "all_action_paths_used_planned_FE",
        *STATE_KEY,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"action-gain training data is missing columns: {missing}")
    if len(SELECTOR_BEHAVIOR_FEATURE_COLUMNS) != 28:
        raise RuntimeError("Behavior action model requires exactly 28 Behavior fields")
    observed = set(frame["candidate_action"].astype(str))
    if observed != set(config.algorithms):
        raise ValueError(
            f"training action portfolio differs from config: {sorted(observed)}"
        )
    if set(frame["dimension"].astype(int)) != {config.dimension}:
        raise ValueError("training data must contain only the configured dimension")
    if not frame["all_action_paths_completed"].astype(bool).all():
        raise ValueError("scientific action labels require completed action paths")
    if not frame["all_action_paths_used_planned_FE"].astype(bool).all():
        raise ValueError("scientific action labels require the planned FE budget")


def _validate_evaluation_frame(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    bundle: dict[str, Any],
) -> None:
    if tuple(bundle["portfolio"]) != config.algorithms:
        raise ValueError("evaluation portfolio differs from the trained model")
    if int(bundle["dimension"]) != config.dimension:
        raise ValueError("evaluation dimension differs from the trained model")
    if int(bundle["FE_total"]) != config.fe_total:
        raise ValueError("evaluation FE_total differs from the trained model")
    if str(bundle["sampling_protocol"]) != config.sampling_protocol:
        raise ValueError("evaluation sampling protocol differs from training")
    if str(bundle["boundary_handling"]) != config.boundary_handling:
        raise ValueError("evaluation boundary handling differs from training")
    if not np.isclose(
        float(bundle["domain_gain_delta"]),
        float(config.domain_gain_delta),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("evaluation domain gain delta differs from training")
    if not np.isclose(
        float(bundle["noise_delta_quantile"]),
        float(config.noise_delta_quantile),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("evaluation noise delta quantile differs from training")
    _validate_action_frame(frame, config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train per-action Behavior models with BBOB family OOF thresholding."
    )
    parser.add_argument("--train-config", default="configs/behavior_with_ela_train.yaml")
    parser.add_argument("--validation-config", default=None)
    parser.add_argument(
        "--output",
        default="results/behavior_with_ela/model/behavior_action_gain",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = train_behavior_action_models(
        train_config_path=args.train_config,
        validation_config_path=args.validation_config,
        output_dir=args.output,
        overwrite=args.overwrite,
    )
    print(
        f"trained {len(summary['portfolio'])} per-action models with "
        f"{summary['oof_switch_rows']} BBOB OOF switch rows; "
        f"threshold={summary['decision_threshold']:.12g}"
    )


if __name__ == "__main__":
    main()
