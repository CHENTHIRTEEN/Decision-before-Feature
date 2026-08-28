from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from behavior_with_ela.action_dataset import GFE_GATE_BEHAVIOR_FEATURE_COLUMNS
from behavior_with_ela.model import (
    RUN_KEY,
    STATE_KEY,
    _block_function_weights,
    _ma_overlaps_heldout,
)
from behavior_with_ela.phase3 import PHASE3_PROTOCOL
from behavior_with_ela.protocol import ExperimentConfig, load_experiment_config
from utility_labels.efficacy import EFFICACY_FORMULA_PROTOCOL, efficacy_log


GFE_GATE_PROTOCOL = "g_fe_query_gate_retrained_three_algorithm_v1"
GFE_TARGET_PROTOCOL = EFFICACY_FORMULA_PROTOCOL
GFE_THRESHOLD_PROTOCOL = "bbob_family_oof_run_first_trigger_utility_v1"
GFE_TARGET_COLUMN = "g_fe_selected_path"
GFE_MODEL_STREAM = 2026082808


def train_gfe_query_gate(
    *,
    train_config_path: str | Path,
    phase3_model_path: str | Path,
    phase3_control_path: str | Path,
    phase3_policy_runs_path: str | Path,
    output_dir: str | Path,
    validation_control_path: str | Path | None = None,
    validation_phase3_policy_runs_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = load_experiment_config(train_config_path)
    phase3_bundle = joblib.load(phase3_model_path)
    _validate_phase3_source(phase3_bundle, config)
    default_algorithm = str(phase3_bundle["default_algorithm"])
    controls = prepare_gfe_controls(pd.read_parquet(phase3_control_path))
    _validate_training_controls(controls)
    phase3_policy_runs = pd.read_parquet(phase3_policy_runs_path)
    target_query_rate = _voi_query_rate(
        phase3_policy_runs,
        evaluation_split="bbob_train_oof",
    )

    output = Path(output_dir)
    expected = (
        output / "gfe_gate_model.joblib",
        output / "oof_predictions.parquet",
        output / "thresholds.parquet",
        output / "oof_policy_runs.parquet",
        output / "oof_matched_query_rate_policy_runs.parquet",
        output / "matched_query_rate_comparison.parquet",
        output / "training_summary.json",
    )
    if any(path.exists() for path in expected) and not overwrite:
        raise FileExistsError(f"G_FE Gate outputs already exist: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in expected:
            path.unlink(missing_ok=True)
        (output / "validation_predictions.parquet").unlink(missing_ok=True)
        (output / "validation_policy_runs.parquet").unlink(missing_ok=True)
        (output / "validation_matched_query_rate_policy_runs.parquet").unlink(
            missing_ok=True
        )

    oof = _family_oof_predictions(controls, config)
    thresholds, selected_threshold, oof_runs = fit_gfe_first_trigger_threshold(
        controls=controls,
        predictions=oof,
        default_algorithm=default_algorithm,
    )
    matched_query_rate_threshold = _select_matched_query_rate_threshold(
        thresholds,
        target_query_rate=target_query_rate,
    )
    thresholds["target_voi_query_rate"] = float(target_query_rate)
    thresholds["query_rate_distance_to_voi"] = np.abs(
        thresholds["query_call_rate"].to_numpy(dtype=float)
        - float(target_query_rate)
    )
    thresholds["selected_matched_query_rate_threshold"] = np.isclose(
        thresholds["threshold"].to_numpy(dtype=float),
        matched_query_rate_threshold,
        rtol=0.0,
        atol=0.0,
    )
    matched_oof_runs = replay_gfe_first_trigger(
        controls=controls,
        predictions=oof,
        threshold=matched_query_rate_threshold,
        default_algorithm=default_algorithm,
    )
    matched_oof_runs["query_policy"] = "g_fe_query_gate_matched_rate"
    matched_oof_runs["evaluation_split"] = "bbob_train_oof"
    model = _fit_model(controls, config, fold_number=0)
    bundle = {
        "gfe_gate_protocol": GFE_GATE_PROTOCOL,
        "target_protocol": GFE_TARGET_PROTOCOL,
        "threshold_protocol": GFE_THRESHOLD_PROTOCOL,
        "model": model,
        "decision_threshold": float(selected_threshold),
        "matched_query_rate_threshold": float(matched_query_rate_threshold),
        "matched_query_rate_target": float(target_query_rate),
        "feature_columns": GFE_GATE_BEHAVIOR_FEATURE_COLUMNS,
        "target_column": GFE_TARGET_COLUMN,
        "portfolio": config.algorithms,
        "default_algorithm": default_algorithm,
        "dimension": config.dimension,
        "FE_total": config.fe_total,
        "population_size": config.population_size,
        "sampling_protocol": config.sampling_protocol,
        "boundary_handling": config.boundary_handling,
        "query_config": config.query,
        "query_selector_protocol": PHASE3_PROTOCOL,
    }
    joblib.dump(bundle, output / "gfe_gate_model.joblib")
    oof.to_parquet(output / "oof_predictions.parquet", index=False)
    thresholds.to_parquet(output / "thresholds.parquet", index=False)
    oof_runs.to_parquet(output / "oof_policy_runs.parquet", index=False)
    matched_oof_runs.to_parquet(
        output / "oof_matched_query_rate_policy_runs.parquet",
        index=False,
    )

    validation_rows = 0
    validation_runs = 0
    matched_comparison_frames = [
        summarize_matched_query_rate_comparison(
            phase3_policy_runs=phase3_policy_runs,
            gfe_matched_runs=matched_oof_runs,
            evaluation_split="bbob_train_oof",
            target_query_rate=target_query_rate,
        )
    ]
    if validation_control_path is not None:
        validation = prepare_gfe_controls(pd.read_parquet(validation_control_path))
        _validate_evaluation_controls(validation)
        predictions = predict_gfe_rows(model, validation)
        runs = replay_gfe_first_trigger(
            controls=validation,
            predictions=predictions,
            threshold=selected_threshold,
            default_algorithm=default_algorithm,
        )
        predictions.to_parquet(
            output / "validation_predictions.parquet",
            index=False,
        )
        runs.to_parquet(output / "validation_policy_runs.parquet", index=False)
        matched_runs = replay_gfe_first_trigger(
            controls=validation,
            predictions=predictions,
            threshold=matched_query_rate_threshold,
            default_algorithm=default_algorithm,
        )
        matched_runs["query_policy"] = "g_fe_query_gate_matched_rate"
        matched_runs["evaluation_split"] = "bbob_validation"
        matched_runs.to_parquet(
            output / "validation_matched_query_rate_policy_runs.parquet",
            index=False,
        )
        if validation_phase3_policy_runs_path is not None:
            validation_phase3_runs = pd.read_parquet(
                validation_phase3_policy_runs_path
            )
            matched_comparison_frames.append(
                summarize_matched_query_rate_comparison(
                    phase3_policy_runs=validation_phase3_runs,
                    gfe_matched_runs=matched_runs,
                    evaluation_split="bbob_validation",
                    target_query_rate=target_query_rate,
                )
            )
        validation_rows = len(validation)
        validation_runs = len(runs)

    matched_comparison = pd.concat(
        matched_comparison_frames,
        ignore_index=True,
    )
    matched_comparison.to_parquet(
        output / "matched_query_rate_comparison.parquet",
        index=False,
    )

    selected = thresholds.loc[thresholds["selected_threshold"]].iloc[0]
    summary = {
        "gfe_gate_protocol": GFE_GATE_PROTOCOL,
        "target_protocol": GFE_TARGET_PROTOCOL,
        "threshold_protocol": GFE_THRESHOLD_PROTOCOL,
        "feature_columns": list(GFE_GATE_BEHAVIOR_FEATURE_COLUMNS),
        "feature_count": len(GFE_GATE_BEHAVIOR_FEATURE_COLUMNS),
        "target_column": GFE_TARGET_COLUMN,
        "portfolio": list(config.algorithms),
        "train_control_rows": int(len(controls)),
        "bbob_train_control_rows": int(
            controls["suite"].astype(str).eq("bbob").sum()
        ),
        "mabbob_train_control_rows": int(
            controls["suite"].astype(str).eq("mabbob").sum()
        ),
        "oof_rows": int(len(oof)),
        "oof_runs": int(len(oof_runs)),
        "decision_threshold": float(selected_threshold),
        "matched_query_rate_threshold": float(matched_query_rate_threshold),
        "matched_query_rate_target": float(target_query_rate),
        "matched_query_rate_observed_oof": float(
            matched_oof_runs["query_triggered"].mean()
        ),
        "function_balanced_mean_g_fe_selected_path": float(
            selected["function_balanced_mean_g_fe_selected_path"]
        ),
        "query_call_rate": float(selected["query_call_rate"]),
        "validation_control_rows": int(validation_rows),
        "validation_runs": int(validation_runs),
        "validation_rows_used_for_fit": 0,
        "validation_rows_used_for_threshold_fit": 0,
        "runtime_used_in_target_or_threshold": False,
    }
    with (output / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def prepare_gfe_controls(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        *GFE_GATE_BEHAVIOR_FEATURE_COLUMNS,
        *STATE_KEY,
        "suite",
        "family",
        "cv_group_id",
        "function_id",
        "component_functions",
        "dimension",
        "FE_total",
        "no_query_continue_log10_loss",
        "no_query_continue_action_loss",
        "query_descriptor_selected_log10_loss",
        "query_descriptor_selected_action_loss",
        "query_descriptor_selected_algorithm",
        "query_descriptor_selected_equals_prefix",
        "epsilon_p",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"G_FE control paths are missing columns: {missing}")
    result = rows.copy()
    gap_skip = result["no_query_continue_action_loss"].to_numpy(dtype=float)
    gap_query = result["query_descriptor_selected_action_loss"].to_numpy(
        dtype=float
    )
    epsilon_p = result["epsilon_p"].to_numpy(dtype=float)
    result[GFE_TARGET_COLUMN] = efficacy_log(
        gap_skip=gap_skip,
        gap_query=gap_query,
        epsilon_p=epsilon_p,
    )
    result["g_fe_selected_path_gt_zero"] = result[GFE_TARGET_COLUMN].gt(0.0)
    result["g_fe_skip_log10_loss"] = result["no_query_continue_log10_loss"]
    result["g_fe_query_selected_log10_loss"] = result[
        "query_descriptor_selected_log10_loss"
    ]
    result["g_fe_skip_gap"] = gap_skip
    result["g_fe_query_selected_gap"] = gap_query
    if not np.isfinite(result[GFE_TARGET_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError("G_FE selected-path targets must be finite")
    return result


def predict_gfe_rows(model: Pipeline, controls: pd.DataFrame) -> pd.DataFrame:
    output = controls.copy()
    output["predicted_g_fe_selected_path"] = np.asarray(
        model.predict(controls[list(GFE_GATE_BEHAVIOR_FEATURE_COLUMNS)]),
        dtype=float,
    )
    if not np.isfinite(
        output["predicted_g_fe_selected_path"].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("G_FE Gate produced non-finite predictions")
    return output


def fit_gfe_first_trigger_threshold(
    *,
    controls: pd.DataFrame,
    predictions: pd.DataFrame,
    default_algorithm: str,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    scores = np.unique(
        predictions["predicted_g_fe_selected_path"].to_numpy(dtype=float)
    )
    thresholds = _score_thresholds(scores)
    rows: list[dict[str, Any]] = []
    replays: dict[float, pd.DataFrame] = {}
    for threshold in thresholds:
        replay = replay_gfe_first_trigger(
            controls=controls,
            predictions=predictions,
            threshold=float(threshold),
            default_algorithm=default_algorithm,
        )
        replays[float(threshold)] = replay
        rows.append(
            {
                "threshold": float(threshold),
                "function_balanced_mean_g_fe_selected_path": float(
                    replay.groupby("cv_group_id")["decision_utility"]
                    .mean()
                    .mean()
                ),
                "mean_g_fe_selected_path": float(
                    replay["decision_utility"].mean()
                ),
                "median_g_fe_selected_path": float(
                    replay["decision_utility"].median()
                ),
                "query_call_rate": float(replay["query_triggered"].mean()),
                "precision_g_fe_gt_zero_under_calls": _called_precision(replay),
                "run_count": int(len(replay)),
            }
        )
    table = pd.DataFrame(rows)
    ordered = table.sort_values(
        [
            "function_balanced_mean_g_fe_selected_path",
            "query_call_rate",
            "threshold",
        ],
        ascending=[False, True, False],
        kind="mergesort",
    )
    selected_index = int(ordered.index[0])
    table["selected_threshold"] = table.index == selected_index
    selected = float(table.loc[selected_index, "threshold"])
    return table, selected, replays[selected]


def replay_gfe_first_trigger(
    *,
    controls: pd.DataFrame,
    predictions: pd.DataFrame,
    threshold: float,
    default_algorithm: str,
) -> pd.DataFrame:
    score = predictions[
        [*STATE_KEY, "predicted_g_fe_selected_path"]
    ].copy()
    states = controls.drop(
        columns=["predicted_g_fe_selected_path"],
        errors="ignore",
    ).merge(score, on=list(STATE_KEY), how="inner", validate="one_to_one")
    run_rows: list[dict[str, Any]] = []
    for run_key, run in states.groupby(list(RUN_KEY), sort=False):
        ordered = run.sort_values(
            ["FE", "decision_opportunity_index"],
            kind="mergesort",
        )
        eligible = ordered.loc[
            ordered["predicted_g_fe_selected_path"].to_numpy(dtype=float)
            > float(threshold)
        ]
        first = ordered.iloc[0]
        prefix = str(first["prefix_algorithm"])
        if eligible.empty:
            selected_algorithm = prefix
            selected_fe = None
            selected_opportunity = None
            selected_score = None
            utility = 0.0
            query_triggered = False
            selected_equals_prefix = True
            selected_log10_loss = float(first["no_query_continue_log10_loss"])
        else:
            selected = eligible.iloc[0]
            selected_algorithm = str(
                selected["query_descriptor_selected_algorithm"]
            )
            selected_fe = int(selected["FE"])
            selected_opportunity = int(selected["decision_opportunity_index"])
            selected_score = float(selected["predicted_g_fe_selected_path"])
            utility = float(selected[GFE_TARGET_COLUMN])
            query_triggered = True
            selected_equals_prefix = bool(selected_algorithm == prefix)
            selected_log10_loss = float(
                selected["query_descriptor_selected_log10_loss"]
            )
        run_rows.append(
            {
                "problem_id": str(run_key[0]),
                "prefix_algorithm": prefix,
                "seed": int(run_key[2]),
                "suite": str(first["suite"]),
                "function_id": str(first["function_id"]),
                "family": str(first["family"]),
                "cv_group_id": str(first["cv_group_id"]),
                "threshold": float(threshold),
                "query_triggered": bool(query_triggered),
                "selected_FE": selected_fe,
                "selected_decision_opportunity_index": selected_opportunity,
                "selected_score": selected_score,
                "selected_algorithm": selected_algorithm,
                "selected_equals_default": bool(
                    selected_algorithm == default_algorithm
                ),
                "selected_terminal_log10_loss": selected_log10_loss,
                "decision_utility": utility,
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
        raise ValueError("G_FE first-trigger replay produced no runs")
    return result.sort_values(list(RUN_KEY), kind="mergesort").reset_index(drop=True)


def _family_oof_predictions(
    controls: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    bbob = controls.loc[controls["suite"].astype(str).eq("bbob")].copy()
    families = tuple(sorted(set(bbob["family"].astype(str))))
    if len(families) < 2:
        raise ValueError("G_FE Gate OOF requires at least two BBOB families")
    predictions: list[pd.DataFrame] = []
    for fold_number, heldout_family in enumerate(families, start=1):
        heldout_functions = set(
            bbob.loc[
                bbob["family"].astype(str).eq(heldout_family),
                "function_id",
            ].astype(str)
        )
        train_mask = ~(
            controls["suite"].astype(str).eq("bbob")
            & controls["family"].astype(str).eq(heldout_family)
        )
        ma_safe = ~controls.apply(
            lambda row: _ma_overlaps_heldout(row, heldout_functions),
            axis=1,
        )
        fold_train = controls.loc[train_mask & ma_safe].copy()
        fold_eval = bbob.loc[
            bbob["family"].astype(str).eq(heldout_family)
        ].copy()
        model = _fit_model(
            fold_train,
            config,
            fold_number=fold_number,
        )
        predicted = predict_gfe_rows(model, fold_eval)
        predicted["oof_fold"] = int(fold_number)
        predicted["heldout_family"] = heldout_family
        predictions.append(predicted)
    result = pd.concat(predictions, ignore_index=True)
    if len(result) != len(bbob) or result.duplicated(list(STATE_KEY)).any():
        raise RuntimeError("G_FE Gate OOF coverage differs from BBOB controls")
    return result


def _fit_model(
    controls: pd.DataFrame,
    config: ExperimentConfig,
    *,
    fold_number: int,
) -> Pipeline:
    random_state = int(
        np.random.SeedSequence(
            [GFE_MODEL_STREAM, int(fold_number), config.dimension]
        ).generate_state(1, dtype=np.uint32)[0]
    )
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=8,
                    max_features="sqrt",
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(
        controls[list(GFE_GATE_BEHAVIOR_FEATURE_COLUMNS)],
        controls[GFE_TARGET_COLUMN].to_numpy(dtype=float),
        regressor__sample_weight=_block_function_weights(controls),
    )
    return model


def validate_gfe_bundle(bundle: dict[str, Any], config: ExperimentConfig) -> None:
    if str(bundle.get("gfe_gate_protocol")) != GFE_GATE_PROTOCOL:
        raise ValueError("G_FE Gate model protocol differs from online runner")
    if tuple(bundle.get("feature_columns", ())) != GFE_GATE_BEHAVIOR_FEATURE_COLUMNS:
        raise ValueError("G_FE Gate feature columns differ from online runner")
    if tuple(bundle.get("portfolio", ())) != config.algorithms:
        raise ValueError("G_FE Gate portfolio differs from online config")
    if str(bundle.get("default_algorithm")) not in config.algorithms:
        raise ValueError("G_FE Gate default algorithm is outside the portfolio")
    for name, value in (
        ("dimension", config.dimension),
        ("FE_total", config.fe_total),
        ("population_size", config.population_size),
    ):
        if int(bundle.get(name, -1)) != int(value):
            raise ValueError(f"G_FE Gate {name} differs from online config")
    if str(bundle.get("sampling_protocol")) != config.sampling_protocol:
        raise ValueError("G_FE Gate sampling protocol differs from online config")
    if str(bundle.get("boundary_handling")) != config.boundary_handling:
        raise ValueError("G_FE Gate boundary handling differs from online config")
    if bundle.get("query_config") != config.query:
        raise ValueError("G_FE Gate Query config differs from online config")
    if str(bundle.get("query_selector_protocol")) != PHASE3_PROTOCOL:
        raise ValueError("G_FE Gate Query Selector protocol differs from Phase 3")
    if not np.isfinite(float(bundle.get("matched_query_rate_threshold", np.nan))):
        raise ValueError("G_FE Gate matched-query-rate threshold must be finite")


def _voi_query_rate(
    runs: pd.DataFrame,
    *,
    evaluation_split: str,
) -> float:
    required = {"evaluation_split", "query_policy", "query_triggered", *RUN_KEY}
    missing = sorted(required.difference(runs.columns))
    if missing:
        raise ValueError(f"Phase 3 policy runs are missing columns: {missing}")
    selected = runs.loc[
        runs["evaluation_split"].astype(str).eq(evaluation_split)
        & runs["query_policy"].astype(str).eq("voi_query")
    ].copy()
    if selected.empty:
        raise ValueError(
            f"Phase 3 policy runs contain no VOI rows for {evaluation_split}"
        )
    if selected.duplicated(list(RUN_KEY)).any():
        raise ValueError("Phase 3 VOI policy runs contain duplicate run keys")
    return float(selected["query_triggered"].astype(bool).mean())


def _select_matched_query_rate_threshold(
    thresholds: pd.DataFrame,
    *,
    target_query_rate: float,
) -> float:
    if not 0.0 <= float(target_query_rate) <= 1.0:
        raise ValueError("target Query rate must lie in [0, 1]")
    required = {
        "threshold",
        "query_call_rate",
        "function_balanced_mean_g_fe_selected_path",
    }
    missing = sorted(required.difference(thresholds.columns))
    if missing:
        raise ValueError(f"G_FE threshold table is missing columns: {missing}")
    ranked = thresholds.copy()
    ranked["query_rate_distance"] = np.abs(
        ranked["query_call_rate"].to_numpy(dtype=float)
        - float(target_query_rate)
    )
    selected = ranked.sort_values(
        [
            "query_rate_distance",
            "function_balanced_mean_g_fe_selected_path",
            "threshold",
        ],
        ascending=[True, False, False],
        kind="mergesort",
    ).iloc[0]
    return float(selected["threshold"])


def summarize_matched_query_rate_comparison(
    *,
    phase3_policy_runs: pd.DataFrame,
    gfe_matched_runs: pd.DataFrame,
    evaluation_split: str,
    target_query_rate: float,
) -> pd.DataFrame:
    required = {
        "evaluation_split",
        "query_policy",
        "query_triggered",
        "selected_terminal_log10_loss",
        "cv_group_id",
        *RUN_KEY,
    }
    missing = sorted(required.difference(phase3_policy_runs.columns))
    if missing:
        raise ValueError(f"Phase 3 matched-rate comparison is missing columns: {missing}")
    missing = sorted(required.difference(gfe_matched_runs.columns))
    if missing:
        raise ValueError(f"G_FE matched-rate comparison is missing columns: {missing}")
    phase3 = phase3_policy_runs.loc[
        phase3_policy_runs["evaluation_split"].astype(str).eq(evaluation_split)
        & phase3_policy_runs["query_policy"].astype(str).isin(
            ("voi_query", "uncertainty_matched_query_rate")
        )
    ].copy()
    gfe = gfe_matched_runs.loc[
        gfe_matched_runs["evaluation_split"].astype(str).eq(evaluation_split)
        & gfe_matched_runs["query_policy"].astype(str).eq(
            "g_fe_query_gate_matched_rate"
        )
    ].copy()
    combined = pd.concat([phase3, gfe], ignore_index=True)
    expected_policies = {
        "voi_query",
        "uncertainty_matched_query_rate",
        "g_fe_query_gate_matched_rate",
    }
    observed = set(combined["query_policy"].astype(str))
    if observed != expected_policies:
        raise ValueError(
            "matched-query-rate comparison does not contain all three policies"
        )
    if combined.duplicated([*RUN_KEY, "query_policy"]).any():
        raise ValueError("matched-query-rate comparison contains duplicate policy runs")
    rows: list[dict[str, Any]] = []
    for policy, group in combined.groupby("query_policy", sort=True):
        function_means = group.groupby("cv_group_id")[
            "selected_terminal_log10_loss"
        ].mean()
        observed_rate = float(group["query_triggered"].astype(bool).mean())
        rows.append(
            {
                "evaluation_split": evaluation_split,
                "query_policy": str(policy),
                "run_count": int(len(group)),
                "function_count": int(group["cv_group_id"].astype(str).nunique()),
                "target_voi_query_rate": float(target_query_rate),
                "observed_query_rate": observed_rate,
                "absolute_query_rate_difference_to_target": abs(
                    observed_rate - float(target_query_rate)
                ),
                "function_balanced_mean_terminal_log10_loss": float(
                    function_means.mean()
                ),
                "median_terminal_log10_loss": float(
                    group["selected_terminal_log10_loss"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def _validate_phase3_source(
    bundle: dict[str, Any],
    config: ExperimentConfig,
) -> None:
    if str(bundle.get("phase3_protocol")) != PHASE3_PROTOCOL:
        raise ValueError("G_FE training requires the current Phase 3 model bundle")
    if tuple(bundle.get("portfolio", ())) != config.algorithms:
        raise ValueError("G_FE training and Phase 3 portfolios differ")
    for name, value in (
        ("dimension", config.dimension),
        ("FE_total", config.fe_total),
        ("population_size", config.population_size),
    ):
        if int(bundle.get(name, -1)) != int(value):
            raise ValueError(f"G_FE training and Phase 3 {name} differ")
    if str(bundle.get("sampling_protocol")) != config.sampling_protocol:
        raise ValueError("G_FE training and Phase 3 sampling protocols differ")
    if str(bundle.get("boundary_handling")) != config.boundary_handling:
        raise ValueError("G_FE training and Phase 3 boundary handling differ")
    if bundle.get("local_landscape_config") != config.local_landscape:
        raise ValueError("G_FE training and Phase 3 local-landscape configs differ")
    if bundle.get("query_config") != config.query:
        raise ValueError("G_FE training and Phase 3 Query configs differ")


def _validate_training_controls(controls: pd.DataFrame) -> None:
    suites = set(controls["suite"].astype(str))
    if not suites.issuperset({"bbob", "mabbob"}):
        raise ValueError(
            "G_FE Gate training requires BBOB and selected MA-BBOB controls"
        )
    _validate_evaluation_controls(controls)


def _validate_evaluation_controls(controls: pd.DataFrame) -> None:
    if controls.empty:
        raise ValueError("G_FE Gate controls must not be empty")
    if controls.duplicated(list(STATE_KEY)).any():
        raise ValueError("G_FE Gate controls contain duplicate decision states")


def _score_thresholds(scores: np.ndarray) -> np.ndarray:
    ordered = np.unique(np.asarray(scores, dtype=float))
    if ordered.size == 0 or not np.isfinite(ordered).all():
        raise ValueError("G_FE threshold scores must be finite and non-empty")
    if ordered.size == 1:
        return np.asarray([ordered[0] - 1e-12, ordered[0]], dtype=float)
    return np.concatenate(
        ([ordered[0] - 1e-12], (ordered[:-1] + ordered[1:]) / 2.0, [ordered[-1]])
    )


def _called_precision(runs: pd.DataFrame) -> float:
    called = runs.loc[runs["query_triggered"].astype(bool)]
    if called.empty:
        return float("nan")
    return float(called["decision_utility"].gt(0.0).mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain the G_FE Query Gate on the three-algorithm protocol."
    )
    parser.add_argument("--train-config", default="configs/behavior_with_ela_train.yaml")
    parser.add_argument(
        "--phase3-model",
        default="results/behavior_with_ela/model/adaptive_query/phase3_models.joblib",
    )
    parser.add_argument(
        "--phase3-controls",
        default="results/behavior_with_ela/model/adaptive_query/query_control_paths.parquet",
    )
    parser.add_argument(
        "--phase3-policy-runs",
        default=(
            "results/behavior_with_ela/model/adaptive_query/"
            "train_query_policy_runs.parquet"
        ),
    )
    parser.add_argument(
        "--validation-controls",
        default=(
            "results/behavior_with_ela/model/adaptive_query/"
            "validation_query_control_paths.parquet"
        ),
    )
    parser.add_argument(
        "--validation-phase3-policy-runs",
        default=(
            "results/behavior_with_ela/model/adaptive_query/"
            "validation_query_policy_runs.parquet"
        ),
    )
    parser.add_argument(
        "--output",
        default="results/behavior_with_ela/model/gfe_query_gate",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    validation_path = Path(args.validation_controls)
    validation_phase3_runs_path = Path(args.validation_phase3_policy_runs)
    summary = train_gfe_query_gate(
        train_config_path=args.train_config,
        phase3_model_path=args.phase3_model,
        phase3_control_path=args.phase3_controls,
        phase3_policy_runs_path=args.phase3_policy_runs,
        validation_control_path=(
            validation_path if validation_path.exists() else None
        ),
        validation_phase3_policy_runs_path=(
            validation_phase3_runs_path
            if validation_phase3_runs_path.exists()
            else None
        ),
        output_dir=args.output,
        overwrite=args.overwrite,
    )
    print(
        f"trained G_FE Gate on {summary['train_control_rows']} states; "
        f"threshold={summary['decision_threshold']:.12g}"
    )


if __name__ == "__main__":
    main()
