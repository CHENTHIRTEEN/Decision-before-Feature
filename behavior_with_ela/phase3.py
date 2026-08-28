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
from sklearn.pipeline import Pipeline

from behavior_with_ela.action_dataset import GFE_GATE_BEHAVIOR_FEATURE_COLUMNS
from behavior_with_ela.model import (
    RUN_KEY,
    STATE_KEY,
    _block_function_weights,
    _family_oof_predictions,
    _fit_models,
    _ma_overlaps_heldout,
    apply_practical_gain_delta,
    fit_first_trigger_threshold,
    predict_action_rows,
    read_action_datasets,
)
from behavior_with_ela.phase2 import PHASE2_FEATURE_GROUPS
from behavior_with_ela.phase3_query_data import query_action_shard_paths
from behavior_with_ela.protocol import ExperimentConfig, load_experiment_config
from landscape_queries.specs import DESCRIPTOR_CHEAP_COLUMNS


PHASE3_PROTOCOL = "adaptive_query_voi_one_switch_v1"
QUERY_NO_DESCRIPTOR_FEATURES = tuple(
    PHASE2_FEATURE_GROUPS["M4_behavior_local_landscape_uncertainty"]
)
QUERY_DESCRIPTOR_FEATURES = (
    *QUERY_NO_DESCRIPTOR_FEATURES,
    *DESCRIPTOR_CHEAP_COLUMNS,
)
VOI_DERIVED_FEATURE_COLUMNS = (
    "pre_top_improve_probability",
    "pre_top2_probability_margin",
    "pre_mean_binary_action_entropy",
    "pre_local_uncertainty_mean",
)
VOI_FEATURE_COLUMNS = (
    *QUERY_NO_DESCRIPTOR_FEATURES,
    *VOI_DERIVED_FEATURE_COLUMNS,
)
GFE_GATE_EXTRA_FEATURE_COLUMNS = tuple(
    column
    for column in GFE_GATE_BEHAVIOR_FEATURE_COLUMNS
    if column not in QUERY_NO_DESCRIPTOR_FEATURES
)
if GFE_GATE_EXTRA_FEATURE_COLUMNS != ("bf_search_maturity_linear",):
    raise RuntimeError("G_FE Gate must add exactly the selected maturity field")
VOI_MODEL_STREAM = 2026082803
VOI_LABELS = ("degrade", "equivalent", "improve")


def train_adaptive_query_policy(
    *,
    train_config_path: str | Path,
    phase1_model_path: str | Path,
    output_dir: str | Path,
    validation_config_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = load_experiment_config(train_config_path)
    phase1_bundle = joblib.load(phase1_model_path)
    default_algorithm = str(phase1_bundle["default_algorithm"])
    practical_delta = float(phase1_bundle["practical_gain_delta"])
    no_query = apply_practical_gain_delta(
        read_action_datasets(config),
        practical_delta,
    )
    query = apply_practical_gain_delta(
        read_query_action_datasets(config),
        practical_delta,
    )
    _validate_query_frames(no_query, query)

    output = Path(output_dir)
    expected = (
        output / "phase3_models.joblib",
        output / "query_selector_oof_predictions.parquet",
        output / "query_selector_thresholds.parquet",
        output / "query_control_paths.parquet",
        output / "voi_oof_predictions.parquet",
        output / "voi_thresholds.parquet",
        output / "train_query_policy_runs.parquet",
        output / "query_policy_summary.parquet",
        output / "phase3_summary.json",
    )
    if any(path.exists() for path in expected) and not overwrite:
        raise FileExistsError(f"Phase 3 outputs already exist; pass --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in expected:
            path.unlink(missing_ok=True)
        (output / "validation_query_control_paths.parquet").unlink(missing_ok=True)
        (output / "validation_query_policy_runs.parquet").unlink(missing_ok=True)

    pre_oof = _integrated_query_oof_predictions(
        no_query,
        config,
        feature_columns=QUERY_NO_DESCRIPTOR_FEATURES,
        fold_offset=50_000,
    )
    _, pre_threshold, _ = fit_first_trigger_threshold(
        action_rows=no_query,
        action_predictions=pre_oof,
        practical_delta=practical_delta,
    )
    query_no_descriptor_oof = _integrated_query_oof_predictions(
        query,
        config,
        feature_columns=QUERY_NO_DESCRIPTOR_FEATURES,
        fold_offset=60_000,
    )
    q0_thresholds, q0_threshold, _ = fit_first_trigger_threshold(
        action_rows=query,
        action_predictions=query_no_descriptor_oof,
        practical_delta=practical_delta,
    )
    query_descriptor_oof = _integrated_query_oof_predictions(
        query,
        config,
        feature_columns=QUERY_DESCRIPTOR_FEATURES,
        fold_offset=70_000,
    )
    q1_thresholds, q1_threshold, _ = fit_first_trigger_threshold(
        action_rows=query,
        action_predictions=query_descriptor_oof,
        practical_delta=practical_delta,
    )
    query_no_descriptor_oof["query_selector"] = "matched_cost_no_descriptors"
    query_descriptor_oof["query_selector"] = "query_with_descriptors"
    q0_thresholds["query_selector"] = "matched_cost_no_descriptors"
    q1_thresholds["query_selector"] = "query_with_descriptors"
    query_selector_predictions = pd.concat(
        [query_no_descriptor_oof, query_descriptor_oof],
        ignore_index=True,
    )
    query_selector_thresholds = pd.concat(
        [q0_thresholds, q1_thresholds],
        ignore_index=True,
    )

    controls = build_query_control_paths(
        no_query_actions=no_query,
        query_actions=query,
        pre_predictions=pre_oof,
        pre_threshold=pre_threshold,
        query_no_descriptor_predictions=query_no_descriptor_oof,
        query_no_descriptor_threshold=q0_threshold,
        query_descriptor_predictions=query_descriptor_oof,
        query_descriptor_threshold=q1_threshold,
        practical_delta=practical_delta,
        default_algorithm=default_algorithm,
    )
    voi_oof = _voi_family_oof_predictions(controls, config)
    voi_thresholds, voi_threshold, voi_runs = fit_voi_threshold(
        controls=controls,
        predictions=voi_oof,
        default_algorithm=default_algorithm,
    )
    voi_runs["query_policy"] = "voi_query"
    bbob_controls = controls.loc[controls["suite"].astype(str).eq("bbob")].copy()
    baseline_runs, uncertainty_threshold = _query_baseline_runs(
        bbob_controls,
        target_query_rate=float(voi_runs["query_triggered"].mean()),
        default_algorithm=default_algorithm,
    )
    train_runs = pd.concat([voi_runs, baseline_runs], ignore_index=True)
    train_runs["evaluation_split"] = "bbob_train_oof"
    policy_summary = summarize_query_policies(train_runs)

    pre_models = _fit_models(
        no_query,
        config,
        fold_number=40_001,
        feature_columns=QUERY_NO_DESCRIPTOR_FEATURES,
    )
    q0_models = _fit_models(
        query,
        config,
        fold_number=40_002,
        feature_columns=QUERY_NO_DESCRIPTOR_FEATURES,
    )
    q1_models = _fit_models(
        query,
        config,
        fold_number=40_003,
        feature_columns=QUERY_DESCRIPTOR_FEATURES,
    )
    voi_model = _fit_voi_model(controls, config, fold_number=40_004)

    validation_runs_output: list[pd.DataFrame] = []
    if validation_config_path is not None:
        validation_config = load_experiment_config(validation_config_path)
        validation_no_query = apply_practical_gain_delta(
            read_action_datasets(validation_config),
            practical_delta,
        )
        validation_query = apply_practical_gain_delta(
            read_query_action_datasets(validation_config),
            practical_delta,
        )
        _validate_query_frames(validation_no_query, validation_query)
        validation_pre_predictions = predict_action_rows(
            pre_models,
            validation_no_query,
            feature_columns=QUERY_NO_DESCRIPTOR_FEATURES,
        )
        validation_q0_predictions = predict_action_rows(
            q0_models,
            validation_query,
            feature_columns=QUERY_NO_DESCRIPTOR_FEATURES,
        )
        validation_q1_predictions = predict_action_rows(
            q1_models,
            validation_query,
            feature_columns=QUERY_DESCRIPTOR_FEATURES,
        )
        validation_controls = build_query_control_paths(
            no_query_actions=validation_no_query,
            query_actions=validation_query,
            pre_predictions=validation_pre_predictions,
            pre_threshold=pre_threshold,
            query_no_descriptor_predictions=validation_q0_predictions,
            query_no_descriptor_threshold=q0_threshold,
            query_descriptor_predictions=validation_q1_predictions,
            query_descriptor_threshold=q1_threshold,
            practical_delta=practical_delta,
            default_algorithm=default_algorithm,
        )
        validation_voi = _predict_voi(voi_model, validation_controls)
        validation_voi_runs = replay_query_policy(
            controls=validation_controls,
            predictions=validation_voi,
            threshold=voi_threshold,
            score_column="predicted_query_improve_probability",
            default_algorithm=default_algorithm,
        )
        validation_voi_runs["query_policy"] = "voi_query"
        validation_baselines, _ = _query_baseline_runs(
            validation_controls,
            target_query_rate=float(voi_runs["query_triggered"].mean()),
            default_algorithm=default_algorithm,
            fixed_uncertainty_threshold=uncertainty_threshold,
        )
        validation_runs = pd.concat(
            [validation_voi_runs, validation_baselines],
            ignore_index=True,
        )
        validation_runs["evaluation_split"] = "bbob_validation"
        validation_controls.to_parquet(
            output / "validation_query_control_paths.parquet",
            index=False,
        )
        validation_runs.to_parquet(
            output / "validation_query_policy_runs.parquet",
            index=False,
        )
        validation_runs_output.append(validation_runs)
        policy_summary = pd.concat(
            [policy_summary, summarize_query_policies(validation_runs)],
            ignore_index=True,
        )

    query_selector_predictions.to_parquet(
        output / "query_selector_oof_predictions.parquet",
        index=False,
    )
    query_selector_thresholds.to_parquet(
        output / "query_selector_thresholds.parquet",
        index=False,
    )
    controls.to_parquet(output / "query_control_paths.parquet", index=False)
    voi_oof.to_parquet(output / "voi_oof_predictions.parquet", index=False)
    voi_thresholds.to_parquet(output / "voi_thresholds.parquet", index=False)
    train_runs.to_parquet(output / "train_query_policy_runs.parquet", index=False)
    policy_summary.to_parquet(output / "query_policy_summary.parquet", index=False)
    joblib.dump(
        {
            "phase3_protocol": PHASE3_PROTOCOL,
            "pre_action_models": pre_models,
            "pre_action_threshold": float(pre_threshold),
            "query_no_descriptor_models": q0_models,
            "query_no_descriptor_threshold": float(q0_threshold),
            "query_descriptor_models": q1_models,
            "query_descriptor_threshold": float(q1_threshold),
            "voi_model": voi_model,
            "voi_threshold": float(voi_threshold),
            "uncertainty_matched_threshold": float(uncertainty_threshold),
            "voi_feature_columns": VOI_FEATURE_COLUMNS,
            "query_no_descriptor_features": QUERY_NO_DESCRIPTOR_FEATURES,
            "query_descriptor_features": QUERY_DESCRIPTOR_FEATURES,
            "practical_gain_delta": practical_delta,
            "default_algorithm": default_algorithm,
            "portfolio": config.algorithms,
            "dimension": config.dimension,
            "FE_total": config.fe_total,
            "population_size": config.population_size,
            "sampling_protocol": config.sampling_protocol,
            "boundary_handling": config.boundary_handling,
            "local_landscape_config": config.local_landscape,
            "query_config": config.query,
        },
        output / "phase3_models.joblib",
    )
    result = {
        "phase3_protocol": PHASE3_PROTOCOL,
        "query_id": config.query.query_id,
        "query_FE": int(
            query["FE_query"].astype(int).drop_duplicates().iloc[0]
        ),
        "no_query_state_action_rows": int(len(no_query)),
        "query_state_action_rows": int(len(query)),
        "control_state_rows": int(len(controls)),
        "bbob_control_state_rows": int(len(bbob_controls)),
        "mabbob_control_state_rows": int(
            controls["suite"].astype(str).eq("mabbob").sum()
        ),
        "voi_threshold": float(voi_threshold),
        "train_voi_query_rate": float(voi_runs["query_triggered"].mean()),
        "validation_included": validation_config_path is not None,
        "validation_rows_used_for_fit": 0,
        "validation_rows_used_for_threshold_fit": 0,
    }
    with (output / "phase3_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def read_query_action_datasets(config: ExperimentConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for suite in config.suites:
        for function in suite.functions:
            path = query_action_shard_paths(config, suite, function)[3]
            if not path.exists():
                raise FileNotFoundError(
                    f"missing query-adjusted action data {path}; run behavior-ela-build-query-actions first"
                )
            frames.append(pd.read_parquet(path))
    if not frames:
        raise ValueError("no query-adjusted action datasets were found")
    return pd.concat(frames, ignore_index=True)


def build_query_control_paths(
    *,
    no_query_actions: pd.DataFrame,
    query_actions: pd.DataFrame,
    pre_predictions: pd.DataFrame,
    pre_threshold: float,
    query_no_descriptor_predictions: pd.DataFrame,
    query_no_descriptor_threshold: float,
    query_descriptor_predictions: pd.DataFrame,
    query_descriptor_threshold: float,
    practical_delta: float,
    default_algorithm: str,
) -> pd.DataFrame:
    pre = _state_action_choices(
        action_rows=no_query_actions,
        predictions=pre_predictions,
        threshold=pre_threshold,
        prefix="pre_no_query",
        default_algorithm=default_algorithm,
    )
    q0 = _state_action_choices(
        action_rows=query_actions,
        predictions=query_no_descriptor_predictions,
        threshold=query_no_descriptor_threshold,
        prefix="query_no_descriptor",
        default_algorithm=default_algorithm,
    )
    q1 = _state_action_choices(
        action_rows=query_actions,
        predictions=query_descriptor_predictions,
        threshold=query_descriptor_threshold,
        prefix="query_descriptor",
        default_algorithm=default_algorithm,
    )
    keys = list(STATE_KEY)
    control = pre.merge(q0, on=keys, how="inner", validate="one_to_one").merge(
        q1,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    query_continue = query_actions.loc[
        query_actions["action_equals_prefix"].astype(bool),
        [
            *keys,
            "action_loss",
            "log10_action_loss",
            "query_continuation_action_loss",
            "query_continuation_log10_action_loss",
        ],
    ].rename(
        columns={
            "action_loss": "sampling_only_continue_action_loss",
            "log10_action_loss": "sampling_only_continue_log10_loss",
            "query_continuation_action_loss": (
                "matched_cost_continue_without_sample_direct_action_loss"
            ),
            "query_continuation_log10_action_loss": (
                "matched_cost_continue_without_sample_direct_log10_loss"
            ),
        }
    )
    no_query_continue = no_query_actions.loc[
        no_query_actions["action_equals_prefix"].astype(bool),
        [*keys, "action_loss", "log10_action_loss"],
    ].rename(
        columns={
            "action_loss": "no_query_continue_action_loss",
            "log10_action_loss": "no_query_continue_log10_loss",
        }
    )
    control = control.merge(
        query_continue,
        on=keys,
        how="inner",
        validate="one_to_one",
    ).merge(
        no_query_continue,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    control["query_information_increment"] = (
        control["query_no_descriptor_selected_log10_loss"]
        - control["query_descriptor_selected_log10_loss"]
    )
    control["query_operational_increment"] = (
        control["pre_no_query_selected_log10_loss"]
        - control["query_descriptor_selected_log10_loss"]
    )
    control["query_matched_cost_no_descriptor_increment"] = (
        control["pre_no_query_selected_log10_loss"]
        - control["query_no_descriptor_selected_log10_loss"]
    )
    control["query_sampling_only_continue_increment"] = (
        control["no_query_continue_log10_loss"]
        - control["sampling_only_continue_log10_loss"]
    )
    values = control["query_operational_increment"].to_numpy(dtype=float)
    control["query_operational_class"] = np.select(
        [values > practical_delta, values < -practical_delta],
        ["improve", "degrade"],
        default="equivalent",
    )
    uncertainty_columns = [
        column
        for column in QUERY_NO_DESCRIPTOR_FEATURES
        if column.endswith("_bootstrap_std")
    ]
    control["pre_local_uncertainty_mean"] = control[uncertainty_columns].mean(
        axis=1,
        skipna=True,
    )
    return control.sort_values([*RUN_KEY, "FE", "decision_opportunity_index"], kind="mergesort").reset_index(drop=True)


def _state_action_choices(
    *,
    action_rows: pd.DataFrame,
    predictions: pd.DataFrame,
    threshold: float,
    prefix: str,
    default_algorithm: str,
) -> pd.DataFrame:
    keys = list(STATE_KEY)
    prediction_key = [*keys, "candidate_action"]
    switch = action_rows.loc[~action_rows["action_equals_prefix"].astype(bool)].merge(
        predictions[
            [
                *prediction_key,
                "predicted_action_class",
                "predicted_improve_probability",
            ]
        ],
        on=prediction_key,
        how="inner",
        validate="one_to_one",
    )
    portfolio_order = {"pso": 0, "shade": 1, "cmaes": 2}
    switch["candidate_order"] = (
        switch["candidate_action"].astype(str).map(portfolio_order).astype(int)
    )
    ranked = switch.sort_values(
        [
            *keys,
            "predicted_improve_probability",
            "candidate_order",
        ],
        ascending=[True] * len(keys) + [False, True],
        kind="mergesort",
    )
    top = ranked.groupby(keys, sort=False).head(1).copy()
    second = ranked.groupby(keys, sort=False).nth(1).reset_index()
    second = second[[*keys, "predicted_improve_probability"]].rename(
        columns={"predicted_improve_probability": "second_probability"}
    )
    top = top.merge(second, on=keys, how="left", validate="one_to_one")
    continue_rows = action_rows.loc[action_rows["action_equals_prefix"].astype(bool)].copy()
    continue_columns = [
        *keys,
        "candidate_action",
        "action_loss",
        "log10_action_loss",
    ]
    context = continue_rows[continue_columns].rename(
        columns={
            "candidate_action": "continue_algorithm",
            "action_loss": "continue_action_loss",
            "log10_action_loss": "continue_log10_loss",
        }
    )
    top = top.merge(context, on=keys, how="inner", validate="one_to_one")
    triggered = top["predicted_improve_probability"].to_numpy(dtype=float) > float(
        threshold
    )
    selected_algorithm = np.where(
        triggered,
        top["candidate_action"].astype(str),
        top["continue_algorithm"].astype(str),
    )
    selected_loss = np.where(
        triggered,
        top["log10_action_loss"].to_numpy(dtype=float),
        top["continue_log10_loss"].to_numpy(dtype=float),
    )
    selected_action_loss = np.where(
        triggered,
        top["action_loss"].to_numpy(dtype=float),
        top["continue_action_loss"].to_numpy(dtype=float),
    )
    probabilities = switch.groupby(keys, sort=False)[
        "predicted_improve_probability"
    ].agg(list)
    entropy_by_key = {
        key if isinstance(key, tuple) else (key,): _mean_binary_entropy(values)
        for key, values in probabilities.items()
    }
    key_values = [tuple(row) for row in top[keys].itertuples(index=False, name=None)]
    output_columns = [*keys]
    if prefix == "pre_no_query":
        output_columns.extend(
            [
                "component_functions",
                "suite",
                "family",
                "cv_group_id",
                "function_id",
                "dimension",
                "FE_ratio",
                "FE_total",
                "remaining_budget_ratio",
                "prefix_best_fitness",
                "benchmark_reference_value",
                "prefix_gap",
                "epsilon_p",
                "is_budget_milestone",
                "budget_milestone_ratio",
                *QUERY_NO_DESCRIPTOR_FEATURES,
                *GFE_GATE_EXTRA_FEATURE_COLUMNS,
            ]
        )
    output = top[output_columns].copy()
    output[f"{prefix}_selected_algorithm"] = selected_algorithm
    output[f"{prefix}_selected_action_loss"] = selected_action_loss
    output[f"{prefix}_selected_log10_loss"] = selected_loss
    output[f"{prefix}_selected_equals_prefix"] = (
        output[f"{prefix}_selected_algorithm"].astype(str)
        == output["prefix_algorithm"].astype(str)
    )
    output[f"{prefix}_selected_equals_default"] = output[
        f"{prefix}_selected_algorithm"
    ].astype(str).eq(str(default_algorithm))
    output[f"{prefix}_handoff_required"] = ~output[
        f"{prefix}_selected_equals_prefix"
    ]
    output[f"{prefix}_handoff_type"] = np.where(
        output[f"{prefix}_handoff_required"].astype(bool),
        "population_transfer_initialization",
        "native_optimizer_state",
    )
    output[f"{prefix}_top_improve_probability"] = top[
        "predicted_improve_probability"
    ].to_numpy(dtype=float)
    output[f"{prefix}_top2_probability_margin"] = (
        top["predicted_improve_probability"].to_numpy(dtype=float)
        - top["second_probability"].to_numpy(dtype=float)
    )
    output[f"{prefix}_mean_binary_action_entropy"] = [
        entropy_by_key[key] for key in key_values
    ]
    if prefix == "pre_no_query":
        output["pre_top_improve_probability"] = output[
            f"{prefix}_top_improve_probability"
        ]
        output["pre_top2_probability_margin"] = output[
            f"{prefix}_top2_probability_margin"
        ]
        output["pre_mean_binary_action_entropy"] = output[
            f"{prefix}_mean_binary_action_entropy"
        ]
    return output


def _mean_binary_entropy(values: list[float]) -> float:
    probabilities = np.clip(np.asarray(values, dtype=float), 1e-12, 1.0 - 1e-12)
    return float(
        np.mean(
            -probabilities * np.log(probabilities)
            - (1.0 - probabilities) * np.log(1.0 - probabilities)
        )
        / np.log(2.0)
    )


def _integrated_query_oof_predictions(
    rows: pd.DataFrame,
    config: ExperimentConfig,
    *,
    feature_columns: tuple[str, ...],
    fold_offset: int,
) -> pd.DataFrame:
    bbob = _family_oof_predictions(
        rows,
        config,
        feature_columns=feature_columns,
    ).copy()
    bbob["oof_partition"] = "bbob_function_family"
    ma = rows.loc[rows["suite"].astype(str).eq("mabbob")].copy()
    if ma.empty:
        return bbob
    if "component_functions" not in rows.columns:
        raise ValueError("selected MA-BBOB Query OOF requires component functions")

    predictions = [bbob]
    groups = tuple(sorted(set(ma["cv_group_id"].astype(str))))
    for fold_index, heldout_group in enumerate(groups, start=1):
        fold_eval = ma.loc[
            ma["cv_group_id"].astype(str).eq(heldout_group)
        ].copy()
        heldout_components: set[int] = set()
        for raw in fold_eval["component_functions"]:
            heldout_components.update(
                int(value) for value in np.asarray(raw, dtype=int).reshape(-1)
            )
        if not heldout_components:
            raise ValueError("selected MA-BBOB OOF fold has no component functions")
        safe = ~rows["component_functions"].apply(
            lambda raw: bool(
                heldout_components.intersection(
                    int(value)
                    for value in np.asarray(raw, dtype=int).reshape(-1)
                )
            )
        )
        fold_train = rows.loc[safe].copy()
        if fold_train.empty:
            raise ValueError(
                "selected MA-BBOB component exclusion leaves no Query training rows"
            )
        models = _fit_models(
            fold_train,
            config,
            fold_number=int(fold_offset + fold_index),
            feature_columns=feature_columns,
        )
        predicted = predict_action_rows(
            models,
            fold_eval,
            feature_columns=feature_columns,
        )
        predicted["oof_fold"] = int(fold_offset + fold_index)
        predicted["heldout_family"] = heldout_group
        predicted["heldout_function_count"] = len(heldout_components)
        predicted["oof_partition"] = "mabbob_component_exclusion"
        predictions.append(predicted)

    result = pd.concat(predictions, ignore_index=True)
    expected = rows.loc[~rows["action_equals_prefix"].astype(bool)]
    keys = [*STATE_KEY, "candidate_action"]
    if len(result) != len(expected):
        raise RuntimeError(
            "integrated Query OOF coverage differs from state-action rows"
        )
    if result.duplicated(keys).any():
        raise RuntimeError("integrated Query OOF contains duplicate state-actions")
    return result


def _voi_family_oof_predictions(
    controls: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    bbob = controls.loc[controls["suite"].astype(str).eq("bbob")].copy()
    families = tuple(sorted(set(bbob["family"].astype(str))))
    predictions: list[pd.DataFrame] = []
    for fold_number, heldout_family in enumerate(families, start=1):
        heldout_functions = set(
            bbob.loc[
                bbob["family"].astype(str).eq(heldout_family), "function_id"
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
        model = _fit_voi_model(fold_train, config, fold_number=fold_number)
        predicted = _predict_voi(model, fold_eval)
        predicted["oof_fold"] = int(fold_number)
        predicted["heldout_family"] = heldout_family
        predictions.append(predicted)
    result = pd.concat(predictions, ignore_index=True)
    keys = [*STATE_KEY]
    if len(result) != len(bbob) or result.duplicated(keys).any():
        raise RuntimeError("VOI OOF prediction coverage differs from BBOB states")
    return result


def _fit_voi_model(
    rows: pd.DataFrame,
    config: ExperimentConfig,
    *,
    fold_number: int,
) -> Pipeline:
    random_state = int(
        np.random.SeedSequence(
            [VOI_MODEL_STREAM, int(fold_number), config.dimension]
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
    weights = _block_function_weights(rows)
    model.fit(
        rows[list(VOI_FEATURE_COLUMNS)],
        rows["query_operational_class"].astype(str),
        classifier__sample_weight=weights,
    )
    return model


def _predict_voi(model: Pipeline, controls: pd.DataFrame) -> pd.DataFrame:
    output = controls.copy()
    features = controls[list(VOI_FEATURE_COLUMNS)]
    output["predicted_query_class"] = model.predict(features).astype(str)
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    labels = np.asarray(model.classes_).astype(str)
    matches = np.flatnonzero(labels == "improve")
    output["predicted_query_improve_probability"] = (
        np.zeros(len(output), dtype=float)
        if len(matches) == 0
        else probabilities[:, int(matches[0])]
    )
    return output


def fit_voi_threshold(
    *,
    controls: pd.DataFrame,
    predictions: pd.DataFrame,
    default_algorithm: str,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    scores = np.unique(
        predictions["predicted_query_improve_probability"].to_numpy(dtype=float)
    )
    thresholds = _score_thresholds(scores)
    rows = []
    replays: dict[float, pd.DataFrame] = {}
    for threshold in thresholds:
        replay = replay_query_policy(
            controls=controls,
            predictions=predictions,
            threshold=float(threshold),
            score_column="predicted_query_improve_probability",
            default_algorithm=default_algorithm,
        )
        replays[float(threshold)] = replay
        rows.append(
            {
                "threshold": float(threshold),
                "function_balanced_mean_query_gain": float(
                    replay.groupby("cv_group_id")["query_policy_gain"].mean().mean()
                ),
                "mean_query_gain": float(replay["query_policy_gain"].mean()),
                "median_query_gain": float(replay["query_policy_gain"].median()),
                "query_rate": float(replay["query_triggered"].mean()),
                "run_count": int(len(replay)),
            }
        )
    table = pd.DataFrame(rows)
    ordered = table.sort_values(
        ["function_balanced_mean_query_gain", "query_rate", "threshold"],
        ascending=[False, True, False],
        kind="mergesort",
    )
    selected_index = int(ordered.index[0])
    table["selected_threshold"] = table.index == selected_index
    selected = float(table.loc[selected_index, "threshold"])
    return table, selected, replays[selected]


def replay_query_policy(
    *,
    controls: pd.DataFrame,
    predictions: pd.DataFrame,
    threshold: float,
    score_column: str,
    default_algorithm: str,
) -> pd.DataFrame:
    keys = [*STATE_KEY]
    score = predictions[[*keys, score_column]].copy()
    states = controls.drop(columns=[score_column], errors="ignore").merge(
        score,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    run_rows = []
    for run_key, run in states.groupby(list(RUN_KEY), sort=False):
        ordered = run.sort_values(["FE", "decision_opportunity_index"], kind="mergesort")
        never = _never_query_run(ordered)
        selected_loss = never["selected_terminal_log10_loss"]
        selected_algorithm = never["selected_algorithm"]
        selected_fe = never["selected_FE"]
        query_triggered = False
        query_score = None
        for _, state in ordered.iterrows():
            if float(state[score_column]) > float(threshold):
                selected_loss = float(state["query_descriptor_selected_log10_loss"])
                selected_algorithm = str(state["query_descriptor_selected_algorithm"])
                selected_fe = int(state["FE"])
                query_triggered = True
                query_score = float(state[score_column])
                break
            if not bool(state["pre_no_query_selected_equals_prefix"]):
                selected_loss = float(state["pre_no_query_selected_log10_loss"])
                selected_algorithm = str(state["pre_no_query_selected_algorithm"])
                selected_fe = int(state["FE"])
                break
        first = ordered.iloc[0]
        prefix = str(first["prefix_algorithm"])
        run_rows.append(
            {
                "problem_id": str(run_key[0]),
                "prefix_algorithm": prefix,
                "seed": int(run_key[2]),
                "function_id": str(first["function_id"]),
                "family": str(first["family"]),
                "cv_group_id": str(first["cv_group_id"]),
                "threshold": float(threshold),
                "query_triggered": bool(query_triggered),
                "query_score": query_score,
                "selected_algorithm": selected_algorithm,
                "selected_FE": selected_fe,
                "selected_terminal_log10_loss": float(selected_loss),
                "never_query_terminal_log10_loss": float(
                    never["selected_terminal_log10_loss"]
                ),
                "query_policy_gain": float(
                    never["selected_terminal_log10_loss"] - selected_loss
                ),
                "selected_equals_default": bool(
                    selected_algorithm == default_algorithm
                ),
                "selected_equals_prefix": bool(selected_algorithm == prefix),
                "handoff_required": bool(selected_algorithm != prefix),
                "handoff_type": (
                    "native_optimizer_state"
                    if selected_algorithm == prefix
                    else "population_transfer_initialization"
                ),
            }
        )
    return pd.DataFrame(run_rows)


def _never_query_run(ordered: pd.DataFrame) -> dict[str, Any]:
    first = ordered.iloc[0]
    prefix = str(first["prefix_algorithm"])
    for _, state in ordered.iterrows():
        if not bool(state["pre_no_query_selected_equals_prefix"]):
            return {
                "selected_algorithm": str(state["pre_no_query_selected_algorithm"]),
                "selected_FE": int(state["FE"]),
                "selected_terminal_log10_loss": float(
                    state["pre_no_query_selected_log10_loss"]
                ),
            }
    return {
        "selected_algorithm": prefix,
        "selected_FE": None,
        "selected_terminal_log10_loss": float(first["no_query_continue_log10_loss"]),
    }


def _query_baseline_runs(
    controls: pd.DataFrame,
    *,
    target_query_rate: float,
    default_algorithm: str,
    fixed_uncertainty_threshold: float | None = None,
) -> tuple[pd.DataFrame, float]:
    never_rows = []
    always_rows = []
    fixed_rows = []
    for run_key, run in controls.groupby(list(RUN_KEY), sort=False):
        ordered = run.sort_values(["FE", "decision_opportunity_index"], kind="mergesort")
        first = ordered.iloc[0]
        never = _never_query_run(ordered)
        never_rows.append(
            _baseline_query_row(
                run_key,
                first,
                never,
                "never_query",
                False,
                default_algorithm=default_algorithm,
            )
        )
        always = {
            "selected_algorithm": str(first["query_descriptor_selected_algorithm"]),
            "selected_FE": int(first["FE"]),
            "selected_terminal_log10_loss": float(
                first["query_descriptor_selected_log10_loss"]
            ),
        }
        always_rows.append(
            _baseline_query_row(
                run_key,
                first,
                always,
                "always_query",
                True,
                never,
                default_algorithm=default_algorithm,
            )
        )
        milestone = ordered.loc[
            ordered["is_budget_milestone"].astype(bool)
            & np.isclose(
                ordered["budget_milestone_ratio"].to_numpy(dtype=float),
                0.30,
                rtol=0.0,
                atol=1e-12,
            )
        ]
        if len(milestone) != 1:
            raise RuntimeError("fixed-time Query requires one 0.30B milestone")
        fixed_state = milestone.iloc[0]
        fixed = {
            "selected_algorithm": str(
                fixed_state["query_descriptor_selected_algorithm"]
            ),
            "selected_FE": int(fixed_state["FE"]),
            "selected_terminal_log10_loss": float(
                fixed_state["query_descriptor_selected_log10_loss"]
            ),
        }
        fixed_rows.append(
            _baseline_query_row(
                run_key,
                first,
                fixed,
                "fixed_030_query",
                True,
                never,
                default_algorithm=default_algorithm,
            )
        )
    if fixed_uncertainty_threshold is None:
        candidates = _score_thresholds(
            controls["pre_mean_binary_action_entropy"].to_numpy(dtype=float)
        )
        choices = []
        for threshold in candidates:
            replay = replay_query_policy(
                controls=controls,
                predictions=controls,
                threshold=float(threshold),
                score_column="pre_mean_binary_action_entropy",
                default_algorithm=default_algorithm,
            )
            choices.append(
                (
                    abs(float(replay["query_triggered"].mean()) - target_query_rate),
                    -float(threshold),
                    float(threshold),
                )
            )
        fixed_uncertainty_threshold = min(choices)[2]
    uncertainty = replay_query_policy(
        controls=controls,
        predictions=controls,
        threshold=fixed_uncertainty_threshold,
        score_column="pre_mean_binary_action_entropy",
        default_algorithm=default_algorithm,
    )
    uncertainty["query_policy"] = "uncertainty_matched_query_rate"
    baseline = pd.concat(
        [
            pd.DataFrame(never_rows),
            pd.DataFrame(always_rows),
            pd.DataFrame(fixed_rows),
            uncertainty,
        ],
        ignore_index=True,
    )
    return baseline, float(fixed_uncertainty_threshold)


def _baseline_query_row(
    run_key: tuple,
    first: pd.Series,
    selected: dict[str, Any],
    policy: str,
    query_triggered: bool,
    never: dict[str, Any] | None = None,
    *,
    default_algorithm: str,
) -> dict[str, Any]:
    reference = selected if never is None else never
    prefix = str(first["prefix_algorithm"])
    algorithm = str(selected["selected_algorithm"])
    return {
        "problem_id": str(run_key[0]),
        "prefix_algorithm": prefix,
        "seed": int(run_key[2]),
        "function_id": str(first["function_id"]),
        "family": str(first["family"]),
        "cv_group_id": str(first["cv_group_id"]),
        "threshold": None,
        "query_triggered": bool(query_triggered),
        "query_score": None,
        "selected_algorithm": algorithm,
        "selected_FE": selected["selected_FE"],
        "selected_terminal_log10_loss": float(
            selected["selected_terminal_log10_loss"]
        ),
        "never_query_terminal_log10_loss": float(
            reference["selected_terminal_log10_loss"]
        ),
        "query_policy_gain": float(
            reference["selected_terminal_log10_loss"]
            - selected["selected_terminal_log10_loss"]
        ),
        "selected_equals_default": bool(algorithm == default_algorithm),
        "selected_equals_prefix": bool(algorithm == prefix),
        "handoff_required": bool(algorithm != prefix),
        "handoff_type": (
            "native_optimizer_state"
            if algorithm == prefix
            else "population_transfer_initialization"
        ),
        "query_policy": policy,
    }


def summarize_query_policies(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, policy), group in runs.groupby(
        ["evaluation_split", "query_policy"],
        sort=False,
    ):
        function_means = group.groupby("cv_group_id")[
            ["selected_terminal_log10_loss", "query_policy_gain"]
        ].mean()
        rows.append(
            {
                "evaluation_split": split,
                "query_policy": policy,
                "run_count": int(len(group)),
                "query_rate": float(group["query_triggered"].mean()),
                "function_balanced_mean_log10_loss": float(
                    function_means["selected_terminal_log10_loss"].mean()
                ),
                "function_balanced_mean_query_gain": float(
                    function_means["query_policy_gain"].mean()
                ),
                "median_query_gain": float(group["query_policy_gain"].median()),
            }
        )
    return pd.DataFrame(rows)


def _score_thresholds(scores: np.ndarray) -> np.ndarray:
    ordered = np.unique(np.asarray(scores, dtype=float))
    if ordered.size == 0 or not np.isfinite(ordered).all():
        raise ValueError("threshold scores must be finite and non-empty")
    if ordered.size == 1:
        return np.asarray([ordered[0] - 1e-12, ordered[0]], dtype=float)
    return np.concatenate(
        ([ordered[0] - 1e-12], (ordered[:-1] + ordered[1:]) / 2.0, [ordered[-1]])
    )


def _validate_query_frames(
    no_query: pd.DataFrame,
    query: pd.DataFrame,
) -> None:
    no_query_required = {
        "all_action_paths_completed",
        "all_action_paths_used_planned_FE",
    }
    no_query_missing = sorted(no_query_required.difference(no_query.columns))
    if no_query_missing:
        raise ValueError(f"no-query matrix is missing columns: {no_query_missing}")
    if not no_query["all_action_paths_completed"].astype(bool).all():
        raise ValueError("no-query labels require completed action paths")
    if not no_query["all_action_paths_used_planned_FE"].astype(bool).all():
        raise ValueError("no-query action paths must use their planned FE budget")
    keys = [*STATE_KEY, "candidate_action"]
    if no_query.duplicated(keys).any() or query.duplicated(keys).any():
        raise ValueError("query matrices contain duplicate state-action rows")
    no_keys = set(map(tuple, no_query[keys].itertuples(index=False, name=None)))
    query_keys = set(map(tuple, query[keys].itertuples(index=False, name=None)))
    if no_keys != query_keys:
        raise ValueError("no-query and query-adjusted matrices cover different state-actions")
    required = {
        *QUERY_DESCRIPTOR_FEATURES,
        "FE_query",
        "FE_action_optimization",
        "query_continuation_log10_action_loss",
        "all_query_adjusted_paths_completed",
        "all_action_paths_used_planned_FE",
    }
    missing = sorted(required.difference(query.columns))
    if missing:
        raise ValueError(f"query-adjusted matrix is missing columns: {missing}")
    if not (
        query["FE"].astype(int)
        + query["FE_query"].astype(int)
        + query["FE_action_optimization"].astype(int)
        == query["FE_total"].astype(int)
    ).all():
        raise ValueError("query-adjusted paths violate the equal-total-FE budget")
    if not query["all_query_adjusted_paths_completed"].astype(bool).all():
        raise ValueError("query-adjusted labels require completed Query and action paths")
    if not query["all_action_paths_used_planned_FE"].astype(bool).all():
        raise ValueError("query-adjusted action paths must use their planned FE budget")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the adaptive Query VOI policy and compare Query controls."
    )
    parser.add_argument("--train-config", default="configs/behavior_with_ela_train.yaml")
    parser.add_argument("--validation-config", default=None)
    parser.add_argument(
        "--phase1-model",
        default="results/behavior_with_ela/model/behavior_action_gain/models.joblib",
    )
    parser.add_argument(
        "--output",
        default="results/behavior_with_ela/model/adaptive_query",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = train_adaptive_query_policy(
        train_config_path=args.train_config,
        phase1_model_path=args.phase1_model,
        validation_config_path=args.validation_config,
        output_dir=args.output,
        overwrite=args.overwrite,
    )
    print(
        f"trained adaptive Query policy on {summary['control_state_rows']} states; "
        f"VOI threshold={summary['voi_threshold']:.12g}"
    )


if __name__ == "__main__":
    main()
