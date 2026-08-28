from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS
from behavior_with_ela.collection import shard_paths
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
    replay_first_trigger,
)
from behavior_with_ela.protocol import (
    CORE_PORTFOLIO,
    ExperimentConfig,
    load_experiment_config,
)
from trajectory.sampling import BUDGET_MILESTONE_RATIOS


RANDOM_SWITCH_STREAM = 2026082704
RANDOM_MATCHED_SWITCH_STREAM = 2026082707
ACTION_LOSS_MODEL_STREAM = 2026082705
TO_SWITCH_MODEL_STREAM = 2026082706
ALGORITHM_CODES = {"pso": 2, "cmaes": 3, "shade": 4}
FIXED_SWITCH_RATIO = 0.30


def compare_phase1_baselines(
    *,
    train_config_path: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    validation_config_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    train_config = load_experiment_config(train_config_path)
    train = read_action_datasets(train_config)
    model_root = Path(model_dir)
    bundle = joblib.load(model_root / "models.joblib")
    main_oof = pd.read_parquet(model_root / "oof_action_predictions.parquet")
    default_algorithm = str(bundle["default_algorithm"])
    practical_delta = float(bundle["practical_gain_delta"])
    train = apply_practical_gain_delta(train, practical_delta)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    expected = (
        output / "train_policy_runs.parquet",
        output / "policy_summary.parquet",
        output / "time_only_oof_predictions.parquet",
        output / "time_only_thresholds.parquet",
        output / "behavior_action_loss_oof_predictions.parquet",
        output / "behavior_action_loss_thresholds.parquet",
        output / "to_switch_oof_predictions.parquet",
        output / "to_switch_thresholds.parquet",
        output / "fixed_transition_mapping.parquet",
        output / "random_matched_calibration.parquet",
        output / "static_portfolio_summary.parquet",
        output / "baseline_models.joblib",
        output / "baseline_summary.json",
    )
    if any(path.exists() for path in expected) and not overwrite:
        raise FileExistsError(f"baseline outputs already exist; pass --overwrite: {output}")
    if overwrite:
        for path in expected:
            path.unlink(missing_ok=True)
        (output / "validation_policy_runs.parquet").unlink(missing_ok=True)

    main_train = replay_first_trigger(
        action_rows=train,
        action_predictions=main_oof,
        threshold=float(bundle["decision_threshold"]),
        practical_delta=practical_delta,
        default_algorithm=default_algorithm,
    )
    main_train["policy_name"] = "behavior_action_gain"

    time_oof = _family_oof_predictions(
        train,
        train_config,
        feature_columns=("bf_fe_ratio",),
    )
    time_thresholds, time_threshold, time_train = fit_first_trigger_threshold(
        action_rows=train,
        action_predictions=time_oof,
        practical_delta=practical_delta,
    )
    time_train = replay_first_trigger(
        action_rows=train,
        action_predictions=time_oof,
        threshold=time_threshold,
        practical_delta=practical_delta,
        default_algorithm=default_algorithm,
    )
    time_train["policy_name"] = "time_only_action_gain"

    action_loss_oof = _action_loss_family_oof_predictions(train, train_config)
    (
        action_loss_thresholds,
        action_loss_threshold,
        action_loss_train,
    ) = fit_first_trigger_threshold(
        action_rows=train,
        action_predictions=action_loss_oof,
        practical_delta=practical_delta,
    )
    action_loss_train = replay_first_trigger(
        action_rows=train,
        action_predictions=action_loss_oof,
        threshold=action_loss_threshold,
        practical_delta=practical_delta,
        default_algorithm=default_algorithm,
    )
    action_loss_train["policy_name"] = "behavior_action_loss_rf"

    to_switch_oof = _to_switch_family_oof_predictions(
        train,
        train_config,
        practical_delta=practical_delta,
    )
    to_switch_thresholds, to_switch_threshold, to_switch_train = (
        fit_first_trigger_threshold(
            action_rows=train,
            action_predictions=to_switch_oof,
            practical_delta=practical_delta,
        )
    )
    to_switch_train = replay_first_trigger(
        action_rows=train,
        action_predictions=to_switch_oof,
        threshold=to_switch_threshold,
        practical_delta=practical_delta,
        default_algorithm=default_algorithm,
    )
    to_switch_train["policy_name"] = "to_switch_style_rf"

    random_matched_calibration = _fit_random_matched_calibration(
        main_train,
        fe_total=train_config.fe_total,
    )
    time_models = _fit_models(
        train,
        train_config,
        fold_number=20_000,
        feature_columns=("bf_fe_ratio",),
    )
    action_loss_model = _fit_action_loss_model(
        train,
        train_config,
        fold_number=20_001,
    )
    to_switch_model = _fit_to_switch_model(
        train,
        train_config,
        fold_number=20_002,
        practical_delta=practical_delta,
    )

    fixed_mapping = fit_fixed_transition_mapping(train, train_config)
    train_runs = pd.concat(
        [
            main_train,
            time_train,
            action_loss_train,
            to_switch_train,
            _continue_policy(train, default_algorithm, practical_delta),
            _random_switch_policy(train, default_algorithm, practical_delta),
            _random_matched_switch_policy(
                train,
                default_algorithm,
                practical_delta,
                calibration=random_matched_calibration,
            ),
            _fixed_switch_policy(
                train,
                fixed_mapping,
                default_algorithm,
                practical_delta,
            ),
            _best_observed_policy(train, default_algorithm, practical_delta),
        ],
        ignore_index=True,
    )
    train_runs["evaluation_split"] = "bbob_train_oof_or_reference"
    all_policy_runs = [train_runs]

    if validation_config_path is not None:
        validation_config = load_experiment_config(validation_config_path)
        validation = read_action_datasets(validation_config)
        validation = apply_practical_gain_delta(validation, practical_delta)
        main_validation = predict_action_rows(bundle["models"], validation)
        main_validation_runs = replay_first_trigger(
            action_rows=validation,
            action_predictions=main_validation,
            threshold=float(bundle["decision_threshold"]),
            practical_delta=practical_delta,
            default_algorithm=default_algorithm,
        )
        main_validation_runs["policy_name"] = "behavior_action_gain"
        time_validation_predictions = predict_action_rows(
            time_models,
            validation,
            feature_columns=("bf_fe_ratio",),
        )
        time_validation_runs = replay_first_trigger(
            action_rows=validation,
            action_predictions=time_validation_predictions,
            threshold=time_threshold,
            practical_delta=practical_delta,
            default_algorithm=default_algorithm,
        )
        time_validation_runs["policy_name"] = "time_only_action_gain"
        action_loss_validation = _predict_action_loss_rows(
            model=action_loss_model,
            action_rows=validation,
            practical_delta=practical_delta,
        )
        action_loss_validation_runs = replay_first_trigger(
            action_rows=validation,
            action_predictions=action_loss_validation,
            threshold=action_loss_threshold,
            practical_delta=practical_delta,
            default_algorithm=default_algorithm,
        )
        action_loss_validation_runs["policy_name"] = "behavior_action_loss_rf"
        to_switch_validation = _predict_to_switch_rows(
            gate_model=to_switch_model,
            action_loss_model=action_loss_model,
            action_rows=validation,
            practical_delta=practical_delta,
        )
        to_switch_validation_runs = replay_first_trigger(
            action_rows=validation,
            action_predictions=to_switch_validation,
            threshold=to_switch_threshold,
            practical_delta=practical_delta,
            default_algorithm=default_algorithm,
        )
        to_switch_validation_runs["policy_name"] = "to_switch_style_rf"
        validation_runs = pd.concat(
            [
                main_validation_runs,
                time_validation_runs,
                action_loss_validation_runs,
                to_switch_validation_runs,
                _continue_policy(validation, default_algorithm, practical_delta),
                _random_switch_policy(validation, default_algorithm, practical_delta),
                _random_matched_switch_policy(
                    validation,
                    default_algorithm,
                    practical_delta,
                    calibration=random_matched_calibration,
                ),
                _fixed_switch_policy(
                    validation,
                    fixed_mapping,
                    default_algorithm,
                    practical_delta,
                ),
                _best_observed_policy(validation, default_algorithm, practical_delta),
            ],
            ignore_index=True,
        )
        validation_runs["evaluation_split"] = "bbob_validation"
        validation_runs.to_parquet(
            output / "validation_policy_runs.parquet", index=False
        )
        all_policy_runs.append(validation_runs)

    policy_runs = pd.concat(all_policy_runs, ignore_index=True)
    summary = summarize_policy_runs(policy_runs)
    static_summary = static_portfolio_summary(
        train_config,
        default_algorithm=default_algorithm,
        evaluation_split="train",
    )
    if validation_config_path is not None:
        static_summary = pd.concat(
            [
                static_summary,
                static_portfolio_summary(
                    load_experiment_config(validation_config_path),
                    default_algorithm=default_algorithm,
                    evaluation_split="validation",
                ),
            ],
            ignore_index=True,
        )

    train_runs.to_parquet(output / "train_policy_runs.parquet", index=False)
    summary.to_parquet(output / "policy_summary.parquet", index=False)
    time_oof.to_parquet(output / "time_only_oof_predictions.parquet", index=False)
    time_thresholds.to_parquet(output / "time_only_thresholds.parquet", index=False)
    action_loss_oof.to_parquet(
        output / "behavior_action_loss_oof_predictions.parquet", index=False
    )
    action_loss_thresholds.to_parquet(
        output / "behavior_action_loss_thresholds.parquet", index=False
    )
    to_switch_oof.to_parquet(
        output / "to_switch_oof_predictions.parquet", index=False
    )
    to_switch_thresholds.to_parquet(
        output / "to_switch_thresholds.parquet", index=False
    )
    fixed_mapping.to_parquet(output / "fixed_transition_mapping.parquet", index=False)
    random_matched_calibration.to_parquet(
        output / "random_matched_calibration.parquet", index=False
    )
    joblib.dump(
        {
            "portfolio": train_config.algorithms,
            "dimension": train_config.dimension,
            "FE_total": train_config.fe_total,
            "population_size": train_config.population_size,
            "sampling_protocol": train_config.sampling_protocol,
            "boundary_handling": train_config.boundary_handling,
            "practical_gain_delta": practical_delta,
            "default_algorithm": default_algorithm,
            "time_only_models": time_models,
            "time_only_threshold": float(time_threshold),
            "behavior_action_loss_model": action_loss_model,
            "behavior_action_loss_threshold": float(action_loss_threshold),
            "to_switch_model": to_switch_model,
            "to_switch_threshold": float(to_switch_threshold),
            "fixed_transition_mapping": fixed_mapping,
            "random_matched_calibration": random_matched_calibration,
        },
        output / "baseline_models.joblib",
    )
    static_summary.to_parquet(output / "static_portfolio_summary.parquet", index=False)
    result = {
        "train_policy_rows": int(len(train_runs)),
        "policy_summary_rows": int(len(summary)),
        "time_only_threshold": float(time_threshold),
        "behavior_action_loss_threshold": float(action_loss_threshold),
        "to_switch_threshold": float(to_switch_threshold),
        "default_algorithm": default_algorithm,
        "validation_included": validation_config_path is not None,
    }
    with (output / "baseline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def _action_loss_family_oof_predictions(
    train: pd.DataFrame,
    config: ExperimentConfig,
    *,
    feature_columns: tuple[str, ...] = SELECTOR_BEHAVIOR_FEATURE_COLUMNS,
) -> pd.DataFrame:
    bbob = train.loc[train["suite"].astype(str).eq("bbob")].copy()
    families = tuple(sorted(set(bbob["family"].astype(str))))
    if len(families) < 2:
        raise ValueError("action-loss OOF requires at least two BBOB landscape families")
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
        fold_eval = bbob.loc[
            bbob["family"].astype(str).eq(heldout_family)
        ].copy()
        model = _fit_action_loss_model(
            fold_train,
            config,
            fold_number=fold_number,
            feature_columns=feature_columns,
        )
        predicted = _predict_action_loss_rows(
            model=model,
            action_rows=fold_eval,
            practical_delta=float(
                fold_eval["practical_gain_delta"].iloc[0]
                if "practical_gain_delta" in fold_eval.columns
                else config.domain_gain_delta
            ),
            feature_columns=feature_columns,
        )
        predicted["oof_fold"] = int(fold_number)
        predicted["heldout_family"] = heldout_family
        predictions.append(predicted)
    result = pd.concat(predictions, ignore_index=True)
    expected = bbob.loc[~bbob["action_equals_prefix"].astype(bool)]
    keys = [*STATE_KEY, "candidate_action"]
    if len(result) != len(expected):
        raise RuntimeError(
            "action-loss OOF prediction coverage differs from BBOB switch rows"
        )
    if result.duplicated(keys).any():
        raise RuntimeError("action-loss OOF predictions contain duplicate rows")
    return result


def _to_switch_family_oof_predictions(
    train: pd.DataFrame,
    config: ExperimentConfig,
    *,
    practical_delta: float,
) -> pd.DataFrame:
    bbob = train.loc[train["suite"].astype(str).eq("bbob")].copy()
    families = tuple(sorted(set(bbob["family"].astype(str))))
    if len(families) < 2:
        raise ValueError("to-switch OOF requires at least two BBOB landscape families")
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
        fold_eval = bbob.loc[
            bbob["family"].astype(str).eq(heldout_family)
        ].copy()
        gate_model = _fit_to_switch_model(
            fold_train,
            config,
            fold_number=fold_number,
            practical_delta=practical_delta,
        )
        action_loss_model = _fit_action_loss_model(
            fold_train,
            config,
            fold_number=10_000 + fold_number,
        )
        predicted = _predict_to_switch_rows(
            gate_model=gate_model,
            action_loss_model=action_loss_model,
            action_rows=fold_eval,
            practical_delta=practical_delta,
        )
        predicted["oof_fold"] = int(fold_number)
        predicted["heldout_family"] = heldout_family
        predictions.append(predicted)
    result = pd.concat(predictions, ignore_index=True)
    expected = bbob.loc[~bbob["action_equals_prefix"].astype(bool)]
    keys = [*STATE_KEY, "candidate_action"]
    if len(result) != len(expected):
        raise RuntimeError("to-switch OOF prediction coverage differs from switch rows")
    if result.duplicated(keys).any():
        raise RuntimeError("to-switch OOF predictions contain duplicate rows")
    return result


def _fit_to_switch_model(
    rows: pd.DataFrame,
    config: ExperimentConfig,
    *,
    fold_number: int,
    practical_delta: float,
) -> Pipeline:
    states, _ = _action_loss_state_matrix(rows, config.algorithms)
    switch_gain = (
        rows.loc[~rows["action_equals_prefix"].astype(bool)]
        .groupby(list(STATE_KEY), sort=False)["action_gain_vs_continue"]
        .max()
        .rename("best_switch_gain")
        .reset_index()
    )
    states = states.merge(
        switch_gain,
        on=list(STATE_KEY),
        how="inner",
        validate="one_to_one",
    )
    labels = states["best_switch_gain"].to_numpy(dtype=float) > float(
        practical_delta
    )
    random_state = int(
        np.random.SeedSequence(
            [TO_SWITCH_MODEL_STREAM, int(fold_number), config.dimension]
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
    weights = _block_function_weights(states)
    model.fit(
        states[list(SELECTOR_BEHAVIOR_FEATURE_COLUMNS)],
        labels,
        classifier__sample_weight=weights,
    )
    return model


def _predict_to_switch_rows(
    *,
    gate_model: Pipeline,
    action_loss_model: Pipeline,
    action_rows: pd.DataFrame,
    practical_delta: float,
) -> pd.DataFrame:
    states, _ = _action_loss_state_matrix(action_rows, CORE_PORTFOLIO)
    probability_matrix = np.asarray(
        gate_model.predict_proba(states[list(SELECTOR_BEHAVIOR_FEATURE_COLUMNS)]),
        dtype=float,
    )
    classes = np.asarray(gate_model.classes_)
    matches = np.flatnonzero(classes.astype(bool))
    if len(matches) > 1:
        raise RuntimeError("to-switch classifier exposes duplicate True classes")
    gate_probability = (
        np.zeros(len(states), dtype=float)
        if len(matches) == 0
        else probability_matrix[:, int(matches[0])]
    )
    gate_class = gate_model.predict(
        states[list(SELECTOR_BEHAVIOR_FEATURE_COLUMNS)]
    ).astype(bool)
    state_scores = states[list(STATE_KEY)].copy()
    state_scores["predicted_switch_probability"] = gate_probability
    state_scores["predicted_switch_class"] = gate_class

    action_predictions = _predict_action_loss_rows(
        model=action_loss_model,
        action_rows=action_rows,
        practical_delta=practical_delta,
    ).merge(
        state_scores,
        on=list(STATE_KEY),
        how="inner",
        validate="many_to_one",
    )
    portfolio_order = {
        algorithm: index for index, algorithm in enumerate(CORE_PORTFOLIO)
    }
    action_predictions["candidate_order"] = (
        action_predictions["candidate_action"].astype(str).map(portfolio_order).astype(int)
    )
    ranked = action_predictions.sort_values(
        [
            *STATE_KEY,
            "predicted_candidate_log10_loss",
            "candidate_order",
        ],
        kind="mergesort",
    )
    selected_keys = set(
        tuple(row)
        for row in ranked.groupby(list(STATE_KEY), sort=False)
        .head(1)[[*STATE_KEY, "candidate_action"]]
        .itertuples(index=False, name=None)
    )
    row_keys = [
        tuple(row)
        for row in action_predictions[[*STATE_KEY, "candidate_action"]].itertuples(
            index=False,
            name=None,
        )
    ]
    selected = np.asarray([key in selected_keys for key in row_keys], dtype=bool)
    action_predictions["candidate_ranked_first"] = selected
    action_predictions["predicted_improve_probability"] = np.where(
        selected,
        action_predictions["predicted_switch_probability"].to_numpy(dtype=float),
        -1.0,
    )
    action_predictions["predicted_action_class"] = np.where(
        selected & action_predictions["predicted_switch_class"].to_numpy(dtype=bool),
        "improve",
        "equivalent",
    )
    action_predictions["action_score_semantics"] = (
        "switch_probability_with_action_loss_target"
    )
    return action_predictions


def _fit_action_loss_model(
    rows: pd.DataFrame,
    config: ExperimentConfig,
    *,
    fold_number: int,
    feature_columns: tuple[str, ...] = SELECTOR_BEHAVIOR_FEATURE_COLUMNS,
) -> Pipeline:
    states, targets = _action_loss_state_matrix(rows, config.algorithms)
    random_state = int(
        np.random.SeedSequence(
            [ACTION_LOSS_MODEL_STREAM, int(fold_number), config.dimension]
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
    weights = _block_function_weights(states)
    model.fit(
        states[list(feature_columns)],
        targets,
        regressor__sample_weight=weights,
    )
    return model


def _predict_action_loss_rows(
    *,
    model: Pipeline,
    action_rows: pd.DataFrame,
    practical_delta: float,
    feature_columns: tuple[str, ...] = SELECTOR_BEHAVIOR_FEATURE_COLUMNS,
) -> pd.DataFrame:
    states, _ = _action_loss_state_matrix(action_rows, CORE_PORTFOLIO)
    predicted = np.asarray(
        model.predict(states[list(feature_columns)]),
        dtype=float,
    )
    if predicted.shape != (len(states), len(CORE_PORTFOLIO)):
        raise RuntimeError("action-loss RF returned an unexpected prediction shape")
    predicted_frame = states[list(STATE_KEY)].copy()
    for algorithm_index, algorithm in enumerate(CORE_PORTFOLIO):
        predicted_frame[f"predicted_log10_loss_{algorithm}"] = predicted[
            :, algorithm_index
        ]
    output: list[pd.DataFrame] = []
    for algorithm in CORE_PORTFOLIO:
        candidate = action_rows.loc[
            action_rows["candidate_action"].astype(str).eq(algorithm)
            & ~action_rows["action_equals_prefix"].astype(bool)
        ].copy()
        if candidate.empty:
            continue
        candidate = candidate.merge(
            predicted_frame,
            on=list(STATE_KEY),
            how="inner",
            validate="many_to_one",
        )
        candidate["predicted_candidate_log10_loss"] = candidate[
            f"predicted_log10_loss_{algorithm}"
        ].to_numpy(dtype=float)
        predicted_continue = np.asarray(
            [
                row[f"predicted_log10_loss_{row['prefix_algorithm']}"]
                for _, row in candidate.iterrows()
            ],
            dtype=float,
        )
        score = (
            predicted_continue
            - candidate["predicted_candidate_log10_loss"].to_numpy(dtype=float)
        )
        candidate["predicted_continue_log10_loss"] = predicted_continue
        candidate["predicted_action_gain"] = score
        candidate["predicted_improve_probability"] = score
        candidate["predicted_action_class"] = np.select(
            [score > practical_delta, score < -practical_delta],
            ["improve", "degrade"],
            default="equivalent",
        )
        candidate["action_score_semantics"] = "predicted_log10_loss_advantage"
        output.append(candidate)
    if not output:
        raise ValueError("action-loss RF produced no switch-action predictions")
    return pd.concat(output, ignore_index=True)


def _action_loss_state_matrix(
    rows: pd.DataFrame,
    portfolio: tuple[str, ...],
) -> tuple[pd.DataFrame, np.ndarray]:
    order = {algorithm: index for index, algorithm in enumerate(portfolio)}
    ordered = rows.copy()
    ordered["candidate_order"] = (
        ordered["candidate_action"].astype(str).map(order)
    )
    if ordered["candidate_order"].isna().any():
        raise ValueError("action-loss data contains an action outside the portfolio")
    ordered = ordered.sort_values(
        [*STATE_KEY, "candidate_order"],
        kind="mergesort",
    )
    counts = ordered.groupby(list(STATE_KEY), sort=False)["candidate_action"].size()
    if not counts.eq(len(portfolio)).all():
        raise ValueError("each action-loss state must contain the complete portfolio")
    states = ordered.groupby(list(STATE_KEY), sort=False).head(1).reset_index(drop=True)
    targets = ordered["log10_action_loss"].to_numpy(dtype=float).reshape(
        len(states), len(portfolio)
    )
    if not np.isfinite(targets).all():
        raise ValueError("action-loss targets must be finite")
    return states, targets


def fit_fixed_transition_mapping(
    train: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    candidates = train.loc[
        train["is_budget_milestone"].astype(bool)
        & np.isclose(
            train["budget_milestone_ratio"].to_numpy(dtype=float),
            FIXED_SWITCH_RATIO,
            rtol=0.0,
            atol=1e-12,
        )
        & ~train["action_equals_prefix"].astype(bool)
    ].copy()
    if candidates.empty:
        raise ValueError("fixed-time baseline found no 0.30B milestone action rows")
    function_means = (
        candidates.groupby(
            ["suite", "cv_group_id", "prefix_algorithm", "candidate_action"],
            as_index=False,
        )["action_gain_vs_continue"]
        .mean()
    )
    suite_means = (
        function_means.groupby(
            ["suite", "prefix_algorithm", "candidate_action"],
            as_index=False,
        )["action_gain_vs_continue"]
        .mean()
    )
    block_means = (
        suite_means.groupby(
            ["prefix_algorithm", "candidate_action"], as_index=False
        )["action_gain_vs_continue"]
        .mean()
        .rename(columns={"action_gain_vs_continue": "train_mean_gain"})
    )
    order = {algorithm: index for index, algorithm in enumerate(config.algorithms)}
    block_means["candidate_order"] = (
        block_means["candidate_action"].map(order).astype(int)
    )
    selected_rows: list[dict[str, Any]] = []
    for prefix in config.algorithms:
        choices = block_means.loc[
            block_means["prefix_algorithm"].astype(str).eq(prefix)
        ].sort_values(
            ["train_mean_gain", "candidate_order"],
            ascending=[False, True],
            kind="mergesort",
        )
        if choices.empty or float(choices.iloc[0]["train_mean_gain"]) <= 0.0:
            selected_rows.append(
                {
                    "prefix_algorithm": prefix,
                    "selected_algorithm": prefix,
                    "train_mean_gain": 0.0,
                    "switch_ratio": FIXED_SWITCH_RATIO,
                }
            )
        else:
            selected_rows.append(
                {
                    "prefix_algorithm": prefix,
                    "selected_algorithm": str(choices.iloc[0]["candidate_action"]),
                    "train_mean_gain": float(choices.iloc[0]["train_mean_gain"]),
                    "switch_ratio": FIXED_SWITCH_RATIO,
                }
            )
    return pd.DataFrame(selected_rows)


def _continue_policy(
    actions: pd.DataFrame,
    default_algorithm: str,
    practical_delta: float,
) -> pd.DataFrame:
    reference = _run_reference(actions)
    rows = []
    for _, item in reference.iterrows():
        rows.append(
            _policy_row(
                reference=item,
                policy_name="continue_current",
                selected_algorithm=str(item["prefix_algorithm"]),
                selected_gain=0.0,
                selected_log10_loss=float(item["continue_log10_loss"]),
                selected_fe=None,
                selected_opportunity=None,
                default_algorithm=default_algorithm,
                practical_delta=practical_delta,
            )
        )
    return pd.DataFrame(rows)


def _best_observed_policy(
    actions: pd.DataFrame,
    default_algorithm: str,
    practical_delta: float,
) -> pd.DataFrame:
    bbob = actions.loc[actions["suite"].astype(str).eq("bbob")].copy()
    reference = _run_reference(actions).set_index(list(RUN_KEY), drop=False)
    rows = []
    order = {algorithm: index for index, algorithm in enumerate(CORE_PORTFOLIO)}
    for run_key, run in bbob.groupby(list(RUN_KEY), sort=False):
        ref = reference.loc[run_key]
        candidates = run.loc[~run["action_equals_prefix"].astype(bool)].copy()
        candidates["candidate_order"] = (
            candidates["candidate_action"].astype(str).map(order).astype(int)
        )
        candidates = candidates.sort_values(
            [
                "action_gain_vs_continue",
                "FE",
                "decision_opportunity_index",
                "candidate_order",
            ],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
        best = candidates.iloc[0]
        if float(best["action_gain_vs_continue"]) <= 0.0:
            selected_algorithm = str(ref["prefix_algorithm"])
            selected_gain = 0.0
            selected_loss = float(ref["continue_log10_loss"])
            selected_fe = None
            selected_opportunity = None
        else:
            selected_algorithm = str(best["candidate_action"])
            selected_gain = float(best["action_gain_vs_continue"])
            selected_loss = float(best["log10_action_loss"])
            selected_fe = int(best["FE"])
            selected_opportunity = int(best["decision_opportunity_index"])
        rows.append(
            _policy_row(
                reference=ref,
                policy_name="best_observed_one_switch",
                selected_algorithm=selected_algorithm,
                selected_gain=selected_gain,
                selected_log10_loss=selected_loss,
                selected_fe=selected_fe,
                selected_opportunity=selected_opportunity,
                default_algorithm=default_algorithm,
                practical_delta=practical_delta,
            )
        )
    return pd.DataFrame(rows)


def _random_switch_policy(
    actions: pd.DataFrame,
    default_algorithm: str,
    practical_delta: float,
) -> pd.DataFrame:
    bbob = actions.loc[
        actions["suite"].astype(str).eq("bbob")
        & ~actions["action_equals_prefix"].astype(bool)
    ].copy()
    reference = _run_reference(actions).set_index(list(RUN_KEY), drop=False)
    grouped_runs = list(bbob.groupby(list(RUN_KEY), sort=False))
    rows = []
    for run_key, run in grouped_runs:
        first = run.iloc[0]
        function_number = int(str(first["function_id"]).split("f")[-1])
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    int(first["seed"]),
                    RANDOM_SWITCH_STREAM,
                    function_number,
                    ALGORITHM_CODES[str(first["prefix_algorithm"])],
                ]
            )
        )
        target_ratio = float(
            BUDGET_MILESTONE_RATIOS[
                int(rng.integers(0, len(BUDGET_MILESTONE_RATIOS)))
            ]
        )
        state_rows = run.loc[
            run["is_budget_milestone"].astype(bool)
            & np.isclose(
                run["budget_milestone_ratio"].to_numpy(dtype=float),
                target_ratio,
                rtol=0.0,
                atol=1e-12,
            )
        ].sort_values("candidate_action", kind="mergesort")
        if len(state_rows) != len(CORE_PORTFOLIO) - 1:
            raise RuntimeError("random baseline requires two switch actions at its milestone")
        selected = state_rows.iloc[int(rng.integers(0, len(state_rows)))]
        ref = reference.loc[run_key]
        rows.append(
            _policy_row(
                reference=ref,
                policy_name="random_one_switch",
                selected_algorithm=str(selected["candidate_action"]),
                selected_gain=float(selected["action_gain_vs_continue"]),
                selected_log10_loss=float(selected["log10_action_loss"]),
                selected_fe=int(selected["FE"]),
                selected_opportunity=int(selected["decision_opportunity_index"]),
                default_algorithm=default_algorithm,
                practical_delta=practical_delta,
            )
        )
    return pd.DataFrame(rows)


def _fit_random_matched_calibration(
    main_oof_runs: pd.DataFrame,
    *,
    fe_total: int,
) -> pd.DataFrame:
    if main_oof_runs.empty:
        raise ValueError("random matched-rate calibration requires OOF policy runs")
    call_rate = float(main_oof_runs["switch_triggered"].astype(bool).mean())
    selected = main_oof_runs.loc[
        main_oof_runs["switch_triggered"].astype(bool), "selected_FE"
    ].dropna()
    trigger_ratios = tuple(
        float(value) / float(fe_total) for value in selected.to_numpy(dtype=float)
    )
    if call_rate > 0.0 and not trigger_ratios:
        raise RuntimeError("positive OOF switch rate requires observed trigger FE values")
    rows = [
        {
            "calibration_source": "bbob_train_oof_behavior_policy",
            "matched_switch_rate": call_rate,
            "trigger_ratio": None,
            "trigger_count": len(trigger_ratios),
        }
    ]
    rows.extend(
        {
            "calibration_source": "bbob_train_oof_behavior_policy",
            "matched_switch_rate": call_rate,
            "trigger_ratio": ratio,
            "trigger_count": len(trigger_ratios),
        }
        for ratio in trigger_ratios
    )
    return pd.DataFrame(rows)


def _random_matched_switch_policy(
    actions: pd.DataFrame,
    default_algorithm: str,
    practical_delta: float,
    *,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    header = calibration.iloc[0]
    call_rate = float(header["matched_switch_rate"])
    trigger_ratios = calibration["trigger_ratio"].dropna().to_numpy(dtype=float)
    if not 0.0 <= call_rate <= 1.0:
        raise ValueError("matched random switch rate must be in [0, 1]")
    bbob = actions.loc[
        actions["suite"].astype(str).eq("bbob")
        & ~actions["action_equals_prefix"].astype(bool)
    ].copy()
    reference = _run_reference(actions).set_index(list(RUN_KEY), drop=False)
    rows = []
    for run_key, run in bbob.groupby(list(RUN_KEY), sort=False):
        first = run.iloc[0]
        function_number = int(str(first["function_id"]).split("f")[-1])
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    int(first["seed"]),
                    RANDOM_MATCHED_SWITCH_STREAM,
                    function_number,
                    ALGORITHM_CODES[str(first["prefix_algorithm"])],
                ]
            )
        )
        ref = reference.loc[run_key]
        should_switch = bool(rng.random() < call_rate)
        if not should_switch:
            rows.append(
                _policy_row(
                    reference=ref,
                    policy_name="random_matched_switch_rate",
                    selected_algorithm=str(ref["prefix_algorithm"]),
                    selected_gain=0.0,
                    selected_log10_loss=float(ref["continue_log10_loss"]),
                    selected_fe=None,
                    selected_opportunity=None,
                    default_algorithm=default_algorithm,
                    practical_delta=practical_delta,
                )
            )
            continue
        target_ratio = float(
            trigger_ratios[int(rng.integers(0, len(trigger_ratios)))]
        )
        states = (
            run.sort_values(["FE", "decision_opportunity_index"], kind="mergesort")
            .groupby(["FE", "decision_opportunity_index"], sort=False)
            .head(1)
            .copy()
        )
        states["ratio_distance"] = np.abs(
            states["FE_ratio"].to_numpy(dtype=float) - target_ratio
        )
        selected_state = states.sort_values(
            ["ratio_distance", "FE", "decision_opportunity_index"],
            kind="mergesort",
        ).iloc[0]
        candidates = run.loc[
            run["FE"].astype(int).eq(int(selected_state["FE"]))
            & run["decision_opportunity_index"].astype(int).eq(
                int(selected_state["decision_opportunity_index"])
            )
        ].sort_values("candidate_action", kind="mergesort")
        selected = candidates.iloc[int(rng.integers(0, len(candidates)))]
        rows.append(
            _policy_row(
                reference=ref,
                policy_name="random_matched_switch_rate",
                selected_algorithm=str(selected["candidate_action"]),
                selected_gain=float(selected["action_gain_vs_continue"]),
                selected_log10_loss=float(selected["log10_action_loss"]),
                selected_fe=int(selected["FE"]),
                selected_opportunity=int(selected["decision_opportunity_index"]),
                default_algorithm=default_algorithm,
                practical_delta=practical_delta,
            )
        )
    return pd.DataFrame(rows)


def _fixed_switch_policy(
    actions: pd.DataFrame,
    mapping: pd.DataFrame,
    default_algorithm: str,
    practical_delta: float,
) -> pd.DataFrame:
    bbob = actions.loc[actions["suite"].astype(str).eq("bbob")].copy()
    reference = _run_reference(actions).set_index(list(RUN_KEY), drop=False)
    target_by_prefix = dict(
        zip(mapping["prefix_algorithm"], mapping["selected_algorithm"], strict=True)
    )
    rows = []
    for run_key, run in bbob.groupby(list(RUN_KEY), sort=False):
        ref = reference.loc[run_key]
        prefix = str(ref["prefix_algorithm"])
        target = str(target_by_prefix[prefix])
        if target == prefix:
            selected_gain = 0.0
            selected_loss = float(ref["continue_log10_loss"])
            selected_fe = None
            selected_opportunity = None
        else:
            candidates = run.loc[
                run["candidate_action"].astype(str).eq(target)
                & run["is_budget_milestone"].astype(bool)
                & np.isclose(
                    run["budget_milestone_ratio"].to_numpy(dtype=float),
                    FIXED_SWITCH_RATIO,
                    rtol=0.0,
                    atol=1e-12,
                )
            ]
            if len(candidates) != 1:
                raise RuntimeError("fixed-time baseline requires one target row at 0.30B")
            selected = candidates.iloc[0]
            selected_gain = float(selected["action_gain_vs_continue"])
            selected_loss = float(selected["log10_action_loss"])
            selected_fe = int(selected["FE"])
            selected_opportunity = int(selected["decision_opportunity_index"])
        rows.append(
            _policy_row(
                reference=ref,
                policy_name="fixed_030_transition",
                selected_algorithm=target,
                selected_gain=selected_gain,
                selected_log10_loss=selected_loss,
                selected_fe=selected_fe,
                selected_opportunity=selected_opportunity,
                default_algorithm=default_algorithm,
                practical_delta=practical_delta,
            )
        )
    return pd.DataFrame(rows)


def _run_reference(actions: pd.DataFrame) -> pd.DataFrame:
    bbob = actions.loc[actions["suite"].astype(str).eq("bbob")].copy()
    rows = []
    for run_key, run in bbob.groupby(list(RUN_KEY), sort=False):
        ordered = run.sort_values(
            ["FE", "decision_opportunity_index"], kind="mergesort"
        )
        continue_rows = ordered.loc[ordered["action_equals_prefix"].astype(bool)]
        if continue_rows.empty:
            raise RuntimeError("run is missing continue-current action outcomes")
        first = ordered.iloc[0]
        rows.append(
            {
                "problem_id": str(run_key[0]),
                "prefix_algorithm": str(run_key[1]),
                "seed": int(run_key[2]),
                "function_id": str(first["function_id"]),
                "family": str(first["family"]),
                "cv_group_id": str(first["cv_group_id"]),
                "continue_log10_loss": float(
                    continue_rows.iloc[0]["log10_action_loss"]
                ),
                "best_observed_one_switch_gain": max(
                    0.0,
                    float(
                        ordered.loc[
                            ~ordered["action_equals_prefix"].astype(bool),
                            "action_gain_vs_continue",
                        ].max()
                    ),
                ),
            }
        )
    return pd.DataFrame(rows)


def _policy_row(
    *,
    reference: pd.Series,
    policy_name: str,
    selected_algorithm: str,
    selected_gain: float,
    selected_log10_loss: float,
    selected_fe: int | None,
    selected_opportunity: int | None,
    default_algorithm: str,
    practical_delta: float,
) -> dict[str, Any]:
    prefix = str(reference["prefix_algorithm"])
    same_prefix = selected_algorithm == prefix
    best_gain = float(reference["best_observed_one_switch_gain"])
    regret = float(best_gain - selected_gain)
    regret_scale = max(best_gain, float(practical_delta), 1e-12)
    return {
        "problem_id": str(reference["problem_id"]),
        "prefix_algorithm": prefix,
        "seed": int(reference["seed"]),
        "function_id": str(reference["function_id"]),
        "family": str(reference["family"]),
        "cv_group_id": str(reference["cv_group_id"]),
        "policy_name": policy_name,
        "switch_triggered": bool(not same_prefix),
        "selected_algorithm": selected_algorithm,
        "selected_FE": selected_fe,
        "selected_decision_opportunity_index": selected_opportunity,
        "selected_action_gain": float(selected_gain),
        "selected_terminal_log10_loss": float(selected_log10_loss),
        "continue_terminal_log10_loss": float(reference["continue_log10_loss"]),
        "best_observed_one_switch_gain": best_gain,
        "one_switch_regret": regret,
        "normalized_one_switch_regret": float(regret / regret_scale),
        "normalized_regret_denominator": regret_scale,
        "acceptable_policy": bool(best_gain - selected_gain <= practical_delta),
        "selected_equals_default": bool(selected_algorithm == default_algorithm),
        "selected_equals_prefix": bool(same_prefix),
        "handoff_required": bool(not same_prefix),
        "handoff_type": (
            "native_optimizer_state"
            if same_prefix
            else "population_transfer_initialization"
        ),
    }


def summarize_policy_runs(rows: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (split, policy), group in rows.groupby(
        ["evaluation_split", "policy_name"], sort=False
    ):
        function_means = group.groupby("cv_group_id")[
            [
                "selected_action_gain",
                "selected_terminal_log10_loss",
                "one_switch_regret",
                "normalized_one_switch_regret",
            ]
        ].mean()
        summaries.append(
            {
                "evaluation_split": split,
                "policy_name": policy,
                "run_count": int(len(group)),
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
                "median_gain": float(group["selected_action_gain"].median()),
                "median_regret": float(group["one_switch_regret"].median()),
                "median_normalized_regret": float(
                    group["normalized_one_switch_regret"].median()
                ),
                "switch_rate": float(group["switch_triggered"].mean()),
                "acceptable_policy_rate": float(group["acceptable_policy"].mean()),
            }
        )
    return pd.DataFrame(summaries)


def static_portfolio_summary(
    config: ExperimentConfig,
    *,
    default_algorithm: str,
    evaluation_split: str,
) -> pd.DataFrame:
    frames = []
    for suite in config.suites:
        if suite.suite != "bbob":
            continue
        for function in suite.functions:
            _, path, _ = shard_paths(config, suite, function)
            frame = pd.read_parquet(path)
            frames.append(frame.loc[frame["algorithm"].astype(str).isin(config.algorithms)])
    final = pd.concat(frames, ignore_index=True)
    sbs = final.loc[final["algorithm"].astype(str).eq(default_algorithm)].copy()
    vbs = (
        final.sort_values(["problem_id", "seed", "log10_gap", "algorithm"], kind="mergesort")
        .groupby(["problem_id", "seed"], sort=False)
        .head(1)
    )
    rows = []
    for name, frame in (("sbs", sbs), ("vbs", vbs)):
        rows.append(
            {
                "evaluation_split": evaluation_split,
                "portfolio_reference": name,
                "run_count": int(len(frame)),
                "function_balanced_mean_log10_gap": float(
                    frame.groupby("function_id")["log10_gap"].mean().mean()
                ),
                "success_rate": float(frame["success"].mean()),
                "default_algorithm": default_algorithm,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Phase 1 Behavior action selection against predefined baselines."
    )
    parser.add_argument("--train-config", default="configs/behavior_with_ela_train.yaml")
    parser.add_argument("--validation-config", default=None)
    parser.add_argument(
        "--model-dir",
        default="results/behavior_with_ela/model/behavior_action_gain",
    )
    parser.add_argument(
        "--output",
        default="results/behavior_with_ela/baselines/phase1",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = compare_phase1_baselines(
        train_config_path=args.train_config,
        validation_config_path=args.validation_config,
        model_dir=args.model_dir,
        output_dir=args.output,
        overwrite=args.overwrite,
    )
    print(
        f"wrote {summary['policy_summary_rows']} policy summaries; "
        f"time-only threshold={summary['time_only_threshold']:.12g}"
    )


if __name__ == "__main__":
    main()
