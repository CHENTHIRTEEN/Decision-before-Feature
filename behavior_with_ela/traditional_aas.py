from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from behavior_with_ela.action_dataset import (
    _TrackedActionObjective,
    _component_functions,
)
from behavior_with_ela.model import _block_function_weights, _ma_overlaps_heldout
from behavior_with_ela.protocol import (
    ExperimentConfig,
    SuiteConfig,
    check_problem_availability,
    function_label,
    load_experiment_config,
    make_experiment_problem,
)
from landscape_queries.cheap import calculate_descriptor_cheap
from landscape_queries.sampling import sample_problem
from landscape_queries.specs import DESCRIPTOR_CHEAP_COLUMNS, get_query_spec
from optimizers import OptimizerSettings, advance_optimizer_state, initialize_optimizer_state


TRADITIONAL_AAS_DATA_PROTOCOL = "fe0_query_static_portfolio_outcomes_v1"
TRADITIONAL_AAS_MODEL_PROTOCOL = "fe0_query_multioutput_rf_static_aas_v1"
TRADITIONAL_AAS_ONLINE_PROTOCOL = "fe0_query_static_aas_online_v1"
TRADITIONAL_AAS_MODEL_STREAM = 2026082807
TRADITIONAL_AAS_FEATURE_COLUMNS = (
    *DESCRIPTOR_CHEAP_COLUMNS,
    "remaining_budget_ratio",
)
TRADITIONAL_AAS_STATE_KEY = ("problem_id", "seed")


def collect_traditional_aas_data(
    *,
    config_path: str | Path,
    only_splits: tuple[str, ...] | None = None,
    only_functions: tuple[int, ...] | None = None,
    workers: int = 1,
    overwrite: bool = False,
) -> dict[str, int]:
    config = load_experiment_config(config_path)
    suites = _selected_suites(config, only_splits)
    tasks = [
        (suite, function)
        for suite in suites
        for function in suite.functions
        if only_functions is None or function in set(only_functions)
    ]
    if not tasks:
        raise ValueError("no Traditional-AAS data shards were selected")
    check_problem_availability(config, tasks)
    if workers < 1:
        raise ValueError("workers must be at least one")
    pending = []
    skipped = 0
    for suite, function in tasks:
        paths = traditional_aas_shard_paths(config, suite, function)
        existing = tuple(path.exists() for path in paths)
        if all(existing) and not overwrite:
            skipped += 1
            continue
        if any(existing) and not all(existing) and not overwrite:
            raise FileExistsError(
                f"Traditional-AAS shard outputs must be regenerated together: {paths[0].parent}"
            )
        pending.append((suite, function))
    written = 0
    if workers == 1:
        for suite, function in pending:
            _collect_and_write_shard(config, suite, function, overwrite)
            written += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _collect_and_write_shard,
                    config,
                    suite,
                    function,
                    overwrite,
                )
                for suite, function in pending
            ]
            for future in as_completed(futures):
                future.result()
                written += 1
    return {"written_shards": written, "skipped_shards": skipped}


def traditional_aas_shard_paths(
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
) -> tuple[Path, Path]:
    directory = (
        config.output_root
        / "traditional_aas"
        / suite.split
        / function_label(suite.suite, function)
        / f"dimension_{config.dimension}"
    )
    return (
        directory / "query_samples.parquet",
        directory / "static_action_outcomes.parquet",
    )


def _collect_and_write_shard(
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    overwrite: bool,
) -> tuple[int, int]:
    sample_path, outcome_path = traditional_aas_shard_paths(config, suite, function)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        sample_path.unlink(missing_ok=True)
        outcome_path.unlink(missing_ok=True)
    samples, outcomes = build_traditional_aas_rows(
        config=config,
        suite=suite,
        function=function,
    )
    pd.DataFrame(samples).to_parquet(sample_path, index=False)
    pd.DataFrame(outcomes).to_parquet(outcome_path, index=False)
    return len(samples), len(outcomes)


def build_traditional_aas_rows(
    *,
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
) -> tuple[list[dict], list[dict]]:
    query_spec = get_query_spec(config.query.query_id)
    query_fe = query_spec.sample_design.sample_size(config.dimension)
    action_budget = config.fe_total - query_fe
    if action_budget < config.population_size:
        raise ValueError("Traditional AAS leaves insufficient optimizer FE")
    samples: list[dict] = []
    outcomes: list[dict] = []
    component_functions = list(_component_functions(suite, function))
    for instance in suite.instances:
        for seed in config.seeds:
            problem = make_experiment_problem(
                suite,
                function=function,
                instance=instance,
                dimension=config.dimension,
                boundary_handling=config.boundary_handling,
            )
            try:
                query_sample = sample_problem(
                    problem=problem,
                    sample_design=query_spec.sample_design,
                    base_seed=seed,
                    function=function,
                    instance=instance,
                    success_gap_target=config.success_gap_target,
                    failure_loss_cap=config.failure_loss_cap,
                )
                if (
                    not bool(query_sample["sample_path_completed"])
                    or int(query_sample["sample_effective_FE"]) != int(query_fe)
                ):
                    raise RuntimeError(
                        "Traditional-AAS Query did not complete its FE budget: "
                        f"status={query_sample['sample_status']}, "
                        f"effective_FE={int(query_sample['sample_effective_FE'])}, "
                        f"planned_FE={int(query_fe)}"
                    )
                descriptors = _query_descriptors(problem, query_sample)
                common = {
                    "data_protocol": TRADITIONAL_AAS_DATA_PROTOCOL,
                    "split": suite.split,
                    "suite": suite.suite,
                    "problem_id": problem.problem_id,
                    "function_id": problem.function_id,
                    "family": problem.family,
                    "cv_group_id": problem.cv_group_id,
                    "component_functions": component_functions,
                    "dimension": problem.dimension,
                    "seed": seed,
                    "FE_total": config.fe_total,
                    "FE_query": query_fe,
                    "FE_action": action_budget,
                    "remaining_budget_ratio": float(action_budget / config.fe_total),
                    "query_id": query_spec.query_id,
                    "sample_design_id": query_spec.sample_design_id,
                    "query_preprocessing_id": query_spec.preprocessing_id,
                    "query_sample_status": str(query_sample["sample_status"]),
                    "query_sample_path_completed": bool(
                        query_sample["sample_path_completed"]
                    ),
                    "query_sample_effective_FE": int(
                        query_sample["sample_effective_FE"]
                    ),
                    "query_sample_best_gap": float(query_sample["query_best_gap"]),
                    "query_sample_first_hit_offset": query_sample[
                        "query_first_hit_offset"
                    ],
                    "boundary_handling": config.boundary_handling,
                    **descriptors,
                }
                samples.append({**common, **query_sample})
                for algorithm in config.algorithms:
                    outcome = _run_static_candidate(
                        config=config,
                        problem=problem,
                        function=function,
                        instance=instance,
                        seed=seed,
                        algorithm=algorithm,
                        action_budget=action_budget,
                        query_sample=query_sample,
                    )
                    if (
                        str(outcome["action_status"]) != "completed"
                        or int(outcome["effective_action_FE"]) != int(action_budget)
                    ):
                        raise RuntimeError(
                            "Traditional-AAS candidate did not complete its FE budget: "
                            f"algorithm={algorithm}, "
                            f"status={outcome['action_status']}, "
                            f"effective_FE={int(outcome['effective_action_FE'])}, "
                            f"planned_FE={int(action_budget)}; "
                            f"{outcome['failure_type']}: {outcome['failure_message']}"
                        )
                    outcomes.append({**common, **outcome})
            finally:
                problem.close()
    return samples, outcomes


def train_traditional_aas(
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
    if tuple(phase1_bundle.get("portfolio", ())) != config.algorithms:
        raise ValueError("Traditional AAS and Phase 1 portfolios differ")
    train = read_traditional_aas_data(config)
    states, targets = _state_matrix(train, config.algorithms)
    _validate_training_sources(states)
    output = Path(output_dir)
    paths = (
        output / "traditional_aas_models.joblib",
        output / "oof_predictions.parquet",
        output / "training_summary.json",
    )
    if any(path.exists() for path in paths) and not overwrite:
        raise FileExistsError(f"Traditional-AAS model outputs already exist: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in paths:
            path.unlink(missing_ok=True)
        (output / "validation_predictions.parquet").unlink(missing_ok=True)

    oof = _family_oof_predictions(states, targets, config)
    model = _fit_model(states, targets, config, fold_number=0)
    validation_rows = 0
    if validation_config_path is not None:
        validation_config = load_experiment_config(validation_config_path)
        validation_long = read_traditional_aas_data(validation_config)
        validation_states, validation_targets = _state_matrix(
            validation_long,
            validation_config.algorithms,
        )
        validation = _predict_model(
            model,
            validation_states,
            validation_targets,
            validation_config.algorithms,
        )
        validation["evaluation_split"] = "bbob_validation"
        validation.to_parquet(output / "validation_predictions.parquet", index=False)
        validation_rows = len(validation)
    bundle = {
        "model_protocol": TRADITIONAL_AAS_MODEL_PROTOCOL,
        "model": model,
        "portfolio": config.algorithms,
        "feature_columns": TRADITIONAL_AAS_FEATURE_COLUMNS,
        "default_algorithm": default_algorithm,
        "dimension": config.dimension,
        "FE_total": config.fe_total,
        "population_size": config.population_size,
        "boundary_handling": config.boundary_handling,
        "query_config": config.query,
    }
    joblib.dump(bundle, output / "traditional_aas_models.joblib")
    oof.to_parquet(output / "oof_predictions.parquet", index=False)
    summary = {
        "model_protocol": TRADITIONAL_AAS_MODEL_PROTOCOL,
        "feature_columns": list(TRADITIONAL_AAS_FEATURE_COLUMNS),
        "feature_count": len(TRADITIONAL_AAS_FEATURE_COLUMNS),
        "portfolio": list(config.algorithms),
        "train_state_rows": int(len(states)),
        "oof_rows": int(len(oof)),
        "validation_rows": int(validation_rows),
        "validation_rows_used_for_fit": 0,
        "query_FE": int(states["FE_query"].iloc[0]),
    }
    with (output / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def read_traditional_aas_data(config: ExperimentConfig) -> pd.DataFrame:
    frames = []
    for suite in config.suites:
        for function in suite.functions:
            path = traditional_aas_shard_paths(config, suite, function)[1]
            if not path.exists():
                raise FileNotFoundError(
                    f"missing Traditional-AAS action data: {path}"
                )
            frames.append(pd.read_parquet(path))
    if not frames:
        raise ValueError("no Traditional-AAS action data were found")
    return pd.concat(frames, ignore_index=True)


def run_traditional_aas_policy(
    *,
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    instance: int,
    seed: int,
    bundle: dict[str, Any],
) -> tuple[dict, list[dict], dict]:
    _validate_online_bundle(bundle, config)
    problem = make_experiment_problem(
        suite,
        function=function,
        instance=instance,
        dimension=config.dimension,
        boundary_handling=config.boundary_handling,
    )
    query_spec = get_query_spec(config.query.query_id)
    query_fe = query_spec.sample_design.sample_size(config.dimension)
    action_budget = config.fe_total - query_fe
    status = "completed"
    failure_type = ""
    failure_message = ""
    selected_algorithm = str(bundle["default_algorithm"])
    query_sample = None
    candidate = None
    try:
        query_sample = sample_problem(
            problem=problem,
            sample_design=query_spec.sample_design,
            base_seed=seed,
            function=function,
            instance=instance,
            success_gap_target=config.success_gap_target,
            failure_loss_cap=config.failure_loss_cap,
        )
        if not bool(query_sample["sample_path_completed"]):
            raise RuntimeError("Traditional-AAS online Query did not complete")
        descriptors = _query_descriptors(problem, query_sample)
        features = {
            **descriptors,
            "remaining_budget_ratio": float(action_budget / config.fe_total),
        }
        frame = pd.DataFrame(
            [{column: features[column] for column in TRADITIONAL_AAS_FEATURE_COLUMNS}]
        )
        prediction = np.asarray(bundle["model"].predict(frame), dtype=float)
        if prediction.shape != (1, len(config.algorithms)):
            raise RuntimeError("Traditional-AAS model returned an unexpected shape")
        selected_algorithm = min(
            config.algorithms,
            key=lambda name: (
                float(prediction[0, config.algorithms.index(name)]),
                config.algorithms.index(name),
            ),
        )
        candidate = _run_static_candidate(
            config=config,
            problem=problem,
            function=function,
            instance=instance,
            seed=seed,
            algorithm=selected_algorithm,
            action_budget=action_budget,
            query_sample=query_sample,
        )
        if str(candidate["action_status"]) != "completed":
            raise RuntimeError("Traditional-AAS selected optimizer path did not complete")
        final_gap = float(candidate["operational_gap"])
        effective_fe = int(query_sample["sample_effective_FE"]) + int(
            candidate["effective_action_FE"]
        )
        query_first = query_sample["query_first_hit_offset"]
        action_first = candidate["first_hit_action_offset"]
        first_hit_fe = (
            int(query_first)
            if query_first is not None
            else (
                None
                if action_first is None
                else int(query_fe + int(action_first))
            )
        )
        reference = problem.reference_value
        if reference is None:
            raise ValueError("Traditional-AAS endpoint requires a reference value")
        best_fitness = float(reference) + final_gap
    except Exception as exc:
        status = "failed"
        failure_type = type(exc).__name__
        failure_message = str(exc)[:500]
        final_gap = config.failure_loss_cap
        best_fitness = None
        effective_fe = int(
            0 if query_sample is None else query_sample["sample_effective_FE"]
        ) + int(0 if candidate is None else candidate["effective_action_FE"])
        first_hit_fe = None
    finally:
        problem.close()
    default_algorithm = str(bundle["default_algorithm"])
    outcome = {
        "policy_protocol": TRADITIONAL_AAS_ONLINE_PROTOCOL,
        "policy_name": "traditional_aas",
        "split": suite.split,
        "suite": suite.suite,
        "problem_id": problem.problem_id,
        "function_id": problem.function_id,
        "family": problem.family,
        "cv_group_id": problem.cv_group_id,
        "dimension": problem.dimension,
        "prefix_algorithm": selected_algorithm,
        "default_algorithm": default_algorithm,
        "seed": seed,
        "FE_total": config.fe_total,
        "policy_status": status,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "effective_FE": effective_fe,
        "best_fitness": best_fitness,
        "benchmark_reference_value": problem.reference_value,
        "final_gap": float(final_gap),
        "log10_gap": float(
            np.log10(
                np.clip(final_gap, config.log10_gap_floor, config.log10_gap_cap)
            )
        ),
        "success": bool(status == "completed" and final_gap <= config.success_gap_target),
        "first_hit_FE": first_hit_fe,
        "query_triggered": True,
        "query_FE": int(
            0 if query_sample is None else query_sample["sample_effective_FE"]
        ),
        "selected_algorithm": selected_algorithm,
        "selected_FE": 0,
        "selected_decision_opportunity_index": None,
        "selected_equals_default": bool(selected_algorithm == default_algorithm),
        "selected_equals_prefix": True,
        "handoff_required": False,
        "handoff_type": "native_optimizer_state",
        "optimizer_initialization_mode": "fresh_optimizer_initialization_after_query",
        "switch_count": 0,
        "boundary_handling": config.boundary_handling,
    }
    resource = {
        "policy_name": "traditional_aas",
        "split": suite.split,
        "problem_id": problem.problem_id,
        "prefix_algorithm": selected_algorithm,
        "seed": seed,
        "timing_source": "single_online_policy_execution_diagnostic",
        "timing_replay_status": status,
    }
    return outcome, [], resource


def _run_static_candidate(
    *,
    config: ExperimentConfig,
    problem,
    function: int,
    instance: int,
    seed: int,
    algorithm: str,
    action_budget: int,
    query_sample: dict,
) -> dict:
    reference = problem.reference_value
    if reference is None:
        raise ValueError("Traditional-AAS action loss requires a reference value")
    tracker = _TrackedActionObjective(
        problem=problem,
        reference_value=float(reference),
        success_gap_target=config.success_gap_target,
        initial_best_fitness=float(reference) + config.failure_loss_cap,
    )
    tracked_problem = tracker.wrapped_problem()
    status = "completed"
    failure_type = ""
    failure_message = ""
    continuation_gap = config.failure_loss_cap
    try:
        settings = OptimizerSettings(
            population_size=config.population_size,
            checkpoint_ratios=(1.0,),
            boundary_handling=config.boundary_handling,
        )
        state = initialize_optimizer_state(
            algorithm=algorithm,
            problem=tracked_problem,
            seed=seed,
            settings=settings,
        )
        advance_optimizer_state(
            state=state,
            problem=tracked_problem,
            fe_budget=action_budget - int(state.evaluations),
        )
        continuation_gap = min(
            max(float(state.best_fitness) - float(reference), 0.0),
            config.failure_loss_cap,
        )
        if tracker.evaluations != action_budget:
            raise RuntimeError("Traditional-AAS candidate used an incorrect action FE budget")
    except Exception as exc:
        status = "failed"
        failure_type = type(exc).__name__
        failure_message = str(exc)[:500]
    query_completed = bool(query_sample["sample_path_completed"])
    operational_gap = (
        min(continuation_gap, float(query_sample["query_best_gap"]))
        if status == "completed" and query_completed
        else config.failure_loss_cap
    )
    return {
        "candidate_action": algorithm,
        "action_status": status,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "planned_action_FE": int(action_budget),
        "effective_action_FE": int(tracker.evaluations),
        "continuation_gap": float(continuation_gap),
        "operational_gap": float(operational_gap),
        "log10_action_loss": float(
            np.log10(
                np.clip(
                    operational_gap,
                    config.log10_gap_floor,
                    config.log10_gap_cap,
                )
            )
        ),
        "first_hit_action_offset": tracker.first_hit_offset,
        "target_hit": bool(
            status == "completed" and operational_gap <= config.success_gap_target
        ),
    }


def _query_descriptors(problem, query_sample: dict) -> dict[str, float | None]:
    if not bool(query_sample["sample_path_completed"]):
        return {column: None for column in DESCRIPTOR_CHEAP_COLUMNS}
    values = calculate_descriptor_cheap(
        np.asarray(query_sample["X"], dtype=float),
        np.asarray(query_sample["y"], dtype=float),
        problem.lower_bounds,
        problem.upper_bounds,
    )
    return {
        column: (
            float(values[column]) if np.isfinite(float(values[column])) else None
        )
        for column in DESCRIPTOR_CHEAP_COLUMNS
    }


def _state_matrix(
    rows: pd.DataFrame,
    portfolio: tuple[str, ...],
) -> tuple[pd.DataFrame, np.ndarray]:
    required = {
        *TRADITIONAL_AAS_FEATURE_COLUMNS,
        *TRADITIONAL_AAS_STATE_KEY,
        "candidate_action",
        "log10_action_loss",
        "suite",
        "family",
        "function_id",
        "cv_group_id",
        "component_functions",
        "FE_query",
        "query_sample_path_completed",
        "query_sample_effective_FE",
        "FE_action",
        "action_status",
        "effective_action_FE",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Traditional-AAS data is missing columns: {missing}")
    order = {name: index for index, name in enumerate(portfolio)}
    ordered = rows.copy()
    ordered["candidate_order"] = ordered["candidate_action"].astype(str).map(order)
    if ordered["candidate_order"].isna().any():
        raise ValueError("Traditional-AAS data contains an action outside the portfolio")
    ordered = ordered.sort_values(
        [*TRADITIONAL_AAS_STATE_KEY, "candidate_order"],
        kind="mergesort",
    )
    counts = ordered.groupby(list(TRADITIONAL_AAS_STATE_KEY), sort=False).size()
    if not counts.eq(len(portfolio)).all():
        raise ValueError("each Traditional-AAS state must contain the complete portfolio")
    states = (
        ordered.groupby(list(TRADITIONAL_AAS_STATE_KEY), sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    targets = ordered["log10_action_loss"].to_numpy(dtype=float).reshape(
        len(states), len(portfolio)
    )
    if not np.isfinite(targets).all():
        raise ValueError("Traditional-AAS targets must be finite")
    if not ordered["query_sample_path_completed"].astype(bool).all():
        raise ValueError("Traditional-AAS labels require completed Query samples")
    if not ordered["query_sample_effective_FE"].astype(int).eq(
        ordered["FE_query"].astype(int)
    ).all():
        raise ValueError("Traditional-AAS Query samples must use the planned FE budget")
    if not ordered["action_status"].astype(str).eq("completed").all():
        raise ValueError("Traditional-AAS labels require completed optimizer paths")
    if not ordered["effective_action_FE"].astype(int).eq(
        ordered["FE_action"].astype(int)
    ).all():
        raise ValueError("Traditional-AAS optimizer paths must use the planned FE budget")
    return states, targets


def _fit_model(
    states: pd.DataFrame,
    targets: np.ndarray,
    config: ExperimentConfig,
    *,
    fold_number: int,
) -> Pipeline:
    random_state = int(
        np.random.SeedSequence(
            [TRADITIONAL_AAS_MODEL_STREAM, int(fold_number), config.dimension]
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
        states[list(TRADITIONAL_AAS_FEATURE_COLUMNS)],
        targets,
        regressor__sample_weight=_block_function_weights(states),
    )
    return model


def _family_oof_predictions(
    states: pd.DataFrame,
    targets: np.ndarray,
    config: ExperimentConfig,
) -> pd.DataFrame:
    bbob_mask = states["suite"].astype(str).eq("bbob").to_numpy()
    bbob = states.loc[bbob_mask].copy()
    families = tuple(sorted(set(bbob["family"].astype(str))))
    if len(families) < 2:
        raise ValueError("Traditional-AAS OOF requires at least two BBOB families")
    predictions = []
    for fold_number, family in enumerate(families, start=1):
        heldout_functions = set(
            bbob.loc[bbob["family"].astype(str).eq(family), "function_id"].astype(str)
        )
        eval_mask = (
            states["suite"].astype(str).eq("bbob")
            & states["family"].astype(str).eq(family)
        ).to_numpy()
        train_mask = ~eval_mask
        ma_safe = ~states.apply(
            lambda row: _ma_overlaps_heldout(row, heldout_functions),
            axis=1,
        ).to_numpy()
        selected = train_mask & ma_safe
        model = _fit_model(
            states.loc[selected].reset_index(drop=True),
            targets[selected],
            config,
            fold_number=fold_number,
        )
        predicted = _predict_model(
            model,
            states.loc[eval_mask].reset_index(drop=True),
            targets[eval_mask],
            config.algorithms,
        )
        predicted["oof_fold"] = fold_number
        predicted["heldout_family"] = family
        predictions.append(predicted)
    result = pd.concat(predictions, ignore_index=True)
    if len(result) != int(np.sum(bbob_mask)):
        raise RuntimeError("Traditional-AAS OOF coverage differs from BBOB states")
    return result


def _predict_model(
    model: Pipeline,
    states: pd.DataFrame,
    targets: np.ndarray,
    portfolio: tuple[str, ...],
) -> pd.DataFrame:
    predicted = np.asarray(
        model.predict(states[list(TRADITIONAL_AAS_FEATURE_COLUMNS)]),
        dtype=float,
    )
    if predicted.shape != targets.shape:
        raise RuntimeError("Traditional-AAS prediction shape differs from targets")
    selected_indices = np.argmin(predicted, axis=1)
    best_indices = np.argmin(targets, axis=1)
    output = states.copy()
    output["selected_algorithm"] = [portfolio[index] for index in selected_indices]
    output["best_observed_static_algorithm"] = [
        portfolio[index] for index in best_indices
    ]
    output["selected_log10_loss"] = targets[
        np.arange(len(targets)), selected_indices
    ]
    output["best_observed_static_log10_loss"] = targets[
        np.arange(len(targets)), best_indices
    ]
    output["static_selection_regret"] = (
        output["selected_log10_loss"] - output["best_observed_static_log10_loss"]
    )
    for index, algorithm in enumerate(portfolio):
        output[f"predicted_log10_loss_{algorithm}"] = predicted[:, index]
        output[f"observed_log10_loss_{algorithm}"] = targets[:, index]
    return output


def _validate_training_sources(states: pd.DataFrame) -> None:
    if not set(states["suite"].astype(str)).issuperset({"bbob", "mabbob"}):
        raise ValueError("Traditional-AAS training requires BBOB and selected MA-BBOB")


def _validate_online_bundle(bundle: dict[str, Any], config: ExperimentConfig) -> None:
    if str(bundle.get("model_protocol")) != TRADITIONAL_AAS_MODEL_PROTOCOL:
        raise ValueError("Traditional-AAS model protocol differs from online runner")
    if tuple(bundle.get("portfolio", ())) != config.algorithms:
        raise ValueError("Traditional-AAS portfolio differs from online config")
    if tuple(bundle.get("feature_columns", ())) != TRADITIONAL_AAS_FEATURE_COLUMNS:
        raise ValueError("Traditional-AAS feature columns differ from online runner")
    for name, value in (
        ("dimension", config.dimension),
        ("FE_total", config.fe_total),
        ("population_size", config.population_size),
    ):
        if int(bundle.get(name, -1)) != int(value):
            raise ValueError(f"Traditional-AAS {name} differs from online config")
    if str(bundle.get("boundary_handling")) != config.boundary_handling:
        raise ValueError("Traditional-AAS boundary handling differs from online config")
    if bundle.get("query_config") != config.query:
        raise ValueError("Traditional-AAS Query config differs from online config")


def _selected_suites(
    config: ExperimentConfig,
    only_splits: tuple[str, ...] | None,
) -> tuple[SuiteConfig, ...]:
    if only_splits is None:
        return config.suites
    requested = set(only_splits)
    missing = requested.difference(suite.split for suite in config.suites)
    if missing:
        raise ValueError(f"requested split is absent from config: {sorted(missing)}")
    return tuple(suite for suite in config.suites if suite.split in requested)


def collect_main() -> None:
    parser = argparse.ArgumentParser(description="Collect FE=0 Traditional-AAS outcomes.")
    parser.add_argument("--config", default="configs/behavior_with_ela_train.yaml")
    parser.add_argument("--only-split", action="append", default=None)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = collect_traditional_aas_data(
        config_path=args.config,
        only_splits=None if args.only_split is None else tuple(args.only_split),
        only_functions=(
            None if args.only_function is None else tuple(args.only_function)
        ),
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(
        f"wrote {result['written_shards']} Traditional-AAS shards; "
        f"skipped {result['skipped_shards']} existing shards"
    )


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train the FE=0 Traditional-AAS baseline.")
    parser.add_argument("--train-config", default="configs/behavior_with_ela_train.yaml")
    parser.add_argument(
        "--phase1-model",
        default=(
            "results/behavior_with_ela/model/behavior_action_gain/models.joblib"
        ),
    )
    parser.add_argument(
        "--validation-config",
        default="configs/behavior_with_ela_validation.yaml",
    )
    parser.add_argument(
        "--output",
        default="results/behavior_with_ela/model/traditional_aas",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = train_traditional_aas(
        train_config_path=args.train_config,
        phase1_model_path=args.phase1_model,
        output_dir=args.output,
        validation_config_path=args.validation_config,
        overwrite=args.overwrite,
    )
    print(
        f"trained Traditional AAS with {result['train_state_rows']} states and "
        f"{result['oof_rows']} BBOB family-OOF predictions"
    )
