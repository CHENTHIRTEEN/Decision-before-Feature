from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter
from math import ceil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from behavior.features import extract_behavior_rows
from benchmarks import make_problem
from experiments.phase1_batch_common import algorithms, fe_total_for_dimension, load_config
from landscape_queries.cheap import calculate_descriptor_cheap
from landscape_queries.consistency import _check_action_losses
from landscape_queries.sampling import sample_problem
from landscape_queries.specs import get_query_spec, get_sample_design_spec
from optimizers import (
    NO_QUERY_TRANSFER_EVENT,
    OptimizerSettings,
    advance_optimizer_state,
    clone_optimizer_state,
    initialize_optimizer_state,
    initialize_transferred_optimizer_state,
)
from selection_reference.action_losses import ACTION_LOSS_PROTOCOL, evaluate_candidate_actions
from selection_reference.build import build_selection_reference
from selection_reference.model import (
    SELECTION_REFERENCE_PROTOCOL,
    SELECTOR_TARGET_TRANSFORM,
    fit_selector_with_cross_family_predictions,
    load_selector_model,
    measure_online_selection_runtime,
    predict_with_main_prefix_cross_family_fits,
    prepare_state_matrix,
    save_selector_model,
    selection_rows,
)
from trajectory.records import TrajectoryRecord
from trajectory.sampling import (
    SAMPLING_METADATA_COLUMNS,
    budget_milestone_metadata,
)
from trajectory.window_statistics import NativeUpdateWindowRecorder


def check_state_action_continuations(*, config_path: Path) -> dict[str, int | str]:
    config = load_config(config_path)
    if str(config["suite"]).lower() != "bbob":
        raise ValueError("selection-reference-check requires a real BBOB configuration")
    portfolio = tuple(algorithms(config))
    population_size = int(config["population_size"])
    settings = OptimizerSettings(population_size=population_size, checkpoint_ratios=(1.0,))
    function = int(config["functions"][0])
    instance = int(config["instances"][0])
    dimension = int(config["dimensions"][0])
    seed = int(config["seeds"][0])
    prefix_budget = 20 * population_size
    action_budget = 10 * population_size
    checked = 0

    problem = make_problem(
        {"suite": "bbob", "function": function, "instance": instance, "dimension": dimension}
    )
    try:
        for prefix_algorithm in portfolio:
            state = initialize_optimizer_state(
                algorithm=prefix_algorithm,
                problem=problem,
                seed=seed,
                settings=settings,
            )
            advance_optimizer_state(state=state, problem=problem, fe_budget=prefix_budget)
            expected_native = advance_optimizer_state(
                state=clone_optimizer_state(state),
                problem=problem,
                fe_budget=action_budget,
            )
            outcomes = evaluate_candidate_actions(
                checkpoint_state=state,
                problem=problem,
                portfolio=portfolio,
                fe_budget=action_budget,
                seed=seed,
                function=function,
                instance=instance,
                checkpoint_fe=int(state.evaluations),
                action_budget_mode="query_adjusted_budget",
                failure_loss_cap=float(config["failure_loss_cap"]),
            )
            native = [row for row in outcomes if row["action"] == "continue_current"]
            reference_value = float(problem.reference_value)
            expected_native_gap = max(float(expected_native.best_fitness) - reference_value, 0.0)
            if (
                len(native) != 1
                or float(native[0]["action_loss_raw"]) != float(expected_native.best_fitness)
                or float(native[0]["action_loss"]) != expected_native_gap
            ):
                raise ValueError("continue_current does not reproduce native optimizer continuation")
            expected_actions = {"continue_current", *set(portfolio).difference({prefix_algorithm})}
            if {str(row["action"]) for row in outcomes} != expected_actions:
                raise ValueError("candidate actions are not continue_current plus the other three algorithms")
            transfers = [
                row for row in outcomes if row["transition_mode"] == "population_transfer_initialization"
            ]
            if len(outcomes) != 4 or len(transfers) != 3:
                raise ValueError("each real BBOB state must have one native and three transfer actions")
            losses = np.asarray([row["action_loss"] for row in outcomes], dtype=float)
            expected_norm = (losses - np.min(losses)) / max(float(np.max(losses) - np.min(losses)), 1e-12)
            observed_norm = np.asarray([row["action_loss_norm"] for row in outcomes], dtype=float)
            if not np.isfinite(observed_norm).all() or not np.allclose(
                observed_norm,
                expected_norm,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("statewise action-loss target transform is inconsistent")
            checked += 1
    finally:
        problem.close()

    _check_query_specific_regression(config=config, portfolio=portfolio, settings=settings)
    _check_action_loss_budget_separation(config=config, portfolio=portfolio, settings=settings)
    print(
        "state-action continuation consistency passed: "
        f"{checked} prefixes, {len(portfolio)} actions per prefix, real BBOB f{function:03d} d{dimension}"
    )
    return {
        "prefixes": checked,
        "actions_per_prefix": len(portfolio),
        "benchmark": f"bbob_f{function:03d}_i{instance:02d}_d{dimension}",
    }


def _check_query_specific_regression(
    *,
    config: dict,
    portfolio: tuple[str, ...],
    settings: OptimizerSettings,
) -> None:
    spec = get_query_spec("descriptor_cheap_invariant")
    sample_design = spec.sample_design
    default_algorithm = portfolio[0]
    prefix_algorithms = portfolio[:2]
    seed = int(config["seeds"][0])
    dimension = int(config["dimensions"][0])
    action_rows = []
    behavior_rows = []
    query_rows = []
    for function in [int(value) for value in config["functions"][:2]]:
        instance = int(config["instances"][0])
        fe_total = fe_total_for_dimension(config, dimension)
        problem = make_problem(
            {"suite": "bbob", "function": function, "instance": instance, "dimension": dimension}
        )
        try:
            sample = sample_problem(
                problem=problem,
                sample_design=sample_design,
                base_seed=0,
                function=function,
                instance=instance,
                success_gap_target=float(config["success_gap_target"]),
                failure_loss_cap=float(config["failure_loss_cap"]),
            )
            started = perf_counter()
            descriptor = calculate_descriptor_cheap(
                sample["X"], sample["y"], sample["lower_bounds"], sample["upper_bounds"]
            )
            runtime_feature = perf_counter() - started
            query_rows.append(
                {
                    "split": "bbob_train",
                    "problem_id": problem.problem_id,
                    "family": problem.family,
                    "dimension": dimension,
                    "query_id": spec.query_id,
                    "query_protocol": spec.protocol,
                    "query_preprocessing_id": spec.preprocessing_id,
                    "sample_design_id": spec.sample_design_id,
                    "runtime_query_sampling": float(sample["runtime_query_sampling"]),
                    "runtime_query_evaluation": float(sample["runtime_query_evaluation"]),
                    "runtime_query_feature_computation": float(runtime_feature),
                    "runtime_query": float(sample["runtime_sampling_evaluation"] + runtime_feature),
                    "feature_status": "ok",
                    "feature_failure": "[]",
                    "feature_group_status": json.dumps(
                        {
                            "descriptor_cheap": {
                                "status": "ok",
                                "runtime_seconds": float(runtime_feature),
                                "nonfinite_columns": [],
                                "warnings": [],
                                "error": "",
                            }
                        },
                        sort_keys=True,
                    ),
                    "feature_nonfinite": "{}",
                    "additional_function_evaluations": 0,
                    "query_feature_columns": json.dumps(list(spec.feature_columns)),
                    **descriptor,
                }
            )
            for prefix_algorithm in prefix_algorithms:
                state, trajectory_rows = _real_prefix_state(
                    problem=problem,
                    prefix_algorithm=prefix_algorithm,
                    seed=seed,
                    settings=settings,
                    fe_total=fe_total,
                )
                behavior_rows.append(
                    {"split": "bbob_train", **extract_behavior_rows(trajectory_rows)[-1]}
                )
                sampling_metadata = {
                    column: trajectory_rows[-1][column]
                    for column in SAMPLING_METADATA_COLUMNS
                }
                fe_query = sample_design.sample_size(dimension)
                query_budget = fe_total - int(state.evaluations) - fe_query
                skip_budget = fe_total - int(state.evaluations)
                if prefix_algorithm == default_algorithm:
                    skip_state = clone_optimizer_state(state)
                    runtime_no_query_handoff = 0.0
                    no_query_transition_mode = "native_optimizer_state"
                else:
                    handoff_started = perf_counter()
                    skip_state = initialize_transferred_optimizer_state(
                        algorithm=default_algorithm,
                        source_state=state,
                        problem=problem,
                        seed=seed,
                        function=function,
                        instance=instance,
                        event=NO_QUERY_TRANSFER_EVENT,
                    )
                    runtime_no_query_handoff = perf_counter() - handoff_started
                    no_query_transition_mode = "population_transfer_initialization"
                skip = advance_optimizer_state(
                    state=skip_state,
                    problem=problem,
                    fe_budget=skip_budget,
                )
                outcomes = evaluate_candidate_actions(
                    checkpoint_state=state,
                    problem=problem,
                    portfolio=portfolio,
                    fe_budget=query_budget,
                    seed=seed,
                    function=function,
                    instance=instance,
                    checkpoint_fe=int(state.evaluations),
                    action_budget_mode="query_adjusted",
                    failure_loss_cap=float(config["failure_loss_cap"]),
                )
                reference_value = float(problem.reference_value)
                skip_raw = float(skip.best_fitness)
                skip_loss = max(skip_raw - reference_value, 0.0)
                common = {
                    "split": "bbob_train",
                    "problem_id": problem.problem_id,
                    "family": problem.family,
                    "dimension": dimension,
                    "prefix_algorithm": prefix_algorithm,
                    "default_algorithm": default_algorithm,
                    "no_query_algorithm": default_algorithm,
                    "seed": seed,
                    "FE": int(state.evaluations),
                    "FE_ratio": float(state.evaluations / fe_total),
                    "FE_total": fe_total,
                    **sampling_metadata,
                    "sample_design_id": spec.sample_design_id,
                    "sample_design_protocol": sample_design.protocol,
                    "FE_query": fe_query,
                    "FE_no_query_optimization": skip_budget,
                    "FE_query_optimization": query_budget,
                    "remaining_budget_ratio": float(query_budget / fe_total),
                    "performance_value_mode": "raw_objective",
                    "performance_loss_mode": "known_optimum_gap",
                    "benchmark_reference_value": reference_value,
                    "p_skip": skip_loss,
                    "p_skip_raw": skip_raw,
                    "loss_skip": skip_loss,
                    "runtime_no_query_handoff": float(runtime_no_query_handoff),
                    "runtime_no_query_optimization": float(skip.runtime_seconds),
                    "no_query_transition_mode": no_query_transition_mode,
                    "action_loss_protocol": ACTION_LOSS_PROTOCOL,
                }
                action_rows.extend({**common, **outcome} for outcome in outcomes)
        finally:
            problem.close()

    action_frame = pd.DataFrame(action_rows)
    behavior_frame = pd.DataFrame(behavior_rows).rename(columns={"algorithm": "prefix_algorithm"})
    query_frame = pd.DataFrame(query_rows)
    states, observed_portfolio = prepare_state_matrix(
        action_frame,
        behavior=behavior_frame,
        query_features=query_frame,
        query_spec=spec,
    )
    main_states = states[
        states["prefix_algorithm"].astype(str) == states["default_algorithm"].astype(str)
    ].reset_index(drop=True)
    cross_probe_states = states[
        states["prefix_algorithm"].astype(str) != states["default_algorithm"].astype(str)
    ].reset_index(drop=True)
    selector, predictions, source = fit_selector_with_cross_family_predictions(
        main_states,
        observed_portfolio,
        spec,
    )
    main_reference = selection_rows(
        states=main_states,
        portfolio=observed_portfolio,
        predictions=predictions,
        prediction_source=source,
        runtime_selection=measure_online_selection_runtime(selector, main_states),
    )
    cross_probe_predictions = predict_with_main_prefix_cross_family_fits(
        training_states=main_states,
        prediction_states=cross_probe_states,
        portfolio=observed_portfolio,
        query_spec=spec,
    )
    cross_probe_reference = selection_rows(
        states=cross_probe_states,
        portfolio=observed_portfolio,
        predictions=cross_probe_predictions,
        prediction_source="cross_cv_group_main_prefix",
        runtime_selection=measure_online_selection_runtime(selector, cross_probe_states),
    )
    reference = pd.concat([main_reference, cross_probe_reference], ignore_index=True)
    if len(reference) != 4 or source != "cross_cv_group":
        raise ValueError("query-specific selector did not produce cross-CV-group predictions")
    if set(cross_probe_reference["selector_prediction_source"].astype(str)) != {
        "cross_cv_group_main_prefix"
    }:
        raise ValueError("cross-probe states did not use main-prefix cross-CV-group fits")
    expected_handoff = ~reference["selected_equals_prefix"].astype(bool)
    if not np.array_equal(reference["handoff_required"].to_numpy(dtype=bool), expected_handoff.to_numpy()):
        raise ValueError("selection reference handoff_required is inconsistent")
    if set(reference["selector_target_transform"].astype(str)) != {SELECTOR_TARGET_TRANSFORM}:
        raise ValueError("selection reference did not freeze the action-loss target transform")
    _check_explicit_action_relation_cases(main_states, observed_portfolio)
    with tempfile.TemporaryDirectory(prefix="decision-before-feature-query-check-") as directory:
        root = Path(directory)
        action_path = root / "action_losses.parquet"
        behavior_root = root / "phase1_refined_sampling"
        behavior_path = (
            behavior_root
            / "bbob_train"
            / "bbob_f001"
            / f"dimension_{dimension}"
            / "behavior.parquet"
        )
        feature_path = root / "features.parquet"
        reference_path = root / "selection_reference.parquet"
        model_path = root / "statewise_selector.joblib"
        pq.write_table(pa.Table.from_pandas(action_frame, preserve_index=False), action_path)
        behavior_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(
                behavior_frame.drop(columns="split"),
                preserve_index=False,
            ),
            behavior_path,
        )
        pq.write_table(pa.Table.from_pandas(query_frame, preserve_index=False), feature_path)
        summary = build_selection_reference(
            query_id=spec.query_id,
            train_action_loss_paths=[action_path],
            predict_action_loss_paths=[],
            behavior_paths=[behavior_root],
            query_feature_paths=[feature_path],
            output_path=reference_path,
            model_output_path=model_path,
            overwrite=True,
        )
        if (
            int(summary["rows"]) != 4
            or int(summary["training_rows"]) != 2
            or int(summary["cross_probe_training_rows"]) != 2
            or not model_path.exists()
        ):
            raise ValueError("query-specific selection-reference artifacts are inconsistent")
        loaded_selector = load_selector_model(model_path)
        if loaded_selector.sample_design_id != spec.sample_design_id:
            raise ValueError("saved selector model lost its query sample design")
        if loaded_selector.selector_target_transform != SELECTOR_TARGET_TRANSFORM:
            raise ValueError("saved selector model lost its action-loss target transform")
        legacy_model_path = root / "legacy_statewise_selector.joblib"
        loaded_selector.protocol = "query_specific_statewise_action_loss_regression_v2"
        save_selector_model(loaded_selector, legacy_model_path)
        try:
            load_selector_model(legacy_model_path)
        except ValueError:
            pass
        else:
            raise ValueError("legacy selector model protocol was not rejected")
        if SELECTION_REFERENCE_PROTOCOL != "query_specific_statewise_action_loss_regression_v6":
            raise ValueError("Selection Reference protocol version is not frozen at v6")
    print("query-specific action-loss regression and utility consistency passed on 2 BBOB families")


def _real_prefix_state(
    *,
    problem,
    prefix_algorithm: str,
    seed: int,
    settings: OptimizerSettings,
    fe_total: int,
):
    state = initialize_optimizer_state(
        algorithm=prefix_algorithm,
        problem=problem,
        seed=seed,
        settings=settings,
    )
    trajectory_rows = []
    window_recorder = NativeUpdateWindowRecorder()
    window_recorder.observe(
        fe=int(state.evaluations),
        native_updates=int(state.generation),
        population=state.population,
        fitness=state.fitness,
        best_fitness=state.best_fitness,
    )
    target_fes = tuple(
        int(ceil(ratio * fe_total / settings.population_size) * settings.population_size)
        for ratio in (0.20, 0.22, 0.24)
    )
    for target_fe in target_fes:
        advance_optimizer_state(
            state=state,
            problem=problem,
            fe_budget=target_fe - int(state.evaluations),
            on_native_update=lambda updated: window_recorder.observe(
                fe=int(updated.evaluations),
                native_updates=int(updated.generation),
                population=updated.population,
                fitness=updated.fitness,
                best_fitness=updated.best_fitness,
            ),
        )
        window_statistics, native_update_history = window_recorder.build(
            fe_total=fe_total,
            problem_id=problem.problem_id,
            algorithm=prefix_algorithm,
        )
        trajectory_rows.append(
            TrajectoryRecord.from_arrays(
                problem_id=problem.problem_id,
                function_id=problem.function_id,
                family=problem.family,
                cv_group_id=problem.cv_group_id,
                dimension=problem.dimension,
                algorithm=prefix_algorithm,
                seed=seed,
                fe=int(state.evaluations),
                fe_total=fe_total,
                native_updates=int(state.generation),
                window_statistics=window_statistics,
                native_update_history=native_update_history,
                population=state.population,
                fitness=state.fitness,
                best_fitness=state.best_fitness,
                sampling_metadata=_milestone_sampling_metadata(
                    actual_fe=int(state.evaluations),
                    fe_total=fe_total,
                    monitor_target_ratio=float(target_fe / fe_total),
                ),
            ).__dict__
        )
    return state, trajectory_rows


def _check_explicit_action_relation_cases(
    states: pd.DataFrame,
    portfolio: tuple[str, ...],
) -> None:
    state = states.iloc[[0]].copy()
    prefix = str(state["prefix_algorithm"].iloc[0])
    other = next(algorithm for algorithm in portfolio if algorithm != prefix)

    def select(target: str, *, default: str | None = None) -> dict:
        case = state.copy()
        if default is not None:
            case["default_algorithm"] = default
            case["no_query_algorithm"] = default
        scores = np.ones((1, len(portfolio)), dtype=float)
        scores[0, portfolio.index(target)] = 0.0
        return selection_rows(
            states=case,
            portfolio=portfolio,
            predictions=scores,
            prediction_source="real_state_relation_contract",
            runtime_selection=0.0,
        ).iloc[0].to_dict()

    native = select(prefix)
    if not (
        native["selected_action"] == "continue_current"
        and bool(native["selected_equals_prefix"])
        and bool(native["selected_equals_default"])
        and not bool(native["handoff_required"])
    ):
        raise ValueError("native continuation relation fields are inconsistent")

    transfer = select(other)
    if not (
        transfer["selected_action"] == other
        and not bool(transfer["selected_equals_prefix"])
        and not bool(transfer["selected_equals_default"])
        and bool(transfer["handoff_required"])
    ):
        raise ValueError("cross-algorithm transfer relation fields are inconsistent")

    cross_prefix = select(prefix, default=other)
    if not (
        bool(cross_prefix["selected_equals_prefix"])
        and not bool(cross_prefix["selected_equals_default"])
        and not bool(cross_prefix["handoff_required"])
    ):
        raise ValueError("selected-vs-prefix and selected-vs-default relations were incorrectly merged")


def _check_action_loss_budget_separation(
    *,
    config: dict,
    portfolio: tuple[str, ...],
    settings: OptimizerSettings,
) -> None:
    function = int(config["functions"][0])
    instance = int(config["instances"][0])
    dimension = int(config["dimensions"][0])
    seed = int(config["seeds"][0])
    fe_total = fe_total_for_dimension(config, dimension)
    prefix_algorithm = portfolio[0]
    problem = make_problem(
        {"suite": "bbob", "function": function, "instance": instance, "dimension": dimension}
    )
    frames = []
    try:
        state = initialize_optimizer_state(
            algorithm=prefix_algorithm,
            problem=problem,
            seed=seed,
            settings=settings,
        )
        advance_optimizer_state(state=state, problem=problem, fe_budget=int(0.20 * fe_total))
        sampling_metadata = _milestone_sampling_metadata(
            actual_fe=int(state.evaluations),
            fe_total=fe_total,
            monitor_target_ratio=0.20,
        )
        for sample_design_id in ("lhs_50d", "lhs_100d"):
            design = get_sample_design_spec(sample_design_id)
            fe_query = design.sample_size(dimension)
            query_budget = fe_total - int(state.evaluations) - fe_query
            skip = advance_optimizer_state(
                state=clone_optimizer_state(state),
                problem=problem,
                fe_budget=fe_total - int(state.evaluations),
            )
            outcomes = evaluate_candidate_actions(
                checkpoint_state=state,
                problem=problem,
                portfolio=portfolio,
                fe_budget=query_budget,
                seed=seed,
                function=function,
                instance=instance,
                checkpoint_fe=int(state.evaluations),
                action_budget_mode="query_adjusted",
                failure_loss_cap=float(config["failure_loss_cap"]),
            )
            reference_value = float(problem.reference_value)
            skip_raw = float(skip.best_fitness)
            skip_loss = max(skip_raw - reference_value, 0.0)
            common = {
                "split": "bbob_train",
                "problem_id": problem.problem_id,
                "family": problem.family,
                "dimension": dimension,
                "prefix_algorithm": prefix_algorithm,
                "default_algorithm": prefix_algorithm,
                "no_query_algorithm": prefix_algorithm,
                "seed": seed,
                "FE": int(state.evaluations),
                "FE_ratio": float(state.evaluations / fe_total),
                "FE_total": fe_total,
                **sampling_metadata,
                "sample_design_id": sample_design_id,
                "sample_design_protocol": design.protocol,
                "FE_query": fe_query,
                "FE_no_query_optimization": fe_total - int(state.evaluations),
                "FE_query_optimization": query_budget,
                "remaining_budget_ratio": float(query_budget / fe_total),
                "performance_value_mode": "raw_objective",
                "performance_loss_mode": "known_optimum_gap",
                "benchmark_reference_value": reference_value,
                "p_skip": skip_loss,
                "p_skip_raw": skip_raw,
                "loss_skip": skip_loss,
                "runtime_no_query_handoff": 0.0,
                "runtime_no_query_optimization": float(skip.runtime_seconds),
                "no_query_transition_mode": "native_optimizer_state",
                "action_loss_protocol": ACTION_LOSS_PROTOCOL,
            }
            frames.append(pd.DataFrame([{**common, **outcome} for outcome in outcomes]))
    finally:
        problem.close()

    combined = pd.concat(frames, ignore_index=True)
    _check_action_losses(combined)
    if set(combined.groupby("sample_design_id")["FE_query"].first().astype(int)) != {500, 1000}:
        raise ValueError("real 10D action-loss consistency run did not preserve 5% and 10% query budgets")
    for query_id in ("descriptor_cheap_invariant", "pflacco_broad_invariant"):
        try:
            prepare_state_matrix(
                combined,
                behavior=pd.DataFrame(),
                query_features=pd.DataFrame(),
                query_spec=get_query_spec(query_id),
            )
        except ValueError as exc:
            if "wrong query-FE action-loss table" not in str(exc):
                raise
        else:
            raise ValueError(f"{query_id} accepted mixed lhs_50d/lhs_100d action losses")
    print("real action-loss budgets remain separated: lhs_50d=500 FE, lhs_100d=1000 FE at 10D")


def _milestone_sampling_metadata(
    *,
    actual_fe: int,
    fe_total: int,
    monitor_target_ratio: float,
) -> dict:
    target = float(monitor_target_ratio)
    metadata = budget_milestone_metadata(target)
    if int(actual_fe) / int(fe_total) < target:
        raise ValueError("consistency sample FE must not precede its monitor target")
    if tuple(metadata) != SAMPLING_METADATA_COLUMNS:
        raise ValueError("consistency sampling metadata does not follow the frozen column order")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Check query-specific shared-state candidate actions on real BBOB.")
    parser.add_argument("--config", type=Path, default=Path("configs/phase1_bbob_train.yaml"))
    args = parser.parse_args()
    check_state_action_continuations(config_path=args.config)


if __name__ == "__main__":
    main()
