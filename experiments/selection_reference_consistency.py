from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter

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
from optimizers import OptimizerSettings, advance_optimizer_state, clone_optimizer_state, initialize_optimizer_state
from selection_reference.action_losses import ACTION_LOSS_PROTOCOL, evaluate_candidate_actions
from selection_reference.build import build_selection_reference
from selection_reference.model import (
    fit_selector_with_cross_family_predictions,
    load_selector_model,
    measure_online_selection_runtime,
    prepare_state_matrix,
    selection_rows,
)
from trajectory.records import TrajectoryRecord
from utility_labels.generation import _utility_row, utility_schema
from utility_labels.validation import validate_utility_label_file


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
            )
            native = [row for row in outcomes if row["action"] == "continue_current"]
            if len(native) != 1 or float(native[0]["action_loss"]) != float(expected_native.best_fitness):
                raise ValueError("continue_current does not reproduce native optimizer continuation")
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
    spec = get_query_spec("descriptor_cheap")
    sample_design = spec.sample_design
    prefix_algorithm = portfolio[0]
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
            state = initialize_optimizer_state(
                algorithm=prefix_algorithm,
                problem=problem,
                seed=seed,
                settings=settings,
            )
            trajectory_rows = []
            for target_fe in (int(0.20 * fe_total), int(0.25 * fe_total), int(0.30 * fe_total)):
                advance_optimizer_state(state=state, problem=problem, fe_budget=target_fe - int(state.evaluations))
                trajectory_rows.append(
                    TrajectoryRecord.from_arrays(
                        problem_id=problem.problem_id,
                        family=problem.family,
                        dimension=dimension,
                        algorithm=prefix_algorithm,
                        seed=seed,
                        fe=int(state.evaluations),
                        fe_total=fe_total,
                        native_updates=int(state.generation),
                        population=state.population,
                        fitness=state.fitness,
                        best_fitness=state.best_fitness,
                    ).__dict__
                )
            behavior_rows.append({"split": "bbob_train", **extract_behavior_rows(trajectory_rows)[-1]})
            sample = sample_problem(
                problem=problem,
                sample_design=sample_design,
                base_seed=0,
                function=function,
                instance=instance,
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
                    "sample_design_id": spec.sample_design_id,
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
            fe_query = sample_design.sample_size(dimension)
            query_budget = fe_total - int(state.evaluations) - fe_query
            skip_started = perf_counter()
            skip = advance_optimizer_state(
                state=clone_optimizer_state(state),
                problem=problem,
                fe_budget=fe_total - int(state.evaluations),
            )
            runtime_skip = perf_counter() - skip_started
            outcomes = evaluate_candidate_actions(
                checkpoint_state=state,
                problem=problem,
                portfolio=portfolio,
                fe_budget=query_budget,
                seed=seed,
                function=function,
                instance=instance,
            )
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
                "sample_design_id": spec.sample_design_id,
                "sample_design_protocol": sample_design.protocol,
                "FE_query": fe_query,
                "FE_no_query_optimization": fe_total - int(state.evaluations),
                "FE_query_optimization": query_budget,
                "remaining_budget_ratio": float(query_budget / fe_total),
                "p_skip": float(skip.best_fitness),
                "runtime_no_query_optimization": float(runtime_skip),
                "no_query_transition_mode": "native_optimizer_state",
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
    selector, predictions, source = fit_selector_with_cross_family_predictions(states, observed_portfolio, spec)
    reference = selection_rows(
        states=states,
        portfolio=observed_portfolio,
        predictions=predictions,
        prediction_source=source,
        runtime_selection=measure_online_selection_runtime(selector, states),
    )
    if len(reference) != 2 or source != "cross_family":
        raise ValueError("query-specific selector did not produce cross-family predictions")
    with tempfile.TemporaryDirectory(prefix="decision-before-feature-query-check-") as directory:
        root = Path(directory)
        action_path = root / "action_losses.parquet"
        behavior_path = root / "behavior.parquet"
        feature_path = root / "features.parquet"
        reference_path = root / "selection_reference.parquet"
        model_path = root / "statewise_selector.joblib"
        pq.write_table(pa.Table.from_pandas(action_frame, preserve_index=False), action_path)
        pq.write_table(pa.Table.from_pandas(behavior_frame, preserve_index=False), behavior_path)
        pq.write_table(pa.Table.from_pandas(query_frame, preserve_index=False), feature_path)
        summary = build_selection_reference(
            query_id=spec.query_id,
            train_action_loss_paths=[action_path],
            predict_action_loss_paths=[],
            behavior_paths=[behavior_path],
            query_feature_paths=[feature_path],
            output_path=reference_path,
            model_output_path=model_path,
        )
        if int(summary["rows"]) != 2 or not model_path.exists():
            raise ValueError("query-specific selection-reference artifacts are inconsistent")
        loaded_selector = load_selector_model(model_path)
        if loaded_selector.sample_design_id != spec.sample_design_id:
            raise ValueError("saved selector model lost its query sample design")
        label = _utility_row(row=pq.read_table(reference_path).to_pylist()[0], query_id=spec.query_id)
        utility_path = root / "utility_labels.parquet"
        pq.write_table(pa.Table.from_pylist([label], schema=utility_schema()), utility_path)
        validate_utility_label_file(utility_path)
    print("query-specific action-loss regression and utility consistency passed on 2 BBOB families")


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
        for sample_design_id in ("lhs_50d", "lhs_100d"):
            design = get_sample_design_spec(sample_design_id)
            fe_query = design.sample_size(dimension)
            query_budget = fe_total - int(state.evaluations) - fe_query
            skip_started = perf_counter()
            skip = advance_optimizer_state(
                state=clone_optimizer_state(state),
                problem=problem,
                fe_budget=fe_total - int(state.evaluations),
            )
            runtime_skip = perf_counter() - skip_started
            outcomes = evaluate_candidate_actions(
                checkpoint_state=state,
                problem=problem,
                portfolio=portfolio,
                fe_budget=query_budget,
                seed=seed,
                function=function,
                instance=instance,
            )
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
                "sample_design_id": sample_design_id,
                "sample_design_protocol": design.protocol,
                "FE_query": fe_query,
                "FE_no_query_optimization": fe_total - int(state.evaluations),
                "FE_query_optimization": query_budget,
                "remaining_budget_ratio": float(query_budget / fe_total),
                "p_skip": float(skip.best_fitness),
                "runtime_no_query_optimization": float(runtime_skip),
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
    for query_id in ("descriptor_cheap", "pflacco_broad"):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Check query-specific shared-state candidate actions on real BBOB.")
    parser.add_argument("--config", type=Path, default=Path("configs/phase1_bbob_train.yaml"))
    args = parser.parse_args()
    check_state_action_continuations(config_path=args.config)


if __name__ == "__main__":
    main()
