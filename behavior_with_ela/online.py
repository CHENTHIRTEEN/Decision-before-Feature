from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS, extract_behavior_rows
from behavior_with_ela.protocol import (
    ExperimentConfig,
    SuiteConfig,
    check_problem_availability,
    load_experiment_config,
    make_experiment_problem,
)
from optimizers import (
    NO_QUERY_TRANSFER_EVENT,
    OptimizerSettings,
    advance_optimizer_state,
    initialize_optimizer_state,
    initialize_transferred_optimizer_state,
)
from trajectory.recorder import TrajectoryRecorder


POLICY_PROTOCOL = "behavior_action_gain_one_switch_first_trigger"


def evaluate_one_switch_online(
    *,
    config_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    only_splits: tuple[str, ...] | None = None,
    only_functions: tuple[int, ...] | None = None,
    initial_algorithm: str = "sbs",
    workers: int = 1,
    overwrite: bool = False,
) -> dict[str, int]:
    config = load_experiment_config(config_path)
    bundle = joblib.load(model_path)
    _validate_model_bundle(bundle, config)
    prefixes = _initial_algorithms(initial_algorithm, bundle, config)
    suites = _selected_suites(config, only_splits)
    tasks = [
        (suite, function)
        for suite in suites
        for function in suite.functions
        if only_functions is None or function in set(only_functions)
    ]
    if not tasks:
        raise ValueError("no online-evaluation functions were selected")
    check_problem_availability(config, tasks)
    if workers < 1:
        raise ValueError("workers must be at least one")

    output = Path(output_dir)
    outcome_path = output / "online_policy_outcomes.parquet"
    opportunity_path = output / "online_opportunities.parquet"
    resource_path = output / "online_deployment_resources.parquet"
    paths = (outcome_path, opportunity_path, resource_path)
    if any(path.exists() for path in paths) and not overwrite:
        raise FileExistsError(f"online outputs already exist; pass --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in paths:
            path.unlink(missing_ok=True)

    function_results: list[tuple[list[dict], list[dict], list[dict]]] = []
    if workers == 1:
        for suite, function in tasks:
            function_results.append(
                _evaluate_function(config, bundle, suite, function, prefixes)
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_function,
                    config,
                    bundle,
                    suite,
                    function,
                    prefixes,
                )
                for suite, function in tasks
            ]
            for future in as_completed(futures):
                function_results.append(future.result())
    outcomes = [row for result in function_results for row in result[0]]
    opportunities = [row for result in function_results for row in result[1]]
    resources = [row for result in function_results for row in result[2]]
    if not outcomes:
        raise ValueError("online evaluation produced no policy outcomes")
    pd.DataFrame(outcomes).sort_values(
        ["split", "problem_id", "prefix_algorithm", "seed"],
        kind="mergesort",
    ).to_parquet(outcome_path, index=False)
    pd.DataFrame(opportunities).sort_values(
        [
            "split",
            "problem_id",
            "prefix_algorithm",
            "seed",
            "FE",
            "decision_opportunity_index",
            "candidate_action",
        ],
        kind="mergesort",
    ).to_parquet(opportunity_path, index=False)
    pd.DataFrame(resources).sort_values(
        ["split", "problem_id", "prefix_algorithm", "seed"],
        kind="mergesort",
    ).to_parquet(resource_path, index=False)
    return {
        "policy_runs": len(outcomes),
        "opportunity_action_rows": len(opportunities),
        "resource_rows": len(resources),
    }


def run_one_switch_policy(
    *,
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    instance: int,
    seed: int,
    prefix_algorithm: str,
    bundle: dict[str, Any],
) -> tuple[dict, list[dict], dict]:
    problem = make_experiment_problem(
        suite,
        function=function,
        instance=instance,
        dimension=config.dimension,
        boundary_handling=config.boundary_handling,
    )
    started = perf_counter()
    evaluation_count = 0
    first_hit_fe: int | None = None
    opportunity_rows: list[dict] = []
    recorder = TrajectoryRecorder(sampling_protocol=config.sampling_protocol)

    def observe_evaluation(point: np.ndarray, value: float) -> None:
        nonlocal evaluation_count, first_hit_fe
        evaluation_count += 1
        reference = problem.reference_value
        if reference is None:
            raise ValueError("online gap evaluation requires a benchmark reference value")
        gap = max(float(value) - float(reference), 0.0)
        if first_hit_fe is None and gap <= config.success_gap_target:
            first_hit_fe = evaluation_count

    status = "completed"
    failure_type = ""
    failure_message = ""
    switch_triggered = False
    selected_algorithm = prefix_algorithm
    selected_fe: int | None = None
    selected_opportunity: int | None = None
    selected_score: float | None = None
    state = None
    try:
        if problem.boundary_handling != config.boundary_handling:
            raise ValueError("online problem boundary handling differs from the experiment config")
        settings = OptimizerSettings(
            population_size=config.population_size,
            sampling_protocol=config.sampling_protocol,
            boundary_handling=config.boundary_handling,
        )
        state = initialize_optimizer_state(
            algorithm=prefix_algorithm,
            problem=problem,
            seed=seed,
            settings=settings,
            on_evaluation=observe_evaluation,
        )
        recorder.observe(
            problem=problem,
            algorithm=prefix_algorithm,
            seed=seed,
            fe=state.evaluations,
            fe_total=config.fe_total,
            native_updates=state.generation,
            population=state.population,
            fitness=state.fitness,
            best_fitness=state.best_fitness,
        )
        last_record_count = 0
        while state.evaluations < config.fe_total and not switch_triggered:
            step = min(
                config.population_size,
                config.fe_total - int(state.evaluations),
            )
            advance_optimizer_state(
                state=state,
                problem=problem,
                fe_budget=step,
                on_native_update=lambda updated: recorder.observe(
                    problem=problem,
                    algorithm=prefix_algorithm,
                    seed=seed,
                    fe=updated.evaluations,
                    fe_total=config.fe_total,
                    native_updates=updated.generation,
                    population=updated.population,
                    fitness=updated.fitness,
                    best_fitness=updated.best_fitness,
                ),
                on_evaluation=observe_evaluation,
            )
            if len(recorder.records) == last_record_count:
                continue
            if len(recorder.records) != last_record_count + 1:
                raise RuntimeError("one native update emitted multiple decision states")
            record = recorder.records[-1]
            opportunity_index = last_record_count
            last_record_count = len(recorder.records)
            behavior = extract_behavior_rows([asdict(record)])[0]
            scores, classes = predict_switch_scores(
                bundle=bundle,
                behavior=behavior,
                prefix_algorithm=prefix_algorithm,
            )
            candidates = [
                algorithm
                for algorithm in config.algorithms
                if algorithm != prefix_algorithm
            ]
            selected_candidate = max(
                candidates,
                key=lambda algorithm: (
                    float(scores[algorithm]),
                    -config.algorithms.index(algorithm),
                ),
            )
            for candidate in candidates:
                opportunity_rows.append(
                    {
                        "policy_protocol": POLICY_PROTOCOL,
                        "split": suite.split,
                        "suite": suite.suite,
                        "problem_id": problem.problem_id,
                        "function_id": problem.function_id,
                        "family": problem.family,
                        "cv_group_id": problem.cv_group_id,
                        "dimension": problem.dimension,
                        "prefix_algorithm": prefix_algorithm,
                        "seed": seed,
                        "FE": int(record.FE),
                        "FE_ratio": float(record.FE_ratio),
                        "decision_opportunity_index": int(opportunity_index),
                        "candidate_action": candidate,
                        "predicted_action_class": classes[candidate],
                        "predicted_improve_probability": float(scores[candidate]),
                        "decision_threshold": float(bundle["decision_threshold"]),
                        "candidate_ranked_first": bool(candidate == selected_candidate),
                        "would_trigger": bool(
                            candidate == selected_candidate
                            and float(scores[candidate])
                            > float(bundle["decision_threshold"])
                        ),
                    }
                )
            candidate_score = float(scores[selected_candidate])
            if candidate_score <= float(bundle["decision_threshold"]):
                continue
            selected_fe = int(record.FE)
            selected_opportunity = int(opportunity_index)
            selected_score = candidate_score
            selected_algorithm = selected_candidate
            switch_triggered = True
            state = initialize_transferred_optimizer_state(
                algorithm=selected_algorithm,
                source_state=state,
                problem=problem,
                seed=seed,
                function=function,
                instance=instance,
                event=NO_QUERY_TRANSFER_EVENT,
            )
            remaining = config.fe_total - selected_fe
            advance_optimizer_state(
                state=state,
                problem=problem,
                fe_budget=remaining,
                on_evaluation=observe_evaluation,
            )
        if not switch_triggered and state.evaluations < config.fe_total:
            advance_optimizer_state(
                state=state,
                problem=problem,
                fe_budget=config.fe_total - int(state.evaluations),
                on_evaluation=observe_evaluation,
            )
        if evaluation_count != config.fe_total:
            raise RuntimeError(
                f"online policy used {evaluation_count} FE, expected {config.fe_total}"
            )
        best_fitness = float(state.best_fitness)
        if not np.isfinite(best_fitness):
            raise FloatingPointError("online policy returned a non-finite best value")
        reference = problem.reference_value
        if reference is None:
            raise ValueError("online endpoint requires a benchmark reference value")
        final_gap = min(
            max(best_fitness - float(reference), 0.0),
            config.failure_loss_cap,
        )
    except Exception as exc:
        status = "failed"
        failure_type = type(exc).__name__
        failure_message = str(exc)[:500]
        best_fitness = (
            None
            if state is None or not np.isfinite(float(state.best_fitness))
            else float(state.best_fitness)
        )
        final_gap = config.failure_loss_cap
    finally:
        problem.close()
    elapsed = float(perf_counter() - started)
    selected_equals_prefix = selected_algorithm == prefix_algorithm
    outcome = {
        "policy_protocol": POLICY_PROTOCOL,
        "split": suite.split,
        "suite": suite.suite,
        "problem_id": problem.problem_id,
        "function_id": problem.function_id,
        "family": problem.family,
        "cv_group_id": problem.cv_group_id,
        "dimension": problem.dimension,
        "prefix_algorithm": prefix_algorithm,
        "seed": seed,
        "FE_total": config.fe_total,
        "policy_status": status,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "effective_FE": int(evaluation_count),
        "best_fitness": best_fitness,
        "benchmark_reference_value": problem.reference_value,
        "final_gap": float(final_gap),
        "log10_gap": float(
            np.log10(
                np.clip(
                    final_gap,
                    config.log10_gap_floor,
                    config.log10_gap_cap,
                )
            )
        ),
        "success": bool(
            status == "completed" and final_gap <= config.success_gap_target
        ),
        "first_hit_FE": first_hit_fe,
        "decision_opportunities_seen": int(
            0 if not opportunity_rows else 1 + max(row["decision_opportunity_index"] for row in opportunity_rows)
        ),
        "switch_triggered": bool(switch_triggered),
        "selected_algorithm": selected_algorithm,
        "selected_FE": selected_fe,
        "selected_decision_opportunity_index": selected_opportunity,
        "selected_score": selected_score,
        "decision_threshold": float(bundle["decision_threshold"]),
        "selected_equals_default": bool(
            selected_algorithm == str(bundle["default_algorithm"])
        ),
        "selected_equals_prefix": bool(selected_equals_prefix),
        "handoff_required": bool(not selected_equals_prefix),
        "handoff_type": (
            "native_optimizer_state"
            if selected_equals_prefix
            else "population_transfer_initialization"
        ),
        "boundary_handling": config.boundary_handling,
    }
    resources = {
        "split": suite.split,
        "problem_id": problem.problem_id,
        "prefix_algorithm": prefix_algorithm,
        "seed": seed,
        "policy_runtime_seconds": elapsed,
        "timing_source": "single_online_policy_execution_diagnostic",
        "timing_replay_status": status,
    }
    return outcome, opportunity_rows, resources


def predict_switch_scores(
    *,
    bundle: dict[str, Any],
    behavior: dict,
    prefix_algorithm: str,
) -> tuple[dict[str, float], dict[str, str]]:
    feature_columns = tuple(bundle["feature_columns"])
    if feature_columns != tuple(SELECTOR_BEHAVIOR_FEATURE_COLUMNS):
        raise ValueError("model feature columns differ from the 28-field Behavior contract")
    frame = pd.DataFrame(
        [{column: behavior[column] for column in feature_columns}]
    )
    scores: dict[str, float] = {}
    classes: dict[str, str] = {}
    for algorithm in bundle["portfolio"]:
        if algorithm == prefix_algorithm:
            continue
        model = bundle["models"][algorithm]
        classes[algorithm] = str(model.predict(frame)[0])
        probabilities = np.asarray(model.predict_proba(frame), dtype=float)
        labels = np.asarray(model.classes_).astype(str)
        matches = np.flatnonzero(labels == "improve")
        scores[algorithm] = (
            0.0 if len(matches) == 0 else float(probabilities[0, int(matches[0])])
        )
    return scores, classes


def _evaluate_function(
    config: ExperimentConfig,
    bundle: dict[str, Any],
    suite: SuiteConfig,
    function: int,
    prefixes: tuple[str, ...],
) -> tuple[list[dict], list[dict], list[dict]]:
    outcomes: list[dict] = []
    opportunities: list[dict] = []
    resources: list[dict] = []
    for instance in suite.instances:
        for seed in config.seeds:
            for prefix in prefixes:
                outcome, rows, resource = run_one_switch_policy(
                    config=config,
                    suite=suite,
                    function=function,
                    instance=instance,
                    seed=seed,
                    prefix_algorithm=prefix,
                    bundle=bundle,
                )
                outcomes.append(outcome)
                opportunities.extend(rows)
                resources.append(resource)
    return outcomes, opportunities, resources


def _validate_model_bundle(bundle: dict[str, Any], config: ExperimentConfig) -> None:
    if tuple(bundle.get("portfolio", ())) != config.algorithms:
        raise ValueError("model portfolio differs from the online experiment config")
    if tuple(bundle.get("feature_columns", ())) != tuple(
        SELECTOR_BEHAVIOR_FEATURE_COLUMNS
    ):
        raise ValueError("online model must use exactly 28 Behavior fields")
    for name in ("dimension", "FE_total", "population_size"):
        config_value = {
            "dimension": config.dimension,
            "FE_total": config.fe_total,
            "population_size": config.population_size,
        }[name]
        if int(bundle.get(name, -1)) != int(config_value):
            raise ValueError(f"model {name} differs from online config")
    if str(bundle.get("sampling_protocol")) != config.sampling_protocol:
        raise ValueError("model sampling protocol differs from online config")
    if str(bundle.get("boundary_handling")) != config.boundary_handling:
        raise ValueError("model boundary handling differs from online config")
    if not np.isfinite(float(bundle.get("decision_threshold", np.nan))):
        raise ValueError("model decision threshold must be finite")
    if not np.isfinite(float(bundle.get("practical_gain_delta", np.nan))):
        raise ValueError("model practical gain threshold must be finite")


def _initial_algorithms(
    value: str,
    bundle: dict[str, Any],
    config: ExperimentConfig,
) -> tuple[str, ...]:
    name = str(value).lower()
    if name == "sbs":
        selected = str(bundle["default_algorithm"])
        if selected not in config.algorithms:
            raise ValueError("trained SBS is outside the online portfolio")
        return (selected,)
    if name == "all":
        return config.algorithms
    if name not in config.algorithms:
        raise ValueError(f"initial algorithm must be sbs, all, or one of {config.algorithms}")
    return (name,)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Behavior action model as an online one-switch policy."
    )
    parser.add_argument("--config", default="configs/behavior_with_ela_cec2017.yaml")
    parser.add_argument(
        "--model",
        default="results/behavior_with_ela/model/behavior_action_gain/models.joblib",
    )
    parser.add_argument(
        "--output",
        default="results/behavior_with_ela/online/cec2017",
    )
    parser.add_argument("--only-split", action="append", default=None)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--initial-algorithm", default="sbs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = evaluate_one_switch_online(
        config_path=args.config,
        model_path=args.model,
        output_dir=args.output,
        only_splits=None if args.only_split is None else tuple(args.only_split),
        only_functions=(
            None if args.only_function is None else tuple(args.only_function)
        ),
        initial_algorithm=args.initial_algorithm,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(
        f"completed {summary['policy_runs']} online policy runs with "
        f"{summary['opportunity_action_rows']} opportunity-action predictions"
    )


if __name__ == "__main__":
    main()
