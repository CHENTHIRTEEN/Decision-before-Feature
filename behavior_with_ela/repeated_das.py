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

from behavior.features import extract_behavior_rows
from behavior_with_ela.local_landscape import LocalLandscapeRecorder
from behavior_with_ela.phase2 import PHASE2_FEATURE_GROUPS
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


REPEATED_DAS_PROTOCOL = "supervised_repeated_das_population_transfer_v1"
PHASE2_ONE_SWITCH_PROTOCOL = "supervised_m4_one_switch_population_transfer_v1"
REPEATED_FEATURE_GROUP = "M4_behavior_local_landscape_uncertainty"


def evaluate_repeated_das(
    *,
    config_path: str | Path,
    phase2_model_path: str | Path,
    output_dir: str | Path,
    only_splits: tuple[str, ...] | None = None,
    only_functions: tuple[int, ...] | None = None,
    initial_algorithm: str = "sbs",
    workers: int = 1,
    overwrite: bool = False,
) -> dict[str, int]:
    config = load_experiment_config(config_path)
    bundle = joblib.load(phase2_model_path)
    _validate_bundle(bundle, config)
    prefixes = _initial_algorithms(initial_algorithm, bundle, config)
    suites = _selected_suites(config, only_splits)
    tasks = [
        (suite, function)
        for suite in suites
        for function in suite.functions
        if only_functions is None or function in set(only_functions)
    ]
    if not tasks:
        raise ValueError("no repeated-DAS functions were selected")
    check_problem_availability(config, tasks)
    if workers < 1:
        raise ValueError("workers must be at least one")

    output = Path(output_dir)
    outcome_path = output / "repeated_das_outcomes.parquet"
    opportunity_path = output / "repeated_das_opportunities.parquet"
    switch_path = output / "repeated_das_switches.parquet"
    resource_path = output / "repeated_das_resources.parquet"
    paths = (outcome_path, opportunity_path, switch_path, resource_path)
    if any(path.exists() for path in paths) and not overwrite:
        raise FileExistsError(f"repeated-DAS outputs already exist; pass --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in paths:
            path.unlink(missing_ok=True)

    results: list[tuple[list[dict], list[dict], list[dict], list[dict]]] = []
    if workers == 1:
        for suite, function in tasks:
            results.append(_evaluate_function(config, bundle, suite, function, prefixes))
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
                results.append(future.result())
    outcomes = [row for result in results for row in result[0]]
    opportunities = [row for result in results for row in result[1]]
    switches = [row for result in results for row in result[2]]
    resources = [row for result in results for row in result[3]]
    pd.DataFrame(outcomes).sort_values(
        ["split", "problem_id", "prefix_algorithm", "seed"],
        kind="mergesort",
    ).to_parquet(outcome_path, index=False)
    pd.DataFrame(opportunities).sort_values(
        ["split", "problem_id", "prefix_algorithm", "seed", "FE", "candidate_action"],
        kind="mergesort",
    ).to_parquet(opportunity_path, index=False)
    switch_frame = pd.DataFrame(switches)
    if switch_frame.empty:
        switch_frame = pd.DataFrame(
            columns=[
                "policy_protocol",
                "split",
                "problem_id",
                "prefix_algorithm",
                "initial_algorithm",
                "seed",
                "switch_index",
                "FE",
                "decision_opportunity_index",
                "source_algorithm",
                "selected_algorithm",
                "selected_equals_default",
                "selected_equals_prefix",
                "handoff_required",
                "handoff_type",
                "predicted_improve_probability",
                "top2_probability_margin",
                "dwell_FE",
            ]
        )
    switch_frame.to_parquet(switch_path, index=False)
    pd.DataFrame(resources).to_parquet(resource_path, index=False)
    return {
        "policy_runs": len(outcomes),
        "opportunity_action_rows": len(opportunities),
        "switch_rows": len(switches),
    }


def run_repeated_das_policy(
    *,
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    instance: int,
    seed: int,
    prefix_algorithm: str,
    bundle: dict[str, Any],
) -> tuple[dict, list[dict], list[dict], dict]:
    return _run_m4_policy(
        config=config,
        suite=suite,
        function=function,
        instance=instance,
        seed=seed,
        prefix_algorithm=prefix_algorithm,
        bundle=bundle,
        repeated=True,
    )


def run_phase2_one_switch_policy(
    *,
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    instance: int,
    seed: int,
    prefix_algorithm: str,
    bundle: dict[str, Any],
) -> tuple[dict, list[dict], list[dict], dict]:
    return _run_m4_policy(
        config=config,
        suite=suite,
        function=function,
        instance=instance,
        seed=seed,
        prefix_algorithm=prefix_algorithm,
        bundle=bundle,
        repeated=False,
    )


def _run_m4_policy(
    *,
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    instance: int,
    seed: int,
    prefix_algorithm: str,
    bundle: dict[str, Any],
    repeated: bool,
) -> tuple[dict, list[dict], list[dict], dict]:
    policy_protocol = (
        REPEATED_DAS_PROTOCOL if repeated else PHASE2_ONE_SWITCH_PROTOCOL
    )
    max_switches = config.repeated_das.max_switches if repeated else 1
    minimum_dwell_fe = config.repeated_das.minimum_dwell_FE if repeated else 0
    hysteresis_margin = (
        config.repeated_das.hysteresis_probability_margin if repeated else 0.0
    )
    problem = make_experiment_problem(
        suite,
        function=function,
        instance=instance,
        dimension=config.dimension,
        boundary_handling=config.boundary_handling,
    )
    started = perf_counter()
    evaluation_count = 0
    global_native_updates = 0
    first_hit_fe: int | None = None
    current_algorithm = prefix_algorithm
    segment_start_fe = 0
    switch_count = 0
    first_switch_fe: int | None = None
    opportunities: list[dict] = []
    switches: list[dict] = []
    recorder = TrajectoryRecorder(sampling_protocol=config.sampling_protocol)
    local = LocalLandscapeRecorder(
        config=config.local_landscape,
        lower_bounds=problem.lower_bounds,
        upper_bounds=problem.upper_bounds,
        seed=seed,
        suite=suite.suite,
        function=function,
        instance=instance,
        dimension=config.dimension,
        algorithm=prefix_algorithm,
    )

    def observe_evaluation(point: np.ndarray, value: float) -> None:
        nonlocal evaluation_count, first_hit_fe
        evaluation_count += 1
        local.observe(point, value)
        reference = problem.reference_value
        if reference is None:
            raise ValueError("repeated DAS requires a benchmark reference value")
        gap = max(float(value) - float(reference), 0.0)
        if first_hit_fe is None and gap <= config.success_gap_target:
            first_hit_fe = evaluation_count

    status = "completed"
    failure_type = ""
    failure_message = ""
    state = None
    try:
        settings = OptimizerSettings(
            population_size=config.population_size,
            sampling_protocol=config.sampling_protocol,
            boundary_handling=config.boundary_handling,
        )
        state = initialize_optimizer_state(
            algorithm=current_algorithm,
            problem=problem,
            seed=seed,
            settings=settings,
            on_evaluation=observe_evaluation,
        )
        recorder.observe(
            problem=problem,
            algorithm=current_algorithm,
            seed=seed,
            fe=evaluation_count,
            fe_total=config.fe_total,
            native_updates=global_native_updates,
            population=state.population,
            fitness=state.fitness,
            best_fitness=state.best_fitness,
        )
        emitted_count = 0
        while evaluation_count < config.fe_total:
            previous_records = len(recorder.records)

            def observe_update(updated) -> None:
                nonlocal global_native_updates
                global_native_updates += 1
                recorder.observe(
                    problem=problem,
                    algorithm=current_algorithm,
                    seed=seed,
                    fe=evaluation_count,
                    fe_total=config.fe_total,
                    native_updates=global_native_updates,
                    population=updated.population,
                    fitness=updated.fitness,
                    best_fitness=updated.best_fitness,
                )

            advance_optimizer_state(
                state=state,
                problem=problem,
                fe_budget=min(
                    config.population_size,
                    config.fe_total - evaluation_count,
                ),
                on_native_update=observe_update,
                on_evaluation=observe_evaluation,
            )
            if len(recorder.records) == previous_records:
                continue
            if len(recorder.records) != previous_records + 1:
                raise RuntimeError("one update emitted multiple repeated-DAS states")
            record = recorder.records[-1]
            opportunity_index = emitted_count
            emitted_count += 1
            landscape = local.snapshot(
                split=suite.split,
                problem_id=record.problem_id,
                function_id=record.function_id,
                family=record.family,
                cv_group_id=record.cv_group_id,
                fe=record.FE,
                fe_total=record.FE_total,
                native_updates=record.native_updates,
                decision_opportunity_index=opportunity_index,
            )
            behavior = extract_behavior_rows([asdict(record)])[0]
            features = {**behavior, **landscape}
            scores, classes = _predict_action_scores(
                bundle=bundle,
                features=features,
                current_algorithm=current_algorithm,
            )
            candidates = [name for name in config.algorithms if name != current_algorithm]
            ranked = sorted(
                candidates,
                key=lambda name: (-scores[name], config.algorithms.index(name)),
            )
            selected = ranked[0]
            top_score = float(scores[selected])
            second_score = float(scores[ranked[1]])
            margin = top_score - second_score
            dwell_fe = int(record.FE) - segment_start_fe
            dwell_ok = dwell_fe >= minimum_dwell_fe
            threshold_ok = top_score > float(
                bundle["thresholds"][REPEATED_FEATURE_GROUP]
            )
            hysteresis_ok = bool(not repeated or margin > hysteresis_margin)
            switch_limit_ok = switch_count < max_switches
            should_switch = bool(
                dwell_ok and threshold_ok and hysteresis_ok and switch_limit_ok
            )
            for candidate in candidates:
                opportunities.append(
                    {
                        "policy_protocol": policy_protocol,
                        "split": suite.split,
                        "suite": suite.suite,
                        "problem_id": problem.problem_id,
                        "function_id": problem.function_id,
                        "family": problem.family,
                        "cv_group_id": problem.cv_group_id,
                        "dimension": problem.dimension,
                        "prefix_algorithm": prefix_algorithm,
                        "current_algorithm": current_algorithm,
                        "seed": seed,
                        "FE": int(record.FE),
                        "FE_ratio": float(record.FE_ratio),
                        "decision_opportunity_index": opportunity_index,
                        "candidate_action": candidate,
                        "predicted_action_class": classes[candidate],
                        "predicted_improve_probability": float(scores[candidate]),
                        "decision_threshold": float(
                            bundle["thresholds"][REPEATED_FEATURE_GROUP]
                        ),
                        "top2_probability_margin": float(margin),
                        "hysteresis_probability_margin": hysteresis_margin,
                        "dwell_FE": dwell_fe,
                        "minimum_dwell_FE": minimum_dwell_fe,
                        "switch_count_before_state": switch_count,
                        "candidate_ranked_first": bool(candidate == selected),
                        "dwell_eligible": bool(dwell_ok),
                        "threshold_eligible": bool(threshold_ok),
                        "hysteresis_eligible": bool(hysteresis_ok),
                        "switch_limit_eligible": bool(switch_limit_ok),
                        "would_switch": bool(should_switch and candidate == selected),
                    }
                )
            if not should_switch:
                continue
            previous_algorithm = current_algorithm
            current_algorithm = selected
            if first_switch_fe is None:
                first_switch_fe = int(record.FE)
            switches.append(
                {
                    "policy_protocol": policy_protocol,
                    "split": suite.split,
                    "problem_id": problem.problem_id,
                    "prefix_algorithm": previous_algorithm,
                    "initial_algorithm": prefix_algorithm,
                    "seed": seed,
                    "switch_index": switch_count + 1,
                    "FE": int(record.FE),
                    "decision_opportunity_index": opportunity_index,
                    "source_algorithm": previous_algorithm,
                    "selected_algorithm": current_algorithm,
                    "selected_equals_default": bool(
                        current_algorithm == str(bundle["default_algorithm"])
                    ),
                    "selected_equals_prefix": False,
                    "handoff_required": True,
                    "handoff_type": "population_transfer_initialization",
                    "predicted_improve_probability": top_score,
                    "top2_probability_margin": margin,
                    "dwell_FE": dwell_fe,
                }
            )
            state = initialize_transferred_optimizer_state(
                algorithm=current_algorithm,
                source_state=state,
                problem=problem,
                seed=seed,
                function=function,
                instance=instance,
                event=NO_QUERY_TRANSFER_EVENT + switch_count,
            )
            switch_count += 1
            segment_start_fe = int(record.FE)
            if not repeated:
                advance_optimizer_state(
                    state=state,
                    problem=problem,
                    fe_budget=config.fe_total - evaluation_count,
                    on_evaluation=observe_evaluation,
                )
                break
        if evaluation_count != config.fe_total:
            raise RuntimeError("repeated DAS did not use the strict total FE budget")
        best_fitness = float(state.best_fitness)
        reference = problem.reference_value
        if reference is None:
            raise ValueError("repeated DAS endpoint requires a reference value")
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
    final_equals_prefix = current_algorithm == prefix_algorithm
    outcome = {
        "policy_protocol": policy_protocol,
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
        "effective_FE": evaluation_count,
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
        "success": bool(status == "completed" and final_gap <= config.success_gap_target),
        "first_hit_FE": first_hit_fe,
        "switch_count": switch_count,
        "first_switch_FE": first_switch_fe,
        "final_algorithm": current_algorithm,
        "selected_algorithm": current_algorithm,
        "selected_equals_default": bool(
            current_algorithm == str(bundle["default_algorithm"])
        ),
        "selected_equals_prefix": bool(final_equals_prefix),
        "handoff_required": bool(not final_equals_prefix),
        "handoff_type": (
            "native_optimizer_state"
            if final_equals_prefix
            else "population_transfer_initialization"
        ),
        "any_handoff_performed": bool(switch_count > 0),
        "total_query_FE": 0,
        "query_count": 0,
        "max_switches": max_switches,
        "minimum_dwell_FE": minimum_dwell_fe,
        "hysteresis_probability_margin": hysteresis_margin,
        "context_restoration": "not_used_population_transfer_only",
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
    return outcome, opportunities, switches, resources


def _predict_action_scores(
    *,
    bundle: dict[str, Any],
    features: dict,
    current_algorithm: str,
) -> tuple[dict[str, float], dict[str, str]]:
    columns = tuple(bundle["feature_groups"][REPEATED_FEATURE_GROUP])
    frame = pd.DataFrame([{column: features[column] for column in columns}])
    scores: dict[str, float] = {}
    classes: dict[str, str] = {}
    for algorithm in bundle["portfolio"]:
        if algorithm == current_algorithm:
            continue
        model = bundle["models"][REPEATED_FEATURE_GROUP][algorithm]
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
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    outcomes = []
    opportunities = []
    switches = []
    resources = []
    for instance in suite.instances:
        for seed in config.seeds:
            for prefix in prefixes:
                outcome, state_rows, switch_rows, resource = run_repeated_das_policy(
                    config=config,
                    suite=suite,
                    function=function,
                    instance=instance,
                    seed=seed,
                    prefix_algorithm=prefix,
                    bundle=bundle,
                )
                outcomes.append(outcome)
                opportunities.extend(state_rows)
                switches.extend(switch_rows)
                resources.append(resource)
    return outcomes, opportunities, switches, resources


def _validate_bundle(bundle: dict[str, Any], config: ExperimentConfig) -> None:
    if tuple(bundle.get("portfolio", ())) != config.algorithms:
        raise ValueError("Phase 2 model portfolio differs from repeated-DAS config")
    if REPEATED_FEATURE_GROUP not in bundle.get("models", {}):
        raise ValueError("Phase 2 model bundle is missing the M4 feature group")
    if tuple(bundle["feature_groups"][REPEATED_FEATURE_GROUP]) != tuple(
        PHASE2_FEATURE_GROUPS[REPEATED_FEATURE_GROUP]
    ):
        raise ValueError("repeated-DAS M4 feature columns differ from Phase 2")
    for name, value in (
        ("dimension", config.dimension),
        ("FE_total", config.fe_total),
        ("population_size", config.population_size),
    ):
        if int(bundle.get(name, -1)) != int(value):
            raise ValueError(f"Phase 2 model {name} differs from repeated-DAS config")
    if str(bundle.get("sampling_protocol")) != config.sampling_protocol:
        raise ValueError("Phase 2 sampling protocol differs from repeated DAS")
    if str(bundle.get("boundary_handling")) != config.boundary_handling:
        raise ValueError("Phase 2 boundary handling differs from repeated DAS")
    if bundle.get("local_landscape_config") != config.local_landscape:
        raise ValueError("Phase 2 local-landscape config differs from repeated DAS")
    if not np.isfinite(float(bundle.get("practical_gain_delta", np.nan))):
        raise ValueError("Phase 2 practical gain threshold must be finite")


def _initial_algorithms(
    value: str,
    bundle: dict[str, Any],
    config: ExperimentConfig,
) -> tuple[str, ...]:
    name = str(value).lower()
    if name == "sbs":
        return (str(bundle["default_algorithm"]),)
    if name == "all":
        return config.algorithms
    if name not in config.algorithms:
        raise ValueError("initial algorithm must be sbs, all, or a portfolio algorithm")
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
        description="Evaluate supervised repeated DAS with dwell and hysteresis."
    )
    parser.add_argument("--config", default="configs/behavior_with_ela_cec2017.yaml")
    parser.add_argument(
        "--phase2-model",
        default="results/behavior_with_ela/model/local_landscape_increment/phase2_models.joblib",
    )
    parser.add_argument(
        "--output",
        default="results/behavior_with_ela/online/repeated_das",
    )
    parser.add_argument("--only-split", action="append", default=None)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--initial-algorithm", default="sbs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = evaluate_repeated_das(
        config_path=args.config,
        phase2_model_path=args.phase2_model,
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
        f"completed {summary['policy_runs']} repeated-DAS runs with "
        f"{summary['switch_rows']} switches"
    )


if __name__ == "__main__":
    main()
