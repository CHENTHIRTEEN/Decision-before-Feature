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
from behavior_with_ela.action_dataset import GFE_GATE_BEHAVIOR_FEATURE_COLUMNS
from behavior_with_ela.gfe_query_gate import validate_gfe_bundle
from behavior_with_ela.local_landscape import LocalLandscapeRecorder
from behavior_with_ela.phase3 import (
    QUERY_DESCRIPTOR_FEATURES,
    QUERY_NO_DESCRIPTOR_FEATURES,
    VOI_FEATURE_COLUMNS,
)
from behavior_with_ela.protocol import (
    ExperimentConfig,
    SuiteConfig,
    check_problem_availability,
    load_experiment_config,
    make_experiment_problem,
)
from landscape_queries.cheap import calculate_descriptor_cheap
from landscape_queries.sampling import sample_problem
from landscape_queries.specs import get_query_spec
from optimizers import (
    NO_QUERY_TRANSFER_EVENT,
    OptimizerSettings,
    advance_optimizer_state,
    initialize_optimizer_state,
    initialize_transferred_optimizer_state,
)
from trajectory.recorder import TrajectoryRecorder


ADAPTIVE_QUERY_ONLINE_PROTOCOL = "adaptive_query_one_switch_online_v1"
GFE_QUERY_POLICIES = (
    "g_fe_query_gate",
    "g_fe_query_gate_matched_rate",
)
QUERY_POLICIES = (
    "voi_query",
    "never_query",
    "always_query",
    "fixed_030_query",
    "uncertainty_query",
    *GFE_QUERY_POLICIES,
)


def evaluate_adaptive_query_online(
    *,
    config_path: str | Path,
    phase3_model_path: str | Path,
    gfe_model_path: str | Path | None = None,
    output_dir: str | Path,
    query_policies: tuple[str, ...] = QUERY_POLICIES,
    only_splits: tuple[str, ...] | None = None,
    only_functions: tuple[int, ...] | None = None,
    initial_algorithm: str = "sbs",
    workers: int = 1,
    overwrite: bool = False,
) -> dict[str, int]:
    config = load_experiment_config(config_path)
    bundle = joblib.load(phase3_model_path)
    _validate_bundle(bundle, config)
    policies = tuple(str(value) for value in query_policies)
    if not policies or any(value not in QUERY_POLICIES for value in policies):
        raise ValueError(f"query policies must belong to {QUERY_POLICIES}")
    gfe_bundle = None
    if set(policies).intersection(GFE_QUERY_POLICIES):
        if gfe_model_path is None or not Path(gfe_model_path).exists():
            raise FileNotFoundError("G_FE Query Gate policy requires its retrained model")
        gfe_bundle = joblib.load(gfe_model_path)
        validate_gfe_bundle(gfe_bundle, config)
        if str(gfe_bundle["default_algorithm"]) != str(bundle["default_algorithm"]):
            raise ValueError("G_FE Gate and Phase 3 use different default algorithms")
    prefixes = _initial_algorithms(initial_algorithm, bundle, config)
    suites = _selected_suites(config, only_splits)
    tasks = [
        (suite, function)
        for suite in suites
        for function in suite.functions
        if only_functions is None or function in set(only_functions)
    ]
    if not tasks:
        raise ValueError("no adaptive-Query functions were selected")
    check_problem_availability(config, tasks)
    if workers < 1:
        raise ValueError("workers must be at least one")

    output = Path(output_dir)
    outcome_path = output / "adaptive_query_online_outcomes.parquet"
    opportunity_path = output / "adaptive_query_online_opportunities.parquet"
    resource_path = output / "adaptive_query_online_resources.parquet"
    paths = (outcome_path, opportunity_path, resource_path)
    if any(path.exists() for path in paths) and not overwrite:
        raise FileExistsError(f"adaptive-Query outputs already exist; pass --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in paths:
            path.unlink(missing_ok=True)

    results: list[tuple[list[dict], list[dict], list[dict]]] = []
    if workers == 1:
        for suite, function in tasks:
            results.append(
                _evaluate_function(
                    config,
                    bundle,
                    gfe_bundle,
                    suite,
                    function,
                    prefixes,
                    policies,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_function,
                    config,
                    bundle,
                    gfe_bundle,
                    suite,
                    function,
                    prefixes,
                    policies,
                )
                for suite, function in tasks
            ]
            for future in as_completed(futures):
                results.append(future.result())
    outcomes = [row for result in results for row in result[0]]
    opportunities = [row for result in results for row in result[1]]
    resources = [row for result in results for row in result[2]]
    pd.DataFrame(outcomes).sort_values(
        ["split", "problem_id", "prefix_algorithm", "seed", "query_policy"],
        kind="mergesort",
    ).to_parquet(outcome_path, index=False)
    pd.DataFrame(opportunities).sort_values(
        [
            "split",
            "problem_id",
            "prefix_algorithm",
            "seed",
            "query_policy",
            "FE",
            "candidate_action",
        ],
        kind="mergesort",
    ).to_parquet(opportunity_path, index=False)
    pd.DataFrame(resources).to_parquet(resource_path, index=False)
    return {
        "policy_runs": len(outcomes),
        "opportunity_action_rows": len(opportunities),
        "resource_rows": len(resources),
    }


def run_adaptive_query_policy(
    *,
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    instance: int,
    seed: int,
    prefix_algorithm: str,
    bundle: dict[str, Any],
    gfe_bundle: dict[str, Any] | None,
    query_policy: str,
) -> tuple[dict, list[dict], dict]:
    if query_policy not in QUERY_POLICIES:
        raise ValueError(f"unknown adaptive Query policy: {query_policy}")
    if query_policy in GFE_QUERY_POLICIES:
        if gfe_bundle is None:
            raise ValueError("G_FE Query Gate policy requires its model bundle")
        validate_gfe_bundle(gfe_bundle, config)
        if str(gfe_bundle["default_algorithm"]) != str(bundle["default_algorithm"]):
            raise ValueError("G_FE Gate and Phase 3 use different default algorithms")
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
    opportunities: list[dict] = []
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
            raise ValueError("adaptive Query requires a benchmark reference value")
        gap = max(float(value) - float(reference), 0.0)
        if first_hit_fe is None and gap <= config.success_gap_target:
            first_hit_fe = evaluation_count

    status = "completed"
    failure_type = ""
    failure_message = ""
    state = None
    selected_algorithm = prefix_algorithm
    selected_fe: int | None = None
    selected_opportunity: int | None = None
    query_triggered = False
    query_fe = 0
    query_best_gap = config.failure_loss_cap
    query_sample_status = "not_run"
    query_runtime_seconds = 0.0
    action_selected = False
    try:
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
        emitted_count = 0
        while evaluation_count < config.fe_total and not action_selected:
            previous_records = len(recorder.records)
            advance_optimizer_state(
                state=state,
                problem=problem,
                fe_budget=min(
                    config.population_size,
                    config.fe_total - evaluation_count,
                ),
                on_native_update=lambda updated: recorder.observe(
                    problem=problem,
                    algorithm=prefix_algorithm,
                    seed=seed,
                    fe=evaluation_count,
                    fe_total=config.fe_total,
                    native_updates=updated.generation,
                    population=updated.population,
                    fitness=updated.fitness,
                    best_fitness=updated.best_fitness,
                ),
                on_evaluation=observe_evaluation,
            )
            if len(recorder.records) == previous_records:
                continue
            if len(recorder.records) != previous_records + 1:
                raise RuntimeError("one update emitted multiple adaptive-Query states")
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
            pre_scores, pre_classes = _predict_action_scores(
                models=bundle["pre_action_models"],
                feature_columns=QUERY_NO_DESCRIPTOR_FEATURES,
                features=features,
                current_algorithm=prefix_algorithm,
                portfolio=config.algorithms,
            )
            ranked = _rank_candidates(pre_scores, config.algorithms, prefix_algorithm)
            pre_selected = ranked[0]
            pre_top = float(pre_scores[pre_selected])
            pre_second = float(pre_scores[ranked[1]])
            pre_margin = pre_top - pre_second
            pre_entropy = _mean_binary_entropy(list(pre_scores.values()))
            local_uncertainty = float(
                np.nanmean(
                    [
                        features[column]
                        for column in QUERY_NO_DESCRIPTOR_FEATURES
                        if column.endswith("_bootstrap_std")
                    ]
                )
            )
            voi_features = {
                **features,
                "pre_top_improve_probability": pre_top,
                "pre_top2_probability_margin": pre_margin,
                "pre_mean_binary_action_entropy": pre_entropy,
                "pre_local_uncertainty_mean": local_uncertainty,
            }
            voi_score, voi_class = _predict_voi(bundle, voi_features)
            gfe_score = (
                _predict_gfe(gfe_bundle, features)
                if query_policy in GFE_QUERY_POLICIES
                else None
            )
            query_eligible = (
                config.fe_total - int(record.FE)
                >= get_query_spec(config.query.query_id).sample_design.sample_size(
                    config.dimension
                )
                + config.query.minimum_post_query_FE
            )
            policy_requests_query = _policy_requests_query(
                query_policy=query_policy,
                record=record,
                opportunity_index=opportunity_index,
                voi_score=voi_score,
                pre_entropy=pre_entropy,
                bundle=bundle,
                gfe_score=gfe_score,
                gfe_bundle=gfe_bundle,
            )
            do_query = bool(query_eligible and policy_requests_query)
            for candidate in ranked:
                opportunities.append(
                    {
                        "policy_protocol": ADAPTIVE_QUERY_ONLINE_PROTOCOL,
                        "query_policy": query_policy,
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
                        "decision_opportunity_index": opportunity_index,
                        "candidate_action": candidate,
                        "predicted_action_class": pre_classes[candidate],
                        "predicted_improve_probability": float(pre_scores[candidate]),
                        "candidate_ranked_first": bool(candidate == pre_selected),
                        "pre_action_threshold": float(bundle["pre_action_threshold"]),
                        "pre_top2_probability_margin": pre_margin,
                        "pre_mean_binary_action_entropy": pre_entropy,
                        "pre_local_uncertainty_mean": local_uncertainty,
                        "predicted_query_class": voi_class,
                        "predicted_query_improve_probability": voi_score,
                        "voi_threshold": float(bundle["voi_threshold"]),
                        "predicted_g_fe_selected_path": gfe_score,
                        "g_fe_decision_threshold": (
                            None
                            if query_policy not in GFE_QUERY_POLICIES
                            else float(
                                gfe_bundle[
                                    "matched_query_rate_threshold"
                                    if query_policy
                                    == "g_fe_query_gate_matched_rate"
                                    else "decision_threshold"
                                ]
                            )
                        ),
                        "query_eligible": bool(query_eligible),
                        "policy_requests_query": bool(policy_requests_query),
                        "query_triggered_here": bool(do_query),
                    }
                )
            if do_query:
                query_started = perf_counter()
                query_spec = get_query_spec(config.query.query_id)
                sample = sample_problem(
                    problem=problem,
                    sample_design=query_spec.sample_design,
                    base_seed=seed,
                    function=function,
                    instance=instance,
                    success_gap_target=config.success_gap_target,
                    failure_loss_cap=config.failure_loss_cap,
                )
                for point, value in zip(sample["X"], sample["y"], strict=True):
                    observe_evaluation(np.asarray(point, dtype=float), float(value))
                query_runtime_seconds = float(perf_counter() - query_started)
                query_triggered = True
                query_fe = int(sample["sample_effective_FE"])
                query_best_gap = float(sample["query_best_gap"])
                query_sample_status = str(sample["sample_status"])
                if not bool(sample["sample_path_completed"]):
                    raise RuntimeError("online Query sample did not complete")
                descriptors = calculate_descriptor_cheap(
                    np.asarray(sample["X"], dtype=float),
                    np.asarray(sample["y"], dtype=float),
                    problem.lower_bounds,
                    problem.upper_bounds,
                )
                post_features = {**features, **descriptors}
                post_scores, _ = _predict_action_scores(
                    models=bundle["query_descriptor_models"],
                    feature_columns=QUERY_DESCRIPTOR_FEATURES,
                    features=post_features,
                    current_algorithm=prefix_algorithm,
                    portfolio=config.algorithms,
                )
                post_ranked = _rank_candidates(
                    post_scores,
                    config.algorithms,
                    prefix_algorithm,
                )
                post_selected = post_ranked[0]
                if float(post_scores[post_selected]) > float(
                    bundle["query_descriptor_threshold"]
                ):
                    selected_algorithm = post_selected
                else:
                    selected_algorithm = prefix_algorithm
                selected_fe = int(record.FE)
                selected_opportunity = opportunity_index
                action_selected = True
            elif query_policy in GFE_QUERY_POLICIES:
                continue
            elif query_policy == "fixed_030_query" and float(record.FE_ratio) < 0.30:
                continue
            elif pre_top > float(bundle["pre_action_threshold"]):
                selected_algorithm = pre_selected
                selected_fe = int(record.FE)
                selected_opportunity = opportunity_index
                action_selected = True
            if action_selected:
                if selected_algorithm != prefix_algorithm:
                    state = initialize_transferred_optimizer_state(
                        algorithm=selected_algorithm,
                        source_state=state,
                        problem=problem,
                        seed=seed,
                        function=function,
                        instance=instance,
                        event=NO_QUERY_TRANSFER_EVENT,
                    )
                remaining = config.fe_total - evaluation_count
                advance_optimizer_state(
                    state=state,
                    problem=problem,
                    fe_budget=remaining,
                    on_evaluation=observe_evaluation,
                )
        if not action_selected and evaluation_count < config.fe_total:
            advance_optimizer_state(
                state=state,
                problem=problem,
                fe_budget=config.fe_total - evaluation_count,
                on_evaluation=observe_evaluation,
            )
        if evaluation_count != config.fe_total:
            raise RuntimeError("adaptive Query policy did not use the strict total FE budget")
        reference = problem.reference_value
        if reference is None:
            raise ValueError("adaptive Query endpoint requires a reference value")
        continuation_gap = min(
            max(float(state.best_fitness) - float(reference), 0.0),
            config.failure_loss_cap,
        )
        final_gap = (
            min(continuation_gap, query_best_gap)
            if query_triggered
            else continuation_gap
        )
        best_fitness = float(reference) + final_gap
    except Exception as exc:
        status = "failed"
        failure_type = type(exc).__name__
        failure_message = str(exc)[:500]
        continuation_gap = config.failure_loss_cap
        final_gap = config.failure_loss_cap
        best_fitness = None
    finally:
        problem.close()
    elapsed = float(perf_counter() - started)
    selected_equals_prefix = selected_algorithm == prefix_algorithm
    outcome = {
        "policy_protocol": ADAPTIVE_QUERY_ONLINE_PROTOCOL,
        "query_policy": query_policy,
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
        "continuation_gap": float(continuation_gap),
        "query_sample_best_gap": float(query_best_gap),
        "final_gap": float(final_gap),
        "log10_gap": float(
            np.log10(
                np.clip(final_gap, config.log10_gap_floor, config.log10_gap_cap)
            )
        ),
        "success": bool(status == "completed" and final_gap <= config.success_gap_target),
        "first_hit_FE": first_hit_fe,
        "query_triggered": bool(query_triggered),
        "query_FE": int(query_fe),
        "query_sample_status": query_sample_status,
        "selected_algorithm": selected_algorithm,
        "selected_FE": selected_fe,
        "selected_decision_opportunity_index": selected_opportunity,
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
        "query_policy": query_policy,
        "split": suite.split,
        "problem_id": problem.problem_id,
        "prefix_algorithm": prefix_algorithm,
        "seed": seed,
        "policy_runtime_seconds": elapsed,
        "query_runtime_seconds": query_runtime_seconds,
        "timing_source": "single_online_policy_execution_diagnostic",
        "timing_replay_status": status,
    }
    return outcome, opportunities, resources


def _predict_action_scores(
    *,
    models: dict[str, Any],
    feature_columns: tuple[str, ...],
    features: dict,
    current_algorithm: str,
    portfolio: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, str]]:
    frame = pd.DataFrame([{column: features[column] for column in feature_columns}])
    scores = {}
    classes = {}
    for algorithm in portfolio:
        if algorithm == current_algorithm:
            continue
        model = models[algorithm]
        classes[algorithm] = str(model.predict(frame)[0])
        probabilities = np.asarray(model.predict_proba(frame), dtype=float)
        labels = np.asarray(model.classes_).astype(str)
        matches = np.flatnonzero(labels == "improve")
        scores[algorithm] = (
            0.0 if len(matches) == 0 else float(probabilities[0, int(matches[0])])
        )
    return scores, classes


def _predict_voi(bundle: dict[str, Any], features: dict) -> tuple[float, str]:
    frame = pd.DataFrame(
        [{column: features[column] for column in VOI_FEATURE_COLUMNS}]
    )
    model = bundle["voi_model"]
    label = str(model.predict(frame)[0])
    probabilities = np.asarray(model.predict_proba(frame), dtype=float)
    labels = np.asarray(model.classes_).astype(str)
    matches = np.flatnonzero(labels == "improve")
    score = 0.0 if len(matches) == 0 else float(probabilities[0, int(matches[0])])
    return score, label


def _predict_gfe(bundle: dict[str, Any] | None, features: dict) -> float:
    if bundle is None:
        raise ValueError("G_FE Gate prediction requires its model bundle")
    frame = pd.DataFrame(
        [
            {
                column: features[column]
                for column in GFE_GATE_BEHAVIOR_FEATURE_COLUMNS
            }
        ]
    )
    score = float(bundle["model"].predict(frame)[0])
    if not np.isfinite(score):
        raise RuntimeError("G_FE Gate produced a non-finite online score")
    return score


def _policy_requests_query(
    *,
    query_policy: str,
    record,
    opportunity_index: int,
    voi_score: float,
    pre_entropy: float,
    bundle: dict[str, Any],
    gfe_score: float | None,
    gfe_bundle: dict[str, Any] | None,
) -> bool:
    if query_policy == "never_query":
        return False
    if query_policy == "always_query":
        return opportunity_index == 0
    if query_policy == "fixed_030_query":
        return bool(
            record.is_budget_milestone
            and record.budget_milestone_ratio is not None
            and np.isclose(
                float(record.budget_milestone_ratio),
                0.30,
                rtol=0.0,
                atol=1e-12,
            )
        )
    if query_policy == "uncertainty_query":
        return pre_entropy > float(bundle["uncertainty_matched_threshold"])
    if query_policy in GFE_QUERY_POLICIES:
        if gfe_score is None or gfe_bundle is None:
            raise ValueError("G_FE Gate policy is missing its online score")
        threshold_name = (
            "matched_query_rate_threshold"
            if query_policy == "g_fe_query_gate_matched_rate"
            else "decision_threshold"
        )
        return gfe_score > float(gfe_bundle[threshold_name])
    return voi_score > float(bundle["voi_threshold"])


def _rank_candidates(
    scores: dict[str, float],
    portfolio: tuple[str, ...],
    current_algorithm: str,
) -> list[str]:
    candidates = [name for name in portfolio if name != current_algorithm]
    return sorted(candidates, key=lambda name: (-scores[name], portfolio.index(name)))


def _mean_binary_entropy(values: list[float]) -> float:
    probabilities = np.clip(np.asarray(values, dtype=float), 1e-12, 1.0 - 1e-12)
    return float(
        np.mean(
            -probabilities * np.log(probabilities)
            - (1.0 - probabilities) * np.log(1.0 - probabilities)
        )
        / np.log(2.0)
    )


def _evaluate_function(
    config: ExperimentConfig,
    bundle: dict[str, Any],
    gfe_bundle: dict[str, Any] | None,
    suite: SuiteConfig,
    function: int,
    prefixes: tuple[str, ...],
    policies: tuple[str, ...],
) -> tuple[list[dict], list[dict], list[dict]]:
    outcomes = []
    opportunities = []
    resources = []
    for instance in suite.instances:
        for seed in config.seeds:
            for prefix in prefixes:
                for policy in policies:
                    outcome, rows, resource = run_adaptive_query_policy(
                        config=config,
                        suite=suite,
                        function=function,
                        instance=instance,
                        seed=seed,
                        prefix_algorithm=prefix,
                        bundle=bundle,
                        gfe_bundle=gfe_bundle,
                        query_policy=policy,
                    )
                    outcomes.append(outcome)
                    opportunities.extend(rows)
                    resources.append(resource)
    return outcomes, opportunities, resources


def _validate_bundle(bundle: dict[str, Any], config: ExperimentConfig) -> None:
    if tuple(bundle.get("portfolio", ())) != config.algorithms:
        raise ValueError("Phase 3 portfolio differs from online config")
    for name, value in (
        ("dimension", config.dimension),
        ("FE_total", config.fe_total),
        ("population_size", config.population_size),
    ):
        if int(bundle.get(name, -1)) != int(value):
            raise ValueError(f"Phase 3 {name} differs from online config")
    if str(bundle.get("sampling_protocol")) != config.sampling_protocol:
        raise ValueError("Phase 3 sampling protocol differs from online config")
    if str(bundle.get("boundary_handling")) != config.boundary_handling:
        raise ValueError("Phase 3 boundary handling differs from online config")
    if bundle.get("local_landscape_config") != config.local_landscape:
        raise ValueError("Phase 3 local-landscape config differs from online config")
    if bundle.get("query_config") != config.query:
        raise ValueError("Phase 3 Query config differs from online config")
    if not np.isfinite(float(bundle.get("practical_gain_delta", np.nan))):
        raise ValueError("Phase 3 practical gain threshold must be finite")


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
        description="Evaluate adaptive Query policies with real Query FE online."
    )
    parser.add_argument("--config", default="configs/behavior_with_ela_cec2017.yaml")
    parser.add_argument(
        "--phase3-model",
        default="results/behavior_with_ela/model/adaptive_query/phase3_models.joblib",
    )
    parser.add_argument(
        "--gfe-model",
        default="results/behavior_with_ela/model/gfe_query_gate/gfe_gate_model.joblib",
    )
    parser.add_argument(
        "--output",
        default="results/behavior_with_ela/online/adaptive_query",
    )
    parser.add_argument("--query-policy", action="append", default=None)
    parser.add_argument("--only-split", action="append", default=None)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--initial-algorithm", default="sbs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = evaluate_adaptive_query_online(
        config_path=args.config,
        phase3_model_path=args.phase3_model,
        gfe_model_path=args.gfe_model,
        output_dir=args.output,
        query_policies=(
            QUERY_POLICIES
            if args.query_policy is None
            else tuple(args.query_policy)
        ),
        only_splits=None if args.only_split is None else tuple(args.only_split),
        only_functions=(
            None if args.only_function is None else tuple(args.only_function)
        ),
        initial_algorithm=args.initial_algorithm,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(
        f"completed {summary['policy_runs']} adaptive-Query online runs with "
        f"{summary['opportunity_action_rows']} opportunity-action predictions"
    )


if __name__ == "__main__":
    main()
