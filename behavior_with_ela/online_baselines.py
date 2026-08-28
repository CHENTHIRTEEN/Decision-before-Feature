from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS, extract_behavior_rows
from behavior_with_ela.baselines import ALGORITHM_CODES, FIXED_SWITCH_RATIO
from behavior_with_ela.protocol import (
    ExperimentConfig,
    SuiteConfig,
    make_experiment_problem,
    suite_code,
)
from optimizers import (
    NO_QUERY_TRANSFER_EVENT,
    OptimizerSettings,
    advance_optimizer_state,
    initialize_optimizer_state,
    initialize_transferred_optimizer_state,
)
from trajectory.recorder import TrajectoryRecorder
from trajectory.sampling import BUDGET_MILESTONE_RATIOS


STATIC_POLICY_PROTOCOL = "static_full_budget_optimizer_v1"
BASELINE_ONE_SWITCH_PROTOCOL = "phase1_baseline_one_switch_online_v1"
RANDOM_ONLINE_STREAM = 2026082805
RANDOM_MATCHED_ONLINE_STREAM = 2026082806

STATIC_POLICY_NAMES = ("continue_current", "sbs")
LEARNED_BASELINE_POLICY_NAMES = (
    "time_only_action_gain",
    "behavior_action_loss_rf",
    "to_switch_style_rf",
    "fixed_030_transition",
    "random_one_switch",
    "random_matched_switch_rate",
)


def run_static_optimizer_policy(
    *,
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    instance: int,
    seed: int,
    prefix_algorithm: str,
    default_algorithm: str,
    policy_name: str,
) -> tuple[dict, list[dict], dict]:
    if policy_name not in STATIC_POLICY_NAMES:
        raise ValueError(f"static policy must belong to {STATIC_POLICY_NAMES}")
    if policy_name == "sbs" and prefix_algorithm != default_algorithm:
        raise ValueError("SBS policy must start from the train-derived default algorithm")
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
    state = None

    def observe_evaluation(point: np.ndarray, value: float) -> None:
        nonlocal evaluation_count, first_hit_fe
        evaluation_count += 1
        reference = problem.reference_value
        if reference is None:
            raise ValueError("static optimizer policy requires a benchmark reference value")
        gap = max(float(value) - float(reference), 0.0)
        if first_hit_fe is None and gap <= config.success_gap_target:
            first_hit_fe = evaluation_count

    status = "completed"
    failure_type = ""
    failure_message = ""
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
        advance_optimizer_state(
            state=state,
            problem=problem,
            fe_budget=config.fe_total - evaluation_count,
            on_evaluation=observe_evaluation,
        )
        if evaluation_count != config.fe_total:
            raise RuntimeError("static optimizer policy did not use the strict total FE budget")
        reference = problem.reference_value
        if reference is None:
            raise ValueError("static optimizer endpoint requires a reference value")
        best_fitness = float(state.best_fitness)
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
    outcome = _outcome_row(
        config=config,
        suite=suite,
        problem=problem,
        seed=seed,
        prefix_algorithm=prefix_algorithm,
        default_algorithm=default_algorithm,
        selected_algorithm=prefix_algorithm,
        selected_fe=None,
        selected_opportunity=None,
        policy_name=policy_name,
        policy_protocol=STATIC_POLICY_PROTOCOL,
        status=status,
        failure_type=failure_type,
        failure_message=failure_message,
        evaluation_count=evaluation_count,
        best_fitness=best_fitness,
        final_gap=final_gap,
        first_hit_fe=first_hit_fe,
    )
    return outcome, [], _resource_row(outcome, elapsed)


def run_phase1_baseline_policy(
    *,
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    instance: int,
    seed: int,
    prefix_algorithm: str,
    bundle: dict[str, Any],
    policy_name: str,
) -> tuple[dict, list[dict], dict]:
    if policy_name not in LEARNED_BASELINE_POLICY_NAMES:
        raise ValueError(
            f"Phase 1 baseline policy must belong to {LEARNED_BASELINE_POLICY_NAMES}"
        )
    _validate_baseline_bundle(bundle, config)
    default_algorithm = str(bundle["default_algorithm"])
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
    state = None
    selected_algorithm = prefix_algorithm
    selected_fe: int | None = None
    selected_opportunity: int | None = None
    selected_score: float | None = None
    decision_complete = False
    random_plan = _random_policy_plan(
        policy_name=policy_name,
        config=config,
        suite=suite,
        function=function,
        instance=instance,
        seed=seed,
        prefix_algorithm=prefix_algorithm,
        bundle=bundle,
    )

    def observe_evaluation(point: np.ndarray, value: float) -> None:
        nonlocal evaluation_count, first_hit_fe
        evaluation_count += 1
        reference = problem.reference_value
        if reference is None:
            raise ValueError("online baseline requires a benchmark reference value")
        gap = max(float(value) - float(reference), 0.0)
        if first_hit_fe is None and gap <= config.success_gap_target:
            first_hit_fe = evaluation_count

    status = "completed"
    failure_type = ""
    failure_message = ""
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
        while evaluation_count < config.fe_total and not decision_complete:
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
                raise RuntimeError("one update emitted multiple online baseline states")
            record = recorder.records[-1]
            opportunity_index = emitted_count
            emitted_count += 1
            behavior = extract_behavior_rows([asdict(record)])[0]
            choice = _baseline_choice(
                policy_name=policy_name,
                bundle=bundle,
                behavior=behavior,
                record=record,
                opportunity_index=opportunity_index,
                prefix_algorithm=prefix_algorithm,
                portfolio=config.algorithms,
                random_plan=random_plan,
            )
            opportunities.extend(
                _opportunity_rows(
                    config=config,
                    suite=suite,
                    problem=problem,
                    seed=seed,
                    prefix_algorithm=prefix_algorithm,
                    default_algorithm=default_algorithm,
                    policy_name=policy_name,
                    record=record,
                    opportunity_index=opportunity_index,
                    choice=choice,
                )
            )
            if not bool(choice["decision_complete"]):
                continue
            decision_complete = True
            selected_algorithm = str(choice["selected_algorithm"])
            selected_score = choice["selected_score"]
            if selected_algorithm != prefix_algorithm:
                selected_fe = int(record.FE)
                selected_opportunity = opportunity_index
                state = initialize_transferred_optimizer_state(
                    algorithm=selected_algorithm,
                    source_state=state,
                    problem=problem,
                    seed=seed,
                    function=function,
                    instance=instance,
                    event=NO_QUERY_TRANSFER_EVENT,
                )
            advance_optimizer_state(
                state=state,
                problem=problem,
                fe_budget=config.fe_total - evaluation_count,
                on_evaluation=observe_evaluation,
            )
        if not decision_complete and evaluation_count < config.fe_total:
            advance_optimizer_state(
                state=state,
                problem=problem,
                fe_budget=config.fe_total - evaluation_count,
                on_evaluation=observe_evaluation,
            )
        if evaluation_count != config.fe_total:
            raise RuntimeError("online baseline did not use the strict total FE budget")
        reference = problem.reference_value
        if reference is None:
            raise ValueError("online baseline endpoint requires a reference value")
        best_fitness = float(state.best_fitness)
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
    outcome = _outcome_row(
        config=config,
        suite=suite,
        problem=problem,
        seed=seed,
        prefix_algorithm=prefix_algorithm,
        default_algorithm=default_algorithm,
        selected_algorithm=selected_algorithm,
        selected_fe=selected_fe,
        selected_opportunity=selected_opportunity,
        policy_name=policy_name,
        policy_protocol=BASELINE_ONE_SWITCH_PROTOCOL,
        status=status,
        failure_type=failure_type,
        failure_message=failure_message,
        evaluation_count=evaluation_count,
        best_fitness=best_fitness,
        final_gap=final_gap,
        first_hit_fe=first_hit_fe,
    )
    outcome["selected_score"] = selected_score
    outcome["decision_complete"] = bool(decision_complete)
    return outcome, opportunities, _resource_row(outcome, elapsed)


def _baseline_choice(
    *,
    policy_name: str,
    bundle: dict[str, Any],
    behavior: dict,
    record,
    opportunity_index: int,
    prefix_algorithm: str,
    portfolio: tuple[str, ...],
    random_plan: dict[str, Any],
) -> dict[str, Any]:
    candidates = tuple(name for name in portfolio if name != prefix_algorithm)
    scores: dict[str, float | None] = {name: None for name in candidates}
    classes: dict[str, str | None] = {name: None for name in candidates}
    threshold: float | None = None
    selected = prefix_algorithm
    selected_score: float | None = None
    decision_complete = False

    if policy_name == "time_only_action_gain":
        frame = pd.DataFrame([{"bf_fe_ratio": behavior["bf_fe_ratio"]}])
        for candidate in candidates:
            model = bundle["time_only_models"][candidate]
            classes[candidate], scores[candidate] = _class_and_probability(
                model,
                frame,
                "improve",
            )
        selected = _highest_score(scores, portfolio)
        selected_score = float(scores[selected])
        threshold = float(bundle["time_only_threshold"])
        decision_complete = selected_score > threshold
    elif policy_name == "behavior_action_loss_rf":
        losses = _predicted_action_losses(bundle, behavior, portfolio)
        for candidate in candidates:
            score = float(losses[prefix_algorithm] - losses[candidate])
            scores[candidate] = score
            classes[candidate] = _gain_class(
                score,
                float(bundle["practical_gain_delta"]),
            )
        selected = _highest_score(scores, portfolio)
        selected_score = float(scores[selected])
        threshold = float(bundle["behavior_action_loss_threshold"])
        decision_complete = selected_score > threshold
    elif policy_name == "to_switch_style_rf":
        losses = _predicted_action_losses(bundle, behavior, portfolio)
        selected = min(candidates, key=lambda name: (losses[name], portfolio.index(name)))
        frame = pd.DataFrame(
            [{column: behavior[column] for column in SELECTOR_BEHAVIOR_FEATURE_COLUMNS}]
        )
        predicted_class, probability = _class_and_probability(
            bundle["to_switch_model"],
            frame,
            True,
        )
        for candidate in candidates:
            scores[candidate] = probability if candidate == selected else -1.0
            classes[candidate] = (
                "improve"
                if candidate == selected and bool(predicted_class)
                else "equivalent"
            )
        selected_score = float(probability)
        threshold = float(bundle["to_switch_threshold"])
        decision_complete = selected_score > threshold
    elif policy_name == "fixed_030_transition":
        at_target = bool(
            record.is_budget_milestone
            and record.budget_milestone_ratio is not None
            and np.isclose(
                float(record.budget_milestone_ratio),
                FIXED_SWITCH_RATIO,
                rtol=0.0,
                atol=1e-12,
            )
        )
        if at_target:
            selected = str(random_plan["selected_algorithm"])
            decision_complete = True
    elif policy_name == "random_one_switch":
        at_target = bool(
            record.is_budget_milestone
            and record.budget_milestone_ratio is not None
            and np.isclose(
                float(record.budget_milestone_ratio),
                float(random_plan["target_ratio"]),
                rtol=0.0,
                atol=1e-12,
            )
        )
        if at_target:
            selected = str(random_plan["selected_algorithm"])
            decision_complete = True
    elif policy_name == "random_matched_switch_rate":
        if not bool(random_plan["should_switch"]):
            return {
                "decision_complete": False,
                "selected_algorithm": prefix_algorithm,
                "selected_score": None,
                "threshold": None,
                "scores": scores,
                "classes": classes,
                "candidate_ranked_first": None,
            }
        if float(record.FE_ratio) + 1e-12 >= float(random_plan["target_ratio"]):
            selected = str(random_plan["selected_algorithm"])
            decision_complete = True

    ranked_first = selected if selected in candidates else None
    return {
        "decision_complete": bool(decision_complete),
        "selected_algorithm": selected,
        "selected_score": selected_score,
        "threshold": threshold,
        "scores": scores,
        "classes": classes,
        "candidate_ranked_first": ranked_first,
    }


def _random_policy_plan(
    *,
    policy_name: str,
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    instance: int,
    seed: int,
    prefix_algorithm: str,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    if policy_name == "fixed_030_transition":
        mapping = bundle["fixed_transition_mapping"]
        rows = mapping.loc[
            mapping["prefix_algorithm"].astype(str).eq(prefix_algorithm)
        ]
        if len(rows) != 1:
            raise ValueError("fixed transition mapping must contain one row per prefix")
        return {
            "target_ratio": FIXED_SWITCH_RATIO,
            "selected_algorithm": str(rows.iloc[0]["selected_algorithm"]),
            "should_switch": True,
        }
    if policy_name not in {"random_one_switch", "random_matched_switch_rate"}:
        return {}
    stream = (
        RANDOM_ONLINE_STREAM
        if policy_name == "random_one_switch"
        else RANDOM_MATCHED_ONLINE_STREAM
    )
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                stream,
                suite_code(suite.suite),
                int(function),
                int(instance),
                config.dimension,
                ALGORITHM_CODES[prefix_algorithm],
            ]
        )
    )
    candidates = tuple(name for name in config.algorithms if name != prefix_algorithm)
    selected_algorithm = candidates[int(rng.integers(0, len(candidates)))]
    if policy_name == "random_one_switch":
        target_ratio = float(
            BUDGET_MILESTONE_RATIOS[
                int(rng.integers(0, len(BUDGET_MILESTONE_RATIOS)))
            ]
        )
        return {
            "target_ratio": target_ratio,
            "selected_algorithm": selected_algorithm,
            "should_switch": True,
        }
    calibration = bundle["random_matched_calibration"]
    call_rate = float(calibration.iloc[0]["matched_switch_rate"])
    trigger_ratios = calibration["trigger_ratio"].dropna().to_numpy(dtype=float)
    should_switch = bool(rng.random() < call_rate)
    if should_switch and len(trigger_ratios) == 0:
        raise ValueError("matched random policy has no trigger ratios")
    target_ratio = (
        None
        if not should_switch
        else float(trigger_ratios[int(rng.integers(0, len(trigger_ratios)))])
    )
    return {
        "target_ratio": target_ratio,
        "selected_algorithm": selected_algorithm,
        "should_switch": should_switch,
    }


def _predicted_action_losses(
    bundle: dict[str, Any],
    behavior: dict,
    portfolio: tuple[str, ...],
) -> dict[str, float]:
    frame = pd.DataFrame(
        [{column: behavior[column] for column in SELECTOR_BEHAVIOR_FEATURE_COLUMNS}]
    )
    values = np.asarray(
        bundle["behavior_action_loss_model"].predict(frame),
        dtype=float,
    )
    if values.shape != (1, len(portfolio)):
        raise RuntimeError("online action-loss RF returned an unexpected shape")
    return {name: float(values[0, index]) for index, name in enumerate(portfolio)}


def _class_and_probability(model, frame: pd.DataFrame, label) -> tuple[Any, float]:
    predicted = model.predict(frame)[0]
    probabilities = np.asarray(model.predict_proba(frame), dtype=float)
    classes = np.asarray(model.classes_)
    matches = np.flatnonzero(classes == label)
    if len(matches) > 1:
        raise RuntimeError("classifier exposes duplicate target classes")
    probability = 0.0 if len(matches) == 0 else float(probabilities[0, int(matches[0])])
    return predicted, probability


def _highest_score(
    scores: dict[str, float | None],
    portfolio: tuple[str, ...],
) -> str:
    return max(
        scores,
        key=lambda name: (
            float(scores[name]),
            -portfolio.index(name),
        ),
    )


def _gain_class(value: float, delta: float) -> str:
    if value > delta:
        return "improve"
    if value < -delta:
        return "degrade"
    return "equivalent"


def _opportunity_rows(
    *,
    config: ExperimentConfig,
    suite: SuiteConfig,
    problem,
    seed: int,
    prefix_algorithm: str,
    default_algorithm: str,
    policy_name: str,
    record,
    opportunity_index: int,
    choice: dict[str, Any],
) -> list[dict]:
    rows = []
    for candidate in config.algorithms:
        if candidate == prefix_algorithm:
            continue
        rows.append(
            {
                "policy_protocol": BASELINE_ONE_SWITCH_PROTOCOL,
                "policy_name": policy_name,
                "split": suite.split,
                "suite": suite.suite,
                "problem_id": problem.problem_id,
                "function_id": problem.function_id,
                "family": problem.family,
                "cv_group_id": problem.cv_group_id,
                "dimension": problem.dimension,
                "prefix_algorithm": prefix_algorithm,
                "default_algorithm": default_algorithm,
                "seed": seed,
                "FE": int(record.FE),
                "FE_ratio": float(record.FE_ratio),
                "decision_opportunity_index": opportunity_index,
                "candidate_action": candidate,
                "predicted_action_class": choice["classes"][candidate],
                "action_score": choice["scores"][candidate],
                "decision_threshold": choice["threshold"],
                "candidate_ranked_first": bool(
                    choice["candidate_ranked_first"] == candidate
                ),
                "would_trigger": bool(
                    choice["decision_complete"]
                    and choice["selected_algorithm"] == candidate
                ),
            }
        )
    return rows


def _outcome_row(
    *,
    config: ExperimentConfig,
    suite: SuiteConfig,
    problem,
    seed: int,
    prefix_algorithm: str,
    default_algorithm: str,
    selected_algorithm: str,
    selected_fe: int | None,
    selected_opportunity: int | None,
    policy_name: str,
    policy_protocol: str,
    status: str,
    failure_type: str,
    failure_message: str,
    evaluation_count: int,
    best_fitness: float | None,
    final_gap: float,
    first_hit_fe: int | None,
) -> dict:
    selected_equals_prefix = selected_algorithm == prefix_algorithm
    return {
        "policy_protocol": policy_protocol,
        "policy_name": policy_name,
        "split": suite.split,
        "suite": suite.suite,
        "problem_id": problem.problem_id,
        "function_id": problem.function_id,
        "family": problem.family,
        "cv_group_id": problem.cv_group_id,
        "dimension": problem.dimension,
        "prefix_algorithm": prefix_algorithm,
        "default_algorithm": default_algorithm,
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
                np.clip(final_gap, config.log10_gap_floor, config.log10_gap_cap)
            )
        ),
        "success": bool(status == "completed" and final_gap <= config.success_gap_target),
        "first_hit_FE": first_hit_fe,
        "selected_algorithm": selected_algorithm,
        "selected_FE": selected_fe,
        "selected_decision_opportunity_index": selected_opportunity,
        "selected_equals_default": bool(selected_algorithm == default_algorithm),
        "selected_equals_prefix": bool(selected_equals_prefix),
        "handoff_required": bool(not selected_equals_prefix),
        "handoff_type": (
            "native_optimizer_state"
            if selected_equals_prefix
            else "population_transfer_initialization"
        ),
        "switch_count": int(not selected_equals_prefix),
        "query_FE": 0,
        "boundary_handling": config.boundary_handling,
    }


def _resource_row(outcome: dict, elapsed: float) -> dict:
    return {
        "policy_name": outcome["policy_name"],
        "split": outcome["split"],
        "problem_id": outcome["problem_id"],
        "prefix_algorithm": outcome["prefix_algorithm"],
        "seed": outcome["seed"],
        "policy_runtime_seconds": float(elapsed),
        "timing_source": "single_online_policy_execution_diagnostic",
        "timing_replay_status": outcome["policy_status"],
    }


def _validate_baseline_bundle(bundle: dict[str, Any], config: ExperimentConfig) -> None:
    if tuple(bundle.get("portfolio", ())) != config.algorithms:
        raise ValueError("baseline model portfolio differs from online config")
    for name, value in (
        ("dimension", config.dimension),
        ("FE_total", config.fe_total),
        ("population_size", config.population_size),
    ):
        if int(bundle.get(name, -1)) != int(value):
            raise ValueError(f"baseline model {name} differs from online config")
    if str(bundle.get("sampling_protocol")) != config.sampling_protocol:
        raise ValueError("baseline sampling protocol differs from online config")
    if str(bundle.get("boundary_handling")) != config.boundary_handling:
        raise ValueError("baseline boundary handling differs from online config")
    if str(bundle.get("default_algorithm")) not in config.algorithms:
        raise ValueError("baseline bundle is missing the train-derived default algorithm")
