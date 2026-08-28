from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd

from behavior_with_ela.adaptive_query_online import (
    _validate_bundle as validate_phase3_bundle,
    run_adaptive_query_policy,
)
from behavior_with_ela.gfe_query_gate import validate_gfe_bundle
from behavior_with_ela.online import (
    _validate_model_bundle as validate_phase1_bundle,
    run_one_switch_policy,
)
from behavior_with_ela.online_baselines import (
    LEARNED_BASELINE_POLICY_NAMES,
    run_phase1_baseline_policy,
    run_static_optimizer_policy,
)
from behavior_with_ela.protocol import (
    ExperimentConfig,
    SuiteConfig,
    check_problem_availability,
    load_experiment_config,
)
from behavior_with_ela.repeated_das import (
    _validate_bundle as validate_phase2_bundle,
    run_phase2_one_switch_policy,
    run_repeated_das_policy,
)
from behavior_with_ela.traditional_aas import (
    _validate_online_bundle as validate_traditional_aas_bundle,
    run_traditional_aas_policy,
)


ONLINE_COMPARISON_PROTOCOL = "behavior_with_ela_complete_policy_comparison_v1"
TIMING_SOURCE = "measured_complete_policy_path"
TIMING_REPETITIONS = 3

QUERY_POLICY_MAP = {
    "query_voi": "voi_query",
    "query_never": "never_query",
    "query_always": "always_query",
    "query_fixed_030": "fixed_030_query",
    "query_uncertainty": "uncertainty_query",
    "g_fe_query_gate": "g_fe_query_gate",
    "g_fe_query_gate_matched_rate": "g_fe_query_gate_matched_rate",
}
ONLINE_COMPARISON_POLICIES = (
    "continue_current",
    "sbs",
    "traditional_aas",
    "random_one_switch",
    "random_matched_switch_rate",
    "fixed_030_transition",
    "time_only_action_gain",
    "behavior_action_loss_rf",
    "to_switch_style_rf",
    "phase1_action_gain",
    "behavior_action_loss_regression_v2",
    "phase2_m4_one_switch",
    "query_never",
    "query_always",
    "query_fixed_030",
    "query_uncertainty",
    "query_voi",
    "g_fe_query_gate",
    "g_fe_query_gate_matched_rate",
    "repeated_das",
)
V2_REGRESSION_POLICY_NAME = "behavior_action_loss_regression_v2"
DEFAULT_V2_REGRESSION_MODEL = (
    "behavior_with_ela/results/analysis_v2/task9_quick_cec/v2_online_bundle.joblib"
)


def evaluate_online_comparison(
    *,
    config_path: str | Path,
    phase1_model_path: str | Path,
    baseline_model_path: str | Path,
    phase2_model_path: str | Path,
    phase3_model_path: str | Path,
    gfe_model_path: str | Path,
    traditional_aas_model_path: str | Path,
    v2_regression_model_path: str | Path = DEFAULT_V2_REGRESSION_MODEL,
    output_dir: str | Path,
    policies: tuple[str, ...] = ONLINE_COMPARISON_POLICIES,
    only_splits: tuple[str, ...] | None = None,
    only_functions: tuple[int, ...] | None = None,
    initial_algorithm: str = "sbs",
    workers: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    selected_policies = _validate_policies(policies)
    phase1_bundle = joblib.load(phase1_model_path)
    validate_phase1_bundle(phase1_bundle, config)
    default_algorithm = str(phase1_bundle["default_algorithm"])
    prefixes = _initial_algorithms(initial_algorithm, default_algorithm, config)

    v2_regression_bundle = _load_required_bundle(
        v2_regression_model_path,
        required=V2_REGRESSION_POLICY_NAME in selected_policies,
    )
    if v2_regression_bundle is not None:
        validate_phase1_bundle(v2_regression_bundle, config)
        if str(v2_regression_bundle.get("model_protocol")) != (
            "behavior_action_loss_regression_v2"
        ):
            raise ValueError(
                "v2 regression bundle carries a different model protocol: "
                f"{v2_regression_bundle.get('model_protocol')}"
            )
        if str(v2_regression_bundle["default_algorithm"]) != default_algorithm:
            raise ValueError(
                "v2 regression and Phase 1 bundles use different default algorithms"
            )

    baseline_bundle = _load_required_bundle(
        baseline_model_path,
        required=bool(set(selected_policies).intersection(LEARNED_BASELINE_POLICY_NAMES)),
    )
    phase2_bundle = _load_required_bundle(
        phase2_model_path,
        required=bool(
            set(selected_policies).intersection(
                {"phase2_m4_one_switch", "repeated_das"}
            )
        ),
    )
    phase3_bundle = _load_required_bundle(
        phase3_model_path,
        required=bool(set(selected_policies).intersection(QUERY_POLICY_MAP)),
    )
    gfe_bundle = _load_required_bundle(
        gfe_model_path,
        required=bool(
            set(selected_policies).intersection(
                {"g_fe_query_gate", "g_fe_query_gate_matched_rate"}
            )
        ),
    )
    traditional_aas_bundle = _load_required_bundle(
        traditional_aas_model_path,
        required="traditional_aas" in selected_policies,
    )
    if baseline_bundle is not None:
        _validate_baseline_default(baseline_bundle, phase1_bundle)
    if phase2_bundle is not None:
        validate_phase2_bundle(phase2_bundle, config)
    if phase3_bundle is not None:
        validate_phase3_bundle(phase3_bundle, config)
    if gfe_bundle is not None:
        validate_gfe_bundle(gfe_bundle, config)
    if traditional_aas_bundle is not None:
        validate_traditional_aas_bundle(traditional_aas_bundle, config)
        if str(traditional_aas_bundle["default_algorithm"]) != default_algorithm:
            raise ValueError(
                "Traditional AAS and Phase 1 bundles use different default algorithms"
            )

    suites = _selected_suites(config, only_splits)
    tasks = [
        (suite, function)
        for suite in suites
        for function in suite.functions
        if only_functions is None or function in set(only_functions)
    ]
    if not tasks:
        raise ValueError("no online comparison functions were selected")
    check_problem_availability(config, tasks)
    if workers < 1:
        raise ValueError("workers must be at least one")

    output = Path(output_dir)
    paths = {
        "outcomes": output / "online_policy_outcomes.parquet",
        "timings": output / "complete_path_timings.parquet",
        "opportunities": output / "online_opportunities.parquet",
        "switches": output / "online_switches.parquet",
        "static_reference": output / "static_portfolio_reference.parquet",
        "summary_table": output / "policy_summary.parquet",
        "paired_contrasts": output / "paired_policy_contrasts.parquet",
        "paired_contrast_summary": output / "paired_policy_contrast_summary.parquet",
        "summary": output / "online_comparison_summary.json",
    }
    if any(path.exists() for path in paths.values()) and not overwrite:
        raise FileExistsError(f"online comparison outputs already exist: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in paths.values():
            path.unlink(missing_ok=True)

    results = []
    arguments = (
        config,
        phase1_bundle,
        baseline_bundle,
        phase2_bundle,
        phase3_bundle,
        gfe_bundle,
        traditional_aas_bundle,
        v2_regression_bundle,
        prefixes,
        selected_policies,
        default_algorithm,
    )
    if workers == 1:
        for suite, function in tasks:
            results.append(_evaluate_function(*arguments, suite, function))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_function,
                    *arguments,
                    suite,
                    function,
                )
                for suite, function in tasks
            ]
            for future in as_completed(futures):
                results.append(future.result())

    outcomes = pd.DataFrame([row for result in results for row in result[0]])
    timings = pd.DataFrame([row for result in results for row in result[1]])
    opportunities = pd.DataFrame([row for result in results for row in result[2]])
    switches = pd.DataFrame([row for result in results for row in result[3]])
    static_reference = pd.DataFrame(
        [row for result in results for row in result[4]]
    )
    if outcomes.empty or timings.empty:
        raise RuntimeError("online comparison produced no policy outcomes or timings")
    if static_reference.empty:
        raise RuntimeError("online comparison produced no static portfolio reference")
    _validate_complete_timings(outcomes, timings)
    outcomes = _attach_measured_timing_medians(outcomes, timings)
    outcomes = _attach_policy_references(outcomes, static_reference)
    summary_table = _summarize_policies(outcomes)
    paired_contrasts = paired_policy_contrasts(outcomes)
    paired_contrast_summary = summarize_paired_policy_contrasts(
        paired_contrasts
    )

    outcome_order = [
        "split",
        "problem_id",
        "prefix_algorithm",
        "seed",
        "policy_name",
    ]
    timing_order = [*outcome_order, "timing_repetition"]
    outcomes.sort_values(outcome_order, kind="mergesort").to_parquet(
        paths["outcomes"], index=False
    )
    timings.sort_values(timing_order, kind="mergesort").to_parquet(
        paths["timings"], index=False
    )
    _write_optional_rows(
        opportunities,
        paths["opportunities"],
        columns=[
            "policy_name",
            "split",
            "problem_id",
            "prefix_algorithm",
            "seed",
            "FE",
            "decision_opportunity_index",
            "candidate_action",
        ],
    )
    static_reference.sort_values(
        ["split", "problem_id", "seed", "candidate_algorithm"],
        kind="mergesort",
    ).to_parquet(paths["static_reference"], index=False)
    _write_optional_rows(
        switches,
        paths["switches"],
        columns=[
            "policy_name",
            "split",
            "problem_id",
            "prefix_algorithm",
            "seed",
            "FE",
        ],
    )
    summary_table.to_parquet(paths["summary_table"], index=False)
    paired_contrasts.to_parquet(paths["paired_contrasts"], index=False)
    paired_contrast_summary.to_parquet(
        paths["paired_contrast_summary"],
        index=False,
    )
    summary = {
        "comparison_protocol": ONLINE_COMPARISON_PROTOCOL,
        "timing_source": TIMING_SOURCE,
        "timing_repetitions": TIMING_REPETITIONS,
        "timing_order": "cyclic_policy_order_within_problem_seed_prefix",
        "policies": list(selected_policies),
        "policy_outcomes": int(len(outcomes)),
        "timing_rows": int(len(timings)),
        "opportunity_rows": int(len(opportunities)),
        "switch_rows": int(len(switches)),
        "static_reference_candidate_rows": int(len(static_reference)),
        "paired_contrast_rows": int(len(paired_contrasts)),
        "paired_contrast_summary_rows": int(len(paired_contrast_summary)),
        "static_reference_deployable_policy": False,
        "strict_total_FE": config.fe_total,
        "boundary_handling": config.boundary_handling,
        "runtime_used_in_scientific_labels": False,
    }
    with paths["summary"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def _evaluate_function(
    config: ExperimentConfig,
    phase1_bundle: dict[str, Any],
    baseline_bundle: dict[str, Any] | None,
    phase2_bundle: dict[str, Any] | None,
    phase3_bundle: dict[str, Any] | None,
    gfe_bundle: dict[str, Any] | None,
    traditional_aas_bundle: dict[str, Any] | None,
    v2_regression_bundle: dict[str, Any] | None,
    prefixes: tuple[str, ...],
    policies: tuple[str, ...],
    default_algorithm: str,
    suite: SuiteConfig,
    function: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    outcomes: list[dict] = []
    timings: list[dict] = []
    opportunities: list[dict] = []
    switches: list[dict] = []
    static_reference: list[dict] = []
    for instance in suite.instances:
        for seed in config.seeds:
            for prefix in prefixes:
                unit_policies = tuple(
                    policy
                    for policy in policies
                    if policy not in {"sbs", "traditional_aas"}
                    or prefix == default_algorithm
                )
                first_outcome: dict[str, dict] = {}
                for repetition in range(TIMING_REPETITIONS):
                    ordered = _cyclic_order(unit_policies, repetition)
                    for execution_order, policy_name in enumerate(ordered):
                        started = perf_counter()
                        outcome, state_rows, switch_rows = _dispatch_policy(
                            config=config,
                            suite=suite,
                            function=function,
                            instance=instance,
                            seed=seed,
                            prefix_algorithm=prefix,
                            policy_name=policy_name,
                            default_algorithm=default_algorithm,
                            phase1_bundle=phase1_bundle,
                            baseline_bundle=baseline_bundle,
                            phase2_bundle=phase2_bundle,
                            phase3_bundle=phase3_bundle,
                            gfe_bundle=gfe_bundle,
                            traditional_aas_bundle=traditional_aas_bundle,
                            v2_regression_bundle=v2_regression_bundle,
                        )
                        elapsed = float(perf_counter() - started)
                        normalized = _normalize_outcome(
                            outcome,
                            policy_name=policy_name,
                            default_algorithm=default_algorithm,
                        )
                        timings.append(
                            _timing_row(
                                normalized,
                                repetition=repetition,
                                execution_order=execution_order,
                                policy_count=len(unit_policies),
                                elapsed=elapsed,
                            )
                        )
                        if repetition == 0:
                            first_outcome[policy_name] = normalized
                            opportunities.extend(
                                {
                                    **row,
                                    "policy_name": policy_name,
                                    "measurement_repetition": 0,
                                }
                                for row in state_rows
                            )
                            switches.extend(
                                {
                                    **row,
                                    "policy_name": policy_name,
                                    "measurement_repetition": 0,
                                }
                                for row in switch_rows
                            )
                outcomes.extend(first_outcome[policy] for policy in unit_policies)
        static_reference.extend(
            _measure_static_portfolio_reference(
                config=config,
                suite=suite,
                function=function,
                instance=instance,
                default_algorithm=default_algorithm,
            )
        )
    return outcomes, timings, opportunities, switches, static_reference


def _dispatch_policy(
    *,
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    instance: int,
    seed: int,
    prefix_algorithm: str,
    policy_name: str,
    default_algorithm: str,
    phase1_bundle: dict[str, Any],
    baseline_bundle: dict[str, Any] | None,
    phase2_bundle: dict[str, Any] | None,
    phase3_bundle: dict[str, Any] | None,
    gfe_bundle: dict[str, Any] | None,
    traditional_aas_bundle: dict[str, Any] | None,
    v2_regression_bundle: dict[str, Any] | None = None,
) -> tuple[dict, list[dict], list[dict]]:
    common = {
        "config": config,
        "suite": suite,
        "function": function,
        "instance": instance,
        "seed": seed,
        "prefix_algorithm": prefix_algorithm,
    }
    if policy_name in {"continue_current", "sbs"}:
        outcome, rows, _ = run_static_optimizer_policy(
            **common,
            default_algorithm=default_algorithm,
            policy_name=policy_name,
        )
        return outcome, rows, []
    if policy_name == "traditional_aas":
        if traditional_aas_bundle is None:
            raise RuntimeError("Traditional-AAS policy has no model bundle")
        outcome, rows, _ = run_traditional_aas_policy(
            config=config,
            suite=suite,
            function=function,
            instance=instance,
            seed=seed,
            bundle=traditional_aas_bundle,
        )
        outcome["comparison_prefix_algorithm"] = prefix_algorithm
        return outcome, rows, []
    if policy_name in LEARNED_BASELINE_POLICY_NAMES:
        if baseline_bundle is None:
            raise RuntimeError("selected baseline policy has no model bundle")
        outcome, rows, _ = run_phase1_baseline_policy(
            **common,
            bundle=baseline_bundle,
            policy_name=policy_name,
        )
        return outcome, rows, []
    if policy_name == "phase1_action_gain":
        outcome, rows, _ = run_one_switch_policy(
            **common,
            bundle=phase1_bundle,
        )
        return outcome, rows, []
    if policy_name == V2_REGRESSION_POLICY_NAME:
        if v2_regression_bundle is None:
            raise RuntimeError("v2 regression policy has no model bundle")
        outcome, rows, _ = run_one_switch_policy(
            **common,
            bundle=v2_regression_bundle,
        )
        return outcome, rows, []
    if policy_name == "phase2_m4_one_switch":
        if phase2_bundle is None:
            raise RuntimeError("Phase 2 one-switch policy has no model bundle")
        outcome, rows, switch_rows, _ = run_phase2_one_switch_policy(
            **common,
            bundle=phase2_bundle,
        )
        return outcome, rows, switch_rows
    if policy_name == "repeated_das":
        if phase2_bundle is None:
            raise RuntimeError("repeated DAS policy has no model bundle")
        outcome, rows, switch_rows, _ = run_repeated_das_policy(
            **common,
            bundle=phase2_bundle,
        )
        return outcome, rows, switch_rows
    if policy_name in QUERY_POLICY_MAP:
        if phase3_bundle is None:
            raise RuntimeError("selected Query policy has no Phase 3 model bundle")
        outcome, rows, _ = run_adaptive_query_policy(
            **common,
            bundle=phase3_bundle,
            gfe_bundle=gfe_bundle,
            query_policy=QUERY_POLICY_MAP[policy_name],
        )
        return outcome, rows, []
    raise ValueError(f"unsupported online comparison policy: {policy_name}")


def _normalize_outcome(
    outcome: dict,
    *,
    policy_name: str,
    default_algorithm: str,
) -> dict:
    row = dict(outcome)
    row["comparison_protocol"] = ONLINE_COMPARISON_PROTOCOL
    row["policy_name"] = policy_name
    row["default_algorithm"] = default_algorithm
    row.setdefault("comparison_prefix_algorithm", row.get("prefix_algorithm"))
    row.setdefault("query_FE", int(row.get("total_query_FE", 0)))
    row.setdefault("switch_count", int(bool(row.get("handoff_required", False))))
    row.setdefault("selected_FE", row.get("first_switch_FE"))
    row.setdefault("selected_decision_opportunity_index", None)
    required = (
        "split",
        "suite",
        "problem_id",
        "function_id",
        "family",
        "cv_group_id",
        "dimension",
        "prefix_algorithm",
        "seed",
        "FE_total",
        "policy_status",
        "effective_FE",
        "final_gap",
        "log10_gap",
        "selected_algorithm",
        "selected_equals_default",
        "selected_equals_prefix",
        "handoff_required",
        "handoff_type",
        "boundary_handling",
    )
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError(f"online policy outcome is missing fields: {missing}")
    effective_fe = int(row["effective_FE"])
    total_fe = int(row["FE_total"])
    if effective_fe > total_fe:
        raise RuntimeError("online policy outcome exceeds the total FE budget")
    if str(row["policy_status"]) == "completed" and effective_fe != total_fe:
        raise RuntimeError("completed online policy does not use the strict total FE budget")
    if bool(row["handoff_required"]) != (not bool(row["selected_equals_prefix"])):
        raise RuntimeError("online policy handoff relation is inconsistent")
    expected_handoff = (
        "population_transfer_initialization"
        if bool(row["handoff_required"])
        else "native_optimizer_state"
    )
    if str(row["handoff_type"]) != expected_handoff:
        raise RuntimeError("online policy handoff type is inconsistent")
    return row


def _timing_row(
    outcome: dict,
    *,
    repetition: int,
    execution_order: int,
    policy_count: int,
    elapsed: float,
) -> dict:
    status = str(outcome["policy_status"])
    timing_status = status if status in {"completed", "timed_out", "failed"} else "failed"
    return {
        "comparison_protocol": ONLINE_COMPARISON_PROTOCOL,
        "policy_name": outcome["policy_name"],
        "split": outcome["split"],
        "suite": outcome["suite"],
        "problem_id": outcome["problem_id"],
        "function_id": outcome["function_id"],
        "family": outcome["family"],
        "cv_group_id": outcome["cv_group_id"],
        "dimension": outcome["dimension"],
        "prefix_algorithm": outcome["prefix_algorithm"],
        "seed": outcome["seed"],
        "timing_repetition": int(repetition),
        "cyclic_execution_order": int(execution_order),
        "cyclic_policy_count": int(policy_count),
        "runtime_complete_policy_path_seconds": float(elapsed),
        "timing_source": TIMING_SOURCE,
        "timing_replay_status": timing_status,
        "effective_FE": int(outcome["effective_FE"]),
        "final_gap": float(outcome["final_gap"]),
        "log10_gap": float(outcome["log10_gap"]),
        "first_hit_FE": outcome.get("first_hit_FE"),
        "selected_algorithm": outcome["selected_algorithm"],
        "selected_FE": outcome.get("selected_FE"),
        "query_FE": int(outcome.get("query_FE", 0)),
        "switch_count": int(outcome.get("switch_count", 0)),
    }


def _validate_complete_timings(outcomes: pd.DataFrame, timings: pd.DataFrame) -> None:
    key = [
        "split",
        "problem_id",
        "prefix_algorithm",
        "seed",
        "policy_name",
    ]
    counts = timings.groupby(key, sort=False)["timing_repetition"].nunique()
    if not counts.eq(TIMING_REPETITIONS).all():
        raise RuntimeError("each policy run must contain three timing repetitions")
    if not timings["timing_source"].astype(str).eq(TIMING_SOURCE).all():
        raise RuntimeError("complete-path timing source is inconsistent")
    total_fe = int(outcomes["FE_total"].iloc[0])
    if timings["effective_FE"].astype(int).gt(total_fe).any():
        raise RuntimeError("a timing replay exceeds the total FE budget")
    completed = timings["timing_replay_status"].astype(str).eq("completed")
    if not timings.loc[completed, "effective_FE"].astype(int).eq(total_fe).all():
        raise RuntimeError("a completed timing replay does not use the strict total FE budget")
    for _, rows in timings.groupby(key, sort=False):
        first = rows.iloc[0]
        if not rows["timing_replay_status"].astype(str).eq(
            str(first["timing_replay_status"])
        ).all():
            raise RuntimeError("timing replay statuses differ within one policy run")
        if not np.allclose(
            rows["final_gap"].to_numpy(dtype=float),
            float(first["final_gap"]),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise RuntimeError("timing replays produced different scientific endpoints")
        for column in ("selected_algorithm", "query_FE", "switch_count"):
            if rows[column].nunique(dropna=False) != 1:
                raise RuntimeError(
                    f"timing replays differ in scientific policy field {column}"
                )


def _attach_measured_timing_medians(
    outcomes: pd.DataFrame,
    timings: pd.DataFrame,
) -> pd.DataFrame:
    key = [
        "split",
        "problem_id",
        "prefix_algorithm",
        "seed",
        "policy_name",
    ]
    summary = (
        timings.groupby(key, as_index=False)["runtime_complete_policy_path_seconds"]
        .median()
        .rename(
            columns={
                "runtime_complete_policy_path_seconds": (
                    "runtime_complete_policy_path_median_seconds"
                )
            }
        )
    )
    summary["timing_repetitions"] = TIMING_REPETITIONS
    summary["timing_source"] = TIMING_SOURCE
    return outcomes.merge(summary, on=key, how="inner", validate="one_to_one")


def _measure_static_portfolio_reference(
    *,
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    instance: int,
    default_algorithm: str,
) -> list[dict]:
    candidates: list[dict] = []
    for seed in config.seeds:
        for algorithm in config.algorithms:
            outcome, _, _ = run_static_optimizer_policy(
                config=config,
                suite=suite,
                function=function,
                instance=instance,
                seed=seed,
                prefix_algorithm=algorithm,
                default_algorithm=default_algorithm,
                policy_name="continue_current",
            )
            normalized = _normalize_outcome(
                outcome,
                policy_name="continue_current",
                default_algorithm=default_algorithm,
            )
            candidates.append(
                {
                    "reference_name": "best_observed_static",
                    "reference_protocol": (
                        "problem_level_seed_aggregated_complete_budget_log10_gap"
                    ),
                    "reference_role": "analysis_only",
                    "deployable_policy": False,
                    "selection_uses_per_seed_hindsight": False,
                    "split": normalized["split"],
                    "suite": normalized["suite"],
                    "problem_id": normalized["problem_id"],
                    "function_id": normalized["function_id"],
                    "family": normalized["family"],
                    "cv_group_id": normalized["cv_group_id"],
                    "dimension": normalized["dimension"],
                    "seed": normalized["seed"],
                    "candidate_algorithm": algorithm,
                    "candidate_status": normalized["policy_status"],
                    "effective_FE": normalized["effective_FE"],
                    "final_gap": normalized["final_gap"],
                    "log10_gap": normalized["log10_gap"],
                    "success": normalized.get("success"),
                    "first_hit_FE": normalized.get("first_hit_FE"),
                    "FE_total": normalized["FE_total"],
                    "boundary_handling": normalized["boundary_handling"],
                }
            )
    frame = pd.DataFrame(candidates)
    expected = len(config.seeds) * len(config.algorithms)
    if len(frame) != expected:
        raise RuntimeError("static reference does not cover every seed-algorithm pair")
    counts = frame.groupby("candidate_algorithm", sort=False)["seed"].nunique()
    if set(counts.index.astype(str)) != set(config.algorithms) or not counts.eq(
        len(config.seeds)
    ).all():
        raise RuntimeError("static reference has incomplete portfolio coverage")
    scores = frame.groupby("candidate_algorithm", sort=False)["log10_gap"].mean()
    minimum = float(scores.min())
    selected = next(
        algorithm
        for algorithm in config.algorithms
        if np.isclose(float(scores.loc[algorithm]), minimum, rtol=0.0, atol=1e-12)
    )
    frame["selected_best_observed_static"] = frame[
        "candidate_algorithm"
    ].astype(str).eq(selected)
    frame["selected_algorithm"] = selected
    frame["selected_algorithm_seed_mean_log10_gap"] = float(scores.loc[selected])
    frame["selection_tie_order"] = ",".join(config.algorithms)
    return frame.to_dict(orient="records")


def _attach_policy_references(
    outcomes: pd.DataFrame,
    static_reference: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["split", "problem_id", "seed"]
    sbs = outcomes.loc[outcomes["policy_name"].astype(str).eq("sbs")]
    if sbs.empty:
        result = outcomes.copy()
        result["sbs_log10_gap"] = np.nan
        result["log10_gain_over_sbs"] = np.nan
    else:
        sbs = sbs[[*keys, "log10_gap"]].rename(
            columns={"log10_gap": "sbs_log10_gap"}
        )
        if sbs.duplicated(keys).any():
            raise RuntimeError("SBS online reference contains duplicate problem-seed rows")
        result = outcomes.merge(sbs, on=keys, how="left", validate="many_to_one")
        result["log10_gain_over_sbs"] = (
            result["sbs_log10_gap"] - result["log10_gap"]
        )
    best_static = static_reference.loc[
        static_reference["selected_best_observed_static"].astype(bool),
        [*keys, "selected_algorithm", "log10_gap"],
    ].rename(
        columns={
            "selected_algorithm": "best_observed_static_algorithm",
            "log10_gap": "best_observed_static_log10_gap",
        }
    )
    if best_static.duplicated(keys).any():
        raise RuntimeError("best-observed static reference contains duplicate rows")
    expected_reference_rows = outcomes[keys].drop_duplicates().shape[0]
    if len(best_static) != expected_reference_rows:
        raise RuntimeError("best-observed static reference coverage differs from outcomes")
    result = result.merge(
        best_static,
        on=keys,
        how="left",
        validate="many_to_one",
    )
    result["sbs_minus_best_observed_static_log10_gap"] = (
        result["sbs_log10_gap"] - result["best_observed_static_log10_gap"]
    )
    result["policy_minus_best_observed_static_log10_gap"] = (
        result["log10_gap"] - result["best_observed_static_log10_gap"]
    )
    continue_reference = outcomes.loc[
        outcomes["policy_name"].astype(str).eq("continue_current"),
        [*keys, "prefix_algorithm", "log10_gap"],
    ].rename(
        columns={
            "prefix_algorithm": "comparison_prefix_algorithm",
            "log10_gap": "continue_log10_gap",
        }
    )
    continue_keys = [*keys, "comparison_prefix_algorithm"]
    if continue_reference.duplicated(continue_keys).any():
        raise RuntimeError("continue-current reference contains duplicate rows")
    result = result.merge(
        continue_reference,
        on=continue_keys,
        how="left",
        validate="many_to_one",
    )
    result["log10_gain_over_continue"] = (
        result["continue_log10_gap"] - result["log10_gap"]
    )
    return result


def _summarize_policies(outcomes: pd.DataFrame) -> pd.DataFrame:
    function_rows = (
        outcomes.groupby(
            ["split", "policy_name", "cv_group_id"],
            as_index=False,
        )
        .agg(
            mean_log10_gap=("log10_gap", "mean"),
            success_rate=("success", "mean"),
            mean_gain_over_continue=("log10_gain_over_continue", "mean"),
            mean_gain_over_sbs=("log10_gain_over_sbs", "mean"),
            mean_sbs_log10_gap=("sbs_log10_gap", "mean"),
            mean_best_observed_static_log10_gap=(
                "best_observed_static_log10_gap",
                "mean",
            ),
            mean_query_FE=("query_FE", "mean"),
            mean_switch_count=("switch_count", "mean"),
            mean_runtime_seconds=(
                "runtime_complete_policy_path_median_seconds",
                "mean",
            ),
            run_count=("problem_id", "size"),
        )
    )
    denominator = (
        function_rows["mean_sbs_log10_gap"]
        - function_rows["mean_best_observed_static_log10_gap"]
    ).to_numpy(dtype=float)
    numerator = (
        function_rows["mean_sbs_log10_gap"]
        - function_rows["mean_log10_gap"]
    ).to_numpy(dtype=float)
    function_rows["static_performance_gap_closed_fraction"] = np.divide(
        numerator,
        denominator,
        out=np.full(len(function_rows), np.nan, dtype=float),
        where=denominator > 1e-12,
    )
    return (
        function_rows.groupby(["split", "policy_name"], as_index=False)
        .agg(
            function_balanced_mean_log10_gap=("mean_log10_gap", "mean"),
            function_balanced_success_rate=("success_rate", "mean"),
            function_balanced_mean_gain_over_continue=(
                "mean_gain_over_continue",
                "mean",
            ),
            function_balanced_mean_gain_over_sbs=("mean_gain_over_sbs", "mean"),
            function_balanced_static_performance_gap_closed_fraction=(
                "static_performance_gap_closed_fraction",
                "mean",
            ),
            function_balanced_mean_query_FE=("mean_query_FE", "mean"),
            function_balanced_mean_switch_count=("mean_switch_count", "mean"),
            function_balanced_mean_runtime_seconds=(
                "mean_runtime_seconds",
                "mean",
            ),
            run_count=("run_count", "sum"),
            function_count=("cv_group_id", "nunique"),
        )
    )


def paired_policy_contrasts(outcomes: pd.DataFrame) -> pd.DataFrame:
    keys = ["split", "problem_id", "prefix_algorithm", "seed"]
    required = {
        *keys,
        "function_id",
        "family",
        "cv_group_id",
        "policy_name",
        "log10_gap",
        "query_FE",
        "switch_count",
    }
    missing = sorted(required.difference(outcomes.columns))
    if missing:
        raise ValueError(f"online outcomes are missing contrast columns: {missing}")
    definitions = (
        (
            "phase2_m4_one_switch",
            "repeated_das",
            "repeated_das_vs_phase2_m4_one_switch",
        ),
        (
            "query_uncertainty",
            "query_voi",
            "query_voi_vs_uncertainty_query",
        ),
        (
            "g_fe_query_gate",
            "query_voi",
            "query_voi_vs_g_fe_query_gate_observed_rates",
        ),
        (
            "g_fe_query_gate_matched_rate",
            "query_voi",
            "query_voi_vs_g_fe_query_gate_train_matched_rate",
        ),
    )
    rows: list[pd.DataFrame] = []
    metadata = [*keys, "function_id", "family", "cv_group_id"]
    values = ["log10_gap", "query_FE", "switch_count"]
    for reference_policy, evaluated_policy, contrast_name in definitions:
        reference = outcomes.loc[
            outcomes["policy_name"].astype(str).eq(reference_policy),
            [*metadata, *values],
        ].rename(
            columns={column: f"reference_{column}" for column in values}
        )
        evaluated = outcomes.loc[
            outcomes["policy_name"].astype(str).eq(evaluated_policy),
            [*keys, *values],
        ].rename(
            columns={column: f"evaluated_{column}" for column in values}
        )
        if reference.empty or evaluated.empty:
            continue
        if reference.duplicated(keys).any() or evaluated.duplicated(keys).any():
            raise RuntimeError(
                f"paired policy contrast contains duplicate run keys: {contrast_name}"
            )
        paired = reference.merge(
            evaluated,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(reference) or len(paired) != len(evaluated):
            raise RuntimeError(
                f"paired policy contrast coverage differs: {contrast_name}"
            )
        paired["contrast_name"] = contrast_name
        paired["reference_policy"] = reference_policy
        paired["evaluated_policy"] = evaluated_policy
        paired["terminal_log10_gain_evaluated_over_reference"] = (
            paired["reference_log10_gap"] - paired["evaluated_log10_gap"]
        )
        paired["query_FE_evaluated_minus_reference"] = (
            paired["evaluated_query_FE"] - paired["reference_query_FE"]
        )
        paired["switch_count_evaluated_minus_reference"] = (
            paired["evaluated_switch_count"]
            - paired["reference_switch_count"]
        )
        rows.append(paired)
    if rows:
        return pd.concat(rows, ignore_index=True).sort_values(
            ["contrast_name", *keys],
            kind="mergesort",
        ).reset_index(drop=True)
    return pd.DataFrame(
        columns=[
            *metadata,
            "reference_log10_gap",
            "reference_query_FE",
            "reference_switch_count",
            "evaluated_log10_gap",
            "evaluated_query_FE",
            "evaluated_switch_count",
            "contrast_name",
            "reference_policy",
            "evaluated_policy",
            "terminal_log10_gain_evaluated_over_reference",
            "query_FE_evaluated_minus_reference",
            "switch_count_evaluated_minus_reference",
        ]
    )


def summarize_paired_policy_contrasts(
    contrasts: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "split",
        "contrast_name",
        "reference_policy",
        "evaluated_policy",
        "run_count",
        "function_count",
        "function_balanced_mean_terminal_log10_gain",
        "median_terminal_log10_gain",
        "fraction_terminal_log10_gain_gt_zero",
        "function_balanced_mean_query_FE_difference",
        "function_balanced_mean_switch_count_difference",
    ]
    if contrasts.empty:
        return pd.DataFrame(columns=columns)
    function_rows = (
        contrasts.groupby(
            [
                "split",
                "contrast_name",
                "reference_policy",
                "evaluated_policy",
                "cv_group_id",
            ],
            as_index=False,
        )
        .agg(
            mean_terminal_log10_gain=(
                "terminal_log10_gain_evaluated_over_reference",
                "mean",
            ),
            mean_query_FE_difference=(
                "query_FE_evaluated_minus_reference",
                "mean",
            ),
            mean_switch_count_difference=(
                "switch_count_evaluated_minus_reference",
                "mean",
            ),
            run_count=("problem_id", "size"),
        )
    )
    summary = (
        function_rows.groupby(
            [
                "split",
                "contrast_name",
                "reference_policy",
                "evaluated_policy",
            ],
            as_index=False,
        )
        .agg(
            function_balanced_mean_terminal_log10_gain=(
                "mean_terminal_log10_gain",
                "mean",
            ),
            function_balanced_mean_query_FE_difference=(
                "mean_query_FE_difference",
                "mean",
            ),
            function_balanced_mean_switch_count_difference=(
                "mean_switch_count_difference",
                "mean",
            ),
            run_count=("run_count", "sum"),
            function_count=("cv_group_id", "nunique"),
        )
    )
    medians = contrasts.groupby(
        ["split", "contrast_name", "reference_policy", "evaluated_policy"],
        as_index=False,
    ).agg(
        median_terminal_log10_gain=(
            "terminal_log10_gain_evaluated_over_reference",
            "median",
        ),
        fraction_terminal_log10_gain_gt_zero=(
            "terminal_log10_gain_evaluated_over_reference",
            lambda values: float(np.mean(np.asarray(values, dtype=float) > 0.0)),
        ),
    )
    return summary.merge(
        medians,
        on=["split", "contrast_name", "reference_policy", "evaluated_policy"],
        how="inner",
        validate="one_to_one",
    )[columns]


def _cyclic_order(policies: tuple[str, ...], repetition: int) -> tuple[str, ...]:
    if not policies:
        raise ValueError("cyclic timing order requires at least one policy")
    shift = int(repetition) % len(policies)
    return (*policies[shift:], *policies[:shift])


def _validate_policies(policies: tuple[str, ...]) -> tuple[str, ...]:
    values = tuple(str(value) for value in policies)
    if not values:
        raise ValueError("at least one online comparison policy is required")
    if len(set(values)) != len(values):
        raise ValueError("online comparison policies must be unique")
    invalid = sorted(set(values).difference(ONLINE_COMPARISON_POLICIES))
    if invalid:
        raise ValueError(f"unsupported online comparison policies: {invalid}")
    return values


def _load_required_bundle(path: str | Path, *, required: bool) -> dict | None:
    source = Path(path)
    if not required:
        return None
    if not source.exists():
        raise FileNotFoundError(f"missing required online model bundle: {source}")
    return joblib.load(source)


def _validate_baseline_default(
    baseline_bundle: dict[str, Any],
    phase1_bundle: dict[str, Any],
) -> None:
    if str(baseline_bundle.get("default_algorithm")) != str(
        phase1_bundle["default_algorithm"]
    ):
        raise ValueError("baseline and Phase 1 bundles use different default algorithms")


def _initial_algorithms(
    value: str,
    default_algorithm: str,
    config: ExperimentConfig,
) -> tuple[str, ...]:
    name = str(value).lower()
    if name == "sbs":
        return (default_algorithm,)
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


def _write_optional_rows(frame: pd.DataFrame, path: Path, *, columns: list[str]) -> None:
    if frame.empty:
        pd.DataFrame(columns=columns).to_parquet(path, index=False)
        return
    existing = [column for column in columns if column in frame.columns]
    frame.sort_values(existing, kind="mergesort").to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run complete online policy paths with three cyclic timing repetitions."
    )
    parser.add_argument("--config", default="configs/behavior_with_ela_cec2017.yaml")
    parser.add_argument(
        "--phase1-model",
        default=(
            "results/behavior_with_ela/model/behavior_action_gain/models.joblib"
        ),
    )
    parser.add_argument(
        "--baseline-model",
        default=(
            "results/behavior_with_ela/baselines/phase1/baseline_models.joblib"
        ),
    )
    parser.add_argument(
        "--phase2-model",
        default=(
            "results/behavior_with_ela/model/local_landscape_increment/"
            "phase2_models.joblib"
        ),
    )
    parser.add_argument(
        "--phase3-model",
        default="results/behavior_with_ela/model/adaptive_query/phase3_models.joblib",
    )
    parser.add_argument(
        "--gfe-model",
        default="results/behavior_with_ela/model/gfe_query_gate/gfe_gate_model.joblib",
    )
    parser.add_argument(
        "--traditional-aas-model",
        default=(
            "results/behavior_with_ela/model/traditional_aas/"
            "traditional_aas_models.joblib"
        ),
    )
    parser.add_argument(
        "--v2-regression-model",
        default=DEFAULT_V2_REGRESSION_MODEL,
    )
    parser.add_argument(
        "--output",
        default="results/behavior_with_ela/online/complete_comparison",
    )
    parser.add_argument("--policy", action="append", default=None)
    parser.add_argument("--only-split", action="append", default=None)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--initial-algorithm", default="sbs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = evaluate_online_comparison(
        config_path=args.config,
        phase1_model_path=args.phase1_model,
        baseline_model_path=args.baseline_model,
        phase2_model_path=args.phase2_model,
        phase3_model_path=args.phase3_model,
        gfe_model_path=args.gfe_model,
        traditional_aas_model_path=args.traditional_aas_model,
        v2_regression_model_path=args.v2_regression_model,
        output_dir=args.output,
        policies=(
            ONLINE_COMPARISON_POLICIES
            if args.policy is None
            else tuple(args.policy)
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
        f"completed {summary['policy_outcomes']} online policy runs with "
        f"{summary['timing_rows']} measured complete-path timings"
    )


if __name__ == "__main__":
    main()
