from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from benchmarks import make_problem
from decision.matched_random import make_matched_random_calibration, matched_random_target
from decision.model_protocol import PREDEFINED_THRESHOLD_MODE, SELECTED_MODEL_ALIAS
from decision.online_controller_evaluate import (
    DecisionControllerModel,
    OnlineSelector,
    _execute_online_policy_path,
    _load_controller,
    _online_timing_base_order,
    _prepare_online_query_inputs,
)
from experiments.phase1_batch_common import (
    load_config,
    selected_dimensions,
    selected_functions,
    validate_dynamic_collection_config,
)
from landscape_queries.specs import MAIN_QUERY_ID, get_query_spec
from optimizers import OptimizerSettings, advance_optimizer_state, initialize_optimizer_state
from optimizers.lshade import advance_lshade_state, initialize_lshade_state
from selection_reference.model import load_selector_model


DEFAULT_CONFIG = Path("configs/cec2017_representative_online_compare.yaml")
DEFAULT_TRAINING_SUMMARY = Path(
    "outputs/recompute_20260825_maturity_ablation/search_maturity_linear/decision/"
    "full_decision_model_training_summary.json"
)
DEFAULT_SELECTOR_MODEL = Path(
    "outputs/recompute_20260825_maturity_ablation/search_maturity_linear/decision/"
    "models/selector__query_full.joblib"
)
TIMING_REPETITIONS = 3
RANDOM_REPETITION = 0
POLICY_ORDER = (
    "predicted_G_FE_gt_0",
    "predicted_g_fe_selected_path_gt_0.2997557291",
    "cmaes",
    "lshade",
    "random_ela",
)


def run_experiment(
    *,
    config_path: Path,
    training_summary_path: Path,
    selector_model_path: Path,
    output_dir: Path,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    only_seeds: list[int] | None,
    overwrite: bool,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"输出目录已有内容：{output_dir}；如需重跑请显式传入 --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    validate_dynamic_collection_config(config)
    if str(config.get("suite", "")).lower() != "cec2017":
        raise ValueError("本入口只接受 CEC2017 配置")
    query_id = str(config.get("query_id", MAIN_QUERY_ID))
    if query_id != MAIN_QUERY_ID:
        raise ValueError("当前在线真实 query 入口只支持 descriptor_cheap_invariant")
    query_spec = get_query_spec(query_id)
    functions = selected_functions(config, only_functions)
    dimensions = selected_dimensions(config, only_dimensions)
    seeds = _selected_seeds(config, only_seeds)

    controller_zero = _load_controller(
        training_summary_path,
        SELECTED_MODEL_ALIAS,
        "zero",
    )
    controller_oof = _load_controller(
        training_summary_path,
        SELECTED_MODEL_ALIAS,
        PREDEFINED_THRESHOLD_MODE,
    )
    selector = OnlineSelector(load_selector_model(selector_model_path))
    _validate_feature_contract(controller_oof, selector)
    if selector.model.query_id != query_id:
        raise ValueError("固定 Selector 与本次 query_id 不一致")
    if not np.isclose(
        float(controller_oof.threshold),
        0.2997557291,
        rtol=0.0,
        atol=2e-10,
    ):
        raise ValueError(
            "线形成熟度模型的预先指定阈值不是 0.2997557291："
            f"{controller_oof.threshold!r}"
        )
    if float(controller_zero.threshold) != 0.0:
        raise ValueError("zero threshold must equal 0")

    calibration = _make_random_calibration(
        training_summary_path=training_summary_path,
        controller=controller_oof,
        query_id=query_id,
        output_dir=output_dir,
        config=config,
    )
    query_feature_rows, query_sample_rows = _prepare_online_query_inputs(
        config=config,
        query_id=query_id,
        query_spec=query_spec,
        functions=functions,
        dimensions=dimensions,
    )
    _write_query_inputs(
        output_dir=output_dir,
        query_feature_rows=query_feature_rows,
        query_sample_rows=query_sample_rows,
    )

    run_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for function in functions:
        for dimension in dimensions:
            for seed in seeds:
                print(f"CEC2017 F{function:02d} D{dimension} seed={seed}", flush=True)
                aggregate_rows, replay_rows = _run_problem_seed(
                    config=config,
                    function=function,
                    dimension=dimension,
                    seed=seed,
                    controller_zero=controller_zero,
                    controller_oof=controller_oof,
                    selector=selector,
                    calibration=calibration,
                    query_feature_row=query_feature_rows[(function, dimension)],
                    query_sample_row=query_sample_rows[(function, dimension)],
                )
                run_rows.extend(aggregate_rows)
                timing_rows.extend(replay_rows)

    run_frame = pd.DataFrame(run_rows)
    timing_frame = pd.DataFrame(timing_rows)
    if run_frame.empty or timing_frame.empty:
        raise ValueError("在线测评没有生成结果")
    _validate_result_coverage(
        run_frame,
        functions=functions,
        dimensions=dimensions,
        seeds=seeds,
    )
    summary_frame = _summarize(run_frame)
    _write_frame(run_frame, output_dir / "online_comparison_run_metrics")
    _write_frame(timing_frame, output_dir / "online_comparison_timing_replays")
    _write_frame(summary_frame, output_dir / "online_comparison_summary")

    config_snapshot = output_dir / "experiment_config_snapshot.yaml"
    config_snapshot.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    report = _make_report(
        config=config,
        config_path=config_path,
        training_summary_path=training_summary_path,
        selector_model_path=selector_model_path,
        run_frame=run_frame,
        summary_frame=summary_frame,
        calibration=calibration,
    )
    report_path = output_dir / "online_comparison_report.md"
    report_path.write_text(report, encoding="utf-8")
    summary = {
        "status": "ok",
        "experiment": "cec2017_representative_online_compare",
        "research_question": "比较固定线形成熟度 Decision + fixed Selector 与 CMA-ES、LSHADE、随机 ELA 的在线优化能力和资源指标",
        "functions": functions,
        "dimensions": dimensions,
        "seeds": seeds,
        "policies": list(POLICY_ORDER),
        "population_size": int(config["population_size"]),
        "fe_total_by_dimension": {
            str(dimension): int(config["FE_total_by_dimension"][dimension])
            for dimension in dimensions
        },
        "controller_zero_threshold": float(controller_zero.threshold),
        "controller_oof_threshold": float(controller_oof.threshold),
        "random_ela_calibration": calibration.payload(),
        "selector_type": str(getattr(selector.model, "selector_type", "")),
        "selector_default_algorithm": selector.sbs_algorithm,
        "decision_model": controller_oof.model_name,
        "decision_feature_group": controller_oof.feature_group,
        "decision_feature_count": len(controller_oof.feature_columns),
        "selector_feature_count": len(selector.model.feature_columns),
        "timing_repetitions": TIMING_REPETITIONS,
        "timing_order_protocol": "cyclic_complete_path_v1",
        "timing_source": "measured_complete_policy_path",
        "result_rows": int(len(run_frame)),
        "timing_rows": int(len(timing_frame)),
        "outputs": {
            "run_metrics": str(output_dir / "online_comparison_run_metrics.parquet"),
            "timing_replays": str(output_dir / "online_comparison_timing_replays.parquet"),
            "summary": str(output_dir / "online_comparison_summary.parquet"),
            "report": str(report_path),
            "config": str(config_snapshot),
            "random_calibration": str(output_dir / "random_ela_calibration.json"),
        },
        "data_leakage_check": {
            "cec2017_rows_used_for_model_fit": 0,
            "cec2017_rows_used_for_selector_fit": 0,
            "validation_rows_used_for_threshold_fit": 0,
            "decision_input_contains_query_descriptors": False,
            "selector_uses_query_descriptors_behavior_and_remaining_budget": True,
            "lshade_is_outside_fixed_selector_portfolio": True,
        },
    }
    (output_dir / "online_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _run_problem_seed(
    *,
    config: dict[str, Any],
    function: int,
    dimension: int,
    seed: int,
    controller_zero: DecisionControllerModel,
    controller_oof: DecisionControllerModel,
    selector: OnlineSelector,
    calibration,
    query_feature_row: dict[str, Any],
    query_sample_row: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fe_total = int(config["FE_total_by_dimension"][dimension])
    random_target = matched_random_target(
        calibration,
        problem_id=f"cec2017_f{function:02d}_d{dimension}",
        prefix_algorithm=selector.sbs_algorithm,
        run_seed=seed,
        dimension=dimension,
        repetition=RANDOM_REPETITION,
    )
    policies = {
        "predicted_G_FE_gt_0": {
            "kind": "online",
            "controller": controller_zero,
            "policy_spec": {"policy_name": "current_controller"},
        },
        "predicted_g_fe_selected_path_gt_0.2997557291": {
            "kind": "online",
            "controller": controller_oof,
            "policy_spec": {"policy_name": "current_controller"},
        },
        "cmaes": {
            "kind": "online",
            "controller": controller_oof,
            "policy_spec": {"policy_name": "sbs_no_query"},
        },
        "random_ela": {
            "kind": "online",
            "controller": controller_oof,
            "policy_spec": {
                "policy_name": "matched_rate_random",
                "random_repetition": RANDOM_REPETITION,
                "matched_trigger_fe_ratio": random_target,
            },
        },
        "lshade": {"kind": "lshade"},
    }
    internal_order = _online_timing_base_order(
        path_count=len(POLICY_ORDER),
        function=function,
        dimension=dimension,
        seed=seed,
        random_repetitions=1,
    )
    stage_a: dict[str, dict[str, Any]] = {}
    replay_rows: list[dict[str, Any]] = []
    for order_position, policy_index in enumerate(internal_order):
        policy_name = POLICY_ORDER[int(policy_index)]
        result = _run_policy_once(
            policy_name=policy_name,
            policy_spec=policies[policy_name],
            config=config,
            function=function,
            dimension=dimension,
            seed=seed,
            fe_total=fe_total,
            selector=selector,
            controller=policies[policy_name].get("controller", controller_oof),
            query_feature_row=query_feature_row,
            query_sample_row=query_sample_row,
        )
        result.update(
            {
                "stage": "stage_a_scientific",
                "timing_repetition": None,
                "timing_order_position": int(order_position),
                "function": int(function),
                "dimension": int(dimension),
                "seed": int(seed),
            }
        )
        stage_a[policy_name] = result

    for timing_repetition in range(TIMING_REPETITIONS):
        rotated = internal_order[timing_repetition:] + internal_order[:timing_repetition]
        for order_position, policy_index in enumerate(rotated):
            policy_name = POLICY_ORDER[int(policy_index)]
            result = _run_policy_once(
                policy_name=policy_name,
                policy_spec=policies[policy_name],
                config=config,
                function=function,
                dimension=dimension,
                seed=seed,
                fe_total=fe_total,
                selector=selector,
                controller=policies[policy_name].get("controller", controller_oof),
                query_feature_row=query_feature_row,
                query_sample_row=query_sample_row,
            )
            result.update(
                {
                    "stage": "stage_b_timing_replay",
                    "timing_source": "measured_complete_policy_path",
                    "timing_repetition": int(timing_repetition),
                    "timing_order_position": int(order_position),
                    "function": int(function),
                    "dimension": int(dimension),
                    "seed": int(seed),
                }
            )
            replay_rows.append(result)

    aggregate_rows = []
    for policy_name in POLICY_ORDER:
        scientific = stage_a[policy_name]
        replays = [
            row for row in replay_rows if row["policy_name"] == policy_name
        ]
        aggregate_rows.append(
            _aggregate_policy_rows(
                scientific=scientific,
                replays=replays,
                policy_name=policy_name,
                function=function,
                dimension=dimension,
                seed=seed,
                random_target=random_target,
            )
        )
    return aggregate_rows, replay_rows


def _run_policy_once(
    *,
    policy_name: str,
    policy_spec: dict[str, Any],
    config: dict[str, Any],
    function: int,
    dimension: int,
    seed: int,
    fe_total: int,
    selector: OnlineSelector,
    controller: DecisionControllerModel,
    query_feature_row: dict[str, Any],
    query_sample_row: dict[str, Any],
) -> dict[str, Any]:
    if policy_spec["kind"] == "lshade":
        return _run_lshade_once(
            config=config,
            function=function,
            dimension=dimension,
            seed=seed,
            fe_total=fe_total,
        ) | {"policy_name": policy_name}

    internal_name = str(policy_spec["policy_spec"]["policy_name"])
    result = _execute_online_policy_path(
        config=config,
        suite="cec2017",
        function=function,
        dimension=dimension,
        seed=seed,
        fe_total=fe_total,
        controller=controller,
        milestone_only_controller=controller,
        behavior_only_controller=controller,
        selector=selector,
        behavior_only_selector=selector,
        pre_run_aas_selector=selector,
        query_feature_row=query_feature_row,
        query_sample_row=query_sample_row,
        policy_spec=policy_spec["policy_spec"],
        sampling_protocol=str(config["sampling_protocol"]),
        decision_check_frequency="dynamic_budget_milestones_and_state_events",
    )
    result["policy_name"] = policy_name
    result["policy_internal_name"] = internal_name
    return result


def _run_lshade_once(
    *,
    config: dict[str, Any],
    function: int,
    dimension: int,
    seed: int,
    fe_total: int,
) -> dict[str, Any]:
    started = perf_counter()
    reference: float | None = None
    first_hit_fe: int | None = None
    first_hit_seconds: float | None = None
    evaluations = 0
    best_value = float("inf")
    failure_class = ""
    failure_message = ""
    problem = None
    try:
        problem = make_problem(
            {
                "suite": "cec2017",
                "function": int(function),
                "instance": 1,
                "dimension": int(dimension),
                "boundary_handling": str(config.get("boundary_handling", "reflect")),
            }
        )
        reference = float(problem.reference_value)
        target = float(config["success_gap_target"])

        def observe(point: np.ndarray, value: float) -> None:
            nonlocal evaluations, best_value, first_hit_fe, first_hit_seconds
            evaluations += 1
            best_value = min(best_value, float(value))
            if first_hit_fe is None and max(float(value) - reference, 0.0) <= target:
                first_hit_fe = int(evaluations)
                first_hit_seconds = float(perf_counter() - started)

        settings = OptimizerSettings(
            population_size=int(config["population_size"]),
            checkpoint_ratios=(1.0,),
            boundary_handling=str(config.get("boundary_handling", "reflect")),
        )
        state = initialize_lshade_state(
            problem=problem,
            seed=int(seed),
            settings=settings,
            fe_total=int(fe_total),
            on_evaluation=observe,
        )
        advance_lshade_state(
            state=state,
            problem=problem,
            fe_budget=int(fe_total) - int(state.evaluations),
            on_evaluation=observe,
        )
        if evaluations != int(fe_total):
            raise RuntimeError(
                f"LSHADE objective-evaluation count={evaluations}, expected={fe_total}"
            )
    except Exception as exc:
        failure_class = type(exc).__name__
        failure_message = str(exc)[:500]
    finally:
        if problem is not None:
            problem.close()
    runtime = float(perf_counter() - started)
    if runtime <= 0.0 or not np.isfinite(runtime):
        raise RuntimeError("LSHADE runtime measurement is invalid")
    completed = not failure_class and evaluations == int(fe_total)
    target_hit = first_hit_fe is not None
    endpoint_gap = (
        max(best_value - float(reference), 0.0)
        if completed and reference is not None
        else float(config["failure_loss_cap"])
    )
    clipped_gap = np.clip(
        endpoint_gap,
        float(config["log10_gap_floor"]),
        float(config["log10_gap_cap"]),
    )
    return {
        "policy_name": "lshade",
        "prefix_algorithm": "lshade",
        "default_algorithm": "lshade",
        "selected_algorithm": "lshade",
        "query_called": False,
        "trigger_FE": None,
        "selected_equals_prefix": True,
        "handoff_required": False,
        "FE_total": int(fe_total),
        "FE_used": int(evaluations),
        "observed_first_hit_FE": first_hit_fe,
        "first_hit_FE": first_hit_fe,
        "first_hit_wall_clock_seconds": first_hit_seconds,
        "target_hit_observed": target_hit,
        "endpoint_success": bool(target_hit and completed),
        "path_completed": completed,
        "path_status": "completed" if completed else "failed",
        "ert_FE_contribution": int(first_hit_fe) if target_hit else int(fe_total),
        "runtime_full_run_wall_clock": runtime,
        "runtime_full_run_wall_clock_median": runtime,
        "time_to_target_seconds": first_hit_seconds,
        "final_gap": float(endpoint_gap),
        "log10_gap": float(np.log10(clipped_gap)),
        "observed_final_gap": float(endpoint_gap),
        "failure_class": failure_class,
        "failure_message": failure_message,
        "timing_source": "measured_complete_policy_path",
    }


def _aggregate_policy_rows(
    *,
    scientific: dict[str, Any],
    replays: list[dict[str, Any]],
    policy_name: str,
    function: int,
    dimension: int,
    seed: int,
    random_target: float | None,
) -> dict[str, Any]:
    if len(replays) != TIMING_REPETITIONS:
        raise ValueError(f"{policy_name} must have exactly three timing replays")
    endpoint_fields = (
        "query_called",
        "selected_algorithm",
        "trigger_FE",
        "observed_first_hit_FE",
        "target_hit_observed",
        "endpoint_success",
        "path_status",
        "FE_used",
    )
    endpoint_consistent = all(
        _same_value(scientific.get(field), replay.get(field))
        for replay in replays
        for field in endpoint_fields
    )
    if not endpoint_consistent:
        raise RuntimeError(f"{policy_name} 的在线计时 replay 与科学端点不一致")
    runtime_values = np.asarray(
        [float(row["runtime_full_run_wall_clock"]) for row in replays], dtype=float
    )
    first_hit_values = [
        row.get("first_hit_wall_clock_seconds") for row in replays
    ]
    finite_first_hit = np.asarray(
        [float(value) for value in first_hit_values if value is not None], dtype=float
    )
    target_hit = bool(scientific.get("target_hit_observed", False))
    time_to_target = (
        float(np.median(finite_first_hit))
        if target_hit and len(finite_first_hit) == TIMING_REPETITIONS
        else None
    )
    full_runtime = float(np.median(runtime_values))
    time_ert_contribution = (
        float(time_to_target) if time_to_target is not None else full_runtime
    )
    result = dict(scientific)
    result.update(
        {
            "policy_name": policy_name,
            "function": int(function),
            "dimension": int(dimension),
            "seed": int(seed),
            "random_trigger_target_ratio": random_target,
            "timing_repetitions": TIMING_REPETITIONS,
            "timing_source": "measured_complete_policy_path",
            "runtime_full_run_wall_clock_median": full_runtime,
            "full_run_wall_clock_seconds": full_runtime,
            "time_to_target_seconds_median": time_to_target,
            "time_ert_seconds_contribution": time_ert_contribution,
            "timing_replay_statuses": [str(row.get("path_status")) for row in replays],
            "timing_replay_status_instability": len(
                {str(row.get("path_status")) for row in replays}
            )
            > 1,
            "endpoint_consistent_across_timing_replays": endpoint_consistent,
        }
    )
    return result


def _make_random_calibration(
    *,
    training_summary_path: Path,
    controller: DecisionControllerModel,
    query_id: str,
    output_dir: Path,
    config: dict[str, Any],
):
    train_oof_path = training_summary_path.with_name("train_oof_predictions.parquet")
    if not train_oof_path.exists():
        raise FileNotFoundError(train_oof_path)
    train_oof = pd.read_parquet(train_oof_path)
    train_oof = train_oof[
        train_oof["model_name"].astype(str).eq(str(controller.model_name))
    ].copy()
    call_column = "decision_run_query_oof_g_fe_selected_path_first_trigger"
    required = {"problem_id", "prefix_algorithm", "seed", "FE", "FE_ratio", call_column}
    missing = required.difference(train_oof.columns)
    if missing:
        raise ValueError(f"train OOF calibration table is missing columns: {sorted(missing)}")
    called_runs = 0
    total_runs = 0
    trigger_ratios: list[float] = []
    run_columns = ["problem_id", "prefix_algorithm", "seed"]
    for _, run_frame in train_oof.groupby(run_columns, sort=True, dropna=False):
        ordered = run_frame.sort_values(["FE"], kind="mergesort")
        total_runs += 1
        called = ordered[ordered[call_column].astype(bool)]
        if called.empty:
            continue
        called_runs += 1
        trigger_ratios.append(float(called.iloc[0]["FE_ratio"]))
    if total_runs <= 0:
        raise ValueError("train OOF calibration table has no trajectory runs")
    call_rate = float(called_runs / total_runs)
    trigger_ratios_array = np.asarray(trigger_ratios, dtype=float)
    calibration = make_matched_random_calibration(
        query_id=query_id,
        query_protocol=get_query_spec(query_id).protocol,
        feature_group=controller.feature_group,
        selected_model=controller.model_name,
        threshold_mode=PREDEFINED_THRESHOLD_MODE,
        run_call_rate=call_rate,
        trigger_fe_ratios=trigger_ratios_array,
        seed=int(config.get("random_ela_stream_code", 20260826)),
    )
    calibration.write(output_dir / "random_ela_calibration.json")
    return calibration


def _validate_feature_contract(
    controller: DecisionControllerModel,
    selector: OnlineSelector,
) -> None:
    decision_feature_cols = list(controller.feature_columns)
    selector_feature_cols = list(selector.model.feature_columns)
    if len(decision_feature_cols) != 29:
        raise ValueError(
            f"Expected 29 Decision behavior features, got {len(decision_feature_cols)}"
        )
    if len(selector_feature_cols) != 43:
        raise ValueError(
            f"Expected 43 Selector features, got {len(selector_feature_cols)}"
        )
    if not any(column.startswith("bf_") for column in decision_feature_cols):
        raise ValueError("Missing behavior features")
    if not any(column.startswith("bf_") for column in selector_feature_cols):
        raise ValueError("Selector is missing behavior features")
    if not any(column.startswith("descriptor_") for column in selector_feature_cols):
        raise ValueError("Missing descriptor features")
    if "remaining_budget_ratio" not in selector_feature_cols:
        raise ValueError("Selector is missing remaining_budget_ratio")
    maturity_columns = {
        "bf_search_maturity",
        "bf_search_maturity_linear",
        "bf_explore_exploit_ratio",
    }
    if maturity_columns.intersection(selector_feature_cols):
        raise ValueError("fixed Selector must not use the three Decision maturity fields")


def _aggregate_group(frame: pd.DataFrame, label: str, group: dict[str, Any]) -> dict[str, Any]:
    hits = float(frame["target_hit_observed"].astype(bool).sum())
    target_count = int(len(frame))
    ert_fe = (
        float(frame["ert_FE_contribution"].astype(float).sum() / hits)
        if hits > 0.0
        else None
    )
    ert_time = (
        float(frame["time_ert_seconds_contribution"].astype(float).sum() / hits)
        if hits > 0.0
        else None
    )
    target_times = pd.to_numeric(
        frame["time_to_target_seconds_median"], errors="coerce"
    ).dropna()
    return {
        "layer": label,
        **group,
        "policy_name": str(frame["policy_name"].iloc[0]),
        "runs": target_count,
        "target_hit_observed_rate": float(hits / target_count),
        "endpoint_success_rate": float(
            frame["endpoint_success"].astype(bool).mean()
        ),
        "ERT_FE": ert_fe,
        "ERT_time_seconds": ert_time,
        "mean_time_to_target_seconds_among_hits": (
            float(target_times.mean()) if len(target_times) else None
        ),
        "median_time_to_target_seconds_among_hits": (
            float(target_times.median()) if len(target_times) else None
        ),
        "mean_log10_gap": float(frame["log10_gap"].astype(float).mean()),
        "median_log10_gap": float(frame["log10_gap"].astype(float).median()),
        "mean_final_gap": float(frame["final_gap"].astype(float).mean()),
        "median_full_run_wall_clock_seconds": float(
            frame["full_run_wall_clock_seconds"].astype(float).median()
        ),
        "mean_trigger_FE_ratio": float(
            pd.to_numeric(frame["trigger_FE_ratio"], errors="coerce").mean()
        )
        if "trigger_FE_ratio" in frame and frame["trigger_FE_ratio"].notna().any()
        else None,
        "query_call_rate": float(frame["query_called"].astype(bool).mean()),
        "handoff_rate": float(frame["handoff_required"].astype(bool).mean()),
        "coverage": float(frame["path_completed"].astype(bool).mean()),
        "timing_replay_instability_rate": float(
            frame["timing_replay_status_instability"].astype(bool).mean()
        ),
    }


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy_name, policy_frame in frame.groupby("policy_name", sort=True):
        rows.append(_aggregate_group(policy_frame, "overall", {}))
        for function, group in policy_frame.groupby("function", sort=True):
            rows.append(
                _aggregate_group(
                    group,
                    "function",
                    {"function": int(function)},
                )
            )
        for dimension, group in policy_frame.groupby("dimension", sort=True):
            rows.append(
                _aggregate_group(
                    group,
                    "dimension",
                    {"dimension": int(dimension)},
                )
            )
        for (function, dimension), group in policy_frame.groupby(
            ["function", "dimension"], sort=True
        ):
            rows.append(
                _aggregate_group(
                    group,
                    "function_dimension",
                    {"function": int(function), "dimension": int(dimension)},
                )
            )
    return pd.DataFrame(rows).sort_values(
        ["layer", "policy_name", "function", "dimension"],
        na_position="first",
    ).reset_index(drop=True)


def _validate_result_coverage(
    frame: pd.DataFrame,
    *,
    functions: list[int],
    dimensions: list[int],
    seeds: list[int],
) -> None:
    expected = len(functions) * len(dimensions) * len(seeds) * len(POLICY_ORDER)
    if len(frame) != expected:
        raise ValueError(f"结果行数={len(frame)}，预期={expected}")
    if set(frame["policy_name"].astype(str)) != set(POLICY_ORDER):
        raise ValueError("策略覆盖不完整")
    for policy_name, group in frame.groupby("policy_name", sort=False):
        if len(group) != len(functions) * len(dimensions) * len(seeds):
            raise ValueError(f"策略 {policy_name} 的函数/维度/seed 覆盖不完整")
    if not frame["endpoint_consistent_across_timing_replays"].astype(bool).all():
        raise ValueError("至少一条策略的三次真实计时 replay 端点不一致")


def _write_query_inputs(
    *,
    output_dir: Path,
    query_feature_rows: dict[tuple[int, int], dict[str, Any]],
    query_sample_rows: dict[tuple[int, int], dict[str, Any]],
) -> None:
    (output_dir / "query_features.json").write_text(
        json.dumps(_json_safe(query_feature_rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "query_samples.json").write_text(
        json.dumps(_json_safe(query_sample_rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value) if not isinstance(value, (str, bool, type(None), dict, list, tuple)) else False:
        return None
    return value


def _write_frame(frame: pd.DataFrame, stem: Path) -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), stem.with_suffix(".parquet"))


def _same_value(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        return bool(np.isclose(float(left), float(right), rtol=1e-12, atol=1e-12))
    return left == right


def _selected_seeds(config: dict[str, Any], only_seeds: list[int] | None) -> list[int]:
    configured = [int(seed) for seed in config["seeds"]]
    if only_seeds is None:
        return configured
    requested = [int(seed) for seed in only_seeds]
    missing = sorted(set(requested).difference(configured))
    if missing:
        raise ValueError(f"requested seeds are absent from config: {missing}")
    return [seed for seed in configured if seed in set(requested)]


def _make_report(
    *,
    config: dict[str, Any],
    config_path: Path,
    training_summary_path: Path,
    selector_model_path: Path,
    run_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    calibration,
) -> str:
    overall = summary_frame[summary_frame["layer"] == "overall"].copy()
    columns = [
        "policy_name",
        "runs",
        "target_hit_observed_rate",
        "endpoint_success_rate",
        "ERT_FE",
        "ERT_time_seconds",
        "median_time_to_target_seconds_among_hits",
        "mean_log10_gap",
        "median_full_run_wall_clock_seconds",
        "query_call_rate",
    ]
    table = overall[columns].sort_values("mean_log10_gap").to_markdown(index=False)
    return "\n".join(
        [
            "# CEC2017 代表性函数在线测评比较",
            "",
            "## 测评定义",
            "",
            f"- 配置：`{config_path}`；函数：{config['functions']}；维度：{config['dimensions']}；seed：{config['seeds']}。",
            "- 函数选择：F1 代表移位旋转单峰结构，F5 代表非线性多峰结构，F9 代表 Levy 型多峰结构，F20/F24 代表组合结构；这五个函数在当前 OPFUNU 适配器中均支持 10/20/30/40/50D。",
            f"- Decision：`{training_summary_path}` 中的线形成熟度组，模型为预先固定的 RandomForestRegressor；严格执行 score > 0 与 score > {0.2997557291}。",
            f"- Selector：`{selector_model_path}`，固定类型为 `{getattr(run_frame, 'selector_type', 'dimension_aware_hybrid_selector')}`；Selector 只用 28 个无成熟度行为、14 个 descriptor 与 remaining budget ratio。",
            "- LSHADE 仅作为额外优化器级基线，不进入固定 Selector 的四动作 portfolio。",
            "- Random ELA 使用 BBOB-train OOF 的调用率与触发位置分布；每个 CEC run 使用一个独立预先生成的随机触发目标，触发后使用同一个固定 Selector。",
            "",
            "## 指标口径",
            "",
            "- `ERT_FE`：命中 run 的目标首次命中 FE 之和，加上未命中 run 的完整 FE 贡献，再除以命中 run 数；它是 FE 口径，不是 wall-clock。",
            "- `ERT_time_seconds`：命中 run 使用三次真实 replay 的首次命中秒数中位数，未命中 run 使用三次完整路径秒数中位数作贡献，再除以命中 run 数；同时报告命中 run 的中位首次命中时间。",
            "- `median_full_run_wall_clock_seconds`：三次 cyclic complete-path replay 的真实完整路径秒数中位数；不使用 component runtime 拼接。",
            "- `mean_log10_gap` 与 `target_hit_observed_rate` 用于优化能力；每条 trajectory 最多一次 query，并保留 query、handoff、失败与三次 replay 状态。",
            "",
            "## Overall",
            "",
            table,
            "",
            "## 数据与执行检查",
            "",
            f"- 生成策略行：{len(run_frame)}；每个函数×维度×seed×策略一行。",
            f"- 真实计时行：{len(run_frame) * TIMING_REPETITIONS}；计时来源为 `measured_complete_policy_path`。",
            f"- Random ELA 调用率：{calibration.run_call_rate:.6g}；触发分布来源：`{calibration.source_split}`。",
            "- CEC2017 行未参与 Decision/Selector 拟合、预处理或阈值拟合。",
            "",
            "## 输出",
            "",
            "- `online_comparison_run_metrics.parquet`：每条正式 run 的科学端点与真实时间汇总。",
            "- `online_comparison_timing_replays.parquet`：三次 cyclic replay 的逐次真实秒数。",
            "- `online_comparison_summary.parquet`：overall、function、dimension 和 function×dimension 汇总。",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="CEC2017 representative online comparison")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--training-summary", type=Path, default=DEFAULT_TRAINING_SUMMARY)
    parser.add_argument("--selector-model", type=Path, default=DEFAULT_SELECTOR_MODEL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    parser.add_argument("--only-seed", type=int, action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir or Path(str(config["output"]))
    run_experiment(
        config_path=args.config,
        training_summary_path=args.training_summary,
        selector_model_path=args.selector_model,
        output_dir=output_dir,
        only_functions=args.only_function,
        only_dimensions=args.only_dimension,
        only_seeds=args.only_seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
