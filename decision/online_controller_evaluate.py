from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from behavior.features import BEHAVIOR_FEATURE_COLUMNS, extract_behavior_rows
from benchmarks import make_problem
from decision.model_protocol import (
    FROZEN_THRESHOLD_MODE,
    SELECTED_MODEL_ALIAS,
    decision_scores,
    resolve_model_name,
)
from experiments.phase1_batch_common import family_name
from experiments.phase1_batch_common import as_int_list, fe_total_for_dimension, load_config, selected_dimensions, selected_functions
from landscape_queries.batch_features import FEATURE_METADATA_COLUMNS
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS, get_query_spec
from optimizers import (
    OptimizerSettings,
    QUERY_TRANSFER_EVENT,
    advance_optimizer_state,
    initialize_optimizer_state,
    initialize_transferred_optimizer_state,
)
from selection_reference.model import StatewiseSelectorModel, load_selector_model, make_selector_features
from trajectory.records import TrajectoryRecord
from trajectory.window_statistics import NativeUpdateWindowRecorder


DEFAULT_CONFIG_PATH = Path("configs/phase1_cec2017_test.yaml")
DEFAULT_TRAIN_CONFIG_PATH = Path("configs/phase1_bbob_train.yaml")
DEFAULT_MODEL_NAME = SELECTED_MODEL_ALIAS
DEFAULT_THRESHOLD_MODE = FROZEN_THRESHOLD_MODE
DEFAULT_RANDOM_QUERY_PROBABILITY = 0.5
DEFAULT_RANDOM_REPETITIONS = 30
DEFAULT_RANDOM_SEED = 1701
DEFAULT_SAMPLING_PROTOCOL = "training_matched"
DENSE_DECISION_CHECK_RATIOS = (
    0.20,
    0.225,
    0.25,
    0.275,
    0.28,
    0.30,
    0.325,
    0.35,
    0.375,
    0.40,
    0.425,
    0.45,
    0.475,
    0.50,
    0.525,
    0.55,
    0.575,
    0.60,
)
SAMPLING_PROTOCOLS = (DEFAULT_SAMPLING_PROTOCOL, "dense_decision_check")
EPS = 1e-12


@dataclass(frozen=True)
class DecisionControllerModel:
    model: Any
    model_name: str
    model_family: str
    threshold_mode: str
    threshold: float
    feature_columns: list[str]
    training_summary_path: Path
    model_path: Path
    query_id: str
    query_protocol: str


@dataclass(frozen=True)
class OnlineSelector:
    model: StatewiseSelectorModel

    @property
    def sbs_algorithm(self) -> str:
        return self.model.default_algorithm

    def select(
        self,
        query_features: dict[str, Any],
        behavior_features: dict[str, Any],
        remaining_ratio: float,
    ) -> tuple[str, float, str, float]:
        features = make_selector_features(
            behavior_features=behavior_features,
            query_features=query_features,
            query_feature_columns=self.model.query_feature_columns,
            remaining_budget_ratio=remaining_ratio,
        )
        selected, _, runtime_selection = self.model.select_one(features)
        return selected, float(remaining_ratio), "random_forest_action_loss_regression", runtime_selection


def evaluate_online_controller(
    *,
    query_id: str,
    query_feature_path: Path,
    config_path: Path,
    train_config_path: Path,
    selector_model_path: Path,
    training_summary_path: Path,
    output_dir: Path,
    model_name: str,
    threshold_mode: str,
    sampling_protocol: str,
    random_query_probability: float,
    random_repetitions: int,
    random_seed: int,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    only_seeds: list[int] | None,
    max_runs: int | None,
    sharded: bool,
    summarize_only: bool,
    workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if workers > 1 and not sharded and not summarize_only:
        raise ValueError("--workers > 1 is only supported with --sharded")
    config = load_config(config_path)
    if str(config["suite"]).lower() not in {"cec2017", "cec2022"}:
        raise ValueError("online controller evaluation currently expects an external CEC suite")
    checkpoint_ratios = _checkpoint_ratios(config, sampling_protocol)
    decision_check_frequency = _decision_check_frequency(sampling_protocol)
    output_dir = _sampling_output_dir(output_dir, sampling_protocol)
    functions = selected_functions(config, only_functions)
    dimensions = selected_dimensions(config, only_dimensions)
    seeds = _selected_seeds(config, only_seeds)
    controller = _load_controller(training_summary_path, model_name, threshold_mode)
    model_name = controller.model_name
    query_spec = get_query_spec(query_id)
    if controller.query_id != query_id or controller.query_protocol != query_spec.protocol:
        raise ValueError("Decision controller query protocol does not match the requested online evaluation")
    selector = None if summarize_only else _fit_online_selector(selector_model_path)
    query_feature_rows = {} if summarize_only else _read_external_query_features(query_feature_path, query_id)
    if selector is not None and selector.model.query_id != query_id:
        raise ValueError("selector model query_id does not match the requested online evaluation")
    checkpoint_plan = {
        dimension: _checkpoint_plan(config, dimension, checkpoint_ratios)
        for dimension in dimensions
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    if summarize_only:
        return _summarize_shards(
            config=config,
            config_path=config_path,
            train_config_path=train_config_path,
            training_summary_path=training_summary_path,
            output_dir=output_dir,
            model_name=model_name,
            threshold_mode=threshold_mode,
            sampling_protocol=sampling_protocol,
            checkpoint_ratios=checkpoint_ratios,
            decision_check_frequency=decision_check_frequency,
            controller=controller,
            random_query_probability=random_query_probability,
            random_repetitions=random_repetitions,
            random_seed=random_seed,
            only_functions=only_functions,
            only_dimensions=only_dimensions,
            only_seeds=only_seeds,
        )

    if sharded:
        return _evaluate_online_controller_sharded(
            config=config,
            config_path=config_path,
            train_config_path=train_config_path,
            training_summary_path=training_summary_path,
            output_dir=output_dir,
            model_name=model_name,
            threshold_mode=threshold_mode,
            sampling_protocol=sampling_protocol,
            checkpoint_ratios=checkpoint_ratios,
            decision_check_frequency=decision_check_frequency,
            controller=controller,
            selector=selector,
            query_feature_rows=query_feature_rows,
            checkpoint_plan=checkpoint_plan,
            random_query_probability=random_query_probability,
            random_repetitions=random_repetitions,
            random_seed=random_seed,
            functions=functions,
            dimensions=dimensions,
            seeds=seeds,
            only_functions=only_functions,
            only_dimensions=only_dimensions,
            only_seeds=only_seeds,
            max_runs=max_runs,
            workers=workers,
            overwrite=overwrite,
        )

    _check_output_paths(output_dir, overwrite)
    rows = []
    run_counter = 0
    for function in functions:
        for dimension in dimensions:
            fe_total = fe_total_for_dimension(config, dimension)
            for seed in seeds:
                if max_runs is not None and run_counter >= max_runs:
                    break
                rows.extend(
                    _evaluate_one_run(
                        config=config,
                        function=function,
                        dimension=dimension,
                        seed=seed,
                        fe_total=fe_total,
                        checkpoint_plan=checkpoint_plan[dimension],
                        controller=controller,
                        selector=selector,
                        query_feature_row=_query_feature_row(query_feature_rows, function=function, dimension=dimension),
                        sampling_protocol=sampling_protocol,
                        decision_check_frequency=decision_check_frequency,
                        random_query_probability=random_query_probability,
                        random_repetitions=random_repetitions,
                        random_seed=random_seed,
                    )
                )
                run_counter += 1
            if max_runs is not None and run_counter >= max_runs:
                break
        if max_runs is not None and run_counter >= max_runs:
            break

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("online controller evaluation produced no rows")
    return _write_online_summary(
        result=result,
        config_path=config_path,
        train_config_path=train_config_path,
        training_summary_path=training_summary_path,
        output_dir=output_dir,
        model_name=model_name,
        threshold_mode=threshold_mode,
        sampling_protocol=sampling_protocol,
        checkpoint_ratios=checkpoint_ratios,
        decision_check_frequency=decision_check_frequency,
        controller=controller,
        default_algorithm=selector.sbs_algorithm,
        random_query_probability=random_query_probability,
        random_repetitions=random_repetitions,
        random_seed=random_seed,
        run_mode="single_output",
        shards={},
    )


def _evaluate_online_controller_sharded(
    *,
    config: dict,
    config_path: Path,
    train_config_path: Path,
    training_summary_path: Path,
    output_dir: Path,
    model_name: str,
    threshold_mode: str,
    sampling_protocol: str,
    checkpoint_ratios: tuple[float, ...],
    decision_check_frequency: str,
    controller: DecisionControllerModel,
    selector: OnlineSelector | None,
    query_feature_rows: dict[tuple[int, int], dict[str, Any]],
    checkpoint_plan: dict[int, list[tuple[float, int]]],
    random_query_probability: float,
    random_repetitions: int,
    random_seed: int,
    functions: list[int],
    dimensions: list[int],
    seeds: list[int],
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    only_seeds: list[int] | None,
    max_runs: int | None,
    workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    if selector is None:
        raise ValueError("sharded online evaluation requires an online selector")
    jobs = []
    skipped_existing_shards = 0
    assigned_base_runs = 0
    for function in functions:
        for dimension in dimensions:
            if max_runs is not None and assigned_base_runs >= max_runs:
                break
            shard_dir = _shard_output_dir(output_dir, str(config["suite"]).lower(), function, dimension)
            shard_path = shard_dir / "online_policy_runs.parquet"
            if shard_path.exists() and not overwrite:
                print(f"skip existing online evaluation shard {shard_path}")
                skipped_existing_shards += 1
                continue

            shard_seeds = list(seeds)
            if max_runs is not None:
                remaining = max_runs - assigned_base_runs
                shard_seeds = shard_seeds[:remaining]
            if not shard_seeds:
                continue
            jobs.append(
                {
                    "config": config,
                    "function": int(function),
                    "dimension": int(dimension),
                    "seeds": [int(seed) for seed in shard_seeds],
                    "checkpoint_plan": checkpoint_plan[dimension],
                    "controller": controller,
                    "selector": selector,
                    "query_feature_row": _query_feature_row(query_feature_rows, function=function, dimension=dimension),
                    "sampling_protocol": sampling_protocol,
                    "decision_check_frequency": decision_check_frequency,
                    "random_query_probability": float(random_query_probability),
                    "random_repetitions": int(random_repetitions),
                    "random_seed": int(random_seed),
                    "output_dir": output_dir,
                    "overwrite": bool(overwrite),
                }
            )
            assigned_base_runs += len(shard_seeds)
        if max_runs is not None and assigned_base_runs >= max_runs:
            break

    shard_results = _run_shard_jobs(jobs, workers)
    written_shards = sum(1 for result in shard_results if result["status"] == "written")
    worker_skipped_existing_shards = sum(1 for result in shard_results if result["status"] == "skipped_existing")
    skipped_existing_shards += worker_skipped_existing_shards
    executed_base_runs = sum(int(result["base_runs_executed"]) for result in shard_results)

    print(
        "finished sharded online evaluation: "
        f"{written_shards} written shards, "
        f"{skipped_existing_shards} existing shards skipped, "
        f"{executed_base_runs} base runs executed, "
        f"{workers} worker(s)"
    )
    return _summarize_shards(
        config=config,
        config_path=config_path,
        train_config_path=train_config_path,
        training_summary_path=training_summary_path,
        output_dir=output_dir,
        model_name=model_name,
        threshold_mode=threshold_mode,
        sampling_protocol=sampling_protocol,
        checkpoint_ratios=checkpoint_ratios,
        decision_check_frequency=decision_check_frequency,
        controller=controller,
        random_query_probability=random_query_probability,
        random_repetitions=random_repetitions,
        random_seed=random_seed,
        only_functions=only_functions,
        only_dimensions=only_dimensions,
        only_seeds=only_seeds,
        shard_run_summary={
            "written_shards": int(written_shards),
            "skipped_existing_shards": int(skipped_existing_shards),
            "executed_base_runs": int(executed_base_runs),
            "submitted_shards": int(len(jobs)),
            "workers": int(workers),
        },
    )


def _run_shard_jobs(jobs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    if not jobs:
        return []
    if workers == 1:
        results = []
        for job in jobs:
            result = _evaluate_online_controller_shard(job)
            _print_shard_result(result)
            results.append(result)
        return results

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_evaluate_online_controller_shard, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            _print_shard_result(result)
            results.append(result)
    return results


def _evaluate_online_controller_shard(job: dict[str, Any]) -> dict[str, Any]:
    config = job["config"]
    function = int(job["function"])
    dimension = int(job["dimension"])
    seeds = [int(seed) for seed in job["seeds"]]
    output_dir = Path(job["output_dir"])
    shard_dir = _shard_output_dir(output_dir, str(config["suite"]).lower(), function, dimension)
    shard_path = shard_dir / "online_policy_runs.parquet"
    if shard_path.exists() and not bool(job["overwrite"]):
        return {
            "status": "skipped_existing",
            "path": str(shard_path),
            "rows": 0,
            "base_runs_executed": 0,
        }

    shard_rows = []
    fe_total = fe_total_for_dimension(config, dimension)
    for seed in seeds:
        shard_rows.extend(
            _evaluate_one_run(
                config=config,
                function=function,
                dimension=dimension,
                seed=seed,
                fe_total=fe_total,
                checkpoint_plan=job["checkpoint_plan"],
                controller=job["controller"],
                selector=job["selector"],
                query_feature_row=job["query_feature_row"],
                sampling_protocol=str(job["sampling_protocol"]),
                decision_check_frequency=str(job["decision_check_frequency"]),
                random_query_probability=float(job["random_query_probability"]),
                random_repetitions=int(job["random_repetitions"]),
                random_seed=int(job["random_seed"]),
            )
        )
    if not shard_rows:
        return {
            "status": "empty",
            "path": str(shard_path),
            "rows": 0,
            "base_runs_executed": 0,
        }
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_frame = pd.DataFrame(shard_rows)
    _write_frame(shard_frame, shard_dir / "online_policy_runs")
    return {
        "status": "written",
        "path": str(shard_path),
        "rows": int(len(shard_frame)),
        "base_runs_executed": int(len(seeds)),
    }


def _print_shard_result(result: dict[str, Any]) -> None:
    if result["status"] == "written":
        print(f"wrote {result['rows']} online policy rows to {result['path']}")
    elif result["status"] == "skipped_existing":
        print(f"skip existing online evaluation shard {result['path']}")
    elif result["status"] == "empty":
        print(f"skip empty online evaluation shard {result['path']}")
    else:
        print(f"finished online evaluation shard {result['path']} with status {result['status']}")


def _summarize_shards(
    *,
    config: dict,
    config_path: Path,
    train_config_path: Path,
    training_summary_path: Path,
    output_dir: Path,
    model_name: str,
    threshold_mode: str,
    sampling_protocol: str,
    checkpoint_ratios: tuple[float, ...],
    decision_check_frequency: str,
    controller: DecisionControllerModel,
    random_query_probability: float,
    random_repetitions: int,
    random_seed: int,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    only_seeds: list[int] | None,
    shard_run_summary: dict[str, int] | None = None,
) -> dict[str, Any]:
    shard_paths = _existing_shard_paths(config, output_dir, only_functions, only_dimensions)
    if not shard_paths:
        raise ValueError(f"no online evaluation shard outputs found under {output_dir}")
    frames = [pq.read_table(path).to_pandas() for path in shard_paths]
    result = pd.concat(frames, ignore_index=True)
    if only_seeds is not None:
        requested = set(int(seed) for seed in only_seeds)
        result = result[result["seed"].astype(int).isin(requested)].reset_index(drop=True)
    if result.empty:
        raise ValueError("online evaluation shard rows are empty after filtering")
    default_algorithms = sorted(result["default_algorithm"].astype(str).unique().tolist())
    default_algorithm = default_algorithms[0] if len(default_algorithms) == 1 else ",".join(default_algorithms)
    return _write_online_summary(
        result=result,
        config_path=config_path,
        train_config_path=train_config_path,
        training_summary_path=training_summary_path,
        output_dir=output_dir,
        model_name=model_name,
        threshold_mode=threshold_mode,
        sampling_protocol=sampling_protocol,
        checkpoint_ratios=checkpoint_ratios,
        decision_check_frequency=decision_check_frequency,
        controller=controller,
        default_algorithm=default_algorithm,
        random_query_probability=random_query_probability,
        random_repetitions=random_repetitions,
        random_seed=random_seed,
        run_mode="sharded",
        shards={
            "discovered_shards": int(len(shard_paths)),
            "paths": [str(path) for path in shard_paths],
            **(shard_run_summary or {}),
        },
    )


def _write_online_summary(
    *,
    result: pd.DataFrame,
    config_path: Path,
    train_config_path: Path,
    training_summary_path: Path,
    output_dir: Path,
    model_name: str,
    threshold_mode: str,
    sampling_protocol: str,
    checkpoint_ratios: tuple[float, ...],
    decision_check_frequency: str,
    controller: DecisionControllerModel,
    default_algorithm: str,
    random_query_probability: float,
    random_repetitions: int,
    random_seed: int,
    run_mode: str,
    shards: dict[str, Any],
) -> dict[str, Any]:
    policy_summary = _policy_summary(result)
    relative_summary = _relative_summary(result)
    random_repetition_summary = _random_repetition_summary(result)

    _write_frame(result, output_dir / "online_policy_runs")
    _write_frame(policy_summary, output_dir / "online_policy_summary")
    _write_frame(relative_summary, output_dir / "online_relative_summary")
    _write_frame(random_repetition_summary, output_dir / "online_random_repetition_summary")

    base_runs = int(result[["problem_id", "dimension", "seed"]].drop_duplicates().shape[0])
    summary = {
        "experiment": "cec_online_controller_evaluation",
        "query_id": str(result["query_id"].iloc[0]),
        "query_protocol": str(result["query_protocol"].iloc[0]),
        "sample_design_id": str(result["sample_design_id"].iloc[0]),
        "run_mode": run_mode,
        "config": str(config_path),
        "train_config": str(train_config_path),
        "training_summary": str(training_summary_path),
        "model_name": model_name,
        "threshold_mode": threshold_mode,
        "sampling_protocol": sampling_protocol,
        "checkpoint_ratios": [float(value) for value in checkpoint_ratios],
        "decision_check_frequency": decision_check_frequency,
        "decision_check_count": int(len(checkpoint_ratios)),
        "threshold": float(controller.threshold),
        "feature_columns": controller.feature_columns,
        "default_algorithm": default_algorithm,
        "random_query_probability": random_query_probability,
        "random_repetitions": random_repetitions,
        "random_seed": random_seed,
        "rows": int(len(result)),
        "base_runs": base_runs,
        "query_group_failure_rows": int(
            (
                result["query_called"].astype(bool)
                & result["query_feature_status"].astype(str).ne("ok")
            ).sum()
        ),
        "policies": sorted(result["policy_name"].astype(str).unique().tolist()),
        "shards": shards,
        "outputs": {
            "policy_runs": str(output_dir / "online_policy_runs.parquet"),
            "policy_summary": str(output_dir / "online_policy_summary.parquet"),
            "relative_summary": str(output_dir / "online_relative_summary.parquet"),
            "random_repetition_summary": str(output_dir / "online_random_repetition_summary.parquet"),
            "report": str(output_dir / "online_controller_evaluation_report.md"),
            "summary": str(output_dir / "online_controller_evaluation_summary.json"),
        },
        "data_leakage_check": {
            "external_rows_used_for_controller_fit": 0,
            "external_rows_used_for_threshold_fit": 0,
            "external_query_features_used_as_controller_input": False,
            "function_id_algorithm_id_or_optimizer_internal_parameters_used_as_controller_input": False,
            "controller_inputs_are_behavior_features_only": True,
        },
        "scope_notes": [
            "All policies start from the SBS/default optimizer probe state when a decision is needed.",
            "Behavior sampling is the decision-check frequency: every checkpoint is also a possible query trigger point.",
            "always_query executes the fixed query at the first checkpoint in the active sampling protocol.",
            "traditional_aas reuses the equivalent always-query plus selector run instead of repeating the continuation.",
            "External group-level query failures are retained in query_feature_status and use the selector's frozen BBOB-train median imputation; affected rows are reported separately.",
            "random_query samples one independent checkpoint trigger stream per repetition and run at each decision-check point.",
            "dense_decision_check is a sensitivity analysis of decision-check frequency, not a passive observation protocol.",
            "The controller and threshold are frozen from BBOB train artifacts.",
            "In sharded mode, existing function/dimension shard outputs are skipped unless --overwrite is passed.",
            "Parallel workers execute independent function/dimension shards; summary outputs are written after shard completion.",
        ],
    }
    summary_path = output_dir / "online_controller_evaluation_summary.json"
    report_path = output_dir / "online_controller_evaluation_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(
        _markdown_report(
            summary=summary,
            policy_summary=policy_summary,
            relative_summary=relative_summary,
            random_repetition_summary=random_repetition_summary,
        ),
        encoding="utf-8",
    )
    print(f"wrote online policy runs to {output_dir / 'online_policy_runs.parquet'}")
    print(f"wrote online controller report to {report_path}")
    return summary


def _shard_output_dir(output_dir: Path, suite: str, function: int, dimension: int) -> Path:
    return output_dir / family_name(suite, function) / f"dimension_{int(dimension)}"


def _existing_shard_paths(
    config: dict,
    output_dir: Path,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
) -> list[Path]:
    suite = str(config["suite"]).lower()
    paths = []
    for function in selected_functions(config, only_functions):
        for dimension in selected_dimensions(config, only_dimensions):
            path = _shard_output_dir(output_dir, suite, function, dimension) / "online_policy_runs.parquet"
            if path.exists():
                paths.append(path)
    return sorted(paths)


def _check_output_paths(output_dir: Path, overwrite: bool) -> None:
    outputs = (
        output_dir / "online_policy_runs.csv",
        output_dir / "online_policy_runs.parquet",
        output_dir / "online_policy_summary.csv",
        output_dir / "online_policy_summary.parquet",
        output_dir / "online_relative_summary.csv",
        output_dir / "online_relative_summary.parquet",
        output_dir / "online_random_repetition_summary.csv",
        output_dir / "online_random_repetition_summary.parquet",
        output_dir / "online_controller_evaluation_report.md",
        output_dir / "online_controller_evaluation_summary.json",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"online evaluation outputs already exist; pass --overwrite: {existing[0]}")


def _selected_seeds(config: dict, only_seeds: list[int] | None) -> list[int]:
    seeds = as_int_list(config, "seeds")
    if only_seeds is None:
        return seeds
    requested = set(int(seed) for seed in only_seeds)
    missing = sorted(requested.difference(seeds))
    if missing:
        raise ValueError(f"requested seeds are not in config: {missing}")
    return [seed for seed in seeds if seed in requested]


def _load_controller(training_summary_path: Path, model_name: str, threshold_mode: str) -> DecisionControllerModel:
    summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    feature_columns = [str(column) for column in summary.get("feature_columns", [])]
    if not feature_columns or not set(feature_columns).issubset(BEHAVIOR_FEATURE_COLUMNS):
        raise ValueError("controller feature columns must be a non-empty subset of behavior features")
    model_name = resolve_model_name(summary, model_name)
    model_path = _model_path(summary, model_name)
    threshold = _threshold(summary, model_name, threshold_mode)
    model_family = _model_family(summary, model_name)
    return DecisionControllerModel(
        model=joblib.load(model_path),
        model_name=model_name,
        model_family=model_family,
        threshold_mode=threshold_mode,
        threshold=threshold,
        feature_columns=feature_columns,
        training_summary_path=training_summary_path,
        model_path=model_path,
        query_id=str(summary.get("query_id", "")),
        query_protocol=str(summary.get("query_protocol", "")),
    )


def _read_external_query_features(path: Path, query_id: str) -> dict[tuple[int, int], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing external query feature table: {path}")
    spec = get_query_spec(query_id)
    frame = pq.read_table(path).to_pandas()
    expected_columns = set(FEATURE_METADATA_COLUMNS) | set(spec.feature_columns)
    observed_columns = set(frame.columns)
    if observed_columns != expected_columns:
        raise ValueError(
            "external query feature table does not exactly match the frozen whitelist; "
            f"missing={sorted(expected_columns - observed_columns)}, "
            f"extra={sorted(observed_columns - expected_columns)}"
        )
    required = {
        "problem_id",
        "function",
        "dimension",
        "query_id",
        "query_protocol",
        "sample_design_id",
        "runtime_query",
        "feature_status",
        "feature_failure",
        "feature_group_status",
        "additional_function_evaluations",
        "query_feature_columns",
        *spec.feature_columns,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"external query feature table is missing columns: {sorted(missing)}")
    if set(frame["query_id"].astype(str)) != {query_id}:
        raise ValueError("external query feature table contains the wrong query_id")
    if set(frame["query_protocol"].astype(str)) != {spec.protocol}:
        raise ValueError("external query feature table contains the wrong query_protocol")
    if set(frame["sample_design_id"].astype(str)) != {spec.sample_design_id}:
        raise ValueError("external query feature table contains the wrong sample design")
    expected_feature_columns = json.dumps(list(spec.feature_columns), ensure_ascii=False)
    if set(frame["query_feature_columns"].astype(str)) != {expected_feature_columns}:
        raise ValueError("external query feature table contains a non-frozen feature-column list")
    if (frame["additional_function_evaluations"].astype(int) != 0).any():
        raise ValueError("external query feature extraction reports additional objective evaluations")
    for row in frame.to_dict(orient="records"):
        group_status = json.loads(str(row["feature_group_status"]))
        if set(group_status) != set(spec.feature_groups):
            raise ValueError("external query feature group status does not cover the frozen groups")
        has_group_failure = any(str(status.get("status")) != "ok" for status in group_status.values())
        expected_status = "failed" if has_group_failure else "ok"
        if str(row["feature_status"]) != expected_status:
            raise ValueError("external query feature_status is inconsistent with group-level status")
    key = ["function", "dimension"]
    if frame.duplicated(key).any():
        raise ValueError("external query feature table contains duplicate function/dimension rows")
    runtimes = frame["runtime_query"].astype(float).to_numpy()
    if not np.isfinite(runtimes).all() or (runtimes < 0.0).any():
        raise ValueError("external query runtime must be finite and non-negative")
    expected_runtime = (
        frame["runtime_sampling_evaluation"].astype(float)
        + frame["runtime_feature_computation"].astype(float)
    ).to_numpy()
    if not np.allclose(runtimes, expected_runtime, rtol=0.0, atol=1e-12):
        raise ValueError("external runtime_query is not sampling evaluation plus feature computation")
    return {
        (int(row["function"]), int(row["dimension"])): row
        for row in frame.to_dict(orient="records")
    }


def _query_feature_row(
    rows: dict[tuple[int, int], dict[str, Any]],
    *,
    function: int,
    dimension: int,
) -> dict[str, Any]:
    key = (int(function), int(dimension))
    if key not in rows:
        raise ValueError(f"missing saved query features for function={function}, dimension={dimension}")
    return rows[key]


def _model_path(summary: dict[str, Any], model_name: str) -> Path:
    for artifact in summary.get("model_artifacts", []):
        if str(artifact.get("model_name")) == model_name:
            path = Path(str(artifact.get("model_path")))
            if not path.exists():
                raise FileNotFoundError(f"missing controller model artifact: {path}")
            return path
    raise ValueError(f"controller model artifact not found: {model_name}")


def _threshold(summary: dict[str, Any], model_name: str, threshold_mode: str) -> float:
    threshold_path = Path(str(summary.get("outputs", {}).get("decision_thresholds", "")))
    if not threshold_path.exists():
        raise FileNotFoundError(f"missing decision threshold table: {threshold_path}")
    thresholds = pq.read_table(threshold_path).to_pandas()
    row = thresholds[
        (thresholds["model_name"].astype(str) == model_name)
        & (thresholds["threshold_mode"].astype(str) == threshold_mode)
    ]
    if len(row) != 1:
        raise ValueError(f"expected one threshold row for {model_name}/{threshold_mode}, found {len(row)}")
    if int(row["validation_rows_used_for_threshold_fit"].iloc[0]) != 0:
        raise ValueError("controller threshold must be fit without held-out rows")
    return float(row["threshold"].iloc[0])


def _model_family(summary: dict[str, Any], model_name: str) -> str:
    fit_path = Path(str(summary.get("outputs", {}).get("model_fit_summary", "")))
    if not fit_path.exists():
        return ""
    rows = pq.read_table(fit_path).to_pandas()
    row = rows[rows["model_name"].astype(str) == model_name]
    return str(row["model_family"].iloc[0]) if len(row) else ""


def _fit_online_selector(selector_model_path: Path) -> OnlineSelector:
    return OnlineSelector(model=load_selector_model(selector_model_path))


def _checkpoint_ratios(config: dict, sampling_protocol: str) -> tuple[float, ...]:
    if sampling_protocol == DEFAULT_SAMPLING_PROTOCOL:
        return tuple(float(value) for value in config["checkpoint_ratios"])
    if sampling_protocol == "dense_decision_check":
        return DENSE_DECISION_CHECK_RATIOS
    raise ValueError(f"unsupported sampling protocol: {sampling_protocol}")


def _decision_check_frequency(sampling_protocol: str) -> str:
    if sampling_protocol == DEFAULT_SAMPLING_PROTOCOL:
        return "training_matched_checkpoint_ratios"
    if sampling_protocol == "dense_decision_check":
        return "dense_0.025_fe_ratio_grid_between_0.20_and_0.60_with_0.28_transition_point"
    raise ValueError(f"unsupported sampling protocol: {sampling_protocol}")


def _sampling_output_dir(output_dir: Path, sampling_protocol: str) -> Path:
    if sampling_protocol == DEFAULT_SAMPLING_PROTOCOL:
        return output_dir
    if sampling_protocol == "dense_decision_check" and not output_dir.name.endswith("_dense_decision_check"):
        return output_dir.with_name(f"{output_dir.name}_dense_decision_check")
    return output_dir


def _checkpoint_plan(config: dict, dimension: int, checkpoint_ratios: tuple[float, ...]) -> list[tuple[float, int]]:
    fe_total = fe_total_for_dimension(config, dimension)
    population_size = int(config["population_size"])
    plan = []
    for ratio in checkpoint_ratios:
        fe = int(min(fe_total, ceil(float(ratio) * fe_total / population_size - EPS) * population_size))
        plan.append((float(ratio), fe))
    return plan


def _evaluate_one_run(
    *,
    config: dict,
    function: int,
    dimension: int,
    seed: int,
    fe_total: int,
    checkpoint_plan: list[tuple[float, int]],
    controller: DecisionControllerModel,
    selector: OnlineSelector,
    query_feature_row: dict[str, Any],
    sampling_protocol: str,
    decision_check_frequency: str,
    random_query_probability: float,
    random_repetitions: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    suite = str(config["suite"]).lower()
    problem = make_problem({"suite": suite, "function": function, "instance": 1, "dimension": dimension})
    try:
        no_query = _run_threshold_policy(
            problem=problem,
            config=config,
            function=function,
            seed=seed,
            fe_total=fe_total,
            checkpoint_plan=checkpoint_plan,
            selector=selector,
            query_feature_row=query_feature_row,
            controller=None,
            policy_name="sbs_no_query",
            trigger_mode="never",
            sampling_protocol=sampling_protocol,
            decision_check_frequency=decision_check_frequency,
            repetition=None,
        )
        always_query = _run_threshold_policy(
            problem=problem,
            config=config,
            function=function,
            seed=seed,
            fe_total=fe_total,
            checkpoint_plan=checkpoint_plan,
            selector=selector,
            query_feature_row=query_feature_row,
            controller=None,
            policy_name="always_query",
            trigger_mode="first_checkpoint",
            sampling_protocol=sampling_protocol,
            decision_check_frequency=decision_check_frequency,
            repetition=None,
        )
        traditional_aas = dict(always_query)
        traditional_aas["policy_name"] = "traditional_aas"
        controller_row = _run_threshold_policy(
            problem=problem,
            config=config,
            function=function,
            seed=seed,
            fe_total=fe_total,
            checkpoint_plan=checkpoint_plan,
            selector=selector,
            query_feature_row=query_feature_row,
            controller=controller,
            policy_name="current_controller",
            trigger_mode="controller",
            sampling_protocol=sampling_protocol,
            decision_check_frequency=decision_check_frequency,
            repetition=None,
        )
        random_rows = [
            _run_threshold_policy(
                problem=problem,
                config=config,
                function=function,
                seed=seed,
                fe_total=fe_total,
                checkpoint_plan=checkpoint_plan,
                selector=selector,
                query_feature_row=query_feature_row,
                controller=None,
                policy_name="random_query_p50",
                trigger_mode="random",
                sampling_protocol=sampling_protocol,
                decision_check_frequency=decision_check_frequency,
                repetition=repetition,
                random_query_probability=random_query_probability,
                random_seed=random_seed,
            )
            for repetition in range(random_repetitions)
        ]
        rows = [no_query, always_query, traditional_aas, controller_row, *random_rows]
        for row in rows:
            row.update(
                {
                    "split": _split_name(config),
                    "suite": suite,
                    "problem_id": problem.problem_id,
                    "family": problem.family,
                    "function": int(function),
                    "dimension": int(dimension),
                    "seed": int(seed),
                    "sampling_protocol": sampling_protocol,
                    "decision_check_frequency": decision_check_frequency,
                    "decision_check_count": int(len(checkpoint_plan)),
                }
            )
        return rows
    finally:
        problem.close()


def _run_threshold_policy(
    *,
    problem,
    config: dict,
    function: int,
    seed: int,
    fe_total: int,
    checkpoint_plan: list[tuple[float, int]],
    selector: OnlineSelector,
    query_feature_row: dict[str, Any],
    controller: DecisionControllerModel | None,
    policy_name: str,
    trigger_mode: str,
    sampling_protocol: str,
    decision_check_frequency: str,
    repetition: int | None,
    random_query_probability: float = 0.5,
    random_seed: int = 1701,
) -> dict[str, Any]:
    default_algorithm = selector.sbs_algorithm
    population_size = int(config["population_size"])
    query_spec = get_query_spec(selector.model.query_id)
    fe_query = query_spec.sample_design.sample_size(problem.dimension)
    if fe_query != int(round(query_spec.sample_design.fe_ratio * fe_total)):
        raise ValueError("online query FE does not match the frozen sample design")
    trajectory_rows: list[dict[str, Any]] = []
    settings = OptimizerSettings(population_size=population_size, checkpoint_ratios=(1.0,))
    started = perf_counter()
    current_state = initialize_optimizer_state(
        algorithm=default_algorithm,
        problem=problem,
        seed=seed,
        settings=settings,
    )
    prefix_algorithm = str(current_state.algorithm)
    window_recorder = NativeUpdateWindowRecorder()
    window_recorder.observe(
        fe=int(current_state.evaluations),
        native_updates=int(current_state.generation),
        population=current_state.population,
        fitness=current_state.fitness,
        best_fitness=current_state.best_fitness,
    )
    runtime_probe = perf_counter() - started
    current_fe = int(current_state.evaluations)
    triggered = False
    trigger_ratio = None
    trigger_score = None
    selected_algorithm = default_algorithm
    selector_remaining_budget_ratio = None
    selector_status = "not_used"
    runtime_query = 0.0
    runtime_selection = 0.0

    for ratio, checkpoint_fe in checkpoint_plan:
        delta = checkpoint_fe - current_fe
        if delta <= 0:
            continue
        continuation = advance_optimizer_state(
            state=current_state,
            problem=problem,
            fe_budget=delta,
            on_native_update=lambda updated: window_recorder.observe(
                fe=int(updated.evaluations),
                native_updates=int(updated.generation),
                population=updated.population,
                fitness=updated.fitness,
                best_fitness=updated.best_fitness,
            ),
        )
        runtime_probe += continuation.runtime_seconds
        current_fe = checkpoint_fe
        window_statistics, native_update_history = window_recorder.build(fe_total=fe_total)
        trajectory_record = TrajectoryRecord.from_arrays(
            problem_id=problem.problem_id,
            family=problem.family,
            dimension=problem.dimension,
            algorithm=default_algorithm,
            seed=seed,
            fe=current_fe,
            fe_total=fe_total,
            native_updates=int(current_state.generation),
            window_statistics=window_statistics,
            native_update_history=native_update_history,
            population=current_state.population,
            fitness=current_state.fitness,
            best_fitness=current_state.best_fitness,
            fe_ratio=ratio,
        )
        trajectory_rows.append(trajectory_record.__dict__)
        behavior_row = extract_behavior_rows([row.copy() for row in trajectory_rows])[-1]
        should_trigger, trigger_score = _should_trigger(
            behavior_row=behavior_row,
            controller=controller,
            trigger_mode=trigger_mode,
            random_query_probability=random_query_probability,
            random_seed=random_seed,
            seed=seed,
            function=function,
            dimension=problem.dimension,
            repetition=repetition,
        )
        if should_trigger:
            triggered = True
            trigger_ratio = ratio
            if str(query_feature_row["problem_id"]) != problem.problem_id:
                raise ValueError("saved query feature row does not match the online problem")
            runtime_query = float(query_feature_row["runtime_query"])
            query_features = {
                column: query_feature_row.get(column)
                for column in selector.model.query_feature_columns
            }
            remaining = max(fe_total - current_fe - fe_query, 0)
            remaining_ratio = round(remaining / fe_total, 6)
            selected_algorithm, selector_remaining_budget_ratio, selector_status, runtime_selection = selector.select(
                query_features,
                behavior_row,
                remaining_ratio,
            )
            break

    if triggered:
        remaining_budget = max(fe_total - current_fe - fe_query, 0)
        after_started = perf_counter()
        if selected_algorithm == prefix_algorithm:
            after_state = current_state
            transition_mode = "native_optimizer_state"
        else:
            after_state = initialize_transferred_optimizer_state(
                algorithm=selected_algorithm,
                source_state=current_state,
                problem=problem,
                seed=seed,
                function=function,
                instance=1,
                event=QUERY_TRANSFER_EVENT,
            )
            transition_mode = "population_transfer_initialization"
        after = advance_optimizer_state(state=after_state, problem=problem, fe_budget=remaining_budget)
        final_performance = float(after.best_fitness)
        runtime_after = float(perf_counter() - after_started)
        fe_after = int(after.evaluations)
        if fe_after != remaining_budget:
            raise ValueError("query continuation did not consume its assigned FE budget")
        fe_used = int(current_fe + fe_query + fe_after)
    else:
        remaining_budget = max(fe_total - current_fe, 0)
        after = advance_optimizer_state(state=current_state, problem=problem, fe_budget=remaining_budget)
        final_performance = float(after.best_fitness)
        runtime_after = float(after.runtime_seconds)
        fe_after = int(after.evaluations)
        if fe_after != remaining_budget:
            raise ValueError("Skip continuation did not consume its assigned FE budget")
        fe_used = int(current_fe + fe_after)
        transition_mode = "native_optimizer_state"

    policy_category = "controller" if policy_name == "current_controller" else "baseline"
    selected_equals_prefix = selected_algorithm == prefix_algorithm
    selected_action = "continue_current" if selected_equals_prefix else selected_algorithm
    handoff_required = not selected_equals_prefix
    if handoff_required != (transition_mode == "population_transfer_initialization"):
        raise ValueError("online handoff_required does not match optimizer_transition_mode")
    return {
        "policy_name": policy_name,
        "policy_category": policy_category,
        "sampling_protocol": sampling_protocol,
        "decision_check_frequency": decision_check_frequency,
        "random_repetition": repetition,
        "prefix_algorithm": prefix_algorithm,
        "default_algorithm": default_algorithm,
        "no_query_algorithm": default_algorithm,
        "selected_algorithm": selected_algorithm,
        "selected_action": selected_action,
        "selected_equals_default": bool(selected_algorithm == default_algorithm),
        "selected_equals_prefix": bool(selected_equals_prefix),
        "handoff_required": bool(handoff_required),
        "skip_switches_from_prefix": bool(default_algorithm != prefix_algorithm),
        "selector_status": selector_status,
        "optimizer_transition_mode": transition_mode,
        "handoff_type": transition_mode,
        "selector_target_transform": selector.model.selector_target_transform,
        "selector_remaining_budget_ratio": selector_remaining_budget_ratio,
        "query_called": bool(triggered),
        "query_id": selector.model.query_id,
        "query_protocol": selector.model.query_protocol,
        "sample_design_id": selector.model.sample_design_id,
        "query_feature_status": str(query_feature_row.get("feature_status", "unknown")) if triggered else "not_called",
        "query_feature_failure": str(query_feature_row.get("feature_failure", "")) if triggered else "",
        "trigger_FE": int(current_fe) if triggered else None,
        "trigger_FE_ratio": float(trigger_ratio) if triggered else None,
        "decision_score": float(trigger_score) if trigger_score is not None else None,
        "decision_threshold": float(controller.threshold) if controller is not None else None,
        "FE_total": int(fe_total),
        "FE_probe": int(current_fe),
        "FE_query": int(fe_query) if triggered else 0,
        "FE_after_decision_optimization": int(fe_after),
        "FE_used": int(fe_used),
        "runtime_probe": float(runtime_probe),
        "runtime_query": float(runtime_query),
        "runtime_selection": float(runtime_selection),
        "runtime_after_decision_optimization": float(runtime_after),
        "runtime_total": float(runtime_probe + runtime_query + runtime_selection + runtime_after),
        "final_performance": final_performance,
    }


def _should_trigger(
    *,
    behavior_row: dict[str, Any],
    controller: DecisionControllerModel | None,
    trigger_mode: str,
    random_query_probability: float,
    random_seed: int,
    seed: int,
    function: int,
    dimension: int,
    repetition: int | None,
) -> tuple[bool, float | None]:
    if trigger_mode == "first_checkpoint":
        return True, None
    if trigger_mode == "never":
        return False, None
    if trigger_mode == "controller":
        if controller is None:
            raise ValueError("controller trigger mode requires a fitted Decision controller model")
        frame = pd.DataFrame([{column: behavior_row[column] for column in controller.feature_columns}])
        score = float(decision_scores(controller.model, frame)[0])
        return bool(score > controller.threshold), score
    if trigger_mode == "random":
        rng = np.random.default_rng(
            np.random.SeedSequence([int(random_seed), int(seed), int(function), int(dimension), int(repetition or 0), int(behavior_row["FE"])])
        )
        return bool(rng.random() < random_query_probability), None
    raise ValueError(f"unknown trigger_mode: {trigger_mode}")


def _policy_summary(rows: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    layers = {
        "overall": [],
        "family": ["family"],
        "dimension": ["dimension"],
        "family_dimension": ["family", "dimension"],
    }
    for layer, columns in layers.items():
        for policy_name, policy_frame in rows.groupby("policy_name", sort=True):
            if columns:
                groups = policy_frame.groupby(columns, dropna=False, sort=True)
            else:
                groups = [((), policy_frame)]
            for values, frame in groups:
                if columns and not isinstance(values, tuple):
                    values = (values,)
                group = dict(zip(columns, values, strict=False)) if columns else {}
                summary_rows.append(_summary_row(frame, layer, group, str(policy_name)))
    return pd.DataFrame(summary_rows).sort_values(["layer", "group", "policy_name"]).reset_index(drop=True)


def _summary_row(frame: pd.DataFrame, layer: str, group: dict[str, Any], policy_name: str) -> dict[str, Any]:
    return {
        "layer": layer,
        "group": _group_label(group),
        "family": group.get("family"),
        "dimension": group.get("dimension"),
        "policy_name": policy_name,
        "rows": int(len(frame)),
        "mean_final_performance": float(frame["final_performance"].mean()),
        "median_final_performance": float(frame["final_performance"].median()),
        "query_call_rate": float(frame["query_called"].mean()),
        "mean_FE_used": float(frame["FE_used"].mean()),
        "mean_runtime_total": float(frame["runtime_total"].mean()),
        "mean_trigger_FE_ratio": float(frame["trigger_FE_ratio"].dropna().mean()) if frame["trigger_FE_ratio"].notna().any() else None,
    }


def _relative_summary(rows: pd.DataFrame) -> pd.DataFrame:
    baseline = rows[rows["policy_name"] == "sbs_no_query"][
        ["problem_id", "dimension", "seed", "final_performance", "runtime_total", "FE_used"]
    ].rename(
        columns={
            "final_performance": "sbs_no_query_final_performance",
            "runtime_total": "sbs_no_query_runtime_total",
            "FE_used": "sbs_no_query_FE_used",
        }
    )
    joined = rows.merge(baseline, on=["problem_id", "dimension", "seed"], how="left")
    joined["final_performance_delta_vs_sbs_no_query"] = (
        joined["final_performance"] - joined["sbs_no_query_final_performance"]
    )
    joined["runtime_delta_vs_sbs_no_query"] = joined["runtime_total"] - joined["sbs_no_query_runtime_total"]
    joined["FE_used_delta_vs_sbs_no_query"] = joined["FE_used"] - joined["sbs_no_query_FE_used"]
    return (
        joined.groupby("policy_name", as_index=False)
        .agg(
            rows=("final_performance", "size"),
            mean_final_performance_delta_vs_sbs_no_query=("final_performance_delta_vs_sbs_no_query", "mean"),
            median_final_performance_delta_vs_sbs_no_query=("final_performance_delta_vs_sbs_no_query", "median"),
            mean_runtime_delta_vs_sbs_no_query=("runtime_delta_vs_sbs_no_query", "mean"),
            mean_FE_used_delta_vs_sbs_no_query=("FE_used_delta_vs_sbs_no_query", "mean"),
            query_call_rate=("query_called", "mean"),
        )
        .sort_values("mean_final_performance_delta_vs_sbs_no_query")
        .reset_index(drop=True)
    )


def _random_repetition_summary(rows: pd.DataFrame) -> pd.DataFrame:
    random_rows = rows[rows["policy_name"] == "random_query_p50"].copy()
    if random_rows.empty:
        return pd.DataFrame()
    summary = (
        random_rows.groupby("random_repetition", as_index=False)
        .agg(
            rows=("final_performance", "size"),
            mean_final_performance=("final_performance", "mean"),
            query_call_rate=("query_called", "mean"),
            mean_FE_used=("FE_used", "mean"),
            mean_runtime_total=("runtime_total", "mean"),
        )
    )
    metric_rows = []
    for metric in ("mean_final_performance", "query_call_rate", "mean_FE_used", "mean_runtime_total"):
        values = summary[metric].astype(float)
        metric_rows.append(
            {
                "policy_name": "random_query_p50",
                "metric": metric,
                "repetitions": int(len(summary)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(metric_rows)


def _write_frame(frame: pd.DataFrame, stem: Path) -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), stem.with_suffix(".parquet"))


def _markdown_report(
    *,
    summary: dict[str, Any],
    policy_summary: pd.DataFrame,
    relative_summary: pd.DataFrame,
    random_repetition_summary: pd.DataFrame,
) -> str:
    overall = policy_summary[policy_summary["layer"] == "overall"][
        [
            "policy_name",
            "rows",
            "mean_final_performance",
            "median_final_performance",
            "query_call_rate",
            "mean_FE_used",
            "mean_runtime_total",
            "mean_trigger_FE_ratio",
        ]
    ].sort_values("mean_final_performance")
    return "\n".join(
        [
            "# CEC online controller evaluation",
            "",
            "## 摘要",
            "",
            f"- Controller: `{summary['model_name']}`，阈值口径 `{summary['threshold_mode']}`。",
            f"- Default/SBS optimizer: `{summary['default_algorithm']}`。",
            f"- Sampling protocol: `{summary['sampling_protocol']}`。",
            f"- Decision-check frequency: `{summary['decision_check_frequency']}`。",
            f"- Checkpoint ratios: `{', '.join(str(value) for value in summary['checkpoint_ratios'])}`。",
            "- 评价单位是完整 optimization run；final performance 越小越好。",
            "- Controller 只使用实时 behavior features；CEC rows 不参与训练、预处理拟合或阈值选择。",
            "- Query 后的 Selection Reference 使用冻结的 BBOB-train statewise action-loss regressor，并连续接收 remaining budget；CEC rows 不参与 selector 拟合。",
            "- 每个 checkpoint 同时是 behavior observation 点和可能触发固定 query 的 decision-check point。",
            "- `always_query` 表示在当前 sampling protocol 的第一个 checkpoint 后必定执行固定 query 的 after-probe baseline。",
            "- `dense_decision_check` 只能解释为决策检查频率敏感性，不是纯被动观测频率实验。",
            "",
            "## Overall Policies",
            "",
            _markdown_table(overall),
            "",
            "## Relative To SBS/No-query",
            "",
            _markdown_table(relative_summary),
            "",
            "## Random Analysis Repetition Summary",
            "",
            _markdown_table(random_repetition_summary),
            "",
            "## Outputs",
            "",
            f"- Policy runs: `{summary['outputs']['policy_runs']}`",
            f"- Policy summary: `{summary['outputs']['policy_summary']}`",
            f"- Relative summary: `{summary['outputs']['relative_summary']}`",
        ]
    ) + "\n"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _group_label(group: dict[str, Any]) -> str:
    if not group:
        return "all"
    return ", ".join(f"{key}={value}" for key, value in group.items())


def _split_name(config: dict) -> str:
    if "split" in config:
        return str(config["split"])
    return Path(config["output"]).stem.removesuffix("_trajectories")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run online CEC controller evaluation with frozen BBOB controller.")
    parser.add_argument("--query-id", choices=sorted(LANDSCAPE_QUERY_SPECS), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH)
    parser.add_argument("--query-features", type=Path, default=None)
    parser.add_argument("--selector-model", type=Path, default=None)
    parser.add_argument("--training-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--threshold-mode", default=DEFAULT_THRESHOLD_MODE)
    parser.add_argument("--sampling-protocol", choices=SAMPLING_PROTOCOLS, default=DEFAULT_SAMPLING_PROTOCOL)
    parser.add_argument("--random-query-probability", type=float, default=DEFAULT_RANDOM_QUERY_PROBABILITY)
    parser.add_argument("--random-repetitions", type=int, default=DEFAULT_RANDOM_REPETITIONS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    parser.add_argument("--only-seed", type=int, action="append", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--sharded", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    split = _split_name(config)
    query_features = args.query_features or Path("results/landscape_queries/features") / args.query_id / split / "features.parquet"
    selector_model = args.selector_model or Path("results/selection_reference") / args.query_id / "statewise_selector.joblib"
    training_summary = (
        args.training_summary
        or Path("results/decision")
        / args.query_id
        / "feature_group_ablation/primary_with_maturity/full_decision_model_training_summary.json"
    )
    output_dir = args.output_dir or Path("results/decision") / args.query_id / split / "online_controller_evaluation"

    evaluate_online_controller(
        query_id=args.query_id,
        query_feature_path=query_features,
        config_path=args.config,
        train_config_path=args.train_config,
        selector_model_path=selector_model,
        training_summary_path=training_summary,
        output_dir=output_dir,
        model_name=args.model_name,
        threshold_mode=args.threshold_mode,
        sampling_protocol=args.sampling_protocol,
        random_query_probability=args.random_query_probability,
        random_repetitions=args.random_repetitions,
        random_seed=args.random_seed,
        only_functions=args.only_function,
        only_dimensions=args.only_dimension,
        only_seeds=args.only_seed,
        max_runs=args.max_runs,
        sharded=args.sharded,
        summarize_only=args.summarize_only,
        workers=args.workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
