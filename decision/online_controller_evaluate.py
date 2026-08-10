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
from ela.features import ELA_FEATURE_COLUMNS, extract_ela_for_problem
from experiments.phase1_batch_common import family_name
from experiments.phase1_batch_common import as_int_list, fe_total_for_dimension, load_config, selected_dimensions, selected_functions
from optimizers import OptimizerSettings, run_optimizer
from optimizers.continuation import run_population_continuation
from selection_reference.build import (
    _best_algorithm_by_problem,
    _bucket_ratio_by_dimension,
    _checkpoint_budget_map,
    _ela_feature_path,
    _fit_selector,
    _predict_algorithms,
    _read_feature_file,
    _read_performance,
    _single_best_solver,
)
from trajectory.records import TrajectoryRecord


DEFAULT_CONFIG_PATH = Path("configs/phase1_cec2017_test.yaml")
DEFAULT_TRAIN_CONFIG_PATH = Path("configs/phase1_bbob_train.yaml")
DEFAULT_TRAINING_SUMMARY_PATH = Path(
    "results/decision/phase1_refined_sampling/feature_group_ablation/"
    "primary_with_maturity/full_decision_model_training_summary.json"
)
DEFAULT_OUTPUT_DIR = Path("results/decision/cec2017_test/online_controller_evaluation")
DEFAULT_MODEL_NAME = "ridge_regression"
DEFAULT_THRESHOLD_MODE = "train_utility"
DEFAULT_RANDOM_ELA_PROBABILITY = 0.5
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
FE_ANALYSIS_RATIO = 0.05
EPS = 1e-12


@dataclass(frozen=True)
class ControllerBundle:
    model: Any
    model_name: str
    model_family: str
    threshold_mode: str
    threshold: float
    feature_columns: list[str]
    training_summary_path: Path
    model_path: Path


@dataclass(frozen=True)
class OnlineSelector:
    sbs_algorithm: str
    buckets: tuple[float, ...]
    models: dict[float, tuple[Any, str]]

    def select(self, ela_features: dict[str, Any], remaining_ratio: float) -> tuple[str, float, str, float]:
        started = perf_counter()
        bucket = min(self.buckets, key=lambda value: (abs(value - remaining_ratio), value))
        model, status = self.models[bucket]
        frame = pd.DataFrame([{column: ela_features[column] for column in ELA_FEATURE_COLUMNS}])
        frame.insert(0, "problem_id", str(ela_features["problem_id"]))
        predicted = _predict_algorithms(model, status, frame, self.sbs_algorithm)
        runtime_selection = perf_counter() - started
        return str(predicted[str(ela_features["problem_id"])]), float(bucket), status, float(runtime_selection)


def evaluate_online_controller(
    *,
    config_path: Path,
    train_config_path: Path,
    training_summary_path: Path,
    output_dir: Path,
    model_name: str,
    threshold_mode: str,
    sampling_protocol: str,
    random_ela_probability: float,
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
    selector = None if summarize_only else _fit_online_selector(train_config_path)
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
            random_ela_probability=random_ela_probability,
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
            checkpoint_plan=checkpoint_plan,
            random_ela_probability=random_ela_probability,
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
                        sampling_protocol=sampling_protocol,
                        decision_check_frequency=decision_check_frequency,
                        random_ela_probability=random_ela_probability,
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
        random_ela_probability=random_ela_probability,
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
    controller: ControllerBundle,
    selector: OnlineSelector | None,
    checkpoint_plan: dict[int, list[tuple[float, int]]],
    random_ela_probability: float,
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
                    "sampling_protocol": sampling_protocol,
                    "decision_check_frequency": decision_check_frequency,
                    "random_ela_probability": float(random_ela_probability),
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
        random_ela_probability=random_ela_probability,
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
                sampling_protocol=str(job["sampling_protocol"]),
                decision_check_frequency=str(job["decision_check_frequency"]),
                random_ela_probability=float(job["random_ela_probability"]),
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
    controller: ControllerBundle,
    random_ela_probability: float,
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
        random_ela_probability=random_ela_probability,
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
    controller: ControllerBundle,
    default_algorithm: str,
    random_ela_probability: float,
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
        "random_ela_probability": random_ela_probability,
        "random_repetitions": random_repetitions,
        "random_seed": random_seed,
        "rows": int(len(result)),
        "base_runs": base_runs,
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
            "external_ela_features_used_as_controller_input": False,
            "function_id_algorithm_id_or_optimizer_internal_parameters_used_as_controller_input": False,
            "controller_inputs_are_behavior_features_only": True,
        },
        "scope_notes": [
            "All policies start from the SBS/default optimizer probe state when a decision is needed.",
            "Behavior sampling is the decision-check frequency: every checkpoint is also a possible ELA trigger point.",
            "always_ela executes ELA at the first checkpoint in the active sampling protocol, so it is an after-probe always-analysis baseline.",
            "random_ela samples one independent checkpoint trigger stream per repetition and run at each decision-check point.",
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


def _load_controller(training_summary_path: Path, model_name: str, threshold_mode: str) -> ControllerBundle:
    summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    feature_columns = [str(column) for column in summary.get("feature_columns", [])]
    if not feature_columns or not set(feature_columns).issubset(BEHAVIOR_FEATURE_COLUMNS):
        raise ValueError("controller feature columns must be a non-empty subset of behavior features")
    model_path = _model_path(summary, model_name)
    threshold = _threshold(summary, model_name, threshold_mode)
    model_family = _model_family(summary, model_name)
    return ControllerBundle(
        model=joblib.load(model_path),
        model_name=model_name,
        model_family=model_family,
        threshold_mode=threshold_mode,
        threshold=threshold,
        feature_columns=feature_columns,
        training_summary_path=training_summary_path,
        model_path=model_path,
    )


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


def _fit_online_selector(train_config_path: Path) -> OnlineSelector:
    train_config = load_config(train_config_path)
    train_features = _read_feature_file(_ela_feature_path(train_config, Path("results/ela")))
    train_performance = _read_performance(train_config, None, None)
    train_budget_map = _checkpoint_budget_map(train_config, train_performance, None)
    buckets = tuple(sorted({remaining for values in train_budget_map.values() for remaining, _ in values}))
    sbs_algorithm = _single_best_solver(train_performance)
    models = {}
    for remaining_ratio in buckets:
        bucket_ratio_by_dimension = _bucket_ratio_by_dimension(train_budget_map, remaining_ratio)
        target = _best_algorithm_by_problem(train_performance, bucket_ratio_by_dimension)
        models[remaining_ratio] = _fit_selector(train_features, target)
    return OnlineSelector(sbs_algorithm=sbs_algorithm, buckets=buckets, models=models)


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
    if sampling_protocol == "dense_decision_check" and output_dir == DEFAULT_OUTPUT_DIR:
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
    controller: ControllerBundle,
    selector: OnlineSelector,
    sampling_protocol: str,
    decision_check_frequency: str,
    random_ela_probability: float,
    random_repetitions: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    suite = str(config["suite"]).lower()
    problem = make_problem({"suite": suite, "function": function, "instance": 1, "dimension": dimension})
    try:
        no_ela = _run_threshold_policy(
            problem=problem,
            config=config,
            function=function,
            seed=seed,
            fe_total=fe_total,
            checkpoint_plan=checkpoint_plan,
            selector=selector,
            controller=None,
            policy_name="sbs_no_ela",
            trigger_mode="never",
            sampling_protocol=sampling_protocol,
            decision_check_frequency=decision_check_frequency,
            repetition=None,
        )
        always_ela = _run_threshold_policy(
            problem=problem,
            config=config,
            function=function,
            seed=seed,
            fe_total=fe_total,
            checkpoint_plan=checkpoint_plan,
            selector=selector,
            controller=None,
            policy_name="always_ela",
            trigger_mode="first_checkpoint",
            sampling_protocol=sampling_protocol,
            decision_check_frequency=decision_check_frequency,
            repetition=None,
        )
        controller_row = _run_threshold_policy(
            problem=problem,
            config=config,
            function=function,
            seed=seed,
            fe_total=fe_total,
            checkpoint_plan=checkpoint_plan,
            selector=selector,
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
                controller=None,
                policy_name="random_ela_p50",
                trigger_mode="random",
                sampling_protocol=sampling_protocol,
                decision_check_frequency=decision_check_frequency,
                repetition=repetition,
                random_ela_probability=random_ela_probability,
                random_seed=random_seed,
            )
            for repetition in range(random_repetitions)
        ]
        rows = [no_ela, always_ela, controller_row, *random_rows]
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
    controller: ControllerBundle | None,
    policy_name: str,
    trigger_mode: str,
    sampling_protocol: str,
    decision_check_frequency: str,
    repetition: int | None,
    random_ela_probability: float = 0.5,
    random_seed: int = 1701,
) -> dict[str, Any]:
    default_algorithm = selector.sbs_algorithm
    population_size = int(config["population_size"])
    fe_analysis = int(FE_ANALYSIS_RATIO * fe_total)
    trajectory_rows: list[dict[str, Any]] = []
    current_population = None
    current_fitness = None
    current_best = np.inf
    current_fe = 0
    runtime_probe = 0.0
    triggered = False
    trigger_ratio = None
    trigger_score = None
    selected_algorithm = default_algorithm
    selected_bucket = None
    selector_status = "not_used"
    runtime_analysis = 0.0
    runtime_selection = 0.0

    for ratio, checkpoint_fe in checkpoint_plan:
        delta = checkpoint_fe - current_fe
        if delta <= 0:
            continue
        if current_fe == 0:
            settings = OptimizerSettings(population_size=population_size, checkpoint_ratios=(1.0,))
            started = perf_counter()
            records = run_optimizer(
                algorithm=default_algorithm,
                problem=problem,
                seed=seed,
                fe_total=checkpoint_fe,
                settings=settings,
            )
            runtime_probe += perf_counter() - started
            record = records[-1]
            current_population = np.asarray(record.population, dtype=float)
            current_fitness = np.asarray(record.fitness, dtype=float)
            current_best = float(record.best_fitness)
        else:
            continuation = run_population_continuation(
                algorithm=default_algorithm,
                problem=problem,
                seed=seed,
                function=function,
                instance=1,
                generation=max(1, current_fe // population_size),
                event=10,
                fe_budget=delta,
                population=current_population,
                fitness=current_fitness,
                best_fitness=current_best,
                settings=OptimizerSettings(population_size=population_size, checkpoint_ratios=(1.0,)),
            )
            runtime_probe += continuation.runtime_seconds
            current_population = np.asarray(continuation.population, dtype=float)
            current_fitness = np.asarray(continuation.fitness, dtype=float)
            current_best = float(continuation.best_fitness)
        current_fe = checkpoint_fe
        trajectory_record = TrajectoryRecord.from_arrays(
            problem_id=problem.problem_id,
            family=problem.family,
            dimension=problem.dimension,
            algorithm=default_algorithm,
            seed=seed,
            fe=current_fe,
            fe_total=fe_total,
            population=current_population,
            fitness=current_fitness,
            best_fitness=current_best,
            fe_ratio=ratio,
        )
        trajectory_rows.append(trajectory_record.__dict__)
        behavior_row = extract_behavior_rows([row.copy() for row in trajectory_rows])[-1]
        should_trigger, trigger_score = _should_trigger(
            behavior_row=behavior_row,
            controller=controller,
            trigger_mode=trigger_mode,
            random_ela_probability=random_ela_probability,
            random_seed=random_seed,
            seed=seed,
            function=function,
            dimension=problem.dimension,
            repetition=repetition,
        )
        if should_trigger:
            triggered = True
            trigger_ratio = ratio
            ela_features = extract_ela_for_problem(
                problem=problem,
                seed=0,
                fe_analysis=fe_analysis,
                function=function,
                instance=1,
            )
            runtime_analysis = float(ela_features["runtime_analysis"])
            ela_features = {"problem_id": problem.problem_id, **ela_features}
            remaining = max(fe_total - current_fe - fe_analysis, 0)
            remaining_ratio = round(remaining / fe_total, 6)
            selected_algorithm, selected_bucket, selector_status, runtime_selection = selector.select(
                ela_features,
                remaining_ratio,
            )
            break

    if triggered:
        remaining_budget = max(fe_total - current_fe - fe_analysis, 0)
        # Triggered selector path uses population transfer; ELA samples only determine the selected algorithm.
        after = run_population_continuation(
            algorithm=selected_algorithm,
            problem=problem,
            seed=seed,
            function=function,
            instance=1,
            generation=max(1, current_fe // population_size),
            event=20,
            fe_budget=remaining_budget,
            population=current_population,
            fitness=current_fitness,
            best_fitness=current_best,
            settings=OptimizerSettings(population_size=population_size, checkpoint_ratios=(1.0,)),
        )
        final_performance = float(after.best_fitness)
        runtime_after = float(after.runtime_seconds)
        fe_after = int(after.evaluations)
        fe_used = int(current_fe + fe_analysis + fe_after)
    else:
        remaining_budget = max(fe_total - current_fe, 0)
        after = run_population_continuation(
            algorithm=default_algorithm,
            problem=problem,
            seed=seed,
            function=function,
            instance=1,
            generation=max(1, current_fe // population_size),
            event=30,
            fe_budget=remaining_budget,
            population=current_population,
            fitness=current_fitness,
            best_fitness=current_best,
            settings=OptimizerSettings(population_size=population_size, checkpoint_ratios=(1.0,)),
        )
        final_performance = float(after.best_fitness)
        runtime_after = float(after.runtime_seconds)
        fe_after = int(after.evaluations)
        fe_used = int(current_fe + fe_after)

    policy_category = "controller" if policy_name == "current_controller" else "baseline"
    return {
        "policy_name": policy_name,
        "policy_category": policy_category,
        "sampling_protocol": sampling_protocol,
        "decision_check_frequency": decision_check_frequency,
        "random_repetition": repetition,
        "default_algorithm": default_algorithm,
        "selected_algorithm": selected_algorithm,
        "selector_status": selector_status,
        "selected_bucket_remaining_ratio": selected_bucket,
        "ela_called": bool(triggered),
        "trigger_FE": int(current_fe) if triggered else None,
        "trigger_FE_ratio": float(trigger_ratio) if triggered else None,
        "decision_score": float(trigger_score) if trigger_score is not None else None,
        "decision_threshold": float(controller.threshold) if controller is not None else None,
        "FE_total": int(fe_total),
        "FE_probe": int(current_fe),
        "FE_analysis": int(fe_analysis) if triggered else 0,
        "FE_after_decision_optimization": int(fe_after),
        "FE_used": int(fe_used),
        "runtime_probe": float(runtime_probe),
        "runtime_analysis": float(runtime_analysis),
        "runtime_selection": float(runtime_selection),
        "runtime_after_decision_optimization": float(runtime_after),
        "runtime_total": float(runtime_probe + runtime_analysis + runtime_selection + runtime_after),
        "final_performance": final_performance,
    }


def _should_trigger(
    *,
    behavior_row: dict[str, Any],
    controller: ControllerBundle | None,
    trigger_mode: str,
    random_ela_probability: float,
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
            raise ValueError("controller trigger mode requires a controller bundle")
        frame = pd.DataFrame([{column: behavior_row[column] for column in controller.feature_columns}])
        score = float(controller.model.predict(frame)[0])
        return bool(score > controller.threshold), score
    if trigger_mode == "random":
        rng = np.random.default_rng(
            np.random.SeedSequence([int(random_seed), int(seed), int(function), int(dimension), int(repetition or 0), int(behavior_row["FE"])])
        )
        return bool(rng.random() < random_ela_probability), None
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
        "ela_call_rate": float(frame["ela_called"].mean()),
        "mean_FE_used": float(frame["FE_used"].mean()),
        "mean_runtime_total": float(frame["runtime_total"].mean()),
        "mean_trigger_FE_ratio": float(frame["trigger_FE_ratio"].dropna().mean()) if frame["trigger_FE_ratio"].notna().any() else None,
    }


def _relative_summary(rows: pd.DataFrame) -> pd.DataFrame:
    baseline = rows[rows["policy_name"] == "sbs_no_ela"][
        ["problem_id", "dimension", "seed", "final_performance", "runtime_total", "FE_used"]
    ].rename(
        columns={
            "final_performance": "sbs_no_ela_final_performance",
            "runtime_total": "sbs_no_ela_runtime_total",
            "FE_used": "sbs_no_ela_FE_used",
        }
    )
    joined = rows.merge(baseline, on=["problem_id", "dimension", "seed"], how="left")
    joined["final_performance_delta_vs_sbs_no_ela"] = (
        joined["final_performance"] - joined["sbs_no_ela_final_performance"]
    )
    joined["runtime_delta_vs_sbs_no_ela"] = joined["runtime_total"] - joined["sbs_no_ela_runtime_total"]
    joined["FE_used_delta_vs_sbs_no_ela"] = joined["FE_used"] - joined["sbs_no_ela_FE_used"]
    return (
        joined.groupby("policy_name", as_index=False)
        .agg(
            rows=("final_performance", "size"),
            mean_final_performance_delta_vs_sbs_no_ela=("final_performance_delta_vs_sbs_no_ela", "mean"),
            median_final_performance_delta_vs_sbs_no_ela=("final_performance_delta_vs_sbs_no_ela", "median"),
            mean_runtime_delta_vs_sbs_no_ela=("runtime_delta_vs_sbs_no_ela", "mean"),
            mean_FE_used_delta_vs_sbs_no_ela=("FE_used_delta_vs_sbs_no_ela", "mean"),
            ela_call_rate=("ela_called", "mean"),
        )
        .sort_values("mean_final_performance_delta_vs_sbs_no_ela")
        .reset_index(drop=True)
    )


def _random_repetition_summary(rows: pd.DataFrame) -> pd.DataFrame:
    random_rows = rows[rows["policy_name"] == "random_ela_p50"].copy()
    if random_rows.empty:
        return pd.DataFrame()
    summary = (
        random_rows.groupby("random_repetition", as_index=False)
        .agg(
            rows=("final_performance", "size"),
            mean_final_performance=("final_performance", "mean"),
            ela_call_rate=("ela_called", "mean"),
            mean_FE_used=("FE_used", "mean"),
            mean_runtime_total=("runtime_total", "mean"),
        )
    )
    metric_rows = []
    for metric in ("mean_final_performance", "ela_call_rate", "mean_FE_used", "mean_runtime_total"):
        values = summary[metric].astype(float)
        metric_rows.append(
            {
                "policy_name": "random_ela_p50",
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
            "ela_call_rate",
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
            "- 每个 checkpoint 同时是 behavior observation 点和可能触发 ELA 的 decision-check point。",
            "- `always_ela` 表示在当前 sampling protocol 的第一个 checkpoint 后必定执行 ELA 的 after-probe baseline。",
            "- `dense_decision_check` 只能解释为决策检查频率敏感性，不是纯被动观测频率实验。",
            "",
            "## Overall Policies",
            "",
            _markdown_table(overall),
            "",
            "## Relative To SBS/No-ELA",
            "",
            _markdown_table(relative_summary),
            "",
            "## Random-ELA Repetition Summary",
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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH)
    parser.add_argument("--training-summary", type=Path, default=DEFAULT_TRAINING_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--threshold-mode", default=DEFAULT_THRESHOLD_MODE)
    parser.add_argument("--sampling-protocol", choices=SAMPLING_PROTOCOLS, default=DEFAULT_SAMPLING_PROTOCOL)
    parser.add_argument("--random-ela-probability", type=float, default=DEFAULT_RANDOM_ELA_PROBABILITY)
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

    evaluate_online_controller(
        config_path=args.config,
        train_config_path=args.train_config,
        training_summary_path=args.training_summary,
        output_dir=args.output_dir,
        model_name=args.model_name,
        threshold_mode=args.threshold_mode,
        sampling_protocol=args.sampling_protocol,
        random_ela_probability=args.random_ela_probability,
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
