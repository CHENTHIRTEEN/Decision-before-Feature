from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from behavior_with_ela.action_dataset import (
    GFE_GATE_BEHAVIOR_FEATURE_COLUMNS,
    _aggregate_state_actions,
    _build_replication_plan,
    _check_checkpoint_state,
    _component_functions,
    _evaluate_state_actions,
    _gain_class,
    _problem_instance,
    _replication_state_key,
    action_shard_paths,
)
from behavior_with_ela.collection import optimizer_checkpoint_shard_path, shard_paths
from behavior_with_ela.local_landscape import (
    LOCAL_LANDSCAPE_FEATURE_COLUMNS,
    LOCAL_LANDSCAPE_METADATA_COLUMNS,
)
from behavior_with_ela.protocol import (
    ExperimentConfig,
    SuiteConfig,
    function_label,
    load_experiment_config,
    make_experiment_problem,
)
from landscape_queries.cheap import calculate_descriptor_cheap
from landscape_queries.sampling import sample_problem
from landscape_queries.specs import DESCRIPTOR_CHEAP_COLUMNS, get_query_spec
from trajectory.sampling import SAMPLING_METADATA_COLUMNS


QUERY_ACTION_PROTOCOL = "state_action_query_adjusted_terminal_v1"
QUERY_BUDGET_MODE = "query_adjusted_equal_total_FE"


def build_query_adjusted_datasets(
    *,
    config_path: str | Path,
    only_splits: tuple[str, ...] | None = None,
    only_functions: tuple[int, ...] | None = None,
    workers: int = 1,
    overwrite: bool = False,
) -> dict[str, int]:
    config = load_experiment_config(config_path)
    suites = _selected_suites(config, only_splits)
    tasks = [
        (suite, function)
        for suite in suites
        for function in suite.functions
        if only_functions is None or function in set(only_functions)
    ]
    if not tasks:
        raise ValueError("no query-adjusted action shards were selected")
    if workers < 1:
        raise ValueError("workers must be at least one")

    pending: list[tuple[SuiteConfig, int]] = []
    skipped = 0
    for suite, function in tasks:
        outputs = query_action_shard_paths(config, suite, function)
        existing = tuple(path.exists() for path in outputs)
        if all(existing) and not overwrite:
            skipped += 1
            continue
        if any(existing) and not all(existing) and not overwrite:
            raise FileExistsError(
                f"query-adjusted shard outputs must be regenerated together: {outputs[0].parent}"
            )
        pending.append((suite, function))

    written = 0
    if workers == 1:
        for suite, function in pending:
            _build_and_write_query_shard(config, suite, function, overwrite)
            written += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _build_and_write_query_shard,
                    config,
                    suite,
                    function,
                    overwrite,
                )
                for suite, function in pending
            ]
            for future in as_completed(futures):
                future.result()
                written += 1
    return {"written_shards": written, "skipped_shards": skipped}


def query_action_shard_paths(
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
) -> tuple[Path, Path, Path, Path]:
    directory = (
        config.output_root
        / "query_actions"
        / suite.split
        / function_label(suite.suite, function)
        / f"dimension_{config.dimension}"
    )
    return (
        directory / "query_samples.parquet",
        directory / "query_state_descriptors.parquet",
        directory / "query_action_repetitions.parquet",
        directory / "query_action_gain_dataset.parquet",
    )


def _build_and_write_query_shard(
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    overwrite: bool,
) -> tuple[int, int]:
    trajectory_path, _, _ = shard_paths(config, suite, function)
    checkpoint_path = optimizer_checkpoint_shard_path(config, suite, function)
    _, no_query_path = action_shard_paths(config, suite, function)
    if (
        not trajectory_path.exists()
        or not checkpoint_path.exists()
        or not no_query_path.exists()
    ):
        raise FileNotFoundError(
            "query-adjusted data requires trajectories, optimizer checkpoints, and no-query action data"
        )
    outputs = query_action_shard_paths(config, suite, function)
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in outputs:
            path.unlink(missing_ok=True)
    trajectories = pd.read_parquet(trajectory_path)
    optimizer_checkpoints = pd.read_parquet(checkpoint_path)
    no_query = pd.read_parquet(no_query_path)
    samples, descriptors, repetitions, aggregate = build_query_adjusted_rows(
        config=config,
        suite=suite,
        function=function,
        trajectory_rows=trajectories.to_dict(orient="records"),
        optimizer_checkpoint_rows=optimizer_checkpoints.to_dict(orient="records"),
        no_query_action_rows=no_query.to_dict(orient="records"),
    )
    pd.DataFrame(samples).to_parquet(outputs[0], index=False)
    pd.DataFrame(descriptors).to_parquet(outputs[1], index=False)
    pd.DataFrame(repetitions).to_parquet(outputs[2], index=False)
    pd.DataFrame(aggregate).to_parquet(outputs[3], index=False)
    return len(repetitions), len(aggregate)


def build_query_adjusted_rows(
    *,
    config: ExperimentConfig,
    suite: SuiteConfig,
    function: int,
    trajectory_rows: list[dict],
    optimizer_checkpoint_rows: list[dict],
    no_query_action_rows: list[dict],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    query_spec = get_query_spec(config.query.query_id)
    if query_spec.sample_design_id != config.query.sample_design_id:
        raise ValueError("configured query and sample design are inconsistent")
    query_fe = query_spec.sample_design.sample_size(config.dimension)
    replication_plan = _build_replication_plan(
        config=config,
        suite=suite,
        function=function,
        trajectory_rows=trajectory_rows,
    )
    state_context = _state_context_map(no_query_action_rows)
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for row in trajectory_rows:
        key = (str(row["problem_id"]), str(row["algorithm"]), int(row["seed"]))
        grouped.setdefault(key, []).append(dict(row))
    checkpoint_by_state: dict[tuple[str, str, int, int, int], bytes] = {}
    for row in optimizer_checkpoint_rows:
        key = (
            str(row["problem_id"]),
            str(row["algorithm"]),
            int(row["seed"]),
            int(row["FE"]),
            int(row["decision_opportunity_index"]),
        )
        if key in checkpoint_by_state:
            raise ValueError("optimizer checkpoint rows contain a duplicate query state")
        checkpoint_by_state[key] = bytes(row["optimizer_state_payload"])

    query_cache: dict[tuple[str, int], tuple[dict, dict[str, float | None]]] = {}
    sample_rows: list[dict] = []
    descriptor_rows: list[dict] = []
    repetition_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    component_functions = _component_functions(suite, function)

    for (problem_id, prefix_algorithm, seed), rows in grouped.items():
        ordered = sorted(rows, key=lambda item: int(item["FE"]))
        instance = _problem_instance(ordered[0], suite.suite)
        problem = make_experiment_problem(
            suite,
            function=function,
            instance=instance,
            dimension=config.dimension,
            boundary_handling=config.boundary_handling,
        )
        try:
            cache_key = (problem_id, seed)
            if cache_key not in query_cache:
                query_sample = sample_problem(
                    problem=problem,
                    sample_design=query_spec.sample_design,
                    base_seed=seed,
                    function=function,
                    instance=instance,
                    success_gap_target=config.success_gap_target,
                    failure_loss_cap=config.failure_loss_cap,
                )
                if (
                    not bool(query_sample["sample_path_completed"])
                    or int(query_sample["sample_effective_FE"]) != int(query_fe)
                ):
                    raise RuntimeError(
                        "independent Query sample did not complete its FE budget: "
                        f"status={query_sample['sample_status']}, "
                        f"effective_FE={int(query_sample['sample_effective_FE'])}, "
                        f"planned_FE={int(query_fe)}"
                    )
                descriptors = _query_descriptors(
                    query_sample,
                    lower=problem.lower_bounds,
                    upper=problem.upper_bounds,
                )
                query_cache[cache_key] = (query_sample, descriptors)
                sample_rows.append(
                    {
                        "split": suite.split,
                        "suite": suite.suite,
                        "problem_id": problem_id,
                        "function_id": str(ordered[0]["function_id"]),
                        "family": str(ordered[0]["family"]),
                        "cv_group_id": str(
                            ordered[0].get("cv_group_id", ordered[0]["function_id"])
                        ),
                        "dimension": config.dimension,
                        "seed": seed,
                        "query_id": query_spec.query_id,
                        "sample_design_id": query_spec.sample_design_id,
                        "query_preprocessing_id": query_spec.preprocessing_id,
                        "query_evaluation_reuse_scope": (
                            "same_problem_seed_hypothetical_decision_states"
                        ),
                        **query_sample,
                        **descriptors,
                    }
                )
            query_sample, descriptors = query_cache[cache_key]
            for opportunity_index, row in enumerate(ordered):
                checkpoint_fe = int(row["FE"])
                checkpoint_key = (
                    problem_id,
                    prefix_algorithm,
                    seed,
                    checkpoint_fe,
                    opportunity_index,
                )
                payload = checkpoint_by_state.get(checkpoint_key)
                if payload is None:
                    raise ValueError(
                        f"query state is missing its optimizer checkpoint: {checkpoint_key}"
                    )
                state = pickle.loads(payload)
                _check_checkpoint_state(state, row)
                action_budget = config.fe_total - checkpoint_fe - query_fe
                if action_budget < config.query.minimum_post_query_FE:
                    raise ValueError("decision state does not retain minimum post-query FE")
                context_key = (
                    problem_id,
                    prefix_algorithm,
                    seed,
                    checkpoint_fe,
                    opportunity_index,
                )
                context = state_context.get(context_key)
                if context is None:
                    raise ValueError(
                        f"query state is missing its no-query context: {context_key}"
                    )
                reason = replication_plan[_replication_state_key(row)]
                repetitions = (
                    config.replication.selected_state_repetitions
                    if reason != "full_coverage_only"
                    else config.replication.full_coverage
                )
                raw = _evaluate_state_actions(
                    config=config,
                    suite=suite,
                    function=function,
                    instance=instance,
                    prefix_algorithm=prefix_algorithm,
                    seed=seed,
                    checkpoint_state=state,
                    problem=problem,
                    checkpoint_fe=checkpoint_fe,
                    repetitions=repetitions,
                    fe_budget_override=action_budget,
                )
                raw = _apply_query_endpoint(
                    rows=raw,
                    query_sample=query_sample,
                    config=config,
                )
                common = {
                    **context,
                    "component_functions": list(component_functions),
                    "query_action_protocol": QUERY_ACTION_PROTOCOL,
                    "action_budget_mode": QUERY_BUDGET_MODE,
                    "query_id": query_spec.query_id,
                    "sample_design_id": query_spec.sample_design_id,
                    "query_preprocessing_id": query_spec.preprocessing_id,
                    "FE_query": int(query_fe),
                    "FE_no_query_optimization": int(config.fe_total - checkpoint_fe),
                    "FE_action_optimization": int(action_budget),
                    "remaining_budget_ratio": float(action_budget / config.fe_total),
                    "query_sample_status": str(query_sample["sample_status"]),
                    "query_sample_path_completed": bool(
                        query_sample["sample_path_completed"]
                    ),
                    "query_sample_effective_FE": int(query_sample["sample_effective_FE"]),
                    "query_sample_best_gap": float(query_sample["query_best_gap"]),
                    "query_sample_first_hit_offset": query_sample[
                        "query_first_hit_offset"
                    ],
                    "query_sample_target_hit_observed": bool(
                        query_sample["sample_target_hit_observed"]
                    ),
                    "query_sample_failure_type": str(
                        query_sample["sample_failure_type"]
                    ),
                    "query_sample_failure_message": str(
                        query_sample["sample_failure_message"]
                    ),
                    **descriptors,
                }
                descriptor_rows.append(
                    {
                        key: common[key]
                        for key in (
                            "split",
                            "suite",
                            "problem_id",
                            "function_id",
                            "family",
                            "cv_group_id",
                            "dimension",
                            "prefix_algorithm",
                            "seed",
                            "FE",
                            "FE_ratio",
                            "FE_total",
                            "decision_opportunity_index",
                            "state_id",
                            "query_id",
                            "sample_design_id",
                            "query_preprocessing_id",
                            "FE_query",
                            "query_sample_status",
                            "query_sample_path_completed",
                            "query_sample_best_gap",
                            *DESCRIPTOR_CHEAP_COLUMNS,
                        )
                    }
                )
                repetition_rows.extend({**common, **item} for item in raw)
                summaries = _aggregate_state_actions(
                    common=common,
                    rows=raw,
                    portfolio=config.algorithms,
                    practical_delta=config.domain_gain_delta,
                )
                for summary in summaries:
                    action_values = [
                        item
                        for item in raw
                        if str(item["candidate_action"])
                        == str(summary["candidate_action"])
                    ]
                    summary["query_continuation_action_loss"] = float(
                        np.median(
                            [
                                item["query_continuation_action_loss"]
                                for item in action_values
                            ]
                        )
                    )
                    summary["query_continuation_log10_action_loss"] = float(
                        np.median(
                            [
                                item["query_continuation_log10_action_loss"]
                                for item in action_values
                            ]
                        )
                    )
                    summary["all_query_adjusted_paths_completed"] = bool(
                        query_sample["sample_path_completed"]
                        and summary["all_action_paths_completed"]
                    )
                aggregate_rows.extend(summaries)
        finally:
            problem.close()
    return sample_rows, descriptor_rows, repetition_rows, aggregate_rows


def _state_context_map(
    no_query_action_rows: list[dict],
) -> dict[tuple[str, str, int, int, int], dict]:
    selected_columns = (
        "split",
        "suite",
        "problem_id",
        "function_id",
        "family",
        "cv_group_id",
        "dimension",
        "boundary_handling",
        "prefix_algorithm",
        "seed",
        "FE",
        "FE_ratio",
        "FE_total",
        "prefix_best_fitness",
        "benchmark_reference_value",
        "prefix_gap",
        "epsilon_p",
        "domain_gain_delta",
        "decision_opportunity_index",
        "planned_action_repetitions",
        "selected_for_additional_repetitions",
        "replication_selection_reason",
        "replication_stratum",
        "state_id",
        *SAMPLING_METADATA_COLUMNS,
        *GFE_GATE_BEHAVIOR_FEATURE_COLUMNS,
        *LOCAL_LANDSCAPE_METADATA_COLUMNS,
        *LOCAL_LANDSCAPE_FEATURE_COLUMNS,
    )
    contexts: dict[tuple[str, str, int, int, int], dict] = {}
    for row in no_query_action_rows:
        key = (
            str(row["problem_id"]),
            str(row["prefix_algorithm"]),
            int(row["seed"]),
            int(row["FE"]),
            int(row["decision_opportunity_index"]),
        )
        if key in contexts:
            continue
        missing = set(selected_columns).difference(row)
        if missing:
            raise ValueError(f"no-query state context is missing columns: {sorted(missing)}")
        contexts[key] = {column: row[column] for column in selected_columns}
    return contexts


def _query_descriptors(
    query_sample: dict,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float | None]:
    if not bool(query_sample["sample_path_completed"]):
        return {column: None for column in DESCRIPTOR_CHEAP_COLUMNS}
    values = calculate_descriptor_cheap(
        np.asarray(query_sample["X"], dtype=float),
        np.asarray(query_sample["y"], dtype=float),
        lower,
        upper,
    )
    return {
        column: (
            float(values[column]) if np.isfinite(float(values[column])) else None
        )
        for column in DESCRIPTOR_CHEAP_COLUMNS
    }


def _apply_query_endpoint(
    *,
    rows: list[dict],
    query_sample: dict,
    config: ExperimentConfig,
) -> list[dict]:
    sample_completed = bool(query_sample["sample_path_completed"])
    query_gap = float(query_sample["query_best_gap"])
    for row in rows:
        row["query_continuation_action_loss"] = float(row["action_loss"])
        row["query_continuation_log10_action_loss"] = float(
            row["log10_action_loss"]
        )
        row["query_continuation_gain_vs_continue"] = float(
            row["action_gain_vs_continue"]
        )
        row["query_continuation_gain_class"] = str(row["action_gain_class"])
        path_completed = sample_completed and str(row["action_status"]) == "completed"
        operational_gap = (
            min(float(row["action_loss"]), query_gap)
            if path_completed
            else config.failure_loss_cap
        )
        row["query_adjusted_path_completed"] = bool(path_completed)
        row["query_sample_improved_terminal"] = bool(
            path_completed and query_gap < float(row["action_loss"])
        )
        row["action_loss"] = float(operational_gap)
        row["log10_action_loss"] = float(
            np.log10(
                np.clip(
                    operational_gap,
                    config.log10_gap_floor,
                    config.log10_gap_cap,
                )
            )
        )
    by_replicate: dict[int, list[dict]] = {}
    for row in rows:
        by_replicate.setdefault(int(row["replicate_id"]), []).append(row)
    for replicate_rows in by_replicate.values():
        continue_rows = [row for row in replicate_rows if row["action_equals_prefix"]]
        if len(continue_rows) != 1:
            raise RuntimeError("query matrix replicate requires one continue action")
        reference = float(continue_rows[0]["log10_action_loss"])
        for row in replicate_rows:
            gain = reference - float(row["log10_action_loss"])
            row["action_gain_vs_continue"] = float(gain)
            row["action_gain_class"] = _gain_class(gain, config.domain_gain_delta)
    return rows


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
        description="Build query-adjusted state-action matrices with real Query FE."
    )
    parser.add_argument("--config", default="configs/behavior_with_ela_train.yaml")
    parser.add_argument("--only-split", action="append", default=None)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = build_query_adjusted_datasets(
        config_path=args.config,
        only_splits=None if args.only_split is None else tuple(args.only_split),
        only_functions=(
            None if args.only_function is None else tuple(args.only_function)
        ),
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(
        f"wrote {summary['written_shards']} query-adjusted action shards; "
        f"skipped {summary['skipped_shards']} existing shards"
    )


if __name__ == "__main__":
    main()
