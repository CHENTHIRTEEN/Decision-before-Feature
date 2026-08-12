from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks import make_problem
from benchmarks.core import Problem
from experiments.phase1_batch_common import algorithms, fe_total_for_dimension, load_config, make_shards
from landscape_queries.specs import SAMPLE_DESIGN_SPECS, get_sample_design_spec
from optimizers import (
    NO_QUERY_TRANSFER_EVENT,
    OptimizerSettings,
    QUERY_TRANSFER_EVENT,
    advance_optimizer_state,
    clone_optimizer_state,
    initialize_optimizer_state,
    initialize_transferred_optimizer_state,
)
from optimizers.state import OptimizerState
from selection_reference.common import split_name, train_derived_sbs


EPS = 1e-12
MIN_LABEL_RATIO = 0.12
ACTION_LOSS_PROTOCOL = "shared_state_query_budget_native_continue_or_population_transfer_v2"
STATE_KEY_COLUMNS = (
    "split",
    "problem_id",
    "family",
    "dimension",
    "prefix_algorithm",
    "seed",
    "FE",
)


def generate_state_action_losses(
    *,
    config_path: Path,
    train_config_path: Path,
    sample_design_id: str,
    output_path: Path,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    default_algorithm: str | None,
    all_prefixes: bool,
    max_states: int | None,
    overwrite: bool,
) -> dict[str, int | str]:
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"state-action loss output already exists; pass --overwrite: {output_path}")
        output_path.unlink()
    config = load_config(config_path)
    sample_design = get_sample_design_spec(sample_design_id)
    split = split_name(config)
    suite = str(config["suite"]).lower()
    portfolio = tuple(str(value) for value in algorithms(config))
    default = str(default_algorithm or train_derived_sbs(train_config_path))
    if default not in portfolio:
        raise ValueError(f"default algorithm {default!r} is not in the configured portfolio")
    settings = OptimizerSettings(population_size=int(config["population_size"]), checkpoint_ratios=(1.0,))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    state_count = 0
    action_count = 0
    problem_cache: dict[tuple[int, int, int], Problem] = {}
    try:
        for shard in make_shards(config, only_functions, only_dimensions):
            trajectory_path = shard.output_path
            if not trajectory_path.exists():
                raise FileNotFoundError(f"missing trajectory shard: {trajectory_path}")
            trajectory_rows = pq.read_table(trajectory_path).to_pylist()
            rows, used_states = _evaluate_shard(
                split=split,
                suite=suite,
                config=config,
                trajectory_rows=trajectory_rows,
                sample_design_id=sample_design.sample_design_id,
                settings=settings,
                problem_cache=problem_cache,
                portfolio=portfolio,
                default_algorithm=default,
                all_prefixes=all_prefixes,
                max_states=None if max_states is None else max_states - state_count,
            )
            if rows:
                if writer is None:
                    writer = pq.ParquetWriter(output_path, _schema())
                writer.write_table(pa.Table.from_pylist(rows, schema=_schema()))
            state_count += used_states
            action_count += len(rows)
            if max_states is not None and state_count >= max_states:
                break
    except BaseException:
        if writer is not None:
            writer.close()
            writer = None
        if output_path.exists():
            output_path.unlink()
        raise
    finally:
        if writer is not None:
            writer.close()
        for problem in problem_cache.values():
            problem.close()
    if state_count == 0:
        raise ValueError("no eligible shared states were evaluated")
    print(f"wrote {action_count} action-loss rows for {state_count} shared states to {output_path}")
    return {"states": state_count, "rows": action_count, "output": str(output_path)}


def evaluate_candidate_actions(
    *,
    checkpoint_state: OptimizerState,
    problem: Problem,
    portfolio: tuple[str, ...],
    fe_budget: int,
    seed: int,
    function: int,
    instance: int,
) -> list[dict[str, float | str]]:
    prefix_algorithm = str(checkpoint_state.algorithm)
    if len(portfolio) != 4 or len(set(portfolio)) != 4:
        raise ValueError("action portfolio must contain exactly four unique algorithms")
    if prefix_algorithm not in portfolio:
        raise ValueError("prefix algorithm must belong to the action portfolio")
    outcomes = []
    for target_algorithm in portfolio:
        started = perf_counter()
        if target_algorithm == prefix_algorithm:
            action = "continue_current"
            state = clone_optimizer_state(checkpoint_state)
            transition_mode = "native_optimizer_state"
        else:
            action = target_algorithm
            state = initialize_transferred_optimizer_state(
                algorithm=target_algorithm,
                source_state=checkpoint_state,
                problem=problem,
                seed=seed,
                function=function,
                instance=instance,
                event=QUERY_TRANSFER_EVENT,
            )
            transition_mode = "population_transfer_initialization"
        result = advance_optimizer_state(state=state, problem=problem, fe_budget=fe_budget)
        outcomes.append(
            {
                "action": action,
                "target_algorithm": target_algorithm,
                "transition_mode": transition_mode,
                "action_loss": float(result.best_fitness),
                "runtime_action_optimization": float(perf_counter() - started),
            }
        )
    _complete_action_diagnostics(outcomes, prefix_algorithm=prefix_algorithm, portfolio=portfolio)
    return outcomes


def _evaluate_shard(
    *,
    split: str,
    suite: str,
    config: dict,
    trajectory_rows: list[dict],
    sample_design_id: str,
    settings: OptimizerSettings,
    problem_cache: dict[tuple[int, int, int], Problem],
    portfolio: tuple[str, ...],
    default_algorithm: str,
    all_prefixes: bool,
    max_states: int | None,
) -> tuple[list[dict], int]:
    if max_states is not None and max_states <= 0:
        return [], 0
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for trajectory_row in trajectory_rows:
        key = (
            str(trajectory_row["problem_id"]),
            str(trajectory_row["algorithm"]),
            int(trajectory_row["seed"]),
        )
        grouped[key].append(trajectory_row)

    output_rows = []
    used_states = 0
    sample_design = get_sample_design_spec(sample_design_id)
    for (problem_id, prefix_algorithm, seed), trajectory_group in grouped.items():
        if not all_prefixes and prefix_algorithm != default_algorithm:
            continue
        function, instance, dimension = _parse_problem_id(problem_id, suite=suite)
        problem_key = (function, instance, dimension)
        if problem_key not in problem_cache:
            problem_cache[problem_key] = make_problem(
                {"suite": suite, "function": function, "instance": instance, "dimension": dimension}
            )
        problem = problem_cache[problem_key]
        state = initialize_optimizer_state(algorithm=prefix_algorithm, problem=problem, seed=seed, settings=settings)
        for trajectory_row in sorted(trajectory_group, key=lambda row: int(row["FE"])):
            checkpoint_fe = int(trajectory_row["FE"])
            delta = checkpoint_fe - int(state.evaluations)
            if delta < 0:
                raise ValueError(f"trajectory checkpoint FE moved backwards for {problem_id} {prefix_algorithm} seed={seed}")
            advance_optimizer_state(state=state, problem=problem, fe_budget=delta)
            _validate_replayed_checkpoint(state, trajectory_row)
            if not _eligible_for_action_loss(trajectory_row, config, sample_design.sample_design_id):
                continue
            fe_total = fe_total_for_dimension(config, dimension)
            fe_query = sample_design.sample_size(dimension)
            if fe_query != int(round(sample_design.fe_ratio * fe_total)):
                raise ValueError("query sample budget does not match FE_total")
            remaining_budget = fe_total - checkpoint_fe - fe_query
            skip_budget = fe_total - checkpoint_fe
            skip_started = perf_counter()
            if default_algorithm == prefix_algorithm:
                skip_state = clone_optimizer_state(state)
                no_query_transition_mode = "native_optimizer_state"
            else:
                skip_state = initialize_transferred_optimizer_state(
                    algorithm=default_algorithm,
                    source_state=state,
                    problem=problem,
                    seed=seed,
                    function=function,
                    instance=instance,
                    event=NO_QUERY_TRANSFER_EVENT,
                )
                no_query_transition_mode = "population_transfer_initialization"
            skip_result = advance_optimizer_state(state=skip_state, problem=problem, fe_budget=skip_budget)
            runtime_skip = perf_counter() - skip_started
            outcomes = evaluate_candidate_actions(
                checkpoint_state=state,
                problem=problem,
                portfolio=portfolio,
                fe_budget=remaining_budget,
                seed=seed,
                function=function,
                instance=instance,
            )
            common = {
                "split": split,
                "problem_id": problem_id,
                "family": str(trajectory_row["family"]),
                "dimension": dimension,
                "prefix_algorithm": prefix_algorithm,
                "default_algorithm": default_algorithm,
                "no_query_algorithm": default_algorithm,
                "seed": seed,
                "FE": checkpoint_fe,
                "FE_ratio": float(trajectory_row["FE_ratio"]),
                "FE_total": fe_total,
                "sample_design_id": sample_design.sample_design_id,
                "sample_design_protocol": sample_design.protocol,
                "FE_query": fe_query,
                "FE_no_query_optimization": skip_budget,
                "FE_query_optimization": remaining_budget,
                "remaining_budget_ratio": float(remaining_budget / fe_total),
                "p_skip": float(skip_result.best_fitness),
                "runtime_no_query_optimization": float(runtime_skip),
                "no_query_transition_mode": no_query_transition_mode,
                "action_loss_protocol": ACTION_LOSS_PROTOCOL,
            }
            output_rows.extend({**common, **outcome} for outcome in outcomes)
            used_states += 1
            if max_states is not None and used_states >= max_states:
                return output_rows, used_states
    return output_rows, used_states


def _complete_action_diagnostics(
    outcomes: list[dict[str, float | str]],
    *,
    prefix_algorithm: str,
    portfolio: tuple[str, ...],
) -> None:
    targets = [str(row["target_algorithm"]) for row in outcomes]
    if len(outcomes) != len(portfolio) or set(targets) != set(portfolio):
        raise ValueError("each shared state must contain exactly one outcome per portfolio algorithm")
    native = [row for row in outcomes if row["transition_mode"] == "native_optimizer_state"]
    if len(native) != 1 or native[0]["target_algorithm"] != prefix_algorithm:
        raise ValueError("each shared state must contain exactly one native continue-current action")
    ordered = sorted(outcomes, key=lambda row: (float(row["action_loss"]), str(row["target_algorithm"])))
    best_algorithm = str(ordered[0]["target_algorithm"])
    best_loss = float(ordered[0]["action_loss"])
    worst_loss = max(float(row["action_loss"]) for row in outcomes)
    scale = max(worst_loss - best_loss, EPS)
    for row in outcomes:
        row["best_observed_algorithm"] = best_algorithm
        row["best_observed_loss"] = best_loss
        row["action_loss_norm"] = float((float(row["action_loss"]) - best_loss) / scale)


def _eligible_for_action_loss(row: dict, config: dict, sample_design_id: str) -> bool:
    fe_total = fe_total_for_dimension(config, int(row["dimension"]))
    fe_prefix = int(row["FE"])
    fe_query = get_sample_design_spec(sample_design_id).sample_size(int(row["dimension"]))
    ratio = float(row["FE_ratio"])
    return ratio >= MIN_LABEL_RATIO and ratio < 1.0 and fe_prefix + fe_query < fe_total


def _validate_replayed_checkpoint(state: OptimizerState, trajectory_row: dict) -> None:
    if not np.array_equal(state.population, np.asarray(trajectory_row["population"], dtype=float)):
        raise ValueError("trajectory population does not match replayed native optimizer state; regenerate trajectories")
    if not np.array_equal(state.fitness, np.asarray(trajectory_row["fitness"], dtype=float)):
        raise ValueError("trajectory fitness does not match replayed native optimizer state; regenerate trajectories")
    if float(state.best_fitness) != float(trajectory_row["best_fitness"]):
        raise ValueError("trajectory best_fitness does not match replayed native optimizer state; regenerate trajectories")
    if int(state.generation) != int(trajectory_row["native_updates"]):
        raise ValueError("trajectory native_updates does not match replayed optimizer generation; regenerate trajectories")


def _parse_problem_id(problem_id: str, *, suite: str) -> tuple[int, int, int]:
    suite_name = str(suite).lower()
    if suite_name == "bbob":
        match = re.match(r"^bbob_f(\d{3})_i(\d+)_d(\d+)$", problem_id)
        if match is None:
            raise ValueError(f"invalid BBOB problem_id: {problem_id}")
        return tuple(int(value) for value in match.groups())
    if suite_name in {"cec2017", "cec2022"}:
        match = re.match(rf"^{suite_name}_f(\d{{2}})_d(\d+)$", problem_id)
        if match is None:
            raise ValueError(f"invalid {suite_name.upper()} problem_id: {problem_id}")
        function, dimension = (int(value) for value in match.groups())
        return function, 1, dimension
    raise ValueError(f"unsupported benchmark suite for state-action loss generation: {suite}")


def _schema() -> pa.Schema:
    fields = [
        ("split", pa.string()),
        ("problem_id", pa.string()),
        ("family", pa.string()),
        ("dimension", pa.int32()),
        ("prefix_algorithm", pa.string()),
        ("default_algorithm", pa.string()),
        ("no_query_algorithm", pa.string()),
        ("seed", pa.int64()),
        ("FE", pa.int64()),
        ("FE_ratio", pa.float64()),
        ("FE_total", pa.int64()),
        ("sample_design_id", pa.string()),
        ("sample_design_protocol", pa.string()),
        ("FE_query", pa.int64()),
        ("FE_no_query_optimization", pa.int64()),
        ("FE_query_optimization", pa.int64()),
        ("remaining_budget_ratio", pa.float64()),
        ("p_skip", pa.float64()),
        ("runtime_no_query_optimization", pa.float64()),
        ("no_query_transition_mode", pa.string()),
        ("action", pa.string()),
        ("target_algorithm", pa.string()),
        ("transition_mode", pa.string()),
        ("action_loss", pa.float64()),
        ("action_loss_norm", pa.float64()),
        ("runtime_action_optimization", pa.float64()),
        ("best_observed_algorithm", pa.string()),
        ("best_observed_loss", pa.float64()),
        ("action_loss_protocol", pa.string()),
    ]
    return pa.schema(fields)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate every portfolio action from each shared optimizer state."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, default=Path("configs/phase1_bbob_train.yaml"))
    parser.add_argument("--sample-design-id", choices=sorted(SAMPLE_DESIGN_SPECS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    parser.add_argument(
        "--default-algorithm",
        choices=("de", "pso", "cmaes", "shade"),
        default=None,
        help="Scoped consistency runs only; formal generation omits this and derives SBS from --train-config.",
    )
    parser.add_argument("--all-prefixes", action="store_true")
    parser.add_argument("--max-states", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate_state_action_losses(
        config_path=args.config,
        train_config_path=args.train_config,
        sample_design_id=args.sample_design_id,
        output_path=args.output,
        only_functions=args.only_function,
        only_dimensions=args.only_dimension,
        default_algorithm=args.default_algorithm,
        all_prefixes=args.all_prefixes,
        max_states=args.max_states,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
