"""Mature post-handoff action-space x action-horizon audit (protocol v1).

Collects mature post-handoff decision checkpoints on BBOB-train + selected
MA-BBOB (10D) under three standardized routes:

  R0_native_cmaes   cmaes from FE=0 (native control, handoff_performed=False)
  R1_pso_to_cmaes   pso 0->2000, population transfer, cmaes thereafter
  R2_shade_to_cmaes shade 0->2000, population transfer, cmaes thereafter

Checkpoints at FE in {3000, 4000, 5000, 6000} (dwell = FE - 2000 on R1/R2).
Every checkpoint forks three action branches {continue cmaes, switch pso,
switch shade}; each branch runs continuously and records its best-so-far at
+500 FE, +1000 FE and terminal (first 40-FE update boundary >= mark for the
two short horizons). A deterministic SeedSequence-based 10% subset of
checkpoints repeats all three actions with R=3 using the formal repetition RNG
fork semantics. Global and segment Behavior are extracted with the formal
extractor from independent NativeUpdateWindowRecorder histories (segment
windows restart at the handoff).

No selector is trained here; the outputs are label tables for the audit.
"""
from __future__ import annotations

import argparse
import resource
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from behavior.features import (
    SELECTOR_BEHAVIOR_FEATURE_COLUMNS,
    extract_behavior_rows,
)
from behavior_with_ela.action_dataset import (
    CONTINUATION_REPETITION_STREAM_OFFSET,
    TRANSFER_REPETITION_EVENT_OFFSET,
)
from behavior_with_ela.protocol import (
    load_experiment_config,
    make_experiment_problem,
    suite_code,
)
from optimizers import (
    NO_QUERY_TRANSFER_EVENT,
    OptimizerSettings,
    advance_optimizer_state,
    clone_optimizer_state,
    initialize_optimizer_state,
    initialize_transferred_optimizer_state,
)
from optimizers.seeding import make_event_rng
from optimizers.state import NATIVE_STREAMS
from trajectory.sampling import SAMPLING_METADATA_COLUMNS
from trajectory.window_statistics import NativeUpdateWindowRecorder


PROTOCOL = "mature_post_handoff_action_horizon_audit_v1"
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/behavior_with_ela_train.yaml"
OUTPUT = ROOT / "behavior_with_ela/results/post_handoff/task11"
CHECKPOINT_FES = (3000, 4000, 5000, 6000)
HANDOFF_FE = 2000
HORIZON_MARKS = (500, 1000)
SEEDS = (1, 2, 3, 4, 5)
REPETITION_FRACTION = 0.10
REPETITION_STREAM = 2026083001
ROUTES = {
    "R0_native_cmaes": {"start": "cmaes", "source": "cmaes", "route_code": 1, "native": True},
    "R1_pso_to_cmaes": {"start": "pso", "source": "pso", "route_code": 2, "native": False},
    "R2_shade_to_cmaes": {"start": "shade", "source": "shade", "route_code": 3, "native": False},
}
ACTIONS = ("continue", "pso", "shade")


def _log10_loss(gap: float, config) -> float:
    return float(
        np.log10(np.clip(gap, config.log10_gap_floor, config.log10_gap_cap))
    )


def _gap(best_fitness: float, reference: float, config) -> float:
    return float(min(max(best_fitness - reference, 0.0), config.failure_loss_cap))


def _audit_metadata(config, fe: int) -> dict:
    values = {
        "sampling_protocol": "mature_post_handoff_audit_checkpoint_v1",
        "sampling_phase": "audit",
        "sampling_triggers": [],
        "is_budget_milestone": False,
        "budget_milestone_ratio": None,
        "is_event_sample": False,
        "monitor_target_ratio": float(fe) / float(config.fe_total),
        "event_index_in_phase": None,
        "event_improvement_resume": False,
        "event_stagnation_onset": False,
        "event_rank_change": False,
        "event_elite_migration": False,
        "event_diversity_recovery": False,
        "event_improvement_resume_metric": 0.0,
        "event_stagnation_onset_metric": 0.0,
        "event_rank_change_metric": 0.0,
        "event_elite_migration_metric": 0.0,
        "event_diversity_recovery_metric": 0.0,
    }
    missing = set(SAMPLING_METADATA_COLUMNS).difference(values)
    if missing:
        raise RuntimeError(f"audit metadata missing columns: {sorted(missing)}")
    return values


def _behavior_record(
    *,
    problem,
    algorithm: str,
    seed: int,
    fe: int,
    config,
    native_updates: int,
    window_statistics: list[dict],
    native_update_history: list[dict],
    population: np.ndarray,
    fitness: np.ndarray,
    best_fitness: float,
) -> dict:
    row = {
        "problem_id": problem.problem_id,
        "function_id": problem.function_id,
        "family": problem.family,
        "cv_group_id": problem.cv_group_id,
        "dimension": problem.dimension,
        "algorithm": algorithm,
        "seed": int(seed),
        "FE": int(fe),
        "FE_ratio": float(fe) / float(config.fe_total),
        "FE_total": int(config.fe_total),
        "native_updates": int(native_updates),
        "window_statistics": window_statistics,
        "native_update_history": native_update_history,
        "population": np.asarray(population, dtype=float),
        "fitness": np.asarray(fitness, dtype=float).reshape(-1),
        "best_fitness": float(best_fitness),
    }
    row.update(_audit_metadata(config, fe))
    return row


class _HorizonBranchTracker:
    """Tracks branch evaluations and best-so-far at the horizon marks."""

    def __init__(self, *, initial_best: float, reference: float, config) -> None:
        self.evaluations = 0
        self.best = float(initial_best)
        self.reference = float(reference)
        self.config = config
        self.marks = list(HORIZON_MARKS)
        self.recorded: dict[int, float] = {}

    def on_evaluation(self, point: np.ndarray, value: float) -> None:
        self.evaluations += 1
        if value < self.best:
            self.best = float(value)

    def on_native_update(self, updated) -> None:
        while self.marks and self.evaluations >= self.marks[0]:
            mark = self.marks.pop(0)
            self.recorded[mark] = _gap(self.best, self.reference, self.config)

    def losses(self, config) -> dict[int, float]:
        terminal_gap = _gap(self.best, self.reference, self.config)
        values = {mark: self.recorded.get(mark) for mark in HORIZON_MARKS}
        values["terminal"] = terminal_gap
        return {
            mark: _log10_loss(gap, config) for mark, gap in values.items()
        }


def _run_branch(
    *,
    checkpoint_state,
    action: str,
    replicate_id: int,
    problem,
    seed: int,
    function: int,
    instance: int,
    config,
) -> tuple[dict[int, float], int]:
    tracker = _HorizonBranchTracker(
        initial_best=checkpoint_state.best_fitness,
        reference=problem.reference_value,
        config=config,
    )
    if action == "continue":
        branch_state = clone_optimizer_state(checkpoint_state)
        if replicate_id > 0:
            rng = make_event_rng(
                seed=seed,
                stream_code=(
                    CONTINUATION_REPETITION_STREAM_OFFSET
                    + NATIVE_STREAMS[checkpoint_state.algorithm]
                ),
                suite_code=problem.suite_code,
                function=function,
                instance=instance,
                dimension=problem.dimension,
                generation=int(checkpoint_state.generation),
                event=int(replicate_id),
            )
            import copy

            branch_state.rng_state = copy.deepcopy(rng.bit_generator.state)
    else:
        event = (
            NO_QUERY_TRANSFER_EVENT
            if replicate_id == 0
            else TRANSFER_REPETITION_EVENT_OFFSET + int(replicate_id)
        )
        branch_state = initialize_transferred_optimizer_state(
            algorithm=action,
            source_state=checkpoint_state,
            problem=problem,
            seed=seed,
            function=function,
            instance=instance,
            event=event,
        )
    remaining = int(config.fe_total) - int(checkpoint_state.evaluations)
    if remaining <= 0:
        raise ValueError("checkpoint leaves no terminal budget")
    advance_optimizer_state(
        state=branch_state,
        problem=problem,
        fe_budget=remaining,
        on_native_update=tracker.on_native_update,
        on_evaluation=tracker.on_evaluation,
    )
    if tracker.evaluations != remaining:
        raise RuntimeError(
            f"branch evaluations {tracker.evaluations} != budget {remaining}"
        )
    losses = tracker.losses(config)
    if any(gap is None for gap in losses.values()):
        raise RuntimeError("a horizon mark was not recorded")
    return losses, remaining


def _sampled_for_repetition(
    *, suite_name: str, function: int, instance: int, seed: int, route: str, fe: int
) -> bool:
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                REPETITION_STREAM,
                suite_code(suite_name),
                int(function),
                int(instance),
                int(seed),
                ROUTES[route]["route_code"],
                int(fe),
            ]
        )
    )
    return bool(rng.random() < REPETITION_FRACTION)


def _collect_route(
    *,
    config,
    suite,
    function: int,
    instance: int,
    seed: int,
    route: str,
) -> tuple[list[dict], list[dict], dict]:
    spec = ROUTES[route]
    problem = make_experiment_problem(
        suite,
        function=function,
        instance=instance,
        dimension=config.dimension,
        boundary_handling=config.boundary_handling,
    )
    reference = problem.reference_value
    settings = OptimizerSettings(
        population_size=config.population_size,
        sampling_protocol=config.sampling_protocol,
        boundary_handling=config.boundary_handling,
    )
    evaluations = 0
    global_updates = 0
    segment_updates = 0
    global_window = NativeUpdateWindowRecorder()
    segment_window = NativeUpdateWindowRecorder()
    segment_active = spec["native"]  # R0: segment == whole run
    current_algorithm = spec["start"]

    def observe_evaluation(point: np.ndarray, value: float) -> None:
        nonlocal evaluations
        evaluations += 1

    def observe_windows(updated, fe: int) -> None:
        nonlocal global_updates, segment_updates
        global_updates += 1
        global_window.observe(
            fe=fe,
            native_updates=global_updates,
            population=updated.population,
            fitness=updated.fitness,
            best_fitness=updated.best_fitness,
        )
        if segment_active:
            segment_updates += 1
            segment_window.observe(
                fe=fe,
                native_updates=segment_updates,
                population=updated.population,
                fitness=updated.fitness,
                best_fitness=updated.best_fitness,
            )

    started = perf_counter()
    state = initialize_optimizer_state(
        algorithm=spec["start"],
        problem=problem,
        seed=seed,
        settings=settings,
        on_evaluation=observe_evaluation,
    )
    observe_windows(state, evaluations)
    handoff_done = segment_active
    checkpoint_rows = []
    checkpoint_states = {}
    pending_checkpoints = list(CHECKPOINT_FES)
    route_budget = max(CHECKPOINT_FES)
    while evaluations < route_budget:
        advance_optimizer_state(
            state=state,
            problem=problem,
            fe_budget=min(config.population_size, route_budget - evaluations),
            on_native_update=lambda updated: observe_windows(updated, evaluations),
            on_evaluation=observe_evaluation,
        )
        if not handoff_done and evaluations >= HANDOFF_FE:
            state = initialize_transferred_optimizer_state(
                algorithm="cmaes",
                source_state=state,
                problem=problem,
                seed=seed,
                function=function,
                instance=instance,
                event=NO_QUERY_TRANSFER_EVENT,
            )
            current_algorithm = "cmaes"
            segment_active = True
            segment_updates = 1
            segment_window = NativeUpdateWindowRecorder()
            segment_window.observe(
                fe=evaluations,
                native_updates=segment_updates,
                population=state.population,
                fitness=state.fitness,
                best_fitness=state.best_fitness,
            )
            handoff_done = True
        if pending_checkpoints and evaluations == pending_checkpoints[0]:
            fe = pending_checkpoints.pop(0)
            windows_g, history_g = global_window.build(
                fe_total=config.fe_total,
                problem_id=problem.problem_id,
                algorithm=current_algorithm,
            )
            windows_s, history_s = segment_window.build(
                fe_total=config.fe_total,
                problem_id=problem.problem_id,
                algorithm=current_algorithm,
            )
            checkpoint_states[fe] = clone_optimizer_state(state)
            checkpoint_rows.append(
                {
                    "fe": fe,
                    "windows_g": windows_g,
                    "history_g": history_g,
                    "windows_s": windows_s,
                    "history_s": history_s,
                    "population": np.asarray(state.population, dtype=float).copy(),
                    "fitness": np.asarray(state.fitness, dtype=float).reshape(-1).copy(),
                    "best_fitness": float(state.best_fitness),
                    "global_updates": global_updates,
                    "segment_updates": segment_updates,
                }
            )
    if pending_checkpoints:
        raise RuntimeError(f"route {route} missed checkpoints {pending_checkpoints}")

    records_g = [
        _behavior_record(
            problem=problem,
            algorithm=current_algorithm,
            seed=seed,
            fe=row["fe"],
            config=config,
            native_updates=row["global_updates"],
            window_statistics=row["windows_g"],
            native_update_history=row["history_g"],
            population=row["population"],
            fitness=row["fitness"],
            best_fitness=row["best_fitness"],
        )
        for row in checkpoint_rows
    ]
    records_s = [
        _behavior_record(
            problem=problem,
            algorithm=current_algorithm,
            seed=seed,
            fe=row["fe"],
            config=config,
            native_updates=row["segment_updates"],
            window_statistics=row["windows_s"],
            native_update_history=row["history_s"],
            population=row["population"],
            fitness=row["fitness"],
            best_fitness=row["best_fitness"],
        )
        for row in checkpoint_rows
    ]
    behavior_g = extract_behavior_rows(records_g)
    behavior_s = extract_behavior_rows(records_s)
    behavior_g = sorted(behavior_g, key=lambda row: int(row["FE"]))
    behavior_s = sorted(behavior_s, key=lambda row: int(row["FE"]))

    state_rows = []
    branch_rows = []
    branch_fe = 0
    for index, checkpoint in enumerate(checkpoint_rows):
        fe = int(checkpoint["fe"])
        sampled = _sampled_for_repetition(
            suite_name=suite.suite,
            function=function,
            instance=instance,
            seed=seed,
            route=route,
            fe=fe,
        )
        bg = {f"bg_{name}": behavior_g[index][name] for name in SELECTOR_BEHAVIOR_FEATURE_COLUMNS}
        bs = {f"bs_{name}": behavior_s[index][name] for name in SELECTOR_BEHAVIOR_FEATURE_COLUMNS}
        state_id = (
            f"{suite.split}:{problem.problem_id}:{route}:seed{int(seed)}:fe{fe}"
        )
        state_rows.append(
            {
                "protocol": PROTOCOL,
                "state_id": state_id,
                "suite": suite.suite,
                "split": suite.split,
                "problem_id": problem.problem_id,
                "function_id": problem.function_id,
                "family": problem.family,
                "cv_group_id": problem.cv_group_id,
                "instance": int(instance),
                "seed": int(seed),
                "route": route,
                "source_algorithm": spec["source"],
                "current_algorithm": current_algorithm,
                "handoff_performed": not spec["native"],
                "handoff_FE": None if spec["native"] else HANDOFF_FE,
                "segment_start_FE": 0 if spec["native"] else HANDOFF_FE,
                "FE": fe,
                "FE_ratio": fe / float(config.fe_total),
                "dwell_FE": fe - (0 if spec["native"] else HANDOFF_FE),
                "global_native_updates": int(checkpoint["global_updates"]),
                "segment_native_updates": int(checkpoint["segment_updates"]),
                "current_best_gap": _gap(checkpoint["best_fitness"], reference, config),
                "current_log10_gap": _log10_loss(
                    _gap(checkpoint["best_fitness"], reference, config), config
                ),
                "sampled_for_repetition": sampled,
                **bg,
                **bs,
            }
        )
        replicates = range(3) if sampled else range(1)
        for action in ACTIONS:
            for replicate_id in replicates:
                losses, used_fe = _run_branch(
                    checkpoint_state=checkpoint_states[fe],
                    action=action,
                    replicate_id=replicate_id,
                    problem=problem,
                    seed=seed,
                    function=function,
                    instance=instance,
                    config=config,
                )
                branch_fe += used_fe
                branch_rows.append(
                    {
                        "state_id": state_id,
                        "route": route,
                        "source_algorithm": spec["source"],
                        "current_algorithm": current_algorithm,
                        "FE": fe,
                        "dwell_FE": fe - (0 if spec["start"] == spec["source"] else HANDOFF_FE),
                        "candidate_action": action,
                        "replicate_id": int(replicate_id),
                        "boundary_fe_500": fe + 520,
                        "boundary_fe_1000": fe + 1000,
                        "recorded_fe_terminal": config.fe_total,
                        "loss_500": losses[500],
                        "loss_1000": losses[1000],
                        "loss_terminal": losses["terminal"],
                    }
                )
    elapsed = perf_counter() - started
    problem.close()
    ledger = {
        "suite": suite.suite,
        "function": int(function),
        "instance": int(instance),
        "seed": int(seed),
        "route": route,
        "base_route_fe": route_budget,
        "branch_fe": branch_fe,
        "states": len(state_rows),
        "branches": len(branch_rows),
        "checkpoints": len(checkpoint_rows),
        "wall_seconds": elapsed,
    }
    return state_rows, branch_rows, ledger


def _collect_unit(job: dict) -> dict:
    config = load_experiment_config(CONFIG)
    suites = {suite.split: suite for suite in config.suites}
    suite = suites[job["split"]]
    state_rows, branch_rows = [], []
    ledgers = []
    for seed in job["seeds"]:
        for route in ROUTES:
            states, branches, ledger = _collect_route(
                config=config,
                suite=suite,
                function=job["function"],
                instance=job["instance"],
                seed=seed,
                route=route,
            )
            state_rows.extend(states)
            branch_rows.extend(branches)
            ledgers.append(ledger)
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "job": job,
        "states": state_rows,
        "branches": branch_rows,
        "ledgers": ledgers,
        "max_rss_mb": rss_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--only-suite", action="append", default=None,
        help="restrict to a split name (bbob_train / mabbob_train)",
    )
    parser.add_argument("--only-function", type=int, action="append", default=None)
    args = parser.parse_args()
    config = load_experiment_config(CONFIG)
    output = OUTPUT / "shards"
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for suite in config.suites:
        if args.only_suite and suite.split not in set(args.only_suite):
            continue
        for function in suite.functions:
            if args.only_function and function not in set(args.only_function):
                continue
            for instance in suite.instances:
                jobs.append(
                    {
                        "split": suite.split,
                        "suite": suite.suite,
                        "function": int(function),
                        "instance": int(instance),
                        "seeds": tuple(SEEDS[: args.seeds]),
                    }
                )
    print(f"[post_handoff_audit] units: {len(jobs)}", flush=True)
    total_states = 0
    total_branches = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_collect_unit, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            job = result["job"]
            shard = output / f"{job['split']}_{suite_name(job)}_f{job['function']:03d}_i{job['instance']}"
            shard.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(result["states"]).to_parquet(
                shard / "states.parquet", index=False
            )
            pd.DataFrame(result["branches"]).to_parquet(
                shard / "branches.parquet", index=False
            )
            ledger = pd.DataFrame(result["ledgers"])
            ledger["max_rss_mb"] = result["max_rss_mb"]
            ledger.to_parquet(shard / "ledger.parquet", index=False)
            total_states += len(result["states"])
            total_branches += len(result["branches"])
            print(
                f"[post_handoff_audit] {index}/{len(jobs)} {shard.name}: "
                f"states={len(result['states'])} branches={len(result['branches'])}",
                flush=True,
            )
    print(
        f"[post_handoff_audit] done: states={total_states} branches={total_branches}",
        flush=True,
    )


def suite_name(job: dict) -> str:
    return str(job["suite"])


if __name__ == "__main__":
    main()
