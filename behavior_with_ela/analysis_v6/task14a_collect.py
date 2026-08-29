"""Task 14A: post-handoff sequential action-space and reset-confound collection.

For every (problem, seed, source A in {SHADE, L-SHADE, CSO}, checkpoint
FE in {2000, 4000, 6000}) of the Task 12 stage-2 development set this script

1. replays the natural source trajectory 0 -> 6000 FE with the formal global
   window recorder (deterministic, identical semantics to Task 12/13);
2. executes the real population-transfer handoff A -> B for the two targets
   B != A (all six directions kept), forces a 1000-FE commitment and saves
   the mature post-handoff checkpoint together with global and segment
   behavior records (segment recorder restarts at the handoff);
3. forks the next-action set {continue B, switch A, switch C} for 1000 FE
   each, with an outcome-blind 10 percent x R=3 repetition plan;
4. runs the mandatory population-preserving reset controls on every
   current=SHADE and current=L-SHADE checkpoint (population, fitness, best,
   evaluations and the L-SHADE reduction schedule are preserved; success
   history, archive and adaptive memory are reset; fresh semantic RNG event).

The reset branches are ledgered separately from the normal action branches.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import resource
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from behavior.features import extract_behavior_rows
from behavior_with_ela.action_dataset import (
    CONTINUATION_REPETITION_STREAM_OFFSET,
    NO_QUERY_TRANSFER_EVENT,
    TRANSFER_REPETITION_EVENT_OFFSET,
)
from behavior_with_ela.analysis_v5.task13.task13_replay import SAMPLING_METADATA
from behavior_with_ela.protocol import (
    load_experiment_config,
    make_experiment_problem,
    suite_code,
)
from optimizers import (
    OptimizerSettings,
    advance_optimizer_state,
    clone_optimizer_state,
    initialize_optimizer_state,
    initialize_transferred_optimizer_state,
)
from optimizers.seeding import make_event_rng
from optimizers.state import NATIVE_STREAMS
from trajectory.records import TrajectoryRecord
from trajectory.window_statistics import NativeUpdateWindowRecorder

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/behavior_with_ela_train.yaml"
HEAVY = ROOT / "behavior_with_ela/results/analysis_v6/task14a"
SOLVERS = ("shade", "lshade", "cso")
SOURCE_CHECKPOINT_FES = (2000, 4000, 6000)
COMMITMENT_FE = 1000
HORIZON_FE = 1000
SEEDS = (1, 2, 3, 4, 5)
REPETITION_FRACTION = 0.10
REPETITION_STREAM = 2026090222
RESET_STREAM_OFFSET = 90000  # distinct from continuation (30k) / transfer offsets
BBOB_FAMILY_FUNCTIONS = {
    "bbob_separable_f01_f05": (1, 2),
    "bbob_low_or_moderate_conditioning_f06_f09": (6, 7),
    "bbob_high_conditioning_unimodal_f10_f14": (10, 11),
    "bbob_multimodal_adequate_global_structure_f15_f19": (15, 16),
    "bbob_multimodal_weak_global_structure_f20_f24": (20, 21),
}
MA_DEFINITION_STRIDE = 2


def collection_jobs(config) -> list[dict]:
    jobs = []
    for suite in config.suites:
        if suite.split == "bbob_train":
            family_map: dict[str, list[int]] = {}
            for function in suite.functions:
                problem = make_experiment_problem(
                    suite,
                    function=int(function),
                    instance=suite.instances[0],
                    dimension=10,
                    boundary_handling="reflect",
                )
                family_map.setdefault(str(problem.family), []).append(int(function))
                problem.close()
            functions = []
            for label in sorted(family_map):
                functions.extend(sorted(family_map[label])[:2])
            instances = suite.instances
        else:
            functions = sorted(suite.functions)[::MA_DEFINITION_STRIDE]
            instances = suite.instances
        for function in functions:
            for instance in instances:
                jobs.append(
                    {
                        "split": suite.split,
                        "suite": suite.suite,
                        "function": int(function),
                        "instance": int(instance),
                        "seeds": tuple(SEEDS),
                    }
                )
    return jobs


def _sampled_for_repetition(
    *, suite_name: str, function: int, instance: int, seed: int, current: str, action: str, fe: int
) -> bool:
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                REPETITION_STREAM,
                suite_code(suite_name),
                int(function),
                int(instance),
                int(seed),
                NATIVE_STREAMS.get(current, 0),
                NATIVE_STREAMS.get(action, 0),
                int(fe),
            ]
        )
    )
    return bool(rng.random() < REPETITION_FRACTION)


def _reset_solver_state(state, fresh_rng_state: dict):
    """Population-preserving reset for SHADE / L-SHADE: keep population,
    fitness, best, evaluations and the reduction schedule; reset success
    history, archive and adaptive memory; install a fresh RNG state and clear
    any partial-generation work."""
    dimension = state.population.shape[1]
    return dataclasses.replace(
        state,
        memory_f=np.full(np.shape(state.memory_f), 0.5, dtype=float),
        memory_cr=np.full(np.shape(state.memory_cr), 0.5, dtype=float),
        archive=np.empty((0, dimension), dtype=float),
        memory_index=0,
        rng_state=copy.deepcopy(fresh_rng_state),
        pending_population=None,
        pending_fitness=None,
        pending_f=None,
        pending_cr=None,
        pending_index=0,
    )


def _behavior_rows(record: TrajectoryRecord) -> list[dict]:
    return extract_behavior_rows([dataclasses.asdict(record)])


def _collect_unit(job: dict) -> dict:
    config = load_experiment_config(CONFIG)
    suites = {suite.split: suite for suite in config.suites}
    suite = suites[job["split"]]
    problem = make_experiment_problem(
        suite,
        function=job["function"],
        instance=job["instance"],
        dimension=config.dimension,
        boundary_handling=config.boundary_handling,
    )
    settings = OptimizerSettings(
        population_size=config.population_size,
        boundary_handling=config.boundary_handling,
    )
    state_rows, branch_rows, reset_rows = [], [], []
    behavior_rows = []
    fe_natural = 0
    started = perf_counter()

    for seed in job["seeds"]:
        for source in SOLVERS:
            evaluations = 0
            global_updates = 0
            global_window = NativeUpdateWindowRecorder()

            def observe_evaluation(point: np.ndarray, value: float) -> None:
                nonlocal evaluations
                evaluations += 1

            def observe_global(updated) -> None:
                nonlocal global_updates
                global_updates += 1
                global_window.observe(
                    fe=evaluations,
                    native_updates=global_updates,
                    population=updated.population,
                    fitness=updated.fitness,
                    best_fitness=updated.best_fitness,
                )

            state = initialize_optimizer_state(
                algorithm=source,
                problem=problem,
                seed=seed,
                settings=settings,
                on_evaluation=observe_evaluation,
            )
            global_window.observe(
                fe=evaluations,
                native_updates=0,
                population=state.population,
                fitness=state.fitness,
                best_fitness=state.best_fitness,
            )
            checkpoint_windows: dict[int, NativeUpdateWindowRecorder] = {}
            checkpoint_updates: dict[int, int] = {}
            checkpoint_states: dict[int, object] = {}
            pending = list(SOURCE_CHECKPOINT_FES)
            while evaluations < max(SOURCE_CHECKPOINT_FES):
                advance_optimizer_state(
                    state=state,
                    problem=problem,
                    fe_budget=min(config.population_size, max(SOURCE_CHECKPOINT_FES) - evaluations),
                    on_native_update=observe_global,
                    on_evaluation=observe_evaluation,
                )
                if pending and evaluations == pending[0]:
                    fe = pending.pop(0)
                    checkpoint_states[fe] = clone_optimizer_state(state)
                    checkpoint_windows[fe] = copy.deepcopy(global_window)
                    checkpoint_updates[fe] = global_updates
            fe_natural += max(SOURCE_CHECKPOINT_FES)

            for target in [s for s in SOLVERS if s != source]:
                for source_fe in SOURCE_CHECKPOINT_FES:
                    source_state = checkpoint_states[source_fe]
                    source_window = checkpoint_windows[source_fe]
                    transferred = initialize_transferred_optimizer_state(
                        algorithm=target,
                        source_state=source_state,
                        problem=problem,
                        seed=seed,
                        function=problem.function_number,
                        instance=problem.instance_number,
                        event=NO_QUERY_TRANSFER_EVENT,
                    )
                    # each direction gets its own copy of the global history so
                    # that commitments never interleave between directions
                    global_window_dir = copy.deepcopy(source_window)
                    dir_updates = checkpoint_updates[source_fe]
                    segment_window = NativeUpdateWindowRecorder()
                    segment_updates = 1
                    segment_window.observe(
                        fe=source_fe,
                        native_updates=segment_updates,
                        population=transferred.population,
                        fitness=transferred.fitness,
                        best_fitness=transferred.best_fitness,
                    )
                    commitment_used = 0

                    def observe_commitment(updated) -> None:
                        nonlocal dir_updates, segment_updates
                        commitment_used_now = commitment_used[0]
                        fe_now = source_fe + commitment_used_now
                        dir_updates += 1
                        segment_updates += 1
                        global_window_dir.observe(
                            fe=fe_now,
                            native_updates=dir_updates,
                            population=updated.population,
                            fitness=updated.fitness,
                            best_fitness=updated.best_fitness,
                        )
                        segment_window.observe(
                            fe=fe_now,
                            native_updates=segment_updates,
                            population=updated.population,
                            fitness=updated.fitness,
                            best_fitness=updated.best_fitness,
                        )

                    def observe_commitment_evals(point: np.ndarray, value: float) -> None:
                        nonlocal commitment_used
                        commitment_used[0] += 1

                    commitment_used = [0]
                    advance_optimizer_state(
                        state=transferred,
                        problem=problem,
                        fe_budget=COMMITMENT_FE,
                        on_native_update=observe_commitment,
                        on_evaluation=observe_commitment_evals,
                    )
                    post = clone_optimizer_state(transferred)
                    post_fe = source_fe + COMMITMENT_FE
                    gap = float(
                        min(
                            max(post.best_fitness - float(problem.reference_value), 0.0),
                            config.failure_loss_cap,
                        )
                    )
                    log10_gap = float(
                        np.log10(np.clip(gap, config.log10_gap_floor, config.log10_gap_cap))
                    )
                    route = f"{source}->{target}"
                    snapshot = global_window_dir.current_snapshot

                    windows_g, history_g = global_window_dir.build(
                        fe_total=config.fe_total,
                        problem_id=problem.problem_id,
                        algorithm=target,
                    )
                    # segment-relative windows: w02/w05/w10 are fractions of
                    # the 1000-FE segment age, so every anchor lies inside the
                    # segment (global-budget windows would reach past the
                    # segment start at exactly one-segment age)
                    windows_s, history_s = segment_window.build(
                        fe_total=COMMITMENT_FE,
                        problem_id=problem.problem_id,
                        algorithm=target,
                    )
                    record_global = TrajectoryRecord.from_arrays(
                        problem_id=problem.problem_id,
                        function_id=problem.function_id,
                        family=problem.family,
                        cv_group_id=problem.cv_group_id,
                        dimension=config.dimension,
                        algorithm=target,
                        seed=int(seed),
                        fe=snapshot.fe,
                        fe_total=config.fe_total,
                        native_updates=snapshot.native_updates,
                        window_statistics=windows_g,
                        native_update_history=history_g,
                        population=snapshot.population,
                        fitness=snapshot.fitness,
                        best_fitness=snapshot.best_fitness,
                        sampling_metadata=dict(SAMPLING_METADATA),
                    )
                    segment_snapshot = segment_window.current_snapshot
                    record_segment = TrajectoryRecord.from_arrays(
                        problem_id=problem.problem_id,
                        function_id=problem.function_id,
                        family=problem.family,
                        cv_group_id=problem.cv_group_id,
                        dimension=config.dimension,
                        algorithm=target,
                        seed=int(seed),
                        fe=segment_snapshot.fe,
                        fe_total=config.fe_total,
                        native_updates=segment_snapshot.native_updates,
                        window_statistics=windows_s,
                        native_update_history=history_s,
                        population=segment_snapshot.population,
                        fitness=segment_snapshot.fitness,
                        best_fitness=segment_snapshot.best_fitness,
                        sampling_metadata=dict(SAMPLING_METADATA),
                    )
                    for kind, extracted in (
                        ("global", _behavior_rows(record_global)),
                        ("segment", _behavior_rows(record_segment)),
                    ):
                        for row in extracted:
                            row["behavior_kind"] = kind
                            row["state_id"] = (
                                f"{suite.split}:{problem.problem_id}:{route}:"
                                f"seed{int(seed)}:src_fe{int(source_fe)}"
                            )
                            row["segment_start"] = int(source_fe)
                            behavior_rows.append(row)

                    state_rows.append(
                        {
                            "state_id": (
                                f"{suite.split}:{problem.problem_id}:{route}:"
                                f"seed{int(seed)}:src_fe{int(source_fe)}"
                            ),
                            "suite": suite.suite,
                            "split": suite.split,
                            "problem_id": problem.problem_id,
                            "family": problem.family,
                            "cv_group_id": problem.cv_group_id,
                            "instance": int(job["instance"]),
                            "seed": int(seed),
                            "source_algorithm": source,
                            "current_algorithm": target,
                            "source_checkpoint_fe": int(source_fe),
                            "FE": int(post_fe),
                            "segment_start": int(source_fe),
                            "segment_age": int(post_fe - source_fe),
                            "best_fitness": float(post.best_fitness),
                            "log10_gap": log10_gap,
                            "population_size": int(post.population.shape[0]),
                            "snapshot_fe": int(snapshot.fe),
                            "snapshot_population_size": int(snapshot.population.shape[0]),
                        }
                    )

                    actions = ["continue", source, *[s for s in SOLVERS if s not in (source, target)]]
                    for action in actions:
                        sampled = _sampled_for_repetition(
                            suite_name=suite.suite,
                            function=problem.function_number,
                            instance=problem.instance_number,
                            seed=seed,
                            current=target,
                            action=action,
                            fe=int(post_fe),
                        )
                        for replicate_id in (range(3) if sampled else range(1)):
                            tracker_best = float(post.best_fitness)
                            branch_evals = 0
                            if action == "continue":
                                branch_state = clone_optimizer_state(post)
                                if replicate_id > 0:
                                    rng = make_event_rng(
                                        seed=seed,
                                        stream_code=(
                                            CONTINUATION_REPETITION_STREAM_OFFSET
                                            + NATIVE_STREAMS[target]
                                        ),
                                        suite_code=problem.suite_code,
                                        function=problem.function_number,
                                        instance=problem.instance_number,
                                        dimension=problem.dimension,
                                        generation=int(post.generation),
                                        event=int(replicate_id),
                                    )
                                    branch_state.rng_state = copy.deepcopy(rng.bit_generator.state)
                            else:
                                event = (
                                    NO_QUERY_TRANSFER_EVENT
                                    if replicate_id == 0
                                    else TRANSFER_REPETITION_EVENT_OFFSET + int(replicate_id)
                                )
                                branch_state = initialize_transferred_optimizer_state(
                                    algorithm=action,
                                    source_state=post,
                                    problem=problem,
                                    seed=seed,
                                    function=problem.function_number,
                                    instance=problem.instance_number,
                                    event=event,
                                )

                            def observe_branch(point: np.ndarray, value: float) -> None:
                                nonlocal branch_evals, tracker_best
                                branch_evals += 1
                                if value < tracker_best:
                                    tracker_best = float(value)

                            advance_optimizer_state(
                                state=branch_state,
                                problem=problem,
                                fe_budget=HORIZON_FE,
                                on_evaluation=observe_branch,
                            )
                            if branch_evals != HORIZON_FE:
                                raise RuntimeError("branch evaluations mismatch")
                            branch_gap = float(
                                min(
                                    max(tracker_best - float(problem.reference_value), 0.0),
                                    config.failure_loss_cap,
                                )
                            )
                            branch_rows.append(
                                {
                                    "state_id": state_rows[-1]["state_id"],
                                    "route": route,
                                    "source_algorithm": source,
                                    "current_algorithm": target,
                                    "source_checkpoint_fe": int(source_fe),
                                    "FE": int(post_fe),
                                    "candidate_action": action,
                                    "replicate_id": int(replicate_id),
                                    "loss_1000": float(
                                        np.log10(
                                            np.clip(
                                                branch_gap, config.log10_gap_floor, config.log10_gap_cap
                                            )
                                        )
                                    ),
                                }
                            )

                    # mandatory reset controls (current = SHADE or L-SHADE)
                    if target in ("shade", "lshade"):
                        rng = make_event_rng(
                            seed=seed,
                            stream_code=NATIVE_STREAMS[target] + RESET_STREAM_OFFSET,
                            suite_code=problem.suite_code,
                            function=problem.function_number,
                            instance=problem.instance_number,
                            dimension=problem.dimension,
                            generation=int(post.generation),
                            event=NO_QUERY_TRANSFER_EVENT,
                        )
                        reset_state = _reset_solver_state(post, rng.bit_generator.state)
                        reset_best = float(post.best_fitness)
                        reset_evals = 0

                        def observe_reset(point: np.ndarray, value: float) -> None:
                            nonlocal reset_evals, reset_best
                            reset_evals += 1
                            if value < reset_best:
                                reset_best = float(value)

                        advance_optimizer_state(
                            state=reset_state,
                            problem=problem,
                            fe_budget=HORIZON_FE,
                            on_evaluation=observe_reset,
                        )
                        if reset_evals != HORIZON_FE:
                            raise RuntimeError("reset evaluations mismatch")
                        reset_gap = float(
                            min(
                                max(reset_best - float(problem.reference_value), 0.0),
                                config.failure_loss_cap,
                            )
                        )
                        reset_rows.append(
                            {
                                "state_id": state_rows[-1]["state_id"],
                                "route": route,
                                "source_algorithm": source,
                                "current_algorithm": target,
                                "source_checkpoint_fe": int(source_fe),
                                "FE": int(post_fe),
                                "control": "reset_current",
                                "loss_1000": float(
                                    np.log10(
                                        np.clip(reset_gap, config.log10_gap_floor, config.log10_gap_cap)
                                    )
                                ),
                                "population_size_preserved": int(post.population.shape[0]),
                                "schedule_max_evaluations": int(getattr(post, "max_evaluations", -1)),
                                "reduction_max_fe": int(getattr(post, "reduction_max_fe", -1)),
                            }
                        )
    problem.close()
    return {
        "job": job,
        "states": state_rows,
        "branches": branch_rows,
        "resets": reset_rows,
        "behavior": behavior_rows,
        "fe_natural": fe_natural,
        "wall_seconds": perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    HEAVY.mkdir(parents=True, exist_ok=True)
    config = load_experiment_config(CONFIG)
    jobs = collection_jobs(config)
    print(f"[task14a] units: {len(jobs)}", flush=True)

    states, branches, resets, behavior = [], [], [], []
    fe_natural_total = 0
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_collect_unit, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            states.extend(result["states"])
            branches.extend(result["branches"])
            resets.extend(result["resets"])
            behavior.extend(result["behavior"])
            fe_natural_total += result["fe_natural"]
            if index % 5 == 0 or index == len(jobs):
                print(
                    f"[task14a] {index}/{len(jobs)} units (states {len(states)})", flush=True
                )

    states_df = pd.DataFrame(states)
    branches_df = pd.DataFrame(branches)
    resets_df = pd.DataFrame(resets)
    behavior_df = pd.DataFrame(behavior)
    states_df.to_parquet(HEAVY / "post_handoff_states.parquet", index=False)
    branches_df.to_parquet(HEAVY / "post_handoff_action_outcomes_1000.parquet", index=False)
    resets_df.to_parquet(HEAVY / "reset_control_outcomes.parquet", index=False)
    behavior_df.to_parquet(HEAVY / "post_handoff_behavior.parquet", index=False)

    commitment_fe = len(states_df) * COMMITMENT_FE
    fork_fe = int(branches_df["loss_1000"].shape[0]) * HORIZON_FE
    reset_fe = int(resets_df.shape[0]) * HORIZON_FE
    ledger = pd.DataFrame(
        [
            {"phase": "source_natural", "fe": int(fe_natural_total)},
            {"phase": "handoff_commitment", "fe": int(commitment_fe)},
            {"phase": "next_action_branch", "fe": int(fork_fe)},
            {"phase": "reset_control", "fe": int(reset_fe)},
            {
                "phase": "total",
                "fe": int(fe_natural_total + commitment_fe + fork_fe + reset_fe),
            },
        ]
    )
    ledger["wall_seconds_total"] = perf_counter() - started
    ledger["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    ledger.to_parquet(HEAVY / "task14a_collection_ledger.parquet", index=False)
    print(ledger.to_string())


if __name__ == "__main__":
    main()
