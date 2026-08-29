"""Task 12 Stage 2: dynamic cross-action screening on a pre-registered,
outcome-blind stratified subset, for the Stage-1 KEEP candidates
{shade, lshade, cso} plus an isolated CMA-ES add-back control branch.

Domain (fixed before outcomes): BBOB train - the two lowest-numbered train
functions of each broad family ({f1,f2},{f6,f7},{f10,f11},{f15,f16},{f20,f21})
x instances 1-3; selected MA-BBOB - every second definition of the sorted 24
(12 definitions) x instance 1; seeds 1-5.

Each candidate runs naturally from FE=0; checkpoints at FE {2000,4000,6000}
are cloned; every checkpoint forks {continue} U {switch to other KEEP
candidates} U {switch to CMA-ES (isolated add-back control)}; every branch
runs 1000 FE (the primary screening horizon). A deterministic 10 percent of
state-action pairs repeats R=3 with the formal RNG fork semantics.
"""
from __future__ import annotations

import argparse
import resource
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

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
from optimizers.seeding import make_rng
from optimizers.state import NATIVE_STREAMS

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/behavior_with_ela_train.yaml"
OUTPUT = ROOT / "behavior_with_ela/results/portfolio_screening/task12/stage2"
CANDIDATES = ("shade", "lshade", "cso")
DOMINANCE_CONTROL = "cmaes"
CHECKPOINT_FES = (2000, 4000, 6000)
HORIZON = 1000
SEEDS = (1, 2, 3, 4, 5)
BBOB_FAMILY_FUNCTIONS = {
    "bbob_separable_f01_f05": (1, 2),
    "bbob_low_or_moderate_conditioning_f06_f09": (6, 7),
    "bbob_high_conditioning_unimodal_f10_f14": (10, 11),
    "bbob_multimodal_adequate_global_structure_f15_f19": (15, 16),
    "bbob_multimodal_weak_global_structure_f20_f24": (20, 21),
}
MA_DEFINITION_STRIDE = 2
REPETITION_FRACTION = 0.10
REPETITION_STREAM = 2026083012


def stage2_jobs(config) -> list[dict]:
    jobs = []
    for suite in config.suites:
        if suite.split == "bbob_train":
            family_map = {}
            for function in suite.functions:
                label = None
                jobs_probe = make_experiment_problem
                # family membership via problem metadata: probe with a cheap call
                label = _bbob_family(suite, int(function))
                family_map.setdefault(label, []).append(int(function))
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


_FAMILY_CACHE: dict[int, str] = {}


def _bbob_family(suite, function: int) -> str:
    if function in _FAMILY_CACHE:
        return _FAMILY_CACHE[function]
    problem = make_experiment_problem(
        suite,
        function=function,
        instance=suite.instances[0],
        dimension=10,
        boundary_handling="reflect",
    )
    family = str(problem.family)
    _FAMILY_CACHE[function] = family
    problem.close()
    return family


def _sampled_for_repetition(*, suite_name: str, function: int, instance: int, seed: int, current: str, action: str, fe: int) -> bool:
    from optimizers.state import NATIVE_STREAMS

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


def _run_horizon_branch(*, checkpoint_state, action: str, replicate_id: int, problem, seed: int, function: int, instance: int, config) -> tuple[float, int, float]:
    started = perf_counter()
    evaluations = 0
    best = float(checkpoint_state.best_fitness)
    reference = float(problem.reference_value)

    def observe_evaluation(point: np.ndarray, value: float) -> None:
        nonlocal evaluations, best
        evaluations += 1
        if value < best:
            best = float(value)

    if action == "continue" or action == checkpoint_state.algorithm:
        branch_state = clone_optimizer_state(checkpoint_state)
        if replicate_id > 0:
            rng = make_rng(
                seed,
                NATIVE_STREAMS[checkpoint_state.algorithm]
                + CONTINUATION_REPETITION_STREAM_OFFSET * replicate_id,
                suite_code=problem.suite_code,
                function=problem.function_number,
                instance=problem.instance_number,
                dimension=problem.dimension,
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
    remaining = int(HORIZON)
    advance_optimizer_state(
        state=branch_state,
        problem=problem,
        fe_budget=remaining,
        on_evaluation=observe_evaluation,
    )
    if evaluations != remaining:
        raise RuntimeError(f"branch evaluations {evaluations} != {remaining}")
    gap = float(
        min(max(best - reference, 0.0), config.failure_loss_cap)
    )
    loss = float(np.log10(np.clip(gap, config.log10_gap_floor, config.log10_gap_cap)))
    return loss, remaining, perf_counter() - started


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
    state_rows = []
    branch_rows = []
    addback_rows = []
    ledgers = []
    for seed in job["seeds"]:
        for current in CANDIDATES:
            evaluations = 0

            def observe_evaluation(point: np.ndarray, value: float) -> None:
                nonlocal evaluations
                evaluations += 1

            state = initialize_optimizer_state(
                algorithm=current,
                problem=problem,
                seed=seed,
                settings=settings,
                on_evaluation=observe_evaluation,
            )
            pending = list(CHECKPOINT_FES)
            checkpoint_states = {}
            terminal_marks = []
            while evaluations < config.fe_total:
                advance_optimizer_state(
                    state=state,
                    problem=problem,
                    fe_budget=min(config.population_size, config.fe_total - evaluations),
                    on_evaluation=observe_evaluation,
                )
                if pending and evaluations == pending[0]:
                    fe = pending.pop(0)
                    checkpoint_states[fe] = clone_optimizer_state(state)
                terminal_marks.append(evaluations)
            terminal_gap = float(
                min(
                    max(state.best_fitness - float(problem.reference_value), 0.0),
                    config.failure_loss_cap,
                )
            )
            terminal_loss = float(
                np.log10(np.clip(terminal_gap, config.log10_gap_floor, config.log10_gap_cap))
            )
            for fe in CHECKPOINT_FES:
                checkpoint_state = checkpoint_states[fe]
                state_id = (
                    f"{suite.split}:{problem.problem_id}:current_{current}:"
                    f"seed{int(seed)}:fe{fe}"
                )
                actions = ["continue"] + [c for c in CANDIDATES if c != current]
                for action in actions:
                    sampled = _sampled_for_repetition(
                        suite_name=suite.suite,
                        function=job["function"],
                        instance=job["instance"],
                        seed=seed,
                        current=current,
                        action=action,
                        fe=fe,
                    )
                    replicates = range(3) if sampled else range(1)
                    for replicate_id in replicates:
                        loss, used_fe, elapsed = _run_horizon_branch(
                            checkpoint_state=checkpoint_state,
                            action=action,
                            replicate_id=replicate_id,
                            problem=problem,
                            seed=seed,
                            function=job["function"],
                            instance=job["instance"],
                            config=config,
                        )
                        branch_rows.append(
                            {
                                "state_id": state_id,
                                "route": f"current_{current}",
                                "source_algorithm": current,
                                "current_algorithm": current,
                                "FE": fe,
                                "dwell_FE": fe,
                                "candidate_action": action,
                                "replicate_id": int(replicate_id),
                                "loss_1000": loss,
                                "terminal_loss_current_run": terminal_loss,
                            }
                        )
                        ledgers.append(
                            {
                                "phase": "stage2_branch",
                                "suite": suite.suite,
                                "fe_used": used_fe,
                                "wall_seconds": elapsed,
                            }
                        )
                # isolated CMA-ES dominance-control branch (analysis deferred
                # until P_balanced is frozen in the report layer)
                addback_sampled = _sampled_for_repetition(
                    suite_name=suite.suite,
                    function=job["function"],
                    instance=job["instance"],
                    seed=seed,
                    current=current,
                    action=DOMINANCE_CONTROL,
                    fe=fe,
                )
                for replicate_id in (range(3) if addback_sampled else range(1)):
                    loss, used_fe, elapsed = _run_horizon_branch(
                        checkpoint_state=checkpoint_state,
                        action=DOMINANCE_CONTROL,
                        replicate_id=replicate_id,
                        problem=problem,
                        seed=seed,
                        function=job["function"],
                        instance=job["instance"],
                        config=config,
                    )
                    addback_rows.append(
                        {
                            "state_id": state_id,
                            "route": f"current_{current}",
                            "source_algorithm": current,
                            "current_algorithm": current,
                            "FE": fe,
                            "dwell_FE": fe,
                            "candidate_action": DOMINANCE_CONTROL,
                            "replicate_id": int(replicate_id),
                            "loss_1000": loss,
                        }
                    )
                    ledgers.append(
                        {
                            "phase": "cmaes_addback_branch",
                            "suite": suite.suite,
                            "fe_used": used_fe,
                            "wall_seconds": elapsed,
                        }
                    )
                state_rows.append(
                    {
                        "state_id": state_id,
                        "suite": suite.suite,
                        "split": suite.split,
                        "problem_id": problem.problem_id,
                        "function_id": problem.function_id,
                        "family": problem.family,
                        "cv_group_id": problem.cv_group_id,
                        "instance": int(job["instance"]),
                        "seed": int(seed),
                        "route": f"current_{current}",
                        "source_algorithm": current,
                        "current_algorithm": current,
                        "handoff_performed": False,
                        "FE": fe,
                        "FE_ratio": fe / float(config.fe_total),
                        "dwell_FE": fe,
                        "terminal_loss_current_run": terminal_loss,
                    }
                )
    problem.close()
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "job": job,
        "states": state_rows,
        "branches": branch_rows,
        "addback": addback_rows,
        "ledgers": ledgers,
        "max_rss_mb": rss_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config = load_experiment_config(CONFIG)
    output = OUTPUT / "shards"
    output.mkdir(parents=True, exist_ok=True)
    jobs = stage2_jobs(config)
    print(f"[task12_stage2] units: {len(jobs)}", flush=True)
    total_states = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_collect_unit, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            job = result["job"]
            shard = output / f"{job['split']}_{job['suite']}_f{job['function']:03d}_i{job['instance']}"
            shard.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(result["states"]).to_parquet(shard / "states.parquet", index=False)
            pd.DataFrame(result["branches"]).to_parquet(shard / "branches.parquet", index=False)
            pd.DataFrame(result["addback"]).to_parquet(shard / "addback.parquet", index=False)
            pd.DataFrame(result["ledgers"]).to_parquet(shard / "ledger.parquet", index=False)
            total_states += len(result["states"])
            if index % 5 == 0 or index == len(jobs):
                print(
                    f"[task12_stage2] {index}/{len(jobs)} done (states so far {total_states})",
                    flush=True,
                )
    print(f"[task12_stage2] done: states={total_states}", flush=True)


if __name__ == "__main__":
    main()
