"""Task 13C/13D: deterministic state-reconstruction replay with the formal
behavior recorder, plus the exact alignment audit against the Task 12
stage-2 states.

The replay re-runs the natural SHADE / L-SHADE / CSO trajectories of the
Task 12 stage-2 problem set (identical problems, instances, seeds,
dimension, settings, boundary handling, RNG semantics and FE accounting)
from FE=0 to FE=6000 only. It generates no action labels: the only new
artifacts are the formal NativeUpdateWindowRecorder snapshots and the
extracted global behavior features at the three decision checkpoints
FE in {2000, 4000, 6000}.

Every checkpoint is aligned against the Task 12.1 checkpoint-gap table
(recovered there from the stage-1 marks). Any state with a log10-gap
difference above 1e-12, or any key mismatch, stops the round: replayed
behavior must never be joined to action outcomes of divergent states.
"""
from __future__ import annotations

import json
import resource
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from behavior.features import extract_behavior_rows
from behavior_with_ela.analysis_v5.task12_1_analysis import TASK12_HEAVY
from behavior_with_ela.protocol import (
    load_experiment_config,
    make_experiment_problem,
)
from optimizers import (
    OptimizerSettings,
    advance_optimizer_state,
    initialize_optimizer_state,
)
from trajectory.records import TrajectoryRecord
from trajectory.sampling import SAMPLING_METADATA_COLUMNS
from trajectory.window_statistics import NativeUpdateWindowRecorder

ROOT = Path(__file__).resolve().parents[3]
T12_1_HEAVY = ROOT / "behavior_with_ela/results/analysis_v5/task12_1"
CONFIG = ROOT / "configs/behavior_with_ela_train.yaml"
HEAVY = ROOT / "behavior_with_ela/results/analysis_v5/task13"
LIGHT = ROOT / "behavior_with_ela/analysis_v5/task13"
CANDIDATES = ("shade", "lshade", "cso")
CHECKPOINT_FES = (2000, 4000, 6000)
REPLAY_FE_LIMIT = 6000
SEEDS = (1, 2, 3, 4, 5)
BBOB_FAMILY_FUNCTIONS = {
    "bbob_separable_f01_f05": (1, 2),
    "bbob_low_or_moderate_conditioning_f06_f09": (6, 7),
    "bbob_high_conditioning_unimodal_f10_f14": (10, 11),
    "bbob_multimodal_adequate_global_structure_f15_f19": (15, 16),
    "bbob_multimodal_weak_global_structure_f20_f24": (20, 21),
}
MA_DEFINITION_STRIDE = 2
ALIGNMENT_TOLERANCE = 1e-12
SAMPLING_METADATA = {
    "sampling_protocol": "task13_state_reconstruction_v1",
    "sampling_phase": "replay_checkpoint",
    "sampling_triggers": [],
    "is_budget_milestone": False,
    "budget_milestone_ratio": None,
    "is_event_sample": False,
    "monitor_target_ratio": 0.0,
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


def replay_jobs(config) -> list[dict]:
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


def _replay_unit(job: dict) -> dict:
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
    checkpoint_rows = []
    behavior_rows = []
    started = perf_counter()
    for seed in job["seeds"]:
        for current in CANDIDATES:
            evaluations = 0
            native_updates = 0
            window = NativeUpdateWindowRecorder()

            def observe_evaluation(point: np.ndarray, value: float) -> None:
                nonlocal evaluations
                evaluations += 1

            def observe_update(updated) -> None:
                nonlocal native_updates
                native_updates += 1
                window.observe(
                    fe=evaluations,
                    native_updates=native_updates,
                    population=updated.population,
                    fitness=updated.fitness,
                    best_fitness=updated.best_fitness,
                )

            state = initialize_optimizer_state(
                algorithm=current,
                problem=problem,
                seed=seed,
                settings=settings,
                on_evaluation=observe_evaluation,
            )
            window.observe(
                fe=evaluations,
                native_updates=0,
                population=state.population,
                fitness=state.fitness,
                best_fitness=state.best_fitness,
            )
            pending = list(CHECKPOINT_FES)
            run_records = []
            run_meta = {}
            while evaluations < REPLAY_FE_LIMIT:
                advance_optimizer_state(
                    state=state,
                    problem=problem,
                    fe_budget=min(config.population_size, REPLAY_FE_LIMIT - evaluations),
                    on_native_update=observe_update,
                    on_evaluation=observe_evaluation,
                )
                if pending and evaluations == pending[0]:
                    fe = pending.pop(0)
                    state_id = (
                        f"{suite.split}:{problem.problem_id}:current_{current}:"
                        f"seed{int(seed)}:fe{fe}"
                    )
                    best = float(state.best_fitness)
                    gap = float(
                        min(
                            max(best - float(problem.reference_value), 0.0),
                            config.failure_loss_cap,
                        )
                    )
                    log10_gap = float(
                        np.log10(np.clip(gap, config.log10_gap_floor, config.log10_gap_cap))
                    )
                    window_statistics, native_history = window.build(
                        fe_total=config.fe_total,
                        problem_id=problem.problem_id,
                        algorithm=current,
                    )
                    snapshot = window.current_snapshot
                    record = TrajectoryRecord.from_arrays(
                        problem_id=problem.problem_id,
                        function_id=problem.function_id,
                        family=problem.family,
                        cv_group_id=problem.cv_group_id,
                        dimension=config.dimension,
                        algorithm=current,
                        seed=int(seed),
                        # the formal extraction contract requires the record to
                        # end exactly at the last completed native update
                        fe=snapshot.fe,
                        fe_total=config.fe_total,
                        native_updates=snapshot.native_updates,
                        window_statistics=window_statistics,
                        native_update_history=native_history,
                        population=snapshot.population,
                        fitness=snapshot.fitness,
                        best_fitness=snapshot.best_fitness,
                        sampling_metadata=dict(SAMPLING_METADATA),
                    )
                    run_records.append(record)
                    run_meta[fe] = {
                        "state_id": state_id,
                        "suite": suite.suite,
                        "split": suite.split,
                        "problem_id": problem.problem_id,
                        "instance": int(job["instance"]),
                        "seed": int(seed),
                        "current_algorithm": current,
                        "FE": fe,
                        "best_fitness": best,
                        "log10_gap": log10_gap,
                        "population_size_at_checkpoint": int(state.population.shape[0]),
                        "generation_at_checkpoint": int(state.generation),
                        "snapshot_fe": int(snapshot.fe),
                        "snapshot_native_updates": int(snapshot.native_updates),
                        "snapshot_population_size": int(snapshot.population.shape[0]),
                    }
            behavior_rows.extend(extract_behavior_rows([asdict(r) for r in run_records]))
            for fe, meta in run_meta.items():
                checkpoint_rows.append(meta)
    replay_fe = len(job["seeds"]) * len(CANDIDATES) * REPLAY_FE_LIMIT
    problem.close()
    return {
        "job": job,
        "checkpoints": checkpoint_rows,
        "behavior": behavior_rows,
        "replay_fe": replay_fe,
        "wall_seconds": perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true", help="load existing replay parquets instead of re-running")
    args = parser.parse_args()
    HEAVY.mkdir(parents=True, exist_ok=True)
    LIGHT.mkdir(parents=True, exist_ok=True)
    config = load_experiment_config(CONFIG)
    jobs = replay_jobs(config)
    states_ref = pd.read_parquet(TASK12_HEAVY / "dynamic_screening_states.parquet")
    gap_ref = pd.read_parquet(T12_1_HEAVY / "states_with_checkpoint_gap.parquet")
    expected_problems = set(states_ref["problem_id"].unique())
    job_problems = set()
    for job in jobs:
        problem = make_experiment_problem(
            {s.split: s for s in config.suites}[job["split"]],
            function=job["function"],
            instance=job["instance"],
            dimension=config.dimension,
            boundary_handling=config.boundary_handling,
        )
        job_problems.add(problem.problem_id)
        problem.close()
    if job_problems != expected_problems:
        raise SystemExit(
            "[task13C] replay problem set does not match the stage-2 state set: "
            f"missing={sorted(expected_problems - job_problems)} extra={sorted(job_problems - expected_problems)}"
        )

    started = perf_counter()
    checkpoints_path = HEAVY / "behavior_replay_checkpoints.parquet"
    behavior_path = HEAVY / "behavior_global_features.parquet"
    if args.reuse and checkpoints_path.exists() and behavior_path.exists():
        checkpoints_df = pd.read_parquet(checkpoints_path)
        total_fe = int(len(checkpoints_df) // len(CHECKPOINT_FES) * REPLAY_FE_LIMIT)
        print(f"[task13C] reusing existing replay artifacts ({len(checkpoints_df)} checkpoints)", flush=True)
    else:
        checkpoints, behavior = [], []
        total_fe = 0
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_replay_unit, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                checkpoints.extend(result["checkpoints"])
                behavior.extend(result["behavior"])
                total_fe += result["replay_fe"]
                if index % 8 == 0 or index == len(jobs):
                    print(f"[task13C] {index}/{len(jobs)} units, FE so far {total_fe}", flush=True)
        checkpoints_df = pd.DataFrame(checkpoints)
        checkpoints_df.to_parquet(checkpoints_path, index=False)
        behavior_df = pd.DataFrame(behavior)
        mapping = checkpoints_df[
            ["state_id", "problem_id", "seed", "current_algorithm", "snapshot_fe"]
        ].rename(columns={"current_algorithm": "algorithm", "snapshot_fe": "FE"})
        behavior_df = behavior_df.merge(mapping, on=["problem_id", "algorithm", "seed", "FE"], validate="many_to_one")
        if len(behavior_df) != len(checkpoints_df) or behavior_df["state_id"].isna().any():
            raise SystemExit("[task13C] behavior rows do not map one-to-one onto replay checkpoints")
        behavior_df.to_parquet(behavior_path, index=False)

    # ---- 13D alignment audit ----
    key = checkpoints_df.copy()
    key["state_key"] = (
        key["split"] + ":" + key["problem_id"] + ":current_" + key["current_algorithm"]
        + ":seed" + key["seed"].astype(str) + ":fe" + key["FE"].astype(str)
    )
    ref = gap_ref[["state_id", "suite", "problem_id", "instance", "seed", "current_algorithm", "FE", "checkpoint_log10_gap"]]
    merged = key.merge(ref, left_on="state_key", right_on="state_id", how="outer", indicator=True)
    if not (merged["_merge"] == "both").all():
        missing = merged.loc[merged["_merge"].ne("both"), "state_key"]
        raise SystemExit(f"[task13D] state set mismatch: {len(missing)} unmatched state ids")
    merged["gap_abs_diff"] = (merged["log10_gap"] - merged["checkpoint_log10_gap"]).abs()
    key_checks = (
        merged["suite_x"].eq(merged["suite_y"])
        & merged["problem_id_x"].eq(merged["problem_id_y"])
        & merged["instance_x"].eq(merged["instance_y"])
        & merged["seed_x"].eq(merged["seed_y"])
        & merged["current_algorithm_x"].eq(merged["current_algorithm_y"])
        & merged["FE_x"].eq(merged["FE_y"])
    )
    if not key_checks.all() or merged["gap_abs_diff"].max() > ALIGNMENT_TOLERANCE:
        bad = merged.loc[~key_checks | merged["gap_abs_diff"].gt(ALIGNMENT_TOLERANCE), "state_id"]
        raise SystemExit(
            f"[task13D] alignment FAILED for {len(bad)} states (max gap diff {merged['gap_abs_diff'].max()}); STOP"
        )
    alignment = merged[
        [
            "state_key",
            "suite_x",
            "problem_id_x",
            "instance_x",
            "seed_x",
            "current_algorithm_x",
            "FE_x",
            "log10_gap",
            "checkpoint_log10_gap",
            "gap_abs_diff",
            "population_size_at_checkpoint",
            "generation_at_checkpoint",
            "snapshot_fe",
            "snapshot_native_updates",
            "snapshot_population_size",
        ]
    ].rename(columns={"state_key": "state_id", "suite_x": "suite", "problem_id_x": "problem_id", "instance_x": "instance",
                      "seed_x": "seed", "current_algorithm_x": "current_algorithm", "FE_x": "FE"})
    alignment.to_parquet(HEAVY / "behavior_replay_alignment.parquet", index=False)

    summary = {
        "replay_executed": True,
        "replay_fe_used": int(total_fe),
        "replay_fe_cap": int(len(jobs) * 5 * 3 * REPLAY_FE_LIMIT),
        "n_states": int(len(alignment)),
        "max_gap_abs_diff": float(merged["gap_abs_diff"].max()),
        "alignment_tolerance": ALIGNMENT_TOLERANCE,
        "all_keys_match": bool(key_checks.all()),
        "lshade_snapshot_population_sizes": sorted(
            alignment.loc[alignment["current_algorithm"].eq("lshade"), "snapshot_population_size"]
            .unique()
            .tolist()
        ),
        "wall_seconds": perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    (LIGHT / "replay_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
