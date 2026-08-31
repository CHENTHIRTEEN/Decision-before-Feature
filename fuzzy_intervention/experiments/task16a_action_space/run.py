from __future__ import annotations

import argparse
import copy
import resource
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from behavior_with_ela.analysis_v6.task14a_collect import collection_jobs
from behavior_with_ela.protocol import (
    load_experiment_config,
    make_experiment_problem,
    suite_code,
)
from fuzzy_intervention.interventions import apply_partial_perturbation
from fuzzy_intervention.probes import AgentHistoryTracker, extract_task16a_probes
from optimizers import (
    OptimizerSettings,
    advance_optimizer_state,
    clone_optimizer_state,
    initialize_optimizer_state,
    initialize_transferred_optimizer_state,
)
from optimizers.seeding import make_indexed_rng
from optimizers.state import NATIVE_STREAMS
from trajectory.window_statistics import NativeUpdateWindowRecorder


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/behavior_with_ela_train.yaml"
OUTPUT = ROOT / "fuzzy_intervention/results/task16a"
SOLVERS = ("shade", "lshade", "cso")
NOMINAL_CHECKPOINTS = (2000, 4000, 6000, 8000)
SEEDS = (1, 2, 3, 4, 5)
ACTION_HORIZON_FE = 1000
REPETITION_FRACTION = 0.10
REPETITION_SELECTION_STREAM = 2026101601
SUBSET_STREAM = 2026101602
VECTOR_STREAM = 2026101603
BRANCH_STREAM_BASE = 2026101700
ACTION_CODES = {
    "continue": 1,
    "perturb_targeted": 2,
    "perturb_random": 3,
    "switch_shade": 11,
    "switch_lshade": 12,
    "switch_cso": 13,
}


def _pending(state):
    value = getattr(state, "pending_population", None)
    return value if value is not None else getattr(state, "pending_positions", None)


def _action_names(current: str) -> tuple[str, ...]:
    switches = tuple(f"switch_{target}" for target in SOLVERS if target != current)
    return ("continue", "perturb_targeted", "perturb_random", *switches)


def _is_repetition_selected(*, problem, seed: int, current: str, nominal_fe: int, action: str) -> bool:
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                REPETITION_SELECTION_STREAM,
                int(problem.suite_code),
                int(problem.function_number),
                int(problem.instance_number),
                int(problem.dimension),
                int(seed),
                int(NATIVE_STREAMS[current]),
                int(nominal_fe),
                int(ACTION_CODES[action]),
            ]
        )
    )
    return bool(rng.random() < REPETITION_FRACTION)


def _event_rng(*, problem, seed: int, current: str, nominal_fe: int, action: str, repetition: int, stream: int):
    unit_number = (
        int(problem.suite_code) * 100_000_000
        + int(problem.function_number) * 100_000
        + int(problem.instance_number) * 100
        + int(problem.dimension)
    )
    return make_indexed_rng(
        seed=int(seed),
        unit_number=unit_number,
        stream_code=int(stream + ACTION_CODES[action] * 100 + NATIVE_STREAMS[current]),
        generation=int(nominal_fe),
        target=int(ACTION_CODES[action]),
        event=int(repetition),
    )


def _semantic_key(*, problem, seed: int, current: str, nominal_fe: int, action: str, repetition: int) -> str:
    return (
        f"suite{int(problem.suite_code)}:f{int(problem.function_number)}:"
        f"i{int(problem.instance_number)}:d{int(problem.dimension)}:seed{int(seed)}:"
        f"current_{current}:fe{int(nominal_fe)}:{action}:rep{int(repetition)}"
    )


def _log10_loss(problem, best_fitness: float, config) -> float:
    gap = min(
        max(float(best_fitness) - float(problem.reference_value), 0.0),
        float(config.failure_loss_cap),
    )
    return float(np.log10(np.clip(gap, config.log10_gap_floor, config.log10_gap_cap)))


def _make_source_row(*, job, problem, seed: int, algorithm: str, nominal_fe: int,
                     state, recorder, native_updates: int, tracker: AgentHistoryTracker) -> tuple[dict, list[dict]]:
    actual_fe = int(state.evaluations)
    if actual_fe > nominal_fe:
        raise RuntimeError("actual source FE must not exceed its nominal checkpoint")
    alignment_gap = int(nominal_fe - actual_fe)
    local_update_fe = int(len(state.population)) if algorithm != "cso" else int(len(state.population) // 2)
    if not 0 <= alignment_gap < local_update_fe:
        raise RuntimeError("source checkpoint is not within one local native update")
    probes = extract_task16a_probes(
        problem=problem,
        state=state,
        algorithm=algorithm,
        seed=seed,
        recorder=recorder,
        native_updates=native_updates,
    )
    state_id = (
        f"{job['split']}:{problem.problem_id}:current_{algorithm}:"
        f"seed{int(seed)}:fe{int(nominal_fe)}"
    )
    primitives = tracker.individual_primitives(state)
    row = {
        "state_id": state_id,
        "suite": str(job["suite"]),
        "split": str(job["split"]),
        "problem_id": problem.problem_id,
        "function_id": problem.function_id,
        "family": problem.family,
        "cv_group_id": problem.cv_group_id,
        "instance": int(problem.instance_number),
        "seed": int(seed),
        "dimension": int(problem.dimension),
        "current_algorithm": algorithm,
        "source_FE": int(nominal_fe),
        "source_FE_nominal": int(nominal_fe),
        "source_FE_actual": actual_fe,
        "source_FE_alignment_gap": alignment_gap,
        "local_native_update_FE": local_update_fe,
        "maturity": float(nominal_fe / 10000.0),
        "population_size": int(len(state.population)),
        "best_fitness": float(state.best_fitness),
        "population": np.asarray(state.population, dtype=float).tolist(),
        "fitness": np.asarray(state.fitness, dtype=float).tolist(),
        "native_updates": int(native_updates),
        "source_state_complete_native_update": bool(_pending(state) is None),
        **probes,
    }
    for primitive in primitives:
        primitive["state_id"] = state_id
    return row, primitives


def _run_action(*, config, problem, source_state, source_row: dict, primitives: list[dict],
                action: str, repetition: int, selected_for_repetition: bool) -> tuple[dict, list[dict]]:
    current = str(source_row["current_algorithm"])
    nominal_fe = int(source_row["source_FE_nominal"])
    seed = int(source_row["seed"])
    state = clone_optimizer_state(source_state)
    branch_rng = _event_rng(
        problem=problem,
        seed=seed,
        current=current,
        nominal_fe=nominal_fe,
        action=action,
        repetition=repetition,
        stream=BRANCH_STREAM_BASE,
    )
    perturb_mode = None
    perturbed_count = 0
    perturb_selected_rows: list[dict] = []
    target = None
    if action.startswith("switch_"):
        target = action.removeprefix("switch_")
        state = initialize_transferred_optimizer_state(
            algorithm=target,
            source_state=source_state,
            problem=problem,
            seed=seed,
            function=problem.function_number,
            instance=problem.instance_number,
            event=int(nominal_fe * 100 + ACTION_CODES[action] * 10 + repetition),
        )
        state.rng_state = copy.deepcopy(branch_rng.bit_generator.state)
    else:
        state.rng_state = copy.deepcopy(branch_rng.bit_generator.state)

    if action in {"perturb_targeted", "perturb_random"}:
        perturb_mode = action.removeprefix("perturb_")
        subset_rng = _event_rng(
            problem=problem,
            seed=seed,
            current=current,
            nominal_fe=nominal_fe,
            action=action,
            repetition=repetition,
            stream=SUBSET_STREAM,
        )
        vector_rng = _event_rng(
            problem=problem,
            seed=seed,
            current=current,
            nominal_fe=nominal_fe,
            action=action,
            repetition=repetition,
            stream=VECTOR_STREAM,
        )
        perturbed_count, perturb_selected_rows = apply_partial_perturbation(
            state=state,
            problem=problem,
            primitives=primitives,
            mode=perturb_mode,
            subset_rng=subset_rng,
            vector_rng=vector_rng,
        )

    remaining = ACTION_HORIZON_FE - perturbed_count
    if remaining < 0:
        raise RuntimeError("perturbation consumed more than the action horizon")
    advance_optimizer_state(state=state, problem=problem, fe_budget=remaining)
    if perturbed_count + remaining != ACTION_HORIZON_FE:
        raise RuntimeError("action FE accounting mismatch")
    semantic_key = _semantic_key(
        problem=problem,
        seed=seed,
        current=current,
        nominal_fe=nominal_fe,
        action=action,
        repetition=repetition,
    )
    outcome = {
        **{key: source_row[key] for key in (
            "state_id", "suite", "split", "problem_id", "function_id", "family",
            "cv_group_id", "instance", "seed", "dimension", "current_algorithm",
            "source_FE", "source_FE_nominal", "source_FE_actual", "source_FE_alignment_gap",
            "maturity", "probe_productivity", "probe_entropy", "probe_stagnation",
        )},
        "action": action,
        "target_algorithm_if_switch": target,
        "perturb_mode": perturb_mode,
        "perturbed_count": int(perturbed_count),
        "loss_terminal": _log10_loss(problem, state.best_fitness, config),
        "repetition_id": int(repetition),
        "is_repetition_selected": bool(selected_for_repetition),
        "semantic_rng_key": semantic_key,
        "actual_action_FE": int(ACTION_HORIZON_FE),
        "terminal_global_FE": int(source_row["source_FE_actual"] + ACTION_HORIZON_FE),
        "native_update_complete_at_endpoint": bool(_pending(state) is None),
    }
    if perturb_mode is not None:
        selected_by_agent = {str(row["agent_id"]): row for row in perturb_selected_rows}
        fitness = np.asarray(source_state.fitness, dtype=float)
        for primitive in primitives:
            agent_id = str(primitive["agent_id"])
            selected = selected_by_agent.get(agent_id)
            index = int(primitive["population_index"])
            metadata = {
                "state_id": source_row["state_id"],
                "suite": source_row["suite"],
                "problem_id": source_row["problem_id"],
                "cv_group_id": source_row["cv_group_id"],
                "seed": seed,
                "current_algorithm": current,
                "source_FE": nominal_fe,
                "action": action,
                "perturb_mode": perturb_mode,
                "repetition_id": int(repetition),
                "semantic_rng_key": semantic_key,
                "agent_id": agent_id,
                "population_index": index,
                "selected_for_perturb": bool(selected is not None),
                "selection_rank_stagnation": int(primitive["selection_rank_stagnation"]),
                "selection_rank_progress": int(primitive["selection_rank_progress"]),
                "individual_stagnation_age_FE": int(primitive["individual_stagnation_age_FE"]),
                "individual_recent_progress": float(primitive["individual_recent_progress"]),
                "fitness_before": float(fitness[index]),
                "fitness_after": float(fitness[index] if selected is None else selected["fitness_after"]),
                "perturb_norm_unitcube": float(0.0 if selected is None else selected["perturb_norm_unitcube"]),
                "became_new_best": bool(False if selected is None else selected["became_new_best"]),
            }
            perturb_selected_rows.append(metadata)
        perturb_selected_rows = perturb_selected_rows[len(selected_by_agent):]
    return outcome, perturb_selected_rows


def _collect_unit(job: dict) -> dict:
    config = load_experiment_config(CONFIG)
    suite = {item.split: item for item in config.suites}[job["split"]]
    problem = make_experiment_problem(
        suite,
        function=job["function"],
        instance=job["instance"],
        dimension=10,
        boundary_handling="reflect",
    )
    settings = OptimizerSettings(population_size=40, boundary_handling="reflect")
    source_rows: list[dict] = []
    primitive_rows: list[dict] = []
    outcome_rows: list[dict] = []
    perturb_rows: list[dict] = []
    source_fe_used = 0
    action_fe_used = 0
    started = perf_counter()

    for seed in SEEDS:
        for algorithm in SOLVERS:
            state = initialize_optimizer_state(
                algorithm=algorithm,
                problem=problem,
                seed=seed,
                settings=settings,
            )
            tracker = AgentHistoryTracker(state)
            recorder = NativeUpdateWindowRecorder()
            native_updates = 0
            recorder.observe(
                fe=int(state.evaluations),
                native_updates=native_updates,
                population=state.population,
                fitness=state.fitness,
                best_fitness=state.best_fitness,
            )
            latest_state = clone_optimizer_state(state)
            latest_tracker = tracker.clone()
            latest_recorder = copy.deepcopy(recorder)
            latest_native_updates = native_updates

            def observe_evaluation(_point, value) -> None:
                tracker.observe_evaluation(state, float(value))

            def observe_update(updated) -> None:
                nonlocal native_updates, latest_state, latest_tracker
                nonlocal latest_recorder, latest_native_updates
                tracker.finish_generation(updated)
                native_updates += 1
                recorder.observe(
                    fe=int(updated.evaluations),
                    native_updates=native_updates,
                    population=updated.population,
                    fitness=updated.fitness,
                    best_fitness=updated.best_fitness,
                )
                latest_state = clone_optimizer_state(updated)
                latest_tracker = tracker.clone()
                latest_recorder = copy.deepcopy(recorder)
                latest_native_updates = native_updates

            for nominal_fe in NOMINAL_CHECKPOINTS:
                while int(state.evaluations) < nominal_fe:
                    pending = _pending(state)
                    if pending is None:
                        tracker.start_generation(state)
                        budget = 1
                    else:
                        budget = min(
                            len(pending) - int(state.pending_index),
                            nominal_fe - int(state.evaluations),
                        )
                    advance_optimizer_state(
                        state=state,
                        problem=problem,
                        fe_budget=int(budget),
                        on_native_update=observe_update,
                        on_evaluation=observe_evaluation,
                    )
                source_state = clone_optimizer_state(latest_state)
                source_tracker = latest_tracker.clone()
                source_recorder = copy.deepcopy(latest_recorder)
                source_row, primitives = _make_source_row(
                    job=job,
                    problem=problem,
                    seed=seed,
                    algorithm=algorithm,
                    nominal_fe=nominal_fe,
                    state=source_state,
                    recorder=source_recorder,
                    native_updates=latest_native_updates,
                    tracker=source_tracker,
                )
                source_rows.append(source_row)
                primitive_rows.extend(primitives)
                for action in _action_names(algorithm):
                    selected = _is_repetition_selected(
                        problem=problem,
                        seed=seed,
                        current=algorithm,
                        nominal_fe=nominal_fe,
                        action=action,
                    )
                    for repetition in (range(3) if selected else range(1)):
                        outcome, perturb = _run_action(
                            config=config,
                            problem=problem,
                            source_state=source_state,
                            source_row=source_row,
                            primitives=primitives,
                            action=action,
                            repetition=int(repetition),
                            selected_for_repetition=selected,
                        )
                        outcome_rows.append(outcome)
                        perturb_rows.extend(perturb)
                        action_fe_used += ACTION_HORIZON_FE
            source_fe_used += max(NOMINAL_CHECKPOINTS)
    problem.close()
    return {
        "job": job,
        "sources": source_rows,
        "primitives": primitive_rows,
        "outcomes": outcome_rows,
        "perturb": perturb_rows,
        "source_fe_used": int(source_fe_used),
        "action_fe_used": int(action_fe_used),
        "wall_seconds": float(perf_counter() - started),
        "peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
    }


def _write_resource_plan(*, workers: int, num_jobs: int) -> pd.DataFrame:
    num_states = num_jobs * len(SEEDS) * len(SOLVERS) * len(NOMINAL_CHECKPOINTS)
    source_fe = num_jobs * len(SEEDS) * len(SOLVERS) * max(NOMINAL_CHECKPOINTS)
    primary_per_action = num_states * ACTION_HORIZON_FE
    expected_repetition = int(round(num_states * 5 * REPETITION_FRACTION * 2 * ACTION_HORIZON_FE))
    throughput = 23_704_000 / 195.0
    rows = [
        ("natural_source_trajectories", num_states, 0, max(NOMINAL_CHECKPOINTS), source_fe),
        ("continue_primary", num_states, num_states, ACTION_HORIZON_FE, primary_per_action),
        ("perturb_targeted_primary", num_states, num_states, ACTION_HORIZON_FE, primary_per_action),
        ("perturb_random_primary", num_states, num_states, ACTION_HORIZON_FE, primary_per_action),
        ("switch_primary", num_states, num_states * 2, ACTION_HORIZON_FE, primary_per_action * 2),
        ("selected_action_repetitions_expected", num_states, int(num_states * 5 * 0.2), ACTION_HORIZON_FE, expected_repetition),
    ]
    frame = pd.DataFrame(
        rows,
        columns=["component", "num_states", "num_branches", "FE_per_branch", "planned_FE"],
    )
    frame["reused_FE"] = 0
    frame["new_FE"] = frame["planned_FE"]
    frame["expected_walltime"] = frame["planned_FE"] / throughput
    frame["workers"] = int(workers)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT / "task16a_resource_plan.parquet", index=False)
    return frame


def _reuse_verification() -> pd.DataFrame:
    rows = [
        {
            "candidate_artifact": "behavior_with_ela/results/analysis_v5/task12/dynamic_action_outcomes_1000.parquet",
            "candidate_scope": "natural Continue/Switch at 2000/4000/6000 FE",
            "full_population_available": False,
            "full_fitness_available": False,
            "adaptive_state_available": False,
            "rng_state_available": False,
            "reuse_eligible": False,
            "reason": "the artifact stores scalar outcomes but not the complete source state required for numerical alignment",
        },
        {
            "candidate_artifact": "behavior_with_ela/results/analysis_v6/task14a/post_handoff_action_outcomes_1000.parquet",
            "candidate_scope": "mature post-handoff states",
            "full_population_available": False,
            "full_fitness_available": False,
            "adaptive_state_available": False,
            "rng_state_available": False,
            "reuse_eligible": False,
            "reason": "state class differs from the required natural incumbent source states",
        },
        {
            "candidate_artifact": "behavior_with_ela/results/trajectories",
            "candidate_scope": "main portfolio trajectories",
            "full_population_available": True,
            "full_fitness_available": True,
            "adaptive_state_available": False,
            "rng_state_available": False,
            "reuse_eligible": False,
            "reason": "the stored algorithm set does not provide complete SHADE/L-SHADE/CSO native optimizer states",
        },
    ]
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUTPUT / "task16a_source_state_reuse_verification.parquet", index=False)
    return frame


def _normalize_probes(sources: pd.DataFrame) -> pd.DataFrame:
    frame = sources.copy()
    for raw, ranked in (
        ("probe_productivity", "probe_productivity_rank"),
        ("probe_entropy", "probe_entropy_rank"),
        ("probe_stagnation", "probe_stagnation_rank"),
    ):
        frame[ranked] = frame.groupby("current_algorithm", sort=True)[raw].rank(
            method="average", pct=True
        )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Task16A natural-state action outcomes")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config = load_experiment_config(CONFIG)
    jobs = collection_jobs(config)
    if len(jobs) != 42:
        raise RuntimeError(f"Task16A expected 42 problems, found {len(jobs)}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_resource_plan(workers=args.workers, num_jobs=len(jobs))
    _reuse_verification()
    print(f"[task16a] collecting {len(jobs)} problem units", flush=True)
    started = perf_counter()
    sources: list[dict] = []
    primitives: list[dict] = []
    outcomes: list[dict] = []
    perturb: list[dict] = []
    ledgers: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_collect_unit, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            sources.extend(result["sources"])
            primitives.extend(result["primitives"])
            outcomes.extend(result["outcomes"])
            perturb.extend(result["perturb"])
            ledgers.append(
                {
                    "split": result["job"]["split"],
                    "suite": result["job"]["suite"],
                    "function": int(result["job"]["function"]),
                    "instance": int(result["job"]["instance"]),
                    "source_FE": int(result["source_fe_used"]),
                    "action_FE": int(result["action_fe_used"]),
                    "new_FE": int(result["source_fe_used"] + result["action_fe_used"]),
                    "reused_FE": 0,
                    "wall_seconds": float(result["wall_seconds"]),
                    "peak_rss_mb": float(result["peak_rss_mb"]),
                    "workers": int(args.workers),
                }
            )
            if index % 5 == 0 or index == len(jobs):
                print(f"[task16a] {index}/{len(jobs)} units, {len(sources)} states", flush=True)

    source_frame = _normalize_probes(pd.DataFrame(sources))
    expected_states = 42 * 3 * 5 * 4
    if len(source_frame) != expected_states or source_frame["state_id"].nunique() != expected_states:
        raise RuntimeError("Task16A source-state coverage mismatch")
    rank_columns = [
        "state_id", "probe_productivity_rank", "probe_entropy_rank", "probe_stagnation_rank"
    ]
    outcome_frame = pd.DataFrame(outcomes).merge(
        source_frame[rank_columns], on="state_id", validate="many_to_one"
    )
    primary = outcome_frame.loc[outcome_frame["repetition_id"].eq(0)]
    counts = primary.groupby("state_id")["action"].nunique()
    if not counts.eq(5).all() or len(counts) != expected_states:
        raise RuntimeError("each source state must contain five concrete primary actions")
    continue_loss = primary.loc[primary["action"].eq("continue")].set_index("state_id")["loss_terminal"]
    outcome_frame["gain_vs_continue"] = (
        outcome_frame["state_id"].map(continue_loss) - outcome_frame["loss_terminal"]
    )
    outcome_frame["is_practical_nondominated"] = pd.NA
    outcome_frame["is_practical_unique_best"] = pd.NA
    repetition_frame = outcome_frame.loc[outcome_frame["is_repetition_selected"]].copy()
    ledger_frame = pd.DataFrame(ledgers)
    ledger_frame.loc[len(ledger_frame)] = {
        "split": "all",
        "suite": "all",
        "function": -1,
        "instance": -1,
        "source_FE": int(ledger_frame["source_FE"].sum()),
        "action_FE": int(ledger_frame["action_FE"].sum()),
        "new_FE": int(ledger_frame["new_FE"].sum()),
        "reused_FE": 0,
        "wall_seconds": float(perf_counter() - started),
        "peak_rss_mb": float(ledger_frame["peak_rss_mb"].max()),
        "workers": int(args.workers),
    }
    source_frame.to_parquet(OUTPUT / "task16a_source_states.parquet", index=False)
    source_frame[[
        "state_id", "suite", "problem_id", "cv_group_id", "seed", "current_algorithm",
        "source_FE", "source_FE_actual", "maturity", "probe_productivity", "probe_entropy",
        "probe_stagnation", "probe_productivity_rank", "probe_entropy_rank", "probe_stagnation_rank",
    ]].to_parquet(OUTPUT / "task16a_probe_values.parquet", index=False)
    pd.DataFrame(primitives).to_parquet(OUTPUT / "task16a_individual_primitives.parquet", index=False)
    outcome_frame.to_parquet(OUTPUT / "task16a_action_outcomes.parquet", index=False)
    repetition_frame.to_parquet(OUTPUT / "task16a_repetition_outcomes.parquet", index=False)
    pd.DataFrame(perturb).to_parquet(OUTPUT / "task16a_perturb_metadata.parquet", index=False)
    ledger_frame.to_parquet(OUTPUT / "task16a_resource_ledger.parquet", index=False)
    print(
        f"[task16a] complete: states={len(source_frame)}, outcomes={len(outcome_frame)}, "
        f"new_FE={int(ledger_frame.iloc[-1]['new_FE'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
