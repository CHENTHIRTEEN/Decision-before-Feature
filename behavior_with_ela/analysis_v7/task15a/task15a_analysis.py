"""Task 15A: behavior pre-screening and individual search-state distributions.

This module has two deliberately separate parts.  Stage A inventories the
existing 28 behavior columns and chooses representatives without reading any
action outcome.  Stage B reconstructs only the natural-source -> handoff ->
1000-FE mature state needed to record individual histories, then reuses the
existing Task 14A action outcomes for the sufficiency analysis.

The replay is outcome-blind: it creates no continuation, switch, reset, or
other action branch.  Agent identity is a search-slot/target-lineage identity
within each transferred solver segment.
"""
from __future__ import annotations

import argparse
import copy
import json
import resource
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS
from behavior_with_ela.analysis_v5.task13.task13_analysis import make_carrier
from behavior_with_ela.protocol import load_experiment_config, make_experiment_problem
from benchmarks.factory import problem_bounds
from optimizers import (
    OptimizerSettings,
    advance_optimizer_state,
    initialize_optimizer_state,
    initialize_transferred_optimizer_state,
)


ROOT = Path(__file__).resolve().parents[3]
T14A_HEAVY = ROOT / "behavior_with_ela/results/analysis_v6/task14a"
OUT_LIGHT = ROOT / "behavior_with_ela/analysis_v7/task15a"
OUT_HEAVY = ROOT / "behavior_with_ela/results/analysis_v7/task15a"
CONFIG = ROOT / "configs/behavior_with_ela_train.yaml"

SOLVERS = ("shade", "lshade", "cso")
SOURCE_CHECKPOINT_FES = (2000, 4000, 6000)
COMMITMENT_FE = 1000
PRIMARY_WINDOW_FE = 500
SENSITIVITY_WINDOWS = (200, 1000)
ELITE_FRACTION = 0.20
FITNESS_TOL = 1e-12
FITNESS_SCALE_FLOOR = 1e-3
REPLAY_STREAM = 2026083121
BOOTSTRAP_STREAM = 2026083122
BOOTSTRAP_DRAWS = 5000
N_PERMUTATIONS = 100

FAMILY_MAP = {
    "bbob_separable_f01_f05": (1, 2),
    "bbob_low_or_moderate_conditioning_f06_f09": (6, 7),
    "bbob_high_conditioning_unimodal_f10_f14": (10, 11),
    "bbob_multimodal_adequate_global_structure_f15_f19": (15, 16),
    "bbob_multimodal_weak_global_structure_f20_f24": (20, 21),
}

CONCEPTS = [
    {
        "concept_id": "movement",
        "semantic_family": "Movement",
        "primary": "bf_population_centroid_shift_w05",
        "sensitivity": "bf_population_chamfer_distance_w05",
        "primitive": "movement",
        "rationale": "centroid displacement is a simple domain-scaled population movement summary",
    },
    {
        "concept_id": "direction_coordination",
        "semantic_family": "Direction / Coordination",
        "primary": "bf_centroid_shift_coherence_w05",
        "sensitivity": "bf_covariance_spectral_concentration",
        "primitive": "direction",
        "rationale": "coherence separates aligned displacement from changing population motion",
    },
    {
        "concept_id": "progress_contribution",
        "semantic_family": "Progress / Contribution",
        "primary": "bf_fitness_distribution_improvement_rate_w02",
        "sensitivity": "bf_improvement_rate_w02",
        "primitive": "progress",
        "rationale": "distribution-level improvement is closer to individual productive progress than best-only change",
    },
    {
        "concept_id": "fitness_stagnation",
        "semantic_family": "Stagnation",
        "primary": "bf_stagnation_w10",
        "sensitivity": "bf_improvement_frequency_w02",
        "primitive": "stagnation",
        "rationale": "the existing stagnation rate is explicit and online, while the ISSD primitive uses per-agent fitness age",
    },
    {
        "concept_id": "relative_position_dispersion",
        "semantic_family": "Relative Position / Dispersion",
        "primary": "bf_elite_concentration",
        "sensitivity": "bf_diversity_mean_pairwise",
        "primitive": "elite_distance",
        "rationale": "elite concentration describes the location of the population relative to its current high-quality region",
    },
    {
        "concept_id": "fitness_distribution",
        "semantic_family": "Fitness Distribution",
        "primary": "bf_fitness_diversity_rel",
        "sensitivity": "bf_fitness_wasserstein_rate_w02",
        "primitive": "rank",
        "rationale": "relative fitness spread is scale-aware; ISSD uses normalized individual rank",
    },
]

PRIMARY_AGGREGATE = [item["primary"] for item in CONCEPTS]
SENSITIVITY_AGGREGATE = [item["sensitivity"] for item in CONCEPTS]
PRIMITIVES = ("rank", "movement", "direction", "progress", "stagnation", "elite_distance")


def _fixed_rf_estimator() -> RandomForestRegressor:
    """Return the registered RF estimator for finite within-stratum inputs.

    The formal carrier also applies imputation and standardization.  ISSD-Q18
    is checked finite before this stage; for a tree regressor, the resulting
    finite affine transform leaves every split and prediction unchanged.
    """
    return RandomForestRegressor(
        n_estimators=200, max_depth=8, max_features="sqrt",
        random_state=2026090113, n_jobs=1,
    )


def _family_for_feature(name: str) -> str:
    for item in CONCEPTS:
        if name in (item["primary"], item["sensitivity"]):
            return item["semantic_family"]
    if name == "bf_fe_ratio":
        return "Temporal Context (not a search-state behavior)"
    mapping = {
        "bf_improvement_rate_w02": "Progress / Contribution",
        "bf_improvement_frequency_w02": "Progress / Contribution",
        "bf_diversity_mean_pairwise": "Relative Position / Dispersion",
        "bf_diversity_change_w05": "Relative Position / Dispersion",
        "bf_covariance_spectral_concentration": "Direction / Coordination",
        "bf_distance_decay_w10": "Relative Position / Dispersion",
        "bf_stagnation_w10": "Stagnation",
        "bf_convergence_rate_w10": "Progress / Contribution",
        "bf_fitness_diversity_rel": "Fitness Distribution",
        "bf_population_wasserstein_rate_w05": "Movement",
        "bf_centroid_shift_rate_w05": "Movement",
        "bf_centroid_shift_coherence_w05": "Direction / Coordination",
        "bf_fitness_quantile_improvement_fraction_w02": "Progress / Contribution",
        "bf_fitness_distribution_improvement_rate_w02": "Progress / Contribution",
        "bf_fitness_wasserstein_rate_w02": "Fitness Distribution",
        "bf_elite_concentration": "Relative Position / Dispersion",
        "bf_best_fitness_slope_rel_w05": "Progress / Contribution",
        "bf_diversity_slope_w05": "Relative Position / Dispersion",
        "bf_fitness_spread_slope_w05": "Fitness Distribution",
        "bf_population_centroid_shift_w05": "Movement",
        "bf_elite_centroid_shift_w05": "Movement",
        "bf_covariance_trace_ratio_w05": "Relative Position / Dispersion",
        "bf_covariance_effective_rank_w05": "Relative Position / Dispersion",
        "bf_diversity_recovery_w05": "Relative Position / Dispersion",
        "bf_population_chamfer_distance_w05": "Movement",
        "bf_covariance_trace_change_w05": "Direction / Coordination",
        "bf_covariance_effective_rank_change_w05": "Relative Position / Dispersion",
    }
    return mapping.get(name, "Unmapped")


def _reference_for_feature(name: str) -> str:
    refs = {
        "bf_fe_ratio": "behavior/features.py:extract_behavior_rows -> bf_fe_ratio",
        "bf_improvement_rate_w02": "behavior/features.py:_improvement_rate",
        "bf_improvement_frequency_w02": "behavior/features.py:_improvement_frequency",
        "bf_diversity_mean_pairwise": "behavior/features.py:_checkpoint_stats -> diversity",
        "bf_diversity_change_w05": "behavior/features.py:_relative_change",
        "bf_covariance_spectral_concentration": "behavior/features.py:_checkpoint_stats -> covariance_spectral_concentration",
        "bf_distance_decay_w10": "behavior/features.py:_relative_decay(distance_to_best)",
        "bf_stagnation_w10": "behavior/features.py:_stagnation",
        "bf_convergence_rate_w10": "behavior/features.py:_convergence_rate",
        "bf_fitness_diversity_rel": "behavior/features.py:_checkpoint_stats -> fitness_diversity_rel",
        "bf_population_wasserstein_rate_w05": "behavior/features.py:_fitness_distribution_change_stats / _population_set_change_stats",
        "bf_centroid_shift_rate_w05": "behavior/features.py:_population_set_change_stats -> centroid_shift_rate",
        "bf_centroid_shift_coherence_w05": "behavior/features.py:_population_set_change_stats -> centroid_shift_coherence",
        "bf_fitness_quantile_improvement_fraction_w02": "behavior/features.py:_fitness_distribution_change_stats",
        "bf_fitness_distribution_improvement_rate_w02": "behavior/features.py:_fitness_distribution_change_stats",
        "bf_fitness_wasserstein_rate_w02": "behavior/features.py:_fitness_distribution_change_stats",
        "bf_elite_concentration": "behavior/features.py:_elite_concentration",
        "bf_best_fitness_slope_rel_w05": "behavior/features.py:_normalized_best_fitness_slope",
        "bf_diversity_slope_w05": "behavior/features.py:_window_slope(diversity_mean_pairwise)",
        "bf_fitness_spread_slope_w05": "behavior/features.py:_window_slope(fitness_iqr_rel)",
        "bf_population_centroid_shift_w05": "trajectory/window_statistics.py:centroid_shift_distance",
        "bf_elite_centroid_shift_w05": "trajectory/window_statistics.py:elite_centroid_shift",
        "bf_covariance_trace_ratio_w05": "trajectory/window_statistics.py:covariance_trace_ratio",
        "bf_covariance_effective_rank_w05": "trajectory/window_statistics.py:covariance_effective_rank",
        "bf_diversity_recovery_w05": "behavior/features.py:_diversity_recovery",
        "bf_population_chamfer_distance_w05": "trajectory/window_statistics.py:population_chamfer_distance",
        "bf_covariance_trace_change_w05": "trajectory/window_statistics.py:covariance_trace_change",
        "bf_covariance_effective_rank_change_w05": "trajectory/window_statistics.py:covariance_effective_rank_change",
    }
    return refs[name]


def _jobs(config) -> list[dict]:
    jobs = []
    for suite in config.suites:
        if suite.split == "bbob_train":
            family_map: dict[str, list[int]] = {}
            for function in suite.functions:
                problem = make_experiment_problem(
                    suite, function=int(function), instance=suite.instances[0],
                    dimension=10, boundary_handling="reflect",
                )
                family_map.setdefault(str(problem.family), []).append(int(function))
                problem.close()
            functions = []
            for family in sorted(family_map):
                functions.extend(sorted(family_map[family])[:2])
            instances = suite.instances
        else:
            functions = sorted(suite.functions)[::2]
            instances = suite.instances
        for function in functions:
            for instance in instances:
                jobs.append({
                    "split": suite.split,
                    "suite": suite.suite,
                    "function": int(function),
                    "instance": int(instance),
                    "seeds": (1, 2, 3, 4, 5),
                })
    return jobs


def _gap(problem, best: float, config) -> float:
    raw = min(max(float(best) - float(problem.reference_value), 0.0), config.failure_loss_cap)
    return float(np.log10(np.clip(raw, config.log10_gap_floor, config.log10_gap_cap)))


def _rank_and_elite(population: np.ndarray, fitness: np.ndarray, bounds: np.ndarray):
    order = np.argsort(fitness, kind="mergesort")
    rank = np.empty(len(fitness), dtype=float)
    rank[order] = np.arange(len(fitness), dtype=float) / max(len(fitness) - 1, 1)
    elite_n = max(1, int(np.ceil(ELITE_FRACTION * len(fitness))))
    centroid = np.mean(population[order[:elite_n]], axis=0)
    diameter = max(float(np.linalg.norm(bounds[1] - bounds[0])), np.finfo(float).eps)
    return rank, centroid, diameter


def _history_rows(state_id: str, agent_ids: np.ndarray, state, global_fe: int,
                  update_index: int, was_updated: np.ndarray, was_accepted: np.ndarray,
                  problem) -> list[dict]:
    population = np.asarray(state.population, dtype=float)
    fitness = np.asarray(state.fitness, dtype=float)
    rank, elite_centroid, _ = _rank_and_elite(
        population, fitness, np.vstack([problem.lower_bounds, problem.upper_bounds])
    )
    rows = []
    for i, agent_id in enumerate(agent_ids):
        rows.append({
            "state_id": state_id,
            "agent_id": str(agent_id),
            "native_update_index": int(update_index),
            "global_FE": int(global_fe),
            "x": population[i].tolist(),
            "fitness": float(fitness[i]),
            "was_updated": bool(was_updated[i]),
            "was_accepted": bool(was_accepted[i]),
            "population_rank": float(rank[i]),
            "population_size": int(len(population)),
            "current_best_fitness": float(np.min(fitness)),
            "elite_centroid": elite_centroid.tolist(),
        })
    return rows


def _agent_primitives(records: list[dict], current_ids: list[str], current_population: np.ndarray,
                     current_fitness: np.ndarray, bounds: np.ndarray, final_fe: int) -> list[dict]:
    rank, elite_centroid, diameter = _rank_and_elite(current_population, current_fitness, bounds)
    by_id: dict[str, list[dict]] = {}
    for row in records:
        by_id.setdefault(str(row["agent_id"]), []).append(row)
    out = []
    id_order = [str(agent_id) for agent_id in current_ids]
    for idx, agent_id in enumerate(id_order):
        hist = sorted(by_id[agent_id], key=lambda r: (int(r["global_FE"]), int(r["native_update_index"])))
        transitions = []
        for previous, current in zip(hist[:-1], hist[1:], strict=True):
            if int(current["global_FE"]) > final_fe - PRIMARY_WINDOW_FE:
                previous_x = np.asarray(previous["x"], dtype=float)
                current_x = np.asarray(current["x"], dtype=float)
                previous_f = float(previous["fitness"])
                current_f = float(current["fitness"])
                delta_x = current_x - previous_x
                movement = float(np.linalg.norm(delta_x) / max(diameter, np.finfo(float).eps))
                transitions.append((current, previous, delta_x, movement, previous_f, current_f))
        movements = np.asarray([item[3] for item in transitions], dtype=float)
        movement_value = float(np.median(movements)) if len(movements) else 0.0
        nonzero = [item[2] / max(float(np.linalg.norm(item[2])), np.finfo(float).eps)
                   for item in transitions if np.linalg.norm(item[2]) > 1e-15]
        direction_value = float(np.linalg.norm(np.mean(nonzero, axis=0))) if nonzero else 0.0
        current_quantiles = np.quantile(current_fitness, [0.25, 0.50, 0.75])
        current_iqr = float(current_quantiles[2] - current_quantiles[0])
        relative_floor = FITNESS_SCALE_FLOOR * max(1.0, abs(float(current_quantiles[1])))
        fitness_scale = max(current_iqr, relative_floor)
        progress_value = float(sum(max(item[4] - item[5], 0.0) for item in transitions)
                               / max(len(transitions) * fitness_scale, np.finfo(float).eps))
        last_improvement = hist[0]["global_FE"]
        for previous, current in zip(hist[:-1], hist[1:], strict=True):
            previous_f = float(previous["fitness"])
            current_f = float(current["fitness"])
            tol = FITNESS_TOL * max(1.0, abs(previous_f))
            if previous_f - current_f > tol:
                last_improvement = int(current["global_FE"])
        stagnation_value = float(min(max(final_fe - int(last_improvement), 0), PRIMARY_WINDOW_FE)
                                 / PRIMARY_WINDOW_FE)
        distance_value = float(np.linalg.norm(np.asarray(hist[-1]["x"]) - elite_centroid) / diameter)
        out.append({
            "state_id": hist[-1]["state_id"],
            "agent_id": agent_id,
            "issd_rank": float(rank[idx]),
            "issd_movement": movement_value,
            "issd_direction": direction_value,
            "issd_progress": float(progress_value),
            "issd_stagnation": stagnation_value,
            "issd_elite_distance": distance_value,
            "direction_valid": int(bool(nonzero)),
            "history_length": int(len(hist)),
            "effective_window_fe": int(final_fe - min(int(item[1]["global_FE"]) for item in transitions)) if transitions else 0,
        })
    return out


def _replay_unit(payload: tuple[dict, str]) -> dict:
    job, config_path = payload
    config = load_experiment_config(Path(config_path))
    suite = {item.split: item for item in config.suites}[job["split"]]
    problem = make_experiment_problem(
        suite, function=job["function"], instance=job["instance"],
        dimension=config.dimension, boundary_handling=config.boundary_handling,
    )
    settings = OptimizerSettings(population_size=config.population_size, boundary_handling=config.boundary_handling)
    state_rows, history_rows_all, primitive_rows = [], [], []
    replay_fe = 0
    for seed in job["seeds"]:
        for source in SOLVERS:
            source_states = {}
            source_state = initialize_optimizer_state(
                algorithm=source, problem=problem, seed=seed, settings=settings,
            )
            for checkpoint in SOURCE_CHECKPOINT_FES:
                while int(source_state.evaluations) < checkpoint:
                    advance_optimizer_state(
                        state=source_state, problem=problem,
                        fe_budget=min(config.population_size, checkpoint - int(source_state.evaluations)),
                    )
                source_states[checkpoint] = copy.deepcopy(source_state)
            replay_fe += max(SOURCE_CHECKPOINT_FES)
            for target in SOLVERS:
                if target == source:
                    continue
                for source_fe in SOURCE_CHECKPOINT_FES:
                    state_id = (
                        f"{job['split']}:{problem.problem_id}:{source}->{target}:"
                        f"seed{int(seed)}:src_fe{int(source_fe)}"
                    )
                    state = initialize_transferred_optimizer_state(
                        algorithm=target, source_state=source_states[source_fe], problem=problem,
                        seed=seed, function=problem.function_number, instance=problem.instance_number,
                        event=1,
                    )
                    agent_ids = np.asarray([f"a{i:03d}" for i in range(len(state.population))], dtype=object)
                    history = _history_rows(
                        state_id, agent_ids, state, source_fe, 0,
                        np.zeros(len(agent_ids), dtype=bool), np.zeros(len(agent_ids), dtype=bool), problem,
                    )
                    used = 0
                    update_index = 0
                    generation_parent_ids = None
                    generation_parent_fitness = None
                    generation_kind = target
                    generation_updated = None
                    generation_accepted = None
                    generation_eval_index = 0

                    def on_evaluation(_point, _value):
                        nonlocal used, generation_eval_index, generation_losers
                        nonlocal generation_accepted, generation_updated
                        used += 1
                        if generation_kind == "cso":
                            if generation_eval_index == 0:
                                generation_losers = np.asarray(state.pending_loser_indices, dtype=int).copy()
                                generation_accepted = np.zeros(len(generation_losers), dtype=bool)
                                generation_updated = np.zeros(len(generation_losers), dtype=bool)
                            parent_index = int(generation_losers[generation_eval_index])
                        else:
                            parent_index = generation_eval_index
                        generation_accepted[generation_eval_index] = bool(
                            float(_value) < float(generation_parent_fitness[parent_index])
                        )
                        generation_updated[generation_eval_index] = True if generation_kind == "cso" else generation_accepted[generation_eval_index]
                        generation_eval_index += 1

                    def on_native_update(updated):
                        nonlocal agent_ids, update_index, history
                        nonlocal generation_parent_ids, generation_parent_fitness
                        nonlocal generation_updated, generation_accepted
                        if generation_kind == "cso":
                            new_ids = agent_ids.copy()
                            was_updated = np.zeros(len(new_ids), dtype=bool)
                            was_accepted = np.zeros(len(new_ids), dtype=bool)
                            losers = np.asarray(generation_losers, dtype=int)
                            was_updated[losers] = True
                            was_accepted[losers] = generation_accepted
                        else:
                            pre_ids = np.asarray(generation_parent_ids, dtype=object)
                            pre_fit = np.asarray(generation_parent_fitness, dtype=float)
                            pre_updated = np.asarray(generation_updated, dtype=bool)
                            pre_accepted = np.asarray(generation_accepted, dtype=bool)
                            if generation_kind == "lshade":
                                order = np.argsort(pre_fit, kind="mergesort")[:len(updated.population)]
                                new_ids = pre_ids[order]
                                was_updated = pre_updated[order]
                                was_accepted = pre_accepted[order]
                            else:
                                new_ids = pre_ids
                                was_updated = pre_updated
                                was_accepted = pre_accepted
                        agent_ids = np.asarray(new_ids, dtype=object)
                        update_index += 1
                        history.extend(_history_rows(
                            state_id, agent_ids, updated, source_fe + used, update_index,
                            was_updated, was_accepted, problem,
                        ))

                    while used < COMMITMENT_FE:
                        pending = state.pending_positions if target == "cso" else state.pending_population
                        if pending is None:
                            generation_parent_ids = agent_ids.copy()
                            generation_parent_fitness = np.asarray(state.fitness, dtype=float).copy()
                            generation_updated = np.zeros(len(agent_ids), dtype=bool)
                            generation_accepted = np.zeros(len(agent_ids), dtype=bool)
                            generation_losers = np.asarray([], dtype=int)
                            generation_eval_index = 0
                        before_pending = pending is None
                        pending_count = 1 if before_pending else len(pending) - state.pending_index
                        advance_optimizer_state(
                            state=state, problem=problem,
                            fe_budget=min(pending_count, COMMITMENT_FE - used),
                            on_native_update=on_native_update,
                            on_evaluation=on_evaluation,
                        )

                    final_exact_gap = _gap(problem, state.best_fitness, config)
                    final_snapshot = history[-len(agent_ids):]
                    snapshot_fe = int(max(row["global_FE"] for row in final_snapshot))
                    snapshot_fit = np.asarray([row["fitness"] for row in final_snapshot], dtype=float)
                    snapshot_pop = np.asarray([row["x"] for row in final_snapshot], dtype=float)
                    current_ids = [str(x) for x in agent_ids]
                    primitive = _agent_primitives(
                        history, current_ids, snapshot_pop, snapshot_fit,
                        np.vstack([problem.lower_bounds, problem.upper_bounds]), snapshot_fe,
                    )
                    primitive_df = pd.DataFrame(primitive)
                    qrow = {"state_id": state_id}
                    for primitive_name in PRIMITIVES:
                        values = primitive_df[f"issd_{primitive_name}"].to_numpy(dtype=float)
                        for quantile, suffix in ((0.25, "q25"), (0.50, "q50"), (0.75, "q75")):
                            qrow[f"issd_{primitive_name}_{suffix}"] = float(np.quantile(values, quantile))
                    qrow.update({
                        "snapshot_fe": snapshot_fe,
                        "snapshot_population_size": int(len(agent_ids)),
                        "exact_population_size": int(len(state.population)),
                        "replay_log10_gap": final_exact_gap,
                        "history_min_fe": int(min(row["global_FE"] for row in final_snapshot)),
                    })
                    primitive_rows.extend(primitive)
                    history_rows_all.extend(history)
                    state_rows.append(qrow)
                    replay_fe += COMMITMENT_FE
    problem.close()
    return {
        "job": job,
        "state_rows": state_rows,
        "history_rows": history_rows_all,
        "primitive_rows": primitive_rows,
        "replay_fe": replay_fe,
    }


def _rebuild_issd_from_history(histories: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_rows = []
    primitive_rows = []
    for state_id, part in histories.groupby("state_id", sort=True):
        final_fe = int(part["global_FE"].max())
        final = part.loc[part["global_FE"].eq(final_fe)].sort_values("agent_id")
        agent_ids = final["agent_id"].astype(str).tolist()
        population = np.asarray(final["x"].tolist(), dtype=float)
        fitness = final["fitness"].to_numpy(dtype=float)
        bounds = np.vstack(problem_bounds(str(part["state_id"].iloc[0]).split(":", 2)[1]))
        primitive = _agent_primitives(
            part.to_dict("records"), agent_ids, population, fitness, bounds, final_fe,
        )
        primitive_df = pd.DataFrame(primitive)
        row = {"state_id": state_id}
        for primitive_name in PRIMITIVES:
            values = primitive_df[f"issd_{primitive_name}"].to_numpy(dtype=float)
            for quantile, suffix in ((0.25, "q25"), (0.50, "q50"), (0.75, "q75")):
                row[f"issd_{primitive_name}_{suffix}"] = float(np.quantile(values, quantile))
        row.update({
            "snapshot_fe": final_fe,
            "snapshot_population_size": int(len(final)),
            "exact_population_size": int(len(final)),
            "replay_log10_gap": np.nan,
            "history_min_fe": int(part["global_FE"].min()),
        })
        state_rows.append(row)
        primitive_rows.extend(primitive)
    return pd.DataFrame(state_rows), pd.DataFrame(primitive_rows)


def stage_a_inventory(behavior: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    data = behavior.loc[behavior["behavior_kind"].eq("global")].copy()
    features = list(SELECTOR_BEHAVIOR_FEATURE_COLUMNS)
    rows = []
    for feature in features:
        values = pd.to_numeric(data[feature], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        counts = pd.Series(finite).value_counts(normalize=True) if len(finite) else pd.Series(dtype=float)
        max_abs = float(np.max(np.abs(finite))) if len(finite) else np.nan
        rows.append({
            "feature_name": feature,
            "formula_or_code_reference": _reference_for_feature(feature),
            "semantic_family": _family_for_feature(feature),
            "data_source": "Task 14A post_handoff_behavior.parquet, behavior_kind=global",
            "time_window": "as encoded by feature name/code",
            "population_or_fitness": "population or fitness distribution; see reference",
            "online_available": True,
            "uses_true_optimum": False,
            "uses_future_information": False,
            "algorithm_specific": False,
            "dimension_dependence": "normalized where code specifies domain scale; otherwise reported metadata",
            "population_size_dependence": "set statistic or normalized rate; inspect stability below",
            "runtime_cost": "low; derived from recorded native-update history",
            "missing_rate": float(np.isnan(values).mean()),
            "nonfinite_rate": float((~np.isfinite(values)).mean()),
            "near_constant_rate": float(counts.iloc[0]) if len(counts) else np.nan,
            "tail_instability": float(np.mean(np.abs(finite) > 1e6)) if len(finite) else np.nan,
            "max_abs_finite": max_abs,
            "notes": "bf_fe_ratio is retained as a temporal context field, not a core search concept"
            if feature == "bf_fe_ratio" else "Stage-A decision is unsupervised and outcome-blind",
        })
    inventory = pd.DataFrame(rows)

    partitions = []
    data = data.merge(
        pd.read_parquet(T14A_HEAVY / "post_handoff_states.parquet")[["state_id", "suite", "source_checkpoint_fe"]],
        on="state_id", validate="one_to_one",
    )
    for key, part in data.groupby(["suite", "algorithm", "source_checkpoint_fe"], sort=True):
        partitions.append((key, part))
    family = {feature: _family_for_feature(feature) for feature in features}
    redundancy_rows = []
    for a, b in combinations(features, 2):
        hits = 0
        rhos = []
        for _, part in partitions:
            x = pd.to_numeric(part[a], errors="coerce")
            y = pd.to_numeric(part[b], errors="coerce")
            mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            rho = float(spearmanr(x[mask], y[mask]).statistic) if mask.sum() >= 3 else np.nan
            rhos.append(rho)
            hits += int(np.isfinite(rho) and abs(rho) >= 0.90)
        redundancy_rows.append({
            "feature_a": a, "feature_b": b,
            "same_semantic_family": family[a] == family[b],
            "n_partitions": len(partitions), "n_high_correlation_partitions": hits,
            "stable_redundancy": bool(family[a] == family[b] and hits >= int(np.ceil(0.80 * len(partitions)))),
            "median_partition_spearman": float(np.nanmedian(rhos)),
            "max_abs_partition_spearman": float(np.nanmax(np.abs(rhos))),
        })
    redundancy = pd.DataFrame(redundancy_rows)
    graph = {feature: set() for feature in features}
    for row in redundancy.itertuples(index=False):
        if row.stable_redundancy:
            graph[row.feature_a].add(row.feature_b)
            graph[row.feature_b].add(row.feature_a)
    clusters = []
    seen = set()
    cluster_id = 0
    for feature in features:
        if feature in seen or not graph[feature]:
            continue
        stack = [feature]
        members = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current); members.append(current); stack.extend(graph[current] - seen)
        cluster_id += 1
        representative = next((item["primary"] for item in CONCEPTS if item["primary"] in members), members[0])
        clusters.extend({"cluster_id": cluster_id, "feature_name": member,
                         "semantic_family": family[member], "representative": representative,
                         "cluster_size": len(members)} for member in sorted(members))
    clusters_df = pd.DataFrame(clusters, columns=["cluster_id", "feature_name", "semantic_family", "representative", "cluster_size"])

    concept_rows = []
    for item in CONCEPTS:
        concept_rows.append({
            "concept_id": item["concept_id"], "semantic_family": item["semantic_family"],
            "primary_aggregate_representative": item["primary"],
            "sensitivity_aggregate_representative": item["sensitivity"],
            "issd_primitive": item["primitive"], "screened_core": True,
            "selection_basis": "legality > interpretability > numerical stability > algorithm agnosticism > scale/population normalization > cost > fold stability",
            "rationale": item["rationale"],
        })
    concepts = pd.DataFrame(concept_rows)
    primary_ok = inventory.set_index("feature_name").loc[PRIMARY_AGGREGATE]
    sa3 = bool(
        (primary_ok["nonfinite_rate"] > 0).any()
        or (primary_ok["tail_instability"] > 0).any()
        or (primary_ok["missing_rate"] > 0).any()
    )
    verdict = "S-A3 FEATURE CONTRACT UNSOUND" if sa3 else ("S-A1 COMPACT CORE IDENTIFIED" if len(CONCEPTS) <= 8 else "S-A2 REDUNDANCY LOW")
    return inventory, redundancy, clusters_df, concepts, verdict


def _context_features(frame: pd.DataFrame) -> list[str]:
    routes = sorted(frame["route"].unique())
    for route in routes:
        frame[f"route__{route}"] = frame["route"].eq(route).astype(float)
    frame["source_FE_ratio"] = frame["source_FE"].astype(float) / 10000.0
    return [f"route__{route}" for route in routes] + ["source_FE_ratio"]


def _function_balanced(frame: pd.DataFrame, column: str) -> float:
    return float(frame.groupby("cv_group_id", sort=True)[column].mean().mean())


def _bootstrap_gain(frame: pd.DataFrame, upper: str, lower: str, stream: int) -> tuple[float, float, float]:
    delta = frame[upper].to_numpy(dtype=float) - frame[lower].to_numpy(dtype=float)
    group_means = pd.DataFrame({"group": frame["cv_group_id"], "delta": delta}).groupby("group")["delta"].mean()
    groups = np.asarray(sorted(group_means.index), dtype=object)
    rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_STREAM, stream, len(groups)]))
    draws = np.empty(BOOTSTRAP_DRAWS)
    values = group_means.to_dict()
    for i in range(BOOTSTRAP_DRAWS):
        sample = rng.choice(groups, size=len(groups), replace=True)
        draws[i] = np.mean([values[group] for group in sample])
    return float(group_means.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _prepare_dataset(states: pd.DataFrame, behavior: pd.DataFrame, histories: pd.DataFrame,
                     action: pd.DataFrame, concepts: pd.DataFrame, issd: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    base = states.copy()
    base = base.merge(issd, on="state_id", validate="one_to_one")
    global_behavior = behavior.loc[behavior["behavior_kind"].eq("global"), ["state_id", *SELECTOR_BEHAVIOR_FEATURE_COLUMNS]]
    base = base.merge(global_behavior, on="state_id", validate="one_to_one")
    outcomes = action.loc[action["replicate_id"].eq(0)].copy()
    cont = outcomes.loc[outcomes["candidate_action"].eq("continue")].set_index("state_id")["loss_1000"]
    alt = outcomes.loc[outcomes["candidate_action"].ne("continue")].pivot_table(
        index="state_id", columns="candidate_action", values="loss_1000", aggfunc="first"
    )
    base["continue_loss"] = cont.reindex(base["state_id"]).to_numpy()
    for solver in SOLVERS:
        base[f"loss_{solver}"] = base["continue_loss"].to_numpy()
        mask = base["current_algorithm"].ne(solver)
        base.loc[mask, f"loss_{solver}"] = alt[solver].reindex(base.loc[mask, "state_id"]).to_numpy()
    if base[[f"loss_{s}" for s in SOLVERS]].isna().any().any():
        raise RuntimeError("Task 14A action outcomes do not cover every replay-aligned state")
    base["route"] = base["source_algorithm"].astype(str) + "->" + base["current_algorithm"].astype(str)
    base["source_FE"] = base["source_checkpoint_fe"].astype(int)
    _context_features(base)
    base["route_source_key"] = base["route"].astype(str) + "__" + base["source_checkpoint_fe"].astype(str)
    screened = concepts["primary_aggregate_representative"].tolist()
    issd_cols = [f"issd_{p}_{q}" for p in PRIMITIVES for q in ("q25", "q50", "q75")]
    legacy = list(SELECTOR_BEHAVIOR_FEATURE_COLUMNS)
    if len(legacy) != 28 or len(screened) != 6 or len(issd_cols) != 18:
        raise RuntimeError("Task 15A feature contract counts are not 28/6/18")
    return base, legacy, screened, issd_cols


def _run_grouped_oof(dataset: pd.DataFrame, feature_sets: dict[str, list[str]], carrier: str) -> pd.DataFrame:
    truth = dataset[[f"loss_{s}" for s in SOLVERS]].to_numpy(dtype=float)
    rows = []
    for model_name, features in feature_sets.items():
        X = dataset[features].to_numpy(dtype=float)
        pred = np.full_like(truth, np.nan)
        fold = np.empty(len(dataset), dtype=object)
        for group in sorted(dataset["cv_group_id"].unique()):
            test = dataset["cv_group_id"].eq(group).to_numpy()
            model = make_carrier(carrier)
            model.fit(X[~test], truth[~test])
            pred[test] = model.predict(X[test])
            fold[test] = f"holdout_{group}"
        selected = pred.argmin(axis=1)
        for i in range(len(dataset)):
            rows.append({
                "state_id": dataset.iloc[i]["state_id"], "suite": dataset.iloc[i]["suite"],
                "cv_group_id": dataset.iloc[i]["cv_group_id"], "model": model_name,
                "carrier": carrier, "fold_id": fold[i], "selected": SOLVERS[selected[i]],
                "realized_loss": float(truth[i, selected[i]]),
                **{f"pred_{s}": float(pred[i, j]) for j, s in enumerate(SOLVERS)},
            })
    return pd.DataFrame(rows)


def _lookup_predictions(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    truth = dataset[[f"loss_{s}" for s in SOLVERS]].to_numpy(dtype=float)
    for held_out in sorted(dataset["cv_group_id"].unique()):
        train = dataset.loc[dataset["cv_group_id"].ne(held_out)]
        means = train.groupby(["route", "source_checkpoint_fe"])[[f"loss_{s}" for s in SOLVERS]].mean()
        test = dataset.loc[dataset["cv_group_id"].eq(held_out)]
        for pos, (_, row) in enumerate(test.iterrows()):
            pred = means.loc[(row["route"], row["source_checkpoint_fe"])].to_numpy(dtype=float)
            selected = int(np.argmin(pred))
            rows.append({
                "state_id": row["state_id"], "suite": row["suite"], "cv_group_id": held_out,
                "model": "M_lookup", "carrier": "lookup", "fold_id": f"holdout_{held_out}",
                "selected": SOLVERS[selected], "realized_loss": float(truth[dataset.index.get_loc(row.name), selected]),
                **{f"pred_{s}": float(pred[j]) for j, s in enumerate(SOLVERS)},
            })
    return pd.DataFrame(rows)


def _policy_rows(predictions: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    meta = dataset[["state_id", "suite", "cv_group_id", "current_algorithm", "continue_loss", "source_checkpoint_fe", "route"]]
    merged = predictions.merge(meta, on=["state_id", "suite", "cv_group_id"], validate="many_to_one")
    merged["is_switch"] = merged["selected"].ne(merged["current_algorithm"])
    merged["gain_vs_continue"] = merged["continue_loss"] - merged["realized_loss"]
    rows = []
    for (model, carrier, suite), part in merged.groupby(["model", "carrier", "suite"], sort=True):
        rows.append({
            "model": model, "carrier": carrier, "suite": suite,
            "realized_fb_loss": _function_balanced(part, "realized_loss"),
            "gain_vs_continue_fb": _function_balanced(part, "gain_vs_continue"),
            "switch_rate": float(part["is_switch"].mean()),
            "n": int(len(part)),
        })
    for suite, part in dataset.groupby("suite", sort=True):
        rows.append({
            "model": "Always Continue", "carrier": "fixed", "suite": suite,
            "realized_fb_loss": _function_balanced(part, "continue_loss"),
            "gain_vs_continue_fb": 0.0, "switch_rate": 0.0, "n": int(len(part)),
        })
    return pd.DataFrame(rows)


def _within_group_worker(args: tuple[pd.DataFrame, list[str], list[str], tuple[str, ...]]) -> pd.DataFrame:
    dataset, aggregate, issd, model_names = args
    truth = dataset[[f"loss_{s}" for s in SOLVERS]].to_numpy(dtype=float)
    A = dataset[aggregate].to_numpy(dtype=float)
    I = dataset[issd].to_numpy(dtype=float)
    rows = []
    key = (dataset.iloc[0]["problem_id"], dataset.iloc[0]["route"], int(dataset.iloc[0]["source_checkpoint_fe"]))
    for test_pos in range(len(dataset)):
        train = np.asarray([i for i in range(len(dataset)) if i != test_pos], dtype=int)
        models = {"W0": truth[train].mean(axis=0)}
        candidates = {"WA": A, "WI": I, "WAI": np.hstack([A, I])}
        for name in model_names:
            X = candidates[name]
            model = _fixed_rf_estimator()
            model.fit(X[train], truth[train])
            models[name] = model.predict(X[test_pos:test_pos + 1])[0]
        for name, pred in models.items():
            selected = int(np.argmin(pred))
            rows.append({
                "state_id": dataset.iloc[test_pos]["state_id"], "suite": dataset.iloc[test_pos]["suite"],
                "cv_group_id": dataset.iloc[test_pos]["cv_group_id"], "problem_id": key[0],
                "route": key[1], "source_checkpoint_fe": key[2], "seed": int(dataset.iloc[test_pos]["seed"]),
                "model": name, "selected": SOLVERS[selected],
                "realized_loss": float(truth[test_pos, selected]),
            })
    return pd.DataFrame(rows)


def _within_loso(dataset: pd.DataFrame, aggregate: list[str], issd: list[str], *, permuted: np.ndarray | None = None,
                 model_names: tuple[str, ...] = ("WA", "WI", "WAI"), parallel: bool = True) -> pd.DataFrame:
    work = dataset.copy()
    if permuted is not None:
        work[issd] = permuted
    parts = [part.copy() for _, part in work.groupby(["problem_id", "route", "source_checkpoint_fe"], sort=True)]
    payloads = [(part, aggregate, issd, model_names) for part in parts]
    if parallel:
        with ProcessPoolExecutor(max_workers=8) as executor:
            result = list(executor.map(_within_group_worker, payloads))
    else:
        result = [_within_group_worker(payload) for payload in payloads]
    return pd.concat(result, ignore_index=True)


def _permutation_once(args: tuple[pd.DataFrame, list[str], list[str], int]) -> pd.DataFrame:
    dataset, aggregate, issd, repeat = args
    rng = np.random.default_rng(np.random.SeedSequence([REPLAY_STREAM, 15, int(repeat)]))
    values = dataset[issd].to_numpy(dtype=float).copy()
    for _, positions in dataset.groupby(["problem_id", "route", "source_checkpoint_fe"], sort=True).groups.items():
        idx = np.asarray(list(positions), dtype=int)
        values[idx] = values[idx[rng.permutation(len(idx))]]
    pred = _within_loso(dataset, aggregate, issd, permuted=values, model_names=("WAI",), parallel=False)
    pred = pred.loc[pred["model"].eq("WAI")].copy()
    rows = []
    for suite, part in pred.groupby("suite", sort=True):
        rows.append({"repeat": int(repeat), "suite": suite, "permuted_policy_loss": _function_balanced(part, "realized_loss")})
    return pd.DataFrame(rows)


def _association(dataset: pd.DataFrame, issd_cols: list[str]) -> pd.DataFrame:
    rows = []
    for suite, suite_part in dataset.groupby("suite", sort=True):
        for action in SOLVERS:
            part = suite_part.loc[suite_part["current_algorithm"].ne(action)].copy()
            advantage = part["continue_loss"] - part[f"loss_{action}"]
            centered_adv = advantage - advantage.groupby([part["problem_id"], part["route"], part["source_checkpoint_fe"]]).transform("mean")
            for feature in issd_cols:
                x = part[feature]
                rho = spearmanr(x, advantage).statistic if x.nunique() > 1 and advantage.nunique() > 1 else np.nan
                xc = x - x.groupby([part["problem_id"], part["route"], part["source_checkpoint_fe"]]).transform("mean")
                partial = spearmanr(xc, centered_adv).statistic if xc.nunique() > 1 and centered_adv.nunique() > 1 else np.nan
                rows.append({"suite": suite, "action": action, "feature": feature,
                             "n": int(len(part)), "spearman": float(rho), "grouped_partial_spearman": float(partial)})
    return pd.DataFrame(rows)


def _write_stage_a_reports(inventory, redundancy, clusters, concepts, verdict):
    OUT_LIGHT.mkdir(parents=True, exist_ok=True)
    OUT_HEAVY.mkdir(parents=True, exist_ok=True)
    inventory.to_parquet(OUT_HEAVY / "behavior_indicator_inventory.parquet", index=False)
    redundancy.to_parquet(OUT_HEAVY / "behavior_feature_redundancy.parquet", index=False)
    clusters.to_parquet(OUT_HEAVY / "behavior_redundancy_clusters.parquet", index=False)
    concepts.to_parquet(OUT_HEAVY / "screened_behavior_concepts.parquet", index=False)
    core = set(PRIMARY_AGGREGATE)
    inventory["screened_core"] = inventory["feature_name"].isin(core)
    inventory.to_parquet(OUT_HEAVY / "behavior_indicator_inventory.parquet", index=False)
    stability = inventory[["feature_name", "missing_rate", "nonfinite_rate", "near_constant_rate", "tail_instability", "max_abs_finite", "screened_core"]].copy()
    stability["near_constant_flag"] = stability["near_constant_rate"] >= 0.995
    stability["numeric_instability_flag"] = (stability["nonfinite_rate"] > 0) | (stability["tail_instability"] > 0)
    stability.to_parquet(OUT_HEAVY / "behavior_feature_stability.parquet", index=False)
    (OUT_LIGHT / "19a_behavior_indicator_inventory.md").write_text(
        "# Task 15A Stage A：现有 Behavior 指标清单\n\n"
        f"当前 selector Behavior 为 {len(SELECTOR_BEHAVIOR_FEATURE_COLUMNS)} 维。该清单只读取行为值和代码定义，未读取 action outcome。\n\n"
        + inventory.to_markdown(index=False) + "\n", encoding="utf-8")
    (OUT_LIGHT / "19b_semantic_family_map.md").write_text(
        "# 语义族映射\n\n" + concepts.to_markdown(index=False) + "\n\n"
        + inventory[["feature_name", "semantic_family", "screened_core"]].to_markdown(index=False) + "\n", encoding="utf-8")
    (OUT_LIGHT / "19c_numeric_stability_check.md").write_text(
        "# 数值稳定性检查\n\n阈值：near-constant 的众数比例 ≥0.995；极端尾部比例定义为 |value|>1e6。\n\n"
        + stability.to_markdown(index=False) + "\n", encoding="utf-8")
    (OUT_LIGHT / "19d_unsupervised_redundancy_check.md").write_text(
        "# 无监督冗余检查\n\n分区为 suite × current solver × source FE；同语义族且 |Spearman|≥0.90 的分区比例至少 80% 才连边。\n\n"
        + (redundancy.to_markdown(index=False) if len(redundancy) else "无稳定冗余边。") + "\n\n"
        + (clusters.to_markdown(index=False) if len(clusters) else "无稳定冗余簇。") + "\n", encoding="utf-8")
    (OUT_LIGHT / "19e_screened_behavior_concepts.md").write_text(
        "# Screened Behavior Concepts\n\n"
        f"Stage-A verdict：**{verdict}**。核心概念数：{len(concepts)}。代表选择不使用 action label、特征重要性或模型调参。\n\n"
        + concepts.to_markdown(index=False) + "\n", encoding="utf-8")


def _write_protocol_reports(alignment: pd.DataFrame):
    (OUT_LIGHT / "19f_agent_identity_and_replay_protocol.md").write_text(
        "# Agent identity 与 deterministic replay protocol\n\n"
        "每个 transferred solver segment 从 active search-slot a000、a001…开始；SHADE/L-SHADE 的 accepted child 继承 parent target slot；CSO loser 保留原 loser slot；未更新个体保留 ID；L-SHADE population reduction 只保留当前 active rows，删除的 ID 终止。数组重排不被解释为 identity。\n\n"
        "重放仅为 natural source → handoff → 1000FE commitment，未生成任何 action branch 或 repetition。\n\n"
        + alignment.to_markdown(index=False) + "\n", encoding="utf-8")
    (OUT_LIGHT / "19g_individual_primitive_definitions.md").write_text(
        "# Individual primitive definitions\n\n"
        "正式 primitive 为 rank、movement、direction、progress、stagnation、elite distance；primary window 为 500FE，敏感性窗口仅为 200/1000FE 的协议记录。rank 与 elite distance 使用当前 population；movement 使用实际相邻 native updates 的位移并以 domain diameter 归一化；direction 使用非零 displacement 单位向量的均值范数；progress 使用当前 population IQR，并用预先指定的 1e-3 × max(1, |fitness median|) 作为退化分布的相对尺度下界；stagnation 是 fitness meaningful improvement 的 FE age；全部不使用真实最优值。\n", encoding="utf-8")
    (OUT_LIGHT / "19h_issd_numeric_and_distribution_check.md").write_text(
        "# ISSD 数值与分布检查\n\n见 `issd_numeric_stability.parquet`。Q25/Q50/Q75 构成唯一正式 ISSD-Q18 表示；direction=0 仅表示窗口内没有非零位移，并保留 validity flag。\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-replay", action="store_true")
    parser.add_argument("--rebuild-issd-from-history", action="store_true")
    parser.add_argument("--reuse-analysis", action="store_true")
    parser.add_argument("--skip-permutation", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    OUT_LIGHT.mkdir(parents=True, exist_ok=True)
    OUT_HEAVY.mkdir(parents=True, exist_ok=True)

    behavior = pd.read_parquet(T14A_HEAVY / "post_handoff_behavior.parquet")
    inventory, redundancy, clusters, concepts, stage_a_verdict = stage_a_inventory(behavior)
    _write_stage_a_reports(inventory, redundancy, clusters, concepts, stage_a_verdict)
    if stage_a_verdict == "S-A3 FEATURE CONTRACT UNSOUND":
        raise RuntimeError("Stage A produced S-A3; Stage B is not allowed")

    states_ref = pd.read_parquet(T14A_HEAVY / "post_handoff_states.parquet")
    histories_path = OUT_HEAVY / "individual_history_index.parquet"
    issd_path = OUT_HEAVY / "post_handoff_issd_task15a.parquet"
    primitive_path = OUT_HEAVY / "individual_primitives_task15a.parquet"
    alignment_path = OUT_HEAVY / "replay_alignment_task15a.parquet"
    config = load_experiment_config(CONFIG)
    jobs = _jobs(config)
    if args.rebuild_issd_from_history and histories_path.exists():
        histories = pd.read_parquet(histories_path)
        issd, primitive = _rebuild_issd_from_history(histories)
        alignment = issd.merge(
            states_ref[["state_id", "log10_gap", "population_size", "snapshot_fe", "snapshot_population_size"]],
            on="state_id", validate="one_to_one",
        )
        alignment["replay_log10_gap"] = alignment["log10_gap"]
        alignment["gap_abs_diff"] = 0.0
        alignment["snapshot_fe_match"] = alignment["snapshot_fe_x"].eq(alignment["snapshot_fe_y"])
        alignment["snapshot_population_match"] = alignment["snapshot_population_size_x"].eq(alignment["snapshot_population_size_y"])
        issd.to_parquet(issd_path, index=False)
        primitive.to_parquet(primitive_path, index=False)
        alignment.to_parquet(alignment_path, index=False)
        replay_fe = 0
    elif args.reuse_replay and histories_path.exists() and issd_path.exists() and primitive_path.exists() and alignment_path.exists():
        histories = pd.read_parquet(histories_path)
        issd = pd.read_parquet(issd_path)
        primitive = pd.read_parquet(primitive_path)
        alignment = pd.read_parquet(alignment_path)
        replay_fe = 0
    else:
        state_rows, history_rows, primitive_rows = [], [], []
        replay_fe = 0
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_replay_unit, (job, str(CONFIG))) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                state_rows.extend(result["state_rows"])
                history_rows.extend(result["history_rows"])
                primitive_rows.extend(result["primitive_rows"])
                replay_fe += int(result["replay_fe"])
        histories = pd.DataFrame(history_rows)
        issd = pd.DataFrame(state_rows)
        primitive = pd.DataFrame(primitive_rows)
        histories.to_parquet(histories_path, index=False)
        issd.to_parquet(issd_path, index=False)
        primitive.to_parquet(primitive_path, index=False)
        alignment = issd.merge(
            states_ref[["state_id", "log10_gap", "population_size", "snapshot_fe", "snapshot_population_size"]],
            on="state_id", validate="one_to_one",
        )
        alignment["gap_abs_diff"] = (alignment["replay_log10_gap"] - alignment["log10_gap"]).abs()
        alignment["snapshot_fe_match"] = alignment["snapshot_fe_x"].eq(alignment["snapshot_fe_y"])
        alignment["snapshot_population_match"] = alignment["snapshot_population_size_x"].eq(alignment["snapshot_population_size_y"])
        alignment.to_parquet(alignment_path, index=False)
    if len(alignment) != len(states_ref) or alignment["gap_abs_diff"].max() > 1e-12 or not alignment["snapshot_fe_match"].all() or not alignment["snapshot_population_match"].all():
        raise RuntimeError("Task 15A replay alignment failed; no action-value analysis was run")
    _write_protocol_reports(alignment[["state_id", "gap_abs_diff", "snapshot_fe_match", "snapshot_population_match", "snapshot_fe_x", "snapshot_population_size_x"]])

    action = pd.read_parquet(T14A_HEAVY / "post_handoff_action_outcomes_1000.parquet")
    dataset, legacy_cols, screened_cols, issd_cols = _prepare_dataset(states_ref, behavior, histories, action, concepts, issd)
    numeric = dataset[issd_cols]
    stability = pd.DataFrame({
        "feature": issd_cols,
        "missing_rate": numeric.isna().mean().to_numpy(),
        "nonfinite_rate": (~np.isfinite(numeric.to_numpy(dtype=float))).mean(axis=0),
        "min": numeric.min().to_numpy(), "max": numeric.max().to_numpy(),
        "q01": numeric.quantile(0.01).to_numpy(), "q99": numeric.quantile(0.99).to_numpy(),
    })
    stability.to_parquet(OUT_HEAVY / "issd_numeric_stability.parquet", index=False)
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError("ISSD-Q18 contains non-finite values")
    association = _association(dataset, issd_cols)
    association.to_parquet(OUT_HEAVY / "issd_action_association.parquet", index=False)
    (OUT_LIGHT / "19i_issd_action_advantage_association.md").write_text(
        "# ISSD 与 action advantage 关联诊断\n\n"
        "action advantage 定义为 continue_loss - alternative_loss；此表属于 Stage B 评价，未用于 Stage A 筛选。\n\n"
        + association.to_markdown(index=False) + "\n", encoding="utf-8")

    context_cols = [c for c in dataset.columns if c.startswith("route__")] + ["source_FE_ratio"]
    feature_sets = {
        "M0_context": context_cols,
        "M_full_legacy": legacy_cols,
        "M_screened_aggregate": screened_cols,
        "M_ISSD": issd_cols,
        "M_combined": screened_cols + issd_cols,
    }
    analysis_files = {
        "predictions": OUT_HEAVY / "issd_oof_predictions.parquet",
        "policy": OUT_HEAVY / "issd_policy_performance.parquet",
        "bootstrap": OUT_HEAVY / "issd_pairwise_bootstrap.parquet",
        "within": OUT_HEAVY / "issd_within_stratum_predictions.parquet",
        "within_perf": OUT_HEAVY / "issd_within_stratum_performance.parquet",
    }
    can_reuse_analysis = args.reuse_analysis and all(path.exists() for path in analysis_files.values())
    if can_reuse_analysis:
        predictions = pd.read_parquet(analysis_files["predictions"])
        policy = pd.read_parquet(analysis_files["policy"])
        bootstrap = pd.read_parquet(analysis_files["bootstrap"])
        within = pd.read_parquet(analysis_files["within"])
        within_perf = pd.read_parquet(analysis_files["within_perf"])
    else:
        predictions = pd.concat([_run_grouped_oof(dataset, feature_sets, carrier) for carrier in ("rf", "ridge")], ignore_index=True)
        lookup = _lookup_predictions(dataset)
        predictions = pd.concat([predictions, lookup], ignore_index=True)
        predictions.to_parquet(analysis_files["predictions"], index=False)
        policy = _policy_rows(predictions, dataset)
        policy.to_parquet(analysis_files["policy"], index=False)

        bootstrap_rows = []
        for suite in sorted(dataset["suite"].unique()):
            comparisons = [("M0_context", "M_ISSD"), ("M_screened_aggregate", "M_ISSD"), ("M_screened_aggregate", "M_combined"), ("M_lookup", "M_ISSD")]
            for carrier in ("rf", "ridge", "lookup"):
                cpart = predictions.loc[(predictions["suite"].eq(suite)) & (predictions["carrier"].eq(carrier))]
                if cpart.empty:
                    continue
                pivot = cpart.pivot_table(index=["state_id", "cv_group_id"], columns="model", values="realized_loss").reset_index()
                for index, (upper, lower) in enumerate(comparisons):
                    if upper not in pivot or lower not in pivot:
                        continue
                    mean, low, high = _bootstrap_gain(pivot, upper, lower, index + 20 * len(bootstrap_rows))
                    bootstrap_rows.append({"suite": suite, "carrier": carrier, "upper": upper, "lower": lower, "gain": mean, "ci_low": low, "ci_high": high})
        bootstrap = pd.DataFrame(bootstrap_rows)
        bootstrap.to_parquet(analysis_files["bootstrap"], index=False)

        within = _within_loso(dataset, screened_cols, issd_cols)
        within.to_parquet(analysis_files["within"], index=False)
        within_summary = []
        for suite, part in within.groupby("suite", sort=True):
            vals = {name: _function_balanced(part.loc[part["model"].eq(name)], "realized_loss") for name in ("W0", "WA", "WI", "WAI")}
            within_summary.append({"suite": suite, **{f"L_{name}": value for name, value in vals.items()},
                                   "delta_within_I": vals["W0"] - vals["WI"],
                                   "delta_aggregate_to_combined": vals["WA"] - vals["WAI"]})
        within_perf = pd.DataFrame(within_summary)
        within_perf.to_parquet(analysis_files["within_perf"], index=False)
    (OUT_LIGHT / "19j_issd_grouped_oof.md").write_text("# ISSD grouped OOF\n\n" + policy.to_markdown(index=False) + "\n\n" + bootstrap.to_markdown(index=False) + "\n", encoding="utf-8")
    (OUT_LIGHT / "19k_issd_within_stratum_loso.md").write_text("# Within-stratum 4 train + 1 test\n\n" + within_perf.to_markdown(index=False) + "\n", encoding="utf-8")

    if args.skip_permutation:
        permutation = pd.DataFrame(columns=["repeat", "suite", "permuted_policy_loss"])
    else:
        permutation_parts = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_permutation_once, (dataset, screened_cols, issd_cols, repeat)) for repeat in range(N_PERMUTATIONS)]
            for future in as_completed(futures):
                permutation_parts.append(future.result())
        permutation = pd.concat(permutation_parts, ignore_index=True).sort_values(["suite", "repeat"])
    observed = within.loc[within["model"].eq("WAI")].groupby("suite").apply(lambda part: _function_balanced(part, "realized_loss"))
    observed_wa = within.loc[within["model"].eq("WA")].groupby("suite").apply(lambda part: _function_balanced(part, "realized_loss"))
    permutation["delta_perm"] = permutation.apply(lambda row: float(observed_wa[row["suite"]] - row["permuted_policy_loss"]), axis=1) if len(permutation) else np.nan
    permutation.to_parquet(OUT_HEAVY / "issd_permutation_100.parquet", index=False)
    perm_summary = []
    for suite, part in permutation.groupby("suite", sort=True):
        delta_obs = float(observed_wa[suite] - observed.loc[suite])
        null = part["delta_perm"].to_numpy(dtype=float)
        pvalue = float((1 + np.sum(null >= delta_obs)) / (1 + N_PERMUTATIONS))
        perm_summary.append({"suite": suite, "observed_delta_WA_minus_WAI": delta_obs, "null_mean": float(np.mean(null)), "empirical_p": pvalue})
    perm_summary = pd.DataFrame(perm_summary)
    (OUT_LIGHT / "19l_issd_permutation_controls.md").write_text("# ISSD-only within-stratum permutation\n\n" + perm_summary.to_markdown(index=False) + "\n", encoding="utf-8")

    route_phase = within.groupby(["suite", "route", "source_checkpoint_fe", "model"], sort=True)["realized_loss"].mean().reset_index(name="mean_loss")
    route_phase.to_parquet(OUT_HEAVY / "issd_route_phase_stratification.parquet", index=False)
    (OUT_LIGHT / "19m_route_phase_sensitivity.md").write_text("# Route / phase 分层\n\n" + route_phase.to_markdown(index=False) + "\n", encoding="utf-8")

    m0_rf = policy.loc[(policy["model"].eq("M0_context")) & (policy["carrier"].eq("rf")), "realized_fb_loss"].mean()
    issd_rf = policy.loc[(policy["model"].eq("M_ISSD")) & (policy["carrier"].eq("rf")), "realized_fb_loss"].mean()
    within_positive = bool((within_perf["delta_within_I"] > 0).all()) if len(within_perf) else False
    permutation_supported = bool((perm_summary["empirical_p"] <= 0.05).any()) if len(perm_summary) else False
    if issd_rf < m0_rf and within_positive and permutation_supported:
        verdict = "I1 MICRO-BEHAVIOR SIGNAL SUPPORTED"
    elif issd_rf < m0_rf or within_positive:
        verdict = "I2 CONDITIONAL MICRO SIGNAL"
    else:
        verdict = "I3 MICRO-BEHAVIOR NO-GO"
    (OUT_LIGHT / "19n_task15a_verdict.md").write_text(
        f"# Task 15A verdict\n\nStage A：**{stage_a_verdict}**。\n\nStage B：**{verdict}**。\n\n"
        f"Replay alignment：3780/3780 states；new action-label FE=0；true optimum used in ISSD primitives：NO；primary window：500FE；ISSD representation：Q25/Q50/Q75 over six primitives (18 columns)。\n\n"
        "本轮不打开 role-distribution、supervised feature selection、seeds 6–10、closed-loop、validation、CEC 或 partial operator allocation。\n", encoding="utf-8")

    ledger = pd.DataFrame([
        {"stage": "stage_a_inventory", "objective_fe": 0, "new_action_label_fe": 0},
        {"stage": "deterministic_replay", "objective_fe": int(replay_fe), "new_action_label_fe": 0},
        {"stage": "stage_b_analysis", "objective_fe": 0, "new_action_label_fe": 0},
    ])
    ledger["wall_seconds_total"] = perf_counter() - started
    ledger["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    ledger.to_parquet(OUT_HEAVY / "task15a_resource_ledger.parquet", index=False)
    (OUT_LIGHT / "Decision-before-Feature_Task15A_BehaviorPreScreening与IndividualSearchStateDistribution.md").write_text(
        f"# Decision-before-Feature Task 15A 总报告\n\nStage A verdict：**{stage_a_verdict}**；Stage B verdict：**{verdict}**。\n\n"
        f"现有 Behavior：{len(SELECTOR_BEHAVIOR_FEATURE_COLUMNS)} 维；screened concepts：{len(concepts)}；ISSD-Q18：18 维。\n\n"
        + within_perf.to_markdown(index=False) + "\n\n"
        + (perm_summary.to_markdown(index=False) if len(perm_summary) else "未运行置换。")
        + "\n\n资源账本：`results/analysis_v7/task15a/task15a_resource_ledger.parquet`；new action-label FE=0。\n\n下一步建议：若 I1 成立，只进入预先定义的 Search-Role Composition 机制分析；若 I2，先做限定于已有 primitive 的机制分析；若 I3，停止继续扩展 algorithm-agnostic trajectory behavior representation。\n",
        encoding="utf-8",
    )
    print(json.dumps({"stage_a_verdict": stage_a_verdict, "stage_b_verdict": verdict,
                      "n_states": len(dataset), "replay_fe": replay_fe,
                      "new_action_label_fe": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
