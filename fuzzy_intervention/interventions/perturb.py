from __future__ import annotations

from math import ceil

import numpy as np


SUBSET_FRACTION = 0.25
PERTURB_SIGMA = 0.05


def _reflect_unit_cube(values: np.ndarray) -> np.ndarray:
    wrapped = np.mod(np.asarray(values, dtype=float), 2.0)
    return np.where(wrapped <= 1.0, wrapped, 2.0 - wrapped)


def _population_view(state) -> np.ndarray:
    return state.positions if state.algorithm == "cso" else state.population


def _select_indices(*, mode: str, state, primitives: list[dict], subset_rng: np.random.Generator) -> list[int]:
    count = len(state.population)
    k = max(1, int(ceil(SUBSET_FRACTION * (count - 1))))
    current_best = int(np.argmin(np.asarray(state.fitness, dtype=float)))
    eligible = [row for row in primitives if int(row["population_index"]) != current_best]
    if len(eligible) < k:
        raise ValueError("insufficient non-best individuals for perturbation")
    if mode == "targeted":
        ordered = sorted(
            eligible,
            key=lambda row: (
                -int(row["individual_stagnation_age_FE"]),
                float(row["individual_recent_progress"]),
                str(row["agent_id"]),
            ),
        )
        return [int(row["population_index"]) for row in ordered[:k]]
    if mode == "random":
        pool = np.asarray([int(row["population_index"]) for row in eligible], dtype=int)
        return sorted(int(index) for index in subset_rng.choice(pool, size=k, replace=False))
    raise ValueError(f"unknown perturb mode: {mode}")


def apply_partial_perturbation(
    *,
    state,
    problem,
    primitives: list[dict],
    mode: str,
    subset_rng: np.random.Generator,
    vector_rng: np.random.Generator,
) -> tuple[int, list[dict]]:
    pending = getattr(state, "pending_population", None)
    if pending is None:
        pending = getattr(state, "pending_positions", None)
    if pending is not None:
        raise ValueError("Task16A perturbation requires a complete native update state")
    selected = _select_indices(mode=mode, state=state, primitives=primitives, subset_rng=subset_rng)
    population = _population_view(state)
    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)
    span = upper - lower
    before = population[selected].copy()
    unit_before = (before - lower) / span
    noise = vector_rng.normal(0.0, 1.0, size=before.shape)
    unit_after = _reflect_unit_cube(unit_before + PERTURB_SIGMA * noise)
    after = lower + unit_after * span
    fitness_before = np.asarray(state.fitness, dtype=float)[selected].copy()
    best_before = float(state.best_fitness)
    fitness_after = np.asarray(problem.evaluate(after), dtype=float)
    population[selected] = after
    state.fitness[selected] = fitness_after
    if state.algorithm == "cso":
        state.velocities[selected] = 0.0
    state.evaluations += len(selected)
    best_local = int(np.argmin(fitness_after))
    if float(fitness_after[best_local]) < float(state.best_fitness):
        state.best_fitness = float(fitness_after[best_local])
        state.best_position = after[best_local].copy()
    primitive_by_index = {int(row["population_index"]): row for row in primitives}
    metadata = []
    for offset, index in enumerate(selected):
        primitive = primitive_by_index[index]
        metadata.append(
            {
                "agent_id": str(primitive["agent_id"]),
                "population_index": int(index),
                "selected_for_perturb": True,
                "selection_rank_stagnation": int(primitive["selection_rank_stagnation"]),
                "selection_rank_progress": int(primitive["selection_rank_progress"]),
                "individual_stagnation_age_FE": int(primitive["individual_stagnation_age_FE"]),
                "individual_recent_progress": float(primitive["individual_recent_progress"]),
                "fitness_before": float(fitness_before[offset]),
                "fitness_after": float(fitness_after[offset]),
                "perturb_norm_unitcube": float(np.linalg.norm(unit_after[offset] - unit_before[offset])),
                "became_new_best": bool(
                    float(fitness_after[offset]) < best_before
                    and float(fitness_after[offset]) == float(state.best_fitness)
                ),
            }
        )
    return len(selected), metadata

