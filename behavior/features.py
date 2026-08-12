from __future__ import annotations

from collections import defaultdict
from math import ceil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks.factory import problem_bounds
from trajectory.validation import validate_trajectory_file


EPS = 1e-12
WINDOW_SHORT = 0.02
WINDOW_MEDIUM = 0.05
WINDOW_LONG = 0.10

BEHAVIOR_METADATA_COLUMNS = (
    "problem_id",
    "family",
    "dimension",
    "algorithm",
    "seed",
    "FE",
    "FE_ratio",
)

BEHAVIOR_WINDOW_METADATA_COLUMNS = (
    "effective_window_ratio_w02",
    "effective_window_fe_w02",
    "effective_native_updates_w02",
    "effective_window_ratio_w05",
    "effective_window_fe_w05",
    "effective_native_updates_w05",
    "effective_window_ratio_w10",
    "effective_window_fe_w10",
    "effective_native_updates_w10",
)

TIME_ONLY_BEHAVIOR_FEATURE_COLUMNS = (
    "bf_fe_ratio",
)

BASE_BEHAVIOR_FEATURE_COLUMNS = (
    "bf_fe_ratio",
    "bf_improvement_rate_w02",
    "bf_improvement_frequency_w02",
    "bf_diversity_mean_pairwise",
    "bf_diversity_change_w05",
    "bf_covariance_spectral_concentration",
    "bf_distance_decay_w10",
    "bf_stagnation_w10",
    "bf_convergence_rate_w10",
)

PRIMARY_BEHAVIOR_FEATURE_COLUMNS = (
    "bf_fitness_diversity",
    "bf_fitness_diversity_rel",
    "bf_population_wasserstein_rate_w05",
    "bf_centroid_shift_rate_w05",
    "bf_centroid_shift_coherence_w05",
    "bf_fitness_quantile_improvement_fraction_w02",
    "bf_fitness_distribution_improvement_rate_w02",
    "bf_fitness_wasserstein_rate_w02",
    "bf_elite_concentration",
    "bf_best_fitness_slope_w05",
    "bf_diversity_slope_w05",
)

DYNAMOREP_LITE_BEHAVIOR_FEATURE_COLUMNS = (
    "bf_robust_fitness_iqr_rel",
    "bf_fitness_spread_slope_w05",
    "bf_population_centroid_shift_w05",
    "bf_elite_centroid_shift_w05",
    "bf_covariance_trace_ratio_w05",
    "bf_covariance_effective_rank_w05",
    "bf_diversity_recovery_w05",
)

MOTION_BEHAVIOR_FEATURE_COLUMNS = (
    "bf_population_chamfer_distance_w05",
    "bf_elite_centroid_shift_w05",
    "bf_covariance_trace_change_w05",
    "bf_covariance_effective_rank_change_w05",
)

MATURITY_BEHAVIOR_FEATURE_COLUMNS = (
    "bf_search_maturity",
    "bf_search_maturity_linear",
    "bf_explore_exploit_ratio",
)

DIAGNOSTIC_BEHAVIOR_FEATURE_COLUMNS = (
    "bf_population_overlap_w05",
    "bf_best_distance_fitness_corr",
)

BEHAVIOR_FEATURE_COLUMNS = (
    BASE_BEHAVIOR_FEATURE_COLUMNS
    + PRIMARY_BEHAVIOR_FEATURE_COLUMNS
    + DYNAMOREP_LITE_BEHAVIOR_FEATURE_COLUMNS
    + MOTION_BEHAVIOR_FEATURE_COLUMNS
    + MATURITY_BEHAVIOR_FEATURE_COLUMNS
    + DIAGNOSTIC_BEHAVIOR_FEATURE_COLUMNS
)

BEHAVIOR_FEATURE_GROUPS = {
    "time_only": TIME_ONLY_BEHAVIOR_FEATURE_COLUMNS,
    "base": BASE_BEHAVIOR_FEATURE_COLUMNS,
    "primary": BASE_BEHAVIOR_FEATURE_COLUMNS + PRIMARY_BEHAVIOR_FEATURE_COLUMNS,
    "primary_with_dynamorep_lite": BASE_BEHAVIOR_FEATURE_COLUMNS + PRIMARY_BEHAVIOR_FEATURE_COLUMNS + DYNAMOREP_LITE_BEHAVIOR_FEATURE_COLUMNS,
    "primary_with_movement": BASE_BEHAVIOR_FEATURE_COLUMNS + PRIMARY_BEHAVIOR_FEATURE_COLUMNS + DYNAMOREP_LITE_BEHAVIOR_FEATURE_COLUMNS + MOTION_BEHAVIOR_FEATURE_COLUMNS,
    "primary_with_maturity": BASE_BEHAVIOR_FEATURE_COLUMNS + PRIMARY_BEHAVIOR_FEATURE_COLUMNS + DYNAMOREP_LITE_BEHAVIOR_FEATURE_COLUMNS + MOTION_BEHAVIOR_FEATURE_COLUMNS + MATURITY_BEHAVIOR_FEATURE_COLUMNS,
    "all_candidates": BEHAVIOR_FEATURE_COLUMNS,
}

BEHAVIOR_COLUMNS = BEHAVIOR_METADATA_COLUMNS + BEHAVIOR_WINDOW_METADATA_COLUMNS + BEHAVIOR_FEATURE_COLUMNS

BEHAVIOR_SCHEMA = pa.schema(
    [
        ("problem_id", pa.string()),
        ("family", pa.string()),
        ("dimension", pa.int32()),
        ("algorithm", pa.string()),
        ("seed", pa.int64()),
        ("FE", pa.int64()),
        ("FE_ratio", pa.float64()),
        ("effective_window_ratio_w02", pa.float64()),
        ("effective_window_fe_w02", pa.int64()),
        ("effective_native_updates_w02", pa.int64()),
        ("effective_window_ratio_w05", pa.float64()),
        ("effective_window_fe_w05", pa.int64()),
        ("effective_native_updates_w05", pa.int64()),
        ("effective_window_ratio_w10", pa.float64()),
        ("effective_window_fe_w10", pa.int64()),
        ("effective_native_updates_w10", pa.int64()),
        *((column, pa.float64()) for column in BEHAVIOR_FEATURE_COLUMNS),
    ]
)


def read_trajectory_rows(path: str | Path) -> list[dict]:
    validate_trajectory_file(path)
    return pq.read_table(path).to_pylist()


def extract_behavior_rows(trajectory_rows: list[dict]) -> list[dict]:
    if not trajectory_rows:
        raise ValueError("trajectory rows must not be empty")

    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for position, row in enumerate(trajectory_rows):
        row["_source_position"] = position
        grouped[(row["algorithm"], row["problem_id"], int(row["seed"]))].append(row)

    behavior_rows: list[dict] = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: item["FE"])
        for index, row in enumerate(ordered):
            row["_behavior_index"] = index
        initial_fitness_iqr = _fitness_shift_invariant_baseline(ordered[0])
        stats = [_checkpoint_stats(row, initial_fitness_iqr=initial_fitness_iqr) for row in ordered]
        for index, row in enumerate(ordered):
            short_window = _window_statistic(row, "w02", WINDOW_SHORT)
            medium_window = _window_statistic(row, "w05", WINDOW_MEDIUM)
            long_window = _window_statistic(row, "w10", WINDOW_LONG)
            native_history = _native_update_history(row)

            metadata = {column: row[column] for column in BEHAVIOR_METADATA_COLUMNS}
            current_ratio = float(row["FE_ratio"])
            window_metadata = {
                **_effective_window_metadata(row, short_window, "w02"),
                **_effective_window_metadata(row, medium_window, "w05"),
                **_effective_window_metadata(row, long_window, "w10"),
            }
            population_change = _population_set_change_stats(row, medium_window)
            fitness_change = _fitness_distribution_change_stats(row, short_window)
            features = {
                "bf_fe_ratio": current_ratio,
                "bf_improvement_rate_w02": _improvement_rate(row, short_window),
                "bf_improvement_frequency_w02": _improvement_frequency(native_history, short_window),
                "bf_diversity_mean_pairwise": stats[index]["diversity"],
                "bf_diversity_change_w05": _relative_change(
                    stats[index]["diversity"],
                    float(medium_window["anchor_diversity_mean_pairwise"]),
                ),
                "bf_covariance_spectral_concentration": stats[index]["covariance_spectral_concentration"],
                "bf_distance_decay_w10": _relative_decay(
                    stats[index]["distance_to_best"],
                    float(long_window["anchor_distance_to_best"]),
                ),
                "bf_stagnation_w10": _stagnation(native_history, row),
                "bf_convergence_rate_w10": _convergence_rate(
                    row,
                    long_window,
                    stats[index],
                ),
                "bf_fitness_diversity": stats[index]["fitness_diversity"],
                "bf_fitness_diversity_rel": stats[index]["fitness_diversity_rel"],
                "bf_population_wasserstein_rate_w05": population_change["wasserstein_rate"],
                "bf_centroid_shift_rate_w05": population_change["centroid_shift_rate"],
                "bf_centroid_shift_coherence_w05": population_change["centroid_shift_coherence"],
                "bf_fitness_quantile_improvement_fraction_w02": fitness_change["quantile_improvement_fraction"],
                "bf_fitness_distribution_improvement_rate_w02": fitness_change["distribution_improvement_rate"],
                "bf_fitness_wasserstein_rate_w02": fitness_change["wasserstein_rate"],
                "bf_elite_concentration": _elite_concentration(row, stats[index]["diversity"]),
                "bf_best_fitness_slope_w05": _window_slope(native_history, medium_window, "best_fitness"),
                "bf_diversity_slope_w05": _window_slope(
                    native_history,
                    medium_window,
                    "diversity_mean_pairwise",
                ),
                "bf_robust_fitness_iqr_rel": _robust_fitness_iqr_rel(row),
                "bf_fitness_spread_slope_w05": _window_slope(native_history, medium_window, "fitness_iqr_rel"),
                "bf_population_centroid_shift_w05": float(medium_window["centroid_shift_distance"]),
                "bf_elite_centroid_shift_w05": float(medium_window["elite_centroid_shift"]),
                "bf_covariance_trace_ratio_w05": float(medium_window["covariance_trace_ratio"]),
                "bf_covariance_effective_rank_w05": float(medium_window["covariance_effective_rank"]),
                "bf_diversity_recovery_w05": _diversity_recovery(row, medium_window),
                "bf_population_chamfer_distance_w05": float(medium_window["population_chamfer_distance"]),
                "bf_covariance_trace_change_w05": float(medium_window["covariance_trace_change"]),
                "bf_covariance_effective_rank_change_w05": float(medium_window["covariance_effective_rank_change"]),
                "bf_population_overlap_w05": float(medium_window["population_overlap"]),
                "bf_best_distance_fitness_corr": _best_distance_fitness_corr(row),
            }
            features.update(_maturity_features(features))
            behavior_rows.append(
                {"_source_position": row["_source_position"], **metadata, **window_metadata, **features}
            )

    ordered_behavior = sorted(behavior_rows, key=lambda item: item["_source_position"])
    return [{column: row[column] for column in BEHAVIOR_COLUMNS} for row in ordered_behavior]


def write_behavior_rows(behavior_rows: list[dict], output_path: str | Path) -> Path:
    if not behavior_rows:
        raise ValueError("no behavior rows to write")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(behavior_rows, schema=BEHAVIOR_SCHEMA)
    pq.write_table(table, path)
    return path


def _checkpoint_stats(row: dict, *, initial_fitness_iqr: float) -> dict[str, float]:
    population = _population(row)
    fitness = _fitness(row)
    problem_id = str(row["problem_id"])
    return {
        "diversity": _mean_pairwise_distance(population, problem_id=problem_id),
        "distance_to_best": _mean_distance_to_population_best(population, fitness, problem_id=problem_id),
        "fitness_diversity": _fitness_diversity(fitness),
        "fitness_diversity_rel": _shift_invariant_fitness_diversity(fitness, initial_fitness_iqr=initial_fitness_iqr),
        "covariance_spectral_concentration": _covariance_spectral_concentration(population),
    }


def _population(row: dict) -> np.ndarray:
    population, _ = _checkpoint_arrays(row)
    return population


def _fitness(row: dict) -> np.ndarray:
    _, fitness = _checkpoint_arrays(row)
    return fitness


def _scale_population_to_unit_cube(population: np.ndarray, *, problem_id: str) -> np.ndarray:
    lower, upper = problem_bounds(problem_id)
    span = upper - lower
    if np.any(span <= 0.0):
        raise ValueError(f"invalid problem bounds for {problem_id}")
    return np.clip((np.asarray(population, dtype=float) - lower) / span, 0.0, 1.0)


def _checkpoint_arrays(row: dict) -> tuple[np.ndarray, np.ndarray]:
    population = np.asarray(row["population"], dtype=float)
    if population.ndim != 2:
        raise ValueError("population must be two-dimensional")
    if population.shape[1] != int(row["dimension"]):
        raise ValueError("population width must match dimension")
    fitness = np.asarray(row["fitness"], dtype=float).reshape(-1)
    if fitness.shape[0] != population.shape[0]:
        raise ValueError("fitness length must match population rows")
    # Stabilize floating-point reductions under row permutations without defining
    # any correspondence between individuals at different checkpoints.
    sort_keys = [fitness]
    sort_keys.extend(population[:, axis] for axis in reversed(range(population.shape[1])))
    order = np.lexsort(tuple(sort_keys))
    return population[order], fitness[order]


def _mean_pairwise_distance(population: np.ndarray, *, problem_id: str) -> float:
    if population.shape[0] < 2:
        return 0.0
    scaled = _scale_population_to_unit_cube(population, problem_id=problem_id)
    deltas = scaled[:, None, :] - scaled[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    upper = distances[np.triu_indices(population.shape[0], k=1)]
    return float(np.mean(upper))


def _mean_distance_to_population_best(population: np.ndarray, fitness: np.ndarray, *, problem_id: str) -> float:
    best = population[int(np.argmin(fitness))]
    scaled = _scale_population_to_unit_cube(population, problem_id=problem_id)
    best_scaled = _scale_population_to_unit_cube(best.reshape(1, -1), problem_id=problem_id)[0]
    distances = np.linalg.norm(scaled - best_scaled, axis=1)
    return float(np.mean(distances))


def _fitness_diversity(fitness: np.ndarray) -> float:
    return float(np.std(fitness))


def _fitness_iqr(fitness: np.ndarray) -> float:
    q75, q25 = np.percentile(np.asarray(fitness, dtype=float).reshape(-1), [75.0, 25.0])
    return float(q75 - q25)


def _shift_invariant_fitness_diversity(fitness: np.ndarray, *, initial_fitness_iqr: float) -> float:
    return float(_fitness_iqr(fitness) / max(initial_fitness_iqr, EPS))


def _fitness_shift_invariant_baseline(row: dict) -> float:
    return _fitness_iqr(_fitness(row))


def _robust_fitness_iqr_rel(row: dict) -> float:
    return float(_fitness_iqr(_fitness(row)) / max(_fitness_shift_invariant_baseline(row), EPS))


def _diversity_recovery(current: dict, window: dict) -> float:
    current_diversity = float(_checkpoint_stats(current, initial_fitness_iqr=_fitness_shift_invariant_baseline(current))["diversity"])
    anchor_diversity = float(window["anchor_diversity_mean_pairwise"])
    if anchor_diversity <= EPS:
        return 0.0
    return float(max(0.0, (current_diversity - anchor_diversity) / (anchor_diversity + EPS)))


def _window_statistic(row: dict, suffix: str, nominal_ratio: float) -> dict:
    values = row.get("window_statistics")
    if not isinstance(values, list):
        raise ValueError("trajectory row is missing native-update window statistics; regenerate trajectories")
    matches = [item for item in values if str(item.get("suffix")) == suffix]
    if len(matches) != 1:
        raise ValueError(f"trajectory row must contain exactly one {suffix} window statistic")
    result = matches[0]
    if abs(float(result["nominal_window_ratio"]) - nominal_ratio) > EPS:
        raise ValueError(f"{suffix} nominal window ratio is inconsistent")
    return result


def _native_update_history(row: dict) -> list[dict]:
    values = row.get("native_update_history")
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError("trajectory row is missing native-update scalar history; regenerate trajectories")
    ordered = sorted(values, key=lambda item: int(item["FE"]))
    if int(ordered[-1]["FE"]) != int(row["FE"]):
        raise ValueError("native-update scalar history must end at the formal checkpoint")
    return ordered


def _effective_window_metadata(current: dict, window: dict, suffix: str) -> dict[str, float | int]:
    keys = (
        f"effective_window_ratio_{suffix}",
        f"effective_window_fe_{suffix}",
        f"effective_native_updates_{suffix}",
    )
    fe_total = int(current["FE_total"])
    fe_delta = int(current["FE"]) - int(window["anchor_FE"])
    native_update_delta = int(current["native_updates"]) - int(window["anchor_native_updates"])
    if fe_delta <= 0 or native_update_delta < 0:
        raise ValueError("window FE must increase and native_updates must not decrease")
    return {
        keys[0]: float(fe_delta / fe_total),
        keys[1]: fe_delta,
        keys[2]: native_update_delta,
    }


def _strict_improvement(previous_best: float, current_best: float) -> bool:
    threshold = EPS * max(1.0, abs(float(previous_best)))
    return float(previous_best) - float(current_best) > threshold


def _window_ratio_delta(current: dict, window: dict) -> float:
    return float((int(current["FE"]) - int(window["anchor_FE"])) / int(current["FE_total"]))


def _improvement_rate(current: dict, window: dict) -> float:
    ratio_delta = _window_ratio_delta(current, window)
    if ratio_delta <= 0.0:
        raise ValueError("strict improvement window must have positive FE span")
    anchor_best = float(window["anchor_best_fitness"])
    scale = max(float(window["fitness_iqr_baseline"]), EPS)
    return float((anchor_best - float(current["best_fitness"])) / scale / ratio_delta)


def _improvement_frequency(history: list[dict], window: dict) -> float:
    rows = [item for item in history if int(item["FE"]) >= int(window["anchor_FE"])]
    intervals = len(rows) - 1
    if intervals <= 0:
        raise ValueError("strict improvement window must contain at least one native-update interval")
    improvements = 0
    for previous, current in zip(rows, rows[1:]):
        if _strict_improvement(previous["best_fitness"], current["best_fitness"]):
            improvements += 1
    return float(improvements / intervals)


def _relative_change(current: float, anchor: float | None) -> float | None:
    if anchor is None:
        return None
    return float((current - anchor) / max(anchor, EPS))


def _relative_decay(current: float, anchor: float | None) -> float | None:
    if anchor is None:
        return None
    return float((anchor - current) / max(anchor, EPS))


def _population_set_change_stats(current: dict, window: dict) -> dict[str, float]:
    ratio_delta = _window_ratio_delta(current, window)
    if ratio_delta <= 0.0:
        raise ValueError("strict population window must have positive FE span")
    wasserstein_distance = float(window["population_wasserstein_distance"])
    centroid_shift = float(window["centroid_shift_distance"])
    coherence = 0.0 if wasserstein_distance <= EPS else _clip_unit(centroid_shift / wasserstein_distance)
    return {
        "wasserstein_rate": float(wasserstein_distance / ratio_delta),
        "centroid_shift_rate": float(centroid_shift / ratio_delta),
        "centroid_shift_coherence": coherence,
    }


def _fitness_distribution_change_stats(current: dict, window: dict) -> dict[str, float]:
    ratio_delta = _window_ratio_delta(current, window)
    if ratio_delta <= 0.0:
        raise ValueError("strict fitness window must have positive FE span")
    scale = max(float(window["fitness_iqr_baseline"]), EPS)
    return {
        "quantile_improvement_fraction": float(window["fitness_quantile_improvement_fraction"]),
        "distribution_improvement_rate": float(window["fitness_mean_improvement"] / scale / ratio_delta),
        "wasserstein_rate": float(window["fitness_wasserstein_distance"] / scale / ratio_delta),
    }


def _covariance_spectral_concentration(population: np.ndarray) -> float:
    population_size, dimension = population.shape
    available_rank = min(dimension, population_size - 1)
    if available_rank <= 1:
        return 0.0
    centered = population - np.mean(population, axis=0)
    eigenvalues = np.maximum(np.linalg.eigvalsh(centered.T @ centered), 0.0)
    total = float(np.sum(eigenvalues))
    if total <= EPS:
        return 0.0
    herfindahl = float(np.sum(eigenvalues * eigenvalues) / (total * total))
    concentration = (available_rank * herfindahl - 1.0) / (available_rank - 1.0)
    return _clip_unit(concentration)


def _elite_concentration(current: dict, population_diversity: float) -> float:
    population = _population(current)
    fitness = _fitness(current)
    if population.shape[0] < 2 or population_diversity <= EPS:
        return 0.0
    elite_count = min(population.shape[0], max(2, int(ceil(0.20 * population.shape[0]))))
    elite_indices = np.argsort(fitness)[:elite_count]
    elite_diversity = _mean_pairwise_distance(population[elite_indices], problem_id=str(current["problem_id"]))
    return float(elite_diversity / max(population_diversity, EPS))


def _window_slope(history: list[dict], window: dict, name: str) -> float:
    rows = [item for item in history if int(item["FE"]) >= int(window["anchor_FE"])]
    ratios = np.asarray([float(row["FE_ratio"]) for row in rows], dtype=float)
    if np.unique(ratios).size < 2:
        raise ValueError("strict slope window must contain at least two native updates")
    values = np.asarray([float(row[name]) for row in rows], dtype=float)
    centered_ratios = ratios - float(np.mean(ratios))
    denominator = float(np.sum(centered_ratios * centered_ratios))
    if denominator <= EPS:
        raise ValueError("strict slope window has zero FE variance")
    centered_values = values - float(np.mean(values))
    return float(np.sum(centered_ratios * centered_values) / denominator)


def _stagnation(history: list[dict], current: dict) -> float:
    last_improvement_fe = int(history[0]["FE"])
    for previous, later in zip(history, history[1:]):
        if _strict_improvement(previous["best_fitness"], later["best_fitness"]):
            last_improvement_fe = int(later["FE"])
    span_ratio = max((int(current["FE"]) - last_improvement_fe) / int(current["FE_total"]), 0.0)
    return float(min(span_ratio, WINDOW_LONG) / WINDOW_LONG)


def _best_distance_fitness_corr(current: dict) -> float | None:
    population = _population(current)
    fitness = _fitness(current)
    if population.shape[0] < 2:
        return None
    best = population[int(np.argmin(fitness))]
    problem_id = str(current["problem_id"])
    scaled = _scale_population_to_unit_cube(population, problem_id=problem_id)
    best_scaled = _scale_population_to_unit_cube(best.reshape(1, -1), problem_id=problem_id)[0]
    distances = np.linalg.norm(scaled - best_scaled, axis=1)
    if float(np.std(distances)) <= EPS or float(np.std(fitness)) <= EPS:
        return None
    return float(min(max(float(np.corrcoef(distances, fitness)[0, 1]), -1.0), 1.0))


def _maturity_features(features: dict[str, float | None]) -> dict[str, float | None]:
    diversity_stabilization = _positive_saturation(_negated(features["bf_diversity_slope_w05"]))
    population_stabilization = _inverse_positive_saturation(features["bf_population_wasserstein_rate_w05"])
    covariance_structure = _unit_interval(features["bf_covariance_spectral_concentration"])
    centroid_coherence = _unit_interval(features["bf_centroid_shift_coherence_w05"])
    if (
        diversity_stabilization is None
        or population_stabilization is None
        or covariance_structure is None
        or centroid_coherence is None
    ):
        exploration_stabilization = None
    else:
        exploration_stabilization = _mean_optional(
            diversity_stabilization,
            population_stabilization,
            covariance_structure,
            centroid_coherence,
        )

    stagnation = _unit_interval(features["bf_stagnation_w10"])
    improvement_saturation = _inverse_positive_saturation(features["bf_improvement_rate_w02"])
    distance_decay = _positive_saturation(features["bf_distance_decay_w10"])
    convergence = _positive_saturation(features["bf_convergence_rate_w10"])
    local_concentration = _inverse_positive_saturation(features["bf_elite_concentration"])
    exploitation_saturation = _mean_optional(stagnation, improvement_saturation, distance_decay, convergence, local_concentration)

    exploration = _mean_optional(
        _positive_saturation(features["bf_diversity_mean_pairwise"]),
        _one_minus_unit(features["bf_covariance_spectral_concentration"]),
        _positive_saturation(features["bf_population_wasserstein_rate_w05"]),
    )
    exploitation = _mean_optional(centroid_coherence, local_concentration, distance_decay, convergence)

    if exploration_stabilization is None or exploitation_saturation is None:
        maturity = None
        maturity_linear = None
    else:
        maturity = float(exploration_stabilization * (1.0 - exploitation_saturation))
        maturity_linear = _unit_interval((exploration_stabilization + (1.0 - exploitation_saturation)) / 2.0)

    if exploration is None or exploitation is None:
        ratio = None
    else:
        ratio = float(exploration / (exploitation + EPS))

    return {
        "bf_search_maturity": maturity,
        "bf_search_maturity_linear": maturity_linear,
        "bf_explore_exploit_ratio": ratio,
    }


def _unit_interval(value: float | None) -> float | None:
    if value is None:
        return None
    return float(min(max(float(value), 0.0), 1.0))


def _clip_unit(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def _one_minus_unit(value: float | None) -> float | None:
    bounded = _unit_interval(value)
    return None if bounded is None else float(1.0 - bounded)


def _positive_saturation(value: float | None) -> float | None:
    if value is None:
        return None
    positive = max(float(value), 0.0)
    return float(positive / (1.0 + positive))


def _inverse_positive_saturation(value: float | None) -> float | None:
    if value is None:
        return None
    return float(1.0 / (1.0 + max(float(value), 0.0)))


def _negated(value: float | None) -> float | None:
    return None if value is None else float(-float(value))


def _mean_optional(*values: float | None) -> float | None:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    return float(np.mean(finite))


def _convergence_rate(
    current: dict,
    window: dict,
    current_stats: dict[str, float],
) -> float:
    ratio_delta = _window_ratio_delta(current, window)
    if ratio_delta <= 0.0:
        raise ValueError("strict convergence window must have positive FE span")
    anchor_diversity = float(window["anchor_diversity_mean_pairwise"])
    return float((anchor_diversity - current_stats["diversity"]) / max(anchor_diversity, EPS) / ratio_delta)
