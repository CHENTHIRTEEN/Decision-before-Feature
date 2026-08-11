from __future__ import annotations

from collections import defaultdict
from math import ceil, sqrt
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment

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
    + MATURITY_BEHAVIOR_FEATURE_COLUMNS
    + DIAGNOSTIC_BEHAVIOR_FEATURE_COLUMNS
)

BEHAVIOR_FEATURE_GROUPS = {
    "time_only": TIME_ONLY_BEHAVIOR_FEATURE_COLUMNS,
    "base": BASE_BEHAVIOR_FEATURE_COLUMNS,
    "primary": BASE_BEHAVIOR_FEATURE_COLUMNS + PRIMARY_BEHAVIOR_FEATURE_COLUMNS,
    "primary_with_maturity": BASE_BEHAVIOR_FEATURE_COLUMNS + PRIMARY_BEHAVIOR_FEATURE_COLUMNS + MATURITY_BEHAVIOR_FEATURE_COLUMNS,
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
        stats = [_checkpoint_stats(row) for row in ordered]
        last_improvement_fe = int(ordered[0]["FE"])

        for index, row in enumerate(ordered):
            if index > 0 and _strict_improvement(ordered[index - 1]["best_fitness"], row["best_fitness"]):
                last_improvement_fe = int(row["FE"])

            short_anchor = _find_anchor(ordered, index, WINDOW_SHORT)
            medium_anchor = _find_anchor(ordered, index, WINDOW_MEDIUM)
            long_anchor = _find_anchor(ordered, index, WINDOW_LONG)

            metadata = {column: row[column] for column in BEHAVIOR_METADATA_COLUMNS}
            current_ratio = float(row["FE_ratio"])
            window_metadata = {
                **_effective_window_metadata(row, short_anchor, "w02"),
                **_effective_window_metadata(row, medium_anchor, "w05"),
                **_effective_window_metadata(row, long_anchor, "w10"),
            }
            population_change = _population_set_change_stats(row, medium_anchor)
            fitness_change = _fitness_distribution_change_stats(row, short_anchor)
            features = {
                "bf_fe_ratio": current_ratio,
                "bf_improvement_rate_w02": _improvement_rate(row, short_anchor),
                "bf_improvement_frequency_w02": _improvement_frequency(ordered, index, short_anchor),
                "bf_diversity_mean_pairwise": stats[index]["diversity"],
                "bf_diversity_change_w05": _relative_change(
                    stats[index]["diversity"],
                    _anchor_stat(stats, medium_anchor, "diversity"),
                ),
                "bf_covariance_spectral_concentration": stats[index]["covariance_spectral_concentration"],
                "bf_distance_decay_w10": _relative_decay(
                    stats[index]["distance_to_best"],
                    _anchor_stat(stats, long_anchor, "distance_to_best"),
                ),
                "bf_stagnation_w10": min(
                    max((int(row["FE"]) - last_improvement_fe) / int(row["FE_total"]), 0.0),
                    WINDOW_LONG,
                )
                / WINDOW_LONG,
                "bf_convergence_rate_w10": _convergence_rate(
                    row,
                    long_anchor,
                    stats[index],
                    _anchor_stats(stats, long_anchor),
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
                "bf_best_fitness_slope_w05": _window_slope(ordered, stats, index, medium_anchor, "best_fitness"),
                "bf_diversity_slope_w05": _window_slope(ordered, stats, index, medium_anchor, "diversity"),
                "bf_population_overlap_w05": _population_overlap(row, medium_anchor, stats[index]["diversity"]),
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


def _checkpoint_stats(row: dict) -> dict[str, float]:
    population = _population(row)
    fitness = _fitness(row)
    dimension = int(row["dimension"])
    return {
        "diversity": _mean_pairwise_distance(population, dimension),
        "distance_to_best": _mean_distance_to_population_best(population, fitness, dimension),
        "fitness_diversity": _fitness_diversity(fitness),
        "fitness_diversity_rel": _relative_fitness_diversity(fitness),
        "covariance_spectral_concentration": _covariance_spectral_concentration(population),
    }


def _population(row: dict) -> np.ndarray:
    population, _ = _checkpoint_arrays(row)
    return population


def _fitness(row: dict) -> np.ndarray:
    _, fitness = _checkpoint_arrays(row)
    return fitness


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


def _mean_pairwise_distance(population: np.ndarray, dimension: int) -> float:
    if population.shape[0] < 2:
        return 0.0
    deltas = population[:, None, :] - population[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    upper = distances[np.triu_indices(population.shape[0], k=1)]
    return float(np.mean(upper) / sqrt(dimension))


def _mean_distance_to_population_best(population: np.ndarray, fitness: np.ndarray, dimension: int) -> float:
    best = population[int(np.argmin(fitness))]
    distances = np.linalg.norm(population - best, axis=1)
    return float(np.mean(distances) / sqrt(dimension))


def _fitness_diversity(fitness: np.ndarray) -> float:
    return float(np.std(fitness))


def _relative_fitness_diversity(fitness: np.ndarray) -> float:
    return float(_fitness_diversity(fitness) / max(abs(float(np.mean(fitness))), EPS))


def _find_anchor(rows: list[dict], current_index: int, window: float) -> dict | None:
    target = float(rows[current_index]["FE_ratio"]) - window
    anchor = None
    for row in rows[:current_index]:
        if float(row["FE_ratio"]) <= target + EPS:
            anchor = row
        else:
            break
    return anchor


def _effective_window_metadata(current: dict, anchor: dict | None, suffix: str) -> dict[str, float | int | None]:
    keys = (
        f"effective_window_ratio_{suffix}",
        f"effective_window_fe_{suffix}",
        f"effective_native_updates_{suffix}",
    )
    if anchor is None:
        return dict.fromkeys(keys)
    fe_total = int(current["FE_total"])
    if fe_total <= 0 or int(anchor["FE_total"]) != fe_total:
        raise ValueError("window endpoints must share one positive FE_total")
    fe_delta = int(current["FE"]) - int(anchor["FE"])
    native_update_delta = int(current["native_updates"]) - int(anchor["native_updates"])
    if fe_delta <= 0 or native_update_delta < 0:
        raise ValueError("window FE must increase and native_updates must not decrease")
    return {
        keys[0]: float(fe_delta / fe_total),
        keys[1]: fe_delta,
        keys[2]: native_update_delta,
    }


def _actual_fe_ratio(row: dict) -> float:
    fe_total = int(row["FE_total"])
    if fe_total <= 0:
        raise ValueError("FE_total must be positive")
    return float(int(row["FE"]) / fe_total)


def _effective_ratio_delta(current: dict, anchor: dict) -> float:
    return float(_actual_fe_ratio(current) - _actual_fe_ratio(anchor))


def _strict_improvement(previous_best: float, current_best: float) -> bool:
    threshold = EPS * max(1.0, abs(float(previous_best)))
    return float(previous_best) - float(current_best) > threshold


def _improvement_rate(current: dict, anchor: dict | None) -> float | None:
    if anchor is None:
        return None
    ratio_delta = _effective_ratio_delta(current, anchor)
    if ratio_delta <= 0.0:
        return None
    scale = max(abs(float(anchor["best_fitness"])), EPS)
    return float((float(anchor["best_fitness"]) - float(current["best_fitness"])) / scale / ratio_delta)


def _improvement_frequency(rows: list[dict], current_index: int, anchor: dict | None) -> float | None:
    if anchor is None:
        return None
    anchor_index = int(anchor["_behavior_index"])
    intervals = current_index - anchor_index
    if intervals <= 0:
        return None
    improvements = 0
    for previous, current in zip(rows[anchor_index:current_index], rows[anchor_index + 1 : current_index + 1]):
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


def _anchor_stat(stats: list[dict[str, float]], anchor: dict | None, name: str) -> float | None:
    anchor_stats = _anchor_stats(stats, anchor)
    return None if anchor_stats is None else anchor_stats[name]


def _anchor_stats(stats: list[dict[str, float]], anchor: dict | None) -> dict[str, float] | None:
    if anchor is None:
        return None
    return stats[anchor["_behavior_index"]]


def _population_set_change_stats(current: dict, anchor: dict | None) -> dict[str, float | None]:
    missing = {
        "wasserstein_rate": None,
        "centroid_shift_rate": None,
        "centroid_shift_coherence": None,
    }
    if anchor is None:
        return missing
    current_population = _population(current)
    anchor_population = _population(anchor)
    if current_population.shape != anchor_population.shape or current_population.shape[0] == 0:
        return missing
    ratio_delta = _effective_ratio_delta(current, anchor)
    if ratio_delta <= 0.0:
        return missing

    dimension = int(current["dimension"])
    pairwise_distances = (
        np.linalg.norm(anchor_population[:, None, :] - current_population[None, :, :], axis=2) / sqrt(dimension)
    )
    anchor_indices, current_indices = linear_sum_assignment(pairwise_distances)
    wasserstein_distance = float(np.mean(pairwise_distances[anchor_indices, current_indices]))
    centroid_shift = float(
        np.linalg.norm(np.mean(current_population, axis=0) - np.mean(anchor_population, axis=0)) / sqrt(dimension)
    )
    coherence = 0.0 if wasserstein_distance <= EPS else _clip_unit(centroid_shift / wasserstein_distance)
    return {
        "wasserstein_rate": float(wasserstein_distance / ratio_delta),
        "centroid_shift_rate": float(centroid_shift / ratio_delta),
        "centroid_shift_coherence": coherence,
    }


def _fitness_distribution_change_stats(current: dict, anchor: dict | None) -> dict[str, float | None]:
    missing = {
        "quantile_improvement_fraction": None,
        "distribution_improvement_rate": None,
        "wasserstein_rate": None,
    }
    if anchor is None:
        return missing
    current_fitness = _fitness(current)
    anchor_fitness = _fitness(anchor)
    if current_fitness.shape != anchor_fitness.shape or current_fitness.size == 0:
        return missing
    ratio_delta = _effective_ratio_delta(current, anchor)
    if ratio_delta <= 0.0:
        return missing

    anchor_quantiles = np.sort(anchor_fitness)
    current_quantiles = np.sort(current_fitness)
    quantile_improvements = anchor_quantiles - current_quantiles
    threshold = EPS * np.maximum(1.0, np.abs(anchor_quantiles))
    scale = max(float(np.mean(np.abs(anchor_quantiles))), EPS)
    return {
        "quantile_improvement_fraction": float(np.mean(quantile_improvements > threshold)),
        "distribution_improvement_rate": float(np.mean(quantile_improvements) / scale / ratio_delta),
        "wasserstein_rate": float(np.mean(np.abs(quantile_improvements)) / scale / ratio_delta),
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
    elite_diversity = _mean_pairwise_distance(population[elite_indices], int(current["dimension"]))
    return float(elite_diversity / max(population_diversity, EPS))


def _window_slope(rows: list[dict], stats: list[dict[str, float]], current_index: int, anchor: dict | None, name: str) -> float | None:
    if anchor is None:
        return None
    anchor_index = int(anchor["_behavior_index"])
    if current_index <= anchor_index:
        return None
    ratios = np.asarray([_actual_fe_ratio(row) for row in rows[anchor_index : current_index + 1]], dtype=float)
    if np.unique(ratios).size < 2:
        return None
    if name == "best_fitness":
        values = np.asarray([float(row["best_fitness"]) for row in rows[anchor_index : current_index + 1]], dtype=float)
    else:
        values = np.asarray([float(item[name]) for item in stats[anchor_index : current_index + 1]], dtype=float)
    centered_ratios = ratios - float(np.mean(ratios))
    denominator = float(np.sum(centered_ratios * centered_ratios))
    if denominator <= EPS:
        return None
    centered_values = values - float(np.mean(values))
    return float(np.sum(centered_ratios * centered_values) / denominator)


def _population_overlap(current: dict, anchor: dict | None, population_diversity: float) -> float | None:
    if anchor is None:
        return None
    current_population = _population(current)
    anchor_population = _population(anchor)
    if current_population.shape[1] != anchor_population.shape[1]:
        return None
    dimension = int(current["dimension"])
    radius = 0.05 * max(population_diversity, EPS)
    distances = np.linalg.norm(current_population[:, None, :] - anchor_population[None, :, :], axis=2) / sqrt(dimension)
    nearest = np.min(distances, axis=1)
    return float(np.mean(nearest <= radius))


def _best_distance_fitness_corr(current: dict) -> float | None:
    population = _population(current)
    fitness = _fitness(current)
    if population.shape[0] < 2:
        return None
    best = population[int(np.argmin(fitness))]
    distances = np.linalg.norm(population - best, axis=1) / sqrt(int(current["dimension"]))
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
    anchor: dict | None,
    current_stats: dict[str, float],
    anchor_stats: dict[str, float] | None,
) -> float | None:
    if anchor is None or anchor_stats is None:
        return None
    ratio_delta = _effective_ratio_delta(current, anchor)
    if ratio_delta <= 0.0:
        return None
    return float((anchor_stats["diversity"] - current_stats["diversity"]) / max(anchor_stats["diversity"], EPS) / ratio_delta)
