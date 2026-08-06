from __future__ import annotations

from collections import defaultdict
from math import log, sqrt
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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

BEHAVIOR_FEATURE_COLUMNS = (
    "bf_fe_ratio",
    "bf_improvement_rate_w02",
    "bf_improvement_frequency_w02",
    "bf_diversity_mean_pairwise",
    "bf_diversity_change_w05",
    "bf_directional_entropy_w05",
    "bf_distance_decay_w10",
    "bf_stagnation_w10",
    "bf_convergence_rate_w10",
)

BEHAVIOR_COLUMNS = BEHAVIOR_METADATA_COLUMNS + BEHAVIOR_FEATURE_COLUMNS

BEHAVIOR_SCHEMA = pa.schema(
    [
        ("problem_id", pa.string()),
        ("family", pa.string()),
        ("dimension", pa.int32()),
        ("algorithm", pa.string()),
        ("seed", pa.int64()),
        ("FE", pa.int64()),
        ("FE_ratio", pa.float64()),
        ("bf_fe_ratio", pa.float64()),
        ("bf_improvement_rate_w02", pa.float64()),
        ("bf_improvement_frequency_w02", pa.float64()),
        ("bf_diversity_mean_pairwise", pa.float64()),
        ("bf_diversity_change_w05", pa.float64()),
        ("bf_directional_entropy_w05", pa.float64()),
        ("bf_distance_decay_w10", pa.float64()),
        ("bf_stagnation_w10", pa.float64()),
        ("bf_convergence_rate_w10", pa.float64()),
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
        last_improvement_ratio = float(ordered[0]["FE_ratio"])

        for index, row in enumerate(ordered):
            if index > 0 and _strict_improvement(ordered[index - 1]["best_fitness"], row["best_fitness"]):
                last_improvement_ratio = float(row["FE_ratio"])

            short_anchor = _find_anchor(ordered, index, WINDOW_SHORT)
            medium_anchor = _find_anchor(ordered, index, WINDOW_MEDIUM)
            long_anchor = _find_anchor(ordered, index, WINDOW_LONG)

            metadata = {column: row[column] for column in BEHAVIOR_METADATA_COLUMNS}
            current_ratio = float(row["FE_ratio"])
            behavior_rows.append(
                {
                    "_source_position": row["_source_position"],
                    **metadata,
                    "bf_fe_ratio": current_ratio,
                    "bf_improvement_rate_w02": _improvement_rate(row, short_anchor),
                    "bf_improvement_frequency_w02": _improvement_frequency(ordered, index, short_anchor),
                    "bf_diversity_mean_pairwise": stats[index]["diversity"],
                    "bf_diversity_change_w05": _relative_change(stats[index]["diversity"], _anchor_stat(stats, medium_anchor, "diversity")),
                    "bf_directional_entropy_w05": _directional_entropy(row, medium_anchor),
                    "bf_distance_decay_w10": _relative_decay(stats[index]["distance_to_best"], _anchor_stat(stats, long_anchor, "distance_to_best")),
                    "bf_stagnation_w10": min(max(current_ratio - last_improvement_ratio, 0.0), WINDOW_LONG) / WINDOW_LONG,
                    "bf_convergence_rate_w10": _convergence_rate(row, long_anchor, stats[index], _anchor_stats(stats, long_anchor)),
                }
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
    }


def _population(row: dict) -> np.ndarray:
    population = np.asarray(row["population"], dtype=float)
    if population.ndim != 2:
        raise ValueError("population must be two-dimensional")
    if population.shape[1] != int(row["dimension"]):
        raise ValueError("population width must match dimension")
    return population


def _fitness(row: dict) -> np.ndarray:
    fitness = np.asarray(row["fitness"], dtype=float).reshape(-1)
    if fitness.shape[0] != len(row["population"]):
        raise ValueError("fitness length must match population rows")
    return fitness


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


def _find_anchor(rows: list[dict], current_index: int, window: float) -> dict | None:
    target = float(rows[current_index]["FE_ratio"]) - window
    anchor = None
    for row in rows[:current_index]:
        if float(row["FE_ratio"]) <= target + EPS:
            anchor = row
        else:
            break
    return anchor


def _strict_improvement(previous_best: float, current_best: float) -> bool:
    threshold = EPS * max(1.0, abs(float(previous_best)))
    return float(previous_best) - float(current_best) > threshold


def _improvement_rate(current: dict, anchor: dict | None) -> float | None:
    if anchor is None:
        return None
    ratio_delta = float(current["FE_ratio"]) - float(anchor["FE_ratio"])
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


def _directional_entropy(current: dict, anchor: dict | None) -> float | None:
    if anchor is None:
        return None
    current_population = _population(current)
    anchor_population = _population(anchor)
    if current_population.shape != anchor_population.shape:
        return None

    displacement = current_population - anchor_population
    norms = np.linalg.norm(displacement, axis=1)
    valid = norms > EPS
    if not np.any(valid):
        return None

    mean_displacement = np.mean(displacement[valid], axis=0)
    mean_norm = float(np.linalg.norm(mean_displacement))
    if mean_norm <= EPS:
        return None

    cosines = np.clip(displacement[valid] @ mean_displacement / (norms[valid] * mean_norm), -1.0, 1.0)
    counts = np.zeros(5, dtype=float)
    counts[0] = float(np.sum(cosines < -0.5))
    counts[1] = float(np.sum((cosines >= -0.5) & (cosines < 0.0)))
    counts[2] = float(np.sum((cosines >= 0.0) & (cosines < 0.5)))
    counts[3] = float(np.sum(cosines >= 0.5))
    probabilities = counts[counts > 0.0] / float(np.sum(counts))
    if probabilities.size == 0:
        return None
    return float(-np.sum(probabilities * np.log(probabilities)) / log(5.0))


def _convergence_rate(
    current: dict,
    anchor: dict | None,
    current_stats: dict[str, float],
    anchor_stats: dict[str, float] | None,
) -> float | None:
    if anchor is None or anchor_stats is None:
        return None
    ratio_delta = float(current["FE_ratio"]) - float(anchor["FE_ratio"])
    if ratio_delta <= 0.0:
        return None
    return float((anchor_stats["diversity"] - current_stats["diversity"]) / max(anchor_stats["diversity"], EPS) / ratio_delta)
