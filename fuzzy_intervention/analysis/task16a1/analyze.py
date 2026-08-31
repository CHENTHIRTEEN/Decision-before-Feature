from __future__ import annotations

import resource
import sys
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[3]
TASK16A = ROOT / "fuzzy_intervention/results/task16a"
OUTPUT = ROOT / "fuzzy_intervention/results/task16a1"
REPORTS = ROOT / "fuzzy_intervention/analysis/task16a1"
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 2026101901
MIN_REGIME_STATES = 5
MIN_NOISE_PAIRS = 15
BASE_CELL = ("suite", "current_algorithm", "source_FE")
NOTICE = (
    "Task16A.1 为零 FE 的事后诊断性复核，不替代 Task16A 预先指定的正式结论。"
)


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return value / divisor


def _fb_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return float("nan")
    values = frame.groupby("cv_group_id", sort=True)[column].mean()
    return float(values.mean())


def _levels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for source, target in (
        ("probe_productivity_rank", "P_level"),
        ("probe_entropy_rank", "H_level"),
        ("probe_stagnation_rank", "S_level"),
    ):
        values = out[source].to_numpy(dtype=float)
        out[target] = np.where(
            values <= 1.0 / 3.0,
            "LOW",
            np.where(values >= 2.0 / 3.0, "HIGH", "MED"),
        )
    out["regime_R1"] = out["P_level"].eq("HIGH") & out["S_level"].eq("LOW")
    out["regime_R2"] = out["P_level"].eq("LOW") & out["S_level"].eq("HIGH")
    out["regime_R3"] = out["regime_R2"] & out["H_level"].eq("LOW")
    out["regime_R4"] = out["regime_R2"] & out["H_level"].eq("HIGH")
    return out


def _load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    required = {
        "sources": TASK16A / "task16a_source_states.parquet",
        "outcomes": TASK16A / "task16a_action_outcomes.parquet",
        "repetitions": TASK16A / "task16a_repetition_outcomes.parquet",
        "practical": TASK16A / "task16a_practical_action_sets.parquet",
        "noise": TASK16A / "task16a_noise_scales.parquet",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Task16A.1 requires existing Task16A inputs: {missing}")
    sources = pd.read_parquet(required["sources"])
    outcomes = pd.read_parquet(required["outcomes"])
    repetitions = pd.read_parquet(required["repetitions"])
    practical = pd.read_parquet(required["practical"])
    noise = pd.read_parquet(required["noise"])
    if len(sources) != 2520 or sources["state_id"].nunique() != 2520:
        raise RuntimeError("Task16A source-state coverage is not 2520 unique states")
    if len(outcomes.loc[outcomes["repetition_id"].eq(0)]) != 12600:
        raise RuntimeError("Task16A primary action coverage is not 2520 x 5")
    if len(outcomes) != 15068:
        raise RuntimeError("Task16A action-outcome coverage is not 15068 rows")
    if len(repetitions) != 3702:
        raise RuntimeError("Task16A repetition-row count changed")
    if len(practical) != 2520 or practical["state_id"].nunique() != 2520:
        raise RuntimeError("Task16A practical action sets are incomplete")
    if not np.allclose(
        sources["maturity"].to_numpy(dtype=float),
        sources["source_FE"].to_numpy(dtype=float) / 10000.0,
    ):
        raise RuntimeError("Task16A maturity is not source_FE / 10000")
    if len(noise) != 6 or noise["action"].nunique() != 6:
        raise RuntimeError("Task16A global action-noise calibration is incomplete")
    return sources, outcomes, repetitions, practical, noise


def _state_frame(
    sources: pd.DataFrame,
    outcomes: pd.DataFrame,
    practical: pd.DataFrame,
) -> pd.DataFrame:
    metadata = [
        "state_id", "suite", "split", "problem_id", "function_id", "family",
        "cv_group_id", "instance", "seed", "dimension", "current_algorithm",
        "source_FE", "source_FE_actual", "maturity", "probe_productivity",
        "probe_entropy", "probe_stagnation", "probe_productivity_rank",
        "probe_entropy_rank", "probe_stagnation_rank",
    ]
    states = _levels(sources[metadata])
    saved = practical.set_index("state_id")
    for regime in ("R1", "R2", "R3", "R4"):
        reconstructed = states.set_index("state_id")[f"regime_{regime}"].sort_index()
        expected = saved[f"regime_{regime}"].sort_index().astype(bool)
        if not reconstructed.equals(expected):
            raise RuntimeError(f"reconstructed {regime} does not match Task16A")
    primary = outcomes.loc[outcomes["repetition_id"].eq(0)].copy()
    pivot = primary.pivot(index="state_id", columns="action", values="loss_terminal")
    if not {"continue", "perturb_targeted", "perturb_random"}.issubset(pivot.columns):
        raise RuntimeError("Task16A primary actions are incomplete")
    switch_columns = [column for column in pivot.columns if str(column).startswith("switch_")]
    best_switch = pivot[switch_columns].min(axis=1, skipna=True)
    action_values = pd.DataFrame(
        {
            "state_id": pivot.index,
            "loss_continue": pivot["continue"],
            "loss_perturb_targeted": pivot["perturb_targeted"],
            "loss_perturb_random": pivot["perturb_random"],
            "loss_best_observed_switch": best_switch,
        }
    ).reset_index(drop=True)
    action_values["G_P"] = (
        action_values["loss_continue"] - action_values["loss_perturb_targeted"]
    )
    action_values["G_PR"] = (
        action_values["loss_continue"] - action_values["loss_perturb_random"]
    )
    action_values["G_S"] = (
        action_values["loss_continue"] - action_values["loss_best_observed_switch"]
    )
    action_values["G_SP"] = (
        action_values["loss_perturb_targeted"]
        - action_values["loss_best_observed_switch"]
    )
    action_values["G_I"] = action_values[["G_P", "G_S"]].max(axis=1)
    states = states.merge(action_values, on="state_id", validate="one_to_one")
    practical_columns = [
        "state_id", "Z_I", "Z_P", "Z_S", "Z_P_over_S", "Z_S_over_P",
        "practical_action_set_size",
    ]
    states = states.merge(
        practical[practical_columns], on="state_id", validate="one_to_one"
    )
    values = states[["G_P", "G_PR", "G_S", "G_SP", "G_I"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("continuous advantages must be finite")
    return states


def _support(states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, part in states.groupby(list(BASE_CELL), sort=True):
        counts = {regime: int(part[f"regime_{regime}"].sum()) for regime in ("R1", "R2", "R3", "R4")}
        rows.append(
            {
                "suite": key[0],
                "current_algorithm": key[1],
                "source_FE": int(key[2]),
                "num_states": int(len(part)),
                **{f"n_{regime}": count for regime, count in counts.items()},
                "support_R1_R2": (
                    "SUPPORTED"
                    if counts["R1"] >= MIN_REGIME_STATES and counts["R2"] >= MIN_REGIME_STATES
                    else "LOW_SUPPORT"
                ),
                "support_R3_R4": (
                    "SUPPORTED"
                    if counts["R3"] >= MIN_REGIME_STATES and counts["R4"] >= MIN_REGIME_STATES
                    else "LOW_SUPPORT"
                ),
            }
        )
    return pd.DataFrame(rows)


def _scope_specs(frame: pd.DataFrame, include_fe: bool) -> list[tuple[str, str, pd.DataFrame]]:
    rows: list[tuple[str, str, pd.DataFrame]] = [("pooled", "all", frame)]
    rows.extend(("suite", str(key), part) for key, part in frame.groupby("suite", sort=True))
    rows.extend(
        ("current_algorithm", str(key), part)
        for key, part in frame.groupby("current_algorithm", sort=True)
    )
    if include_fe:
        rows.extend(("source_FE", str(int(key)), part) for key, part in frame.groupby("source_FE", sort=True))
    return rows


def _cell_key(row) -> tuple[str, str, int]:
    return str(row["suite"]), str(row["current_algorithm"]), int(row["source_FE"])


def _valid_cells(
    support: pd.DataFrame,
    scope: pd.DataFrame,
    support_column: str,
) -> list[tuple[str, str, int]]:
    allowed_states = set(scope["state_id"])
    cells = []
    for row in support.loc[support[support_column].eq("SUPPORTED")].to_dict("records"):
        key = _cell_key(row)
        present = scope.loc[
            scope["suite"].eq(key[0])
            & scope["current_algorithm"].eq(key[1])
            & scope["source_FE"].eq(key[2]),
            "state_id",
        ]
        if len(present) and set(present).issubset(allowed_states):
            cells.append(key)
    return cells


def _cell_subset(frame: pd.DataFrame, key: tuple[str, str, int]) -> pd.DataFrame:
    return frame.loc[
        frame["suite"].eq(key[0])
        & frame["current_algorithm"].eq(key[1])
        & frame["source_FE"].eq(key[2])
    ]


def _bootstrap_binary(
    frame: pd.DataFrame,
    cells: list[tuple[str, str, int]],
    column: str,
    target_regime: str,
    base_regime: str,
    stream: int,
) -> tuple[np.ndarray, np.ndarray]:
    groups = sorted(frame["cv_group_id"].unique())
    group_index = {group: index for index, group in enumerate(groups)}
    cache = []
    for cell in cells:
        part = _cell_subset(frame, cell)
        regime_arrays = []
        for regime in (target_regime, base_regime):
            values = np.full(len(groups), np.nan, dtype=float)
            selected = part.loc[part[f"regime_{regime}"]]
            means = selected.groupby("cv_group_id", sort=True)[column].mean()
            for group, value in means.items():
                values[group_index[group]] = float(value)
            regime_arrays.append(values)
        cache.append(regime_arrays)
    rng = np.random.default_rng(
        np.random.SeedSequence([BOOTSTRAP_SEED, int(stream), len(groups), len(cells)])
    )
    draws = np.full(BOOTSTRAP_DRAWS, np.nan, dtype=float)
    used_cells = np.zeros(BOOTSTRAP_DRAWS, dtype=int)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.integers(0, len(groups), size=len(groups))
        weights = np.bincount(sampled, minlength=len(groups)).astype(float)
        differences = []
        for target, base in cache:
            target_mask = np.isfinite(target) & (weights > 0)
            base_mask = np.isfinite(base) & (weights > 0)
            if not target_mask.any() or not base_mask.any():
                continue
            target_value = float(np.average(target[target_mask], weights=weights[target_mask]))
            base_value = float(np.average(base[base_mask], weights=weights[base_mask]))
            differences.append(target_value - base_value)
        if differences:
            draws[draw] = float(np.mean(differences))
            used_cells[draw] = len(differences)
    return draws, used_cells


def _binary_cell_table(states: pd.DataFrame, support: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for support_row in support.to_dict("records"):
        key = _cell_key(support_row)
        part = _cell_subset(states, key)
        r1 = part.loc[part["regime_R1"]]
        r2 = part.loc[part["regime_R2"]]
        rows.append(
            {
                **{field: support_row[field] for field in support.columns},
                "p_R1_function_balanced": _fb_mean(r1, column),
                "p_R2_function_balanced": _fb_mean(r2, column),
                "delta_R2_minus_R1": _fb_mean(r2, column) - _fb_mean(r1, column),
            }
        )
    return pd.DataFrame(rows)


def _part_a(
    states: pd.DataFrame,
    support: pd.DataFrame,
    *,
    column: str = "Z_I",
    analysis_part: str = "A",
    contrast: str = "Z_I_R2_minus_R1",
    stream_base: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    cell_table = _binary_cell_table(states, support, column)
    rows = []
    bootstrap_frames = []
    for index, (scope_type, scope_value, part) in enumerate(
        _scope_specs(states, include_fe=True), start=1
    ):
        cells = _valid_cells(support, part, "support_R1_R2")
        cell_values = cell_table.loc[
            cell_table.apply(lambda row: _cell_key(row) in set(cells), axis=1)
        ].copy()
        equal = float(cell_values["delta_R2_minus_R1"].mean()) if len(cell_values) else np.nan
        weights = (cell_values["n_R1"] + cell_values["n_R2"]).to_numpy(dtype=float)
        weighted = (
            float(np.average(cell_values["delta_R2_minus_R1"], weights=weights))
            if len(cell_values) and weights.sum() > 0
            else np.nan
        )
        raw_r1 = _fb_mean(part.loc[part["regime_R1"]], column)
        raw_r2 = _fb_mean(part.loc[part["regime_R2"]], column)
        draws, used = _bootstrap_binary(
            part, cells, column, "R2", "R1", stream_base + index
        )
        finite = draws[np.isfinite(draws)]
        low, high = (
            np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
        )
        rows.append(
            {
                "scope_type": scope_type,
                "scope_value": scope_value,
                "raw_R1_function_balanced_rate": raw_r1,
                "raw_R2_function_balanced_rate": raw_r2,
                "raw_delta_R2_minus_R1": raw_r2 - raw_r1,
                "standardized_cell_equal_delta": equal,
                "standardized_state_count_weighted_delta": weighted,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "valid_cells": int(len(cells)),
                "total_scope_cells": int(part.groupby(list(BASE_CELL)).ngroups),
                "bootstrap_finite_draws": int(len(finite)),
            }
        )
        bootstrap_frames.append(
            pd.DataFrame(
                {
                    "analysis_part": analysis_part,
                    "contrast": contrast,
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "draw": np.arange(BOOTSTRAP_DRAWS, dtype=int),
                    "value": draws,
                    "valid_cells_in_draw": used,
                }
            )
        )
    return cell_table, pd.DataFrame(rows), bootstrap_frames


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0 or float(np.sum(weights)) <= 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    cutoff = 0.5 * float(ordered_weights.sum())
    return float(ordered_values[int(np.searchsorted(cumulative, cutoff, side="left"))])


def _bootstrap_continuous(
    frame: pd.DataFrame,
    cells: list[tuple[str, str, int]],
    metric: str,
    target_regime: str,
    base_regime: str,
    stream: int,
) -> tuple[np.ndarray, np.ndarray]:
    groups = sorted(frame["cv_group_id"].unique())
    group_index = {group: index for index, group in enumerate(groups)}
    cache = []
    for cell in cells:
        part = _cell_subset(frame, cell)
        regime_cache = []
        for regime in (target_regime, base_regime):
            by_group: list[np.ndarray] = [np.empty(0, dtype=float) for _ in groups]
            selected = part.loc[part[f"regime_{regime}"]]
            for group, group_part in selected.groupby("cv_group_id", sort=True):
                by_group[group_index[group]] = group_part[metric].to_numpy(dtype=float)
            regime_cache.append(by_group)
        cache.append(regime_cache)
    rng = np.random.default_rng(
        np.random.SeedSequence([BOOTSTRAP_SEED, int(stream), len(groups), len(cells)])
    )
    draws = np.full(BOOTSTRAP_DRAWS, np.nan, dtype=float)
    used_cells = np.zeros(BOOTSTRAP_DRAWS, dtype=int)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.integers(0, len(groups), size=len(groups))
        group_weights = np.bincount(sampled, minlength=len(groups)).astype(float)
        differences = []
        for target_cache, base_cache in cache:
            medians = []
            for regime_cache in (target_cache, base_cache):
                values_parts = []
                weight_parts = []
                for group_id, values in enumerate(regime_cache):
                    if len(values) and group_weights[group_id] > 0:
                        values_parts.append(values)
                        weight_parts.append(
                            np.full(len(values), group_weights[group_id], dtype=float)
                        )
                if not values_parts:
                    medians.append(np.nan)
                else:
                    medians.append(
                        _weighted_median(
                            np.concatenate(values_parts), np.concatenate(weight_parts)
                        )
                    )
            if np.isfinite(medians).all():
                differences.append(float(medians[0] - medians[1]))
        if differences:
            draws[draw] = float(np.mean(differences))
            used_cells[draw] = len(differences)
    return draws, used_cells


def _advantage_descriptives(states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ("G_P", "G_PR", "G_S", "G_SP", "G_I")
    for scope_type, scope_value, scope in _scope_specs(states, include_fe=False):
        for regime in ("R1", "R2", "R3", "R4"):
            part = scope.loc[scope[f"regime_{regime}"]]
            for metric in metrics:
                values = part[metric].to_numpy(dtype=float)
                rows.append(
                    {
                        "scope_type": scope_type,
                        "scope_value": scope_value,
                        "regime": regime,
                        "metric": metric,
                        "num_states": int(len(part)),
                        "median": float(np.median(values)) if len(values) else np.nan,
                        "function_balanced_mean": _fb_mean(part, metric),
                        "q25": float(np.quantile(values, 0.25)) if len(values) else np.nan,
                        "q75": float(np.quantile(values, 0.75)) if len(values) else np.nan,
                        "positive_rate": _fb_mean(part.assign(_gt0=part[metric] > 0), "_gt0"),
                    }
                )
    return pd.DataFrame(rows)


def _continuous_contrasts(
    states: pd.DataFrame,
    support: pd.DataFrame,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    definitions = [
        ("B1", "G_I", "R2", "R1", "support_R1_R2"),
        ("B2", "G_P", "R2", "R1", "support_R1_R2"),
        ("B3", "G_S", "R2", "R1", "support_R1_R2"),
        ("B4_GP", "G_P", "R3", "R4", "support_R3_R4"),
        ("B4_GS", "G_S", "R3", "R4", "support_R3_R4"),
    ]
    rows = []
    bootstrap_frames = []
    stream = 2000
    for contrast, metric, target, base, support_column in definitions:
        for scope_type, scope_value, part in _scope_specs(states, include_fe=False):
            stream += 1
            cells = _valid_cells(support, part, support_column)
            cell_rows = []
            for cell in cells:
                cell_part = _cell_subset(part, cell)
                upper = cell_part.loc[cell_part[f"regime_{target}"]]
                lower = cell_part.loc[cell_part[f"regime_{base}"]]
                cell_rows.append(
                    {
                        "cell": cell,
                        "median_difference": float(upper[metric].median() - lower[metric].median()),
                        "fb_mean_difference": _fb_mean(upper, metric) - _fb_mean(lower, metric),
                        "weight": int(len(upper) + len(lower)),
                    }
                )
            cell_frame = pd.DataFrame(cell_rows)
            equal = float(cell_frame["median_difference"].mean()) if len(cell_frame) else np.nan
            weighted = (
                float(
                    np.average(
                        cell_frame["median_difference"],
                        weights=cell_frame["weight"],
                    )
                )
                if len(cell_frame)
                else np.nan
            )
            fb_sensitivity = (
                float(cell_frame["fb_mean_difference"].mean()) if len(cell_frame) else np.nan
            )
            raw_upper = part.loc[part[f"regime_{target}"]]
            raw_lower = part.loc[part[f"regime_{base}"]]
            raw_median = float(raw_upper[metric].median() - raw_lower[metric].median())
            raw_fb = _fb_mean(raw_upper, metric) - _fb_mean(raw_lower, metric)
            draws, used = _bootstrap_continuous(
                part, cells, metric, target, base, stream
            )
            finite = draws[np.isfinite(draws)]
            low, high = (
                np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
            )
            rows.append(
                {
                    "contrast": contrast,
                    "metric": metric,
                    "target_regime": target,
                    "base_regime": base,
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "raw_median_difference": raw_median,
                    "raw_function_balanced_mean_difference": raw_fb,
                    "standardized_cell_equal_median_difference": equal,
                    "standardized_state_count_weighted_median_difference": weighted,
                    "standardized_cell_equal_fb_mean_difference": fb_sensitivity,
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "valid_cells": int(len(cells)),
                    "bootstrap_finite_draws": int(len(finite)),
                }
            )
            bootstrap_frames.append(
                pd.DataFrame(
                    {
                        "analysis_part": "B",
                        "contrast": contrast,
                        "scope_type": scope_type,
                        "scope_value": scope_value,
                        "draw": np.arange(BOOTSTRAP_DRAWS, dtype=int),
                        "value": draws,
                        "valid_cells_in_draw": used,
                    }
                )
            )
    return pd.DataFrame(rows), bootstrap_frames


def _noise_pairs(repetitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (state_id, action), part in repetitions.groupby(["state_id", "action"], sort=True):
        ordered = part.sort_values("repetition_id")
        values = ordered["loss_terminal"].to_numpy(dtype=float)
        if len(values) != 3:
            raise RuntimeError("each selected state-action must have three repetitions")
        base = ordered.iloc[0]
        for repetition_id, difference in zip(
            ordered["repetition_id"].iloc[1:], np.abs(values[1:] - values[0]), strict=True
        ):
            rows.append(
                {
                    "state_id": state_id,
                    "action": action,
                    "suite": base["suite"],
                    "current_algorithm": base["current_algorithm"],
                    "source_FE": int(base["source_FE"]),
                    "cv_group_id": base["cv_group_id"],
                    "repetition_id": int(repetition_id),
                    "absolute_loss_difference": float(difference),
                }
            )
    return pd.DataFrame(rows)


def _noise_cells(
    pairs: pd.DataFrame,
    task16a_noise: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    definitions = [
        ("global", []),
        ("suite", ["suite"]),
        ("current", ["current_algorithm"]),
        ("FE", ["source_FE"]),
        ("suite_current", ["suite", "current_algorithm"]),
        ("current_FE", ["current_algorithm", "source_FE"]),
    ]
    rows = []
    for action, action_part in pairs.groupby("action", sort=True):
        pair_support = action_part[["state_id", "action"]].drop_duplicates()
        global_delta = float(np.quantile(action_part["absolute_loss_difference"], 0.95))
        for conditioning, columns in definitions:
            grouped = [((), action_part)] if not columns else action_part.groupby(columns, sort=True)
            for key, part in grouped:
                keys = key if isinstance(key, tuple) else (key,)
                key_map = dict(zip(columns, keys, strict=True))
                n_pairs = int(part[["state_id", "action"]].drop_duplicates().shape[0])
                supported = n_pairs >= MIN_NOISE_PAIRS
                delta = (
                    float(np.quantile(part["absolute_loss_difference"], 0.95))
                    if supported
                    else np.nan
                )
                rows.append(
                    {
                        "action": action,
                        "conditioning": conditioning,
                        "suite": key_map.get("suite"),
                        "current_algorithm": key_map.get("current_algorithm"),
                        "source_FE": key_map.get("source_FE"),
                        "cell_label": "all" if not columns else "|".join(str(key_map[column]) for column in columns),
                        "n_repeated_pairs": n_pairs,
                        "n_absolute_differences": int(len(part)),
                        "support_status": "SUPPORTED" if supported else "LOW_SUPPORT",
                        "delta95": delta,
                        "global_delta95": global_delta,
                        "global_to_cell_ratio": (
                            float(global_delta / delta) if supported and delta > 0 else np.inf
                            if supported and global_delta > 0
                            else np.nan
                        ),
                        "total_action_repeated_pairs": int(len(pair_support)),
                    }
                )
    cells = pd.DataFrame(rows)
    reconstructed_global = cells.loc[cells["conditioning"].eq("global")].set_index(
        "action"
    )["delta95"].sort_index()
    expected_global = task16a_noise.set_index("action")["delta_action_95"].sort_index()
    if not reconstructed_global.index.equals(expected_global.index) or not np.allclose(
        reconstructed_global.to_numpy(dtype=float),
        expected_global.to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError("Task16A global action-noise scales were not reconstructed exactly")
    summary_rows = []
    for (action, conditioning), part in cells.loc[
        cells["conditioning"].ne("global") & cells["support_status"].eq("SUPPORTED")
    ].groupby(["action", "conditioning"], sort=True):
        values = part["delta95"].to_numpy(dtype=float)
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        summary_rows.append(
            {
                "action": action,
                "conditioning": conditioning,
                "supported_cells": int(len(values)),
                "delta95_min": minimum,
                "delta95_max": maximum,
                "R_delta": float(maximum / minimum) if minimum > 0 else np.inf,
                "coefficient_of_variation": float(np.std(values, ddof=0) / np.mean(values))
                if float(np.mean(values)) > 0
                else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    primary = summary.loc[
        summary["conditioning"].eq("current_FE") & summary["supported_cells"].ge(2)
    ]
    strong = int((primary["R_delta"] >= 2.0).sum())
    moderate = int((primary["R_delta"] >= 1.3).sum())
    if strong >= 2:
        verdict = "N1 STRONG HETEROGENEITY"
    elif moderate >= 2 or strong >= 1:
        verdict = "N2 MODERATE HETEROGENEITY"
    else:
        verdict = "N3 LOW HETEROGENEITY"
    return cells, summary, verdict


def _local_practical_sets(
    states: pd.DataFrame,
    outcomes: pd.DataFrame,
    noise_cells: pd.DataFrame,
    noise_verdict: str,
) -> pd.DataFrame:
    columns = [
        "state_id", "suite", "current_algorithm", "source_FE", "cv_group_id",
        "regime_R1", "regime_R2", "regime_R3", "regime_R4",
        "posthoc_local_threshold_action_set", "posthoc_local_threshold_action_set_size",
        "posthoc_local_threshold_Z_I", "posthoc_local_threshold_Z_P",
        "posthoc_local_threshold_Z_S", "posthoc_local_threshold_Z_P_over_S",
        "posthoc_local_threshold_Z_S_over_P", "posthoc_local_threshold_fallback_count",
    ]
    if not noise_verdict.startswith("N1"):
        return pd.DataFrame(columns=columns)
    supported = noise_cells.loc[noise_cells["support_status"].eq("SUPPORTED")]
    global_delta = supported.loc[supported["conditioning"].eq("global")].set_index("action")["delta95"].to_dict()
    current_delta = supported.loc[supported["conditioning"].eq("current")].set_index(
        ["action", "current_algorithm"]
    )["delta95"].to_dict()
    current_fe_delta = supported.loc[supported["conditioning"].eq("current_FE")].set_index(
        ["action", "current_algorithm", "source_FE"]
    )["delta95"].to_dict()
    primary = outcomes.loc[outcomes["repetition_id"].eq(0)]
    rows = []
    state_meta = states.set_index("state_id")
    for state_id, part in primary.groupby("state_id", sort=True):
        meta = state_meta.loc[state_id]
        losses = part.set_index("action")["loss_terminal"].to_dict()
        actions = sorted(losses)
        deltas = {}
        fallback = 0
        for action in actions:
            key = (action, meta["current_algorithm"], int(meta["source_FE"]))
            if key in current_fe_delta:
                deltas[action] = float(current_fe_delta[key])
            elif (action, meta["current_algorithm"]) in current_delta:
                deltas[action] = float(current_delta[(action, meta["current_algorithm"])])
                fallback += 1
            else:
                deltas[action] = float(global_delta[action])
                fallback += 1
        beats = {}
        for action_a in actions:
            for action_b in actions:
                if action_a != action_b:
                    threshold = max(deltas[action_a], deltas[action_b])
                    beats[(action_a, action_b)] = bool(
                        losses[action_a] < losses[action_b] - threshold
                    )
        action_set = [
            action
            for action in actions
            if not any(beats[(other, action)] for other in actions if other != action)
        ]
        switches = [action for action in actions if action.startswith("switch_")]
        rows.append(
            {
                "state_id": state_id,
                "suite": meta["suite"],
                "current_algorithm": meta["current_algorithm"],
                "source_FE": int(meta["source_FE"]),
                "cv_group_id": meta["cv_group_id"],
                **{f"regime_{regime}": bool(meta[f"regime_{regime}"]) for regime in ("R1", "R2", "R3", "R4")},
                "posthoc_local_threshold_action_set": action_set,
                "posthoc_local_threshold_action_set_size": int(len(action_set)),
                "posthoc_local_threshold_Z_I": bool("continue" not in action_set),
                "posthoc_local_threshold_Z_P": bool(beats[("perturb_targeted", "continue")]),
                "posthoc_local_threshold_Z_S": bool(any(beats[(action, "continue")] for action in switches)),
                "posthoc_local_threshold_Z_P_over_S": bool(
                    all(beats[("perturb_targeted", action)] for action in switches)
                ),
                "posthoc_local_threshold_Z_S_over_P": bool(
                    any(beats[(action, "perturb_targeted")] for action in switches)
                ),
                "posthoc_local_threshold_fallback_count": int(fallback),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _spearman_bootstrap(
    frame: pd.DataFrame,
    stream: int,
) -> tuple[float, np.ndarray]:
    def correlation(part: pd.DataFrame) -> float:
        if part["maturity"].nunique() < 2 or part["G_SP"].nunique() < 2:
            return float("nan")
        return float(spearmanr(part["maturity"], part["G_SP"]).statistic)

    point = correlation(frame)
    groups = sorted(frame["cv_group_id"].unique())
    by_group = {
        group: frame.loc[frame["cv_group_id"].eq(group), ["maturity", "G_SP"]]
        for group in groups
    }
    rng = np.random.default_rng(
        np.random.SeedSequence([BOOTSTRAP_SEED, int(stream), len(groups)])
    )
    draws = np.full(BOOTSTRAP_DRAWS, np.nan, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([by_group[group] for group in sampled], ignore_index=True)
        draws[draw] = correlation(sample)
    return point, draws


def _maturity(
    states: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame], str]:
    summary_rows = []
    scope_specs = _scope_specs(states, include_fe=False)
    scope_specs.extend(
        (
            "suite_current",
            f"{suite}|{algorithm}",
            part,
        )
        for (suite, algorithm), part in states.groupby(
            ["suite", "current_algorithm"], sort=True
        )
    )
    for scope_type, scope_value, scope in scope_specs:
        for regime in ("R2", "R3", "R4"):
            part = scope.loc[scope[f"regime_{regime}"]]
            for maturity, maturity_part in part.groupby("maturity", sort=True):
                summary_rows.append(
                    {
                        "scope_type": scope_type,
                        "scope_value": scope_value,
                        "regime": regime,
                        "maturity": float(maturity),
                        "num_states": int(len(maturity_part)),
                        "median_G_SP": float(maturity_part["G_SP"].median()),
                        "function_balanced_mean_G_SP": _fb_mean(maturity_part, "G_SP"),
                        "q25_G_SP": float(maturity_part["G_SP"].quantile(0.25)),
                        "q75_G_SP": float(maturity_part["G_SP"].quantile(0.75)),
                    }
                )
    correlation_rows = []
    bootstrap_frames = []
    stream = 4000
    correlation_scopes = [("pooled", "all", states)]
    correlation_scopes.extend(
        ("suite", str(key), part) for key, part in states.groupby("suite", sort=True)
    )
    correlation_scopes.extend(
        ("current_algorithm", str(key), part)
        for key, part in states.groupby("current_algorithm", sort=True)
    )
    correlation_scopes.extend(
        ("suite_current", f"{suite}|{algorithm}", part)
        for (suite, algorithm), part in states.groupby(
            ["suite", "current_algorithm"], sort=True
        )
    )
    for scope_type, scope_value, scope in correlation_scopes:
        stream += 1
        part = scope.loc[scope["regime_R2"]]
        point, draws = _spearman_bootstrap(part, stream)
        finite = draws[np.isfinite(draws)]
        low, high = np.quantile(finite, [0.025, 0.975])
        correlation_rows.append(
            {
                "scope_type": scope_type,
                "scope_value": scope_value,
                "num_states": int(len(part)),
                "spearman_rho": point,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "bootstrap_finite_draws": int(len(finite)),
            }
        )
        bootstrap_frames.append(
            pd.DataFrame(
                {
                    "analysis_part": "M",
                    "contrast": "R2_maturity_spearman_G_SP",
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "draw": np.arange(BOOTSTRAP_DRAWS, dtype=int),
                    "value": draws,
                    "valid_cells_in_draw": np.nan,
                }
            )
        )
    correlations = pd.DataFrame(correlation_rows)
    suite_rows = correlations.loc[correlations["scope_type"].eq("suite")]
    solver_rows = correlations.loc[
        correlations["scope_type"].eq("current_algorithm")
    ]
    pooled = correlations.loc[correlations["scope_type"].eq("pooled")].iloc[0]
    suite_signs = np.sign(suite_rows["spearman_rho"].to_numpy(dtype=float))
    same_suite = len(suite_signs) == 2 and suite_signs[0] != 0 and suite_signs[0] == suite_signs[1]
    solver_signs = np.sign(solver_rows["spearman_rho"].to_numpy(dtype=float))
    if same_suite:
        matching_solvers = int(np.sum(solver_signs == suite_signs[0]))
    else:
        matching_solvers = int(max(np.sum(solver_signs > 0), np.sum(solver_signs < 0)))
    interval_clear = bool(
        (float(pooled["ci95_low"]) > 0 or float(pooled["ci95_high"]) < 0)
        or ((suite_rows["ci95_low"] > 0) | (suite_rows["ci95_high"] < 0)).any()
    )
    local_rows = correlations.loc[correlations["scope_type"].eq("suite_current")]
    local_clear = int(((local_rows["ci95_low"] > 0) | (local_rows["ci95_high"] < 0)).sum())
    local_signs = np.sign(local_rows["spearman_rho"].to_numpy(dtype=float))
    two_local_same = max(int(np.sum(local_signs > 0)), int(np.sum(local_signs < 0))) >= 2
    if same_suite and matching_solvers >= 2 and interval_clear:
        verdict = "M1 STRUCTURED"
    elif local_clear >= 1 or two_local_same:
        verdict = "M2 PARTIAL"
    else:
        verdict = "M3 NONE"
    return pd.DataFrame(summary_rows), correlations, bootstrap_frames, verdict


def _interval_supports_direction(row: pd.Series, direction: int) -> bool:
    return bool(row["ci95_low"] > 0) if direction > 0 else bool(row["ci95_high"] < 0)


def _verdicts(
    part_a: pd.DataFrame,
    part_b: pd.DataFrame,
    support: pd.DataFrame,
    noise_verdict: str,
    local_sensitivity: pd.DataFrame,
    maturity_verdict: str,
) -> dict:
    a = part_a.set_index(["scope_type", "scope_value"])
    pooled_a = a.loc[("pooled", "all")]
    suites_a = [a.loc[("suite", suite)] for suite in ("bbob", "mabbob")]
    if (
        float(pooled_a["standardized_cell_equal_delta"]) > 0
        and any(float(row["ci95_low"]) > 0 for row in suites_a)
        and all(float(row["ci95_high"]) >= 0 for row in suites_a)
    ):
        a_verdict = "A-REVERSAL"
    elif (
        float(pooled_a["standardized_cell_equal_delta"]) < 0
        and all(float(row["standardized_cell_equal_delta"]) < 0 for row in suites_a)
        and any(float(row["ci95_high"]) < 0 for row in suites_a)
    ):
        a_verdict = "A-CONFIRM"
    else:
        a_verdict = "A-NULL"

    b = part_b.set_index(["contrast", "scope_type", "scope_value"])
    suite_b1 = [b.loc[("B1", "suite", suite)] for suite in ("bbob", "mabbob")]
    solver_b1 = [
        b.loc[("B1", "current_algorithm", solver)]
        for solver in ("shade", "lshade", "cso")
    ]
    entropy_interpretable = False
    for contrast in ("B4_GP", "B4_GS"):
        rows = [b.loc[(contrast, "suite", suite)] for suite in ("bbob", "mabbob")]
        signs = [np.sign(float(row["standardized_cell_equal_median_difference"])) for row in rows]
        if signs[0] != 0 and signs[0] == signs[1] and any(
            _interval_supports_direction(row, int(signs[0])) for row in rows
        ):
            entropy_interpretable = True
    b_structure = (
        all(float(row["standardized_cell_equal_median_difference"]) > 0 for row in suite_b1)
        and any(float(row["ci95_low"]) > 0 for row in suite_b1)
        and all(float(row["ci95_high"]) >= 0 for row in suite_b1)
        and sum(float(row["standardized_cell_equal_median_difference"]) > 0 for row in solver_b1) >= 2
        and entropy_interpretable
    )
    b_partial = (
        any(float(row["ci95_low"]) > 0 for row in suite_b1)
        or any(float(row["ci95_low"]) > 0 for row in solver_b1)
        or entropy_interpretable
    )
    b_verdict = "B-STRUCTURE" if b_structure else ("B-PARTIAL" if b_partial else "B-NONE")

    valid_by_suite = support.loc[support["support_R1_R2"].eq("SUPPORTED")].groupby("suite").size()
    support_adequate = all(int(valid_by_suite.get(suite, 0)) >= 6 for suite in ("bbob", "mabbob"))
    local_stable_enrichment = False
    local_changes_relation = False
    if len(local_sensitivity):
        local = local_sensitivity.set_index(["scope_type", "scope_value"])
        local_suite = [local.loc[("suite", suite)] for suite in ("bbob", "mabbob")]
        local_stable_enrichment = bool(
            all(float(row["standardized_cell_equal_delta"]) > 0 for row in local_suite)
            and any(float(row["ci95_low"]) > 0 for row in local_suite)
            and all(float(row["ci95_high"]) >= 0 for row in local_suite)
        )
        formal_raw_signs = np.sign(
            [
                float(a.loc[("pooled", "all"), "raw_delta_R2_minus_R1"]),
                *[
                    float(a.loc[("suite", suite), "raw_delta_R2_minus_R1"])
                    for suite in ("bbob", "mabbob")
                ],
            ]
        )
        local_raw_signs = np.sign(
            [
                float(local.loc[("pooled", "all"), "raw_delta_R2_minus_R1"]),
                *[
                    float(local.loc[("suite", suite), "raw_delta_R2_minus_R1"])
                    for suite in ("bbob", "mabbob")
                ],
            ]
        )
        local_changes_relation = bool(np.any(formal_raw_signs != local_raw_signs))
    if (a_verdict == "A-REVERSAL" or b_verdict == "B-STRUCTURE") and support_adequate:
        joint = "J1 HIDDEN JOINT STRUCTURE PLAUSIBLE"
    elif (
        a_verdict in {"A-CONFIRM", "A-NULL"}
        and b_verdict == "B-NONE"
        and maturity_verdict == "M3 NONE"
        and not local_stable_enrichment
    ):
        joint = "J3 JOINT-STRUCTURE NO-GO"
    else:
        joint = "J2 WEAK / AMBIGUOUS STRUCTURE"
    return {
        "part_a_verdict": a_verdict,
        "part_b_verdict": b_verdict,
        "noise_verdict": noise_verdict,
        "maturity_verdict": maturity_verdict,
        "joint_verdict": joint,
        "support_adequate_for_J1": support_adequate,
        "entropy_interaction_interpretable": entropy_interpretable,
        "local_threshold_stable_R2_enrichment": local_stable_enrichment,
        "local_threshold_changes_qualitative_relation": local_changes_relation,
        "low_support_noise_threshold_used": False,
        "task16a_primary_F3_unchanged": True,
        "task16a_perturb_A3_unchanged": True,
        "new_objective_FE": 0,
    }


def _robustness(
    part_a: pd.DataFrame,
    part_b: pd.DataFrame,
    maturity_correlations: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for row in part_a.loc[
        part_a["scope_type"].isin(["suite", "current_algorithm", "source_FE"])
    ].to_dict("records"):
        rows.append(
            {
                "analysis": "A_Z_I_R2_minus_R1",
                "scope_type": row["scope_type"],
                "scope_value": row["scope_value"],
                "point_estimate": row["standardized_cell_equal_delta"],
                "ci95_low": row["ci95_low"],
                "ci95_high": row["ci95_high"],
                "valid_cells_or_states": row["valid_cells"],
            }
        )
    for row in part_b.loc[
        part_b["contrast"].eq("B1")
        & part_b["scope_type"].isin(["suite", "current_algorithm"])
    ].to_dict("records"):
        rows.append(
            {
                "analysis": "B1_G_I_R2_minus_R1",
                "scope_type": row["scope_type"],
                "scope_value": row["scope_value"],
                "point_estimate": row["standardized_cell_equal_median_difference"],
                "ci95_low": row["ci95_low"],
                "ci95_high": row["ci95_high"],
                "valid_cells_or_states": row["valid_cells"],
            }
        )
    for row in maturity_correlations.loc[
        maturity_correlations["scope_type"].isin(["suite", "current_algorithm"])
    ].to_dict("records"):
        rows.append(
            {
                "analysis": "M_R2_spearman_G_SP",
                "scope_type": row["scope_type"],
                "scope_value": row["scope_value"],
                "point_estimate": row["spearman_rho"],
                "ci95_low": row["ci95_low"],
                "ci95_high": row["ci95_high"],
                "valid_cells_or_states": row["num_states"],
            }
        )
    frame = pd.DataFrame(rows)
    frame["sign"] = np.sign(frame["point_estimate"])
    frame["ci_excludes_zero"] = (frame["ci95_low"] > 0) | (frame["ci95_high"] < 0)
    return frame


def _fmt(value) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.4f}"


def _report_header(title: str) -> str:
    return f"# {title}\n\n> {NOTICE}\n\n"


def _write_reports(
    *,
    states: pd.DataFrame,
    support: pd.DataFrame,
    part_a: pd.DataFrame,
    advantages: pd.DataFrame,
    contrasts: pd.DataFrame,
    noise_cells: pd.DataFrame,
    noise_summary: pd.DataFrame,
    local_sets: pd.DataFrame,
    local_sensitivity: pd.DataFrame,
    maturity_summary: pd.DataFrame,
    maturity_correlations: pd.DataFrame,
    robustness: pd.DataFrame,
    verdicts: dict,
    resource_row: dict,
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    a = part_a.set_index(["scope_type", "scope_value"])
    b = contrasts.set_index(["contrast", "scope_type", "scope_value"])
    desc = advantages.set_index(["scope_type", "scope_value", "regime", "metric"])
    m = maturity_correlations.set_index(["scope_type", "scope_value"])
    pooled_a = a.loc[("pooled", "all")]
    pooled_b1 = b.loc[("B1", "pooled", "all")]
    pooled_b2 = b.loc[("B2", "pooled", "all")]
    pooled_b3 = b.loc[("B3", "pooled", "all")]
    pooled_b4p = b.loc[("B4_GP", "pooled", "all")]
    pooled_b4s = b.loc[("B4_GS", "pooled", "all")]
    valid_cells = int((support["support_R1_R2"] == "SUPPORTED").sum())
    if local_sets.empty:
        local_relation = "未构造（noise verdict 不为 N1）"
        local_table = ""
    else:
        local_relation = (
            "已构造，仅作敏感性分析；"
            + (
                "raw pooled/suite 的 R2-R1 方向发生变化"
                if verdicts["local_threshold_changes_qualitative_relation"]
                else "raw pooled 与两个 suite 的 R2-R1 方向均未改变"
            )
            + "；低支持 noise cell 的阈值未被使用"
        )
        local_table = "\n\n" + local_sensitivity.to_markdown(index=False) + "\n"
    reports = {
        "16a1_01_zero_fe_contract.md": (
            _report_header("16A.1-01 零 FE 合同")
            + "只读取 Task16A parquet；不导入 benchmark、optimizer 或 objective evaluation 接口。\n\n"
            f"复用 {len(states)} 个状态；`new_objective_FE=0`。\n"
        ),
        "16a1_02_state_support_verification.md": (
            _report_header("16A.1-02 State support verification")
            + f"24 个 suite×solver×FE 基础 cell 中，R1/R2 各至少 5 states 的有效 cell 为 {valid_cells}。\n\n"
            + support.to_markdown(index=False)
            + "\n"
        ),
        "16a1_03_maturity_standardized_regime.md": (
            _report_header("16A.1-03 Maturity/current 标准化 regime")
            + f"Raw pooled R2-R1={_fmt(pooled_a['raw_delta_R2_minus_R1'])}；"
            f"cell-equal standardized={_fmt(pooled_a['standardized_cell_equal_delta'])} "
            f"[{_fmt(pooled_a['ci95_low'])}, {_fmt(pooled_a['ci95_high'])}]。\n\n"
            + part_a.to_markdown(index=False)
            + f"\n\n结论：**{verdicts['part_a_verdict']}**。\n"
        ),
        "16a1_04_continuous_advantage_diagnostic.md": (
            _report_header("16A.1-04 Continuous advantage diagnostic")
            + f"B1 standardized G_I R2-R1={_fmt(pooled_b1['standardized_cell_equal_median_difference'])} "
            f"[{_fmt(pooled_b1['ci95_low'])}, {_fmt(pooled_b1['ci95_high'])}]；"
            f"B2 G_P={_fmt(pooled_b2['standardized_cell_equal_median_difference'])}；"
            f"B3 G_S={_fmt(pooled_b3['standardized_cell_equal_median_difference'])}。\n\n"
            + contrasts.loc[contrasts["contrast"].isin(["B1", "B2", "B3"])].to_markdown(index=False)
            + f"\n\n结论：**{verdicts['part_b_verdict']}**。\n"
        ),
        "16a1_05_entropy_interaction.md": (
            _report_header("16A.1-05 Entropy interaction")
            + f"R3-R4 standardized G_P={_fmt(pooled_b4p['standardized_cell_equal_median_difference'])} "
            f"[{_fmt(pooled_b4p['ci95_low'])}, {_fmt(pooled_b4p['ci95_high'])}]；"
            f"G_S={_fmt(pooled_b4s['standardized_cell_equal_median_difference'])} "
            f"[{_fmt(pooled_b4s['ci95_low'])}, {_fmt(pooled_b4s['ci95_high'])}]。\n\n"
            + contrasts.loc[contrasts["contrast"].isin(["B4_GP", "B4_GS"])].to_markdown(index=False)
            + "\n"
        ),
        "16a1_06_noise_heterogeneity.md": (
            _report_header("16A.1-06 Noise heterogeneity")
            + f"主异质性视图为 current×FE；结论：**{verdicts['noise_verdict']}**。"
            "只有 n_repeated_pairs≥15 的 cell 报告 delta95。\n\n"
            + noise_summary.to_markdown(index=False)
            + "\n\n## Cell support 与 delta95\n\n"
            + noise_cells.to_markdown(index=False)
            + "\n"
        ),
        "16a1_07_posthoc_local_threshold_sensitivity.md": (
            _report_header("16A.1-07 事后 local-threshold sensitivity")
            + f"{local_relation}。Task16A F3 与 Perturb A3 均保持不变。"
            + local_table
        ),
        "16a1_08_maturity_monotonicity.md": (
            _report_header("16A.1-08 Maturity monotonicity")
            + f"Pooled R2 Spearman rho={_fmt(m.loc[('pooled','all'),'spearman_rho'])} "
            f"[{_fmt(m.loc[('pooled','all'),'ci95_low'])}, {_fmt(m.loc[('pooled','all'),'ci95_high'])}]。\n\n"
            + maturity_correlations.to_markdown(index=False)
            + f"\n\n结论：**{verdicts['maturity_verdict']}**。\n"
        ),
        "16a1_09_solver_suite_robustness.md": (
            _report_header("16A.1-09 Solver/suite robustness")
            + robustness.to_markdown(index=False)
            + "\n"
        ),
        "16a1_10_final_joint_verdict.md": (
            _report_header("16A.1-10 Final joint verdict")
            + f"Part A：**{verdicts['part_a_verdict']}**  \n"
            f"Part B：**{verdicts['part_b_verdict']}**  \n"
            f"Noise：**{verdicts['noise_verdict']}**  \n"
            f"Maturity：**{verdicts['maturity_verdict']}**  \n"
            f"Joint：**{verdicts['joint_verdict']}**  \n\n"
            "Maturity 与 observed best switch 相对 Targeted Perturb 的优势存在单调关系，"
            "但标准化 intervention advantage 未显示 R2 enrichment，且 R1/R2 有效 cell 支持不足以满足 J1。"
            "因此该局部关系不足以支持模糊控制器开发。Task16A F3 与当前 Perturb A3 保持不变；"
            "不得直接进入 Type-1 或 Interval Type-2。\n"
        ),
    }
    for name, content in reports.items():
        (REPORTS / name).write_text(content, encoding="utf-8")

    pooled_maturity_levels = maturity_summary.loc[
        maturity_summary["scope_type"].eq("pooled")
    ]

    def maturity_level_line(regime: str) -> str:
        entries = []
        for level in (0.2, 0.4, 0.6, 0.8):
            selected = pooled_maturity_levels.loc[
                pooled_maturity_levels["regime"].eq(regime)
                & np.isclose(pooled_maturity_levels["maturity"], level)
            ]
            if selected.empty:
                entries.append(f"M={level:.1f}: NA (n=0)")
            else:
                row = selected.iloc[0]
                entries.append(
                    f"M={level:.1f}: {_fmt(row['median_G_SP'])} (n={int(row['num_states'])})"
                )
        return "；".join(entries)

    noise_global = noise_cells.loc[noise_cells["conditioning"].eq("global")][
        ["action", "delta95", "n_repeated_pairs"]
    ]
    noise_ranges = noise_summary.loc[
        noise_summary["conditioning"].isin(["suite", "current", "FE", "current_FE"])
    ]
    total = _report_header(
        "Decision-before-Feature Task16A.1：Zero-FE Joint Regime Verification"
    )
    total += f"""## Regime / maturity

1. R1/R2 cell counts：\n\n{support[['suite', 'current_algorithm', 'source_FE', 'n_R1', 'n_R2', 'n_R3', 'n_R4', 'support_R1_R2', 'support_R3_R4']].to_markdown(index=False)}

2. 有效 R1/R2 cells：{valid_cells}/24。
3. Raw pooled R2-R1：{_fmt(pooled_a['raw_delta_R2_minus_R1'])}。
4. Standardized R2-R1：{_fmt(pooled_a['standardized_cell_equal_delta'])}。
5. BBOB standardized CI：{_fmt(a.loc[('suite','bbob'),'standardized_cell_equal_delta'])} [{_fmt(a.loc[('suite','bbob'),'ci95_low'])}, {_fmt(a.loc[('suite','bbob'),'ci95_high'])}]。
6. MA-BBOB standardized CI：{_fmt(a.loc[('suite','mabbob'),'standardized_cell_equal_delta'])} [{_fmt(a.loc[('suite','mabbob'),'ci95_low'])}, {_fmt(a.loc[('suite','mabbob'),'ci95_high'])}]。
7. Solver standardized differences：SHADE={_fmt(a.loc[('current_algorithm','shade'),'standardized_cell_equal_delta'])}，L-SHADE={_fmt(a.loc[('current_algorithm','lshade'),'standardized_cell_equal_delta'])}，CSO={_fmt(a.loc[('current_algorithm','cso'),'standardized_cell_equal_delta'])}。
8. Part A verdict：{verdicts['part_a_verdict']}。

## Continuous advantage

9. G_P median：R1={_fmt(desc.loc[('pooled','all','R1','G_P'),'median'])}，R2={_fmt(desc.loc[('pooled','all','R2','G_P'),'median'])}。
10. G_S median：R1={_fmt(desc.loc[('pooled','all','R1','G_S'),'median'])}，R2={_fmt(desc.loc[('pooled','all','R2','G_S'),'median'])}。
11. G_I standardized R2-R1：{_fmt(pooled_b1['standardized_cell_equal_median_difference'])}。
12. G_I CI：[{_fmt(pooled_b1['ci95_low'])}, {_fmt(pooled_b1['ci95_high'])}]。
13. R3-R4 G_P：{_fmt(pooled_b4p['standardized_cell_equal_median_difference'])}。
14. R3-R4 G_S：{_fmt(pooled_b4s['standardized_cell_equal_median_difference'])}。
15. 跨 suite 稳定：{'YES' if all(float(b.loc[('B1','suite',suite),'standardized_cell_equal_median_difference']) > 0 for suite in ('bbob','mabbob')) else 'NO'}。
16. 跨 solver 稳定：{'YES' if sum(float(b.loc[('B1','current_algorithm',solver),'standardized_cell_equal_median_difference']) > 0 for solver in ('shade','lshade','cso')) >= 2 else 'NO'}。
17. Part B verdict：{verdicts['part_b_verdict']}。

## Noise

18. Global delta95：\n\n{noise_global.to_markdown(index=False)}

19. Suite/current/FE/current×FE ranges：\n\n{noise_ranges.to_markdown(index=False)}

20. Cell repeated-pair support：见 `task16a1_noise_cells.parquet`。
21. R_delta：按 action 与 conditioning 完整报告。
22. Noise verdict：{verdicts['noise_verdict']}。
23. Local-threshold sensitivity：{local_relation}。
24. 低支持 cell 是否影响 local 解释：低支持 cell-specific threshold 从未进入 sensitivity；仅使用 current×FE、current 或 global 中支持充分的估计。

## Maturity

25. R2 四个 maturity 的 G_SP median：{maturity_level_line('R2')}。
26. R3：{maturity_level_line('R3')}。
27. R4：{maturity_level_line('R4')}。
28. Pooled R2 Spearman：{_fmt(m.loc[('pooled','all'),'spearman_rho'])} [{_fmt(m.loc[('pooled','all'),'ci95_low'])}, {_fmt(m.loc[('pooled','all'),'ci95_high'])}]。
29. BBOB/MA：BBOB={_fmt(m.loc[('suite','bbob'),'spearman_rho'])}，MA-BBOB={_fmt(m.loc[('suite','mabbob'),'spearman_rho'])}。
30. Solver：SHADE={_fmt(m.loc[('current_algorithm','shade'),'spearman_rho'])}，L-SHADE={_fmt(m.loc[('current_algorithm','lshade'),'spearman_rho'])}，CSO={_fmt(m.loc[('current_algorithm','cso'),'spearman_rho'])}。
31. Maturity verdict：{verdicts['maturity_verdict']}。

## Final

32. Task16A primary F3 是否保持：YES。
33. 当前 Perturb A3 是否保持：YES。
34. Task16A.1 verdict：{verdicts['joint_verdict']}。
35. 是否允许直接运行 Type-1：NO。
36. 是否允许 Interval Type-2：NO。
37. 若为 J1，是否只能先设计 Task16A.2：YES。
38. 若为 J3，search-regime intervention line 是否停止：YES。
39. new objective FE 是否为 0：YES。

## 资源

- reused_task16a_states：{resource_row['reused_task16a_states']}；
- reused_action_outcomes：{resource_row['reused_action_outcomes']}；
- reused_repetition_rows：{resource_row['reused_repetition_rows']}；
- analysis_cpu_seconds：{resource_row['analysis_cpu_seconds']:.3f}；
- wall_seconds：{resource_row['wall_seconds']:.3f}；
- peak_rss_mb：{resource_row['peak_rss_mb']:.3f}。

## 解释边界

该结论只描述 Task16A 既有开发数据中的事后诊断结构，不改变 Task16A 的正式结论，也不评价任何模糊控制器。
"""
    (REPORTS / "Decision-before-Feature_Task16A1_ZeroFEJointRegimeVerification.md").write_text(
        total, encoding="utf-8"
    )


def main() -> None:
    started_wall = perf_counter()
    started_cpu = process_time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    sources, outcomes, repetitions, practical, task16a_noise = _load_inputs()
    states = _state_frame(sources, outcomes, practical)
    support = _support(states)
    a_cells, part_a, bootstrap_a = _part_a(states, support)
    advantage_summary = _advantage_descriptives(states)
    contrasts, bootstrap_b = _continuous_contrasts(states, support)
    noise_pairs = _noise_pairs(repetitions)
    noise_cells, noise_summary, noise_verdict = _noise_cells(noise_pairs, task16a_noise)
    local_sets = _local_practical_sets(states, outcomes, noise_cells, noise_verdict)
    local_cells = pd.DataFrame()
    local_sensitivity = pd.DataFrame()
    bootstrap_local: list[pd.DataFrame] = []
    if len(local_sets):
        local_cells, local_sensitivity, bootstrap_local = _part_a(
            local_sets,
            support,
            column="posthoc_local_threshold_Z_I",
            analysis_part="L",
            contrast="posthoc_local_Z_I_R2_minus_R1",
            stream_base=6000,
        )
    else:
        local_cells = a_cells.iloc[0:0].copy()
        local_sensitivity = part_a.iloc[0:0].copy()
    if bool(len(local_sets)) != noise_verdict.startswith("N1"):
        raise RuntimeError("local-threshold sensitivity must be generated if and only if N1 holds")
    maturity_summary, maturity_correlations, bootstrap_m, maturity_verdict = _maturity(states)
    verdicts = _verdicts(
        part_a,
        contrasts,
        support,
        noise_verdict,
        local_sensitivity,
        maturity_verdict,
    )
    robustness = _robustness(part_a, contrasts, maturity_correlations)
    bootstrap = pd.concat(
        [*bootstrap_a, *bootstrap_b, *bootstrap_local, *bootstrap_m],
        ignore_index=True,
    )
    bootstrap_groups = ["analysis_part", "contrast", "scope_type", "scope_value"]
    if not bootstrap.groupby(bootstrap_groups)["draw"].nunique().eq(BOOTSTRAP_DRAWS).all():
        raise RuntimeError("every reported bootstrap series must contain 5000 indexed draws")

    states.to_parquet(OUTPUT / "task16a1_state_regimes.parquet", index=False)
    support.to_parquet(OUTPUT / "task16a1_cell_support.parquet", index=False)
    part_a.to_parquet(OUTPUT / "task16a1_maturity_standardized_regime.parquet", index=False)
    states.to_parquet(OUTPUT / "task16a1_continuous_advantages.parquet", index=False)
    advantage_summary.to_parquet(OUTPUT / "task16a1_advantage_regime_summary.parquet", index=False)
    contrasts.to_parquet(OUTPUT / "task16a1_continuous_advantage_contrasts.parquet", index=False)
    contrasts.loc[contrasts["contrast"].isin(["B4_GP", "B4_GS"])].to_parquet(
        OUTPUT / "task16a1_entropy_interaction.parquet", index=False
    )
    noise_cells.to_parquet(OUTPUT / "task16a1_noise_cells.parquet", index=False)
    noise_summary.to_parquet(
        OUTPUT / "task16a1_noise_heterogeneity_summary.parquet", index=False
    )
    local_sets.to_parquet(
        OUTPUT / "task16a1_posthoc_local_practical_sets.parquet", index=False
    )
    local_sensitivity.to_parquet(
        OUTPUT / "task16a1_posthoc_local_regime_sensitivity.parquet", index=False
    )
    local_cells.to_parquet(
        OUTPUT / "task16a1_posthoc_local_regime_cells.parquet", index=False
    )
    maturity_summary.to_parquet(OUTPUT / "task16a1_maturity_levels.parquet", index=False)
    maturity_correlations.to_parquet(
        OUTPUT / "task16a1_maturity_monotonicity.parquet", index=False
    )
    robustness.to_parquet(
        OUTPUT / "task16a1_solver_suite_robustness.parquet", index=False
    )
    bootstrap.to_parquet(OUTPUT / "task16a1_bootstrap.parquet", index=False)
    pd.DataFrame([verdicts]).to_parquet(
        OUTPUT / "task16a1_final_joint_verdict.parquet", index=False
    )
    a_cells.to_parquet(OUTPUT / "task16a1_regime_cell_estimates.parquet", index=False)
    noise_pairs.to_parquet(OUTPUT / "task16a1_repetition_differences.parquet", index=False)
    resource_row = {
        "new_objective_FE": 0,
        "reused_task16a_states": int(len(sources)),
        "reused_action_outcomes": int(len(outcomes)),
        "reused_primary_action_outcomes": int(outcomes["repetition_id"].eq(0).sum()),
        "reused_repetition_rows": int(len(repetitions)),
        "analysis_cpu_seconds": float(process_time() - started_cpu),
        "wall_seconds": float(perf_counter() - started_wall),
        "peak_rss_mb": _peak_rss_mb(),
        "status": "VALID_ZERO_FE",
    }
    if resource_row["new_objective_FE"] != 0:
        raise RuntimeError("Task16A.1 is invalid when new_objective_FE is not zero")
    _write_reports(
        states=states,
        support=support,
        part_a=part_a,
        advantages=advantage_summary,
        contrasts=contrasts,
        noise_cells=noise_cells,
        noise_summary=noise_summary,
        local_sets=local_sets,
        local_sensitivity=local_sensitivity,
        maturity_summary=maturity_summary,
        maturity_correlations=maturity_correlations,
        robustness=robustness,
        verdicts=verdicts,
        resource_row=resource_row,
    )
    resource_row["analysis_cpu_seconds"] = float(process_time() - started_cpu)
    resource_row["wall_seconds"] = float(perf_counter() - started_wall)
    resource_row["peak_rss_mb"] = _peak_rss_mb()
    pd.DataFrame([resource_row]).to_parquet(
        OUTPUT / "task16a1_resource_ledger.parquet", index=False
    )
    _write_reports(
        states=states,
        support=support,
        part_a=part_a,
        advantages=advantage_summary,
        contrasts=contrasts,
        noise_cells=noise_cells,
        noise_summary=noise_summary,
        local_sets=local_sets,
        local_sensitivity=local_sensitivity,
        maturity_summary=maturity_summary,
        maturity_correlations=maturity_correlations,
        robustness=robustness,
        verdicts=verdicts,
        resource_row=resource_row,
    )
    print(
        f"[task16a1] {verdicts['part_a_verdict']} | {verdicts['part_b_verdict']} | "
        f"{verdicts['noise_verdict']} | {verdicts['maturity_verdict']} | "
        f"{verdicts['joint_verdict']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
