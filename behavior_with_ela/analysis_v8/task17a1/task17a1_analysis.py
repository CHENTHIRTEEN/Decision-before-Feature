"""Task 17A.1 zero-FE statistical consistency re-analysis.

The module reads only Task 17A state, pair, representation, decision-distance,
and nearest-neighbor products. It does not import or call benchmark objectives,
optimizers, feature extractors, ELA code, or learned models.
"""
from __future__ import annotations

import json
import resource
import sys
import warnings
from pathlib import Path
from time import perf_counter, process_time
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr


ROOT = Path(__file__).resolve().parents[3]
OLD_LIGHT = ROOT / "behavior_with_ela/analysis_v8/task17a"
OLD_HEAVY = ROOT / "behavior_with_ela/results/analysis_v8/task17a"
OUT_LIGHT = ROOT / "behavior_with_ela/analysis_v8/task17a1"
OUT_HEAVY = ROOT / "behavior_with_ela/results/analysis_v8/task17a1"
FIGURES = OUT_LIGHT / "figures"

BOOTSTRAP_DRAWS = 5000
PERMUTATIONS = 100
RANDOM_NEIGHBOR_CONTROLS = 100
MASTER_SEED = 2026083101
BOOTSTRAP_STREAM = 2026083111
PERMUTATION_STREAM = 2026083112
CROSS_DOMAIN_STREAM = 2026083113
LADDER_STREAM = 2026083114
NEW_OBJECTIVE_FE = 0

DOMAINS = ("natural", "post_handoff")
SCOPES = ("bbob", "mabbob", "pooled")
BASE_REPRESENTATIONS = ("compact6", "global28")
POST_LADDER = (
    "compact6",
    "global28",
    "segment_matched28",
    "issd18",
    "compact_issd24",
)
MARGIN_COLUMNS = (
    "normalized_margin_shade_lshade",
    "normalized_margin_shade_cso",
    "normalized_margin_lshade_cso",
)
DOMAIN_CODES = {"natural": 1, "post_handoff": 2}
SCOPE_CODES = {"bbob": 1, "mabbob": 2, "pooled": 3}
REPRESENTATION_CODES = {
    "compact6": 1,
    "global28": 2,
    "segment_old28": 3,
    "segment_matched28": 4,
    "issd18": 5,
    "compact_issd24": 6,
}


def _seed_sequence(*codes: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([MASTER_SEED, *[int(code) for code in codes]])


def _scope(frame: pd.DataFrame, suite: str) -> pd.DataFrame:
    if suite == "pooled":
        return frame
    return frame.loc[frame["suite"].eq(suite)]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        return float(spearmanr(x, y).statistic)


def _within_rank_alignment(frame: pd.DataFrame) -> float:
    """The single statistic used by point, bootstrap, and permutation paths."""
    return _spearman(
        frame["representation_rank_within_stratum"].to_numpy(dtype=float),
        frame["decision_rank_within_stratum"].to_numpy(dtype=float),
    )


def _nn_delta(frame: pd.DataFrame) -> float:
    return float(
        np.median(frame["decision_distance_random_median"])
        - np.median(frame["decision_distance_observable_nn"])
    )


def _ava_rate(frame: pd.DataFrame) -> float:
    return float(np.mean(frame["decision_distance_linf"].to_numpy(dtype=float) > 1.0))


def _median_decision_distance(frame: pd.DataFrame) -> float:
    return float(np.median(frame["decision_distance_linf"].to_numpy(dtype=float)))


def _quantile_interval(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def _group_codes(frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    groups = sorted(frame["cv_group_id"].astype(str).unique())
    values = frame["cv_group_id"].astype(str).to_numpy()
    group_to_code = {group: index for index, group in enumerate(groups)}
    codes = np.asarray([group_to_code[value] for value in values], dtype=int)
    if len(codes) != len(frame):
        raise RuntimeError("empty cv_group_id block")
    return groups, codes


def _bootstrap_multiplicities(
    n_groups: int,
    stream_codes: tuple[int, ...],
    stream: int = BOOTSTRAP_STREAM,
) -> np.ndarray:
    rng = np.random.default_rng(
        _seed_sequence(stream, *stream_codes, n_groups)
    )
    return rng.multinomial(
        n_groups,
        np.full(n_groups, 1.0 / n_groups),
        size=BOOTSTRAP_DRAWS,
    )


def _weighted_median_draws(
    values: np.ndarray,
    group_codes: np.ndarray,
    multiplicities: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_codes = np.asarray(group_codes, dtype=int)[order]
    output = np.empty(len(multiplicities), dtype=float)
    batch_size = 32 if len(values) > 20000 else 128
    for start in range(0, len(multiplicities), batch_size):
        stop = min(start + batch_size, len(multiplicities))
        weights = multiplicities[start:stop, sorted_codes]
        cumulative = np.cumsum(weights, axis=1)
        totals = cumulative[:, -1]
        lower_positions = (totals - 1) // 2
        upper_positions = totals // 2
        lower_indices = np.argmax(cumulative > lower_positions[:, None], axis=1)
        upper_indices = np.argmax(cumulative > upper_positions[:, None], axis=1)
        output[start:stop] = (
            sorted_values[lower_indices] + sorted_values[upper_indices]
        ) / 2.0
    return output


def _weighted_spearman_draws(
    frame: pd.DataFrame,
    group_codes: np.ndarray,
    multiplicities: np.ndarray,
) -> np.ndarray:
    x = frame["representation_rank_within_stratum"].to_numpy(dtype=float)
    y = frame["decision_rank_within_stratum"].to_numpy(dtype=float)
    unique_x, x_codes = np.unique(x, return_inverse=True)
    unique_y, y_codes = np.unique(y, return_inverse=True)
    contingency_by_group = np.zeros(
        (multiplicities.shape[1], len(unique_x), len(unique_y)), dtype=float
    )
    np.add.at(contingency_by_group, (group_codes, x_codes, y_codes), 1.0)
    output = np.empty(len(multiplicities), dtype=float)
    for draw, group_multiplicity in enumerate(multiplicities):
        contingency = np.tensordot(
            group_multiplicity,
            contingency_by_group,
            axes=(0, 0),
        )
        x_counts = contingency.sum(axis=1)
        y_counts = contingency.sum(axis=0)
        total = float(x_counts.sum())
        x_midrank = np.cumsum(x_counts) - (x_counts - 1.0) / 2.0
        y_midrank = np.cumsum(y_counts) - (y_counts - 1.0) / 2.0
        mean_rank = (total + 1.0) / 2.0
        x_centered = x_midrank - mean_rank
        y_centered = y_midrank - mean_rank
        covariance = float(x_centered @ contingency @ y_centered)
        variance_x = float(np.sum(x_counts * x_centered**2))
        variance_y = float(np.sum(y_counts * y_centered**2))
        output[draw] = (
            covariance / np.sqrt(variance_x * variance_y)
            if variance_x > 0 and variance_y > 0
            else np.nan
        )
    return output


def _group_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    stream_codes: tuple[int, ...],
) -> tuple[float, np.ndarray]:
    groups, codes = _group_codes(frame)
    multiplicities = _bootstrap_multiplicities(len(groups), stream_codes)
    positions = [np.flatnonzero(codes == index) for index in range(len(groups))]
    first_indices = np.concatenate(
        [
            np.tile(positions[index], int(count))
            for index, count in enumerate(multiplicities[0])
            if count > 0
        ]
    )
    if statistic is _within_rank_alignment:
        draws = _weighted_spearman_draws(frame, codes, multiplicities)
    elif statistic is _ava_rate:
        successes = np.bincount(
            codes,
            weights=(frame["decision_distance_linf"].to_numpy(dtype=float) > 1.0),
            minlength=len(groups),
        )
        totals = np.bincount(codes, minlength=len(groups))
        draws = (multiplicities @ successes) / (multiplicities @ totals)
    elif statistic is _median_decision_distance:
        draws = _weighted_median_draws(
            frame["decision_distance_linf"].to_numpy(dtype=float),
            codes,
            multiplicities,
        )
    elif statistic is _nn_delta:
        random_median = _weighted_median_draws(
            frame["decision_distance_random_median"].to_numpy(dtype=float),
            codes,
            multiplicities,
        )
        observable_median = _weighted_median_draws(
            frame["decision_distance_observable_nn"].to_numpy(dtype=float),
            codes,
            multiplicities,
        )
        draws = random_median - observable_median
    else:
        draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
        for draw, multiplicity in enumerate(multiplicities):
            indices = np.concatenate(
                [
                    np.tile(positions[index], int(count))
                    for index, count in enumerate(multiplicity)
                    if count > 0
                ]
            )
            draws[draw] = float(statistic(frame.iloc[indices]))
    first_direct = float(statistic(frame.iloc[first_indices]))
    if not np.isclose(draws[0], first_direct, rtol=1e-12, atol=1e-12, equal_nan=True):
        raise RuntimeError("grouped bootstrap optimization changed the requested statistic")
    return float(statistic(frame)), draws


def _group_mean_sensitivity(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    stream_codes: tuple[int, ...],
) -> tuple[float, float, float, int]:
    values = []
    for _, part in frame.groupby("cv_group_id", sort=True):
        value = float(statistic(part))
        if np.isfinite(value):
            values.append(value)
    array = np.asarray(values, dtype=float)
    if not len(array):
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(
        _seed_sequence(BOOTSTRAP_STREAM, *stream_codes, len(array))
    )
    draws = rng.choice(array, size=(BOOTSTRAP_DRAWS, len(array)), replace=True).mean(axis=1)
    low, high = _quantile_interval(draws)
    return float(np.mean(array)), low, high, int(len(array))


def _matched_group_difference(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    stream_codes: tuple[int, ...],
) -> tuple[float, np.ndarray, int]:
    reference_groups = set(reference["cv_group_id"].astype(str))
    comparison_groups = set(comparison["cv_group_id"].astype(str))
    groups = sorted(reference_groups & comparison_groups)
    if not groups:
        return float("nan"), np.full(BOOTSTRAP_DRAWS, np.nan), 0
    reference = reference.loc[reference["cv_group_id"].astype(str).isin(groups)].copy()
    comparison = comparison.loc[comparison["cv_group_id"].astype(str).isin(groups)].copy()
    group_to_code = {group: index for index, group in enumerate(groups)}
    reference_codes = np.asarray(
        [group_to_code[value] for value in reference["cv_group_id"].astype(str)],
        dtype=int,
    )
    comparison_codes = np.asarray(
        [group_to_code[value] for value in comparison["cv_group_id"].astype(str)],
        dtype=int,
    )
    reference_point = float(statistic(reference))
    comparison_point = float(statistic(comparison))
    multiplicities = _bootstrap_multiplicities(
        len(groups),
        stream_codes,
        stream=CROSS_DOMAIN_STREAM,
    )
    if statistic is _within_rank_alignment:
        reference_draws = _weighted_spearman_draws(
            reference,
            reference_codes,
            multiplicities,
        )
        comparison_draws = _weighted_spearman_draws(
            comparison,
            comparison_codes,
            multiplicities,
        )
    elif statistic is _ava_rate:
        reference_successes = np.bincount(
            reference_codes,
            weights=(reference["decision_distance_linf"].to_numpy(dtype=float) > 1.0),
            minlength=len(groups),
        )
        comparison_successes = np.bincount(
            comparison_codes,
            weights=(comparison["decision_distance_linf"].to_numpy(dtype=float) > 1.0),
            minlength=len(groups),
        )
        reference_totals = np.bincount(reference_codes, minlength=len(groups))
        comparison_totals = np.bincount(comparison_codes, minlength=len(groups))
        reference_draws = (multiplicities @ reference_successes) / (
            multiplicities @ reference_totals
        )
        comparison_draws = (multiplicities @ comparison_successes) / (
            multiplicities @ comparison_totals
        )
    elif statistic is _nn_delta:
        reference_draws = _weighted_median_draws(
            reference["decision_distance_random_median"].to_numpy(dtype=float),
            reference_codes,
            multiplicities,
        ) - _weighted_median_draws(
            reference["decision_distance_observable_nn"].to_numpy(dtype=float),
            reference_codes,
            multiplicities,
        )
        comparison_draws = _weighted_median_draws(
            comparison["decision_distance_random_median"].to_numpy(dtype=float),
            comparison_codes,
            multiplicities,
        ) - _weighted_median_draws(
            comparison["decision_distance_observable_nn"].to_numpy(dtype=float),
            comparison_codes,
            multiplicities,
        )
    else:
        reference_positions = [
            np.flatnonzero(reference_codes == index) for index in range(len(groups))
        ]
        comparison_positions = [
            np.flatnonzero(comparison_codes == index) for index in range(len(groups))
        ]
        reference_draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
        comparison_draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
        for draw, multiplicity in enumerate(multiplicities):
            reference_index = np.concatenate(
                [
                    np.tile(reference_positions[index], int(count))
                    for index, count in enumerate(multiplicity)
                    if count > 0
                ]
            )
            comparison_index = np.concatenate(
                [
                    np.tile(comparison_positions[index], int(count))
                    for index, count in enumerate(multiplicity)
                    if count > 0
                ]
            )
            reference_draws[draw] = statistic(reference.iloc[reference_index])
            comparison_draws[draw] = statistic(comparison.iloc[comparison_index])
    draws = comparison_draws - reference_draws
    return comparison_point - reference_point, draws, len(groups)


def _read_inputs() -> dict[str, pd.DataFrame]:
    names = {
        "states": "task17a_state_decision_signatures.parquet",
        "pairs": "task17a_within_stratum_pairs.parquet",
        "decision": "task17a_decision_distances.parquet",
        "representation": "task17a_representation_distances.parquet",
        "nn": "task17a_nn_consistency.parquet",
        "random_pairs": "task17a_random_pair_controls.parquet",
        "old_alignment": "task17a_alignment_summary.parquet",
        "old_aliasing": "task17a_aliasing_rates.parquet",
        "old_shift": "task17a_natural_post_shift.parquet",
        "old_ladder": "task17a_representation_ladder.parquet",
        "old_heterogeneity": "task17a_heterogeneity_summary.parquet",
    }
    missing = [OLD_HEAVY / filename for filename in names.values() if not (OLD_HEAVY / filename).exists()]
    if missing:
        raise FileNotFoundError("missing Task 17A products: " + ", ".join(map(str, missing)))
    return {name: pd.read_parquet(OLD_HEAVY / filename) for name, filename in names.items()}


def _validate_and_merge(
    inputs: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    states = inputs["states"].copy()
    pairs = inputs["pairs"].copy()
    decision = inputs["decision"].copy()
    representation = inputs["representation"].copy()

    expected_states = {"natural": 1890, "post_handoff": 3780}
    expected_pairs = {"natural": 3780, "post_handoff": 7560}
    if states.groupby("domain").size().to_dict() != expected_states:
        raise RuntimeError("state counts do not match the Task 17A contract")
    if pairs.groupby("domain").size().to_dict() != expected_pairs:
        raise RuntimeError("pair counts do not match the Task 17A contract")
    if not states["state_id"].is_unique:
        raise RuntimeError("state_id is not unique")

    stratum_sizes = states.groupby(["domain", "stratum_id"], sort=False).size()
    seed_sets = states.groupby(["domain", "stratum_id"], sort=False)["seed"].agg(
        lambda values: tuple(sorted(map(int, values)))
    )
    if not stratum_sizes.eq(5).all():
        raise RuntimeError("a state stratum does not contain exactly five states")
    if not seed_sets.map(lambda values: values == (1, 2, 3, 4, 5)).all():
        raise RuntimeError("a state stratum does not contain exactly seeds 1-5")

    pair_key = ["domain", "stratum_id", "seed_i", "seed_j"]
    representation_key = ["domain", "representation", "stratum_id", "seed_i", "seed_j"]
    if pairs.duplicated(pair_key).any() or decision.duplicated(pair_key).any():
        raise RuntimeError("state-pair identity is not unique")
    if representation.duplicated(representation_key).any():
        raise RuntimeError("representation-pair identity is not unique")
    if not (pairs["seed_i"].astype(int) < pairs["seed_j"].astype(int)).all():
        raise RuntimeError("pair seed order is not canonical")
    if not pairs.groupby(["domain", "stratum_id"], sort=False).size().eq(10).all():
        raise RuntimeError("a stratum does not contain exactly ten unordered pairs")

    state_domain = states.set_index("state_id")["domain"].astype(str)
    domain_i = pairs["state_i"].map(state_domain)
    domain_j = pairs["state_j"].map(state_domain)
    if domain_i.isna().any() or domain_j.isna().any():
        raise RuntimeError("pair-to-state join is incomplete")
    if not domain_i.eq(pairs["domain"]).all() or not domain_j.eq(pairs["domain"]).all():
        raise RuntimeError("a pair crosses domains")

    identity_columns = [
        "domain",
        "suite",
        "pair_index",
        "stratum_id",
        "cv_group_id",
        "problem_id",
        "state_i",
        "state_j",
        "seed_i",
        "seed_j",
        "state_position_i",
        "state_position_j",
    ]
    pair_identity = pairs[identity_columns].merge(
        decision[
            [
                *identity_columns,
                "decision_distance_linf",
                "decision_distance_l2",
            ]
        ],
        on=identity_columns,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if len(pair_identity) != 11340 or not pair_identity["_merge"].eq("both").all():
        raise RuntimeError("decision-distance join is not exact")
    pair_identity = pair_identity.drop(columns="_merge")

    merged = representation.merge(
        decision[
            [
                "domain",
                "pair_index",
                "decision_distance_linf",
                "decision_distance_l2",
                "action_set_i",
                "action_set_j",
            ]
        ],
        on=["domain", "pair_index"],
        how="left",
        validate="many_to_one",
    )
    if merged[["decision_distance_linf", "decision_distance_l2"]].isna().any().any():
        raise RuntimeError("representation-to-decision join is incomplete")
    expected_representation_rows = 52920
    if len(merged) != expected_representation_rows:
        raise RuntimeError(
            f"expected {expected_representation_rows} representation rows, got {len(merged)}"
        )

    state_identity = states[
        [
            "domain",
            "suite",
            "state_id",
            "stratum_id",
            "cv_group_id",
            "problem_id",
            "seed",
            "route",
            "current_algorithm",
            "source_FE",
        ]
    ].copy()
    state_identity["state_identity_unique"] = True
    state_identity["new_objective_fe"] = NEW_OBJECTIVE_FE
    pair_identity["pair_identity_unique"] = True
    pair_identity["cross_domain"] = False
    pair_identity["new_objective_fe"] = NEW_OBJECTIVE_FE
    return states, pair_identity, merged, state_identity


def _rank_pair_distances(merged: pd.DataFrame) -> pd.DataFrame:
    ranked = merged.copy()
    group_columns = ["domain", "representation", "stratum_id"]
    sizes = ranked.groupby(group_columns, sort=False)["pair_index"].transform("size")
    if not sizes.eq(10).all():
        raise RuntimeError("rank normalization requires exactly ten pairs per stratum")
    representation_rank = ranked.groupby(group_columns, sort=False)[
        "representation_distance_l1_mean"
    ].rank(method="average")
    decision_rank = ranked.groupby(group_columns, sort=False)[
        "decision_distance_linf"
    ].rank(method="average")
    ranked["representation_rank_within_stratum"] = (representation_rank - 1.0) / 9.0
    ranked["decision_rank_within_stratum"] = (decision_rank - 1.0) / 9.0
    ranked["rank_tie_method"] = "average"
    ranked["rank_normalization"] = "(rank-1)/(n_pairs-1)"
    ranked["n_pairs_in_stratum"] = sizes.astype(int)
    ranked["new_objective_fe"] = NEW_OBJECTIVE_FE
    return ranked


def _permuted_decision_distance(
    states: pd.DataFrame,
    pairs: pd.DataFrame,
    domain: str,
    repeat: int,
) -> pd.DataFrame:
    domain_states = states.loc[states["domain"].eq(domain)].reset_index(drop=True)
    domain_pairs = pairs.loc[pairs["domain"].eq(domain)].sort_values("pair_index").reset_index(drop=True)
    margins = domain_states[list(MARGIN_COLUMNS)].to_numpy(dtype=float)
    permuted = margins.copy()
    rng = np.random.default_rng(
        _seed_sequence(PERMUTATION_STREAM, DOMAIN_CODES[domain], repeat)
    )
    for _, part in domain_states.groupby("stratum_id", sort=True):
        positions = part.index.to_numpy(dtype=int)
        permuted[positions] = margins[positions[rng.permutation(len(positions))]]
    position_i = domain_pairs["state_position_i"].to_numpy(dtype=int)
    position_j = domain_pairs["state_position_j"].to_numpy(dtype=int)
    values = np.max(np.abs(permuted[position_i] - permuted[position_j]), axis=1)
    output = domain_pairs[
        ["domain", "suite", "pair_index", "stratum_id", "cv_group_id"]
    ].copy()
    output["decision_distance_linf"] = values
    ranks = output.groupby("stratum_id", sort=False)["decision_distance_linf"].rank(
        method="average"
    )
    output["decision_rank_within_stratum"] = (ranks - 1.0) / 9.0
    return output


def _run_permutations(
    states: pd.DataFrame,
    pair_identity: pd.DataFrame,
    ranked: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        domain_ranked = ranked.loc[ranked["domain"].eq(domain)]
        for repeat in range(PERMUTATIONS):
            permuted = _permuted_decision_distance(
                states,
                pair_identity,
                domain,
                repeat,
            )
            for representation in sorted(domain_ranked["representation"].unique()):
                observable = domain_ranked.loc[
                    domain_ranked["representation"].eq(representation),
                    [
                        "domain",
                        "suite",
                        "pair_index",
                        "stratum_id",
                        "cv_group_id",
                        "representation_rank_within_stratum",
                    ],
                ]
                combined = observable.merge(
                    permuted[
                        [
                            "domain",
                            "pair_index",
                            "decision_rank_within_stratum",
                        ]
                    ],
                    on=["domain", "pair_index"],
                    validate="one_to_one",
                )
                for suite in SCOPES:
                    subset = _scope(combined, suite)
                    rows.append(
                        {
                            "domain": domain,
                            "suite": suite,
                            "representation": representation,
                            "repeat": repeat,
                            "rho_rank_within_null": _within_rank_alignment(subset),
                            "statistic_function": "_within_rank_alignment",
                            "permutation_unit": "stratum_id",
                            "signature_permutation": "within_stratum_state_permutation",
                            "new_objective_fe": NEW_OBJECTIVE_FE,
                        }
                    )
    return pd.DataFrame(rows)


def _corrected_alignment(
    ranked: pd.DataFrame,
    permutations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    bootstrap_parts = []
    for (domain, representation), part in ranked.groupby(
        ["domain", "representation"], sort=True
    ):
        for suite in SCOPES:
            subset = _scope(part, suite)
            point, draws = _group_bootstrap(
                subset,
                _within_rank_alignment,
                (
                    DOMAIN_CODES[domain],
                    REPRESENTATION_CODES[representation],
                    SCOPE_CODES[suite],
                    1,
                ),
            )
            low, high = _quantile_interval(draws)
            group_mean, group_low, group_high, n_groups = _group_mean_sensitivity(
                subset,
                _within_rank_alignment,
                (
                    DOMAIN_CODES[domain],
                    REPRESENTATION_CODES[representation],
                    SCOPE_CODES[suite],
                    2,
                ),
            )
            null = permutations.loc[
                permutations["domain"].eq(domain)
                & permutations["representation"].eq(representation)
                & permutations["suite"].eq(suite),
                "rho_rank_within_null",
            ].to_numpy(dtype=float)
            null_mean = float(np.mean(null))
            q95 = float(np.quantile(null, 0.95))
            q975 = float(np.quantile(null, 0.975))
            empirical_p = float((1 + np.sum(null >= point)) / (1 + len(null)))
            if point > 0 and low > 0 and point > q975 and empirical_p <= 0.05:
                verdict = "CA1 ROBUST ALIGNMENT"
            elif point > 0:
                verdict = "CA2 WEAK ALIGNMENT"
            else:
                verdict = "CA3 NO ALIGNMENT"
            summary_rows.append(
                {
                    "domain": domain,
                    "suite": suite,
                    "representation": representation,
                    "n_pairs": len(subset),
                    "n_cv_groups": n_groups,
                    "rho_rank_within": point,
                    "rho_rank_within_ci_low": low,
                    "rho_rank_within_ci_high": high,
                    "rho_group_mean_sensitivity": group_mean,
                    "rho_group_mean_sensitivity_ci_low": group_low,
                    "rho_group_mean_sensitivity_ci_high": group_high,
                    "permutation_null_mean": null_mean,
                    "permutation_q95": q95,
                    "permutation_q97_5": q975,
                    "empirical_p": empirical_p,
                    "alignment_verdict": verdict,
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "bootstrap_unit": "cv_group_id",
                    "permutations": PERMUTATIONS,
                    "statistic_function": "_within_rank_alignment",
                    "new_objective_fe": NEW_OBJECTIVE_FE,
                }
            )
            bootstrap_parts.append(
                pd.DataFrame(
                    {
                        "domain": domain,
                        "suite": suite,
                        "representation": representation,
                        "draw": np.arange(BOOTSTRAP_DRAWS, dtype=int),
                        "rho_rank_within": draws,
                        "bootstrap_unit": "cv_group_id",
                        "statistic_function": "_within_rank_alignment",
                        "new_objective_fe": NEW_OBJECTIVE_FE,
                    }
                )
            )
    return pd.DataFrame(summary_rows), pd.concat(bootstrap_parts, ignore_index=True)


def _corrected_nn(nn: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (domain, representation), part in nn.groupby(
        ["domain", "representation"], sort=True
    ):
        for suite in SCOPES:
            subset = _scope(part, suite)
            point, draws = _group_bootstrap(
                subset,
                _nn_delta,
                (
                    DOMAIN_CODES[domain],
                    REPRESENTATION_CODES[representation],
                    SCOPE_CODES[suite],
                    21,
                ),
            )
            low, high = _quantile_interval(draws)
            observable = float(np.median(subset["decision_distance_observable_nn"]))
            random = float(np.median(subset["decision_distance_random_median"]))
            rows.append(
                {
                    "domain": domain,
                    "suite": suite,
                    "representation": representation,
                    "n_states": len(subset),
                    "observable_nn_median_decision_distance": observable,
                    "random_nn_median_decision_distance": random,
                    "delta_nn": point,
                    "delta_nn_ci_low": low,
                    "delta_nn_ci_high": high,
                    "r_nn": 1.0 - observable / random if random > 0 else np.nan,
                    "random_neighbor_controls": RANDOM_NEIGHBOR_CONTROLS,
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "bootstrap_unit": "cv_group_id",
                    "new_objective_fe": NEW_OBJECTIVE_FE,
                }
            )
    return pd.DataFrame(rows)


def _local_pair_selection(ranked: pd.DataFrame) -> pd.DataFrame:
    ordered = ranked.sort_values(
        [
            "domain",
            "representation",
            "stratum_id",
            "representation_distance_l1_mean",
            "seed_i",
            "seed_j",
            "pair_index",
        ]
    ).copy()
    group_columns = ["domain", "representation", "stratum_id"]
    ordered["local_order"] = ordered.groupby(group_columns, sort=False).cumcount() + 1
    minimum = ordered.groupby(group_columns, sort=False)[
        "representation_distance_l1_mean"
    ].transform("min")
    ordered["nearest_distance_tie"] = ordered[
        "representation_distance_l1_mean"
    ].eq(minimum)
    ordered["nearest_distance_tie_count"] = ordered.groupby(
        group_columns, sort=False
    )["nearest_distance_tie"].transform("sum").astype(int)
    selected = ordered.loc[ordered["local_order"].le(2)].copy()
    selected["selected_for_local_ava10"] = selected["local_order"].eq(1)
    selected["selected_for_local_ava20"] = True
    selected["tie_break"] = "seed_i_then_seed_j_then_pair_index"
    selected["new_objective_fe"] = NEW_OBJECTIVE_FE
    counts = selected.groupby(group_columns, sort=False)["local_order"].agg(
        local10=lambda values: int(np.sum(values == 1)),
        local20="size",
    )
    if not counts["local10"].eq(1).all():
        raise RuntimeError("local AVA10 must select exactly one pair per stratum")
    if not counts["local20"].eq(2).all():
        raise RuntimeError("local AVA20 must select exactly two pairs per stratum")
    return selected


def _matched_local_random_reduction(
    local: pd.DataFrame,
    random_pairs: pd.DataFrame,
    stream_codes: tuple[int, ...],
) -> tuple[float, float, float]:
    groups = sorted(
        set(local["cv_group_id"].astype(str))
        & set(random_pairs["cv_group_id"].astype(str))
    )
    group_to_code = {group: index for index, group in enumerate(groups)}
    local = local.loc[local["cv_group_id"].astype(str).isin(groups)].copy()
    random_pairs = random_pairs.loc[
        random_pairs["cv_group_id"].astype(str).isin(groups)
    ].copy()
    local_codes = np.asarray(
        [group_to_code[value] for value in local["cv_group_id"].astype(str)],
        dtype=int,
    )
    random_codes = np.asarray(
        [group_to_code[value] for value in random_pairs["cv_group_id"].astype(str)],
        dtype=int,
    )

    local_median = _median_decision_distance(local)
    random_median = _median_decision_distance(random_pairs)
    point = 1.0 - local_median / random_median if random_median > 0 else np.nan
    multiplicities = _bootstrap_multiplicities(
        len(groups),
        stream_codes,
    )
    sampled_local = _weighted_median_draws(
        local["decision_distance_linf"].to_numpy(dtype=float),
        local_codes,
        multiplicities,
    )
    sampled_random = _weighted_median_draws(
        random_pairs["decision_distance_linf"].to_numpy(dtype=float),
        random_codes,
        multiplicities,
    )
    draws = 1.0 - sampled_local / sampled_random
    low, high = _quantile_interval(draws)
    return point, low, high


def _local_ava(
    ranked: pd.DataFrame,
    selected: pd.DataFrame,
    random_pairs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (domain, representation), all_part in ranked.groupby(
        ["domain", "representation"], sort=True
    ):
        selected_part = selected.loc[
            selected["domain"].eq(domain)
            & selected["representation"].eq(representation)
        ]
        for suite in SCOPES:
            all_subset = _scope(all_part, suite)
            local20 = _scope(selected_part, suite)
            local10 = local20.loc[local20["selected_for_local_ava10"]]
            random_subset = _scope(
                random_pairs.loc[random_pairs["domain"].eq(domain)], suite
            )
            all_point, all_draws = _group_bootstrap(
                all_subset,
                _ava_rate,
                (
                    DOMAIN_CODES[domain],
                    REPRESENTATION_CODES[representation],
                    SCOPE_CODES[suite],
                    31,
                ),
            )
            ava10, ava10_draws = _group_bootstrap(
                local10,
                _ava_rate,
                (
                    DOMAIN_CODES[domain],
                    REPRESENTATION_CODES[representation],
                    SCOPE_CODES[suite],
                    32,
                ),
            )
            ava20, ava20_draws = _group_bootstrap(
                local20,
                _ava_rate,
                (
                    DOMAIN_CODES[domain],
                    REPRESENTATION_CODES[representation],
                    SCOPE_CODES[suite],
                    33,
                ),
            )
            sg_local, sg_draws = _group_bootstrap(
                local10,
                _median_decision_distance,
                (
                    DOMAIN_CODES[domain],
                    REPRESENTATION_CODES[representation],
                    SCOPE_CODES[suite],
                    34,
                ),
            )
            sg_random = _median_decision_distance(random_subset)
            reduction, reduction_low, reduction_high = _matched_local_random_reduction(
                local10,
                random_subset,
                (
                    DOMAIN_CODES[domain],
                    REPRESENTATION_CODES[representation],
                    SCOPE_CODES[suite],
                    35,
                ),
            )
            all_low, all_high = _quantile_interval(all_draws)
            ava10_low, ava10_high = _quantile_interval(ava10_draws)
            ava20_low, ava20_high = _quantile_interval(ava20_draws)
            sg_low, sg_high = _quantile_interval(sg_draws)
            rows.append(
                {
                    "domain": domain,
                    "suite": suite,
                    "representation": representation,
                    "n_all_pairs": len(all_subset),
                    "n_local_ava10_pairs": len(local10),
                    "n_local_ava20_pairs": len(local20),
                    "ava_all": all_point,
                    "ava_all_ci_low": all_low,
                    "ava_all_ci_high": all_high,
                    "local_ava10": ava10,
                    "local_ava10_ci_low": ava10_low,
                    "local_ava10_ci_high": ava10_high,
                    "local_ava20": ava20,
                    "local_ava20_ci_low": ava20_low,
                    "local_ava20_ci_high": ava20_high,
                    "local_aliasing_ratio": ava10 / all_point if all_point > 0 else np.nan,
                    "sg_local": sg_local,
                    "sg_local_ci_low": sg_low,
                    "sg_local_ci_high": sg_high,
                    "sg_random": sg_random,
                    "local_sufficiency_reduction": reduction,
                    "local_sufficiency_reduction_ci_low": reduction_low,
                    "local_sufficiency_reduction_ci_high": reduction_high,
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "bootstrap_unit": "cv_group_id",
                    "new_objective_fe": NEW_OBJECTIVE_FE,
                }
            )
    return pd.DataFrame(rows)


def _heterogeneity(decision: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        domain_part = decision.loc[decision["domain"].eq(domain)]
        suite_rates = {
            suite: _ava_rate(_scope(domain_part, suite)) for suite in ("bbob", "mabbob")
        }
        for suite in SCOPES:
            subset = _scope(domain_part, suite)
            raw_rate = _ava_rate(subset)
            balanced, low, high, n_groups = _group_mean_sensitivity(
                subset,
                _ava_rate,
                (DOMAIN_CODES[domain], SCOPE_CODES[suite], 41),
            )
            if suite == "pooled":
                if (
                    min(raw_rate, balanced) >= 0.10
                    and min(suite_rates.values()) >= 0.05
                ):
                    verdict = "DH1 NONTRIVIAL"
                elif max(raw_rate, balanced) >= 0.05:
                    verdict = "DH2 WEAK"
                else:
                    verdict = "DH3 DEGENERATE"
            elif min(raw_rate, balanced) >= 0.05:
                verdict = "DH1 NONTRIVIAL"
            elif max(raw_rate, balanced) >= 0.05:
                verdict = "DH2 WEAK"
            else:
                verdict = "DH3 DEGENERATE"
            rows.append(
                {
                    "domain": domain,
                    "suite": suite,
                    "n_pairs": len(subset),
                    "decision_distance_gt1_rate": raw_rate,
                    "cv_group_balanced_gt1_rate": balanced,
                    "cv_group_balanced_ci_low": low,
                    "cv_group_balanced_ci_high": high,
                    "n_cv_groups": n_groups,
                    "weighting_difference": balanced - raw_rate,
                    "heterogeneity_verdict": verdict,
                    "bootstrap_unit": "cv_group_id",
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "new_objective_fe": NEW_OBJECTIVE_FE,
                }
            )
    return pd.DataFrame(rows)


def _natural_post_shift(
    ranked: pd.DataFrame,
    selected: pd.DataFrame,
    nn: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    local10 = selected.loc[selected["selected_for_local_ava10"]]
    for representation in BASE_REPRESENTATIONS:
        for suite in SCOPES:
            natural_alignment = _scope(
                ranked.loc[
                    ranked["domain"].eq("natural")
                    & ranked["representation"].eq(representation)
                ],
                suite,
            )
            post_alignment = _scope(
                ranked.loc[
                    ranked["domain"].eq("post_handoff")
                    & ranked["representation"].eq(representation)
                ],
                suite,
            )
            rho_difference, rho_draws, n_groups_rho = _matched_group_difference(
                natural_alignment,
                post_alignment,
                _within_rank_alignment,
                (REPRESENTATION_CODES[representation], SCOPE_CODES[suite], 51),
            )

            natural_ava = _scope(
                local10.loc[
                    local10["domain"].eq("natural")
                    & local10["representation"].eq(representation)
                ],
                suite,
            )
            post_ava = _scope(
                local10.loc[
                    local10["domain"].eq("post_handoff")
                    & local10["representation"].eq(representation)
                ],
                suite,
            )
            ava_difference, ava_draws, n_groups_ava = _matched_group_difference(
                natural_ava,
                post_ava,
                _ava_rate,
                (REPRESENTATION_CODES[representation], SCOPE_CODES[suite], 52),
            )

            natural_nn = _scope(
                nn.loc[
                    nn["domain"].eq("natural")
                    & nn["representation"].eq(representation)
                ],
                suite,
            )
            post_nn = _scope(
                nn.loc[
                    nn["domain"].eq("post_handoff")
                    & nn["representation"].eq(representation)
                ],
                suite,
            )
            nn_difference, nn_draws, n_groups_nn = _matched_group_difference(
                natural_nn,
                post_nn,
                _nn_delta,
                (REPRESENTATION_CODES[representation], SCOPE_CODES[suite], 53),
            )

            rho_low, rho_high = _quantile_interval(rho_draws)
            ava_low, ava_high = _quantile_interval(ava_draws)
            nn_low, nn_high = _quantile_interval(nn_draws)
            weakening_support = int(rho_high < 0) + int(ava_low > 0) + int(nn_high < 0)
            strong_opposite = int(rho_low > 0) + int(ava_high < 0) + int(nn_low > 0)
            rows.append(
                {
                    "representation": representation,
                    "suite": suite,
                    "rho_natural": _within_rank_alignment(natural_alignment),
                    "rho_post_handoff": _within_rank_alignment(post_alignment),
                    "delta_rho_post_minus_natural": rho_difference,
                    "delta_rho_ci_low": rho_low,
                    "delta_rho_ci_high": rho_high,
                    "local_ava10_natural": _ava_rate(natural_ava),
                    "local_ava10_post_handoff": _ava_rate(post_ava),
                    "delta_local_ava10_post_minus_natural": ava_difference,
                    "delta_local_ava10_ci_low": ava_low,
                    "delta_local_ava10_ci_high": ava_high,
                    "delta_nn_natural": _nn_delta(natural_nn),
                    "delta_nn_post_handoff": _nn_delta(post_nn),
                    "delta_nn_post_minus_natural": nn_difference,
                    "delta_nn_ci_low": nn_low,
                    "delta_nn_ci_high": nn_high,
                    "post_weakening_supported_metric_count": weakening_support,
                    "strong_opposite_metric_count": strong_opposite,
                    "matched_cv_groups_rho": n_groups_rho,
                    "matched_cv_groups_ava": n_groups_ava,
                    "matched_cv_groups_nn": n_groups_nn,
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "bootstrap_unit": "matched_cv_group_id",
                    "new_objective_fe": NEW_OBJECTIVE_FE,
                }
            )
    return pd.DataFrame(rows)


def _leave_one_group_out_ava_improvement(
    baseline: pd.DataFrame,
    comparison: pd.DataFrame,
) -> tuple[bool, float, float]:
    groups = sorted(
        set(baseline["cv_group_id"].astype(str))
        & set(comparison["cv_group_id"].astype(str))
    )
    differences = []
    for group in groups:
        baseline_subset = baseline.loc[baseline["cv_group_id"].astype(str).ne(group)]
        comparison_subset = comparison.loc[
            comparison["cv_group_id"].astype(str).ne(group)
        ]
        differences.append(_ava_rate(comparison_subset) - _ava_rate(baseline_subset))
    if not differences:
        return False, float("nan"), float("nan")
    return bool(max(differences) < 0), float(min(differences)), float(max(differences))


def _representation_ladder(
    ranked: pd.DataFrame,
    selected: pd.DataFrame,
    nn: pd.DataFrame,
    alignment: pd.DataFrame,
    nn_summary: pd.DataFrame,
    local_ava: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    local10 = selected.loc[
        selected["domain"].eq("post_handoff")
        & selected["selected_for_local_ava10"]
    ]
    post_ranked = ranked.loc[ranked["domain"].eq("post_handoff")]
    post_nn = nn.loc[nn["domain"].eq("post_handoff")]
    for representation in POST_LADDER:
        for suite in SCOPES:
            result_alignment = alignment.loc[
                alignment["domain"].eq("post_handoff")
                & alignment["suite"].eq(suite)
                & alignment["representation"].eq(representation)
            ].iloc[0]
            result_nn = nn_summary.loc[
                nn_summary["domain"].eq("post_handoff")
                & nn_summary["suite"].eq(suite)
                & nn_summary["representation"].eq(representation)
            ].iloc[0]
            result_ava = local_ava.loc[
                local_ava["domain"].eq("post_handoff")
                & local_ava["suite"].eq(suite)
                & local_ava["representation"].eq(representation)
            ].iloc[0]

            if representation == "global28":
                rho_difference = ava_difference = nn_difference = 0.0
                rho_draws = ava_draws = nn_draws = np.zeros(BOOTSTRAP_DRAWS)
                loo_stable, loo_min, loo_max = True, 0.0, 0.0
                matched_groups = int(result_alignment["n_cv_groups"])
            else:
                baseline_alignment = _scope(
                    post_ranked.loc[post_ranked["representation"].eq("global28")],
                    suite,
                )
                comparison_alignment = _scope(
                    post_ranked.loc[post_ranked["representation"].eq(representation)],
                    suite,
                )
                rho_difference, rho_draws, matched_groups = _matched_group_difference(
                    baseline_alignment,
                    comparison_alignment,
                    _within_rank_alignment,
                    (
                        LADDER_STREAM,
                        REPRESENTATION_CODES[representation],
                        SCOPE_CODES[suite],
                        61,
                    ),
                )
                baseline_ava = _scope(
                    local10.loc[local10["representation"].eq("global28")], suite
                )
                comparison_ava = _scope(
                    local10.loc[local10["representation"].eq(representation)], suite
                )
                ava_difference, ava_draws, _ = _matched_group_difference(
                    baseline_ava,
                    comparison_ava,
                    _ava_rate,
                    (
                        LADDER_STREAM,
                        REPRESENTATION_CODES[representation],
                        SCOPE_CODES[suite],
                        62,
                    ),
                )
                baseline_nn = _scope(
                    post_nn.loc[post_nn["representation"].eq("global28")], suite
                )
                comparison_nn = _scope(
                    post_nn.loc[post_nn["representation"].eq(representation)], suite
                )
                nn_difference, nn_draws, _ = _matched_group_difference(
                    baseline_nn,
                    comparison_nn,
                    _nn_delta,
                    (
                        LADDER_STREAM,
                        REPRESENTATION_CODES[representation],
                        SCOPE_CODES[suite],
                        63,
                    ),
                )
                loo_stable, loo_min, loo_max = _leave_one_group_out_ava_improvement(
                    baseline_ava,
                    comparison_ava,
                )
            rho_low, rho_high = _quantile_interval(rho_draws)
            ava_low, ava_high = _quantile_interval(ava_draws)
            nn_low, nn_high = _quantile_interval(nn_draws)
            rows.append(
                {
                    "domain": "post_handoff",
                    "suite": suite,
                    "representation": representation,
                    "rho_rank_within": result_alignment["rho_rank_within"],
                    "rho_ci_low": result_alignment["rho_rank_within_ci_low"],
                    "rho_ci_high": result_alignment["rho_rank_within_ci_high"],
                    "alignment_verdict": result_alignment["alignment_verdict"],
                    "local_ava10": result_ava["local_ava10"],
                    "local_ava10_ci_low": result_ava["local_ava10_ci_low"],
                    "local_ava10_ci_high": result_ava["local_ava10_ci_high"],
                    "delta_nn": result_nn["delta_nn"],
                    "delta_nn_ci_low": result_nn["delta_nn_ci_low"],
                    "delta_nn_ci_high": result_nn["delta_nn_ci_high"],
                    "delta_rho_vs_global28": rho_difference,
                    "delta_rho_vs_global28_ci_low": rho_low,
                    "delta_rho_vs_global28_ci_high": rho_high,
                    "delta_local_ava10_vs_global28": ava_difference,
                    "delta_local_ava10_vs_global28_ci_low": ava_low,
                    "delta_local_ava10_vs_global28_ci_high": ava_high,
                    "delta_nn_vs_global28": nn_difference,
                    "delta_nn_vs_global28_ci_low": nn_low,
                    "delta_nn_vs_global28_ci_high": nn_high,
                    "ava_improvement_after_every_group_omission": loo_stable,
                    "ava_loo_difference_min": loo_min,
                    "ava_loo_difference_max": loo_max,
                    "matched_cv_groups": matched_groups,
                    "reference_representation": "global28",
                    "new_objective_fe": NEW_OBJECTIVE_FE,
                }
            )
    return pd.DataFrame(rows)


def _collision_examples(
    selected: pd.DataFrame,
    states: pd.DataFrame,
) -> pd.DataFrame:
    local10 = selected.loc[selected["selected_for_local_ava10"]].copy()
    examples = []
    for (domain, suite), part in local10.groupby(["domain", "suite"], sort=True):
        chosen = (
            part.sort_values(
                ["decision_distance_linf", "representation_distance_l1_mean"],
                ascending=[False, True],
            )
            .drop_duplicates("problem_id", keep="first")
            .head(10)
            .copy()
        )
        chosen["example_rank"] = np.arange(1, len(chosen) + 1, dtype=int)
        examples.append(chosen)
    output = pd.concat(examples, ignore_index=True)
    state_columns = [
        "state_id",
        "loss_shade",
        "loss_lshade",
        "loss_cso",
        *MARGIN_COLUMNS,
        "A_ND_members",
    ]
    state_i = states[state_columns].rename(
        columns={column: f"{column}_i" for column in state_columns if column != "state_id"}
    )
    state_j = states[state_columns].rename(
        columns={column: f"{column}_j" for column in state_columns if column != "state_id"}
    )
    output = output.merge(
        state_i.rename(columns={"state_id": "state_i"}),
        on="state_i",
        validate="many_to_one",
    )
    output = output.merge(
        state_j.rename(columns={"state_id": "state_j"}),
        on="state_j",
        validate="many_to_one",
    )
    output["selection_rule"] = (
        "highest decision distance among stratum-local observable-nearest pairs; "
        "at most one row per problem"
    )
    output["new_objective_fe"] = NEW_OBJECTIVE_FE
    return output


def _statistical_contract_check() -> pd.DataFrame:
    rows = [
        {
            "issue_id": "P1_mixed_correlation_estimand",
            "old_behavior": "pooled pairwise Spearman point with mean group-Spearman interval",
            "corrected_behavior": "point and grouped bootstrap both recompute within-stratum rank Spearman",
            "old_estimand": "pooled distance correlation plus group-mean correlation",
            "corrected_estimand": "rho_rank_within",
            "scientific_impact": "point, interval, permutation, and CA classification now refer to one quantity",
            "changes_point_estimate": True,
            "changes_ci": True,
            "changes_verdict": True,
        },
        {
            "issue_id": "P2_bootstrap_grouping",
            "old_behavior": "cv_group_id concatenated with problem_id",
            "corrected_behavior": "cv_group_id is the sole resampling unit",
            "old_estimand": "instance-split grouped summary",
            "corrected_estimand": "function-level grouped dependence",
            "scientific_impact": "BBOB instances remain in the same dependence block",
            "changes_point_estimate": False,
            "changes_ci": True,
            "changes_verdict": True,
        },
        {
            "issue_id": "P3_global_near_region",
            "old_behavior": "suite-wide bottom 10 percent representation distance",
            "corrected_behavior": "one observable-nearest pair per ten-pair stratum",
            "old_estimand": "global AVA10",
            "corrected_estimand": "stratum-local AVA10",
            "scientific_impact": "near-state mismatch now answers the within-context question",
            "changes_point_estimate": True,
            "changes_ci": True,
            "changes_verdict": True,
        },
        {
            "issue_id": "P4_fallback_verdict",
            "old_behavior": "V2 assigned when other branches did not trigger",
            "corrected_behavior": "C1-C4 each has explicit evidence conditions",
            "old_estimand": "fallback category",
            "corrected_estimand": "evidence-defined mechanism category",
            "scientific_impact": "the final category is no longer interpreted as direct support by default",
            "changes_point_estimate": False,
            "changes_ci": False,
            "changes_verdict": True,
        },
        {
            "issue_id": "P5_permutation_between_stratum_scale_mixing",
            "old_behavior": "permuted decision distances correlated on pooled raw scales",
            "corrected_behavior": "both distances are rank-normalized inside every stratum after permutation",
            "old_estimand": "pooled-scale permutation correlation",
            "corrected_estimand": "within-stratum geometry independence null",
            "scientific_impact": "the null targets local ordering rather than between-stratum scale",
            "changes_point_estimate": False,
            "changes_ci": False,
            "changes_verdict": True,
        },
    ]
    output = pd.DataFrame(rows)
    output["new_objective_fe"] = NEW_OBJECTIVE_FE
    return output


def _read_markdown_table(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8").splitlines()
    table_lines = []
    started = False
    for line in lines:
        if line.startswith("|"):
            table_lines.append(line)
            started = True
        elif started:
            break
    if len(table_lines) < 3:
        raise RuntimeError(f"no markdown table found in {path}")
    header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in table_lines[2:]
    ]
    frame = pd.DataFrame(rows, columns=header)
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().all():
            frame[column] = converted
    return frame


def _derive_verdicts(
    heterogeneity: pd.DataFrame,
    alignment: pd.DataFrame,
    local_ava: pd.DataFrame,
    shift: pd.DataFrame,
    ladder: pd.DataFrame,
) -> dict[str, object]:
    pooled_heterogeneity = heterogeneity.loc[heterogeneity["suite"].eq("pooled")]
    dh1 = bool(pooled_heterogeneity["heterogeneity_verdict"].eq("DH1 NONTRIVIAL").all())

    base_pooled_alignment = alignment.loc[
        alignment["suite"].eq("pooled")
        & alignment["representation"].isin(BASE_REPRESENTATIONS)
    ]
    base_suite_alignment = alignment.loc[
        alignment["suite"].isin(("bbob", "mabbob"))
        & alignment["representation"].isin(BASE_REPRESENTATIONS)
    ]
    all_base_ca1 = bool(
        len(base_pooled_alignment) == 4
        and base_pooled_alignment["alignment_verdict"].eq("CA1 ROBUST ALIGNMENT").all()
    )
    no_suite_inverse = bool(
        ~(
            (base_suite_alignment["rho_rank_within"] <= 0)
            & (base_suite_alignment["rho_rank_within_ci_high"] <= 0)
        ).any()
    )
    any_alignment_support = bool(
        (base_pooled_alignment["alignment_verdict"] != "CA3 NO ALIGNMENT").any()
    )
    all_base_rho_above_zero = bool((base_pooled_alignment["rho_rank_within"] > 0).all())

    post_local = local_ava.loc[
        local_ava["domain"].eq("post_handoff")
        & local_ava["suite"].eq("pooled")
        & local_ava["representation"].isin(BASE_REPRESENTATIONS)
    ]
    substantial_local_aliasing = bool(
        len(post_local) == 2 and post_local["local_ava10"].ge(0.30).all()
    )
    observable_proximity_helps = bool(
        len(post_local) == 2 and post_local["local_aliasing_ratio"].lt(1.0).all()
    )

    if not dh1:
        final_verdict = "C4 DECISION HETEROGENEITY NOT ROBUST"
    elif (
        all_base_ca1
        and no_suite_inverse
        and substantial_local_aliasing
        and observable_proximity_helps
    ):
        final_verdict = (
            "C1 ROBUST PARTIAL DECISION ALIGNMENT WITH SUBSTANTIAL RESIDUAL ALIASING"
        )
    elif all_base_rho_above_zero and any_alignment_support:
        final_verdict = "C2 WEAK OR FRAGILE DECISION ALIGNMENT"
    else:
        final_verdict = "C3 NO RELIABLE OBSERVABLE DECISION ALIGNMENT"

    pooled_shift = shift.loc[shift["suite"].eq("pooled")]
    suite_shift = shift.loc[shift["suite"].isin(("bbob", "mabbob"))]
    if (
        pooled_shift["post_weakening_supported_metric_count"].ge(2).any()
        and not suite_shift["strong_opposite_metric_count"].gt(0).any()
    ):
        shift_verdict = "DS1 POST-HANDOFF WEAKENING"
    elif (
        shift["post_weakening_supported_metric_count"].gt(0).any()
        and shift["strong_opposite_metric_count"].gt(0).any()
    ):
        shift_verdict = "DS3 MIXED SHIFT"
    else:
        shift_verdict = "DS2 NO ROBUST SHIFT"

    alternatives = [representation for representation in POST_LADDER if representation != "global28"]
    clear_gain = False
    any_gain = False
    for representation in alternatives:
        suite_rows = ladder.loc[
            ladder["representation"].eq(representation)
            & ladder["suite"].isin(("bbob", "mabbob"))
        ]
        ava_lower_both = bool(
            len(suite_rows) == 2
            and suite_rows["delta_local_ava10_vs_global28"].lt(0).all()
        )
        ava_interval_support = bool(
            suite_rows["delta_local_ava10_vs_global28_ci_high"].lt(0).any()
        )
        rho_not_lower_both = bool(suite_rows["delta_rho_vs_global28"].ge(0).all())
        nn_not_lower_both = bool(suite_rows["delta_nn_vs_global28"].ge(0).all())
        leave_one_group_stable = bool(
            suite_rows["ava_improvement_after_every_group_omission"].all()
        )
        clear_gain = clear_gain or (
            ava_lower_both
            and ava_interval_support
            and (rho_not_lower_both or nn_not_lower_both)
            and leave_one_group_stable
        )
        any_gain = any_gain or bool(
            suite_rows[
                [
                    "delta_rho_vs_global28",
                    "delta_nn_vs_global28",
                ]
            ].gt(0).any().any()
            or suite_rows["delta_local_ava10_vs_global28"].lt(0).any()
        )
    if clear_gain:
        ladder_verdict = "LR1 CLEARLY BETTER REPRESENTATION EXISTS"
    elif any_gain:
        ladder_verdict = "LR2 PARTIAL OR INCONSISTENT"
    else:
        ladder_verdict = "LR3 NO MEANINGFUL REPRESENTATION GAIN"

    if final_verdict.startswith("C1"):
        scientific_statement = (
            "Observable search behavior shows statistically detectable within-context "
            "alignment with observed alternate-action decision geometry, while substantial "
            "action-value aliasing remains among locally nearest observable states."
        )
    elif final_verdict.startswith("C2"):
        scientific_statement = (
            "Evidence for behavior-decision alignment is weak or statistically fragile "
            "after unifying the estimand and dependence structure."
        )
    elif final_verdict.startswith("C3"):
        scientific_statement = (
            "Observed alternate-action decision heterogeneity persists, but the tested "
            "behavior representations do not reliably preserve its local geometry."
        )
    else:
        scientific_statement = (
            "Trajectory-specific alternate-action heterogeneity is not sufficiently robust "
            "under the corrected grouped analysis to support the mechanism study."
        )
    return {
        "heterogeneity_verdict": "DH1 NONTRIVIAL" if dh1 else "DH NOT ROBUST",
        "final_verdict": final_verdict,
        "shift_verdict": shift_verdict,
        "ladder_verdict": ladder_verdict,
        "all_base_pooled_ca1": all_base_ca1,
        "no_suite_significantly_inverse": no_suite_inverse,
        "substantial_post_local_aliasing": substantial_local_aliasing,
        "observable_proximity_reduces_aliasing": observable_proximity_helps,
        "scientific_statement": scientific_statement,
        "behavior_sufficient_for_dynamic_selection": False,
        "new_selector_allowed": False,
        "solver_internal_state_allowed": False,
        "seeds_6_10_allowed": False,
        "cec_allowed": False,
        "next_step_requires_task17a1_based_redesign": True,
        "new_objective_fe": NEW_OBJECTIVE_FE,
    }


def _old_vs_corrected(
    inputs: dict[str, pd.DataFrame],
    alignment: pd.DataFrame,
    nn_summary: pd.DataFrame,
    local_ava: pd.DataFrame,
    shift: pd.DataFrame,
    verdicts: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_numeric(
        metric: str,
        domain: str,
        suite: str,
        representation: str,
        old_value: float,
        corrected_value: float,
        old_low: float = np.nan,
        old_high: float = np.nan,
        corrected_low: float = np.nan,
        corrected_high: float = np.nan,
        reason: str = "",
    ) -> None:
        rows.append(
            {
                "metric": metric,
                "domain": domain,
                "suite": suite,
                "representation": representation,
                "old_value": str(old_value),
                "corrected_value": str(corrected_value),
                "numeric_difference": corrected_value - old_value,
                "old_ci_low": old_low,
                "old_ci_high": old_high,
                "corrected_ci_low": corrected_low,
                "corrected_ci_high": corrected_high,
                "reason": reason,
            }
        )

    old_alignment = inputs["old_alignment"]
    for corrected in alignment.itertuples(index=False):
        old = old_alignment.loc[
            old_alignment["domain"].eq(corrected.domain)
            & old_alignment["suite"].eq(corrected.suite)
            & old_alignment["representation"].eq(corrected.representation)
        ].iloc[0]
        add_numeric(
            "rho",
            corrected.domain,
            corrected.suite,
            corrected.representation,
            float(old["rho_observed_pooled"]),
            float(corrected.rho_rank_within),
            float(old["rho_ci_low"]),
            float(old["rho_ci_high"]),
            float(corrected.rho_rank_within_ci_low),
            float(corrected.rho_rank_within_ci_high),
            "stratum-local ranks and one shared correlation estimand",
        )
        add_numeric(
            "permutation_q97_5",
            corrected.domain,
            corrected.suite,
            corrected.representation,
            float(old["permutation_q97_5"]),
            float(corrected.permutation_q97_5),
            reason="permuted decision distances are rank-normalized inside each stratum",
        )

    old_nn = _read_markdown_table(OLD_LIGHT / "17a07_nearest_neighbor_consistency.md")
    for corrected in nn_summary.itertuples(index=False):
        old = old_nn.loc[
            old_nn["domain"].eq(corrected.domain)
            & old_nn["suite"].eq(corrected.suite)
            & old_nn["representation"].eq(corrected.representation)
        ].iloc[0]
        add_numeric(
            "delta_nn",
            corrected.domain,
            corrected.suite,
            corrected.representation,
            float(old["delta_nn"]),
            float(corrected.delta_nn),
            float(old["delta_nn_ci_low"]),
            float(old["delta_nn_ci_high"]),
            float(corrected.delta_nn_ci_low),
            float(corrected.delta_nn_ci_high),
            "interval now resamples cv_group_id and recomputes the overall median difference",
        )

    old_aliasing = inputs["old_aliasing"]
    for corrected in local_ava.itertuples(index=False):
        old = old_aliasing.loc[
            old_aliasing["domain"].eq(corrected.domain)
            & old_aliasing["suite"].eq(corrected.suite)
            & old_aliasing["representation"].eq(corrected.representation)
        ].iloc[0]
        add_numeric(
            "ava10",
            corrected.domain,
            corrected.suite,
            corrected.representation,
            float(old["ava_10"]),
            float(corrected.local_ava10),
            float(old["ava_10_ci_low"]),
            float(old["ava_10_ci_high"]),
            float(corrected.local_ava10_ci_low),
            float(corrected.local_ava10_ci_high),
            "one nearest pair per stratum replaces a suite-wide cutoff",
        )

    old_shift = inputs["old_shift"]
    for corrected in shift.itertuples(index=False):
        old = old_shift.loc[
            old_shift["suite"].eq(corrected.suite)
            & old_shift["representation"].eq(corrected.representation)
        ].iloc[0]
        add_numeric(
            "natural_post_delta_rho",
            "cross_domain",
            corrected.suite,
            corrected.representation,
            float(old["delta_rho_post_minus_natural"]),
            float(corrected.delta_rho_post_minus_natural),
            float(old["delta_rho_ci_low"]),
            float(old["delta_rho_ci_high"]),
            float(corrected.delta_rho_ci_low),
            float(corrected.delta_rho_ci_high),
            "matched cv_group resampling with corrected rank alignment",
        )

    old_verdict = json.loads((OLD_LIGHT / "task17a_verdict.json").read_text(encoding="utf-8"))
    rows.extend(
        [
            {
                "metric": "natural_post_shift_verdict",
                "domain": "cross_domain",
                "suite": "pooled",
                "representation": "compact6|global28",
                "old_value": "H-SUFF=" + str(old_verdict["h_suff_supported"]),
                "corrected_value": str(verdicts["shift_verdict"]),
                "numeric_difference": np.nan,
                "old_ci_low": np.nan,
                "old_ci_high": np.nan,
                "corrected_ci_low": np.nan,
                "corrected_ci_high": np.nan,
                "reason": "matched differences replace independent or problem-level summaries",
            },
            {
                "metric": "representation_ladder_verdict",
                "domain": "post_handoff",
                "suite": "all",
                "representation": "representation_ladder",
                "old_value": str(old_verdict["ladder_verdict"]),
                "corrected_value": str(verdicts["ladder_verdict"]),
                "numeric_difference": np.nan,
                "old_ci_low": np.nan,
                "old_ci_high": np.nan,
                "corrected_ci_low": np.nan,
                "corrected_ci_high": np.nan,
                "reason": "comparison uses local AVA10 and matched differences versus Global28",
            },
            {
                "metric": "final_verdict",
                "domain": "all",
                "suite": "all",
                "representation": "all",
                "old_value": str(old_verdict["final_verdict"]),
                "corrected_value": str(verdicts["final_verdict"]),
                "numeric_difference": np.nan,
                "old_ci_low": np.nan,
                "old_ci_high": np.nan,
                "corrected_ci_low": np.nan,
                "corrected_ci_high": np.nan,
                "reason": "explicit C1-C4 evidence conditions replace the fallback branch",
            },
        ]
    )
    output = pd.DataFrame(rows)
    output["new_objective_fe"] = NEW_OBJECTIVE_FE
    return output


def _suite_robustness(
    heterogeneity: pd.DataFrame,
    alignment: pd.DataFrame,
    nn_summary: pd.DataFrame,
    local_ava: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["domain", "suite", "representation"]
    output = alignment.loc[alignment["suite"].isin(("bbob", "mabbob"))][
        [
            *keys,
            "rho_rank_within",
            "rho_rank_within_ci_low",
            "rho_rank_within_ci_high",
            "alignment_verdict",
        ]
    ].merge(
        nn_summary[[*keys, "delta_nn", "delta_nn_ci_low", "delta_nn_ci_high"]],
        on=keys,
        validate="one_to_one",
    )
    output = output.merge(
        local_ava[
            [
                *keys,
                "local_ava10",
                "local_ava10_ci_low",
                "local_ava10_ci_high",
                "local_aliasing_ratio",
            ]
        ],
        on=keys,
        validate="one_to_one",
    )
    heterogeneity_lookup = heterogeneity.set_index(["domain", "suite"])[
        "cv_group_balanced_gt1_rate"
    ]
    output["cv_group_balanced_gt1_rate"] = [
        heterogeneity_lookup.loc[(row.domain, row.suite)]
        for row in output.itertuples(index=False)
    ]
    output["new_objective_fe"] = NEW_OBJECTIVE_FE
    return output


def _make_figures(
    ranked: pd.DataFrame,
    alignment: pd.DataFrame,
    local_ava: pd.DataFrame,
    nn_summary: pd.DataFrame,
) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = []

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True, sharey=True)
    for axis, domain in zip(axes, DOMAINS):
        subset = ranked.loc[
            ranked["domain"].eq(domain)
            & ranked["representation"].eq("global28")
        ]
        metric = alignment.loc[
            alignment["domain"].eq(domain)
            & alignment["suite"].eq("pooled")
            & alignment["representation"].eq("global28")
        ].iloc[0]
        density = axis.hexbin(
            subset["representation_rank_within_stratum"],
            subset["decision_rank_within_stratum"],
            gridsize=18,
            mincnt=1,
            cmap="viridis",
        )
        axis.plot([0, 1], [0, 1], color="#555555", linewidth=0.8, linestyle="--")
        axis.set_title("Natural" if domain == "natural" else "Post-handoff")
        axis.set_xlabel("Observable distance rank within stratum")
        axis.text(
            0.03,
            0.97,
            (
                f"rho={metric['rho_rank_within']:.3f}\n"
                f"95% CI [{metric['rho_rank_within_ci_low']:.3f}, "
                f"{metric['rho_rank_within_ci_high']:.3f}]\n"
                f"perm q97.5={metric['permutation_q97_5']:.3f}"
            ),
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#AAAAAA", "alpha": 0.9},
        )
    axes[0].set_ylabel("Decision distance rank within stratum")
    figure.suptitle("Corrected Within-Stratum Alignment: Global28")
    figure.subplots_adjust(left=0.08, right=0.88, bottom=0.14, top=0.86, wspace=0.12)
    color_axis = figure.add_axes((0.91, 0.18, 0.016, 0.64))
    figure.colorbar(density, cax=color_axis, label="Pair count")
    path = FIGURES / "figure_a_corrected_within_stratum_alignment.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for axis, domain in zip(axes, DOMAINS):
        representations = (
            BASE_REPRESENTATIONS if domain == "natural" else POST_LADDER
        )
        subset = local_ava.loc[
            local_ava["domain"].eq(domain)
            & local_ava["suite"].eq("pooled")
            & local_ava["representation"].isin(representations)
        ].set_index("representation").loc[list(representations)]
        positions = np.arange(len(subset))
        values = subset["local_ava10"].to_numpy(dtype=float)
        errors = np.vstack(
            [
                values - subset["local_ava10_ci_low"].to_numpy(dtype=float),
                subset["local_ava10_ci_high"].to_numpy(dtype=float) - values,
            ]
        )
        axis.errorbar(
            positions,
            values,
            yerr=errors,
            fmt="o",
            color="#B23A48" if domain == "natural" else "#2A7F62",
            capsize=4,
            linewidth=1.5,
        )
        axis.axhline(
            float(subset["ava_all"].iloc[0]),
            color="#555555",
            linestyle="--",
            linewidth=1.0,
            label="All within-stratum pairs",
        )
        axis.set_xticks(positions, [name.replace("_", "\n") for name in representations])
        axis.set_title("Natural" if domain == "natural" else "Post-handoff")
        axis.set_xlabel("Representation")
        axis.set_ylim(0, 1)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Local AVA10: P(decision distance > 1)")
    figure.suptitle("Stratum-Local AVA10 by Representation")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path = FIGURES / "figure_b_local_ava10_representation_check.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    colors = {"natural": "#2A6F97", "post_handoff": "#C45D32"}
    metric_specs = (
        (
            axes[0],
            alignment,
            "rho_rank_within",
            "rho_rank_within_ci_low",
            "rho_rank_within_ci_high",
            "Within-stratum rho",
        ),
        (
            axes[1],
            local_ava,
            "local_ava10",
            "local_ava10_ci_low",
            "local_ava10_ci_high",
            "Local AVA10",
        ),
        (
            axes[2],
            nn_summary,
            "delta_nn",
            "delta_nn_ci_low",
            "delta_nn_ci_high",
            "NN median reduction",
        ),
    )
    positions = np.arange(len(BASE_REPRESENTATIONS))
    width = 0.34
    for axis, table, value_column, low_column, high_column, title in metric_specs:
        for offset, domain in zip((-width / 2, width / 2), DOMAINS):
            subset = table.loc[
                table["domain"].eq(domain)
                & table["suite"].eq("pooled")
                & table["representation"].isin(BASE_REPRESENTATIONS)
            ].set_index("representation").loc[list(BASE_REPRESENTATIONS)]
            values = subset[value_column].to_numpy(dtype=float)
            errors = np.vstack(
                [
                    values - subset[low_column].to_numpy(dtype=float),
                    subset[high_column].to_numpy(dtype=float) - values,
                ]
            )
            axis.bar(
                positions + offset,
                values,
                width,
                color=colors[domain],
                label="Natural" if domain == "natural" else "Post-handoff",
                alpha=0.9,
            )
            axis.errorbar(
                positions + offset,
                values,
                yerr=errors,
                fmt="none",
                ecolor="#222222",
                capsize=3,
                linewidth=1.0,
            )
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set_xticks(positions, ("Compact6", "Global28"))
        axis.set_title(title)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Natural and Post-handoff: Corrected Primary Metrics")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path = FIGURES / "figure_c_natural_post_primary_metrics.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)
    return paths


def _resource_usage(
    wall_start: float,
    cpu_start: float,
    figures: list[Path],
) -> pd.DataFrame:
    peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_rss_mb = (
        peak_rss / (1024.0 * 1024.0)
        if sys.platform == "darwin"
        else peak_rss / 1024.0
    )
    return pd.DataFrame(
        [
            {
                "new_objective_fe": NEW_OBJECTIVE_FE,
                "natural_states_reused": 1890,
                "post_states_reused": 3780,
                "natural_pairs_reused": 3780,
                "post_pairs_reused": 7560,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "permutations": PERMUTATIONS,
                "random_neighbor_controls": RANDOM_NEIGHBOR_CONTROLS,
                "analysis_cpu_seconds": process_time() - cpu_start,
                "wall_seconds": perf_counter() - wall_start,
                "peak_rss_mb": peak_rss_mb,
                "figures": ";".join(str(path.relative_to(ROOT)) for path in figures),
                "input_scope": "Task 17A products in the current project",
            }
        ]
    )


def _consistency_checks(
    states: pd.DataFrame,
    pair_identity: pd.DataFrame,
    selected: pd.DataFrame,
    alignment: pd.DataFrame,
    bootstrap: pd.DataFrame,
    permutations: pd.DataFrame,
    products: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    group_columns = ["domain", "representation", "stratum_id"]
    local_counts = selected.groupby(group_columns, sort=False)["local_order"].agg(
        local10=lambda values: int(np.sum(values == 1)),
        local20="size",
    )
    checks = [
        (
            "C01_shared_alignment_statistic",
            set(alignment["statistic_function"])
            == set(bootstrap["statistic_function"])
            == set(permutations["statistic_function"])
            == {"_within_rank_alignment"},
            "point, grouped bootstrap, and permutation call one core function",
        ),
        (
            "C02_cv_group_block_resampling",
            alignment["bootstrap_unit"].eq("cv_group_id").all()
            and bootstrap["bootstrap_unit"].eq("cv_group_id").all(),
            "resampling unit is cv_group_id without problem_id subdivision",
        ),
        (
            "C03_one_local_ava10_pair_per_stratum",
            local_counts["local10"].eq(1).all(),
            "every domain-representation-stratum contributes one pair",
        ),
        (
            "C04_two_local_ava20_pairs_per_stratum",
            local_counts["local20"].eq(2).all(),
            "every domain-representation-stratum contributes two pairs",
        ),
        (
            "C05_no_cross_domain_pairs",
            not pair_identity["cross_domain"].any()
            and set(pair_identity["domain"]) == set(DOMAINS),
            "both state endpoints match the pair domain",
        ),
        (
            "C06_zero_new_objective_fe",
            NEW_OBJECTIVE_FE == 0
            and all(
                "new_objective_fe" in table.columns
                and table["new_objective_fe"].eq(0).all()
                for table in products.values()
            ),
            "all result tables report zero newly evaluated objective FE",
        ),
        (
            "C07_state_and_pair_counts",
            len(states) == 5670 and len(pair_identity) == 11340,
            "1890 plus 3780 states and 3780 plus 7560 pairs",
        ),
        (
            "C08_draw_counts",
            bootstrap.groupby(["domain", "suite", "representation"]).size().eq(
                BOOTSTRAP_DRAWS
            ).all()
            and permutations.groupby(["domain", "suite", "representation"]).size().eq(
                PERMUTATIONS
            ).all(),
            "all requested grouped draws and permutations are present",
        ),
    ]
    output = pd.DataFrame(checks, columns=["check_id", "passed", "detail"])
    output["new_objective_fe"] = NEW_OBJECTIVE_FE
    if not output["passed"].all():
        failures = output.loc[~output["passed"], "check_id"].tolist()
        raise RuntimeError("Task 17A.1 consistency checks failed: " + ", ".join(failures))
    return output


def _write_report(
    filename: str,
    title: str,
    sections: list[tuple[str, str]],
) -> None:
    preamble = (
        "Task17A.1 是零 FE 的统计正确性与估计对象复核。目的不是获得更漂亮的显著性，"
        "而是判断 Task17A 哪些机制结论在统一统计口径后仍然成立。"
    )
    text = [f"# {title}", "", preamble, ""]
    for heading, body in sections:
        text.extend([f"## {heading}", "", body, ""])
    (OUT_LIGHT / filename).write_text("\n".join(text).rstrip() + "\n", encoding="utf-8")


def _reports(
    states: pd.DataFrame,
    state_identity: pd.DataFrame,
    pair_identity: pd.DataFrame,
    contract_check: pd.DataFrame,
    consistency: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    alignment: pd.DataFrame,
    permutations: pd.DataFrame,
    nn_summary: pd.DataFrame,
    selected: pd.DataFrame,
    local_ava: pd.DataFrame,
    shift: pd.DataFrame,
    ladder: pd.DataFrame,
    collisions: pd.DataFrame,
    comparison: pd.DataFrame,
    robustness: pd.DataFrame,
    resource_usage: pd.DataFrame,
    verdicts: dict[str, object],
) -> None:
    group_semantics = (
        states.groupby(["suite", "cv_group_id"], sort=True)["problem_id"]
        .nunique()
        .groupby(level=0)
        .agg(n_cv_groups="size", min_problems_per_group="min", max_problems_per_group="max")
        .reset_index()
    )
    identity_summary = pd.DataFrame(
        [
            {
                "domain": domain,
                "states": int((state_identity["domain"] == domain).sum()),
                "strata": int(state_identity.loc[state_identity["domain"].eq(domain), "stratum_id"].nunique()),
                "pairs": int((pair_identity["domain"] == domain).sum()),
                "states_per_stratum": 5,
                "pairs_per_stratum": 10,
                "new_objective_fe": NEW_OBJECTIVE_FE,
            }
            for domain in DOMAINS
        ]
    )
    pooled_alignment = alignment.loc[alignment["suite"].eq("pooled")]
    pooled_nn = nn_summary.loc[nn_summary["suite"].eq("pooled")]
    pooled_ava = local_ava.loc[local_ava["suite"].eq("pooled")]
    pooled_shift = shift.loc[shift["suite"].eq("pooled")]
    tie_summary = (
        selected.loc[selected["selected_for_local_ava10"]]
        .groupby(["domain", "representation"], sort=True)[
            "nearest_distance_tie_count"
        ]
        .agg(
            n_strata="size",
            strata_with_nearest_tie=lambda values: int(np.sum(values > 1)),
            maximum_tie_count="max",
        )
        .reset_index()
    )
    comparison_short = comparison.loc[
        comparison["metric"].isin(
            (
                "rho",
                "delta_nn",
                "ava10",
                "natural_post_shift_verdict",
                "representation_ladder_verdict",
                "final_verdict",
            )
        )
        & comparison["suite"].isin(("pooled", "all"))
    ]

    _write_report(
        "17a1_01_zero_fe_and_statistical_contract.md",
        "17a1_01 Zero-FE and Statistical Contract",
        [
            (
                "执行范围",
                "只读取 Task17A 已生成的 states、pairs、representation distances、decision distances、NN 与随机 pair 对照；未导入或调用 objective、optimizer、ELA、特征提取器或学习模型。",
            ),
            (
                "随机与统计参数",
                pd.DataFrame(
                    [
                        {
                            "master_seed": MASTER_SEED,
                            "bootstrap_draws": BOOTSTRAP_DRAWS,
                            "permutations": PERMUTATIONS,
                            "random_neighbor_controls": RANDOM_NEIGHBOR_CONTROLS,
                            "bootstrap_unit": "cv_group_id",
                            "new_objective_fe": NEW_OBJECTIVE_FE,
                        }
                    ]
                ).to_markdown(index=False),
            ),
            ("一致性检查", consistency.to_markdown(index=False)),
        ],
    )
    _write_report(
        "17a1_02_task17a_issue_check.md",
        "17a1_02 Task17A Issue Check",
        [("已确认问题", contract_check.to_markdown(index=False))],
    )
    _write_report(
        "17a1_03_data_identity_and_group_semantics.md",
        "17a1_03 Data Identity and Group Semantics",
        [
            ("身份结果", identity_summary.to_markdown(index=False)),
            ("cv_group_id 语义", group_semantics.to_markdown(index=False)),
            (
                "解释",
                "BBOB 的 cv_group_id 是函数编号，同一函数的三个实例属于同一 grouped dependence 单元；MA-BBOB 的 cv_group_id 是候选函数编号，当前每组一个实例。既有 Task12-15 grouped-OOF 也直接按 cv_group_id 留组。",
            ),
        ],
    )
    _write_report(
        "17a1_04_corrected_within_stratum_alignment.md",
        "17a1_04 Corrected Within-Stratum Alignment",
        [
            (
                "主估计对象",
                "每个十-pair stratum 内分别以 average rank 将 representation distance 与 decision distance 映射到 [0,1]，随后合并正式 strata 计算 Spearman。point 与 grouped bootstrap 均重新计算该统计量。",
            ),
            ("Pooled 结果", pooled_alignment.to_markdown(index=False)),
            ("全套件结果", alignment.to_markdown(index=False)),
        ],
    )
    permutation_summary = permutations.groupby(
        ["domain", "suite", "representation"], sort=True
    )["rho_rank_within_null"].agg(
        null_mean="mean",
        q95=lambda values: values.quantile(0.95),
        q97_5=lambda values: values.quantile(0.975),
    ).reset_index()
    _write_report(
        "17a1_05_corrected_permutation_null.md",
        "17a1_05 Corrected Permutation Null",
        [
            (
                "定义",
                "每次在各 stratum 的五个 state 间置换三维 decision signature，重算十个 decision distances，再执行层内 average-rank normalization；observable ranks 保持不变。",
            ),
            ("结果", permutation_summary.to_markdown(index=False)),
        ],
    )
    _write_report(
        "17a1_06_corrected_nearest_neighbor.md",
        "17a1_06 Corrected Nearest Neighbor",
        [
            (
                "定义",
                "observable NN 仍从同一 stratum 的四个其他 states 中选择；point 为整体 random median 减 observable median，区间在每次 cv_group 重采样后重算同一差值。R_NN=1-observable median/random median。",
            ),
            ("Pooled 结果", pooled_nn.to_markdown(index=False)),
            ("全套件结果", nn_summary.to_markdown(index=False)),
        ],
    )
    _write_report(
        "17a1_07_local_action_value_aliasing.md",
        "17a1_07 Local Action-Value Aliasing",
        [
            (
                "离散定义",
                "Local AVA10 每个十-pair stratum 取 observable distance 最小的一个 pair；Local AVA20 取最小的两个 pairs。不构造 local AVA5。距离并列时仅用 seed_i、seed_j、pair_index 作确定性次序。",
            ),
            ("最近距离并列", tie_summary.to_markdown(index=False)),
            ("Pooled 结果", pooled_ava.to_markdown(index=False)),
            ("全套件结果", local_ava.to_markdown(index=False)),
        ],
    )
    _write_report(
        "17a1_08_natural_post_shift.md",
        "17a1_08 Natural and Post-handoff Shift",
        [
            (
                "Matched resampling",
                "Natural 与 Post 在相同 cv_group_id 上做 matched group resampling；每个 draw 分别重算 rho、local AVA10 与 NN benefit 后取 Post-Natural。",
            ),
            ("Pooled 结果", pooled_shift.to_markdown(index=False)),
            ("全套件结果", shift.to_markdown(index=False)),
            ("分类", str(verdicts["shift_verdict"])),
        ],
    )
    _write_report(
        "17a1_09_representation_ladder.md",
        "17a1_09 Comparative Representation Check",
        [
            (
                "边界",
                "Compact6、Global28、SegmentMatched28、ISSD18 与 Compact+ISSD24 不是严格嵌套维度链；这里只比较其统计结果，不把相邻项描述为加入同一种信息。",
            ),
            ("结果", ladder.to_markdown(index=False)),
            ("分类", str(verdicts["ladder_verdict"])),
        ],
    )
    collision_columns = [
        "domain",
        "suite",
        "example_rank",
        "problem_id",
        "representation",
        "stratum_id",
        "state_i",
        "state_j",
        "representation_distance_l1_mean",
        "decision_distance_linf",
        "A_ND_members_i",
        "A_ND_members_j",
    ]
    _write_report(
        "17a1_10_collision_examples.md",
        "17a1_10 Local Collision Examples",
        [
            (
                "选择规则",
                "只从每个 stratum 的 observable-nearest pair 中按 decision distance 降序选例；每个 domain-suite 最多 10 个、每个 problem 最多一个。",
            ),
            ("例子", collisions[collision_columns].to_markdown(index=False)),
        ],
    )
    _write_report(
        "17a1_11_old_vs_corrected_comparison.md",
        "17a1_11 Old and Corrected Comparison",
        [
            ("关键变化", comparison_short.to_markdown(index=False)),
            (
                "边界",
                "Task17A 原文件与结果均保留；本轮不覆盖旧结论，只把统一估计对象后的结果写入 task17a1 目录。",
            ),
        ],
    )
    _write_report(
        "17a1_12_solver_suite_robustness.md",
        "17a1_12 Solver-Suite Robustness",
        [("BBOB 与 MA-BBOB", robustness.to_markdown(index=False))],
    )
    _write_report(
        "17a1_13_resource_usage.md",
        "17a1_13 Resource Usage",
        [("实测资源", resource_usage.to_markdown(index=False))],
    )
    _write_report(
        "17a1_14_final_verdict.md",
        "17a1_14 Final Verdict",
        [
            ("分类", f"**{verdicts['final_verdict']}**"),
            ("异质性", heterogeneity.to_markdown(index=False)),
            ("科学表述", str(verdicts["scientific_statement"])),
            (
                "执行边界",
                "本轮不支持直接加入 solver internal state，不训练新 selector，不增加 seeds 6-10，不运行 CEC，不执行闭环。新增 objective FE = 0；下一步只能依据 Task17A.1 分类重新设计。",
            ),
            ("机器可读字段", "```json\n" + json.dumps(verdicts, indent=2, ensure_ascii=False) + "\n```"),
        ],
    )

    total_sections = [
        ("Correctness", contract_check.to_markdown(index=False)),
        ("Data Identity and Grouping", identity_summary.to_markdown(index=False) + "\n\n" + group_semantics.to_markdown(index=False)),
        ("Decision Heterogeneity", heterogeneity.to_markdown(index=False)),
        ("Corrected Alignment", pooled_alignment.to_markdown(index=False)),
        ("Nearest Neighbor", pooled_nn.to_markdown(index=False)),
        ("Local Aliasing", pooled_ava.to_markdown(index=False)),
        ("Natural and Post-handoff", pooled_shift.to_markdown(index=False) + "\n\n分类：**" + str(verdicts["shift_verdict"]) + "**。"),
        ("Representation Comparison", ladder.to_markdown(index=False) + "\n\n分类：**" + str(verdicts["ladder_verdict"]) + "**。"),
        ("Old and Corrected", comparison_short.to_markdown(index=False)),
        ("Final", f"**{verdicts['final_verdict']}**\n\n{verdicts['scientific_statement']}"),
        ("Scope Boundary", "不直接加入 solver internal state；不训练新 selector；不增加 seeds 6-10；不运行 CEC；不执行闭环。new objective FE = 0。下一步必须以 Task17A.1 分类为依据重新设计。"),
    ]
    _write_report(
        "Decision-before-Feature_Task17A1_CorrectedDecisionGeometry与LocalAliasing.md",
        "Decision-before-Feature Task17A.1 Corrected Decision Geometry and Local Aliasing",
        total_sections,
    )


def run_analysis() -> dict[str, object]:
    wall_start = perf_counter()
    cpu_start = process_time()
    OUT_LIGHT.mkdir(parents=True, exist_ok=True)
    OUT_HEAVY.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    inputs = _read_inputs()
    states, pair_identity, merged, state_identity = _validate_and_merge(inputs)
    ranked = _rank_pair_distances(merged)
    permutations = _run_permutations(states, pair_identity, ranked)
    alignment, alignment_bootstrap = _corrected_alignment(ranked, permutations)
    nn_summary = _corrected_nn(inputs["nn"])
    selected = _local_pair_selection(ranked)
    random_pairs = inputs["random_pairs"].merge(
        pair_identity[["domain", "pair_index", "suite"]],
        on=["domain", "pair_index"],
        how="left",
        validate="many_to_one",
    )
    if random_pairs["suite"].isna().any():
        raise RuntimeError("random-pair suite join is incomplete")
    local_ava = _local_ava(ranked, selected, random_pairs)
    heterogeneity = _heterogeneity(inputs["decision"])
    shift = _natural_post_shift(ranked, selected, inputs["nn"])
    ladder = _representation_ladder(
        ranked,
        selected,
        inputs["nn"],
        alignment,
        nn_summary,
        local_ava,
    )
    collisions = _collision_examples(selected, states)
    verdicts = _derive_verdicts(
        heterogeneity,
        alignment,
        local_ava,
        shift,
        ladder,
    )
    comparison = _old_vs_corrected(
        inputs,
        alignment,
        nn_summary,
        local_ava,
        shift,
        verdicts,
    )
    robustness = _suite_robustness(
        heterogeneity,
        alignment,
        nn_summary,
        local_ava,
    )
    contract_check = _statistical_contract_check()
    figures = _make_figures(ranked, alignment, local_ava, nn_summary)
    usage = _resource_usage(wall_start, cpu_start, figures)

    products = {
        "task17a1_statistical_contract_check": contract_check,
        "task17a1_state_identity": state_identity,
        "task17a1_pair_identity": pair_identity,
        "task17a1_stratum_rank_distances": ranked,
        "task17a1_corrected_alignment": alignment,
        "task17a1_corrected_alignment_bootstrap": alignment_bootstrap,
        "task17a1_corrected_permutations": permutations,
        "task17a1_corrected_nn": nn_summary,
        "task17a1_local_nearest_pairs": selected,
        "task17a1_local_ava": local_ava,
        "task17a1_heterogeneity": heterogeneity,
        "task17a1_natural_post_shift": shift,
        "task17a1_representation_ladder": ladder,
        "task17a1_collision_examples": collisions,
        "task17a1_old_vs_corrected": comparison,
        "task17a1_solver_suite_robustness": robustness,
        "task17a1_resource_usage": usage,
    }
    consistency = _consistency_checks(
        states,
        pair_identity,
        selected,
        alignment,
        alignment_bootstrap,
        permutations,
        products,
    )
    products["task17a1_consistency_checks"] = consistency

    if NEW_OBJECTIVE_FE != 0:
        raise RuntimeError("TASK17A1_INVALID_NEW_FE_ATTEMPT")
    for name, table in products.items():
        if "new_objective_fe" not in table.columns or not table["new_objective_fe"].eq(0).all():
            raise RuntimeError(f"{name} does not satisfy the zero-FE contract")
        table.to_parquet(OUT_HEAVY / f"{name}.parquet", index=False)

    (OUT_LIGHT / "task17a1_verdict.json").write_text(
        json.dumps(verdicts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _reports(
        states,
        state_identity,
        pair_identity,
        contract_check,
        consistency,
        heterogeneity,
        alignment,
        permutations,
        nn_summary,
        selected,
        local_ava,
        shift,
        ladder,
        collisions,
        comparison,
        robustness,
        usage,
        verdicts,
    )
    return {
        "final_verdict": verdicts["final_verdict"],
        "shift_verdict": verdicts["shift_verdict"],
        "ladder_verdict": verdicts["ladder_verdict"],
        "new_objective_fe": NEW_OBJECTIVE_FE,
        "all_consistency_checks_passed": bool(consistency["passed"].all()),
        "output_directory": str(OUT_LIGHT.relative_to(ROOT)),
        "heavy_output_directory": str(OUT_HEAVY.relative_to(ROOT)),
    }


def main() -> None:
    print(json.dumps(run_analysis(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
