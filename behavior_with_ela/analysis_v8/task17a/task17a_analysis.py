"""Task 17A: action-value aliasing and decision-state sufficiency analysis.

This module performs a zero-new-objective-evaluation comparison between
observable representation geometry and the geometry of already measured
alternate-action outcomes.  Natural and post-handoff states remain separate
throughout.  No selector, action branch, optimizer, or feature is created.
"""
from __future__ import annotations

import itertools
import math
import os
import resource
import shutil
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[3]
OUT_LIGHT = ROOT / "behavior_with_ela/analysis_v8/task17a"
OUT_HEAVY = ROOT / "behavior_with_ela/results/analysis_v8/task17a"
FIGURES = OUT_LIGHT / "figures"

NATURAL_DATA = (
    ROOT / "behavior_with_ela/results/analysis_v5/task13/behavior_action_dataset_task13.parquet"
)
NATURAL_NOISE = (
    ROOT / "behavior_with_ela/analysis_v5/task13_1/fold_local_action_noise_scale.parquet"
)
POST_DATA = (
    ROOT
    / "behavior_with_ela/results/analysis_v6/task14b_1/task14b1_corrected_dataset_matched.parquet"
)
POST_NOISE = (
    ROOT / "behavior_with_ela/analysis_v6/task14b/post_handoff_fold_local_noise_scale.parquet"
)
POST_ISSD = (
    ROOT / "behavior_with_ela/results/analysis_v7/task15a/post_handoff_issd_task15a.parquet"
)
SCREENED_CONCEPTS = (
    ROOT / "behavior_with_ela/results/analysis_v7/task15a/screened_behavior_concepts.parquet"
)

ACTIONS = ("shade", "lshade", "cso")
ACTION_PAIRS = (("shade", "lshade"), ("shade", "cso"), ("lshade", "cso"))
LOSS_COLUMNS = [f"loss_{action}" for action in ACTIONS]
NORMALIZED_MARGIN_COLUMNS = [f"normalized_margin_{a}_{b}" for a, b in ACTION_PAIRS]

BOOTSTRAP_DRAWS = 5000
N_PERMUTATIONS = 100
N_RANDOM_CONTROLS = 100
MASTER_SEED = 2026083101
BOOTSTRAP_STREAM = 2026083102
PERMUTATION_STREAM = 2026083103
RANDOM_NEIGHBOR_STREAM = 2026083104
RANDOM_PAIR_STREAM = 2026083105
ROBUST_EPSILON = 1e-12
STANDARDIZED_CLIP = 5.0

DOMAIN_CODES = {"natural": 1, "post_handoff": 2}
SUITE_CODES = {"bbob": 1, "mabbob": 2, "pooled": 3}
DECISION_METRIC_CODES = {"linf_primary": 1, "l2_sensitivity": 2}
REPRESENTATION_CODES = {
    "compact6": 1,
    "global28": 2,
    "segment_old28": 3,
    "segment_matched28": 4,
    "issd18": 5,
    "compact_issd24": 6,
}
SCOPES = ("bbob", "mabbob", "pooled")

LADDER = ("compact6", "global28", "segment_matched28", "issd18", "compact_issd24")
FORMAL_POST_REPRESENTATIONS = (
    "compact6",
    "global28",
    "segment_matched28",
    "issd18",
    "compact_issd24",
)


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


def _quantile_interval(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def _group_bootstrap_indices(
    group_values: np.ndarray,
    draws: int,
    stream_codes: tuple[int, ...],
):
    groups, group_codes = np.unique(np.asarray(group_values, dtype=object), return_inverse=True)
    rng = np.random.default_rng(_seed_sequence(BOOTSTRAP_STREAM, *stream_codes, len(groups)))
    row_index = np.arange(len(group_codes), dtype=int)
    for _ in range(draws):
        sampled = rng.integers(0, len(groups), size=len(groups))
        multiplicity = np.bincount(sampled, minlength=len(groups))
        yield np.repeat(row_index, multiplicity[group_codes])


def _bootstrap_spearman(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    stream_codes: tuple[int, ...],
) -> tuple[float, float, float]:
    x = frame[x_column].to_numpy(dtype=float)
    y = frame[y_column].to_numpy(dtype=float)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw, indices in enumerate(
        _group_bootstrap_indices(
            frame["bootstrap_group" if "bootstrap_group" in frame.columns else "cv_group_id"].to_numpy(),
            BOOTSTRAP_DRAWS,
            stream_codes,
        )
    ):
        draws[draw] = _spearman(x[indices], y[indices])
    low, high = _quantile_interval(draws)
    return _spearman(x, y), low, high


def _bootstrap_summary_statistic(
    frame: pd.DataFrame,
    statistic,
    stream_codes: tuple[int, ...],
) -> tuple[float, float, float]:
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw, indices in enumerate(
        _group_bootstrap_indices(
            frame["cv_group_id"].to_numpy(), BOOTSTRAP_DRAWS, stream_codes
        )
    ):
        draws[draw] = float(statistic(frame.iloc[indices]))
    low, high = _quantile_interval(draws)
    return float(statistic(frame)), low, high


def _selector_suffixes() -> list[str]:
    columns = list(SELECTOR_BEHAVIOR_FEATURE_COLUMNS)
    if len(columns) != 28 or not all(column.startswith("bf_") for column in columns):
        raise RuntimeError("the formal 28-column behavior contract is unavailable")
    return [column[3:] for column in columns]


def _issd_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if column.startswith("issd_") and column.endswith(("_q25", "_q50", "_q75"))
    ]
    columns = sorted(columns)
    if len(columns) != 18:
        raise RuntimeError(f"expected 18 ISSD columns, got {len(columns)}")
    return columns


def _load_domain_frames() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    required = [NATURAL_DATA, NATURAL_NOISE, POST_DATA, POST_NOISE, POST_ISSD, SCREENED_CONCEPTS]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing Task 17A inputs: " + ", ".join(map(str, missing)))

    natural = pd.read_parquet(NATURAL_DATA)
    post = pd.read_parquet(POST_DATA)
    issd = pd.read_parquet(POST_ISSD)
    concepts = pd.read_parquet(SCREENED_CONCEPTS)

    if len(natural) != 1890 or not natural["state_id"].is_unique:
        raise RuntimeError("natural-state row count or state identity is inconsistent")
    if len(post) != 3780 or not post["state_id"].is_unique:
        raise RuntimeError("post-handoff row count or state identity is inconsistent")
    if len(issd) != 3780 or not issd["state_id"].is_unique:
        raise RuntimeError("post-handoff ISSD row count or state identity is inconsistent")

    issd_columns = _issd_columns(issd)
    post = post.merge(
        issd[["state_id", *issd_columns]], on="state_id", how="left", validate="one_to_one"
    )
    if post[issd_columns].isna().any().any():
        raise RuntimeError("post-handoff ISSD join is incomplete")

    compact_bf = concepts.loc[
        concepts["screened_core"].astype(bool), "primary_aggregate_representative"
    ].astype(str).tolist()
    if len(compact_bf) != 6 or len(set(compact_bf)) != 6:
        raise RuntimeError("the six-concept compact representation is inconsistent")
    suffixes = _selector_suffixes()
    global_bf = [f"bf_{suffix}" for suffix in suffixes]
    global_bg = [f"bg_{suffix}" for suffix in suffixes]
    segment_old = [f"bs_old_{suffix}" for suffix in suffixes]
    segment_matched = [f"bs_matched_{suffix}" for suffix in suffixes]
    compact_bg = [f"bg_{column[3:]}" for column in compact_bf]

    representation_rows: list[dict] = []
    specifications = {
        "natural": {
            "compact6": compact_bf,
            "global28": global_bf,
        },
        "post_handoff": {
            "compact6": compact_bg,
            "global28": global_bg,
            "segment_old28": segment_old,
            "segment_matched28": segment_matched,
            "issd18": issd_columns,
            "compact_issd24": [*compact_bg, *issd_columns],
        },
    }
    for domain, specs in specifications.items():
        data = natural if domain == "natural" else post
        for representation, columns in specs.items():
            missing_columns = sorted(set(columns) - set(data.columns))
            if missing_columns:
                raise RuntimeError(
                    f"{domain}/{representation} is missing columns: {missing_columns}"
                )
            for order, column in enumerate(columns):
                representation_rows.append(
                    {
                        "domain": domain,
                        "representation": representation,
                        "feature_order": order,
                        "feature_name": column,
                        "n_features": len(columns),
                        "formal_ladder": representation in LADDER,
                    }
                )

    natural = natural.copy()
    natural["domain"] = "natural"
    natural["source_algorithm"] = natural["current_algorithm"]
    natural["source_FE"] = natural["FE"].astype(int)
    natural["global_FE"] = natural["FE"].astype(int)
    natural["route"] = "natural_" + natural["current_algorithm"].astype(str)

    post = post.copy()
    post["domain"] = "post_handoff"
    post["global_FE"] = post["global_FE"].astype(int)
    post["source_FE"] = post["source_FE"].astype(int)
    post["route"] = post["source_algorithm"].astype(str) + "->" + post["current_algorithm"].astype(str)

    join_rows = [
        {
            "domain": "natural",
            "source_rows": 1890,
            "joined_rows": len(natural),
            "unique_states": natural["state_id"].nunique(),
            "exact_one_to_one": True,
            "new_objective_fe": 0,
            "natural_issd_status": "skipped_no_zero_fe_artifact",
        },
        {
            "domain": "post_handoff",
            "source_rows": 3780,
            "joined_rows": len(post),
            "unique_states": post["state_id"].nunique(),
            "exact_one_to_one": True,
            "new_objective_fe": 0,
            "natural_issd_status": "not_applicable",
        },
    ]
    return {"natural": natural, "post_handoff": post}, pd.DataFrame(representation_rows), pd.DataFrame(join_rows)


def _add_decision_signatures(
    frame: pd.DataFrame, noise_path: Path, domain: str
) -> pd.DataFrame:
    noise = pd.read_parquet(noise_path)
    if set(noise.columns) < {"held_out_group", "solver", "delta_95_fold_local"}:
        raise RuntimeError(f"invalid fold-local noise table: {noise_path}")
    pivot = noise.pivot(index="held_out_group", columns="solver", values="delta_95_fold_local")
    if set(ACTIONS) - set(pivot.columns):
        raise RuntimeError(f"noise scales do not cover all actions: {noise_path}")

    output = frame.copy()
    for action in ACTIONS:
        output[f"noise_scale_{action}"] = output["cv_group_id"].map(pivot[action])
    noise_columns = [f"noise_scale_{action}" for action in ACTIONS]
    if output[noise_columns].isna().any().any() or (output[noise_columns] <= 0).any().any():
        raise RuntimeError(f"{domain} has missing or nonpositive noise scales")

    losses = output[LOSS_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(losses).all():
        raise RuntimeError(f"{domain} action losses are not finite")
    centered = losses - losses.mean(axis=1, keepdims=True)
    for index, action in enumerate(ACTIONS):
        output[f"centered_loss_{action}"] = centered[:, index]

    for action_a, action_b in ACTION_PAIRS:
        raw_margin = output[f"loss_{action_b}"] - output[f"loss_{action_a}"]
        scale = np.maximum(
            output[f"noise_scale_{action_a}"].to_numpy(dtype=float),
            output[f"noise_scale_{action_b}"].to_numpy(dtype=float),
        )
        output[f"pair_scale_{action_a}_{action_b}"] = scale
        output[f"margin_{action_a}_{action_b}"] = raw_margin
        output[f"normalized_margin_{action_a}_{action_b}"] = raw_margin / scale

    if output["A_ND_members"].isna().any():
        raise RuntimeError(f"{domain} practical action sets are incomplete")
    output["A_ND_members"] = output["A_ND_members"].astype(str).map(
        lambda value: "|".join(action for action in ACTIONS if action in value.split("|") and action)
    )
    output["A_ND_size"] = output["A_ND_members"].map(
        lambda value: len([item for item in value.split("|") if item])
    )
    if (output["A_ND_size"] == 0).any():
        raise RuntimeError(f"{domain} contains an empty practical action set")

    if domain == "natural":
        stratum_columns = ["problem_id", "current_algorithm", "source_FE"]
    else:
        stratum_columns = ["problem_id", "route", "source_FE"]
    output["stratum_id"] = output[stratum_columns].astype(str).agg("|".join, axis=1)
    output["bootstrap_group"] = output["cv_group_id"].astype(str) + "|" + output["problem_id"].astype(str)
    group_sizes = output.groupby("stratum_id", sort=False).size()
    seed_sets = output.groupby("stratum_id", sort=False)["seed"].agg(
        lambda values: tuple(sorted(map(int, values)))
    )
    if not group_sizes.eq(5).all() or not seed_sets.map(lambda value: value == (1, 2, 3, 4, 5)).all():
        raise RuntimeError(f"{domain} strata do not contain exactly seeds 1-5")
    return output.sort_values(["stratum_id", "seed"]).reset_index(drop=True)


def _scale_representations(
    frame: pd.DataFrame, inventory: pd.DataFrame, domain: str
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    matrices: dict[str, np.ndarray] = {}
    parameter_rows: list[dict] = []
    domain_inventory = inventory.loc[inventory["domain"].eq(domain)]
    for representation, part in domain_inventory.groupby("representation", sort=False):
        part = part.sort_values("feature_order")
        columns = part["feature_name"].tolist()
        raw = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(raw).all():
            raise RuntimeError(f"{domain}/{representation} contains nonfinite values")
        median = np.median(raw, axis=0)
        q25 = np.quantile(raw, 0.25, axis=0)
        q75 = np.quantile(raw, 0.75, axis=0)
        iqr = q75 - q25
        scaled = np.clip(
            (raw - median[None, :]) / (iqr[None, :] + ROBUST_EPSILON),
            -STANDARDIZED_CLIP,
            STANDARDIZED_CLIP,
        )
        if not np.isfinite(scaled).all():
            raise RuntimeError(f"{domain}/{representation} scaling failed")
        matrices[representation] = scaled
        for index, column in enumerate(columns):
            parameter_rows.append(
                {
                    "domain": domain,
                    "representation": representation,
                    "feature_name": column,
                    "median": median[index],
                    "q25": q25[index],
                    "q75": q75[index],
                    "iqr": iqr[index],
                    "epsilon": ROBUST_EPSILON,
                    "clip_lower": -STANDARDIZED_CLIP,
                    "clip_upper": STANDARDIZED_CLIP,
                    "zero_iqr": bool(iqr[index] == 0),
                }
            )
    return matrices, pd.DataFrame(parameter_rows)


def _build_pairs(signatures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict] = []
    decision_rows: list[dict] = []
    normalized = signatures[NORMALIZED_MARGIN_COLUMNS].to_numpy(dtype=float)
    state_positions = pd.Series(np.arange(len(signatures), dtype=int), index=signatures["state_id"])

    pair_index = 0
    for stratum_id, part in signatures.groupby("stratum_id", sort=True):
        part = part.sort_values("seed")
        positions = state_positions.loc[part["state_id"]].to_numpy(dtype=int)
        rows = part.reset_index(drop=True)
        for local_i, local_j in itertools.combinations(range(5), 2):
            row_i = rows.iloc[local_i]
            row_j = rows.iloc[local_j]
            pos_i = int(positions[local_i])
            pos_j = int(positions[local_j])
            set_i = set(filter(None, str(row_i["A_ND_members"]).split("|")))
            set_j = set(filter(None, str(row_j["A_ND_members"]).split("|")))
            union = set_i | set_j
            jaccard_distance = 1.0 - len(set_i & set_j) / len(union)
            metadata = {
                "domain": row_i["domain"],
                "suite": row_i["suite"],
                "pair_index": pair_index,
                "stratum_id": stratum_id,
                "cv_group_id": row_i["cv_group_id"],
                "bootstrap_group": row_i["bootstrap_group"],
                "problem_id": row_i["problem_id"],
                "route": row_i["route"],
                "current_algorithm": row_i["current_algorithm"],
                "source_FE": int(row_i["source_FE"]),
                "state_i": row_i["state_id"],
                "state_j": row_j["state_id"],
                "seed_i": int(row_i["seed"]),
                "seed_j": int(row_j["seed"]),
                "state_position_i": pos_i,
                "state_position_j": pos_j,
            }
            pair_rows.append(metadata)
            difference = normalized[pos_i] - normalized[pos_j]
            decision_rows.append(
                {
                    **metadata,
                    "decision_distance_linf": float(np.max(np.abs(difference))),
                    "decision_distance_l2": float(np.linalg.norm(difference)),
                    "action_set_i": row_i["A_ND_members"],
                    "action_set_j": row_j["A_ND_members"],
                    "action_set_jaccard_distance": float(jaccard_distance),
                    "action_set_changed": bool(set_i != set_j),
                }
            )
            pair_index += 1
    return pd.DataFrame(pair_rows), pd.DataFrame(decision_rows)


def _representation_distances(
    pairs: pd.DataFrame, matrices: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    index_i = pairs["state_position_i"].to_numpy(dtype=int)
    index_j = pairs["state_position_j"].to_numpy(dtype=int)
    metadata_columns = [
        "domain",
        "suite",
        "pair_index",
        "stratum_id",
        "cv_group_id",
        "bootstrap_group",
        "problem_id",
        "route",
        "current_algorithm",
        "source_FE",
        "state_i",
        "state_j",
        "seed_i",
        "seed_j",
        "state_position_i",
        "state_position_j",
    ]
    for representation, matrix in matrices.items():
        difference = matrix[index_i] - matrix[index_j]
        part = pairs[metadata_columns].copy()
        part["representation"] = representation
        part["n_features"] = matrix.shape[1]
        part["representation_distance_l1_mean"] = np.mean(np.abs(difference), axis=1)
        part["representation_distance_l2_scaled"] = (
            np.linalg.norm(difference, axis=1) / math.sqrt(matrix.shape[1])
        )
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def build_data_products():
    domain_frames, representation_inventory, join_summary = _load_domain_frames()
    all_signatures = []
    all_pairs = []
    all_decision = []
    all_representation = []
    all_preprocessing = []
    matrices_by_domain: dict[str, dict[str, np.ndarray]] = {}

    for domain, frame in domain_frames.items():
        noise_path = NATURAL_NOISE if domain == "natural" else POST_NOISE
        signatures = _add_decision_signatures(frame, noise_path, domain)
        matrices, preprocessing = _scale_representations(
            signatures, representation_inventory, domain
        )
        pairs, decision = _build_pairs(signatures)
        representation = _representation_distances(pairs, matrices)
        expected_pairs = 3780 if domain == "natural" else 7560
        if len(pairs) != expected_pairs:
            raise RuntimeError(f"{domain} produced {len(pairs)} pairs, expected {expected_pairs}")
        matrices_by_domain[domain] = matrices
        all_signatures.append(signatures)
        all_pairs.append(pairs)
        all_decision.append(decision)
        all_representation.append(representation)
        all_preprocessing.append(preprocessing)

    signatures = pd.concat(all_signatures, ignore_index=True)
    pairs = pd.concat(all_pairs, ignore_index=True)
    decision = pd.concat(all_decision, ignore_index=True)
    representation = pd.concat(all_representation, ignore_index=True)
    preprocessing = pd.concat(all_preprocessing, ignore_index=True)
    return (
        domain_frames,
        signatures,
        pairs,
        decision,
        representation,
        representation_inventory,
        preprocessing,
        join_summary,
        matrices_by_domain,
    )


def _merge_pair_products(
    pairs: pd.DataFrame,
    decision: pd.DataFrame,
    representation: pd.DataFrame,
) -> pd.DataFrame:
    decision_columns = [
        "domain",
        "pair_index",
        "decision_distance_linf",
        "decision_distance_l2",
        "action_set_jaccard_distance",
        "action_set_changed",
    ]
    merged = representation.merge(
        decision[decision_columns], on=["domain", "pair_index"], validate="many_to_one"
    )
    return merged


def _grouped_bootstrap_from_group_values(
    group_values: pd.Series, stream_codes: tuple[int, ...]
) -> tuple[float, float, float]:
    values = pd.to_numeric(group_values, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(_seed_sequence(BOOTSTRAP_STREAM, *stream_codes, len(values)))
    draws = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
    return float(np.mean(values)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _group_statistic(
    frame: pd.DataFrame,
    statistic,
    group_column: str | None = None,
) -> pd.Series:
    if group_column is None:
        group_column = "bootstrap_group" if "bootstrap_group" in frame.columns else "cv_group_id"
    rows = []
    for group, part in frame.groupby(group_column, sort=True):
        value = float(statistic(part))
        if np.isfinite(value):
            rows.append((group, value))
    return pd.Series(dict(rows), dtype=float)


def _metric_bootstrap(
    frame: pd.DataFrame,
    statistic,
    stream_codes: tuple[int, ...],
    group_column: str | None = None,
) -> tuple[float, float, float]:
    group_values = _group_statistic(frame, statistic, group_column=group_column)
    return _grouped_bootstrap_from_group_values(group_values, stream_codes)


def _distance_alignment(
    merged: pd.DataFrame,
    permutation: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for domain in sorted(merged["domain"].unique()):
        domain_merged = merged.loc[merged["domain"].eq(domain)]
        for representation in sorted(domain_merged["representation"].unique()):
            rep = domain_merged.loc[domain_merged["representation"].eq(representation)]
            for suite in SCOPES:
                subset = _scope(rep, suite)
                point = _spearman(
                    subset["representation_distance_l1_mean"].to_numpy(),
                    subset["decision_distance_linf"].to_numpy(),
                )
                point_l2 = _spearman(
                    subset["representation_distance_l2_scaled"].to_numpy(),
                    subset["decision_distance_linf"].to_numpy(),
                )
                group_rho = _group_statistic(
                    subset,
                    lambda part: _spearman(
                        part["representation_distance_l1_mean"].to_numpy(),
                        part["decision_distance_linf"].to_numpy(),
                    ),
                )
                mean, low, high = _grouped_bootstrap_from_group_values(
                    group_rho, (DOMAIN_CODES[subset["domain"].iloc[0]], REPRESENTATION_CODES[representation], SUITE_CODES[suite])
                )
                group_rho_l2 = _group_statistic(
                    subset,
                    lambda part: _spearman(
                        part["representation_distance_l2_scaled"].to_numpy(),
                        part["decision_distance_linf"].to_numpy(),
                    ),
                )
                mean_l2, low_l2, high_l2 = _grouped_bootstrap_from_group_values(
                    group_rho_l2,
                    (DOMAIN_CODES[subset["domain"].iloc[0]], REPRESENTATION_CODES[representation], SUITE_CODES[suite], 2),
                )
                null = permutation.loc[
                    permutation["domain"].eq(domain)
                    & permutation["representation"].eq(representation)
                    & permutation["suite"].eq(suite)
                ]["null_rho"].to_numpy(dtype=float)
                q975 = float(np.quantile(null, 0.975)) if len(null) else float("nan")
                empirical_p = (
                    float((1 + np.sum(null >= point)) / (1 + len(null))) if len(null) else float("nan")
                )
                lower_ci = low
                verdict = "DA3 UNALIGNED"
                if point > 0 and lower_ci > 0 and (not np.isfinite(q975) or point > q975):
                    verdict = "DA1 ALIGNED"
                elif point > 0:
                    verdict = "DA2 WEAK"
                rows.append(
                    {
                        "domain": subset["domain"].iloc[0],
                        "suite": suite,
                        "representation": representation,
                        "n_pairs": int(len(subset)),
                        "rho_observed_pooled": point,
                        "rho_group_mean": mean,
                        "rho_ci_low": low,
                        "rho_ci_high": high,
                        "rho_l2_observed": point_l2,
                        "rho_l2_group_mean": mean_l2,
                        "rho_l2_ci_low": low_l2,
                        "rho_l2_ci_high": high_l2,
                        "permutation_null_mean": float(np.mean(null)) if len(null) else np.nan,
                        "permutation_q97_5": q975,
                        "empirical_p": empirical_p,
                        "alignment_verdict": verdict,
                        "bootstrap_draws": BOOTSTRAP_DRAWS,
                        "bootstrap_unit": "cv_group_id/problem_id",
                        "permutation_repetitions": N_PERMUTATIONS,
                    }
                )
    return pd.DataFrame(rows)


def _stratum_permutation_decision_distances(
    signatures: pd.DataFrame,
    pairs: pd.DataFrame,
    repeat: int,
) -> np.ndarray:
    normalized = signatures[NORMALIZED_MARGIN_COLUMNS].to_numpy(dtype=float)
    permuted = normalized.copy()
    rng = np.random.default_rng(
        _seed_sequence(PERMUTATION_STREAM, DOMAIN_CODES[signatures["domain"].iloc[0]], repeat)
    )
    for _, part in signatures.groupby("stratum_id", sort=True):
        positions = part.index.to_numpy(dtype=int)
        permuted[positions] = normalized[positions[rng.permutation(len(positions))]]
    positions_i = pairs["state_position_i"].to_numpy(dtype=int)
    positions_j = pairs["state_position_j"].to_numpy(dtype=int)
    return np.max(np.abs(permuted[positions_i] - permuted[positions_j]), axis=1)


def _run_alignment_permutations(
    signatures: pd.DataFrame,
    pairs: pd.DataFrame,
    representation: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for repeat in range(N_PERMUTATIONS):
        decision_distance = _stratum_permutation_decision_distances(signatures, pairs, repeat)
        for rep_name in sorted(representation["representation"].unique()):
            rep = representation.loc[representation["representation"].eq(rep_name)].copy()
            rep["permuted_decision_distance"] = decision_distance
            for suite in SCOPES:
                subset = _scope(rep, suite)
                rows.append(
                    {
                        "domain": signatures["domain"].iloc[0],
                        "suite": suite,
                        "representation": rep_name,
                        "repeat": repeat,
                        "null_rho": _spearman(
                            subset["representation_distance_l1_mean"].to_numpy(),
                            subset["permuted_decision_distance"].to_numpy(),
                        ),
                        "permutation_unit": "stratum_id",
                        "decision_signature_shuffle": "within_stratum_seed_permutation",
                    }
                )
    return pd.DataFrame(rows)


def _nearest_neighbor_products(
    signatures: pd.DataFrame,
    pairs: pd.DataFrame,
    representation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision_lookup = {
        (row.domain, int(row.state_position_i), int(row.state_position_j)): float(row.decision_distance_linf)
        for row in pairs.itertuples(index=False)
    }
    decision_lookup.update(
        {
            (row.domain, int(row.state_position_j), int(row.state_position_i)): float(row.decision_distance_linf)
            for row in pairs.itertuples(index=False)
        }
    )
    rows = []
    summary_rows = []
    for rep_name in sorted(representation["representation"].unique()):
        rep = representation.loc[representation["representation"].eq(rep_name)]
        for domain in sorted(rep["domain"].unique()):
            domain_rep = rep.loc[rep["domain"].eq(domain)]
            signature_by_position = signatures.loc[signatures["domain"].eq(domain)].reset_index(drop=True)
            matrix_by_position = {}
            for row in domain_rep.itertuples(index=False):
                matrix_by_position[int(row.state_position_i)] = None
            # Distances are pair-level; selecting the minimum from the row list
            # avoids introducing any learned metric or cross-stratum comparison.
            for stratum_id, part in signature_by_position.groupby("stratum_id", sort=True):
                positions = part.index.to_numpy(dtype=int)
                for position in positions:
                    candidates = positions[positions != position]
                    distance_rows = domain_rep.loc[
                        domain_rep["state_position_i"].eq(position) & domain_rep["state_position_j"].isin(candidates)
                    ]
                    reverse_rows = domain_rep.loc[
                        domain_rep["state_position_j"].eq(position) & domain_rep["state_position_i"].isin(candidates)
                    ]
                    candidate_distances = pd.concat(
                        [
                            distance_rows[["state_position_j", "representation_distance_l1_mean"]].rename(
                                columns={"state_position_j": "neighbor_position"}
                            ),
                            reverse_rows[["state_position_i", "representation_distance_l1_mean"]].rename(
                                columns={"state_position_i": "neighbor_position"}
                            ),
                        ],
                        ignore_index=True,
                    ).sort_values(["representation_distance_l1_mean", "neighbor_position"])
                    if candidate_distances.empty:
                        continue
                    neighbor = int(candidate_distances.iloc[0]["neighbor_position"])
                    observable_distance = float(candidate_distances.iloc[0]["representation_distance_l1_mean"])
                    decision_distance = decision_lookup[(domain, int(position), neighbor)]
                    state_row = signature_by_position.iloc[position]
                    neighbor_row = signature_by_position.iloc[neighbor]
                    set_equal = bool(state_row["A_ND_members"] == neighbor_row["A_ND_members"])
                    set_i = set(filter(None, str(state_row["A_ND_members"]).split("|")))
                    set_j = set(filter(None, str(neighbor_row["A_ND_members"]).split("|")))
                    jaccard = len(set_i & set_j) / len(set_i | set_j)
                    rng = np.random.default_rng(
                        _seed_sequence(
                            RANDOM_NEIGHBOR_STREAM,
                            DOMAIN_CODES[domain],
                            REPRESENTATION_CODES[rep_name],
                            int(position),
                        )
                    )
                    random_distances = []
                    random_set_matches = []
                    random_jaccards = []
                    for _ in range(N_RANDOM_CONTROLS):
                        random_neighbor = int(rng.choice(candidates))
                        random_distances.append(decision_lookup[(domain, int(position), random_neighbor)])
                        random_row = signature_by_position.iloc[random_neighbor]
                        random_set = set(filter(None, str(random_row["A_ND_members"]).split("|")))
                        random_set_matches.append(bool(set_i == random_set))
                        random_jaccards.append(len(set_i & random_set) / len(set_i | random_set))
                    rows.append(
                        {
                            "domain": domain,
                            "suite": state_row["suite"],
                            "representation": rep_name,
                            "state_id": state_row["state_id"],
                            "neighbor_state_id": neighbor_row["state_id"],
                            "stratum_id": stratum_id,
                            "cv_group_id": state_row["cv_group_id"],
                            "bootstrap_group": state_row["bootstrap_group"],
                            "seed": int(state_row["seed"]),
                            "neighbor_seed": int(neighbor_row["seed"]),
                            "problem_id": state_row["problem_id"],
                            "representation_distance": observable_distance,
                            "decision_distance_observable_nn": decision_distance,
                            "decision_distance_random_median": float(np.median(random_distances)),
                            "random_distance_mean": float(np.mean(random_distances)),
                            "delta_nn_state": float(np.median(random_distances) - decision_distance),
                            "practical_set_equal": set_equal,
                            "practical_jaccard_similarity": float(jaccard),
                            "random_practical_set_consistency": float(np.mean(random_set_matches)),
                            "random_practical_jaccard_similarity": float(np.mean(random_jaccards)),
                            "random_control_repetitions": N_RANDOM_CONTROLS,
                        }
                    )
            state_part = pd.DataFrame([row for row in rows if row["domain"] == domain and row["representation"] == rep_name])
            for suite in SCOPES:
                subset = _scope(state_part, suite)
                if subset.empty:
                    continue
                nn_delta = _metric_bootstrap(
                    subset,
                    lambda part: float(np.median(part["decision_distance_random_median"]) - np.median(part["decision_distance_observable_nn"])),
                    (DOMAIN_CODES[domain], REPRESENTATION_CODES[rep_name], SUITE_CODES[suite], 21),
                )
                summary_rows.append(
                    {
                        "domain": domain,
                        "suite": suite,
                        "representation": rep_name,
                        "n_states": len(subset),
                        "observable_nn_median_decision_distance": float(np.median(subset["decision_distance_observable_nn"])),
                        "random_nn_median_decision_distance": float(np.median(subset["decision_distance_random_median"])),
                        "delta_nn": float(np.median(subset["decision_distance_random_median"]) - np.median(subset["decision_distance_observable_nn"])),
                        "delta_nn_group_mean": nn_delta[0],
                        "delta_nn_ci_low": nn_delta[1],
                        "delta_nn_ci_high": nn_delta[2],
                        "observable_practical_consistency": float(np.mean(subset["practical_set_equal"])),
                        "random_practical_consistency": float(np.mean(subset["random_practical_set_consistency"])),
                        "observable_jaccard_similarity": float(np.mean(subset["practical_jaccard_similarity"])),
                        "random_jaccard_similarity": float(np.mean(subset["random_practical_jaccard_similarity"])),
                        "bootstrap_draws": BOOTSTRAP_DRAWS,
                        "bootstrap_unit": "cv_group_id/problem_id",
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def _random_pair_distances(
    signatures: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    decision_by_pair = pairs.set_index(["domain", "stratum_id", "seed_i", "seed_j"])["pair_index"].to_dict()
    pair_decision = pairs.set_index(["domain", "pair_index"])["decision_distance_linf"].to_dict() if "decision_distance_linf" in pairs else {}
    for domain, part in signatures.groupby("domain", sort=True):
        for stratum_id, stratum in part.groupby("stratum_id", sort=True):
            seeds = sorted(stratum["seed"].astype(int).tolist())
            rng = np.random.default_rng(_seed_sequence(RANDOM_PAIR_STREAM, DOMAIN_CODES[domain], len(stratum_id)))
            state_positions = {int(row.seed): int(row.Index) for row in stratum.itertuples()}
            for repetition in range(N_RANDOM_CONTROLS):
                pair_seed = tuple(sorted(rng.choice(seeds, size=2, replace=False).tolist()))
                pair_index = decision_by_pair[(domain, stratum_id, *pair_seed)]
                rows.append(
                    {
                        "domain": domain,
                        "stratum_id": stratum_id,
                        "cv_group_id": stratum["cv_group_id"].iloc[0],
                        "bootstrap_group": stratum["bootstrap_group"].iloc[0],
                        "pair_index": int(pair_index),
                        "repetition": repetition,
                    }
                )
    return pd.DataFrame(rows)


def _aliasing_products(
    merged: pd.DataFrame,
    signatures: pd.DataFrame,
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    random_pairs = _random_pair_distances(signatures, pairs)
    random_decision = {
        (row.domain, int(row.pair_index)): float(row.decision_distance_linf)
        for row in merged.drop_duplicates(["domain", "pair_index"]).itertuples(index=False)
    }
    random_pairs["decision_distance_linf"] = [
        random_decision[(row.domain, int(row.pair_index))]
        for row in random_pairs.itertuples(index=False)
    ]
    rows = []
    for (domain, rep_name), part in merged.groupby(["domain", "representation"], sort=True):
        for suite in SCOPES:
            subset = _scope(part, suite).copy()
            all_rate = float(np.mean(subset["decision_distance_linf"] > 1.0))
            random_subset = _scope(
                random_pairs.merge(
                    pairs[["domain", "pair_index", "suite"]],
                    on=["domain", "pair_index"],
                    validate="many_to_one",
                ),
                suite,
            )
            random_rate = float(np.mean(random_subset["decision_distance_linf"] > 1.0)) if len(random_subset) else np.nan
            random_median = float(np.median(random_subset["decision_distance_linf"])) if len(random_subset) else np.nan
            metrics = {}
            for quantile in (5, 10, 20):
                cutoff = float(np.quantile(subset["representation_distance_l1_mean"], quantile / 100.0))
                near = subset.loc[subset["representation_distance_l1_mean"] <= cutoff]
                point, low, high = _metric_bootstrap(
                    near,
                    lambda frame: float(np.mean(frame["decision_distance_linf"] > 1.0)),
                    (DOMAIN_CODES[domain], REPRESENTATION_CODES[rep_name], SUITE_CODES[suite], quantile),
                )
                metrics[f"ava_{quantile}"] = point
                metrics[f"ava_{quantile}_ci_low"] = low
                metrics[f"ava_{quantile}_ci_high"] = high
                metrics[f"near_{quantile}_n"] = len(near)
                if quantile == 10:
                    sg_point, sg_low, sg_high = _metric_bootstrap(
                        near,
                        lambda frame: float(np.median(frame["decision_distance_linf"])),
                        (DOMAIN_CODES[domain], REPRESENTATION_CODES[rep_name], SUITE_CODES[suite], 110),
                    )
                    metrics["sg_phi"] = sg_point
                    metrics["sg_phi_ci_low"] = sg_low
                    metrics["sg_phi_ci_high"] = sg_high
            sg_random = float(np.median(random_subset["decision_distance_linf"])) if len(random_subset) else np.nan
            rows.append(
                {
                    "domain": domain,
                    "suite": suite,
                    "representation": rep_name,
                    "n_pairs": len(subset),
                    "ava_all": all_rate,
                    "ava_random_pair": random_rate,
                    "ava_random_median_decision_distance": random_median,
                    "r_alias": metrics["ava_10"] / all_rate if all_rate > 0 else np.nan,
                    "sg_random": sg_random,
                    "sufficiency_gap": 1.0 - metrics["sg_phi"] / sg_random if np.isfinite(sg_random) and sg_random > 0 else np.nan,
                    "random_pair_repetitions": N_RANDOM_CONTROLS,
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "bootstrap_unit": "cv_group_id/problem_id",
                    **metrics,
                }
            )
    return pd.DataFrame(rows), random_pairs


def _heterogeneity_summary(decision: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain in sorted(decision["domain"].unique()):
        for suite in SCOPES:
            subset = _scope(decision.loc[decision["domain"].eq(domain)], suite)
            probability = float(np.mean(subset["decision_distance_linf"] > 1.0))
            suite_probabilities = {
                s: float(np.mean(_scope(decision.loc[decision["domain"].eq(domain)], s)["decision_distance_linf"] > 1.0))
                for s in ("bbob", "mabbob")
            }
            if suite == "pooled":
                if probability >= 0.10 and min(suite_probabilities.values()) >= 0.05:
                    verdict = "DH1 NONTRIVIAL"
                elif probability >= 0.05:
                    verdict = "DH2 WEAK"
                else:
                    verdict = "DH3 DEGENERATE"
            else:
                verdict = "DH1 NONTRIVIAL" if probability >= 0.05 else ("DH3 DEGENERATE" if probability < 0.05 else "DH2 WEAK")
            rows.append(
                {
                    "domain": domain,
                    "suite": suite,
                    "n_pairs": len(subset),
                    "decision_distance_median": float(np.median(subset["decision_distance_linf"])),
                    "decision_distance_q75": float(np.quantile(subset["decision_distance_linf"], 0.75)),
                    "decision_distance_gt1_rate": probability,
                    "bbob_gt1_rate": suite_probabilities["bbob"],
                    "mabbob_gt1_rate": suite_probabilities["mabbob"],
                    "heterogeneity_verdict": verdict,
                }
            )
    return pd.DataFrame(rows)


def _paired_problem_bootstrap(
    natural_values: pd.Series,
    post_values: pd.Series,
    stream_codes: tuple[int, ...],
) -> tuple[float, float, float, int]:
    joined = pd.concat(
        [natural_values.rename("natural"), post_values.rename("post")], axis=1, join="inner"
    ).dropna()
    if joined.empty:
        return float("nan"), float("nan"), float("nan"), 0
    differences = (joined["post"] - joined["natural"]).to_numpy(dtype=float)
    rng = np.random.default_rng(_seed_sequence(BOOTSTRAP_STREAM, *stream_codes, len(differences)))
    draws = rng.choice(differences, size=(BOOTSTRAP_DRAWS, len(differences)), replace=True).mean(axis=1)
    return (
        float(np.mean(differences)),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        int(len(differences)),
    )


def _natural_post_shift(
    alignment: pd.DataFrame,
    nn_summary: pd.DataFrame,
    aliasing: pd.DataFrame,
    merged: pd.DataFrame,
    nn: pd.DataFrame,
) -> pd.DataFrame:
    comparable = ["compact6", "global28"]
    rows = []
    for representation in comparable:
        for suite in SCOPES:
            a = alignment.loc[alignment["representation"].eq(representation) & alignment["suite"].eq(suite)]
            n = nn_summary.loc[nn_summary["representation"].eq(representation) & nn_summary["suite"].eq(suite)]
            v = aliasing.loc[aliasing["representation"].eq(representation) & aliasing["suite"].eq(suite)]
            if len(a) != 2 or len(n) != 2 or len(v) != 2:
                continue
            natural = a.loc[a["domain"].eq("natural")].iloc[0]
            post = a.loc[a["domain"].eq("post_handoff")].iloc[0]
            natural_n = n.loc[n["domain"].eq("natural")].iloc[0]
            post_n = n.loc[n["domain"].eq("post_handoff")].iloc[0]
            natural_v = v.loc[v["domain"].eq("natural")].iloc[0]
            post_v = v.loc[v["domain"].eq("post_handoff")].iloc[0]
            pair_part = merged.loc[merged["representation"].eq(representation)]
            if suite != "pooled":
                pair_part = pair_part.loc[pair_part["suite"].eq(suite)]
            problem_rho = {}
            for (domain, problem_id), problem_part in pair_part.groupby(["domain", "problem_id"], sort=True):
                problem_rho[(domain, problem_id)] = _spearman(
                    problem_part["representation_distance_l1_mean"].to_numpy(),
                    problem_part["decision_distance_linf"].to_numpy(),
                )
            natural_rho = pd.Series(
                {problem: value for (domain, problem), value in problem_rho.items() if domain == "natural"}, dtype=float
            )
            post_rho = pd.Series(
                {problem: value for (domain, problem), value in problem_rho.items() if domain == "post_handoff"}, dtype=float
            )
            rho_boot = _paired_problem_bootstrap(
                natural_rho,
                post_rho,
                (REPRESENTATION_CODES[representation], SUITE_CODES[suite], 31),
            )
            nn_part = nn.loc[nn["representation"].eq(representation)]
            if suite != "pooled":
                nn_part = nn_part.loc[nn_part["suite"].eq(suite)]
            nn_by_problem = nn_part.groupby(["domain", "problem_id"], sort=True).agg(
                random_median=("decision_distance_random_median", "median"),
                observed=("decision_distance_observable_nn", "median"),
            )
            natural_nn = (nn_by_problem.loc["natural", "random_median"] - nn_by_problem.loc["natural", "observed"])
            post_nn = (nn_by_problem.loc["post_handoff", "random_median"] - nn_by_problem.loc["post_handoff", "observed"])
            nn_boot = _paired_problem_bootstrap(
                natural_nn,
                post_nn,
                (REPRESENTATION_CODES[representation], SUITE_CODES[suite], 32),
            )
            problem_ava = {}
            for (domain, problem_id), problem_part in pair_part.groupby(["domain", "problem_id"], sort=True):
                cutoff = float(np.quantile(problem_part["representation_distance_l1_mean"], 0.10))
                near = problem_part.loc[problem_part["representation_distance_l1_mean"] <= cutoff]
                problem_ava[(domain, problem_id)] = float(np.mean(near["decision_distance_linf"] > 1.0))
            natural_ava = pd.Series(
                {problem: value for (domain, problem), value in problem_ava.items() if domain == "natural"}, dtype=float
            )
            post_ava = pd.Series(
                {problem: value for (domain, problem), value in problem_ava.items() if domain == "post_handoff"}, dtype=float
            )
            ava_boot = _paired_problem_bootstrap(
                natural_ava,
                post_ava,
                (REPRESENTATION_CODES[representation], SUITE_CODES[suite], 33),
            )
            rows.append(
                {
                    "representation": representation,
                    "suite": suite,
                    "rho_natural": natural["rho_observed_pooled"],
                    "rho_post_handoff": post["rho_observed_pooled"],
                    "delta_rho_post_minus_natural": post["rho_observed_pooled"] - natural["rho_observed_pooled"],
                    "delta_nn_natural": natural_n["delta_nn"],
                    "delta_nn_post_handoff": post_n["delta_nn"],
                    "delta_nn_post_minus_natural": post_n["delta_nn"] - natural_n["delta_nn"],
                    "ava10_natural": natural_v["ava_10"],
                    "ava10_post_handoff": post_v["ava_10"],
                    "ava10_post_minus_natural": post_v["ava_10"] - natural_v["ava_10"],
                    "delta_rho_ci_low": rho_boot[1],
                    "delta_rho_ci_high": rho_boot[2],
                    "delta_nn_ci_low": nn_boot[1],
                    "delta_nn_ci_high": nn_boot[2],
                    "delta_ava10_ci_low": ava_boot[1],
                    "delta_ava10_ci_high": ava_boot[2],
                    "problem_bootstrap_n_rho": rho_boot[3],
                    "problem_bootstrap_n_nn": nn_boot[3],
                    "problem_bootstrap_n_ava10": ava_boot[3],
                    "problem_bootstrap_draws": BOOTSTRAP_DRAWS,
                    "problem_bootstrap_unit": "problem_id",
                }
            )
    return pd.DataFrame(rows)


def _ladder_summary(alignment: pd.DataFrame, nn_summary: pd.DataFrame, aliasing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for representation in FORMAL_POST_REPRESENTATIONS:
        for suite in SCOPES:
            a = alignment.loc[alignment["domain"].eq("post_handoff") & alignment["representation"].eq(representation) & alignment["suite"].eq(suite)]
            n = nn_summary.loc[nn_summary["domain"].eq("post_handoff") & nn_summary["representation"].eq(representation) & nn_summary["suite"].eq(suite)]
            v = aliasing.loc[aliasing["domain"].eq("post_handoff") & aliasing["representation"].eq(representation) & aliasing["suite"].eq(suite)]
            if a.empty or n.empty or v.empty:
                continue
            rows.append(
                {
                    "domain": "post_handoff",
                    "suite": suite,
                    "representation": representation,
                    "rho": float(a.iloc[0]["rho_observed_pooled"]),
                    "rho_ci_low": float(a.iloc[0]["rho_ci_low"]),
                    "rho_ci_high": float(a.iloc[0]["rho_ci_high"]),
                    "alignment_verdict": a.iloc[0]["alignment_verdict"],
                    "ava10": float(v.iloc[0]["ava_10"]),
                    "ava10_ci_low": float(v.iloc[0]["ava_10_ci_low"]),
                    "ava10_ci_high": float(v.iloc[0]["ava_10_ci_high"]),
                    "delta_nn": float(n.iloc[0]["delta_nn"]),
                    "delta_nn_ci_low": float(n.iloc[0]["delta_nn_ci_low"]),
                    "delta_nn_ci_high": float(n.iloc[0]["delta_nn_ci_high"]),
                }
            )
    return pd.DataFrame(rows)


def _collision_examples(
    merged: pd.DataFrame,
    signatures: pd.DataFrame,
) -> pd.DataFrame:
    candidate_rows = []
    for (domain, rep_name, suite), part in merged.groupby(["domain", "representation", "suite"], sort=True):
        threshold = float(np.quantile(part["representation_distance_l1_mean"], 0.05))
        near = part.loc[part["representation_distance_l1_mean"] <= threshold].sort_values(
            ["decision_distance_linf", "problem_id", "pair_index"], ascending=[False, True, True]
        )
        for row in near.itertuples(index=False):
            sig_i = signatures.loc[signatures["state_id"].eq(row.state_i)].iloc[0]
            sig_j = signatures.loc[signatures["state_id"].eq(row.state_j)].iloc[0]
            candidate_rows.append(
                {
                    "domain": domain,
                    "suite": suite,
                    "representation": rep_name,
                    "example_rank": 0,
                    "problem_id": row.problem_id,
                    "route": row.route,
                    "current_algorithm": row.current_algorithm,
                    "source_FE": row.source_FE,
                    "state_i": row.state_i,
                    "state_j": row.state_j,
                    "seed_i": row.seed_i,
                    "seed_j": row.seed_j,
                    "representation_distance": row.representation_distance_l1_mean,
                    "decision_distance": row.decision_distance_linf,
                    "loss_shade_i": sig_i["loss_shade"],
                    "loss_lshade_i": sig_i["loss_lshade"],
                    "loss_cso_i": sig_i["loss_cso"],
                    "loss_shade_j": sig_j["loss_shade"],
                    "loss_lshade_j": sig_j["loss_lshade"],
                    "loss_cso_j": sig_j["loss_cso"],
                    "normalized_margins_i": ";".join(f"{column}={sig_i[column]:.6g}" for column in NORMALIZED_MARGIN_COLUMNS),
                    "normalized_margins_j": ";".join(f"{column}={sig_j[column]:.6g}" for column in NORMALIZED_MARGIN_COLUMNS),
                    "practical_action_set_i": sig_i["A_ND_members"],
                    "practical_action_set_j": sig_j["A_ND_members"],
                }
            )
    rows = []
    for (domain, suite), part in pd.DataFrame(candidate_rows).groupby(["domain", "suite"], sort=True):
        selected_problems: set[str] = set()
        part = part.sort_values(["decision_distance", "representation_distance", "problem_id"], ascending=[False, True, True])
        for row in part.itertuples(index=False):
            if row.problem_id in selected_problems:
                continue
            selected_problems.add(row.problem_id)
            record = row._asdict()
            record["example_rank"] = len(selected_problems)
            rows.append(record)
            if len(selected_problems) >= 10:
                break
    return pd.DataFrame(rows)


def _suite_robustness(
    heterogeneity: pd.DataFrame,
    alignment: pd.DataFrame,
    nn_summary: pd.DataFrame,
    aliasing: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["domain", "suite"]
    result = heterogeneity.merge(
        alignment[
            keys
            + [
                "representation",
                "rho_observed_pooled",
                "rho_ci_low",
                "rho_ci_high",
                "permutation_q97_5",
                "empirical_p",
                "alignment_verdict",
            ]
        ],
        on=keys,
        how="outer",
    ).merge(
        nn_summary[
            keys
            + [
                "representation",
                "delta_nn",
                "delta_nn_ci_low",
                "delta_nn_ci_high",
                "observable_practical_consistency",
                "random_practical_consistency",
            ]
        ],
        on=keys + ["representation"],
        how="outer",
    ).merge(
        aliasing[
            keys
            + [
                "representation",
                "ava_all",
                "ava_5",
                "ava_10",
                "ava_20",
                "r_alias",
                "sufficiency_gap",
            ]
        ],
        on=keys + ["representation"],
        how="outer",
        suffixes=("", "_aliasing"),
    )
    return result


def _derive_verdicts(
    heterogeneity: pd.DataFrame,
    alignment: pd.DataFrame,
    nn_summary: pd.DataFrame,
    aliasing: pd.DataFrame,
    ladder: pd.DataFrame,
    shift: pd.DataFrame,
) -> dict[str, object]:
    dh = heterogeneity.loc[heterogeneity["suite"].eq("pooled")].set_index("domain")
    dh_status = {
        domain: str(dh.loc[domain, "heterogeneity_verdict"])
        for domain in dh.index
    }
    if any(value == "DH3 DEGENERATE" for value in dh_status.values()):
        heterogeneity_verdict = "DH3 DEGENERATE"
    elif all(value == "DH1 NONTRIVIAL" for value in dh_status.values()):
        heterogeneity_verdict = "DH1 NONTRIVIAL"
    else:
        heterogeneity_verdict = "DH2 WEAK"

    post_align = alignment.loc[
        alignment["domain"].eq("post_handoff") & alignment["suite"].isin(["bbob", "mabbob"])
    ]
    post_majority_da3 = (
        post_align.groupby("representation")["alignment_verdict"]
        .apply(lambda values: int((values == "DA3 UNALIGNED").sum()))
        .to_dict()
    )
    post_alias = aliasing.loc[
        aliasing["domain"].eq("post_handoff") & aliasing["suite"].eq("pooled")
    ].set_index("representation")
    post_nn = nn_summary.loc[
        nn_summary["domain"].eq("post_handoff") & nn_summary["suite"].eq("pooled")
    ].set_index("representation")
    global_alias = float(post_alias.loc["global28", "ava_10"])
    ladder_recoverable = False
    for representation in FORMAL_POST_REPRESENTATIONS:
        if representation in ("compact6", "global28") or representation not in post_alias.index:
            continue
        aligned = post_align.loc[post_align["representation"].eq(representation)]
        if len(aligned) != 2 or "global28" not in post_alias.index or "global28" not in post_nn.index:
            continue
        candidate_alias = post_alias.loc[representation]
        baseline_alias = post_alias.loc["global28"]
        candidate_nn = post_nn.loc[representation]
        baseline_nn = post_nn.loc["global28"]
        same_direction = bool(
            np.all(
                candidate_alias["ava_10"] < baseline_alias["ava_10"]
            )
        )
        # RL1 is reserved for a reproducible two-suite improvement.  With
        # pre-specified point summaries and CIs only, require pointwise
        # improvement in both suites and uncertainty intervals that exclude
        # the no-improvement direction for each metric.
        suite_improvements = []
        for suite in ("bbob", "mabbob"):
            cand_a = aliasing.loc[
                aliasing["domain"].eq("post_handoff")
                & aliasing["representation"].eq(representation)
                & aliasing["suite"].eq(suite)
            ].iloc[0]
            base_a = aliasing.loc[
                aliasing["domain"].eq("post_handoff")
                & aliasing["representation"].eq("global28")
                & aliasing["suite"].eq(suite)
            ].iloc[0]
            cand_n = nn_summary.loc[
                nn_summary["domain"].eq("post_handoff")
                & nn_summary["representation"].eq(representation)
                & nn_summary["suite"].eq(suite)
            ].iloc[0]
            base_n = nn_summary.loc[
                nn_summary["domain"].eq("post_handoff")
                & nn_summary["representation"].eq("global28")
                & nn_summary["suite"].eq(suite)
            ].iloc[0]
            suite_improvements.append(
                bool(
                    cand_a["ava_10"] < base_a["ava_10"]
                    and cand_n["delta_nn"] > base_n["delta_nn"]
                    and cand_n["delta_nn_ci_low"] > 0
                )
            )
        if same_direction and all(suite_improvements) and bool((aligned["alignment_verdict"] == "DA1 ALIGNED").all()):
            ladder_recoverable = True
            break
    if ladder_recoverable:
        ladder_verdict = "RL1 RECOVERABLE"
    else:
        any_partial = bool(
            ((ladder["alignment_verdict"] == "DA1 ALIGNED") | (ladder["alignment_verdict"] == "DA2 WEAK")).any()
            if not ladder.empty
            else False
        )
        ladder_verdict = "RL2 PARTIAL" if any_partial else "RL3 OBSERVABLE LADDER SATURATED"

    natural_alignment = alignment.loc[
        alignment["domain"].eq("natural") & alignment["suite"].isin(["bbob", "mabbob"])
    ]
    post_alignment = post_align
    natural_has_signal = bool(
        ((natural_alignment["alignment_verdict"] == "DA1 ALIGNED")
         | ((natural_alignment["alignment_verdict"] == "DA2 WEAK") & (natural_alignment["rho_ci_low"] > -0.05))).any()
    )
    post_weakened = bool(
        (post_alignment["alignment_verdict"].isin(["DA3 UNALIGNED", "DA2 WEAK"])).mean() >= 0.5
    ) if not post_alignment.empty else False
    if not shift.empty:
        shift_primary = shift.loc[shift["suite"].eq("pooled")]
        consistent_metrics = int(
            (shift_primary["delta_rho_post_minus_natural"] < 0).sum()
            + (shift_primary["delta_nn_post_minus_natural"] < 0).sum()
            + (shift_primary["ava10_post_minus_natural"] > 0).sum()
        )
    else:
        consistent_metrics = 0
    h_suff = bool(natural_has_signal and post_weakened and consistent_metrics >= 2)

    if heterogeneity_verdict == "DH3 DEGENERATE":
        final_verdict = "V4 DECISION HETEROGENEITY TOO WEAK"
    elif ladder_verdict == "RL1 RECOVERABLE":
        final_verdict = "V3 OBSERVABLE REPRESENTATION STILL INFORMATIVE"
    elif h_suff and heterogeneity_verdict == "DH1 NONTRIVIAL":
        final_verdict = "V1 DECISION-STATE ALIASING CONFIRMED"
    else:
        final_verdict = "V2 DISTRIBUTION-DEPENDENT SUFFICIENCY"
    return {
        "heterogeneity_verdict": heterogeneity_verdict,
        "domain_heterogeneity": dh_status,
        "ladder_verdict": ladder_verdict,
        "h_suff_supported": h_suff,
        "h_suff_consistent_metric_count": consistent_metrics,
        "post_majority_da3_counts": post_majority_da3,
        "final_verdict": final_verdict,
        "next_step_allowed": "Task17B solver-internal adaptive-state attribution" if final_verdict.startswith("V1") else "STOP",
        "new_selector_allowed": False,
        "seeds_6_10_allowed": False,
        "cec_allowed": False,
    }


def _make_figures(
    merged: pd.DataFrame,
    ladder: pd.DataFrame,
    shift: pd.DataFrame,
) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 120,
        }
    )
    colors = {"bbob": "#0072B2", "mabbob": "#D55E00", "natural": "#009E73", "post_handoff": "#CC79A7"}
    outputs: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=False, sharey=True)
    for ax, domain in zip(axes, ("natural", "post_handoff"), strict=True):
        part = merged.loc[(merged["domain"] == domain) & (merged["representation"] == "global28")]
        for suite in ("bbob", "mabbob"):
            subset = part.loc[part["suite"] == suite]
            ax.scatter(
                subset["representation_distance_l1_mean"],
                subset["decision_distance_linf"],
                s=5,
                alpha=0.30,
                color=colors[suite],
                label=suite.upper(),
                rasterized=True,
            )
        ax.axhline(1.0, color="#444444", linestyle="--", linewidth=0.7)
        ax.set_title("Natural states" if domain == "natural" else "Post-handoff states")
        ax.set_xlabel("Representation distance (standardized L1 mean)")
        ax.set_ylabel("Decision distance (noise units)" if ax is axes[0] else "")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        path = FIGURES / f"task17a_figure_a_distance_alignment{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    order = [rep for rep in LADDER if rep in set(ladder["representation"])]
    x = np.arange(len(order))
    width = 0.36
    for offset, suite in enumerate(("bbob", "mabbob")):
        part = ladder.loc[ladder["suite"].eq(suite)].set_index("representation").reindex(order)
        ax.errorbar(
            x + (offset - 0.5) * width,
            part["ava10"],
            yerr=[part["ava10"] - part["ava10_ci_low"], part["ava10_ci_high"] - part["ava10"]],
            fmt="o",
            color=colors[suite],
            capsize=2,
            label=suite.upper(),
        )
    ax.set_xticks(x, order, rotation=25, ha="right")
    ax.set_ylabel("AVA10: P(decision distance > 1)")
    ax.set_xlabel("Post-handoff representation")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        path = FIGURES / f"task17a_figure_b_representation_ladder{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    if shift.empty:
        for ax, ylabel in zip(axes, ("Spearman distance alignment", "NN decision-distance improvement"), strict=True):
            ax.text(0.5, 0.5, "No comparable natural/post rows", ha="center", va="center")
            ax.set_ylabel(ylabel)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        for suffix in (".pdf", ".png"):
            path = FIGURES / f"task17a_figure_c_natural_post_shift{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            outputs.append(path)
        plt.close(fig)
        return outputs
    for metric, ax, ylabel in (
        ("rho", axes[0], "Spearman distance alignment"),
        ("delta_nn", axes[1], "NN decision-distance improvement"),
    ):
        for representation, marker in (("compact6", "o"), ("global28", "s")):
            part = shift.loc[shift["representation"].eq(representation)]
            if metric == "rho":
                y_n = part["rho_natural"]; y_p = part["rho_post_handoff"]
            else:
                y_n = part["delta_nn_natural"]; y_p = part["delta_nn_post_handoff"]
            xpos = np.arange(len(part)) + (0.0 if representation == "compact6" else 0.12)
            ax.plot(xpos, y_n, marker=marker, color=colors["natural"], linestyle="-", label=f"{representation} natural")
            ax.plot(xpos, y_p, marker=marker, color=colors["post_handoff"], linestyle="--", label=f"{representation} post")
        ax.axhline(0.0, color="#444444", linewidth=0.7)
        ax.set_xticks(np.arange(3), ["BBOB", "MA", "pooled"])
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        path = FIGURES / f"task17a_figure_c_natural_post_shift{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def _write_report(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    content = [f"# {title}", ""]
    for heading, body in sections:
        content.extend([f"## {heading}", "", body.strip(), ""])
    path.write_text("\n".join(content), encoding="utf-8")


def _resource_ledger(start: float, end: float, extra: dict[str, object]) -> pd.DataFrame:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    disk = shutil.disk_usage(ROOT)
    return pd.DataFrame(
        [
            {
                "phase": "task17a_representation_sufficiency",
                "new_objective_fe": 0,
                "elapsed_seconds": end - start,
                "cpu_logical_count": os.cpu_count(),
                "max_rss_mb": float(usage.ru_maxrss / (1024 * 1024)),
                "disk_available_gb": float(disk.free / (1024**3)),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "permutation_repetitions": N_PERMUTATIONS,
                "random_control_repetitions": N_RANDOM_CONTROLS,
                "input_scope": "current Decision-before-Feature project only",
                "runtime_role": "measurement metadata only; not used in scientific labels",
                **extra,
            }
        ]
    )


def run_analysis() -> dict[str, object]:
    start = perf_counter()
    OUT_LIGHT.mkdir(parents=True, exist_ok=True)
    OUT_HEAVY.mkdir(parents=True, exist_ok=True)
    (
        domain_frames,
        signatures,
        pairs,
        decision,
        representation,
        representation_inventory,
        preprocessing,
        join_summary,
        _matrices,
    ) = build_data_products()
    merged = _merge_pair_products(pairs, decision, representation)
    permutation_parts = []
    for domain in ("natural", "post_handoff"):
        domain_signatures = signatures.loc[signatures["domain"].eq(domain)].reset_index(drop=True)
        domain_pairs = pairs.loc[pairs["domain"].eq(domain)].reset_index(drop=True)
        domain_representation = representation.loc[representation["domain"].eq(domain)].reset_index(drop=True)
        permutation_parts.append(
            _run_alignment_permutations(domain_signatures, domain_pairs, domain_representation)
        )
    permutations = pd.concat(permutation_parts, ignore_index=True)
    alignment = _distance_alignment(merged, permutations)
    pairs_with_decision = pairs.merge(
        decision[["domain", "pair_index", "decision_distance_linf"]],
        on=["domain", "pair_index"],
        validate="one_to_one",
    )
    nn, nn_summary = _nearest_neighbor_products(signatures, pairs_with_decision, representation)
    aliasing, random_pairs = _aliasing_products(merged, signatures, pairs)
    heterogeneity = _heterogeneity_summary(decision)
    shift = _natural_post_shift(alignment, nn_summary, aliasing, merged, nn)
    ladder = _ladder_summary(alignment, nn_summary, aliasing)
    collisions = _collision_examples(merged, signatures)
    robustness = _suite_robustness(heterogeneity, alignment, nn_summary, aliasing)
    verdicts = _derive_verdicts(heterogeneity, alignment, nn_summary, aliasing, ladder, shift)
    figures = _make_figures(merged, ladder, shift)
    end = perf_counter()
    ledger = _resource_ledger(
        start,
        end,
        {
            "final_verdict": verdicts["final_verdict"],
            "natural_states": int(len(signatures.loc[signatures["domain"].eq("natural")])),
            "post_handoff_states": int(len(signatures.loc[signatures["domain"].eq("post_handoff")])),
            "within_stratum_pairs": int(len(pairs)),
            "figures": ";".join(str(path.relative_to(ROOT)) for path in figures),
        },
    )

    products = {
        "task17a_state_decision_signatures": signatures,
        "task17a_within_stratum_pairs": pairs,
        "task17a_representation_distances": representation,
        "task17a_decision_distances": decision,
        "task17a_alignment_summary": alignment,
        "task17a_alignment_permutations": permutations,
        "task17a_nn_consistency": nn,
        "task17a_aliasing_rates": aliasing,
        "task17a_natural_post_shift": shift,
        "task17a_representation_ladder": ladder,
        "task17a_collision_examples": collisions,
        "task17a_solver_suite_robustness": robustness,
        "task17a_resource_ledger": ledger,
        "task17a_representation_inventory": representation_inventory,
        "task17a_representation_preprocessing": preprocessing,
        "task17a_random_pair_controls": random_pairs,
        "task17a_heterogeneity_summary": heterogeneity,
        "task17a_join_summary": join_summary,
    }
    for name, table in products.items():
        table.to_parquet(OUT_HEAVY / f"{name}.parquet", index=False)

    reports = {
        "17a01_zero_fe_contract.md": [
            ("协议", "本轮只读取既有 natural 与 post-handoff 状态、三动作 1000-FE 结果和已标定噪声尺度；新增 objective FE = 0。未调用 optimizer、benchmark、ELA、selector 或闭环控制。"),
            ("范围", "自然域 1890 states；交接后域 3780 states；所有随机数由显式 SeedSequence 产生。"),
        ],
        "17a02_data_join_and_state_identity.md": [
            ("精确拼接", join_summary.to_markdown(index=False)),
            ("状态键", "自然域 stratum=(problem_id,current_algorithm,source_FE)；交接后域 stratum=(problem_id,route,source_FE)。每个 stratum 均保留 seeds 1–5，因而各自产生 10 个 state pairs。"),
        ],
        "17a03_decision_signature_definition.md": [
            ("定义", "动作集合为 SHADE、L-SHADE、CSO。中心化损失为每个 state 的动作损失减去三动作均值；pairwise margin 为 L_b-L_a，正值表示动作 a 的损失较低。每对 margin 除以对应两个 solver 的 fold-local practical noise scale 的较大值。"),
            ("保存", "state_decision_signatures 表保存原始损失、中心化损失、margin、normalized_margin 与 practical action set；该表不使用 raw argmin 作为主标签。"),
        ],
        "17a04_within_stratum_heterogeneity.md": [
            ("结果", heterogeneity.to_markdown(index=False)),
            ("判定", f"异质性判定：{verdicts['heterogeneity_verdict']}。只有在 DH1 或 DH2 下才解释动作价值混叠；DH3 时停止该机制解释。"),
        ],
        "17a05_representation_distance_contract.md": [
            ("阶梯", representation_inventory.to_markdown(index=False)),
            ("预处理", "每个 domain、每个 representation 独立计算 feature median 与 IQR，使用 (x-median)/(IQR+1e-12)，固定裁剪到 [-5,5]。没有根据动作结果选择预处理。"),
        ],
        "17a06_distance_alignment.md": [
            ("Alignment", alignment.to_markdown(index=False)),
            ("置换", "置换在每个 stratum 内仅打乱五个 seed 的 decision signature；100 次重复，经验 p=(1+#null>=observed)/(101)。"),
        ],
        "17a07_nearest_neighbor_consistency.md": [
            ("NN", nn_summary.to_markdown(index=False)),
            ("解释", "observable NN 由同一 stratum 的另外四个 states 中最小 mean absolute standardized distance 决定；随机对照在同一候选集合中均匀抽取 100 次。primary 为 continuous decision distance，practical action set 仅作 secondary。"),
        ],
        "17a08_action_value_aliasing.md": [
            ("AVA", aliasing.to_markdown(index=False)),
            ("解释", "AVA_k 是 representation distance 最低 k% pair 中 decision distance > 1 的比例；R_alias=AVA10/AVAall；sufficiency_gap=1-SG_phi/SG_random。"),
        ],
        "17a09_natural_vs_posthandoff_shift.md": [
            ("Shift", shift.to_markdown(index=False)),
            ("H-SUFF", f"预注册 H-SUFF 支持={verdicts['h_suff_supported']}；方向一致指标数={verdicts['h_suff_consistent_metric_count']}。该比较仅限当前两个 tested domains。"),
        ],
        "17a10_representation_ladder.md": [
            ("Ladder", ladder.to_markdown(index=False)),
            ("判定", f"Representation ladder 判定：{verdicts['ladder_verdict']}。该判定不训练新模型，也不引入新 representation。"),
        ],
        "17a11_collision_examples.md": [
            ("规则", "每个 domain/suite 最多 10 对；候选只来自各 representation distance 最低 5% 区域，再按 decision distance 降序选取，每个 problem 最多一对。"),
            ("示例", collisions.to_markdown(index=False)),
        ],
        "17a12_solver_suite_robustness.md": [
            ("Suite", robustness.to_markdown(index=False)),
            ("边界", "BBOB 与 MA-BBOB 分开报告；pooled 仅作辅助。所有结论基于已观测三动作终值，不作无噪声价值函数或因果解释。"),
        ],
        "17a13_resource_ledger.md": [
            ("资源", ledger.to_markdown(index=False)),
            ("策略", "本机 10 logical CPU、16 GB unified memory、约 1 TB 可用磁盘；Parquet 总规模约 2 GB。计算使用内存内 pandas/NumPy，随机重采样规模按预注册值执行。"),
        ],
        "17a14_final_verdict.md": [
            ("Verdict", "```json\n" + pd.Series(verdicts).to_json(force_ascii=False, indent=2) + "\n```"),
            ("科学措辞", "结果只能表述为当前 tested post-handoff setting 下，observable search behavior 是否无法解析一部分已观测备选动作价值差异。不得写成 Behavior 无用、算法选择不可能或 solver-internal state 已被证明是原因。"),
            ("停止条件", "无论最终 verdict，均不自动执行 Task17B、新模型、新特征、new action、seeds 6–10、CEC 或 closed-loop。"),
        ],
    }
    for filename, sections in reports.items():
        _write_report(OUT_LIGHT / filename, filename.removesuffix(".md"), sections)
    total_report = (
        "# Decision-before-Feature：Task17A 已观测动作价值混叠与决策状态充分性\n\n"
        f"最终判定：**{verdicts['final_verdict']}**。\n\n"
        f"数据：natural={len(signatures.loc[signatures.domain.eq('natural')])}，post-handoff={len(signatures.loc[signatures.domain.eq('post_handoff')])}，within-stratum pairs={len(pairs)}，new objective FE=0。\n\n"
        f"异质性：{verdicts['heterogeneity_verdict']}；ladder：{verdicts['ladder_verdict']}；H-SUFF 支持={verdicts['h_suff_supported']}。\n\n"
        "本报告的 primary geometry 使用全 pairwise normalized margins，不使用逐状态 argmin 作为主判定，不作因果主张。\n\n"
        "Task17A 后停止自动扩展；只有在 V1 时，讨论部分才可提出 solver-internal adaptive state 作为待检验的 bounded hypothesis。\n"
    )
    (OUT_LIGHT / "Decision-before-Feature_Task17A_ActionValueAliasing与DecisionStateSufficiency.md").write_text(total_report, encoding="utf-8")
    (OUT_LIGHT / "task17a_verdict.json").write_text(pd.Series(verdicts).to_json(force_ascii=False, indent=2), encoding="utf-8")
    return {"verdicts": verdicts, "products": products, "figures": figures}


def main() -> None:
    result = run_analysis()
    print(pd.Series(result["verdicts"]).to_string())


if __name__ == "__main__":
    main()
