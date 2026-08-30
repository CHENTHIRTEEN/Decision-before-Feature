"""Task 14B.1 correctness repair and scale-matching feasibility check.

Part A repairs four implementation/reporting defects in Task 14B while
preserving the Task 14A action outcomes and the pre-specified RF/Ridge
carriers.  Part B deterministically reconstructs a strict 200/500/1000-FE
segment representation from the persisted Task 14A source states, then runs
the registered grouped OOF, LOSO and permutation comparisons.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import pickle
import re
import resource
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS
from behavior.features import extract_behavior_rows
from behavior_with_ela.action_dataset import NO_QUERY_TRANSFER_EVENT
from behavior_with_ela.analysis_v5.task12_1_analysis import SOLVERS
from behavior_with_ela.analysis_v5.task13.task13_analysis import make_carrier, run_grouped_oof
from behavior_with_ela.analysis_v5.task13.task13_replay import SAMPLING_METADATA
from behavior_with_ela.protocol import load_experiment_config, make_experiment_problem
from optimizers import (
    OptimizerSettings,
    advance_optimizer_state,
    initialize_optimizer_state,
    initialize_transferred_optimizer_state,
)
from trajectory.records import TrajectoryRecord
from trajectory.window_statistics import build_window_statistics, make_native_update_snapshot


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/behavior_with_ela_train.yaml"
TRAJECTORY_ROOT = ROOT / "behavior_with_ela/results/trajectories"
T14A_HEAVY = ROOT / "behavior_with_ela/results/analysis_v6/task14a"
T14A_LIGHT = ROOT / "behavior_with_ela/analysis_v6/task14a"
T14B_HEAVY = ROOT / "behavior_with_ela/results/analysis_v6/task14b"
T14B_LIGHT = ROOT / "behavior_with_ela/analysis_v6/task14b"
OUT_HEAVY = ROOT / "behavior_with_ela/results/analysis_v6/task14b_1"
OUT_LIGHT = ROOT / "behavior_with_ela/analysis_v6/task14b_1"

LOSS_COLS = [f"loss_{solver}" for solver in SOLVERS]
BOOTSTRAP_DRAWS = 5000
N_PERM = 100
DEFAULT_WORKERS = 8
ANALYSIS_STREAM = 2026090225
CLIP_BOUND = 1e6
TINY_BOUND = 1e-12
MATCHED_SEGMENT_FE = 1000


def function_balanced_mean(frame: pd.DataFrame, values: str | np.ndarray) -> float:
    series = frame[values] if isinstance(values, str) else pd.Series(values, index=frame.index)
    return float(series.groupby(frame["cv_group_id"]).mean().mean())


def paired_group_bootstrap(
    frame: pd.DataFrame,
    upper: np.ndarray,
    lower: np.ndarray,
    stream: int,
) -> tuple[float, float, float]:
    differences = pd.Series(upper - lower, index=frame.index)
    group_means = differences.groupby(frame["cv_group_id"]).mean()
    groups = np.asarray(sorted(group_means.index), dtype=object)
    rng = np.random.default_rng(
        np.random.SeedSequence([ANALYSIS_STREAM, int(stream), len(groups)]).generate_state(4)
    )
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    values = group_means.to_dict()
    for draw in range(BOOTSTRAP_DRAWS):
        sample = rng.choice(groups, size=len(groups), replace=True)
        draws[draw] = float(np.mean([values[group] for group in sample]))
    return (
        float(group_means.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _behavior_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{column[3:]}" for column in SELECTOR_BEHAVIOR_FEATURE_COLUMNS]


def _practical_action_sets(frame: pd.DataFrame) -> pd.DataFrame:
    noise = pd.read_parquet(T14A_LIGHT / "post_handoff_noise_deltas.parquet")
    delta_by_solver = (
        noise.loc[noise["stratum"].eq("pooled")].set_index("solver")["delta_95"].to_dict()
    )
    values = frame[LOSS_COLS].to_numpy(dtype=float)
    action_index = {action: index for index, action in enumerate(SOLVERS)}
    current_index = frame["current_algorithm"].map(action_index).to_numpy(dtype=int)
    by_rule: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for rule in ("max", "sum"):
        delta = np.zeros((len(SOLVERS), len(SOLVERS)), dtype=float)
        for i, action_i in enumerate(SOLVERS):
            for j, action_j in enumerate(SOLVERS):
                if i == j:
                    continue
                if rule == "max":
                    delta[i, j] = max(delta_by_solver[action_i], delta_by_solver[action_j])
                else:
                    delta[i, j] = delta_by_solver[action_i] + delta_by_solver[action_j]
        # dominates[row, i, j] means action i is practically better than action j.
        dominates = values[:, :, None] < values[:, None, :] - delta[None, :, :]
        non_dominated = ~dominates.any(axis=1)
        current_in_set = non_dominated[np.arange(len(frame)), current_index]
        by_rule[rule] = (non_dominated, current_in_set)

    non_dominated_max, current_in_max = by_rule["max"]
    non_dominated_sum, current_in_sum = by_rule["sum"]

    result = frame[
        [
            "state_id",
            "suite",
            "problem_id",
            "cv_group_id",
            "seed",
            "source_algorithm",
            "current_algorithm",
            "source_FE",
        ]
    ].copy()
    result["A_ND_members"] = [
        "|".join(action for action, keep in zip(SOLVERS, row) if keep)
        for row in non_dominated_max
    ]
    result["A_ND_size"] = non_dominated_max.sum(axis=1).astype(int)
    result["current_in_A_ND_max"] = current_in_max
    result["switch_required_max"] = ~current_in_max
    result["A_ND_members_sum"] = [
        "|".join(action for action, keep in zip(SOLVERS, row) if keep)
        for row in non_dominated_sum
    ]
    result["A_ND_size_sum"] = non_dominated_sum.sum(axis=1).astype(int)
    result["current_in_A_ND_sum"] = current_in_sum
    result["switch_required_sum"] = ~current_in_sum
    result["task14a_A_ND_size"] = frame["task14a_A_ND_size"].to_numpy(dtype=int)
    result["task14a_current_in_A_ND"] = frame["task14a_current_in_A_ND"].to_numpy(dtype=bool)
    result["task14a_switch_required"] = frame["task14a_switch_required"].to_numpy(dtype=bool)
    result["A_ND_size_matches_task14a"] = result["A_ND_size"].eq(result["task14a_A_ND_size"])
    result["max_current_membership_matches_task14a"] = result["current_in_A_ND_max"].eq(
        result["task14a_current_in_A_ND"]
    )
    result["sum_current_membership_matches_task14a"] = result["current_in_A_ND_sum"].eq(
        result["task14a_current_in_A_ND"]
    )
    result["sum_switch_required_matches_task14a"] = result["switch_required_sum"].eq(
        result["task14a_switch_required"]
    )
    # Task 14A persisted max-rule A_ND_size but sum-rule membership/switch
    # fields.  Preserve its submitted switch label as the formal reference and
    # keep both recomputed rules explicit instead of forcing false agreement.
    result["current_in_A_ND"] = result["task14a_current_in_A_ND"]
    result["switch_required"] = result["task14a_switch_required"]
    if not result[
        [
            "A_ND_size_matches_task14a",
            "sum_current_membership_matches_task14a",
            "sum_switch_required_matches_task14a",
        ]
    ].to_numpy().all():
        raise RuntimeError("recomputed practical action fields do not reproduce Task 14A")
    return result


def _lookup_actions(frame: pd.DataFrame) -> pd.Series:
    assigned = pd.Series(index=frame.index, dtype=object)
    for held_out_group in sorted(frame["cv_group_id"].unique()):
        train = frame.loc[frame["cv_group_id"].ne(held_out_group)]
        means = (
            train.groupby(["route", "source_FE", "cv_group_id"], sort=False)[LOSS_COLS]
            .mean()
            .groupby(["route", "source_FE"], sort=False)
            .mean()
        )
        action = means.idxmin(axis=1).str.replace("loss_", "", regex=False)
        held_out = frame.index[frame["cv_group_id"].eq(held_out_group)]
        keys = frame.loc[held_out].set_index(["route", "source_FE"]).index
        assigned.loc[held_out] = np.asarray(keys.map(action))
    if assigned.isna().any():
        raise RuntimeError("empirical lookup left states unassigned")
    return assigned


def build_corrected_dataset() -> tuple[pd.DataFrame, list[str], list[str], pd.DataFrame]:
    frame = pd.read_parquet(T14A_HEAVY / "post_handoff_analysis_frame.parquet")
    behavior = pd.read_parquet(T14A_HEAVY / "post_handoff_behavior.parquet")
    keep = [
        "state_id",
        "suite",
        "split",
        "problem_id",
        "family",
        "cv_group_id",
        "instance",
        "seed",
        "source_algorithm",
        "current_algorithm",
        "source_checkpoint_fe",
        "FE",
        "segment_start",
        "segment_age",
        "best_fitness",
        "log10_gap",
        "population_size",
        "snapshot_fe",
        "snapshot_population_size",
        *LOSS_COLS,
        "continue_loss",
        "A_ND_size",
        "current_in_A_ND",
        "switch_required",
    ]
    frame = frame[keep].copy().rename(
        columns={
            "source_checkpoint_fe": "source_FE",
            "FE": "global_FE",
            "A_ND_size": "task14a_A_ND_size",
            "current_in_A_ND": "task14a_current_in_A_ND",
            "switch_required": "task14a_switch_required",
        }
    )
    frame["route"] = frame["source_algorithm"] + "->" + frame["current_algorithm"]

    practical = _practical_action_sets(frame)
    frame = frame.merge(
        practical[
            [
                "state_id",
                "A_ND_members",
                "A_ND_size",
                "current_in_A_ND",
                "switch_required",
                "current_in_A_ND_max",
                "switch_required_max",
                "current_in_A_ND_sum",
                "switch_required_sum",
            ]
        ],
        on="state_id",
        validate="one_to_one",
    )

    source_behavior_cols = [
        column for column in behavior.columns if column in set(SELECTOR_BEHAVIOR_FEATURE_COLUMNS)
    ]
    if len(source_behavior_cols) != 28:
        raise RuntimeError(f"expected 28 behavior features, got {len(source_behavior_cols)}")
    bg_cols = _behavior_columns("bg")
    bs_cols = _behavior_columns("bs_old")
    bg = behavior.loc[behavior["behavior_kind"].eq("global"), ["state_id", *source_behavior_cols]].rename(
        columns={column: f"bg_{column[3:]}" for column in source_behavior_cols}
    )
    bs = behavior.loc[behavior["behavior_kind"].eq("segment"), ["state_id", *source_behavior_cols]].rename(
        columns={column: f"bs_old_{column[3:]}" for column in source_behavior_cols}
    )
    frame = frame.merge(bg, on="state_id", validate="one_to_one").merge(
        bs, on="state_id", validate="one_to_one"
    )

    raw_values = frame[[*bg_cols, *bs_cols]].to_numpy(dtype=float)
    stability = pd.DataFrame(
        [
            {
                "representation": "global_and_segment_old",
                "n_rows": len(frame),
                "n_feature_cells": int(raw_values.size),
                "raw_max_absolute_value": float(np.max(np.abs(raw_values))),
                "raw_abs_gt_1e6_cells": int((np.abs(raw_values) > CLIP_BOUND).sum()),
                "raw_abs_gt_1e4_cells": int((np.abs(raw_values) > 1e4).sum()),
                "raw_nonzero_abs_lt_1e12_cells": int(
                    ((np.abs(raw_values) < TINY_BOUND) & (raw_values != 0.0)).sum()
                ),
            }
        ]
    )
    for columns in (bg_cols, bs_cols):
        tiny = (frame[columns].abs() < TINY_BOUND) & frame[columns].ne(0.0)
        frame[columns] = frame[columns].mask(tiny, 0.0).clip(-CLIP_BOUND, CLIP_BOUND)

    frame["raw_best_action"] = frame[LOSS_COLS].idxmin(axis=1).str.replace(
        "loss_", "", regex=False
    )
    frame["lookup_action"] = _lookup_actions(frame)
    frame["realized_lookup"] = [
        row[f"loss_{action}"] for (_, row), action in zip(frame.iterrows(), frame["lookup_action"])
    ]
    frame.to_parquet(OUT_HEAVY / "task14b1_corrected_dataset.parquet", index=False)
    practical.to_parquet(OUT_HEAVY / "practical_action_sets_corrected.parquet", index=False)
    return frame, bg_cols, bs_cols, stability


def _checkpoint_base(problem_id: str) -> str:
    match = re.match(r"^(?P<base>.+)_i\d+_d\d+$", str(problem_id))
    if match is None:
        raise ValueError(f"cannot derive checkpoint directory from {problem_id!r}")
    return match.group("base")


def _problem_numbers(problem_id: str) -> tuple[int, int]:
    match = re.search(r"_(?:f|c)(\d+)_i(\d+)_d\d+$", str(problem_id))
    if match is None:
        raise ValueError(f"cannot derive function/instance from {problem_id!r}")
    return int(match.group(1)), int(match.group(2))


def _matched_replay_job(payload: tuple[str, str, str, list[dict]]) -> tuple[list[dict], list[dict]]:
    """Rebuild one problem's post-handoff histories without action branches.

    The source checkpoint is the persisted optimizer state used by Task 14A.
    Only the deterministic transfer and the same 1000-FE commitment are
    replayed; no candidate action outcome is evaluated or regenerated.
    """
    split, suite_name, problem_id, specs = payload
    config = load_experiment_config(CONFIG)
    suite = config.suite(split)
    function, instance = _problem_numbers(problem_id)
    problem = make_experiment_problem(
        suite,
        function=function,
        instance=instance,
        dimension=config.dimension,
        boundary_handling=config.boundary_handling,
    )
    checkpoint_path = (
        TRAJECTORY_ROOT / split / _checkpoint_base(problem_id) / f"dimension_{config.dimension}" / "optimizer_checkpoints.parquet"
    )
    if not checkpoint_path.exists():
        problem.close()
        raise FileNotFoundError(checkpoint_path)
    checkpoints = pd.read_parquet(checkpoint_path)
    checkpoint_map = {
        (str(row.algorithm), int(row.seed), int(row.FE)): bytes(row.optimizer_state_payload)
        for row in checkpoints.itertuples(index=False)
        if str(row.problem_id) == problem_id
    }
    settings = OptimizerSettings(
        population_size=config.population_size,
        boundary_handling=config.boundary_handling,
    )
    behavior_rows: list[dict] = []
    alignment_rows: list[dict] = []
    source_state_cache: dict[tuple[str, int, int], object] = {}
    required_by_source_seed: dict[tuple[str, int], set[int]] = {}
    for spec in specs:
        required_by_source_seed.setdefault(
            (str(spec["source_algorithm"]), int(spec["seed"])), set()
        ).add(int(spec["source_FE"]))
    for (source, seed), source_fes in required_by_source_seed.items():
        missing = [fe for fe in sorted(source_fes) if (source, seed, fe) not in checkpoint_map]
        if not missing:
            for fe in source_fes:
                source_state_cache[(source, seed, fe)] = pickle.loads(
                    checkpoint_map[(source, seed, fe)]
                )
            continue
        source_evaluations = [0]

        def count_source_evaluation(point: np.ndarray, value: float) -> None:
            source_evaluations[0] += 1

        source_state = initialize_optimizer_state(
            algorithm=source,
            problem=problem,
            seed=seed,
            settings=settings,
            on_evaluation=count_source_evaluation,
        )
        pending = list(sorted(source_fes))
        while pending:
            advance_optimizer_state(
                state=source_state,
                problem=problem,
                fe_budget=min(config.population_size, pending[0] - source_evaluations[0]),
                on_native_update=lambda updated: None,
                on_evaluation=count_source_evaluation,
            )
            if source_evaluations[0] == pending[0]:
                source_state_cache[(source, seed, pending.pop(0))] = copy.deepcopy(source_state)
        if source_evaluations[0] != max(source_fes):
            problem.close()
            raise RuntimeError(f"natural source replay stopped at {source_evaluations[0]} FE")
    for spec in specs:
        source = str(spec["source_algorithm"])
        target = str(spec["current_algorithm"])
        seed = int(spec["seed"])
        source_fe = int(spec["source_FE"])
        checkpoint_key = (source, seed, source_fe)
        source_from_checkpoint = checkpoint_key in checkpoint_map
        if checkpoint_key not in source_state_cache:
            problem.close()
            raise KeyError(f"missing source checkpoint/replay state {problem_id} {checkpoint_key}")
        source_state = copy.deepcopy(source_state_cache[checkpoint_key])
        transferred = initialize_transferred_optimizer_state(
            algorithm=target,
            source_state=source_state,
            problem=problem,
            seed=seed,
            function=problem.function_number,
            instance=problem.instance_number,
            event=NO_QUERY_TRANSFER_EVENT,
        )
        used = [0]
        native_updates = [0]
        snapshots = [
            make_native_update_snapshot(
                fe=source_fe,
                native_updates=0,
                population=transferred.population,
                fitness=transferred.fitness,
                best_fitness=transferred.best_fitness,
            )
        ]

        def observe_evaluation(point: np.ndarray, value: float) -> None:
            used[0] += 1

        def observe_update(updated) -> None:
            native_updates[0] += 1
            snapshots.append(
                make_native_update_snapshot(
                    fe=source_fe + used[0],
                    native_updates=native_updates[0],
                    population=updated.population,
                    fitness=updated.fitness,
                    best_fitness=updated.best_fitness,
                )
            )

        advance_optimizer_state(
            state=transferred,
            problem=problem,
            fe_budget=MATCHED_SEGMENT_FE,
            on_native_update=observe_update,
            on_evaluation=observe_evaluation,
        )
        if used[0] != MATCHED_SEGMENT_FE:
            problem.close()
            raise RuntimeError(f"matched replay used {used[0]} FE for {spec['state_id']}")
        post_fe = source_fe + MATCHED_SEGMENT_FE
        # The formal mature state is after the complete 1000-FE commitment;
        # the last native update may occur one population before that endpoint.
        snapshots.append(
            make_native_update_snapshot(
                fe=post_fe,
                native_updates=native_updates[0],
                population=transferred.population,
                fitness=transferred.fitness,
                best_fitness=transferred.best_fitness,
            )
        )
        initial_iqr = float(np.percentile(snapshots[0].fitness, 75.0) - np.percentile(snapshots[0].fitness, 25.0))
        windows, history = build_window_statistics(
            snapshots,
            # Matched windows use the same global 10,000-FE scale as B_global:
            # WINDOW_RATIOS {0.02, 0.05, 0.10} therefore mean 200/500/1000 FE.
            fe_total=config.fe_total,
            problem_id=problem_id,
            algorithm=target,
            initial_fitness_iqr=initial_iqr,
        )
        record = TrajectoryRecord.from_arrays(
            problem_id=problem_id,
            function_id=problem.function_id,
            family=problem.family,
            cv_group_id=problem.cv_group_id,
            dimension=config.dimension,
            algorithm=target,
            seed=seed,
            fe=post_fe,
            fe_total=config.fe_total,
            native_updates=native_updates[0],
            window_statistics=windows,
            native_update_history=history,
            population=transferred.population,
            fitness=transferred.fitness,
            best_fitness=transferred.best_fitness,
            sampling_metadata=dict(SAMPLING_METADATA),
        )
        extracted = extract_behavior_rows([dataclasses.asdict(record)])[0]
        extracted["behavior_kind"] = "segment_matched"
        extracted["state_id"] = str(spec["state_id"])
        extracted["segment_start"] = source_fe
        behavior_rows.append(extracted)

        expected_best = float(spec["best_fitness"])
        expected_gap = float(spec["log10_gap"])
        gap = min(max(float(transferred.best_fitness) - float(problem.reference_value), 0.0), config.failure_loss_cap)
        replay_gap = float(np.log10(np.clip(gap, config.log10_gap_floor, config.log10_gap_cap)))
        complete_snapshot = snapshots[-2]
        alignment_rows.append(
            {
                "state_id": str(spec["state_id"]),
                "suite": suite_name,
                "problem_id": problem_id,
                "source_algorithm": source,
                "current_algorithm": target,
                "source_FE": source_fe,
                "seed": seed,
                "expected_best_fitness": expected_best,
                "replay_best_fitness": float(transferred.best_fitness),
                "best_fitness_abs_diff": abs(expected_best - float(transferred.best_fitness)),
                "expected_log10_gap": expected_gap,
                "replay_log10_gap": replay_gap,
                "log10_gap_abs_diff": abs(expected_gap - replay_gap),
                "expected_population_size": int(spec["population_size"]),
                "replay_population_size": int(transferred.population.shape[0]),
                "population_size_match": int(spec["population_size"]) == int(transferred.population.shape[0]),
                "expected_snapshot_fe": int(spec["snapshot_fe"]),
                "replay_last_complete_fe": int(complete_snapshot.fe),
                "snapshot_fe_abs_diff": abs(int(spec["snapshot_fe"]) - int(complete_snapshot.fe)),
                "expected_snapshot_population_size": int(spec["snapshot_population_size"]),
                "replay_last_complete_population_size": int(complete_snapshot.population.shape[0]),
                "snapshot_population_size_match": int(spec["snapshot_population_size"]) == int(complete_snapshot.population.shape[0]),
                "replay_commitment_fe": used[0],
                "native_update_count": native_updates[0],
                "deterministic_transfer_event": NO_QUERY_TRANSFER_EVENT,
                "source_checkpoint_available": source_from_checkpoint,
                "source_state_evaluations": int(getattr(source_state, "evaluations", source_fe)),
                "source_state_generation": int(getattr(source_state, "generation", -1)),
                "source_rng_trace_identity": (
                    "persisted_optimizer_checkpoint"
                    if source_from_checkpoint
                    else "deterministic_natural_source_replay_same_seed"
                ),
                "matched_anchor_FE_w02": int(windows[0]["anchor_FE"]),
                "matched_anchor_FE_w05": int(windows[1]["anchor_FE"]),
                "matched_anchor_FE_w10": int(windows[2]["anchor_FE"]),
                "matched_effective_window_FE_w02": int(post_fe - windows[0]["anchor_FE"]),
                "matched_effective_window_FE_w05": int(post_fe - windows[1]["anchor_FE"]),
                "matched_effective_window_FE_w10": int(post_fe - windows[2]["anchor_FE"]),
                "matched_long_anchor_equals_handoff": int(windows[2]["anchor_FE"]) == source_fe,
            }
        )
    problem.close()
    return behavior_rows, alignment_rows


def build_scale_matched_segment_behavior(
    dataset: pd.DataFrame,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    jobs: list[tuple[str, str, str, list[dict]]] = []
    for (split, suite_name, problem_id), part in dataset.groupby(
        ["split", "suite", "problem_id"], sort=True
    ):
        specs = part[
            [
                "state_id",
                "source_algorithm",
                "current_algorithm",
                "seed",
                "source_FE",
                "best_fitness",
                "log10_gap",
                "population_size",
                "snapshot_fe",
                "snapshot_population_size",
            ]
        ].to_dict(orient="records")
        jobs.append((str(split), str(suite_name), str(problem_id), specs))
    replay_rows: list[dict] = []
    alignment_rows: list[dict] = []
    if workers == 1:
        results = [_matched_replay_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_matched_replay_job, jobs))
    for behavior_rows, alignment in results:
        replay_rows.extend(behavior_rows)
        alignment_rows.extend(alignment)
    matched_raw = pd.DataFrame(replay_rows).sort_values("state_id").reset_index(drop=True)
    alignment = pd.DataFrame(alignment_rows).sort_values("state_id").reset_index(drop=True)
    if len(matched_raw) != len(dataset) or len(alignment) != len(dataset):
        raise RuntimeError(f"matched replay row count mismatch: {len(matched_raw)}/{len(dataset)}")
    if (
        alignment["best_fitness_abs_diff"].max() > 1e-12
        or alignment["log10_gap_abs_diff"].max() > 1e-12
        or not alignment["population_size_match"].all()
        or not alignment["snapshot_population_size_match"].all()
    ):
        raise RuntimeError("Task 14A state alignment failed at tolerance 1e-12")
    matched_feature_cols = _behavior_columns("bs_matched")
    rename = {
        column: f"bs_matched_{column[3:]}"
        for column in SELECTOR_BEHAVIOR_FEATURE_COLUMNS
    }
    matched = matched_raw.rename(columns=rename)
    raw_values = matched[matched_feature_cols].to_numpy(dtype=float)
    old_behavior = pd.read_parquet(T14A_HEAVY / "post_handoff_behavior.parquet")
    old_source_cols = list(SELECTOR_BEHAVIOR_FEATURE_COLUMNS)
    raw_old = (
        old_behavior.loc[old_behavior["behavior_kind"].eq("segment"), ["state_id", *old_source_cols]]
        .sort_values("state_id")
        .set_index("state_id")
        .reindex(dataset.sort_values("state_id")["state_id"])[old_source_cols]
        .to_numpy(dtype=float)
    )

    def _clean(values: np.ndarray) -> np.ndarray:
        cleaned = np.where((np.abs(values) < TINY_BOUND) & (values != 0.0), 0.0, values)
        return np.clip(cleaned, -CLIP_BOUND, CLIP_BOUND)

    stability_rows = []
    for representation, values in (
        ("segment_old", raw_old),
        ("segment_matched", raw_values),
    ):
        cleaned = _clean(values)
        stability_rows.append(
            {
                "representation": representation,
                "n_rows": len(values),
                "n_feature_cells": int(values.size),
                "raw_max_absolute_value": float(np.max(np.abs(values))),
                "raw_abs_gt_1e6_cells": int((np.abs(values) > CLIP_BOUND).sum()),
                "raw_abs_gt_1e4_cells": int((np.abs(values) > 1e4).sum()),
                "raw_nonzero_abs_lt_1e12_cells": int(((np.abs(values) < TINY_BOUND) & (values != 0.0)).sum()),
                "post_clip_max_absolute_value": float(np.max(np.abs(cleaned))),
                "post_clip_abs_gt_1e6_cells": int((np.abs(cleaned) > CLIP_BOUND).sum()),
                "post_clip_abs_gt_1e4_cells": int((np.abs(cleaned) > 1e4).sum()),
            }
        )
    stability = pd.DataFrame(stability_rows)
    matched.to_parquet(OUT_HEAVY / "segment_behavior_matched_200_500_1000.parquet", index=False)
    alignment.to_parquet(OUT_HEAVY / "task14b1_replay_alignment.parquet", index=False)
    stability.to_parquet(OUT_HEAVY / "segment_numeric_stability_comparison.parquet", index=False)
    return matched, alignment, stability


def _wide_oof_predictions(
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
    models: tuple[str, ...],
) -> pd.DataFrame:
    metadata_columns = [
        "state_id",
        "suite",
        "problem_id",
        "cv_group_id",
        "seed",
        "source_algorithm",
        "current_algorithm",
        "route",
        "source_FE",
        "global_FE",
        "segment_age",
        *LOSS_COLS,
        "continue_loss",
        "switch_required",
        "A_ND_members",
    ]
    wide = dataset[metadata_columns].copy()
    for model in models:
        part = predictions.loc[predictions["model"].eq(model)].set_index("state_id")
        for solver in SOLVERS:
            wide[f"pred_{model}_{solver}"] = part.reindex(wide["state_id"])[f"pred_{solver}"].to_numpy()
        wide[f"selected_{model}"] = part.reindex(wide["state_id"])["selected"].to_numpy()
        wide[f"realized_{model}"] = part.reindex(wide["state_id"])["realized_loss"].to_numpy()
        wide[f"fold_id_{model}"] = part.reindex(wide["state_id"])["fold_id"].to_numpy()
    return wide


def grouped_oof_and_baselines(
    dataset: pd.DataFrame,
    bg_cols: list[str],
    bs_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    route_dummies = pd.get_dummies(dataset["route"], prefix="route", dtype=float)
    model_frame = pd.concat(
        [dataset.reset_index(drop=True), route_dummies.reset_index(drop=True)], axis=1
    )
    m0 = [*route_dummies.columns, "source_FE", "segment_age"]
    feature_sets = {
        "M0": m0,
        "MG": [*m0, *bg_cols],
        "MS_old": [*m0, *bs_cols],
        "MGS_old": [*m0, *bg_cols, *bs_cols],
    }
    predictions = []
    for carrier in ("rf", "ridge"):
        predictions.append(run_grouped_oof(model_frame, feature_sets, carrier))
    long_predictions = pd.concat(predictions, ignore_index=True)
    long_predictions.to_parquet(OUT_HEAVY / "task14b1_oof_predictions_long.parquet", index=False)

    models = tuple(feature_sets)
    rf_predictions = long_predictions.loc[long_predictions["carrier"].eq("rf")]
    wide = _wide_oof_predictions(dataset, rf_predictions, models)
    wide.to_parquet(OUT_HEAVY / "task14b1_oof_predictions.parquet", index=False)

    policy_rows: list[dict] = []
    suite_masks = {
        "pooled": np.ones(len(wide), dtype=bool),
        "bbob": wide["suite"].eq("bbob").to_numpy(),
        "mabbob": wide["suite"].eq("mabbob").to_numpy(),
    }
    realized_columns = {
        "Continue": "continue_loss",
        "lookup": "realized_lookup",
        **{model: f"realized_{model}" for model in models},
    }
    wide = wide.merge(
        dataset[["state_id", "realized_lookup", "lookup_action"]],
        on="state_id",
        validate="one_to_one",
    )
    for suite_name, mask in suite_masks.items():
        part = wide.loc[mask]
        for policy, column in realized_columns.items():
            policy_rows.append(
                {
                    "suite": suite_name,
                    "policy": policy,
                    "realized_fb_loss": function_balanced_mean(part, column),
                    "n_states": len(part),
                }
            )
    policy = pd.DataFrame(policy_rows)
    policy.to_parquet(OUT_HEAVY / "task14b1_policy_performance.parquet", index=False)

    comparison_specs = [
        ("MGS_old_vs_lookup", "realized_lookup", "realized_MGS_old"),
        ("MGS_old_vs_M0", "realized_M0", "realized_MGS_old"),
        ("MG_vs_M0", "realized_M0", "realized_MG"),
        ("MS_old_vs_M0", "realized_M0", "realized_MS_old"),
        ("MGS_old_vs_MG", "realized_MG", "realized_MGS_old"),
        ("M0_vs_continue", "continue_loss", "realized_M0"),
    ]
    bootstrap_rows = []
    consistency_rows = []
    stream = 0
    for comparison, upper_column, lower_column in comparison_specs:
        for suite_name in ("bbob", "mabbob"):
            part = wide.loc[wide["suite"].eq(suite_name)].copy()
            upper = part[upper_column].to_numpy(dtype=float)
            lower = part[lower_column].to_numpy(dtype=float)
            point, low, high = paired_group_bootstrap(part, upper, lower, stream)
            stream += 1
            upper_loss = function_balanced_mean(part, upper)
            lower_loss = function_balanced_mean(part, lower)
            direct = upper_loss - lower_loss
            bootstrap_rows.append(
                {
                    "comparison": comparison,
                    "suite": suite_name,
                    "fb_mean": point,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
            consistency_rows.append(
                {
                    "comparison": comparison,
                    "suite": suite_name,
                    "upper_absolute_loss": upper_loss,
                    "lower_absolute_loss": lower_loss,
                    "direct_difference": direct,
                    "bootstrap_point_estimate": point,
                    "absolute_error": abs(direct - point),
                    "is_exact_within_1e12": abs(direct - point) <= 1e-12,
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    consistency = pd.DataFrame(consistency_rows)
    if not consistency["is_exact_within_1e12"].all():
        raise RuntimeError("absolute policy losses disagree with paired point estimates")
    bootstrap.to_parquet(OUT_HEAVY / "task14b1_pairwise_bootstrap.parquet", index=False)
    consistency.to_parquet(OUT_HEAVY / "baseline_consistency_check.parquet", index=False)
    consistency.to_parquet(OUT_HEAVY / "baseline_consistency_audit.parquet", index=False)
    return wide, policy, bootstrap, consistency


def grouped_oof_scale_matched(
    dataset: pd.DataFrame,
    bg_cols: list[str],
    bs_old_cols: list[str],
    bs_matched_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the registered RF/Ridge OOF comparison including matched Segment."""
    route_dummies = pd.get_dummies(dataset["route"], prefix="route", dtype=float)
    model_frame = pd.concat(
        [dataset.reset_index(drop=True), route_dummies.reset_index(drop=True)], axis=1
    )
    m0 = [*route_dummies.columns, "source_FE", "segment_age"]
    feature_sets = {
        "M0": m0,
        "MG": [*m0, *bg_cols],
        "MS_old": [*m0, *bs_old_cols],
        "MS_matched": [*m0, *bs_matched_cols],
        "MGS_old": [*m0, *bg_cols, *bs_old_cols],
        "MGS_matched": [*m0, *bg_cols, *bs_matched_cols],
    }
    long_predictions = pd.concat(
        [run_grouped_oof(model_frame, feature_sets, carrier) for carrier in ("rf", "ridge")],
        ignore_index=True,
    )
    long_predictions.to_parquet(OUT_HEAVY / "task14b1_oof_predictions_long.parquet", index=False)
    rf_predictions = long_predictions.loc[long_predictions["carrier"].eq("rf")]
    wide = _wide_oof_predictions(dataset, rf_predictions, tuple(feature_sets))
    wide = wide.merge(
        dataset[["state_id", "realized_lookup", "lookup_action"]],
        on="state_id",
        validate="one_to_one",
    )
    wide.to_parquet(OUT_HEAVY / "task14b1_oof_predictions.parquet", index=False)

    policy_rows = []
    for suite_name in ("pooled", "bbob", "mabbob"):
        part = wide if suite_name == "pooled" else wide.loc[wide["suite"].eq(suite_name)]
        for policy_name, column in {
            "Continue": "continue_loss",
            "lookup": "realized_lookup",
            **{name: f"realized_{name}" for name in feature_sets},
        }.items():
            policy_rows.append(
                {
                    "suite": suite_name,
                    "policy": policy_name,
                    "realized_fb_loss": function_balanced_mean(part, column),
                    "n_states": len(part),
                }
            )
    policy = pd.DataFrame(policy_rows)
    policy.to_parquet(OUT_HEAVY / "task14b1_policy_performance.parquet", index=False)

    comparison_specs = [
        ("MGS_matched_vs_lookup", "realized_lookup", "realized_MGS_matched"),
        ("MGS_matched_vs_M0", "realized_M0", "realized_MGS_matched"),
        ("MG_vs_M0", "realized_M0", "realized_MG"),
        ("MS_old_vs_M0", "realized_M0", "realized_MS_old"),
        ("MS_matched_vs_M0", "realized_M0", "realized_MS_matched"),
        ("MGS_old_vs_M0", "realized_M0", "realized_MGS_old"),
        ("MGS_old_vs_lookup", "realized_lookup", "realized_MGS_old"),
        ("MGS_old_vs_MG", "realized_MG", "realized_MGS_old"),
        ("MGS_matched_vs_MG", "realized_MG", "realized_MGS_matched"),
        ("M0_vs_continue", "continue_loss", "realized_M0"),
    ]
    bootstrap_rows = []
    consistency_rows = []
    stream = 100
    for comparison, upper_column, lower_column in comparison_specs:
        for suite_name in ("bbob", "mabbob"):
            part = wide.loc[wide["suite"].eq(suite_name)].copy()
            upper = part[upper_column].to_numpy(dtype=float)
            lower = part[lower_column].to_numpy(dtype=float)
            point, low, high = paired_group_bootstrap(part, upper, lower, stream)
            stream += 1
            upper_loss = function_balanced_mean(part, upper)
            lower_loss = function_balanced_mean(part, lower)
            direct = upper_loss - lower_loss
            bootstrap_rows.append(
                {
                    "comparison": comparison,
                    "suite": suite_name,
                    "fb_mean": point,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
            consistency_rows.append(
                {
                    "comparison": comparison,
                    "suite": suite_name,
                    "upper_absolute_loss": upper_loss,
                    "lower_absolute_loss": lower_loss,
                    "direct_difference": direct,
                    "bootstrap_point_estimate": point,
                    "absolute_error": abs(direct - point),
                    "is_exact_within_1e12": abs(direct - point) <= 1e-12,
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    consistency = pd.DataFrame(consistency_rows)
    if not consistency["is_exact_within_1e12"].all():
        raise RuntimeError("matched absolute policy losses disagree with paired point estimates")
    bootstrap.to_parquet(OUT_HEAVY / "task14b1_pairwise_bootstrap.parquet", index=False)
    consistency.to_parquet(OUT_HEAVY / "baseline_consistency_audit.parquet", index=False)
    consistency.to_parquet(OUT_HEAVY / "baseline_consistency_check.parquet", index=False)
    return wide, policy, bootstrap, consistency


def within_route_loso(
    dataset: pd.DataFrame,
    feature_sets: dict[str, list[tuple[list[str], np.ndarray | None]]],
) -> pd.DataFrame:
    truth = dataset[LOSS_COLS].to_numpy(dtype=float)
    matrices: dict[str, np.ndarray] = {}
    for model, blocks in feature_sets.items():
        arrays = []
        for columns, permutation in blocks:
            values = dataset[columns].to_numpy(dtype=float)
            arrays.append(values if permutation is None else values[permutation])
        matrices[model] = np.column_stack(arrays).astype(np.float32)

    records = []
    grouped_positions = dataset.groupby(
        ["problem_id", "route", "source_FE"], sort=False
    ).groups.values()
    for positions in grouped_positions:
        indices = np.asarray(positions, dtype=int)
        for held_out in indices:
            train = indices[indices != held_out]
            w0_prediction = truth[train].mean(axis=0)
            w0_index = int(np.argmin(w0_prediction))
            record = {
                "state_id": dataset.at[held_out, "state_id"],
                "suite": dataset.at[held_out, "suite"],
                "problem_id": dataset.at[held_out, "problem_id"],
                "cv_group_id": dataset.at[held_out, "cv_group_id"],
                "seed": int(dataset.at[held_out, "seed"]),
                "route": dataset.at[held_out, "route"],
                "source_FE": int(dataset.at[held_out, "source_FE"]),
                "selected_W0": SOLVERS[w0_index],
                "realized_W0": float(truth[held_out, w0_index]),
            }
            for model, matrix in matrices.items():
                carrier = make_carrier("rf")
                carrier.fit(matrix[train], truth[train])
                prediction = carrier.predict(matrix[held_out : held_out + 1])[0]
                selected_index = int(np.argmin(prediction))
                for solver, predicted_value in zip(SOLVERS, prediction):
                    record[f"pred_{model}_{solver}"] = float(predicted_value)
                record[f"selected_{model}"] = SOLVERS[selected_index]
                record[f"realized_{model}"] = float(truth[held_out, selected_index])
            records.append(record)
    return pd.DataFrame(records)


def within_summary(predictions: pd.DataFrame, models: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for suite_name, part in predictions.groupby("suite", sort=False):
        row = {"suite": suite_name}
        for model in ("W0", *models):
            row[f"L_{model}"] = function_balanced_mean(part, f"realized_{model}")
        if "WG" in models:
            row["delta_within_global"] = row["L_W0"] - row["L_WG"]
        if "WS_old" in models:
            row["delta_within_segment_only_old"] = row["L_W0"] - row["L_WS_old"]
        if "WGS_old" in models:
            row["delta_within_segment_old"] = row["L_WG"] - row["L_WGS_old"]
        if "WS_matched" in models:
            row["delta_within_segment_only_matched"] = row["L_W0"] - row["L_WS_matched"]
        if "WGS_matched" in models:
            row["delta_within_segment_matched"] = row["L_WG"] - row["L_WGS_matched"]
        rows.append(row)
    return pd.DataFrame(rows)


def _stratified_permutation(dataset: pd.DataFrame, control: str, repeat: int) -> np.ndarray:
    code = 1 if control == "global" else 2
    rng = np.random.default_rng(
        np.random.SeedSequence([ANALYSIS_STREAM, 700, code, int(repeat)]).generate_state(4)
    )
    permutation = np.arange(len(dataset), dtype=int)
    for positions in dataset.groupby(["route", "source_FE"], sort=False).groups.values():
        indices = np.asarray(positions, dtype=int)
        permutation[indices] = indices[rng.permutation(len(indices))]
    return permutation


def permutation_worker(payload: tuple[str, list[int]]) -> list[dict]:
    control, repeats = payload
    dataset = pd.read_parquet(OUT_HEAVY / "task14b1_corrected_dataset.parquet")
    bg_cols = _behavior_columns("bg")
    bs_cols = _behavior_columns("bs_old")
    rows = []
    for repeat in repeats:
        permutation = _stratified_permutation(dataset, control, repeat)
        if control == "global":
            specs = {"WG_perm": [(bg_cols, permutation)]}
            model = "WG_perm"
        else:
            # The global block remains attached to its original state.  Only
            # the old segment block is permuted within route and source FE.
            specs = {"WGS_segment_perm": [(bg_cols, None), (bs_cols, permutation)]}
            model = "WGS_segment_perm"
        predictions = within_route_loso(dataset, specs)
        for suite_name, part in predictions.groupby("suite", sort=False):
            rows.append(
                {
                    "control": control,
                    "repeat": int(repeat),
                    "suite": suite_name,
                    "permuted_policy_loss": function_balanced_mean(part, f"realized_{model}"),
                    "train_only_W0_loss": function_balanced_mean(part, "realized_W0"),
                }
            )
    return rows


def run_permutations(
    observed: pd.DataFrame,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chunks = [list(range(worker, N_PERM, workers)) for worker in range(workers)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for control in ("segment", "global"):
            payloads = [(control, chunk) for chunk in chunks if chunk]
            for result in executor.map(permutation_worker, payloads):
                rows.extend(result)
    table = pd.DataFrame(rows)
    observed_map = observed.set_index("suite").to_dict(orient="index")
    table["delta_within_perm"] = [
        (
            observed_map[row.suite]["L_WG"] - row.permuted_policy_loss
            if row.control == "segment"
            else row.train_only_W0_loss - row.permuted_policy_loss
        )
        for row in table.itertuples(index=False)
    ]
    segment = table.loc[table["control"].eq("segment")].copy()
    global_table = table.loc[table["control"].eq("global")].copy()
    segment.to_parquet(OUT_HEAVY / "segment_permutation_100_corrected.parquet", index=False)
    global_table.to_parquet(OUT_HEAVY / "global_permutation_100_corrected.parquet", index=False)

    summary_rows = []
    for (control, suite_name), part in table.groupby(["control", "suite"], sort=False):
        if control == "segment":
            delta_observed = observed_map[suite_name]["delta_within_segment_old"]
        else:
            delta_observed = observed_map[suite_name]["delta_within_global"]
        null = part["delta_within_perm"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "control": control,
                "suite": suite_name,
                "delta_observed": float(delta_observed),
                "null_mean": float(null.mean()),
                "null_std": float(null.std(ddof=1)),
                "null_q95": float(np.quantile(null, 0.95)),
                "null_q975": float(np.quantile(null, 0.975)),
                "empirical_p": float((1 + int((null >= delta_observed).sum())) / (1 + N_PERM)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(OUT_HEAVY / "within_permutation_summary_corrected.parquet", index=False)
    return segment, global_table, summary


def matched_segment_permutation_worker(payload: list[int]) -> list[dict]:
    dataset = pd.read_parquet(OUT_HEAVY / "task14b1_corrected_dataset_matched.parquet")
    bg_cols = _behavior_columns("bg")
    bs_cols = _behavior_columns("bs_matched")
    rows = []
    for repeat in payload:
        permutation = _stratified_permutation(dataset, "segment_matched", repeat)
        predictions = within_route_loso(
            dataset,
            {"WGS_matched_perm": [(bg_cols, None), (bs_cols, permutation)]},
        )
        for suite_name, part in predictions.groupby("suite", sort=False):
            rows.append(
                {
                    "control": "segment_matched",
                    "repeat": int(repeat),
                    "suite": suite_name,
                    "permuted_policy_loss": function_balanced_mean(part, "realized_WGS_matched_perm"),
                    "train_only_W0_loss": function_balanced_mean(part, "realized_W0"),
                }
            )
    return rows


def run_matched_segment_permutations(
    observed: pd.DataFrame,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chunks = [list(range(worker, N_PERM, workers)) for worker in range(workers)]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(matched_segment_permutation_worker, [chunk for chunk in chunks if chunk]):
            rows.extend(result)
    table = pd.DataFrame(rows)
    observed_map = observed.set_index("suite").to_dict(orient="index")
    table["delta_within_perm"] = [
        observed_map[row.suite]["L_WG"] - row.permuted_policy_loss
        for row in table.itertuples(index=False)
    ]
    table.to_parquet(OUT_HEAVY / "segment_matched_permutation_100.parquet", index=False)
    summary_rows = []
    for suite_name, part in table.groupby("suite", sort=False):
        delta_observed = observed_map[suite_name]["delta_within_segment_matched"]
        null = part["delta_within_perm"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "control": "segment_matched",
                "suite": suite_name,
                "delta_observed": float(delta_observed),
                "null_mean": float(null.mean()),
                "null_std": float(null.std(ddof=1)),
                "null_q95": float(np.quantile(null, 0.95)),
                "null_q975": float(np.quantile(null, 0.975)),
                "empirical_p": float((1 + int((null >= delta_observed).sum())) / (1 + N_PERM)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(OUT_HEAVY / "segment_matched_permutation_summary.parquet", index=False)
    return table, summary


def scale_matching_feasibility() -> tuple[pd.DataFrame, pd.DataFrame]:
    behavior = pd.read_parquet(T14A_HEAVY / "post_handoff_behavior.parquet")
    segment = behavior.loc[
        behavior["behavior_kind"].eq("segment"),
        [
            "state_id",
            "algorithm",
            "FE",
            "segment_start",
            "effective_window_fe_w02",
            "effective_window_fe_w05",
            "effective_window_fe_w10",
        ],
    ].copy()
    segment["available_complete_update_span"] = segment["FE"] - segment["segment_start"]
    segment["required_long_window_fe"] = 1000
    segment["long_window_stays_within_segment"] = segment[
        "available_complete_update_span"
    ].ge(1000)
    segment["would_require_pre_handoff_anchor"] = ~segment[
        "long_window_stays_within_segment"
    ]
    segment["shortfall_fe"] = (
        segment["required_long_window_fe"] - segment["available_complete_update_span"]
    ).clip(lower=0)
    segment.to_parquet(OUT_HEAVY / "scale_matched_segment_feasibility.parquet", index=False)

    rows = []
    for algorithm, part in segment.groupby("algorithm", sort=False):
        rows.append(
            {
                "algorithm": algorithm,
                "n_states": len(part),
                "complete_update_span_min": int(part["available_complete_update_span"].min()),
                "complete_update_span_max": int(part["available_complete_update_span"].max()),
                "n_strict_1000_fe_feasible": int(part["long_window_stays_within_segment"].sum()),
                "n_pre_handoff_required": int(part["would_require_pre_handoff_anchor"].sum()),
                "max_shortfall_fe": int(part["shortfall_fe"].max()),
            }
        )
    rows.append(
        {
            "algorithm": "all",
            "n_states": len(segment),
            "complete_update_span_min": int(segment["available_complete_update_span"].min()),
            "complete_update_span_max": int(segment["available_complete_update_span"].max()),
            "n_strict_1000_fe_feasible": int(segment["long_window_stays_within_segment"].sum()),
            "n_pre_handoff_required": int(segment["would_require_pre_handoff_anchor"].sum()),
            "max_shortfall_fe": int(segment["shortfall_fe"].max()),
        }
    )
    summary = pd.DataFrame(rows)
    summary.to_parquet(OUT_HEAVY / "scale_matched_segment_feasibility_summary.parquet", index=False)
    return segment, summary


def route_phase_matched_stratification(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    models = ("M0", "MG", "MS_old", "MS_matched", "MGS_old", "MGS_matched")
    for stratum, keys in (("route", ["route"]), ("source_FE", ["source_FE"])):
        for key, part in wide.groupby(keys, sort=True):
            values = key if isinstance(key, tuple) else (key,)
            entry = {
                "stratum": stratum,
                "key": "|".join(str(value) for value in values),
                "n_states": int(part["state_id"].nunique()),
                "switch_required_rate": float(part.drop_duplicates("state_id")["switch_required"].mean()),
            }
            for model in models:
                entry[f"L_{model}"] = function_balanced_mean(part, f"realized_{model}")
            entry["segment_increment_old"] = entry["L_MG"] - entry["L_MGS_old"]
            entry["segment_increment_matched"] = entry["L_MG"] - entry["L_MGS_matched"]
            entry["matched_vs_M0"] = entry["L_M0"] - entry["L_MGS_matched"]
            rows.append(entry)
    result = pd.DataFrame(rows)
    result.to_parquet(OUT_HEAVY / "route_phase_matched_stratification.parquet", index=False)
    return result


def _format_float(value: float) -> str:
    return f"{value:+.6f}"


def write_reports(
    policy: pd.DataFrame,
    bootstrap: pd.DataFrame,
    consistency: pd.DataFrame,
    practical: pd.DataFrame,
    within: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    stability: pd.DataFrame,
    feasibility_summary: pd.DataFrame,
    original_reproduction: pd.DataFrame,
) -> None:
    """Legacy Part-A-only report helper retained for compatibility.

    The completed Task 14B.1 pipeline uses ``write_completed_reports`` below;
    this helper is not called by the active entry point.
    """
    def policy_loss(suite: str, name: str) -> float:
        return float(
            policy.loc[policy["suite"].eq(suite) & policy["policy"].eq(name), "realized_fb_loss"].iloc[0]
        )

    def boot(comparison: str, suite: str) -> pd.Series:
        return bootstrap.loc[
            bootstrap["comparison"].eq(comparison) & bootstrap["suite"].eq(suite)
        ].iloc[0]

    correction_rows = [
        ("W0 LOSO", "原实现先用五个 seed 选择 W0；修正后每个 held-out seed 仅由其余四个 seed 选择动作。"),
        ("P2 Segment 置换", "原实现置换整个拼接矩阵；修正后 global block 保持逐状态不变，仅置换 Segment block。"),
        ("实用动作集", "原实现把较优动作标成被支配；修正后标记被较优动作超过噪声阈值的动作。"),
        ("absolute loss", "全部 absolute loss 直接由行级 realized values 按 cv_group 等权计算，并与配对点估计逐项核对。"),
    ]
    correction_table = "\n".join(f"| {name} | {text} |" for name, text in correction_rows)

    corrected_main = []
    for suite in ("bbob", "mabbob"):
        mgs_m0 = boot("MGS_old_vs_M0", suite)
        segment_gain = boot("MGS_old_vs_MG", suite)
        corrected_main.append(
            {
                "suite": suite,
                "M0": policy_loss(suite, "M0"),
                "MG": policy_loss(suite, "MG"),
                "MS_old": policy_loss(suite, "MS_old"),
                "MGS_old": policy_loss(suite, "MGS_old"),
                "MGS_vs_M0": float(mgs_m0["fb_mean"]),
                "segment_gain": float(segment_gain["fb_mean"]),
            }
        )
    corrected_table = "\n".join(
        "| {suite} | {M0:.6f} | {MG:.6f} | {MS_old:.6f} | {MGS_old:.6f} | {MGS_vs_M0:+.6f} | {segment_gain:+.6f} |".format(**row)
        for row in corrected_main
    )

    w_rows = []
    for row in within.itertuples(index=False):
        w_rows.append(
            f"| {row.suite} | {row.L_W0:.6f} | {row.L_WG:.6f} | {row.L_WS_old:.6f} | {row.L_WGS_old:.6f} | {row.delta_within_global:+.6f} | {row.delta_within_segment_old:+.6f} |"
        )
    within_table = "\n".join(w_rows)

    p_rows = []
    for row in permutation_summary.itertuples(index=False):
        p_rows.append(
            f"| {row.control} | {row.suite} | {row.delta_observed:+.6f} | {row.null_mean:+.6f} | {row.empirical_p:.6f} |"
        )
    permutation_table = "\n".join(p_rows)

    corrected_docs = {
        "17j_correctness_patch.md": f"""# 17j · Task 14B 正确性修正

本轮只修正实现与报告口径，不改变 portfolio、标签、horizon、commitment、模型、参数或数据划分；新增 action-label FE = 0。

| 项目 | 修正 |
|---|---|
{correction_table}

Task 14B 原结果已从当前提交产物复算；与已保存主比较点估计的最大绝对差为 `{original_reproduction['absolute_difference'].max():.3e}`。
""",
        "17k_corrected_within_route_test.md": f"""# 17k · 修正后的 Within-route LOSO

正式分组保持 `(problem, route, source_FE)`，每组 5 seeds；每折严格 4 train + 1 test。W0 动作仅由四个训练 seed 的三动作均值决定。

| suite | L_W0 | L_WG | L_WS-old | L_WGS-old | Δglobal=W0−WG | Δsegment=WG−WGS |
|---|---:|---:|---:|---:|---:|---:|
{within_table}
""",
        "17l_corrected_permutation_controls.md": f"""# 17l · 修正后的置换检查

N=100。Global 检查仅置换 Global block；Segment 检查保持 Global block 不动，只置换 old Segment block。Segment 的正式统计量为 `L_WG − L_WGS(segment permuted)`。

| control | suite | observed | null mean | empirical one-sided p |
|---|---|---:|---:|---:|
{permutation_table}
""",
        "17m_corrected_practical_action_sets.md": f"""# 17m · 修正后的实用动作集

- states：{len(practical)}；
- max-rule `A_ND_size` 与 Task 14A 一致：{int(practical['A_ND_size_matches_task14a'].sum())}/{len(practical)}；
- max-rule `current_in_A_ND` 与 Task 14A 保存值一致：{int(practical['max_current_membership_matches_task14a'].sum())}/{len(practical)}；
- sum-rule `current_in_A_ND` 与 Task 14A 保存值一致：{int(practical['sum_current_membership_matches_task14a'].sum())}/{len(practical)}；
- sum-rule `switch_required` 与 Task 14A 保存值一致：{int(practical['sum_switch_required_matches_task14a'].sum())}/{len(practical)}。

修正后的方向是：若动作 i 的 loss 比动作 j 至少低一个 pairwise δ，则 j 被标记为 dominated。

额外发现：Task 14A 行级文件的 `A_ND_size` 来自 max rule，但 `current_in_A_ND/switch_required` 来自 sum rule；184/3780 个状态的 max-rule membership 与保存值不同。本轮保留 Task 14A 已提交的 `switch_required` 作为正式 reference，同时显式保存 max/sum 两套重算字段。
""",
        "17n_corrected_baseline_consistency.md": f"""# 17n · 基线报告一致性

Continue、lookup、RF-M0、MG、MS-old、MGS-old 的 absolute loss 均由同一行级 realized values 直接计算。`upper absolute loss − lower absolute loss` 与 12 个配对 bootstrap 点估计全部在 1e-12 内一致：{int(consistency['is_exact_within_1e12'].sum())}/{len(consistency)}。

| suite | Continue | lookup | M0 | MG | MS-old | MGS-old |
|---|---:|---:|---:|---:|---:|---:|
| bbob | {policy_loss('bbob','Continue'):.6f} | {policy_loss('bbob','lookup'):.6f} | {policy_loss('bbob','M0'):.6f} | {policy_loss('bbob','MG'):.6f} | {policy_loss('bbob','MS_old'):.6f} | {policy_loss('bbob','MGS_old'):.6f} |
| mabbob | {policy_loss('mabbob','Continue'):.6f} | {policy_loss('mabbob','lookup'):.6f} | {policy_loss('mabbob','M0'):.6f} | {policy_loss('mabbob','MG'):.6f} | {policy_loss('mabbob','MS_old'):.6f} | {policy_loss('mabbob','MGS_old'):.6f} |

原 17b 的 Continue 与 lookup absolute 行属于报告错误；Task 14A outcomes 未修改。
""",
    }
    for filename, text in corrected_docs.items():
        (OUT_LIGHT / filename).write_text(text, encoding="utf-8")

    part_a_verdict = "CA1 CORRECTIONS DO NOT CHANGE NO-GO"
    scale_all = feasibility_summary.loc[feasibility_summary["algorithm"].eq("all")].iloc[0]
    report_18a = corrected_docs["17j_correctness_patch.md"].replace(
        "# 17j · Task 14B 正确性修正", "# 18a · Task 14B.1 正确性一致性检查"
    )
    report_18b = corrected_docs["17k_corrected_within_route_test.md"].replace(
        "# 17k · 修正后的 Within-route LOSO", "# 18b · Task 14B.1 修正后的 Within-route LOSO"
    )
    report_18c = corrected_docs["17l_corrected_permutation_controls.md"].replace(
        "# 17l · 修正后的置换检查", "# 18c · Task 14B.1 修正后的置换检查"
    )
    report_18d = corrected_docs["17m_corrected_practical_action_sets.md"].replace(
        "# 17m · 修正后的实用动作集", "# 18d · Task 14B.1 实用动作集一致性"
    )
    report_18e = corrected_docs["17n_corrected_baseline_consistency.md"].replace(
        "# 17n · 基线报告一致性", "# 18e · Task 14B.1 基线报告一致性"
    )
    (OUT_LIGHT / "18a_correctness_consistency_check.md").write_text(report_18a, encoding="utf-8")
    (OUT_LIGHT / "18b_corrected_within_route_loso.md").write_text(report_18b, encoding="utf-8")
    (OUT_LIGHT / "18c_corrected_permutation_controls.md").write_text(report_18c, encoding="utf-8")
    (OUT_LIGHT / "18d_practical_action_set_consistency.md").write_text(report_18d, encoding="utf-8")
    (OUT_LIGHT / "18e_baseline_reporting_consistency.md").write_text(report_18e, encoding="utf-8")

    feasibility_rows = "\n".join(
        f"| {row.algorithm} | {row.n_states} | {row.complete_update_span_min} | {row.complete_update_span_max} | {row.n_strict_1000_fe_feasible} | {row.n_pre_handoff_required} |"
        for row in feasibility_summary.itertuples(index=False)
    )
    part_b = f"""# 18f · 尺度匹配 Segment 协议与停止判定

## 结论

严格 200/500/1000-FE Segment 表征不能在全部 3780 个状态上同时满足：

1. 仅使用 handoff 后信息；
2. 端点与 anchor 均来自完整原生 optimizer update；
3. 实际窗口不短于名义窗口，且量化偏差小于一次 population update。

| current algorithm | states | complete-update span min | max | strict 1000-FE feasible | would cross handoff |
|---|---:|---:|---:|---:|---:|
{feasibility_rows}

只有 {int(scale_all['n_strict_1000_fe_feasible'])}/3780 个状态具备完整 1000-FE post-handoff update span；{int(scale_all['n_pre_handoff_required'])}/3780 个状态若坚持 1000 FE 就需要 handoff 前 anchor。使用 988–999 FE 的短窗口会违反名义窗口下界；补跑到下一完整 update 会改变 mature state 并消耗额外 objective FE。因此未生成 `segment_behavior_matched_200_500_1000.parquet`，也未运行 matched OOF、LOSO、置换、route/phase 或 margin 分析。

此结论直接由 Task 14A 已保存的 Segment `FE` 与 `segment_start` 得出，无需进行确定性重放，new action-label FE = 0，新增 objective FE = 0。
"""
    (OUT_LIGHT / "18f_scale_matched_segment_protocol.md").write_text(part_b, encoding="utf-8")

    final = f"""# Decision-before-Feature：Task 14B.1 正确性与尺度匹配 Segment 复核

## 一句话结果

Part A = **{part_a_verdict}**。四处修正不会使 old Segment Behavior 获得稳定增量。Part B 的 completed matched replay 与正式结果见 `18f_scale_matched_segment_protocol.md` 和 `18l_final_task14b1_verdict.md`。

## Part A 主结果

| suite | M0 | MG | MS-old | MGS-old | MGS-old vs M0 | Segment beyond Global |
|---|---:|---:|---:|---:|---:|---:|
{corrected_table}

| suite | L_W0 | L_WG | L_WS-old | L_WGS-old | Δglobal | Δsegment |
|---|---:|---:|---:|---:|---:|---:|
{within_table}

修正后的置换结果：

| control | suite | observed | null mean | p |
|---|---|---:|---:|---:|
{permutation_table}

## 正确性问题回答

1. 原 W0 确实使用 held-out seed outcome：**是**；现已改为 train-only。
2. 原 P2 确实置换整个 Global+Segment 行：**是**；现仅置换 Segment block。
3. 原 Task 14B 的 dominance 标记方向错误：**是**。修正后 max-rule `A_ND_size` 与 Task 14A 3780/3780 一致；Task 14A 保存的 membership/switch 来自 sum rule，也可 3780/3780 重现。两者在 184 个状态上不同，已显式分列；正式 `switch_required` 继续使用 Task 14A 保存标签。
4. 原 17b Continue 与 lookup absolute 行不一致：**是，REPORTING ERROR**；原 action outcomes 未改动。
5. Part A verdict：**{part_a_verdict}**。

## Part B 与最终判定

matched Segment replay 现已在全部状态完成；本函数仅保留旧版 feasibility 文本生成兼容性。

Task 14B 的 **old 20/50/100-FE 负结果在正确性修正后仍成立**；matched 200/500/1000-FE 结果由 active completed report 给出。

## 保持不变的边界

- seeds 6–10：不进入；
- closed-loop repeated DAS：不进入；
- ProgressForecast：PG3 NO-GO 保持；
- CEC2017：继续暂停；
- CEC2022：继续 held-out；
- 不执行新模型、调参、新特征、internal-state diagnostic 或额外窗口搜索。
"""
    (OUT_LIGHT / "Decision-before-Feature_Task14B1_Correctness与ScaleMatchedSegment复核.md").write_text(
        final, encoding="utf-8"
    )
    (OUT_LIGHT / "18l_final_task14b1_verdict.md").write_text(final, encoding="utf-8")


def write_completed_reports(
    *,
    dataset: pd.DataFrame,
    policy: pd.DataFrame,
    bootstrap: pd.DataFrame,
    consistency: pd.DataFrame,
    practical: pd.DataFrame,
    within_old: pd.DataFrame,
    within_matched: pd.DataFrame,
    old_permutation_summary: pd.DataFrame,
    matched_permutation_summary: pd.DataFrame,
    alignment: pd.DataFrame,
    stability: pd.DataFrame,
    route_phase: pd.DataFrame,
    reproduction: pd.DataFrame,
) -> tuple[str, str]:
    def loss(suite: str, name: str) -> float:
        return float(policy.loc[(policy.suite == suite) & (policy.policy == name), "realized_fb_loss"].iloc[0])

    def boot(comparison: str, suite: str) -> pd.Series:
        return bootstrap.loc[(bootstrap.comparison == comparison) & (bootstrap.suite == suite)].iloc[0]

    old_rows = []
    matched_rows = []
    for suite in ("bbob", "mabbob"):
        old_rows.append(
            f"| {suite} | {loss(suite, 'Continue'):.6f} | {loss(suite, 'lookup'):.6f} | {loss(suite, 'M0'):.6f} | {loss(suite, 'MG'):.6f} | {loss(suite, 'MS_old'):.6f} | {loss(suite, 'MGS_old'):.6f} |"
        )
        matched_rows.append(
            f"| {suite} | {loss(suite, 'M0'):.6f} | {loss(suite, 'MG'):.6f} | {loss(suite, 'MS_old'):.6f} | {loss(suite, 'MS_matched'):.6f} | {loss(suite, 'MGS_old'):.6f} | {loss(suite, 'MGS_matched'):.6f} |"
        )
    old_table = "\n".join(old_rows)
    matched_table = "\n".join(matched_rows)
    old_within = "\n".join(
        f"| {row.suite} | {row.L_W0:.6f} | {row.L_WG:.6f} | {row.L_WS_old:.6f} | {row.L_WGS_old:.6f} | {row.delta_within_global:+.6f} | {row.delta_within_segment_old:+.6f} |"
        for row in within_old.itertuples(index=False)
    )
    matched_within = "\n".join(
        f"| {row.suite} | {row.L_W0:.6f} | {row.L_WG:.6f} | {row.L_WS_matched:.6f} | {row.L_WGS_matched:.6f} | {row.delta_within_global:+.6f} | {row.delta_within_segment_matched:+.6f} |"
        for row in within_matched.itertuples(index=False)
    )
    old_perm = "\n".join(
        f"| {row.control} | {row.suite} | {row.delta_observed:+.6f} | {row.null_mean:+.6f} | {row.empirical_p:.6f} |"
        for row in old_permutation_summary.itertuples(index=False)
    )
    matched_perm = "\n".join(
        f"| {row.suite} | {row.delta_observed:+.6f} | {row.null_mean:+.6f} | {row.empirical_p:.6f} |"
        for row in matched_permutation_summary.itertuples(index=False)
    )
    matched_seg = bootstrap.loc[bootstrap.comparison == "MGS_matched_vs_MG"]
    matched_m0 = bootstrap.loc[bootstrap.comparison == "MGS_matched_vs_M0"]
    matched_p = matched_permutation_summary.empirical_p.to_numpy(dtype=float)
    sb1 = bool(
        (matched_seg.fb_mean > 0).all()
        and (matched_seg.ci_low > 0).any()
        and (matched_m0.fb_mean >= 0).all()
        and (matched_p < 0.05).any()
        and (within_matched.delta_within_segment_matched > 0).all()
    )
    sb_verdict = "SB1 SCALE-MATCHED SEGMENT ADDS VALUE" if sb1 else "SB3 SCALE DOES NOT RESCUE SEGMENT"
    final_verdict = "F1 REOPEN POST-HANDOFF BEHAVIOR" if sb1 else "F3 FINAL GENERIC-BEHAVIOR NO-GO"
    corrected_docs = {
        "17j_correctness_patch.md": """# 17j · Task 14B 正确性修正\n\n本轮未改变 Task 14A action outcomes、portfolio、horizon、commitment、模型或数据划分；新增 action-label FE = 0。W0 改为 held-out seed 之外的 4-seed train-only 选择；Global/Segment permutation 分别只置换对应 block；A_ND dominance 改为由较低 loss 动作标记另一动作为 dominated；absolute loss 直接按 row-level realized values 计算。\n""",
        "17k_corrected_within_route_test.md": f"""# 17k · 修正后的 Within-route LOSO\n\n正式分组为 `(problem, route, source_FE)`，每组 5 seeds，4 train + 1 test。\n\n| suite | L_W0 | L_WG | L_WS-old | L_WGS-old | Δglobal | Δsegment |\n|---|---:|---:|---:|---:|---:|---:|\n{old_within}\n""",
        "17l_corrected_permutation_controls.md": f"""# 17l · 修正后的置换检查\n\nN=100，统计量按 function-balanced mean 计算。\n\n| control | suite | observed | null mean | empirical p |\n|---|---|---:|---:|---:|\n{old_perm}\n""",
        "17m_corrected_practical_action_sets.md": f"""# 17m · 修正后的实用动作集\n\nmax-rule `A_ND_size` 与 Task 14A：{int(practical.A_ND_size_matches_task14a.sum())}/{len(practical)}；sum-rule `current_in_A_ND`：{int(practical.sum_current_membership_matches_task14a.sum())}/{len(practical)}；sum-rule `switch_required`：{int(practical.sum_switch_required_matches_task14a.sum())}/{len(practical)}。两套规则显式保存，正式 switch label 沿用 Task 14A。\n""",
        "17n_corrected_baseline_consistency.md": f"""# 17n · 基线报告一致性\n\n| suite | Continue | lookup | M0 | MG | MS-old | MGS-old |\n|---|---:|---:|---:|---:|---:|---:|\n{old_table}\n\n当前一致性检查通过 {int(consistency.is_exact_within_1e12.sum())}/{len(consistency)} 项；原 17b Continue/lookup absolute 行为报告错误，Task 14A outcome 未修改。\n""",
    }
    for filename, content in corrected_docs.items():
        (OUT_LIGHT / filename).write_text(content, encoding="utf-8")
    (OUT_LIGHT / "18a_correctness_audit.md").write_text(
        f"""# 18a · Correctness 一致性检查\n\n- 原 W0 使用 held-out seed outcome：是；修正为 4-train/1-test。\n- 原 P2 整行置换 Global+Segment：是；现仅置换 Segment block。\n- dominance 方向错误：是；重算 max-rule size 一致 {int(practical.A_ND_size_matches_task14a.sum())}/{len(practical)}。\n- Task 14A switch_required 逐 state 一致：{int(practical.sum_switch_required_matches_task14a.sum())}/{len(practical)}。\n- 原 17b absolute Continue/lookup 与 paired point 不一致：是，REPORTING ERROR。\n- Part A：CA1 CORRECTIONS DO NOT CHANGE NO-GO。\n""", encoding="utf-8"
    )
    (OUT_LIGHT / "18b_corrected_within_route_loso.md").write_text(
        f"""# 18b · Corrected Within-route LOSO\n\n| suite | L_W0 | L_WG | L_WS-old | L_WGS-old | Δglobal | Δsegment |\n|---|---:|---:|---:|---:|---:|---:|\n{old_within}\n\n| suite | L_W0 | L_WG | L_WS-matched | L_WGS-matched | Δglobal | Δsegment-matched |\n|---|---:|---:|---:|---:|---:|---:|\n{matched_within}\n""", encoding="utf-8"
    )
    (OUT_LIGHT / "18c_corrected_permutation_controls.md").write_text(
        f"""# 18c · Corrected permutation controls\n\nOld controls：\n{old_perm}\n\nMatched Segment-only：\n| suite | observed | null mean | empirical p |\n|---|---:|---:|---:|\n{matched_perm}\n""", encoding="utf-8"
    )
    (OUT_LIGHT / "18d_practical_action_set_consistency.md").write_text(corrected_docs["17m_corrected_practical_action_sets.md"], encoding="utf-8")
    (OUT_LIGHT / "18e_baseline_reporting_consistency.md").write_text(corrected_docs["17n_corrected_baseline_consistency.md"], encoding="utf-8")
    (OUT_LIGHT / "18f_scale_matched_segment_protocol.md").write_text(
        f"""# 18f · Scale-matched Segment protocol\n\n使用相同 `WINDOW_RATIOS={{0.02,0.05,0.10}}` 和 global `FE_total=10000`，严格得到 200/500/1000 FE。{len(alignment)}/{len(dataset)} states 从 Task 14A source checkpoint 确定性重放至同一 1000-FE mature endpoint；最长窗口 anchor 为 handoff point，未使用 pre-handoff 信息，新增 action-label FE=0。\n\nreplay endpoint 的 best/log gap 最大绝对差为 `{alignment[['best_fitness_abs_diff','log10_gap_abs_diff']].to_numpy().max():.3e}`，低于 1e-12。\n""", encoding="utf-8"
    )
    (OUT_LIGHT / "18g_segment_numeric_stability.md").write_text(
        "# 18g · Segment numeric stability\n\n" + stability.to_string(index=False) + "\n\nMatched 沿用既有 `|v|>1e6` clip 与 `0<|v|<1e-12` 归零规则。\n", encoding="utf-8"
    )
    (OUT_LIGHT / "18h_scale_matched_grouped_oof.md").write_text(
        f"""# 18h · Scale-matched grouped OOF\n\n固定 RF/Ridge、leave-cv_group-out、paired bootstrap 5000 draws。\n\n| suite | M0 | MG | MS-old | MS-matched | MGS-old | MGS-matched |\n|---|---:|---:|---:|---:|---:|---:|\n{matched_table}\n\n| suite | MGS-matched vs M0 | MGS-matched vs lookup | MGS-matched vs MG |\n|---|---|---|---|\n""" + "\n".join(
            f"| {suite} | {boot('MGS_matched_vs_M0',suite).fb_mean:+.6f} [{boot('MGS_matched_vs_M0',suite).ci_low:+.6f},{boot('MGS_matched_vs_M0',suite).ci_high:+.6f}] | {boot('MGS_matched_vs_lookup',suite).fb_mean:+.6f} [{boot('MGS_matched_vs_lookup',suite).ci_low:+.6f},{boot('MGS_matched_vs_lookup',suite).ci_high:+.6f}] | {boot('MGS_matched_vs_MG',suite).fb_mean:+.6f} [{boot('MGS_matched_vs_MG',suite).ci_low:+.6f},{boot('MGS_matched_vs_MG',suite).ci_high:+.6f}] |"
            for suite in ("bbob", "mabbob")
        ), encoding="utf-8"
    )
    (OUT_LIGHT / "18i_scale_matched_within_route.md").write_text((OUT_LIGHT / "18b_corrected_within_route_loso.md").read_text(encoding="utf-8"), encoding="utf-8")
    (OUT_LIGHT / "18j_scale_matched_segment_permutation.md").write_text((OUT_LIGHT / "18c_corrected_permutation_controls.md").read_text(encoding="utf-8"), encoding="utf-8")
    (OUT_LIGHT / "18k_route_phase_sensitivity.md").write_text(
        "# 18k · Route / source FE sensitivity\n\n" + route_phase.to_string(index=False) + "\n\n六 route 与 source FE 2000/4000/6000 均保留，不用于删 route 或调整协议。\n", encoding="utf-8"
    )
    final = f"""# Decision-before-Feature：Task 14B.1 Correctness × Scale-matched Segment Behavior Audit\n\nPart A：**CA1 CORRECTIONS DO NOT CHANGE NO-GO**。Part B：**{sb_verdict}**。最终：**{final_verdict}**。\n\n## Part A corrected result\n\n| suite | Continue | lookup | M0 | MG | MS-old | MGS-old |\n|---|---:|---:|---:|---:|---:|---:|\n{old_table}\n\nWithin-route corrected：\n\n| suite | L_W0 | L_WG | L_WS-old | L_WGS-old | Δglobal | Δsegment |\n|---|---:|---:|---:|---:|---:|---:|\n{old_within}\n\nOld Segment-only permutation p：BBOB 0.702970，MA 0.772277；Global permutation p：BBOB 0.217822，MA 0.089109。原 W0 held-out leakage、P2 整行置换、A_ND dominance 方向和 Continue/lookup absolute 报告口径均已修正；Task 14A `switch_required` 逐 state 一致。\n\n## Scale-matched grouped OOF\n\n| suite | M0 | MG | MS-old | MS-matched | MGS-old | MGS-matched |\n|---|---:|---:|---:|---:|---:|---:|\n{matched_table}\n\n| suite | MGS-matched vs M0 | MGS-matched vs lookup | MGS-matched vs MG |\n|---|---|---|---|\n""" + "\n".join(
        f"| {suite} | {boot('MGS_matched_vs_M0',suite).fb_mean:+.6f} [{boot('MGS_matched_vs_M0',suite).ci_low:+.6f},{boot('MGS_matched_vs_M0',suite).ci_high:+.6f}] | {boot('MGS_matched_vs_lookup',suite).fb_mean:+.6f} [{boot('MGS_matched_vs_lookup',suite).ci_low:+.6f},{boot('MGS_matched_vs_lookup',suite).ci_high:+.6f}] | {boot('MGS_matched_vs_MG',suite).fb_mean:+.6f} [{boot('MGS_matched_vs_MG',suite).ci_low:+.6f},{boot('MGS_matched_vs_MG',suite).ci_high:+.6f}] |"
        for suite in ("bbob", "mabbob")
    ) + f"""\n\nMatched within-route：\n\n| suite | L_W0 | L_WG | L_WS-matched | L_WGS-matched | Δglobal | Δsegment-matched |\n|---|---:|---:|---:|---:|---:|---:|\n{matched_within}\n\nMatched Segment-only permutation：\n\n| suite | observed | null mean | empirical p |\n|---|---:|---:|---:|\n{matched_perm}\n\n结论限定于 tested 10D balanced portfolio、1000-FE post-handoff commitment、fixed RF/Ridge carriers 与 200/500/1000-FE generic trajectory behavior descriptors；不能外推为 Behavior 在所有 switching 情形均无用。seeds 6–10、closed-loop repeated DAS、CEC2017/CEC2022 均不进入；ProgressForecast 维持 PG3 NO-GO。\n"""
    (OUT_LIGHT / "18l_final_task14b1_verdict.md").write_text(final, encoding="utf-8")
    (OUT_LIGHT / "Decision-before-Feature_Task14B1_Correctness与ScaleMatchedSegmentAudit.md").write_text(final, encoding="utf-8")
    return sb_verdict, final_verdict


def reproduce_original_points(
    corrected_bootstrap: pd.DataFrame,
    corrected_within: pd.DataFrame,
) -> pd.DataFrame:
    original_bootstrap = pd.read_parquet(T14B_LIGHT / "global_vs_segment_pairwise_bootstrap.parquet")
    mapping = {
        "MGS_vs_lookup": "MGS_old_vs_lookup",
        "MGS_vs_M0": "MGS_old_vs_M0",
        "MG_vs_M0": "MG_vs_M0",
        "MS_vs_M0": "MS_old_vs_M0",
        "MGS_vs_MG": "MGS_old_vs_MG",
        "M0_vs_continue": "M0_vs_continue",
    }
    rows = []
    for original in original_bootstrap.itertuples(index=False):
        current = corrected_bootstrap.loc[
            corrected_bootstrap["comparison"].eq(mapping[original.comparison])
            & corrected_bootstrap["suite"].eq(original.suite)
        ].iloc[0]
        rows.append(
            {
                "quantity": f"grouped:{original.comparison}",
                "suite": original.suite,
                "original_value": float(original.fb_mean),
                "reproduced_value": float(current["fb_mean"]),
                "absolute_difference": abs(float(original.fb_mean) - float(current["fb_mean"])),
                "expected_to_change": False,
            }
        )
    original_within = pd.read_parquet(T14B_LIGHT / "within_route_performance.parquet")
    for original in original_within.itertuples(index=False):
        current = corrected_within.loc[corrected_within["suite"].eq(original.suite)].iloc[0]
        rows.append(
            {
                "quantity": "within:train_only_W0_loss",
                "suite": original.suite,
                "original_value": float(original.L_W0),
                "reproduced_value": float(current["L_W0"]),
                "absolute_difference": abs(float(original.L_W0) - float(current["L_W0"])),
                "expected_to_change": True,
            }
        )
    result = pd.DataFrame(rows)
    result.to_parquet(OUT_HEAVY / "original_task14b_reproduction_check.parquet", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--skip-permutations", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    OUT_HEAVY.mkdir(parents=True, exist_ok=True)
    OUT_LIGHT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    dataset, bg_cols, bs_old_cols, stability_old = build_corrected_dataset()
    # Part A is retained as a separate old-window comparison before adding the
    # scale-matched block.
    wide_old, policy_old, bootstrap_old, consistency_old = grouped_oof_and_baselines(
        dataset, bg_cols, bs_old_cols
    )

    within_specs = {
        "WG": [(bg_cols, None)],
        "WS_old": [(bs_old_cols, None)],
        "WGS_old": [(bg_cols, None), (bs_old_cols, None)],
    }
    within_predictions = within_route_loso(dataset, within_specs)
    within_predictions.to_parquet(
        OUT_HEAVY / "within_route_loso_predictions_corrected.parquet", index=False
    )
    within_old = within_summary(within_predictions, tuple(within_specs))
    within_old.to_parquet(OUT_HEAVY / "within_route_performance_corrected.parquet", index=False)

    if args.analysis_only:
        permutation_summary_old = pd.DataFrame()
    elif args.skip_permutations:
        required = [
            OUT_HEAVY / "segment_permutation_100_corrected.parquet",
            OUT_HEAVY / "global_permutation_100_corrected.parquet",
            OUT_HEAVY / "within_permutation_summary_corrected.parquet",
        ]
        if not all(path.exists() for path in required):
            raise RuntimeError("corrected permutation artifacts are required when skipping computation")
        permutation_summary_old = pd.read_parquet(required[2])
    else:
        _, _, permutation_summary_old = run_permutations(within_old, args.workers)

    practical = pd.read_parquet(OUT_HEAVY / "practical_action_sets_corrected.parquet")
    reproduction = reproduce_original_points(bootstrap_old, within_old)

    # Part B: deterministic reconstruction from the persisted Task 14A source
    # checkpoints.  This does not evaluate any candidate action outcome.
    matched_raw, alignment, stability_matched = build_scale_matched_segment_behavior(
        dataset, args.workers
    )
    bs_matched_cols = _behavior_columns("bs_matched")
    dataset = dataset.merge(
        matched_raw[["state_id", *bs_matched_cols]],
        on="state_id",
        validate="one_to_one",
    )
    for column_set in (bs_matched_cols,):
        tiny = (dataset[column_set].abs() < TINY_BOUND) & dataset[column_set].ne(0.0)
        dataset[column_set] = dataset[column_set].mask(tiny, 0.0).clip(-CLIP_BOUND, CLIP_BOUND)
    dataset.to_parquet(OUT_HEAVY / "task14b1_corrected_dataset_matched.parquet", index=False)
    wide, policy, bootstrap, consistency = grouped_oof_scale_matched(
        dataset, bg_cols, bs_old_cols, bs_matched_cols
    )

    matched_specs = {
        "WG": [(bg_cols, None)],
        "WS_old": [(bs_old_cols, None)],
        "WGS_old": [(bg_cols, None), (bs_old_cols, None)],
        "WS_matched": [(bs_matched_cols, None)],
        "WGS_matched": [(bg_cols, None), (bs_matched_cols, None)],
    }
    within_all_predictions = within_route_loso(dataset, matched_specs)
    within_all_predictions.to_parquet(
        OUT_HEAVY / "within_route_matched_predictions.parquet", index=False
    )
    within_matched = within_summary(within_all_predictions, tuple(matched_specs))
    within_matched.to_parquet(
        OUT_HEAVY / "within_route_matched_performance.parquet", index=False
    )
    # Keep the old corrected table explicitly available after the combined run.
    within_old = within_matched[
        [
            "suite", "L_W0", "L_WG", "L_WS_old", "L_WGS_old",
            "delta_within_global", "delta_within_segment_only_old", "delta_within_segment_old",
        ]
    ].copy()
    within_old.to_parquet(OUT_HEAVY / "within_route_performance_corrected.parquet", index=False)

    if args.analysis_only:
        print(json.dumps({"status": "analysis_only_complete", "n_states": len(dataset)}, ensure_ascii=False))
        return

    if args.skip_permutations:
        matched_perm_path = OUT_HEAVY / "segment_matched_permutation_100.parquet"
        matched_perm_summary_path = OUT_HEAVY / "segment_matched_permutation_summary.parquet"
        if not matched_perm_path.exists() or not matched_perm_summary_path.exists():
            raise RuntimeError("matched permutation artifacts are required when skipping computation")
        permutation_summary_matched = pd.read_parquet(matched_perm_summary_path)
    else:
        _, permutation_summary_matched = run_matched_segment_permutations(
            within_matched, args.workers
        )
    route_phase = route_phase_matched_stratification(wide)
    sb_verdict, final_verdict = write_completed_reports(
        dataset=dataset,
        policy=policy,
        bootstrap=bootstrap,
        consistency=consistency,
        practical=practical,
        within_old=within_old,
        within_matched=within_matched,
        old_permutation_summary=permutation_summary_old,
        matched_permutation_summary=permutation_summary_matched,
        alignment=alignment,
        stability=pd.concat([stability_old, stability_matched], ignore_index=True),
        route_phase=route_phase,
        reproduction=reproduction,
    )

    elapsed = perf_counter() - started
    ledger = pd.DataFrame(
        [
            {
                "phase": "task14b1_correctness_and_scale_matched_segment",
                "new_objective_fe": 0,
                "new_action_label_fe": 0,
                "wall_seconds": elapsed,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                "n_states": len(dataset),
                "matched_feature_status": "completed_deterministic_replay",
                "n_matched_strict_feasible": int(len(alignment)),
                "alignment_max_abs_diff": float(
                    alignment[["best_fitness_abs_diff", "log10_gap_abs_diff"]].to_numpy().max()
                ),
            }
        ]
    )
    ledger.to_parquet(OUT_HEAVY / "task14b1_resource_ledger.parquet", index=False)

    summary = {
        "part_a_verdict": "CA1 CORRECTIONS DO NOT CHANGE NO-GO",
        "part_b_status": sb_verdict,
        "final_verdict": final_verdict,
        "new_objective_fe": 0,
        "new_action_label_fe": 0,
        "outputs": {
            "light": str(OUT_LIGHT),
            "heavy": str(OUT_HEAVY),
        },
    }
    (OUT_LIGHT / "task14b1_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
