"""Shared helpers for the analysis_v4 feasibility diagnostics (Task 11A-11H)."""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "behavior_with_ela" / "results"
V4 = ROOT / "behavior_with_ela" / "analysis_v4"
V4_HEAVY = RESULTS / "analysis_v4"
V3_HEAVY = RESULTS / "analysis_v3"

from behavior.features import SELECTOR_BEHAVIOR_FEATURE_COLUMNS  # noqa: E402
from behavior_with_ela.model import (  # noqa: E402
    STATE_KEY,
    apply_practical_gain_delta,
    read_action_datasets,
    read_action_repetitions,
)
from behavior_with_ela.protocol import load_experiment_config  # noqa: E402

TRAIN_CONFIG = ROOT / "configs/behavior_with_ela_train.yaml"
VALIDATION_CONFIG = ROOT / "configs/behavior_with_ela_validation.yaml"
PHASE1_MODEL = RESULTS / "model/behavior_action_gain/models.joblib"

PORTFOLIO = ("pso", "shade", "cmaes")
SUCCESS_LOG10_TARGET = -8.0
LOG10_FLOOR = -12.0  # log10_gap_floor of the train/validation configs


def load_config():
    return load_experiment_config(TRAIN_CONFIG)


def noise_deltas() -> dict[str, float]:
    cached = V3_HEAVY / "noise_deltas.json"
    if not cached.exists():
        raise FileNotFoundError(
            "noise deltas missing; run analysis_v3/task1_per_prefix.py first"
        )
    return {key: float(value) for key, value in json.loads(cached.read_text()).items()}


def load_action_frames():
    """Aggregate action-loss frames: (train incl. MA, validation, delta)."""
    import joblib

    config = load_experiment_config(TRAIN_CONFIG)
    validation_config = load_experiment_config(VALIDATION_CONFIG)
    bundle = joblib.load(PHASE1_MODEL)
    delta = float(bundle["practical_gain_delta"])
    train = apply_practical_gain_delta(read_action_datasets(config), delta)
    validation = apply_practical_gain_delta(
        read_action_datasets(validation_config), delta
    )
    return config, bundle, delta, train, validation


def state_action_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per state with the three observed action losses and best action."""
    order = {algorithm: index for index, algorithm in enumerate(PORTFOLIO)}
    rows = frame.copy()
    rows["candidate_order"] = rows["candidate_action"].astype(str).map(order)
    rows = rows.sort_values([*STATE_KEY, "candidate_order"], kind="mergesort")
    counts = rows.groupby(list(STATE_KEY), sort=False)["candidate_action"].size()
    if not counts.eq(len(PORTFOLIO)).all():
        raise ValueError("each state must contain the complete action portfolio")
    losses = rows["log10_action_loss"].to_numpy(dtype=float).reshape(
        counts.shape[0], len(PORTFOLIO)
    )
    first = rows.groupby(list(STATE_KEY), sort=False).head(1).reset_index(drop=True)
    best_index = losses.argmin(axis=1)
    sorted_losses = np.sort(losses, axis=1)
    result = first[
        [
            "split", "suite", "problem_id", "function_id", "family", "cv_group_id",
            "prefix_algorithm", "seed", "FE", "FE_ratio", "decision_opportunity_index",
            "sampling_phase", "is_budget_milestone", "is_event_sample",
        ]
    ].copy()
    for index, algorithm in enumerate(PORTFOLIO):
        result[f"loss_{algorithm}"] = losses[:, index]
    result["best_action"] = [PORTFOLIO[index] for index in best_index]
    result["best_loss"] = losses[np.arange(len(losses)), best_index]
    result["second_best_loss"] = sorted_losses[:, 1]
    result["action_margin"] = sorted_losses[:, 1] - sorted_losses[:, 0]
    prefix_index = np.array(
        [PORTFOLIO.index(a) for a in first["prefix_algorithm"].astype(str)]
    )
    result["continue_loss"] = losses[np.arange(len(losses)), prefix_index]
    switch_losses = np.where(
        np.arange(len(PORTFOLIO))[None, :] == prefix_index[:, None],
        np.inf,
        losses,
    )
    result["best_switch_gain"] = result["continue_loss"].to_numpy(dtype=float) - (
        switch_losses.min(axis=1)
    )
    return result


def distribution_table(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    counts = frame.groupby([*group_columns, "best_action"], sort=False).size()
    totals = frame.groupby(group_columns, sort=False).size().rename("total")
    table = counts.reset_index(name="count").merge(
        totals.reset_index(), on=group_columns, how="left"
    )
    table["share"] = table["count"] / table["total"]
    return table


def entropy(counts: np.ndarray) -> float:
    counts = counts[counts > 0].astype(float)
    if counts.size == 0:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def conditional_entropy(frame: pd.DataFrame, group_columns: list[str]) -> dict:
    total = len(frame)
    conditional = 0.0
    per_group = []
    for key, group in frame.groupby(group_columns, sort=False):
        weight = len(group) / total
        value = entropy(group["best_action"].value_counts().to_numpy())
        conditional += weight * value
        per_group.append(
            {
                **{column: str(key[index]) for index, column in enumerate(group_columns)},
                "states": int(len(group)),
                "entropy_bits": value,
            }
        )
    return {
        "conditional_entropy_bits": float(conditional),
        "per_group": pd.DataFrame(per_group),
    }


def function_balanced(values: pd.Series, groups: pd.Series) -> float:
    frame = pd.DataFrame({"value": values, "group": groups})
    return float(frame.groupby("group")["value"].mean().mean())


def save_table(table, name: str, task: str) -> Path:
    target_dir = V4 / task
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / name
    if name.endswith(".parquet"):
        table.to_parquet(path, index=False)
    elif name.endswith(".json"):
        path.write_text(json.dumps(table, indent=2, ensure_ascii=False, default=float))
    else:
        table.to_csv(path, index=False)
    return path


def save_heavy_table(table, name: str, task: str) -> Path:
    target_dir = V4_HEAVY / task
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / name
    table.to_parquet(path, index=False)
    return path


def json_dumps(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=float)
