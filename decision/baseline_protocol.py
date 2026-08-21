"""Shared protocol for dynamic algorithm-selection baseline policies.

The module keeps policy scheduling separate from action-outcome data.  A baseline
may only select an action when its input table contains the corresponding
observed action outcome columns; no baseline silently reuses query outcomes as
switch outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from decision.sampling_opportunities import STATE_KEY_COLUMNS

RUN_KEY_COLUMNS = STATE_KEY_COLUMNS[:-1]
PORTFOLIO = ("de", "pso", "cmaes", "shade")
BASELINE_PROTOCOL = "dynamic_action_baseline_contract_v1"


@dataclass(frozen=True)
class BaselinePolicySpec:
    name: str
    policy_kind: str
    max_actions_per_run: int = 1
    requires_action_outcomes: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.policy_kind:
            raise ValueError("baseline policy identity must not be empty")
        if self.max_actions_per_run != 1:
            raise ValueError("formal baseline protocol currently permits one action per run")


def validate_baseline_state_frame(frame: pd.DataFrame, *, artifact: str) -> None:
    required = {*STATE_KEY_COLUMNS, "FE_ratio", "prefix_algorithm"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{artifact} is missing baseline state columns: {missing}")
    if frame.duplicated(list(STATE_KEY_COLUMNS)).any():
        raise ValueError(f"{artifact} contains duplicate baseline state keys")
    if not np.isfinite(frame["FE_ratio"].to_numpy(dtype=float)).all():
        raise ValueError(f"{artifact} contains non-finite FE ratios")
    if ((frame["FE_ratio"].to_numpy(dtype=float) <= 0.0) | (frame["FE_ratio"].to_numpy(dtype=float) >= 1.0)).any():
        raise ValueError(f"{artifact} FE ratios must lie in (0, 1)")


def first_action_mask(frame: pd.DataFrame, candidates: np.ndarray) -> np.ndarray:
    """Return exactly the first candidate state per run."""
    validate_baseline_state_frame(frame, artifact="baseline state frame")
    values = np.asarray(candidates, dtype=bool).reshape(-1)
    if len(values) != len(frame):
        raise ValueError("baseline candidate mask does not align with state frame")
    output = np.zeros(len(frame), dtype=bool)
    for _, group in frame.assign(_candidate=values).groupby(list(RUN_KEY_COLUMNS), sort=True, dropna=False):
        ordered = group.sort_values(["FE", *(["decision_opportunity_index"] if "decision_opportunity_index" in group else [])])
        hit = np.flatnonzero(ordered["_candidate"].to_numpy(dtype=bool))
        if hit.size:
            row_position = frame.index.get_indexer([ordered.index[int(hit[0])]])[0]
            if row_position < 0:
                raise ValueError("baseline ordered state is missing from the source frame")
            output[row_position] = True
    return output


def fixed_one_switch_mask(frame: pd.DataFrame, *, switch_fe_ratio: float) -> np.ndarray:
    """Select the first opportunity at or after a frozen switch ratio."""
    ratio = float(switch_fe_ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError("fixed switch FE ratio must lie in (0, 1)")
    return first_action_mask(frame, frame["FE_ratio"].to_numpy(dtype=float) >= ratio)


def random_one_switch_mask(
    frame: pd.DataFrame,
    *,
    switch_fe_ratios: Iterable[float],
    seed: int,
) -> np.ndarray:
    """Choose one frozen random switch ratio independently for each run."""
    ratios = np.asarray(tuple(float(value) for value in switch_fe_ratios), dtype=float)
    if ratios.size == 0 or not np.isfinite(ratios).all() or ((ratios <= 0.0) | (ratios >= 1.0)).any():
        raise ValueError("random switch FE ratios must be finite and lie in (0, 1)")
    validate_baseline_state_frame(frame, artifact="baseline state frame")
    output = np.zeros(len(frame), dtype=bool)
    for run_number, (_, group) in enumerate(frame.groupby(list(RUN_KEY_COLUMNS), sort=True, dropna=False)):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 2026082001, run_number]))
        target = float(rng.choice(ratios))
        ordered = group.sort_values(["FE", *(["decision_opportunity_index"] if "decision_opportunity_index" in group else [])])
        candidates = ordered["FE_ratio"].to_numpy(dtype=float) >= target
        if candidates.any():
            selected_index = ordered.index[int(np.flatnonzero(candidates)[0])]
            row_position = frame.index.get_indexer([selected_index])[0]
            if row_position < 0:
                raise ValueError("random baseline ordered state is missing from the source frame")
            output[row_position] = True
    return output


def validate_action_outcome_columns(frame: pd.DataFrame, *, algorithms: Iterable[str] = PORTFOLIO) -> None:
    required = {"FE", "prefix_algorithm", "remaining_budget_ratio"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"action-outcome baseline input is missing columns: {missing}")
    action_loss_matrix(frame, algorithms=algorithms)


def action_loss_matrix(
    frame: pd.DataFrame,
    *,
    algorithms: Iterable[str] = PORTFOLIO,
) -> np.ndarray:
    """Extract a per-action loss matrix from long or wide action tables."""
    algorithm_names = tuple(str(value).lower() for value in algorithms)
    wide_columns = [f"action_loss_{algorithm}" for algorithm in algorithm_names]
    if set(wide_columns).issubset(frame.columns):
        matrix = frame[wide_columns].to_numpy(dtype=float)
    elif {"target_algorithm", "action_loss"}.issubset(frame.columns):
        if frame.duplicated([*STATE_KEY_COLUMNS, "target_algorithm"]).any():
            raise ValueError("long action-outcome input contains duplicate state/action rows")
        pivot = frame.pivot(index=list(STATE_KEY_COLUMNS), columns="target_algorithm", values="action_loss")
        missing = sorted(set(algorithm_names).difference(str(value).lower() for value in pivot.columns))
        if missing:
            raise ValueError(f"long action-outcome input is missing algorithms: {missing}")
        matrix = pivot[[algorithm for algorithm in algorithm_names]].to_numpy(dtype=float)
    else:
        raise ValueError(
            "SwitchBenefit-RF requires either action_loss_<algorithm> columns or "
            "target_algorithm/action_loss long-form columns"
        )
    if not np.isfinite(matrix).all() or (matrix < 0.0).any():
        raise ValueError("action losses must be finite and non-negative")
    return matrix
