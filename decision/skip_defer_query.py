"""Skip–Defer–Query 三向决策.

给定预测区间 [l, u]:

  Query  if l > delta_practical
  Skip   if u < 0
  Defer  otherwise

Defer 在最近的下一个 fine opportunity 复查；
Skip 进入 cooldown，只在下一个 coarse milestone 复查。
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class DecisionAction(str, Enum):
    QUERY = "query"
    SKIP = "skip"
    DEFER = "defer"


# Fine opportunities: all frozen milestones + accepted behavior events
# Coarse opportunities: phase boundaries or at least 0.05 FE-ratio gap
COARSE_MILESTONE_RATIOS = (0.20, 0.30, 0.42, 0.50, 0.60)


def skip_defer_query_decision(
    *,
    lower_bound: np.ndarray | float,
    upper_bound: np.ndarray | float,
    delta_practical: float,
) -> np.ndarray | DecisionAction:
    """Classify each opportunity into Query / Skip / Defer.

    Parameters
    ----------
    lower_bound
        Predicted lower quantile / conformal lower bound.
    upper_bound
        Predicted upper quantile / conformal upper bound.
    delta_practical
        Minimum practically meaningful efficacy threshold.
    """
    is_scalar = np.isscalar(lower_bound) and np.isscalar(upper_bound)
    lower = np.atleast_1d(np.asarray(lower_bound, dtype=float))
    upper = np.atleast_1d(np.asarray(upper_bound, dtype=float))

    actions = np.empty(len(lower), dtype=object)
    for i in range(len(lower)):
        if lower[i] > float(delta_practical):
            actions[i] = DecisionAction.QUERY
        elif upper[i] < 0.0:
            actions[i] = DecisionAction.SKIP
        else:
            actions[i] = DecisionAction.DEFER

    if is_scalar:
        return actions[0]
    return actions


def next_review_ratio(
    *,
    current_ratio: float,
    action: DecisionAction,
    available_ratios: list[float],
) -> float | None:
    """Determine the next review FE ratio based on action type.

    Defer: next fine opportunity (nearest available ratio >= current).
    Skip: next coarse milestone (>= current + 0.05 or next phase boundary).
    Query: None (no further review needed).
    """
    if action == DecisionAction.QUERY:
        return None

    current = float(current_ratio)
    available = sorted(float(r) for r in available_ratios)

    if action == DecisionAction.DEFER:
        for r in available:
            if r > current + 1e-12:
                return r
        return None

    if action == DecisionAction.SKIP:
        coarse = sorted(set(COARSE_MILESTONE_RATIOS) & set(available))
        if not coarse:
            coarse = available
        for r in coarse:
            if r >= current + 0.05 - 1e-12:
                return r
        return None

    return None
