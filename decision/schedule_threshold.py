"""Schedule × threshold 联合冻结.

正式策略不是只选 tau，而是联合选择 (T_D*, tau*)，最大化 run-level OOF mean efficacy。

J(T_D, tau) = (1/R) sum_r C_r(T_D, tau)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_DECISION_CHECKS = 7


@dataclass(frozen=True)
class ScheduleThresholdResult:
    schedule: tuple[float, ...]
    threshold: float
    mean_efficacy: float
    call_rate: float
    n_called: int


def _run_level_contribution(
    labels: pd.DataFrame,
    run_key_columns: list[str],
    score_column: str,
    threshold: float,
    g_fe_column: str = "g_fe",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (contributions, called_flags) per run."""
    run_ids = labels[run_key_columns].astype(str).agg("|".join, axis=1).to_numpy()
    scores = np.asarray(labels[score_column].astype(float))
    g_fe = np.asarray(labels[g_fe_column].astype(float))

    contributions = []
    called_flags = []
    for run_id in np.unique(run_ids):
        mask = run_ids == run_id
        run_scores = scores[mask]
        run_g_fe = g_fe[mask]

        trigger = np.flatnonzero(run_scores > threshold)
        if len(trigger) > 0:
            contributions.append(float(run_g_fe[int(trigger[0])]))
            called_flags.append(True)
        else:
            contributions.append(0.0)
            called_flags.append(False)
    return np.asarray(contributions), np.asarray(called_flags)


def _select_schedule_ratios(
    fe_ratios: np.ndarray,
    max_checks: int = MAX_DECISION_CHECKS,
) -> list[tuple[float, ...]]:
    """Generate candidate schedule subsets of at most max_checks milestones."""
    unique = sorted(set(float(r) for r in fe_ratios))
    if len(unique) <= max_checks:
        return [tuple(unique)]
    # Select evenly-spaced subsets
    candidates = [tuple(unique)]
    step = len(unique) // max_checks
    if step > 1:
        sparse = tuple(unique[::step][:max_checks])
        if sparse not in candidates:
            candidates.append(sparse)
    # Also add just the 12 formal milestones
    formal = tuple(
        r for r in unique
        if r in {0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.60}
    )
    if formal and formal not in candidates:
        candidates.append(formal)
    return candidates


def joint_schedule_threshold_selection(
    *,
    labels: pd.DataFrame,
    run_key_columns: list[str],
    score_column: str,
    fe_ratio_column: str = "FE_ratio",
    g_fe_column: str = "g_fe",
    threshold_grid: np.ndarray | None = None,
    max_checks: int = MAX_DECISION_CHECKS,
) -> ScheduleThresholdResult:
    """Select (T_D*, tau*) maximizing run-level OOF mean efficacy."""
    if threshold_grid is None:
        threshold_grid = np.linspace(-2.0, 2.0, 81)
    else:
        threshold_grid = np.asarray(threshold_grid, dtype=float)

    fe_ratios = np.asarray(labels[fe_ratio_column].astype(float))
    candidate_schedules = _select_schedule_ratios(fe_ratios, max_checks=max_checks)

    best: ScheduleThresholdResult | None = None
    for schedule in candidate_schedules:
        mask = np.isin(fe_ratios, np.asarray(schedule))
        sub = labels.loc[mask].copy()
        if sub.empty:
            continue
        for tau in threshold_grid:
            contributions, called = _run_level_contribution(
                labels=sub,
                run_key_columns=run_key_columns,
                score_column=score_column,
                threshold=float(tau),
                g_fe_column=g_fe_column,
            )
            mean_eff = float(np.mean(contributions)) if len(contributions) > 0 else 0.0
            call_rate = float(np.mean(called)) if len(called) > 0 else 0.0
            n_called = int(np.sum(called))
            if best is None or mean_eff > best.mean_efficacy:
                best = ScheduleThresholdResult(
                    schedule=schedule,
                    threshold=float(tau),
                    mean_efficacy=mean_eff,
                    call_rate=call_rate,
                    n_called=n_called,
                )
    if best is None:
        raise ValueError("no valid schedule × threshold combination found")
    return best
