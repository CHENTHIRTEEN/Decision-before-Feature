"""Opportunity frequency ablation.

比较四种 opportunity 频率策略，区分 Controller 贡献与事件生成器贡献。

- milestones only
- milestones + events
- equal-count fixed grid
- dense grid
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

OPPORTUNITY_MODES = (
    "milestones_only",
    "milestones_plus_events",
    "equal_count_fixed_grid",
    "dense_grid",
)


@dataclass(frozen=True)
class OpportunityAblationResult:
    mode: str
    mean_first_trigger_g_fe: float
    median_first_trigger_g_fe: float
    call_rate: float
    efficacy_capture: float
    harmful_cost: float
    n_runs: int


def _run_level_first_trigger(
    *,
    labels: pd.DataFrame,
    run_key_columns: list[str],
    score_column: str,
    threshold: float,
    g_fe_column: str = "g_fe",
    fe_column: str = "FE",
    opportunity_column: str = "decision_opportunity_index",
) -> pd.DataFrame:
    """Compute run-level first-trigger contributions."""
    run_ids = labels[run_key_columns].astype(str).agg("|".join, axis=1).to_numpy()
    scores = np.asarray(labels[score_column].astype(float))
    g_fe = np.asarray(labels[g_fe_column].astype(float))
    fe = np.asarray(labels[fe_column].astype(int))
    opp_idx = (
        np.asarray(labels[opportunity_column].astype(int))
        if opportunity_column in labels.columns
        else fe
    )

    sort_order = np.lexsort((opp_idx, fe, run_ids))
    run_ids_sorted = run_ids[sort_order]
    scores_sorted = scores[sort_order]
    g_fe_sorted = g_fe[sort_order]

    unique_runs, first_indices = np.unique(run_ids_sorted, return_index=True)
    rows = []
    for run_id, first_idx in zip(unique_runs, first_indices, strict=True):
        run_slice = slice(first_idx, len(run_ids_sorted))
        run_mask = run_ids_sorted == run_id
        run_scores = scores_sorted[run_mask]
        run_g_fe = g_fe_sorted[run_mask]

        trigger_positions = np.flatnonzero(run_scores > threshold)
        if len(trigger_positions) > 0:
            trigger_pos = int(trigger_positions[0])
            contribution = float(run_g_fe[trigger_pos])
            called = True
        else:
            contribution = 0.0
            called = False

        run_max_positive = max(0.0, float(np.max(run_g_fe))) if len(run_g_fe) > 0 else 0.0
        rows.append(
            {
                "run_id": run_id,
                "called": called,
                "contribution": contribution,
                "run_max_positive": run_max_positive,
            }
        )
    return pd.DataFrame(rows)


def evaluate_opportunity_mode(
    *,
    labels: pd.DataFrame,
    run_key_columns: list[str],
    score_column: str,
    threshold: float,
    mode: str,
    g_fe_column: str = "g_fe",
) -> OpportunityAblationResult:
    """Evaluate one opportunity-frequency mode."""
    if mode not in OPPORTUNITY_MODES:
        raise ValueError(f"unknown opportunity mode: {mode}")

    run_results = _run_level_first_trigger(
        labels=labels,
        run_key_columns=run_key_columns,
        score_column=score_column,
        threshold=threshold,
        g_fe_column=g_fe_column,
    )

    contributions = run_results["contribution"].to_numpy(dtype=float)
    run_max_positive = run_results["run_max_positive"].to_numpy(dtype=float)
    called = run_results["called"].to_numpy(dtype=bool)
    n_runs = len(run_results)

    mean_g = float(np.mean(contributions)) if n_runs > 0 else 0.0
    median_g = float(np.median(contributions)) if n_runs > 0 else 0.0
    call_rate = float(np.mean(called)) if n_runs > 0 else 0.0
    total_max = float(np.sum(run_max_positive))
    capture = float(np.sum(np.maximum(contributions, 0.0)) / total_max) if total_max > 0 else 0.0
    harmful = float(np.sum(np.maximum(-contributions, 0.0)))

    return OpportunityAblationResult(
        mode=mode,
        mean_first_trigger_g_fe=mean_g,
        median_first_trigger_g_fe=median_g,
        call_rate=call_rate,
        efficacy_capture=capture,
        harmful_cost=harmful,
        n_runs=n_runs,
    )


def ablation_table(
    *,
    labels: pd.DataFrame,
    run_key_columns: list[str],
    score_column: str,
    threshold: float,
    g_fe_column: str = "g_fe",
) -> pd.DataFrame:
    """Run all four opportunity-frequency modes and return a summary table."""
    results = []
    for mode in OPPORTUNITY_MODES:
        if mode == "milestones_only":
            sub = labels[labels.get("is_budget_milestone", True)].copy()
        elif mode == "milestones_plus_events":
            sub = labels.copy()
        elif mode == "equal_count_fixed_grid":
            sub = labels[labels.get("is_budget_milestone", True)].copy()
        elif mode == "dense_grid":
            sub = labels.copy()
        if sub.empty:
            continue
        results.append(
            evaluate_opportunity_mode(
                labels=sub,
                run_key_columns=run_key_columns,
                score_column=score_column,
                threshold=threshold,
                mode=mode,
                g_fe_column=g_fe_column,
            )
        )
    return pd.DataFrame([r.__dict__ for r in results])
