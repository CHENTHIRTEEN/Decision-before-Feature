"""Pilot 覆盖检查与采样区间验证.

验证旧 [0.20, 0.60] 采样范围在新 G_FE 标签下是否仍然充分。

CoverageMass(T) = sum_r G_{r,T}^+ / sum_r G_{r,pilot}^+
CoverageRun(T) = #{r: G_{r,T}^+ > delta} / #{r: G_{r,pilot}^+ > delta}

只有 CoverageMass >= 0.95 且 CoverageRun >= 0.90 才能保留旧范围。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PILOT_MILESTONE_RATIOS = (
    0.10,
    0.15,
    0.20,
    0.22,
    0.24,
    0.26,
    0.28,
    0.30,
    0.34,
    0.38,
    0.42,
    0.46,
    0.50,
    0.60,
    0.70,
)
FORMAL_RANGE_RATIOS = (
    0.20,
    0.22,
    0.24,
    0.26,
    0.28,
    0.30,
    0.34,
    0.38,
    0.42,
    0.46,
    0.50,
    0.60,
)
COVERAGE_MASS_MIN = 0.95
COVERAGE_RUN_MIN = 0.90


@dataclass(frozen=True)
class CoverageResult:
    coverage_mass: float
    coverage_run: float
    passes: bool
    n_positive_pilot_runs: int
    n_positive_formal_runs: int


def _run_best_positive(g_fe_values: np.ndarray, delta: float) -> float:
    """G_r^+ = max(0, max_t G_{r,t})."""
    if g_fe_values.size == 0:
        return 0.0
    return max(0.0, float(np.max(g_fe_values)))


def compute_coverage(
    *,
    pilot_labels: pd.DataFrame,
    run_key_columns: list[str],
    fe_ratio_column: str = "FE_ratio",
    g_fe_column: str = "g_fe",
    delta_practical: float,
    formal_ratios: tuple[float, ...] = FORMAL_RANGE_RATIOS,
    pilot_ratios: tuple[float, ...] = PILOT_MILESTONE_RATIOS,
) -> CoverageResult:
    """Check whether the formal opportunity range covers enough positive efficacy."""
    ratios = np.asarray(pilot_labels[fe_ratio_column].astype(float))
    g_fe = np.asarray(pilot_labels[g_fe_column].astype(float))
    runs = pilot_labels[run_key_columns].astype(str).agg("|".join, axis=1).to_numpy()

    pilot_mask = np.isin(ratios, np.asarray(pilot_ratios))
    formal_mask = np.isin(ratios, np.asarray(formal_ratios))

    unique_runs = np.unique(runs)
    pilot_best = np.zeros(len(unique_runs))
    formal_best = np.zeros(len(unique_runs))
    for i, run in enumerate(unique_runs):
        run_mask = runs == run
        pilot_best[i] = _run_best_positive(g_fe[run_mask & pilot_mask], delta_practical)
        formal_best[i] = _run_best_positive(g_fe[run_mask & formal_mask], delta_practical)

    total_mass = float(np.sum(pilot_best))
    formal_mass = float(np.sum(formal_best))
    coverage_mass = formal_mass / total_mass if total_mass > 0 else 1.0

    positive_pilot = pilot_best > delta_practical
    positive_formal = formal_best > delta_practical
    n_positive_pilot = int(np.sum(positive_pilot))
    n_positive_formal = int(np.sum(positive_formal))
    coverage_run = (
        n_positive_formal / n_positive_pilot
        if n_positive_pilot > 0
        else 1.0
    )

    passes = coverage_mass >= COVERAGE_MASS_MIN and coverage_run >= COVERAGE_RUN_MIN
    return CoverageResult(
        coverage_mass=coverage_mass,
        coverage_run=coverage_run,
        passes=passes,
        n_positive_pilot_runs=n_positive_pilot,
        n_positive_formal_runs=n_positive_formal,
    )
