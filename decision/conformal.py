"""Conformal prediction interval calibration for G_FE.

提供 split conformal 预测区间，满足边际覆盖 P(G_FE ∈ [l, u]) >= 1 - alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConformalInterval:
    lower: np.ndarray
    upper: np.ndarray
    alpha: float
    n_calibration: int


def calibrate_split_conformal(
    *,
    calibration_predictions: np.ndarray,
    calibration_targets: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Compute split-conformal half-width from calibration residuals."""
    residuals = np.abs(
        np.asarray(calibration_predictions, dtype=float)
        - np.asarray(calibration_targets, dtype=float)
    )
    n = len(residuals)
    if n == 0:
        raise ValueError("conformal calibration requires at least one calibration sample")
    quantile_level = np.ceil((1.0 - alpha) * (n + 1) / n).clip(1, n) / n
    return float(np.quantile(residuals, quantile_level))


def conformal_interval(
    *,
    predictions: np.ndarray,
    calibration_half_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build [lower, upper] intervals from point predictions and half-width."""
    preds = np.asarray(predictions, dtype=float)
    lower = preds - calibration_half_width
    upper = preds + calibration_half_width
    return lower, upper


def evaluate_conformal_coverage(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    calibration_half_width: float,
) -> float:
    """Empirical marginal coverage on test data."""
    preds = np.asarray(predictions, dtype=float)
    targs = np.asarray(targets, dtype=float)
    lower = preds - calibration_half_width
    upper = preds + calibration_half_width
    return float(np.mean((targs >= lower) & (targs <= upper)))


def calibrate_grouped_conformal(
    *,
    oof_predictions: pd.DataFrame,
    prediction_column: str,
    target_column: str,
    group_column: str,
    alpha: float = 0.1,
) -> dict[str, float]:
    """Calibrate per-group conformal half-widths from OOF predictions."""
    groups = oof_predictions[group_column].astype(str).unique()
    half_widths: dict[str, float] = {}
    for group in groups:
        mask = oof_predictions[group_column].astype(str).eq(str(group))
        preds = oof_predictions.loc[mask, prediction_column].to_numpy(dtype=float)
        targets = oof_predictions.loc[mask, target_column].to_numpy(dtype=float)
        if len(preds) > 0:
            half_widths[str(group)] = calibrate_split_conformal(
                calibration_predictions=preds,
                calibration_targets=targets,
                alpha=alpha,
            )
    return half_widths
