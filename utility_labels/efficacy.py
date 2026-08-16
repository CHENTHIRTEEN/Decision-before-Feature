"""ELA Efficacy (G_FE) label computation.

方案 A 核心标签：等总 FE 下的性能功效。

G_FE = log((E_skip + epsilon_p) / (E_query + epsilon_p))

其中 epsilon_p 是问题尺度协变稳定项，runtime 不进入主标签。
"""

from __future__ import annotations

import numpy as np

EPSILON_0 = 1e-30
ETA_DEFAULT = 0.01
EFFICACY_FORMULA_PROTOCOL = "equal_total_fe_log_gap_ratio_v1"

# ── 稳定项 ──────────────────────────────────────────────


def problem_scale_epsilon(
    *,
    prefix_gap: np.ndarray | float,
    problem_scale: np.ndarray | float,
    eta: float = ETA_DEFAULT,
    epsilon_0: float = EPSILON_0,
) -> np.ndarray | float:
    """epsilon_p = eta * max(E_prefix, S_p, epsilon_0).

    All three terms are non-negative. ``problem_scale`` is a suite-level
    reference scale (e.g. 1.0 for normalised benchmarks, or the known
    domain extent for engineering problems).
    """
    e_prefix = np.asarray(prefix_gap, dtype=float)
    s_p = np.asarray(problem_scale, dtype=float)
    return float(eta) * np.maximum(np.maximum(e_prefix, s_p), float(epsilon_0))


# ── 主功效 ──────────────────────────────────────────────


def efficacy_log(
    *,
    gap_skip: np.ndarray,
    gap_query: np.ndarray,
    epsilon_p: np.ndarray | float,
) -> np.ndarray:
    """G_FE = log((E_skip + eps) / (E_query + eps)).

    Uses natural log so that G_FE = log(10) corresponds to one order of
    magnitude improvement.  Both gaps must be non-negative and finite.
    """
    e_skip = np.asarray(gap_skip, dtype=float)
    e_query = np.asarray(gap_query, dtype=float)
    eps = np.asarray(epsilon_p, dtype=float)
    if not np.isfinite(e_skip).all() or (e_skip < 0.0).any():
        raise ValueError("efficacy gap_skip must be finite and non-negative")
    if not np.isfinite(e_query).all() or (e_query < 0.0).any():
        raise ValueError("efficacy gap_query must be finite and non-negative")
    if not np.isfinite(eps).all() or (eps <= 0.0).any():
        raise ValueError("efficacy epsilon_p must be finite and strictly positive")
    return np.log((e_skip + eps) / (e_query + eps))


def efficacy_bounded(
    *,
    gap_skip: np.ndarray,
    gap_query: np.ndarray,
    epsilon_p: np.ndarray | float,
) -> np.ndarray:
    """Bounded sensitivity metric in [-1, 1].

    G_bounded = (E_skip - E_query) / max(E_skip, E_query, epsilon_p)
    """
    e_skip = np.asarray(gap_skip, dtype=float)
    e_query = np.asarray(gap_query, dtype=float)
    eps = np.asarray(epsilon_p, dtype=float)
    denom = np.maximum(np.maximum(e_skip, e_query), eps)
    return (e_skip - e_query) / denom


# ── 派生标签 ────────────────────────────────────────────


def meaningful_efficacy_label(
    g_fe: np.ndarray,
    delta_practical: float,
) -> np.ndarray:
    """Y_eff = I[G_FE > delta_practical]."""
    return np.asarray(g_fe, dtype=float) > float(delta_practical)


def positive_efficacy_label(g_fe: np.ndarray) -> np.ndarray:
    """I[G_FE > 0]."""
    return np.asarray(g_fe, dtype=float) > 0.0
