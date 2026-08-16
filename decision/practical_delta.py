"""最小实际意义功效阈值 delta_practical.

delta_practical = max(delta_domain, delta_noise)

delta_noise 由共享状态重复 continuation 的功效波动分位数估计。
"""

from __future__ import annotations

import numpy as np


def estimate_delta_noise(
    g_fe_rep1: np.ndarray,
    g_fe_rep2: np.ndarray,
    alpha: float = 0.95,
) -> float:
    """delta_noise = Q_{1-alpha}(|G1 - G2| / 2).

    Parameters
    ----------
    g_fe_rep1, g_fe_rep2
        Paired efficacy estimates from two continuation repetitions
        on the *same* shared state.
    alpha
        Quantile level; default 0.95.
    """
    g1 = np.asarray(g_fe_rep1, dtype=float)
    g2 = np.asarray(g_fe_rep2, dtype=float)
    half_diff = np.abs(g1 - g2) / 2.0
    finite = np.isfinite(half_diff)
    if not finite.any():
        raise ValueError("no finite efficacy pairs for delta_noise estimation")
    return float(np.quantile(half_diff[finite], float(alpha)))


def estimate_delta_practical(
    *,
    delta_domain: float,
    g_fe_rep1: np.ndarray | None = None,
    g_fe_rep2: np.ndarray | None = None,
    alpha: float = 0.95,
) -> float:
    """delta_practical = max(delta_domain, delta_noise)."""
    dd = float(delta_domain)
    if g_fe_rep1 is not None and g_fe_rep2 is not None:
        dn = estimate_delta_noise(g_fe_rep1, g_fe_rep2, alpha=alpha)
        return max(dd, dn)
    return dd
