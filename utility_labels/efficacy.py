"""ELA Efficacy (G_FE) label computation.

方案 A 核心标签：等总 FE 下的性能功效。

G_FE = log((E_skip + epsilon_p) / (E_query + epsilon_p))

其中 epsilon_p 是问题尺度协变稳定项，runtime 不进入主标签。

支持 paired continuation repetitions 以稳定标签噪声。
"""

from __future__ import annotations

import numpy as np

EPSILON_0 = 1e-30
ETA_DEFAULT = 0.01
EFFICACY_FORMULA_PROTOCOL = "equal_total_fe_log_gap_ratio_v1"

# Paired continuation repetitions protocol
PAIRED_REPETITION_PROTOCOL = "paired_continuation_repetitions_v1"
DEFAULT_REPETITIONS = 3

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


# ── Paired continuation repetitions ─────────────────────────────────


def efficacy_from_repetitions(
    gap_skip_reps: list[np.ndarray],
    gap_query_reps: list[np.ndarray],
    epsilon_p: np.ndarray | float,
    aggregation: str = "median",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """从多次 continuation 计算稳健 G_FE 及统计量。

    Parameters
    ----------
    gap_skip_reps : list of np.ndarray
        每个元素是一次 no-query continuation 的 gap_skip 数组。
    gap_query_reps : list of np.ndarray
        每个元素是一次 query continuation 的 gap_query 数组。
    epsilon_p : np.ndarray or float
        问题尺度协变稳定项。
    aggregation : str
        聚合方式："median" (默认)、"mean"、"trimmed_mean"。

    Returns
    -------
    g_fe_robust : np.ndarray
        稳健的 G_FE 值。
    stats : dict
        包含 g_fe_rep_*, g_fe_median, g_fe_mean, g_fe_std, g_fe_ci_low, g_fe_ci_high, sign_flip_rate。
    """
    if len(gap_skip_reps) != len(gap_query_reps):
        raise ValueError("gap_skip_reps and gap_query_reps must have the same length")
    if len(gap_skip_reps) < 1:
        raise ValueError("at least one repetition is required")

    n_reps = len(gap_skip_reps)
    n_states = len(gap_skip_reps[0])

    # 确保所有 repetition 都有相同数量的 states
    for i, (s, q) in enumerate(zip(gap_skip_reps, gap_query_reps)):
        if len(s) != n_states:
            raise ValueError(f"repetition {i}: gap_skip has {len(s)} states, expected {n_states}")
        if len(q) != n_states:
            raise ValueError(f"repetition {i}: gap_query has {len(q)} states, expected {n_states}")

    eps = np.asarray(epsilon_p, dtype=float)
    if eps.shape != (n_states,) and eps.shape != ():
        raise ValueError(f"epsilon_p must be scalar or length {n_states}, got shape {eps.shape}")

    # 计算每次 repetition 的 G_FE
    g_fe_reps = []
    for i in range(n_reps):
        g_fe = efficacy_log(
            gap_skip=gap_skip_reps[i],
            gap_query=gap_query_reps[i],
            epsilon_p=eps,
        )
        g_fe_reps.append(g_fe)

    # Stack 成 (n_reps, n_states)
    g_fe_all = np.column_stack(g_fe_reps)  # shape: (n_states, n_reps)

    # 计算统计量
    g_fe_median = np.median(g_fe_all, axis=1)
    g_fe_mean = np.mean(g_fe_all, axis=1)
    g_fe_std = (
        np.std(g_fe_all, axis=1, ddof=1)
        if n_reps > 1
        else np.zeros(n_states, dtype=float)
    )

    # With one repetition there is no sampling interval to estimate.
    if n_reps > 1:
        from scipy.stats import t

        t_critical = t.ppf(0.975, df=n_reps - 1)
        g_fe_ci_margin = t_critical * g_fe_std / np.sqrt(n_reps)
    else:
        g_fe_ci_margin = np.zeros(n_states, dtype=float)
    g_fe_ci_low = g_fe_median - g_fe_ci_margin
    g_fe_ci_high = g_fe_median + g_fe_ci_margin

    # sign flip rate: 每个 state 里，repetition 之间 sign 变化的比例
    signs = np.sign(g_fe_all)  # -1, 0, +1
    # 对每个 state，计算 sign 变化次数
    sign_flips = []
    for state_idx in range(n_states):
        state_signs = signs[state_idx]
        # 比较连续 repetition 之间的 sign 变化
        flips = 0
        for j in range(1, n_reps):
            if state_signs[j] != state_signs[j - 1] and state_signs[j] != 0 and state_signs[j - 1] != 0:
                flips += 1
        sign_flips.append(flips / max(n_reps - 1, 1))
    sign_flip_rate = np.array(sign_flips)

    # 选择聚合方式
    if aggregation == "median":
        g_fe_robust = g_fe_median
    elif aggregation == "mean":
        g_fe_robust = g_fe_mean
    elif aggregation == "trimmed_mean":
        # 小样本不截尾；只有至少三次重复时才截去两端极值。
        sorted_vals = np.sort(g_fe_all, axis=1)
        trim = n_reps // 4
        g_fe_robust = (
            g_fe_mean
            if trim == 0 or 2 * trim >= n_reps
            else np.mean(sorted_vals[:, trim:-trim], axis=1)
        )
    else:
        raise ValueError(f"unsupported aggregation: {aggregation}")

    # 构建 repetition 字段名
    rep_fields = {}
    for i in range(n_reps):
        rep_fields[f"g_fe_rep_{i + 1}"] = g_fe_all[:, i]

    stats = {
        **rep_fields,
        "g_fe_median": g_fe_median,
        "g_fe_mean": g_fe_mean,
        "g_fe_std": g_fe_std,
        "g_fe_ci_low": g_fe_ci_low,
        "g_fe_ci_high": g_fe_ci_high,
        "g_fe_ci_width": g_fe_ci_high - g_fe_ci_low,
        "sign_flip_rate": sign_flip_rate,
        "n_repetitions": n_reps,
    }

    return g_fe_robust, stats
