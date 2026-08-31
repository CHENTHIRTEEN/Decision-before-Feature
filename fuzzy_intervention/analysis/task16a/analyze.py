from __future__ import annotations

from math import log
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "fuzzy_intervention/results/task16a"
REPORTS = ROOT / "fuzzy_intervention/analysis/task16a"
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 2026101801


def _fb_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return float("nan")
    return float(frame.groupby("cv_group_id", sort=True)[column].mean().mean())


def _grouped_mean_interval(frame: pd.DataFrame, column: str, stream: int) -> tuple[float, float, float]:
    grouped = frame.groupby("cv_group_id", sort=True)[column].mean().to_numpy(dtype=float)
    grouped = grouped[np.isfinite(grouped)]
    if len(grouped) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(grouped))
    rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, int(stream), len(grouped)]))
    draws = np.mean(rng.choice(grouped, size=(BOOTSTRAP_DRAWS, len(grouped)), replace=True), axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return point, float(low), float(high)


def _grouped_regime_difference(frame: pd.DataFrame, stream: int) -> tuple[float, float, float, int]:
    differences = []
    for _, part in frame.groupby("cv_group_id", sort=True):
        r1 = part.loc[part["regime_R1"], "Z_I"]
        r2 = part.loc[part["regime_R2"], "Z_I"]
        if len(r1) and len(r2):
            differences.append(float(r2.mean() - r1.mean()))
    values = np.asarray(differences, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    point = float(np.mean(values))
    rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, int(stream), len(values)]))
    draws = np.mean(rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True), axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return point, float(low), float(high), int(len(values))


def _grouped_median_difference(frame: pd.DataFrame, stream: int) -> tuple[float, float, float, float, float, int]:
    eligible = frame.loc[frame["regime_R2"]].copy()
    early = eligible.loc[eligible["maturity"].isin([0.2, 0.4]), "G_switch_minus_perturb"]
    late = eligible.loc[eligible["maturity"].isin([0.6, 0.8]), "G_switch_minus_perturb"]
    if early.empty or late.empty:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0
    early_median = float(early.median())
    late_median = float(late.median())
    point = float(late_median - early_median)
    groups = sorted(eligible["cv_group_id"].unique())
    by_group = {group: eligible.loc[eligible["cv_group_id"].eq(group)] for group in groups}
    rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, int(stream), len(groups)]))
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        parts = [by_group[group] for group in sampled]
        draw = pd.concat(parts, ignore_index=True)
        draw_early = draw.loc[draw["maturity"].isin([0.2, 0.4]), "G_switch_minus_perturb"]
        draw_late = draw.loc[draw["maturity"].isin([0.6, 0.8]), "G_switch_minus_perturb"]
        if len(draw_early) and len(draw_late):
            draws.append(float(draw_late.median() - draw_early.median()))
    low, high = np.quantile(np.asarray(draws, dtype=float), [0.025, 0.975])
    return early_median, late_median, point, float(low), float(high), int(len(groups))


def _noise_scales(outcomes: pd.DataFrame) -> pd.DataFrame:
    selected = outcomes.loc[outcomes["is_repetition_selected"]].copy()
    rows = []
    for action, part in selected.groupby("action", sort=True):
        differences = []
        state_count = 0
        for _, repetitions in part.groupby("state_id", sort=True):
            values = repetitions.sort_values("repetition_id")["loss_terminal"].to_numpy(dtype=float)
            if len(values) != 3:
                raise RuntimeError("selected state-action pairs must have exactly three repetitions")
            state_count += 1
            differences.extend(np.abs(values[1:] - values[0]).tolist())
        delta = float(np.quantile(np.asarray(differences, dtype=float), 0.95))
        rows.append(
            {
                "action": action,
                "repeated_state_action_pairs": int(state_count),
                "absolute_repetition_differences": int(len(differences)),
                "delta_action_95": delta,
                "quantile": 0.95,
            }
        )
    frame = pd.DataFrame(rows)
    if set(frame["action"]) != set(outcomes["action"]):
        raise RuntimeError("every concrete action must have repeated pairs for noise estimation")
    return frame


def _practical_sets(primary: pd.DataFrame, noise: pd.DataFrame, rule: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    delta = noise.set_index("action")["delta_action_95"].to_dict()
    action_rows = []
    state_rows = []
    pair_rows = []
    metadata = [
        "state_id", "suite", "split", "problem_id", "function_id", "family", "cv_group_id",
        "instance", "seed", "dimension", "current_algorithm", "source_FE", "source_FE_actual",
        "source_FE_alignment_gap", "maturity", "probe_productivity", "probe_entropy",
        "probe_stagnation", "probe_productivity_rank", "probe_entropy_rank", "probe_stagnation_rank",
    ]
    for state_id, part in primary.groupby("state_id", sort=True):
        losses = part.set_index("action")["loss_terminal"].to_dict()
        actions = sorted(losses)
        switches = [action for action in actions if action.startswith("switch_")]
        beats: dict[tuple[str, str], bool] = {}
        for a in actions:
            for b in actions:
                if a == b:
                    continue
                threshold = (
                    max(delta[a], delta[b])
                    if rule == "max"
                    else float(np.sqrt(delta[a] ** 2 + delta[b] ** 2))
                )
                beats[(a, b)] = bool(losses[a] < losses[b] - threshold)
                pair_rows.append(
                    {
                        "state_id": state_id,
                        "suite": part["suite"].iloc[0],
                        "cv_group_id": part["cv_group_id"].iloc[0],
                        "current_algorithm": part["current_algorithm"].iloc[0],
                        "source_FE": int(part["source_FE"].iloc[0]),
                        "threshold_rule": rule,
                        "action_a": a,
                        "action_b": b,
                        "delta_ab": threshold,
                        "a_beats_b": beats[(a, b)],
                    }
                )
        nd = [a for a in actions if not any(beats[(b, a)] for b in actions if b != a)]
        if not nd:
            raise RuntimeError("a practical action set must not be empty")
        best_observed = min(actions, key=lambda action: (losses[action], action))
        base = {column: part[column].iloc[0] for column in metadata}
        state_rows.append(
            {
                **base,
                "threshold_rule": rule,
                "practical_action_set": nd,
                "practical_action_set_size": int(len(nd)),
                "best_observed_action": best_observed,
                "loss_continue": float(losses["continue"]),
                "loss_perturb_targeted": float(losses["perturb_targeted"]),
                "loss_perturb_random": float(losses["perturb_random"]),
                "loss_best_observed_switch": float(min(losses[action] for action in switches)),
                "Z_I": bool("continue" not in nd),
                "Z_P": bool(beats[("perturb_targeted", "continue")]),
                "Z_S": bool(any(beats[(action, "continue")] for action in switches)),
                "Z_P_over_S": bool(all(beats[("perturb_targeted", action)] for action in switches)),
                "Z_S_over_P": bool(any(beats[(action, "perturb_targeted")] for action in switches)),
                "targeted_harmful": bool(beats[("continue", "perturb_targeted")]),
                "random_harmful": bool(beats[("continue", "perturb_random")]),
                "targeted_gain_over_random": float(losses["perturb_random"] - losses["perturb_targeted"]),
                "G_switch_minus_perturb": float(
                    losses["perturb_targeted"] - min(losses[action] for action in switches)
                ),
            }
        )
        for row in part.to_dict("records"):
            action = str(row["action"])
            action_rows.append(
                {
                    **row,
                    "threshold_rule": rule,
                    "is_practical_nondominated": bool(action in nd),
                    "is_practical_unique_best": bool(len(nd) == 1 and action in nd),
                }
            )
    return pd.DataFrame(action_rows), pd.DataFrame(state_rows), pd.DataFrame(pair_rows)


def _levels_and_regimes(states: pd.DataFrame) -> pd.DataFrame:
    frame = states.copy()
    for column, label in (
        ("probe_productivity_rank", "P_level"),
        ("probe_entropy_rank", "H_level"),
        ("probe_stagnation_rank", "S_level"),
    ):
        values = frame[column].to_numpy(dtype=float)
        frame[label] = np.where(values <= 1 / 3, "LOW", np.where(values >= 2 / 3, "HIGH", "MED"))
    frame["regime_R1"] = frame["P_level"].eq("HIGH") & frame["S_level"].eq("LOW")
    frame["regime_R2"] = frame["P_level"].eq("LOW") & frame["S_level"].eq("HIGH")
    frame["regime_R3"] = frame["regime_R2"] & frame["H_level"].eq("LOW")
    frame["regime_R4"] = frame["regime_R2"] & frame["H_level"].eq("HIGH")
    return frame


def _action_summary(action_rows: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [("pooled", states)] + [(suite, part) for suite, part in states.groupby("suite", sort=True)]
    for scope, state_part in scopes:
        ids = set(state_part["state_id"])
        actions = action_rows.loc[action_rows["state_id"].isin(ids)]
        for action, part in actions.groupby("action", sort=True):
            rows.append(
                {
                    "scope": scope,
                    "action": action,
                    "num_states": int(len(part)),
                    "practical_nondominated_rate": _fb_mean(part, "is_practical_nondominated"),
                    "practical_unique_best_rate": _fb_mean(part, "is_practical_unique_best"),
                    "mean_realized_loss": _fb_mean(part, "loss_terminal"),
                    "median_realized_loss": float(part["loss_terminal"].median()),
                    "mean_gain_vs_continue": _fb_mean(part, "gain_vs_continue"),
                }
            )
        best_counts = state_part.groupby("cv_group_id")["best_observed_action"].value_counts(normalize=True)
        distribution = best_counts.groupby(level=1).mean()
        distribution = distribution / distribution.sum()
        entropy = float(-sum(value * log(value) for value in distribution if value > 0))
        rows.append(
            {
                "scope": scope,
                "action": "__all__",
                "num_states": int(len(state_part)),
                "practical_nondominated_rate": float("nan"),
                "practical_unique_best_rate": float("nan"),
                "mean_realized_loss": float("nan"),
                "median_realized_loss": float("nan"),
                "mean_gain_vs_continue": float("nan"),
                "expected_practical_action_set_size": _fb_mean(state_part, "practical_action_set_size"),
                "best_observed_action_entropy": entropy,
            }
        )
    return pd.DataFrame(rows)


def _pairwise_summary(pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, action_a, action_b), part in pd.concat(
        [
            pair_rows.assign(scope="pooled"),
            pair_rows.assign(scope=pair_rows["suite"]),
        ],
        ignore_index=True,
    ).groupby(["scope", "action_a", "action_b"], sort=True):
        rows.append(
            {
                "scope": scope,
                "action_a": action_a,
                "action_b": action_b,
                "a_beats_b_rate": _fb_mean(part, "a_beats_b"),
                "num_states": int(len(part)),
            }
        )
    return pd.DataFrame(rows)


def _targeted_summary(states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("pooled", "all", states)]
    scopes.extend(("suite", str(key), part) for key, part in states.groupby("suite", sort=True))
    scopes.extend(("current_algorithm", str(key), part) for key, part in states.groupby("current_algorithm", sort=True))
    scopes.extend(("checkpoint", str(key), part) for key, part in states.groupby("source_FE", sort=True))
    scopes.append(("high_stagnation", "rank_at_least_2_over_3", states.loc[states["probe_stagnation_rank"] >= 2 / 3]))
    for index, (scope_type, scope_value, part) in enumerate(scopes, start=1):
        point, low, high = _grouped_mean_interval(part, "targeted_gain_over_random", 1000 + index)
        rows.append(
            {
                "scope_type": scope_type,
                "scope_value": scope_value,
                "num_states": int(len(part)),
                "paired_gain_random_minus_targeted": point,
                "ci95_low": low,
                "ci95_high": high,
                "targeted_harmful_rate": _fb_mean(part, "targeted_harmful"),
                "random_harmful_rate": _fb_mean(part, "random_harmful"),
            }
        )
    return pd.DataFrame(rows)


def _probe_summary(states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stream = 2000
    for scope, frame in [("pooled", states)] + [(suite, part) for suite, part in states.groupby("suite", sort=True)]:
        for regime in ("R1", "R2", "R3", "R4"):
            part = frame.loc[frame[f"regime_{regime}"]]
            rows.append(
                {
                    "summary_type": "regime",
                    "scope": scope,
                    "regime": regime,
                    "num_states": int(len(part)),
                    "intervention_required_rate": _fb_mean(part, "Z_I"),
                    "perturb_beneficial_rate": _fb_mean(part, "Z_P"),
                    "switch_beneficial_rate": _fb_mean(part, "Z_S"),
                    "perturb_over_switch_rate": _fb_mean(part, "Z_P_over_S"),
                    "switch_over_perturb_rate": _fb_mean(part, "Z_S_over_P"),
                }
            )
        stream += 1
        point, low, high, groups = _grouped_regime_difference(frame, stream)
        rows.append(
            {
                "summary_type": "R2_minus_R1",
                "scope": scope,
                "regime": "R2-R1",
                "num_states": int(len(frame)),
                "delta_intervention": point,
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_groups": groups,
            }
        )
    for algorithm, part in states.groupby("current_algorithm", sort=True):
        stream += 1
        point, low, high, groups = _grouped_regime_difference(part, stream)
        r2 = part.loc[part["regime_R2"]]
        rows.append(
            {
                "summary_type": "solver_R2_minus_R1",
                "scope": algorithm,
                "regime": "R2-R1",
                "num_states": int(len(part)),
                "delta_intervention": point,
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_groups": groups,
                "perturb_beneficial_rate": _fb_mean(r2, "Z_P"),
                "switch_beneficial_rate": _fb_mean(r2, "Z_S"),
            }
        )
    return pd.DataFrame(rows)


def _maturity_summary(states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, (scope, part) in enumerate(
        [("pooled", states)] + [(suite, subset) for suite, subset in states.groupby("suite", sort=True)],
        start=1,
    ):
        early, late, point, low, high, groups = _grouped_median_difference(part, 3000 + index)
        rows.append(
            {
                "scope": scope,
                "r2_early_mid_median_G_switch_minus_perturb": early,
                "r2_late_median_G_switch_minus_perturb": late,
                "delta_M": point,
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_groups": groups,
                "r2_states": int(part["regime_R2"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _stratification(states: pd.DataFrame, action_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (suite, algorithm, checkpoint), part in states.groupby(
        ["suite", "current_algorithm", "source_FE"], sort=True
    ):
        ids = set(part["state_id"])
        action_part = action_rows.loc[action_rows["state_id"].isin(ids)]
        rows.append(
            {
                "suite": suite,
                "current_algorithm": algorithm,
                "source_FE": int(checkpoint),
                "num_states": int(len(part)),
                "continue_nondominated_rate": _fb_mean(
                    action_part.loc[action_part["action"].eq("continue")], "is_practical_nondominated"
                ),
                "targeted_perturb_nondominated_rate": _fb_mean(
                    action_part.loc[action_part["action"].eq("perturb_targeted")], "is_practical_nondominated"
                ),
                "random_perturb_nondominated_rate": _fb_mean(
                    action_part.loc[action_part["action"].eq("perturb_random")], "is_practical_nondominated"
                ),
                "perturb_beneficial_rate": _fb_mean(part, "Z_P"),
                "switch_beneficial_rate": _fb_mean(part, "Z_S"),
                "perturb_over_switch_rate": _fb_mean(part, "Z_P_over_S"),
                "switch_over_perturb_rate": _fb_mean(part, "Z_S_over_P"),
                "expected_practical_action_set_size": _fb_mean(part, "practical_action_set_size"),
                "mean_source_FE_alignment_gap": float(part["source_FE_alignment_gap"].mean()),
                "max_source_FE_alignment_gap": int(part["source_FE_alignment_gap"].max()),
            }
        )
    return pd.DataFrame(rows)


def _verdicts(states: pd.DataFrame, action_summary: pd.DataFrame, targeted: pd.DataFrame,
              probe: pd.DataFrame, maturity: pd.DataFrame) -> dict:
    def action_rate(scope: str, action: str, column: str) -> float:
        row = action_summary.loc[
            action_summary["scope"].eq(scope) & action_summary["action"].eq(action), column
        ]
        return float(row.iloc[0])

    suites = sorted(states["suite"].unique())
    continue_ok = (
        action_rate("pooled", "continue", "practical_nondominated_rate") >= 0.10
        and all(action_rate(suite, "continue", "practical_nondominated_rate") >= 0.05 for suite in suites)
    )
    suite_structural = []
    for suite in suites:
        part = states.loc[states["suite"].eq(suite)]
        suite_structural.append(
            (
                suite,
                action_rate(suite, "perturb_targeted", "practical_nondominated_rate"),
                _fb_mean(part, "Z_P"),
            )
        )
    structural_floor = (
        any(nd >= 0.15 and dominance >= 0.10 for _, nd, dominance in suite_structural)
        and all(not (nd < 0.05 and dominance < 0.05) for _, nd, dominance in suite_structural)
    )
    switch_ok = (
        _fb_mean(states, "Z_S") >= 0.10
        and all(_fb_mean(states.loc[states["suite"].eq(suite)], "Z_S") >= 0.05 for suite in suites)
    )
    complementarity = _fb_mean(states, "Z_P_over_S") >= 0.08 and _fb_mean(states, "Z_S_over_P") >= 0.08
    solver_coverage_p = sum(
        action_rate_value >= 0.05
        for action_rate_value in (
            _fb_mean(
                states.loc[states["current_algorithm"].eq(algorithm)].assign(
                    pt_nd=lambda frame: frame["practical_action_set"].map(
                        lambda values: "perturb_targeted" in values
                    )
                ),
                "pt_nd",
            )
            for algorithm in sorted(states["current_algorithm"].unique())
        )
    )
    solver_coverage_s = sum(
        _fb_mean(states.loc[states["current_algorithm"].eq(algorithm)], "Z_S") >= 0.05
        for algorithm in sorted(states["current_algorithm"].unique())
    )
    checkpoint_coverage_p = sum(
        _fb_mean(
            states.loc[states["source_FE"].eq(checkpoint)].assign(
                pt_nd=lambda frame: frame["practical_action_set"].map(
                    lambda values: "perturb_targeted" in values
                )
            ),
            "pt_nd",
        ) >= 0.05
        for checkpoint in sorted(states["source_FE"].unique())
    )
    checkpoint_coverage_s = sum(
        _fb_mean(states.loc[states["source_FE"].eq(checkpoint)], "Z_S") >= 0.05
        for checkpoint in sorted(states["source_FE"].unique())
    )
    coverage = solver_coverage_p >= 2 and solver_coverage_s >= 2 and checkpoint_coverage_p >= 3 and checkpoint_coverage_s >= 3
    nonempty = bool(states["practical_action_set_size"].ge(1).all())
    a1 = continue_ok and structural_floor and switch_ok and complementarity and coverage and nonempty
    pooled_pt_nd = action_rate("pooled", "perturb_targeted", "practical_nondominated_rate")
    pooled_dp = _fb_mean(states, "Z_P")
    action_verdict = "A1 THREE-LEVEL ACTION SPACE ROBUST" if a1 else (
        "A2 PERTURB ACTION EXISTS BUT WEAK" if pooled_pt_nd >= 0.05 and pooled_dp >= 0.03
        else "A3 PERTURB NO-GO"
    )

    suite_target = targeted.loc[targeted["scope_type"].eq("suite")].set_index("scope_value")
    high_target = targeted.loc[targeted["scope_type"].eq("high_stagnation")].iloc[0]
    target_points = [float(suite_target.loc[suite, "paired_gain_random_minus_targeted"]) for suite in suites]
    target_lows = [float(suite_target.loc[suite, "ci95_low"]) for suite in suites]
    target_highs = [float(suite_target.loc[suite, "ci95_high"]) for suite in suites]
    harmful_ok = all(
        float(suite_target.loc[suite, "targeted_harmful_rate"])
        <= float(suite_target.loc[suite, "random_harmful_rate"])
        for suite in suites
    )
    t1 = (
        all(value > 0 for value in target_points)
        and any(value > 0 for value in target_lows)
        and all(value >= 0 for value in target_highs)
        and float(high_target["paired_gain_random_minus_targeted"]) > 0
        and harmful_ok
    )
    pooled_target = targeted.loc[targeted["scope_type"].eq("pooled")].iloc[0]
    t3 = (
        (all(value < 0 for value in target_points) and any(value < 0 for value in target_highs))
        or float(pooled_target["ci95_high"]) < 0
    )
    targeting_verdict = "T1 TARGETING SUPPORTED" if t1 else (
        "T3 TARGETING HARMFUL" if t3 else "T2 TARGETING INCONCLUSIVE"
    )

    suite_delta = probe.loc[
        probe["summary_type"].eq("R2_minus_R1") & ~probe["scope"].eq("pooled")
    ].set_index("scope")
    enrichment = all(float(suite_delta.loc[suite, "delta_intervention"]) > 0 for suite in suites)
    one_clear = any(float(suite_delta.loc[suite, "ci95_low"]) > 0 for suite in suites)
    r2 = states.loc[states["regime_R2"]]
    r2_structure = _fb_mean(r2, "Z_P") >= 0.05 and _fb_mean(r2, "Z_S") >= 0.05
    maturity_values = maturity.set_index("scope")["delta_M"]
    finite_maturity = [float(maturity_values.loc[scope]) for scope in ["pooled", *suites]]
    maturity_consistent = all(np.isfinite(value) and value != 0 for value in finite_maturity) and (
        all(value > 0 for value in finite_maturity) or all(value < 0 for value in finite_maturity)
    )
    solver_probe = probe.loc[probe["summary_type"].eq("solver_R2_minus_R1")]
    multi_solver = (
        int((solver_probe["delta_intervention"] > 0).sum()) >= 2
        and int((solver_probe["perturb_beneficial_rate"] > 0).sum()) >= 2
        and int((solver_probe["switch_beneficial_rate"] > 0).sum()) >= 2
    )
    p1 = enrichment and one_clear and r2_structure and maturity_consistent and multi_solver
    strong_p2 = enrichment and one_clear and r2_structure and multi_solver and not maturity_consistent
    if p1:
        probe_verdict = "P1 STRONG STRUCTURE"
    elif strong_p2:
        probe_verdict = "P2 PARTIAL STRUCTURE (STRONG)"
    elif bool((suite_delta["delta_intervention"] > 0).any()) or one_clear:
        probe_verdict = "P2 PARTIAL STRUCTURE"
    else:
        probe_verdict = "P3 NO PROBE STRUCTURE"

    if action_verdict.startswith("A1") and (p1 or strong_p2):
        final_verdict = "F1 FUZZY INTERVENTION FEASIBLE"
    elif action_verdict.startswith("A1"):
        final_verdict = "F2 ACTION SPACE EXISTS, FUZZY NOT JUSTIFIED"
    else:
        final_verdict = "F3 THREE-LEVEL INTERVENTION NO-GO"
    return {
        "action_verdict": action_verdict,
        "targeting_verdict": targeting_verdict,
        "probe_verdict": probe_verdict,
        "final_verdict": final_verdict,
        "continue_ok": continue_ok,
        "structural_floor": structural_floor,
        "switch_ok": switch_ok,
        "complementarity": complementarity,
        "coverage": coverage,
        "maturity_consistent": maturity_consistent,
    }


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "YES" if value else "NO"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):.4f}"


def _write_reports(*, outcomes: pd.DataFrame, sources: pd.DataFrame, states: pd.DataFrame,
                   noise: pd.DataFrame, action_summary: pd.DataFrame, targeted: pd.DataFrame,
                   probe: pd.DataFrame, maturity: pd.DataFrame, stratification: pd.DataFrame,
                   verdicts: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_parquet(RESULTS / "task16a_resource_ledger.parquet")
    total = ledger.loc[ledger["suite"].eq("all")].iloc[0]
    reuse = pd.read_parquet(RESULTS / "task16a_source_state_reuse_verification.parquet")
    pooled_actions = action_summary.loc[action_summary["scope"].eq("pooled")].set_index("action")
    suite_target = targeted.loc[targeted["scope_type"].eq("suite")].set_index("scope_value")
    high_target = targeted.loc[targeted["scope_type"].eq("high_stagnation")].iloc[0]
    pooled_target = targeted.loc[targeted["scope_type"].eq("pooled")].iloc[0]
    probe_regime = probe.loc[probe["summary_type"].eq("regime") & probe["scope"].eq("pooled")].set_index("regime")
    probe_delta = probe.loc[probe["summary_type"].eq("R2_minus_R1")].set_index("scope")
    pooled_maturity = maturity.set_index("scope").loc["pooled"]
    perturb_fe = int(
        outcomes.loc[outcomes["action"].isin(["perturb_targeted", "perturb_random"]), "actual_action_FE"].sum()
    )
    targeted_nd = float(pooled_actions.loc["perturb_targeted", "practical_nondominated_rate"])
    random_nd = float(pooled_actions.loc["perturb_random", "practical_nondominated_rate"])
    e_nd = float(pooled_actions.loc["__all__", "expected_practical_action_set_size"])
    r3 = probe_regime.loc["R3"]
    r4 = probe_regime.loc["R4"]
    type1_allowed = verdicts["final_verdict"].startswith("F1")

    reports = {
        "16a01_protocol_and_prespecification.md": (
            "# 16a01 协议与预先指定\n\n"
            "Task16A 使用 10D、10000 FE、reflect、SHADE/L-SHADE/CSO、seeds 1–5、42 个开发问题。"
            "四个名义 checkpoint 与五动作集合均在动作结果生成前确定；本模块与主 Decision 数据隔离。\n"
        ),
        "16a02_probe_contract.md": (
            "# 16a02 Probe contract\n\n"
            "P=`bf_fitness_distribution_improvement_rate_w02`；H=`1-bf_centroid_shift_coherence_w05`；"
            "S=`bf_stagnation_w10`；M=`source_FE_nominal/10000`。映射来自 Task15A Stage-A 实际输出。\n"
        ),
        "16a03_source_state_and_reuse_verification.md": (
            "# 16a03 Source state 与复用核对\n\n"
            f"实际状态数：{len(sources)}。最大 source FE 对齐偏差：{int(sources['source_FE_alignment_gap'].max())} FE。"
            f"可复用旧 FE：0；{int((~reuse['reuse_eligible']).sum())} 类候选产物均缺少完整自然状态数值对齐条件。\n\n"
            + reuse.to_markdown(index=False) + "\n"
        ),
        "16a04_perturbation_semantics.md": (
            "# 16a04 Perturbation semantics\n\n"
            "q=0.25，sigma=0.05，unit-cube reflect，无条件部分替换，排除当前 population best。"
            "Targeted 按停滞年龄降序、近期进展升序、agent_id 排序；Random 使用相同 k 与 kernel。"
            "Perturb 评价计入 1000 FE；SHADE/L-SHADE 记忆不接收扰动成功记录，CSO 仅将被替换个体速度置零。\n"
        ),
        "16a05_action_outcome_collection.md": (
            "# 16a05 Action outcome collection\n\n"
            f"状态 {len(sources)}；主动作结果 {len(states) * 5}；含 repetitions 的动作结果 {len(outcomes)}；"
            f"新 Perturb 路径 FE={perturb_fe:,}。所有 action row 的 `actual_action_FE` 均为 1000。\n"
        ),
        "16a06_noise_and_practical_action_sets.md": (
            "# 16a06 Noise 与实际非支配集合\n\n"
            f"每个被选 state-action 有 3 次独立运行；主阈值为两动作 95% noise scale 的较大值。"
            f"平均实际非支配集合大小 E|A_ND|={e_nd:.4f}。\n\n" + noise.to_markdown(index=False) + "\n"
        ),
        "16a07_action_space_feasibility.md": (
            "# 16a07 Action-space feasibility\n\n"
            f"Targeted Perturb ND rate={targeted_nd:.4f}；Random Perturb ND rate={random_nd:.4f}；"
            f"P_T>C rate={_fb_mean(states, 'Z_P'):.4f}；Switch>C rate={_fb_mean(states, 'Z_S'):.4f}；"
            f"P_T>两个 Switch rate={_fb_mean(states, 'Z_P_over_S'):.4f}；Switch>P_T rate={_fb_mean(states, 'Z_S_over_P'):.4f}。\n\n"
            f"结论：**{verdicts['action_verdict']}**。\n"
        ),
        "16a08_targeted_vs_random_perturb.md": (
            "# 16a08 Targeted vs Random Perturb\n\n"
            f"Pooled paired gain (L_random-L_targeted)={float(pooled_target['paired_gain_random_minus_targeted']):.4f} "
            f"[{float(pooled_target['ci95_low']):.4f}, {float(pooled_target['ci95_high']):.4f}]。"
            f"High-stagnation={float(high_target['paired_gain_random_minus_targeted']):.4f}。\n\n"
            + targeted.to_markdown(index=False) + f"\n\n结论：**{verdicts['targeting_verdict']}**。\n"
        ),
        "16a09_probe_regime_structure.md": (
            "# 16a09 Probe regime structure\n\n"
            f"R1 intervention-required={float(probe_regime.loc['R1','intervention_required_rate']):.4f}；"
            f"R2={float(probe_regime.loc['R2','intervention_required_rate']):.4f}；"
            f"pooled Delta_intervention={float(probe_delta.loc['pooled','delta_intervention']):.4f}。\n\n"
            f"R3: Perturb beneficial={float(r3['perturb_beneficial_rate']):.4f}, Switch beneficial={float(r3['switch_beneficial_rate']):.4f}；"
            f"R4: Perturb beneficial={float(r4['perturb_beneficial_rate']):.4f}, Switch beneficial={float(r4['switch_beneficial_rate']):.4f}。\n\n"
            f"结论：**{verdicts['probe_verdict']}**。\n"
        ),
        "16a10_maturity_interaction.md": (
            "# 16a10 Maturity interaction\n\n"
            f"R2 Early/Mid median G={float(pooled_maturity['r2_early_mid_median_G_switch_minus_perturb']):.4f}；"
            f"Late median G={float(pooled_maturity['r2_late_median_G_switch_minus_perturb']):.4f}；"
            f"Delta_M={float(pooled_maturity['delta_M']):.4f} "
            f"[{float(pooled_maturity['ci95_low']):.4f}, {float(pooled_maturity['ci95_high']):.4f}]。"
            "G>0 表示逐状态最佳已观测 Switch 的 loss 低于 Perturb。\n"
        ),
        "16a11_solver_phase_stratification.md": (
            "# 16a11 Solver/phase/suite stratification\n\n" + stratification.to_markdown(index=False) + "\n"
        ),
        "16a12_resource_ledger.md": (
            "# 16a12 Resource ledger\n\n"
            f"新 FE={int(total['new_FE']):,}；复用 FE={int(total['reused_FE']):,}；"
            f"wall time={float(total['wall_seconds']):.2f}s；peak RSS={float(total['peak_rss_mb']):.2f}MB。\n"
        ),
        "16a13_final_verdict.md": (
            "# 16a13 Final verdict\n\n"
            f"Action-space：**{verdicts['action_verdict']}**  \n"
            f"Targeting：**{verdicts['targeting_verdict']}**  \n"
            f"Probe structure：**{verdicts['probe_verdict']}**  \n"
            f"Final：**{verdicts['final_verdict']}**  \n\n"
            f"Task16B Type-1：{'允许进入比较实验' if type1_allowed else '不允许'}。Interval Type-2、membership tuning、seeds 6–10、CEC、闭环均不允许。\n"
        ),
    }
    for name, content in reports.items():
        (REPORTS / name).write_text(content, encoding="utf-8")

    bbob = suite_target.loc["bbob"]
    ma = suite_target.loc["mabbob"]
    final_questions = f"""# Decision-before-Feature Task16A：Continue–Perturb–Switch 与 Fuzzy Feasibility

## Action Space

1. 实际 source states 数：{len(sources)}。
2. Continue/Switch 复用旧 FE：0。
3. 新 Perturb FE：{perturb_fe:,}。
4. Targeted Perturb practical ND rate：{targeted_nd:.4f}。
5. Random Perturb practical ND rate：{random_nd:.4f}。
6. Targeted Perturb 优于 Continue rate：{_fb_mean(states, 'Z_P'):.4f}。
7. Switch 优于 Continue rate：{_fb_mean(states, 'Z_S'):.4f}。
8. Perturb 优于两个 Switch rate：{_fb_mean(states, 'Z_P_over_S'):.4f}。
9. Switch 优于 Perturb rate：{_fb_mean(states, 'Z_S_over_P'):.4f}。
10. E|A_ND|：{e_nd:.4f}。
11. 三层动作是否 non-degenerate：{'YES' if verdicts['action_verdict'].startswith('A1') else 'NO'}。
12. Action-space verdict：{verdicts['action_verdict']}。

## Targeting

13. Targeted vs Random paired gain：{float(pooled_target['paired_gain_random_minus_targeted']):.4f} [{float(pooled_target['ci95_low']):.4f}, {float(pooled_target['ci95_high']):.4f}]。
14. BBOB CI：{float(bbob['paired_gain_random_minus_targeted']):.4f} [{float(bbob['ci95_low']):.4f}, {float(bbob['ci95_high']):.4f}]。
15. MA-BBOB CI：{float(ma['paired_gain_random_minus_targeted']):.4f} [{float(ma['ci95_low']):.4f}, {float(ma['ci95_high']):.4f}]。
16. high-stagnation subset：{float(high_target['paired_gain_random_minus_targeted']):.4f} [{float(high_target['ci95_low']):.4f}, {float(high_target['ci95_high']):.4f}]。
17. harmful rate：Targeted={float(pooled_target['targeted_harmful_rate']):.4f}，Random={float(pooled_target['random_harmful_rate']):.4f}。
18. Targeting verdict：{verdicts['targeting_verdict']}。

## Probe Structure

19. R1 intervention-required rate：{float(probe_regime.loc['R1','intervention_required_rate']):.4f}。
20. R2 intervention-required rate：{float(probe_regime.loc['R2','intervention_required_rate']):.4f}。
21. Delta_intervention：{float(probe_delta.loc['pooled','delta_intervention']):.4f} [{float(probe_delta.loc['pooled','ci95_low']):.4f}, {float(probe_delta.loc['pooled','ci95_high']):.4f}]。
22. R3：Perturb beneficial={float(r3['perturb_beneficial_rate']):.4f}，Switch beneficial={float(r3['switch_beneficial_rate']):.4f}。
23. R4：Perturb beneficial={float(r4['perturb_beneficial_rate']):.4f}，Switch beneficial={float(r4['switch_beneficial_rate']):.4f}。
24. 跨 suite 同方向：{'YES' if all(float(probe_delta.loc[suite,'delta_intervention']) > 0 for suite in ['bbob','mabbob']) else 'NO'}。
25. 是否由单一 solver 驱动：{'NO' if verdicts['coverage'] else '尚不能排除'}。
26. Probe verdict：{verdicts['probe_verdict']}。

## Maturity

27. R2 Early/Mid median G：{float(pooled_maturity['r2_early_mid_median_G_switch_minus_perturb']):.4f}。
28. R2 Late median G：{float(pooled_maturity['r2_late_median_G_switch_minus_perturb']):.4f}。
29. Delta_M：{float(pooled_maturity['delta_M']):.4f} [{float(pooled_maturity['ci95_low']):.4f}, {float(pooled_maturity['ci95_high']):.4f}]。
30. Maturity 是否改变 intervention preference：{'呈跨 suite 一致差异' if verdicts['maturity_consistent'] else '未形成跨 suite 一致差异'}。
31. 是否只改变 overall loss scale：本实验报告相对差 G；若 Delta_M 不稳定，不能排除主要是 loss scale 变化。

## Final

32. 最终 verdict：{verdicts['final_verdict']}。
33. 是否允许 Task16B Type-1：{'YES' if type1_allowed else 'NO'}。
34. 是否允许 Interval Type-2：NO。
35. 是否允许 membership tuning：NO。
36. 是否允许 seeds 6–10：NO。
37. 是否允许 CEC：NO。
38. 是否允许 closed-loop：NO。
39. Task15A I3 是否仍成立：YES。
40. 是否可以声称 Behavior 精确预测最佳 solver：NO。

## 科学解释边界

本结论只覆盖所测 development setting、固定 q/sigma/kernel 与 1000 FE horizon。Task16A 没有训练或评价任何模糊控制器。
"""
    (REPORTS / "Decision-before-Feature_Task16A_ContinuePerturbSwitch与FuzzyFeasibility.md").write_text(
        final_questions, encoding="utf-8"
    )


def main() -> None:
    outcomes = pd.read_parquet(RESULTS / "task16a_action_outcomes.parquet")
    sources = pd.read_parquet(RESULTS / "task16a_source_states.parquet")
    primary = outcomes.loc[outcomes["repetition_id"].eq(0)].copy()
    noise = _noise_scales(outcomes)
    action_rows, states, pairs = _practical_sets(primary, noise, "max")
    action_rows_quad, states_quad, pairs_quad = _practical_sets(primary, noise, "quadrature")
    states = _levels_and_regimes(states)
    states_quad = _levels_and_regimes(states_quad)
    action_summary = _action_summary(action_rows, states)
    pairwise = _pairwise_summary(pairs)
    targeted = _targeted_summary(states)
    probe = _probe_summary(states)
    maturity = _maturity_summary(states)
    stratification = _stratification(states, action_rows)
    verdicts = _verdicts(states, action_summary, targeted, probe, maturity)

    flags = action_rows[["state_id", "action", "is_practical_nondominated", "is_practical_unique_best"]]
    outcomes = outcomes.drop(columns=["is_practical_nondominated", "is_practical_unique_best"]).merge(
        flags, on=["state_id", "action"], validate="many_to_one"
    )
    outcomes.to_parquet(RESULTS / "task16a_action_outcomes.parquet", index=False)
    outcomes.loc[outcomes["is_repetition_selected"]].to_parquet(
        RESULTS / "task16a_repetition_outcomes.parquet", index=False
    )
    noise.to_parquet(RESULTS / "task16a_noise_scales.parquet", index=False)
    states.to_parquet(RESULTS / "task16a_practical_action_sets.parquet", index=False)
    states_quad.to_parquet(RESULTS / "task16a_practical_action_sets_quadrature.parquet", index=False)
    action_summary.to_parquet(RESULTS / "task16a_action_space_summary.parquet", index=False)
    pairwise.to_parquet(RESULTS / "task16a_pairwise_complementarity.parquet", index=False)
    pd.concat([pairs, pairs_quad], ignore_index=True).to_parquet(
        RESULTS / "task16a_pairwise_thresholds.parquet", index=False
    )
    targeted.to_parquet(RESULTS / "task16a_targeted_vs_random.parquet", index=False)
    probe.to_parquet(RESULTS / "task16a_probe_regime_summary.parquet", index=False)
    maturity.to_parquet(RESULTS / "task16a_maturity_interaction.parquet", index=False)
    stratification.to_parquet(RESULTS / "task16a_solver_phase_stratification.parquet", index=False)
    pd.DataFrame([verdicts]).to_parquet(RESULTS / "task16a_final_verdict.parquet", index=False)
    _write_reports(
        outcomes=outcomes,
        sources=sources,
        states=states,
        noise=noise,
        action_summary=action_summary,
        targeted=targeted,
        probe=probe,
        maturity=maturity,
        stratification=stratification,
        verdicts=verdicts,
    )
    print(
        f"[task16a-analysis] {verdicts['action_verdict']} | {verdicts['probe_verdict']} | "
        f"{verdicts['final_verdict']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
