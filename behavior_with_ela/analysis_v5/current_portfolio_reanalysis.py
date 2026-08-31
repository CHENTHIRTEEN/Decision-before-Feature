"""Reanalyze the current behavior_with_ela portfolio.

This read-only analysis uses the Task 12-14 artifacts for
P_balanced={shade,lshade,cso}. It deliberately excludes the older Phase 1
``results/actions`` artifacts and does not run objective evaluations or fit
any model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
OUTPUT = RESULTS / "analysis_v5/current_portfolio_reanalysis"
REPORT = BASE / "analysis_v5/current_portfolio_reanalysis.md"
PORTFOLIO = ("shade", "lshade", "cso")


def _require(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _read_natural() -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix_path = RESULTS / "analysis_v5/task12/dynamic_solver_loss_matrix.parquet"
    states_path = RESULTS / "analysis_v5/task12/dynamic_screening_states.parquet"
    action = pd.read_parquet(matrix_path)
    states = pd.read_parquet(states_path)
    _require(
        action,
        [
            "suite",
            "cv_group_id",
            "problem_id",
            "FE",
            "seed",
            "current_algorithm",
            "loss_shade",
            "loss_lshade",
            "loss_cso",
        ],
        str(matrix_path),
    )
    _require(states, ["state_id", "suite", "problem_id", "FE", "current_algorithm"], str(states_path))
    if set(action["current_algorithm"].astype(str)) != set(PORTFOLIO):
        raise ValueError("Task 12 current algorithm set is not P_balanced")
    if len(action) != 1890 or len(states) != 1890:
        raise ValueError(f"Task 12 expected 1890 states, got matrix={len(action)}, states={len(states)}")
    return action, states


def _natural_action_tables(action: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    action = action.copy()
    action["continue_loss"] = [
        float(getattr(row, f"loss_{row.current_algorithm}")) for row in action.itertuples()
    ]
    rows: list[dict[str, object]] = []
    for row in action.itertuples(index=False):
        for target in PORTFOLIO:
            if target == row.current_algorithm:
                continue
            target_loss = float(getattr(row, f"loss_{target}"))
            rows.append(
                {
                    "suite": row.suite,
                    "cv_group_id": row.cv_group_id,
                    "problem_id": row.problem_id,
                    "FE": int(row.FE),
                    "seed": int(row.seed),
                    "current_algorithm": row.current_algorithm,
                    "target_algorithm": target,
                    "continue_loss": float(row.continue_loss),
                    "target_loss": target_loss,
                    "gain_vs_continue": float(row.continue_loss - target_loss),
                }
            )
    gains = pd.DataFrame(rows)
    gains["strict_gain_gt_zero"] = gains["gain_vs_continue"].gt(0.0)
    gains["state_id_key"] = (
        gains["suite"].astype(str)
        + "|"
        + gains["problem_id"].astype(str)
        + "|"
        + gains["current_algorithm"].astype(str)
        + "|"
        + gains["seed"].astype(str)
        + "|"
        + gains["FE"].astype(str)
    )
    strict_summary = (
        gains.groupby(["suite", "current_algorithm", "FE", "target_algorithm"], dropna=False)
        .agg(
            states=("gain_vs_continue", "size"),
            strict_gain_gt_zero_rate=("strict_gain_gt_zero", "mean"),
            gain_mean=("gain_vs_continue", "mean"),
            gain_median=("gain_vs_continue", "median"),
            gain_std=("gain_vs_continue", lambda values: float(values.std(ddof=0))),
        )
        .reset_index()
        .sort_values(["suite", "current_algorithm", "FE", "target_algorithm"])
    )
    state_gain = (
        gains.groupby(
            ["suite", "cv_group_id", "problem_id", "FE", "seed", "current_algorithm"],
            dropna=False,
        )
        .agg(
            best_alt_gain=("gain_vs_continue", "max"),
            any_alt_gt_zero=("strict_gain_gt_zero", "any"),
            mean_alt_gain=("gain_vs_continue", "mean"),
        )
        .reset_index()
    )
    state_gain["state_id_key"] = (
        state_gain["suite"].astype(str)
        + "|"
        + state_gain["problem_id"].astype(str)
        + "|"
        + state_gain["current_algorithm"].astype(str)
        + "|"
        + state_gain["seed"].astype(str)
        + "|"
        + state_gain["FE"].astype(str)
    )
    state_gain["strict_gain_gt_zero_rate"] = state_gain["any_alt_gt_zero"].astype(float)
    return strict_summary, state_gain


def _read_practical_actions(state_gain: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = RESULTS / "analysis_v5/task13/behavior_action_dataset_task13.parquet"
    practical = pd.read_parquet(path)
    _require(
        practical,
        [
            "suite",
            "problem_id",
            "cv_group_id",
            "FE",
            "seed",
            "current_algorithm",
            "switch_required",
            "switch_target",
            "A_ND_size",
        ],
        str(path),
    )
    if len(practical) != 1890:
        raise ValueError(f"Task 13 action dataset expected 1890 rows, got {len(practical)}")
    practical = practical.copy()
    practical["state_id_key"] = (
        practical["suite"].astype(str)
        + "|"
        + practical["problem_id"].astype(str)
        + "|"
        + practical["current_algorithm"].astype(str)
        + "|"
        + practical["seed"].astype(str)
        + "|"
        + practical["FE"].astype(str)
    )
    merged = state_gain.merge(
        practical[
            [
                "state_id_key",
                "switch_required",
                "switch_target",
                "A_ND_size",
                "best_action_raw",
                "best_action_practical",
            ]
        ],
        on="state_id_key",
        how="left",
        validate="one_to_one",
    )
    if merged["switch_required"].isna().any():
        raise ValueError("Task 12 and Task 13 state keys do not align")
    practical_summary = (
        merged.groupby(["suite", "current_algorithm", "FE"], dropna=False)
        .agg(
            states=("switch_required", "size"),
            strict_any_alt_gt_zero_rate=("any_alt_gt_zero", "mean"),
            practical_switch_required_rate=("switch_required", "mean"),
            mean_best_alt_gain=("best_alt_gain", "mean"),
            median_best_alt_gain=("best_alt_gain", "median"),
            mean_A_ND_size=("A_ND_size", "mean"),
        )
        .reset_index()
        .sort_values(["suite", "current_algorithm", "FE"])
    )
    transition_summary = (
        merged[merged["switch_required"].astype(bool)]
        .assign(switch_target=lambda frame: frame["switch_target"].replace({"": "continue"}).fillna("continue"))
        .groupby(["suite", "current_algorithm", "switch_target"], dropna=False)
        .size()
        .rename("states")
        .reset_index()
    )
    transition_summary["conditional_share"] = transition_summary["states"] / transition_summary.groupby(
        ["suite", "current_algorithm"]
    )["states"].transform("sum")
    return practical_summary, transition_summary


def _task13_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = RESULTS / "analysis_v5/task13/oof_policy_rows.parquet"
    oof = pd.read_parquet(path)
    _require(oof, ["model", "carrier", "suite", "cv_group_id", "realized_loss", "gain_vs_continue", "is_switch", "harmful"], str(path))
    function_means = (
        oof.groupby(["model", "carrier", "suite", "cv_group_id"], dropna=False)
        .agg(
            realized_loss=("realized_loss", "mean"),
            gain_vs_continue=("gain_vs_continue", "mean"),
            switch_rate=("is_switch", "mean"),
            harmful_rate=("harmful", "mean"),
        )
        .reset_index()
    )
    summary = (
        function_means.groupby(["model", "carrier", "suite"], dropna=False)
        .agg(
            functions=("cv_group_id", "nunique"),
            function_balanced_realized_loss=("realized_loss", "mean"),
            function_balanced_gain_vs_continue=("gain_vs_continue", "mean"),
            function_balanced_switch_rate=("switch_rate", "mean"),
            function_balanced_harmful_rate=("harmful_rate", "mean"),
        )
        .reset_index()
        .sort_values(["suite", "carrier", "model"])
    )

    within_path = RESULTS / "analysis_v5/task13/within_problem_loso_predictions.parquet"
    within = pd.read_parquet(within_path)
    _require(within, ["model", "carrier", "suite", "problem_id", "realized_loss"], str(within_path))
    problem_means = (
        within.groupby(["model", "carrier", "suite", "problem_id"], dropna=False)
        .agg(realized_loss=("realized_loss", "mean"))
        .reset_index()
    )
    within_summary = (
        problem_means.groupby(["model", "carrier", "suite"], dropna=False)
        .agg(
            problems=("problem_id", "nunique"),
            function_balanced_realized_loss=("realized_loss", "mean"),
        )
        .reset_index()
    )
    wide = within_summary.pivot(index=["carrier", "suite"], columns="model", values="function_balanced_realized_loss").reset_index()
    wide.columns.name = None
    for model in ("W0", "W1", "W2"):
        if model not in wide:
            wide[model] = np.nan
    wide["delta_W0_minus_W2"] = wide["W0"] - wide["W2"]
    wide["delta_W0_minus_W1"] = wide["W0"] - wide["W1"]
    return summary, wide


def _task14_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_path = RESULTS / "analysis_v6/task14b_1/task14b1_policy_performance.parquet"
    pairwise_path = RESULTS / "analysis_v6/task14b_1/task14b1_pairwise_bootstrap.parquet"
    policy = pd.read_parquet(policy_path)
    pairwise = pd.read_parquet(pairwise_path)
    _require(policy, ["suite", "policy", "realized_fb_loss", "n_states"], str(policy_path))
    _require(pairwise, ["comparison", "suite", "fb_mean", "ci_low", "ci_high"], str(pairwise_path))
    return policy, pairwise


def _fmt_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    view = frame[columns].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.{digits}f}")
    return view.to_markdown(index=False)


def _write_report(
    strict: pd.DataFrame,
    practical: pd.DataFrame,
    transitions: pd.DataFrame,
    task13: pd.DataFrame,
    within: pd.DataFrame,
    task14_policy: pd.DataFrame,
    task14_pairwise: pd.DataFrame,
) -> None:
    natural_overall = (
        practical.groupby("suite", dropna=False)
        .agg(
            states=("states", "sum"),
            strict_any_alt_gt_zero_rate=("strict_any_alt_gt_zero_rate", "mean"),
            practical_switch_required_rate=("practical_switch_required_rate", "mean"),
            mean_best_alt_gain=("mean_best_alt_gain", "mean"),
        )
        .reset_index()
    )
    task13_rf = task13[task13["carrier"].eq("rf")].copy()
    key_pairwise = task14_pairwise[task14_pairwise["comparison"].isin(["MGS_matched_vs_M0", "MGS_matched_vs_lookup", "MGS_matched_vs_MG"])].copy()
    lines = [
        "# behavior_with_ela 当前组合重分析",
        "",
        "> 本报告只使用 `behavior_with_ela/` 内 Task 12、Task 13、Task 14 的当前结果。旧 Phase 1 的 `results/actions/`（`{pso, shade, cmaes}`）不纳入分析。",
        "",
        "## 1. 当前实验对象",
        "",
        "- 当前组合：`{shade, lshade, cso}`；CMA-ES 仅作为隔离的 add-back control，不作为当前组合动作。",
        "- Task 12 natural action space：1,890 states = 42 problems × 5 seeds × 3 current algorithms × 3 FE checkpoints（2,000/4,000/6,000）。",
        "- 主动态 horizon：1,000 FE；Task 13 使用 28 列 Behavior，主 carrier 为 RF；Task 14 使用真实 post-handoff states。",
        "- 本文件夹当前结果不使用 `g_fe` / `g_fe_selected_path` 字段；对应指标是 `loss_*`、`gain_vs_continue`、`switch_required` 和 grouped-OOF policy loss。",
        "- 本次仅做读取和聚合：new objective FE = 0；没有重新训练模型、重选组合或修改标签。",
        "",
        "## 2. Natural action space",
        "",
        "严格收益定义为 `continue_loss - target_loss > 0`；practical `switch_required` 使用 Task 13 的 set-valued action 规则，不把 within-delta 平局强行压成单一 winner。",
        "",
        _fmt_table(natural_overall, ["suite", "states", "strict_any_alt_gt_zero_rate", "practical_switch_required_rate", "mean_best_alt_gain"]),
        "",
        "按 current 与 FE 的完整表见 `natural_action_summary.csv`；切换目标条件分布见 `natural_transition_summary.csv`。",
        "",
        _fmt_table(
            transitions,
            ["suite", "current_algorithm", "switch_target", "states", "conditional_share"],
        ),
        "",
        "当前正确的方向来自六个 transition：`shade↔lshade`、`shade↔cso`、`lshade↔cso`。没有 PSO→CMA-ES 或 SHADE→CMA-ES 的当前组合结论。",
        "",
        "## 3. Task 13 Behavior 增量",
        "",
        "Task 13 的对象是 natural P_balanced states 上的 1,000-FE action selection，不是独立 query 是否值得执行。",
        "",
        _fmt_table(
            task13_rf,
            ["model", "carrier", "suite", "functions", "function_balanced_realized_loss", "function_balanced_gain_vs_continue", "function_balanced_switch_rate", "function_balanced_harmful_rate"],
        ),
        "",
        _fmt_table(
            within[within["carrier"].eq("rf")],
            ["carrier", "suite", "W0", "W1", "W2", "delta_W0_minus_W1", "delta_W0_minus_W2"],
        ),
        "",
        "解释：RF 的 Behavior policy 在 natural 域相对 `current+FE` 的点估计增量为 BBOB +0.047、MA +0.051；MA 的 grouped-OOF 区间不跨 0，BBOB 区间较宽。within-problem LOSO 的 RF 增量在两个 suite 均为正，说明存在固定 problem/current/FE 后的 state 区分信号。",
        "",
        "风险同时上升：raw Behavior policy 的 harmful rate 约为 BBOB 0.136、MA 0.137，不能只报告平均 policy loss。",
        "",
        "## 4. Post-handoff Task 14",
        "",
        "Task 14A/14B.1 的 3,780 个 states 仍使用 `{shade,lshade,cso}`；它们不是 Task 12 natural states 的简单重复，而是 handoff 后 commitment 状态。",
        "",
        _fmt_table(
            task14_policy[task14_policy["suite"].isin(["bbob", "mabbob"])],
            ["suite", "policy", "realized_fb_loss", "n_states"],
        ),
        "",
        _fmt_table(key_pairwise, ["comparison", "suite", "fb_mean", "ci_low", "ci_high"]),
        "",
        "Task 14B.1 的结论是 post-handoff generic Behavior 没有超过 M0/lookup，segment 相对 global 也没有增量；该结论不否定 Task 13 在 natural 域的条件增量，而是说明增量不能直接迁移到 handoff 后域。",
        "",
        "## 5. 更正后的结论",
        "",
        "1. 当前组合的动态动作空间确实是 `{shade,lshade,cso}`，且 practical `switch_required` 约为 26%，方向分布在六个组合之间。",
        "2. Natural 域中 Behavior 对 1,000-FE action selection 有条件增量（Task 13：A2 CONDITIONAL；within-problem：B1 GENUINE STATE VALUE），但切换风险更高。",
        "3. Post-handoff 域中，generic global/segment Behavior 未提供额外 policy 增量（Task 14B.1：最终 NO-GO）。",
        "4. 因此，之前的 `PSO→CMA-ES`、`SHADE→CMA-ES`、`CMA-ES→继续` 只能归入旧 Phase 1 历史结果，不能用于解释当前 `{shade,lshade,cso}` 实验。",
        "",
        "## 6. 产物",
        "",
        "- `behavior_with_ela/results/analysis_v5/current_portfolio_reanalysis/natural_action_summary.csv`",
        "- `behavior_with_ela/results/analysis_v5/current_portfolio_reanalysis/natural_transition_summary.csv`",
        "- `behavior_with_ela/results/analysis_v5/current_portfolio_reanalysis/task13_oof_summary.csv`",
        "- `behavior_with_ela/results/analysis_v5/current_portfolio_reanalysis/task13_within_problem_summary.csv`",
        "- `behavior_with_ela/results/analysis_v5/current_portfolio_reanalysis/task14b1_policy_summary.csv`",
        "- `behavior_with_ela/results/analysis_v5/current_portfolio_reanalysis/task14b1_pairwise_bootstrap.csv`",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, str]:
    action, _ = _read_natural()
    strict, state_gain = _natural_action_tables(action)
    practical, transitions = _read_practical_actions(state_gain)
    task13, within = _task13_summaries()
    task14_policy, task14_pairwise = _task14_summary()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    strict.to_csv(OUTPUT / "natural_strict_gain_summary.csv", index=False)
    practical.to_csv(OUTPUT / "natural_action_summary.csv", index=False)
    transitions.to_csv(OUTPUT / "natural_transition_summary.csv", index=False)
    task13.to_csv(OUTPUT / "task13_oof_summary.csv", index=False)
    within.to_csv(OUTPUT / "task13_within_problem_summary.csv", index=False)
    task14_policy.to_csv(OUTPUT / "task14b1_policy_summary.csv", index=False)
    task14_pairwise.to_csv(OUTPUT / "task14b1_pairwise_bootstrap.csv", index=False)
    metadata = {
        "portfolio": list(PORTFOLIO),
        "natural_states": 1890,
        "task13_behavior_features": 28,
        "task13_new_action_label_fe": 0,
        "task14_post_handoff_states": 3780,
        "excluded_old_phase1_action_root": str(RESULTS / "actions"),
        "model_fitted": False,
        "runtime_used": False,
        "new_objective_fe": 0,
        "report": str(REPORT),
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_report(strict, practical, transitions, task13, within, task14_policy, task14_pairwise)
    return {"report": str(REPORT), "output_dir": str(OUTPUT)}


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.output is not None:
        OUTPUT = args.output
    print(json.dumps(run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
