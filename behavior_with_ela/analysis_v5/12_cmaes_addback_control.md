# 12 · CMAES Add-Back Control（Task 12P）

- 日期：2026-08-30
- 本报告与 11 报告合并撰写于 `11_portfolio_pareto_selection_and_12_cmaes_addback.md`（§2）。执行顺序合规：add-back 分支在采集期隔离存放，分析在 P_balanced 预先固定后进行，未反馈任何选择。

## 1. 结果（4 动作空间：P_balanced ∪ {CMAES}）

| suite | SBS(4) | cmaes practical win | Δ_dynamic(3→4) | max dominance(4) | P(varies)(4) | verdict |
|---|---|---:|---:|---:|---:|---|
| bbob | cmaes | 0.271 | 0.114→0.094（0.83） | 0.490 | 0.651 | **NO COLLAPSE** |
| mabbob | cmaes | 0.374 | 0.085→0.083（0.97） | 0.411 | 0.689 | **NO COLLAPSE** |

practical entropy：1.25/1.31 → 1.74/1.75 bits（加入 cmaes 后动作多样性上升）。

## 2. 判读

1. cmaes 是 4 动作空间的最强单项（SBS），但 practical 支配率 0.41–0.49，与 Task 11 的 0.92 支配完全不同；
2. Δ_dynamic 基本无损（比值 0.83/0.97），action space 没有坍塌；
3. 机制：Task 11 的支配是 cmaes 轨迹占据的状态分布的属性；当轨迹由互补候选占据时，cmaes 只是强动作之一——**dominance travels with the state distribution**（实验观察，待外部确认后才能作为论文结论）。

## 3. 含义

- P_balanced（含或不含 cmaes 作为第五动作）都保有非退化动态空间；正式组合是否加入 cmaes 属于下一阶段在 deployment 目标下的设计决策（cmaes 提升 SBS 但压缩多样性约 0.02 log10 的 Δ_dynamic）；
- ProgressForecast 仍不恢复：需先通过 Behavior Incremental Test（工作单 §42）。
