# 15h · Task 14A 就绪判定（Task 13.1 GO Gate）

- 日期：2026-08-29。Gate 条件（工作单 §28）：$\text{Verdict}_1\in\{S1,S2\}$ 且 $\text{Verdict}_3\in\{W1,W2\}$。

## 1. 三层 Verdict 汇总

| Verdict | 结果 | 依据 |
|---|---|---|
| 1 Behavior vs Strong Baselines | **S2 CONDITIONAL** | Raw M2 全方向点估计为正；MA vs lookup/Continue 的证据较强（lookup CI>0），bbob 全部 CI 穿 0；相对 Continue 近似打平（+0.005）；约 47%（bbob）的 Task 13 Δ_B 来自 RF-M0 弱于 empirical lookup（15b） |
| 2 Risk Control | **R1 RISK-CONTROL FEASIBLE** | 预注册 κ=0.5/1.0（max）同时实现：harmful 0.136/0.137→0.071/0.078（κ=0.5），switch 0.59–0.64→0.26–0.42，vs Continue/Lookup 增益不为负且不低于 raw M2，相对 raw M2 的损失差 CI 无明确劣化（κ=0.5：bbob +0.016 [−0.037,+0.065]、MA +0.001 [−0.015,+0.015]）；raw M2 不在 Pareto 前沿上（15d） |
| 3 Within-Problem Robustness | **W1 ROBUST** | 观测 +0.0188/+0.0161，100 次 permutation 经验 p=0.0099（两 suite 一致，超全部 null） |

$$
\boxed{\text{Task 14A GO：S2 + W1，且 R1 允许把 risk-aware margin policy 带入后续 deployment design}}
$$

## 2. Task 14A 执行约束（按工作单 §28–29）

1. **只做 action-space / reset 审计**：6 方向（SHADE↔L-SHADE、SHADE↔CSO、L-SHADE↔CSO）post-handoff 1000-FE commitment + fork {continue, switch other two}；
2. **同轮执行 reset controls**（SHADE-current / L-SHADE-current 的 population-preserving reset 分支）——分离 solver identity 效应与 memory/restart 效应，这是必做项；
3. **seeds 6–10**（未参与 Task 12/13 开发的 confirmation seeds，train-domain 不变，不触碰 validation/CEC）——建议采用；
4. **禁止在 14A 训练 segment Behavior**（B_global vs B_segment vs 并集留给 14B，且以 post-handoff action space non-degenerate 为前提）；
5. transition 风险分层（15g）作为分方向报告模板：shade-current 两方向的高 harmful（0.29–0.51）需在 handoff 后复检，但**不得据此删方向**；
6. R1 语义：margin policy（pooled solver δ、固定 κ 网格）可进入 deployment design 讨论，但最终 κ 的确定仍需 14A 后的确认数据，本轮不预先确定。

## 3. 封存项

- ProgressForecast@1000FE：**PG3 NO-GO 不变**；margin gate ≠ progress gate（预测的相对动作优势 ≠ 绝对 current-solver progress），不得因 margin 成功而复活；
- CEC2017 formal = PAUSED；CEC2022 = HELD OUT；CEC 不参与 threshold/risk/direction/model/portfolio 的任何选择。
