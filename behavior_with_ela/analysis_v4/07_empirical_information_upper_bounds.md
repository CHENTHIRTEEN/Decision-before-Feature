# 07 · Empirical Information Upper Bounds（Task 11K）

- 日期：2026-08-30
- 方法：在训练任何 Behavior model 之前，比较不同信息集下的经验最优动作策略（函数平衡平均损失）与 state-wise 上界的差距。策略构造：对每个信息组（group）与动作 a，计算函数平衡平均损失，取组内最优动作；策略损失 = 各状态按其所属组的选中动作的平均损失。**任何模型都不可能超过这些 oracle**。
- 产物：`analysis_v4/task11/empirical_upper_bounds.parquet`。

## 1. 主表（log10 损失，越小越好；pooled BBOB+MA）

| horizon | 信息集 | 全组选中动作 | 策略损失 | state-wise 上界 | 剩余可得增益 $\Delta_{state,h}$ | continue 参照 |
|---|---|---|---:|---:|---:|---:|
| 500 | current only | continue | −3.385 | −3.394 | 0.009 | −3.385 |
| 500 | route | 全部组 = continue | −3.385 | −3.394 | 0.009 | −3.385 |
| 500 | route+FE | 全部 12 组 = continue | −3.385 | −3.394 | 0.009 | −3.385 |
| **1000** | current only | continue | −3.763 | −3.776 | **0.013** | −3.763 |
| **1000** | route | 全部组 = continue | −3.763 | −3.776 | **0.013** | −3.763 |
| **1000** | route+FE | 全部 12 组 = continue | −3.763 | −3.776 | **0.013** | −3.763 |
| terminal | current only | continue | −7.113 | −7.155 | 0.042 | −7.113 |
| terminal | route | 全部组 = continue | −7.113 | −7.155 | 0.042 | −7.113 |
| terminal | route+FE | 全部 12 组 = continue | −7.113 | −7.155 | 0.042 | −7.113 |

## 2. 判读（这是本轮最关键的表）

1. **每个信息层级、每个 group、每个 horizon 的经验最优动作都是 continue CMAES**。知道 route、FE 甚至全部能知道的东西，经验最优策略不发生任何变化。
2. **$\Delta_{state,1000}=0.013\log_{10}\approx 0$**（工作单判据原文）：完美逐状态选择相对 always-continue 的可得增益只有 0.013 log10，而 1000-FE 的重复噪声 $\delta_{1000,95}\approx0.098$——oracle 增益比噪声低约 7 倍。terminal 上也只有 0.042（$\delta_{T,95}\approx0.267$ 的六分之一）。
3. 因此**不是"模型不够好"，而是任务本身没有价值密度**：任何 Behavior/ELA/Progress 特征、任何模型类、任何阈值，在本 portfolio 的 mature post-handoff 动作任务上的可得改进都被封顶在 0.01–0.04 log10。
4. 与 Task 9 的对照：initial one-switch 任务中 state-wise oracle 相对 always-continue 的增益约 1.85–2.21（Task 9/10 的 gain 口径），其中 prefix 身份吸收了几乎全部（Task 9 报告）；本轮 mature post-handoff 域中连 oracle 增益本身都消失了——**handoff 完成后问题从"可学习但已退化为查表"进一步恶化为"无可学习信号"**。

## 3. 对 11N（secondary SHADE-route）的证据

- $P(a^\star_{1000}=\text{shade})$（argmax 口径、含并列）均值 15.4%，但其中绝大多数是"三动作并列、无真实改善"的状态；真实 escape（$G_{1000}>\delta_{1000,95}$）仅 3.3%；
- 跨 functions 的稳定 $G_{1000}(\text{cmaes}\to\text{shade})>\delta_{1000,95}$ 模式不存在（escape 率 3.3%，未超过阈值意义上的"非平凡"）。
- **结论：不建议下一阶段生成 mature SHADE post-handoff routes**——在当前 portfolio 上继续投入分支成本没有价值密度支撑。

## 4. 结论

结合 04 报告：本轮的三重证据（I 分布、K 上界、I4 escape）一致指向 **Action-space = DEGENERATE**。下一阶段应按工作单 Case 3 进入 Portfolio Sufficiency Pilot（候选沿用预定义：lbest-PSO、L-SHADE、IPOP-CMAES），而不是训练任何 segment/progress 模型。
