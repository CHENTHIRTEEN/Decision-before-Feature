# 08 · ProgressForecast 数据结构与含义（Task 11O + GO 条件核对）

- 日期：2026-08-30
- 本轮不训练 ProgressForecast；仅按工作单要求为未来准备好数据结构并核对 GO 条件。

## 1. 已保存的 Progress 数据结构

`results/analysis_v4/task11/progress_labels_current.parquet`（4,680 行，每 mature state 一行）：

| 字段 | 定义 |
|---|---|
| `progress_500_current` | $(\ell_t-\ell^{continue}_{t+500})/(500/B)$，**只用 continue 分支**的 500-FE 表现 |
| `progress_1000_current` | $(\ell_t-\ell^{continue}_{t+1000})/(1000/B)$，同上 |
| 附带 | state_id、route、source/current、FE、dwell、cv_group、global `bg_*` 28 维、segment `bs_*` 28 维（在 `mature_state_behavior.parquet`，按 state_id 关联） |

未使用 best-candidate 分支构造 progress 标签；未训练任何 Progress Predictor。

## 2. ProgressForecast GO 条件核对（工作单 §22）

| 条件 | 内容 | 结果 |
|---|---|---|
| A | 1000-FE post-handoff action space 至少 WEAKLY NON-DEGENERATE | **不满足**（04/07 报告：DEGENERATE，oracle 增益 0.013 << δ95≈0.098） |
| B | 存在 continue vs switch 的实质 variation | **不满足**（escape≈3%，practical best=continue 占 92–96%） |
| C | $B^{segment}$ 或 $[B^{global},B^{segment}]$ 相对 source+current+FE+dwell 有增量 | **不适用**（无动作差异可预测；且本轮未训练任何模型） |

三条条件全部不满足或不适用的根因是同一个：**在当前三算法 portfolio 下，mature post-handoff 段内不存在超越噪声的动作价值**（可得上界 0.013 log10）。这阻断的不是 ProgressForecast 的预测能力假设，而是其下游动作选择的意义。

## 3. Progress-vs-Action 的关系（Low Progress ≠ Switch 的先验检验）

本轮数据允许一个零成本的先导检查（正式检验留给 ProgressForecast 可行性研究）：progress 标签在 4,680 个状态上的分布显示，无论 progress 高低，continue 都是经验最优动作（07 报告）。因此即便未来 Behavior 能完美预测 progress，"低 progress → 切换"的策略在本 portfolio 上的期望收益上界仍是 0.013 log10。**Progress Gate 的价值同样被动作空间封顶。**

## 4. 成本视角（工作单 §27 的 branch 节省估计）

若未来只对 progress 触发的状态做分支（以本轮 progress_1000_current 的训练分位数触发）：

| 触发规则 | 触发率 ρ | 分支成本相对全量审计 |
|---|---:|---:|
| bottom 20% | 0.20 | −80%（≈23M vs 115M FE） |
| bottom 30% | 0.20（并列堆积） | −80% |
| bottom 40% | 0.40 | −60% |

（分位数在 progress 退化区有大量并列，bottom20 与 bottom30 的触发率相同。）该估计仅说明未来 Gate 的成本效率，不构成训练 Gate 的依据。

## 5. 结论

- **不允许进入 ProgressForecast full pipeline**（条件 A/B 失败）。
- 也不允许进入 Behavior incremental test：在动作空间退化的域上测试特征增量没有可兑现的目标。
- 本轮保存的 states × multi-horizon × 双 Behavior 表为未来（若 Portfolio Pilot 引入互补算法使 action space 复活）提供了可直接复用的标注基础设施：同一批 checkpoint 协议可扩展到新动作，无需重跑 route 基座。
