# 04 · Mature Post-Handoff Action-Space 审计（Task 11I）

- 日期：2026-08-30
- 数据：4,680 个 mature post-handoff 状态 × 3 动作 × 3 horizon（真实标签，未训练任何模型）。practical 判定用 03 报告的 $\delta_{h,95}$。
- 产物：`analysis_v4/task11/{best_action_distributions,conditional_entropies,practical_action_sets,escape_rates}.parquet`。

## I1. Best-action 分布（argmin，按 route；pooled BBOB+MA）

| horizon | route | continue | pso | shade |
|---|---|---:|---:|---:|
| 500 | R0 native | 0.930 | 0.006 | 0.064 |
| 500 | R1 pso→cmaes | 0.910 | 0.010 | 0.080 |
| 500 | R2 shade→cmaes | 0.921 | 0.010 | 0.069 |
| 1000 | R0 | 0.931 | 0.010 | 0.060 |
| 1000 | R1 | 0.917 | 0.017 | 0.066 |
| 1000 | R2 | 0.919 | 0.026 | 0.055 |
| terminal | R0 | 0.872 | 0.077 | 0.051 |
| terminal | R1 | 0.844 | 0.096 | 0.060 |
| terminal | R2 | 0.849 | 0.072 | 0.078 |

按 suite / family / definition / FE 的分层（`best_action_distributions.parquet`）同型：continue 占 84–96%。argmin 的少数偏离大量来自并列/近并列状态（见 I3：δ95 下约一半状态没有唯一 practical winner）。

## I2. Conditional entropy（bits；最大值 $\log_2 3=1.585$）

| horizon | $H(A^\star_h)$ | $H(\cdot|route)$ | $H(\cdot|route,FE)$ | $H(\cdot|source,current,FE)$ |
|---|---:|---:|---:|---:|
| 500 | 0.439 | 0.438 | 0.423 | 0.423 |
| 1000 | 0.454 | 0.452 | 0.438 | 0.438 |
| terminal | 0.741 | 0.738 | 0.722 | 0.722 |

条件化几乎不降低熵——剩余"不确定性"主要是近并列噪声，而非可利用的结构性变化。

## I3. Practical best-action set

| horizon | δ | 平均 acceptable 数 | 唯一 best 占比 |
|---|---|---:|---:|
| 500 | 95 | 1.889 | 0.510 |
| 500 | 50 | 1.211 | 0.891 |
| 1000 | 95 | 1.817 | 0.543 |
| 1000 | 50 | 1.182 | 0.903 |
| terminal | 95 | 1.878 | 0.490 |
| terminal | 50 | 1.344 | 0.752 |

δ95 下约一半状态的所有三个动作都 practical 等价——真实 margin 远小于噪声 δ95 本身（见 07 报告 K 段）。

## I4. CMAES escape opportunity（$P(\max_{a\ne\text{cmaes}}G_h>\delta_{h,95})$）

| horizon | R0 native | R1 pso→cmaes | R2 shade→cmaes |
|---|---:|---:|---:|
| 500 | 0.033 | 0.039 | 0.035 |
| **1000** | **0.035** | **0.035** | **0.029** |
| terminal | 0.047 | 0.053 | 0.074 |

**RQ3 答案：CMAES 在 mature post-handoff 状态的实际 escape rate 在 1000-FE horizon 仅为 2.9–3.5%**，且跨 route 几乎相同（native 与 transfer 无差异）；terminal 也仅 4.7–7.4%。当前三算法 portfolio 下，mature cmaes 状态几乎不存在超越短程噪声的离开机会。

## Verdict A（Action-space）：**DEGENERATE**

判据逐条核对（工作单 §23C）：
1. 1000-FE practical best 几乎始终为 continue cmaes（91.7–93.1%，route+FE 分组的经验最优动作全部是 continue）；✅
2. route+FE 已解释几乎全部 attainable action value（经验最优策略在各信息层级无差别，全部退化为 always-continue）；✅
3. state-wise upper bound 相对 always-continue 的提升仅 0.013 log10（1000 horizon），比噪声 δ95 低一个数量级；✅
4. CMAES escape 具有可重复但接近零的结构（≈3%，低于 5% 且不随 route 变化）。✅

$$\boxed{\text{Repeated action space is degenerate under current portfolio}}$$

按工作单 §23，若最终确认，下一阶段转入 **Portfolio Sufficiency Pilot**（lbest-PSO / L-SHADE / IPOP-CMAES），而不是 ProgressForecast；不立即扩 seed（本判定已远超"borderline"，扩样本不改变结论方向）。
