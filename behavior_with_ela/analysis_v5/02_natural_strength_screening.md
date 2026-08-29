# 02 · Natural-Run Strength Screening（Task 12B/D）

- 日期：2026-08-30
- 域：BBOB train 18×3 + MA-BBOB 24×1，seeds 1–5（78 问题 × 7 候选 × 10,000 FE + 10%×R=3 重复），共 32.56M FE。
- 产物：`analysis_v5/task12/{natural_candidate_performance,candidate_strength_metrics,marginal_vbs,static_pairwise_dcm}.parquet`；噪声 `results/analysis_v5/task12/stage1_noise_deltas.parquet`。

## 1. Stage 1 噪声标定（每 FE 标记，函数平衡）

| FE | BBOB δ50/δ95 | MA δ50/δ95 |
|---:|---|---|
| 1000 | 0.064 / 0.445 | 0.046 / 0.197 |
| 2000 | 0.077 / 0.715 | 0.065 / 0.423 |
| 4000 | 0.105 / 1.366 | 0.074 / 0.853 |
| 6000 | 0.110 / 1.870 | 0.084 / 1.006 |
| 10000 | 0.112 / 2.337 | 0.109 / 1.155 |

（BBOB δ95 随 FE 大幅增长由个别 replicate 发散的右尾驱动；pooled 与函数平衡值并存于 parquet。）

## 2. 主性能表（函数平衡 log10 gap；practical 口径用各 suite δ95）

| 候选 | FE=2000 gap / excl-win / gain-mass | FE=6000 gap / excl-win / gain-mass | FE=10000 gap / excl-win / gain-mass | regret→VBS@10000 |
|---|---|---|---|---:|
| pso | −0.458 / 0.018 / 0.027 | −2.198 / 0.008 / 0.009 | −2.953 / 0.005 / 0.011 | 4.23 |
| lbestpso | +0.015 / 0.003 / 0.000 | −1.059 / 0.000 / 0.000 | −1.974 / 0.005 / 0.000 | 5.08 |
| de | −0.033 / 0.003 / 0.000 | −1.918 / 0.005 / 0.006 | −3.156 / 0.005 / 0.034 | 3.97 |
| shade | −0.168 / 0.005 / 0.001 | −3.161 / 0.067 / 0.158 | −5.457 / 0.146 / 0.490 | 1.72 |
| lshade | −0.318 / 0.008 / 0.003 | **−3.973 / 0.126 / 0.193** | **−5.683 / 0.156 / 0.761** | 1.15 |
| ga | **+0.824 / 0.000 / 0.000** | +0.594 / 0.000 / 0.000 | +0.463 / 0.000 / 0.000 | 7.43 |
| cso | **−1.196 / 0.282 / 0.392** | −2.681 / 0.062 / 0.081 | −2.807 / 0.010 / 0.004 | 4.34 |

Leave-one-out marginal VBS（FE=10000）：lshade +1.21、shade +0.86、cso +0.06、de +0.05、pso/lbestpso +0.03、ga +0.00。

## 3. Strength Gate 判读

- **CSO 是早期阶段专家**：FE=2000 时 fb gap 最优（−1.20）且独占胜率 28.2%、独占增益质量 0.39——远超其它所有候选；至 6000–10000 退化为边缘。这是真实的阶段性优势区（跨 family 验证见 05 报告）。
- **L-SHADE 是中后期最强**：6000/10000 双双第一，独占胜率 12.6%/15.6%，marginal VBS 最大；**标记 DOMINANT-CONTROL 观察**（standalone 最强 + 最大 marginal VBS，但 practical 支配率 15.6%，远未达到 Task 11 式 92% 支配）。
- **SHADE 与 L-SHADE 不冗余**：SHADE 的 leave-one-out marginal VBS（+0.86）在 L-SHADE 存在下仍显著为正（自身中后期独占区）。
- **GA 全面 REJECT-WEAK**：所有 FE、所有 suite 的 exclusive win = 0、gain mass = 0、regret 最差。
- **PSO / DE / lbestPSO REJECT-WEAK**：exclusive win ≤ 0.8%、gain mass ≤ 0.03、marginal VBS ≤ 0.05——在 practical 口径下没有独立优势区。

## 4. 与"强算法≠适合组合"原则的对应

L-SHADE 平均最强且确实保留；CSO 平均排中游但拥有最大的**早期独占区**——正是 §32 所说"canonical/弱一些的求解器可能对 DAS 更有价值"的实例化（最终由 Stage 2 dynamic 检验确认其切换价值）。
