# 07 · Dynamic Noise Calibration 判读（Task 12J 汇总）

- 日期：2026-08-30
- 数据：1,890 状态 × 3 动作，10% × R=3（明细 `results/analysis_v5/task12/`；汇总 `analysis_v5/task12/dynamic_noise_deltas.parquet`）。

## 1. 主判据

- **BBOB：δ1000,95 ≈ 0.0876**（各动作 0.075–0.101）
- **MA：δ1000,95 ≈ 0.0839**（各动作 0.058–0.125）
- pooled 敏感性：≈0.056–0.135，与 suite-specific 一致量级。

## 2. 三个要点

1. **逐 suite/逐算法标定是必要的**：cso 在 MA 上的 δ95（0.125）几乎是 lshade 在 MA 上（0.058）的两倍——统一套用单一 δ 会同时误判两个方向。
2. **与 Task 11 的 CMAES-state 噪声（0.098）同量级但不可沿用**：本轮状态为 shade/lshade/cso 自然轨迹 checkpoint，重复结构不同；实测值在此上下验证了"必须重标定"的要求。
3. **判定规则**：后续 08–11 报告的 practical 关系一律以 |Δ| > suite δ95 为准；Δ 的函数级 bootstrap（2000 次，按 cv_group 重采样）同时报告，避免把 state 行当独立样本。

（本报告为 06 的统计补充；主叙事见 08。）
