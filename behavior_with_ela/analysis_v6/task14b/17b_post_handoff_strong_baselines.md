# 17b · 强基线阶梯（Task 14B 强部署基线）

- 日期：2026-08-29。同一 3780 post-handoff states；RF 正式 carrier；grouped leave-cv_group-out；fb realized loss。产物：`task14b_policy_performance.parquet`、`global_vs_segment_pairwise_bootstrap.parquet`、`post_handoff_lookup_oof`（lookup 逐 state 值在 OOF 表）。

## 1. 阶梯（fb log10 loss，RF）

| 基线 | BBOB | MA | pooled |
|---|---:|---:|---:|
| B0 Always Continue | −1.9978 | −5.0540 | −3.665 |
| B1 Empirical route+sourceFE lookup | **−1.8980** | **−5.0007** | — |
| B2 RF-M0 [route, sourceFE, segmentAge] | −1.9835 | −5.0431 | −3.6524 |
| B3 MG | −1.9543 | −5.0190 | −3.6260 |
| B4 MS | −1.9421 | −5.0058 | −3.6132 |
| B5 MGS | −1.9413 | −5.0239 | −3.6228 |

## 2. 配对比较（5000 draws，95% CI）

| 比较 | BBOB | MA |
|---|---:|---:|
| M0 vs Continue | +0.0143 [−0.008, +0.046] | +0.0109 [+0.003, +0.019] |
| **MGS vs Lookup** | **−0.0433 [−0.073, −0.021]**（显著劣于 lookup） | −0.0232 [−0.058, +0.021] |
| MGS vs M0 | −0.0422 [−0.067, −0.021] | −0.0192 [−0.058, +0.035] |
| MG vs M0 | −0.0292 [−0.047, −0.011] | −0.0241 [−0.058, +0.023] |
| MS vs M0 | −0.0414 [−0.071, −0.013] | −0.0373 [−0.081, +0.019] |
| **MGS vs MG（segment 增量）** | −0.0129 [−0.035, +0.002] | +0.0049 [−0.009, +0.018] |

## 3. 判读

1. **全部 Behavior 模型（MG/MS/MGS/BG-only/BS-only）劣于 M0 与 lookup**：bbob 上为显著负增量（CI 上界 <0），ma 上点估计亦为负；
2. RF-M0 ≈ empirical lookup（bbob 差 −0.001），M0 阶梯成立；
3. **segment 增量 MGS−MG ≈ 0**（bbob −0.013 CI 含 0、ma +0.005 CI 含 0）；
4. 机制：Behavior 模型把 switch rate 抬到 0.60–0.64（基率 0.21–0.23），而成熟 post-handoff 状态上 continue 是强默认（16g/16h），过度切换直接损失平均性能——与 Task 14A §16h 的迁移失败诊断一致。

→ Verdict A（Post-Handoff Behavior Increment）= **A3 NO-GO**；Verdict B 详见解 17c/17d。
