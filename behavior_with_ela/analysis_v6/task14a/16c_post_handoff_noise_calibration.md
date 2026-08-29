# 16c · Post-Handoff Practical Noise 标定（Task 14A 噪声重标定）

- 日期：2026-08-29。语义：post-handoff state-action 对的 10% × R=3（outcome-blind SeedSequence 抽样，REPETITION_STREAM=2026090222），deviation = |loss − state-action 中位|，cv_group 函数平衡 Q95。**不复用 natural-state Task 12 δ**。产物：`post_handoff_noise_deltas.parquet`。

## 1. Pooled（solver-cell 合并口径，与 Task 13.1 部署尺度同语义）

| solver | δ50 | δ95 | 重复 cell 数 |
|---|---:|---:|---:|
| shade | 0.0 | 0.0872 | 394 |
| lshade | 0.0 | 0.0950 | 396 |
| cso | 0.0 | **0.0481** | 352 |

**cso 的 post-handoff 短程噪声（0.048）明显低于其 natural 值（0.086/0.125）**——成熟 cso 状态的分支结果更稳定；shade/lshade 与 natural 量级相当。

## 2. 按 route 的 δ95（诊断）

| route | cso | lshade | shade |
|---|---:|---:|---:|
| shade→lshade | 0.050 | 0.155 | 0.144 |
| shade→cso | 0.117 | 0.061 | 0.051 |
| lshade→shade | 0.060 | 0.101 | 0.092 |
| lshade→cso | 0.110 | 0.132 | 0.096 |
| cso→shade | 0.025 | 0.032 | 0.043 |
| cso→lshade | 0.009 | 0.057 | 0.061 |

route 间差异明显（cso→lshade 全列 ≤0.06 vs shade→lshade 高达 0.14–0.16）——post-handoff 噪声是 route 条件化的；主判定使用 pooled solver 口径，route 口径留作 16d/16g 的解释性参照。

## 3. 使用方式

- practical $A_{ND}$ 主口径 = solver δ95 的 max 组合；quadrature/sum 为敏感性；
- 若直接配对重复足够可估 $\delta_{ij,95}^{paired}$——本轮重复按 state-action 抽样、跨动作配对数不足（与 Task 12.1 同样的 10%×R=3 设计），故按预注册 fallback 使用 solver 级保守组合。
