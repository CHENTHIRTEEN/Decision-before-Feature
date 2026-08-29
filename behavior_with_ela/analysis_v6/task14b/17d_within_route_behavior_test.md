# 17d · Within-Route Genuine State Test（Task 14B-B）

- 日期：2026-08-29。固定 (problem, route, sourceFE)（126 组 × 5 seeds），LOSO 4 train / 1 test：W0 组内均值基线；WG = B_global；WS = B_segment；WGS = 并集。RF 正式 carrier。产物：`within_route_loso_predictions.parquet`、`within_route_performance.parquet`。

## 1. 主结果

| suite | L_W0 | L_WG | L_WS | L_WGS | Δ_global (W0−WG) | Δ_segment-only (W0−WS) | **Δ_segment (WG−WGS)** |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBOB | −2.0583 | −2.0145 | −2.0146 | −2.0145 | **−0.0437** | −0.0437 | **−0.0001** |
| MA | −5.1195 | −5.0868 | −5.0864 | −5.0857 | **−0.0326** | −0.0331 | −0.0011 |

**两类 Behavior 在 within-route 下均为负增量**：换挡后同 route 同 phase 的不同 state 之间，B_global 与 B_segment 都不能带来优于组内均值基线的动作选择。

## 2. 与 natural 域（Task 13/13.1）的对照

| 量 | natural 域 | post-handoff 域 |
|---|---:|---:|
| Δ_within,global | +0.019 / +0.016（p=0.0099） | **−0.044 / −0.033** |
| Δ_within,segment | — | −0.000 / −0.001 |

natural 域的 within-problem 真实信号**在换挡后域完全消失且反号**。

## 3. 结论

工作单的核心问题——"recent segment history 是否含有 global history 未表达的 state information"——答案为 **NO**（Δ_segment ≈ 0，且两侧都无正增量）。
