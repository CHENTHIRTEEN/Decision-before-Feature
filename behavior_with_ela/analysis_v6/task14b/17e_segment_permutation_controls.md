# 17e · Within-Route Permutation Controls（Task 14B-E，N=100）

- 日期：2026-08-29。协议：P1 在 (route, sourceFE) 层内置换 B_global（检验 WG 增量 null）；P2 仅置换 B_segment、保持 B_global 不动（直接建立 segment 增量 null）。RF 正式 carrier；null 统计量与观测统计量同形（P2 用 fb(W0−WGS_perm) 对 fb(W0−WGS_obs)）。经验 p = (1+#{Δ_perm≥Δ_obs})/(1+100)。产物：`segment_permutation_100.parquet`、`global_permutation_100.parquet`、`within_permutation_summary.parquet`。

## 1. 结果

| control | suite | Δ observed | null mean | null std | q95 | q97.5 | **empirical p** |
|---|---|---:|---:|---:|---:|---:|---:|
| P2 segment | bbob | −0.0438 | −0.0445 | 0.0018 | −0.0413 | −0.0409 | **0.337** |
| P2 segment | ma | −0.0337 | −0.0368 | 0.0040 | −0.0301 | −0.0290 | 0.228 |
| P1 global | bbob | −0.0437 | −0.0448 | 0.0020 | −0.0414 | −0.0409 | **0.267** |
| P1 global | ma | −0.0326 | −0.0380 | 0.0043 | −0.0313 | −0.0305 | 0.129 |

## 2. 判读

- 观测值**落在 null 分布的中心**（≈null mean），远未超过 q95——与 Task 13 natural 域"观测超全部 100 个 null（p=0.0099）"形成鲜明对照；
- 换挡后域内 Behavior（global 或 segment）的 within-route 信号**不存在**，置换检验与 OOF/LOSO 的负增量结论互证；
- 实现声明：本轮曾出现临时实现的 W0 泄漏与 float32 溢出两个问题，均已修复并全量重跑（denormal 碎片快照 + float32 矩阵转换，见 17a §3）。
