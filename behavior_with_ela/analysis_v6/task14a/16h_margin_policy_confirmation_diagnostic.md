# 16h · 预先固定 Margin Policy 在 Post-Handoff 状态上的确认诊断（Task 14A §24–25）

- 日期：2026-08-29。载体：Task 13 正式 RF pipeline（同参数/同特征契约）在**全部 1890 development states** 上拟合一次（§25 允许的预先固定跨 group 模型），直接应用于 3780 个 post-handoff 状态（B_global 28 列 + current one-hot + FE_ratio）；阈值尺度 = Task 13.1 pooled 部署常数（对新状态而言即"仅训练域"合法常数）。**确认诊断：不得也不曾用于重选 κ。**
- 产物：`margin_policy_confirmation_diagnostic.parquet`。

## 1. 结果（one-step realized，vs Continue）

| κ | suite | switch rate | fb loss | gain vs Continue | harmful rate |
|---:|---|---:|---:|---:|---:|
| 0.0（raw） | pooled | 0.551 | +0.029 | **−0.029** | 0.166 |
| 0.0 | bbob | 0.560 | +0.018 | −0.018 | 0.163 |
| 0.0 | ma | 0.528 | +0.038 | −0.038 | 0.173 |
| 0.5 | pooled | 0.333 | +0.009 | −0.009 | 0.097 |
| 0.5 | bbob | 0.336 | −0.003 | **+0.003** | 0.092 |
| 0.5 | ma | 0.324 | +0.019 | −0.019 | 0.112 |
| 1.0 | pooled | 0.274 | +0.003 | −0.003 | 0.074 |
| 1.0 | bbob | 0.281 | +0.006 | +0.006 | 0.070 |
| 1.0 | ma | 0.255 | +0.011 | −0.011 | 0.082 |

## 2. 判读

1. **natural 域调校的预先固定策略不向 post-handoff 域迁移出正增益**：pooled 下三个 κ 的 one-step gain 均为负或 ≈0（−0.029/−0.009/−0.003）；只有 bbob 在 κ=0.5/1.0 下勉强为正（+0.003/+0.006）；
2. 与 16d 一致：成熟换挡后 **continue 是强默认**（switch-required 0.21–0.23、P1 退化为 Always Continue），natural 域学到的切换倾向在新域整体偏激进；
3. 风险面仍随 κ 单调改善（harmful 0.166→0.097→0.074）——margin 的**风险控制**语义可迁移，**性能增益**语义不可迁移；
4. 按工作单要求：本诊断不改变 κ=0.5/1.0 两个 pre-fixed candidate 的地位；14B 若在 post-handoff 域部署策略，必须以 post-handoff 开发数据重新拟合策略（含 current-preserving margin），而不是沿用 natural 域模型。

## 3. 边界

- 该迁移缺口**不是** 14B GO/NO-GO 的判据（14B 的对象是 B_global vs B_segment 的增量，基线在各自域内重建）；
- 未做任何 per-suite/per-route 阈值适配（那属于 threshold tuning，本轮禁止）。
