# 17g · Post-Handoff 域内 Margin Policy（Task 14B，固定 κ=0/0.5/1.0）

- 日期：2026-08-29。margin 来自 **post-handoff 域内重训的 MGS OOF 预测**（非 natural 模型）；pair scale 采用 **fold-local** 校准（每 held-out cv_group 仅用训练 groups 的 post-handoff 重复，Task 13.1-H 语义）；κ∈{0, 0.5, 1.0} 预注册，不选优。产物：`post_handoff_fold_local_noise_scale.parquet`、`post_handoff_margin_policy_rows.parquet`、`post_handoff_margin_policy_summary.parquet`。

## 1. fold-local δ_{a,95}^{(-g)}

post-handoff 重复的 per-solver 尺度：cso 0.048 / shade 0.087 / lshade 0.095（pooled 参照）；fold 间波动与 Task 13.1-H 相同量级（排除任一 group 变化 ≤~5%）。

## 2. MGS margin 策略（max 口径）

| κ | suite | fb loss | gain vs Continue | switch rate | harmful rate |
|---:|---|---:|---:|---:|---:|
| 0.0（raw） | bbob | −1.9413 | −0.0278 | 0.637 | 0.170 |
| 0.0（raw） | ma | −5.0239 | −0.0083 | 0.603 | 0.157 |
| 0.5 | bbob | −1.9495 | −0.0197 | 0.363 | **0.109** |
| 0.5 | ma | −5.0247 | −0.0075 | 0.367 | **0.119** |
| 1.0 | bbob | −1.9653 | −0.0039 | 0.199 | **0.046** |
| 1.0 | ma | −5.0271 | −0.0051 | 0.207 | **0.064** |

sum 口径：κ=0.5 harmful 0.059/0.076；κ=1.0：bbob gain **+0.0021**（≈打平）、harmful 0.032/0.032。

## 3. margin 排序意义

Spearman(margin, realized switch gain)：pooled 0.127（bbob 0.122 / ma 0.142）——弱单调；margin 三分位的切换增益：底部 −0.126 / 中部 −0.026 / 顶部 **−0.044**——**顶部三分位未转正**，margin 排序意义在 post-handoff 域 MGS 上**弱且不足以支撑正增益策略**（与 natural 域 Task 13.1 的可用排序形成对照）。

## 4. 判读

- κ 阈值在 post-handoff 域内重训模型上仍实现**单调的风险下降**（harmful 0.17→0.05–0.06）；
- 但**平均性能不再保留**（gain 全负，至多打平）→ Verdict C = **C2 TRADEOFF**（风险下降、性能受损），非 C1、亦非 C3（margin 语义未完全失效，risk 方向仍可用）；
- κ=0.5/1.0 两个 pre-fixed candidate 维持原标签，本轮不选择。
