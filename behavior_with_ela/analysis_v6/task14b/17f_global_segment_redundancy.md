# 17f · Global↔Segment 表征冗余诊断（Task 14B Representation Redundancy）

- 日期：2026-08-29。定义：leave-cv_group-out 下 Ridge(α=1) 跨表征回归 OOF R²：$R^2(B^{segment}\leftarrow B^{global})$ 与 $R^2(B^{global}\leftarrow B^{segment})$（28 输出合并 R²）。产物：`global_segment_redundancy_oof.parquet`。

## 1. 结果

| 方向 | OOF R² mean | 范围 |
|---|---:|---|
| bs ← bg | **−8.7×10¹⁰**（折间 −9.1×10¹¹ ～ −0.81） | 剧烈为负 |
| bg ← bs | −2.2×10⁷（折间 −1.6×10⁸ ～ −6.5×10⁵） | 剧烈为负 |

## 2. 如实判读：该诊断在本域**不可解释**

OOF R² 大幅为负不是"两个表征互不包含"的合法证据，而是**跨 group 的特征分布漂移 × 长尾特征**（|v| 至 1e6 裁剪后仍有组间量级差）在 Ridge 外推下的灾难性失败——Ridge 对未见 group 的均值/尺度外推完全失效。诊断按工作单 §27 的目的（cross-group reconstructability）**退化为不可用**，本轮如实标注为 UNINFORMATIVE，不做任何"谁包含谁"的推断。

## 3. 可用的替代证据

- 逐特征相关：bg↔bs 平均 0.446，5 列恒等（慢变特征），27 列中等以下相关——两类表征**携带部分共享、部分独立的信息**，但（见 17b/17c/17d）**这些信息都不能转化为 post-handoff 域的 next-action 选择增益**；
- 表征冗余问题对 verdict 无影响：Verdict A 已经是 A3（Behavior 整体无增量），冗余问题不再进入决策链。
