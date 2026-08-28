# Task 7：selected MA-BBOB 训练增广价值检验

- 日期：2026-08-28
- 问题：当前训练使用 BBOB train + selected MA-BBOB（24 个 definition），增广是否真的提升泛化？
- 设计：Model B（仅 BBOB train 行）与 Model BM（BBOB+MA-BBOB，即 Phase 1 参照）同特征（28 维 Behavior）、同 RF、同 sample weighting、同阈值协议、同 BBOB family OOF 折与同一 validation。另做 MA-BBOB definition 分组 5 折 OOF（按 definition 分组，非按行随机；折分配经 SeedSequence 预先固定，记录于 `definition_fold_assignment.json`）。无新 objective 运行。

## 1. B vs BM（函数平衡）

| 模型 | split | AP | balanced acc | macro F1 | gain | 归一化 regret | harmful switch rate |
|---|---|---:|---:|---:|---:|---:|---:|
| B（BBOB only） | bbob train OOF | 0.2834 | 0.5493 | 0.5333 | 0.6754 | 0.4628 | 0.0947 |
| **BM（BBOB+MA）** | bbob train OOF | 0.2642 | 0.5683 | 0.5426 | **0.8467** | **0.4023** | — |
| B（BBOB only） | bbob validation | 0.3658 | 0.4854 | 0.4837 | 1.1976 | 0.3186 | 0.0236 |
| **BM（BBOB+MA）** | bbob validation | — | — | — | **1.4481** | **0.2561** | — |

## 2. MA-BBOB definition 分组 OOF（模型在 MA 域内的表现，分数级）

5 折 × 24 definitions：mean improve AP = **0.6043**，mean gain Spearman = **0.6834**（逐折明细见 `mabbob_definition_oof.csv`；fold 3 最弱 AP 0.509 / ρ 0.337）。

## 3. 结论

1. **MA-BBOB 增广有效**：validation gain +0.25（1.448 vs 1.198）、归一化 regret −0.06、train OOF gain +0.17。B-only 在 AP 上略高（0.283 vs 0.264）但策略收益全面更低——排序指标与策略收益的分离再次提示单点 AP 不是主指标。
2. 增广的收益与泄漏防护相互独立：Task 0 已验证无 BBOB held-out 函数身份经 MA 组件泄漏；本消融的差异来自训练分布增广本身。
3. MA-BBOB 内部 definition 分组 OOF 表现良好（AP 0.604），说明模型在组合景观域内同样可学习，24 个 selected definitions 的内部留出诊断不需要后验挑选。
4. 结论：**保留 MA-BBOB 增广作为正式训练协议的一部分**。

产物：`analysis_v2/task7/`（augmentation_summary.parquet、mabbob_definition_oof.csv、definition_fold_assignment.json）、`results/analysis_v2/task7/`（Model B OOF 与 validation run 明细）。
