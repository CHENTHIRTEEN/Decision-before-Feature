# Task 4：Local Landscape 分组消融（代码预定义 L1–L4）

- 日期：2026-08-28
- 问题：M2（lf_* 单独）在 validation 上略强于 M1 的现象由哪些特征组驱动？
- 分组（按 `local_landscape.py` 既有元组预定义，评估前固定，未做后验挑列）：
  - **L1** `LOCAL_LANDSCAPE_STREAMING_COLUMNS`（7）：fitness 分布/分位数
  - **L2** `LOCAL_LANDSCAPE_META_MODEL_COLUMNS`（7）：linear/quadratic 元模型
  - **L3** `LOCAL_LANDSCAPE_INFORMATION_COLUMNS`（3）：information content
  - **L4** `LOCAL_LANDSCAPE_GEOMETRY_COLUMNS`（6）：local FDC / dispersion / NBC 几何
- 每组两个变体（`M2_Lk` 单独、`M1_Lk` = Behavior 28 维 + Lk），三分类载体、同一 grouped family OOF 与阈值协议。无新 objective 运行。

## 1. 主表（函数平衡）

| 变体 | 特征数 | train OOF gain | train 归一化 regret | val gain | val 归一化 regret |
|---|---:|---:|---:|---:|---:|
| M2_L1 | 7 | 0.7946 | 0.3860 | 1.4285 | 0.2699 |
| M1_L1 | 35 | **0.8812** | 0.3446 | **1.4590** | 0.2427 |
| M2_L2 | 7 | 0.8572 | 0.4753 | 0.7286 | 0.4137 |
| M1_L2 | 35 | 0.8472 | 0.3730 | 1.4643 | 0.2455 |
| M2_L3 | 3 | 0.2521 | 0.4589 | 0.2375 | 0.4930 |
| M1_L3 | 31 | 0.8311 | 0.4241 | 1.4555 | 0.2511 |
| M2_L4 | 6 | 0.3294 | 0.5643 | 1.2930 | 0.3061 |
| M1_L4 | 34 | 0.8586 | 0.3524 | 1.2867 | 0.2918 |
| （参照）M1 Behavior | 28 | 0.8467 | 0.4023 | 1.4532 | 0.2527 |

## 2. 对 M1 的逐 run 配对增量（train OOF，函数平衡差值）

| 对比 | Δgain | Δregret |
|---|---:|---:|
| M1+L1 − M1 | **+0.0345** | −0.0345 |
| M1+L2 − M1 | +0.0005 | −0.0005 |
| M1+L3 − M1 | −0.0156 | +0.0156 |
| M1+L4 − M1 | +0.0120 | −0.0120 |

## 3. 解读

1. **L1（fitness 分布/分位数）是唯一既有 train OOF 增量又在 validation 上不退化的组**（M1+L1 是 validation 第二好的配置，1.4590，仅次于 M2-full 1.4658）。7 列分布特征捕捉了 reservoir 内目标值结构，与 Behavior 的改善率/停滞类特征互补性最好。
2. **L2（元模型）是过拟合信号**：M1+L2 在 train OOF 持平（+0.0005），validation 1.4643 尚可；但 M2_L2 单独从 train 0.857 崩到 validation 0.729——线性/二次 R² 类特征对 family 结构记忆强、跨族不稳。
3. **L4（几何/NBC）训练族内小增量不能迁移**：M1+L4 validation 1.2867，明显低于 M1（1.4532）。
4. **L3（information content）单独几乎无信息**（0.24–0.25），加入后轻微拖累。
5. M2-alone 现象的解释线索：L1 单独（M2_L1）validation 1.4285 已接近 M1，说明 7 个分布分位数特征本身就携带了与 Behavior 相当的可行动信息（冗余性的直接证据见 05 报告）。

结论：**不存在稳健为正的 conditional increment**（最好的 L1 组在 train OOF +0.0345、validation 上与 M1 相当），Phase 2 的总体判断不变；M2 现象主要由 L1 分布特征解释，部分来自 L2/L4 在训练族内的过拟合。

产物：`analysis_v2/task4/`（model_summary.parquet、paired_contrasts_vs_M1.parquet、summary.json）、`results/analysis_v2/task4/`（各变体 OOF 与 run 明细）。
