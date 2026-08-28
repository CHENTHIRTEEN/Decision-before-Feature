# Task 5：Behavior 与 Local Landscape 冗余与分歧分析

- 日期：2026-08-28
- 目标：判断 Local Landscape 是否包含 Behavior 条件下的额外可行动信息；量化 A1（Behavior action-loss RF）与 A3（Behavior+Local action-loss RF）的分歧及其真实后果。
- 全部基于已有 parquet 与已训练模型，无新 objective 运行。

## 1. 特征层相关性（bf 28 维 × lf 23 维，Spearman，20k 子样本）

| 统计量 | 数值 |
|---|---:|
| 平均 |相关| | 0.2725 |
| 最大 |相关| | 0.9251 |
| |相关|>0.5 的配对占比 | 15.4% |
| |相关|>0.7 的配对占比 | 4.7% |
| 有 |相关|>0.7 lf 伙伴的 bf 特征数 | 8 / 28 |
| 有 |相关|>0.7 bf 伙伴的 lf 特征数 | 9 / 23 |

原始特征层并非全局高度相关（均值仅 0.27），但存在一个中等规模的强相关子集。完整矩阵：`spearman_bf_vs_lf.csv`。

## 2. 跨组可预测性（多输出 RF，80/20 划分，中位数填充）

| 方向 | 目标特征 R² 中位数 | R²>0.5 的目标占比 |
|---|---:|---:|
| bf → lf | 0.3762 | 26.1% |
| **lf → bf** | **0.4754** | **46.4%** |

lf 特征对 bf 特征的预测能力强于反向——local landscape 采样自当前种群，携带了生成它的搜索行为的投影。这是两组信息大量重叠的机制性证据。明细：`cross_group_predictability.csv`。

## 3. A1 vs A3 分歧的真实后果（train OOF，25,469 states / 1,620 runs）

| 层级 | 统计 | 数值 |
|---|---|---:|
| state | top-1 动作分歧率 | 1.60%（406 states） |
| state | 分歧处 A3 更优 / A1 更优 | 68 / **338** |
| state | 强制采用 A3 的真实增益差 | mean **−2.581**，median −2.299 |
| run | selected action 一致率 | 98.64% |
| run | selected FE 一致率 | 99.44% |
| run | 强制采用 A3 的平均增益差 | −0.0633 |

结论：**两组模型的可行动信息高度重叠**——分歧位置上 local landscape 增广反而系统性更差（1:5 的优劣比），强制采用 A3 排序无任何收益。lf→bf 的可预测性 + 分歧处 A1 占优 + A3 置换重要性中 lf 块占比 0.698 却几乎不改变决策，共同支持"两组特征编码的信息大量重叠、在当前单次切换设定下不构成互补"的判断。

## 4. 与 M2 现象的关系

M2（lf 单独）≈ M1 的现象与上述机制一致：lf 特征能部分重建 bf 特征（lf→bf R² 中位 0.475），因此单独使用时信息量接近，合并时不产生新增可行动信息。但注意 Task 4 的发现：L1 分布特征是其中最互补的子集（M1+L1 增量 +0.0345），不能排除在更细粒度（如 L1 与 Behavior 的特定子集组合）下存在小的互补空间——这留作后续发展性分析，不改变当前裁决。

产物：`analysis_v2/task5/`（summary.json、spearman_bf_vs_lf.csv、cross_group_predictability.csv）。
