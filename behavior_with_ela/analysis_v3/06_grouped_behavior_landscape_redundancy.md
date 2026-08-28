# 06 · Task 9F：grouped 口径下 Behavior–Local Landscape 冗余的重新检验

- 日期：2026-08-29
- 问题：Task 5 的 `lf→bf R² 中位 0.475` 基于 80/20 行级随机划分，而同一 state 的三个 action 行携带完全相同的特征向量，同一 state 的重复行可能同时落入训练与测试，造成乐观偏置。grouped 口径下信息重叠是否仍成立？
- 方法：
  1. **state 去重**：唯一键 (problem_id, prefix_algorithm, seed, FE, decision_opportunity_index)，每 state 一行（25,469 BBOB + MA states，取 continue 行）。
  2. **两个方向**：bf→lf（28→23）、lf→bf（23→28），多输出 `RandomForestRegressor(100, depth 12)`，折内中位数填充。
  3. **两个分组方案**：A. leave-BBOB-family-out（MA 行按组件泄漏防护并入训练，只在 held-out family 的 BBOB states 上评价）；B. 按 problem_id 分组 5 折（54 个 BBOB problem + 24 个 MA definition 各自成组，折分配经 SeedSequence 预先固定，stream 2026082919）。
  4. 不做随机行划分；不以 full-train 模型在训练行上做置换重要性（旧 Task 5 的 block permutation importance 因此不再复述，见 `summary.json` 的说明字段）。
- 产物：`analysis_v3/task6/grouped_cross_predictability.csv`（逐目标 R²）、`summary.json`。

## 1. 主表（pooled OOF R²，逐目标汇总）

| 方案：方向 | 目标 R² 中位 | R²>0.5 目标占比 | 负 R² 目标占比 |
|---|---:|---:|---:|
| leave-family-out：bf→lf | **0.186** | **0.0%** | 26.1% |
| leave-family-out：lf→bf | **0.118** | 10.7% | 25.0% |
| grouped-by-problem：bf→lf | 0.236 | 0.0% | 4.3% |
| grouped-by-problem：lf→bf | **0.391** | 32.1% | 3.6% |

（对照：旧行级划分 lf→bf 中位 0.475、46.4% 目标 R²>0.5——确认为乐观偏置。3 个近似常数目标 `lf_linear_r2_change / lf_quadratic_r2_change / lf_information_sign_change_rate` 的 pooled R² 无定义，已排除在占比之外。）

## 2. 结构解读

1. **跨 BBOB family 时互相可预测性大幅缩水**：lf→bf 中位 R² 从 0.475 降到 0.118，仅 10.7% 的 bf 目标 R²>0.5；bf→lf 则没有任何目标达到 0.5。Task 4/5 依据行级划分得出的"两组特征信息大量重叠"在跨函数泛化意义上**不成立**。
2. **同函数内重叠仍然可观但集中于多样性/散布类特征**：problem 分组下 lf→bf 的头部目标是 `bf_diversity_mean_pairwise`（0.81）、`bf_population_chamfer_distance_w05`（0.77）、`bf_population_wasserstein_rate_w05`（0.74）、`bf_elite_concentration`（0.64）——local landscape 种群采出的 y 分布本质上是生成它的搜索行为的投影，但这主要是**函数内、轨迹内**的耦合；对 improvement/stagnation 类行为特征几乎无预测力（`bf_improvement_rate_w02` R² = -0.07）。
3. 结论措辞：**information overlap 在 grouped OOF 后仅部分成立且限定于函数内**；它不能作为"Behavior 可由 landscape descriptors 替代"的跨函数机制证据。

## 3. A1 vs A3 的 OOF action 分歧（更直接的 actionable-overlap 证据，复述）

- state 级 top-1 动作分歧率 **1.60%**（25,469 states 中 406 个）；分歧处强制采用 A3 的真实增益差 mean **-2.58**、median -2.30（68 : 338 劣于 A1）；run 级 selected action 一致率 98.6%。
- 该证据不依赖行级划分，保持不变：**在动作层面，加入 Local Landscape 没有产生可利用的互补信息**。此结论与冗余重检的缩水并不矛盾——A3 的失败方式是"复述 A1 已有信息 + 引入族内过拟合"，而不是"提供另一路信息"。

## 4. 对论文表述的影响

- 旧表述"lf→bf R² 中位 0.475，46% 目标可被 lf 以 R²>0.5 预测"必须撤回，改为本报告的 grouped 口径数字，并注明行级划分的 state 重复泄漏问题。
- 保留的机制结论："搜索行为与轨迹派生 landscape descriptors 都编码了求解器–问题交互信息；其共享部分主要是函数内的种群几何投影（多样性/散布类），在动态选择设定下两组特征的可行动信息高度重叠（A1 vs A3 分歧证据），且均不构成跨函数的互补信息。"
