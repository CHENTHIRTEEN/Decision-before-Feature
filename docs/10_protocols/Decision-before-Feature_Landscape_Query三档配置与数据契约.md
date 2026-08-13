# Decision-before-Feature 三档 Landscape Query 配置与数据契约

## 1. 构念边界

当前 16 维实现由 LHS、目标值分布、样本距离、中心距离、相关性、线性模型拟合度与梯度范数组成。它不是完整 ELA，也不是 pflacco 标准 feature groups 的子集实现。因此第一篇论文的主研究问题限定为：

> 算法无关搜索行为能否预测调用一项固定低成本 landscape descriptor query 的效用？

论文结论只适用于实际执行并计费的 query。不得外推到完整 ELA、全部 pflacco features、NeurELA、Deep-ELA 或任意 landscape representation。

## 2. 预先定义的三档配置

| `query_id` | 角色 | 样本 | 总 FE 占比 | 特征 | 后端 |
| --- | --- | ---: | ---: | ---: | --- |
| `descriptor_cheap` | 主 query | `lhs_50d` | 5% | 16 | 仓库内固定描述符 |
| `pflacco_standard` | 配置稳健性 | `lhs_50d` | 5% | 37 | pflacco 1.2.2 |
| `pflacco_broad` | 配置稳健性 | `lhs_100d` | 10% | 52 | pflacco 1.2.2 |

`descriptor_cheap` 与 `pflacco_standard` 必须读取同一份 `lhs_50d` 的 (X,y)，不得分别采样。`pflacco_broad` 使用独立 `lhs_100d`。主 query 在查看 validation 结果前已经固定，不能依据结果改选。

standard 包含 PCA、NBC、dispersion、information content 和 ELA distribution。broad 在 standard 基础上加入 ELA level-set 与 sample-derived fitness-distance correlation。不包含完整 quadratic `ela_meta`、cell mapping，也不包含需要额外函数评价的 local、curvature、convexity、Sobol 和 length-scale groups。

## 3. 采样与特征边界

主环境只负责生成 benchmark 问题的 LHS 样本并保存 (X,y)：

```text
results/landscape_queries/samples/{sample_design_id}/{split}/samples.parquet
```

样本行保存 `sample_design_id`、`sampling_protocol`、`sample_seed`、`sample_size`、`FE_query`、`runtime_query_sampling`、`runtime_query_evaluation`、兼容汇总字段 `runtime_sampling_evaluation`、边界、(X) 和 (y)。随机种子由 base seed、stream code、function、instance、dimension 与整数 design code 共同进入 `numpy.random.SeedSequence`；不得使用字符串哈希。

特征结果保存到：

```text
results/landscape_queries/features/{query_id}/{split}/features.parquet
```

每行必须包含 `query_id`、`query_protocol`、`sample_design_id`、固定 `query_feature_columns`、`runtime_query_feature_computation`、兼容别名 `runtime_feature_computation`、`feature_status`、逐 group 状态、非有限值记录与失败信息，并满足：

```text
runtime_sampling_evaluation = runtime_query_sampling + runtime_query_evaluation
runtime_query = runtime_query_sampling + runtime_query_evaluation + runtime_query_feature_computation
additional_function_evaluations = 0
```

单个数学上未定义或非有限的特征保存为 null，并在 `feature_nonfinite` 中记录；不得替换为自定义常数。group 异常将该 group 的列保存为 null，并在 `feature_group_status` 与 `feature_failure` 中记录。

## 4. pflacco 隔离环境

主项目不依赖 pflacco。标准特征只可由 `tools/pflacco_query/` 的 Python 3.11 环境提取，该环境固定 `pflacco==1.2.2` 及其兼容 NumPy、Pandas、SciPy 与 scikit-learn。工具只读 Parquet 样本并写 Parquet 特征；不导入 benchmark，不评价目标函数，不安装 R 包，不在运行时下载依赖，也不回退到自定义公式。

NBC 固定 `dist_tie_breaker="first"`；information content 使用显式整数 `SeedSequence` seed。每个 group 的返回列必须与 pflacco 1.2.2 白名单完全一致，`*.costs_runtime` 只用于组内诊断，不进入 selector feature 列。

## 5. Query-specific Selector、Utility 与 Decision

`lhs_50d` 只生成一份逐共享状态 action-loss 表，供 cheap 与 standard 分别训练 Selector；`lhs_100d` 单独生成 action-loss 表。action-loss 输入包含 `sample_design_id` 与 `FE_query`，不包含 behavior 或 query features。构建 Selector 时才连接 behavior 与指定 query feature 表。

每个 query 独立拟合 Selector、Utility target 和 Decision Model：

```text
results/selection_reference/{query_id}/
results/utility_labels/{query_id}/
results/decision/{query_id}/
```

Selector artifact 保存 `query_id`、`query_protocol`、`sample_design_id` 和实际 `query_feature_columns`。Decision Model 输入仍严格限定为算法无关 `bf_*` behavior；query id、query features、function、dimension、algorithm 和优化器内部状态都只作 metadata 或分层报告。项目不训练动态 query-type selector。

模型比较、特征组消融、阈值分析、baseline、成本—性能与外部评价命令均要求显式 `--query-id`。命令据此推导 query-specific 默认目录，并核对 artifact 中的 `query_id`、`query_protocol` 与 `sample_design_id`；不同 query 的 dataset、summary 或 prediction 表不能交叉读取。

活动标签字段为 `FE_query`、`runtime_query`、`p_query`、`u_query_lamT_*` 与 `need_query_lamT_*`。旧 `results/ela/`、`FE_analysis`、`p_ela`、`u_ela_*`、`need_ela_*` 和缺少新协议字段的 artifact 均为撤回结果，活动读取器必须明确拒绝。

## 6. 数据质量与失败处理

BBOB train/validation 不允许 group-level extraction failure。单个 null 只使用 BBOB-train median imputation；任何在 BBOB train 上整列缺失的 query feature 都阻止 Selector 拟合。

外部 benchmark 允许保留 group-level failure，并由已拟合 Selector 的 BBOB-train median 执行 fallback。所有受影响行必须单独报告 `feature_status` 与 `feature_failure`；只要存在 group-level failure，就不能给出该 query 的无条件外部泛化结论。

`query-consistency` 是非 pytest 的可执行一致性入口。它检查三档预算、feature whitelist、cheap/standard 的共享样本键、零额外函数评价、BBOB group failure、整列缺失和 action-loss 预算隔离。`utility-labels-validate` 逐行重算：

```text
p_query = selected_action_loss
performance_gain_raw = potential_gain_raw - selector_regret_raw
u_query_lamT_* = performance_gain_norm - lambda_T * time_cost_norm
need_query_lamT_* = (u_query_lamT_* > 0)
```

## 7. 报告与结论口径

分别报告 (U_{cheap})、(U_{standard}) 与 (U_{broad})，并对每档报告 Never Query、Always Query、Random Query、Traditional AAS、SBS 和 VBS。每档至少包含 query FE、运行时间、失败率、selector regret、action-loss 回归性能、Utility 分布、调用率、效用捕获、最终优化性能，以及 function-family 层面的配对效应量与区间。

若三档结论一致，只能表述为“结论在三个预定义 query 配置上具有稳健性”。若结论不一致，必须报告 representation 与成本依赖性，不能隐藏结果或重新定义 query。NeurELA 和 Deep-ELA 本轮只用于说明 landscape representation 的异质性。

## 8. 固定执行顺序

```text
trajectory 重生成
→ behavior
→ lhs_50d / lhs_100d samples
→ descriptor_cheap / pflacco_standard / pflacco_broad features
→ lhs_50d 与 lhs_100d 两档 action losses
→ 三个 Selectors
→ 三套 Utility labels
→ 三套 Decision Models 与 baselines
```

在 query 配置、采样边界、隔离 pflacco 提取与一致性命令通过前，不启动 72 个 trajectory shard 的全量重生成。截至 2026-08-11，这些前置检查已在真实 BBOB 10D 关键路径通过；本轮仍未启动正式 shards。
