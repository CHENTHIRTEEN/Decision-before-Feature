# Decision-before-Feature 三档 Landscape Query 配置与数据契约

## 1. 构念边界

当前 14 维实现由 LHS、目标值分布、样本距离、中心距离、相关性、线性模型拟合度与梯度范数组成。统一 median/IQR preprocessing 后恒为 0/1 的 `descriptor_y_median`、`descriptor_y_iqr` 已删除；query ID、样本、FE 和 action losses 不变。它不是完整 ELA，也不是 pflacco 标准 feature groups 的子集实现。因此第一篇论文的主研究问题限定为：

> 算法无关搜索行为能否预测调用一项固定低成本 landscape descriptor query 的效用？

论文结论只适用于实际执行并计费的 query。不得外推到完整 ELA、全部 pflacco features、NeurELA、Deep-ELA 或任意 landscape representation。

## 2. 预先定义的三档配置

| `query_id` | 角色 | 样本 | 总 FE 占比 | 特征 | 后端 |
| --- | --- | ---: | ---: | ---: | --- |
| `descriptor_cheap_invariant` | 主 query | `lhs_50d` | 5% | 14 | 仓库内固定描述符 |
| `pflacco_standard_invariant` | 配置稳健性 | `lhs_50d` | 5% | 37 | pflacco 1.2.2 |
| `pflacco_broad_invariant` | 配置稳健性 | `lhs_100d` | 10% | 52 | pflacco 1.2.2 |

`descriptor_cheap_invariant` 与 `pflacco_standard_invariant` 必须读取同一份 `lhs_50d` 的 (X,y)，不得分别采样。`pflacco_broad_invariant` 使用独立 `lhs_100d`。BBOB-validation 已在旧流程中被查看，不能再声称主 query 是相对该集合事前固定；从本次协议冻结起，不得依据 BBOB-validation、CEC2017 或后续外部结果改选 query。

每个静态 problem/sample design 只有一个由 problem key 冻结的 LHS realization；optimizer seed、decision state 与 action 共用该 realization。因此主结果是固定 query realization 下的条件 estimand，不包含 query-sampling randomness。没有多个独立 LHS replicates 时不得宣称对 sampling randomness 稳健；若在查看正式 policy outcomes 前增加 replicates，只能作为预设 sampling-robustness sensitivity，并把 replicate 纳入配对键和层级统计。

cheap--standard 共享相同 sample size、sample realization 与 action budget，可用于 representation comparison。standard--broad 同时改变 representation、sample size、sample realization 与 action budget，只能解释为整体 query configuration 差异，不能把观察差异拆成任一单独来源。

`lhs_50d`、`lhs_100d`、`lhs_5d`、`lhs_10d`、`lhs_20d` 只定义“每维样本数”，不是固定 FE ratio。实际 `FE_query` 仍由 `FE_query = sample_size_per_dimension × dimension` 给出，`FE_query / FE_total` 由具体基准预算决定。该设计允许同一采样设计在未来预算变化时复用，而不把预算口径写死在采样设计名义中。

这三个 query 的统一前处理版本固定为 `query_preprocessing_id = unit_cube_x__median_iqr_y_v1`：所有坐标先映射到 unit cube，再对目标值做样本内 median/IQR robust 标准化；IQR 退化时使用冻结 fallback。仓库内 cheap 与隔离 pflacco 两档提取器均已实现该前处理。该 preprocessing 只改变送入特征计算的数组，不改变原始样本文件中的 `X, y, lower_bounds, upper_bounds`。实现完成不等于正式数据或结果完成，正式 features 仍须通过三档一致性检查后生成。

standard 包含 PCA、NBC、dispersion、information content 和 ELA distribution。broad 在 standard 基础上加入 ELA level-set 与 sample-derived fitness-distance correlation。不包含完整 quadratic `ela_meta`、cell mapping，也不包含需要额外函数评价的 local、curvature、convexity、Sobol 和 length-scale groups。

## 3. 采样与特征边界

主环境只负责生成 benchmark 问题的 LHS 样本并保存 (X,y)：

```text
results/landscape_queries/samples/{sample_design_id}/{split}/samples.parquet
```

样本行保存 `sample_design_id`、`sampling_protocol`、`sample_seed`、`sample_size`、`FE_query`、`runtime_query_sampling`、`runtime_query_evaluation`、兼容汇总字段 `runtime_sampling_evaluation`、`query_first_hit_offset`、边界、(X) 和 (y)。其中 `sample_size = sample_size_per_dimension × dimension`，而不是固定 FE ratio。`query_first_hit_offset` 记录 query sample 内首次达到 suite success target 的评价偏移；它用于主 operational Query path 的 success/ERT，不能进入 Selector 或 Decision 输入。随机种子由 base seed、stream code、function、instance、dimension 与整数 design code 共同进入 `numpy.random.SeedSequence`；不得使用字符串哈希。

特征结果保存到：

```text
results/landscape_queries/features/{query_id}/{split}/features.parquet
```

每行必须包含 `query_id`、`query_protocol`、`query_preprocessing_id`、`sample_design_id`、固定 `query_feature_columns`、`runtime_query_feature_computation`、兼容别名 `runtime_feature_computation`、`feature_status`、逐 group 状态、非有限值记录与失败信息，并满足：

```text
runtime_sampling_evaluation = runtime_query_sampling + runtime_query_evaluation
runtime_query = runtime_query_sampling + runtime_query_evaluation + runtime_query_feature_computation
additional_function_evaluations = 0
```

单个数学上未定义或非有限的特征保存为 null，并在 `feature_nonfinite` 中记录；不得替换为自定义常数。group 异常将该 group 的列保存为 null，并在 `feature_group_status` 与 `feature_failure` 中记录。

## 4. pflacco 隔离环境

主项目不依赖 pflacco。标准特征只可由 `tools/pflacco_query/` 的 Python 3.11 环境提取，该环境固定 `pflacco==1.2.2` 及其兼容 NumPy、Pandas、SciPy 与 scikit-learn。工具只读 Parquet 样本，对 X/y 执行上述 invariant preprocessing 后送入 pflacco，并写 Parquet features；不导入 benchmark，不评价目标函数，不安装 R 包，不在运行时下载依赖，也不回退到自定义公式。终点评价字段可随行保留为 metadata，但不得进入 `query_feature_columns` whitelist。

NBC 固定 `dist_tie_breaker="first"`；information content 使用显式整数 `SeedSequence` seed。每个 group 的返回列必须与 pflacco 1.2.2 白名单完全一致，`*.costs_runtime` 只用于组内诊断，不进入 selector feature 列。

## 5. Query-specific Selector、Utility 与 Decision

`lhs_50d` 只生成一份逐共享状态 action-loss 表，供 cheap 与 standard 分别训练 Selector；`lhs_100d` 单独生成 action-loss 表。action-loss 输入包含 `sample_design_id` 与 `FE_query`，不包含 behavior 或 query features。构建 Selector 时才连接 behavior 与指定 query feature 表。

Selector action target 使用 continuation-only outcomes。主 operational Query path 虽不把 query sample 插入 optimizer population，但 terminal best、`observed_first_hit_FE`、`target_hit_observed` 与 ERT 必须计入 query sample 的真实 objective evaluations；`endpoint_success` 另要求 continuation path 完成。另保存 continuation-only gap 与 `query_sample_best_contribution_log10_gap`。同一 query-adjusted action matrix 上的 `query_feature_predictive_increment_log10_gap` 必须排除 sample best，避免把随机采样直接找到更优点归因于 descriptors。

每个 query 独立拟合 Selector、Utility target 和 Decision Model。若未来新增预算档，只需新增对应的 `sample_design_id` 与相应配置，不需要改动 `query_id` 的语义：

```text
results/selection_reference/{query_id}/
results/utility_labels/{query_id}/
results/decision/{query_id}/
```

Selector artifact 保存 `query_id`、`query_protocol`、`query_preprocessing_id`、`sample_design_id` 和实际 `query_feature_columns`。Decision Model 输入仍严格限定为算法无关 `bf_*` behavior；query id、query features、function、dimension、algorithm 和优化器内部状态都只作 metadata 或分层报告。项目不训练动态 query-type selector。

模型比较、特征组消融、阈值分析、baseline、成本—性能与外部评价命令均要求显式 `--query-id`。命令据此推导 query-specific 默认目录，并核对 artifact 中的 `query_id`、`query_protocol`、`query_preprocessing_id` 与 `sample_design_id`；不同 query 的 dataset、summary 或 prediction 表不能交叉读取。

活动标签必须至少保存 `FE_query`、query 组件 runtime、Stage-A 科学端点来源及五条 gap/observed-first-hit/target-hit/path-completion/endpoint-success/planned/effective FE、Stage-B 三次 decision-state future-path raw/censored runtime、逐次 status/effective FE/timeout/completion、三类一致性与两类 instability、FE=0 policy wall-clock、截断后的五条 `log10_gap`、`u_query_joint_lamT_*`、`u_behavior_only_full_budget_lamT_*`、`query_operational_increment_lamT_*`、三类 matched-acquisition increment 及对应布尔标签。`first_hit_FE/success` 若保留，只能分别作为 `observed_first_hit_FE/target_hit_observed` 的兼容别名。旧单一 `u_query_lamT_*`、`results/ela/`、`FE_analysis`、`p_ela`、`u_ela_*`、`need_ela_*` 和缺少当前协议字段的 artifact 均为撤回结果，活动读取器必须明确拒绝。

## 6. 数据质量与失败处理

BBOB train/validation 不允许 group-level extraction failure。单个 null 只使用 BBOB-train median imputation；任何在 BBOB train 上整列缺失的 query feature 都阻止 Selector 拟合。

外部 benchmark 允许保留 group-level failure，并由已拟合 Selector 的 BBOB-train median 执行 fallback。所有受影响行必须单独报告 `feature_status` 与 `feature_failure`；只要存在 group-level failure，就不能给出该 query 的无条件外部泛化结论。

BBOB train/validation 与 CEC2017 使用配置固定的 `failure_loss_cap=1e20`、取 `log10` 前 raw-gap floor/cap `1e-12/1e20`、`success_gap_target=1e-8`、单 state-action path timeout `3600 s` 和逐 objective evaluation first-hit 记录。timeout/failed 行计路径失败并保留；失败前已经观察到的 target hit 不抹除，标准 ERT 使用该 observed first hit，只有未命中项计完整 planned budget。CEC2022 与工程问题必须各自先固定同类字段及 constraint rule。

`query-consistency` 是非 pytest 的可执行一致性入口。它检查三档预算、feature whitelist、cheap/standard 的共享样本键、零额外函数评价、BBOB group failure、整列缺失和 action-loss 预算隔离。`utility-labels-validate` 逐行重算五条路径及加法分解。令 `ell_*` 是 Stage-A raw gap 先应用 suite floor/cap 后的 `log10_gap`；Stage-B 每次 replay 的 censored time 在 completed 时等于 raw，在 timed-out/failed 时为 `max(raw, role timeout)`，`T_*` 是三次 censored future-path wall-clock 的中位数。raw observed median 只作诊断。共享 prefix 视为 sunk cost；Stage-B 不改写科学端点，也不选择性补跑：

```text
p_query = selected_query_action_loss
p_behavior = selected_behavior_only_action_loss
u_query_joint_lamT_* = (ell_skip - ell_query) - lambda_T * (log10(T_query) - log10(T_skip))
u_behavior_only_full_budget_lamT_* = (ell_skip - ell_behavior) - lambda_T * (log10(T_behavior) - log10(T_skip))
query_operational_increment_lamT_* = u_query_joint_lamT_* - u_behavior_only_full_budget_lamT_*
need_query_joint_lamT_* = (u_query_joint_lamT_* > 0)
```

## 7. 报告与结论口径

分别报告三个 query 配置的 joint Utility、Behavior-only Utility 与 query operational increment；`I_q` 同时覆盖全 eligible states 与 Proposed-triggered states，并明确其包含 query FE/runtime、sample best、预算差和 Selector 差异，不是纯信息效应或因果 estimand。若 `U_joint>0` 而 `I_q<=0`，只能支持联合路径优于 SBS。每档同时报告五路径的 descriptor-use、state-only-vs-sampling 与 sampling-direct 增量及逐行加法一致性。每档报告 Never/SBS、Always Query、`matched_rate_random`、`pre_run_aas_fe0`、`milestone_only_T0`、`self_thresholded_behavior_only`、Proposed 和静态 VBS reference，并至少包含 query FE、三次 raw/censored future-path 时间及两种中位数、FE=0 policy wall-clock、失败率、selector regret、action-loss 回归性能、Utility 分布、调用率、效用捕获、`log10_gap`、target-hit rate、endpoint-success rate、ERT，以及 function 层面的配对效应量与区间。matched-acquisition 分解只作固定协议下的操作性解释。

若三档结论一致，只能表述为“结论在三个预定义 query 配置上具有稳健性”。若结论不一致，必须报告 representation 与成本依赖性，不能隐藏结果或重新定义 query。NeurELA 和 Deep-ELA 本轮只用于说明 landscape representation 的异质性。

## 8. 固定执行顺序

```text
trajectory 重生成
→ behavior
→ lhs_50d / lhs_100d samples
→ descriptor_cheap_invariant / pflacco_standard_invariant / pflacco_broad_invariant features
→ lhs_50d 与 lhs_100d 两档 action losses
→ 三个 Selectors
→ 三套 Utility labels
→ 三个 Decision Models 与 baselines
```

在 query 配置、采样边界、隔离 pflacco 提取与一致性命令通过前，不启动 72 个 trajectory shard 的全量重生成。截至 2026-08-11，这些前置检查已在真实 BBOB 10D 关键路径通过；本轮仍未启动正式 shards。
