# min_support 问题归因矩阵（历史）

本文基于当前 `results/decision/min_support/` 下已有诊断输出整理，不重训模型，不生成新 utility labels，不使用 ELA features，不改变正式 phase1 配置。

## 总体判断

当前 min_support 结果已经支持如下判断：

- Decision-before-Feature 的机会存在：`changed_algorithm_validation` 中逐状态最佳动作的 `utility_sum=64.7292`，说明不是所有状态都应跳过 ELA。
- 当前可部署 Decision 策略仍不稳定：gated Pareto 中 `all_validation` 的可部署 frontier 仍由 `No ELA / SBS` 主导；`changed_algorithm_validation` 上仅有小幅收益。
- 当前主要瓶颈不是单一模型族，而是数据覆盖、行为来源混杂、阈值迁移和 utility 来源耦合共同造成。

## 归因矩阵

| 问题类别 | 已有证据 | 对 Decision-before-Feature 结论的影响 | 下一轮最小修复实验 |
|---|---|---|---|
| label coverage | `fe_transition_model_sensitivity` 唯一 state 统计：train 中 `changed_algorithm` 1040 rows，仅 9 rows 满足 `U_ELA>0`，比例 `0.87%`；validation 中 `changed_algorithm` 1900 rows，有 245 rows 满足 `U_ELA>0`，比例 `12.89%`。`U_ELA>0` 主要集中在 validation 的 `FE_ratio=0.30-0.552`。 | 训练集几乎没有覆盖“算法切换且 ELA 有用”的状态，模型阈值容易学成“几乎不调用 ELA”。因此当前结果更能证明机会区域存在，不能证明 Decision Model 已经学到稳定泛化规则。 | 不改变正式 phase1；新增 min_support train 覆盖扩展。重点补 `changed_algorithm`、`FE_ratio=0.30-0.55`、与 validation `bbob_f005/bbob_f024` 行为相近但不破坏 family split 的 train family/problem/seed。输出扩展 labels 和诊断模型副本。 |
| algorithm-behavior confounding | 唯一 state 统计显示 prefix algorithm 下的 `U_ELA>0` 分布很不均衡。train changed rows：`cmaes=0/260`、`de=3/260`、`pso=0/260`、`shade=6/260`；validation changed rows：`cmaes=133/475`、`de=24/475`、`pso=31/475`、`shade=57/475`。Decision 输入无 algorithm id，但 behavior features 会携带算法轨迹风格。 | 模型可能把“算法产生的行为签名”当成 ELA utility 信号。该影响不违反协议，因为输入仍是算法无关轨迹行为；但如果 train/validation 的 prefix behavior 分布不同，会削弱泛化解释。 | 新增只读诊断：按 `prefix_algorithm` 分层比较 behavior feature 分布、score 分布、`U_ELA>0` 覆盖、误调用成本。保持 algorithm id 仅用于分组，不作为 Decision 输入。必要时做 prefix-balanced sampling 或 prefix-stratified reporting。 |
| threshold transfer / calibration | `family_stage_threshold_transfer` 显示 train family 与 validation family 交集为 0。`family_stage_train_utility_threshold` 在 `changed_algorithm_validation` 的 44 个 family-stage groups 中 100% 缺失；train family-stage transfer `utility_sum=0`，而 validation descriptive family-stage opportunity 为 LightGBM `54.0103`、RF `56.6553`、XGBoost `54.6588`。gated Pareto 中 all_validation 仍不超过 `No ELA / SBS`。 | 当前 family-stage threshold 的 validation 机会是描述性上界，不能作为可部署策略。train-derived 阈值迁移失败首先来自 family key 不可转移，其次是 train 中 `U_ELA>0` 稀疏导致阈值过保守。 | 改做可迁移阈值诊断：比较 FE-only stage threshold、score quantile threshold、behavior-bucket threshold、search-maturity bucket threshold。不使用 held-out family 作为阈值 key；只用已有 predictions 先做后处理评估。 |
| downstream ELA-based selection pipeline quality | `selector_transition_diagnostic` 显示 0.25 到 0.30 的 changed_algorithm 出现主要由 performance bucket 边界驱动：`stage_025_to_030_performance_bucket_change_rate=1.0`，`selected_algorithm_change_rate=0.75`。`performance_bucket_sensitivity` 和 `bucket_smoothing_diagnostic` 均支持 sparse nearest bucket mapping 会造成 selected_algorithm 不连续。 | 我们不研究算法选择器本身，但 `U_ELA=(P_skip-P_ELA)-lambda C_ELA` 中的 `P_ELA` 依赖固定的 ELA-based selection pipeline。若该下游路径在 bucket 边界不稳定，Decision Model 看到的 utility label 会混入下游选择路径的离散误差。 | 保持 selector 固定为实验组件；新增 sensitivity reporting：在现有 selection_reference 上报告 nearest/lower/upper/interpolated bucket 对 `selected_algorithm` 与 utility proxy 的影响。论文中将其写成 fixed downstream pipeline limitation，而不是方法贡献。 |
| same_algorithm random continuation noise | `label_source_check` 显示 `selected_algorithm == default_algorithm` 时仍存在 `U_ELA>0`：same_algorithm overall 为 52/1680，比例 `3.10%`；changed_algorithm overall 为 27/720，比例 `3.75%`。在 fe_transition unique validation 中 same_algorithm 为 34/2100，比例 `1.62%`，其 `sum_positive_utility=3.1410`，远小于 changed_algorithm 的 `64.7292`。 | same_algorithm 中的 `U_ELA` 更像共享前缀后续随机差异参照，而不是 ELA 改变算法带来的主要收益。若与 changed_algorithm 完全等权训练，会污染 Decision Model 的监督信号。 | 继续保留 same_algorithm_reference，但训练诊断中分开报告。比较 full-label、changed-only、changed-weighted 训练副本；正式 label 不改，论文中把 same_algorithm 写作 continuation randomness control。 |
| behavior feature insufficiency | `f024_behavior_separability` 显示现有 behavior features 在 target holdout 上有一定排序能力但不充分：RF top-20% capture `0.3077`，Logistic top-20% capture `0.3846`，best existing single-feature F1 `0.5946`。候选特征诊断中 `cf_elite_centroid_shift_norm` 在 target holdout FE=0.50 上 F1 `0.6341`、recall `1.0`，提示现有 bf_* 仍可能缺少 population/fitness 形态信息。two-feature guard 在 target holdout 表现较好，但在 non-holdout/extension_train 上 precision 降至 `0.30`，`f024_i02_d10` 归因为 sample coverage insufficient with behavior feature insufficiency。 | 当前 behavior features 能捕捉一部分 search maturity，但对 f024 late-stage 的有用 ELA 与误调用区域区分不稳。换模型族不能完全解决特征空间重叠问题。 | 保持不使用 ELA/function/algorithm id；最小新增算法无关 population/fitness behavior features，如 elite centroid shift、elite fitness gap、diversity recovery。先做候选特征诊断与可迁移阈值评估，再决定是否进入正式 extractor。 |

## 证据文件索引

- Label coverage 与模型预测：`results/decision/min_support/fe_transition_model_sensitivity/model_sensitivity_train_predictions.parquet`；`results/decision/min_support/fe_transition_model_sensitivity/model_sensitivity_predictions.parquet`
- Gated Pareto：`results/decision/min_support/gated_pareto_diagnostic/gated_pareto_frontier.parquet`；`results/decision/min_support/gated_pareto_diagnostic/gated_pareto_points.parquet`
- Family-stage 阈值迁移：`results/decision/min_support/family_stage_threshold_transfer/family_stage_threshold_transfer_summary.json`；`results/decision/min_support/family_stage_threshold_transfer/family_stage_transfer_failure_reasons.parquet`
- Label source 与 same_algorithm 参照：`results/decision/min_support/label_source_check/label_source_summary.parquet`；`results/decision/min_support/label_source_check/label_source_layered_summary.parquet`
- Selector transition 与 bucket sensitivity：`results/decision/min_support/selector_transition_diagnostic/selector_transition_diagnostic_summary.json`；`results/decision/min_support/performance_bucket_sensitivity/performance_bucket_sensitivity_summary.json`；`results/decision/min_support/bucket_smoothing_diagnostic/bucket_smoothing_diagnostic_summary.json`
- Behavior overlap 与 f024 可分性：`results/decision/min_support/behavior_overlap/behavior_overlap_pairwise.parquet`；`results/decision/min_support/f024_behavior_separability/f024_behavior_separability_summary.json`；`results/decision/min_support/f024_behavior_feature_candidates/f024_behavior_feature_candidate_summary.json`；`results/decision/min_support/f024_two_feature_guard_stability/f024_two_feature_guard_domain_performance.parquet`；`results/decision/min_support/f024_i02_d10_guard_failure/f024_i02_d10_guard_failure_summary.json`

## 优先级建议

1. 先补 label coverage：尤其是 train 中 `changed_algorithm` 且 `U_ELA>0` 的 0.30-0.55 区域。
2. 同步做 prefix_algorithm 分层诊断，确认模型是否过度利用算法行为签名。
3. 将 family-stage threshold 改为可迁移的 stage/behavior-bucket calibration。
4. 保留 same_algorithm_reference 作为随机续跑差异参照，不与 changed_algorithm 混为同一解释。
5. 对 f024 late-stage 增加少量算法无关 population/fitness behavior feature 候选。
