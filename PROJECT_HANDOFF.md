# Decision-before-Feature 项目交接记录

本文用于跨对话同步当前项目状态。开始新任务时依次阅读：

1. `AGENTS.md`
2. `README.md`
3. 本文件
4. `DEVELOPMENT_DECISIONS.md`
5. 与任务直接相关的 `docs/` 文档

所有工作只使用当前项目目录。

## 当前阶段（2026-08-12）

> 重建式 continuation 已被确认会在每个 decision checkpoint 丢失优化器内部状态；旧 behavior 还错误地把 population 行号当作跨代个体身份。实现现已改为完整状态原生推进和 permutation-invariant 集合特征；当前阶段是“从 trajectory 开始重生成全部正式内部证据”，不得继续沿用旧 behavior、标签或模型。

实现修正已经完成：

- DE、PSO、CMA-ES、SHADE 均有显式完整状态对象和 RNG state；
- 同算法 checkpoint 使用原生状态保存/恢复，不再重新初始化；
- 第一篇论文主标签只使用训练集 SBS 产生的 prefix，No-query 原生继续同一 SBS 完整状态；
- 全 prefix 标签只用于 cross-probe robustness、leave-one-probe-out 与 algorithm-agnostic 泛化；
- No-query 或 Query 分支只有在目标算法不同于 prefix 时才执行一次 population transfer 初始化；
- 标签显式区分 `selected_equals_default`、`selected_equals_prefix`、`handoff_required` 与 `skip_switches_from_prefix`；
- 在线主协议使用与离线训练相同的动态输出状态，全部策略共享同一连续状态与 decision opportunities；
- 真实 BBOB 上的 checkpoint 保存/恢复与不中断运行逐状态完全一致。
- behavior 的跨 checkpoint 空间变化使用经验 Wasserstein、centroid shift 与集合协方差形状，fitness 变化使用排序后的经验分位数；不保存 individual ID 或 ancestry。
- trajectory 保存 `FE_total`、完成的 `native_updates`、三个窗口的轻量集合/fitness 统计和最近10%预算内的逐 update 标量历史；w02/w05/w10 anchor 从逐次完整原生 update 中选择，量化误差小于一次 update。behavior 保存实际 `effective_window_ratio_*`、`effective_window_fe_*`、`effective_native_updates_*`，这些窗口测量 metadata 不进入 Decision 输入。
- 行为字段合同已冻结为 34 个唯一输出、31 个正式输入和 3 个诊断字段，`T0/B1/B2/B3=1/19/25/31`。`bf_fitness_diversity_rel` 是唯一的相对 IQR 字段，baseline 为优化器初始化后、任何原生 update 前的已评估 population IQR。
- `behavior-permutation-check` 在真实 BBOB 上对四种优化器逐状态独立打乱 population 行序，冻结行为特征与 9 个窗口测量字段必须保持逐值一致。
- 主采样协议为 `phase1_dynamic_budget_event_v1`：`0.20–0.60` 步长 `0.01` 的候选监测网格、12 个预算里程碑加状态事件，每个跨过至少一个 `0.01` 监测网格的完整原生 update 只判定一次事件；里程碑合并行不消耗 event-only 配额、最小间隔锚点或 `event_index_in_phase`，每个 run 输出 12–18 个状态。
- Selection Reference 当前协议为 `query_specific_statewise_action_loss_regression_v5`：动作集合为 `continue_current` 加其余三个 portfolio algorithms，固定多输出 Random Forest 预测 `statewise_minmax_observed_action_loss`，remaining budget 连续输入，train 行使用 function-family cross-fitting；旧静态 bucket classifier 不再生成正式标签。
- Utility label 增加 `best_observed_algorithm/loss`、`potential_gain_raw` 与 `selector_regret_raw`；handoff 影响和 query sampling FE 均不重复扣除。
- Selection Reference、Utility、Decision dataset 与在线策略输出统一保存 `selected_equals_default`、`selected_equals_prefix`、`handoff_required`、`no_query_algorithm` 和 `handoff_type`；活动分析不再生成 selected-vs-default 字符串分层。
- 三档 `LandscapeQuerySpec`、共享/独立 LHS 样本边界、隔离 pflacco 1.2.2 提取和 query-generic 数据契约已实现。
- query-sensitive 的模型比较、阈值、baseline、成本—性能和外部评价命令均要求显式 `--query-id`，并从 query-specific 目录读取；模型冻结后的 Q10 邻近带复查使用独立输出目录。
- Decision feature-group、baseline 与成本—性能比较已加入 `time_only_controller`：数学输入 `X={FE_ratio}`，实现列仅为逐行等于 `FE_ratio` 的 `bf_fe_ratio`；Time-only 与主 Controller 的预测文件、模型名、逐状态样本和实际推理时间分别核对。
- Decision Model 活动候选已收敛为 LDA、Logistic Regression 与 Ridge；主选择使用 BBOB-train 嵌套 function-family OOF decision utility，完整 train 的 family-OOF 分数冻结 `oof_utility` 阈值，BBOB-validation 只作冻结评价。连续 Utility RMSE 只对 Ridge 计算。
- 真实 BBOB 10D 上 cheap/standard 共享样本检查与 broad 52 列提取检查均已通过；40D cheap 距离计算使用等价的低内存 `pdist`/`cKDTree` 路径并通过真实提取检查。
- 优化器状态、behavior行排列不变性、逐状态四动作 loss、Selector 目标变换、显式动作关系、Utility 分解、function-family 配置、Decision 输入边界与全仓编译检查通过。
- 本轮没有启动 72 个正式 trajectory shards。

必须重新完成：

- 覆盖生成 BBOB train/validation trajectory shards，并重提取 behavior；
- 生成 state-action loss shards，重新拟合 Selection Reference，再生成 utility labels 和 Decision dataset；
- 重新执行三候选嵌套 OOF 选择、冻结阈值、baseline、消融和成本—性能评价；
- 完整 CEC2017 29 functions × 3 dimensions × 30 seeds 外部评价；
- CEC2022 和工程问题外部评价；
- 论文级统计推断和最终图表。

## 正式数据协议

BBOB train families：

```text
1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 15, 16, 17, 18, 20, 21, 22, 23
```
BBOB validation families：

```text
5, 9, 13, 14, 19, 24
```

共同设置：

```text
dimensions: 10, 20, 40
instances: 1, 2, 3
optimizer seeds: 1 ... 30
algorithms: de, pso, cmaes, shade
population_size: 40
FE_total: 1000 * dimension
sampling_protocol: phase1_dynamic_budget_event_v1
monitor_grid: 0.20--0.60, step 0.01
budget_milestones: 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.60
states_per_run: 12--18
final_performance: one row per problem_id/algorithm/seed at FE=FE_total
```

`final_performance.parquet` 与上述 `0.20–0.60` decision trajectory 分开保存。SBS 只读取完整预算终值表：先按 `problem_id × algorithm` 对全部 seeds 的 `best_fitness` 取算术均值，再在每个 problem 内按越小越好排名，最后按 algorithm 跨 problem 平均排名；平均排名并列时按冻结 portfolio 顺序 `de, pso, cmaes, shade` 决定。不得从 trajectory 的最后一个 decision state 推导 SBS。

trajectory 与 `final_performance.parquet` 作为同一 shard 输出 pair 发布；覆盖采集期间不得并发运行 behavior extraction、SBS 或其他下游读取。中断后若 pair 为 missing/partial，必须用 `--overwrite` 成对重生成。

正式配置：

- `configs/phase1_bbob_train.yaml`
- `configs/phase1_bbob_validation.yaml`
- `configs/phase1_cec2017_test.yaml`

## 旧结果状态（不得作为论文证据）

以下数值来自重建式 continuation 和旧 identity-dependent behavior 数据，仅用于说明被撤回结果的范围；完成全链路重生成前不得称为当前正式结果。

Decision dataset：

- train：194,400 rows，`U_query>0` 比例 6.1322%；
- validation：64,800 rows，`U_query>0` 比例 5.7793%。

当前主输入候选：

- 正式 feature-group 消融只包含 `T0/B1/B2/B3=1/19/25/31`；T0 只含 `bf_fe_ratio`，用于判断 Controller 是否只学习调用阶段；
- 旧 `primary_with_maturity` 的 23 字段计数只属于本节已撤回结果；当前 `primary_with_maturity` 与 `all_candidates` 都是 B3 的兼容别名，严格等于 31 个正式输入；
- `all_candidates` 不含 `diagnostic_only` 的 3 个字段，也不是独立的第五个正式消融组；
- function、dimension、algorithm、query feature 和优化器内部参数不进入模型输入。

统一模型比较的最佳可部署策略：

```text
model: LDA classifier
threshold source: BBOB train only
withdrawn validation query call rate: 0.0378395
validation mean decision utility: 0.00432497
validation utility capture: 0.561033
validation precision among calls: 0.312806
```

调参后的 shrinkage LDA：

```text
validation mean decision utility: 0.00435820
validation utility capture: 0.560990
validation precision among calls: 0.315074
```

该差异只记录旧流程当时的观察，不能继续据此选择 LDA 或保留 shrinkage 调参。新协议只比较三个固定候选，并在重生成后的 BBOB-train 上按嵌套 family-OOF decision utility 重新选择。

## 结果解释边界

当前没有可作为论文证据的内部 validation 数值。上节数值只记录撤回范围，不能证明搜索行为可预测任一 query 的效用，也不能支持 LDA、Always Query 或 Random Analysis 的比较结论。旧 16 维 landscape 表的构念命名同样无效：它只能解释为现已命名的 `descriptor_cheap`，不能代表完整 ELA 或 pflacco。

不能作为最终论文结论：

- 旧 Ridge baseline/Pareto 是最新 LDA 的最终结果；
- preliminary CEC2017 在线运行证明了跨 benchmark 泛化；
- top-10% capture 高就等价于正平均效用；
- Search Maturity 单独造成性能改善；
- exact utility magnitude 已经良好校准。

详细数值和来源见 `docs/30_results/phase1_current_results.md`。

## 当前数据目录

截至 2026-08-11，所有旧实验产物已移出活动路径，本机封存在 `results/archive/withdrawn_20260811/` 及原有 `results/archive/` 子目录；这些目录受 `.gitignore` 排除，不会同步到 GitHub。当前没有可直接复用的正式模型或结果。

当前活动目录契约：

```text
results/phase1_refined_sampling/
results/landscape_queries/samples/{sample_design_id}/{split}/
results/landscape_queries/features/{query_id}/{split}/
results/selection_reference/{query_id}/
results/utility_labels/{query_id}/
results/decision/{query_id}/
```

主 query 是 `descriptor_cheap`；`pflacco_standard` 与 `pflacco_broad` 只用于配置稳健性。旧 `results/ela/` 及缺少 query 协议字段的 artifact 已撤回且不得读取。

preliminary CEC2017 运行放在：

```text
results/archive/cec2017_preliminary/
```

该目录只用于定位实现问题，不作为当前结果入口。

## 清理后的代码边界

已从当前运行面删除：

- MVE 和旧 phase1 配置；
- preliminary/min_support 专用 Decision 脚本和命令；
- 外置 `full_training_readiness` 启动许可；训练入口直接执行数据、输入字段、标签和 family split 检查；
- 被正式 phase1 取代的旧采样结果、模型、预测和图表已移出活动路径并在本机封存；
- WPS 中间图片、缓存、`.DS_Store`；
- 可再生的 Random Forest 大模型文件和重复 threshold-sweep CSV。
- Decision Model 的 Random Forest、XGBoost、LightGBM、MLP、Linear SVM 等扩展候选，以及依赖 validation utility 的分类特征工程调参和旧模型解释脚本；撤回数值仍保留在结果说明中。

preliminary/min_support 的关键归因矩阵与说明文档保存在 `docs/archive/min_support/`，只用于研究脉络追溯。

## 绝对约束

- 不访问项目目录之外的历史文件。
- 不改变 function-family split、冻结的 `phase1_dynamic_budget_event_v1` 采样参数或等总 FE 预算。
- 不将 query features、function id、dimension、algorithm id 或优化器内部参数放入 Decision 输入。
- 不使用测试 benchmark 拟合 preprocessing、模型或 threshold。
- 不使用 BBOB-validation 选择 Decision 模型或 threshold；活动候选只允许 LDA、Logistic Regression 与 Ridge。
- 不在线训练 controller 作为主实验。
- 不重复扣除 query FE 成本。
- 不将 VBS 写成现实可部署方法。
- 不将 preliminary/min_support、MVE 或旧采样结果写成正式结论。
- 不引入 pytest、测试目录、JSON Schema、文件身份或执行解锁机制。
- 随机流只使用显式整数和 `numpy.random.SeedSequence`。

## 下一步

推荐按以下顺序继续：

1. 已完成：optimizer-state、Selection Reference、三档样本/特征与 broad 真实提取一致性检查。
2. 下一运行阶段对 BBOB train/validation 的 72 个 trajectory shards 使用 `--overwrite` 全量重生成，再覆盖提取 permutation-invariant behavior。
3. 按 `lhs_50d` 与 `lhs_100d` 生成两档 action-loss shards，再独立构建三个 Selector、三套 Utility labels和三套 Decision dataset/model；检查 query 预算、`selector_target_transform`、三个显式动作关系字段与 best-observed-action 分解逐行一致。
4. 对每档执行三候选嵌套 family-OOF 模型选择、完整 train OOF 阈值冻结、baseline、ablation 和成本—性能评价；每档必须同表报告 `time_only_controller` 与主 Controller，并在内部证据重建后再启动 CEC2017、CEC2022 和工程问题评价。

可直接复制的下一步 prompt：

```text
请阅读 AGENTS.md、README.md、PROJECT_HANDOFF.md、DEVELOPMENT_DECISIONS.md 与三档 Landscape Query 协议。三档 query 与完整状态一致性检查已通过；现在按 `phase1_dynamic_budget_event_v1` 重生成 BBOB train/validation 的 72 个 trajectory shards 与 permutation-invariant behavior。每个 run 必须包含 12 个里程碑并在冻结事件规则下产生 12–18 个状态；`FE_ratio=FE/FE_total`，状态连接使用整数 `FE`，不用浮点 ratio 作键。随后生成共享的 `lhs_50d` action losses 和独立的 `lhs_100d` action losses，分别构建 `descriptor_cheap`、`pflacco_standard`、`pflacco_broad` 的 Selector、Utility labels、Decision dataset/model 与 baselines。每档 Selector 必须使用四个唯一动作、多输出 Random Forest、`statewise_minmax_observed_action_loss`，并逐行核对 `selected_equals_default`、`selected_equals_prefix`、`handoff_required` 与 `handoff_type`。每档 Decision Model 只比较固定的 LDA、Logistic Regression 与 Ridge，以 BBOB-train 嵌套 function-family OOF decision utility 选择模型，并用完整 BBOB-train family-OOF 分数冻结 `oof_utility` threshold；BBOB-validation 只作评价。每档必须同表报告 `time_only_controller`（`X={FE_ratio}`，实现列仅 `bf_fe_ratio`）与主 Controller，使用同名模型、相同 OOF 过程、held-out family 和外部评价口径。必须逐阶段运行一致性命令，检查 sampling metadata、整数 FE 状态键双向覆盖、query_id/query_protocol/sample_design_id、FE_query、p_query、runtime_query、Utility 分解和 function-family split，不得混用不同预算或旧 `results/ela` artifact，不得修改冻结采样参数、算法池、等总 FE 或 Decision 输入边界。
```
