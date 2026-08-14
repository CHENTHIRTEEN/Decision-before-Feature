# Decision-before-Feature

本项目研究一个前置于特征计算的资源决策问题：在黑盒优化过程中，是否值得为当前搜索状态执行一项预先定义的 landscape-analysis query。项目不设计新的优化算法，而是用离线轨迹和监督学习估计固定 query 的状态依赖效用。

## 当前状态

截至 2026-08-12，四种优化器已改为完整状态推进：同一算法在 checkpoint 间保留内部状态与 RNG；只有 selector 确实切换算法时，才执行一次显式的 population transfer 初始化。可用 `uv run optimizer-state-check` 在真实 BBOB 问题上检查连续运行与多次 checkpoint 保存/恢复的一致性。

Behavior extractor 同时已改为 permutation-invariant 的种群集合统计：跨窗口的空间变化使用经验 Wasserstein、centroid shift 和协方差谱集中度，fitness 变化使用经验分位数分布；不再把 population 行号解释为跨代个体身份。运行时逐次记录完整原生 update 的轻量窗口统计，正式 behavior state 的 w02/w05/w10 anchor 不再从稀疏输出状态中选择；若名义 FE 不能整除一次原生 update，则取不晚于目标位置的最近完整 update，误差严格小于一次 update，并保存 `effective_window_ratio_*`、`effective_window_fe_*` 与 `effective_native_updates_*`。所有 rate/slope 使用实际 `ΔFE/FE_total`，这些窗口字段只作 metadata，不进入 Decision 输入。

正式状态采样已冻结为 `phase1_dynamic_budget_event_v1`：在 `0.20–0.60` 上按 `0.01` 候选网格监测，保留 12 个预定义预算里程碑，并依据 improvement resume、stagnation onset、effective-rank change、elite migration 与 diversity recovery 补充事件状态；每个跨过至少一个 0.01 监测网格的完整原生 update 只判定一次事件。同一 update 跨过多个监测点时，若包含预算里程碑，则里程碑与事件合并为一行，且该行不消耗 event-only 配额、最小间隔锚点或 `event_index_in_phase`；若不含里程碑，则以最新跨过的监测点作为名义节点。每个 run 输出 12–18 个状态。`FE_ratio` 始终是实际 `FE/FE_total`，名义里程碑另存 `budget_milestone_ratio`，状态连接使用整数 `FE` 而非浮点 ratio。完整预算终值另存为每个 `problem_id × algorithm × seed` 在 `FE=FE_total` 恰好一行的 `final_performance.parquet`；该表与 `0.20–0.60` decision trajectory 分离，不能把 `0.60` 的最后一个 decision state 当作完整预算终值。

Selection Reference 已改为逐共享状态候选动作损失回归：每个 state 对 `continue_current` 和其余三个 portfolio algorithm 分别进行真实 continuation，`remaining_budget_ratio` 作为连续输入；不再按静态 problem label 和 nearest performance bucket 选择算法。固定的多输出 Random Forest 预测 `statewise_minmax_observed_action_loss`，训练行使用按 function family 的交叉拟合预测，验证与外部评价加载仅由 BBOB train 拟合的 selector model。

三档 query 的实现边界检查已通过：真实 BBOB 10D 上 cheap/standard 共享 `lhs_50d` 样本，broad 独立使用 `lhs_100d` 并由隔离的 pflacco 1.2.2 环境完整提取 52 列；三档均无额外函数评价。72 个正式 trajectory shards 尚未启动。

Decision Model 的活动候选固定为 LDA、Logistic Regression 与 Ridge。模型由 BBOB-train 上嵌套 function-family OOF decision utility 选择：内层 OOF 拟合外层阈值，外层 OOF 比较模型；最终阈值由完整 BBOB-train 的 family-OOF 分数冻结，BBOB-validation 只作冻结评价。Random Forest、XGBoost、LightGBM、MLP 及分类特征工程搜索已退出 Decision Model 活动调参路径；Selection Reference 中预先定义的 Random Forest action-loss regression 不受此约束。

此前生成的 BBOB trajectory 使用了重建式 continuation，旧 behavior 含有依赖行号对应关系的字段，旧 landscape 表又把 16 个自定义描述符笼统称为 ELA。旧 utility labels、Decision dataset、模型、baseline 和成本—性能结果因此全部撤回；必须从 trajectory 开始按依赖顺序重新生成。已有 CEC2017 在线结果同样不能用于外部结论。

这些撤回产物已全部移出活动结果路径，本机仅在 `results/archive/` 下封存且不进入 Git。当前不存在可复用的正式模型、checkpoint 或论文结果表。

当前结果的完整口径见 [docs/30_results/phase1_current_results.md](docs/30_results/phase1_current_results.md)，跨对话状态见 [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md)。

## 冻结实验协议

- 训练：BBOB 10D / 20D / 40D。
- 验证：BBOB 10D / 20D / 40D，按 function family 与训练集隔离。
- 外部测试：CEC2017、CEC2022，工程问题作为后续外部验证。
- 算法池：DE、PSO、CMA-ES、SHADE。
- 主采样协议：`phase1_dynamic_budget_event_v1`；12 个必选预算里程碑为 `0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.60`，事件状态使每个 run 总计 12–18 行。
- 主 query 固定为 `descriptor_cheap`：16 个自定义低成本描述符，使用 `lhs_50d`，即 5% 总 FE。
- `pflacco_standard`（37 维，`lhs_50d`）与 `pflacco_broad`（52 维，`lhs_100d`，10% 总 FE）只用于预先定义的配置稳健性实验；不得根据 validation 结果改选主 query。
- Decision 输入仅来自 permutation-invariant 的算法无关搜索行为；function、dimension、algorithm、query feature 和优化器内部状态只能作为 metadata 或分层报告字段。
- Decision feature-group 正式消融固定为 `T0/B1/B2/B3=1/19/25/31`；T0 的数学输入是 `X={FE_ratio}`，实现为逐行等于 `FE_ratio` 的 `bf_fe_ratio`。`all_candidates` 仅是 B3 的兼容别名，不含 3 个诊断字段，也不作为第五个独立消融组。
- Decision Model 活动候选严格为 LDA、Logistic Regression、Ridge；主选择指标为 nested function-family OOF decision utility。AUROC、Average Precision、Spearman 为辅助指标，连续 Utility RMSE 只对 Ridge 定义。
- BBOB-validation 不参与 preprocessing、模型、候选选择或 threshold 拟合；部署阈值模式固定为 `oof_utility`。
- 主效用标签为 `u_query_lamT_1`；query 的 FE 成本通过减少后续优化预算体现，不重复按 FE 数量扣除。主时间成本为 Query 与 No-query 两条完整路径的有符号端到端 wall-clock 相对差，纯分析计算开销另存诊断字段。
- `benchmark_reference_value` 和所有 gap 字段只用于离线标签和最终评价，不进入 Behavior、Selection Reference 输入或 Decision X；使用已知最优值计算离线标签并不意味着在线优化器知道最优值。
- 第一篇论文主 probe/default 固定为训练集 SBS；SBS 只从 BBOB-train 的完整预算 `final_performance.parquet` 计算：先在每个 `problem_id × algorithm` 上对全部 seeds 的 `best_fitness` 取算术均值，再在每个 problem 内对算法排名，最后按算法跨 problem 平均排名；平均排名并列按冻结 portfolio 顺序 `de, pso, cmaes, shade` 决定。No-query 原生继续当前 SBS 的完整 checkpoint state。
- Query 后选择当前 prefix 时原生继续；选择其他算法时采用一次 checkpoint population transfer；query 采样点不并入后续优化 population。
- 多 prefix 行单独用于 cross-probe robustness、leave-one-probe-out 与 algorithm-agnostic 泛化，不进入主 Decision 数据。
- 标签显式保存 `selected_equals_default`、`selected_equals_prefix`、`handoff_required` 和 `skip_switches_from_prefix`，不再生成含义模糊的 selected-vs-default 字符串分层。
- `no_query_algorithm` 显式保存 No-query 分支算法并等于 `default_algorithm`；`handoff_type` 显式保存 Query-selected action 的 transition 类型并等于 `query_transition_mode`；`handoff_required` 等价于 `handoff_type == population_transfer_initialization`。
- 逐状态最小 action loss 称为 `best observed action`，只用于潜在性能差与 selector regret 诊断，不称为 oracle，也不进入 Decision 输入。

## 正式入口

正式配置只有：

- `configs/phase1_bbob_train.yaml`
- `configs/phase1_bbob_validation.yaml`
- `configs/phase1_cec2017_test.yaml`

主要命令可通过 `uv run <command> --help` 查看参数：

Selection Reference、Utility、Decision、baseline 与外部评价入口均显式区分 `query_id`。Decision 分析命令使用 `--query-id` 从 `results/decision/{query_id}/` 推导默认输入和输出；显式传入的 artifact 仍必须通过 query 协议核对。

- 数据采集：`phase1-plan-shards`、`phase1-collect-batch`、`phase1-check-trajectory-shards`、`optimizer-state-check`、`behavior-permutation-check`
- Query 与标签：`query-sample-batch`、`query-extract-cheap`、隔离的 `tools/pflacco_query/extract.py`、`query-consistency`、`selection-reference-check`、`selection-reference-evaluate-actions`、`selection-reference-build`、`utility-labels-generate-batch`
- Decision 数据与模型：`decision-materialize-training-data`、`decision-train-full`、`decision-check-model-protocol`、`decision-compare-feature-groups`
- 决策分析：`decision-handoff-learnability`、`decision-threshold-sweep`、`decision-compare-controller-baselines`、`decision-controller-cost-performance`
- 外部评价：`decision-external-test-predict`、`decision-online-controller-evaluate`、`decision-cec-online-score-distribution`

## 目录

```text
configs/               正式实验配置
benchmarks/            BBOB、CEC benchmark 适配
optimizers/            DE、PSO、CMA-ES、SHADE 的完整状态与原生 continuation
trajectory/            轨迹字段、记录与数据质量检查
behavior/              算法无关行为特征
landscape_queries/     三档 query 规格、LHS 样本、cheap 提取和一致性检查
tools/pflacco_query/   Python 3.11 + pflacco 1.2.2 隔离提取环境
selection_reference/   逐状态候选动作损失与离线算法选择参考
utility_labels/        共享前缀配对续跑与效用标签
decision/              Decision 数据、模型、baseline 与外部评价
experiments/           正式分片采集和配置检查
docs/                  主规格、协议、结果摘要、扩展和历史归档
results/               本机生成结果；默认不提交 Git
```

三档结果按 `query_id` 隔离：`results/selection_reference/{query_id}/`、`results/utility_labels/{query_id}/` 和 `results/decision/{query_id}/`。旧 `results/ela/` 以及缺少 `query_id`、`query_protocol`、`sample_design_id` 的标签或模型不属于活动读取契约。

## 文档优先级

1. `AGENTS.md`
2. `DEVELOPMENT_DECISIONS.md`
3. `docs/00_master/`
4. `docs/10_protocols/`
5. `docs/30_results/`

`docs/archive/` 只保存研究脉络，不是当前协议或运行入口。
