# Decision-before-Feature

本项目研究一个前置于特征计算的资源决策问题：在黑盒优化过程中，是否值得为当前搜索状态执行一项预先定义的 landscape-analysis query。项目不设计新的优化算法，而是用离线轨迹和监督学习估计固定 query 的状态依赖效用。

## 当前状态

截至 2026-08-11，四种优化器已改为完整状态推进：同一算法在 checkpoint 间保留内部状态与 RNG；只有 selector 确实切换算法时，才执行一次显式的 population transfer 初始化。可用 `uv run optimizer-state-check` 在真实 BBOB 问题上检查连续运行与多次 checkpoint 保存/恢复的一致性。

Behavior extractor 同时已改为 permutation-invariant 的种群集合统计：跨 checkpoint 的空间变化使用经验 Wasserstein、centroid shift 和协方差谱集中度，fitness 变化使用经验分位数分布；不再把 population 行号解释为跨代个体身份。

Selection Reference 已改为逐共享状态候选动作损失回归：每个 state 对 `continue_current` 和其余 portfolio algorithm 分别进行真实 continuation，`remaining_budget_ratio` 作为连续输入；不再按静态 problem label 和 nearest performance bucket 选择算法。训练行使用按 function family 的交叉拟合预测，验证与外部评价加载仅由 BBOB train 拟合的 selector model。

三档 query 的实现边界检查已通过：真实 BBOB 10D 上 cheap/standard 共享 `lhs_50d` 样本，broad 独立使用 `lhs_100d` 并由隔离的 pflacco 1.2.2 环境完整提取 52 列；三档均无额外函数评价。72 个正式 trajectory shards 尚未启动。

此前生成的 BBOB trajectory 使用了重建式 continuation，旧 behavior 含有依赖行号对应关系的字段，旧 landscape 表又把 16 个自定义描述符笼统称为 ELA。旧 utility labels、Decision dataset、模型、baseline 和成本—性能结果因此全部撤回；必须从 trajectory 开始按依赖顺序重新生成。已有 CEC2017 在线结果同样不能用于外部结论。

这些撤回产物已全部移出活动结果路径，本机仅在 `results/archive/` 下封存且不进入 Git。当前不存在可复用的正式模型、checkpoint 或论文结果表。

当前结果的完整口径见 [docs/30_results/phase1_current_results.md](docs/30_results/phase1_current_results.md)，跨对话状态见 [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md)。

## 冻结实验协议

- 训练：BBOB 10D / 20D / 40D。
- 验证：BBOB 10D / 20D / 40D，按 function family 与训练集隔离。
- 外部测试：CEC2017、CEC2022，工程问题作为后续外部验证。
- 算法池：DE、PSO、CMA-ES、SHADE。
- 主 checkpoint ratios：`0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60`。
- 主 query 固定为 `descriptor_cheap`：16 个自定义低成本描述符，使用 `lhs_50d`，即 5% 总 FE。
- `pflacco_standard`（37 维，`lhs_50d`）与 `pflacco_broad`（52 维，`lhs_100d`，10% 总 FE）只用于预先定义的配置稳健性实验；不得根据 validation 结果改选主 query。
- Decision 输入仅来自 permutation-invariant 的算法无关搜索行为；function、dimension、algorithm、query feature 和优化器内部状态只能作为 metadata 或分层报告字段。
- 主效用标签为 `u_query_lamT_1`；query 的 FE 成本通过减少后续优化预算体现，不重复扣除。
- 第一篇论文主 probe/default 固定为训练集 SBS；No-query 原生继续当前 SBS 的完整 checkpoint state。
- Query 后选择当前 prefix 时原生继续；选择其他算法时采用一次 checkpoint population transfer；query 采样点不并入后续优化 population。
- 多 prefix 行单独用于 cross-probe robustness、leave-one-probe-out 与 algorithm-agnostic 泛化，不进入主 Decision 数据。
- 标签显式保存 `selected_equals_default`、`selected_equals_prefix` 和 `skip_switches_from_prefix`；`same_algorithm` 仅是前者的兼容名称。
- 逐状态最小 action loss 称为 `best observed action`，只用于潜在性能差与 selector regret 诊断，不称为 oracle，也不进入 Decision 输入。

## 正式入口

正式配置只有：

- `configs/phase1_bbob_train.yaml`
- `configs/phase1_bbob_validation.yaml`
- `configs/phase1_cec2017_test.yaml`

主要命令可通过 `uv run <command> --help` 查看参数：

Selection Reference、Utility、Decision、baseline 与外部评价入口均显式区分 `query_id`。Decision 分析命令使用 `--query-id` 从 `results/decision/{query_id}/` 推导默认输入和输出；显式传入的 artifact 仍必须通过 query 协议核对。

- 数据采集：`phase1-plan-shards`、`phase1-collect-batch`、`phase1-check-trajectory-shards`、`optimizer-state-check`
- Query 与标签：`query-sample-batch`、`query-extract-cheap`、隔离的 `tools/pflacco_query/extract.py`、`query-consistency`、`selection-reference-check`、`selection-reference-evaluate-actions`、`selection-reference-build`、`utility-labels-generate-batch`
- Decision 数据与模型：`decision-materialize-training-data`、`decision-train-full`、`decision-compare-feature-groups`、`decision-model-benchmark`
- 决策分析：`decision-classifier-tune`、`decision-threshold-sweep`、`decision-compare-controller-baselines`、`decision-controller-cost-performance`
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
