# Decision-before-Feature 文档索引

## 当前入口

当前文档按以下顺序使用：

1. `../AGENTS.md`：最高优先级开发与实验约束。
2. `../DEVELOPMENT_DECISIONS.md`：已冻结的可执行协议。
3. `00_master/`：研究问题、数学定义、论文结构和引用边界。
4. `10_protocols/`：数据、标签、模型、baseline 和成本评价协议。
5. `30_results/`：只记录已历史旧结果的影响范围；完整两阶段链重生成后再新增正式结果。
6. `20_extensions/`：尚未进入主实验的扩展问题。
7. `archive/`：已被当前规格取代的历史材料。

项目整体状态请先读 `../README.md` 和 `../PROJECT_HANDOFF.md`。

## 当前实现与结果

截至 2026-08-14：

- 四种优化器的完整状态 continuation 与真实 BBOB 一致性检查已实现；
- 逐共享状态 action-loss Selection Reference、连续 remaining-budget regression 与部分一致性检查已实现；
- 旧 BBOB trajectory、utility labels、259,200 行 Decision dataset 和模型结果受重建式 continuation 影响，均不得作为正式证据；
- 活动 Utility 已改为截断 `log10_gap` 差减完整路径 `log10` runtime ratio，并分开 joint、Behavior-only、operational increment 与 query-feature predictive increment；
- 当前必须从 trajectory 开始重生成一次 state-action matrices，再由 fold-specific Selector 选择动作并三次重跑 selected complete paths；现有 action-component 计时不能代替该阶段；
- 旧 CEC2017 在线结果作废，完整外部评价须在内部链路重生成后执行。

详细数值见：

- `30_results/phase1_current_results.md`

## 00_master

- `Decision-before-Feature Master Research Specification.md`：最高层研究规格。
- `Decision-before-Feature_数学定义与方法章节.md`：方法与数学定义。
- `Decision-before-Feature_科学结论公式与引用关系核对.md`：论断、公式和引用边界。
- `Decision-before-Feature 完整论文结构与Contribution设计.md`：论文结构和 contribution 蓝图。

## 10_protocols

- `Decision-before-Feature_Landscape_Query三档配置与数据契约.md`（三档 query、共享样本、pflacco 隔离、活动字段与失败处理）
- `Decision-before-Feature Behavior Feature Taxonomy与指标选择协议.md`
- `Decision-before-Feature_实验数据生成与算法无关行为建模设计.md`
- `Decision-before-Feature 完整实验Pipeline与代码架构设计.md`
- `Decision-before-Feature_Offline Utility Label构建协议.md`
- `Decision-before-Feature_phase1_refined_sampling_指标口径冻结.md`
- `Decision-before-Feature_phase1_utility_label_column_spec.md`
- `Decision-before-Feature Algorithm Portfolio与Selection Reference设计.md`
- `Decision-before-Feature_逐状态动作损失Selection Reference修订.md`
- `Decision-before-Feature Decision Model设计与训练协议.md`
- `Decision-before-Feature Baseline与公平比较协议.md`
- `Decision-before-Feature_Search Maturity理论设计.md`
- `Decision-before-Feature Search Maturity与ELA Utility关系验证设计.md`
- `Decision-before-Feature_Decision_Model计算成本与资源开销分析设计.md`

正式 phase1 的关键冻结口径：

```text
BBOB train/validation dimensions: 10, 20, 40
sampling protocol: phase1_dynamic_budget_event_v1
monitor grid: 0.20--0.60, step 0.01
budget milestones: 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.60
states per run: 12--18
final performance: separate final_performance.parquet, one FE=FE_total row per problem_id/algorithm/seed
behavior columns: 34 outputs / 31 formal inputs / 3 diagnostic-only
feature groups: T0/B1/B2/B2+Motion/B2+Maturity/B3 = 1/19/25/28/28/31
primary query: descriptor_cheap_invariant = 14 columns / lhs_50d / 5% FE
robustness queries: pflacco_standard_invariant = 37; pflacco_broad_invariant = 52
primary target: G_FE
operational increment: query_operational_increment
predictive diagnostic: query_feature_predictive_increment_log10_gap
algorithm switch: population transfer
Selection Reference: statewise action-loss regression with continuous remaining budget
query sample reuse as optimizer population: false
Decision X: algorithm-agnostic behavior only
policy unit: run-level first trigger, at most once
```

SBS 只从相应 fit functions 的完整预算终值表计算：raw gap 按配置截断并取 `log10_gap`，再按 run → static problem（function × dimension × instance）→ fixed dimension stratum → function 等权聚合，选择均值最低算法；并列按 `de,pso,cmaes,shade`。outer/inner/full-train 分别重算。`0.20–0.60` decision trajectory 不提供 SBS 终值；`all_candidates` 只是 B3 别名，`primary_with_maturity` 只对应 B2+Maturity。

统一 median/IQR preprocessing 后恒为 0/1 的 `descriptor_y_median`、`descriptor_y_iqr` 已从主 query 活动 whitelist 删除，因此 cheap 从 16 列改为 14 列；query ID、`lhs_50d`、5% FE 和既有 action-loss 设计不变。

正式运行前仍有 blockers：replay planner 已有枚举能力，但 offline decision-state-to-terminal runner、物化实测 plan 与 Stage-A Skip 复用尚未闭合；资源/排期未确认；CEC2017 F2/F30 函数集口径需核对；CEC2022 与工程问题的 suite endpoint 和 constraint rule 尚未冻结。

## 20_extensions

当前只保留仍可能进入后续正式实验的扩展：

- `Decision-before-Feature_特征信息必要性与ELA信息价值验证设计.md`
- `Decision-before-Feature_维度与泛化实验设计.md`

preliminary/min_support 的扩展计划和归因记录已移动到 `archive/min_support/`。

## 30_results

- `phase1_current_results.md`：已历史旧 BBOB 结果的范围、不能迁移的结论及当前正式证据缺口；不是当前正式结果页。

## 90_literature

项目内已收集且与当前研究直接相关的本地参考文献。这里只保存文献 PDF，不保存实验数据或其他项目材料。

- `How MHs exploit.pdf`
- `HaywardL25TEC MHs similarity Behavior analysis.pdf`
- `Per-run Algorithm Selection with Warm-starting Using Trajectory-Based Features.pdf`
- `Trajectory-based Algorithm Selection with Warm-starting.pdf`

论文主架构图保存在 `assets/figures/decision-before-feature-architecture.png`。

## archive

只保存被当前规格取代的草案和 preliminary 研究脉络。不得从该目录提取当前运行参数、结果数值或开发入口。

本次整理范围和恢复位置见 `archive/cleanup_20260811.md`。
