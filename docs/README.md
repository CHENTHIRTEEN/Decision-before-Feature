# Decision-before-Feature 文档索引

## 当前入口

当前文档按以下顺序使用：

1. `../AGENTS.md`：最高优先级开发与实验约束。
2. `../DEVELOPMENT_DECISIONS.md`：已冻结的可执行协议。
3. `00_master/`：研究问题、数学定义、论文结构和引用边界。
4. `10_protocols/`：数据、标签、模型、baseline 和成本评价协议。
5. `30_results/`：已撤回旧结果的影响范围；完整状态重生成后再更新正式结果。
6. `20_extensions/`：尚未进入主实验的扩展问题。
7. `archive/`：已被当前规格取代的历史材料。

项目整体状态请先读 `../README.md` 和 `../PROJECT_HANDOFF.md`。

## 当前实现与结果

截至 2026-08-11：

- 四种优化器的完整状态 continuation 与真实 BBOB 一致性检查已实现；
- 逐共享状态 action-loss Selection Reference、连续 remaining-budget regression 与对应一致性检查已实现；
- 旧 BBOB trajectory、utility labels、259,200 行 Decision dataset 和模型结果受重建式 continuation 影响，均不得作为正式证据；
- 当前必须从 trajectory 开始重生成，再重新选择 Decision Model；
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
- `Decision-before-Feature Search Maturity与Query Utility关系验证设计.md`
- `Decision-before-Feature_Decision_Model计算成本与资源开销分析设计.md`

正式 phase1 的关键冻结口径：

```text
BBOB train/validation dimensions: 10, 20, 40
checkpoint ratios: 0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60
target: u_query_lamT_1
algorithm switch: population transfer
Selection Reference: statewise action-loss regression with continuous remaining budget
query sample reuse as optimizer population: false
Decision X: algorithm-agnostic behavior only
```

## 20_extensions

当前只保留仍可能进入后续正式实验的扩展：

- `Decision-before-Feature_特征信息必要性与ELA信息价值验证设计.md`
- `Decision-before-Feature_维度与泛化实验设计.md`

preliminary/min_support 的扩展计划和归因记录已移动到 `archive/min_support/`。

## 30_results

- `phase1_current_results.md`：当前正式 BBOB 内部结果、模型选择、结果边界和外部评价状态。

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
