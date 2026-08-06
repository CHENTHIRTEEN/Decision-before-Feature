# Decision-before-Feature 文档索引

本文档说明 `docs/` 的当前有效结构。

整理原则：

- 不删除历史内容；
- 当前主协议优先；
- 早期草案只保留研究脉络；
- 与 `DEVELOPMENT_DECISIONS.md` 冲突时，以 `DEVELOPMENT_DECISIONS.md` 为准。

---

# 当前 phase1 启动状态

当前已完成 min_support 链路验证和问题归因，准备进入正式 BBOB phase1 完整数据集实验。

关键持久化结论：

- min_support 问题归因矩阵：`results/decision/min_support/problem_attribution_matrix.md`
- 正式 phase1 checkpoint ratios 已根据 min_support 结果冻结为：

```text
0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60
```

裁决理由：

- min_support 中 `changed_algorithm` 且 `U_ELA>0` 的主要机会集中在 `FE_ratio=0.30-0.55`。
- `0.20/0.25/0.28` 用于保留 transition 前后参照。
- `0.60` 用于保留机会区之后的衰减参照。
- very early 和 late endpoint 采样不进入正式 phase1 主频率；如需研究，应另设扩展实验。

正式 phase1 需要同步处理的 min_support 问题：

- label coverage：补足 train 中 `changed_algorithm` 且 `U_ELA>0` 的覆盖。
- algorithm-behavior confounding：按 `prefix_algorithm` 做分层诊断，但不作为 Decision 输入。
- threshold transfer/calibration：避免使用 held-out family 作为可部署 threshold key，优先使用可迁移 stage / behavior-bucket / score-quantile 校准。
- downstream ELA-based selection pipeline quality：作为固定下游组件限制报告，不作为本文方法贡献。
- same_algorithm random continuation noise：保留为共享前缀续跑随机差异参照。
- behavior feature insufficiency：优先诊断算法无关 population/fitness behavior feature 候选。

---

# 00_master

当前研究主线与论文写作入口。

- `00_master/Decision-before-Feature Master Research Specification.md`
  - 最高层研究规格。
  - 定义研究问题、主流程、数据 split、baseline、评价协议。
- `00_master/Decision-before-Feature_数学定义与方法章节.md`
  - 数学定义和方法章节草案。
  - 定义 Analysis Selection、行为状态、离线效用标签和 Decision Model。
- `00_master/Decision-before-Feature_科学结论公式与引用关系核对.md`
  - 论文结论、公式和文献依据的核对表。
  - 用于区分文献直接支持、方法学支持、思想来源和项目原创假设。
- `00_master/Decision-before-Feature 完整论文结构与Contribution设计.md`
  - 论文结构、contribution 和章节写作蓝图。

---

# 10_protocols

当前可执行协议和模块级设计。

- `10_protocols/Decision-before-Feature Behavior Feature Taxonomy与指标选择协议.md`
  - 行为特征定义、FE-ratio窗口、prior behavior metrics关系。
- `10_protocols/Decision-before-Feature_实验数据生成与算法无关行为建模设计.md`
  - trajectory 数据生成、记录字段和算法无关输入约束。
- `10_protocols/Decision-before-Feature 完整实验Pipeline与代码架构设计.md`
  - 完整实验工程流程。
  - 注意：验证方式采用真实小规模实验、数据质量检查和一致性检查，不使用测试框架替代真实实验。
- `10_protocols/Decision-before-Feature_Offline Utility Label构建协议.md`
  - 离线效用标签生成协议。
  - 开发口径使用 `utility_labels`，避免把普通标签生成写成独立算法贡献。
- `10_protocols/Decision-before-Feature Algorithm Portfolio与Selection Reference设计.md`
  - Algorithm portfolio 和 selection reference 设计。
  - `Selection Reference` 只用于离线标签生成，不作为部署阶段组件。
- `10_protocols/Decision-before-Feature Decision Model设计与训练协议.md`
  - Decision Model 输入、输出、训练形式、模型选择和消融。
- `10_protocols/Decision-before-Feature Baseline与公平比较协议.md`
  - Never ELA、Always ELA、Random Analysis、Traditional AAS、SBS、VBS 和公平预算协议。
- `10_protocols/Decision-before-Feature_Search Maturity理论设计.md`
  - Search Maturity 理论构造。
- `10_protocols/Decision-before-Feature Search Maturity与ELA Utility关系验证设计.md`
  - Search Maturity 与 ELA Utility 的关系验证实验。
- `10_protocols/Decision-before-Feature_Decision_Model计算成本与资源开销分析设计.md`
  - Decision Model 推理成本、资源节省和开销分析。

---

# 20_extensions

扩展问题和非主协议笔记。

- `20_extensions/min_support_late_stage_coverage_extension_plan.md`
  - min_support late-stage coverage extension 计划。
  - 当前完整问题归因矩阵保存在 `results/decision/min_support/problem_attribution_matrix.md`。
- `20_extensions/Decision-before-Feature_特征信息必要性与ELA信息价值验证设计.md`
  - Full ELA、Compact ELA、特征信息必要性等扩展问题。
- `20_extensions/Decision-before-Feature_维度与泛化实验设计.md`
  - 维度泛化扩展笔记。
  - 当前主 BBOB train / validation 不包含 50D；50D / 100D 只能作为另设扩展实验。

---

# 90_literature

本项目已收集的本地参考文献 PDF。

- `90_literature/How MHs exploit.pdf`
- `90_literature/HaywardL25TEC MHs similarity Behavior analysis.pdf`

---

# archive

早期研究方案和对话整理，仅保留历史脉络。

这些文档不作为当前实验协议、代码开发协议或论文主线的直接来源。

- `archive/Decision-before-Feature_研究方案.md`
- `archive/Decision-before-Feature_实验方案.md`
- `archive/Decision-before-Feature_对话整理与研究脉络.md`
- `archive/Decision-before-Feature_实验协议与代码设计.md`

---

# 当前已处理的冲突

- BBOB validation：
  - 主协议使用 BBOB function family validation split，维度为 10D / 20D / 40D。
  - BBOB 50D 不进入当前主协议。
- 验证方式：
  - 不建立 `tests/` 目录。
  - 不引入 `pytest`。
  - 使用真实小规模 benchmark 运行、数据质量检查和一致性检查。
- 术语：
  - 开发和协议文档使用 `Offline Utility Label` / `utility_labels`。
  - Algorithm selection 的离线参考使用 `Selection Reference`。
  - 共享前缀上的两分支续跑称为“共享前缀配对续跑”。
  - 普通公式、引用和结果检查称为“核对”或“检查”。
