# Decision-before-Feature 项目交接记录

本文档用于在不同对话之间持续同步项目状态。每次开始新对话时，先阅读：

1. `AGENTS.md`
2. 本文件
3. `DEVELOPMENT_DECISIONS.md`
4. 与当前任务直接相关的 `docs/` 研究设计文档

除非用户明确指定外，所有工作只允许使用当前项目目录内的文件。

---

## 当前任务状态

最近一次任务：

> 阅读 `AGENTS.md` 和 `docs/` 下所有研究设计 Markdown 文档，先不写代码，整理研究目标、冻结实验协议、软件模块、逻辑风险和开发路线图。

当前状态：

- 已完成研究设计阅读与口头总结。
- 已确认并持久化开发前约束冲突裁决：`DEVELOPMENT_DECISIONS.md`。
- 尚未开始实现代码。
- 尚未生成实验数据。
- 尚未建立项目源码目录。
- 尚未运行任何 benchmark 或优化器。

---

## 已完成内容

已阅读并整理：

- `AGENTS.md`
- `docs/` 下所有 Markdown 研究设计文档

已确认核心研究目标：

> 在黑盒优化中，Landscape Analysis / ELA 本身是否值得执行。

项目不是设计新优化算法，也不是提出新 ELA feature；核心是将 ELA 从默认固定步骤改成资源感知决策对象。

已确认主流程：

```text
Black-box problem
-> cheap optimization probe
-> algorithm-agnostic behavior
-> Search Maturity
-> ELA Utility
-> Decision-before-Feature
-> run or skip ELA
```

已确认冻结协议要点：

- 主学习范式：offline trajectory collection + supervised learning。
- 禁止在线控制器训练作为主实验。
- 训练：BBOB，10D / 20D / 40D。
- 验证：BBOB，50D。
- 测试：CEC2017 / CEC2022 / engineering problems。
- 划分：Function Family Split。
- 禁止随机 function instance split。
- 算法池：DE / PSO / CMA-ES / SHADE 或 L-SHADE。
- Decision 输入只能使用算法无关行为特征。
- Decision 输入禁止 Function ID、Dimension、Algorithm ID、算法内部参数、ELA feature。
- Checkpoint 使用 FE ratio，不使用固定 100 FE。
- Utility 标签离线生成。
- 必须包含 Never ELA、Always ELA、Random Analysis、Traditional AAS、SBS、VBS。

---

## 需要实现的软件模块

建议按以下模块边界开发：

- `configs/`：benchmark、optimizer、feature、model、experiment 配置。
- `benchmarks/`：BBOB、CEC2017、CEC2022、工程问题接口。
- `optimizers/`：DE、PSO、CMA-ES、SHADE/L-SHADE 统一运行接口。
- `trajectory/`：轨迹采集、checkpoint 记录、schema、Parquet 输出。
- `behavior/`：progress、diversity、exploration、exploitation、统一 extractor。
- `maturity/` 或归入 `behavior/`：ES、XS、Search Maturity 计算。
- `ela/`：ELA 采样、feature extraction、成本记录、selector 输入准备。
- `selection_reference/` 或 `portfolio_reference/`：算法性能矩阵、SBS/VBS、现实 selector。
- `utility_labels/`：共享前缀配对续跑、`P_skip/P_ELA`、Utility label 数据集生成。
- `decision/`：dataset、RF/XGBoost/LightGBM 训练、阈值选择、预测、解释分析。
- `experiments/`：RQ1-RQ5 与 decision overhead。
- `evaluation/`：指标、统计检验、分层 bootstrap、Pareto 分析、结果表格与图。
- `results/` 和 `logs/`：结构化实验输出与运行记录。

---

## 当前卡点

当前没有技术卡点，因为尚未开始代码实现。

但正式开发前需要用户或项目文档进一步冻结以下细节：

- BBOB function family 的具体 train/validation/test family 列表。
- CEC2017 / CEC2022 的具体维度、函数范围和预算。
- `FE_total`、`FE_prefix`、`FE_analysis`、`FE_optimization` 的精确定义。
- ELA 采样点是否允许复用到后续优化。
- Utility 中 FE 成本是否已经通过减少优化预算体现，避免重复扣除。
- Default optimizer 主设置使用 SBS 还是固定 CMA-ES/DE。
- `lambda` 主值和敏感性分析取值。
- Search Maturity 中 ES、XS 的具体公式、窗口和归一化。
- 方向熵的高维离散方案、零位移处理和平滑方式。

---

## 下一步计划

推荐开发路线：

1. 冻结配置与命名
   - 明确 benchmark split、维度、预算、checkpoint、seed、lambda。
   - 明确所有输出字段名，避免后期实验口径变化。

2. 建立基础目录和配置
   - 只创建必要模块骨架。
   - 不引入测试目录、pytest、JSON Schema 或代码生成器。

3. 实现 benchmark 与 optimizer 基础层
   - BBOB / CEC 接口。
   - DE / PSO / CMA-ES / SHADE-L-SHADE 统一接口。
   - 严格 FE budget 与 seed 管理。

4. 实现 trajectory collector
   - 按 FE ratio 记录 population、fitness、best fitness、metadata。
   - 保存 Parquet 或 JSON 结构化数据。

5. 实现 behavior extractor
   - FE ratio、improvement rate/frequency、diversity/change、directional entropy、distance decay、stagnation、convergence rate。
   - 检查尺度归一化、维度影响、算法人口规模影响。

6. 实现 ELA 路径与 selection reference
   - 建立算法性能矩阵。
   - 训练 ELA selector。
   - 报告 SBS、VBS 和现实 selector 表现。

7. 实现 Utility label generation
   - 基于共享 checkpoint 状态生成 Skip ELA 与 Run ELA 分支。
   - 保存 `P_skip`、`P_ELA`、cost、`U_ELA`。

8. 实现 Decision Model
   - 训练 RF / XGBoost / LightGBM 回归模型。
   - 做 threshold selection、OOD 测试、消融和解释分析。

9. 完成论文实验
   - RQ1 utility distribution。
   - RQ2 behavior prediction。
   - RQ3 end-to-end comparison。
   - RQ4 cross-benchmark OOD。
   - RQ5 ablation and explanation。
   - decision overhead and cost-performance Pareto。

---

## 绝对不要再踩的坑

- 不要访问当前项目目录之外的任何历史代码、旧实验、旧数据或旧文档。
- 不要把 ELA feature 输入 Decision Model。
- 不要把 Function ID、Dimension、Algorithm ID、算法内部参数输入 Decision Model。
- 不要随机划分 function instance；必须使用 Function Family Split。
- 不要用测试函数训练 Decision Model、selection reference 或 threshold。
- 不要把固定 100 FE 当 checkpoint；使用 FE ratio。
- 不要在线训练 controller 作为主实验。
- 不要把 `P_skip - P_ELA` 用在越大越好的指标上，除非先统一成越小越好的 loss/error。
- 不要重复扣除 ELA FE 成本：如果 ELA FE 已经从优化预算中扣除，Utility 中不能再扣同一笔 FE。
- 不要把 VBS 当现实可部署方法；它只能作为理论上界。
- 不要声称 Search Maturity 是已有成熟概念；它是本文提出的中间表征。
- 不要把 `M_t=ES_t(1-XS_t)` 写成已有文献公式；必须通过实验验证。
- 不要用 `p > 0.05` 证明两种策略等价；需要等价性分析或置信区间。
- 不要按 checkpoint 直接独立统计“大多数不需要 ELA”；同一轨迹 checkpoint 高度相关，必须分层统计。
- 不要说 CEC 天然就是 OOD；需要说明与 BBOB 的结构差异和分布差异。
- 不要引入文件哈希、checksum、manifest、receipt、source closure、quarantine、执行解锁机制等工程机制。
- 不要引入 pytest、测试目录、JSON Schema、schema registry 或测试依赖；如需验证，使用真实实验运行的数据质量检查和一致性检查。
- 不要使用 Python 内置 `hash()` 或 `hashlib` 派生实验随机数；随机数使用显式整数 seed 和 `numpy.random.SeedSequence`。
- 不要把普通结果检查命名为 audit / auditor / auditing；使用 evaluation、validation、consistency check、data quality check 等领域术语。
- 不要使用 `oracle/`、`tests/`、`schema_registry/`、`audit/` 作为开发主目录名；开发主线按 `DEVELOPMENT_DECISIONS.md` 执行。

---

## 下次对话启动建议

如果要继续开发，可以直接说：

```text
请阅读 AGENTS.md、PROJECT_HANDOFF.md 和 DEVELOPMENT_DECISIONS.md，然后从 Phase 1 开始建立配置和目录骨架。
```

如果要先冻结实验细节，可以直接说：

```text
请阅读 AGENTS.md、PROJECT_HANDOFF.md 和 DEVELOPMENT_DECISIONS.md，然后帮我冻结 benchmark split、FE 预算、lambda 和 Search Maturity 公式。
```

如果要继续科研设计，可以直接说：

```text
请阅读 AGENTS.md、PROJECT_HANDOFF.md、DEVELOPMENT_DECISIONS.md 和相关 docs，继续检查 Decision-before-Feature 的实验逻辑风险。
```
