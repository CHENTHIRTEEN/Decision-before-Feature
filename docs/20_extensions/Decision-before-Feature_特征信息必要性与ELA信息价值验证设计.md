# Decision-before-Feature 特征信息必要性与表示依赖性验证设计

## 1. 文档定位

本文档讨论固定 landscape representation 的信息增量与表示依赖性。旧版文档使用 “Full ELA / Compact ELA” 二分法，但当前 16 维描述符既不是完整 ELA，也不是 pflacco 的完整 feature sets；该二分法已由三档 Landscape Query 协议取代。

活动配置以 `docs/10_protocols/Decision-before-Feature_Landscape_Query三档配置与数据契约.md` 为准：

- `descriptor_cheap`：16 个自定义低成本描述符，`lhs_50d`，5% FE，第一篇论文唯一主 query；
- `pflacco_standard`：37 个预定义 pflacco 1.2.2 特征，`lhs_50d`，5% FE；
- `pflacco_broad`：52 个预定义 pflacco 1.2.2 特征，`lhs_100d`，10% FE。

standard 与 broad 只用于预先定义的配置稳健性实验，不得根据 validation 结果改选主 query。NeurELA、Deep-ELA 及其他学习式或动态表示不在本轮实现范围内。

---

## 2. 两个不同问题

### 2.1 固定 query 是否值得执行

第一篇论文的主问题是：

> 算法无关搜索行为能否预测调用固定 `descriptor_cheap` query 的状态依赖效用？

比较 No-query 与 Run Query，并计算：

$$
U_{query}=(P_{skip}-p_{query})-\lambda_T C_T-\lambda_M C_M.
$$

其中 `FE_query` 已通过减少 Run Query 的后续优化预算进入 $p_{query}$，不得重复按 FE 数量扣除。主 $C_T$ 比较 Query 与 No-query 的完整端到端 wall-clock：`runtime_query` 包含采样、样本目标评价与特征计算时间，同时两条路径的后续优化运行时间相减；纯分析计算开销另作诊断。

### 2.2 结论是否依赖 representation

配置稳健性问题是：

> 在不改变 Decision 输入边界、function-family split、`phase1_dynamic_budget_event_v1` 采样参数、算法池和等总 FE 原则时，效用与调用决策是否随预定义 query 配置改变？

分别报告：

$$
U_{cheap},\quad U_{standard},\quad U_{broad}.
$$

三档必须独立训练 Selector、Utility target 和 Decision Model。`query_id` 只用于数据隔离和协议检查，不作为模型输入；本轮不训练动态 query-type selector。

---

## 3. 共享前缀配对续跑协议

每个状态从同一完整 optimizer checkpoint state 构造两条路径：

1. No-query：按主协议原生继续训练集 SBS/default；
2. Run Query：付出固定 query FE，使用该 query 的 Selector 选择动作，再以减少后的后续优化预算运行。

若 query 后仍选择 prefix algorithm，必须恢复完整内部状态原生继续；若切换算法，只转移 population、fitness 与 best-so-far position，并记录 population-transfer initialization。Query 的 LHS 样本不得并入优化 population。

cheap 与 standard 共享完全相同的 `lhs_50d` 样本和 5% FE action-loss 表；broad 使用独立 `lhs_100d` 样本与 10% FE action-loss 表。不同 `sample_design_id` 或 `FE_query` 的 action losses 不得混用。

---

## 4. 信息增量与成本必须分开

以下两种现象不能混写：

### 情况 A：性能差为正但净效用不大于零

$$
P_{skip}-p_{query}>0,\qquad U_{query}\leq0.
$$

这表示 selector 路径的最终性能差不足以覆盖预定义的非 FE 成本。

### 情况 B：query feature 对动作损失预测的增量有限

在 Selection Reference 内比较使用同一拆分与同一 action-loss 表的模型：

- behavior-only；
- behavior + 该 query 的固定 feature columns。

如果忽略 query runtime 后，selector regret 或 action-loss 回归性能改善仍很小，只能说明该固定 representation 对当前动作选择任务的增量有限，不能外推为所有 ELA 信息无用。

Decision Model 本身仍只接收算法无关 behavior；上述 feature 增量比较属于下游 Selector 的分析，不得把 query feature 泄漏到 Decision 输入。

---

## 5. 统计与报告单位

同一 trajectory 的 checkpoint 高度相关，不能把所有行当作独立样本。分层单位为：

```text
function family -> function instance -> dimension -> optimizer seed -> sampled state (integer FE)
```

每档 query 至少报告：

- `FE_query`、`runtime_query` 和 query failure rate；
- selector regret 与 action-loss regression performance；
- Utility 分布、调用率、utility capture 和最终优化性能；
- function-family 层面的配对效应量与区间；
- Never Query、Always Query、Random Analysis、Traditional AAS、SBS 与 VBS。

不能用 `p > 0.05` 证明等价。如果提出实质等价，必须预先定义等价界，并使用区间或等价性检验。

---

## 6. 失败与缺失值解释

BBOB train/validation 不允许 group-level extraction failure。单个未定义值保存为 null，只使用 BBOB-train median imputation；任何整列缺失都阻止模型拟合。

外部 benchmark 可以记录 query failure，并使用冻结的 BBOB-train median fallback。但只要存在 group-level failure，就必须单独报告，不能形成该 query 的无条件跨 benchmark 泛化结论。

---

## 7. 允许的结论

若三档效应方向和主要结论一致，只能写：

> 结论在三个预定义 landscape-query 配置上具有稳健性。

不得写成：

- 对完整 ELA 成立；
- 对全部 pflacco feature groups 成立；
- 对 NeurELA、Deep-ELA 或任意 landscape representation 成立。

若三档结论不一致，应报告 representation dependence，并解释差异是否与 feature group、样本量、失败率或计算成本有关；不得隐藏结果、事后改选主 query 或重定义配置。

---

## 8. 后续扩展边界

特征组消融、Progressive ELA、NeurELA、Deep-ELA 与动态 representation selection 均可成为后续独立研究问题，但不进入第一篇论文主实验。本轮不根据 SHAP 排名删减特征，不实现 learned query type selection，也不把三档配置包装成完整 landscape analysis 的覆盖性比较。
