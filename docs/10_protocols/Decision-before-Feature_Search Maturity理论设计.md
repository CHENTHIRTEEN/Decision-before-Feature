# Decision-before-Feature Search Maturity 理论设计

> 唯一活动定义（2026-08-14）。Search Maturity 是由既有 Behavior 变量确定性计算的三维基函数组，不是独立观测、latent state、收敛判据、因果中介或由 Utility 反向定义的标签。旧“分析就绪状态”“有效中间状态”和预设倒 U 关系全部退出。

## 1. 研究用途与边界

本项目只检验一个有限问题：在固定数据、固定线性 Decision 候选、固定 first-trigger 评价和固定 query 下，预定义 Maturity 基函数是否改善对 Query/full-Selector 联合效用的预测。该问题不等于验证一个普适搜索阶段理论，也不支持把 Maturity 解释为 query 价值的原因。

Search Maturity 的上游仅含 query 前可得、算法无关且对 population 行排列不变的 Behavior。它不读取 query descriptors、function/dimension/algorithm identity、benchmark reference、known optimum、gap、Selector prediction、action loss、Utility 或未来轨迹。

## 2. 确定性定义

令冻结 Behavior 变换得到探索稳定化分量 (ES_t\in[0,1]) 与开发饱和分量 (XS_t\in[0,1])。两者只用于构造以下活动字段：

\[
M_t=ES_t(1-XS_t),
\]

\[
M_t^{linear}=\frac{ES_t+(1-XS_t)}{2},
\]

以及由既有探索与开发 summaries 计算的

\[
R_t^{EE}=\frac{E_t}{X_t+10^{-12}}.
\]

对应字段固定为：

```text
bf_search_maturity
bf_search_maturity_linear
bf_explore_exploit_ratio
```

这些量不增加原始信息；它们只把现有 Behavior 进行非线性重参数化。`bf_search_maturity` 不要求随 FE 单调，也不定义“早/中/晚成熟”类别。任何单调、U 形、倒 U 形或分段关系都必须由冻结评价结果描述，不能写入定义。

## 3. 与相邻概念的区别

| 概念 | 本项目中的含义 |
|---|---|
| `FE_ratio` | 已消耗预算比例；主纯时间 baseline 的唯一输入 |
| convergence | 特定 Behavior 指标描述的种群或最优值变化，不等于已知最优 gap |
| exploration/exploitation | 搜索群体的可观测行为 summaries |
| Search Maturity | 上述 Behavior 的三项确定性派生基函数 |
| Query Utility | 离线 outcome 与资源成本定义的监督目标，不参与 Maturity 构造 |

因此，Maturity 与 Utility 的相关性不能证明构念独立存在；Maturity 消融有增量也只能说明该预定义变换有助于当前候选模型。

## 4. 冻结比较

正式 feature groups 为：

| 组 | 内容 | 字段数 |
|---|---|---:|
| T0 | 仅 `bf_fe_ratio` | 1 |
| B1 | core permutation-invariant Behavior | 19 |
| B2 | B1 + longitudinal set dynamics | 25 |
| B2+Motion | B2 + 三项 set-motion 字段 | 28 |
| B2+Maturity | B2 + 三项 Maturity 字段 | 28 |
| B3 | B2 + Motion + Maturity | 31 |

Maturity 的预设比较是 `B2+Maturity - B2` 与 `B3 - (B2+Motion)`；Motion 的对应比较是 `B2+Motion - B2` 与 `B3 - (B2+Maturity)`。六组使用同一 Decision 数据、同名冻结模型家族、完整 outer/inner function 链、train-only preprocessing 和各组自身的 train-only first-trigger threshold。不得根据结果选取兄弟组、增加交互项或修改 Maturity 公式。

RQ2 的 Behavior-versus-time 主比较只在 12 个预算 milestones 上进行：`milestone_only_T0` 与 milestone-only B3 使用完全相同的行。事件机会由 Behavior 触发，完整动态 schedule 上的 T0 只能作为 `schedule_conditioned_T0` sensitivity。

## 5. 评价对象

主要证据是 BBOB-train outer-function OOF 的 run-level first-trigger `u_query_joint_lamT_1` 差，并按预设统计层级报告效应与区间。逐状态 Spearman、分箱曲线和单变量图只作描述性分析；它们不替代政策效用，也不用于改写公式、筛选 bins 或选择曲线形状。

对 Maturity 图形不画预期方向线。应展示样本量、function-balanced estimate 与不确定性，并分别报告 BBOB-train OOF、冻结 BBOB-validation 以及各外部 suite。外部结果不参与特征、模型或 threshold 选择。

## 6. 可支持与不可支持的结论

若两项预设 Maturity contrasts 的效应与区间支持改进，可写：

> 在所评估的固定线性模型、query、机会分布和 benchmark 上，预定义 Maturity 变换提高了 first-trigger 联合效用预测。

若结果不稳定或区间跨越项目内 operational tolerance，应写“预测增量未建立”或“结果依赖 suite/function”，不得以个别相关系数替代。第一篇论文不作确认性等价声明；普通或 simultaneous CI 只描述区间相对项目内 tolerance 的位置。

任何结果均不得写成：

- Maturity 是真实潜在搜索阶段；
- Maturity 导致 query 有价值；
- 存在普适最佳分析窗口；
- 倒 U 关系已由定义或理论保证；
- Maturity 提供了 Behavior 之外的新信息。

## 7. 数据泄漏与失败判据

Maturity 字段必须在每个 state 只由该时点及之前的完整 native-update 历史计算。窗口 anchor、实际窗口 FE/ratio 与 native-update 数只作 metadata，不进入 Decision X。若计算读取未来状态、known optimum、gap 或 query/outcome 字段，该行失效并须从 Behavior 依赖位置重生成。

如果 `B2+Maturity` 与 B2 的列差不严格等于三项活动 Maturity 字段，或 `B3` 不是 Motion 与 Maturity 的并集，则不得启动正式消融。若 BBOB train 上任一 Maturity 列全缺失，也不得通过固定常数填充后继续。

## 8. 结果保存

每组保存 outer fold、inner fold、模型名、threshold mode、first-trigger state、run-level Utility、Stage-A 最终 gap/`observed_first_hit_FE`/`target_hit_observed`/`path_completed`/`endpoint_success`/planned/effective FE/科学失败、Stage-B 三次 timing replay 的 wall-clock/status/effective FE/timeout/completion 及中位数/instability、独立 FE=0 policy wall-clock、call/trigger/handoff 状态与失败字段。Stage-B outcome 只作一致性与计时稳定性检查，不替换 Stage-A 科学端点；不得选择性重跑。结果表必须保留 `feature_group` 的六个活动名称；`all_candidates` 只可作为 B3 兼容别名，`primary_with_maturity` 只可解析为 B2+Maturity，不能生成新的比较组。
