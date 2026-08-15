# Decision-before-Feature Search Maturity 与 Query Utility 关系验证设计

> 唯一活动验证协议（2026-08-15）。本文中的 Search Maturity 是 Behavior 的确定性派生基函数，Query Utility 使用预指定 Stage-A 单次科学运行经配置截断后的 `log10_gap`，以及 Stage-B 三次 decision-state-to-terminal censored future-path wall-clock 中位数；raw observed median 只作诊断。本文不预设二者为单调、U 形或倒 U 关系，也不把描述性关联解释为因果关系。

## 1. 研究问题

本实验回答两个可证伪问题：

1. 在相同预算 milestones 上，B3 Behavior 是否比仅含 `FE_ratio` 的 T0 提供更高的 run-level first-trigger 联合效用？
2. 在 B2 已含原始 Behavior 的条件下，三项确定性 Maturity 变换是否改善固定模型家族的 first-trigger 联合效用？

第一个问题评价 Behavior 是否超出纯时间；第二个问题只评价预定义非线性基函数，不评价新信息来源或潜在状态。

## 2. 目标量

对 `skip`、`query_joint`、`query_matched_state_only`、`sampling_only_continue_current` 和 `behavior_only_full_budget` 五条状态后未来路径，先把非负 benchmark-reference raw gap 按 suite 配置截断：

\[
\ell_k=\log_{10}\!\left(\min(\max(g_k,10^{-12}),10^{20})\right),
\quad k\in\{skip,q,b\}.
\]

预指定 Stage-A 每条路径只运行一次，固定 terminal gap、`observed_first_hit_FE`、`target_hit_observed`、`target_hit_before_failure`、`path_completed`、`endpoint_success`、planned/effective FE 与科学失败状态；标准 ERT 使用 `target_hit_observed`。Stage-B 再从同一 complete optimizer state 和相同 RNG state 到 terminal 真实运行预定三次，但只用于计时。按 `cyclic_complete_path_v1` 循环移动 canonical path order，逐次保存 raw observed runtime、status、observed hit、path completion、endpoint success、effective FE 与失败字段。completed repetition 的 censored time 等于 raw，timed-out/failed repetition 为 `max(raw, role timeout)`，$T_k$ 是三次 censored time 的中位数；raw median 只作诊断。路径身份、completed replays 内部 endpoint 和 Stage-A→completed replay endpoint 一致性分别保存，Stage-B status instability 与跨阶段 completion instability 也分别保存。三次不得选择性补跑或覆盖 Stage-A 科学端点。共享 prefix 是 sunk cost，不进入 $T_k$；FE=0→terminal 政策 wall-clock 另作端点。

主联合效用、Behavior-only 效用与 query 操作性增量为：

\[
U_q^{joint}=(\ell_{skip}-\ell_q)
-\lambda_T\left(\log_{10}T_q-\log_{10}T_{skip}\right),
\]

\[
U_b=(\ell_{skip}-\ell_b)
-\lambda_T\left(\log_{10}T_b-\log_{10}T_{skip}\right),
\]

\[
I_q=(\ell_b-\ell_q)
-\lambda_T\left(\log_{10}T_q-\log_{10}T_b\right)
=U_q^{joint}-U_b.
\]

主情景固定 `lambda_time=1, lambda_memory=0`，表示 performance gap 与 runtime 的十进制数量级变化等权；`lambda_time={0,0.25,0.5,1,2}` 只作完整敏感性分析。所有旧的 raw-gap max-scale Utility、相对时间差 Utility 和一次计时标签均失效。

## 3. 数据与机会集合

主数据限于 `descriptor_cheap_invariant`、fold-specific SBS prefix、BBOB-train 与 `phase1_dynamic_budget_event_v1`。主 T0 对比只保留 12 个固定预算 milestones；事件机会依赖 Behavior，完整动态 schedule 上的 T0 仅作为 `schedule_conditioned_T0` sensitivity。

一条 run 的机会按整数 `FE` 和 `decision_opportunity_index` 排序。threshold 第一次越过后，该政策不再观察后续状态。模型选择、threshold、call rate、precision、policy Utility 与 capture 都按 run-level first-trigger 计算。

## 4. 冻结实验组

| 组 | 字段数 | 用途 |
|---|---:|---|
| T0 | 1 | milestone-only 纯时间参照 |
| B1 | 19 | core permutation-invariant Behavior |
| B2 | 25 | 加 longitudinal set dynamics |
| B2+Motion | 28 | B2 加三项 Motion |
| B2+Maturity | 28 | B2 加三项 Maturity |
| B3 | 31 | B2 同时加 Motion 与 Maturity |

主要预设 contrasts 为 `B1-T0`、`B2-B1`、`B2+Motion-B2`、`B2+Maturity-B2`、`B3-(B2+Motion)` 与 `B3-(B2+Maturity)`。所有组共享同一数据范围、完整嵌套 function 链和已选模型名；每组只用自身 train OOF score 冻结 first-trigger threshold，不重新选模型或按结果选择 feature group。

## 5. 完整嵌套要求

每个 outer holdout function 的证据必须由 outer-fit functions 独立生成。outer-fit 内每个 inner holdout 又必须只使用 inner-fit functions 计算 `SBS_inner`、拟合/cross-fit 两类 Selector、生成三类 Utility、拟合 preprocessing/Decision 并评价 inner holdout。inner holdout 不得通过预制 Selector 或 Utility label 进入其自身 threshold 拟合。

outer threshold 冻结后，使用 `SBS_outer` 和 outer-fit 全量组件只评价 outer holdout 一次。完整 BBOB-train OOF threshold 也执行同样的 fold-specific 上游链。BBOB-validation 与外部 suite 不参与公式、特征、模型、threshold 或分箱选择。

## 6. 主要与辅助分析

主要分析使用 function-balanced run-level first-trigger `u_query_joint_lamT_1`。效应按 run → static problem → fixed dimension stratum → function 聚合，function 为顶层统计单位。

辅助分析可以报告：

- 单个 Maturity 字段与 joint Utility 的 function-balanced Spearman；
- 预先固定分位数 bins 中的 Utility、gap 和 runtime 分布；
- 不同 function、dimension、query 配置的描述性异质性；
- Ridge 标准化系数或 LDA/Logistic 判别方向的 fold 稳定性。

这些分析不用于筛选特征、选择曲线形状或产生新的 threshold。若图形呈现非单调关系，只能描述实际估计与区间，不能事后命名“最佳成熟区间”。

## 7. 推断

对已见 BBOB-validation 使用 10,000 次条件配对 bootstrap：固定保留六个 validation functions、全部 dimensions 与 instances 1/2/3 对应的全部 static problems，只在每个固定 static problem 内配对重抽 optimizer seeds；每个抽中 seed/run 的完整有序 state 簇不可拆分。function-resampling 只作函数组成敏感性。上述边界只称项目内 operational tolerance，区间只描述该固定有限集，不能恢复独立确认性评价资格或推断到 transformed-instance 超总体。

RQ5 的六个预设 contrasts 改为估计性 family。BBOB-validation 对六个固定且已见 function effects 的 sign-flip 额外要求 signs 可交换；双侧 exact raw p 最小为 0.03125，Holm 后最小 adjusted p 为 0.1875，所以在 0.05 下不可能拒绝。逐 function effects、固定有限集效应与条件 CI 为主，adjusted p 只作假设敏感辅助，不作函数超总体推断。

## 8. 结果解释

支持 Maturity 的最低证据是相应 first-trigger Utility contrast 的效应、区间与外部方向；单个相关系数、训练拟合或一张平滑曲线都不充分。结果仅允许表述为“该预定义变换在所评估设置中提供/未提供预测增量”。

若已见 BBOB-validation 上 milestone-only B3 未超出 T0 或 dimension-stratified T0，则“Behavior 超出阶段及已知维度上下文的预测价值”在该固定有限集上未建立；即使超出，也仍只是开发期条件证据，train outer OOF 只作开发诊断。若 Maturity contrasts 未超出项目内 operational tolerance，则不得声称 Maturity 有独立贡献。若不同 suite 方向不一致，必须报告 representation/suite dependence。

## 9. 输出

结果表必须保存 `query_id`、outer/inner function-family split、feature group、模型、threshold mode、first-trigger state、joint/Behavior-only/increment Utility、matched-acquisition 三项增量、Stage-A 五条 `log10_gap`/observed-hit/target-hit/path-completion/endpoint-success/planned/effective FE/科学失败、Stage-B 五条 future paths 的三次 raw/censored wall-clock、逐次状态、三类一致性与两类 instability、FE=0 policy wall-clock、call/trigger/handoff、coverage 与失败字段。不得读取任何旧 `u_query_lamT_*` 或逐状态 policy 汇总产物。
