# Decision-before-Feature Algorithm Portfolio 与 Selection Reference 设计

> 唯一活动协议（2026-08-14）。Selection Reference 是固定下游性能回归组件，不是本文提出的新算法。旧 problem-level classifier、remaining-budget bucket 与从重建 state 续跑的标签全部退出。

## 1. Portfolio

活动算法池固定为：

```text
DE
PSO
CMA-ES
SHADE
```

本轮不得增加、删除或以 L-SHADE 替换 SHADE。四种算法使用项目内统一 complete-state optimizer interface。算法参数和内部状态不进入 Decision 或 Selector features；algorithm identity 只定义动作列和 metadata。

## 2. SBS、prefix 与 VBS

SBS 只由相应 fit functions 的完整预算 `final_performance.parquet` 计算：对每个算法按 run → static problem（function × dimension × instance）→ fixed dimension stratum → function 等权聚合配置截断后的 `log10_gap`，选取均值最低者；并列按 `de,pso,cmaes,shade`。该定义与主性能端点和 function 顶层权重一致，不使用平均 rank 丢弃效应量。

外层评价用 `SBS_outer`，内层评价用 `SBS_inner`，二者只由各自 fit functions 计算。完整 BBOB-train SBS 仅用于最终重拟合和已见 BBOB-validation/external deployment；BBOB-validation 不再具有未查看确认集资格。主行 `prefix_algorithm == default_algorithm == fold-specific SBS`。

静态 VBS 是不可部署的 problem-level hindsight reference。对每个 `problem = function × instance × dimension`，先对四算法各自的完整预算 clipped `log10_gap` 跨 optimizer seeds 取算术均值，选择均值最低算法（并列按 `de,pso,cmaes,shade`），再用该算法的逐 seed paired outcomes 进入后续汇总。不得逐 seed 选择最小算法。共享 state 上已运行 continuation actions 的最小 loss 称 `best observed action`；它不是 VBS，也不称 oracle。

## 3. 共享 state 与动作集合

每个 state 保存 population、fitness、best-so-far、generation/native-update index、optimizer-specific dynamics 与 RNG。动作集合固定为：

\[
\mathcal A(s_t)=\{\texttt{continue_current}\}\cup
(\{DE,PSO,CMA\mbox{-}ES,SHADE\}\setminus\{a_t\}).
\]

因此始终是四个互不重复动作：

- `continue_current` 保留完整当前算法 state 与 RNG，原生推进；
- 其余三个动作转移 checkpoint population、fitness 与 best-so-far，初始化新算法内部状态一次；
- 跨算法 transition 记为 `population_transfer_initialization`；
- query sample 不并入 optimizer population。

不得把“当前算法动作”和 `continue_current` 同时列为两个动作。

## 4. 两套预算矩阵

令总预算为 (B)、prefix 已用 FE 为 (e_t)、query FE 为 (FE_q)。每个共享 state 生成：

1. Query-adjusted matrix：四动作均使用 (B-e_t-FE_q)，终端 raw loss 为 (L_q(s_t,a))；
2. Behavior-only full-budget matrix：四动作均使用 (B-e_t)，终端 raw loss 为 (L_b(s_t,a))。

full-budget `continue_current` 与 Skip 语义相同，只计算一次。cheap/standard 共用 5% sample design 下语义相同的 action outcomes；broad 使用独立 10% outcomes。预算不同的动作结果不能复用。

动作矩阵的 Selector target 使用 continuation outcomes。主 operational Query path 虽不把 query sample 插入 optimizer population，但 terminal best、`observed_first_hit_FE` 与标准 ERT 必须计入 query sample 的真实 objective evaluations；`target_hit_observed`、`path_completed` 与 `endpoint_success` 分列。因此另保存 Query continuation-only outcome 与 query-sample-best contribution，不能把两者混为 Selector 改进。

所有动作从同一复制 state 和冻结的 action RNG stream 开始。失败动作不删除，按 suite 配置保留有限 target 与失败状态；缺失 pair 另执行运行前冻结的双向极端 sensitivity，已保留的科学 path failure 不当作 missing pair。

## 5. Selector target 与模型

每套矩阵使用 suite 预先固定的 $g_{\min},g_{\max}$ 变换 continuation-only raw observed action loss。令 $a_c$ 为 `continue_current`：

\[
Y_a=\log_{10}(\operatorname{clip}(L_a,g_{\min},g_{\max}))
-\log_{10}(\operatorname{clip}(L_{a_c},g_{\min},g_{\max})).
\]

固定模型为多输出 `RandomForestRegressor`，预测四个动作的 $Y_a$，部署选择预测最小动作；`continue_current` 的 target 恒为 0。主产物保存 `selector_target_transform=clipped_log10_gap_advantage_vs_continue_current`、raw action range、near ties 与 raw regret sensitivity。旧 `statewise_minmax_observed_action_loss` 只作预设 target sensitivity，不生成主 selected action 或 Utility。Utility 读取真实 selected-path endpoint，不直接使用 Selector target。

Query Selector 输入：

```text
B3 Behavior
当前 query descriptors
query-adjusted remaining_budget_ratio
```

Behavior-only full-budget Selector 输入：

```text
B3 Behavior
full-budget remaining_budget_ratio
```

function ID、dimension、prefix algorithm ID、seed、known optimum/gap、action losses 与 optimizer internal state 不进入 Selector features。

## 6. Query-feature predictive diagnostic

除主 Query Selector 与 full-budget Behavior-only Selector 外，另拟合第三个 `query_adjusted_state_only_selector`：输入 B3 Behavior 与 query-adjusted remaining ratio，不含 query descriptors；target 和动作矩阵与 Query Selector 完全相同。它不是主 full-budget Behavior-only Selector。两个 query-adjusted Selectors 都用 OOF predictions 在同一四动作 outcomes 上选择动作，不新增 action losses，定义：

\[
\Delta_{qfeat}^{pred}=\ell_{q,state\mbox{-}only}-\ell_{q,full},
\]

字段名为：

```text
query_feature_predictive_increment_log10_gap
```

正值表示在相同 query-adjusted budget 和 observed action outcomes 下，加入 query features 的 OOF Selector 选择获得更低的 `log10_gap`。该诊断不需要新增 action losses，只比较同一矩阵中两个 OOF Selector 的所选 outcome。它只称“query features 的 OOF 边际预测贡献”，不是纯信息价值、因果效应或主策略指标。

主 `query_operational_increment_lamT_*` 则比较 Query 路径与 full-budget Behavior-only 路径，包含 query FE、decision-state-to-terminal future-path 时间与不同剩余预算；共享 prefix 视为 sunk cost，FE=0 policy wall-clock 另报。二者不得混称。

## 7. Fold-specific 拟合

Decision outer holdout 的所有 Selector 必须只由 outer-fit functions 拟合；outer-fit Decision labels 使用 outer-fit 内 cross-fitted Selector predictions。每个 Decision inner holdout 还必须只读 inner-fit functions：重算 `SBS_inner`，在 inner-fit 内 cross-fit Selector 生成 Decision fit labels，再用 inner-fit 全量 Selector 生成 inner-holdout Utility。

完整 BBOB-train 的部署 threshold 和 Random calibration 同样使用端到端 function-OOF 上游链。不得由 full-train Selector 先生成整表 labels，再仅对 Decision 分 fold。

所有 Selector fit 在各 fit fold 内使用 function → fixed dimension stratum → static problem → optimizer run 等权，再把 run 权重均分给其 states；权重缩放到平均 row weight 为 1。旧 unweighted state-row fit 只作敏感性。当前 `selection_reference.model` 尚未把该权重接到 RandomForest pipeline，因此在实现前仍是正式运行 blocker。

## 8. 诊断与字段

Selection Reference、Utility、Decision dataset 与在线输出必须保存：

```text
selected_equals_default
selected_equals_prefix
handoff_required
handoff_type
skip_switches_from_prefix
no_query_algorithm
```

并满足 `handoff_required = not selected_equals_prefix = (handoff_type == population_transfer_initialization)`。不得生成 `label_source` 或用模糊字符串代替。

Selector 评价至少报告 OOF selected observed `log10_gap`、best-observed regret、动作一致率、raw action range、near-tie rate、failure/coverage 和 query-feature predictive diagnostic。选择一致率不能替代实际 outcome。

## 9. 失败与运行条件

BBOB train/validation 与 CEC2017 固定 `failure_loss_cap=1e20`、取对数前 raw-gap floor/cap `1e-12/1e20`、`success_gap_target=1e-8`、单 state-action path timeout `3600 s`，并逐 objective evaluation 记录 `observed_first_hit_FE`。`target_hit_observed := observed_first_hit_FE != null`；`target_hit_before_failure := target_hit_observed and not path_completed`；`endpoint_success := target_hit_observed and path_completed`。timeout/failed path 的 final-gap endpoint按失败 cap 保留，但失败前命中的 first hit 不抹除；标准 ERT 使用 `target_hit_observed`，未命中项计完整 planned budget。CEC2022/工程问题必须先冻结同类字段与 constraint rule。

缺失共享 state、动作矩阵不完整、预算混用、fold source 不明或 population transfer 字段不一致时，相应 Selector/Utility 行失效并须从 action-loss 依赖位置重生成。

## 10. 结果边界

Selection Reference 只为固定 query downstream path 提供现实动作选择。它的性能不构成新算法贡献。`best observed action` 只诊断潜在动作差与 Selector regret；跨算法 transition 已包含在 observed loss 中，不能作为主 Utility 的额外性能罚项重复扣除。
