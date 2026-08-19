# Decision-before-Feature 逐状态动作损失 Selection Reference 修订

## 1. 修订结论

本修订回应“当前 Selection Reference 与在线任务并不完全匹配”的方法问题。结论如下：

1. 原实现按 `problem × remaining-budget bucket` 预测一个静态最佳算法，再把该算法用于不同 seed、不同 prefix 和不同 checkpoint state。这个标签不等同于“从当前共享状态接管后，哪个动作的最终 loss 最小”，属于重要的构念失配。
2. 正式 Selection Reference 改为逐共享状态运行候选动作，监督目标改为连续动作损失，而不是最佳算法类别。
3. `remaining_budget_ratio` 作为连续输入，不再使用 nearest performance bucket；相邻 checkpoint 不再因 bucket 映射本身产生离散跳变。
4. 主 Utility 仍评价是否调用当前 `query_id + selector` 下游流程，不把算法选择本身改写为本文贡献。
5. 候选动作的实测 loss 已包含 native continuation 或 population transfer 的结果，query FE 也已通过减少后续优化预算计入；二者不能在主公式中再次相减。方案 A 下，主功效改为等总 FE 的 `G_FE = log((E_skip + epsilon_p) / (E_query + epsilon_p))`，runtime 不进入主功效。字段规范以《最小 Action Loss 字段规范 v1》为准：每条 action 记录必须同时保留行标识、科学端点、censored runtime 和一个 canonical loss（`action_loss`）；旧 Utility 变体仅作兼容诊断。
6. 正式链路同时生成 `behavior_only_full_budget` 四动作 outcome 与 Selector；`query_operational_increment` 仅作为旧口径兼容与辅助诊断，不再作为主实验标签。

## 2. 与既有协议的冲突及裁决

| 建议 | 与现有协议的关系 | 裁决 |
|---|---|---|
| 对每个共享状态运行全部候选动作 | 增加离线数据生成成本，但仍属于 offline trajectory collection，不是在线 controller 训练 | 接受，作为正式 Selection Reference 标签来源 |
| 把 `continue current` 加入动作集合 | 与“同算法原生 continuation、跨算法 Population Transfer”完全一致 | 接受；它是 prefix algorithm 的语义动作，不与同名算法动作重复计数 |
| 预测连续的 \(\widehat L(s_t,a)\) | 保留选错动作的严重程度，比 multiclass label 更适合效用计算 | 接受；实现预测逐状态归一化动作损失 |
| 将 remaining budget 连续输入模型 | 与冻结动态采样状态不冲突 | 接受；删除 nearest-bucket 选择逻辑 |
| 将 function ID、dimension、prefix algorithm ID 放入 Decision 输入 | 违反算法无关 Decision 输入边界 | 拒绝；这些字段只作 metadata 和分层报告 |
| 将 query features 放入 Selection Reference | Selection Reference 是 Query 后的固定 query-specific selector | 接受；query features 不进入 Decision Model |
| 将算法无关 behavior 放入 Selection Reference | 它描述当前 population、历史和成熟度，且 Query 前已经可得 | 接受；必须同时提供 full-budget behavior-only 主对照，query-adjusted state-only 只能作诊断 |
| 在主功效中再扣 query sampling FE | 等总 FE 协议已通过减少 continuation budget 计入 | 拒绝，避免重复计费；主功效直接使用 `G_FE` |
| 将 handoff 单列后再从实测 selector loss 中扣除 | population-transfer 影响已经进入实测 action loss | 拒绝作为主公式的额外减项；只作预先定义 transition mode 的稳健性比较 |
| 把逐状态最小 loss 称为 VBS 或 oracle | 它只是已运行动作中的最小值 | 拒绝；称为“逐状态最佳已观测动作”或 `best observed action` |

## 3. 研究问题与实验单位

研究问题保持不变：

> 给定当前算法无关搜索行为，Query/full-Selector 联合路径是否优于原生 SBS continuation；相对于不执行 query 的 `behavior_only_full_budget` 选择路径，固定 query 是否提供操作性净增量？

Selection Reference 的实验单位是一个共享状态：

\[
s_t=(X_t,y_t,H_t,B_t),
\]

其中 \(X_t,y_t\) 是当前 population 与 fitness，\(H_t\) 是截至 checkpoint 的搜索历史。定义 Query-adjusted budget \(B_t^q=B-e_t-FE_q\) 和 full budget \(B_t^b=B-e_t\)。源码不把 population 行号解释为个体身份；模型使用 permutation-invariant behavior summaries 表示 \((X_t,y_t,H_t)\)。

## 4. 动作集合与真实损失

对 prefix algorithm 为 \(a_t\) 的共享状态，唯一动作集合定义为：

\[
\mathcal A(s_t)=\{\text{continue-current}\}\cup
\bigl(\{\mathrm{DE},\mathrm{PSO},\mathrm{CMA\mbox{-}ES},\mathrm{SHADE}\}\setminus\{a_t\}\bigr).
\]

因此四算法组合始终产生四个互不重复的动作：

- `continue_current`：复制完整 checkpoint state，保留优化器内部动态量与 RNG，原生推进；
- 其他三个算法动作：从同一 population、fitness 和 best-so-far 执行一次 `population_transfer_initialization`，随后推进；
- Query outcome matrix 的所有动作使用相同的 \(B_t^q\)；
- `behavior_only_full_budget` outcome matrix 的同四个动作使用相同的 \(B_t^b\)，且不生成或读取 query features；
- query 采样点不并入 continuation population。

两套真实动作损失分别记为：

\[
L_q(s_t,a),\qquad L_b(s_t,a).
\]

逐状态最佳已观测动作为：

\[
a_{t,q}^{\mathrm{best\ observed}}=\arg\min_a L_q(s_t,a),\qquad
a_{t,b}^{\mathrm{best\ observed}}=\arg\min_a L_b(s_t,a).
\]

该量是离线诊断参照，不是现实可部署方法，也不作 Decision Model 输入。

## 5. Selection Reference 模型

两类 Selector 都是固定的多输出性能回归，但输入和预算不可混用：

\[
\widehat{\boldsymbol L}_q(s_t)
=f^q_\theta\!\left(
\text{behavior}(s_t),
\phi_{query}(p),
B_t^q/FE_{total}
\right),
\qquad
\widehat{\boldsymbol L}_b(s_t)
=f^b_\theta\!\left(
\text{behavior}(s_t),
B_t^b/FE_{total}
\right).
\]

训练时，原始 BBOB gap 在不同 problem 上尺度不同。为与主 terminal `log10_gap` 风险同单位，并使 `continue_current` 成为固定参照，主目标使用：

\[
Y(s_t,a)=\log_{10}(\operatorname{clip}(L(s_t,a),g_{\min},g_{\max}))
-\log_{10}(\operatorname{clip}(L(s_t,a_{continue}),g_{\min},g_{\max})).
\]

部署时选择：

\[
\hat a_t=\arg\min_a \widehat{Y}(s_t,a).
\]

两类模型都使用相同的预设多输出 `RandomForestRegressor` 参数，但分别拟合，不共享不适用的输入。训练集 Selection Reference 行采用按 BBOB function 分组的交叉拟合预测；held-out BBOB validation 和外部 benchmark 只使用全体 BBOB train functions 拟合的最终模型。`function_id`、`dimension`、`prefix_algorithm`、seed 和 optimizer internal state 不进入 selector features。

端到端 outer evaluation 还要求 fold-specific 重拟合。对每个 Decision outer holdout function group，先只用 outer-fit functions 计算 `SBS_outer`，再在 outer-fit 内 cross-fit 两类 Selector 生成 Decision 训练标签；outer holdout 的 Utility 只能由 outer-fit 全量 Selector 预测生成。每个 Decision inner holdout 还必须只用 inner-fit functions 重算 `SBS_inner`、cross-fit/拟合两类 Selector并生成 Utility，不能复用含 inner holdout 信息的 outer-fit 上游标签。outer/inner holdout 不得进入其评价链的 SBS、Selector、目标变换、Decision preprocessing 或 threshold。完整 BBOB-train 的 cross-fit Selection Reference 不能冒充上述 fold-specific 链。

模型产物和主 Selection Reference 行固定保存 `selector_target_transform=clipped_log10_gap_advantage_vs_continue_current`；`statewise_minmax_observed_action_loss` 只用于单独标记的 target sensitivity。Selection Reference、Utility、Decision dataset 与在线输出同时保存 `selected_equals_default`、`selected_equals_prefix` 和 `handoff_required`；其中 `handoff_required = not selected_equals_prefix`，并与 `handoff_type == population_transfer_initialization` 逐行一致。

## 6. Utility 与诊断分解

Selector 主 target 是第 5 节基于 continuation outcomes 的相对 continue-current 截断 `log10_gap`；Utility 不直接使用该 target。Stage-A 预指定单次 Selection Reference outcome 是 terminal gap、`observed_first_hit_FE`、`target_hit_observed`、`target_hit_before_failure`、`path_completed`、`endpoint_success` 与 planned/effective FE 的唯一科学来源。Query sample 不进入 continuation population，但主 operational Query terminal best、observed first hit 与标准 ERT 必须计入真实 query sample evaluations；ERT 使用 `target_hit_observed`，完整路径且命中的 endpoint 另由 `endpoint_success` 表示。方案 A 下，主功效不再以 `query_operational_increment_lamT_*` 为唯一标签，而改用 `G_FE`；`query_operational_increment_lamT_*` 仅作为兼容诊断。令五条 Stage-A operational 路径的非负 benchmark-reference raw gap 在取对数前按 suite 配置截断；另保存 query continuation-only gap 与 `query_sample_best_contribution_log10_gap`。对 BBOB train/validation 与 CEC2017：

\[
\ell_k=\log_{10}\!\left(\min(\max(g_k,10^{-12}),10^{20})\right).
\]

Stage-B 每次 timing-only replay 保留 raw observed wall-clock；completed repetition 的 censored time 等于 raw，timed-out/failed repetition 的 censored time 为 `max(raw, role timeout)`。令 (T_{skip},T_q,T_b) 为同一 complete state/RNG 起点下三次 censored 完整路径时间的中位数；raw median 只作诊断。主联合策略效用定义为：

\[
U_{query}^{joint}(s_t)=
(\ell_{skip}-\ell_q)
-\lambda_T(\log_{10}T_q-\log_{10}T_{skip}).
\]

Behavior-only full-budget 效用与 query 操作性增量分别为：

\[
U_b(s_t)=(\ell_{skip}-\ell_b)
-\lambda_T(\log_{10}T_b-\log_{10}T_{skip}),
\]

\[
I_q(s_t)=(\ell_b-\ell_q)
-\lambda_T(\log_{10}T_q-\log_{10}T_b)
=U_{query}^{joint}(s_t)-U_b(s_t).
\]

其中 query sampling FE 已通过 Query path 使用较少 continuation FE 体现，不再额外相减。旧 `u_query_joint_lamT_1` / `query_operational_increment_lamT_1` 仅保留为过渡兼容字段；方案 A 的主标签是 `G_FE`，并由 `g_fe`、`g_fe_bounded`、`g_fe_gt_zero`、`g_fe_gt_practical` 派生。`query_operational_increment_lamT_1` 若仍输出，只作辅助诊断，并必须明确解释为 fixed-model、action-budget 与 transition rule 下相对 Behavior-only 路径的操作性净增量，不是主功效。它包含 query FE/runtime、sample best、预算差和 Selector 差异，不是纯信息效应或因果 estimand。若 `U_joint>0` 而 `I_q<=0`，只能支持联合路径优于 SBS。主 `lambda_T=1` 表示 gap 与 runtime 的十进制数量级变化等权；memory 的主权重为 0、另作端点。

Query-adjusted state-only Selector 与 full Query Selector 在同一四动作矩阵上比较 OOF selected continuation-only `log10_gap`，输出 `query_feature_predictive_increment_log10_gap`。该诊断排除 query sample best，不新增 action losses，只表示 query features 的 OOF 边际预测贡献；不得把随机采样直接找到更优点归因于 features，也不得与主功效 `G_FE` 混称。正式五路径仍可保留 `query_matched_state_only` 与 `sampling_only_continue_current` 作为辅助分解，但这些只作为旧 utility/诊断兼容，不再作为主实验标签。

`runtime_selection` 必须包含单状态模型推理和动作选择，不能只计已有预测分数上的 `argmin` 时间；`runtime_handoff` 与 `runtime_query_optimization` 必须拆分保存，避免把 transfer 初始化隐含在后续优化中。

同一 complete state/action 的 selected Skip、Query 与 Behavior-only 完整路径必须在 Stage B 使用固定线程并从相同显式随机状态真实 replay 预定 3 次；canonical path order 按 `cyclic_complete_path_v1` 在 repetition 间循环移位。逐次保存 repetition/order、raw/censored 组件与完整路径 wall-clock、status、observed hit、path completion、endpoint success、effective FE 与 timeout。路径身份、completed replays 内部 endpoint、Stage-A→completed replay endpoint 一致性分别保存；Stage-B status instability 与跨阶段 completion instability 也分别保存。replay effective FE 不覆盖 Stage-A 字段，任何 repetition 不整批拒绝且不得选择性补跑。主结果另报 Stage-A `log10_gap`、Stage-B 主 censored `log10` wall-clock ratio 与 raw timing diagnostics，且不得按结果改选主 `lambda_T=1`。`lambda_T={0,0.25,0.5,1,2}` 只作完整敏感性分析。旧 max-scale/relative-time 以及把快速失败 raw runtime 作为主成本的 Utility 全部失效。

逐状态最佳已观测动作只用于以下恒等分解：

\[
V_{potential}=L_{noquery}-L_{best\ observed},
\]

\[
R_{selector}=L_{selector}-L_{best\ observed},
\]

\[
L_{noquery}-L_{selector}=V_{potential}-R_{selector}.
\]

该分解可以区分“当前状态没有可实现的动作性能差”和“现实 selector 未选到已观测最佳动作”。它不能把 handoff 独立识别为一个可加减的因果效应。若要研究 transition mode，必须预先定义对照初始化、使用同一共享状态和预算分别运行，并作为稳健性实验报告。

## 7. Baseline

主 Decision 实验包含 Never Query/SBS、Always Query、`matched_rate_random`、`pre_run_aas_fe0`、VBS、`milestone_only_T0`、`self_thresholded_behavior_only` 和 Proposed。`matched_trigger_behavior_only` 与 Proposed 使用相同首次触发 state，只用于计算政策级 query 操作性增量；它不替代 self-thresholded baseline。另保留以下 Selection Reference 诊断：

- state-only performance regression：behavior + continuous budget，不含 query features；
- query-only performance regression：query features + continuous budget，不含 behavior；
- full statewise selector：behavior + query features + continuous budget；
- 逐状态最佳已观测动作：仅作离线上界诊断；
- 原静态 bucket classifier：仅作被替代方法对照，不再生成正式 Utility 标签。

## 8. 数据泄漏控制

- BBOB train 与 validation 继续按 function-ID split 隔离；该设计只称 `cv_group_id = function_id`，不声称经典 landscape-family 泛化；
- train utility labels 使用 `cv_group_id = function_id` selector predictions，不使用对同一 function 的 in-sample predictions；
- Decision outer holdout function 不参与该 fold 的 `SBS_outer`、两类 Selector、Utility、Decision、inner threshold 或 Random calibration；每个 inner holdout 同样不参与其评价链的 `SBS_inner`、Selector、Utility 或 preprocessing；
- validation、CEC2017、CEC2022 和工程问题不参与 selector、Decision Model、preprocessing 或 threshold 拟合；
- 逐状态最佳已观测动作、observed action losses 和 selector regret 只作为离线标签或诊断字段，不进入可部署模型输入；
- 主 Decision 数据仍只保留 `prefix_algorithm == default_algorithm == train-derived SBS`，全 prefix 数据另存为 cross-probe robustness。

## 9. 结果保存方式

正式生成顺序为：

```text
native-state trajectories + complete-budget outcomes
-> permutation-invariant behavior
-> shared query samples and query features
-> query-adjusted + full-budget state-action loss shards
-> outer-fold-specific SBS and two fitted/cross-fitted selectors
-> joint/behavior-only/operational-increment utility-label shards
-> Decision datasets, inner first-trigger thresholds and outer evaluation
```

建议目录：

```text
results/selection_reference/{query_id}/query_adjusted_action_losses/{split}/{function}/dimension_*/action_losses.parquet
results/selection_reference/behavior_only_full_budget/action_losses/{split}/{function}/dimension_*/action_losses.parquet
results/selection_reference/{query_id}/selection_reference.parquet
results/selection_reference/{query_id}/statewise_selector.joblib
results/selection_reference/behavior_only_full_budget/statewise_selector.joblib
results/utility_labels/{query_id}/
```

主检查命令：

```bash
uv run --frozen optimizer-state-check
uv run --frozen selection-reference-check
uv run --frozen selection-reference-evaluate-actions --help
uv run --frozen selection-reference-build --help
```

旧静态 bucket Selection Reference、由其生成的 utility labels 和依赖这些标签的模型不得与本协议输出混用。
