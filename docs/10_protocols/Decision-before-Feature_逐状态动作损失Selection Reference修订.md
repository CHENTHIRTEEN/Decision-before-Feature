# Decision-before-Feature 逐状态动作损失 Selection Reference 修订

## 1. 修订结论

本修订回应“当前 Selection Reference 与在线任务并不完全匹配”的方法问题。结论如下：

1. 原实现按 `problem × remaining-budget bucket` 预测一个静态最佳算法，再把该算法用于不同 seed、不同 prefix 和不同 checkpoint state。这个标签不等同于“从当前共享状态接管后，哪个动作的最终 loss 最小”，属于重要的构念失配。
2. 正式 Selection Reference 改为逐共享状态运行候选动作，监督目标改为连续动作损失，而不是最佳算法类别。
3. `remaining_budget_ratio` 作为连续输入，不再使用 nearest performance bucket；相邻 checkpoint 不再因 bucket 映射本身产生离散跳变。
4. 主 Utility 仍评价是否调用当前 `query_id + selector` 下游流程，不把算法选择本身改写为本文贡献。
5. 候选动作的实测 loss 已包含 native continuation 或 population transfer 的结果，query FE 也已通过减少后续优化预算计入；二者不能在主公式中再次相减。主公式只额外扣除尚未进入 loss 的时间、内存等非 FE 成本。

## 2. 与既有协议的冲突及裁决

| 建议 | 与现有协议的关系 | 裁决 |
|---|---|---|
| 对每个共享状态运行全部候选动作 | 增加离线数据生成成本，但仍属于 offline trajectory collection，不是在线 controller 训练 | 接受，作为正式 Selection Reference 标签来源 |
| 把 `continue current` 加入动作集合 | 与“同算法原生 continuation、跨算法 Population Transfer”完全一致 | 接受；它是 prefix algorithm 的语义动作，不与同名算法动作重复计数 |
| 预测连续的 \(\widehat L(s_t,a)\) | 保留选错动作的严重程度，比 multiclass label 更适合效用计算 | 接受；实现预测逐状态归一化动作损失 |
| 将 remaining budget 连续输入模型 | 与固定 checkpoint ratio 不冲突 | 接受；删除 nearest-bucket 选择逻辑 |
| 将 function ID、dimension、prefix algorithm ID 放入 Decision 输入 | 违反算法无关 Decision 输入边界 | 拒绝；这些字段只作 metadata 和分层报告 |
| 将 query features 放入 Selection Reference | Selection Reference 是 Query 后的固定 query-specific selector | 接受；query features 不进入 Decision Model |
| 将算法无关 behavior 放入 Selection Reference | 它描述当前 population、历史和成熟度，且 Query 前已经可得 | 接受，但必须增加 state-only selector 消融，分离 behavior 自身的选择信息 |
| 在主 Utility 中再扣 query sampling FE | 等总 FE 协议已通过减少 continuation budget 计入 | 拒绝，避免重复计费 |
| 将 handoff 单列后再从实测 selector loss 中扣除 | population-transfer 影响已经进入实测 action loss | 拒绝作为主公式的额外减项；只作预先定义 transition mode 的稳健性比较 |
| 把逐状态最小 loss 称为 VBS 或 oracle | 它只是已运行动作中的最小值 | 拒绝；称为“逐状态最佳已观测动作”或 `best observed action` |

## 3. 研究问题与实验单位

研究问题保持不变：

> 给定当前算法无关搜索行为，是否值得支付当前固定 query 的计算与预算成本并调用下游 selector？

Selection Reference 的实验单位是一个共享状态：

\[
s_t=(X_t,y_t,H_t,B_t),
\]

其中 \(X_t,y_t\) 是当前 population 与 fitness，\(H_t\) 是截至 checkpoint 的搜索历史，\(B_t\) 是扣除 query sampling FE 后的连续剩余预算。源码不把 population 行号解释为个体身份；模型使用 permutation-invariant behavior summaries 表示 \((X_t,y_t,H_t)\)。

## 4. 动作集合与真实损失

对 prefix algorithm 为 \(a_t\) 的共享状态，唯一动作集合定义为：

\[
\mathcal A(s_t)=\{\text{continue-current}\}\cup
\bigl(\{\mathrm{DE},\mathrm{PSO},\mathrm{CMA\mbox{-}ES},\mathrm{SHADE}\}\setminus\{a_t\}\bigr).
\]

因此四算法组合始终产生四个互不重复的动作：

- `continue_current`：复制完整 checkpoint state，保留优化器内部动态量与 RNG，原生推进；
- 其他三个算法动作：从同一 population、fitness 和 best-so-far 执行一次 `population_transfer_initialization`，随后推进；
- 所有动作使用相同的 query-adjusted remaining FE budget；
- query 采样点不并入 continuation population。

真实动作损失记为：

\[
L(s_t,a).
\]

逐状态最佳已观测动作为：

\[
a_t^{\mathrm{best\ observed}}=\arg\min_{a\in\mathcal A(s_t)} L(s_t,a).
\]

该量是离线诊断参照，不是现实可部署方法，也不作 Decision Model 输入。

## 5. Selection Reference 模型

Selector 为多输出性能回归：

\[
\widehat{\boldsymbol L}(s_t)
=f_\theta\!\left(
\text{behavior}(s_t),
\phi_{query}(p),
B_t/FE_{total}
\right).
\]

训练时，原始 BBOB objective 在不同 problem 上尺度与偏移不同。为保留同一 state 内的动作排序并避免跨问题 raw-loss 尺度支配，目标使用：

\[
\widetilde L(s_t,a)=
\frac{L(s_t,a)-\min_b L(s_t,b)}
{\max_b L(s_t,b)-\min_b L(s_t,b)+\epsilon}.
\]

部署时选择：

\[
\hat a_t=\arg\min_a \widehat{\widetilde L}(s_t,a).
\]

当前实现使用一个多输出 `RandomForestRegressor`。训练集 Selection Reference 行采用按 BBOB function family 分组的交叉拟合预测；held-out BBOB validation 和外部 benchmark 只使用全体 BBOB train families 拟合的最终模型。`function_id`、`dimension`、`prefix_algorithm`、seed 和 optimizer internal state 不进入 selector features。

## 6. Utility 与诊断分解

主路径仍定义为：

\[
U_{query}(s_t)=
\underbrace{L_{noquery}(s_t)-L_{selector}(s_t)}_{\text{observed performance gain}}
-\lambda_T C_T(s_t)-\lambda_M C_M(s_t).
\]

其中 query sampling FE 已通过 \(L_{selector}\) 使用较少 continuation FE 体现，不再额外相减。
时间成本中的 `runtime_selection` 必须包含单状态模型推理和动作选择，不能只计已有预测分数上的 `argmin` 时间。

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

主 Decision 实验继续包含：Never Query、Always Query、Random Analysis、Traditional AAS、SBS 和 VBS。另增加与本修订直接相关的 Selection Reference 诊断：

- state-only performance regression：behavior + continuous budget，不含 query features；
- query-only performance regression：query features + continuous budget，不含 behavior；
- full statewise selector：behavior + query features + continuous budget；
- 逐状态最佳已观测动作：仅作离线上界诊断；
- 原静态 bucket classifier：仅作被替代方法对照，不再生成正式 Utility 标签。

## 8. 数据泄漏控制

- BBOB train 与 validation 继续按冻结 function-family split 隔离；
- train utility labels 使用 cross-family selector predictions，不使用对同一 family 的 in-sample predictions；
- validation、CEC2017、CEC2022 和工程问题不参与 selector、Decision Model、preprocessing 或 threshold 拟合；
- 逐状态最佳已观测动作、observed action losses 和 selector regret 只作为离线标签或诊断字段，不进入可部署模型输入；
- 主 Decision 数据仍只保留 `prefix_algorithm == default_algorithm == train-derived SBS`，全 prefix 数据另存为 cross-probe robustness。

## 9. 结果保存方式

正式生成顺序为：

```text
native-state trajectories
-> permutation-invariant behavior
-> query features
-> state-action loss shards
-> statewise selection reference + fitted selector model
-> utility-label shards
-> Decision dataset and model evaluation
```

建议目录：

```text
results/selection_reference/{query_id}/state_action_losses/{split}/{family}/dimension_*/action_losses.parquet
results/selection_reference/{query_id}/selection_reference.parquet
results/selection_reference/{query_id}/statewise_selector.joblib
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
