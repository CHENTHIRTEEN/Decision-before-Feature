# Decision-before-Feature Offline Utility Label 构建协议

> 实现同步（2026-08-11）：旧 phase1 utility-label shards 由重建式 continuation 生成，已撤回正式证据资格。完整 optimizer state 修正后，必须覆盖生成 trajectory、behavior 与 utility labels，再训练 Decision Model。

> Selection Reference 修订（2026-08-11）：正式标签必须连接逐共享状态 action-loss regression 输出；静态 problem label、remaining-budget bucket classifier 和 nearest-bucket prediction 不再是正式标签来源。

# 1. 离线效用标签目标

Decision-before-Feature需要监督信号：

模型需要知道：

> 当前问题是否值得执行所评估的固定 query。

因此需要构造：

Offline Utility Label。

---

# 2. 离线效用标签基本思想

对于同一个问题：

构造两个策略。

两个策略必须从同一个共享 checkpoint state 派生。

该 state 包括：

- checkpoint population；
- checkpoint fitness；
- checkpoint best-so-far fitness 与 position；
- optimizer generation 与 RNG state；
- PSO velocity/personal best/global best，CMA-ES strategy state，SHADE memory/archive 等算法内部动态量。

主实验采用 Population Transfer：

- 第一篇论文主行满足 `prefix_algorithm == default_algorithm ==` 训练集 SBS，No-query 原生继续该完整 checkpoint state；
- 全 prefix 稳健性行仍以训练集 SBS 作为 `default_algorithm`；若 default 与 prefix 不同，Skip 分支执行一次 population transfer；
- 如果 Run Query 仍选择 prefix algorithm，则复制同一完整状态并原生继续；
- 如果 Run Query 选择了不同算法，新算法继承 population、fitness 和 best fitness；
- 新算法不继承前缀算法内部状态；
- Best-so-far Warm Start 不进入主标签口径，只能作为后续稳健性分析候选；
- query 采样点不复用到后续优化 population。

在生成 Utility 前，必须先对每个共享状态完整运行唯一候选动作集合：`continue_current` 加其余三个 portfolio algorithms。动作表保存 raw loss、逐状态 normalized action loss、transition mode、best observed action 和 action runtime；随后只用 BBOB train states 拟合连续预算 Selector。

Trajectory Parquet 按 `phase1_dynamic_budget_event_v1` 保存输出状态的 population、fitness、best-so-far、`FE_total`、已完成的 `native_updates`、三个名义窗口的轻量集合/fitness 统计、最近10%预算内的逐 update 标量历史、`optimizer_state_mode` 与全部 sampling metadata，不把逐 update 的 RNG、velocity、evolution path、archive 等完整内部状态重复写入每一行。Utility 生成按 `(problem_id, prefix_algorithm, seed)` 原生重放一次前缀，在每个整数 `FE` 状态上对 population、fitness、best-so-far、`native_updates` 与 sampling metadata 做逐值一致性检查；随后复制内存中的完整状态生成两条分支。任一状态不一致时必须停止并重新生成 trajectory，不得退回 population-only 重建。

正式状态键为 `(split, problem_id, family, dimension, prefix_algorithm, seed, FE)`。eligible behavior、action-loss、Selection Reference 与 Utility label 必须在该键上双向完全覆盖；不使用浮点 `FE_ratio` 作 join key。

首轮离线状态采样不使用 decision score，每行 `sample_weight=1`。完整 BBOB-train family-OOF 上的 Q10 threshold-neighborhood 仅在模型与 threshold 冻结后用于 online 附加复查，BBOB-validation 与外部测试不参与带宽拟合，所有策略共享相同决策机会。

## Strategy A: No-query

直接优化。

流程：

    Problem

    ↓

    Prefix Optimizer Complete State

    ↓

    Default Algorithm

    （主协议中与 prefix 同为训练集 SBS）

    ↓

    Final Performance

得到：

$$
P_{skip}
$$

---

## Strategy B: Run Query

执行：

    Problem

    ↓

    Fixed Query

    ↓

    Algorithm Selection

    ↓

    Selected Optimizer

    ↓

    Final Performance

得到：

$$
p_{query}
$$

这里的 \(p_{query}\) 表示：

> 付出固定 query 成本后，selection reference 选择的算法从同一 checkpoint population 继续优化得到的最终 performance。

它不是围绕 best-so-far 重新初始化后得到的 performance。

---

# 3. Utility定义

## 3.1 Performance Gain

对于最小化问题：

$$
G=P_{skip}-p_{query}
$$

如果：

$$
G>0
$$

表示固定 query 路径取得更低的最终 performance。

---

## 3.2 Query Cost

固定 query 的资源账本包括：

### Function Evaluation Cost

Query 采样：

$$
FE_{query}
$$

主协议采用等总 FE，`FE_query_optimization = FE_total - FE_prefix - FE_query`。因此 sampling FE 已通过减少 Query continuation budget 进入 $p_{query}$，不得在 Utility 中再次扣除。

---

### Feature Computation Cost

包括：

- PCA
- nearest neighbor
- meta-model

---

### Runtime Cost

CPU time。

---

## 3.3 Final Utility

定义：

$$
U_{query}=G-\lambda_T C_T-\lambda_M C_M,
$$

其中 $C_T$ 是 Query 与 No-query 两条完整路径的有符号端到端 wall-clock 相对差，$C_M$ 是尚未进入 performance loss 的额外内存成本。Query 样本评价时间进入 Query 总时间，同时 Query 分支少执行的后续优化时间通过两条路径相减抵消。纯 feature/selector/handoff 计算开销只作诊断。若未来采用额外 query FE 而非等总 FE，必须另设实验条件和公式。

逐状态 best observed action 定义为：

$$
L_{best\ observed}(s_t)=\min_{a\in\mathcal A(s_t)}L(s_t,a).
$$

诊断分解为：

$$
V_{potential}=L_{noquery}-L_{best\ observed},
$$

$$
R_{selector}=L_{selector}-L_{best\ observed},
$$

$$
G=V_{potential}-R_{selector}.
$$

跨算法 Population Transfer 的影响已经包含在 $L(s_t,a)$ 中，不能再把 `handoff cost` 作为主 Utility 的独立减项。只有比较预先定义的不同初始化协议时，才能把其性能差作为稳健性结果单独报告。

---

# 4. 为什么不用简单标签？

错误方式：

    Fixed query improves final performance?

    Yes/No

问题：

丢失收益规模。

例如：

Problem A:

提升100%，成本100。

Problem B:

提升1%，成本100。

二者都会得到Yes。

但是价值不同。

---

因此推荐：

Regression：

预测：

$$
U_{query}
$$

---

# 5. Algorithm Selection Reference

每个 query 路径必须在实验前固定。

推荐：

使用标准Algorithm Portfolio。

例如：

- DE
- PSO
- CMA-ES
- SHADE

流程：

    Shared-state action losses on BBOB train

    ↓

    query features + algorithm-agnostic state behavior + continuous remaining budget

    ↓

    Multi-output action-loss regression

    ↓

    Minimum predicted-loss action

训练行采用 function-family grouped cross-fitting；held-out rows 只使用 BBOB train 拟合模型。逐状态最佳已观测动作只用于诊断，不作为可部署输入。

---

# 6. Prefix 与部署默认算法

离线 trajectory 覆盖 DE、PSO、CMA-ES 和 SHADE，但第一篇论文主 Decision 数据只保留由训练集 SBS 产生的 prefix：

```text
prefix_algorithm = default_algorithm = train-derived SBS
```

因此主 No-query 路径是原生继续当前 SBS；Query 路径在 `continue_current` 与其余三个 portfolio algorithm 构成的四个唯一动作中选择。选择 `continue_current` 时原生继续，选择其他算法时执行一次 population transfer。

其他 prefix algorithm 的标签单独进入 cross-probe robustness、leave-one-probe-out 与 algorithm-agnostic 泛化分析，不与主训练或主结果汇总。为使这些行可解释，`default_algorithm` 仍表示训练集 SBS；当它不同于 prefix 时，Skip 分支的切换必须由 `skip_switches_from_prefix=true` 和 `no_query_transition_mode=population_transfer_initialization` 显式记录。

每行还必须记录：

```text
selected_equals_default = (selected_algorithm == default_algorithm)
selected_equals_prefix = (selected_algorithm == prefix_algorithm)
handoff_required = not selected_equals_prefix
skip_switches_from_prefix = (default_algorithm != prefix_algorithm)
no_query_algorithm = default_algorithm
handoff_type = query_transition_mode
```

`handoff_required` 必须与 `handoff_type == population_transfer_initialization` 逐行相等。三个显式动作关系字段分别用于报告，不再生成 selected-vs-default 字符串别名。

`no_query_algorithm` 是 No-query 实际使用算法的显式字段；`handoff_type` 是 Query-selected action transition 的详细字段，并与 `query_transition_mode` 逐行相等。这些字段均不进入 Decision 输入。

在线部署的初始 default optimizer 仍由训练集 SBS 确定。若在线 query 未触发，SBS 的同一个完整状态持续推进；若 query 后 Selector 选择其他算法，才执行 population transfer。固定 CMA-ES 或 DE 只作为部署默认算法敏感性分析。

---

# 7. Lambda设置

不能随意设置。

推荐：

## Multi-lambda Analysis

例如：

$$
\lambda \in {0,0.25,0.5,1,2}
$$

观察：

Decision稳定性。

---

## Normalized Utility

由于性能和成本量纲不同：

先归一化：

$$
U= \Delta P_{norm} - \lambda C_{norm}
$$

---

# 8. Offline Utility Label Generation Pipeline

    For each problem:

    Run optimizer trajectories

    Generate behavior states

    Run No-query pipeline

    Record P_skip

    Run all Query candidate actions from the shared state

    Fit/apply the statewise Selection Reference

    Select the observed loss corresponding to the predicted action

    Record p_query

    Calculate:

    U_query

    Save:

    behavior state

    utility label

正式生成顺序不得颠倒：`state-action losses -> selection reference -> utility labels`。旧静态 selector 生成的标签与本协议不兼容。

---

# 9. Data Split Requirement

离线效用标签生成必须遵守：

function family split。

禁止：

同一个function family同时出现在train/test。

原因：

避免：

- shifted leakage
- rotated leakage
- noisy variant leakage

---

# 10. Offline Utility Label Quality Check

必须验证：

## Stability

不同random seed：

Utility是否稳定。

还必须报告 `selector_regret_raw`、`selected_matches_best_observed` 与各 transition mode 的分层结果。

---

## Sensitivity

不同：

- optimizer portfolio
- lambda

是否改变结论。

---

# 11. Decision Model Training

输入：

behavior state。

输出：

$$
\hat U_{query}
$$

损失：

$$
MSE(U,\hat U)
$$

决策：

$$
\hat U>0
$$

---

# 12. 最终评价

比较：

## Always Query

性能最高但成本高。

## Never Query

成本最低。

## Proposed

自动权衡。

指标：

- optimization quality
- total FE
- runtime
- Pareto efficiency

---

# 13. 关键实验问题

RQ1:

Query Utility是否存在明显分布？

RQ2:

Behavior是否能预测Utility？

RQ3:

Utility-label-based Decision 是否减少无效 query 调用？

RQ4:

Utility定义是否对lambda鲁棒？
