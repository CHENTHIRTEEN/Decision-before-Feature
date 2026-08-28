# Decision-before-Feature 新方案采样协议

> 文档性质：离线数据采集与在线测评采样协议草案  
> 适用阶段：10D、`{PSO, SHADE, CMA-ES}`、BBOB + selected MA-BBOB 训练，CEC 10D 外部测评  
> 核心目标：保证离线训练数据与在线部署状态分布同构，并避免“每个 FE 都做完整 Behavior / 每个 FE 都做决策 / 每个 FE 都分支”的不必要计算

---

# 1. 总体原则

新方案中的采样必须区分三个不同频率：

1. **FE-level bookkeeping**：每次 objective evaluation 只维护极少量在线统计；
2. **Native-update-level monitoring**：每完成一次算法原生更新，更新 Behavior 与可选的 local landscape state；
3. **Decision-opportunity-level decision**：仅在预定义的决策机会构造正式输入并调用 action predictor。

因此：

$$
\boxed{
\text{每个 FE 记录轻量信息}
\neq
\text{每个 FE 计算完整 Behavior}
\neq
\text{每个 FE 做算法选择}
}
$$

离线与在线必须使用同一套 eligibility / monitoring 规则。

---

# 2. 三种时间尺度

## 2.1 FE-level

每次函数评价：

$$
x_i\rightarrow f(x_i)
$$

只维护：

- 当前 FE；
- 当前 best-so-far；
- 是否发生 improvement；
- first-hit 信息；
- 必要的 streaming scalar statistics；
- 若以后实现 reservoir，则进行 O(1) reservoir update。

不在这一层计算完整 population Behavior。

---

## 2.2 Native-update level

每完成一个完整的算法原生更新后，再更新 population-based Behavior。

例如：

- PSO：一个完整粒子群 update 完成；
- SHADE：一代 mutation / crossover / selection 完成；
- CMA-ES：一代 sampling / evaluation / distribution update 完成。

此时具有完整：

$$
P_t=\{x_1,\ldots,x_N\},
$$

$$
F_t=\{f_1,\ldots,f_N\}.
$$

建议记录：

- population；
- fitness；
- best fitness；
- native update index；
- FE；
- Behavior 所需的 rolling-window summaries。

---

# 3. Behavior monitoring

当前 Behavior 应继续使用 algorithm-agnostic、permutation-invariant 表示。

建议至少覆盖：

## Improvement / stagnation

- improvement rate；
- improvement frequency；
- best-fitness slope；
- stagnation；
- convergence rate。

## Population diversity

- mean pairwise diversity；
- diversity change；
- diversity slope；
- diversity recovery。

## Distribution dynamics

- fitness quantile improvement；
- fitness distribution improvement rate；
- fitness Wasserstein rate；
- fitness spread slope。

## Population motion

- population Wasserstein rate；
- centroid shift；
- centroid-shift coherence；
- Chamfer distance。

## Geometry

- covariance spectral concentration；
- covariance effective rank；
- effective-rank change。

---

# 4. Rolling window

保留当前三档窗口：

$$
w_{02}=0.02B,
$$

$$
w_{05}=0.05B,
$$

$$
w_{10}=0.10B.
$$

其中：

$$
B=FE_{total}.
$$

对于 10D 主实验：

$$
B=10000.
$$

因此：

$$
w_{02}=200FE,
$$

$$
w_{05}=500FE,
$$

$$
w_{10}=1000FE.
$$

Behavior recorder 只需要保留覆盖最近 $0.10B$ 所需的 native-update snapshots，不需要永久保存完整历史 population。

---

# 5. Monitor 与 Decision State 必须区分

推荐继续保留当前 dynamic-budget-event 设计。

## Monitor grid

在：

$$
FE/B\in[0.20,0.60]
$$

区间内，以：

$$
0.01B
$$

为 monitor 步长。

Monitor 只用于：

- 更新事件指标；
- 判断是否达到 milestone；
- 判断是否出现 Behavior event。

不是每个 monitor point 都要产生正式 decision state。

---

# 6. 固定 Budget Milestones

建议继续使用当前 milestone：

$$
\{
0.20,
0.22,
0.24,
0.26,
0.28,
0.30,
0.34,
0.38,
0.42,
0.46,
0.50,
0.60
\}.
$$

每条 run 至少产生：

$$
12
$$

个 milestone decision states。

---

# 7. Event-triggered Decision States

建议保留以下 Behavior event：

- improvement resume；
- stagnation onset；
- effective-rank / rank change；
- elite migration；
- diversity recovery。

每个 early / mid / late phase 最多保留：

$$
2
$$

个 event-only states。

因此每条 trajectory 正式 decision states 约为：

$$
12\sim18.
$$

---

# 8. 阶段划分

仅作为采样配额和后续分析使用，不作为算法 action 的硬约束。

建议：

$$
\text{early}: 0.20\le FE/B<0.30,
$$

$$
\text{mid}: 0.30\le FE/B<0.50,
$$

$$
\text{late}: 0.50\le FE/B\le0.60.
$$

注意：

> phase 只用于 reporting / sampling balance，不预先规定 PSO、SHADE、CMA-ES 分别属于哪个阶段。

---

# 9. 在线测评协议

在线测评时不需要每个 FE 构造完整 Behavior。

完整流程：

```text
objective evaluation
      ↓
update FE / best / lightweight statistics
      ↓
native update completed?
      ├── No → continue
      └── Yes
            ↓
      update Behavior monitor
            ↓
      crossed monitor grid?
            ├── No → continue
            └── Yes
                  ↓
          milestone or event?
              ├── No → continue
              └── Yes
                    ↓
              Decision State
                    ↓
             action predictor
```

---

# 10. Phase 1 在线策略

第一阶段不执行 extra ELA Query。

在 decision state $S_t$：

$$
B_t
\rightarrow
\text{action predictor}.
$$

假设当前算法是 PSO，则候选动作：

$$
\mathcal A(S_t)
=
\{
continue\ PSO,
switch\ SHADE,
switch\ CMAES
\}.
$$

模型预测每个候选动作相对于 continue 的未来优势。

然后选择：

$$
a_t^\star.
$$

第一阶段限制：

$$
N_{switch}\le1.
$$

一旦发生切换，后续不再产生第二次 switch decision。

---

# 11. Phase 2/3 在线 ELA 触发逻辑

后续加入 ELA 时，不应默认在每个 decision state 执行 Query。

应先使用：

$$
B_t
$$

或：

$$
[B_t,L_t^{local},U_t]
$$

完成初步 action prediction。

只有当满足以下类型的条件时，才评估 Query：

- top-1 / top-2 action probability margin 太小；
- action predictor entropy 较高；
- local landscape uncertainty 较高；
- Query VOI predictor 认为额外信息可能改变决策；
- 剩余预算足以覆盖 Query FE。

如果 Query 触发：

$$
FE_t
\rightarrow
FE_t+c_q.
$$

Query 消耗真实 FE。

随后：

$$
[B_t,L_t^{local},\Phi_q]
$$

重新进行 action selection。

---

# 12. 离线数据采集总体结构

离线采集不是：

> 每个 FE 都 fork 三个算法。

正确结构是：

```text
完整基础 trajectory
        ↓
native-update Behavior monitoring
        ↓
12–18 decision states
        ↓
只在这些 state 保存 checkpoint
        ↓
从同一 checkpoint fork candidate actions
        ↓
运行到统一 horizon
        ↓
构造 action-gain label
```

---

# 13. 离线基础 trajectory

对于：

$$
\text{BBOB}_{10D}
+
\text{selected MA-BBOB}_{10D}
$$

三个算法分别独立运行：

$$
PSO,\ SHADE,\ CMAES.
$$

每条基础 trajectory 从：

$$
FE=0
$$

运行到：

$$
FE=B=10000.
$$

过程中：

### FE-level

记录轻量信息。

### Native-update-level

记录 Behavior snapshot。

### Decision-opportunity-level

保存完整 checkpoint。

---

# 14. 离线正式 checkpoint

每个 decision state 必须保存足够恢复 continuation 的完整状态：

$$
S_t
=
(
P_t,
F_t,
best_t,
optimizer\ internal\ state,
RNG\ state,
FE_t
).
$$

同时保存：

- Behavior features；
- sampling trigger；
- sampling phase；
- milestone / event metadata；
- prefix algorithm；
- remaining budget。

---

# 15. 离线 action branching

对于一个来自 PSO 的状态：

$$
S_t,
$$

离线运行：

```text
                  continue PSO
                /
S_t -----------+----> SHADE
                \
                  CMA-ES
```

全部从同一个 state 出发。

对于 SHADE prefix：

```text
                  continue SHADE
                /
S_t -----------+----> PSO
                \
                  CMA-ES
```

对于 CMA-ES prefix 同理。

---

# 16. 第一阶段统一 horizon

第一阶段只使用 terminal horizon：

$$
\boxed{
H=B-FE_t
}
$$

即每个 action 从当前 state 一直运行到：

$$
FE=B.
$$

先验证：

> 当前状态下，哪个 action 最终更优？

短期 / 中期 horizon 后续再扩展。

---

# 17. Action Loss

定义 action terminal loss：

$$
L_{s,a}.
$$

对于 BBOB / MA-BBOB，可以使用 reference-optimum-based log gap：

$$
L_{s,a}
=
\log_{10}
\left(
\max(
f^{best}_{s,a}-f^\star,
\epsilon
)
\right).
$$

所有 action 在同一个 state 内使用统一预算终点。

---

# 18. Action Gain

以继续当前算法为 reference：

$$
\boxed{
G_{s,a}
=
L_{s,continue}
-
L_{s,a}
}
$$

解释：

- $G_{s,a}>0$：切换到 $a$ 更好；
- $G_{s,a}=0$：相近；
- $G_{s,a}<0$：切换更差。

对于 `continue_current`：

$$
G_{s,continue}=0.
$$

---

# 19. 三分类标签

建议主监督目标为：

$$
Y_{s,a}
=
\begin{cases}
Improve,&G_{s,a}>\delta_{practical},\\
Equivalent,&|G_{s,a}|\le\delta_{practical},\\
Degrade,&G_{s,a}<-\delta_{practical}.
\end{cases}
$$

不以极小数值差异制造唯一 best action。

---

# 20. Acceptable Action Set

定义：

$$
L_s^\star
=
\min_a L_{s,a},
$$

$$
\mathcal A_{acc}(s)
=
\left\{
a:
L_{s,a}-L_s^\star
\le
\delta_{practical}
\right\}.
$$

若多个动作属于 acceptable set，则选择其中任意一个都不视为错误。

---

# 21. 离线数据表结构

建议训练数据以：

$$
(state,\ candidate\ action)
$$

为基本单位，而不是每个 state 只保存一个 best-solver label。

示例：

| state_id | prefix_algorithm | Behavior | candidate_action | future_loss | gain_vs_continue | class |
|---|---|---|---|---:|---:|---|
| S001 | PSO | ... | PSO | 2.40 | 0.00 | Equivalent |
| S001 | PSO | ... | SHADE | 1.60 | +0.80 | Improve |
| S001 | PSO | ... | CMAES | 2.65 | -0.25 | Degrade |

---

# 22. Offline 与 Online 必须同分布

必须保证：

$$
\boxed{
\text{offline eligible states}
=
\text{online eligible states}
}
$$

如果在线仅在：

$$
0.20B\sim0.60B
$$

的 milestone / event state 决策，那么离线训练也只能使用这些状态作为主样本。

不要离线额外采：

$$
0.05B,\ 0.10B,\ 0.90B
$$

再让模型在线只在另一段区间部署。

---

# 23. Base seeds

第一阶段建议：

$$
10
$$

个 base optimizer seeds。

当前 10D 规模：

### BBOB train

$$
18\times3\times10\times3=1620
$$

条基础 trajectories。

### BBOB validation

$$
6\times3\times10\times3=540.
$$

### selected MA-BBOB

$$
24\times10\times3=720.
$$

合计：

$$
2880
$$

条基础 trajectories。

---

# 24. State 数量估计

如果每条 trajectory 平均：

$$
15
$$

个 decision states：

$$
2880\times15
=
43200
$$

个 states。

每个 state 3 个 actions：

$$
43200\times3
=
129600
$$

个 state-action outcomes。

对于第一阶段 tabular action prediction 已经足够。

---

# 25. Repetition 协议

不建议所有 state-action 全部做三重复。

第一遍：

$$
R=1
$$

覆盖全部正式 state-action。

再对预先定义的 subset：

$$
R=3
$$

或：

$$
R=5.
$$

建议 subset 由以下 strata 预先抽取：

- function；
- prefix algorithm；
- early / mid / late；
- milestone / event；
- easy / medium / hard；
- 固定随机比例。

避免根据第一次 outcome 事后决定是否补跑。

---

# 26. Repetition 用途

用于估计：

$$
Var[G_{s,a}],
$$

$$
P(G_{s,a}>0),
$$

以及：

$$
\text{sign-flip rate}.
$$

由此估计：

$$
\delta_{noise}.
$$

最终：

$$
\delta_{practical}
=
\max(
\delta_{domain},
\delta_{noise}
).
$$

---

# 27. 后续加入 Local Landscape 时的采样

如果进入 Phase 2：

### 每 FE

仅递推：

- fitness moments；
- reservoir；
- cheap scalar statistics。

### 每 native update

更新：

- local information content；
- local meta-model；
- covariance-conditioned landscape statistics。

### 每 decision state

正式计算：

- local FDC；
- local dispersion；
- approximate NBC；
- bootstrap uncertainty。

这样不需要在每个 FE 做完整在线 ELA。

---

# 28. 后续加入独立 ELA Query 时的离线采样

当研究 adaptive Query 时，每个 selected decision state 需要两类 action matrix。

## No-query matrix

$$
M_t^B
=
\{
L_t^B(a)
\}_{a\in\mathcal A}.
$$

预算：

$$
B-FE_t.
$$

## Query-adjusted matrix

先消耗：

$$
c_q
$$

FE。

然后：

$$
M_t^{B+Q}
=
\{
L_t^{B+Q}(a)
\}_{a\in\mathcal A}.
$$

剩余预算：

$$
B-FE_t-c_q.
$$

---

# 29. Query 阶段建议保留四条控制路径

为了拆分信息效果与成本效果，建议至少保留：

### A. Behavior-only

不 Query，完整剩余预算。

### B. Query + descriptors

花 Query FE，并使用 descriptors 进行动作选择。

### C. Matched Query Cost / no descriptors

花同样 Query FE，但 selector 不看 descriptors。

### D. Sampling-only continue

花同样 Query FE，然后继续当前算法。

这样可以区分：

- Query sample direct effect；
- descriptor information effect；
- FE opportunity cost；
- selector effect。

---

# 30. 第一阶段与后续阶段的边界

## Phase 1

只做：

$$
\boxed{
Behavior \rightarrow Action Gain
}
$$

不做 extra ELA。

## Phase 2

加入：

$$
Trajectory\mbox{-}derived\ Local\ Landscape.
$$

## Phase 3

加入：

$$
Adaptive\ Query.
$$

## Phase 4

加入：

$$
Dynamic\ Soft\mbox{-}ERT.
$$

## Phase 5

加入：

$$
Repeated\ DAS.
$$

---

# 31. CEC 10D 在线测评

CEC 10D 使用与离线完全一致的 monitoring / decision opportunity。

进入 CEC 后保持以下内容不变：

- Behavior extractor；
- rolling windows；
- milestone；
- event thresholds；
- action predictor；
- preprocessing；
- practical threshold；
- portfolio；
- FE budget；
- population size；
- handoff protocol；
- one-switch rule。

CEC outcome 不用于重新训练或调 threshold。

---

# 32. CEC 在线流程

```text
CEC 10D
   ↓
run current solver
   ↓
FE-level lightweight bookkeeping
   ↓
native-update Behavior monitoring
   ↓
milestone / event
   ↓
Decision State
   ↓
Behavior action predictor
   ↓
continue or switch
   ↓
one-switch policy
   ↓
terminal outcome
```

如果后续进入 Adaptive Query 阶段，则在 action uncertainty 足够高且 Query VOI 为正时增加 ELA branch。

---

# 33. 最终推荐的第一阶段采样协议

```text
Benchmark:
    BBOB 10D + selected MA-BBOB 10D

External evaluation:
    CEC 10D

Algorithms:
    PSO
    SHADE
    CMA-ES

FE_total:
    10000

Population:
    40

Base seeds:
    10

FE-level:
    best-so-far
    first-hit
    lightweight statistics

Native-update-level:
    full population snapshot
    fitness
    Behavior monitor

Monitor range:
    0.20B - 0.60B

Monitor step:
    0.01B

Budget milestones:
    0.20
    0.22
    0.24
    0.26
    0.28
    0.30
    0.34
    0.38
    0.42
    0.46
    0.50
    0.60

Event-only states:
    max 2 per phase

Decision states per trajectory:
    approximately 12-18

Offline branching:
    only at decision states

Actions:
    continue current
    switch candidate 1
    switch candidate 2

Main horizon:
    terminal

Main target:
    action gain vs continue

Policy:
    one-switch

Repetitions:
    R=1 full coverage
    R=3 or 5 pre-specified subset

Extra ELA:
    not used in Phase 1
```

---

# 34. 一句话总结

在线：

> **每个 FE 只维护轻量统计，每个 native update 更新 Behavior，只在 milestone / event decision state 调用模型；Behavior 足够时直接 continue/switch，后续若引入 ELA，则仅在信息不足且 Query 有价值时额外花 FE。**

离线：

> **使用完全相同的 monitoring 和 decision-state 机制生成 prefix trajectory，只在正式 decision state 保存 checkpoint，并从同一 checkpoint 分别运行所有候选 action 到统一 horizon，构造 action-gain supervision；只有研究 Query 时才额外生成 query-adjusted branches。**

最终原则：

$$
\boxed{
\text{high-frequency monitoring}
+
\text{low-frequency decision}
+
\text{decision-state-only branching}
}
$$
