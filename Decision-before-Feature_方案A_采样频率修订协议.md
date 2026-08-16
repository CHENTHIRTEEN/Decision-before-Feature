# Decision-before-Feature 方案 A：等总 FE 性能功效的采样频率修订协议

> 版本：v1.0  
> 文档性质：论文级实验协议与代码实现规范  
> 适用方案：方案 A，等总函数评价预算下的 ELA 性能功效  
> 主标签：$G_{\mathrm{FE}}$  
> 适用任务：连续单目标黑盒优化、运行中至多一次 Landscape Query、下游算法选择与原生 continuation / population transfer  
> 公式规范：所有数学公式均采用标准 Markdown `$...$` 与 `$$...$$`

---

## 摘要

方案 A 将 ELA 的主科学评价从依赖程序运行时间和人为权重的混合 Utility，改为等总函数评价预算下的客观性能功效：

$$
G_{\mathrm{FE}}(s_t,q)
=
\log
\frac{
E_{\mathrm{noquery}}+\epsilon_p
}{
E_{\mathrm{query}}+\epsilon_p
}.
$$

其中，Query 所消耗的函数评价预算已经通过减少 Query 路径的后续优化预算进入最终结果，因此主标签不再强制依赖 wall-clock。

这一修改不要求推翻原有“高频观测、低频决策、共享前缀、每条 run 至多一次 Query”的实验骨架，但需要对采样协议作以下修订：

1. **原生轨迹观测继续保持高频**：每次完整 optimizer update 后更新轻量行为历史；
2. **离线分叉状态保持稀疏**：完整 checkpoint 只在冻结的 milestone 或 behavior-event opportunity 保存；
3. **旧的 $[0.20,0.60]$ 候选区间必须经过新标签 Pilot 重新验证**，不能继续由已撤回的旧 Utility 结果担保；
4. **正式在线策略继续采用 run-level one-shot first-trigger**，但 decision schedule 和 score threshold 必须在 BBOB train 的 grouped OOF 结果上联合冻结；
5. **milestone + event 机会生成器必须做独立消融**，以区分机会生成规则与 Controller 本身的贡献；
6. **Trajectory-based zero-FE descriptor 应在轨迹采集阶段同步保留**，避免正式运行后无法补充最近邻强基线；
7. **若后续采用 Skip–Defer–Query 三向策略**，则 Defer 与 Skip 必须拥有不同的复查频率，否则它们在执行层面没有区别。

本协议的核心原则是：

> **高频观察搜索状态，低频生成昂贵标签，更低频执行 Landscape Query。**

---

# 1. 文档目标与研究边界

## 1.1 文档目标

本文只解决方案 A 下的采样频率问题，包括：

- 原始优化轨迹多久观测一次；
- Behavior rolling windows 如何获得稳定支撑；
- 哪些状态需要保存完整 optimizer checkpoint；
- 哪些状态需要进行 No-query / Query 配对续跑；
- 在线 Controller 在哪些状态获得决策机会；
- decision schedule 与 threshold 如何冻结；
- 采样频率如何做公平消融；
- 哪些修改会迫使重新生成 trajectory，哪些只需重算下游结果。

本文不重新定义：

- Algorithm Portfolio；
- Landscape Query 的具体特征组；
- Selection Reference 的模型结构；
- Handoff 的具体实现；
- 方案 B 的 Behavior-only 增量价值标签；
- 模糊控制器的完整规则库。

## 1.2 方案 A 的核心问题

方案 A 回答：

> 在相同总 FE 预算下，从当前共享前缀状态执行固定 Landscape Query，并据此选择后续算法，是否比 No-query 路径获得更好的最终解？

它不首先回答：

> Query 的 CPU 时间值多少钱？

也不首先回答：

> ELA 是否比 Behavior-only Selector 更有增量价值？

后两者分别属于独立资源评价和方案 B。

## 1.3 采样协议的科学角色

采样协议不是普通工程参数。它会决定：

- Controller 能看到哪些搜索阶段；
- 哪些状态有真实 $G_{\mathrm{FE}}$ 标签；
- 早期高价值或晚期高价值状态是否被系统性漏掉；
- 事件触发规则是否替 Controller 提前完成了大部分筛选；
- 在线 first-trigger 是否因检查过密而增加误触发；
- 不同 run 的状态数是否造成训练权重偏差；
- OOD 测试时是否出现训练未覆盖的 decision opportunity。

因此，采样频率必须被版本化、冻结并单独消融。

---

# 2. 统一符号与定义

## 2.1 基本符号

| 符号 | 含义 |
|---|---|
| $p$ | 黑盒优化问题实例 |
| $d$ | 问题维度 |
| $B$ | 总函数评价预算，即 `FE_total` |
| $r$ | 一条完整 problem/instance/seed/prefix run |
| $u$ | 原生 optimizer update 索引 |
| $t$ | 候选 decision opportunity 索引 |
| $b_t$ | 当前状态已消耗的 FE |
| $\rho_t=b_t/B$ | 当前实际 FE ratio |
| $s_t$ | 当前完整 optimizer checkpoint state |
| $B_t$ | 当前 trajectory prefix 的 Behavior 表示 |
| $q$ | 固定 Landscape Query 配置 |
| $c_q$ | Query 消耗的 FE |
| $\Phi_t^{(q)}$ | Query 返回的 landscape descriptors |
| $L_{0,t}$ | No-query 路径终端损失 |
| $L_{q,t}$ | Query 路径终端损失 |
| $E_{0,t}$ | No-query 路径相对于合法参考值的终端 gap |
| $E_{q,t}$ | Query 路径相对于合法参考值的终端 gap |
| $G_{\mathrm{FE},t}$ | 当前状态下的等总 FE 性能功效 |
| $z_t$ | Controller 输出的 efficacy score |
| $\tau$ | 在线 score threshold |
| $\delta_{\mathrm{practical}}$ | 最小实际意义功效阈值 |

## 2.2 完整状态

每个候选分叉状态必须包含：

$$
s_t
=
\left(
X_t,
F_t,
\text{best}_t,
\text{optimizer internal state},
\text{RNG state},
\text{pending native update state}
\right).
$$

仅保存 population、fitness 和 best-so-far，不足以称为完整同算法 continuation。

## 2.3 等总 FE 路径

### No-query 路径

$$
FE_{0,t}^{\mathrm{remain}}
=
B-b_t.
$$

### Query 路径

$$
FE_{q,t}^{\mathrm{remain}}
=
B-b_t-c_q.
$$

必须满足：

$$
FE_{0,t}^{\mathrm{used}}
=
FE_{q,t}^{\mathrm{used}}
=
B.
$$

Query 的 FE 机会成本已经进入 $L_{q,t}$，不得在 $G_{\mathrm{FE}}$ 中重复扣除。

## 2.4 主功效标签

对已知最优值或合法 benchmark reference value 的问题：

$$
E_{0,t}
=
\max\left(f_{0,t}-f^\star,0\right),
$$

$$
E_{q,t}
=
\max\left(f_{q,t}-f^\star,0\right).
$$

定义问题尺度协变稳定项：

$$
\epsilon_p
=
\eta
\max
\left(
E_{\mathrm{prefix},t},
S_p,
\epsilon_0
\right).
$$

主功效为：

$$
\boxed{
G_{\mathrm{FE},t}
=
\log
\frac{
E_{0,t}+\epsilon_p
}{
E_{q,t}+\epsilon_p
}
}
$$

解释：

- $G_{\mathrm{FE},t}>0$：Query 路径更好；
- $G_{\mathrm{FE},t}=0$：两条路径等价；
- $G_{\mathrm{FE},t}<0$：Query 路径有害；
- $G_{\mathrm{FE},t}>\delta_{\mathrm{practical}}$：Query 具有实际意义上的正功效。

---

# 3. 四个必须分离的采样时钟

方案 A 的正式实现必须区分四个时钟。禁止继续用同一份 checkpoint 列表同时控制轨迹观测、Behavior窗口、离线分叉和在线决策。

## 3.1 时钟 A：原生 optimizer update

记为：

$$
u=1,2,\ldots,U_r.
$$

每完成一次完整原生更新，就更新轻量历史。

不同算法中的原生更新定义：

| 算法 | 原生 update |
|---|---|
| DE | 一次完整 target population 更新 |
| SHADE | 一次完整 generation 更新，包括 memory/archive 更新 |
| PSO | 一次全群位置、速度和 personal-best 更新 |
| CMA-ES | 一次完整 ask–tell、mean/covariance/path/step-size 更新 |

### 每次原生 update 应更新

```text
native_update_index
FE
FE_ratio
best_fitness
fitness_quantiles
fitness_iqr
fitness_std
population_diversity
centroid
elite_centroid
covariance_trace
covariance_effective_rank
improvement_count
stagnation_length
rolling event statistics
trajectory reservoir / trajectory descriptor accumulator
```

### 每次原生 update 不应执行

- 不保存完整 population 到正式主表；
- 不运行 Landscape Query；
- 不生成 action-loss branches；
- 不调用最终 Controller；
- 不改变 optimizer RNG；
- 不重建 optimizer state。

## 3.2 时钟 B：Behavior rolling windows

Behavior窗口继续使用 FE-ratio 口径：

$$
W
\in
\{0.02,0.05,0.10\}.
$$

对窗口 $W$：

$$
\mathcal H_t^{(W)}
=
\left\{
\text{native-update summaries }h_u:
0\le
\frac{FE_t-FE_u}{B}
\le W
\right\}.
$$

每个窗口必须保存：

```text
nominal_window_ratio
effective_window_ratio
effective_window_fe
effective_native_updates
window_start_FE
window_end_FE
```

### 支撑条件

设最小 update 支撑数为 $m_W$，则：

$$
|\mathcal H_t^{(W)}|\ge m_W.
$$

若不满足：

- 标记为 insufficient support；
- 不使用零值冒充“没有变化”；
- 不复制上一个状态的特征；
- 不偷偷扩大到更长窗口却仍命名为 `w02`。

## 3.3 时钟 C：离线标签 opportunity

只有在冻结的 candidate opportunity 处才：

- 保存完整 optimizer checkpoint；
- 提取正式 Behavior；
- 生成 independent Query；
- 执行 No-query / Query 配对续跑；
- 生成 $G_{\mathrm{FE}}$。

离线 opportunity 数应远低于原生 update 数：

$$
N_{\mathrm{offline\ opportunity}}
\ll
U_r.
$$

## 3.4 时钟 D：在线 decision opportunity

在线 Controller 只在冻结的机会集合：

$$
\mathcal T_{\mathrm{online}}
\subseteq
\mathcal T_{\mathrm{offline}}
$$

上进行预测。

每条 run 至多触发一次 Query：

$$
N_{\mathrm{query},r}\le1.
$$

这一约束适用于：

- Proposed；
- Always Query after probe；
- Random Query；
- Fuzzy Gate；
- 所有 first-trigger baseline。

---

# 4. 采样频率修订的总体裁决

## 4.1 可以沿用的部分

以下设计继续沿用：

1. 每次原生 optimizer update 更新轻量 Behavior 历史；
2. 完整 checkpoint 只在稀疏候选状态保存；
3. 使用 FE ratio 而不是 iteration 作为跨算法主尺度；
4. 使用 milestone + behavior event 的混合 opportunity generator；
5. 所有策略共享相同的候选机会集合；
6. 每条 run 最多执行一次 Query；
7. 模型选择、阈值拟合和评价采用 run-level first-trigger；
8. BBOB validation、CEC 和工程问题不得修改 decision schedule。

## 4.2 必须修改的部分

1. 旧 $[0.20,0.60]$ 状态区间必须在新 $G_{\mathrm{FE}}$ 标签下重新验证；
2. 在正式全量采集前增加早期与晚期 Pilot 候选点；
3. 采样范围选择必须依据功效质量覆盖，而不是仅依据正标签行数；
4. milestone + event generator 必须与 milestone-only、equal-count fixed grid 比较；
5. online schedule 和 threshold 必须按 run-level OOF 目标联合冻结；
6. trajectory-based zero-FE descriptor 必须在原始采集阶段可构造；
7. 所有状态必须有明确且唯一的 `decision_opportunity_index`；
8. 同一 FE 上的 milestone 与多个 event 应合并为一个状态；
9. 不同 run 的状态数不能通过逐行损失无意中改变模型权重；
10. 若采用 Skip–Defer–Query，必须定义不同复查节奏。

---

# 5. 新标签下的采样范围 Pilot

## 5.1 为什么必须重新验证范围

旧采样范围主要来自旧 Utility 下的机会分布。新方案改变了：

- 主标签不再扣除 wall-clock；
- 使用 benchmark-reference gap 与 log-ratio；
- Query功效按等总 FE 终端结果定义；
- continuation、Behavior与Selector均已重构。

因此，旧结论：

```text
高价值机会主要位于 0.30–0.55
```

不能直接迁移到新的 $G_{\mathrm{FE}}$。

## 5.2 Pilot候选范围

建议小规模 Pilot 使用：

$$
\mathcal T_{\mathrm{pilot}}^{\mathrm{milestone}}
=
\{
0.10,
0.15,
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
0.60,
0.70
\}.
$$

这一网格不是正式论文最终 schedule，而是用于验证正式覆盖边界。

## 5.3 Pilot函数与运行规模

建议：

```text
BBOB functions: F1, F3, F15, F24
Dimension: D=10
Instances: 1
Seeds: 1,2,3
Prefix algorithms: DE, PSO, CMA-ES, SHADE
Query: 主 cheap query
Opportunity: expanded milestones + frozen event generator
```

函数作用：

- F1：简单可分离单峰；
- F3：非线性单峰/低条件复杂性代表；
- F15：多峰代表；
- F24：复杂组合结构代表。

Pilot的目标不是估计论文最终性能，而是检查采样覆盖与实现一致性。

## 5.4 采样范围覆盖指标

对 run $r$，全 Pilot 候选范围中的最佳正功效为：

$$
G_{r,\mathrm{pilot}}^+
=
\max
\left(
0,
\max_{t\in\mathcal T_{\mathrm{pilot}}}
G_{r,t}
\right).
$$

对候选正式范围 $\mathcal T$：

$$
G_{r,\mathcal T}^+
=
\max
\left(
0,
\max_{t\in\mathcal T}
G_{r,t}
\right).
$$

定义功效质量覆盖率：

$$
\boxed{
\mathrm{CoverageMass}(\mathcal T)
=
\frac{
\sum_r G_{r,\mathcal T}^+
}{
\sum_r G_{r,\mathrm{pilot}}^+
}
}
$$

定义正机会 run 覆盖率：

$$
\boxed{
\mathrm{CoverageRun}(\mathcal T)
=
\frac{
\#\{r:G_{r,\mathcal T}^+>\delta_{\mathrm{practical}}\}
}{
\#\{r:G_{r,\mathrm{pilot}}^+>\delta_{\mathrm{practical}}\}
}
}
$$

## 5.5 正式范围冻结规则

当前 $[0.20,0.60]$ 可继续使用，当且仅当训练 Pilot 支持：

$$
\mathrm{CoverageMass}([0.20,0.60])\ge0.95,
$$

且：

$$
\mathrm{CoverageRun}([0.20,0.60])\ge0.90.
$$

同时要求：

- 0.10–0.20 的正功效质量不形成稳定独立峰；
- 0.60–0.70 的正功效质量不形成稳定独立峰；
- 结论在单峰、多峰和复杂函数上不完全相反；
- 不使用 BBOB validation 或 CEC 决定正式范围。

上述 0.95 与 0.90 是项目预设判据，不是文献定理，必须在正式运行前冻结。

## 5.6 可能的冻结结果

### 结果 A：旧范围充分

正式保留：

$$
[0.20,0.60].
$$

### 结果 B：早期机会不可忽略

增加：

$$
0.10,
0.15.
$$

但应控制总机会数，可能删去冗余的 0.22 或 0.26。

### 结果 C：晚期机会不可忽略

增加：

$$
0.70.
$$

同时检查剩余预算是否足以支付 Query FE 与最小 continuation FE。

### 结果 D：机会主要由事件而非阶段决定

保留较稀疏 milestone，用 event opportunities 补充，但必须进行 equal-count 消融，防止事件规则成为隐藏主模型。

---

# 6. 正式离线机会生成协议

## 6.1 推荐起始 milestone schedule

若 Pilot 支持旧范围，正式主协议使用：

$$
\mathcal T_{M}
=
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

阶段划分：

```text
early: [0.20, 0.30)
mid:   [0.30, 0.50)
late:  [0.50, 0.60]
```

## 6.2 Monitor grid

Behavior event只在监测网格上判定：

$$
\rho
\in
\{0.20,0.21,\ldots,0.60\}.
$$

注意：

- monitor grid不是完整状态保存频率；
- monitor grid不是Controller调用频率；
- monitor grid只决定何时检查事件条件；
- 每个完整原生 update最多判定一次事件 crossing。

## 6.3 Event类型

冻结候选事件：

```text
improvement_resume
stagnation_onset
rank_change
elite_migration
diversity_recovery
```

每个事件必须：

- 只依赖 Query前可观测 Behavior；
- 不读取未来续跑结果；
- 不读取Selector所选动作；
- 不使用 function ID；
- 不使用 algorithm-specific control parameters；
- 采用边界归一化和尺度稳定的指标。

## 6.4 Event-only配额

建议：

```text
maximum event-only states per phase = 2
minimum event-only FE-ratio gap = 0.02
```

同一 update 同时满足 milestone 和 event 时：

- 合并为一个 opportunity；
- `sampling_triggers`保存全部触发原因；
- 不消耗 event-only配额；
- 不重复保存完整状态；
- 不生成多个相同 FE 的决策行。

## 6.5 每条run的状态数量

若保持12个 milestone，且每阶段最多2个 event-only状态，则：

$$
12
\le
N_{\mathrm{state},r}
\le
18.
$$

下游不得假设每条run有完全相同的状态数。

## 6.6 状态顺序

每条run必须保存：

```text
decision_opportunity_index
native_update_index
FE
FE_ratio
sampling_phase
sampling_triggers
is_budget_milestone
is_event_sample
```

排序固定为：

```text
FE, decision_opportunity_index
```

同一run不得存在两个不同决策状态共享完全相同的 `FE`。

---

# 7. 离线分叉频率与标签生成

## 7.1 每个候选状态生成的路径

方案 A 的最小正式分支为：

```text
shared prefix state s_t
    ├── No-query continuation
    └── Query q + downstream selector + continuation
```

每个状态至少保存：

```text
p_noquery_raw
p_query_raw
benchmark_reference_value
loss_noquery
loss_query
query_fe
remaining_fe_noquery
remaining_fe_query
selected_action
handoff_type
runtime components
```

并在下游派生：

```text
g_fe
meaningful_gain_label
```

## 7.2 不应在每次原生update分叉

若每个原生update都做Query/No-query续跑，成本约为：

$$
O
\left(
N_{\mathrm{run}}
\times
U_r
\times
N_{\mathrm{branch}}
\right),
$$

不仅计算量不可控，还会产生大量高度相关状态。

正式策略应保持：

$$
N_{\mathrm{branch\ state},r}
\approx12\text{--}18,
$$

而不是按每1% FE甚至每代分叉。

## 7.3 状态权重

一条run的状态数可能不同。训练时应至少提供两种权重口径：

### 行等权

$$
w_{r,t}^{\mathrm{row}}=1.
$$

### Run等权

$$
w_{r,t}^{\mathrm{run}}
=
\frac{1}{N_{\mathrm{state},r}}.
$$

主 first-trigger策略选择本身必须以run为单位，不得因某条run拥有更多event状态而获得更高权重。

行等权可保留为敏感性分析。

---

# 8. Trajectory-based zero-FE descriptor的采样频率

## 8.1 为什么必须同步保留

Trajectory-based algorithm selection使用优化器已经评价过的点计算landscape descriptors，从而避免额外Query FE。该基线直接挑战独立ELA的必要性。

正式trajectory采集结束后，如果只保留12–18个checkpoint population，就无法恢复所有曾评价但未留在population中的点。

因此，trajectory-based Query所需信息必须在原始运行时同步维护。

## 8.2 推荐实现一：确定性reservoir

维护最多：

$$
N_{\mathrm{reservoir}}=50d
$$

个历史评价点。

每次新评价：

```text
update deterministic reservoir
update count
update coverage metadata
```

要求：

- 使用独立semantic RNG；
- 不改变optimizer RNG；
- 同seed逐值可复现；
- 不增加目标函数评价；
- 不把整个reservoir复制到每个state row。

## 8.3 推荐实现二：在线累计描述符

对可增量更新的特征直接维护：

```text
count
mean
variance
quantiles / sketches
coordinate bounds
fitness distribution summaries
longitudinal phase summaries
```

在候选状态处输出trajectory descriptor。

## 8.4 保存频率

- reservoir/accumulator：每次函数评价更新；
- trajectory descriptor：每个离线candidate state输出；
- 完整historical points：可选，只在小规模验证子集保存。

---

# 9. 在线主策略的采样频率

## 9.1 主策略保持二元one-shot first-trigger

为避免同时修改标签、控制逻辑和机会生成器，方案 A 的第一轮正式主实验建议采用：

```text
Query
vs
Continue
```

给定score：

$$
z_{r,t}=f_\theta(B_{r,t}).
$$

首次触发时刻：

$$
t_r(\tau)
=
\min
\left\{
t:z_{r,t}>\tau
\right\}.
$$

若不存在：

$$
N_{\mathrm{query},r}=0.
$$

若存在：

$$
N_{\mathrm{query},r}=1.
$$

## 9.2 在线检查频率

主在线策略可以使用：

- 与正式离线相同的 milestone + event opportunities；
- 或从离线机会集合中选择train-only冻结的子集。

必须满足：

$$
\mathcal T_{\mathrm{online}}
\subseteq
\mathcal T_{\mathrm{offline}}.
$$

不允许在CEC上发现某些时刻表现更好后再增加检查点。

## 9.3 Schedule与threshold联合选择

正式策略不是只选择$\tau$，而是选择：

$$
(\mathcal T_D^*,\tau^*)
=
\arg\max_{\mathcal T_D,\tau}
\frac{1}{R}
\sum_{r=1}^{R}
C_r(\mathcal T_D,\tau),
$$

其中：

$$
C_r
=
\begin{cases}
G_{r,t_r}, & \text{首次触发},\\
0, & \text{未触发}.
\end{cases}
$$

建议约束：

$$
|\mathcal T_D|\le7.
$$

这可减少重复相关检查和累计误触发。

## 9.4 Train-only选择流程

```text
BBOB train family-grouped outer folds
    ↓
fit model on inner train families
    ↓
produce holdout-family scores
    ↓
simulate complete run-level first-trigger policy
    ↓
select schedule + threshold by OOF mean efficacy
    ↓
refit final model on all BBOB train
    ↓
freeze schedule + threshold
```

BBOB validation只评价，不参与：

- schedule选择；
- threshold选择；
- event阈值选择；
- practical efficacy boundary选择。

---

# 10. 高频检查为何可能有害

## 10.1 累计误触发

若每个机会的条件误触发概率约为$p_k$，则至少一次误触发概率为：

$$
P(\mathrm{false\ trigger})
=
1-
\prod_{k=1}^{K}
(1-p_k).
$$

即使单点错误率不高，增加$K$也会提高整条run的误调用概率。

## 10.2 状态相关性

相邻状态：

$$
B_{t+1}\approx B_t.
$$

过密检查通常增加的是重复信息，而不是独立证据。

## 10.3 Event机会的隐藏优势

milestone + event策略可能在Controller之前就过滤掉大量普通状态。

因此必须比较：

```text
milestones only
milestones + events
equal-count fixed grid
dense grid
```

若milestone + events显著更好，需要进一步判断：

- 是Controller更会判断；
- 还是event generator已经把高功效状态挑出来。

---

# 11. Opportunity频率敏感性实验

## 11.1 Sparse

约4个固定检查点，例如：

$$
\{0.20,0.30,0.42,0.55\}.
$$

## 11.2 Medium

主策略，约5–7个train-selected机会。

## 11.3 Dense

使用全部正式milestone + event机会，约12–18个/run。

## 11.4 Equal-count fixed

为每条run提供与event策略相同数量的固定机会，但不使用event规则。

## 11.5 每组必须重新校准

不同机会集合必须分别在BBOB train OOF上拟合：

- model；
- threshold；
- first-trigger policy。

不能把一个schedule下的threshold直接复制到另一个schedule。

## 11.6 报告指标

### 功效

- mean first-trigger $G_{\mathrm{FE}}$；
- median first-trigger $G_{\mathrm{FE}}$；
- $P(G_{\mathrm{FE}}>0)$；
- $P(G_{\mathrm{FE}}>\delta_{\mathrm{practical}})$；
- efficacy capture；
- harmful early-trigger cost。

### 策略

- Query call rate；
- mean trigger FE ratio；
- median trigger FE ratio；
- positive-opportunity run recall；
- helpful-call precision；
- no-call missed-positive run rate。

### 优化性能

- final log regret；
- anytime AUC；
- 与Never Query差异；
- 与Always Query等价性；
- 与Fixed-checkpoint Query差异。

### 资源

- Behavior extraction runtime；
- Controller inference runtime；
- Query FE；
- end-to-end runtime；
- peak RSS。

资源指标单独报告，不进入主$G_{\mathrm{FE}}$。

---

# 12. Skip–Defer–Query扩展中的频率修改

## 12.1 为什么不能让Skip和Defer相同

若两者都执行：

```text
当前不Query
下一个同样的checkpoint再次判断
```

则它们在策略执行上完全相同。

## 12.2 双频复查

### Query

```text
立即执行Query
后续不再检查
```

### Defer

```text
证据不足
在最近的下一个fine opportunity复查
```

### Skip

```text
当前证据明确不支持Query
进入cooldown
只在下一个coarse milestone或阶段边界复查
```

形式：

$$
t_{k+1}
=
\begin{cases}
\operatorname{next}_{\mathrm{fine}}(t_k),
& d_k=\mathrm{Defer},\\[4pt]
\operatorname{next}_{\mathrm{coarse}}(t_k),
& d_k=\mathrm{Skip}.
\end{cases}
$$

## 12.3 推荐fine/coarse定义

```text
fine opportunities:
all frozen milestones + accepted behavior events

coarse opportunities:
phase boundaries or at least 0.05 FE-ratio gap
```

## 12.4 第一篇论文的地位

- 二元first-trigger：主结果；
- Skip–Defer–Query：扩展/消融；
- Type-1 fuzzy：可解释增强；
- Interval Type-2：仅在边界不稳定被数据证实后再做。

---

# 13. 公平基线的频率对齐

所有在线baseline必须使用相同的机会集合，除非其定义本身发生在优化开始前。

## 13.1 Never Query

- 接收相同机会；
- 永不Query；
- 不因检查而改变optimizer trajectory。

## 13.2 After-probe Always Query

- 在第一个有效在线机会Query；
- 与Traditional AAS at start区分。

## 13.3 Fixed-checkpoint Query

- checkpoint只由BBOB train选择；
- validation/test不得修改。

## 13.4 Call-rate-matched Random

- 与Proposed总run-level调用率匹配；
- 每条run最多一次Query；
- 使用相同机会集合。

## 13.5 Trigger-time-matched Random

- 匹配Proposed的trigger FE-ratio分布；
- 排除收益仅来自“较晚调用”的可能性。

## 13.6 Opportunity-generator baseline

```text
milestone-only gate
milestone+event gate
equal-count fixed gate
```

必须使用相同模型容量和相同训练数据范围。

---

# 14. 数据契约

## 14.1 Native-update summary表

```text
run key
native_update_index
FE
FE_ratio
best_fitness
fitness_iqr
fitness_std
population_diversity
centroid summary
elite centroid summary
covariance trace
covariance effective rank
improvement count
stagnation length
trajectory reservoir count
```

## 14.2 Opportunity表

```text
run key
decision_opportunity_index
native_update_index
FE
FE_ratio
sampling_protocol
sampling_phase
sampling_triggers
is_budget_milestone
budget_milestone_ratio
is_event_sample
event flags
effective window support
```

## 14.3 完整checkpoint表或文件

```text
state key
population
fitness
best-so-far
optimizer internal state
RNG state
pending update state
```

## 14.4 方案A标签事实表

```text
state key
query_id
query_fe
remaining_fe_noquery
remaining_fe_query
p_noquery_raw
p_query_raw
benchmark_reference_value
loss_noquery
loss_query
selected_action
handoff_type
runtime components
```

## 14.5 派生表

```text
g_fe
g_fe_gt_zero
g_fe_gt_practical
predicted_efficacy
prediction interval
first-trigger decision
run-level policy contribution
```

## 14.6 禁止进入Decision X的字段

- function ID；
- family；
- dimension；
- algorithm ID；
- selected action；
- handoff outcome；
- Query features；
- action losses；
- $G_{\mathrm{FE}}$；
- benchmark reference value；
- future performance；
- event的未来结果解释字段。

允许进入Decision X的阶段信息仅限冻结的：

```text
FE_ratio
algorithm-agnostic behavior fields
readiness/maturity fields
```

---

# 15. 共享first-trigger实现规范

必须有唯一共享实现：

```python
def first_trigger_mask(frame, scores, threshold, run_key, order_columns):
    ...
```

该实现同时用于：

1. threshold sweep；
2. nested OOF model selection；
3. full-train OOF threshold freezing；
4. validation evaluation；
5. external online replay；
6. baseline comparison；
7. opportunity-frequency sensitivity。

## 15.1 必须满足的性质

对任意run：

$$
\sum_t\mathbb I(\mathrm{call}_{r,t})\le1.
$$

触发位置必须为：

$$
t_r
=
\min
\{t:z_{r,t}>\tau\}.
$$

## 15.2 评价分母

平均策略功效必须按run计算：

$$
J
=
\frac{1}{R}
\sum_{r=1}^{R}
C_r.
$$

不得按状态行计算：

$$
\frac{1}{N_{\mathrm{row}}}
\sum_{r,t}
G_{r,t}
\mathbb I(z_{r,t}>\tau).
$$

---

# 16. Efficacy capture与频率评价

对run $r$：

$$
G_r^+
=
\max
\left(
0,
\max_t G_{r,t}
\right).
$$

若策略在$t_r$触发：

$$
C_r
=G_{r,t_r};
$$

否则：

$$
C_r=0.
$$

## 16.1 Call rate

$$
\mathrm{CallRate}
=
\frac{
\#\{r:\text{called}\}
}{R}.
$$

## 16.2 Helpful precision

$$
\mathrm{Precision}
=
\frac{
\#\{r:\text{called and }C_r>\delta_{\mathrm{practical}}\}
}{
\#\{r:\text{called}\}
}.
$$

## 16.3 Positive-opportunity recall

$$
\mathrm{Recall}
=
\frac{
\#\{r:\text{called and }C_r>\delta_{\mathrm{practical}}\}
}{
\#\{r:G_r^+>\delta_{\mathrm{practical}}\}
}.
$$

## 16.4 Efficacy capture

$$
\mathrm{Capture}
=
\frac{
\sum_r\max(C_r,0)
}{
\sum_r G_r^+
}.
$$

分母必须与模型无关，不能根据每个模型的score排序重新定义。

## 16.5 Harmful early-trigger cost

$$
\mathrm{HarmfulCost}
=
\sum_r
\max(-C_r,0).
$$

若某run第一次触发为负，但后续存在正功效状态，还应报告：

```text
harmful_early_trigger_miss_runs
```

---

# 17. 实现一致性检查

## 17.1 轨迹被动观测检查

同seed下比较：

```text
uninterrupted run
vs
run with native-update observation
vs
run with milestone/event opportunity detection
```

必须逐值一致：

- final best；
- optimizer state；
- RNG state；
- native update count；
- FE count。

## 17.2 Window检查

对每个状态和窗口：

$$
W_{\mathrm{nominal}}
\le
W_{\mathrm{effective}}
<
W_{\mathrm{nominal}}
+
\frac{FE_{\mathrm{one\ native\ update}}}{B}.
$$

## 17.3 Opportunity唯一性检查

每条run：

```text
(run key, FE)
```

必须唯一。

## 17.4 First-trigger一致性检查

同一模型、同一threshold下：

```text
threshold sweep
nested OOF table
validation table
online simulator
```

必须产生相同触发行。

## 17.5 Query预算检查

每个状态：

$$
FE_{q,t}^{\mathrm{remain}}
=
B-b_t-c_q
\ge0.
$$

并设置最小 continuation 条件：

$$
B-b_t-c_q
\ge
FE_{\mathrm{continuation,min}}.
$$

## 17.6 标签重算检查

从事实字段独立重算：

$$
G_{\mathrm{FE}}.
$$

保存值必须与重算值逐行一致。

---

# 18. 小规模真实Pilot的GO / NO-GO

## 18.1 GO条件

正式启动全量BBOB前，至少满足：

1. milestone/event观测不改变原生轨迹；
2. 所有Behavior窗口均有真实native-update支撑；
3. expanded Pilot能够稳定估计早、中、晚功效分布；
4. 正式候选区间达到预设CoverageMass与CoverageRun；
5. first-trigger四处实现一致；
6. 每条run调用次数不超过1；
7. $G_{\mathrm{FE}}$可从事实字段重算；
8. trajectory descriptor可在0额外FE下构造；
9. sparse/medium/dense三种频率均可复现；
10. Query后剩余预算在所有候选点有效。

## 18.2 NO-GO条件

任一情况出现则暂停全量采集：

- 0.10–0.20或0.60–0.70包含大量稳定正功效，但正式区间未覆盖；
- event opportunity与milestone发生重复状态；
- 机会观测改变optimizer RNG；
- first-trigger在不同模块结果不一致；
- dense schedule因重复Query逻辑产生一条run多次调用；
- trajectory基线无法从正式采集数据构造；
- $G_{\mathrm{FE}}$依赖未保存的中间事实；
- 采样范围由validation或CEC结果反向确定。

---

# 19. 推荐配置草案

```yaml
sampling_protocol:
  id: efficacy_first_sampling_v1

trajectory_observation:
  cadence: every_complete_native_update
  intervention: false
  store_lightweight_summary: true

behavior_windows:
  fe_ratio:
    - 0.02
    - 0.05
    - 0.10
  store_effective_support: true
  insufficient_support_policy: missing

offline_range_pilot:
  enabled: true
  milestone_ratios:
    - 0.10
    - 0.15
    - 0.20
    - 0.22
    - 0.24
    - 0.26
    - 0.28
    - 0.30
    - 0.34
    - 0.38
    - 0.42
    - 0.46
    - 0.50
    - 0.60
    - 0.70
  coverage_mass_min: 0.95
  coverage_run_min: 0.90

formal_opportunities:
  milestone_ratios:
    - 0.20
    - 0.22
    - 0.24
    - 0.26
    - 0.28
    - 0.30
    - 0.34
    - 0.38
    - 0.42
    - 0.46
    - 0.50
    - 0.60
  monitor_grid_start: 0.20
  monitor_grid_end: 0.60
  monitor_grid_step: 0.01
  events:
    - improvement_resume
    - stagnation_onset
    - rank_change
    - elite_migration
    - diversity_recovery
  max_event_only_per_phase: 2
  min_event_only_ratio_gap: 0.02
  merge_same_update_triggers: true

trajectory_query_baseline:
  enabled: true
  reservoir_size_per_dimension: 50
  extra_fe: 0
  separate_rng_stream: true

online_policy:
  semantics: one_shot_first_trigger
  max_query_calls_per_run: 1
  opportunity_source: frozen_train_protocol
  schedule_selection: grouped_oof_run_utility
  max_decision_checks: 7
  threshold_selection: grouped_oof_run_utility

external_test:
  refit_schedule: false
  refit_threshold: false
  refit_event_rules: false
```

---

# 20. 执行顺序

## Step 1：完成采样范围Pilot

- 扩展0.10–0.70；
- 运行最小真实BBOB子集；
- 计算CoverageMass与CoverageRun；
- 冻结正式范围。

## Step 2：冻结opportunity generator

- milestone规则；
- event规则；
- 配额；
- 间隔；
- 合并规则；
- 版本ID。

## Step 3：验证被动观测

- uninterrupted；
- observed；
- opportunity-monitored；
- 三者逐值一致。

## Step 4：实现trajectory zero-FE基线

- reservoir或incremental descriptor；
- independent RNG；
- 不改变轨迹；
- 状态级输出。

## Step 5：正式采集BBOB trajectories

- 54 train shards；
- 18 validation shards；
- 同时发布trajectory与final performance；
- 不在采集期间读取未完成shard。

## Step 6：生成方案A标签

- No-query路径；
- Query路径；
- benchmark gap；
- $G_{\mathrm{FE}}$；
- practical label。

## Step 7：训练与冻结在线schedule

- grouped nested OOF；
- run-level first-trigger；
- schedule + threshold联合选择；
- validation只评价。

## Step 8：频率消融

- milestones only；
- milestones + events；
- equal-count fixed；
- sparse/medium/dense。

## Step 9：外部测试

- CEC2017；
- CEC2022；
- MA-BBOB；
- 任何外部suite均不得重调频率。

---

# 21. 论文中的推荐表述

## 21.1 方法部分

> Search behavior is monitored after every complete native optimizer update, whereas expensive paired continuation labels are generated only at a sparse set of frozen budget milestones and behavior-triggered opportunities. The online controller is evaluated under a one-shot first-trigger policy, and both the opportunity schedule and score threshold are selected exclusively from grouped out-of-fold predictions on the training function families.

## 21.2 采样范围说明

> The formal opportunity range is not inherited from withdrawn preliminary utility results. We first conduct a train-only coverage pilot over an expanded FE-ratio range and retain the smallest opportunity interval that captures a prespecified fraction of the positive efficacy mass and positive-opportunity runs.

## 21.3 消融说明

> To separate the contribution of the learned controller from that of the hand-crafted opportunity generator, we compare milestone-only, milestone-plus-event, equal-count fixed-grid, and dense opportunity protocols under independently recalibrated run-level first-trigger policies.

---

# 22. 参考文献与思路来源

## [R1] Trajectory-based Algorithm Selection with Warm-starting

Jankovic, A., Vermetten, D., Kostovska, A., de Nobel, J., Eftimov, T., & Doerr, C. (2022). *Trajectory-based Algorithm Selection with Warm-starting*. IEEE Congress on Evolutionary Computation.

支持内容：

- 使用初始算法轨迹中的已评价样本减少独立特征采样开销；
- 固定A1预算后进行一次算法选择与warm-start；
- fixed-budget target precision与log-performance评价；
- 自适应切换时机被列为后续方向。

不直接支持：

- 本文的milestone+event机会生成器；
- $G_{\mathrm{FE}}$定义；
- train-only功效质量覆盖准则。

## [R2] Per-run Algorithm Selection with Warm-Starting Using Trajectory-Based Features

Kostovska, A., Jankovic, A., Vermetten, D., de Nobel, J., Wang, H., Eftimov, T., & Doerr, C. (2022). *Per-run Algorithm Selection with Warm-Starting Using Trajectory-Based Features*. PPSN 2022.

支持内容：

- default optimizer prefix；
- trajectory features与时间序列特征；
- 不同后续预算下最优算法会变化；
- 跨benchmark轨迹覆盖不足会导致泛化下降。

不直接支持：

- 本项目的动态event采样；
- first-trigger efficacy threshold；
- 0.20–0.60正式区间。

## [R3] DynamoRep

Cenikj, G., Petelin, G., Doerr, C., Korošec, P., & Eftimov, T. (2023). *DynamoRep: Trajectory-Based Population Dynamics for Classification of Black-box Optimization Problems*.

支持内容：

- longitudinal trajectory statistics包含problem–algorithm interaction信息；
- 轨迹统计可在0额外FE下提取；
- 将所有轨迹点静态混合可能丢失时间结构；
- 初始多次迭代的统计可用于低成本表示。

不直接支持：

- 本文的功效标签；
- opportunity schedule；
- 每条run一次Query协议。

## [R4] Hyperparameter Control Using Fuzzy Logic

Roy, N., Beauthier, C., & Mayer, A. (2025). *Hyperparameter Control Using Fuzzy Logic: Evolving Policies for Adaptive Fuzzy Particle Swarm Optimization Algorithm*. Evolutionary Computation, 33(2), 279–308.

支持内容：

- 反馈probes应数量少、非冗余且有明确意义；
- probes应归一化并避免原始fitness尺度；
- 控制规则可以在训练benchmark上系统优化；
- 复杂控制器可能过拟合。

不直接支持：

- 用模糊控制定义真实ELA功效；
- 本文二元first-trigger主协议；
- 本文的采样覆盖指标。

## [R5] ELA Sampling Sensitivity

Renau, Q., Doerr, C., Dreo, J., & Doerr, B. (2020). *Exploratory Landscape Analysis Is Strongly Sensitive to the Sampling Strategy*. PPSN 2020.

支持内容：

- ELA特征对采样方法和样本规模敏感；
- Landscape Query配置必须版本化、冻结并进行敏感性分析。

---

# 23. 最终冻结建议

方案 A 下，采样频率不应整体推翻，而应按以下结论修订：

1. **原生轨迹观测保持每次完整optimizer update一次。**
2. **Behavior窗口从高频native-update历史中精确计算。**
3. **完整状态与昂贵配对标签只在12–18个稀疏milestone/event机会生成。**
4. **在正式全量运行前，用0.10–0.70小规模Pilot重新验证旧的0.20–0.60区间。**
5. **正式在线主策略继续使用run-level one-shot first-trigger。**
6. **在线schedule和threshold只用BBOB train grouped OOF结果冻结。**
7. **milestone+event必须与milestone-only和equal-count fixed grid做公平消融。**
8. **Trajectory-based zero-FE descriptor必须在原始采集阶段同步保留。**
9. **不同机会频率必须分别重新校准，不能共享同一threshold。**
10. **Skip–Defer–Query只作为扩展，并为Skip与Defer定义不同复查频率。**

一句话概括：

> **方案 A 不要求增加昂贵标签的密度，而要求提高被动观测的时间分辨率、重新验证候选区间，并确保在线决策频率由run-level功效而不是旧Utility或人工直觉决定。**
