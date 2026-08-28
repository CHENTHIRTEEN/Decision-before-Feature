# Decision-before-Feature 改进方案：从 G_FE 二元门控到行为–景观协同的动态算法选择

> 文档性质：研究方案重构草案  
> 适用项目：`CHENTHIRTEEN/Decision-before-Feature`  
> 形成依据：当前仓库 `main` 分支的方法、协议与代码；今日讨论的 AS-LGBM / Soft-ERT、trajectory-based algorithm selection、DynamoRep、per-run warm-starting、switch-benefit prediction、probing trajectories 与 RL-DAS 等工作。  
> 目标：保留当前项目已经完成且有价值的基础设施，修正 `G_FE -> 是否 Query/是否切换 -> Selector` 两级串联系统的目标错位，将研究问题重构为 **行为优先、在线景观估计、动作条件化、查询按需、可重复决策的动态算法选择**。

---

# 1. 结论先行

当前项目已经具备相当完整的实验基础，包括：

- DE、PSO、CMA-ES、SHADE 的完整状态推进与 checkpoint continuation；
- 跨算法 `population_transfer_initialization`；
- permutation-invariant 的 Behavior representation；
- 多窗口 improvement、diversity、fitness distribution、population motion、effective rank、stagnation 等行为特征；
- 动态 decision opportunity 采样；
- 同一共享状态下的多动作真实 continuation；
- Behavior-only、Query、State-only 等 Selector 路径；
- grouped-by-function 的学习与评价协议。

这些资产不应该推倒重来。

需要重构的主要不是 optimizer 或 Behavior，而是**决策问题本身**。

当前主逻辑大体可概括为：

```text
当前优化状态
    ↓
Behavior
    ↓
G_FE predictor
    ↓
“Query / 不 Query”
    ↓
若 Query
    ↓
提取 landscape descriptors
    ↓
Selector
    ↓
继续当前算法 / 切换算法
```

这套设计的问题在于：`G_FE` 是一个**完整路径的结果变量**，而不是“候选算法动作的价值”。它同时受到 Query FE、Query sample 是否直接找到好点、descriptor 是否有增量信息、Selector 是否选对、population transfer 是否适配、剩余预算等因素影响。

因此：

> **“是否值得 Query”与“是否值得切换算法”不应由同一个 G_FE 二元门控变量承担。**

建议将主研究问题改为：

> **在优化运行过程中，能否利用算法行为与逐步更新的景观信息，对每个候选算法动作的未来收益进行方向性判断，并仅在当前信息不足时支付额外 FE 获取景观 Query，从而实现成本感知的动态算法选择？**

新的核心流程应为：

```text
优化状态 S_t
   │
   ├── Behavior B_t                         0 额外 FE
   │
   ├── Trajectory / Local Landscape L_t    0 额外 FE
   │
   └── Uncertainty U_t                     0 额外 FE
             │
             ▼
      Action-value predictor
             │
   ┌─────────┼──────────┐
   ▼         ▼          ▼
continue   switch     information insufficient
current     A_j              │
                            ▼
                      optional Query
                            │
                            ▼
                    update landscape belief
                            │
                            ▼
                    re-rank candidate actions
                            │
                            ▼
                       execute action
                            │
                            ▼
                  next decision opportunity
```

论文核心不再是“Feature 前要不要 Feature”，而升级为：

> **Behavior-first Dynamic Algorithm Selection with Adaptive Landscape Information Acquisition**

更保守、也更准确的中文定位可以写成：

> **行为–景观协同的成本感知动态算法选择**

---

# 2. 为什么当前 G_FE 两阶段架构存在结构性问题

## 2.1 G_FE 的语义不是“切换收益”

当前代码中的基础 Efficacy 定义为：

$$
G_{\mathrm{FE}}
=
\log
\frac{E_{\mathrm{skip}}+\epsilon_p}
     {E_{\mathrm{query}}+\epsilon_p}.
$$

它比较的是：

- 一条 no-query / skip 路径；
- 一条 query 路径。

而 query 路径内部还可能包含：

1. Query 消耗的 FE；
2. Query sample 自身直接发现更好点；
3. landscape descriptor；
4. Query-enhanced Selector；
5. Selector 选择的新算法；
6. 跨算法 population transfer；
7. 更少的剩余 optimization FE。

因此：

$$
G_{\mathrm{FE}}
\neq
\text{switch benefit}.
$$

更准确地说：

$$
G_{\mathrm{FE}}
=
\text{acquisition}
+
\text{sample endpoint}
+
\text{information}
+
\text{selection}
+
\text{handoff}
+
\text{budget effect}
+
\text{stochasticity}.
$$

如果直接用它回答“现在是否值得切算法”，目标已经混杂。

## 2.2 Gate 与 Selector 的学习目标不一致

当前两级系统相当于：

### Gate

预测：

$$
P(G_{\mathrm{FE}}>0\mid B_t).
$$

### Selector

预测：

$$
a_t^\star
=
\arg\min_a \hat L_{t,a}.
$$

Gate 并不知道最终 Selector 会选择哪个候选动作，也不知道不同动作的收益差异。

例如当前状态下：

| 动作 | 未来损失 |
|---|---:|
| continue DE | 1.00 |
| PSO | 1.02 |
| CMA-ES | 0.35 |
| SHADE | 0.97 |

真正值得预测的是：

$$
G_t(\mathrm{CMAES})
=
L_t(\mathrm{continue})-L_t(\mathrm{CMAES}),
$$

而不是一个与具体 action 无关的 `G_FE`。

否则 Gate 学到的只是：

> “历史上类似状态中，整个 Query+Selector pipeline 最后有没有占便宜。”

这和：

> “如果现在切到 CMA-ES，会不会提升？”

不是同一个问题。

## 2.3 两级串联存在误差级联

当前策略成功需要同时满足：

1. G_FE predictor 判断 Query 值得；
2. Query descriptors 有效；
3. Selector 选出好动作；
4. warm-start / population transfer 成功；
5. 剩余预算足以兑现切换收益。

任何一层出错都会让整个状态被标成 Query 无效。

于是 Decision Model 容易把 Selector 的错误当成“ELA 不值得”。

这使第一层模型的科学解释很困难：

$$
\text{Gate error}
\quad\text{vs}\quad
\text{Selector error}
\quad\text{vs}\quad
\text{Query information failure}
$$

无法天然分离。

## 2.4 当前项目已经拥有 Behavior-only Selector，这反而说明应该先做动态动作预测

仓库目前已经有：

- `query_full`
- `state_only`
- `behavior_only_full_budget`
- `pre_run_query_only`

等 Selector 输入模式。

这意味着项目实际上已经具备回答更基础问题的条件：

> **只用 Behavior，能否直接预测当前状态下各候选算法的未来收益？**

如果答案是“能”，那么先预测 `G_FE` 再决定是否调用 Selector，就绕了一圈。

更自然的顺序应变为：

$$
B_t
\rightarrow
\{G_t(a)\}_{a\in\mathcal A}
\rightarrow
\text{动作决策}.
$$

只有当：

$$
\max_a \hat G_t(a)
$$

的置信度不足，才考虑获取额外 landscape information。

## 2.5 固定 Query 与持续累积的搜索信息割裂

当前独立 Query 的优点是接近 global landscape probe，但它把 landscape information 设计为“有或无”的离散事件。

事实上运行过程中已经不断产生：

$$
\{x_i,f(x_i)\}_{i\le t}.
$$

这些数据可以持续更新很多景观统计量：

- fitness distribution；
- local meta-model；
- covariance spectrum；
- dispersion；
- local FDC；
- information content；
- local level-set separability；
- approximate NBC；
- uncertainty。

因此应该先使用：

$$
\text{trajectory-derived landscape information}
$$

再判断是否需要：

$$
\text{independent global query}.
$$

也就是：

> **Query 应该补信息，不应该负责从零制造 landscape information。**

## 2.6 One-shot first-trigger 限制了“动态算法选择”的含义

当前每条 run 最多 Query 一次，这对于第一篇 Decision-before-Feature 论文是合理的控制实验，但对于 DAS 来说过于受限。

已有 trajectory switching 和 RL-DAS 工作都指向同一个事实：

> 某个算法在搜索早期、中期和后期的相对优势可能不同。

因此最终系统应允许：

$$
A_1
\rightarrow
A_3
\rightarrow
A_3
\rightarrow
A_2
$$

而不是只有：

$$
A_1
\rightarrow
A_2.
$$

不过**不建议第一阶段直接上 repeated DAS**。应先证明 action-conditioned behavior prediction 成立，再逐步扩展。

## 2.7 G_FE 当前文档口径需要重新统一

当前仓库不同文件对 `G_FE` 的描述存在值得统一的地方：

- `utility_labels/efficacy.py` 将 G_FE 描述为严格等总 FE 下的性能功效，并明确 runtime 不进入主标签；
- 当前部分数学/Utility 文档中仍存在带 runtime 项的 G_FE / Utility 表述；
- AGENTS 中又明确规定 action-loss 科学标签不得混入 wall-clock。

建议在新方案中彻底拆开：

### Scientific performance gain

只用 FE-indexed optimization performance：

$$
G^{\mathrm{perf}}.
$$

### Runtime / deployment overhead

单独报告：

$$
C^{\mathrm{time}}.
$$

### Decision utility

只有在明确给出偏好权重时才计算：

$$
U
=
G^{\mathrm{perf}}
-
\lambda C^{\mathrm{time}}.
$$

不要继续让 `G_FE` 同时承担“性能事实”和“资源偏好”。

---

# 3. 新的研究问题

建议将研究问题分为四层。

## RQ1：Behavior 是否足以预测动态算法动作的未来收益？

$$
P
\left(
G_{t,H}(a)>\delta
\mid B_t
\right)
$$

其中：

- $B_t$：当前算法无关 Behavior；
- $a$：候选后续算法；
- $H$：未来评价 horizon；
- $\delta$：最小实际意义改善。

这是必须最先验证的核心假设。

如果这一层不成立，后续所有复杂融合都没有意义。

## RQ2：运行轨迹中的在线景观信息能否在 Behavior 之外提供增量价值？

定义：

$$
L_t^{local}
=
\text{landscape statistics estimated from trajectory}.
$$

比较：

$$
P(G>0\mid B_t)
$$

与：

$$
P(G>0\mid B_t,L_t^{local}).
$$

真正的问题不是“ELA 准不准”，而是：

$$
I(Y;L_t^{local}\mid B_t)
$$

是否明显大于 0。

也就是：

> **知道 Behavior 后，在线 landscape information 还能增加多少决策信息？**

## RQ3：什么时候值得额外做独立 Landscape Query？

在已有：

$$
(B_t,L_t^{local},U_t)
$$

的情况下，Query 不再是默认动作，而是一个 information-acquisition action。

定义 Query 前决策损失：

$$
R_t^{pre}.
$$

Query 后重新选动作得到：

$$
R_t^{post}.
$$

则 Query 的操作性价值可写为：

$$
V_q
=
R_t^{pre}
-
R_t^{post}
-
C_q.
$$

其中 $C_q$ 应至少包含：

- Query 占用的 FE 所造成的 optimization opportunity cost；
- 可选的实际 runtime 维度。

只有：

$$
E[V_q\mid s_t]>0
$$

时才 Query。

## RQ4：行为–景观协同策略能否支持 repeated DAS？

在每个 decision interval：

$$
s_t
\rightarrow
a_t
\rightarrow
s_{t+1}
\rightarrow
a_{t+1}.
$$

最终研究：

$$
\pi(a_t\mid B_t,L_t,U_t,B_{\mathrm{remain}}).
$$

这里才是真正意义上的 repeated dynamic algorithm selection。

---

# 4. 新的状态表示

建议将状态拆成三条信息流。

## 4.1 Stream A：算法行为 Behavior

保留当前 `behavior/features.py` 已经实现的 permutation-invariant 体系。

当前已有的重要字段包括：

### Improvement / stagnation

- `bf_improvement_rate_w02`
- `bf_improvement_frequency_w02`
- `bf_best_fitness_slope_rel_w05`
- `bf_stagnation_w10`
- `bf_convergence_rate_w10`

### Population diversity

- `bf_diversity_mean_pairwise`
- `bf_diversity_change_w05`
- `bf_diversity_slope_w05`
- `bf_diversity_recovery_w05`

### Distribution dynamics

- `bf_fitness_quantile_improvement_fraction_w02`
- `bf_fitness_distribution_improvement_rate_w02`
- `bf_fitness_wasserstein_rate_w02`
- `bf_fitness_spread_slope_w05`

### Population motion

- `bf_population_wasserstein_rate_w05`
- `bf_centroid_shift_rate_w05`
- `bf_centroid_shift_coherence_w05`
- `bf_population_chamfer_distance_w05`

### Geometry

- `bf_covariance_spectral_concentration`
- `bf_covariance_effective_rank_w05`
- `bf_covariance_effective_rank_change_w05`

这一部分非常适合直接作为新方案的 Behavior backbone，不建议重做。

---

# 5. Stream B：无额外 FE 的在线景观估计

这一层是今天讨论后建议新增的核心模块。

不要把它称为 traditional/global ELA。

建议命名：

> **Trajectory-derived Local Landscape State**

或者简写：

$$
L_t^{local}.
$$

它只使用已经被 optimizer 评价过的点，因此：

$$
FE_{\mathrm{extra}}=0.
$$

## 5.1 第一类：可以精确递推的 Streaming Features

### Fitness distribution

在线更新：

$$
\mu_t,\quad
\sigma_t,\quad
\mathrm{skew}_t,\quad
\mathrm{kurt}_t.
$$

同时维护：

$$
Q_{0.1},Q_{0.25},Q_{0.5},Q_{0.75},Q_{0.9}.
$$

推荐字段：

```text
lf_y_mean
lf_y_std
lf_y_skew
lf_y_kurtosis
lf_y_q10
lf_y_q50
lf_y_q90
```

注意最好对 fitness 做当前协议一致的 scale normalization。

## 5.2 Online Meta-model

利用 trajectory reservoir 在线拟合：

### Linear

$$
\hat f(x)
=
\beta_0+\beta^\top x.
$$

### Quadratic

$$
\hat f(x)
=
\beta_0+\beta^\top x+x^\top Qx.
$$

在线记录：

```text
lf_linear_r2
lf_quadratic_r2
lf_quadratic_gain_over_linear
lf_model_residual_rel
lf_design_condition
```

与传统一次性 ELA 最大不同是，还可以利用：

$$
\Delta R^2_t
=
R^2_t-R^2_{t-w}.
$$

这反映当前访问区域是否逐渐变得可建模。

## 5.3 Incremental PCA / covariance structure

已有 Behavior 已经使用 covariance spectrum，因此这里不要重复同义特征。

Local landscape 模块只保留与目标值关联后确有额外意义的结构，例如：

- weighted PCA；
- elite vs whole covariance difference；
- fitness-conditioned effective dimension。

否则纯 X-space PCA 很容易只是重复 population geometry。

## 5.4 Rolling Information Content

根据最近窗口的 fitness change：

$$
\Delta f_i=f_{i+1}-f_i
$$

构造符号序列：

$$
s_i\in\{-1,0,+1\}.
$$

计算：

- transition entropy；
- sign-change rate；
- local ruggedness proxy。

这一类特别适合 online window。

---

# 6. Stream C：Reservoir + Monte Carlo / Bootstrap 的结构景观估计

对难以完全递推的 ELA 类特征，不建议每次重新计算全部历史点。

维护固定大小：

$$
M
$$

的 trajectory reservoir。

例如：

$$
M=200
$$

或按维度：

$$
M=c\,d.
$$

候选机制：

- uniform reservoir sampling；
- elite + random stratified reservoir；
- recency-weighted reservoir；
- diversity-preserving reservoir。

第一阶段推荐最简单、最可解释的：

> **50% uniform historical + 25% recent + 25% elite**

作为预注册 sensitivity，而不是一开始优化 reservoir policy。

## 6.1 Approximate FDC

在 reservoir 上计算当前参考点的 distance-fitness association。

需要强调：

> trajectory FDC 不是 global FDC。

命名应为：

```text
lf_local_fdc
```

而不是 `ELA_FDC`。

## 6.2 Approximate Dispersion

比较 elite subset 与 reservoir 的距离分布：

$$
D_{\mathrm{elite}}
\quad\text{vs}\quad
D_{\mathrm{all}}.
$$

输出：

```text
lf_elite_dispersion_ratio
lf_elite_dispersion_shift
```

## 6.3 Approximate NBC

在固定 reservoir 上使用 kNN / nearest-better graph：

```text
lf_nbc_mean_distance
lf_nbc_distance_cv
lf_nbc_cluster_proxy
```

第一阶段不建议追求完整 flacco NBC 等价实现。

---

# 7. 在线特征必须带 uncertainty

这是与传统 ELA 形成区别的关键。

对 reservoir 做 bootstrap：

$$
R_t^{(1)},\ldots,R_t^{(K)}.
$$

得到：

$$
\phi_t^{(1)},\ldots,\phi_t^{(K)}.
$$

然后记录：

$$
\mu_{\phi,t}
=
\frac1K
\sum_k \phi_t^{(k)},
$$

$$
\sigma_{\phi,t}
=
\sqrt{
\frac1{K-1}
\sum_k
(\phi_t^{(k)}-\mu_{\phi,t})^2
}.
$$

于是状态中不仅有：

> “local FDC = 0.42”

还包含：

> “local FDC = 0.42 ± 0.18”。

建议至少给关键在线特征记录：

```text
lf_*_mean
lf_*_std
lf_*_ci_width
```

不需要对所有特征都 bootstrap，优先：

- local FDC；
- dispersion；
- meta-model R²；
- information content。

---

# 8. 新的动作定义：先预测“动作是否有提升”

这是整个重构最关键的一步。

当前状态：

$$
S_t.
$$

动作集保持：

$$
\mathcal A_t
=
\{
\mathrm{continue\_current},
\mathrm{DE},
\mathrm{PSO},
\mathrm{CMAES},
\mathrm{SHADE}
\}
$$

并去掉和 current algorithm 重复的一个 switch action。

从同一完整 checkpoint 分支运行。

## 8.1 动作收益标签

对候选 action $a$：

$$
G_{t,H}(a)
=
\log
\frac{
E_{t,H}(\mathrm{continue})+\epsilon_t
}{
E_{t,H}(a)+\epsilon_t
}.
$$

解释：

- $G>0$：切到 $a$ 优于继续当前算法；
- $G=0$：等价；
- $G<0$：切换更差。

这里的 $H$ 不要只设一个。

推荐：

$$
H\in
\{0.1B,\ 0.2B,\ B_{\mathrm{remain}}\}.
$$

分别表示：

- short-term gain；
- medium-term gain；
- terminal gain。

## 8.2 不建议主任务精确回归 G

第一阶段主任务建议三分类：

$$
Y_{t,H,a}
=
\begin{cases}
+1, & G_{t,H}(a)>\delta_H,\\
0,  & |G_{t,H}(a)|\le\delta_H,\\
-1, & G_{t,H}(a)<-\delta_H.
\end{cases}
$$

即：

```text
Improve
Equivalent
Degrade
```

其中：

$$
\delta_H
$$

由 paired continuation noise 与 practical effect threshold 共同确定。

这比精确预测：

$$
\hat G=0.173621
$$

更符合实际决策目标。

---

# 9. Repetition 设计：保留当前 paired continuation 思路

当前 `efficacy.py` 已经加入：

- median；
- std；
- CI；
- sign flip rate。

这一思想应该从 G_FE 扩展到每个 action gain：

```text
action_gain_rep_1
action_gain_rep_2
action_gain_rep_3
action_gain_median
action_gain_std
action_gain_ci_low
action_gain_ci_high
action_gain_sign_flip_rate
```

当：

$$
CI
$$

跨过 practical-equivalence 区间时，可以直接给：

```text
uncertain
```

而不是强行制造标签。

---

# 10. Soft-ERT 应该怎么接入

不建议第一阶段直接把当前所有 gap labels 全部换成 Soft-ERT。

原因：

1. BBOB 已知 optimum，当前 gap label 更直接；
2. dynamic Soft-ERT 需要重新定义 state/horizon target；
3. Soft-ERT 本身是 portfolio-relative；
4. 加入/删除候选算法会改变 soft target；
5. 在短 horizon 上 hitting-time 可能很稀疏。

因此建议分两步。

## 10.1 第一阶段主标签

使用当前最成熟的 equal-FE relative log-gap action gain：

$$
G_{t,H}(a).
$$

## 10.2 第二阶段 unknown-optimum sensitivity

构造 Dynamic Soft-ERT。

对共享状态 $S_t$、未来 horizon $H$、所有 action 与 repetition，定义：

$$
f^{soft}_{t,H}
=
\min_{a,r,\tau\le H}
f_{a,r,t+\tau}.
$$

再定义宽松 target：

$$
f^{target}_{t,H}
=
f^{soft}_{t,H}
+
\epsilon_{t,H}.
$$

对每个 action 统计 reaching target 的 FE，得到：

$$
SoftERT_{t,H}(a).
$$

最后定义：

$$
G^{SE}_{t,H}(a)
=
SoftERT_{t,H}(\mathrm{continue})
-
SoftERT_{t,H}(a).
$$

模型仍只预测：

```text
Improve / Equivalent / Degrade
```

而不回归 Soft-ERT 数值。

## 10.3 Soft-ERT 的研究作用

如果：

$$
\operatorname{sign}
G_{t,H}(a)
$$

与：

$$
\operatorname{sign}
G^{SE}_{t,H}(a)
$$

在 BBOB 上有较高一致性，并且 Dynamic Soft-ERT 在 unknown-optimum benchmark / engineering problems 上仍可构造，那么可以形成第二层贡献：

> **不依赖已知最优值的动态 action-superiority label。**

这比把 Soft-ERT 直接塞进主模型安全得多。

---

# 11. Selector 应改成 Action-conditioned Predictor

当前 Selector 是 action-loss regression。

新方案中仍然可以保留 RF / LightGBM 等 tabular model 作为下游 predictor，但 target 改成动作相对收益。

两种实现都可以。

## 11.1 方案 A：Per-action model

对每个 target algorithm 训练：

$$
M_a(B_t,L_t,B_{\mathrm{remain}})
\rightarrow
P(Y_{t,a}=+1).
$$

优点：

- 不需要把 algorithm ID 作为 feature；
- 保持 Behavior 的 algorithm-independent 表述；
- 容易解释各算法需要什么状态；
- 适合当前四算法小 portfolio。

第一篇建议优先采用。

## 11.2 方案 B：Unified action-conditioned model

输入：

$$
[B_t,L_t,B_{\mathrm{remain}},e(a)]
$$

其中：

$$
e(a)
$$

是 action ID / embedding。

优点：

- 更容易扩展大量算法；
- 能共享统计强度。

缺点：

- action identity 会成为显式 predictor；
- 更容易学到算法先验捷径；
- OOD 到新算法比较困难。

第一篇不建议作为主方法。

---

# 12. Query 不再是 Gate，而是 Information Acquisition Action

这是对 Decision-before-Feature 原始思想最重要的保留和升级。

原逻辑：

```text
Behavior -> Query?
```

新逻辑：

```text
Behavior + online local landscape
       ↓
action prediction
       ↓
是否已经足够确定？
       │
   ┌───┴───┐
   │       │
  是       否
   │       │
直接选    评估 Query VOI
动作       │
       ┌───┴───┐
       │       │
    VOI<=0   VOI>0
       │       │
    不 Query  Query
```

---

# 13. Query Value of Information

设 Query 前：

$$
\hat a^{pre}
=
\arg\max_a
P(Y_{t,a}=+1\mid B_t,L_t).
$$

Query 后：

$$
\hat a^{post}
=
\arg\max_a
P(Y_{t,a}=+1\mid B_t,L_t,\Phi_q).
$$

离线可直接计算 Query 是否改变了动作以及改变后是否减少 observed action loss。

定义信息增量：

$$
V^{info}_{t,q}
=
L_{t,\hat a^{pre}}
-
L_{t,\hat a^{post}}.
$$

这一步必须在**相同 action budget matrix**上比较，从而隔离 descriptor 使用本身。

再定义 operation-level Query value：

$$
V^{op}_{t,q}
=
L^{noquery}_{t,\hat a^{pre}}
-
L^{query}_{t,\hat a^{post}}.
$$

后者才包含：

- Query FE；
- sample direct improvement；
- 更少剩余预算。

因此新系统应同时保留：

```text
query_information_increment
query_operational_increment
```

但不再用一个 `G_FE` 把二者合并成第一层 Gate 的唯一真值。

---

# 14. Query 触发条件

第一阶段不需要强化学习。

可以直接定义可解释规则。

例如：

### 条件 1：动作收益不确定

$$
\max_a P(+1)
<
\tau_p.
$$

### 条件 2：top-2 margin 太小

$$
P_1-P_2
<
\tau_m.
$$

### 条件 3：Online landscape uncertainty 较高

$$
U_t
>
\tau_u.
$$

### 条件 4：剩余预算足够覆盖 Query

$$
B_{\mathrm{remain}}
>
c_q+B_{\min}^{switch}.
$$

满足后再预测：

$$
P(V_q^{op}>0).
$$

只有超过阈值才 Query。

这样 Query 的科学意义变成：

> **解决动作选择不确定性。**

而不是：

> **先预测 ELA 有没有用，再希望 ELA 帮 Selector 选对。**

---

# 15. 第一篇新方案不要直接做 repeated switching

建议严格按阶段推进。

## Phase 0：保留现有项目作为对照

不要删除现有 G_FE 路径。

把它预先指定为：

```text
DBF-v1: G_FE Query Gate + Query Selector
```

作为 baseline。

## Phase 1：Behavior-only Switch Benefit Prediction

这是最重要的 feasibility test。

不使用 Query。

输入：

$$
B_t.
$$

预测：

$$
Y_{t,H,a}.
$$

要求回答：

> **在不做 ELA 的情况下，当前行为能否预测某个候选 solver 比继续当前 solver 更好？**

### 16.1 Baselines

至少包括：

1. Continue-current；
2. SBS；
3. Random switch；
4. Fixed-time 1-switch；
5. Existing Behavior-only action-loss RF；
6. To-Switch-style RF；
7. Time-only model；
8. B3 Behavior model。

### 16.2 Phase 1 Go / No-Go

只有满足以下条件才进入下一阶段：

- Behavior-only 在 grouped function OOD 上明显优于 time-only；
- action-gain sign AP / balanced accuracy 有稳定增益；
- selected action regret 显著低于 Continue / Random；
- 不同 dimension 不出现完全反向结论；
- paired repetitions 的 sign-flip rate 可接受。

如果 Phase 1 不成立，应停止扩展 online ELA，而不是继续堆模型。

---

# 17. Phase 2：加入无额外 FE 的 Local Landscape

比较：

```text
M0: time-only
M1: Behavior
M2: Local landscape only
M3: Behavior + Local landscape
M4: Behavior + Local landscape + uncertainty
```

核心检验：

$$
\Delta_{\mathrm{landscape}|B}
=
Metric(M3)-Metric(M1).
$$

如果：

$$
\Delta_{\mathrm{landscape}|B}
\approx0,
$$

那么说明 Behavior 已经吸收了大部分信息。

这是一个有意义的科学结论，不是“实验失败”。

---

# 18. Phase 3：Adaptive Query

此时才加入独立 LHS / descriptor Query。

比较：

```text
Never Query
Always Query
Fixed-time Query
Current G_FE Gate
Uncertainty Gate
VOI Query
```

主问题：

> 同样的 query rate 下，VOI Query 是否带来更低 action regret / terminal loss？

推荐使用 matched-query-rate comparison。

---

# 19. Phase 4：Repeated DAS

只有前三阶段有效才进入。

将一条 run 划为多个 action interval：

$$
[t_0,t_1),
[t_1,t_2),
\dots
$$

在每个机会：

1. 更新 Behavior；
2. 更新 Local Landscape；
3. 更新 uncertainty；
4. 预测 candidate action gain；
5. 必要时 Query；
6. 选择动作；
7. 执行一个 dwell interval。

## 19.1 防止算法抖动

必须增加：

### Minimum dwell time

$$
\Delta FE_{\min}.
$$

### Hysteresis

只有：

$$
\hat G(a_{new})-\hat G(a_{current})
>
\delta_{switch}
$$

才切换。

### Maximum switches

第一版建议：

$$
N_{switch}\le 3.
$$

### Switching cost

population transfer 本身可以进入实际 terminal performance，不必人为重复扣 FE。

wall-clock 额外单独报告。

---

# 20. Algorithm Context Restoration

当前仓库跨算法主要使用 population transfer。

Repeated DAS 后，建议增加可选的 algorithm context cache：

```text
context[DE]
context[PSO]
context[CMAES]
context[SHADE]
```

如果：

```text
CMAES -> DE -> CMAES
```

第二次回到 CMA-ES 时可以比较：

1. fresh transfer；
2. restore previous CMA-ES internal state；
3. restore internal state + update current population。

这一步参考 RL-DAS 的 context restoration 思想。

第一版 repeated DAS 可只使用现有 population transfer，context restoration 放到扩展实验。

---

# 21. 数据集构造应该从“Query outcome”改为“State × Action outcome”

当前共享 checkpoint continuation 资产非常适合新任务。

核心训练表建议变成：

```text
state_id
problem_id
dimension
prefix_algorithm
seed
FE
remaining_budget_ratio

behavior features ...

local landscape features ...
local landscape uncertainty ...

candidate_action
horizon
replicate_id

terminal_loss
action_gain_vs_continue
action_gain_class

soft_ert_action          # extension
soft_ert_gain            # extension

handoff_required
handoff_type
endpoint_status
```

关键单位是：

$$
(S_t,a,H,r).
$$

而不是：

$$
(S_t,\text{Query path}).
$$

---

# 22. Decision opportunity

当前：

```text
phase1_dynamic_budget_event_v1
```

可以保留。

但角色发生变化：

原来：

> 这些是“是否 Query”的候选机会。

现在：

> 这些是“是否继续 / switch / acquire information”的候选动态决策状态。

12–18 states/run 对第一阶段离线研究已经足够，不建议立刻提高密度。

Repeated DAS 时再缩小为少量可达机会，避免路径组合爆炸。

---

# 23. 计算成本控制

如果每个 state 都：

$$
4\ actions
\times
3\ reps
\times
3\ horizons
$$

成本会迅速爆炸。

建议：

## Stage A

使用现有完整 terminal action continuation 一次，建立粗标签。

## Stage B

只对以下状态做 3 repetitions：

- action margin 小；
- label sign 接近 0；
- 多模型意见不一致；
- event states；
- stratified sampled stable states。

这相当于 active replication，而不是每行无差别 3 次。

但正式论文必须预注册 replication selection rule，不能看 outcome 后补跑。

---

# 24. 模型选择建议

第一阶段请保持朴素。

## Action gain classifier

推荐：

- Logistic Regression；
- Random Forest；
- LightGBM 作为 selector sensitivity；
- calibrated probability。

不建议一开始：

- Transformer；
- LSTM；
- PPO；
- GNN。

原因很简单：

> 当前首先需要证明“信息存在”，不是证明“大模型能拟合”。

---

# 25. 时间序列模型放在第二轮

如果 tabular aggregated Behavior 已证明有信息，再比较：

```text
aggregated Behavior
vs
raw trajectory sequence
```

候选：

- MiniROCKET；
- 1D-CNN；
- TCN。

这可以直接对接 probing trajectories 与 trajectory classifier 文献。

---

# 26. 评价指标

不能只报告 classification accuracy。

## 26.1 Action prediction

- macro F1；
- balanced accuracy；
- Average Precision；
- Brier score；
- calibration curve。

## 26.2 Selection quality

最重要：

$$
Regret_t
=
L_{selected}
-
\min_a L_a.
$$

报告：

- mean regret；
- median regret；
- normalized regret；
- fraction within practical-equivalent action set。

## 26.3 Optimization outcome

- terminal log-gap；
- ERT / Soft-ERT；
- fixed-budget precision；
- success rate。

## 26.4 Dynamic policy

- switch count；
- first switch FE；
- total query FE；
- query rate；
- fraction of switches that improved performance；
- net terminal gain over continue-current；
- VBS–SBS gap closed。

---

# 27. Acceptable Action Set

不要坚持唯一 best solver label。

定义：

$$
\mathcal A_{acc}(S_t)
=
\left\{
a:
L_a-L_{\min}
\le
\delta_{\mathrm{practical}}
\right\}.
$$

如果多个算法性能统计上/实践上等价：

> 选中任意一个都算 acceptable。

这比强制：

$$
y=\arg\min_a L_a
$$

稳定得多，也更接近 AS-LGBM acceptable accuracy 的思想，但建议使用 practical-equivalence margin，而不是单纯依赖 $p>0.05$。

---

# 28. OOD 与 split

现有 grouped-by-function 原则应保留并强化。

禁止：

```text
same function / same instance 的不同 run
随机分散到 train/test
```

至少报告：

1. leave-function-out；
2. leave-dimension-out；
3. MA-BBOB structural OOD；
4. CEC development；
5. future CEC2022 / engineering confirmation。

---

# 29. 新的实验矩阵

| ID | Behavior | Local Landscape | Uncertainty | Extra Query | Repeated Switch | 目的 |
|---|---:|---:|---:|---:|---:|---|
| E0 | ✗ | ✗ | ✗ | ✗ | ✗ | SBS / continue |
| E1 | ✓ | ✗ | ✗ | ✗ | ✗ | Behavior-only DAS |
| E2 | ✗ | ✓ | ✗ | ✗ | ✗ | Local landscape only |
| E3 | ✓ | ✓ | ✗ | ✗ | ✗ | 协同基本模型 |
| E4 | ✓ | ✓ | ✓ | ✗ | ✗ | uncertainty 增量 |
| E5 | ✓ | ✓ | ✓ | Always | ✗ | Always Query |
| E6 | ✓ | ✓ | ✓ | G_FE Gate | ✗ | 当前 DBF baseline |
| E7 | ✓ | ✓ | ✓ | VOI Gate | ✗ | 新 Query policy |
| E8 | ✓ | ✓ | ✓ | VOI Gate | ✓ | 完整 repeated DAS |

---

# 30. 关键消融

## A. Behavior group

```text
T0
B1
B2
B2+Motion
B3
```

保留当前设计。

## B. Local landscape

```text
streaming distribution
+ meta-model
+ information content
+ reservoir geometry
```

逐组加入。

## C. Uncertainty

```text
point estimate
vs
estimate + bootstrap uncertainty
```

## D. Query

```text
never
always
fixed schedule
G_FE
uncertainty
VOI
```

## E. Label

```text
terminal log-gap gain
short-horizon gain
multi-horizon gain
dynamic Soft-ERT gain
```

---

# 31. 当前代码的建议重构方式

不要在现有目录里硬塞所有逻辑。

建议新增两层。

## 31.1 `online_landscape/`

```text
online_landscape/
    __init__.py
    streaming_statistics.py
    recursive_meta_model.py
    rolling_information.py
    reservoir.py
    local_fdc.py
    local_dispersion.py
    local_nbc.py
    uncertainty.py
    features.py
```

职责：

> 只使用 optimization 已经产生的 evaluation points，维护 Local Landscape State。

明确禁止：

> 在该模块内部额外调用 objective evaluator。

## 31.2 `dynamic_selection/`

```text
dynamic_selection/
    __init__.py
    action_gain_labels.py
    action_gain_dataset.py
    action_models.py
    acceptable_actions.py
    query_value.py
    controller.py
    repeated_policy.py
    evaluation.py
```

---

# 32. 现有模块如何处理

## `behavior/`

**保留。**

当前 permutation-invariant representation 是新方案的重要资产。

只建议增加：

- feature normalization contracts；
- optional raw trajectory export。

## `optimizers/`

**保留。**

完整 state continuation 和 population transfer 是新方案最难重建的基础设施之一。

## `trajectory/`

**保留。**

`phase1_dynamic_budget_event_v1` 第一阶段继续使用。

以后再增加 repeated-policy reachable-state logging。

## `selection_reference/`

建议逐步迁移为：

> action outcome / action gain reference

而不是只作为 Query 下游 selector。

原 action-loss matrix 可以继续利用。

## `utility_labels/efficacy.py`

`G_FE` 不删除。

重新定位成：

```text
legacy / DBF-v1 query operational efficacy baseline
```

新建：

```text
dynamic_selection/action_gain_labels.py
```

承载：

$$
G_{t,H}(a).
$$

## `decision/`

当前 Query Decision Model 保留作为 baseline。

不要继续把它扩成新主方法。

新主策略放到：

```text
dynamic_selection/controller.py
```

避免旧 first-trigger Query 语义与 repeated DAS 混在一起。

## `landscape_queries/`

保留为：

> independent global information acquisition

不再承担全部 landscape representation。

---

# 33. 建议的开发顺序

## Step 1：暂停扩展 G_FE Gate

当前 `G_FE -> Query` 主线保留可运行状态，但不再继续加复杂模型。

## Step 2：从已有 action-loss matrix 构造 action gain labels

先不跑新实验，检查现有正式协议能否直接派生：

$$
G_t(a)
=
L_{continue}-L_a.
$$

如果可以，就先生成：

```text
action_gain_dataset.parquet
```

## Step 3：做 Behavior-only feasibility

训练：

```text
B3 -> action gain class
```

做 grouped function OOF。

这是新方向第一个必须闭合的实验。

## Step 4：实现 online local landscape

只做少量高价值特征：

1. fitness moments / quantiles；
2. recursive linear/quadratic R²；
3. rolling information content；
4. local FDC；
5. elite dispersion；
6. bootstrap std。

第一版不要复制完整 61 ELA。

## Step 5：验证 Behavior + Landscape 增量

只有这一层显示明确增量，才进入 Query。

## Step 6：重定义 Query 为 VOI

训练：

$$
P(V_q^{op}>0\mid B,L,U).
$$

## Step 7：one-switch 在线 replay

先使用新 controller，但仍最多切一次。

和当前 DBF 做完全公平比较。

## Step 8：repeated switching pilot

仅 BBOB 10D 小规模验证：

- 最多 2 次 switch；
- 固定 dwell；
- 不做 RL。

## Step 9：再考虑 contextual bandit / RL

只有 supervised repeated DAS 已经证明：

$$
\text{state}\rightarrow\text{action gain}
$$

稳定成立，RL 才有必要。

---

# 34. 第一篇论文建议收缩到哪里

如果希望风险可控，我建议第一篇新工作不要同时声称：

- 新 Behavior；
- 新 ELA；
- 新 uncertainty；
- 新 Query；
- repeated DAS；
- RL；
- Soft-ERT。

这会变成一辆装满贡献点但没有刹车的卡车。

更合理的第一篇范围：

## 主贡献 1

**Algorithm-agnostic Behavior for action-conditioned switch-benefit prediction**

$$
B_t
\rightarrow
P(G_t(a)>\delta).
$$

## 主贡献 2

**Zero-additional-FE local landscape state and its incremental value beyond Behavior**

$$
B_t+L_t^{local}.
$$

## 主贡献 3

**Adaptive Query only under decision uncertainty**

用 VOI 证明：

> 独立 landscape probe 不必默认执行。

仍然限制：

$$
N_{switch}\le1.
$$

Repeated DAS 留给下一阶段。

---

# 35. 如果要直接冲更高完整度

则完整方法可定义为：

> **Sequential Landscape–Behavior Co-Decision for Dynamic Algorithm Selection**

状态：

$$
Z_t
=
[
B_t,\,
L_t^{local},\,
U_t,\,
B_{\mathrm{remain}}
].
$$

候选动作：

$$
a_t
\in
\{
continue,\ switch\_DE,\ switch\_PSO,\ switch\_CMAES,\ switch\_SHADE,\ query
\}.
$$

其中 `query` 是 information action，不直接对应 optimizer transition。

Optimizer action value：

$$
Q_t(a)
=
P
\left(
G_{t,H}(a)>\delta
\mid Z_t
\right).
$$

Query value：

$$
Q_t(q)
=
P
\left(
V_{t,q}^{op}>0
\mid Z_t
\right).
$$

决策：

$$
a_t^\star
=
\arg\max_{a\in\mathcal A_t}
\mathbb E[U_t(a)].
$$

这才是完整的 Behavior–Landscape Online Co-decision。

---

# 36. 与相关工作的边界

## Trajectory-based Algorithm Selection / Per-run AS

已有工作已经证明：

- 可以从 initial optimizer trajectory 提取特征；
- 可以预测第二算法性能；
- 可以 warm-start；
- 可以运行中切一次。

因此不能声称：

> “首次使用 trajectory 做动态 AS”。

## DynamoRep

已有工作证明：

- population trajectory 的低成本描述统计能携带丰富 problem-algorithm interaction 信息。

当前 Behavior 模块已经比原始 DynamoRep 更进一步，采用 permutation-invariant set dynamics。

因此可以把 DynamoRep 作为 Behavior representation 的重要来源之一，但不应声称 Behavior representation 概念本身全新。

## To Switch or Not to Switch

这项工作已经直接研究：

> sliding-window trajectory features 能否预测某个 switch point 的切换收益。

因此：

> “预测是否值得切换”

不能单独作为创新。

你需要强调区别：

1. action-conditioned multiple candidate solvers；
2. Behavior 与 online landscape 的增量信息分解；
3. Query 作为可选择的信息获取动作；
4. practical equivalence / uncertainty；
5. unknown-optimum dynamic label extension；
6. 最终 repeated DAS。

## Probing Trajectories

已有工作证明：

> 短 solver trajectory 可以作为 algorithm-centric instance representation 并用于 algorithm selection。

因此 Behavior-only AS 不是空白。

真正的问题应是：

> **自然运行产生的行为状态能否直接支持 online action selection，以及何时还需要额外 landscape information。**

## RL-DAS

RL-DAS 已经做 repeated selection，并将 landscape + algorithmic features 放入 MDP。

因此如果以后使用 PPO，不能把：

> “RL + DAS”

作为贡献。

你的区别应来自：

- heterogeneous solver portfolio；
- generic behavior representation；
- explicit value of landscape information；
- action-gain supervision；
- unknown-optimum label；
- query cost / uncertainty。

---

# 37. 关键风险

## 风险 1：Behavior 只能识别 problem，而不能预测 action advantage

如果：

$$
B_t
$$

能分类 BBOB function，但不能预测：

$$
G_t(a),
$$

则 DynamoRep 类结果不能自动支持 DAS。

因此 Phase 1 必须先做。

## 风险 2：Local Landscape 与 Behavior 高度冗余

可能结果：

$$
I(Y;L\mid B)\approx0.
$$

这不是坏事。

它意味着：

> 行为本身已经是更有效的 problem-algorithm interaction representation。

此时 Query 的必要性会进一步降低。

## 风险 3：Switch benefit 主要由 warm-start 机制决定

必须比较：

- native continue；
- population transfer；
- cold restart sensitivity。

否则模型可能学的是：

> 哪种 handoff 实现更占便宜。

## 风险 4：Dynamic Soft-ERT 不稳定

如果 horizon 太短，hitting target 事件可能稀少。

因此 Soft-ERT 必须先作为 sensitivity，而不是主标签。

## 风险 5：Repeated DAS 组合爆炸

不要离线枚举整条：

$$
4^T
$$

policy tree。

应从 one-step action outcome 学习开始。

Repeated controller 用 model rollout，而不是穷举所有路径。

---

# 38. 明确的 Stop Conditions

为了避免项目无限膨胀，建议预注册：

### Stop A

若 Behavior-only action prediction 在 grouped OOD 上不优于 time-only / SBS：

> 暂停 DAS 方向。

### Stop B

若 `Behavior + Local Landscape` 相对 Behavior 无稳定增量：

> 不把 online landscape 作为主贡献。

### Stop C

若 VOI Query 在 matched query rate 下不能优于 current G_FE Gate / uncertainty Gate：

> Query 只保留 baseline，不做主方法。

### Stop D

若 repeated switching 相对 best one-switch 没有稳定 terminal gain：

> 第一篇保持 one-switch，不使用 RL。

---

# 39. 推荐的新论文主线

我建议把原来的：

> **Decision-before-Feature**

保留为问题来源，而方法主线改成：

> **Decision-with-Progressive-Information**

但论文标题不建议强行造新术语。

更稳妥的题目方向：

### 方向 A

**Behavior-Guided Dynamic Algorithm Selection with Adaptive Landscape Information Acquisition**

### 方向 B

**Online Landscape–Behavior Co-Decision for Dynamic Algorithm Selection in Black-Box Optimization**

### 方向 C

**When Is Landscape Information Needed? Behavior-First Dynamic Algorithm Selection for Black-Box Optimization**

其中 C 最贴合项目最初的科学问题。

---

# 40. 推荐的核心假设

## H1

在固定共享状态与候选 action 下，短窗口算法行为包含预测未来相对 action gain 的信息：

$$
I(Y_{t,a};B_t)>0.
$$

## H2

trajectory-derived local landscape state 在 Behavior 条件下仍提供增量信息：

$$
I(Y_{t,a};L_t^{local}\mid B_t)>0.
$$

## H3

extra landscape query 的收益具有明显状态依赖性：

$$
P(V_q>0\mid Z_t)
$$

并非常数。

## H4

只在 decision uncertainty 高且 Query VOI 为正时获取独立 landscape information，可在较低 query FE 下达到或超过 Always Query。

## H5

Action-conditioned supervised policy 在异构 solver portfolio 上可以优于：

- SBS；
- continue；
- Random；
- fixed switch；
- static AAS；
- G_FE Query Gate；
- behavior-only static selector。

---

# 41. 最终建议

当前项目**不需要推倒重来**。

最值得保留的是：

1. optimizer state continuation；
2. population transfer；
3. Behavior representation；
4. dynamic state sampling；
5. grouped function OOD；
6. same-state action outcome generation；
7. query isolation；
8. Behavior-only / Query selector infrastructure。

最应该停止继续扩展的是：

> **把 G_FE 作为上游唯一 gate，再让另一个 Selector 决定真正 optimizer action。**

建议新的因果顺序不是：

```text
Behavior
 -> Query efficacy
 -> Query
 -> Selector
 -> action
```

而是：

```text
Behavior + free online landscape
 -> candidate action gain
 -> action confidence
 -> [if needed] additional Query
 -> updated action gain
 -> optimizer action
```

因此新工作的真正核心可以浓缩成一句话：

> **先判断“下一步哪个算法动作有意义”，再判断“现有信息是否足以做这个决定”；只有不足时才购买额外 landscape information。**

这比“先判断 ELA 值不值得，再让 ELA 帮忙选择算法”逻辑更直接，也更接近 dynamic algorithm selection 的实际决策结构。

---

# 42. 最小可执行版本

如果下一步只做一件事，建议不要立刻实现 online ELA。

先完成：

$$
\boxed{
B_t
\rightarrow
\operatorname{sign}
\left(
L_{continue}
-
L_a
\right)
}
$$

对：

```text
a ∈ {DE, PSO, CMA-ES, SHADE}
```

构造 Behavior-only action gain dataset。

只要这个实验成立，后续：

- online landscape；
- uncertainty；
- Query VOI；
- Soft-ERT；
- repeated DAS；

才有可靠的研究基础。

---

# 43. 参考文献

1. Jankovic, A., Eftimov, T., Doerr, C. **Towards Feature-Based Performance Regression Using Trajectory Data.** EvoApplications, 2021. DOI: 10.1007/978-3-030-72699-7_38.
2. Jankovic, A., Vermetten, D., Kostovska, A., de Nobel, J., Eftimov, T., Doerr, C. **Trajectory-based Algorithm Selection with Warm-starting.** IEEE CEC, 2022. DOI: 10.1109/CEC55065.2022.9870222.
3. Kostovska, A., Jankovic, A., Vermetten, D., de Nobel, J., Wang, H., Eftimov, T., Doerr, C. **Per-run Algorithm Selection with Warm-Starting Using Trajectory-Based Features.** PPSN XVII, 2022. DOI: 10.1007/978-3-031-14714-2_4.
4. Cenikj, G., Petelin, G., Doerr, C., Korošec, P., Eftimov, T. **DynamoRep: Trajectory-Based Population Dynamics for Classification of Black-box Optimization Problems.** GECCO, 2023. DOI: 10.1145/3583131.3590401.
5. Vermetten, D., Wang, H., Sim, K., Hart, E. **To Switch or not to Switch: Predicting the Benefit of Switching between Algorithms based on Trajectory Features.** EvoApplications, 2023. DOI: 10.1007/978-3-031-30229-9_22.
6. Renau, Q., Hart, E. **On the Utility of Probing Trajectories for Algorithm-Selection.** EvoApplications, 2024. arXiv:2401.12745.
7. Guo, H., Ma, Y., Ma, Z., et al. **Deep Reinforcement Learning for Dynamic Algorithm Selection: A Proof-of-Principle Study on Differential Evolution.** IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2024. DOI: 10.1109/TSMC.2024.3374889.
8. Janković, A., Doerr, C. **Adaptive Landscape Analysis.** GECCO Companion, 2019. DOI: 10.1145/3319619.3326905.
9. Guo, Q., Wang, H., Tian, Y. **Automated algorithm selection for black-box optimization using light gradient boosting machine.** Swarm and Evolutionary Computation, 2025, 98:102071.
