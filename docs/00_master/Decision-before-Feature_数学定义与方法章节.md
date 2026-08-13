# Decision-before-Feature 数学定义与方法章节

## 1. 方法定位

本文研究的问题不是传统的“如何利用 landscape features 选择算法”，而是进一步研究：

> 在执行预先定义的 landscape-analysis query 之前，是否值得执行该固定 query。

第一篇论文的主 query 是 `descriptor_cheap`，即 16 维自定义低成本描述符；它不代表完整 ELA 或完整 pflacco。`pflacco_standard` 与 `pflacco_broad` 仅用于预先定义的配置稳健性实验。

因此，将问题定义为Analysis Selection Problem。

传统Automated Algorithm Selection流程：

    Optimization Problem
            |
            v
    Query Feature Extraction
            |
            v
    Algorithm Selection
            |
            v
    Optimizer

本文提出：

    Optimization Problem

    |
            v

    Cheap Search Behavior Observation

    |
            v

    Decision-before-Feature

|                       |
| --------------------- |
|                       |
| No-query      Run Query |

    |              |

    Default      Algorithm Selection

    Optimizer

核心思想：

Feature extraction本身也是一种需要优化的资源消耗。

---

# 2. Problem Formulation

## 2.1 Optimization Problem

设黑盒优化问题为：

$$
p \in \mathcal{P}
$$

目标：

$$
\min f_p(x)
$$

其中：

- (p)：问题实例
- (x)：决策变量
- (f_p)：未知目标函数

---

## 2.2 Traditional Algorithm Selection

传统方法：

首先计算：

$$
\phi(p)
$$

其中：

$$
\phi
$$

表示query feature extractor。

在本文的在线状态任务中，固定下游 Selection Reference 还接收 permutation-invariant 搜索行为和连续剩余预算，并预测每个候选动作的连续 loss：

$$
\widehat{\boldsymbol L}(s_t)
=S\!\left(\phi(p),\operatorname{behavior}(s_t),B_t/FE_{total}\right),
$$

$$
\hat a_t=\arg\min_a\widehat L(s_t,a).
$$

离线监督由同一共享状态上的 `continue_current` 与其余 portfolio actions 真实 continuation 产生；不使用静态 problem label 或 nearest performance bucket。

该流程对应已有 feature-based algorithm selection 范式：先获取 landscape features，再用监督学习模型从算法组合中选择优化器。本文不把该 selector 作为新贡献，而是把它作为固定下游 Selection Reference；本文研究的是调用该 selector 之前是否值得执行所评估的固定 query。

---

## 2.3 Limitation

固定 query 存在额外成本：

$$
C_{query}
$$

包括：

- Function Evaluation Cost
- CPU Time
- Memory Cost

因此：

不能默认：

$$
U_{query}(p)>0
$$

---

# 3. Analysis Selection Problem

定义决策变量：

$$
d\in\{0,1\}
$$

其中：

$$
d=1
$$

表示执行固定 query。

$$
d=0
$$

表示不执行 query。

目标：

最大化：

$$
U(d,p)
$$

即：

$$
U=Performance-Cost
$$

---

# 4. Query Utility Label

为了训练Decision Module，需要构造监督信号。

## 4.1 No-query performance

直接优化：

$$
P_{skip}
$$

表示：

不执行 query 时的性能。

---

## 4.2 Query-based performance

执行：

$$
Problem \rightarrow Fixed\ Query \rightarrow Algorithm
Selection
$$

得到：

$$
p_{query}
$$

---

## 4.3 Utility

定义：

$$
G=P_{skip}-p_{query}
$$

表示固定 query 路径相对 No-query 路径的性能差。

最终：

$$
U_{query}=G-\lambda_T C_T-\lambda_M C_M.
$$

主协议采用等总 FE；query sampling FE 通过减少 Query continuation budget 进入 $p_{query}$，不得再次扣除。$C_T$ 表示两条路径的时间差，$C_M$ 表示尚未进入 performance loss 的额外内存成本。

主时间成本使用两条完整路径的有符号 wall-clock 相对差。令 $T_{query,total}$ 包含 query 采样、样本目标评价、特征计算、选择、必要的 population transfer 初始化与 Query 后续优化，$T_{skip,total}$ 包含 No-query 必要的 transfer 初始化与后续优化，则

$$
C_T=\frac{T_{query,total}-T_{skip,total}}{\max(T_{skip,total},10^{-12})}.
$$

因此 query 样本评价时间进入 Query 总时间，但 Query 分支因等总 FE 而少执行的后续优化时间也被抵消。纯 feature/selection/handoff 计算开销只作诊断分解，不替代主 $C_T$。

令：

$$
P_{best\ observed}=\min_{a\in\mathcal A(s_t)}L(s_t,a),
$$

则：

$$
G=(P_{skip}-P_{best\ observed})-(p_{query}-P_{best\ observed}).
$$

第一项是逐状态潜在性能差，第二项是 selector regret。Population Transfer 的影响已经包含在 observed action loss 中，不作为独立成本重复相减。

决策：

$$
U_{query}>0
$$

执行固定 query。

否则：

不执行 query。

---

# 5. Search Behavior State Representation

Decision-before-Feature不能使用query feature作为输入，否则产生循环。

因此输入必须来自低成本搜索行为。

定义：

$$
s_t
$$

表示时间t的搜索状态。

组成：

$$
s_t=[B_t,P_t]
$$

其中：

B：

Behavior information。

P：

Progress information。

---

# 6. Behavior Features

## 6.1 Diversity

描述种群空间分布。

例如：

平均距离：

$$
D_t
$$

---

## 6.2 Improvement Rate

描述优化收益：

$$
r_t=\frac{FE_t}{FE_{max}}
$$

令 anchor 为逐次完整原生 update 历史中满足下式的最近 update：

$$
r_a \le r_t-w
$$

则：

$$
I_t^{(w)}=
\frac{f_{best}(a)-f_{best}(t)}
{\max(\operatorname{IQR}(f_{init}),\epsilon)}
\cdot
\frac{1}{r_t-r_a}
$$

其中 $f_{init}$ 是优化器初始化后、任何原生 update 前已评估 population 的 fitness 向量。该 IQR 也是所有 shift-invariant fitness-change rate 的共享稳健尺度。

---

## 6.3 Permutation-invariant Population Change

令当前与anchor种群分别为等权经验分布 $P_t$ 与 $P_a$，种群规模为 $N$。跨checkpoint不假定行号表示同一个体。

空间分布变化使用经验Wasserstein-1：

$$
W_t^{(w)}=
\frac{1}{r_t-r_a}
\min_{\pi\in S_N}
\frac{1}{N}\sum_{i=1}^{N}
\frac{\lVert x_i^{(a)}-x_{\pi(i)}^{(t)}\rVert_2}{\sqrt d}.
$$

centroid shift rate定义为：

$$
G_t^{(w)}=
\frac{\lVert \mu_t-\mu_a\rVert_2}
{\sqrt d\,(r_t-r_a)}.
$$

centroid shift coherence为未除预算间隔的centroid shift与Wasserstein distance之比；无集合变化时取0，其余限制在 $[0,1]$。

当前种群的协方差谱集中度使用特征值的归一化Herfindahl concentration。令 $q=\min(d,N-1)$，则：

$$
C_t=
\frac{q\frac{\sum_j\lambda_j^2}{(\sum_j\lambda_j)^2}-1}{q-1}.
$$

退化协方差取0。上述统计只依赖种群集合，不依赖个体身份或亲子关系。

---

## 6.4 Stagnation

描述连续无改善时间。

令：

$$
r_{last}
$$

表示最近一次best-fitness严格改善对应的FE ratio，则：

$$
S_t^{(w)}=
\frac{\min(\max(r_t-r_{last},0),w)}{w}
$$

---

## 6.5 Distance Decay

描述population向当前population-best收缩的程度。

令：

$$
d_t=\frac{1}{N}\sum_i\frac{\|x_i-x_{best,t}\|_2}{\sqrt d}
$$

则：

$$
DD_t^{(w)}=
\frac{d_a-d_t}{\max(d_a,\epsilon)}
$$

其中：

$$
x_{best,t}
$$

是当前population中的best solution，不是真实全局最优。

---

## 6.6 Convergence Rate

描述种群diversity下降速度：

$$
CR_t^{(w)}=
\frac{D_a-D_t}{\max(D_a,\epsilon)}
\cdot
\frac{1}{r_t-r_a}
$$

---

## 6.7 Relation to Prior Behavioral Metrics

上述指标来自已有metaheuristic behavior analysis的启发，但不是逐式复现。

与 *How do metaheuristics exploit?* 的区别：

- 该论文指标用于late-stage exploitation behavior diagnostics/control；
- 本文指标用于预测Query Utility；
- 该论文可使用iteration窗口、exploitation phase划分和reference optimum；
- 本文使用FE-ratio窗口，不使用真实全局最优、function identity或query feature。

与 *Determining Metaheuristic Similarity Using Behavioral Analysis* 的区别：

- 该论文构造whole-run behavioral footprint，用于algorithm similarity clustering；
- 本文构造checkpoint-level behavior state，用于Analysis Selection；
- 本文仅保留diversity change、improvement、convergence、distance/locality等可低成本改写的行为语义；
- 本文不使用whole-run knee-point extraction、STN locality指标、IN communication指标、DBSCAN clustering或frequency map。

论文表述应采用：

> budget-normalized behavior states inspired by prior behavioral analysis metrics

而不是：

> the same exploitation or similarity metrics as prior work.

---

## 6.8 Communication Behavior

适用于群智能算法：

描述个体之间的信息传播。

---

# 7. Search Maturity

## 7.1 Definition

Search Maturity定义：

> 当前搜索过程是否已经产生足够的信息，使进一步Landscape
> Analysis具有价值。

区别：

  概念                       含义

---

  Convergence                是否接近最优
  Exploration/Exploitation   当前搜索行为
  Search Maturity            分析价值是否形成

---

# 7.2 Two-factor formulation

搜索成熟度不是单调增长。

因此定义两个因素。

## Exploration Stabilization

$$
ES_t
$$

表示探索是否由随机状态转向结构化。

---

## Exploitation Saturation

$$
XS_t
$$

表示开发是否过度。

---

定义：

$$
M_t=ES_t(1-XS_t)
$$

解释：

低成熟：

信息不足。

中成熟：

最适合分析。

高成熟：

可能已经过度收缩。

---

# 8. Decision Module

定义：

$$
\hat U=f_\theta(s_t)
$$

输入：

搜索行为状态。

输出：

Query Utility。

训练目标：

$$
\min_{\theta}(U_{query}-\hat U)^2
$$

---

# 9. Algorithm

    Input:
    Black-box problem p

    1. Run cheap probing optimization

    2. Collect search behavior

    3. Extract behavior state

    4. Predict fixed-query utility

    5. If utility > 0:

    Execute the fixed query

    Run algorithm selection

    Else:

    Run default optimizer

    6. Return solution

---

# 10. Expected Contributions

1. 提出Analysis Selection Problem。
2. 提出Decision-before-Feature框架。
3. 建立搜索行为与所评估固定 query 效用之间的关系。
4. 提供资源感知的自动算法选择流程。
