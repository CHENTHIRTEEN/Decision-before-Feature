# Decision-before-Feature 完整论文结构与Contribution设计

## 1. 论文定位

## 1.1 核心研究领域

本文属于：

- Automated Algorithm Selection (AAS)
- Exploratory Landscape Analysis (ELA)
- Behavior Analysis of Metaheuristic Optimization
- Resource-aware Optimization

核心研究问题：

> 在黑盒优化中，所评估的固定 landscape-analysis query 是否值得执行？

第一篇论文的主 query 固定为 `descriptor_cheap`；standard/broad 只用于预先定义的配置稳健性，结论不外推到其他 landscape representation。

---

# 2. 论文核心故事（Research Story）

## 2.1 领域现状

现有自动算法选择流程：

    Black-box Problem

    ↓

    Landscape Feature Extraction

    ↓

    Algorithm Selection

    ↓

    Optimizer

其中：

ELA用于：

- 描述问题结构
- 建立problem-feature映射
- 预测算法适应性

大量研究集中于：

- 新Query Feature设计
- 更强Algorithm Selector
- 更复杂Machine Learning模型

---

## 2.2 现有研究缺口

现有方法隐含：

    Feature extraction is always useful

但是：

固定 query 存在额外成本：

$$
C_{query}
$$

包括：

- Sampling FE
- Feature computation
- Runtime

对于部分问题：

$$
Benefit_{query}<Cost_{query}
$$

因此：

执行该固定 query 本身也是一个优化决策。

---

# 3. 新问题定义

## Analysis Selection Problem

传统：

    Which algorithm should be selected?

本文：

    Should the evaluated fixed query be performed?

形成两阶段决策：

    Stage 1:

    Should analyze?

    Stage 2:

    Which algorithm?

---

# 4. 核心方法框架

    Unknown Black-box Problem

    |

    ↓

    Cheap Search Behavior Observation

    |

    ↓

    Search Maturity Estimation

    |

    ↓

    Query Utility Prediction

    |

    ↓

    Decision-before-Feature

    -------------------

    |                 |

    No-query          Run Query

    |                 |

    Default Solver   Algorithm Selection

---

# 5. Contribution设计

# Contribution 1

## Redefining Automated Algorithm Selection

提出：

Analysis Selection Problem。

贡献：

将所评估的固定 landscape-analysis query 从固定步骤转变为可优化决策。

创新点：

传统：

Problem → Feature → Algorithm

本文：

Problem → Decide Feature → Algorithm

---

# Contribution 2

## Behavior-aware Decision Framework

提出：

利用优化过程动态行为预测所评估固定 query 的价值。

输入：

Search Behavior State。

包括：

- Improvement rate
- Population diversity
- Population Wasserstein change
- Centroid shift coherence
- Covariance spectral concentration
- Fitness distribution change
- Stagnation
- Communication

输出：

Query Utility。

---

# Contribution 3

## Search Maturity Representation

提出：

Search Maturity。

核心思想：

不是判断：

搜索是否收敛。

而判断：

搜索是否已经产生足够的信息。

区别：

  概念                       关注

---

  Convergence                距离最优
  Exploration/Exploitation   搜索行为
  Search Maturity            分析价值

---

# Contribution 4

## Resource-aware Landscape Analysis

提出：

Utility-based decision。

定义：

$$
U_{query} = PerformanceGain - \lambda_T TimeCost - \lambda_M MemoryCost
$$

等总 FE 协议下，query sampling FE 已通过减少 Query continuation budget 计入 PerformanceGain，不重复扣除。

实现：

性能和成本动态权衡。

---

# 6. 论文结构设计

# Section 1 Introduction

## 目标

建立：

为什么需要Decision-before-Feature。

---

## Paragraph 1

介绍自动算法选择。

说明：

不同优化问题适合不同算法。

---

## Paragraph 2

介绍ELA。

说明：

ELA成为AAS重要工具。

---

## Paragraph 3

指出问题。

现有工作：

关注：

- Better features
- Better selectors

忽略：

> Whether feature extraction itself is worthwhile.

---

## Paragraph 4

提出本文观点。

Whether to execute the evaluated fixed landscape-analysis query should itself be treated as a resource-aware decision.

---

## Paragraph 5

贡献总结。

---

# Section 2 Related Work

## 2.1 Automated Algorithm Selection

介绍：

- Rice framework
- Algorithm portfolio
- Meta-learning

强调：

已有工作关注algorithm decision。

---

## 2.2 Exploratory Landscape Analysis

内容：

- Classical ELA
- flacco/pflacco
- DeepELA
- NeurELA

强调：

现有 feature-based selection 通常默认 landscape information 已被获取。

---

## 2.3 Algorithm Behavior Analysis

介绍：

搜索行为指标。

包括：

- Exploration
- Exploitation
- Diversity
- Entropy

说明：

行为信息可以作为低成本决策依据。

---

## 2.4 Hyper-heuristic and Adaptive Control

联系：

动态决策。

区别：

本文决定：

是否分析。

---

# Section 3 Problem Formulation

## 3.1 Traditional AAS

定义：

$$
A^*=S(\phi(p))
$$

---

## 3.2 Analysis Selection

定义：

$$
d\in \{0,1\}
$$

其中：

0:

Skip analysis

1:

Run analysis

---

## 3.3 Utility Function

定义：

$$
U(d,p)
$$

展开：

$$
U=PerformanceGain-\lambda_T TimeCost-\lambda_M MemoryCost.
$$

---

# Section 4 Proposed Method

# 4.1 Search Behavior Extraction

说明：

低成本信息。

Feature groups：

## Progress

- FE ratio
- improvement

## Population

- diversity

## Behavior

- population distribution change
- fitness distribution change
- stagnation

---

# 4.2 Search Maturity

定义：

搜索信息成熟状态。

包括：

- Exploration Stabilization
- Exploitation Saturation

---

# 4.3 Query Utility Prediction

模型：

输入：

behavior state

输出：

$$
\hat U_{query}
$$

---

# 4.4 Decision Mechanism

规则：

    if predicted utility > 0:

    execute the fixed query

    else:

    skip

---

# Section 5 Experimental Design

## RQ1

主 `descriptor_cheap` query 是否在所有状态都有净收益？

实验：

Cost-benefit analysis。

---

## RQ2

Behavior 是否包含固定 query 的效用信息？

实验：

Utility prediction。

---

## RQ3

Decision-before-Feature是否有效？

Baseline：

- Never Query
- Always Query
- Traditional AAS
- Random Decision

---

## RQ4

是否具有泛化能力？

设置：

BBOB → CEC

Function family split。

---

## RQ5

为什么有效？

SHAP解释。

---

# Section 6 Results Analysis

## 6.1 Utility Distribution

证明：

主 `descriptor_cheap` query 存在 `U_{cheap}\leq0` 的状态。

---

## 6.2 Prediction Results

证明：

behavior 可以预测固定 query 的效用。

---

## 6.3 Optimization Results

证明：

降低成本同时保持性能。

---

## 6.4 OOD Results

证明：

不是benchmark记忆。

---

## 6.5 Behavior Interpretation

解释：

什么搜索状态适合执行固定 query。

---

# Section 7 Discussion

讨论：

## 为什么不是新的Query Feature？

因为：

问题不是Feature不足。

而是：

所评估固定 query 的特征是否值得计算。

---

## 与NeurELA关系

NeurELA：

学习更好的表示。

本文：

决定何时值得执行所评估的固定 query。

二者互补。

---

## 局限

- 依赖离线效用标签生成
- 当前针对continuous optimization
- Portfolio有限

---

# Section 8 Conclusion

总结：

本文提出：

Decision-before-Feature。

核心贡献：

将所评估的固定 landscape-analysis query 纳入优化决策。

未来：

- Progressive ELA
- Neural Decision
- Multi-objective extension
