# Decision-before-Feature 研究脉络整理（v2）

## 1. 核心研究问题

当前自动算法选择（Automated Algorithm Selection, AAS）的典型流程：

    Black-box Problem
            |
            v
    Landscape Feature Extraction (ELA)
            |
            v
    Algorithm Selection
            |
            v
    Optimizer

该流程隐含一个重要假设：

> 对任意黑盒优化问题，进行Landscape Analysis都是有价值的。

但是现实中ELA具有额外成本：

-   函数评价次数（Function Evaluations, FE）
-   CPU时间
-   特征计算成本

对于简单问题：

    ELA cost > algorithm selection benefit

因此提出：

# Decision-before-Feature

核心思想：

在执行昂贵Feature Extraction之前，首先判断：

> 当前问题是否值得进行Landscape Analysis。

新的流程：

    Black-box Problem

            |
            v

    Cheap Search Behavior Observation

            |
            v

    Decision Module

            |
       ----------------
       |              |
     Skip            Run ELA

       |              |
    Default       Algorithm Selection
    Optimizer

------------------------------------------------------------------------

# 2. 与已有ELA工作的关系

## 2.1 Classical ELA

经典ELA通过采样优化问题，构造Landscape描述。

主要Feature：

-   Distribution
-   Dispersion
-   Information Content
-   PCA
-   Meta-model
-   Local structure

优势：

-   可解释
-   成熟
-   大量算法选择研究采用

不足：

-   Feature计算昂贵
-   高维困难
-   默认Feature一定值得计算

------------------------------------------------------------------------

## 2.2 DeepELA / NeurELA

发展趋势：

从人工Feature转向自动表示学习。

DeepELA：

学习Landscape embedding。

NeurELA：

进一步考虑动态优化状态：

    Population
    Fitness
    Trajectory

            |
            v

    Neural representation

区别：

Classical ELA：

Problem -\> Feature

NeurELA：

Optimization State -\> Representation

------------------------------------------------------------------------

# 3. Behavior Analysis作为Decision依据

## 3.1 Algorithm behavior

已有研究表明：

算法行为可以被量化。

典型行为类别：

1.  Exploration
2.  Exploitation
3.  Locality
4.  Communication
5.  Evaluation Effort

行为指标包括：

-   Diversity
-   Improvement rate
-   Convergence speed
-   Population movement
-   Communication

------------------------------------------------------------------------

## 3.2 Exploration / Exploitation指标

重要指标：

### Directional Entropy

描述搜索方向分散程度。

### Distance Decay

描述个体距离最优区域变化。

### Stagnation

描述搜索停滞程度。

这些指标不是最终研究目标，而是：

Decision Evidence。

------------------------------------------------------------------------

# 4. Search Maturity概念

定义：

Search Maturity表示：

> 当前搜索过程是否已经产生足够的信息，使Landscape Analysis具有价值。

区别：

  概念                       含义
  -------------------------- ------------------
  Convergence                是否接近最优
  Exploration/Exploitation   搜索行为状态
  Search Maturity            是否具备分析价值

可能状态：

## Low maturity

-   搜索随机性高
-   行为不稳定
-   Landscape信息不足

## Medium maturity

-   搜索结构形成
-   行为稳定
-   最适合执行ELA

## High maturity

-   强收敛
-   信息增长有限

------------------------------------------------------------------------

# 5. Label设计

不要直接预测：

Need ELA?

建议预测：

ELA Utility。

定义：

    U_ELA =
    Performance Gain
    -
    lambda * Analysis Cost

其中：

Performance Gain：

-   ERT改善
-   error降低
-   success rate提升

Cost：

-   ELA sampling FE
-   CPU time

决策：

    if U_ELA > 0:

        Run ELA

    else:

        Skip ELA

------------------------------------------------------------------------

# 6. Benchmark划分原则

## 关键原则

避免function leakage。

原因：

BBOB/CEC中：

shift rotation noise

通常只是同一个base function变化。

因此：

不要随机划分instance。

------------------------------------------------------------------------

# 推荐协议

## Level 1

IID：

同benchmark随机划分。

目的：

基础能力。

------------------------------------------------------------------------

## Level 2

Function family split：

训练：

部分function family

测试：

未见family。

目的：

验证跨landscape泛化。

------------------------------------------------------------------------

## Level 3

Cross benchmark OOD：

Train:

BBOB

Test:

CEC2017/CEC2022

目的：

验证真实迁移。

------------------------------------------------------------------------

# 7. 相关论文划分方式启发

已有PSO参数控制研究：

按照：

-   unimodal
-   multimodal
-   separable
-   non-separable
-   composition

分析算法表现。

该方式适合：

解释算法行为。

但不适合作为你的主要train/test split。

你的目标是：

behavior generalization。

因此：

function family split更合理。

------------------------------------------------------------------------

# 8. 最终研究假设

H1:

ELA并非所有问题都具有正收益。

H2:

优化过程中的行为信息可以预测ELA价值。

H3:

Decision-before-Feature能够降低分析成本，同时保持算法选择收益。

H4:

基于行为的决策具有跨benchmark泛化能力。
