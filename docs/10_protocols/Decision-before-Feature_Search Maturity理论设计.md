# Decision-before-Feature Search Maturity 理论设计

## 1. 研究定位

Search Maturity（搜索成熟度）是 Decision-before-Feature
框架中的核心中间概念。

传统自动算法选择关注：

    Problem
       |
       v
    Landscape Feature
       |
       v
    Algorithm Selection

而本文关注：

    Problem

       |
       v

    Early Search Behavior

       |
       v

    Search Maturity

       |
       v

    Should Landscape Analysis be performed?

Search Maturity 的目标不是预测优化结果，而是判断：

> 当前优化过程是否已经产生足够的信息，使额外Landscape
> Analysis具有正向收益。

------------------------------------------------------------------------

# 2. 与已有概念的区别

## 2.1 与Convergence区别

Convergence回答：

> 搜索是否接近最优。

例如：

best fitness变化。

Search Maturity回答：

> 搜索过程是否已经暴露问题结构。

二者可能不同。

例：

一个简单Sphere：

快速收敛。

但是：

ELA可能没有价值。

一个复杂多峰问题：

尚未收敛。

但是：

搜索行为已经包含丰富Landscape信息。

------------------------------------------------------------------------

## 2.2 与Exploration/Exploitation区别

Exploration/Exploitation描述：

当前搜索行为。

Search Maturity描述：

当前搜索行为是否达到分析条件。

因此：

Exploration/Exploitation是输入。

Search Maturity是状态。

------------------------------------------------------------------------

# 3. 理论假设

提出三个假设。

## H1

搜索行为随优化过程变化，并包含问题结构信息。

形式：

$$ s_t \rightarrow p $$

其中：

(s_t)为搜索状态。

------------------------------------------------------------------------

## H2

存在一个搜索阶段，使Landscape Analysis收益最大。

即：

$$ U_{ELA}(t) $$

不是单调函数。

------------------------------------------------------------------------

## H3

Search Maturity可以作为ELA决策依据。

即：

$$ M_t \rightarrow U_{ELA} $$

------------------------------------------------------------------------

# 4. Search State Representation

定义：

$$ s_t $$

由以下信息组成。

------------------------------------------------------------------------

## 4.1 Progress Information

### Function Evaluation Ratio

$$ r_t=\frac{FE_t}{FE_{max}} $$

表示当前预算消耗。

------------------------------------------------------------------------

### Improvement Rate

$$ I_t= \frac{f_{best}(t-k)-f_{best}(t)} {k} $$

表示优化收益。

------------------------------------------------------------------------

## 4.2 Population Information

### Diversity

例如：

平均个体距离：

$$ D_t= \frac{2}{N(N-1)}
\sum_i\sum_j\|x_i-x_j\| $$

------------------------------------------------------------------------

### Population Spread

描述搜索区域覆盖程度。

------------------------------------------------------------------------

## 4.3 Behavioral Information

来源：

算法行为分析。

包括：

-   exploration
-   exploitation
-   communication
-   locality

------------------------------------------------------------------------

# 5. Search Maturity构造

## 5.1 为什么不是单调指标？

简单定义：

$$ M_t=f(convergence) $$

存在问题。

因为：

收敛越高不代表越值得ELA。

例如：

过早收敛：

    information gain ↓

------------------------------------------------------------------------

# 5.2 双因素模型

提出：

Search Maturity由两个因素共同决定。

## Exploration Stabilization (ES)

表示：

探索是否从随机状态转变为结构化状态。

输入：

-   diversity下降速度
-   entropy变化
-   trajectory稳定性

定义：

$$ ES_t \in [0,1]$$

高：

搜索模式稳定。

------------------------------------------------------------------------

## Exploitation Saturation (XS)

表示：

开发是否达到饱和。

输入：

-   stagnation
-   improvement decay
-   local concentration

定义：

$$ XS_t \in [0,1]$$

高：

继续搜索收益降低。

------------------------------------------------------------------------

# 5.3 Mature Window

定义：

$$ M_t=ES_t(1-XS_t) $$

解释：

## Early stage

$$ ES低 $$

搜索未形成结构。

------------------------------------------------------------------------

## Middle stage

$$ ES高, XS低 $$

最佳分析窗口。

------------------------------------------------------------------------

## Late stage

$$ XS高 $$

搜索可能已经过度收缩。

------------------------------------------------------------------------

# 6. Search Maturity估计方法

## 方法A：Explicit Index

直接计算：

$$ M_t $$

优点：

-   可解释
-   易分析

缺点：

需要人工设计。

------------------------------------------------------------------------

## 方法B：Latent Representation

利用：

Autoencoder

或者

MLP

学习：

$$ z_t=g(s_t) $$

其中：

z表示潜在成熟状态。

------------------------------------------------------------------------

## 方法C：Supervised Estimation

使用ELA Utility作为监督：

$$ g(s_t)\rightarrow U_{ELA} $$

间接获得成熟度。

------------------------------------------------------------------------

# 7. 实验验证

需要证明：

## Experiment 1

Search Maturity与ELA Utility存在关系。

分析：

Spearman correlation。

------------------------------------------------------------------------

## Experiment 2

比较：

Without maturity

vs

With maturity

------------------------------------------------------------------------

## Experiment 3

不同function family下：

分析成熟窗口差异。

------------------------------------------------------------------------

# 8. 预期贡献

Search Maturity不是新的Landscape Feature。

而是：

> A decision-oriented representation of optimization process information
> sufficiency.

它连接：

Behavior Analysis

和

Landscape Analysis Decision。
