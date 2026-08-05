# Decision-before-Feature Search Maturity与ELA Utility关系验证设计

## 1. 文档定位

本文档用于验证 Decision-before-Feature 框架中的核心理论假设：

> 搜索行为不是简单描述优化过程，而是能够反映当前问题是否已经具备进行Landscape
> Analysis的价值。

整体关系：

    Search Behavior Features

            |

            v

    Search Maturity

            |

            v

    ELA Utility

            |

            v

    Decision-before-Feature

本实验的目标：

证明 Search Maturity 不是人为构造的概念，而是连接：

-   Optimization Behavior
-   Landscape Analysis Decision

之间的有效中间状态。

------------------------------------------------------------------------

# 2. 核心理论假设

## H1: Search Behavior包含ELA价值信息

优化过程中的行为状态：

$$ s_t $$

能够预测：

$$ U_{ELA} $$

即：

$$ s_t \rightarrow U_{ELA} $$

------------------------------------------------------------------------

## H2: Search Maturity比单一行为指标具有更强解释能力

单个指标：

例如：

-   entropy
-   diversity
-   stagnation

只能描述局部行为。

Search Maturity融合：

多个行为维度：

形成：

$$ M_t $$

能够更准确表示：

分析就绪状态。

------------------------------------------------------------------------

## H3: ELA Utility与Search Maturity存在结构关系

不是简单：

$$ M \uparrow \Rightarrow U \uparrow $$

而应该存在：

中间最优区域。

即：

倒U关系：

    ELA Utility

          ^
          |

          |        ***
          |      **   **
          |    **
          |
          +---------------->

              Search Maturity

------------------------------------------------------------------------

# 3. 实验数据基础

使用：

Offline trajectory dataset。

来源：

多个优化算法：

-   DE
-   PSO
-   CMA-ES
-   SHADE

Benchmark：

BBOB训练集。

记录：

不同FE阶段：

-   behavior state
-   ELA Utility

------------------------------------------------------------------------

# 4. Search Maturity计算方案

## 4.1 Explicit Maturity Index

基于：

两个因素。

------------------------------------------------------------------------

## Exploration Stabilization (ES)

表示：

探索是否从随机转向结构化。

输入：

-   diversity下降趋势
-   entropy变化
-   movement稳定性

输出：

$$ ES_t\in[0,1]$$

------------------------------------------------------------------------

## Exploitation Saturation (XS)

表示：

开发是否过度饱和。

输入：

-   stagnation
-   improvement decay
-   population concentration

输出：

$$ XS_t\in[0,1]$$

------------------------------------------------------------------------

## Maturity

定义：

$$ M_t=ES_t(1-XS_t) $$

------------------------------------------------------------------------

# 5. 关系验证实验

# Experiment 1: Behavior Feature与ELA Utility相关性

## 目标

验证：

单个行为指标是否包含ELA价值信息。

------------------------------------------------------------------------

## 方法

计算：

Spearman correlation：

$$ \rho(feature,U_{ELA}) $$

分析：

-   improvement rate
-   diversity
-   entropy
-   stagnation

------------------------------------------------------------------------

## 预期结果

不同指标存在相关性。

但是：

单指标解释能力有限。

------------------------------------------------------------------------

# Experiment 2: Search Maturity与ELA Utility关系

## 目标

验证：

Maturity是否比单指标更接近ELA价值。

------------------------------------------------------------------------

## 方法

计算：

$$ \rho(M,U_{ELA}) $$

并绘制：

二维关系图。

------------------------------------------------------------------------

## 分析

观察：

低成熟：

ELA价值低。

中成熟：

ELA价值最高。

高成熟：

ELA价值下降。

------------------------------------------------------------------------

# Experiment 3: Maturity-based Decision

## 目标

验证：

Search Maturity是否可以指导ELA决策。

------------------------------------------------------------------------

## 方法

比较：

### Rule-based

例如：

$$ M_t>\theta $$

执行ELA。

------------------------------------------------------------------------

### ML-based

输入：

Behavior features。

输出：

Utility。

------------------------------------------------------------------------

比较：

-   Utility prediction
-   Decision accuracy

------------------------------------------------------------------------

# Experiment 4: Ablation

## Ablation A

Without Search Maturity

结构：

    Behavior

    ↓

    Utility Prediction

------------------------------------------------------------------------

## Ablation B

With Search Maturity

结构：

    Behavior

    ↓

    Maturity

    ↓

    Utility Prediction

------------------------------------------------------------------------

比较：

-   R2
-   MAE
-   Spearman

验证：

Maturity是否提供额外信息。

------------------------------------------------------------------------

# Experiment 5: Cross-function-family Analysis

目标：

验证不同Landscape类型下：

Maturity-Utility关系是否稳定。

分析：

不同function family：

-   smooth
-   ill-conditioned
-   multimodal
-   composition

------------------------------------------------------------------------

# 6. 可视化设计

## Figure 1

Behavior State Space

展示：

不同搜索阶段。

------------------------------------------------------------------------

## Figure 2

Search Maturity Curve

横轴：

FE ratio

纵轴：

Maturity。

展示：

不同算法的变化。

------------------------------------------------------------------------

## Figure 3

Maturity vs ELA Utility

展示：

倒U关系。

------------------------------------------------------------------------

## Figure 4

Utility Prediction Comparison

比较：

-   single features
-   behavior vector
-   maturity representation

------------------------------------------------------------------------

# 7. 关键评价指标

## Correlation

-   Pearson
-   Spearman

## Prediction

-   MAE
-   RMSE
-   R2

## Decision

-   Accuracy
-   F1
-   AUROC

## End-to-end

-   Optimization performance
-   FE saving
-   Pareto efficiency

------------------------------------------------------------------------

# 8. 可能的审稿质疑与回应

## Q1:

Search Maturity是否只是人为定义？

回应：

通过：

-   correlation analysis
-   ablation
-   prediction improvement

证明其有效性。

------------------------------------------------------------------------

## Q2:

为什么不用单个behavior指标？

回应：

单指标无法描述：

探索稳定化和开发饱和之间的关系。

------------------------------------------------------------------------

## Q3:

Maturity是否等价于convergence？

回应：

比较：

-   convergence indicators
-   maturity indicators

证明二者不同。

------------------------------------------------------------------------

# 9. 最终研究结论目标

证明：

    Optimization Behavior

            |

            v

    Search Maturity

            |

            v

    ELA Utility

            |

            v

    Analysis Selection

形成稳定的信息链。

这使Decision-before-Feature不仅是一个机器学习分类器，而成为一种基于搜索状态的信息感知Landscape
Analysis决策框架。
