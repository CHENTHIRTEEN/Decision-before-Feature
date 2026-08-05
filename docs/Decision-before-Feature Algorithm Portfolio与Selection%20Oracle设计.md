# Decision-before-Feature Algorithm Portfolio与Selection Oracle设计

## 1. 文档定位

本文档定义 Decision-before-Feature 框架中的 Algorithm Portfolio 与
Selection Oracle。

核心目标：

构建稳定、公平、可复现的 ELA Utility Oracle。

Oracle 的作用：

不是提出新的算法选择方法，而是提供一个可靠的离线参考：

> 如果执行Landscape Analysis，理论上可以获得多少收益。

因此，Selection Oracle 的质量直接影响 Decision Model 的训练标签质量。

------------------------------------------------------------------------

# 2. 整体流程

传统流程：

    Problem

    ↓

    ELA Feature Extraction

    ↓

    Algorithm Selection Model

    ↓

    Optimizer

Decision-before-Feature：

    Problem

    ↓

    Decision Module

    ↓

          ----------------

          |              |

       Skip ELA       Run ELA

          |              |

    Default       ELA + Selection Oracle

    Optimizer          |

                       v

                 Selected Optimizer

其中：

Selection Oracle只用于离线生成标签。

部署阶段：

不需要执行Oracle。

------------------------------------------------------------------------

# 3. Algorithm Portfolio设计原则

## 3.1 为什么需要Portfolio

ELA的价值来自：

> Feature是否能够帮助选择更合适的优化算法。

因此：

如果Portfolio过小：

例如：

只有DE。

那么：

ELA几乎没有选择空间。

Utility被低估。

如果Portfolio过大：

包含大量相似算法。

会导致：

-   Selection难度增加
-   Oracle不稳定
-   计算成本增加

因此需要：

性能覆盖 + 稳定性之间平衡。

------------------------------------------------------------------------

# 4. 推荐Algorithm Portfolio

第一阶段建议：

连续单目标黑盒优化。

包含：

## Differential Evolution (DE)

特点：

-   全局探索能力强
-   参数稳定
-   常用于black-box optimization

------------------------------------------------------------------------

## CMA-ES

特点：

-   强局部搜索能力
-   适合连续变量
-   对旋转问题鲁棒

------------------------------------------------------------------------

## PSO

特点：

-   群体智能代表算法
-   行为分析方便

------------------------------------------------------------------------

## SHADE / L-SHADE

特点：

-   参数自适应
-   CEC竞赛常用强baseline

------------------------------------------------------------------------

最终Portfolio：

    A =
    {
    DE,
    CMA-ES,
    PSO,
    L-SHADE
    }

------------------------------------------------------------------------

# 5. Algorithm Selection Oracle定义

给定：

问题：

$$ p $$

ELA Feature：

$$ \phi(p) $$

Portfolio：

$$ A $$

Selection模型：

$$ S $$

选择：

$$ a^*=S(\phi(p)) $$

得到：

$$ P_{ELA} $$

------------------------------------------------------------------------

# 6. Selection Model设计

## 6.1 基础方案

使用经典Algorithm Selection模型。

输入：

ELA features。

输出：

algorithm label。

模型：

-   Random Forest
-   XGBoost
-   LightGBM

原因：

-   小样本稳定
-   可解释
-   与ELA研究传统一致

------------------------------------------------------------------------

## 6.2 训练数据生成

对于每个problem：

运行所有算法：

    DE
    CMA-ES
    PSO
    L-SHADE

            |

            v

    Performance Matrix

形成：

Algorithm Portfolio Performance Matrix。

------------------------------------------------------------------------

# 7. Oracle公平性要求

## 7.1 禁止使用测试信息

Selection Oracle训练：

只能使用：

training problems。

测试问题：

只能用于最终评价。

------------------------------------------------------------------------

## 7.2 固定Algorithm Budget

所有算法：

必须：

-   相同FE预算
-   相同运行次数
-   相同停止条件

否则：

Oracle偏向某个算法。

------------------------------------------------------------------------

## 7.3 Random Seed控制

每个：

problem × algorithm

至少：

30 independent runs。

保存：

-   mean
-   median
-   std
-   best

------------------------------------------------------------------------

# 8. Selection Oracle性能评价

Oracle本身需要验证。

比较：

## Single Best Solver (SBS)

整个训练集平均最好的算法。

------------------------------------------------------------------------

## Virtual Best Solver (VBS)

理论上每个问题选择最优算法。

这是上限。

------------------------------------------------------------------------

## ELA Selector

实际Selection Oracle。

目标：

接近VBS。

------------------------------------------------------------------------

指标：

-   ERT
-   final error
-   regret

------------------------------------------------------------------------

# 9. 避免Oracle过强或过弱

## 过弱情况

如果：

ELA Selector≈SBS

说明：

ELA没有产生有效选择。

------------------------------------------------------------------------

## 过强情况

如果：

ELA Selector接近VBS

但是：

训练测试泄漏。

需要检查：

function family split。

------------------------------------------------------------------------

# 10. 与ELA Utility的关系

最终：

No ELA：

$$ P_{skip} $$

ELA：

$$ P_{ELA} $$

其中：

$$ P_{ELA} $$

来自：

ELA + Selection Oracle。

Utility：

$$ U_{ELA} = (P_{skip}-P_{ELA}) - \lambda C_{ELA} $$

------------------------------------------------------------------------

# 11. Selection Oracle实验

## Experiment A

验证Portfolio覆盖能力。

分析：

不同算法在哪些function family占优。

------------------------------------------------------------------------

## Experiment B

验证ELA Selector。

比较：

-   SBS
-   Random Selection
-   ELA Selection
-   VBS

------------------------------------------------------------------------

## Experiment C

验证Utility稳定性。

不同：

-   lambda
-   seed
-   portfolio

------------------------------------------------------------------------

# 12. 后续扩展

## Multi-fidelity Portfolio

不同预算选择不同算法。

------------------------------------------------------------------------

## Dynamic Algorithm Portfolio

根据Search Behavior动态调整候选算法。

------------------------------------------------------------------------

## Joint Decision and Selection

未来：

联合学习：

是否ELA

以及

选择哪个算法。

------------------------------------------------------------------------

# 13. 实现建议

代码模块：

    selection_oracle/

    ├── portfolio.py

    ├── performance_matrix.py

    ├── selector.py

    ├── train_selector.py

    ├── evaluate_selector.py

    └── oracle_generator.py

------------------------------------------------------------------------

# 14. 核心原则总结

1.  Selection Oracle不是论文创新点，而是可靠实验基础。

2.  Portfolio必须覆盖不同搜索行为。

3.  Oracle训练必须严格避免test leakage。

4.  ELA Utility的可信度依赖Selection Oracle稳定性。

5.  Decision-before-Feature的创新重点仍然是：

Analysis Selection，而不是Algorithm Selection。
