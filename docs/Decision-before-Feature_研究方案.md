# Decision-before-Feature 研究方案（详细版）

# 1. 研究定位

方向：

Automated Algorithm Selection + Exploratory Landscape Analysis +
Behavior Analysis

核心问题：

传统AAS流程默认：

Feature Extraction always helps。

本文研究：

Feature Extraction itself should be optimized.

------------------------------------------------------------------------

# 2. 科学问题

## Q1

什么时候Landscape Analysis值得执行？

## Q2

优化过程中的搜索行为是否包含问题结构信息？

## Q3

能否建立资源感知的算法选择流程？

------------------------------------------------------------------------

# 3. 方法框架

    Unknown Problem

          |
          v

    Initial Optimization Probe

          |
          v

    Behavior Feature Extraction

          |
          v

    Search Maturity Estimation

          |
          v

    ELA Utility Prediction

          |
          v

    Decision-before-Feature

          |
     ------------------
     |                |

    Skip             ELA

     |                |

    Optimizer     Algorithm Selection

------------------------------------------------------------------------

# 4. Behavior Feature设计

输入必须廉价。

包括：

## Optimization progress

-   FE ratio
-   iteration ratio
-   improvement rate
-   fitness variance

## Population behavior

-   diversity
-   population spread
-   movement distance

## Exploration/Exploitation

-   entropy
-   distance decay
-   stagnation frequency

------------------------------------------------------------------------

# 5. Decision Module

第一阶段：

机器学习。

推荐：

-   Random Forest
-   XGBoost
-   LightGBM

原因：

-   可解释
-   小数据稳定
-   可做SHAP分析

未来：

-   Transformer
-   RL

------------------------------------------------------------------------

# 6. 论文创新点

## Contribution 1

提出新的问题：

是否应该执行Landscape Analysis。

## Contribution 2

提出Behavior-aware Decision-before-Feature框架。

## Contribution 3

建立Search Behavior到ELA Utility之间关系。

## Contribution 4

提出跨benchmark泛化评价协议。

------------------------------------------------------------------------

# 7. 潜在扩展

Progressive ELA:

逐步增加feature预算。

Behavior-aware ELA:

融合Landscape和Trajectory。

Neural Decision:

学习复杂search state。
