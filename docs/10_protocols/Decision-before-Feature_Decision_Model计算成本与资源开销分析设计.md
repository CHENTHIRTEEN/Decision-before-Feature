# Decision-before-Feature Decision Model计算成本与资源开销分析设计

## 1. 文档定位

本文档补充 Decision-before-Feature 研究中的一个重要审稿风险：

> 使用机器学习模型进行决策本身是否引入了过高额外成本？

潜在审稿质疑：

> The proposed decision mechanism introduces an additional machine
> learning model. Does the overhead of this controller outweigh the
> saved cost from avoiding landscape analysis?

因此，需要证明：

1.  Decision Module 本身成本远低于 ELA；
2.  Decision 过程不会抵消跳过 ELA 带来的收益；
3.  方法优化的是总资源消耗，而不是单纯增加一个预测模型。

------------------------------------------------------------------------

# 2. 核心思想

Decision-before-Feature 的目标不是：

> 增加一个复杂机器学习模型。

而是：

> 使用极低成本的信息判断是否值得调用高成本 Landscape Analysis。

传统流程：

    Problem

    ↓

    ELA Feature Extraction

    ↓

    Algorithm Selection

成本：

$$
C_{traditional}=C_{ELA}+C_{selection}
$$

------------------------------------------------------------------------

本文流程：

    Problem

    ↓

    Cheap Behavior Extraction

    ↓

    Decision Model

    ↓

    ELA / Skip

成本：

$$
C_{DBF}=C_{behavior}+C_{decision}+C_{optional\ ELA}
$$

核心目标：

证明：

$$
C_{behavior}+C_{decision}\ll C_{ELA}
$$

------------------------------------------------------------------------

# 3. Decision Module定位

Decision Model不是：

-   优化器；
-   参数控制器；
-   新的算法选择器。

它是：

## Analysis Selection Controller

任务：

预测：

$$
\hat U_{ELA}
$$

其中：

$$
U_{ELA}=PerformanceGain-\lambda Cost
$$

决策：

如果：

$$
\hat U_{ELA}>0
$$

执行ELA。

否则：

跳过ELA。

------------------------------------------------------------------------

# 4. 为什么不能使用复杂模型

如果使用：

-   Transformer；
-   大规模神经网络；
-   强化学习控制器；

可能产生：

    为了减少ELA成本

    ↓

    引入更复杂模型

    ↓

    模型成本接近ELA

这违背研究目标。

因此第一篇论文推荐：

-   Random Forest；
-   XGBoost；
-   LightGBM。

原因：

1.  训练成本低；
2.  推理速度快；
3.  适合tabular behavior data；
4.  支持SHAP解释。

------------------------------------------------------------------------

# 5. Decision阶段成本组成

总决策成本：

$$
C_{decision}=C_{behavior}+C_{model}
$$

其中：

## 5.1 Behavior Extraction Cost

来源：

已有优化状态：

-   population；
-   fitness history；
-   trajectory。

不需要额外目标函数评价。

因此：

$$
FE_{decision}=0
$$

------------------------------------------------------------------------

## 5.2 Model Inference Cost

包括：

-   特征归一化；
-   模型预测。

对于：

Random Forest / XGBoost：

通常为毫秒级。

------------------------------------------------------------------------

# 6. 新增实验：Decision Overhead Analysis

需要回答：

> Is the decision overhead negligible compared with the saved analysis
> cost?

------------------------------------------------------------------------

## Experiment 1: Runtime Breakdown

比较：

### Always ELA

包括：

-   ELA sampling；
-   ELA feature computation；
-   selector。

------------------------------------------------------------------------

### Decision-before-Feature

包括：

-   behavior extraction；
-   model inference；
-   被调用的ELA。

报告：

$$
Ratio=
\frac{C_{decision}}{C_{ELA}}
$$

目标：

证明：

$$
Ratio\ll1
$$

------------------------------------------------------------------------

# 7. Cost-adjusted Utility

不能只比较最终优化性能。

定义：

$$
U_{net}=Performance-\lambda(C_{analysis}+C_{decision})
$$

------------------------------------------------------------------------

## Always ELA

$$
U_A=Performance_A-\lambda C_{ELA}
$$

------------------------------------------------------------------------

## Decision-before-Feature

$$
U_D=Performance_D-\lambda(C_{decision}+C_{ELA\ executed})
$$

目标：

证明：

$$
U_D>U_A
$$

------------------------------------------------------------------------

# 8. 函数评价次数（FE）分析

黑盒优化中，FE是核心成本。

## Always ELA

$$
FE_A=FE_{ELA}+FE_{optimization}
$$

------------------------------------------------------------------------

## Decision-before-Feature

$$
FE_D=FE_{probe}+FE_{optional\ ELA}+FE_{optimization}
$$

需要证明：

Decision阶段：

不引入额外FE。

------------------------------------------------------------------------

# 9. 训练成本与部署成本区分

## Offline Cost

一次性成本：

-   trajectory generation；
-   oracle generation；
-   model training。

------------------------------------------------------------------------

## Online Decision Cost

实际部署成本：

-   behavior extraction；
-   model inference。

论文重点关注：

online decision efficiency。

------------------------------------------------------------------------

# 10. 推荐新增实验

## Experiment A

Decision overhead comparison

比较：

-   Always ELA；
-   Decision-before-Feature。

指标：

-   runtime；
-   FE；
-   memory。

------------------------------------------------------------------------

## Experiment B

Model complexity ablation

比较：

-   Logistic Regression；
-   Random Forest；
-   XGBoost；
-   MLP。

分析：

模型复杂度是否带来实际收益。

------------------------------------------------------------------------

## Experiment C

Cost-performance Pareto

横轴：

$$
Cost
$$

纵轴：

$$
Optimization\ Quality
$$

比较：

-   Never ELA；
-   Always ELA；
-   Proposed。

------------------------------------------------------------------------

# 11. 论文推荐表述

不要写：

> We use machine learning to decide whether ELA is needed.

推荐：

> The decision module introduces negligible overhead because it operates
> on already available search states and requires no additional
> objective evaluations.

中文：

> 决策模块仅利用优化过程中已有搜索状态，不引入额外函数评价，其计算开销远低于
> Landscape Analysis。

------------------------------------------------------------------------

# 12. 最终研究逻辑

    昂贵Landscape Analysis

            ↑

    需要低成本决策机制

            ↑

    已有搜索行为信息

            ↑

    Decision-before-Feature

最终目标：

证明：

> 一个廉价的信息筛选机制可以避免大量无效Landscape
> Analysis，同时保持自动算法选择收益。
