# Decision-before-Feature Decision Model计算成本与资源开销分析设计

> 实现同步（2026-08-11）：旧成本—性能报告依赖撤回的 trajectory、旧 landscape 构念与 Ridge controller，不是当前证据。三档 query 必须分别重跑本协议，见 `../30_results/phase1_current_results.md`。

## 1. 文档定位

本文档补充 Decision-before-Feature 研究中的一个重要审稿风险：

> 使用机器学习模型进行决策本身是否引入了过高额外成本？

潜在审稿质疑：

> The proposed decision mechanism introduces an additional machine
> learning model. Does the overhead of this controller outweigh the
> saved cost from avoiding landscape analysis?

因此，需要证明：

1.  Decision Module 本身成本相对所评估固定 query 足够小；
2.  Decision 过程不会抵消避免无效 query 调用带来的资源差；
3.  方法优化的是总资源消耗，而不是单纯增加一个预测模型。

------------------------------------------------------------------------

# 2. 核心思想

Decision-before-Feature 的目标不是：

> 增加一个复杂机器学习模型。

而是：

> 使用低成本行为信息判断是否值得调用所评估的固定 landscape-analysis query。

传统流程：

    Problem

    ↓

    Query Feature Extraction

    ↓

    Algorithm Selection

成本：

$$
C_{traditional}=C_{query}+C_{selection}
$$

------------------------------------------------------------------------

本文流程：

    Problem

    ↓

    Cheap Behavior Extraction

    ↓

    Decision Model

    ↓

    Run Query / No-query

成本：

$$
C_{DBF}=C_{behavior}+C_{decision}+C_{optional\ query}
$$

核心目标：

证明：

$$
C_{behavior}+C_{decision}\ll C_{query}
$$

------------------------------------------------------------------------

## 2.1 Online behavior sampling protocol

在线测评中的行为采样频率定义为：

```text
decision-check frequency
```

含义：

- 每个 checkpoint 都是 behavior observation 点；
- 每个 checkpoint 也是 controller、Random Analysis 和 Always Query 可以触发固定 query 的决策点；
- Always Query 在当前 sampling protocol 的第一个 checkpoint 后触发；
- Random Analysis 在每个 decision-check point 独立判断是否触发；
- controller 在每个 decision-check point 使用 behavior features 预测是否触发。

主在线测评使用训练 / label 同口径 checkpoint ratios：

```text
0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60
```

密集采样仅作为敏感性分析：

```text
0.20, 0.225, 0.25, 0.275, 0.28, 0.30, 0.325, 0.35, 0.375,
0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60
```

在 query 尚未触发时，密集 checkpoint 只观察同一个连续 optimizer state，不改变同 seed 的原生优化轨迹。密集采样仍只作为决策检查频率敏感性分析，因为额外检查点会改变 controller、Random Analysis 和 Always Query 的触发机会；它不替代主在线测评。

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
\hat U_{query}
$$

其中：

$$
U_{query}=PerformanceGain-\lambda_T TimeCost-\lambda_M MemoryCost
$$

等总 FE 条件下，query sampling FE 已通过减少 Query continuation budget 进入 PerformanceGain；Population Transfer 的影响也已进入 selected action loss，二者不得再次扣除。

决策：

如果：

$$
\hat U_{query}>0
$$

执行固定 query。

否则：

不执行 query。

------------------------------------------------------------------------

# 4. 为什么不能使用复杂模型

如果使用：

-   Transformer；
-   大规模神经网络；
-   强化学习控制器；

可能产生：

    为了减少固定 query 成本

    ↓

    引入更复杂模型

    ↓

    模型成本接近固定 query

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

### Always Query

包括：

-   query sampling；
-   query feature computation；
-   selector。

------------------------------------------------------------------------

### Decision-before-Feature

包括：

-   behavior extraction；
-   model inference；
-   被调用的固定 query。

报告：

$$
Ratio=
\frac{C_{decision}}{C_{query}}
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

## Always Query

$$
U_A=Performance_A-\lambda C_{query}
$$

------------------------------------------------------------------------

## Decision-before-Feature

$$
U_D=Performance_D-\lambda(C_{decision}+C_{query\ executed})
$$

目标：

证明：

$$
U_D>U_A
$$

------------------------------------------------------------------------

# 8. 函数评价次数（FE）分析

黑盒优化中，FE是核心成本。

## Always Query

$$
FE_A=FE_{query}+FE_{optimization}
$$

------------------------------------------------------------------------

## Decision-before-Feature

$$
FE_D=FE_{probe}+FE_{optional\ query}+FE_{optimization}
$$

需要证明：

Decision阶段：

不引入额外FE。

------------------------------------------------------------------------

# 9. 训练成本与部署成本区分

## Offline Cost

一次性成本：

-   trajectory generation；
-   offline utility label generation；
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

-   Always Query；
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

-   Never Query；
-   Always Query；
-   Proposed。

------------------------------------------------------------------------

# 11. 论文推荐表述

不要写：

> We use machine learning to decide whether the evaluated fixed landscape query should be executed.

推荐：

> The decision module introduces negligible overhead because it operates
> on already available search states and requires no additional
> objective evaluations.

中文：

> 决策模块仅利用优化过程中已有搜索状态，不引入额外函数评价；其计算开销与所评估
> 固定 query 的采样评价及特征计算时间分别报告。

------------------------------------------------------------------------

# 12. 最终研究逻辑

    所评估的固定 landscape-analysis query

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
