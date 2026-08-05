# Decision-before-Feature ELA Utility Oracle 构建协议

# 1. Oracle目标

Decision-before-Feature需要监督信号：

模型需要知道：

> 当前问题是否值得执行ELA。

因此需要构造：

ELA Utility Oracle。

------------------------------------------------------------------------

# 2. Oracle基本思想

对于同一个问题：

构造两个策略。

## Strategy A: Skip ELA

直接优化。

流程：

    Problem

    ↓

    Default Optimizer

    ↓

    Final Performance

得到：

$$ P_{skip} $$

------------------------------------------------------------------------

## Strategy B: Run ELA

执行：

    Problem

    ↓

    ELA

    ↓

    Algorithm Selection

    ↓

    Selected Optimizer

    ↓

    Final Performance

得到：

$$ P_{ELA} $$

------------------------------------------------------------------------

# 3. Utility定义

## 3.1 Performance Gain

对于最小化问题：

$$ G=P_{skip}-P_{ELA} $$

如果：

$$ G>0 $$

表示ELA提升性能。

------------------------------------------------------------------------

## 3.2 Analysis Cost

ELA成本：

$$ C_{ELA} $$

包括：

### Function Evaluation Cost

ELA采样：

$$ FE_{ELA} $$

------------------------------------------------------------------------

### Feature Computation Cost

包括：

-   PCA
-   nearest neighbor
-   meta-model

------------------------------------------------------------------------

### Runtime Cost

CPU time。

------------------------------------------------------------------------

## 3.3 Final Utility

定义：

$$ U_{ELA}=G-\lambda C_{ELA} $$

其中：

($\lambda$)

控制性能和成本权衡。

------------------------------------------------------------------------

# 4. 为什么不用简单标签？

错误方式：

    ELA improves?

    Yes/No

问题：

丢失收益规模。

例如：

Problem A:

提升100%，成本100。

Problem B:

提升1%，成本100。

二者都会得到Yes。

但是价值不同。

------------------------------------------------------------------------

因此推荐：

Regression：

预测：

$$ U_{ELA} $$

------------------------------------------------------------------------

# 5. Algorithm Selection Oracle

ELA路径必须固定。

推荐：

使用标准Algorithm Portfolio。

例如：

-   DE
-   PSO
-   CMA-ES
-   SHADE

流程：

    ELA Features

    ↓

    Selection Model

    ↓

    Best optimizer

------------------------------------------------------------------------

# 6. Default Optimizer选择

需要固定。

推荐：

两种设置。

## Setting A

Single Best Solver

使用训练集表现最佳算法。

------------------------------------------------------------------------

## Setting B

General baseline

例如：

CMA-ES或者DE。

------------------------------------------------------------------------

实验中必须报告：

不同default选择对结果影响。

------------------------------------------------------------------------

# 7. Lambda设置

不能随意设置。

推荐：

## Multi-lambda Analysis

例如：

$$ \lambda \in {0,0.25,0.5,1,2} $$

观察：

Decision稳定性。

------------------------------------------------------------------------

## Normalized Utility

由于性能和成本量纲不同：

先归一化：

$$ U= \Delta P_{norm} - \lambda C_{norm} $$

------------------------------------------------------------------------

# 8. Oracle Generation Pipeline

    For each problem:

        Run optimizer trajectories

        Generate behavior states


        Run Skip-ELA pipeline

        Record P_skip


        Run ELA pipeline

        Record P_ELA


        Calculate:

        U_ELA


        Save:

        behavior state

        utility label

------------------------------------------------------------------------

# 9. Data Split Requirement

Oracle生成必须遵守：

function family split。

禁止：

同一个function family同时出现在train/test。

原因：

避免：

-   shifted leakage
-   rotated leakage
-   noisy variant leakage

------------------------------------------------------------------------

# 10. Oracle Quality Validation

必须验证：

## Stability

不同random seed：

Utility是否稳定。

------------------------------------------------------------------------

## Sensitivity

不同：

-   optimizer portfolio
-   lambda

是否改变结论。

------------------------------------------------------------------------

# 11. Decision Model Training

输入：

behavior state。

输出：

$$ \hat U_{ELA} $$

损失：

$$ MSE(U,\hat U) $$

决策：

$$ \hat U>0 $$

------------------------------------------------------------------------

# 12. 最终评价

比较：

## Always ELA

性能最高但成本高。

## Never ELA

成本最低。

## Proposed

自动权衡。

指标：

-   optimization quality
-   total FE
-   runtime
-   Pareto efficiency

------------------------------------------------------------------------

# 13. 关键实验问题

RQ1:

ELA Utility是否存在明显分布？

RQ2:

Behavior是否能预测Utility？

RQ3:

Oracle-based Decision是否减少无效ELA？

RQ4:

Utility定义是否对lambda鲁棒？
