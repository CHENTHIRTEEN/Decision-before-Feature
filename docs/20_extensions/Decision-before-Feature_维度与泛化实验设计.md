# Decision-before-Feature 维度与泛化实验设计

> 状态同步（2026-08-11）：主 BBOB 仍为 10D/20D/40D；CEC2017 外部配置已冻结为 10D/30D/50D，但完整外部结果尚未生成。CEC2022 与工程问题维度仍待单独冻结。

状态说明：

本文档作为维度泛化扩展实验笔记保留，不定义当前主协议。

当前主协议以 `DEVELOPMENT_DECISIONS.md` 和
`docs/00_master/Decision-before-Feature Master Research Specification.md`
为准：

- BBOB train / validation 均使用 10D / 20D / 40D；
- 当前 COCO `bbob` suite 不支持 50D；
- BBOB 50D / 100D 只能作为另设扩展实验，不能混入主 validation。

## 1. 问题背景

函数维度会影响：

-   搜索难度
-   行为分布
-   收敛速度
-   固定 query 的效用

因此需要合理设计训练和测试维度。

------------------------------------------------------------------------

# 2. 基本原则

Decision-before-Feature学习：

$$ Behavior \rightarrow Query\ Utility $$

不是：

$$ Dimension \rightarrow Decision $$

因此：

dimension不应该成为主要预测依据。

------------------------------------------------------------------------

# 3. 推荐维度设置

## Training

BBOB：

维度：

    10D
    20D
    40D

目的：

让模型学习：

不同规模下稳定行为。

------------------------------------------------------------------------

## Validation

BBOB：

    10D
    20D
    40D

用于：

模型调参与阈值选择。

注意：

    50D 不进入当前 BBOB 主 validation。

------------------------------------------------------------------------

## Test

Cross benchmark：

CEC2017 / CEC2022

维度：

    10D
    30D
    50D

------------------------------------------------------------------------

# 4. 为什么不固定单一维度

如果：

训练20D

测试20D

模型可能学习：

dimension-specific pattern。

不能证明：

behavior generalization。

------------------------------------------------------------------------

# 5. 为什么不直接10D训练100D测试

差距过大。

失败可能来自：

distribution shift过强。

因此：

采用渐进式泛化。

------------------------------------------------------------------------

# 6. Dimension作为分析因素

虽然不作为主要输入。

但是结果分析应该报告：

不同维度：

-   Decision accuracy
-   Utility prediction
-   Cost saving

------------------------------------------------------------------------

# 7. FE预算设置

不推荐：

所有维度固定FE。

推荐：

预算随维度缩放。

例如：

$$ FE \propto D $$

或者采用：

BBOB标准预算。

------------------------------------------------------------------------

# 8. 泛化实验体系

## Experiment 1

IID：

同benchmark同维度。

目的：

验证基本能力。

------------------------------------------------------------------------

## Experiment 2

Dimension generalization：

训练：

10/20/40D

测试：

50/100D

------------------------------------------------------------------------

## Experiment 3

Cross benchmark OOD：

BBOB训练

CEC测试。

------------------------------------------------------------------------

## Experiment 4

Cross algorithm generalization：

Leave-one-algorithm-out。

------------------------------------------------------------------------

# 9. 最终冻结方案

Training:

    BBOB

    D={10,20,40}

    Algorithms:

    DE
    PSO
    CMA-ES
    SHADE

Testing:

    CEC2017
    CEC2022

    D={10,30,50}

输入：

只使用：

-   behavior
-   progress
-   trajectory

不使用：

-   function ID
-   dimension
-   algorithm parameter
-   query feature

------------------------------------------------------------------------

# 10. 核心目标

最终证明：

模型学习的是：

    Optimization behavior

    ↓

    Analysis usefulness

而不是：

    Function identity

    ↓

    Decision
