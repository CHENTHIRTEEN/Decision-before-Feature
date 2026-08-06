# Decision-before-Feature 实验方案（面向Vibe开发）

> 归档说明：本文档为早期实验草案，仅保留历史脉络；当前实验协议、代码开发和论文主线以 `DEVELOPMENT_DECISIONS.md`、`docs/README.md` 及 `docs/00_master/`、`docs/10_protocols/` 中的有效文档为准。

# 1. 实验目标

验证：

RQ1:

ELA是否总有收益？

RQ2:

Behavior是否可以预测ELA Utility？

RQ3:

Decision-before-Feature是否降低成本？

RQ4:

是否具有OOD泛化？

------------------------------------------------------------------------

# 2. 数据生成

Benchmark:

## Training

BBOB

## Validation/Test

CEC2017

CEC2022

## Optional

Engineering problems

------------------------------------------------------------------------

# 3. 优化算法池

建议：

连续优化算法：

-   DE
-   PSO
-   CMA-ES
-   SHADE

原因：

避免模型学习单一算法行为。

------------------------------------------------------------------------

# 4. 数据采集

每隔固定FE记录：

Population:

-   positions
-   fitness

Trajectory:

-   best fitness history
-   diversity

Behavior:

-   entropy
-   stagnation
-   improvement

------------------------------------------------------------------------

# 5. Label生成

离线计算。

对于每个problem：

执行：

## Always ELA

    Problem
     -> ELA
     -> Algorithm Selection
     -> Performance

## Never ELA

    Problem
     -> Default optimizer

计算：

    U_ELA =
    gain - lambda*cost

作为label。

------------------------------------------------------------------------

# 6. 训练测试划分

禁止：

随机function instance split。

原因：

shift/rotation/noise可能属于同一family。

采用：

## Family-level split

训练：

function families A

测试：

function families B

## Cross benchmark

Train:

BBOB

Test:

CEC

------------------------------------------------------------------------

# 7. Baseline

必须包含：

1.  Always ELA

2.  Never ELA

3.  Random decision

4.  Traditional AAS

------------------------------------------------------------------------

# 8. 评价指标

## Optimization

-   ERT
-   final error
-   success rate

## Decision

-   MAE
-   R2
-   Spearman
-   AUROC

## Cost

-   FE
-   CPU time

------------------------------------------------------------------------

# 9. 核心实验

## Experiment 1

ELA Cost-Benefit Analysis

证明：

ELA并非总值得。

------------------------------------------------------------------------

## Experiment 2

Behavior prediction

证明：

behavior包含ELA价值信息。

------------------------------------------------------------------------

## Experiment 3

End-to-end comparison

比较：

Always ELA

vs

Decision-before-Feature

------------------------------------------------------------------------

## Experiment 4

OOD Generalization

BBOB训练：

CEC测试。

------------------------------------------------------------------------

## Experiment 5

Pareto Analysis

绘制：

Cost-Performance Pareto frontier。

------------------------------------------------------------------------

## Experiment 6

Ablation

删除：

-   exploration features
-   exploitation features
-   progress features

------------------------------------------------------------------------

# 10. 可解释分析

使用：

-   SHAP
-   feature importance

回答：

哪些行为因素决定ELA价值。

------------------------------------------------------------------------

# 11. Vibe开发注意事项

实验代码应模块化：

    data_generation/

    optimizer/

    behavior_features/

    ela/

    decision_model/

    evaluation/

每一步保存：

-   config
-   random seed
-   metadata
-   trajectory

保证可复现实验。
