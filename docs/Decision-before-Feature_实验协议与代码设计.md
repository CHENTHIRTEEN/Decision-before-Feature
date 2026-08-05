# Decision-before-Feature 实验协议与代码设计

本文档用于指导实验开发。

目标：

建立可复现、可扩展的Decision-before-Feature实验框架。

------------------------------------------------------------------------

# 1. Overall Pipeline

    Benchmark Problems

            |

            v

    Optimizer Trajectory Generation

            |

            v

    Behavior Feature Extraction

            |

            v

    ELA Oracle Generation

            |

            v

    Decision Model Training

            |

            v

    OOD Evaluation

------------------------------------------------------------------------

# 2. Benchmark Protocol

## 2.1 Training Dataset

推荐：

BBOB。

原因：

-   标准黑盒优化测试
-   函数类型丰富
-   适合行为学习

------------------------------------------------------------------------

## 2.2 Validation/Test Dataset

推荐：

CEC2017

CEC2022

扩展：

Engineering Optimization Problems。

------------------------------------------------------------------------

# 3. Function Split Rules

禁止：

随机instance split。

原因：

shift、rotation、noise版本可能属于同一function family。

------------------------------------------------------------------------

采用：

## Function Family Split

训练：

部分function family。

测试：

未见function family。

------------------------------------------------------------------------

## Cross Benchmark OOD

Train:

BBOB

Test:

CEC

验证：

模型是否学习搜索规律，而不是函数编号。

------------------------------------------------------------------------

# 4. Optimizer Pool

至少包含：

-   Differential Evolution
-   Particle Swarm Optimization
-   CMA-ES
-   SHADE

目的：

避免模型学习单一算法行为。

------------------------------------------------------------------------

# 5. Trajectory Logging

每固定FE保存：

    iteration

    function evaluations

    population position

    population fitness

    best fitness

    diversity

    trajectory information

保存格式：

推荐：

parquet/json。

------------------------------------------------------------------------

# 6. Behavior Feature Pipeline

输入：

optimizer trajectory。

输出：

behavior vector。

包括：

## Progress

-   FE ratio
-   improvement rate
-   convergence speed

## Population

-   diversity
-   spread

## Exploration/Exploitation

-   entropy
-   stagnation
-   distance decay

------------------------------------------------------------------------

# 7. ELA Oracle Generation

离线生成训练标签。

------------------------------------------------------------------------

## Always ELA

流程：

    Problem

    ↓

    ELA

    ↓

    Algorithm Selection

    ↓

    Performance

------------------------------------------------------------------------

## Never ELA

流程：

    Problem

    ↓

    Default Optimizer

    ↓

    Performance

------------------------------------------------------------------------

计算：

    U_ELA =
    Performance Gain
    -
    lambda * ELA Cost

作为label。

------------------------------------------------------------------------

# 8. Decision Model

Baseline:

-   Logistic Regression
-   Random Forest

Main:

-   XGBoost
-   LightGBM

Optional:

-   MLP

------------------------------------------------------------------------

输入：

Behavior Features。

禁止：

输入ELA Feature。

------------------------------------------------------------------------

输出：

$$ \hat U_{ELA} $$

------------------------------------------------------------------------

# 9. Experimental Questions

## RQ1

ELA是否总有收益？

实验：

Cost-benefit analysis。

------------------------------------------------------------------------

## RQ2

搜索行为是否可以预测ELA价值？

指标：

-   R2
-   MAE
-   Spearman

------------------------------------------------------------------------

## RQ3

Decision-before-Feature是否优于传统流程？

比较：

-   Always ELA
-   Never ELA
-   Random Decision
-   Traditional AAS

------------------------------------------------------------------------

## RQ4

是否具有OOD泛化？

训练：

BBOB

测试：

CEC

------------------------------------------------------------------------

## RQ5

为什么有效？

分析：

-   SHAP
-   Feature importance

------------------------------------------------------------------------

# 10. Cost-performance Pareto

横轴：

Total Cost

包括：

-   ELA FE
-   Optimization FE
-   CPU time

纵轴：

Optimization Performance。

比较：

-   Always ELA
-   Never ELA
-   Proposed

目标：

证明：

更优成本性能平衡。

------------------------------------------------------------------------

# 11. Repository Structure

    decision_before_feature/

    ├── benchmarks/

    ├── optimizers/

    ├── trajectory/

    ├── behavior/

    ├── ela/

    ├── decision/

    ├── experiments/

    ├── evaluation/

    └── results/

------------------------------------------------------------------------

# 12. Reproducibility Rules

每个实验保存：

    config.yaml

    random seed

    benchmark version

    optimizer parameters

    model parameters

    timestamp

------------------------------------------------------------------------

# 13. Vibe/Codex开发约束

禁止：

1.  修改实验协议。

2.  使用test函数训练。

3.  使用ELA feature作为Decision输入。

4.  随意增加baseline。

新增实验必须：

-   说明目的
-   记录配置
-   保留随机种子
-   输出结构化结果。

------------------------------------------------------------------------

# 14. 后续扩展

## Progressive ELA

动态决定Feature预算。

## Behavior-aware ELA

融合Landscape和Search Behavior。

## Neural Decision Module

学习复杂搜索状态。
