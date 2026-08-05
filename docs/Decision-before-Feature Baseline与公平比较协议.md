# Decision-before-Feature Baseline与公平比较协议

## 1. 文档定位

本文档定义 Decision-before-Feature 论文中的实验比较协议。

目标：

回答审稿人最关心的问题：

> Proposed method的提升是否来自合理的决策机制，而不是比较不公平？

核心原则：

1.  所有方法共享相同优化预算；
2.  所有方法共享相同benchmark；
3.  所有方法使用相同统计评价协议；
4.  明确区分：
    -   Analysis Selection
    -   Algorithm Selection
    -   Optimization Performance

------------------------------------------------------------------------

# 2. 实验比较总体框架

                        Black-box Problem

                               |

            --------------------------------------

            |                 |                  |

        Never ELA        Always ELA       Decision-before-Feature

            |                 |                  |

     Default Solver    ELA + Selector        Decision

                                                 |

                                      -------------------

                                      |                 |

                                   Skip             Run ELA

                                      |                 |

                                Default Solver    ELA + Selector

------------------------------------------------------------------------

# 3. Baseline体系

## Baseline 1: Never ELA (No Analysis)

### 定义

完全不执行Landscape Analysis。

流程：

    Problem

    ↓

    Default Optimizer

    ↓

    Solution

### 作用

代表：

最低分析成本策略。

用于回答：

> 如果完全不分析问题，性能如何？

------------------------------------------------------------------------

# Baseline 2: Always ELA

### 定义

所有问题都执行ELA。

流程：

    Problem

    ↓

    ELA Feature Extraction

    ↓

    Algorithm Selection

    ↓

    Optimizer

### 作用

代表传统AAS流程。

用于回答：

> 如果无限相信ELA，是否最优？

------------------------------------------------------------------------

# Baseline 3: Random Analysis

## 定义

随机决定是否执行ELA。

例如：

概率：

$$ p_{ELA}=0.5 $$

### 作用

排除：

简单减少ELA调用带来的偶然收益。

------------------------------------------------------------------------

# Baseline 4: Traditional Algorithm Selection

## 定义

经典：

ELA + ML Selector。

不包含Decision Module。

用于比较：

你的创新是否来自：

Decision-before-Feature。

------------------------------------------------------------------------

# Baseline 5: Single Best Solver (SBS)

## 定义

整个训练集合中平均性能最佳算法。

例如：

DE/CMA-ES/SHADE中选择一个。

### 作用

提供强基准。

------------------------------------------------------------------------

# Baseline 6: Virtual Best Solver (VBS)

## 定义

每个问题事后选择最佳算法。

理论上限。

作用：

衡量：

Algorithm Portfolio还有多少提升空间。

------------------------------------------------------------------------

# 4. Proposed Method

## Decision-before-Feature

流程：

    Problem

    ↓

    Cheap Behavior Observation

    ↓

    Decision Model

    ↓

    Estimate U_ELA


    if U_ELA > 0:

          ELA

          Algorithm Selection


    else:

          Default Optimization

------------------------------------------------------------------------

# 5. 公平预算协议

## 5.1 Function Evaluation Budget

所有方法：

必须共享总FE预算。

定义：

$$ FE_{total} = FE_{analysis} + FE_{optimization} $$

------------------------------------------------------------------------

## 5.2 ELA成本计算

ELA阶段：

包括：

-   sampling FE
-   feature calculation

不能隐藏ELA成本。

------------------------------------------------------------------------

## 5.3 Optimization Budget

ELA方法：

必须扣除分析阶段消耗。

否则：

Always ELA天然占优势。

------------------------------------------------------------------------

# 6. Algorithm Portfolio公平协议

所有Algorithm Selection方法：

使用同一Portfolio。

例如：

$$ A= { DE, PSO, CMA-ES, L-SHADE } $$

禁止：

不同方法使用不同候选算法。

------------------------------------------------------------------------

# 7. Benchmark公平协议

## 7.1 Function Family Split

禁止：

random instance split。

原因：

BBOB/CEC中的：

-   shift
-   rotation
-   noise

可能属于同一function family。

------------------------------------------------------------------------

## 7.2 OOD Evaluation

推荐：

训练：

BBOB

测试：

CEC2017

CEC2022

进一步：

engineering problems。

------------------------------------------------------------------------

# 8. 评价指标体系

## 8.1 Optimization Performance

### Final Error

$$ f(x)-f(x^*) $$

------------------------------------------------------------------------

### ERT

Expected Running Time。

------------------------------------------------------------------------

### Success Rate

达到目标精度的问题比例。

------------------------------------------------------------------------

# 8.2 Decision Performance

如果预测：

$$ U_{ELA} $$

评价：

-   MAE
-   RMSE
-   R²
-   Spearman correlation

如果转为decision：

-   Accuracy
-   F1
-   AUROC

------------------------------------------------------------------------

# 8.3 Resource Efficiency

必须报告：

## FE Cost

$$ FE_{total} $$

------------------------------------------------------------------------

## Runtime

包括：

-   ELA
-   Decision model
-   Optimization

------------------------------------------------------------------------

# 9. Cost-performance Pareto Protocol

## 为什么需要

Decision-before-Feature不是追求单一最高性能。

目标：

性能-成本平衡。

------------------------------------------------------------------------

横轴：

$$ Cost $$

纵轴：

$$ Performance $$

比较：

-   Never ELA
-   Always ELA
-   Proposed

------------------------------------------------------------------------

# 10. 统计检验协议

## Repeated Runs

每个：

problem × algorithm

至少：

30 runs。

------------------------------------------------------------------------

## Statistical Tests

推荐：

非参数检验。

例如：

-   Wilcoxon signed-rank
-   Friedman test
-   Holm post-hoc

原因：

优化结果通常非正态。

------------------------------------------------------------------------

# 11. Ablation Protocol

必须包括：

## Ablation A

去除Behavior Feature。

验证：

行为信息是否必要。

------------------------------------------------------------------------

## Ablation B

去除Search Maturity。

直接：

Behavior → Utility。

验证：

中间状态价值。

------------------------------------------------------------------------

## Ablation C

不同Decision模型。

比较：

-   RF
-   XGBoost
-   MLP

------------------------------------------------------------------------

## Ablation D

不同Utility定义。

比较：

不同lambda。

------------------------------------------------------------------------

# 12. OOD公平协议

训练：

不能看到测试函数。

包括：

-   Decision Model
-   Selection Model

------------------------------------------------------------------------

测试：

只进行：

deployment。

------------------------------------------------------------------------

# 13. 常见审稿质疑与回应

## Q1:

是不是只是一个分类器？

回应：

Decision-before-Feature定义了新的Analysis Selection Problem。

------------------------------------------------------------------------

## Q2:

为什么不用全部ELA？

回应：

实验展示ELA存在负Utility区域。

------------------------------------------------------------------------

## Q3:

是否只是benchmark记忆？

回应：

采用function-family split和cross-benchmark OOD。

------------------------------------------------------------------------

## Q4:

收益是否来自更多计算？

回应：

统一FE预算，并报告analysis cost。

------------------------------------------------------------------------

# 14. 实验结果呈现建议

## Table 1

Benchmark statistics。

------------------------------------------------------------------------

## Table 2

Algorithm portfolio。

------------------------------------------------------------------------

## Table 3

Prediction performance。

------------------------------------------------------------------------

## Table 4

End-to-end optimization performance。

------------------------------------------------------------------------

## Figure 1

Framework。

------------------------------------------------------------------------

## Figure 2

ELA Utility distribution。

------------------------------------------------------------------------

## Figure 3

Cost-performance Pareto。

------------------------------------------------------------------------

## Figure 4

OOD generalization。

------------------------------------------------------------------------

## Figure 5

SHAP interpretation。

------------------------------------------------------------------------

# 15. 最终比较逻辑

论文核心不是证明：

Decision-before-Feature永远最好。

而是证明：

    Always ELA

    ↓

    高性能

    但高成本


    Never ELA

    ↓

    低成本

    但性能不足


    Decision-before-Feature

    ↓

    自动平衡性能与分析成本

这才符合Resource-aware Automated Algorithm Selection的研究目标。
