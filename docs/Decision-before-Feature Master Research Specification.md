# Decision-before-Feature Master Research Specification

## 0. Document Purpose

本文档是 Decision-before-Feature 项目的最高层研究规范。

用途：

1. 指导论文研究设计；
2. 指导后续 Vibe/Codex 实验开发；
3. 固化方法、数据、实验和工程约束；
4. 避免开发过程中偏离论文核心问题。

核心原则：

> 本研究不是设计新的优化算法，而是研究在黑盒优化中，Landscape Analysis
> 本身是否值得执行。

---

# 1. Research Positioning

## 1.1 Research Area

所属领域：

- Automated Algorithm Selection (AAS)
- Exploratory Landscape Analysis (ELA)
- Metaheuristic Behavior Analysis
- Resource-aware Optimization

---

## 1.2 Core Research Question

传统流程：

    Problem

    ↓

    Landscape Feature Extraction

    ↓

    Algorithm Selection

    ↓

    Optimizer

隐含假设：

    Feature extraction is always beneficial

本文提出：

    Analysis itself should be selected.

即：

> 在执行Landscape Analysis之前，判断是否值得付出分析成本。

---

# 2. Problem Definition

## 2.1 Analysis Selection Problem

给定黑盒问题：

$$
p
$$

决策：

$$
d\in\{0,1\}
$$

其中：

0:

Skip Landscape Analysis

1:

Run Landscape Analysis

目标：

最大化：

$$
Utility(d,p)
$$

---

# 3. Core Framework

整体框架：

    Unknown Black-box Problem

    |

    v

    Cheap Optimization Probe

    |

    v

    Algorithm-agnostic Behavior Extraction

    |

    v

    Search Maturity Estimation

    |

    v

    ELA Utility Prediction

    |

    v

    Decision-before-Feature

    ------------------

    |                |

    Skip ELA        Run ELA

    |                |

    Default Solver   ELA + Algorithm Selection

---

# 4. Offline Learning Strategy

## 4.1 Decision

采用：

Offline trajectory collection + supervised learning。

不采用：

Online controller training。

---

## 4.2 Reason

原因：

1. ELA Utility需要离线计算；
2. 避免credit assignment问题；
3. 保证Analysis Selection问题独立。

---

# 5. Optimization Experience Dataset

## 5.1 Optimizer Pool

必须包含多种搜索机制：

- Differential Evolution
- Particle Swarm Optimization
- CMA-ES
- SHADE/L-SHADE

目的：

学习通用搜索行为。

---

# 6. Algorithm-agnostic Behavior Representation

## 6.1 禁止输入

不允许：

Algorithm-specific parameters。

例如：

PSO:

- inertia
- c1
- c2

DE:

- F
- CR

CMA-ES:

- covariance
- sigma

原因：

避免模型学习算法身份。

---

## 6.2 Behavior Feature Taxonomy

### Progress

- FE ratio
- improvement rate
- improvement frequency

### Diversity

- population diversity
- diversity change

### Exploration

- directional entropy
- movement range

### Exploitation

- distance decay
- stagnation
- convergence rate

---

# 7. Search Maturity

## Definition

Search Maturity表示：

> 当前搜索过程是否已经产生足够的信息，使Landscape Analysis具有价值。

---

## Difference

  概念                       含义

---

  Convergence                是否接近最优
  Exploration/Exploitation   搜索行为
  Search Maturity            分析价值

---

## Conceptual Model

由两个因素组成：

### Exploration Stabilization

探索是否形成稳定结构。

### Exploitation Saturation

开发是否过度饱和。

成熟度不是单调增加：

存在最佳分析窗口。

---

# 8. ELA Utility Oracle

## 8.1 Skip ELA

得到：

$$
P_{skip}
$$

---

## 8.2 Run ELA

流程：

    Problem

    ↓

    ELA Feature Extraction

    ↓

    Algorithm Selection

    ↓

    Optimizer

得到：

$$
P_{ELA}
$$

---

## 8.3 Utility

定义：

$$
U_{ELA} = (P_{skip}-P_{ELA}) -\lambda C_{ELA}
$$

其中：

成本包括：

- Sampling FE
- Feature computation
- Runtime

---

# 9. Decision Model

## Input

Algorithm-agnostic behavior state。

不包含：

- Function ID
- Dimension
- Algorithm ID
- ELA Feature

---

## Output

预测：

$$
\hat U_{ELA}
$$

---

## Recommended Models

Baseline:

- Logistic Regression
- Random Forest

Main:

- XGBoost
- LightGBM

Optional:

- MLP

---

# 10. Dataset Protocol

## Training

BBOB：

Dimensions:

- 10D
- 20D
- 40D

Algorithms:

- DE
- PSO
- CMA-ES
- SHADE

---

## Validation

BBOB:

50D

---

## Testing

OOD:

- CEC2017
- CEC2022
- Engineering problems

---

# 11. Function Split Rules

禁止：

random instance split。

原因：

shift/rotation/noise可能属于同一function family。

采用：

Function-family split。

---

# 12. Checkpoint Protocol

禁止：

固定100 FE。

采用：

FE ratio。

推荐：

    0.5%
    1%
    2%
    5%
    10%
    20%
    50%
    100%

---

# 13. Baseline Protocol

必须包含：

## Never ELA

最低分析成本。

## Always ELA

传统ELA流程。

## Random Analysis

随机决策。

## Traditional AAS

ELA + Selector。

## SBS

Single Best Solver。

## VBS

Virtual Best Solver。

---

# 14. Evaluation Protocol

## Optimization Metrics

- Final error
- ERT
- Success rate

## Decision Metrics

- MAE
- RMSE
- R2
- Spearman
- AUROC

## Resource Metrics

- FE cost
- Runtime

---

# 15. Core Research Questions

## RQ1

ELA是否总有收益？

验证：

ELA negative utility。

---

## RQ2

Behavior是否可以预测ELA Utility？

---

## RQ3

Decision-before-Feature是否减少无效ELA？

---

## RQ4

是否具有跨benchmark泛化能力？

---

## RQ5

为什么有效？

使用：

- SHAP
- Feature importance

---

# 16. Required Ablations

## A1

Without Search Maturity

## A2

Without Exploration Features

## A3

Without Exploitation Features

## A4

Algorithm-specific features

验证：

算法无关行为表示优势。

---

# 17. Repository Specification

    decision_before_feature/

    ├── configs/

    ├── benchmarks/

    ├── optimizers/

    ├── trajectory/

    ├── behavior/

    ├── ela/

    ├── oracle/

    ├── decision/

    ├── experiments/

    ├── evaluation/

    └── results/

---

# 18. Vibe/Codex Development Rules

禁止：

1. 修改benchmark split；
2. 使用test数据训练；
3. 输入ELA feature到Decision Model；
4. 输入algorithm-specific parameter；
5. 未记录配置新增实验；
6. 删除失败实验结果。

---

# 19. Development Order

Phase 1:

Trajectory Collector

Phase 2:

Behavior Extractor

Phase 3:

ELA Utility Oracle

Phase 4:

Decision Model

Phase 5:

OOD Evaluation

Phase 6:

Paper Experiment Reproduction

---

# 20. Final Research Statement

本文最终希望证明：

    Optimization Experience

    ↓

    Algorithm-agnostic Search Behavior

    ↓

    Search Maturity

    ↓

    ELA Utility

    ↓

    Analysis Selection

    ↓

    Resource-efficient Algorithm Selection

核心贡献：

> Landscape Analysis itself should become an object of optimization.
