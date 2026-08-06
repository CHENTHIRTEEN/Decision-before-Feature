# Decision-before-Feature Decision Model设计与训练协议

## 1. 文档定位

本文档定义 Decision-before-Feature 框架中的 Decision Model。

目标：

根据低成本搜索行为状态，在不执行ELA之前预测：

$$ U_{ELA} $$

从而决定：

是否执行Landscape Analysis。

核心：

Decision Model不是优化器，而是：

Analysis Selection Controller。

------------------------------------------------------------------------

# 2. Decision Model总体流程

    Search Behavior State

            |

            v

    Feature Representation

            |

            v

    Decision Model

            |

            v

    ELA Utility Prediction

            |

            v

    Decision

输出：

$$ \hat U_{ELA} $$

规则：

如果：

$$ \hat U_{ELA}>0 $$

执行ELA。

否则：

跳过ELA。

------------------------------------------------------------------------

# 3. 输入设计

## 3.1 输入来源

输入来自：

Optimization Trajectory。

不允许：

-   ELA feature
-   Function ID
-   Algorithm parameter

原因：

避免信息泄漏。

------------------------------------------------------------------------

# 3.2 Feature Vector

定义：

$$ x_t $$

包括：

## Progress

-   FE ratio
-   improvement rate
-   improvement frequency

## Diversity

-   population diversity
-   diversity change

## Exploration

-   directional entropy

## Exploitation

-   distance decay
-   stagnation
-   convergence rate

## Optional

Search Maturity。

------------------------------------------------------------------------

# 4. 模型目标设计

## 4.1 Regression形式（推荐）

预测：

$$ \hat U=f_\theta(x) $$

训练目标：

$$ L= (U_{ELA}-\hat U)^2 $$

优势：

保留收益大小。

------------------------------------------------------------------------

## 4.2 Classification形式（辅助）

标签：

$$ y=
\begin{cases}
1,&U_{ELA}>0\\
0,&otherwise
\end{cases}
$$

输出：

$$ P(U>0) $$

缺点：

损失收益信息。

------------------------------------------------------------------------

# 5. 推荐模型体系

## Baseline Model

### Logistic Regression

目的：

线性可分性分析。

------------------------------------------------------------------------

### Random Forest

优势：

-   稳定
-   可解释
-   SHAP支持

------------------------------------------------------------------------

## Main Model

### XGBoost / LightGBM

原因：

适合：

-   tabular behavior data
-   非线性关系
-   小中规模数据

------------------------------------------------------------------------

## Advanced Model

### MLP

用于验证：

神经网络是否进一步提升。

------------------------------------------------------------------------

# 6. 是否加入Search Maturity

设计两个版本。

------------------------------------------------------------------------

## Model A: Direct Utility Prediction

    Behavior Features

            ↓

    Decision Model

            ↓

    U_ELA

------------------------------------------------------------------------

## Model B: Maturity-aware Model

    Behavior Features

            ↓

    Search Maturity

            ↓

    Decision Model

            ↓

    U_ELA

比较：

验证Search Maturity价值。

------------------------------------------------------------------------

# 7. 训练数据构造

每个样本：

对应一个搜索状态。

格式：

    (problem,
    algorithm,
    dimension,
    FE_ratio,
    behavior_state,
    U_ELA)

其中：

algorithm只用于分析。

不进入模型。

------------------------------------------------------------------------

# 8. 数据划分

## 训练集

BBOB：

不同function family。

------------------------------------------------------------------------

## 验证集

用于：

-   hyperparameter tuning
-   threshold selection

------------------------------------------------------------------------

## 测试集

严格OOD：

-   unseen function family
-   CEC benchmark
-   unseen dimension

------------------------------------------------------------------------

# 9. 防止模型学习shortcut

## 禁止输入

### Function identity

否则：

记忆函数。

### Dimension

否则：

学习复杂度。

### Algorithm ID

否则：

学习算法。

### ELA Feature

否则：

循环。

------------------------------------------------------------------------

# 10. Threshold设计

不能简单固定：

$$ 0 $$

需要验证。

------------------------------------------------------------------------

方法1：

Utility threshold

$$ \theta=0 $$

理论方案。

------------------------------------------------------------------------

方法2：

Validation tuning

在validation集：

选择：

$$ \theta $$

使：

Cost-performance最优。

------------------------------------------------------------------------

# 11. 评价指标

## Utility Prediction

-   MAE
-   RMSE
-   R2
-   Spearman

------------------------------------------------------------------------

## Decision

-   Accuracy
-   F1
-   AUROC

------------------------------------------------------------------------

## End-to-end

比较：

-   final optimization performance
-   total FE
-   runtime

------------------------------------------------------------------------

# 12. Ablation设计

## Ablation A

Direct Behavior Model

vs

Maturity-aware Model

验证：

Search Maturity。

------------------------------------------------------------------------

## Ablation B

去除Exploration features

验证：

探索信息贡献。

------------------------------------------------------------------------

## Ablation C

去除Exploitation features

验证：

开发信息贡献。

------------------------------------------------------------------------

## Ablation D

加入Algorithm-specific features

验证：

算法无关设计优势。

------------------------------------------------------------------------

# 13. 可解释性分析

推荐：

SHAP。

分析：

哪些行为指标影响：

$$ \hat U_{ELA} $$

例如：

发现：

-   高停滞
-   中等entropy
-   稳定下降diversity

更可能需要ELA。

------------------------------------------------------------------------

# 14. 训练流程

    Offline Dataset

            |

            v

    Feature Normalization

            |

            v

    Train Decision Model

            |

            v

    Validation Threshold

            |

            v

    OOD Test

            |

            v

    End-to-end Evaluation

------------------------------------------------------------------------

# 15. 最终冻结方案

第一篇论文推荐：

模型：

-   Random Forest
-   XGBoost
-   LightGBM

任务：

Regression预测ELA Utility。

输入：

Algorithm-agnostic Behavior。

输出：

Expected ELA Utility。

决策：

Utility-aware threshold。

目标：

证明：

搜索行为可以支持资源感知Landscape Analysis决策。
