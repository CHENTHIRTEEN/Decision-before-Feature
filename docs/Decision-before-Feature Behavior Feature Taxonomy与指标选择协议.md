# Decision-before-Feature Behavior Feature Taxonomy与指标选择协议

## 1. 文档定位

本文档定义 Decision-before-Feature 框架中的 Behavior Feature 输入体系。

核心目标：

构建一种：

> Algorithm-agnostic, low-cost, decision-oriented search behavior
> representation

用于预测：

$$ U_{ELA} $$

即：

执行Landscape Analysis是否具有正收益。

------------------------------------------------------------------------

# 2. Feature设计原则

## Principle 1: Algorithm-agnostic

Behavior Feature必须来自：

优化过程。

不依赖：

-   PSO参数
-   DE参数
-   CMA-ES内部矩阵

原因：

目标是学习：

$$ Search Behavior \rightarrow Analysis Utility $$

而不是：

$$ Algorithm Parameter \rightarrow Decision $$

------------------------------------------------------------------------

## Principle 2: Low-cost

Feature计算成本必须远低于ELA。

允许：

-   population statistics
-   fitness history
-   trajectory

禁止：

-   additional objective evaluations
-   landscape probing

------------------------------------------------------------------------

## Principle 3: Interpretable

每个Feature应该具有明确优化含义。

避免：

直接使用不可解释embedding作为唯一输入。

------------------------------------------------------------------------

# 3. Feature Taxonomy总体结构

Behavior Feature分为五类：

    Search Behavior State

            |

    --------------------------------

    |          |          |          |

    Progress  Diversity  Exploration  Exploitation



                 Communication

------------------------------------------------------------------------

# 4. Category A: Optimization Progress Features

## 4.1 Function Evaluation Ratio

定义：

$$ r_t=\frac{FE_t}{FE_{max}} $$

作用：

描述当前搜索阶段。

注意：

不是直接学习维度或者时间。

------------------------------------------------------------------------

## 4.2 Improvement Rate

定义：

$$ IR_t= \frac{
f_{best}(t-k)-f_{best}(t)
}{k} $$

含义：

当前搜索收益。

高：

仍有优化潜力。

低：

可能进入停滞。

------------------------------------------------------------------------

## 4.3 Improvement Frequency

过去窗口：

改善次数比例。

用于区分：

偶然改善

和

持续改善。

------------------------------------------------------------------------

# 5. Category B: Population Diversity Features

## 5.1 Mean Pairwise Distance

定义：

$$ D_t= \frac{2}{N(N-1)}
\sum_i\sum_j ||x_i-x_j|| $$

含义：

种群空间分散程度。

------------------------------------------------------------------------

## 5.2 Diversity Change Rate

定义：

$$ \Delta D_t=D_t-D_{t-k} $$

解释：

下降：

搜索集中。

增加：

探索增强。

------------------------------------------------------------------------

## 5.3 Population Spread

统计：

-   variance
-   coordinate range
-   centroid movement

------------------------------------------------------------------------

# 6. Category C: Exploration Features

目标：

描述搜索空间探索能力。

------------------------------------------------------------------------

## 6.1 Directional Entropy

来源：

exploration/exploitation行为分析。

含义：

搜索方向的不确定性。

高：

方向分散。

低：

方向一致。

------------------------------------------------------------------------

## 6.2 Search Radius

描述：

当前搜索区域范围。

------------------------------------------------------------------------

## 6.3 Exploration Rate

可定义：

基于：

-   diversity
-   movement
-   entropy

综合形成。

注意：

不作为最终决策标签。

仅作为输入。

------------------------------------------------------------------------

# 7. Category D: Exploitation Features

描述：

开发已有区域的能力。

------------------------------------------------------------------------

## 7.1 Distance Decay

定义：

个体距离最优位置变化。

$$ d_t=||x_i-x_{best}|| $$

观察：

距离是否快速下降。

------------------------------------------------------------------------

## 7.2 Stagnation

定义：

连续无改善次数。

含义：

判断：

开发是否饱和。

------------------------------------------------------------------------

## 7.3 Convergence Rate

描述：

种群收缩速度。

------------------------------------------------------------------------

# 8. Category E: Communication Features

主要用于群智能算法。

目标：

描述：

个体之间信息传播。

------------------------------------------------------------------------

可能指标：

-   neighborhood similarity
-   information sharing intensity
-   leader influence

注意：

由于DE/CMA-ES不存在显式communication，

第一版本可以暂不加入主模型。

作为扩展实验。

------------------------------------------------------------------------

# 9. 推荐第一版Feature集合

为了避免feature explosion。

第一篇论文建议：

## Core Features

### Progress

-   FE ratio
-   improvement rate
-   improvement frequency

### Diversity

-   diversity
-   diversity change rate

### Exploration

-   directional entropy

### Exploitation

-   distance decay
-   stagnation
-   convergence rate

总计：

约8-12个Feature。

------------------------------------------------------------------------

# 10. 不推荐直接加入的Feature

## Algorithm-specific

禁止：

PSO:

-   inertia
-   c1
-   c2

DE:

-   F
-   CR

CMA:

-   covariance

------------------------------------------------------------------------

## High-level ELA Features

禁止：

-   ruggedness
-   modality
-   autocorrelation

原因：

这些属于Landscape Feature。

而Decision-before-Feature需要在ELA之前决策。

------------------------------------------------------------------------

# 11. Feature Selection Protocol

不能直接人工删除。

建议：

三个阶段。

------------------------------------------------------------------------

## Stage 1: Expert-designed core set

使用理论确定：

8-12个核心指标。

------------------------------------------------------------------------

## Stage 2: Feature importance analysis

训练：

RF/XGBoost。

分析：

-   gain importance
-   permutation importance
-   SHAP

------------------------------------------------------------------------

## Stage 3: Compact model

比较：

Full behavior set

vs

Selected behavior set

证明：

少量行为足够。

------------------------------------------------------------------------

# 12. Ablation Design

## Ablation A

Only Progress

验证：

简单优化进度是否足够。

------------------------------------------------------------------------

## Ablation B

Progress + Diversity

验证：

种群结构价值。

------------------------------------------------------------------------

## Ablation C

Progress + Diversity + Exploration/Exploitation

完整行为模型。

------------------------------------------------------------------------

## Ablation D

加入algorithm-specific feature

验证：

算法特征是否降低泛化。

------------------------------------------------------------------------

# 13. Feature与Search Maturity关系

Behavior Feature不是最终目标。

关系：

    Behavior Features

            |

            v

    Search Maturity

            |

            v

    ELA Utility Prediction

            |

            v

    Decision

------------------------------------------------------------------------

# 14. 预期实验假设

H1:

低成本行为指标能够预测ELA Utility。

H2:

算法无关行为比算法参数具有更好的OOD泛化。

H3:

组合Progress + Diversity + Exploration/Exploitation获得最佳性能。

------------------------------------------------------------------------

# 15. 最终冻结方案

Decision输入：

包含：

    FE ratio

    Improvement rate

    Improvement frequency

    Diversity

    Diversity change

    Directional entropy

    Distance decay

    Stagnation

    Convergence rate

不包含：

    Function ID

    Dimension

    Algorithm ID

    Algorithm parameters

    ELA features

目标：

学习：

> Unknown optimization problem下，搜索行为是否已经足以支持Landscape
> Analysis决策。
