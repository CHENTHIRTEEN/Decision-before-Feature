# Decision-before-Feature 实验数据生成与算法无关行为建模设计

> 实现同步（2026-08-11）：旧 72 个 BBOB trajectory/behavior shards 由重建式 continuation 与 identity-dependent behavior 生成，已撤回。完整状态与三档 query 一致性检查通过后，仍须从 trajectory 开始全量重生成；本轮修订未启动该重生成。

## 1. 文档定位

本文档补充 Decision-before-Feature 框架中的数据生成部分。

重点解决：

1.  Offline训练数据如何生成？
2.  优化过程记录哪些信息？
3.  如何避免学习到算法特有参数？
4.  如何保证行为表示具有跨算法泛化能力？

------------------------------------------------------------------------

# 2. Offline Learning总体方案

Decision-before-Feature不采用边优化边训练控制器。

采用：

Offline trajectory collection + supervised decision learning。

流程：

    Benchmark Problems

            |

            v

    Multiple Optimizers

    (DE / PSO / CMA-ES / SHADE)

            |

            v

    Optimization Trajectory Database

            |

            v

    Algorithm-agnostic Behavior Extraction

            |

            v

    Offline Utility Label

            |

            v

    Decision Model Training

原因：

## 2.1 Label需要离线获得

Decision模型需要预测：

$$ U_{query} $$

该值需要比较：

-   No-query
-   Run Query

因此必须事后计算。

------------------------------------------------------------------------

## 2.2 避免credit assignment问题

如果在线训练：

    Optimizer

    ↓

    Controller

    ↓

    Decision

    ↓

    Performance

最终性能提升无法区分来自：

-   控制器
-   固定 query
-   算法变化
-   随机因素

因此第一篇工作采用offline更容易形成清晰科学问题。

------------------------------------------------------------------------

# 3. 数据采集目标

目标不是学习：

某个算法什么时候调参数。

目标：

学习：

> 搜索行为是否包含足够信息判断所评估固定 query 的价值。

因此采集：

Algorithm-agnostic Search Behavior。

------------------------------------------------------------------------

# 4. 为什么不能记录算法内部参数

禁止作为Decision输入：

## PSO

-   inertia weight ω
-   c1
-   c2

## DE

-   F
-   CR
-   mutation strategy

## CMA-ES

-   covariance matrix
-   sigma

原因：

这些属于：

Algorithm-specific state。

如果使用：

模型可能学习：

    PSO parameter state

    ↓

    Decision

而不是：

    Search behavior

    ↓

    Decision

导致：

跨算法泛化失败。

------------------------------------------------------------------------

# 5. 推荐记录信息

## 5.1 通用优化状态

所有population-based optimizer均可获得。

### Fitness progress

-   best fitness
-   mean fitness
-   median fitness

其中所有 fitness 相关的尺度化口径优先采用 shift-invariant 稳健尺度，避免目标函数整体平移导致的数值漂移。movement / direction / success 类逐个体统计仅在算法身份稳定时作为诊断数据，主建模仍采用 permutation-invariant 的集合级版本。
### Improvement rate

$$ IR_t= \frac{f_{best}(t-k)-f_{best}(t)}{k} $$

------------------------------------------------------------------------

## 5.2 Population Behavior

### Diversity

描述种群空间覆盖。

例如：

平均距离先按问题边界归一化到单位超立方体后计算：

$$ D_t $$

------------------------------------------------------------------------

### Population spread

包括：

-   variance
-   centroid shift（按搜索空间边界归一化后计算）
-   covariance spectral concentration

------------------------------------------------------------------------

## 5.3 Exploration / Exploitation Behavior

来自算法行为分析研究。

包括：

### Exploration

-   diversity change（基于边界归一化坐标）
-   population Wasserstein change rate（先按搜索空间边界归一化）
-   centroid shift coherence（先按搜索空间边界归一化）

### Fitness distribution

-   quantile improvement fraction
-   mean distribution improvement rate
-   fitness Wasserstein rate

### Exploitation

-   distance decay（基于边界归一化后的 population-best 距离）
-   stagnation
-   convergence speed（基于边界归一化坐标的 diversity 下降）

------------------------------------------------------------------------

## 5.4 Trajectory Features

包括：

-   fitness curve slope
-   improvement frequency
-   change point
-   trajectory stability

------------------------------------------------------------------------

# 6. 时间尺度设计

## 6.1 不推荐固定FE窗口

例如：

每100 FE记录。

原因：

不同：

-   算法
-   维度
-   问题复杂度

具有不同时间尺度。

------------------------------------------------------------------------

# 6.2 推荐FE比例采样

使用：

$$ r=\frac{FE}{FE_{max}} $$

例如：

    0.5%
    1%
    2%
    5%
    10%
    20%
    50%
    100%

优势：

跨维度泛化。

------------------------------------------------------------------------

# 6.3 多尺度行为窗口

状态：

$$ s_t $$

包含：

## Short-term

最近2%预算：

-   improvement

## Medium-term

最近5%预算：

-   diversity change

## Long-term

最近10%预算：

-   stagnation

------------------------------------------------------------------------

# 7. 算法差异问题

不同算法确实具有不同探索-开发转换速度。

例如：

PSO：

较早聚集。

DE：

探索更持续。

CMA-ES：

分布自适应。

因此：

不能简单认为：

    10% budget

    =

    same search phase

------------------------------------------------------------------------

# 8. 如何解决算法行为差异

## 方法1：多算法训练

训练数据包含：

-   DE
-   PSO
-   CMA-ES
-   SHADE

让模型看到多种搜索模式。

------------------------------------------------------------------------

## 方法2：Algorithm ID不作为输入

保存：

algorithm metadata。

但是Decision输入：

不包含algorithm。

目的：

学习：

algorithm-independent behavior。

------------------------------------------------------------------------

## 方法3：Leave-one-algorithm-out验证

例如：

训练：

DE + PSO + CMA-ES

测试：

SHADE

验证：

行为表示是否跨算法。

------------------------------------------------------------------------

# 9. 最终推荐数据格式

每个checkpoint：

``` json
{
problem_id,

function_family,

dimension,

algorithm,

FE,

FE_ratio,

FE_total,

native_updates,

window_statistics,

native_update_history,

effective_window_ratio_w02/w05/w10,

effective_window_fe_w02/w05/w10,

effective_native_updates_w02/w05/w10,


behavior:
{
diversity,

improvement_rate,

population_wasserstein_rate,

centroid_shift_coherence,

covariance_spectral_concentration,

fitness_distribution_change,

distance_decay,

stagnation,

trajectory_features
}

}
```

algorithm字段：

用于分析。

不用于Decision输入。

------------------------------------------------------------------------

# 10. 核心研究假设

如果上述设计成立：

说明：

> 不同算法虽然内部机制不同，但在相同优化问题上会产生具有共享语义的搜索行为状态，而这些状态能够用于判断是否值得执行Landscape
> Analysis。

这正是Decision-before-Feature区别于传统adaptive optimizer的核心。
