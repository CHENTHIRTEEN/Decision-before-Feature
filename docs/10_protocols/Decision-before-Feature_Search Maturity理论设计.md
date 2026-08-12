# Decision-before-Feature Search Maturity 理论设计

> 实现同步（2026-08-11）：Search Maturity 已改用 permutation-invariant 集合统计。旧 maturity 字段依赖identity-dependent behavior，相关消融与模型结果已失效；新数据重生成前不能评价该公式的预测作用。

## 1. 研究定位

Search Maturity（搜索成熟度）是 Decision-before-Feature
框架中的核心中间概念。

传统自动算法选择关注：

    Problem
       |
       v
    Landscape Feature
       |
       v
    Algorithm Selection

而本文关注：

    Problem

       |
       v

    Early Search Behavior

       |
       v

    Search Maturity

       |
       v

    Should the evaluated fixed landscape query be performed?

Search Maturity 的目标不是预测优化结果，而是判断：

> 当前优化过程是否已经产生足够的信息，使所评估固定 query 的
> $U_{query}$ 大于 0。

------------------------------------------------------------------------

# 2. 与已有概念的区别

## 2.1 与Convergence区别

Convergence回答：

> 搜索是否接近最优。

例如：

best fitness变化。

Search Maturity回答：

> 搜索过程是否已经暴露问题结构。

二者可能不同。

例：

一个简单Sphere：

快速收敛。

但是：

当前固定 query 可能没有价值。

一个复杂多峰问题：

尚未收敛。

但是：

搜索行为已经包含丰富Landscape信息。

------------------------------------------------------------------------

## 2.2 与Exploration/Exploitation区别

Exploration/Exploitation描述：

当前搜索行为。

Search Maturity描述：

当前搜索行为是否达到分析条件。

因此：

Exploration/Exploitation是输入。

Search Maturity是状态。

------------------------------------------------------------------------

# 3. 理论假设

提出三个假设。

## H1

搜索行为随优化过程变化，并包含问题结构信息。

形式：

$$ s_t \rightarrow p $$

其中：

(s_t)为搜索状态。

------------------------------------------------------------------------

## H2

可能存在一个搜索阶段，使当前固定 query 的效用较高。

即：

$$ U_{query}(t) $$

不是单调函数。

------------------------------------------------------------------------

## H3

Search Maturity 可以作为固定 query 决策的候选依据。

即：

$$ M_t \rightarrow U_{query} $$

------------------------------------------------------------------------

# 4. Search State Representation

定义：

$$ s_t $$

由以下信息组成。

------------------------------------------------------------------------

## 4.1 Progress Information

### Function Evaluation Ratio

$$ r_t=\frac{FE_t}{FE_{max}} $$

表示当前预算消耗。

------------------------------------------------------------------------

### Improvement Rate

$$ I_t= \frac{f_{best}(t-k)-f_{best}(t)} {k} $$

表示优化收益。

------------------------------------------------------------------------

## 4.2 Population Information

### Diversity

例如：

平均个体距离先按问题搜索空间边界映射到单位超立方体后计算：

$$ D_t= \frac{2}{N(N-1)}
\sum_i\sum_j\|\tilde{x}_i-\tilde{x}_j\| $$

------------------------------------------------------------------------

### Population Spread

描述搜索区域覆盖程度。

------------------------------------------------------------------------

## 4.3 Behavioral Information

来源：

算法行为分析。

包括：

-   exploration
-   exploitation
-   communication
-   locality

------------------------------------------------------------------------

# 5. Search Maturity构造

## 5.1 为什么不是单调指标？

简单定义：

$$ M_t=f(convergence) $$

存在问题。

因为：

收敛越高不代表越值得执行固定 query。

例如：

过早收敛：

    information gain ↓

------------------------------------------------------------------------

# 5.2 双因素模型

提出：

Search Maturity由两个因素共同决定。

## Exploration Stabilization (ES)

表示：

探索是否从随机状态转变为结构化状态。

输入：

-   diversity下降速度（基于边界归一化坐标）
-   population Wasserstein变化率的稳定程度（先按搜索空间边界归一化）
-   covariance spectral concentration
-   centroid shift coherence（先按搜索空间边界归一化）

当前可执行形式令：

$$
g_+(z)=\frac{\max(z,0)}{1+\max(z,0)},
\qquad
g_-(z)=\frac{1}{1+\max(z,0)}.
$$

其中 fitness 相关输入使用 shift-invariant 稳健尺度：以优化器初始化后、任何原生 update 前的已评估 population fitness IQR 作为 baseline，而不是以首个输出状态、均值或原始标准差作尺度。`bf_fitness_diversity_rel` 是唯一的当前 IQR 相对初始化 IQR 字段；另一旧字段在文档定义上与其重复，代码却错误地以当前 IQR 自归一化为近常数，现已删除。movement / direction / success 类逐个体量只在可稳定保留身份语义的算法上作为诊断信息；主成熟度只接受 permutation-invariant 的集合级代理量。DynamoRep-lite 其余六项为 `bf_fitness_spread_slope_w05`、`bf_population_centroid_shift_w05`、`bf_elite_centroid_shift_w05`、`bf_covariance_trace_ratio_w05`、`bf_covariance_effective_rank_w05` 与 `bf_diversity_recovery_w05`，并在完整原生 update 历史上按预算归一化窗口计算。`bf_best_distance_fitness_corr` 与 `bf_population_overlap_w05` 仅保留为诊断字段，不进入 Search Maturity 主体计算。

则：

$$
ES_t=\operatorname{mean}\left(
g_+(-\text{diversity slope}),
g_-(\text{population Wasserstein rate}),
\text{covariance spectral concentration},
\text{centroid shift coherence}
\right).
$$

当5%窗口anchor不存在时，$ES_t$ 与由其派生的maturity为缺失值，不使用当前集合形状单独填充时间变化信息。

定义：

$$ ES_t \in [0,1]$$

高：

搜索模式稳定。

------------------------------------------------------------------------

## Exploitation Saturation (XS)

表示：

开发是否达到饱和。

输入：

-   stagnation
-   improvement decay
-   local concentration

当前 $XS_t$ 仍由stagnation、improvement-rate saturation、distance decay、convergence rate和elite concentration的变换均值构成，不引入算法内部状态。

定义：

$$ XS_t \in [0,1]$$

高：

继续搜索收益降低。

------------------------------------------------------------------------

# 5.3 Mature Window

定义：

$$ M_t=ES_t(1-XS_t) $$

辅助字段 `bf_explore_exploit_ratio` 使用：

$$
E_t=\operatorname{mean}\left(
g_+(\text{diversity}),
1-\text{covariance spectral concentration},
g_+(\text{population Wasserstein rate})
\right),
$$

$$
X_t=\operatorname{mean}\left(
\text{centroid shift coherence},
g_-(\text{elite concentration}),
g_+(\text{distance decay}),
g_+(\text{convergence rate})
\right),
\qquad
R_t=\frac{E_t}{X_t+\epsilon}.
$$

解释：

## Early stage

$$ ES低 $$

搜索未形成结构。

------------------------------------------------------------------------

## Middle stage

$$ ES高, XS低 $$

最佳分析窗口。

------------------------------------------------------------------------

## Late stage

$$ XS高 $$

搜索可能已经过度收缩。

------------------------------------------------------------------------

# 6. Search Maturity估计方法

## 方法A：Explicit Index

直接计算：

$$ M_t $$

优点：

-   可解释
-   易分析

缺点：

需要人工设计。

------------------------------------------------------------------------

## 方法B：Latent Representation（本轮不实现）

学习式潜在表示会改变 Decision Model 构念、训练成本和候选范围，不进入第一篇论文的活动协议。当前只使用显式、算法无关且 permutation-invariant 的 behavior 与 Search Maturity 派生字段；如后续单独研究 latent representation，必须另设研究问题与预定义比较，不得与本轮 LDA、Logistic Regression、Ridge 结果混报。

------------------------------------------------------------------------

## 方法C：Supervised Estimation

使用Query Utility作为监督：

$$ g(s_t)\rightarrow U_{query} $$

间接获得成熟度。

------------------------------------------------------------------------

# 7. 实验验证

需要证明：

## Experiment 1

Search Maturity与Query Utility存在关系。

分析：

Spearman correlation。

------------------------------------------------------------------------

## Experiment 2

比较：

Without maturity

vs

With maturity

------------------------------------------------------------------------

## Experiment 3

不同function family下：

分析成熟窗口差异。

------------------------------------------------------------------------

# 8. 预期贡献

Search Maturity不是新的Landscape Feature。

而是：

> A decision-oriented representation of optimization process information
> sufficiency.

它连接：

Behavior Analysis

和

Fixed Landscape Query Decision。
