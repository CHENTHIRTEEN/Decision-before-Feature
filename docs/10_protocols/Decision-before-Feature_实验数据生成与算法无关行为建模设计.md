# Decision-before-Feature 实验数据生成与算法无关行为建模设计

> 实现同步（2026-08-11）：旧 72 个 BBOB trajectory/behavior shards 由重建式 continuation 与 identity-dependent behavior 生成，已撤回。完整状态与三档 query 一致性检查通过后，仍须从 trajectory 开始全量重生成；本轮修订未启动该重生成。

## 1. 文档定位

本文档补充 Decision-before-Feature 框架中的 `offline trajectory collection` 与 `behavior extraction` 部分。

### 1.1 术语约定

为与 `Behavior Feature Taxonomy` 和 `Decision Model` 两份协议保持一致，本文统一采用以下术语：

- `search trajectory`：单次优化运行按时间展开的状态序列；
- `behavior state`：从 `search trajectory` 中按预算比例、阶段和事件聚合得到的低成本行为表示；
- `query utility`：固定 query 在当前状态下的效用值，记为 $U_{query}$；
- `offline trajectory collection`：面向监督学习的离线轨迹收集；
- `online evaluation`：在与训练一致的流式状态接口上进行推理和切换评估；
- `behavior extraction`：从轨迹中提取算法无关行为特征的过程。

重点解决：

1.  offline training data 如何生成？
2.  优化过程记录哪些信息？
3.  如何避免学习到算法特有参数？
4.  如何保证行为表示具有跨算法泛化能力？

------------------------------------------------------------------------

# 2. Offline Learning总体方案

Decision-before-Feature 不采用边优化边训练控制器。

采用：

`offline trajectory collection` + `supervised decision learning`。

## 1.2 术语对齐规则

- 文中优先使用 `search trajectory` 指代原始运行序列；
- 使用 `behavior state` 指代从轨迹中提取的模型输入；
- 使用 `decision model` 指代监督学习模型；
- 使用 `decision controller` 指代部署态推理模块；
- 使用 `online evaluation` 指代与训练一致的流式评估过程。

流程：

    Benchmark Problems

            |

            v

    Multiple Optimizers

    (DE / PSO / CMA-ES / SHADE)

            |

            v

    search trajectory database

            |

            v

    behavior extraction

            |

            v

    offline utility labeling

            |

            v

    decision model training

            |

            v

    decision model training

原因：

## 2.1 Label需要离线获得

`decision model` 需要预测：

$$ U_{query} $$

该值需要比较：

-   no-query
-   run query

并通过 `decision threshold` 映射为最终决策。

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

所有 fitness 相关尺度化均使用优化器初始化后、任何原生 update 前的已评估 population fitness IQR，避免目标函数整体平移导致的数值漂移。`bf_fitness_diversity_rel` 是唯一的相对 IQR 字段；另一旧字段在文档定义上与其重复，代码却错误地以当前 IQR 自归一化为近常数，现已删除。movement / direction / success 类逐个体统计仅在算法身份稳定时作为诊断数据，主建模采用 permutation-invariant 的集合级版本。冻结的 `DynamoRep-lite` 低成本补充组包括 fitness spread slope、population centroid shift、elite centroid shift、covariance trace ratio、covariance effective rank 与 diversity recovery；这些特征在逐次完整原生 update 历史上计算，不依赖个体跨代身份。`bf_best_distance_fitness_corr` 与 `bf_population_overlap_w05` 不进入主模型，仅保留为 `diagnostic_only`。
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

正式 phase1 不使用全程固定 ratio 列表，而使用 `phase1_dynamic_budget_event_v1`：

    monitor grid: 0.20--0.60, step 0.01
    budget milestones:
      0.20, 0.22, 0.24, 0.26, 0.28,
      0.30, 0.34, 0.38, 0.42, 0.46,
      0.50, 0.60
    event-only states: at most 2 per phase
    event-only minimum actual ratio gap: 0.02
    states per run: 12--18

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

## 6.4 与 Decision Model 的接口对齐

离线数据生成与在线测评必须使用同一套状态表示接口，建议约定如下 streaming API：

- `observe(snapshot)`：接收单次更新或一个预算片段的状态；
- `update_window()`：更新多尺度窗口统计；
- `emit_features()`：输出当前 decision 特征向量；
- `maybe_decide()`：返回是否执行 query 以及当前 decision score。

这样做的目的不是引入在线训练，而是确保离线生成的轨迹切片和 online 决策看到的是同一类信息。

------------------------------------------------------------------------

## 2.3 与 decision model 的接口对齐

离线数据生成与 online evaluation 应复用同一套流式状态接口，建议约定如下术语：

- `budget milestone`：预定义的 FE-ratio 检查点；
- `event-triggered sample`：在状态突变点补充的观测；
- `trajectory window`：按 FE-ratio 对齐的窗口；
- `decision score`：用于是否执行 query 的连续分数；
- `decision threshold`：将 `decision score` 映射为 query 决策的阈值。

推荐实现接口：

- `observe(snapshot)`：接收一个状态快照；
- `update_window()`：更新窗口统计；
- `emit_features()`：输出当前 `behavior state`；
- `maybe_decide()`：输出是否执行 query 及其 `decision score`。

这样做的目的是让离线训练和 online evaluation 对同一种状态空间建模，从而降低 train-test mismatch。

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

# 9. 最终冻结数据格式

每个输出状态：

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

sampling_protocol,

sampling_phase,

sampling_triggers,

is_budget_milestone,

budget_milestone_ratio,

is_event_sample,

monitor_target_ratio,

event_index_in_phase,

event_* flags and metrics,

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

跨 trajectory、behavior、action-loss、Selection Reference、Utility 与 Decision materialization 的状态键为 `(split, problem_id, family, dimension, prefix_algorithm, seed, FE)`。`FE` 是实际整数函数评价数；`FE_ratio=FE/FE_total` 只作 metadata 和模型阶段特征，不作 join key。名义里程碑另存 `budget_milestone_ratio`。

完整预算算法性能另写入同 shard 目录的 `final_performance.parquet`，每个 `problem_id × algorithm × seed` 在 `FE=FE_total` 恰好一行。该表不属于 `0.20–0.60` decision trajectory，不参与 behavior window 或 Decision state join；其用途是冻结训练集 SBS 及静态完整预算性能基准。SBS 先按 `problem_id × algorithm` 对全部 seeds 的终值取算术均值，再逐 problem 排名，最后按 algorithm 跨 problem 平均排名；最终平均排名并列时按冻结 portfolio 顺序 `de, pso, cmaes, shade` 决定。

首轮离线采样不由 decision score 决定，不做事后样本重加权（`sample_weight=1`）。Q10 threshold-neighborhood 只能在模型与 threshold 冻结后用于 online 附加复查，所有策略必须共享相同 decision opportunities。

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
