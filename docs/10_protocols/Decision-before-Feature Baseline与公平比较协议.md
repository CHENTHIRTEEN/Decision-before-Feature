# Decision-before-Feature Baseline与公平比较协议

> 实现同步（2026-08-11）：Never Query、Always Query、Random Analysis、Traditional AAS、SBS skip reference 和 Time-only Controller 的代码路径已实现；Traditional AAS 与 Always Query 共享“固定 query + 同一 Selector”的等价运行结果，不重复 continuation。VBS 仍由静态 per-problem 完整候选结果单独计算，不以逐状态 `best observed action` 替代。现有表依赖撤回的 trajectory、旧 16 维构念与旧 Ridge controller，不是当前性能证据。三档 query 必须分别重跑本协议后才能形成论文表。详见 `../30_results/phase1_current_results.md`。

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

        Never Query        Always Query       Decision-before-Feature

            |                 |                  |

     Default Solver    Query + Selector      Decision

                                                 |

                                      -------------------

                                      |                 |

                                   No-query         Run Query

                                      |                 |

                                Default Solver    Query + Selector

------------------------------------------------------------------------

# 3. Baseline体系

## Baseline 1: Never Query (No Analysis)

### 定义

完全不执行所评估的固定 landscape-analysis query。

第一篇论文主协议中，Default Optimizer 是仅由 BBOB train 确定的 SBS，probe 也由同一 SBS 产生；Never Query 原生继续当前完整状态，不重启、不调参、不切换算法。

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

# Baseline 2: Always Query

### 定义

所有问题都执行当前 `query_id` 对应的固定 query。

流程：

    Problem

    ↓

    Query Feature Extraction

    ↓

    Algorithm Selection

    ↓

    Optimizer

### 作用

代表传统AAS流程。

用于回答：

> 如果始终调用该固定 query，是否最优？

------------------------------------------------------------------------

# Baseline 3: Random Analysis

## 定义

随机决定是否执行当前固定 query。

例如：

概率：

$$ p_{query}=0.5 $$

### 作用

排除：

简单减少 query 调用带来的偶然收益。

------------------------------------------------------------------------

# Baseline 4: Traditional Algorithm Selection

## 定义

经典：

固定 query + ML Selector。

正式实现使用与 Proposed Method 完全相同的逐状态 action-loss regression Selector，只是在 `phase1_dynamic_budget_event_v1` 输出的共享 decision opportunities 上总是调用，不使用 Decision Module。不得为该 baseline 单独保留静态 problem classifier、nearest bucket 或不同的决策机会集合。

不包含Decision Module。

用于比较：

你的创新是否来自：

Decision-before-Feature。

------------------------------------------------------------------------

# Baseline 5: Single Best Solver (SBS)

## 定义

SBS 只从 BBOB-train 的完整预算 `final_performance.parquet` 计算。该表与 `0.20–0.60` decision trajectory 分开保存；每个 `problem_id × algorithm × seed` 在 `FE=FE_total` 恰好一行。冻结计算顺序为：

1. 对每个 `problem_id × algorithm` 的全部 optimizer seeds 的 `best_fitness` 取算术均值；
2. 在每个 problem 内按 `best_fitness` 越小越好对四个算法排名，并列使用平均排名；
3. 对每个 algorithm 跨 problem 计算平均排名，选择平均排名最小者；若最终平均排名并列，按冻结 portfolio 顺序 `de, pso, cmaes, shade` 决定。

不得从 trajectory 的最后一个 decision state（最大仅约为 `FE_ratio=0.60`）估计完整预算 SBS，也不得先跨 problem 平均原始 loss。

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

VBS 是静态 per-problem 标准上界。共享 state 上从已运行 continuation actions 取最小 loss 时，另称为 `best observed action`；它只用于 selector regret 和潜在性能差诊断，不作为现实 baseline 执行。

------------------------------------------------------------------------

# Baseline 7: Time-only Controller

## 定义

使用与 Proposed Controller 相同的监督学习与 threshold 协议，但输入严格限定为：

$$
X_{time}=\{FE\_ratio\}.
$$

代码使用 `BEHAVIOR_FEATURE_GROUPS["time_only"] = ("bf_fe_ratio",)`；`bf_fe_ratio` 必须逐行等于 `FE_ratio`。

### 公平比较要求

- 与 Proposed Controller 使用同一 materialized dataset、目标列与 function-family split；
- 比较同名模型，或明确逐模型报告，不能给 Time-only 分配更弱的模型候选；
- preprocessing、模型参数和 threshold 只能由 BBOB train 拟合；
- validation 与外部 benchmark 只用于评价；
- 使用同一 query、Selector、总 FE 与 Utility 口径。

### 作用

用于回答：

> Controller 是否只是学会在哪个优化阶段调用固定 landscape-analysis query？

若 Proposed Controller 没有在 held-out families 与外部 benchmark 上稳定优于 Time-only Controller，则不能声称算法无关搜索行为提供了阶段信息之外的预测价值。

------------------------------------------------------------------------

# 4. Proposed Method

## Decision-before-Feature

流程：

    Problem

    ↓

    Cheap Behavior Observation
