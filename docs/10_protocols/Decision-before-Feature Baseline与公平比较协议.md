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

    ↓

    Decision Model

    ↓

    Estimate U_query


    if U_query > 0:

          Fixed Query

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

## 5.2 Query 成本计算

Query 阶段：

包括：

-   sampling FE
-   feature calculation

不能隐藏 query 成本。

------------------------------------------------------------------------

## 5.3 Optimization Budget

Query 方法：

必须扣除分析阶段消耗。

等总 FE 主协议通过 `FE_query_optimization = FE_total - FE_prefix - FE_query` 扣除 sampling FE；Utility 不能再减同一笔 FE。只额外计算 feature/selector runtime 与内存等未进入 final loss 的成本。

否则：

Always Query天然占优势。

------------------------------------------------------------------------

## 5.4 算法切换后的初始化公平性

所有会在共享动态采样状态后调用 selector 的方法：

- Always Query；
- Random Analysis 中的 Run Query 分支；
- Traditional AAS；
- Decision-before-Feature 中的 Run Query 分支。

必须使用同一 Population Transfer 口径。

具体规则：

- 切换后的算法直接使用当前 checkpoint 的 `population`、`fitness` 和 `best_fitness`；
- 切换后的算法重新初始化自身内部状态；
- 不使用 Best-so-far Warm Start 作为主实验默认设置；
- query 采样点不注入后续优化 population；
- `FE_query_optimization = FE_total - FE_prefix - FE_query`。

主结果只使用 `prefix_algorithm=default_algorithm=训练集 SBS` 的行。全 prefix 行若 No-query 需要切到 default，必须以 `skip_switches_from_prefix` 明示，且只能进入独立稳健性分析。

这样比较的是是否执行固定 query 以及 selector 选择的算法，而不是额外重启策略或 query 样本复用策略。

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

主模型选择评价 nested function-family OOF decision mean utility。辅助分数指标统一报告 AUROC、Average Precision 与 Spearman；连续 Utility RMSE 只对 Ridge 报告。策略层报告调用率、utility capture、precision under calls 与 mean decision utility。

------------------------------------------------------------------------

# 8.3 Resource Efficiency

必须报告：

## FE Cost

$$ FE_{total} $$

------------------------------------------------------------------------

## Runtime

包括：

-   固定 query
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

-   Never Query
-   Always Query
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

正式 feature-group 消融固定为 `T0/B1/B2/B3=1/19/25/31`。T0 是只保留 `FE_ratio` 的 Time-only Controller；B1、B2、B3 依次增加冻结的算法无关行为信息。`all_candidates` 仅是 B3 的兼容代码别名，不含诊断字段，也不增加第五个正式组。

四组必须使用相同 Decision dataset、decision opportunities、function-family split、同名模型、train-only preprocessing、threshold 与评价指标。LDA、Logistic Regression、Ridge 属于固定模型候选比较；不同 $\lambda$ 属于预先定义的效用敏感性分析，二者均不得伪装成额外 feature group。

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

为什么不始终执行 query？

回应：

实验展示主 `descriptor_cheap` query 存在 `U_{cheap}\leq0` 的区域。

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

Query Utility distribution。

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

    Always Query

    ↓

    高性能

    但高成本


    Never Query

    ↓

    低成本

    但性能不足


    Decision-before-Feature

    ↓

    自动平衡性能与分析成本

这才符合Resource-aware Automated Algorithm Selection的研究目标。
