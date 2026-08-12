# Decision-before-Feature 完整实验Pipeline与代码架构设计

> 实现同步（2026-08-11）：trajectory、behavior、三档 landscape query、selection reference、utility labels、Decision dataset、模型比较和内部评价模块均已实现。现有正式证据仍须从 trajectory 开始重生成；本文后续目录树中的细粒度文件名属于设计分解，当前实际模块和命令以项目根目录 `README.md` 为准。

## 1. 文档定位

本文档定义 Decision-before-Feature 的完整实验工程架构。

目标：

将论文方法转换为可复现的软件系统。

覆盖：

1.  Benchmark管理
2.  优化轨迹生成
3.  Behavior Feature提取
4.  Offline Utility Label生成
5.  Decision Model训练
6.  OOD测试
7.  结果分析

------------------------------------------------------------------------

# 2. 总体实验Pipeline

                    Benchmark Pool

                          |

                          v

              Optimizer Trajectory Generation

                          |

                          v

                 Trajectory Database

                          |

                          v

              Algorithm-agnostic Behavior Extraction

                          |

                          v

                  Search State Dataset

                          |

                          v

          Shared-state Candidate Action Losses

                          |

                          v

       Continuous-budget Selection Reference

                          |

              -----------------------------

              |                           |

              v                           v

       Offline Utility Label          Behavior Analysis


              |

              v

       Decision Model Training


              |

              v

          OOD Evaluation


              |

              v

     End-to-end Cost-performance Analysis

------------------------------------------------------------------------

# 3. 项目目录设计

推荐结构：

    decision_before_feature/
    ├──AGENTS.md
    
    ├── configs/

    │   ├── benchmark.yaml

    │   ├── optimizer.yaml

    │   ├── feature.yaml

    │   └── model.yaml

    ├── docs

    ├── benchmarks/

    │   ├── bbob/

    │   ├── cec/

    │   └── engineering/


    ├── optimizers/

    │   ├── de/

    │   ├── pso/

    │   ├── cmaes/

    │   └── shade/


    ├── trajectory/

    │   ├── collector.py

    │   ├── recorder.py

    │   └── schema.py


    ├── behavior/

    │   ├── progress.py

    │   ├── diversity.py

    │   ├── exploration.py

    │   ├── exploitation.py

    │   └── extractor.py


    ├── ela/

    │   ├── pflacco_wrapper.py

    │   ├── feature_extraction.py

    │   └── selector.py


    ├── selection_reference/

    │   ├── action_losses.py

    │   ├── model.py

    │   └── build.py

    ├── utility_labels/

    │   ├── generation.py

    │   └── batch_generation.py


    ├── decision/

    │   ├── dataset.py

    │   ├── train.py

    │   ├── predict.py

    │   └── explain.py


    ├── experiments/

    │   ├── rq1_ela_cost.py

    │   ├── rq2_prediction.py

    │   ├── rq3_end_to_end.py

    │   ├── rq4_ood.py

    │   └── rq5_ablation.py


    ├── results/

    └── logs/

------------------------------------------------------------------------

# 4. Trajectory Generation模块

## 4.1 输入

Problem:

    function_id

    family_id

    dimension

    bounds

    budget

Optimizer:

    algorithm

    parameters

    seed

------------------------------------------------------------------------

## 4.2 输出

每个动态采样状态保存：

    {
    problem_id,

    family_id,

    dimension,

    algorithm,

    seed,

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


    population,

    fitness,

    best_fitness,

    optimizer_state_mode
    }

------------------------------------------------------------------------

# 5. 动态状态采样策略

禁止：

固定100 FE。

正式 phase1 冻结为 `phase1_dynamic_budget_event_v1`：

    monitor grid: 0.20--0.60, step 0.01
    budget milestones:
      0.20, 0.22, 0.24, 0.26, 0.28,
      0.30, 0.34, 0.38, 0.42, 0.46,
      0.50, 0.60
    event-only cap: 2 per early/mid/late phase
    event-only minimum actual FE-ratio gap: 0.02
    states per run: 12--18

事件使用 improvement resume、stagnation onset、effective-rank change、elite migration 和 diversity recovery 的冻结阈值与再武装规则。每个跨过至少一个 `0.01` 监测网格的完整原生 update 只判定一次事件；跨过的监测点含里程碑时，以该里程碑为合并行名义节点且不消耗 event-only 配额、最小间隔锚点或 `event_index_in_phase`；不含里程碑时，以最新跨过的监测点为名义节点。冻结的 `population_size=40` 与 `FE_total=1000D` 保证一次 update 不会同时跨过两个预算里程碑。被 gap/quota 抑制落盘的 crossing 仍推进再武装状态。同一完整 update 上的里程碑与多事件只输出一行。`FE_ratio=FE/FE_total`，名义里程碑单独写入 `budget_milestone_ratio`。

整数状态键冻结为 `(split, problem_id, family, dimension, prefix_algorithm, seed, FE)`。浮点 `FE_ratio` 只作 metadata/模型阶段输入，不用作 join key。首轮离线采样不使用模型分数，且正式 estimator/threshold 的每行 `sample_weight=1`。模型冻结后的 Q10 阈值邻近带只用于 online 附加复查，所有策略共享同一附加机会集合。

每次完整 optimizer run 还必须在同 shard 目录输出独立的 `final_performance.parquet`：每个 `problem_id × algorithm × seed` 仅有 `FE=FE_total` 的一行。该终值表与 decision trajectory 隔离，不进入 behavior extraction。训练集 SBS 只从该表计算，顺序固定为“每 problem/algorithm 对全部 seeds 取 `best_fitness` 算术均值 → 每 problem 内排名 → 每 algorithm 跨 problem 平均排名”；最终平均排名并列时按冻结 portfolio 顺序 `de, pso, cmaes, shade` 决定。

同一 shard 的 trajectory 与 `final_performance.parquet` 必须成对发布。覆盖采集期间不得并发启动 behavior extraction、SBS 或其他下游读取；中断留下的 missing/partial pair 必须成对重生成，不得补写单个文件。

------------------------------------------------------------------------

# 6. Behavior Feature模块

输入：

Trajectory checkpoint。

输出：

Behavior vector。

------------------------------------------------------------------------

## Progress

计算：

-   FE ratio
-   improvement rate
-   improvement frequency

------------------------------------------------------------------------

## Diversity

计算：

-   mean distance
-   variance
-   diversity change

------------------------------------------------------------------------

## Exploration

计算：

-   population Wasserstein change rate
-   centroid shift rate and coherence
-   covariance spectral concentration
-   fitness distribution change

------------------------------------------------------------------------

## Exploitation

计算：

-   distance decay
-   stagnation
-   convergence rate

------------------------------------------------------------------------

输出：

    behavior_vector.npy

------------------------------------------------------------------------

# 7. Offline Utility Label模块

No-query 与 Run Query 必须从同一个共享完整 optimizer checkpoint state 生成。

主实验采用 Population Transfer：

- 第一篇论文主协议令 prefix/default 都为训练集 SBS，No-query 原生继续该完整内部状态与 RNG state；全 prefix 行只用于独立稳健性分析；
- Run Query 选择同一算法时也原生继续完整状态；
- 切换算法时只转移算法无关搜索状态；
- 不使用 Best-so-far Warm Start；
- 不复用 query 采样点。

正式 Selection Reference 必须先对每个共享 state 运行 `continue_current` 与其余三个 portfolio actions，保存 observed action loss；随后用 query features、算法无关 behavior 和连续 remaining budget 训练 multi-output action-loss regressor，目标固定为 `statewise_minmax_observed_action_loss`。Selection Reference、Utility、Decision dataset 与在线输出逐行保存 `selected_equals_default`、`selected_equals_prefix` 和 `handoff_required`；静态 problem label、nearest performance bucket 与 selected-vs-default 字符串别名不再进入正式生成链。

## 7.1 No-query 路径

    Problem

    ↓

    Prefix optimizer native state

    ↓

    Performance

保存：

$$ P_{skip} $$

------------------------------------------------------------------------

## 7.2 Run Query 路径

    Problem

    ↓

    Fixed Query

    ↓

    Algorithm Selector

    ↓

    Selected optimizer

    ↓

    Performance

保存：

$$ p_{query} $$

------------------------------------------------------------------------

## 7.3 Utility

生成：

$$ U_{query} = (P_{skip}-p_{query}) - \lambda_T C_T-\lambda_M C_M. $$

Query sampling FE 已通过减少 Query continuation budget 计入 $p_{query}$；Population Transfer 的影响已进入 observed action loss，均不得重复扣除。另保存：

```text
potential_gain_raw = P_skip - P_best_observed
selector_regret_raw = p_query - P_best_observed
performance_gain_raw = potential_gain_raw - selector_regret_raw
```

保存：

    utility_labels.parquet

字段：

    behavior_state

    U_query

    metadata

------------------------------------------------------------------------

# 8. Decision Dataset设计

每个样本：

    {
    behavior_features,

    search_maturity(optional),

    utility_label,

    metadata

    }

metadata：

保存：

-   problem
-   family
-   dimension
-   algorithm
-   seed

但训练时：

不输入metadata。

------------------------------------------------------------------------

# 9. Decision Model模块

## Training

输入：

behavior features

输出：连续 Utility 预测或 `U_query>0` 分类分数。

活动模型候选：

-   LDA；
-   Logistic Regression；
-   Ridge。

按 BBOB-train nested function-family OOF decision utility 选择候选，完整 train family-OOF 冻结 `oof_utility` threshold；BBOB-validation 只作评价。

------------------------------------------------------------------------

## Prediction

部署流程：

    Unknown Problem

    ↓

    Cheap optimization probe

    ↓

    Behavior extraction

    ↓

    Decision Model

    ↓

    Run Query / No-query

------------------------------------------------------------------------

# 10. Experiment模块

## RQ1

Fixed-query cost-benefit

脚本：

    rq1_ela_cost.py

输出：

-   utility distribution
-   negative utility ratio

------------------------------------------------------------------------

## RQ2

Behavior prediction

脚本：

    rq2_prediction.py

输出：

-   nested OOF decision utility
-   AUROC
-   Average Precision
-   Spearman
-   Ridge RMSE

------------------------------------------------------------------------

## RQ3

End-to-end

脚本：

    rq3_end_to_end.py

比较：

-   Never Query
-   Always Query
-   Traditional AAS
-   Proposed

------------------------------------------------------------------------

## RQ4

OOD

脚本：

    rq4_ood.py

测试：

-   unseen family
-   unseen dimension
-   CEC

------------------------------------------------------------------------

## RQ5

Ablation

脚本：

    rq5_ablation.py

------------------------------------------------------------------------

# 11. 数据格式规范

推荐：

Parquet。

原因：

-   支持大规模trajectory
-   保留schema
-   读取速度快

------------------------------------------------------------------------

# 12. 实验配置管理

所有实验必须由yaml控制。

例如：

    config:

    benchmark:
      name: bbob
      dimension: 20


    optimizer:
      name: CMAES


    budget:
      fe_ratio: 0.1


    model:
      name: xgboost

    seed:
      value: 42

------------------------------------------------------------------------

# 13. 可重复性要求

每次运行保存：

    run_id

    config.yaml

    git_commit

    random_seed

    timestamp

------------------------------------------------------------------------

# 14. Vibe/Codex开发约束

## 禁止

1.  修改实验协议。

2.  修改train/test划分。

3.  使用test函数生成训练数据。

4.  将query feature加入Decision输入。

5.  添加baseline但不记录原因。

------------------------------------------------------------------------

## 新增代码要求

每个模块必须：

-   有README
-   有真实小规模验证入口
-   有数据质量检查或一致性检查
-   有配置文件
-   有结果保存

------------------------------------------------------------------------

# 15. 开发顺序建议

## Phase 1

完成trajectory collector。

验证：

四算法可以稳定运行。

------------------------------------------------------------------------

## Phase 2

完成behavior extractor。

验证：

feature分布。

------------------------------------------------------------------------

## Phase 3

完成offline utility label generation。

验证：

utility合理。

------------------------------------------------------------------------

## Phase 4

训练Decision Model。

------------------------------------------------------------------------

## Phase 5

完成OOD和论文实验。

------------------------------------------------------------------------

# 16. 最终工程目标

形成：

    Optimization Experience Dataset

                +

    Behavior Representation

                +

    Offline Utility Label

                +

    Decision Model

                =

    Decision-before-Feature Framework

该系统应支持后续扩展：

-   Progressive ELA
-   Neural Decision
-   Multi-objective Optimization
-   Online Adaptive Analysis
