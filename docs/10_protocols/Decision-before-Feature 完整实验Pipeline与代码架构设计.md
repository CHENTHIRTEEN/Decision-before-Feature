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

每个checkpoint保存：

    {
    problem_id,

    family_id,

    dimension,

    algorithm,

    seed,

    FE_ratio,

    iteration,


    population,

    fitness,

    best_fitness
    }

------------------------------------------------------------------------

# 5. Checkpoint策略

禁止：

固定100 FE。

采用：

FE ratio。

正式 phase1：

    0.20, 0.25, 0.28, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55, 0.60

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

正式 Selection Reference 必须先对每个共享 state 运行 `continue_current` 与其余三个 portfolio actions，保存 observed action loss；随后用 query features、算法无关 behavior 和连续 remaining budget 训练 multi-output action-loss regressor。静态 problem label 和 nearest performance bucket 不再进入正式生成链。

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

输出：

$$ \hat U_{query} $$

模型：

Baseline:

-   Logistic Regression
-   Random Forest

Main:

-   XGBoost
-   LightGBM

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

-   MAE
-   R2
-   Spearman

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
