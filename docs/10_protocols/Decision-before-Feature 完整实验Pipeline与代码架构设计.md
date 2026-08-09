# Decision-before-Feature 完整实验Pipeline与代码架构设计

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


    ├── oracle/

    │   ├── generate_utility.py

    │   ├── portfolio.py

    │   └── selector_training.py


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

例如：

    0.005

    0.01

    0.02

    0.05

    0.10

    0.20

    0.50

    1.00

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

-   directional entropy
-   movement range

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

Skip ELA 与 Run ELA 必须从同一个共享 checkpoint state 生成。

主实验采用 Population Transfer：

- 使用 checkpoint population、fitness 和 best fitness 继续优化；
- 切换算法时只转移算法无关搜索状态；
- 不使用 Best-so-far Warm Start；
- 不复用 ELA 采样点。

## 7.1 Skip ELA路径

    Problem

    ↓

    Default optimizer

    ↓

    Performance

保存：

$$ P_{skip} $$

------------------------------------------------------------------------

## 7.2 Run ELA路径

    Problem

    ↓

    ELA

    ↓

    Algorithm Selector

    ↓

    Selected optimizer

    ↓

    Performance

保存：

$$ P_{ELA} $$

------------------------------------------------------------------------

## 7.3 Utility

生成：

$$ U_{ELA} = (P_{skip}-P_{ELA}) - \lambda C_{ELA} $$

保存：

    oracle_dataset.parquet

字段：

    behavior_state

    U_ELA

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

$$ \hat U_{ELA} $$

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

    Run ELA / Skip ELA

------------------------------------------------------------------------

# 10. Experiment模块

## RQ1

ELA Cost-benefit

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

-   Never ELA
-   Always ELA
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

4.  将ELA feature加入Decision输入。

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
