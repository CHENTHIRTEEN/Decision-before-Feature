# Decision-before-Feature Master Research Specification

> 实现同步（2026-08-11）：完整 optimizer-state continuation 已替换 checkpoint 重建机制。旧 BBOB labels、模型和评价结果已撤回；必须从 trajectory 开始重生成后再确定主模型。Landscape query 同时改为三档预定义配置：`descriptor_cheap` 是第一篇论文唯一主 query，`pflacco_standard` 与 `pflacco_broad` 只作配置稳健性实验。实现状态见 `../../README.md`、`../../PROJECT_HANDOFF.md` 与 `../10_protocols/Decision-before-Feature_Landscape_Query三档配置与数据契约.md`。

## 0. Document Purpose

本文档是 Decision-before-Feature 项目的最高层研究规范。

用途：

1. 指导论文研究设计；
2. 指导后续 Vibe/Codex 实验开发；
3. 固化方法、数据、实验和工程约束；
4. 避免开发过程中偏离论文核心问题。

核心原则：

> 本研究不是设计新的优化算法，而是研究在黑盒优化中，所评估的固定
> landscape-analysis query 是否值得执行。

构念边界：当前主 query 是 16 维自定义低成本描述符，不代表完整 ELA、完整 pflacco 或一般意义上的所有 Landscape Analysis。第一篇论文不得把三档结果外推到未评价的 pflacco feature groups、NeurELA、Deep-ELA 或其他 landscape representation。

---

# 1. Research Positioning

## 1.1 Research Area

所属领域：

- Automated Algorithm Selection (AAS)
- Landscape analysis and Exploratory Landscape Analysis (ELA)
- Metaheuristic Behavior Analysis
- Resource-aware Optimization

---

## 1.2 Core Research Question

传统流程：

    Problem

    ↓

    Fixed Landscape-query Feature Extraction

    ↓

    Algorithm Selection

    ↓

    Optimizer

隐含假设：

    The evaluated feature query is always beneficial

本文提出：

    A fixed analysis query should be selected conditionally.

即：

> 在执行预先定义的 landscape-analysis query 之前，判断是否值得付出该配置的采样与计算成本。

---

# 2. Problem Definition

## 2.1 Analysis Selection Problem

给定黑盒问题：

$$
p
$$

决策：

$$
d\in\{0,1\}
$$

其中：

0:

No-query

1:

Run the evaluated fixed query

目标：

最大化：

$$
Utility(d,p)
$$

---

# 3. Core Framework

整体框架：

    Unknown Black-box Problem

    |

    v

    Cheap Optimization Probe

    |

    v

    Algorithm-agnostic Behavior Extraction

    |

    v

    Search Maturity Estimation

    |

    v

    Query Utility Prediction

    |

    v

    Decision-before-Feature

    ------------------

    |                |

    No-query        Run Query

    |                |

    Default Solver   Query + Algorithm Selection

---

# 4. Offline Learning Strategy

## 4.1 Decision

采用：

Offline trajectory collection + supervised learning。

不采用：

Online controller training。

---

## 4.2 Reason

原因：

1. Query Utility需要离线计算；
2. 避免credit assignment问题；
3. 保证Analysis Selection问题独立。

---

# 5. Optimization Experience Dataset

## 5.1 Optimizer Pool

必须包含多种搜索机制：

- Differential Evolution
- Particle Swarm Optimization
- CMA-ES
- SHADE/L-SHADE

目的：

学习通用搜索行为。

---

# 6. Algorithm-agnostic Behavior Representation

## 6.1 禁止输入

不允许：

Algorithm-specific parameters。

例如：

PSO:

- inertia
- c1
- c2

DE:

- F
- CR

CMA-ES:

- covariance
- sigma

原因：

避免模型学习算法身份。

---

## 6.2 Behavior Feature Taxonomy

### Progress

- FE ratio
- improvement rate
- improvement frequency

### Diversity

- population diversity
- diversity change

### Exploration

- population Wasserstein change rate
- centroid shift and shift coherence
- covariance spectral concentration

### Fitness Distribution

- quantile improvement fraction
- mean distribution improvement rate
- one-dimensional fitness Wasserstein rate

### Exploitation

- distance decay
- stagnation
- convergence rate

---

## 6.3 Relation to Prior Behavioral Metrics

本项目的behavior representation受两类metaheuristic behavior analysis文献启发：

- exploitation behavior diagnostics；
- metaheuristic behavioral similarity analysis。

但当前指标不是对已有论文指标的逐式复现。

关键口径：

1. *How do metaheuristics exploit?* 中的distance-to-reference decay、directional entropy和stagnation indicators主要用于分析和调节late-stage exploitation behavior，并使用iteration窗口和exploitation phase划分；本文不采用需要跨代个体对应关系的directional entropy，也不实现其direction bins，而以permutation-invariant集合分布变化指标替代。
2. *Determining Metaheuristic Similarity Using Behavioral Analysis* 中的behavioral characteristics主要用于whole-run algorithm similarity，包括diversity/accuracy/convergence/locality/communication/evaluation-effort等整段搜索特征。
3. 本项目只继承其中可解释、低成本、算法无关的行为语义，并改写为checkpoint-level、FE-ratio-normalized、permutation-invariant behavior state。

因此论文主线必须表述为：

> inspired by prior behavioral analysis metrics

或：

> adapted into budget-normalized algorithm-agnostic behavior states

不得表述为：

> directly using the metrics from prior work

当前主模型不使用：

- true global optimum；
- function identity；
- query feature；
- whole-run knee-point extraction；
- STN/IN similarity metrics；
- DBSCAN frequency map。

这些内容可作为相关工作或扩展实验背景，不进入Decision Model主输入。

---

# 7. Search Maturity

## Definition

Search Maturity表示：

> 当前搜索过程是否已经产生足够的信息，使所评估的固定 query 具有价值。

---

## Difference

  概念                       含义

---

  Convergence                是否接近最优
  Exploration/Exploitation   搜索行为
  Search Maturity            分析价值

---

## Conceptual Model

由两个因素组成：

### Exploration Stabilization

探索是否形成稳定结构。

### Exploitation Saturation

开发是否过度饱和。

成熟度不是单调增加：

存在最佳分析窗口。

---

# 8. Offline Utility Label

No-query 与 Run Query 的 paired continuation 使用同一完整 optimizer checkpoint state。

主实验固定为 Population Transfer：

- 第一篇论文主协议令 prefix/default 都等于训练集 SBS，No-query 原生继续该 SBS 的 population、fitness、内部动态量、generation、best-so-far 与 RNG state；
- 若 Run Query 仍选择 prefix algorithm，则从同一完整状态原生继续，但后续优化预算扣除 `FE_query`；
- 若 selector 切换算法，新算法重新初始化自身内部状态；
- 跨算法初始化只转移 checkpoint population、fitness 与 best-so-far position，并明确记为 population transfer；
- 不使用 Best-so-far Warm Start；
- 不复用 query 采样点。

全 prefix trajectory 只用于 cross-probe robustness、leave-one-probe-out 与 algorithm-agnostic 泛化。正式标签必须保存 `selected_equals_default`、`selected_equals_prefix`、`skip_switches_from_prefix`、`no_query_algorithm` 和 `handoff_type`；其中 `no_query_algorithm=default_algorithm`，`handoff_type=query_transition_mode`。多 prefix 数据中的 `same_algorithm` 不得解释为“继续当前算法”。

## 8.1 No-query

得到：

$$
P_{skip}
$$

---

## 8.2 Run Query

流程：

    Problem

    ↓

    Query Feature Extraction

    ↓

    Algorithm Selection

    ↓

    Optimizer

得到：

$$
p_{query}
$$

正式 Selection Reference 不是 problem 级静态分类器。对每个共享 checkpoint state，离线运行 `continue_current` 与其余 portfolio actions，记录连续 action loss；Selector 使用 query features、算法无关 behavior 与连续 `remaining_budget_ratio` 预测各动作损失并选择预测值最小的动作。训练行使用按 function family 的交叉拟合预测，validation/test 不参与拟合。

---

## 8.3 Utility

定义：

$$
U_{query} = (P_{skip}-p_{query}) -\lambda_T C_T-\lambda_M C_M
$$

其中：

Query sampling FE 已通过减少 Run Query 分支的后续优化预算进入 $p_{query}$，不得再次扣除。$C_T$ 与 $C_M$ 只表示尚未进入 performance loss 的 feature/selector runtime 与额外内存成本。

逐状态最佳已观测动作只用于诊断分解：

$$
P_{skip}-p_{query}
=
(P_{skip}-P_{best\ observed})
-(p_{query}-P_{best\ observed}).
$$

跨算法 Population Transfer 的影响已包含在 observed action loss 中，不作为额外减项重复计入。

---

# 9. Decision Model

## Input

Algorithm-agnostic behavior state。

不包含：

- Function ID
- Dimension
- Algorithm ID
- Query Feature

---

## Output

预测：

$$
\hat U_{query}
$$

---

## Recommended Models

活动候选固定为：

- LDA classification：预测 `U_query > 0` 的分类分数；
- Logistic Regression classification：预测 `U_query > 0` 的分类分数；
- Ridge regression：预测连续 `U_query`。

不继续把 Random Forest、XGBoost、LightGBM、MLP 或其变体加入 Decision Model 活动模型搜索。Selection Reference 中固定的 action-loss Random Forest regression 属于不同组件，不受本条影响。

主模型按 BBOB-train 上 nested function-family OOF decision mean utility 选择。每个外层 family fold 的 threshold 只由对应内层 family-OOF 分数拟合；随后用完整 BBOB-train family-OOF 分数冻结 `oof_utility` threshold，并在完整 BBOB-train 重拟合模型。BBOB-validation 只用于冻结后的性能评价。

---

# 10. Dataset Protocol

## Training

BBOB：

Dimensions:

- 10D
- 20D
- 40D

Algorithms:

- DE
- PSO
- CMA-ES
- SHADE

---

## Validation

BBOB：

- function families: 5, 9, 13, 14, 19, 24
- dimensions: 10D / 20D / 40D

注意：

- 当前 COCO `bbob` suite 不支持 50D。
- BBOB 50D 不进入主协议。
- 如需 50D / 100D 泛化，必须另设扩展实验并选择 COCO 支持的 suite。

---

## Testing

OOD:

- CEC2017
- CEC2022
- Engineering problems

---

# 11. Function Split Rules

禁止：

random instance split。

原因：

shift/rotation/noise可能属于同一function family。

采用：

Function-family split。

---

# 12. Checkpoint Protocol

禁止：

固定100 FE。

采用：

FE ratio。

正式 phase1 checkpoint ratios：

    20%, 25%, 28%, 30%, 35%, 40%, 45%, 50%, 55%, 60%

Selection Reference 将扣除 query sampling FE 后的 `remaining_budget_ratio` 作为连续输入，不使用 nearest bucket。

---

# 13. Baseline Protocol

必须包含：

## Never Query

最低分析成本。

## Always Query

固定 query 的传统始终调用流程。

## Random Analysis

随机决策。

## Traditional AAS

Query + Selector。

## SBS

Single Best Solver。

## VBS

Virtual Best Solver。

VBS 保留为标准静态算法选择上界；共享状态上已运行候选动作的最小 loss 另称为 `best observed action`，不得与 VBS 或现实可部署方法混称。

## Time-only Controller

输入严格为 $X=\{FE\_ratio\}$，并与 Proposed Controller 使用同名模型、同一 train-only preprocessing/threshold、同一 held-out family 和外部评价口径。用于判断主 Controller 是否只学习了调用阶段。

---

# 14. Evaluation Protocol

## Optimization Metrics

- Final error
- ERT
- Success rate

## Decision Metrics

- 主选择：nested function-family OOF decision mean utility
- 辅助：AUROC、Average Precision、Spearman
- 连续 Utility 回归：Ridge 的 RMSE
- 策略：query call rate、utility capture、precision under calls、最终优化性能

分类概率或判别分数不与连续 Utility 直接计算 RMSE。

## Resource Metrics

- FE cost
- Runtime

---

# 15. Core Research Questions

## RQ1

主 `descriptor_cheap` query 是否在所有状态都有净收益？

验证：

`U_{cheap} \leq 0` 的状态比例、效应量与区间。

---

## RQ2

Behavior是否可以预测Query Utility？

---

## RQ3

Decision-before-Feature 是否减少无效 query 调用？

---

## RQ4

是否具有跨benchmark泛化能力？

---

## RQ5

为什么有效？

使用：

- SHAP
- Feature importance

---

# 16. Required Ablations

## A1

Without Search Maturity

## A2

Without Exploration Features

## A3

Without Exploitation Features

## A4

Algorithm-specific features

验证：

算法无关行为表示优势。

---

# 17. Repository Specification

    decision_before_feature/

    ├── configs/

    ├── benchmarks/

    ├── optimizers/

    ├── trajectory/

    ├── behavior/

    ├── landscape_queries/

    ├── utility_labels/

    ├── selection_reference/

    ├── decision/

    ├── experiments/

    ├── evaluation/

    └── results/

---

# 18. Vibe/Codex Development Rules

禁止：

1. 修改benchmark split；
2. 使用test数据训练；
3. 输入query feature到Decision Model；
4. 输入algorithm-specific parameter；
5. 未记录配置新增实验；
6. 删除失败实验结果。

---

# 19. Development Order

Phase 1:

Trajectory Collector

Phase 2:

Behavior Extractor

Phase 3:

Offline Utility Label

Phase 4:

Decision Model

Phase 5:

OOD Evaluation

Phase 6:

Paper Experiment Reproduction

---

# 20. Final Research Statement

本文最终希望证明：

    Optimization Experience

    ↓

    Algorithm-agnostic Search Behavior

    ↓

    Search Maturity

    ↓

    Query Utility

    ↓

    Analysis Selection

    ↓

    Resource-efficient Algorithm Selection

核心贡献：

> The decision to execute an evaluated fixed landscape-analysis query should itself be optimized.

该结论只覆盖 `descriptor_cheap` 主配置；若 standard/broad 方向一致，只能进一步表述为“在三个预定义 query 配置上具有稳健性”。若方向不一致，必须报告 representation dependence，不能重新定义 query 或选择性隐藏结果。
