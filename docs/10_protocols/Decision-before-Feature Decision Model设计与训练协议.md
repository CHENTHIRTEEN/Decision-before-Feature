# Decision-before-Feature Decision Model设计与训练协议

> 实现同步（2026-08-11）：旧 18 模型比较依赖重建式 continuation 标签，已撤回正式证据资格。活动候选现固定为 LDA、Logistic Regression 与 Ridge；完整状态 trajectory 与 utility labels 重生成后，按 nested function-family OOF decision utility 重新选择，不预设任一候选胜出。BBOB-validation 只作冻结评价。

## 1. 文档定位

本文档定义 Decision-before-Feature 框架中的 `decision model`。

### 1.1 术语约定

为与 `Behavior Feature Taxonomy` 和 `实验数据生成` 两份协议保持一致，本文统一采用以下术语：

- `search trajectory`：单次优化运行中的状态序列；
- `behavior state`：从 `search trajectory` 聚合得到的低成本输入表示；
- `query utility`：固定 query 在当前状态下的效用值，记为 $U_{query}$；
- `decision model`：基于 `behavior state` 预测 `query utility` 的模型；
- `decision controller`：输出是否执行 query 的控制器；
- `online evaluation`：在流式状态接口上进行推理、阈值判定与切换评估；
- `decision score`：用于 query 决策的连续分数，记为 $s(x)$。

目标：

根据低成本 `behavior state`，在执行固定 query 之前预测：

$$ U_{query} $$

从而决定：

是否执行所评估的固定 landscape-analysis query。

核心：

`decision model` 不是优化器，而是：

`decision controller`。

## 1.2 术语对齐规则

- 文中优先使用 `decision model` 指代监督学习组件；
- 使用 `decision controller` 指代上线后的决策执行模块；
- 使用 `behavior state` 指代面向模型输入的行为表示；
- 使用 `search trajectory` 指代行为表示的原始来源；
- 使用 `online evaluation` 指代部署态评估过程。

------------------------------------------------------------------------

# 2. Decision Model总体流程

    Search Behavior State

            |

            v

    Feature Representation

            |

            v

    Decision Model

            |

            v

    Query Utility Prediction

            |

            v

    Decision

输出是连续 Utility 预测或 `U_query>0` 分类分数，统一记为 decision score $s(x)$。部署规则固定为 $s(x)>\theta_{OOF}$ 时执行 query，其中 $\theta_{OOF}$ 只由 BBOB-train family-OOF 分数拟合。

------------------------------------------------------------------------

# 3. 输入设计

## 3.1 输入来源

输入来自：

Optimization Trajectory。

不允许：

-   query feature
-   Function ID
-   Algorithm parameter

原因：

避免信息泄漏。

------------------------------------------------------------------------

# 3.2 Feature Vector

定义：

$$ x_t $$

包括：

fitness 相关输入采用 shift-invariant 稳健尺度，以优化器初始化后、任何原生 update 前的已评估 population fitness IQR 归一化。`bf_fitness_diversity_rel` 是唯一的当前 IQR 相对初始化 IQR 字段；另一旧字段在文档定义上与其重复，代码却错误地以当前 IQR 自归一化为近常数，现已删除。DynamoRep-lite 的其余 6 项为 fitness-spread slope、population/elite centroid shift、covariance trace ratio、covariance effective rank 与 diversity recovery。当前行为输出/正式输入/诊断字段计数冻结为 34/31/3，`T0/B1/B2/B3=1/19/25/31`；T0 严格只含 `bf_fe_ratio`。

## Progress

-   FE ratio
-   improvement rate
-   improvement frequency

## Diversity

-   population diversity
-   diversity change

## Exploration

-   population Wasserstein change rate
-   centroid shift coherence
-   covariance spectral concentration

## Fitness Distribution

-   quantile improvement fraction
-   mean distribution improvement rate
-   fitness Wasserstein rate

## Exploitation

-   distance decay
-   stagnation
-   convergence rate

## Optional

Search Maturity。

------------------------------------------------------------------------

# 4. 模型目标设计

## 4.1 Regression形式

预测：

$$ \hat U=f_\theta(x) $$

训练目标：

$$ L= (U_{query}-\hat U)^2 $$

优势：

保留收益大小。

------------------------------------------------------------------------

## 4.2 Classification形式

标签：

$$ y=
\begin{cases}
1,&U_{query}>0\\
0,&otherwise
\end{cases}
$$

输出：

$$ P(U>0) $$

分类目标不直接拟合 Utility 大小，成本差异由 OOF decision-utility threshold 选择体现。分类分数不得直接与连续 Utility 计算 RMSE。

------------------------------------------------------------------------

# 5. 固定模型体系

活动候选严格为：

1. LDA classifier：`LinearDiscriminantAnalysis()`；
2. Logistic Regression classifier：`C=1.0`、`class_weight="balanced"`；
3. Ridge regression：`alpha=1.0`。

三个候选均使用 Pipeline 内的 BBOB-train median imputation 与 standard scaling；每个 OOF fit fold 独立拟合 preprocessing。Random Forest、XGBoost、LightGBM、MLP、SVM、核近似和额外特征工程不进入活动 Decision 候选或超参数搜索。Selection Reference 的 Random Forest action-loss regression 是固定下游组件，不属于 Decision Model 候选。

------------------------------------------------------------------------

# 6. 是否加入Search Maturity

设计两个版本。

------------------------------------------------------------------------

## Model A: Direct Utility Prediction

    Behavior Features

            ↓

    Decision Model

            ↓

    U_query

------------------------------------------------------------------------

## Model B: Maturity-aware Model

    Behavior Features

            ↓

    Search Maturity

            ↓

    Decision Model

            ↓

    U_query

比较：

验证Search Maturity价值。

------------------------------------------------------------------------

# 7. 训练数据构造

每个样本：

对应一个搜索状态。

格式：

    (problem,
    algorithm,
    dimension,
    FE_ratio,
    behavior_state,
    U_query)

其中：

algorithm只用于分析。

不进入模型。

第一篇论文主训练表只保留 `prefix_algorithm == default_algorithm ==` 训练集 SBS 且 `skip_switches_from_prefix == false` 的状态。完整多 prefix 表单独用于 cross-probe robustness、leave-one-probe-out 和 algorithm-agnostic 泛化，不进入主模型拟合、主 threshold 选择或主结果汇总。

`selected_equals_default`、`selected_equals_prefix`、`handoff_required` 与 `skip_switches_from_prefix` 只用于数据范围检查和分层报告，同样不进入模型输入。活动 Decision 数据和报告不生成 selected-vs-default 字符串别名。

------------------------------------------------------------------------

# 8. 数据划分

## 训练集

BBOB：

不同function family。

------------------------------------------------------------------------

## 验证集

只用于冻结后的内部性能评价，不参与：

- preprocessing 拟合；
- 模型或候选选择；
- threshold 拟合；
- feature-group、checkpoint 或 query 配置改选。

------------------------------------------------------------------------

## 测试集

严格OOD：

-   unseen function family
-   CEC benchmark
-   unseen dimension

------------------------------------------------------------------------

# 9. 防止模型学习shortcut

## 禁止输入

### Function identity

否则：

记忆函数。

### Dimension

否则：

学习复杂度。

### Algorithm ID

否则：

学习算法。

### Query Feature

否则：

循环。

------------------------------------------------------------------------

# 10. 模型选择与 Threshold 设计

## 10.1 嵌套 function-family OOF 模型选择

- 外层：BBOB-train families 的 5-fold GroupKFold，用于评价每个候选的 decision utility；
- 内层：每个外层 fit 部分再做 4-fold family OOF，只用内层 OOF score 与 Utility 拟合该外层 threshold；
- 主选择指标：拼接全部外层 holdout 决策后的 mean decision utility；
- 指标相同时按预先固定候选顺序 LDA、Logistic Regression、Ridge 决定，不读取 validation。

## 10.2 冻结部署 threshold

选模之外，对完整 BBOB-train 做 5-fold family OOF，使用每行仅由其他 train families 拟合的 score 选择：

$$
\theta_{OOF}=\arg\max_{\theta}\sum_i \mathbf{1}[s_i^{OOF}>\theta]U_i.
$$

Utility 和相同时选择调用行更少的 threshold。随后在完整 BBOB-train 上重拟合估计器，冻结 $\theta_{OOF}$，再评价 BBOB-validation 与外部 benchmark。`theta=0` 只作为固定参考，不参与主选择。

分阶段 threshold 只能从 BBOB-train OOF 信息拟合并作为预先定义的稳健性分析；不得在 validation threshold grid 上选择。

------------------------------------------------------------------------

# 11. 评价指标

## 主选择

- nested function-family OOF decision mean utility。

## 辅助分数指标

- AUROC；
- Average Precision；
- Spearman。

## 连续 Utility 回归

- Ridge RMSE。

LDA 与 Logistic Regression 的分类分数不报告连续 Utility RMSE。

## 决策策略

- query call rate；
- utility capture；
- precision under calls；
- mean decision utility。

------------------------------------------------------------------------

## End-to-end

比较：

-   final optimization performance
-   total FE
-   runtime

------------------------------------------------------------------------

# 12. Ablation设计

## Time-only baseline（必须报告）

定义：

$$
X_{time}=\{FE\_ratio\}.
$$

实现列固定为 `bf_fe_ratio`。behavior 数据质量检查与 Decision materialization 必须保证：

$$
bf\_fe\_ratio = FE\_ratio
$$

逐行成立。使用 `bf_fe_ratio` 是为了保持活动模型输入来自 `BEHAVIOR_FEATURE_GROUPS`，不表示额外引入行为信息。

正式 feature-group 消融只比较 `T0/B1/B2/B3`（兼容代码入口分别为 `time_only`、`primary`/对应 B1 入口、`primary_with_dynamorep_lite`、`primary_with_maturity`）。`all_candidates` 严格等于 B3，只是兼容别名，不含诊断字段，也不单列为第五组。四组必须使用：

- 完全相同的 materialized Decision dataset；
- 完全相同的 BBOB train 与 held-out function-family validation；
- 完全相同的三个固定模型候选和随机 seed；
- 完全相同的 nested family-OOF 过程；
- B3 选择出的同名模型；
- 仅由 train family-OOF 分数拟合的 decision threshold；
- 完全相同的 Utility prediction、调用率、效用捕获和最终性能指标。

该 baseline 回答：

> Controller 是否只是学会在哪个优化阶段调用固定 landscape-analysis query？

解释规则：

- 若完整行为模型在 held-out families 和外部 benchmark 上没有稳定优于 `time_only`，不能声称搜索行为提供了超出阶段信息的预测价值；
- 若完整行为模型优于 `time_only`，只能说明所测行为变量提供了阶段之外的增量预测信息，仍需报告配对效应量与区间；
- 不得依据 validation 上 `time_only` 的结果改变 `phase1_dynamic_budget_event_v1` 采样参数、主 query 或调用预算。

首轮离线 Decision 样本只来自冻结的预算里程碑与状态事件，不由模型分数决定，每行 `sample_weight=1`。冻结模型后的 online 附加复查可使用阈值邻近带，其带宽为完整 BBOB-train family-OOF 上 `abs(score-threshold)` 的第 10 百分位数。BBOB-validation 与外部测试不拟合带宽，controller 与全部 baselines 必须共享同一附加机会集合。

------------------------------------------------------------------------

正式结果按 T0、B1、B2、B3 四个嵌套输入组报告，字段数分别为 1、19、25、31。T0 检验阶段信息；B1 到 B3 依次检验冻结行为集合提供的增量信息。`all_candidates` 与 `primary_with_maturity` 均映射到 B3，不重复报告；`diagnostic_only` 与算法特定参数不进入正式模型输入。若后续单独移除某个构念或引入算法身份信息，应作为预先定义的扩展实验，不得替代本冻结四组。

------------------------------------------------------------------------

# 13. 可解释性分析

推荐：

SHAP。

分析：

哪些行为指标影响：

$$ \hat U_{query} $$

例如：

发现：

-   高停滞
-   较低population Wasserstein变化率
-   较高covariance spectral concentration
-   稳定下降diversity

更可能值得执行固定 query。

------------------------------------------------------------------------

# 14. 训练流程

    Offline Dataset

            |

            v

    Feature Normalization

            |

            v

    Train Decision Model

            |

            v

    Nested Train-family OOF Model Selection

            |

            v

    Full-train OOF Threshold Freeze

            |

            v

    OOD Test

            |

            v

    End-to-end Evaluation

------------------------------------------------------------------------

# 15. 最终冻结方案

第一篇论文冻结：

模型候选：LDA、Logistic Regression、Ridge。

任务：分类 `U_query>0` 或回归连续 Query Utility，由 nested family-OOF decision utility 统一选择。

输入：

Algorithm-agnostic Behavior。

输出：

Decision score；Ridge 分数解释为预测 Utility，分类分数解释为正 Utility 排序分数。

决策：

完整 BBOB-train family-OOF 冻结的 `oof_utility` threshold。

目标：

证明：

算法无关搜索行为可以支持是否执行所评估固定 landscape-analysis query 的资源决策。
