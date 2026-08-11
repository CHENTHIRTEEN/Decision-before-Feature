# Decision-before-Feature Algorithm Portfolio与Selection Reference设计

> 协议修订（2026-08-11）：正式 Selection Reference 已从 problem 级静态分类与 nearest performance bucket 改为逐共享状态 action-loss regression。旧分类器只保留为被替代方法对照，不再生成正式 Utility 标签。详细裁决见 `Decision-before-Feature_逐状态动作损失Selection Reference修订.md`。

> 实现同步（2026-08-11）：逐状态 Selection Reference 的生成、拟合和接口检查已实现，但现存 BBOB trajectory 不具备完整 optimizer-state continuation 字段，因此正式 action losses、Selection Reference 和下游标签尚未重生成。它只用于离线标签与外部运行中的固定算法选择路径，不进入 Decision 输入，也不作为本文算法贡献。当前结果边界见 `../30_results/phase1_current_results.md`。

## 1. 文档定位

本文档定义 Decision-before-Feature 框架中的 Algorithm Portfolio 与
Selection Reference。

核心目标：

构建稳定、公平、可复现的 Offline Utility Label。

Selection Reference 的作用：

不是提出新的算法选择方法，而是提供一个可靠的离线参考：

> 如果执行所评估的固定 landscape-analysis query，在当前逐状态动作选择协议下可以获得多少性能差。

因此，Selection Reference 的质量直接影响 Decision Model 的训练标签质量。

------------------------------------------------------------------------

# 2. 整体流程

传统流程：

    Problem

    ↓

    Query Feature Extraction

    ↓

    Algorithm Selection Model

    ↓

    Optimizer

Decision-before-Feature：

    Problem

    ↓

    Decision Module

    ↓

          ----------------

          |              |

       No-query       Run Query

          |              |

    Default       Query + Selection Reference

    Optimizer          |

                       v

                 Selected Optimizer

其中：

完整 state-action loss 表只用于离线生成标签。部署式评价中，如果 gate 调用 query，则仍需执行已冻结的 Selector model；外部 benchmark 不参与该模型拟合。

------------------------------------------------------------------------

# 3. Algorithm Portfolio设计原则

## 3.1 为什么需要Portfolio

固定 query 的效用可能来自：

> Feature是否能够帮助选择更合适的优化算法。

因此：

如果Portfolio过小：

例如：

只有DE。

那么：

query feature 几乎没有可影响的选择空间。

Utility被低估。

如果Portfolio过大：

包含大量相似算法。

会导致：

-   Selection难度增加
-   Selection Reference不稳定
-   计算成本增加

因此需要：

性能覆盖 + 稳定性之间平衡。

------------------------------------------------------------------------

# 4. 推荐Algorithm Portfolio

第一阶段建议：

连续单目标黑盒优化。

包含：

## Differential Evolution (DE)

特点：

-   全局探索能力强
-   参数稳定
-   常用于black-box optimization

------------------------------------------------------------------------

## CMA-ES

特点：

-   强局部搜索能力
-   适合连续变量
-   对旋转问题鲁棒

------------------------------------------------------------------------

## PSO

特点：

-   群体智能代表算法
-   行为分析方便

------------------------------------------------------------------------

## SHADE / L-SHADE

特点：

-   参数自适应
-   CEC竞赛常用强baseline

------------------------------------------------------------------------

最终Portfolio：

    A =
    {
    DE,
    CMA-ES,
    PSO,
    L-SHADE
    }

------------------------------------------------------------------------

# 5. Algorithm Selection Reference定义

给定当前共享 checkpoint state：

$$s_t=(X_t,y_t,H_t,B_t),$$

唯一动作集合为：

$$
\mathcal A(s_t)=\{\text{continue-current}\}\cup(A\setminus\{a_t\}),
$$

其中 `continue-current` 对 prefix algorithm 使用完整状态原生 continuation，其余动作使用一次 Population Transfer。对每个动作真实运行：

$$L(s_t,a),\qquad a\in\mathcal A(s_t).$$

Selector 预测连续动作损失：

$$
\widehat{\boldsymbol L}(s_t)
=S\!\left(\phi(p),\operatorname{behavior}(s_t),B_t/FE_{total}\right),
$$

并选择：

$$
\hat a_t=\arg\min_a\widehat L(s_t,a).
$$

逐状态真实最小值称为 `best observed action`：

$$
a_t^{best\ observed}=\arg\min_a L(s_t,a).
$$

它只作离线诊断参照，不是现实可部署方法。

------------------------------------------------------------------------

# 6. Selection Model设计

## 6.0 文献依据与本文定位

ELA-based algorithm selection 已有明确文献依据。

Bischl et al. (2012) 将 Exploratory Landscape Analysis 与成本敏感学习结合，用 ELA 特征训练按实例选择算法的模型。Kerschke and Trautmann (2019) 进一步在连续黑盒优化中系统使用 ELA 特征与机器学习构建 algorithm selection model，并与 Single Best Solver 和 Virtual Best Solver 比较。Kerschke et al. (2019) 对 Automated Algorithm Selection 进行了综述，明确了 per-instance algorithm selection 的标准形式：

    problem instance features
            |
            v
    supervised selection model
            |
            v
    selected algorithm from a portfolio

因此，本项目可以采用：

    query features -> supervised selector -> selected_algorithm

作为下游 Algorithm Selection Reference。

但是：

Selection Reference 不是本文提出的新算法选择方法。

本文创新点是：

    是否应该在当前搜索状态下执行所评估的固定 query

而不是：

    执行固定 query 后如何重新发明 algorithm selector

写作时应使用如下定位：

> We use a query-specific supervised selector as a fixed downstream selection reference, following established feature-based per-instance algorithm selection studies. The contribution is the preceding analysis-selection problem: deciding whether the evaluated fixed landscape-analysis query should be executed before invoking such a selector.

引用建议：

- Bischl et al. (2012), *Algorithm Selection Based on Exploratory Landscape Analysis and Cost-Sensitive Learning*；
- Kerschke and Trautmann (2019), *Automated Algorithm Selection on Continuous Black-Box Problems by Combining Exploratory Landscape Analysis and Machine Learning*；
- Kerschke et al. (2019), *Automated Algorithm Selection: Survey and Perspectives*。

### 6.0.1 ELA-based selector 方法谱系

已有 ELA-based algorithm selection 文献不只包含单一的
`features -> classifier -> algorithm label` 形式。为了避免把当前
`selection_reference` 误写成本文创新点，本文档将相关方法分为以下几类。

| 类别 | 代表工作 | 训练目标 | 输出 | 与当前项目关系 |
|---|---|---|---|---|
| cost-sensitive classification | Bischl et al. (2012) | 以候选算法的 expected runtime 定义 example-specific label cost，用 one-sided support vector regression 学习低成本选择 | portfolio algorithm | 这是 ELA-based selection reference 的早期直接出处，强调 wrong selection 的成本不应等同处理。 |
| direct classification | Kerschke and Trautmann (2019) | 直接预测 best-performing optimizer；文中比较 classification、regression、paired regression 三类策略，并使用 rpart、SVM、Random Forest、XGBoost、MARS 等模型 | portfolio algorithm label | 原静态 `RandomForestClassifier` 属于这一类；现仅作被替代方法对照。 |
| performance regression | Kerschke and Trautmann (2019); Jankovic and Doerr (2020); Jankovic et al. (2021) | 对每个候选算法预测 performance，例如 fixed-target 或 fixed-budget performance，再选择预测性能最好的算法 | predicted best algorithm, plus predicted performance | 当前正式 `selection_reference` 属于这一类，但监督单位改为共享 state，目标为真实 continuation action loss。 |
| pairwise regression | Kerschke and Trautmann (2019) | 对每对算法预测 performance difference，再聚合 pairwise 胜负关系 | portfolio algorithm | 可减少直接 multiclass label 对小 performance gap 的敏感性；适合作为后续诊断而非本文主创新。 |
| algorithm configuration | Belkhir et al. (2017); Prager et al. (2020) | 用 query features 选择同一算法框架下的参数、模块或配置，例如 modular CMA-ES 的 classifier chains | algorithm configuration | 说明 ELA 不仅能做 algorithm selection，也能做 per-instance configuration；当前项目仍限定在 portfolio algorithm selection。 |
| Deep-ELA / learned landscape representation | van Stein et al. (2023); Seiler et al. (2025) | 用 VAE、transformer 等学习 landscape representation，替代或补充手工 query features | features for AS/AAC or downstream meta-learning | 属于近期表示学习扩展。当前项目不采用 Deep-ELA，以保持固定下游 selector 简洁和可解释。 |
| MO-ELA | Preuss et al. (2026) | 为连续多目标优化构造 non-dominated sorting、descriptive statistics、PCA、graph、gradient information 等 feature groups | features for multi-objective AAS | 当前项目是单目标黑盒优化；该方向只用于 related work 边界说明。 |
| benchmarking risks | Tanabe (2022); Kerschke and Trautmann (2019); Jankovic et al. (2021) | 分析 algorithm portfolio、dimension、cross-validation、pre-solver、performance measure、hyperparameter tuning 对 AS 系统评价的影响 | evaluation methodology | 支持本文把 `selection_reference` 质量作为 data condition 和 diagnostic risk，而不是无条件假设 selector 稳定泛化。 |

因此，当前项目的正式定位是：

    query features + algorithm-agnostic state behavior + continuous remaining budget
    -> multi-output action-loss regression
    -> selected_algorithm

当前模型为 `RandomForestRegressor`。分类器版本只用于说明被修订构念失配的来源，不再生成正式标签。

这一路线有明确文献依据，但不是本文贡献。本文贡献仍然是：

    search behavior -> decide whether to execute the fixed query

而不是：

    propose a new query-specific algorithm selector

论文写作中应避免以下表述：

- “本文提出一种新的 query selector”；
- “RandomForestClassifier selector 是本文主要方法贡献”；
- “selection_reference 的 `selected=VBS` 提升等价于 `U_query` 提升”。

更合适的表述是：

> The selection reference follows the established performance-regression
> paradigm for feature-based algorithm selection, with supervision defined by
> candidate continuations from each shared search state. It is used as a fixed downstream
> component for constructing offline query-utility labels. The proposed
> Decision-before-Feature model addresses a preceding decision problem:
> whether the evaluated query and its downstream selector should be invoked
> at the current search state.

补充引用建议：

- Jankovic and Doerr (2020), *Landscape-Aware Fixed-Budget Performance Regression and Algorithm Selection for Modular CMA-ES Variants*；
- Jankovic et al. (2021), *The Impact of Hyper-Parameter Tuning for Landscape-Aware Performance Regression and Algorithm Selection*；
- Kostovska et al. (2023), *Comparing Algorithm Selection Approaches on Black-Box Optimization Problems*；
- Tanabe (2022), *Benchmarking Feature-based Algorithm Selection Systems for Black-box Numerical Optimization*；
- Belkhir et al. (2017), *Per Instance Algorithm Configuration of CMA-ES with Limited Budget*；
- Prager et al. (2020), *Per-Instance Configuration of the Modularized CMA-ES by Means of Classifier Chains and Exploratory Landscape Analysis*；
- van Stein et al. (2023), *DoE2Vec: Deep-learning Based Features for Exploratory Landscape Analysis*；
- Seiler et al. (2025), *Deep-ELA: Deep Exploratory Landscape Analysis with Self-Supervised Pretrained Transformers for Single- and Multi-Objective Continuous Optimization Problems*；
- Preuss et al. (2026), *MO-ELA: Rigorously Expanding Exploratory Landscape Features for Automated Algorithm Selection in Continuous Multi-Objective Optimisation*。

可核对的引用入口：

- Bischl et al. (2012), GECCO 2012, DOI `10.1145/2330163.2330209`；
- Kerschke and Trautmann (2019), *Evolutionary Computation*, DOI `10.1162/evco_a_00236`；
- Kerschke et al. (2019), *Evolutionary Computation*, DOI `10.1162/evco_a_00242`；
- Jankovic and Doerr (2020), GECCO 2020, DOI `10.1145/3377930.3390183`；
- Jankovic et al. (2021), GECCO 2021, DOI `10.1145/3449639.3459406`；
- Kostovska et al. (2023), GECCO Companion 2023, DOI `10.1145/3583133.3590697`；
- Tanabe (2022), *IEEE Transactions on Evolutionary Computation*, DOI `10.1109/TEVC.2022.3169770`；
- Belkhir et al. (2017), GECCO 2017, DOI `10.1145/3071178.3071343`；
- Prager et al. (2020), IEEE SSCI 2020, DOI `10.1109/SSCI47803.2020.9308510`；
- van Stein et al. (2023), GECCO 2023, DOI `10.1145/3583133.3590609`；
- Seiler et al. (2025), *Evolutionary Computation*, DOI `10.1162/evco_a_00367`；
- Preuss et al. (2026), arXiv `2602.00098`。

------------------------------------------------------------------------

## 6.1 基础方案

使用逐状态 multi-output performance regression。

输入：

```text
query features
+ permutation-invariant algorithm-agnostic behavior
+ continuous remaining_budget_ratio
```

输出：

每个候选算法动作的 predicted normalized action loss。

当前正式 `selection_reference` 的模型口径：

- `RandomForestRegressor`
- 200 trees
- `min_samples_leaf=2`
- 一个输出头对应一个 portfolio algorithm
- BBOB train 行使用 function-family grouped cross-fitting predictions
- held-out rows 使用全体 BBOB train families 拟合的最终模型

可作为后续 sensitivity 或 robustness baseline 的模型：

-   linear / ridge performance regression
-   XGBoost
-   LightGBM
-   per-action Random Forest regression
-   pairwise regression selector

原因：

-   保留选错动作的严重程度
-   连续处理 remaining budget
-   避免 nearest bucket 造成的离散跳变
-   与 feature-based algorithm selection 研究传统一致
-   便于与 SBS、VBS、best observed action 和 state-only ablation 比较

------------------------------------------------------------------------

## 6.2 训练数据生成

对于每个 eligible shared state，运行 `continue_current` 和其余三个算法动作，形成 `state × action` loss matrix。每个动作使用相同 query-adjusted remaining FE budget；同算法原生续跑，跨算法 Population Transfer。

为消除不同 BBOB problem 的 objective offset 与尺度对回归损失的支配，训练目标为逐状态归一化 action loss：

$$
\widetilde L(s_t,a)=
\frac{L(s_t,a)-\min_bL(s_t,b)}
{\max_bL(s_t,b)-\min_bL(s_t,b)+\epsilon}.
$$

该变换保持同一 state 内的 argmin。raw action loss 同时保留，用于最终性能、潜在性能差和 selector regret 计算。

------------------------------------------------------------------------

# 7. Selection Reference公平性要求

## 7.1 禁止使用测试信息

Selection Reference训练：

只能使用：

training problems。

测试问题：

只能用于最终评价。

------------------------------------------------------------------------

## 7.2 固定Algorithm Budget

同一共享状态的所有候选动作：

必须：

-   相同FE预算
-   相同 state 与 seed 口径
-   相同停止条件

否则：

Selection Reference偏向某个算法。

------------------------------------------------------------------------

## 7.3 Random Seed控制

每个：

problem × algorithm

至少：

30 independent runs。

保存：

-   mean
-   median
-   std
-   best

------------------------------------------------------------------------

# 8. Selection Reference性能评价

Selection Reference本身需要验证。

比较：

## Single Best Solver (SBS)

整个训练集平均最好的算法。

------------------------------------------------------------------------

## Virtual Best Solver (VBS)

理论上每个静态问题选择最优算法。

这是上限。

共享 checkpoint state 上从已运行动作中取最小 loss 时，必须另称为 `best observed action`。它用于逐状态 selector regret，不与 VBS 混称。

------------------------------------------------------------------------

## Query-specific Selector

实际Selection Reference。

目标：

在 held-out function families 上减小相对于 best observed action 的 selector regret，并改善实际 Query path loss。`selected_matches_best_observed` 只是辅助指标，不能替代 regret 与 end-to-end Utility。

------------------------------------------------------------------------

指标：

-   ERT
-   final error
-   regret

------------------------------------------------------------------------

# 9. 避免Selection Reference过强或过弱

## 过弱情况

如果：

Query-specific Selector≈SBS

说明：

该固定 query 没有带来有效选择差异。

------------------------------------------------------------------------

## 过强情况

如果：

Query-specific Selector 接近 VBS

但是：

训练测试泄漏。

需要检查：

function family split。

------------------------------------------------------------------------

## 9.1 当前正式实验中的解释边界

Selection Reference 的质量直接影响 $p_{query}$ 和 $U_{query}$，但它仍是固定下游实验组件，不是本文的算法贡献。

第一篇论文主 probe/default 固定为训练集 SBS：No-query 原生继续当前 SBS，Query 后 statewise selector 从唯一动作集合中选择，选择 `continue_current` 时继续同一完整状态，选择其他算法时执行一次 population transfer。其他 prefix algorithm 只用于 cross-probe robustness、leave-one-probe-out 与 algorithm-agnostic 泛化，不得混入主结果。

正式输出必须保存 `selected_equals_default`、`selected_equals_prefix`、`skip_switches_from_prefix`、`no_query_algorithm`、`handoff_type`、`best_observed_algorithm`、`best_observed_loss` 与 `selector_regret_raw`。`no_query_algorithm=default_algorithm`；`handoff_type=query_transition_mode`，描述 Query-selected action 使用原生 continuation 还是 Population Transfer。`same_algorithm` / `changed_algorithm` 只作为 selected-vs-default 报告分层；多 prefix 数据中的实际行动变化以 `selected_equals_prefix` 为准。正式报告同时给出 SBS、静态 VBS、逐状态 best observed action 与现实 selector 的性能差；选择一致率不能替代 observed utility。

preliminary selector 的覆盖和 bucket 不连续问题已归入 `docs/archive/min_support/`，不得作为正式 phase1 数值来源。

------------------------------------------------------------------------

# 10. 与Query Utility的关系

最终：

No-query：

$$ P_{skip} $$

Run Query：

$$ p_{query} $$

其中 $p_{query}$ 来自固定 query + statewise Selection Reference 的真实 selected-action continuation。

Utility：

$$ U_{query} = (P_{skip}-p_{query}) - \lambda_T C_T-\lambda_M C_M. $$

Query sampling FE 已通过减少 Query continuation budget 计入 $p_{query}$；Population Transfer 的影响也已进入 action loss，二者均不得重复扣除。

诊断分解：

$$
P_{skip}-p_{query}
=(P_{skip}-P_{best\ observed})-(p_{query}-P_{best\ observed}).
$$

------------------------------------------------------------------------

# 11. Selection Reference实验

## Experiment A

验证Portfolio覆盖能力。

分析：

不同算法在哪些function family占优。

------------------------------------------------------------------------

## Experiment B

验证 statewise query-specific Selector。

比较：

-   SBS
-   Random Selection
-   Query-specific Selection
-   VBS
-   best observed action（逐状态诊断）
-   state-only selector（不含 query features）
-   query-only selector（不含 behavior）

------------------------------------------------------------------------

## Experiment C

验证Utility稳定性。

不同：

-   lambda
-   seed
-   portfolio

------------------------------------------------------------------------

# 12. 后续扩展

## Multi-fidelity Portfolio

不同预算选择不同算法。

------------------------------------------------------------------------

## Dynamic Algorithm Portfolio

根据Search Behavior动态调整候选算法。

------------------------------------------------------------------------

## Joint Decision and Selection

未来：

联合学习：

是否执行固定 query

以及

选择哪个算法。

------------------------------------------------------------------------

# 13. 实现建议

当前代码模块：

    selection_reference/

    ├── action_losses.py

    ├── model.py

    └── build.py

    utility_labels/

    ├── generation.py

    └── batch_generation.py

------------------------------------------------------------------------

# 14. 核心原则总结

1.  Selection Reference不是论文创新点，而是可靠实验基础。

2.  Portfolio必须覆盖不同搜索行为。

3.  Selection Reference训练必须严格避免test leakage。

4.  Selection Reference 的监督单位必须与在线共享状态动作匹配；不得用静态 problem label 或 nearest bucket 代替。

5.  Query Utility的可信度依赖Selection Reference在 held-out function families 上的 selector regret 与 end-to-end performance。

6.  Decision-before-Feature的创新重点仍然是：

Analysis Selection，而不是Algorithm Selection。
