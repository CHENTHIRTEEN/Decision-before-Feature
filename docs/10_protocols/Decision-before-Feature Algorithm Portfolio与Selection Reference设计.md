# Decision-before-Feature Algorithm Portfolio与Selection Reference设计

## 1. 文档定位

本文档定义 Decision-before-Feature 框架中的 Algorithm Portfolio 与
Selection Reference。

核心目标：

构建稳定、公平、可复现的 Offline Utility Label。

Selection Reference 的作用：

不是提出新的算法选择方法，而是提供一个可靠的离线参考：

> 如果执行Landscape Analysis，理论上可以获得多少收益。

因此，Selection Reference 的质量直接影响 Decision Model 的训练标签质量。

------------------------------------------------------------------------

# 2. 整体流程

传统流程：

    Problem

    ↓

    ELA Feature Extraction

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

       Skip ELA       Run ELA

          |              |

    Default       ELA + Selection Reference

    Optimizer          |

                       v

                 Selected Optimizer

其中：

Selection Reference只用于离线生成标签。

部署阶段：

不需要执行Selection Reference。

------------------------------------------------------------------------

# 3. Algorithm Portfolio设计原则

## 3.1 为什么需要Portfolio

ELA的价值来自：

> Feature是否能够帮助选择更合适的优化算法。

因此：

如果Portfolio过小：

例如：

只有DE。

那么：

ELA几乎没有选择空间。

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

给定：

问题：

$$ p $$

ELA Feature：

$$ \phi(p) $$

Portfolio：

$$ A $$

Selection模型：

$$ S $$

选择：

$$ a^*=S(\phi(p)) $$

得到：

$$ P_{ELA} $$

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

    ELA features -> supervised selector -> selected_algorithm

作为下游 Algorithm Selection Reference。

但是：

Selection Reference 不是本文提出的新算法选择方法。

本文创新点是：

    是否应该在当前搜索状态下执行 ELA

而不是：

    执行 ELA 后如何重新发明 algorithm selector

写作时应使用如下定位：

> We use an ELA-based supervised selector as a fixed downstream selection reference, following established feature-based per-instance algorithm selection studies. The contribution of this work is the preceding analysis-selection problem: deciding whether landscape analysis should be executed before invoking such a selector.

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
| direct classification | Kerschke and Trautmann (2019) | 直接预测 best-performing optimizer；文中比较 classification、regression、paired regression 三类策略，并使用 rpart、SVM、Random Forest、XGBoost、MARS 等模型 | portfolio algorithm label | 当前 `selection_reference` 对应这一类：`ELA features -> RandomForestClassifier -> selected_algorithm`。 |
| performance regression | Kerschke and Trautmann (2019); Jankovic and Doerr (2020); Jankovic et al. (2021) | 对每个候选算法预测 performance，例如 fixed-target 或 fixed-budget performance，再选择预测性能最好的算法 | predicted best algorithm, plus predicted performance | 更贴近 $P_{ELA}$ 与 $U_{ELA}$ 的数值口径，可作为后续 sensitivity baseline，但不是当前正式组件。 |
| pairwise regression | Kerschke and Trautmann (2019) | 对每对算法预测 performance difference，再聚合 pairwise 胜负关系 | portfolio algorithm | 可减少直接 multiclass label 对小 performance gap 的敏感性；适合作为后续诊断而非本文主创新。 |
| algorithm configuration | Belkhir et al. (2017); Prager et al. (2020) | 用 ELA features 选择同一算法框架下的参数、模块或配置，例如 modular CMA-ES 的 classifier chains | algorithm configuration | 说明 ELA 不仅能做 algorithm selection，也能做 per-instance configuration；当前项目仍限定在 portfolio algorithm selection。 |
| Deep-ELA / learned landscape representation | van Stein et al. (2023); Seiler et al. (2025) | 用 VAE、transformer 等学习 landscape representation，替代或补充手工 ELA features | features for AS/AAC or downstream meta-learning | 属于近期表示学习扩展。当前项目不采用 Deep-ELA，以保持固定下游 selector 简洁和可解释。 |
| MO-ELA | Preuss et al. (2026) | 为连续多目标优化构造 non-dominated sorting、descriptive statistics、PCA、graph、gradient information 等 feature groups | features for multi-objective AAS | 当前项目是单目标黑盒优化；该方向只用于 related work 边界说明。 |
| benchmarking risks | Tanabe (2022); Kerschke and Trautmann (2019); Jankovic et al. (2021) | 分析 algorithm portfolio、dimension、cross-validation、pre-solver、performance measure、hyperparameter tuning 对 AS 系统评价的影响 | evaluation methodology | 支持本文把 `selection_reference` 质量作为 data condition 和 diagnostic risk，而不是无条件假设 selector 稳定泛化。 |

因此，当前项目的正式定位是：

    ELA features -> direct supervised classifier -> selected_algorithm

其中 classifier 当前为诊断和 label 生成中使用的 `RandomForestClassifier`。

这一路线有明确文献依据，但不是本文贡献。本文贡献仍然是：

    search behavior -> decide whether to execute ELA

而不是：

    propose a new ELA-based algorithm selector

论文写作中应避免以下表述：

- “本文提出一种新的 ELA selector”；
- “RandomForestClassifier selector 是本文主要方法贡献”；
- “selection_reference 的 `selected=VBS` 提升等价于 `U_ELA` 提升”。

更合适的表述是：

> The selection reference follows the established direct supervised
> ELA-based algorithm selection paradigm. It is used as a fixed downstream
> component for constructing offline ELA utility labels. The proposed
> Decision-before-Feature model addresses a preceding decision problem:
> whether the ELA computation and its downstream selector should be invoked
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

使用经典 direct supervised Algorithm Selection 模型。

输入：

ELA features。

输出：

algorithm label。

当前正式 `selection_reference` 的模型口径：

- `RandomForestClassifier`

可作为后续 sensitivity 或 robustness baseline 的模型：

-   Random Forest
-   XGBoost
-   LightGBM
-   Logistic Regression
-   kNN / nearest-neighbor
-   performance regression selector
-   pairwise regression selector

原因：

-   小样本稳定
-   可解释
-   与ELA研究传统一致
-   便于与 SBS、VBS、stage-wise majority 等 baseline 比较

------------------------------------------------------------------------

## 6.2 训练数据生成

对于每个problem：

运行所有算法：

    DE
    CMA-ES
    PSO
    L-SHADE

            |

            v

    Performance Matrix

形成：

Algorithm Portfolio Performance Matrix。

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

所有算法：

必须：

-   相同FE预算
-   相同运行次数
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

理论上每个问题选择最优算法。

这是上限。

------------------------------------------------------------------------

## ELA Selector

实际Selection Reference。

目标：

接近VBS。

------------------------------------------------------------------------

指标：

-   ERT
-   final error
-   regret

------------------------------------------------------------------------

# 9. 避免Selection Reference过强或过弱

## 过弱情况

如果：

ELA Selector≈SBS

说明：

ELA没有产生有效选择。

------------------------------------------------------------------------

## 过强情况

如果：

ELA Selector接近VBS

但是：

训练测试泄漏。

需要检查：

function family split。

------------------------------------------------------------------------

## 9.1 当前 min_support 诊断对应的风险

当前 min_support 结果显示，Selection Reference 的质量本身是影响 $P_{ELA}$ 与 $U_{ELA}$ 的重要数据条件。

已有诊断：

- `results/decision/min_support/selection_reference_selector_ablation/selector_ablation_diagnostic.md`
- `results/decision/min_support/selection_reference_selector_ablation/selector_ablation_summary.csv`
- `results/decision/min_support/selection_reference_selector_ablation/selector_ablation_target_family_summary.csv`

主要现象：

1. 当前 RandomForestClassifier selector 在训练 split 上 `selected_algorithm = vbs_algorithm` 的比例达到 100%，但在 validation split 上下降到约 42%。

2. 限制深度的 RandomForestClassifier 没有改善 validation 上的 `selected_algorithm = vbs_algorithm` 比例，说明问题不能简单归因于树深度过大。

3. validation 上 `bbob_f005`、`bbob_f019` 和 `bbob_f024` 的错配最集中：

   - `bbob_f005` 中 VBS 经常为 `pso` 或 `shade`，当前 selector 经常选择 `cmaes`；
   - `bbob_f019` 中 10D 的 VBS 主要为 `cmaes`，当前 selector 经常选择 `shade`；
   - `bbob_f024` 的错配具有 dimension 和 budget stage 依赖，20D 中期存在部分可用选择信号，但 10D 与阶段边界不稳定。

4. SBS 或 stage-wise majority selector 在部分 validation family 上提高了 `selected_algorithm = vbs_algorithm` 比例，但 trajectory-bucket proxy 的 $P_{ELA}$ 或 $U_{ELA}$ 并不总是同步改善。

这说明：

`selected_algorithm = vbs_algorithm` 是必要诊断指标，但不能单独替代真实的 paired continuation utility。

当前可作出的谨慎判断是：

> min_support 中 ELA selector 的泛化失败很可能同时来自训练 family 覆盖不足、performance bucket 稀疏、VBS 标签噪声和 ELA feature 表达不足。RandomForestClassifier 的过拟合是观察到的表现形式，数据条件不足是重要来源，但不能在没有扩展实验前断言为唯一原因。

论文写作中应将该结果表述为：

- 固定下游 selection pipeline 的限制；
- utility label 质量的影响因素；
- 后续正式 phase1 数据集需要验证的稳定性问题。

不得将当前 min_support 的 ELA selector 写成已经稳定泛化的 algorithm selection 方法。

------------------------------------------------------------------------

# 10. 与ELA Utility的关系

最终：

No ELA：

$$ P_{skip} $$

ELA：

$$ P_{ELA} $$

其中：

$$ P_{ELA} $$

来自：

ELA + Selection Reference。

Utility：

$$ U_{ELA} = (P_{skip}-P_{ELA}) - \lambda C_{ELA} $$

------------------------------------------------------------------------

# 11. Selection Reference实验

## Experiment A

验证Portfolio覆盖能力。

分析：

不同算法在哪些function family占优。

------------------------------------------------------------------------

## Experiment B

验证ELA Selector。

比较：

-   SBS
-   Random Selection
-   ELA Selection
-   VBS

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

是否ELA

以及

选择哪个算法。

------------------------------------------------------------------------

# 13. 实现建议

代码模块：

    selection_oracle/

    ├── portfolio.py

    ├── performance_matrix.py

    ├── selector.py

    ├── train_selector.py

    ├── evaluate_selector.py

    └── oracle_generator.py

------------------------------------------------------------------------

# 14. 核心原则总结

1.  Selection Reference不是论文创新点，而是可靠实验基础。

2.  Portfolio必须覆盖不同搜索行为。

3.  Selection Reference训练必须严格避免test leakage。

4.  ELA Utility的可信度依赖Selection Reference稳定性。

5.  Decision-before-Feature的创新重点仍然是：

Analysis Selection，而不是Algorithm Selection。
