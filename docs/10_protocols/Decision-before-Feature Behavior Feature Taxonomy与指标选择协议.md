# Decision-before-Feature Behavior Feature Taxonomy与指标选择协议

> 实现同步（2026-08-12）：当前 extractor 生成 25 个 permutation-invariant、算法无关 `bf_*` 字段；正式模型比较的主组 `primary_with_maturity` 使用其中 23 个，`bf_population_overlap_w05` 与 `bf_best_distance_fitness_corr` 保留为诊断字段，不进入主组。w02/w05/w10 已改为逐次完整原生 update 统计，不再从稀疏正式 checkpoint 中选择 anchor。旧 behavior 曾把 population 行号解释为跨代个体身份，已撤回并不得复用。

## 1. 文档定位

本文档定义 Decision-before-Feature 框架中的 Behavior Feature 输入体系。

核心目标：

构建一种：

> Algorithm-agnostic, low-cost, decision-oriented search behavior
> representation

用于预测：

$$ U_{query} $$

即：

所评估固定 query 的 $U_{query}$ 是否大于 0。

------------------------------------------------------------------------

# 2. Feature设计原则

## Principle 1: Algorithm-agnostic

Behavior Feature必须来自：

优化过程。

不依赖：

-   PSO参数
-   DE参数
-   CMA-ES内部矩阵

原因：

目标是学习：

$$ Search Behavior \rightarrow Analysis Utility $$

而不是：

$$ Algorithm Parameter \rightarrow Decision $$

------------------------------------------------------------------------

## Principle 2: Low-cost

Behavior feature 计算成本必须相对所评估固定 query 足够小。

允许：

-   population statistics
-   fitness history
-   trajectory

禁止：

-   additional objective evaluations
-   landscape probing

------------------------------------------------------------------------

## Principle 3: Interpretable

每个Feature应该具有明确优化含义。

避免：

直接使用不可解释embedding作为唯一输入。

------------------------------------------------------------------------

# 3. Feature Taxonomy总体结构

Behavior Feature分为五类：

    Search Behavior State

            |

    --------------------------------

    |          |          |          |

    Progress  Diversity  Exploration  Exploitation



                 Communication

------------------------------------------------------------------------

# 4. Category A: Optimization Progress Features

## 4.1 Function Evaluation Ratio

定义：

$$ r_t=\frac{FE_t}{FE_{max}} $$

作用：

描述当前搜索阶段。

注意：

不是直接学习维度或者时间。

------------------------------------------------------------------------

## 4.2 Improvement Rate

定义：

令：

$$ r_t=\frac{FE_t}{FE_{max}} $$

在预算比例窗口：

$$ w=0.02 $$

内取满足：

$$ r_a \le r_t-w $$

的最近历史检查点作为anchor，则当前实现使用：

$$ IR_t^{(0.02)}=
\frac{f_{best}(a)-f_{best}(t)}
{\max(|f_{best}(a)|,\epsilon)}
\cdot
\frac{1}{r_t-r_a}
$$

含义：

单位FE预算比例下的相对best-fitness改善。

高：

仍有优化潜力。

低：

可能进入停滞。

------------------------------------------------------------------------

## 4.3 Improvement Frequency

过去窗口：

改善次数比例。

当前实现使用同一：

$$ w=0.02 $$

预算比例窗口，统计窗口内相邻检查点发生严格best-fitness改善的比例。

用于区分：

偶然改善

和

持续改善。

------------------------------------------------------------------------

# 5. Category B: Population Diversity Features

## 5.1 Mean Pairwise Distance

定义：

$$ D_t= \frac{2}{N(N-1)}
\sum_{i<j}\frac{\|x_i-x_j\|_2}{\sqrt{d}} $$

含义：

种群空间分散程度。

------------------------------------------------------------------------

## 5.2 Diversity Change Rate

定义：

在预算比例窗口：

$$ w=0.05 $$

内取anchor：

$$ \Delta D_t^{(0.05)}=
\frac{D_t-D_a}{\max(D_a,\epsilon)}
$$

解释：

下降：

搜索集中。

增加：

探索增强。

------------------------------------------------------------------------

## 5.3 Population Spread

统计：

-   variance
-   coordinate range
-   centroid shift
-   covariance spectral concentration

------------------------------------------------------------------------

# 6. Category C: Set-based Distribution Features

目标：

描述population与fitness经验分布的当前形状和跨窗口变化。

------------------------------------------------------------------------

## 6.1 Population Wasserstein Change

当前与anchor population均视为等权经验分布，不按行建立个体对应关系。当前实现使用 $w=0.05$ 的FE-ratio窗口，并通过最小费用一一匹配计算经验Wasserstein-1：

$$
W_t^{(0.05)}=
\frac{1}{r_t-r_a}
\min_{\pi\in S_N}
\frac{1}{N}\sum_i
\frac{\lVert x_i^{(a)}-x_{\pi(i)}^{(t)}\rVert_2}{\sqrt d}.
$$

该数值表示种群空间分布每单位预算比例的变化幅度，不表示真实个体运动。

------------------------------------------------------------------------

## 6.2 Centroid Shift与Coherence

centroid shift rate定义为：

$$
G_t^{(0.05)}=
\frac{\lVert \mu_t-\mu_a\rVert_2}{\sqrt d\,(r_t-r_a)}.
$$

coherence定义为未除预算跨度的centroid shift与Wasserstein distance之比。无集合变化时取0，其余裁剪到 $[0,1]$。该比值描述总体平移占集合分布变化的比例，不解释为个体方向一致性。

------------------------------------------------------------------------

## 6.3 Covariance与Fitness Distribution

当前population的协方差谱集中度使用归一化Herfindahl concentration。令 $q=\min(d,N-1)$，协方差非负特征值为 $\lambda_j$：

$$
C_t=\frac{q\frac{\sum_j\lambda_j^2}{(\sum_j\lambda_j)^2}-1}{q-1}.
$$

退化协方差取0。fitness跨窗口变化先分别排序为经验分位数，令 $\delta_i=f_{(i)}^{(a)}-f_{(i)}^{(t)}$，再计算：

- 改善分位数比例；
- $\operatorname{mean}(\delta_i)$ 经anchor平均绝对fitness和FE-ratio跨度归一化后的分布改善率；
- $\operatorname{mean}(|\delta_i|)$ 经相同尺度归一化的一维Wasserstein变化率。

这些字段只表示经验分布变化，不表示同一个体的success、improvement variance或best improvement share。

------------------------------------------------------------------------

# 7. Category D: Exploitation Features

描述：

开发已有区域的能力。

------------------------------------------------------------------------

## 7.1 Distance Decay

定义：

个体距离当前population-best位置的平均距离变化。

$$ d_t=\frac{1}{N}\sum_i\frac{\|x_i-x_{best,t}\|_2}{\sqrt d} $$

当前实现使用：

$$ w=0.10 $$

预算比例窗口：

$$ DD_t^{(0.10)}=
\frac{d_a-d_t}{\max(d_a,\epsilon)}
$$

观察：

距离是否快速下降。

注意：

这里不使用真实全局最优位置，也不使用benchmark function identity。

------------------------------------------------------------------------

## 7.2 Stagnation

定义：

距离最近一次best-fitness严格改善已经消耗的预算比例。

当前实现使用：

$$ w=0.10 $$

并归一化为：

$$ S_t^{(0.10)}=
\frac{\min(\max(r_t-r_{last\ improvement},0),0.10)}{0.10}
$$

含义：

判断：

开发是否饱和。

------------------------------------------------------------------------

## 7.3 Convergence Rate

描述：

种群收缩速度。

当前实现使用：

$$ w=0.10 $$

预算比例窗口内的相对diversity下降率：

$$ CR_t^{(0.10)}=
\frac{D_a-D_t}{\max(D_a,\epsilon)}
\cdot
\frac{1}{r_t-r_a}
$$

------------------------------------------------------------------------

# 8. Category E: Communication Features

主要用于群智能算法。

目标：

描述：

个体之间信息传播。

------------------------------------------------------------------------

可能指标：

-   neighborhood similarity
-   information sharing intensity
-   leader influence

注意：

由于DE/CMA-ES不存在显式communication，

第一版本可以暂不加入主模型。

作为扩展实验。

------------------------------------------------------------------------

# 9. 推荐第一版Feature集合

为了避免feature explosion。

第一篇论文建议：

## Core Features

### Progress

-   FE ratio
-   improvement rate
-   improvement frequency

### Diversity

-   diversity
-   diversity change rate

### Exploration

-   population Wasserstein change rate
-   centroid shift coherence
-   covariance spectral concentration

### Fitness Distribution

-   quantile improvement fraction
-   mean distribution improvement rate
-   fitness Wasserstein rate

### Exploitation

-   distance decay
-   stagnation
-   convergence rate

总计：

约8-12个Feature。

------------------------------------------------------------------------

# 10. Relation to Exploitation Behavior Metrics

## 10.1 定位

本项目的行为特征不是对已有 exploitation behavior 论文指标的逐字复现。

当前任务是：

$$ Search\ Behavior \rightarrow Query\ Utility $$

因此所有输入必须满足：

-   algorithm-agnostic；
-   low-cost；
-   不使用query feature；
-   不使用function identity；
-   不依赖真实全局最优位置；
-   可在不同维度、预算和优化器之间比较。

所以本文采用：

> budget-normalized behavior state

而不是：

> whole-run behavioral footprint

或：

> late-stage exploitation trigger.

------------------------------------------------------------------------

## 10.2 与 How do metaheuristics exploit? 的关系

*How do metaheuristics exploit?* 使用 distance-to-reference decay、directional entropy 和 stagnation indicators 描述粒子级 exploitation dynamics，并用global-best在50-iteration sliding window内的relative improvement低于5%来划分late-stage exploitation phase。其方向统计要求agent displacement具有跨窗口身份语义，本文不直接采用；directional entropy及其direction bins已由permutation-invariant集合分布变化指标替代，不属于活动实现。

本项目与其关系如下。

| 指标 | 原论文口径 | 本项目口径 | 使用方式 |
|---|---|---|---|
| improvement rate | 用global-best的relative improvement阈值辅助确定exploitation phase | 2% FE-ratio窗口内best-fitness相对改善，并除以预算比例跨度 | 启发，不是复现 |
| population distribution change | 原论文的directional entropy依赖agent displacement | 使用经验Wasserstein、centroid shift/coherence和协方差谱集中度，不建立个体对应关系 | 只继承行为变化动机，不复现方向熵 |
| distance decay | 跟踪agent到真实或估计reference optimum的距离衰减 | 跟踪population成员到当前population-best的平均距离衰减，不使用真实全局最优 | 启发并替换reference |
| stagnation | 区分movement stagnation、personal-best stagnation和global stagnation | 使用最近一次best-fitness严格改善后的FE-ratio间隔，截断并归一化到10%预算窗口 | 改写为算法无关停滞状态 |

关键区别：

-   原论文的目标是分析和调节exploitation behavior；
-   本项目的目标是预测所评估固定 query 的 $U_{query}$ 是否大于 0；
-   原论文可以使用已知benchmark optimum进行后验行为研究；
-   本项目的decision输入不能使用真实全局最优或function identity；
-   原论文以iteration窗口和phase划分为主；
-   本项目以FE-ratio窗口构造任意检查点上的行为状态。

因此，论文中应写作：

> The proposed features are inspired by exploitation-oriented behavioral diagnostics, but are not intended as a direct reproduction of those metrics. We adapt the underlying behavioral notions into budget-normalized, algorithm-agnostic state variables suitable for predicting the utility of the evaluated fixed landscape query before computing its features.

不建议写作：

> We use the exploitation metrics proposed in that paper.

因为当前定义已经改变。

------------------------------------------------------------------------

## 10.3 与 Determining Metaheuristic Similarity Using Behavioral Analysis 的关系

Hayward and Engelbrecht提出20个behavioral characteristics，用于比较metaheuristics的behavioral similarity。其指标分为：

-   Exploration；
-   Exploitation；
-   Locality in the Search Space；
-   Communication；
-   Evaluation Effort。

该论文的核心方法是将整段搜索过程压缩成whole-run behavioral vector，并用聚类和frequency map比较算法行为相似性。本项目不采用whole-run similarity clustering，而是采用checkpoint-level behavior state预测Query Utility。

与本项目直接相关或启发关系如下。

严格来说，第一版主模型没有逐式直接复现Hayward and Engelbrecht的20个characteristics；直接继承的是trajectory-based behavioral characterization这一研究视角，以及其中若干可低成本重写的行为语义。

| Hayward and Engelbrecht指标 | 原论文含义 | 本项目处理 |
|---|---|---|
| DRoC | diversity temporal curve第一段斜率，用于描述从exploration到exploitation的转变 | 启发 `bf_diversity_change_w05`，但本项目使用5% FE-ratio局部相对变化，不使用two-piecewise whole-run slope |
| PPS | exploration结束时的population spread | 启发population diversity输入；本项目使用当前检查点mean pairwise distance |
| ARoC-A | 基于到global optimum的Euclidean distance变化，描述找到promising locality的速度 | 不直接使用；因为需要真实global optimum |
| ARoC-B | 基于objective gap变化，描述exploration阶段改善速度 | 启发 `bf_improvement_rate_w02`，但本项目使用当前best-fitness相对改善，不使用global optimum gap |
| PFV | exploration结束时与global minimum的objective gap | 不使用；因为会引入benchmark optimum信息 |
| CRoC | diversity curve第二段斜率，描述个体向单点收敛速度 | 启发 `bf_convergence_rate_w10`，但本项目使用10% FE-ratio局部相对diversity下降率 |
| LRoC-A | objective value在exploitation阶段的局部改善速度 | 与 `bf_improvement_rate_w02` 的语义相关，但不按phase/knee-point划分 |
| ER | exploitation开始时到global optimum的distance，表示promising locality半径 | 启发 `bf_distance_decay_w10`，但本项目替换为到当前population-best的平均距离 |
| nshared / ntotal / nend | 基于search trajectory network的locality指标 | 第一版不纳入；计算更重，并依赖trajectory discretization |
| BSoI / IRoF / best-strength / nbest | 基于interaction network的communication指标 | 第一版不纳入；不同算法的显式communication定义不一致 |
| number of function evaluations / ERT variants / %INFEASIBLE | evaluation effort指标 | 不作为主behavior输入；FE budget和query cost在资源口径中单独处理 |

可直接引用的观点或明确启发的部分：

-   使用behavioral characteristics而不是仅用最终性能来描述metaheuristic search behavior；
-   使用diversity、accuracy/improvement、convergence和locality等低成本trajectory statistics；
-   将behavior划分为exploration、exploitation、locality、communication和evaluation effort的思路；
-   `DRoC` 对本项目diversity change的启发；
-   `CRoC` 对本项目convergence rate的启发；
-   `ARoC-B` 和 `LRoC-A` 对本项目improvement rate的启发；
-   `ER` 和distance-to-optimum accuracy对本项目distance decay的启发。

不直接采用的部分：

-   whole-run 20维behavioral footprint；
-   two-piecewise linear approximation和knee-point划分；
-   依赖global optimum的accuracy指标；
-   STN locality指标；
-   IN communication指标；
-   DBSCAN clustering和frequency map similarity analysis。

原因：

这些设计服务于：

$$ Algorithm\ Similarity $$

而本项目服务于：

$$ Query\ Utility\ Prediction $$

两者的估计对象不同。

------------------------------------------------------------------------

## 10.4 论文可用表述

可在方法章节使用：

> Our behavior representation is motivated by prior behavioral analyses of metaheuristics, especially studies that characterize exploration, exploitation, diversity change, convergence, locality, and communication from optimization trajectories. However, our objective differs from whole-run algorithm similarity analysis and late-stage exploitation control. We therefore define checkpoint-level behavior states on a normalized evaluation-budget axis. Improvement rate, diversity change, permutation-invariant population and fitness distribution changes, distance decay, stagnation, and convergence rate are computed over fixed FE-ratio windows, so that the resulting variables are comparable across dimensions and optimizers and can be used before computing query features.

可在相关工作章节使用：

> Hayward and Engelbrecht proposed a behavioral characteristic suite for determining metaheuristic similarity, including diversity-rate, accuracy-rate, convergence-rate, locality, communication, and evaluation-effort characteristics. We adopt this trajectory-based view of metaheuristic behavior, but do not use their whole-run footprinting, knee-point extraction, STN/IN metrics, or clustering protocol. Instead, we retain only low-cost and algorithm-agnostic behavioral notions and reformulate them as local budget-normalized state variables for predicting Query Utility.

可在指标定义后使用：

> The distance and stagnation variables should be interpreted as decision-oriented behavioral summaries rather than estimates of true distance to the global optimum. In particular, distance decay is computed relative to the current population-best solution, and stagnation is measured by elapsed FE ratio since the last strict best-fitness improvement. This avoids using problem identifiers, global optima, or query-derived landscape information in the decision input.

------------------------------------------------------------------------

# 11. 不推荐直接加入的Feature

## Algorithm-specific

禁止：

PSO:

-   inertia
-   c1
-   c2

DE:

-   F
-   CR

CMA:

-   covariance

------------------------------------------------------------------------

## High-level Query Features

禁止：

-   ruggedness
-   modality
-   autocorrelation

原因：

这些属于Landscape Feature。

而 Decision-before-Feature 需要在固定 query 执行之前决策。

------------------------------------------------------------------------

# 12. Feature Selection Protocol

不能直接人工删除。

建议：

三个阶段。

------------------------------------------------------------------------

## Stage 1: Expert-designed core set

使用理论确定：

8-12个核心指标。

------------------------------------------------------------------------

## Stage 2: Feature importance analysis

只在训练集 OOF 上分析三个活动线性候选：

-   标准化 Logistic/Ridge 系数；
-   LDA 判别方向；
-   train-family permutation importance。

不为特征重要性另行引入 RF/XGBoost 或据 validation 结果筛选输入列。

------------------------------------------------------------------------

## Stage 3: Compact model

比较：

Full behavior set

vs

Selected behavior set

证明：

少量行为足够。

------------------------------------------------------------------------

# 13. Ablation Design

## Ablation A

Only Progress

验证：

简单优化进度是否足够。

------------------------------------------------------------------------

## Ablation B

Progress + Diversity

验证：

种群结构价值。

------------------------------------------------------------------------

## Ablation C

Progress + Diversity + Exploration/Exploitation

完整行为模型。

------------------------------------------------------------------------

## Ablation D

加入algorithm-specific feature

验证：

算法特征是否降低泛化。

------------------------------------------------------------------------

# 14. Feature与Search Maturity关系

Behavior Feature不是最终目标。

关系：

    Behavior Features

            |

            v

    Search Maturity

            |

            v

    Query Utility Prediction

            |

            v

    Decision

------------------------------------------------------------------------

# 15. 预期实验假设

H1:

低成本行为指标能够预测Query Utility。

H2:

算法无关行为比算法参数具有更好的OOD泛化。

H3:

组合Progress + Diversity + Exploration/Exploitation获得最佳性能。

------------------------------------------------------------------------

# 16. 当前实现中的正式Feature分类

当前 `behavior.features` 中的 `BEHAVIOR_FEATURE_COLUMNS` 共包含25个permutation-invariant算法无关行为特征。

这些特征只从已记录的 checkpoint population、fitness、best fitness、FE、FE_total、native update计数、FE ratio 和 dimension 计算，不使用额外目标函数调用，不使用query feature，不使用function identity、algorithm identity 或优化器内部参数。native update计数仅用于窗口跨度记录，不进入特征集合。

| 类别 | Feature | 实现列名 | 口径 | Feature group |
|---|---|---|---|---|
| Progress | FE ratio | `bf_fe_ratio` | 当前 `FE/FE_total`；逐行强制等于元数据 `FE_ratio` | time_only, base, primary, primary_with_maturity, all_candidates |
| Progress | improvement rate | `bf_improvement_rate_w02` | 名义2% FE-ratio、按完整原生 update 对齐的窗口内 best fitness 相对改善率 | base, primary, primary_with_maturity, all_candidates |
| Progress | improvement frequency | `bf_improvement_frequency_w02` | 名义2%窗口内相邻原生 update 发生严格 best-fitness 改善的比例 | base, primary, primary_with_maturity, all_candidates |
| Diversity | population diversity | `bf_diversity_mean_pairwise` | population平均两两距离，除以 `sqrt(dimension)` | base, primary, primary_with_maturity, all_candidates |
| Diversity | diversity change | `bf_diversity_change_w05` | 名义5%、按完整原生 update 对齐窗口内 population diversity 相对变化 | base, primary, primary_with_maturity, all_candidates |
| Diversity | diversity slope | `bf_diversity_slope_w05` | 名义5%窗口内逐 update diversity 对实际 FE ratio 的线性斜率 | primary, primary_with_maturity, all_candidates |
| Diversity | fitness diversity | `bf_fitness_diversity` | 当前 checkpoint fitness values 的标准差 | primary, primary_with_maturity, all_candidates |
| Diversity | relative fitness diversity | `bf_fitness_diversity_rel` | fitness标准差除以 `abs(mean fitness)+epsilon` | primary, primary_with_maturity, all_candidates |
| Set change | population Wasserstein rate | `bf_population_wasserstein_rate_w05` | 5%窗口等权经验Wasserstein-1，除以 `sqrt(dimension)` 与实际FE-ratio跨度 | primary, primary_with_maturity, all_candidates |
| Set change | centroid shift rate | `bf_centroid_shift_rate_w05` | 5%窗口centroid距离，除以 `sqrt(dimension)` 与实际FE-ratio跨度 | primary, primary_with_maturity, all_candidates |
| Set change | centroid shift coherence | `bf_centroid_shift_coherence_w05` | centroid shift占经验Wasserstein distance的比例 | primary, primary_with_maturity, all_candidates |
| Set shape | covariance spectral concentration | `bf_covariance_spectral_concentration` | 当前population协方差特征值的归一化Herfindahl concentration | base, primary, primary_with_maturity, all_candidates |
| Set change | population overlap | `bf_population_overlap_w05` | 当前population到5%窗口anchor population的近邻重叠比例 | all_candidates |
| Convergence | distance decay | `bf_distance_decay_w10` | 10% FE-ratio窗口内到当前population-best平均距离的相对下降 | base, primary, primary_with_maturity, all_candidates |
| Convergence | stagnation | `bf_stagnation_w10` | 最近一次 best-fitness 严格改善后的预算比例间隔，截断到10%窗口 | base, primary, primary_with_maturity, all_candidates |
| Convergence | convergence slope | `bf_convergence_rate_w10` | 10% FE-ratio窗口内 diversity 相对下降率 | base, primary, primary_with_maturity, all_candidates |
| Fitness distribution | quantile improvement fraction | `bf_fitness_quantile_improvement_fraction_w02` | 2%窗口内改善的经验fitness分位数比例 | primary, primary_with_maturity, all_candidates |
| Convergence | best fitness slope | `bf_best_fitness_slope_w05` | 5% FE-ratio窗口内 best fitness 对 FE ratio 的线性斜率 | primary, primary_with_maturity, all_candidates |
| Fitness distribution | mean improvement rate | `bf_fitness_distribution_improvement_rate_w02` | 排序fitness的平均带符号改善量，除以anchor尺度与实际FE-ratio跨度 | primary, primary_with_maturity, all_candidates |
| Fitness distribution | Wasserstein rate | `bf_fitness_wasserstein_rate_w02` | 排序fitness的平均绝对分位数变化，除以anchor尺度与实际FE-ratio跨度 | primary, primary_with_maturity, all_candidates |
| Elite | elite concentration | `bf_elite_concentration` | top-20% elite population diversity 与总体 diversity 的比值 | primary, primary_with_maturity, all_candidates |
| Elite | best-distance fitness correlation | `bf_best_distance_fitness_corr` | 个体到当前population-best距离与fitness的相关系数 | all_candidates |
| State | Search maturity | `bf_search_maturity` | `ES_t(1-XS_t)` 的可执行行为特征版本 | primary_with_maturity, all_candidates |
| State | linear Search maturity | `bf_search_maturity_linear` | maturity 的线性合成备选形式 | primary_with_maturity, all_candidates |
| State | exploration/exploitation ratio | `bf_explore_exploit_ratio` | 行为探索分量除以行为开发分量 | primary_with_maturity, all_candidates |

窗口测量 metadata 不属于 `BEHAVIOR_FEATURE_COLUMNS`。对 `suffix in {w02,w05,w10}`，behavior 输出：

```text
effective_window_ratio_suffix = (FE_t - FE_anchor) / FE_total
effective_window_fe_suffix = FE_t - FE_anchor
effective_native_updates_suffix = native_updates_t - native_updates_anchor
```

具体列名为：

```text
effective_window_ratio_w02
effective_window_fe_w02
effective_native_updates_w02
effective_window_ratio_w05
effective_window_fe_w05
effective_native_updates_w05
effective_window_ratio_w10
effective_window_fe_w10
effective_native_updates_w10
```

anchor 从逐次完整原生 generation/update 的运行历史中选择，不再使用稀疏正式 checkpoint 序列。对名义窗口 (W\in\{0.02,0.05,0.10\})，目标位置为 `FE_t - round(W * FE_total)`；若该位置不是完整 update 边界，则选择不晚于目标位置的最近完整 update。因此实际跨度满足 `round(W*FE_total) <= effective_window_fe < round(W*FE_total) + population_size`。trajectory 行只保存三个 anchor 到当前 checkpoint 的轻量集合/fitness 比较统计，以及最近10%预算内用于 improvement frequency、stagnation 和 slope 的逐 update 标量历史；不重复保存逐 update 的 RNG 或完整优化器内部状态。rate与slope一律使用实际 FE 跨度。`native_updates` 差值只作计算来源记录，不作为 Decision 或 Selector 输入。

其中：

- `time_only` 只使用 `bf_fe_ratio`，数学输入为 $X=\{FE\_ratio\}$；它是阶段信息 baseline，用于判断完整 Controller 的性能是否只是来自“在哪个优化阶段调用 Query”；
- `base` 使用9个紧凑、permutation-invariant行为特征；旧identity-dependent base不再保留为活动组；
- `primary` 加入低成本的population与fitness集合变化特征；
- `primary_with_maturity` 在 `primary` 基础上加入 Search Maturity 及其备选状态特征；
- `all_candidates` 进一步加入 `bf_population_overlap_w05` 与 `bf_best_distance_fitness_corr`，只作为消融和诊断口径，不作为主结论的单独证据。

------------------------------------------------------------------------

# 17. 最终冻结方案

Decision输入：

包含：

    BEHAVIOR_FEATURE_GROUPS 中指定的 bf_* 行为特征集合

不包含：

    Function ID

    Dimension

    Algorithm ID

    Algorithm parameters

    query features

目标：

学习：

> Unknown optimization problem下，搜索行为是否已经足以支持Landscape
> Analysis决策。
