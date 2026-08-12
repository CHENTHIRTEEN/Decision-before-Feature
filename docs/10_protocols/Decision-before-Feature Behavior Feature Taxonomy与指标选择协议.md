# Decision-before-Feature Behavior Feature Taxonomy与指标选择协议

> 实现同步（2026-08-12）：extractor 生成 34 个唯一、permutation-invariant、算法无关 `bf_*` 字段，其中 31 个进入正式模型/selector，3 个仅作诊断。`T0/B1/B2/B3` 分别为 1/19/25/31 个字段。旧 `29/27` 是加入 DynamoRep-lite 前的快照；旧 `36` 的 tuple 同时重复列出 elite 项，并包含文档定义与 `bf_fitness_diversity_rel` 重复的 IQR 项；该旧 IQR 实现还错误地以当前 IQR 自归一化，非零 IQR 时近似恒为 1。过渡口径 `35/32` 是移除重复 elite 项后仍保留该 IQR 项的输出/正式输入计数。三组旧计数均已废止。w02/w05/w10 均来自逐次完整原生 update 历史。

## 1. 文档定位

本文档定义 Decision-before-Feature 框架中的 Behavior Feature 输入体系。

### 1.1 术语约定

为保证三份协议文档中的表述一致，本文统一采用以下术语：

- `search trajectory`：单次优化运行中随时间产生的状态序列；
- `behavior state`：从 `search trajectory` 聚合得到的、用于决策的低成本行为表示；
- `query utility`：固定 query 在当前状态下的效用值，记为 $U_{query}$；
- `decision model`：根据 `behavior state` 预测 `query utility` 并输出是否执行 query 的模型；
- `decision controller`：输出是否执行 query 的控制模块；
- `online evaluation`：在与训练一致的流式状态接口上进行推理与切换评估。

核心目标：

构建一种：

> algorithm-agnostic, low-cost, decision-oriented `behavior state`

用于预测：

$$ U_{query} $$

即：

所评估固定 query 的 $U_{query}$ 是否大于 0。

## 1.1 术语约定

为避免三份协议之间术语漂移，本文统一使用以下表达：

- `behavior feature`：算法无关行为特征；
- `behavior feature vector`：某一时刻或某一窗口下的行为特征向量；
- `behavior state`：行为特征向量与其对应的时间/预算上下文；
- `trajectory window`：按 FE-ratio 对齐的轨迹窗口；
- `event-triggered sample`：在状态突变点额外插入的观测；
- `budget milestone`：预定义的 FE-ratio 预算检查点；
- `query`：固定的 landscape-analysis query；
- `query utility`：$U_{query}$；
- `decision score`：用于判断是否执行 query 的连续分数 $s(x)$；
- `online evaluation`：使用与训练一致的流式状态接口进行推理和评估。

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

的最近完整原生 update 作为 anchor（若目标不在 update 边界，取不晚于目标的最近 update），则当前实现使用：

$$ IR_t^{(0.02)}=
\frac{f_{best}(a)-f_{best}(t)}
{\max(\operatorname{IQR}(f_{init}),\epsilon)}
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

预算比例窗口，统计窗口内相邻完整原生 update 发生严格 best-fitness 改善的比例。

用于区分：

偶然改善

和

持续改善。

------------------------------------------------------------------------

# 5. Category B: Population Diversity Features

## 5.1 Mean Pairwise Distance

定义：

先将每一维按问题边界映射到单位超立方体：

$$ \tilde{x}_{i,j}=\frac{x_{i,j}-l_j}{u_j-l_j} $$

然后在归一化坐标上计算：

$$ D_t= \frac{2}{N(N-1)}
\sum_{i<j}\|\tilde{x}_i-\tilde{x}_j\|_2 $$

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

当前与anchor population均视为等权经验分布，不按行建立个体对应关系。当前实现使用 $w=0.05$ 的FE-ratio窗口，并先将两侧 population 按问题边界映射到单位超立方体，再通过最小费用一一匹配计算经验Wasserstein-1：

$$
\tilde{x}_{i,j}=\frac{x_{i,j}-l_j}{u_j-l_j}
$$

$$
W_t^{(0.05)}=
\frac{1}{r_t-r_a}
\min_{\pi\in S_N}
\frac{1}{N}\sum_i
\lVert \tilde{x}_i^{(a)}-\tilde{x}_{\pi(i)}^{(t)}\rVert_2.
$$

该数值表示种群空间分布每单位预算比例的变化幅度，不表示真实个体运动。

------------------------------------------------------------------------

## 6.2 Centroid Shift与Coherence

centroid shift rate定义为：

先在单位超立方体中计算 centroid，再除以预算跨度：

$$
G_t^{(0.05)}=
\frac{\lVert \tilde{\mu}_t-\tilde{\mu}_a\rVert_2}{r_t-r_a}.
$$

coherence定义为未除预算跨度的centroid shift与Wasserstein distance之比。无集合变化时取0，其余裁剪到 $[0,1]$。该比值描述总体平移占集合分布变化的比例，不解释为个体方向一致性。

------------------------------------------------------------------------

## 6.3 Covariance与Fitness Distribution

当前population的协方差谱集中度使用归一化Herfindahl concentration。该量仍在原始坐标系上计算，因为它反映的是方差结构占比而不是绝对位置距离。令 $q=\min(d,N-1)$，协方差非负特征值为 $\lambda_j$：

$$
C_t=\frac{q\frac{\sum_j\lambda_j^2}{(\sum_j\lambda_j)^2}-1}{q-1}.
$$

退化协方差取0。fitness跨窗口变化先分别排序为经验分位数，令 $\delta_i=f_{(i)}^{(a)}-f_{(i)}^{(t)}$，再计算：

- 改善分位数比例；
- $\operatorname{mean}(\delta_i)$ 经优化器初始化后、任何原生 update 前已评估 population 的 fitness IQR 和 FE-ratio 跨度归一化后的分布改善率；
- $\operatorname{mean}(|\delta_i|)$ 经同一初始化 population IQR 和 FE-ratio 跨度归一化的一维 Wasserstein 变化率。

这些字段只表示经验分布变化，不表示同一个体的success、improvement variance或best improvement share。

------------------------------------------------------------------------

## 6.4 Movement / Direction / Success Diagnostics

当前代际逐个体的 movement、direction 和 success 定义默认要求跨 checkpoint 的第 `i` 行代表同一个体。这个假设对 PSO 只在稳定粒子编号下近似成立，对 DE / SHADE 只在 target index 或同源位点追踪下部分成立，对 CMA-ES 则不成立，因为每一代是重新采样的一批点，population 第 `i` 行不具备稳定跨代身份。

因此，下列逐个体特征不再作为严格的跨算法主特征：

- `bf_movement_magnitude`
- `bf_movement_diversity`
- `bf_direction_consistency_w05`
- `bf_directional_entropy_w05`
- `bf_success_rate_w02`
- `bf_improvement_variance_w02`
- `bf_best_improvement_ratio_w02`

推荐处理是将其拆成两个版本：

### 主模型：Permutation-invariant 版本

用集合层面的变化替代逐行对应：

- population Chamfer distance
- elite centroid shift
- covariance trace change
- covariance effective-rank change

这些量只依赖 checkpoint 集合分布，不依赖个体身份，因此可以作为算法无关主模型特征或诊断特征的候选。movement / direction / success 的主模型版本只在集合层面出现，不再保留逐个体强绑定口径。

### 诊断上界：Identity-aware 版本

仅对确实保留稳定个体身份语义的算法计算 movement / direction / success 特征，并将其作为 algorithm-specific 上界或诊断对照，不进入主 Decision 模型，也不与 permutation-invariant 主特征混报。

------------------------------------------------------------------------

# 7. Category D: Exploitation Features

描述：

开发已有区域的能力。

------------------------------------------------------------------------

## 7.1 Distance Decay

定义：

个体距离当前population-best位置的平均距离变化。当前实现先将 population 与 best 按问题边界归一化到单位超立方体，再计算平均距离。

$$ d_t=\frac{1}{N}\sum_i\lVert \tilde{x}_i-\tilde{x}_{best,t}\rVert_2 $$

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

预算比例窗口内的相对diversity下降率，其中 diversity 先基于边界归一化后的坐标计算：

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

> Our behavior representation is motivated by prior behavioral analyses of metaheuristics, especially studies that characterize exploration, exploitation, diversity change, convergence, locality, and communication from optimization trajectories. However, our objective differs from whole-run algorithm similarity analysis and late-stage exploitation control. We therefore define checkpoint-level behavior states on a normalized evaluation-budget axis. Improvement rate, diversity change, permutation-invariant population and fitness distribution changes, distance decay, stagnation, and convergence rate are computed over fixed FE-ratio windows, while movement / direction / success quantities are only retained as algorithm-specific diagnostics when stable individual identity exists. The resulting variables remain comparable across dimensions and optimizers and can be used before computing query features.

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

正式消融只比较以下四个嵌套输入组：

| 组 | 字段数 | 作用 |
|---|---:|---|
| T0 | 1 | 仅阶段信息 `FE_ratio` |
| B1 | 19 | core permutation-invariant behavior |
| B2 | 25 | 加入 DynamoRep-lite longitudinal set dynamics |
| B3 | 31 | 加入 movement 与 maturity behavior |

四组必须共享相同样本、模型候选、function-family split、train-only preprocessing、threshold 与评价口径。`all_candidates` 只是 B3 的兼容别名；`diagnostic_only` 不进入任何正式组。算法特定参数不进入 Decision 输入，因此不在本 feature-group 消融中加入算法特定字段。

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

当前 `BEHAVIOR_FEATURE_COLUMNS` 共 34 个唯一输出：`SELECTOR_BEHAVIOR_FEATURE_COLUMNS` 为 31 个正式输入，`diagnostic_only` 为 3 个诊断字段。`T0/B1/B2/B3` 的冻结计数为 1/19/25/31。

trajectory 的 `window_statistics` 与 `native_update_history` 都会记录底层 `fitness_iqr_rel`：前者保存各窗口共享的当前端点统计，后者保存逐原生 update 序列以计算 fitness-spread slope。它们是计算来源 metadata，不是两个 `bf_*` 字段，也不计入上述 34/31/3 合同。

正式特征组织为四个主组加一个诊断组：`T0`（time-only）、`B1`（core behavior / set-based baseline）、`B2`（longitudinal set dynamics，即 DynamoRep-lite）、`B3`（readiness-aware / maturity-aware），并将 `diagnostic_only` 作为独立诊断桶。`primary_with_dynamorep_lite` 是 `B2` 的兼容代码入口；`primary_with_maturity` 与 `all_candidates` 都是 `B3` 的兼容别名；三者均不得增加新的正式消融组。`diagnostic_only` 包含原始 fitness 标准差、population overlap 与 best-distance fitness correlation，且不进入 B3/`all_candidates`。

这些特征只从已记录的 checkpoint population、fitness、best fitness、FE、FE_total、native update计数、FE ratio 和 dimension 计算，不使用额外目标函数调用，不使用query feature，不使用function identity、algorithm identity 或优化器内部参数。native update计数仅用于窗口跨度记录，不进入特征集合。

## 16.1 轨迹采样与在线测评的修订原则

结合 trajectory-based algorithm selection、warm-starting 与 ELA sampling sensitivity 的近期工作，离线训练和 online 测评统一使用同一套冻结的多分辨率流式接口。修订原则如下：

1. 采样频率按搜索阶段分层，而不是全程固定。早期阶段保留高密度采样，中期阶段采用中密度，后期阶段降低里程碑频率。
2. 采样触发由“预算里程碑 + 冻结状态事件”共同决定。预算里程碑保证不同 run 可比，事件捕捉预定义的状态转折。
3. 首轮离线采样只使用预算里程碑与冻结状态事件，不由模型分数决定。模型冻结后的 online 附加复查才可使用 Q10 阈值邻近带，且所有比较策略共享相同决策机会。
4. 主模型和诊断特征必须分离。`bf_best_distance_fitness_corr` 与 `bf_population_overlap_w05` 保持为 `diagnostic_only`，不直接参与主 selector 的训练与部署。
5. 特征窗口保留 longitudinal 顺序信息，`w02/w05/w10` 严格从逐次完整原生 update 历史选择 anchor，不由输出状态密度决定。

这些修订并不改变当前特征定义本身，而是改变它们被采样、对齐和消费的方式，使离线建模与 online 测评共享一致的状态空间。

冻结的主采样协议为 `phase1_dynamic_budget_event_v1`。候选监测网格为 `0.20–0.60`、步长 `0.01`，在不早于目标比例的首个完整原生 update 上检查。必选预算里程碑按阶段为：early `[0.20,0.30)` 使用 `0.20/0.22/0.24/0.26/0.28`；mid `[0.30,0.50)` 使用 `0.30/0.34/0.38/0.42/0.46`；late `[0.50,0.60]` 使用 `0.50/0.60`。

事件采用未达到到达到的上穿与再武装：w02 improvement frequency 从 0 变为大于 0，回到 0 后再武装；stagnation span 达到 0.05，出现新的严格 best improvement 后再武装；w05 covariance effective-rank 相对变化绝对值达到 0.20（低于 0.10 再武装）；elite-centroid shift 除以 `sqrt(d)` 达到 0.05（低于 0.025 再武装）；diversity recovery 达到 0.10（低于 0.05 再武装）。每个跨过至少一个 `0.01` 监测网格的完整原生 update 只判定一次事件；若同一 update 跨过多个监测点且其中含预算里程碑，则以该里程碑为合并行名义节点，事件不消耗 event-only 配额、最小间隔锚点或 `event_index_in_phase`；若不含里程碑，则以最新跨过的监测点为名义节点。冻结的 update ratio 跨度不超过 `0.01`，不会同时跨过两个预算里程碑。每阶段最多 2 个 event-only 状态，event-only 状态的实际 FE-ratio 间隔至少 0.02；被 gap/quota 抑制落盘的 crossing 仍推进再武装状态。每个 run 因此输出 12–18 个状态，且不做事后样本重加权。

`FE_ratio` 始终保存实际 `FE/FE_total`；名义节点另存 `budget_milestone_ratio`，跨表状态键使用整数 `FE` 而非浮点 ratio。正式离线训练采样不由模型分数决定，不做事后样本重加权（每行 `sample_weight=1`）。冻结模型后的 online 额外复查才可使用阈值邻近带：带宽为完整 BBOB-train family-OOF 上 `abs(score-threshold)` 的第 10 百分位数；BBOB-validation 与外部测试不得拟合该带宽，且附加复查机会必须同时提供给所有被比较策略。

| 类别 | Feature | 实现列名 | 口径 | Feature group |
|---|---|---|---|---|
| Progress | FE ratio | `bf_fe_ratio` | 当前 `FE/FE_total`；逐行强制等于元数据 `FE_ratio` | T0, B1, B2, B3 |
| Progress | improvement rate | `bf_improvement_rate_w02` | 名义2% FE-ratio、按完整原生 update 对齐的窗口内 best fitness 相对改善率；分母使用初始化已评估 population 的 fitness IQR 稳健尺度 | B1, B2, B3 |
| Progress | improvement frequency | `bf_improvement_frequency_w02` | 名义2%窗口内相邻原生 update 发生严格 best-fitness 改善的比例 | B1, B2, B3 |
| Diversity | population diversity | `bf_diversity_mean_pairwise` | population平均两两距离，除以 `sqrt(dimension)` | B1, B2, B3 |
| Diversity | diversity change | `bf_diversity_change_w05` | 名义5%、按完整原生 update 对齐窗口内 population diversity 相对变化 | B1, B2, B3 |
| Diversity | diversity slope | `bf_diversity_slope_w05` | 名义5%窗口内逐 update diversity 对实际 FE ratio 的线性斜率 | B1, B2, B3 |
| Diversity | fitness diversity | `bf_fitness_diversity` | 当前 checkpoint fitness values 的标准差；原始版本不进入正式主模型，仅保留为兼容/诊断口径 | diagnostic_only |
| Diversity | relative fitness diversity | `bf_fitness_diversity_rel` | 当前 fitness IQR 相对优化器初始化后、任何原生 update 前已评估 population IQR 的比值；shift-invariant 稳健尺度 | B1, B2, B3 |
| Set change | population Wasserstein rate | `bf_population_wasserstein_rate_w05` | 5%窗口等权经验Wasserstein-1，除以 `sqrt(dimension)` 与实际FE-ratio跨度 | B1, B2, B3 |
| Set change | centroid shift rate | `bf_centroid_shift_rate_w05` | 5%窗口centroid距离，除以 `sqrt(dimension)` 与实际FE-ratio跨度 | B1, B2, B3 |
| Set change | centroid shift coherence | `bf_centroid_shift_coherence_w05` | centroid shift占经验Wasserstein distance的比例 | B1, B2, B3 |
| Set shape | covariance spectral concentration | `bf_covariance_spectral_concentration` | 当前population协方差特征值的归一化Herfindahl concentration | B1, B2, B3 |
| DynamoRep-lite | fitness spread slope | `bf_fitness_spread_slope_w05` | 5% FE-ratio 窗口内 fitness IQR 的局部斜率 | B2, B3 |
| DynamoRep-lite | population centroid shift | `bf_population_centroid_shift_w05` | 5%窗口内 population centroid 的边界归一化 shift | B2, B3 |
| DynamoRep-lite | elite centroid shift | `bf_elite_centroid_shift_w05` | 5%窗口内 elite centroid 的边界归一化 shift | B2, B3 |
| DynamoRep-lite | covariance trace ratio | `bf_covariance_trace_ratio_w05` | 5%窗口内 covariance trace 相对 anchor 的比值 | B2, B3 |
| DynamoRep-lite | covariance effective rank | `bf_covariance_effective_rank_w05` | 5%窗口内 covariance effective rank | B2, B3 |
| DynamoRep-lite | diversity recovery | `bf_diversity_recovery_w05` | 5%窗口内 diversity 相对回升幅度 | B2, B3 |
| Movement proxy | population Chamfer distance | `bf_population_chamfer_distance_w05` | 5%窗口内 anchor 与 current population 的边界归一化 Chamfer 风格集合距离 | B3 |
| Movement proxy | covariance trace change | `bf_covariance_trace_change_w05` | 5%窗口内 covariance trace 的相对变化 | B3 |
| Movement proxy | covariance effective-rank change | `bf_covariance_effective_rank_change_w05` | 5%窗口内 covariance effective-rank 的相对变化 | B3 |
| Set change | population overlap | `bf_population_overlap_w05` | 当前population到5%窗口anchor population的近邻重叠比例；依赖距离半径与population size，仅作诊断 | diagnostic_only |
| Convergence | distance decay | `bf_distance_decay_w10` | 10% FE-ratio窗口内到当前population-best平均距离的相对下降 | B1, B2, B3 |
| Convergence | stagnation | `bf_stagnation_w10` | 最近一次 best-fitness 严格改善后的预算比例间隔，截断到10%窗口 | B1, B2, B3 |
| Convergence | convergence slope | `bf_convergence_rate_w10` | 10% FE-ratio窗口内 diversity 相对下降率 | B1, B2, B3 |
| Fitness distribution | quantile improvement fraction | `bf_fitness_quantile_improvement_fraction_w02` | 2%窗口内改善的经验fitness分位数比例 | B1, B2, B3 |
| Convergence | best fitness slope | `bf_best_fitness_slope_w05` | 5% FE-ratio窗口内 best fitness 对 FE ratio 的线性斜率 | B1, B2, B3 |
| Fitness distribution | mean improvement rate | `bf_fitness_distribution_improvement_rate_w02` | 排序fitness的平均带符号改善量，除以初始化 population 的 fitness IQR 与实际FE-ratio跨度 | B1, B2, B3 |
| Fitness distribution | Wasserstein rate | `bf_fitness_wasserstein_rate_w02` | 排序fitness的平均绝对分位数变化，除以初始化 population 的 fitness IQR 与实际FE-ratio跨度 | B1, B2, B3 |
| Elite | elite concentration | `bf_elite_concentration` | top-20% elite population diversity 与总体 diversity 的比值 | B1, B2, B3 |
| Elite | best-distance fitness correlation | `bf_best_distance_fitness_corr` | 个体到当前population-best距离与fitness的相关系数；接近局部 landscape proxy，仅作诊断 | diagnostic_only |
| State | Search maturity | `bf_search_maturity` | `ES_t(1-XS_t)` 的可执行行为特征版本 | B3 |
| State | linear Search maturity | `bf_search_maturity_linear` | maturity 的线性合成备选形式 | B3 |
| State | exploration/exploitation ratio | `bf_explore_exploit_ratio` | 行为探索分量除以行为开发分量 | B3 |

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
- `diagnostic_only` 包含 `bf_fitness_diversity`、`bf_population_overlap_w05` 与 `bf_best_distance_fitness_corr`；
- `all_candidates` 是 B3 的兼容别名，严格等于 31 个正式输入，既不含 `diagnostic_only` 的 3 个字段，也不作为第五个独立正式消融组；正式 feature-group 消融只有 `T0/B1/B2/B3`。

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
