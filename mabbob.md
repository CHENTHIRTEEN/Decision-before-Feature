# MA-BBOB 方案总结

## 核心思路

本次实验不建议继续采用“BBOB 固定若干函数 × 30 seeds × 大量重复”的数据采集方式，而应将数据采集目标改为：

> **Landscape coverage + Behavior coverage + Action discrimination**
>
> 景观覆盖 + 行为覆盖 + 动作区分度

函数多样性只是第一层。真正需要验证的是：所选函数是否能够驱动 PSO、DE、CMA-ES、SHADE 进入足够丰富的搜索状态，并且不同动作在这些状态下确实产生不同结果。

## 数据源与算法配置建议

- 数据源推荐采用 **BBOB + 受控采样的 MA-BBOB**。
- 暂时保留现有四个算法：**PSO、DE、CMA-ES、SHADE**，不建议现在增加第五个算法。
- 边界处理默认应采用 **reflect**，因为简单 `clip` 会造成明显的 boundary accumulation，并扭曲 population geometry；`clip` 只作为敏感性分析基线保留。
- 正式采集前，应先进行一次 **portfolio adequacy pilot**，用于判断当前函数集与算法配置是否足以支持研究目标。

## 重点验证事项

当前四个算法在机制层面已经足够形成论文所需的行为异质性，因此接下来的重点不是继续扩充算法数量，而是谨慎确认：

1. **DE 与 SHADE 的具体实现**是否存在过度相似的问题；
2. 两者的**参数设置和预算**是否合理，避免因实现或调参不当而削弱行为差异；
3. 统一设置的 **population size = 40** 是否使四个算法被配置成不自然、缺乏代表性的版本；
4. 在 pilot 中检查不同函数、状态和动作是否确实表现出足够的差异。

## 结论

目前的优先级应是验证实验组合的充分性与自然性，而不是盲目增加函数、seed 或算法。推荐先完成 BBOB 与受控 MA-BBOB 的小规模 portfolio adequacy pilot，再根据景观覆盖、行为覆盖和动作区分度的结果决定正式采集方案。

在边界处理上，正式实验默认应采用 `reflect`，因为它比 `clip` 更少制造边界堆积，也更不容易把行为特征学成“大家都被压到边上了”。`clip` 可以继续保留为敏感性分析中的 baseline，用于量化边界处理本身对行为特征和最终性能的影响。

## 为什么不能只使用 BBOB

BBOB 的 24 个函数已经经过有意识的设计，并覆盖了典型的优化困难类型，包括：

| 类别                                  | BBOB 函数 |
| ------------------------------------- | --------- |
| Separable                             | F1–F5     |
| Low/moderate conditioning             | F6–F9     |
| High-conditioning unimodal            | F10–F14   |
| Multimodal, adequate global structure | F15–F19   |
| Multimodal, weak global structure     | F20–F24   |

因此，BBOB 已经覆盖 separability、conditioning、multimodality、global structure，以及 ridge、valley、plateau 等经典困难。COCO 对这组函数的设计目标，本身就是通过可解释的拓扑结构观察算法行为与缺陷。

但 BBOB 仍然是由 24 个离散函数族构成的“孤岛”。仅使用这些固定函数，难以充分观察同一算法在连续变化的 landscape 上发生的 behavior regime transition，也难以构造足够丰富的中间状态。例如，DE 可能从 exploration 逐步转向 stagnation，CMA-ES 可能从 fast contraction 转向 ill-conditioning adaptation。这类连续变化的行为状态正是 MA-BBOB 的主要价值所在。

## 为什么不能只使用 MA-BBOB

MA-BBOB 的价值在于通过函数组合

> **Fᵢ ↔ Fⱼ**

在 BBOB 函数之间生成大量中间景观。已有研究表明，MA-BBOB 能够填补 BBOB 原始问题之间的一部分 instance space，而且组合函数上的算法性能变化不一定在 ELA 空间中呈现平滑变化；组合问题的最佳算法也可能不同于两个 component function 各自的最佳算法。因此，MA-BBOB 很适合制造算法行为发生转变所需的“中间态”。

不过，MA-BBOB 不能完全替代 BBOB。其作者指出，随机 many-affine problems 可能集中在部分 landscape 区域；一些纯 BBOB 的极端结构（例如 linear slope，以及部分 unimodal landscape）在多函数非平凡组合中容易消失。默认 \(T = 0.85\) 时，平均约 3.6 个 BBOB component 具有非零权重，这还可能使随机组合天然偏向复杂混合结构。

## 推荐方案：Pure BBOB anchors + MA-BBOB bridges

在实现层面，建议默认边界处理使用 `reflect`，并将 `clip` 保留为对照策略。我们已经在正式 pilot 中验证，`clip` 会系统性增加 boundary accumulation，而 `reflect` 能更好地保留搜索动力学与几何结构。

最合理的数据源组合是：

> **Pure BBOB anchors + MA-BBOB bridges**
>
> 纯 BBOB 锚点 + MA-BBOB 桥接景观

具体而言：

- 使用具有代表性的纯 BBOB 函数作为 **anchors**，保留其清晰、可解释的极端结构；
- 使用受控采样的 MA-BBOB 组合填补函数之间的 instance space，生成连续或半连续的中间景观；
- 不将 MA-BBOB 的随机组合直接视为完整 benchmark，而是控制组合数量、权重与复杂度，避免样本过度集中于复杂混合区域；
- 在 pilot 中同时检查纯函数与组合函数是否覆盖不同的行为 regime，并确认动作差异在两类景观上都能被识别。

因此，研究目标不是获得尽可能多的 benchmark，而是构造一条从可解释的 BBOB 锚点通向丰富中间态的 landscape 轨迹，使同一算法在连续变化的景观上展现可分析、可区分的行为转移。

## 行为驱动的问题选择

这是新数据方案区别于单纯 AS-LGBM 式数据构造的关键。研究目标不是学习从 ELA 特征到算法标签的静态映射：

> \(ELA \rightarrow algorithm\)

而是学习从时变行为状态到搜索动作的映射：

> \(B_t \rightarrow G_{FE}\)

因此，最终训练问题不能只依据 ELA diversity 选择，而应同时考虑：

> **Landscape diversity + Observed behavior diversity**

也就是说，一个问题即使在 ELA 空间中与其他问题相距较远，如果它没有诱发新的搜索行为，也不一定值得纳入；相反，某些景观几何上相近，但能触发新的行为 regime 或 action response，应当优先保留。

## 两阶段采集流程

### 阶段一：廉价筛选候选池

首先生成较大的候选池，例如约 **200–300 个 MA-BBOB definitions**。候选池中的每个问题不必立即进行完整的 state-action branching，只需采用低成本的行为筛选：

- 使用 4 个算法：DE、PSO、CMA-ES、SHADE；
- 使用 2 个 prefix seeds；
- 在 10D 上运行；
- 先执行正常搜索轨迹，不进行昂贵的 state-action 分支实验；
- 从轨迹中提取现有的 31 维正式 Behavior 特征。

这些特征包括 improvement、diversity、fitness distribution、Wasserstein population movement、centroid shift、elite concentration、covariance、stagnation、convergence、movement 和 maturity 等。对每个问题 \(p\)，可构造跨算法的行为表示：

\[
Z^{behavior}_p = [Z_{DE}, Z_{PSO}, Z_{CMA}, Z_{SHADE}].
\]

每个算法再按 **early、middle、late** 三个阶段进行摘要，以保留搜索行为随时间的变化。

### 阶段二：基于覆盖度选择正式子集

将标准化后的 landscape 表示与行为表示拼接：

\[
[\Phi^{landscape}_p,\; Z^{behavior}_p].
\]

在这个联合空间中进行 maximin coverage selection，而不是简单随机抽取。可以采用：

- farthest-point sampling；
- k-medoids；
- maximin design。

最终目标是从约 300 个候选问题中选择约 **40–60 个 MA-BBOB problems**，使正式子集尽可能覆盖整个候选行为空间。

候选池与正式标签的职责应明确区分：候选池可以有约 300 个问题，但只有约 40–60 个问题需要生成完整的 full labels 和昂贵的 state-action 数据。这样可以把预算从重复运行转移到真正有信息量的 landscape 与 behavior coverage 上。

## Maximin coverage 的选择原则

最终 MA-BBOB 子集不是为了保留最多问题，而是为了覆盖最广的联合空间。对于候选问题集合 \(\mathcal{C}\)，可将选择目标理解为最大化未选候选点到已选集合的最近距离：

\[
\max_{\mathcal{S}\subset\mathcal{C}}\; \min_{p\in\mathcal{C}}\min_{q\in\mathcal{S}} d(p,q),
\]

其中距离应基于标准化后的 landscape 与 observed behavior 联合表示。实际实现可根据计算成本选择近似算法，不必拘泥于某一种具体方法。

## 建议的正式数据集规模

下面的规模可作为 pilot 通过 coverage audit 后的初始版本，而不是不可调整的文献规定：

| 部分                      |                      Train |   Internal OOD validation |
| ------------------------- | -------------------------: | ------------------------: |
| Pure BBOB                 | 18 functions × 2 instances | 6 functions × 2 instances |
| MA-BBOB pairwise bridges  |                      30–36 |                     12–18 |
| MA-BBOB sparse 3/4-way    |                      10–15 |                      6–10 |
| Behavior-maximin selected |                      10–15 |          不再进行后验挑选 |
| Dimensions                |                   10D、20D |                  10D、20D |
| Prefix seeds              |                        5–8 |                       5–8 |
| Algorithms                |           DE/PSO/CMA/SHADE |                      同上 |

40D 建议在协议开发阶段作为 **dimension stress test**：

\[
D_{train}=\{10,20\},\qquad D_{stress}=40.
\]

如果论文不主张跨维度泛化，则模型冻结后可以再将 40D 加回正式训练；如果主张跨维度泛化，则应将 40D 严格作为独立压力测试或 OOD 维度，而不是在开发过程中随意混入训练集。

## 避免 MA-BBOB 的 component leakage

当前的函数划分为：

\[
\mathcal{F}_{train}=\{1,2,3,4,6,7,8,10,11,12,15,16,17,18,20,21,22,23\},
\]

\[
\mathcal{F}_{val}=\{5,9,13,14,19,24\}.
\]

这组 18/6 split 应继续保留，因为 validation functions 覆盖了 BBOB 的主要类别，并且已有仓库配置按该划分组织了训练与验证。

MA-BBOB 训练问题只能由 \(\mathcal{F}_{train}\) 中的 component 生成；validation MA-BBOB 应由 \(\mathcal{F}_{val}\) 中的 component 生成，或至少保证其 component composition 与训练集合完全隔离。

禁止出现如下情况：

```text
Train:      0.5 F1 + 0.5 F24
Validation: F24
```

虽然 F24 没有作为纯函数直接出现在训练标签中，但它已经通过混合函数进入训练数据，随后再声称 F24 是 unseen，会破坏 function-family OOD 结论。验证集的 component 不能以任何形式从训练混合函数中泄漏。

## 分层构造新的问题集合

新的问题集合不应“随机多采”，而应按照研究目的分层设计：

\[
\mathcal{P} = \mathcal{P}_{\mathrm{anchor}} \cup \mathcal{P}_{\mathrm{bridge}} \cup \mathcal{P}_{\mathrm{behavior}}
\]

### 第一层：BBOB Anchor Set

建议保留全部 24 个 BBOB 函数，而不是只挑选其中一部分。24 个函数本身已经提供了具有代表性的 landscape extremes；真正需要增加的是后续的 action branching，而不是通过减少或反复运行基础函数来节省数据量。

当前仓库已有较为合理的 function-family split：

\[
\mathcal{F}_{\mathrm{train}} = \{1,2,3,4,6,7,8,10,11,12,15,16,17,18,20,21,22,23\}
\]

\[
\mathcal{F}_{\mathrm{val}} = \{5,9,13,14,19,24\}
\]

这组 18/6 划分建议继续保留。6 个 validation functions 覆盖 BBOB 的五大类别：F5、F9、F13/F14、F19、F24。当前仓库也已经按照这一划分使用 10/20/40D、3 instances、30 seeds 和四个算法，因此不必优先改动 anchor 层的 split。

真正需要改变的是数据预算的方向：不要继续纵向堆叠 30 个 seeds 的重复，而应把预算横向投入到更多 landscape 与 behavior coverage 上。

### 第二层：MA-BBOB Controlled Bridge Set

不建议直接随机采样 100 或 500 个 MA-BBOB 问题，然后仅以数量来宣称多样性。应明确控制 component composition，使组合有清晰的结构解释。

可将五类 BBOB landscape category 抽象为 \(C_1,\ldots,C_5\)。两两组合共有：

\[
\binom{5}{2}=10
\]

种 cross-category combinations。例如：

- \(C_1 + C_3\)：separable + high-conditioning unimodal；
- \(C_2 + C_5\)：moderate-conditioned + multimodal weak structure。

这些跨类别组合对于 Behavior Controller 尤其有价值，因为它们更可能制造算法搜索机制发生切换的条件，而不是简单重复某一类 landscape。

建议 MA-BBOB 至少包含三种组合复杂度：

- \(K=2\) 的 pairwise mixtures，作为主要的可解释桥接结构；
- \(K=3\) 的 sparse mixtures，提供更丰富但仍可分析的中间态；
- 少量 \(K=4\) 的 complex mixtures，用于测试组合复杂度增加后算法行为是否发生新的变化。

不建议主要使用很稠密的 24 个函数加权组合。MA-BBOB 允许使用多个 BBOB components，但组合数量与 weight sampling 会显著改变 problem distribution；过度稠密的组合容易让样本偏向复杂混合结构，并削弱纯函数极端性质的可解释性。

### 权重与桥接轨迹

对于 pairwise mixture，可采用：

\[
f_{\alpha}=\alpha f_i+(1-\alpha)f_j
\]

并使用示例权重：

\[
\alpha\in\{0.25,0.5,0.75\}
\]

这样每个 function pair 都能形成一条简洁的行为过渡轨迹：

> mostly \(F_i\) → balanced transition → mostly \(F_j\)

相比于无结构的随机采样，这种设计能够更直接地回答研究问题：当 landscape 从一种结构逐渐转向另一种结构时，同一算法何时、以何种方式从一种 behavior regime 转移到另一种 regime。

## 数据预算的重新分配

Anchor 层可继续保留现有的 30 seeds，以支持随机性估计和统计检验；但不应让所有新增预算都用于同一函数上的重复运行。新增预算应优先支持：

1. 不同 BBOB 类别之间的 controlled bridges；
2. 不同组合复杂度 \(K=2,3,4\)；
3. 权重从 \(F_i\) 主导到 \(F_j\) 主导的过渡点；
4. 能够触发算法行为变化的代表性 landscape 状态。

核心原则是把预算从“纵向重复”转移到“横向覆盖”：从单纯增加 seeds，转向增加 landscape coverage、behavior coverage 和 action discrimination。