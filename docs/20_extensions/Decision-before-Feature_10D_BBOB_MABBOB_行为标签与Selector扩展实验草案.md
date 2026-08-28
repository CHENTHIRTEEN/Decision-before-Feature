# Decision-before-Feature：10D BBOB + MA-BBOB 行为、标签与 Selector 扩展实验草案

> 创建日期：2026-08-27
>
> 状态：待细化的研究扩展草案，不替代当前主协议，不授权启动新数据生成或模型训练
>
> 参考工作：RL-DAS 与 AS-LGBM
>
> 本轮目的：先记录研究方向、证据边界、实验骨架与待定项，后续再确定字段、公式、配置和运行顺序

---

## 1. 研究范围与现有协议的关系

本扩展研究 10D 黑盒优化，问题集合由当前项目已经定义的 BBOB 与 MA-BBOB 训练/评价集合组成，算法池缩减为 `PSO`、`SHADE` 和 `CMA-ES`。该三算法设定只属于本扩展，不回写当前四算法主协议。总预算沿用 10D 的 `FE_total=10000`，种群规模保持 40，边界处理保持 `reflect`，科学端点继续采用严格等总 FE 的最终 gap、target hit 与 ERT；wall-clock 只通过完整在线路径实测进入部署评价和资源分析，不进入 Decision 标签。

系统仍保留两个不同职责的学习组件。Decision Model 在 landscape query 发生前，只读取算法无关 Behavior，判断是否值得执行 query。Selector 在 query 完成后，读取扩充 Behavior、ELA descriptors 和连续 `remaining_budget_ratio`，从三算法动作集合中选择后续动作。对于任一 prefix algorithm，动作集合为 `continue_current` 加另外两个算法，因此每个状态共有三个互不重复动作。

在当前项目定义中，`g_fe_selected_path` 衡量 Query-selected path 相对 Skip path 的等总 FE 性能差，不是单独的算法更换标签。算法是否更换继续由 `selected_equals_prefix` 与 `handoff_required = not selected_equals_prefix` 表示。因此本扩展保留 `g_fe_selected_path` 作为直接监督基线和所有候选模型的共同政策评价指标；最终采用哪一种训练标签，则由 BBOB-train nested function-group OOF 比较决定。算法更换率、首次更换位置、换算法后的端点差和 handoff failure 作为独立下游结果报告。

---

## 2. 本扩展的研究问题

### RQ-E1：扩充 Behavior 是否改善 query 决策？

在固定 Decision 模型、Selector、query 成本、三算法池与 run-level first-trigger 规则下，从 RL-DAS 的动态状态表示中提炼的无额外评价、算法无关行为量，能否提高 BBOB-train nested function-group OOF 的 first-trigger `g_fe_selected_path`，并在 MA-BBOB 评价问题上保持同一效应方向？

### RQ-E2：哪一种 Decision 候选标签最有用？

分别以 `g_fe_selected_path`、ERT-compatible hitting-FE difference、AS-LGBM softERT 思路和 RL-DAS 短期改进 reward 作为监督目标训练独立 Decision 模型时，哪一个模型在共同的 BBOB-train nested function-group OOF 比较中获得较高的 first-trigger `g_fe_selected_path`？这些候选只有在逐状态定义、信息边界和 OOF 评价均闭合后才能进入活动比较，不能依据 validation 结果临时选择。

### RQ-E3：扩充 Behavior 与不同原生 `pflacco` ELA 组如何共同影响 Selector？

在相同 query 样本、相同 query FE、相同 state-action outcomes 与相同 function-group OOF 划分下，比较现有 14 维 descriptors、当前 37/52 维 `pflacco` 组，以及受 AS-LGBM 启发的 61 维候选组，判断哪些 ELA 组在扩充 Behavior 已可用时仍能降低 Selector regret 并改善最终 `g_fe_selected_path`。

---

## 3. 两项参考工作的可借鉴内容与边界

### 3.1 RL-DAS

RL-DAS 将动态算法选择建模为强化学习问题，在固定 period 上重复选择差分进化算法。公开实现的活动状态包含 9 个量：归一化当前 best、fitness-distance correlation、dispersion、dispersion ratio、negative slope coefficient、average neutral ratio、non-improvable ratio、non-worsenable ratio 和优化进度。代码还维护算法动作后的 best/worst movement history，但该部分带有动作身份和特定 portfolio 语义，不能进入本项目的算法无关 Decision 输入。

RL-DAS 的若干状态量依赖额外局部采样及其目标函数评价。例如 population evolvability、negative slope、neutral ratio、non-improvable/non-worsenable ratio 和 average delta fitness 都读取采样动作的 fitness。它们不能原样加入 query 前 Behavior，否则会产生未计费的额外 FE。可借鉴的是这些量表达的行为关系，而不是其额外采样实现。

RL-DAS 公开代码的活动 reward 为

$$
r_t=\max\!\left(\frac{f_{\mathrm{best},t-1}-f_{\mathrm{best},t}}{f_{\mathrm{best},0}},0\right),
$$

其中分母由初始 population best 给出。该 reward 表示一个选择周期内的非负相对改进。它对目标值平移、分母接近零或分母符号敏感，也没有直接比较 Query 与 Skip，因此不能原样替代 `g_fe_selected_path`。本扩展只把它作为“固定未来 FE 窗口的密集改进监督”灵感来源。

### 3.2 AS-LGBM

AS-LGBM 的公开实现从 `pflacco` 计算 61 个 ELA 特征，涉及 `ela_meta`、`ela_distribution`、`ela_level`、dispersion、information content、nearest-better clustering、PCA 和 fitness-distance correlation。其代码为每个算法保留 30 次运行的 hitting-time 数据，失败用 `-1` 表示；`cal_ert` 用固定失败预算惩罚并除以成功次数，再用最小 ERT 对应的算法作为分类标签。该思路适合启发 Selector 的 target-reaching 评价，但它是跨重复运行聚合量，不是天然的逐状态、逐 run Decision 标签。

AS-LGBM 论文使用了 softERT 表述，但当前公开仓库的 Python 文件和主 notebook 中没有直接提供名为 `softERT` 的函数或可独立核对的公式。因而 softERT 的精确定义目前尚无法从公开代码确认。本草案只保留该候选位置；在获得并逐式核对论文定义前，不自行补写公式，也不让它进入活动标签。

---

## 4. Decision Behavior 扩充方案

### 4.1 保留的基础输入

基线 `D0` 保留当前 Decision 输入：28 个无成熟度 Behavior，加上由 BBOB-train nested function-group OOF 选择出的一个 Search Maturity 形式。Decision 仍不读取 ELA、query sample、function ID、suite、dimension、algorithm ID、optimizer-specific state、known optimum gap 或未来 action outcomes。活动 Decision estimator 继续使用固定的 `RandomForestRegressor(n_estimators=200, max_depth=8, max_features=sqrt)`。

### 4.2 RL-DAS 启发的无额外 FE 候选

新增候选只允许从逐次完整原生 optimizer update 的已发生历史计算，不调用额外目标函数评价，不读取算法内部参数，不依赖跨代个体身份。初步候选分为四组：

| 候选组 | 初步行为含义 | 与 RL-DAS 的关系 | 实现约束 |
|---|---|---|---|
| `R_dispersion` | elite 子集相对全 population 的离散度差与比例 | 对应 dispersion / dispersion ratio | 使用当前 population 与 fitness 排序；单位立方体坐标；排列不变 |
| `R_fdc` | 当前 population fitness 与到当前 best 点距离的稳健相关 | 对应 fitness-distance correlation | 优先评估现有 `bf_best_distance_fitness_corr` 是否从 diagnostic 提升为候选；不得使用 known optimum |
| `R_transition_response` | 相邻完整 update 的 fitness 分位响应斜率、优于前一 best 的新点比例 | 对应 negative slope 与 population evolvability 的行为含义 | 使用集合级分位数，不要求 parent-offspring 对应，不新增评价 |
| `R_neutrality` | 相邻完整 update 的改善、近似持平和变差比例及稳健尺度化平均变化 | 对应 neutral、non-improvable、non-worsenable 与 average delta fitness | 近似持平阈值必须由 train-only 稳健尺度预先确定；不得使用固定绝对 fitness 差跨函数比较 |

当前已有 `bf_fitness_quantile_improvement_fraction_w02`、`bf_fitness_distribution_improvement_rate_w02`、fitness Wasserstein、diversity、centroid motion 和 covariance change，因此新增字段必须先做定义重叠检查。若某候选只是现有字段的单调变换或近似重复，则不新增字段，而是在文档中记录对应关系。

### 4.3 Behavior 消融顺序

Behavior 消融采用逐组、非穷举方案，避免把所有候选一次性加入造成不可解释的维度扩张：

1. `D0`：现有 28 个无成熟度 Behavior + 选定的一个 maturity 形式；
2. `D1`：`D0 + R_dispersion`；
3. `D2`：`D0 + R_fdc`；
4. `D3`：`D0 + R_transition_response`；
5. `D4`：`D0 + R_neutrality`；
6. `D5`：只合并在 train nested OOF 中显示互补贡献且通过共线性/缺失率检查的候选组。

选择依据是 BBOB-train 的 function-balanced first-trigger mean `g_fe_selected_path`，并同时报告 call rate、handoff rate、terminal gap、coverage 和每组新增计算时间。MA-BBOB formal 不参与候选组选择；validation 只在 Behavior 组和 threshold 确定后评价，不参与字段选择。

---

## 5. Decision 标签与辅助监督：候选模型独立训练与选择

### 5.1 共同比较原则

本章所称“辅助监督”是候选训练目标，不表示把多个目标同时输入一个模型。每个候选标签单独训练一套 Decision 模型，不使用多任务学习，不把多个标签线性组合，也不把某一标签作为另一标签的 sample weight。各模型使用完全相同的 Decision X、fit-fold preprocessing、固定 `RandomForestRegressor(n_estimators=200, max_depth=8, max_features=sqrt)`、BBOB-train outer/inner function groups、训练权重、机会集合和 first-trigger 重建规则。所有候选标签统一规定为“数值越大，执行 Query 越有利”；若来源定义方向相反，必须在标签定义阶段预先乘以 $-1$，不能在结果产生后调整方向。

“最有用”只按共同的外层 OOF 政策目标判断，而不按各候选标签自身的 RMSE、相关系数或平均值判断。具体流程为：候选标签分别生成训练目标、分别拟合 Decision RF、分别在 inner OOF 中确定 threshold，再分别在 outer holdout functions 上按 run-level first-trigger 规则重建政策。最终按拼接 outer holdouts 后的 function-balanced mean first-trigger `g_fe_selected_path` 选择一个标签模型。threshold objective 并列时先选择触发 runs 较少者，仍并列时选择数值较大的 threshold。ERT、target-hit rate、terminal gap、call rate、handoff rate、coverage 与标签 RMSE 同时报告，但不改写主排序。

### 5.2 候选 GFE 标签

候选 `C_GFE` 使用严格等总 FE 的

$$
G_{FE}(s)=g\_fe\_selected\_path(s)
=\log\frac{E_{\mathrm{skip}}(s)+\epsilon_p}
{E_{\mathrm{query,selected}}(s)+\epsilon_p}.
$$

Query 路径包含 query sample FE、sample best、Selector action、必要 handoff 与较短 continuation；Skip 路径继续当前算法。runtime、wall-clock 和 component time 均不进入该标签。该候选是当前研究问题的直接监督基线。

### 5.3 ERT-compatible hitting-FE 标签

标准 ERT 保留为政策级科学端点，不能直接拆成普通逐状态标签。对每个固定 `function/definition × target` 层，标准 ERT 按

$$
ERT=\frac{\sum_i T_i}{\sum_i I_i}
$$

从重复运行联合重算，其中 $I_i$ 表示是否观察到 target hit，未命中运行的 $T_i$ 计完整 planned FE。由于分母依赖一组重复运行的命中数，标准 ERT 本身不进入逐状态回归。

为形成可训练的独立候选 `C_HITFE`，对每条状态和路径先定义 ERT-compatible 的单 run hitting-FE contribution：

$$
C_p(s)=
\begin{cases}
\max(FE_{\mathrm{hit},p}-FE_s,1), & \text{该路径观察到 target hit},\\
B_p(s), & \text{该路径未观察到 target hit},
\end{cases}
$$

其中 $p\in\{\mathrm{skip},\mathrm{query,selected}\}$，$B_p(s)$ 是从状态起算、已经计入 query FE 后的该路径完整 planned future FE。候选标签为

$$
Y_{\mathrm{HITFE}}(s)=
\log\frac{C_{\mathrm{skip}}(s)+\epsilon_{FE}}
{C_{\mathrm{query,selected}}(s)+\epsilon_{FE}}.
$$

该量只比较单 run 的 hitting-FE contribution，不称为 ERT，也不替代最终政策 ERT。两条路径均未命中时标签为 0；只有一条路径命中时，符号由两条 contribution 的大小确定。标准 ERT 仍从 candidate policy 的完整 outer-holdout/validation runs 联合重算。

### 5.4 softERT 候选标签

在论文公式完成来源核对后，softERT 可形成独立候选 `C_SOFT_ERT`。它必须先转换为同一状态上 Query-selected path 相对 Skip path 的连续差，并保证数值越大表示 Query 越有利。必须明确 success target 集合、失败 FE 贡献、跨 target 聚合方式、跨 seed 聚合方式、全失败时的定义以及是否能对每个计划状态产生有限标签。

`C_SOFT_ERT` 单独训练一套固定 RF，不与 `C_GFE` 或 `C_HITFE` 组合。它不得读取 wall-clock，也不替代最终 gap、标准 ERT 或共同的模型选择目标。若精确公式与本项目等总 FE、逐状态 first-trigger 或信息边界不兼容，则不进入活动候选集合。

### 5.5 RL-DAS reward 启发的候选标签

为避免直接使用对目标值平移敏感的原 reward，本扩展候选采用固定未来 FE 窗口的等预算 log-gap 改善差：

$$
R_h(s)=\log\frac{E_{\mathrm{skip}}^{(h)}(s)+\epsilon_p}
{E_{\mathrm{query,selected}}^{(h)}(s)+\epsilon_p},
$$

其中 $h$ 是从状态 $s$ 起算的预先指定未来 FE 窗口，Query 和 Skip 均计入同样的未来总 FE；terminal 窗口对应 `C_GFE`。每个预先指定的较短窗口分别构成一个候选 `C_REWARD_h` 并单独训练模型，不把多个窗口合并成多任务 loss。窗口集合必须在读取 validation 前确定；若需要从多个 $h$ 中选择，也在同一 BBOB-train outer/inner OOF 链内完成。

### 5.6 候选集合与选择规则

| 候选 ID | 训练标签 | 当前状态 | 共同外层评价 |
|---|---|---|---|
| `C_GFE` | terminal `g_fe_selected_path` | 可定义；直接监督基线 | first-trigger `g_fe_selected_path` |
| `C_HITFE` | `Y_HITFE` | 可定义；需确定 `epsilon_FE` | 同上，并报告标准 ERT |
| `C_SOFT_ERT` | softERT Query-vs-Skip difference | 待论文公式核对 | 同上，并报告标准 ERT |
| `C_REWARD_h` | 固定 future-FE window 的 `R_h` | 待确定窗口集合 | 同上 |

标签候选只有在所有计划训练状态上定义为有限数值，或具有预先指定且不删除运行的失败处理时，才能进入活动比较。候选选择先比较 function-balanced mean first-trigger `g_fe_selected_path`；若候选差值落在项目内预设 Utility tolerance 内，先选择 call rate 较低者，仍不能区分时按 `C_GFE`、`C_HITFE`、`C_SOFT_ERT`、`C_REWARD_h` 的顺序选择；多个 `C_REWARD_h` 仍并列时选择较长的 future-FE window。标签确定后，用同一候选名完成 full BBOB-train OOF threshold、最终 refit 和 validation；validation 不重新选择标签。拼接 outer-OOF 数值承担标签选择作用，不能再表述为已选择程序的无偏性能估计。

---

## 6. Selector 的 Behavior + ELA 扩展

### 6.1 Selector 基础结构

本扩展的 Selector 输入固定为：扩充后的无成熟度 Behavior、当前 query 的 ELA descriptors 和连续 `remaining_budget_ratio`。Search Maturity 三个候选字段仍不进入 Selector。10D 不需要 dimension routing，`dimension` 只作固定 strata metadata，不加入 feature columns。

Selector 主 target 继续使用相对 `continue_current` 的

$$
Y_{s,a}=\log_{10}\!\left(\operatorname{clip}(L_{s,a},g_{\min},g_{\max})\right)
-\log_{10}\!\left(\operatorname{clip}(L_{s,\mathrm{continue}},g_{\min},g_{\max})\right),
$$

其中 `continue_current` 的 target 恒为 0。三算法条件下，主 10D Selector 继续使用适配三输出的 `formal_multioutput_rf`；三对 one-vs-one RF 聚合作为 Selector sensitivity。selected action 必须逐行保存 `selected_equals_default`、`selected_equals_prefix`、`handoff_required` 和 `handoff_type`。

### 6.2 ELA 候选组

为区分“descriptor 组成”与“query sample FE”，主要 ELA 组先在同一 `lhs_50d` 样本上比较。10D 时每次 query 使用 500 个样本点，即总预算的 5%。初步候选为：

| 组 | 维数 | 原生实现 | 作用 |
|---|---:|---|---|
| `Q14_cheap` | 14 | 当前 `descriptor_cheap_invariant` | 现有主 query 基线 |
| `Q37_pflacco_standard` | 37 | PCA + NBC + dispersion + information content + distribution | 当前原生 `pflacco` 标准组 |
| `Q52_pflacco_broad_matched` | 52 | Q37 + level + fitness-distance correlation | 在同一 `lhs_50d` 上做匹配 FE 的 broad 组 |
| `Q61_as_lgbm_native` | 61 | Q52 + 9 个 `ela_meta` 输出 | 与 AS-LGBM 公开实现的原生特征家族对齐 |

`Q61_as_lgbm_native` 新增的 9 个字段为 simple linear、linear-with-interactions、simple quadratic 和 quadratic-with-interactions 的 adjusted $R^2$，线性模型 intercept、系数最小值、最大值、最大/最小比值，以及 simple quadratic condition number。`pflacco` 的 runtime 字段不进入 Selector X。所有组继续使用当前 unit-cube X 与 median/IQR Y preprocessing；非有限值与 group failure 显式保留并使用 fit-fold 内预处理，不允许删除失败问题后重算结果。

AS-LGBM 代码导入了 local optima network 功能，但其活动 61 维 `Cal_ELA` 返回值没有包含该组。考虑到额外局部搜索成本和与当前 query FE 账本的不一致，本扩展不加入 local optima network descriptors。

### 6.3 增加或减少 ELA 的决策规则

首先把 `Q14/Q37/Q52/Q61` 当作四个预先指定实验条件完整报告，不在 validation 上挑选。若后续需要形成一个部署用 ELA 组，只允许在 BBOB-train 的 nested function-group OOF 中，按以下顺序决定：先比较 OOF Selector selected-action terminal `log10_gap` 与 regret，再比较端到端 first-trigger `g_fe_selected_path`、failure coverage 和 query feature computation time。若新增组没有稳定改善，优先选择字段更少且 failure rate 更低的组。

减少 ELA 时只做 feature-family 级消融，不做逐列递归搜索。允许消融的单位为 `pca`、`nbc`、`dispersion`、`information_content`、`ela_distribution`、`ela_level`、`fitness_distance_correlation` 和 `ela_meta`。任何 family 保留/删除决定均只读取 train OOF，不读取 validation。

---

## 7. 数据角色、划分与评价

本扩展沿用当前整合配置的数据角色，但只读取 10D：训练侧为 BBOB-train 18 个函数与 `mabbob_formal` 24 个定义；评价侧为 BBOB-validation 6 个函数与 `mabbob_validation` 18 个定义。评价侧保持参数不变且不得参与 preprocessing、Behavior 组选择、ELA 组选择、候选标签选择、模型选择或 threshold 拟合。

全部交叉验证按 suite-aware `cv_group_id` 分组。BBOB 以 function ID 为 group，MA-BBOB 以完整 problem definition 为 group；同一 function/definition 的 instance、seed、prefix algorithm 与全部有序 states 必须留在同一 fold。禁止随机 state 划分、随机 trajectory 划分或让同一 function/definition 跨 fit 与 holdout。

模型选择、threshold、call rate、handoff rate和 Utility 全部使用 run-level first-trigger。每条 trajectory 最多触发一次 query；未触发 run 的 Utility 为 0；触发后状态不可达。逐状态 AUROC、Average Precision、Spearman 和 auxiliary-label RMSE 只作 score diagnostics。

Behavior 组、ELA 组、标签形式、模型与 threshold 的主选择只使用 BBOB-train function-balanced nested OOF。候选和 threshold 确定后，`mabbob_formal` 才可按当前整合配置进入最终 refit；同时保留“不加入 `mabbob_formal` 的 BBOB-only refit”作为训练扩充敏感性，以区分组件选择与生成问题训练扩充。评价侧先分别报告 BBOB-validation 与 `mabbob_validation`，再报告当前项目已经定义的两层 50/50 有限集均值；不把问题定义当作来自无限 function 超总体的随机样本。

---

## 8. 分阶段实验矩阵

为控制计算量并保留解释性，本扩展按四阶段推进：

| 阶段 | 固定项 | 比较项 | 主要选择依据 |
|---|---|---|---|
| E1 Behavior | `Q14_cheap`、`C_GFE`、固定 RF | `D0-D5` | BBOB-train nested OOF first-trigger `G_FE` |
| E2 标签 | E1 选定 Behavior、`Q14_cheap` | `C_GFE/C_HITFE/C_SOFT_ERT/C_REWARD_h` 独立模型 | 共同 outer-OOF first-trigger `G_FE` |
| E3 Selector ELA | E1 与 E2 已确定的 Behavior/标签 | `Q14/Q37/Q52/Q61` 与 family 消融 | OOF action regret、selected-path gap、端到端 `G_FE` |
| E4 联合评价 | 所有组件保持参数不变 | proposed 与 baselines | validation 两层有限集结果、coverage 与 failure sensitivity |

强制 baselines 包括 Never ELA、Always ELA、Random Analysis、Traditional AAS、SBS、VBS 与 Time-only Controller。另增加两个组件基线：现有 Behavior + Q14 Selector，以及扩充 Behavior + Q14 Selector，以分离 Behavior 扩充与 ELA 扩充的贡献。

---

## 9. 结果字段与保存要求

每个状态至少保存以下关系和端点：

- `g_fe_selected_path`、`g_fe`（仅最佳已观测动作诊断）、Skip/Query selected terminal gap；
- `selected_action`、`selected_equals_default`、`selected_equals_prefix`、`handoff_required`、`handoff_type`；
- standard ERT 所需的 `observed_first_hit_FE`、`target_hit_observed`、planned FE 与 target；
- `decision_label_candidate_id`、标签公式版本、标签方向与模型选择 fold role；
- `soft_ert_candidate_id`、公式版本和 target set，仅在 `C_SOFT_ERT` 启用后生成；
- `reward_horizon_id`、$R_h$、future FE window，仅在对应 `C_REWARD_h` 启用后生成；
- Behavior feature-group ID、ELA query ID、query sample design、feature columns 与 group failure；
- first-trigger FE、call rate、handoff rate、action/selector/query failure、coverage；
- runtime 只保存完整 online policy/path 实测及其状态，不生成 component-runtime 拼接结果。

不同 ELA 组、Behavior 组和标签组必须使用不同输出目录或显式 condition ID，不能混读后再依据结果补字段。

---

## 10. 数据泄漏与信息边界检查

1. Decision X 只能包含 query 前 Behavior；ELA 只进入 Selector。
2. 所有新增 Behavior 必须来自已经发生的完整原生 update，不新增目标函数评价。
3. known optimum、benchmark reference value 和 gap 只用于离线标签与最终评价，不进入 Behavior 或 Selector X。
4. `C_HITFE`、`C_SOFT_ERT`、$R_h$ 与标准 ERT 都不能回流为 input feature。
5. Selector 的扩充 Behavior join 继续使用完整状态键；不得因三算法池而放松 `prefix_algorithm`、seed 或整数 FE 对齐。
6. validation 不参与任何字段、候选标签、公式、模型、threshold 或 ELA family 选择。
7. 三算法实验产生的新 Selection Reference、Utility、Decision dataset 和模型必须与当前四算法结果隔离，不能复用后者的 action labels 或 selected-path Utility。

---

## 11. 后续需要细化的事项

以下内容在本轮有意保持待定，需用户后续决定或提供论文正文后再写入可执行协议：

1. AS-LGBM softERT 的精确公式、target 集合、失败项与聚合顺序；
2. `R_transition_response` 与 `R_neutrality` 的最终数学定义、稳健尺度和字段名；
3. `C_HITFE` 的 `epsilon_FE` 与 $R_h$ 的 future FE 窗口集合；每个窗口将作为独立候选模型，不做多任务或标签加权；
4. `Q52_pflacco_broad_matched` 与 `Q61_as_lgbm_native` 是否统一使用 `lhs_50d`，以及是否另设 `lhs_100d` 采样量敏感性；
5. 三输出 `formal_multioutput_rf` 与三对 pairwise RF 的具体超参数是否完全沿用当前 Selector；
6. 训练侧 BBOB/MA-BBOB 的 50/50 权重是否保持，或另做 suite-weight sensitivity；
7. 新增 Behavior 的缺失率、计算成本和共线性淘汰阈值；
8. 实验运行预算、分片规格与结果目录命名。

在上述事项确定前，本文件只作为研究设计草案，不修改 `behavior/features.py`、Landscape Query whitelist、Selector 代码、Decision 标签代码或正式配置。

---

## 12. 参考来源

1. Guo, H. et al. *Deep Reinforcement Learning for Dynamic Algorithm Selection: A Proof-of-Principle Study on Differential Evolution*. IEEE Transactions on Systems, Man, and Cybernetics: Systems 54(7), 4247–4259 (2024). DOI: [10.1109/TSMC.2024.3374889](https://doi.org/10.1109/TSMC.2024.3374889). [arXiv:2403.02131](https://arxiv.org/abs/2403.02131). [RL-DAS code](https://github.com/MetaEvo/RL-DAS).
2. Guo, Q., Wang, H. & Tian, Y. *Automated algorithm selection for black-box optimization using light gradient boosting machine*. Swarm and Evolutionary Computation 98, 102071 (2025). DOI: [10.1016/j.swevo.2025.102071](https://doi.org/10.1016/j.swevo.2025.102071). [AS-LGBM code](https://github.com/HandingWangXDGroup/AS-LGBM).
3. 当前项目实现依据：`behavior/features.py`、`landscape_queries/specs.py`、`tools/pflacco_query/extract.py`、`configs/phase1_train.yaml` 与 `configs/phase1_validation.yaml`。
