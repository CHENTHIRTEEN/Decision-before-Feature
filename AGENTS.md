# AGENTS.md

# Decision-before-Feature 项目开发规范（中文版）

## 0. 文档目的

本文档是 Decision-before-Feature 项目的最高优先级开发规范。

用途：

1. 指导 Vibe/Codex 等智能开发代理进行代码开发；
2. 固化论文研究设计；
3. 保证实验协议不被随意修改；
4. 保证代码实现与论文目标一致；
5. 保证当前实验与历史实验严格隔离。

---

# 0.1 实验隔离（Experiment Isolation）规范（最高优先级）

## 背景说明

当前主机环境中可能存在大量历史科研项目、实验代码、文档、数据和结果文件。

这些内容可能来自：

- 之前的研究方向；
- 已废弃实验；
- 不同论文方案；
- 旧版本代码；
- 不同实验协议。

这些历史内容与当前 Decision-before-Feature 项目没有必然关系。

## 核心规则

### 禁止访问当前实验目录之外的任何内容

Agent 执行任务时：

**禁止主动读取、搜索、分析或引用当前实验目录之外的任何文件。**

包括但不限于：

- 其他代码仓库；
- 其他实验目录；
- 历史 Python 代码；
- 旧实验配置；
- 旧结果文件；
- 旧 Markdown 文档；
- 旧论文材料；
- 旧模型 checkpoint；
- 其他项目数据。

## 当前项目唯一可信来源

Agent 只能使用：

1. 当前 Decision-before-Feature 项目目录；
2. 当前项目 docs 文档；
3. 当前项目配置文件；
4. 当前项目生成的数据；
5. 当前项目生成结果。

其他位置默认视为：

不可使用的历史信息。

## 禁止行为

禁止：

- 扫描用户主目录寻找旧代码；
- grep 全盘寻找类似实现；
- 复制其他项目代码；
- 使用旧实验结果；
- 使用旧模型或旧数据。

如果需要参考历史项目：

必须由用户明确指定文件路径。

## 缺少文件处理

如果当前项目缺少：

- 数据；
- 代码；
- 配置；
- 模型；

Agent 必须：

1. 明确报告缺失；
2. 提出重新生成方案；
3. 不得自行搜索历史目录。

---

# 0.2 禁止的工程机制

不得在源码、配置、脚本或输出接口中实现：

- 文件哈希、checksum、digest、canonical-byte 比较或其他完整性身份系统；
- receipt、manifest、authorization、source closure、append-only、quarantine 或执行解锁机制；
- frozen/successor/v2/v3 式代码复制、一次性运行声明或隐藏阶段门；
- dry、smoke、synthetic validation、resource calibration 或与真实科学运行无关的替代工作流；
- pytest、JSON Schema、schema registry、测试目录或测试依赖；
- 代码生成器、脚本调用脚本、循环依赖、同一逻辑的多版本并存；
- 仅用于证明文件来源、字节一致、提交身份或运行许可的代码。

不使用 Python 内置 `hash()` 或 `hashlib` 派生实验随机数。随机数必须通过显式整数 seed、unit number、stream code、generation、target 和 event 交给 `numpy.random.SeedSequence`。

---

# 0.3学术方法与表述

论文、研究方案、配置、源码符号和输出字段必须使用与实际科学操作对应的领域术语，不得用治理或问责隐喻包装普通实验步骤。

- 算法比较使用“性能评价”“基准比较”或 `evaluation/benchmarking`；
- 实验设计与数据条件使用“有效性检查”“一致性检查”或“数据质量检查”；
- 模型假设与异常行为使用“模型诊断”“误差分析”或“失败分析”；
- 超参数、变换和样本选择影响使用“敏感性分析”或“稳健性分析”；
- 组件贡献使用“消融实验”，引用和数值对应关系使用“来源核对”或“交叉核对”。

不得使用“正向/负向”“正面/负面”“积极/消极”概括结果。必须写明指标、比较顺序、数值符号的含义、效应量、区间和阈值。`positive/negative` 仅可表示数学正负号、布尔条件或已有明确定义的类别标签。

“反事实”及 `counterfactual` 只可用于具有正式定义的潜在结果、结构因果模型、反事实解释/公平性或离策略评价，并须给出 intervention、estimand 和识别假设。共享状态上的多动作完整运行不属于该用法，不得据此提出因果主张。

`oracle` 只可表示现实中不可获得、由额外信息定义的理想决策规则；若只是从已运行候选中取最小 loss，必须称为“逐状态最佳动作”或 `best observed action`。不得使用 `headroom`、`gross gain`、`paid/free information`、`sham`、`pressure test`、`bundle`、`cost view` 等含义不精确或跨领域隐喻；应分别写成明确的性能差、实验条件、稳健性实验、特征集合和预算口径。`stress test` 仅在预先定义扰动、强度和失败判据时使用。

新术语或缩写只有在没有通行名称且给出数学定义、计算方法和与既有术语的区别时才能引入；不得用宣传性名称替代标准任务、基线、指标或统计方法。

除非研究对象本身是算法问责、偏差/歧视检测、合规评价或独立第三方检查，不得把普通的完整性检查、结果复核、字段检查、统计诊断或文献核对命名为“xx审计”、`audit`、`auditor` 或 `auditing`，也不得把它们写成科学贡献或独立实验阶段。此类任务在本项目中应优先命名为“consistency check”“verification”“validation”或“benchmark check”。若确需引用“algorithmic auditing”这类外部文献概念时，必须说明检查对象、规范性准则、检查者角色与独立性、证据程序和报告产物。

上述用语遵循领域文献的通常分工：COCO 与生物启发优化实验指南使用 benchmark、performance measure、reference algorithm、validation 和 statistical analysis；算法选择文献使用 SBS、VBS、portfolio、feature cost 与 performance gap；AOS 文献使用 operator selection、credit assignment 和 reward。机器学习中的 counterfactual 有因果模型或预测解释的正式含义，不能泛化为任意备选运行。Raji 等的 algorithmic auditing 只对应外部问责语境，不应泛化为本项目的普通检查或验证步骤；本项目默认优先使用 consistency / verification / validation / check。

---

# 0.4 对话输出规范

除非用户明确要求使用其他语言，Agent 的对话更新和最终输出必须使用中文。

每次对话结束时，最终输出除说明当前结果外，还必须提供“下一步建议”。如果下一步任务业务复杂、容易偏离实验协议或适合另开新对话推进，必须给出一段可直接复制使用的下一步 prompt。

---

# 0.5 用户观点核查与命令执行顺序

收到用户命令后，Agent 必须先识别其中是否包含可核查的事实判断、实验假设、因果解释或预期结论，并基于当前项目文档、配置、代码、数据和可复现检查独立判断其是否成立，然后再执行命令。

- 观点与当前证据一致时，在用户授权范围内继续执行；
- 观点仅部分成立时，必须说明成立范围与不成立部分，并在不改变用户目标和实验协议的前提下按修正后的理解执行；
- 观点与当前证据或实验协议冲突时，必须先给出具体依据，不得为了服从命令而把该观点写入代码、文档或实验结论；若不同处理方案会实质改变研究问题、数据、成本、风险或不可逆操作，则先请求用户决定；
- 当前项目内缺少足够证据时，必须明确标记为“尚无法验证”，不得将推测表述为已证实事实；可在授权范围内先执行不会依赖该观点成立的安全部分；
- 文件命名、排版、接口风格等偏好性要求，以及用户明确授权的安全操作，不属于需要证明真假的“观点”，不得据此增加不必要的执行阻塞。

该规则属于 Agent 的推理与沟通顺序，不得实现为源码中的审批门、receipt、manifest、authorization、执行解锁或其他被第 0.2 节禁止的工程机制。只有判断会影响执行方案或研究结论时，才需要在对话中显式报告核查结果。

---

# 1. 项目研究目标

本项目研究：

> 在黑盒优化中，Landscape Analysis 本身是否值得执行。

不是设计新的优化算法。

核心流程：

黑盒问题

↓

廉价优化探测

↓

算法无关搜索行为

↓

Search Maturity

↓

ELA Utility

↓

Decision-before-Feature

---

# 2. 数据与模型规范

采用：

Offline trajectory collection + supervised learning。

禁止在线控制器训练作为主实验。

优化算法池：

- DE
- PSO
- CMA-ES
- SHADE

Decision输入：

只允许算法无关行为：

- improvement rate
- diversity
- entropy
- stagnation
- distance decay

benchmark_reference_value、已知最优值 gap 以及所有 gap 字段只可用于离线标签和最终评价，不得进入 Behavior、Selection Reference 输入或 Decision X。

禁止：

- PSO参数；
- DE参数；
- CMA-ES内部参数；
- ELA Feature；
- Function ID；
- Algorithm ID。

Decision Model 活动候选固定为：

- LDA；
- Logistic Regression；
- Ridge。

模型主选择必须使用 BBOB-train 上的 nested function-family OOF decision utility；完整 BBOB-train 的 family-OOF 分数用于冻结 `oof_utility` threshold。BBOB-validation 只作冻结评价，不参与 preprocessing、选模、特征筛选或 threshold 拟合。AUROC、Average Precision、Spearman 为辅助指标；连续 Utility RMSE 只对 Ridge 定义。不得继续增加 Random Forest、XGBoost、LightGBM、MLP 或其变体作为活动 Decision Model 候选；Selection Reference 中固定的 action-loss regression 不受此条限制。

所有顺序策略的活动单位是完整 trajectory。每条 trajectory 最多执行一次 query；机会按整数 `FE` 排序，同一 FE 若存在多行再按 `decision_opportunity_index` 排序。给定 threshold，只在最早满足 `score > threshold` 的机会触发；未触发 run 的政策 Utility 为 0，首次触发后的状态在该策略下不可达。模型选择、inner/full-train threshold、validation、baseline、call rate、precision、Utility capture 和全部策略指标必须使用这一 run-level first-trigger 规则；逐状态 AUROC、Average Precision、Spearman 和 Ridge RMSE 只能作辅助 score 诊断。

Selection Reference 固定使用四个互不重复动作：`continue_current` 加其余三个 portfolio algorithms。模型使用多输出 `RandomForestRegressor`，输入为 query features、算法无关 behavior 与连续 `remaining_budget_ratio`。主目标变换固定为 `clipped_log10_gap_advantage_vs_continue_current`：

$$
Y_{s,a}=\log_{10}\!\left(\operatorname{clip}(L_{s,a},g_{\min},g_{\max})\right)
-\log_{10}\!\left(\operatorname{clip}(L_{s,\mathrm{continue}},g_{\min},g_{\max})\right).
$$

`continue_current` 的主 target 恒为 0。原 `statewise_minmax_observed_action_loss` 只作预设 Selector target 敏感性分析，不得生成主 selected action、Utility、Decision label 或政策评价。

正式 Utility 必须同时物化且逐状态配对五条路径：`skip`、`query_joint`、`query_matched_state_only`、`sampling_only_continue_current` 与 `behavior_only_full_budget`。主 estimand 是 `query_joint` 相对 `skip` 的 query+selector 联合策略效用；`behavior_only_full_budget` 使用 query 前 Behavior、`FE_query=0` 和完整剩余预算的四动作 Selector。`query_matched_state_only` 与 `query_joint` 使用相同 query realization、sample endpoint、query-adjusted 四动作 outcome 和剩余预算，只移除 query descriptors；`sampling_only_continue_current` 执行同一 query acquisition 后原生继续当前算法。五路径分解只表示固定模型、预算和 transition rule 下的操作性分解，不作因果解释。

Selection Reference、Utility、Decision dataset 与在线输出必须保存 `selected_equals_default`、`selected_equals_prefix` 和 `handoff_required`；其中 `handoff_required = not selected_equals_prefix`，并与 `handoff_type == population_transfer_initialization` 逐行一致。不得生成 `label_source` 或以 `same_algorithm/changed_algorithm` 代替这些显式关系。

Behavior 的 w02、w05、w10 必须基于逐次完整原生 optimizer update 的运行历史，不得从稀疏正式 checkpoint 中选择窗口 anchor。若名义 FE 位置不落在完整 update 边界，选择不晚于目标位置的最近完整 update，实际窗口必须不小于名义窗口且偏差小于一次 population update；实际 FE、ratio 与 native update 数只作 metadata，不进入 Decision Model 输入。

---

# 3. 实验协议规范

训练：

BBOB：

- 10D
- 20D
- 40D

测试：

- CEC2017
- CEC2022
- 工程问题

采用：

Function Family Split。

禁止：

随机函数实例划分。

---

# 4. ELA Utility

令 $\ell_k$ 为路径 $k$ 的 suite-specific floor/cap 截断后 terminal `log10_gap`，$T_k$ 为同一 decision state 到 terminal 的三次删失 wall-clock 中位数。主 `lambda_time=1` 时：

$$
U_{query}^{joint}=(\ell_{skip}-\ell_{query})
-\lambda_T\log_{10}(T_{query}/T_{skip}).
$$

主 Decision target 固定为 `u_query_joint_lamT_1`，标签必须离线生成。还必须保存 Behavior-only Utility、Query 相对 Behavior-only 的 `query_operational_increment_lamT_1`、matched-acquisition descriptor-use increment、state-only-vs-sampling increment 和 sampling-direct increment，并逐行满足五路径加法分解。联合效用、操作性增量和 query-feature 预测诊断不得互相替代。

---

# 5. Baseline

必须包含：

- Never ELA；
- Always ELA；
- Random Analysis；
- Traditional AAS；
- SBS；
- VBS。
- Time-only Controller：$X=\{FE\_ratio\}$，用于检验 Controller 是否只学习调用阶段。

---

# 6. 代码开发规则

修改代码前：

1. 阅读当前项目文档；
2. 检查实验协议；
3. 不改变研究假设；
4. 说明修改原因；
5. 保持模块独立。

新增实验必须说明：

1. 研究问题；
2. baseline；
3. 是否存在数据泄漏；
4. 结果保存方式。

---

# 7. Git规范

提交格式：

    [类型] 描述

例如：

    [feature] add behavior extractor

    [experiment] add CEC evaluation

禁止提交：

- 临时文件；
- 大规模原始数据；
- checkpoint；
- 自动生成文件。

---

# 8. 最终目标

实现：

优化经验数据

↓

算法无关搜索行为

↓

Search Maturity

↓

ELA Utility

↓

Decision-before-Feature

↓

资源感知算法选择。

最终目标：

> 证明 Landscape Analysis 本身也应该成为优化对象。
