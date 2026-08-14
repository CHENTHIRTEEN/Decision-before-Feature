# 文献阅读与引用边界记录

本记录汇总三组材料：此前通过本地 Zotero 核读的 10 篇 Elsevier 期刊论文、当前项目 `docs/90_literature` 中的 5 篇独立全文，以及与本文方法相邻的 Zotero 研究。Zotero parent key 是文献条目标识，attachment key 是实际读取的附件标识，二者均不同于 BibTeX citation key。Jankovic et al. (2022) 当前只以项目内 PDF 为证据来源，未绑定 Zotero parent 或 attachment。

撤回状态核对口径：截至 2026-08-13，10 个 DOI 的 Crossref 类型均为 `journal-article`，`relation`、`update-to`、`updated-by` 均为空；Zotero 索引全文未检出 `retracted` 或 `withdrawn` 标记。这表示本轮核对未发现撤回信息，不构成永久状态保证。

## A. 已核读的 10 篇 Elsevier 期刊论文

### A.1 Cenikj et al. (2025)

- 题名：*Landscape features in single-objective continuous optimization: Have we hit a wall in algorithm selection generalization?*
- 作者：Gjorgjina Cenikj；Gašper Petelin；Moritz Seiler；Nikola Cenikj；Tome Eftimov
- 期刊／年份：*Swarm and Evolutionary Computation*，2025
- DOI：`10.1016/j.swevo.2025.101894`
- Zotero parent／attachment key：`EI7EEHFN`／`JD9BN76B`
- BibTeX key：`cenikjLandscapeFeaturesSingleobjective2025`
- 阅读载体：Zotero 索引 PDF 全文；核读摘要、数据划分、特征—性能对齐、AS 结果、讨论与局限。
- 可支撑：相似实例跨训练/测试会使 AS 评价偏乐观；该文的 problem split 与 problem-combination split 中，多组景观表示未优于简单基线；SBS 是关键基线；ELA 对采样和变换敏感。
- 不可支撑：本项目 family split 或阈值已有效；BBOB-validation、CEC 或工程泛化已经成立；行为特征必然优于 ELA；ELA 普遍不值得执行。
- 正文用途：Introduction 的泛化风险；Related Work 的严格划分与 SBS；Experimental Setup 的 function-family split 动机。

### A.2 Cenikj et al. (2026)

- 题名：*A survey of features used for representing black-box single-objective continuous optimization*
- 作者：Gjorgjina Cenikj；Ana Nikolikj；Gašper Petelin；Niki van Stein；Carola Doerr；Tome Eftimov
- 期刊／年份：*Swarm and Evolutionary Computation*，2026
- DOI：`10.1016/j.swevo.2026.102288`
- Zotero parent／attachment key：`3UUR344L`／`ZR8FCXYS`
- BibTeX key：`cenikjSurveyFeaturesUsed2026`
- 阅读载体：Zotero 索引 PDF 全文；核读特征 taxonomy、ML pipeline、轨迹特征、成本、泛化挑战与统计报告建议。
- 可支撑：区分问题景观、算法、高层交互和轨迹特征；静态 ELA 通常需要额外样本；轨迹特征可复用既有评价但仍有计算开销；现有研究过度依赖 BBOB，跨 benchmark 证据有限。
- 不可支撑：本项目五个行为量已充分定义 Search Maturity；行为特征零成本；当前 Decision Model 或跨数据集结果已经有效；应新增深度模型。
- 正文用途：最新特征 taxonomy；静态 ELA 与轨迹表示的成本区别；Reproducibility 的样本量、重复、维度、方差和效应量要求。

### A.3 Korošec and Eftimov (2024)

- 题名：*Opt2Vec — a continuous optimization problem representation based on the algorithm's behavior: A case study on problem classification*
- 作者：Peter Korošec；Tome Eftimov
- 期刊／年份：*Information Sciences*，2024
- DOI：`10.1016/j.ins.2024.121134`
- Zotero parent／attachment key：`QVEKYZRI`／`XXEXY6KA`
- BibTeX key：`korosecOpt2VecContinuousOptimization2024`
- 阅读载体：Zotero 索引 PDF 全文；核读数据生成、群体状态表示、autoencoder、分类实验与讨论。
- 可支撑：群体状态可用于学习 problem--algorithm interaction；可从算法实际访问区域进行动态表征；单次迭代群体在该文问题分类任务中含可辨识信息。
- 不可支撑：Opt2Vec 已验证算法选择或 ELA utility；问题分类准确率等于跨族 decision utility；本项目应采用 autoencoder；Opt2Vec 输入天然算法无关。
- 正文用途：trajectory representation；说明搜索历史是可学习的信息源，同时明确任务差异。

### A.4 Ochoa et al. (2021)

- 题名：*Search trajectory networks: A tool for analysing and visualising the behaviour of metaheuristics*
- 作者：Gabriela Ochoa；Katherine M. Malan；Christian Blum
- 期刊／年份：*Applied Soft Computing*，2021
- DOI：`10.1016/j.asoc.2021.107492`
- Zotero parent／attachment key：`8KA6TKB7`／`IXNZLH44`
- BibTeX key：`ochoaSearchTrajectoryNetworks2021`
- 阅读载体：Zotero 索引 PDF 全文；核读 STN 定义、日志构建、连续/组合案例、指标与结论。
- 可支撑：搜索行为可由正常运行轨迹分析，无需额外目标函数采样；可刻画收敛、吸引区域、循环和算法差异；适用于 evolutionary、swarm 和 local search。
- 不可支撑：STN 指标等同于本项目五个行为特征；STN 已证明 ELA 调用时机；无额外 sampling 等于无计算开销。
- 正文用途：算法无关搜索行为研究背景；复用 native optimizer history 的动机。

### A.5 Malan and Engelbrecht (2013)

- 题名：*A survey of techniques for characterising fitness landscapes and some possible ways forward*
- 作者：Katherine M. Malan；Andries P. Engelbrecht
- 期刊／年份：*Information Sciences*，2013
- DOI：`10.1016/j.ins.2013.04.015`
- Zotero parent／attachment key：`55KGWGLI`／`GHUP2DTT`
- BibTeX key：`malanSurveyTechniquesCharacterising2013`
- 阅读载体：Zotero 索引 PDF 全文；核读景观要素、技术 taxonomy、search independence、实用性讨论与结论。
- 可支撑：景观分析包含不同假设、采样需求和 search dependence；许多精确定义实际需样本近似；单一特征通常不足以映射算法性能。
- 不可支撑：本项目 `U_ELA` 定义或 λ 设置；特定 ELA 预算/阈值最优；ELA 一定提升优化性能。
- 正文用途：FLA/ELA 历史与实用边界；信息成本必须服务于决策的研究动机。

### A.6 Zhou et al. (2024)

- 题名：*Adaptive multi-population artificial bee colony algorithm based on fitness landscape analysis*
- 作者：Xinyu Zhou；Xiaocui Zhang；Weifeng Gao；Hui Wang；Yong Ma
- 期刊／年份：*Applied Soft Computing*，2024
- DOI：`10.1016/j.asoc.2024.111952`
- Zotero parent／attachment key：`MAGSF953`／`A6D77FY5`
- BibTeX key：`zhouAdaptiveMultipopulationArtificial2024`
- 阅读载体：Zotero 索引 PDF 全文；核读景观驱动机制、消融、CEC2013/CEC2017、工程问题、统计比较和结论。
- 可支撑：已有群体智能方法用景观特征决定子群体数量及划分；景观信息可作为算法内部自适应输入；该类工作关注求解器机制。
- 不可支撑：其 CEC 或工程结果是本项目泛化证据；其收益已扣除 ELA 成本；ABC 专用机制证明算法无关行为跨算法有效。
- 正文用途：landscape-aware optimizer 实例；反衬本文不是设计新的群体智能算法。

### A.7 Yang et al. (2025)

- 题名：*Meta-black-box optimization for evolutionary algorithms: Review and perspective*
- 作者：Xu Yang；Rui Wang；Kaiwen Li；Hisao Ishibuchi
- 期刊／年份：*Swarm and Evolutionary Computation*，2025
- DOI：`10.1016/j.swevo.2024.101838`
- Zotero parent／attachment key：`M8BIK3XB`／`UB3BKVB6`
- BibTeX key：`yangMetablackboxOptimizationEvolutionary2025`
- 阅读载体：Zotero 索引 PDF 全文；核读形式化定义、优化对象、方法 taxonomy、泛化策略、benchmark 和结论。
- 可支撑：自动算法研究可优化参数、算子、结构和学习规则；泛化评价需要多样实例、交叉验证和稳健指标；信息获取可放在更广义自动优化背景中讨论。
- 不可支撑：本文是在线 MetaBBO controller；本项目已具备跨任务泛化；综述展望可代替本项目证据。
- 正文用途：自动算法设计背景；区分本项目的“分析调用决策”与求解器更新规则。

### A.8 Guo et al. (2025)

- 题名：*Automated algorithm selection for black-box optimization using light gradient boosting machine*
- 作者：Qingbin Guo；Handing Wang；Ye Tian
- 期刊／年份：*Swarm and Evolutionary Computation*，2025
- DOI：`10.1016/j.swevo.2025.102071`
- Zotero parent／attachment key：`D7WRXPLP`／`LGGDT7NT`
- BibTeX key：`guoAutomatedAlgorithmSelection2025`
- 阅读载体：Zotero 索引 ScienceDirect snapshot 全文；PDF 子项 `4ZDYX4VR` 存在但无索引全文。核读实例生成、ELA、selector、SBS、交叉验证、FE 成本和结果。
- 可支撑：Traditional AAS 的典型流程是采样、ELA、监督选择器和算法组合；SBS 是正式基线；特征所需 FE 应计入评价；预算会限制特征集合。
- 不可支撑：本项目应采用 LightGBM；其生成数据集准确率可迁移到本项目；其低成本结论等同于本项目 ELA cost；本项目泛化已成立。
- 正文用途：Traditional AAS 与 SBS；说明特征评价成本属于选择过程。

### A.9 Gomes Pereira de Lacerda et al. (2021)

- 题名：*A systematic literature review on general parameter control for evolutionary and swarm-based algorithms*
- 作者：Marcelo Gomes Pereira de Lacerda；Luis Filipe de Araujo Pessoa；Fernando Buarque de Lima Neto；Teresa Bernarda Ludermir；Herbert Kuchen
- 期刊／年份：*Swarm and Evolutionary Computation*，2021
- DOI：`10.1016/j.swevo.2020.100777`
- Zotero parent／attachment key：`7VPS77HD`／`E4UYFHR2`
- BibTeX key：`gomespereiradelacerdaSystematicLiteratureReview2021`
- 阅读载体：Zotero 索引 ScienceDirect snapshot 全文；PDF 子项 `YFPNYAID` 存在但无索引全文。核读 tuning/control 定义、纳入排除标准、方法分类和结论。
- 可支撑：tuning 是运行前设置，control 是运行中调整；numerical parameter control 与 operator selection 应区分；有助于避免术语混用。
- 不可支撑：AOS 的完整 taxonomy；ELA utility；在线 controller 应作为本项目主实验；本项目属于 parameter control。
- 正文用途：术语边界；说明本文采用 offline trajectory collection + supervised learning，而非在线参数控制。

### A.10 Sallam et al. (2017)

- 题名：*Landscape-based adaptive operator selection mechanism for differential evolution*
- 作者：Karam M. Sallam；Saber M. Elsayed；Ruhul A. Sarker；Daryl L. Essam
- 期刊／年份：*Information Sciences*，2017
- DOI：`10.1016/j.ins.2017.08.028`
- Zotero parent／attachment key：`B4KXFCJ3`／`UJGRVVFN`
- BibTeX key：`sallamLandscapebasedAdaptiveOperator2017`
- 阅读载体：Zotero 索引 ScienceDirect snapshot 全文；PDF 子项 `4M4TT5JY` 存在但无索引全文。核读 AOS 背景、景观度量、DE 算子选择机制、实验和结论。
- 可支撑：AOS 可依据历史表现和景观信息选择算子；作者明确讨论离线景观分析的计算成本、训练测试依赖和新问题退化风险；既有方法回答“如何使用景观信息”。
- 不可支撑：本项目是 DE operator selection；DE 内部指标是算法无关行为；该文 CEC 结果证明本项目 ELA utility 或跨 benchmark 泛化。
- 正文用途：landscape-aware AOS 代表；直接支持“分析可能有收益，也有成本和迁移风险”。

## A.11 Elsevier 组统一引用边界

这 10 篇文献共同支持以下研究动机：静态景观特征已用于算法选择和自适应搜索，但具有采样、计算及泛化限制；优化轨迹可以提供 problem--algorithm interaction 信息。因此，在获取 ELA 之前预测其预期收益能否覆盖成本，是合理且尚需本项目正式实验回答的问题。

它们不能替代以下本项目产物：RQ1--RQ5 的正式统计结果、BBOB-train family-OOF 模型与阈值、BBOB-validation 冻结评价、CEC2017/CEC2022/工程问题泛化、ELA 成本扣除后的净效用，以及各 baseline 的最终性能差。上述内容在对应正式产物生成前不得写成已证实结论。

## B. `docs/90_literature` 项目内全文

目录核对结果：当前目录恰有 5 个 PDF，分别对应下列 5 篇独立论文。Jankovic et al. (CEC 2022) 与 Kostovska et al. (PPSN 2022) 具有不同题名、作者表、DOI、方法扩展与实验范围，必须作为两篇文献分别引用。当前目录不存在第二份重复的 PPSN PDF，也不存在把这两篇合并为一条记录的依据。

### B.1 Jankovic et al. (2022), CEC

- 题名：*Trajectory-based Algorithm Selection with Warm-starting*
- 出处／年份：*2022 IEEE Congress on Evolutionary Computation (CEC)*，2022，pp. 1--8
- DOI：`10.1109/CEC55065.2022.9870222`
- Zotero parent／attachment：不适用／不适用；证据来源为项目内 PDF `docs/90_literature/Trajectory-based Algorithm Selection with Warm-starting.pdf`
- BibTeX key：`jankovicTrajectorybasedAlgorithmSelection2022`
- 核读范围：PDF 全文；重点核读固定 CMA-ES 前缀、trajectory-based ELA、候选算法性能预测、warm-start 方式、预算处理、BBOB 5D 实验范围与 future work。
- 可支撑：由优化前缀产生的已评价点可以用于构建轨迹 ELA，从而避免另取一份全局景观样本；所构建表示随后用于选择第二阶段算法；算法切换需要明确的 warm-start 状态传递。
- 不可支撑：该文没有定义独立 query 的 paired skip/query utility，也没有在 descriptor 尚不存在时决定是否执行 query；复用前缀 FE 不等于表示构建没有时间或内存成本；其 BBOB 5D 结果不能证明本文的 BBOB-validation 或 CEC 泛化。
- 正文用途：Introduction 与 Related Work 中最接近的 per-run algorithm selection 前例；用于区分“表示已构建后的 portfolio action”与“表示构建前的 query-execution decision”。

### B.2 Kostovska et al. (2022), PPSN

- 题名：*Per-run Algorithm Selection with Warm-Starting Using Trajectory-Based Features*
- 出处／年份：*Parallel Problem Solving from Nature -- PPSN XVII*，2022，pp. 46--60
- DOI：`10.1007/978-3-031-14714-2_4`
- Zotero parent／attachment：`87C8LMEP`／`DH6L9G8V`
- BibTeX key：`kostovskaPerrunAlgorithmSelection2022`
- 核读范围：项目内 PDF 与 Zotero 索引全文；重点核读 trajectory ELA、CMA-ES 内部状态时间序列、组合表示、固定 switch point、六个第二阶段候选、warm-start 与 BBOB-to-YABBOB 评价。
- 可支撑：per-run selection 可以综合轨迹 ELA 与优化器内部时间序列来预测候选算法表现；BBOB 上训练的模型在 YABBOB 上性能下降，说明跨 benchmark 转移必须直接评价；作者仅将 trajectory-coverage mismatch 讨论为一种可能解释。
- 不可支撑：CMA-ES 内部变量不是本文允许的算法无关行为输入；该文的 direct switch recommendation 不是 query/skip 标签；其 YABBOB 结果只能说明跨套件评价的必要性，不能作为本文外部泛化证据。
- 正文用途：Related Work 中精确比较输入类型、标签和 transition；Discussion 中支持把跨 benchmark 表现写成待实证检验而非预期结论。

### B.3 Cenikj et al. (2023), DynamoRep

- 题名：*DynamoRep: Trajectory-Based Population Dynamics for Classification of Black-box Optimization Problems*
- 出处／年份：*Proceedings of the Genetic and Evolutionary Computation Conference (GECCO)*，2023，pp. 813--821
- DOI：`10.1145/3583131.3590401`
- Zotero parent／attachment：`6G44RBAL`／`V236D7CJ`
- BibTeX key：`cenikjDynamoRepTrajectoryBasedPopulation2023`
- 核读范围：项目内 PDF 与 Zotero 索引全文；核读纵向 population summaries、维度与群体规模相关的表示、分类标签、数据生成、计算量讨论、3D BBOB 范围与局限。
- 可支撑：跨代的 population-level summaries 可以刻画 problem--algorithm interaction；使用原生优化轨迹可以省去独立目标函数采样；纵向集合动态是轻量行为摘要的合理灵感来源。
- 不可支撑：问题分类准确率不等同于 query utility；DynamoRep 的代内坐标／适应度 min、max、mean、std 对 population 行重排本身不敏感，但其分类器按 optimizer 分别训练，坐标级表示长度随维数和代数增长；3D BBOB 问题分类不等同于固定维、跨优化器的 query-utility state，也不能证明本文 B2/B3 特征或 Search Maturity 有效。
- 正文用途：Related Work 的 trajectory representation；Method 中解释 DynamoRep-lite 只借鉴“纵向集合动态”思想，同时采用本文冻结的算法无关集合统计与窗口定义。

### B.4 Hayward and Engelbrecht (2025)

- 题名：*Determining Metaheuristic Similarity Using Behavioral Analysis*
- 出处／年份：*IEEE Transactions on Evolutionary Computation*，2025，29(1): 262--274
- DOI：`10.1109/TEVC.2023.3346672`
- Zotero parent／attachment：`DT4Z63IC`／`YK6ZN4NA`
- BibTeX key：`haywardDeterminingMetaheuristicSimilarity2025`
- 核读范围：项目内 PDF 与 Zotero 索引全文；核读行为特征定义、whole-run profiling、相似性与聚类分析、测试函数、维度、预算及方法限制。
- 可支撑：元启发式可通过 exploration、exploitation、locality、communication 与 evaluation effort 等多个行为维度比较，而不只比较最终性能；行为是 problem-dependent 的。
- 不可支撑：部分量依赖已知最优解、持续个体身份、局部邻域或交互结构，不能直接作为本文的部署时算法无关输入；whole-run similarity 不是 state-conditional query utility；聚类结果不验证 Search Maturity。
- 正文用途：Related Work 的 behavior characterization；Method 中说明为何本文只采用当前状态可获得、算法无关且集合置换不变的行为量。

### B.5 Mbasso et al. (2026)

- 题名：*How do metaheuristics exploit? A particle-level behavioral study across the CEC 2025 benchmark functions*
- 出处／年份：*Evolutionary Intelligence*，2026，19: 77
- DOI：`10.1007/s12065-026-01190-7`
- Zotero parent／attachment：`GTVR5UVZ`／`M3STB69F`
- BibTeX key：`mbassoHowMetaheuristicsExploit2026`
- 核读范围：项目内 PDF 与 Zotero 索引全文；核读 particle movement、directional entropy、stagnation、distance-to-reference、行为监测开销、优化器特定 intervention、CEC 2025 实验与局限。
- 可支撑：distance decay、directional entropy 与 stagnation 是可解释的搜索行为维度；复用搜索历史虽不新增独立景观样本，行为监测仍有计算开销；本文只借鉴 distance-decay 与 stagnation 的行为语义，并重新定义距离参照。
- 不可支撑：optimizer-specific intervention 不是 query acquisition；particle-wise directional entropy 依赖跨代个体身份，因而未进入本文冻结输入；其 CEC 2025 数值、干预收益或结论不能证明本文 RQ1--RQ5、Search Maturity 或外部泛化。
- 正文用途：Related Work 与 Method 的行为 taxonomy；Results RQ5 的描述性解释边界，避免把行为相关性写成因果作用。

## C. Zotero 邻近研究

### C.1 Kerschke and Trautmann (2019)

- 题名：*Automated Algorithm Selection on Continuous Black-box Problems by Combining Exploratory Landscape Analysis and Machine Learning*
- 出处／年份：*Evolutionary Computation*，2019，27(1): 99--127
- DOI：`10.1162/evco_a_00236`
- Zotero parent／attachment：`USBZSEJH`／`3UJ2ENYL`
- BibTeX key：`kerschkeAutomatedAlgorithmSelection2019`
- 核读范围：Zotero 索引 PDF 全文；核读 ELA--machine-learning pipeline、portfolio、SBS/VBS、feature cost、leave-one-function-out 评价与局限。
- 可支撑：静态 ELA 与监督学习可以把问题实例映射到 portfolio action；SBS/VBS 是算法选择的重要参照；特征采样 FE 应纳入总预算；function-level split 比随机 run/instance split 更能暴露泛化难度。
- 不可支撑：descriptor 已计算后的 action selection 不等同于 descriptor 计算前的 query decision；leave-one-function-out 不等同于本文 function-family split；该文性能不验证本文 Selection Reference 或 Decision Model。
- 正文用途：Introduction 与 Related Work 的传统 ELA-based AAS 基线；Experimental Setup 中 SBS/VBS 和 function-family split 的文献脉络。

### C.2 Prager and Trautmann (2024), pflacco

- 题名：*Pflacco: Feature-based Landscape Analysis of Continuous and Constrained Optimization Problems in Python*
- 出处／年份：*Evolutionary Computation*，2024，32(3): 211--216
- DOI：`10.1162/evco_a_00341`
- Zotero parent／attachment：`RKR5X7CC`／`2YV9PIV5`
- BibTeX key：`pragerPflaccoFeaturebasedLandscape2024`
- 核读范围：Zotero 索引 PDF 全文；核读软件范围、支持的 feature groups、输入样本要求、连续与约束问题接口及计算注意事项。
- 可支撑：不同景观特征族具有不同输入、采样假设与计算要求；复现实验必须明确实际 query、样本设计、特征组和失败特征处理，而不能只写软件包名称。
- 不可支撑：软件可用不意味着 descriptor query 没有成本；该文未证明某一固定采样率（包括本文预设的 50d）普遍充分；不能据此声称所有 ELA query 等价。
- 正文用途：Related Work 的 feature acquisition；Experimental Setup 与 Reproducibility 中限定 query configuration 和成本口径。

### C.3 Renau and Hart (2024)

- 题名：*On the Utility of Probing Trajectories for Algorithm-Selection*
- 出处／年份：*Applications of Evolutionary Computation*，2024，pp. 98--114
- DOI：`10.1007/978-3-031-56852-7_7`
- Zotero parent／attachment：`M48DY73P`／`9KNUEEIT`
- BibTeX key：`renauUtilityProbingTrajectories2024`
- 核读范围：仅核对 Zotero bibliographic metadata、DOI 与摘要；当前证据未达到全文核读级别，因而所有支持均严格限制在摘要明示内容。
- 可支撑：摘要提出用短的、algorithm-centric probing trajectory 表示实例并服务于后续 algorithm selection；摘要将此方法与较昂贵的 landscape-based representations 作成本动机上的比较。
- 不可支撑：不能从摘要推断具体特征、分类器、split、warm-start 细节、效应大小或完整成本核算；该文不支持本文 query utility 公式、行为输入有效性或跨套件泛化。
- 正文用途：Related Work 中作为 trajectory/probing selection 的邻近方向，并在句内明确其证据仅来自摘要；不承担方法细节或数值论据。

### C.4 Renau and Hart (2025)

- 题名：*Algorithm Selection with Probing Trajectories: Benchmarking the Choice of Classifier Model*
- 出处／年份：*Applications of Evolutionary Computation*，2025
- DOI：`10.1007/978-3-031-90062-4_28`
- Zotero parent／attachment：`45FUDCHK`／`XIV9Y2WW`
- BibTeX key：`renauAlgorithmSelectionProbing2025b`
- 核读范围：Zotero 索引 PDF 全文，共 18 页；核读 probing-trajectory 定义、17 类时间序列分类器、LOIO/LOPO 划分、BBOB 任务范围、结果与局限。
- 可支撑：probing trajectory 是由短期 objective-value sequence 构成的 algorithm-centric 输入；分类器选择会影响此类时序输入的选择表现；leave-one-problem-out 比 leave-one-instance-out 更能暴露未见问题上的困难。
- 不可支撑：该文仍在 representation 已构建后直接选择 portfolio algorithm，不定义独立 query 的 paired skip/query utility；BBOB 分类器结果不能证明本文行为输入、阈值或跨套件泛化；不得把其分类器候选加入本文冻结的 LDA、Logistic Regression、Ridge 集合。
- 正文用途：Related Work 中补充 probing-trajectory 的模型与划分敏感性；支持把预测模型和 problem split 写成经验设计的一部分，同时精确区分其 action-selection 目标与本文 query-acquisition 目标。

### C.5 van der Blom and Vermetten (2026)

- 题名：*On the Influence of the Feature Computation Budget on Per-instance Algorithm Selection for Black-box Optimization*
- 出处／年份：arXiv preprint `arXiv:2605.04954`，2026
- DOI：`10.48550/arXiv.2605.04954`
- Zotero parent／attachment：`FM6J4CM4`／`6SQ5NWA3`
- BibTeX key：`blomInfluenceFeatureComputation2026`
- 核读范围：Zotero 索引全文；核读 feature-computation budget、总 FE 预算口径、不同 sampling fractions、portfolio/问题/维度/目标预算条件及 scenario-dependent 结果。
- 可支撑：用于特征计算的 FE 可以与所选优化器共享同一总预算；per-instance AS 表现对 feature budget 敏感；较合适的特征预算比例会随实验情景变化。
- 不可支撑：该预印本不提供跨 portfolio、维度、benchmark 与 performance criterion 普遍最优的预算比例；固定地考察若干 sampling fractions 不等同于在某个搜索状态先预测是否执行 query；不能用其结果补写本文 RQ1 或 RQ3。
- 正文用途：Introduction、Related Work 与 Experimental Setup 中论证 equal-total-budget 与 state/configuration-dependent acquisition 的必要性，同时标明其预印本身份。

### C.6 Guo et al. (2024), RL-DAS

- 题名：*Deep Reinforcement Learning for Dynamic Algorithm Selection: A Proof-of-Principle Study on Differential Evolution*
- 出处／年份：*IEEE Transactions on Systems, Man, and Cybernetics: Systems*，2024，54(7): 4247--4259
- DOI：`10.1109/TSMC.2024.3374889`
- Zotero parent／attachment：`7L6WZAID`／`5WQV85BP`
- BibTeX key：`guoDeepReinforcementLearning2024b`
- 核读范围：Zotero 索引全文；正式期刊元数据经 Crossref 题名与 DOI 交叉核对；核读 MDP 定义、PPO policy、landscape/history/context state、DE variant portfolio、动态调度、训练评价范围与 proof-of-principle 限定。
- 可支撑：dynamic algorithm selection 可以被表述为运行中重复选择 DE 变体的强化学习问题；其状态可同时使用景观、算法历史和上下文信息。
- 不可支撑：在线 RL scheduler 与本文 offline supervised query gate 的学习问题、动作空间和信息边界不同；DE-variant/CEC 条件下的结果不能证明跨算法或外部 suite 的 query utility；不能据此把在线 controller training 加入本文主实验。
- 正文用途：Related Work 的 dynamic algorithm selection 边界；用于说明本文决定的是信息获取而非反复改变求解器更新规则。

### C.7 Eimer et al. (2021), DACBench

- 题名：*DACBench: A Benchmark Library for Dynamic Algorithm Configuration*
- 出处／年份：*Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence (IJCAI)*，2021，pp. 1668--1674
- DOI：`10.24963/ijcai.2021/230`
- Zotero parent／attachment：`TRLTGG3D`／`FT8SKFUQ`
- BibTeX key：`eimerDACBenchBenchmarkLibrary2021`
- 核读范围：Zotero 索引 PDF 全文，共 7 页；核读 DAC 定义、state/action/reward/decision-frequency 构成、CMA-ES 与 ModEA benchmark、训练／测试实例和结论。
- 可支撑：动态算法配置把目标算法状态映射为超参数或组件动作，并需要明确 state、action、reward、instance set 与决策频率；其基准包含 CMA-ES 步长适配与 ModEA 组件的重复选择。
- 不可支撑：DACBench 控制的是优化器超参数或组件，而不是是否获取额外信息；其 benchmark 结果不验证本文 query utility、行为输入、BBOB-validation 或外部泛化。
- 正文用途：Related Work 中限定 dynamic configuration 与 Decision-before-Feature 的 action 语义，避免把信息获取 gate 写成参数控制或重复算法组件选择。

## D. 跨文献比较轴

下表只比较任务定义和证据边界，不将任何已发表结果转写为本文实验结果。`Split` 列记录原研究的评价边界或当前可核读范围；“未由摘要确认”表示不得据此继续推断。

| 研究 | Decision target | Information time | Input | Label／分析目标 | Cost treatment | Transition | Split／评价边界 |
|---|---|---|---|---|---|---|---|
| Kerschke and Trautmann (2019) | 选择 portfolio algorithm | 静态 ELA 已计算之后 | 独立样本的 ELA | performance-derived algorithm selection | 计入 feature sampling FE | 启动所选算法 | leave-one-function-out；不是 family split |
| Jankovic et al. (2022) | 选择第二阶段算法 | 固定 CMA-ES 前缀及 trajectory ELA 之后 | CMA-ES 前缀上的 ELA | 候选算法 fixed-budget performance | 复用前缀 FE；无 paired query/skip net utility | 单次 algorithm-specific warm-start | 5D BBOB、instance-group evaluation；CEC 留作未来工作 |
| Kostovska et al. (2022) | 选择第二阶段候选 | 固定 switch point 之后 | trajectory ELA、CMA-ES 内部时间序列或二者 | 候选 performance regression | 无独立 ELA sample；未把 acquisition/computation 合为 gate utility | 从 CMA-ES 单次 warm-start 切换 | BBOB 5D/10D；另测 BBOB-to-YABBOB |
| Renau and Hart (2024) | 选择 portfolio algorithm | probing trajectory 构建之后 | 短的 algorithm-centric probe sequence | algorithm-selection target | 摘要只支持“相对昂贵表示”的成本动机 | probe 后选择；细节未由摘要确认 | 具体 split 未由摘要确认 |
| Renau and Hart (2025) | 选择 portfolio algorithm | probing trajectory 构建之后 | 短期 objective-value time series | algorithm-selection classification | 比较 trajectory budget 与分类器计算；不定义 query/skip utility | probe 后一次选择 | BBOB 10D；分别采用 LOIO 与 LOPO，不含跨 suite 验证 |
| DynamoRep (2023) | 问题分类，不作在线 action decision | 观察多代 population dynamics 之后 | 纵向 population summaries | problem class | 复用已评价点；表示计算仍有成本 | 无 | 3D BBOB 分类范围 |
| Hayward and Engelbrecht (2025) | 行为相似性描述与聚类 | 完整运行之后 | 多维 whole-run behavioral measures | behavioral profile／cluster similarity | evaluation effort 是行为维度，不是 acquisition utility | 无 | 固定函数、20D、群体与预算条件；不是 selector split |
| Mbasso et al. (2026) | 行为诊断并触发 optimizer-specific intervention | 监测到 exploitation-related pattern 之后 | particle movement、entropy、stagnation、distance | 行为诊断与 optimizer performance | 报告监测开销；不定义独立 query utility | 修改 PSO/DE/GWO 内部行为 | CEC 2025；不证明 noisy/dynamic/real-world 转移 |
| van der Blom and Vermetten (2026) | 特征预算条件下的 per-instance algorithm selection | 特征已计算之后 | 静态景观特征与预算情景 | algorithm-selection performance | feature FE 与 optimizer 共享总预算 | 选择算法并用剩余预算优化 | 结论依赖 portfolio、问题、维度、预算与评价情景 |
| RL-DAS (2024) | 运行中重复选择 DE variant | 每次动态调度时，已有 landscape/history/context | 信息丰富的 RL state | MDP reward | 重复 feature/state acquisition；不是 paired query/skip utility | 多次切换或调度候选 DE 状态 | proof-of-principle 的 DE/CEC 条件；不证明外部转移 |
| DACBench (2021) | 动态设置 hyperparameter 或 algorithm component | 每个配置时点的目标算法状态 | benchmark-defined state 与 instance context | 累积 reward／solution quality | state 查询与配置频率属于 benchmark 定义；不估计独立 query acquisition utility | 重复参数或组件控制 | 含 CMA-ES 与 ModEA 等 DAC benchmark；不是 query/skip 评价 |
| Decision-before-Feature | 执行或跳过固定 descriptor query | query descriptors 尚不存在时 | 算法无关 behavior 与允许的连续预算上下文 | matched-state、cost-adjusted query utility | 同时区分 query FE、时间与内存口径；只扣除非重复成本 | query 后由独立 Selection Reference 选择 native continuation 或 population-transfer initialization | BBOB-train function-family OOF 选模/阈值；BBOB-validation 与外部套件仅冻结评价，当前尚无泛化结论 |

## E. 精确研究缺口与引用边界

在本记录实际核读的文献范围内，尚未识别到同时满足下列条件的方法：

1. 决策对象是是否执行一个预先固定、独立采样的 landscape-descriptor query，而不是直接选择算法、算子、参数或问题类别；
2. 决策发生在 query descriptors 生成之前，输入仅含原生优化历史中的算法无关行为与允许的预算上下文；
3. 监督标签来自共享状态上的匹配 query/skip 路径，并将下游性能差与非重复的 FE、时间和内存资源量放入明确效用口径；
4. query gate 与下游 portfolio action selector 分离，并显式区分 native continuation 与 population-transfer initialization；
5. 选模和阈值只使用 BBOB-training function families，held-out BBOB、CEC 与工程问题不反馈到 preprocessing、特征选择、模型或阈值。

因此可使用的表述是：**“Among the literature reviewed here, we did not identify a method that makes the same pre-query, cost-adjusted information-acquisition decision.”** 这是一项由当前核读语料限定的 gap 陈述，不是绝对首创声明。文献只支持问题的合理性、邻近方法的任务边界以及冻结评价设计；它们不支持将 RQ1--RQ5、BBOB-validation、CEC2017、CEC2022 或工程问题结果写成已经验证，也不支持使用任何已撤回实验数值。

## F. 与当前 BibTeX 的题名和 DOI 对照

以下 13 篇核心／邻近文献均与 `docs/40_manuscript/references.bib` 中对应条目的题名和 DOI 一致：

- `jankovicTrajectorybasedAlgorithmSelection2022` — `10.1109/CEC55065.2022.9870222`
- `kostovskaPerrunAlgorithmSelection2022` — `10.1007/978-3-031-14714-2_4`
- `cenikjDynamoRepTrajectoryBasedPopulation2023` — `10.1145/3583131.3590401`
- `haywardDeterminingMetaheuristicSimilarity2025` — `10.1109/TEVC.2023.3346672`
- `mbassoHowMetaheuristicsExploit2026` — `10.1007/s12065-026-01190-7`
- `kerschkeAutomatedAlgorithmSelection2019` — `10.1162/evco_a_00236`
- `pragerPflaccoFeaturebasedLandscape2024` — `10.1162/evco_a_00341`
- `renauUtilityProbingTrajectories2024` — `10.1007/978-3-031-56852-7_7`
- `renauAlgorithmSelectionProbing2025b` — `10.1007/978-3-031-90062-4_28`
- `blomInfluenceFeatureComputation2026` — `10.48550/arXiv.2605.04954`
- `guoDeepReinforcementLearning2024b` — `10.1109/TSMC.2024.3374889`
- `eimerDACBenchBenchmarkLibrary2021` — `10.24963/ijcai.2021/230`
- `mersmannExploratoryLandscapeAnalysis2011` — `10.1145/2001576.2001690`

Renau and Hart (2025) 已按 Zotero 全文证据纳入正文和 BibTeX；其用途仅限 probing-trajectory 分类器与问题划分敏感性，不承担本文 RQ1--RQ5 的结果证据。

## G. 本轮新增的 Zotero 全文

### G.1 Renau et al. (2020)

- 题名：*Exploratory Landscape Analysis Is Strongly Sensitive to the Sampling Strategy*
- 出处／年份：*Parallel Problem Solving from Nature -- PPSN XVI*，2020，pp. 139--153
- DOI：`10.1007/978-3-030-58115-2_10`
- Zotero parent key：`B2FSDYT3`
- BibTeX key：`renauExploratoryLandscapeAnalysis2020`
- 核读范围：Zotero 索引 PDF 全文；题名、页码和 DOI 经 Crossref 核对。
- 可支撑：ELA 估计依赖采样策略；训练与部署阶段使用不同采样设计会影响下游预测。
- 不可支撑：其 5D BBOB 分类结果不能证明本文 query utility、成本效益或跨套件泛化。
- 正文用途：Related Work 中说明采样设计是表示定义的一部分。

### G.2 Hayward and Engelbrecht (2026)

- 题名：*Survey and Analysis of Metaheuristic Search Behavior Characterization: A Case Study on Particle Swarm Optimization Variants*
- 出处／年份：*Swarm Intelligence*，2026，20(1): 1--33
- DOI：`10.1007/s11721-025-00254-1`
- Zotero parent key：`YH9IGJFD`
- BibTeX key：`haywardSurveyAnalysisMetaheuristic2026`
- 核读范围：Zotero 索引 PDF 全文；正式期刊元数据经 Crossref 核对。
- 可支撑：行为指标需同时考虑敏感性、冗余性、单次运行可计算性和部署时信息边界，不能因可计算就默认有辨识力。
- 不可支撑：PSO variants 的案例不能验证本文冻结的五类行为输入，也不能证明这些输入跨 DE、PSO、CMA-ES 和 SHADE 完全算法无关。
- 正文用途：Related Work 中限定行为指标选择与解释。

### G.3 Janković and Doerr (2019), Adaptive Landscape Analysis

- 题名：*Adaptive Landscape Analysis*
- 出处／年份：*Proceedings of the Genetic and Evolutionary Computation Conference Companion*，2019，pp. 2032--2035
- DOI：`10.1145/3319619.3326905`
- Zotero parent／attachment key：`KISBBP7P`／`8ZJH65KR`
- BibTeX key：`jankovicAdaptiveLandscapeAnalysis2019a`
- 核读范围：Zotero PDF 全文，共 4 页；核读 local-feature sampling、CMA-ES/BBOB 设置、feature evolution 与 future work。
- 可支撑：该文在 CMA-ES 达到预定 target levels 时，从当前分布额外抽取 2,000 个点并计算 56 个不需要进一步评价的 ELA descriptors，以初步研究局部 feature values 如何随搜索推进而变化。
- 不可支撑：文中所谓 cheap features 不表示 2,000 点样本没有 FE 成本；该研究未在采样前决定是否获取 descriptors，也未把 descriptors 与 selector performance、paired query/skip utility 或本文外部泛化联系起来。
- 正文用途：Related Work 中区分“运行中重复观察 local ELA”与“descriptor 尚不存在时决定是否执行 query”。

### G.4 Pei et al. (2025), Adaptive Operator Selection Survey

- 题名：*Adaptive Operator Selection for Meta-Heuristics: A Survey*
- 出处／年份：*IEEE Transactions on Artificial Intelligence*，2025，6(8): 1991--2012
- DOI：`10.1109/TAI.2025.3545792`
- Zotero parent／attachment key：`IKLF35F4`／`SGFLAAM6`
- BibTeX key：`peiAdaptiveOperatorSelection2025`
- 核读范围：Zotero PDF 全文；核读 AOS 任务定义、stateless/state-based taxonomy、credit assignment、state features 与 operator-selection action。
- 可支撑：AOS 的在线动作是在 metaheuristic 内分配 search operators；stateless 方法依赖历史 reward/credit，state-based 方法可使用 solution、problem 与 optimization-process features。
- 不可支撑：本文的 query gate 属于 AOS；AOS survey 的 operator performance 能验证 query acquisition utility；本文应改为在线 credit assignment。
- 正文用途：Related Work 中精确区分 search-operator action 与 information-acquisition action。

### G.5 Petelin and Cenikj (2025), Benchmarking Pitfalls

- 题名：*The Pitfalls of Benchmarking in Algorithm Selection: What We Are Getting Wrong*
- 出处／年份：*Proceedings of the Genetic and Evolutionary Computation Conference*，2025，pp. 1181--1189
- DOI：`10.1145/3712256.3726336`
- Zotero 正式条目 key：`MKAGUF56`；核读附件来自同文预印本 parent／attachment `NJTR6BEZ`／`JWD2GPPY`；会议名与页码由 DOI 的 Crossref 正式元数据交叉核对
- BibTeX key：`petelinPitfallsBenchmarkingAlgorithm2025`
- 核读范围：Zotero PDF 全文，共 9 页；核读 leave-instance-out/leave-problem-out 对比、非信息特征示例、scale-sensitive target 与 scale-free selection metric。
- 可支撑：共享函数的 instance-level 划分可能奖励问题类别识别而非 unseen-problem algorithm ranking；scale-sensitive performance-regression error 可能改善却不提高实际 selection ranking。
- 不可支撑：本文 function-family split 或 statewise-normalized utility 已经有效；严格 split 本身能保证外部泛化；该文结果可替代本文 RQ2/RQ4 评价。
- 正文用途：Related Work 与 Experimental Setup 中说明 function-family partition、statewise action-loss normalization 和 utility-based model selection 的动机，同时保留这些设计仍需正式评价的边界。

### G.6 Mersmann et al. (2011), Exploratory Landscape Analysis

- 题名：*Exploratory Landscape Analysis*
- 出处／年份：*Proceedings of the 13th Annual Conference on Genetic and Evolutionary Computation*，2011，pp. 829--836
- DOI：`10.1145/2001576.2001690`
- Zotero parent／attachment key：`D5PQYQTL`／`VGAV4KEV`
- BibTeX key：`mersmannExploratoryLandscapeAnalysis2011`
- 核读范围：Zotero 正式条目的题录与摘要；作者、会议、页码和 DOI 由 Zotero BibTeX 导出核对。附件未提供可用的 Zotero 索引全文，因此未作全文层面的扩展主张。
- 可支撑：ELA 汇集用于认识未知优化问题性质的技术；该文提出以相对低成本、计算机生成的底层特征联系问题表征与自动算法选择。
- 不可支撑：所谓 relatively cheap 表示 query 没有 FE 或计算成本；该文预先决定是否获取 descriptors；这些特征能预测本文的 statewise query utility；本文的 held-out BBOB 或外部泛化已经成立。
- 正文用途：Related Work 首段作为 ELA 原始概念出处；后续 `pflacco`、采样敏感性与 ELA-based AAS 文献分别承担实现、表示条件和下游应用主张。
