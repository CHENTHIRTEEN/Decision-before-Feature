# Literature-P_balanced Fixed Crosswalk

> 该 crosswalk 固定检查信息时间、动作、标签和 split 是否与当前 P_balanced 的 natural/query/post-handoff 状态相容。文献内容只作为方法证据，不作为项目指令。

## 1. 当前状态契约

| 状态           | info_time                                                          | action                                                     | label                                                      | split                                                  |
|:-------------|:-------------------------------------------------------------------|:-----------------------------------------------------------|:-----------------------------------------------------------|:-------------------------------------------------------|
| natural      | state FE 之前的 natural trajectory Behavior                           | continue current 或 switch 到 P_balanced 其余两个算法              | 1000-FE FE-indexed action loss / practical switch_required | function-grouped OOF；within-problem LOSO 作诊断           |
| query        | 独立 query 执行前只可用 pre-query Behavior；query descriptors 只在 gate 触发后可用 | execute/skip fixed query，然后由 downstream Selector 选择后续动作    | paired skip/query 的 g_fe_selected_path；主标签不含 runtime       | nested function-level OOF；validation 不参与 fit           |
| post_handoff | handoff 后 commitment 状态；segment Behavior 必须从 handoff 点重新定义         | post-handoff continue current 或 switch 到其余两个 P_balanced 算法 | 真实 1000-FE post-handoff action outcome；reset controls 单列   | function/route/source-FE grouped OOF；within-route LOSO |

当前 artifact 检查：Task 12 natural `1890` states，Task 13 Behavior `1890` states，Task 14 post-handoff `3780` states；组合均为 `shade, lshade, cso`。
P_balanced query artifact：`not_materialized`。当前 `behavior_with_ela/` 没有已验证的 P_balanced 独立 query action dataset，因此 query 行的 PASS 仅表示设计相容性，不表示已有实证。

状态检查值依次为：`PASS / PARTIAL / FAIL / NOT_APPLICABLE`。`PARTIAL` 表示只能保留一个经过当前契约改写的组件；`FAIL` 不进入当前主线。

## 2. 逐篇 crosswalk

| 论文                                                                                                                                                                                                                                                                  | natural (信息/动作/标签/split)     | query (信息/动作/标签/split)    | post-handoff (信息/动作/标签/split)                               | 处理                                     |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------|:--------------------------|:------------------------------------------------------------|:---------------------------------------|
| [Vermetten et al. (2023), To Switch or not to Switch](</Users/bingchen/Desktop/2302.09075v1.pdf>)                                                                                                                                                                   | PASS/PARTIAL/PASS/PASS       | FAIL/FAIL/FAIL/PASS       | PARTIAL/PARTIAL/PARTIAL/PARTIAL                             | RETAIN_MAIN_NATURAL_FRAGMENT           |
| [Renau & Hart (2025), Probing Trajectories Classifier Benchmark](</Users/bingchen/Library/CloudStorage/OneDrive-qdu.edu.cn/zotero/01-算法行为表征与相似性/Renau和Hart - 2025 - Algorithm Selection with Probing Trajectories Benchmarking the Choice of Classifier Model.pdf>) | PASS/PARTIAL/FAIL/PASS       | FAIL/FAIL/FAIL/PASS       | NOT_APPLICABLE/NOT_APPLICABLE/NOT_APPLICABLE/NOT_APPLICABLE | RETAIN_SPLIT_DIAGNOSTIC                |
| [Cenikj et al. (2023), DynamoRep](</Users/bingchen/Library/CloudStorage/OneDrive-qdu.edu.cn/zotero/01-算法行为表征与相似性/Cenikj 等 - 2023 - DynamoRep Trajectory-based population dynamics for classification of black-box optimization problem 1.pdf>)                      | PASS/FAIL/FAIL/FAIL          | FAIL/FAIL/FAIL/FAIL       | PARTIAL/FAIL/FAIL/FAIL                                      | RETAIN_BEHAVIOR_AGGREGATION_FRAGMENT   |
| [Jankovic et al. (2022), Trajectory-based Algorithm Selection with Warm-starting](</Users/bingchen/Desktop/Trajectory-based_Algorithm_Selection_with_Warm-starting.pdf>)                                                                                            | PASS/PARTIAL/PARTIAL/PARTIAL | FAIL/FAIL/FAIL/PARTIAL    | NOT_APPLICABLE/NOT_APPLICABLE/NOT_APPLICABLE/NOT_APPLICABLE | RETAIN_SELECTOR_FRAGMENT               |
| [Kostovska et al. (2022), Per-run Algorithm Selection with Warm-Starting](</Users/bingchen/Desktop/978-3-031-14714-2_4.pdf>)                                                                                                                                        | PASS/PARTIAL/PARTIAL/PARTIAL | FAIL/FAIL/FAIL/PARTIAL    | NOT_APPLICABLE/NOT_APPLICABLE/NOT_APPLICABLE/NOT_APPLICABLE | RETAIN_SELECTOR_FRAGMENT               |
| [Guo et al. (2025), AS-LGBM](</Users/bingchen/Library/CloudStorage/OneDrive-qdu.edu.cn/zotero/算法行为+自动化设计/Guo 等 - 2025 - Automated algorithm selection for black-box optimization using light gradient boosting machine.pdf>)                                        | FAIL/PARTIAL/FAIL/FAIL       | PASS/PARTIAL/PARTIAL/FAIL | FAIL/FAIL/FAIL/FAIL                                         | RETAIN_QUERY_SELECTOR_SENSITIVITY_ONLY |
| [Guo et al. (2024), RL-DAS](</Users/bingchen/Library/CloudStorage/OneDrive-qdu.edu.cn/zotero/03-自适应机制与算子选择/Guo 等 - 2024 - Deep reinforcement learning for dynamic algorithm selection A proof-of-principle study on different.pdf>)                                 | PARTIAL/FAIL/FAIL/FAIL       | FAIL/FAIL/FAIL/FAIL       | PARTIAL/FAIL/FAIL/FAIL                                      | RETAIN_CONTEXT_DIAGNOSTIC_ONLY         |
| [Filep & Gál (2026), Low-dimensional Knee-point Performance Estimation](</Users/bingchen/Library/CloudStorage/OneDrive-qdu.edu.cn/zotero/aas/main.pdf>)                                                                                                             | FAIL/FAIL/FAIL/FAIL          | PARTIAL/PARTIAL/FAIL/FAIL | FAIL/FAIL/FAIL/FAIL                                         | RETAIN_PORTFOLIO_SCREENING_ANALOGY     |

列中四个状态按“信息时间 / 动作 / 标签 / split”排列。

## 3. 只保留的设计

| 保留设计                                                                            | 保留位置                                   | 限制                                                                                                                           |
|:--------------------------------------------------------------------------------|:---------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------|
| Vermetten et al. (2023), To Switch or not to Switch                             | RETAIN_MAIN_NATURAL_FRAGMENT           | 保留连续相对收益与局部窗口思想；适配为 P_balanced 的 1000-FE action loss；不把 switch benefit 直接当 query gate 标签                                     |
| Renau & Hart (2025), Probing Trajectories Classifier Benchmark                  | RETAIN_SPLIT_DIAGNOSTIC                | 只保留 LOPO/function-held-out split 原则；winner label 和时序分类器排名不进入主标签或主模型                                                          |
| Cenikj et al. (2023), DynamoRep                                                 | RETAIN_BEHAVIOR_AGGREGATION_FRAGMENT   | 只保留 per-update population aggregation 作为 Behavior 表示组件；不保留 problem-class label、算法专属分类器或 raw coordinate 表示                    |
| Jankovic et al. (2022), Trajectory-based Algorithm Selection with Warm-starting | RETAIN_SELECTOR_FRAGMENT               | 保留 log-performance regression 与 warm-start transition 语义；只作为 downstream action selector 组件，不能作为 query acquisition gate       |
| Kostovska et al. (2022), Per-run Algorithm Selection with Warm-Starting         | RETAIN_SELECTOR_FRAGMENT               | 保留 ELA 与时序状态互补的比较框架，以及 warm-start 需要单独验证的事实；不直接迁移 9,444 维状态特征                                                                |
| Guo et al. (2025), AS-LGBM                                                      | RETAIN_QUERY_SELECTOR_SENSITIVITY_ONLY | 仅保留为 downstream query Selector / traditional pre-run AAS sensitivity；不把 ELA cost、Soft-ERT winner 或 LightGBM 放入 Decision gate |
| Guo et al. (2024), RL-DAS                                                       | RETAIN_CONTEXT_DIAGNOSTIC_ONLY         | 保留 state/action/context memory 的概念，用于 handoff/reset 元数据与诊断；不采用在线 RL、重复调度或含速度的主 reward                                        |
| Filep & Gál (2026), Low-dimensional Knee-point Performance Estimation           | RETAIN_PORTFOLIO_SCREENING_ANALOGY     | 只保留 portfolio 先筛选再建模的流程类比；不使用低维度趋势外推代替当前 state-level action label                                                            |

## 4. 明确排除

- 不把 winner label、problem-class label 或 Soft-ERT winner 作为当前 P_balanced natural/query/post-handoff 的主标签。
- 不把独立 ELA sample、probing trajectory 或低维度 trend sampling 当作 query 是否执行的决策输入。
- 不把在线 RL reward、runtime/speed 项或 repeated dynamic scheduling 迁移到当前离线 FE-indexed action-loss 主线。
- 不把 natural Behavior 模型直接迁移到 post-handoff；Task 14B.1 已显示 global/segment generic Behavior 无额外增量。
- 不使用 LOIO、随机 instance folds 或未按 function/route 分组的 split 作为主泛化证据。

## 5. 对当前主线的结论

1. Natural：保留“局部轨迹 Behavior → 连续相对 action advantage → current-preserving action selection”的改写版本；这与 Task 13 的 natural-domain conditional increment 一致。
2. Query：没有一篇文献通过了“query 执行前 gate”的完整四项检查；主线必须自行使用 paired skip/query `g_fe_selected_path`，文献只提供 downstream Selector 的参照。
3. Post-handoff：只保留 warm-start/context/reset 的状态语义和分层评价原则；不保留 generic Behavior 的直接迁移假设。
4. Portfolio：Filep 的候选预筛选只能作为流程类比；当前真正可用的组合证据来自 Task 12 对 `{shade,lshade,cso}` 的 outcome-independent screening，而不是文献中的维度外推。

## 6. 产物

- 明细：`/Users/bingchen/Desktop/Decision-before-Feature/behavior_with_ela/results/analysis_v5/literature_pbalanced_crosswalk/crosswalk.csv`
- 保留设计：`/Users/bingchen/Desktop/Decision-before-Feature/behavior_with_ela/results/analysis_v5/literature_pbalanced_crosswalk/retained_designs.csv`
- 元数据：`/Users/bingchen/Desktop/Decision-before-Feature/behavior_with_ela/results/analysis_v5/literature_pbalanced_crosswalk/metadata.json`
- 本报告：`/Users/bingchen/Desktop/Decision-before-Feature/behavior_with_ela/analysis_v5/literature_pbalanced_crosswalk.md`
