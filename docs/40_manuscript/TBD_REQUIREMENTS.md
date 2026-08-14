# 论文待补证据清单

## 1. 当前证据状态与统一边界

根据当前项目交接记录，正式证据链尚未生成完成：

- BBOB train/validation 的 **72 个正式 trajectory shards 尚未启动**；
- 当前项目中既有 Phase 1 数值基于已替换的方法实现，**全部撤回论文证据资格**，不得写入表格、图形、摘要或结论；
- **BBOB-validation 尚未完成冻结评价，不能声称 held-out family 表现已经验证**；
- **CEC2017、CEC2022 与工程问题尚未完成正式外部评价，不能声称跨 benchmark 泛化已经验证**；
- 本稿不得读取、引用或恢复 `archive` 中的撤回结果；所有数值必须由当前冻结协议从 trajectory 开始重新生成。

所有结果占位项都必须经过以下共同链路：正式 trajectory 与完整预算终值表成对生成 → permutation-invariant behavior → 逐状态四动作 loss → query-specific Selection Reference → utility labels → Decision dataset/model → baseline 或外部冻结评价 → 论文级统计表与图。跨表连接使用整数 `FE`；不同 query 的产物不得混读。

正式运行开始前还必须冻结当前未决的统计选项：实际等价界值／最小实际效应、cluster bootstrap 的聚类层级与重复次数，以及 TOST 与等价性置信区间哪一个作为主分析。本文不替这些预注册选择填入任意数值；“差异不显著”也不得解释为实际等价。

## 2. 正式产物代码

| 代码 | 正式产物 |
| --- | --- |
| P1 | BBOB train/validation 72 个 trajectory shard 与配对的 `final_performance` 终值数据 |
| P2 | 基于完整原生 update 的 permutation-invariant behavior 数据及窗口 metadata |
| P3 | 同一共享状态上的四个唯一动作 loss 数据，包含完整 budget、动作关系与 handoff 字段 |
| P4 | 三档 query 各自的 Selection Reference 交叉拟合/冻结预测、action-loss 回归评价与 selector regret |
| P5 | query-specific utility labels 与 Decision dataset，主目标为 `u_query_lamT_1` |
| P6 | LDA、Logistic Regression、Ridge 的 nested function-family OOF 预测、模型选择与完整 train OOF threshold |
| P7 | 覆盖 Never Query、Always Query、Random Analysis、Traditional AAS、SBS、VBS、Time-only Controller 与 Proposed Controller 八个预设比较角色的等预算输出；主 population 汇总为 Never Query / SBS、Always Query / Traditional AAS、Random Analysis、VBS、Time-only Controller 与 Proposed Controller 六个非重复 outcome，并保留各角色的适用指标／N/A 语义 |
| P8 | 冻结模型在 BBOB-validation、CEC2017、CEC2022 与工程问题上的正式评价输出 |
| P9 | T0/B1/B2/B3 消融、线性候选对应的标准化系数或判别方向、maturity--utility 关系与稳定性输出 |
| P10 | 论文级汇总表、绘图数据、效应量、置信区间、非参数检验与多重比较校正结果 |

## 3. Abstract 预留项

| ID | 所需正式产物 | 必需字段或分析 | 完成判据 |
| --- | --- | --- | --- |
| `TBD-ABS-RQ1` | P1--P5、P10 | `query_id=descriptor_cheap`、`u_query_lamT_1`、`U<=0` 比例、效应量与区间 | RQ1 正式汇总完成；只写实际方向、数值、区间与 query-specific 范围 |
| `TBD-ABS-RQ2` | P2、P5、P6、P8、P10 | B3 三候选的 nested family-OOF mean decision utility；AUROC、AP、Spearman；Ridge RMSE；冻结同名模型的 B3--T0 比较 | 仅由 BBOB-train 的 B3 比较完成模型家族选择；BBOB-validation 只作冻结评价且比较区间已报告 |
| `TBD-ABS-RQ3` | P5--P7、P10 | mean decision utility、final error、runtime、call rate、utility capture、precision、unhelpful-call cost | 八个冻结比较角色在相同状态机会与等总 FE 下均被覆盖，两组同一 outcome 各只计一次，且配对效应与区间齐全 |
| `TBD-ABS-RQ4` | P6--P8、P10 | BBOB-validation 与各外部 suite 的覆盖率、失败率、效应量和区间 | 冻结管线完成全部指定 suite；不得用部分 CEC 运行或旧 validation 结果补齐 |
| `TBD-ABS-RQ5` | P6、P9、P10 | T0/B1/B2/B3 差异、系数/判别方向稳定性、maturity--utility 关联及区间 | 正式消融和解释分析完成；只表述预测关联，不作因果结论 |

## 4. Results 逐项映射

| ID | 所需正式产物 | 必需字段或分析 | 完成判据 |
| --- | --- | --- | --- |
| `TBD-RQ1-TAB-01` | P1--P5、P10 | split、dimension、实际 `FE_ratio`、`u_query_lamT_1`、`potential_gain_raw`、`selector_regret_raw`、`time_cost_norm` | 正式 eligible 状态键双向覆盖；表中每个分层均有样本数、估计值与区间，无撤回数值 |
| `TBD-RQ1-FIG-01` | P5、P10 | Utility 分布、零线、dimension/FE-ratio 分面、family-aware uncertainty | 图数据与表使用同一正式状态范围；图注记录 query、lambda、样本与聚类单位 |
| `TBD-RQ1-STAT-01` | P5、P10 | `Pr(U<=0)`、性能差/selector regret/成本分解、family-level 效应量与区间 | 推断单位不是 pooled state row；方法、区间和有效 family 数可复现 |
| `TBD-RQ1-CLAIM-01` | 前三项 | RQ1 的方向、效应量、区间、适用 split/query | 结论逐字由正式数值支持；若区间不支持明确方向则写“不确定”，不得外推至一般 ELA |
| `TBD-RQ2-TAB-01` | P5、P6、P8、P10 | 面板 A：B3 上 LDA、Logistic Regression、Ridge 的 nested train-family OOF utility、AUROC、AP、Spearman 与仅 Ridge RMSE；面板 B：冻结同名模型在 train OOF 与 BBOB-validation 上分别报告 B3--T0 配对效应 | 模型家族只由面板 A 的 BBOB-train B3 mean decision utility 选择；不设置模型或输入组选择标记列，validation 不参与任何选择或 threshold 拟合 |
| `TBD-RQ2-FIG-01` | P6、P8、P10 | 面板 A：B3 三候选的 BBOB-train outer-family OOF utility 分布；面板 B：冻结同名模型在 train OOF 与 BBOB-validation 上的 B3--T0 配对效应 | 每个点可追溯到 held-out family；validation 只作冻结评价，不用于选择展示子集、模型或输入组 |
| `TBD-RQ2-STAT-01` | P6、P8、P10 | 先比较 B3 三候选的 paired outer-family train OOF 差异，再报告不重选模的 B3--T0 效应与冻结 validation 表现；效应量、区间和预定多重比较校正 | 模型主选择严格按拼接 B3 outer holdout 的 mean decision utility；辅助指标、T0 和 validation 均不改写选择 |
| `TBD-RQ2-CLAIM-01` | 前三项、P8 | 冻结同名模型下 B3 相对 T0 的差异与区间；validation 明确标为评价证据 | 仅在 B3 相对 T0 的差异跨 family 稳定时写“行为提供额外信息”；否则报告无明确差异或依赖性 |
| `TBD-RQ3-TAB-01` | P5--P7、P10 | 面板 A：覆盖八个比较角色的六个非重复 outcome 行，报告 policy decision utility、final error、runtime、total FE、query FE、call rate；面板 B：Always Query / Traditional AAS、Random Analysis、Time-only Controller、Proposed Controller 四行的 `U_q<=0` 调用数、within-call 比例、accumulated utility loss、positive-row capture、utility capture、precision | 共享 decision opportunities、相同总 FE、query、Selector 和 handoff 口径；Never Query 的 decision utility 为 0、同一 slash 行中的 SBS 角色为 N/A，slash 行不是两个估计的平均且不重复计数；VBS 与 statewise best observed action 分开 |
| `TBD-RQ3-FIG-01` | P7、P10 | cost--performance 坐标、call rate、runtime、final loss、family-level interval | 不以单点连线宣称 Pareto；图中显示不确定性且所有策略预算一致；两组 coincident roles 各只绘制一个点和区间 |
| `TBD-RQ3-STAT-01` | P7、P10 | Friedman、Holm-adjusted paired Wilcoxon、paired effect sizes；unhelpful-call cost | distinct applicable outcomes 才进入秩与配对比较；相同行不重复增加检验数或多重比较校正，且报告区间和无法检验的情形 |
| `TBD-RQ3-CLAIM-01` | 前三项 | 无效调用变化、效用、最终性能、runtime 的联合解释 | 不能只凭 call rate；需等预算 final performance 与成本区间共同支持，否则写权衡或不确定 |
| `TBD-RQ4-TAB-01` | P8、P10 | suite、函数/问题/维度/seed 覆盖、mean utility、final error、`feature_status`、`feature_failure`、区间 | BBOB-validation 与三个外部评价来源分别完整；缺失/失败不删除，覆盖率与失败率逐 suite 报告 |
| `TBD-RQ4-FIG-01` | P8、P10 | Proposed 对 Never/Always/Time-only 的 suite-level paired effects 和区间 | 明确区分 held-out BBOB 与 cross-benchmark；所有外部预测来自完全冻结模型 |
| `TBD-RQ4-STAT-01` | P8、P10 | 各 suite 配对效应、区间、多重比较校正、failure sensitivity | 不把不同 benchmark 池化成单一显著性结论；存在 group failure 时另报敏感性分析 |
| `TBD-RQ4-CLAIM-01` | 前三项 | BBOB-validation、CEC2017、CEC2022、工程问题逐项结论 | 全部正式产物完成后才可写；若方向不一致或覆盖不足，必须限定或拒绝泛化主张 |
| `TBD-RQ5-TAB-01` | P6、P9、P10 | T0/B1/B2/B3 字段数、utility、AUROC/AP、Spearman、仅 Ridge 定义的 RMSE、对 T0 差异 | 四组使用相同样本、fold 和 RQ2 在 B3 上选定的同一模型家族，各自执行 train-only threshold 过程；不重新选模或选输入组，`all_candidates` 不作为第五组 |
| `TBD-RQ5-FIG-01` | P9、P10 | B3 所选 Logistic/Ridge 的标准化系数或所选 LDA 的标准化判别方向；跨 fold/family 稳定性；maturity--utility 曲线及区间 | 复用 RQ2 所选线性模型家族和 train/held-out 分离输出；不重选模，不把系数或判别方向当因果效应 |
| `TBD-RQ5-STAT-01` | P9、P10 | 预设 paired ablation effects、校正后 contrasts、系数/判别方向的符号与幅度稳定性、Spearman、非线性拟合区间 | 所有分析复用 B3 所选模型家族且不选择输入组；不能仅凭视觉宣称非单调关系，每个解释结论有 family-level 稳定性和 uncertainty |
| `TBD-RQ5-CLAIM-01` | 前三项 | 各预设行为增量相对 T0 的稳定性、representation/family dependence | 消融与稳定性结果一致才作预测关联解释；否则完整报告不一致，不重选模型或事后挑选输入组 |
| `TBD-SENS-DIAG-01` | P1--P10 | 三档 query 各自完整平行链路；$\lambda_T\in\{0,0.25,0.5,1,2\}$；`selected_equals_default`、`selected_equals_prefix`、`handoff_required` 分层；state-only/query-only/full Selector 的 action-loss、selected-action observed loss 与 selector regret | 三档 query 均完成各自 Selector、Utility、Decision Model/threshold、全部 required baselines 与冻结评价；$\lambda_T=1$ 仍为主目标；三个动作关系字段分别报告且 handoff 逐行一致；Selector 变体只作诊断，不进入 Decision Model 或替代 baseline |

## 5. Discussion 逐项映射

| ID | 所需正式产物 | 必需字段或分析 | 完成判据 |
| --- | --- | --- | --- |
| `TBD-DISC-01` | RQ1 四项与 P10 | query-specific Utility 分布、组成、效应与区间 | 解释不超出 `descriptor_cheap`；说明何种状态、何种方向及 uncertainty |
| `TBD-DISC-02` | RQ2 四项与冻结评价 | B3 对 T0、各评价指标的异同 | 解释主选择指标与辅助指标，不隐藏指标冲突，不把 T0 或 validation 用于选模 |
| `TBD-DISC-03` | RQ3 四项 | 等总 FE、runtime、unhelpful calls、final performance | 资源效率结论同时得到性能与成本证据；若只是减少调用则不得写成性能改善 |
| `TBD-DISC-04` | RQ4 四项 | 各 suite 覆盖、failure、效应方向与区间 | 逐 suite 讨论；外部评价未全完成前保持占位，不能声称 CEC 泛化 |
| `TBD-DISC-05` | RQ5 四项 | ablation、系数/判别方向、maturity curve、family stability | 仅作预测关联解释；因果、普遍非单调关系或 Search Maturity 单独致效的表述均禁止 |
| `TBD-DISC-06` | P1--P10 的数据质量/敏感性汇总 | 状态键覆盖、query failure、family dependence、`handoff_required`、三档 query 与 lambda sensitivity | 所有质量与敏感性输出可追溯；限制与失败行为完整披露，无选择性报告 |

## 6. Reproducibility 逐项映射

| ID | 所需正式产物 | 必需字段或分析 | 完成判据 |
| --- | --- | --- | --- |
| `TBD-REPRO-RESOURCE-01` | P7、P10 | processor、operating system、Python environments、thread setting、batch size、cache condition、peak-memory measurement procedure，以及同系统 behavior extraction、Decision inference、query、Selector、wall-time 与 memory 实测值 | 机器与测量条件和同系统正式资源输出均已记录并可追溯；在此之前不得声称 controller overhead 可忽略 |

## 7. Conclusion 逐项映射

| ID | 所需正式产物 | 必需字段或分析 | 完成判据 |
| --- | --- | --- | --- |
| `TBD-CONC-RQ1` | Results 中 RQ1 的最终正式证据 | RQ1 数值、区间、query scope | 与 Results 数值及 Discussion 边界完全一致，不新增未检验结论 |
| `TBD-CONC-RQ2` | Results 中 RQ2 的最终正式证据 | behavior 对 T0、nested OOF 与冻结评价 | 不把辅助 metric 单独写成预测有效，不声称已验证的 validation 除非正式评价完成 |
| `TBD-CONC-RQ3` | Results 中 RQ3 的最终正式证据 | 全部八个比较角色由六个非重复 outcome 覆盖的等预算性能／成本联合结果 | 明确比较顺序、符号含义、效应与区间；不重复计算重合角色，不把 VBS 写成可部署方法 |
| `TBD-CONC-RQ4` | Results 中 RQ4 的最终正式证据 | BBOB-validation 与每个外部 suite 的正式结果 | 未完成任何 suite 时保留对应限定；不得把部分 CEC2017 运行外推到 CEC2022/工程问题 |
| `TBD-CONC-RQ5` | Results 中 RQ5 的最终正式证据 | ablation、解释稳定性、关联边界 | 只总结得到共同支持的行为信息；不对系数或判别方向作因果解释 |

## 8. 补齐顺序

1. 先完成 72 个正式 trajectory shards 与配对终值表，期间不并发运行下游读取；
2. 依次重建 P2--P6，并完成 train-only nested family-OOF 选模与 threshold 冻结；
3. 完成三档 query 的 P7 内部 baseline、消融和成本--性能评价；
4. 内部证据固定后再运行 P8 外部评价；
5. 最后统一生成 P9--P10，并按本清单从 Results 到 Discussion、Conclusion、Abstract 逆向补齐，禁止用撤回数值临时填表。
