# Phase1 历史结果摘要（已撤回正式证据资格）

> 2026-08-14 统一撤回说明：本页数值来自 population-only 重建式 continuation、把 population 行号当作跨代个体身份的旧 Behavior、problem 级静态标签与 nearest performance bucket 的旧 Selection Reference、max-scale/线性相对时间 Utility（历史），以及被笼统称为 ELA 的 16 维自定义描述符。因此旧 behavior、landscape features、selection reference、utility labels、Decision Model 与下游评价不得作为论文证据。当前主 `descriptor_cheap_invariant` 已在统一 median/IQR preprocessing 后删除恒为 0/1 的 `descriptor_y_median`、`descriptor_y_iqr`，为 14 维；该构念修正不能追溯性修复本页数值。

> 当前模型协议已另行é¢åæå®：活动候选只包括 LDA、Logistic Regression 与 Ridge，按 BBOB-train 完整 nested function OOF 的 run-level first-trigger 主功效选择（方案 A 为 `G_FE`；主标签为 `G_FE`）；`oof_utility_first_trigger` threshold、matched-rate Random calibration 与完整 train OOF score 都必须来自 fold-specific SBS/Selectors/Utility 上游链。下述撤回数值只保留历史记录，不能作为候选缩减、模型选择、阈值或“分类边界”解释的经验依据。

旧 `results/ela/`、旧 `u_ela_*` / `need_ela_*` 标签和历史模型缺少 `query_id`、`query_protocol`、`sample_design_id` 与固定 feature 列信息。它们不是活动数据契约的一部分；新读取入口必须明确失败，不提供兼容层。

当前有效的离线标签与最终评价可使用 `benchmark_reference_value` 与 known-optimum gap 口径，但这些字段只允许留在离线标签和评价层，不能进入 Behavior、Selection Reference 输入或 Decision X。

## 1. 文档定位

本文仅保留截至 2026-08-11 已生成结果的撤回记录，用于定位受影响范围，不表示当前有效性能评价。当前没有一项正式内部或外部结果。

本文只记录已撤回产物的历史范围和失效原因，不陈述任何当前有效性能结论，也不把内部 validation 历史数值扩展为 CEC2017、CEC2022 或工程问题结论。

## 2. 数据范围

| 项目 | BBOB train | BBOB validation |
| --- | ---: | ---: |
| function IDs | 18 | 6 |
| dimensions | 10, 20, 40 | 10, 20, 40 |
| instances | 1, 2, 3 | 1, 2, 3 |
| optimizer seeds | 1–30 | 1–30 |
| algorithms | DE, PSO, CMA-ES, SHADE | DE, PSO, CMA-ES, SHADE |
| checkpoints per run | 10 | 10 |
| Decision rows | 194,400 | 64,800 |
| `U_ELA > 0` rows | 11,921 | 3,745 |
| `U_ELA > 0` rate | 6.1322% | 5.7793% |

历史模型曾使用 `primary_with_maturity` 的 23 个 `bf_*` 行为字段，但其中 7 个字段依赖跨 checkpoint 的行号对应关系，现已退出活动列契约。当前列契约为 34 个唯一输出、31 个正式输入与 3 个诊断字段，六组 `T0/B1/B2/B2+Motion/B2+Maturity/B3=1/19/25/28/28/31`；`primary_with_maturity` 当前只解析为 B2+Maturity，`all_candidates` 才是 B3。旧 23 字段模型数值不得迁移到当前列契约下解释。

## 3. 已撤回的模型比较记录

关键结果如下：

| 口径 | 模型 | BBOB validation 结果 |
| --- | --- | ---: |
| 最低 regression RMSE | RBF-Nystroem SVR | 0.128917 |
| 最高 Spearman | Linear SVR | 0.456383 |
| 最高 average precision | Softmax Logistic | 0.176193 |
| 最佳 train-derived threshold 决策效用 | LDA | 0.00432497 |
| LDA ELA call rate | LDA | 0.0378395 |
| LDA efficacy/utility capture | LDA | 0.561033 |
| LDA 调用中 `U_ELA>0` 比例 | LDA | 0.312806 |
| 最高 top-10% efficacy/utility capture | Softmax Logistic | 0.646442 |

这些指标回答不同问题，不能只按 RMSE 选模型。历史流程曾据此选择 LDA 并解释 top-10% capture；该选择未包含 fold-specific SBS/Selectors/Utility、first-trigger reconstruction 或当前 Utility，因此已经撤回，不能迁移到活动模型选择。

## 4. 调参与消融

shrinkage=0.5 的 LDA 在同一 validation 上得到：

- ELA call rate：0.0374691；
- mean decision efficacy (辅助口径 utility)：0.0043582；
- efficacy/utility capture：0.560990；
- precision：0.315074。

相对未调 LDA 的旧效用差异很小，但该观察没有活动证据资格。当前不得保留 shrinkage 调参、预设 LDA 或把它作为敏感性候选。

特征组消融表明：

- 9-field base 行为组的模型结论弱于扩展算法无关行为组；
- 旧 `primary_with_maturity` 曾在旧 threshold 下改变 Ridge 决策效用，但不能迁移到当前 B2+Maturity；
- `all_candidates` 是当前 B3 的兼容别名，严格等于 31 个正式输入且不含 3 个诊断字段；它不是第七组；
- `bf_best_distance_fitness_corr` 仍按近似 landscape proxy 的诊断字段处理，不作为主模型证据。

## 5. Baseline 与成本—性能结果的使用边界

当前 baseline/Pareto 报告使用的是较早的 `primary_with_maturity + Ridge` controller：

- Ridge call rate：0.0266204；
- mean decision efficacy (辅助口径 utility)：0.00265025；
- efficacy/utility capture：0.343901；
- precision：0.267246；
- Always ELA mean efficacy (辅助口径 utility)：-0.0915294；
- Random Analysis (`p=0.5`) mean efficacy (辅助口径 utility) 约为 -0.0458；
- Never ELA / SBS skip reference 的相对 utility 为 0。

这些历史数值仅记录曾经进行过的 Always/Random/Ridge 比较，不再支持任何内部 validation 结论，也不预设重生成后 LDA 或 Ridge 仍是最优模型。baseline、消融和 Pareto 表必须在新 trajectory、action losses（按《最小 Action Loss 字段规范 v1》保留 canonical `action_loss`）与 utility labels 上全部重算。

## 6. 外部评价状态

CEC2017 的 10D/30D/50D、30 seeds、等总 FE 预算和动态采样已写入配置；函数集已按官方 29 题口径é¢åæå®为 F1, F3-F30。此前的 F2/F30 blocker 已在不查看 policy outcome 的前提下闭合。

当前已有的在线 CEC2017 结果只用于以下检查：

- online score 分布；
- 旧 main 与 dense decision-check frequency 的差异（仅作已撤回历史记录，不是当前动态采样比较）；
- zero 与 train-derived threshold 的触发差异；
- 少量 function/dimension 对上的 targeted opportunity 诊断。

这些运行仍使用 Ridge controller，部分报告只覆盖单个函数/维度和少量完整运行。它们已归入 preliminary 结果，不构成跨 benchmark 泛化证据。

## 7. 当前可写与不可写结论

在完整状态数据重生成前，以下旧结论也不可写入论文：

- 在 held-out BBOB function IDs（不是经典 landscape families）上，算法无关行为分数包含可用于筛选高效用 query 状态的信息；
- train-derived LDA threshold 的平均决策效用大于 Never ELA 的零基线，并显著减少 Always ELA 的无效调用；
- 撤回结果中表现较高的旧分层实际对应 `selected_equals_default=false`，不能用于判断 Query 路径是否继续 prefix。重生成后必须分别使用 `selected_equals_default`、`selected_equals_prefix`、`handoff_required` 与 `skip_switches_from_prefix`，且主结果只使用 SBS prefix。

暂时不能写：

- Decision Model 已在 CEC2017、CEC2022 或工程问题上稳定泛化；
- LDA 已完成端到端 baseline/Pareto 的最终比较；
- Search Maturity 单独导致性能改善；
- efficacy/utility magnitude 已被准确校准；多个回归模型的 validation R² 仍较弱。

## 8. 下一轮正式任务

1. 覆盖生成 BBOB train/validation trajectory shards，并通过完整 state、native-update window 与 checkpoint 一致性检查。
2. 重提取 Behavior，执行 Stage-A 两套四动作 matrices 与 FE=0 outcomes 各一次；用这些单次 outcomes 固定科学 gap、`observed_first_hit_FE`、`target_hit_observed`、`path_completed`、`endpoint_success` 与 planned/effective FE，并由 fold-specific SBS/Selectors 生成 OOF selected actions。
3. Replay planner 已有枚举能力；下一步实现 offline decision-state-to-terminal runner，物化并核对 plan，对 Skip/Query/Behavior-only 及 FE=0 policy paths 各执行三次 Stage-B timing-only replay，保存逐次 status/effective FE/timeout/completion、完成端点一致性与 instability；将 Stage-A 科学端点和 Stage-B 计时中位数组合成新 Utility 与方案 A 主标签 `G_FE`。不得用 replay outcome 改写科学端点或选择性补跑。
4. 对 LDA、Logistic Regression 与 Ridge 重新执行完整 `cv_group_id = function_id` nested OOF、first-trigger model selection、threshold/Random calibration、六组消融、baselines 与估计性统计分析。BBOB-validation 只作已见固定六函数内部评价。
5. CEC2017 已按官方 29 题口径闭合，只作已见外部开发评价；CEC2022 和工程问题须在首次 outcome 前é¢åæå® suite endpoints、constraint rule 与分析计划，才可承担前瞻确认。

当前额外 blockers 是 `cv_group_id = function_id` Selector artifact 路由、runner、物化实测 replay plan、Stage-A 共享/复用裁决、BBOB instance-aware online endpoint、cluster-balanced Selector/Decision fit、资源排期与真实 evaluator timing。12 个 mandatory milestones 的平均 prefix ratio 为 0.35；只含这些 states 时，仅跨 matrices 共享的 main cheap 为 215.709732B FE、三档为 350.202636B FE；进一步复用基础 trajectory 时为 210.992292B/345.485196B；保持当前 main producer 时主 query 为 225.144612B FE，三档当前量待枚举。event-only states 尚未计入。现 CEC2017 online evaluator 另需 11.5884B planned FE，已见 BBOB-validation 全 instances 需 5.5944B 但当前不可执行；这些只是 mandatory-only 算术情景，不是严格下界。
