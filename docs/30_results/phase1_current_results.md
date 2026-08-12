# Phase1 旧结果摘要（已撤回正式证据资格）

> 2026-08-11 方法修正：本页数值来自 population-only 重建式 continuation、把 population 行号当作跨代个体身份的旧 behavior、problem 级静态标签与 nearest performance bucket 的旧 Selection Reference，以及被笼统称为 ELA 的 16 维自定义描述符。前三项分别造成内部状态丢失、非 permutation-invariant 行为表示和在线共享状态动作标签失配；第四项造成 landscape representation 构念越界。因此旧 behavior、landscape features、selection reference、utility labels、Decision Model 与下游评价不得作为论文证据。完成 native optimizer-state trajectory、集合级 behavior、三档 query、逐状态 action losses、连续预算 selector、labels 和模型的全链路重生成后，必须用新结果替换本页。

> 当前模型协议已另行冻结：活动候选只包括 LDA、Logistic Regression 与 Ridge，按 BBOB-train nested function-family OOF decision utility 选择，并由完整 train family-OOF 分数冻结 `oof_utility` threshold。下述撤回数值只保留历史记录，不能作为候选缩减、模型选择、阈值或“分类边界”解释的经验依据。

旧 `results/ela/`、旧 `u_ela_*` / `need_ela_*` 标签和旧模型缺少 `query_id`、`query_protocol`、`sample_design_id` 与固定 feature 列信息。它们不是活动数据契约的一部分；新读取入口必须明确失败，不提供兼容层。

## 1. 文档定位

本文仅保留截至 2026-08-11 已生成结果的撤回记录，用于定位受影响范围，不表示当前有效性能评价。

本文只陈述当前数据支持的性能评价结果，不把内部 validation 结果扩展为 CEC2017、CEC2022 或工程问题结论。

## 2. 数据范围

| 项目 | BBOB train | BBOB validation |
| --- | ---: | ---: |
| function families | 18 | 6 |
| dimensions | 10, 20, 40 | 10, 20, 40 |
| instances | 1, 2, 3 | 1, 2, 3 |
| optimizer seeds | 1–30 | 1–30 |
| algorithms | DE, PSO, CMA-ES, SHADE | DE, PSO, CMA-ES, SHADE |
| checkpoints per run | 10 | 10 |
| Decision rows | 194,400 | 64,800 |
| `U_ELA > 0` rows | 11,921 | 3,745 |
| `U_ELA > 0` rate | 6.1322% | 5.7793% |

旧模型曾使用 `primary_with_maturity` 的 23 个 `bf_*` 行为字段，但其中 7 个字段依赖跨 checkpoint 的行号对应关系，现已退出活动列契约。新实现仍保留 23 个主输入字段，但全部跨窗口计算改为 permutation-invariant 的种群或 fitness 分布统计；本页模型数值不得迁移到新列契约下解释。

## 3. 已撤回的模型比较记录

完整模型比较表位于：

- 本机归档：`results/archive/withdrawn_20260811/decision/phase1_refined_sampling/model_benchmark_comparison/model_benchmark_comparison_report.md`

关键结果如下：

| 口径 | 模型 | BBOB validation 结果 |
| --- | --- | ---: |
| 最低 regression RMSE | RBF-Nystroem SVR | 0.128917 |
| 最高 Spearman | Linear SVR | 0.456383 |
| 最高 average precision | Softmax Logistic | 0.176193 |
| 最佳 train-derived threshold 决策效用 | LDA | 0.00432497 |
| LDA ELA call rate | LDA | 0.0378395 |
| LDA utility capture | LDA | 0.561033 |
| LDA 调用中 `U_ELA>0` 比例 | LDA | 0.312806 |
| 最高 top-10% utility capture | Softmax Logistic | 0.646442 |

这些指标回答不同问题，不能只按 RMSE 选模型。当前部署式主口径选择 LDA，是因为其阈值完全由 train split 确定，并在 held-out BBOB families 上取得最大的平均决策效用。Softmax Logistic 的 top-10% capture 更高，但对应 top-10% 平均 observed utility 仍为负，因此只适合作为固定调用预算下的排序结果。

## 4. 调参与消融

分类器调参结果位于：

- 本机归档：`results/archive/withdrawn_20260811/decision/phase1_refined_sampling/classifier_feature_engineering_tuning/classifier_feature_engineering_tuning_report.md`

shrinkage=0.5 的 LDA 在同一 validation 上得到：

- ELA call rate：0.0374691；
- mean decision utility：0.0043582；
- utility capture：0.560990；
- precision：0.315074。

相对未调 LDA 的效用增量很小。当前可将未调 LDA 作为更简洁的主模型，把 shrinkage LDA 作为敏感性结果。

特征组消融表明：

- 9-field base 行为组的模型结论弱于扩展算法无关行为组；
- `primary_with_maturity` 在 train-derived threshold 下明显改善 Ridge 决策效用；
- `all_candidates` 含诊断字段，不得直接替代主输入组；
- `bf_best_distance_fitness_corr` 仍按近似 landscape proxy 的诊断字段处理，不作为主模型证据。

## 5. Baseline 与成本—性能结果的使用边界

当前 baseline/Pareto 报告使用的是较早的 `primary_with_maturity + Ridge` controller：

- Ridge call rate：0.0266204；
- mean decision utility：0.00265025；
- utility capture：0.343901；
- precision：0.267246；
- Always ELA mean utility：-0.0915294；
- Random Analysis (`p=0.5`) mean utility 约为 -0.0458；
- Never ELA / SBS skip reference 的相对 utility 为 0。

这些旧数值仅记录曾经进行过的 Always/Random/Ridge 比较，不再支持任何内部 validation 结论，也不预设重生成后 LDA 或 Ridge 仍是最优模型。baseline、消融和 Pareto 表必须在新 trajectory、action losses 与 utility labels 上全部重算。

## 6. 外部评价状态

CEC2017 正式配置已经冻结：29 个函数、10D/30D/50D、30 seeds、等总 FE 预算和训练同口径 checkpoint ratios。

当前已有的在线 CEC2017 结果只用于以下检查：

- online score 分布；
- main 与 dense decision-check frequency 的差异；
- zero 与 train-derived threshold 的触发差异；
- 少量 function/dimension 对上的 targeted opportunity 诊断。

这些运行仍使用 Ridge controller，部分报告只覆盖单个函数/维度和少量完整运行。它们已归入 preliminary 结果，不构成跨 benchmark 泛化证据。

## 7. 当前可写与不可写结论

在完整状态数据重生成前，以下旧结论也不可写入论文：

- 在 held-out BBOB function families 上，算法无关行为分数包含可用于筛选高效用 ELA 状态的信息；
- train-derived LDA threshold 的平均决策效用大于 Never ELA 的零基线，并显著减少 Always ELA 的无效调用；
- 撤回结果中表现较高的旧分层实际对应 `selected_equals_default=false`，不能用于判断 Query 路径是否继续 prefix。重生成后必须分别使用 `selected_equals_default`、`selected_equals_prefix`、`handoff_required` 与 `skip_switches_from_prefix`，且主结果只使用 SBS prefix。

暂时不能写：

- Decision Model 已在 CEC2017、CEC2022 或工程问题上稳定泛化；
- LDA 已完成端到端 baseline/Pareto 的最终比较；
- Search Maturity 单独导致性能改善；
- utility magnitude 已被准确校准；多个回归模型的 validation R² 仍较弱。

## 8. 下一轮正式任务

1. 覆盖生成 BBOB train/validation trajectory shards，并通过完整状态与 checkpoint 一致性检查。
2. 重提取 behavior，重新生成 Selection Reference、utility labels 和 Decision dataset。
3. 对 LDA、Logistic Regression 与 Ridge 重新执行 nested family-OOF 选择、完整 train OOF threshold 冻结、baseline、feature ablation 和成本—性能评价，不预设任一候选胜出。
4. 内部链路完成后，按完整 CEC2017 配置执行外部评价，再扩展 CEC2022 和工程问题。
