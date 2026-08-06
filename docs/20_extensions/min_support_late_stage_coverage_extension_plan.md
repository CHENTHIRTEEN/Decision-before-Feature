# min_support late-stage coverage extension 诊断计划

## 定位

这个扩展只服务于 min_support 诊断问题：

> 当前 changed_algorithm late-stage `U_ELA>0` 行没有被模型捕获，是因为训练集中缺少同阶段效用标签覆盖，还是因为 behavior feature 本身不能区分这些状态？

该扩展不属于正式 phase1 配置，不改变 `configs/phase1_bbob_train.yaml`、`configs/phase1_bbob_validation.yaml`，不重训正式模型，不修改原始 utility label。扩展后的任何模型只作为诊断模型，不能报告为正式 phase1 结果。

## late_stage_label_coverage 依据

现有输出 `results/decision/min_support/late_stage_label_coverage/late_stage_label_coverage_summary.json` 给出的关键结果：

- train changed_algorithm late stage：`0 / 180` 行满足 `u_ela_lamT_1 > 0`。
- validation changed_algorithm late stage：`27 / 400` 行满足 `u_ela_lamT_1 > 0`。
- validation 中这些行全部位于 `FE_ratio = 0.50`。
- 集中 family 为 `bbob_f005` 与 `bbob_f024`。
- 五个已评估模型在 late stage 都是 `no_train_positive_same_stage`。

因此，当前最小实验不能判断 behavior feature 是否能学习 late-stage 有用 ELA 区域；训练集在该阶段没有可学习的 `U_ELA>0` 支撑。

## 最小补样范围

新增配置文件：

`configs/min_support_bbob_train_late_stage_extension.yaml`

配置兼容现有 batch collector，使用最小笛卡尔积外壳：

| 字段 | 取值 | 理由 |
| --- | --- | --- |
| suite | `bbob` | 与 min_support 保持一致 |
| split | `min_support_bbob_train_late_stage_extension` | 与正式 train/validation 隔离 |
| functions | `[5, 24]` | validation late-stage `U_ELA>0` 集中 family |
| instances | `[1, 2]` | 覆盖目标 problem_id 所需 instance |
| dimensions | `[10, 20]` | 覆盖目标 problem_id 所需 dimension |
| seeds | `[1, 2, 3]` | 给诊断训练提供同阶段样本，并保留现有 validation seeds 4/5 作 seed holdout |
| checkpoint_ratios | `[0.50]` | 所有 validation late-stage `U_ELA>0` 行都在该 FE_ratio |
| algorithms | `[de, pso, cmaes, shade]` | 与 min_support 一致，保留 prefix_algorithm 差异 |
| FE_total_by_dimension | `10: 10000`, `20: 20000` | 与 min_support 一致 |
| population_size | `40` | 与 min_support 一致 |

科学目标最小 problem/seed cell：

| family | problem_id | FE_ratio | train seeds | 保留的现有 validation holdout seeds | 观察到的算法切换 |
| --- | --- | ---: | --- | --- | --- |
| `bbob_f005` | `bbob_f005_i01_d10` | 0.50 | 1, 2, 3 | 4, 5 | `cmaes -> de` |
| `bbob_f005` | `bbob_f005_i01_d20` | 0.50 | 1, 2, 3 | 4, 5 | `cmaes -> shade` |
| `bbob_f024` | `bbob_f024_i01_d10` | 0.50 | 1, 2, 3 | 5 | `cmaes -> shade` |
| `bbob_f024` | `bbob_f024_i02_d10` | 0.50 | 1, 2, 3 | 4 | `cmaes -> shade` |
| `bbob_f024` | `bbob_f024_i02_d20` | 0.50 | 1, 2, 3 | 5 | `cmaes -> shade` |

这 5 个 problem_id × 3 个 seed 是科学目标的最小补样单元。由于当前 collector 按 `function × instance × dimension × seed × algorithm` 生成，配置外壳还会包含 `bbob_f005_i02_d10`、`bbob_f005_i02_d20`、`bbob_f024_i01_d20` 的 seeds 1-3；这些额外行作为同 family、同 FE_ratio 的参照，不进入正式结果。

配置外壳预计产生：

- `2 functions × 2 instances × 2 dimensions × 3 seeds × 4 algorithms = 96` 个 optimizer trajectory runs。
- 每条 trajectory 只记录 `FE_ratio = 0.50` 一个 checkpoint。

如果后续新增精确 cell runner，则可把运行量降为 `5 problem_id × 3 seeds × 4 algorithms = 60` 个 optimizer trajectory runs；当前计划不新增该 runner。

## 诊断流程

1. 按扩展配置采集 trajectory，并抽取 behavior。
2. 用原始 utility label 构建协议生成扩展 split 的 utility label；标签生成仍可使用 ELA selector 参考信息，但 Decision 输入只允许 behavior feature。
3. 构造诊断训练集副本：`min_support_bbob_train` 原始 labels 加上 `min_support_bbob_train_late_stage_extension` labels。原始 labels 文件不覆盖。
4. 用已有模型族和既有训练协议训练诊断模型副本；不新增模型族，不重训正式模型。
5. threshold 只从诊断训练集 train predictions 计算：zero、train_utility、stage_train_utility。
6. 在现有 validation 中分开评估：
   - changed_algorithm validation rows；
   - same_algorithm reference rows；
   - target holdout seeds；
   - 非目标 validation rows。
7. 所有输出按 `family`、`dimension`、`FE_ratio`、`problem_id`、`seed` 分层，并保留 `selected_algorithm != default_algorithm` 与 `selected_algorithm == default_algorithm` 分组。

建议输出目录：

- `results/phase1/min_support_bbob_train_late_stage_extension/`
- `results/behavior/min_support_bbob_train_late_stage_extension/`
- `results/utility_labels/min_support_bbob_train_late_stage_extension/utility_labels.parquet`
- `results/decision/min_support/late_stage_coverage_extension/`

## 判断标准

若加入扩展 labels 后，诊断模型在 train extension 中能给 `U_ELA>0` 行更高 score，并在 holdout seeds 上提高 `U_ELA` 捕获率，同时误调用成本没有同步放大，则当前主要问题是 min_support 训练覆盖不足。

若 train extension 中 `U_ELA>0` 行仍低分，说明当前 behavior feature 或模型族对 late-stage 有用 ELA 状态的表达不足。

若 train extension 能被高分捕获，但 holdout seeds 或非目标 validation 仍低捕获，则问题更像是同 family 内行为分布迁移，需要扩大 problem_id 或 seed 覆盖，而不是简单调阈值。

若只有降低 stage threshold 才能捕获，但 changed_algorithm validation 上误调用成本明显增加，则问题主要来自阈值校准与 `U_ELA>0` 稀疏性之间的权衡。

## 协议边界

- 不使用 ELA features 作为 Decision 输入。
- 不使用 function id、dimension、algorithm id 作为 Decision 输入；这些字段只用于分层输出。
- 不改变正式 phase1 配置。
- 不修改原始 utility label。
- 不引入 pytest、测试目录、JSON Schema、hash、checksum 或 digest 机制。
- 不把该扩展结果写成正式 Function Family Split 泛化证据；它只验证 late-stage coverage 缺口是否会限制 behavior-only Decision Model 学习。
