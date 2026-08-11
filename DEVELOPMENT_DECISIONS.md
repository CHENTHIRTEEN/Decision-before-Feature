# Decision-before-Feature 开发前裁决

本文档记录已经确认的项目级裁决，用于解决 `AGENTS.md` 与 `docs/` 研究设计文档之间的约束冲突。

优先级规则：

1. `AGENTS.md` 是最高优先级。
2. 本文档用于解释如何在开发中落实 `AGENTS.md` 和研究文档。
3. 若 `docs/` 中早期方案与本文档冲突，开发时按本文档执行。

当前状态（2026-08-11）：优化器 continuation 已改为完整状态原生推进。此前由重建式 continuation 生成的 trajectory、utility labels、Decision dataset、模型和评价结果已撤回正式证据资格，必须从 trajectory 开始重新生成。preliminary/MVE 口径仍退出当前运行面。

---

## 1. 验证方式

裁决：

- 不建立 `tests/` 目录。
- 不引入 `pytest`、测试依赖、JSON Schema 或 schema registry。
- 使用真实小规模实验运行、数据质量检查和一致性检查验证模块。

允许：

- 轻量字段定义。
- 运行前后的数据质量检查。
- 真实 benchmark 上的最小可验证实验。

禁止：

- dry、smoke、synthetic validation。
- 与真实科学运行无关的替代工作流。
- 用测试框架替代真实实验验证。

---

## 2. 数据字段约定

裁决：

- 可以使用轻量字段定义来描述 trajectory、behavior、utility label 的列。
- 不实现 JSON Schema、schema registry 或任何强制注册系统。
- 文件名和模块名避免使用会暗示治理机制的命名。

开发口径：

- `schema` 一词只表示普通数据字段约定。
- 不用于文件身份、完整性、授权、运行许可或执行解锁。

---

## 3. Oracle 术语

裁决：

- 代码目录不使用 `oracle/` 作为主模块名。
- `Query Utility Oracle` 在开发中改称 `utility_labels` 或 `offline utility label generation`。
- `Selection Oracle` 在开发中改称 `selection_reference`。
- VBS 保留为理论上界，不作为现实可部署方法。

推荐目录：

```text
utility_labels/
selection_reference/
portfolio_reference/
```

避免目录：

```text
oracle/
selection_oracle/
oracle_generator/
```

---

## 4. 共享前缀配对续跑

裁决：

- 不使用“反事实”或 `counterfactual` 描述共享状态上的多分支完整运行。
- 使用“共享前缀配对续跑”或 `shared-prefix paired continuation`。
- 不提出因果主张。

开发口径：

```text
same complete optimizer checkpoint state
-> No-query: use the train-derived SBS/default; continue natively when default == prefix, otherwise use one population-transfer initialization
-> Run Query, same algorithm: native continuation with the query-adjusted budget
-> Run Query, changed algorithm: one population-transfer initialization
-> paired comparison
```

完整 checkpoint state 必须包括 population、fitness、generation、best-so-far、优化器内部动态量和 RNG state。不得把仅含 population/fitness 的算法重建称为同算法 continuation。

---

## 4.1 逐状态 Selection Reference

裁决：

- 不再用 problem 级静态最佳算法标签和 nearest performance bucket 生成正式 Selection Reference。
- 对每个共享状态运行唯一动作集合：`continue_current` 加其余三个 portfolio algorithm；同算法使用完整状态原生 continuation，跨算法使用一次 population transfer。
- 保存每个 `state × action` 的 observed final loss；逐状态最小值称为 `best_observed_action`，不称为 oracle。
- Selector 预测逐状态归一化 action loss，并把 `remaining_budget_ratio` 作为连续输入。
- Selector 可使用 query features、permutation-invariant algorithm-agnostic behavior 和连续剩余预算；这些 selector 输入不改变 Decision Model 的禁止输入边界。
- 正式 train Selection Reference 输出必须来自 function-family cross-fitting；validation 和外部 benchmark 只允许使用 BBOB train 拟合的最终模型。
- 原静态 bucket classifier 只可作为被替代方法诊断，不得继续生成正式 Utility 标签。

Utility 分解口径：

```text
potential_gain_raw = loss_noquery - loss_best_observed
selector_regret_raw = loss_selector - loss_best_observed
performance_gain_raw = potential_gain_raw - selector_regret_raw
```

Population transfer 的影响已经包含在各 action 的 observed loss 中，不能在主 Utility 再扣一次 `handoff cost`。Query sampling FE 已通过减少 Query 分支后续优化预算体现，也不能重复扣除；主 Utility 只额外扣除尚未进入 performance loss 的 time/memory 等成本。

对应协议见：

`docs/10_protocols/Decision-before-Feature_逐状态动作损失Selection Reference修订.md`

---

## 5. Train / Validation / Test

裁决：

- Train: BBOB, 10D / 20D / 40D。
- Validation: BBOB，按第 5.1 节正式 BBOB trajectory 协议执行。
- Test: CEC2017 / CEC2022。
- Engineering problems: 在 CEC2017 / CEC2022 之后作为外部验证或扩展。

禁止：

- 用 CEC 或 engineering problems 调参。
- 用测试函数训练 Decision Model、selection reference 或 threshold。
- 随机 function instance split。

---

## 5.1 正式 BBOB trajectory 数据采集协议

裁决：

- BBOB trajectory 采集只使用 COCO `bbob` suite。
- 不使用手写 benchmark 函数作为 BBOB 替代。
- BBOB function family 以函数编号为单位，记为 `bbob_f001` 至 `bbob_f024`。
- `problem_id` 保存具体问题粒度，例如 `bbob_f001_i01_d10`。
- `family` 保存 function family 粒度，例如 `bbob_f001`。
- 禁止按 instance、seed 或 dimension 随机拆分 train / validation。

正式 function family split：

```text
train:
1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 15, 16, 17, 18, 20, 21, 22, 23

validation:
5, 9, 13, 14, 19, 24
```

正式维度：

```text
train dimensions:
10, 20, 40

validation dimensions:
10, 20, 40
```

50D 裁决：

- 当前 COCO `bbob` suite 不支持 50D。
- 早期文档中的 BBOB 50D validation 视为历史方案，不进入主协议。
- 如需 50D / 100D 泛化，必须另设扩展实验并选择 COCO 支持的 suite，不得混入主 BBOB validation。

正式采集重复设置：

```text
instances:
1, 2, 3

optimizer seeds:
1 ... 30
```

预算口径：

```text
FE_total = 1000 * D

10D: 10000
20D: 20000
40D: 40000
```

人口规模：

```text
population_size = 40
```

理由：

- 上述预算均可被 `population_size = 40` 整除，便于保存完整 population checkpoint。
- Checkpoint 继续使用 FE ratio，不使用固定 FE 间隔。

正式 checkpoint ratios：

```text
0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60
```

裁决理由：

- preliminary 覆盖分析显示 `changed_algorithm` 且 `U_query>0` 的主要机会集中在 `FE_ratio=0.30-0.55`；该观察只用于冻结采样范围，不作为正式结果。
- `0.20` 保留为较早成熟度参照，`0.25` 和 `0.28` 用于提高 0.20 到 0.30 之间的行为与效用分辨率；这些 checkpoint 不再与任何 performance bucket 绑定。
- `0.60` 保留为机会区之后的衰减参照。
- very early checkpoints 例如 `0.005-0.15` 和 late endpoints 例如 `0.75/1.00` 不进入正式 phase1 主采样频率；如需研究 early/late 行为，应另设扩展实验。

在线测评行为采样口径：

- 在线测评中的行为采样频率定义为 `decision-check frequency`。
- 每个采样点同时是 behavior observation 点，也是 controller、Random Analysis 和 Always Query 可以触发固定 query 的决策点。
- 主在线测评使用训练 / label 同口径 checkpoint ratios：`0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60`。
- 密集采样只作为敏感性分析，使用：`0.20, 0.225, 0.25, 0.275, 0.28, 0.30, 0.325, 0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60`。
- 在尚未触发 query 时，主采样与密集采样都只观察同一个连续优化器状态；增加 checkpoint 不得重新初始化算法，也不得改变同 seed 的原生搜索轨迹。
- 密集采样仍是决策检查频率敏感性分析，因为增加检查点会增加 controller 的触发机会；这一影响来自决策机会，而不是优化器重启。
- 主结论只使用训练同口径 online sampling protocol。

正式配置文件：

```text
configs/phase1_bbob_train.yaml
configs/phase1_bbob_validation.yaml
```

输出路径：

```text
results/phase1_refined_sampling/bbob_train_trajectories.parquet
results/phase1_refined_sampling/bbob_validation_trajectories.parquet
```

---

## 5.2 正式 BBOB trajectory 分片运行策略

裁决：

- 正式 BBOB train / validation 采集不得写入单个 Parquet 文件作为主运行方式。
- 正式采集采用按 `split / function family / dimension` 分片输出。
- 单文件输出不是正式采集入口；正式运行只使用分片输出。

推荐分片路径：

```text
results/phase1_refined_sampling/bbob_train/bbob_f001/dimension_10/trajectories.parquet
results/phase1_refined_sampling/bbob_validation/bbob_f005/dimension_10/trajectories.parquet
```

分片粒度：

```text
train:
18 function families * 3 dimensions = 54 shards

validation:
6 function families * 3 dimensions = 18 shards
```

每个正式 shard 包含：

```text
3 instances * 30 optimizer seeds * 4 algorithms * 10 checkpoints = 3600 trajectory rows
```

续跑口径：

- 若目标 shard 文件已存在，默认跳过。
- 显式传入 `--overwrite` 时允许重新生成目标 shard。
- 单个 shard 失败时，只重跑该 shard。
- 不实现哈希、checksum、manifest、receipt、append-only 或执行解锁机制。
- 文件存在性只作为人工续跑便利判断，不作为数据身份或完整性证明。

正式采集前必须先运行 shard plan：

```text
uv run phase1-plan-shards --config configs/phase1_bbob_train.yaml
uv run phase1-plan-shards --config configs/phase1_bbob_validation.yaml
```

---

## 5.3 Behavior 的跨 checkpoint 对应关系

裁决：

- Decision 主输入必须对 checkpoint 内 population 行排列不变。
- 不在 trajectory 中增加 individual ID 或 ancestry；DE、PSO、SHADE 的稳定行序也不作为统一输入协议的前提。
- population 跨窗口变化使用等权经验 Wasserstein-1、centroid shift、centroid/Wasserstein coherence 和当前协方差谱集中度。
- fitness 跨窗口变化使用排序后的经验分位数，计算改善分位数比例、平均分布改善率和一维 Wasserstein 变化率。
- 所有变化量按实际 `FE_ratio_t-FE_ratio_anchor` 归一化；空间距离同时除以 `sqrt(dimension)`。
- CMA-ES 每代样本不存在稳定个体身份，因此不得使用 row-wise displacement、row-wise fitness improvement 或由其派生的方向统计。
- 旧 behavior、utility labels、Decision dataset、模型和评价结果不得与新集合特征混用。

---

## 6. Dimension 使用

裁决：

- `dimension` 必须保存为 metadata。
- `dimension` 不进入 Decision Model 输入列。
- `dimension` 仅用于 split、分组报告和 OOD 分析。

同类规则：

- `function_id`、`algorithm_id`、算法内部参数、query feature 都不进入 Decision Model。
- `algorithm` 可保存为 metadata，但不作为模型输入。

---

## 7. Utility 成本账本

裁决：

- 主协议采用等总 FE 预算。
- 固定 query 消耗的 FE 通过减少后续优化预算体现。
- 若 query FE 已经从优化预算中扣除，Utility 中不能再扣同一笔 FE 成本。
- Utility 中额外扣除 wall-time、feature computation、memory 等非 FE 成本。

必须保存：

```text
FE_total
FE_prefix
FE_query
FE_no_query_optimization
FE_query_optimization
runtime_query
runtime_selection
runtime_no_query_optimization
runtime_query_optimization
```

主标签口径：

```text
U_query = performance_gain - lambda_time * time_cost - lambda_memory * memory_cost
```

若某实验允许额外 query FE，必须另设清晰公式并单独记录：

```text
U_query = performance_gain
        - lambda_FE * extra_FE_cost
        - lambda_time * time_cost
        - lambda_memory * memory_cost
```

---

## 8. Performance 方向

裁决：

- Utility 标签主性能量使用越小越好的 loss / error / regret。
- 对最小化问题：

```text
performance_gain = P_skip - p_query
```

- success rate 等越大越好的指标只用于报告，不直接放入该差值。

若使用综合指标，必须先统一方向并冻结公式。

---

## 9. Default Optimizer

裁决：

- 在线部署主设置使用 SBS 作为初始 default optimizer。
- SBS 只由训练集平均表现确定。
- 固定 CMA-ES 或 DE 可作为敏感性分析。
- 第一篇论文的主 probe/default 都固定为训练集 SBS；主 Decision 数据只保留 `prefix_algorithm == default_algorithm` 的行。
- 主 No-query 路径原生继续 SBS prefix 的完整状态，不重启、不改参数、不改变总预算口径。
- DE、PSO、CMA-ES、SHADE 的全 prefix 标签继续生成，但只进入 cross-probe robustness、leave-one-probe-out 和 algorithm-agnostic 泛化分析，不得混入主结果。
- 在非主 prefix 行中，Skip 使用训练集 SBS/default；若 default 与 prefix 不同，必须显式记录一次 population transfer，不能称为原生 continuation。

历史小规模链路验证已退出当前运行面，不再保留对应配置或结果入口。

---

## 10. SHADE / L-SHADE

裁决：

- 软件接口统一使用 `shade` 或 `shade_family`。
- 当前完整状态实现使用 SHADE；不得在本轮重生成中切换为 L-SHADE。

### 10.1 完整状态实现的算法定义

为保证初始化、完整运行、checkpoint 恢复和分支续跑使用同一逻辑，四种算法均由 `optimizers/state.py` 的状态推进器执行，不再混用外部完整运行器与本地 continuation adapter。

- DE：`rand/1/bin`，固定 `F=0.5`、`CR=0.9`。
- PSO：固定 `w=0.72`、`c1=c2=1.49`，速度上限为每维搜索区间的 `0.2`。
- CMA-ES：population size 等于正式配置，`mu=lambda/2`，包含 rank-one/rank-mu covariance update、step-size path、covariance path 和完整 eigensystem state；初始 mean 为边界中点，初始 sigma 为平均边界跨度的 `0.3`。
- SHADE：memory size 为 `5`，保留 `M_F`、`M_CR`、archive 与 memory index，使用 current-to-pbest/1 和成功历史更新。

上述定义属于新的数据生成机制。旧 pymoo/cma 与 population-only adapter 混合生成的结果不得与新结果合并。

---

## 11. Random Analysis

裁决：

- 统一术语为 `Random Analysis`。
- 定义：以固定概率决定 Run Query 或 No-query。
- 主配置：

```text
query_run_probability = 0.5
```

代码配置名：

```text
random_analysis
```

---

## 12. 运行记录

裁决：

- 不自动采集 `git_commit`。
- 不实现哈希、checksum、digest、manifest、receipt、source closure、quarantine 或执行解锁机制。
- 运行记录保存普通实验元数据。

允许保存：

```text
config
seed
benchmark version
optimizer settings
model settings
timestamp
```

---

## 13. 正式实验与 preliminary 运行

裁决：

- 正式论文结果只来自冻结配置、完整 function-family split 和预定重复次数。
- preliminary 运行只用于定位实现或资源问题，必须与正式输出目录隔离。
- preliminary 运行不得进入论文主表、模型选择或 threshold 拟合。
- 完整论文实验执行 30 independent runs、完整 baselines、外部评价和统计分析。

---

## 14. 三档 Landscape Query 与表示范围

裁决：

- 第一篇论文只以 `descriptor_cheap` 为主 query；它是固定 16 维自定义低成本描述符，不得称为 Full ELA 或完整 pflacco。
- `pflacco_standard` 与 `pflacco_broad` 是预先定义的配置稳健性实验，不得根据 validation 结果替换主 query。
- 三档分别学习 Selector、Utility target 与 Decision Model，不把 `query_id` 作为模型输入，也不训练动态 query-type selector。
- 所有 query-sensitive 的 Selection Reference、Utility、Decision、baseline 与外部评价入口必须显式接收 `query_id`，并核对 `query_protocol`、`sample_design_id` 与实际列；默认输出只能写入对应的 query 目录。
- `Always Query` 与 `Traditional AAS` 在当前协议中都是“固定 query + 同一 Selector”，因此共享一次运行结果并使用两个报告标签，不重复等价 continuation。VBS 必须由静态 per-problem 完整候选结果计算，不能以逐状态 `best observed action` 替代。
- NeurELA、Deep-ELA、Progressive ELA 和其他学习式或动态 landscape representation 只用于说明表示异质性，本轮不实现。

---

## 15. Search Maturity

裁决：

- 正式模型同时比较 Direct Behavior -> Utility 与 Maturity-aware 表征。
- Search Maturity 可作为 Phase 3 的派生字段。
- Phase 5 比较两条模型：

```text
Direct Behavior Model
Maturity-aware Model
```

- 不预设 Search Maturity 一定有效。
- `M_t = ES_t(1 - XS_t)` 是本文启发式定义，必须通过消融和 OOD 结果验证。

---

## 16. 开发主线

冻结开发主线：

```text
configs
-> benchmarks / optimizers
-> trajectory
-> behavior
-> landscape_queries
-> selection_reference
-> utility_labels
-> decision
-> evaluation
```

避免主目录名：

```text
oracle
tests
schema_registry
audit
```

## 16.1 算法切换后的初始化口径

裁决：

- 主实验的跨算法切换采用 Population Transfer；同算法路径采用原生完整状态 continuation。
- 主协议中 `prefix_algorithm == default_algorithm ==` 训练集 SBS，因此 No-query 必须原生继续该完整 checkpoint state。
- 全 prefix 稳健性数据中，Skip 使用 default algorithm；只有 `default_algorithm != prefix_algorithm` 时才执行一次 population transfer，并将 `skip_switches_from_prefix` 记为真。
- Run Query 若选择 prefix algorithm，必须继续同一完整 checkpoint state，只减少 query 消耗的后续优化预算。
- selector、Always Query、Random Analysis 和 Traditional AAS 只有在共享 checkpoint 后确实切换算法时，才使用该 checkpoint 的 `population`、`fitness` 和 `best-so-far position` 初始化新算法。
- Best-so-far Warm Start 不作为主实验默认口径，只能作为后续稳健性分析候选。
- query 采样点不并入后续优化 population；固定 query 只提供 selector 所需特征，并通过减少 `FE_query_optimization` 体现 FE 成本。

实现含义：

- 新算法继承的是算法无关搜索状态：位置、fitness 和当前 best。
- 新算法不继承前缀算法内部状态。
- DE、PSO、CMA-ES 和 SHADE 只在跨算法切换时按新算法初始化内部状态；该操作必须标记为 `population_transfer_initialization`，不得称为原生 continuation。
- 同算法路径保存并恢复完整状态：DE 保留 generation 与 RNG；PSO 额外保留 velocity、personal/global best；CMA-ES 额外保留 mean、covariance、sigma、evolution paths 和 strategy state；SHADE 额外保留 `M_F`、`M_CR`、archive 与 memory index。
- `p_query` 表示付出固定 query 成本后，selection reference 选择的算法从同一 checkpoint population 继续优化得到的 final performance；不是围绕 best-so-far 重启得到的 performance。

---

## 17. 当前仍需单独冻结的实验细节

当前仍需确定：

- CEC2022 与工程问题的具体函数范围、维度、预算和重复次数。
- 主 `lambda_time`、`lambda_memory` 及其敏感性分析取值。
- LDA 主 controller 的最终导出格式和外部评价加载路径。
- 论文级分层统计、不确定性区间和多重比较口径。

CEC2017 已由 `configs/phase1_cec2017_test.yaml` 冻结为 29 个函数、10D / 30D / 50D、30 seeds 和 `FE_total=1000D`；在修改该配置前必须先说明研究问题、数据泄漏风险和结果保存方式。

已经冻结并进入正式 phase1 的细节：

- 主 query 为 `descriptor_cheap`（16 维、`lhs_50d`、5% FE）；`pflacco_standard`（37 维、`lhs_50d`、5% FE）与 `pflacco_broad`（52 维、`lhs_100d`、10% FE）只作预定义配置稳健性。
- cheap 与 standard 共享完全相同的 (X,y)；broad 使用独立样本与 action-loss 预算。
- 主环境不依赖 pflacco；标准特征只由 `tools/pflacco_query/` 的 Python 3.11、pflacco 1.2.2 环境从 Parquet 样本提取。
- BBOB train / validation function family split，见第 5.1 节。
- BBOB train / validation dimensions：10D / 20D / 40D。
- Phase1 主 checkpoint ratios：`0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60`。
- `dimension`、`function_id`、`algorithm_id`、算法内部参数、query features 不进入 Decision Model 输入；这些字段只作为 metadata 和分层诊断使用。
- 必须保存 `selected_equals_default`、`selected_equals_prefix` 和 `skip_switches_from_prefix`；三者分别回答 selector 是否选择 SBS、query 路径是否继续当前算法、No-query 是否离开当前算法。
- `same_algorithm` 是 `selected_equals_default` 的报告分层名，不得在多 prefix 数据中解释为“query 后继续当前算法”。
- 主协议因 `prefix_algorithm == default_algorithm`，此时 `selected_equals_default == selected_equals_prefix`，语义才可合并。
- Selection Reference / query-conditioned selection pipeline 是固定下游组件，不作为本文方法贡献点。
- 算法切换后的主初始化口径为 Population Transfer；query 采样点不复用到后续优化。
