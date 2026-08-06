# Decision-before-Feature 开发前裁决

本文档记录已经确认的项目级裁决，用于解决 `AGENTS.md` 与 `docs/` 研究设计文档之间的约束冲突。

优先级规则：

1. `AGENTS.md` 是最高优先级。
2. 本文档用于解释如何在开发中落实 `AGENTS.md` 和研究文档。
3. 若 `docs/` 中早期方案与本文档冲突，开发时按本文档执行。

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
- `ELA Utility Oracle` 在开发中改称 `utility_labels` 或 `offline utility label generation`。
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
same checkpoint state
-> Skip ELA continuation
-> Run ELA continuation
-> paired comparison
```

---

## 5. Train / Validation / Test

裁决：

- Train: BBOB, 10D / 20D / 40D。
- Validation: BBOB，按第 5.1 节正式 BBOB trajectory 协议执行。
- Test: CEC2017 / CEC2022。
- Engineering problems: 最终外部验证或扩展，不进入 MVE。

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

- 当前 min_support 诊断显示 `changed_algorithm` 且 `U_ELA>0` 的主要机会集中在 `FE_ratio=0.30-0.55`。
- `0.20` 保留为 transition 前参照，`0.25` 和 `0.28` 用于捕捉 0.20 到 0.30 之间的 selection-reference / performance-bucket 过渡。
- `0.60` 保留为机会区之后的衰减参照。
- very early checkpoints 例如 `0.005-0.15` 和 late endpoints 例如 `0.75/1.00` 不进入正式 phase1 主采样频率；如需研究 early/late 行为，应另设扩展实验。

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
- 单文件输出仅保留给 MVE 和小规模链路验证。

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

## 6. Dimension 使用

裁决：

- `dimension` 必须保存为 metadata。
- `dimension` 不进入 Decision Model 输入列。
- `dimension` 仅用于 split、分组报告和 OOD 分析。

同类规则：

- `function_id`、`algorithm_id`、算法内部参数、ELA feature 都不进入 Decision Model。
- `algorithm` 可保存为 metadata，但不作为模型输入。

---

## 7. Utility 成本账本

裁决：

- 主协议采用等总 FE 预算。
- ELA 消耗的 FE 通过减少后续优化预算体现。
- 若 ELA FE 已经从优化预算中扣除，Utility 中不能再扣同一笔 FE 成本。
- Utility 中额外扣除 wall-time、feature computation、memory 等非 FE 成本。

必须保存：

```text
FE_total
FE_prefix
FE_analysis
FE_optimization
runtime_analysis
runtime_decision
runtime_optimization
```

主标签口径：

```text
U_ELA = performance_gain - lambda_time * time_cost - lambda_memory * memory_cost
```

若某实验允许额外 ELA FE，必须另设清晰公式并单独记录：

```text
U_ELA = performance_gain
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
performance_gain = P_skip - P_ELA
```

- success rate 等越大越好的指标只用于报告，不直接放入该差值。

若使用综合指标，必须先统一方向并冻结公式。

---

## 9. Default Optimizer

裁决：

- 完整实验主设置使用 SBS 作为 default optimizer。
- SBS 只由训练集平均表现确定。
- 固定 CMA-ES 或 DE 可作为敏感性分析。

MVE 口径：

- MVE 可先用固定 DE 或 CMA-ES 验证链路。
- MVE 结果不能作为论文结论。

---

## 10. SHADE / L-SHADE

裁决：

- 软件接口统一使用 `shade` 或 `shade_family`。
- MVE 可先实现 SHADE。
- 完整实验优先使用 L-SHADE；若实现成本过高，可冻结为 SHADE 并在配置中说明。

---

## 11. Random Analysis

裁决：

- 统一术语为 `Random Analysis`。
- 定义：以固定概率决定 Run ELA 或 Skip ELA。
- 主配置：

```text
p_ELA = 0.5
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

## 13. MVE 与完整论文实验

裁决：

- Minimum Viable Experiment 只验证软件链路。
- MVE 不产生论文主结论。
- MVE 可以使用少量 function families、少量 seeds、少量 dimensions。
- 完整论文实验才执行 30 independent runs、完整 baselines、OOD 和统计分析。

---

## 14. Compact ELA / Progressive ELA

裁决：

- 第一篇主实验不实现 Compact ELA 或 Progressive ELA。
- Phase 4 的 ELA 路径固定为 Full ELA。
- Compact ELA 和 Progressive ELA 保留为 discussion 或后续扩展。

---

## 15. Search Maturity

裁决：

- MVE 先实现 Direct Behavior -> Utility。
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

---

## 17. 当前仍需单独冻结的实验细节

这些不是文档冲突，而是开发前仍需确定的实验参数：

- CEC2017 / CEC2022 的具体函数范围、维度和预算。
- `FE_prefix`、`FE_analysis`、`FE_optimization` 的具体数值。
- ELA 采样点是否允许复用到后续优化。
- 主 `lambda_time`、`lambda_memory` 和敏感性分析取值。
- Search Maturity 中 ES、XS 的具体公式、窗口和归一化。
- 方向熵的高维离散方案、零位移处理和平滑方式。

已经由 min_support 结果冻结并进入正式 phase1 的细节：

- BBOB train / validation function family split，见第 5.1 节。
- BBOB train / validation dimensions：10D / 20D / 40D。
- Phase1 主 checkpoint ratios：`0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60`。
- `dimension`、`function_id`、`algorithm_id`、算法内部参数、ELA features 不进入 Decision Model 输入；这些字段只作为 metadata 和分层诊断使用。
- `same_algorithm` rows 作为共享前缀续跑随机差异参照，不应与 `changed_algorithm` rows 混为同一解释。
- Selection Reference / ELA-based selection pipeline 是固定下游组件，不作为本文方法贡献点。
