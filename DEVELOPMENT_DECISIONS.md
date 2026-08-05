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
- Validation: BBOB, 50D。
- Test: CEC2017 / CEC2022。
- Engineering problems: 最终外部验证或扩展，不进入 MVE。

禁止：

- 用 CEC 或 engineering problems 调参。
- 用测试函数训练 Decision Model、selection reference 或 threshold。
- 随机 function instance split。

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

- BBOB function family 的具体 train / validation / test family 列表。
- CEC2017 / CEC2022 的具体函数范围、维度和预算。
- `FE_total`、`FE_prefix`、`FE_analysis`、`FE_optimization` 的具体数值。
- ELA 采样点是否允许复用到后续优化。
- 主 `lambda_time`、`lambda_memory` 和敏感性分析取值。
- Search Maturity 中 ES、XS 的具体公式、窗口和归一化。
- 方向熵的高维离散方案、零位移处理和平滑方式。
