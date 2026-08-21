# DBF 实验改进 TODO 清单

> 整理自 `2026-08-19_对话汇总_Vibe实施说明.md`，对照当前实验代码状态。
> 按 **必须改 / 建议改 / 可选改** 三档，再按文件组织。
>
> **✅ MA-BBOB diversity pilot 已实施**：参考 RL-DAS 方法，已完成 manifest 驱动的 42 定义 diversity 池 + 24 个正式子集选择。详见 `scheme_mabbob_diversity_pilot.md`。

---

## 目录

- [必须改](#必须改)
- [建议改](#建议改)
- [可选改](#可选改)
- [执行顺序建议](#执行顺序建议)

---

# 必须改

## 1) `utility_labels/efficacy.py`

### TODO 1.1：把 G_FE 标签从"单次"升级为"paired repetitions 的稳健标签"

当前 `g_fe` 定义正确，但只依赖单次 continuation。文档强调 `G_FE` 标签噪声可能很强。

需要增加：
- 对同一 state 的多次 continuation 结果输入
- 输出字段：
  - `g_fe_rep_1 ... g_fe_rep_R`
  - `g_fe_median`
  - `g_fe_std`
  - `g_fe_ci_low / g_fe_ci_high`
  - `sign_flip_rate`
- 训练时默认使用稳健聚合值（median 或 trimmed mean），不直接依赖单次 continuation

### TODO 1.2：补充 multi-horizon efficacy 的派生接口

保留 `g_fe_final`，额外支持：
- `g_fe_h100d`
- `g_fe_h200d`
- `g_fe_h500d`

短 horizon 只作为辅助，不替代 final efficacy。

---

## 2) `utility_labels/generation.py`

### TODO 2.1：把 utility label 生成流程改成支持 paired continuation

`paired_utility_label_view(...)` 需要接收多次 continuation 的汇总结果。
输出标签时保留 repetition-level 统计，在最终 parquet 中保存稳健标签与方差信息。

### TODO 2.2：补充 query / behavior / state-only / sampling-only 的一致性诊断字段

保留 `g_fe_rep_*`、`g_fe_median`、`g_fe_std`、`g_fe_sign_flip_rate`，用于后续分析标签噪声。

---

## 3) `decision/compare_controller_baselines.py`

### TODO 3.1：加入 Behavior-only DAS 作为正式 baseline

只用 `B_t`（trajectory behavior）的动态选择器，与 DBF 主方法做公平比较。

### TODO 3.2：加入 Fixed-1Switch 和 Random-1Switch

对齐 switch rate、trigger time 分布，作为动态算法选择的基础对照。

### TODO 3.3：加入 SwitchBenefit-RF

用 trajectory / behavior 预测切换收益，作为 DBF 的强相关 baseline。

---

## 4) `decision/online_controller_evaluate.py`

### TODO 4.1：强化 first-trigger 检查

- 每条 run 最多触发一次 Query
- 触发后不能重复 Query

### TODO 4.2：强化 no-query leakage 检查

若 controller 没触发 Query：
- 不生成 query samples
- 不计算 query features
- 不留任何 query 结果痕迹

### TODO 4.3：强化 query RNG 隔离

- Query RNG 必须与 optimizer continuation RNG 完全独立
- 相同状态应可复现相同 Query
- Query 不得改变后续随机序列

---

## 5) `decision/sampling_opportunities.py`

### TODO 5.1：把 decision opportunity 检查做成更强的协议约束

保证 sampling opportunity 的 state key 唯一，保证 milestone / event row 语义清晰，为后续 first-trigger replay 对齐提供硬约束。

### TODO 5.2：扩展 opportunity 统计字段

增加：
- `decision_opportunity_index`
- `sampling_opportunity_type`
- `trigger_phase`
- `trigger_reason`

---

## 6) `trajectory/sampling.py`

### TODO 6.1：把 current monitor grid 之外的机会点做 pilot

当前固定区间 `0.20–0.60`，需要验证：
- 早期（`0.10–0.20`）是否有收益
- late-stage（`0.60–0.70`）是否有收益
- 旧中段区间是否漏掉高价值 state

### TODO 6.2：保留现有冻结协议，但增加可审计的扩展版本

不直接改正式冻结协议，先通过 pilot 产出证据，再决定是否调整正式 grid。

---

## 7) `configs/phase1_bbob_train.yaml`

### TODO 7.1：补齐 10D / 20D / 40D 的分层实验配置

- `10D`：开发主实验
- `20D`：主泛化验证
- `40D`：stress test

### TODO 7.2：为 OOD / 泛化实验预留更清晰的 split 组织

train / validation / stress test 不要混在同一配置里。

---

## 8) `configs/phase1_bbob_validation.yaml`

### TODO 8.1：保持 function-family OOD 的严格隔离

确认 validation functions 没有任何训练泄漏。训练过程中生成的任何派生数据都不能"间接见过"validation family。

---

## 9) `configs/phase1_mabbob_formal.yaml`

### TODO 9.1：把正式 MA-BBOB 子集从"编号子集"升级为"coverage-selected 子集"

✅ **已完成**：已实施 manifest 驱动的 42 定义 diversity 池 + 24 个分层选择正式子集。
- 新增 `experiments/cli/generate_mabbob_diversity_pool.py` 生成结构化池
- 新增 `experiments/cli/select_mabbob_formal_subset.py` 做分层 maximin 选择
- `configs/phase1_mabbob_formal.yaml` 已改为 manifest 驱动
- 正式子集：24 个定义（8 anchor + 8 pairwise + 4 triple + 4 dense）

### TODO 9.2：增加正式实验的多维配置入口

目前正式配置偏 `10D`，建议预留 `20D` / `40D` 的独立配置。

---

## 10) `configs/phase1_mabbob_pilot.yaml` 和 `configs/phase1_mabbob_pilot_101_200.yaml`

### TODO 10.1：把两个 pilot 批次用于 coverage analysis，而不是直接当正式子集

✅ **已完成**：新 diversity pilot（42 定义）已生成，用于 coverage analysis。
- 旧 1-100 和 101-200 批次仍保留作为历史对照基线
- 新 pilot 配置：`results/mabbob_diversity_pilot/phase1_mabbob_diversity_pilot.yaml`

### TODO 10.2：明确 anchor / bridge / sparse-mixture 的分层标签

✅ **已完成**：manifest 中已显式标记：
- `bridge_type`: anchor / pairwise_bridge / sparse_3way_bridge / dense_bridge
- `strata_tag`: anchor / C1×C2 / triple_* / dense_*
- `profile_tag`: default / dominant_expand / dominant_contract / flat / balanced / graduated / dominant
- `variant_tag`: anchor / pairwise_* / triple_* / dense_*

---

## 11) `benchmarks/mabbob.py`

### TODO 11.1：检查并明确 MA-BBOB candidate 的结构语义

✅ **已完成**：
- 新增 `MABBOBDefinition` dataclass 显式记录结构语义
- 新增 `bridge_type` / `xopt_mode` / `profile_tag` / `variant_tag` 字段
- manifest 中每个 entry 都带完整结构元数据

### TODO 11.2：确保 train/test component leakage 不会发生

✅ **已完成**：
- manifest 中每个 entry 带 `is_val_component` 标记
- 选择脚本默认排除 val component（F9, F24）进入 train 正式集
- leakage audit 已集成到选择逻辑

---

## 12) `selection_reference/build.py`

### TODO 12.1：让 selector reference 明确支持 behavior-only baseline

不要只保留 query-driven selector reference，要能生成 behavior-only 的对照参考。

### TODO 12.2：把 acceptable action / selector regret 所需字段补全

输出：
- `selector_regret`
- `acceptable_action`
- `acceptable_accuracy`
- `top1_probability`
- `probability_margin`

---

## 13) `decision/model_protocol.py`

### TODO 13.1：把强 baseline 模型协议固定下来

明确：
- `classification` baseline
- `regression` baseline
- `per-action regression` 的协议入口

### TODO 13.2：保留多种 selector 形式但冻结主协议

- multiclass
- per-action regression
- behavior-only vs query-enhanced

---

# 建议改

## 14) `trajectory/query.py`

### TODO 14.1：为 query reservoir 加更完整的可复现信息

增加：
- query seed 的组成字段
- query 触发 state 标识
- query 与 continuation 的隔离检查结果

### TODO 14.2：为 query feature sample size 做分档实验支持

支持 `5D` / `10D` / `20D` / `50D`，方便做 sample-size sensitivity。

---

## 15) `utility_labels/batch_generation.py`

### TODO 15.1：加入标签质量报告里的噪声统计

- `g_fe` 分布
- `sign_flip_rate`
- `median vs mean` 差异
- `CI width`

### TODO 15.2：在汇总报表里区分"publication evidence"和"engineering consistency"

增加 label stability summary、query/no-query balance、positive efficacy ratio。

---

## 16) `decision/online_controller_evaluate.py` 和 `decision/compare_controller_baselines.py`

### TODO 16.1：加入 acceptable action set

不要把唯一最佳算法当作唯一真值，把"近似等价算法"归入 acceptable set。

### TODO 16.2：加入 selector regret

输出：
- `selector_regret_raw`
- `selector_regret_norm`

作为主评价之一。

---

## 17) `utility_labels/fields.py`

### TODO 17.1：扩展正式标签字段

新增字段类目：
- repetition-level efficacy
- horizon-level efficacy
- uncertainty statistics
- acceptable action / regret

### TODO 17.2：统一旧 utility 列名与新 g_fe 字段的兼容层

保留兼容诊断，但主论文输出应以 `g_fe` 系列为主。

---

## 18) `trajectory/records.py`、`trajectory/final_performance.py`

### TODO 18.1：补充 paired continuation 所需的记录字段

- `replicate_id`
- `continuation_branch_id`
- `query_branch_id`
- `paired_status`

### TODO 18.2：补充 final performance 的 uncertainty 结构

- `final_gap`
- `final_gap_ci`
- `final_gap_median`
- `final_gap_std`

---

## 19) `experiments/phase1_batch_common.py`

### TODO 19.1：让实验配置校验支持更多维度与 split 组合

- 10D / 20D / 40D
- train / validation / stress-test

### TODO 19.2：把 future pilot 与 formal 的差异检查得更严格

检查项：
- population size
- FE budget
- algorithm portfolio
- sampling protocol
- query protocol

---

## 20) `experiments/cli/phase1_collect_batch.py`

### TODO 20.1：把 shard 采集的元数据补全

附带：
- `sampling_opportunity_type`
- `decision_opportunity_index`
- `trigger_phase`
- `trigger_reason`

### TODO 20.2：为后续 paired continuation 数据生成预留接口

当前只采 trajectories 还不够，需要能把同一 state 的多个 continuation 关联起来。

---

# 可选改

## 21) `decision/skip_defer_query.py`

### TODO 21.1：如果后续想做三向控制，再把它纳入正式实验

`skip` / `defer` / `query` 三向控制，当前可以先不进主协议，作为扩展分支。

---

## 22) `decision/conformal.py`

### TODO 22.1：如果后面要做风险控制，可以加入 conformal calibration

用于阈值稳健化，但不建议现在就进入主协议。

---

## 23) `decision/practical_delta.py`

### TODO 23.1：把 practical threshold 的研究做成单独消融

不要让阈值变成主贡献，只作为辅助决策参数。

---

## 24) `decision/pilot_coverage.py`

### TODO 24.1：把 coverage 选点做成正式工具

- landscape coverage
- behavior coverage
- action discrimination
- maximin score

这个模块很适合承接正式 MA-BBOB 子集选择。

---

## 25) `decision/nested_learning.py`

### TODO 25.1：如果要做更完整的 OOD / nested CV，继续增强这里

但只在主数据协议冻结后再扩，否则会把工程复杂度拉太高。

---

## 26) `landscape_queries/cheap.py`、`landscape_queries/batch_features.py`

### TODO 26.1：如果要扩 query representation，再在这里做

- `descriptor_cheap`
- `pflacco_standard`
- `pflacco_broad`

目前主协议可以先只保留 `cheap`，后续再做 sensitivity analysis。

---

# 执行顺序建议

## 第一批先做（P0）

| 优先级 | 文件 | 任务 |
|-------|------|------|
| P0 | `utility_labels/efficacy.py` | 1.1, 1.2 |
| P0 | `utility_labels/generation.py` | 2.1, 2.2 |
| P0 | `decision/compare_controller_baselines.py` | 3.1, 3.2, 3.3 |
| P0 | `decision/online_controller_evaluate.py` | 4.1, 4.2, 4.3 |
| P0 | `decision/sampling_opportunities.py` | 5.1, 5.2 |

## 第二批再做（P1）

| 优先级 | 文件 | 任务 |
|-------|------|------|
| P1 | `trajectory/sampling.py` | 6.1, 6.2 |
| P1 | `configs/phase1_bbob_train.yaml` | 7.1, 7.2 |
| P1 | `configs/phase1_bbob_validation.yaml` | 8.1 |
| P1 | `configs/phase1_mabbob_formal.yaml` | 9.1, 9.2 |
| P1 | `benchmarks/mabbob.py` | 11.1, 11.2 |
| P1 | `configs/phase1_mabbob_pilot.yaml` | 10.1, 10.2 |

## 第三批补充（P2）

| 优先级 | 文件 | 任务 |
|-------|------|------|
| P2 | `selection_reference/build.py` | 12.1, 12.2 |
| P2 | `decision/model_protocol.py` | 13.1, 13.2 |
| P2 | `utility_labels/batch_generation.py` | 15.1, 15.2 |
| P2 | `trajectory/query.py` | 14.1, 14.2 |
| P2 | `decision/online_controller_evaluate.py` | 16.1, 16.2 |
| P2 | `utility_labels/fields.py` | 17.1, 17.2 |
| P2 | `trajectory/records.py` | 18.1, 18.2 |
| P2 | `experiments/phase1_batch_common.py` | 19.1, 19.2 |
| P2 | `experiments/cli/phase1_collect_batch.py` | 20.1, 20.2 |

---

> 最后一次更新：2026-08-20
> 来源：`2026-08-19_对话汇总_Vibe实施说明.md`