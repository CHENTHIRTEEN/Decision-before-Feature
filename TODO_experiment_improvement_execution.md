# DBF 实验改进 TODO：P0–P2 落地版

> 目标：把 DBF 实验改造成能够支撑“equal-FE efficacy + 条件 Query + 下游 Selector + 强 DAS 对照”的可复现实验协议。
>
> 使用方式：按 P0 → P1 → P2 顺序实施。每项任务都包含目标、涉及文件、输入/输出、依赖和验收标准。未通过验收标准前，不得生成下一阶段正式数据。
>
> 状态约定：
> - `[ ]` 未开始
> - `[-]` 进行中
> - `[x]` 已完成
> - `[blocked]` 被依赖项阻塞
>
> **✅ MA-BBOB diversity pilot 已实施**：P0.6 已完成，生成 42 定义 diversity 池 + 24 个正式子集。详见 `scheme_mabbob_diversity_pilot.md`。

---

## 0. 总体依赖关系

```text
P0.1 paired continuation
        ↓
P0.2 robust efficacy label contract
        ↓
P0.3 strong DAS baselines ─────┐
        ↓                       │
P0.4 online protocol checks     │
        ↓                       │
P0.5 opportunity-range pilot    │
        ↓                       │
P0.6 coverage-driven MA-BBOB ───┘
        ↓
P1.1 multi-horizon efficacy
P1.2 dimension layering
P1.3 acceptable action + regret
P1.4 OOD leakage audit
P1.5 label-noise audit
        ↓
P2 extensions: skip/defer/query, calibration, diversify, external benchmarks
```

### 正式数据生成前的硬门槛

- [ ] Equal-FE accounting 在所有路径通过。
- [ ] Query 未触发时没有 query sample、objective evaluation 或 query feature。
- [ ] 每条 run 的 Query 次数不超过 1。
- [ ] Query RNG 与 continuation RNG 隔离且可复现。
- [ ] paired continuation 的重复数、聚合方法和标签版本已冻结。
- [ ] train / validation / OOD 的 component composition 已审计。
- [ ] MA-BBOB formal subset 已通过 coverage audit。
- [ ] baseline 使用相同总 FE、相同 opportunity、相同 handoff 语义。

---

# P0：必须先完成

## P0.1 稳定 `G_FE` 标签：paired continuation repetitions

### 目标

将同一个 decision state 的单次 Query/No-query continuation 标签升级为可估计不确定性的 paired repeated continuation 标签。主科学量仍然是 final equal-FE efficacy，不改变定义。

### 涉及文件

- `utility_labels/efficacy.py`
- `utility_labels/generation.py`
- `utility_labels/fields.py`
- `trajectory/records.py`
- `trajectory/final_performance.py`
- `experiments/phase1_batch_common.py`
- 新增或扩展 continuation replay 入口（优先放在现有 action-loss / selection-reference 流程中，避免另造不兼容的数据管线）

### 实施任务

- [ ] 在配置中增加明确的 `efficacy_repetitions`，Pilot 使用 `1, 3, 5` 做敏感性比较，正式协议先冻结为 `3`。
- [ ] 为每个 state/path/repetition 保存唯一键：
  `state_key + path + repetition_index`。
- [ ] 明确 paired 语义：同一 state、同一总额外 FE、No-query 与 Query 使用相同 repetition 配对规则。
- [ ] 保存每次 continuation 的 terminal gap、effective FE、completion status 和失败信息。
- [x] 在 `efficacy.py` 增加 repetition 聚合函数：median、mean、trimmed mean、std、sign-flip rate。
- [x] 先提供 deterministic 95% t-interval CI，并通过 `g_fe_uncertainty_protocol` 记录方法；paired bootstrap 留作后续敏感性分析。
- [ ] 主标签字段明确区分：
  - `g_fe_final_single`
  - `g_fe_final_median`
  - `g_fe_final_mean`
  - `g_fe_final_std`
  - `g_fe_final_ci_low`
  - `g_fe_final_ci_high`
  - `g_fe_final_sign_flip_rate`
- [ ] 设置 `PRIMARY_EFFICACY_VALUE_COLUMN` 的正式版本；训练默认使用 median 聚合值。
- [ ] 对 `R=1` 保留兼容路径，但不得把它与 `R=3` 正式结果混在同一训练集而不标记版本。

### 输入

- 已保存的 decision states
- Query / No-query continuation replay 结果
- Query cost 与 continuation cost
- `log10_gap_floor`、`log10_gap_cap`

### 输出

- 带 repetition 字段的 utility label parquet
- label protocol/version 字段
- repetition consistency report

### 验收标准

- [ ] 每个正式 state/path 都有规定数量的 repetition，或有明确失败状态，不允许静默缺行。
- [ ] 所有 repetition 的总额外 FE 相同。
- [ ] `g_fe_median` 能由 repetition-level primitive facts 重新计算。
- [ ] `sign_flip_rate` 与 repetition-level 符号一致。
- [ ] 同一输入重复运行生成完全相同的标签文件。
- [ ] `validate_utility_label_file` 能拒绝 repetition 不完整、FE 不一致、协议版本错误的数据。

### 依赖

无；但 P0.2、P0.3 和正式数据生成依赖本任务。

---

## P0.2 固化 robust efficacy label contract

### 目标

让所有下游模型、baseline 和报表明确使用同一个主标签，不再混用旧 Utility 与新的 `g_fe` 语义。

### 涉及文件

- `utility_labels/efficacy.py`
- `utility_labels/fields.py`
- `utility_labels/generation.py`
- `utility_labels/validation.py`
- `utility_labels/batch_generation.py`
- `decision/train_full_decision_model.py`
- `decision/compare_controller_baselines.py`

### 实施任务

- [x] 定义唯一主标签协议的 repetition-aware contract：当前以 `equal_total_fe_log_ratio_v1` 为基础，新增 repetition/aggregation/uncertainty 字段；`g_fe` 已由稳健聚合路径写出，后续仅需在正式数据中冻结 protocol ID。
- [x] 将 `g_fe_final_median` 映射为正式 `g_fe` 的运行时写出逻辑已接入；R=1 兼容路径会保留重复列为空，避免伪装成多次重复。
- [x] 所有当前 controller 训练入口默认读取正式 `g_fe`，旧 `u_*` 只作为 Behavior-only 兼容诊断。
- [x] 将 `delta_practical` 保持为校准产物，不写入原始 efficacy 计算。
- [x] 为 label schema 增加 `g_fe_n_repetitions`、`g_fe_aggregation`、`g_fe_uncertainty_protocol` 和 repetition series columns。
- [x] 更新 validation，检查标签协议、重复数和聚合方法（兼容 R=1 与正式 repetition-aware 输出）。

### 验收标准

- [ ] `grep` 检查训练入口后，主训练路径只有一个正式 efficacy target。
- [ ] 旧 Utility 字段只能作为兼容诊断，不影响主标签。
- [ ] label summary 能显示不同 repetition 聚合方式的差异。

---

## P0.3 补齐强 DAS baseline

### 目标

证明 DBF 的收益来自“是否值得支付独立信息获取成本”，而不是来自一个弱的动态算法选择器。

### 涉及文件

- `decision/compare_controller_baselines.py`
- `decision/train_full_decision_model.py`
- `decision/model_protocol.py`
- `selection_reference/build.py`
- `selection_reference/model.py`
- `decision/static_portfolio_reference.py`
- 建议新增：
  - `decision/baselines_behavior_only.py`
  - `decision/baselines_switch.py`
  - `decision/baselines_switch_benefit.py`
  - `decision/baseline_protocol.py`

### 必须实现的 baseline

| ID | 方法 | 额外 FE | 动态 | 目的 |
|---|---|---:|---:|---|
| S0 | SBS / CMA-ES | 0 | 否 | 最佳单算法基线 |
| D0 | Random-1Switch | 0 | 是 | 排除随机切换收益 |
| D1 | Fixed-1Switch | 0 | 是 | 排除固定时机收益 |
| D2 | Behavior-only DAS | 0 | 是 | 强免费信息基线 |
| D3 | Trajectory-based AS | 0 | 是 | 零额外 FE trajectory AS |
| D4 | SwitchBenefit-RF | 0 | 是 | 直接预测切换收益 |
| A0 | Always Query + downstream selector | Query FE | 是 | 排除 gate 贡献 |
| P | DBF | 条件 Query FE | 是 | Proposed |
| O1 | Single-switch DynVBS | 0 | oracle | 动态上界 |

### 实施任务

- [x] 新增 `decision/baseline_protocol.py`，定义统一 policy spec、state key 校验、one-action-per-run 和固定/随机 switch 调度接口。
- [x] 固定 action 语义协议：`continue_current`、`switch_to_X`；不得把 continue 隐式实现为 perturb。
- [-] `Behavior-only DAS` 当前已有 prediction/threshold 入口；仍需补齐独立 run-level action replay 以满足完整 DAS baseline。
- [x] `Fixed-1Switch` / `Random-1Switch` 的调度接口已在 `baseline_protocol.py` 中统一；仍需接入正式 run replay。
- [ ] `SwitchBenefit-RF` 训练 per-target-action 的 switch benefit 或 future loss regression。
- [ ] `Always Query` 使用与 DBF 相同的 Query protocol、sample size 和 downstream selector。
- [ ] 每个 baseline 输出 run-level 结果，而不是只输出 state-level prediction。
- [x] `compare_controller_baselines.py` 已输出 baseline protocol 摘要，并支持可选 Fixed/Random/SwitchBenefit 输入路径；实际 prediction/action-loss 生成与 run-level 汇总仍待完成。

### 统一输入/输出

输入：

- 相同 initial probe
- 相同 decision opportunity set
- 相同 prefix state
- 相同总 FE
- 相同算法 portfolio
- 相同 handoff 实现

输出：

- `run_id`
- `method_id`
- `query_count`
- `switch_count`
- `trigger_FE`
- `selected_action`
- `final_raw_fitness`
- `final_gap`
- `g_fe`
- `selector_regret`
- `runtime_total`
- `status`

### 验收标准

- [ ] 所有 baseline 通过 equal-FE check。
- [ ] 所有 baseline 使用相同 seed 集和 opportunity set。
- [ ] `Random-1Switch` 的匹配误差有报告。
- [ ] `Behavior-only DAS` 不读取 query feature 列。
- [ ] `Always Query` 的 Query 次数恒为 1，DBF 可为 0 或 1。
- [ ] baseline 的 run-level 结果可以直接进行配对统计检验。

### 依赖

P0.1/P0.2 的标签与 action outcome contract。

---

## P0.4 严格在线协议：first-trigger、no-query leakage、semantic RNG

### 目标

保证离线 replay 与在线 simulator 语义一致，且未触发 Query 的 run 不会获得任何 Query 信息。

### 涉及文件

- `decision/online_controller_evaluate.py`
- `decision/query_contract.py`
- `decision/sampling_opportunities.py`
- `landscape_queries/sampling.py`
- `trajectory/query.py`
- `optimizers/seeding.py`
- `experiments/cli/tiny_end_to_end_check.py`
- 建议新增：`decision/protocol_checks.py`

### 实施任务

- [ ] 为每次在线 run 维护 `query_count`，Query 后立即锁定，不允许第二次触发。
- [ ] 记录 `first_trigger_state_key`、`first_trigger_FE`、`trigger_score`、`trigger_threshold`。
- [ ] 当 controller 未触发时，query sample generation 函数不得被调用。
- [ ] 记录 query generation、objective evaluation、feature extraction 三类计数，并在 no-query run 中验证为零。
- [ ] 实现 semantic query seed：
  `h(problem_id, dimension, optimizer_seed, trigger_FE, query_id)`。
- [ ] 明确 Query RNG stream 与 optimizer continuation RNG stream 的不同 code。
- [ ] 增加 replay test：相同 state + 相同 protocol 得到相同 samples/features；插入 Query 不改变 continuation RNG 序列。
- [ ] 增加 offline/online first-trigger 对齐检查。

### 验收标准

- [ ] `query_count <= 1` 对每条 run 恒成立。
- [ ] no-query run 的 query sample、objective eval、feature access 计数均为 0。
- [ ] Query seed 在输出数据中可追溯。
- [ ] 相同状态重放得到 bitwise 或协议允许误差内一致的 Query 结果。
- [ ] Query 前后 optimizer continuation 的随机序列隔离测试通过。

---

## P0.5 Decision opportunity range pilot

### 目标

验证当前冻结的 `0.20–0.60` 监测范围是否遗漏早期或晚期高价值 Query state。

### 涉及文件

- `trajectory/sampling.py`
- `decision/sampling_opportunities.py`
- `experiments/cli/phase1_collect_batch.py`
- `configs/phase1_bbob_pilot.yaml`
- 建议新增：`configs/phase1_opportunity_range_pilot.yaml`
- 建议新增：`experiments/cli/opportunity_range_pilot.py`

### 实施任务

- [ ] 增加 pilot-only monitor grid：覆盖 `0.10–0.70`。
- [ ] 正式冻结协议保持不变，扩展 grid 必须使用独立 protocol ID。
- [ ] 在相同 prefix trajectory 上比较不同 grid 的 state coverage。
- [ ] 计算每个区间的：state 数、positive efficacy rate、median efficacy、label variance、first-trigger distribution。
- [ ] 分析 milestone-only、event-only、milestone+event 三类 opportunity。
- [ ] 根据 pilot 结果决定正式范围；决定过程只能使用 train/pilot，不能使用外部 test。

### 验收标准

- [ ] 产出 opportunity coverage report。
- [ ] 报告至少包含 `0.10–0.20`、`0.20–0.60`、`0.60–0.70` 的比较。
- [ ] 扩展协议没有影响默认 formal config。
- [ ] 每个 opportunity row 有准确的实际 FE、FE ratio、native update 数和触发原因。

---

## P0.6 Coverage-driven MA-BBOB formal subset

### 目标

用 landscape + observed behavior + action discrimination 选择 MA-BBOB formal subset，替代单纯按 candidate ID 选样。

### 涉及文件

- `benchmarks/mabbob.py`
- `behavior/batch_extraction.py`
- `behavior/extraction.py`
- `behavior/features.py`
- `decision/pilot_coverage.py`
- `configs/phase1_mabbob_pilot.yaml`
- `configs/phase1_mabbob_pilot_101_200.yaml`
- `configs/phase1_mabbob_formal.yaml`
- 建议新增：`experiments/cli/select_mabbob_formal_subset.py`

### 实施任务

- [ ] 为每个 candidate 输出结构 metadata：components、weights、mixture arity、bridge type、dominant component。
- [ ] 为每个 candidate 汇总跨算法 behavior representation，保留 early/middle/late 信息。
- [ ] 计算 action discrimination：各 state 的 action loss spread、best-vs-worst gap、acceptable action count。
- [ ] 对 landscape、behavior、action discrimination 分别标准化，再按冻结权重拼接。
- [ ] 实现 farthest-point/maximin selection，支持分层约束。
- [ ] 选择时保留 pairwise bridge、sparse 3-way mixture 和不同 weight profile 的代表。
- [ ] 输出 selection manifest，记录候选集、特征协议、距离度量、seed、选中理由。
- [ ] formal config 只引用 manifest 中选中的 candidate IDs，不再手工维护不可解释编号。

### 验收标准

- [ ] formal subset 的 selection 可以由 manifest 完全复现。
- [ ] 选中集覆盖 landscape、behavior、action discrimination 三个空间。
- [ ] train / validation component leakage audit 通过。
- [ ] selection 不使用 Query efficacy labels 或外部 test 数据。
- [ ] formal config 与 manifest 的 candidate IDs 完全一致。

---

# P1：应尽快完成

## P1.1 Multi-horizon efficacy

### 涉及文件

- `utility_labels/efficacy.py`
- `utility_labels/generation.py`
- `utility_labels/fields.py`
- `utility_labels/validation.py`
- `trajectory/final_performance.py`

### 实施任务

- [ ] 定义 horizon protocol，例如 `H ∈ {100D, 200D, 500D, final}`。
- [ ] 每个 horizon 采用 equal-FE：Query branch 的 Query FE 必须从 continuation FE 中扣除。
- [ ] 保存 `g_fe_h100d`、`g_fe_h200d`、`g_fe_h500d`、`g_fe_final`。
- [ ] 对每个 horizon 计算 MAE、R²、Spearman、sign persistence。
- [ ] 计算 `P_persist(H) = P[sign(g_H)=sign(g_final)]`。
- [ ] 检查 delayed payoff：短期负、最终正的比例。

### 验收标准

- [ ] 所有 horizon 通过 equal-FE check。
- [ ] horizon 不足以支付 Query cost 的 state 被明确排除或标记。
- [ ] final efficacy 仍是主标签。
- [ ] 产出 multi-horizon analysis report。

---

## P1.2 维度分层：10D / 20D / 40D

### 涉及文件

- `configs/phase1_bbob_train.yaml`
- `configs/phase1_bbob_validation.yaml`
- `configs/phase1_mabbob_formal.yaml`
- `experiments/phase1_batch_common.py`
- `benchmarks/mabbob.py`
- `experiments/cli/phase1_check_config.py`

### 实施任务

- [ ] 明确 `10D` 为开发/主实验，`20D` 为泛化，`40D` 为 stress test。
- [ ] 为不同维度建立独立 output root，避免 parquet 混写。
- [ ] 模型训练和 scaler 只从声明的 train dimensions fit。
- [ ] stress test 只使用冻结模型，不重新校准阈值或 selector。
- [ ] 报告维度内、跨维度和 stress test 三类结果。

### 验收标准

- [ ] 配置校验拒绝未声明维度或预算不匹配的输入。
- [ ] 40D 数据不会进入只声明 10D/20D 的训练过程。
- [ ] 外部维度测试不触发任何 fit/recalibration。

---

## P1.3 Acceptable action set 与 selector regret

### 涉及文件

- `selection_reference/action_losses.py`
- `selection_reference/model.py`
- `selection_reference/build.py`
- `utility_labels/generation.py`
- `utility_labels/fields.py`
- `decision/online_controller_evaluate.py`

### 实施任务

- [ ] 定义 practical action tolerance `delta_a`，只从 train calibration 得到。
- [ ] 对每个 state 计算：
  `A_acc = {a: L_a - L_min <= delta_a}`。
- [ ] 输出 `acceptable_action`、`acceptable_action_count`、`acceptable_accuracy`。
- [ ] 输出 `selector_regret_raw = L_selected - L_min`。
- [ ] 同时报告 top-1 accuracy、acceptable accuracy、mean/median regret。
- [ ] 若有 repetition，acceptable set 应基于稳健 action loss 或等价性检验。

### 验收标准

- [ ] action tolerance 只使用 train 数据拟合。
- [ ] 外部 test 不重新选择 `delta_a`。
- [ ] selector 输出中包含 selected action、best action、regret 和 acceptable flag。
- [ ] 近似等价算法不会被硬性标为错误 action。

---

## P1.4 OOD 与 component leakage audit

### 涉及文件

- `benchmarks/bbob.py`
- `benchmarks/mabbob.py`
- `experiments/phase1_batch_common.py`
- `decision/nested_learning.py`
- `decision/external_test_predict.py`
- `experiments/cli/phase1_validate.py`
- `experiments/cli/phase1_check_config.py`

### 实施任务

- [ ] 建立 split manifest：BBOB train、BBOB validation、CEC test、MA-BBOB train/validation。
- [ ] 对每个 MA-BBOB candidate 展开 component set，做集合交叉检查。
- [ ] 禁止 validation component 出现在 train mixture 中。
- [ ] 记录 model fit scope、scaler fit scope、threshold fit scope、selector fit scope。
- [ ] 外部 test 只允许 `predict/evaluate`，禁止 fit、recalibrate、feature selection。
- [ ] 输出 leakage audit report 和失败 candidate 列表。

### 验收标准

- [ ] 所有 split 的 component intersection 符合协议。
- [ ] CEC/外部测试日志中没有 fit 或 recalibration 操作。
- [ ] 任一泄漏都能在正式数据生成前使 pipeline 失败。

---

## P1.5 标签噪声审计

### 涉及文件

- `utility_labels/batch_generation.py`
- `utility_labels/validation.py`
- `utility_labels/generation.py`
- 建议新增：`experiments/cli/label_noise_report.py`

### 实施任务

- [ ] 汇总 `g_fe` 的 mean、median、std、quantiles。
- [ ] 汇总 `sign_flip_rate` 与 high-uncertainty state 比例。
- [ ] 汇总 median 与 mean 的差异。
- [ ] 汇总 CI width 与 opportunity phase、algorithm、dimension、family 的关系。
- [ ] 统计单次标签与重复稳健标签的 sign agreement。
- [ ] 对高噪声 state 做单独 sensitivity analysis，不直接删除。

### 验收标准

- [ ] report 至少按 split、dimension、prefix algorithm、sampling phase 分组。
- [ ] 明确正式训练采用的聚合方法。
- [ ] 高噪声标签有数量、比例和处理策略记录。

---

# P2：可扩展项

## P2.1 Skip / Defer / Query 三向控制

### 涉及文件

- `decision/skip_defer_query.py`
- `decision/online_controller_evaluate.py`
- `decision/model_protocol.py`
- `decision/compare_controller_baselines.py`

### 进入条件

- P0 gate 已通过。
- single-query protocol 已稳定。
- defer 的 FE 语义和重复触发上限已定义。

### 任务

- [ ] 定义 `skip`、`defer`、`query` 的状态机。
- [ ] 明确 defer 后允许的下一次机会和最大次数。
- [ ] 为三向动作定义独立标签和评估指标。
- [ ] 先做 ablation，不改变 DBF 主协议。

---

## P2.2 Conformal / quantile / fuzzy calibration

### 涉及文件

- `decision/conformal.py`
- `decision/practical_delta.py`
- `decision/schedule_threshold.py`
- `decision/model_protocol.py`

### 进入条件

- P0.2 robust label contract 已冻结。
- P1.5 已确认标签噪声和 calibration 需求。
- train/validation/test calibration scope 已审计。

### 任务

- [ ] 增加 efficacy prediction interval 或 quantile output。
- [ ] 在 train/validation 上选择 coverage/threshold protocol。
- [ ] CEC/test 只调用冻结 calibration artifact。
- [ ] 报告 coverage、abstention、query rate 和 final performance。

---

## P2.3 Diversify / restart action

### 涉及文件

- 新增或扩展 `actions/continue_action.py`
- 新增或扩展 `actions/switch_action.py`
- `optimizers/state.py`
- 各算法实现文件
- `selection_reference/action_losses.py`
- `selection_reference/model.py`

### 进入条件

- P0 baseline action semantics 已冻结。
- `continue_current` 与 `switch_to_X` 的结果稳定。
- 可以为 DE/SHADE、PSO、CMA-ES 定义语义正确的 restart/diversify。

### 任务

- [ ] 为 DE/SHADE 定义 partial population restart 或 jitter。
- [ ] 为 PSO 同时处理 position、velocity、personal/global best。
- [ ] 为 CMA-ES 定义 step-size/covariance/distribution restart，而非只扰动 population。
- [ ] 将 diversify 作为独立 action ID。
- [ ] Query 与 behavior-only 两侧使用相同 action set。

---

## P2.4 外部 benchmark 与工程问题

### 涉及文件

- `configs/phase1_cec2017_test.yaml`
- `configs/prospective_suites.yaml`
- `decision/external_test_predict.py`
- `experiments/cli/pilot_online_eval.py`

### 进入条件

- P0/P1 所有 hard gate 通过。
- 模型、scaler、threshold、selector、query protocol 均已冻结。
- 外部测试路径已经通过 leakage audit。

### 任务

- [ ] CEC2017 只执行 frozen predict/evaluate。
- [ ] 分别报告 seed、instance、function-family、dimension OOD。
- [ ] 记录 query FE、query call rate、wall-clock 和 final performance。
- [ ] 后续再增加 CEC2022 或 engineering benchmark。

---

# 推荐实施顺序与状态表

## P0

| ID | 任务 | 主要文件 | 依赖 | 状态 |
|---|---|---|---|---|
| P0.1 | paired continuation 与稳健标签 | `utility_labels/*`, `trajectory/*` | 无 | [ ] |
| P0.2 | 固化 efficacy label contract | `utility_labels/*`, `decision/train_full_decision_model.py` | P0.1 | [ ] |
| P0.3 | Behavior-only / switch baselines | `decision/*`, `selection_reference/*` | P0.2 | [ ] |
| P0.4 | first-trigger / leakage / RNG checks | `decision/online_controller_evaluate.py`, `trajectory/query.py` | P0.2 | [ ] |
| P0.5 | opportunity range pilot | `trajectory/sampling.py`, `configs/*` | P0.1 | [ ] |
| P0.6 | MA-BBOB coverage-driven selection | `benchmarks/mabbob.py`, `decision/pilot_coverage.py` | P0.1, P0.5 | [x] |

## P1

| ID | 任务 | 主要文件 | 依赖 | 状态 |
|---|---|---|---|---|
| P1.1 | multi-horizon efficacy | `utility_labels/*` | P0.1/P0.2 | [ ] |
| P1.2 | 10D/20D/40D 分层 | `configs/*`, `phase1_batch_common.py` | P0.4 | [ ] |
| P1.3 | acceptable action + regret | `selection_reference/*`, `utility_labels/*` | P0.3 | [ ] |
| P1.4 | OOD/component leakage audit | `benchmarks/*`, `decision/nested_learning.py` | P0.6 | [ ] |
| P1.5 | label noise audit | `utility_labels/batch_generation.py` | P0.1 | [ ] |

## P2

| ID | 任务 | 主要文件 | 依赖 | 状态 |
|---|---|---|---|---|
| P2.1 | skip/defer/query | `decision/skip_defer_query.py` | P0 全部 | [ ] |
| P2.2 | conformal/quantile/fuzzy | `decision/conformal.py` | P0.2, P1.5 | [ ] |
| P2.3 | diversify/restart | `optimizers/state.py`, `actions/*` | P0.3 | [ ] |
| P2.4 | CEC/工程 benchmark | `configs/*`, `decision/external_test_predict.py` | P0/P1 hard gates | [ ] |

---

# 每阶段交付物

## P0 交付物

- `efficacy_label_protocol_v2` 数据合同
- paired continuation replay 数据
- robust efficacy label parquet
- baseline protocol 与 run-level results
- online protocol check report
- opportunity range pilot report
- ✅ MA-BBOB coverage selection manifest (`results/mabbob_diversity_pilot/mabbob_formal_selection_manifest.json`)
- formal data generation gate report

## P1 交付物

- multi-horizon efficacy report
- dimension generalization report
- acceptable action / selector regret report
- OOD/component leakage audit report
- label noise audit report

## P2 交付物

- 扩展动作或控制策略的独立 ablation report
- frozen calibration artifact
- external benchmark evaluation report

---

# 正式数据生成前的最终检查表

- [ ] 标签主定义仍为 final equal-FE `G_FE`。
- [ ] paired repetition 数与 aggregation 已冻结。
- [ ] Query cost 已计入总 FE，所有路径 exact equal-FE。
- [ ] Behavior-only DAS 是强 baseline，不使用 query features。
- [ ] Fixed/Random/SwitchBenefit baseline 已运行并完成公平性检查。
- [ ] Query 未触发时没有任何 query 数据生成或访问。
- [ ] 每条 run 最多一次 Query。
- [ ] Query RNG 和 continuation RNG 隔离。
- [ ] opportunity range pilot 已完成，正式 grid 有依据。
- [x] MA-BBOB subset 由可复现 manifest 选出。
- [ ] train/validation/test component leakage audit 通过。
- [ ] 所有 scaler、threshold、selector、calibration artifact 已冻结。
- [ ] P1 报表所需字段已经进入数据合同。

> 最后更新：2026-08-20
> 版本：P0–P2 executable implementation plan v1
