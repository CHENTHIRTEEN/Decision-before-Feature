# Decision-before-Feature Behavior Feature Taxonomy 与指标选择协议

> 唯一活动协议（2026-08-14）。Decision X 只由 query 前可得、算法无关且对 population 行排列不变的 Behavior 构成。旧跨 checkpoint 行号对应、四组消融和结果后特征筛选全部退出。

## 1. 构念与信息边界

Behavior 描述当前 optimizer 产生的可观测搜索过程，不描述 benchmark 身份或 optimizer 内部机制。允许使用当前及过去完整 native updates 的 population 集合、fitness 分布、best-so-far、实际 FE 与预算；禁止读取未来状态。

以下内容不得进入 Decision X：

- query features、query samples 或 query status；
- function/suite ID、dimension、instance；
- prefix/default/selected algorithm ID；
- optimizer-specific parameters、internal dynamics 或 RNG state；
- benchmark reference、known optimum、gap；
- observed action losses、Selector predictions/regret、Utility；
- nominal milestone/event label、window metadata。

算法 identity 可作分层 metadata。称为 algorithm-agnostic 只表示输入不显式编码算法，不保证不同 optimizer 产生相同 Behavior 分布。

## 2. 完整 native-update 窗口

名义窗口为 `w02=0.02`、`w05=0.05`、`w10=0.10` 的总 FE 比例。对当前 emitted state (FE_t) 和名义跨度 (W)，目标 anchor 为：

\[
FE_t-\operatorname{round}(W\,FE_{total}).
\]

若目标不是完整 native update 边界，选择不晚于目标的最近完整 update。因此实际窗口不得小于名义窗口，且偏差严格小于一次 population update：

```text
round(W*FE_total) <= effective_window_fe
effective_window_fe < round(W*FE_total) + population_size
```

anchor 必须来自逐次完整 native-update history，不能从稀疏正式 decision checkpoints 选择。所有 rate/slope 使用实际 `effective_window_fe/FE_total`。以下字段只作计算来源 metadata，不进入 Decision X：

```text
effective_window_ratio_w02/w05/w10
effective_window_fe_w02/w05/w10
effective_native_updates_w02/w05/w10
```

## 3. 活动字段

### 3.1 T0/base（9 字段）

```text
bf_fe_ratio
bf_improvement_rate_w02
bf_improvement_frequency_w02
bf_diversity_mean_pairwise
bf_diversity_change_w05
bf_covariance_spectral_concentration
bf_distance_decay_w10
bf_stagnation_w10
bf_convergence_rate_w10
```

其中 `bf_fe_ratio` 必须逐行等于实际 `FE/FE_total`。fitness 变化用初始化已评价 population 的 fitness IQR 作 shift-invariant robust scale；不得使用 known optimum。

### 3.2 Primary set/distribution fields（10 字段）

```text
bf_fitness_diversity_rel
bf_population_wasserstein_rate_w05
bf_centroid_shift_rate_w05
bf_centroid_shift_coherence_w05
bf_fitness_quantile_improvement_fraction_w02
bf_fitness_distribution_improvement_rate_w02
bf_fitness_wasserstein_rate_w02
bf_elite_concentration
bf_best_fitness_slope_rel_w05
bf_diversity_slope_w05
```

population/fitness 跨窗口比较把两端视为经验集合或经验分布，不假定同一行在跨代代表同一个体。

### 3.3 Longitudinal set dynamics（6 字段）

```text
bf_fitness_spread_slope_w05
bf_population_centroid_shift_w05
bf_elite_centroid_shift_w05
bf_covariance_trace_ratio_w05
bf_covariance_effective_rank_w05
bf_diversity_recovery_w05
```

这些 summaries 借鉴 trajectory-based population dynamics 的一般思想，但不是对任一既有方法的完整复现。论文只写“inspired/adapted”，不声称直接使用既有指标集合。

### 3.4 Set-motion（3 字段）

```text
bf_population_chamfer_distance_w05
bf_covariance_trace_change_w05
bf_covariance_effective_rank_change_w05
```

三项都基于集合形状变化，不依赖跨代个体对应。

### 3.5 Maturity basis（3 字段）

```text
bf_search_maturity
bf_search_maturity_linear
bf_explore_exploit_ratio
```

它们是既有 Behavior 的确定性变换，不增加原始信息。`bf_search_maturity=ES_t(1-XS_t)` 不定义真实阶段、不要求随 FE 单调，也不预设与 Utility 的曲线形状。其作用只通过预设消融评价。

### 3.6 Diagnostic-only（3 字段）

```text
bf_fitness_diversity
bf_population_overlap_w05
bf_best_distance_fitness_corr
```

这些字段可以用于误差分析，但不进入任一正式 Decision/Selector feature group。总输出为 34 个唯一 `bf_*` 字段，其中 31 个是活动输入、3 个仅诊断。

## 4. 冻结 feature groups

| 组 | 组成 | 字段数 |
|---|---|---:|
| T0 | 仅 `bf_fe_ratio` | 1 |
| B1 | base + primary | 19 |
| B2 | B1 + longitudinal set dynamics | 25 |
| B2+Motion | B2 + set-motion | 28 |
| B2+Maturity | B2 + Maturity basis | 28 |
| B3 | B2 + set-motion + Maturity basis | 31 |

兼容名称只作解析：

```text
time_only -> T0
primary -> B1
primary_with_dynamorep_lite -> B2
primary_with_movement -> B2+Motion
primary_with_maturity -> B2+Maturity
all_candidates -> B3
```

兼容名不能产生第七组；`all_candidates` 严格等于 B3，不含 diagnostic-only 字段。

## 5. Time-only 公平比较

主 RQ2 的 T0 只在 12 个固定预算 milestones 上拟合、threshold 与评价，称为 `milestone_only_T0`。用于直接比较的 B3 必须限制到相同 milestone rows。事件机会的出现由 Behavior 条件决定，因此完整 dynamic schedule 上的 `schedule_conditioned_T0` 只能作 sensitivity，不能支持“Behavior 超出时间”的主结论。另设按 dimension 预先分层、每层仍只输入 `bf_fe_ratio` 的 `dimension_stratified_T0`，控制调用前已知的 dimension/绝对预算上下文而不把 dimension 加入主 Decision X。

## 6. 模型与消融

六组使用同一个由 B3 outer-function OOF first-trigger Utility 选定的模型名。每组在完整 outer/inner function 链中独立拟合 train-only preprocessing，并用自身端到端 train OOF score冻结 first-trigger threshold；不得让各组重新选择最有利模型。

预设 RQ5 contrasts 为：

```text
B1 - T0
B2 - B1
B2+Motion - B2
B2+Maturity - B2
B3 - (B2+Motion)
B3 - (B2+Maturity)
```

这些 contrasts 检验预测增量，不检验因果贡献。不得按 observed result 删除兄弟组、修改 Maturity 公式、增加交互项或筛选单列。

## 7. Missingness 与 preprocessing

每个 fit fold 内用训练行中位数 impute，再标准化；holdout、BBOB-validation 与外部数据不参与 imputer/scaler。训练 fold 内整列缺失阻止该组拟合，不能用全局常数或 validation 值填充。缺失率和触发来源按字段/function/dimension 报告。

## 8. 一致性检查

活动 Behavior 表必须满足：

- 状态键与 trajectory 双向覆盖；
- `bf_fe_ratio=FE/FE_total`；
- 三个实际窗口不少于名义跨度且误差小于一次 update；
- w02/w05/w10 anchors 均为 retained complete native updates；
- population 排列后全部正式字段数值不变；
- 不读取未来 native update、query、gap、action loss 或算法内部字段；
- 六组列集合与 1/19/25/28/28/31 计数严格一致。

任一失败都要求从相应 trajectory/Behavior shard 重生成；不得通过删除行或修改输入组绕过。

## 9. 结果解释

BBOB-validation 已被旧流程查看；即使 B3 在该固定六函数有限集上超出 milestone-only T0 与 dimension-stratified T0，也只能作为开发期条件证据，不能称独立确认。train outer OOF 同样只作开发诊断。Maturity contrasts 超出 B2 只支持“预定义确定性基函数对固定模型有用”。如果区间不支持差异或落入项目内 operational tolerance，应报告差异未建立；不得预设 Behavior 或 Maturity 必然有效。
