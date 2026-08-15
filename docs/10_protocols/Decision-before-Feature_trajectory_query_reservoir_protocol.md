# Decision-before-Feature trajectory query reservoir protocol

> 活动诊断协议（2026-08-14）。Reservoir 只保存已评价点流的固定容量代表性子集，用于零额外 FE 的 trajectory-query 诊断。它不是完整评价档案，也不替代主 `descriptor_cheap_invariant` 独立 LHS query。

## 1. 用途与 estimand 隔离

每个 optimizer run 在 objective value 返回时，把已评价点传给 online reservoir，不触发额外函数评价。Reservoir snapshot 可用于 trajectory-derived descriptors、零额外 FE 诊断和与独立 query 的敏感性比较。

该诊断必须与主独立 query 分开：

```text
query_id = descriptor_cheap_invariant
query_source_mode = trajectory_reservoir_zero_extra_fe
query_protocol = trajectory_query_reservoir_v1
query_preprocessing_id = unit_cube_x__median_iqr_y_v1
reservoir_size = 50 * dimension
```

相同 `query_id` 只表示同一 16 列 descriptor 定义；`query_source_mode` 与 `query_protocol` 区分独立 LHS acquisition 和 trajectory reservoir。两者的 FE、样本分布、Utility 与结论不能合并。Reservoir 不进入第一篇论文的主 Query/full-Selector estimand。

## 2. Replacement randomness

对第 (i) 个已评价点使用标准 reservoir replacement sampling。随机状态只能由显式整数输入构造：

```text
base_seed
unit_number
stream_code
generation
target_code
event_code
suite_code
function_number
instance_number
dimension
algorithm_code
run_seed
```

这些整数直接交给 `numpy.random.SeedSequence`。不得对 `problem_id`、`family`、algorithm 字符串或 `query_protocol` 做字符串哈希，也不得使用 Python `hash()`、`hashlib` 或“semantic seed”。同一已冻结整数映射与 base seed 必须重现相同 replacement stream。

## 3. 与 native update 的对齐

Reservoir 按 objective evaluation 更新，但 snapshot 只在 trajectory collector 实际 emitted 的完整 native-update state 上写出。每个 snapshot 必须对应同一行的实际整数 `FE`；`FE_ratio=FE/FE_total`，跨表只以整数 `FE` 连接。

若 wall-clock timeout 或其他停止条件发生在一次 population/native update 中间：

1. 不写出部分 update 的 optimizer state；
2. trajectory 与 reservoir 都回到最后一个完整 native update 的 emitted state；
3. 保留实际已耗评价数、timeout 与 failure metadata，但不把部分 state 伪装为可续跑 decision state；
4. 若没有新的完整 update，则不新增 snapshot。

`FE_total` 的完整预算 outcome 保存在 `final_performance.parquet`，不要求也不允许为了对齐终值而伪造一个 reservoir/decision snapshot。decision trajectory 的最后 emitted state 可以早于 `FE_total`。

## 4. Snapshot 字段

每行至少保存：

```text
split
problem_id
family
function
instance
dimension
algorithm
seed
FE
FE_ratio
FE_total
native_updates
query_id
query_source_mode
query_protocol
query_preprocessing_id
query_feature_columns
trajectory_query_reservoir_size
trajectory_query_seen_count
trajectory_sample_count
trajectory_sample_coverage_ratio
reservoir_stream_code
reservoir_event_code
trajectory_query_runtime
feature_status
feature_count
feature_failure
feature_group_status
feature_nonfinite
```

14 个 descriptor columns 与同一行保存，必须严格匹配 `descriptor_cheap_invariant` whitelist；已删除的两个归一化恒量不得重新生成。`trajectory_query_seen_count` 是截至该 emitted FE 已交给 reservoir 的评价点数；它不得小于 sample count，也不得大于实际评价流计数。

## 5. 一致性检查

活动数据必须满足：

- 每个 snapshot 的状态键在 trajectory 中恰好存在一行；
- snapshot `FE` 是整数完整 update 边界，ratio 可由它重算；
- snapshot 不超前读取未来评价点；
- reservoir 不增加 FE；
- sample count 不超过 `50D`，seen count 单调不减；
- time truncation 后只保留最后完整 update；
- `FE_total` 只来自真实 emitted state，不由终值表补造；
- replacement RNG 只使用冻结整数 SeedSequence inputs；
- independent-LHS 与 reservoir 产物由 `query_source_mode/query_protocol` 隔离。

任一键缺失、部分 update 被写成 state、字符串 seed 参与 RNG 或未来点评价泄漏时，相应 run 的 reservoir 诊断失效并须重生成。

## 6. 解释限制

Reservoir 是评价点流的随机子样本，不等于完整 prefix archive。它受 optimizer 采样分布影响，因此基于它的 descriptor 不是算法无关的独立 landscape sample。它可以回答“既有评价流是否提供有用的 trajectory descriptors”，但不能回答独立 landscape query 是否值得额外采样，也不能把零额外 FE 结果与主 query Utility 直接比较为同一 estimand。
