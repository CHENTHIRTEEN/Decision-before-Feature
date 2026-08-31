Task17B 是零新增目标函数评估的机制分解。所有正式区间均按 `cv_group_id` 分组 bootstrap 5000 次；`new_objective_FE = 0`。

# 17b_02 Repetition topology 与 RNG 核查

| domain       | component_type    | action_or_pair   |   t1_cells |   t2_states |   t3_states |   cv_groups |   decision_strata | outcome_blind_selection   | formal_support   | support_status   |
|:-------------|:------------------|:-----------------|-----------:|------------:|------------:|------------:|------------------:|:--------------------------|:-----------------|:-----------------|
| post_handoff | T1_action         | shade            |        394 |           0 |           6 |          22 |               306 | YES                       | True             | DESCRIPTIVE_T1   |
| post_handoff | T1_action         | lshade           |        396 |           0 |           6 |          22 |               306 | YES                       | True             | DESCRIPTIVE_T1   |
| post_handoff | T1_action         | cso              |        352 |           0 |           6 |          22 |               300 | YES                       | True             | DESCRIPTIVE_T1   |
| post_handoff | T2_pair           | shade|lshade     |        790 |          40 |           6 |          14 |                40 | YES                       | True             | FORMAL           |
| post_handoff | T2_pair           | shade|cso        |        746 |          46 |           6 |          12 |                44 | YES                       | True             | FORMAL           |
| post_handoff | T2_pair           | lshade|cso       |        748 |          40 |           6 |          13 |                40 | YES                       | True             | FORMAL           |
| post_handoff | T3_full_signature | shade|lshade|cso |       1142 |           0 |           6 |           3 |                 6 | YES                       | False            | UNAVAILABLE      |
| natural      | T1_action         | shade            |        179 |           0 |           1 |          22 |               153 | YES                       | True             | DESCRIPTIVE_T1   |
| natural      | T1_action         | lshade           |        186 |           0 |           1 |          22 |               150 | YES                       | True             | DESCRIPTIVE_T1   |
| natural      | T1_action         | cso              |        192 |           0 |           1 |          22 |               157 | YES                       | True             | DESCRIPTIVE_T1   |
| natural      | T2_pair           | shade|lshade     |        365 |          18 |           1 |          14 |                17 | YES                       | False            | LOW_SUPPORT      |
| natural      | T2_pair           | shade|cso        |        371 |          22 |           1 |          13 |                22 | YES                       | False            | LOW_SUPPORT      |
| natural      | T2_pair           | lshade|cso       |        378 |          19 |           1 |          11 |                19 | YES                       | False            | LOW_SUPPORT      |
| natural      | T3_full_signature | shade|lshade|cso |        557 |           0 |           1 |           1 |                 1 | YES                       | False            | UNAVAILABLE      |

## RNG 与 clone 语义

| check_id                     | status               | detail                                                                                                     |
|:-----------------------------|:---------------------|:-----------------------------------------------------------------------------------------------------------|
| outcome_blind_selection      | PASS                 | sampling uses SeedSequence coordinates only; no action outcome enters selection                            |
| source_state_identity        | PASS                 | every branch starts from the same saved population, fitness, best value, and FE state                      |
| continue_clone               | PASS                 | continue uses an independent full-state clone before installing its repetition RNG                         |
| transfer_initialization      | PASS_WITH_SCOPE_NOTE | target initialization reuses the same source state; CSO velocities vary only through the target action RNG |
| semantic_rng_separation      | PASS                 | replicate and target-action coordinates select disjoint semantic RNG streams                               |
| cross_action_coupling        | PASS_BY_CONSTRUCTION | target solver stream codes differ; no shared mutable optimizer state is reused                             |
| empirical_independence_claim | NOT_CLAIMED          | stream separation is verified from code; finite PRNG outcomes do not prove probabilistic independence      |
| primary_eligibility          | PASS                 | all post-handoff T1/T2 rows satisfy the source-state and RNG design contract                               |

CSO transfer 的初始 velocity 由 target-action RNG 生成，因此 target 内部数组不逐字相同；差异完全由预先指定的 action RNG 产生，source population/fitness/best/FE 保持相同。该随机初始化属于被重复测量的动作核。
