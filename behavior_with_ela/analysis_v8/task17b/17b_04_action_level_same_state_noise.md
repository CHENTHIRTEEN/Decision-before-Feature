Task17B 是零新增目标函数评估的机制分解。所有正式区间均按 `cv_group_id` 分组 bootstrap 5000 次；`new_objective_FE = 0`。

# 17b_04 Action-level same-state noise

| action   |   median |    q75 |    q90 |     q95 |   P_N_gt1 |   repeated_states |   cv_groups | repetition_weight_sensitive   |
|:---------|---------:|-------:|-------:|--------:|----------:|------------------:|------------:|:------------------------------|
| cso      |   0.3441 | 2.3370 | 6.3879 | 11.1504 |    0.3617 |               352 |          22 | False                         |
| lshade   |   0.4203 | 1.7557 | 3.8305 |  5.6442 |    0.3577 |               396 |          22 | False                         |
| shade    |   0.5727 | 1.9776 | 4.2440 |  6.2579 |    0.3841 |               394 |          22 | False                         |

所有 unordered repetition pairs 作描述；每 state 固定 key 0/1 的敏感性结果保存在同一 summary parquet。
