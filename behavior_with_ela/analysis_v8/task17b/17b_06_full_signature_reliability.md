Task17B 是零新增目标函数评估的机制分解。所有正式区间均按 `cv_group_id` 分组 bootstrap 5000 次；`new_objective_FE = 0`。

# 17b_06 Full-signature reliability

| domain       | suite   |   t3_states |   t3_cv_groups | track_status                        | reason                                                 |
|:-------------|:--------|------------:|---------------:|:------------------------------------|:-------------------------------------------------------|
| post_handoff | bbob    |           6 |              3 | UNAVAILABLE_INSUFFICIENT_T3_SUPPORT | pooled post-handoff requires 50 states and 8 cv groups |
| post_handoff | mabbob  |           0 |              0 | UNAVAILABLE_INSUFFICIENT_T3_SUPPORT | pooled post-handoff requires 50 states and 8 cv groups |
| post_handoff | pooled  |           6 |              3 | UNAVAILABLE_INSUFFICIENT_T3_SUPPORT | pooled post-handoff requires 50 states and 8 cv groups |
| natural      | bbob    |           0 |              0 | UNAVAILABLE_INSUFFICIENT_T3_SUPPORT | pooled post-handoff requires 50 states and 8 cv groups |
| natural      | mabbob  |           1 |              1 | UNAVAILABLE_INSUFFICIENT_T3_SUPPORT | pooled post-handoff requires 50 states and 8 cv groups |
| natural      | pooled  |           1 |              1 | UNAVAILABLE_INSUFFICIENT_T3_SUPPORT | pooled post-handoff requires 50 states and 8 cv groups |

未生成 synthetic signature，也未把模型化噪声当作直接证据。
