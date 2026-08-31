Task17B 是零新增目标函数评估的机制分解。所有正式区间均按 `cv_group_id` 分组 bootstrap 5000 次；`new_objective_FE = 0`。

# 17b_08 Full-vector residual decomposition

| domain       | suite   |   t3_states |   t3_cv_groups | track_status                        |   AVA_noise |   AVA_local_T3 |   E_res_full |
|:-------------|:--------|------------:|---------------:|:------------------------------------|------------:|---------------:|-------------:|
| post_handoff | bbob    |           6 |              3 | UNAVAILABLE_INSUFFICIENT_T3_SUPPORT |         nan |            nan |          nan |
| post_handoff | mabbob  |           0 |              0 | UNAVAILABLE_INSUFFICIENT_T3_SUPPORT |         nan |            nan |          nan |
| post_handoff | pooled  |           6 |              3 | UNAVAILABLE_INSUFFICIENT_T3_SUPPORT |         nan |            nan |          nan |
| natural      | bbob    |           0 |              0 | UNAVAILABLE_INSUFFICIENT_T3_SUPPORT |         nan |            nan |          nan |
| natural      | mabbob  |           1 |              1 | UNAVAILABLE_INSUFFICIENT_T3_SUPPORT |         nan |            nan |          nan |
| natural      | pooled  |           1 |              1 | UNAVAILABLE_INSUFFICIENT_T3_SUPPORT |         nan |            nan |          nan |

T3 未达到门槛，因此不计算 full-vector residual 或 decision-geometry repeatability reference。
