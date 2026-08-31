Task17B 是零新增目标函数评估的机制分解。所有正式区间均按 `cv_group_id` 分组 bootstrap 5000 次；`new_objective_FE = 0`。

# 17b_09 Decision-relevant state 核查

U0=0.5471，U1=0.4529，U2=0.2243，U3=0.1902。

| subset   | action_pair   |   states |   strata_with_pairs |   state_fraction |   A_local |   median_D_local |   winner_mismatch_rate |
|:---------|:--------------|---------:|--------------------:|-----------------:|----------:|-----------------:|-----------------------:|
| U1       | shade|lshade  |     1712 |            471.0000 |           0.4529 |    0.7197 |           2.4203 |                 0.3418 |
| U1       | shade|cso     |     1712 |            471.0000 |           0.4529 |    0.7643 |           2.8569 |                 0.3418 |
| U1       | lshade|cso    |     1712 |            471.0000 |           0.4529 |    0.7367 |           2.3083 |                 0.3418 |
| U3       | shade|lshade  |      719 |            199.0000 |           0.1902 |    0.7035 |           1.9653 |                 0.2261 |
| U3       | shade|cso     |      719 |            199.0000 |           0.1902 |    0.7839 |           2.6968 |                 0.2261 |
| U3       | lshade|cso    |      719 |            199.0000 |           0.1902 |    0.7085 |           2.2517 |                 0.2261 |

U1/U3 来自 observed practical action sets，只是 secondary diagnostic，不称为 ground truth。T3 不足，U1-repeat 不可构造。
