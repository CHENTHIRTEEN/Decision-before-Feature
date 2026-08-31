Task17B 是零新增目标函数评估的机制分解。所有正式区间均按 `cv_group_id` 分组 bootstrap 5000 次；`new_objective_FE = 0`。

# 17b_03 T1/T2/T3 支持门槛

| suite   | track   | action_pair      |   states |   cv_groups |   state_threshold |   cv_group_threshold | formal_support   | support_status   |
|:--------|:--------|:-----------------|---------:|------------:|------------------:|---------------------:|:-----------------|:-----------------|
| bbob    | T2      | shade|lshade     |       24 |           9 |                15 |                    5 | True             | FORMAL           |
| bbob    | T2      | shade|cso        |       38 |           9 |                15 |                    5 | True             | FORMAL           |
| bbob    | T2      | lshade|cso       |       32 |           9 |                15 |                    5 | True             | FORMAL           |
| bbob    | T3      | shade|lshade|cso |        6 |           3 |                20 |                    5 | False            | UNAVAILABLE      |
| mabbob  | T2      | shade|lshade     |       16 |           5 |                15 |                    5 | True             | FORMAL           |
| mabbob  | T2      | shade|cso        |        8 |           3 |                15 |                    5 | False            | LOW_SUPPORT      |
| mabbob  | T2      | lshade|cso       |        8 |           4 |                15 |                    5 | False            | LOW_SUPPORT      |
| mabbob  | T3      | shade|lshade|cso |        0 |           0 |                20 |                    5 | False            | UNAVAILABLE      |
| pooled  | T2      | shade|lshade     |       40 |          14 |                30 |                    8 | True             | FORMAL           |
| pooled  | T2      | shade|cso        |       46 |          12 |                30 |                    8 | True             | FORMAL           |
| pooled  | T2      | lshade|cso       |       40 |          13 |                30 |                    8 | True             | FORMAL           |
| pooled  | T3      | shade|lshade|cso |        6 |           3 |                50 |                    8 | False            | UNAVAILABLE      |

T3 pooled 只有 6 states / 3 groups，full-signature track 不可用。
